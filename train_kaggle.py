#!/usr/bin/env python3
"""One training run of the P-SVM vs CoRe comparison, sized for Kaggle.

Phase 1 - state only, no images, no API (runs on CPU sessions):
  env_dense    MetaWorld's dense reward                        (paper row: Env Dense)
  env_sparse   1 when the task is solved, else 0               (paper row: Env Sparse)
  frm          CoRe's saved LLM-written reward code            (paper rows: FRM)
  svm          success visitation matching, simulator labels   (ours)
  svm_sparse   svm + success reward (the SVM paper's full reward)
  svm_seeded   svm whose discriminator starts with ~10 successful episodes of a trained policy (demonstrations)
  frm_svm      CoRe's saved LLM-written reward + svm: CoRe with SVM in place of the VLM part
  frm_sparse   CoRe's saved LLM-written reward + success bit (control: is SVM more than the success bit?)
  frm_psvm     frm_svm with no ground truth: the SVM's success weights come from the VLM judging 3 frames
               of each episode (only those frames are rendered)

Phase 2 - CoRe's protocol with rendered images and the image reward model (T4):
  core         CoRe re-run: FRM refined by the LLM + RRM     (paper row: CoRe)
  rrm_only     RRM alone                                       (paper row: RRM)
  rrm_sparse   RRM + success reward (control: is the success bit alone enough?)
  rrm_svm      RRM + oracle SVM (main test)
  rrm_psvm     RRM + P-SVM: success weights q graded 0-10 by the VLM from each episode's final frame

Every hyperparameter defaults to CoRe's run_meta.sh / config/train_CoRe.yaml, and
the training loop follows train_CoRe.py step for step. The only differences are
the reward put in the FRM slot, lossless image storage, and skipping work whose
result is never used (replay images after the last RRM relabel).

Example:
  python train_kaggle.py --mode svm --task peg-insert-side-v2 --seed 0 --out runs
"""
import argparse
import glob
import importlib.util
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict, deque
from types import SimpleNamespace

import numpy as np
import torch
import yaml

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from psvm.compat import install_hydra_shim, target_pos, reset_env, step_env, scripted_policy  # noqa: E402

install_hydra_shim()
# MuJoCo picks its OpenGL backend when it is first imported, so utils / ReplayBuffer
# (which import MetaWorld) are imported in main() after MUJOCO_GL is set.
utils = ReplayBuffer = Frame = SVMReward = None


def import_simulation_modules():
    global utils, ReplayBuffer, Frame, SVMReward
    import utils as _utils
    from replay_buffer import ReplayBuffer as _ReplayBuffer
    from psvm.frames import Frame as _Frame
    from psvm.svm import SVMReward as _SVMReward
    utils, ReplayBuffer, Frame, SVMReward = _utils, _ReplayBuffer, _Frame, _SVMReward

PHASE1_MODES = ['env_dense', 'env_sparse', 'frm', 'svm', 'svm_sparse', 'svm_seeded', 'frm_svm', 'frm_sparse', 'frm_psvm']
JUDGED_MODES = ['rrm_psvm', 'frm_psvm']  # SVM success weights graded by the VLM instead of the simulator
PHASE2_MODES = ['core', 'rrm_only', 'rrm_sparse', 'rrm_svm', 'rrm_psvm']
SVM_MODES = ['svm', 'svm_sparse', 'svm_seeded', 'rrm_svm', 'rrm_psvm', 'frm_svm', 'frm_psvm']
TASKS = ['peg-insert-side-v2', 'hammer-v2', 'button-press-topdown-v2',
         'soccer-v2', 'sweep-into-v2', 'drawer-open-v2', 'dial-turn-v2']

# CoRe's MetaWorld settings (run_meta.sh on top of config/train_CoRe.yaml)
DEFAULTS = dict(
    num_train_steps=500000, num_seed_steps=1000, num_pre_steps=19000,
    eval_frequency=10000, num_eval_episodes=10, replay_buffer_capacity=200000, reward_alpha=0.5,
    segment=20, RRM_batch=20, max_feedback=500, num_interact=4000, RRM_lr=3e-4, RRM_update=10,
    reset_update=100, ensemble_size=3, large_batch=10, image_size=300,
    conv_kernel_sizes=[5, 3, 3, 3], conv_n_channels=[16, 32, 64, 128], conv_strides=[3, 2, 2, 2], kl_weight=5,
    FRM_align_step=25000, FRM_align_max_steps=100000,
    # success visitation matching (ours)
    svm_hidden=256, svm_depth=2, svm_lr=3e-4, svm_batch=512, svm_steps=50, svm_min_effective=5.0,
    svm_clip=1.0, svm_scale=1.0, svm_seed_episodes=10,
)


class NullLogger:
    """agent.update() logs through this; the run's own logs are written as JSON lines."""
    def log(self, *args, **kwargs):
        pass


def rss_gb():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == 'darwin' else r / 1e6


def load_frm(task):
    name = f"metaworld_{task.replace('-', '_').lower()}_frm"
    path = os.path.join(PROJECT_DIR, 'envs', 'frm_code', name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reward_function, path


def make_agent(obs_dim, act_dim, action_range, device):
    with open(os.path.join(PROJECT_DIR, 'config', 'agent', 'sac.yaml')) as f:
        cfg = yaml.safe_load(f)
    params = dict(cfg['agent']['params'])
    critic = {'class': cfg['double_q_critic']['class'], 'params': dict(cfg['double_q_critic']['params'])}
    actor = {'class': cfg['diag_gaussian_actor']['class'], 'params': dict(cfg['diag_gaussian_actor']['params'])}
    for c in (critic, actor):
        c['params'].update(obs_dim=obs_dim, action_dim=act_dim)
    for k, v in params.items():  # YAML reads values like 3e-4 as strings
        if isinstance(v, str):
            try:
                params[k] = float(v)
            except ValueError:
                pass
    params.update(obs_dim=obs_dim, action_dim=act_dim, action_range=action_range, device=device,
                  critic_cfg=critic, actor_cfg=actor)
    from agent.sac import SACAgent
    return SACAgent(**params)


def render_frame(env, task, size):
    """CoRe's image preprocessing for MetaWorld (train_CoRe.py)."""
    import cv2
    rgb = env.render()[::-1, :, :]
    if 'drawer' in task or 'sweep' in task:
        rgb = rgb[100:400, 100:400, :]
    return cv2.resize(rgb, (size, size))


class Run:
    def __init__(self, args, cfg, run_dir):
        self.args, self.cfg, self.run_dir = args, cfg, run_dir
        self.mode, self.task = args.mode, args.task
        self.phase2 = self.mode in PHASE2_MODES
        self.device = args.device
        self.files = {k: open(os.path.join(run_dir, f'{k}.jsonl'), 'a') for k in ('eval', 'episodes', 'events')}
        utils.set_seed_everywhere(args.seed)

        self.env = utils.make_metaworld_env(SimpleNamespace(env=f'metaworld_{self.task}', seed=args.seed))
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        action_range = [float(self.env.action_space.low.min()), float(self.env.action_space.high.max())]
        self.agent = make_agent(obs_dim, act_dim, action_range, self.device)

        self.alpha = 0.0  # weight on RRM; CoRe switches it on after the pre-training steps
        self.replay = ReplayBuffer(self.env.observation_space.shape, self.env.action_space.shape,
                                   int(cfg['replay_buffer_capacity']), self.device,
                                   image_size=cfg['image_size'] if self.phase2 else 0, reward_alpha=0.0,
                                   image_backend=args.frame_backend, png_level=args.png_level)
        self.store_images = self.phase2

        self.RRM = None
        if self.phase2:
            from RRM.reward_model import RewardModel
            self.RRM = RewardModel(
                obs_dim, act_dim, ensemble_size=cfg['ensemble_size'], size_segment=cfg['segment'],
                activation='tanh', lr=cfg['RRM_lr'], mb_size=cfg['RRM_batch'], large_batch=cfg['large_batch'],
                label_margin=0.0, teacher_beta=-1, teacher_gamma=1, teacher_eps_mistake=0,
                teacher_eps_skip=0, teacher_eps_equal=0, capacity=cfg['max_feedback'],
                env_name=f'metaworld_{self.task}', log_dir=run_dir, cached_label_path=None,
                image_height=cfg['image_size'], image_width=cfg['image_size'], resize_factor=1, resnet=0,
                conv_kernel_sizes=cfg['conv_kernel_sizes'], conv_strides=cfg['conv_strides'],
                conv_n_channels=cfg['conv_n_channels'], kl_weight=cfg['kl_weight'],
                frame_backend=args.frame_backend, png_level=args.png_level, label_source=args.labels,
                vlm_workers=args.vlm_workers, save_label_images=args.save_label_images)

        # the reward that fills CoRe's formal-reward slot
        self.frm, self.frm_gen = None, None
        self.last_frm = 0.0  # frm_svm: this step's FRM value, kept in the replay buffer when the SVM relabels
        if self.mode in ('frm', 'core', 'frm_svm', 'frm_sparse', 'frm_psvm'):
            self.frm, frm_path = load_frm(self.task)
            if self.mode == 'core' and args.llm_refine:
                from FRM.reward_code import FRMCode
                from FRM.misc import file_to_string
                sys.path.append(run_dir)
                sys.path.append(os.path.join(PROJECT_DIR, 'envs', 'env_info'))
                sys.path.append(os.path.join(PROJECT_DIR, 'envs', 'frm_code'))
                frm_cfg = SimpleNamespace(env=f'metaworld_{self.task}', num_train_steps=cfg['num_train_steps'],
                                          num_seed_steps=cfg['num_seed_steps'])
                self.frm_gen = FRMCode(frm_cfg, PROJECT_DIR, run_dir, self.RRM, llm_model=args.llm_model,
                                       sample_num=4, once_sample_num=4, patch_env_source=False)
                self.frm = self.frm_gen.dynamic_import_frm(os.path.basename(frm_path)[:-3])
                self.frm_gen.best_frm_response = file_to_string(frm_path)

        self.svm = None
        if self.mode in SVM_MODES:
            self.svm = SVMReward(obs_dim, act_dim, device=self.device, hidden=cfg['svm_hidden'],
                                 depth=cfg['svm_depth'], lr=cfg['svm_lr'], batch_size=cfg['svm_batch'],
                                 steps_per_update=cfg['svm_steps'], min_effective=cfg['svm_min_effective'],
                                 clip_beta=cfg['svm_clip'], scale=cfg['svm_scale'], seed=args.seed)
            if self.mode == 'svm_seeded':
                self.seed_svm(cfg['svm_seed_episodes'])
        self.judge = None
        if self.mode in JUDGED_MODES and args.success_judge == 'vlm':
            from psvm.judge import SuccessJudge
            self.judge = SuccessJudge(f'metaworld_{self.task}', PROJECT_DIR, style=args.judge_style)
        self.ep_frames = []  # rrm_psvm: frames at 1/2, 3/4 and the end of the episode, for the judge

        self.step = 0
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.svm_on_step = None
        self.first_success_step = None
        self.frm_refine_failures = 0
        self.start = time.time()
        # set by continue_from() when this run continues one that was stopped early
        self.hours_offset = 0.0
        self.continued_from = []
        self.episode0, self.interact0, self.avg_returns0 = 0, 0, []
        self.update_from = 0
        # periodic save points (--save_every_min), so a run killed without a clean stop loses little
        self.last_save, self.last_save_step = time.time(), None
        self._label_png = {'buffer_seg1': [], 'buffer_seg2': []}  # PNGs of labelled frames already encoded

    # ------------------------------------------------------------------ logging
    def write(self, kind, **row):
        row = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in row.items()}
        self.files[kind].write(json.dumps(row) + '\n')
        self.files[kind].flush()

    # ------------------------------------------------------------------ rewards
    def structural_reward(self, obs, action, next_obs, env_reward, success):
        """Reward placed in CoRe's formal-reward slot; returns (reward, FRM components)."""
        m = self.mode
        if m == 'env_dense':
            return float(env_reward), None
        if m in ('env_sparse', 'rrm_sparse'):
            return float(success), None
        if m in ('frm', 'core'):
            r, info = self.frm(next_obs, action, target_pos(self.env))
            return float(np.clip(r, -1, 1)), info
        if m == 'frm_sparse':
            r, info = self.frm(next_obs, action, target_pos(self.env))
            return float(np.clip(r, -1, 1)) + float(success), info
        if m in ('frm_svm', 'frm_psvm'):
            r, info = self.frm(next_obs, action, target_pos(self.env))
            self.last_frm = float(np.clip(r, -1, 1))
            return self.last_frm + self.svm.reward(obs, action), info
        if m in SVM_MODES:
            r = self.svm.reward(obs, action)
            return r + (float(success) if m == 'svm_sparse' else 0.0), None
        return 0.0, None  # rrm_only

    def success_weight(self, frames, success, true_return, max_steps):
        """rrm_psvm: the episode's success weight q, graded by the VLM from frames of the episode
        (--success_judge scripted: a dense-reward progress proxy, for testing without a VLM)."""
        if self.judge is None:
            return float(np.clip(true_return / (10.0 * max_steps), 0.0, 1.0)) if not success else 1.0
        q = self.judge.grade(frames)
        return 0.0 if q is None else q  # no usable answer: count it as a failure

    def svm_relabel_fn(self, obses, actions, successes, bases):
        r = self.svm.reward_batch(obses, actions)
        if self.mode == 'svm_sparse':
            return r + successes
        return r + bases if self.mode in ('frm_svm', 'frm_psvm') else r

    def seed_svm(self, n_episodes):
        """svm_seeded: give the discriminator successful episodes before training, from a trained
        policy (--seed_actor, e.g. the Env Dense run's actor) or MetaWorld's scripted policy.
        Diagnostic only: it separates 'SVM cannot get started' from 'SVM does not help'."""
        env = utils.make_metaworld_env(SimpleNamespace(env=f'metaworld_{self.task}', seed=self.args.seed + 10000))
        if self.args.seed_actor:
            helper = make_agent(env.observation_space.shape[0], env.action_space.shape[0],
                                [float(env.action_space.low.min()), float(env.action_space.high.max())], self.device)
            helper.actor.load_state_dict(torch.load(self.args.seed_actor, map_location=self.device))
            helper.train(False)
            get_action = lambda o: helper.act(o, sample=True)  # sampled: CoRe's fixed start would make greedy demos identical
            source = self.args.seed_actor
        else:
            scripted = scripted_policy(self.task)
            get_action = scripted.get_action
            source = 'scripted'
        kept = 0
        for _ in range(n_episodes * 3):
            obs, done, success = reset_env(env), False, 0.0
            obs_l, act_l = [], []
            while not done:
                action = np.clip(get_action(obs), -1, 1).astype(np.float32)
                obs_l.append(obs)
                act_l.append(action)
                obs, _, done, info = step_env(env, action)
                success = max(success, info['success'])
            if success:
                self.svm.add_episode(np.array(obs_l), np.array(act_l), q=1.0, y=1.0)
                kept += 1
            if kept >= n_episodes:
                break
        self.write('events', step=0, event='svm_seeded', successes=kept, source=source)
        print(f"[svm_seeded] added {kept} successful episodes from {source}")

    # ------------------------------------------------------------------ RRM (CoRe)
    def learn_rrm(self):
        t0 = time.time()
        labeled = self.RRM.uniform_sampling()
        self.total_feedback += self.RRM.mb_size
        self.labeled_feedback += labeled
        t_query = time.time() - t0
        acc = 0.0
        if self.labeled_feedback > 0:
            for _ in range(self.cfg['RRM_update']):
                self.RRM.train()
                acc = float(np.mean(self.RRM.train_reward()))
                if acc > 0.97:
                    break
        self.write('events', step=self.step, event='rrm_update', labeled=self.labeled_feedback,
                   total=self.total_feedback, rrm_acc=acc, vlm_acc=float(self.RRM.vlm_label_acc),
                   query_s=t_query, train_s=time.time() - t0 - t_query)

    def relabel_rrm(self):
        self.RRM.eval()
        self.replay.relabel_with_predictor(self.RRM, self.step)
        self.RRM.train()

    def rrm_reward(self, rgb):
        image = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        image = image.reshape(1, 3, image.shape[1], image.shape[2])
        self.RRM.eval()
        r = self.RRM.r_hat(image)
        self.RRM.train()
        return float(r)

    # ------------------------------------------------------------------ evaluation (as train_CoRe.evaluate)
    def evaluate(self):
        successes, returns = [], []
        for _ in range(self.cfg['num_eval_episodes']):
            obs, done, ep_ret, ep_succ = reset_env(self.env), False, 0.0, 0.0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, info = step_env(self.env, action)
                ep_ret += reward
                ep_succ = max(ep_succ, info['success'])
            successes.append(ep_succ)
            returns.append(ep_ret)
        sr = 100.0 * float(np.mean(successes))
        self.write('eval', step=self.step, success_rate=sr, true_return=float(np.mean(returns)),
                   hours=self.hours_offset + (time.time() - self.start) / 3600, rss_gb=rss_gb())
        print(f"[eval] step {self.step}  success {sr:.1f}%  ({(time.time() - self.start) / 3600:.2f} h, RSS {rss_gb():.1f} GB)")

    # ------------------------------------------------------------------ main loop (as train_CoRe.run)
    def run(self):
        cfg, args = self.cfg, self.args
        warmup = cfg['num_seed_steps'] + cfg['num_pre_steps']
        max_steps = self.env._max_episode_steps
        start_step = self.step  # 0, or the step a continued run starts from
        episode, done = self.episode0, True
        interact_count = self.interact0
        avg_train_true_return = deque(self.avg_returns0, maxlen=10)
        ep_obs, ep_act, step_frm_dict = [], [], []
        ep_ret = ep_true = ep_struct = ep_rrm = ep_succ = 0.0
        last_print = time.time()
        timed_out = False

        while self.step <= cfg['num_train_steps']:
            if done:
                if self.step > start_step:
                    if ep_succ and self.first_success_step is None:
                        self.first_success_step = self.step
                        self.write('events', step=self.step, event='first_success')
                    svm_changed = False
                    q = ep_succ if self.mode not in JUDGED_MODES else self.success_weight(
                        self.ep_frames or [rgb], ep_succ, ep_true, max_steps)
                    if self.svm is not None:
                        self.svm.add_episode(np.array(ep_obs), np.array(ep_act), q=q, y=ep_succ)
                        was_ready = self.svm.ready
                        svm_changed = self.svm.update()
                        if self.svm.ready and not was_ready:
                            self.svm_on_step = self.step
                            self.write('events', step=self.step, event='svm_on', **self.svm.last)
                        if svm_changed:
                            self.replay.relabel_structural(self.svm_relabel_fn)
                    if self.frm_gen is not None and self.step <= cfg['FRM_align_max_steps']:
                        self.frm_policy_feedback(ep_succ, step_frm_dict)
                    self.write('episodes', episode=episode, step=self.step, success=ep_succ, true_return=ep_true,
                               reward_return=ep_ret, struct_return=ep_struct, rrm_return=ep_rrm,
                               svm_ready=bool(self.svm.ready) if self.svm else None,
                               labeled_feedback=self.labeled_feedback,
                               **({'q': q} if self.mode in JUDGED_MODES else {}),
                               **({k: v for k, v in self.svm.last.items()} if svm_changed else {}))
                    avg_train_true_return.append(ep_true)

                if self.step > start_step and self.step % cfg['eval_frequency'] == 0:
                    self.evaluate()
                if args.time_limit_hours and (time.time() - self.start) / 3600 > args.time_limit_hours:
                    timed_out = True
                    break
                if args.save_every_min and self.step > start_step and time.time() - self.last_save > 60 * args.save_every_min:
                    self.save_point()

                obs = reset_env(self.env)
                if self.frm_gen is not None:
                    self.frm_gen.target_pos = target_pos(self.env)
                done = False
                ep_ret = ep_true = ep_struct = ep_rrm = ep_succ = 0.0
                ep_obs, ep_act, step_frm_dict = [], [], []
                episode_step = 0
                self.ep_frames = []
                episode += 1

            # action
            if self.step < cfg['num_seed_steps']:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # CoRe: switch on and update the residual reward model
            if self.phase2 and self.step == warmup:
                self.alpha = cfg['reward_alpha']
                self.replay.reward_alpha = cfg['reward_alpha']
                self.RRM.change_batch(1)
                margin = np.mean(avg_train_true_return) * (cfg['segment'] / max_steps)
                self.RRM.set_teacher_thres_skip(margin)
                self.RRM.set_teacher_thres_equal(margin)
                self.learn_rrm()
                self.relabel_rrm()
                self.agent.reset_critic()
                self.agent.update_after_reset(self.replay, NullLogger(), self.step,
                                              gradient_update=cfg['reset_update'], policy_update=True)
                interact_count = 0
            elif self.phase2 and self.step > warmup:
                if self.total_feedback < cfg['max_feedback'] and interact_count == cfg['num_interact']:
                    self.RRM.change_batch(1)
                    margin = np.mean(avg_train_true_return) * (cfg['segment'] / max_steps)
                    self.RRM.set_teacher_thres_skip(margin)
                    self.RRM.set_teacher_thres_equal(margin)
                    if self.RRM.mb_size + self.total_feedback > cfg['max_feedback']:
                        self.RRM.set_batch(cfg['max_feedback'] - self.total_feedback)
                    self.learn_rrm()
                    self.relabel_rrm()
                    interact_count = 0

            if self.step >= cfg['num_seed_steps'] and self.step >= self.update_from:
                self.agent.update(self.replay, NullLogger(), self.step, 1)

            next_obs, env_reward, done, info = step_env(self.env, action)
            success = float(info['success'])

            frame, r_rrm = None, 0.0
            if self.phase2:
                t_render = time.time()
                rgb = render_frame(self.env, self.task, cfg['image_size'])
                if self.step == 0:
                    from psvm.compat import gl_renderer
                    self.write('events', step=0, event='renderer', renderer=gl_renderer(),
                               first_frame_ms=(time.time() - t_render) * 1e3)
                frame = Frame(rgb, args.png_level)
                if self.mode == 'rrm_psvm' and (episode_step + 1) in (max_steps // 2, 3 * max_steps // 4, max_steps):
                    self.ep_frames.append(rgb)
                if self.step >= warmup:
                    r_rrm = self.rrm_reward(rgb)
            elif self.mode == 'frm_psvm' and (episode_step + 1) in (max_steps // 2, 3 * max_steps // 4, max_steps):
                # state-only training: render just the frames the judge sees
                self.ep_frames.append(render_frame(self.env, self.task, cfg['image_size']))

            r_struct, frm_info = self.structural_reward(obs, action, next_obs, env_reward, success)
            if frm_info is not None:
                step_frm_dict.append(frm_info)

            done_f = float(done)
            done_no_max = 0 if episode_step + 1 == max_steps else done_f
            reward_hat = (1.0 - self.alpha) * r_struct + self.alpha * r_rrm

            self.replay.add(obs, action, reward_hat, r_struct, r_rrm, next_obs, done_f, done_no_max,
                            image=frame if self.store_images else None, success=success, base=self.last_frm)
            if self.phase2 and self.total_feedback < cfg['max_feedback']:
                self.RRM.add_data(obs, action, env_reward, done_f, img=frame)

            ep_obs.append(obs)
            ep_act.append(np.asarray(action, dtype=np.float32))
            ep_ret += reward_hat
            ep_true += env_reward
            ep_struct += (1.0 - self.alpha) * r_struct
            ep_rrm += self.alpha * r_rrm
            ep_succ = max(ep_succ, success)

            obs = next_obs
            episode_step += 1
            self.step += 1
            interact_count += 1

            # once the last labels are used, CoRe empties the RRM data; replay frames are never read again
            if self.phase2 and not self.RRM.preference_data_emptyed and self.total_feedback >= cfg['max_feedback']:
                self.RRM.empty_preference_data()
                self.replay.drop_images()
                self.store_images = False
                self.write('events', step=self.step, event='rrm_done', labeled=self.labeled_feedback)

            if time.time() - last_print > 120:
                sps = (self.step - start_step) / max(time.time() - self.start, 1e-6)
                print(f"[{self.mode} {self.task} s{args.seed}] step {self.step}  {sps:.0f} steps/s  "
                      f"RSS {rss_gb():.1f} GB  feedback {self.labeled_feedback}")
                last_print = time.time()

        return timed_out

    def frm_policy_feedback(self, ep_succ, step_frm_dict):
        """CoRe's FRM refinement: collect policy feedback, ask the LLM every FRM_align_step steps."""
        totals = defaultdict(float)
        for d in step_frm_dict:
            for k, v in d.items():
                totals[k] += v
        md = self.frm_gen.metric_dict
        md.setdefault("socre", []).append(round(ep_succ, 2))
        for k, v in totals.items():
            md.setdefault(k, []).append(round(v, 2))
        if self.step % self.cfg['FRM_align_step'] == 0:
            freq = max(int(len(md["socre"]) // 10), 1)
            try:
                rf = self.frm_gen.FRM_generate(episode_freq=freq, cur_frm=self.frm)
            except (Exception, SystemExit) as e:
                # LLM unavailable (e.g. quota): keep the current formal reward and keep training
                self.frm_refine_failures += 1
                self.write('events', step=self.step, event='frm_refine_failed', reason=str(e)[:300])
                print(f"[frm] refinement failed, keeping the current formal reward: {str(e)[:200]}")
                return
            if rf is not None:
                self.frm = rf
                self.replay.relabel_with_rf(self.frm, target_pos(self.env), self.step)
            self.write('events', step=self.step, event='frm_refine', improved=rf is not None)

    # ------------------------------------------------------------------ continuing a run stopped early
    def save_continue_state(self, model_dir):
        """What a later continue_from() needs besides the networks: optimizer states, the entropy
        weight, the SVM's episodes and the frames of the labelled segments (those live in /tmp,
        which a Kaggle session wipes). The replay buffer is not saved."""
        state = dict(log_alpha=self.agent.log_alpha.detach().cpu(),
                     actor_opt=self.agent.actor_optimizer.state_dict(),
                     critic_opt=self.agent.critic_optimizer.state_dict(),
                     alpha_opt=self.agent.log_alpha_optimizer.state_dict())
        if self.RRM is not None:
            state['rrm_opt'] = self.RRM.opt.state_dict()
        if self.svm is not None:
            state.update(svm_store=self.svm.store, svm_opt=self.svm.opt.state_dict(), svm_rng=self.svm.rng)
        torch.save(state, os.path.join(model_dir, f'state_{self.step}.pt'))
        if self.phase2 and not self.RRM.preference_data_emptyed and self.labeled_feedback > 0:
            from psvm.frames import encode_png
            n_frames = (self.RRM.capacity if self.RRM.buffer_full else self.RRM.buffer_index) * self.cfg['segment']
            blobs = {}
            for name in ('buffer_seg1', 'buffer_seg2'):
                frames, done = getattr(self.RRM, name), self._label_png[name]  # labelled frames never change
                for a in range(len(done), n_frames, 256):
                    done.extend(encode_png(f, self.args.png_level) for f in frames.take(np.arange(a, min(a + 256, n_frames))))
                blobs[name] = done[:n_frames]
            path = os.path.join(self.run_dir, 'label_frames.pkl')
            with open(path + '.tmp', 'wb') as f:
                pickle.dump(blobs, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(path + '.tmp', path)

    def save_point(self):
        """Save everything continue_from() needs at this step, written as saved_at_<step>.json last so
        a kill during the save leaves the previous save point intact; then drop the previous one."""
        model_dir = os.path.join(self.run_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)
        self.agent.save(model_dir, self.step)
        if self.RRM is not None:
            self.RRM.save(model_dir, self.step)
        if self.svm is not None:
            self.svm.save(os.path.join(model_dir, f'svm_{self.step}.pt'))
        self.save_continue_state(model_dir)
        record = dict(steps=self.step, labeled_feedback=self.labeled_feedback,
                      hours=self.hours_offset + (time.time() - self.start) / 3600,
                      first_success_step=self.first_success_step, svm_on_step=self.svm_on_step,
                      frm_refine_failures=self.frm_refine_failures, continued_from=self.continued_from)
        path = os.path.join(self.run_dir, f'saved_at_{self.step}.json')
        with open(path + '.tmp', 'w') as f:
            json.dump(record, f, indent=2)
        os.replace(path + '.tmp', path)
        prev = self.last_save_step
        if prev is not None and prev != self.step:
            for old in (glob.glob(os.path.join(model_dir, f'*_{prev}.pt')) + glob.glob(os.path.join(model_dir, f'reward_model_{prev}_*.pt'))
                        + [os.path.join(self.run_dir, f'saved_at_{prev}.json')]):
                if os.path.exists(old):
                    os.remove(old)
        self.last_save, self.last_save_step = time.time(), self.step
        self.write('events', step=self.step, event='saved', labeled=self.labeled_feedback)

    def label_frames(self, n):
        """Frames of the first n labelled segment pairs, as get(name, start, stop) -> uint8 frames.
        From label_frames.pkl (written when a run stops early) or label_frames.tgz (the RRM's
        seg1_*/seg2_* frame files copied by hand from /tmp/psvm_frames before the session ended)."""
        n_frames, size = n * self.cfg['segment'], self.cfg['image_size']
        path = os.path.join(self.run_dir, 'label_frames.pkl')
        if os.path.exists(path):
            from psvm.frames import decode_png
            with open(path, 'rb') as f:
                blobs = pickle.load(f)
            if min(len(v) for v in blobs.values()) < n_frames:
                return None, 'label_frames.pkl has fewer frames than labels'
            get = lambda name, a, b: np.stack([decode_png(x) for x in blobs['buffer_' + name][a:b]])
            return get, 'label_frames.pkl'
        path = os.path.join(self.run_dir, 'label_frames.tgz')
        if os.path.exists(path):
            tmp = tempfile.mkdtemp(prefix='label_frames_', dir=os.environ.get('PSVM_DISK_DIR') or None)
            self._label_tmp = tmp
            subprocess.run(['tar', '-xzf', path, '-C', tmp], check=True)
            names = sorted(os.listdir(tmp))
            found = {k: [x for x in names if x.startswith(k + '_')] for k in ('seg1', 'seg2')}
            if any(len(v) != 1 for v in found.values()):
                return None, f'label_frames.tgz should hold one seg1_* and one seg2_* file, found {names}'
            arrays = {}
            for k, (name,) in found.items():
                frames = np.memmap(os.path.join(tmp, name), dtype=np.uint8, mode='r').reshape(-1, size, size, 3)
                if len(frames) < n_frames or not frames[n_frames - 1].any():
                    return None, f'{name} does not hold all {n} labelled segments'
                arrays[k] = frames
            return (lambda name, a, b: np.array(arrays[name][a:b])), 'label_frames.tgz'
        return None, 'no saved label frames (label_frames.pkl / label_frames.tgz)'

    def trim_after(self, step):
        """Drop what an attempt that continued from `step` and was then killed (no clean stop) wrote:
        log rows after the stop, and label files of rounds after it (moved to vlm_label_set_discarded)."""
        for f in self.files.values():
            f.close()
        kept = {}
        for kind in ('eval', 'episodes', 'events'):
            path = os.path.join(self.run_dir, f'{kind}.jsonl')
            rows = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
            keep = [r for r in rows if r['step'] < step or (r['step'] == step and r.get('event') != 'continued')]
            kept[kind] = keep
            if len(keep) != len(rows):
                with open(path + '.tmp', 'w') as f:
                    f.writelines(json.dumps(r) + '\n' for r in keep)
                os.replace(path + '.tmp', path)
        self.files = {k: open(os.path.join(self.run_dir, f'{k}.jsonl'), 'a') for k in ('eval', 'episodes', 'events')}
        n_rounds = sum(1 for r in kept['events'] if r.get('event') == 'rrm_update')
        label_files = sorted(glob.glob(os.path.join(self.run_dir, 'vlm_label_set', '*.pkl')))
        if len(label_files) > n_rounds:
            aside = os.path.join(self.run_dir, 'vlm_label_set_discarded')
            os.makedirs(aside, exist_ok=True)
            for path in label_files[n_rounds:]:
                shutil.move(path, os.path.join(aside, os.path.basename(path)))

    def restore_rrm_labels(self):
        """Refill the RRM's labelled-segment buffer exactly as the stopped run filled it:
        labels, VLM weights and states from vlm_label_set/*.pkl (one file per round), frames
        from label_frames(). Returns (number of labels restored, source or reason)."""
        rounds = []
        for path in sorted(glob.glob(os.path.join(self.run_dir, 'vlm_label_set', '*.pkl'))):
            with open(path, 'rb') as f:
                _, _, vlm_labels, vlm_weights, sa_t_1, sa_t_2, _, _ = pickle.load(f)
            if len(vlm_labels):
                rounds.append((vlm_labels, vlm_weights, sa_t_1, sa_t_2))
        n = sum(len(r[0]) for r in rounds)
        if n == 0:
            return 0, 'no labels yet'
        if n != self.labeled_feedback:
            return 0, f'{n} labels in vlm_label_set but the run had {self.labeled_feedback}'
        get, source = self.label_frames(n)
        try:
            if get is None:
                return 0, source
            seg, size, i = self.cfg['segment'], self.cfg['image_size'], 0
            for vlm_labels, vlm_weights, sa_t_1, sa_t_2 in rounds:
                k = len(vlm_labels)
                img_t_1 = get('seg1', i * seg, (i + k) * seg).reshape(k, seg, size, size, 3)
                img_t_2 = get('seg2', i * seg, (i + k) * seg).reshape(k, seg, size, size, 3)
                self.RRM.put_queries(img_t_1, img_t_2, vlm_labels, sa_t_1, sa_t_2, vlm_weights)
                i += k
            return n, source
        finally:
            if getattr(self, '_label_tmp', None):
                shutil.rmtree(self._label_tmp, ignore_errors=True)

    def continue_from(self, stopped):
        """Continue a run that stopped early (incomplete.json) from the step it stopped at.
        Restored: the networks, the counters and label schedule, the RRM's labels, and (if saved)
        the optimizers, entropy weight and SVM episodes. Not restored: the replay buffer, which
        starts empty; SAC updates resume after num_seed_steps new steps refill it."""
        cfg, step = self.cfg, int(stopped['steps'])
        self.trim_after(step)
        model_dir = os.path.join(self.run_dir, 'models')
        load = lambda name: torch.load(os.path.join(model_dir, name), map_location=self.device, weights_only=False)
        self.agent.actor.load_state_dict(load(f'actor_{step}.pt'))
        self.agent.critic.load_state_dict(load(f'critic_{step}.pt'))
        self.agent.critic_target.load_state_dict(load(f'critic_target_{step}.pt'))
        extra = load(f'state_{step}.pt') if os.path.exists(os.path.join(model_dir, f'state_{step}.pt')) else None
        if extra:
            with torch.no_grad():
                self.agent.log_alpha.copy_(extra['log_alpha'].to(self.agent.log_alpha.device))
            self.agent.actor_optimizer.load_state_dict(extra['actor_opt'])
            self.agent.critic_optimizer.load_state_dict(extra['critic_opt'])
            self.agent.log_alpha_optimizer.load_state_dict(extra['alpha_opt'])
        if self.RRM is not None:
            for m in range(self.RRM.de):
                self.RRM.ensemble[m].load_state_dict(load(f'reward_model_{step}_{m}.pt'))
            if extra and 'rrm_opt' in extra:
                self.RRM.opt.load_state_dict(extra['rrm_opt'])
        svm_file = f'svm_{step}.pt' if os.path.exists(os.path.join(model_dir, f'svm_{step}.pt')) else 'svm.pt'
        if self.svm is not None and os.path.exists(os.path.join(model_dir, svm_file)):
            d = load(svm_file)
            self.svm.disc.load_state_dict(d['disc'])
            self.svm.mean, self.svm.std, self.svm.ready = d['mean'], d['std'], d['ready']
            if extra and 'svm_store' in extra:
                self.svm.store, self.svm.rng = extra['svm_store'], extra['svm_rng']
                self.svm.opt.load_state_dict(extra['svm_opt'])
            elif self.svm.ready:
                self.svm.ready = False  # its episodes were not saved: switch on again from new ones

        # counters, from the run's own logs
        read = lambda kind: [json.loads(l) for l in open(os.path.join(self.run_dir, f'{kind}.jsonl'))]
        episodes = [e for e in read('episodes') if e['step'] <= step]
        rounds = [e for e in read('events') if e['event'] == 'rrm_update' and e['step'] <= step]
        self.step = step
        self.labeled_feedback = int(stopped.get('labeled_feedback', 0))
        self.total_feedback = int(rounds[-1]['total']) if rounds else 0
        self.first_success_step = stopped.get('first_success_step')
        self.svm_on_step = stopped.get('svm_on_step')
        self.frm_refine_failures = int(stopped.get('frm_refine_failures', 0))
        self.hours_offset = float(stopped.get('hours', 0.0))
        self.continued_from = list(stopped.get('continued_from', [])) + [step]
        self.episode0 = episodes[-1]['episode'] if episodes else 0
        self.avg_returns0 = [e['true_return'] for e in episodes[-10:]]
        max_steps = self.env._max_episode_steps
        if rounds:  # keep the label schedule; collect at least one episode before the next round
            self.interact0 = min(step - rounds[-1]['step'], cfg['num_interact'] - max_steps)
        self.update_from = step + cfg['num_seed_steps']

        restored, note = 0, ''
        if self.phase2:
            if step > cfg['num_seed_steps'] + cfg['num_pre_steps']:
                self.alpha = cfg['reward_alpha']
                self.replay.reward_alpha = cfg['reward_alpha']
            if self.total_feedback >= cfg['max_feedback']:
                self.RRM.empty_preference_data()
                self.replay.drop_images()
                self.store_images = False
            else:
                restored, note = self.restore_rrm_labels()
        utils.set_seed_everywhere(self.args.seed + step)
        self.write('events', step=step, event='continued', from_step=step, hours_before=self.hours_offset,
                   restored_labels=restored,
                   labels_of=self.labeled_feedback, labels_note=note, optimizers_restored=bool(extra),
                   replay='empty; SAC updates resume after %d new steps' % cfg['num_seed_steps'])
        print(f"[continue] from step {step}: {restored}/{self.labeled_feedback} labels restored ({note}), "
              f"optimizers {'restored' if extra else 'new'}, replay buffer empty")

    def close(self, timed_out, reason=None):
        model_dir = os.path.join(self.run_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)
        self.agent.save(model_dir, self.step)
        if self.RRM is not None:
            self.RRM.save(model_dir, self.step)
        if self.svm is not None:
            self.svm.save(os.path.join(model_dir, 'svm.pt'))
        if timed_out:
            self.save_continue_state(model_dir)
        summary = dict(mode=self.mode, task=self.task, seed=self.args.seed, steps=self.step,
                       complete=not timed_out, hours=self.hours_offset + (time.time() - self.start) / 3600,
                       first_success_step=self.first_success_step, svm_on_step=self.svm_on_step,
                       labeled_feedback=self.labeled_feedback, peak_rss_gb=rss_gb(),
                       frm_refine_failures=self.frm_refine_failures, stop_reason=reason)
        if self.continued_from:
            summary['continued_from'] = self.continued_from
        with open(os.path.join(self.run_dir, 'done.json' if not timed_out else 'incomplete.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        for f in self.files.values():
            f.close()
        print("[done]", json.dumps(summary))


def stopped_early(run_dir):
    """The record of the point a run can continue from (networks saved at that step), else None:
    incomplete.json after a clean early stop; if the run was killed without one, its newest save
    point. What the killed attempt did after that point is lost."""
    if os.path.exists(os.path.join(run_dir, 'done.json')):
        return None
    usable = lambda r: r and int(r.get('steps', 0)) > 0 and os.path.exists(
        os.path.join(run_dir, 'models', f"actor_{int(r['steps'])}.pt"))
    path = os.path.join(run_dir, 'incomplete.json')
    if os.path.exists(path):  # a clean stop is always the latest point
        stopped = json.load(open(path))
        return stopped if usable(stopped) else None
    # killed without a clean stop: the newest save point (saved_at_*: periodic, stopped_at_*: the stop
    # an attempt continued from, or rebuilt from the 'continued' event of an older version)
    records = [json.load(open(p)) for p in glob.glob(os.path.join(run_dir, 'saved_at_*.json'))
               + glob.glob(os.path.join(run_dir, 'stopped_at_*.json'))] + [stop_from_logs(run_dir)]
    records = [r for r in records if usable(r)]
    return max(records, key=lambda r: int(r['steps'])) if records else None


def stop_from_logs(run_dir):
    """Rebuild the stop record from the last 'continued' event (runs continued before stopped_at_*.json existed)."""
    path = os.path.join(run_dir, 'events.jsonl')
    events = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    cont = [e for e in events if e.get('event') == 'continued']
    if not cont:
        return None
    step = int(cont[-1]['from_step'])
    before = [e for e in events if e['step'] <= step and e.get('event') != 'continued']
    first = lambda name: next((e['step'] for e in before if e.get('event') == name), None)
    rounds = [e for e in before if e.get('event') == 'rrm_update']
    evals = [json.loads(l) for l in open(os.path.join(run_dir, 'eval.jsonl'))]
    evals = [e for e in evals if e['step'] <= step]
    hours = cont[-1].get('hours_before', evals[-1]['hours'] if evals else 0.0)
    return dict(steps=step, labeled_feedback=rounds[-1]['labeled'] if rounds else 0, hours=hours,
                first_success_step=first('first_success'), svm_on_step=first('svm_on'),
                frm_refine_failures=sum(1 for e in before if e.get('event') == 'frm_refine_failed'),
                continued_from=sorted({int(e['from_step']) for e in cont if int(e['from_step']) < step}))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', required=True, choices=PHASE1_MODES + PHASE2_MODES)
    p.add_argument('--task', required=True, choices=TASKS)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='runs')
    p.add_argument('--device', default='auto', help="auto | cpu | cuda")
    p.add_argument('--threads', type=int, default=0, help="torch CPU threads (0 = torch default)")
    p.add_argument('--labels', default='vlm', choices=['vlm', 'scripted'],
                   help="RRM labels: 'vlm' as in the paper, 'scripted' = true preferences (free, testing only)")
    p.add_argument('--vlm_workers', type=int, default=4, help="parallel VLM queries per feedback session")
    p.add_argument('--llm_model', default=os.environ.get('LLM_MODEL', 'gpt-4.1-mini'))
    p.add_argument('--no_llm_refine', dest='llm_refine', action='store_false',
                   help="core mode without LLM refinement (initial FRM + RRM; not the paper's CoRe)")
    p.add_argument('--frame_backend', default=os.environ.get('PSVM_FRAME_BACKEND', 'png'), choices=['ram', 'png', 'disk'],
                   help="image storage: png/disk are lossless and save RAM; ram = original")
    p.add_argument('--png_level', type=int, default=1)
    p.add_argument('--save_label_images', action='store_true', help="keep images in saved label files (large)")
    p.add_argument('--success_judge', default='vlm', choices=['vlm', 'scripted'],
                   help="rrm_psvm: grade episodes with the VLM, or a dense-reward proxy (testing only)")
    p.add_argument('--judge_style', default='stages', choices=['stages', 'success', 'scale'],
                   help="rrm_psvm: furthest stage of a task checklist over 3 frames, or a 0-10 grade of the final frame")
    p.add_argument('--seed_actor', default='', help="svm_seeded: actor .pt of a trained policy that solves the task")
    p.add_argument('--time_limit_hours', type=float, default=0, help="stop early (run marked incomplete)")
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--save_every_min', type=float, default=30,
                   help="save a point to continue from every this many minutes (0 = only at a clean stop)")
    p.add_argument('--no_continue', dest='continue_runs', action='store_false',
                   help="start a run that stopped early from step 0 instead of continuing it")
    p.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE',
                   help="override a hyperparameter, e.g. --set num_train_steps=20000 (smoke tests only)")
    args = p.parse_args()

    cfg = dict(DEFAULTS)
    for kv in args.set:
        k, v = kv.split('=', 1)
        if k not in cfg:
            raise SystemExit(f"unknown setting {k}")
        cfg[k] = yaml.safe_load(v)
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.mode in PHASE2_MODES or args.mode == 'frm_psvm':
        # headless GPU rendering on Linux (Kaggle); macOS needs MUJOCO_GL=glfw
        os.environ.setdefault('MUJOCO_GL', 'egl' if sys.platform.startswith('linux') else 'glfw')
        if os.environ['MUJOCO_GL'] in ('egl', 'osmesa'):
            os.environ.setdefault('PYOPENGL_PLATFORM', os.environ['MUJOCO_GL'])
        os.environ.setdefault('PSVM_SAVE_VLM_VIDEOS', '0')
    else:
        os.environ.setdefault('MUJOCO_GL', 'disable')  # phase 1 never renders
    import_simulation_modules()

    # runs with changed hyperparameters (--set, smoke tests) live apart so they never count as real runs
    run_dir = os.path.abspath(os.path.join(args.out, '_smoke' if args.set else '', args.task, args.mode, f'seed{args.seed}'))
    if os.path.exists(os.path.join(run_dir, 'done.json')) and not args.overwrite:
        print(f"already finished: {run_dir}")
        return
    lock = os.path.join(run_dir, 'running.pid')
    if os.path.exists(lock):
        try:
            other = int(open(lock).read().strip())
            os.kill(other, 0)  # raises if that process is gone
            print(f"this run is already being trained by process {other}: {run_dir}")
            return
        except (ValueError, OSError):
            pass  # stale lock from a run that died
    stopped = None if (args.overwrite or not args.continue_runs) else stopped_early(run_dir)
    if stopped is None and os.path.exists(run_dir):
        for name in ('eval.jsonl', 'episodes.jsonl', 'events.jsonl', 'incomplete.json', 'done.json',
                     'label_frames.pkl', 'label_frames.tgz'):
            if os.path.exists(os.path.join(run_dir, name)):
                os.remove(os.path.join(run_dir, name))
        for name in ('vlm_output', 'vlm_label_set', 'vlm_label_set_discarded', 'models'):  # leftovers of an earlier attempt
            shutil.rmtree(os.path.join(run_dir, name), ignore_errors=True)
        for path in glob.glob(os.path.join(run_dir, 'stopped_at_*.json')) + glob.glob(os.path.join(run_dir, 'saved_at_*.json')):
            os.remove(path)
    os.makedirs(run_dir, exist_ok=True)
    if stopped is None:
        with open(os.path.join(run_dir, 'config.json'), 'w') as f:
            json.dump(dict(args=vars(args), cfg=cfg, vlm_model=os.environ.get('VLM_MODEL'),
                           torch=torch.__version__, cuda=torch.cuda.is_available()), f, indent=2)
    with open(lock, 'w') as f:
        f.write(str(os.getpid()))
    os.chdir(run_dir)  # CoRe writes VLM/LLM files relative to the working directory

    try:
        from FRM.code_agent import DailyQuotaError
        run = Run(args, cfg, run_dir)
        if stopped is not None:
            run.continue_from(stopped)
            with open(os.path.join(run_dir, f"stopped_at_{int(stopped['steps'])}.json"), 'w') as f:
                json.dump(stopped, f, indent=2)  # kept: if this attempt is killed, the next start continues from here
            if os.path.exists(os.path.join(run_dir, 'incomplete.json')):
                os.remove(os.path.join(run_dir, 'incomplete.json'))
        try:
            timed_out, reason = run.run(), None
        except DailyQuotaError as e:
            # the VLM key's daily quota is used up: no more labels today, stop cleanly (marked incomplete)
            print(f"[stop] {e}")
            run.write('events', step=run.step, event='vlm_daily_quota', reason=str(e)[:300])
            timed_out, reason = True, 'VLM daily quota used up; restart after the quota resets'
        run.close(timed_out, reason)
    finally:
        if os.path.exists(lock):
            os.remove(lock)


if __name__ == '__main__':
    main()
