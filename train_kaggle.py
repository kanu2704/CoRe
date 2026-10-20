#!/usr/bin/env python3
"""One training run of the P-SVM vs CoRe comparison, sized for Kaggle.

Phase 1 - state only, no images, no API (runs on CPU sessions):
  env_dense    MetaWorld's dense reward                        (paper row: Env Dense)
  env_sparse   1 when the task is solved, else 0               (paper row: Env Sparse)
  frm          CoRe's saved LLM-written reward code            (paper rows: FRM)
  svm          success visitation matching, simulator labels   (ours)
  svm_sparse   svm + success reward (the SVM paper's full reward)
  svm_seeded   svm whose discriminator starts with scripted successes (diagnostic only)

Phase 2 - CoRe's protocol with rendered images and the image reward model (T4):
  core         CoRe re-run: FRM refined by the LLM + RRM     (paper row: CoRe)
  rrm_only     RRM alone                                       (paper row: RRM)
  rrm_sparse   RRM + success reward (control: is the success bit alone enough?)
  rrm_svm      RRM + oracle SVM (main test)

Every hyperparameter defaults to CoRe's run_meta.sh / config/train_CoRe.yaml, and
the training loop follows train_CoRe.py step for step. The only differences are
the reward put in the FRM slot, lossless image storage, and skipping work whose
result is never used (replay images after the last RRM relabel).

Example:
  python train_kaggle.py --mode svm --task peg-insert-side-v2 --seed 0 --out runs
"""
import argparse
import importlib.util
import json
import os
import sys
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

PHASE1_MODES = ['env_dense', 'env_sparse', 'frm', 'svm', 'svm_sparse', 'svm_seeded']
PHASE2_MODES = ['core', 'rrm_only', 'rrm_sparse', 'rrm_svm']
SVM_MODES = ['svm', 'svm_sparse', 'svm_seeded', 'rrm_svm']
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
        if self.mode in ('frm', 'core'):
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

        self.step = 0
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.svm_on_step = None
        self.first_success_step = None
        self.start = time.time()

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
        if m in SVM_MODES:
            r = self.svm.reward(obs, action)
            return r + (float(success) if m == 'svm_sparse' else 0.0), None
        return 0.0, None  # rrm_only

    def svm_relabel_fn(self, obses, actions, successes):
        r = self.svm.reward_batch(obses, actions)
        return r + successes if self.mode == 'svm_sparse' else r

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
            get_action = lambda o: helper.act(o, sample=False)
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
                   hours=(time.time() - self.start) / 3600, rss_gb=rss_gb())
        print(f"[eval] step {self.step}  success {sr:.1f}%  ({(time.time() - self.start) / 3600:.2f} h, RSS {rss_gb():.1f} GB)")

    # ------------------------------------------------------------------ main loop (as train_CoRe.run)
    def run(self):
        cfg, args = self.cfg, self.args
        warmup = cfg['num_seed_steps'] + cfg['num_pre_steps']
        max_steps = self.env._max_episode_steps
        episode, done = 0, True
        interact_count = 0
        avg_train_true_return = deque([], maxlen=10)
        ep_obs, ep_act, step_frm_dict = [], [], []
        ep_ret = ep_true = ep_struct = ep_rrm = ep_succ = 0.0
        last_print = time.time()
        timed_out = False

        while self.step <= cfg['num_train_steps']:
            if done:
                if self.step > 0:
                    if ep_succ and self.first_success_step is None:
                        self.first_success_step = self.step
                        self.write('events', step=self.step, event='first_success')
                    svm_changed = False
                    if self.svm is not None:
                        self.svm.add_episode(np.array(ep_obs), np.array(ep_act), q=ep_succ, y=ep_succ)
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
                               **({k: v for k, v in self.svm.last.items()} if svm_changed else {}))
                    avg_train_true_return.append(ep_true)

                if self.step > 0 and self.step % cfg['eval_frequency'] == 0:
                    self.evaluate()
                if args.time_limit_hours and (time.time() - self.start) / 3600 > args.time_limit_hours:
                    timed_out = True
                    break

                obs = reset_env(self.env)
                if self.frm_gen is not None:
                    self.frm_gen.target_pos = target_pos(self.env)
                done = False
                ep_ret = ep_true = ep_struct = ep_rrm = ep_succ = 0.0
                ep_obs, ep_act, step_frm_dict = [], [], []
                episode_step = 0
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

            if self.step >= cfg['num_seed_steps']:
                self.agent.update(self.replay, NullLogger(), self.step, 1)

            next_obs, env_reward, done, info = step_env(self.env, action)
            success = float(info['success'])

            frame, r_rrm = None, 0.0
            if self.phase2:
                rgb = render_frame(self.env, self.task, cfg['image_size'])
                frame = Frame(rgb, args.png_level)
                if self.step >= warmup:
                    r_rrm = self.rrm_reward(rgb)

            r_struct, frm_info = self.structural_reward(obs, action, next_obs, env_reward, success)
            if frm_info is not None:
                step_frm_dict.append(frm_info)

            done_f = float(done)
            done_no_max = 0 if episode_step + 1 == max_steps else done_f
            reward_hat = (1.0 - self.alpha) * r_struct + self.alpha * r_rrm

            self.replay.add(obs, action, reward_hat, r_struct, r_rrm, next_obs, done_f, done_no_max,
                            image=frame if self.store_images else None, success=success)
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
                sps = self.step / max(time.time() - self.start, 1e-6)
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
            rf = self.frm_gen.FRM_generate(episode_freq=freq, cur_frm=self.frm)
            if rf is not None:
                self.frm = rf
                self.replay.relabel_with_rf(self.frm, target_pos(self.env), self.step)
            self.write('events', step=self.step, event='frm_refine', improved=rf is not None)

    def close(self, timed_out):
        model_dir = os.path.join(self.run_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)
        self.agent.save(model_dir, self.step)
        if self.RRM is not None:
            self.RRM.save(model_dir, self.step)
        if self.svm is not None:
            self.svm.save(os.path.join(model_dir, 'svm.pt'))
        summary = dict(mode=self.mode, task=self.task, seed=self.args.seed, steps=self.step,
                       complete=not timed_out, hours=(time.time() - self.start) / 3600,
                       first_success_step=self.first_success_step, svm_on_step=self.svm_on_step,
                       labeled_feedback=self.labeled_feedback, peak_rss_gb=rss_gb())
        with open(os.path.join(self.run_dir, 'done.json' if not timed_out else 'incomplete.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        for f in self.files.values():
            f.close()
        print("[done]", json.dumps(summary))


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
    p.add_argument('--frame_backend', default='png', choices=['ram', 'png', 'disk'],
                   help="image storage: png/disk are lossless and save RAM; ram = original")
    p.add_argument('--png_level', type=int, default=1)
    p.add_argument('--save_label_images', action='store_true', help="keep images in saved label files (large)")
    p.add_argument('--seed_actor', default='', help="svm_seeded: actor .pt of a trained policy that solves the task")
    p.add_argument('--time_limit_hours', type=float, default=0, help="stop early (run marked incomplete)")
    p.add_argument('--overwrite', action='store_true')
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
    if args.mode in PHASE2_MODES:
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
    if os.path.exists(run_dir):
        for name in ('eval.jsonl', 'episodes.jsonl', 'events.jsonl', 'incomplete.json'):
            if os.path.exists(os.path.join(run_dir, name)):
                os.remove(os.path.join(run_dir, name))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(dict(args=vars(args), cfg=cfg, vlm_model=os.environ.get('VLM_MODEL'),
                       torch=torch.__version__, cuda=torch.cuda.is_available()), f, indent=2)
    os.chdir(run_dir)  # CoRe writes VLM/LLM files relative to the working directory

    run = Run(args, cfg, run_dir)
    timed_out = run.run()
    run.close(timed_out)


if __name__ == '__main__':
    main()
