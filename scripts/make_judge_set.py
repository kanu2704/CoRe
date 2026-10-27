#!/usr/bin/env python3
"""Build a test set for VLM success judges: real attempts with known outcomes.

For each task, trained policies from earlier runs act in CoRe's environment: successful
attempts come from policies that solve the task, failures and near-misses from policies that
did not (and from good policies with random actions mixed in). Each attempt keeps the frames
the P-SVM judge sees (1/2, 3/4 and the end of the episode), the simulator's success flag and
the dense environment return (a measure of real progress).

  MUJOCO_GL=glfw python scripts/make_judge_set.py --runs runs_mac --out runs_mac/judge_set --per_class 20
"""
import argparse
import glob
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault('MUJOCO_GL', 'egl' if sys.platform.startswith('linux') else 'glfw')

import train_kaggle as tk  # noqa: E402

TASKS = ['peg-insert-side-v2', 'hammer-v2', 'button-press-topdown-v2']
# (mode, kind): policies expected to succeed, and policies that did not learn the task
GOOD = ['frm_svm', 'env_dense', 'svm_seeded']
BAD = ['svm', 'frm', 'env_sparse']


def actors(runs, task, modes, good):
    out = []
    for mode in modes:
        for d in sorted(glob.glob(os.path.join(runs, task, mode, 'seed*'))):
            if not os.path.exists(os.path.join(d, 'done.json')):
                continue
            ep = [json.loads(l) for l in open(os.path.join(d, 'episodes.jsonl'))][-50:]
            rate = sum(e['success'] > 0 for e in ep) / max(1, len(ep))
            files = sorted(glob.glob(os.path.join(d, 'models', 'actor_*.pt')))
            if files and ((rate >= 0.8) if good else (rate <= 0.4)):
                out.append((f'{mode}/{os.path.basename(d)}', files[-1]))
    return out


def attempt(env, agent, task, size, noise, rng):
    obs, done, step, frames, ret, success = tk.reset_env(env), False, 0, [], 0.0, 0.0
    max_steps = env._max_episode_steps
    while not done:
        with tk.utils.eval_mode(agent):
            action = agent.act(obs, sample=True)
        if noise and rng.random() < noise:
            action = env.action_space.sample()
        obs, r, done, info = tk.step_env(env, action)
        ret += r
        success = max(success, float(info['success']))
        step += 1
        if step in (max_steps // 2, 3 * max_steps // 4, max_steps):
            frames.append(tk.render_frame(env, task, size))
    return frames, success, ret


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', default='runs_mac')
    p.add_argument('--out', default='runs_mac/judge_set')
    p.add_argument('--per_class', type=int, default=20, help="successes and failures to keep per task")
    p.add_argument('--image_size', type=int, default=300)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    tk.import_simulation_modules()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    meta, frames_all = [], []
    for task in TASKS:
        env = tk.utils.make_metaworld_env(SimpleNamespace(env=f'metaworld_{task}', seed=args.seed + 20000))
        agent = tk.make_agent(env.observation_space.shape[0], env.action_space.shape[0],
                              [float(env.action_space.low.min()), float(env.action_space.high.max())], 'cpu')
        kept = {1.0: 0, 0.0: 0}
        # good policies as they are, good policies with random actions mixed in (near-misses), bad policies
        sources = [(n, f, 0.0) for n, f in actors(args.runs, task, GOOD, True)] + \
                  [(n + '+noise', f, 0.3) for n, f in actors(args.runs, task, GOOD, True)] + \
                  [(n, f, 0.0) for n, f in actors(args.runs, task, BAD, False)]
        if not sources:
            print(f"{task}: no finished runs to take policies from")
            continue
        for _ in range(40 * args.per_class):
            if min(kept.values()) >= args.per_class:
                break
            name, path, noise = sources[rng.integers(len(sources))]
            agent.actor.load_state_dict(torch.load(path, map_location='cpu'))
            frames, success, ret = attempt(env, agent, task, args.image_size, noise, rng)
            if kept[success] >= args.per_class:
                continue
            kept[success] += 1
            meta.append(dict(i=len(frames_all), task=task, source=name, success=success, true_return=round(ret, 1)))
            frames_all.append(np.stack(frames))
        print(f"{task}: {kept[1.0]} successes, {kept[0.0]} failures", flush=True)
    np.savez_compressed(os.path.join(args.out, 'frames.npz'), frames=np.stack(frames_all))
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print(f"saved {len(meta)} attempts to {args.out}")


if __name__ == '__main__':
    main()
