#!/usr/bin/env python3
"""Try the P-SVM success judge on a running Phase 2 run's latest episodes, while it trains.

Reads the episodes' frames from the run's replay frame file (frame_backend disk, before the
images are dropped after the last label round), grades each episode with each judging style
and prints Qwen's raw answers next to the grades. Also writes judge_check.png (one row of
frames per episode) next to the run, to look at.

  source kaggle/env.sh
  $PY scripts/judge_check.py /kaggle/working/runs/peg-insert-side-v2/rrm_psvm/seed0 --episodes 6
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import psvm.judge as J  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_dir')
    p.add_argument('--episodes', type=int, default=6)
    p.add_argument('--styles', default='stages,scale')
    p.add_argument('--frames_dir', default=os.environ.get('PSVM_DISK_DIR', '/tmp/psvm_frames'))
    args = p.parse_args()

    cfg = json.load(open(os.path.join(args.run_dir, 'config.json')))
    task, size = cfg['args']['task'], cfg['cfg']['image_size']
    pid = open(os.path.join(args.run_dir, 'running.pid')).read().strip()
    files = glob.glob(os.path.join(args.frames_dir, f'replay_{pid}_*.u8'))
    if len(files) != 1:
        raise SystemExit(f"replay frames of process {pid} not found in {args.frames_dir} "
                         "(frame_backend must be disk, and images are dropped after the last label round)")
    frames = np.memmap(files[0], dtype=np.uint8, mode='r').reshape(-1, size, size, 3)
    events = [json.loads(l) for l in open(os.path.join(args.run_dir, 'events.jsonl'))]
    offset = max([e['from_step'] for e in events if e['event'] == 'continued'], default=0)  # replay restarts on continue
    episodes = [json.loads(l) for l in open(os.path.join(args.run_dir, 'episodes.jsonl'))][-args.episodes:]

    raw = []
    orig = J.parse_grade
    J.parse_grade = lambda t, top=10: (raw.append(t), orig(t, top))[1]
    judges = {s: J.SuccessJudge(f'metaworld_{task}', ROOT, style=s) for s in args.styles.split(',')}
    rows = []
    for ep in episodes:
        end = ep['step'] - offset  # replay index after the episode's last step
        idx = [end - 251, end - 126, end - 1]
        if min(idx) < 0:
            continue
        shots = [np.array(frames[i]) for i in idx]
        line = f"episode {ep['episode']:4d} (step {ep['step']}, solved {int(ep['success'])}, logged q {ep.get('q')})"
        for name, judge in judges.items():
            raw.clear()
            q = judge.grade(shots)
            line += f" | {name}: {q} (Qwen said {raw[-1]!r})" if raw else f" | {name}: {q}"
        print(line, flush=True)
        rows.append(np.concatenate([cv2.resize(s, (150, 150)) for s in shots], axis=1))
    if rows:
        out = os.path.join(args.run_dir, 'judge_check.png')
        cv2.imwrite(out, cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))
        print(f"frames (1/2, 3/4, end of each episode): {out}")


if __name__ == '__main__':
    main()
