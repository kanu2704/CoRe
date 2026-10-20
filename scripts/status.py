#!/usr/bin/env python3
"""Show progress of all runs in a runs folder (plus RAM / GPU use on Linux).

  python scripts/status.py /kaggle/working/runs
"""
import glob
import json
import os
import subprocess
import sys


def last_line(path):
    try:
        with open(path) as f:
            lines = [l for l in f if l.strip()]
        return json.loads(lines[-1]) if lines else None
    except OSError:
        return None


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else 'runs'
    rows = []
    cfgs = sorted(glob.glob(os.path.join(root, '*', '*', 'seed*', 'config.json')))
    cfgs += sorted(glob.glob(os.path.join(root, '_smoke', '*', '*', 'seed*', 'config.json')))
    for cfg in cfgs:
        d = os.path.dirname(cfg)
        task, mode, seed = d.split(os.sep)[-3:]
        if os.sep + '_smoke' + os.sep in d:
            mode = mode + '*'
        if os.path.exists(os.path.join(d, 'done.json')):
            state = 'done'
        elif os.path.exists(os.path.join(d, 'incomplete.json')):
            state = 'stopped early'
        else:
            state = 'running/failed'
        ev = last_line(os.path.join(d, 'eval.jsonl'))
        ep = last_line(os.path.join(d, 'episodes.jsonl'))
        step = ep['step'] if ep else 0
        sr = f"{ev['success_rate']:.0f}%" if ev else '-'
        hours = f"{ev['hours']:.2f}" if ev else '-'
        rows.append((task, mode, seed, state, step, sr, hours, ep.get('labeled_feedback', '') if ep else ''))
    print(f"{'task':26s} {'mode':11s} {'seed':6s} {'state':15s} {'step':>7s} {'last eval':>9s} {'hours':>6s} {'labels':>6s}")
    for r in rows:
        print(f"{r[0]:26s} {r[1]:11s} {r[2]:6s} {r[3]:15s} {r[4]:7d} {r[5]:>9s} {r[6]:>6s} {str(r[7]):>6s}")
    if not rows:
        print("(no runs yet)")
    elif any(r[1].endswith('*') for r in rows):
        print("* = smoke test (short, not part of the results)")
    for cmd in (['free', '-g'], ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                                 '--format=csv,noheader']):
        try:
            print(subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass


if __name__ == '__main__':
    main()
