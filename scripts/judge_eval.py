#!/usr/bin/env python3
"""How well does a VLM judge success? Grades every attempt of a judge test set
(scripts/make_judge_set.py) and compares with the simulator's success flag.

Uses the VLM_* settings (VLM_API_KEY, VLM_BASE_URL, VLM_MODEL, VLM_RPM), e.g. Gemini's free tier:
  set -a; source ~/.psvm_keys.env; set +a
  python scripts/judge_eval.py runs_mac/judge_set --styles success,stages
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from psvm.judge import SuccessJudge  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('set_dir')
    p.add_argument('--styles', default='success,stages')
    p.add_argument('--limit', type=int, default=0, help="grade only the first N attempts (0 = all)")
    args = p.parse_args()
    meta = json.load(open(os.path.join(args.set_dir, 'meta.json')))
    frames = np.load(os.path.join(args.set_dir, 'frames.npz'))['frames']
    if args.limit:
        meta = meta[:args.limit]
    model = os.environ.get('VLM_MODEL', '?')
    results = []
    for style in args.styles.split(','):
        judges = {}
        for m in meta:
            env = f"metaworld_{m['task']}"
            judge = judges.setdefault(env, SuccessJudge(env, ROOT, style=style))
            q = judge.grade(list(frames[m['i']]))
            results.append(dict(m, style=style, q=q, answer=judge.last_answer))
            print(f"{style:8s} {m['task']:24s} {m['source']:28s} success {int(m['success'])}  q {q}  ({judge.last_answer!r})", flush=True)

    print(f"\n=== {model} ===")
    by = defaultdict(list)
    for r in results:
        by[(r['style'], r['task'])].append(r)
        by[(r['style'], 'all tasks')].append(r)
    for (style, task), rs in sorted(by.items()):
        ok = [r for r in rs if r['q'] is not None]
        succ = [r for r in ok if r['success']]
        fail = [r for r in ok if not r['success']]
        # a 'success' call: yes, or the last stage (stage 4 = done) for the checklist
        called = lambda r: r['q'] >= 1.0
        tpr = sum(called(r) for r in succ) / max(1, len(succ))
        fpr = sum(called(r) for r in fail) / max(1, len(fail))
        line = (f"{style:8s} {task:24s} n={len(rs):3d} unanswered={len(rs) - len(ok)}  "
                f"successes called success {100 * tpr:5.1f}%  failures called success {100 * fpr:5.1f}%  "
                f"balanced accuracy {50 * (tpr + 1 - fpr):5.1f}%")
        if style != 'success' and succ and fail:
            line += f"  mean q: successes {np.mean([r['q'] for r in succ]):.2f}, failures {np.mean([r['q'] for r in fail]):.2f}"
        if len(fail) > 2:
            qs, ret = [r['q'] for r in fail], [r['true_return'] for r in fail]
            if np.std(qs) > 0:
                line += f"  | among failures, corr(q, real progress) {np.corrcoef(qs, ret)[0, 1]:+.2f}"
        print(line)
    out = os.path.join(args.set_dir, f"judge_results_{model.replace('/', '_')}.json")
    json.dump(results, open(out, 'w'), indent=1)
    print(f"answers saved to {out}")


if __name__ == '__main__':
    main()
