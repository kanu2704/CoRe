#!/usr/bin/env python3
"""Merge finished runs (from any number of folders / Kaggle sessions / accounts) into
comparison tables against the paper's numbers, gap closure G, verdicts and learning curves.

  python scripts/summarize.py runs other_session_runs ... --out results
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

TASK_ORDER = ['peg-insert-side-v2', 'hammer-v2', 'button-press-topdown-v2',
              'soccer-v2', 'sweep-into-v2', 'drawer-open-v2', 'dial-turn-v2']

# CoRe paper (ICML 2026), final success rate %, Tables 1 and 2
PAPER = {
    'env_dense': {'soccer-v2': 100.0, 'sweep-into-v2': 98.0, 'drawer-open-v2': 100.0, 'button-press-topdown-v2': 100.0,
                  'dial-turn-v2': 100.0, 'hammer-v2': 97.3, 'peg-insert-side-v2': 100.0},
    'env_sparse': {'soccer-v2': 100.0, 'sweep-into-v2': 60.0, 'drawer-open-v2': 100.0, 'button-press-topdown-v2': 66.7,
                   'dial-turn-v2': 100.0, 'hammer-v2': 60.7, 'peg-insert-side-v2': 0.0},
    'frm': {'soccer-v2': '100 / 100', 'sweep-into-v2': '36.0 / 93.3', 'drawer-open-v2': '100 / 100',
            'button-press-topdown-v2': '17.3 / 59.3', 'dial-turn-v2': '34.7 / 89.0', 'hammer-v2': '34.4 / 54.0',
            'peg-insert-side-v2': '50.0 / 88.0'},  # FRM0 (first reward) / FRM4 (after 4 LLM refinements)
    'core': {'soccer-v2': 100.0, 'sweep-into-v2': 97.3, 'drawer-open-v2': 100.0, 'button-press-topdown-v2': 100.0,
             'dial-turn-v2': 98.0, 'hammer-v2': 98.0, 'peg-insert-side-v2': 100.0},
    'rrm_only': {'soccer-v2': 100.0, 'sweep-into-v2': 94.0, 'drawer-open-v2': 88.7, 'button-press-topdown-v2': 8.7,
                 'dial-turn-v2': 36.0, 'hammer-v2': 49.3, 'peg-insert-side-v2': 36.0},
}
LABELS = {
    'env_dense': 'A1 Env Dense', 'env_sparse': 'A2 Env Sparse', 'frm': 'A3 FRM only', 'svm': 'A4 Oracle SVM',
    'svm_sparse': 'A5 SVM + sparse', 'svm_seeded': 'A6 SVM seeded (diag.)',
    'core': 'B1 CoRe re-run', 'rrm_only': 'B2 RRM only', 'rrm_sparse': 'B3 RRM + success', 'rrm_svm': 'B4 RRM + oracle SVM',
    'rrm_psvm': 'B5 RRM + P-SVM (VLM-graded q)',
}
PHASE1 = ['env_dense', 'env_sparse', 'frm', 'svm', 'svm_sparse', 'svm_seeded']
PHASE2 = ['core', 'rrm_only', 'rrm_sparse', 'rrm_svm', 'rrm_psvm']


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_runs(roots):
    runs = {}
    for root in roots:
        for cfg_path in glob.glob(os.path.join(root, '**', 'config.json'), recursive=True):
            d = os.path.dirname(cfg_path)
            with open(cfg_path) as f:
                cfg = json.load(f)
            a = cfg['args']
            done_path = os.path.join(d, 'done.json')
            summary = json.load(open(done_path)) if os.path.exists(done_path) else {}
            evals = read_jsonl(os.path.join(d, 'eval.jsonl'))
            if not evals:
                continue
            key = (a['task'], a['mode'], a['seed'])
            run = dict(task=a['task'], mode=a['mode'], seed=a['seed'], dir=d, complete=bool(summary.get('complete')),
                       steps=np.array([e['step'] for e in evals]), sr=np.array([e['success_rate'] for e in evals]),
                       hours=summary.get('hours'), first_success=summary.get('first_success_step'),
                       svm_on=summary.get('svm_on_step'), labels=cfg['args'].get('labels', 'vlm'),
                       vlm_model=cfg.get('vlm_model'), smoke=bool(cfg['args'].get('set')),
                       continued_from=summary.get('continued_from'))
            if key not in runs or (run['complete'] and not runs[key]['complete']):
                runs[key] = run
    return list(runs.values())


def metrics(run):
    sr, steps = run['sr'], run['steps']
    t80 = next((int(s) for s, v in zip(steps, sr) if v >= 80), None)
    return dict(final=float(sr[-1]), last5=float(sr[-5:].mean()), auc=float(sr.mean()), t80=t80)


def fmt(vals):
    if not vals:
        return '-'
    v = np.array(vals)
    return f"{v.mean():.1f} ± {v.std():.1f} [{v.min():.0f}–{v.max():.0f}]"


def group(runs, include_smoke=False):
    g = defaultdict(list)
    for r in runs:
        if r['complete'] and (include_smoke or not r['smoke']):
            g[(r['task'], r['mode'])].append(r)
    return g


def table(task, modes, g, lines):
    lines.append(f"| Run | n | Final success (mean ± std [min–max]) | Last-5 evals | Learning-curve area | Paper |")
    lines.append("|---|---|---|---|---|---|")
    for m in modes:
        rs = g.get((task, m), [])
        if not rs:
            continue
        ms = [metrics(r) for r in rs]
        paper = PAPER.get(m, {}).get(task, '-')
        lines.append(f"| {LABELS[m]} | {len(rs)} | {fmt([x['final'] for x in ms])} | {fmt([x['last5'] for x in ms])} | "
                     f"{fmt([x['auc'] for x in ms])} | {paper} |")


def mean_final(g, task, mode, key='final'):
    rs = g.get((task, mode), [])
    return float(np.mean([metrics(r)[key] for r in rs])) if rs else None


def phase1_verdict(task, g):
    svm = g.get((task, 'svm'), [])
    if not svm:
        return None
    a2, a3, a4 = (mean_final(g, task, m) for m in ('env_sparse', 'frm', 'svm'))
    on = [r['svm_on'] for r in svm]
    started = sum(1 for s in on if s is not None)
    if started < len(svm) / 2:
        return f"Can't get started: SVM switched on in {started}/{len(svm)} seeds. No verdict; check A6 and Phase 2."
    early = np.median([s for s in on if s is not None])
    msg = f"SVM switched on in {started}/{len(svm)} seeds (median step {early:.0f}). "
    if a3 is not None and a2 is not None and a4 >= a3 and a4 >= a2 + 20:
        return msg + f"Mechanism strong: A4 {a4:.1f} ≥ A3 {a3:.1f} and ≥ A2 {a2:.1f} + 20."
    if a2 is not None and a4 < a2 + 20:
        return msg + f"Mechanism weak: A4 {a4:.1f} is not clearly above A2 (success flag as reward) {a2:.1f}."
    return msg + f"Mixed: A4 {a4:.1f}, A3 {a3}, A2 {a2}."


def phase2_verdict(task, g):
    b1, b2, b3, b4 = (mean_final(g, task, m) for m in ('core', 'rrm_only', 'rrm_sparse', 'rrm_svm'))
    if None in (b1, b2, b4):
        return None, None
    gap = b1 - b2
    if gap < 15:
        return None, f"Gap CoRe − RRM-only is only {gap:.1f} points: G can't be measured on this task."
    G = (b4 - b2) / gap
    if G >= 0.5 and b3 is not None and b4 - b3 >= 10:
        v = f"Strong enough here: G = {G:.2f} and B4 beats the success-bit control by {b4 - b3:.1f} points."
    elif G >= 0.5 and b3 is None:
        v = f"Promising: G = {G:.2f}. Run B3 (RRM + success) to rule out the success-bit leak."
    elif G >= 0.5:
        v = f"Unclear: G = {G:.2f} but B4 is within 10 points of the success-bit control ({b3:.1f})."
    else:
        v = f"Not strong enough here: G = {G:.2f} < 0.5."
    return G, v


def plot(task, modes, g, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    drawn = False
    for m in modes:
        rs = g.get((task, m), [])
        if not rs:
            continue
        n = min(len(r['sr']) for r in rs)
        steps = rs[0]['steps'][:n]
        y = np.stack([r['sr'][:n] for r in rs])
        ax.plot(steps, y.mean(0), label=f"{LABELS[m]} (n={len(rs)})")
        ax.fill_between(steps, y.mean(0) - y.std(0), y.mean(0) + y.std(0), alpha=0.15)
        drawn = True
    if not drawn:
        plt.close(fig)
        return False
    ax.set_xlabel('environment steps')
    ax.set_ylabel('success rate (%)')
    ax.set_ylim(-2, 102)
    ax.set_title(task)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('roots', nargs='+')
    p.add_argument('--out', default='results')
    p.add_argument('--include_smoke', action='store_true', help="also count short test runs (pipeline checks only)")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs = load_runs(args.roots)
    g = group(runs, args.include_smoke)
    tasks = [t for t in TASK_ORDER if any(k[0] == t for k in g)]
    lines = ["# P-SVM vs CoRe: results", ""]
    models = sorted({r['vlm_model'] for r in runs if r['mode'] in PHASE2 and r['labels'] == 'vlm'} - {None})
    if models:
        lines.append(f"VLM used for RRM labels: {', '.join(models)}" + ("  ⚠ more than one model!" if len(models) > 1 else ""))
    scripted = sum(1 for r in runs if r['mode'] in PHASE2 and r['labels'] == 'scripted' and not r['smoke'])
    if scripted:
        lines.append(f"⚠ {scripted} Phase 2 runs used scripted labels; they are not comparable to the paper.")
    continued = [r for r in runs if r.get('continued_from') and not r['smoke']]
    if continued:
        lines.append("Continued after an early stop (replay buffer restarted empty at that step): " + ', '.join(
            f"{r['mode']}/{r['task']}/s{r['seed']} from step {', '.join(map(str, r['continued_from']))}" for r in continued))
    incomplete = [r for r in runs if not r['complete'] and not r['smoke']]
    if incomplete:
        lines.append(f"Incomplete runs left out: " + ', '.join(f"{r['mode']}/{r['task']}/s{r['seed']}" for r in incomplete))
    lines.append("")

    verdicts = []
    for task in tasks:
        lines += [f"## {task}", ""]
        if any((task, m) in g for m in PHASE1):
            lines.append("**Phase 1 (state only, no VLM)** — paper FRM column is FRM0 / FRM4")
            lines.append("")
            table(task, PHASE1, g, lines)
            v = phase1_verdict(task, g)
            if v:
                lines += ["", f"Phase 1 verdict: {v}"]
            if plot(task, PHASE1, g, os.path.join(args.out, f'phase1_{task}.png')):
                lines.append(f"![phase 1 {task}](phase1_{task}.png)")
            lines.append("")
        if any((task, m) in g for m in PHASE2):
            lines.append("**Phase 2 (RRM + VLM labels)**")
            lines.append("")
            table(task, PHASE2, g, lines)
            G, v = phase2_verdict(task, g)
            if v:
                lines += ["", f"Phase 2: {v}"]
                verdicts.append((task, G))
            if plot(task, PHASE2, g, os.path.join(args.out, f'phase2_{task}.png')):
                lines.append(f"![phase 2 {task}](phase2_{task}.png)")
            lines.append("")

    measured = [(t, G) for t, G in verdicts if G is not None]
    if measured:
        strong = sum(1 for _, G in measured if G >= 0.5)
        lines.append("## Overall")
        lines.append(f"G ≥ 0.5 on {strong} of {len(measured)} measured tasks. The plan's rule needs at least 2 tasks "
                     f"(and B4 clearly above B3) before calling P-SVM strong enough.")
    lines.append("")
    lines.append("Noise rule: with 3–5 seeds only trust differences of about 20 points or more, or seed ranges that don't overlap.")

    text = "\n".join(lines)
    with open(os.path.join(args.out, 'summary.md'), 'w') as f:
        f.write(text + "\n")
    with open(os.path.join(args.out, 'runs.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['task', 'mode', 'seed', 'complete', 'final', 'last5', 'auc', 'steps_to_80', 'first_success_step',
                    'svm_on_step', 'hours', 'labels', 'vlm_model', 'dir'])
        for r in sorted(runs, key=lambda r: (r['task'], r['mode'], r['seed'])):
            m = metrics(r)
            w.writerow([r['task'], r['mode'], r['seed'], r['complete'], m['final'], m['last5'], m['auc'], m['t80'],
                        r['first_success'], r['svm_on'], r['hours'], r['labels'], r['vlm_model'], r['dir']])
    print(text)
    print(f"\nwrote {args.out}/summary.md, runs.csv and plots")


if __name__ == '__main__':
    main()
