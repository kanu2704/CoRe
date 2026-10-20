#!/usr/bin/env python3
"""Run a plan from scripts/plans.py inside one Kaggle session (or on any machine).

* skips runs that already finished (done.json) here or in --done_dirs
  (e.g. results from earlier sessions or teammates' accounts, attached as a dataset)
* runs up to --parallel runs at once, spread over --gpus
* never starts a run that cannot finish before --session_hours, and tells each
  run to stop cleanly before the session ends

Examples:
  python scripts/launch.py --plan phase1 --parallel 2 --threads 2          # CPU session
  python scripts/launch.py --plan phase2_peg --parallel 1 --est_hours 4    # GPU session
  python scripts/launch.py --plan smoke_phase2 --labels scripted           # free pipeline check
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from plans import PLANS, SMOKE_SETTINGS  # noqa: E402


def finished(out_dirs, mode, task, seed):
    """A real (non-smoke) run of this variant already finished in one of out_dirs."""
    for d in out_dirs:
        run = os.path.join(d, task, mode, f'seed{seed}')
        if os.path.exists(os.path.join(run, 'done.json')):
            try:
                if not json.load(open(os.path.join(run, 'config.json')))['args'].get('set'):
                    return True
            except (OSError, ValueError, KeyError):
                pass
    return False


def seed_actor_for(out_dirs, task):
    """svm_seeded uses a trained Env Dense actor (Phase 1, A1) that solves the task."""
    for d in out_dirs:
        for seed_dir in sorted(glob.glob(os.path.join(d, task, 'env_dense', 'seed*'))):
            actors = sorted(glob.glob(os.path.join(seed_dir, 'models', 'actor_*.pt')))
            if actors and os.path.exists(os.path.join(seed_dir, 'done.json')):
                return actors[-1]
    return ''


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--plan', required=True, choices=sorted(PLANS))
    p.add_argument('--out', default='runs')
    p.add_argument('--done_dirs', nargs='*', default=[], help="other result folders to treat as finished")
    p.add_argument('--parallel', type=int, default=1)
    p.add_argument('--gpus', default='', help="comma list of GPU ids, e.g. 0,1 (default: all visible / none)")
    p.add_argument('--threads', type=int, default=2, help="torch CPU threads per run")
    p.add_argument('--session_hours', type=float, default=11.5, help="stop launching before this (Kaggle: 12 h)")
    p.add_argument('--est_hours', type=float, default=3.0, help="expected hours per run (updated as runs finish)")
    p.add_argument('--labels', default='vlm', choices=['vlm', 'scripted'])
    p.add_argument('--vlm_workers', type=int, default=4)
    p.add_argument('--only', nargs='*', default=[], help="filters like mode=svm task=hammer-v2 seed=0")
    p.add_argument('--max_runs', type=int, default=0,
                   help="start at most this many runs (e.g. 1 per day with a free-tier VLM key)")
    p.add_argument('--extra', nargs=argparse.REMAINDER, default=[], help="passed through to train_kaggle.py")
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()

    out = os.path.abspath(args.out)
    out_dirs = [out] + [os.path.abspath(d) for d in args.done_dirs]
    queue = []
    for mode, task, seed in PLANS[args.plan]:
        f = dict(kv.split('=', 1) for kv in args.only)
        if f.get('mode', mode) != mode or f.get('task', task) != task or f.get('seed', str(seed)) != str(seed):
            continue
        if args.plan not in SMOKE_SETTINGS and finished(out_dirs, mode, task, seed):
            continue
        queue.append((mode, task, seed))

    print(f"[launch] plan {args.plan}: {len(PLANS[args.plan])} runs, {len(queue)} still to do")
    if args.max_runs:
        queue = queue[:args.max_runs]
        print(f"[launch] --max_runs {args.max_runs}: this session starts {len(queue)}")
    for q in queue:
        print("   ", *q)
    if args.dry_run or not queue:
        return

    gpus = [g for g in args.gpus.split(',') if g] or [None]
    start = time.time()
    est = min(args.est_hours, 0.5) if args.plan in SMOKE_SETTINGS else args.est_hours
    running = []  # (proc, run, gpu, t0, logfile)
    results = []

    def elapsed_h():
        return (time.time() - start) / 3600

    while queue or running:
        # collect finished runs
        for item in list(running):
            proc, run, gpu, t0, log = item
            if proc.poll() is not None:
                running.remove(item)
                log.close()
                hours = (time.time() - t0) / 3600
                mode, task, seed = run
                ok = os.path.exists(os.path.join(out, '_smoke' if args.plan in SMOKE_SETTINGS else '', task, mode,
                                                 f'seed{seed}', 'done.json'))
                results.append((run, proc.returncode, hours, ok))
                done_hours = [r[2] for r in results if r[3]]
                if done_hours:  # plan with the slowest finished run so far
                    est = max(done_hours)
                print(f"[launch] finished {run} in {hours:.2f} h, exit {proc.returncode}, complete={ok}")

        # start new runs
        free_gpus = [g for g in gpus if sum(1 for r in running if r[2] == g) < max(1, args.parallel // len(gpus))]
        while queue and len(running) < args.parallel and free_gpus:
            remaining = args.session_hours - elapsed_h()
            if remaining < est:
                print(f"[launch] {remaining:.1f} h left in the session < {est:.1f} h per run: not starting more")
                queue = []
                break
            mode, task, seed = queue.pop(0)
            gpu = free_gpus[0]
            cmd = [sys.executable, os.path.join(ROOT, 'train_kaggle.py'), '--mode', mode, '--task', task,
                   '--seed', str(seed), '--out', out, '--threads', str(args.threads),
                   '--labels', args.labels, '--vlm_workers', str(args.vlm_workers),
                   '--time_limit_hours', f'{max(remaining - 0.25, 0.1):.2f}']
            if args.plan in SMOKE_SETTINGS:
                cmd += ['--overwrite', '--set'] + SMOKE_SETTINGS[args.plan]
            if mode == 'svm_seeded':
                actor = seed_actor_for(out_dirs, task)
                if actor:
                    cmd += ['--seed_actor', actor]
                elif task != 'button-press-topdown-v2':
                    print(f"[launch] skip svm_seeded {task} s{seed}: run env_dense first (its actor seeds the SVM)")
                    continue
            cmd += args.extra
            env = dict(os.environ)
            env['PYTHONUNBUFFERED'] = '1'  # progress lines reach stdout.log immediately (tail -f works)
            if gpu is not None:
                env['CUDA_VISIBLE_DEVICES'] = gpu
            run_dir = os.path.join(out, '_smoke' if args.plan in SMOKE_SETTINGS else '', task, mode, f'seed{seed}')
            os.makedirs(run_dir, exist_ok=True)
            log = open(os.path.join(run_dir, 'stdout.log'), 'w')
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
            running.append((proc, (mode, task, seed), gpu, time.time(), log))
            print(f"[launch] started {mode} {task} seed{seed}" + (f" on GPU {gpu}" if gpu else ""))
            free_gpus = [g for g in gpus if sum(1 for r in running if r[2] == g) < max(1, args.parallel // len(gpus))]
        time.sleep(10)

    print(f"[launch] session done after {elapsed_h():.2f} h")
    for run, code, hours, ok in results:
        print(f"   {'ok  ' if ok else 'FAIL'} {run}  {hours:.2f} h")
    with open(os.path.join(out, f'launch_{args.plan}_{int(start)}.json'), 'w') as f:
        json.dump([dict(run=r, exit=c, hours=h, complete=ok) for r, c, h, ok in results], f, indent=2)


if __name__ == '__main__':
    main()
