#!/usr/bin/env python3
"""Copy finished runs from earlier sessions / teammates into one runs folder.

Sources can be folders (searched recursively) or .zip files made by
kaggle/save_results.sh. A finished run is copied unless the destination already has a
finished copy of the same (task, mode, seed). A run that stopped early with its networks
saved is copied too (unless the destination is finished or further along), so that
train_kaggle.py can continue it.

  python scripts/merge_runs.py /kaggle/input /tmp/prev_runs.zip --into /kaggle/working/runs
"""
import argparse
import glob
import json
import os
import re
import shutil
import tempfile
import zipfile


def run_dirs(root):
    for cfg in glob.glob(os.path.join(root, '**', 'config.json'), recursive=True):
        d = os.path.dirname(cfg)
        try:
            a = json.load(open(cfg))['args']
            yield d, a['task'], a['mode'], a['seed'], bool(a.get('set'))
        except (KeyError, ValueError, OSError):
            continue


def stopped_step(d):
    """Step a run can continue from, as train_kaggle.stopped_early() finds it (incomplete.json, or the
    stop a continued-then-killed run started from), if its networks were saved there."""
    if os.path.exists(os.path.join(d, 'done.json')):
        return None
    step = None
    try:
        step = int(json.load(open(os.path.join(d, 'incomplete.json')))['steps'])
    except (OSError, ValueError, KeyError):
        saved = [int(re.findall(r'\d+', os.path.basename(p))[0])
                 for p in glob.glob(os.path.join(d, 'stopped_at_*.json')) + glob.glob(os.path.join(d, 'saved_at_*.json'))]
        if saved:
            step = max(saved)
        else:
            try:
                cont = [json.loads(l) for l in open(os.path.join(d, 'events.jsonl')) if '"continued"' in l]
                step = int(cont[-1]['from_step']) if cont else None
            except (OSError, ValueError, KeyError):
                step = None
    return step if step and os.path.exists(os.path.join(d, 'models', f'actor_{step}.pt')) else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('sources', nargs='+')
    p.add_argument('--into', required=True)
    args = p.parse_args()
    into = os.path.abspath(args.into)
    copied = stopped = skipped = 0
    tmp = tempfile.mkdtemp()
    roots = []
    for src in args.sources:
        for path in ([src] if not os.path.isdir(src) else [src] + glob.glob(os.path.join(src, '**', '*.zip'), recursive=True)):
            if path.endswith('.zip') and os.path.isfile(path):
                out = os.path.join(tmp, str(len(roots)))
                with zipfile.ZipFile(path) as z:
                    z.extractall(out)
                roots.append(out)
            elif os.path.isdir(path):
                roots.append(path)
    for root in roots:
        for d, task, mode, seed, smoke in run_dirs(root):
            if os.path.abspath(d).startswith(into + os.sep):
                continue
            dest = os.path.join(into, '_smoke' if smoke else '', task, mode, f'seed{seed}')
            src_done = os.path.exists(os.path.join(d, 'done.json'))
            src_step = None if src_done else stopped_step(d)
            if os.path.exists(os.path.join(dest, 'done.json')) or (not src_done and (
                    src_step is None or (stopped_step(dest) or -1) >= src_step)):
                skipped += 1
                continue
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.copytree(d, dest)
            if src_done:
                copied += 1
            else:
                stopped += 1
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[merge] copied {copied} finished runs into {into} ({skipped} skipped: unfinished or already there)")
    if stopped:
        print(f"[merge] copied {stopped} runs that stopped early; train_kaggle.py continues them where they stopped")


if __name__ == '__main__':
    main()
