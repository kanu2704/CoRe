#!/usr/bin/env python3
"""Copy finished runs from earlier sessions / teammates into one runs folder.

Sources can be folders (searched recursively) or .zip files made by
kaggle/save_results.sh. A run is copied unless the destination already has a
finished copy of the same (task, mode, seed).

  python scripts/merge_runs.py /kaggle/input /tmp/prev_runs.zip --into /kaggle/working/runs
"""
import argparse
import glob
import json
import os
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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('sources', nargs='+')
    p.add_argument('--into', required=True)
    args = p.parse_args()
    into = os.path.abspath(args.into)
    copied = skipped = 0
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
            if os.path.exists(os.path.join(dest, 'done.json')) or not src_done:
                skipped += 1
                continue
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.copytree(d, dest)
            copied += 1
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[merge] copied {copied} finished runs into {into} ({skipped} skipped: unfinished or already there)")


if __name__ == '__main__':
    main()
