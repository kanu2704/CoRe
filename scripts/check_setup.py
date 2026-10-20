#!/usr/bin/env python3
"""Check a machine (e.g. a Kaggle session) before spending quota.

  python scripts/check_setup.py            # versions, CPU, RAM, disk, MetaWorld
  python scripts/check_setup.py --render   # + headless rendering, frame compression, Phase 2 RAM estimate
  python scripts/check_setup.py --api      # + one test call to the VLM (and the LLM if configured)
"""
import argparse
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def meminfo_gb():
    try:
        with open('/proc/meminfo') as f:
            info = {l.split(':')[0]: int(l.split()[1]) for l in f}
        return info['MemTotal'] / 1e6, info['MemAvailable'] / 1e6
    except OSError:
        return None, None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--render', action='store_true')
    p.add_argument('--api', action='store_true')
    p.add_argument('--task', default='peg-insert-side-v2')
    args = p.parse_args()

    if args.render:
        os.environ.setdefault('MUJOCO_GL', 'egl' if sys.platform.startswith('linux') else 'glfw')
        if os.environ['MUJOCO_GL'] in ('egl', 'osmesa'):
            os.environ.setdefault('PYOPENGL_PLATFORM', os.environ['MUJOCO_GL'])
    else:
        os.environ.setdefault('MUJOCO_GL', 'disable')

    import numpy as np
    import torch
    import mujoco
    print(f"python {sys.version.split()[0]}  torch {torch.__version__}  numpy {np.__version__}  mujoco {mujoco.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}", end='')
    if torch.cuda.is_available():
        print("  GPUs: " + ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())))
    else:
        print()
    total, avail = meminfo_gb()
    print(f"CPU cores: {os.cpu_count()}   RAM total/available: {total and round(total, 1)} / {avail and round(avail, 1)} GB")
    for d in ['.', '/tmp', '/kaggle/working']:
        if os.path.exists(d):
            print(f"disk free at {d}: {shutil.disk_usage(d).free / 1e9:.1f} GB")

    from psvm.compat import install_hydra_shim, reset_env, step_env
    install_hydra_shim()
    from types import SimpleNamespace
    import utils
    env = utils.make_metaworld_env(SimpleNamespace(env=f'metaworld_{args.task}', seed=0))
    obs = reset_env(env)
    t0 = time.time()
    for _ in range(500):
        obs, r, done, info = step_env(env, env.action_space.sample())
    print(f"MetaWorld {args.task}: OK, obs {obs.shape}, {500 / (time.time() - t0):.0f} sim steps/s")

    if args.render:
        import cv2
        from psvm.frames import encode_png, decode_png
        reset_env(env)
        frames, t0 = [], time.time()
        for _ in range(50):
            step_env(env, env.action_space.sample())
            rgb = env.render()[::-1]
            frames.append(cv2.resize(rgb, (300, 300)))
        render_ms = (time.time() - t0) / 50 * 1e3
        t0 = time.time()
        blobs = [encode_png(f) for f in frames]
        enc_ms = (time.time() - t0) / 50 * 1e3
        t0 = time.time()
        ok = all(np.array_equal(decode_png(b), f) for b, f in zip(blobs, frames))
        dec_ms = (time.time() - t0) / 50 * 1e3
        kb = np.mean([len(b) for b in blobs]) / 1e3
        print(f"rendering ({os.environ['MUJOCO_GL']}): {render_ms:.1f} ms/frame; PNG {kb:.0f} KB/frame "
              f"({270 / kb:.1f}x smaller), encode {enc_ms:.1f} ms, decode {dec_ms:.1f} ms, lossless={ok}")
        # Phase 2 peak: replay frames up to the last RRM update (~116k steps) + RRM episode frames (100 x 500)
        # + labelled segments (500 pairs x 2 x 20 frames) + ~3 GB for Python, CUDA, MuJoCo
        est_png = (116_000 + 50_000 + 20_000) * kb / 1e6 + 3
        est_ram = (116_000 + 50_000 + 20_000) * 270 / 1e6 + 3
        print(f"estimated Phase 2 peak RAM per run: ~{est_png:.0f} GB with --frame_backend png, "
              f"~{est_ram:.0f} GB with ram")
        if avail:
            fit = int(avail // est_png)
            print(f"-> about {fit} Phase 2 run(s) fit in this machine's available RAM at once")

    if args.api:
        from FRM.code_agent import make_client
        from RRM.preference_label import model_type as vlm_model
        try:
            c = make_client('vlm')
            r = c.chat.completions.create(model=vlm_model, messages=[{"role": "user", "content": "Reply with the word OK."}],
                                          temperature=0)
            print(f"VLM {vlm_model}: {r.choices[0].message.content.strip()[:40]!r}")
        except Exception as e:
            print(f"VLM {vlm_model}: FAILED: {e}")
        llm = os.environ.get('LLM_MODEL', 'gpt-4.1-mini')
        try:
            c = make_client('llm')
            r = c.chat.completions.create(model=llm, messages=[{"role": "user", "content": "Reply with the word OK."}])
            print(f"LLM {llm}: {r.choices[0].message.content.strip()[:40]!r}  (only needed for 'core' runs)")
        except Exception as e:
            print(f"LLM {llm}: FAILED: {e}  (only needed for 'core' runs)")


if __name__ == '__main__':
    main()
