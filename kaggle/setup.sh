#!/usr/bin/env bash
# Prepare a Kaggle session (GPU or CPU) for train_kaggle.py. Run from anywhere:
#   bash kaggle/setup.sh
# Creates an isolated Python 3.11 environment in /tmp (not saved with the notebook
# output) with the pinned versions in kaggle/requirements.txt.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${PSVM_VENV:-/tmp/psvm-venv}
pip install -q uv
# re-running in the same session reuses the environment (the install below is then a quick no-op)
[ -x "$VENV/bin/python" ] || uv venv -q --python 3.11 "$VENV"
PY="$VENV/bin/python"
uv pip install -q --python "$PY" -r kaggle/requirements.txt

# headless OpenGL (EGL) for MuJoCo rendering in Phase 2
(apt-get update -qq && apt-get install -y -qq libegl1 libgl1 libglib2.0-0) >/dev/null 2>&1 || \
  echo "note: could not apt-get install libegl1/libgl1 (fine for Phase 1)"

echo "$PY" > "${PSVM_PY_FILE:-/tmp/psvm_python}"
"$PY" -c "import torch, mujoco, numpy; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '| mujoco', mujoco.__version__, '| numpy', numpy.__version__)"
echo "Python for runs: $PY"
