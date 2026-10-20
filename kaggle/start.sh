#!/usr/bin/env bash
# First command in every new Kaggle session (VS Code terminal connected through the tunnel).
#
#   bash /kaggle/input/*/kaggle/start.sh              # code attached to the notebook as a dataset
#   bash /kaggle/working/CoRe/kaggle/start.sh         # code you uploaded / unzipped there yourself
#
# It copies the code to /kaggle/working/CoRe, builds the pinned Python environment (~3 min),
# and merges earlier results so finished runs are skipped. Options:
#   --runs-dataset USER/SLUG   also download earlier results saved with save_results.sh --upload
set -euo pipefail
WORK="${PSVM_WORK:-/kaggle/working}"   # overridable for testing off Kaggle
INPUT="${PSVM_INPUT:-/kaggle/input}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DEST=${WORK}/CoRe
RUNS=${WORK}/runs
RUNS_DATASET=""
while [ $# -gt 0 ]; do
  case "$1" in
    --runs-dataset) RUNS_DATASET="$2"; shift 2 ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done

# 1. code -> /kaggle/working/CoRe (/kaggle/input is read-only)
if [ "$SRC" != "$DEST" ]; then
  mkdir -p "$DEST"
  cp -r "$SRC/." "$DEST/"
  chmod -R u+w "$DEST"   # dataset files are read-only
  echo "[start] code copied from $SRC"
fi
cd "$DEST"
chmod +x kaggle/*.sh

# 2. pinned Python environment in /tmp (rebuilt each session, never saved as output)
export PSVM_PY_FILE="${PSVM_PY_FILE:-/tmp/psvm_python}"
bash kaggle/setup.sh
PY="$(cat "$PSVM_PY_FILE")"

# 3. earlier results: anything attached under /kaggle/input, plus an optional results dataset
mkdir -p "$RUNS"
SOURCES=()
[ -d ${INPUT} ] && SOURCES+=(${INPUT})
if [ -n "$RUNS_DATASET" ]; then
  set -a; [ -f "$HOME/.psvm_keys.env" ] && . "$HOME/.psvm_keys.env"; set +a
  [ -n "${KAGGLE_API_TOKEN:-}" ] || unset KAGGLE_API_TOKEN
  [ -n "${KAGGLE_KEY:-}" ] || unset KAGGLE_KEY        # empty values would confuse the CLI
  uv pip install -q --python "$PY" -U kaggle   # recent CLI understands the new KAGGLE_API_TOKEN
  KAGGLE_BIN="$(dirname "$PY")/kaggle"
  rm -rf /tmp/prev_runs && mkdir -p /tmp/prev_runs
  "$KAGGLE_BIN" datasets download -d "$RUNS_DATASET" -p /tmp/prev_runs --unzip
  SOURCES+=(/tmp/prev_runs)
fi
if [ ${#SOURCES[@]} -gt 0 ]; then "$PY" scripts/merge_runs.py "${SOURCES[@]}" --into "$RUNS"; fi

# 4. API keys file
if [ ! -f "$HOME/.psvm_keys.env" ]; then
  cp kaggle/keys.example.env "$HOME/.psvm_keys.env"
  echo "[start] created ~/.psvm_keys.env: open it (code ~/.psvm_keys.env) and paste your keys"
fi

cat <<EOF

[start] ready. Next, in this terminal:
  cd $DEST
  source kaggle/env.sh                 # loads keys + Python
  \$PY scripts/check_setup.py --render --api
  bash kaggle/run.sh smoke_phase2 --labels scripted
EOF
