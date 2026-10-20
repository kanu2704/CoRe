#!/usr/bin/env bash
# Summarize and package everything in /kaggle/working/runs. Do this before the session ends:
# an interactive Kaggle session's files are lost when it stops.
#
#   bash kaggle/save_results.sh            # -> /kaggle/working/psvm_runs_<time>.zip (right-click > Download in VS Code)
#   bash kaggle/save_results.sh --upload   # also save it as your private Kaggle dataset <KAGGLE_USERNAME>/psvm-runs
#                                          #   (next session: bash .../start.sh --runs-dataset <KAGGLE_USERNAME>/psvm-runs)
set -euo pipefail
WORK="${PSVM_WORK:-/kaggle/working}"   # overridable for testing off Kaggle
INPUT="${PSVM_INPUT:-/kaggle/input}"
cd "$(dirname "$0")/.."
source kaggle/env.sh >/dev/null

"$PY" scripts/summarize.py ${WORK}/runs --out ${WORK}/results > ${WORK}/results_summary.log 2>&1 \
  && echo "[save] summary: ${WORK}/results/summary.md" \
  || echo "[save] summary failed, see ${WORK}/results_summary.log"

STAMP=$(date +%Y%m%d_%H%M)
ZIP="${WORK}/psvm_runs_${STAMP}.zip"
(cd ${WORK} && { zip -qr "$ZIP" runs results 2>/dev/null || "$PY" -m zipfile -c "$ZIP" runs results; })
echo "[save] packaged $ZIP ($(du -h "$ZIP" | cut -f1))"

if [ "${1:-}" = "--upload" ]; then
  [ -n "${KAGGLE_USERNAME:-}" ] && { [ -n "${KAGGLE_API_TOKEN:-}" ] || [ -n "${KAGGLE_KEY:-}" ]; } || \
    { echo "[save] set KAGGLE_USERNAME and KAGGLE_API_TOKEN (or legacy KAGGLE_KEY) in ~/.psvm_keys.env"; exit 1; }
  [ -n "${KAGGLE_API_TOKEN:-}" ] || unset KAGGLE_API_TOKEN
  [ -n "${KAGGLE_KEY:-}" ] || unset KAGGLE_KEY        # empty values would confuse the CLI
  # recent CLI (understands the new KAGGLE_API_TOKEN), installed into the run environment
  uv pip install -q --python "$PY" -U kaggle
  KAGGLE_BIN="$(dirname "$PY")/kaggle"
  UP=/tmp/psvm_upload
  rm -rf "$UP" && mkdir -p "$UP"
  cp "$ZIP" "$UP/psvm_runs.zip"
  cat > "$UP/dataset-metadata.json" <<EOF
{"title": "psvm-runs", "id": "${KAGGLE_USERNAME}/psvm-runs", "licenses": [{"name": "CC0-1.0"}]}
EOF
  if "$KAGGLE_BIN" datasets version -p "$UP" -m "runs $STAMP" >/dev/null 2>&1; then
    echo "[save] updated private dataset ${KAGGLE_USERNAME}/psvm-runs"
  else
    "$KAGGLE_BIN" datasets create -p "$UP" && echo "[save] created private dataset ${KAGGLE_USERNAME}/psvm-runs"
  fi
fi
