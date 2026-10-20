#!/usr/bin/env bash
# One command per Kaggle session: sets everything up if needed, then starts the next experiment run.
#   bash kaggle/go.sh
# GPU session -> the next Phase 2 run (about 7-8 h).  CPU session -> Phase 1 runs.
# Check on it with:  bash kaggle/check.sh     Save before stopping the session:  bash kaggle/save_results.sh
set -euo pipefail
cd "$(dirname "$0")/.."

[ -x "$(cat "${PSVM_PY_FILE:-/tmp/psvm_python}" 2>/dev/null)" ] || bash kaggle/start.sh
source kaggle/env.sh >/dev/null
if [ -z "${VLM_API_KEY:-}" ]; then
  echo "Your API keys are missing. Run:  code ~/.psvm_keys.env   paste your keys, save, then run this again."
  exit 1
fi

if "$PY" -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  bash kaggle/run.sh phase2_peg --max_runs "${RUNS:-1}" --est_hours 8 >/dev/null
  echo "Started the next Phase 2 run (Peg Insert). It takes about 7-8 hours and keeps going if VS Code disconnects."
else
  bash kaggle/run.sh phase1 >/dev/null
  echo "Started Phase 1 runs (no GPU needed)."
fi
echo "Check on it:                       bash kaggle/check.sh"
echo "Before you stop the Kaggle session: bash kaggle/save_results.sh"
