#!/usr/bin/env bash
# Start a plan in the background. It keeps running if VS Code disconnects, but stops when the
# Kaggle session ends, so the launcher only starts runs that can finish in the time left.
#
#   bash kaggle/run.sh smoke_phase2 --labels scripted   # free pipeline check (~15 min on a T4)
#   bash kaggle/run.sh phase2_peg                       # GPU session: one Phase 2 run at a time
#   bash kaggle/run.sh phase1                           # CPU session: two runs at a time
# Extra options go to scripts/launch.py, e.g. --est_hours 4, --only seed=0, --parallel 2.
set -euo pipefail
WORK="${PSVM_WORK:-/kaggle/working}"   # overridable for testing off Kaggle
INPUT="${PSVM_INPUT:-/kaggle/input}"
cd "$(dirname "$0")/.."
source kaggle/env.sh >/dev/null
[ -n "$PY" ] || { echo "run kaggle/start.sh first"; exit 1; }
PLAN="${1:?usage: bash kaggle/run.sh PLAN [launch.py options]}"; shift

# hours left in this Kaggle session (sessions end after 12 h; the container started with the session)
if [ -z "${SESSION_HOURS:-}" ]; then
  ELAPSED=$(ps -o etimes= -p 1 2>/dev/null | tr -d ' ' || echo 0)
  SESSION_HOURS=$(awk -v e="${ELAPSED:-0}" 'BEGIN { h = 11.7 - e / 3600; if (h < 0.5) h = 0.5; printf "%.2f", h }')
fi

GPUS=$("$PY" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
if [ "$GPUS" -gt 0 ]; then
  DEFAULTS=(--gpus 0 --parallel 1 --threads 4)
else
  DEFAULTS=(--parallel 2 --threads 2)
fi

mkdir -p ${WORK}/runs
LOG="${WORK}/runs/launch_${PLAN}_$(date +%m%d_%H%M).log"
nohup "$PY" scripts/launch.py --plan "$PLAN" --out ${WORK}/runs --session_hours "$SESSION_HOURS" \
  "${DEFAULTS[@]}" "$@" > "$LOG" 2>&1 &
echo "[run] $PLAN started in the background (pid $!), ~${SESSION_HOURS} h left in this session"
echo "      follow:   tail -f $LOG"
echo "      progress: \$PY scripts/status.py ${WORK}/runs"
