#!/usr/bin/env bash
# Plain progress report:  bash kaggle/check.sh
cd "$(dirname "$0")/.."
source kaggle/env.sh >/dev/null
WORK="${PSVM_WORK:-/kaggle/working}"

if pgrep -f train_kaggle.py >/dev/null; then echo "A run is training right now."; else echo "Nothing is training right now."; fi
echo
"$PY" scripts/status.py "$WORK/runs" 2>/dev/null
echo
# for runs still in progress: the latest label update (labels should grow by ~20 each time)
for ev in "$WORK"/runs/*/*/seed*/events.jsonl; do
  [ -f "$ev" ] || continue
  d="$(dirname "$ev")"
  [ -f "$d/done.json" ] && continue
  last="$(grep rrm_update "$ev" | tail -1)"
  [ -n "$last" ] && echo "$(basename "$(dirname "$d")") $(basename "$d"): $last"
done
# overall agreement of the VLM's labels with the true preference, over all label rounds so far
for ev in "$WORK"/runs/*/*/seed*/events.jsonl; do
  [ -f "$ev" ] || continue
  "$PY" - "$ev" <<'PYEOF'
import json, sys, os
rows = [json.loads(l) for l in open(sys.argv[1]) if '"rrm_update"' in l]
n = c = 0
for r in rows:
    k = r['labeled'] - n
    c += r['vlm_acc'] * k
    n = r['labeled']
if n:
    run = os.path.relpath(os.path.dirname(sys.argv[1]), os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(sys.argv[1])))))
    print(f"{run}: Gemini agreement so far {100 * c / n:.0f}% over {n} labels ({len(rows)} rounds)")
PYEOF
done
