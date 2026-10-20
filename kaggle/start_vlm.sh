#!/usr/bin/env bash
# Serve a Qwen vision-language model from Hugging Face for the image reward model's preference labels:
# free, no daily request limit. CoRe's code talks to it like Gemini.
# One server per listed GPU; a label round's questions are split between them. On a T4 x2 session the
# default uses both GPUs: GPU 1 is free, and GPU 0's training run pauses while labels are being made.
#   bash kaggle/start_vlm.sh                                   # Qwen3-VL-4B-Instruct (float16, ~9 GB per GPU)
#   PSVM_VLM_GPUS="1" bash kaggle/start_vlm.sh                 # second GPU only
#   PSVM_VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct PSVM_VLM_4BIT=1 bash kaggle/start_vlm.sh   # 8B, 4-bit, slower
# Logs: /tmp/vlm_server_<gpu>.log
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${PSVM_VLM_MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
GPUS="${PSVM_VLM_GPUS:-1 0}"
BASE_PORT="${PSVM_VLM_PORT:-8008}"
VENV="${PSVM_VLM_VENV:-/tmp/vlm-venv}"
export HF_HOME="${HF_HOME:-/tmp/hf}"

command -v uv >/dev/null || pip install -q uv
[ -x "$VENV/bin/python" ] || uv venv -q --python 3.11 "$VENV"
uv pip install -q --python "$VENV/bin/python" -r kaggle/vlm_requirements.txt

URLS=()
i=0
for GPU in $GPUS; do
  PORT=$((BASE_PORT + i)); i=$((i + 1))
  URLS+=("http://127.0.0.1:$PORT/v1")
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "[vlm] GPU $GPU: already running on port $PORT"; continue
  fi
  echo "[vlm] GPU $GPU: loading $MODEL (first time: downloads from Hugging Face, a few minutes)..."
  CUDA_VISIBLE_DEVICES="$GPU" nohup "$VENV/bin/python" kaggle/vlm_server.py --model "$MODEL" --port "$PORT" \
    ${PSVM_VLM_4BIT:+--load-in-4bit} > "/tmp/vlm_server_$GPU.log" 2>&1 &
  PID=$!
  for _ in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    if ! kill -0 "$PID" 2>/dev/null; then echo "[vlm] server on GPU $GPU failed:"; tail -n 20 "/tmp/vlm_server_$GPU.log"; exit 1; fi
    sleep 5
  done
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "[vlm] GPU $GPU not ready after 20 min"; exit 1; }
done

JOINED=$(IFS=,; echo "${URLS[*]}")
cat > /tmp/psvm_local_vlm.env <<ENVEOF
VLM_BASE_URL=$JOINED
VLM_MODEL=$MODEL
VLM_API_KEY=local
VLM_RPM=0
ENVEOF
echo "[vlm] ready: $MODEL on GPU(s) $GPUS (labels now come from it, not Gemini)"
