#!/usr/bin/env bash
# Serve a Qwen vision-language model from Hugging Face on the second GPU, for the image reward
# model's preference labels: free, no daily request limit. CoRe's code talks to it like Gemini.
#   bash kaggle/start_vlm.sh                                   # Qwen3-VL-4B-Instruct (float16, ~9 GB)
#   PSVM_VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct PSVM_VLM_4BIT=1 bash kaggle/start_vlm.sh   # 8B, 4-bit, slower
# Log: /tmp/vlm_server.log
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${PSVM_VLM_MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
PORT="${PSVM_VLM_PORT:-8008}"
GPU="${PSVM_VLM_GPU:-1}"
VENV="${PSVM_VLM_VENV:-/tmp/vlm-venv}"

if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "[vlm] already running on port $PORT"
else
  command -v uv >/dev/null || pip install -q uv
  [ -x "$VENV/bin/python" ] || uv venv -q --python 3.11 "$VENV"
  uv pip install -q --python "$VENV/bin/python" -r kaggle/vlm_requirements.txt
  echo "[vlm] downloading and loading $MODEL from Hugging Face on GPU $GPU (first time: a few minutes)..."
  HF_HOME="${HF_HOME:-/tmp/hf}" CUDA_VISIBLE_DEVICES="$GPU" nohup "$VENV/bin/python" kaggle/vlm_server.py \
    --model "$MODEL" --port "$PORT" ${PSVM_VLM_4BIT:+--load-in-4bit} > /tmp/vlm_server.log 2>&1 &
  PID=$!
  for _ in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    if ! kill -0 "$PID" 2>/dev/null; then echo "[vlm] server failed to start:"; tail -n 20 /tmp/vlm_server.log; exit 1; fi
    sleep 5
  done
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "[vlm] not ready after 20 min, see /tmp/vlm_server.log"; exit 1; }
fi
cat > /tmp/psvm_local_vlm.env <<ENVEOF
VLM_BASE_URL=http://127.0.0.1:$PORT/v1
VLM_MODEL=$MODEL
VLM_API_KEY=local
VLM_RPM=0
ENVEOF
echo "[vlm] ready: $MODEL at http://127.0.0.1:$PORT/v1 (labels now come from it, not Gemini)"
