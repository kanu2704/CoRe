# Load API keys and the run Python into the current terminal:
#   source kaggle/env.sh
# Keys come from ~/.psvm_keys.env (see kaggle/keys.example.env); if the VLM key is missing there,
# Kaggle Secrets attached to the tunnel notebook are tried (GEMINI_API_KEY, OPENAI_API_KEY).

KEYS="${PSVM_KEYS:-$HOME/.psvm_keys.env}"
if [ -f "$KEYS" ]; then
  set -a; . "$KEYS"; set +a
else
  echo "[env] no $KEYS yet: cp kaggle/keys.example.env $KEYS and fill in your keys"
fi

if [ -z "${VLM_API_KEY:-}" ] && command -v python3 >/dev/null; then
  _k=$(python3 -c "from kaggle_secrets import UserSecretsClient as U; print(U().get_secret('GEMINI_API_KEY'))" 2>/dev/null || true)
  if [ -n "$_k" ]; then export VLM_API_KEY="$_k"; export LLM_API_KEY="${LLM_API_KEY:-$_k}"; echo "[env] VLM key loaded from Kaggle Secrets"; fi
  unset _k
fi

# defaults (free Gemini key for both roles)
export VLM_BASE_URL="${VLM_BASE_URL:-https://generativelanguage.googleapis.com/v1beta/openai/}"
export VLM_MODEL="${VLM_MODEL:-gemini-3.5-flash-lite}"
export LLM_BASE_URL="${LLM_BASE_URL:-$VLM_BASE_URL}"
export LLM_MODEL="${LLM_MODEL:-gemini-3.8-flash}"
export LLM_API_KEY="${LLM_API_KEY:-${VLM_API_KEY:-}}"

PY="$(cat "${PSVM_PY_FILE:-/tmp/psvm_python}" 2>/dev/null || true)"
if [ -z "$PY" ]; then echo "[env] run kaggle/start.sh first (it builds the Python environment)"; fi
export PY
if [ -n "${VLM_API_KEY:-}" ]; then _ks=set; else _ks=MISSING; fi
echo "[env] VLM ${VLM_MODEL} (key ${_ks}) | LLM ${LLM_MODEL} | Python ${PY:-none}"
unset _ks
