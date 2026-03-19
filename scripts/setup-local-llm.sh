#!/usr/bin/env bash
set -euo pipefail

# Start a llama.cpp server with Qwen3.5-9B for AI slide descriptions.
# Requires llama.cpp: https://github.com/ggml-org/llama.cpp
#
# Install llama.cpp:
#   # Linux (build from source):
#   git clone https://github.com/ggml-org/llama.cpp
#   cd llama.cpp && cmake -B build -G Ninja && cmake --build build -j$(nproc)
#   cp build/bin/llama-server ~/.local/bin/
#
#   # macOS:
#   brew install llama.cpp
#
# Usage:
#   ./scripts/setup-local-llm.sh
#   # Or with custom settings:
#   SLIDEGEIST_LLM_PORT=8081 ./scripts/setup-local-llm.sh

HOST="${SLIDEGEIST_LLM_HOST:-127.0.0.1}"
PORT="${SLIDEGEIST_LLM_PORT:-8081}"
MODEL_DIR="${SLIDEGEIST_LLM_MODEL_DIR:-$HOME/.local/share/slidegeist-llm/models}"
MODEL_FILE="${SLIDEGEIST_LLM_MODEL_FILE:-Qwen3.5-9B-Q4_K_M.gguf}"
MODEL_URL="${SLIDEGEIST_LLM_MODEL_URL:-https://huggingface.co/lmstudio-community/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf?download=true}"
THREADS="${SLIDEGEIST_LLM_THREADS:-4}"
CTX_SIZE="${SLIDEGEIST_LLM_CTX:-32768}"
NGL="${SLIDEGEIST_LLM_NGL:-99}"

# Find llama-server binary
SERVER_BIN=""
for candidate in \
  "${LLAMA_SERVER_BIN:-}" \
  "$(command -v llama-server 2>/dev/null || true)" \
  "$HOME/.local/bin/llama-server" \
  "$HOME/.local/llama.cpp/llama-server"; do
  if [ -n "$candidate" ] && [ -x "$candidate" ]; then
    SERVER_BIN="$candidate"
    break
  fi
done

if [ -z "$SERVER_BIN" ]; then
  echo "llama-server binary not found." >&2
  echo "" >&2
  echo "Install llama.cpp:" >&2
  echo "  # From source:" >&2
  echo "  git clone https://github.com/ggml-org/llama.cpp" >&2
  echo "  cd llama.cpp && cmake -B build -G Ninja && cmake --build build -j\$(nproc)" >&2
  echo "  cp build/bin/llama-server ~/.local/bin/" >&2
  echo "" >&2
  echo "  # macOS:" >&2
  echo "  brew install llama.cpp" >&2
  exit 1
fi

if curl -fsS --max-time 2 "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
  echo "llama-server already running at http://${HOST}:${PORT}"
  exit 0
fi

mkdir -p "$MODEL_DIR"
MODEL_PATH="$MODEL_DIR/$MODEL_FILE"
if [ ! -s "$MODEL_PATH" ]; then
  echo "Downloading $MODEL_FILE to $MODEL_PATH (~5.3 GB)"
  curl -fL --retry 3 --retry-delay 2 -o "$MODEL_PATH.tmp" "$MODEL_URL"
  mv "$MODEL_PATH.tmp" "$MODEL_PATH"
fi

echo "Starting llama.cpp server at http://$HOST:$PORT (model=$MODEL_FILE)"

exec "$SERVER_BIN" \
  -m "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  -c "$CTX_SIZE" \
  --threads "$THREADS" \
  -ngl "$NGL" \
  --parallel 1 \
  --alias "qwen3.5-9b" \
  --no-webui
