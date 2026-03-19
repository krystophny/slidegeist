#!/usr/bin/env bash
set -euo pipefail

# Start a Voxtype daemon with OpenAI-compatible STT service.
# Requires voxtype from the feature/single-daemon-openai-stt-api branch:
#   https://github.com/Trivernis/voxtype
#
# Build voxtype:
#   git clone https://github.com/Trivernis/voxtype
#   cd voxtype
#   git checkout feature/single-daemon-openai-stt-api
#   cargo build --release
#   cp target/release/voxtype ~/.local/bin/
#
# Usage:
#   ./scripts/setup-voxtype-stt.sh
#   # Or with custom settings:
#   SLIDEGEIST_STT_PORT=8427 SLIDEGEIST_STT_MODEL=large-v3-turbo ./scripts/setup-voxtype-stt.sh

VOXTYPE_BIN="${VOXTYPE_BIN:-voxtype}"
HOST="${SLIDEGEIST_STT_HOST:-127.0.0.1}"
PORT="${SLIDEGEIST_STT_PORT:-8427}"
MODEL="${SLIDEGEIST_STT_MODEL:-large-v3-turbo}"
LANGUAGE="${SLIDEGEIST_STT_LANGUAGE:-auto}"
THREADS="${SLIDEGEIST_STT_THREADS:-4}"

if ! command -v "$VOXTYPE_BIN" >/dev/null 2>&1; then
  echo "voxtype binary not found: $VOXTYPE_BIN" >&2
  echo "" >&2
  echo "Install voxtype (requires Rust toolchain):" >&2
  echo "  git clone https://github.com/Trivernis/voxtype" >&2
  echo "  cd voxtype" >&2
  echo "  git checkout feature/single-daemon-openai-stt-api" >&2
  echo "  cargo build --release" >&2
  echo "  cp target/release/voxtype ~/.local/bin/" >&2
  exit 1
fi

if curl -fsS --max-time 2 "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "STT service already running at http://${HOST}:${PORT}"
  exit 0
fi

echo "Starting voxtype STT at http://$HOST:$PORT (model=$MODEL language=$LANGUAGE)"

export VOXTYPE_SERVICE_ENABLED=true
export VOXTYPE_SERVICE_HOST="$HOST"
export VOXTYPE_SERVICE_PORT="$PORT"
export VOXTYPE_LANGUAGE="$LANGUAGE"
export VOXTYPE_MODEL="$MODEL"
export VOXTYPE_THREADS="$THREADS"

exec "$VOXTYPE_BIN" \
  --service \
  --service-host "$HOST" \
  --service-port "$PORT" \
  --model "$MODEL" \
  --language "$LANGUAGE" \
  --threads "$THREADS" \
  daemon
