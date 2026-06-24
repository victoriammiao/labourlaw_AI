#!/usr/bin/env bash
# Start the local RAG API for MaxKB.
set -e

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${RAG_API_PYTHON:-$(command -v python)}"
HOST="${RAG_API_HOST:-0.0.0.0}"
PORT="${RAG_API_PORT:-8000}"

cd "$PROJECT"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python interpreter not found: $PYTHON" >&2
  exit 1
fi

exec "$PYTHON" -m uvicorn rag_api:app --host "$HOST" --port "$PORT"
