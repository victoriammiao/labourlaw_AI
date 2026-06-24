#!/usr/bin/env bash
# Start the local RAG API for MaxKB.
set -e

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PYTHON="/root/miniconda3/envs/llm_course/bin/python"
PYTHON="${RAG_API_PYTHON:-$DEFAULT_PYTHON}"
HOST="${RAG_API_HOST:-0.0.0.0}"
PORT="${RAG_API_PORT:-8000}"

cd "$PROJECT"

if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python)"
  echo "Configured Python not found; using $PYTHON instead."
fi

exec "$PYTHON" -m uvicorn rag_api:app --host "$HOST" --port "$PORT"
