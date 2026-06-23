#!/usr/bin/env bash
# Start the local RAG API for MaxKB.
set -e

PROJECT="/root/autodl-tmp/projects/Labor Law Legal Advisor"
PYTHON="/root/miniconda3/envs/llm_course/bin/python"
HOST="${RAG_API_HOST:-0.0.0.0}"
PORT="${RAG_API_PORT:-8000}"

cd "$PROJECT"

exec "$PYTHON" -m uvicorn rag_api:app --host "$HOST" --port "$PORT"
