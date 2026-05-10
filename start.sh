#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8501}"
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"

export API_BASE="${API_BASE:-http://${BACKEND_HOST}:${BACKEND_PORT}}"
export VECTORSTORE_PATH="${VECTORSTORE_PATH:-./data/vectorstore}"

mkdir -p "${VECTORSTORE_PATH}"

python -m uvicorn main:app --host "${BACKEND_HOST}" --port "${BACKEND_PORT}" &
BACKEND_PID=$!

cleanup() {
  kill "${BACKEND_PID}" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

streamlit run frontend/app.py \
  --server.port "${PORT}" \
  --server.address "0.0.0.0" \
  --server.headless "true"
