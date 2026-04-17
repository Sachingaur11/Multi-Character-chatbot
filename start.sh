#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

source ../venv/bin/activate

cleanup() {
    echo ""
    echo "Shutting down..."
    kill "$UVICORN_PID" "$STREAMLIT_PID" 2>/dev/null
    wait "$UVICORN_PID" "$STREAMLIT_PID" 2>/dev/null
    exit 0
}

trap cleanup INT TERM

echo "Starting FastAPI backend..."
uvicorn main:app --reload &
UVICORN_PID=$!

echo "Starting Streamlit frontend..."
streamlit run app.py &
STREAMLIT_PID=$!

echo "Both servers running. Press Ctrl+C to stop."
wait
