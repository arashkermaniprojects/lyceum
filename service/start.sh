#!/usr/bin/env bash
# Launch the SeVim HTTP shim. Binds to loopback only.
set -eu
cd "$(dirname "$0")/.."

VENV_PY="/home/ara/Documents/Programming/agentic_systems/Video Lecture Generator/local_models/venv/bin/python3"
LOG="/tmp/sevim_service.log"

nohup "$VENV_PY" -m uvicorn service.app:app \
  --host 127.0.0.1 --port 8003 --log-level info \
  > "$LOG" 2>&1 &

echo "launched PID $!"
echo "log: $LOG"
echo "probe: curl -s http://127.0.0.1:8003/health"
