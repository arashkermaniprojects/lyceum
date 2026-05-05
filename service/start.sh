#!/usr/bin/env bash
# Launch the SeVim HTTP shim. Binds to loopback only.
#
# The Python interpreter defaults to the repo-local virtualenv;
# override via $LYCEUM_PY.  Override the log path via $LYCEUM_LOG.
set -eu
cd "$(dirname "$0")/.."

VENV_PY="${LYCEUM_PY:-$(pwd)/.venv/bin/python3}"
LOG="${LYCEUM_LOG:-/tmp/sevim_service.log}"

nohup "$VENV_PY" -m uvicorn service.app:app \
  --host 127.0.0.1 --port 8003 --log-level info \
  > "$LOG" 2>&1 &

echo "launched PID $!"
echo "log: $LOG"
echo "probe: curl -s http://127.0.0.1:8003/health"
