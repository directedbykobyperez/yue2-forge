#!/bin/bash
# Launch the yue2-forge dashboard on http://localhost:8000
# Env knobs: RUN_NAME (default my_lora), FORGE_TITLE, FORGE_TOTAL
cd "$(dirname "$0")"
export RUN_NAME="${RUN_NAME:-my_lora}"
PY=/workspace/yue2venv/bin/python
if [ ! -f "$PY" ]; then
  PY=python3
fi
nohup "$PY" ui/server.py >/tmp/forge_ui.log 2>&1 &
echo "dashboard live: http://localhost:8000  (pid $!, log /tmp/forge_ui.log)"
