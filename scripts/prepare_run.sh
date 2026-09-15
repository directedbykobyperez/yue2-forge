#!/bin/bash
# yue2-forge dataset prep for UI runs: prep_real -> cursor_prep -> ar_prep, staged.
# Writes progress to /workspace/prep_status.json for the dashboard.
# Usage: bash scripts/prepare_run.sh   (refuses while training runs)
set -u
export HF_HOME="${HF_HOME:-/workspace/hf}"
export REG_PACK="${REG_PACK:-/workspace/real/regularizer/minted_regularizer_pack.pt}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY=/workspace/yue2venv/bin/python
STATUS=/workspace/prep_status.json
LOG=/workspace/prep.log

stage() { printf '{"stage":"%s","detail":"%s","done":false,"error":""}' "$1" "$2" > "$STATUS"; echo "[prep] $1: $2" | tee -a "$LOG"; }
finish() { printf '{"stage":"done","detail":"dataset ready — start training","done":true,"error":""}' > "$STATUS"; echo "[prep] DONE" | tee -a "$LOG"; }
fail() { printf '{"stage":"error","detail":%s,"done":false,"error":%s}' "$(printf '%s' "$1" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')" "$(printf '%s' "$1" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')" > "$STATUS"; echo "[prep] FAILED: $1" | tee -a "$LOG"; exit 1; }

if pgrep -f "ar_train|ar_lora_" >/dev/null; then echo "training running — prep refused"; exit 1; fi

: > "$LOG"
stage "prep" "MERT features + VAE latents + prefixes"
$PY "$REPO_DIR/scripts/prep_real.py" >>"$LOG" 2>&1 || fail "prep_real failed (see prep.log)"
stage "cursor" "vocal stems + lyric alignment (slowest stage)"
$PY "$REPO_DIR/scripts/cursor_prep.py" >>"$LOG" 2>&1 || fail "cursor_prep failed (see prep.log)"
stage "ar_prep" "tokenizing + regularizer mix"
$PY "$REPO_DIR/scripts/ar_prep.py" /workspace/tok/tokenizer_head_joint_v4.pt >>"$LOG" 2>&1 || fail "ar_prep failed (see prep.log)"
finish
