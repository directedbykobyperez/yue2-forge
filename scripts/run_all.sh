#!/bin/bash
# yue2-forge full run: prep -> cursor align -> ar dataset -> train.
# Assumes install.sh done + dataset in /workspace/real/artist/ (see docs/DATASET.md).
# Env: RUN_NAME (default my_lora), VRAM_MODE (low|high, auto-detect if unset).
# Training runs in background; logs to /workspace/ar_train.log.
set -e
export RUN_NAME="${RUN_NAME:-my_lora}"
export HF_HOME="${HF_HOME:-/workspace/hf}"
export VRAM_MODE="${VRAM_MODE:-}"
export SCHED_STEPS=3000 CK_FROM=600 CK_EVERY=200
# Training hyperparams (override via env)
export AR_KL_WEIGHT="${AR_KL_WEIGHT:-0.04}"
export AR_LR_MULTIPLIER="${AR_LR_MULTIPLIER:-1.0}"
export ABC_DROPOUT="${ABC_DROPOUT:-0.5}"
export TRAIN_WINDOW="${TRAIN_WINDOW:-1500}"
export COT="${COT:-off}"
export AR_MAX_TOKENS="${AR_MAX_TOKENS:-0}"
export SHEETSAGE_TASK="${SHEETSAGE_TASK:-full}"
export TOKENIZER="${TOKENIZER:-community}"
export REG_PACK="${REG_PACK:-/workspace/real/regularizer/minted_regularizer_pack.pt}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY=/workspace/yue2venv/bin/python

echo "==> [1/4] prep_real (MERT + VAE + Codec + ABC sheets)"
$PY "$REPO_DIR/scripts/prep_real.py"
echo "==> [2/4] cursor_prep (demucs stems + lyric alignment)"
$PY "$REPO_DIR/scripts/cursor_prep.py"
echo "==> [3/4] ar_prep (tokenize + regularizer mix)"
$PY "$REPO_DIR/scripts/ar_prep.py" /workspace/tok/tokenizer_head_joint_v4.pt
echo "==> [4/4] train: $RUN_NAME (1600 steps, rank 64, background)"
cd "$REPO_DIR/scripts"
nohup $PY ar_train.py "$RUN_NAME" 1600 64 0.5 none 1e-4 0.08 >>/workspace/ar_train.log 2>&1 &
echo "training pid $!. watch: tail -f /workspace/ar_train.log"
echo "dashboard: FORGE_OUT=/workspace/tok/full/$RUN_NAME $PY $REPO_DIR/ui/server.py  (port 8000)"
echo "auto-sampler: RUN_NAME=$RUN_NAME bash $REPO_DIR/scripts/watch_run.sh"
