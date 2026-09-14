#!/bin/bash
# yue2-forge full run: prep -> cursor align -> ar dataset -> train.
# Assumes install.sh done + dataset in /workspace/real/artist/ (see docs/DATASET.md).
# Env: RUN_NAME (default my_lora). Training runs in background; logs to /workspace/ar_train.log.
set -e
export RUN_NAME="${RUN_NAME:-my_lora}"
export HF_HOME="${HF_HOME:-/workspace/hf}"
export SCHED_STEPS=3000 CK_FROM=600 CK_EVERY=200
export REG_PACK="${REG_PACK:-/workspace/real/regularizer/minted_regularizer_pack.pt}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY=/workspace/yue2venv/bin/python

echo "==> [1/4] prep_real (MERT + VAE + prefixes)"
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
