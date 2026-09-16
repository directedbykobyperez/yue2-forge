#!/bin/bash
# Checkpoint watcher: when a new step-N.pt lands, pause training, render a
# fixed-prompt sample with it, then resume training from that checkpoint.
OUT=/workspace/tok/full/${RUN_NAME:-my_lora}
GEN=${GEN_DIR:-/workspace/tok/full/gen}
NAR=${NAR_CK:-/workspace/tok/nar_lora_joint_v4.pt}
STYLE=${SAMPLE_STYLE_TRACK:-sample}
LYR=${SAMPLE_LYRICS:-/workspace/sample_lyrics.txt}
export HF_HOME=/workspace/hf SCHED_STEPS=3000 CK_FROM=600 CK_EVERY=200 TOTAL_STEPS=${TOTAL_STEPS:-1600}
CKPTS=${CKPTS:-"600 800 1000 1200 1400 1600"}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

train_pid() { pgrep -f "ar_lora_" | head -1; }
have_sample() { ls $GEN/${RUN_NAME:-my_lora}_s$1.mp3 $GEN/${RUN_NAME:-my_lora}_s$1.flac >/dev/null 2>&1; }

sample_step() {
  local t=$1; export CKSTEP=$t
  echo "{\"phase\": \"sampling\", \"step\": $t}" > /workspace/ui/state.json
  echo "[watcher] sampling step-$t" >> /workspace/watcher.log
  if [ "$t" -lt 1600 ]; then
    for i in $(seq 1 90); do
      if grep -qE "step $((t+1)) loss|EVAL step $t " /workspace/ar_train.log 2>/dev/null; then break; fi
      [ -z "$(train_pid)" ] && break
      sleep 10
    done
    P=$(train_pid); [ -n "$P" ] && kill "$P"
    for i in $(seq 1 12); do
      [ -z "$(train_pid)" ] && break
      sleep 5
    done
    for i in $(seq 1 12); do
      FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
      [ "${FREE:-0}" -gt 18000 ] && break
      sleep 5
    done
    echo "[watcher] gpu ready, free VRAM: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) MiB" >> /workspace/watcher.log
  else
    for i in $(seq 1 120); do
      grep -q "RESULT ${RUN_NAME:-my_lora}" /workspace/ar_train.log 2>/dev/null && break
      [ -z "$(train_pid)" ] && sleep 10 && break
      sleep 10
    done
  fi
  SEED=12
  if [ -f /workspace/sample_cfg.json ]; then
    SEED=$(python3 -c "import json; c=json.load(open('/workspace/sample_cfg.json')); b=int(c.get('seed',12)); print(b + int(__import__('os').environ.get('CKSTEP','0'))//200 if c.get('walk') else b)" 2>/dev/null || echo 12)
  fi
  echo "[watcher] sampling with seed $SEED" >> /workspace/watcher.log
  /workspace/yue2venv/bin/python -u "$SCRIPT_DIR/ar_generate.py" \
    $OUT/step-$t.pt $NAR ${RUN_NAME:-my_lora}_s$t $STYLE $LYR "$SEED" >> /workspace/gen.log 2>&1
  if [ -f $GEN/${RUN_NAME:-my_lora}_s$t.flac ]; then
    ffmpeg -y -v error -i $GEN/${RUN_NAME:-my_lora}_s$t.flac -codec:a libmp3lame -qscale:a 4 \
      $GEN/${RUN_NAME:-my_lora}_s$t.mp3 2>>/workspace/gen.log || true
  fi
  # automatic: every sampled checkpoint also lands as .safetensors (user-facing format)
  /workspace/yue2venv/bin/python "$SCRIPT_DIR/export_safetensors.py" $OUT/step-$t.pt >> /workspace/gen.log 2>&1 || true
  echo "[watcher] sample step-$t done" >> /workspace/watcher.log
}

echo "[watcher] started" > /workspace/watcher.log
echo "{\"phase\": \"training\"}" > /workspace/ui/state.json
while true; do
  latest=""
  for t in $CKPTS; do
    if [ -f $OUT/step-$t.pt ] && ! have_sample $t; then latest=$t; fi
  done
  if [ -n "$latest" ]; then
    sample_step "$latest"
    if [ "$latest" -eq "$TOTAL_STEPS" ]; then
      /workspace/yue2venv/bin/python "$SCRIPT_DIR/export_safetensors.py" --all $OUT >> /workspace/gen.log 2>&1 || true
      echo "{\"phase\": \"done\"}" > /workspace/ui/state.json
      echo "[watcher] ALL DONE" >> /workspace/watcher.log
      break
    fi
    echo "{\"phase\": \"training\"}" > /workspace/ui/state.json
    START_STEP=$latest nohup /workspace/yue2venv/bin/python -u ar_lora_cont.py \
      ${RUN_NAME:-my_lora} $((TOTAL_STEPS-latest)) 64 0.5 $OUT/step-$latest.pt 1e-4 0.08 \
      >> /workspace/ar_train.log 2>&1 &
    echo "[watcher] resumed from $latest pid $!" >> /workspace/watcher.log
  fi
  sleep 30
done
