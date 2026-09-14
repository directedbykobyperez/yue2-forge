#!/bin/bash
# yue2-forge installer — run ONCE on a fresh GPU server (tested on NVIDIA L4, 22GB VRAM).
# Usage: git clone <this-repo> /workspace/yue2-forge && cd /workspace/yue2-forge && bash install.sh
set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
export HF_HOME="${HF_HOME:-/workspace/hf}"
export DEBIAN_FRONTEND=noninteractive
# Optional: export HF_TOKEN=... first (needed only for gated/private downloads).

echo "==> [1/6] system deps (ffmpeg, git-lfs)"
apt-get update -qq && apt-get install -y -qq ffmpeg git-lfs >/dev/null
git lfs install >/dev/null 2>&1 || true

echo "==> [2/6] yue2-infer @ pinned commit"
if [ ! -d /workspace/yue2-infer ]; then
  git clone https://github.com/multimodal-art-projection/YuE.git /workspace/yue2-infer
fi
cd /workspace/yue2-infer && git checkout 92a73cc7 2>/dev/null || true
cd "$REPO_DIR"

echo "==> [3/6] venv + torch + deps (takes a while)"
python3 -m venv /workspace/yue2venv
/workspace/yue2venv/bin/pip install --upgrade pip -q
/workspace/yue2venv/bin/pip install "torch==2.10.0" --index-url https://download.pytorch.org/whl/cu128
/workspace/yue2venv/bin/pip install -e /workspace/yue2-infer -r "$REPO_DIR/requirements.txt"
/workspace/yue2venv/bin/python -c "import torch, yue2; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"

echo "==> [4/6] tokenizer weights (Mothersuperior v4)"
mkdir -p /workspace/tok
huggingface-cli download Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4 \
  tokenizer_head_joint_v4.pt nar_lora_joint_v4.pt --local-dir /workspace/tok

echo "==> [5/6] base models (YuE2-3B, Vae, MERT-v2-FullSong)"
huggingface-cli download m-a-p/YuE2-3B --local-dir-use-symlinks False 2>&1 | tail -1
huggingface-cli download m-a-p/YuE2-Vae --local-dir-use-symlinks False 2>&1 | tail -1
huggingface-cli download m-a-p/MERT-v2-FullSong --local-dir-use-symlinks False 2>&1 | tail -1

echo "==> [6/6] minted regularizer pack (~100MB, 4,732 songs)"
mkdir -p /workspace/real
huggingface-cli download Mothersuperior/yue2-minted-corpus regularizer/minted_regularizer_pack.pt \
  --repo-type dataset --local-dir /workspace/real

echo ""
echo "INSTALL DONE. Next:"
echo "  1. Read docs/DATASET.md and place songs in /workspace/real/artist/"
echo "  2. export RUN_NAME=my_lora TRIGGER=mytrigger"
echo "  3. bash scripts/run_all.sh  (prep -> cursor -> ar_prep -> train -> sample)"
