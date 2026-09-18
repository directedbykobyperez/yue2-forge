"""Shared VRAM helpers for yue2-forge training scripts.

Env: VRAM_MODE = low | high  (default: auto-detect based on available VRAM)
     INT8_MODEL_PATH — path to INT8 safetensors (default: /workspace/comfyui/yue2_3b_int8_convrot.safetensors)
     LOW_VRAM_LEN — MAXLEN for low-VRAM mode (default: 8192)
     HIGH_VRAM_LEN — MAXLEN for high-VRAM mode (default: 12288)
     SHEETSAGE_MODEL_PATH — path to SheetSage2 model (default: /workspace/comfyui/sheetsage2_bf16.safetensors)

Model options:
  VRAM_MODE=low  → INT8 model (3.7GB VRAM, fits 16GB GPUs)
  VRAM_MODE=high → bf16 model (full precision, needs 22GB+ VRAM)
"""
import os, glob, torch

# Model paths
INT8_MODEL_DEFAULT = "/workspace/comfyui/yue2_3b_int8_convrot.safetensors"
BF16_SNAP_PATTERN = "/workspace/hf/hub/models--m-a-p--YuE2-3B/snapshots/*"
SHEETSAGE_DEFAULT = "/workspace/comfyui/sheetsage2_bf16.safetensors"

def detect_vram_gb():
    """Return total GPU VRAM in GB, or 0 if no GPU."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.get_device_properties(0).total_mem / (1024**3)

def get_vram_mode():
    """Resolve VRAM_MODE: explicit env var > auto-detect by VRAM."""
    mode = os.environ.get("VRAM_MODE", "").strip().lower()
    if mode in ("low", "high"):
        return mode
    vram = detect_vram_gb()
    if vram > 0 and vram < 20:
        return "low"
    return "high"

def get_model_info():
    """Return dict with model paths and availability status."""
    int8_path = os.environ.get("INT8_MODEL_PATH", INT8_MODEL_DEFAULT)
    snaps = glob.glob(BF16_SNAP_PATTERN)
    bf16_path = snaps[0] if snaps else None
    sheetsage_path = os.environ.get("SHEETSAGE_MODEL_PATH", SHEETSAGE_DEFAULT)
    return {
        "vram_mode": get_vram_mode(),
        "vram_gb": round(detect_vram_gb(), 1),
        "int8_available": os.path.exists(int8_path),
        "int8_path": int8_path,
        "bf16_available": bf16_path is not None,
        "bf16_path": bf16_path,
        "sheetsage_available": os.path.exists(sheetsage_path),
        "sheetsage_path": sheetsage_path,
        "maxlen": get_maxlen(),
    }

def get_maxlen():
    mode = get_vram_mode()
    if mode == "low":
        return int(os.environ.get("LOW_VRAM_LEN", "8192"))
    return int(os.environ.get("HIGH_VRAM_LEN", "12288"))

def load_yue2_model(device="cuda"):
    """Load YuE2-3B model. INT8 single-file in low mode, HF bf16 snapshot in high mode."""
    mode = get_vram_mode()
    if mode == "low":
        return _load_int8(device)
    return _load_bf16(device)

def _load_int8(device):
    """Load from single-file INT8 safetensors."""
    from yue2.modeling_yue2 import YuE2ForCausalLM
    path = os.environ.get("INT8_MODEL_PATH", INT8_MODEL_DEFAULT)
    if not os.path.exists(path):
        raise FileNotFoundError(f"INT8 model not found: {path}. Download it or set INT8_MODEL_PATH.")
    print(f"[VRAM low] Loading INT8 model from {path}", flush=True)
    model = YuE2ForCausalLM.from_pretrained(path, local_files_only=True, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.eval().to(device)
    model.requires_grad_(False)
    print(f"[VRAM low] INT8 model loaded. VRAM: {torch.cuda.memory_allocated()/2**30:.1f}G", flush=True)
    return model

def _load_bf16(device):
    """Load from HF snapshot (bf16)."""
    from yue2.modeling_yue2 import YuE2ForCausalLM
    snaps = glob.glob(BF16_SNAP_PATTERN)
    if not snaps:
        raise FileNotFoundError("No YuE2-3B snapshot found in /workspace/hf/hub/. Run install.sh or download the model.")
    snap = snaps[0]
    print(f"[VRAM high] Loading bf16 model from {snap}", flush=True)
    model = YuE2ForCausalLM.from_pretrained(snap, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval().to(device)
    model.requires_grad_(False)
    print(f"[VRAM high] bf16 model loaded. VRAM: {torch.cuda.memory_allocated()/2**30:.1f}G", flush=True)
    return model, snap

def get_snap_path():
    """Return tokenizer snapshot path (needed for tokenizer loading)."""
    snaps = glob.glob(BF16_SNAP_PATTERN)
    if not snaps:
        raise FileNotFoundError("No YuE2-3B snapshot found.")
    return snaps[0]

def load_sheetsage2(device="cuda"):
    """Load SheetSage2 model for ABC sheet generation."""
    path = os.environ.get("SHEETSAGE_MODEL_PATH", SHEETSAGE_DEFAULT)
    if not os.path.exists(path):
        raise FileNotFoundError(f"SheetSage2 model not found: {path}. Download it first.")
    print(f"[SheetSage2] Loading from {path}", flush=True)
    # SheetSage2 is loaded via the yue2 tokenizer module
    from yue2.tokenization_yue2 import SheetSage2Transcriber
    transcriber = SheetSage2Transcriber(path)
    transcriber.to(device)
    print(f"[SheetSage2] Loaded. VRAM: {torch.cuda.memory_allocated()/2**30:.1f}G", flush=True)
    return transcriber
