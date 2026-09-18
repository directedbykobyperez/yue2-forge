"""Shared VRAM helpers for yue2-forge training scripts.

Env: VRAM_MODE = low | high  (default: auto-detect based on available VRAM)
     INT8_MODEL_PATH — path to INT8 safetensors (default: /workspace/comfyui/yue2_3b_int8_convrot.safetensors)
     LOW_VRAM_LEN — MAXLEN for low-VRAM mode (default: 8192)
     HIGH_VRAM_LEN — MAXLEN for high-VRAM mode (default: 12288)
"""
import os, glob, torch

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
    path = os.environ.get("INT8_MODEL_PATH", "/workspace/comfyui/yue2_3b_int8_convrot.safetensors")
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
    snaps = glob.glob("/workspace/hf/hub/models--m-a-p--YuE2-3B/snapshots/*")
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
    snaps = glob.glob("/workspace/hf/hub/models--m-a-p--YuE2-3B/snapshots/*")
    if not snaps:
        raise FileNotFoundError("No YuE2-3B snapshot found.")
    return snaps[0]
