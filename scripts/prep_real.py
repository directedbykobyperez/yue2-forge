"""Cache Latents: Precompute and cache all training data to disk.
-> /workspace/real/prep/<name>/{mert.npy, lat.npy, codec.npy, prefix.npy, abc_sheet.txt}
-> /workspace/real/prep/_cache_manifest.json (tracks cot mode, prevents stale caches)

Env: SHEETSAGE_TASK=full|melody|off (default: full)
     TOKENIZER=community|mert (default: community)
     COT=off|full|melody (default: off, must match ar_prep.py)
"""
import os, glob, json, hashlib, numpy as np, torch, soundfile as sf
from scipy.signal import resample_poly
from math import gcd
os.environ.setdefault("HF_HOME","/workspace/hf")
from transformers import AutoModel, AutoFeatureExtractor
from yue2.modeling_vae import YuE2VAE
from yue2.protocol import SongRequest, token_prefixes
from yue2.tokenization_yue2 import YuE2TextTokenizer

SRC = "/workspace/real/artist"
OUT = "/workspace/real/prep"
os.makedirs(OUT, exist_ok=True)
dev = "cuda"

snap = lambda n: glob.glob(f"/workspace/hf/hub/models--m-a-p--{n}/snapshots/*")[0]
tok = YuE2TextTokenizer(snap("YuE2-3B") + "/qwen.tiktoken")

# ── Load models ──────────────────────────────────────────────────────
print("[Cache] Loading VAE...", flush=True)
vae = YuE2VAE.from_pretrained(snap("YuE2-Vae"), decoder_only=False, device=dev, local_files_only=True)

print("[Cache] Loading MERT for features...", flush=True)
proc = AutoFeatureExtractor.from_pretrained("m-a-p/MERT-v2-FullSong", trust_remote_code=True)
mert = AutoModel.from_pretrained("m-a-p/MERT-v2-FullSong", trust_remote_code=True).to(dev).eval()

TOKENIZER = os.environ.get("TOKENIZER", "community")
codec_tokenizer = None
if TOKENIZER == "community":
    try:
        print("[Cache] Loading community tokenizer (Mothersuperior)...", flush=True)
        from transformers import AutoModel as AutoModelCT
        codec_tokenizer = AutoModelCT.from_pretrained(
            "Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4",
            trust_remote_code=True
        ).to(dev).eval()
        print("[Cache] Community tokenizer loaded.", flush=True)
    except Exception as e:
        print(f"[Cache] WARNING: Community tokenizer failed: {e}, falling back to MERT", flush=True)
        TOKENIZER = "mert"

SHEETSAGE_TASK = os.environ.get("SHEETSAGE_TASK", "full")
COT = os.environ.get("COT", "off")
sheetsage_model = None
if SHEETSAGE_TASK != "off" and COT != "off":
    try:
        print(f"[Cache] Loading SheetSage2 for task={SHEETSAGE_TASK}...", flush=True)
        sheetsage_model = AutoModel.from_pretrained("m-a-p/SheetSage2", trust_remote_code=True, local_files_only=True).eval().to(dev)
        print("[Cache] SheetSage2 loaded.", flush=True)
    except Exception as e:
        print(f"[Cache] WARNING: SheetSage2 failed: {e}", flush=True)
        sheetsage_model = None

# ── Feature extraction ───────────────────────────────────────────────
def mert_features(mono24):
    """Extract MERT L20 features at 25Hz."""
    CH = 24000 * 30
    chunks = [mono24[s:s+CH] for s in range(0, len(mono24), CH)]
    chunks = [c for c in chunks if len(c) >= 24000]
    full = [c for c in chunks if len(c) == CH]
    tail = [c for c in chunks if len(c) < CH]
    feats = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for group in ([full] if full else []) + [[c] for c in tail]:
            inp = {k: v.to(dev) for k, v in proc(group, sampling_rate=24000, return_tensors="pt").items()}
            feats.append(mert(**inp, output_hidden_states=True).hidden_states[20].reshape(-1, 1024))
    H = torch.cat(feats, 0).float()
    T25 = int(round(len(mono24) / 24000 * 25))
    return torch.nn.functional.interpolate(H.T[None], size=T25, mode="linear", align_corners=False)[0].T.half().cpu().numpy()

def encode_audio(st48):
    """Encode audio to VAE latents [T, 64]."""
    out = []
    CH = 48000 * 60
    with torch.inference_mode():
        for s in range(0, len(st48), CH):
            seg = st48[s:s+CH]
            if len(seg) < 1920:
                break
            out.append(vae.encode(torch.tensor(seg.T[None]))[0].T.float().cpu())
    return torch.cat(out, 0).numpy()

def encode_codec(mono24):
    """Encode audio to codec tokens using community tokenizer or MERT."""
    if codec_tokenizer is not None:
        # Community tokenizer: direct audio-to-token
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            audio_tensor = torch.tensor(mono24).unsqueeze(0).to(dev)
            tokens = codec_tokenizer.encode(audio_tensor)
            return tokens.squeeze(0).cpu().numpy().astype(np.int32)
    else:
        # Fallback: MERT features -> quantize (approximation)
        feat = mert_features(mono24)
        # Simple quantization for now (real tokenizer is better)
        return (feat.mean(axis=-1) * 100).astype(np.int32)

def generate_abc_sheet(audio_path, task="full"):
    """Generate ABC sheet from audio using SheetSage2."""
    if sheetsage_model is None:
        return None
    try:
        melody_only = task != "full"
        prompts = ["timestamp", "downbeat_meter", "structure", "key"]
        if task == "full":
            prompts += ["chord_full", "melody_full"]
        else:
            prompts += ["melody_vocal" if task == "melody-vocal" else "melody_full"]
        result = sheetsage_model.transcribe(
            str(audio_path), prompts=prompts, dtype="bf16",
            preset="default", melody_only=melody_only
        )
        if result.get("abc") and not result.get("abc_error"):
            return result["abc"]
    except Exception as e:
        print(f"  [SheetSage2] Error: {e}", flush=True)
    return None

def parse_txt_file(path):
    """Parse .txt file with caption + lyrics format.
    
    Format:
        Caption/description of the song
        [Verse 1]
        First line of lyrics
        Second line of lyrics
        
        [Chorus]
        Chorus line 1
        
        [Bridge]
        Bridge line
    
    Or simple format (no section headers):
        Caption/description
        Lyrics line 1
        Lyrics line 2
    """
    if not os.path.exists(path):
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return None, None
    
    lines = content.split("\n")
    caption_lines = []
    lyric_lines = []
    in_lyrics = False
    
    for line in lines:
        stripped = line.strip()
        # Check for section headers like [Verse 1], [Chorus], [Bridge]
        if stripped.startswith("[") and stripped.endswith("]") and any(kw in stripped.lower() for kw in ["verse", "chorus", "bridge", "intro", "outro", "pre-chorus", "hook", "refrain"]):
            in_lyrics = True
            lyric_lines.append(stripped)
        elif in_lyrics:
            lyric_lines.append(stripped)
        elif not caption_lines:
            # First non-empty line is the caption
            if stripped:
                caption_lines.append(stripped)
        else:
            # Lines after caption but before section headers are still caption
            if stripped:
                caption_lines.append(stripped)
    
    caption = " ".join(caption_lines).strip()[:1500]
    lyrics = "\n".join(lyric_lines).strip() if lyric_lines else "[instrumental]"
    
    # If no section headers found, treat all lines after first as lyrics
    if not lyric_lines and len(lines) > 1:
        caption = lines[0].strip()[:1500]
        lyrics = "\n".join(lines[1:]).strip()
    
    return caption, lyrics

# ── Cache manifest ───────────────────────────────────────────────────
def load_manifest():
    path = f"{OUT}/_cache_manifest.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"cot": None, "tokenizer": None, "songs": []}

def save_manifest(manifest):
    path = f"{OUT}/_cache_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)

manifest = load_manifest()
needs_rebuild = (manifest.get("cot") != COT or manifest.get("tokenizer") != TOKENIZER)

if needs_rebuild:
    print(f"[Cache] Mode changed (cot: {manifest.get('cot')} -> {COT}, tokenizer: {manifest.get('tokenizer')} -> {TOKENIZER})", flush=True)
    print("[Cache] Clearing old cache...", flush=True)
    for d in glob.glob(f"{OUT}/*"):
        if os.path.isdir(d) and not os.path.basename(d).startswith("_"):
            import shutil
            shutil.rmtree(d)
    manifest = {"cot": COT, "tokenizer": TOKENIZER, "songs": []}

# ── Process songs ────────────────────────────────────────────────────
for f in sorted(glob.glob(f"{SRC}/*.flac")):
    name = os.path.basename(f)[:-5]
    d = f"{OUT}/{name}"
    
    # Skip if already cached
    if os.path.exists(f"{d}/prefix.npy") and name in manifest.get("songs", []):
        print(f"{name}: cached, skipping", flush=True)
        continue
    
    os.makedirs(d, exist_ok=True)
    
    # Load audio
    a, sr = sf.read(f, dtype="float32")
    a = np.stack([a, a], 1) if a.ndim == 1 else a
    
    # Resample to 48kHz (VAE) and 24kHz (MERT)
    g = gcd(sr, 48000)
    st48 = resample_poly(a, 48000//g, sr//g, axis=0).astype(np.float32) if sr != 48000 else a
    g = gcd(sr, 24000)
    m24 = resample_poly(a.mean(1), 24000//g, sr//g).astype(np.float32)
    
    # Extract features
    M = mert_features(m24)
    Z = encode_audio(st48)
    n = min(len(M), len(Z))
    
    # Save MERT features and VAE latents
    np.save(f"{d}/mert.npy", M[:n])
    np.save(f"{d}/lat.npy", Z[:n].astype(np.float32))
    
    # Encode codec tokens
    codec = encode_codec(m24)
    np.save(f"{d}/codec.npy", codec[:n].astype(np.int32))
    
    # Parse .txt file with new format
    txt_path = f"{SRC}/{name}.txt"
    cap, lyr = parse_txt_file(txt_path)
    if cap is None:
        # Fallback to old format
        cap = open(txt_path).read().split("===LYRICS===")[0].replace("Global Metadata:", "").strip()
        cap = " ".join(cap.split())[:1500]
        lyr_path = f"{SRC}/{name}.lyrics.txt"
        lyr = open(lyr_path).read().strip() if os.path.exists(lyr_path) else "[instrumental]"
    
    # Build prefix with ABC sheet if cot != off
    abc_sheet = None
    if COT != "off" and sheetsage_model is not None:
        abc_sheet = generate_abc_sheet(f, task=SHEETSAGE_TASK)
        if abc_sheet:
            with open(f"{d}/abc_sheet.txt", "w") as af:
                af.write(abc_sheet)
            print(f"  [SheetSage2] ABC sheet saved ({len(abc_sheet)} chars)", flush=True)
    
    # Build prefix
    if COT != "off" and abc_sheet:
        pre = token_prefixes(SongRequest(style=cap, lyrics=lyr, cot=COT, seed=1, id="real", abc=abc_sheet), tok)
    else:
        pre = token_prefixes(SongRequest(style=cap, lyrics=lyr, cot="off", seed=1, id="real"), tok)
    
    np.save(f"{d}/prefix.npy", np.array(pre, dtype=np.int64))
    
    # Update manifest
    if name not in manifest["songs"]:
        manifest["songs"].append(name)
    save_manifest(manifest)
    
    print(f"{name}: frames {n} ({n/25/60:.1f} min) prefix {len(pre)} codec {len(codec)}", flush=True)

print("CACHE LATENTS DONE", flush=True)
