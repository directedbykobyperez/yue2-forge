"""Prep real tracks for the NAR-teacher loss: MERT-v2-FullSong L20 @25Hz (instance-normed later), VAE latents [T,64], cot=off prefix ids.
-> /workspace/real/prep/<name>/{mert.npy,lat.npy,prefix.npy}. Held-out track kept too (used only for eval)."""
import os, glob, re, json, numpy as np, torch, soundfile as sf
from scipy.signal import resample_poly
from math import gcd
os.environ.setdefault("HF_HOME","/workspace/hf")
from transformers import AutoModel, AutoFeatureExtractor
from yue2.modeling_vae import YuE2VAE
from yue2.protocol import SongRequest, token_prefixes
from yue2.tokenization_yue2 import YuE2TextTokenizer
SRC="/workspace/real/artist"; OUT="/workspace/real/prep"; os.makedirs(OUT,exist_ok=True); dev="cuda"
snap=lambda n: glob.glob(f"/workspace/hf/hub/models--m-a-p--{n}/snapshots/*")[0]; tok=YuE2TextTokenizer(snap("YuE2-3B")+"/qwen.tiktoken")
proc=AutoFeatureExtractor.from_pretrained("m-a-p/MERT-v2-FullSong", trust_remote_code=True); mert=AutoModel.from_pretrained("m-a-p/MERT-v2-FullSong", trust_remote_code=True).to(dev).eval()
vae=YuE2VAE.from_pretrained(snap("YuE2-Vae"), decoder_only=False, device=dev, local_files_only=True)
def mert_l20(mono24):
    CH=24000*30; chunks=[mono24[s:s+CH] for s in range(0,len(mono24),CH)]; chunks=[c for c in chunks if len(c)>=24000]; full=[c for c in chunks if len(c)==CH]; tail=[c for c in chunks if len(c)<CH]; feats=[]
    with torch.inference_mode(), torch.autocast("cuda",dtype=torch.bfloat16):
        for group in ([full] if full else [])+[[c] for c in tail]:
            inp={k:v.to(dev) for k,v in proc(group, sampling_rate=24000, return_tensors="pt").items()}; feats.append(mert(**inp, output_hidden_states=True).hidden_states[20].reshape(-1,1024))
    H=torch.cat(feats,0).float(); T25=int(round(len(mono24)/24000*25)); return torch.nn.functional.interpolate(H.T[None], size=T25, mode="linear", align_corners=False)[0].T.half().cpu().numpy()
def latents(st48):
    out=[]; CH=48000*60
    with torch.inference_mode():
        for s in range(0,len(st48),CH):
            seg=st48[s:s+CH]
            if len(seg)<1920: break
            out.append(vae.encode(torch.tensor(seg.T[None]))[0].T.float().cpu())
    return torch.cat(out,0).numpy()
for f in sorted(glob.glob(f"{SRC}/*.flac")):
    name=os.path.basename(f)[:-5]; d=f"{OUT}/{name}"
    if os.path.exists(f"{d}/prefix.npy"): continue
    os.makedirs(d,exist_ok=True); a,sr=sf.read(f, dtype="float32"); a=np.stack([a,a],1) if a.ndim==1 else a
    g=gcd(sr,48000); st48=resample_poly(a,48000//g,sr//g,axis=0).astype(np.float32) if sr!=48000 else a
    g=gcd(sr,24000); m24=resample_poly(a.mean(1),24000//g,sr//g).astype(np.float32)
    M=mert_l20(m24); Z=latents(st48); n=min(len(M),len(Z)); np.save(f"{d}/mert.npy",M[:n]); np.save(f"{d}/lat.npy",Z[:n].astype(np.float32))
    cap=open(f"{SRC}/{name}.txt").read().split("===LYRICS===")[0].replace("Global Metadata:","").strip(); style=" ".join(cap.split())[:1500]
    lyr=open(f"{SRC}/{name}.lyrics.txt").read().strip() if os.path.exists(f"{SRC}/{name}.lyrics.txt") else "[instrumental]"
    pre=token_prefixes(SongRequest(style=style,lyrics=lyr,cot="off",seed=1,id="real"),tok); np.save(f"{d}/prefix.npy",np.array(pre,dtype=np.int64))
    print(f"{name}: frames {n} ({n/25/60:.1f} min) prefix {len(pre)} mert-lat frame diff {len(M)-len(Z)}", flush=True)
print("PREP DONE", flush=True)
