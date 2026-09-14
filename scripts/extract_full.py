"""Full-corpus MERT-v2-FullSong features, layers KEEP only, 25 Hz, fp16 [len(KEEP),T,1024]. Batches all 30 s chunks of a track in one forward."""
import glob, os, sys, time, json, numpy as np, torch, soundfile as sf
from scipy.signal import resample_poly
from math import gcd
from transformers import AutoModel, AutoFeatureExtractor
ROOT="/workspace/yue2-corpus/tracks"; OUT="/workspace/tok/full/feats"; os.makedirs(OUT, exist_ok=True)
NAME="m-a-p/MERT-v2-FullSong"; KEEP=[1,8,14,20]
LIMIT=int(sys.argv[1]) if len(sys.argv)>1 else 10**9
tracks=[d for d in sorted(glob.glob(f"{ROOT}/*")) if os.path.exists(f"{d}/semantic.npy") and os.path.exists(f"{d}/item.json")][:LIMIT]
t0=time.time()
proc=AutoFeatureExtractor.from_pretrained(NAME, trust_remote_code=True)
model=AutoModel.from_pretrained(NAME, trust_remote_code=True).cuda().eval()
sr_t=getattr(proc,"sampling_rate",24000); CH=sr_t*30
print(f"loaded in {time.time()-t0:.0f}s | {len(tracks)} tracks | keep layers {KEEP}", flush=True)
done=0
for i,d in enumerate(tracks):
    pid=os.path.basename(d); out=f"{OUT}/{pid}.npy"
    if os.path.exists(out): continue
    a,sr=sf.read(f"{d}/audio.flac", dtype="float32"); a=a.mean(1) if a.ndim==2 else a
    g=gcd(sr,sr_t); a=resample_poly(a, sr_t//g, sr//g).astype(np.float32)
    chunks=[a[s:s+CH] for s in range(0,len(a),CH)]; chunks=[c for c in chunks if len(c)>=sr_t]
    full=[c for c in chunks if len(c)==CH]; tail=[c for c in chunks if len(c)<CH]
    feats=[]
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
        for group in ([full] if full else [])+[[c] for c in tail]:
            inp=proc(group, sampling_rate=sr_t, return_tensors="pt")
            inp={k:v.cuda() for k,v in inp.items()}
            o=model(**inp, output_hidden_states=True)
            hs=torch.stack([o.hidden_states[k] for k in KEEP])   # [K,B,T,C]
            K,B,T,C=hs.shape; feats.append(hs.reshape(K,B*T,C))  # chunks are in time order
    Hh=torch.cat(feats,1).float()
    T25=int(round(len(a)/sr_t*25))
    H25=torch.nn.functional.interpolate(Hh.permute(0,2,1), size=T25, mode="linear", align_corners=False).permute(0,2,1)
    np.save(out, H25.half().cpu().numpy()); done+=1
    if done%25==0: print(f"{i+1}/{len(tracks)} {pid} frames25={T25} {time.time()-t0:.0f}s", flush=True)
json.dump({"keep":KEEP}, open(f"{OUT}/../info.json","w")); print("EXTRACT DONE", done, f"{time.time()-t0:.0f}s", flush=True)
