"""Lyric-cursor targets for the artist's tracks: demucs vocal stem -> MMS_FA forced alignment of the FULL lyrics -> per word (start,end,score,char0,char1)
-> /workspace/real/prep/<name>/cursor_words.npy. Char offsets are into the exact lyrics string used in the AR prefix."""
import os, re, glob, subprocess, numpy as np, torch, torchaudio, soundfile as sf
from scipy.signal import resample_poly
from math import gcd
SRC="/workspace/real/artist"; LYD="/workspace/real/artist_lyrics"; ST="/workspace/real/stems"; RP="/workspace/real/prep"
bundle=torchaudio.pipelines.MMS_FA; model=bundle.get_model(with_star=False).cuda().eval(); labels=bundle.get_labels(star=None); d={c:i for i,c in enumerate(labels)}
def words_of(lyr):
    out=[]; pos=0
    for line in lyr.split("\n"):
        if not re.match(r"^\s*\[.*\]\s*$",line):
            for m in re.finditer(r"\S+",line):
                w=re.sub(r"[^a-z']","",m.group().lower().replace("’","'"))
                if re.search(r"[a-z]",w): out.append((pos+m.start(),pos+m.end(),w))
        pos+=len(line)+1
    return out
for f in sorted(glob.glob(f"{SRC}/*.flac")):
    name=os.path.basename(f)[:-5]; lyr=open(f"{LYD}/{name}.lyrics.txt").read().strip(); outp=f"{RP}/{name}/cursor_words.npy"
    if lyr.strip()=="[instrumental]" or os.path.exists(outp): print("skip",name,flush=True); continue
    stem=f"{ST}/htdemucs/{name}/vocals.wav"
    if not os.path.exists(stem): subprocess.run(["/workspace/yue2venv/bin/python","-m","demucs","--two-stems=vocals","-n","htdemucs","-o",ST,f],check=True,capture_output=True)
    a,sr=sf.read(stem,dtype="float32"); a=a.mean(1) if a.ndim==2 else a; g=gcd(sr,16000); a=resample_poly(a,16000//g,sr//g).astype(np.float32)
    ws=words_of(lyr); toks=[[d[c] for c in w if c in d] for _,_,w in ws]; keep=[i for i,t in enumerate(toks) if t]; flat=[t for i in keep for t in toks[i]]
    with torch.inference_mode(): em=torch.log_softmax(model(torch.tensor(a[None]).cuda())[0],-1)
    ali,sc=torchaudio.functional.forced_align(em,torch.tensor([flat],device="cuda"),blank=0); spans=torchaudio.functional.merge_tokens(ali[0],sc[0].exp()); fps=len(a)/16000/em.shape[1]
    rows=[]; k=0
    for i in keep:
        seg=spans[k:k+len(toks[i])]; k+=len(toks[i]); rows.append((seg[0].start*fps, seg[-1].end*fps, float(np.mean([s.score for s in seg])), ws[i][0], ws[i][1]))
    arr=np.array(rows,dtype=np.float32); np.save(outp,arr); st=arr[:,0]
    print(f"{name}: {len(rows)} words | {st.min():.0f}s..{arr[:,1].max():.0f}s of {len(a)/16000:.0f}s | mean score {arr[:,2].mean():.2f} | monotonic {bool((np.diff(st)>=0).all())} | max gap {np.diff(st).max():.1f}s", flush=True)
print("CURSOR PREP DONE", flush=True)
