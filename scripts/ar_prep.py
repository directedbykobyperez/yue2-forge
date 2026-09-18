"""AR dataset in score-free (cot=off) format: prefix ids (instructions+tags+lyrics) + codec tokens.
Artist tracks: tokens from the joint_v1 head. Minted: true semantic tokens. -> /workspace/real/ar/dataset.pt
Env: COT=off|full|melody (default: off)"""
import os, glob, json, hashlib, numpy as np, torch, torch.nn as nn, sys
os.environ.setdefault("HF_HOME","/workspace/hf")
from yue2.protocol import SongRequest, token_prefixes
from yue2.tokenization_yue2 import YuE2TextTokenizer
HEAD_CK=sys.argv[1] if len(sys.argv)>1 else "/workspace/tok/full/joint_v1/head_best.pt"
W="/workspace/tok/full"; ROOT="/workspace/yue2-corpus/tracks"; RP="/workspace/real/prep"; dev="cuda"; VOCAB=32768; WIN=512; D=512; L=8; H=8
snap=glob.glob("/workspace/hf/hub/models--m-a-p--YuE2-3B/snapshots/*")[0]; tok=YuE2TextTokenizer(snap+"/qwen.tiktoken")
COT=os.environ.get("COT","off")
print(f"COT mode: {COT}", flush=True)

class Tok(nn.Module):
    def __init__(s, din):
        super().__init__(); s.inp=nn.Linear(din,D); s.pos=nn.Parameter(torch.zeros(1,WIN,D))
        layer=nn.TransformerEncoderLayer(D,H,4*D,dropout=0.1,batch_first=True,norm_first=True,activation="gelu"); s.enc=nn.TransformerEncoder(layer,L); s.norm=nn.LayerNorm(D); s.head=nn.Linear(D,VOCAB)
    def forward(s,x): return s.head(s.norm(s.enc(s.inp(x)+s.pos[:,:x.shape[1]])))
head=Tok(1024).to(dev).eval(); head.load_state_dict(torch.load(HEAD_CK,map_location=dev)["model"])
def instnorm(x): x=x.astype(np.float32); return (x-x.mean(0))/(x.std(0)+1e-5)
@torch.no_grad()
def predict(x):
    T=len(x); out=np.zeros(T,dtype=np.int64); starts=list(range(0,max(1,T-WIN+1),WIN//2))
    if starts[-1]+WIN<T: starts.append(max(0,T-WIN))
    for s0 in starts:
        xw=x[s0:s0+WIN]; n=len(xw)
        if n<WIN: xw=np.pad(xw,((0,WIN-n),(0,0)))
        with torch.autocast("cuda",dtype=torch.bfloat16): pred=head(torch.tensor(xw[None],device=dev))[0,:n].float().argmax(-1).cpu().numpy()
        lo=s0+(0 if s0==0 else WIN//4); hi=s0+n-(0 if s0+n>=T else WIN//4); out[lo:hi]=pred[lo-s0:hi-s0]
    return out

def load_abc_sheet(name):
    """Load ABC sheet from prep directory."""
    path = f"{RP}/{name}/abc_sheet.txt"
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read().strip()
    return None

def build_prefix_with_abc(style, lyrics, abc_sheet, cot_mode):
    """Build prefix with ABC sheet if available."""
    if cot_mode == "off" or abc_sheet is None:
        return token_prefixes(SongRequest(style=style,lyrics=lyr,cot="off",seed=1,id="c"),tok)
    # For cot=full or cot=melody, we need to include the ABC sheet in the prompt
    # The YuE2 protocol expects the ABC sheet after [Lyrics]
    # We'll use the off-mode prefix for now and add ABC sheet handling later
    return token_prefixes(SongRequest(style=style,lyrics=lyr,cot="off",seed=1,id="c"),tok)

data=[]
for d in sorted(glob.glob(f"{RP}/*")):
    name=os.path.basename(d); toks=predict(instnorm(np.load(f"{d}/mert.npy")))
    src="/workspace/real/artist"; cap=open(f"{src}/{name}.txt").read().split("===LYRICS===")[0].replace("Global Metadata:","").strip(); style=" ".join(cap.split())[:1500]
    fl=f"/workspace/real/artist_lyrics/{name}.lyrics.txt"; lyr=open(fl).read().strip() if os.path.exists(fl) else (open(f"{src}/{name}.lyrics.txt").read().strip() if os.path.exists(f"{src}/{name}.lyrics.txt") else "[instrumental]")
    abc_sheet = load_abc_sheet(name)
    data.append(dict(name=name, src="artist", style=style, lyrics=lyr, prefix=token_prefixes(SongRequest(style=style,lyrics=lyr,cot="off",seed=1,id="c"),tok), codec=toks.astype(np.int32), abc_sheet=abc_sheet))
    if abc_sheet:
        print(f"  {name}: ABC sheet loaded ({len(abc_sheet)} chars)", flush=True)
print("artist", len(data), "mean frames", int(np.mean([len(x["codec"]) for x in data])), flush=True)
PACK=os.environ.get("REG_PACK","/workspace/real/regularizer/minted_regularizer_pack.pt")
pack=torch.load(PACK, map_location="cpu", weights_only=False); nm=0
for r_ in pack:
    y=np.asarray(r_["codec"]).astype(np.int32); sd=int(hashlib.md5(r_["name"].encode()).hexdigest(),16)%100000
    data.append(dict(name=r_["name"], src=r_["src"], style=r_["style"], lyrics=r_["lyrics"], prefix=token_prefixes(SongRequest(style=r_["style"],lyrics=r_["lyrics"],cot="off",seed=sd,id=r_["name"]),tok), codec=y, abc_sheet=None)); nm+=1
print("minted", nm, flush=True); os.makedirs("/workspace/real/ar", exist_ok=True); torch.save(data, "/workspace/real/ar/dataset.pt"); print("AR PREP DONE", len(data), flush=True)
