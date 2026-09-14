"""v2 head: MERT L20 -> semantic token. Fixes vs v0 recipe: (1) per-track instance normalization of features (domain shift),
(2) augmented-audio variants mixed in (USE_AUG=1), (3) soft targets over the LM-embedding neighbours of the true code, (4) dropout 0.2.
usage: train_v2.py <outname> <steps> <use_aug 0|1>"""
import glob, json, math, os, random, time, hashlib, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32=True
NAME=sys.argv[1]; STEPS=int(sys.argv[2]); USE_AUG=int(sys.argv[3]); ALPHA=0.25; TAU=0.05; DROP=0.2
W="/workspace/tok/full"; ROOT="/workspace/yue2-corpus/tracks"; FE=f"{W}/feats"; FA=f"{W}/feats_aug"; OUT=f"{W}/{NAME}"; os.makedirs(OUT,exist_ok=True)
VOCAB=32768; WIN=512; BATCH=32; LR=3e-4; D=512; L=8; H=8; dev="cuda"
pids=[os.path.basename(f)[:-4] for f in sorted(glob.glob(f"{FE}/*.npy"))]; held=lambda p: int(hashlib.md5(p.encode()).hexdigest(),16)%20==0
train=[p for p in pids if not held(p)]; val=[p for p in pids if held(p)]; print(f"{NAME}: train {len(train)} val {len(val)} aug {USE_AUG}", flush=True)
def instnorm(x): x=x.astype(np.float32); return ((x-x.mean(0))/(x.std(0)+1e-5)).astype(np.float16)
def load(p):
    y=np.load(f"{ROOT}/{p}/semantic.npy").astype(np.int64); xs=[instnorm(np.load(f"{FE}/{p}.npy")[3])]
    if USE_AUG: xs+=[instnorm(np.load(f)) for f in sorted(glob.glob(f"{FA}/{p}_k*.npy"))]
    n=min(min(len(x) for x in xs),len(y)); return [x[:n] for x in xs], y[:n]
t0=time.time(); train_data=[load(p) for p in train]; val_data=[load(p) for p in val]; print(f"loaded {time.time()-t0:.0f}s variants/track {np.mean([len(x) for x,_ in train_data]):.2f}", flush=True)
NB=torch.tensor(np.load(f"{W}/sem_nbr_idx.npy").astype(np.int64),device=dev); NC=torch.tensor(np.load(f"{W}/sem_nbr_cos.npy"),device=dev); NW=torch.softmax(NC/TAU,dim=1)  # [V,16]
def batch(data, bs, aug=True):
    xs, ys = [], []
    for _ in range(bs):
        vs,y=random.choice(data); x=random.choice(vs) if aug else vs[0]; s=random.randint(0,max(0,len(x)-WIN)); xw=x[s:s+WIN].astype(np.float32); yw=y[s:s+WIN]
        if len(xw)<WIN: pad=WIN-len(xw); xw=np.pad(xw,((0,pad),(0,0))); yw=np.pad(yw,(0,pad),constant_values=-100)
        xs.append(xw); ys.append(yw)
    return torch.tensor(np.stack(xs),device=dev), torch.tensor(np.stack(ys),device=dev)
class Tok(nn.Module):
    def __init__(s, din):
        super().__init__(); s.inp=nn.Linear(din,D); s.pos=nn.Parameter(torch.zeros(1,WIN,D)); nn.init.normal_(s.pos,std=0.02)
        layer=nn.TransformerEncoderLayer(D,H,4*D,dropout=DROP,batch_first=True,norm_first=True,activation="gelu")
        s.enc=nn.TransformerEncoder(layer,L); s.norm=nn.LayerNorm(D); s.head=nn.Linear(D,VOCAB)
    def forward(s,x): return s.head(s.norm(s.enc(s.inp(x)+s.pos[:,:x.shape[1]])))
def soft_loss(lg, y):
    m=y!=-100; lg=lg[m].float(); y=y[m]; logp=F.log_softmax(lg,-1)
    hard=-logp.gather(1,y[:,None])[:,0]; soft=-(logp.gather(1,NB[y])*NW[y]).sum(1)
    return ((1-ALPHA)*hard+ALPHA*soft).mean()
m=Tok(1024).to(dev); opt=torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.05, betas=(0.9,0.95)); sched=lambda st: LR*min(1, st/200)*0.5*(1+math.cos(math.pi*min(st,STEPS)/STEPS))
@torch.no_grad()
def evaluate(n=64):
    m.eval(); top1=top5=nbr=tot=0
    for _ in range(n):
        x,y=batch(val_data,16,aug=False)
        with torch.autocast("cuda",dtype=torch.bfloat16): lg=m(x)
        lg=lg.float(); mask=y!=-100; t5=lg.topk(5,-1).indices; p1=t5[...,0]
        top1+=(p1==y)[mask].sum().item(); top5+=(t5==y[...,None]).any(-1)[mask].sum().item(); tot+=mask.sum().item()
        yy=y.clamp(min=0); nbr+=((p1==y)|(NB[yy]==p1[...,None]).any(-1))[mask].sum().item()
    m.train(); return top1/tot, top5/tot, nbr/tot
t0=time.time(); best=(0,0,0); log=open(f"{OUT}/train.log","a")
for st in range(1,STEPS+1):
    for g in opt.param_groups: g["lr"]=sched(st)
    x,y=batch(train_data,BATCH)
    with torch.autocast("cuda",dtype=torch.bfloat16): lg=m(x)
    loss=soft_loss(lg,y); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    if st%500==0 or st==STEPS:
        t1,t5,nb=evaluate(); msg=f"EVAL {NAME} step {st} loss {loss.item():.3f} top1 {t1:.4f} top5 {t5:.4f} top1-or-nbr {nb:.4f} {time.time()-t0:.0f}s"; print(msg, flush=True); log.write(msg+"\n"); log.flush()
        if t1>best[0]: best=(t1,t5,nb); torch.save({"model":m.state_dict(),"cfg":dict(NAME=NAME,VOCAB=VOCAB,WIN=WIN,D=D,L=L,H=H,instnorm=True)}, f"{OUT}/best.pt")
json.dump({"name":NAME,"best_top1":best[0],"top5":best[1],"top1_or_nbr":best[2],"steps":STEPS,"aug":USE_AUG,"train_tracks":len(train),"val_tracks":len(val)}, open(f"{OUT}/result.json","w"))
print(f"RESULT {NAME}: top1 {best[0]:.4f} top5 {best[1]:.4f} top1-or-nbr {best[2]:.4f}", flush=True)
