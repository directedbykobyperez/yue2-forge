"""AR-branch LoRA: teach YuE2's AR to write token streams (joint_v1 head dialect) for an artist from style+lyrics (cot=off), regularized with minted songs.
usage: ar_lora.py <name> <steps> <rank> <artist_frac>
Env: VRAM_MODE=low|high, SCHED_STEPS, CK_FROM, CK_EVERY"""
import os, sys, glob, math, time, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
sys.path.insert(0, os.path.dirname(__file__))
from vram_helpers import load_yue2_model, get_snap_path, get_maxlen, get_vram_mode
os.environ.setdefault("HF_HOME","/workspace/hf"); torch.backends.cuda.matmul.allow_tf32=True
from yue2.protocol import CODEC_OFFSET, MUSIC_END
from yue2.nar import attention as nar_attention
NAME=sys.argv[1]; STEPS=int(sys.argv[2]); RANK=int(sys.argv[3]); CFRAC=float(sys.argv[4]); INIT=sys.argv[5] if len(sys.argv)>5 else "none"; LR=float(sys.argv[6]) if len(sys.argv)>6 else 1e-4; OUT=f"/workspace/tok/full/{NAME}"; os.makedirs(OUT,exist_ok=True); dev="cuda"
MAXLEN=get_maxlen(); ACC=2; CHUNK=1024; SCHED=int(os.environ.get('SCHED_STEPS','3000')); CK_FROM=int(os.environ.get('CK_FROM','600')); CK_EVERY=int(os.environ.get('CK_EVERY','200'))
result=load_yue2_model(dev)
if isinstance(result, tuple): model, snap = result
else: model, snap = result, get_snap_path()
bb=model.model
if get_vram_mode() == "low":
    try: bb.gradient_checkpointing_enable()
    except Exception: pass
class LoRALinear(nn.Module):
    def __init__(s, base, r):
        super().__init__(); s.base=base; s.A=nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device)*(1/math.sqrt(base.in_features))); s.B=nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
    def forward(s,x): return s.base(x)+((x.float()@s.A.T)@s.B.T).to(x.dtype)
lora=[]
for layer in bb.layers:
    for mod,names in ((layer.self_attn,("q_proj","k_proj","v_proj","o_proj")),(layer.mlp,("gate_proj","up_proj","down_proj"))):
        for n in names: l=LoRALinear(getattr(mod,n),RANK); setattr(mod,n,l); lora+=[l.A,l.B]
if INIT!="none":
    ck=torch.load(INIT,map_location=dev)
    with torch.no_grad(): [p.copy_(v.to(dev)) for p,v in zip(lora,ck["lora"])]
    print("init from", INIT, flush=True)
print(f"AR LoRA params {sum(p.numel() for p in lora)/1e6:.1f}M lr {LR}", flush=True)
opt=torch.optim.AdamW(lora,lr=LR,weight_decay=0.0,betas=(0.9,0.95))
data=torch.load("/workspace/real/ar/dataset.pt", weights_only=False); artist=[x for x in data if x["src"]=="artist"]; minted=[x for x in data if x["src"]=="minted"]; mval=[x for x in data if x["src"]=="minted_val"][:6]
print(f"artist {len(artist)} minted {len(minted)} val {len(mval)}", flush=True)
def seq_of(item):
    pre=list(item["prefix"]); cod=[int(c)+CODEC_OFFSET for c in item["codec"]]; room=MAXLEN-len(pre)-1
    body=cod[:room]+([MUSIC_END] if len(cod)<=room else []); return torch.tensor([pre+body],device=dev), len(pre)
def ar_layer(layer,x,cos_,sin_):
    q,k,v=layer.self_attn.project_qkv(layer.input_layernorm(x),cos_,sin_); h=nar_attention(q[0],k[0],v[0],causal=True)
    x=x+layer.self_attn.o_proj(h.flatten(1)[None]); return x+layer.mlp(layer.post_attention_layernorm(x))
def lm_loss(ids, Lp, grad=True):
    x=bb.embed_tokens(ids); Lq=ids.shape[1]; cos_,sin_=bb.rotary_emb(torch.arange(Lq,device=dev)[None])
    for layer in bb.layers: x=checkpoint(ar_layer,layer,x,cos_,sin_,use_reentrant=False) if grad else ar_layer(layer,x,cos_,sin_)
    h=bb.norm(x[0,Lp-1:-1]); tgt=ids[0,Lp:]; tot=0.
    for s in range(0,h.shape[0],CHUNK):
        lg=model.lm_head(h[s:s+CHUNK]).float(); tot=tot+F.cross_entropy(lg,tgt[s:s+CHUNK],reduction="sum")
    return tot/h.shape[0]
@torch.no_grad()
def evaluate():
    r={}
    for tag,items in (("minted_val",mval),("artist",artist[:6])):
        tot=0
        for it in items: ids,Lp=seq_of(it); tot+=lm_loss(ids,Lp,grad=False).item()
        r[tag]=tot/len(items)
    return r
e=evaluate(); print(f"EVAL step 0 minted_val {e['minted_val']:.3f} artist {e['artist']:.3f}", flush=True); log=open(f"{OUT}/train.log","a"); log.write(str(e)+"\n"); t0=time.time(); best=e["artist"]
def save(path): torch.save({"lora":[p.detach().cpu() for p in lora],"rank":RANK,"targets":"ar self_attn qkvo + mlp gate/up/down"}, path)
for st in range(1,STEPS+1):
    for g in opt.param_groups: g["lr"]=LR*min(1,st/50)*(0.2+0.8*0.5*(1+math.cos(math.pi*min(st,SCHED)/SCHED)))
    for _ in range(ACC):
        it=random.choice(artist) if random.random()<CFRAC else random.choice(minted); ids,Lp=seq_of(it); loss=lm_loss(ids,Lp)/ACC; loss.backward()
    torch.nn.utils.clip_grad_norm_(lora,1.0); opt.step(); opt.zero_grad(set_to_none=True)
    if st<=3 or st%20==0: print(f"step {st} loss {loss.item()*ACC:.3f} len {ids.shape[1]} {time.time()-t0:.0f}s mem {torch.cuda.max_memory_allocated()/2**30:.1f}G", flush=True)
    if st%100==0 or st==STEPS:
        e=evaluate(); msg=f"EVAL step {st} minted_val {e['minted_val']:.3f} artist {e['artist']:.3f} {time.time()-t0:.0f}s"; print(msg, flush=True); log.write(msg+"\n"); log.flush()
        if e["artist"]<best: best=e["artist"]; save(f"{OUT}/best.pt")
        save(f"{OUT}/last.pt")
        if st>=CK_FROM and (st-CK_FROM)%CK_EVERY==0: save(f"{OUT}/step-{st}.pt")
print(f"RESULT {NAME}: best artist CE {best:.3f}", flush=True)
