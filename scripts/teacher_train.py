"""NAR-as-teacher fine-tune of the MERT->semantic head on REAL audio (no token labels):
  real window: MERT L20 -> head -> straight-through hard tokens (soft grads via codec embedding) -> frozen YuE2 AR prefill + NAR velocity
               at random t on the TRUE VAE latents -> MSE(v_pred, noise - x1). Backprop into the head only.
  minted window: soft-target CE (v2 recipe) to keep minted accuracy.
Eval: held-out real track NAR loss (fixed windows/noise/t) + minted held-out top-1. usage: teacher_train.py <name> <steps> <lambda_nar> <init_ckpt>"""
import os, sys, glob, json, math, time, random, hashlib, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
os.environ.setdefault("HF_HOME","/workspace/hf"); torch.backends.cuda.matmul.allow_tf32=True
from yue2.modeling_yue2 import YuE2ForCausalLM
from yue2.protocol import CODEC_OFFSET, MUSIC_END
from yue2.nar import attention as nar_attention
NAME=sys.argv[1]; STEPS=int(sys.argv[2]); LAM=float(sys.argv[3]); INIT=sys.argv[4]; HOLD=os.environ.get("HOLD_TRACK","")  # held-out track name for the real-audio metric; defaults to the first track
W="/workspace/tok/full"; ROOT="/workspace/yue2-corpus/tracks"; RP="/workspace/real/prep"; OUT=f"{W}/{NAME}"; os.makedirs(OUT,exist_ok=True); dev="cuda"
VOCAB=32768; WIN=512; D=512; L=8; H=8; LR=1e-4; ALPHA=0.25; TAU=0.05; MB=16
snap=glob.glob("/workspace/hf/hub/models--m-a-p--YuE2-3B/snapshots/*")[0]
model=YuE2ForCausalLM.from_pretrained(snap, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval().to(dev); model.requires_grad_(False)
bb=model.model; Ecodec=bb.embed_tokens.weight[CODEC_OFFSET:CODEC_OFFSET+VOCAB]   # [V,2048] bf16, frozen
class Tok(nn.Module):
    def __init__(s, din):
        super().__init__(); s.inp=nn.Linear(din,D); s.pos=nn.Parameter(torch.zeros(1,WIN,D))
        layer=nn.TransformerEncoderLayer(D,H,4*D,dropout=0.1,batch_first=True,norm_first=True,activation="gelu"); s.enc=nn.TransformerEncoder(layer,L); s.norm=nn.LayerNorm(D); s.head=nn.Linear(D,VOCAB)
    def forward(s,x): return s.head(s.norm(s.enc(s.inp(x)+s.pos[:,:x.shape[1]])))
head=Tok(1024).to(dev); head.load_state_dict(torch.load(INIT,map_location=dev)["model"]); opt=torch.optim.AdamW(head.parameters(),lr=LR,weight_decay=0.05,betas=(0.9,0.95))
def instnorm(x): x=x.astype(np.float32); return (x-x.mean(0))/(x.std(0)+1e-5)
# real data
real=[]; hold=None
for d in sorted(glob.glob(f"{RP}/*")):
    item=dict(name=os.path.basename(d), mert=instnorm(np.load(f"{d}/mert.npy")), lat=np.load(f"{d}/lat.npy"), prefix=[int(v) for v in np.load(f"{d}/prefix.npy")])
    if item["name"]==HOLD: hold=item
    else: real.append(item)
if hold is None: hold=real.pop(0); print("HOLD_TRACK not set; holding out", hold["name"], flush=True)
print(f"real train tracks {len(real)} | held-out {hold['name'] if hold else None}", flush=True)
# minted data (soft-target CE)
held=lambda p: int(hashlib.md5(p.encode()).hexdigest(),16)%20==0; pids=[os.path.basename(f)[:-4] for f in sorted(glob.glob(f"{W}/feats/*.npy"))]
def load_m(p):
    y=np.load(f"{ROOT}/{p}/semantic.npy").astype(np.int64); x=np.load(f"{W}/feats/{p}.npy")[3]; n=min(len(x),len(y)); return instnorm(x[:n]).astype(np.float16), y[:n]
mtrain=[load_m(p) for p in pids if not held(p)]; mval=[load_m(p) for p in pids if held(p)]
NB=torch.tensor(np.load(f"{W}/sem_nbr_idx.npy").astype(np.int64),device=dev); NW=torch.softmax(torch.tensor(np.load(f"{W}/sem_nbr_cos.npy"),device=dev)/TAU,dim=1)
def mbatch(data,bs):
    xs,ys=[],[]
    for _ in range(bs):
        x,y=random.choice(data); s=random.randint(0,max(0,len(x)-WIN)); xw=x[s:s+WIN].astype(np.float32); yw=y[s:s+WIN]
        if len(xw)<WIN: pad=WIN-len(xw); xw=np.pad(xw,((0,pad),(0,0))); yw=np.pad(yw,(0,pad),constant_values=-100)
        xs.append(xw); ys.append(yw)
    return torch.tensor(np.stack(xs),device=dev), torch.tensor(np.stack(ys),device=dev)
def soft_ce(lg,y):
    m=y!=-100; lg=lg[m].float(); y=y[m]; logp=F.log_softmax(lg,-1); return ((1-ALPHA)*(-logp.gather(1,y[:,None])[:,0])+ALPHA*(-(logp.gather(1,NB[y])*NW[y]).sum(1))).mean()
# --- differentiable NAR loss
def ar_layer(layer,x,cos_,sin_):
    q,k,v=layer.self_attn.project_qkv(layer.input_layernorm(x),cos_,sin_); h=nar_attention(q[0],k[0],v[0],causal=True)
    x=x+layer.self_attn.o_proj(h.flatten(1)[None]); return x+layer.mlp(layer.post_attention_layernorm(x)), k[0], v[0]
def nar_layer(layer,h,ak,av,ncos,nsin):
    q,k,v=layer.nar_self_attn.project_qkv(layer.nar_input_layernorm(h),ncos,nsin); a=nar_attention(q[0],torch.cat((ak,k[0])),torch.cat((av,v[0])))
    h=h+layer.nar_self_attn.o_proj(a.flatten(1)[None]); return h+layer.nar_mlp(layer.nar_pre_mlp_layernorm(h))
def nar_loss(prefix, codec_emb, x1, t, noise, grad=True):
    """prefix list[int]; codec_emb [T,2048] (differentiable); x1 [T,64]; scalar t; noise [T,64]."""
    T=codec_emb.shape[0]; pre=bb.embed_tokens(torch.tensor([prefix],device=dev))[0]; end=bb.embed_tokens(torch.tensor([MUSIC_END],device=dev))
    x=torch.cat((pre,codec_emb.to(pre.dtype),end),0)[None]; Lq=x.shape[1]; cos_,sin_=bb.rotary_emb(torch.arange(Lq,device=dev)[None]); cache=[]
    for layer in bb.layers:
        x,k,v=checkpoint(ar_layer,layer,x,cos_,sin_,use_reentrant=False) if grad else ar_layer(layer,x,cos_,sin_); cache.append((k,v))
    xt=t*noise+(1-t)*x1; target=noise-x1; N=T+2; ncos,nsin=bb.rotary_emb(torch.arange(Lq,Lq+N,device=dev)[None])
    pe=model.latent_pos_embed(torch.arange(N,device=dev).clamp(max=model.config.max_latent_frames-1))[None]; sh=model._shift_t_value(float(np.clip(np.log(t/(1-t)),-20,20)),dev,torch.bfloat16)
    h=model.vae2llm(F.pad(xt.to(torch.bfloat16),(0,0,1,1))[None])+model.time_embedder(sh.expand(N))[None]+pe
    for layer,(ak,av) in zip(bb.layers,cache): h=checkpoint(nar_layer,layer,h,ak,av,ncos,nsin,use_reentrant=False) if grad else nar_layer(layer,h,ak,av,ncos,nsin)
    return F.mse_loss(model.llm2vae(bb.norm(h))[0,1:-1].float(),target)
def st_embed(logits):  # straight-through: forward = hard token embedding, backward = softmax-weighted embedding
    p=torch.softmax(logits.float(),-1); hard=F.one_hot(p.argmax(-1),VOCAB).float(); return ((hard+(p-p.detach())).to(Ecodec.dtype))@Ecodec
def real_window(item, s=None):
    n=min(len(item["mert"]),len(item["lat"])); s=random.randint(0,n-WIN) if s is None else s; return item["mert"][s:s+WIN], item["lat"][s:s+WIN]
@torch.no_grad()
def evaluate():
    head.eval(); n=min(len(hold["mert"]),len(hold["lat"])); tot=0; g=torch.Generator(device="cpu").manual_seed(123)
    for s in [n//4, n//2, 3*n//4]:
        m,z=real_window(hold,s)
        with torch.autocast("cuda",dtype=torch.bfloat16): lg=head(torch.tensor(m[None],device=dev))[0]
        emb=Ecodec[lg.float().argmax(-1)]; noise=torch.randn(WIN,64,generator=g).to(dev)
        for t in (0.2,0.5,0.8): tot+=nar_loss(hold["prefix"],emb,torch.tensor(z,device=dev),t,noise,grad=False).item()
    t1=tot_=0
    for _ in range(24):
        x,y=mbatch(mval,16)
        with torch.autocast("cuda",dtype=torch.bfloat16): lg=head(x)
        mk=y!=-100; t1+=(lg.float().argmax(-1)==y)[mk].sum().item(); tot_+=mk.sum().item()
    head.train(); return tot/9, t1/tot_
h0,a0=evaluate(); print(f"EVAL step 0 real_nar {h0:.4f} minted_top1 {a0:.4f}", flush=True); log=open(f"{OUT}/train.log","a"); log.write(f"EVAL step 0 real_nar {h0:.4f} minted_top1 {a0:.4f}\n"); best=h0
t0=time.time()
for st in range(1,STEPS+1):
    for gp in opt.param_groups: gp["lr"]=LR*min(1,st/50)*0.5*(1+math.cos(math.pi*st/STEPS))
    item=random.choice(real); m,z=real_window(item); t=float(np.clip(np.random.beta(2,2),0.05,0.95)); noise=torch.randn(WIN,64,device=dev)
    with torch.autocast("cuda",dtype=torch.bfloat16): lg=head(torch.tensor(m[None],device=dev))[0]
    ln=nar_loss(item["prefix"],st_embed(lg),torch.tensor(z,device=dev),t,noise)
    x,y=mbatch(mtrain,MB)
    with torch.autocast("cuda",dtype=torch.bfloat16): lgm=head(x)
    lc=soft_ce(lgm,y); loss=lc+LAM*ln; opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(),1.0); opt.step()
    if st<=3 or st%25==0: print(f"step {st} nar {ln.item():.4f} ce {lc.item():.3f} {time.time()-t0:.0f}s mem {torch.cuda.max_memory_allocated()/2**30:.1f}G", flush=True)
    if st%100==0 or st==STEPS:
        hr,acc=evaluate(); msg=f"EVAL step {st} real_nar {hr:.4f} minted_top1 {acc:.4f} {time.time()-t0:.0f}s"; print(msg, flush=True); log.write(msg+"\n"); log.flush()
        if hr<best: best=hr; torch.save({"model":head.state_dict(),"cfg":dict(NAME=NAME,instnorm=True)}, f"{OUT}/best.pt")
        torch.save({"model":head.state_dict()}, f"{OUT}/last.pt")
print(f"RESULT {NAME}: best real_nar {best:.4f} (start {h0:.4f})", flush=True)
