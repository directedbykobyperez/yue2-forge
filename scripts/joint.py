"""Head and/or NAR-LoRA training on REAL audio with the flow loss as teacher.
usage: joint.py <name> <steps> <train_head 0|1> <train_lora 0|1> <init_head.pt> <init_lora.pt|none> [rank]
Real window: MERT -> head -> straight-through tokens -> (LoRA'd) NAR flow loss on true VAE latents. Head also gets minted soft-CE each step;
in LoRA mode 25% of flow windows are minted (true tokens). Eval = held-out real flow loss (fixed windows/t/noise), minted top-1, repeat rate.
Ends by rendering the held-out track with the best head+NAR."""
import os, sys, glob, json, math, time, random, hashlib, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, soundfile as sf
from torch.utils.checkpoint import checkpoint
os.environ.setdefault("HF_HOME","/workspace/hf"); torch.backends.cuda.matmul.allow_tf32=True
from yue2.modeling_yue2 import YuE2ForCausalLM
from yue2.modeling_vae import YuE2VAE
from yue2.protocol import CODEC_OFFSET, MUSIC_END, SongRequest, token_prefixes
from yue2.tokenization_yue2 import YuE2TextTokenizer
from yue2.nar import attention as nar_attention, synthesize
NAME=sys.argv[1]; STEPS=int(sys.argv[2]); TRAIN_HEAD=int(sys.argv[3]); TRAIN_LORA=int(sys.argv[4]); INIT_HEAD=sys.argv[5]; INIT_LORA=sys.argv[6]; RANK=int(sys.argv[7]) if len(sys.argv)>7 else 32; HOLD=os.environ.get("HOLD_TRACK","")  # held-out track name for the real-audio metric; defaults to the first track
W="/workspace/tok/full"; ROOT="/workspace/yue2-corpus/tracks"; RP="/workspace/real/prep"; OUT=f"{W}/{NAME}"; os.makedirs(OUT,exist_ok=True); dev="cuda"
VOCAB=32768; WIN=512; D=512; L=8; H=8; LR_HEAD=1e-4; LR_LORA=5e-5; LR_IO=2e-5; ALPHA=0.25; TAU=0.05; MB=16; MINTED_FLOW_P=0.25
snap=glob.glob("/workspace/hf/hub/models--m-a-p--YuE2-3B/snapshots/*")[0]; vsnap=glob.glob("/workspace/hf/hub/models--m-a-p--YuE2-Vae/snapshots/*")[0]
model=YuE2ForCausalLM.from_pretrained(snap, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval().to(dev); model.requires_grad_(False); bb=model.model
tok=YuE2TextTokenizer(snap+"/qwen.tiktoken"); Ecodec=bb.embed_tokens.weight[CODEC_OFFSET:CODEC_OFFSET+VOCAB]
class LoRALinear(nn.Module):
    def __init__(s, base, r):
        super().__init__(); s.base=base; s.A=nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device)*(1/math.sqrt(base.in_features))); s.B=nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
    def forward(s,x): return s.base(x)+((x.float()@s.A.T)@s.B.T).to(x.dtype)
lora_params=[]
for layer in bb.layers:
    for mod,names in ((layer.nar_self_attn,("q_proj","k_proj","v_proj","o_proj")),(layer.nar_mlp,("gate_proj","up_proj","down_proj"))):
        for n in names: l=LoRALinear(getattr(mod,n),RANK); setattr(mod,n,l); lora_params+=[l.A,l.B]
model.vae2llm.float(); model.llm2vae.float(); io_params=list(model.vae2llm.parameters())+list(model.llm2vae.parameters())
def load_lora(path):
    ck=torch.load(path,map_location=dev)
    with torch.no_grad():
        for p,v in zip(lora_params,ck["lora"]): p.copy_(v.to(dev))
        model.vae2llm.load_state_dict({k:v.float() for k,v in ck["io"]["vae2llm"].items()}); model.llm2vae.load_state_dict({k:v.float() for k,v in ck["io"]["llm2vae"].items()})
def save_lora(path): torch.save({"lora":[p.detach().cpu() for p in lora_params],"io":{"vae2llm":model.vae2llm.state_dict(),"llm2vae":model.llm2vae.state_dict()},"rank":RANK}, path)
if INIT_LORA!="none": load_lora(INIT_LORA); print("loaded NAR LoRA", INIT_LORA, flush=True)
for p in lora_params+io_params: p.requires_grad_(bool(TRAIN_LORA))
class Tok(nn.Module):
    def __init__(s, din):
        super().__init__(); s.inp=nn.Linear(din,D); s.pos=nn.Parameter(torch.zeros(1,WIN,D))
        layer=nn.TransformerEncoderLayer(D,H,4*D,dropout=0.1,batch_first=True,norm_first=True,activation="gelu"); s.enc=nn.TransformerEncoder(layer,L); s.norm=nn.LayerNorm(D); s.head=nn.Linear(D,VOCAB)
    def forward(s,x): return s.head(s.norm(s.enc(s.inp(x)+s.pos[:,:x.shape[1]])))
head=Tok(1024).to(dev); head.load_state_dict(torch.load(INIT_HEAD,map_location=dev)["model"]); head.requires_grad_(bool(TRAIN_HEAD))
groups=[]
if TRAIN_HEAD: groups.append({"params":list(head.parameters()),"lr":LR_HEAD,"weight_decay":0.05})
if TRAIN_LORA: groups+=[{"params":lora_params,"lr":LR_LORA,"weight_decay":0.0},{"params":io_params,"lr":LR_IO,"weight_decay":0.0}]
opt=torch.optim.AdamW(groups,betas=(0.9,0.95)); base_lrs=[g["lr"] for g in opt.param_groups]
def instnorm(x): x=x.astype(np.float32); return (x-x.mean(0))/(x.std(0)+1e-5)
real=[]; hold=None
for d in sorted(glob.glob(f"{RP}/*")):
    item=dict(name=os.path.basename(d), mert=instnorm(np.load(f"{d}/mert.npy")), lat=np.load(f"{d}/lat.npy"), prefix=[int(v) for v in np.load(f"{d}/prefix.npy")]); n=min(len(item["mert"]),len(item["lat"])); item["mert"]=item["mert"][:n]; item["lat"]=item["lat"][:n]
    if item["name"]==HOLD: hold=item
    else: real.append(item)
if hold is None: hold=real.pop(0); print("HOLD_TRACK not set; holding out", hold["name"], flush=True)
held=lambda p: int(hashlib.md5(p.encode()).hexdigest(),16)%20==0; pids=[os.path.basename(f)[:-4] for f in sorted(glob.glob(f"{W}/feats/*.npy"))]
def load_m(p):
    y=np.load(f"{ROOT}/{p}/semantic.npy").astype(np.int64); x=np.load(f"{W}/feats/{p}.npy")[3]; n=min(len(x),len(y)); return instnorm(x[:n]).astype(np.float16), y[:n]
mtrain=[load_m(p) for p in pids if not held(p)]; mval=[load_m(p) for p in pids if held(p)]; mtrain_pids=[p for p in pids if not held(p)]
print(f"{NAME}: head {TRAIN_HEAD} lora {TRAIN_LORA} | real {len(real)} tracks, held-out {hold['name']} | minted {len(mtrain)}/{len(mval)}", flush=True)
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
def ar_layer(layer,x,cos_,sin_):
    q,k,v=layer.self_attn.project_qkv(layer.input_layernorm(x),cos_,sin_); h=nar_attention(q[0],k[0],v[0],causal=True)
    x=x+layer.self_attn.o_proj(h.flatten(1)[None]); return x+layer.mlp(layer.post_attention_layernorm(x)), k[0], v[0]
def nar_layer(layer,h,ak,av,ncos,nsin):
    q,k,v=layer.nar_self_attn.project_qkv(layer.nar_input_layernorm(h),ncos,nsin); a=nar_attention(q[0],torch.cat((ak,k[0])),torch.cat((av,v[0])))
    h=h+layer.nar_self_attn.o_proj(a.flatten(1)[None]); return h+layer.nar_mlp(layer.nar_pre_mlp_layernorm(h))
def flow_loss(prefix, codec_emb, x1, t, noise, grad_ar, grad_nar):
    pre=bb.embed_tokens(torch.tensor([prefix],device=dev))[0]; end=bb.embed_tokens(torch.tensor([MUSIC_END],device=dev)); x=torch.cat((pre,codec_emb.to(pre.dtype),end),0)[None]
    Lq=x.shape[1]; cos_,sin_=bb.rotary_emb(torch.arange(Lq,device=dev)[None]); cache=[]
    if grad_ar:
        for layer in bb.layers: x,k,v=checkpoint(ar_layer,layer,x,cos_,sin_,use_reentrant=False); cache.append((k,v))
    else:
        with torch.no_grad():
            for layer in bb.layers: x,k,v=ar_layer(layer,x,cos_,sin_); cache.append((k,v))
    T=codec_emb.shape[0]; xt=t*noise+(1-t)*x1; target=noise-x1; N=T+2; ncos,nsin=bb.rotary_emb(torch.arange(Lq,Lq+N,device=dev)[None])
    pe=model.latent_pos_embed(torch.arange(N,device=dev).clamp(max=model.config.max_latent_frames-1))[None]; sh=model._shift_t_value(float(np.clip(np.log(t/(1-t)),-20,20)),dev,torch.bfloat16)
    h=model.vae2llm(F.pad(xt,(0,0,1,1))[None].float()).to(torch.bfloat16)+model.time_embedder(sh.expand(N))[None]+pe
    for layer,(ak,av) in zip(bb.layers,cache): h=checkpoint(nar_layer,layer,h,ak,av,ncos,nsin,use_reentrant=False) if (grad_ar or grad_nar) else nar_layer(layer,h,ak,av,ncos,nsin)
    return F.mse_loss(model.llm2vae(bb.norm(h)[0,1:-1].float()),target)
def st_embed(logits):
    p=torch.softmax(logits.float(),-1); idx=p.argmax(-1); hard=F.one_hot(idx,VOCAB).float(); return ((hard+(p-p.detach())).to(Ecodec.dtype))@Ecodec, idx
def real_window(item,s=None):
    n=len(item["lat"]); s=random.randint(0,n-WIN) if s is None else s; return item["mert"][s:s+WIN], torch.tensor(item["lat"][s:s+WIN],device=dev)
@torch.no_grad()
def evaluate():
    head.eval(); n=len(hold["lat"]); tot=0; g=torch.Generator(device="cpu").manual_seed(123); reps=[]
    for s in (n//4,n//2,3*n//4):
        m,z=real_window(hold,s)
        with torch.autocast("cuda",dtype=torch.bfloat16): idx=head(torch.tensor(m[None],device=dev))[0].float().argmax(-1)
        reps.append(float((idx[1:]==idx[:-1]).float().mean())); noise=torch.randn(WIN,64,generator=g).to(dev)
        for t in (0.2,0.5,0.8): tot+=flow_loss(hold["prefix"],Ecodec[idx],z,t,noise,False,False).item()
    t1=tot_=0
    for _ in range(24):
        x,y=mbatch(mval,16)
        with torch.autocast("cuda",dtype=torch.bfloat16): lg=head(x)
        mk=y!=-100; t1+=(lg.float().argmax(-1)==y)[mk].sum().item(); tot_+=mk.sum().item()
    head.train(); return tot/9, t1/tot_, float(np.mean(reps))
def save_all(tag):
    torch.save({"model":head.state_dict(),"cfg":dict(NAME=NAME,instnorm=True)}, f"{OUT}/head_{tag}.pt")
    if TRAIN_LORA: save_lora(f"{OUT}/lora_{tag}.pt")
e0,a0,r0=evaluate(); msg=f"EVAL step 0 real_nar {e0:.4f} minted_top1 {a0:.4f} real_repeat {r0:.3f}"; print(msg, flush=True); log=open(f"{OUT}/train.log","a"); log.write(msg+"\n"); best=e0; save_all("best"); t0=time.time()
for st in range(1,STEPS+1):
    mult=min(1,st/50)*(0.2+0.8*0.5*(1+math.cos(math.pi*st/STEPS)))
    for g_,b in zip(opt.param_groups,base_lrs): g_["lr"]=b*mult
    if TRAIN_LORA and random.random()<MINTED_FLOW_P:   # minted regularizer for the NAR: true tokens, minted latents
        p=random.choice(mtrain_pids); d=f"{ROOT}/{p}"; r=json.load(open(f"{d}/request.json")); y=np.load(f"{d}/semantic.npy").astype(np.int64); z=np.load(f"{d}/latent.npy"); n=min(len(y),len(z)); s=random.randint(0,max(0,n-WIN))
        pre=token_prefixes(SongRequest(style=r["style"],lyrics=r["lyrics"],cot="off",seed=r["seed"],id=p),tok); t=float(np.clip(np.random.beta(2,2),0.02,0.98))
        ln=flow_loss(pre,Ecodec[torch.tensor(y[s:s+WIN],device=dev)],torch.tensor(z[s:s+WIN],device=dev),t,torch.randn(WIN,64,device=dev),False,True); lc=torch.zeros((),device=dev)
    else:
        item=random.choice(real); m,z=real_window(item); t=float(np.clip(np.random.beta(2,2),0.05,0.95)); noise=torch.randn(WIN,64,device=dev)
        with torch.autocast("cuda",dtype=torch.bfloat16): lg=head(torch.tensor(m[None],device=dev))[0]
        emb,idx=st_embed(lg) if TRAIN_HEAD else (Ecodec[lg.float().argmax(-1)],None)
        ln=flow_loss(item["prefix"],emb,z,t,noise,bool(TRAIN_HEAD),bool(TRAIN_LORA))
        if TRAIN_HEAD:
            x,y=mbatch(mtrain,MB)
            with torch.autocast("cuda",dtype=torch.bfloat16): lgm=head(x)
            lc=soft_ce(lgm,y)
        else: lc=torch.zeros((),device=dev)
    loss=ln+lc; opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_([p for g_ in opt.param_groups for p in g_["params"]],1.0); opt.step()
    if st<=3 or st%25==0: print(f"step {st} nar {ln.item():.4f} ce {lc.item():.3f} {time.time()-t0:.0f}s mem {torch.cuda.max_memory_allocated()/2**30:.1f}G", flush=True)
    if st%100==0 or st==STEPS:
        e,a,rp=evaluate(); msg=f"EVAL step {st} real_nar {e:.4f} minted_top1 {a:.4f} real_repeat {rp:.3f} {time.time()-t0:.0f}s"; print(msg, flush=True); log.write(msg+"\n"); log.flush()
        if e<best: best=e; save_all("best")
        save_all("last")
print(f"RESULT {NAME}: best real_nar {best:.4f} (start {e0:.4f})", flush=True)
# ---- render held-out with best
head.load_state_dict(torch.load(f"{OUT}/head_best.pt",map_location=dev)["model"]); head.eval()
if TRAIN_LORA: load_lora(f"{OUT}/lora_best.pt")
model.vae2llm.to(torch.bfloat16); model.llm2vae.to(torch.bfloat16)
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
toks=predict(hold["mert"]); print(f"held-out tokens: unique {len(set(toks.tolist()))/len(toks):.2f} repeat {float((toks[1:]==toks[:-1]).mean()):.4f}", flush=True)
with torch.inference_mode(): z=synthesize(model, hold["prefix"], [int(v) for v in toks], 4242, steps=32).float().cpu()
vae=YuE2VAE.from_pretrained(vsnap, decoder_only=True, device=dev, local_files_only=True)
with torch.inference_mode(): audio=vae.decode_tiled(z.T[None].contiguous(), core_frames=750, halo_frames=16, output_device="cpu")
sf.write(f"{W}/listen_real/real_pred_{NAME}.flac", audio[0].float().clamp(-1,1).T.numpy(), 48000, subtype="PCM_24"); print("RENDER DONE", flush=True)
