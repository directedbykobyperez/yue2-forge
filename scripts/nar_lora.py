"""NAR LoRA on REAL latents: adapt YuE2's NAR branch (nar_self_attn q/k/v/o + nar_mlp gate/up/down, rank R) + full vae2llm/llm2vae
to real artist audio, conditioned on OUR head's tokens (frozen head, precomputed). AR branch frozen (prefix cache under no_grad).
25% of steps use minted windows (true tokens + minted latents) as a regularizer. Eval = held-out real track flow loss (fixed windows/t/noise).
At the end: render the held-out track with the LoRA'd NAR from head tokens. usage: nar_lora.py <name> <steps> <rank> <head_ckpt>"""
import os, sys, glob, json, math, time, random, hashlib, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, soundfile as sf
from torch.utils.checkpoint import checkpoint
os.environ.setdefault("HF_HOME","/workspace/hf"); torch.backends.cuda.matmul.allow_tf32=True
from yue2.modeling_yue2 import YuE2ForCausalLM
from yue2.modeling_vae import YuE2VAE
from yue2.protocol import CODEC_OFFSET, MUSIC_END, SongRequest, token_prefixes
from yue2.tokenization_yue2 import YuE2TextTokenizer
from yue2.nar import attention as nar_attention, synthesize
NAME=sys.argv[1]; STEPS=int(sys.argv[2]); R=int(sys.argv[3]); HEAD_CK=sys.argv[4]; HOLD=os.environ.get("HOLD_TRACK","")  # held-out track name for the real-audio metric; defaults to the first track
W="/workspace/tok/full"; ROOT="/workspace/yue2-corpus/tracks"; RP="/workspace/real/prep"; OUT=f"{W}/{NAME}"; os.makedirs(OUT,exist_ok=True); dev="cuda"
VOCAB=32768; WIN=768; HWIN=512; D=512; L=8; H=8; LR=1e-4; LR_IO=3e-5; ACC=2; MINTED_P=0.25
snap=glob.glob("/workspace/hf/hub/models--m-a-p--YuE2-3B/snapshots/*")[0]; vsnap=glob.glob("/workspace/hf/hub/models--m-a-p--YuE2-Vae/snapshots/*")[0]
model=YuE2ForCausalLM.from_pretrained(snap, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval().to(dev); model.requires_grad_(False); bb=model.model
tok=YuE2TextTokenizer(snap+"/qwen.tiktoken")
# ---- LoRA
class LoRALinear(nn.Module):
    def __init__(s, base, r, alpha=None):
        super().__init__(); s.base=base; s.r=r; s.scale=(alpha or r)/r
        s.A=nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device)*(1/math.sqrt(base.in_features))); s.B=nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
    def forward(s,x): return s.base(x)+((x.float()@s.A.T)@s.B.T).to(x.dtype)*s.scale
lora_params=[]
for layer in bb.layers:
    for mod,names in ((layer.nar_self_attn,("q_proj","k_proj","v_proj","o_proj")),(layer.nar_mlp,("gate_proj","up_proj","down_proj"))):
        for n in names: l=LoRALinear(getattr(mod,n),R); setattr(mod,n,l); lora_params+=[l.A,l.B]
io_params=[]
for m in (model.vae2llm, model.llm2vae):
    m.float(); m.requires_grad_(True); io_params+=list(m.parameters())
print(f"LoRA params {sum(p.numel() for p in lora_params)/1e6:.1f}M + io {sum(p.numel() for p in io_params)/1e6:.2f}M", flush=True)
opt=torch.optim.AdamW([{"params":lora_params,"lr":LR},{"params":io_params,"lr":LR_IO}],weight_decay=0.0,betas=(0.9,0.95))
# ---- frozen head -> tokens for real tracks (cached)
class Tok(nn.Module):
    def __init__(s, din):
        super().__init__(); s.inp=nn.Linear(din,D); s.pos=nn.Parameter(torch.zeros(1,HWIN,D))
        layer=nn.TransformerEncoderLayer(D,H,4*D,dropout=0.1,batch_first=True,norm_first=True,activation="gelu"); s.enc=nn.TransformerEncoder(layer,L); s.norm=nn.LayerNorm(D); s.head=nn.Linear(D,VOCAB)
    def forward(s,x): return s.head(s.norm(s.enc(s.inp(x)+s.pos[:,:x.shape[1]])))
head=Tok(1024).to(dev).eval(); head.load_state_dict(torch.load(HEAD_CK,map_location=dev)["model"])
def instnorm(x): x=x.astype(np.float32); return (x-x.mean(0))/(x.std(0)+1e-5)
@torch.no_grad()
def predict(x):
    T=len(x); out=np.zeros(T,dtype=np.int64); starts=list(range(0,max(1,T-HWIN+1),HWIN//2))
    if starts[-1]+HWIN<T: starts.append(max(0,T-HWIN))
    for s0 in starts:
        xw=x[s0:s0+HWIN]; n=len(xw)
        if n<HWIN: xw=np.pad(xw,((0,HWIN-n),(0,0)))
        with torch.autocast("cuda",dtype=torch.bfloat16): pred=head(torch.tensor(xw[None],device=dev))[0,:n].float().argmax(-1).cpu().numpy()
        lo=s0+(0 if s0==0 else HWIN//4); hi=s0+n-(0 if s0+n>=T else HWIN//4); out[lo:hi]=pred[lo-s0:hi-s0]
    return out
htag=hashlib.md5(HEAD_CK.encode()).hexdigest()[:6]; real=[]; hold=None
for d in sorted(glob.glob(f"{RP}/*")):
    tf=f"{d}/tokens_{htag}.npy"
    if not os.path.exists(tf): np.save(tf, predict(instnorm(np.load(f"{d}/mert.npy"))).astype(np.int32))
    item=dict(name=os.path.basename(d), lat=np.load(f"{d}/lat.npy"), tokens=np.load(tf).astype(np.int64), prefix=[int(v) for v in np.load(f"{d}/prefix.npy")])
    n=min(len(item["lat"]),len(item["tokens"])); item["lat"]=item["lat"][:n]; item["tokens"]=item["tokens"][:n]
    if item["name"]==HOLD: hold=item
    else: real.append(item)
if hold is None: hold=real.pop(0); print("HOLD_TRACK not set; holding out", hold["name"], flush=True)
print(f"real train {len(real)} tracks, held-out {hold['name']}", flush=True)
held=lambda p: int(hashlib.md5(p.encode()).hexdigest(),16)%20==0
mpids=[os.path.basename(f)[:-4] for f in sorted(glob.glob(f"{W}/feats/*.npy")) if not held(os.path.basename(f)[:-4])]
def minted_item(p):
    d=f"{ROOT}/{p}"; r=json.load(open(f"{d}/request.json")); y=np.load(f"{d}/semantic.npy").astype(np.int64); z=np.load(f"{d}/latent.npy"); n=min(len(y),len(z))
    return dict(name=p, lat=z[:n], tokens=y[:n], prefix=token_prefixes(SongRequest(style=r["style"],lyrics=r["lyrics"],cot="off",seed=r["seed"],id=p),tok))
# ---- NAR forward with trainable NAR branch; AR prefill under no_grad
def ar_layer(layer,x,cos_,sin_):
    q,k,v=layer.self_attn.project_qkv(layer.input_layernorm(x),cos_,sin_); h=nar_attention(q[0],k[0],v[0],causal=True)
    x=x+layer.self_attn.o_proj(h.flatten(1)[None]); return x+layer.mlp(layer.post_attention_layernorm(x)), k[0], v[0]
def nar_layer(layer,h,ak,av,ncos,nsin):
    q,k,v=layer.nar_self_attn.project_qkv(layer.nar_input_layernorm(h),ncos,nsin); a=nar_attention(q[0],torch.cat((ak,k[0])),torch.cat((av,v[0])))
    h=h+layer.nar_self_attn.o_proj(a.flatten(1)[None]); return h+layer.nar_mlp(layer.nar_pre_mlp_layernorm(h))
def flow_loss(prefix, codec, x1, t, noise, grad=True):
    with torch.no_grad():
        ar=torch.tensor([prefix+[int(c)+CODEC_OFFSET for c in codec]+[MUSIC_END]],device=dev); Lq=ar.shape[1]; cos_,sin_=bb.rotary_emb(torch.arange(Lq,device=dev)[None]); x=bb.embed_tokens(ar); cache=[]
        for layer in bb.layers: x,k,v=ar_layer(layer,x,cos_,sin_); cache.append((k,v))
    T=len(codec); xt=t*noise+(1-t)*x1; target=noise-x1; N=T+2; ncos,nsin=bb.rotary_emb(torch.arange(Lq,Lq+N,device=dev)[None])
    pe=model.latent_pos_embed(torch.arange(N,device=dev).clamp(max=model.config.max_latent_frames-1))[None]; sh=model._shift_t_value(float(np.clip(np.log(t/(1-t)),-20,20)),dev,torch.bfloat16)
    h=model.vae2llm(F.pad(xt,(0,0,1,1))[None].float()).to(torch.bfloat16)+model.time_embedder(sh.expand(N))[None]+pe
    for layer,(ak,av) in zip(bb.layers,cache): h=checkpoint(nar_layer,layer,h,ak,av,ncos,nsin,use_reentrant=False) if grad else nar_layer(layer,h,ak,av,ncos,nsin)
    return F.mse_loss(model.llm2vae(bb.norm(h)[0,1:-1].float()),target)
def window(item, s=None, win=WIN):
    n=len(item["lat"]); s=random.randint(0,max(0,n-win)) if s is None else s; return item["tokens"][s:s+win], torch.tensor(item["lat"][s:s+win],device=dev)
@torch.no_grad()
def evaluate():
    n=len(hold["lat"]); tot=0; g=torch.Generator(device="cpu").manual_seed(123)
    for s in (n//4,n//2,3*n//4):
        c,z=window(hold,s,512); noise=torch.randn(512,64,generator=g).to(dev)
        for t in (0.2,0.5,0.8): tot+=flow_loss(hold["prefix"],c,z,t,noise,grad=False).item()
    return tot/9
def save(path): torch.save({"lora":[p.detach().cpu() for p in lora_params],"io":{"vae2llm":model.vae2llm.state_dict(),"llm2vae":model.llm2vae.state_dict()},"rank":R,"head":HEAD_CK}, path)
e0=evaluate(); print(f"EVAL step 0 real_nar {e0:.4f}", flush=True); best=e0; log=open(f"{OUT}/train.log","a"); log.write(f"EVAL step 0 real_nar {e0:.4f}\n"); t0=time.time()
for st in range(1,STEPS+1):
    lr_mult=min(1,st/50)*(0.2+0.8*0.5*(1+math.cos(math.pi*st/STEPS)))
    for gp,base in zip(opt.param_groups,(LR,LR_IO)): gp["lr"]=base*lr_mult
    for _ in range(ACC):
        item=minted_item(random.choice(mpids)) if random.random()<MINTED_P else random.choice(real)
        c,z=window(item); t=float(np.clip(np.random.beta(2,2),0.02,0.98)); noise=torch.randn(len(c),64,device=dev)
        loss=flow_loss(item["prefix"],c,z,t,noise)/ACC; loss.backward()
    torch.nn.utils.clip_grad_norm_(lora_params+io_params,1.0); opt.step(); opt.zero_grad(set_to_none=True)
    if st<=3 or st%25==0: print(f"step {st} loss {loss.item()*ACC:.4f} {time.time()-t0:.0f}s mem {torch.cuda.max_memory_allocated()/2**30:.1f}G", flush=True)
    if st%100==0 or st==STEPS:
        e=evaluate(); msg=f"EVAL step {st} real_nar {e:.4f} {time.time()-t0:.0f}s"; print(msg, flush=True); log.write(msg+"\n"); log.flush()
        if e<best: best=e; save(f"{OUT}/best.pt")
        save(f"{OUT}/last.pt")
print(f"RESULT {NAME}: best real_nar {best:.4f} (start {e0:.4f})", flush=True)
# ---- render held-out with best LoRA
ck=torch.load(f"{OUT}/best.pt",map_location=dev)
with torch.no_grad():
    for p,v in zip(lora_params,ck["lora"]): p.copy_(v.to(dev))
    model.vae2llm.load_state_dict(ck["io"]["vae2llm"]); model.llm2vae.load_state_dict(ck["io"]["llm2vae"])
model.vae2llm.to(torch.bfloat16); model.llm2vae.to(torch.bfloat16)
with torch.inference_mode(): z=synthesize(model, hold["prefix"], [int(v) for v in hold["tokens"]], 4242, steps=32).float().cpu()
vae=YuE2VAE.from_pretrained(vsnap, decoder_only=True, device=dev, local_files_only=True)
with torch.inference_mode(): audio=vae.decode_tiled(z.T[None].contiguous(), core_frames=750, halo_frames=16, output_device="cpu")
sf.write(f"{W}/listen_real/real_pred_{NAME}.flac", audio[0].float().clamp(-1,1).T.numpy(), 48000, subtype="PCM_24"); print("RENDER DONE", flush=True)
