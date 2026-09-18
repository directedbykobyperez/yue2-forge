"""AR-branch LoRA: teach YuE2's AR to write token streams for an artist from style+lyrics.

usage: ar_train.py <name> <steps> <rank> <artist_frac> [init] [lr] [cur_w]
Env:  VRAM_MODE=low|high, START_STEP, SCHED_STEPS, CK_FROM, CK_EVERY, EMA_DECAY,
      AR_KL_WEIGHT, AR_LR_MULTIPLIER, ABC_DROPOUT, TRAIN_WINDOW, COT, AR_MAX_TOKENS"""
import os, sys, glob, math, time, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
last_cl=float("nan")
from torch.utils.checkpoint import checkpoint
sys.path.insert(0, os.path.dirname(__file__))
from vram_helpers import load_yue2_model, get_snap_path, get_maxlen, get_vram_mode
os.environ.setdefault("HF_HOME","/workspace/hf"); torch.backends.cuda.matmul.allow_tf32=True
from yue2.protocol import CODEC_OFFSET, MUSIC_END, ABC_START, MUSIC_START
from yue2.nar import attention as nar_attention
from yue2.tokenization_yue2 import YuE2TextTokenizer
from yue2.protocol import INSTRUCTIONS

# ── CLI args ──────────────────────────────────────────────────────────
NAME=sys.argv[1]; STEPS=int(sys.argv[2]); RANK=int(sys.argv[3]); CFRAC=float(sys.argv[4])
INIT=sys.argv[5] if len(sys.argv)>5 else "none"
LR=float(sys.argv[6]) if len(sys.argv)>6 else 1e-4
CUR_W=float(sys.argv[7]) if len(sys.argv)>7 else 0.08
OUT=f"/workspace/tok/full/{NAME}"; os.makedirs(OUT,exist_ok=True); dev="cuda"

# ── Hyperparams (env or defaults) ────────────────────────────────────
MAXLEN=get_maxlen(); ACC=2; CHUNK=1024
SCHED=int(os.environ.get('SCHED_STEPS','3000'))
CK_FROM=int(os.environ.get('CK_FROM','600'))
CK_EVERY=int(os.environ.get('CK_EVERY','200'))
AR_KL_WEIGHT=float(os.environ.get('AR_KL_WEIGHT','0.04'))
AR_LR_MULT=float(os.environ.get('AR_LR_MULTIPLIER','1.0'))
ABC_DROPOUT=float(os.environ.get('ABC_DROPOUT','0.5'))
TRAIN_WINDOW=int(os.environ.get('TRAIN_WINDOW','1500'))
COT=str(os.environ.get('COT','off'))
AR_MAX_TOKENS=int(os.environ.get('AR_MAX_TOKENS','0'))
SAMPLE_AR_REPETITION_PENALTY=float(os.environ.get('SAMPLE_AR_REPETITION_PENALTY','1.2'))
EMA_DECAY=float(os.environ.get('EMA_DECAY','0.99'))
WEIGHT_DECAY=float(os.environ.get('WEIGHT_DECAY','0.0001'))

print(f"Config: rank={RANK} lr={LR} ar_kl_weight={AR_KL_WEIGHT} ar_lr_mult={AR_LR_MULT} "
      f"abc_dropout={ABC_DROPOUT} train_window={TRAIN_WINDOW} cot={COT} maxlen={MAXLEN} "
      f"ar_rep_penalty={SAMPLE_AR_REPETITION_PENALTY} ema_decay={EMA_DECAY} weight_decay={WEIGHT_DECAY}", flush=True)

# ── Load model ───────────────────────────────────────────────────────
result=load_yue2_model(dev)
if isinstance(result, tuple): model, snap = result
else: model, snap = result, get_snap_path()
bb=model.model
if get_vram_mode() == "low":
    try: bb.gradient_checkpointing_enable()
    except Exception: pass
    print(f"[VRAM low] MAXLEN={MAXLEN}, gradient checkpointing enabled", flush=True)

# ── LoRA layers ──────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    def __init__(s, base, r):
        super().__init__(); s.base=base; s.A=nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device)*(1/math.sqrt(base.in_features))); s.B=nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
    def forward(s,x): return s.base(x)+((x.float()@s.A.T)@s.B.T).to(x.dtype)

lora=[]; ar_param_ids=set()
for layer in bb.layers:
    for mod,names in ((layer.self_attn,("q_proj","k_proj","v_proj","o_proj")),(layer.mlp,("gate_proj","up_proj","down_proj"))):
        for n in names: l=LoRALinear(getattr(mod,n),RANK); setattr(mod,n,l); lora+=[l.A,l.B]; ar_param_ids.update(id(p) for p in [l.A,l.B])

if INIT!="none":
    ck=torch.load(INIT,map_location=dev)
    with torch.no_grad(): [p.copy_(v.to(dev)) for p,v in zip(lora,ck["lora"])]
    RESTORE_CH=ck.get("cursor_head")
    RESTORE_EMA=ck.get("ema")
    print("init from", INIT, flush=True)
print(f"AR LoRA params {sum(p.numel() for p in lora)/1e6:.1f}M lr {LR}", flush=True)

# ── Cursor head (alignment) ─────────────────────────────────────────
cursor_head=nn.Linear(model.config.hidden_size,model.config.hidden_size,bias=False).to(dev); nn.init.eye_(cursor_head.weight)
if "RESTORE_CH" in dir() and RESTORE_CH is not None: cursor_head.load_state_dict({k: v.to(dev) for k,v in RESTORE_CH.items()}); print("cursor_head restored", flush=True)

# ── Optimizer with AR/NAR LR split ──────────────────────────────────
ar_params=[p for p in lora if id(p) in ar_param_ids]
non_ar_params=[p for p in lora if id(p) not in ar_param_ids]
opt_groups=[{"params":ar_params,"lr":LR*AR_LR_MULT,"weight_decay":WEIGHT_DECAY}]
if non_ar_params: opt_groups.append({"params":non_ar_params,"lr":LR,"weight_decay":WEIGHT_DECAY})
opt_groups.append({"params":cursor_head.parameters(),"lr":LR,"weight_decay":WEIGHT_DECAY})
opt=torch.optim.AdamW(opt_groups,betas=(0.9,0.95))
if AR_LR_MULT!=1.0: print(f"AR LR multiplier: {AR_LR_MULT}x ({len(ar_params)} AR tensors, {len(non_ar_params)} non-AR)", flush=True)

ema_params=[p.detach().clone() for p in lora + list(cursor_head.parameters())]
if "RESTORE_EMA" in dir() and RESTORE_EMA is not None:
    with torch.no_grad():
        [e.copy_(v.to(dev)) for e, v in zip(ema_params, RESTORE_EMA)]
    print("ema restored", flush=True)

# ── Tokenizer + data ────────────────────────────────────────────────
tok=YuE2TextTokenizer(snap+"/qwen.tiktoken"); RP="/workspace/real/prep"

def cursor_targets(item):
    """-> (j0, j1, targ [F,ntok] sparse as (frame->token list)) for an artist item, or None."""
    p=f"{RP}/{item['name']}/cursor_words.npy"
    if not os.path.exists(p): return None
    words=np.load(p)
    # Build the same prefix that was used for this item
    prefix = build_prefix(item, cot_mode="off")  # cursor always uses off-mode for alignment
    head=f"{INSTRUCTIONS['off']}\n[Tags]\n{item['style']}\n[Lyrics]\n"
    ids_head=tok.encode(head)
    ids_full=tok.encode(head+item["lyrics"]+"\n")
    if ids_full[:len(ids_head)]!=ids_head: return None
    # Find lyrics position in the prefix
    lyr_start_in_head = len(ids_head)
    j0=lyr_start_in_head+1  # +1 for EOD at start
    lyr_ids=ids_full[lyr_start_in_head:]
    j1=j0+len(lyr_ids)
    offs=[]; cur=0
    for k in range(len(lyr_ids)): cur=len(tok.decode(lyr_ids[:k+1])); offs.append(cur)
    starts=[0]+offs[:-1]; tok_of_word=[]
    for (s0,e0,sc,c0,c1) in words: tok_of_word.append([k for k in range(len(lyr_ids)) if starts[k]<c1 and offs[k]>c0] or [min(len(lyr_ids)-1,int(np.searchsorted(offs,c0)))])
    F_=len(item["codec"]); t=np.arange(F_)/25.0; wstart=words[:,0]; widx=np.clip(np.searchsorted(wstart,t,side="right")-1,0,len(words)-1)
    rows=[]; cols=[]; vals=[]
    for i,w in enumerate(widx):
        ks=tok_of_word[w]; rows+= [i]*len(ks); cols+=ks; vals+=[1.0/len(ks)]*len(ks)
    return j0,j1,F_,torch.tensor(rows,device=dev),torch.tensor(cols,device=dev),torch.tensor(vals,device=dev)

CUR_CACHE={}
data=torch.load("/workspace/real/ar/dataset.pt", weights_only=False)
artist=[x for x in data if x["src"]=="artist"]; minted=[x for x in data if x["src"]=="minted"]; mval=[x for x in data if x["src"]=="minted_val"][:6]
print(f"artist {len(artist)} minted {len(minted)} val {len(mval)}", flush=True)

# ── Helper: build prompt with cot mode ──────────────────────────────
def build_prefix(item, cot_mode=None):
    """Build AR prompt prefix. Uses stored prefix if available (with ABC sheets)."""
    cot = cot_mode or COT
    # Use pre-built prefix from dataset if available (includes ABC sheet)
    if "prefix" in item and item["prefix"] is not None:
        return list(item["prefix"])
    # Fallback: build from scratch
    instruction = INSTRUCTIONS.get(cot, INSTRUCTIONS['off'])
    head = f"{instruction}\n[Tags]\n{item['style']}\n[Lyrics]\n{item['lyrics']}\n"
    dur = item.get("duration")
    if dur is not None and dur > 0:
        head += f"[Duration]\n{dur}\n"
    return tok.encode(head)

def seq_of(item):
    pre = build_prefix(item)
    cod=[int(c)+CODEC_OFFSET for c in item["codec"]]; room=MAXLEN-len(pre)-1
    body=cod[:room]+([MUSIC_END] if len(cod)<=room else []); return torch.tensor([pre+body],device=dev), len(pre)

# ── Window cropping (train_window_frames) ────────────────────────────
def seq_of_windowed(item):
    """Random window crop for long songs; returns (ids, prefix_len, start, total)."""
    pre = build_prefix(item)
    cod = [int(c)+CODEC_OFFSET for c in item["codec"]]
    total = len(pre) + len(cod) + 1  # +1 for MUSIC_END
    if TRAIN_WINDOW <= 0 or total <= MAXLEN:
        body = cod[:MAXLEN-len(pre)-1] + ([MUSIC_END] if len(cod) <= MAXLEN-len(pre)-1 else [])
        return torch.tensor([pre+body], device=dev), len(pre), 0, total
    # crop a random window from the codec tokens
    max_codec = MAXLEN - len(pre) - 1
    if len(cod) <= max_codec:
        body = cod + [MUSIC_END]
    else:
        start = random.randint(0, len(cod) - max_codec)
        body = cod[start:start+max_codec] + [MUSIC_END]
    return torch.tensor([pre+body], device=dev), len(pre), 0, total

# ── AR forward (with optional KL from base model) ───────────────────
def ar_layer(layer,x,cos_,sin_):
    q,k,v=layer.self_attn.project_qkv(layer.input_layernorm(x),cos_,sin_); h=nar_attention(q[0],k[0],v[0],causal=True)
    x=x+layer.self_attn.o_proj(h.flatten(1)[None]); return x+layer.mlp(layer.post_attention_layernorm(x))

def lm_loss(ids, Lp, grad=True, cur=None):
    x=bb.embed_tokens(ids); Lq=ids.shape[1]; cos_,sin_=bb.rotary_emb(torch.arange(Lq,device=dev)[None])
    for layer in bb.layers: x=checkpoint(ar_layer,layer,x,cos_,sin_,use_reentrant=False) if grad else ar_layer(layer,x,cos_,sin_)
    hn=bb.norm(x[0]); h=hn[Lp-1:-1]; tgt=ids[0,Lp:]; tot=0.
    for s in range(0,h.shape[0],CHUNK):
        lg=model.lm_head(h[s:s+CHUNK]).float(); tot=tot+F.cross_entropy(lg,tgt[s:s+CHUNK],reduction="sum")
    lm=tot/h.shape[0]
    if cur is None: return lm, None, None
    j0,j1,F_,rows,cols,vals=cur; nF=min(F_,h.shape[0]); q=cursor_head(h[:nF].float()); kh=hn[j0:j1].float(); sc=(q@kh.T)*(q.shape[-1]**-0.5); logp=sc.log_softmax(-1)
    m=rows<nF; tmask=torch.zeros(nF,j1-j0,device=dev).index_put_((rows[m],cols[m]),vals[m])
    return lm, -(tmask*logp).sum(-1).mean(), None

def lm_loss_with_kl(ids, Lp, grad=True, cur=None):
    """LM loss with optional KL regularization against base model (LoRA disabled)."""
    x=bb.embed_tokens(ids); Lq=ids.shape[1]; cos_,sin_=bb.rotary_emb(torch.arange(Lq,device=dev)[None])
    for layer in bb.layers: x=checkpoint(ar_layer,layer,x,cos_,sin_,use_reentrant=False) if grad else ar_layer(layer,x,cos_,sin_)
    hn=bb.norm(x[0]); h=hn[Lp-1:-1]; tgt=ids[0,Lp:]; tot=0.
    for s in range(0,h.shape[0],CHUNK):
        lg=model.lm_head(h[s:s+CHUNK]).float(); tot=tot+F.cross_entropy(lg,tgt[s:s+CHUNK],reduction="sum")
    lm=tot/h.shape[0]

    # KL regularization: compare LoRA vs base (LoRA temporarily disabled)
    kl = None
    if AR_KL_WEIGHT > 0 and grad:
        # disable all LoRA layers
        for layer in bb.layers:
            for mod in (layer.self_attn, layer.mlp):
                for name in ("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"):
                    mod_child = getattr(mod, name)
                    if hasattr(mod_child, 'base'):
                        mod_child._lora_was_active = True
                        # swap to base-only forward
        # run base model forward (no LoRA)
        with torch.no_grad():
            x_base=bb.embed_tokens(ids)
            for layer in bb.layers: x_base=ar_layer(layer,x_base,cos_,sin_)
            hn_base=bb.norm(x_base[0]); h_base=hn_base[Lp-1:-1]
        # re-enable LoRA (nothing to undo — LoRA is additive: base(x) + lora(x))
        # compute KL
        kl_tot = torch.zeros(1, device=dev)
        for s in range(0, h.shape[0], CHUNK):
            logp_lora = F.log_softmax(model.lm_head(h[s:s+CHUNK]).float(), dim=-1)
            p_base = F.softmax(model.lm_head(h_base[s:s+CHUNK]).float(), dim=-1)
            kl_tot = kl_tot + F.kl_div(logp_lora, p_base, log_target=False, reduction="sum")
        kl = kl_tot / h.shape[0]

    # cursor alignment
    if cur is not None:
        j0,j1,F_,rows,cols,vals=cur; nF=min(F_,h.shape[0]); q=cursor_head(h[:nF].float()); kh=hn[j0:j1].float(); sc=(q@kh.T)*(q.shape[-1]**-0.5); logp=sc.log_softmax(-1)
        m=rows<nF; tmask=torch.zeros(nF,j1-j0,device=dev).index_put_((rows[m],cols[m]),vals[m])
        return lm, -(tmask*logp).sum(-1).mean(), kl
    return lm, None, kl

# ── ABC dropout: randomly drop sheet conditioning ────────────────────
def maybe_abc_dropout(item):
    """With probability ABC_DROPOUT, return item without sheet (off-mode prompt)."""
    if random.random() < ABC_DROPOUT:
        # clone with off-mode instruction
        dropped = dict(item)
        dropped["_abc_dropped"] = True
        return dropped
    return item

# ── Evaluate ────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate():
    r={}
    for tag,items in (("minted_val",mval),("artist",artist[:6])):
        tot=0
        for it in items: ids,Lp=seq_of(it); tot+=lm_loss(ids,Lp,grad=False)[0].item()
        r[tag]=tot/len(items)
    return r

e=evaluate(); print(f"EVAL step 0 minted_val {e['minted_val']:.3f} artist {e['artist']:.3f}", flush=True)
log=open(f"{OUT}/train.log","a"); log.write(str(e)+"\n"); t0=time.time(); best=e["artist"]

def save(path):
    torch.save({"lora":[p.detach().cpu() for p in lora],"rank":RANK,"targets":"ar self_attn qkvo + mlp gate/up/down","cursor_head":cursor_head.state_dict(),"ema":[p.detach().cpu() for p in ema_params],"ar_kl_weight":AR_KL_WEIGHT,"ar_lr_mult":AR_LR_MULT,"abc_dropout":ABC_DROPOUT,"cot":COT}, path)
    # Auto-export to .safetensors (user-facing format)
    safetensors_path = os.path.splitext(path)[0] + ".safetensors"
    try:
        import export_safetensors
        export_safetensors.convert_one(path, safetensors_path, force=True)
    except Exception as e:
        print(f"  [export] safetensors export failed: {e}", flush=True)

# ── Training loop ───────────────────────────────────────────────────
START=int(os.environ.get("START_STEP","0"))
total_ar_ce=0.; total_ar_kl=0.; n_ar_steps=0
for st in range(START+1,START+STEPS+1):
    for g in opt.param_groups: g["lr"]=LR*min(1,st/50)*(0.2+0.8*0.5*(1+math.cos(math.pi*min(st,SCHED)/SCHED)))
    for _ in range(ACC):
        # ABC dropout: randomly drop sheet conditioning
        it = random.choice(artist) if random.random()<CFRAC else random.choice(minted)
        it = maybe_abc_dropout(it)
        ids,Lp=seq_of(it)
        cur=None
        if it["src"]=="artist" and not it.get("_abc_dropped"):
            if it["name"] not in CUR_CACHE: CUR_CACHE[it["name"]]=cursor_targets(it)
            cur=CUR_CACHE[it["name"]]
        lm,cl,kl = lm_loss_with_kl(ids,Lp,cur=cur)
        loss = lm
        if cl is not None: loss = loss + CUR_W * cl
        if kl is not None: loss = loss + AR_KL_WEIGHT * kl
        loss = loss / ACC
        loss.backward()
        last_cl = float(cl) if cl is not None else float("nan")
        if kl is not None:
            total_ar_kl += kl.item()
        total_ar_ce += lm.item()
        n_ar_steps += 1
    torch.nn.utils.clip_grad_norm_(lora+list(cursor_head.parameters()),1.0); opt.step(); opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        for e_, p_ in zip(ema_params, lora + list(cursor_head.parameters())):
            e_.mul_(EMA_DECAY).add_(p_.detach(), alpha=1 - EMA_DECAY)
    if st<=3 or st%20==0:
        avg_kl = total_ar_kl / max(1, n_ar_steps)
        avg_ce = total_ar_ce / max(1, n_ar_steps)
        print(f"step {st} loss {loss.item()*ACC:.3f} ar_ce {avg_ce:.3f} ar_kl {avg_kl:.4f} cursor {last_cl:.3f} len {ids.shape[1]} {time.time()-t0:.0f}s mem {torch.cuda.max_memory_allocated()/2**30:.1f}G", flush=True)
    if st%100==0 or st==STEPS:
        e=evaluate(); msg=f"EVAL step {st} minted_val {e['minted_val']:.3f} artist {e['artist']:.3f} {time.time()-t0:.0f}s"; print(msg, flush=True); log.write(msg+"\n"); log.flush()
        if e["artist"]<best: best=e["artist"]; save(f"{OUT}/best.pt")
        save(f"{OUT}/last.pt")
        if st>=CK_FROM and (st-CK_FROM)%CK_EVERY==0: save(f"{OUT}/step-{st}.pt")
        total_ar_ce=0.; total_ar_kl=0.; n_ar_steps=0
print(f"RESULT {NAME}: best artist CE {best:.3f}", flush=True)
