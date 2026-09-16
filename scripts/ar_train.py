"""AR-branch LoRA: teach YuE2's AR to write token streams (joint_v1 head dialect) for an artist from style+lyrics (cot=off), regularized with minted songs.
usage: ar_lora.py <name> <steps> <rank> <artist_frac>"""
import os, sys, glob, math, time, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
last_cl=float("nan")
from torch.utils.checkpoint import checkpoint
os.environ.setdefault("HF_HOME","/workspace/hf"); torch.backends.cuda.matmul.allow_tf32=True
from yue2.modeling_yue2 import YuE2ForCausalLM
from yue2.protocol import CODEC_OFFSET, MUSIC_END
from yue2.nar import attention as nar_attention
from yue2.tokenization_yue2 import YuE2TextTokenizer
from yue2.protocol import INSTRUCTIONS
NAME=sys.argv[1]; STEPS=int(sys.argv[2]); RANK=int(sys.argv[3]); CFRAC=float(sys.argv[4]); INIT=sys.argv[5] if len(sys.argv)>5 else "none"; LR=float(sys.argv[6]) if len(sys.argv)>6 else 1e-4; CUR_W=float(sys.argv[7]) if len(sys.argv)>7 else 0.08; OUT=f"/workspace/tok/full/{NAME}"; os.makedirs(OUT,exist_ok=True); dev="cuda"
MAXLEN=12288; ACC=2; CHUNK=1024; SCHED=int(os.environ.get('SCHED_STEPS','3000')); CK_FROM=int(os.environ.get('CK_FROM','600')); CK_EVERY=int(os.environ.get('CK_EVERY','200'))
snap=glob.glob("/workspace/hf/hub/models--m-a-p--YuE2-3B/snapshots/*")[0]
model=YuE2ForCausalLM.from_pretrained(snap, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval().to(dev); model.requires_grad_(False); bb=model.model
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
    RESTORE_CH=ck.get("cursor_head")
    RESTORE_EMA=ck.get("ema")
    print("init from", INIT, flush=True)
print(f"AR LoRA params {sum(p.numel() for p in lora)/1e6:.1f}M lr {LR}", flush=True)
cursor_head=nn.Linear(model.config.hidden_size,model.config.hidden_size,bias=False).to(dev); nn.init.eye_(cursor_head.weight)
if "RESTORE_CH" in dir() and RESTORE_CH is not None: cursor_head.load_state_dict({k: v.to(dev) for k,v in RESTORE_CH.items()}); print("cursor_head restored", flush=True)
opt=torch.optim.AdamW([{"params":lora,"lr":LR,"weight_decay":0.0},{"params":cursor_head.parameters(),"lr":LR,"weight_decay":0.0}],betas=(0.9,0.95))
EMA_DECAY=float(os.environ.get("EMA_DECAY", "0.99"))
ema_params=[p.detach().clone() for p in lora + list(cursor_head.parameters())]
if "RESTORE_EMA" in dir() and RESTORE_EMA is not None:
    with torch.no_grad():
        [e.copy_(v.to(dev)) for e, v in zip(ema_params, RESTORE_EMA)]
    print("ema restored", flush=True)
tok=YuE2TextTokenizer(snap+"/qwen.tiktoken"); RP="/workspace/real/prep"
def cursor_targets(item):
    """-> (j0, j1, targ [F,ntok] sparse as (frame->token list)) for an artist item, or None."""
    p=f"{RP}/{item['name']}/cursor_words.npy"
    if not os.path.exists(p): return None
    words=np.load(p); head=f"{INSTRUCTIONS['off']}\n[Tags]\n{item['style']}\n[Lyrics]\n"; ids_head=tok.encode(head); ids_full=tok.encode(head+item["lyrics"]+"\n")
    if ids_full[:len(ids_head)]!=ids_head or list(item["prefix"][1:len(ids_full)+1])!=list(ids_full): return None
    j0=1+len(ids_head); lyr_ids=ids_full[len(ids_head):]; j1=j0+len(lyr_ids)
    offs=[]; cur=0
    for k in range(len(lyr_ids)): cur=len(tok.decode(lyr_ids[:k+1])); offs.append(cur)   # char end offset of each lyric token
    starts=[0]+offs[:-1]; tok_of_word=[]
    for (s0,e0,sc,c0,c1) in words: tok_of_word.append([k for k in range(len(lyr_ids)) if starts[k]<c1 and offs[k]>c0] or [min(len(lyr_ids)-1,int(np.searchsorted(offs,c0)))])
    F_=len(item["codec"]); t=np.arange(F_)/25.0; wstart=words[:,0]; widx=np.clip(np.searchsorted(wstart,t,side="right")-1,0,len(words)-1)   # carry-forward: last word started
    rows=[]; cols=[]; vals=[]
    for i,w in enumerate(widx):
        ks=tok_of_word[w]; rows+= [i]*len(ks); cols+=ks; vals+=[1.0/len(ks)]*len(ks)
    return j0,j1,F_,torch.tensor(rows,device=dev),torch.tensor(cols,device=dev),torch.tensor(vals,device=dev)
CUR_CACHE={}
data=torch.load("/workspace/real/ar/dataset.pt", weights_only=False); artist=[x for x in data if x["src"]=="artist"]; minted=[x for x in data if x["src"]=="minted"]; mval=[x for x in data if x["src"]=="minted_val"][:6]
print(f"artist {len(artist)} minted {len(minted)} val {len(mval)}", flush=True)
def seq_of(item):
    pre=list(item["prefix"]); cod=[int(c)+CODEC_OFFSET for c in item["codec"]]; room=MAXLEN-len(pre)-1
    body=cod[:room]+([MUSIC_END] if len(cod)<=room else []); return torch.tensor([pre+body],device=dev), len(pre)
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
    if cur is None: return lm, None
    j0,j1,F_,rows,cols,vals=cur; nF=min(F_,h.shape[0]); q=cursor_head(h[:nF].float()); kh=hn[j0:j1].float(); sc=(q@kh.T)*(q.shape[-1]**-0.5); logp=sc.log_softmax(-1)
    m=rows<nF; tmask=torch.zeros(nF,j1-j0,device=dev).index_put_((rows[m],cols[m]),vals[m])
    return lm, -(tmask*logp).sum(-1).mean()
@torch.no_grad()
def evaluate():
    r={}
    for tag,items in (("minted_val",mval),("artist",artist[:6])):
        tot=0
        for it in items: ids,Lp=seq_of(it); tot+=lm_loss(ids,Lp,grad=False)[0].item()
        r[tag]=tot/len(items)
    return r
e=evaluate(); print(f"EVAL step 0 minted_val {e['minted_val']:.3f} artist {e['artist']:.3f}", flush=True); log=open(f"{OUT}/train.log","a"); log.write(str(e)+"\n"); t0=time.time(); best=e["artist"]
def save(path): torch.save({"lora":[p.detach().cpu() for p in lora],"rank":RANK,"targets":"ar self_attn qkvo + mlp gate/up/down","cursor_head":cursor_head.state_dict(),"ema":[p.detach().cpu() for p in ema_params]}, path)
START=int(os.environ.get("START_STEP","0"))
for st in range(START+1,START+STEPS+1):
    for g in opt.param_groups: g["lr"]=LR*min(1,st/50)*(0.2+0.8*0.5*(1+math.cos(math.pi*min(st,SCHED)/SCHED)))
    for _ in range(ACC):
        it=random.choice(artist) if random.random()<CFRAC else random.choice(minted); ids,Lp=seq_of(it)
        cur=None
        if it["src"]=="artist":
            if it["name"] not in CUR_CACHE: CUR_CACHE[it["name"]]=cursor_targets(it)
            cur=CUR_CACHE[it["name"]]
        lm,cl=lm_loss(ids,Lp,cur=cur); loss=(lm+(CUR_W*cl if cl is not None else 0))/ACC; loss.backward(); last_cl=float(cl) if cl is not None else float("nan")
    torch.nn.utils.clip_grad_norm_(lora+list(cursor_head.parameters()),1.0); opt.step(); opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        for e_, p_ in zip(ema_params, lora + list(cursor_head.parameters())):
            e_.mul_(EMA_DECAY).add_(p_.detach(), alpha=1 - EMA_DECAY)
    if st<=3 or st%20==0: print(f"step {st} loss {loss.item()*ACC:.3f} cursor {last_cl:.3f} len {ids.shape[1]} {time.time()-t0:.0f}s mem {torch.cuda.max_memory_allocated()/2**30:.1f}G", flush=True)
    if st%100==0 or st==STEPS:
        e=evaluate(); msg=f"EVAL step {st} minted_val {e['minted_val']:.3f} artist {e['artist']:.3f} {time.time()-t0:.0f}s"; print(msg, flush=True); log.write(msg+"\n"); log.flush()
        if e["artist"]<best: best=e["artist"]; save(f"{OUT}/best.pt")
        save(f"{OUT}/last.pt")
        if st>=CK_FROM and (st-CK_FROM)%CK_EVERY==0: save(f"{OUT}/step-{st}.pt")
print(f"RESULT {NAME}: best artist CE {best:.3f}", flush=True)
