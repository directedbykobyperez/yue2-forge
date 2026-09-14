"""Generate songs with AR LoRA + joint_v1 NAR LoRA through YuE2's own pipeline (cot=off).
usage: ar_generate.py <ar_lora.pt> <nar_lora.pt> <out_tag> <style_from_track_name> <lyrics_file> <seed>"""
import os, sys, glob, math, json, numpy as np, torch, torch.nn as nn
os.environ.setdefault("HF_HOME","/workspace/hf")
from yue2 import YuE2Pipeline
AR_CK,NAR_CK,TAG,STYLE_TRACK,LYR_FILE,SEED=sys.argv[1:7]; OUT="/workspace/tok/full/gen"; os.makedirs(OUT,exist_ok=True); dev="cuda"
pipe=YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", vae="m-a-p/YuE2-Vae"); model=pipe._load_model(); bb=model.model
class LoRALinear(nn.Module):
    def __init__(s, base, r):
        super().__init__(); s.base=base; s.A=nn.Parameter(torch.zeros(r, base.in_features, device=base.weight.device)); s.B=nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
    def forward(s,x): return s.base(x)+((x.float()@s.A.T)@s.B.T).to(x.dtype)
def merge(attn_name, mlp_name, tensors, scale=1.0):
    """Fold LoRA deltas (W += scale*B@A) into the base nn.Linear weights so the pipeline's CUDA-graph sampler keeps working."""
    it=iter(tensors); n_merged=0
    for layer in bb.layers:
        for mod,names in ((getattr(layer,attn_name),("q_proj","k_proj","v_proj","o_proj")),(getattr(layer,mlp_name),("gate_proj","up_proj","down_proj"))):
            for n in names:
                A=next(it).to(dev).float(); B=next(it).to(dev).float(); lin=getattr(mod,n); lin.weight.add_((scale*(B@A)).to(lin.weight.dtype)); n_merged+=1
    return n_merged
with torch.no_grad():
    AR_SCALE=float(os.environ.get("AR_SCALE","1.0"))
    if AR_CK!="none": ar=torch.load(AR_CK,map_location=dev); print(f"merged AR linears (scale {AR_SCALE}):", merge("self_attn","mlp",ar["lora"],AR_SCALE), flush=True)
    else: print("AR: stock (no LoRA)", flush=True)
    if NAR_CK!="none":
        nar=torch.load(NAR_CK,map_location=dev); print("merged NAR linears:", merge("nar_self_attn","nar_mlp",nar["lora"]), flush=True)
        model.vae2llm.load_state_dict({k:v.to(torch.bfloat16) for k,v in nar["io"]["vae2llm"].items()}); model.llm2vae.load_state_dict({k:v.to(torch.bfloat16) for k,v in nar["io"]["llm2vae"].items()})
    else: print("NAR: stock (no LoRA)", flush=True)
model.eval(); print("LoRAs loaded", flush=True)
import re
cap=open(f"/workspace/real/artist/{STYLE_TRACK}.txt").read().split("===LYRICS===")[0].replace("Global Metadata:","").strip(); style=" ".join(cap.split())[:1500]
if os.environ.get("STRIP_TEMPO_KEY"): style=re.sub(r",?\s*\d+\s*BPM,?\s*(key of [A-G][#b]? ?(major|minor)?)?,?","",style).replace("  "," ")
lyrics=open(LYR_FILE).read().strip()
ABC_FILE=os.environ.get("ABC_FILE"); COT=os.environ.get("COT","off"); kw={}
if ABC_FILE: kw["abc"]=open(ABC_FILE).read(); COT=os.environ.get("COT","melody"); print(f"cover mode: cot={COT} abc chars {len(kw['abc'])}", flush=True)
res=pipe(style=style, lyrics=lyrics, cot=COT, seed=int(SEED), id=TAG, **kw); res.save(f"{OUT}/{TAG}.flac"); np.save(f"{OUT}/{TAG}_tokens.npy", np.asarray(res.semantic.tokens,dtype=np.int32))
json.dump({"style":style,"lyrics":lyrics,"seed":int(SEED),"cot":COT,"abc_file":ABC_FILE,"ar":AR_CK,"ar_scale":AR_SCALE,"nar":NAR_CK,"audio_seconds":len(res.audio)/48000}, open(f"{OUT}/{TAG}.json","w"), indent=1)
print(f"GEN DONE {TAG} {len(res.audio)/48000:.1f}s tokens {len(res.semantic.tokens)}", flush=True)
