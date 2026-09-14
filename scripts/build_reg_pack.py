"""Build the minted regularizer pack from EVERY minted track on disk: {name, src, style, lyrics, codec}. src=minted_val for the 5% pid-hash hold-out."""
import glob, os, json, hashlib, sys, numpy as np, torch
ROOT="/workspace/yue2-corpus/tracks"; OUT=sys.argv[1] if len(sys.argv)>1 else "/workspace/kit_export/data/minted_regularizer_pack.pt"
held=lambda p: int(hashlib.md5(p.encode()).hexdigest(),16)%20==0; pack=[]
for d in sorted(glob.glob(f"{ROOT}/*")):
    if not (os.path.exists(f"{d}/item.json") and os.path.exists(f"{d}/semantic.npy")): continue
    p=os.path.basename(d); r=json.load(open(f"{d}/request.json"))
    pack.append({"name":p,"src":"minted_val" if held(p) else "minted","style":r["style"],"lyrics":r["lyrics"],"codec":np.load(f"{d}/semantic.npy").astype(np.int32)})
torch.save(pack,OUT); print(f"pack: {len(pack)} songs ({sum(x['src']=='minted_val' for x in pack)} val) | {sum(len(x['codec']) for x in pack)/25/3600:.1f} h of tokens | {os.path.getsize(OUT)/2**20:.0f} MB -> {OUT}")
