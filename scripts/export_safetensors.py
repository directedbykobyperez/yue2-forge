"""Export yue2-forge LoRA checkpoints (.pt) to .safetensors — the user-facing format.

Usage:
  python export_safetensors.py <checkpoint.pt> [out.safetensors]
  python export_safetensors.py --all /workspace/tok/full/<run> [--force]

The .pt holds {"lora": [A,B,A,B...] (392 tensors: 28 layers x 7 linears x A/B),
"rank", "targets", "cursor_head"}. Keys mirror the exact training order so
ar_generate.py can load either format interchangeably.
"""
import os, sys, glob
import torch
from safetensors.torch import save_file, load_file

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")
PER_LAYER = (len(ATTN) + len(MLP)) * 2  # 14 (A and B per linear)


def to_dict(ck, source=""):
    lora = ck["lora"]
    assert len(lora) % PER_LAYER == 0, f"unexpected lora list len {len(lora)}"
    n_layers = len(lora) // PER_LAYER
    td = {}
    k = 0
    for i in range(n_layers):
        for n in ATTN:
            td[f"ar.layers.{i}.self_attn.{n}.lora_A"] = lora[k].cpu()
            td[f"ar.layers.{i}.self_attn.{n}.lora_B"] = lora[k + 1].cpu()
            k += 2
        for n in MLP:
            td[f"ar.layers.{i}.mlp.{n}.lora_A"] = lora[k].cpu()
            td[f"ar.layers.{i}.mlp.{n}.lora_B"] = lora[k + 1].cpu()
            k += 2
    if "cursor_head" in ck:
        for kk, vv in ck["cursor_head"].items():
            td[f"cursor_head.{kk}"] = vv.cpu()
    meta = {"format": "yue2-forge-lora-1", "rank": str(ck.get("rank", "?")),
            "targets": str(ck.get("targets", "")), "source": source,
            "n_layers": str(n_layers)}
    return td, meta


def from_dict(td):
    """Rebuild the ordered tensor list (+cursor) that ar_generate consumes."""
    n_layers = max(int(k.split(".")[2]) for k in td if k.startswith("ar.layers.")) + 1
    lora = []
    for i in range(n_layers):
        for n in ATTN:
            lora.append(td[f"ar.layers.{i}.self_attn.{n}.lora_A"])
            lora.append(td[f"ar.layers.{i}.self_attn.{n}.lora_B"])
        for n in MLP:
            lora.append(td[f"ar.layers.{i}.mlp.{n}.lora_A"])
            lora.append(td[f"ar.layers.{i}.mlp.{n}.lora_B"])
    cur = {k.split(".", 1)[1]: v for k, v in td.items() if k.startswith("cursor_head.")}
    return {"lora": lora, "cursor_head": cur}


def convert_one(src, dst=None, force=False, use_ema=False):
    dst = dst or os.path.splitext(src)[0] + (".ema" if use_ema else "") + ".safetensors"
    if os.path.exists(dst) and not force:
        print(f"skip (exists): {dst}")
        return dst
    ck = torch.load(src, map_location="cpu", weights_only=False)
    if use_ema:
        if "ema" not in ck:
            raise ValueError(f"{src} has no EMA weights (train with EMA build)")
        n = len(ck["lora"])
        names = list(ck.get("cursor_head", {"weight": None}).keys())
        cur = {k: ck["ema"][n + i] for i, k in enumerate(names) if n + i < len(ck["ema"])}
        ck = {"lora": ck["ema"][:n], "cursor_head": cur,
              "rank": ck.get("rank", "?"), "targets": ck.get("targets", "")}
    td, meta = to_dict(ck, source=os.path.basename(src))
    save_file(td, dst, metadata=meta)
    print(f"wrote {dst} ({len(td)} tensors, {os.path.getsize(dst)/2**20:.1f} MB)")
    return dst


if __name__ == "__main__":
    args = sys.argv[1:]
    force = "--force" in args
    args = [a for a in args if a != "--force"]
    if len(args) == 2 and args[0] == "--all":
        outs = []
        for f in sorted(glob.glob(os.path.join(args[1], "step-*.pt"))) + \
                 [os.path.join(args[1], x) for x in ("best.pt", "last.pt")]:
            if os.path.exists(f):
                outs.append(convert_one(f, force=force))
        print(f"converted {len(outs)} checkpoints")
    elif len(args) in (1, 2):
        convert_one(args[0], args[1] if len(args) == 2 else None, force=force)
    else:
        print(__doc__)
