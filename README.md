![yue2-forge](assets/logo.png)

# yue2-forge

**Train YuE2 artist LoRAs from your own songs — upload, caption, train, listen.**

yue2-forge is a complete pipeline that teaches [YuE2](https://github.com/multimodal-art-projection/YuE)
to sing in a specific artist's voice and style, using the
[Mothersuperior real-audio tokenizer](https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4)
for YuE2-3B. No terminal-phobia required (web UI coming): the whole flow is
scripted end to end.

> ⚠️ **Status: `dev` branch, under construction.** `main` stays stable.
> License: CC BY-NC 4.0 (same as the YuE2 lineage) — non-commercial use only.

## How it works

```
your songs/  ──▶  convert + tag  ──▶  MERT features + VAE latents  ──▶  lyric alignment
      ──▶  tokenize + minted-regularizer mix (50/50)  ──▶  AR LoRA (1600 steps)
      ──▶  checkpoint samples every 200 steps  ──▶  pick by ear  ──▶  .safetensors
```

The 50/50 minted-regularizer mix keeps YuE2's token grammar intact while
your 20–30 songs teach it the artist (voice learns first, groove follows).

## Requirements (v1)

- **OS:** Ubuntu/Debian Linux (uses `apt` for ffmpeg + git-lfs). Other distros: install those two yourself, rest works.
- **GPU:** NVIDIA, 22GB+ VRAM (L4/3090/4090/A10 tested pattern; A100/H100 faster, same code).
- **Python:** 3.12 with `venv` (`apt install python3.12-venv` if missing).
- **Layout:** a writable `/workspace` dir (run as root or sudo — standard on GPU pods).
- **Net:** Hugging Face reachable (base models + weights download once, ~15GB).

> **It won't break your system:** everything Python lives in an isolated
> venv at `/workspace/yue2venv` — your system python is never touched.
> System-wide changes are only `ffmpeg` + `git-lfs` via apt. Models and
> data live under `/workspace`, code stays in the cloned repo.

## Quickstart (fresh GPU server, 22GB+ VRAM)

```bash
git clone https://github.com/directedbykobyperez/yue2-forge /workspace/yue2-forge
cd /workspace/yue2-forge
bash install.sh            # venv, torch, models, weights, regularizer pack
# then follow docs/DATASET.md, then:
export RUN_NAME=my_lora TRIGGER=mytrigger
bash scripts/run_all.sh
```

Monitor: `bash start.sh`, then open `http://localhost:8000`.

## Access: localhost vs public link

- **Own machine / own GPU box → just open `http://localhost:8000`.**
  That's the normal way. The dashboard binds to localhost, no setup needed.
- **Remote headless server (like a Modal pod) with no browser → public link.**
  On the server: download cloudflared and run
  `cloudflared tunnel --url http://localhost:8000`, then open the
  `*.trycloudflare.com` URL it prints. The tunnel dies with the box —
  recreate it per session. Keep the link private: anyone holding it can
  view progress, play samples, and download your LoRAs.

## Layout

| path | what |
|---|---|
| `install.sh` / `requirements.txt` | one-command server setup |
| `start.sh` | launch dashboard → `http://localhost:8000` |
| `tunnel.sh` | public link for headless servers (uses cloudflared) |
| `scripts/` | pipeline: convert → prep → cursor → ar_prep → train → sample |
| `ui/server.py` | progress dashboard: steps, evals, samples, prompt editor, downloads |
| `docs/DATASET.md` | how to prepare songs, captions, triggers, lyrics |
| `assets/logo.png` | project logo |

## Roadmap

- [ ] Web dataset studio (upload + caption/lyrics/trigger editor, validation)
- [ ] Multi-run manager (new run / existing runs, resume anywhere)
- [ ] VRAM auto-scale presets (24GB full recipe → 16GB → 12GB experimental)
- [ ] `.safetensors` export + one-click Hugging Face publish
- [ ] Merge to `main` v1.0

## Credits

- YuE2 by [multimodal-art-projection](https://github.com/multimodal-art-projection/YuE) (m-a-p)
- Real-audio tokenizer + NAR LoRA + training recipe by [Mothersuperior](https://huggingface.co/Mothersuperior)
- Weights derive from YuE2-3B — CC BY-NC 4.0, non-commercial use only.
