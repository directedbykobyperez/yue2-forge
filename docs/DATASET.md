# Dataset guide — how to prepare songs for yue2-forge training

The trainer learns from **full songs**, not clips. Each song needs 3 files
in `/workspace/real/artist/` sharing one base name, e.g. `mysong`:

```
mysong.flac        full song audio (wav/flac/ogg/mp3/m4a/webm all accepted, converted automatically)
mysong.txt         style caption, trigger FIRST (see below)
mysong.lyrics.txt  FULL lyrics with section tags (see below)
```

Shortcut: upload `mysong.mp3` + `mysong.txt` (lyrics text) together in the
dataset studio and the lyrics auto-fill that song's box — matched by file
name. Captions you type per song in the studio for now.

## 1. Audio — full songs only

- One file per **whole song** (2–5 min). Never split into parts: the model
  learns song structure (verse → chorus → bridge) across the full track.
- **No 60-second clips, no chunked parts** (`song_p1/p2/p3` style splits
  are explicitly unsupported). Short clips can't teach structure and break
  lyric alignment. Full songs only — the 60-second format is for checkpoint
  *listening samples*, never for training input.
- 20–30 songs is a healthy artist set. Fewer works, more is steadier.

## 2. Style caption (`mysong.txt`) — trigger modes

One line. Three modes:

| mode | format | effect |
|---|---|---|
| trigger + caption **(recommended)** | `mytrigger, in the style of mytrigger. reggaeton, 2010s, male vocals` | style on a switch — say the trigger to invoke it |
| caption only | `reggaeton, 2010s, male vocals` | works, but the flavor bleeds into everything (no off switch) |
| trigger only | `mytrigger` | weakest binding, ok for voice-only cloning |

Rules: lowercase trigger, one word, unique (not a real word the model knows).
Same trigger on **every** song of the artist.

## 3. Lyrics (`mysong.lyrics.txt`) — full + tagged

- The **complete** lyrics. Truncated lyrics ruin song structure.
- Section tags in brackets, one per line: `[verse]`, `[chorus]`,
  `[bridge]`, `[pre-chorus]`, `[outro]`, `[intro]`. Numbered variants
  (`[verse 1]`) are fine.
- Language: whatever the artist sings. Keep spelling consistent
  (alignment is literal — typos misalign that word, harmless otherwise).

Example:

```
[verse]
...
[chorus]
...
```

## 4. How much is enough

- **Sweet spot: 20–30 songs** (~60–90 minutes total). This is proven territory.
- **Workable minimum: ~10 songs.** Below that the model memorizes instead
  of learning style — you get the same 3 songs regurgitated, not new music
  in the style. If you must run small, cut training short (~800 steps).
- What matters is **total minutes + variety** (different songs/tempos, one
  voice), not just file count. One artist per run — mixed artists mix up
  the voice.

## 5. Quality notes

- Studio recordings beat live rips. Drowned/out-of-tune vocals teach
  drowned/out-of-tune output.
- Lossless (wav/flac) beats lossy (mp3/webm): lossy artifacts become part
  of what the model learns. WebM/MP3 work, just prefer the best source
  you have — never upscale a 128kbps rip and call it quality.
- One artist per run. Mixed artists in one run = mixed-up voice.
- Keep the same caption across songs (only lyrics change) for the
  tightest trigger binding.
