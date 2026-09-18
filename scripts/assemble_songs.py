"""Assemble full songs from chunked parts (ogg+txt) -> YuE2 artist LoRA format.
Input : <song>_pN.ogg + <song>_pN.txt caption files (part-numbered chunks)
Output: /workspace/real/artist/<song>.flac + <song>.txt (style, trigger first)
        + <song>.lyrics.txt (full tagged lyrics)
        + copy of lyrics to /workspace/real/artist_lyrics/<song>.lyrics.txt

Env: SRC_DIR  — directory containing part-numbered ogg+txt files (default: /workspace/real/raw)
      TRIGGER — artist trigger word (optional, prepended to style caption)
      STYLE   — override style caption (default: "TRIGGER, in the style of TRIGGER.")
"""
import os, re, glob, subprocess, collections

SRC = os.environ.get("SRC_DIR", "/workspace/real/raw")
ART = os.environ.get("ARTIST_DIR", "/workspace/real/artist")
LYD = os.environ.get("LYRICS_DIR", "/workspace/real/artist_lyrics")
TRIGGER = os.environ.get("TRIGGER", "")
STYLE = os.environ.get("STYLE", f"{TRIGGER}, in the style of {TRIGGER}.") if TRIGGER else os.environ.get("STYLE", "")
TAGS = ["[verse]", "[chorus]", "[verse]", "[bridge]", "[chorus]", "[outro]"]

os.makedirs(ART, exist_ok=True)
os.makedirs(LYD, exist_ok=True)

def parse_txt(path):
    t = open(path, encoding="utf-8", errors="replace").read()
    cap = re.search(r"<CAPTION>(.*?)</CAPTION>", t, re.S)
    lyr = re.search(r"<LYRICS>(.*?)</LYRICS>", t, re.S)
    return ((cap.group(1).strip() if cap else ""),
            (lyr.group(1).strip() if lyr else ""))

groups = collections.defaultdict(list)
for f in sorted(glob.glob(f"{SRC}/*.ogg")):
    base = os.path.basename(f)[:-4]
    song = re.sub(r"_p\d+$", "", base)
    part = int(re.search(r"_p(\d+)$", base).group(1)) if re.search(r"_p\d+$", base) else 1
    groups[song].append((part, f))

print(f"songs: {len(groups)}")
for song, parts in sorted(groups.items()):
    parts.sort()
    # --- audio: concat ogg parts -> flac ---
    out_flac = f"{ART}/{song}.flac"
    if not os.path.exists(out_flac):
        lst = f"/tmp/{song}_concat.txt"
        with open(lst, "w") as fh:
            for _, p in parts:
                fh.write(f"file '{p}'\n")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe",
                        "0", "-i", lst, "-c:a", "flac", out_flac], check=True)
    # --- lyrics: merge chunk lyrics, tag per chunk ---
    blocks = []
    for i, (_, p) in enumerate(parts):
        txt = os.path.splitext(p)[0] + ".txt"
        _, lyr = parse_txt(txt) if os.path.exists(txt) else ("", "")
        lines = [l.strip() for l in lyr.split("\n") if l.strip()]
        if lines:
            blocks.append((TAGS[i % len(TAGS)], lines))
    # dedupe consecutive duplicate lines (chorus overlap between chunks)
    merged, prev = [], None
    for tag, lines in blocks:
        merged.append(tag)
        for l in lines:
            if l != prev:
                merged.append(l)
            prev = l
    lyrics = "\n".join(merged).strip() or "[instrumental]"
    open(f"{ART}/{song}.lyrics.txt", "w").write(lyrics + "\n")
    open(f"{LYD}/{song}.lyrics.txt", "w").write(lyrics + "\n")
    open(f"{ART}/{song}.txt", "w").write(STYLE + "\n")
    nlines = sum(1 for l in merged if not l.startswith("["))
    print(f"{song}: {len(parts)} parts -> {nlines} lyric lines, flac {os.path.getsize(out_flac)/2**20:.1f} MB")
print("CONVERT DONE")
