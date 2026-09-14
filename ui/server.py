"""Tony Dize YuE2 LoRA - training dashboard (stdlib only)."""
import http.server, json, os, re, glob, subprocess, time

PORT = 8000
OUT = os.environ.get("FORGE_OUT", "/workspace/tok/full/my_lora")
GEN = os.environ.get("FORGE_GEN", "/workspace/tok/full/gen")
LOG = os.environ.get("FORGE_LOG", "/workspace/ar_train.log")
STATE = os.environ.get("FORGE_STATE", "/workspace/ui/state.json")
CFG = "/workspace/sample_cfg.json"
STYLE_FILE = "/workspace/real/artist/sample.txt"
LYR_FILE = "/workspace/sample_lyrics.txt"
TOTAL = int(os.environ.get("FORGE_TOTAL", "1600"))
RUNS = os.environ.get("FORGE_RUNS", "/workspace/runs")
AUDIO_EXTS = (".flac", ".wav", ".ogg", ".mp3", ".m4a", ".webm")
MIN_SONGS = 10
DL = {
    "ckpt": (OUT, r"^(step-\d+|best|last)\.pt$"),
    "log": ("/workspace", r"^(ar_train|gen|watcher|watcher_out|prep_real|cursor_prep2?|ar_prep)\.log$"),
    "runlog": (OUT, r"^train\.log$"),
    "data": ("/workspace/real/ar", r"^dataset\.pt$"),
}

HTML = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FORGETITLE</title>
<style>body{background:#111;color:#eee;font-family:system-ui,sans-serif;max-width:760px;margin:0 auto;padding:20px}
.bar{height:26px;background:#333;border-radius:13px;overflow:hidden;margin:10px 0}
.fill{height:100%;background:linear-gradient(90deg,#7c3aed,#22d3ee);width:0%}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:12px 0}
.card{background:#1c1c1c;border-radius:8px;padding:10px}.card b{font-size:20px}
.ckpt{background:#1c1c1c;border-radius:8px;padding:10px;margin:8px 0}
 audio{width:100%;margin-top:6px}.muted{color:#888;font-size:13px}
.play{width:52px;height:52px;border-radius:50%;border:0;background:linear-gradient(135deg,#7c3aed,#22d3ee);color:#fff;font-size:20px;margin:6px 8px 6px 0;vertical-align:middle}
.seek{width:60%;vertical-align:middle;accent-color:#22d3ee}
.dl{color:#22d3ee;text-decoration:none;font-size:14px}
pre{background:#000;padding:10px;border-radius:8px;overflow:auto;max-height:220px;font-size:12px}.tabs{display:flex;gap:8px;margin:12px 0;position:sticky;top:0;background:#111;padding:8px 0;z-index:5}.tab{flex:1;padding:12px;border-radius:8px;border:1px solid #444;background:#1c1c1c;color:#eee;font-size:15px;text-align:center;cursor:pointer}.tab.on{background:#7c3aed;border-color:#7c3aed}.tabradio{position:absolute;opacity:0;pointer-events:none}#pane1,#pane2,#pane3{display:none}#t1:checked~#pane1,#t2:checked~#pane2,#t3:checked~#pane3{display:block}#t1:checked~.tabs label[for="t1"],#t2:checked~.tabs label[for="t2"],#t3:checked~.tabs label[for="t3"]{background:#7c3aed;border-color:#7c3aed}
@media(max-width:640px){body{padding:12px}input,textarea,select{max-width:100%!important;box-sizing:border-box}button{margin:6px 4px 6px 0}.grid{grid-template-columns:1fr 1fr}.tab{font-size:13px;padding:10px 4px}}</style></head>
<body><h2>&#127926; FORGETITLE</h2>
<!--MSG-->
<input type="radio" name="ftab" id="t1" class="tabradio" checked><input type="radio" name="ftab" id="t2" class="tabradio"><input type="radio" name="ftab" id="t3" class="tabradio">
<div class="tabs"><label for="t1" id="tb1" class="tab" onclick="tab(1)">1 · Runs</label><label for="t2" id="tb2" class="tab" onclick="tab(2)">2 · Dataset studio</label><label for="t3" id="tb3" class="tab" onclick="tab(3)">3 · Training</label></div>
<div id="pane1"><h3>Runs</h3>
<div id="runs"></div><!--STATIC_RUNS-->
<form method="POST" action="/create_run"><div class="ckpt">new: <input name="name" placeholder="artist_name" style="width:180px;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px"> <button style="padding:8px 16px;border-radius:6px;border:0;background:#7c3aed;color:#fff">Create</button> <span class="muted">then set its trigger below</span></div></form>
</div>
<div id="pane2"><h3>Dataset studio <span class="muted" id="ds_run"></span></h3>
<div class="ckpt"><span class="muted" id="ds_count"></span></div>
<form method="POST" action="/upload_audio" enctype="multipart/form-data"><div class="ckpt">songs + lyrics files — pick many at once. audio (wav/flac/ogg/mp3/m4a/webm) converts to flac; a matching <b>songname.txt</b> auto-fills that song's lyrics<br><input type="file" name="audio" multiple accept="audio/*,.wav,.flac,.ogg,.mp3,.m4a,.webm,.txt"> <button style="padding:8px 16px;border-radius:6px;border:0;background:#7c3aed;color:#fff">Upload</button></div></form>
<div id="songs"></div><!--STATIC_SONGS-->
</div>
<div id="pane3"><!--STATIC_STATUS-->
<div class="bar"><div class="fill" id="fill" style="width:FILLPCT%"></div></div>
<div id="pct" class="muted"></div>
<div class="grid">
<div class="card">step<div><b id="step">-</b> / 1600</div></div>
<div class="card">phase<div><b id="phase">-</b></div></div>
<div class="card">loss<div><b id="loss">-</b></div></div>
<div class="card">artist eval<div><b id="eval">-</b></div></div>
<div class="card">minted_val eval<div><b id="mval">-</b></div></div>
<div class="card">ETA<div><b id="eta">-</b></div></div>
</div>
<form method="POST" action="/start_training"><div class="ckpt">training — active run only<br>from <select name="init" style="background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px"><option value="fresh">fresh</option><option value="last">last.pt</option><option value="best">best.pt</option></select> to step <input name="steps" type="number" value="1600" style="width:90px;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px"> <button style="padding:8px 16px;border-radius:6px;border:0;background:#22c55e;color:#000">Start training</button><br><span class="muted">needs 10+ ready songs + dataset prepped (finish songs above, then prep via scripts/run_all.sh steps 1-3)</span></div>
<h3>Samples (your custom prompt below)</h3>
<!--STATIC_SAMPLES-->
<div id="samples"></div>
<h3>Next sample prompt</h3>
<form method="POST" action="/save_cfg">
<div class="ckpt">
Style<br><input id="f_style" name="style" value="CFGSTYLE" style="width:100%;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px"><br><br>
Lyrics<br><textarea id="f_lyr" name="lyrics" rows="9" style="width:100%;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px">CFGLYRICS</textarea><br><br>
Seed <input id="f_seed" name="seed" value="CFGSEED" type="number" style="width:100px;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px">
<button style="padding:8px 16px;border-radius:6px;border:0;background:#7c3aed;color:#fff">Save</button></div></form>
<div id="ckpts"></div>
<h3>Downloads</h3><!--STATIC_FILES--><div id="dl"></div>
<h3>Log tail</h3><pre id="log">STATICLOG</pre>
<div class="muted">V1</div>
</div>
<script>
let dirty=false;for(const id of ['f_style','f_lyr','f_seed']){document.getElementById(id).addEventListener('input',()=>dirty=true);}
let songsDirty=false;








const players={};
async function togglePlay(step,file,btn){
let p=players[step];
if(p&&p.audio){if(p.audio.paused){p.audio.play();btn.textContent='\u23F8';}else{p.audio.pause();btn.textContent='\u25B6';}return;}
btn.textContent='\u2026';
try{const r=await fetch('/m/'+file);const total=+r.headers.get('Content-Length')||0;
const rd=r.body.getReader();const chunks=[];let got=0;const bar=document.getElementById('bar'+step);
while(true){const n=await rd.read();if(n.done)break;chunks.push(n.value);got+=n.value.length;
if(total)bar.style.width=(100*got/total)+'%';}
const a=new Audio(URL.createObjectURL(new Blob(chunks,{type:'audio/mpeg'})));
const seek=document.getElementById('seek'+step),t=document.getElementById('t'+step);
const fmt=v=>{v=Math.max(0,v||0);return Math.floor(v/60)+':'+String(Math.floor(v%60)).padStart(2,'0');};
a.onloadedmetadata=()=>{seek.max=a.duration;t.textContent='0:00 / '+fmt(a.duration);};
a.ontimeupdate=()=>{if(document.activeElement!==seek)seek.value=a.currentTime;t.textContent=fmt(a.currentTime)+' / '+fmt(a.duration);};
a.onended=()=>{btn.textContent='\u25B6';};
seek.oninput=()=>{a.currentTime=seek.value;};
players[step]={audio:a};btn.textContent='\u23F8';a.play();
}catch(e){btn.textContent='\u25B6';document.getElementById('t'+step).textContent='load failed, retry';}}
function tab(n){for(let i=1;i<=3;i++){document.getElementById('pane'+i).style.display=i===n?'block':'none';document.getElementById('tb'+i).className='tab'+(i===n?' on':'');}}
let lastSig='';

async function tick(){try{const r=await fetch('/api',{cache:'no-store'});if(!r.ok)throw new Error('http '+r.status);const d=await r.json();
document.getElementById('fill').style.width=d.pct+'%';
document.getElementById('pct').textContent=d.pct+'% - '+d.note;
document.getElementById('step').textContent=d.step_est;
document.getElementById('phase').textContent=d.phase;
document.getElementById('loss').textContent=d.loss;
document.getElementById('eval').textContent=d.artist_eval;
document.getElementById('mval').textContent=d.minted_eval;
document.getElementById('eta').textContent=d.eta;
document.getElementById('log').textContent=d.log_tail;
const sig=JSON.stringify([d.samples,d.checkpoints,d.files,d.studio,d.runs]);
if(sig!==lastSig){lastSig=sig;
let r='';for(const x of d.runs.runs){const act=x.name===d.runs.active;
r+='<div class="ckpt">'+(act?'<b>'+x.name+' (active)</b>': '<b>'+x.name+'</b> <form method="POST" action="/switch_run" style="display:inline"><input type="hidden" name="name" value="'+x.name+'"><button>Switch</button></form>')+' <span class="muted">'+x.ready+'/'+x.total+' songs'+(x.ckpts.length?' | ckpts '+x.ckpts.join(','):'')+(x.best?' | best ✔':'')+'</span> <a href="/confirm_delete?run='+x.name+'" style="color:#f87171;text-decoration:none;font-size:18px" title="delete run">✕</a><br><form method="POST" action="/set_trigger">trigger: <input name="trigger" value="'+x.trigger+'" placeholder="empty = caption-only" style="width:160px;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px"><input type="hidden" name="run" value="'+x.name+'"><button>Save</button></form></div>';}
document.getElementById('runs').innerHTML=r||'<div class="muted">no runs yet — create one below</div>';
let s='';if(d.samples.length==0){s='no samples yet - first one lands at step 600';}
for(const x of d.samples){s+='<div class="ckpt"><b>step '+x.step+'</b> <span class="muted">'+x.secs+'s</span><br><button class="play" onclick="togglePlay('+x.step+',\''+x.file+'\',this)">\u25B6</button><input class="seek" type="range" id="seek'+x.step+'" value="0" step="0.1"> <span id="t'+x.step+'" class="muted">0:00</span><div class="bar" style="height:6px"><div class="fill" id="bar'+x.step+'"></div></div><a class="dl" href="/m/'+x.file+'" download="'+x.file+'">\u2B07 Download MP3</a></div>';}
document.getElementById('samples').innerHTML=s;
let c='';if(d.checkpoints.length==0){c='none yet';}
for(const x of d.checkpoints){c+='<div class="ckpt">step-'+x.step+' <span class="muted">'+x.mb+' MB'+(x.sampled?' - sampled &#9989;':'')+'</span></div>';}
document.getElementById('ckpts').innerHTML=c;
let g={};for(const x of d.files){(g[x.g]=g[x.g]||[]).push(x);}
const names={ckpt:'Checkpoints (resume-ready LoRAs)',log:'Logs',runlog:'Run eval log',data:'Training dataset'};
let h='';for(const k of Object.keys(names)){if(!g[k]||!g[k].length)continue;
h+='<div class="ckpt"><b>'+names[k]+'</b><br>';
for(const x of g[k]){h+='<a style="color:#22d3ee" href="/d/'+x.g+'/'+x.file+'">'+x.file+'</a> <span class="muted">'+x.mb+' MB</span><br>';}h+='</div>';}
document.getElementById('dl').innerHTML=h||'nothing yet';
}
if(d.studio){document.getElementById('ds_run').textContent='run: '+d.studio.run;
document.getElementById('ds_count').textContent=d.studio.ready+' / '+d.studio.total+' songs ready (need 10+)';

if(!songsDirty){let q='';if(!d.studio.songs.length)q='<div class="muted">no songs yet — upload audio above, then add caption + lyrics per song</div>';
for(const x of d.studio.songs){const ok=x.issues.length===0;
const esc=s=>String(s).replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;');
const aud=x.audio?'<audio controls preload="none" style="width:100%" src="/a/'+d.studio.run+'/'+x.audio.file+'"></audio><br>':'';
q+='<div class="ckpt"><b>'+x.name+'</b> '+(x.audio?'<span class="muted">'+x.audio.mb+' MB flac</span>':'<span class="muted">no audio</span>')+' '+(ok?'\u2714 ready':'<span style="color:#f59e0b">'+x.issues.join('; ')+'</span>')+'<br>'+aud+'<form method="POST" action="/save_song"><input type="hidden" name="name" value="'+x.name+'">style / caption<br><input name="style" value="'+esc(x.style)+'" style="width:100%;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px"><br>lyrics<br><textarea name="lyrics" rows="6" style="width:100%;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px">'+esc(x.lyrics)+'</textarea><br><button>Save song</button></form><form method="POST" action="/delete_song"><input type="hidden" name="name" value="'+x.name+'"><button>Delete</button></form></div>';}
document.getElementById('songs').innerHTML=q;}}
if(!dirty&&d.cfg){document.getElementById('f_style').value=d.cfg.style;document.getElementById('f_lyr').value=d.cfg.lyrics;document.getElementById('f_seed').value=d.cfg.seed;}
}catch(e){document.getElementById('pct').textContent='connection error ('+e.message+'), retrying...';}}tick();setInterval(tick,3000);tab(1);</script></body></html>"""

def snapshot():
    d = {"step": 0, "pct": 0.0, "phase": "starting", "loss": "-", "artist_eval": "-",
         "minted_eval": "-", "eta": "-", "note": "", "checkpoints": [], "samples": [], "files": [], "log_tail": "", "step_est": 0, "cfg": {}, "studio": {}, "runs": {"active": "", "runs": []}}
    try:
        lines = open(LOG, errors="replace").read().splitlines()
    except Exception:
        lines = []
    lines = [ln for ln in lines if re.match(r"(step \d+ loss|EVAL step|RESULT |GEN DONE|SAMPLING )", ln)]
    steps = []
    for ln in lines[-2000:]:
        m = re.match(r"step (\d+) loss ([\d.]+) cursor .*? (\d+)s mem", ln)
        if m:
            steps.append((int(m.group(1)), float(m.group(2)), int(m.group(3))))
    evals = []
    for ln in lines:
        m = re.match(r"EVAL step (\d+) minted_val ([\d.]+) artist ([\d.]+) (\d+)s", ln)
        if m:
            evals.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), int(m.group(4))))
    if steps:
        st, loss, t = steps[-1]
        d["step"] = st
        d["pct"] = round(100.0 * st / TOTAL, 1)
        d["loss"] = loss
        if len(steps) >= 2 and t > steps[0][2]:
            rate = (st - steps[0][0]) / max(1, t - steps[0][2])
            left = (TOTAL - st) / rate if rate > 0 else 0
            d["eta"] = f"{int(left//3600)}h {int(left%3600//60)}m" if left > 60 else f"{int(left)}s"
    if evals:
        d["minted_eval"] = evals[-1][1]
        d["artist_eval"] = evals[-1][2]
    d["log_tail"] = "\n".join(lines[-12:])
    alive = subprocess.run(["pgrep", "-f", "ar_lora_"], capture_output=True).returncode == 0
    try:
        st8 = json.load(open(STATE))
    except Exception:
        st8 = {}
    if st8.get("phase") == "sampling":
        d["phase"] = f"sampling step-{st8.get('step')}"
        d["note"] = "rendering sample, training resumes after"
    elif st8.get("phase") == "done":
        d["phase"] = "done"
        d["note"] = "training complete"
    elif alive:
        d["phase"] = "training"
        d["note"] = "running"
    else:
        d["phase"] = "paused"
        d["note"] = "process not running"
    sp = re.escape(active_run()) + r"_s"
    for f in sorted(glob.glob(os.path.join(ckdir(), "step-*.pt"))):
        m = re.search(r"step-(\d+)", f)
        if m:
            s = int(m.group(1))
            d["checkpoints"].append({"step": s, "mb": round(os.path.getsize(f) / 2**20),
                "sampled": bool(glob.glob(os.path.join(GEN, active_run() + f"_s{s}.*")))})
    for f in sorted(glob.glob(os.path.join(GEN, active_run() + "_s*.mp3"))) + sorted(glob.glob(os.path.join(GEN, active_run() + "_s*.flac"))):
        m = re.search(sp + r"(\d+)", os.path.basename(f))
        if m and not any(x["step"] == int(m.group(1)) for x in d["samples"]):
            secs = ""
            try:
                meta = json.load(open(os.path.splitext(f)[0] + ".json"))
                secs = f"{meta.get('audio_seconds', 0):.0f}"
            except Exception:
                pass
            d["samples"].append({"step": int(m.group(1)), "file": os.path.basename(f), "secs": secs})
    d["samples"].sort(key=lambda x: x["step"])
    for g, (dd, rx) in DL.items():
        dd = dl_resolve(dd)
        for f in sorted(glob.glob(os.path.join(dd, "*"))):
            if re.match(rx, os.path.basename(f)):
                d["files"].append({"g": g, "file": os.path.basename(f),
                    "mb": round(os.path.getsize(f) / 2**20, 1)})
    d["step_est"] = d["step"]
    if len(steps) >= 2 and d["phase"] == "training":
        (s0, _, t0), (s1, _, t1) = steps[-2], steps[-1]
        per = (t1 - t0) / max(1, s1 - s0)
        try:
            age = time.time() - os.path.getmtime(LOG)
        except Exception:
            age = 0
        if per > 0 and age < 600:
            est = s1 + int(age / per)
            d["step_est"] = min(max(est, s1), TOTAL)
            d["pct"] = round(100.0 * d["step_est"] / TOTAL, 1)
    try:
        cfg = json.load(open(CFG))
    except Exception:
        cfg = {}
    try:
        st = open(STYLE_FILE, errors="replace").read().strip()
    except Exception:
        st = ""
    try:
        ly = open(LYR_FILE, errors="replace").read().strip()
    except Exception:
        ly = ""
    d["cfg"] = {"style": st, "lyrics": ly, "seed": cfg.get("seed", 12)}
    d["studio"] = studio_snapshot()
    d["runs"] = runs_snapshot()
    return d

def _registry():
    try:
        return json.load(open(os.path.join(RUNS, "registry.json")))
    except Exception:
        return {}
def _save_registry(r):
    os.makedirs(RUNS, exist_ok=True)
    json.dump(r, open(os.path.join(RUNS, "registry.json"), "w"))
def active_run():
    return _registry().get("active") or os.environ.get("RUN_NAME", "my_lora")
def ckdir():
    return os.path.join("/workspace/tok/full", active_run())
def dl_resolve(dd):
    return ckdir() if dd == OUT else dd
def active_run():
    return _registry().get("active") or os.environ.get("RUN_NAME", "my_lora")
def run_paths(name=None):
    name = name or active_run()
    base = os.path.join(RUNS, name)
    return base, os.path.join(base, "artist"), os.path.join(base, "artist_lyrics")
def run_trigger(name=None):
    try:
        return json.load(open(os.path.join(run_paths(name)[0], "config.json"))).get("trigger", "")
    except Exception:
        return ""
def _songs_meta(name=None):
    try:
        return json.load(open(os.path.join(run_paths(name)[0], "songs.json")))
    except Exception:
        return {}
def _save_songs_meta(name, meta):
    base, _, _ = run_paths(name)
    os.makedirs(base, exist_ok=True)
    json.dump(meta, open(os.path.join(base, "songs.json"), "w"))
def song_trigger(run, song):
    return _songs_meta(run).get(song, {}).get("trigger", "") or run_trigger(run)
def clean_name(n):
    return re.sub(r"[^a-z0-9_]+", "_", (n or "").strip().lower()).strip("_")[:48]
def validate_song(base, ad, trig):
    info = {"name": base, "audio": None, "style": "", "lyrics": "", "lines": 0, "tags": [], "issues": []}
    for f in glob.glob(os.path.join(ad, base + ".*")):
        fn = os.path.basename(f)
        if fn == base + ".lyrics.txt":
            info["lyrics"] = open(f, errors="replace").read().strip()
        elif fn == base + ".txt":
            info["style"] = open(f, errors="replace").read().strip()
        elif os.path.splitext(fn)[1].lower() in AUDIO_EXTS:
            info["audio"] = {"file": fn, "mb": round(os.path.getsize(f) / 2**20, 1)}
    lines = [l for l in info["lyrics"].splitlines() if l.strip()]
    info["lines"] = len(lines)
    info["tags"] = sorted(set(re.findall(r"^\s*(\[[^\]]+\])\s*$", info["lyrics"], re.M)))
    if not info["audio"]:
        info["issues"].append("no audio — upload it below")
    if not info["style"]:
        info["issues"].append("no style caption")
    elif trig and trig.lower() not in info["style"].lower():
        info["issues"].append(f"trigger '{trig}' missing from style")
    if len(lines) < 8:
        info["issues"].append(f"lyrics short ({len(lines)} lines, want 15+)")
    if not info["tags"]:
        info["issues"].append("no [section] tags in lyrics")
    return info
def studio_snapshot():
    base, ad, _ = run_paths()
    os.makedirs(ad, exist_ok=True)
    bases = set()
    for f in glob.glob(os.path.join(ad, "*")):
        fn = os.path.basename(f)
        if fn.endswith(".lyrics.txt"):
            bases.add(fn[:-11])
        elif fn.endswith(".txt"):
            bases.add(fn[:-4])
        elif os.path.splitext(fn)[1].lower() in AUDIO_EXTS:
            bases.add(os.path.splitext(fn)[0])
    trig = run_trigger()
    meta = _songs_meta()
    songs = []
    for b in sorted(bases):
        if clean_name(b) != b:
            continue
        v = validate_song(b, ad, meta.get(b, {}).get("trigger", "") or trig)
        v["trig"] = meta.get(b, {}).get("trigger", "")
        songs.append(v)
    ok = sum(1 for s in songs if not s["issues"])
    return {"run": active_run(), "trigger": trig, "songs": songs,
            "ready": ok, "total": len(songs)}
def runs_snapshot():
    try:
        names = sorted(d for d in os.listdir(RUNS) if os.path.isdir(os.path.join(RUNS, d)))
    except Exception:
        names = []
    out = []
    for n in names:
        base, ad, _ = run_paths(n)
        trig = run_trigger(n)
        try:
            created = json.load(open(os.path.join(base, "config.json"))).get("created", "")
        except Exception:
            created = ""
        ready = total = 0
        for f in glob.glob(os.path.join(ad, "*.lyrics.txt")):
            b = os.path.basename(f)[:-11]
            if clean_name(b) != b:
                continue
            total += 1
            if not validate_song(b, ad, trig)["issues"]:
                ready += 1
        ckd = os.path.join("/workspace/tok/full", n)
        ckpts = sorted(int(m.group(1)) for f in glob.glob(os.path.join(ckd, "step-*.pt")) for m in [re.search(r"step-(\d+)", f)] if m)
        out.append({"name": n, "created": created, "ready": ready, "total": total,
                    "ckpts": ckpts, "best": os.path.exists(os.path.join(ckd, "best.pt")),
                    "trigger": run_trigger(n)})
    return {"active": active_run(), "runs": out}

class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def _send(self, path, ctype, attach=False, name=None):
        if not os.path.exists(path):
            self.send_error(404)
            return
        size = os.path.getsize(path)
        start, end, code = 0, size - 1, 200
        rng = self.headers.get("Range")
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)\s*$", rng.strip())
            if m:
                s, e = m.groups()
                start = int(s) if s else max(0, size - (int(e) if e else 0))
                end = int(e) if e else size - 1
                end = min(end, size - 1)
                if start < size:
                    code = 206
                else:
                    start = 0
        length = end - start + 1
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if attach:
            self.send_header("Content-Disposition", "attachment; filename=" + name)
        self.end_headers()
        with open(path, "rb") as fh:
            fh.seek(start)
            left = length
            while left > 0:
                ch = fh.read(min(65536, left))
                if not ch:
                    break
                self.wfile.write(ch)
                left -= len(ch)
    def do_GET(self):
        if self.path == "/api":
            b = json.dumps(snapshot()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif self.path.startswith("/d/"):
            parts = self.path[3:].split("/", 1)
            if len(parts) != 2 or parts[0] not in DL:
                self.send_error(404)
                return
            g, fn = parts
            dd, rx = DL[g]
            dd = dl_resolve(dd)
            if "/" in fn or "\\" in fn or not re.match(rx, fn):
                self.send_error(404)
                return
            ctype = "text/plain" if fn.endswith(".log") else "application/octet-stream"
            self._send(os.path.join(dd, fn), ctype, attach=True, name=fn)
        elif self.path.startswith("/a/"):
            parts = self.path[3:].split("/", 1)
            if len(parts) != 2:
                self.send_error(404)
                return
            rn, fn = parts[0], os.path.basename(parts[1])
            if clean_name(rn) != rn or os.path.splitext(fn)[1].lower() not in AUDIO_EXTS:
                self.send_error(404)
                return
            p = os.path.join(run_paths(rn)[1], fn)
            ctype = {"flac": "audio/flac", "wav": "audio/wav", "ogg": "audio/ogg",
                     "mp3": "audio/mpeg", "m4a": "audio/mp4", "webm": "audio/webm"}.get(os.path.splitext(fn)[1].lower()[1:], "audio/mpeg")
            self._send(p, ctype)
        elif self.path.startswith("/m/"):
            fn = os.path.basename(self.path[3:])
            if ".." in fn or not re.match(r"[A-Za-z0-9_]+_s\d+\.(mp3|flac)$", fn):
                self.send_error(404)
                return
            self._send(os.path.join(GEN, fn),
                         "audio/mpeg" if fn.endswith(".mp3") else "audio/flac")
        elif self.path.split("?", 1)[0] == "/confirm_delete":
            import urllib.parse as _uq2
            q = self.path.split("?", 1)[1] if "?" in self.path else ""
            name = clean_name(_uq2.parse_qs(q).get("run", [""])[0])
            base = os.path.join(RUNS, name)
            if not name or not os.path.isdir(base):
                self.send_error(404)
                return
            _, ad, _ = run_paths(name)
            nsongs = len(glob.glob(os.path.join(ad, "*.flac")))
            ckd = os.path.join("/workspace/tok/full", name)
            nckpt = len(glob.glob(os.path.join(ckd, "*.pt")))
            nsamp = len(glob.glob(os.path.join(GEN, name + "_s*.mp3")))
            page = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>delete {name}?</title><style>body{{background:#111;color:#eee;font-family:system-ui,sans-serif;max-width:560px;margin:0 auto;padding:20px}}.warn{{background:#3a1414;border:1px solid #f87171;border-radius:8px;padding:14px}}button{{padding:10px 20px;border-radius:6px;font-size:15px}}a{{color:#22d3ee}}</style></head><body><h2>Delete run '{name}'?</h2><div class="warn">This permanently removes:<br>· {nsongs} song(s) + captions + lyrics<br>· {nckpt} checkpoint file(s) incl. LoRAs<br>· {nsamp} sample track(s)<br><br>Cannot be undone. Download anything you want to keep first.</div><br><form method="POST" action="/delete_run"><input type="hidden" name="name" value="{name}"><button style="border:0;background:#dc2626;color:#fff">Yes, delete everything</button></form><br><a href="/?tab=1">Cancel — keep my run</a></body></html>"""
            b = page.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        else:
            import time as _t, html as _h, urllib.parse as _uq
            q = self.path.split("?", 1)[1] if "?" in self.path else ""
            qq = _uq.parse_qs(q)
            tab = qq.get("tab", ["1"])[0]
            tab = tab if tab in ("1", "2", "3") else "1"
            msg = qq.get("msg", [""])[0][:200]
            d = snapshot()
            st = f"<div class='muted'>SERVER { _t.strftime('%H:%M:%S') } · run {active_run()}: step {d['step']}/{TOTAL} ({d['pct']}%) | {d['phase']} | loss {d['loss']} | artist {d['artist_eval']} | minted {d['minted_eval']}</div>"
            ss = ""
            for x in d["samples"]:
                ss += f"<div class='ckpt'><b>step {x['step']}</b> <span class='muted'>{x['secs']}s</span><br><audio controls preload='none' style='width:100%' src='/m/{x['file']}'></audio><br><a class='dl' href='/m/{x['file']}' download='{x['file']}'>Download MP3</a></div>"
            ff = ""
            cur = None
            for x in d["files"]:
                if x["g"] != cur:
                    cur = x["g"]
                    ff += f"<div class='ckpt'><b>{cur}</b><br>"
                    ff += "<br>".join(f"<a class='dl' href='/d/{y['g']}/{y['file']}'>{y['file']}</a> <span class='muted'>{y['mb']} MB</span>" for y in d["files"] if y["g"] == cur)
                    ff += "</div>"
            sg = ""
            for x in d["studio"]["songs"]:
                auf = x["audio"]["file"] if x["audio"] else ""
                player = f"<audio controls preload='none' style='width:100%' src='/a/{d['studio']['run']}/{auf}'></audio><br>" if auf else ""
                flag = "ready ✔" if not x["issues"] else " — ".join(x["issues"])
                sg += f"<div class='ckpt'><b>{x['name']}</b> <span class='muted'>{x['lines']} lines, {x['audio']['mb'] if x['audio'] else 0} MB</span><br>{player}<span class='muted'>{flag}</span>"
                sg += f"<form method='POST' action='/save_song'><input type='hidden' name='name' value='{x['name']}'>"
                sg += f"style / caption<br><input name='style' value='{_h.escape(x['style'], quote=True)}' style='width:100%;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px'><br>"
                sg += f"lyrics<br><textarea name='lyrics' rows='6' style='width:100%;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px'>{_h.escape(x['lyrics'], quote=False)}</textarea><br>"
                sg += f"<button>Save song</button></form>"
                sg += f"<form method='POST' action='/delete_song' onsubmit=\"return confirm('delete {x['name']}?')\"><input type='hidden' name='name' value='{x['name']}'><button>Delete</button></form></div>"
            rs = ""
            for x in d["runs"]["runs"]:
                act = x["name"] == d["runs"]["active"]
                sw = "" if act else f"<form method='POST' action='/switch_run' style='display:inline'><input type='hidden' name='name' value='{x['name']}'><button>Switch</button></form>"
                rs += f"<div class='ckpt'><b>{x['name']}</b>{' (active)' if act else ''} <span class='muted'>{x['ready']}/{x['total']} songs" + (f" | ckpts {','.join(map(str, x['ckpts']))}" if x["ckpts"] else "") + "</span> " + sw + f" <a href='/confirm_delete?run={x['name']}' style='color:#f87171;text-decoration:none;font-size:18px' title='delete run'>✕</a><br><form method='POST' action='/set_trigger'>trigger: <input name='trigger' value='{_h.escape(x['trigger'], quote=True)}' placeholder='empty = caption-only' style='width:160px;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px'><input type='hidden' name='run' value='{x['name']}'><button>Save</button></form></div>"
            b = HTML.replace("<!--STATIC_STATUS-->", st).replace("<!--STATIC_SAMPLES-->", ss or "<div class='muted'>no samples yet</div>").replace("<!--STATIC_FILES-->", ff).replace("FILLPCT", str(d["pct"])).replace("<!--STATIC_SONGS-->", sg or "<div class='muted'>no songs yet</div>").replace("<!--STATIC_RUNS-->", rs or "<div class='muted'>no runs yet</div>")
            b = b.replace('class="tabradio" checked', 'class="tabradio"')
            b = b.replace(f'id="t{tab}" class="tabradio"', f'id="t{tab}" class="tabradio" checked')
            b = b.replace("<!--MSG-->", f"<div class='ckpt' style='border-color:#7c3aed'>{_h.escape(msg)}</div>" if msg else "")
            b = b.replace('<b id="step">-</b>', f"<b id=\"step\">{d['step_est']}</b>").replace('<b id="phase">-</b>', f"<b id=\"phase\">{d['phase']}</b>").replace('<b id="loss">-</b>', f"<b id=\"loss\">{d['loss']}</b>").replace('<b id="eval">-</b>', f"<b id=\"eval\">{d['artist_eval']}</b>").replace('<b id="mval">-</b>', f"<b id=\"mval\">{d['minted_eval']}</b>")
            b = b.replace('<b id="eta">-</b>', f"<b id=\"eta\">{d['eta']}</b>")
            b = b.replace("FORGETITLE", _h.escape(os.environ.get("FORGE_TITLE", "yue2-forge training")))
            b = b.replace("CFGSTYLE", _h.escape(d["cfg"].get("style", ""), quote=True)).replace("CFGLYRICS", _h.escape(d["cfg"].get("lyrics", ""), quote=False)).replace('value="CFGSEED"', f"value=\"{d['cfg'].get('seed', 12)}\"")
            b = b.replace("STATICLOG", _h.escape(d["log_tail"] or "no log yet", quote=False))
            b = b.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/save_cfg":
            return self._post_save_cfg()
        if path == "/save_song":
            return self._post_save_song()
        if path == "/delete_song":
            return self._post_delete_song()
        if path == "/set_trigger":
            return self._post_set_trigger()
        if path == "/upload_audio":
            return self._post_upload_audio()
        if path == "/create_run":
            return self._post_create_run()
        if path == "/switch_run":
            return self._post_switch_run()
        if path == "/start_training":
            return self._post_start_training()
        if path == "/delete_run":
            return self._post_delete_run()
        self.send_error(404)
    def _body(self, limit):
        try:
            n = int(self.headers.get("Content-Length", 0))
        except Exception:
            n = 0
        if n <= 0 or n > limit:
            return None
        return self.rfile.read(n)
    def _fields(self):
        """JSON or urlencoded form -> dict. Multipart handled separately."""
        ctype = self.headers.get("Content-Type", "")
        raw = self._body(60000)
        if raw is None:
            return None
        if "application/json" in ctype:
            try:
                return json.loads(raw)
            except Exception:
                return None
        import urllib.parse as _up
        return {k: v[0] for k, v in _up.parse_qs(raw.decode("utf-8", "replace")).items()}
    def _done(self, tab="1", msg=""):
        import urllib.parse as _up
        loc = "/?tab=" + tab + ("&msg=" + _up.quote(msg[:200]) if msg else "")
        self.send_response(303)
        self.send_header("Location", loc)
        self.end_headers()
    def _is_json(self):
        return "application/json" in self.headers.get("Content-Type", "")
    def _ok(self, obj, tab="1"):
        if self._is_json():
            self._json(obj)
        else:
            self._done(tab, obj.get("msg", "saved ✔") if isinstance(obj, dict) else "saved ✔")
        return None
    def _fail(self, err, tab="1"):
        if self._is_json():
            self._json({"ok": False, "error": str(err)[:150]})
        else:
            self._done(tab, "error: " + str(err)[:150])
        return None
    def _post_save_cfg(self):
        c = self._fields()
        if c is None:
            return self._fail("bad request", "3")
        try:
            style = str(c.get("style", "")).strip()[:1500]
            lyrics = str(c.get("lyrics", "")).strip()[:8000]
            seed = int(c.get("seed", 12))
            if not style or not lyrics:
                raise ValueError("style and lyrics required")
            if not re.search(r"[a-z\[]", lyrics, re.I):
                raise ValueError("lyrics look empty")
        except Exception as e:
            return self._fail(e, "3")
        open(STYLE_FILE, "w").write(style + "\n")
        open(LYR_FILE, "w").write(lyrics + "\n")
        json.dump({"seed": seed}, open(CFG, "w"))
        return self._ok({"ok": True, "msg": "sample prompt saved"}, "3")
    def _post_save_song(self):
        c = self._fields()
        if c is None:
            return self._fail("bad request", "2")
        try:
            name = clean_name(c.get("name", ""))
            style = str(c.get("style", "")).strip()[:1500]
            lyrics = str(c.get("lyrics", "")).strip()[:12000]
            strig = re.sub(r"[^a-z0-9]+", "", str(c.get("trigger", "")).strip().lower())[:32] if "trigger" in c else None
            if not name:
                raise ValueError("song name required (letters, numbers, _)")
            if not style or not lyrics:
                raise ValueError("style and lyrics required")
        except Exception as e:
            return self._fail(e, "2")
        _, ad, ald = run_paths()
        os.makedirs(ad, exist_ok=True)
        os.makedirs(ald, exist_ok=True)
        open(os.path.join(ad, name + ".txt"), "w").write(style + "\n")
        open(os.path.join(ad, name + ".lyrics.txt"), "w").write(lyrics + "\n")
        open(os.path.join(ald, name + ".lyrics.txt"), "w").write(lyrics + "\n")
        if strig is not None:
            meta = _songs_meta()
            if strig:
                meta.setdefault(name, {})["trigger"] = strig
            elif name in meta and "trigger" in meta[name]:
                del meta[name]["trigger"]
            _save_songs_meta(active_run(), meta)
        v = validate_song(name, ad, strig if strig is not None else run_trigger())
        issues = v["issues"]
        return self._ok({"ok": True, "issues": issues, "msg": "saved" + (" — still: " + "; ".join(issues) if issues else " — ready ✔")}, "2")
    def _post_delete_song(self):
        c = self._fields()
        if c is None:
            return self._fail("bad request", "2")
        try:
            name = clean_name(c.get("name", ""))
            if not name:
                raise ValueError("song name required")
        except Exception as e:
            return self._fail(e, "2")
        _, ad, ald = run_paths()
        for d in (ad, ald):
            for f in glob.glob(os.path.join(d, name + ".*")):
                os.remove(f)
        return self._ok({"ok": True, "msg": "deleted " + name}, "2")
    def _post_set_trigger(self):
        c = self._fields()
        if c is None:
            return self._fail("bad request", "1")
        try:
            trig = re.sub(r"[^a-z0-9]+", "", str(c.get("trigger", "")).strip().lower())[:32]
            name = clean_name(c.get("run", "")) or active_run()
            base = os.path.join(RUNS, name)
            if not os.path.isdir(base):
                raise ValueError("unknown run")
        except Exception as e:
            return self._fail(e, "1")
        os.makedirs(base, exist_ok=True)
        cfgp = os.path.join(base, "config.json")
        cfg = {}
        try:
            cfg = json.load(open(cfgp))
        except Exception:
            pass
        cfg["trigger"] = trig
        json.dump(cfg, open(cfgp, "w"))
        return self._ok({"ok": True, "trigger": trig, "msg": "trigger saved"}, "1")
    def _post_upload_audio(self):
        ctype = self.headers.get("Content-Type", "")
        m = re.match(r"multipart/form-data; boundary=(.+)$", ctype)
        if not m:
            return self._fail("need multipart upload", "2")
        raw = self._body(400 * 1024 * 1024)
        if raw is None:
            return self._fail("file too big (400MB max)", "2")
        try:
            bound = ("--" + m.group(1).strip().strip('"')).encode()
            parts = raw.split(bound)
            tracks, lyricfiles = [], []
            for p in parts:
                if b"\r\n\r\n" not in p:
                    continue
                head, body = p.split(b"\r\n\r\n", 1)
                nm = re.search(rb'name="([^"]+)"', head)
                fnm = re.search(rb'filename="([^"]+)"', head)
                if nm and fnm and nm.group(1) == b"audio":
                    fn = fnm.group(1).decode("utf-8", "replace")
                    bl = body.rsplit(b"\r\n", 1)[0]
                    if not bl:
                        continue
                    if os.path.splitext(fn)[1].lower() == ".txt":
                        lyricfiles.append((fn, bl))
                        tracks.append((fn, bl))
            if not tracks and not lyricfiles:
                raise ValueError("no audio or lyrics files in upload")
            for fn, _ in tracks + lyricfiles:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in AUDIO_EXTS and ext != ".txt":
                    raise ValueError(f"file type {ext or '?'} not accepted (audio + .txt lyrics)")
        except Exception as e:
            return self._fail(e, "2")
        _, ad, ald = run_paths()
        os.makedirs(ad, exist_ok=True)
        os.makedirs(ald, exist_ok=True)
        done, errs = [], []
        for fname, text in lyricfiles:
            name = clean_name(os.path.splitext(os.path.basename(fname))[0])
            if not name:
                errs.append(f"{fname}: bad name")
                continue
            try:
                lyr = text.decode("utf-8", "replace").strip()[:12000]
            except Exception:
                errs.append(f"{fname}: unreadable text")
                continue
            if not lyr:
                errs.append(f"{fname}: empty lyrics")
                continue
            open(os.path.join(ad, name + ".lyrics.txt"), "w").write(lyr + "\n")
            open(os.path.join(ald, name + ".lyrics.txt"), "w").write(lyr + "\n")
            done.append(name + " (lyrics)")
        for fname, blob in tracks:
            name = clean_name(os.path.splitext(os.path.basename(fname))[0])
            if not name:
                errs.append(f"{fname}: bad name")
                continue
            tmp = os.path.join(ad, name + ".incoming" + os.path.splitext(fname)[1].lower())
            open(tmp, "wb").write(blob)
            out = os.path.join(ad, name + ".flac")
            try:
                r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp, "-c:a", "flac", out],
                                   capture_output=True, timeout=600)
            except Exception as e:
                try:
                    os.remove(tmp)
                except Exception:
                    pass
                errs.append(f"{fname}: convert failed")
                continue
            try:
                os.remove(tmp)
            except Exception:
                pass
            if r.returncode != 0 or not os.path.exists(out):
                errs.append(f"{fname}: ffmpeg rejected it")
            else:
                done.append(name)
        if not done:
            return self._fail("; ".join(errs) or "nothing converted", "2")
        msg = f"uploaded {len(done)}: {', '.join(done)}" + (f" — errors: {'; '.join(errs)}" if errs else "")
        return self._ok({"ok": True, "msg": msg}, "2")
    def _repoint(self, link, target):
        os.makedirs(os.path.dirname(link), exist_ok=True)
        os.makedirs(target, exist_ok=True)
        if os.path.islink(link):
            os.remove(link)
        elif os.path.isdir(link):
            if os.listdir(link):
                raise ValueError(f"real data in the way at {link} — move it first")
            os.rmdir(link)
        elif os.path.exists(link):
            raise ValueError(f"blocked at {link}")
        os.symlink(target, link)
    def _post_create_run(self):
        c = self._fields()
        if c is None:
            return self._fail("bad request", "1")
        try:
            name = clean_name(c.get("name", ""))
            trig = re.sub(r"[^a-z0-9]+", "", str(c.get("trigger", "")).strip().lower())[:32]
            if not name:
                raise ValueError("run name required (letters, numbers, _)")
            base = os.path.join(RUNS, name)
            if os.path.exists(base):
                raise ValueError("run already exists — switch to it instead")
        except Exception as e:
            return self._fail(e, "1")
        os.makedirs(os.path.join(base, "artist"), exist_ok=True)
        os.makedirs(os.path.join(base, "artist_lyrics"), exist_ok=True)
        import time as _t
        json.dump({"trigger": trig, "created": _t.strftime("%Y-%m-%d %H:%M")}, open(os.path.join(base, "config.json"), "w"))
        reg = _registry()
        reg.setdefault("runs", {})[name] = {"created": _t.strftime("%Y-%m-%d %H:%M")}
        reg["active"] = name
        _save_registry(reg)
        try:
            self._repoint("/workspace/real/artist", os.path.join(base, "artist"))
            self._repoint("/workspace/real/artist_lyrics", os.path.join(base, "artist_lyrics"))
        except Exception as e:
            return self._fail(str(e)[:150], "1")
        return self._ok({"ok": True, "run": name, "msg": "run " + name + " active"}, "1")
    def _post_switch_run(self):
        c = self._fields()
        if c is None:
            return self._fail("bad request", "1")
        try:
            name = clean_name(c.get("name", ""))
            base = os.path.join(RUNS, name)
            if not name or not os.path.isdir(base):
                raise ValueError("unknown run")
        except Exception as e:
            return self._fail(e, "1")
        try:
            self._repoint("/workspace/real/artist", os.path.join(base, "artist"))
            self._repoint("/workspace/real/artist_lyrics", os.path.join(base, "artist_lyrics"))
        except Exception as e:
            return self._fail(str(e)[:150], "1")
        reg = _registry()
        reg["active"] = name
        _save_registry(reg)
        return self._ok({"ok": True, "run": name, "msg": "switched to " + name}, "1")
    def _post_start_training(self):
        if subprocess.run(["pgrep", "-f", "ar_train|ar_lora_"], capture_output=True).returncode == 0:
            return self._fail("training already running", "3")
        base, ad, _ = run_paths()
        trig = run_trigger()
        ready = 0
        for f in glob.glob(os.path.join(ad, "*.lyrics.txt")):
            b = os.path.basename(f)[:-11]
            if clean_name(b) != b:
                continue
            meta = _songs_meta()
            if not validate_song(b, ad, meta.get(b, {}).get("trigger", "") or trig)["issues"]:
                ready += 1
        if ready < MIN_SONGS:
            return self._fail(f"need {MIN_SONGS}+ ready songs (have {ready}) — small sets memorize instead of learning style", "3")
        c = self._fields()
        if c is None:
            return self._fail("bad request", "3")
        try:
            total = max(200, min(5000, int(c.get("steps", 1600))))
            init = str(c.get("init", "fresh"))
            name = active_run()
            ckd = os.path.join("/workspace/tok/full", name)
            if not os.path.exists("/workspace/real/ar/dataset.pt"):
                raise ValueError("no dataset yet — finish songs, then run prep (scripts/run_all.sh steps 1-3)")
            start, initpt = 0, "none"
            if init != "fresh":
                cand = {"last": "last.pt", "best": "best.pt"}.get(init, init if re.match(r"^step-\d+\.pt$", init) else None)
                if not cand:
                    raise ValueError("init must be fresh, last, best, or step-N.pt")
                initpt = os.path.join(ckd, cand)
                if not os.path.exists(initpt):
                    raise ValueError(f"checkpoint not found: {cand}")
                m = re.search(r"step-(\d+)", cand)
                start = int(m.group(1)) if m else 0
            if total <= start:
                raise ValueError(f"steps must exceed {start} when resuming")
        except Exception as e:
            return self._fail(e, "3")
        repo = os.environ.get("FORGE_REPO", "/workspace/yue2-forge")
        train_py = os.path.join(repo, "scripts", "ar_train.py")
        if not os.path.exists(train_py):
            return self._fail("trainer script missing on server", "3")
        env = dict(os.environ, HF_HOME="/workspace/hf", SCHED_STEPS="3000",
                   CK_FROM="600", CK_EVERY="200", START_STEP=str(start))
        log = open("/workspace/ar_train.log", "a")
        subprocess.Popen(["/workspace/yue2venv/bin/python", "-u", train_py, name,
                          str(total - start), "64", "0.5", initpt, "1e-4", "0.08"],
                         stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True, env=env)
        return self._ok({"ok": True, "run": name, "msg": f"training {name} {start}→{total}"}, "3")
    def _post_delete_run(self):
        if subprocess.run(["pgrep", "-f", "ar_train|ar_lora_"], capture_output=True).returncode == 0:
            return self._fail("stop training first — refusing to delete under a live run", "1")
        c = self._fields()
        if c is None:
            return self._fail("bad request", "1")
        name = clean_name(c.get("name", ""))
        base = os.path.join(RUNS, name)
        if not name or not os.path.isdir(base):
            return self._fail("unknown run", "1")
        import shutil
        shutil.rmtree(base, ignore_errors=True)
        shutil.rmtree(os.path.join("/workspace/tok/full", name), ignore_errors=True)
        for f in glob.glob(os.path.join(GEN, name + "_s*")):
            try:
                os.remove(f)
            except Exception:
                pass
        for link in ("/workspace/real/artist", "/workspace/real/artist_lyrics"):
            if os.path.islink(link) and name in os.readlink(link):
                try:
                    os.remove(link)
                except Exception:
                    pass
        reg = _registry()
        try:
            del reg.get("runs", {})[name]
        except KeyError:
            pass
        if reg.get("active") == name:
            rest = [d for d in os.listdir(RUNS) if os.path.isdir(os.path.join(RUNS, d))] if os.path.isdir(RUNS) else []
            if rest:
                reg["active"] = sorted(rest)[0]
            else:
                reg.pop("active", None)
        _save_registry(reg)
        return self._ok({"ok": True, "msg": "deleted run " + name + " (songs, checkpoints, samples)"}, "1")
    def _json(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

http.server.ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
