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
AUDIO_EXTS = (".flac", ".wav", ".ogg", ".mp3", ".m4a")
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
pre{background:#000;padding:10px;border-radius:8px;overflow:auto;max-height:220px;font-size:12px}.tabs{display:flex;gap:8px;margin:12px 0;position:sticky;top:0;background:#111;padding:8px 0;z-index:5}.tab{flex:1;padding:12px;border-radius:8px;border:1px solid #444;background:#1c1c1c;color:#eee;font-size:15px;text-align:center;cursor:pointer}.tab.on{background:#7c3aed;border-color:#7c3aed}.tabradio{position:absolute;opacity:0;pointer-events:none}#pane1,#pane2,#pane3{display:none}#t1:checked~#pane1,#t2:checked~#pane2,#t3:checked~#pane3{display:block}#t1:checked~.tabs label[for="t1"],#t2:checked~.tabs label[for="t2"],#t3:checked~.tabs label[for="t3"]{background:#7c3aed;border-color:#7c3aed}</style></head>
<body><h2>&#127926; FORGETITLE</h2>
<input type="radio" name="ftab" id="t1" class="tabradio" checked><input type="radio" name="ftab" id="t2" class="tabradio"><input type="radio" name="ftab" id="t3" class="tabradio">
<div class="tabs"><label for="t1" id="tb1" class="tab on" onclick="tab(1)">1 · Runs</label><label for="t2" id="tb2" class="tab" onclick="tab(2)">2 · Dataset studio</label><label for="t3" id="tb3" class="tab" onclick="tab(3)">3 · Training</label></div>
<div id="pane1"><h3>Runs</h3>
<div id="runs"></div>
<div class="ckpt">new run<br>name <input id="nr_name" placeholder="artist_name" style="width:180px;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px"> trigger <input id="nr_trig" placeholder="oneword or empty" style="width:160px;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px"> <button onclick="mkRun()" style="padding:8px 16px;border-radius:6px;border:0;background:#7c3aed;color:#fff">Create + switch</button> <span id="nr_msg" class="muted"></span></div>
</div>
<div id="pane2"><h3>Dataset studio <span class="muted" id="ds_run"></span></h3>
<div class="ckpt">trigger word — empty means caption-only mode (style bleeds into everything)<br><input id="ds_trig" style="width:200px;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px"> <button onclick="saveTrig()" style="padding:8px 16px;border-radius:6px;border:0;background:#7c3aed;color:#fff">Save</button> <span id="trig_msg" class="muted"></span><br><span class="muted" id="ds_count"></span></div>
<div class="ckpt">new song audio — pick many at once (wav/flac/ogg/mp3/m4a, each converts to flac)<br><input type="file" id="up_file" multiple accept="audio/*,.wav,.flac,.ogg,.mp3,.m4a"> name (single file only) <input id="up_name" placeholder="song_name" style="width:180px;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px"> <button onclick="upAudio()" style="padding:8px 16px;border-radius:6px;border:0;background:#7c3aed;color:#fff">Upload</button> <span id="up_msg" class="muted"></span></div>
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
<div class="ckpt">training — active run only<br>from <select id="tr_init" style="background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px"><option value="fresh">fresh</option><option value="last">last.pt</option><option value="best">best.pt</option></select> to step <input id="tr_steps" type="number" value="1600" style="width:90px;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px"> <button onclick="startTrain()" style="padding:8px 16px;border-radius:6px;border:0;background:#22c55e;color:#000">Start training</button> <span id="tr_msg" class="muted"></span><br><span class="muted">needs 10+ ready songs + dataset prepped (finish songs above, then prep via scripts/run_all.sh steps 1-3)</span></div>
<h3>Samples (your custom prompt below)</h3>
<!--STATIC_SAMPLES-->
<div id="samples"></div>
<h3>Next sample prompt</h3>
<div class="ckpt">
Style<br><input id="f_style" value="CFGSTYLE" style="width:100%;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px"><br><br>
Lyrics<br><textarea id="f_lyr" rows="9" style="width:100%;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px">CFGLYRICS</textarea><br><br>
Seed <input id="f_seed" value="CFGSEED" type="number" style="width:100px;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px">
<button onclick="saveCfg()" style="padding:8px 16px;border-radius:6px;border:0;background:#7c3aed;color:#fff">Save</button>
<span id="saved" class="muted"></span></div>
<div id="ckpts"></div>
<h3>Downloads</h3><!--STATIC_FILES--><div id="dl"></div>
<h3>Log tail</h3><pre id="log">STATICLOG</pre>
<div class="muted">V1</div>
</div>
<script>
let dirty=false;for(const id of ['f_style','f_lyr','f_seed']){document.getElementById(id).addEventListener('input',()=>dirty=true);}
let trigDirty=false,songsDirty=false;
document.getElementById('ds_trig').addEventListener('input',()=>trigDirty=true);
async function saveTrig(){const r=await fetch('/set_trigger',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({trigger:document.getElementById('ds_trig').value})});const d=await r.json();document.getElementById('trig_msg').textContent=d.ok?('saved: '+(d.trigger||'(caption-only mode)')):('error: '+d.error);trigDirty=false;tick();}
async function saveSong(n){const st=document.getElementById('st_'+n).value,ly=document.getElementById('ly_'+n).value,tg=document.getElementById('tg_'+n).value;
const r=await fetch('/save_song',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:n,style:st,lyrics:ly,trigger:tg})});const d=await r.json();
document.getElementById('msg_'+n).textContent=d.ok?('saved'+(d.issues&&d.issues.length?' — still: '+d.issues.join('; '):' — ready ✔')):('error: '+d.error);songsDirty=false;tick();}
async function delSong(n){if(!confirm('delete '+n+'?'))return;await fetch('/delete_song',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:n})});songsDirty=false;tick();}
async function mkRun(){const m=document.getElementById('nr_msg');
const r=await fetch('/create_run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:document.getElementById('nr_name').value,trigger:document.getElementById('nr_trig').value})});const d=await r.json();
m.textContent=d.ok?('created + active: '+d.run):('error: '+d.error);tick();}
async function switchRun(n){await fetch('/switch_run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:n})});tick();}
async function startTrain(){const m=document.getElementById('tr_msg');m.textContent='launching...';
const r=await fetch('/start_training',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({init:document.getElementById('tr_init').value,steps:parseInt(document.getElementById('tr_steps').value||'1600')})});const d=await r.json();
m.textContent=d.ok?('training '+d.run+' '+d.from_step+'→'+d.to_step):('error: '+d.error);tick();}
async function upAudio(){const fs=document.getElementById('up_file').files;const m=document.getElementById('up_msg');
if(!fs.length){m.textContent='pick files first';return;}
const only=document.getElementById('up_name').value;let ok=0;
for(let i=0;i<fs.length;i++){const f=fs[i];
const n=(fs.length===1&&only)?only:f.name.replace(/\.[^.]+$/,'');
m.textContent='uploading '+(i+1)+'/'+fs.length+': '+f.name+'...';
const fd=new FormData();fd.append('audio',f,f.name);
try{const r=await fetch('/upload_audio?name='+encodeURIComponent(n),{method:'POST',body:fd});const d=await r.json();
if(d.ok)ok++;else{m.textContent='error on '+f.name+': '+d.error;break;}}catch(e){m.textContent='upload failed: '+e.message;break;}}
m.textContent+=' — done '+ok+'/'+fs.length+' — now add captions + lyrics below';
document.getElementById('up_file').value='';document.getElementById('up_name').value='';tick();}
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
async function saveCfg(){const b={style:document.getElementById('f_style').value,lyrics:document.getElementById('f_lyr').value,seed:parseInt(document.getElementById('f_seed').value||'12')};
const r=await fetch('/save_cfg',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});const d=await r.json();
document.getElementById('saved').textContent=d.ok?'saved - applies from next sample':'error: '+(d.error||'unknown');dirty=false;}
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
r+='<div class="ckpt">'+(act?'<b>'+x.name+' (active)</b>': '<b>'+x.name+'</b> <button onclick="switchRun(\''+x.name+'\')" style="padding:4px 12px;border-radius:6px;border:1px solid #666;background:#222;color:#eee">Switch</button>')+' <span class="muted">'+x.ready+'/'+x.total+' songs'+(x.ckpts.length?' | ckpts '+x.ckpts.join(','):'')+(x.best?' | best ✔':'')+'</span></div>';}
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
if(!trigDirty)document.getElementById('ds_trig').value=d.studio.trigger||'';
if(!songsDirty){let q='';if(!d.studio.songs.length)q='<div class="muted">no songs yet — upload audio above, then add caption + lyrics per song</div>';
for(const x of d.studio.songs){const ok=x.issues.length===0;
const esc=s=>String(s).replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;');
const aud=x.audio?'<audio controls preload="none" style="width:100%" src="/a/'+d.studio.run+'/'+x.audio.file+'"></audio><br>':'';
q+='<div class="ckpt"><b>'+x.name+'</b> '+(x.audio?'<span class="muted">'+x.audio.mb+' MB flac</span>':'<span class="muted">no audio</span>')+' '+(ok?'\u2714 ready':'<span style="color:#f59e0b">'+x.issues.join('; ')+'</span>')+'<br>'+aud+'trigger (empty = run default)<br><input id="tg_'+x.name+'" oninput="songsDirty=true" value="'+esc(x.trig)+'" placeholder="run default: '+esc(d.studio.trigger||'(none)')+'" style="width:220px;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px"><br>style / caption<br><input id="st_'+x.name+'" oninput="songsDirty=true" value="'+esc(x.style)+'" style="width:100%;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px"><br>lyrics<br><textarea id="ly_'+x.name+'" oninput="songsDirty=true" rows="6" style="width:100%;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px">'+esc(x.lyrics)+'</textarea><br><button onclick="saveSong(\''+x.name+'\')" style="padding:6px 14px;border-radius:6px;border:0;background:#7c3aed;color:#fff">Save</button> <button onclick="delSong(\''+x.name+'\')" style="padding:6px 14px;border-radius:6px;border:1px solid #666;background:#222;color:#eee">Delete</button> <span id="msg_'+x.name+'" class="muted"></span></div>';}
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
                    "ckpts": ckpts, "best": os.path.exists(os.path.join(ckd, "best.pt"))})
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
                     "mp3": "audio/mpeg", "m4a": "audio/mp4"}.get(os.path.splitext(fn)[1].lower()[1:], "audio/mpeg")
            self._send(p, ctype)
        elif self.path.startswith("/m/"):
            fn = os.path.basename(self.path[3:])
            if ".." in fn or not re.match(r"[A-Za-z0-9_]+_s\d+\.(mp3|flac)$", fn):
                self.send_error(404)
                return
            self._send(os.path.join(GEN, fn),
                         "audio/mpeg" if fn.endswith(".mp3") else "audio/flac")
        else:
            import time as _t, html as _h
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
                sg += f"<div class='ckpt'><b>{x['name']}</b> <span class='muted'>{x['lines']} lines, {x['audio']['mb'] if x['audio'] else 0} MB</span><br>{player}<span class='muted'>{flag}</span></div>"
            b = HTML.replace("<!--STATIC_STATUS-->", st).replace("<!--STATIC_SAMPLES-->", ss or "<div class='muted'>no samples yet</div>").replace("<!--STATIC_FILES-->", ff).replace("FILLPCT", str(d["pct"])).replace("<!--STATIC_SONGS-->", sg or "<div class='muted'>no songs yet</div>")
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
        self.send_error(404)
    def _body(self, limit):
        try:
            n = int(self.headers.get("Content-Length", 0))
        except Exception:
            n = 0
        if n <= 0 or n > limit:
            return None
        return self.rfile.read(n)
    def _post_save_cfg(self):
        raw = self._body(20000)
        if raw is None:
            self._json({"ok": False, "error": "bad size"})
            return
        try:
            c = json.loads(raw)
            style = str(c.get("style", "")).strip()[:1500]
            lyrics = str(c.get("lyrics", "")).strip()[:8000]
            seed = int(c.get("seed", 12))
            if not style or not lyrics:
                raise ValueError("style and lyrics required")
            if not re.search(r"[a-z\[]", lyrics, re.I):
                raise ValueError("lyrics look empty")
        except Exception as e:
            self._json({"ok": False, "error": str(e)[:120]})
            return
        open(STYLE_FILE, "w").write(style + "\n")
        open(LYR_FILE, "w").write(lyrics + "\n")
        json.dump({"seed": seed}, open(CFG, "w"))
        self._json({"ok": True})
    def _post_save_song(self):
        raw = self._body(60000)
        if raw is None:
            self._json({"ok": False, "error": "bad size"})
            return
        try:
            c = json.loads(raw)
            name = clean_name(c.get("name", ""))
            style = str(c.get("style", "")).strip()[:1500]
            lyrics = str(c.get("lyrics", "")).strip()[:12000]
            strig = re.sub(r"[^a-z0-9]+", "", str(c.get("trigger", "")).strip().lower())[:32] if "trigger" in c else None
            if not name:
                raise ValueError("song name required (letters, numbers, _)")
            if not style or not lyrics:
                raise ValueError("style and lyrics required")
        except Exception as e:
            self._json({"ok": False, "error": str(e)[:120]})
            return
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
        self._json({"ok": True, "issues": v["issues"]})
    def _post_delete_song(self):
        raw = self._body(2000)
        if raw is None:
            self._json({"ok": False, "error": "bad size"})
            return
        try:
            name = clean_name(json.loads(raw).get("name", ""))
            if not name:
                raise ValueError("song name required")
        except Exception as e:
            self._json({"ok": False, "error": str(e)[:120]})
            return
        _, ad, ald = run_paths()
        for d in (ad, ald):
            for f in glob.glob(os.path.join(d, name + ".*")):
                os.remove(f)
        self._json({"ok": True})
    def _post_set_trigger(self):
        raw = self._body(2000)
        if raw is None:
            self._json({"ok": False, "error": "bad size"})
            return
        try:
            trig = re.sub(r"[^a-z0-9]+", "", str(json.loads(raw).get("trigger", "")).strip().lower())[:32]
        except Exception as e:
            self._json({"ok": False, "error": str(e)[:120]})
            return
        base, _, _ = run_paths()
        os.makedirs(base, exist_ok=True)
        cfgp = os.path.join(base, "config.json")
        cfg = {}
        try:
            cfg = json.load(open(cfgp))
        except Exception:
            pass
        cfg["trigger"] = trig
        json.dump(cfg, open(cfgp, "w"))
        self._json({"ok": True, "trigger": trig})
    def _post_upload_audio(self):
        ctype = self.headers.get("Content-Type", "")
        m = re.match(r"multipart/form-data; boundary=(.+)$", ctype)
        if not m:
            self._json({"ok": False, "error": "need multipart upload"})
            return
        raw = self._body(400 * 1024 * 1024)
        if raw is None:
            self._json({"ok": False, "error": "file too big (400MB max)"})
            return
        try:
            qs = self.path.split("?", 1)[1] if "?" in self.path else ""
            name = clean_name(re.findall(r"(?:^|&)name=([^&]*)", qs)[0] if re.findall(r"(?:^|&)name=([^&]*)", qs) else "")
            if not name:
                raise ValueError("song name required (?name=...)")
            bound = ("--" + m.group(1).strip().strip('"')).encode()
            parts = raw.split(bound)
            blob, fname = None, "upload.bin"
            for p in parts:
                if b'name="audio"' in p.split(b"\r\n\r\n", 1)[0]:
                    head, blob = p.split(b"\r\n\r\n", 1)
                    blob = blob.rsplit(b"\r\n", 1)[0]
                    fm = re.search(rb'filename="([^"]+)"', head)
                    if fm:
                        fname = fm.group(1).decode("utf-8", "replace")
                    break
            if not blob:
                raise ValueError("no audio file field")
            ext = os.path.splitext(fname)[1].lower()
            if ext not in AUDIO_EXTS:
                raise ValueError(f"audio type {ext or '?'} not accepted (wav/flac/ogg/mp3/m4a)")
        except Exception as e:
            self._json({"ok": False, "error": str(e)[:120]})
            return
        _, ad, _ = run_paths()
        os.makedirs(ad, exist_ok=True)
        tmp = os.path.join(ad, name + ".incoming" + ext)
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
            self._json({"ok": False, "error": f"audio convert failed: {str(e)[:100]}"})
            return
        try:
            os.remove(tmp)
        except Exception:
            pass
        if r.returncode != 0 or not os.path.exists(out):
            err = (r.stderr or b"").decode("utf-8", "replace")[-200:]
            self._json({"ok": False, "error": f"ffmpeg rejected it ({err or 'unknown'})"})
            return
        self._json({"ok": True, "mb": round(os.path.getsize(out) / 2**20, 1)})
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
        raw = self._body(4000)
        if raw is None:
            self._json({"ok": False, "error": "bad size"})
            return
        try:
            c = json.loads(raw)
            name = clean_name(c.get("name", ""))
            trig = re.sub(r"[^a-z0-9]+", "", str(c.get("trigger", "")).strip().lower())[:32]
            if not name:
                raise ValueError("run name required (letters, numbers, _)")
            base = os.path.join(RUNS, name)
            if os.path.exists(base):
                raise ValueError("run already exists — switch to it instead")
        except Exception as e:
            self._json({"ok": False, "error": str(e)[:120]})
            return
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
            self._json({"ok": False, "error": str(e)[:150]})
            return
        self._json({"ok": True, "run": name})
    def _post_switch_run(self):
        raw = self._body(2000)
        if raw is None:
            self._json({"ok": False, "error": "bad size"})
            return
        try:
            name = clean_name(json.loads(raw).get("name", ""))
            base = os.path.join(RUNS, name)
            if not name or not os.path.isdir(base):
                raise ValueError("unknown run")
        except Exception as e:
            self._json({"ok": False, "error": str(e)[:120]})
            return
        try:
            self._repoint("/workspace/real/artist", os.path.join(base, "artist"))
            self._repoint("/workspace/real/artist_lyrics", os.path.join(base, "artist_lyrics"))
        except Exception as e:
            self._json({"ok": False, "error": str(e)[:150]})
            return
        reg = _registry()
        reg["active"] = name
        _save_registry(reg)
        self._json({"ok": True, "run": name})
    def _post_start_training(self):
        if subprocess.run(["pgrep", "-f", "ar_train|ar_lora_"], capture_output=True).returncode == 0:
            self._json({"ok": False, "error": "training already running"})
            return
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
            self._json({"ok": False, "error": f"need {MIN_SONGS}+ ready songs (have {ready}) — small sets memorize instead of learning style"})
            return
        raw = self._body(4000)
        if raw is None:
            self._json({"ok": False, "error": "bad size"})
            return
        try:
            c = json.loads(raw) if raw else {}
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
            self._json({"ok": False, "error": str(e)[:150]})
            return
        repo = os.environ.get("FORGE_REPO", "/workspace/yue2-forge")
        train_py = os.path.join(repo, "scripts", "ar_train.py")
        if not os.path.exists(train_py):
            self._json({"ok": False, "error": "trainer script missing on server"})
            return
        env = dict(os.environ, HF_HOME="/workspace/hf", SCHED_STEPS="3000",
                   CK_FROM="600", CK_EVERY="200", START_STEP=str(start))
        log = open("/workspace/ar_train.log", "a")
        subprocess.Popen(["/workspace/yue2venv/bin/python", "-u", train_py, name,
                          str(total - start), "64", "0.5", initpt, "1e-4", "0.08"],
                         stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True, env=env)
        self._json({"ok": True, "run": name, "from_step": start, "to_step": total})
    def _json(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

http.server.ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
