"""YuE2 LoRA Training | @directedbykobyperez"""
import http.server, json, os, re, glob, subprocess, time
try:
    import torch
except ImportError:
    torch = None

PORT = int(os.environ.get("PORT", 8000))
OUT = os.environ.get("FORGE_OUT", "/workspace/tok/full/my_lora")
GEN = os.environ.get("FORGE_GEN", "/workspace/tok/full/gen")
LOG = os.environ.get("FORGE_LOG", "/workspace/ar_train.log")
STATE = os.environ.get("FORGE_STATE", "/workspace/ui/state.json")
CFG = os.environ.get("FORGE_CFG", "/workspace/sample_cfg.json")
STYLE_FILE = "/workspace/real/artist/sample.txt"
LYR_FILE = "/workspace/sample_lyrics.txt"
TOTAL = int(os.environ.get("FORGE_TOTAL", "800"))
RUNS = os.environ.get("FORGE_RUNS", "/workspace/runs")
AUDIO_EXTS = (".flac", ".wav", ".ogg", ".mp3", ".m4a", ".webm")
MIN_SONGS = 7
DL = {
    "ckpt": (OUT, r"^(step-\d+|best|last)\.pt$"),
    "log": ("/workspace", r"^(ar_train|gen|watcher|watcher_out|prep_real|cursor_prep2?|ar_prep)\.log$"),
    "runlog": (OUT, r"^train\.log$"),
    "data": ("/workspace/real/ar", r"^dataset\.pt$"),
}

HTML = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
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
pre{background:#000;padding:10px;border-radius:8px;overflow:auto;max-height:220px;font-size:12px}.tabs{display:flex;gap:8px;margin:12px 0;position:sticky;top:0;background:#111;padding:8px 0;z-index:5}.tab{flex:1;padding:12px;border-radius:8px;border:1px solid #444;background:#1c1c1c;color:#eee;font-size:15px;text-align:center;cursor:pointer;text-decoration:none;display:block}.tab.on{background:#7c3aed;border-color:#7c3aed}
@media(max-width:640px){body{padding:12px}input,textarea,select{max-width:100%!important;box-sizing:border-box}button{margin:6px 4px 6px 0}.grid{grid-template-columns:1fr 1fr}.tab{font-size:13px;padding:10px 4px}}.hdr{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap}.hdr h2{margin:0 0 8px}#gpu{text-align:right;font-size:12px;color:#888;line-height:1.6}#gpu b{color:#ccc}.pill{display:inline-flex;align-items:center;gap:7px;font-size:12px;color:#888;background:#1c1c1c;border:1px solid #333;border-radius:20px;padding:6px 12px}.dot{width:8px;height:8px;border-radius:50%;background:#22c55e}.ptools{display:flex;align-items:center;gap:10px;margin:8px 0}.ptools .padd{margin-left:auto;padding:6px 12px;border-radius:6px;border:1px solid #444;background:#1c1c1c;color:#ddd;font-size:13px;cursor:pointer}.ptools .padd:hover{background:#262626}.pcard{background:#161616;border:1px solid #2C2C2C;border-radius:10px;margin:8px 0;overflow:hidden}.pcard.collapsed .pbody{display:none}.phead{display:flex;align-items:center;gap:8px;padding:9px 12px;background:#1c1c1c;cursor:pointer;user-select:none}.phead b{font-size:13px;flex:1;margin:0}.pcard.collapsed .phead b::after{content:" — click to expand";color:#888;font-weight:400;font-size:12px}.phead .x{color:#f87171;font-size:16px;line-height:1;border:0;background:none;cursor:pointer;padding:2px 6px;border-radius:4px}.phead .x:hover{background:#3a1414}.pbody{padding:10px 12px 12px}.pbody input,.pbody textarea{width:100%;box-sizing:border-box;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px;font-family:inherit}.pbody .seedbox{width:110px}.plabel{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:#888;margin:10px 0 4px}.plabel:first-child{margin-top:0}</style></head>
<body><div class="hdr"><h2>&#127926; FORGETITLE</h2><div id="gpu">
<!--STATIC_GPU--></div></div>
<!--MSG-->
<div class="tabs"><div class="tab on" data-tab="1">1 · Runs</div><div class="tab" data-tab="2">2 · Dataset studio</div><div class="tab" data-tab="3">3 · Training</div><div class="tab" data-tab="4">4 · Logs</div><div class="tab" data-tab="5">5 · Generate</div></div>
<div id="pane1" class="pane"><h3>Runs</h3>
<div id="runs"></div><!--STATIC_RUNS-->
<form method="POST" action="/create_run"><div class="ckpt">new: <input name="name" placeholder="artist_name" style="width:180px;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px"> <button style="padding:8px 16px;border-radius:6px;border:0;background:#7c3aed;color:#fff">Create</button> <span class="muted">then set its trigger below</span></div></form>
</div>
<div id="pane2" class="pane" style="display:none"><h3>Dataset studio <span class="muted" id="ds_run"></span></h3>
<div class="ckpt"><span class="muted" id="ds_count"></span></div>
<div class="ckpt" style="background:#1a1a2e;border:1px solid #333;border-radius:8px;padding:10px;margin-bottom:8px"><b>Format guide</b> <span class="muted">(per song):</span><br><pre style="background:#000;border-radius:6px;padding:8px;margin:6px 0;font-size:12px;color:#22d3ee">a dreamy pop ballad about sunset love
[Verse 1]
walking down the golden shore
waves crashing at my feet

[Chorus]
oh the sunset glow
lighting up our souls

[Bridge]
time stands still tonight</pre><span class="muted">First line = caption/style. Then [Verse], [Chorus], [Bridge] sections with lyrics. Upload .txt files alongside audio to auto-fill.</span></div>
<form method="POST" action="/upload_audio" enctype="multipart/form-data"><div class="ckpt">Upload audio (WAV/FLAC/OGG/MP3/M4A/WebM) + matching .txt files.<br><span class="muted">.txt format: first line = caption, then [Verse]/[Chorus] lyrics (see below)</span><br><input type="file" name="audio" multiple accept="audio/*,.wav,.flac,.ogg,.mp3,.m4a,.webm,.txt"> <button style="padding:8px 16px;border-radius:6px;border:0;background:#7c3aed;color:#fff">Upload</button></div></form>
<div class="ckpt">
<label class="muted">Import from HuggingFace</label><br>
<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:6px">
<input id="hf_token" type="password" placeholder="HF token (hf_...)" style="flex:1;min-width:180px;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px">
<input id="hf_repo" type="text" placeholder="repo: user/dataset-name" style="flex:1;min-width:180px;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px">
<button onclick="hfList()" style="padding:8px 16px;border-radius:6px;border:0;background:#7c3aed;color:#fff">HuggingFace</button>
</div>
<div id="hf_status" class="muted" style="font-size:12px;margin-bottom:4px"></div>
<div id="hf_files" style="max-height:300px;overflow-y:auto"></div>
</div>
<div id="songs"></div>
</div>
<div id="pane3" class="pane" style="display:none"><!--STATIC_STATUS-->
<iframe src="/live-mini" style="width:100%;height:118px;border:1px solid #2C2C2C;border-radius:10px;overflow:hidden" scrolling="no" title="live progress"></iframe>
<h3>Loss graph</h3>
<div style="display:flex;align-items:center;gap:12px;margin:6px 0;flex-wrap:wrap">
<label class="muted" style="font-size:12px"><input type="checkbox" id="lg_loss" checked onchange="renderLossChart()"> loss</label>
<label class="muted" style="font-size:12px"><input type="checkbox" id="lg_artist" checked onchange="renderLossChart()"> artist</label>
<label class="muted" style="font-size:12px"><input type="checkbox" id="lg_minted" checked onchange="renderLossChart()"> minted</label>
<span class="muted" style="font-size:12px">smooth:</span>
<input type="range" id="lg_smooth" min="1" max="50" value="5" style="width:80px;accent-color:#22d3ee" oninput="renderLossChart()">
<label class="muted" style="font-size:12px"><input type="checkbox" id="lg_log" onchange="renderLossChart()"> Log Y</label>
</div>
<div style="background:#1c1c1c;border-radius:8px;padding:8px;margin-bottom:8px"><canvas id="lossCanvas" height="160"></canvas></div><!--LOSSGRAPH-->
<div class="bar"><div class="fill" id="fill" style="width:FILLPCT%"></div></div>
<div id="pct" class="muted"></div>


<div class="grid">
<div class="card">step<div><b id="step">-</b> / 800</div></div>
<div class="card">phase<div><b id="phase">-</b></div></div>
<div class="card">loss<div><b id="loss">-</b></div></div>
<div class="card">speed<div><b id="speed">-</b></div></div>
<div class="card">artist eval<div><b id="eval">-</b></div></div>
<div class="card">minted_val eval<div><b id="mval">-</b></div></div>
<div class="card">ETA<div><b id="eta">-</b></div></div>

<div class="ckpt">dataset prep — features + stems + tokens (10–40 min by dataset size, runs in background)<br><form method="POST" action="/prepare_dataset"><button style="padding:8px 16px;border-radius:6px;border:0;background:#f59e0b;color:#000">Prepare dataset</button></form> <span class="muted" id="prep_line"></span></div>
<!--STATIC_PREP-->
<form method="POST" action="/start_training"><div class="ckpt">training — active run only<br>
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:8px">
<div><label class="muted">model precision</label><br><select name="vram_mode" style="background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px;width:100%"><option value="low">INT8 (low VRAM, 16GB+)</option><option value="high">bf16 (full, 22GB+)</option></select></div>
<div><label class="muted">cot mode</label><br><select name="cot" style="background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px;width:100%"><option value="off">off (no ABC sheet)</option><option value="full">full (chords + melody)</option><option value="melody">melody only</option></select></div>
<div><label class="muted">tokenizer</label><br><select name="tokenizer" style="background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px;width:100%"><option value="community">Community (Mothersuperior)</option><option value="mert">MERT (fallback)</option></select></div>
</div>
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:8px">
<div><label class="muted">ar_kl_weight</label><br><input name="ar_kl_weight" type="number" step="0.01" value="0.04" style="width:100%;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px"></div>
<div><label class="muted">ar_lr_multiplier</label><br><input name="ar_lr_multiplier" type="number" step="0.1" value="1.0" style="width:100%;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px"></div>
<div><label class="muted">ar_repetition_penalty</label><br><input name="sample_ar_repetition_penalty" type="number" step="0.1" value="1.2" style="width:100%;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px"></div>
</div>
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:8px">
<div><label class="muted">abc_dropout</label><br><input name="abc_dropout" type="number" step="0.1" value="0.5" style="width:100%;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px"></div>
<div><label class="muted">train_window</label><br><input name="train_window" type="number" value="1500" style="width:100%;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px"></div>
<div><label class="muted">ema_decay</label><br><input name="ema_decay" type="number" step="0.01" value="0.99" style="width:100%;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px"></div>
</div>
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:8px">
<div><label class="muted">learning_rate</label><br><input name="lr" type="number" step="0.00001" value="0.0001" style="width:100%;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px"></div>
<div><label class="muted">weight_decay</label><br><input name="weight_decay" type="number" step="0.0001" value="0.0001" style="width:100%;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px"></div>
<div><label class="muted">save_every</label><br><input name="save_every" type="number" value="250" style="width:100%;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px"></div>
</div>
from <select name="init" style="background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px"><option value="fresh">fresh</option><option value="last">last.pt</option><option value="best">best.pt</option></select> rank <select name="rank"><option value="16">16</option><option value="32">32</option><option value="64" selected>64</option><option value="128">128</option></select> to step <input name="steps" type="number" value="800" style="width:90px;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px">
<button style="padding:8px 16px;border-radius:6px;border:0;background:#22c55e;color:#000">Start training</button>
<button type="button" onclick="resumeTrain()" style="padding:8px 16px;border-radius:6px;border:0;background:#f59e0b;color:#000;margin-left:8px">Resume training</button><br><span class="muted">needs 7+ ready songs + dataset prepped (finish songs above, then prep via scripts/run_all.sh steps 1-3)</span>
<div id="resume_info" style="display:none;margin-top:8px;padding:8px;background:#1c1c1c;border:1px solid #333;border-radius:6px;font-size:12px"></div>
<script>
async function resumeTrain(){
  const info=document.getElementById('resume_info');info.style.display='block';info.innerHTML='checking last.pt...';
  try{
    const r=await fetch('/resume_info');const d=await r.json();
    if(!d.exists){info.innerHTML='<span style="color:#f59e0b">no last.pt found — start training first</span>';return;}
    info.innerHTML=`last.pt: step ${d.step} | best loss: ${d.best_loss} | rank ${d.rank}<br>
    <form method="POST" action="/start_training" style="display:inline">
    <input type="hidden" name="init" value="last"><input type="hidden" name="rank" value="${d.rank}">
    <input type="hidden" name="steps" value="${d.target_steps}">
    <input type="hidden" name="ar_kl_weight" value="0.04"><input type="hidden" name="ar_lr_multiplier" value="1.0">
    <input type="hidden" name="ar_repetition_penalty" value="1.2"><input type="hidden" name="abc_dropout" value="0.5">
    <input type="hidden" name="train_window" value="1500"><input type="hidden" name="ema_decay" value="0.99">
    <input type="hidden" name="weight_decay" value="0.0001"><input type="hidden" name="lr" value="0.0001">
    <input type="hidden" name="save_every" value="250"><input type="hidden" name="cot" value="off">
    <input type="hidden" name="tokenizer" value="community"><input type="hidden" name="sheetsage_task" value="full">
    <input type="hidden" name="vram_mode" value="low">
    <button style="padding:6px 12px;border-radius:5px;border:0;background:#22c55e;color:#000;margin-top:6px">confirm resume from step ${d.step}</button>
    </form>`;
  }catch(e){info.innerHTML='<span style="color:#f87171">error: '+e.message+'</span>';}
}
</script></div></form>
<h3>Samples (your custom prompt below)</h3>
<!--STATIC_SAMPLES-->
<div id="samples"></div>
<h3>Sample prompts</h3>
<form method="POST" action="/save_cfg">
<!--STATIC_PROMPTS-->
<div class="ckpt"><label class="muted"><input type="checkbox" name="walk" value="1"WALKCHECKED> walk seed per checkpoint</label>
<button style="padding:8px 16px;border-radius:6px;border:0;background:#7c3aed;color:#fff">Save prompts</button></div></form>
<div id="ckpts"></div>
<h3>Downloads</h3><!--STATIC_FILES--><div id="dl"></div>


</div>
</div>
<div id="pane4" class="pane" style="display:none"><h3>Log tail</h3><pre id="log">STATICLOG</pre></div>
<div id="pane5" class="pane" style="display:none">
<h3>Generate songs</h3>
<div class="ckpt">
<label class="muted">Select checkpoint</label><br>
<select id="gen_ckpt" style="background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px;width:100%">
<!--GEN_CKPTS-->
</select>
</div>
<div class="ckpt">
<label class="muted">Model precision</label><br>
<select id="gen_precision" style="background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px;width:100%">
<option value="bf16">BF16 (faster, needs more VRAM)</option>
<option value="int8">INT8 (lower VRAM usage)</option>
</select>
</div>
<div class="ckpt">
<label class="muted">Style (caption)</label><br>
<textarea id="gen_style" rows="3" style="width:100%;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px;font-family:monospace" placeholder="In the style of Artist Name, genre tags..."></textarea>
</div>
<div class="ckpt">
<label class="muted">Lyrics</label><br>
<textarea id="gen_lyrics" rows="6" style="width:100%;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px;font-family:monospace" placeholder="[Verse]
Your lyrics here...
[Chorus]
More lyrics..."></textarea>
</div>
<div class="ckpt">
<label class="muted">Seed</label><br>
<input id="gen_seed" type="number" value="12" style="width:120px;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px">
</div>
<div class="grid">
<div class="ckpt">
<label class="muted">CFG Scale</label><br>
<input id="gen_cfg" type="number" value="1.0" step="0.1" min="1.0" max="20.0" style="width:80px;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px">
<span class="muted">1.0 = no guidance</span>
</div>
<div class="ckpt">
<label class="muted">CoT Mode</label><br>
<select id="gen_cot" style="background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px;width:100%">
<option value="off" selected>off (default)</option>
<option value="melody">melody</option>
<option value="full">full</option>
</select>
</div>
</div>
<div class="grid">
<div class="ckpt">
<label class="muted">Temperature</label><br>
<input id="gen_temp" type="number" value="1.0" step="0.05" min="0.1" max="2.0" style="width:80px;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px">
</div>
<div class="ckpt">
<label class="muted">Top P</label><br>
<input id="gen_top_p" type="number" value="0.95" step="0.05" min="0.1" max="1.0" style="width:80px;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px">
</div>
</div>
<div class="grid">
<div class="ckpt">
<label class="muted">Top K</label><br>
<input id="gen_top_k" type="number" value="50" step="1" min="0" max="200" style="width:80px;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px">
</div>
<div class="ckpt">
<label class="muted">Rep. Penalty</label><br>
<input id="gen_rep_pen" type="number" value="1.2" step="0.05" min="1.0" max="3.0" style="width:80px;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px">
</div>
</div>
<div class="ckpt">
<button onclick="startGenerate()" style="padding:8px 16px;border-radius:6px;border:0;background:#7c3aed;color:#fff">Generate</button>
<span id="gen_status" class="muted" style="margin-left:8px"></span>
</div>
<div id="gen_progress" style="display:none" class="ckpt">
<label class="muted">Generating...</label><br>
<div class="bar"><div class="fill" id="gen_bar" style="width:0%"></div></div>
<span id="gen_eta" class="muted"></span>
</div>
<h3>Generated songs</h3>
<div id="gen_list"><!--GEN_LIST--></div>
<script>
async function loadGenCkpts(){
  try{
    const r=await fetch('/api');const d=await r.json();
    const sel=document.getElementById('gen_ckpt');
    sel.innerHTML='';
    const base=document.createElement('option');
    base.value='none';base.textContent='🎵 Base model (no LoRA)';
    sel.appendChild(base);
    if(d.checkpoints&&d.checkpoints.length){
      d.checkpoints.forEach(c=>{
        const o=document.createElement('option');
        o.value=c.pt;o.textContent='step-'+c.step+' ('+c.mb+' MB)';
        sel.appendChild(o);
      });
    }
  }catch(e){}
}
async function loadGenList(){
  try{
    const r=await fetch('/gen_list');const d=await r.json();
    const el=document.getElementById('gen_list');
    if(!d.files||!d.files.length){el.innerHTML='<div class="muted">no generated songs yet</div>';return;}
    el.innerHTML='';
    d.files.forEach(f=>{
      el.innerHTML+='<div class="ckpt" style="display:flex;align-items:center;gap:8px"><span style="flex:1">'+f.name+' <span class="muted">'+f.size+'</span></span><audio controls src="/gen/'+f.name+'" style="height:32px"></audio><a href="/gen/'+f.name+'" download style="color:#22d3ee">⬇</a></div>';
    });
  }catch(e){}
}
async function startGenerate(){
  const ckpt=document.getElementById('gen_ckpt').value;
  if(!ckpt){alert('select a checkpoint first');return;}
  const style=document.getElementById('gen_style').value;
  const lyrics=document.getElementById('gen_lyrics').value;
  const seed=document.getElementById('gen_seed').value;
  const precision=document.getElementById('gen_precision').value;
  const cfg=document.getElementById('gen_cfg').value;
  const cot=document.getElementById('gen_cot').value;
  const temp=document.getElementById('gen_temp').value;
  const top_p=document.getElementById('gen_top_p').value;
  const top_k=document.getElementById('gen_top_k').value;
  const rep_pen=document.getElementById('gen_rep_pen').value;
  document.getElementById('gen_progress').style.display='block';
  document.getElementById('gen_status').textContent='starting...';
  try{
    const r=await fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({ckpt,style,lyrics,seed:parseInt(seed)||12,precision,cfg:parseFloat(cfg)||1.0,cot,temp:parseFloat(temp)||1.0,top_p:parseFloat(top_p)||0.95,top_k:parseInt(top_k)||50,rep_pen:parseFloat(rep_pen)||1.2})});
    const d=await r.json();
    if(d.error){document.getElementById('gen_status').textContent='error: '+d.error;return;}
    document.getElementById('gen_status').textContent='generating... eta '+d.eta;
    pollGen(d.job_id);
  }catch(e){document.getElementById('gen_status').textContent='error: '+e.message;}
}
function pollGen(jobId){
  const iv=setInterval(async()=>{
    try{
      const r=await fetch('/gen_status?job='+jobId);const d=await r.json();
      if(d.done){
        clearInterval(iv);
        document.getElementById('gen_progress').style.display='none';
        document.getElementById('gen_status').textContent=d.error?'error: '+d.error:'done!';
        loadGenList();
      }else{
        document.getElementById('gen_bar').style.width=d.pct+'%';
        document.getElementById('gen_eta').textContent=d.status+' · '+d.eta;
      }
    }catch(e){}
  },3000);
}
loadGenCkpts();loadGenList();
</script>
</div>
<script>
let hfSelected=new Set();
async function hfList(){
  const token=document.getElementById('hf_token').value.trim();
  const repo=document.getElementById('hf_repo').value.trim();
  if(!repo){document.getElementById('hf_status').textContent='enter a repo name';return;}
  document.getElementById('hf_status').textContent='loading...';
  document.getElementById('hf_files').innerHTML='';
  hfSelected.clear();
  try{
    const r=await fetch('/hf_list',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token,repo})});
    const d=await r.json();
    if(d.error){document.getElementById('hf_status').textContent='error: '+d.error;return;}
    document.getElementById('hf_status').textContent=d.files.length+' files found';
    let html='<table style="width:100%;font-size:12px;border-collapse:collapse">';
    d.files.forEach((f,i)=>{
      const ext=f.name.split('.').pop().toLowerCase();
      const isAudio=['wav','flac','ogg','mp3','m4a','webm'].includes(ext);
      const isTxt=ext==='txt';
      const icon=isAudio?'🎵':isTxt?'📝':'📄';
      const canAdd=isAudio||isTxt;
      html+='<tr style="border-bottom:#333 1px solid">';
      html+='<td style="padding:4px">'+icon+' <span class="muted">'+f.name+'</span> <span class="muted">('+f.size+')</span></td>';
      if(canAdd) html+='<td style="padding:4px;text-align:right"><button onclick="hfToggle(this)" data-name="'+f.name+'" style="background:#22c55e;color:#fff;border:0;border-radius:4px;padding:2px 8px;cursor:pointer">+</button></td>';
      html+='</tr>';
    });
    html+='</table>';
    html+='<button onclick="hfImport()" style="margin-top:8px;padding:8px 16px;border-radius:6px;border:0;background:#7c3aed;color:#fff">Import selected to dataset</button>';
    document.getElementById('hf_files').innerHTML=html;
  }catch(e){document.getElementById('hf_status').textContent='error: '+e.message;}
}
function hfToggle(btn){
  const name=btn.dataset.name;
  if(hfSelected.has(name)){hfSelected.delete(name);btn.textContent='+';btn.style.background='#22c55e';}
  else{hfSelected.add(name);btn.textContent='\u2212';btn.style.background='#ef4444';}
  document.getElementById('hf_status').textContent=hfSelected.size+' selected';
}
async function hfImport(){
  if(!hfSelected.size){document.getElementById('hf_status').textContent='select files first';return;}
  const token=document.getElementById('hf_token').value.trim();
  const repo=document.getElementById('hf_repo').value.trim();
  document.getElementById('hf_status').textContent='importing '+hfSelected.size+' files...';
  try{
    const r=await fetch('/hf_import',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token,repo,files:Array.from(hfSelected)})});
    const d=await r.json();
    if(d.error){document.getElementById('hf_status').textContent='error: '+d.error;return;}
    document.getElementById('hf_status').textContent=d.msg;
    tick();
  }catch(e){document.getElementById('hf_status').textContent='error: '+e.message;}
}
let songsDirty=false;
var activeTab=localStorage.getItem('forge_tab')||'1';
function switchTab(t){activeTab=t;localStorage.setItem('forge_tab',t);document.querySelectorAll('.pane').forEach(function(p){p.style.display='none';});document.getElementById('pane'+t).style.display='';document.querySelectorAll('.tab').forEach(function(b){b.classList.toggle('on',b.dataset.tab===t);});}
document.addEventListener('DOMContentLoaded',function(){switchTab(activeTab);document.querySelectorAll('.tab').forEach(function(b){b.addEventListener('click',function(e){e.preventDefault();switchTab(this.dataset.tab);});});});
document.addEventListener('click',function(e){
var t=e.target;
if(t.classList.contains('play')){togglePlay(t.dataset.k,t.dataset.file,t);return;}
var a=t.closest('[data-action]');if(!a)return;
var act=a.dataset.action,i=parseInt(a.dataset.idx);
if(act==='toggle'){document.getElementById('pc'+i).classList.toggle('collapsed');}
else if(act==='del'){var c=document.getElementById('pc'+i);c.querySelector("[name='style_"+i+"']").value='';c.querySelector("[name='lyrics_"+i+"']").value='';c.querySelector("[name='seed_"+i+"']").value='12';c.classList.add('collapsed');}
else if(act==='add'){for(var j=0;j<4;j++){var c=document.getElementById('pc'+j);if(c.classList.contains('collapsed')){c.classList.remove('collapsed');c.querySelector("[name='style_"+j+"']").focus();return;}}}
});








const players={};
async function togglePlay(key,file,btn){
let p=players[key];
if(p&&p.audio){if(p.audio.paused){p.audio.play();btn.textContent='\u23F8';}else{p.audio.pause();btn.textContent='\u25B6';}return;}
btn.textContent='\u2026';
try{const r=await fetch('/m/'+file);const total=+r.headers.get('Content-Length')||0;
const rd=r.body.getReader();const chunks=[];let got=0;const bar=document.getElementById('bar'+key);
while(true){const n=await rd.read();if(n.done)break;chunks.push(n.value);got+=n.value.length;
if(total)bar.style.width=(100*got/total)+'%';}
const a=new Audio(URL.createObjectURL(new Blob(chunks,{type:'audio/mpeg'})));
const seek=document.getElementById('seek'+key),t=document.getElementById('t'+key);
const fmt=v=>{v=Math.max(0,v||0);return Math.floor(v/60)+':'+String(Math.floor(v%60)).padStart(2,'0');};
a.onloadedmetadata=()=>{seek.max=a.duration;t.textContent='0:00 / '+fmt(a.duration);};
a.ontimeupdate=()=>{if(document.activeElement!==seek)seek.value=a.currentTime;t.textContent=fmt(a.currentTime)+' / '+fmt(a.duration);};
a.onended=()=>{btn.textContent='\u25B6';};
seek.oninput=()=>{a.currentTime=seek.value;};
players[key]={audio:a};btn.textContent='\u23F8';a.play();
}catch(e){btn.textContent='\u25B6';document.getElementById('t'+key).textContent='load failed, retry';}}
let lastSig='';

async function tick(){try{const r=await fetch('/api',{cache:'no-store'});if(!r.ok)throw new Error('http '+r.status);const d=await r.json();
document.getElementById('fill').style.width=d.pct+'%';
document.getElementById('pct').textContent=d.pct+'% - '+d.note;
document.getElementById('step').textContent=d.step_est;
document.getElementById('phase').textContent=d.phase;
document.getElementById('loss').textContent=d.loss;
document.getElementById('speed').textContent=d.speed;
document.getElementById('eval').textContent=d.artist_eval;
document.getElementById('mval').textContent=d.minted_eval;
document.getElementById('eta').textContent=d.eta;
document.getElementById('log').textContent=d.log_tail;
const sig=JSON.stringify([d.samples,d.checkpoints,d.files,d.studio,d.runs,d.prep]);
if(sig!==lastSig){lastSig=sig;
let r='';for(const x of d.runs.runs){const act=x.name===d.runs.active;
r+='<div class="ckpt">'+(act?'<b>'+x.name+' (active)</b>': '<b>'+x.name+'</b> <form method="POST" action="/switch_run" style="display:inline"><input type="hidden" name="name" value="'+x.name+'"><button>Switch</button></form>')+' <span class="muted">'+x.ready+'/'+x.total+' songs'+(x.ckpts.length?' | ckpts '+x.ckpts.join(','):'')+(x.best?' | best ✔':'')+'</span> <a href="/confirm_delete?run='+x.name+'" style="color:#f87171;text-decoration:none;font-size:18px" title="delete run">✕</a><br><form method="POST" action="/set_trigger">trigger: <input name="trigger" value="'+x.trigger+'" placeholder="empty = caption-only" style="width:160px;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px"><input type="hidden" name="run" value="'+x.name+'"> caption: <select name="template"><option value="full"'+(x.template!=='short'?' selected':'')+'>trigger, in the style of…</option><option value="short"'+(x.template==='short'?' selected':'')+'>trigger, caption</option></select> <button>Save</button></form></div>';}
document.getElementById('runs').innerHTML=r||'<div class="muted">no runs yet — create one below</div>';
let s='';if(d.samples.length==0){s='no samples yet - first one lands at step 600';}
for(const x of d.samples){const k=x.step+'p'+x.p;const pl=x.p>=0?' · prompt '+(x.p+1):'';s+='<div class="ckpt"><b>step '+x.step+pl+'</b> <span class="muted">'+x.secs+'s</span><br><button class="play" data-k="'+k+'" data-file="'+x.file+'">\u25B6</button><input class="seek" type="range" id="seek'+k+'" value="0" step="0.1"> <span id="t'+k+'" class="muted">0:00</span><div class="bar" style="height:6px"><div class="fill" id="bar'+k+'"></div></div><a class="dl" href="/m/'+x.file+'" download="'+x.file+'">\u2B07 Download MP3</a></div>';}
document.getElementById('samples').innerHTML=s;
let c='';if(d.checkpoints.length==0){c='none yet';}
for(const x of d.checkpoints){c+='<div class="ckpt" style="display:flex;align-items:center;gap:8px"><span style="flex:1">step-'+x.step+' <span class="muted">'+x.mb+' MB'+(x.sampled?' - sampled &#9989;':'')+'</span></span><button onclick="deleteCkpt('+x.step+')" style="padding:4px 10px;border-radius:5px;border:1px solid #f87171;background:#3a1414;color:#f87171;font-size:12px;cursor:pointer">delete</button></div>';}
document.getElementById('ckpts').innerHTML=c;
let g={};for(const x of d.files){(g[x.g]=g[x.g]||[]).push(x);}
const names={ckpt:'Checkpoints (resume-ready LoRAs)',log:'Logs',runlog:'Run eval log',data:'Training dataset'};
let h='';for(const k of Object.keys(names)){if(!g[k]||!g[k].length)continue;
h+='<div class="ckpt"><b>'+names[k]+'</b><br>';
for(const x of g[k]){h+='<a style="color:#22d3ee" href="/d/'+x.g+'/'+x.file+'">'+x.file+'</a> <span class="muted">'+x.mb+' MB</span><br>';}h+='</div>';}
document.getElementById('dl').innerHTML=h||'nothing yet';
document.getElementById('prep_line').textContent='prep: '+d.prep.stage+' — '+d.prep.detail;
}
if(d.studio){document.getElementById('ds_run').textContent='run: '+d.studio.run;
document.getElementById('ds_count').textContent=d.studio.ready+' / '+d.studio.total+' songs ready (need 7+)';

if(!songsDirty){let q='';if(!d.studio.songs.length)q='<div class="muted">no songs yet — upload audio above, then add caption + lyrics per song</div>';
for(const x of d.studio.songs){const ok=x.issues.length===0;
const esc=s=>String(s).replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;');
const aud=x.audio?'<audio controls preload="none" style="width:100%" src="/a/'+d.studio.run+'/'+x.audio.file+'"></audio><br>':'';
// Combine style + lyrics into single textarea
const combined=x.style+'\\n'+x.lyrics;
q+='<div class="ckpt"><b>'+x.name+'</b> '+(x.audio?'<span class="muted">'+x.audio.mb+' MB flac</span>':'<span class="muted">no audio</span>')+' '+(ok?'\u2714 ready':'<span style="color:#f59e0b">'+x.issues.join('; ')+'</span>')+(x.notes&&x.notes.length?'<br><span class="muted">note: '+x.notes.join('; ')+'</span>':'')+'<br>'+aud+'<form method="POST" action="/save_song"><input type="hidden" name="name" value="'+x.name+'"><label class="muted" style="font-size:12px">caption + lyrics (first line = caption, then [Verse]/[Chorus] sections)</label><br><textarea name="content" rows="8" style="width:100%;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px;font-family:monospace">'+esc(combined)+'</textarea><br><button>Save song</button></form><form method="POST" action="/delete_song"><input type="hidden" name="name" value="'+x.name+'"><button>Delete</button></form></div>';}
document.getElementById('songs').innerHTML=q;}}

}catch(e){document.getElementById('pct').textContent='connection error ('+e.message+'), retrying...';}}tick();setInterval(tick,3000);

/* --- interactive loss chart (Chart.js) --- */
let lossChart=null, lossData={};
function smoothPts(pts,win){if(win<=1)return pts;const r=[];for(let i=0;i<pts.length;i++){const sl=pts.slice(Math.max(0,i-win+1),i+1);r.push({x:pts[i].x,y:sl.reduce((a,b)=>a+b.y,0)/sl.length});}return r;}
async function fetchLoss(){try{const r=await fetch('/api/loss');if(!r.ok)return;lossData=await r.json();renderLossChart();}catch(e){}}
function renderLossChart(){
const ctx=document.getElementById('lossCanvas');if(!ctx)return;
const showLoss=document.getElementById('lg_loss').checked;
const showArtist=document.getElementById('lg_artist').checked;
const showMinted=document.getElementById('lg_minted').checked;
const win=parseInt(document.getElementById('lg_smooth').value)||5;
const logY=document.getElementById('lg_log').checked;
const datasets=[];
if(showLoss&&lossData.loss&&lossData.loss.length>1){datasets.push({label:'loss',data:smoothPts(lossData.loss,win),borderColor:'#FF3B30',backgroundColor:'rgba(255,59,48,0.08)',borderWidth:1.5,pointRadius:0,tension:0.3,fill:true});}
if(showArtist&&lossData.artist&&lossData.artist.length>1){datasets.push({label:'artist',data:smoothPts(lossData.artist,win),borderColor:'#22d3ee',backgroundColor:'rgba(34,211,238,0.08)',borderWidth:1.5,pointRadius:0,tension:0.3,fill:true});}
if(showMinted&&lossData.minted&&lossData.minted.length>1){datasets.push({label:'minted',data:smoothPts(lossData.minted,win),borderColor:'#A0A0A0',backgroundColor:'rgba(160,160,160,0.08)',borderWidth:1.5,pointRadius:0,tension:0.3,fill:true});}
if(lossChart){lossChart.destroy();lossChart=null;}
if(!datasets.length){ctx.getContext('2d').clearRect(0,0,ctx.width,ctx.height);return;}
lossChart=new Chart(ctx,{type:'line',data:{datasets},options:{responsive:true,maintainAspectRatio:false,animation:false,interaction:{mode:'index',intersect:false},scales:{x:{type:'linear',title:{display:true,text:'step',color:'#888'},ticks:{color:'#888'},grid:{color:'#333'}},y:{type:logY?'logarithmic':'linear',title:{display:true,text:'value',color:'#888'},ticks:{color:'#888'},grid:{color:'#333'}}},plugins:{legend:{labels:{color:'#ccc',boxWidth:12,padding:8}},tooltip:{backgroundColor:'#1c1c1c',borderColor:'#444',borderWidth:1,titleColor:'#eee',bodyColor:'#ccc'}}}});
}
fetchLoss();setInterval(fetchLoss,15000);

async function deleteCkpt(step){if(!confirm('Delete checkpoint step-'+step+'? This cannot be undone.'))return;
try{const r=await fetch('/delete_ckpt',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({step:step})});const d=await r.json();if(d.msg)document.getElementById('pct').textContent=d.msg;tick();}catch(e){document.getElementById('pct').textContent='delete failed';}}
</script></body></html>"""

def snapshot():
    d = {"step": 0, "pct": 0.0, "phase": "starting", "loss": "-", "artist_eval": "-",
         "minted_eval": "-", "eta": "-", "note": "", "checkpoints": [], "samples": [], "files": [], "log_tail": "", "step_est": 0, "cfg": {}, "studio": {}, "runs": {"active": "", "runs": []}, "prep": {}, "gpu": {}, "loss_svg": "", "speed": "-"}
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
    # speed: sec/iter from last two logged steps
    speed = "-"
    if len(steps) >= 2:
        s0, _, t0 = steps[-2]
        s1, _, t1 = steps[-1]
        ds = s1 - s0
        if ds > 0:
            sec_per = (t1 - t0) / ds
            speed = f"{sec_per:.1f}s/iter"
    d["speed"] = speed
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
    for f in sorted(glob.glob(os.path.join(ckdir(), "step-*.pt")) + glob.glob(os.path.join(ckdir(), "step-*.safetensors"))):
        m = re.search(r"step-(\d+)", f)
        if m:
            s = int(m.group(1))
            d["checkpoints"].append({"step": s, "pt": f, "mb": round(os.path.getsize(f) / 2**20),
                "sampled": bool(glob.glob(os.path.join(GEN, active_run() + f"_s{s}.*")))})
    for f in sorted(glob.glob(os.path.join(GEN, active_run() + "_s*.mp3"))) + sorted(glob.glob(os.path.join(GEN, active_run() + "_s*.flac"))):
        m = re.search(sp + r"(\d+)(?:p(\d+))?", os.path.basename(f))
        if m and not any(x["file"] == os.path.basename(f) for x in d["samples"]):
            secs = ""
            try:
                meta = json.load(open(os.path.splitext(f)[0] + ".json"))
                secs = f"{meta.get('audio_seconds', 0):.0f}"
            except Exception:
                pass
            d["samples"].append({"step": int(m.group(1)), "p": int(m.group(2)) if m.group(2) is not None else -1,
                                 "file": os.path.basename(f), "secs": secs})
    d["samples"].sort(key=lambda x: (x["step"], x["p"]))
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
    prompts = cfg.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        try:
            st0 = open(STYLE_FILE, errors="replace").read().strip()
        except Exception:
            st0 = ""
        try:
            ly0 = open(LYR_FILE, errors="replace").read().strip()
        except Exception:
            ly0 = ""
        prompts = [{"style": st0, "lyrics": ly0, "seed": cfg.get("seed", 12)}] if (st0 or ly0) else []
    clean = []
    for p in prompts[:4]:
        if isinstance(p, dict):
            clean.append({"style": str(p.get("style", ""))[:1500],
                          "lyrics": str(p.get("lyrics", ""))[:8000],
                          "seed": int(p.get("seed", 12) or 12)})
    d["cfg"] = {"prompts": clean, "walk": bool(cfg.get("walk")),
                "style": clean[0]["style"] if clean else "",
                "lyrics": clean[0]["lyrics"] if clean else "",
                "seed": clean[0]["seed"] if clean else 12}
    d["studio"] = studio_snapshot()
    d["runs"] = runs_snapshot()
    d["prep"] = prep_status()
    d["gpu"] = gpu_snapshot()
    d["loss_svg"] = loss_svg(loss_series())
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
def run_template(name=None):
    try:
        t = json.load(open(os.path.join(run_paths(name)[0], "config.json"))).get("template", "full")
        return t if t in ("full", "short") else "full"
    except Exception:
        return "full"
def trig_prefix(trig, template="full"):
    if not trig:
        return ""
    return f"{trig}, " if template == "short" else f"{trig}, in the style of {trig}. "
def split_style(raw, trig):
    """disk full caption -> box display caption (any trigger prefix hidden)."""
    raw = (raw or "").strip()
    if trig:
        m = re.match(r"(?i)" + re.escape(trig) + r"\s*,\s*in the style of\s+" + re.escape(trig) + r"\.\s*", raw)
        if m:
            return raw[m.end():].strip()
        m = re.match(r"(?i)" + re.escape(trig) + r",\s*", raw)
        if m:
            return raw[m.end():].strip()
    m = re.match(r"(?i)[a-z0-9]+\s*,\s*in the style of\s+[a-z0-9]+\.\s*", raw)
    if m:
        return raw[m.end():].strip()
    return raw
def full_style(display, trig, template="full"):
    """box caption -> disk caption (trigger ensured first, never doubled)."""
    display = (display or "").strip()
    if trig:
        display = split_style(display, trig)
        return trig_prefix(trig, template) + display
    return display
def clean_name(n):
    return re.sub(r"[^a-z0-9_]+", "_", (n or "").strip().lower()).strip("_")[:48]
def validate_song(base, ad, trig):
    info = {"name": base, "audio": None, "style": "", "lyrics": "", "lines": 0, "tags": [], "issues": []}
    for f in glob.glob(os.path.join(ad, base + ".*")):
        fn = os.path.basename(f)
        if fn == base + ".lyrics.txt":
            info["lyrics"] = open(f, errors="replace").read().strip()
        elif fn == base + ".txt":
            info["_raw_style"] = open(f, errors="replace").read().strip()
        elif os.path.splitext(fn)[1].lower() in AUDIO_EXTS:
            info["audio"] = {"file": fn, "mb": round(os.path.getsize(f) / 2**20, 1)}
    lines = [l for l in info["lyrics"].splitlines() if l.strip()]
    info["lines"] = len(lines)
    info["tags"] = sorted(set(re.findall(r"^\s*(\[[^\]]+\])\s*$", info["lyrics"], re.M)))
    info["notes"] = []
    if not info["audio"]:
        info["issues"].append("no audio — upload it below")
    info["style"] = split_style(info.pop("_raw_style", ""), trig)
    if not info["style"]:
        info["issues"].append("no style caption")
    if len(lines) < 8:
        info["issues"].append(f"lyrics short ({len(lines)} lines, want 15+)")
    if not info["tags"]:
        info["notes"].append("no [Verse]/[Chorus] tags — add section headers for better structure (see format guide above)")
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
                    "trigger": run_trigger(n), "template": run_template(n)})
    return {"active": active_run(), "runs": out}

def prep_status():
    try:
        return json.load(open("/workspace/prep_status.json"))
    except Exception:
        return {"stage": "idle", "detail": "not started", "done": False, "error": ""}
def loss_series():
    """Parse train log -> {loss:[(step,v)], artist:[...], minted:[...]}."""
    out = {"loss": [], "artist": [], "minted": []}
    try:
        lines = open(LOG, errors="replace").read().splitlines()
    except Exception:
        return out
    for ln in lines:
        m = re.match(r"step (\d+) loss ([\d.]+) cursor", ln)
        if m:
            out["loss"].append((int(m.group(1)), float(m.group(2))))
        m = re.match(r"EVAL step (\d+) minted_val ([\d.]+) artist ([\d.]+)", ln)
        if m:
            out["minted"].append((int(m.group(1)), float(m.group(2))))
            out["artist"].append((int(m.group(1)), float(m.group(3))))
    return out
def gpu_snapshot():
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit,fan.speed,clocks.current.graphics,clocks.current.memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=8)
        if q.returncode != 0:
            return {}
        p = [x.strip() for x in q.stdout.splitlines()[0].split(",")]
        vram_gb = float(p[4]) / 1024
        vram_mode = "low" if vram_gb < 20 else "high"
        return {"name": p[0], "temp": p[1] + "°C", "load": p[2] + "%",
                "mem": f"{float(p[3]) / 1024:.1f} / {float(p[4]) / 1024:.1f} GB",
                "mempct": round(100 * float(p[3]) / max(1, float(p[4]))),
                "pwr": f"{p[5]} / {p[6]} W",
                "fan": p[7] + "%" if len(p) > 7 and p[7] not in ("N/A", "[N/A]") else "-",
                "clk_gpu": p[8] + " MHz" if len(p) > 8 and p[8] not in ("N/A", "[N/A]") else "-",
                "clk_mem": p[9] + " MHz" if len(p) > 9 and p[9] not in ("N/A", "[N/A]") else "-",
                "vram_mode": os.environ.get("VRAM_MODE", vram_mode)}
    except Exception:
        return {}
def loss_svg(series, w=620, h=180):
    """Dark SVG chart, no JS needed. Returns '' when empty."""
    pts = [(s, v) for k in ("loss", "artist", "minted") for s, v in series[k]]
    if len(pts) < 2:
        return ""
    xs = [p[0] for p in pts]
    vs = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs) or 1
    v0, v1 = min(vs), max(vs)
    if v1 - v0 < 1e-6:
        v1 = v0 + 1
    pad = 34
    def X(s):
        return pad + (s - x0) / (x1 - x0) * (w - pad - 8) if x1 > x0 else pad
    def Y(v):
        return 8 + (1 - (v - v0) / (v1 - v0)) * (h - 16)
    def smooth(seq, win=5):
        sm = []
        for i in range(len(seq)):
            w_ = seq[max(0, i - win + 1):i + 1]
            sm.append((seq[i][0], sum(v for _, v in w_) / len(w_)))
        return sm
    colors = {"loss": "#FF3B30", "artist": "#22d3ee", "minted": "#A0A0A0"}
    el = [f"<svg viewBox='0 0 {w} {h}' style='width:100%' role='img'>"]
    for k in ("minted", "artist", "loss"):
        seq = smooth(series[k])
        if len(seq) < 2:
            continue
        el.append("<polyline fill='none' stroke='" + colors[k] + "' stroke-width='1.6' points='" +
                  " ".join(f"{X(s):.1f},{Y(v):.1f}" for s, v in seq) + "'/>")
    el.append(f"<text x='4' y='{Y(v1) + 3}' fill='#A0A0A0' font-size='9'>{v1:.1f}</text>")
    el.append(f"<text x='4' y='{Y(v0) + 3}' fill='#A0A0A0' font-size='9'>{v0:.1f}</text>")
    el.append(f"<text x='{w - 44}' y='{h - 2}' fill='#A0A0A0' font-size='9'>step {x1}</text>")
    el.append("</svg>")
    leg = "".join(f"<span class='muted'><span style='color:{c}'>●</span> {k} </span>" for k, c in colors.items())
    return "".join(el) + "<div>" + leg + "</div>"
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
        if self.path == "/resume_info":
            base, _, _ = run_paths()
            last_pt = os.path.join(base, "last.pt")
            best_pt = os.path.join(base, "best.pt")
            info = {"exists": False, "step": 0, "best_loss": 999, "rank": 64, "target_steps": 800}
            if os.path.exists(last_pt):
                try:
                    if torch:
                        ck = torch.load(last_pt, map_location="cpu", weights_only=False)
                        info["exists"] = True
                        info["rank"] = ck.get("rank", 64)
                        # Get best loss
                        if os.path.exists(best_pt):
                            best_ck = torch.load(best_pt, map_location="cpu", weights_only=False)
                            info["best_loss"] = best_ck.get("best_loss", 999)
                    else:
                        info["exists"] = True
                except Exception:
                    pass
                # Get step from last.pt step file
                import glob as g
                step_files = g.glob(os.path.join(base, "step-*.pt"))
                if step_files:
                    try:
                        max_step = max(int(re.search(r"step-(\d+)", f).group(1)) for f in step_files if re.search(r"step-(\d+)", f))
                        info["step"] = max_step
                    except Exception:
                        pass
            b = json.dumps(info).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif self.path == "/gen_list":
            gen_dir = "/workspace/tok/full/gen"
            files = []
            if os.path.exists(gen_dir):
                for f in sorted(os.listdir(gen_dir)):
                    if f.endswith((".ogg", ".mp3", ".flac", ".wav")):
                        sz = os.path.getsize(os.path.join(gen_dir, f))
                        files.append({"name": f, "size": f"{sz/1024:.0f} KB"})
            b = json.dumps({"files": files}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif self.path.startswith("/gen_status"):
            import urllib.parse
            params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            job_id = params.get("job", [""])[0]
            status_file = f"/workspace/gen_jobs/{job_id}.json"
            if os.path.exists(status_file):
                try:
                    data = json.load(open(status_file))
                except Exception:
                    data = {"done": True, "error": "status file corrupted"}
            else:
                data = {"done": False, "pct": 0, "status": "waiting", "eta": "unknown"}
            b = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif self.path == "/api":
            b = json.dumps(snapshot()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif self.path == "/api/loss":
            series = loss_series()
            # convert to Chart.js format: [{x: step, y: value}, ...]
            out = {k: [{"x": s, "y": v} for s, v in v] for k, v in series.items()}
            b = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
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
        elif self.path.startswith("/assets/"):
            fn = os.path.basename(self.path[8:])
            if fn not in ("logo.png",):
                self.send_error(404)
                return
            repo = os.environ.get("FORGE_REPO", "/workspace/yue2-forge")
            self._send(os.path.join(repo, "assets", fn), "image/png")
        elif self.path.startswith("/m/"):
            fn = os.path.basename(self.path[3:])
            if ".." in fn or not re.match(r"[A-Za-z0-9_]+_s\d+(p\d+)?\.(mp3|flac)$", fn):
                self.send_error(404)
                return
            self._send(os.path.join(GEN, fn),
                         "audio/mpeg" if fn.endswith(".mp3") else "audio/flac")
        elif self.path.startswith("/gen/"):
            fn = os.path.basename(self.path[5:])
            if ".." in fn or not re.match(r".+\.(ogg|mp3|flac|wav)$", fn):
                self.send_error(404)
                return
            gen_dir = "/workspace/tok/full/gen"
            ctype = {"ogg": "audio/ogg", "mp3": "audio/mpeg", "flac": "audio/flac", "wav": "audio/wav"}.get(fn.rsplit(".", 1)[-1], "audio/ogg")
            self._send(os.path.join(gen_dir, fn), ctype)
        elif self.path.split("?", 1)[0] in ("/live", "/live-mini"):
            import time as _t2, html as _h2
            d = snapshot()
            mini = self.path.split("?", 1)[0] == "/live-mini"
            import time as _t2, html as _h2
            d = snapshot()
            rows = ""
            for x in d["checkpoints"]:
                rows += f"<div class='ckpt'>step-{x['step']} <span class='muted'>{x['mb']} MB{' · sampled ✔' if x['sampled'] else ''}</span></div>"
            smp = ""
            for x in d["samples"]:
                pl = f" · prompt {x['p'] + 1}" if x['p'] >= 0 else ""
                smp += f"<div class='ckpt'><b>step {x['step']}{pl}</b> <span class='muted'>{x['secs']}s</span> <a class='dl' href='/m/{x['file']}'>play/download</a></div>"
            if mini:
                page = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="20"><style>body{{background:#0D0D0D;color:#fff;font-family:system-ui,sans-serif;margin:0;padding:10px 12px}}a{{color:#22d3ee;font-size:12px}}.bar{{height:8px;background:#1E1E1E;border:1px solid #2C2C2C;border-radius:5px;overflow:hidden;margin:6px 0}}.fill{{height:100%;background:linear-gradient(90deg,#7c3aed,#22d3ee);width:{d['pct']}%}}.t{{font-size:14px}}.muted{{color:#A0A0A0;font-size:12px}}</style></head><body><div class="bar"><div class="fill"></div></div><div class="t">step {d['step_est']}/{TOTAL} ({d['pct']}%) · {d['phase']} · loss {d['loss']} · eval {d['artist_eval']}</div><div class="muted">auto-refreshes · {_t2.strftime('%H:%M:%S')}</div></body></html>"""
            else:
                page = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="20"><title>YuE2 LoRA Training | @directedbykobyperez</title><style>body{{background:#0D0D0D;color:#fff;font-family:system-ui,sans-serif;max-width:640px;margin:0 auto;padding:18px}}h2{{font-size:20px}}.bar{{height:12px;background:#1E1E1E;border:1px solid #2C2C2C;border-radius:6px;overflow:hidden}}.fill{{height:100%;background:linear-gradient(90deg,#7c3aed,#22d3ee);width:{d['pct']}%}}.ckpt{{background:#1E1E1E;border:1px solid #2C2C2C;border-radius:10px;padding:10px;margin:8px 0}}.muted{{color:#A0A0A0;font-size:13px}}.dl{{color:#22d3ee}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}.card{{background:#1E1E1E;border:1px solid #2C2C2C;border-radius:10px;padding:10px;color:#A0A0A0;font-size:12px}}.card b{{font-size:19px;color:#fff}}</style></head><body><h2>forge live <span class="muted">· auto every 20s · {_t2.strftime('%H:%M:%S')}</span></h2><div class="bar"><div class="fill"></div></div><div class="muted">run {active_run()} · step {d['step_est']}/{TOTAL} ({d['pct']}%) · {d['phase']} · loss {d['loss']}</div><div class="grid"><div class="card">artist eval<div><b>{d['artist_eval']}</b></div></div><div class="card">minted eval<div><b>{d['minted_eval']}</b></div></div><div class="card">ETA<div><b>{d['eta']}</b></div></div><div class="card">prep<div><b>{d['prep'].get('stage', 'idle')}</b></div></div></div><h3 style="font-size:12px;text-transform:uppercase;letter-spacing:.14em;color:#A0A0A0">checkpoints</h3>{rows or "<div class='muted'>none yet</div>"}<h3 style="font-size:12px;text-transform:uppercase;letter-spacing:.14em;color:#A0A0A0">samples</h3>{smp or "<div class='muted'>none yet</div>"}<p><a class="dl" href="/">← full dashboard</a></p></body></html>"""
            b = page.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
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
            page = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>delete {name}?</title><style>body{{background:#111;color:#eee;font-family:system-ui,sans-serif;max-width:560px;margin:0 auto;padding:20px}}.warn{{background:#3a1414;border:1px solid #f87171;border-radius:8px;padding:14px}}button{{padding:10px 20px;border-radius:6px;font-size:15px}}a{{color:#22d3ee}}</style></head><body><h2>Delete run '{name}'?</h2><div class="warn">This permanently removes:<br>· {nsongs} song(s) + captions + lyrics<br>· {nckpt} checkpoint file(s) incl. LoRAs<br>· {nsamp} sample track(s)<br><br>Cannot be undone. Download anything you want to keep first.</div><br><form method="POST" action="/delete_run"><input type="hidden" name="name" value="{name}"><button style="border:0;background:#dc2626;color:#fff">Yes, delete everything</button></form><br><a href="/">Cancel — keep my run</a></body></html>"""
            b = page.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        else:
            import time as _t, html as _h, urllib.parse as _uq
            msg = ""
            if "?" in self.path:
                q = self.path.split("?", 1)[1]
                qq = _uq.parse_qs(q)
                msg = qq.get("msg", [""])[0][:200]
            d = snapshot()
            st = f"<div class='muted'>SERVER { _t.strftime('%H:%M:%S') } · run {active_run()}: step {d['step']}/{TOTAL} ({d['pct']}%) | {d['phase']} | loss {d['loss']} | artist {d['artist_eval']} | minted {d['minted_eval']}</div>"
            ss = ""
            for x in d["samples"]:
                pl = f" · prompt {x['p'] + 1}" if x['p'] >= 0 else ""
                ss += f"<div class='ckpt'><b>step {x['step']}{pl}</b> <span class='muted'>{x['secs']}s</span><br><audio controls preload='none' style='width:100%' src='/m/{x['file']}'></audio><br><a class='dl' href='/m/{x['file']}' download='{x['file']}'>Download MP3</a></div>"
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
                sg += f"<div class='ckpt'><b>{x['name']}</b> <span class='muted'>{x['lines']} lines, {x['audio']['mb'] if x['audio'] else 0} MB</span><br>{player}<span class='muted'>{flag}</span>" + (f"<br><span class='muted'>note: {' — '.join(x['notes'])}</span>" if x["notes"] else "")
                sg += f"<form method='POST' action='/save_song'><input type='hidden' name='name' value='{x['name']}'>"
                sg += f"style / caption<br><input name='style' value='{_h.escape(x['style'], quote=True)}' style='width:100%;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px'><br>"
                sg += f"lyrics<br><textarea name='lyrics' rows='6' style='width:100%;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px'>{_h.escape(x['lyrics'], quote=False)}</textarea><br>"
                sg += f"<button>Save song</button></form>"
                sg += f"<form method='POST' action='/delete_song' onsubmit=\"return confirm('delete {x['name']}?')\"><input type='hidden' name='name' value='{x['name']}'><button>Delete</button></form></div>"
            rs = ""
            for x in d["runs"]["runs"]:
                act = x["name"] == d["runs"]["active"]
                sw = "" if act else f"<form method='POST' action='/switch_run' style='display:inline'><input type='hidden' name='name' value='{x['name']}'><button>Switch</button></form>"
                rs += f"<div class='ckpt'><b>{x['name']}</b>{' (active)' if act else ''} <span class='muted'>{x['ready']}/{x['total']} songs" + (f" | ckpts {','.join(map(str, x['ckpts']))}" if x["ckpts"] else "") + "</span> " + sw + f" <a href='/confirm_delete?run={x['name']}' style='color:#f87171;text-decoration:none;font-size:18px' title='delete run'>✕</a><br><form method='POST' action='/set_trigger'>trigger: <input name='trigger' value='{_h.escape(x['trigger'], quote=True)}' placeholder='empty = caption-only' style='width:160px;background:#000;color:#eee;border:1px solid #444;border-radius:6px;padding:8px'><input type='hidden' name='run' value='{x['name']}'> caption: <select name='template'><option value='full'{(' selected' if x['template'] != 'short' else '')}>trigger, in the style of…</option><option value='short'{(' selected' if x['template'] == 'short' else '')}>trigger, caption</option></select> <button>Save</button></form></div>"
            pp = f"<div class='muted'>prep: {d['prep'].get('stage', 'idle')} — {d['prep'].get('detail', '')}</div>"
            pc = "<div class='ptools'><span class='muted' style='margin-right:8px'>up to 4 — one render each per checkpoint</span><button type='button' class='padd' data-action='add'>+ Add prompt</button></div>"
            plist = d["cfg"].get("prompts", [])
            while len(plist) < 4:
                plist = plist + [{"style": "", "lyrics": "", "seed": 12}]
            for i, p in enumerate(plist[:4]):
                st = str(p.get("style", "") or "")
                ly = str(p.get("lyrics", "") or "")
                sd = p.get("seed", 12)
                filled = bool(st.strip() or ly.strip())
                cls = "pcard" if filled else "pcard collapsed"
                pc += (f"<div class='{cls}' id='pc{i}'><div class='phead' data-action='toggle' data-idx='{i}'><b>prompt {i + 1}</b>"
                       f"<button type='button' class='x' data-action='del' data-idx='{i}' title='clear inputs'>clean</button></div>"
                       f"<div class='pbody'><div class='plabel'>Style / caption</div>"
                       f"<input name='style_{i}' value='{_h.escape(st, quote=True)}'>"
                       f"<div class='plabel'>Lyrics</div>"
                       f"<textarea name='lyrics_{i}' rows='5'>{_h.escape(ly, quote=False)}</textarea>"
                       f"<div class='plabel'>Seed</div>"
                       f"<input class='seedbox' name='seed_{i}' value='{sd}' type='number'></div></div>")
            g = d["gpu"]
            gp = (f"<b>{_h.escape(g.get('name', 'GPU'))}</b><br>🌡 {g.get('temp', '-')} · load {g.get('load', '-')} · {g.get('mem', '-')} ({g.get('mempct', 0)}%) · {g.get('pwr', '-')} · fan {g.get('fan', '-')} · GPU {g.get('clk_gpu', '-')} · mem {g.get('clk_mem', '-')} · VRAM: {g.get('vram_mode', '?')}" if g else "<span class='muted'>no GPU visible</span>")
            b = HTML.replace("<!--STATIC_STATUS-->", st).replace("<!--STATIC_SAMPLES-->", ss or "<div class='muted'>no samples yet</div>").replace("<!--STATIC_FILES-->", ff).replace("FILLPCT", str(d["pct"])).replace("<!--STATIC_RUNS-->", rs or "<div class='muted'>no runs yet</div>").replace("<!--STATIC_PREP-->", pp).replace("<!--STATIC_GPU-->", gp).replace("<!--LOSSGRAPH-->", d["loss_svg"] or "<div class='muted'>no training data yet</div>").replace("<!--STATIC_PROMPTS-->", pc)
            b = b.replace("<!--MSG-->", f"<div class='ckpt' style='border-color:#7c3aed'>{_h.escape(msg)}</div>" if msg else "")
            b = b.replace("<!--STATIC_PHASE-->", f"<span class='pill'><span class='dot'></span>{_h.escape(str(d['phase']))} · {d['step_est']}/{TOTAL}</span>")
            b = b.replace('<b id="step">-</b>', f"<b id=\"step\">{d['step_est']}</b>").replace('<b id="phase">-</b>', f"<b id=\"phase\">{d['phase']}</b>").replace('<b id="loss">-</b>', f"<b id=\"loss\">{d['loss']}</b>").replace('<b id="eval">-</b>', f"<b id=\"eval\">{d['artist_eval']}</b>").replace('<b id="mval">-</b>', f"<b id=\"mval\">{d['minted_eval']}</b>")
            b = b.replace('<b id="eta">-</b>', f"<b id=\"eta\">{d['eta']}</b>")
            b = b.replace("FORGETITLE", _h.escape(os.environ.get("FORGE_TITLE", "YuE2 LoRA Training | @directedbykobyperez")))
            for _i in (1, 2, 3, 4):
                b = b.replace(f"<!--PANE{_i}-->", "").replace(f"<!--ENDPANE{_i}-->", "")
            b = b.replace("CFGSTYLE", _h.escape(d["cfg"].get("style", ""), quote=True)).replace("CFGLYRICS", _h.escape(d["cfg"].get("lyrics", ""), quote=False)).replace('value="CFGSEED"', f"value=\"{d['cfg'].get('seed', 12)}\"").replace("WALKCHECKED", " checked" if d["cfg"].get("walk") else "")
            b = b.replace("STATICLOG", _h.escape(d["log_tail"] or "no log yet", quote=False))
            b = b.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Cache-Control", "no-store")
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
        if path == "/hf_list":
            return self._post_hf_list()
        if path == "/hf_import":
            return self._post_hf_import()
        if path == "/create_run":
            return self._post_create_run()
        if path == "/switch_run":
            return self._post_switch_run()
        if path == "/start_training":
            return self._post_start_training()
        if path == "/delete_run":
            return self._post_delete_run()
        if path == "/prepare_dataset":
            return self._post_prepare_dataset()
        if path == "/delete_ckpt":
            return self._post_delete_ckpt()
        if path == "/generate":
            return self._post_generate()
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
        loc = "/" + ("?msg=" + _up.quote(msg[:200]) if msg else "")
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
            walk = bool(str(c.get("walk", "")).strip().lower() in ("1", "on", "true", "yes"))
            raw_prompts = c.get("prompts")
            if (not isinstance(raw_prompts, list) or not raw_prompts) and any(f"style_{i}" in c for i in range(4)):
                raw_prompts = [{"style": c.get(f"style_{i}", ""), "lyrics": c.get(f"lyrics_{i}", ""),
                                "seed": c.get(f"seed_{i}", 12)} for i in range(4)]
            prompts = []
            if isinstance(raw_prompts, list) and raw_prompts:
                for p in raw_prompts[:4]:
                    st = str(p.get("style", "")).strip()[:1500]
                    ly = str(p.get("lyrics", "")).strip()[:8000]
                    if not st or not ly:
                        continue
                    try:
                        sd = int(str(p.get("seed", "")).strip() or 12)
                    except Exception:
                        sd = 12
                    prompts.append({"style": st, "lyrics": ly, "seed": sd})
                if not prompts:
                    raise ValueError("at least one complete prompt required")
            else:
                style = str(c.get("style", "")).strip()[:1500]
                lyrics = str(c.get("lyrics", "")).strip()[:8000]
                try:
                    seed = int(str(c.get("seed", "")).strip() or 12)
                except Exception:
                    seed = 12
                if not style or not lyrics:
                    raise ValueError("style and lyrics required")
                if not re.search(r"[a-z\[]", lyrics, re.I):
                    raise ValueError("lyrics look empty")
                prompts = [{"style": style, "lyrics": lyrics, "seed": seed}]
        except Exception as e:
            return self._fail(e, "3")
        try:
            os.makedirs(os.path.dirname(STYLE_FILE) or ".", exist_ok=True)
            open(STYLE_FILE, "w").write(prompts[0]["style"] + "\n")
            open(LYR_FILE, "w").write(prompts[0]["lyrics"] + "\n")
        except Exception:
            pass
        try:
            os.makedirs(os.path.dirname(CFG) or ".", exist_ok=True)
            json.dump({"seed": prompts[0]["seed"], "walk": walk, "prompts": prompts}, open(CFG, "w"))
        except Exception as e:
            return self._fail(f"cannot save prompt config: {e}", "3")
        return self._ok({"ok": True, "msg": f"{len(prompts)} sample prompt(s) saved"}, "3")
    def _post_save_song(self):
        c = self._fields()
        if c is None:
            return self._fail("bad request", "2")
        try:
            name = clean_name(c.get("name", ""))
            # New combined format: content = "caption\n[Verse 1]\nlyrics..."
            content = str(c.get("content", "")).strip()[:15000]
            strig = re.sub(r"[^a-z0-9]+", "", str(c.get("trigger", "")).strip().lower())[:32] if "trigger" in c else None
            if not name:
                raise ValueError("song name required (letters, numbers, _)")
            if not content:
                raise ValueError("content required")
            # Parse combined content: first line = caption, rest = lyrics
            lines = content.split("\n")
            style = lines[0].strip()[:1500] if lines else ""
            lyrics = "\n".join(lines[1:]).strip()[:12000] if len(lines) > 1 else "[instrumental]"
        except Exception as e:
            return self._fail(e, "2")
        _, ad, ald = run_paths()
        os.makedirs(ad, exist_ok=True)
        os.makedirs(ald, exist_ok=True)
        open(os.path.join(ad, name + ".txt"), "w").write(full_style(style, run_trigger(), run_template()) + "\n")
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
        tmpl = str(c.get("template", "")).strip().lower()
        if tmpl in ("full", "short"):
            cfg["template"] = tmpl
        else:
            tmpl = cfg.get("template", "full")
        json.dump(cfg, open(cfgp, "w"))
        n = 0
        if trig:
            for f in glob.glob(os.path.join(base, "artist", "*.txt")):
                if f.endswith(".lyrics.txt"):
                    continue
                try:
                    cur = open(f, errors="replace").read().strip()
                except Exception:
                    continue
                want = full_style(cur, trig, tmpl)
                if want != cur:
                    open(f, "w").write(want + "\n")
                    n += 1
        return self._ok({"ok": True, "trigger": trig, "msg": f"trigger saved ({n} captions updated)"}, "1")
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
                    else:
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
                content = text.decode("utf-8", "replace").strip()[:15000]
            except Exception:
                errs.append(f"{fname}: unreadable text")
                continue
            if not content:
                errs.append(f"{fname}: empty lyrics")
                continue
            # Parse combined format: first line = caption, rest = lyrics
            lines = content.split("\n")
            style = lines[0].strip()[:1500] if lines else ""
            lyrics = "\n".join(lines[1:]).strip()[:12000] if len(lines) > 1 else "[instrumental]"
            open(os.path.join(ad, name + ".txt"), "w").write(full_style(style, run_trigger(), run_template()) + "\n")
            open(os.path.join(ad, name + ".lyrics.txt"), "w").write(lyrics + "\n")
            open(os.path.join(ald, name + ".lyrics.txt"), "w").write(lyrics + "\n")
            done.append(name + " (lyrics + caption)")
        for fname, blob in tracks:
            name = clean_name(os.path.splitext(os.path.basename(fname))[0])
            if not name:
                errs.append(f"{fname}: bad name")
                continue
            if not blob:
                errs.append(f"{fname}: arrived empty (0 bytes) — retry the upload")
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
            elif os.path.getsize(out) < 10240:
                try:
                    os.remove(out)
                except Exception:
                    pass
                errs.append(f"{fname}: convert produced nothing usable — retry")
            else:
                done.append(name)
        if not done:
            return self._fail("; ".join(errs) or "nothing converted", "2")
        msg = f"uploaded {len(done)}: {', '.join(done)}" + (f" — errors: {'; '.join(errs)}" if errs else "")
        return self._ok({"ok": True, "msg": msg}, "2")
    def _post_hf_list(self):
        c = self._fields()
        if c is None:
            return self._fail("bad request", "2")
        token = c.get("token", "").strip()
        repo = c.get("repo", "").strip()
        if not repo:
            return self._fail("enter a repo name", "2")
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        url = f"https://huggingface.co/api/datasets/{repo}/tree/main"
        try:
            import urllib.request, urllib.error
            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req, timeout=15)
            data = json.loads(resp.read())
            files = []
            AUDIO_EXTS_HF = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".webm"}
            for item in data:
                name = item.get("path", "")
                ext = os.path.splitext(name)[1].lower()
                if ext in AUDIO_EXTS_HF or ext == ".txt":
                    size = item.get("size", 0)
                    if size > 50 * 1024 * 1024:
                        continue
                    size_str = f"{size / 1024 / 1024:.1f}MB" if size > 1024 * 1024 else f"{size / 1024:.0f}KB"
                    files.append({"name": name, "size": size_str})
            return self._ok({"files": files}, "2")
        except urllib.error.HTTPError as e:
            if e.code == 401:
                return self._fail("auth failed — check your token", "2")
            if e.code == 404:
                return self._fail("repo not found", "2")
            return self._fail(f"HF error: {e.code}", "2")
        except Exception as e:
            return self._fail(str(e), "2")
    def _post_hf_import(self):
        c = self._fields()
        if c is None:
            return self._fail("bad request", "2")
        token = c.get("token", "").strip()
        repo = c.get("repo", "").strip()
        files = c.get("files", [])
        if not repo or not files:
            return self._fail("repo and files required", "2")
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        _, ad, ald = run_paths()
        os.makedirs(ad, exist_ok=True)
        os.makedirs(ald, exist_ok=True)
        done, errs = [], []
        import urllib.request
        AUDIO_EXTS_HF = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".webm"}
        for fname in files:
            url = f"https://huggingface.co/datasets/{repo}/resolve/main/{fname}"
            name = clean_name(os.path.splitext(os.path.basename(fname))[0])
            if not name:
                errs.append(f"{fname}: bad name")
                continue
            try:
                req = urllib.request.Request(url, headers=headers)
                resp = urllib.request.urlopen(req, timeout=120)
                data = resp.read()
            except Exception as e:
                errs.append(f"{fname}: download failed ({e})")
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext == ".txt":
                try:
                    content = data.decode("utf-8", "replace").strip()[:15000]
                except Exception:
                    errs.append(f"{fname}: unreadable")
                    continue
                if not content:
                    errs.append(f"{fname}: empty")
                    continue
                lines = content.split("\n")
                style = lines[0].strip()[:1500] if lines else ""
                lyrics = "\n".join(lines[1:]).strip()[:12000] if len(lines) > 1 else "[instrumental]"
                open(os.path.join(ad, name + ".txt"), "w").write(full_style(style, run_trigger(), run_template()) + "\n")
                open(os.path.join(ad, name + ".lyrics.txt"), "w").write(lyrics + "\n")
                open(os.path.join(ald, name + ".lyrics.txt"), "w").write(lyrics + "\n")
                done.append(name + " (lyrics)")
            elif ext in AUDIO_EXTS_HF:
                tmp = os.path.join(ad, name + ".incoming" + ext)
                open(tmp, "wb").write(data)
                out = os.path.join(ad, name + ".flac")
                try:
                    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp, "-c:a", "flac", out],
                                       capture_output=True, timeout=600)
                except Exception:
                    try: os.remove(tmp)
                    except: pass
                    errs.append(f"{fname}: convert failed")
                    continue
                try: os.remove(tmp)
                except: pass
                if r.returncode != 0 or not os.path.exists(out):
                    errs.append(f"{fname}: ffmpeg rejected")
                elif os.path.getsize(out) < 10240:
                    try: os.remove(out)
                    except: pass
                    errs.append(f"{fname}: bad audio")
                else:
                    done.append(name)
            else:
                errs.append(f"{fname}: skipped (unknown type)")
        if not done:
            return self._fail("; ".join(errs) or "nothing imported", "2")
        msg = f"imported {len(done)}: {', '.join(done)}" + (f" — errors: {'; '.join(errs)}" if errs else "")
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
            total = max(200, min(5000, int(c.get("steps", 800))))
            rank = max(8, min(128, int(c.get("rank", 64))))
            base, _, _ = run_paths()
            try:
                rconf = json.load(open(os.path.join(base, "config.json")))
            except Exception:
                rconf = {}
            if init != "fresh" and "last_rank" in rconf and rconf["last_rank"] != rank:
                raise ValueError(f"rank changed ({rconf['last_rank']}→{rank}) — resume needs same rank, or start fresh")
            rconf["last_rank"] = rank
            os.makedirs(base, exist_ok=True)
            json.dump(rconf, open(os.path.join(base, "config.json"), "w"))
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
                   CK_FROM="600", CK_EVERY=str(c.get("save_every", "250")), START_STEP=str(start),
                   VRAM_MODE=str(c.get("vram_mode", "low")),
                   COT=str(c.get("cot", "off")),
                   TOKENIZER=str(c.get("tokenizer", "community")),
                   AR_KL_WEIGHT=str(c.get("ar_kl_weight", "0.04")),
                   AR_LR_MULTIPLIER=str(c.get("ar_lr_multiplier", "1.0")),
                   ABC_DROPOUT=str(c.get("abc_dropout", "0.5")),
                   TRAIN_WINDOW=str(c.get("train_window", "1500")),
                   SHEETSAGE_TASK=str(c.get("sheetsage_task", "full")),
                   SAMPLE_AR_REPETITION_PENALTY=str(c.get("sample_ar_repetition_penalty", "1.2")),
                   EMA_DECAY=str(c.get("ema_decay", "0.99")),
                   WEIGHT_DECAY=str(c.get("weight_decay", "0.0001")),
                   LR=str(c.get("lr", "0.0001")))
        log = open("/workspace/ar_train.log", "a")
        subprocess.Popen(["/workspace/yue2venv/bin/python", "-u", train_py, name,
                          str(total - start), str(rank), "0.5", initpt, "1e-4", "0.08"],
                         stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True, env=env)
        return self._ok({"ok": True, "run": name, "msg": f"training {name} {start}→{total}"}, "3")
    def _post_prepare_dataset(self):
        if subprocess.run(["pgrep", "-f", "ar_train|ar_lora_"], capture_output=True).returncode == 0:
            return self._fail("training already running", "3")
        if subprocess.run(["pgrep", "-f", "prepare_run"], capture_output=True).returncode == 0:
            return self._fail("prep already running", "3")
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
            return self._fail(f"need {MIN_SONGS}+ ready songs (have {ready})", "3")
        repo = os.environ.get("FORGE_REPO", "/workspace/yue2-forge")
        prep_sh = os.path.join(repo, "scripts", "prepare_run.sh")
        if not os.path.exists(prep_sh):
            return self._fail("prepare script missing on server", "3")
        log = open("/workspace/prep.log", "a")
        subprocess.Popen(["bash", prep_sh], stdout=log, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True,
                         env=dict(os.environ, HF_HOME="/workspace/hf",
                                  REG_PACK="/workspace/real/regularizer/minted_regularizer_pack.pt"))
        return self._ok({"ok": True, "msg": "prep started (prep → cursor → ar_prep)"}, "3")
    def _post_delete_ckpt(self):
        c = self._fields()
        if c is None:
            return self._fail("bad request", "3")
        try:
            step = int(c.get("step", 0))
            if step <= 0:
                raise ValueError("bad step number")
        except Exception as e:
            return self._fail(e, "3")
        ckd = os.path.join("/workspace/tok/full", active_run())
        path = os.path.join(ckd, f"step-{step}.pt")
        if not os.path.exists(path):
            return self._fail(f"checkpoint step-{step} not found", "3")
        if subprocess.run(["pgrep", "-f", "ar_train|ar_lora_"], capture_output=True).returncode == 0:
            return self._fail("stop training first before deleting checkpoints", "3")
        try:
            os.remove(path)
        except Exception as e:
            return self._fail(str(e), "3")
        return self._ok({"ok": True, "msg": f"deleted step-{step}.pt"}, "3")
    def _post_generate(self):
        import uuid, threading
        c = self._fields()
        if c is None:
            return self._fail("bad request", "3")
        ckpt = c.get("ckpt", "")
        style = c.get("style", "")
        lyrics = c.get("lyrics", "")
        seed = int(c.get("seed", 12))
        precision = c.get("precision", "int8")
        cfg = float(c.get("cfg", 1.0))
        cot = c.get("cot", "off")
        temp = float(c.get("temp", 1.0))
        top_p = float(c.get("top_p", 0.95))
        top_k = int(c.get("top_k", 50))
        rep_pen = float(c.get("rep_pen", 1.2))
        if not ckpt:
            return self._fail("select a checkpoint", "3")
        if not os.path.exists(ckpt):
            return self._fail(f"checkpoint not found: {ckpt}", "3")
        job_id = str(uuid.uuid4())[:8]
        job_dir = "/workspace/gen_jobs"
        os.makedirs(job_dir, exist_ok=True)
        job_file = os.path.join(job_dir, f"{job_id}.json")
        json.dump({"done": False, "pct": 0, "status": "starting", "eta": "estimating..."}, open(job_file, "w"))
        def run_gen():
            try:
                name = active_run()
                nar = os.path.join("/workspace/tok/full", name, "joint_v1.pt")
                if not os.path.exists(nar):
                    nar = "none"
                style_file = f"/workspace/gen_jobs/_style_{job_id}.txt"
                lyr_file = f"/workspace/gen_jobs/_lyrics_{job_id}.txt"
                open(style_file, "w").write(style)
                open(lyr_file, "w").write(lyrics)
                out_tag = f"gen_{job_id}"
                json.dump({"done": False, "pct": 30, "status": "loading model", "eta": "~60s"}, open(job_file, "w"))
                env = dict(os.environ, HF_HOME="/workspace/hf", VRAM_MODE=precision)
                env["CFG_SCALE"] = str(cfg)
                env["COT"] = cot
                env["TEMPERATURE"] = str(temp)
                env["TOP_P"] = str(top_p)
                env["TOP_K"] = str(top_k)
                env["REPETITION_PENALTY"] = str(rep_pen)
                if precision == "int8":
                    env["YUE2_MODEL"] = "/workspace/comfyui/yue2_3b_int8_convrot.safetensors"
                repo = os.environ.get("FORGE_REPO", "/workspace/yue2-forge")
                gen_py = os.path.join(repo, "scripts", "ar_generate.py")
                log = open("/workspace/gen.log", "a")
                proc = subprocess.Popen(
                    ["/workspace/yue2venv/bin/python", "-u", gen_py,
                     ckpt, nar, out_tag, f"@{style_file}", lyr_file, str(seed)],
                    stdout=log, stderr=subprocess.STDOUT, env=env
                )
                json.dump({"done": False, "pct": 50, "status": "generating", "eta": "~45s"}, open(job_file, "w"))
                proc.wait()
                gen_dir = "/workspace/tok/full/gen"
                # Convert to ogg
                src_flac = os.path.join(gen_dir, f"{out_tag}.flac")
                dst_ogg = os.path.join(gen_dir, f"{out_tag}.ogg")
                if os.path.exists(src_flac):
                    subprocess.run(["ffmpeg", "-y", "-i", src_flac, "-c:a", "libopus", "-b:a", "64k", dst_ogg],
                                   capture_output=True)
                    os.remove(src_flac)
                # Cleanup temp files
                for f in (style_file, lyr_file):
                    try: os.remove(f)
                    except: pass
                if proc.returncode == 0:
                    json.dump({"done": True, "pct": 100, "status": "complete", "eta": "",
                               "file": f"{out_tag}.ogg", "error": None}, open(job_file, "w"))
                else:
                    json.dump({"done": True, "pct": 100, "status": "failed", "eta": "",
                               "error": f"exit code {proc.returncode}"}, open(job_file, "w"))
            except Exception as e:
                json.dump({"done": True, "pct": 100, "status": "error", "eta": "",
                           "error": str(e)[:200]}, open(job_file, "w"))
        t = threading.Thread(target=run_gen, daemon=True)
        t.start()
        return self._json({"ok": True, "job_id": job_id, "eta": "~60s"})
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
