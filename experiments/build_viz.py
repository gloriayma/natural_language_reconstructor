#!/usr/bin/env python
"""
Build viz/roundtrip.html — a self-contained interactive explorer for the
embed->unembed round-trip results.

Reads every results/<model>/ dir that has meta.json + results.json (script v2),
packs a compact form of the data (token spellings + logprobs + ranks), gzips it,
base64s it, and injects it into the HTML template below. Decoded token strings
are reconstructed client-side (byte-level-BPE inverse map / SentencePiece rule),
which keeps the payload roughly half the size.

Rerun after more models finish:  python experiments/build_viz.py
"""
import base64
import datetime
import glob
import gzip
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "viz", "roundtrip.html")


def tok_style(rows):
    """Detect tokenizer spelling convention from token strings."""
    sample = "".join(r.get("token") or "" for r in rows[:200])
    if "Ġ" in sample or "Ċ" in sample:  # Ġ / Ċ
        return "bpe"
    if "▁" in sample:  # ▁
        return "sp"
    return "bpe"


def pack_variant(v):
    if not v:
        return None
    out = {
        "top": [[t["token"], round(math.log(max(t["prob"], 1e-12)), 2)] for t in v["top"]],
        "self": v["self_rank"],
    }
    if "next_rank" in v:
        out["next"] = v["next_rank"]
    return out


def pack_model(mdir):
    meta = json.load(open(os.path.join(mdir, "meta.json")))
    if meta.get("script_version") != 2:
        return None
    rows = json.load(open(os.path.join(mdir, "results.json")))
    probes = ["emb"] + [f"layer_{i}" for i in meta["probe_layers"]] + ["final_logits"]

    words, cur = [], None
    for r in rows:
        if cur is None or r["word"] != cur["w"] or r.get("pos", 0) <= len(cur["toks"]) - 1:
            # new word entry (rows are emitted word-by-word, pos ascending)
            if cur is None or r["word"] != cur["w"] or r.get("pos", 0) == 0:
                cur = {"w": r["word"], "cat": r["category"], "toks": []}
                words.append(cur)
        if r.get("n_tokens", 0) == 0:
            cur["note"] = r.get("note", "no tokens")
            continue
        tok = {"t": r["token"], "probes": {}}
        for p in probes:
            rec = r["probes"].get(p)
            if not rec:
                continue
            tok["probes"][p] = {k: pack_variant(rec.get(k)) for k in ("raw", "normed") if rec.get(k)}
        cur["toks"].append(tok)

    return {
        "name": meta["model"],
        "tied": bool(meta["tied_weights_actual_same_tensor"]),
        "layers": meta["n_layers"],
        "hidden": meta["hidden_size"],
        "vocab": meta["vocab_size_tokenizer"],
        "style": tok_style(rows),
        "probes": probes,
        "words": words,
    }


def main():
    models = []
    for mdir in sorted(glob.glob(os.path.join(RESULTS, "*/"))):
        if not (os.path.exists(os.path.join(mdir, "meta.json"))
                and os.path.exists(os.path.join(mdir, "results.json"))):
            continue
        try:
            m = pack_model(mdir)
        except Exception as e:  # partial/old dirs shouldn't kill the build
            print(f"skip {mdir}: {e}")
            continue
        if m:
            models.append(m)
            print(f"packed {m['name']}: {len(m['words'])} words, {len(m['probes'])} probes")

    payload = json.dumps({"models": models}, separators=(",", ":"), ensure_ascii=False)
    b64 = base64.b64encode(gzip.compress(payload.encode())).decode()
    stamp = datetime.date.today().isoformat()

    html = (TEMPLATE
            .replace("__PAYLOAD__", b64)
            .replace("__STAMP__", stamp)
            .replace("__NMODELS__", str(len(models))))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write(html)
    print(f"\nwrote {OUT}: {len(html)/1024:.0f} KB ({len(models)} models, payload {len(b64)/1024:.0f} KB b64)")


TEMPLATE = r"""<title>Round-Trip Lens</title>
<style>
:root{
  --bg:#F6F7F9; --surface:#FFFFFF; --ink:#1B2430; --muted:#5C6875; --line:#E3E7EC;
  --accent:#0E7C7B; --accent-soft:#0E7C7B22; --seq:#0E7C7B;
  --next:#7C5CBF; --hit-bg:#E3F2EA; --hit-ink:#1C6B45; --chipmax:58;
}
@media (prefers-color-scheme: dark){:root{
  --bg:#0E1319; --surface:#151C24; --ink:#E7ECF2; --muted:#93A1B0; --line:#242E3A;
  --accent:#3FBFB6; --accent-soft:#3FBFB622; --seq:#2FA69E;
  --next:#A78BFA; --hit-bg:#14342A; --hit-ink:#6FD9A6; --chipmax:46;
}}
:root[data-theme="light"]{
  --bg:#F6F7F9; --surface:#FFFFFF; --ink:#1B2430; --muted:#5C6875; --line:#E3E7EC;
  --accent:#0E7C7B; --accent-soft:#0E7C7B22; --seq:#0E7C7B;
  --next:#7C5CBF; --hit-bg:#E3F2EA; --hit-ink:#1C6B45; --chipmax:58;
}
:root[data-theme="dark"]{
  --bg:#0E1319; --surface:#151C24; --ink:#E7ECF2; --muted:#93A1B0; --line:#242E3A;
  --accent:#3FBFB6; --accent-soft:#3FBFB622; --seq:#2FA69E;
  --next:#A78BFA; --hit-bg:#14342A; --hit-ink:#6FD9A6; --chipmax:46;
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);margin:0;
  font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
header{padding:26px 28px 18px;border-bottom:1px solid var(--line)}
header h1{margin:0;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:19px;font-weight:600;letter-spacing:.14em;text-transform:uppercase}
header h1 .dim{color:var(--accent)}
header p{margin:4px 0 0;color:var(--muted);font-size:13.5px;max-width:64ch}
.app{display:grid;grid-template-columns:296px 1fr;gap:0;align-items:start}
@media (max-width:880px){.app{grid-template-columns:1fr}}
.rail{position:sticky;top:0;padding:20px 22px;display:flex;flex-direction:column;gap:16px}
@media (max-width:880px){.rail{position:static}}
.field label{display:block;font-size:11px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--muted);margin-bottom:5px}
select{width:100%;padding:7px 9px;border:1px solid var(--line);border-radius:7px;
  background:var(--surface);color:var(--ink);font:inherit;font-size:14px}
select:focus-visible,button:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.seg{display:flex;border:1px solid var(--line);border-radius:7px;overflow:hidden;background:var(--surface)}
.seg button{flex:1;padding:6px 8px;border:0;background:transparent;color:var(--muted);
  font:inherit;font-size:13px;cursor:pointer}
.seg button[aria-pressed="true"]{background:var(--accent-soft);color:var(--ink);font-weight:600}
.postabs{display:flex;flex-wrap:wrap;gap:6px}
.postabs button{padding:5px 10px;border:1px solid var(--line);border-radius:7px;
  background:var(--surface);color:var(--ink);cursor:pointer;font-size:13px;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;white-space:pre}
.postabs button[aria-pressed="true"]{border-color:var(--accent);background:var(--accent-soft);font-weight:600}
.meta{border:1px solid var(--line);border-radius:9px;background:var(--surface);
  padding:12px 14px;font-size:13px;display:grid;grid-template-columns:auto 1fr;gap:3px 12px}
.meta dt{color:var(--muted)} .meta dd{margin:0;font-variant-numeric:tabular-nums}
.tiedpill{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;font-weight:600}
.tiedpill.tied{background:var(--hit-bg);color:var(--hit-ink)}
.tiedpill.untied{background:var(--accent-soft);color:var(--ink)}
.legend{font-size:12.5px;color:var(--muted);display:flex;flex-direction:column;gap:6px}
.legend .row{display:flex;align-items:center;gap:8px}
.swatch{width:26px;height:16px;border-radius:5px;flex:none;
  background:color-mix(in oklab,var(--seq) 45%,var(--surface))}
.ringdemo{width:26px;height:16px;border-radius:5px;flex:none;background:var(--surface);border:2px solid var(--accent)}
.ringdemo.dashed{border:2px dashed var(--next)}
.stack{padding:20px 28px 40px;min-width:0}
.stackhead{font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin:0 0 10px}
.stackhead b{color:var(--ink);letter-spacing:0;text-transform:none;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:14px}
.probe{display:grid;grid-template-columns:96px minmax(0,1fr) auto;gap:14px;
  padding:9px 0;position:relative}
.probe + .probe{border-top:1px dashed var(--line)}
.plabel{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12.5px;
  color:var(--muted);display:flex;align-items:flex-start;gap:8px;padding-top:6px}
.plabel .node{width:9px;height:9px;border-radius:50%;background:var(--accent);flex:none;margin-top:4px}
.plabel.minor .node{background:var(--line)}
.plabel .nm{white-space:nowrap}
.chips{display:flex;flex-wrap:wrap;gap:6px;min-width:0}
.chip{border:2px solid transparent;border-radius:7px;padding:3px 8px 4px;cursor:default;
  background:color-mix(in oklab,var(--seq) calc(var(--p)*1%),var(--surface));
  display:inline-flex;align-items:baseline;gap:7px;max-width:100%}
.chip .tok{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px;
  white-space:pre;overflow:hidden;text-overflow:ellipsis;max-width:22ch}
.chip .lp{font-size:11px;color:var(--muted);font-variant-numeric:tabular-nums;white-space:nowrap}
.chip.self{border-color:var(--accent)}
.chip.nexttok{border-style:dashed;border-color:var(--next)}
.chip .tag{font-size:10px;font-weight:700;letter-spacing:.06em;text-transform:uppercase}
.chip.self .tag{color:var(--accent)}
.chip.nexttok .tag{color:var(--next)}
.ranks{display:flex;flex-direction:column;gap:4px;align-items:flex-end;padding-top:3px}
.badge{font-size:11.5px;padding:2px 8px;border-radius:99px;background:var(--surface);
  border:1px solid var(--line);color:var(--muted);font-variant-numeric:tabular-nums;white-space:nowrap}
.badge b{color:var(--ink);font-weight:600}
.badge.hit{background:var(--hit-bg);border-color:transparent;color:var(--hit-ink)}
.badge.hit b{color:var(--hit-ink)}
.note{color:var(--muted);font-size:13px;font-style:italic;padding:8px 0}
footer{padding:14px 28px 30px;color:var(--muted);font-size:12px;border-top:1px solid var(--line)}
#tip{position:fixed;z-index:10;pointer-events:none;background:var(--ink);color:var(--bg);
  border-radius:7px;padding:7px 10px;font-size:12px;max-width:340px;display:none;line-height:1.45}
#tip .mono{word-break:break-all}
#tip .k{opacity:.65}
@media (prefers-reduced-motion:no-preference){.chip{transition:border-color .12s}}
</style>

<header>
  <h1>Round-Trip <span class="dim">Lens</span></h1>
  <p>Pick a model and a word: each row unembeds the residual stream at one probe point,
     from the raw embedding down to the model&rsquo;s real output. Chips are the top-10 tokens
     (darker&nbsp;=&nbsp;more probable) with log-probs; badges track where the input token itself
     (&ldquo;self&rdquo;) and the word&rsquo;s true next token (&ldquo;next&rdquo;) rank.</p>
</header>

<div class="app">
<aside class="rail">
  <div class="field"><label for="msel">Model</label><select id="msel"></select></div>
  <div class="field"><label for="wsel">Word</label><select id="wsel"></select></div>
  <div class="field"><label>Token position</label><div class="postabs" id="ptabs"></div></div>
  <div class="field"><label>Unembedding</label>
    <div class="seg" id="vseg">
      <button data-v="normed" aria-pressed="true">logit lens (final norm)</button>
      <button data-v="raw" aria-pressed="false">raw</button>
    </div></div>
  <div class="field"><label>Token display</label>
    <div class="seg" id="dseg">
      <button data-d="decoded" aria-pressed="true">decoded</button>
      <button data-d="tokens" aria-pressed="false">raw tokens</button>
    </div></div>
  <dl class="meta" id="meta"></dl>
  <div class="legend">
    <div class="row"><span class="swatch"></span> fill intensity &prop; probability</div>
    <div class="row"><span class="ringdemo"></span> solid ring = the input token itself</div>
    <div class="row"><span class="ringdemo dashed"></span> dashed ring = word&rsquo;s actual next token</div>
    <div class="row">&ldquo;output&rdquo; row = model&rsquo;s own head (always its true path)</div>
  </div>
</aside>
<main class="stack">
  <p class="stackhead" id="stackhead"></p>
  <div id="rows"></div>
</main>
</div>
<footer>generated __STAMP__ &middot; __NMODELS__ model(s) &middot; rebuild: <span class="mono">python experiments/build_viz.py</span> &middot; data: <span class="mono">results/&lt;model&gt;/results.json</span></footer>
<div id="tip" role="status"></div>

<script>
const PAYLOAD="__PAYLOAD__";
let DATA=null;
const S={m:0,w:0,pos:0,variant:"normed",display:"decoded"};

/* ---------- token decoding (client-side) ---------- */
const byteOf=(()=>{const m={};const inc=new Set();
  const rng=[[33,126],[161,172],[174,255]];
  for(const[a,b]of rng)for(let i=a;i<=b;i++)inc.add(i);
  for(let b=0;b<256;b++)if(inc.has(b))m[String.fromCharCode(b)]=b;
  let n=0;for(let b=0;b<256;b++)if(!inc.has(b)){m[String.fromCharCode(256+n)]=b;n++;}
  return m;})();
const td=new TextDecoder("utf-8",{fatal:false}),te=new TextEncoder();
function decodeBPE(tok){const bytes=[];
  for(const ch of tok){const b=byteOf[ch];
    if(b===undefined){for(const x of te.encode(ch))bytes.push(x);}else bytes.push(b);}
  return td.decode(new Uint8Array(bytes));}
function decodeSP(tok){
  const m=tok.match(/^<0x([0-9A-Fa-f]{2})>$/);
  if(m)return td.decode(new Uint8Array([parseInt(m[1],16)]));
  return tok.replaceAll("▁"," ");}
function visible(s){ // make control chars + edge spaces visible without lying about content
  let out="";
  for(const ch of s){const c=ch.codePointAt(0);
    if(c===0)out+="␀"; else if(c===10)out+="␊"; else if(c===9)out+="␉";
    else if(c<32||c===127)out+=String.fromCodePoint(0x2400+Math.min(c,0x20)); else out+=ch;}
  return out.replace(/^ +| +$/g,m=>"␣".repeat(m.length))||"␣";}
function disp(model,tok){
  if(S.display==="tokens")return tok;
  return visible(model.style==="sp"?decodeSP(tok):decodeBPE(tok));}

/* ---------- rendering ---------- */
const $=id=>document.getElementById(id);
const fmt=n=>n.toLocaleString("en-US");
function probeLabel(p,model){
  if(p==="emb")return"emb";
  if(p==="final_logits")return"output";
  return"L"+p.slice(6);}
function render(){
  const model=DATA.models[S.m],word=model.words[S.w];
  /* meta card */
  $("meta").innerHTML=
    `<dt>weights</dt><dd><span class="tiedpill ${model.tied?"tied":"untied"}">${model.tied?"tied":"untied"}</span></dd>`+
    `<dt>layers</dt><dd>${model.layers}</dd><dt>hidden</dt><dd>${fmt(model.hidden)}</dd>`+
    `<dt>vocab</dt><dd>${fmt(model.vocab)}</dd><dt>spelling</dt><dd>${model.style==="sp"?"SentencePiece ▁":"byte-level BPE Ġ"}</dd>`;
  /* position tabs */
  const pt=$("ptabs");pt.innerHTML="";
  word.toks.forEach((t,i)=>{const b=document.createElement("button");
    b.textContent=disp(model,t.t);b.setAttribute("aria-pressed",String(i===S.pos));
    b.onclick=()=>{S.pos=i;render();};pt.appendChild(b);});
  if(word.note){$("rows").innerHTML=`<p class="note">“${word.w}” ${word.note} in this tokenizer.</p>`;
    $("stackhead").textContent="";return;}
  const tok=word.toks[S.pos],nextTok=word.toks[S.pos+1]?.t??null;
  $("stackhead").innerHTML=`trajectory for token <b>${escapeHtml(disp(model,tok.t))}</b>`+
    ` (${S.pos+1} of ${word.toks.length}${nextTok?`, true next: <b>${escapeHtml(disp(model,nextTok))}</b>`:""})`;
  /* probe rows */
  const rows=$("rows");rows.innerHTML="";
  for(const p of model.probes){
    const rec=tok.probes[p];if(!rec)continue;
    const v=p==="final_logits"?rec.raw:(rec[S.variant]||rec.raw);if(!v)continue;
    const row=document.createElement("div");row.className="probe";
    const major=p==="emb"||p==="final_logits";
    row.innerHTML=`<div class="plabel${major?"":" minor"}"><span class="node"></span><span class="nm">${probeLabel(p,model)}</span></div>`;
    const chips=document.createElement("div");chips.className="chips";
    v.top.forEach(([t,lp],k)=>{
      const c=document.createElement("button");c.className="chip";
      const prob=Math.exp(lp),pmax=getComputedStyle(document.documentElement).getPropertyValue("--chipmax");
      c.style.setProperty("--p",String(Math.max(5,Math.min(+pmax,+pmax*Math.sqrt(prob)))));
      if(t===tok.t)c.classList.add("self");else if(nextTok&&t===nextTok)c.classList.add("nexttok");
      c.innerHTML=`${c.classList.contains("self")?'<span class="tag">self</span>':c.classList.contains("nexttok")?'<span class="tag">next</span>':""}`+
        `<span class="tok">${escapeHtml(disp(model,t))}</span><span class="lp">${lp.toFixed(2).replace("-","−")}</span>`;
      c.addEventListener("pointerenter",e=>tip(e,model,t,lp,k));
      c.addEventListener("pointermove",e=>place(e));
      c.addEventListener("pointerleave",hideTip);
      c.addEventListener("focus",e=>tip(e,model,t,lp,k));c.addEventListener("blur",hideTip);
      chips.appendChild(c);});
    row.appendChild(chips);
    const ranks=document.createElement("div");ranks.className="ranks";
    ranks.innerHTML=`<span class="badge${v.self===1?" hit":""}">self <b>#${fmt(v.self)}</b></span>`+
      (v.next!=null?`<span class="badge${v.next===1?" hit":""}">next <b>#${fmt(v.next)}</b></span>`:"");
    row.appendChild(ranks);rows.appendChild(row);}
}
function escapeHtml(s){return s.replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));}
/* ---------- tooltip ---------- */
function tip(e,model,t,lp,k){const el=$("tip");
  const dec=model.style==="sp"?decodeSP(t):decodeBPE(t);
  el.innerHTML=`<div><span class="k">#${k+1}</span> <span class="mono">${escapeHtml(t)}</span></div>`+
    `<div><span class="k">decoded</span> <span class="mono">${escapeHtml(JSON.stringify(dec))}</span></div>`+
    `<div><span class="k">logprob</span> ${lp.toFixed(2)} <span class="k">&middot; p</span> ${(Math.exp(lp)*100).toPrecision(3)}%</div>`;
  el.style.display="block";place(e);}
function place(e){const el=$("tip"),m=14;let x=e.clientX+m,y=e.clientY+m;
  const r=el.getBoundingClientRect();
  if(x+r.width>innerWidth-8)x=e.clientX-r.width-m;
  if(y+r.height>innerHeight-8)y=e.clientY-r.height-m;
  el.style.left=x+"px";el.style.top=y+"px";}
function hideTip(){$("tip").style.display="none";}
/* ---------- controls ---------- */
function initControls(){
  const ms=$("msel");
  DATA.models.forEach((m,i)=>{const o=document.createElement("option");
    o.value=i;o.textContent=`${m.name}  (${m.tied?"tied":"untied"}, ${m.layers}L)`;ms.appendChild(o);});
  ms.onchange=()=>{S.m=+ms.value;S.w=Math.min(S.w,DATA.models[S.m].words.length-1);S.pos=0;fillWords();render();};
  function fillWords(){
    const ws=$("wsel");ws.innerHTML="";let g=null,cat=null;
    DATA.models[S.m].words.forEach((w,i)=>{
      if(w.cat!==cat){cat=w.cat;g=document.createElement("optgroup");
        g.label=cat.replaceAll("_"," ");ws.appendChild(g);}
      const o=document.createElement("option");o.value=i;
      o.textContent=JSON.stringify(w.w).slice(1,-1)||w.w;g.appendChild(o);});
    ws.value=S.w;
    ws.onchange=()=>{S.w=+ws.value;S.pos=0;render();};}
  fillWords();
  for(const[segId,key]of[["vseg","variant"],["dseg","display"]])
    $(segId).querySelectorAll("button").forEach(b=>b.onclick=()=>{
      S[key]=b.dataset.v||b.dataset.d;
      $(segId).querySelectorAll("button").forEach(x=>x.setAttribute("aria-pressed",String(x===b)));
      render();});
}
/* ---------- boot ---------- */
(async()=>{
  try{
    const bin=Uint8Array.from(atob(PAYLOAD),c=>c.charCodeAt(0));
    const stream=new Blob([bin]).stream().pipeThrough(new DecompressionStream("gzip"));
    DATA=JSON.parse(await new Response(stream).text());
  }catch(err){
    document.querySelector(".stack").innerHTML=`<p class="note">Could not unpack data (${escapeHtml(String(err))}). This page needs a browser with DecompressionStream (2023+).</p>`;
    return;}
  // default to a fun multi-token word if present
  const w0=DATA.models[0].words.findIndex(w=>w.cat==="multi_token_by_design");
  if(w0>0)S.w=w0;
  initControls();$("wsel").value=S.w;render();
})();
</script>
"""

if __name__ == "__main__":
    main()
