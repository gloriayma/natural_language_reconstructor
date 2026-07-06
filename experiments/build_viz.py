#!/usr/bin/env python
"""
Build viz/roundtrip.html — a self-contained interactive explorer for the
embed->unembed round-trip results.

Reads every results/<model>/ dir that has meta.json + results.json (script v2),
packs a compact form of the data (token spellings + logprobs + ranks), gzips it,
base64s it, and injects it into the HTML template below. Decoded token strings
are reconstructed client-side (byte-level-BPE inverse map / SentencePiece rule),
which keeps the payload roughly half the size.

Styling follows gloria.ma (per CLAUDE.md HTML instructions): system-ui, cream
#fffdf8 ground, warm ink #2d2926, tan borders #ded4c7 @ 8px radius, terracotta
#9a5b4f as the single accent (here: the probability ramp), small-caps labels,
italic hints, light-only by design (her site has no dark mode).

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
    if "Ġ" in sample or "Ċ" in sample:
        return "bpe"
    if "▁" in sample:
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
    if "final1_rank" in v:
        out["f1"] = v["final1_rank"]
    return out


def pack_model(mdir):
    meta = json.load(open(os.path.join(mdir, "meta.json")))
    if meta.get("script_version") not in (2, 3):
        return None
    rows = json.load(open(os.path.join(mdir, "results.json")))
    probes = ["emb"] + [f"layer_{i}" for i in meta["probe_layers"]] + ["final_logits"]

    words, cur = [], None
    for r in rows:
        if cur is None or r["word"] != cur["w"] or r.get("pos", 0) == 0:
            cur = {"w": r["word"], "cat": r["category"], "toks": []}
            words.append(cur)
        if r.get("n_tokens", 0) == 0:
            cur["note"] = r.get("note", "no tokens")
            continue
        tok = {"t": r["token"], "probes": {}}
        if r.get("final_top1_token") is not None:
            tok["ft"] = r["final_top1_token"]
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
        except Exception as e:
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


TEMPLATE = r"""<title>round-trip lens</title>
<style>
:root{
  --bg:#fffdf8; --ink:#2d2926; --muted:#6f675f; --line:#ded4c7; --line-strong:#b9ab98;
  --accent:#9a5b4f; --accent-soft:#f3e6e0; --chipmax:52;
  color-scheme: light; /* gloria.ma is light-only; so is this page, deliberately */
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);margin:0;
  font-family:system-ui,sans-serif;font-size:15px;line-height:1.6}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
header{max-width:1140px;margin:0 auto;padding:2.5rem 2rem 1.2rem}
header h1{margin:0;font-size:1.35rem;font-weight:600}
header h1 .dim{color:var(--accent)}
header p{margin:.4rem 0 0;color:var(--muted);font-size:.92rem;max-width:66ch}
.app{max-width:1140px;margin:0 auto;display:grid;grid-template-columns:288px minmax(0,1fr);
  gap:1rem;align-items:start;padding:0 2rem}
@media (max-width:880px){.app{grid-template-columns:1fr}}
.rail{position:sticky;top:1rem;display:flex;flex-direction:column;gap:1rem;padding:.5rem 0}
@media (max-width:880px){.rail{position:static}}
.field label{display:block;font-variant:small-caps;letter-spacing:.08em;
  color:var(--muted);margin-bottom:.3rem}
select{width:100%;padding:.5rem .6rem;border:1px solid var(--line);border-radius:8px;
  background:var(--bg);color:var(--ink);font:inherit;font-size:.92rem}
select:focus,button:focus-visible{outline:none;border-color:var(--line-strong)}
button:focus-visible{outline:2px solid var(--line-strong);outline-offset:1px}
.seg{display:flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}
.seg button{flex:1;padding:.4rem .5rem;border:0;background:transparent;color:var(--muted);
  font:inherit;font-size:.86rem;cursor:pointer}
.seg button[aria-pressed="true"]{background:var(--accent-soft);color:var(--ink);font-weight:600}
.postabs{display:flex;flex-wrap:wrap;gap:.4rem}
.postabs button{padding:.3rem .65rem;border:1px solid var(--line);border-radius:8px;
  background:var(--bg);color:var(--ink);cursor:pointer;font-size:.86rem;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;white-space:pre}
.postabs button[aria-pressed="true"]{border-color:var(--accent);background:var(--accent-soft);font-weight:600}
.meta{border:1px solid var(--line);border-radius:8px;padding:.8rem 1rem;margin:0;
  font-size:.86rem;display:grid;grid-template-columns:auto 1fr;gap:.15rem .8rem}
.meta dt{color:var(--muted);font-variant:small-caps;letter-spacing:.05em}
.meta dd{margin:0;font-variant-numeric:tabular-nums}
.tiedpill{display:inline-block;padding:0 .55rem;border-radius:999px;font-size:.8rem;font-weight:600}
.tiedpill.tied{background:var(--accent-soft);color:var(--accent)}
.tiedpill.untied{border:1px solid var(--line-strong);color:var(--muted)}
.legend{font-size:.82rem;color:var(--muted);font-style:italic;display:flex;
  flex-direction:column;gap:.35rem}
.legend .row{display:flex;align-items:center;gap:.5rem}
.swatch{width:26px;height:15px;border-radius:5px;flex:none;
  background:color-mix(in oklab,var(--accent) 42%,var(--bg))}
.ringdemo{width:26px;height:15px;border-radius:5px;flex:none;background:var(--bg);
  border:2px solid var(--accent)}
.ringdemo.dashed{border:2px dashed var(--ink)}
.stack{padding:.5rem 0 3rem;min-width:0}
.chartcard{border:1px solid var(--line);border-radius:8px;padding:.8rem 1rem .5rem;margin:0 0 1rem}
.chartcard h2{margin:0;font-variant:small-caps;letter-spacing:.08em;color:var(--muted);
  font-size:.9rem;font-weight:600}
.chartcard .sub{margin:0 0 .3rem;font-size:.8rem;color:var(--muted);font-style:italic}
.chartcard svg{display:block;width:100%;height:auto}
.chartlegend{display:flex;gap:1.2rem;font-size:.78rem;color:var(--muted);
  font-style:italic;padding:.2rem 0 .3rem}
.chartlegend .row{display:flex;align-items:center;gap:.45rem}
.chartlegend .ln{width:26px;height:0;border-top:2px solid var(--accent);flex:none}
.chartlegend .ln.dashed{border-top:2px dashed var(--ink)}
.stackhead{font-variant:small-caps;letter-spacing:.08em;color:var(--muted);margin:0 0 .6rem}
.stackhead b{color:var(--ink);letter-spacing:0;font-variant:normal;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.95rem}
.probe{display:grid;grid-template-columns:92px minmax(0,1fr) auto;gap:.9rem;padding:.55rem 0}
.probe + .probe{border-top:1px dashed var(--line)}
.plabel{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.82rem;
  color:var(--muted);display:flex;align-items:flex-start;gap:.5rem;padding-top:.35rem}
.plabel .node{width:9px;height:9px;border-radius:50%;background:var(--accent);flex:none;margin-top:5px}
.plabel.minor .node{background:var(--line)}
.plabel .nm{white-space:nowrap}
.chips{display:flex;flex-wrap:wrap;gap:.4rem;min-width:0}
.chip{border:2px solid transparent;border-radius:8px;padding:.15rem .5rem .2rem;cursor:default;
  background:color-mix(in oklab,var(--accent) calc(var(--p)*1%),var(--bg));
  display:inline-flex;align-items:baseline;gap:.45rem;max-width:100%}
.chip .tok{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.86rem;
  white-space:pre;overflow:hidden;text-overflow:ellipsis;max-width:22ch}
.chip .lp{font-size:.74rem;color:var(--muted);font-variant-numeric:tabular-nums;white-space:nowrap}
.chip.self{border-color:var(--accent)}
.chip.nexttok{border-style:dashed;border-color:var(--ink)}
.chip .tag{font-size:.66rem;font-weight:700;letter-spacing:.06em;font-variant:small-caps}
.chip.self .tag{color:var(--accent)}
.chip.nexttok .tag{color:var(--ink)}
.ranks{display:flex;flex-direction:column;gap:.25rem;align-items:flex-end;padding-top:.2rem}
.badge{font-size:.78rem;padding:.05rem .55rem;border-radius:999px;
  border:1px solid var(--line);color:var(--muted);font-variant-numeric:tabular-nums;white-space:nowrap}
.badge b{color:var(--ink);font-weight:600}
.badge.hit{background:var(--accent-soft);border-color:transparent;color:var(--accent)}
.badge.hit b{color:var(--accent)}
.note{color:var(--muted);font-size:.9rem;font-style:italic;padding:.5rem 0}
footer{max-width:1140px;margin:0 auto;padding:1rem 2rem 2.5rem;color:var(--muted);
  font-size:.8rem;font-style:italic}
#tip{position:fixed;z-index:10;pointer-events:none;background:var(--ink);color:var(--bg);
  border-radius:8px;padding:.45rem .65rem;font-size:.8rem;max-width:340px;display:none;line-height:1.5}
#tip .mono{word-break:break-all}
#tip .k{opacity:.6}
@media (prefers-reduced-motion:no-preference){.chip{transition:border-color .12s}}
</style>

<header>
  <h1>round-trip <span class="dim">lens</span></h1>
  <p>pick a model and a word: each row unembeds the residual stream at one probe point,
     from the raw embedding down to the model&rsquo;s real output. chips are the top-10 tokens
     (darker&nbsp;=&nbsp;more probable) with log-probs; badges track where the input token itself
     (&ldquo;self&rdquo;) and the word&rsquo;s true next token (&ldquo;next&rdquo;) rank.</p>
</header>

<div class="app">
<aside class="rail">
  <div class="field"><label for="msel">model</label><select id="msel"></select></div>
  <div class="field"><label for="wsel">word</label><select id="wsel"></select></div>
  <div class="field"><label>token position</label><div class="postabs" id="ptabs"></div></div>
  <div class="field"><label>unembedding</label>
    <div class="seg" id="vseg">
      <button data-v="normed" aria-pressed="true">logit lens (final norm)</button>
      <button data-v="raw" aria-pressed="false">raw</button>
    </div></div>
  <div class="field"><label>token display</label>
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
  <div id="chart"></div>
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
function visible(s){
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
function probeLabel(p){
  if(p==="emb")return"emb";
  if(p==="final_logits")return"output";
  return"L"+p.slice(6);}
function render(){
  const model=DATA.models[S.m],word=model.words[S.w];
  $("meta").innerHTML=
    `<dt>weights</dt><dd><span class="tiedpill ${model.tied?"tied":"untied"}">${model.tied?"tied":"untied"}</span></dd>`+
    `<dt>layers</dt><dd>${model.layers}</dd><dt>hidden</dt><dd>${fmt(model.hidden)}</dd>`+
    `<dt>vocab</dt><dd>${fmt(model.vocab)}</dd><dt>spelling</dt><dd>${model.style==="sp"?"SentencePiece ▁":"byte-level BPE Ġ"}</dd>`;
  const pt=$("ptabs");pt.innerHTML="";
  word.toks.forEach((t,i)=>{const b=document.createElement("button");
    b.textContent=disp(model,t.t);b.setAttribute("aria-pressed",String(i===S.pos));
    b.onclick=()=>{S.pos=i;render();};pt.appendChild(b);});
  if(word.note){$("rows").innerHTML=`<p class="note">“${word.w}” ${word.note} in this tokenizer.</p>`;
    $("stackhead").textContent="";$("chart").innerHTML="";return;}
  const tok=word.toks[S.pos],nextTok=word.toks[S.pos+1]?.t??null;
  $("stackhead").innerHTML=`trajectory for token <b>${escapeHtml(disp(model,tok.t))}</b>`+
    ` (${S.pos+1} of ${word.toks.length}${nextTok?`, true next: <b>${escapeHtml(disp(model,nextTok))}</b>`:""})`;
  const rows=$("rows");rows.innerHTML="";
  for(const p of model.probes){
    const rec=tok.probes[p];if(!rec)continue;
    const v=p==="final_logits"?rec.raw:(rec[S.variant]||rec.raw);if(!v)continue;
    const row=document.createElement("div");row.className="probe";
    const major=p==="emb"||p==="final_logits";
    row.innerHTML=`<div class="plabel${major?"":" minor"}"><span class="node"></span><span class="nm">${probeLabel(p)}</span></div>`;
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
  renderChart(model,tok);
}
/* ---------- rank-trajectory chart ---------- */
function gridLabel(d){return d<3?String(10**d):d<6?(10**(d-3))+"k":(10**(d-6))+"M";}
function renderChart(model,tok){
  const host=$("chart");
  const pts=[];
  for(const p of model.probes){
    const rec=tok.probes[p];if(!rec)continue;
    const v=p==="final_logits"?rec.raw:(rec[S.variant]||rec.raw);if(!v)continue;
    pts.push({x:p==="emb"?0:p==="final_logits"?model.layers+1:+p.slice(6),
      label:probeLabel(p),self:v.self,f1:v.f1});}
  if(pts.length<2){host.innerHTML="";return;}
  const hasF1=pts.some(q=>q.f1!=null);
  const W=760,H=240,L=46,R=86,T=14,B=32,pw=W-L-R,ph=H-T-B;
  const dec=Math.max(1,Math.ceil(Math.log10(model.vocab)));
  const xmax=model.layers+1;
  const px=x=>L+x/xmax*pw, py=r=>T+Math.log10(Math.max(1,r))/dec*ph;
  let g="";
  for(let d=0;d<=dec;d++){const y=T+d/dec*ph;
    g+=`<line x1="${L}" y1="${y}" x2="${W-R}" y2="${y}" style="stroke:var(--line)" stroke-width="1"/>`+
       `<text x="${L-7}" y="${y+3.5}" text-anchor="end" font-size="10.5" style="fill:var(--muted)">${gridLabel(d)}</text>`;}
  for(const q of pts)
    g+=`<text x="${px(q.x)}" y="${H-B+16}" text-anchor="middle" font-size="10" style="fill:var(--muted)">${q.label==="output"?"out":q.label.replace("L","")}</text>`;
  g+=`<text x="${L-7}" y="${T-4}" text-anchor="end" font-size="9.5" font-style="italic" style="fill:var(--muted)">rank</text>`;
  const series=[["self","var(--accent)","",q=>q.self]];
  if(hasF1)series.push(["final top-1","var(--ink)","5 4",q=>q.f1]);
  let marks="";
  for(const[nm,col,dash,get]of series){
    const line=pts.filter(q=>get(q)!=null).map(q=>`${px(q.x)},${py(get(q))}`).join(" ");
    g+=`<polyline points="${line}" fill="none" style="stroke:${col}" stroke-width="2"${dash?` stroke-dasharray="${dash}"`:""}/>`;
    const last=pts[pts.length-1];
    g+=`<text x="${px(last.x)+8}" y="${py(get(last))+3.5}" font-size="10.5" style="fill:var(--ink)">${nm}</text>`;
    for(const q of pts){if(get(q)==null)continue;
      marks+=`<circle cx="${px(q.x)}" cy="${py(get(q))}" r="4" style="fill:var(--bg);stroke:${col}" stroke-width="2"/>`+
        `<circle class="pt" data-nm="${nm}" data-lb="${q.label}" data-r="${get(q)}" cx="${px(q.x)}" cy="${py(get(q))}" r="10" fill="transparent"/>`;}}
  const sub=hasF1&&tok.ft!=null
    ?`final top-1 = what the model actually predicts after this token: <span class="mono">${escapeHtml(disp(model,tok.ft))}</span> &middot; rank 1 is at the top, log scale`
    :`rank 1 is at the top, log scale`;
  host.innerHTML=`<section class="chartcard"><h2>rank trajectory</h2>
    <p class="sub">${sub}</p>
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="rank of self and final top-1 token across layers">${g}${marks}</svg>
    <div class="chartlegend"><span class="row"><span class="ln"></span> self (the input token)</span>${hasF1?'<span class="row"><span class="ln dashed"></span> final top-1</span>':""}</div>
    </section>`;
  host.querySelectorAll(".pt").forEach(c=>{
    const show=e=>{const el=$("tip");
      el.innerHTML=`<div>${c.dataset.lb} &middot; ${c.dataset.nm}</div><div><span class="k">rank</span> #${fmt(+c.dataset.r)}</div>`;
      el.style.display="block";place(e);};
    c.addEventListener("pointerenter",show);c.addEventListener("pointermove",place);
    c.addEventListener("pointerleave",hideTip);});
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
  const w0=DATA.models[0].words.findIndex(w=>w.cat==="multi_token_by_design");
  if(w0>0)S.w=w0;
  initControls();$("wsel").value=S.w;render();
})();
</script>
"""

if __name__ == "__main__":
    main()
