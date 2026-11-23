"""Build the single-file Chathouse medallion pipeline dashboard.

Reads output/metrics/bundle.json (from scripts/compute_metrics.py) and writes
dashboard/chathouse_medallion_dashboard.html with Plotly inlined.
"""
import datetime as dt
import json
import os
from collections import defaultdict
from pathlib import Path

import plotly

ROOT = Path(__file__).resolve().parents[1]
B = json.loads((ROOT / "output" / "metrics" / "bundle.json").read_text())
PLOTLY = (Path(plotly.__file__).parent / "package_data" / "plotly.min.js").read_text().replace("</script", "<\\/script")

n = lambda x: f"{int(round(x)):,}"
pct = lambda x, d=1: f"{x * 100:.{d}f}%"

# ---------- derived numbers ----------
R, D = B["rows"], B["distinct"]
daily = B["gold_daily_activity"]
last_date = B["span"][1][:10]
partial = [d["event_date"] for d in daily if d["event_date"] == last_date and B["span"][1][11:13] < "23"]
full = [d for d in daily if d["event_date"] not in partial]
week1 = sum(d["conversations"] for d in full[:7]) / 7
week_last = sum(d["conversations"] for d in full[-7:]) / 7
growth = week_last / week1 - 1

models = defaultdict(int)
for m in B["gold_model_mix_daily"]:
    models[m["model"]] += m["conversations"]
tot_conv = sum(models.values())
gpt4 = models["gpt-4-0314"] / tot_conv
g4 = [m["share_of_day"] for m in B["gold_model_mix_daily"] if m["model"] == "gpt-4-0314"]

turns = B["gold_turns_distribution"]
one_turn = turns[0]["conversations"] / tot_conv
multi = 1 - one_turn
ret = B["gold_return_rates"]
ret_overall = sum(r["returned_within_7_days"] for r in ret) / sum(r["new_hashed_ips"] for r in ret)
ret_min = min(r["return_rate_7d"] for r in ret)
ret_max = max(r["return_rate_7d"] for r in ret)

H = B["heavy"]
heavy_share_ip = H["heavy"] / H["ips"]
heavy_share_conv = H["heavy_convs"] / H["convs"]
conc = B["gold_user_concentration"]
single_ips = conc[0]["hashed_ips"]

langs = sorted(B["gold_language_mix"], key=lambda x: -x["conversations"])
ctry = sorted(B["gold_country_mix"], key=lambda x: -x["conversations"])

dates = [dt.date.fromisoformat(d["event_date"]) for d in daily]
occ = defaultdict(int)
for d in dates:
    occ[d.isoweekday()] += 1
heat = [[0.0] * 24 for _ in range(7)]
for h in B["gold_hourly_pattern"]:
    heat[h["iso_weekday"] - 1][h["hour_utc"]] = h["conversations"] / occ[h["iso_weekday"]]
by_hour = [sum(heat[w][h] for w in range(7)) for h in range(24)]
peak_h, low_h = max(range(24), key=lambda i: by_hour[i]), min(range(24), key=lambda i: by_hour[i])
wk = [sum(heat[w]) for w in range(7)]
wd_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
peak_wd, low_wd = max(range(7), key=lambda i: wk[i]), min(range(7), key=lambda i: wk[i])

# heavy IP sensitivity on daily metrics
ip_drop = 1 - sum(d["active_hashed_ips_excl_heavy"] for d in full) / sum(d["active_hashed_ips"] for d in full)
conv_drop = sum(d["heavy_ip_conversations"] for d in full) / sum(d["conversations"] for d in full)

dq = B["dq"]
dq_pass = sum(d["status"] == "PASS" for d in dq)
rec = B["recon"]
rec_ok = sum(r["match"] for r in rec)
dims = defaultdict(lambda: dict(failed=0, total=0, checks=0, passed=0))
for d in dq:
    x = dims[d["dimension"]]
    x["failed"] += d["failed"]; x["total"] += d["total"]; x["checks"] += 1; x["passed"] += d["status"] == "PASS"
dim_rows = [dict(dimension=k, score=1 - v["failed"] / v["total"], checks=v["checks"], passed=v["passed"]) for k, v in dims.items()]
dim_rows.append(dict(dimension="Reconciliation", score=rec_ok / len(rec), checks=len(rec), passed=rec_ok))
dim_rows.sort(key=lambda r: -r["score"])
gold_rows = {k: len(v) for k, v in B.items() if k.startswith("gold_")}
excluded = R["bronze_conv"] - R["valid_conv"]
t = B["timings"]
t_total = sum(t.values())

DATA = dict(B, partial=partial, heat=heat, dimRows=dim_rows, wd=wd_names)

I = dict(  # insight text
    daily=f"Volume grows through the window. The last full week averages {n(week_last)} conversations a day against {n(week1)} in the first week, a gain of {pct(growth, 0)}.",
    models=f"GPT-3.5 handles {pct(1 - gpt4)} of conversations and GPT-4 handles {pct(gpt4)}. GPT-4 opens at {pct(g4[0], 0)} on day one, then ranges from {pct(min(g4[1:]), 0)} to {pct(max(g4[1:]), 0)}.",
    turns=f"{pct(one_turn)} of conversations stop after one turn. {pct(multi)} continue, and {pct(turns[3]['conversations'] / tot_conv)} run past 10 turns.",
    ret=f"{pct(ret_overall)} of new hashed IPs return within 7 days. Daily cohorts range from {pct(ret_min)} to {pct(ret_max)}.",
    conc=f"{n(H['heavy'])} heavy hashed IPs ({pct(heavy_share_ip, 2)} of all) hold {pct(heavy_share_conv)} of conversations. {pct(single_ips / H['ips'])} of IPs send one conversation.",
    lang=f"{langs[0]['language']} leads with {pct(langs[0]['share'])}, then {langs[1]['language']} at {pct(langs[1]['share'])} and {langs[2]['language']} at {pct(langs[2]['share'])}. {ctry[0]['country']} is the top country at {pct(ctry[0]['share'])}.",
    heat=f"Activity peaks at {peak_h:02d}:00 UTC and bottoms at {low_h:02d}:00 UTC. {wd_names[peak_wd]} is the busiest day and {wd_names[low_wd]} the quietest, per calendar day.",
    prompt=f"The median user prompt has {n(B['prompt']['med'])} characters. The mean has {n(B['prompt']['mean'])}. The longest 5% exceed {n(B['prompt']['p95'])} characters.",
    heavy=f"Removing heavy IPs lowers daily active IPs by {pct(ip_drop)} and daily conversations by {pct(conv_drop)}. User counts stay stable. Volume metrics need the heavy-IP flag.",
)

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chathouse Medallion Pipeline Dashboard</title>
<style>
:root{--navy:#14213D;--red:#D7263D;--ink:#1F2937;--mute:#667085;--line:#E5E7EB;--bg:#F5F6F9;--card:#fff;--bronze:#B5733A;--silver:#8A93A6;--gold:#C9971C;--ok:#1E8E5A;--okbg:#E6F4EC;--slate:#6B7A99;--soft:#A9B9D6}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
header{background:var(--navy);color:#fff;padding:36px 0 30px;border-bottom:4px solid var(--red)}
.wrap{max-width:1180px;margin:0 auto;padding:0 20px}
header h1{margin:0 0 6px;font-size:30px;letter-spacing:-.3px}
header p{margin:0;color:#C9D2E6;max-width:760px}
.badges{margin-top:16px;display:flex;flex-wrap:wrap;gap:8px}
.badge{background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.22);border-radius:99px;padding:4px 12px;font-size:13px}
.badge.ok{background:var(--okbg);color:var(--ok);border-color:#BFE3CF;font-weight:600}
nav{position:sticky;top:0;z-index:10;background:#fff;border-bottom:1px solid var(--line)}
nav .wrap{display:flex;gap:4px;overflow-x:auto}
nav a{padding:13px 16px;color:var(--mute);text-decoration:none;font-weight:600;font-size:14px;border-bottom:3px solid transparent;white-space:nowrap}
nav a:hover{color:var(--navy)}nav a.on{color:var(--navy);border-color:var(--red)}
section{padding:34px 0 6px}
h2{margin:0 0 4px;font-size:22px;color:var(--navy)}
.lead{margin:0 0 18px;color:var(--mute);max-width:820px}
.kpis{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-top:22px}
.kpi{background:var(--card);border-radius:10px;padding:14px 16px;border:1px solid var(--line);box-shadow:0 1px 2px rgba(16,24,40,.05)}
.kpi b{display:block;font-size:26px;color:var(--navy);line-height:1.15}
.kpi span{font-size:12.5px;color:var(--mute)}
.kpi.good b{color:var(--ok)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin-bottom:16px;min-width:0;overflow-x:auto}
.card h3{margin:0 0 2px;font-size:16px;color:var(--navy)}
.take{margin:0 0 8px;color:var(--mute);font-size:13.5px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.grid2>.card{margin-bottom:0}.grid2{min-width:0}.gap{height:16px}
.flow{display:grid;grid-template-columns:1fr 84px 1fr 84px 1fr;align-items:stretch;gap:0}
.layer{border-radius:12px;padding:16px 18px;color:#fff;position:relative}
.layer.b{background:linear-gradient(160deg,#C4824A,#9C5F2C)}.layer.s{background:linear-gradient(160deg,#9AA3B6,#6F788C)}.layer.g{background:linear-gradient(160deg,#D8A92B,#B07F0C)}
.layer h4{margin:0 0 2px;font-size:18px}.layer .sub{font-size:12.5px;opacity:.9;margin-bottom:10px}
.layer ul{list-style:none;margin:0;padding:0;font-size:13px}
.layer li{display:flex;justify-content:space-between;gap:8px;padding:4px 0;border-top:1px solid rgba(255,255,255,.25)}
.layer li b{font-variant-numeric:tabular-nums}
.arrow{display:flex;flex-direction:column;align-items:center;justify-content:center;color:var(--mute);font-size:11px;text-align:center;padding:0 6px;line-height:1.3}
.arrow i{font-style:normal;font-size:26px;color:var(--navy);line-height:1}
table{width:100%;border-collapse:collapse;font-size:13.5px}@media(max-width:900px){table{min-width:560px}}
th{text-align:left;color:var(--mute);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.4px;border-bottom:2px solid var(--line);padding:8px 8px}
td{padding:8px;border-bottom:1px solid var(--line);vertical-align:middle}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.pill{display:inline-block;padding:2px 10px;border-radius:99px;font-size:12px;font-weight:700}
.pill.pass{background:var(--okbg);color:var(--ok)}.pill.fail{background:#FDE8EA;color:var(--red)}
.meter{height:8px;background:var(--line);border-radius:8px;overflow:hidden;min-width:90px}.meter i{display:block;height:100%;background:var(--navy)}
.dim{display:grid;grid-template-columns:130px 1fr 64px 54px;gap:10px;align-items:center;margin:9px 0;font-size:13.5px}
.dim .meter{height:12px}.dim .meter i{background:var(--ok)}
.tiles{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:6px}
.tile{background:var(--bg);border-radius:8px;padding:12px;text-align:center}.tile b{display:block;font-size:22px;color:var(--navy)}.tile span{font-size:12px;color:var(--mute)}
.chart{width:100%}
.note{font-size:12.5px;color:var(--mute)}
footer{padding:26px 0 40px;color:var(--mute);font-size:12.5px}
footer a{color:var(--navy)}
@media(max-width:900px){.kpis{grid-template-columns:repeat(2,1fr)}.grid2{grid-template-columns:1fr}.flow{grid-template-columns:1fr}.arrow{flex-direction:row;gap:8px;padding:8px 0}.arrow i{transform:rotate(90deg)}header h1{font-size:24px}.dim{grid-template-columns:100px 1fr 54px}.dim span:last-child{display:none}}
@media print{nav{display:none}.card{break-inside:avoid}}
</style>
</head>
<body>
<header><div class="wrap">
<h1>Chathouse Medallion Pipeline Dashboard</h1>
<p>Bronze, silver and gold layers of the EDA analytics model built on WildChat conversation logs, with the evaluation metrics of every layer.</p>
<div class="badges">
<span class="badge">Coverage: __SPAN__ UTC</span>
<span class="badge">__NCONV__ conversations in gold</span>
<span class="badge ok">__DQP__ of __DQN__ quality checks pass</span>
<span class="badge ok">__RECOK__ of __RECN__ reconciliations match</span>
</div></div></header>
<nav><div class="wrap"><a href="#pipeline" class="on">Pipeline</a><a href="#analytics">Analytics</a><a href="#evaluation">Evaluation</a><a href="#catalog">Gold catalog</a></div></nav>
<div class="wrap">
<div class="kpis">
<div class="kpi"><b>__NCONV__</b><span>Valid conversations</span></div>
<div class="kpi"><b>__NIPS__</b><span>Hashed IPs</span></div>
<div class="kpi"><b>__NLANG__</b><span>Languages</span></div>
<div class="kpi"><b>__NCTRY__</b><span>Countries</span></div>
<div class="kpi good"><b>__DQP__/__DQN__</b><span>Quality checks pass</span></div>
<div class="kpi good"><b>__RECOK__/__RECN__</b><span>Reconciliations match</span></div>
</div>

<section id="pipeline">
<h2>Medallion pipeline</h2>
<p class="lead">Raw logs land in bronze. Silver types, flattens and cleans them. Gold holds eight aggregate tables that feed every chart on this page.</p>
<div class="flow">
<div class="layer b"><h4>Bronze</h4><div class="sub">Raw Parquet, schema as received</div><ul>
<li><span>Conversations</span><b>__BCONV__</b></li><li><span>Utterances (nested)</span><b>__BUTT__</b></li><li><span>Columns</span><b>__BCOLS__</b></li><li><span>Source files</span><b>Parquet</b></li></ul></div>
<div class="arrow"><i>&rarr;</i>Flatten, type, default nulls</div>
<div class="layer s"><h4>Silver</h4><div class="sub">Conformed, keyed and validated</div><ul>
<li><span>silver_conversations</span><b>__SCONV__</b></li><li><span>silver_utterances</span><b>__SUTT__</b></li><li><span>dim_user_proxy</span><b>__NIPS__</b></li><li><span>valid_conversations</span><b>__NCONV__</b></li></ul></div>
<div class="arrow"><i>&rarr;</i>Aggregate valid rows</div>
<div class="layer g"><h4>Gold</h4><div class="sub">Eight analytics tables</div><ul>
__GOLDLIST__</ul></div>
</div>
<div class="gap"></div>
<div class="grid2">
<div class="card"><h3>Row retention by layer</h3><p class="take">Silver keeps every bronze row. The valid view drops __EXCL__ conversations (__EXCLP__) with an empty first input.</p><div id="c-retention" class="chart" style="height:290px"></div></div>
<div class="card"><h3>Stage build time</h3><p class="take">The bronze scan and silver build take __TTOT__ seconds in total on a four-core machine.</p><div id="c-timing" class="chart" style="height:290px"></div></div>
</div>
</section>

<section id="analytics">
<h2>EDA analytics</h2>
<p class="lead">Each chart reads one gold table. The last day (4 May) ends at 07:49 UTC and appears in a lighter shade.</p>
<div class="card"><h3>Daily conversations and active hashed IPs</h3><p class="take">__I_DAILY__</p><div id="c-daily" class="chart" style="height:380px"></div></div>
<div class="grid2">
<div class="card"><h3>Model mix by day</h3><p class="take">__I_MODELS__</p><div id="c-models" class="chart" style="height:340px"></div></div>
<div class="card"><h3>Conversation length in turns</h3><p class="take">__I_TURNS__</p><div id="c-turns" class="chart" style="height:340px"></div></div>
</div>
<div class="gap"></div>
<div class="grid2">
<div class="card"><h3>7-day return rate by first-seen cohort</h3><p class="take">__I_RET__</p><div id="c-return" class="chart" style="height:340px"></div></div>
<div class="card"><h3>User concentration</h3><p class="take">__I_CONC__</p><div id="c-conc" class="chart" style="height:340px"></div></div>
</div>
<div class="gap"></div>
<div class="grid2">
<div class="card"><h3>Top languages</h3><p class="take">__I_LANG__</p><div id="c-lang" class="chart" style="height:380px"></div></div>
<div class="card"><h3>Top countries</h3><p class="take">Country comes from IP geolocation. __NCTRY__ countries appear in the logs.</p><div id="c-ctry" class="chart" style="height:380px"></div></div>
</div>
<div class="gap"></div>
<div class="card"><h3>Activity by weekday and hour (UTC)</h3><p class="take">__I_HEAT__</p><div id="c-heat" class="chart" style="height:330px"></div></div>
<div class="card"><h3>Prompt length</h3><p class="take">__I_PROMPT__</p>
<div class="tiles"><div class="tile"><b>__PMED__</b><span>Median characters</span></div><div class="tile"><b>__PMEAN__</b><span>Mean characters</span></div><div class="tile"><b>__PP95__</b><span>95th percentile</span></div></div></div>
</section>

<section id="evaluation">
<h2>Evaluation metrics</h2>
<p class="lead">Every layer is scored on completeness, validity, uniqueness, consistency and reconciliation. A check passes when its failed share stays within its threshold.</p>
<div class="grid2">
<div class="card"><h3>Quality score by dimension</h3><p class="take">Score is the share of rows that pass. Reconciliation is the share of matched totals.</p><div id="dimbars"></div></div>
<div class="card"><h3>Completeness by bronze column</h3><p class="take">Twelve of fourteen columns have no nulls. State is the only gap above 1%.</p><div id="c-nulls" class="chart" style="height:300px"></div></div>
</div>
<div class="gap"></div>
<div class="card"><h3>Data quality checks</h3><p class="take">All __DQN__ checks sit inside their thresholds. The bar shows how much of the allowed failure share each check uses.</p>
<table><thead><tr><th>Check</th><th>Dimension</th><th class="num">Rows failed</th><th class="num">Failed share</th><th class="num">Threshold</th><th>Threshold use</th><th>Status</th></tr></thead><tbody id="dqrows"></tbody></table></div>
<div class="card"><h3>Gold to silver reconciliation</h3><p class="take">Every gold table sums back to the silver source. No rows gain or loss between layers.</p>
<table><thead><tr><th>Check</th><th class="num">Upstream</th><th class="num">Downstream</th><th>Status</th></tr></thead><tbody id="recrows"></tbody></table></div>
<div class="card"><h3>Metric stability</h3><p class="take">__I_HEAVY__</p>
<div class="tiles"><div class="tile"><b>__IPDROP__</b><span>Daily active IPs lost without heavy IPs</span></div><div class="tile"><b>__CONVDROP__</b><span>Daily conversations lost without heavy IPs</span></div><div class="tile"><b>__RETALL__</b><span>7-day return rate (all cohorts)</span></div></div></div>
</section>

<section id="catalog">
<h2>Gold catalog</h2>
<p class="lead">Eight tables serve the analytics layer. Each table reconciles to valid_conversations.</p>
<div class="card"><table><thead><tr><th>Table</th><th>Grain</th><th class="num">Rows</th><th>Feeds</th></tr></thead><tbody>
__CATALOG__
</tbody></table></div>
</section>
<footer>Source data: WildChat-1M, Allen Institute for AI (<a href="https://huggingface.co/datasets/allenai/WildChat-1M">huggingface.co/datasets/allenai/WildChat-1M</a>). Stack: DuckDB, Parquet, Plotly. IPs stay hashed in the source and appear here only as counts. Code: <a href="https://github.com/ambarishsengupta/Chathouse">github.com/ambarishsengupta/Chathouse</a>.</footer>
</div>
<script>__PLOTLY__</script>
<script>
const D=__DATA__;
const C={navy:'#14213D',red:'#D7263D',slate:'#6B7A99',soft:'#A9B9D6',line:'#E5E7EB',amber:'#C9971C'};
const base={font:{family:'-apple-system,Segoe UI,Roboto,Arial,sans-serif',size:12.5,color:'#1F2937'},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',margin:{l:56,r:16,t:8,b:44},legend:{orientation:'h',y:1.12,x:0}};
const ax={gridcolor:C.line,zeroline:false,linecolor:C.line};
const cfg={displayModeBar:false,responsive:true};
const lay=(o)=>Object.assign({},base,o);
const fmt=d3=>d3;
const dates=D.gold_daily_activity.map(d=>d.event_date);
const isPart=d=>D.partial.includes(d);
Plotly.newPlot('c-daily',[
 {type:'bar',name:'Conversations',x:dates,y:D.gold_daily_activity.map(d=>d.conversations),marker:{color:dates.map(d=>isPart(d)?C.soft:C.navy)},hovertemplate:'%{x}<br>%{y:,} conversations<extra></extra>'},
 {type:'scatter',mode:'lines+markers',name:'Active hashed IPs',x:dates,y:D.gold_daily_activity.map(d=>d.active_hashed_ips),line:{color:C.red,width:2.5},marker:{size:5},hovertemplate:'%{x}<br>%{y:,} hashed IPs<extra></extra>'}
],lay({xaxis:Object.assign({},ax,{tickformat:'%d %b',nticks:13}),yaxis:Object.assign({},ax,{title:'Count'})}),cfg);
const mods=[...new Set(D.gold_model_mix_daily.map(m=>m.model))].sort();
const mcol={'gpt-3.5-turbo-0301':C.soft,'gpt-4-0314':C.navy};
Plotly.newPlot('c-models',mods.map(m=>({type:'bar',name:m,x:D.gold_model_mix_daily.filter(r=>r.model==m).map(r=>r.event_date),y:D.gold_model_mix_daily.filter(r=>r.model==m).map(r=>r.share_of_day),marker:{color:mcol[m]||C.slate},hovertemplate:'%{x}<br>%{y:.1%}<extra>'+m+'</extra>'})),
 lay({barmode:'stack',xaxis:Object.assign({},ax,{tickformat:'%d %b',nticks:8}),yaxis:Object.assign({},ax,{tickformat:'.0%',range:[0,1]})}),cfg);
const tt=D.gold_turns_distribution,tsum=tt.reduce((a,b)=>a+b.conversations,0);
Plotly.newPlot('c-turns',[{type:'bar',x:tt.map(t=>t.turn_bucket),y:tt.map(t=>t.conversations),marker:{color:[C.navy,C.slate,C.soft,C.red]},text:tt.map(t=>(t.conversations/tsum*100).toFixed(1)+'%'),textposition:'outside',cliponaxis:false,hovertemplate:'%{x} turns<br>%{y:,} conversations<extra></extra>'}],
 lay({margin:{l:56,r:16,t:22,b:44},xaxis:Object.assign({},ax,{title:'Turns per conversation'}),yaxis:Object.assign({},ax,{title:'Conversations',range:[0,Math.max(...tt.map(t=>t.conversations))*1.15]})}),cfg);
const rr=D.gold_return_rates;
Plotly.newPlot('c-return',[
 {type:'scatter',mode:'lines+markers',name:'Return rate',x:rr.map(r=>r.cohort_date),y:rr.map(r=>r.return_rate_7d),line:{color:C.navy,width:2.5},marker:{size:6},hovertemplate:'%{x}<br>%{y:.1%}<extra></extra>'},
 {type:'scatter',mode:'lines',name:'All cohorts',x:[rr[0].cohort_date,rr[rr.length-1].cohort_date],y:[__RETRAW__,__RETRAW__],line:{color:C.red,dash:'dash',width:2},hoverinfo:'skip'}
],lay({xaxis:Object.assign({},ax,{tickformat:'%d %b',nticks:8}),yaxis:Object.assign({},ax,{tickformat:'.0%',rangemode:'tozero',title:'Returned within 7 days'})}),cfg);
const cc=D.gold_user_concentration,ipT=cc.reduce((a,b)=>a+b.hashed_ips,0),cvT=cc.reduce((a,b)=>a+b.conversations,0);
Plotly.newPlot('c-conc',[
 {type:'bar',name:'Share of hashed IPs',x:cc.map(c=>c.conversations_per_ip),y:cc.map(c=>c.hashed_ips/ipT),marker:{color:C.soft},hovertemplate:'%{x} conversations per IP<br>%{y:.1%} of IPs<extra></extra>'},
 {type:'bar',name:'Share of conversations',x:cc.map(c=>c.conversations_per_ip),y:cc.map(c=>c.conversations/cvT),marker:{color:C.navy},hovertemplate:'%{x} conversations per IP<br>%{y:.1%} of conversations<extra></extra>'}
],lay({barmode:'group',xaxis:Object.assign({},ax,{title:'Conversations per hashed IP'}),yaxis:Object.assign({},ax,{tickformat:'.0%'})}),cfg);
const hb=(id,rows,key,color)=>{const r=rows.slice(0,10).reverse();Plotly.newPlot(id,[{type:'bar',orientation:'h',y:r.map(x=>x[key]),x:r.map(x=>x.share),marker:{color:color},text:r.map(x=>(x.share*100).toFixed(1)+'%'),textposition:'outside',cliponaxis:false,hovertemplate:'%{y}<br>%{x:.1%}<extra></extra>'}],lay({margin:{l:110,r:48,t:8,b:36},xaxis:Object.assign({},ax,{tickformat:'.0%',range:[0,Math.max(...r.map(x=>x.share))*1.18]}),yaxis:Object.assign({},ax,{gridcolor:'rgba(0,0,0,0)'})}),cfg)};
hb('c-lang',[...D.gold_language_mix].sort((a,b)=>b.conversations-a.conversations),'language',C.navy);
hb('c-ctry',[...D.gold_country_mix].sort((a,b)=>b.conversations-a.conversations),'country',C.slate);
Plotly.newPlot('c-heat',[{type:'heatmap',z:D.heat,x:[...Array(24).keys()].map(h=>String(h).padStart(2,'0')),y:D.wd,colorscale:[[0,'#EEF1F8'],[.5,'#6B7A99'],[1,'#14213D']],colorbar:{title:'Conv. per day',thickness:12},hovertemplate:'%{y} %{x}:00 UTC<br>%{z:.0f} conversations per day<extra></extra>'}],
 lay({margin:{l:90,r:10,t:8,b:44},xaxis:Object.assign({},ax,{title:'Hour of day (UTC)'}),yaxis:Object.assign({},ax,{autorange:'reversed',gridcolor:'rgba(0,0,0,0)'})}),cfg);
// pipeline charts
const R=D.rows;
Plotly.newPlot('c-retention',[{type:'bar',orientation:'h',y:['Gold (valid)','Silver','Bronze'],x:[R.valid_conv,R.silver_conv,R.bronze_conv],marker:{color:['#C9971C','#8A93A6','#B5733A']},text:[R.valid_conv,R.silver_conv,R.bronze_conv].map(v=>v.toLocaleString('en-US')),textposition:'inside',insidetextanchor:'end',textfont:{color:'#fff',size:13},hovertemplate:'%{y}<br>%{x:,} conversations<extra></extra>'}],
 lay({margin:{l:100,r:16,t:8,b:36},xaxis:Object.assign({},ax,{title:'Conversations',range:[0,R.bronze_conv*1.02]}),yaxis:Object.assign({},ax,{gridcolor:'rgba(0,0,0,0)'})}),cfg);
const tk=Object.keys(D.timings);
Plotly.newPlot('c-timing',[{type:'bar',orientation:'h',y:tk.slice().reverse(),x:tk.map(k=>D.timings[k]).reverse(),marker:{color:C.navy},text:tk.map(k=>D.timings[k].toFixed(2)+' s').reverse(),textposition:'outside',cliponaxis:false,hovertemplate:'%{y}<br>%{x:.2f} s<extra></extra>'}],
 lay({margin:{l:150,r:50,t:8,b:36},xaxis:Object.assign({},ax,{title:'Seconds',range:[0,Math.max(...tk.map(k=>D.timings[k]))*1.25]}),yaxis:Object.assign({},ax,{gridcolor:'rgba(0,0,0,0)'})}),cfg);
// evaluation
document.getElementById('dimbars').innerHTML=D.dimRows.map(r=>`<div class="dim"><b>${r.dimension}</b><div class="meter"><i style="width:${(r.score*100).toFixed(1)}%"></i></div><span><b>${(r.score*100).toFixed(1)}%</b></span><span class="note">${r.passed}/${r.checks}</span></div>`).join('');
const nr=Object.entries(D.null_rates).sort((a,b)=>b[1]-a[1]).slice(0,6).reverse();
Plotly.newPlot('c-nulls',[{type:'bar',orientation:'h',y:nr.map(x=>x[0]),x:nr.map(x=>x[1]),marker:{color:nr.map(x=>x[1]>0.01?C.red:C.soft)},text:nr.map(x=>(x[1]*100).toFixed(2)+'%'),textposition:'outside',cliponaxis:false,hovertemplate:'%{y}<br>%{x:.2%} null<extra></extra>'}],
 lay({margin:{l:130,r:56,t:8,b:36},xaxis:Object.assign({},ax,{tickformat:'.0%',range:[0,0.22],title:'Null share'}),yaxis:Object.assign({},ax,{gridcolor:'rgba(0,0,0,0)'})}),cfg);
const f=(x)=>x.toLocaleString('en-US');
document.getElementById('dqrows').innerHTML=D.dq.map(d=>{const use=d.limit>0?Math.min(1,d.share/d.limit):0;return `<tr><td>${d.check}</td><td>${d.dimension}</td><td class="num">${f(d.failed)}</td><td class="num">${(d.share*100).toFixed(2)}%</td><td class="num">${d.limit===0?'0%':(d.limit*100).toFixed(0)+'%'}</td><td><div class="meter"><i style="width:${(use*100).toFixed(0)}%;${use>0.8?'background:var(--red)':''}"></i></div></td><td><span class="pill ${d.status==='PASS'?'pass':'fail'}">${d.status}</span></td></tr>`}).join('');
document.getElementById('recrows').innerHTML=D.recon.map(r=>`<tr><td>${r.check}</td><td class="num">${f(r.silver)}</td><td class="num">${f(r.gold)}</td><td><span class="pill ${r.match?'pass':'fail'}">${r.match?'MATCH':'GAP'}</span></td></tr>`).join('');
// scrollspy
const links=[...document.querySelectorAll('nav a')],secs=links.map(a=>document.querySelector(a.getAttribute('href')));
addEventListener('scroll',()=>{let i=0;secs.forEach((s,k)=>{if(s.getBoundingClientRect().top<120)i=k});links.forEach((a,k)=>a.classList.toggle('on',k===i))},{passive:true});
</script>
</body>
</html>
"""

catalog_meta = [
    ("gold_daily_activity", "One row per UTC day", "Daily chart, stability metrics"),
    ("gold_model_mix_daily", "Day and model", "Model mix chart"),
    ("gold_turns_distribution", "Turn bucket", "Conversation length chart"),
    ("gold_language_mix", "Language", "Language chart"),
    ("gold_country_mix", "Country", "Country chart"),
    ("gold_hourly_pattern", "Weekday and hour", "Heatmap"),
    ("gold_return_rates", "First-seen cohort day", "Return rate chart"),
    ("gold_user_concentration", "Conversations-per-IP bucket", "Concentration chart"),
]
gold_list = "".join(f"<li><span>{k}</span><b>{gold_rows[k]}</b></li>" for k, _, _ in catalog_meta)
catalog = "".join(f"<tr><td><code>{k}</code></td><td>{g}</td><td class='num'>{gold_rows[k]:,}</td><td>{f}</td></tr>" for k, g, f in catalog_meta)

span0 = dt.date.fromisoformat(B["span"][0][:10]).strftime("%-d %b %Y")
span1 = dt.date.fromisoformat(B["span"][1][:10]).strftime("%-d %b %Y")

rep = {
    "__PLOTLY__": PLOTLY, "__DATA__": json.dumps(DATA, default=str),
    "__SPAN__": f"{span0} to {span1}", "__NCONV__": n(R["valid_conv"]), "__NIPS__": n(R["dim_user"]),
    "__NLANG__": str(D["languages"]), "__NCTRY__": str(D["countries"]),
    "__DQP__": str(dq_pass), "__DQN__": str(len(dq)), "__RECOK__": str(rec_ok), "__RECN__": str(len(rec)),
    "__BCONV__": n(R["bronze_conv"]), "__BUTT__": n(R["bronze_utt"]), "__BCOLS__": str(len(B["null_rates"])),
    "__SCONV__": n(R["silver_conv"]), "__SUTT__": n(R["silver_utt"]),
    "__GOLDLIST__": gold_list, "__CATALOG__": catalog,
    "__EXCL__": n(excluded), "__EXCLP__": pct(excluded / R["bronze_conv"], 2), "__TTOT__": f"{t_total:.1f}",
    "__PMED__": n(B["prompt"]["med"]), "__PMEAN__": n(B["prompt"]["mean"]), "__PP95__": n(B["prompt"]["p95"]),
    "__IPDROP__": pct(ip_drop), "__CONVDROP__": pct(conv_drop), "__RETALL__": pct(ret_overall), "__RETRAW__": repr(ret_overall),
}
for k, v in I.items():
    rep[f"__I_{k.upper()}__"] = v
out = HTML
for k, v in rep.items():
    out = out.replace(k, v)
assert "__" not in out.replace("__proto__", "").split("<script>")[0], "unreplaced placeholder"
dest = ROOT / "dashboard" / "chathouse_medallion_dashboard.html"
dest.parent.mkdir(exist_ok=True)
dest.write_text(out, encoding="utf-8")
print(dest, round(dest.stat().st_size / 1e6, 2), "MB")
print(json.dumps(I, indent=1))
