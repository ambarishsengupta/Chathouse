"""Compute pipeline, quality and reconciliation metrics for the Chathouse medallion pipeline.

Reads the bronze Parquet files, rebuilds the silver layer in memory, checks the gold tables
in output/gold against silver, and writes output/metrics/bundle.json for the dashboard and deck.
"""
import glob, json, os, time
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]
DATA_GLOB = os.environ.get("CHATHOUSE_DATA_GLOB", str(ROOT / "*.parquet"))
GOLD = ROOT / "output" / "gold"
OUT = ROOT / "output" / "metrics"
HEAVY_IP_MIN_CONVS = 100

files = sorted(glob.glob(DATA_GLOB))
assert files, "No Parquet files match the data path"
con = duckdb.connect()
sql = lambda q: con.sql(q).fetchall()
timings = {}

def timed(name, q):
    t = time.time(); con.execute(q); timings[name] = round(time.time() - t, 2)

# Bronze and silver layers
timed("Bronze scan", f"create table bronze as select * from read_parquet({files!r})")
timed("Silver conversations", """create table sc as
 select conversation[1].turn_identifier as conversation_id, conversation_hash, model, cast(timestamp as timestamp) as ts_utc,
        cast(timestamp as date) as event_date, turn, language, coalesce(country,'Unknown') as country,
        coalesce(state,'Unknown') as state, hashed_ip, header."user-agent" as user_agent,
        len(conversation) as utterance_count,
        coalesce(trim(conversation[1].content),'') = '' as empty_first_input
 from bronze""")
timed("Silver utterances", "create table su as select conversation[1].turn_identifier as cid, unnest(conversation) as u from bronze")
timed("User proxy dimension", f"""create table dim as select hashed_ip, count(*) as c from sc where not empty_first_input group by 1""")
con.execute("create table su2 as select cid as conversation_id, u.turn_identifier, u.role, len(u.content) as char_len from su")

rows = dict(
    bronze_conv=sql("select count(*) from bronze")[0][0],
    bronze_utt=int(sql("select sum(len(conversation)) from bronze")[0][0]),
    silver_conv=sql("select count(*) from sc")[0][0],
    silver_utt=sql("select count(*) from su")[0][0],
    valid_conv=sql("select count(*) from sc where not empty_first_input")[0][0],
    dim_user=sql("select count(*) from dim")[0][0],
)
span = [str(x) for x in sql("select min(ts_utc), max(ts_utc) from sc")[0]]
distinct = dict(zip(["models", "languages", "countries", "days"], sql(
    "select count(distinct model), count(distinct language), count(distinct country), count(distinct event_date) from sc")[0]))
cols = [r[0] for r in sql("describe bronze")]
null_rates = {c: sql(f'select avg(("{c}" is null)::int) from bronze')[0][0] for c in cols}

# Data quality checks
checks = [
 ("conversation_id is unique and not null", 0.0, "select count(*)-count(distinct conversation_id)+count(*) filter (where conversation_id is null) from sc", "sc", "Uniqueness"),
 ("utterance key is unique", 0.0, "select count(*)-count(distinct (conversation_id,turn_identifier,role)) from su2", "su2", "Uniqueness"),
 ("utterance count equals twice the turn count", 0.0, "select count(*) filter (where utterance_count != 2*turn) from sc", "sc", "Consistency"),
 ("role is user or assistant", 0.0, "select count(*) filter (where role not in ('user','assistant')) from su2", "su2", "Validity"),
 ("timestamp is present", 0.0, "select count(*) filter (where ts_utc is null) from sc", "sc", "Completeness"),
 ("moderation category spellings agree", 0.0, """select count(*) filter (where m.categories."self-harm" is distinct from m.categories.self_harm
      or m.categories."hate/threatening" is distinct from m.categories.hate_threatening
      or m.categories."violence/graphic" is distinct from m.categories.violence_graphic)
      from (select unnest(openai_moderation) as m from bronze)""", "bronze", "Consistency"),
 ("first user input is not empty", 0.02, "select count(*) filter (where empty_first_input) from sc", "sc", "Validity"),
 ("conversation_hash is unique", 0.01, "select count(*)-count(distinct conversation_hash) from sc", "sc", "Uniqueness"),
 ("country is known", 0.01, "select count(*) filter (where country='Unknown') from sc", "sc", "Completeness"),
 ("state is known", 0.25, "select count(*) filter (where state='Unknown') from sc", "sc", "Completeness"),
 ("user agent is present", 0.05, "select count(*) filter (where user_agent is null) from sc", "sc", "Completeness"),
]
dq = []
for name, limit, q, table, dim in checks:
    failed = int(con.sql(q).fetchone()[0]); total = int(con.sql(f"select count(*) from {table}").fetchone()[0])
    share = failed / total
    dq.append(dict(check=name, failed=failed, total=total, share=share, limit=limit,
                   status="PASS" if share <= limit else "FAIL", dimension=dim))

# Gold reconciliation against silver
g = lambda t: f"read_parquet('{GOLD}/{t}.parquet')"
valid = "select count(*) from sc where not empty_first_input"
recon = []
def rec(name, gold_q, silver_q):
    a, b = int(sql(gold_q)[0][0]), int(sql(silver_q)[0][0])
    recon.append(dict(check=name, gold=a, silver=b, match=a == b))
for t in ["gold_daily_activity", "gold_model_mix_daily", "gold_language_mix", "gold_country_mix",
          "gold_hourly_pattern", "gold_turns_distribution", "gold_user_concentration"]:
    rec(f"{t}: conversations", f"select sum(conversations) from {g(t)}", valid)
rec("gold_user_concentration: hashed IPs", f"select sum(hashed_ips) from {g('gold_user_concentration')}", "select count(*) from dim")
rec("bronze to silver: conversation rows", "select count(*) from sc", "select count(*) from bronze")
rec("bronze to silver: utterance rows", "select count(*) from su", "select sum(len(conversation)) from bronze")

out = dict(rows=rows, span=span, distinct=distinct, null_rates=null_rates, dq=dq, recon=recon, timings=timings)
for f in sorted(glob.glob(str(GOLD / "gold_*.parquet"))):
    df = con.sql(f"select * from read_parquet('{f}')").df()
    for c in df.columns:
        if str(df[c].dtype).startswith("datetime") or str(df[c].dtype) == "object":
            df[c] = df[c].astype(str)
    out[os.path.basename(f)[:-8]] = json.loads(df.to_json(orient="records"))
out["heavy"] = con.sql(f"""select count(*) as ips, count(*) filter (where c >= {HEAVY_IP_MIN_CONVS}) as heavy,
    sum(c)::bigint as convs, (sum(c) filter (where c >= {HEAVY_IP_MIN_CONVS}))::bigint as heavy_convs from dim""").df().to_dict("records")[0]
out["prompt"] = con.sql("select median(char_len) as med, avg(char_len) as mean, quantile_cont(char_len, 0.95) as p95 from su2 where role='user'").df().to_dict("records")[0]
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "bundle.json").write_text(json.dumps(out, default=str))
print("PASS", sum(d["status"] == "PASS" for d in dq), "of", len(dq), "| reconciled", sum(r["match"] for r in recon), "of", len(recon))
print(out["heavy"], out["prompt"], timings)
