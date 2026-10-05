"""Independent re-check of the D12 time facts (verifier's own code; no imports
from the original scripts).

Method differences from s1_timeline.py:
- files are listed with os.scandir recursion, not os.walk;
- log windows are merged by marking every covered SECOND in a set (occupancy
  count), not by sorting and merging intervals;
- pipeline concurrency is counted per second;
- audit timing and prompt counts are read straight from ~/.claude/history.jsonl
  (read only; no prompt text is written out).
Output: verification/out/v1_timing.json
"""
import collections
import datetime
import hashlib
import json
import os

MT = "<REPO_ROOT>/projects/maxtoki"
REPO = "<REPO_ROOT>"
OUT = os.path.join(MT, "checks/d12_deployment_facts/verification/out")
CUT = datetime.datetime(2026, 5, 8, 6, 0).timestamp()
PIPES = ["spectral-geometry-217M", "attention-grn-217M", "topology-141-217M", "sae-atlas-217M",
         "circuit-tracing-217M", "exhaustive-mapping-217M", "manifold-discovery-217M", "longevity-mechinterp-217M"]


def files(top):
    stack = [top]
    while stack:
        d = stack.pop()
        with os.scandir(d) as it:
            for e in it:
                if e.name.startswith("._"):
                    continue
                if e.is_dir(follow_symlinks=False):
                    if e.name in ("node_modules", ".git", "__pycache__"):
                        continue
                    stack.append(e.path)
                elif e.is_file(follow_symlinks=False):
                    yield e.path, e.stat(follow_symlinks=False)


def eff(st):
    # a file copied in with an old preserved mtime has birth > mtime
    return st.st_birthtime if st.st_birthtime > st.st_mtime + 60 else st.st_mtime


fmt = lambda t: datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")

# ---------------------------------------------------------------- whole deployment
recs = []
for top in ("runs", "summaries", "audits", "setup"):
    for p, st in files(os.path.join(MT, top)):
        recs.append((eff(st), os.path.relpath(p, MT), st))
dep = [r for r in recs if r[0] < CUT]
first, last = min(dep), max(dep)
days = sorted({fmt(r[0])[:10] for r in dep})
hours = {fmt(r[0])[:13] for r in dep}
whole = dict(n_files=len(dep), n_excluded_after_cutoff=len(recs) - len(dep),
             n_excluded_born_before_cutoff=sum(1 for r in recs if r[0] >= CUT and r[2].st_birthtime < CUT),
             first=(fmt(first[0]), first[1]), last=(fmt(last[0]), last[1]),
             span_days=round((last[0] - first[0]) / 86400, 3), n_active_days=len(days), active_days=days,
             n_clock_hours=len(hours))

# ---------------------------------------------------------------- log windows, minute bins
per_min = {}
logs = []
for p in PIPES:
    mins = set()
    for path, st in files(os.path.join(MT, "runs", p)):
        if not path.endswith(".log") or st.st_mtime >= CUT:
            continue
        a, b = st.st_birthtime, st.st_mtime
        logs.append(dict(pipeline=p, log=os.path.relpath(path, MT), start=fmt(a), end=fmt(b), h=round((b - a) / 3600, 3)))
        # exact to the second: mark every whole second inside [a, b)
        mins.update(range(int(round(a)), int(round(b))))
    per_min[p] = mins
all_min = set().union(*per_min.values())
conc = collections.Counter()
for m in all_min:
    conc[sum(m in per_min[p] for p in PIPES)] += 1
first_4 = min((m for m in all_min if sum(m in per_min[p] for p in PIPES) == 4), default=None)
logwin = dict(
    n_logs=len(logs),
    per_pipeline_h={p: round(len(per_min[p]) / 3600, 2) for p in PIPES},
    sum_over_pipelines_h=round(sum(len(v) for v in per_min.values()) / 3600, 2),
    overlap_aware_h=round(len(all_min) / 3600, 2),
    hours_by_n_running={k: round(v / 3600, 2) for k, v in sorted(conc.items())},
    hours_with_2_or_more=round(sum(v for k, v in conc.items() if k >= 2) / 3600, 2),
    first_second_with_4=fmt(first_4) if first_4 else None,
    longest_single_log_windows=sorted(logs, key=lambda x: -x["h"])[:8],
    note="second-level occupancy: each log window [birth, mtime) is marked second by second; union = number of marked seconds",
)

# work blocks: all run-folder file writes plus log windows of a pipeline,
# cut wherever nothing happens for more than 3 hours (time-ordered scan)
blocks = {}
for p in PIPES:
    ev = [(r[0], r[0]) for r in dep if r[1].startswith(f"runs/{p}/")]
    ev += [(datetime.datetime.strptime(l["start"], "%Y-%m-%d %H:%M:%S").timestamp(),
            datetime.datetime.strptime(l["end"], "%Y-%m-%d %H:%M:%S").timestamp()) for l in logs if l["pipeline"] == p]
    ev.sort()
    tot, s, e = 0.0, ev[0][0], ev[0][1]
    for a, b in ev[1:]:
        if a - e > 3 * 3600:
            tot += e - s
            s, e = a, b
        else:
            e = max(e, b)
    tot += e - s
    blocks[p] = round(tot / 3600, 2)
logwin["work_blocks_gap3h_h"] = blocks

# ---------------------------------------------------------------- audit timing
st = lambda rel: eff(os.stat(os.path.join(MT, rel)))
spec = eff(os.stat(os.path.join(REPO, "pipelines/audit-recurring-review-issues.md")))
H = os.path.expanduser("~/.claude/history.jsonl")
rows = [json.loads(l) for l in open(H)]
aud = sorted(r["timestamp"] / 1000 for r in rows if r.get("sessionId", "").startswith("2a9284b9")
             and fmt(r["timestamp"] / 1000).startswith("2026-05-07"))
rep, v1, v2 = st("audits/audit-20260507.md"), st("audits/audit-20260507-completion.md"), st("audits/audit-20260507-completion-v2.md")
# prompts by position in the session that day (checked by time in the output)
audit = dict(
    session_prompt_times_may7=[fmt(t) for t in aud],
    n_prompts_session_may7=len(aud),
    spec_file=fmt(spec), report_file=fmt(rep), completion_v1=fmt(v1), completion_v2=fmt(v2),
)
t_spec_req, t_apply, t_fix, t_done, t_fix2 = aud[5], aud[6], aud[7], aud[8], aud[9]
audit.update(
    used_prompts=dict(spec_request=fmt(t_spec_req), apply=fmt(t_apply), fix=fmt(t_fix), done_q=fmt(t_done), fix2=fmt(t_fix2)),
    spec_min=round((spec - t_spec_req) / 60, 1), sweep_min=round((rep - t_apply) / 60, 1),
    round1_min=round((v1 - t_fix) / 60, 1), round2_min=round((v2 - t_fix2) / 60, 1),
    total_from_apply_h=round((v2 - t_apply) / 3600, 3), total_from_spec_request_h=round((v2 - t_spec_req) / 3600, 3),
)

# ---------------------------------------------------------------- prompt counts
S = {"17b19da0", "965ec1c8", "8fb71b84", "3af9282c", "8be46b0f", "b9b1ea22", "2a9284b9", "58d03bad", "5137b56d", "862b8d39"}
lo, hi = datetime.datetime(2026, 4, 15).timestamp(), datetime.datetime(2026, 5, 9).timestamp()
win = [r for r in rows if lo <= r["timestamp"] / 1000 < hi and r.get("project") == REPO]
mx = [r for r in win if r["sessionId"][:8] in S]
per_sess = collections.Counter(r["sessionId"][:8] for r in mx)
per_day = collections.Counter(fmt(r["timestamp"] / 1000)[:10] for r in mx)
prompts = dict(history_sha256=hashlib.sha256(open(H, "rb").read()).hexdigest(), n_rows=len(rows),
               n_same_folder_in_window=len(win), n_maxtoki_sessions=len(mx), per_session=dict(per_sess),
               per_day_all_maxtoki_sessions=dict(sorted(per_day.items())),
               n_session_17b19da0=per_sess["17b19da0"], n_session_862b8d39=per_sess["862b8d39"])

res = dict(whole=whole, log_windows=logwin, audit=audit, prompts=prompts)
os.makedirs(OUT, exist_ok=True)
json.dump(res, open(os.path.join(OUT, "v1_timing.json"), "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in res.items() if k != "log_windows"}, indent=1, default=str)[:4000])
print(json.dumps({k: v for k, v in logwin.items() if k != "longest_single_log_windows"}, indent=1))
for l in logwin["longest_single_log_windows"]:
    print(l)
