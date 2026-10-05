"""D12 step 1: file and log timeline for the April-May 2026 MaxToki deployment.

Reads only file metadata (size, birth time, modification time) under
runs/, summaries/, audits/, setup/ and the content of the *.log files and a
few run_config.json / STATUS.md files for logged durations.

Three ways to measure compute time (each pipeline, and all pipelines together
with overlaps counted once):
  A. log windows      = union of [log file creation, log file last write]
  B. activity sessions = union of log windows and single file writes, merged
                         when two events are closer than G minutes (G = 15/30/60)
  C. logged durations  = durations the scripts themselves printed or saved
                         (curated list below; each value is read from a file)
Outputs: out/file_index.csv, out/timeline.json
"""
import collections
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import (MT, REPO, OUT, PIPELINES, CUTOFF, iso, day, sha256_file,
                    eff_time, walk, union, total_hours, dump)

TOPS = ["runs", "summaries", "audits", "setup"]

SUMMARY_PREFIX = {
    "spectral-geometry": "spectral-geometry-217M",
    "attention-grn": "attention-grn-217M",
    "topology-141": "topology-141-217M",
    "sae-atlas": "sae-atlas-217M",
    "circuit-tracing": "circuit-tracing-217M",
    "exhaustive-mapping": "exhaustive-mapping-217M",
    "manifold-discovery": "manifold-discovery-217M",
    "longevity": "longevity-mechinterp-217M",
}


def owner(rel):
    parts = rel.split("/")
    if parts[0] == "runs":
        return parts[1]
    if parts[0] == "summaries":
        for k, v in SUMMARY_PREFIX.items():
            if parts[1].startswith(k):
                return v
        return "summaries-other"
    return parts[0]  # audits, setup


def is_audit_file(rel):
    b = os.path.basename(rel)
    return rel.startswith("audits/") or b.startswith("audit_") or "/audit_" in rel or \
        any(seg.startswith(("phase8t_chance_baseline", "edge_density_chance_baseline",
                            "synthetic_positive_control", "groupkfold_", "phase8t_positive_tf",
                            "seed_stability", "iter_04_replication", "cell_level_benchmark_lite",
                            "bootstrap_pearson", "external_bootstrap"))
            for seg in rel.split("/"))


# ---------------------------------------------------------------- file index
recs = []
for top in TOPS:
    for p in walk(os.path.join(MT, top)):
        try:
            st = os.stat(p)
        except OSError:
            continue
        e, copied = eff_time(st)
        rel = os.path.relpath(p, MT)
        recs.append(dict(rel=rel, owner=owner(rel), size=st.st_size, mtime=st.st_mtime,
                         birth=getattr(st, "st_birthtime", st.st_mtime), eff=e,
                         copied_old_mtime=copied, post_cutoff=e >= CUTOFF,
                         audit=is_audit_file(rel)))
# the audit spec lives outside projects/maxtoki
spec = os.path.join(REPO, "pipelines", "audit-recurring-review-issues.md")
st = os.stat(spec)
e, copied = eff_time(st)
recs.append(dict(rel="../../pipelines/audit-recurring-review-issues.md", owner="audit-spec",
                 size=st.st_size, mtime=st.st_mtime, birth=st.st_birthtime, eff=e,
                 copied_old_mtime=copied, post_cutoff=e >= CUTOFF, audit=True))

with open(os.path.join(OUT, "file_index.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["rel_path", "owner", "size_bytes", "mtime", "birth", "effective_time",
                "copied_old_mtime", "post_cutoff", "audit_related"])
    for r in sorted(recs, key=lambda r: r["rel"]):
        w.writerow([r["rel"], r["owner"], r["size"], iso(r["mtime"]), iso(r["birth"]),
                    iso(r["eff"]), r["copied_old_mtime"], r["post_cutoff"], r["audit"]])

dep = [r for r in recs if not r["post_cutoff"] and r["owner"] != "audit-spec"]
post = [r for r in recs if r["post_cutoff"]]

# ---------------------------------------------------------------- whole deployment
first = min(dep, key=lambda r: r["eff"])
last = max(dep, key=lambda r: r["eff"])
first_runs = min([r for r in dep if r["rel"].startswith("runs/")], key=lambda r: r["eff"])
days_all = sorted({day(r["eff"]) for r in dep})
hours_all = sorted({iso(r["eff"])[:13] for r in dep})
deployment = dict(
    n_files_in_window=len(dep),
    n_files_excluded_post_cutoff=len(post),
    post_cutoff_examples=sorted({r["rel"] for r in post})[:10],
    n_files_copied_with_old_mtime=sum(r["copied_old_mtime"] for r in dep),
    first_artefact=dict(path=first["rel"], time=iso(first["eff"])),
    first_run_folder_artefact=dict(path=first_runs["rel"], time=iso(first_runs["eff"])),
    last_artefact=dict(path=last["rel"], time=iso(last["eff"])),
    span_days=round((last["eff"] - first["eff"]) / 86400, 2),
    active_days=days_all,
    n_active_days=len(days_all),
    n_distinct_clock_hours_with_a_write=len(hours_all),
    files_per_day=dict(collections.Counter(day(r["eff"]) for r in dep)),
)

# ---------------------------------------------------------------- per pipeline file windows
per_pipe = {}
for p in PIPELINES:
    rs = [r for r in dep if r["owner"] == p]
    byday = collections.defaultdict(list)
    for r in rs:
        byday[day(r["eff"])].append(r["eff"])
    per_pipe[p] = dict(
        n_files=len(rs),
        first=iso(min(r["eff"] for r in rs)),
        last=iso(max(r["eff"] for r in rs)),
        days={d: dict(first=iso(min(v))[11:16], last=iso(max(v))[11:16], n_files=len(v))
              for d, v in sorted(byday.items())},
    )

# ---------------------------------------------------------------- logs
tq = re.compile(r"(\d+)/(\d+) \[(\d+(?::\d+){1,2})<")


def tosec(s):
    p = [int(x) for x in s.split(":")]
    return p[0] * 60 + p[1] if len(p) == 2 else p[0] * 3600 + p[1] * 60 + p[2]


logs = []
for p in PIPELINES:
    for path in walk(os.path.join(MT, "runs", p)):
        if not path.endswith(".log"):
            continue
        st = os.stat(path)
        if st.st_mtime >= CUTOFF:
            continue
        txt = open(path, errors="ignore").read()
        bars, cur = [], None
        for seg in re.split(r"[\r\n]", txt):
            for m in tq.finditer(seg):
                n, tot, el = int(m.group(1)), int(m.group(2)), tosec(m.group(3))
                if cur is None or tot != cur[0] or n < cur[1]:
                    if cur:
                        bars.append(cur)
                    cur = [tot, n, el]
                else:
                    cur[1], cur[2] = n, el
        if cur:
            bars.append(cur)
        dev = sorted(set(m.group(1).lower() for m in re.finditer(
            r"(?:device[\"'=: ]+|on |Device: )(mps|cuda|cpu)\b", txt, re.I)))
        logs.append(dict(pipeline=p, log=os.path.relpath(path, MT), birth=st.st_birthtime,
                         mtime=st.st_mtime, size=st.st_size, sha256=sha256_file(path),
                         window_h=(st.st_mtime - st.st_birthtime) / 3600,
                         tqdm_bars_ge_60s_min=round(sum(b[2] for b in bars if b[2] >= 60) / 60, 2),
                         device_mentions=dev))

log_rows = [dict(l, birth=iso(l["birth"]), mtime=iso(l["mtime"]), window_h=round(l["window_h"], 3))
            for l in logs]

A = {}
all_iv = []
for p in PIPELINES:
    iv = [(l["birth"], l["mtime"]) for l in logs if l["pipeline"] == p]
    u = union(iv)
    all_iv += iv
    A[p] = dict(n_logs=len(iv), hours=round(total_hours(u), 2),
                windows=[f"{iso(a)[5:16]} -> {iso(b)[11:16]}" for a, b in u if b - a >= 300])
u_all = union(all_iv)
# concurrency of pipelines (sweep line over per-pipeline unions of log windows)
ev = []
for p in PIPELINES:
    for a, b in union([(l["birth"], l["mtime"]) for l in logs if l["pipeline"] == p]):
        ev += [(a, 1, p), (b, -1, p)]
ev.sort(key=lambda x: (x[0], x[1]))
conc_h = collections.Counter()
cur, last_t, active, max_c, max_when = 0, None, set(), 0, None
for t_, d_, p_ in ev:
    if last_t is not None and cur > 0:
        conc_h[cur] += (t_ - last_t) / 3600
    if d_ == 1:
        active.add(p_)
    else:
        active.discard(p_)
    cur = len(active)
    if cur > max_c:
        max_c, max_when = cur, (iso(t_), sorted(active))
    last_t = t_
concurrency = dict(hours_by_n_pipelines_running={k: round(v, 2) for k, v in sorted(conc_h.items())},
                   hours_with_2_or_more=round(sum(v for k, v in conc_h.items() if k >= 2), 2),
                   max_pipelines_at_once=max_c, first_time_max_reached=max_when)
A_total = dict(sum_of_pipelines_h=round(sum(v["hours"] for v in A.values()), 2),
               overlap_aware_h=round(total_hours(u_all), 2))
long_windows = [dict(log=l["log"], hours=round(l["window_h"], 2)) for l in logs if l["window_h"] > 6]

# ---------------------------------------------------------------- B: activity sessions
B = {}
for G in (15, 30, 60):
    per, allv = {}, []
    for p in PIPELINES:
        pts = [(r["eff"], r["eff"]) for r in dep if r["owner"] == p and r["rel"].startswith("runs/")]
        pts += [(l["birth"], l["mtime"]) for l in logs if l["pipeline"] == p]
        u = union(pts, gap=G * 60)
        per[p] = dict(hours=round(total_hours(u), 2), n_sessions=len(u))
        allv += u
    B[f"gap_{G}min"] = dict(per_pipeline=per,
                            sum_of_pipelines_h=round(sum(v["hours"] for v in per.values()), 2),
                            overlap_aware_h=round(total_hours(union(allv)), 2))

# work sessions: same events merged when closer than 3 hours (day-level blocks)
SESS3H = {}
for p in PIPELINES:
    pts = [(r["eff"], r["eff"]) for r in dep if r["owner"] == p and r["rel"].startswith("runs/")]
    pts += [(l["birth"], l["mtime"]) for l in logs if l["pipeline"] == p]
    SESS3H[p] = [dict(start=iso(a), end=iso(b), hours=round((b - a) / 3600, 2)) for a, b in union(pts, gap=3 * 3600)]

# ---------------------------------------------------------------- C: logged durations (curated)
def grab(rel, pattern, how="sum", scale=1.0, flags=0):
    txt = open(os.path.join(MT, rel), errors="ignore").read()
    txt = txt.replace("\r", "\n")
    vals = [float(m.group(1).replace(",", "")) for m in re.finditer(pattern, txt, flags)]
    if not vals:
        raise SystemExit(f"pattern not found: {rel} :: {pattern}")
    v = sum(vals) if how == "sum" else (vals[-1] if how == "last" else max(vals))
    return dict(file=rel, pattern=pattern, how=how, n_matches=len(vals), seconds=round(v * scale, 1))


def jfield(rel, key):
    d = json.load(open(os.path.join(MT, rel)))
    return dict(file=rel, pattern=f"json:{key}", how="field", n_matches=1, seconds=round(float(d[key]), 1))


R = "runs/"
C = {
    "spectral-geometry-217M": {
        "phase0 extraction (total_phase_seconds)": jfield(R + "spectral-geometry-217M/outputs/phase0/run_config.json", "total_phase_seconds"),
        "phase8 stability, 2 re-extractions (tqdm bars)": dict(file=R + "spectral-geometry-217M/outputs/phase8.log", pattern="tqdm bars >= 60 s", how="sum", n_matches=None,
                                                               seconds=round(next(l for l in logs if l["log"].endswith("spectral-geometry-217M/outputs/phase8.log"))["tqdm_bars_ge_60s_min"] * 60, 1)),
        "autoloop executor+brainstormer turns (driver dt)": grab(R + "spectral-geometry-217M/autoloop/runtime/driver.log", r"dt=(\d+)s"),
    },
    "attention-grn-217M": {
        "phase0 K562 217M": jfield(R + "attention-grn-217M/outputs/phase0/run_config.json", "total_phase_seconds"),
        "phase0 RPE1": jfield(R + "attention-grn-217M/outputs/phase0_rpe1/run_config.json", "total_phase_seconds"),
        "phase0 Adamson": jfield(R + "attention-grn-217M/outputs/phase0_adamson/run_config.json", "total_phase_seconds"),
        "phase0 K562 1B (200 cells)": jfield(R + "attention-grn-217M/outputs/phase0_k562_1b/run_config.json", "total_phase_seconds"),
        "phase0b value-weighted (total_fwd_seconds)": jfield(R + "attention-grn-217M/outputs/phase0b/run_config.json", "total_fwd_seconds"),
    },
    "topology-141-217M": {
        "phase0 lung": jfield(R + "topology-141-217M/outputs/phase0/lung/run_config.json", "wall_seconds"),
        "phase0 external lung": jfield(R + "topology-141-217M/outputs/phase0/external_lung/run_config.json", "wall_seconds"),
        "audit re-extraction lung seed 43": jfield(R + "topology-141-217M/outputs/seed_stability_phase0/seed43/lung/run_config.json", "wall_seconds"),
    },
    "sae-atlas-217M": {
        "12-layer extraction+SAE training+annotation (sum of 'Lx DONE in')": grab(R + "sae-atlas-217M/outputs/full_12layer.log", r"L\d+ DONE in (\d+)s"),
    },
    "circuit-tracing-217M": {
        "trace L0/L3/L6/L9 (sum of 'complete: ... in Ns')": grab(R + "circuit-tracing-217M/outputs/circuit_trace.log", r"complete: \d+ edges in (\d+)s"),
    },
    "exhaustive-mapping-217M": {
        "experiment 1 (last 'min elapsed')": grab(R + "exhaustive-mapping-217M/outputs/experiment1.log", r"([\d.]+)min elapsed", how="last", scale=60),
        "experiment 2 rerun (wall-clock)": grab(R + "exhaustive-mapping-217M/outputs/experiment2_rerun.log", r"wall-clock: ([\d.]+) min", how="last", scale=60),
    },
    "manifold-discovery-217M": {
        "STATUS.md iteration-log 'done in Ns' entries (one per step)": grab(R + "manifold-discovery-217M/STATUS.md", r"\n- 2026-[^\n]*?(?:done|completed) in ([\d,]+)s\b"),
    },
    "longevity-mechinterp-217M": {
        "stage 1 extraction (last elapsed=)": grab(R + "longevity-mechinterp-217M/outputs/stage1_run.log", r"elapsed=(\d+)s", how="last"),
    },
}
C_tot = {p: round(sum(v["seconds"] for v in d.values()) / 3600, 2) for p, d in C.items()}

# ---------------------------------------------------------------- audit day
aud = [r for r in recs if r["audit"] and not r["post_cutoff"]]
aud_may7 = [r for r in aud if day(r["eff"]) == "2026-05-07"]
audit = dict(
    n_audit_files=len(aud),
    n_on_2026_05_07=len(aud_may7),
    days=sorted({day(r["eff"]) for r in aud}),
    first=dict(path=min(aud, key=lambda r: r["eff"])["rel"], time=iso(min(r["eff"] for r in aud))),
    last=dict(path=max(aud, key=lambda r: r["eff"])["rel"], time=iso(max(r["eff"] for r in aud))),
    key_files={r["rel"]: iso(r["eff"]) for r in aud if r["rel"].startswith(("audits/", "../../pipelines"))},
    note="audit-related = files in audits/, audit_* scripts, their output folders, and the audit spec",
)
if any(day(r["eff"]) != "2026-05-07" for r in aud):
    audit["files_not_on_may7"] = sorted({r["rel"] for r in aud if day(r["eff"]) != "2026-05-07"})

res = dict(deployment=deployment, per_pipeline_files=per_pipe,
           A_log_windows=dict(per_pipeline=A, total=A_total, logs_longer_than_6h=long_windows, concurrency=concurrency),
           B_activity_sessions=B, work_blocks_gap3h=SESS3H,
           C_logged_durations=dict(items=C, per_pipeline_hours=C_tot,
                                   note="lower bound: covers only steps whose scripts printed or saved a duration"),
           audit_files=audit, logs=log_rows)
print(dump("timeline.json", res))
print(json.dumps(dict(deployment={k: deployment[k] for k in ("first_artefact", "last_artefact", "span_days", "n_active_days", "n_distinct_clock_hours_with_a_write", "n_files_excluded_post_cutoff")},
                      A=A_total, B={k: (v["sum_of_pipelines_h"], v["overlap_aware_h"]) for k, v in B.items()},
                      C=C_tot, audit={k: audit[k] for k in ("first", "last", "days", "n_audit_files")}), indent=1))
