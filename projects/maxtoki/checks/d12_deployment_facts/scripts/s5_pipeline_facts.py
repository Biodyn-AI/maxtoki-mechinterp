"""D11 + D12 step 5: per-pipeline facts read from run files.

- phases in each spec (from the spec's '### Phase/Stage/Experiment N' headings)
- phases with run evidence (output/log names, '[Phase N]' markers in logs), dated
- device evidence (logs, run_config.json, script defaults)
- per-cell token budget (max_len in scripts; recorded sequence lengths)
- input format (single cell vs multi-cell trajectory)
- loop mechanics that existed in code or files
- topology-141 scope: hypotheses touched, strict max-null layers, dual-axis CV scope
- H115 / H118 search
Outputs: out/pipeline_facts.json
"""
import collections
import csv
import datetime
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import MT, REPO, OUT, PIPELINES, CUTOFF, iso, day, eff_time, walk, dump, sha256_file

SPECS = {
    "spectral-geometry-217M": "pipelines/residual-stream-spectral-geometry.md",
    "attention-grn-217M": "pipelines/attention-grn-extraction-and-evaluation.md",
    "topology-141-217M": "pipelines/topology-geometry-141-hypotheses.md",
    "sae-atlas-217M": "pipelines/sparse-autoencoders/01-sae-atlas.md",
    "circuit-tracing-217M": "pipelines/sparse-autoencoders/02-causal-circuit-tracing.md",
    "exhaustive-mapping-217M": "pipelines/sparse-autoencoders/03-exhaustive-mapping-and-steering.md",
    "manifold-discovery-217M": "pipelines/manifold-discovery-extraction-compactification.md",
    "longevity-mechinterp-217M": "pipelines/longevity-mechinterp-donor-aware.md",
}

# Session windows used to label when a phase first ran (local time).
# base = first block of work on the pipeline; same_session = later days of the
# same agent session before the human moved on; later = separate later sessions.
T = lambda s: datetime.datetime.strptime(s, "%Y-%m-%d %H:%M").timestamp()
WINDOWS = {
    "spectral-geometry-217M": dict(base=(T("2026-04-16 18:57"), T("2026-04-17 10:42")), same_session=None),
    "attention-grn-217M": dict(base=(T("2026-04-15 23:46"), T("2026-04-16 02:01")), same_session=(T("2026-04-16 02:01"), T("2026-04-16 18:47"))),
    "topology-141-217M": dict(base=(T("2026-05-03 19:15"), T("2026-05-03 20:42")), same_session=None),
    "sae-atlas-217M": dict(base=(T("2026-04-17 15:48"), T("2026-04-17 23:21")), same_session=(T("2026-04-17 23:21"), T("2026-04-19 22:22"))),
    "circuit-tracing-217M": dict(base=(T("2026-04-19 22:20"), T("2026-04-20 02:28")), same_session=(T("2026-04-20 02:28"), T("2026-04-20 11:14"))),
    "exhaustive-mapping-217M": dict(base=(T("2026-04-20 14:15"), T("2026-04-20 20:40")), same_session=None),
    "manifold-discovery-217M": dict(base=(T("2026-05-03 19:38"), T("2026-05-04 08:32")), same_session=None),
    "longevity-mechinterp-217M": dict(base=(T("2026-05-05 16:10"), T("2026-05-05 17:51")), same_session=None),
}
AUDIT_WIN = (T("2026-05-07 17:50"), T("2026-05-07 21:11"))

MULTI = {  # output/log names that cover several phases and whose logs carry no [Phase N] markers
    "phase78": ["7", "8"], "phase1bc": ["1b", "1c"],
}
MARKER_LOGS = ("phases_345", "phases_67", "phases2_to_8", "phases12_13", "remaining")  # use markers, not names


def label_for(p, t, audit):
    w = WINDOWS[p]
    if audit or AUDIT_WIN[0] <= t <= AUDIT_WIN[1]:
        return "audit repair (2026-05-07)"
    if w["base"][0] <= t <= w["base"][1]:
        return "base run"
    if w["same_session"] and w["same_session"][0] <= t <= w["same_session"][1]:
        return "added later in the same session"
    return "later session (extension)"


# ------------------------------------------------------------ A. spec phases
spec_phases = {}
for p, s in SPECS.items():
    txt = open(os.path.join(REPO, s)).read()
    labs = re.findall(r"^###\s+(?:\d+\.\d+\s+)?(Phase|Stage|Experiment)\s+(\d+[a-z]?)\b", txt, re.M)
    spec_phases[p] = dict(spec=s, sha256=sha256_file(os.path.join(REPO, s)),
                          labels=[f"{a} {b}" for a, b in labs], n=len(labs))

# ------------------------------------------------------------ B. phase evidence
evidence = {}
for p in PIPELINES:
    ev = collections.defaultdict(list)
    root = os.path.join(MT, "runs", p)
    for path in walk(root):
        st = os.stat(path)
        e, _ = eff_time(st)
        if e >= CUTOFF:
            continue
        rel = os.path.relpath(path, root)
        if rel.startswith(("scripts/", "atlas/src", "atlas/dist", "atlas/public", "atlas-data", "atlas/node_modules")):
            continue
        is_audit = "audit" in rel or any(k in rel for k in (
            "chance_baseline", "positive_tf", "synthetic_positive_control", "groupkfold_", "seed_stability",
            "iter_04_replication", "cell_level_benchmark_lite", "bootstrap"))
        names = set()
        for seg in rel.split("/"):
            low = seg.lower()
            for k, v in MULTI.items():
                if low.startswith(k):
                    names.update(f"phase {x}" for x in v)
            m = re.match(r"phase[_-]?(\d+[a-z]?)(?![\d])", low)
            if m and not any(low.startswith(k) for k in MULTI) and not low.startswith(MARKER_LOGS):
                names.add(f"phase {m.group(1)}")
            m = re.match(r"stage(\d+)", low)
            if m and p == "exhaustive-mapping-217M":
                names.add("phase 13 (stage-3 synthesis)")
            elif m:
                names.add(f"stage {m.group(1)}")
            m = re.match(r"experiments?_?(\d)(?:_(\d))?", low)
            if m:
                names.add(f"experiment {m.group(1)}")
                if m.group(2):
                    names.add(f"experiment {m.group(2)}")
            m = re.match(r"iter_(\d+)", low)
            if m:
                names.add(f"autoloop iter {int(m.group(1))}")
        if path.endswith(".log"):
            txt = open(path, errors="ignore").read()
            skipped = set(re.findall(r"Phase (\d+[a-z]?) skipped", txt))
            for m in re.finditer(r"\[Phase (\d+[a-z]?)\]", txt):
                if m.group(1) not in skipped:
                    names.add(f"phase {m.group(1)}")
        for n in names:
            ev[n].append((e, rel, is_audit))
    out = {}
    for n, items in ev.items():
        items.sort()
        non_audit = [x for x in items if not x[2]]
        first = (non_audit or items)[0]
        out[n] = dict(first=iso(first[0]), last=iso(items[-1][0]), n_files=len(items),
                      example=first[1], label=label_for(p, first[0], not non_audit))
    evidence[p] = dict(sorted(out.items(), key=lambda kv: kv[1]["first"]))

# ------------------------------------------------------------ C. devices
devices = {}
tl = json.load(open(os.path.join(OUT, "timeline.json")))
for p in PIPELINES:
    logs = [l for l in tl["logs"] if l["pipeline"] == p]
    rc = {}
    for f in glob.glob(f"{MT}/runs/{p}/**/run_config.json", recursive=True):
        if os.stat(f).st_mtime >= CUTOFF or "/._" in f:
            continue
        d = json.load(open(f))
        rc[os.path.relpath(f, MT)] = d.get("device")
    scr = [s for s in glob.glob(f"{MT}/runs/{p}/**/*.py", recursive=True)
           if "/._" not in s and "/node_modules/" not in s and os.stat(s).st_mtime < CUTOFF]
    mps_default, cpu_forced, cuda_any = [], [], []
    for s in scr:
        t = open(s, errors="ignore").read()
        rel = os.path.relpath(s, MT)
        if re.search(r"[\"']mps[\"']\s+if\s+torch\.backends\.mps\.is_available\(\)", t) or re.search(r"device\s*=\s*[\"']mps[\"']", t):
            mps_default.append(rel)
        for m in re.finditer(r"^\s*(?:DEVICE|device|SAE_DEVICE)\s*=\s*[\"']cpu[\"'].*$", t, re.M):
            cpu_forced.append(f"{rel}: {m.group(0).strip()[:90]}")
        if "cuda" in t:
            cuda_any.append(rel)
    devices[p] = dict(
        logs_total=len(logs),
        logs_mentioning_mps=sum("mps" in l["device_mentions"] for l in logs),
        logs_mentioning_cuda=sum("cuda" in l["device_mentions"] for l in logs),
        logs_mentioning_cpu_device=[l["log"] for l in logs if "cpu" in l["device_mentions"]],
        run_config_device_values=dict(collections.Counter(v for v in rc.values() if v)),
        scripts_total=len(scr), scripts_with_mps_default=len(mps_default),
        scripts_forcing_cpu=cpu_forced, scripts_mentioning_cuda=[os.path.relpath(s, MT) for s in scr if "cuda" in open(s, errors="ignore").read()],
    )
sae_train = dict(
    topk_sae_default=re.search(r"device[^\n]*=\s*[\"'](\w+)[\"']", open(MT + "/setup/topk_sae.py").read()).group(0),
    full_12layer_log_line=[l for l in open(MT + "/runs/sae-atlas-217M/outputs/full_12layer.log", errors="ignore").read().splitlines() if "device=" in l][:1],
)

# ------------------------------------------------------------ D. token budget
def jget(rel, *keys):
    d = json.load(open(os.path.join(MT, rel)))
    for k in keys:
        d = d[k]
    return d


maxlen = {}
for p in PIPELINES:
    vals = collections.Counter()
    for s in glob.glob(f"{MT}/runs/{p}/**/*.py", recursive=True):
        if "/._" in s or "/node_modules/" in s or os.stat(s).st_mtime >= CUTOFF:
            continue
        t = open(s, errors="ignore").read()
        for m in re.finditer(r"(?:MAX_LEN|MAX_SEQ_LEN|MAX_GENES_PER_CELL)\s*=\s*(?:int\(os\.environ\.get\([^,]+,\s*\")?(\d+)", t):
            vals[int(m.group(1))] += 1
        for m in re.finditer(r"max_len\s*=\s*(\d+)", t):
            vals[int(m.group(1))] += 1
        if "L_cap = tokenizer.model_input_size" in t:
            vals["model_input_size (4096, setup/maxtoki_adapter.py default)"] += 1
    maxlen[p] = dict(vals)

sae_pos = 1019996
recorded = {
    "spectral-geometry-217M": {"TS immune, 2,000 cells (phase0 run_config mean_seq_len)": jget("runs/spectral-geometry-217M/outputs/phase0/run_config.json", "mean_seq_len")},
    "attention-grn-217M": {
        "K562 217M, 2,000 cells": jget("runs/attention-grn-217M/outputs/phase0/run_config.json", "mean_seq_len"),
        "RPE1": jget("runs/attention-grn-217M/outputs/phase0_rpe1/run_config.json", "mean_seq_len"),
        "Adamson": jget("runs/attention-grn-217M/outputs/phase0_adamson/run_config.json", "mean_seq_len"),
        "K562 1B, 200 cells": jget("runs/attention-grn-217M/outputs/phase0_k562_1b/run_config.json", "mean_seq_len"),
    },
    "sae-atlas-217M": {
        "K562 500 cells: positions / cells (full_12layer.log '1,019,996 positions', N_ctrl=500)": sae_pos / 500,
        "TS immune 500 cells, phase 9 (phase9.log 'TS positions: 886,965')": 886965 / 500,
    },
}
# longevity: the key is nested; find it
es = json.load(open(MT + "/runs/longevity-mechinterp-217M/outputs/stage1_20260505/extraction_stats.json"))
def find(d, key):
    if isinstance(d, dict):
        for k, v in d.items():
            if k == key:
                return v
            r = find(v, key)
            if r is not None:
                return r
    return None
recorded["longevity-mechinterp-217M"] = {"TS immune subset, 2,974 cells (extraction_stats.json mean_seq_len)": find(es, "mean_seq_len")}
import numpy as np
mani = {}
for panel in ("internal", "external", "zeroshot", "lung_control", "lung_nonhema"):
    z = np.load(f"{MT}/runs/manifold-discovery-217M/outputs/phase1/cells_{panel}.npz")
    L = z["seq_lens"]
    tid = z["token_ids"]
    mani[panel] = dict(n_cells_tokenized=int(len(L)), mean=round(float(L.mean()), 1), median=float(np.median(L)),
                       max=int(L.max()), n_at_4096_cap=int((L >= 4096).sum()),
                       frac_at_cap=round(float((L >= 4096).mean()), 4), max_token_id=int(tid.max()))
    del tid, z
recorded["manifold-discovery-217M"] = dict(per_panel_from_npz=mani,
    note="all tokenized cells of each panel; the forward pass used a stratified subset for external (12,000) and zeroshot (5,120) per STATUS.md")
for p in ("topology-141-217M", "circuit-tracing-217M", "exhaustive-mapping-217M"):
    recorded[p] = "not recorded in any run file"
# sanity: positions in the SAE log
assert "1,019,996 positions" in open(MT + "/runs/sae-atlas-217M/outputs/full_12layer.log", errors="ignore").read()
assert "TS positions: 886,965" in open(MT + "/runs/sae-atlas-217M/outputs/phase9.log", errors="ignore").read()

# ------------------------------------------------------------ E. input format
adapter = open(MT + "/setup/maxtoki_adapter.py").read()
tokd = json.load(open(MT + "/setup/token_dictionary.json"))
non_gene = {k: v for k, v in tokd.items() if not k.startswith("ENSG")}
fmt = {}
for p in PIPELINES:
    scr = [s for s in glob.glob(f"{MT}/runs/{p}/**/*.py", recursive=True)
           if "/._" not in s and "/node_modules/" not in s and os.stat(s).st_mtime < CUTOFF]
    one_cell = [os.path.relpath(s, MT) for s in scr if re.search(r"token_ids\[None,\s*:\]|token_ids\[i,\s*:L\]|token_ids\[ci:ci\+1\]|torch\.tensor\(tokens", open(s, errors="ignore").read())]
    special = [os.path.relpath(s, MT) for s in scr if re.search(r"<boq>|<eoq>|time_token|20275\s*\+|>= ?20275", open(s, errors="ignore").read())]
    fmt[p] = dict(n_scripts_building_one_cell_sequences=len(one_cell), scripts=one_cell,
                  scripts_using_time_or_query_tokens=special)
input_format = dict(
    per_pipeline=fmt,
    token_dictionary_entries=len(tokd), non_gene_tokens=len(non_gene),
    time_tokens=sum(1 for k in non_gene if re.fullmatch(r"-?\d+", k)),
    time_and_query_token_id_range=[min(v for k, v in non_gene.items() if v >= 20275), max(non_gene.values())],
    hf_vocab_size=json.load(open(MT + "/setup/MaxToki-217M-HF/config.json"))["vocab_size"],
    adapter_keeps_only_gene_ids_below_20275="v < 20275" in adapter,
    adapter_sequence="<bos> + genes ranked by median-normalised expression, clipped to max_len-2, + <eos>" if "order[: max_len - 2]" in adapter else "?",
)

# ------------------------------------------------------------ F. loops
drv = MT + "/runs/spectral-geometry-217M/autoloop/run_maxtoki_autoloop.py"
drv_t = open(drv).read()
scr_json = sorted(glob.glob(MT + "/runs/spectral-geometry-217M/autoloop/iterations/*/executor_hypothesis_screen.json"))
hyps = [h for f in scr_json for h in json.load(open(f))["hypotheses"]]
dlog = open(MT + "/runs/spectral-geometry-217M/autoloop/runtime/driver.log").read().splitlines()
topo = MT + "/runs/topology-141-217M/scripts/phase11_autoloop.py"
topo_t = open(topo).read()
asum = json.load(open(MT + "/runs/topology-141-217M/outputs/phase11_autoloop/autoloop_summary.json"))
mr = open(MT + "/runs/topology-141-217M/outputs/phase11_autoloop/MANUAL_REVIEW.md").read()
tbl = re.findall(r"^\|\s*(iter_0\d[^|]*)\|\s*([^|]+)\|\s*([^|]+)\|", mr, re.M)
changed = [(a.strip(), b.strip(), c.strip()) for a, b, c in tbl if b.strip().strip("*").upper() != c.strip().strip("*").upper()]
status = open(MT + "/runs/manifold-discovery-217M/STATUS.md").read()
mani_scripts = [s for s in glob.glob(MT + "/runs/manifold-discovery-217M/scripts/*.py") if os.stat(s).st_mtime < CUTOFF and "/._" not in s]
loops = {
    "spectral-geometry-217M": dict(
        kind="two-agent autoloop driver (executor + brainstormer), separate CLI processes",
        driver=os.path.relpath(drv, MT),
        model_default=re.search(r'CLAUDE_MODEL = os\.environ\.get\("CLAUDE_MODEL", "(\w+)"\)', drv_t).group(1),
        driver_log_first_line=dlog[0][:140], driver_log_last_line=dlog[-1],
        n_spawns=sum("spawning" in l for l in dlog),
        n_spawns_with_model_sonnet=sum("--model sonnet" in l for l in dlog),
        stop_rule_in_code="stop after 2 consecutive executor failures (failure = missing report / missing or empty hypothesis JSON / no data file); also STOP file or max iteration",
        stop_rule_lines=[i + 1 for i, l in enumerate(drv_t.splitlines()) if "consecutive_failures >= 2" in l],
        retirement_in_code="none: the driver passes the `retired` flags back into the next prompt but never acts on them",
        n_hypotheses_in_screens=len(hyps), n_marked_retired=sum(bool(h.get("retired")) for h in hyps),
        iterations_run=sorted({os.path.basename(os.path.dirname(f)) for f in glob.glob(MT + "/runs/spectral-geometry-217M/autoloop/iterations/*/executor_prompt.md")}),
        outcome="aborted after 2 consecutive executor failures; both failures were API connection errors",
        api_errors=[l.strip()[:80] for f in sorted(glob.glob(MT + "/runs/spectral-geometry-217M/autoloop/runtime/iter_*.log")) for l in open(f).read().splitlines() if "API Error" in l],
        reported_in_pipeline_summary=("autoloop" in open(MT + "/summaries/spectral-geometry-217M-FINAL_SUMMARY.md").read().lower()) and ("iter_0002" in open(MT + "/summaries/spectral-geometry-217M-FINAL_SUMMARY.md").read()),
        summary_phase10_line=[l[:160] for l in open(MT + "/summaries/spectral-geometry-217M-FINAL_SUMMARY.md").read().splitlines() if l.startswith("- **Phase 10**")],
    ),
    "topology-141-217M": dict(
        kind="one Python script with 6 hard-coded hypotheses; no LLM call inside the loop",
        script=os.path.relpath(topo, MT),
        decision_rule_lines=[i + 1 for i, l in enumerate(topo_t.splitlines()) if re.search(r"PROMOTE if|INCONCLUSIVE if|RETIRE otherwise", l)],
        stop_rule="3 consecutive RETIRE", stop_rule_line=[i + 1 for i, l in enumerate(topo_t.splitlines()) if "consecutive_negatives >= 3" in l],
        coded_outcome=dict(n_iterations=asum["n_iterations"], n_promote=asum["n_promote"], n_inconclusive=asum["n_inconclusive"], n_retire=asum["n_retire"]),
        stop_rule_fired="stopping early" in open(MT + "/runs/topology-141-217M/logs/phase11.log", errors="ignore").read(),
        manual_review_changed=changed, n_changed=len(changed),
    ),
    "manifold-discovery-217M": dict(
        kind="Claude Code /loop wake-ups reading and updating a hand-kept STATUS.md tracker",
        status_line_3=status.splitlines()[2][:160],
        iterations_dir_files=len([x for x in os.listdir(MT + "/runs/manifold-discovery-217M/iterations") if not x.startswith("._")]),
        scripts_mentioning_retire=sum("retire" in open(s, errors="ignore").read().lower() for s in mani_scripts),
        n_scripts=len(mani_scripts),
        stop_statements=[l[:120] for l in status.splitlines() if "STOPPED" in l or "Loop ending" in l],
    ),
    "others": "attention-grn, sae-atlas, circuit-tracing, exhaustive-mapping, longevity: no loop code; the agent ran phases in one chat session and the human asked 'is anything remaining?' / 'proceed' (see human_prompts.json categories scope_check / continue)",
}

# ------------------------------------------------------------ G. topology-141 scope
rows = list(csv.DictReader(open(MT + "/runs/topology-141-217M/outputs/phase12/h141_strict_margins.csv")))
gk = json.load(open(MT + "/runs/topology-141-217M/outputs/groupkfold_h123/summary.json"))
hyp_ids = collections.defaultdict(set)
for s in glob.glob(MT + "/runs/topology-141-217M/scripts/*.py"):
    if os.stat(s).st_mtime >= CUTOFF or "/._" in s:
        continue
    for h in set(re.findall(r"\bH(\d{1,3})\b", open(s, errors="ignore").read())):
        hyp_ids[int(h)].add(os.path.basename(s))
topology = dict(
    readme_scope_line=[l for l in open(MT + "/runs/topology-141-217M/README.md").read().splitlines() if "headline backbone" in l][:1],
    summary_not_rescreened_line=[l.strip() for l in open(MT + "/runs/topology-141-217M/FINAL_SUMMARY.md").read().splitlines() if "130+" in l][:1],
    numbered_hypotheses_named_in_scripts=sorted(hyp_ids),
    n_numbered_hypotheses_named_in_scripts=len(hyp_ids),
    n_numbered_hypotheses_with_code=len(hyp_ids) - 2,
    autoloop_extra_hypotheses=[r["hypothesis_id"] for r in asum["per_iter"]],
    id_1_note="ID 1 matches both hypothesis H01 (tested, phase5) and the homology dimension 'H1' (not a hypothesis)",
    ids_only_referenced_not_tested={"H30": "named as the source-paper companion of H-hyp (phase11_autoloop.py:13)",
                                    "H136": "named as the source of a support matrix that was not rebuilt (phase10_h139_sectional_anisotropy.py:25)"},
    immune_domain=json.load(open(MT + "/runs/topology-141-217M/outputs/phase0/immune/run_config.json")),
    new_domain_extractions={d: {k: json.load(open(MT + f"/runs/topology-141-217M/outputs/phase0/{d}/run_config.json")).get(k) for k in ("n_cells", "max_len", "device", "seed", "wall_seconds")} for d in ("lung", "external_lung")},
    strict_max_null=dict(layers=sorted({int(r["layer"]) for r in rows}), domains=sorted({r["domain"] for r in rows}),
                         metrics=sorted({r["metric"] for r in rows}), n_tests=len(rows),
                         n_pass=sum(int(r["passes_strict"]) for r in rows),
                         code_line=[i + 1 for i, l in enumerate(open(MT + "/runs/topology-141-217M/scripts/phase12_strict_max_null.py").read().splitlines()) if "layer_set = [3, 6, 9]" in l],
                         n_layer_states_available=12),
    dual_axis_cv=dict(scope=gk["scope"], n_folds=gk["n_folds"], domains=sorted({r["domain"] for r in gk["by_layer"]}),
                      layers=sorted({r["layer"] for r in gk["by_layer"]}),
                      n_test_pairs_dual_per_layer=sorted({r["n_test_pairs_dual"] for r in gk["by_layer"]}),
                      auc_dual_axis_range=[round(min(r["auc_dual_axis_disjoint"] for r in gk["by_layer"]), 3), round(max(r["auc_dual_axis_disjoint"] for r in gk["by_layer"]), 3)],
                      hypothesis="H123 only", written=iso(os.stat(MT + "/runs/topology-141-217M/outputs/groupkfold_h123/summary.json").st_mtime)),
)

# ------------------------------------------------------------ H. H115 / H118
def count_hits(top, pat=r"\bH11[58]\b"):
    n, files = 0, []
    for path in walk(top):
        if "/revision/" in path:
            continue
        if not path.endswith((".md", ".py", ".json", ".csv", ".tex", ".txt", ".log", ".tsv")):
            continue
        try:
            if os.path.getsize(path) > 50e6:
                continue
            t = open(path, errors="ignore").read()
        except OSError:
            continue
        k = len(re.findall(pat, t))
        if k:
            n += k
            files.append(os.path.relpath(path, REPO))
    return n, files


h115 = {}
for sub in ["projects/maxtoki/runs", "projects/maxtoki/summaries", "projects/maxtoki/audits", "projects/maxtoki/setup", "pipelines"] + \
        sorted(os.path.relpath(d, REPO) for d in glob.glob(MT + "/paper*")):
    n, files = count_hits(os.path.join(REPO, sub))
    h115[sub] = dict(hits=n, files=files[:8])
plos_main_lines = [i + 1 for i, l in enumerate(open(MT + "/paper-plos-one/main.tex", errors="ignore").read().splitlines()) if re.search(r"H11[58]", l)]
mani_ids = sorted({int(x) for x in re.findall(r"\bH(\d{2,3})\b", open(MT + "/runs/manifold-discovery-217M/STATUS.md").read() + open(MT + "/runs/manifold-discovery-217M/FINAL_SUMMARY.md").read())})

res = dict(spec_phases=spec_phases, phase_evidence=evidence, session_windows={k: {kk: ([iso(x) for x in vv] if vv else None) for kk, vv in v.items()} for k, v in WINDOWS.items()},
           devices=devices, sae_training_device=sae_train, token_budget=dict(max_len_values_in_scripts=maxlen, recorded_sequence_lengths=recorded,
           note="recorded lengths include the <bos> and <eos> tokens; gene tokens per cell = length - 2"),
           input_format=input_format, loops=loops, topology=topology,
           h115_h118_search=dict(pattern=r"\bH11[58]\b", per_folder=h115, plos_main_tex_lines=plos_main_lines,
                                 note="paper-plos-one/revision/ (this revision's own investigation notes) is excluded", manifold_hypothesis_ids_in_status_and_summary=[mani_ids[0], mani_ids[-1]]))
print(dump("pipeline_facts.json", res))
for p in PIPELINES:
    print(p, spec_phases[p]["n"], {k: v["label"][:6] + " " + v["first"][5:16] for k, v in evidence[p].items()})
print(json.dumps(dict(devices={p: (d["logs_mentioning_mps"], d["logs_mentioning_cuda"], d["run_config_device_values"], len(d["scripts_forcing_cpu"])) for p, d in devices.items()},
                      maxlen=maxlen, recorded=recorded, loops_topo=loops["topology-141-217M"]["n_changed"], topo=topology["strict_max_null"],
                      h115={k: v["hits"] for k, v in h115.items()}), indent=1, default=str))
