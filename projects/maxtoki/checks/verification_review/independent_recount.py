#!/usr/bin/env python3
"""Independent re-derivation of the checker's key numbers (verification pass, 2026-10-01).

Written by the verifying agent, not by the checker's author. It reads the specs, the run files and the
checker's per-item CSVs, and recomputes the headline numbers with its own code:
  V1  template sections (S1) from raw '## ' headings of the 8 deployed specs;
  V2  code-pointer totals from results/spec_lint_coderefs.csv (no use of checker_summary.json);
  V3  number tracing pooled over DISTINCT numbers per run (Wilson; bootstrap over numbers; bootstrap over runs);
  V4  a second chance null: numbers written in OTHER runs' summaries, searched in this run's outputs,
      re-weighted to this run's mix of number classes (decimal_1dp, decimal_2+dp, integer>=10, sci);
  V5  manifold gate objects re-read from the JSON files, with the frozen file's panel rule
      (five gates on the internal panel, four on the external panel).
Seed 20261001. CPU only. Writes verification_review/results.json and run_config.json (sha256 of inputs).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

SEED = 20261001
ROOT = Path("<REPO_ROOT>")
PROJ = ROOT / "projects/maxtoki"
CHK = PROJ / "checks"
RES = CHK / "results"
OUT = CHK / "verification_review"
SPECS = ["pipelines/attention-grn-extraction-and-evaluation.md", "pipelines/residual-stream-spectral-geometry.md",
         "pipelines/topology-geometry-141-hypotheses.md", "pipelines/manifold-discovery-extraction-compactification.md",
         "pipelines/longevity-mechinterp-donor-aware.md", "pipelines/sparse-autoencoders/01-sae-atlas.md",
         "pipelines/sparse-autoencoders/02-causal-circuit-tracing.md",
         "pipelines/sparse-autoencoders/03-exhaustive-mapping-and-steering.md"]
TEMPLATE = ["overview", "source", "inputs", "outputs", "dependencies", "methodology", "code references",
            "parameters", "validation", "known pitfalls"]
inputs: list[Path] = []


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - w, 4), round(c + w, 4)]


def v1_sections():
    out = {}
    for s in SPECS:
        p = ROOT / s
        inputs.append(p)
        heads, code = [], False
        for ln in p.read_text(encoding="utf-8").splitlines():
            if ln.startswith("```"):
                code = not code
            elif not code and ln.startswith("## "):
                heads.append(re.sub(r"^##\s+(\d+\.\s*)?", "", ln).strip().lower())
        first10 = heads[:10]
        ok = all(first10[i].startswith(TEMPLATE[i]) for i in range(10)) if len(first10) == 10 else False
        out[s.split("/")[-1]] = dict(ok=ok, missing=[t for t in TEMPLATE if not any(h.startswith(t) for h in heads)])
    return dict(n_fail=sum(not v["ok"] for v in out.values()), per_spec=out)


def v2_coderefs():
    p = RES / "spec_lint_coderefs.csv"
    inputs.append(p)
    rows = [r for r in csv.DictReader(open(p)) if r["spec"] in SPECS]
    st = Counter(r["resolve_status"] for r in rows)
    return dict(n_refs=len(rows), required_form=sum(r["required_form"] == "True" for r in rows),
                exists=sum(r["exists"] == "True" for r in rows),
                as_written=sum(r["exists"] == "True" and r["resolve_status"] == "ok" for r in rows),
                pattern=st.get("pattern", 0),
                not_resolved=sum(r["exists"] == "False" and r["resolve_status"] != "pattern" for r in rows),
                in_clone_without_git=sum(r["exists"] == "True" and r["clone_has_git"] == "False" for r in rows),
                symbols_not_found=sum(s.count("NOT FOUND") for s in (r["symbols"] for r in rows)))


def load_distinct():
    p = RES / "number_trace_numbers.csv"
    inputs.append(p)
    rows = [r for r in csv.DictReader(open(p)) if r["scope"] == "run_summary"]
    d = {}
    for r in rows:
        d.setdefault((r["run"], round(float(r["value"]), 12), round(float(r["half_width"]), 15), r["pct"]), r)
    return rows, list(d.values())


def found(vals, x, h, pct):
    slack = h * 1e-9 + 1e-12 * max(1.0, abs(x))
    for s in ((1.0, 100.0) if pct else (1.0,)):
        lo, hi = (x - h - slack) / s, (x + h + slack) / s
        i = int(np.searchsorted(vals, lo, side="left"))
        if i < len(vals) and vals[i] <= hi:
            return True
    return False


def v3_v4_numbers():
    rows, u = load_distinct()
    runs = sorted({r["run"] for r in u})
    tr = np.array([r["traced"] == "True" for r in u], float)
    ch = np.array([float(r["chance_rate"]) for r in u])
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(u), (2000, len(u)))
    per = {}
    for R in runs:
        vp = RES / "cache" / f"{R}.values.npy"
        inputs.append(vp)
        vals = np.load(vp)
        mine = [r for r in u if r["run"] == R]
        own = {round(float(r["value"]), 12) for r in mine}
        mix = Counter(r["cls"] for r in mine)
        foreign = {}
        for r in u:
            if r["run"] != R and round(float(r["value"]), 12) not in own:
                foreign.setdefault((round(float(r["value"]), 12), round(float(r["half_width"]), 15), r["pct"]), r)
        rate = {}
        for c in mix:
            f = [r for r in foreign.values() if r["cls"] == c]
            rate[c] = float(np.mean([found(vals, float(r["value"]), float(r["half_width"]), r["pct"] == "True")
                                     for r in f])) if f else float("nan")
        okc = [c for c in mix if not math.isnan(rate[c])]
        per[R] = dict(n=len(mine), traced=float(np.mean([r["traced"] == "True" for r in mine])),
                      tool_chance=float(np.mean([float(r["chance_rate"]) for r in mine])),
                      foreign_chance_class_matched=sum(mix[c] * rate[c] for c in okc) / sum(mix[c] for c in okc),
                      n_foreign=len(foreign), class_mix=dict(mix), foreign_rate_by_class=rate)
    keys = ("traced", "tool_chance", "foreign_chance_class_matched")
    N = sum(p["n"] for p in per.values())
    pooled = {k: sum(p["n"] * p[k] for p in per.values()) / N for k in keys}
    B = []
    for _ in range(2000):
        pick = [runs[i] for i in rng.integers(0, len(runs), len(runs))]
        n = sum(per[R]["n"] for R in pick)
        B.append([sum(per[R]["n"] * per[R][k] for R in pick) / n for k in keys])
    B = np.array(B)
    pc = lambda a: [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
    return dict(
        pooled_rows=len(rows), distinct_numbers=len(u), rows_repeating_a_number=len(rows) - len(u),
        traced=int(tr.sum()), traced_share=round(float(tr.mean()), 4), traced_wilson=wilson(int(tr.sum()), len(u)),
        tool_chance=round(float(ch.mean()), 4), tool_chance_ci_numbers=pc(ch[idx].mean(1)),
        excess_tool=round(float((tr - ch).mean()), 4), excess_tool_ci_numbers=pc((tr - ch)[idx].mean(1)),
        weak=sum(r["weak"] == "True" for r in u),
        untraced_not_in_spec=sum(r["traced"] == "False" and r["also_in_spec"] == "False" for r in u),
        pooled_by_run_weighting={k: round(v, 4) for k, v in pooled.items()},
        run_cluster_ci={k: pc(B[:, i]) for i, k in enumerate(keys)},
        excess_tool_run_cluster_ci=pc(B[:, 0] - B[:, 1]),
        excess_foreign=round(pooled["traced"] - pooled["foreign_chance_class_matched"], 4),
        excess_foreign_run_cluster_ci=pc(B[:, 0] - B[:, 2]),
        per_run=per,
        methods="unit = distinct number per run; Wilson over numbers; '_ci_numbers' = percentile bootstrap over "
                "numbers (2,000); 'run_cluster' = percentile bootstrap over the 8 runs (2,000)")


def v5_gates():
    rep = PROJ / "runs/manifold-discovery-217M/reports"
    spec_p = rep / "quality_gates_spec.json"
    inputs.append(spec_p)
    spec = json.loads(spec_p.read_text())
    gates = spec["gates"]
    alias = {"trustworthiness": "trustworthiness_min", "random_holdout": "random_holdout_correlation_min",
             "donor_holdout": "donor_holdout_correlation_min", "branch_holdout": "clade_branch_holdout_correlation_min",
             "category_holdout": "clade_branch_holdout_correlation_min"}
    out = []
    for fn in ("quality_gates_let_anchor.json", "h38_lite_quality_gates.json", "external_validation_external.json",
               "external_validation_lung_control.json", "external_validation_lung_nonhema.json",
               "zeroshot_transfer_anchor_head.json"):
        p = rep / fn
        inputs.append(p)
        o = json.loads(p.read_text())
        objs = [("positive", o["positive"])] if "positive" in o else [("(root)", o)]
        for name, d in objs:
            rec = d.get("passes", d.get("all_gates_pass"))
            have = {alias[k] for k in d if k in alias}
            panel = o.get("panel", "internal (no panel key; internal-panel anchors)" if "positive" in o else None)
            need = set(gates) if str(panel).startswith("internal") else (
                set(gates) - {"blocked_permutation_p_max", "n_permutations_min"} if panel in ("external", "lung_control") else None)
            computed_ok = all((d[k] >= gates[alias[k]]) for k in d if k in alias)
            out.append(dict(file=fn, object=name, panel=panel, recorded_pass=rec, computed_gates_pass=computed_ok,
                            required_gates_missing=sorted(need - have) if need is not None else "rule does not name this panel",
                            expected_to_fail=o.get("expected_to_fail")))
    return out


def main():
    res = dict(V1_sections=v1_sections(), V2_coderefs=v2_coderefs(), V3_V4_numbers=v3_v4_numbers(), V5_gates=v5_gates())
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str))
    me = Path(__file__).resolve()
    cfg = dict(tool=str(me.relative_to(ROOT)), tool_sha256=sha(me), started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               python=sys.version.split()[0], seed=SEED,
               inputs=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p), bytes=p.stat().st_size) for p in sorted(set(inputs))])
    (OUT / "run_config.json").write_text(json.dumps(cfg, indent=1))
    n = res["V3_V4_numbers"]
    print("V1 S1 failures:", res["V1_sections"]["n_fail"], "of", len(SPECS))
    print("V2", res["V2_coderefs"])
    print("V3 distinct", n["distinct_numbers"], "traced", n["traced"], n["traced_wilson"], "tool chance", n["tool_chance"],
          "excess", n["excess_tool"], "run-CI", n["excess_tool_run_cluster_ci"])
    print("V4 foreign chance", n["pooled_by_run_weighting"], "excess_foreign", n["excess_foreign"], n["excess_foreign_run_cluster_ci"])
    for g in res["V5_gates"]:
        print("V5", g)


if __name__ == "__main__":
    main()
