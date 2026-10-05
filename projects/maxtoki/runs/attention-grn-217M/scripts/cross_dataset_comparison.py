"""Cross-dataset comparison: K562 vs RPE1 verdict on the four mandatory criteria.

Reads phase12_verdict.json and phase12_verdict_rpe1.json, builds a side-by-side
replication table and a combined Markdown report.
"""
from __future__ import annotations

import json
from pathlib import Path

PROJ = Path(__file__).resolve().parents[3]
OUT = PROJ / "runs/attention-grn-217M/outputs"

k = json.loads((OUT / "phase12_verdict.json").read_text())
r_path = OUT / "phase12_verdict_rpe1.json"
r = json.loads(r_path.read_text()) if r_path.exists() else None

lines = [
    "# Cross-context replication — MaxToki-217M, K562 vs RPE1\n",
    "One-line summary of the attention-GRN pipeline verdict on both cell lines.\n",
    "## Headline\n",
    f"- **K562:** {k['overall']['verdict']}",
]
if r:
    lines.append(f"- **RPE1:** {r['overall']['verdict']}")
else:
    lines.append("- **RPE1:** not yet run")
lines.append("")

lines.append("## Four mandatory criteria\n")
lines.append("| Criterion | K562 | RPE1 |")
lines.append("|---|---|---|")
for cname in [
    "C1_attention_beats_trivial_baselines",
    "C2_incremental_value_over_gene_features",
    "C3_residualized_retains_signal",
    "C4_top_heads_causal",
]:
    kv = k["criteria"].get(cname, {})
    kvp = "PASS" if kv.get("pass") else ("FAIL" if kv.get("pass") is False else "–")
    if r:
        rv = r["criteria"].get(cname, {})
        rvp = "PASS" if rv.get("pass") else ("FAIL" if rv.get("pass") is False else "–")
    else:
        rvp = "–"
    lines.append(f"| {cname} | {kvp} | {rvp} |")
lines.append("")

def _fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else str(v)

lines.append("## Headline metrics\n")
lines.append("| Metric | K562 | RPE1 |")
lines.append("|---|---|---|")

def _pull(x, *keys, default=None):
    for k2 in keys:
        if x is None:
            return default
        x = x.get(k2) if isinstance(x, dict) else None
    return x if x is not None else default

pairs = [
    ("attention mean AUROC (primary layer)",
        _pull(k, "criteria", "C1_attention_beats_trivial_baselines", "details", "attention_mean_auroc"),
        _pull(r, "criteria", "C1_attention_beats_trivial_baselines", "details", "attention_mean_auroc"),
    ),
    ("incremental ΔAUROC (cross-pert gene+attn vs gene-only)",
        _pull(k, "criteria", "C2_incremental_value_over_gene_features", "delta_auroc"),
        _pull(r, "criteria", "C2_incremental_value_over_gene_features", "delta_auroc"),
    ),
    ("residualized TRRUST AUROC (OLS cross-fit)",
        _pull(k, "criteria", "C3_residualized_retains_signal", "residualized_auroc"),
        _pull(r, "criteria", "C3_residualized_retains_signal", "residualized_auroc"),
    ),
    ("baseline TRRUST AUROC",
        _pull(k, "criteria", "C3_residualized_retains_signal", "baseline_auroc"),
        _pull(r, "criteria", "C3_residualized_retains_signal", "baseline_auroc"),
    ),
    ("top-5 TRRUST-head ablation Δ AUROC (primary layer)",
        _pull(k, "criteria", "C4_top_heads_causal", "delta_top5"),
        _pull(r, "criteria", "C4_top_heads_causal", "delta_top5"),
    ),
]
for label, kv, rv in pairs:
    lines.append(f"| {label} | {_fmt(kv)} | {_fmt(rv)} |")
lines.append("")

lines.append("## Scope\n")
lines.append("- Dataset K562: Replogle concat h5ad, non-targeting NT controls, 2000 cells")
lines.append("- Dataset RPE1: ReplogleWeissman2022_rpe1.h5ad, `control` category, 2000 cells")
lines.append("- Reference network: TRRUST only")
lines.append("- N_hvg: 1500 per dataset (intersected with MaxToki Ensembl vocab)")
lines.append(f"- K562 primary layer: L{_pull(k, 'criteria', 'C4_top_heads_causal')  and 8}  (Llama decoder, 11 layers)")
lines.append("- K562 ran all phases (0, 0b, 1, 2, 3, 4, 12). RPE1 scope: 0, 1, 2, 3, 12 (no Phase 0b or Phase 4 to fit compute budget).")

out = OUT / "cross_dataset_report.md"
out.write_text("\n".join(lines))
print(f"wrote: {out}")
