"""Phase 12 — Verdict and run report.

Reads all Phase 0–4 outputs and emits a pass/fail verdict on the four
mandatory criteria required to advertise the attention-derived edges as
a gene regulatory network.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import os
PROJ = Path(__file__).resolve().parents[3]
DATASET = os.environ.get("DATASET", "k562").lower()
_SUFFIX = os.environ.get("OUT_SUFFIX", "" if DATASET == "k562" else f"_{DATASET}")
OUT = PROJ / "runs/attention-grn-217M/outputs"
VERDICT_FILE = OUT / f"phase12_verdict{_SUFFIX}.json"
REPORT_FILE = OUT / f"run_report{_SUFFIX}.md"

def _safe_load_json(p):
    return json.loads(p.read_text()) if p.exists() else None

def _safe_load_csv(p):
    return pd.read_csv(p) if p.exists() else None

phase0_cfg = _safe_load_json(OUT / f"phase0{_SUFFIX}/run_config.json")
phase1_cfg = _safe_load_json(OUT / f"phase1{_SUFFIX}/run_config.json")
phase1_wilc = _safe_load_csv(OUT / f"phase1{_SUFFIX}/wilcoxon_baseline_vs_edges.csv")
phase1_auc = _safe_load_csv(OUT / f"phase1{_SUFFIX}/per_perturbation_auroc.csv")
phase2_res = _safe_load_csv(OUT / f"phase2{_SUFFIX}/incremental_value_results.csv")
phase2_delta = _safe_load_csv(OUT / f"phase2{_SUFFIX}/delta_auroc_summary.csv")
phase2_cfg = _safe_load_json(OUT / f"phase2{_SUFFIX}/run_config.json")
phase3_resid = _safe_load_csv(OUT / f"phase3{_SUFFIX}/residualization_results.csv")
phase3_null = _safe_load_json(OUT / f"phase3{_SUFFIX}/null_results.json")
phase3_match = _safe_load_json(OUT / f"phase3{_SUFFIX}/propensity_matching_results.json")
phase4_res = _safe_load_csv(OUT / f"phase4{_SUFFIX}/ablation_results.csv")

verdict = {"criteria": {}}

# --- C1: Attention beats trivial gene-level baselines --------------------
# Pass if attention_primary has higher mean AUROC than variance/mean/dropout
# and Wilcoxon p < 0.05 in all three direct comparisons.
c1_pass = None
if phase1_auc is not None and phase1_wilc is not None:
    mean_aucs = {c[4:]: float(phase1_auc[c].mean()) for c in phase1_auc.columns
                 if c.startswith("auc_")}
    attn_mean = mean_aucs.get("attention_primary", float("nan"))
    c1_pass = True
    details = {"attention_mean_auroc": attn_mean, "beat_baselines": []}
    for base in ["gene_variance", "gene_mean_expr", "gene_one_minus_dropout"]:
        base_mean = mean_aucs.get(base, float("nan"))
        won = attn_mean > base_mean
        wilc = phase1_wilc[(phase1_wilc.baseline == base)
                           & (phase1_wilc.reference == "attention_primary")]
        p = float(wilc["wilcoxon_p_bh"].iloc[0]) if len(wilc) else float("nan")
        significant = won and (p < 0.05)
        details["beat_baselines"].append({
            "baseline": base, "baseline_auroc": base_mean,
            "attention_won": bool(won), "wilcoxon_p_bh": p,
            "significant": bool(significant),
        })
        if not significant:
            c1_pass = False
    verdict["criteria"]["C1_attention_beats_trivial_baselines"] = {
        "pass": c1_pass, "details": details,
    }

# --- C2: Attention adds incremental value over gene features -------------
# Pass if ΔAUROC(gene_plus_attn - gene_only) > 0.005 under cross-pert GKF.
c2_pass = None
if phase2_delta is not None:
    sel = phase2_delta[(phase2_delta.split == "cross_pert")
                       & (phase2_delta.model == "logreg")
                       & (phase2_delta.feature_set == "gene_plus_attn")]
    if len(sel):
        delta = float(sel["delta_auroc"].iloc[0])
        c2_pass = bool(delta > 0.005)
        verdict["criteria"]["C2_incremental_value_over_gene_features"] = {
            "pass": c2_pass, "delta_auroc": delta,
            "threshold": 0.005,
            "split": "cross_pert", "model": "logreg",
        }

# --- C3: Residualized attention retains TRRUST signal --------------------
# Pass if OLS-residualized attention retains > 50% of the above-chance signal.
c3_pass = None
if phase3_resid is not None:
    row = phase3_resid[(phase3_resid.edge == "attention")
                       & (phase3_resid.model == "ols")]
    if len(row):
        frac = float(row["fraction_above_chance_retained"].iloc[0])
        c3_pass = bool(frac > 0.5)
        verdict["criteria"]["C3_residualized_retains_signal"] = {
            "pass": c3_pass,
            "fraction_above_chance_retained": frac,
            "threshold": 0.5,
            "baseline_auroc": float(row["baseline_trrust_auroc"].iloc[0]),
            "residualized_auroc": float(row["residualized_trrust_auroc"].iloc[0]),
        }

# --- C4: Top-ranked heads carry causal weight ----------------------------
# Pass if ablating top-5 TRRUST heads reduces TRRUST AUROC by > 0.02 AND by
# more than the median random-5 ablation.
c4_pass = None
if phase4_res is not None:
    def _row(c): return phase4_res[phase4_res.condition == c]
    baseline = _row("baseline")
    top5 = _row("top5_trrust")
    if len(baseline) and len(top5):
        base_auc = float(baseline["trrust_auroc_primary"].iloc[0])
        top5_auc = float(top5["trrust_auroc_primary"].iloc[0])
        delta_top5 = base_auc - top5_auc
        rand_conds = phase4_res[phase4_res.condition.str.startswith("random5")]
        rand_deltas = [base_auc - float(r["trrust_auroc_primary"])
                       for _, r in rand_conds.iterrows()]
        rand_median = float(np.median(rand_deltas)) if rand_deltas else float("nan")
        c4_pass = bool((delta_top5 > 0.02) and (delta_top5 > rand_median))
        verdict["criteria"]["C4_top_heads_causal"] = {
            "pass": c4_pass,
            "baseline_auroc": base_auc,
            "top5_auroc": top5_auc,
            "delta_top5": delta_top5,
            "random5_deltas": rand_deltas,
            "random5_median_delta": rand_median,
            "threshold": 0.02,
        }

# Overall verdict
decided = [c for c, v in verdict["criteria"].items() if v.get("pass") is not None]
all_pass = all(v.get("pass") is True for v in verdict["criteria"].values()
               if v.get("pass") is not None)
verdict["overall"] = {
    "criteria_evaluated": len(decided),
    "criteria_total": 4,
    "all_pass": bool(all_pass),
    "verdict": (
        "PASS — attention edges carry evidence compatible with a regulatory interpretation"
        if all_pass and len(decided) == 4
        else ("FAIL — at least one mandatory criterion failed"
              if any(v.get("pass") is False for v in verdict["criteria"].values())
              else "INCONCLUSIVE — some criteria not evaluated in this scoped run")
    ),
}

# Scope notes — phases NOT run
verdict["scope_notes"] = {
    "phases_run": ["0a", "1", "2", "3", "4", "12"],
    "phases_skipped": [
        "0b (value-weighted edges)",
        "5 (cross-context replication — RPE1, Adamson, Dixit, Shifrut, Tian)",
        "6 (CSSI)",
        "7 (biological characterization — STRING, Reactome, KEGG, GO)",
        "8 (detectability)",
        "9 (ortholog transfer)",
        "10 (pseudotime)",
        "11 (batch leakage, calibration, TRRUST circularity, HVG protocol, mediation)",
    ],
    "dataset": "Replogle K562 non-targeting controls only",
    "reference_network": "TRRUST only (no STRING/Reactome/KEGG/GO)",
    "n_ctrl": phase0_cfg.get("n_ctrl") if phase0_cfg else None,
    "n_hvg": phase0_cfg.get("n_hvg") if phase0_cfg else None,
    "curveball_null_iters": phase3_null["curveball"]["n_iter"] if phase3_null else None,
}

with open(VERDICT_FILE, "w") as f:
    json.dump(verdict, f, indent=2)
print(f"verdict written: {VERDICT_FILE}")

# --- Markdown report ------------------------------------------------------
lines = ["# Attention-GRN pipeline — MaxToki-217M run report\n"]
lines.append(f"**Overall verdict:** {verdict['overall']['verdict']}\n")

if phase0_cfg:
    lines.append("## Phase 0 — extraction\n")
    for k in ["model", "n_ctrl", "n_hvg", "max_len", "model_n_layers",
              "model_n_heads", "mean_seq_len", "mean_fwd_seconds",
              "total_phase_seconds", "device"]:
        if k in phase0_cfg:
            lines.append(f"- {k}: `{phase0_cfg[k]}`")
    lines.append("")

if phase1_auc is not None:
    lines.append("## Phase 1 — trivial baselines\n")
    mean_aucs = {c[4:]: float(phase1_auc[c].mean()) for c in phase1_auc.columns
                 if c.startswith("auc_")}
    for k, v in sorted(mean_aucs.items(), key=lambda x: -x[1]):
        lines.append(f"- mean AUROC `{k}`: **{v:.4f}**")
    lines.append("")

if phase1_wilc is not None:
    lines.append("### Paired Wilcoxon vs attention/correlation (n={})\n".format(
        int(phase1_wilc["n_valid"].iloc[0]) if len(phase1_wilc) else 0))
    lines.append("| baseline | reference | mean Δ | Wilcoxon p (BH) |")
    lines.append("|---|---|---|---|")
    for _, r in phase1_wilc.iterrows():
        lines.append(
            f"| {r['baseline']} | {r['reference']} | "
            f"{r['mean_delta']:+.4f} | {r['wilcoxon_p_bh']:.2e} |"
        )
    lines.append("")

if phase2_delta is not None:
    lines.append("## Phase 2 — incremental value (ΔAUROC vs gene-only)\n")
    lines.append("| split | model | feature_set | gene_only | new | ΔAUROC |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in phase2_delta.iterrows():
        lines.append(
            f"| {r['split']} | {r['model']} | {r['feature_set']} | "
            f"{r['gene_only_auroc']:.4f} | {r['new_auroc']:.4f} | "
            f"{r['delta_auroc']:+.4f} |"
        )
    lines.append("")

if phase3_resid is not None:
    lines.append("## Phase 3a — cross-fitted residualization\n")
    lines.append("| edge | model | R² | baseline AUROC | residual AUROC | Δ lost |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in phase3_resid.iterrows():
        lines.append(
            f"| {r['edge']} | {r['model']} | {r['r2_train_mean']:.3f} | "
            f"{r['baseline_trrust_auroc']:.4f} | {r['residualized_trrust_auroc']:.4f} "
            f"| {r['delta_auroc_lost']:+.4f} |"
        )
    lines.append("")

if phase3_null:
    lines.append("## Phase 3b — degree-preserving + label-shuffle nulls\n")
    lines.append(
        f"- observed attention TRRUST AUROC: **"
        f"{phase3_null['baseline_trrust_auroc_attention']:.4f}**"
    )
    cb = phase3_null["curveball"]
    ls = phase3_null["label_shuffle"]
    lines.append(
        f"- curveball null (n={cb['n_iter']}): mean={cb['null_mean']:.4f} "
        f"± {cb['null_std']:.4f}  z = {cb['z']:.2f}"
    )
    lines.append(
        f"- label-shuffle null (n={ls['n_iter']}): mean={ls['null_mean']:.4f} "
        f"± {ls['null_std']:.4f}  z = {ls['z']:.2f}"
    )
    lines.append("")

if phase3_match:
    lines.append("## Phase 3c — propensity matching\n")
    lines.append(f"- n matched pairs: `{phase3_match['n_matched_pairs']}`")
    lines.append(f"- matched positive rate: `{phase3_match['positive_rate_matched']:.4f}`")
    mg = phase3_match["matched_gkf_auroc"]
    lines.append(
        f"- matched GroupKFold AUROC: gene_only={mg['gene_only']:.4f}  "
        f"gene+attn={mg['gene_plus_attn']:.4f}  gene+corr={mg['gene_plus_corr']:.4f}  "
        f"(Δattn={mg['delta_attn']:+.4f}, Δcorr={mg['delta_corr']:+.4f})"
    )
    lines.append("")

if phase4_res is not None:
    lines.append("## Phase 4 — 6-condition causal head ablation\n")
    lines.append("| condition | n_masked | TRRUST AUROC (primary) | ΔAUROC | best layer |")
    lines.append("|---|---|---|---|---|")
    for _, r in phase4_res.iterrows():
        delta = r.get("delta_primary", "–")
        delta_s = f"{delta:+.4f}" if isinstance(delta, (int, float)) else "–"
        lines.append(
            f"| {r['condition']} | {r['n_heads_masked']} | "
            f"{r['trrust_auroc_primary']:.4f} | {delta_s} | L{r['best_layer']} |"
        )
    lines.append("")

lines.append("## Verdict on four mandatory criteria\n")
for cname, cval in verdict["criteria"].items():
    status = "PASS" if cval.get("pass") else ("FAIL" if cval.get("pass") is False else "–")
    lines.append(f"- **{cname}** → `{status}`")
lines.append(f"\n**Overall:** {verdict['overall']['verdict']}")

lines.append("\n## Scope\n")
lines.append(f"- dataset: `{verdict['scope_notes']['dataset']}`")
lines.append(f"- reference network: `{verdict['scope_notes']['reference_network']}`")
lines.append(f"- N_ctrl: `{verdict['scope_notes']['n_ctrl']}`  "
             f"N_hvg: `{verdict['scope_notes']['n_hvg']}`")
lines.append(f"- curveball null iters: `{verdict['scope_notes']['curveball_null_iters']}`")
lines.append(f"\n**Phases run:** {', '.join(verdict['scope_notes']['phases_run'])}")
lines.append(f"**Phases skipped:** " + "; ".join(verdict['scope_notes']['phases_skipped']))

with open(REPORT_FILE, "w") as f:
    f.write("\n".join(lines))
print(f"report written: {REPORT_FILE}")
