"""Deterministic checks behind the MaxToki error ledger (item C-prep).

Reads only saved files. CPU only. No model forward pass.
Writes verify_results.json next to this script, plus the sha256 of every
input it read (for run_config.json).

Each check has an id (V01, V02, ...). The ledger cites these ids in its
deterministic_evidence column. A check "passes" when the saved files show
the error the ledger describes.

Seeds: bootstrap seed 20261001, curveball seeds copied from the deployed
script (42 + trial).
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

MT = Path("<REPO_ROOT>/projects/maxtoki")
R = MT / "runs"
S = MT / "summaries"
A = MT / "audits"
PL = Path("<REPO_ROOT>/pipelines")
TRRUST = Path("<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv")
TS_IMMUNE = Path("<DATA_ROOT>/biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune.h5ad")
OUT = Path(__file__).resolve().parent / "verify_results.json"
BOOT_SEED = 20261001
N_BOOT = 2000

INPUTS: dict[str, str] = {}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def use(p: Path) -> Path:
    """Register an input file (hash recorded once)."""
    key = str(p)
    if key not in INPUTS:
        INPUTS[key] = sha256(p) if p.stat().st_size < 2_000_000_000 else "not-hashed(>2GB)"
    return p


def find_lines(p: Path, pattern: str, flags=0) -> list[int]:
    rx = re.compile(pattern, flags)
    out = []
    with open(use(p), encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            if rx.search(line):
                out.append(i)
    return out


def jload(p: Path):
    with open(use(p)) as f:
        return json.load(f)


results: dict[str, dict] = {}


def check(cid: str, desc: str):
    def deco(fn):
        t0 = time.time()
        try:
            res = fn()
            res.setdefault("pass", None)
        except Exception as e:  # keep going; record the failure
            res = {"pass": None, "error": repr(e)}
        res["description"] = desc
        res["seconds"] = round(time.time() - t0, 2)
        results[cid] = res
        print(f"{cid}: pass={res.get('pass')}  {desc}", flush=True)
        return fn
    return deco


# ---------------------------------------------------------------------------
# Circuit tracing: Phase 11 sign bug and the corrected metric
# ---------------------------------------------------------------------------
@check("V01", "Phase 11 counts actual_lfc<0 and never uses predicted_inhibitory")
def _():
    p = R / "circuit-tracing-217M/scripts/remaining_phases.py"
    count_line = find_lines(p, r"if \(actual_lfc < 0\)")
    pred_def = find_lines(p, r"predicted_inhibitory\s*=")
    pred_uses = [l for l in find_lines(p, r"predicted_inhibitory") if l not in pred_def]
    j = jload(R / "circuit-tracing-217M/outputs/phase11_crispri_validation.json")
    return {"count_line": count_line, "predicted_inhibitory_defined_at": pred_def,
            "predicted_inhibitory_other_uses": pred_uses, "phase11_json": j,
            "pass": bool(count_line) and not pred_uses}


@check("V02", "Corrected CRISPRi records: model vs always-decrease, balanced accuracy, MCC; grouped bootstrap by silenced gene")
def _():
    p = R / "circuit-tracing-217M/outputs/groupkfold_crispri/per_pair.parquet"
    df = pd.read_parquet(use(p), columns=["source", "predicted_inhibitory", "actual_inhibitory"])
    pred = df["predicted_inhibitory"].to_numpy(bool)
    obs = df["actual_inhibitory"].to_numpy(bool)
    n = len(df)
    # Way 1: pooled arrays
    acc = float((pred == obs).mean())
    p_obs = float(obs.mean()); p_pred = float(pred.mean())
    always_dec = p_obs
    tpr = float((pred & obs).sum() / obs.sum()); tnr = float((~pred & ~obs).sum() / (~obs).sum())
    bal = (tpr + tnr) / 2
    indep = p_pred * p_obs + (1 - p_pred) * (1 - p_obs)
    tp = float((pred & obs).sum()); tn = float((~pred & ~obs).sum())
    fp = float((pred & ~obs).sum()); fn = float((~pred & obs).sum())
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    # Way 2: per-source confusion counts, then pooled sums
    g = df.assign(tp=pred & obs, tn=~pred & ~obs, fp=pred & ~obs, fn=~pred & obs).groupby("source")[["tp", "tn", "fp", "fn"]].sum()
    C = g.to_numpy(float)
    acc2 = (C[:, 0].sum() + C[:, 1].sum()) / C.sum()
    dec2 = (C[:, 0].sum() + C[:, 3].sum()) / C.sum()
    # grouped bootstrap over silenced genes (sources)
    rng = np.random.default_rng(BOOT_SEED)
    k = C.shape[0]
    d_acc, d_bal = np.empty(N_BOOT), np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, k, k)
        tp_, tn_, fp_, fn_ = C[idx].sum(0)
        tot = tp_ + tn_ + fp_ + fn_
        a_ = (tp_ + tn_) / tot
        dec_ = (tp_ + fn_) / tot
        d_acc[b] = a_ - max(dec_, 1 - dec_)
        d_bal[b] = 0.5 * (tp_ / (tp_ + fn_) + tn_ / (tn_ + fp_)) - 0.5
    q = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    return {"n_pairs": n, "n_sources": int(k), "accuracy": acc, "accuracy_way2": float(acc2),
            "always_decrease": always_dec, "always_decrease_way2": float(dec2),
            "pred_decrease_rate": p_pred, "balanced_accuracy": bal, "independence_expectation": indep,
            "mcc": float(mcc),
            "acc_minus_best_constant": acc - max(always_dec, 1 - always_dec),
            "acc_minus_best_constant_ci95": q(d_acc), "frac_boot_above0_acc": float((d_acc > 0).mean()),
            "balacc_minus_half_ci95": q(d_bal), "frac_boot_above0_bal": float((d_bal > 0).mean()),
            "bootstrap": {"unit": "silenced (source) gene", "method": "percentile, resample sources with replacement, pooled confusion counts", "reps": N_BOOT, "seed": BOOT_SEED},
            "pass": bool(acc < always_dec and abs(acc - acc2) < 1e-12)}


@check("V03", "Deployed 'GroupKFold' summary: 5-fold mean, per-source mean, n sources vs summary text (53.55%, 966)")
def _():
    j = jload(R / "circuit-tracing-217M/outputs/groupkfold_crispri/summary.json")
    txt_lines_5355 = find_lines(S / "circuit-tracing-217M-FINAL_SUMMARY.md", r"53\.55")
    v2_966 = find_lines(A / "audit-20260507-completion-v2.md", r"966")
    folds = None
    for key in ("fold_accuracies", "groupkfold_fold_accuracies", "folds"):
        if key in j:
            folds = j[key]
    flat = json.dumps(j)
    return {"summary_json": j, "summary_md_lines_with_53.55": txt_lines_5355, "audit_v2_lines_with_966": v2_966,
            "pass": ("0.5347" in flat) and bool(txt_lines_5355)}


@check("V04", "Deployed A2 script trains nothing inside its 'GroupKFold' loop")
def _():
    p = R / "circuit-tracing-217M/scripts/audit_a2_groupkfold_crispri.py"
    src = open(use(p)).read()
    fit_calls = [m.start() for m in re.finditer(r"\.fit\(", src)]
    gkf = find_lines(p, r"GroupKFold|fold")
    boot = find_lines(p, r"bootstrap|n_boot|N_BOOT")
    return {"n_fit_calls": len(fit_calls), "fold_lines": gkf[:20], "bootstrap_lines": boot[:10],
            "pass": len(fit_calls) == 0}


@check("V05", "Circuit edges: zero edges at target layer = source layer + 1; inhibitory share; edge count")
def _():
    p = R / "circuit-tracing-217M/outputs/circuit_edges.csv"
    use(p)
    counts = {}
    n = 0; n_inh = 0
    for ch in pd.read_csv(p, usecols=["src_layer", "tgt_layer", "sign"], chunksize=500_000):
        n += len(ch); n_inh += int((ch["sign"] == "inhibitory").sum())
        for (s, t), c in ch.groupby(["src_layer", "tgt_layer"]).size().items():
            counts[f"{s}->{t}"] = counts.get(f"{s}->{t}", 0) + int(c)
    next_layer = {s: counts.get(f"{s}->{s+1}", 0) for s in (0, 3, 6, 9)}
    hook_lines = {"clean_hidden": find_lines(R / "circuit-tracing-217M/scripts/circuit_trace.py", r"hidden_states\[l\]"),
                  "patched": find_lines(R / "circuit-tracing-217M/scripts/circuit_trace.py", r"h_patched = "),
                  "hook_on_layers_src": find_lines(R / "circuit-tracing-217M/scripts/circuit_trace.py", r"layers\[src_l\]\.register_forward_hook")}
    return {"n_edges": n, "inhibitory_share": n_inh / n, "edges_at_src_plus_1": next_layer,
            "edges_by_layer_pair": counts, "code_lines": hook_lines,
            "ratio_vs_52116": n / 52116,
            "summary_22x_lines": find_lines(S / "circuit-tracing-217M-FINAL_SUMMARY.md", r"22× denser"),
            "pass": all(v == 0 for v in next_layer.values())}


# ---------------------------------------------------------------------------
# Exhaustive mapping: block-deletion hook, triplet overwrite, steering
# ---------------------------------------------------------------------------
@check("V06", "Exp 1: every feature has 0 edges at L6; edges per feature nearly constant")
def _():
    d = R / "exhaustive-mapping-217M/outputs/experiment1"
    per = []; l6 = []
    for f in sorted(d.glob("feature_F*.json")):
        j = json.load(open(f)); per.append(j["total_edges"]); l6.append(j["edges_by_layer"].get("6", None))
    use(d / "exhaustive_summary.json")
    per = np.array(per)
    code = {k: find_lines(R / f"exhaustive-mapping-217M/scripts/{k}", r"register_forward_hook|hidden_states\[")
            for k in ("experiment1_exhaustive.py", "experiment2_rerun.py", "experiment3_rerun.py", "experiments_2_3.py")}
    return {"n_feature_files": len(per), "all_L6_zero": all(x == 0 for x in l6), "total_edges": int(per.sum()),
            "mean": float(per.mean()), "sd": float(per.std(ddof=1)), "min": int(per.min()), "max": int(per.max()),
            "hook_and_hidden_state_lines": code,
            "summary_range_claim_lines": find_lines(S / "exhaustive-mapping-217M-FINAL_SUMMARY.md", r"range 0–5,400"),
            "note": "feature files are read in bulk; only the summary JSON is hashed",
            "pass": all(x == 0 for x in l6)}


@check("V07", "Triplets: 4 triplets; 2,980 = targets; ratios follow exactly from ABC=C, AB=B, AC=C, BC=C")
def _():
    out = {}
    for tag in ("experiment2", "experiment2_v2"):
        j = jload(R / f"exhaustive-mapping-217M/outputs/{tag}/combinatorial_summary.json")
        rows = []
        for t in j.get("triplets", []):
            ab, bc = t.get("pairwise_ratio_AB"), t.get("pairwise_ratio_BC")
            if ab is None or bc is None:
                continue
            # C = 1; BC = C/(B+C) -> B; AB = B/(A+B) -> A
            B = 1 / bc - 1; A_ = B / ab - B
            rows.append({"triplet": t["triplet"], "A:B:C": [round(A_, 3), round(B, 3), 1.0],
                         "AC_obs": t.get("pairwise_ratio_AC"), "AC_pred": round(1 / (A_ + 1), 4),
                         "threeway_obs": t.get("threeway_ratio"), "threeway_pred": round(1 / (A_ + B + 1), 4),
                         "marg_obs": t.get("marginal_C_given_AB"), "marg_pred": round(1 - B, 4),
                         "n_targets_with_effect": t.get("n_targets_with_effect"), "n_superadditive": t.get("n_superadditive")})
        out[tag] = {"n_triplets": j.get("n_triplets"), "rows": rows}
    v2 = out["experiment2_v2"]["rows"]
    ok = all(abs(r["AC_obs"] - r["AC_pred"]) <= 2e-4 and abs(r["threeway_obs"] - r["threeway_pred"]) <= 2e-4
             and abs(r["marg_obs"] - r["marg_pred"]) <= 2e-4 for r in v2)
    out["summary_not_a_code_bug_line"] = find_lines(S / "exhaustive-mapping-217M-FINAL_SUMMARY.md", r"not a code bug")
    out["pass"] = ok and out["experiment2_v2"]["n_triplets"] == 4
    return out


@check("V08", "Steering: delta_s identical across 3 features per layer and across alpha 2 vs 5")
def _():
    j = jload(R / "exhaustive-mapping-217M/outputs/experiment3/steering_summary.json")
    a5 = j["per_layer_summary_alpha5"]; a2 = j["per_layer_summary_alpha2"]
    per = {}
    for f in j["features"]:
        L = f"L{f['layer']}"
        per.setdefault(L, []).append({"feature": f["feature"], "direction": f.get("direction"),
                                      "ds_a5": f["alpha_results"]["5"]["mean_delta_s"],
                                      "ds_a2": f["alpha_results"]["2"]["mean_delta_s"],
                                      "top_gene_a5": f["alpha_results"]["5"]["top_upregulated"][0]["gene"]})
    ratio = {L: a5[L]["mean_delta_s"] / a2[L]["mean_delta_s"] for L in a5}
    spread = {L: float(np.ptp([x["ds_a5"] for x in v])) for L, v in per.items()}
    sig = np.load(use(R / "exhaustive-mapping-217M/outputs/experiment3/state_signatures.npz"))
    keys = list(sig.keys())
    cos = None
    if "g_early_logits" in keys and "g_late_logits" in keys:
        a, b = sig["g_early_logits"].ravel().astype(float), sig["g_late_logits"].ravel().astype(float)
        cos = float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    # file-time window of the steering outputs (summary says "~4 min exp3")
    d3 = R / "exhaustive-mapping-217M/outputs/experiment3"
    mt = sorted(p.stat().st_mtime for p in d3.iterdir() if not p.name.startswith("._"))
    window_min = (mt[-1] - mt[0]) / 60 if mt else None
    return {"alpha5_over_alpha2": ratio, "per_feature": per, "max_spread_within_layer_a5": spread,
            "L0_over_L3": a5["L0"]["mean_delta_s"] / a5["L3"]["mean_delta_s"],
            "L0_over_L11": a5["L0"]["mean_delta_s"] / a5["L11"]["mean_delta_s"],
            "signature_keys": keys, "cos_g_early_g_late": cos,
            "summary_0.999_line": find_lines(S / "exhaustive-mapping-217M-FINAL_SUMMARY.md", r"≈ 0\.999"),
            "experiment3_file_time_window_minutes": window_min,
            "summary_4min_line": find_lines(S / "exhaustive-mapping-217M-FINAL_SUMMARY.md", r"~4 min exp3"),
            "pass": all(abs(r - 1) < 0.05 for r in ratio.values())}


# ---------------------------------------------------------------------------
# Attention-GRN: curveball null, verdict inconsistencies
# ---------------------------------------------------------------------------
def _extract_func(src_path: Path, name: str):
    tree = ast.parse(open(use(src_path)).read())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            mod = ast.Module(body=[node], type_ignores=[])
            ns = {"np": np}
            exec(compile(mod, str(src_path), "exec"), ns)
            return ns[name], node.lineno
    raise KeyError(name)


@check("V09", "Curveball null: count draws identical to the observed TRRUST matrix (exact deployed function and seeds)")
def _():
    script = R / "attention-grn-217M/scripts/phase3_residualization.py"
    cb, lineno = _extract_func(script, "curveball_permute")
    tr = pd.read_csv(use(TRRUST), sep="\t", header=None, names=["tf", "target", "mode", "pmid"])
    tr["tf"] = tr["tf"].str.upper(); tr["target"] = tr["target"].str.upper()
    out = {"function_line": lineno, "n_iter_setting_lines": find_lines(script, r"PHASE3_CURVEBALL_N")}
    for suffix in ("", "_rpe1", "_adamson", "_k562_1b"):
        gf = pd.read_csv(use(R / f"attention-grn-217M/outputs/phase0{suffix}/gene_features.csv"))
        sym = [s.upper() for s in gf["symbol"]]
        idx = {s: i for i, s in enumerate(sym)}
        n = len(sym)
        M = np.zeros((n, n), dtype=np.int8)
        for a, b in zip(tr["tf"], tr["target"]):
            i, j = idx.get(a), idx.get(b)
            if i is not None and j is not None and i != j:
                M[i, j] = 1
        n_edges = int(M.sum())
        tf_rows = [i for i in range(n) if M[i].sum() >= 3 and sym[i] in set(tr["tf"])]
        ident_all = ident_tf = 0
        for trial in range(50):
            P = cb(M, n_iter=5 * n_edges, seed=42 + trial)
            ident_all += int(np.array_equal(P, M))
            ident_tf += int(np.array_equal(P[tf_rows], M[tf_rows]))
        nr = jload(R / f"attention-grn-217M/outputs/phase3{suffix}/null_results.json")
        out[suffix or "_k562"] = {"n_trrust_edges": n_edges, "n_tf_rows_scored": len(tf_rows),
                                  "identical_full_matrix": ident_all, "identical_on_scored_rows": ident_tf,
                                  "reported_curveball": nr["curveball"]}
    out["pass"] = out["_k562"]["identical_on_scored_rows"] >= 40
    return out


@check("V10", "Attention verdict files vs FINAL_SUMMARY text (1B C3, Adamson C3, RPE1 baselines, p-values, 5x, CRISPRa)")
def _():
    v = {}
    for tag, f in (("k562", "phase12_verdict.json"), ("rpe1", "phase12_verdict_rpe1.json"),
                   ("adamson", "phase12_verdict_adamson.json"), ("k562_1b", "phase12_verdict_k562_1b.json")):
        j = jload(R / f"attention-grn-217M/outputs/{f}")
        c = j["criteria"]
        v[tag] = {"C3_pass": c["C3_residualized_retains_signal"]["pass"],
                  "C3_fraction": c["C3_residualized_retains_signal"].get("fraction_above_chance_retained"),
                  "C1_beats": [(b["baseline"], b["attention_won"], b["wilcoxon_p_bh"]) for b in c["C1_attention_beats_trivial_baselines"]["details"]["beat_baselines"]]}
    fs = S / "attention-grn-217M-FINAL_SUMMARY.md"
    lines = {"row_1b_C3_FAIL": find_lines(fs, r"^\| k562_1b .*FAIL \| FAIL \| FAIL"),
             "collapses_every_run": find_lines(fs, r"collapses to chance after OLS cross-fit in every run"),
             "no_run_beats_any": find_lines(fs, r"No run's attention beats any"),
             "pvals_2e-52_5e-11": find_lines(fs, r"2e-52|5e-11"),
             "five_x": find_lines(fs, r"5× model-size"),
             "crispra": find_lines(fs, r"CRISPRa"),
             "loader_crispra": find_lines(MT / "setup/dataset_loader.py", r"CRISPRa"),
             "phase0_any_model_name": find_lines(R / "attention-grn-217M/scripts/phase0_any.py", r'"model": "MaxToki-217M-HF"'),
             "rpe1_report_k562_scope": find_lines(S / "attention-grn-217M-rpe1-report.md", r"K562 non-targeting controls only"),
             "adamson_report_k562_scope": find_lines(S / "attention-grn-217M-adamson-report.md", r"K562 non-targeting controls only")}
    cfg1b = jload(R / "attention-grn-217M/outputs/phase0_k562_1b/run_config.json")
    size_217 = 217; size_1b = 1000
    return {"verdicts": v, "summary_lines": lines, "phase0_1b_config": {k: cfg1b.get(k) for k in ("n_ctrl", "N_ctrl", "n_cells", "model", "device") if k in cfg1b},
            "pass": v["k562_1b"]["C3_pass"] is True and bool(lines["row_1b_C3_FAIL"])}


# ---------------------------------------------------------------------------
# SAE atlas: Phase 8t, A8 re-run, Phase 6
# ---------------------------------------------------------------------------
@check("V11", "Phase 8t: TFs with any responding feature; alphabetical TF choice; layer mismatch; ~5 control cells")
def _():
    pr = pd.read_csv(use(R / "sae-atlas-217M/outputs/phase8_true/perturbation_response.csv"))
    tfs = pr[pr["is_tf"] == True]["target"].tolist()
    detected = pr[(pr["is_tf"] == True) & (pr["n_responding"] > 0)][["target", "n_responding"]].values.tolist()
    rp = R / "sae-atlas-217M/scripts/remaining_phases.py"
    a8 = R / "sae-atlas-217M/scripts/audit_a8_rerun_phase8t_targeted.py"
    per_tf = jload(R / "sae-atlas-217M/outputs/phase8t_positive_tf_rerun/per_tf_response.json")
    swn = pd.read_csv(use(R / "sae-atlas-217M/outputs/phase8t_positive_tf_rerun/specificity_with_null.csv"))
    bad_above = swn[(swn["above_random"] == True) & (swn["p_null_specific_>=1"] > 0.05)][["ground_truth", "threshold", "p_null_specific_>=1"]].values.tolist()
    return {"n_tfs": len(tfs), "tfs_sorted_alphabetically": tfs == sorted(tfs), "first_last": [tfs[0], tfs[-1]],
            "tfs_with_responding_features": detected,
            "ctrl_from_phase0_layer_file_line": find_lines(rp, r"layer_\{PROBE_LAYER:02d\}_activations\.npy"),
            "perturbed_hidden_plus1_line": find_lines(rp, r"hidden\[PROBE_LAYER \+ 1\]"),
            "ctrl_first_10000_rows_line": find_lines(rp, r"h_ctrl\[:min\(10000"),
            "a8_hidden_plus1_lines": find_lines(a8, r"PROBE_LAYER \+ 1"),
            "a8_ctrl_10000_lines": find_lines(a8, r"10000|10_000"),
            "gata1_effects": per_tf["GATA1"]["feature_stats"],
            "above_random_true_with_p_gt_0.05": bad_above,
            "sae_trained_on_hidden_layer_idx_line": find_lines(R / "sae-atlas-217M/scripts/full_12layer_pipeline.py", r"hl = hidden\[layer_idx\]"),
            "pass": len(detected) == 3 and bool(bad_above)}


@check("V12", "Phase 6 patching applies the layer-5 SAE to hidden_states[6]")
def _():
    p = R / "sae-atlas-217M/scripts/phase6_patching.py"
    return {"sae_layer_line": find_lines(p, r"PROBE_LAYER = 5"),
            "hidden_plus1_line": find_lines(p, r"hidden_states\[PROBE_LAYER \+ 1\]"),
            "sae_ckpt_line": find_lines(p, r"layer_\{PROBE_LAYER:02d\}/sae_final\.pt"),
            "sae_training_extract_line": find_lines(R / "sae-atlas-217M/scripts/phase0_extract_positions.py", r"hidden\[li\]"),
            "pass": bool(find_lines(p, r"hidden_states\[PROBE_LAYER \+ 1\]"))}


# ---------------------------------------------------------------------------
# Manifold discovery
# ---------------------------------------------------------------------------
@check("V13", "Manifold: five frozen gates vs four computed; permutation gate never computed; null passes two gates")
def _():
    spec = jload(R / "manifold-discovery-217M/reports/quality_gates_spec.json")
    qg = jload(R / "manifold-discovery-217M/reports/quality_gates_let_anchor.json")
    scripts = sorted((R / "manifold-discovery-217M/scripts").glob("*.py"))
    perm_gate_hits = []
    for s in scripts:
        if s.name.startswith("audit_"):
            continue
        for ln in find_lines(s, r"blocked_permutation|n_permutations|permutation_p"):
            perm_gate_hits.append(f"{s.name}:{ln}")
    nul = qg["null_shuffled"]
    rp = R / "manifold-discovery-217M/planning/research_plan.md"
    return {"gates_frozen": spec["gates"], "promotion_rules": spec.get("promotion_rules"),
            "verdict": qg.get("verdict"), "null": nul,
            "null_passes": {k: nul[k] >= 0.2 for k in ("random_holdout", "donor_holdout", "branch_holdout")} | {"trust": nul["trustworthiness"] >= 0.8},
            "scripts_computing_permutation_gate": perm_gate_hits,
            "research_plan_fail_all_four_line": find_lines(rp, r"must FAIL all four"),
            "research_plan_fail_at_least_one_line": find_lines(rp, r"must fail at least one"),
            "pass": not perm_gate_hits}


@check("V14", "Manifold: lung control trust < 0.80; zero-shot donors inside external donors; H103 zero-shot k=5; bootstrap method")
def _():
    lung = jload(R / "manifold-discovery-217M/reports/external_validation_lung_nonhema.json")
    ext = pd.read_csv(use(R / "manifold-discovery-217M/artifacts/anchors/anchor_meta_external.csv"))
    zs = pd.read_csv(use(R / "manifold-discovery-217M/artifacts/anchors/anchor_meta_zeroshot.csv"))
    inn = pd.read_csv(use(R / "manifold-discovery-217M/artifacts/anchors/anchor_meta_internal.csv"))
    zsd, exd, ind = set(zs.donor_id), set(ext.donor_id), set(inn.donor_id)
    tis_shared = len(set(zs.tissue) & set(ext.tissue))
    h103 = jload(R / "manifold-discovery-217M/reports/zeroshot_h103.json")
    boot = jload(R / "manifold-discovery-217M/reports/external_validation_external_bootstrap.json")
    fs = R / "manifold-discovery-217M/FINAL_SUMMARY.md"
    ss = S / "manifold-discovery-217M-FINAL_SUMMARY.md"
    return {"lung_trust": lung["trustworthiness"], "lung_trust_below_0.80": lung["trustworthiness"] < 0.8,
            "summary_trust_passes_weakly_lines": {"summaries": find_lines(ss, r"trust passes weakly"), "runs": find_lines(fs, r"trust passes weakly")},
            "summary_fails_3_4_lines": {"summaries": find_lines(ss, r"fails 3/4"), "runs": find_lines(fs, r"fails 3/4")},
            "n_zs_donors": len(zsd), "n_ext_donors": len(exd), "zs_subset_of_ext": zsd <= exd,
            "zs_internal_overlap": len(zsd & ind), "zs_tissues": len(set(zs.tissue)), "zs_tissues_shared_with_ext": tis_shared,
            "summary_disjoint_lines": {"summaries": find_lines(ss, r"Disjoint from both internal and external|12 disjoint donors"),
                                       "runs": find_lines(fs, r"Disjoint from both internal and external|12 disjoint donors")},
            "internal_anchor_count_by_donor_top": inn.donor_id.value_counts().head(3).to_dict(),
            "h103_zeroshot": h103, "h103_summary_7_donors_lines": find_lines(ss, r"18 B-lineage anchors / 7 donors"),
            "bootstrap_method": {k: boot[k] for k in ("n_bootstrap", "bootstrap_method", "subsample_size")},
            "pass": lung["trustworthiness"] < 0.8 and zsd <= exd}


# ---------------------------------------------------------------------------
# Spectral geometry
# ---------------------------------------------------------------------------
@check("V15", "Spectral: CKA at L0 equals 1; 'disjoint' samples overlap; Phase 9b CI uses 80% subsample without replacement")
def _():
    cka = pd.read_csv(use(R / "spectral-geometry-217M/outputs/phase8/cka_per_layer.csv"))
    l0 = cka[cka.layer == 0].iloc[0].to_dict()
    import h5py
    with h5py.File(TS_IMMUNE, "r") as f:
        obs = f["obs"]
        idx_name = obs.attrs.get("_index", "_index")
        if isinstance(idx_name, bytes):
            idx_name = idx_name.decode()
        n_total = int(obs[idx_name].shape[0])
    INPUTS[str(TS_IMMUNE)] = "not-hashed(19.8GB; only obs index length read)"
    samples = {}
    for s in (42, 43, 44):
        samples[s] = np.sort(np.random.default_rng(s).choice(n_total, size=2000, replace=False))
    ov = {f"{a}-{b}": int(len(np.intersect1d(samples[a], samples[b]))) for a, b in ((42, 43), (42, 44), (43, 44))}
    ov2 = {f"{a}-{b}": len(set(samples[a].tolist()) & set(samples[b].tolist())) for a, b in ((42, 43), (42, 44), (43, 44))}
    bs = R / "spectral-geometry-217M/scripts/audit_a3_bootstrap_cross_model_pearson.py"
    return {"cka_L0": l0, "n_total_cells": n_total, "overlap_intersect1d": ov, "overlap_sets": ov2,
            "sample_code_lines": {"phase0": find_lines(R / "spectral-geometry-217M/scripts/phase0_extract.py", r"rng_sample\.choice"),
                                  "phase8": find_lines(R / "spectral-geometry-217M/scripts/phase8_stability.py", r"rng\.choice\(n_total")},
            "summary_disjoint_lines": find_lines(S / "spectral-geometry-217M-FINAL_SUMMARY.md", r"3 disjoint cell samples"),
            "phase9b_docstring_with_replacement": find_lines(bs, r"with replacement"),
            "phase9b_replace_false": find_lines(bs, r"replace=False"),
            "feature_shuffle_null_lines": find_lines(R / "spectral-geometry-217M/scripts/phase1_svd.py", r"shuffle|permut"),
            "pass": l0["cka_mean"] == 1.0 and all(v > 0 for v in ov.values())}


@check("V16", "Spectral autoloop ran Claude Sonnet; a timed-out executor (rc=124) passed validation; stop rule counts crashes")
def _():
    log = R / "spectral-geometry-217M/autoloop/runtime/driver.log"
    return {"model_line": find_lines(log, r"model=sonnet"), "rc124_valid_true": find_lines(log, r"rc=124 .*valid=True"),
            "abort_line": find_lines(log, r"2 consecutive executor failures"), "pass": bool(find_lines(log, r"rc=124 .*valid=True"))}


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------
@check("V17", "Topology: iter_04 replication verdict text vs numbers; synthetic check ~0; dual-axis AUROC below 0.5; N_NULL=8; manual relabels")
def _():
    j = jload(R / "topology-141-217M/outputs/phase11_autoloop/iter_04_replication/summary.json")
    flat = json.dumps(j)
    by = pd.read_csv(use(R / "topology-141-217M/outputs/groupkfold_h123/by_layer.csv"))
    auto = jload(R / "topology-141-217M/outputs/phase11_autoloop/autoloop_summary.json")
    return {"aggregate_means": j.get("aggregate_means_across_3_domains"), "synthetic": j.get("synthetic_positive_control"),
            "verdict_says_all_above_0.6": "above 0.6" in flat,
            "dual_axis": {"min": float(by.auc_dual_axis_disjoint.min()), "max": float(by.auc_dual_axis_disjoint.max()),
                           "n_layers_below_0.5": int((by.auc_dual_axis_disjoint < 0.5).sum()), "n_test_pairs": sorted(set(by.n_test_pairs_dual))},
            "phase10_n_null_line": find_lines(R / "topology-141-217M/scripts/phase10_h139_sectional_anisotropy.py", r"N_NULL\s*=\s*8"),
            "autoloop_code_decisions": json.dumps(auto)[:600],
            "manual_review_novel_positive_lines": find_lines(R / "topology-141-217M/outputs/phase11_autoloop/MANUAL_REVIEW.md", r"NOVEL POSITIVE"),
            "stop_rule_lines": find_lines(R / "topology-141-217M/scripts/phase11_autoloop.py", r"consecutive"),
            "pass": ("above 0.6" in flat) and float(by.auc_dual_axis_disjoint.max()) < 0.5}


# ---------------------------------------------------------------------------
# Specs, hardware, audit arithmetic
# ---------------------------------------------------------------------------
@check("V18", "Deployed specs: section headings vs the ten-section template")
def _():
    specs = ["attention-grn-extraction-and-evaluation.md", "residual-stream-spectral-geometry.md",
             "topology-geometry-141-hypotheses.md", "manifold-discovery-extraction-compactification.md",
             "longevity-mechinterp-donor-aware.md", "sparse-autoencoders/01-sae-atlas.md",
             "sparse-autoencoders/02-causal-circuit-tracing.md", "sparse-autoencoders/03-exhaustive-mapping-and-steering.md"]
    out = {}
    for s in specs:
        p = PL / s
        heads = [l.strip() for l in open(use(p), encoding="utf-8") if l.startswith("## ")]
        out[s] = {"has_code_references": any("code reference" in h.lower() for h in heads), "n_h2": len(heads)}
    n_missing = sum(1 for v in out.values() if not v["has_code_references"])
    curve_spec = find_lines(PL / "attention-grn-extraction-and-evaluation.md", r"200 permuted GRNs|degree_null_n_curveball")
    return {"specs": out, "n_missing_code_references": n_missing, "spec_curveball_200_lines": curve_spec, "pass": n_missing == 7}


@check("V19", "Hardware: device strings in run logs and configs (mps vs cuda/A100)")
def _():
    hits = {"mps": 0, "cuda": 0, "a100": 0}
    files = 0
    for p in list(R.rglob("*.log")) + list(R.rglob("run_config.json")):
        if "node_modules" in str(p) or p.name.startswith("._"):
            continue
        try:
            t = open(p, encoding="utf-8", errors="replace").read().lower()
        except Exception:
            continue
        files += 1
        hits["mps"] += int("mps" in t)
        hits["cuda"] += int("cuda" in t)
        hits["a100"] += int("a100" in t)
    return {"files_scanned": files, "files_with_term": hits, "pass": hits["a100"] == 0 and hits["mps"] > 0}


@check("V20", "Audit v2 matrix: only two lines give v1/v2 cell counts; v1 split not an integer count")
def _():
    v2 = A / "audit-20260507-completion-v2.md"
    lines = find_lines(v2, r"v[012] \(")
    fits = []
    for c in range(81):
        for p_ in range(81 - c):
            u = 80 - c - p_
            if round(100 * c / 80) == 68 and round(100 * p_ / 80) == 26 and round(100 * u / 80) == 6:
                fits.append((c, p_, u))
    return {"v2_matrix_lines": lines, "integer_splits_matching_68_26_6": fits,
            "v0_line": find_lines(v2, r"49 / 23 / 8"), "v2_line": find_lines(v2, r"60 / 18 / 2"),
            "zero_untouched_line": find_lines(v2, r"0 untouched"), "pass": len(fits) <= 1}


@check("V21", "Manifold summary copies: stale 'n/t' rows, top-10 factor share, H38 zero-shot fail")
def _():
    fa = jload(R / "manifold-discovery-217M/reports/factor_ablation.json")
    ss = S / "manifold-discovery-217M-FINAL_SUMMARY.md"
    fs = R / "manifold-discovery-217M/FINAL_SUMMARY.md"
    h38zs = jload(R / "manifold-discovery-217M/reports/zeroshot_H38_lite.json")
    return {"top4": fa["top4_pct_of_total_impact"], "top10": fa["top10_pct_of_total_impact"],
            "summary_top10_33pct_lines": find_lines(ss, r"Top-10 factors \| ~33%"),
            "summaries_copy_nt_rows": find_lines(ss, r"H38 LITE signalling \| 0\.814 \| n/t"),
            "runs_copy_h38_backfill_rows": find_lines(fs, r"0\.787 \(3/4 gates"),
            "h38_zeroshot_trust": h38zs.get("trustworthiness"),
            "pass": abs(fa["top10_pct_of_total_impact"] - 0.33) > 0.03}


@check("V22", "Longevity summary says every other pipeline used L5; attention-grn primary layer is L8")
def _():
    lf = R / "longevity-mechinterp-217M/FINAL_SUMMARY.md"
    return {"longevity_every_pipeline_L5_lines": find_lines(lf, r"used\s*$|canonical mid-stack tap used"),
            "attention_primary_layer_default_8": find_lines(R / "attention-grn-217M/scripts/phase3_residualization.py", r'"PHASE1_PRIMARY_LAYER", "8"'),
            "pass": bool(find_lines(R / "attention-grn-217M/scripts/phase3_residualization.py", r'"PHASE1_PRIMARY_LAYER", "8"'))}


@check("V23", "Circuit Phase 11 original record set used labelled edges only; corrected set used all edges")
def _():
    rp = R / "circuit-tracing-217M/scripts/remaining_phases.py"
    a2 = R / "circuit-tracing-217M/scripts/audit_a2_groupkfold_crispri.py"
    return {"has_both_labels_lines": find_lines(rp, r"has_both_labels"),
            "tmp_atlas_path_lines": find_lines(rp, r"<TMP>/maxtoki-atlas-data-12"),
            "a2_edge_filter_lines": find_lines(a2, r"top20_genes|feature_catalog"),
            "pass": bool(find_lines(rp, r"has_both_labels"))}


if __name__ == "__main__":
    t0 = time.time()
    payload = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "python": sys.version.split()[0],
               "numpy": np.__version__, "pandas": pd.__version__, "boot_seed": BOOT_SEED,
               "checks": results, "inputs_sha256": INPUTS, "wall_seconds": round(time.time() - t0, 1)}
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=1, default=str)
    print("wrote", OUT)
