"""v2 step 3 — Endpoint A: knockdown-response (per-perturbation) evaluation.

For each run and each evaluated perturbation p:
  positives = genes called responding by the DE rule (see v2_02_rederive_de.py);
  negatives = the other genes of the run's 1,500 (perturbed gene removed);
  attention score of candidate g = attn_L8[p, g] (row = perturbed gene as query);
  gene-variance score of g = variance of g's log1p(CP10k) in the run's Phase-0 control
  cells (same for every perturbation; no model involved).
Summary = mean over perturbations of the per-perturbation AUROC.

Two routes to the per-perturbation AUROCs:
  route 1: deployed file outputs/phase1*/per_perturbation_auroc.csv;
  route 2: re-derived labels (v2_02) + own tie-safe rank AUROC.
They must agree. Intervals: percentile bootstrap over perturbations (2,000 reps).
Test: the one the deployed run used — paired Wilcoxon signed-rank on the
per-perturbation AUROCs (scipy, zero_method="zsplit", two-sided), BH over the
13 tests of that run's table (phase1_trivial_baselines.py:297-349). Also BH across the
4 runs for the variance-vs-attention test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v2_common import (RUNS, V2, N_BOOT, SEED, phase_dir, load_gene_features,  # noqa: E402
                       load_attention_layer, auroc_rank, percentile_ci, bh, save_json,
                       tie_self_test)

OUTD = V2 / "knockdown"
OUTD.mkdir(parents=True, exist_ok=True)

# the 13-test family of the deployed Phase-1 table, in deployed order
FAMILY = [(b, r) for b in ["gene_variance", "gene_mean_expr", "gene_one_minus_dropout",
                           "gene_tf_out_degree"]
          for r in ["attention_primary", "attention_primary_norm", "spearman"]] + \
         [("attention_primary", "spearman")]


def wilcox_p(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(stats.wilcoxon(a[m], b[m], zero_method="zsplit").pvalue)


def main():
    res = {"tie_self_test": tie_self_test(), "runs": {}}
    raw_p_var_attn = {}
    for ri, run in enumerate(RUNS):
        gf = load_gene_features(run)
        G = len(gf)
        A = load_attention_layer(run)
        np.fill_diagonal(A, 0.0)
        gvar = gf["variance"].to_numpy(np.float64)
        d = np.load(V2 / "de_labels" / f"{run}.npz")
        syms, hs, Y = d["pert_symbol"], d["pert_hvg_idx"], d["labels"]
        rows = []
        for s, h, y in zip(syms, hs, Y):
            valid = np.ones(G, bool); valid[h] = False
            yy = y[valid]
            a_att = auroc_rank(yy, A[h][valid])
            a_var = auroc_rank(yy, gvar[valid])
            rows.append({"pert_symbol": str(s), "n_pos": int(yy.sum()), "n_neg": int((~yy).sum()),
                         "auc_attention": a_att, "auc_variance": a_var,
                         "auc_attention_sklearn": float(roc_auc_score(yy, A[h][valid])),
                         "auc_variance_sklearn": float(roc_auc_score(yy, gvar[valid]))})
        mine = pd.DataFrame(rows)
        mine.to_csv(OUTD / f"per_perturbation_{run}.csv", index=False)

        dep = pd.read_csv(phase_dir("phase1", run) / "per_perturbation_auroc.csv")
        mg = dep.merge(mine, on="pert_symbol", how="outer", indicator=True)
        both = mg[mg["_merge"] == "both"]
        check = {
            "n_deployed": int(len(dep)), "n_rederived": int(len(mine)), "n_matched": int(len(both)),
            "n_positive_counts_equal": int((both["n_de_positive"] == both["n_pos"]).sum()),
            "max_abs_diff_attention": float((both["auc_attention_primary"] - both["auc_attention"]).abs().max()),
            "max_abs_diff_variance": float((both["auc_gene_variance"] - both["auc_variance"]).abs().max()),
            "max_abs_diff_rank_vs_sklearn": float(max((mine["auc_attention"] - mine["auc_attention_sklearn"]).abs().max(),
                                                      (mine["auc_variance"] - mine["auc_variance_sklearn"]).abs().max())),
        }
        # labels vs the deployed Phase-3 pair file (is_de written by a third copy of the DE code)
        fp = pd.read_csv(phase_dir("phase3", run) / "full_pair_dataset.csv",
                         usecols=["pert_symbol", "target_hvg", "is_de"])
        idx = {str(s): i for i, s in enumerate(syms)}
        pi = fp["pert_symbol"].astype(str).map(idx)
        ok = pi.notna()
        mine_lab = Y[pi[ok].astype(int).to_numpy(), fp.loc[ok, "target_hvg"].to_numpy()]
        check["phase3_pairs"] = int(len(fp))
        check["phase3_pairs_matched"] = int(ok.sum())
        check["phase3_label_disagreements"] = int((mine_lab != fp.loc[ok, "is_de"].to_numpy().astype(bool)).sum())
        check["rederived_pairs_total"] = int(sum(G - 1 for _ in syms))

        # bootstrap over perturbations
        att = mine["auc_attention"].to_numpy(); var = mine["auc_variance"].to_numpy()
        n = len(att)
        rng = np.random.default_rng(SEED + 100 + ri)
        B = rng.integers(0, n, size=(N_BOOT, n))
        b_att = att[B].mean(1); b_var = var[B].mean(1); b_gap = b_var[:] - b_att
        # deployed Wilcoxon family, recomputed from the deployed per-perturbation file
        fam_p = []
        for b, r in FAMILY:
            fam_p.append(wilcox_p(dep[f"auc_{b}"].to_numpy(), dep[f"auc_{r}"].to_numpy()))
        fam_p = np.array(fam_p)
        fam_bh = bh(fam_p)
        dep_w = pd.read_csv(phase_dir("phase1", run) / "wilcoxon_baseline_vs_edges.csv")
        p_mine = wilcox_p(var, att)
        raw_p_var_attn[run] = fam_p[0]
        res["runs"][run] = {
            "checks_vs_deployed": check,
            "n_perturbations": int(n),
            "n_positive_pairs": int(mine["n_pos"].sum()),
            "n_negative_pairs": int(mine["n_neg"].sum()),
            "median_positives_per_perturbation": float(mine["n_pos"].median()),
            "attention_mean_auroc": float(att.mean()),
            "attention_ci95": percentile_ci(b_att),
            "variance_mean_auroc": float(var.mean()),
            "variance_ci95": percentile_ci(b_var),
            "gap_variance_minus_attention": float(var.mean() - att.mean()),
            "gap_ci95": percentile_ci(b_gap),
            "n_perturbations_variance_gt_attention": int((var > att).sum()),
            "wilcoxon_variance_vs_attention": {
                "test": "paired Wilcoxon signed-rank, zero_method='zsplit', two-sided, on per-perturbation AUROCs",
                "p_raw_recomputed_from_deployed_csv": float(fam_p[0]),
                "p_raw_deployed_file": float(dep_w["wilcoxon_p_raw"].iloc[0]),
                "p_bh_within_run_13_tests_recomputed": float(fam_bh[0]),
                "p_bh_within_run_deployed_file": float(dep_w["wilcoxon_p_bh"].iloc[0]),
                "max_abs_diff_family_raw_p": float(np.max(np.abs(fam_p - dep_w["wilcoxon_p_raw"].to_numpy()))),
                "p_raw_from_rederived_aurocs": p_mine,
            },
        }
        print(run, {k: v for k, v in res["runs"][run].items() if k != "checks_vs_deployed"})
        print("  checks:", check)
    runs = list(RUNS)
    across = bh(np.array([raw_p_var_attn[r] for r in runs]))
    for r, q in zip(runs, across):
        res["runs"][r]["wilcoxon_variance_vs_attention"]["p_bh_across_4_runs"] = float(q)
    res["bootstrap"] = {"unit": "perturbation", "method": "percentile, resample perturbations with replacement",
                        "n_reps": N_BOOT, "seed_base": SEED + 100}
    save_json(res, OUTD / "knockdown_summary.json")


if __name__ == "__main__":
    main()
