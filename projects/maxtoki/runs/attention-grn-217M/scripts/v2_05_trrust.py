"""v2 step 5 — Endpoint B: TRRUST curated-edge evaluation with gene-level baselines.

Deployed evaluation (phase3_residualization.py:86-141), reproduced exactly:
  * gene set = the run's 1,500 genes; TRRUST edges with both TF and target in it, TF != target;
  * evaluated TFs = TFs in the gene set with >= 3 such targets;
  * for each evaluated TF, candidates = the other 1,499 genes; positive = TRRUST target;
  * all (TF, candidate) pairs of all evaluated TFs pooled into ONE AUROC.
Scores compared on the same pairs:
  attention      attn_L8[TF, candidate]   (row = TF as query)
  variance       variance of candidate in control cells (one number per gene)
  mean, 1-dropout, |Spearman| (co-expression in the same control cells)
  gene3          logistic regression on candidate mean, variance, dropout; 5-fold
                 GroupKFold by TF; out-of-fold predictions pooled (the fitted fold models
                 are held fixed in the bootstrap)
Intervals: percentile bootstrap over TFs (2,000 reps; a TF drawn k times has all its
pairs weighted k). Secondary view: mean over TFs of the per-TF AUROC.
Supplement: pooled attention AUROC at every stored layer (not used for any choice).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v2_common import (RUNS, V2, N_BOOT, SEED, phase_dir, load_gene_features,  # noqa: E402
                       load_attention_layer, n_layers_stored, load_trrust, trrust_matrix,
                       evaluated_tf_rows, auroc_rank, percentile_ci, save_json, WeightedAUROC)

OUTD = V2 / "trrust"
OUTD.mkdir(parents=True, exist_ok=True)


def pooled_pairs(tf_rows, G):
    tf_of, cand = [], []
    for k, r in enumerate(tf_rows):
        c = np.delete(np.arange(G), r)
        tf_of.append(np.full(len(c), k)); cand.append(c)
    return np.concatenate(tf_of), np.concatenate(cand)


def main():
    trrust = load_trrust()
    res = {}
    per_layer_rows = []
    for ri, run in enumerate(RUNS):
        gf = load_gene_features(run)
        G = len(gf)
        sym = [s.upper() for s in gf["symbol"]]
        E = trrust_matrix(sym, trrust)
        tf_rows = evaluated_tf_rows(sym, trrust, E)
        k_of, cand = pooled_pairs(tf_rows, G)
        src = tf_rows[k_of]
        y = E[src, cand].astype(bool)
        A = load_attention_layer(run); np.fill_diagonal(A, 0.0)
        sp = np.abs(np.load(phase_dir("phase0", run) / "spearman_edges.npy").astype(np.float32))
        np.fill_diagonal(sp, 0.0)
        gm = gf["mean_expr"].to_numpy(np.float64); gv = gf["variance"].to_numpy(np.float64)
        gd = gf["dropout_rate"].to_numpy(np.float64)
        scores = {
            "attention": A[src, cand].astype(np.float64),
            "variance": gv[cand],
            "mean_expr": gm[cand],
            "one_minus_dropout": 1.0 - gd[cand],
            "spearman_abs": sp[src, cand].astype(np.float64),
        }
        # gene3: grouped CV by TF
        X = np.column_stack([gm[cand], gv[cand], gd[cand]]).astype(np.float32)
        oof = np.zeros(len(y)); fold_auc = []
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups=k_of):
            m = Pipeline([("sc", StandardScaler()),
                          ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))]).fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
            if 0 < y[te].sum() < len(te):
                fold_auc.append(float(roc_auc_score(y[te], oof[te])))
        scores["gene3_logreg_oof"] = oof

        pooled = {k: auroc_rank(y, s) for k, s in scores.items()}
        pooled_sk = {k: float(roc_auc_score(y, s)) for k, s in scores.items()}
        # per-TF view
        per_tf = []
        for k, r in enumerate(tf_rows):
            m = k_of == k
            row = {"tf": sym[r], "n_targets": int(y[m].sum())}
            for name, s in scores.items():
                row[name] = auroc_rank(y[m], s[m])
            per_tf.append(row)
        per_tf = pd.DataFrame(per_tf)
        per_tf.to_csv(OUTD / f"per_tf_{run}.csv", index=False)

        # bootstrap over TFs
        nT = len(tf_rows)
        fast = {k: WeightedAUROC(y, s) for k, s in scores.items()}
        for k in scores:
            assert abs(fast[k]() - pooled[k]) < 1e-9
        rng = np.random.default_rng(SEED + 500 + ri)
        boot = {k: [] for k in scores}
        boot_tfmean = {k: [] for k in scores}
        tfvals = {k: per_tf[k].to_numpy() for k in scores}
        for _ in range(N_BOOT):
            draw = rng.integers(0, nT, nT)
            cnt = np.bincount(draw, minlength=nT)
            w = cnt[k_of]
            for k in scores:
                boot[k].append(fast[k](w))
                boot_tfmean[k].append(float(np.nanmean(tfvals[k][draw])))
        boot = {k: np.array(v) for k, v in boot.items()}
        boot_tfmean = {k: np.array(v) for k, v in boot_tfmean.items()}

        out = {
            "n_trrust_edges_in_gene_set": int(E.sum()),
            "n_tfs_evaluated": int(nT), "tfs": [sym[r] for r in tf_rows],
            "n_positive_pairs": int(y.sum()), "n_negative_pairs": int((~y).sum()),
            "n_candidate_genes_with_any_positive": int(len(np.unique(cand[y]))),
            "attention_exact_zero_fraction": float((scores["attention"] == 0).mean()),
            "pooled_auroc": pooled,
            "pooled_auroc_sklearn_check": pooled_sk,
            "pooled_ci95": {k: percentile_ci(v) for k, v in boot.items()},
            "gene3_fold_mean_auroc": float(np.mean(fold_auc)), "gene3_n_valid_folds": len(fold_auc),
            "gaps": {},
            "per_tf_mean_auroc": {k: float(np.nanmean(tfvals[k])) for k in scores},
            "per_tf_mean_ci95": {k: percentile_ci(v) for k, v in boot_tfmean.items()},
            "n_tfs_variance_gt_attention": int((per_tf["variance"] > per_tf["attention"]).sum()),
            "n_tfs_gene3_gt_attention": int((per_tf["gene3_logreg_oof"] > per_tf["attention"]).sum()),
        }
        for b in ["variance", "gene3_logreg_oof", "mean_expr", "one_minus_dropout", "spearman_abs"]:
            g = boot[b] - boot["attention"]
            out["gaps"][f"{b}_minus_attention"] = {
                "point": pooled[b] - pooled["attention"], "ci95": percentile_ci(g),
                "frac_boot_gt0": float((g > 0).mean())}
            gt = boot_tfmean[b] - boot_tfmean["attention"]
            out["gaps"][f"{b}_minus_attention_per_tf_mean"] = {
                "point": out["per_tf_mean_auroc"][b] - out["per_tf_mean_auroc"]["attention"],
                "ci95": percentile_ci(gt)}
        # layer supplement
        L = n_layers_stored(run)
        for li in range(L):
            Al = load_attention_layer(run, li); np.fill_diagonal(Al, 0.0)
            per_layer_rows.append({"run": run, "layer": li, "n_layers": L,
                                   "pooled_trrust_auroc_attention": auroc_rank(y, Al[src, cand])})
        res[run] = out
        print(run, {k: out[k] for k in ["n_tfs_evaluated", "n_positive_pairs", "pooled_auroc", "gene3_fold_mean_auroc"]})
        print("   gaps:", {k: (round(v["point"], 4), [round(x, 4) for x in v["ci95"]]) for k, v in out["gaps"].items()})
    pd.DataFrame(per_layer_rows).to_csv(OUTD / "per_layer_trrust_attention_auroc.csv", index=False)
    res["_method"] = {"bootstrap_unit": "TF (evaluated TF row)", "n_reps": N_BOOT,
                      "ci": "percentile", "seed_base": SEED + 500,
                      "gene3": "logistic regression, StandardScaler, class_weight=balanced; GroupKFold(5) by TF; pooled out-of-fold predictions"}
    save_json(res, OUTD / "trrust_summary.json")


if __name__ == "__main__":
    main()
