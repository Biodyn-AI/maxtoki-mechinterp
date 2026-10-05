"""v2 step 4 — what the deployed 'incremental delta AUROC' measured (knockdown endpoint).

Deployed (phase2_incremental_value.py): one row per (perturbation, candidate gene) pair,
label = responding gene (same DE rule as Phase 1). Logistic regression (StandardScaler +
LogisticRegression(class_weight='balanced', max_iter=2000, lbfgs)) under 5-fold
GroupKFold grouped by perturbation ('cross_pert'). Metric = mean over the 5 held-out folds
of the AUROC on pooled held-out pairs. Delta = mean AUROC(gene features + attention)
minus mean AUROC(gene features only). Gene features = candidate gene's mean, variance,
dropout in control cells. No CI was computed; GBDT was not run (has_gbdt=false).

Here: rebuild the pair table from the re-derived labels, reproduce the deployed means,
and add a 95% CI for the delta by bootstrap over perturbations of the pooled
out-of-fold AUROC difference (the fitted fold models are held fixed).
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
                       load_attention_layer, percentile_ci, save_json, WeightedAUROC)
import json  # noqa: E402

OUTD = V2 / "knockdown"


def mk():
    return Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"))])


def main():
    res = {}
    for ri, run in enumerate(RUNS):
        part = OUTD / f"incremental_{run}.json"
        if part.exists():
            res[run] = json.load(open(part))
            print(run, "cached")
            continue
        gf = load_gene_features(run)
        G = len(gf)
        A = load_attention_layer(run); np.fill_diagonal(A, 0.0)
        d = np.load(V2 / "de_labels" / f"{run}.npz")
        hs, Y = d["pert_hvg_idx"], d["labels"]
        pid, tgt, lab, att = [], [], [], []
        for k, (h, y) in enumerate(zip(hs, Y)):
            t = np.delete(np.arange(G), h)
            pid.append(np.full(len(t), k)); tgt.append(t); lab.append(y[t]); att.append(A[h, t])
        pid = np.concatenate(pid); tgt = np.concatenate(tgt)
        y = np.concatenate(lab).astype(np.int64); att = np.concatenate(att)
        gm = gf["mean_expr"].to_numpy(np.float32)[tgt]
        gv = gf["variance"].to_numpy(np.float32)[tgt]
        gd = gf["dropout_rate"].to_numpy(np.float32)[tgt]
        X_gene = np.column_stack([gm, gv, gd]).astype(np.float32)
        X_both = np.column_stack([gm, gv, gd, att]).astype(np.float32)

        # check against the deployed pair table
        dep = pd.read_csv(phase_dir("phase2", run) / "pair_dataset.csv",
                          usecols=["pert_idx", "target_hvg_idx", "label", "attention_edge"])
        same_rows = (len(dep) == len(y)) and np.array_equal(dep["pert_idx"].to_numpy(), pid) \
            and np.array_equal(dep["target_hvg_idx"].to_numpy(), tgt)
        lab_equal = bool(same_rows and np.array_equal(dep["label"].to_numpy(), y))
        att_maxdiff = float(np.max(np.abs(dep["attention_edge"].to_numpy() - att))) if same_rows else float("nan")

        folds = list(GroupKFold(n_splits=5).split(X_gene, y, groups=pid))
        oof = {"gene_only": np.zeros(len(y)), "gene_plus_attn": np.zeros(len(y))}
        fold_auc = {"gene_only": [], "gene_plus_attn": []}
        for tr, te in folds:
            for name, X in [("gene_only", X_gene), ("gene_plus_attn", X_both)]:
                m = mk().fit(X[tr], y[tr])
                p = m.predict_proba(X[te])[:, 1]
                oof[name][te] = p
                fold_auc[name].append(float(roc_auc_score(y[te], p)))
        mean_go = float(np.mean(fold_auc["gene_only"])); mean_ga = float(np.mean(fold_auc["gene_plus_attn"]))
        dsum = pd.read_csv(phase_dir("phase2", run) / "delta_auroc_summary.csv")
        drow = dsum[(dsum.split == "cross_pert") & (dsum.model == "logreg") & (dsum.feature_set == "gene_plus_attn")].iloc[0]

        pooled_go = float(roc_auc_score(y, oof["gene_only"]))
        pooled_ga = float(roc_auc_score(y, oof["gene_plus_attn"]))
        # bootstrap over perturbations: integer weights = times each perturbation is drawn
        rng = np.random.default_rng(SEED + 400 + ri)
        n_p = len(hs)
        f_ga = WeightedAUROC(y, oof["gene_plus_attn"]); f_go = WeightedAUROC(y, oof["gene_only"])
        assert abs(f_ga() - pooled_ga) < 1e-9 and abs(f_go() - pooled_go) < 1e-9
        deltas = []
        for _ in range(N_BOOT):
            w = np.bincount(rng.integers(0, n_p, n_p), minlength=n_p)[pid]
            deltas.append(f_ga(w) - f_go(w))
        deltas = np.array(deltas)
        res[run] = {
            "n_pairs": int(len(y)), "n_perturbations": int(n_p), "positive_rate": float(y.mean()),
            "pair_table_rows_identical_to_deployed": bool(same_rows),
            "labels_identical_to_deployed": lab_equal,
            "attention_max_abs_diff_vs_deployed": att_maxdiff,
            "fold_mean_auroc_gene_only": mean_go, "fold_mean_auroc_gene_plus_attn": mean_ga,
            "fold_mean_delta": mean_ga - mean_go,
            "deployed_gene_only": float(drow["gene_only_auroc"]),
            "deployed_gene_plus_attn": float(drow["new_auroc"]),
            "deployed_delta": float(drow["delta_auroc"]),
            "per_fold_delta": list(np.array(fold_auc["gene_plus_attn"]) - np.array(fold_auc["gene_only"])),
            "pooled_oof_auroc_gene_only": pooled_go, "pooled_oof_auroc_gene_plus_attn": pooled_ga,
            "pooled_oof_delta": pooled_ga - pooled_go,
            "pooled_oof_delta_ci95": percentile_ci(deltas),
        }
        save_json(res[run], part)
        print(run, res[run], flush=True)
    res["_method"] = ("cross-perturbation 5-fold GroupKFold logistic regression; CI = percentile bootstrap "
                      f"over perturbations ({N_BOOT} reps) of the pooled out-of-fold AUROC difference, "
                      "fold models held fixed")
    save_json(res, OUTD / "incremental_value_check.json")


if __name__ == "__main__":
    main()
