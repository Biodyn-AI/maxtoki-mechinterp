"""Independent check of the incremental-value CI for the 1B K562 run (knockdown endpoint),
the only run where v2 reports a CI above zero (+0.0021 [+0.0010, +0.0031]).

Pair table = deployed outputs/phase2_k562_1b/pair_dataset.csv (label, target mean /
variance / dropout, attention_edge), float64.
Route 1: GroupKFold(5) by perturbation, logistic regression (standardised, balanced),
  pooled out-of-fold AUROC difference (gene+attention minus gene only); 95% CI by
  explicit-replication bootstrap over perturbations (2,000 reps, seed 31), sklearn AUROC.
Route 2: refitting noise. 20 random 5-fold splits by perturbation (perturbations
  shuffled with seed 41+i before GroupKFold); pooled OOF delta for each.
Output: outputs/v2_eval/verification/incremental_1B_check.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path("<REPO_ROOT>/projects/maxtoki/runs/attention-grn-217M/outputs")
VER = OUT / "v2_eval" / "verification"


def oof(X, y, groups, folds):
    p = np.zeros(len(y))
    for tr, te in folds:
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        p[te] = m.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def main():
    d = pd.read_csv(OUT / "phase2_k562_1b/pair_dataset.csv")
    y = d["label"].to_numpy(int)
    g = d["pert_idx"].to_numpy()
    Xg = d[["target_mean", "target_variance", "target_dropout"]].to_numpy(np.float64)
    Xb = np.column_stack([Xg, d["attention_edge"].to_numpy(np.float64)])
    folds = list(GroupKFold(5).split(Xg, y, g))
    pg, pb = oof(Xg, y, g, folds), oof(Xb, y, g, folds)
    delta = roc_auc_score(y, pb) - roc_auc_score(y, pg)
    fold_mean = float(np.mean([roc_auc_score(y[te], pb[te]) - roc_auc_score(y[te], pg[te]) for _, te in folds]))
    ug = np.unique(g)
    rows_of = {k: np.flatnonzero(g == k) for k in ug}
    rng = np.random.default_rng(31)
    boot = []
    for _ in range(2000):
        ix = np.concatenate([rows_of[k] for k in rng.choice(ug, len(ug), replace=True)])
        boot.append(roc_auc_score(y[ix], pb[ix]) - roc_auc_score(y[ix], pg[ix]))
    boot = np.array(boot)
    refit = []
    for i in range(20):
        perm = np.random.default_rng(41 + i).permutation(ug)
        g2 = perm[np.searchsorted(ug, g)]
        f2 = list(GroupKFold(5).split(Xg, y, g2))
        refit.append(roc_auc_score(y, oof(Xb, y, g2, f2)) - roc_auc_score(y, oof(Xg, y, g2, f2)))
    refit = np.array(refit)
    res = {"n_pairs": int(len(y)), "n_perturbations": int(len(ug)),
           "fold_mean_delta": fold_mean, "pooled_oof_delta": float(delta),
           "bootstrap_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
           "refit_20_splits": {"mean": float(refit.mean()), "sd": float(refit.std(ddof=1)),
                               "min": float(refit.min()), "max": float(refit.max())},
           "_method": "route 1 explicit-replication bootstrap over perturbations (2000, seed 31), fold models fixed; "
                      "route 2 refit on 20 random perturbation-grouped 5-fold splits"}
    json.dump(res, open(VER / "incremental_1B_check.json", "w"), indent=1)
    print(res)


if __name__ == "__main__":
    main()
