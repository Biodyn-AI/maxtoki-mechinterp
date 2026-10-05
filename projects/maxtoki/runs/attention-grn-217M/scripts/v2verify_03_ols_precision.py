"""Check: is the deployed OLS residualisation (float32 sklearn LinearRegression) really the
least-squares fit?

For each run, fit the deployed OLS (StandardScaler + LinearRegression on
[src mean, src var, tgt mean, tgt var, tgt dropout] of every ordered off-diagonal gene
pair, 5-fold KFold(shuffle=True, random_state=42)) three ways:
  f32_sklearn  inputs float32 (as deployed and as v2_07)
  f64_sklearn  same pipeline, inputs float64
  f64_lstsq    numpy lstsq with an intercept, float64
Record per fold the training R^2 (a true least-squares fit has the LARGEST training R^2
of any linear fit on that fold, so if float32 R^2 < float64 R^2 the float32 result is
not the least-squares solution), the coefficients, and the TRRUST pooled AUROC of the
cross-fitted residual (deployed evaluation: evaluated TFs with >= 3 targets, all other
genes as candidates, one pooled AUROC).
Output: outputs/v2_eval/verification/ols_precision_check.json
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path("<REPO_ROOT>/projects/maxtoki/runs/attention-grn-217M/outputs")
VER = OUT / "v2_eval" / "verification"
TRRUST = Path("<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv")
RUNS = {"217M_K562": "", "217M_RPE1": "_rpe1", "217M_Adamson": "_adamson", "1B_K562": "_k562_1b"}


def main():
    with open(TRRUST) as f:
        pairs = [(r[0].upper(), r[1].upper()) for r in csv.reader(f, delimiter="\t") if len(r) >= 2]
    res = {}
    for run, sfx in RUNS.items():
        gf = pd.read_csv(OUT / f"phase0{sfx}/gene_features.csv")
        G = len(gf)
        idx = {s.upper(): i for i, s in enumerate(gf["symbol"])}
        edges = {(idx[a], idx[b]) for a, b in pairs if a in idx and b in idx and a != b}
        outdeg = np.bincount([a for a, _ in edges], minlength=G)
        tfs = np.array(sorted({idx[a] for a, _ in pairs if a in idx and outdeg[idx[a]] >= 3}))
        src = np.repeat(tfs, G - 1)
        cand = np.concatenate([np.array([g for g in range(G) if g != t]) for t in tfs])
        y = np.array([(s, c) in edges for s, c in zip(src, cand)], dtype=int)
        A32 = np.array(np.load(OUT / f"phase0{sfx}/attention_edges_layer_mean.npy", mmap_mode="r")[8], dtype=np.float32)
        np.fill_diagonal(A32, 0.0)
        ii, jj = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
        mk = ii != jj
        s_all, t_all = ii[mk], jj[mk]
        del ii, jj
        folds = list(KFold(5, shuffle=True, random_state=42).split(np.zeros(len(s_all))))
        out = {"raw_attention_auroc": float(roc_auc_score(y, A32[src, cand]))}
        for name in ["f32_sklearn", "f64_sklearn", "f64_lstsq"]:
            dt = np.float32 if name.startswith("f32") else np.float64
            gm, gv, gd = (gf[c].to_numpy(dt) for c in ["mean_expr", "variance", "dropout_rate"])
            X = np.column_stack([gm[s_all], gv[s_all], gm[t_all], gv[t_all], gd[t_all]]).astype(dt)
            yy = A32[s_all, t_all].astype(dt)
            resid = np.empty(len(yy), dtype=np.float64)
            r2, coefs = [], []
            for tr, te in folds:
                if name == "f64_lstsq":
                    Xd = np.column_stack([np.ones(len(tr)), X[tr]])
                    b, *_ = np.linalg.lstsq(Xd, yy[tr], rcond=None)
                    ptr = Xd @ b
                    pte = np.column_stack([np.ones(len(te)), X[te]]) @ b
                    coefs.append(b[1:].tolist())
                else:
                    m = Pipeline([("sc", StandardScaler()), ("ols", LinearRegression())]).fit(X[tr], yy[tr])
                    ptr = m.predict(X[tr]); pte = m.predict(X[te])
                    coefs.append((m[-1].coef_ / m[0].scale_).astype(float).tolist())  # raw-scale slopes
                ytr = yy[tr].astype(np.float64)
                r2.append(float(1 - ((ytr - ptr) ** 2).sum() / ((ytr - ytr.mean()) ** 2).sum()))
                resid[te] = yy[te].astype(np.float64) - pte.astype(np.float64)
            R = np.zeros((G, G)); R[s_all, t_all] = resid
            out[name] = {"residualised_auroc": float(roc_auc_score(y, R[src, cand])),
                         "train_r2_per_fold": r2, "raw_scale_slopes_fold0": coefs[0]}
            del X, yy, resid, R
        res[run] = out
        print(run, json.dumps({k: (v if not isinstance(v, dict) else {kk: v[kk] for kk in ["residualised_auroc", "train_r2_per_fold"]}) for k, v in out.items()}), flush=True)
    res["_method"] = ("deployed OLS residualisation refitted in float32 (as deployed) and float64 (two routes); "
                      "training R^2 per fold; TRRUST pooled AUROC of the cross-fitted residual")
    json.dump(res, open(VER / "ols_precision_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
