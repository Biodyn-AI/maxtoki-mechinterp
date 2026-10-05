"""v2 step 7 — residualised attention on the TRRUST endpoint, per run, with CIs.

Residualisation = predict the attention score of every ordered gene pair (i != j, all
1,500 x 1,499 pairs) from gene-level features only, with 5-fold cross-fitting over pairs
(KFold shuffle, random_state=42, as deployed), and keep observed - predicted.
The residual is then scored on the TRRUST endpoint exactly like raw attention.

  ols5  deployed version (phase3_residualization.py:148-214): StandardScaler + OLS on
        [source mean, source variance, target mean, target variance, target dropout].
        Must reproduce the deployed residualised AUROCs. Inputs are float32, as deployed.
  ols5_f64  [added in verification, 2026-10-01] the same OLS with float64 inputs. In
        float32, sklearn LinearRegression does not reach the least-squares fit on these
        data (training R^2 is lower than the float64 fit in K562, Adamson and 1B; see
        outputs/v2_eval/verification/ols_precision_check.json). ols5_f64 is the correct
        OLS residual; ols5 is kept only as the replica of the deployed number.
  hgb6  non-linear check: HistGradientBoostingRegressor on the same three gene features
        (mean, variance, dropout) for both genes of the pair (6 features).
CI: percentile bootstrap over evaluated TFs (2,000 reps), fold models held fixed.
Usage: python v2_07_residualise.py [run ...] | combine   (resumable per run; a cached run
without ols5_f64 gets only that variant added, with the same bootstrap draws)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v2_common import (RUNS, V2, N_BOOT, SEED, phase_dir, load_gene_features,  # noqa: E402
                       load_attention_layer, load_trrust, trrust_matrix, evaluated_tf_rows,
                       auroc_rank, percentile_ci, save_json, WeightedAUROC)

OUTD = V2 / "residualised"
OUTD.mkdir(parents=True, exist_ok=True)


def residualise(Aoff, feats, model_fn):
    resid = np.empty_like(Aoff)
    r2 = []
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=42).split(feats):
        m = model_fn().fit(feats[tr], Aoff[tr])
        resid[te] = Aoff[te] - m.predict(feats[te])
        pt = m.predict(feats[tr])
        r2.append(1 - ((Aoff[tr] - pt) ** 2).sum() / (((Aoff[tr] - Aoff[tr].mean()) ** 2).sum() + 1e-12))
    return resid, float(np.mean(r2))


def run_one(ri, run):
    out_path = OUTD / f"resid_{run}.json"
    prev = json.load(open(out_path)) if out_path.exists() else None
    if prev is not None and "ols5_f64" in prev:
        print(run, "cached"); return
    t0 = time.time()
    gf = load_gene_features(run)
    G = len(gf)
    sym = [s.upper() for s in gf["symbol"]]
    trrust = load_trrust()
    E = trrust_matrix(sym, trrust)
    tf_rows = evaluated_tf_rows(sym, trrust, E)
    A = load_attention_layer(run).astype(np.float32); np.fill_diagonal(A, 0.0)
    gm = gf["mean_expr"].to_numpy(np.float32); gv = gf["variance"].to_numpy(np.float32)
    gd = gf["dropout_rate"].to_numpy(np.float32)
    ii, jj = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    mask = ii != jj
    src, tgt = ii[mask], jj[mask]            # same pair order as deployed _flatten_pairwise
    del ii, jj
    Aoff = A[src, tgt]
    # evaluated pooled pairs
    k_of = np.concatenate([np.full(G - 1, k) for k in range(len(tf_rows))])
    e_src = tf_rows[k_of]
    e_tgt = np.concatenate([np.delete(np.arange(G), r) for r in tf_rows])
    y = E[e_src, e_tgt].astype(bool)
    # index of each evaluated pair in the flat off-diagonal order: i*(G-1) + j - (j > i)
    flat = e_src * (G - 1) + e_tgt - (e_tgt > e_src)
    res = {"raw_attention_auroc": auroc_rank(y, A[e_src, e_tgt])}
    gm64, gv64, gd64 = (gf[c].to_numpy(np.float64) for c in ["mean_expr", "variance", "dropout_rate"])
    variants = {
        "ols5": (lambda: np.column_stack([gm[src], gv[src], gm[tgt], gv[tgt], gd[tgt]]),
                 lambda: Pipeline([("sc", StandardScaler()), ("ols", LinearRegression())]), Aoff),
        "ols5_f64": (lambda: np.column_stack([gm64[src], gv64[src], gm64[tgt], gv64[tgt], gd64[tgt]]),
                     lambda: Pipeline([("sc", StandardScaler()), ("ols", LinearRegression())]),
                     Aoff.astype(np.float64)),
        "hgb6": (lambda: np.column_stack([gm[src], gv[src], gd[src], gm[tgt], gv[tgt], gd[tgt]]),
                 lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1,
                                                       max_leaf_nodes=31, random_state=SEED), Aoff),
    }
    if prev is not None:                       # keep the cached variants; add only ols5_f64
        res = prev
        variants = {"ols5_f64": variants["ols5_f64"]}
    rng = np.random.default_rng(SEED + 800 + ri)
    nT = len(tf_rows)
    draws = [np.bincount(rng.integers(0, nT, nT), minlength=nT) for _ in range(N_BOOT)]
    f_raw = WeightedAUROC(y, A[e_src, e_tgt])
    raw_boot = np.array([f_raw(c[k_of]) for c in draws])
    res["raw_attention_ci95"] = percentile_ci(raw_boot)
    for name, (mkX, fn, target) in variants.items():
        t1 = time.time()
        X = mkX()
        R, r2 = residualise(target, X, fn)
        s = R[flat].astype(np.float64)
        auc = auroc_rank(y, s)
        f = WeightedAUROC(y, s)
        boot = np.array([f(c[k_of]) for c in draws])
        res[name] = {"residualised_auroc": auc, "ci95": percentile_ci(boot),
                     "drop_from_raw": res["raw_attention_auroc"] - auc,
                     "drop_ci95": percentile_ci(raw_boot - boot),
                     "r2_train_mean": r2, "seconds": time.time() - t1}
        del X
        print(run, name, res[name], flush=True)
    dep = np.genfromtxt(phase_dir("phase3", run) / "residualization_results.csv", delimiter=",",
                        names=True, dtype=None, encoding=None)
    for row in dep:
        if row["edge"] == "attention":
            res[f"deployed_{row['model']}_residualised_auroc"] = float(row["residualized_trrust_auroc"])
    res["n_tfs"] = int(nT); res["n_positive_pairs"] = int(y.sum())
    res["seconds" if prev is None else "seconds_ols5_f64_added"] = time.time() - t0
    save_json(res, out_path)


def combine():
    allr = {r: json.load(open(OUTD / f"resid_{r}.json")) for r in RUNS}
    allr["_method"] = {"bootstrap_unit": "TF", "n_reps": N_BOOT, "ci": "percentile", "seed_base": SEED + 800,
                       "cross_fit": "KFold(5, shuffle=True, random_state=42) over all ordered gene pairs",
                       "ols5": "float32 inputs, replica of the deployed number (not the least-squares fit in K562, Adamson, 1B)",
                       "ols5_f64": "float64 inputs, the correct OLS residual (added in verification 2026-10-01)"}
    save_json(allr, OUTD / "residualised_summary.json")
    for r in RUNS:
        a = allr[r]
        print(r, "raw", round(a["raw_attention_auroc"], 4), "ols5", round(a["ols5"]["residualised_auroc"], 4),
              [round(x, 4) for x in a["ols5"]["ci95"]], "deployed", round(a["deployed_ols_residualised_auroc"], 4),
              "hgb6", round(a["hgb6"]["residualised_auroc"], 4), [round(x, 4) for x in a["hgb6"]["ci95"]],
              "ols5_f64", round(a["ols5_f64"]["residualised_auroc"], 4), [round(x, 4) for x in a["ols5_f64"]["ci95"]],
              "drop", round(a["ols5_f64"]["drop_from_raw"], 4), [round(x, 4) for x in a["ols5_f64"]["drop_ci95"]])


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["combine"]:
        combine()
    else:
        for ri, r in enumerate(RUNS):
            if not args or r in args:
                run_one(ri, r)
