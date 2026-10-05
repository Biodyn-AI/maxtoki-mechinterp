"""v2 step 8 — supplement: the TRRUST endpoint at every stored layer (not only layer 8).

Layer 8 was fixed before the runs (depth rule, runs/attention-grn-217M/README.md).
Other layers give higher raw TRRUST AUROCs, so a reader will ask whether they pass the
controls. For each run and layer:
  * pooled attention AUROC and the gap to the 3-feature gene model (v2_05 out-of-fold
    predictions are recomputed here the same way), 95% CI by TF bootstrap (2,000 reps);
  * Curveball null z and Phipson-Smyth p, using the SAME 1,000 null networks as v2_06
    (same seed and trade count; layer 8 must reproduce v2_06 exactly);
  * a layer-selection-aware test: statistic = max over layers of the null-standardised
    AUROC; its null = the same max computed on each null network (family-wise over layers).
Usage: python v2_08_layers.py [run ...] | combine
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v2_common import (RUNS, V2, N_BOOT, SEED, load_gene_features, load_attention_layer,  # noqa: E402
                       n_layers_stored, load_trrust, trrust_matrix, evaluated_tf_rows,
                       percentile_ci, save_json, WeightedAUROC, bh)
from v2_06_curveball import curveball, Scorer, N_DRAWS, N_TRADES  # noqa: E402

OUTD = V2 / "layers"
OUTD.mkdir(parents=True, exist_ok=True)


def run_one(ri, run):
    out = OUTD / f"layers_{run}.json"
    if out.exists():
        print(run, "cached"); return
    gf = load_gene_features(run); G = len(gf)
    sym = [s.upper() for s in gf["symbol"]]
    trrust = load_trrust()
    E = trrust_matrix(sym, trrust)
    tf_rows = evaluated_tf_rows(sym, trrust, E); nT = len(tf_rows)
    M = E[tf_rows]
    obs_rows = [set(np.flatnonzero(M[k]).tolist()) for k in range(nT)]
    forb = tf_rows.copy()
    L = n_layers_stored(run)
    layers = [load_attention_layer(run, li) for li in range(L)]
    for A in layers:
        np.fill_diagonal(A, 0.0)
    scorers = [Scorer(A[tf_rows].astype(np.float64), tf_rows, G) for A in layers]
    obs = np.array([s(obs_rows)[0] for s in scorers])
    # same null networks as v2_06
    rng = np.random.default_rng(SEED + 600 + ri)
    null = np.empty((N_DRAWS, L))
    for d in range(N_DRAWS):
        r = curveball(obs_rows, forb, N_TRADES, rng)
        null[d] = [s(r)[0] for s in scorers]
    mu, sd = null.mean(0), null.std(0, ddof=1)
    z = (obs - mu) / sd
    p = (1 + (null >= obs).sum(0)) / (1 + N_DRAWS)
    zn = (null - mu) / sd
    maxz_obs = z.max()
    p_max = float((1 + (zn.max(1) >= maxz_obs).sum()) / (1 + N_DRAWS))
    # gene3 OOF (same as v2_05) and TF-bootstrap gaps
    k_of = np.concatenate([np.full(G - 1, k) for k in range(nT)])
    cand = np.concatenate([np.delete(np.arange(G), r) for r in tf_rows])
    src = tf_rows[k_of]
    y = E[src, cand].astype(bool)
    X = np.column_stack([gf["mean_expr"].to_numpy()[cand], gf["variance"].to_numpy()[cand],
                         gf["dropout_rate"].to_numpy()[cand]]).astype(np.float32)
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups=k_of):
        m = Pipeline([("sc", StandardScaler()),
                      ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))]).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    f_g = WeightedAUROC(y, oof)
    f_a = [WeightedAUROC(y, A[src, cand].astype(np.float64)) for A in layers]
    rngb = np.random.default_rng(SEED + 900 + ri)
    gaps = np.empty((N_BOOT, L))
    for b in range(N_BOOT):
        w = np.bincount(rngb.integers(0, nT, nT), minlength=nT)[k_of]
        g = f_g(w)
        gaps[b] = [g - fa(w) for fa in f_a]
    rows = []
    for li in range(L):
        ci = percentile_ci(gaps[:, li])
        rows.append({"run": run, "layer": li, "n_layers": L, "attention_auroc": float(obs[li]),
                     "gene3_auroc": float(f_g()), "gene3_minus_attention": float(f_g() - obs[li]),
                     "gap_ci_low": ci[0], "gap_ci_high": ci[1],
                     "null_mean": float(mu[li]), "null_sd": float(sd[li]), "z": float(z[li]), "p": float(p[li])})
    save_json({"rows": rows, "max_z_over_layers": float(maxz_obs), "argmax_layer": int(np.argmax(z)),
               "p_max_over_layers": p_max}, out)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(run, "max z", round(maxz_obs, 2), "at layer", int(np.argmax(z)), "p(max over layers)", round(p_max, 4), flush=True)


def combine():
    allr = {r: json.load(open(OUTD / f"layers_{r}.json")) for r in RUNS}
    df = pd.concat([pd.DataFrame(allr[r]["rows"]) for r in RUNS])
    df.to_csv(OUTD / "per_layer_trrust.csv", index=False)
    runs = list(RUNS)
    q = bh(np.array([allr[r]["p_max_over_layers"] for r in runs]))
    summ = {r: {"max_z_over_layers": allr[r]["max_z_over_layers"], "argmax_layer": allr[r]["argmax_layer"],
                "p_max_over_layers": allr[r]["p_max_over_layers"], "p_bh_across_4_runs": float(qq)}
            for r, qq in zip(runs, q)}
    summ["_method"] = ("same 1,000 Curveball networks as v2_06; per-layer z; family-wise test uses the max over "
                       "layers of z computed identically on every null network; Phipson-Smyth p; BH across runs")
    save_json(summ, OUTD / "layers_summary.json")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["combine"]:
        combine()
    else:
        for ri, r in enumerate(RUNS):
            if not args or r in args:
                run_one(ri, r)
