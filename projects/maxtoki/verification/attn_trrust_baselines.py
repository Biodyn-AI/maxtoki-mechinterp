"""Read-only re-analysis of the attention-GRN TRRUST endpoint.

For each of the 4 runs, reproduce the Phase-3 TRRUST evaluation exactly
(same TRRUST file, same HVG grid, same tf_mask >=3 targets, same pooled
AUROC), then score the SAME labels with gene-level baselines that the
pipeline never computed on this endpoint (target variance, mean, 1-dropout).
Also: count positives/negatives, per-TF AUROCs, TF-cluster bootstrap CI of
(variance - attention), and a mixing check of the Phase-3 curveball null.
No model forward pass. Loads one (G,G) layer slice via mmap.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OUT = Path("<REPO_ROOT>/projects/maxtoki/runs/attention-grn-217M/outputs")
TRRUST_TSV = Path("<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv")
PRIMARY_LAYER = 8
SEED = 42
RUNS = {"217M_K562": "", "217M_RPE1": "_rpe1", "217M_Adamson": "_adamson", "1B_K562": "_k562_1b"}

trrust = pd.read_csv(TRRUST_TSV, sep="\t", header=None, names=["tf", "target", "mode", "pmid"])
trrust["tf"] = trrust["tf"].str.upper()
trrust["target"] = trrust["target"].str.upper()


def curveball_permute(mat, n_iter, seed):
    """Verbatim copy of phase3_residualization.py:221-246."""
    rng = np.random.default_rng(seed)
    rows = [set(np.where(r)[0]) for r in mat.astype(bool)]
    G = len(rows)
    n_changed_swaps = 0
    for _ in range(n_iter):
        i, j = rng.choice(G, 2, replace=False)
        a, b = rows[i], rows[j]
        inter = a & b
        sym = (a | b) - inter
        if len(sym) < 2:
            continue
        sym_list = list(sym)
        rng.shuffle(sym_list)
        k = len(a) - len(inter)
        new_a = inter | set(sym_list[:k])
        new_b = inter | set(sym_list[k:])
        if new_a != a:
            n_changed_swaps += 1
        rows[i] = new_a
        rows[j] = new_b
    out = np.zeros_like(mat)
    for r_idx, cols in enumerate(rows):
        for c in cols:
            out[r_idx, c] = 1
    return out, n_changed_swaps


def curveball_mixed(mat, n_iter, seed):
    """Same trade rule, but pairs drawn only among non-empty rows (so trades happen)."""
    rng = np.random.default_rng(seed)
    nz = np.where(mat.sum(1) > 0)[0]
    rows = {int(r): set(np.where(mat[r])[0]) for r in nz}
    keys = list(rows.keys())
    for _ in range(n_iter):
        i, j = rng.choice(len(keys), 2, replace=False)
        i, j = keys[i], keys[j]
        a, b = rows[i], rows[j]
        inter = a & b
        sym = (a | b) - inter
        if len(sym) < 2:
            continue
        sym_list = list(sym)
        rng.shuffle(sym_list)
        k = len(a) - len(inter)
        rows[i] = inter | set(sym_list[:k])
        rows[j] = inter | set(sym_list[k:])
    out = np.zeros_like(mat)
    for r, cols in rows.items():
        for c in cols:
            out[r, c] = 1
    return out


results = {}
for name, sfx in RUNS.items():
    gf = pd.read_csv(OUT / f"phase0{sfx}" / "gene_features.csv")
    G = len(gf)
    sym = [s.upper() for s in gf["symbol"]]
    s2i = {s: i for i, s in enumerate(sym)}
    attn_all = np.load(OUT / f"phase0{sfx}" / "attention_edges_layer_mean.npy", mmap_mode="r")
    attn = np.array(attn_all[PRIMARY_LAYER], dtype=np.float32)
    del attn_all
    np.fill_diagonal(attn, 0.0)
    sp = np.load(OUT / f"phase0{sfx}" / "spearman_edges.npy").astype(np.float32)
    np.fill_diagonal(sp, 0.0)
    sp = np.abs(sp)

    E = np.zeros((G, G), dtype=np.int8)
    for tf, tg in zip(trrust["tf"], trrust["target"]):
        i = s2i.get(tf); j = s2i.get(tg)
        if i is not None and j is not None and i != j:
            E[i, j] = 1
    tf_in_hvg = set(trrust["tf"]) & set(sym)
    tf_mask = np.zeros(G, bool)
    for tf in tf_in_hvg:
        i = s2i[tf]
        if E[i].sum() >= 3:
            tf_mask[i] = True
    tf_rows = np.where(tf_mask)[0]

    gvar = gf["variance"].to_numpy(np.float64)
    gmean = gf["mean_expr"].to_numpy(np.float64)
    g1md = 1.0 - gf["dropout_rate"].to_numpy(np.float64)

    def pooled(score_fn, labels=E):
        s, l, grp = [], [], []
        for r in tf_rows:
            m = np.ones(G, bool); m[r] = False
            s.append(score_fn(r)[m]); l.append(labels[r][m]); grp.append(np.full(m.sum(), r))
        return np.concatenate(s), np.concatenate(l), np.concatenate(grp)

    scorers = {
        "attention_L8": lambda r: attn[r],
        "spearman_abs": lambda r: sp[r],
        "target_variance": lambda r: gvar,
        "target_mean_expr": lambda r: gmean,
        "target_one_minus_dropout": lambda r: g1md,
    }
    res = {"n_edges_in_hvg_grid": int(E.sum()), "n_nonempty_rows": int((E.sum(1) > 0).sum()),
           "n_tfs_evaluated": int(tf_mask.sum()), "tfs": [sym[r] for r in tf_rows]}
    pooled_scores = {}
    for k, fn in scorers.items():
        s, l, grp = pooled(fn)
        pooled_scores[k] = s
        res[f"auroc_pooled_{k}"] = float(roc_auc_score(l, s))
    res["n_positive"] = int(l.sum()); res["n_negative"] = int(len(l) - l.sum()); res["n_pairs"] = int(len(l))
    # per-TF AUROCs
    per_tf = []
    for r in tf_rows:
        m = np.ones(G, bool); m[r] = False
        y = E[r][m]
        per_tf.append({"tf": sym[r], "n_pos": int(y.sum()),
                       "attn": float(roc_auc_score(y, attn[r][m])),
                       "var": float(roc_auc_score(y, gvar[m]))})
    per_tf = pd.DataFrame(per_tf)
    res["per_tf"] = per_tf.to_dict(orient="records")
    res["per_tf_mean_attn"] = float(per_tf["attn"].mean())
    res["per_tf_mean_var"] = float(per_tf["var"].mean())
    res["per_tf_n_var_beats_attn"] = int((per_tf["var"] > per_tf["attn"]).sum())
    # TF-cluster bootstrap of pooled (var - attn)
    rng = np.random.default_rng(SEED)
    grp_ids = np.unique(grp)
    idx_by = {g: np.where(grp == g)[0] for g in grp_ids}
    diffs = []
    for _ in range(2000):
        pick = rng.choice(grp_ids, len(grp_ids), replace=True)
        ii = np.concatenate([idx_by[g] for g in pick])
        yy = l[ii]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        diffs.append(roc_auc_score(yy, pooled_scores["target_variance"][ii]) -
                     roc_auc_score(yy, pooled_scores["attention_L8"][ii]))
    diffs = np.array(diffs)
    res["boot_var_minus_attn_mean"] = float(diffs.mean())
    res["boot_var_minus_attn_ci95"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    res["boot_frac_var_gt_attn"] = float((diffs > 0).mean())

    # Curveball mixing check: verbatim null, first 20 seeds
    n_iter = 5 * int(E.sum())
    changed_cells, changed_swaps = [], []
    for t in range(20):
        P, nsw = curveball_permute(E, n_iter, SEED + t)
        changed_cells.append(int((P != E).sum()))
        changed_swaps.append(nsw)
    res["verbatim_curveball_n_iter"] = n_iter
    res["verbatim_curveball_changed_cells_first20"] = changed_cells
    res["verbatim_curveball_effective_swaps_first20"] = changed_swaps
    # Mixed curveball null, n=200
    s_att, _, _ = pooled(scorers["attention_L8"])
    null = []
    for t in range(200):
        P = curveball_mixed(E, max(n_iter, 2000), SEED + 10_000 + t)
        _, lp, _ = pooled(scorers["attention_L8"], labels=P)
        null.append(roc_auc_score(lp, s_att))
    null = np.array(null)
    obs = res["auroc_pooled_attention_L8"]
    res["mixed_curveball_null_mean"] = float(null.mean())
    res["mixed_curveball_null_std"] = float(null.std())
    res["mixed_curveball_z"] = float((obs - null.mean()) / max(null.std(), 1e-12))
    res["mixed_curveball_p_upper"] = float((1 + (null >= obs).sum()) / (1 + len(null)))
    results[name] = res
    print(name, json.dumps({k: v for k, v in res.items() if k != "per_tf"}, indent=1), flush=True)
    print(per_tf.to_string(index=False), flush=True)

with open(Path(sys.argv[0]).with_suffix(".json"), "w") as f:
    json.dump(results, f, indent=1)
