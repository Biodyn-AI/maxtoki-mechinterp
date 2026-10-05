"""Sensitivity: the TRRUST variance baseline when gene variance is computed on a single
log scale. For 217M K562, 1B K562 and 217M Adamson the h5ad X already holds
log1p(CP10k) values, and Phase 0 applied a second scaling + log1p before computing
gene variance. Here variance = variance of the stored values over the same Phase-0
control cells (single log). TRRUST evaluation as deployed (evaluated TFs with >= 3
targets, all other genes as candidates, pooled AUROC). TF-bootstrap CI of
(variance - attention), 2,000 reps, seed 61+run_index, explicit replication.
Output: outputs/v2_eval/verification/single_log_trrust_check.json
"""
import csv, json, sys
from pathlib import Path
import h5py, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
PROJ = Path("<REPO_ROOT>/projects/maxtoki")
OUT = PROJ / "runs/attention-grn-217M/outputs"
sys.path.insert(0, str(PROJ / "setup"))
from dataset_loader import resolve as load_ds  # noqa: E402
TRRUST = Path("<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv")
pairs = [(r[0].upper(), r[1].upper()) for r in csv.reader(open(TRRUST), delimiter="\t") if len(r) >= 2]
res = {}
for ri, (run, sfx, dsk) in enumerate([("217M_K562", "", "k562"), ("217M_Adamson", "_adamson", "adamson"), ("1B_K562", "_k562_1b", "k562")]):
    gf = pd.read_csv(OUT / f"phase0{sfx}/gene_features.csv"); G = len(gf)
    rows = np.sort(pd.read_csv(OUT / f"phase0{sfx}/control_cells.csv")["cell_idx_global"].to_numpy())
    cols = gf["var_idx"].to_numpy()
    ds = load_ds(dsk)
    with h5py.File(ds.h5_path, "r") as f:
        X = f["X"]
        if isinstance(X, h5py.Dataset):
            V = np.vstack([X[rows[a:a + 500], :][:, cols] for a in range(0, len(rows), 500)]).astype(np.float64)
        else:
            indptr = X["indptr"][:]; V = np.zeros((len(rows), G))
            pos = {c: k for k, c in enumerate(cols)}
            for i, r in enumerate(rows):
                d = X["data"][indptr[r]:indptr[r + 1]]; ix = X["indices"][indptr[r]:indptr[r + 1]]
                for v, c in zip(d, ix):
                    k = pos.get(int(c))
                    if k is not None:
                        V[i, k] = v
    var1 = V.var(0); var2 = gf["variance"].to_numpy(np.float64)
    idx = {s.upper(): i for i, s in enumerate(gf["symbol"])}
    edges = {(idx[a], idx[b]) for a, b in pairs if a in idx and b in idx and a != b}
    outdeg = np.bincount([a for a, _ in edges], minlength=G)
    tfs = np.array(sorted({idx[a] for a, _ in pairs if a in idx and outdeg[idx[a]] >= 3}))
    src = np.repeat(tfs, G - 1); cand = np.concatenate([[g for g in range(G) if g != t] for t in tfs]).astype(int)
    grp = np.repeat(np.arange(len(tfs)), G - 1)
    y = np.array([(s, c) in edges for s, c in zip(src, cand)], dtype=int)
    A = np.array(np.load(OUT / f"phase0{sfx}/attention_edges_layer_mean.npy", mmap_mode="r")[8], dtype=np.float64); np.fill_diagonal(A, 0)
    att = A[src, cand]
    rng = np.random.default_rng(61 + ri); byk = [np.flatnonzero(grp == k) for k in range(len(tfs))]
    gaps = []
    for _ in range(2000):
        ix = np.concatenate([byk[k] for k in rng.integers(0, len(tfs), len(tfs))])
        gaps.append(roc_auc_score(y[ix], var1[cand][ix]) - roc_auc_score(y[ix], att[ix]))
    res[run] = {"variance_single_log_auroc": float(roc_auc_score(y, var1[cand])),
                "variance_deployed_auroc": float(roc_auc_score(y, var2[cand])),
                "attention_auroc": float(roc_auc_score(y, att)),
                "gap_single_log_variance_minus_attention_ci95": [float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))],
                "spearman_var_single_vs_deployed": float(pd.Series(var1).corr(pd.Series(var2), method="spearman"))}
    print(run, res[run], flush=True)
json.dump(res, open(OUT / "v2_eval/verification/single_log_trrust_check.json", "w"), indent=1)
