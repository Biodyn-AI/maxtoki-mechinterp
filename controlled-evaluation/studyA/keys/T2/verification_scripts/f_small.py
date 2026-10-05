import numpy as np, pandas as pd, json
from sklearn.metrics import roc_auc_score
PKG = "<EVAL_ROOT>/studyA/tasks/T2-paper/data"
P = pd.read_parquet(f"{PKG}/gene_pairs.parquet")
y = (P.lfc.values < 0); f = (P.n_inhibitory_evidence / P.evidence).values
g = P.silenced_gene.astype("category").cat.codes.values; NG = g.max() + 1
# per-gene AUROC
au = []
for k in range(NG):
    m = g == k
    if y[m].min() != y[m].max() and np.unique(f[m]).size > 1:
        au.append(roc_auc_score(y[m], f[m]))
out = {"mean_per_gene_auroc": float(np.mean(au)), "n_genes_auroc": len(au)}
# gene bootstrap of pooled AUROC via weights (500 reps)
rng = np.random.default_rng(17); idx_by_g = [np.flatnonzero(g == k) for k in range(NG)]
bs = []
for _ in range(500):
    w = np.bincount(rng.integers(0, NG, NG), minlength=NG)[g]
    bs.append(roc_auc_score(y, f, sample_weight=w))
out["auroc_boot_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
print(json.dumps(out))
