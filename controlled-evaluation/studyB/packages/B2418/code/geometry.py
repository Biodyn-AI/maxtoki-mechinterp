"""Global shape of the cell-embedding cloud for three single-cell foundation models.

For each model, reads data/embeddings/<model>_gut.npz (per-cell mean-pooled layer-11 residual,
one row per cell, same 4,269 cells for every model) and measures how compact the cloud is:

  participation_ratio  (sum lambda)^2 / sum lambda^2 over all covariance eigenvalues
  pc1_share            lambda_1 / sum lambda
  d90                  smallest number of PCs whose cumulative share reaches 0.90
  norm_mean, norm_cv   mean and coefficient of variation of the per-cell L2 norm

Lower participation ratio / d90, higher PC1 share and lower norm CV = more compact.
Stability: the same statistics on 10 random halves of the cells (seed 0).

Run from the package root:
  python code/geometry.py
Writes outputs/geometry.json, outputs/geometry_table.tsv, outputs/run_log.txt.
"""
import os, json
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB = os.path.join(ROOT, "data", "embeddings")
OUT = os.path.join(ROOT, "outputs")
MODELS = ["scgpt", "geneformer", "state"]
TISSUE = "gut"
N_HALVES = 10
SEED = 0


def spectrum(X):
    X = X.astype(np.float64)
    Xc = X - X.mean(0)
    C = Xc.T @ Xc / (len(Xc) - 1)
    lam = np.clip(np.linalg.eigvalsh(C)[::-1], 0.0, None)
    return lam


def shape_stats(X):
    lam = spectrum(X)
    r = lam / lam.sum()
    norms = np.linalg.norm(X.astype(np.float64), axis=1)
    return dict(
        participation_ratio=float(lam.sum() ** 2 / (lam ** 2).sum()),
        pc1_share=float(r[0]),
        d90=int(np.searchsorted(np.cumsum(r), 0.90) + 1),
        norm_mean=float(norms.mean()),
        norm_cv=float(norms.std() / norms.mean()),
    )


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {}
    for m in MODELS:
        z = np.load(os.path.join(EMB, f"{m}_{TISSUE}.npz"), allow_pickle=True)
        data[m] = (z["emb"], z["cell_idx"])
    ref = data[MODELS[0]][1]
    for m in MODELS[1:]:
        assert np.array_equal(data[m][1], ref), f"{m}: cells differ from {MODELS[0]}"

    rng = np.random.default_rng(SEED)
    halves = [np.sort(rng.choice(len(ref), len(ref) // 2, replace=False)) for _ in range(N_HALVES)]

    res = {}
    for m in MODELS:
        X = data[m][0]
        rec = dict(n_cells=int(X.shape[0]), width=int(X.shape[1]), **shape_stats(X))
        hs = [shape_stats(X[h]) for h in halves]
        for k in ("participation_ratio", "pc1_share", "d90"):
            v = np.array([h[k] for h in hs], float)
            rec[f"{k}_halves_min"] = float(v.min())
            rec[f"{k}_halves_max"] = float(v.max())
        res[m] = rec

    ranks = {}
    for k, lower_is_compact in (("participation_ratio", True), ("pc1_share", False),
                                ("d90", True), ("norm_cv", True)):
        order = sorted(MODELS, key=lambda m: res[m][k] if lower_is_compact else -res[m][k])
        ranks[k] = order
    out = dict(tissue=TISSUE, layer=11, pooling="mean over expressed gene tokens",
               n_halves=N_HALVES, seed=SEED, models=res, most_to_least_compact=ranks)
    json.dump(out, open(os.path.join(OUT, "geometry.json"), "w"), indent=1)

    cols = ["n_cells", "width", "participation_ratio", "pc1_share", "d90", "norm_mean", "norm_cv",
            "participation_ratio_halves_min", "participation_ratio_halves_max",
            "pc1_share_halves_min", "pc1_share_halves_max", "d90_halves_min", "d90_halves_max"]
    with open(os.path.join(OUT, "geometry_table.tsv"), "w") as f:
        f.write("model\t" + "\t".join(cols) + "\n")
        for m in MODELS:
            f.write(m + "\t" + "\t".join(
                f"{res[m][c]:.4f}" if isinstance(res[m][c], float) else str(res[m][c]) for c in cols) + "\n")

    lines = [f"tissue={TISSUE} layer=11 n_cells={len(ref)} halves={N_HALVES}",
             f"{'model':11s} {'width':>5s} {'PR':>7s} {'PC1':>6s} {'d90':>4s} {'norm':>7s} {'normCV':>7s}"
             f"   PR halves      PC1 halves     d90 halves"]
    for m in MODELS:
        r = res[m]
        lines.append(
            f"{m:11s} {r['width']:5d} {r['participation_ratio']:7.2f} {r['pc1_share']:6.3f} {r['d90']:4d} "
            f"{r['norm_mean']:7.2f} {r['norm_cv']:7.4f}   "
            f"{r['participation_ratio_halves_min']:5.2f}-{r['participation_ratio_halves_max']:<6.2f}  "
            f"{r['pc1_share_halves_min']:.3f}-{r['pc1_share_halves_max']:.3f}    "
            f"{int(r['d90_halves_min'])}-{int(r['d90_halves_max'])}")
    for k, order in ranks.items():
        lines.append(f"most->least compact by {k}: {' > '.join(order)}")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(OUT, "run_log.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
