"""Significance test for branch-point curvature in single-cell foundation-model embeddings.

Statistic, per model x tissue (full embedding, no PCA):

    curvature = CV R2[degree-2 polynomial kernel ridge] - CV R2[linear kernel ridge]

where both ridges predict diffusion pseudotime from the cell embedding.

Null: blocked permutations of pseudotime. Cells are sorted by pseudotime and cut into
consecutive blocks of BLOCK cells; the block order is shuffled and the labels are put back on
the sorted cells. The same statistic is computed on each permuted target.

Also reports a bootstrap 95% CI of the curvature (cells resampled from the out-of-fold
predictions; decoders are not refit).

Each training fold's Gram matrix depends only on X, so it is eigendecomposed once. Every alpha
and every null draw is then a matrix-vector product.

Run from the package root:
    python code/curvature_significance.py
Writes outputs/curvature_significance.json and outputs/results_table.tsv.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "outputs")

SEED = 0
N_FOLDS = 5
ALPHAS = [1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4]
MODELS = ["geneformer", "scgpt"]
TISSUES = ["lung", "gut", "pancreas"]
SPECIES = {"lung": "human", "gut": "human", "pancreas": "mouse"}

N_NULL = 200   # p-value floor = 1/201
N_BOOT = 1000
BLOCK = 50     # cells per consecutive pseudotime block


def r2(y, p):
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def null_draw(y, rng, block=BLOCK):
    """Shuffle consecutive pseudotime-ordered blocks of labels; keeps local label smoothness."""
    n = len(y)
    order = np.argsort(y)
    ys = y[order]
    nb = int(np.ceil(n / block))
    bounds = [(i * block, min((i + 1) * block, n)) for i in range(nb)]
    perm = rng.permutation(nb)
    shuffled = np.concatenate([ys[bounds[b][0]:bounds[b][1]] for b in perm])
    out = np.empty(n, dtype=float)
    out[order] = shuffled
    return out


class FoldSolver:
    """Kernel ridge over fixed CV folds, with each training Gram eigendecomposed once.

    For K_tr = U diag(lam) U^T:  pred_te = (K_te,tr U) diag(1 / (lam + alpha)) (U^T y_tr).
    (K_te,tr U) and U do not depend on y, so they are reused across alphas and targets.
    """

    def __init__(self, K, n, seed=SEED):
        self.n = n
        self.folds = []
        for tr, te in KFold(N_FOLDS, shuffle=True, random_state=seed).split(np.arange(n)):
            lam, U = np.linalg.eigh(K[np.ix_(tr, tr)])
            KteU = K[np.ix_(te, tr)] @ U
            self.folds.append((tr, te, lam, U, KteU))

    def best_r2(self, y):
        """CV R2 at the best alpha on the grid; returns (r2, alpha, out-of-fold predictions)."""
        preds = np.zeros((len(ALPHAS), self.n))
        for tr, te, lam, U, KteU in self.folds:
            Uty = U.T @ y[tr]
            for j, a in enumerate(ALPHAS):
                preds[j, te] = KteU @ (Uty / (lam + a))
        scores = [r2(y, preds[j]) for j in range(len(ALPHAS))]
        j = int(np.argmax(scores))
        return float(scores[j]), ALPHAS[j], preds[j]


def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    q = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def run(model, tissue):
    z = np.load(os.path.join(DATA, f"{model}_{tissue}.npz"))
    X = z["emb"].astype(np.float64)
    y = z["pseudotime"].astype(np.float64)
    n = len(y)
    Xz = StandardScaler().fit_transform(X)
    d = Xz.shape[1]

    gamma = 1.0 / d
    Klin = gamma * (Xz @ Xz.T) + 1.0
    Kpoly = Klin ** 2

    S_lin = FoldSolver(Klin, n)
    S_poly = FoldSolver(Kpoly, n)

    lin_r2, a_lin, p_lin = S_lin.best_r2(y)
    poly_r2, a_poly, p_poly = S_poly.best_r2(y)
    curv = poly_r2 - lin_r2

    rng = np.random.default_rng(SEED)
    null = np.empty(N_NULL)
    for i in range(N_NULL):
        yn = null_draw(y, rng)
        null[i] = S_poly.best_r2(yn)[0] - S_lin.best_r2(yn)[0]

    rb = np.random.default_rng(SEED + 1)
    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        b = rb.integers(0, n, n)
        boot[i] = r2(y[b], p_poly[b]) - r2(y[b], p_lin[b])

    return dict(
        model=model, tissue=tissue, species=SPECIES[tissue], n_cells=n, dim=d,
        linear_r2=lin_r2, poly2_r2=poly_r2, curvature=float(curv),
        alpha_linear=a_lin, alpha_poly=a_poly,
        null=dict(p_value=float((np.sum(null >= curv) + 1) / (N_NULL + 1)),
                  null_mean=float(null.mean()), null_sd=float(null.std()),
                  null_p95=float(np.percentile(null, 95)),
                  z=float((curv - null.mean()) / (null.std() + 1e-12)),
                  above_null_p95=bool(curv > np.percentile(null, 95))),
        bootstrap=dict(mean=float(boot.mean()),
                       ci_lo=float(np.percentile(boot, 2.5)),
                       ci_hi=float(np.percentile(boot, 97.5)),
                       frac_below_zero=float(np.mean(boot <= 0))),
    )


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = [run(m, t) for t in TISSUES for m in MODELS]
    q = bh_fdr([r["null"]["p_value"] for r in rows])
    for r, qi in zip(rows, q):
        r["null"]["q_bh"] = float(qi)

    settings = dict(seed=SEED, n_folds=N_FOLDS, alphas=ALPHAS, n_null=N_NULL, n_boot=N_BOOT,
                    block=BLOCK)
    with open(os.path.join(OUT, "curvature_significance.json"), "w") as f:
        json.dump(dict(settings=settings, results={f"{r['model']}_{r['tissue']}": r for r in rows}),
                  f, indent=1)

    cols = ["model", "tissue", "species", "n_cells", "linear_r2", "poly2_r2", "curvature",
            "null_mean", "null_sd", "z", "p_value", "q_bh", "boot_ci_lo", "boot_ci_hi"]
    lines = ["\t".join(cols)]
    for r in rows:
        s, b = r["null"], r["bootstrap"]
        lines.append("\t".join([
            r["model"], r["tissue"], r["species"], str(r["n_cells"]),
            f"{r['linear_r2']:.4f}", f"{r['poly2_r2']:.4f}", f"{r['curvature']:+.4f}",
            f"{s['null_mean']:+.4f}", f"{s['null_sd']:.4f}", f"{s['z']:+.2f}",
            f"{s['p_value']:.4f}", f"{s['q_bh']:.4f}",
            f"{b['ci_lo']:+.4f}", f"{b['ci_hi']:+.4f}"]))
    with open(os.path.join(OUT, "results_table.tsv"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
