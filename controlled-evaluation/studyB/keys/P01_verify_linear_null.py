"""P01 verification: linear-truth nulls that keep the real fitted linear direction.

Not part of any package. Reads the B1448 package (data + FoldSolver) read-only.
For each cell: fit the linear kernel ridge in-sample at the CV-selected alpha (yhat, exactly
linear in X), then build synthetic targets yhat + noise, where noise is the real in-sample
residual vector, either permuted (P) or sign-flipped (W, keeps heteroscedasticity), at the
in-sample scale (P, W) or rescaled to the real CV residual variance (Ps, Ws). The statistic
(poly CV R2 - linear CV R2, same folds, same alpha grid) is computed on each target.

Run:  maxtoki-framework-eval/bin/python studyB/keys/P01_verify_linear_null.py
About 2-4 minutes on CPU, < 0.5 GB.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import sys
sys.dont_write_bytecode = True  # do not create __pycache__ inside the package

import importlib.util
import numpy as np
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..", "packages", "B1448")
spec = importlib.util.spec_from_file_location("pkg", os.path.join(PKG, "code", "curvature_significance.py"))
pkg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pkg)

N_DRAW = 200
CELLS = [f"{m}_{t}" for t in ("lung", "gut", "pancreas") for m in ("geneformer", "scgpt")]


def main():
    print("cell                 curv     | null mean (P, Ps, W, Ws)            | p (P, Ps, W, Ws)")
    allp = {k: [] for k in ("P", "Ps", "W", "Ws")}
    for name in CELLS:
        z = np.load(os.path.join(PKG, "data", f"{name}.npz"))
        X = z["emb"].astype(np.float64)
        y = z["pseudotime"].astype(np.float64)
        n = len(y)
        Xz = StandardScaler().fit_transform(X)
        Klin = (Xz @ Xz.T) / Xz.shape[1] + 1.0
        Kpoly = Klin ** 2
        Sl, Sp = pkg.FoldSolver(Klin, n), pkg.FoldSolver(Kpoly, n)
        lr, al, plin = Sl.best_r2(y)
        curv = Sp.best_r2(y)[0] - lr
        yhat = Klin @ np.linalg.solve(Klin + al * np.eye(n), y)
        res = y - yhat
        k = np.sqrt(np.var(y - plin) / np.var(res))
        rng = np.random.default_rng(0)
        null = {key: np.empty(N_DRAW) for key in allp}
        for i in range(N_DRAW):
            ep = rng.permutation(res)
            ew = res * rng.choice([-1.0, 1.0], size=n)
            for key, e in (("P", ep), ("Ps", k * ep), ("W", ew), ("Ws", k * ew)):
                ys = yhat + e
                null[key][i] = Sp.best_r2(ys)[0] - Sl.best_r2(ys)[0]
        ps = {key: (np.sum(v >= curv) + 1) / (N_DRAW + 1) for key, v in null.items()}
        for key in allp:
            allp[key].append(ps[key])
        print(f"{name:20s} {curv:+.4f} | " + " ".join(f"{null[key].mean():+.4f}" for key in allp)
              + " | " + " ".join(f"{ps[key]:.4f}" for key in allp), flush=True)
    for key, p in allp.items():
        q = pkg.bh_fdr(p)
        print(f"BH q ({key}): " + " ".join(f"{v:.4f}" for v in q) + f"  -> {int(np.sum(q < 0.05))}/6 significant")


if __name__ == "__main__":
    main()
