"""Phase 1 — Global spectral decomposition.

Per-layer SVD of the gene-embedding matrix from Phase 0; effective rank,
TwoNN intrinsic dimensionality, participation ratio, SV1 variance fraction,
plus the feature-shuffle null (sparsity-artefact control).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[3]
PHASE0 = PROJ / "runs/spectral-geometry-217M/outputs/phase0"
OUT = PROJ / "runs/spectral-geometry-217M/outputs/phase1"
OUT.mkdir(parents=True, exist_ok=True)

SEED = int(os.environ.get("PHASE1_SEED", "42"))
TOPK_SV = int(os.environ.get("PHASE1_TOPK_SV", "7"))

print("=" * 70)
print("PHASE 1 — Global spectral decomposition")
print("=" * 70)

E = np.load(PHASE0 / "layer_gene_embeddings.npy")  # (L, G, H)
n_layers, n_genes, hidden = E.shape
counts = np.load(PHASE0 / "gene_counts.npy")
nonzero_genes = counts > 0
print(f"  input: {E.shape}  non-zero genes: {int(nonzero_genes.sum())}/{n_genes}")


def _two_nn(X, seed=0):
    """Facco 2017 TwoNN intrinsic dimensionality."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n < 4:
        return float("nan")
    from scipy.spatial.distance import cdist
    idx = rng.choice(n, size=min(n, 500), replace=False)
    D = cdist(X[idx], X)
    D.sort(axis=1)
    r1 = D[:, 1]
    r2 = D[:, 2]
    ok = (r1 > 1e-9) & (r2 > r1)
    mu = r2[ok] / r1[ok]
    mu.sort()
    # CDF fit
    F = (np.arange(1, len(mu) + 1)) / len(mu)
    mask = F < 0.9
    if mask.sum() < 10:
        return float("nan")
    x = np.log(mu[mask])
    y = -np.log(1.0 - F[mask] + 1e-12)
    d, _ = np.polyfit(x, y, 1)
    return float(d)


def _effective_rank(sigmas):
    p = sigmas ** 2
    p = p / max(p.sum(), 1e-12)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def _participation_ratio(sigmas):
    s2 = sigmas ** 2
    return float(s2.sum() ** 2 / max((s2 ** 2).sum(), 1e-12))


def _svd_and_metrics(M, seed):
    """M: (n_genes, hidden). Mean-centre columns, SVD, compute dimensionality metrics."""
    mu = M.mean(axis=0, keepdims=True)
    Mc = M - mu
    U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    out = {
        "n_genes": int(M.shape[0]),
        "effective_rank": _effective_rank(S),
        "participation_ratio": _participation_ratio(S),
        "sv1_variance_fraction": float(S[0] ** 2 / (S ** 2).sum()),
        "svd_sigmas": S.tolist(),
    }
    out["twonn_intrinsic_dim"] = _two_nn(Mc, seed=seed)
    return out, U, S, Vt


# Run per-layer on non-zero-gene subset
print("\n[1] Per-layer SVD + metrics...")
per_layer_metrics = []
per_layer_sv = []  # spectral coords of each gene on SV1..SVk
for li in range(n_layers):
    M = E[li][nonzero_genes]
    m, U, S, Vt = _svd_and_metrics(M, seed=SEED + li)
    m["layer"] = li
    per_layer_metrics.append(m)
    # Spectral coordinates: U[:, 0..TOPK_SV] * S[0..TOPK_SV]
    k = min(TOPK_SV, U.shape[1])
    coords = U[:, :k] * S[:k][None, :]     # (n_genes_nz, TOPK_SV)
    per_layer_sv.append(coords)
    print(f"  L{li:2d}: ER={m['effective_rank']:7.3f}  PR={m['participation_ratio']:7.3f}  "
          f"SV1%={m['sv1_variance_fraction']:.3f}  TwoNN={m['twonn_intrinsic_dim']:.2f}")

df_metrics = pd.DataFrame(per_layer_metrics)
df_metrics[["layer", "n_genes", "effective_rank", "participation_ratio",
            "sv1_variance_fraction", "twonn_intrinsic_dim"]].to_csv(
    OUT / "per_layer_metrics.csv", index=False
)

sv_array = np.stack(per_layer_sv, axis=0)  # (L, n_nonzero, TOPK_SV)
np.save(OUT / "gene_svd_coords.npy", sv_array.astype(np.float32))

# Depth correlations
from scipy.stats import spearmanr
layers = np.arange(n_layers)
er_vals = np.array([m["effective_rank"] for m in per_layer_metrics])
pr_vals = np.array([m["participation_ratio"] for m in per_layer_metrics])
sv1_vals = np.array([m["sv1_variance_fraction"] for m in per_layer_metrics])
twonn_vals = np.array([m["twonn_intrinsic_dim"] for m in per_layer_metrics])

rho_er, p_er = spearmanr(layers, er_vals)
rho_pr, p_pr = spearmanr(layers, pr_vals)
rho_sv1, p_sv1 = spearmanr(layers, sv1_vals)
rho_2nn, p_2nn = spearmanr(layers, twonn_vals)
print(f"\n  depth correlations:")
print(f"    effective_rank: ρ={rho_er:+.3f}  p={p_er:.2e}")
print(f"    participation_ratio: ρ={rho_pr:+.3f}  p={p_pr:.2e}")
print(f"    sv1_variance_fraction: ρ={rho_sv1:+.3f}  p={p_sv1:.2e}")
print(f"    twonn_intrinsic_dim: ρ={rho_2nn:+.3f}  p={p_2nn:.2e}")

# ------------------------------------------------------------------------
# Feature-shuffle null (sparsity-artefact control)
# ------------------------------------------------------------------------
print("\n[2] Feature-shuffle null on final layer...")
rng = np.random.default_rng(SEED + 777)
M_last = E[-1][nonzero_genes].copy()
M_shuf = M_last.copy()
for j in range(hidden):
    rng.shuffle(M_shuf[:, j])
m_shuf, _, _, _ = _svd_and_metrics(M_shuf, seed=SEED + 999)
print(f"  observed L{n_layers-1}: ER={er_vals[-1]:.3f}")
print(f"  feature-shuffle: ER={m_shuf['effective_rank']:.3f}  "
      f"(expected near the dimension count; the compression is learned, not architectural)")

with open(OUT / "null_feature_shuffle.json", "w") as f:
    json.dump({
        "observed_effective_rank_final_layer": float(er_vals[-1]),
        "feature_shuffle_effective_rank_final_layer": m_shuf["effective_rank"],
        "ratio_shuffle_vs_observed": m_shuf["effective_rank"] / max(er_vals[-1], 1e-6),
    }, f, indent=2)

with open(OUT / "run_config.json", "w") as f:
    json.dump({
        "n_layers": n_layers, "n_genes": n_genes, "hidden_size": hidden,
        "n_nonzero_genes": int(nonzero_genes.sum()),
        "topk_sv": TOPK_SV,
        "depth_correlations": {
            "effective_rank": {"rho": float(rho_er), "p": float(p_er)},
            "participation_ratio": {"rho": float(rho_pr), "p": float(p_pr)},
            "sv1_variance_fraction": {"rho": float(rho_sv1), "p": float(p_sv1)},
            "twonn_intrinsic_dim": {"rho": float(rho_2nn), "p": float(p_2nn)},
        },
    }, f, indent=2)

print(f"\nPHASE 1 COMPLETE — outputs: {OUT}")
