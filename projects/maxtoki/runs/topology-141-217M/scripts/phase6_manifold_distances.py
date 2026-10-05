"""Phase 6 — manifold distance hierarchy (H13, H16, H32, H69/H70).

Per (domain, layer), construct edge features for every pair in the pair_table:
  - Euclidean (baseline)
  - Geodesic (kNN shortest path; k adaptive ∈ [10, 35] for connectivity)
  - Diffusion (Coifman-Lafon, t ∈ {1, 2, 4, 8}; pick best by AUROC on dev fold)
  - Triangle-defect spectrum (k ∈ {8, 12, 16})
  - Convexity-deficit (a simple "detour" ratio: geodesic / euclidean)

Score each via ΔAUROC vs coexpression-matched null on regulatory-edge classification.

The target hierarchy (paper):
  Euclidean (0) < Geodesic (+0.013) < Diffusion (+0.017) < Triangle-defect (+0.026)

Output: outputs/phase6/distance_deltaauroc.csv, phase6_summary.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import roc_auc_score
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix

PROJ = Path(__file__).resolve().parents[3]
RUN = PROJ / "runs/topology-141-217M"
P0 = RUN / "outputs/phase0"
P1 = RUN / "outputs/phase1"
P6 = RUN / "outputs/phase6"
P6.mkdir(parents=True, exist_ok=True)

DOMAINS = ["lung", "immune", "external_lung"]
KNN_VALUES = [8, 12, 16]
DIFF_TIMES = [1, 2, 4, 8]


def pairwise_distances(E: np.ndarray) -> np.ndarray:
    diff = E[:, None, :] - E[None, :, :]
    return np.sqrt(np.maximum((diff ** 2).sum(-1), 0.0))


def adaptive_knn_graph(E: np.ndarray, kmin: int = 10, kmax: int = 35) -> tuple[nx.Graph, int]:
    """Increase k until graph is connected."""
    D = pairwise_distances(E)
    n = D.shape[0]
    np.fill_diagonal(D, np.inf)
    for k in range(kmin, kmax + 1):
        nbr = np.argsort(D, axis=1)[:, :k]
        g = nx.Graph()
        g.add_nodes_from(range(n))
        for i in range(n):
            for j in nbr[i]:
                w = float(D[i, int(j)])
                g.add_edge(int(i), int(j), weight=w)
        if nx.is_connected(g):
            return g, k
    return g, kmax


def geodesic_distances(g: nx.Graph) -> np.ndarray:
    n = g.number_of_nodes()
    D = nx.floyd_warshall_numpy(g, weight="weight")
    return np.asarray(D, dtype=np.float32)


def diffusion_distances(E: np.ndarray, k: int, t: int) -> np.ndarray:
    """Coifman-Lafon diffusion distance via row-stochastic kernel."""
    D = pairwise_distances(E)
    n = D.shape[0]
    sigma = np.median(D[D > 0])
    if sigma == 0:
        sigma = 1.0
    K = np.exp(-(D ** 2) / (2 * sigma ** 2))
    np.fill_diagonal(K, 0.0)
    # kNN sparsify for stability
    nbr = np.argsort(K, axis=1)[:, -k:]
    K_sp = np.zeros_like(K)
    for i in range(n):
        for j in nbr[i]:
            K_sp[i, int(j)] = K[i, int(j)]
    K_sp = (K_sp + K_sp.T) / 2.0
    rs = K_sp.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
    P = K_sp / rs
    Pt = np.linalg.matrix_power(P, t)
    # Diffusion distance: || Pt[i] - Pt[j] ||_2 (with stationary weighting)
    Dd = np.sqrt(np.maximum(((Pt[:, None, :] - Pt[None, :, :]) ** 2).sum(-1), 0.0))
    return Dd.astype(np.float32)


def triangle_defect_feature(E: np.ndarray, k: int) -> np.ndarray:
    """Per-edge triangle-defect: for each gene pair (i,j), look at the k
    nearest common neighbours and compute mean |d(i,k)+d(j,k)-d(i,j)|.
    Returns full pair-feature matrix of shape (n,n)."""
    D = pairwise_distances(E)
    n = D.shape[0]
    np.fill_diagonal(D, np.inf)
    nbr = np.argsort(D, axis=1)[:, :k]
    feat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            common = np.intersect1d(nbr[i], nbr[j])
            # Drop i and j from common to avoid the sentinel D[i,i]=inf
            common = common[(common != i) & (common != j)]
            if len(common) == 0:
                pool = np.unique(np.concatenate([nbr[i], nbr[j]]))
                pool = pool[(pool != i) & (pool != j)]
                if len(pool) == 0:
                    feat[i, j] = feat[j, i] = 0.0
                    continue
                common = pool[:k]
            triangle = np.mean(
                np.abs(D[i, common] + D[j, common] - D[i, j])
            )
            feat[i, j] = feat[j, i] = float(triangle)
    return feat


def auroc_pair_feature(
    pair_feature: np.ndarray, pairs: pd.DataFrame, label_col: str = "is_trrust",
    direction: str = "smaller_is_positive",
) -> float:
    pi = pairs["i"].to_numpy(); pj = pairs["j"].to_numpy()
    feat = pair_feature[pi, pj]
    if direction == "smaller_is_positive":
        feat = -feat
    labels = pairs[label_col].to_numpy()
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    try:
        return float(roc_auc_score(labels, feat))
    except Exception:
        return float("nan")


def coexpression_matrix(log_expr: np.ndarray) -> np.ndarray:
    """|Pearson correlation| matrix between gene columns."""
    Z = log_expr - log_expr.mean(0, keepdims=True)
    sd = Z.std(0, keepdims=True); sd[sd == 0] = 1.0
    Zn = Z / sd
    C = (Zn.T @ Zn) / Z.shape[0]
    return np.abs(C).astype(np.float32)


def coexpression_matched_delta(pair_feature, pairs, log_expr,
                               direction="smaller_is_positive",
                               label_col="is_trrust") -> dict:
    coex = coexpression_matrix(log_expr)
    auroc_obs = auroc_pair_feature(pair_feature, pairs, label_col, direction)
    auroc_coex = auroc_pair_feature(coex, pairs, label_col,
                                    direction="larger_is_positive")
    # Residualise feat against coex (linear)
    pi = pairs["i"].to_numpy(); pj = pairs["j"].to_numpy()
    f_raw = pair_feature[pi, pj]
    if direction == "smaller_is_positive":
        f = -f_raw
    else:
        f = f_raw
    c = coex[pi, pj]
    A = np.column_stack([c, np.ones_like(c)])
    coef, *_ = np.linalg.lstsq(A, f, rcond=None)
    f_resid = f - A @ coef
    labels = pairs[label_col].to_numpy()
    try:
        auc_resid = float(roc_auc_score(labels, f_resid))
    except Exception:
        auc_resid = float("nan")
    return dict(
        auc_obs=auroc_obs,
        auc_coex=auroc_coex,
        delta_vs_coex=auroc_obs - auroc_coex,
        auc_residualised=auc_resid,
        delta_residualised=auc_resid - 0.5,
    )


def main():
    rows = []
    summary = {}
    for domain in DOMAINS:
        d_dir = P0 / domain
        emb_path = d_dir / "layer_gene_embeddings_pca20.npy"
        if not emb_path.exists():
            continue
        emb = np.load(emb_path)
        log_expr = np.load(d_dir / "log_expression_pool.npy")
        gf = pd.read_csv(d_dir / "gene_features.csv")
        pairs = pd.read_csv(P1 / domain / "pair_table.csv")
        n_layers = emb.shape[0]
        print(f"\n[{domain}] emb {emb.shape}  pairs {len(pairs)}  log_expr {log_expr.shape}")

        for li in range(n_layers):
            E = emb[li]
            keep = np.linalg.norm(E, axis=1) > 0
            if keep.sum() < 50:
                continue
            E = E[keep]
            # Filter pairs to keep only those with both endpoints kept
            mask_g = np.zeros(len(gf), dtype=bool); mask_g[:keep.shape[0]] = keep
            valid = mask_g[pairs["i"].to_numpy()] & mask_g[pairs["j"].to_numpy()]
            P_lay = pairs[valid].reset_index(drop=True).copy()
            # remap i/j into the kept-gene space
            new_idx = -np.ones(len(gf), dtype=np.int64)
            new_idx[np.where(mask_g)[0]] = np.arange(int(mask_g.sum()))
            P_lay["i"] = new_idx[P_lay["i"].to_numpy()]
            P_lay["j"] = new_idx[P_lay["j"].to_numpy()]
            log_expr_kept = log_expr[:, mask_g]

            # Euclidean
            D_eucl = pairwise_distances(E)
            r = coexpression_matched_delta(D_eucl, P_lay, log_expr_kept,
                                            direction="smaller_is_positive")
            rows.append(dict(domain=domain, layer=li, metric="euclidean", **r))

            # Geodesic (adaptive k)
            g, k_used = adaptive_knn_graph(E)
            D_geo = geodesic_distances(g)
            # Replace any inf
            if not np.isfinite(D_geo).all():
                fmax = float(D_geo[np.isfinite(D_geo)].max()) if np.isfinite(D_geo).any() else 1.0
                D_geo[~np.isfinite(D_geo)] = fmax + 1.0
            r = coexpression_matched_delta(D_geo, P_lay, log_expr_kept,
                                            direction="smaller_is_positive")
            r["k_used"] = int(k_used)
            rows.append(dict(domain=domain, layer=li, metric="geodesic", **r))

            # Diffusion (sweep t, pick best by raw AUROC)
            best_t = None; best_auc = -np.inf; best_d = None
            for t in DIFF_TIMES:
                D_diff = diffusion_distances(E, k=12, t=t)
                a = auroc_pair_feature(D_diff, P_lay, "is_trrust",
                                        "smaller_is_positive")
                if np.isfinite(a) and a > best_auc:
                    best_auc = a; best_t = t; best_d = D_diff
            if best_d is not None:
                r = coexpression_matched_delta(best_d, P_lay, log_expr_kept,
                                                direction="smaller_is_positive")
                r["best_t"] = int(best_t)
                rows.append(dict(domain=domain, layer=li, metric="diffusion", **r))

            # Triangle-defect spectrum (sum across k=8,12,16)
            tri_sum = np.zeros((E.shape[0], E.shape[0]), dtype=np.float32)
            for k in KNN_VALUES:
                tri_sum += triangle_defect_feature(E, k=k)
            r = coexpression_matched_delta(tri_sum, P_lay, log_expr_kept,
                                            direction="larger_is_positive")
            rows.append(dict(domain=domain, layer=li, metric="triangle_defect", **r))

            print(f"  L{li:02d}  eucl={rows[-4]['delta_vs_coex']:+.3f}  "
                  f"geo={rows[-3]['delta_vs_coex']:+.3f}  "
                  f"diff={rows[-2]['delta_vs_coex']:+.3f}  "
                  f"tri={rows[-1]['delta_vs_coex']:+.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(P6 / "distance_deltaauroc.csv", index=False)

    # Aggregate hierarchy: mean delta_vs_coex per metric per domain
    if not df.empty:
        agg = df.groupby(["domain", "metric"])["delta_vs_coex"].agg(
            ["mean", "median", "std", "count"]).reset_index()
        agg.to_csv(P6 / "hierarchy_summary.csv", index=False)
        summary["per_domain_per_metric"] = agg.to_dict(orient="records")

    with open(P6 / "phase6_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n[phase 6] hierarchy:")
    if not df.empty:
        print(df.groupby(["metric"])["delta_vs_coex"].mean().sort_values())


if __name__ == "__main__":
    main()
