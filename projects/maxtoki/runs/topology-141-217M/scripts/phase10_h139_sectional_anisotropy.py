"""Phase 10 — H139 sectional-anisotropy spot-check.

Closes the Phase 10 "partial" item in the run README: H23 Forman direction was
already covered in phase16; this script adds the H139 sectional-anisotropy
spot-check.

H139 (source paper, iter_0052) builds 12 per-edge sectional descriptors over a
kNN graph in PCA space (anisotropy / planarity / omnivariance from local
covariance eigenvalues, plus PC1 alignment terms and an asymmetric support
margin), classifies regulatory edges with a cross-validated logistic
regression, and compares against a triangle-defect-only baseline (H70) under
three null families:

  - endpoint_swap        : per-edge feature-vector swap (src/tgt symmetry test)
  - sectional_row_shuffle: row permutation of the feature matrix
  - label_permutation    : permutation of the regulatory labels

Spot-check scope (faithful in spirit; not the full 25-row autoloop sweep):

  - 3 domains (lung, immune, external_lung) × layer 11 × seed 42
  - one split regime per domain (the existing pair_table from phase 1)
  - 8 null draws per family (matches iter_0052 H139_NULL_PERM)
  - support_dir replaced with kNN-rank asymmetry: rank of j in i's kNN list
    minus rank of i in j's kNN list (the iter_0051 reference uses a
    pipeline-internal H70/H136-derived support matrix that we do not
    reconstruct here; this is the documented spot-check simplification)

Outputs:
  outputs/phase10/h139_sectional_spot_check.csv
  outputs/phase10/phase10_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

PROJ = Path(__file__).resolve().parents[3]
RUN = PROJ / "runs/topology-141-217M"
P0 = RUN / "outputs/phase0"
P1 = RUN / "outputs/phase1"
P10 = RUN / "outputs/phase10"
P10.mkdir(parents=True, exist_ok=True)

DOMAINS = ["lung", "immune", "external_lung"]
LAYER = 11
KNN = 12
N_CV = 3
N_NULL = 8
SEED = 42


def safe_unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-10 else np.zeros_like(v)


def local_sectional_geometry(points: np.ndarray, n_neighbors: int) -> dict:
    """Per-node (anisotropy, planarity, omnivariance, PC1) from kNN covariance."""
    n_nodes = points.shape[0]
    k = max(4, min(n_neighbors, n_nodes - 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(points)
    _, idx = nbrs.kneighbors(points)

    anis = np.zeros(n_nodes)
    plan = np.zeros(n_nodes)
    omni = np.zeros(n_nodes)
    pc1 = np.zeros((n_nodes, points.shape[1]))
    for i in range(n_nodes):
        neigh = idx[i, 1:]
        if neigh.size < 3:
            continue
        block = points[neigh] - points[neigh].mean(0, keepdims=True)
        cov = (block.T @ block) / max(1, block.shape[0] - 1)
        evals, evecs = np.linalg.eigh(cov)
        evals = np.clip(evals[::-1], 1e-8, None)
        vecs = evecs[:, ::-1]
        denom = float(evals[0] + evals[1] + evals[2])
        anis[i] = float((evals[0] - evals[1]) / max(1e-8, denom))
        plan[i] = float((evals[1] - evals[2]) / max(1e-8, evals[0]))
        omni[i] = float((evals[0] * evals[1] * evals[2]) ** (1.0 / 3.0)
                        / max(1e-8, denom))
        pc1[i] = safe_unit(vecs[:, 0])
    return dict(anisotropy=anis, planarity=plan, omnivariance=omni, pc1=pc1, knn_idx=idx)


def knn_rank_support(knn_idx: np.ndarray) -> np.ndarray:
    """support_dir[i,j] = (1 + position of j in i's neighbour list)^-1, 0 if absent.

    Asymmetric, scale-free spot-check stand-in for the paper's H70/H136-derived
    support matrix. Closer to i ⇒ higher support; absent ⇒ 0. Then
    support_margin in the edge feature reads (rank^-1 from i to j) minus
    (rank^-1 from j to i)."""
    n = knn_idx.shape[0]
    sup = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        # column 0 is i itself; skip it
        for r, j in enumerate(knn_idx[i, 1:], start=1):
            sup[i, int(j)] = 1.0 / r
    return sup


def sectional_edge_features(points: np.ndarray, src: np.ndarray, tgt: np.ndarray,
                            support_dir: np.ndarray, node_geo: dict) -> np.ndarray:
    """12-dim per-edge sectional feature, faithful port of iter_0051."""
    anis = node_geo["anisotropy"]
    plan = node_geo["planarity"]
    omni = node_geo["omnivariance"]
    pc1 = node_geo["pc1"]

    src = np.asarray(src, dtype=int)
    tgt = np.asarray(tgt, dtype=int)
    edge_vec = points[tgt] - points[src]
    edge_u = np.vstack([safe_unit(v) for v in edge_vec])

    src_align = np.abs(np.sum(pc1[src] * edge_u, axis=1))
    tgt_align = np.abs(np.sum(pc1[tgt] * (-edge_u), axis=1))
    pair_align = np.abs(np.sum(pc1[src] * pc1[tgt], axis=1))
    support_margin = support_dir[src, tgt] - support_dir[tgt, src]

    anis_src = anis[src]
    anis_tgt = anis[tgt]
    anis_gap = np.abs(anis_src - anis_tgt)
    anis_mean = 0.5 * (anis_src + anis_tgt)
    feat = np.column_stack([
        anis_src, anis_tgt, anis_gap, anis_mean,
        0.5 * (plan[src] + plan[tgt]),
        0.5 * (omni[src] + omni[tgt]),
        support_margin,
        src_align, tgt_align, pair_align,
        anis_gap * support_margin,
        anis_mean * pair_align,
    ])
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)


def swapped_sectional_features(features: np.ndarray) -> np.ndarray:
    """Endpoint-symmetry null: faithful port of iter_0051's column re-shuffle."""
    x = np.asarray(features, dtype=float)
    out = x.copy()
    out[:, 0] = x[:, 1]
    out[:, 1] = x[:, 0]
    out[:, 6] = -x[:, 6]
    out[:, 7] = x[:, 8]
    out[:, 8] = x[:, 7]
    out[:, 10] = -x[:, 10]
    return out


def cv_auroc(X: np.ndarray, y: np.ndarray, seed: int = SEED) -> float:
    if len(np.unique(y)) < 2 or X.shape[0] < 12:
        return float("nan")
    skf = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=seed)
    preds = np.zeros_like(y, dtype=float)
    for tr, te in skf.split(X, y):
        scaler = StandardScaler().fit(X[tr])
        Xt = scaler.transform(X[tr])
        Xv = scaler.transform(X[te])
        model = LogisticRegression(max_iter=200, solver="liblinear")
        model.fit(Xt, y[tr])
        preds[te] = model.predict_proba(Xv)[:, 1]
    try:
        return float(roc_auc_score(y, preds))
    except Exception:
        return float("nan")


def triangle_defect_pair_feature(points: np.ndarray, src: np.ndarray,
                                 tgt: np.ndarray, knn_idx: np.ndarray) -> np.ndarray:
    """Per-edge triangle-defect: mean |d(i,k)+d(j,k)-d(i,j)| over common kNN of i and j."""
    n = points.shape[0]
    diff = points[:, None, :] - points[None, :, :]
    D = np.sqrt(np.maximum((diff ** 2).sum(-1), 0.0))
    feat = np.zeros(len(src), dtype=np.float32)
    for e in range(len(src)):
        i = int(src[e]); j = int(tgt[e])
        common = np.intersect1d(knn_idx[i, 1:], knn_idx[j, 1:])
        common = common[(common != i) & (common != j)]
        if common.size == 0:
            pool = np.unique(np.concatenate([knn_idx[i, 1:], knn_idx[j, 1:]]))
            pool = pool[(pool != i) & (pool != j)]
            if pool.size == 0:
                continue
            common = pool[: KNN]
        feat[e] = float(np.mean(np.abs(D[i, common] + D[j, common] - D[i, j])))
    return feat


def run_one_domain(domain: str, rng: np.random.Generator) -> dict:
    d_dir = P0 / domain
    emb = np.load(d_dir / "layer_gene_embeddings_pca20.npy")  # (n_layers, n_genes, 20)
    pairs = pd.read_csv(P1 / domain / "pair_table.csv")

    E = emb[LAYER]
    keep = np.linalg.norm(E, axis=1) > 0
    if keep.sum() < 50:
        return dict(domain=domain, status="skipped_low_coverage")
    E = E[keep]
    n_genes_full = emb.shape[1]
    new_idx = -np.ones(n_genes_full, dtype=np.int64)
    new_idx[np.where(keep)[0]] = np.arange(int(keep.sum()))
    valid = (new_idx[pairs["i"].to_numpy()] >= 0) & (new_idx[pairs["j"].to_numpy()] >= 0)
    P_lay = pairs[valid].reset_index(drop=True).copy()
    P_lay["i"] = new_idx[P_lay["i"].to_numpy()]
    P_lay["j"] = new_idx[P_lay["j"].to_numpy()]

    src = P_lay["i"].to_numpy()
    tgt = P_lay["j"].to_numpy()
    y = P_lay["is_trrust"].to_numpy().astype(int)
    n_pairs = len(y)
    n_pos = int(y.sum())
    if n_pos < 6 or n_pos > n_pairs - 6:
        return dict(domain=domain, status="skipped_label_imbalance",
                    n_pairs=n_pairs, n_pos=n_pos)

    print(f"[{domain}] L{LAYER}  n_genes={E.shape[0]}  n_pairs={n_pairs}  n_pos={n_pos}")
    node_geo = local_sectional_geometry(E, n_neighbors=KNN)
    support = knn_rank_support(node_geo["knn_idx"])
    sec_feat = sectional_edge_features(E, src, tgt, support, node_geo)

    auc_sec = cv_auroc(sec_feat, y)
    tri_feat = triangle_defect_pair_feature(E, src, tgt, node_geo["knn_idx"])
    auc_tri = cv_auroc(tri_feat[:, None], y)
    delta_vs_h70 = auc_sec - auc_tri

    null_swap = np.empty(N_NULL)
    null_rowshuf = np.empty(N_NULL)
    null_label = np.empty(N_NULL)
    for p in range(N_NULL):
        null_swap[p] = cv_auroc(swapped_sectional_features(sec_feat), y, seed=SEED + p)
        perm = rng.permutation(sec_feat.shape[0])
        null_rowshuf[p] = cv_auroc(sec_feat[perm], y, seed=SEED + p)
        y_perm = rng.permutation(y)
        null_label[p] = cv_auroc(sec_feat, y_perm, seed=SEED + p)
    null_swap_q95 = float(np.nanquantile(null_swap, 0.95))
    null_rowshuf_q95 = float(np.nanquantile(null_rowshuf, 0.95))
    null_label_q95 = float(np.nanquantile(null_label, 0.95))
    null_max_q95 = max(null_swap_q95, null_rowshuf_q95, null_label_q95)

    return dict(
        domain=domain, status="ok", layer=LAYER, n_pairs=n_pairs, n_pos=n_pos,
        auc_sec=auc_sec, auc_tri_h70=auc_tri, delta_vs_h70=delta_vs_h70,
        null_swap_mean=float(np.nanmean(null_swap)),
        null_swap_q95=null_swap_q95,
        null_rowshuf_mean=float(np.nanmean(null_rowshuf)),
        null_rowshuf_q95=null_rowshuf_q95,
        null_label_mean=float(np.nanmean(null_label)),
        null_label_q95=null_label_q95,
        null_max_q95=null_max_q95,
        null_gap=auc_sec - null_max_q95,
        strict_positive=bool(auc_sec - null_max_q95 > 0),
        directional_positive=bool(delta_vs_h70 > 0),
    )


def main():
    rng = np.random.default_rng(SEED)
    rows = [run_one_domain(d, rng) for d in DOMAINS]
    df = pd.DataFrame(rows)
    df.to_csv(P10 / "h139_sectional_spot_check.csv", index=False)
    summary = {
        "scope": dict(layer=LAYER, n_cv=N_CV, n_null=N_NULL, knn=KNN, seed=SEED,
                       feature_dim=12, support_dir="knn_rank_asymmetry"),
        "per_domain": rows,
        "aggregate": {
            "n_domains": int((df["status"] == "ok").sum()) if "status" in df else 0,
            "mean_auc_sec": float(df.get("auc_sec", pd.Series(dtype=float)).mean()),
            "mean_auc_tri_h70": float(df.get("auc_tri_h70", pd.Series(dtype=float)).mean()),
            "mean_delta_vs_h70": float(df.get("delta_vs_h70", pd.Series(dtype=float)).mean()),
            "mean_null_gap": float(df.get("null_gap", pd.Series(dtype=float)).mean()),
            "n_directional_positive": int(df.get("directional_positive", pd.Series(dtype=bool)).sum()),
            "n_strict_positive": int(df.get("strict_positive", pd.Series(dtype=bool)).sum()),
        },
    }
    with open(P10 / "phase10_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
