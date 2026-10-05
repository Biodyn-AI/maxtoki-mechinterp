"""Phase 16 — combined exploratory hypothesis screen.

Tests in one pass:
  H-lineage : cell-type marker clustering (B/T/NK/Myeloid lineages) — do
              canonical markers cluster tighter than random gene sets in
              each layer's kNN graph?
  H-depth   : layer-wise H1 persistence trajectory + intrinsic dimensionality
              by layer (twoNN estimator).
  H-hub     : TF-degree centrality alignment — do high-TRRUST-degree TFs sit
              at higher betweenness/eigenvector centrality in the kNN graph?
  H23 sanity: Forman curvature directional check — high-curvature edges
              should be LESS likely to be regulatory (paper's expected sig).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import roc_auc_score
from ripser import ripser

PROJ = Path(__file__).resolve().parents[3]
RUN = PROJ / "runs/topology-141-217M"
P0 = RUN / "outputs/phase0"
P1 = RUN / "outputs/phase1"
P16 = RUN / "outputs/phase16_extended"
P16.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(RUN / "scripts"))
from phase6_manifold_distances import pairwise_distances
from phase123_splits_nulls import load_trrust_signed

DOMAINS = ["lung", "immune", "external_lung"]
KNN = 12

# Canonical cell-type lineage markers (immune-focused; use HUGO symbols)
LINEAGE_MARKERS = {
    "T_cell": ["CD3D", "CD3E", "CD3G", "CD4", "CD8A", "CD8B", "LCK", "ZAP70",
               "TCF7", "LEF1"],
    "B_cell": ["CD19", "CD79A", "CD79B", "MS4A1", "PAX5", "BLK", "VPREB3",
               "FCRL1"],
    "NK_cell": ["NCAM1", "KLRD1", "KLRF1", "NKG7", "GNLY", "GZMB", "PRF1",
                "FCGR3A"],
    "Myeloid": ["CD14", "CD68", "CD163", "MRC1", "MERTK", "LYZ", "VCAN",
                "S100A8", "S100A9"],
    "Dendritic": ["CLEC9A", "CLEC10A", "CD1C", "FCER1A", "ITGAX"],
    "Granulocyte": ["ELANE", "MPO", "AZU1", "DEFA3", "CSF3R"],
}


def lineage_purity(emb: np.ndarray, syms: np.ndarray, marker_set: list[str],
                    knn: int = KNN, n_null: int = 200, seed: int = 42) -> dict:
    """For markers that are in `syms`, compute the within-lineage Mean Reciprocal
    Rank: for each marker, what fraction of its k nearest neighbors are also
    markers? Compare to random gene-sets of the same size.
    """
    rng = np.random.default_rng(seed)
    sym_set = set(s.upper() for s in syms)
    marker_in = [s for s in marker_set if s.upper() in sym_set]
    if len(marker_in) < 3:
        return {"error": f"only {len(marker_in)} markers in pool"}
    sym_to_i = {s.upper(): i for i, s in enumerate(syms)}
    marker_idx = np.array([sym_to_i[s.upper()] for s in marker_in])
    D = pairwise_distances(emb)
    np.fill_diagonal(D, np.inf)
    nbr = np.argsort(D, axis=1)[:, :knn]
    # purity: for each marker, fraction of its kNN that are also markers
    marker_set_idx = set(marker_idx.tolist())
    obs = np.array([np.mean([1 if int(j) in marker_set_idx else 0
                              for j in nbr[i]]) for i in marker_idx]).mean()
    # Null: random gene sets of same size
    null = []
    n_genes = len(syms)
    for k in range(n_null):
        rand_idx = rng.choice(n_genes, size=len(marker_idx), replace=False)
        rand_set = set(rand_idx.tolist())
        null.append(np.array([np.mean([1 if int(j) in rand_set else 0
                                         for j in nbr[i]]) for i in rand_idx]).mean())
    null = np.array(null)
    return {
        "n_markers_in_pool": len(marker_in),
        "obs_purity": float(obs),
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "null_p95": float(np.percentile(null, 95)),
        "z": float((obs - null.mean()) / (null.std() + 1e-9)),
        "delta": float(obs - null.mean()),
    }


def twonn_intrinsic_dim(emb: np.ndarray) -> float:
    """Two-nearest-neighbor intrinsic dimension estimator (Facco 2017)."""
    n = emb.shape[0]
    D = pairwise_distances(emb)
    np.fill_diagonal(D, np.inf)
    sorted_D = np.sort(D, axis=1)
    r1 = sorted_D[:, 0]; r2 = sorted_D[:, 1]
    valid = (r1 > 0) & (r2 > 0)
    mu = r2[valid] / r1[valid]
    mu = mu[mu > 1.001]
    if len(mu) < 5:
        return float("nan")
    F = np.arange(1, len(mu) + 1) / (len(mu) + 1)
    F = F[np.argsort(mu)]
    sorted_mu = np.sort(mu)
    # Fit -log(1 - F) ~ d * log(mu)
    y = -np.log(1 - F + 1e-12)
    x = np.log(sorted_mu)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 5:
        return float("nan")
    d = float(np.polyfit(x[valid], y[valid], 1)[0])
    return d


def total_h1(emb: np.ndarray) -> float:
    res = ripser(emb, maxdim=1, n_perm=None)
    h1 = res["dgms"][1]
    if h1.size == 0:
        return 0.0
    h1 = h1[np.isfinite(h1[:, 1])]
    if h1.size == 0:
        return 0.0
    return float((h1[:, 1] - h1[:, 0]).sum())


def tf_degree_centrality(emb: np.ndarray, syms: np.ndarray,
                          trrust: pd.DataFrame, knn: int = KNN,
                          n_perm: int = 100, seed: int = 42) -> dict:
    """For each gene in pool, compute its TRRUST out-degree (TF→targets)
    and its eigenvector + betweenness centrality in the kNN graph.
    Test: Spearman correlation between TF-degree and centrality, vs
    permutation null shuffling TF-degree across genes.
    """
    n = emb.shape[0]
    sym_to_i = {s.upper(): i for i, s in enumerate(syms)}
    tf_count = np.zeros(n, dtype=np.int32)
    for tf, targets in trrust.groupby("tf"):
        if tf.upper() in sym_to_i:
            tf_count[sym_to_i[tf.upper()]] = int(len(targets))
    if tf_count.max() == 0:
        return {"error": "no TFs in pool"}

    # Build kNN graph
    D = pairwise_distances(emb)
    np.fill_diagonal(D, np.inf)
    nbr = np.argsort(D, axis=1)[:, :knn]
    g = nx.Graph(); g.add_nodes_from(range(n))
    for i in range(n):
        for j in nbr[i]:
            g.add_edge(int(i), int(j))
    # Use eigenvector centrality (often more stable than betweenness for kNN)
    try:
        ec = nx.eigenvector_centrality_numpy(g)
    except Exception:
        ec = nx.degree_centrality(g)
    cent = np.array([ec.get(i, 0) for i in range(n)])
    rs_obs = float(pd.Series(tf_count).corr(pd.Series(cent), method="spearman"))

    rng = np.random.default_rng(seed)
    null = []
    for k in range(n_perm):
        perm = rng.permutation(n)
        null.append(float(pd.Series(tf_count[perm]).corr(pd.Series(cent), method="spearman")))
    null = np.array(null)
    return {
        "n_tfs_in_pool": int((tf_count > 0).sum()),
        "spearman_obs": rs_obs,
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "z": float((rs_obs - null.mean()) / (null.std() + 1e-9)),
        "p95_null": float(np.percentile(null, 95)),
    }


def forman_curvature_check(
    emb: np.ndarray, pairs: pd.DataFrame, knn: int = KNN,
) -> dict:
    """Per source paper H23: high Forman curvature edges should be LESS
    likely to be regulatory (AUROC 0.34-0.39 — i.e., curvature predicts
    NON-regulatory edges).

    Forman curvature for an edge (u,v) in unweighted graph:
        F(u,v) = 2 - deg(u) - deg(v) + Σ neighbours
    We use a simplified Forman: F = 2 - deg(u) - deg(v) + |N(u) ∩ N(v)|.
    Higher F = "less curved" / less hub-like.
    """
    n = emb.shape[0]
    D = pairwise_distances(emb)
    np.fill_diagonal(D, np.inf)
    nbr = np.argsort(D, axis=1)[:, :knn]
    g = nx.Graph(); g.add_nodes_from(range(n))
    for i in range(n):
        for j in nbr[i]:
            g.add_edge(int(i), int(j))
    # Build per-pair Forman feature matrix
    F = np.zeros((n, n), dtype=np.float32)
    deg = dict(g.degree())
    for u, v in g.edges():
        common = len(set(g.neighbors(u)) & set(g.neighbors(v)))
        f_uv = 2 - deg[u] - deg[v] + common
        F[u, v] = F[v, u] = f_uv
    # AUROC of F on TRRUST regulatory pairs (only over edges that exist)
    pi = pairs["i"].to_numpy(); pj = pairs["j"].to_numpy()
    feat = F[pi, pj]
    # Restrict to pairs that are graph edges (otherwise F is 0)
    has_edge = np.array([g.has_edge(int(a), int(b)) for a, b in zip(pi, pj)])
    if has_edge.sum() < 10:
        return {"error": "too few edges in graph"}
    feat_e = feat[has_edge]
    y_e = pairs["is_trrust"].to_numpy()[has_edge]
    if y_e.sum() == 0 or y_e.sum() == len(y_e):
        return {"error": "degenerate labels on edge subset"}
    # Test direction: paper says higher F (less curved) → more regulatory? OR
    # lower F → more regulatory? Their test reports AUROC 0.34-0.39 for "high
    # curvature → regulatory", meaning AUROC < 0.5 is the EXPECTED direction.
    auroc = float(roc_auc_score(y_e, feat_e))
    auroc_inv = float(roc_auc_score(y_e, -feat_e))
    return {
        "n_pair_edges_in_graph": int(has_edge.sum()),
        "auroc_high_curvature_is_regulatory": auroc,
        "auroc_inverted": auroc_inv,
        "expected_paper": "0.34–0.39 (below chance)",
        "directional_replicates": bool(auroc < 0.45),
    }


def main():
    trrust = load_trrust_signed()
    lineage_rows = []
    depth_rows = []
    hub_rows = []
    forman_rows = []
    for domain in DOMAINS:
        emb_path = P0 / domain / "layer_gene_embeddings_pca20.npy"
        emb = np.load(emb_path)
        gf = pd.read_csv(P0 / domain / "gene_features.csv")
        pairs = pd.read_csv(P1 / domain / "pair_table.csv")
        syms = gf["symbol"].astype(str).to_numpy()
        n_layers = emb.shape[0]
        print(f"\n[{domain}] running H-lineage + H-depth + H-hub + H23 on emb {emb.shape}")

        # H-lineage on layer L6 (mid-depth) per domain
        layer_for_lineage = min(6, n_layers - 1)
        E = emb[layer_for_lineage]
        keep = np.linalg.norm(E, axis=1) > 0
        E = E[keep]; syms_kept = syms[keep]
        for lin, markers in LINEAGE_MARKERS.items():
            r = lineage_purity(E, syms_kept, markers)
            r.update(domain=domain, layer=layer_for_lineage, lineage=lin)
            lineage_rows.append(r)
            if "obs_purity" in r:
                print(f"  L{layer_for_lineage} {lin:>12s}  purity={r['obs_purity']:.3f}  "
                      f"null={r['null_mean']:.3f}  z={r['z']:+.2f}  "
                      f"({r['n_markers_in_pool']} markers)")

        # H-depth: H1 persistence + intrinsic dim per layer
        for li in range(n_layers):
            E = emb[li]
            keep = np.linalg.norm(E, axis=1) > 0
            E = E[keep]
            d = twonn_intrinsic_dim(E)
            h1 = total_h1(E)
            depth_rows.append(dict(domain=domain, layer=li, twonn_dim=d, total_h1=h1))
        print(f"  depth twonn dims: " +
              " ".join(f"{r['twonn_dim']:.1f}" for r in depth_rows[-n_layers:]))
        print(f"  depth h1       : " +
              " ".join(f"{r['total_h1']:.0f}" for r in depth_rows[-n_layers:]))

        # H-hub: TF-degree vs centrality at L6
        E = emb[layer_for_lineage]
        keep = np.linalg.norm(E, axis=1) > 0
        E = E[keep]; syms_kept = syms[keep]
        r = tf_degree_centrality(E, syms_kept, trrust)
        r.update(domain=domain, layer=layer_for_lineage)
        hub_rows.append(r)
        if "spearman_obs" in r:
            print(f"  H-hub TF-deg vs centrality: ρ={r['spearman_obs']:+.3f}  "
                  f"null={r['null_mean']:+.3f}  z={r['z']:+.2f}")

        # H23 Forman curvature direction
        E = emb[layer_for_lineage]
        keep = np.linalg.norm(E, axis=1) > 0
        E = E[keep]
        # filter pairs to kept genes
        valid = keep[pairs["i"].to_numpy()] & keep[pairs["j"].to_numpy()]
        new_idx = -np.ones(len(gf), dtype=np.int64)
        new_idx[np.where(keep)[0]] = np.arange(int(keep.sum()))
        P_lay = pairs[valid].reset_index(drop=True).copy()
        P_lay["i"] = new_idx[P_lay["i"].to_numpy()]
        P_lay["j"] = new_idx[P_lay["j"].to_numpy()]
        r = forman_curvature_check(E, P_lay)
        r.update(domain=domain, layer=layer_for_lineage)
        forman_rows.append(r)
        if "auroc_high_curvature_is_regulatory" in r:
            print(f"  H23 Forman: AUROC={r['auroc_high_curvature_is_regulatory']:.3f}  "
                  f"(paper expects 0.34–0.39 if direction replicates)  "
                  f"replicates_direction={r['directional_replicates']}")

    pd.DataFrame(lineage_rows).to_csv(P16 / "h_lineage_marker_purity.csv", index=False)
    pd.DataFrame(depth_rows).to_csv(P16 / "h_depth_persistence_dim.csv", index=False)
    pd.DataFrame(hub_rows).to_csv(P16 / "h_hub_tf_degree.csv", index=False)
    pd.DataFrame(forman_rows).to_csv(P16 / "h23_forman.csv", index=False)

    summary = {}
    # Lineage
    df = pd.DataFrame(lineage_rows)
    df_ok = df.dropna(subset=["z"]) if "z" in df.columns else df
    if not df_ok.empty:
        per_domain = {}
        for dom, sub in df_ok.groupby("domain"):
            sub = sub[sub["z"].notna()]
            per_domain[dom] = dict(
                n_lineages_tested=int(len(sub)),
                n_lineages_significant=int((sub["z"] > 1.96).sum()),
                mean_z=float(sub["z"].mean()),
                top_lineage=str(sub.loc[sub["z"].idxmax(), "lineage"])
                            if not sub.empty and sub["z"].notna().any() else "none",
                top_z=float(sub["z"].max()),
            )
        summary["lineage"] = per_domain

    # Depth
    df = pd.DataFrame(depth_rows)
    if not df.empty:
        per_domain = {}
        for dom, sub in df.groupby("domain"):
            per_domain[dom] = dict(
                twonn_dim_per_layer=[float(d) for d in sub["twonn_dim"].tolist()],
                total_h1_per_layer=[float(h) for h in sub["total_h1"].tolist()],
                dim_compression_l0_to_l_last=(
                    float(sub.iloc[0]["twonn_dim"] / sub.iloc[-1]["twonn_dim"])
                    if sub.iloc[-1]["twonn_dim"] != 0 and np.isfinite(sub.iloc[-1]["twonn_dim"])
                    else float("nan")
                ),
            )
        summary["depth"] = per_domain

    # Hub
    df = pd.DataFrame(hub_rows)
    df_ok = df[df["z"].notna()] if "z" in df.columns else pd.DataFrame()
    if not df_ok.empty:
        per_domain = {}
        for dom, sub in df_ok.groupby("domain"):
            r = sub.iloc[0]
            per_domain[dom] = dict(
                spearman_obs=float(r["spearman_obs"]),
                z=float(r["z"]),
                significant=bool(abs(r["z"]) > 1.96),
            )
        summary["hub"] = per_domain

    # Forman
    df = pd.DataFrame(forman_rows)
    if "auroc_high_curvature_is_regulatory" in df.columns:
        per_domain = {}
        for dom, sub in df.groupby("domain"):
            r = sub.iloc[0]
            per_domain[dom] = dict(
                auroc_high_curvature=float(r["auroc_high_curvature_is_regulatory"]),
                replicates_paper_direction=bool(r["directional_replicates"]),
            )
        summary["forman"] = per_domain

    with open(P16 / "phase16_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n[phase 16] summary:")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
