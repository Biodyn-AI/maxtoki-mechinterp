"""Phase 7+8 — community structure (H16) and signed motif-community hardening
(H116, H123 — the strongest finding in the source paper).

Per (domain, layer):
  - Build kNN graph at k=12 on PCA(20) embedding
  - Run Louvain communities
  - H16: AUROC of community co-membership for predicting TRRUST regulatory pairs
  - H116: TRRUST sign-motif × community alignment, vs sign-shuffle null
  - H123: as H116 but with TF-identity-preserving sign shuffle + TF/target degree strata

Output: outputs/phase78/{h16,h116,h123}.csv + phase78_summary.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import roc_auc_score

PROJ = Path(__file__).resolve().parents[3]
RUN = PROJ / "runs/topology-141-217M"
P0 = RUN / "outputs/phase0"
P1 = RUN / "outputs/phase1"
P78 = RUN / "outputs/phase78"
P78.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(RUN / "scripts"))
from phase123_splits_nulls import load_trrust_signed
from phase6_manifold_distances import (
    pairwise_distances, triangle_defect_feature, coexpression_matrix,
)

DOMAINS = ["lung", "immune", "external_lung"]
KNN = 12
N_NULL = 100


def louvain_communities(g: nx.Graph, seed: int = 42) -> np.ndarray:
    """Returns array of length n_nodes with community labels."""
    communities = nx.community.louvain_communities(g, seed=seed)
    labels = np.zeros(g.number_of_nodes(), dtype=np.int32)
    for ci, comm in enumerate(communities):
        for n in comm:
            labels[int(n)] = ci
    return labels


def comembership_matrix(labels: np.ndarray) -> np.ndarray:
    return (labels[:, None] == labels[None, :]).astype(np.float32)


def h16_score(emb: np.ndarray, pairs: pd.DataFrame, k: int = 12) -> dict:
    D = pairwise_distances(emb)
    n = D.shape[0]
    np.fill_diagonal(D, np.inf)
    nbr = np.argsort(D, axis=1)[:, :k]
    g = nx.Graph(); g.add_nodes_from(range(n))
    for i in range(n):
        for j in nbr[i]:
            g.add_edge(int(i), int(j))
    labels = louvain_communities(g, seed=42)
    com = comembership_matrix(labels)
    pi = pairs["i"].to_numpy(); pj = pairs["j"].to_numpy()
    feat = com[pi, pj]
    y = pairs["is_trrust"].to_numpy()
    if y.sum() == 0 or y.sum() == len(y):
        return dict(auc=float("nan"), n_communities=int(labels.max() + 1))
    auc = float(roc_auc_score(y, feat))
    return dict(auc=auc, n_communities=int(labels.max() + 1))


def signed_motif_community_score(
    emb: np.ndarray,
    pairs: pd.DataFrame,
    sym_to_idx: dict[str, int],
    trrust: pd.DataFrame,
    triangle_baseline: np.ndarray,
    k: int = 12,
    null_seed_base: int = 100,
    n_null: int = N_NULL,
    use_degree_strata: bool = False,
) -> dict:
    """For each gene pair, compute: do they share a TF whose annotated sign is
    consistent with their geometric placement (same community = activation,
    different community = repression)?

    Then test whether this signed-motif × community feature predicts TRRUST
    regulatory edges *over and above* the triangle-defect baseline.

    Null: TF-identity-preserving sign shuffles (preserve the TF→target graph,
    permute the sign labels). H123 adds: stratify the null by TF-degree and
    target-degree.
    """
    D = pairwise_distances(emb)
    n = D.shape[0]
    np.fill_diagonal(D, np.inf)
    nbr = np.argsort(D, axis=1)[:, :k]
    g = nx.Graph(); g.add_nodes_from(range(n))
    for i in range(n):
        for j in nbr[i]:
            g.add_edge(int(i), int(j))
    labels = louvain_communities(g, seed=42)
    com = comembership_matrix(labels)

    # Build per-pair signed-motif feature:
    #   For each TRRUST edge (TF→target, sign), if TF and target both in pool,
    #   contribute +1 to feature[TF, target] when (sign>0 AND same community)
    #   or (sign<0 AND different community); -1 otherwise.
    # Then for any unordered pair, sum contributions over all TFs they share.
    pi = pairs["i"].to_numpy(); pj = pairs["j"].to_numpy()
    n_pairs = len(pairs)

    def compute_feature(sign_array: np.ndarray) -> np.ndarray:
        # Build a (n_genes, n_genes) signed-motif-consistency matrix
        feat = np.zeros((n, n), dtype=np.float32)
        for sign, row in zip(sign_array, trrust_in_pool.itertuples()):
            tf = sym_to_idx[row.tf]; tgt = sym_to_idx[row.target]
            same_com = bool(com[tf, tgt])
            consistency = 1.0 if (
                (sign > 0 and same_com) or (sign < 0 and not same_com)
            ) else (-1.0 if sign != 0 else 0.0)
            feat[tf, tgt] += consistency
            feat[tgt, tf] += consistency
        return feat[pi, pj]

    # Filter TRRUST to pairs in this pool
    trrust_in_pool = trrust[
        trrust["tf"].isin(sym_to_idx) & trrust["target"].isin(sym_to_idx)
    ].reset_index(drop=True)
    if len(trrust_in_pool) == 0:
        return dict(error="no TRRUST edges in pool")

    obs_signs = trrust_in_pool["sign"].to_numpy()
    feat_obs = compute_feature(obs_signs)
    y = pairs["is_trrust"].to_numpy()

    # Combine with triangle-defect baseline via simple z-score sum
    tri_pair = triangle_baseline[pi, pj]
    # standardise both
    def z(x):
        x = np.asarray(x, dtype=np.float32)
        s = x.std() + 1e-12
        return (x - x.mean()) / s
    combo_obs = z(feat_obs) + z(tri_pair)
    if y.sum() == 0 or y.sum() == len(y):
        return dict(error="degenerate labels")
    auc_combo = float(roc_auc_score(y, combo_obs))
    auc_tri = float(roc_auc_score(y, tri_pair))
    auc_motif = float(roc_auc_score(y, feat_obs))

    # Null: shuffle signs while keeping TF→target structure
    rng = np.random.default_rng(null_seed_base)
    null_aucs_motif = []
    null_aucs_combo = []
    for nk in range(n_null):
        if use_degree_strata:
            # Stratified by TF degree quantile and target degree quantile
            # Compute TF/target degree from TRRUST in-pool edges
            tf_deg = trrust_in_pool.groupby("tf").size()
            tgt_deg = trrust_in_pool.groupby("target").size()
            # Quantile-bin both into 4 strata each
            tf_q = pd.qcut(trrust_in_pool["tf"].map(tf_deg), q=4, duplicates="drop",
                            labels=False)
            tgt_q = pd.qcut(trrust_in_pool["target"].map(tgt_deg), q=4, duplicates="drop",
                             labels=False)
            stratum = (tf_q.fillna(0).astype(int) * 10 +
                       tgt_q.fillna(0).astype(int)).to_numpy()
            shuffled = obs_signs.copy()
            for s_val in np.unique(stratum):
                mask = stratum == s_val
                idxs = np.where(mask)[0]
                perm = rng.permutation(idxs)
                shuffled[idxs] = obs_signs[perm]
        else:
            shuffled = obs_signs.copy()
            rng.shuffle(shuffled)
        feat_null = compute_feature(shuffled)
        combo_null = z(feat_null) + z(tri_pair)
        try:
            null_aucs_motif.append(roc_auc_score(y, feat_null))
            null_aucs_combo.append(roc_auc_score(y, combo_null))
        except Exception:
            pass

    null_aucs_motif = np.array(null_aucs_motif)
    null_aucs_combo = np.array(null_aucs_combo)

    return dict(
        auc_motif=auc_motif,
        auc_triangle_baseline=auc_tri,
        auc_combined=auc_combo,
        delta_motif_vs_baseline=auc_motif - auc_tri,
        delta_combo_vs_baseline=auc_combo - auc_tri,
        null_motif_p95=float(np.percentile(null_aucs_motif, 95)),
        null_combo_p95=float(np.percentile(null_aucs_combo, 95)),
        delta_motif_vs_null_p95=auc_motif - float(np.percentile(null_aucs_motif, 95)),
        delta_combo_vs_null_p95=auc_combo - float(np.percentile(null_aucs_combo, 95)),
        null_motif_mean=float(null_aucs_motif.mean()),
        null_combo_mean=float(null_aucs_combo.mean()),
        n_trrust_in_pool=int(len(trrust_in_pool)),
        use_degree_strata=use_degree_strata,
    )


def main():
    trrust = load_trrust_signed()
    h16_rows = []
    h116_rows = []
    h123_rows = []
    for domain in DOMAINS:
        d_dir = P0 / domain
        emb_path = d_dir / "layer_gene_embeddings_pca20.npy"
        if not emb_path.exists():
            continue
        emb = np.load(emb_path)
        gf = pd.read_csv(d_dir / "gene_features.csv")
        pairs = pd.read_csv(P1 / domain / "pair_table.csv")
        n_layers = emb.shape[0]
        sym_to_idx = {s.upper(): i for i, s in enumerate(gf["symbol"].astype(str))}
        print(f"\n[{domain}] emb {emb.shape}  pairs {len(pairs)}")

        for li in range(n_layers):
            E = emb[li]
            keep = np.linalg.norm(E, axis=1) > 0
            if keep.sum() < 50:
                continue
            E = E[keep]
            n = E.shape[0]
            mask_g = np.zeros(len(gf), dtype=bool); mask_g[:keep.shape[0]] = keep
            new_idx = -np.ones(len(gf), dtype=np.int64)
            new_idx[np.where(mask_g)[0]] = np.arange(int(mask_g.sum()))
            valid = mask_g[pairs["i"].to_numpy()] & mask_g[pairs["j"].to_numpy()]
            P_lay = pairs[valid].reset_index(drop=True).copy()
            P_lay["i"] = new_idx[P_lay["i"].to_numpy()]
            P_lay["j"] = new_idx[P_lay["j"].to_numpy()]

            kept_syms = [s for s, m in zip(gf["symbol"].astype(str).str.upper(),
                                           mask_g.tolist()) if m]
            sym_to_idx_kept = {s: i for i, s in enumerate(kept_syms)}

            # H16: community alignment AUROC
            r16 = h16_score(E, P_lay, k=KNN)
            r16.update(domain=domain, layer=li)
            h16_rows.append(r16)

            # Triangle-defect baseline
            tri = sum(triangle_defect_feature(E, k=k) for k in [8, 12, 16])

            # H116: signed motif × community, simple sign shuffle
            r116 = signed_motif_community_score(
                E, P_lay, sym_to_idx_kept, trrust, tri, k=KNN,
                use_degree_strata=False,
            )
            r116.update(domain=domain, layer=li, hypothesis="H116")
            h116_rows.append(r116)

            # H123: same but with TF/target degree-stratified sign shuffle
            r123 = signed_motif_community_score(
                E, P_lay, sym_to_idx_kept, trrust, tri, k=KNN,
                use_degree_strata=True,
            )
            r123.update(domain=domain, layer=li, hypothesis="H123")
            h123_rows.append(r123)

            print(f"  L{li:02d}  H16 auc={r16.get('auc'):.3f}  "
                  f"H116 Δcombo={r116.get('delta_combo_vs_null_p95', float('nan')):+.3f}  "
                  f"H123 Δcombo={r123.get('delta_combo_vs_null_p95', float('nan')):+.3f}")

    pd.DataFrame(h16_rows).to_csv(P78 / "h16_community.csv", index=False)
    pd.DataFrame(h116_rows).to_csv(P78 / "h116_signed_motif.csv", index=False)
    pd.DataFrame(h123_rows).to_csv(P78 / "h123_signed_motif_degree_strata.csv", index=False)

    summary = {}
    for tag, rows in [("h16", h16_rows), ("h116", h116_rows), ("h123", h123_rows)]:
        df = pd.DataFrame(rows)
        if df.empty:
            continue
        per_domain = {}
        for domain, sub in df.groupby("domain"):
            if tag == "h16":
                per_domain[domain] = dict(
                    n_layers=int(len(sub)),
                    n_layers_above_chance=int((sub["auc"] > 0.5).sum()),
                    mean_auc=float(sub["auc"].mean()),
                )
            else:
                col = "delta_combo_vs_null_p95"
                per_domain[domain] = dict(
                    n_layers=int(len(sub)),
                    mean_delta=float(sub[col].mean()),
                    n_layers_positive_null_gap=int((sub[col] > 0).sum()),
                    fraction_layers_positive=float((sub[col] > 0).mean()),
                    n_layers_significant=int((sub[col] > 0).sum()),
                )
        summary[tag] = per_domain

    with open(P78 / "phase78_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[phase 7+8] summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
