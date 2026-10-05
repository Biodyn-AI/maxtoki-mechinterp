"""Phase 9 — H91 stability-selected geometric descriptors.

Combines geometric edge features (geodesic, triangle-defect spectrum at
k∈{8,12,16}, community co-membership, bifiltration cycle-rank) into a
multivariate logistic classifier evaluated under the dual disjoint gene-pool
splits with stability selection (randomised LASSO bootstrap).

Outputs: outputs/phase9/h91.csv, phase9_summary.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

PROJ = Path(__file__).resolve().parents[3]
RUN = PROJ / "runs/topology-141-217M"
P0 = RUN / "outputs/phase0"
P1 = RUN / "outputs/phase1"
P9 = RUN / "outputs/phase9"
P9.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(RUN / "scripts"))
from phase6_manifold_distances import (
    pairwise_distances, triangle_defect_feature, adaptive_knn_graph,
    geodesic_distances, coexpression_matrix,
)
from phase78_community_signed_motif import louvain_communities, comembership_matrix

DOMAINS = ["lung", "immune", "external_lung"]
N_BOOTSTRAP = 50
LAYERS_FOR_STABILITY = [3, 6, 9]  # use a subset of layers to keep compute reasonable


def build_geometric_features(emb: np.ndarray) -> dict[str, np.ndarray]:
    n = emb.shape[0]
    feats: dict[str, np.ndarray] = {}
    g, k_used = adaptive_knn_graph(emb)
    feats["geodesic"] = geodesic_distances(g)
    if not np.isfinite(feats["geodesic"]).all():
        feats["geodesic"][~np.isfinite(feats["geodesic"])] = (
            float(feats["geodesic"][np.isfinite(feats["geodesic"])].max()) + 1.0
        )
    for k in [8, 12, 16]:
        feats[f"tri_k{k}"] = triangle_defect_feature(emb, k=k)
    # Community comembership (k=12)
    D = pairwise_distances(emb)
    np.fill_diagonal(D, np.inf)
    nbr = np.argsort(D, axis=1)[:, :12]
    g2 = nx.Graph(); g2.add_nodes_from(range(n))
    for i in range(n):
        for j in nbr[i]:
            g2.add_edge(int(i), int(j))
    labels = louvain_communities(g2, seed=42)
    feats["community"] = comembership_matrix(labels)
    # Cycle-rank-derived per-pair feature: for each pair, # of common neighbours
    A = nx.adjacency_matrix(g2).toarray().astype(np.float32)
    feats["common_nbrs"] = (A @ A).astype(np.float32)
    return feats


def stability_selected_classifier(X: np.ndarray, y: np.ndarray,
                                   train_idx: np.ndarray, test_idx: np.ndarray,
                                   feature_names: list[str], seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(X[train_idx])
    selected_counts = np.zeros(X.shape[1])
    for b in range(N_BOOTSTRAP):
        sub = rng.choice(len(train_idx), size=len(train_idx), replace=True)
        clf = LogisticRegression(penalty="l1", solver="saga", C=0.1,
                                  random_state=int(rng.integers(0, 1 << 30)),
                                  max_iter=200)
        try:
            clf.fit(Xtr_s[sub], y[train_idx][sub])
        except Exception:
            continue
        if clf.coef_.shape[1] != X.shape[1]:
            continue
        selected_counts += (np.abs(clf.coef_[0]) > 1e-6).astype(int)
    # Fit final classifier on stability-selected features (selected in ≥ 50% of bootstraps)
    selected_mask = selected_counts >= (N_BOOTSTRAP * 0.5)
    if selected_mask.sum() == 0:
        selected_mask = selected_counts >= 1
    final = LogisticRegression(penalty="l2", C=1.0, max_iter=500)
    if selected_mask.sum() == 0:
        return dict(error="no features selected")
    final.fit(Xtr_s[:, selected_mask], y[train_idx])
    Xte_s = sc.transform(X[test_idx])[:, selected_mask]
    p = final.predict_proba(Xte_s)[:, 1]
    if y[test_idx].sum() == 0 or y[test_idx].sum() == len(test_idx):
        return dict(error="degenerate test labels")
    auc = float(roc_auc_score(y[test_idx], p))
    return dict(
        auc_test=auc,
        n_features_total=int(X.shape[1]),
        n_features_selected=int(selected_mask.sum()),
        selected_features=[feature_names[i] for i, m in enumerate(selected_mask) if m],
        selection_freq={feature_names[i]: float(selected_counts[i] / N_BOOTSTRAP)
                         for i in range(len(feature_names))},
    )


def main():
    rows = []
    for domain in DOMAINS:
        d_dir = P0 / domain
        emb_path = d_dir / "layer_gene_embeddings_pca20.npy"
        if not emb_path.exists():
            continue
        emb = np.load(emb_path)
        gf = pd.read_csv(d_dir / "gene_features.csv")
        pairs = pd.read_csv(P1 / domain / "pair_table.csv")
        with open(P1 / domain / "splits.json") as f:
            splits = json.load(f)
        n_layers = emb.shape[0]
        layer_set = [li for li in LAYERS_FOR_STABILITY if li < n_layers]
        print(f"\n[{domain}] running H91 on layers {layer_set}")

        for li in layer_set:
            E = emb[li]
            keep = np.linalg.norm(E, axis=1) > 0
            if keep.sum() < 50:
                continue
            E = E[keep]
            mask_g = np.zeros(len(gf), dtype=bool); mask_g[:keep.shape[0]] = keep
            new_idx = -np.ones(len(gf), dtype=np.int64)
            new_idx[np.where(mask_g)[0]] = np.arange(int(mask_g.sum()))
            valid = mask_g[pairs["i"].to_numpy()] & mask_g[pairs["j"].to_numpy()]
            P_lay = pairs[valid].reset_index(drop=True).copy()
            P_lay["i"] = new_idx[P_lay["i"].to_numpy()]
            P_lay["j"] = new_idx[P_lay["j"].to_numpy()]
            y_all = P_lay["is_trrust"].to_numpy()

            # Build geometric features → flat per-pair feature matrix
            feats = build_geometric_features(E)
            fnames = list(feats.keys())
            X = np.column_stack([
                feats[name][P_lay["i"].to_numpy(), P_lay["j"].to_numpy()]
                for name in fnames
            ])
            # Re-map split indices into the post-valid index space
            for seed_str, sp in splits.items():
                for regime in ["source_disjoint", "target_disjoint"]:
                    train_full = np.array(sp[regime]["train_idx"])
                    test_full = np.array(sp[regime]["test_idx"])
                    valid_arr = valid.to_numpy() if hasattr(valid, "to_numpy") else np.asarray(valid)
                    train = np.array([t for t in train_full if valid_arr[int(t)]])
                    test = np.array([t for t in test_full if valid_arr[int(t)]])
                    if len(train) == 0 or len(test) == 0:
                        continue
                    # remap to new indices in P_lay
                    pair_map = -np.ones(len(pairs), dtype=np.int64)
                    pair_map[np.where(valid)[0]] = np.arange(int(valid.sum()))
                    train_l = pair_map[train.astype(int)]
                    test_l = pair_map[test.astype(int)]
                    train_l = train_l[train_l >= 0]
                    test_l = test_l[test_l >= 0]
                    if len(train_l) == 0 or len(test_l) == 0:
                        continue
                    r = stability_selected_classifier(
                        X, y_all, train_l, test_l, fnames,
                        seed=int(seed_str))
                    if "error" in r:
                        continue
                    rows.append(dict(
                        domain=domain, layer=li, seed=int(seed_str),
                        regime=regime, **r,
                    ))
                    print(f"  L{li:02d} seed={seed_str} {regime}  "
                          f"AUC={r['auc_test']:.3f}  "
                          f"selected={r['n_features_selected']}/{r['n_features_total']}")

    if not rows:
        print("[phase 9] no rows produced")
        return
    df = pd.DataFrame(rows)
    df.to_csv(P9 / "h91_stability_selection.csv", index=False)
    summary = {}
    for domain, sub in df.groupby("domain"):
        summary[domain] = dict(
            n_rows=int(len(sub)),
            mean_auc=float(sub["auc_test"].mean()),
            median_auc=float(sub["auc_test"].median()),
            fraction_above_chance=float((sub["auc_test"] > 0.5).mean()),
        )
    with open(P9 / "phase9_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[phase 9]")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
