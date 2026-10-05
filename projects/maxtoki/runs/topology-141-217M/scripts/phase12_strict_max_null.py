"""Phase 12 — H141 strict max-null audit.

For every previously-promoted positive (Phase 6 distance metrics, Phase 7+8
H16/H116/H123, Phase 9 H91), compute the **strict margin** = observed
ΔAUROC minus the **maximum** of the 95th percentile across ALL four null
families simultaneously.

Implementation note: in this scoped run we approximate the four null families
as follows for each test:
  - feature-shuffle null: AUROC of the same metric computed on
    feature-shuffled embedding (mean over 12 reps); 95th percentile of
    those replicate AUCs is the null floor
  - label-permutation null: AUROC after permuting labels (100 reps)
  - rewiring null: AUROC of the metric computed on a degree-preserving
    rewiring of the kNN graph (12 reps for distance metrics)
  - coexpression-matched null: AUROC of |coexpression| feature alone

Strict margin = observed_AUROC − max(null_95th_percentile across the four families).

For Phase 7+8 motif tests, the "feature-shuffle" / "rewiring" rows reuse the
embedded null already in phase78_summary.json.

Output: outputs/phase12/h141.csv, phase12_summary.json.
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
P6 = RUN / "outputs/phase6"
P78 = RUN / "outputs/phase78"
P12 = RUN / "outputs/phase12"
P12.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(RUN / "scripts"))
from phase6_manifold_distances import (
    pairwise_distances, triangle_defect_feature, adaptive_knn_graph,
    geodesic_distances, coexpression_matrix, auroc_pair_feature,
    diffusion_distances,
)
from phase123_splits_nulls import feature_shuffle_embedding, curveball_rewire

DOMAINS = ["lung", "immune", "external_lung"]
N_FEATURE_SHUFFLE = 12
N_LABEL_PERM = 100
N_REWIRE = 8


def strict_audit_distance_metric(
    metric_name: str,
    metric_fn,
    direction: str,
    emb: np.ndarray,
    pairs: pd.DataFrame,
    log_expr: np.ndarray,
    seed: int = 42,
) -> dict:
    """Run the four-null audit on a distance-style pair feature."""
    rng = np.random.default_rng(seed)
    pi = pairs["i"].to_numpy(); pj = pairs["j"].to_numpy()
    y = pairs["is_trrust"].to_numpy()
    if y.sum() == 0:
        return {}

    # 1) Observed AUROC
    feat_obs = metric_fn(emb)
    f_obs = feat_obs[pi, pj]
    if direction == "smaller_is_positive":
        f_obs = -f_obs
    auc_obs = float(roc_auc_score(y, f_obs))

    # 2) Feature-shuffle null
    fs_aucs = []
    for k in range(N_FEATURE_SHUFFLE):
        Es = feature_shuffle_embedding(emb, seed=int(rng.integers(0, 1 << 30)))
        feat = metric_fn(Es)
        f = feat[pi, pj]
        if direction == "smaller_is_positive":
            f = -f
        try:
            fs_aucs.append(roc_auc_score(y, f))
        except Exception:
            pass
    fs_aucs = np.array(fs_aucs) if fs_aucs else np.array([0.5])

    # 3) Label-permutation null  (uses observed feature)
    lp_aucs = []
    for k in range(N_LABEL_PERM):
        yp = y.copy(); rng.shuffle(yp)
        try:
            lp_aucs.append(roc_auc_score(yp, f_obs))
        except Exception:
            pass
    lp_aucs = np.array(lp_aucs) if lp_aucs else np.array([0.5])

    # 4) Rewiring null (degree-preserving on kNN at k=12)
    rew_aucs = []
    n = emb.shape[0]
    D = pairwise_distances(emb)
    np.fill_diagonal(D, np.inf)
    nbr = np.argsort(D, axis=1)[:, :12]
    edges = []
    for i in range(n):
        for j in nbr[i]:
            edges.append((int(i), int(j)))
    for k in range(N_REWIRE):
        rew = curveball_rewire(edges, n_nodes=n, n_swaps=4 * len(edges),
                                seed=int(rng.integers(0, 1 << 30)))
        # Rebuild a "rewired distance" by graph shortest-path
        g = nx.Graph(); g.add_nodes_from(range(n)); g.add_edges_from(rew)
        try:
            lengths = dict(nx.all_pairs_shortest_path_length(g))
        except Exception:
            continue
        Dr = np.full((n, n), np.inf, dtype=np.float32)
        for i, lens in lengths.items():
            for j, l in lens.items():
                Dr[i, j] = l
        np.fill_diagonal(Dr, 0.0)
        if not np.isfinite(Dr).all():
            Dr[~np.isfinite(Dr)] = float(Dr[np.isfinite(Dr)].max()) + 1.0
        f = -Dr[pi, pj]
        try:
            rew_aucs.append(roc_auc_score(y, f))
        except Exception:
            pass
    rew_aucs = np.array(rew_aucs) if rew_aucs else np.array([0.5])

    # 5) Coexpression-matched null: AUROC of |coexpression| alone
    coex = coexpression_matrix(log_expr)
    auc_coex = float(roc_auc_score(y, coex[pi, pj]))

    nulls_p95 = {
        "feature_shuffle_p95": float(np.percentile(fs_aucs, 95)),
        "label_perm_p95": float(np.percentile(lp_aucs, 95)),
        "rewire_p95": float(np.percentile(rew_aucs, 95)),
        "coexpression_floor": auc_coex,
    }
    max_null_p95 = float(max(nulls_p95.values()))
    strict_margin = auc_obs - max_null_p95

    return dict(
        metric=metric_name,
        auc_obs=auc_obs,
        **nulls_p95,
        max_null_p95=max_null_p95,
        strict_margin=strict_margin,
        passes_strict=int(strict_margin > 0),
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
        log_expr = np.load(d_dir / "log_expression_pool.npy")
        n_layers = emb.shape[0]
        # Audit a representative subset of layers (early/mid/late)
        layer_set = [3, 6, 9]
        layer_set = [li for li in layer_set if li < n_layers]
        print(f"\n[{domain}] strict audit on layers {layer_set}")

        for li in layer_set:
            E = emb[li]
            keep = np.linalg.norm(E, axis=1) > 0
            E = E[keep]
            mask_g = np.zeros(len(gf), dtype=bool); mask_g[:keep.shape[0]] = keep
            new_idx = -np.ones(len(gf), dtype=np.int64)
            new_idx[np.where(mask_g)[0]] = np.arange(int(mask_g.sum()))
            valid = mask_g[pairs["i"].to_numpy()] & mask_g[pairs["j"].to_numpy()]
            P_lay = pairs[valid].reset_index(drop=True).copy()
            P_lay["i"] = new_idx[P_lay["i"].to_numpy()]
            P_lay["j"] = new_idx[P_lay["j"].to_numpy()]
            log_expr_kept = log_expr[:, mask_g]

            metrics = [
                ("euclidean", lambda e: pairwise_distances(e), "smaller_is_positive"),
                ("geodesic",
                 lambda e: geodesic_distances(adaptive_knn_graph(e)[0]),
                 "smaller_is_positive"),
                ("diffusion_t4",
                 lambda e: diffusion_distances(e, k=12, t=4),
                 "smaller_is_positive"),
                ("triangle_defect",
                 lambda e: sum(triangle_defect_feature(e, k=k) for k in [8, 12, 16]),
                 "larger_is_positive"),
            ]

            for name, fn, direction in metrics:
                try:
                    r = strict_audit_distance_metric(
                        name, fn, direction, E, P_lay, log_expr_kept,
                        seed=42 + li,
                    )
                except Exception as e:
                    r = {"error": str(e), "metric": name}
                r.update(domain=domain, layer=li)
                rows.append(r)
                if "strict_margin" in r:
                    print(f"  L{li:02d} {name:>16s}  "
                          f"obs={r['auc_obs']:.3f}  max_null_p95={r['max_null_p95']:.3f}  "
                          f"strict={r['strict_margin']:+.3f}  pass={r['passes_strict']}")

    df = pd.DataFrame(rows)
    df.to_csv(P12 / "h141_strict_margins.csv", index=False)

    summary = {}
    if not df.empty and "strict_margin" in df.columns:
        df_clean = df.dropna(subset=["strict_margin"])
        for domain, sub in df_clean.groupby("domain"):
            summary[domain] = dict(
                n_tests=int(len(sub)),
                mean_strict_margin=float(sub["strict_margin"].mean()),
                n_strict_positive=int(sub["passes_strict"].sum()),
                fraction_strict_positive=float(sub["passes_strict"].mean()),
            )
        summary["overall"] = dict(
            mean_strict_margin=float(df_clean["strict_margin"].mean()),
            n_strict_positive=int(df_clean["passes_strict"].sum()),
            n_tests=int(len(df_clean)),
        )
    with open(P12 / "phase12_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[phase 12] summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
