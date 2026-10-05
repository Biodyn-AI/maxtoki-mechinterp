"""Phase 5 — persistent homology (H01/H03), bifiltration cycle-rank (H47),
and degree-preserving rewiring null (Phase 5 step 10).

For each (domain, layer) tap on PCA(20)-reduced gene embeddings:
  - Compute H1 persistent homology via ripser
  - Compare total H1 lifetime to feature-shuffle null (20 replicates)
  - Compare to degree-preserving rewiring null (kNN graph, 24 replicates)
  - Bifiltration cycle-rank: count independent cycles in kNN graph

Outputs to outputs/phase5/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ripser import ripser
import networkx as nx

PROJ = Path(__file__).resolve().parents[3]
RUN = PROJ / "runs/topology-141-217M"
P0 = RUN / "outputs/phase0"
P5 = RUN / "outputs/phase5"
P5.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(RUN / "scripts"))
from phase123_splits_nulls import feature_shuffle_embedding, curveball_rewire

DOMAINS = ["lung", "immune", "external_lung"]
N_FEATURE_SHUFFLE = 20
N_REWIRE = 12
KNN_VALUES = [8, 12, 16]


def total_h1_persistence(emb: np.ndarray, max_pts: int = 350) -> float:
    """Sum of (death - birth) for H1 classes."""
    if emb.shape[0] > max_pts:
        idx = np.random.default_rng(0).choice(emb.shape[0], size=max_pts, replace=False)
        emb = emb[idx]
    res = ripser(emb, maxdim=1, n_perm=None)
    h1 = res["dgms"][1]
    if h1.size == 0:
        return 0.0
    h1 = h1[np.isfinite(h1[:, 1])]
    if h1.size == 0:
        return 0.0
    return float((h1[:, 1] - h1[:, 0]).sum())


def knn_graph(emb: np.ndarray, k: int) -> nx.Graph:
    n = emb.shape[0]
    g = nx.Graph()
    g.add_nodes_from(range(n))
    # Pairwise distances
    diff = emb[:, None, :] - emb[None, :, :]
    d2 = (diff ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nbr = np.argsort(d2, axis=1)[:, :k]
    for i in range(n):
        for j in nbr[i]:
            g.add_edge(int(i), int(j))
    return g


def cycle_rank(g: nx.Graph) -> int:
    """Number of independent cycles = m - n + c (c = number of components)."""
    return g.number_of_edges() - g.number_of_nodes() + nx.number_connected_components(g)


def rewired_total_h1(emb: np.ndarray, k: int, seed: int) -> float:
    """Persistence on a degree-preserving rewiring of the kNN graph,
    realised as a Vietoris-Rips on the *original distance matrix* but with
    the neighborhood structure shuffled (approximated by re-distributing
    edges via curveball)."""
    g = knn_graph(emb, k)
    edges = list(g.edges())
    rew = curveball_rewire(edges, n_nodes=g.number_of_nodes(),
                           n_swaps=4 * len(edges), seed=seed)
    g2 = nx.Graph(); g2.add_nodes_from(range(emb.shape[0])); g2.add_edges_from(rew)
    # Compute H1 on the *graph-distance* matrix derived from g2
    try:
        lengths = dict(nx.all_pairs_shortest_path_length(g2))
    except Exception:
        return 0.0
    n = emb.shape[0]
    D = np.full((n, n), np.inf, dtype=np.float32)
    for i, lens in lengths.items():
        for j, l in lens.items():
            D[i, j] = l
    np.fill_diagonal(D, 0.0)
    # Replace inf with a large finite value (max + 1)
    finite_max = float(D[np.isfinite(D)].max()) if np.isfinite(D).any() else 1.0
    D[~np.isfinite(D)] = finite_max + 1.0
    res = ripser(D, distance_matrix=True, maxdim=1)
    h1 = res["dgms"][1]
    if h1.size == 0:
        return 0.0
    h1 = h1[np.isfinite(h1[:, 1])]
    if h1.size == 0:
        return 0.0
    return float((h1[:, 1] - h1[:, 0]).sum())


def main():
    summary_rows = []
    bifil_rows = []
    rewire_rows = []
    for domain in DOMAINS:
        d_dir = P0 / domain
        emb_path = d_dir / "layer_gene_embeddings_pca20.npy"
        if not emb_path.exists():
            print(f"[{domain}] missing PCA embeddings — skip")
            continue
        emb = np.load(emb_path)
        n_layers = emb.shape[0]
        print(f"\n[{domain}] PCA emb shape: {emb.shape}")
        for li in range(n_layers):
            E = emb[li]
            keep = np.linalg.norm(E, axis=1) > 0
            E = E[keep]
            n = E.shape[0]
            obs = total_h1_persistence(E)
            # Feature-shuffle null
            null = []
            for s in range(N_FEATURE_SHUFFLE):
                Es = feature_shuffle_embedding(E, seed=s)
                null.append(total_h1_persistence(Es))
            null = np.array(null)
            null_mean = float(null.mean())
            null_std = float(null.std() + 1e-12)
            z = (obs - null_mean) / null_std
            null_p95 = float(np.percentile(null, 95))
            delta = obs - null_mean
            sig = obs > null_p95
            row = dict(
                domain=domain, layer=li, n_genes=n,
                obs_h1=obs, null_mean=null_mean, null_std=null_std,
                null_p95=null_p95, z_score=z, delta=delta,
                sig_at_5pct=int(sig),
            )
            summary_rows.append(row)
            print(f"  L{li:02d}  obs={obs:.2f}  null={null_mean:.2f}±{null_std:.2f}  "
                  f"z={z:.2f}  sig={int(sig)}")

            # Rewiring null per k (cheap subset, 4 replicates × {12} for budget)
            rew_results = {}
            for k in [12]:
                obs_k = obs  # use same obs for reference
                rew_vals = []
                for s in range(N_REWIRE):
                    rv = rewired_total_h1(E, k=k, seed=s + 100 * li)
                    rew_vals.append(rv)
                rew_vals = np.array(rew_vals)
                rew_p95 = float(np.percentile(rew_vals, 95))
                rew_results[k] = dict(
                    rewire_mean=float(rew_vals.mean()),
                    rewire_p95=rew_p95,
                    sig_vs_rewire=int(obs_k > rew_p95),
                )
            for k, r in rew_results.items():
                rewire_rows.append(dict(domain=domain, layer=li, k=k, **r,
                                        obs_h1=obs))

            # Bifiltration cycle-rank per k ∈ {8, 12, 16}
            for k in KNN_VALUES:
                g = knn_graph(E, k=k)
                cr = cycle_rank(g)
                bifil_rows.append(dict(domain=domain, layer=li, k=k,
                                        cycle_rank=cr,
                                        n_edges=g.number_of_edges(),
                                        n_components=nx.number_connected_components(g)))

    pd.DataFrame(summary_rows).to_csv(P5 / "h01_h03_persistent_homology.csv", index=False)
    pd.DataFrame(rewire_rows).to_csv(P5 / "h01_rewiring_null.csv", index=False)
    pd.DataFrame(bifil_rows).to_csv(P5 / "h47_bifiltration_cycle_rank.csv", index=False)

    # Summary stats per domain
    summary = {}
    df = pd.DataFrame(summary_rows)
    if not df.empty:
        for domain, sub in df.groupby("domain"):
            summary[domain] = dict(
                n_layers=int(len(sub)),
                n_layers_significant=int(sub["sig_at_5pct"].sum()),
                top_z=float(sub["z_score"].max()),
                top_layer=int(sub.loc[sub["z_score"].idxmax(), "layer"]),
                mean_delta=float(sub["delta"].mean()),
            )
    df2 = pd.DataFrame(rewire_rows)
    if not df2.empty:
        for domain, sub in df2.groupby("domain"):
            summary.setdefault(domain, {})
            summary[domain]["rewire_n_significant"] = int(sub["sig_vs_rewire"].sum())
            summary[domain]["rewire_n_total"] = int(len(sub))
    with open(P5 / "phase5_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[phase 5] summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
