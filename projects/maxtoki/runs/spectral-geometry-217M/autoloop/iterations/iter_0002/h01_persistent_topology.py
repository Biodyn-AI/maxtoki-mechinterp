"""H01: H0 Persistent topology of gene embeddings via cosine-distance filtration.

Proper Betti-0 persistence barcode (union-find on sorted cosine distances).
Compares real geometry to feature-shuffle null across all 12 MaxToki layers.
"""
import numpy as np
import pandas as pd
import os
import time

BASE = '<REPO_ROOT>/projects/maxtoki/runs/spectral-geometry-217M'
OUT = f'{BASE}/autoloop/iterations/iter_0002'
os.makedirs(OUT, exist_ok=True)

emb = np.load(f'{BASE}/outputs/phase0/layer_gene_embeddings.npy')  # (12, 1500, 1232)
n_layers, n_genes, d_model = emb.shape
print(f'Embeddings: {emb.shape}')


def cosine_dist_matrix(X):
    """Pairwise cosine distances for (n, d) array."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_n = X / (norms + 1e-10)
    cos_sim = X_n @ X_n.T
    np.fill_diagonal(cos_sim, 1.0)
    return (1.0 - cos_sim).astype(np.float32)


def h0_persistence(dist_matrix):
    """Betti-0 persistence: component lifetimes via union-find on MST edges.

    Returns sorted array of component death-distances (birth=0 for all).
    """
    n = dist_matrix.shape[0]
    triu_i, triu_j = np.triu_indices(n, k=1)
    edge_dists = dist_matrix[triu_i, triu_j]
    order = np.argsort(edge_dists)

    parent = np.arange(n, dtype=np.int32)
    rank = np.zeros(n, dtype=np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    deaths = []
    n_comp = n
    for k in order:
        d = edge_dists[k]
        i, j = int(triu_i[k]), int(triu_j[k])
        ri, rj = find(i), find(j)
        if ri != rj:
            if rank[ri] < rank[rj]:
                ri, rj = rj, ri
            parent[rj] = ri
            if rank[ri] == rank[rj]:
                rank[ri] += 1
            deaths.append(d)
            n_comp -= 1
            if n_comp == 1:
                break

    return np.array(deaths, dtype=np.float32)


def persistence_metrics(lifetimes):
    """Summary statistics for a persistence diagram."""
    if len(lifetimes) == 0:
        return dict(total=0., max_=0., n_long=0, entropy=0.)
    total = float(lifetimes.sum())
    p = lifetimes / (total + 1e-12)
    entropy = float(-np.sum(p * np.log(p + 1e-12)))
    return dict(
        total=total,
        max_=float(lifetimes.max()),
        n_long=int((lifetimes > 0.10).sum()),
        n_vlong=int((lifetimes > 0.20).sum()),
        entropy=entropy,
        mean_=float(lifetimes.mean()),
        p95=float(np.percentile(lifetimes, 95)),
    )


rng = np.random.default_rng(42)
rows = []

for layer in range(n_layers):
    X = emb[layer]
    t0 = time.time()

    # Real
    D_real = cosine_dist_matrix(X)
    lt_real = h0_persistence(D_real)
    m_real = persistence_metrics(lt_real)

    # Feature-shuffle null (permute within each dimension)
    X_null = X.copy()
    for dim in range(d_model):
        X_null[:, dim] = rng.permutation(X_null[:, dim])
    D_null = cosine_dist_matrix(X_null)
    lt_null = h0_persistence(D_null)
    m_null = persistence_metrics(lt_null)

    t1 = time.time()

    row = {'layer': layer}
    for k, v in m_real.items():
        row[f'real_{k}'] = v
    for k, v in m_null.items():
        row[f'null_{k}'] = v
    row['persistence_ratio'] = m_real['total'] / (m_null['total'] + 1e-12)
    row['max_lifetime_ratio'] = m_real['max_'] / (m_null['max_'] + 1e-12)
    row['n_long_ratio'] = m_real['n_long'] / (m_null['n_long'] + 1e-3)
    row['entropy_ratio'] = m_real['entropy'] / (m_null['entropy'] + 1e-12)
    row['time_s'] = t1 - t0
    rows.append(row)

    print(f'L{layer:02d}: ratio={row["persistence_ratio"]:.3f}, '
          f'max_real={m_real["max_"]:.4f}, max_null={m_null["max_"]:.4f}, '
          f'n_long_real={m_real["n_long"]}, n_long_null={m_null["n_long"]}, '
          f't={t1-t0:.1f}s')

df = pd.DataFrame(rows)
out_path = f'{OUT}/h01_persistent_topology.csv'
df.to_csv(out_path, index=False)
print(f'\nSaved: {out_path}')
print(df[['layer', 'persistence_ratio', 'max_lifetime_ratio', 'n_long_ratio']].to_string())
