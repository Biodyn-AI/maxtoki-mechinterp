"""H03: Gene kNN graph spectral gap (Fiedler value) and STRING community cohesion.

Tests: (1) does algebraic connectivity of gene-embedding kNN graph change non-monotonically
       across MaxToki layers?
       (2) are STRING PPI pairs over-represented as kNN neighbors at later layers?
       Both compared to ER-null.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import eigsh
import json
import os
import time

BASE = '<REPO_ROOT>/projects/maxtoki/runs/spectral-geometry-217M'
BIOM = '<DATA_ROOT>/biodyn-nmi-paper'
OUT = f'{BASE}/autoloop/iterations/iter_0002'
os.makedirs(OUT, exist_ok=True)

emb = np.load(f'{BASE}/outputs/phase0/layer_gene_embeddings.npy')
genes = pd.read_csv(f'{BASE}/outputs/phase0/gene_features.csv')
n_layers, n_genes, d_model = emb.shape
gene_symbols = [s.upper() for s in genes['symbol']]
sym_to_idx = {s: i for i, s in enumerate(gene_symbols)}

# STRING pairs present in HVGs
with open(f'{BIOM}/results/biological_impact/reference_edge_sets/string_ppi_edges.json') as f:
    string_data = json.load(f)

string_pairs_700 = set()
string_pairs_900 = set()
for pair in string_data.get('pairs_700', []):
    g1, g2 = pair[0].upper(), pair[1].upper()
    if g1 in sym_to_idx and g2 in sym_to_idx:
        i, j = sym_to_idx[g1], sym_to_idx[g2]
        if i != j:
            string_pairs_700.add((min(i, j), max(i, j)))
for pair in string_data.get('pairs_900', []):
    g1, g2 = pair[0].upper(), pair[1].upper()
    if g1 in sym_to_idx and g2 in sym_to_idx:
        i, j = sym_to_idx[g1], sym_to_idx[g2]
        if i != j:
            string_pairs_900.add((min(i, j), max(i, j)))

print(f'STRING pairs in HVGs: 700={len(string_pairs_700)}, 900={len(string_pairs_900)}')


def build_knn_adjacency(X, k=15):
    """Build symmetric kNN adjacency matrix (cosine similarity)."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_n = (X / (norms + 1e-10)).astype(np.float32)
    sim = X_n @ X_n.T
    np.fill_diagonal(sim, -np.inf)

    n = sim.shape[0]
    rows, cols = [], []
    for i in range(n):
        nn = np.argpartition(-sim[i], k)[:k]
        for j in nn:
            if i != j:
                rows += [i, j]
                cols += [j, i]

    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A = (A > 0).astype(np.float64)
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def normalized_laplacian_fiedler(A):
    """Second eigenvalue of normalized graph Laplacian."""
    d = np.array(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_isq = diags(d_inv_sqrt)
    L = eye(A.shape[0], format='csr') - D_isq @ A @ D_isq
    try:
        vals = eigsh(L, k=6, which='SM', return_eigenvectors=False, tol=1e-5, maxiter=3000)
        vals = np.sort(np.real(vals))
        return float(vals[1])
    except Exception as e:
        print(f'  eigsh warning: {e}')
        return np.nan


def string_precision_at_k(A, string_pairs):
    """Fraction of STRING pairs that appear as direct kNN neighbors."""
    if not string_pairs:
        return np.nan
    A_arr = A.toarray()
    hits = sum(1 for (i, j) in string_pairs if A_arr[i, j] > 0)
    return hits / len(string_pairs)


def null_string_precision(n_nodes, n_edges, n_pairs, n_rep=200, seed=42):
    """Expected fraction under random edge assignment (hypergeometric approx)."""
    total_possible = n_nodes * (n_nodes - 1) / 2
    # P(edge | random) = n_edges / total_possible
    return (n_edges / total_possible), np.sqrt(
        (n_edges / total_possible) * (1 - n_edges / total_possible) / n_pairs
    )


# Main loop
K = 15
rows_out = []

for layer in range(n_layers):
    X = emb[layer]
    t0 = time.time()

    A = build_knn_adjacency(X, k=K)
    n_edges = A.nnz // 2

    fv = normalized_laplacian_fiedler(A)
    p700 = string_precision_at_k(A, string_pairs_700)
    p900 = string_precision_at_k(A, string_pairs_900)

    # Null for STRING precision
    null_p700, null_se700 = null_string_precision(n_genes, n_edges, len(string_pairs_700))
    null_p900, null_se900 = null_string_precision(n_genes, n_edges, len(string_pairs_900))
    z700 = (p700 - null_p700) / (null_se700 + 1e-12)
    z900 = (p900 - null_p900) / (null_se900 + 1e-12)

    t1 = time.time()
    print(f'L{layer:02d}: fiedler={fv:.4f}, prec700={p700:.4f}(z={z700:.2f}), '
          f'prec900={p900:.4f}(z={z900:.2f}), n_edges={n_edges}, t={t1-t0:.1f}s')

    rows_out.append({
        'layer': layer,
        'fiedler_value': fv,
        'n_edges': n_edges,
        'string_prec_700': p700,
        'string_prec_900': p900,
        'null_prec_700': null_p700,
        'null_prec_900': null_p900,
        'z_700': z700,
        'z_900': z900,
        'time_s': t1 - t0,
    })

# Compute Fiedler null from random ER graph (do once at L0 edge count)
A_l0 = build_knn_adjacency(emb[0], k=K)
n_edges_l0 = A_l0.nnz // 2
rng = np.random.default_rng(42)
null_fiedlers = []
for rep in range(8):
    # Random symmetric adjacency with same edge count
    pairs_idx = np.array(
        [rng.choice(n_genes, 2, replace=False) for _ in range(n_edges_l0)]
    )
    r, c = pairs_idx[:, 0], pairs_idx[:, 1]
    A_rand = csr_matrix(
        (np.ones(2 * len(r)), (np.concatenate([r, c]), np.concatenate([c, r]))),
        shape=(n_genes, n_genes)
    )
    A_rand = (A_rand > 0).astype(np.float64)
    A_rand.setdiag(0)
    A_rand.eliminate_zeros()
    null_fiedlers.append(normalized_laplacian_fiedler(A_rand))
null_f_mean = float(np.nanmean(null_fiedlers))
null_f_std = float(np.nanstd(null_fiedlers))

df = pd.DataFrame(rows_out)
df['fiedler_null_mean'] = null_f_mean
df['fiedler_null_std'] = null_f_std
df['fiedler_z'] = (df['fiedler_value'] - null_f_mean) / (null_f_std + 1e-10)

out_path = f'{OUT}/h03_spectral_gap.csv'
df.to_csv(out_path, index=False)
print(f'\nSaved: {out_path}')
print(df[['layer', 'fiedler_value', 'fiedler_z', 'z_700', 'z_900']].to_string())
