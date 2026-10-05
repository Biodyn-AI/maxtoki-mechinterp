"""H3-BC: STRING kNN pair persistence across layers — decay-rate vs confidence tier.

Tests whether STRING pairs that remain kNN neighbors across many layers
are enriched for the higher-confidence (900+) tier vs the 700-900 tier.
Channel-level weights (coexpression/experimental/textmining) are unavailable
in the local STRING JSON; 700-vs-900 tier split serves as confidence proxy.

Null: gene-label shuffle (permute gene identity assignment, recompute STRING
pair adjacency in original kNN graphs).
"""
import json
import time
import numpy as np
import pandas as pd
from scipy import stats

BASE = '<REPO_ROOT>/projects/maxtoki/runs/spectral-geometry-217M'
BIOM = '<DATA_ROOT>'
OUT  = f'{BASE}/autoloop/iterations/iter_0003'

print("=" * 70)
print("H3-BC: STRING kNN pair persistence — confidence tier vs layer persistence")
print("=" * 70)
t0_total = time.time()

# ── Data ─────────────────────────────────────────────────────────────────────
emb   = np.load(f'{BASE}/outputs/phase0/layer_gene_embeddings.npy')
genes = pd.read_csv(f'{BASE}/outputs/phase0/gene_features.csv')
n_layers, n_genes, d_model = emb.shape
print(f"Embeddings: {emb.shape}")

with open(f'{BIOM}/biodyn-nmi-paper/results/biological_impact/reference_edge_sets/string_ppi_edges.json') as f:
    string_data = json.load(f)

gene_symbols = [s.upper() for s in genes['symbol']]
sym_to_idx   = {s: i for i, s in enumerate(gene_symbols)}

# Build pair lists (deduplicated by (min,max) index)
seen_700  = set()
pairs_700 = []  # (i, j)
for a, b in string_data.get('pairs_700', []):
    i, j = sym_to_idx.get(a.upper()), sym_to_idx.get(b.upper())
    if i is None or j is None or i == j:
        continue
    key = (min(i,j), max(i,j))
    if key not in seen_700:
        seen_700.add(key)
        pairs_700.append(key)

seen_900  = set()
pairs_900 = []
for a, b in string_data.get('pairs_900', []):
    i, j = sym_to_idx.get(a.upper()), sym_to_idx.get(b.upper())
    if i is None or j is None or i == j:
        continue
    key = (min(i,j), max(i,j))
    if key not in seen_900:
        seen_900.add(key)
        pairs_900.append(key)

pairs_900_set = set(pairs_900)
pairs_arr     = np.array(pairs_700, dtype=np.int32)
is_900        = np.array([p in pairs_900_set for p in pairs_700], dtype=bool)
n_pairs       = len(pairs_700)
print(f"STRING pairs in HVGs: 700={n_pairs}, 900={is_900.sum()}, 700-900 only={n_pairs - is_900.sum()}")

# ── Step 1: Build kNN graphs & check pair adjacency (12 layers) ──────────────
K = 15
knn_sets = []
t0 = time.time()
for layer in range(n_layers):
    X  = emb[layer].astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_n = X / (norms + 1e-10)
    sim = X_n @ X_n.T
    np.fill_diagonal(sim, -np.inf)
    nn_idx = np.argpartition(-sim, K, axis=1)[:, :K]
    knn_set = set()
    for gi in range(n_genes):
        for nj in nn_idx[gi]:
            if gi != nj:
                knn_set.add((min(gi, nj), max(gi, nj)))
    knn_sets.append(knn_set)
    print(f"  L{layer:02d}: kNN graph built, |edges|={len(knn_set)}, t={time.time()-t0:.1f}s")

# Real pair adjacency
adj = np.zeros((n_pairs, n_layers), dtype=bool)
for li, knn_set in enumerate(knn_sets):
    for pi, (i, j) in enumerate(pairs_arr):
        adj[pi, li] = (int(i), int(j)) in knn_set

np.save(f'{OUT}/h3bc_pair_adjacency.npy', adj)
layer_count = adj.sum(axis=1).astype(int)  # (n_pairs,)

# ── Step 2: Categorize pairs ─────────────────────────────────────────────────
def categorize_pair(row):
    lc = row.sum()
    if lc == 12:              return 'always'
    if lc == 0:               return 'never'
    if row[0] and lc == 1:    return 'L0_only'
    early = row[0] and row[:4].sum() >= 3 and row[4:].sum() < 2
    if early:                 return 'early'
    if row[8:].sum() >= 3:    return 'late'
    return 'mixed'

categories = [categorize_pair(adj[pi]) for pi in range(n_pairs)]

cat_df = pd.DataFrame({
    'gene_a':      [genes['symbol'].iloc[int(i)] for i, _ in pairs_arr],
    'gene_b':      [genes['symbol'].iloc[int(j)] for _, j in pairs_arr],
    'layer_count': layer_count,
    'category':    categories,
    'is_900_tier': is_900,
})
cat_df.to_csv(f'{OUT}/h3bc_pair_categories.csv', index=False)

print("\n--- Category counts ---")
for cat in ['always', 'early', 'late', 'L0_only', 'mixed', 'never']:
    mask = cat_df['category'] == cat
    print(f"  {cat:10s}: n={mask.sum():4d} | "
          f"mean_900={is_900[mask.values].mean():.2f} | "
          f"mean_lc={layer_count[mask.values].mean():.1f}")

# ── Step 3: Spearman(layer_count, is_900) ────────────────────────────────────
rho_all, p_all = stats.spearmanr(layer_count, is_900.astype(float))
print(f"\nSpearman(layer_count, is_900_tier) all: rho={rho_all:.4f}, p={p_all:.4f}")

lc_900 = layer_count[is_900]
lc_700_900 = layer_count[~is_900]
mw_stat, mw_p = stats.mannwhitneyu(lc_900, lc_700_900, alternative='two-sided')
med_900     = float(np.median(lc_900))
med_700_900 = float(np.median(lc_700_900))
print(f"Mann-Whitney 900+ vs 700-900: stat={mw_stat:.0f}, p={mw_p:.4f}")
print(f"  median 900+={med_900:.1f}, 700-900={med_700_900:.1f}, diff={med_900-med_700_900:.1f}")

# Separate Spearman by tier subgroup (within 700-900 and within 900+)
rho_700_only, p_700_only = stats.spearmanr(
    layer_count[~is_900], np.zeros(int((~is_900).sum())))  # trivial; just report group stats

# ── Step 4: Gene-label shuffle null (100 permutations) ───────────────────────
print("\nComputing gene-label shuffle null (100 permutations)...")
rng = np.random.default_rng(42)
null_mean_lc = []
null_rho     = []
pairs_i = pairs_arr[:, 0]
pairs_j = pairs_arr[:, 1]

t_null = time.time()
for perm in range(100):
    shuf = rng.permutation(n_genes)
    shuf_lc = np.zeros(n_pairs, dtype=int)
    for li, knn_set in enumerate(knn_sets):
        ni = shuf[pairs_i]
        nj = shuf[pairs_j]
        for pi in range(n_pairs):
            a, b = min(int(ni[pi]), int(nj[pi])), max(int(ni[pi]), int(nj[pi]))
            if (a, b) in knn_set:
                shuf_lc[pi] += 1
    null_mean_lc.append(float(shuf_lc.mean()))
    r, _ = stats.spearmanr(shuf_lc, is_900.astype(float))
    null_rho.append(r)
    if perm % 20 == 0:
        print(f"  perm {perm+1}/100: mean_lc={null_mean_lc[-1]:.3f}, rho={r:.4f}  t={time.time()-t_null:.1f}s")

null_mean_lc_arr = np.array(null_mean_lc)
null_rho_arr     = np.array(null_rho)

real_mean_lc = float(layer_count.mean())
z_lc = (real_mean_lc - null_mean_lc_arr.mean()) / (null_mean_lc_arr.std() + 1e-12)
z_rho = (rho_all - null_rho_arr.mean()) / (null_rho_arr.std() + 1e-12)

print(f"\nReal mean layer_count: {real_mean_lc:.3f}")
print(f"Null mean layer_count: {null_mean_lc_arr.mean():.3f} ± {null_mean_lc_arr.std():.3f}")
print(f"Z (layer_count): {z_lc:.2f}")
print(f"\nReal Spearman rho: {rho_all:.4f}")
print(f"Null Spearman rho: {null_rho_arr.mean():.4f} ± {null_rho_arr.std():.4f}")
print(f"Z (rho): {z_rho:.2f}")

# ── Save results ──────────────────────────────────────────────────────────────
decay_df = pd.DataFrame([
    {'test': 'spearman_lc_vs_900tier_all',    'rho_or_z': rho_all, 'pval': p_all,
     'null_mean': null_rho_arr.mean(), 'null_std': null_rho_arr.std(), 'z_vs_null': z_rho,
     'n_pairs': n_pairs},
    {'test': 'mannwhitney_900plus_vs_700_900', 'rho_or_z': med_900 - med_700_900, 'pval': mw_p,
     'null_mean': float('nan'), 'null_std': float('nan'), 'z_vs_null': float('nan'),
     'n_pairs': n_pairs},
    {'test': 'null_z_mean_layer_count',        'rho_or_z': real_mean_lc, 'pval': float('nan'),
     'null_mean': null_mean_lc_arr.mean(), 'null_std': null_mean_lc_arr.std(), 'z_vs_null': z_lc,
     'n_pairs': n_pairs},
])
decay_df.to_csv(f'{OUT}/h3bc_decay_confidence.csv', index=False)

# Category channel stats (here: median layer_count per category by tier)
cat_stat_rows = []
for cat in ['always', 'early', 'late', 'L0_only', 'mixed', 'never']:
    m = np.array(categories) == cat
    if m.sum() == 0:
        continue
    for tier, tmask in [('all', np.ones(n_pairs, bool)), ('900plus', is_900), ('700_900', ~is_900)]:
        mm = m & tmask
        if mm.sum() == 0:
            continue
        cat_stat_rows.append({
            'category': cat, 'tier': tier, 'n': int(mm.sum()),
            'median_lc': float(np.median(layer_count[mm])),
            'mean_lc': float(np.mean(layer_count[mm])),
            'frac_900tier': float(is_900[mm].mean()),
        })
pd.DataFrame(cat_stat_rows).to_csv(f'{OUT}/h3bc_category_channel_stats.csv', index=False)

print(f"\nTotal time: {time.time()-t0_total:.1f}s")
print(f"Saved: h3bc_pair_adjacency.npy, h3bc_pair_categories.csv, h3bc_decay_confidence.csv, h3bc_category_channel_stats.csv")
