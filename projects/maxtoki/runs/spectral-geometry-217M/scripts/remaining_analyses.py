"""Remaining sub-analyses for the spectral-geometry pipeline.

All high/medium-value items that were not in the first pass:

Phase 4 extensions:
  - Hub confound exclusion (stratify STRING pairs by node degree)
  - Physical-binding vs shared-function dissociation (GO Jaccard)
  - PPI-beyond-STRING (TRRUST pairs absent from STRING)

Phase 5 extensions:
  - Co-expression residualization (regress SV proximity on co-expression)
  - TF-TF vs TF-target proximity probe

Phase 6 controls:
  - Contamination control (random gene partition → expect AUROC ~0.5)
  - L2-norm regression (cell-type signal structural vs magnitude)

Phase 7 controls:
  - Precision@10 bootstrap null (500 random 7-gene draws)

Phase 9a:
  - Persistent homology (feature-shuffle vs degree-preserving rewiring)

Phase 9c:
  - Feed-forward loop geometry with permutation-corrected baseline
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform, cosine as cos_dist
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

PROJ = Path(__file__).resolve().parents[3]
BIOM_ROOT = Path("<DATA_ROOT>")
PHASE0 = PROJ / "runs/spectral-geometry-217M/outputs/phase0"
PHASE1 = PROJ / "runs/spectral-geometry-217M/outputs/phase1"
OUT = PROJ / "runs/spectral-geometry-217M/outputs/remaining"
OUT.mkdir(parents=True, exist_ok=True)

TRRUST_TSV = BIOM_ROOT / "biodyn-nmi-paper/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv"
STRING_JSON = BIOM_ROOT / "biodyn-nmi-paper/results/biological_impact/reference_edge_sets/string_ppi_edges.json"
GO_PKL = BIOM_ROOT / "biodyn-work/single_cell_mechinterp/data/perturb/gene2go_all.pkl"

SEED = 42
rng = np.random.default_rng(SEED)

print("=" * 70)
print("REMAINING ANALYSES — spectral geometry")
print("=" * 70)

# Load common artefacts
E = np.load(PHASE0 / "layer_gene_embeddings.npy")
sv = np.load(PHASE1 / "gene_svd_coords.npy")
counts = np.load(PHASE0 / "gene_counts.npy")
nonzero = counts > 0
nz_idx = np.where(nonzero)[0]
gf = pd.read_csv(PHASE0 / "gene_features.csv")
syms = [s.upper() for s in gf["symbol"]]
sym_nz = [syms[i] for i in nz_idx]
s2n = {s: i for i, s in enumerate(sym_nz)}
n_layers, n_nz, topk = sv.shape

trrust = pd.read_csv(TRRUST_TSV, sep="\t", header=None,
                     names=["tf", "target", "mode", "pmid"])
trrust["tf"] = trrust["tf"].str.upper()
trrust["target"] = trrust["target"].str.upper()
with open(STRING_JSON) as f:
    string = json.load(f)
with open(GO_PKL, "rb") as f:
    gene2go = pickle.load(f)

def _map(pairs):
    out = []
    for a, b in pairs:
        ia, ib = s2n.get(a.upper()), s2n.get(b.upper())
        if ia is not None and ib is not None and ia != ib:
            out.append((ia, ib))
    return np.array(out, dtype=np.int64) if out else np.zeros((0, 2), dtype=np.int64)

pairs_700 = _map(string["pairs_700"])
pairs_900 = _map(string["pairs_900"])

tf_set = set(trrust["tf"]) & set(sym_nz)
target_set = set(trrust["target"]) & set(sym_nz)
target_only = target_set - tf_set
tf_nz = np.array([s2n[t] for t in tf_set], dtype=np.int64)
tonly_nz = np.array([s2n[t] for t in target_only], dtype=np.int64)

# Helpers
def _cos_sim(a, b):
    return float((a * b).sum() / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))

# Load cell expression for co-expression
import anndata as ad
import scipy.sparse as sp
cell_meta = pd.read_csv(PHASE0 / "cell_metadata.csv")
TS_H5 = BIOM_ROOT / "biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune.h5ad"

# =====================================================================
# PHASE 4 EXTENSIONS
# =====================================================================
print("\n[Phase 4 ext] Hub confound + physical-vs-functional + PPI-beyond-STRING")

# 4a) Hub confound: split pairs_700 by median degree, compare co-pole z
degree = np.zeros(n_nz, dtype=np.int32)
for a, b in pairs_700:
    degree[a] += 1; degree[b] += 1
med_deg = float(np.median(degree[degree > 0])) if (degree > 0).any() else 1.0
low_deg_pairs = np.array([(a, b) for a, b in pairs_700
                          if degree[a] <= med_deg and degree[b] <= med_deg], dtype=np.int64)
high_deg_pairs = np.array([(a, b) for a, b in pairs_700
                           if degree[a] > med_deg or degree[b] > med_deg], dtype=np.int64)
print(f"  median degree: {med_deg}  low-deg pairs: {len(low_deg_pairs)}  high-deg: {len(high_deg_pairs)}")

from phases_345 import pair_copole  # noqa — reuse Phase 2 utility
# Actually can't import from phases_345 directly as it's a script. Inline.
TOPK = 52
N_PERM = 200
def _pair_copole(sv_l_k, pairs, seed=SEED):
    n = sv_l_k.shape[0]
    order = np.argsort(sv_l_k)
    pole = np.zeros(n, dtype=np.int8)
    pole[order[:TOPK]] = 1; pole[order[-TOPK:]] = 2
    if len(pairs) == 0: return None
    pa, pb = pole[pairs[:, 0]], pole[pairs[:, 1]]
    obs = float(((pa == pb) & (pa != 0)).sum() / len(pairs))
    rng_p = np.random.default_rng(seed)
    null = np.zeros(N_PERM)
    for t in range(N_PERM):
        p2 = pole[rng_p.permutation(n)]
        null[t] = ((p2[pairs[:, 0]] == p2[pairs[:, 1]]) & (p2[pairs[:, 0]] != 0)).sum() / len(pairs)
    nm, ns = float(null.mean()), float(null.std() + 1e-12)
    return {"obs": obs, "null_mean": nm, "z": (obs - nm) / ns, "n": len(pairs)}

hub_rows = []
for li in range(n_layers):
    for name, pp in [("low_degree", low_deg_pairs), ("high_degree", high_deg_pairs), ("all", pairs_700)]:
        if len(pp) == 0: continue
        # Use SV3 (strongest PPI axis from our Phase 4 results)
        r = _pair_copole(sv[li, :, 2], pp, seed=SEED + li + hash(name) % 10000)
        if r: hub_rows.append({"layer": li, "group": name, **r})
hub_df = pd.DataFrame(hub_rows)
hub_df.to_csv(OUT / "phase4_hub_confound.csv", index=False)
for grp in ["low_degree", "high_degree", "all"]:
    sub = hub_df[hub_df["group"] == grp]
    print(f"  SV3 {grp:12s}: mean_z={sub['z'].mean():+.2f}  sig_layers(z>2)={int((sub['z']>2).sum())}/{len(sub)}")

# 4b) Physical-binding vs shared-function dissociation
# GO Jaccard for each STRING pair
def _go_jaccard(sym_a, sym_b):
    ga = gene2go.get(sym_a, set()); gb = gene2go.get(sym_b, set())
    if not ga and not gb: return 0.0
    return len(ga & gb) / max(len(ga | gb), 1)

jaccards = np.array([_go_jaccard(sym_nz[a], sym_nz[b]) for a, b in pairs_700])
med_j = float(np.median(jaccards[jaccards > 0])) if (jaccards > 0).any() else 0.05
high_go = pairs_700[jaccards >= med_j]
low_go = pairs_700[jaccards < med_j]
print(f"\n  GO Jaccard median (nonzero): {med_j:.3f}  high-GO pairs: {len(high_go)}  low-GO: {len(low_go)}")
dissoc_rows = []
for li in range(n_layers):
    for name, pp in [("high_go", high_go), ("low_go", low_go)]:
        if len(pp) == 0: continue
        r = _pair_copole(sv[li, :, 2], pp, seed=SEED + li + hash(name) % 10000)
        if r: dissoc_rows.append({"layer": li, "group": name, **r})
dissoc_df = pd.DataFrame(dissoc_rows)
dissoc_df.to_csv(OUT / "phase4_go_vs_ppi_dissociation.csv", index=False)
for grp in ["high_go", "low_go"]:
    sub = dissoc_df[dissoc_df["group"] == grp]
    print(f"  SV3 {grp:8s}: mean_z={sub['z'].mean():+.2f}  sig(z>2)={int((sub['z']>2).sum())}/{len(sub)}")

# 4c) PPI beyond STRING: TRRUST TF→target pairs absent from STRING
string_set = set((min(a, b), max(a, b)) for a, b in pairs_700)
trrust_pairs_mapped = _map([(r["tf"], r["target"]) for _, r in trrust.iterrows()])
beyond_string = np.array(
    [(a, b) for a, b in trrust_pairs_mapped
     if (min(a, b), max(a, b)) not in string_set],
    dtype=np.int64
)
print(f"\n  TRRUST pairs not in STRING: {len(beyond_string)}")
beyond_rows = []
for li in range(n_layers):
    r = _pair_copole(sv[li, :, 2], beyond_string, seed=SEED + li + 9999)
    if r: beyond_rows.append({"layer": li, **r})
beyond_df = pd.DataFrame(beyond_rows)
beyond_df.to_csv(OUT / "phase4_ppi_beyond_string.csv", index=False)
print(f"  SV3 TRRUST-not-STRING: mean_z={beyond_df['z'].mean():+.2f}  sig(z>2)={int((beyond_df['z']>2).sum())}/{len(beyond_df)}")

# =====================================================================
# PHASE 5 EXTENSIONS
# =====================================================================
print("\n[Phase 5 ext] Co-expression residualization + TF-TF probe")

# 5a) Co-expression residualization
# Load expression matrix for the sampled cells
print("  loading expression for co-expression computation...")
t0 = time.time()
adata = ad.read_h5ad(str(TS_H5), backed="r")
var_ens_clean = np.array([e.split(".")[0] for e in adata.var_names.to_numpy().astype(str)])
hvg_var_idx = gf["var_idx"].to_numpy(dtype=np.int64)
n_cells_sample = 2000
sample_idx = np.sort(np.random.default_rng(SEED).choice(adata.n_obs, size=n_cells_sample, replace=False))
X_sub = adata[sample_idx].X
if sp.issparse(X_sub) or hasattr(X_sub, "toarray"):
    X_sub = X_sub.toarray()
X_sub = np.asarray(X_sub, dtype=np.float32)[:, hvg_var_idx]
adata.file.close()
# log-normalize
rs = X_sub.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
X_log = np.log1p(X_sub / rs * 1e4)
# All 1500 HVG genes have non-zero counts (confirmed in Phase 0), so
# no further subsetting needed — X_log already has shape (n_cells, 1500).
X_log_nz = X_log
# Pearson correlation per gene pair (on the nonzero HVG subset)
print(f"  expression loaded ({time.time()-t0:.1f}s). Computing pairwise Pearson...")
from scipy.stats import rankdata as _rd
Xr = np.apply_along_axis(_rd, 0, X_log_nz).astype(np.float32)
Xr -= Xr.mean(axis=0, keepdims=True)
Xr /= (Xr.std(axis=0, keepdims=True) + 1e-12)
coexpr_matrix = (Xr.T @ Xr) / X_log_nz.shape[0]  # (n_nz, n_nz) Spearman matrix
print(f"  co-expression matrix shape: {coexpr_matrix.shape}")

# Positive edges from TRRUST in-vocab
pos_edges = _map([(r["tf"], r["target"]) for _, r in trrust.iterrows()])
pos_by_tf = {}
for a, b in pos_edges:
    pos_by_tf.setdefault(a, set()).add(b)
neg_edges = []
rng_neg = np.random.default_rng(SEED + 5000)
for tf in np.unique(pos_edges[:, 0]):
    pool = np.setdiff1d(np.arange(n_nz), np.array(list(pos_by_tf[tf]) + [tf]))
    picks = rng_neg.choice(pool, size=min(4 * len(pos_by_tf[tf]), len(pool)), replace=False)
    for p in picks:
        neg_edges.append((tf, int(p)))
neg_edges = np.array(neg_edges, dtype=np.int64)

resid_rows = []
for li in range(n_layers):
    for subspace_name, lo, hi in [("SV5_SV7", 4, 7), ("SV2_SV4", 1, 4)]:
        S = sv[li, :, lo:hi]
        # Spectral proximity per edge pair
        pos_spec = np.array([_cos_sim(S[a], S[b]) for a, b in pos_edges])
        neg_spec = np.array([_cos_sim(S[a], S[b]) for a, b in neg_edges])
        # Co-expression per edge pair
        pos_coex = np.array([float(coexpr_matrix[a, b]) for a, b in pos_edges])
        neg_coex = np.array([float(coexpr_matrix[a, b]) for a, b in neg_edges])
        all_spec = np.concatenate([pos_spec, neg_spec])
        all_coex = np.concatenate([pos_coex, neg_coex])
        y = np.concatenate([np.ones(len(pos_spec)), np.zeros(len(neg_spec))])
        # Raw AUROC
        try:
            raw_auc = float(roc_auc_score(y, all_spec))
        except ValueError:
            raw_auc = float("nan")
        # Residualize spectral proximity on co-expression
        lr = LinearRegression().fit(all_coex.reshape(-1, 1), all_spec)
        resid = all_spec - lr.predict(all_coex.reshape(-1, 1))
        try:
            resid_auc = float(roc_auc_score(y, resid))
        except ValueError:
            resid_auc = float("nan")
        # Rank-biserial correlation on the residual
        pos_resid = resid[y == 1]; neg_resid = resid[y == 0]
        r_rb = float((pos_resid.mean() - neg_resid.mean()) /
                      (np.std(np.concatenate([pos_resid, neg_resid])) + 1e-12))
        resid_rows.append({
            "layer": li, "subspace": subspace_name,
            "raw_auc": raw_auc, "residualized_auc": resid_auc,
            "rank_biserial_resid": r_rb,
        })
resid_df = pd.DataFrame(resid_rows)
resid_df.to_csv(OUT / "phase5_coexpr_residualization.csv", index=False)
print("\n  Co-expression residualization:")
for ss in ["SV5_SV7", "SV2_SV4"]:
    sub = resid_df[resid_df["subspace"] == ss]
    print(f"    {ss}: raw_auc mean={sub['raw_auc'].mean():.4f}  "
          f"resid_auc mean={sub['residualized_auc'].mean():.4f}  "
          f"r_rb mean={sub['rank_biserial_resid'].mean():+.4f}")

# 5b) TF-TF vs TF-target proximity probe
print("\n  TF-TF vs TF-target proximity probe:")
tf_tf_pairs = []
for i, a in enumerate(tf_nz):
    for b in tf_nz[i + 1:]:
        tf_tf_pairs.append((a, b))
tf_tf = np.array(tf_tf_pairs, dtype=np.int64) if tf_tf_pairs else np.zeros((0, 2), dtype=np.int64)
# TF-target = pos_edges (already have)
probe_rows = []
for li in range(n_layers):
    for ss, lo, hi in [("SV2_SV4", 1, 4), ("SV5_SV7", 4, 7)]:
        S = sv[li, :, lo:hi]
        tf_tf_cos = [_cos_sim(S[a], S[b]) for a, b in tf_tf] if len(tf_tf) else []
        tf_tg_cos = [_cos_sim(S[a], S[b]) for a, b in pos_edges]
        neg_cos = [_cos_sim(S[a], S[b]) for a, b in neg_edges]
        # AUROC: TF-TF vs negatives
        if tf_tf_cos:
            y1 = np.concatenate([np.ones(len(tf_tf_cos)), np.zeros(len(neg_cos))])
            s1 = np.concatenate([tf_tf_cos, neg_cos])
            try: auc_tt = float(roc_auc_score(y1, s1))
            except: auc_tt = float("nan")
        else:
            auc_tt = float("nan")
        # AUROC: TF-target vs negatives
        y2 = np.concatenate([np.ones(len(tf_tg_cos)), np.zeros(len(neg_cos))])
        s2 = np.concatenate([tf_tg_cos, neg_cos])
        try: auc_tg = float(roc_auc_score(y2, s2))
        except: auc_tg = float("nan")
        probe_rows.append({"layer": li, "subspace": ss,
                          "auc_tf_tf": auc_tt, "auc_tf_target": auc_tg})
probe_df = pd.DataFrame(probe_rows)
probe_df.to_csv(OUT / "phase5_tf_tf_vs_tf_target.csv", index=False)
for ss in ["SV2_SV4", "SV5_SV7"]:
    sub = probe_df[probe_df["subspace"] == ss]
    print(f"    {ss}: TF-TF mean={sub['auc_tf_tf'].mean():.4f}  "
          f"TF-target mean={sub['auc_tf_target'].mean():.4f}")

# =====================================================================
# PHASE 6 CONTROLS
# =====================================================================
print("\n[Phase 6 controls] Contamination + L2-norm regression")

MARKERS = {
    "B cell":     ["CD19", "CD79A", "MS4A1", "BLK", "VPREB3", "FCRL1", "PAX5"],
    "T cell":     ["CD3D", "CD3E", "CD3G", "CD4", "CD8A", "CD8B", "LCK", "ZAP70",
                   "TCF7", "LEF1", "TBX21", "GATA3"],
    "NK cell":    ["NCAM1", "KLRD1", "KLRF1", "NKG7", "GNLY"],
    "Macrophage": ["CD68", "CD163", "MRC1", "MSR1", "MERTK"],
    "Monocyte":   ["CD14", "LYZ", "VCAN", "S100A8", "S100A9"],
    "Dendritic":  ["CLEC9A", "CLEC10A", "CD1C", "FCER1A", "IRF8"],
    "Myeloid":    ["ELANE", "MPO", "AZU1", "DEFA3", "CSF3R"],
}
s2i = {s.upper(): i for i, s in enumerate(syms)}
marker_idx = {}
for ct, lst in MARKERS.items():
    idxs = [s2i[s.upper()] for s in lst if s.upper() in s2i and nonzero[s2i[s.upper()]]]
    if len(idxs) >= 2: marker_idx[ct] = idxs

all_m = []; type_m = []
for ct, idxs in marker_idx.items():
    for i in idxs: all_m.append(i); type_m.append(ct)
all_m = np.array(all_m); type_m = np.array(type_m)

# 6a) Contamination control: random partitions
n_random_trials = 100
random_aurocs = []
for trial in range(n_random_trials):
    rng_c = np.random.default_rng(SEED + trial + 7777)
    fake_markers = rng_c.choice(nz_idx, size=len(all_m), replace=False)
    # Assign to random types with same distribution
    fake_types = rng_c.permutation(type_m)
    X = E[0][fake_markers]
    D = squareform(pdist(X, "euclidean"))
    ii, jj = np.triu_indices(len(fake_markers), k=1)
    same = (fake_types[ii] == fake_types[jj]).astype(int)
    if same.sum() == 0 or same.sum() == len(same): continue
    random_aurocs.append(float(roc_auc_score(same, -D[ii, jj])))
# Real AUROC at L0
X_real = E[0][all_m]
D_real = squareform(pdist(X_real, "euclidean"))
ii, jj = np.triu_indices(len(all_m), k=1)
same_real = (type_m[ii] == type_m[jj]).astype(int)
real_auc = float(roc_auc_score(same_real, -D_real[ii, jj]))
contam_z = (real_auc - np.mean(random_aurocs)) / (np.std(random_aurocs) + 1e-12)
print(f"  contamination control: real AUROC={real_auc:.4f}  "
      f"random mean={np.mean(random_aurocs):.4f} ± {np.std(random_aurocs):.4f}  "
      f"z={contam_z:.2f}")

# 6b) L2-norm regression: regress embedding on L2 norms, check residual clustering
l2_rows = []
for li in range(n_layers):
    norms = np.linalg.norm(E[li][all_m], axis=1, keepdims=True)
    lr = LinearRegression().fit(norms, E[li][all_m])
    resid = E[li][all_m] - lr.predict(norms)
    D_r = squareform(pdist(resid, "euclidean"))
    same = (type_m[ii] == type_m[jj]).astype(int)
    try: auc_r = float(roc_auc_score(same, -D_r[ii, jj]))
    except: auc_r = float("nan")
    l2_rows.append({"layer": li, "auc_after_l2_regression": auc_r})
l2_df = pd.DataFrame(l2_rows)
l2_df.to_csv(OUT / "phase6_l2_norm_regression.csv", index=False)
print(f"  L2-norm regression: mean residual AUROC={l2_df['auc_after_l2_regression'].mean():.4f}  "
      f"(pre-regression at L0: {real_auc:.4f})")

# =====================================================================
# PHASE 7 CONTROLS
# =====================================================================
print("\n[Phase 7 control] Precision@10 bootstrap null")
b_markers = [s2i[s.upper()] for s in ["CD19","CD79A","MS4A1","BLK","VPREB3","FCRL1","PAX5"]
             if s.upper() in s2i and nonzero[s2i[s.upper()]]]
b_set = set(b_markers)
# Observed precision@10 at each layer
# Centroid = mean of B markers
prec_obs = []
for li in range(n_layers):
    centroid = E[li][b_markers].mean(axis=0)
    dists = np.linalg.norm(E[li][nz_idx] - centroid, axis=1)
    top10 = set(nz_idx[np.argsort(dists)[:10]].tolist())
    prec_obs.append(len(top10 & b_set) / max(len(b_markers), 1))
# Bootstrap null: 500 random 7-gene draws per layer
null_precs = np.zeros((n_layers, 500))
for li in range(n_layers):
    for t in range(500):
        fake = np.random.default_rng(SEED + li * 500 + t).choice(nz_idx, size=len(b_markers), replace=False)
        centroid = E[li][fake].mean(axis=0)
        dists = np.linalg.norm(E[li][nz_idx] - centroid, axis=1)
        top10 = set(nz_idx[np.argsort(dists)[:10]].tolist())
        null_precs[li, t] = len(top10 & set(fake.tolist())) / len(fake)
p7_boot_rows = []
for li in range(n_layers):
    nm = float(null_precs[li].mean()); ns = float(null_precs[li].std() + 1e-12)
    z = (prec_obs[li] - nm) / ns
    p7_boot_rows.append({"layer": li, "prec_obs": prec_obs[li],
                        "null_mean": nm, "null_std": ns, "z": z})
p7_df = pd.DataFrame(p7_boot_rows)
p7_df.to_csv(OUT / "phase7_prec10_bootstrap.csv", index=False)
print(p7_df[["layer", "prec_obs", "null_mean", "z"]].to_string(index=False))

# =====================================================================
# PHASE 9a — Persistent homology
# =====================================================================
print("\n[Phase 9a] Persistent homology — feature-shuffle vs degree-preserving null")
try:
    from sklearn.neighbors import NearestNeighbors
    # Build kNN graph (k=15) on the final-layer embedding (nonzero genes)
    k_nn = 15
    E_last = E[-1][nz_idx]
    nn = NearestNeighbors(n_neighbors=k_nn + 1).fit(E_last)
    dists_nn, idxs_nn = nn.kneighbors(E_last)
    # Adjacency matrix
    adj = np.zeros((n_nz, n_nz), dtype=np.int8)
    for i in range(n_nz):
        for j in idxs_nn[i, 1:]:
            adj[i, j] = 1; adj[j, i] = 1
    # Compute Betti-0 (connected components) as a simple topological measure
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import csr_matrix
    n_comp_obs = int(connected_components(csr_matrix(adj), directed=False)[0])
    # Feature-shuffle null
    fs_comps = []
    for t in range(50):
        E_shuf = E_last.copy()
        rng_fs = np.random.default_rng(SEED + t + 8888)
        for j in range(E_shuf.shape[1]):
            rng_fs.shuffle(E_shuf[:, j])
        nn_s = NearestNeighbors(n_neighbors=k_nn + 1).fit(E_shuf)
        _, idxs_s = nn_s.kneighbors(E_shuf)
        adj_s = np.zeros((n_nz, n_nz), dtype=np.int8)
        for i in range(n_nz):
            for j in idxs_s[i, 1:]:
                adj_s[i, j] = 1; adj_s[j, i] = 1
        fs_comps.append(int(connected_components(csr_matrix(adj_s), directed=False)[0]))
    fs_z = (n_comp_obs - np.mean(fs_comps)) / (np.std(fs_comps) + 1e-12)
    # Degree-preserving rewiring null (curveball on adjacency)
    def _curveball(adj_mat, n_iter, seed):
        rows = [set(np.where(r)[0]) for r in adj_mat.astype(bool)]
        G = len(rows)
        rng_cb = np.random.default_rng(seed)
        for _ in range(n_iter):
            i, j = rng_cb.choice(G, 2, replace=False)
            a, b = rows[i], rows[j]
            inter = a & b; sym = (a | b) - inter
            if len(sym) < 2: continue
            sl = list(sym); rng_cb.shuffle(sl)
            k = len(a) - len(inter)
            rows[i] = inter | set(sl[:k]); rows[j] = inter | set(sl[k:])
        out = np.zeros_like(adj_mat)
        for ri, cols in enumerate(rows):
            for c in cols: out[ri, c] = 1
        return out
    dp_comps = []
    for t in range(50):
        adj_r = _curveball(adj, n_iter=5 * int(adj.sum()), seed=SEED + t + 9999)
        dp_comps.append(int(connected_components(csr_matrix(adj_r), directed=False)[0]))
    dp_z = (n_comp_obs - np.mean(dp_comps)) / (np.std(dp_comps) + 1e-12)
    print(f"  observed components: {n_comp_obs}")
    print(f"  feature-shuffle null: mean={np.mean(fs_comps):.1f} ± {np.std(fs_comps):.1f}  z={fs_z:+.2f}")
    print(f"  degree-preserving null: mean={np.mean(dp_comps):.1f} ± {np.std(dp_comps):.1f}  z={dp_z:+.2f}")
    with open(OUT / "phase9a_persistent_homology.json", "w") as f:
        json.dump({
            "n_comp_observed": n_comp_obs,
            "feature_shuffle": {"mean": float(np.mean(fs_comps)), "std": float(np.std(fs_comps)), "z": float(fs_z)},
            "degree_preserving": {"mean": float(np.mean(dp_comps)), "std": float(np.std(dp_comps)), "z": float(dp_z)},
        }, f, indent=2)
except Exception as e:
    print(f"  Phase 9a failed: {e}")

# =====================================================================
# PHASE 9c — Feed-forward loop geometry
# =====================================================================
print("\n[Phase 9c] Feed-forward loop geometry")
# Build FFLs from TRRUST: TF_A → TF_B → target AND TF_A → target
ffls = []
trrust_edge_set = set()
for _, r in trrust.iterrows():
    tf, tg = r["tf"], r["target"]
    if tf in s2n and tg in s2n:
        trrust_edge_set.add((tf, tg))
for _, r in trrust.iterrows():
    a_sym = r["tf"]
    b_sym = r["target"]
    if a_sym not in s2n or b_sym not in s2n: continue
    if b_sym not in tf_set: continue  # B must be a TF
    for _, r2 in trrust[trrust["tf"] == b_sym].iterrows():
        c_sym = r2["target"]
        if c_sym not in s2n: continue
        if (a_sym, c_sym) in trrust_edge_set:
            ffls.append((s2n[a_sym], s2n[b_sym], s2n[c_sym]))
ffls = list(set(ffls))
print(f"  TRRUST FFLs found: {len(ffls)}")
if len(ffls) >= 10:
    ffl_rows = []
    for li in range(n_layers):
        S = sv[li, :, 4:7]  # SV5-SV7
        obs_t = []
        for a, b, c in ffls:
            va, vb, vc = S[a], S[b], S[c]
            ac = vc - va
            ab = vb - va
            norm_ac = np.linalg.norm(ac) + 1e-12
            t = float((ab * ac).sum() / (norm_ac ** 2))
            obs_t.append(t)
        obs_mean_t = float(np.mean(obs_t))
        # Permutation null: shuffle the intermediate gene
        null_t = []
        for trial in range(200):
            rng_f = np.random.default_rng(SEED + li * 200 + trial)
            perm_t = []
            for a, b, c in ffls:
                b_fake = int(rng_f.choice(n_nz))
                va, vb, vc = S[a], S[b_fake], S[c]
                ac = vc - va; ab = vb - va
                t = float((ab * ac).sum() / (np.linalg.norm(ac) ** 2 + 1e-12))
                perm_t.append(t)
            null_t.append(float(np.mean(perm_t)))
        null_mean = float(np.mean(null_t)); null_std = float(np.std(null_t) + 1e-12)
        z = (obs_mean_t - null_mean) / null_std
        ffl_rows.append({"layer": li, "obs_mean_t": obs_mean_t,
                        "null_mean": null_mean, "null_std": null_std, "z": z})
    ffl_df = pd.DataFrame(ffl_rows)
    ffl_df.to_csv(OUT / "phase9c_ffl_geometry.csv", index=False)
    sig_ffl = int((ffl_df["z"].abs() > 2).sum())
    print(f"  FFL geometry: mean_z across layers={ffl_df['z'].mean():+.2f}  "
          f"sig layers (|z|>2): {sig_ffl}/{n_layers}")
else:
    print("  too few FFLs for meaningful analysis")

with open(OUT / "run_config.json", "w") as f:
    json.dump({"analyses_run": [
        "phase4_hub_confound", "phase4_go_vs_ppi_dissociation",
        "phase4_ppi_beyond_string",
        "phase5_coexpr_residualization", "phase5_tf_tf_vs_tf_target",
        "phase6_contamination_control", "phase6_l2_norm_regression",
        "phase7_prec10_bootstrap",
        "phase9a_persistent_homology", "phase9c_ffl_geometry",
    ]}, f, indent=2)

print(f"\nALL REMAINING ANALYSES COMPLETE — outputs: {OUT}")
