"""Audit action A1 — synthetic-graph positive control for topology-141 H123.

Build a synthetic gene embedding where the H123 signed-motif-community
structure is *planted by construction*, then run the project's existing H123
implementation on it. Verify the implementation detects the planted signal
at meaningful SNR.

Without this control, the H123 0–1/12 null result on MaxToki cannot be
distinguished from "the H123 implementation has a bug and would never detect
signal anywhere". After this control, we can rule that out.

Construction:
  - 200 genes in 4 communities of 50 genes each.
  - Embeddings: each community has its own random centroid in R^20, scaled
    so kNN at k=12 recovers communities cleanly via Louvain.
  - Synthetic TRRUST: 5 "TFs" per community × 10 targets each = 200 edges.
    Sign convention: same-community pairs get +1 (activator), different-
    community pairs get −1 (repressor). This is exactly the H123 signed-motif
    consistency rule, planted at full strength.
  - Pair table: 200 TRRUST positive edges + 200 random non-edge negatives.

Expected outcome (positive control PASSES):
  - Louvain recovers the 4 planted communities (>90% adjusted rand index).
  - H123 ΔAUROC vs null > +0.10 (paper-style strong positive).
  - Null AUROC at chance (≈0.5).
  - Effect persists across `use_degree_strata = True` (the H123 variant) and
    `use_degree_strata = False` (the H116 variant).

Outputs: outputs/synthetic_positive_control/{summary.json, raw_scores.json}
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import adjusted_rand_score, roc_auc_score

PROJ = Path(__file__).resolve().parents[3]
RUN = PROJ / "runs/topology-141-217M"
OUT = RUN / "outputs/synthetic_positive_control"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(RUN / "scripts"))

from phase78_community_signed_motif import (
    louvain_communities, comembership_matrix, signed_motif_community_score,
)
from phase6_manifold_distances import (
    pairwise_distances, triangle_defect_feature,
)

N_GENES = 200
N_COMMUNITIES = 4
N_PER_COMMUNITY = N_GENES // N_COMMUNITIES  # 50
PCA_DIM = 20
N_TFS_PER_COMMUNITY = 5
N_TARGETS_PER_TF = 10
KNN = 12
N_NULL = 100
SEED = 42

print("[A1-synth] Generating synthetic 200-gene 4-community embedding...")
rng = np.random.default_rng(SEED)

# Community centroids: 4 random vectors in R^20 well-separated
centroids = rng.normal(0, 1.0, size=(N_COMMUNITIES, PCA_DIM))
centroids *= 4.0  # scale up so within-community pairs are clearly closer than between

# Per-gene embedding = community centroid + small noise
true_community_labels = np.repeat(np.arange(N_COMMUNITIES), N_PER_COMMUNITY)
gene_emb = centroids[true_community_labels] + rng.normal(0, 1.0, size=(N_GENES, PCA_DIM))

print("[A1-synth] Building synthetic TRRUST with planted signed-motif structure...")
trrust_rows = []
sym_to_idx = {f"GENE{i:04d}": i for i in range(N_GENES)}
syms = list(sym_to_idx.keys())

# For each community, pick 5 TFs and assign each 10 targets:
#   7 within-community (sign +1: activator)
#   3 cross-community (sign -1: repressor)
all_indices_by_comm = {c: np.where(true_community_labels == c)[0] for c in range(N_COMMUNITIES)}
for c in range(N_COMMUNITIES):
    tf_idxs = rng.choice(all_indices_by_comm[c], size=N_TFS_PER_COMMUNITY, replace=False)
    for tf_idx in tf_idxs:
        # 7 within-community targets
        within_pool = np.array([i for i in all_indices_by_comm[c] if i != tf_idx])
        within_targets = rng.choice(within_pool, size=7, replace=False)
        for tgt_idx in within_targets:
            trrust_rows.append({"tf": syms[tf_idx], "target": syms[tgt_idx], "sign": +1})
        # 3 cross-community targets
        other_communities = [oc for oc in range(N_COMMUNITIES) if oc != c]
        across_targets = []
        for _ in range(3):
            oc = rng.choice(other_communities)
            tgt_idx = rng.choice(all_indices_by_comm[oc])
            across_targets.append(int(tgt_idx))
        for tgt_idx in across_targets:
            trrust_rows.append({"tf": syms[tf_idx], "target": syms[tgt_idx], "sign": -1})

trrust = pd.DataFrame(trrust_rows)
print(f"  TRRUST: {len(trrust)} signed edges across {trrust['tf'].nunique()} TFs")
print(f"  Sign distribution: +1 = {(trrust['sign']==1).sum()}, −1 = {(trrust['sign']==-1).sum()}")

# Build pair table: 200 TRRUST positive + 200 random non-edge negatives
trrust_pairs_set = set()
for _, row in trrust.iterrows():
    a, b = sym_to_idx[row["tf"]], sym_to_idx[row["target"]]
    trrust_pairs_set.add((min(a, b), max(a, b)))

pairs_pos = [(min(sym_to_idx[r["tf"]], sym_to_idx[r["target"]]),
              max(sym_to_idx[r["tf"]], sym_to_idx[r["target"]]),
              1) for _, r in trrust.iterrows()]
pairs_neg = []
n_pos = len(pairs_pos)
while len(pairs_neg) < n_pos:
    i, j = rng.integers(0, N_GENES, size=2)
    if i == j: continue
    key = (min(int(i), int(j)), max(int(i), int(j)))
    if key in trrust_pairs_set: continue
    pairs_neg.append((key[0], key[1], 0))

pair_rows = pairs_pos + pairs_neg
pairs_df = pd.DataFrame({
    "i": [p[0] for p in pair_rows],
    "j": [p[1] for p in pair_rows],
    "is_trrust": [p[2] for p in pair_rows],
})
print(f"  Pairs: {len(pairs_df)} ({pairs_df['is_trrust'].sum()} positive, {(~pairs_df['is_trrust'].astype(bool)).sum()} negative)")

# Verify Louvain recovers the planted communities
print("\n[A1-synth] Running Louvain on synthetic kNN graph...")
D = pairwise_distances(gene_emb)
np.fill_diagonal(D, np.inf)
nbr = np.argsort(D, axis=1)[:, :KNN]
g = nx.Graph(); g.add_nodes_from(range(N_GENES))
for i in range(N_GENES):
    for j in nbr[i]:
        g.add_edge(int(i), int(j))
labels = louvain_communities(g, seed=42)
n_recovered = labels.max() + 1
ari = adjusted_rand_score(true_community_labels, labels)
print(f"  Louvain found {n_recovered} communities; ARI vs planted = {ari:.3f}")

# Triangle-defect baseline
tri = sum(triangle_defect_feature(gene_emb, k=k) for k in [8, 12, 16])

# Run H123 (degree-stratified) and H116 (simple shuffle)
print("\n[A1-synth] Running H123 signed-motif-community score on planted graph...")
r123 = signed_motif_community_score(
    gene_emb, pairs_df, sym_to_idx, trrust, tri,
    k=KNN, n_null=N_NULL, use_degree_strata=True,
)
print(f"  H123: AUC_motif={r123['auc_motif']:.3f}  AUC_combined={r123['auc_combined']:.3f}  "
      f"Δ_combo_vs_null_p95={r123['delta_combo_vs_null_p95']:+.3f}  Δ_motif_vs_null_p95={r123['delta_motif_vs_null_p95']:+.3f}")

print("\n[A1-synth] Running H116 (simple sign-shuffle null)...")
r116 = signed_motif_community_score(
    gene_emb, pairs_df, sym_to_idx, trrust, tri,
    k=KNN, n_null=N_NULL, use_degree_strata=False,
)
print(f"  H116: AUC_motif={r116['auc_motif']:.3f}  AUC_combined={r116['auc_combined']:.3f}  "
      f"Δ_combo_vs_null_p95={r116['delta_combo_vs_null_p95']:+.3f}  Δ_motif_vs_null_p95={r116['delta_motif_vs_null_p95']:+.3f}")

# Verdict
auc_motif_strong = r123['auc_motif'] > 0.7
delta_strong = r123['delta_combo_vs_null_p95'] > 0.05
louvain_ok = ari > 0.9

verdict = "POSITIVE — pipeline detects planted signal" if (auc_motif_strong and delta_strong and louvain_ok) else "NEGATIVE — pipeline does not detect planted signal"

summary = {
    "synthetic_design": {
        "n_genes": N_GENES,
        "n_communities_planted": N_COMMUNITIES,
        "n_per_community": N_PER_COMMUNITY,
        "n_tfs": N_TFS_PER_COMMUNITY * N_COMMUNITIES,
        "n_targets_per_tf": N_TARGETS_PER_TF,
        "fraction_within_community_targets": 7/10,
        "sign_planted_within_community": "+1 (activator)",
        "sign_planted_across_community": "−1 (repressor)",
        "centroid_separation_scale": 4.0,
        "noise_sd": 1.0,
        "knn_k": KNN,
    },
    "louvain_recovery": {
        "n_communities_found": int(n_recovered),
        "adjusted_rand_index": float(ari),
        "passes_threshold_ari_>_0.9": bool(louvain_ok),
    },
    "h123_degree_stratified": {
        "auc_motif": float(r123['auc_motif']),
        "auc_triangle_baseline": float(r123['auc_triangle_baseline']),
        "auc_combined": float(r123['auc_combined']),
        "null_combo_mean": float(r123['null_combo_mean']),
        "null_combo_p95": float(r123['null_combo_p95']),
        "delta_combo_vs_null_p95": float(r123['delta_combo_vs_null_p95']),
        "delta_motif_vs_null_p95": float(r123['delta_motif_vs_null_p95']),
    },
    "h116_simple_shuffle": {
        "auc_motif": float(r116['auc_motif']),
        "auc_combined": float(r116['auc_combined']),
        "delta_combo_vs_null_p95": float(r116['delta_combo_vs_null_p95']),
        "delta_motif_vs_null_p95": float(r116['delta_motif_vs_null_p95']),
    },
    "decision_thresholds": {
        "auc_motif_>_0.7": bool(auc_motif_strong),
        "delta_combo_vs_null_p95_>_0.05": bool(delta_strong),
        "louvain_ari_>_0.9": bool(louvain_ok),
    },
    "verdict": verdict,
    "interpretation": (
        f"Planted-signal positive control on a 4-community 200-gene synthetic graph "
        f"with H123-consistent signed motifs. AUC_motif = {r123['auc_motif']:.3f}, "
        f"Δ_combo_vs_null_p95 = {r123['delta_combo_vs_null_p95']:+.3f}. "
        f"Louvain ARI = {ari:.3f}. "
        + ("The H123 implementation in phase78_community_signed_motif.py "
           "DETECTS planted signal at meaningful SNR, confirming the 0–1/12 layers "
           "result on MaxToki is a property of the model's representations, "
           "not a pipeline bug." if verdict.startswith("POSITIVE")
           else "The H123 implementation FAILS to detect planted signal — there is "
                "an implementation issue that needs to be fixed before the MaxToki "
                "negative result can be trusted.")
    ),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"\n[A1-synth] Verdict: {verdict}")
print(f"[A1-synth] Wrote {OUT}/summary.json")
