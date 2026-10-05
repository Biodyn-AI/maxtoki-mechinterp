"""Phase 16 — Graph-DAG rescue: refine the H107 antigen-receptor axis from
flat 4-bucket Hamming (trust 0.771, DIRECTIONAL) to a 2-level hierarchical
tree with shortest-path distance.

Hypothesis under test: the trust ceiling at ~0.77 on Hamming/ordinal rulers
arises because Hamming gives only {0, 1, 2} distinct distances and many
anchors land at distance-0 to each other (tied neighbours), which the
trustworthiness metric (k=15 neighbour ranks) struggles to preserve. The
H65 stage DAG (the only ruler family that clears 0.80 trust internally) has
34-stage branching tree with integer distances 0..9 — much richer.

Phase 16 builds an analogous tree over H107's 4 antigen-receptor buckets:

  Root  (virtual hema_progenitor)
   ├── tcr_ab (bucket node)
   │     ├── thymocyte (depth 0)
   │     ├── naive_T (depth 1)         [naive CD4 T, naive CD8 T]
   │     ├── mature_T (depth 2)        [CD4 T, CD8 T, T cell, mature αβ T]
   │     ├── Treg / NKT (depth 3)      [regulatory T, mature NKT, type i NKT]
   │     └── effector_T (depth 4)      [CD4 helper, cytotoxic effector, MAIT, γδ]
   ├── bcr (bucket node)
   │     ├── transitional_B (depth 0)
   │     ├── naive_B (depth 1)
   │     ├── memory_B (depth 2)
   │     ├── class_switched (depth 3)
   │     ├── plasmablast (depth 4)
   │     └── plasma (depth 5)
   ├── innate_lymphoid (bucket node)
   │     ├── ILC (depth 0)
   │     └── NK (depth 1)
   └── myeloid (bucket node)
         ├── progenitor (depth 0)      [HSC, HPC, CMP, EPC]
         ├── granulocyte (depth 1)     [neut, mast, baso, eosinophil]
         ├── DC (depth 2)              [DC, mDC, pDC]
         ├── monocyte (depth 2)        [classical, intermediate, non-classical]
         └── macrophage (depth 3)      [mac, microglia, kupffer, alveolar, ...]

Distance metric (shortest path on tree):
  - same sub-class: 0  (still tied — diagnostic)
  - same bucket, different sub-class: |sub_a - sub_b|
  - different buckets: sub_a + 1 (to bucket_a) + 1 (bucket_a → root) +
                       1 (root → bucket_b) + sub_b = sub_a + sub_b + 2

Distance range: 0 .. 11 (cross-bucket maxes around 5+5+2). Same anchor
mapping as H107 → roughly the same n_anchors_kept (≈290).

Same evaluator as Sweep #3 (anchor-level, 4-gate + 3-gate fallback +
DIRECTIONAL handling, paired global-shuffle null). If trust passes 0.80
under this richer ruler, that confirms the Hamming-ceiling hypothesis. If
trust is still ≤0.78, the ceiling is on the underlying biology, not the
ruler-resolution.

Outputs:
  reports/phase16_graph_dag_rescue.json
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.manifold import trustworthiness

PROJ = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN / "scripts"))
from phase5_let_anchor import (  # type: ignore
    train_let, build_pooled_drift, random_holdout_corr, grouped_holdout_corr, SEED,
)

ART = RUN / "artifacts"; REP = RUN / "reports"


def _ct(s): return s.strip().lower()


# (bucket, sub_class, sub_depth) per cell-type
H107_GRAPH = {
    # --- TCR-αβ bucket ---
    "thymocyte":                                                       ("tcr_ab", "thymocyte",      0),
    "double negative thymocyte":                                       ("tcr_ab", "thymocyte",      0),
    "double-positive, alpha-beta thymocyte":                           ("tcr_ab", "thymocyte",      0),
    "naive thymus-derived cd4-positive, alpha-beta t cell":            ("tcr_ab", "naive_T",        1),
    "naive thymus-derived cd8-positive, alpha-beta t cell":            ("tcr_ab", "naive_T",        1),
    "cd4-positive, alpha-beta t cell":                                 ("tcr_ab", "mature_T",       2),
    "cd8-positive, alpha-beta t cell":                                 ("tcr_ab", "mature_T",       2),
    "cd4-positive, alpha-beta memory t cell":                          ("tcr_ab", "mature_T",       2),
    "cd8-positive, alpha-beta memory t cell":                          ("tcr_ab", "mature_T",       2),
    "t cell":                                                          ("tcr_ab", "mature_T",       2),
    "mature alpha-beta t cell":                                        ("tcr_ab", "mature_T",       2),
    "regulatory t cell":                                               ("tcr_ab", "Treg_NKT",       3),
    "type i nk t cell":                                                ("tcr_ab", "Treg_NKT",       3),
    "mature nk t cell":                                                ("tcr_ab", "Treg_NKT",       3),
    "cd4-positive helper t cell":                                      ("tcr_ab", "effector_T",     4),
    "cd8-positive, alpha-beta cytokine secreting effector t cell":     ("tcr_ab", "effector_T",     4),
    "mucosal invariant t cell":                                        ("tcr_ab", "effector_T",     4),
    "gamma-delta t cell":                                              ("tcr_ab", "effector_T",     4),
    # --- BCR bucket ---
    "transitional stage b cell":                                       ("bcr", "transitional",      0),
    "naive b cell":                                                    ("bcr", "naive_B",           1),
    "b cell":                                                          ("bcr", "naive_B",           1),
    "memory b cell":                                                   ("bcr", "memory_B",          2),
    "class switched memory b cell":                                    ("bcr", "class_switched",    3),
    "plasmablast":                                                     ("bcr", "plasmablast",       4),
    "plasma cell":                                                     ("bcr", "plasma",            5),
    # --- innate-lymphoid bucket ---
    "innate lymphoid cell":                                            ("innate_lymphoid", "ILC",   0),
    "natural killer cell":                                             ("innate_lymphoid", "NK",    1),
    # --- myeloid + progenitor bucket ---
    "hematopoietic stem cell":                                         ("myeloid", "progenitor",    0),
    "hematopoietic precursor cell":                                    ("myeloid", "progenitor",    0),
    "common myeloid progenitor":                                       ("myeloid", "progenitor",    0),
    "common lymphoid progenitor":                                      ("myeloid", "progenitor",    0),
    "granulocyte monocyte progenitor cell":                            ("myeloid", "progenitor",    0),
    "erythroid progenitor cell":                                       ("myeloid", "progenitor",    0),
    "erythrocyte":                                                     ("myeloid", "progenitor",    0),
    "platelet":                                                        ("myeloid", "progenitor",    0),
    "neutrophil":                                                      ("myeloid", "granulocyte",   1),
    "mature neutrophil":                                               ("myeloid", "granulocyte",   1),
    "mast cell":                                                       ("myeloid", "granulocyte",   1),
    "basophil":                                                        ("myeloid", "granulocyte",   1),
    "eosinophil":                                                      ("myeloid", "granulocyte",   1),
    "myeloid dendritic cell":                                          ("myeloid", "DC",            2),
    "conventional dendritic cell":                                     ("myeloid", "DC",            2),
    "dendritic cell":                                                  ("myeloid", "DC",            2),
    "cd1c-positive myeloid dendritic cell":                            ("myeloid", "DC",            2),
    "cd141-positive myeloid dendritic cell":                           ("myeloid", "DC",            2),
    "plasmacytoid dendritic cell":                                     ("myeloid", "DC",            2),
    "monocyte":                                                        ("myeloid", "monocyte",      2),
    "classical monocyte":                                              ("myeloid", "monocyte",      2),
    "intermediate monocyte":                                           ("myeloid", "monocyte",      2),
    "non-classical monocyte":                                          ("myeloid", "monocyte",      2),
    "macrophage":                                                      ("myeloid", "macrophage",    3),
    "alveolar macrophage":                                             ("myeloid", "macrophage",    3),
    "tissue-resident macrophage":                                      ("myeloid", "macrophage",    3),
    "microglial cell":                                                 ("myeloid", "macrophage",    3),
    "kupffer cell":                                                    ("myeloid", "macrophage",    3),
    "elicited macrophage":                                             ("myeloid", "macrophage",    3),
}


def graph_distance(a: tuple, b: tuple) -> float:
    """Shortest path on the antigen-receptor hierarchical tree.
    a, b are (bucket, sub_class, sub_depth) tuples.
    """
    bucket_a, _sub_a, depth_a = a
    bucket_b, _sub_b, depth_b = b
    if bucket_a == bucket_b:
        return float(abs(depth_a - depth_b))
    # cross-bucket: depth_a → bucket_a (depth_a) → root (1) → bucket_b (1) → depth_b (depth_b)
    return float(depth_a + 1 + 1 + depth_b)


def build_graph_ruler(cell_types: list[str]):
    nodes = []
    keep_mask = []
    for ct in cell_types:
        spec = H107_GRAPH.get(_ct(ct))
        nodes.append(spec)
        keep_mask.append(spec is not None)
    keep_mask = np.array(keep_mask)
    n = len(cell_types)
    D_full = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, n):
            if not keep_mask[j]:
                continue
            d = graph_distance(nodes[i], nodes[j])
            D_full[i, j] = d
            D_full[j, i] = d
    return D_full, nodes, keep_mask


def evaluate(feat, anchor_meta, gates):
    cell_types = anchor_meta["cell_type"].astype(str).tolist()
    D_full, nodes, keep = build_graph_ruler(cell_types)
    if keep.sum() < 30:
        return {"verdict": "INSUFFICIENT_COVERAGE", "n_anchors_kept": int(keep.sum())}
    idx = np.where(keep)[0]
    feat_k = feat[idx]
    D = D_full[np.ix_(idx, idx)]
    nodes_k = [nodes[i] for i in idx]
    anchor_meta_k = anchor_meta.loc[keep].reset_index(drop=True)

    print(f"  n_anchors_kept = {keep.sum()}/{len(cell_types)}")
    print(f"  D unique values: {sorted(np.unique(D).tolist())[:20]}")
    print(f"  D max = {D.max():.1f}, mean (off-diag) = {D[D>0].mean():.2f}")

    # Per-bucket and per-subclass primary groups
    primary_bucket = np.array([nd[0] for nd in nodes_k])
    primary_subclass = np.array([f"{nd[0]}|{nd[1]}" for nd in nodes_k])

    print("[1] Train head, gates...")
    head, z, _ = train_let(feat_k, D, label="H107_graph", verbose=False)
    trust = float(trustworthiness(feat_k, z, n_neighbors=15))
    rand_corr, _ = random_holdout_corr(feat_k, D, n_iters=5, frac=0.2)
    donor_corr, _ = grouped_holdout_corr(
        feat_k, D, anchor_meta_k["donor_id"].astype(str).to_numpy(), "H107_graph-donor"
    )
    bucket_corr, _ = grouped_holdout_corr(feat_k, D, primary_bucket, "H107_graph-bucket")
    subclass_corr, _ = grouped_holdout_corr(feat_k, D, primary_subclass, "H107_graph-subclass")

    print("[2] Null (global label shuffle)...")
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(idx))
    nodes_shuf = [nodes_k[p] for p in perm]
    n_k = len(idx)
    D_null = np.zeros((n_k, n_k), dtype=np.float32)
    for i in range(n_k):
        for j in range(i + 1, n_k):
            d = graph_distance(nodes_shuf[i], nodes_shuf[j])
            D_null[i, j] = d
            D_null[j, i] = d
    head_n, z_n, _ = train_let(feat_k, D_null, label="H107_graph_null", verbose=False)
    trust_n = float(trustworthiness(feat_k, z_n, n_neighbors=15))
    rand_n, _ = random_holdout_corr(feat_k, D_null, n_iters=5, frac=0.2)
    donor_n, _ = grouped_holdout_corr(
        feat_k, D_null, anchor_meta_k["donor_id"].astype(str).to_numpy(), "H107_graph-donor-null"
    )
    bucket_n, _ = grouped_holdout_corr(
        feat_k, D_null, np.array([nd[0] for nd in nodes_shuf]), "H107_graph-bucket-null"
    )

    def _ge(x, t):
        if x is None:
            return False
        return (not (isinstance(x, float) and math.isnan(x))) and x >= t

    pos4 = _ge(trust, gates["trustworthiness_min"]) and _ge(rand_corr, gates["random_holdout_correlation_min"]) and _ge(donor_corr, gates["donor_holdout_correlation_min"]) and _ge(bucket_corr, gates["clade_branch_holdout_correlation_min"])
    null4 = _ge(trust_n, gates["trustworthiness_min"]) and _ge(rand_n, gates["random_holdout_correlation_min"]) and _ge(donor_n, gates["donor_holdout_correlation_min"]) and _ge(bucket_n, gates["clade_branch_holdout_correlation_min"])
    pos3 = _ge(trust, gates["trustworthiness_min"]) and _ge(rand_corr, gates["random_holdout_correlation_min"]) and _ge(donor_corr, gates["donor_holdout_correlation_min"])
    null3 = _ge(trust_n, gates["trustworthiness_min"]) and _ge(rand_n, gates["random_holdout_correlation_min"]) and _ge(donor_n, gates["donor_holdout_correlation_min"])
    trust_subthreshold = not _ge(trust, gates["trustworthiness_min"])
    rand_dir = _ge(rand_corr, 0.5 + gates["random_holdout_correlation_min"])
    donor_dir = _ge(donor_corr, 0.5 + gates["donor_holdout_correlation_min"])

    if pos4 and not null4: verdict = "POSITIVE"
    elif pos3 and not null3 and not pos4: verdict = "POSITIVE_3GATE_FALLBACK"
    elif trust_subthreshold and (rand_dir or donor_dir): verdict = "DIRECTIONAL_TRUST_SUBTHRESHOLD"
    else: verdict = "INCONCLUSIVE"

    return {
        "label": "H107_graph_dag_rescue",
        "n_anchors_kept": int(keep.sum()),
        "ruler_max": float(D.max()),
        "ruler_unique_distances": sorted(set(np.unique(D).tolist())),
        "positive_trust": trust,
        "positive_random": float(rand_corr),
        "positive_donor": float(donor_corr) if not (isinstance(donor_corr, float) and math.isnan(donor_corr)) else None,
        "positive_bucket_holdout": float(bucket_corr) if not (isinstance(bucket_corr, float) and math.isnan(bucket_corr)) else None,
        "positive_subclass_holdout": float(subclass_corr) if not (isinstance(subclass_corr, float) and math.isnan(subclass_corr)) else None,
        "null_trust": trust_n,
        "null_random": float(rand_n),
        "null_donor": float(donor_n) if not (isinstance(donor_n, float) and math.isnan(donor_n)) else None,
        "null_bucket": float(bucket_n) if not (isinstance(bucket_n, float) and math.isnan(bucket_n)) else None,
        "verdict": verdict,
    }


def main():
    print("=" * 70)
    print("PHASE 16 — Graph-DAG rescue for H107 antigen-receptor axis")
    print("=" * 70)
    t_phase = time.time()

    centroids = np.load(ART / "anchors/centroids_internal.npy")
    anchor_meta = pd.read_csv(ART / "anchors/anchor_meta_internal.csv")
    op_idx = json.loads((ART / "operators/operator_index.json").read_text())
    op_npz = np.load(ART / "operators/pooled_drift_components.npz")
    A_e, A_m, A_l = op_npz["A_early"], op_npz["A_mid"], op_npz["A_late"]
    partition = op_idx["block_partition"]
    feat = build_pooled_drift(centroids, A_e, A_m, A_l, partition)
    feat = (feat - feat.mean(0)) / (feat.std(0) + 1e-6)
    gates = json.loads((REP / "quality_gates_spec.json").read_text())["gates"]

    r = evaluate(feat, anchor_meta, gates)
    r["elapsed_s"] = float(time.time() - t_phase)

    print("\n" + "=" * 70)
    print(f"VERDICT: {r['verdict']}")
    print(f"  POS  trust={r['positive_trust']:.3f}  rand={r['positive_random']:.3f}  donor={r['positive_donor']}  bucket={r.get('positive_bucket_holdout')}  subclass={r.get('positive_subclass_holdout')}")
    print(f"  NULL trust={r['null_trust']:.3f}  rand={r['null_random']:.3f}  donor={r.get('null_donor')}  bucket={r.get('null_bucket')}")
    print(f"  Compared to flat H107 Hamming: trust 0.771 (DIRECTIONAL).")
    print(f"  Total elapsed: {r['elapsed_s']:.1f}s")

    (REP / "phase16_graph_dag_rescue.json").write_text(json.dumps(r, indent=2))
    print(f"  → {REP / 'phase16_graph_dag_rescue.json'}")


if __name__ == "__main__":
    main()
