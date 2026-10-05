"""Phase 3a sweep #2 — ordinal-distance rulers (maturation chains).

Sweep #1 used Hamming distance over multi-label category sets. Sweep #2 uses
*ordinal* depth assignments (single numeric axis per anchor) and L1 distance —
this is structurally similar to the H65 graph-distance ruler.

Each candidate assigns each anchor a numeric depth in a maturation chain.
Distance = |depth_i - depth_j|. Null = shuffled depths.

Outputs:
  reports/hypothesis_registry_sweep2.json
  reports/hypothesis_registry_sweep2.csv
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
    LETHead, train_let, build_pooled_drift, random_holdout_corr, grouped_holdout_corr, SEED,
)

ART = RUN / "artifacts"; REP = RUN / "reports"


def _ct(s): return s.strip().lower()


# H101 — continuous progenitor → terminal depth (whole-hematopoiesis maturation)
# 0 = HSC/pluripotent, 6 = terminal effector
H101_DEPTH = {
    "hematopoietic stem cell": 0, "hematopoietic precursor cell": 0,
    "common myeloid progenitor": 1, "common lymphoid progenitor": 1,
    "granulocyte monocyte progenitor cell": 2, "erythroid progenitor cell": 2,
    "transitional stage b cell": 2, "double-positive, alpha-beta thymocyte": 2,
    "double negative thymocyte": 2, "thymocyte": 2,
    "monocyte": 3, "intermediate monocyte": 3, "naive b cell": 3,
    "naive thymus-derived cd4-positive, alpha-beta t cell": 3,
    "naive thymus-derived cd8-positive, alpha-beta t cell": 3,
    "classical monocyte": 4, "non-classical monocyte": 4,
    "memory b cell": 4, "class switched memory b cell": 4,
    "cd4-positive, alpha-beta t cell": 4, "cd4-positive, alpha-beta memory t cell": 4,
    "cd8-positive, alpha-beta t cell": 4, "cd8-positive, alpha-beta memory t cell": 4,
    "regulatory t cell": 4, "t cell": 4, "mature alpha-beta t cell": 4,
    "innate lymphoid cell": 4, "natural killer cell": 4, "type i nk t cell": 4,
    "mucosal invariant t cell": 4, "gamma-delta t cell": 4, "b cell": 4,
    "myeloid dendritic cell": 4, "conventional dendritic cell": 4, "dendritic cell": 4,
    "cd1c-positive myeloid dendritic cell": 4, "cd141-positive myeloid dendritic cell": 4,
    "plasmacytoid dendritic cell": 4, "basophil": 4, "eosinophil": 4, "mast cell": 4,
    "macrophage": 5, "alveolar macrophage": 5, "tissue-resident macrophage": 5,
    "kupffer cell": 5, "microglial cell": 5, "elicited macrophage": 5,
    "neutrophil": 5, "mature neutrophil": 5,
    "cd4-positive helper t cell": 5,
    "cd8-positive, alpha-beta cytokine secreting effector t cell": 5,
    "platelet": 5, "megakaryocyte": 5, "erythroid lineage cell": 5,
    "plasma cell": 6, "plasmablast": 5,
    "erythrocyte": 6,
}

# H102 — T-cell-only maturation depth
H102_DEPTH = {
    "double negative thymocyte": 0,
    "double-positive, alpha-beta thymocyte": 1,
    "thymocyte": 1,
    "naive thymus-derived cd4-positive, alpha-beta t cell": 2,
    "naive thymus-derived cd8-positive, alpha-beta t cell": 2,
    "cd4-positive, alpha-beta t cell": 3,
    "cd8-positive, alpha-beta t cell": 3,
    "t cell": 3, "mature alpha-beta t cell": 3,
    "cd4-positive, alpha-beta memory t cell": 4,
    "cd8-positive, alpha-beta memory t cell": 4,
    "regulatory t cell": 4,
    "gamma-delta t cell": 4,
    "mucosal invariant t cell": 4,
    "cd4-positive helper t cell": 5,
    "cd8-positive, alpha-beta cytokine secreting effector t cell": 5,
}

# H103 — B-cell-only maturation depth
H103_DEPTH = {
    "transitional stage b cell": 0,
    "naive b cell": 1, "b cell": 1,
    "memory b cell": 2,
    "class switched memory b cell": 3,
    "plasmablast": 4,
    "plasma cell": 5,
}

# H104 — monocyte → macrophage maturation
H104_DEPTH = {
    "classical monocyte": 0,
    "monocyte": 0,
    "intermediate monocyte": 1,
    "non-classical monocyte": 2,
    "macrophage": 3, "tissue-resident macrophage": 3, "alveolar macrophage": 3,
    "kupffer cell": 3, "microglial cell": 3, "elicited macrophage": 3,
}

# H105 — granulocyte vs (mono + DC) myeloid axis (within myeloid)
H105_DEPTH = {
    # progenitors
    "common myeloid progenitor": 0, "granulocyte monocyte progenitor cell": 1,
    # mono branch
    "monocyte": 2, "classical monocyte": 2, "intermediate monocyte": 2, "non-classical monocyte": 2,
    "macrophage": 3, "tissue-resident macrophage": 3, "alveolar macrophage": 3,
    "kupffer cell": 3, "microglial cell": 3, "elicited macrophage": 3,
    # dendritic
    "myeloid dendritic cell": 4, "conventional dendritic cell": 4, "dendritic cell": 4,
    "cd1c-positive myeloid dendritic cell": 4, "cd141-positive myeloid dendritic cell": 4,
    "plasmacytoid dendritic cell": 4,
    # granulocyte
    "neutrophil": 5, "mature neutrophil": 5,
    "basophil": 5, "eosinophil": 5, "mast cell": 5,
}

# H106 — proliferation-state proxy (naive < memory < effector < cycling)
H106_DEPTH = {
    "hematopoietic stem cell": 3, "hematopoietic precursor cell": 3,
    "common myeloid progenitor": 4, "common lymphoid progenitor": 4,
    "granulocyte monocyte progenitor cell": 4, "erythroid progenitor cell": 4,
    "transitional stage b cell": 3, "double-positive, alpha-beta thymocyte": 4,
    "double negative thymocyte": 3, "thymocyte": 3,
    "naive thymus-derived cd4-positive, alpha-beta t cell": 0,
    "naive thymus-derived cd8-positive, alpha-beta t cell": 0,
    "naive b cell": 0, "b cell": 0,
    "monocyte": 1, "classical monocyte": 1, "intermediate monocyte": 1, "non-classical monocyte": 1,
    "cd4-positive, alpha-beta t cell": 1, "cd4-positive, alpha-beta memory t cell": 1,
    "cd8-positive, alpha-beta t cell": 1, "cd8-positive, alpha-beta memory t cell": 1,
    "memory b cell": 1, "class switched memory b cell": 1,
    "regulatory t cell": 1,
    "innate lymphoid cell": 1, "natural killer cell": 2, "type i nk t cell": 2,
    "mucosal invariant t cell": 1, "gamma-delta t cell": 1, "t cell": 1, "mature alpha-beta t cell": 1,
    "myeloid dendritic cell": 1, "conventional dendritic cell": 1, "dendritic cell": 1,
    "cd1c-positive myeloid dendritic cell": 1, "cd141-positive myeloid dendritic cell": 1,
    "plasmacytoid dendritic cell": 1,
    "macrophage": 2, "tissue-resident macrophage": 2, "alveolar macrophage": 2,
    "kupffer cell": 2, "microglial cell": 2, "elicited macrophage": 2,
    "neutrophil": 2, "mature neutrophil": 2, "basophil": 2, "eosinophil": 2, "mast cell": 2,
    "cd4-positive helper t cell": 2,
    "cd8-positive, alpha-beta cytokine secreting effector t cell": 2,
    "plasmablast": 4, "plasma cell": 0,
    "erythrocyte": 5, "erythroid lineage cell": 4, "platelet": 5, "megakaryocyte": 4,
}


CANDIDATES = [
    {"name": "H101_progenitor_to_terminal", "depth_map": H101_DEPTH,
     "description": "Whole-hematopoiesis maturation depth (HSC=0 → terminal=6)"},
    {"name": "H102_T_cell_maturation", "depth_map": H102_DEPTH,
     "description": "T-cell-only maturation: DN→DP→naive→memory→effector"},
    {"name": "H103_B_cell_maturation", "depth_map": H103_DEPTH,
     "description": "B-cell-only maturation: transitional→naive→memory→class-switched→plasmablast→plasma"},
    {"name": "H104_mono_macro_axis", "depth_map": H104_DEPTH,
     "description": "Classical mono → intermediate → non-classical → macrophage"},
    {"name": "H105_myeloid_branches", "depth_map": H105_DEPTH,
     "description": "Myeloid sub-branches as ordinal: prog→mono/macro→DC→granulocyte"},
    {"name": "H106_proliferation_proxy", "depth_map": H106_DEPTH,
     "description": "Cell-state proliferation proxy: naive(0)→activated(1)→effector(2)→cycling/progenitor(3-4)→post-mitotic(5)"},
]


def build_ordinal_ruler(cell_types: list[str], depth_map: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """L1 distance on depth assignments. Returns (D, depths)."""
    n = len(cell_types)
    depths = np.array([depth_map.get(_ct(ct), -1) for ct in cell_types], dtype=np.float32)
    keep = depths >= 0
    D_full = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        D_full[i] = np.abs(depths - depths[i])
    return D_full, depths


def evaluate(cand, feat, anchor_meta, gates):
    name = cand["name"]
    cell_types = anchor_meta["cell_type"].astype(str).tolist()
    D_full, depths = build_ordinal_ruler(cell_types, cand["depth_map"])
    keep = depths >= 0
    if keep.sum() < 30:
        return {"name": name, "verdict": "INSUFFICIENT_COVERAGE", "n_anchors_kept": int(keep.sum())}
    feat_k = feat[keep]
    D = D_full[np.ix_(np.where(keep)[0], np.where(keep)[0])]
    depths_k = depths[keep]
    anchor_meta_k = anchor_meta.loc[keep].reset_index(drop=True)
    if D.max() == 0:
        return {"name": name, "verdict": "DEGENERATE_RULER", "n_anchors_kept": int(keep.sum())}

    head, z, _ = train_let(feat_k, D, label=name, verbose=False)
    trust = trustworthiness(feat_k, z, n_neighbors=15)
    rand_corr, _ = random_holdout_corr(feat_k, D, n_iters=5, frac=0.2)
    donor_corr, _ = grouped_holdout_corr(
        feat_k, D, anchor_meta_k["donor_id"].astype(str).to_numpy(), f"{name}-donor"
    )
    # For ordinal candidates, "branch" holdout uses the depth value as a discrete group
    depth_group = depths_k.astype(int).astype(str)
    depth_corr, _ = grouped_holdout_corr(feat_k, D, depth_group, f"{name}-depth")

    # Null: shuffle depths
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(depths_k))
    depths_shuffled = depths_k[perm]
    D_null = np.zeros_like(D)
    for i in range(len(depths_k)):
        D_null[i] = np.abs(depths_shuffled - depths_shuffled[i])
    head_n, z_n, _ = train_let(feat_k, D_null, label=f"{name}_null", verbose=False)
    trust_n = trustworthiness(feat_k, z_n, n_neighbors=15)
    rand_n, _ = random_holdout_corr(feat_k, D_null, n_iters=5, frac=0.2)
    donor_n, _ = grouped_holdout_corr(feat_k, D_null, anchor_meta_k["donor_id"].astype(str).to_numpy(), f"{name}-donor-null")
    depth_n, _ = grouped_holdout_corr(feat_k, D_null, depths_shuffled.astype(int).astype(str), f"{name}-depth-null")

    pos_passes = (
        trust >= gates["trustworthiness_min"]
        and (not (isinstance(rand_corr, float) and math.isnan(rand_corr))) and rand_corr >= gates["random_holdout_correlation_min"]
        and (not (isinstance(donor_corr, float) and math.isnan(donor_corr))) and donor_corr >= gates["donor_holdout_correlation_min"]
        and (not (isinstance(depth_corr, float) and math.isnan(depth_corr))) and depth_corr >= gates["clade_branch_holdout_correlation_min"]
    )
    null_passes = (
        trust_n >= gates["trustworthiness_min"]
        and (not (isinstance(rand_n, float) and math.isnan(rand_n))) and rand_n >= gates["random_holdout_correlation_min"]
        and (not (isinstance(donor_n, float) and math.isnan(donor_n))) and donor_n >= gates["donor_holdout_correlation_min"]
        and (not (isinstance(depth_n, float) and math.isnan(depth_n))) and depth_n >= gates["clade_branch_holdout_correlation_min"]
    )
    if pos_passes and not null_passes:
        verdict = "POSITIVE"
    elif (trust >= gates["trustworthiness_min"]
          and (not (isinstance(rand_corr, float) and math.isnan(rand_corr))) and rand_corr >= gates["random_holdout_correlation_min"]
          and (not (isinstance(donor_corr, float) and math.isnan(donor_corr))) and donor_corr >= gates["donor_holdout_correlation_min"]
          and not pos_passes):
        verdict = "POSITIVE_3GATE_FALLBACK"
    elif not pos_passes:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "PIPELINE_ARTEFACT"

    return {
        "name": name,
        "description": cand["description"],
        "n_anchors_kept": int(keep.sum()),
        "n_distinct_depths": int(len(np.unique(depths_k))),
        "ruler_max": float(D.max()),
        "positive_trust": float(trust),
        "positive_random": float(rand_corr) if not (isinstance(rand_corr, float) and math.isnan(rand_corr)) else None,
        "positive_donor": float(donor_corr) if not (isinstance(donor_corr, float) and math.isnan(donor_corr)) else None,
        "positive_depth": float(depth_corr) if not (isinstance(depth_corr, float) and math.isnan(depth_corr)) else None,
        "null_trust": float(trust_n),
        "null_random": float(rand_n) if not (isinstance(rand_n, float) and math.isnan(rand_n)) else None,
        "null_donor": float(donor_n) if not (isinstance(donor_n, float) and math.isnan(donor_n)) else None,
        "null_depth": float(depth_n) if not (isinstance(depth_n, float) and math.isnan(depth_n)) else None,
        "verdict": verdict,
    }


def main():
    print("=" * 70)
    print("PHASE 3a SWEEP #2 — ordinal-distance rulers (maturation chains)")
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

    results = []
    for cand in CANDIDATES:
        print(f"\n--- {cand['name']}: {cand['description']} ---")
        t0 = time.time()
        r = evaluate(cand, feat, anchor_meta, gates)
        r["elapsed_s"] = float(time.time() - t0)
        if "positive_trust" in r:
            print(f"  POS trust={r['positive_trust']:.3f} rand={r.get('positive_random')} donor={r.get('positive_donor')} depth={r.get('positive_depth')}")
            print(f"  NULL trust={r['null_trust']:.3f} rand={r.get('null_random')} donor={r.get('null_donor')} depth={r.get('null_depth')}")
        print(f"  VERDICT: {r['verdict']}  ({r['elapsed_s']:.1f}s)")
        results.append(r)

    out = {
        "scope_note": "Phase 3a Sweep #2 — ordinal depth rulers, L1 distance. Anchor-level on H65 internal panel.",
        "n_candidates": len(results),
        "candidates": results,
        "elapsed_s": float(time.time() - t_phase),
    }
    (REP / "hypothesis_registry_sweep2.json").write_text(json.dumps(out, indent=2))
    df = pd.DataFrame([{k: v for k, v in r.items()} for r in results])
    df.to_csv(REP / "hypothesis_registry_sweep2.csv", index=False)
    print("\n=== Sweep #2 SUMMARY ===")
    for r in results:
        if "positive_trust" in r:
            print(f"{r['name']:35s} {r['verdict']:30s} trust={r['positive_trust']:.3f}/{r['null_trust']:.3f}")
        else:
            print(f"{r['name']:35s} {r['verdict']:30s}")
    print(f"\nTotal time: {time.time()-t_phase:.1f}s")


if __name__ == "__main__":
    main()
