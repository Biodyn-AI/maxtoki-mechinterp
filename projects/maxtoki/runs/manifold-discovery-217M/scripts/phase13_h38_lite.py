"""Phase 13 LITE — H38 second-manifold with a curated signaling-category ruler.

Source paper's H38 case uses OmniPath ligand-receptor interaction-path distances on
30 hand-picked LR-pair anchors. We don't have OmniPath wired up here, so we test the
same conceptual question with a simpler proxy:

  Question: does MaxToki-217M's residual carry an *intercellular communication*
            geometry alongside the H65 developmental geometry?

  Proxy ruler: each H65 internal anchor's cell_type is mapped to a coarse signalling
               category (cytokine producer / chemokine producer / lipid-metabolic /
               antigen-presenting / cytotoxic / antibody-producer / progenitor).
               H38 ruler distance = Hamming distance on the (cell-type → categories)
               assignment.

This reuses the existing internal-panel centroids and operator library. We train a
fresh LET head against the H38 ruler and check whether it also passes the four
quality gates (with a paired null).

Outputs:
  reports/h38_lite_quality_gates.json
  reports/h38_lite_categories.csv  (cell_type → categories assignment)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.manifold import trustworthiness
from scipy.stats import spearmanr

PROJ = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN / "scripts"))
from phase5_let_anchor import (  # type: ignore
    LETHead, train_let, build_pooled_drift, random_holdout_corr, grouped_holdout_corr, SEED,
)

ART = RUN / "artifacts"; REP = RUN / "reports"

# ----- Curated cell-type → signalling-category assignment.
# Categories (multi-label):
#   cyto = cytokine producer (IL-2/IL-4/IFN-γ/etc.)
#   chemo = chemokine producer (CXCL/CCL family)
#   lipid = lipid-metabolic / phagocytic
#   antigen = antigen-presenting (MHC-II)
#   cytotox = cytotoxic effector (perforin/granzyme/granulocyte degranulation)
#   ab = antibody producer
#   prog = progenitor / not yet committed to signalling role
CELL_TYPE_TO_CATEGORIES = {
    # T-lineage
    "naive thymus-derived cd4-positive, alpha-beta t cell": {"prog"},
    "cd4-positive, alpha-beta t cell": {"cyto"},
    "cd4-positive, alpha-beta memory t cell": {"cyto", "antigen"},
    "cd4-positive helper t cell": {"cyto"},
    "regulatory t cell": {"cyto"},
    "naive thymus-derived cd8-positive, alpha-beta t cell": {"prog"},
    "cd8-positive, alpha-beta t cell": {"cytotox"},
    "cd8-positive, alpha-beta memory t cell": {"cytotox"},
    "cd8-positive, alpha-beta cytokine secreting effector t cell": {"cyto", "cytotox"},
    "double-positive, alpha-beta thymocyte": {"prog"},
    "double negative thymocyte": {"prog"},
    "mucosal invariant t cell": {"cyto", "cytotox"},
    "gamma-delta t cell": {"cytotox"},
    "t cell": {"cyto"},
    "thymocyte": {"prog"},
    "mature alpha-beta t cell": {"cyto", "cytotox"},
    "innate lymphoid cell": {"cyto"},
    "natural killer cell": {"cytotox", "cyto"},
    "type i nk t cell": {"cytotox", "cyto"},
    # B-lineage
    "b cell": {"antigen"},
    "naive b cell": {"antigen"},
    "memory b cell": {"antigen"},
    "class switched memory b cell": {"antigen"},
    "transitional stage b cell": {"prog"},
    "plasma cell": {"ab"},
    "plasmablast": {"ab"},
    # Dendritic
    "plasmacytoid dendritic cell": {"cyto", "antigen"},
    "conventional dendritic cell": {"antigen"},
    "dendritic cell": {"antigen"},
    "myeloid dendritic cell": {"antigen"},
    "cd1c-positive myeloid dendritic cell": {"antigen"},
    "cd141-positive myeloid dendritic cell": {"antigen"},
    # Mono/macro
    "monocyte": {"chemo", "lipid"},
    "classical monocyte": {"chemo", "lipid"},
    "non-classical monocyte": {"chemo"},
    "intermediate monocyte": {"chemo", "lipid"},
    "macrophage": {"lipid", "antigen", "chemo"},
    "alveolar macrophage": {"lipid", "antigen"},
    "tissue-resident macrophage": {"lipid", "antigen"},
    "kupffer cell": {"lipid", "antigen"},
    "microglial cell": {"lipid"},
    "elicited macrophage": {"lipid", "chemo"},
    # Granulocyte
    "neutrophil": {"chemo", "cytotox"},
    "mature neutrophil": {"chemo", "cytotox"},
    "basophil": {"cyto"},
    "eosinophil": {"cytotox"},
    "mast cell": {"cyto"},
    # Erythroid / megakaryocyte
    "erythrocyte": set(),
    "erythroid lineage cell": {"prog"},
    "erythroid progenitor cell": {"prog"},
    "platelet": {"chemo"},
    "megakaryocyte": {"prog"},
    # Stem / progenitor
    "hematopoietic stem cell": {"prog"},
    "hematopoietic precursor cell": {"prog"},
    "common myeloid progenitor": {"prog"},
    "common lymphoid progenitor": {"prog"},
    "granulocyte monocyte progenitor cell": {"prog"},
}

ALL_CATEGORIES = ["cyto", "chemo", "lipid", "antigen", "cytotox", "ab", "prog"]


def categorize(cell_type_str: str) -> set[str]:
    return CELL_TYPE_TO_CATEGORIES.get(cell_type_str.strip().lower(), set())


def build_h38_ruler(cell_types: list[str]) -> np.ndarray:
    """Hamming distance on category-membership indicator vectors (n × n_categories)."""
    n = len(cell_types)
    cats = [categorize(c) for c in cell_types]
    indicator = np.zeros((n, len(ALL_CATEGORIES)), dtype=np.float32)
    for i, cs in enumerate(cats):
        for ci, c in enumerate(ALL_CATEGORIES):
            if c in cs:
                indicator[i, ci] = 1.0
    # Pairwise Hamming
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        D[i] = (indicator != indicator[i]).sum(axis=1)
    return D


def main():
    print("=" * 70)
    print("PHASE 13 LITE — H38 second-manifold with curated signalling ruler")
    print("=" * 70)
    t_phase = time.time()

    centroids = np.load(ART / "anchors/centroids_internal.npy")
    anchor_meta = pd.read_csv(ART / "anchors/anchor_meta_internal.csv")
    op_idx = json.loads((ART / "operators/operator_index.json").read_text())
    op_npz = np.load(ART / "operators/pooled_drift_components.npz")
    A_e, A_m, A_l = op_npz["A_early"], op_npz["A_mid"], op_npz["A_late"]
    partition = op_idx["block_partition"]

    # Build pooled-drift feature
    feat = build_pooled_drift(centroids, A_e, A_m, A_l, partition)
    feat = (feat - feat.mean(0)) / (feat.std(0) + 1e-6)

    cell_types = anchor_meta["cell_type"].astype(str).tolist()
    d_target = build_h38_ruler(cell_types)
    print(f"  d_target shape={d_target.shape}  min={d_target.min()} median={np.median(d_target):.1f} max={d_target.max()}")

    # Save categories CSV
    rows_c = []
    for ct in sorted(set(cell_types)):
        rows_c.append({"cell_type": ct, "categories": ",".join(sorted(categorize(ct)))})
    pd.DataFrame(rows_c).to_csv(REP / "h38_lite_categories.csv", index=False)

    # Coverage check
    n_uncategorized = sum(1 for ct in cell_types if not categorize(ct))
    print(f"  cell_types with no category: {n_uncategorized}/{len(cell_types)}")

    # Filter out anchors with no category (zero indicator vector → fits everything trivially)
    keep = np.array([bool(categorize(ct)) for ct in cell_types])
    feat_k = feat[keep]; D_k = d_target[np.ix_(np.where(keep)[0], np.where(keep)[0])]
    cell_types_k = [ct for ct, k in zip(cell_types, keep) if k]
    anchor_meta_k = anchor_meta.loc[keep].reset_index(drop=True)
    print(f"  kept {keep.sum()}/{len(cell_types)} anchors with at least one category")

    # ---- Train H38 LET head
    print("\n[1] Training H38 anchor LET head...")
    head, z, _ = train_let(feat_k, D_k, label="H38_pos", verbose=False)

    print("\n[2] Trustworthiness, random/donor/branch holdouts...")
    trust = trustworthiness(feat_k, z, n_neighbors=15)
    rand_corr, _ = random_holdout_corr(feat_k, D_k, n_iters=10, frac=0.2)
    donor_corr, _ = grouped_holdout_corr(feat_k, D_k, anchor_meta_k["donor_id"].astype(str).to_numpy(), "h38-donor")
    # Branch holdout: define "branch" as the most common category for each anchor, fall back to "_unk"
    primary_cat = []
    for ct in cell_types_k:
        cats = sorted(categorize(ct))
        primary_cat.append(cats[0] if cats else "_unk")
    branch_corr, _ = grouped_holdout_corr(feat_k, D_k, np.array(primary_cat), "h38-cat")
    print(f"  H38   trust={trust:.3f}  rand={rand_corr:.3f}  donor={donor_corr:.3f}  category-holdout={branch_corr:.3f}")

    # ---- Null: shuffle category-vector across anchors (preserves marginal but breaks per-cell-type signal)
    print("\n[3] Null: shuffled categories...")
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(cell_types_k))
    cell_types_shuffled = [cell_types_k[p] for p in perm]
    D_null = build_h38_ruler(cell_types_shuffled)
    head_n, z_n, _ = train_let(feat_k, D_null, label="H38_null", verbose=False)
    trust_n = trustworthiness(feat_k, z_n, n_neighbors=15)
    rand_n, _ = random_holdout_corr(feat_k, D_null, n_iters=10, frac=0.2)
    donor_n, _ = grouped_holdout_corr(feat_k, D_null, anchor_meta_k["donor_id"].astype(str).to_numpy(), "h38-donor-null")
    primary_cat_shuffled = [primary_cat[p] for p in perm]
    branch_n, _ = grouped_holdout_corr(feat_k, D_null, np.array(primary_cat_shuffled), "h38-cat-null")
    print(f"  null  trust={trust_n:.3f}  rand={rand_n:.3f}  donor={donor_n:.3f}  category-holdout={branch_n:.3f}")

    # ---- Verdict
    gates = json.loads((REP / "quality_gates_spec.json").read_text())["gates"]
    pos_passes = (
        trust >= gates["trustworthiness_min"]
        and rand_corr >= gates["random_holdout_correlation_min"]
        and donor_corr >= gates["donor_holdout_correlation_min"]
        and branch_corr >= gates["clade_branch_holdout_correlation_min"]
    )
    null_passes = (
        trust_n >= gates["trustworthiness_min"]
        and rand_n >= gates["random_holdout_correlation_min"]
        and donor_n >= gates["donor_holdout_correlation_min"]
        and branch_n >= gates["clade_branch_holdout_correlation_min"]
    )
    if pos_passes and not null_passes:
        verdict = "POSITIVE: H38 LITE passes all gates and shuffled null fails"
    elif not pos_passes:
        verdict = "INCONCLUSIVE: H38 LITE does not pass all gates"
    else:
        verdict = "PIPELINE_ARTEFACT: null also passes — branch-specificity not demonstrated"

    out = {
        "scope_note": (
            "PHASE 13 LITE — second-manifold proxy. NOT source paper's H38 with OmniPath "
            "LR-pair anchors and 100k cells. Uses existing internal-panel anchors (n=keep) "
            "with a curated cell-type → signalling-category Hamming-distance ruler."
        ),
        "n_anchors_kept": int(keep.sum()),
        "n_anchors_dropped_no_category": int((~keep).sum()),
        "n_categories": len(ALL_CATEGORIES),
        "categories": ALL_CATEGORIES,
        "ruler_range": [float(D_k.min()), float(D_k.median()) if hasattr(D_k, 'median') else float(np.median(D_k)), float(D_k.max())],
        "positive": {
            "trustworthiness": float(trust),
            "random_holdout": float(rand_corr),
            "donor_holdout": float(donor_corr),
            "category_holdout": float(branch_corr),
            "passes": bool(pos_passes),
        },
        "null_shuffled": {
            "trustworthiness": float(trust_n),
            "random_holdout": float(rand_n),
            "donor_holdout": float(donor_n),
            "category_holdout": float(branch_n),
            "passes": bool(null_passes),
        },
        "verdict": verdict,
        "elapsed_s": float(time.time() - t_phase),
    }
    (REP / "h38_lite_quality_gates.json").write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 70)
    print(f"VERDICT: {verdict}")
    print(f"Total time: {time.time()-t_phase:.1f}s")


if __name__ == "__main__":
    main()
