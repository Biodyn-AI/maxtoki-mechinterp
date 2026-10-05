"""Phase 3a — autonomous manifold-hypothesis sweep over multiple candidate rulers.

Question: does MaxToki-217M's residual encode *other* biological geometries beyond
H65 (developmental) and H38 (signalling categories)?

Approach: sweep N candidate rulers, each defined as a (cell_type → category-set)
mapping plus Hamming distance. For each, train an anchor-trained LET-10D head on
the existing internal-panel centroids and evaluate the four quality gates +
paired null (within-domain global label shuffle).

Each candidate produces a row in the hypothesis registry with verdict
{POSITIVE, INCONCLUSIVE, PIPELINE_ARTEFACT}.

Outputs:
  reports/hypothesis_registry.json
  reports/hypothesis_registry.csv
  iterations/iter_<branch>/  (per-branch reports + heads)
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
ITER = RUN / "iterations"; ITER.mkdir(parents=True, exist_ok=True)


# ============================================================
# CANDIDATE MANIFOLDS — each is a function cell_type → category set,
# plus a description and a primary-axis label for the holdout group.
# ============================================================

def _ct(s: str) -> str:
    return s.strip().lower()


# H92 — naive vs memory vs effector immune state
NAIVE = {"naive thymus-derived cd4-positive, alpha-beta t cell", "naive thymus-derived cd8-positive, alpha-beta t cell",
        "naive b cell", "double-positive, alpha-beta thymocyte", "double negative thymocyte", "transitional stage b cell",
        "thymocyte"}
MEMORY = {"cd4-positive, alpha-beta memory t cell", "cd8-positive, alpha-beta memory t cell", "memory b cell",
          "class switched memory b cell", "regulatory t cell"}
EFFECTOR = {"cd8-positive, alpha-beta cytokine secreting effector t cell", "cd4-positive helper t cell",
            "natural killer cell", "type i nk t cell", "neutrophil", "mature neutrophil", "elicited macrophage",
            "mucosal invariant t cell", "gamma-delta t cell"}
TERMINAL = {"plasma cell", "plasmablast", "macrophage", "alveolar macrophage", "tissue-resident macrophage",
            "kupffer cell", "microglial cell", "erythrocyte", "platelet"}
PROGENITOR_92 = {"hematopoietic stem cell", "hematopoietic precursor cell", "common myeloid progenitor",
                 "common lymphoid progenitor", "granulocyte monocyte progenitor cell", "erythroid progenitor cell",
                 "megakaryocyte"}

def cat_h92(ct: str) -> set[str]:
    ct = _ct(ct)
    out = set()
    if ct in NAIVE: out.add("naive")
    if ct in MEMORY: out.add("memory")
    if ct in EFFECTOR: out.add("effector")
    if ct in TERMINAL: out.add("terminal")
    if ct in PROGENITOR_92: out.add("progenitor")
    return out


# H93 — lymphoid vs myeloid lineage axis (binary lineage)
LYMPHOID = {"naive thymus-derived cd4-positive, alpha-beta t cell", "cd4-positive, alpha-beta t cell",
            "cd4-positive, alpha-beta memory t cell", "cd4-positive helper t cell", "regulatory t cell",
            "naive thymus-derived cd8-positive, alpha-beta t cell", "cd8-positive, alpha-beta t cell",
            "cd8-positive, alpha-beta memory t cell", "cd8-positive, alpha-beta cytokine secreting effector t cell",
            "double-positive, alpha-beta thymocyte", "double negative thymocyte", "thymocyte",
            "mucosal invariant t cell", "gamma-delta t cell", "t cell", "mature alpha-beta t cell",
            "innate lymphoid cell", "natural killer cell", "type i nk t cell",
            "b cell", "naive b cell", "memory b cell", "class switched memory b cell",
            "transitional stage b cell", "plasma cell", "plasmablast", "common lymphoid progenitor"}
MYELOID = {"plasmacytoid dendritic cell", "conventional dendritic cell", "dendritic cell", "myeloid dendritic cell",
           "cd1c-positive myeloid dendritic cell", "cd141-positive myeloid dendritic cell",
           "monocyte", "classical monocyte", "non-classical monocyte", "intermediate monocyte",
           "macrophage", "alveolar macrophage", "tissue-resident macrophage", "kupffer cell", "microglial cell",
           "elicited macrophage",
           "neutrophil", "mature neutrophil", "basophil", "eosinophil", "mast cell",
           "common myeloid progenitor", "granulocyte monocyte progenitor cell", "erythroid progenitor cell"}

def cat_h93(ct: str) -> set[str]:
    ct = _ct(ct)
    out = set()
    if ct in LYMPHOID: out.add("lymphoid")
    if ct in MYELOID: out.add("myeloid")
    return out


# H94 — adaptive vs innate immunity
ADAPTIVE = {"naive thymus-derived cd4-positive, alpha-beta t cell", "cd4-positive, alpha-beta t cell",
            "cd4-positive, alpha-beta memory t cell", "cd4-positive helper t cell", "regulatory t cell",
            "naive thymus-derived cd8-positive, alpha-beta t cell", "cd8-positive, alpha-beta t cell",
            "cd8-positive, alpha-beta memory t cell", "cd8-positive, alpha-beta cytokine secreting effector t cell",
            "double-positive, alpha-beta thymocyte", "double negative thymocyte", "thymocyte",
            "t cell", "mature alpha-beta t cell",
            "b cell", "naive b cell", "memory b cell", "class switched memory b cell",
            "transitional stage b cell", "plasma cell", "plasmablast"}
INNATE = {"innate lymphoid cell", "natural killer cell", "type i nk t cell",
          "mucosal invariant t cell", "gamma-delta t cell",
          "plasmacytoid dendritic cell", "conventional dendritic cell", "dendritic cell", "myeloid dendritic cell",
          "cd1c-positive myeloid dendritic cell", "cd141-positive myeloid dendritic cell",
          "monocyte", "classical monocyte", "non-classical monocyte", "intermediate monocyte",
          "macrophage", "alveolar macrophage", "tissue-resident macrophage", "kupffer cell", "microglial cell",
          "elicited macrophage", "neutrophil", "mature neutrophil", "basophil", "eosinophil", "mast cell"}

def cat_h94(ct: str) -> set[str]:
    ct = _ct(ct)
    out = set()
    if ct in ADAPTIVE: out.add("adaptive")
    if ct in INNATE: out.add("innate")
    return out


# H95 — phagocytic / cytotoxic / humoral / regulatory effector axes (multi-label)
PHAGOCYTIC = {"monocyte", "classical monocyte", "non-classical monocyte", "intermediate monocyte",
              "macrophage", "alveolar macrophage", "tissue-resident macrophage", "kupffer cell", "microglial cell",
              "elicited macrophage", "neutrophil", "mature neutrophil", "dendritic cell", "myeloid dendritic cell",
              "conventional dendritic cell"}
CYTOTOXIC = {"cd8-positive, alpha-beta t cell", "cd8-positive, alpha-beta memory t cell",
             "cd8-positive, alpha-beta cytokine secreting effector t cell", "natural killer cell", "type i nk t cell",
             "gamma-delta t cell", "mucosal invariant t cell", "eosinophil"}
HUMORAL = {"plasma cell", "plasmablast", "memory b cell", "class switched memory b cell", "naive b cell", "b cell"}
REGULATORY = {"regulatory t cell", "tissue-resident macrophage", "alveolar macrophage"}

def cat_h95(ct: str) -> set[str]:
    ct = _ct(ct)
    out = set()
    if ct in PHAGOCYTIC: out.add("phagocytic")
    if ct in CYTOTOXIC: out.add("cytotoxic")
    if ct in HUMORAL: out.add("humoral")
    if ct in REGULATORY: out.add("regulatory")
    return out


# H96 — tissue-residency vs circulating
RESIDENT = {"alveolar macrophage", "tissue-resident macrophage", "kupffer cell", "microglial cell",
            "elicited macrophage", "memory b cell", "cd4-positive, alpha-beta memory t cell",
            "cd8-positive, alpha-beta memory t cell"}
CIRCULATING = {"monocyte", "classical monocyte", "non-classical monocyte", "intermediate monocyte",
               "neutrophil", "mature neutrophil", "natural killer cell", "type i nk t cell",
               "naive thymus-derived cd4-positive, alpha-beta t cell", "naive thymus-derived cd8-positive, alpha-beta t cell",
               "naive b cell", "plasmablast", "erythrocyte", "platelet"}

def cat_h96(ct: str) -> set[str]:
    ct = _ct(ct)
    out = set()
    if ct in RESIDENT: out.add("resident")
    if ct in CIRCULATING: out.add("circulating")
    return out


# H97 — myeloid maturation chain (granulocyte vs mono/macro vs DC)
GRANULOCYTE_BR = {"neutrophil", "mature neutrophil", "basophil", "eosinophil", "mast cell"}
MONO_MACRO_BR = {"monocyte", "classical monocyte", "non-classical monocyte", "intermediate monocyte",
                 "macrophage", "alveolar macrophage", "tissue-resident macrophage", "kupffer cell", "microglial cell",
                 "elicited macrophage"}
DC_BR = {"plasmacytoid dendritic cell", "conventional dendritic cell", "dendritic cell", "myeloid dendritic cell",
         "cd1c-positive myeloid dendritic cell", "cd141-positive myeloid dendritic cell"}

def cat_h97(ct: str) -> set[str]:
    ct = _ct(ct)
    out = set()
    if ct in GRANULOCYTE_BR: out.add("granulocyte")
    if ct in MONO_MACRO_BR: out.add("mono_macro")
    if ct in DC_BR: out.add("dendritic")
    return out


CANDIDATES = [
    {"name": "H92_naive_memory_effector_terminal", "categorize": cat_h92,
     "description": "Immune-state axis: naive / memory / effector / terminal / progenitor"},
    {"name": "H93_lymphoid_myeloid", "categorize": cat_h93,
     "description": "Binary lymphoid vs myeloid lineage axis"},
    {"name": "H94_adaptive_innate", "categorize": cat_h94,
     "description": "Adaptive vs innate immunity"},
    {"name": "H95_effector_modality", "categorize": cat_h95,
     "description": "Phagocytic / cytotoxic / humoral / regulatory effector axes (multi-label)"},
    {"name": "H96_resident_circulating", "categorize": cat_h96,
     "description": "Tissue-resident vs circulating"},
    {"name": "H97_myeloid_subbranches", "categorize": cat_h97,
     "description": "Myeloid sub-branches: granulocyte / mono-macro / dendritic"},
]


def build_ruler(cell_types: list[str], categorize, all_cats: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Returns (D, indicator). Hamming distance on category indicators."""
    n = len(cell_types)
    indicator = np.zeros((n, len(all_cats)), dtype=np.float32)
    for i, ct in enumerate(cell_types):
        cs = categorize(ct)
        for ci, c in enumerate(all_cats):
            if c in cs:
                indicator[i, ci] = 1.0
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        D[i] = (indicator != indicator[i]).sum(axis=1)
    return D, indicator


def evaluate_candidate(cand, feat, anchor_meta, gates) -> dict:
    name = cand["name"]
    categorize = cand["categorize"]
    cell_types = anchor_meta["cell_type"].astype(str).tolist()
    cats_per_anchor = [categorize(ct) for ct in cell_types]
    keep = np.array([bool(c) for c in cats_per_anchor])
    if keep.sum() < 30:
        return {"name": name, "verdict": "INSUFFICIENT_COVERAGE", "n_anchors_kept": int(keep.sum())}
    all_cats = sorted(set().union(*[c for c in cats_per_anchor if c]))
    cell_types_k = [cell_types[i] for i in np.where(keep)[0]]
    feat_k = feat[keep]
    anchor_meta_k = anchor_meta.loc[keep].reset_index(drop=True)

    D, indicator = build_ruler(cell_types_k, categorize, all_cats)
    if D.max() == 0:
        return {"name": name, "verdict": "DEGENERATE_RULER", "n_anchors_kept": int(keep.sum())}

    # ---- Train head + 4 gates
    head, z, _ = train_let(feat_k, D, label=name, verbose=False)
    trust = trustworthiness(feat_k, z, n_neighbors=15)
    rand_corr, _ = random_holdout_corr(feat_k, D, n_iters=5, frac=0.2)
    donor_corr, _ = grouped_holdout_corr(
        feat_k, D, anchor_meta_k["donor_id"].astype(str).to_numpy(), f"{name}-donor"
    )
    primary_cat = []
    for ct in cell_types_k:
        cs = sorted(categorize(ct))
        primary_cat.append(cs[0] if cs else "_unk")
    cat_corr, _ = grouped_holdout_corr(feat_k, D, np.array(primary_cat), f"{name}-cat")

    # ---- Null: globally shuffle category assignments
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(cell_types_k))
    cell_types_shuffled = [cell_types_k[p] for p in perm]
    D_null, _ = build_ruler(cell_types_shuffled, categorize, all_cats)
    head_n, z_n, _ = train_let(feat_k, D_null, label=f"{name}_null", verbose=False)
    trust_n = trustworthiness(feat_k, z_n, n_neighbors=15)
    rand_n, _ = random_holdout_corr(feat_k, D_null, n_iters=5, frac=0.2)
    donor_n, _ = grouped_holdout_corr(
        feat_k, D_null, anchor_meta_k["donor_id"].astype(str).to_numpy(), f"{name}-donor-null"
    )
    primary_cat_shuffled = [primary_cat[p] for p in perm]
    cat_n, _ = grouped_holdout_corr(feat_k, D_null, np.array(primary_cat_shuffled), f"{name}-cat-null")

    pos_passes = (
        trust >= gates["trustworthiness_min"]
        and rand_corr >= gates["random_holdout_correlation_min"]
        and donor_corr >= gates["donor_holdout_correlation_min"]
        and cat_corr >= gates["clade_branch_holdout_correlation_min"]
    )
    null_passes = (
        trust_n >= gates["trustworthiness_min"]
        and rand_n >= gates["random_holdout_correlation_min"]
        and donor_n >= gates["donor_holdout_correlation_min"]
        and cat_n >= gates["clade_branch_holdout_correlation_min"]
    )
    if pos_passes and not null_passes:
        verdict = "POSITIVE"
    elif not pos_passes:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "PIPELINE_ARTEFACT"

    return {
        "name": name,
        "description": cand["description"],
        "n_anchors_kept": int(keep.sum()),
        "n_categories": len(all_cats),
        "categories": all_cats,
        "ruler_max": float(D.max()),
        "positive_trust": float(trust),
        "positive_random": float(rand_corr),
        "positive_donor": float(donor_corr),
        "positive_category": float(cat_corr),
        "null_trust": float(trust_n),
        "null_random": float(rand_n),
        "null_donor": float(donor_n),
        "null_category": float(cat_n),
        "verdict": verdict,
    }


def main():
    print("=" * 70)
    print("PHASE 3a — autonomous manifold-hypothesis sweep")
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
        r = evaluate_candidate(cand, feat, anchor_meta, gates)
        r["elapsed_s"] = float(time.time() - t0)
        if "positive_trust" in r:
            print(f"  POS trust={r['positive_trust']:.3f} rand={r['positive_random']:.3f} donor={r['positive_donor']:.3f} cat={r['positive_category']:.3f}")
            print(f"  NULL trust={r['null_trust']:.3f} rand={r['null_random']:.3f} donor={r['null_donor']:.3f} cat={r['null_category']:.3f}")
        print(f"  VERDICT: {r['verdict']}  ({r['elapsed_s']:.1f}s)")
        results.append(r)

    # ---- Persist registry
    registry = {
        "scope_note": "Phase 3a manifold-hypothesis sweep — anchor-level on H65 internal panel. Each candidate = (cell_type → categories) Hamming ruler.",
        "n_candidates": len(results),
        "candidates": results,
        "elapsed_s": float(time.time() - t_phase),
    }
    (REP / "hypothesis_registry.json").write_text(json.dumps(registry, indent=2))

    # CSV form for easy reading
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "categories"} for r in results])
    df.to_csv(REP / "hypothesis_registry.csv", index=False)

    print("\n" + "=" * 70)
    print("REGISTRY SUMMARY")
    print("=" * 70)
    for r in results:
        if "positive_trust" in r:
            print(f"{r['name']:40s} {r['verdict']:18s} trust={r['positive_trust']:.3f}/{r['null_trust']:.3f}  cat={r['positive_category']:.3f}/{r['null_category']:.3f}")
        else:
            print(f"{r['name']:40s} {r['verdict']:18s}")
    print(f"\nTotal time: {time.time()-t_phase:.1f}s")


if __name__ == "__main__":
    main()
