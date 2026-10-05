"""Phase 14 — Sweep #3: fresh manifold candidates over axes not probed in
sweeps #1 (Hamming categories H92–H97) and #2 (ordinal maturation H101–H106).

Six candidates spanning two ruler families:

  Hamming (multi-label cell-type → category indicator, Hamming distance)
    H107  Antigen-receptor type      (TCR-ab / BCR / innate-lymphoid / myeloid+prog)
    H109  Lineage compartment        (lymphocyte / MNP / granulocyte / EryMega / prog)
    H110  Tissue class               (circulating / 1° lymphoid / 2° lymphoid / mucosal / parenchymal)

  Ordinal (cell-type → integer depth, L1 distance)
    H108  Cytotoxic potential        (0=none, 1=phagocytic-killer, 2=cytotoxic-lymphoid)
    H111  Lifespan/turnover proxy    (0=hours-days, 1=days-weeks, 2=weeks-months, 3=months-years)
    H112  Adaptive→Innate axis       (0=naive-adaptive ... 3=pure-innate)

Each candidate evaluated anchor-level on the H65 internal panel against the
existing pooled-drift feature, with paired global-shuffle null. Verdict rule
matches sweeps #1+#2 (4-gate primary; 3-gate fallback when the branch/depth
holdout is undefined due to small group cardinality).

Outputs:
  reports/hypothesis_registry_sweep3.json
  reports/hypothesis_registry_sweep3.csv
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
def _ts(s): return s.strip().lower()


# ============================================================
# H107 — Antigen-receptor type (Hamming, 4 categories)
# ============================================================
TCR_AB = {
    "cd4-positive, alpha-beta t cell", "cd8-positive, alpha-beta t cell",
    "cd4-positive, alpha-beta memory t cell", "cd8-positive, alpha-beta memory t cell",
    "cd4-positive helper t cell", "cd8-positive, alpha-beta cytokine secreting effector t cell",
    "naive thymus-derived cd4-positive, alpha-beta t cell",
    "naive thymus-derived cd8-positive, alpha-beta t cell",
    "regulatory t cell", "t cell", "mature alpha-beta t cell",
    "double-positive, alpha-beta thymocyte", "double negative thymocyte", "thymocyte",
    "mucosal invariant t cell",
}
BCR_ = {
    "b cell", "naive b cell", "memory b cell", "class switched memory b cell",
    "transitional stage b cell", "plasma cell", "plasmablast",
}
INNATE_LYMPHOID_NO_AR = {
    "natural killer cell", "innate lymphoid cell",
    "type i nk t cell", "mature nk t cell",  # NKT have semi-invariant TCR; biologically innate-like
    "gamma-delta t cell",  # γδ T are innate-like, distinct receptor; group with innate
}
MYELOID_PROG_NO_AR = {
    # myeloid mononuclear phagocytes
    "monocyte", "classical monocyte", "non-classical monocyte", "intermediate monocyte",
    "macrophage", "alveolar macrophage", "tissue-resident macrophage", "kupffer cell",
    "microglial cell", "elicited macrophage",
    # dendritic
    "myeloid dendritic cell", "conventional dendritic cell", "dendritic cell",
    "cd1c-positive myeloid dendritic cell", "cd141-positive myeloid dendritic cell",
    "plasmacytoid dendritic cell",
    # granulocytes
    "neutrophil", "mature neutrophil", "basophil", "eosinophil", "mast cell",
    # progenitors / erythromegakaryocyte
    "hematopoietic stem cell", "hematopoietic precursor cell",
    "common myeloid progenitor", "common lymphoid progenitor",
    "granulocyte monocyte progenitor cell", "erythroid progenitor cell",
    "erythrocyte", "platelet", "megakaryocyte", "erythroid lineage cell",
}

def cat_h107(ct):
    ct = _ct(ct); out = set()
    if ct in TCR_AB: out.add("tcr_ab")
    if ct in BCR_: out.add("bcr")
    if ct in INNATE_LYMPHOID_NO_AR: out.add("innate_lymphoid")
    if ct in MYELOID_PROG_NO_AR: out.add("myeloid_progenitor")
    return out


# ============================================================
# H109 — Lineage compartment refined (Hamming, 5 categories)
# ============================================================
LYMPHOCYTE = TCR_AB | BCR_ | INNATE_LYMPHOID_NO_AR
MNP_ = {  # mononuclear phagocytes
    "monocyte", "classical monocyte", "non-classical monocyte", "intermediate monocyte",
    "macrophage", "alveolar macrophage", "tissue-resident macrophage", "kupffer cell",
    "microglial cell", "elicited macrophage",
    "myeloid dendritic cell", "conventional dendritic cell", "dendritic cell",
    "cd1c-positive myeloid dendritic cell", "cd141-positive myeloid dendritic cell",
    "plasmacytoid dendritic cell",
}
GRANULOCYTE = {"neutrophil", "mature neutrophil", "basophil", "eosinophil", "mast cell"}
ERY_MEGA = {"erythrocyte", "erythroid progenitor cell", "platelet", "megakaryocyte", "erythroid lineage cell"}
PROGENITOR_109 = {
    "hematopoietic stem cell", "hematopoietic precursor cell",
    "common myeloid progenitor", "common lymphoid progenitor",
    "granulocyte monocyte progenitor cell",
    "thymocyte", "double-positive, alpha-beta thymocyte", "double negative thymocyte",
}

def cat_h109(ct):
    ct = _ct(ct); out = set()
    if ct in LYMPHOCYTE - PROGENITOR_109: out.add("lymphocyte")
    if ct in MNP_: out.add("mnp")
    if ct in GRANULOCYTE: out.add("granulocyte")
    if ct in ERY_MEGA: out.add("ery_mega")
    if ct in PROGENITOR_109: out.add("progenitor")
    return out


# ============================================================
# H110 — Tissue class (Hamming, 5 buckets) — uses tissue field, not cell_type
# ============================================================
TISSUE_CIRCULATING = {"blood"}
TISSUE_PRIMARY_LYMPHOID = {"bone marrow", "thymus"}
TISSUE_SECONDARY_LYMPHOID = {"spleen", "lymph node", "inguinal lymph node"}
TISSUE_MUCOSAL = {
    "lung", "trachea", "mucosa of stomach", "stomach smooth muscle",
    "muscularis mucosae of stomach",
    "duodenum", "ascending colon", "sigmoid colon",
    "skin of chest", "skin of abdomen",
    "cornea", "sclera", "eye", "lacrimal gland", "retinal neural layer",
}
# everything else → parenchymal_other (kidney, liver, bladder, muscle, etc.)

def cat_h110_from_tissue(tissue):
    t = _ts(tissue); out = set()
    if t in TISSUE_CIRCULATING: out.add("circulating")
    elif t in TISSUE_PRIMARY_LYMPHOID: out.add("primary_lymphoid")
    elif t in TISSUE_SECONDARY_LYMPHOID: out.add("secondary_lymphoid")
    elif t in TISSUE_MUCOSAL: out.add("mucosal")
    else: out.add("parenchymal_other")
    return out


# ============================================================
# H108 — Cytotoxic potential (Ordinal 0..2)
# ============================================================
H108_DEPTH = {
    # 0: non-cytotoxic
    "b cell": 0, "naive b cell": 0, "memory b cell": 0, "class switched memory b cell": 0,
    "transitional stage b cell": 0, "plasma cell": 0, "plasmablast": 0,
    "naive thymus-derived cd4-positive, alpha-beta t cell": 0,
    "naive thymus-derived cd8-positive, alpha-beta t cell": 0,
    "cd4-positive, alpha-beta t cell": 0,  # helper, not classically cyto
    "cd4-positive, alpha-beta memory t cell": 0,
    "cd4-positive helper t cell": 0,
    "regulatory t cell": 0,
    "innate lymphoid cell": 0,
    "common lymphoid progenitor": 0, "common myeloid progenitor": 0,
    "hematopoietic stem cell": 0, "hematopoietic precursor cell": 0,
    "thymocyte": 0, "double negative thymocyte": 0, "double-positive, alpha-beta thymocyte": 0,
    "erythroid progenitor cell": 0, "erythrocyte": 0, "platelet": 0,
    "myeloid dendritic cell": 0, "conventional dendritic cell": 0,
    "dendritic cell": 0, "plasmacytoid dendritic cell": 0,
    "cd1c-positive myeloid dendritic cell": 0, "cd141-positive myeloid dendritic cell": 0,
    # 1: phagocytic killer / antimicrobial degranulation (myeloid effector)
    "monocyte": 1, "classical monocyte": 1, "non-classical monocyte": 1, "intermediate monocyte": 1,
    "macrophage": 1, "alveolar macrophage": 1, "tissue-resident macrophage": 1,
    "kupffer cell": 1, "microglial cell": 1, "elicited macrophage": 1,
    "neutrophil": 1, "mature neutrophil": 1, "basophil": 1, "eosinophil": 1, "mast cell": 1,
    # 2: cytotoxic granular lymphocyte / armed effector
    "cd8-positive, alpha-beta t cell": 2,
    "cd8-positive, alpha-beta memory t cell": 2,
    "cd8-positive, alpha-beta cytokine secreting effector t cell": 2,
    "natural killer cell": 2,
    "type i nk t cell": 2, "mature nk t cell": 2,
    "gamma-delta t cell": 2,
    "mucosal invariant t cell": 2,
    "t cell": 1,                # generic T at level 1 (mixed cyto/non)
    "mature alpha-beta t cell": 1,
}


# ============================================================
# H111 — Lifespan / turnover proxy (Ordinal 0..3)
# ============================================================
H111_DEPTH = {
    # 0: short-lived granulocytes / activated effectors (hours-days)
    "neutrophil": 0, "mature neutrophil": 0, "basophil": 0, "mast cell": 0, "eosinophil": 0,
    # 1: medium turnover (days-weeks): monocytes, DC, eryth, platelet, progenitors
    "monocyte": 1, "classical monocyte": 1, "non-classical monocyte": 1, "intermediate monocyte": 1,
    "myeloid dendritic cell": 1, "conventional dendritic cell": 1, "dendritic cell": 1,
    "plasmacytoid dendritic cell": 1,
    "cd1c-positive myeloid dendritic cell": 1, "cd141-positive myeloid dendritic cell": 1,
    "erythrocyte": 1, "platelet": 1, "erythroid progenitor cell": 1,
    "common myeloid progenitor": 1, "common lymphoid progenitor": 1,
    "granulocyte monocyte progenitor cell": 1,
    "hematopoietic stem cell": 1, "hematopoietic precursor cell": 1,
    "thymocyte": 1, "double negative thymocyte": 1, "double-positive, alpha-beta thymocyte": 1,
    # 2: long-circulating lymphocytes (weeks-months)
    "b cell": 2, "naive b cell": 2, "transitional stage b cell": 2,
    "naive thymus-derived cd4-positive, alpha-beta t cell": 2,
    "naive thymus-derived cd8-positive, alpha-beta t cell": 2,
    "cd4-positive, alpha-beta t cell": 2, "cd8-positive, alpha-beta t cell": 2,
    "t cell": 2, "mature alpha-beta t cell": 2,
    "cd4-positive helper t cell": 2, "cd8-positive, alpha-beta cytokine secreting effector t cell": 2,
    "regulatory t cell": 2, "natural killer cell": 2, "innate lymphoid cell": 2,
    "type i nk t cell": 2, "mature nk t cell": 2,
    "gamma-delta t cell": 2, "mucosal invariant t cell": 2,
    # 3: long-lived terminally differentiated (months-years)
    "plasma cell": 3, "memory b cell": 3, "class switched memory b cell": 3,
    "cd4-positive, alpha-beta memory t cell": 3, "cd8-positive, alpha-beta memory t cell": 3,
    "macrophage": 3, "alveolar macrophage": 3, "tissue-resident macrophage": 3,
    "kupffer cell": 3, "microglial cell": 3, "elicited macrophage": 3,
}


# ============================================================
# H112 — Adaptive→Innate axis (Ordinal 0..3)
# ============================================================
H112_DEPTH = {
    # 0: pure naive adaptive
    "naive thymus-derived cd4-positive, alpha-beta t cell": 0,
    "naive thymus-derived cd8-positive, alpha-beta t cell": 0,
    "naive b cell": 0, "transitional stage b cell": 0,
    "double negative thymocyte": 0, "double-positive, alpha-beta thymocyte": 0,
    "thymocyte": 0,
    # 1: experienced adaptive (memory/effector lymphocyte)
    "cd4-positive, alpha-beta t cell": 1, "cd8-positive, alpha-beta t cell": 1,
    "cd4-positive, alpha-beta memory t cell": 1, "cd8-positive, alpha-beta memory t cell": 1,
    "cd4-positive helper t cell": 1, "cd8-positive, alpha-beta cytokine secreting effector t cell": 1,
    "regulatory t cell": 1,
    "b cell": 1, "memory b cell": 1, "class switched memory b cell": 1,
    "plasma cell": 1, "plasmablast": 1,
    "t cell": 1, "mature alpha-beta t cell": 1,
    # 2: innate-like adaptive (NKT, γδ T, MAIT — semi-invariant TCR / unconventional)
    "type i nk t cell": 2, "mature nk t cell": 2,
    "gamma-delta t cell": 2, "mucosal invariant t cell": 2,
    # 3: pure innate
    "natural killer cell": 3, "innate lymphoid cell": 3,
    "monocyte": 3, "classical monocyte": 3, "non-classical monocyte": 3, "intermediate monocyte": 3,
    "macrophage": 3, "alveolar macrophage": 3, "tissue-resident macrophage": 3,
    "kupffer cell": 3, "microglial cell": 3, "elicited macrophage": 3,
    "myeloid dendritic cell": 3, "conventional dendritic cell": 3, "dendritic cell": 3,
    "cd1c-positive myeloid dendritic cell": 3, "cd141-positive myeloid dendritic cell": 3,
    "plasmacytoid dendritic cell": 3,
    "neutrophil": 3, "mature neutrophil": 3, "basophil": 3, "eosinophil": 3, "mast cell": 3,
    # progenitors and erythromega: leave UNASSIGNED → filtered out
}


CANDIDATES = [
    {"name": "H107_antigen_receptor_type", "type": "hamming", "categorize": cat_h107,
     "description": "Antigen-receptor type: TCR-αβ / BCR / innate-lymphoid / myeloid-prog (Hamming)"},
    {"name": "H108_cytotoxic_potential", "type": "ordinal", "depth_map": H108_DEPTH,
     "description": "Cytotoxic potential ordinal: 0 non-cyto / 1 phagocytic-killer / 2 cyto-lymphoid"},
    {"name": "H109_lineage_compartment_refined", "type": "hamming", "categorize": cat_h109,
     "description": "Refined lineage: lymphocyte / MNP / granulocyte / ery-mega / progenitor"},
    {"name": "H110_tissue_class", "type": "hamming_tissue", "categorize": cat_h110_from_tissue,
     "description": "Tissue class: circulating / 1° lymph / 2° lymph / mucosal / parenchymal"},
    {"name": "H111_lifespan_turnover", "type": "ordinal", "depth_map": H111_DEPTH,
     "description": "Cell lifespan ordinal: 0 hours-days / 1 days-weeks / 2 weeks-months / 3 months-years"},
    {"name": "H112_adaptive_innate_ordinal", "type": "ordinal", "depth_map": H112_DEPTH,
     "description": "Adaptive→Innate ordinal: 0 naive / 1 experienced / 2 innate-like / 3 pure innate"},
]


def build_hamming_ruler(labels, categorize):
    cats_per = [categorize(x) for x in labels]
    keep = np.array([bool(c) for c in cats_per])
    if keep.sum() < 30:
        return None, None, None, keep
    all_cats = sorted(set().union(*[c for c in cats_per if c]))
    n = len(labels)
    indicator = np.zeros((n, len(all_cats)), dtype=np.float32)
    for i, cs in enumerate(cats_per):
        for ci, c in enumerate(all_cats):
            if c in cs:
                indicator[i, ci] = 1.0
    idx = np.where(keep)[0]
    indicator_k = indicator[idx]
    D = (indicator_k[:, None, :] != indicator_k[None, :, :]).sum(axis=-1).astype(np.float32)
    primary = []
    for i in idx:
        cs = sorted(cats_per[i])
        primary.append(cs[0] if cs else "_unk")
    return D, np.array(primary), all_cats, keep


def build_ordinal_ruler(cell_types, depth_map):
    n = len(cell_types)
    depths = np.array([depth_map.get(_ct(ct), -1) for ct in cell_types], dtype=np.float32)
    keep = depths >= 0
    if keep.sum() < 30:
        return None, None, keep
    idx = np.where(keep)[0]
    d_k = depths[idx]
    D = np.abs(d_k[:, None] - d_k[None, :]).astype(np.float32)
    return D, d_k, keep


def evaluate(cand, feat, anchor_meta, gates):
    name = cand["name"]
    if cand["type"] == "hamming":
        labels = anchor_meta["cell_type"].astype(str).tolist()
        D, primary_group, all_cats, keep = build_hamming_ruler(labels, cand["categorize"])
        ruler_kind = "hamming_celltype"
    elif cand["type"] == "hamming_tissue":
        labels = anchor_meta["tissue"].astype(str).tolist()
        D, primary_group, all_cats, keep = build_hamming_ruler(labels, cand["categorize"])
        ruler_kind = "hamming_tissue"
    else:
        labels = anchor_meta["cell_type"].astype(str).tolist()
        D, depths_k, keep = build_ordinal_ruler(labels, cand["depth_map"])
        primary_group = depths_k.astype(int).astype(str) if D is not None else None
        all_cats = None
        ruler_kind = "ordinal"

    if D is None:
        return {"name": name, "verdict": "INSUFFICIENT_COVERAGE", "n_anchors_kept": int(keep.sum())}
    if D.max() == 0:
        return {"name": name, "verdict": "DEGENERATE_RULER", "n_anchors_kept": int(keep.sum())}

    feat_k = feat[keep]
    anchor_meta_k = anchor_meta.loc[keep].reset_index(drop=True)

    head, z, _ = train_let(feat_k, D, label=name, verbose=False)
    trust = trustworthiness(feat_k, z, n_neighbors=15)
    rand_corr, _ = random_holdout_corr(feat_k, D, n_iters=5, frac=0.2)
    donor_corr, _ = grouped_holdout_corr(
        feat_k, D, anchor_meta_k["donor_id"].astype(str).to_numpy(), f"{name}-donor"
    )
    cat_corr, _ = grouped_holdout_corr(feat_k, D, np.array(primary_group), f"{name}-cat")

    # Null: globally shuffle the *primary label* used to build the ruler
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(int(keep.sum()))
    if cand["type"] == "ordinal":
        depths_shuf = depths_k[perm]
        D_null = np.abs(depths_shuf[:, None] - depths_shuf[None, :]).astype(np.float32)
        primary_shuf = depths_shuf.astype(int).astype(str)
    else:
        # rebuild Hamming D by shuffling the underlying labels (cell_type or tissue)
        labels_k = [labels[i] for i in np.where(keep)[0]]
        labels_shuf = [labels_k[p] for p in perm]
        D_null_full, primary_shuf, _, _ = build_hamming_ruler(labels_shuf, cand["categorize"])
        D_null = D_null_full

    head_n, z_n, _ = train_let(feat_k, D_null, label=f"{name}_null", verbose=False)
    trust_n = trustworthiness(feat_k, z_n, n_neighbors=15)
    rand_n, _ = random_holdout_corr(feat_k, D_null, n_iters=5, frac=0.2)
    donor_n, _ = grouped_holdout_corr(
        feat_k, D_null, anchor_meta_k["donor_id"].astype(str).to_numpy(), f"{name}-donor-null"
    )
    cat_n, _ = grouped_holdout_corr(feat_k, D_null, np.array(primary_shuf), f"{name}-cat-null")

    def _ge(x, t):
        return (not (isinstance(x, float) and math.isnan(x))) and x >= t

    pos4 = (
        _ge(trust, gates["trustworthiness_min"])
        and _ge(rand_corr, gates["random_holdout_correlation_min"])
        and _ge(donor_corr, gates["donor_holdout_correlation_min"])
        and _ge(cat_corr, gates["clade_branch_holdout_correlation_min"])
    )
    null4 = (
        _ge(trust_n, gates["trustworthiness_min"])
        and _ge(rand_n, gates["random_holdout_correlation_min"])
        and _ge(donor_n, gates["donor_holdout_correlation_min"])
        and _ge(cat_n, gates["clade_branch_holdout_correlation_min"])
    )
    pos3 = (
        _ge(trust, gates["trustworthiness_min"])
        and _ge(rand_corr, gates["random_holdout_correlation_min"])
        and _ge(donor_corr, gates["donor_holdout_correlation_min"])
    )
    null3 = (
        _ge(trust_n, gates["trustworthiness_min"])
        and _ge(rand_n, gates["random_holdout_correlation_min"])
        and _ge(donor_n, gates["donor_holdout_correlation_min"])
    )

    trust_subthreshold = not _ge(trust, gates["trustworthiness_min"])
    rand_directional = _ge(rand_corr, 0.5 + gates["random_holdout_correlation_min"])
    donor_directional = _ge(donor_corr, 0.5 + gates["donor_holdout_correlation_min"])

    if pos4 and not null4:
        verdict = "POSITIVE"
    elif pos3 and not null3 and not pos4:
        verdict = "POSITIVE_3GATE_FALLBACK"
    elif trust_subthreshold and (rand_directional or donor_directional):
        verdict = "DIRECTIONAL_TRUST_SUBTHRESHOLD"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "name": name,
        "description": cand["description"],
        "ruler_kind": ruler_kind,
        "n_anchors_kept": int(keep.sum()),
        "n_categories": (len(all_cats) if all_cats is not None else None),
        "categories": all_cats,
        "ruler_max": float(D.max()),
        "positive_trust": float(trust),
        "positive_random": float(rand_corr) if not (isinstance(rand_corr, float) and math.isnan(rand_corr)) else None,
        "positive_donor": float(donor_corr) if not (isinstance(donor_corr, float) and math.isnan(donor_corr)) else None,
        "positive_category": float(cat_corr) if not (isinstance(cat_corr, float) and math.isnan(cat_corr)) else None,
        "null_trust": float(trust_n),
        "null_random": float(rand_n) if not (isinstance(rand_n, float) and math.isnan(rand_n)) else None,
        "null_donor": float(donor_n) if not (isinstance(donor_n, float) and math.isnan(donor_n)) else None,
        "null_category": float(cat_n) if not (isinstance(cat_n, float) and math.isnan(cat_n)) else None,
        "verdict": verdict,
    }


def main():
    print("=" * 70)
    print("PHASE 14 SWEEP #3 — fresh manifold candidates (mixed Hamming + ordinal)")
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
            print(f"  POS trust={r['positive_trust']:.3f} rand={r.get('positive_random')} donor={r.get('positive_donor')} cat={r.get('positive_category')}")
            print(f"  NULL trust={r['null_trust']:.3f} rand={r.get('null_random')} donor={r.get('null_donor')} cat={r.get('null_category')}")
        print(f"  VERDICT: {r['verdict']}  ({r['elapsed_s']:.1f}s)")
        results.append(r)

    out = {
        "scope_note": "Phase 14 Sweep #3 — fresh axes mixing Hamming + ordinal rulers, anchor-level on H65 internal panel.",
        "n_candidates": len(results),
        "candidates": results,
        "elapsed_s": float(time.time() - t_phase),
    }
    (REP / "hypothesis_registry_sweep3.json").write_text(json.dumps(out, indent=2))
    df_rows = []
    for r in results:
        row = {k: v for k, v in r.items() if k != "categories"}
        df_rows.append(row)
    pd.DataFrame(df_rows).to_csv(REP / "hypothesis_registry_sweep3.csv", index=False)
    print("\n=== Sweep #3 SUMMARY ===")
    for r in results:
        if "positive_trust" in r:
            print(f"{r['name']:38s} {r['verdict']:32s} trust={r['positive_trust']:.3f}/{r['null_trust']:.3f}")
        else:
            print(f"{r['name']:38s} {r['verdict']:32s}")
    print(f"\nTotal time: {time.time()-t_phase:.1f}s")


if __name__ == "__main__":
    main()
