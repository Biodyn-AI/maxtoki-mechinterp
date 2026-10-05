"""Phase 15 — unified external + zero-shot validation for any anchor-level
manifold candidate (H38 LITE, H95 effector modality, or any Sweep #3 positive).

Given a candidate name + ruler-spec, this script:

  1. Trains the LET head on the H65 internal panel using the candidate's ruler.
  2. Applies the FROZEN head to the external panel; reports trust + 4 holdouts.
  3. Applies the FROZEN head to the zeroshot panel; reports same.

Verdict per panel uses the same 4-gate-with-3-gate-fallback rule as Sweep
#3 + the existing phase7_h103_external.py.

Outputs:
  reports/external_validation_<NAME>.json
  reports/zeroshot_<NAME>.json

Usage:
  python scripts/phase15_validate_candidate.py <candidate_name>

Where <candidate_name> ∈ {H38_lite, H95, H107, H108, H109, H110, H111, H112}.
"""
from __future__ import annotations

import argparse
import json
import math
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
    LETHead, train_let, build_pooled_drift, SEED,
)
from phase14_sweep3 import (  # type: ignore
    cat_h107, cat_h109, cat_h110_from_tissue,
    H108_DEPTH, H111_DEPTH, H112_DEPTH,
    build_hamming_ruler as _build_ham,
    build_ordinal_ruler as _build_ord,
)
from phase13_h38_lite import CELL_TYPE_TO_CATEGORIES as H38_CATS  # type: ignore
from phase3a_manifold_sweep import cat_h95  # type: ignore
from phase3a_sweep2_ordinal import H101_DEPTH  # type: ignore

ART = RUN / "artifacts"; REP = RUN / "reports"


# ----- candidate registry: (kind, ruler-builder, label-source) -----
def _h38_categorize(ct: str):
    return H38_CATS.get(ct.strip().lower(), set())


CAND_REGISTRY = {
    "H38_lite": {"kind": "hamming_ct", "categorize": _h38_categorize, "label": "H38_LITE_signalling"},
    "H95":      {"kind": "hamming_ct", "categorize": cat_h95,         "label": "H95_effector_modality"},
    "H101":     {"kind": "ordinal", "depth_map": H101_DEPTH, "label": "H101_whole_hema_maturation"},
    "H107":     {"kind": "hamming_ct", "categorize": cat_h107,        "label": "H107_antigen_receptor_type"},
    "H109":     {"kind": "hamming_ct", "categorize": cat_h109,        "label": "H109_lineage_compartment_refined"},
    "H110":     {"kind": "hamming_tissue", "categorize": cat_h110_from_tissue, "label": "H110_tissue_class"},
    "H108":     {"kind": "ordinal", "depth_map": H108_DEPTH, "label": "H108_cytotoxic_potential"},
    "H111":     {"kind": "ordinal", "depth_map": H111_DEPTH, "label": "H111_lifespan_turnover"},
    "H112":     {"kind": "ordinal", "depth_map": H112_DEPTH, "label": "H112_adaptive_innate_ordinal"},
}


def build_ruler_for_panel(spec, anchor_meta):
    if spec["kind"] == "hamming_ct":
        labels = anchor_meta["cell_type"].astype(str).tolist()
        D, primary, all_cats, keep = _build_ham(labels, spec["categorize"])
    elif spec["kind"] == "hamming_tissue":
        labels = anchor_meta["tissue"].astype(str).tolist()
        D, primary, all_cats, keep = _build_ham(labels, spec["categorize"])
    else:
        labels = anchor_meta["cell_type"].astype(str).tolist()
        D, depths_k, keep = _build_ord(labels, spec["depth_map"])
        primary = depths_k.astype(int).astype(str) if D is not None else None
        all_cats = None
    return D, primary, all_cats, keep


def per_group_spearman(z, D, group_arr, min_size=3):
    out = []
    for g in pd.Series(group_arr).unique():
        mask = group_arr == g
        if mask.sum() < min_size:
            continue
        idx = np.where(mask)[0]
        zt = z[idx]
        zn = zt / (np.linalg.norm(zt, axis=1, keepdims=True) + 1e-9)
        cos_t = np.clip(zn @ zn.T, -1 + 1e-7, 1 - 1e-7)
        d_hat = np.arccos(cos_t)
        D_t = D[np.ix_(idx, idx)]
        triu = np.triu_indices(len(idx), k=1)
        if len(triu[0]) == 0 or D_t[triu].std() < 1e-6:
            continue
        rho, _ = spearmanr(d_hat[triu], D_t[triu])
        if not np.isnan(rho):
            out.append(float(rho))
    return out


def random_holdout_spearman(z, D, n_iters=20, frac=0.2, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(D); out = []
    for _ in range(n_iters):
        idx = rng.permutation(n)
        n_test = max(2, int(round(n * frac)))
        test = sorted(idx[:n_test])
        zt = z[test]; zn = zt / (np.linalg.norm(zt, axis=1, keepdims=True) + 1e-9)
        cos_t = np.clip(zn @ zn.T, -1 + 1e-7, 1 - 1e-7)
        D_t = D[np.ix_(test, test)]
        triu = np.triu_indices(len(test), k=1)
        if len(triu[0]) == 0 or D_t[triu].std() < 1e-6:
            continue
        rho, _ = spearmanr(np.arccos(cos_t)[triu], D_t[triu])
        if not np.isnan(rho):
            out.append(float(rho))
    return out


def evaluate_frozen(head, feat_int_mean, feat_int_std, A_e, A_m, A_l, partition,
                     panel_centroids, panel_meta, ruler_spec, panel_name, gates):
    """Apply frozen head to a panel; compute gates + verdict."""
    f_raw = build_pooled_drift(panel_centroids, A_e, A_m, A_l, partition)
    f = (f_raw - feat_int_mean) / feat_int_std

    D, primary, all_cats, keep = build_ruler_for_panel(ruler_spec, panel_meta)
    if D is None:
        return {"panel": panel_name, "verdict": f"INSUFFICIENT_COVERAGE_{panel_name}",
                "n_anchors_kept": int(keep.sum()) if keep is not None else 0}
    if D.max() == 0:
        return {"panel": panel_name, "verdict": f"DEGENERATE_RULER_{panel_name}",
                "n_anchors_kept": int(keep.sum())}
    f_k = f[keep]
    panel_meta_k = panel_meta.loc[keep].reset_index(drop=True)

    with torch.no_grad():
        z = head(torch.from_numpy(f_k).float())[0].numpy()

    trust = float(trustworthiness(f_k, z, n_neighbors=15))
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    cos = np.clip(zn @ zn.T, -1 + 1e-7, 1 - 1e-7)
    d_hat = np.arccos(cos)
    triu = np.triu_indices(len(D), k=1)
    rho_global, _ = spearmanr(d_hat[triu], D[triu])

    rand = random_holdout_spearman(z, D)
    donors = panel_meta_k["donor_id"].astype(str).to_numpy()
    donor = per_group_spearman(z, D, donors)
    cat = per_group_spearman(z, D, np.array(primary))

    rand_mean = float(np.mean(rand)) if rand else float("nan")
    donor_mean = float(np.mean(donor)) if donor else float("nan")
    cat_mean = float(np.mean(cat)) if cat else float("nan")

    def _ge(x, t):
        return (not (isinstance(x, float) and math.isnan(x))) and x >= t

    pass_trust = _ge(trust, gates["trustworthiness_min"])
    pass_rand = _ge(rand_mean, gates["random_holdout_correlation_min"])
    pass_donor = _ge(donor_mean, gates["donor_holdout_correlation_min"])
    pass_cat = _ge(cat_mean, gates["clade_branch_holdout_correlation_min"])
    n_passes = sum([pass_trust, pass_rand, pass_donor, pass_cat])

    if n_passes == 4:
        verdict = f"POSITIVE_{panel_name}"
    elif n_passes >= 3 and pass_trust:
        verdict = f"POSITIVE_3GATE_FALLBACK_{panel_name}"
    else:
        verdict = f"FAIL_{panel_name}"

    return {
        "panel": panel_name,
        "n_anchors_kept": int(keep.sum()),
        "n_donors": int(panel_meta_k["donor_id"].nunique()),
        "n_categories": (len(all_cats) if all_cats is not None else None),
        "ruler_max": float(D.max()),
        "trustworthiness": trust,
        "random_holdout": rand_mean,
        "donor_holdout": donor_mean,
        "category_holdout": cat_mean,
        "global_correlation": float(rho_global),
        "pass_trust": bool(pass_trust),
        "pass_random": bool(pass_rand),
        "pass_donor": bool(pass_donor),
        "pass_category": bool(pass_cat),
        "n_gates_passing": int(n_passes),
        "verdict": verdict,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help=f"candidate ∈ {sorted(CAND_REGISTRY.keys())}")
    ap.add_argument("--skip-zeroshot", action="store_true")
    ap.add_argument("--skip-external", action="store_true")
    args = ap.parse_args()

    if args.name not in CAND_REGISTRY:
        print(f"Unknown candidate {args.name}; valid: {sorted(CAND_REGISTRY)}")
        sys.exit(2)

    spec = CAND_REGISTRY[args.name]
    label = spec["label"]
    print("=" * 70)
    print(f"PHASE 15 — frozen-head transfer for {label}")
    print("=" * 70)
    t_phase = time.time()

    centroids_int = np.load(ART / "anchors/centroids_internal.npy")
    anchor_meta_int = pd.read_csv(ART / "anchors/anchor_meta_internal.csv")
    op_idx = json.loads((ART / "operators/operator_index.json").read_text())
    op_npz = np.load(ART / "operators/pooled_drift_components.npz")
    A_e, A_m, A_l = op_npz["A_early"], op_npz["A_mid"], op_npz["A_late"]
    partition = op_idx["block_partition"]
    f_int_raw = build_pooled_drift(centroids_int, A_e, A_m, A_l, partition)
    feat_mean = f_int_raw.mean(0); feat_std = f_int_raw.std(0) + 1e-6
    f_int = (f_int_raw - feat_mean) / feat_std

    D_int, primary_int, all_cats_int, keep_int = build_ruler_for_panel(spec, anchor_meta_int)
    if D_int is None or D_int.max() == 0:
        print(f"  insufficient internal coverage for {label}")
        sys.exit(3)
    f_int_k = f_int[keep_int]
    print(f"  internal anchors kept: {keep_int.sum()}/{len(anchor_meta_int)}")

    print("[1] Training anchor LET head on internal panel...")
    head, z_int, _ = train_let(f_int_k, D_int, label=f"{label}_internal", verbose=False)
    head.eval()

    gates = json.loads((REP / "quality_gates_spec.json").read_text())["gates"]

    out = {"branch": label, "elapsed_s_total": None}

    if not args.skip_external:
        print("\n[2] External panel...")
        centroids_ext = np.load(ART / "anchors/centroids_external.npy")
        anchor_meta_ext = pd.read_csv(ART / "anchors/anchor_meta_external.csv")
        ext = evaluate_frozen(head, feat_mean, feat_std, A_e, A_m, A_l, partition,
                              centroids_ext, anchor_meta_ext, spec, "external", gates)
        print(json.dumps(ext, indent=2))
        out["external"] = ext
        outpath_ext = REP / f"external_validation_{args.name}.json"
        outpath_ext.write_text(json.dumps(ext, indent=2))
        print(f"  → {outpath_ext}")

    if not args.skip_zeroshot:
        print("\n[3] Zero-shot panel...")
        centroids_zs = np.load(ART / "anchors/centroids_zeroshot.npy")
        anchor_meta_zs = pd.read_csv(ART / "anchors/anchor_meta_zeroshot.csv")
        zs = evaluate_frozen(head, feat_mean, feat_std, A_e, A_m, A_l, partition,
                             centroids_zs, anchor_meta_zs, spec, "zeroshot", gates)
        print(json.dumps(zs, indent=2))
        out["zeroshot"] = zs
        outpath_zs = REP / f"zeroshot_{args.name}.json"
        outpath_zs.write_text(json.dumps(zs, indent=2))
        print(f"  → {outpath_zs}")

    out["elapsed_s_total"] = float(time.time() - t_phase)
    (REP / f"phase15_{args.name}_summary.json").write_text(json.dumps(out, indent=2))

    print(f"\nTotal time: {time.time()-t_phase:.1f}s")


if __name__ == "__main__":
    main()
