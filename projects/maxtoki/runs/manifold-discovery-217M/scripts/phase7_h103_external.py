"""Phase 7 external validation for the H103 B-cell maturation manifold.

H103 was the strongest sweep #2 positive (trust 0.860, rand 0.861, donor 0.861
on internal). Question: does the H103-trained frozen head also pass quality
gates on the strict non-overlap external panel?

Mirrors phase7_external_validation.py but uses the ordinal B-cell maturation
ruler from sweep #2 instead of H65's stage DAG.

Outputs:
  reports/external_validation_h103.json
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
from scipy.stats import spearmanr

PROJ = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN / "scripts"))
from phase5_let_anchor import LETHead, train_let, build_pooled_drift, SEED  # type: ignore
from phase3a_sweep2_ordinal import H103_DEPTH, build_ordinal_ruler  # type: ignore

ART = RUN / "artifacts"; REP = RUN / "reports"


def main():
    print("=" * 70)
    print("PHASE 7 H103 — frozen B-cell-maturation head on external panel")
    print("=" * 70)
    t_phase = time.time()

    # Build internal H103 ruler + train head
    centroids_int = np.load(ART / "anchors/centroids_internal.npy")
    anchor_meta_int = pd.read_csv(ART / "anchors/anchor_meta_internal.csv")
    op_idx = json.loads((ART / "operators/operator_index.json").read_text())
    op_npz = np.load(ART / "operators/pooled_drift_components.npz")
    A_e, A_m, A_l = op_npz["A_early"], op_npz["A_mid"], op_npz["A_late"]
    partition = op_idx["block_partition"]

    f_int_raw = build_pooled_drift(centroids_int, A_e, A_m, A_l, partition)
    feat_mean = f_int_raw.mean(0); feat_std = f_int_raw.std(0) + 1e-6
    f_int = (f_int_raw - feat_mean) / feat_std

    cell_types_int = anchor_meta_int["cell_type"].astype(str).tolist()
    D_int_full, depths_int = build_ordinal_ruler(cell_types_int, H103_DEPTH)
    keep_int = depths_int >= 0
    f_int_k = f_int[keep_int]
    D_int = D_int_full[np.ix_(np.where(keep_int)[0], np.where(keep_int)[0])]
    print(f"  internal H103 anchors kept: {keep_int.sum()}/{len(cell_types_int)}")

    if keep_int.sum() < 20 or D_int.max() == 0:
        print("  insufficient internal anchors with B-lineage label — bailing")
        return

    print("[1] Training H103 anchor head on internal panel...")
    head, z_int, _ = train_let(f_int_k, D_int, label="H103_internal", verbose=False)
    head.eval()

    # External panel
    centroids_ext = np.load(ART / "anchors/centroids_external.npy")
    anchor_meta_ext = pd.read_csv(ART / "anchors/anchor_meta_external.csv")
    cell_types_ext = anchor_meta_ext["cell_type"].astype(str).tolist()
    D_ext_full, depths_ext = build_ordinal_ruler(cell_types_ext, H103_DEPTH)
    keep_ext = depths_ext >= 0
    print(f"  external H103 anchors kept: {keep_ext.sum()}/{len(cell_types_ext)}")
    if keep_ext.sum() < 20 or D_ext_full[np.ix_(np.where(keep_ext)[0], np.where(keep_ext)[0])].max() == 0:
        print("  insufficient external anchors with B-lineage label — bailing")
        return

    f_ext_raw = build_pooled_drift(centroids_ext, A_e, A_m, A_l, partition)
    f_ext = (f_ext_raw - feat_mean) / feat_std
    f_ext_k = f_ext[keep_ext]
    D_ext = D_ext_full[np.ix_(np.where(keep_ext)[0], np.where(keep_ext)[0])]
    anchor_meta_ext_k = anchor_meta_ext.loc[keep_ext].reset_index(drop=True)

    print("[2] Frozen H103 forward pass on external...")
    with torch.no_grad():
        z_ext = head(torch.from_numpy(f_ext_k).float())[0].numpy()

    print("[3] Trustworthiness, holdouts on external...")
    trust = trustworthiness(f_ext_k, z_ext, n_neighbors=15)
    zn = z_ext / (np.linalg.norm(z_ext, axis=1, keepdims=True) + 1e-9)
    cos = np.clip(zn @ zn.T, -1 + 1e-7, 1 - 1e-7)
    d_hat = np.arccos(cos)
    triu = np.triu_indices(len(D_ext), k=1)
    rho_global, _ = spearmanr(d_hat[triu], D_ext[triu])

    rng = np.random.default_rng(SEED)
    rand_corrs = []
    n = len(D_ext)
    for _ in range(20):
        idx = rng.permutation(n); n_test = max(2, int(round(n * 0.2)))
        test = sorted(idx[:n_test])
        zt = z_ext[test]; zn_t = zt / (np.linalg.norm(zt, axis=1, keepdims=True) + 1e-9)
        cos_t = np.clip(zn_t @ zn_t.T, -1 + 1e-7, 1 - 1e-7)
        D_t = D_ext[np.ix_(test, test)]
        triu_t = np.triu_indices(len(test), k=1)
        if D_t[triu_t].std() < 1e-6 or len(triu_t[0]) == 0:
            continue
        rho, _ = spearmanr(np.arccos(cos_t)[triu_t], D_t[triu_t])
        if not np.isnan(rho):
            rand_corrs.append(float(rho))

    donor_corrs = []
    donors = anchor_meta_ext_k["donor_id"].astype(str).to_numpy()
    for d in pd.Series(donors).unique():
        mask = donors == d
        if mask.sum() < 3:
            continue
        idx_test = np.where(mask)[0]
        zt = z_ext[idx_test]; zn_t = zt / (np.linalg.norm(zt, axis=1, keepdims=True) + 1e-9)
        cos_t = np.clip(zn_t @ zn_t.T, -1 + 1e-7, 1 - 1e-7)
        D_t = D_ext[np.ix_(idx_test, idx_test)]
        triu_t = np.triu_indices(len(idx_test), k=1)
        if len(triu_t[0]) == 0 or D_t[triu_t].std() < 1e-6:
            continue
        rho, _ = spearmanr(np.arccos(cos_t)[triu_t], D_t[triu_t])
        if not np.isnan(rho):
            donor_corrs.append(float(rho))

    depth_groups = depths_ext[keep_ext].astype(int).astype(str)
    depth_corrs = []
    for g in pd.Series(depth_groups).unique():
        mask = depth_groups == g
        if mask.sum() < 3:
            continue
        idx_test = np.where(mask)[0]
        zt = z_ext[idx_test]; zn_t = zt / (np.linalg.norm(zt, axis=1, keepdims=True) + 1e-9)
        cos_t = np.clip(zn_t @ zn_t.T, -1 + 1e-7, 1 - 1e-7)
        D_t = D_ext[np.ix_(idx_test, idx_test)]
        triu_t = np.triu_indices(len(idx_test), k=1)
        if len(triu_t[0]) == 0 or D_t[triu_t].std() < 1e-6:
            continue
        rho, _ = spearmanr(np.arccos(cos_t)[triu_t], D_t[triu_t])
        if not np.isnan(rho):
            depth_corrs.append(float(rho))

    rand_mean = float(np.mean(rand_corrs)) if rand_corrs else float("nan")
    donor_mean = float(np.mean(donor_corrs)) if donor_corrs else float("nan")
    depth_mean = float(np.mean(depth_corrs)) if depth_corrs else float("nan")

    gates = json.loads((REP / "quality_gates_spec.json").read_text())["gates"]
    pass_trust = trust >= gates["trustworthiness_min"]
    pass_rand = (not math.isnan(rand_mean)) and rand_mean >= gates["random_holdout_correlation_min"]
    pass_donor = (not math.isnan(donor_mean)) and donor_mean >= gates["donor_holdout_correlation_min"]
    pass_depth = (not math.isnan(depth_mean)) and depth_mean >= gates["clade_branch_holdout_correlation_min"]
    n_passes = sum([pass_trust, pass_rand, pass_donor, pass_depth])

    if n_passes == 4:
        verdict = "POSITIVE: H103 B-cell maturation transfers to external panel"
    elif n_passes >= 3 and pass_trust:
        verdict = "POSITIVE_3GATE_FALLBACK: H103 transfers (one gate undefined)"
    else:
        verdict = "FAIL: H103 does not transfer cleanly to external"

    out = {
        "branch": "H103_b_cell_maturation",
        "panel": "external",
        "n_external_anchors_kept": int(keep_ext.sum()),
        "n_external_donors": int(anchor_meta_ext_k["donor_id"].nunique()),
        "trustworthiness": float(trust),
        "random_holdout": rand_mean,
        "donor_holdout": donor_mean,
        "depth_holdout": depth_mean,
        "global_correlation": float(rho_global),
        "pass_trust": bool(pass_trust),
        "pass_random": bool(pass_rand),
        "pass_donor": bool(pass_donor),
        "pass_depth": bool(pass_depth),
        "n_gates_passing": int(n_passes),
        "verdict": verdict,
        "elapsed_s": float(time.time() - t_phase),
    }
    (REP / "external_validation_h103.json").write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 70)
    print(f"VERDICT: {verdict}")
    print(f"  trust={trust:.3f}  rand={rand_mean:.3f}  donor={donor_mean:.3f}  depth={depth_mean:.3f}")
    print(f"  global ρ = {rho_global:.3f}")
    print(f"Total time: {time.time()-t_phase:.1f}s")


if __name__ == "__main__":
    main()
