"""Phase 17 — Cross-axis orthogonality test: H65 (developmental DAG) × H95
(effector modality Hamming).

Question: are H65 and H95 encoded as ORTHOGONAL axes in MaxToki-217M's
residual stream, or do they collapse onto the same direction? FINAL_SUMMARY
calls the residual a "multi-axis biological hub" — Phase 17 quantifies that
claim.

Method
------
1. Restrict to anchors mapped by BOTH rulers (~208/290 — H95 leaves out
   progenitor/erythroid/non-immune cell types).
2. Train two independent LET heads on the same anchors and same features:
   - H_h65: 10D head trained against H65 stage-DAG distance.
   - H_h95: 10D head trained against H95 effector-modality Hamming.
3. For each head, compute pairwise latent arc-cos distances over the
   restricted anchor set: D̂_h65 (n×n), D̂_h95 (n×n).
4. Orthogonality = Spearman correlation between D̂_h65 and D̂_h95 over the
   upper triangle. Low → orthogonal axes; high → collapsed.
5. Sanity checks:
   - Spearman(D̂_h65, D_target_h65)  — head fits its own ruler
   - Spearman(D̂_h95, D_target_h95)  — head fits its own ruler
   - Spearman(D_target_h65, D_target_h95)  — ground-truth ruler overlap
6. Compare to a paired NULL: scrambled-H95 ruler. If H_h95 is real, the
   real-H95 head should produce a markedly different latent than the
   shuffled-H95 head; if H_h95 is collapsed onto H_h65, the difference is
   small.

Outputs:
  reports/phase17_cross_axis_h65_h95.json
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
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness

PROJ = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN / "scripts"))
from phase5_let_anchor import train_let, build_pooled_drift, SEED  # type: ignore
from phase3a_manifold_sweep import cat_h95, build_ruler  # type: ignore

ART = RUN / "artifacts"; REP = RUN / "reports"


def latent_distance(z: np.ndarray) -> np.ndarray:
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    cos = np.clip(zn @ zn.T, -1 + 1e-7, 1 - 1e-7)
    return np.arccos(cos)


def main():
    print("=" * 70)
    print("PHASE 17 — Cross-axis orthogonality H65 × H95")
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

    # H65 ruler — already cached
    D_h65_full = np.load(ART / "anchors/d_target_internal.npy")
    cell_types = anchor_meta["cell_type"].astype(str).tolist()

    # H95 ruler — build from existing categorize fn
    cats_per = [cat_h95(ct) for ct in cell_types]
    keep_h95 = np.array([bool(c) for c in cats_per])
    if keep_h95.sum() < 30:
        print("Insufficient H95 coverage; aborting"); return
    all_cats = sorted(set().union(*[c for c in cats_per if c]))
    n_full = len(cell_types)
    indicator = np.zeros((n_full, len(all_cats)), dtype=np.float32)
    for i, cs in enumerate(cats_per):
        for ci, c in enumerate(all_cats):
            if c in cs: indicator[i, ci] = 1.0

    # Intersection: H95 ⊂ H65 (since all H95 cell-types are in H65)
    idx_int = np.where(keep_h95)[0]
    n = len(idx_int)
    feat_k = feat[idx_int]
    cell_types_k = [cell_types[i] for i in idx_int]
    indicator_k = indicator[idx_int]
    D_h65 = D_h65_full[np.ix_(idx_int, idx_int)]
    D_h95 = (indicator_k[:, None, :] != indicator_k[None, :, :]).sum(axis=-1).astype(np.float32)
    print(f"  Anchor intersection (H65 ∩ H95): {n}/{n_full}")
    print(f"  D_h65 max/median: {D_h65.max():.1f}/{np.median(D_h65):.1f}")
    print(f"  D_h95 max/median: {D_h95.max():.1f}/{np.median(D_h95):.1f}")

    # Ground-truth ruler overlap
    triu = np.triu_indices(n, k=1)
    rho_target, _ = spearmanr(D_h65[triu], D_h95[triu])
    print(f"  Spearman(D_target_h65, D_target_h95) = {rho_target:.3f}")

    # Train head H_h65 on intersection-restricted anchors
    print("\n[1] Train H_h65 on intersection-restricted anchors...")
    head_h65, z_h65, _ = train_let(feat_k, D_h65, label="H_h65_intersect", verbose=False)
    head_h65.eval()
    D_hat_h65 = latent_distance(z_h65)

    # Train head H_h95 on intersection-restricted anchors
    print("[2] Train H_h95 on intersection-restricted anchors...")
    head_h95, z_h95, _ = train_let(feat_k, D_h95, label="H_h95_intersect", verbose=False)
    head_h95.eval()
    D_hat_h95 = latent_distance(z_h95)

    # Self-fit Spearmans
    rho_self_h65, _ = spearmanr(D_hat_h65[triu], D_h65[triu])
    rho_self_h95, _ = spearmanr(D_hat_h95[triu], D_h95[triu])
    print(f"  Self-fit ρ(D̂_h65, D_target_h65) = {rho_self_h65:.3f}")
    print(f"  Self-fit ρ(D̂_h95, D_target_h95) = {rho_self_h95:.3f}")

    # Trustworthiness on each
    trust_h65 = float(trustworthiness(feat_k, z_h65, n_neighbors=15))
    trust_h95 = float(trustworthiness(feat_k, z_h95, n_neighbors=15))
    print(f"  Trustworthiness  H65 head: {trust_h65:.3f}   H95 head: {trust_h95:.3f}")

    # ORTHOGONALITY MEASURE
    rho_overlap, _ = spearmanr(D_hat_h65[triu], D_hat_h95[triu])
    print(f"\n[3] Cross-head latent-distance Spearman: ρ(D̂_h65, D̂_h95) = {rho_overlap:.3f}")
    print(f"   (compare to ground-truth ruler overlap ρ_target = {rho_target:.3f})")
    delta = rho_overlap - rho_target
    print(f"   Δ (head_overlap − target_overlap) = {delta:+.3f}")

    # Paired null: scrambled H95 ruler
    print("\n[4] Paired null — head trained on scrambled H95 ruler...")
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    indicator_shuf = indicator_k[perm]
    D_h95_null = (indicator_shuf[:, None, :] != indicator_shuf[None, :, :]).sum(axis=-1).astype(np.float32)
    head_null, z_null, _ = train_let(feat_k, D_h95_null, label="H_h95_null", verbose=False)
    D_hat_null = latent_distance(z_null)
    rho_null_self, _ = spearmanr(D_hat_null[triu], D_h95_null[triu])
    rho_null_overlap_h65, _ = spearmanr(D_hat_h65[triu], D_hat_null[triu])
    rho_null_overlap_target_h95, _ = spearmanr(D_hat_null[triu], D_h95[triu])
    trust_null = float(trustworthiness(feat_k, z_null, n_neighbors=15))
    print(f"   null self-fit ρ = {rho_null_self:.3f}   trust = {trust_null:.3f}")
    print(f"   ρ(D̂_h65, D̂_null_h95) = {rho_null_overlap_h65:.3f}  (vs real H95 head: {rho_overlap:.3f})")

    # Verdict
    if rho_overlap < 0.45 and rho_self_h65 > 0.7 and rho_self_h95 > 0.5:
        verdict = ("ORTHOGONAL: H65 and H95 occupy substantially independent "
                   "subspaces of the residual stream (both heads fit their "
                   "own rulers but their pairwise distances are weakly correlated).")
    elif rho_overlap > 0.85 * rho_self_h65:
        verdict = ("COLLAPSED: H_h95's pairwise-distance structure is mostly "
                   "explained by H_h65's. The two axes are not independently "
                   "encoded.")
    else:
        verdict = ("PARTIALLY_ORTHOGONAL: H_h65 and H_h95 share some variance "
                   "but each head also captures structure not available from "
                   "the other.")

    out = {
        "scope_note": "Phase 17 cross-axis orthogonality test for H65 × H95.",
        "n_anchors_intersection": int(n),
        "ruler_overlap_target": float(rho_target),
        "self_fit_h65": float(rho_self_h65),
        "self_fit_h95": float(rho_self_h95),
        "trust_h65": trust_h65,
        "trust_h95": trust_h95,
        "head_overlap_rho": float(rho_overlap),
        "head_overlap_minus_target_overlap": float(delta),
        "null_self_fit": float(rho_null_self),
        "null_trust": trust_null,
        "null_overlap_with_h65_head": float(rho_null_overlap_h65),
        "null_overlap_with_target_h95": float(rho_null_overlap_target_h95),
        "verdict": verdict,
        "elapsed_s": float(time.time() - t_phase),
    }
    (REP / "phase17_cross_axis_h65_h95.json").write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 70)
    print(f"VERDICT: {verdict}")
    print(f"  rho_overlap (real H95 head) = {rho_overlap:.3f}")
    print(f"  rho_overlap (null H95 head) = {rho_null_overlap_h65:.3f}")
    print(f"  ground-truth ruler overlap  = {rho_target:.3f}")
    print(f"  Total time: {out['elapsed_s']:.1f}s")
    print(f"  → {REP/'phase17_cross_axis_h65_h95.json'}")


if __name__ == "__main__":
    main()
