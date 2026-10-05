"""Phase 8 — frozen-head zero-shot transfer to multi-donor panel.

Train-once, transfer-many: take the Phase-5 anchor-trained LET head trained
on the internal panel only, apply it (without retraining) to a separate
multi-donor zero-shot panel, and evaluate the same four quality gates on
the held-out panel.

Outputs:
  reports/zeroshot_transfer_anchor_head.json
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
    LETHead, train_let, build_pooled_drift, random_holdout_corr, grouped_holdout_corr,
    SEED, LATENT_DIM,
)

ART = RUN / "artifacts"; REP = RUN / "reports"
DEVICE = "cpu"


def main():
    print("=" * 70)
    print("PHASE 8 — frozen-head zero-shot transfer to zeroshot panel")
    print("=" * 70)
    t_phase = time.time()

    # ---- internal-trained head (re-train from saved weights, since Phase 5 saved state_dict)
    centroids_int = np.load(ART / "anchors/centroids_internal.npy")
    d_target_int = np.load(ART / "anchors/d_target_internal.npy")
    op_idx = json.loads((ART / "operators/operator_index.json").read_text())
    op_npz = np.load(ART / "operators/pooled_drift_components.npz")
    A_e, A_m, A_l = op_npz["A_early"], op_npz["A_mid"], op_npz["A_late"]
    partition = op_idx["block_partition"]

    # Build internal pooled-drift feature; remember mean/std to apply to external
    f_int_raw = build_pooled_drift(centroids_int, A_e, A_m, A_l, partition)
    feat_mean = f_int_raw.mean(0); feat_std = f_int_raw.std(0) + 1e-6
    f_int = (f_int_raw - feat_mean) / feat_std

    # Re-train the anchor LET head on internal (this matches Phase 5; deterministic via SEED)
    print("[1] Re-training anchor LET head on internal panel (deterministic seed)...")
    head, z_int, _ = train_let(f_int, d_target_int, label="internal_anchor", verbose=False)
    head.eval()

    # ---- apply to zeroshot panel (FROZEN head)
    print("\n[2] Loading zeroshot centroids and ruler...")
    centroids_zs = np.load(ART / "anchors/centroids_zeroshot.npy")
    d_target_zs = np.load(ART / "anchors/d_target_zeroshot.npy")
    anchor_meta_zs = pd.read_csv(ART / "anchors/anchor_meta_zeroshot.csv")
    print(f"  centroids: {centroids_zs.shape}")

    f_zs_raw = build_pooled_drift(centroids_zs, A_e, A_m, A_l, partition)
    f_zs = (f_zs_raw - feat_mean) / feat_std  # use INTERNAL mean/std

    print("\n[3] Frozen forward pass (no retraining)...")
    with torch.no_grad():
        x = torch.from_numpy(f_zs).float()
        z_zs = head(x)[0].numpy()

    print("\n[4] Trustworthiness, random, donor, branch holdouts on zeroshot panel...")
    trust = trustworthiness(f_zs, z_zs, n_neighbors=15)

    # Use the standard Spearman-on-arccos for direct evaluation since the head is frozen
    zn = z_zs / (np.linalg.norm(z_zs, axis=1, keepdims=True) + 1e-9)
    cos = zn @ zn.T
    cos = np.clip(cos, -1 + 1e-7, 1 - 1e-7)
    d_hat = np.arccos(cos)
    triu = np.triu_indices(len(d_target_zs), k=1)
    rho_global, _ = spearmanr(d_hat[triu], d_target_zs[triu])

    # For the holdout-style metrics we still hold out subsets of zeroshot anchors
    # but DO NOT retrain — we just measure correlation on the held-out subset
    rng = np.random.default_rng(SEED)
    rand_corrs, donor_corrs, branch_corrs = [], [], []
    n_zs = len(d_target_zs)
    for _ in range(20):
        idx = rng.permutation(n_zs)
        n_test = max(2, int(round(n_zs * 0.2)))
        test = sorted(idx[:n_test])
        zt = z_zs[test]; zn_t = zt / (np.linalg.norm(zt, axis=1, keepdims=True) + 1e-9)
        cos_t = np.clip(zn_t @ zn_t.T, -1 + 1e-7, 1 - 1e-7)
        d_hat_t = np.arccos(cos_t)
        D_t = d_target_zs[np.ix_(test, test)]
        triu_t = np.triu_indices(len(test), k=1)
        rho, _ = spearmanr(d_hat_t[triu_t], D_t[triu_t])
        rand_corrs.append(float(rho))

    donors = anchor_meta_zs["donor_id"].astype(str).to_numpy()
    for d in pd.Series(donors).unique():
        mask = donors == d
        if mask.sum() < 3:
            continue
        idx_test = np.where(mask)[0]
        zt = z_zs[idx_test]; zn_t = zt / (np.linalg.norm(zt, axis=1, keepdims=True) + 1e-9)
        cos_t = np.clip(zn_t @ zn_t.T, -1 + 1e-7, 1 - 1e-7)
        d_hat_t = np.arccos(cos_t)
        D_t = d_target_zs[np.ix_(idx_test, idx_test)]
        triu_t = np.triu_indices(len(idx_test), k=1)
        if len(triu_t[0]) == 0:
            continue
        rho, _ = spearmanr(d_hat_t[triu_t], D_t[triu_t])
        if not np.isnan(rho):
            donor_corrs.append(float(rho))

    dag = json.loads((RUN / "planning/h65_stage_dag.json").read_text())
    stage_to_branch = {k: v["branch"] for k, v in dag["stage_to_branch"].items()}
    branches = anchor_meta_zs["hema_stage"].map(stage_to_branch).fillna("_unk").to_numpy()
    for b in pd.Series(branches).unique():
        if b == "_unk":
            continue
        mask = branches == b
        if mask.sum() < 3:
            continue
        idx_test = np.where(mask)[0]
        zt = z_zs[idx_test]; zn_t = zt / (np.linalg.norm(zt, axis=1, keepdims=True) + 1e-9)
        cos_t = np.clip(zn_t @ zn_t.T, -1 + 1e-7, 1 - 1e-7)
        d_hat_t = np.arccos(cos_t)
        D_t = d_target_zs[np.ix_(idx_test, idx_test)]
        triu_t = np.triu_indices(len(idx_test), k=1)
        if len(triu_t[0]) == 0:
            continue
        rho, _ = spearmanr(d_hat_t[triu_t], D_t[triu_t])
        if not np.isnan(rho):
            branch_corrs.append(float(rho))

    rand_mean = float(np.mean(rand_corrs))
    donor_mean = float(np.mean(donor_corrs)) if donor_corrs else float("nan")
    branch_mean = float(np.mean(branch_corrs)) if branch_corrs else float("nan")

    gates = json.loads((REP / "quality_gates_spec.json").read_text())["gates"]
    passes = (
        trust >= gates["trustworthiness_min"]
        and rand_mean >= gates["random_holdout_correlation_min"]
        and donor_mean >= gates["donor_holdout_correlation_min"]
        and branch_mean >= gates["clade_branch_holdout_correlation_min"]
    )
    report = {
        "branch": "H65_anchor_LET10D",
        "panel": "zeroshot_multi_donor",
        "frozen_from": "internal_anchor",
        "n_anchors": int(len(d_target_zs)),
        "n_donors": int(anchor_meta_zs["donor_id"].nunique()),
        "n_tissues": int(anchor_meta_zs["tissue"].nunique()),
        "global_correlation": float(rho_global),
        "trustworthiness": float(trust),
        "random_holdout": rand_mean,
        "donor_holdout": donor_mean,
        "branch_holdout": branch_mean,
        "all_gates_pass": bool(passes),
        "verdict": "POSITIVE: frozen head transfers to zero-shot panel" if passes else "FAIL: zero-shot transfer does not pass all gates",
        "elapsed_s": float(time.time() - t_phase),
    }
    (REP / "zeroshot_transfer_anchor_head.json").write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 70)
    print(f"VERDICT: {report['verdict']}")
    print(f"  trust={trust:.3f}  rand={rand_mean:.3f}  donor={donor_mean:.3f}  branch={branch_mean:.3f}")
    print(f"  global ρ on full zeroshot panel: {rho_global:.3f}")
    print(f"Total time: {time.time()-t_phase:.1f}s")


if __name__ == "__main__":
    main()
