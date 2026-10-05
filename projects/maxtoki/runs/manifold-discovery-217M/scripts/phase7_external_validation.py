"""Phase 7 — frozen-head external validation on a held-out panel.

Apply the frozen Phase-5 anchor LET head (trained on internal) to a held-out
panel and evaluate the four quality gates. Use this for:
  - strict non-overlap external panel  (expected to PASS)
  - lung negative control panel        (expected to FAIL — especially trust + branch)

Usage:
  python phase7_external_validation.py <panel>

  where <panel> is one of: external, lung_control, zeroshot

Outputs:
  reports/external_validation_<panel>.json
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
    LETHead, train_let, build_pooled_drift, SEED,
)

ART = RUN / "artifacts"; REP = RUN / "reports"
DEVICE = "cpu"


def main():
    if len(sys.argv) != 2:
        print("Usage: phase7_external_validation.py <panel: external|lung_control|zeroshot>")
        sys.exit(1)
    panel = sys.argv[1]
    assert panel in {"external", "lung_control", "zeroshot", "lung_nonhema"}

    print("=" * 70)
    print(f"PHASE 7 — frozen-head external validation on panel='{panel}'")
    print("=" * 70)
    t_phase = time.time()

    # ---- internal-trained head (re-train deterministic via SEED)
    centroids_int = np.load(ART / "anchors/centroids_internal.npy")
    d_target_int = np.load(ART / "anchors/d_target_internal.npy")
    op_idx = json.loads((ART / "operators/operator_index.json").read_text())
    op_npz = np.load(ART / "operators/pooled_drift_components.npz")
    A_e, A_m, A_l = op_npz["A_early"], op_npz["A_mid"], op_npz["A_late"]
    partition = op_idx["block_partition"]
    f_int_raw = build_pooled_drift(centroids_int, A_e, A_m, A_l, partition)
    feat_mean = f_int_raw.mean(0); feat_std = f_int_raw.std(0) + 1e-6
    f_int = (f_int_raw - feat_mean) / feat_std

    print(f"[1] Re-training anchor LET head on internal panel (deterministic)...")
    head, _, _ = train_let(f_int, d_target_int, label="internal_anchor", verbose=False)
    head.eval()

    # ---- held-out panel
    print(f"\n[2] Loading {panel} panel...")
    centroids = np.load(ART / f"anchors/centroids_{panel}.npy")
    anchor_meta = pd.read_csv(ART / f"anchors/anchor_meta_{panel}.csv")
    print(f"  centroids: {centroids.shape}")

    # For lung_control: build the H65 ruler over its anchors using their stage labels
    # (lung_control stage labels were derived in Phase 1A and may include non-hematopoietic
    # stages → 99 sentinel for unmappable, but here we use whatever the script saved).
    if panel in {"external", "zeroshot"}:
        d_target = np.load(ART / f"anchors/d_target_{panel}.npy")
    else:
        # lung_control: build ruler from its anchor metadata (same stage DAG)
        from phase1bc_hidden_states_and_centroids import build_ruler  # type: ignore
        stages = anchor_meta["hema_stage"].astype(str).tolist()
        d_target = build_ruler(stages)
        np.save(ART / f"anchors/d_target_{panel}.npy", d_target)

    f_raw = build_pooled_drift(centroids, A_e, A_m, A_l, partition)
    f = (f_raw - feat_mean) / feat_std

    print(f"\n[3] Frozen forward pass...")
    with torch.no_grad():
        x = torch.from_numpy(f).float()
        z = head(x)[0].numpy()

    print(f"\n[4] Trustworthiness, holdouts on {panel}...")
    trust = trustworthiness(f, z, n_neighbors=15)
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    cos = np.clip(zn @ zn.T, -1 + 1e-7, 1 - 1e-7)
    d_hat = np.arccos(cos)
    triu = np.triu_indices(len(d_target), k=1)
    rho_global, _ = spearmanr(d_hat[triu], d_target[triu])

    rng = np.random.default_rng(SEED)
    rand_corrs, donor_corrs, branch_corrs = [], [], []
    n = len(d_target)

    for _ in range(20):
        idx = rng.permutation(n); n_test = max(2, int(round(n * 0.2)))
        test = sorted(idx[:n_test])
        zt = z[test]; zn_t = zt / (np.linalg.norm(zt, axis=1, keepdims=True) + 1e-9)
        cos_t = np.clip(zn_t @ zn_t.T, -1 + 1e-7, 1 - 1e-7)
        D_t = d_target[np.ix_(test, test)]
        triu_t = np.triu_indices(len(test), k=1)
        rho, _ = spearmanr(np.arccos(cos_t)[triu_t], D_t[triu_t])
        if not np.isnan(rho):
            rand_corrs.append(float(rho))

    donors = anchor_meta["donor_id"].astype(str).to_numpy()
    for d in pd.Series(donors).unique():
        mask = donors == d
        if mask.sum() < 3:
            continue
        idx_test = np.where(mask)[0]
        zt = z[idx_test]; zn_t = zt / (np.linalg.norm(zt, axis=1, keepdims=True) + 1e-9)
        cos_t = np.clip(zn_t @ zn_t.T, -1 + 1e-7, 1 - 1e-7)
        D_t = d_target[np.ix_(idx_test, idx_test)]
        triu_t = np.triu_indices(len(idx_test), k=1)
        if len(triu_t[0]) == 0:
            continue
        rho, _ = spearmanr(np.arccos(cos_t)[triu_t], D_t[triu_t])
        if not np.isnan(rho):
            donor_corrs.append(float(rho))

    dag = json.loads((RUN / "planning/h65_stage_dag.json").read_text())
    stage_to_branch = {k: v["branch"] for k, v in dag["stage_to_branch"].items()}
    branches = anchor_meta["hema_stage"].map(stage_to_branch).fillna("_unk").to_numpy()
    for b in pd.Series(branches).unique():
        if b == "_unk":
            continue
        mask = branches == b
        if mask.sum() < 3:
            continue
        idx_test = np.where(mask)[0]
        zt = z[idx_test]; zn_t = zt / (np.linalg.norm(zt, axis=1, keepdims=True) + 1e-9)
        cos_t = np.clip(zn_t @ zn_t.T, -1 + 1e-7, 1 - 1e-7)
        D_t = d_target[np.ix_(idx_test, idx_test)]
        triu_t = np.triu_indices(len(idx_test), k=1)
        if len(triu_t[0]) == 0:
            continue
        rho, _ = spearmanr(np.arccos(cos_t)[triu_t], D_t[triu_t])
        if not np.isnan(rho):
            branch_corrs.append(float(rho))

    rand_mean = float(np.mean(rand_corrs)) if rand_corrs else float("nan")
    donor_mean = float(np.mean(donor_corrs)) if donor_corrs else float("nan")
    branch_mean = float(np.mean(branch_corrs)) if branch_corrs else float("nan")

    gates = json.loads((REP / "quality_gates_spec.json").read_text())["gates"]
    passes = (
        trust >= gates["trustworthiness_min"]
        and (rand_mean >= gates["random_holdout_correlation_min"])
        and (np.isnan(donor_mean) or donor_mean >= gates["donor_holdout_correlation_min"])
        and (np.isnan(branch_mean) or branch_mean >= gates["clade_branch_holdout_correlation_min"])
    )
    expected_to_fail = panel in {"lung_control", "lung_nonhema"}
    if expected_to_fail:
        diagnostic = "GOOD: negative control fails" if not passes else "BAD: negative control passed (gates too loose or pipeline confound)"
    else:
        diagnostic = "POSITIVE: external panel passes all gates" if passes else "FAIL: external panel does not pass all gates"

    report = {
        "branch": "H65_anchor_LET10D",
        "panel": panel,
        "expected_to_fail": expected_to_fail,
        "frozen_from": "internal_anchor",
        "n_anchors": int(len(d_target)),
        "n_donors": int(anchor_meta["donor_id"].nunique()),
        "n_tissues": int(anchor_meta["tissue"].nunique()),
        "global_correlation": float(rho_global),
        "trustworthiness": float(trust),
        "random_holdout": rand_mean,
        "donor_holdout": donor_mean,
        "branch_holdout": branch_mean,
        "all_gates_pass": bool(passes),
        "diagnostic": diagnostic,
        "elapsed_s": float(time.time() - t_phase),
    }
    (REP / f"external_validation_{panel}.json").write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 70)
    print(f"VERDICT: {diagnostic}")
    print(f"  trust={trust:.3f}  rand={rand_mean:.3f}  donor={donor_mean:.3f}  branch={branch_mean:.3f}")
    print(f"  global ρ on full panel: {rho_global:.3f}")
    print(f"Total time: {time.time()-t_phase:.1f}s")


if __name__ == "__main__":
    main()
