"""Phase 10 — multi-stage compaction chain.

Compaction chain on L10H6 (Phase 9 winner):
  full-drift  → compact-V1 (L10H6 single head)  → rank-{8,16,32,64} SVD  → hard sparse

At each stage:
1. Replace the operator
2. Retrain the LET head on the compressed feature
3. Evaluate the four quality gates on internal panel
4. Save metrics + head weights

Outputs:
  artifacts/heads/compact/<variant>.pt
  reports/compaction_chain.csv     per-variant gate scores
  reports/compaction_chain.json    full results
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

PROJ = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN / "scripts"))
from phase5_let_anchor import (  # type: ignore
    LETHead, train_let, random_holdout_corr, grouped_holdout_corr,
    LATENT_DIM, EPOCHS, SEED,
)

ART = RUN / "artifacts"
HEADS = ART / "heads/compact"; HEADS.mkdir(parents=True, exist_ok=True)
REP = RUN / "reports"

DEVICE = "cpu"


def evaluate_variant(
    features: np.ndarray, d_target: np.ndarray, anchor_meta: pd.DataFrame,
    branch_per_anchor: np.ndarray, label: str,
) -> tuple[dict, torch.nn.Module]:
    print(f"\n--- {label} ---")
    head, z, _ = train_let(features, d_target, label=label, verbose=False)
    trust = trustworthiness(features, z, n_neighbors=15)
    rand_corr, _ = random_holdout_corr(features, d_target, n_iters=5, frac=0.2)
    donor_corr, _ = grouped_holdout_corr(
        features, d_target, anchor_meta["donor_id"].astype(str).to_numpy(), f"{label}-donor"
    )
    branch_corr, _ = grouped_holdout_corr(
        features, d_target, branch_per_anchor, f"{label}-branch"
    )
    metrics = {
        "label": label,
        "feature_dim": int(features.shape[1]),
        "trustworthiness": float(trust),
        "random_holdout": float(rand_corr),
        "donor_holdout": float(donor_corr),
        "branch_holdout": float(branch_corr),
    }
    print(f"  trust={trust:.3f}  rand={rand_corr:.3f}  donor={donor_corr:.3f}  branch={branch_corr:.3f}")
    return metrics, head


def block_pool(centroids: np.ndarray, layers: list[int]) -> np.ndarray:
    return centroids[:, layers, :].mean(axis=1)


def build_pooled_drift_feature(centroids, A_early, A_mid, A_late, partition):
    early = [i + 1 for i in partition["early"]]
    mid = [i + 1 for i in partition["mid"]]
    late = [i + 1 for i in partition["late"]]
    x_e = block_pool(centroids, early); x_m = block_pool(centroids, mid); x_l = block_pool(centroids, late)
    y_e = x_e @ A_early; y_m = x_m @ A_mid; y_l = x_l @ A_late
    f = np.concatenate([y_e - y_m, y_m - y_l], axis=1).astype(np.float32)
    return (f - f.mean(0)) / (f.std(0) + 1e-6)


def build_single_head_feature(centroids, A_lh, layer_idx_for_input):
    """Single-head feature: hidden state at layer's input @ A_lh.T → R^head_dim."""
    x = centroids[:, layer_idx_for_input + 1, :]  # post-block index
    f = (x @ A_lh.T).astype(np.float32)
    return (f - f.mean(0)) / (f.std(0) + 1e-6)


def main():
    print("=" * 70)
    print("PHASE 10 — multi-stage compaction chain")
    print("=" * 70)
    t_phase = time.time()

    # ---- inputs
    centroids = np.load(ART / "anchors/centroids_internal.npy")
    d_target = np.load(ART / "anchors/d_target_internal.npy")
    anchor_meta = pd.read_csv(ART / "anchors/anchor_meta_internal.csv")
    op_idx = json.loads((ART / "operators/operator_index.json").read_text())
    op_npz = np.load(ART / "operators/pooled_drift_components.npz")
    A_early = op_npz["A_early"]; A_mid = op_npz["A_mid"]; A_late = op_npz["A_late"]
    partition = op_idx["block_partition"]

    dag = json.loads((RUN / "planning/h65_stage_dag.json").read_text())
    stage_to_branch = {k: v["branch"] for k, v in dag["stage_to_branch"].items()}
    branch_per_anchor = anchor_meta["hema_stage"].map(stage_to_branch).fillna("_unk").to_numpy()

    head_screen = pd.read_csv(REP / "head_layer_screen.csv")
    top1 = head_screen.iloc[0]
    L_top, H_top = int(top1["layer"]), int(top1["head"])
    print(f"\n[1] Compact-V1 single-head winner: L{L_top}H{H_top} (trust={top1['trustworthiness']:.3f})")

    A_top = np.load(ART / f"operators/layer{L_top:02d}_head{H_top}.npy").astype(np.float32)
    print(f"  A shape {A_top.shape}")

    # ---- chain
    chain_metrics = []

    # 0. full-drift reference
    f_full = build_pooled_drift_feature(centroids, A_early, A_mid, A_late, partition)
    m, h = evaluate_variant(f_full, d_target, anchor_meta, branch_per_anchor, "full_drift_reference")
    m["mb_size"] = float((A_early.nbytes + A_mid.nbytes + A_late.nbytes) / 1e6)
    chain_metrics.append(m)
    torch.save(h.state_dict(), HEADS / "full_drift_reference.pt")

    # 1. compact-V1: L10H6 single head
    f_top = build_single_head_feature(centroids, A_top, layer_idx_for_input=L_top)
    m, h = evaluate_variant(f_top, d_target, anchor_meta, branch_per_anchor, f"compact_top1_L{L_top}H{H_top}")
    m["mb_size"] = float(A_top.nbytes / 1e6)
    chain_metrics.append(m)
    torch.save(h.state_dict(), HEADS / f"compact_top1_L{L_top}H{H_top}.pt")

    # 2. truncated SVD surrogates of A_top
    U, S, Vt = np.linalg.svd(A_top, full_matrices=False)
    print(f"\n[2] SVD of A_L{L_top}H{H_top}: |S|={S.shape}, S[:5]={S[:5].round(3)}")
    for r in [8, 16, 32, 64]:
        if r > len(S):
            continue
        A_r = (U[:, :r] * S[:r]) @ Vt[:r, :]
        f_r = build_single_head_feature(centroids, A_r.astype(np.float32), layer_idx_for_input=L_top)
        m, h = evaluate_variant(f_r, d_target, anchor_meta, branch_per_anchor, f"rank{r}_svd")
        m["mb_size"] = float((U[:, :r].nbytes + S[:r].nbytes + Vt[:r, :].nbytes) / 1e6)
        chain_metrics.append(m)
        torch.save(h.state_dict(), HEADS / f"rank{r}_svd.pt")

    # 3. hard-sparse: 16 factors × 60 read/60 write genes per factor
    r_sparse_factors = 16
    n_genes_per_factor = 60
    A_sparse = np.zeros_like(A_top)
    if r_sparse_factors <= len(S):
        for k in range(r_sparse_factors):
            u_k = U[:, k]; v_k = Vt[k, :]; s_k = S[k]
            top_u = np.argsort(np.abs(u_k))[-n_genes_per_factor:]
            top_v = np.argsort(np.abs(v_k))[-n_genes_per_factor:]
            u_sparse = np.zeros_like(u_k); u_sparse[top_u] = u_k[top_u]
            v_sparse = np.zeros_like(v_k); v_sparse[top_v] = v_k[top_v]
            A_sparse += s_k * np.outer(u_sparse, v_sparse)
        f_s = build_single_head_feature(centroids, A_sparse.astype(np.float32), layer_idx_for_input=L_top)
        m, h = evaluate_variant(f_s, d_target, anchor_meta, branch_per_anchor,
                                f"hard_sparse_{r_sparse_factors}f_{n_genes_per_factor}g")
        # Effective storage: r_sparse_factors × (head_dim_sparse + hidden_sparse) entries
        eff_entries = r_sparse_factors * (n_genes_per_factor + n_genes_per_factor) + r_sparse_factors  # +S
        m["mb_size"] = float(eff_entries * 4 / 1e6)
        chain_metrics.append(m)
        torch.save(h.state_dict(), HEADS / f"hard_sparse_{r_sparse_factors}f_{n_genes_per_factor}g.pt")

    # ---- persist
    df = pd.DataFrame(chain_metrics)
    df.to_csv(REP / "compaction_chain.csv", index=False)
    (REP / "compaction_chain.json").write_text(json.dumps(chain_metrics, indent=2))

    print("\n" + "=" * 70)
    print("COMPACTION CHAIN SUMMARY")
    print("=" * 70)
    print(df[["label", "feature_dim", "mb_size", "trustworthiness", "branch_holdout", "donor_holdout"]].to_string(index=False))

    print(f"\nPhase 10 complete in {time.time()-t_phase:.1f}s")


if __name__ == "__main__":
    main()
