"""Phase 9 — head/layer attribution scan: 88 single-head LET-10D runs.

For each of the 88 (layer, head) units, train a single-head LET-10D variant on
the same internal anchors and the same H65 ruler. Score each variant by:
- trustworthiness on internal panel
- internal random holdout Spearman
- branch holdout Spearman (the most discriminative gate from Phase 5)

The single-head feature for unit (l, h) is `x_l @ A_{l,h}^T` ∈ R^{154}, where
x_l is the anchor centroid's hidden state at layer-input l (i.e. centroid index
l) — the residual stream feeding into layer l.

The output is a ranked CSV that drives Phase 10 compaction (the top-1 head
becomes the compact-V1 single-head operator).
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
sys.path.insert(0, str(PROJ / "scripts"))  # not strictly needed
sys.path.insert(0, str(PROJ / "setup"))

# Reuse Phase 5 helpers
sys.path.insert(0, str(RUN / "scripts"))
from phase5_let_anchor import LETHead, train_let, random_holdout_corr, grouped_holdout_corr  # type: ignore
from phase5_let_anchor import LATENT_DIM, EPOCHS, SEED  # type: ignore

ART = RUN / "artifacts"
REP = RUN / "reports"
HEADS_OUT = ART / "heads/per_head_scan"; HEADS_OUT.mkdir(parents=True, exist_ok=True)

DEVICE = "cpu"


def main():
    print("=" * 70)
    print("PHASE 9 — head/layer attribution scan (88 single-head LET-10D)")
    print("=" * 70)
    t_phase = time.time()

    centroids = np.load(ART / "anchors/centroids_internal.npy")  # (290, 12, 1232)
    d_target = np.load(ART / "anchors/d_target_internal.npy")
    anchor_meta = pd.read_csv(ART / "anchors/anchor_meta_internal.csv")
    op_idx = json.loads((ART / "operators/operator_index.json").read_text())

    n_layers = op_idx["n_layers"]
    n_heads = op_idx["n_heads"]
    head_dim = op_idx["head_dim"]
    print(f"  centroids {centroids.shape}  layers={n_layers} heads={n_heads} head_dim={head_dim}")

    dag = json.loads((RUN / "planning/h65_stage_dag.json").read_text())
    stage_to_branch = {k: v["branch"] for k, v in dag["stage_to_branch"].items()}
    branch_per_anchor = anchor_meta["hema_stage"].map(stage_to_branch).fillna("_unk").to_numpy()

    rows = []
    for li in range(n_layers):
        x_l = centroids[:, li + 1, :]  # input to layer li in residual space (post layer li-1, pre layer li)
        # actually layer-input is centroid index `li` (post-block li-1) — adjusting: we treat
        # centroid index `li+1` as "post-block li" (output of layer li). For "input to layer li"
        # use centroid index `li` (post layer li-1). We'll use `li+1` as layer-output for consistency
        # with Phase 5 (which uses the layer-output side).
        for hi in range(n_heads):
            A = np.load(ART / f"operators/layer{li:02d}_head{hi}.npy").astype(np.float32)  # (154, 1232)
            # Apply: x_l @ A.T  (note: A is (head_dim, hidden), so A.T is (hidden, head_dim))
            features = (x_l @ A.T).astype(np.float32)
            # Standardise per-feature
            features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)
            # Train LET-10D on full panel, evaluate gates
            t0 = time.time()
            head, z, _ = train_let(features, d_target, label=f"L{li}H{hi}", verbose=False)
            trust = trustworthiness(features, z, n_neighbors=15)
            rand_corr, _ = random_holdout_corr(features, d_target, n_iters=3, frac=0.2)  # 3 iters for speed
            branch_corr, _ = grouped_holdout_corr(features, d_target, branch_per_anchor, f"L{li}H{hi}-branch")
            elapsed = time.time() - t0
            rows.append({
                "layer": li,
                "head": hi,
                "feature_dim": features.shape[1],
                "trustworthiness": float(trust),
                "random_holdout": float(rand_corr),
                "branch_holdout": float(branch_corr),
                "elapsed_s": float(elapsed),
            })
            print(f"  L{li}H{hi}: trust={trust:.3f}  rand={rand_corr:.3f}  branch={branch_corr:.3f}  ({elapsed:.1f}s)")
            # Save the head
            torch.save(head.state_dict(), HEADS_OUT / f"let_L{li:02d}H{hi}.pt")

    df = pd.DataFrame(rows)
    df["score"] = (
        0.5 * df["trustworthiness"]
        + 0.25 * df["random_holdout"]
        + 0.25 * df["branch_holdout"]
    )
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df.to_csv(REP / "head_layer_screen.csv", index=False)

    top = df.head(10)
    print("\n=== TOP 10 (by composite score) ===")
    for _, r in top.iterrows():
        print(f"  L{int(r['layer'])}H{int(r['head'])}: trust={r['trustworthiness']:.3f}  "
              f"rand={r['random_holdout']:.3f}  branch={r['branch_holdout']:.3f}  score={r['score']:.3f}")

    summary = {
        "n_total_heads": int(len(df)),
        "n_passing_trust": int((df["trustworthiness"] >= 0.80).sum()),
        "n_passing_branch": int((df["branch_holdout"] >= 0.20).sum()),
        "top1": {"layer": int(top.iloc[0]["layer"]), "head": int(top.iloc[0]["head"]),
                  "trustworthiness": float(top.iloc[0]["trustworthiness"]),
                  "branch_holdout": float(top.iloc[0]["branch_holdout"])},
        "elapsed_s": float(time.time() - t_phase),
    }
    (REP / "head_layer_screen_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nPhase 9 complete in {time.time()-t_phase:.1f}s")
    print(f"Top-1 head: L{summary['top1']['layer']}H{summary['top1']['head']}")


if __name__ == "__main__":
    main()
