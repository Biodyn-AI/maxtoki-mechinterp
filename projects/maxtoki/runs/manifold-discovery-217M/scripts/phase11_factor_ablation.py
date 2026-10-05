"""Phase 11 — leave-one-factor-out ablation on rank-64 SVD compact operator.

Procedure (per pipeline §6 Phase 11):
1. SVD-decompose A_L10H6 → keep top-64 rank-1 factors u_k s_k v_k^T.
2. Train fixed Stage-2 (LET head) + Stage-3 (linear probes) on the rank-64
   surrogate's features.
3. For each factor k in 0..63: zero it out → recompute features (using the
   training-set standardisation) → push through the FROZEN LET head and
   FROZEN probes → measure drop in each of 5 endpoints.
4. Rank factors by total pooled impact.
5. Test core sufficiency: keep only top-N core factors (zero all others) →
   re-evaluate (still using frozen heads).
6. Save top read/write genes (=hidden-feature loadings, used for downstream
   GO enrichment in a follow-up).

Outputs:
  reports/factor_ablation.json
  reports/factor_ablation_per_endpoint.csv
  reports/factor_gene_loadings.csv
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from scipy.stats import spearmanr

PROJ = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN / "scripts"))
from phase5_let_anchor import LETHead, train_let, SEED  # type: ignore

ART = RUN / "artifacts"
REP = RUN / "reports"
DEVICE = "cpu"


def build_centroid_input(centroids, layer_idx_for_input):
    return centroids[:, layer_idx_for_input + 1, :]  # post-block of layer L


def project_through_operator(x, A):
    """Apply per-head operator: feature = x @ A.T → R^head_dim."""
    return (x @ A.T).astype(np.float32)


def standardise(X, mean, std):
    return (X - mean) / std


def fit_linear_classifier_full(X, y, n_classes):
    model = nn.Linear(X.shape[1], n_classes)
    nn.init.normal_(model.weight, std=0.01)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).long()
    for _ in range(800):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(Xt), yt)
        loss.backward(); opt.step()
    return model


def fit_linear_regressor_full(X, y):
    model = nn.Linear(X.shape[1], 1)
    nn.init.normal_(model.weight, std=0.01)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).float()
    for _ in range(800):
        opt.zero_grad()
        loss = ((model(Xt).squeeze(-1) - yt) ** 2).mean()
        loss.backward(); opt.step()
    return model


def evaluate_endpoints(z_intact, z_test, anchor_meta, branch_y, stage_y, depths,
                        cd_mask, cd_y, mm_mask, mm_y,
                        n_branch_classes, n_stage_classes,
                        probes):
    """Apply frozen probes to z_test; report endpoint scores."""
    Xt = torch.from_numpy(z_test).float()
    out = {}
    with torch.no_grad():
        # branch
        logits = probes["branch"](Xt)
        preds = logits.argmax(-1).numpy()
        out["branch_balanced_acc"] = balanced_accuracy_score(branch_y, preds)
        out["branch_macro_f1"] = f1_score(branch_y, preds, average="macro", zero_division=0)
        # stage
        logits = probes["stage"](Xt)
        preds = logits.argmax(-1).numpy()
        out["stage_balanced_acc"] = balanced_accuracy_score(stage_y, preds)
        out["stage_macro_f1"] = f1_score(stage_y, preds, average="macro", zero_division=0)
        # cd4/cd8
        logits = probes["cd"](Xt[cd_mask])
        probs = torch.softmax(logits, -1).numpy()
        try:
            out["cd_auroc"] = roc_auc_score(cd_y[cd_mask], probs[:, 1])
        except ValueError:
            out["cd_auroc"] = float("nan")
        # mono/macro
        logits = probes["mm"](Xt[mm_mask])
        probs = torch.softmax(logits, -1).numpy()
        try:
            out["mm_auroc"] = roc_auc_score(mm_y[mm_mask], probs[:, 1])
        except ValueError:
            out["mm_auroc"] = float("nan")
        # pseudotime
        pred = probes["pt"](Xt).squeeze(-1).numpy()
        rho, _ = spearmanr(pred, depths)
        out["pseudotime_spearman"] = float(rho if not np.isnan(rho) else 0.0)
    return out


def main():
    print("=" * 70)
    print("PHASE 11 — leave-one-factor-out ablation on rank-64 SVD")
    print("=" * 70)
    t_phase = time.time()

    centroids = np.load(ART / "anchors/centroids_internal.npy")
    d_target = np.load(ART / "anchors/d_target_internal.npy")
    anchor_meta = pd.read_csv(ART / "anchors/anchor_meta_internal.csv")
    op_idx = json.loads((ART / "operators/operator_index.json").read_text())

    head_screen = pd.read_csv(REP / "head_layer_screen.csv")
    L_top, H_top = int(head_screen.iloc[0]["layer"]), int(head_screen.iloc[0]["head"])
    print(f"  using compact-V1 head L{L_top}H{H_top}")

    A_top = np.load(ART / f"operators/layer{L_top:02d}_head{H_top}.npy").astype(np.float32)
    U, S, Vt = np.linalg.svd(A_top, full_matrices=False)
    R = 64
    A_64 = (U[:, :R] * S[:R]) @ Vt[:R, :]
    print(f"  SVD: |S|={len(S)}, top-{R} factors retained")

    # Build intact features (rank-64) and standardisation params
    x_layer = build_centroid_input(centroids, L_top)
    f_intact_raw = project_through_operator(x_layer, A_64.astype(np.float32))
    feat_mean = f_intact_raw.mean(0); feat_std = f_intact_raw.std(0) + 1e-6
    f_intact = standardise(f_intact_raw, feat_mean, feat_std)

    # Train Stage-2 LET head on intact features
    print("\n[1] Training fixed Stage-2 LET head on rank-64 intact features...")
    head, z_intact, _ = train_let(f_intact, d_target, label="rank64_fixed", verbose=False)
    head.eval()

    # Labels and groups
    dag = json.loads((RUN / "planning/h65_stage_dag.json").read_text())
    stage_to_branch = {k: v["branch"] for k, v in dag["stage_to_branch"].items()}
    stage_to_depth = {k: v["depth"] for k, v in dag["stage_to_branch"].items()}
    stages = anchor_meta["hema_stage"].astype(str).to_numpy()
    branches = np.array([stage_to_branch.get(s, "_unk") for s in stages])
    depths = np.array([stage_to_depth.get(s, -1) for s in stages], dtype=np.float32)
    branch_set = sorted(set(branches.tolist()) - {"_unk"})
    stage_set = sorted(set(stages.tolist()))
    branch_to_idx = {b: i for i, b in enumerate(branch_set)}
    stage_to_idx = {s: i for i, s in enumerate(stage_set)}
    branch_y = np.array([branch_to_idx.get(b, -1) for b in branches])
    stage_y = np.array([stage_to_idx[s] for s in stages])
    is_cd4 = np.array([s in {"Naive_CD4", "Memory_CD4", "Treg"} for s in stages])
    is_cd8 = np.array([s in {"Naive_CD8", "Memory_CD8"} for s in stages])
    cd_mask = is_cd4 | is_cd8; cd_y = is_cd8.astype(np.int64)
    is_mono = np.array([s in {"Cl_Mono", "NC_Mono"} for s in stages])
    is_macro = np.array([s == "Macrophage" for s in stages])
    mm_mask = is_mono | is_macro; mm_y = is_macro.astype(np.int64)

    # Train Stage-3 probes on z_intact (no holdout — we measure ablation deltas vs intact baseline)
    print("\n[2] Training fixed Stage-3 probes on z_intact...")
    probes = {
        "branch": fit_linear_classifier_full(z_intact[branch_y >= 0], branch_y[branch_y >= 0], len(branch_set)),
        "stage": fit_linear_classifier_full(z_intact, stage_y, len(stage_set)),
        "cd": fit_linear_classifier_full(z_intact[cd_mask], cd_y[cd_mask], 2),
        "mm": fit_linear_classifier_full(z_intact[mm_mask], mm_y[mm_mask], 2),
        "pt": fit_linear_regressor_full(z_intact, depths),
    }
    for p in probes.values():
        p.eval()

    # For evaluation we use ALL anchors (since we're measuring necessity, not generalisation)
    intact_metrics = evaluate_endpoints(
        z_intact, z_intact, anchor_meta,
        branch_y[branch_y >= 0], stage_y, depths,
        cd_mask, cd_y, mm_mask, mm_y,
        len(branch_set), len(stage_set), probes,
    )
    print("\n[3] Intact baseline (rank-64, no factors removed):")
    for k, v in intact_metrics.items():
        print(f"  {k:30s}  {v:.4f}")

    # Helper: project frozen LET head on a feature
    @torch.no_grad()
    def let_z(features):
        ft = torch.from_numpy(features).float()
        return head(ft)[0].numpy()

    # Filter z_intact to branches we have labels for (for branch endpoint)
    branch_eval_idx = np.where(branch_y >= 0)[0]
    branch_y_eval = branch_y[branch_eval_idx]

    print("\n[4] Leave-one-factor-out ablation (64 factors)...")
    rows = []
    for k in range(R):
        s_k = S[k]; u_k = U[:, k]; v_k = Vt[k, :]
        A_minus = A_64 - s_k * np.outer(u_k, v_k)
        f_raw = project_through_operator(x_layer, A_minus.astype(np.float32))
        f = standardise(f_raw, feat_mean, feat_std)
        z = let_z(f)
        m = evaluate_endpoints(
            z_intact, z, anchor_meta,
            branch_y_eval, stage_y, depths,
            cd_mask, cd_y, mm_mask, mm_y,
            len(branch_set), len(stage_set), probes,
        )
        # Re-evaluate branch using the labelled subset
        Xt = torch.from_numpy(z[branch_eval_idx]).float()
        with torch.no_grad():
            preds = probes["branch"](Xt).argmax(-1).numpy()
        m["branch_balanced_acc"] = float(balanced_accuracy_score(branch_y_eval, preds))
        m["branch_macro_f1"] = float(f1_score(branch_y_eval, preds, average="macro", zero_division=0))
        # Pooled drop relative to intact (sum of squared drops, equal weight)
        drop = sum(max(0.0, intact_metrics[k_e] - m[k_e]) for k_e in intact_metrics if not np.isnan(m[k_e]))
        rows.append({"factor": int(k), "singular_value": float(s_k), "pooled_drop": float(drop), **m})
    df = pd.DataFrame(rows).sort_values("pooled_drop", ascending=False).reset_index(drop=True)
    df.to_csv(REP / "factor_ablation_per_endpoint.csv", index=False)

    print("\n[5] Top-10 factors by pooled drop:")
    print(df.head(10)[["factor", "singular_value", "pooled_drop", "branch_balanced_acc", "stage_balanced_acc", "cd_auroc", "mm_auroc", "pseudotime_spearman"]].to_string(index=False))

    # Cumulative impact concentration
    total_drop = df["pooled_drop"].sum()
    df["pct_of_total"] = df["pooled_drop"] / max(total_drop, 1e-9)
    cum = df["pct_of_total"].cumsum().to_numpy()
    n_for_50 = int((cum < 0.50).sum() + 1)
    n_for_66 = int((cum < 0.66).sum() + 1)
    n_for_80 = int((cum < 0.80).sum() + 1)

    # Core sufficiency — keep top 4
    core_factors = df.head(4)["factor"].astype(int).tolist()
    print(f"\n[6] Core-sufficiency test: only factors {core_factors} retained")
    A_core = np.zeros_like(A_64)
    for k in core_factors:
        A_core += S[k] * np.outer(U[:, k], Vt[k, :])
    f_core = standardise(project_through_operator(x_layer, A_core.astype(np.float32)), feat_mean, feat_std)
    z_core = let_z(f_core)
    Xt = torch.from_numpy(z_core[branch_eval_idx]).float()
    with torch.no_grad():
        preds_branch_core = probes["branch"](Xt).argmax(-1).numpy()
    core_metrics = evaluate_endpoints(
        z_intact, z_core, anchor_meta,
        branch_y_eval, stage_y, depths,
        cd_mask, cd_y, mm_mask, mm_y,
        len(branch_set), len(stage_set), probes,
    )
    core_metrics["branch_balanced_acc"] = float(balanced_accuracy_score(branch_y_eval, preds_branch_core))
    print("  core endpoints:")
    for k_e, v in core_metrics.items():
        print(f"    {k_e:30s}  {v:.4f}  (intact {intact_metrics[k_e]:.4f})")

    # Gene loadings for the 4 core factors
    print("\n[7] Saving gene loadings for core factors (top 60 read/write per factor)...")
    rows_g = []
    for rank, k in enumerate(core_factors):
        u_k = U[:, k]; v_k = Vt[k, :]
        # u_k indexes head_dim (154) — these are "value-vector positions", not genes
        # v_k indexes hidden_size (1232) — these are "residual hidden positions"
        # We do not have a direct gene mapping for either dimension on MaxToki (the o_proj
        # is in residual space, not gene space). We save top-loaded indices for downstream
        # interpretation; mapping to genes requires SAE decoders or unembedding analysis.
        top_u = np.argsort(np.abs(u_k))[-60:][::-1]
        top_v = np.argsort(np.abs(v_k))[-60:][::-1]
        rows_g.append({
            "rank_in_core": rank,
            "factor_index": int(k),
            "singular_value": float(S[k]),
            "top_u_indices": ",".join(map(str, top_u.tolist())),
            "top_u_values": ",".join(f"{u_k[i]:.4f}" for i in top_u),
            "top_v_indices": ",".join(map(str, top_v.tolist())),
            "top_v_values": ",".join(f"{v_k[i]:.4f}" for i in top_v),
        })
    pd.DataFrame(rows_g).to_csv(REP / "factor_gene_loadings.csv", index=False)

    summary = {
        "compact_head": f"L{L_top}H{H_top}",
        "rank": R,
        "intact_metrics": intact_metrics,
        "n_for_50pct_impact": int(n_for_50),
        "n_for_66pct_impact": int(n_for_66),
        "n_for_80pct_impact": int(n_for_80),
        "top4_pct_of_total_impact": float(df.head(4)["pct_of_total"].sum()),
        "top10_pct_of_total_impact": float(df.head(10)["pct_of_total"].sum()),
        "core_factors": core_factors,
        "core_metrics": core_metrics,
        "elapsed_s": float(time.time() - t_phase),
    }
    (REP / "factor_ablation.json").write_text(json.dumps(summary, indent=2))

    print(f"\n[8] Summary:")
    print(f"  factors needed for 50%/66%/80% pooled impact: {n_for_50} / {n_for_66} / {n_for_80}")
    print(f"  top-4 explain {summary['top4_pct_of_total_impact']*100:.1f}% of pooled impact")
    print(f"  core ({core_factors}) sufficiency: branch {core_metrics['branch_balanced_acc']:.3f} (intact {intact_metrics['branch_balanced_acc']:.3f})")
    print(f"\nPhase 11 complete in {time.time()-t_phase:.1f}s")


if __name__ == "__main__":
    main()
