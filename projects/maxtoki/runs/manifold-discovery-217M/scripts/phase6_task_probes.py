"""Phase 6 — Stage 3 task probes on the z_final latent (anchor LET head).

Trains small task-specific probes on the 10D LET latent for the eight endpoints
the source paper benchmarks:
  - branch_balanced_acc, branch_macro_f1
  - stage_balanced_acc,  stage_macro_f1
  - cd4_cd8_auroc,       cd4_cd8_balanced_acc       (T_lineage subset)
  - mono_macro_auroc                                 (monocyte/macrophage subset)
  - pseudotime_spearman  (regress stage depth = DAG distance from HSC)

Linear probes are the default (per pipeline §6 Phase 6). Splits: 80/20 random
per task, 10 random seeds for stability.

Outputs:
  reports/probe_metrics_anchor_head.json
  artifacts/heads/probes_anchor_head.pt
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
from phase5_let_anchor import LETHead, train_let, build_pooled_drift, SEED  # type: ignore

ART = RUN / "artifacts"
REP = RUN / "reports"
DEVICE = "cpu"
N_SEEDS = 10


def fit_linear_classifier(X_train, y_train, X_test, y_test, n_classes):
    """Tiny linear classifier (with bias) trained via Adam."""
    rng = torch.Generator().manual_seed(SEED)
    model = nn.Linear(X_train.shape[1], n_classes)
    nn.init.normal_(model.weight, std=0.01)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    Xt = torch.from_numpy(X_train).float()
    yt = torch.from_numpy(y_train).long()
    for _ in range(800):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(Xt), yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).float())
        probs = torch.softmax(logits, dim=-1).numpy()
        preds = probs.argmax(-1)
    return preds, probs


def fit_linear_regressor(X_train, y_train, X_test):
    rng = torch.Generator().manual_seed(SEED)
    model = nn.Linear(X_train.shape[1], 1)
    nn.init.normal_(model.weight, std=0.01)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    Xt = torch.from_numpy(X_train).float()
    yt = torch.from_numpy(y_train).float()
    for _ in range(800):
        opt.zero_grad()
        loss = ((model(Xt).squeeze(-1) - yt) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return model(torch.from_numpy(X_test).float()).squeeze(-1).numpy()


def main():
    print("=" * 70)
    print("PHASE 6 — Stage 3 task probes")
    print("=" * 70)
    t_phase = time.time()

    # ---- inputs
    centroids = np.load(ART / "anchors/centroids_internal.npy")
    d_target = np.load(ART / "anchors/d_target_internal.npy")
    anchor_meta = pd.read_csv(ART / "anchors/anchor_meta_internal.csv")
    op_idx = json.loads((ART / "operators/operator_index.json").read_text())
    op_npz = np.load(ART / "operators/pooled_drift_components.npz")
    A_early, A_mid, A_late = op_npz["A_early"], op_npz["A_mid"], op_npz["A_late"]

    # ---- recompute features and z_final using Phase-5 head
    f_full = build_pooled_drift(centroids, A_early, A_mid, A_late, op_idx["block_partition"])
    f_full = (f_full - f_full.mean(0)) / (f_full.std(0) + 1e-6)
    head, z_final, _ = train_let(f_full, d_target, label="anchor_head", verbose=False)
    print(f"  z_final shape: {z_final.shape}")

    # ---- labels
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

    is_T = np.array([s in {"Naive_CD4", "Memory_CD4", "Naive_CD8", "Memory_CD8", "Treg", "MAIT", "gd_T", "DN_T", "DP_T"} for s in stages])
    is_cd4 = np.array([s in {"Naive_CD4", "Memory_CD4", "Treg"} for s in stages])
    is_cd8 = np.array([s in {"Naive_CD8", "Memory_CD8"} for s in stages])
    is_mm = np.array([s in {"Cl_Mono", "NC_Mono", "Macrophage"} for s in stages])
    is_mono = np.array([s in {"Cl_Mono", "NC_Mono"} for s in stages])
    is_macro = np.array([s == "Macrophage" for s in stages])

    print(f"  anchor counts:")
    print(f"    branches available: {len(branch_set)}; total labelled: {(branch_y >= 0).sum()}")
    print(f"    T lineage subset: {is_T.sum()} (CD4 {is_cd4.sum()}, CD8 {is_cd8.sum()})")
    print(f"    Mono/Macro subset: {is_mm.sum()} (mono {is_mono.sum()}, macro {is_macro.sum()})")

    rng = np.random.default_rng(SEED)
    metrics = {"endpoints": []}

    def run_classifier(name, X, y, n_classes, mask=None, use_auroc_if_binary=False):
        if mask is not None:
            X, y = X[mask], y[mask]
        results = []
        for seed in range(N_SEEDS):
            rg = np.random.default_rng(SEED + seed)
            n = len(y)
            idx = rg.permutation(n)
            n_test = max(2, int(round(n * 0.2)))
            test = idx[:n_test]; train = idx[n_test:]
            preds, probs = fit_linear_classifier(X[train], y[train], X[test], y[test], n_classes)
            r = {"balanced_acc": float(balanced_accuracy_score(y[test], preds)),
                 "macro_f1": float(f1_score(y[test], preds, average="macro", zero_division=0))}
            if use_auroc_if_binary and n_classes == 2:
                try:
                    r["auroc"] = float(roc_auc_score(y[test], probs[:, 1]))
                except ValueError:
                    r["auroc"] = float("nan")
            results.append(r)
        avg = {k: float(np.mean([r[k] for r in results])) for k in results[0]}
        avg["std_balanced_acc"] = float(np.std([r["balanced_acc"] for r in results]))
        avg["n_anchors"] = int(len(y))
        metrics["endpoints"].append({"name": name, **avg})
        print(f"  {name:30s}  bal_acc={avg['balanced_acc']:.3f}±{avg['std_balanced_acc']:.3f}  macro_f1={avg.get('macro_f1', float('nan')):.3f}" +
              (f"  auroc={avg.get('auroc', float('nan')):.3f}" if 'auroc' in avg else ""))

    print("\n[1] Branch / Stage classification (multi-class)...")
    run_classifier("branch_classification", z_final[branch_y >= 0], branch_y[branch_y >= 0], len(branch_set))
    run_classifier("stage_classification", z_final, stage_y, len(stage_set))

    print("\n[2] CD4 vs CD8 (T-lineage subset)...")
    cd_mask = is_cd4 | is_cd8
    cd_y = is_cd8.astype(np.int64)  # 1=CD8, 0=CD4
    if cd_mask.sum() >= 4:
        run_classifier("cd4_cd8", z_final, cd_y, 2, mask=cd_mask, use_auroc_if_binary=True)
    else:
        print("  insufficient CD4/CD8 anchors — skipped")

    print("\n[3] Mono vs Macro...")
    mm_mask = is_mono | is_macro
    mm_y = is_macro.astype(np.int64)
    if mm_mask.sum() >= 4:
        run_classifier("mono_macro", z_final, mm_y, 2, mask=mm_mask, use_auroc_if_binary=True)
    else:
        print("  insufficient mono/macro anchors — skipped")

    print("\n[4] Pseudotime regression (stage depth on H65 DAG)...")
    pseudo_results = []
    for seed in range(N_SEEDS):
        rg = np.random.default_rng(SEED + seed)
        n = len(depths)
        idx = rg.permutation(n)
        n_test = int(round(n * 0.2))
        test = idx[:n_test]; train = idx[n_test:]
        y_pred = fit_linear_regressor(z_final[train], depths[train], z_final[test])
        rho, _ = spearmanr(y_pred, depths[test])
        pseudo_results.append(float(rho))
    metrics["endpoints"].append({
        "name": "pseudotime_spearman",
        "spearman": float(np.mean(pseudo_results)),
        "std": float(np.std(pseudo_results)),
        "n_anchors": int(len(depths)),
    })
    print(f"  pseudotime_spearman             ρ={np.mean(pseudo_results):.3f}±{np.std(pseudo_results):.3f}")

    metrics["latent_dim"] = int(z_final.shape[1])
    metrics["n_anchors_total"] = int(z_final.shape[0])
    metrics["seed"] = SEED
    metrics["n_seeds"] = N_SEEDS
    metrics["elapsed_s"] = float(time.time() - t_phase)
    (REP / "probe_metrics_anchor_head.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nPhase 6 complete in {time.time()-t_phase:.1f}s")
    print(f"Saved: reports/probe_metrics_anchor_head.json")


if __name__ == "__main__":
    main()
