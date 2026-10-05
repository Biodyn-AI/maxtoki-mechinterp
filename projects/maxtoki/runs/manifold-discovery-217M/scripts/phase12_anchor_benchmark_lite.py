"""Phase 12 (lite) — anchor-level benchmark vs simple baselines.

Source paper's Phase 12 is a cell-level Robust-V2 88-split campaign across 8
baselines including scVI and Palantir. That's infeasible on this MBA: needs
per-cell features (we only saved anchor centroids), needs scVI training, etc.

Lite scope: anchor-level leave-one-donor-out splits, 5 endpoints, 4 simple
baselines vs the extracted H65 head. Trades source-paper-comparable absolute
numbers for tractability and a meaningful relative comparison.

Baselines (all anchor-level features built from data we already have):
  - raw_log1p_pseudo: pseudotime regression on log1p(centroid mean over layers)
  - pca10_anchor:     PCA-10 of anchor centroids (averaged over layers)
  - svd10_anchor:     SVD-10 of same
  - frozen_maxtoki_avgpool: anchor centroid averaged over all layers (1232-d)

Endpoints: branch_balanced_acc, stage_balanced_acc, cd_auroc, mm_auroc, pseudotime_spearman.

Splits: 11 leave-one-donor-out (one per internal-panel donor).

Outputs:
  reports/phase12_anchor_benchmark_lite.json
  reports/phase12_anchor_benchmark_lite.csv  (per split × baseline × endpoint)
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
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from scipy.stats import spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests

PROJ = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN / "scripts"))
from phase5_let_anchor import LETHead, train_let, build_pooled_drift, SEED  # type: ignore

ART = RUN / "artifacts"; REP = RUN / "reports"


def fit_clf(X_train, y_train, X_test, n_classes, epochs=600, lr=0.05):
    model = nn.Linear(X_train.shape[1], n_classes)
    nn.init.normal_(model.weight, std=0.01)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, yt = torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(Xt), yt)
        loss.backward(); opt.step()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).float())
        probs = torch.softmax(logits, -1).numpy()
        return logits.argmax(-1).numpy(), probs


def fit_reg(X_train, y_train, X_test, epochs=600, lr=0.05):
    model = nn.Linear(X_train.shape[1], 1)
    nn.init.normal_(model.weight, std=0.01)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, yt = torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((model(Xt).squeeze(-1) - yt) ** 2).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        return model(torch.from_numpy(X_test).float()).squeeze(-1).numpy()


def main():
    print("=" * 70)
    print("PHASE 12 (lite) — anchor-level benchmark vs simple baselines")
    print("=" * 70)
    t_phase = time.time()

    centroids = np.load(ART / "anchors/centroids_internal.npy")  # (290, 12, 1232)
    d_target = np.load(ART / "anchors/d_target_internal.npy")
    anchor_meta = pd.read_csv(ART / "anchors/anchor_meta_internal.csv")
    op_idx = json.loads((ART / "operators/operator_index.json").read_text())
    op_npz = np.load(ART / "operators/pooled_drift_components.npz")
    A_e, A_m, A_l = op_npz["A_early"], op_npz["A_mid"], op_npz["A_late"]
    partition = op_idx["block_partition"]

    # ---- Build candidate features
    avg_pool = centroids.mean(axis=1)  # (290, 1232)
    pseudo_log1p = np.log1p(np.clip(avg_pool - avg_pool.min(), 0, None))
    pca10 = PCA(n_components=10, random_state=SEED).fit_transform(avg_pool)
    svd10 = TruncatedSVD(n_components=10, random_state=SEED).fit_transform(avg_pool)
    drift = build_pooled_drift(centroids, A_e, A_m, A_l, partition)
    drift_std = (drift - drift.mean(0)) / (drift.std(0) + 1e-6)
    print(f"  features: avg_pool {avg_pool.shape}, pca10 {pca10.shape}, svd10 {svd10.shape}, drift {drift.shape}")

    # Per-baseline feature builder for the extracted head: train LET on train split → z
    # Source paper's "extracted head" benchmark uses the LET-10D latent as the comparator feature.

    # Labels
    dag = json.loads((RUN / "planning/h65_stage_dag.json").read_text())
    stage_to_branch = {k: v["branch"] for k, v in dag["stage_to_branch"].items()}
    stage_to_depth = {k: v["depth"] for k, v in dag["stage_to_branch"].items()}
    stages = anchor_meta["hema_stage"].astype(str).to_numpy()
    branches = np.array([stage_to_branch.get(s, "_unk") for s in stages])
    depths = np.array([stage_to_depth.get(s, -1) for s in stages], dtype=np.float32)
    is_cd4 = np.array([s in {"Naive_CD4", "Memory_CD4", "Treg"} for s in stages])
    is_cd8 = np.array([s in {"Naive_CD8", "Memory_CD8"} for s in stages])
    is_mono = np.array([s in {"Cl_Mono", "NC_Mono"} for s in stages])
    is_macro = np.array([s == "Macrophage" for s in stages])
    cd_mask = is_cd4 | is_cd8; cd_y = is_cd8.astype(np.int64)
    mm_mask = is_mono | is_macro; mm_y = is_macro.astype(np.int64)

    branch_set = sorted(set(branches.tolist()) - {"_unk"})
    stage_set = sorted(set(stages.tolist()))
    branch_to_idx = {b: i for i, b in enumerate(branch_set)}
    stage_to_idx = {s: i for i, s in enumerate(stage_set)}

    donors = anchor_meta["donor_id"].astype(str).to_numpy()
    unique_donors = sorted(np.unique(donors))
    print(f"  donors (splits): {unique_donors}")

    # Build feature dicts
    def get_features():
        return {
            "raw_log1p": pseudo_log1p,
            "pca10": pca10,
            "svd10": svd10,
            "frozen_maxtoki_avgpool": avg_pool,
            "extracted_let_drift_anchor": drift_std,
        }

    rows = []
    for split_id, holdout_donor in enumerate(unique_donors):
        train_mask = donors != holdout_donor
        test_mask = ~train_mask
        n_train, n_test = int(train_mask.sum()), int(test_mask.sum())
        if n_test < 3:
            print(f"  skipping donor {holdout_donor}: n_test={n_test}")
            continue
        feats = get_features()

        # For extracted head, retrain LET on training fold; z = projected latent
        head, z_full, _ = train_let(drift_std[train_mask], d_target[np.ix_(np.where(train_mask)[0], np.where(train_mask)[0])], label=f"split-{split_id}", verbose=False)
        with torch.no_grad():
            x_full = torch.from_numpy(drift_std).float()
            z_all = head(x_full)[0].numpy()
        feats["extracted_let_z10"] = z_all

        for b_name, X in feats.items():
            X_train = X[train_mask]; X_test = X[test_mask]
            # Drop test classes not seen in train
            # branch
            train_b_mask = train_mask.copy()
            train_b_idx = np.where(train_b_mask)[0]
            tb_y = np.array([branch_to_idx.get(b, -1) for b in branches])
            keep_train = train_b_idx[tb_y[train_b_idx] >= 0]
            keep_test = np.where(test_mask)[0]
            keep_test = keep_test[tb_y[keep_test] >= 0]
            if len(keep_train) < 5 or len(keep_test) < 2:
                bb_acc = float("nan")
            else:
                preds, _ = fit_clf(X[keep_train], tb_y[keep_train], X[keep_test], n_classes=len(branch_set))
                bb_acc = float(balanced_accuracy_score(tb_y[keep_test], preds))
            # stage
            ts_y = np.array([stage_to_idx[s] for s in stages])
            preds, _ = fit_clf(X[train_mask], ts_y[train_mask], X[test_mask], n_classes=len(stage_set))
            sb_acc = float(balanced_accuracy_score(ts_y[test_mask], preds))
            # cd4/cd8
            cd_train_idx = np.where(train_mask & cd_mask)[0]
            cd_test_idx = np.where(test_mask & cd_mask)[0]
            cd_auroc = float("nan")
            if len(cd_train_idx) >= 4 and len(cd_test_idx) >= 2:
                preds, probs = fit_clf(X[cd_train_idx], cd_y[cd_train_idx], X[cd_test_idx], n_classes=2)
                if len(np.unique(cd_y[cd_test_idx])) >= 2:
                    cd_auroc = float(roc_auc_score(cd_y[cd_test_idx], probs[:, 1]))
            # mono/macro
            mm_train_idx = np.where(train_mask & mm_mask)[0]
            mm_test_idx = np.where(test_mask & mm_mask)[0]
            mm_auroc = float("nan")
            if len(mm_train_idx) >= 4 and len(mm_test_idx) >= 2:
                preds, probs = fit_clf(X[mm_train_idx], mm_y[mm_train_idx], X[mm_test_idx], n_classes=2)
                if len(np.unique(mm_y[mm_test_idx])) >= 2:
                    mm_auroc = float(roc_auc_score(mm_y[mm_test_idx], probs[:, 1]))
            # pseudotime
            y_pred = fit_reg(X[train_mask], depths[train_mask], X[test_mask])
            try:
                pt_rho, _ = spearmanr(y_pred, depths[test_mask])
                pt_rho = float(pt_rho)
            except Exception:
                pt_rho = float("nan")
            rows.append({
                "split": split_id, "holdout_donor": holdout_donor,
                "baseline": b_name,
                "n_train": n_train, "n_test": n_test,
                "branch_balanced_acc": bb_acc,
                "stage_balanced_acc": sb_acc,
                "cd_auroc": cd_auroc,
                "mm_auroc": mm_auroc,
                "pseudotime_spearman": pt_rho,
            })
        print(f"  split {split_id+1}/{len(unique_donors)} (donor {holdout_donor}, n_test={n_test}) done")

    df = pd.DataFrame(rows)
    df.to_csv(REP / "phase12_anchor_benchmark_lite.csv", index=False)

    # Summary: mean per baseline × endpoint
    endpoints = ["branch_balanced_acc", "stage_balanced_acc", "cd_auroc", "mm_auroc", "pseudotime_spearman"]
    summary = df.groupby("baseline")[endpoints].mean().to_dict()
    summary_table = df.groupby("baseline")[endpoints].agg(["mean", "std"]).round(3)
    print("\n=== MEAN ± STD per baseline × endpoint (over splits) ===")
    print(summary_table.to_string())

    # Paired Wilcoxon: extracted_let_z10 vs each baseline, BH-FDR across endpoints
    results = []
    extracted = "extracted_let_z10"
    other_baselines = [b for b in df["baseline"].unique() if b != extracted]
    for ep in endpoints:
        for b in other_baselines:
            d_ext = df[df["baseline"] == extracted].set_index("split")[ep]
            d_b = df[df["baseline"] == b].set_index("split")[ep]
            paired = pd.concat([d_ext, d_b], axis=1, keys=["extracted", b]).dropna()
            if len(paired) < 3:
                results.append({"endpoint": ep, "baseline": b, "n_paired": int(len(paired)), "p": float("nan"), "delta_mean": float("nan")})
                continue
            try:
                stat, p = wilcoxon(paired["extracted"], paired[b])
            except ValueError:
                p = float("nan")
            results.append({
                "endpoint": ep, "baseline": b,
                "n_paired": int(len(paired)),
                "delta_mean": float((paired["extracted"] - paired[b]).mean()),
                "p": float(p),
            })
    res_df = pd.DataFrame(results)
    valid = ~res_df["p"].isna()
    if valid.any():
        res_df.loc[valid, "bh_q"] = multipletests(res_df.loc[valid, "p"].to_numpy(), method="fdr_bh")[1]
    res_df["sig_at_05"] = res_df["bh_q"] < 0.05
    res_df.to_csv(REP / "phase12_anchor_benchmark_lite_pairwise.csv", index=False)

    print("\n=== PAIRED WILCOXON (extracted_let_z10 vs each baseline), BH-FDR ===")
    print(res_df.to_string(index=False))

    out = {
        "scope_note": (
            "PHASE 12 LITE — anchor-level leave-one-donor-out (11 splits) over 5 endpoints "
            "and 5 features. NOT cell-level Robust-V2 with scVI/Palantir as in source paper §6 "
            "Phase 12. Use these as relative-comparison signal only."
        ),
        "n_splits": int(df["split"].nunique()),
        "n_baselines": int(df["baseline"].nunique()),
        "endpoints": endpoints,
        "mean_per_baseline": {k: {bs: float(v) for bs, v in d.items()} for k, d in summary.items()},
        "elapsed_s": float(time.time() - t_phase),
    }
    (REP / "phase12_anchor_benchmark_lite.json").write_text(json.dumps(out, indent=2))
    print(f"\nPhase 12 (lite) complete in {time.time()-t_phase:.1f}s")


if __name__ == "__main__":
    main()
