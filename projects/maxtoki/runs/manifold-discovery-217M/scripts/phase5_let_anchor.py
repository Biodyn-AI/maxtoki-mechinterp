"""Phase 5 (anchor-trained variant) — initial LET head training + four-gate check.

Trains the lightweight Stage-2 adaptor g_θ : ℝ^{2464} → ℝ^{10} on the 290 internal
anchors using the LET (Latent Embedding Transfer) objective from the source
paper. Then evaluates the four hierarchical quality gates on the same panel:
trustworthiness, random/donor/branch holdout Spearman.

LET objective (paper §2.3):
    z = W_enc (x − b)
    d̂_{ij} = β arccos(cos(z_i, z_j))
    L = ||d̂ − d_target||² + α ||W_enc^T z + b − x||²

Pooled-drift feature for MaxToki (cell or anchor centroid):
    x_block = mean_{ℓ ∈ block} hidden_state_{ℓ}      ∈ ℝ^{1232}
    y_block = x_block @ A_block                       ∈ ℝ^{1232}
    f_drift  = concat(y_early − y_mid, y_mid − y_late) ∈ ℝ^{2464}

Outputs (under runs/manifold-discovery-217M/):
  artifacts/heads/let_anchor_internal.pt   trained head weights
  reports/let_anchor_train.json            loss curve, hyperparameters
  reports/quality_gates_let_anchor.json    four-gate result on internal panel + null branch
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
from sklearn.manifold import trustworthiness
from scipy.stats import spearmanr

PROJ = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parents[1]

ART = RUN / "artifacts"
ART_HEADS = ART / "heads"; ART_HEADS.mkdir(parents=True, exist_ok=True)
REP = RUN / "reports"; REP.mkdir(parents=True, exist_ok=True)

DEVICE = "cpu"  # LET is small (290 anchors, 2464-dim); CPU avoids MPS contention with Phase 1B

# Hyperparameters — start from pipeline §9 known-pitfalls #5 default suggestions
ALPHA_RECON = 0.1
BETA_DIST = 1.0 / np.pi
LATENT_DIM = 10
EPOCHS = 1500
LR = 5e-3
SEED = 42


def block_pool(centroids: np.ndarray, layers: list[int]) -> np.ndarray:
    """centroids: (n_anchors, n_layers+1, hidden). Return (n_anchors, hidden) pooled across layers."""
    sub = centroids[:, layers, :]
    return sub.mean(axis=1)


def build_pooled_drift(centroids: np.ndarray, A_early, A_mid, A_late, partition):
    # operator_index.json keys are "early"/"mid"/"late". Layer indices are
    # 0..n_layers-1 in transformer-output space; centroids index 0 is the
    # post-embedding hidden state, indices 1..n_layers are post-block outputs.
    # Convert layer indices to centroid indices by adding 1 (so layer-0 input
    # comes from centroid index 0, layer-1 input from centroid index 1, etc.)
    early = [i + 1 for i in partition["early"]]
    mid = [i + 1 for i in partition["mid"]]
    late = [i + 1 for i in partition["late"]]
    x_e = block_pool(centroids, early)
    x_m = block_pool(centroids, mid)
    x_l = block_pool(centroids, late)
    y_e = x_e @ A_early
    y_m = x_m @ A_mid
    y_l = x_l @ A_late
    return np.concatenate([y_e - y_m, y_m - y_l], axis=1).astype(np.float32)


class LETHead(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, init_beta: float = 1.0):
        super().__init__()
        self.W = nn.Linear(in_dim, latent_dim, bias=False)
        self.b = nn.Parameter(torch.zeros(in_dim))
        # β is a learnable scalar that fits arc-cos distances to the ruler scale.
        # Without this, fixing β to 1/π forces d̂ ∈ [0,1] which can't match a ruler in [0,9].
        self.log_beta = nn.Parameter(torch.tensor(float(np.log(init_beta))))
        nn.init.normal_(self.W.weight, std=0.01)

    @property
    def beta(self):
        return torch.exp(self.log_beta)

    def forward(self, x):
        z = self.W(x - self.b)
        recon = self.W.weight.T @ z.T
        recon = recon.T + self.b
        return z, recon


def let_loss(z: torch.Tensor, x: torch.Tensor, recon: torch.Tensor, d_target: torch.Tensor, beta: torch.Tensor, alpha=ALPHA_RECON):
    z_norm = z / (z.norm(dim=1, keepdim=True) + 1e-9)
    cos = z_norm @ z_norm.T
    cos = cos.clamp(-1 + 1e-7, 1 - 1e-7)
    d_hat = beta * torch.arccos(cos)
    fit = ((d_hat - d_target) ** 2).mean()
    rec = ((recon - x) ** 2).mean()
    return fit + alpha * rec, fit.item(), rec.item()


def train_let(features: np.ndarray, d_target: np.ndarray, label: str, verbose: bool = True):
    torch.manual_seed(SEED)
    n, d_in = features.shape
    # initialize β to match d_target scale: max d̂ at orthogonal latent = β * π/2 should approach max d_target
    init_beta = max(1.0, float(d_target.max()) / (np.pi / 2.0 + 1e-6))
    head = LETHead(d_in, LATENT_DIM, init_beta=init_beta).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    x = torch.from_numpy(features).to(DEVICE)
    D = torch.from_numpy(d_target.astype(np.float32)).to(DEVICE)
    history = []
    t0 = time.time()
    for epoch in range(EPOCHS):
        opt.zero_grad()
        z, recon = head(x)
        loss, fit, rec = let_loss(z, x, recon, D, beta=head.beta)
        loss.backward()
        opt.step()
        if verbose and (epoch % 100 == 0 or epoch == EPOCHS - 1):
            history.append({
                "epoch": epoch, "loss": loss.item(), "fit": fit, "rec": rec,
                "beta": float(head.beta.item()),
            })
            print(f"  [{label}] epoch {epoch:4d}  loss {loss.item():.4f}  fit {fit:.4f}  rec {rec:.4f}  β {head.beta.item():.3f}")
    print(f"  [{label}] trained {EPOCHS} epochs in {time.time()-t0:.1f}s")
    z_final = head(x)[0].detach().cpu().numpy()
    return head, z_final, history


def random_holdout_corr(features, d_target, n_iters=20, frac=0.2, seed=SEED):
    """Train on (1-frac), measure Spearman between latent arc-cos distances and d_target on the holdout."""
    rng = np.random.default_rng(seed)
    n = features.shape[0]
    corrs = []
    for _ in range(n_iters):
        idx = rng.permutation(n)
        n_test = int(round(n * frac))
        test = sorted(idx[:n_test])
        train = sorted(idx[n_test:])
        f_train = features[train]
        D_train = d_target[np.ix_(train, train)]
        head, _, _ = train_let(f_train, D_train, label="rand-holdout", verbose=False)
        with torch.no_grad():
            x_test = torch.from_numpy(features[test]).to(DEVICE)
            z_test, _ = head(x_test)
            z_test = z_test.detach().cpu().numpy()
        # arc-cos pairwise
        zn = z_test / (np.linalg.norm(z_test, axis=1, keepdims=True) + 1e-9)
        cos = zn @ zn.T
        cos = np.clip(cos, -1 + 1e-7, 1 - 1e-7)
        d_hat = np.arccos(cos)
        D_test = d_target[np.ix_(test, test)]
        triu = np.triu_indices(len(test), k=1)
        rho, _ = spearmanr(d_hat[triu], D_test[triu])
        corrs.append(float(rho))
    return float(np.mean(corrs)), float(np.std(corrs))


def grouped_holdout_corr(features, d_target, group_labels, label_name, n_min_anchors=3):
    """For each unique group, hold it out; evaluate on the held-out group."""
    rng = np.random.default_rng(SEED + 1)
    corrs = []
    groups = pd.Series(group_labels).unique()
    print(f"  [{label_name}] groups: {len(groups)}")
    for g in groups:
        train_mask = group_labels != g
        test_mask = ~train_mask
        n_test = int(test_mask.sum())
        if n_test < n_min_anchors:
            continue
        train = np.where(train_mask)[0]
        test = np.where(test_mask)[0]
        f_train = features[train]
        D_train = d_target[np.ix_(train, train)]
        head, _, _ = train_let(f_train, D_train, label=f"{label_name}-{g}", verbose=False)
        with torch.no_grad():
            x_test = torch.from_numpy(features[test]).to(DEVICE)
            z_test, _ = head(x_test)
            z_test = z_test.detach().cpu().numpy()
        zn = z_test / (np.linalg.norm(z_test, axis=1, keepdims=True) + 1e-9)
        cos = zn @ zn.T
        cos = np.clip(cos, -1 + 1e-7, 1 - 1e-7)
        d_hat = np.arccos(cos)
        D_test = d_target[np.ix_(test, test)]
        triu = np.triu_indices(n_test, k=1)
        if len(triu[0]) == 0:
            continue
        rho, _ = spearmanr(d_hat[triu], D_test[triu])
        if not np.isnan(rho):
            corrs.append(float(rho))
    return (float(np.mean(corrs)) if corrs else float("nan")), len(corrs)


def main():
    print("=" * 70)
    print("PHASE 5 (anchor-trained) — initial LET head + four-gate check")
    print("=" * 70)
    t_phase = time.time()

    print("\n[1] Loading internal anchors and operator library...")
    centroids = np.load(ART / "anchors/centroids_internal.npy")
    d_target = np.load(ART / "anchors/d_target_internal.npy")
    d_target_null = np.load(ART / "anchors/d_target_internal_null_shuffled.npy")
    anchor_meta = pd.read_csv(ART / "anchors/anchor_meta_internal.csv")
    op_idx = json.loads((ART / "operators/operator_index.json").read_text())
    op_npz = np.load(ART / "operators/pooled_drift_components.npz")
    A_early = op_npz["A_early"].astype(np.float32)
    A_mid = op_npz["A_mid"].astype(np.float32)
    A_late = op_npz["A_late"].astype(np.float32)
    partition = op_idx["block_partition"]
    print(f"  centroids: {centroids.shape}  d_target: {d_target.shape}")
    print(f"  A_early: {A_early.shape}  A_mid: {A_mid.shape}  A_late: {A_late.shape}")

    print("\n[2] Building pooled-drift feature...")
    features = build_pooled_drift(centroids, A_early, A_mid, A_late, partition)
    print(f"  features: {features.shape}")

    # Standardise feature scale to keep optimization stable
    feat_mean = features.mean(axis=0, keepdims=True)
    feat_std = features.std(axis=0, keepdims=True) + 1e-6
    features = (features - feat_mean) / feat_std

    print("\n[3] Training LET head on full internal panel (positive branch H65)...")
    head, z_final, hist_pos = train_let(features, d_target, label="H65")
    torch.save(head.state_dict(), ART_HEADS / "let_anchor_internal.pt")

    print("\n[4] Trustworthiness (k=15)...")
    trust = trustworthiness(features, z_final, n_neighbors=15)
    print(f"  trustworthiness = {trust:.4f}")

    print("\n[5] Random holdout (20-fold, frac 0.2)...")
    rand_mean, rand_std = random_holdout_corr(features, d_target, n_iters=10, frac=0.2)
    print(f"  random holdout Spearman = {rand_mean:.4f} ± {rand_std:.4f}")

    print("\n[6] Donor holdout...")
    donor_corr, n_donor_groups = grouped_holdout_corr(
        features, d_target, anchor_meta["donor_id"].astype(str).to_numpy(), "donor"
    )
    print(f"  donor holdout Spearman = {donor_corr:.4f} (over {n_donor_groups} donors)")

    print("\n[7] Branch holdout (using stage_to_branch on H65 DAG)...")
    dag = json.loads((RUN / "planning/h65_stage_dag.json").read_text())
    stage_to_branch = {k: v["branch"] for k, v in dag["stage_to_branch"].items()}
    branch_per_anchor = anchor_meta["hema_stage"].map(stage_to_branch).fillna("_unk").to_numpy()
    branch_corr, n_branch_groups = grouped_holdout_corr(features, d_target, branch_per_anchor, "branch")
    print(f"  branch holdout Spearman = {branch_corr:.4f} (over {n_branch_groups} branches)")

    print("\n[8] Null branch (within-branch shuffled stage labels) — must FAIL gates...")
    head_null, z_null, _ = train_let(features, d_target_null, label="H65_null")
    trust_null = trustworthiness(features, z_null, n_neighbors=15)
    rand_null_mean, _ = random_holdout_corr(features, d_target_null, n_iters=10, frac=0.2)
    donor_null, _ = grouped_holdout_corr(
        features, d_target_null, anchor_meta["donor_id"].astype(str).to_numpy(), "donor_null"
    )
    branch_null, _ = grouped_holdout_corr(features, d_target_null, branch_per_anchor, "branch_null")

    gates = json.loads((REP / "quality_gates_spec.json").read_text())["gates"]
    pos = {
        "trustworthiness": float(trust),
        "random_holdout": float(rand_mean),
        "donor_holdout": float(donor_corr),
        "branch_holdout": float(branch_corr),
    }
    pos["passes"] = (
        pos["trustworthiness"] >= gates["trustworthiness_min"]
        and pos["random_holdout"] >= gates["random_holdout_correlation_min"]
        and pos["donor_holdout"] >= gates["donor_holdout_correlation_min"]
        and pos["branch_holdout"] >= gates["clade_branch_holdout_correlation_min"]
    )
    null = {
        "trustworthiness": float(trust_null),
        "random_holdout": float(rand_null_mean),
        "donor_holdout": float(donor_null),
        "branch_holdout": float(branch_null),
    }
    null["passes"] = (
        null["trustworthiness"] >= gates["trustworthiness_min"]
        and null["random_holdout"] >= gates["random_holdout_correlation_min"]
        and null["donor_holdout"] >= gates["donor_holdout_correlation_min"]
        and null["branch_holdout"] >= gates["clade_branch_holdout_correlation_min"]
    )
    report = {
        "branch": "H65_anchor_LET10D",
        "panel": "internal",
        "feature": "pooled_drift_2464",
        "head_variant": "anchor_trained_baseline",
        "n_anchors": int(centroids.shape[0]),
        "gates_thresholds": gates,
        "positive": pos,
        "null_shuffled": null,
        "verdict": (
            "POSITIVE: H65 passes all four gates and null fails at least one"
            if pos["passes"] and not null["passes"]
            else (
                "INCONCLUSIVE: H65 fails one or more gates" if not pos["passes"]
                else "PIPELINE_ARTEFACT_RISK: null also passes — branch is not specific"
            )
        ),
    }
    (REP / "quality_gates_let_anchor.json").write_text(json.dumps(report, indent=2))
    (REP / "let_anchor_train.json").write_text(json.dumps({
        "history_pos": hist_pos,
        "hyperparameters": {
            "alpha_recon": ALPHA_RECON, "beta_dist": BETA_DIST, "latent_dim": LATENT_DIM,
            "epochs": EPOCHS, "lr": LR, "seed": SEED, "device": DEVICE,
        },
    }, indent=2))

    print("\n" + "=" * 70)
    print(f"VERDICT: {report['verdict']}")
    print(f"  H65   trust={trust:.3f}  rand={rand_mean:.3f}  donor={donor_corr:.3f}  branch={branch_corr:.3f}")
    print(f"  null  trust={trust_null:.3f}  rand={rand_null_mean:.3f}  donor={donor_null:.3f}  branch={branch_null:.3f}")
    print(f"Total time: {time.time()-t_phase:.1f}s")


if __name__ == "__main__":
    main()
