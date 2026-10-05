"""Shared helpers for the donor-level CI prototype.

All model code is COPIED from
runs/manifold-discovery-217M/scripts/phase5_let_anchor.py (lines 47-135)
so that we never import from the repo (import would write __pycache__ into it).
Read-only access to the run artefacts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr

RUN = Path("<REPO_ROOT>/projects/maxtoki/runs/manifold-discovery-217M")
ART = RUN / "artifacts"
REP = RUN / "reports"
OUT = Path("<AGENT_TMP>/<SESSION_DIR>/"
           "d422dc33-6d6f-4b2b-bbf7-f274d8a82fae/scratchpad/understand/manifold_ci")

# ---- hyperparameters, copied from phase5_let_anchor.py:47-52
ALPHA_RECON = 0.1
LATENT_DIM = 10
EPOCHS = 1500
LR = 5e-3
SEED = 42
DEVICE = "cpu"


def block_pool(centroids, layers):
    return centroids[:, layers, :].mean(axis=1)


def build_pooled_drift(centroids, A_early, A_mid, A_late, partition):
    early = [i + 1 for i in partition["early"]]
    mid = [i + 1 for i in partition["mid"]]
    late = [i + 1 for i in partition["late"]]
    x_e = block_pool(centroids, early); x_m = block_pool(centroids, mid); x_l = block_pool(centroids, late)
    y_e = x_e @ A_early; y_m = x_m @ A_mid; y_l = x_l @ A_late
    return np.concatenate([y_e - y_m, y_m - y_l], axis=1).astype(np.float32)


class LETHead(nn.Module):
    def __init__(self, in_dim, latent_dim, init_beta=1.0):
        super().__init__()
        self.W = nn.Linear(in_dim, latent_dim, bias=False)
        self.b = nn.Parameter(torch.zeros(in_dim))
        self.log_beta = nn.Parameter(torch.tensor(float(np.log(init_beta))))
        nn.init.normal_(self.W.weight, std=0.01)

    @property
    def beta(self):
        return torch.exp(self.log_beta)

    def forward(self, x):
        z = self.W(x - self.b)
        recon = (self.W.weight.T @ z.T).T + self.b
        return z, recon


def let_loss(z, x, recon, d_target, beta, alpha=ALPHA_RECON):
    z_norm = z / (z.norm(dim=1, keepdim=True) + 1e-9)
    cos = (z_norm @ z_norm.T).clamp(-1 + 1e-7, 1 - 1e-7)
    d_hat = beta * torch.arccos(cos)
    fit = ((d_hat - d_target) ** 2).mean()
    rec = ((recon - x) ** 2).mean()
    return fit + alpha * rec


def train_let(features, d_target):
    torch.manual_seed(SEED)
    n, d_in = features.shape
    init_beta = max(1.0, float(d_target.max()) / (np.pi / 2.0 + 1e-6))
    head = LETHead(d_in, LATENT_DIM, init_beta=init_beta).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    x = torch.from_numpy(np.ascontiguousarray(features))
    D = torch.from_numpy(d_target.astype(np.float32))
    for _ in range(EPOCHS):
        opt.zero_grad()
        z, recon = head(x)
        loss = let_loss(z, x, recon, D, beta=head.beta)
        loss.backward()
        opt.step()
    head.eval()
    return head


def load_operators():
    op_idx = json.loads((ART / "operators/operator_index.json").read_text())
    npz = np.load(ART / "operators/pooled_drift_components.npz")
    return npz["A_early"], npz["A_mid"], npz["A_late"], op_idx["block_partition"]


def load_panel(panel):
    c = np.load(ART / f"anchors/centroids_{panel}.npy")
    d = np.load(ART / f"anchors/d_target_{panel}.npy")
    m = pd.read_csv(ART / f"anchors/anchor_meta_{panel}.csv")
    return c, d, m


def branch_labels(meta):
    dag = json.loads((RUN / "planning/h65_stage_dag.json").read_text())
    s2b = {k: v["branch"] for k, v in dag["stage_to_branch"].items()}
    return meta["hema_stage"].map(s2b).fillna("_unk").to_numpy()


def arccos_dist(z):
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    return np.arccos(np.clip(zn @ zn.T, -1 + 1e-7, 1 - 1e-7))


def euclid(X):
    X = X.astype(np.float64)
    sq = (X * X).sum(1)
    D2 = sq[:, None] + sq[None, :] - 2 * X @ X.T
    np.maximum(D2, 0, out=D2)
    return np.sqrt(D2)


def trust_masked(DX, DZ, origin, k=15):
    """Trustworthiness (Venna & Kaski; same formula as sklearn.manifold.trustworthiness)
    computed from precomputed Euclidean distance matrices, where pairs that are copies of
    the SAME original anchor (origin[i]==origin[j]) are excluded like self-pairs.
    With no duplicates this equals sklearn's value (checked in 01_reproduce.py)."""
    n = DX.shape[0]
    same = origin[:, None] == origin[None, :]
    DXm = np.where(same, np.inf, DX)
    DZm = np.where(same, np.inf, DZ)
    ind_X = np.argsort(DXm, axis=1, kind="stable")
    inv = np.empty((n, n), dtype=np.int64)
    rows = np.arange(n)[:, None]
    inv[rows, ind_X] = np.arange(1, n + 1)[None, :]
    ind_Z = np.argsort(DZm, axis=1, kind="stable")[:, :k]
    ranks = inv[rows, ind_Z] - k
    t = ranks[ranks > 0].sum()
    return float(1.0 - t * (2.0 / (n * k * (2.0 * n - 3.0 * k - 1.0))))


def group_spearman(Dhat, Dt, groups, origin, min_n=3, skip="_unk"):
    """Mean over groups (>= min_n members) of Spearman(latent arc-cos dist, ruler dist) on
    within-group pairs, excluding pairs that are copies of the same original anchor.
    With no duplicates this equals the within-branch / within-donor loops of
    phase7_external_validation.py:116-150."""
    vals = []
    for g in pd.unique(groups):
        if g == skip:
            continue
        idx = np.where(groups == g)[0]
        if len(idx) < min_n:
            continue
        iu = np.triu_indices(len(idx), k=1)
        a = idx[iu[0]]; b = idx[iu[1]]
        keep = origin[a] != origin[b]
        if keep.sum() < 2:
            continue
        rho, _ = spearmanr(Dhat[a[keep], b[keep]], Dt[a[keep], b[keep]])
        if not np.isnan(rho):
            vals.append(float(rho))
    return float(np.mean(vals)) if vals else float("nan"), len(vals)
