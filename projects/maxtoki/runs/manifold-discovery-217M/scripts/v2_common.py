"""v2 shared helpers for the manifold-discovery interval / facts revision (item D8).

All model code is COPIED (not imported) from scripts/phase5_let_anchor.py lines 47-135 and
scripts/phase7_external_validation.py lines 94-150, so that running v2 scripts never writes
into the existing run files (no __pycache__ writes: sys.dont_write_bytecode is set).
Only NEW files are written, under outputs/v2_intervals/ .

Read-only inputs: artifacts/anchors/*, artifacts/operators/*, artifacts/heads/let_anchor_internal.pt,
planning/h65_stage_dag.json, reports/*.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr

RUN = Path("<REPO_ROOT>/projects/maxtoki/runs/manifold-discovery-217M")
ART = RUN / "artifacts"
REP = RUN / "reports"
PLAN = RUN / "planning"
OUT = RUN / "outputs" / "v2_intervals"
OUT.mkdir(parents=True, exist_ok=True)

# ---- hyperparameters copied from phase5_let_anchor.py:47-52
ALPHA_RECON = 0.1
LATENT_DIM = 10
EPOCHS = 1500
LR = 5e-3
SEED = 42
DEVICE = "cpu"

GATE_TRUST = 0.80
GATE_CORR = 0.20


# ------------------------------------------------------------------ provenance
def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_run_config(name: str, inputs: list[Path], extra: dict) -> Path:
    cfg = {
        "script": name,
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "inputs": {str(p): sha256(p) for p in inputs},
        **extra,
    }
    p = OUT / f"run_config_{name}.json"
    p.write_text(json.dumps(cfg, indent=2))
    return p


CORE_INPUTS = [
    ART / "anchors/centroids_internal.npy", ART / "anchors/d_target_internal.npy",
    ART / "anchors/d_target_internal_null_shuffled.npy", ART / "anchors/anchor_meta_internal.csv",
    ART / "anchors/centroids_external.npy", ART / "anchors/d_target_external.npy",
    ART / "anchors/anchor_meta_external.csv",
    ART / "anchors/centroids_zeroshot.npy", ART / "anchors/d_target_zeroshot.npy",
    ART / "anchors/anchor_meta_zeroshot.csv",
    ART / "anchors/centroids_lung_nonhema.npy", ART / "anchors/d_target_lung_nonhema.npy",
    ART / "anchors/anchor_meta_lung_nonhema.csv",
    ART / "operators/pooled_drift_components.npz", ART / "operators/operator_index.json",
    ART / "heads/let_anchor_internal.pt", PLAN / "h65_stage_dag.json", REP / "quality_gates_spec.json",
]


# ------------------------------------------------------------------ features / head (copied)
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


def train_let(features: np.ndarray, d_target: np.ndarray):
    """Identical maths and RNG use to phase5_let_anchor.train_let (prints/history removed)."""
    torch.manual_seed(SEED)
    n, d_in = features.shape
    init_beta = max(1.0, float(d_target.max()) / (np.pi / 2.0 + 1e-6))
    head = LETHead(d_in, LATENT_DIM, init_beta=init_beta).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    x = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
    D = torch.from_numpy(np.ascontiguousarray(d_target, dtype=np.float32))
    for _ in range(EPOCHS):
        opt.zero_grad()
        z, recon = head(x)
        loss = let_loss(z, x, recon, D, beta=head.beta)
        loss.backward()
        opt.step()
    head.eval()
    return head


def head_z(head, f: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return head(torch.from_numpy(np.ascontiguousarray(f, dtype=np.float32)))[0].numpy()


# ------------------------------------------------------------------ data
def load_operators():
    op_idx = json.loads((ART / "operators/operator_index.json").read_text())
    npz = np.load(ART / "operators/pooled_drift_components.npz")
    return npz["A_early"], npz["A_mid"], npz["A_late"], op_idx["block_partition"]


def load_panel(panel):
    c = np.load(ART / f"anchors/centroids_{panel}.npy", mmap_mode="r")
    d = np.load(ART / f"anchors/d_target_{panel}.npy")
    m = pd.read_csv(ART / f"anchors/anchor_meta_{panel}.csv")
    return c, d, m


def stage_dag():
    return json.loads((PLAN / "h65_stage_dag.json").read_text())


def branch_labels(meta):
    s2b = {k: v["branch"] for k, v in stage_dag()["stage_to_branch"].items()}
    return meta["hema_stage"].map(s2b).fillna("_unk").to_numpy()


def stage_distance_table():
    """All-pairs shortest-path distances on the 34-node H65 stage DAG (undirected), as in
    phase1bc_hidden_states_and_centroids.build_ruler (99 = disconnected)."""
    dag = stage_dag()
    nodes = list(dag["stage_to_branch"].keys())
    adj = {n: set() for n in nodes}
    for a, b in dag["edges"]:
        adj[a].add(b); adj[b].add(a)
    idx = {n: i for i, n in enumerate(nodes)}
    T = np.full((len(nodes), len(nodes)), 99.0, dtype=np.float32)
    for s in nodes:
        dist = {s: 0}; q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1; q.append(v)
        for t, dd in dist.items():
            T[idx[s], idx[t]] = dd
    return nodes, idx, T


def ruler_from_stages(stages, idx, T):
    s = np.array([idx[x] for x in stages])
    D = T[np.ix_(s, s)].copy()
    np.fill_diagonal(D, 0.0)
    return D


class Frozen:
    """Everything needed to score panels with the frozen Phase-5 head."""

    def __init__(self):
        self.A_e, self.A_m, self.A_l, self.part = load_operators()
        c_int, self.d_int, self.m_int = load_panel("internal")
        self.F_int_raw = build_pooled_drift(np.asarray(c_int), self.A_e, self.A_m, self.A_l, self.part)
        self.mu = self.F_int_raw.mean(0)
        self.sd = self.F_int_raw.std(0) + 1e-6
        self.F_int = (self.F_int_raw - self.mu) / self.sd
        self.head = None

    def fit_head(self):
        self.head = train_let(self.F_int, self.d_int)
        return self.head

    def features(self, panel):
        c, d, m = load_panel(panel)
        f = (build_pooled_drift(np.asarray(c), self.A_e, self.A_m, self.A_l, self.part) - self.mu) / self.sd
        return f.astype(np.float32), d, m


# ------------------------------------------------------------------ metrics
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
    """Trustworthiness (Venna & Kaski 2006; same formula as sklearn.manifold.trustworthiness) from
    precomputed Euclidean distance matrices. Pairs that are copies of the SAME original anchor
    (origin[i]==origin[j]) are excluded like self-pairs. With no copies it equals sklearn's value
    (checked in v2_00_reproduce.py)."""
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


def group_spearman(Dhat, Dt, groups, origin, min_n=3, skip=("_unk",), detail=False):
    """Mean over groups (>= min_n members) of Spearman(latent arc-cos dist, ruler dist) on
    within-group pairs, excluding pairs that are copies of one anchor. Groups whose ruler is
    constant give NaN and are dropped (as in phase7_external_validation.py:132-150)."""
    vals, rows = [], []
    for g in pd.unique(groups):
        if g in skip:
            continue
        idx = np.where(groups == g)[0]
        if len(idx) < min_n:
            continue
        iu = np.triu_indices(len(idx), k=1)
        a = idx[iu[0]]; b = idx[iu[1]]
        keep = origin[a] != origin[b]
        rho = np.nan
        if keep.sum() >= 2:
            x = Dhat[a[keep], b[keep]]; y = Dt[a[keep], b[keep]]
            if np.ptp(y) > 0 and np.ptp(x) > 0:
                rho = float(spearmanr(x, y)[0])
        if not np.isnan(rho):
            vals.append(rho)
        rows.append({"group": str(g), "n_anchors": int(len(idx)), "n_distinct_anchors": int(len(set(origin[idx]))),
                     "n_pairs": int(keep.sum()), "rho": None if np.isnan(rho) else rho})
    mean = float(np.mean(vals)) if vals else float("nan")
    if detail:
        return mean, len(vals), rows
    return mean, len(vals)


def global_spearman(Dhat, Dt, origin):
    iu = np.triu_indices(len(origin), k=1)
    keep = origin[iu[0]] != origin[iu[1]]
    return float(spearmanr(Dhat[iu][keep], Dt[iu][keep])[0])


def random_holdout_frozen(z, d, n_iter=20, frac=0.2, seed=SEED):
    """phase7_external_validation.py:101-114 (frozen head: no refit)."""
    rng = np.random.default_rng(seed)
    n = len(d); out = []
    for _ in range(n_iter):
        idx = rng.permutation(n); n_test = max(2, int(round(n * frac)))
        test = sorted(idx[:n_test])
        Dh = arccos_dist(z[test]); Dt = d[np.ix_(test, test)]
        iu = np.triu_indices(len(test), k=1)
        rho = spearmanr(Dh[iu], Dt[iu])[0]
        if not np.isnan(rho):
            out.append(float(rho))
    return float(np.mean(out)) if out else float("nan")


def pct_ci(a, lo=2.5, hi=97.5):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    return [float(np.percentile(a, lo)), float(np.percentile(a, hi))]


def summarize(arr, gate=None):
    arr = np.asarray(arr, float)
    ok = arr[~np.isnan(arr)]
    out = {"ci95_percentile": pct_ci(ok), "mean": float(ok.mean()), "median": float(np.median(ok)),
           "sd": float(ok.std(ddof=1)), "n_valid": int(len(ok)), "n_nan": int(np.isnan(arr).sum())}
    if gate is not None:
        out[f"frac_below_{gate}"] = float(np.mean(ok < gate))
    return out


# ------------------------------------------------------------------ batched trainer (same maths, many heads at once)
def _init_W(d_in):
    """The exact initial W of train_let: torch.manual_seed(42) then LETHead(...) (kaiming draw, then normal std 0.01)."""
    torch.manual_seed(SEED)
    h = LETHead(d_in, LATENT_DIM, 1.0)
    return h.W.weight.detach().clone()


def train_let_batched(feat_list, D_list):
    """Fit len(feat_list) independent LET heads in one batched computation.

    Mathematically identical to calling train_let on each (same init W from seed 42, b = 0, log beta from the ruler
    max, same loss with means over the member's own n^2 pairs and n*d entries, Adam lr 5e-3 for 1500 steps; Adam acts
    element-wise so one optimiser over stacked parameters equals independent optimisers). Members are padded to a
    common n with masks. Floating-point summation order differs from the one-at-a-time trainer, so results agree with
    train_let only up to optimiser noise (quantified in v2_02b_batched_check.py).
    Returns a list of (W (10 x d), b (d,), log_beta) numpy tuples."""
    Bn = len(feat_list); d_in = feat_list[0].shape[1]
    ns = [f.shape[0] for f in feat_list]; nmax = max(ns)
    X = torch.zeros(Bn, nmax, d_in); Dt = torch.zeros(Bn, nmax, nmax); M = torch.zeros(Bn, nmax)
    for i, (f, D) in enumerate(zip(feat_list, D_list)):
        X[i, :ns[i]] = torch.from_numpy(np.ascontiguousarray(f, dtype=np.float32))
        Dt[i, :ns[i], :ns[i]] = torch.from_numpy(np.ascontiguousarray(D, dtype=np.float32))
        M[i, :ns[i]] = 1.0
    nvec = torch.tensor(ns, dtype=torch.float32)
    MM = M[:, :, None] * M[:, None, :]
    W0 = _init_W(d_in)
    W = W0[None].repeat(Bn, 1, 1).clone().requires_grad_(True)            # (B, 10, d)
    b = torch.zeros(Bn, d_in, requires_grad=True)
    lb0 = [float(np.log(max(1.0, float(D.max()) / (np.pi / 2.0 + 1e-6)))) for D in D_list]
    log_beta = torch.tensor(lb0, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([W, b, log_beta], lr=LR)
    for _ in range(EPOCHS):
        opt.zero_grad()
        xc = X - b[:, None, :]                                              # (B, n, d)
        z = torch.bmm(xc, W.transpose(1, 2))                                 # (B, n, 10)
        recon = torch.bmm(z, W) + b[:, None, :]                              # (B, n, d)
        zn = z / (z.norm(dim=2, keepdim=True) + 1e-9)
        cos = torch.bmm(zn, zn.transpose(1, 2)).clamp(-1 + 1e-7, 1 - 1e-7)
        d_hat = torch.exp(log_beta)[:, None, None] * torch.arccos(cos)
        fit = (((d_hat - Dt) ** 2) * MM).sum((1, 2)) / (nvec ** 2)
        rec = (((recon - X) ** 2).sum(2) * M).sum(1) / (nvec * d_in)
        loss = (fit + ALPHA_RECON * rec).sum()
        loss.backward()
        opt.step()
    return [(W[i].detach().numpy().copy(), b[i].detach().numpy().copy(), float(log_beta[i])) for i in range(Bn)]


def z_from_params(params, f):
    W, b, _ = params
    return ((f - b[None, :]) @ W.T).astype(np.float32)
