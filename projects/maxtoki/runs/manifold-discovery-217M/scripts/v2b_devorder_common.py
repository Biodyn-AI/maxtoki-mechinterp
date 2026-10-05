"""v2b (item D8b) shared helpers: does MaxToki-217M carry a hematopoietic developmental ordering beyond
cell-type identity?

Gate code: the run's own functions are IMPORTED read-only from scripts/phase5_let_anchor.py
(train_let, random_holdout_corr, grouped_holdout_corr, build_pooled_drift). The frozen-panel gate code of
phase7_external_validation.py / phase8_zeroshot_transfer.py / phase15_validate_candidate.py lives inside their
main() functions, so it is copied here line for line (frozen_gates_h65, frozen_gates_cat) and checked against
the deployed numbers in v2b_devorder_03_frozen.py.

Two helpers re-implement phase5 holdouts so that the held-out latent coordinates can be kept
(random_holdout_detail, grouped_holdout_detail). They use the same RNG calls, the same train_let and the same
Spearman, and are checked to give the same numbers as the imported originals (v2b_devorder_02_pool.py,
task 'check').

Nothing in this module writes outside outputs/v2b_devorder/. sys.dont_write_bytecode is set before any import
of run scripts, so no .pyc files are written into scripts/__pycache__.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import contextlib
import hashlib
import io
import json
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, rankdata

RUN = Path("<REPO_ROOT>/projects/maxtoki/runs/manifold-discovery-217M")
SCR = RUN / "scripts"
ART = RUN / "artifacts"
REP = RUN / "reports"
PLAN = RUN / "planning"
OUT = RUN / "outputs" / "v2b_devorder"
FEAT = OUT / "features"
HEADS = OUT / "heads"
for _d in (OUT, FEAT, HEADS):
    _d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCR))
with contextlib.redirect_stdout(io.StringIO()):
    import phase5_let_anchor as P5  # noqa: E402  (read-only import; prints nothing at import)

SEED = P5.SEED                 # 42
GATE_TRUST = 0.80
GATE_CORR = 0.20
PANELS = ["internal", "external", "zeroshot", "lung_nonhema"]
BLOOD = ["internal", "external", "zeroshot"]


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


def write_run_config(name: str, inputs: list, extra: dict) -> Path:
    cfg = {"script": name, "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
           "torch_threads": torch.get_num_threads(),
           "inputs": {str(p): sha256(Path(p)) for p in inputs}, **extra}
    p = OUT / f"run_config_{name}.json"
    p.write_text(json.dumps(cfg, indent=2, default=str))
    return p


# ------------------------------------------------------------------ metadata, stages, rulers
def meta(panel: str) -> pd.DataFrame:
    return pd.read_csv(ART / f"anchors/anchor_meta_{panel}.csv")


def stage_dag():
    return json.loads((PLAN / "h65_stage_dag.json").read_text())


STAGE_TO_BRANCH = {k: v["branch"] for k, v in stage_dag()["stage_to_branch"].items()}
STAGE_NODES = list(stage_dag()["stage_to_branch"].keys())


def branches_of(stages) -> np.ndarray:
    return pd.Series(list(stages)).map(STAGE_TO_BRANCH).fillna("_unk").to_numpy()


def stage_distance_table():
    """All-pairs shortest path on the 34-node stage graph (undirected), 99 if disconnected;
    the same rule as phase1bc_hidden_states_and_centroids.build_ruler."""
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


_NODES, _IDX, _T = stage_distance_table()


def h65_ruler(stages) -> np.ndarray:
    s = np.array([_IDX[x] for x in stages])
    D = _T[np.ix_(s, s)].copy()
    np.fill_diagonal(D, 0.0)
    return D


# ------------------------------------------------------------------ features
def load_feat(rep: str, panel: str) -> np.ndarray:
    return np.load(FEAT / f"{rep}__{panel}.npy").astype(np.float32)


def list_reps() -> list:
    return sorted({p.name.split("__")[0] for p in FEAT.glob("*__internal.npy")})


# ------------------------------------------------------------------ head fitting (the run's own trainer)
def fit(features: np.ndarray, D: np.ndarray):
    """phase5_let_anchor.train_let (seed 42, 1500 epochs, Adam 5e-3); its progress print is silenced."""
    with contextlib.redirect_stdout(io.StringIO()):
        head, z, _ = P5.train_let(np.ascontiguousarray(features, dtype=np.float32), D, label="v2b", verbose=False)
    head.eval()
    return head, z


def head_params(head) -> dict:
    sd = head.state_dict()
    return {"W": sd["W.weight"].numpy().copy(), "b": sd["b"].numpy().copy(), "log_beta": float(sd["log_beta"])}


def z_from(params: dict, f: np.ndarray) -> np.ndarray:
    """Same maths as LETHead.forward: z = W (x - b), in float32 torch to match the run."""
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(f, dtype=np.float32))
        W = torch.from_numpy(params["W"]); b = torch.from_numpy(params["b"])
        return ((x - b) @ W.T).numpy()


def arccos_dist(z: np.ndarray) -> np.ndarray:
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    return np.arccos(np.clip(zn @ zn.T, -1 + 1e-7, 1 - 1e-7))


def _rho(x, y):
    if len(x) < 2 or np.ptp(y) == 0 or np.ptp(x) == 0:
        return float("nan")
    r = spearmanr(x, y)[0]
    return float(r)


def pair_rhos(Dh: np.ndarray, Dt: np.ndarray, ct: np.ndarray | None):
    """Spearman over all i<j pairs, and over pairs whose cell_type differs (ct given)."""
    n = Dh.shape[0]
    iu = np.triu_indices(n, k=1)
    allr = _rho(Dh[iu], Dt[iu]) if len(iu[0]) else float("nan")
    if ct is None:
        return allr, float("nan")
    keep = ct[iu[0]] != ct[iu[1]]
    diff = _rho(Dh[iu][keep], Dt[iu][keep]) if keep.sum() >= 2 else float("nan")
    return allr, diff


def nanmean(v):
    v = [x for x in v if x is not None and not np.isnan(x)]
    return (float(np.mean(v)) if v else float("nan")), len(v)


# ------------------------------------------------------------------ internal holdouts that keep z
def random_holdout_detail(features, D, ct, n_iters=10, frac=0.2, seed=SEED):
    """phase5_let_anchor.random_holdout_corr, plus the different-cell-type-pair Spearman."""
    rng = np.random.default_rng(seed)
    n = features.shape[0]
    allv, diffv = [], []
    for _ in range(n_iters):
        idx = rng.permutation(n)
        n_test = int(round(n * frac))
        test = sorted(idx[:n_test]); train = sorted(idx[n_test:])
        head, _ = fit(features[train], D[np.ix_(train, train)])
        zt = z_from(head_params(head), features[test])
        a, d = pair_rhos(arccos_dist(zt), D[np.ix_(test, test)], None if ct is None else ct[test])
        allv.append(a); diffv.append(d)
    return {"mean": float(np.mean(allv)), "sd": float(np.std(allv)), "values": allv,
            "diff_ct_mean": nanmean(diffv)[0], "diff_ct_values": diffv}


def grouped_holdout_detail(features, D, groups, ct, n_min_anchors=3, keep_z=False):
    """phase5_let_anchor.grouped_holdout_corr, plus the different-cell-type-pair Spearman.
    Groups whose held-out ruler is constant give NaN in the original and are dropped there; here they are
    skipped before fitting (same result, fewer fits)."""
    groups = np.asarray(groups)
    rows = {}
    for g in pd.Series(groups).unique():
        test = np.where(groups == g)[0]
        if len(test) < n_min_anchors:
            continue
        Dt = D[np.ix_(test, test)]
        iu = np.triu_indices(len(test), k=1)
        if len(iu[0]) == 0 or np.ptp(Dt[iu]) == 0:
            continue
        train = np.where(groups != g)[0]
        head, _ = fit(features[train], D[np.ix_(train, train)])
        zt = z_from(head_params(head), features[test])
        a, d = pair_rhos(arccos_dist(zt), Dt, None if ct is None else ct[test])
        rows[str(g)] = {"n": int(len(test)), "rho": a, "rho_diff_ct": d}
        if keep_z:
            rows[str(g)]["test_idx"] = test.tolist()
            rows[str(g)]["z"] = zt.tolist()
    m, k = nanmean([r["rho"] for r in rows.values()])
    md, kd = nanmean([r["rho_diff_ct"] for r in rows.values()])
    return {"mean": m, "n_groups": k, "diff_ct_mean": md, "n_groups_diff_ct": kd, "per_group": rows}


# ------------------------------------------------------------------ frozen-panel gates (copied from phase7/8)
def frozen_gates_h65(f, z, D, donors, branches, ct, rand_iters=20):
    """phase7_external_validation.py:94-150 / phase8_zeroshot_transfer.py:76-139 (identical maths),
    plus different-cell-type-pair versions. f: standardised panel features; z: frozen head output."""
    from sklearn.manifold import trustworthiness
    out = {"trust": float(trustworthiness(f, z, n_neighbors=15))}
    Dh = arccos_dist(z)
    out["global"], out["global_diff_ct"] = pair_rhos(Dh, D, ct)
    rng = np.random.default_rng(SEED)
    n = len(D); ra, rd = [], []
    for _ in range(rand_iters):
        idx = rng.permutation(n); n_test = max(2, int(round(n * 0.2)))
        test = sorted(idx[:n_test])
        a, d = pair_rhos(Dh[np.ix_(test, test)], D[np.ix_(test, test)], ct[test])
        if not np.isnan(a):
            ra.append(a)
        rd.append(d)
    out["random"] = nanmean(ra)[0]; out["random_diff_ct"] = nanmean(rd)[0]
    for name, lab in (("donor", donors), ("branch", branches)):
        va, vd, per = [], [], {}
        for g in pd.Series(lab).unique():
            if g == "_unk":
                continue
            m = np.where(lab == g)[0]
            if len(m) < 3:
                continue
            a, d = pair_rhos(Dh[np.ix_(m, m)], D[np.ix_(m, m)], ct[m])
            if not np.isnan(a):
                va.append(a)
            if not np.isnan(d):
                vd.append(d)
            per[str(g)] = {"n": int(len(m)), "rho": a, "rho_diff_ct": d}
        out[name] = nanmean(va)[0]; out[name + "_n"] = len(va)
        out[name + "_diff_ct"] = nanmean(vd)[0]; out[name + "_diff_ct_n"] = len(vd)
        out[name + "_per_group"] = per
    out["passes_4"] = bool(out["trust"] >= GATE_TRUST and out["random"] >= GATE_CORR
                           and out["donor"] >= GATE_CORR and out["branch"] >= GATE_CORR)
    return out


def frozen_gates_cat(f, z, D, donors, groups, ct):
    """phase15_validate_candidate.evaluate_frozen (random 20 subsets with seed 42, per-group Spearman over groups
    of >= 3 anchors with a non-constant ruler), plus different-cell-type-pair versions."""
    from sklearn.manifold import trustworthiness
    out = {"trust": float(trustworthiness(f, z, n_neighbors=15)) if len(f) > 30 else float("nan")}
    Dh = arccos_dist(z)
    out["global"], out["global_diff_ct"] = pair_rhos(Dh, D, ct)
    rng = np.random.default_rng(SEED)
    n = len(D); ra, rd = [], []
    for _ in range(20):
        idx = rng.permutation(n); n_test = max(2, int(round(n * 0.2)))
        test = sorted(idx[:n_test])
        Dt = D[np.ix_(test, test)]; iu = np.triu_indices(len(test), 1)
        if len(iu[0]) == 0 or Dt[iu].std() < 1e-6:
            continue
        a, d = pair_rhos(Dh[np.ix_(test, test)], Dt, ct[test])
        if not np.isnan(a):
            ra.append(a)
        rd.append(d)
    out["random"] = nanmean(ra)[0]; out["random_diff_ct"] = nanmean(rd)[0]
    for name, lab in (("donor", donors), ("category", groups)):
        va, vd = [], []
        for g in pd.Series(lab).unique():
            m = np.where(lab == g)[0]
            if len(m) < 3:
                continue
            Dt = D[np.ix_(m, m)]; iu = np.triu_indices(len(m), 1)
            if len(iu[0]) == 0 or Dt[iu].std() < 1e-6:
                continue
            a, d = pair_rhos(Dh[np.ix_(m, m)], Dt, ct[m])
            if not np.isnan(a):
                va.append(a)
            if not np.isnan(d):
                vd.append(d)
        out[name] = nanmean(va)[0]; out[name + "_n"] = len(va)
        out[name + "_diff_ct"] = nanmean(vd)[0]
    return out


# ------------------------------------------------------------------ structured null for H65: class -> stage map
def h65_classes():
    """A 'class' is a (cell_type, hema_stage) pair present in any blood panel. Stage is derived from
    free_annotation > cell_type, so one cell_type can carry several stages; the class is the finest label the
    anchors were built from (apart from donor and tissue)."""
    rows = []
    for p in BLOOD:
        m = meta(p)
        rows += list(zip(m["cell_type"].astype(str), m["hema_stage"].astype(str)))
    cls = sorted(set(rows))
    return cls


def permuted_stage_map(classes, rng):
    """Permute the class -> stage map among classes of the same branch (branch = branch of the class's stage).
    The multiset of stages per branch is kept, so every anchor stays in its branch."""
    by_branch = {}
    for c in classes:
        by_branch.setdefault(STAGE_TO_BRANCH[c[1]], []).append(c)
    new = {}
    for b, cs in sorted(by_branch.items()):
        stages = [c[1] for c in cs]
        perm = rng.permutation(len(cs))
        for c, j in zip(cs, perm):
            new[c] = stages[j]
    return new


def anchor_classes(m: pd.DataFrame):
    return list(zip(m["cell_type"].astype(str), m["hema_stage"].astype(str)))
