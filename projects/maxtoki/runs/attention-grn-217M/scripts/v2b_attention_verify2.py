"""v2b_attention_verify2 — second independent check of revision item D1b (attention
direction and rank baselines). CPU only. No model forward pass, no model weights, no
tokeniser call.

Written from scratch. Does NOT import v2b_attention.py, v2b_attention_verify.py or
v2_common.py. Own code for: score variants, AUROC (sklearn per perturbation; own
TF-by-TF count matrix for pooled TRRUST AUROC), bootstraps (own seeds), TRRUST matrix
(own TSV parser), a CHECKERBOARD-SWAP degree-preserving null (a different sampler from
the Curveball used by D1b and by the first check), OLS residualisation (own folds), BH.

Inputs it reads (read-only): phase0*/attention_edges_layer_mean.npy (layer 8),
phase0*/attention_pair_counts.npy, phase0*/spearman_edges.npy, phase0*/gene_features.csv,
v2_eval/de_labels/<run>.npz, D1b order counts outputs/v2b_attention/order/order_counts_<run>.npy
(checked against invariants here; the token order itself was rebuilt twice already, by D1b
with the tokeniser and by the first check with its own rank code), D1b single-log variance
CSVs, the TRRUST TSV.

New checks not in D1b or the first check:
  * knockdown: target mean expression as a baseline (is F just "target is highly expressed"?)
  * does the rank-conditioned edge still carry order? (within-row Spearman of Ec with F)
  * TRRUST: leave-one-TF-out stability of the RPE1 symmetric null result
  * TRRUST: null test of the DIFFERENCE symmetric attention - co-expression
  * TRRUST: symmetric attention residualised on co-expression (+ gene features + F)

Subcommands:  kd | trvec <run> raw|resA|resB | trnull <run> d0 d1 | treval <run> | bh | fig | config
Outputs: outputs/v2b_attention/verification2/
"""
from __future__ import annotations

import os
import sys
sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJ = Path("<REPO_ROOT>/projects/maxtoki")
RD = PROJ / "runs/attention-grn-217M"
OUT = RD / "outputs"
D1B = OUT / "v2b_attention"
VER = D1B / "verification2"
VER.mkdir(parents=True, exist_ok=True)
SUF = {"217M_K562": "", "217M_RPE1": "_rpe1", "217M_Adamson": "_adamson", "1B_K562": "_k562_1b"}
RUNS = list(SUF)
LOGRUNS = {"217M_K562", "217M_Adamson", "1B_K562"}
TRRUST_TSV = Path("<DATA_ROOT>/biodyn-nmi-paper/"
                  "src/02_cssi_method/cssi_real_data/results/trrust_human.tsv")
LAYER = 8
SEED = 50505            # own seed; differs from D1b (20261001 + offsets) and first check (777001)
NB = 2000
NDRAW = 1000
SWAPS_PER_EDGE = 20     # accepted checkerboard swaps per draw = 20 x number of edges
FAM6 = ["forward", "transpose", "sym_mean", "sym_max", "rank_conditioned", "order_share_F"]


def p0(run):
    return OUT / f"phase0{SUF[run]}"


def jdump(o, p):
    def d(x):
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.bool_,)):
            return bool(x)
        raise TypeError(type(x))
    with open(p, "w") as f:
        json.dump(o, f, indent=1, default=d)


def ci(x):
    x = np.asarray(x, float)
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    run_min = 1.0
    for i in range(n - 1, -1, -1):
        run_min = min(run_min, p[o[i]] * n / (i + 1))
        q[o[i]] = run_min
    return q


# ------------------------------------------------------------------ scores
def load_scores(run):
    a = np.load(p0(run) / "attention_edges_layer_mean.npy", mmap_mode="r")
    E = np.array(a[LAYER], dtype=np.float64)
    del a
    G = E.shape[0]
    pc = np.load(p0(run) / "attention_pair_counts.npy").astype(np.int64)
    C = np.load(D1B / "order" / f"order_counts_{run}.npy").astype(np.int64)
    off = ~np.eye(G, dtype=bool)
    inv = {
        "C_nonneg": bool((C >= 0).all()),
        "C_le_pair_count": bool((C <= pc).all()),
        "C_plus_CT_eq_pair_count_offdiag": bool(np.array_equal((C + C.T)[off], pc[off])),
        "E_pos_where_C0_and_npair_pos": int(((C == 0) & (E > 0) & (pc > 0) & off).sum()),
        "E_zero_where_Cpos": int(((C > 0) & (E == 0) & off).sum()),
        "E_pos_where_npair0": int(((pc == 0) & (E > 0) & off).sum()),
    }
    np.fill_diagonal(E, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        F = np.where(pc > 0, C / pc, 0.0)
        Ec = np.where(C > 0, E / F, 0.0)          # = E * n_pair / C
    np.fill_diagonal(F, 0.0)
    np.fill_diagonal(Ec, 0.0)
    co = np.abs(np.load(p0(run) / "spearman_edges.npy").astype(np.float64))
    np.fill_diagonal(co, 0.0)
    S = {"forward": E, "transpose": E.T.copy(), "sym_mean": (E + E.T) / 2.0,
         "sym_max": np.maximum(E, E.T), "rank_conditioned": Ec, "order_share_F": F,
         "coexpr": co}
    gf = pd.read_csv(p0(run) / "gene_features.csv")
    base = {"variance": gf["variance"].to_numpy(float), "mean_expr": gf["mean_expr"].to_numpy(float),
            "dropout_neg": -gf["dropout_rate"].to_numpy(float)}
    if run in LOGRUNS:
        sl = pd.read_csv(D1B / "order" / f"gene_variance_single_log_{run}.csv")
        base["variance_single_log"] = sl["variance_single_log"].to_numpy(float)
    return S, base, gf, pc, C, inv


# ------------------------------------------------------------------ endpoint A
def cmd_kd():
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr, wilcoxon
    res = {}
    for ri, run in enumerate(RUNS):
        t0 = time.time()
        S, base, gf, pc, C, inv = load_scores(run)
        d = np.load(OUT / "v2_eval" / "de_labels" / f"{run}.npz")
        hs, Y = d["pert_hvg_idx"], d["labels"]
        G = S["forward"].shape[0]
        names = list(S) + list(base)
        A = {k: np.empty(len(hs)) for k in names}
        rho_ec_f, rho_ec_mean = [], []
        for i, (h, y) in enumerate(zip(hs, Y)):
            keep = np.arange(G) != h
            yy = y[keep]
            for k in S:
                A[k][i] = roc_auc_score(yy, S[k][h][keep])
            for k in base:
                A[k][i] = roc_auc_score(yy, base[k][keep])
            m = keep & (C[h] > 0)
            rho_ec_f.append(spearmanr(S["rank_conditioned"][h, m], S["order_share_F"][h, m]).correlation)
            rho_ec_mean.append(spearmanr(S["rank_conditioned"][h, m], gf["mean_expr"].to_numpy()[m]).correlation)
        # compare with D1b per-perturbation file
        dd = pd.read_csv(D1B / "knockdown" / f"per_perturbation_{run}.csv")
        mapn = {"forward": "forward", "transpose": "transpose", "sym_mean": "sym_mean", "sym_max": "sym_max",
                "rank_conditioned": "rank_conditioned", "order_share_F": "order_share_F",
                "coexpr": "coexpr_abs_spearman", "variance": "gene_variance",
                "variance_single_log": "gene_variance_single_log"}
        maxdiff = {k: float(np.max(np.abs(A[k] - dd[f"auc_{v}"].to_numpy()))) for k, v in mapn.items() if k in A}
        rng = np.random.default_rng(SEED + 10 * ri)
        n = len(hs)
        B = rng.integers(0, n, (NB, n))
        bm = {k: A[k][B].mean(1) for k in names}
        o = {"n_perturbations": n, "invariants": inv, "max_abs_diff_vs_D1b_per_pert": maxdiff,
             "median_within_row_spearman_Ec_vs_F": float(np.nanmedian(rho_ec_f)),
             "median_within_row_spearman_Ec_vs_target_mean": float(np.nanmedian(rho_ec_mean)),
             "scores": {}}
        for k in names:
            s = {"mean": float(A[k].mean()), "ci": ci(bm[k]),
                 "variance_minus": float(A["variance"].mean() - A[k].mean()), "variance_minus_ci": ci(bm["variance"] - bm[k])}
            if k != "forward":
                s["minus_forward"] = float(A[k].mean() - A["forward"].mean())
                s["minus_forward_ci"] = ci(bm[k] - bm["forward"])
            if k != "order_share_F":
                s["minus_F"] = float(A[k].mean() - A["order_share_F"].mean())
                s["minus_F_ci"] = ci(bm[k] - bm["order_share_F"])
            if k != "variance":
                s["wilcoxon_p_vs_variance"] = float(wilcoxon(A[k], A["variance"], zero_method="zsplit").pvalue)
            o["scores"][k] = s
        o["seconds"] = time.time() - t0
        res[run] = o
        print(run, {k: (round(v["mean"], 3), [round(x, 3) for x in v["ci"]]) for k, v in o["scores"].items()},
              maxdiff, f"{o['seconds']:.0f}s", flush=True)
    res["_method"] = {"auroc": "sklearn roc_auc_score per perturbation, perturbed gene removed; mean over perturbations",
                      "ci": f"percentile bootstrap over perturbations, {NB} reps, seed {SEED}+10*run_index"}
    jdump(res, VER / "knockdown_v2.json")


# ------------------------------------------------------------------ endpoint B
def trrust_edges(symbols):
    s2i = {s.upper(): i for i, s in enumerate(symbols)}
    tg = {}
    with open(TRRUST_TSV) as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 2:
                continue
            a, b = row[0].strip().upper(), row[1].strip().upper()
            if a in s2i and b in s2i and a != b:
                tg.setdefault(s2i[a], set()).add(s2i[b])
    tfs = sorted(i for i, t in tg.items() if len(t) >= 3)
    return tfs, {i: tg[i] for i in tfs}


class Pooled:
    """Pooled AUROC over TF blocks from a TF-by-TF count matrix: exact for any integer
    TF weights (bootstrap) — sum_{k,l} w_k w_l A_kl / sum_{k,l} w_k w_l npos_k nneg_l."""

    def __init__(self, blocks_s, blocks_y):
        nT = len(blocks_s)
        self.npos = np.array([y.sum() for y in blocks_y], float)
        self.nneg = np.array([(~y).sum() for y in blocks_y], float)
        negs = [np.sort(s[~y]) for s, y in zip(blocks_s, blocks_y)]
        A = np.zeros((nT, nT))
        for k in range(nT):
            ps = blocks_s[k][blocks_y[k]]
            for l in range(nT):
                lo = np.searchsorted(negs[l], ps, "left")
                hi = np.searchsorted(negs[l], ps, "right")
                A[k, l] = (lo + 0.5 * (hi - lo)).sum()
        self.A = A

    def __call__(self, w=None):
        if w is None:
            w = np.ones(len(self.npos))
        return float(w @ self.A @ w / ((w * self.npos).sum() * (w * self.nneg).sum()))


def trrust_setup(run):
    S, base, gf, pc, C, inv = load_scores(run)
    G = len(gf)
    tfs, targets = trrust_edges(list(gf["symbol"].astype(str)))
    cand = [np.delete(np.arange(G), t) for t in tfs]
    ylist = [np.isin(c, sorted(targets[t])) for c, t in zip(cand, tfs)]
    return S, base, gf, pc, C, inv, G, tfs, targets, cand, ylist


def cmd_trvec(run, part):
    """Stage 1: pooled raw scores and residualised scores (own OLS via normal equations,
    own 5-fold KFold over all ordered off-diagonal pairs). part = raw | resA | resB."""
    from sklearn.model_selection import KFold
    t0 = time.time()
    ri = RUNS.index(run)
    S, base, gf, pc, C, inv, G, tfs, targets, cand, ylist = trrust_setup(run)
    out = {}
    if part == "raw":
        for k in S:
            out[k] = np.concatenate([S[k][t, c] for t, c in zip(tfs, cand)])
        for k in base:
            out[k] = np.concatenate([base[k][c] for c in cand])
        out["_y"] = np.concatenate(ylist)
        np.savez(VER / f"vec_raw_{run}.npz", **out)
        jdump(inv, VER / f"invariants_{run}.json")
        print(run, "raw", len(out["_y"]), int(out["_y"].sum()), f"{time.time() - t0:.0f}s")
        return
    ii, jj = np.nonzero(~np.eye(G, dtype=bool))
    gm, gv, gd = (gf[c].to_numpy(float) for c in ["mean_expr", "variance", "dropout_rate"])
    gene6 = np.column_stack([gm[ii], gv[ii], gd[ii], gm[jj], gv[jj], gd[jj]])
    Fx = np.column_stack([S["order_share_F"][ii, jj], (pc[ii, jj] == 0).astype(float)])
    cox = S["coexpr"][ii, jj][:, None]
    folds = list(KFold(5, shuffle=True, random_state=SEED + ri).split(ii))
    flatmap = np.full((G, G), -1, dtype=np.int64)
    flatmap[ii, jj] = np.arange(len(ii))
    pf = np.concatenate([flatmap[t, c] for t, c in zip(tfs, cand)])

    def ols_resid(t, X):
        r = np.empty_like(t)
        for tr, te in folds:
            mu, sd = X[tr].mean(0), X[tr].std(0)
            keepc = sd > 0                       # drop constant columns (e.g. no n_pair = 0 pairs)
            mu, sd = mu[keepc], sd[keepc]
            Z = np.column_stack([np.ones(len(tr)), (X[tr][:, keepc] - mu) / sd])
            b = np.linalg.solve(Z.T @ Z, Z.T @ t[tr])
            r[te] = t[te] - np.column_stack([np.ones(len(te)), (X[te][:, keepc] - mu) / sd]) @ b
        return r[pf]
    if part == "resA":       # D1b main model, own folds
        XA = np.column_stack([gene6, Fx])
        for k in ["forward", "transpose", "sym_mean", "sym_max", "rank_conditioned", "coexpr"]:
            out[f"{k}|gene6+F"] = ols_resid(S[k][ii, jj], XA)
            print(k, f"{time.time() - t0:.0f}s", flush=True)
        out["order_share_F|gene6"] = ols_resid(S["order_share_F"][ii, jj], gene6)
    elif part == "resB":     # new: remove co-expression from attention
        XB = np.column_stack([cox, gene6, Fx])
        qs = np.quantile(cox[:, 0], np.linspace(0, 1, 51))
        bins = np.clip(np.searchsorted(qs, cox[:, 0], "right") - 1, 0, 49)
        for k in ["transpose", "sym_mean", "sym_max", "forward"]:
            t = S[k][ii, jj]
            out[f"{k}|coexpr_ols"] = ols_resid(t, cox)
            out[f"{k}|coexpr+gene6+F"] = ols_resid(t, XB)
            bm = np.bincount(bins, weights=t, minlength=50) / np.maximum(np.bincount(bins, minlength=50), 1)
            out[f"{k}|coexpr_50bins"] = (t - bm[bins])[pf]
            print(k, f"{time.time() - t0:.0f}s", flush=True)
    np.savez(VER / f"vec_{part}_{run}.npz", **out)
    print(run, part, f"{time.time() - t0:.0f}s")


def cmd_trnull(run, d0, d1):
    """Stage 2: degree-preserving null by CHECKERBOARD swaps on the TF x gene 0/1 matrix.
    The cell (TF, its own gene) is forbidden. Each draw starts from the observed network
    and makes SWAPS_PER_EDGE x n_edges ACCEPTED swaps. Draw d uses its own generator
    default_rng([SEED, 300 + run_index, d]), so draws can be made in chunks."""
    t0 = time.time()
    ri = RUNS.index(run)
    _, _, gf, _, _, _, G, tfs, targets, cand, ylist = trrust_setup(run)
    obs = [(k, c) for k, t in enumerate(tfs) for c in sorted(targets[t])]
    ne = len(obs)
    tfg = [int(t) for t in tfs]
    need = SWAPS_PER_EDGE * ne
    res = []
    att_tot = acc_tot = 0
    for d in range(d0, d1):
        rng = np.random.default_rng([SEED, 300 + ri, d])
        edges = list(obs)
        es = set(edges)
        acc = 0
        while acc < need:
            I = rng.integers(0, ne, (4 * need, 2))
            for i, j in I:
                att_tot += 1
                (k1, c1), (k2, c2) = edges[i], edges[j]
                if k1 == k2 or c1 == c2 or (k1, c2) in es or (k2, c1) in es or c2 == tfg[k1] or c1 == tfg[k2]:
                    continue
                es.discard((k1, c1)); es.discard((k2, c2))
                es.add((k1, c2)); es.add((k2, c1))
                edges[i] = (k1, c2); edges[j] = (k2, c1)
                acc += 1
                if acc == need:
                    break
        acc_tot += acc
        k = np.array([e[0] for e in edges]); c = np.array([e[1] for e in edges])
        tk = np.array(tfg)[k]
        res.append(np.sort(k * (G - 1) + c - (c > tk)))
    np.save(VER / f"null_chunk_{run}_{d0:04d}_{d1:04d}.npy", np.array(res, dtype=np.int32))
    jdump({"attempts": att_tot, "accepted": acc_tot}, VER / f"null_chunk_{run}_{d0:04d}_{d1:04d}.json")
    print(run, d0, d1, f"acc rate {acc_tot / att_tot:.3f}", f"{time.time() - t0:.0f}s")


def cmd_treval(run):
    """Stage 3: pooled AUROC, TF-bootstrap CIs, null z / p, differences, leave-one-TF-out."""
    from scipy.stats import rankdata
    from sklearn.metrics import roc_auc_score
    t0 = time.time()
    ri = RUNS.index(run)
    _, _, gf, _, _, _, G, tfs, targets, cand, ylist = trrust_setup(run)
    nT = len(tfs)
    vec = {}
    for part in ["raw", "resA", "resB"]:
        z = np.load(VER / f"vec_{part}_{run}.npz")
        vec.update({k: z[k] for k in z.files})
    y = vec.pop("_y").astype(bool)
    chunks = sorted(VER.glob(f"null_chunk_{run}_*.npy"))
    pos = np.concatenate([np.load(p) for p in chunks]).astype(np.int64)
    meta = [json.load(open(str(p)[:-4] + ".json")) for p in chunks]
    assert pos.shape[0] == NDRAW, pos.shape
    # null checks
    obs = np.flatnonzero(y)
    tf_of = pos // (G - 1)
    tfa = np.array(tfs)
    col = pos - tf_of * (G - 1)
    col = col + (col >= tfa[tf_of])                     # back to gene index
    obs_tf = obs // (G - 1)
    obs_col = obs - obs_tf * (G - 1); obs_col = obs_col + (obs_col >= tfa[obs_tf])
    rows_ok = all(np.array_equal(np.bincount(t, minlength=nT), np.bincount(obs_tf, minlength=nT)) for t in tf_of)
    cols_ok = all(np.array_equal(np.bincount(c, minlength=G), np.bincount(obs_col, minlength=G)) for c in col)
    forb = int((col == tfa[tf_of]).any(1).sum())
    so = set(obs.tolist())
    jac = np.array([1 - len(so & set(p.tolist())) / len(so | set(p.tolist())) for p in pos])
    off = np.cumsum([0] + [len(c) for c in cand])
    blocks = lambda v: [v[off[k]:off[k + 1]] for k in range(nT)]  # noqa: E731
    PA = {k: Pooled(blocks(v), ylist) for k, v in vec.items()}
    auc = {k: PA[k]() for k in vec}
    sk = max(abs(auc[k] - roc_auc_score(y, vec[k])) for k in ["forward", "sym_mean", "coexpr", "variance"])
    rng = np.random.default_rng(SEED + 100 + ri)
    W = np.array([np.bincount(rng.integers(0, nT, nT), minlength=nT) for _ in range(NB)], float)
    boot = {k: np.array([PA[k](w) for w in W]) for k in vec}

    def null_auc(v, P_):
        r = rankdata(v)
        n1 = P_.shape[1]
        return (r[P_].sum(1) - n1 * (n1 + 1) / 2) / (n1 * (len(v) - n1))
    nulls = {k: null_auc(v, pos) for k, v in vec.items()}

    def zt(o, nv):
        sd = nv.std(ddof=1)
        if sd < 1e-12:
            return {"null_mean": float(nv.mean()), "null_sd": 0.0, "z": None, "p": None}
        return {"null_mean": float(nv.mean()), "null_sd": float(sd), "z": float((o - nv.mean()) / sd),
                "p": float((1 + (nv >= o - 1e-12).sum()) / (1 + len(nv)))}
    out = {"run": run, "n_tfs": nT, "n_pos": int(y.sum()), "n_neg": int((~y).sum()), "sklearn_check": sk,
           "null_checks": {"n_draws": int(len(pos)), "rows_kept": rows_ok, "cols_kept": cols_ok,
                           "draws_with_forbidden_cell": forb, "draws_equal_observed": int((jac == 0).sum()),
                           "mean_jaccard": float(jac.mean()), "min_jaccard": float(jac.min()),
                           "swap_acceptance": sum(m["accepted"] for m in meta) / sum(m["attempts"] for m in meta)},
           "scores": {}}
    rawk = [k for k in vec if "|" not in k]
    for k in vec:
        o = {"auroc": auc[k], "ci": ci(boot[k])}
        o.update(zt(auc[k], nulls[k]))
        if k in rawk and k not in ("forward",):
            o["minus_forward"] = auc[k] - auc["forward"]; o["minus_forward_ci"] = ci(boot[k] - boot["forward"])
        if k != "coexpr":
            o["minus_coexpr"] = auc[k] - auc["coexpr"]; o["minus_coexpr_ci"] = ci(boot[k] - boot["coexpr"])
            o["minus_coexpr_null"] = zt(auc[k] - auc["coexpr"], nulls[k] - nulls["coexpr"])
        out["scores"][k] = o
    # leave-one-TF-out
    kd_of = np.concatenate([np.full(len(c), k) for k, c in enumerate(cand)])
    loto = {}
    for k in ["forward", "transpose", "sym_mean", "sym_max", "coexpr"]:
        rows = []
        for drop in range(nT):
            keep = kd_of != drop
            newidx = np.cumsum(keep) - 1
            v = vec[k][keep]
            m = tf_of != drop
            pnew = np.array([newidx[p[mm]] for p, mm in zip(pos, m)])
            a = float(roc_auc_score(y[keep], v))
            nv = null_auc(v, pnew)
            rows.append({"dropped_tf": str(gf["symbol"].iloc[tfs[drop]]), "n_targets": len(targets[tfs[drop]]),
                         "auroc": a, "z": float((a - nv.mean()) / nv.std(ddof=1)),
                         "p": float((1 + (nv >= a - 1e-12).sum()) / (1 + len(nv)))})
        loto[k] = {"min_z": min(r["z"] for r in rows), "max_z": max(r["z"] for r in rows),
                   "max_p": max(r["p"] for r in rows), "rows": rows}
    out["leave_one_tf_out"] = loto
    out["seconds"] = time.time() - t0
    jdump(out, VER / f"trrust_{run}.json")
    print(run, out["n_tfs"], out["n_pos"], "sk", sk, out["null_checks"])
    for k, o in out["scores"].items():
        print(f"  {k:28s} {o['auroc']:.3f} [{o['ci'][0]:.3f},{o['ci'][1]:.3f}] null {o['null_mean']:.3f} z "
              + ("  -  " if o["z"] is None else f"{o['z']:+.2f} p {o['p']:.3f}")
              + (f" | -co {o['minus_coexpr']:+.3f} [{o['minus_coexpr_ci'][0]:+.3f},{o['minus_coexpr_ci'][1]:+.3f}]"
                 + ("" if o['minus_coexpr_null']['z'] is None else f" z {o['minus_coexpr_null']['z']:+.2f} p {o['minus_coexpr_null']['p']:.3f}")
                 if "minus_coexpr" in o else "")
              + (f" | -fwd {o['minus_forward']:+.3f} [{o['minus_forward_ci'][0]:+.3f},{o['minus_forward_ci'][1]:+.3f}]" if "minus_forward" in o else ""))
    print("  LOTO", {k: (round(v["min_z"], 2), round(v["max_z"], 2), round(v["max_p"], 3)) for k, v in loto.items()})
    print(f"{out['seconds']:.0f}s")


def cmd_extra():
    """How much do symmetric attention and |Spearman| co-expression overlap? Spearman over
    all ordered off-diagonal pairs and over the pooled TRRUST pairs; co-expression with
    symmetric attention removed (50 quantile bins of attention); null z with the own
    checkerboard draws."""
    from scipy.stats import rankdata, spearmanr
    out = {}
    for ri, run in enumerate(RUNS):
        S, base, gf, pc, C, inv, G, tfs, targets, cand, ylist = trrust_setup(run)
        ii, jj = np.nonzero(~np.eye(G, dtype=bool))
        y = np.concatenate(ylist)
        pos = np.concatenate([np.load(p) for p in sorted(VER.glob(f"null_chunk_{run}_*.npy"))]).astype(np.int64)
        flatmap = np.full((G, G), -1, dtype=np.int64)
        flatmap[ii, jj] = np.arange(len(ii))
        pf = np.concatenate([flatmap[t, c] for t, c in zip(tfs, cand)])
        o = {}
        for k in ["sym_mean", "sym_max", "forward", "transpose"]:
            a = S[k][ii, jj]; co = S["coexpr"][ii, jj]
            o[f"spearman_{k}_vs_coexpr_all_pairs"] = float(spearmanr(a, co).correlation)
            o[f"spearman_{k}_vs_coexpr_trrust_pairs"] = float(spearmanr(a[pf], co[pf]).correlation)
            qs = np.quantile(a, np.linspace(0, 1, 51))
            bins = np.clip(np.searchsorted(qs, a, "right") - 1, 0, 49)
            bm = np.bincount(bins, weights=co, minlength=50) / np.maximum(np.bincount(bins, minlength=50), 1)
            v = (co - bm[bins])[pf]
            r = rankdata(v); n1 = int(y.sum()); N = len(v) - n1
            auc = (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * N)
            nv = (r[pos].sum(1) - n1 * (n1 + 1) / 2) / (n1 * N)
            o[f"coexpr_minus_{k}_50bins"] = {"auroc": float(auc), "null_mean": float(nv.mean()),
                                             "z": float((auc - nv.mean()) / nv.std(ddof=1)),
                                             "p": float((1 + (nv >= auc - 1e-12).sum()) / (1 + len(nv)))}
        out[run] = o
        print(run, json.dumps(o))
    jdump(out, VER / "overlap_attention_coexpr.json")


def cmd_bh():
    R = {r: json.load(open(VER / f"trrust_{r}.json")) for r in RUNS}
    fam = [(r, k) for r in RUNS for k in FAM6]
    q = bh([R[r]["scores"][k]["p"] for r, k in fam])
    resfam = [(r, (k + "|gene6+F") if k != "order_share_F" else "order_share_F|gene6") for r in RUNS for k in FAM6]
    qr = bh([R[r]["scores"][k]["p"] for r, k in resfam])
    out = {"raw_24": {f"{r}:{k}": {"z": R[r]["scores"][k]["z"], "p": R[r]["scores"][k]["p"], "q": float(qq)}
                      for (r, k), qq in zip(fam, q)},
           "resid_24": {f"{r}:{k}": {"auroc": R[r]["scores"][k]["auroc"], "z": R[r]["scores"][k]["z"],
                                     "p": R[r]["scores"][k]["p"], "q": float(qq)} for (r, k), qq in zip(resfam, qr)}}
    # D1b q recomputed from D1b p
    d = json.load(open(D1B / "trrust" / "trrust_variants_summary.json"))
    pd1 = [d[r]["scores"][k]["null"]["p_one_sided"] for r, k in fam]
    qd1 = bh(pd1)
    out["D1b_q_recomputed_max_abs_diff"] = float(max(abs(a - d[r]["scores"][k]["null"]["q_bh_all_24_tests"])
                                                    for (r, k), a in zip(fam, qd1)))
    jdump(out, VER / "bh.json")
    for k, v in out["raw_24"].items():
        if v["q"] < 0.2:
            print("raw", k, round(v["z"], 2), v["p"], round(v["q"], 3))
    for k, v in out["resid_24"].items():
        if v["q"] < 0.2:
            print("resid", k, round(v["auroc"], 3), round(v["z"], 2), v["p"], round(v["q"], 3))
    print("D1b q recomputed diff", out["D1b_q_recomputed_max_abs_diff"])


def cmd_fig():
    """Check fig_attention_variants.csv against own recomputation (knockdown means and
    TRRUST pooled AUROCs) and check its structure."""
    fig = pd.read_csv(D1B / "fig_attention_variants.csv")
    kd = json.load(open(VER / "knockdown_v2.json"))
    tr = {r: json.load(open(VER / f"trrust_{r}.json")) for r in RUNS}
    mapn = {"forward": "forward", "transpose": "transpose", "sym_mean": "sym_mean", "sym_max": "sym_max",
            "rank_conditioned": "rank_conditioned", "order_share_F": "order_share_F",
            "coexpr_abs_spearman": "coexpr", "gene_variance": "variance",
            "gene_variance_single_log": "variance_single_log"}
    diffs = []
    for _, r in fig.iterrows():
        if r["score"] not in mapn:
            continue
        mine = None
        if r["endpoint"] == "knockdown" and r["measure"] == "mean per-perturbation AUROC":
            mine = kd[r["run"]]["scores"][mapn[r["score"]]]["mean"]
        if r["endpoint"] == "trrust" and r["measure"] == "pooled AUROC":
            mine = tr[r["run"]]["scores"][mapn[r["score"]]]["auroc"]
        if mine is not None:
            diffs.append(abs(mine - r["value"]))
    struct = {
        "n_rows": int(len(fig)),
        "columns": list(fig.columns),
        "endpoints": sorted(fig["endpoint"].unique().tolist()),
        "measures": sorted(fig["measure"].unique().tolist()),
        "every_variant_and_F_in_knockdown_all_runs": bool(all(
            ((fig.run == r) & (fig.endpoint == "knockdown") & (fig.score == s)).any() for r in RUNS for s in FAM6)),
        "every_variant_and_F_in_trrust_all_runs": bool(all(
            ((fig.run == r) & (fig.endpoint == "trrust") & (fig.score == s) & (fig.measure == "pooled AUROC")).any()
            for r in RUNS for s in FAM6)),
        "rows_with_ci_but_no_ci_method": int((fig.ci_low.notna() & fig.ci_method.isna()).sum()),
        "ci_methods": sorted(fig["ci_method"].dropna().unique().tolist()),
        "n_values_compared_with_own_recomputation": len(diffs),
        "max_abs_diff_vs_own": float(max(diffs)) if diffs else None,
    }
    jdump(struct, VER / "figcheck.json")
    print(json.dumps(struct, indent=1))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def cmd_config():
    cache_p = VER / "_sha_cache.json"
    cache = json.load(open(cache_p)) if cache_p.exists() else {}

    def h(p):
        p = Path(p)
        key = f"{p}|{p.stat().st_size}|{p.stat().st_mtime}"
        if key not in cache:
            cache[key] = sha(p)
            json.dump(cache, open(cache_p, "w"))
        return {"path": str(p), "bytes": p.stat().st_size, "sha256": cache[key]}
    ins = [TRRUST_TSV]
    for r in RUNS:
        ins += [p0(r) / "attention_edges_layer_mean.npy", p0(r) / "attention_pair_counts.npy",
                p0(r) / "spearman_edges.npy", p0(r) / "gene_features.csv",
                OUT / "v2_eval" / "de_labels" / f"{r}.npz", D1B / "order" / f"order_counts_{r}.npy",
                D1B / "knockdown" / f"per_perturbation_{r}.csv"]
        if r in LOGRUNS:
            ins.append(D1B / "order" / f"gene_variance_single_log_{r}.csv")
    ins += [D1B / "trrust" / "trrust_variants_summary.json", D1B / "fig_attention_variants.csv"]
    outs = sorted(p for p in VER.iterdir() if p.is_file() and not p.name.startswith(("._", "_sha")) and p.name != "run_config.json")
    import sklearn
    import scipy
    cfg = {"item": "D1b attention, second independent check", "device": "CPU only; no model, no tokeniser call",
           "python": sys.version.split()[0],
           "packages": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                        "scikit-learn": sklearn.__version__},
           "seeds": {"master": SEED, "knockdown_bootstrap": "SEED+10*run_index", "trrust_bootstrap": "SEED+100+run_index",
                     "checkerboard_null": "SEED+200+run_index", "kfold": "SEED+run_index"},
           "parameters": {"layer": LAYER, "n_bootstrap": NB, "null_draws": NDRAW,
                          "accepted_swaps_per_draw": f"{SWAPS_PER_EDGE} x n_edges"},
           "run_order": RUNS,
           "inputs": [h(p) for p in ins], "script": h(Path(__file__)), "outputs": [h(p) for p in outs]}
    jdump(cfg, VER / "run_config.json")
    print("inputs", len(cfg["inputs"]), "outputs", len(cfg["outputs"]))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "kd":
        cmd_kd()
    elif cmd == "trvec":
        cmd_trvec(sys.argv[2], sys.argv[3])
    elif cmd == "trnull":
        cmd_trnull(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "treval":
        cmd_treval(sys.argv[2])
    elif cmd == "extra":
        cmd_extra()
    elif cmd == "bh":
        cmd_bh()
    elif cmd == "fig":
        cmd_fig()
    elif cmd == "config":
        cmd_config()
    else:
        raise SystemExit(__doc__)
