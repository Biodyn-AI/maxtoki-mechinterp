"""v2b_attention_verify — independent check of revision item D1b (attention direction and
rank baselines). CPU only. No model forward pass, no model weights.

Written separately from v2b_attention.py. It does NOT import v2b_attention.py or
v2_common.py. Own code for: token order, the order counts C, AUROC (sklearn), bootstraps
(own seeds, explicit resampling), TRRUST matrix, a boolean-matrix Curveball null (own
sampler, own seed), OLS residualisation (own fold split), BH.

Subcommands
  order <run>        rebuild token order with own rank code -> C; compare with D1b C
  knockdown          Endpoint A for all variants, all runs
  trrust_scores <run>   Endpoint B scores: raw, gene baselines, own gene model, own OLS residuals
  trrust_null <run>     own boolean Curveball null networks
  trrust_eval <run>     pooled AUROC, TF-bootstrap CIs, null z / p, differences
  bh                 BH recomputation from D1b JSONs and from own p-values
  figcheck           fig/table CSV values vs D1b JSONs
  config             run_config.json for verification outputs
Outputs: outputs/v2b_attention/verification/
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJ = Path("<REPO_ROOT>/projects/maxtoki")
RD = PROJ / "runs/attention-grn-217M"
OUT = RD / "outputs"
D1B = OUT / "v2b_attention"
VER = D1B / "verification"
VER.mkdir(parents=True, exist_ok=True)
SUF = {"217M_K562": "", "217M_RPE1": "_rpe1", "217M_Adamson": "_adamson", "1B_K562": "_k562_1b"}
DSKEY = {"217M_K562": "k562", "217M_RPE1": "rpe1", "217M_Adamson": "adamson", "1B_K562": "k562"}
RUNS = list(SUF)
TRRUST_TSV = Path("<DATA_ROOT>/biodyn-nmi-paper/"
                  "src/02_cssi_method/cssi_real_data/results/trrust_human.tsv")
LAYER = 8
MYSEED = 777001          # own seed, different from the D1b / v2 seeds
NB = 2000


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
        raise TypeError(type(x))
    json.dump(o, open(p, "w"), indent=1, default=d)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ order (own code)
def cmd_order(run):
    import h5py
    t0 = time.time()
    sys.path.insert(0, str(PROJ / "setup"))
    from dataset_loader import resolve      # metadata only (path, var ids)
    ds = resolve(DSKEY[run])
    tokd = json.load(open(PROJ / "setup/token_dictionary.json"))
    med = pickle.load(open("<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/"
                           "crispri_validation/data/gene_median_dictionary_gc104M.pkl", "rb"))
    keep = np.array([i for i, e in enumerate(ds.var_ensembl) if isinstance(e, str) and e in tokd], dtype=np.int64)
    medv = np.array([med[ds.var_ensembl[i]] for i in keep], dtype=np.float32)
    hv = pd.read_csv(p0(run) / "hvg_gene_table.csv")
    G = len(hv)
    var2h = -np.ones(ds.n_genes_total, dtype=np.int64)
    var2h[hv["var_idx"].to_numpy()] = np.arange(G)
    rows = pd.read_csv(p0(run) / "control_cells.csv")["cell_idx_global"].to_numpy(np.int64)
    C = np.zeros((G, G), dtype=np.int32)
    PC = np.zeros((G, G), dtype=np.int32)
    with h5py.File(ds.h5_path, "r") as f:
        Xd = f["X"]
        dense = isinstance(Xd, h5py.Dataset)
        if not dense:
            indptr = Xd["indptr"][:]
        for ci, r in enumerate(rows):
            if dense:
                x = np.asarray(Xd[r, :], dtype=np.float32)
            else:
                x = np.zeros(ds.n_genes_total, np.float32)
                a, b = int(indptr[r]), int(indptr[r + 1])
                x[Xd["indices"][a:b]] = Xd["data"][a:b].astype(np.float32)
            e = x[keep]
            nz = np.flatnonzero(e > 0)
            v = e[nz] / medv[nz]
            v = np.nan_to_num(v, nan=0.0, posinf=0.0)
            o = np.argsort(-v)[:2046]                 # max_len 2048 minus bos/eos
            seq_var = keep[nz[o]]                     # var index in sequence order
            h = var2h[seq_var]
            h = h[h >= 0]                             # HVGs in sequence order
            k = len(h)
            if k < 2:
                continue
            # C[later, earlier] += 1 : strictly lower triangle in sequence order
            L = np.tril(np.ones((k, k), dtype=np.int32), -1)
            C[np.ix_(h, h)] += L
            PC[np.ix_(h, h)] += 1
    np.fill_diagonal(C, 0)
    Cd = np.load(D1B / "order" / f"order_counts_{run}.npy")
    pcs = np.load(p0(run) / "attention_pair_counts.npy")
    E = np.load(p0(run) / "attention_edges_layer_mean.npy", mmap_mode="r")[LAYER].astype(np.float64)
    off = ~np.eye(G, dtype=bool)
    m = off & (pcs > 0)
    # per-head zero pattern on one layer too (stronger than head mean)
    out = {"run": run,
           "own_C_equals_D1b_C": bool(np.array_equal(C, Cd)),
           "n_C_mismatch": int((C != Cd).sum()),
           "own_pair_counts_equal_saved_offdiag": bool(np.array_equal(PC[off], pcs[off])),
           "own_C_plus_CT_equals_saved_pc_offdiag": bool(np.array_equal((C + C.T)[off], pcs[off])),
           "E_pos_where_C0": int(((C == 0) & (E > 0) & m).sum()),
           "E_zero_where_Cpos": int(((C > 0) & (E == 0) & m).sum()),
           "frac_pairs_C0": float(((C == 0) & m).sum() / m.sum()),
           "seconds": time.time() - t0}
    # exact identity E = F * Ec only needs E = 0 where C = 0; check direction convention too:
    # mean over pairs with 0 < C < n_pair of E / (C / n_pair) should be positive and finite
    F = np.where(pcs > 0, C / np.maximum(pcs, 1), 0.0)
    mm = m & (C > 0)
    lE, lF = np.log(E[mm]), np.log(F[mm])
    b = np.polyfit(lF, lE, 1)
    out["r2_logE_on_logF"] = float(1 - np.var(lE - np.polyval(b, lF)) / np.var(lE))
    out["slope_logE_on_logF"] = float(b[0])
    from scipy.stats import spearmanr
    out["spearman_E_F_all"] = float(spearmanr(E[m], F[m]).correlation)
    jdump(out, VER / f"order_{run}.json")
    print(json.dumps(out))


# ------------------------------------------------------------------ scores (own build)
def scores(run):
    E = np.load(p0(run) / "attention_edges_layer_mean.npy", mmap_mode="r")[LAYER].astype(np.float64)
    np.fill_diagonal(E, 0.0)
    pc = np.load(p0(run) / "attention_pair_counts.npy").astype(np.float64)
    C = np.load(D1B / "order" / f"order_counts_{run}.npy").astype(np.float64)   # checked by `order`
    F = np.divide(C, pc, out=np.zeros_like(C), where=pc > 0)
    Ec = np.divide(E * pc, C, out=np.zeros_like(C), where=C > 0)
    np.fill_diagonal(F, 0); np.fill_diagonal(Ec, 0)
    sp = np.abs(np.load(p0(run) / "spearman_edges.npy").astype(np.float64)); np.fill_diagonal(sp, 0)
    return {"forward": E, "transpose": E.T.copy(), "sym_mean": (E + E.T) / 2, "sym_max": np.maximum(E, E.T),
            "rank_conditioned": Ec, "order_share_F": F, "coexpr_abs_spearman": sp}, pc


def gene_var(run):
    gf = pd.read_csv(p0(run) / "gene_features.csv")
    return gf


# ------------------------------------------------------------------ Endpoint A
def cmd_knockdown():
    from sklearn.metrics import roc_auc_score
    res = {}
    for ri, run in enumerate(RUNS):
        S, pc = scores(run)
        gf = gene_var(run)
        S["gene_variance"] = np.tile(gf["variance"].to_numpy(np.float64), (len(gf), 1))
        d = np.load(OUT / "v2_eval" / "de_labels" / f"{run}.npz")
        hs, Y = d["pert_hvg_idx"], d["labels"].astype(bool)
        G = Y.shape[1]
        A = {k: np.array([roc_auc_score(y[np.arange(G) != h], M[h][np.arange(G) != h]) for h, y in zip(hs, Y)])
             for k, M in S.items()}
        n = len(hs)
        rng = np.random.default_rng(MYSEED + ri)
        idx = rng.integers(0, n, (NB, n))
        o = {"n_perts": int(n)}
        for k in A:
            o[k] = {"mean": float(A[k].mean()), "ci": np.percentile(A[k][idx].mean(1), [2.5, 97.5]).tolist(),
                    "minus_forward": float(A[k].mean() - A["forward"].mean()),
                    "minus_forward_ci": np.percentile((A[k] - A["forward"])[idx].mean(1), [2.5, 97.5]).tolist(),
                    "variance_minus": float(A["gene_variance"].mean() - A[k].mean()),
                    "variance_minus_ci": np.percentile((A["gene_variance"] - A[k])[idx].mean(1), [2.5, 97.5]).tolist(),
                    "n_score_gt_forward": int((A[k] > A["forward"]).sum())}
        # compare with D1b per-perturbation file
        dp = pd.read_csv(D1B / "knockdown" / f"per_perturbation_{run}.csv")
        o["max_abs_diff_vs_D1b_per_pert"] = {k: float(np.max(np.abs(A[k] - dp[f"auc_{k}"]))) for k in A if f"auc_{k}" in dp}
        res[run] = o
        print(run, {k: round(v["mean"], 4) for k, v in o.items() if isinstance(v, dict) and "mean" in v},
              o["max_abs_diff_vs_D1b_per_pert"], flush=True)
    res["_method"] = {"auroc": "sklearn roc_auc_score per perturbation (perturbed gene removed)",
                      "ci": f"percentile bootstrap over perturbations, {NB} reps, seed {MYSEED}+run_index (own seed)"}
    jdump(res, VER / "knockdown_verify.json")


# ------------------------------------------------------------------ Endpoint B
def trrust_pairs(run):
    gf = gene_var(run)
    sym = [s.upper() for s in gf["symbol"]]
    s2i = {s: i for i, s in enumerate(sym)}
    t = pd.read_csv(TRRUST_TSV, sep="\t", header=None, usecols=[0, 1])
    G = len(sym)
    M = np.zeros((G, G), bool)
    for a, b in zip(t[0].str.upper(), t[1].str.upper()):
        if a in s2i and b in s2i and s2i[a] != s2i[b]:
            M[s2i[a], s2i[b]] = True
    tfs = np.array(sorted(i for i in range(G) if M[i].sum() >= 3 and sym[i] in set(t[0].str.upper())))
    return gf, M, tfs


def curveball_bool(B, forb_col, n_trades, rng):
    """Own Curveball on a boolean (rows x cols) matrix. forb_col[r] = column never allowed in row r."""
    B = B.copy()
    R = B.shape[0]
    for _ in range(n_trades):
        a, b = rng.choice(R, 2, replace=False)
        ra, rb = B[a], B[b]
        only_a = np.flatnonzero(ra & ~rb)
        only_b = np.flatnonzero(rb & ~ra)
        # a column forbidden for the other row cannot move
        mov_a = only_a[only_a != forb_col[b]]
        mov_b = only_b[only_b != forb_col[a]]
        pool = np.concatenate([mov_a, mov_b])
        if len(pool) == 0:
            continue
        rng.shuffle(pool)
        na = len(mov_a)
        B[a, mov_a] = False; B[b, mov_b] = False
        B[a, pool[:na]] = True; B[b, pool[na:]] = True
    return B


def _trrust_base(run):
    S, pc = scores(run)
    gf, M, tfs = trrust_pairs(run)
    G = M.shape[0]
    nT = len(tfs)
    cand = np.array([j for k in range(nT) for j in range(G) if j != tfs[k]])
    kk = np.repeat(np.arange(nT), G - 1)
    src = tfs[kk]
    y = M[src, cand]
    return S, pc, gf, M, tfs, G, nT, cand, kk, src, y


def cmd_trrust_scores(run):
    """All pooled-pair scores (raw, gene baselines, own gene model, own OLS residuals) -> npz."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    t0 = time.time()
    ri = RUNS.index(run)
    S, pc, gf, M, tfs, G, nT, cand, kk, src, y = _trrust_base(run)
    sc = {k: V[src, cand] for k, V in S.items()}
    sc["gene_variance"] = gf["variance"].to_numpy(np.float64)[cand]
    if run != "217M_RPE1":
        sl = pd.read_csv(D1B / "order" / f"gene_variance_single_log_{run}.csv")["variance_single_log"].to_numpy()
        sc["gene_variance_single_log"] = sl[cand]
    # 3-feature gene model, own code: logistic regression, 5 folds grouped by TF (own random grouping)
    Xg = np.column_stack([gf[c].to_numpy(np.float64)[cand] for c in ["mean_expr", "variance", "dropout_rate"]])
    fold_of_tf = np.random.default_rng(MYSEED + 50 + ri).permutation(np.arange(nT) % 5)
    oof = np.zeros(len(y))
    for f in range(5):
        te = fold_of_tf[kk] == f
        ss = StandardScaler().fit(Xg[~te])
        lr = LogisticRegression(max_iter=3000, class_weight="balanced").fit(ss.transform(Xg[~te]), y[~te])
        oof[te] = lr.predict_proba(ss.transform(Xg[te]))[:, 1]
    sc["gene_model_3feat_owngroups"] = oof
    # residualisation (own): OLS on gene6 (+ F + indicator n_pair = 0 for edges), own 5-fold split
    ii, jj = np.nonzero(~np.eye(G, dtype=bool))
    gm, gv, gd = (gf[c].to_numpy(np.float64) for c in ["mean_expr", "variance", "dropout_rate"])
    X6 = np.column_stack([gm[ii], gv[ii], gd[ii], gm[jj], gv[jj], gd[jj]])
    XF = np.column_stack([X6, S["order_share_F"][ii, jj], (pc[ii, jj] == 0).astype(float)])
    fold = np.random.default_rng(MYSEED + 70 + ri).integers(0, 5, len(ii)).astype(np.int8)
    flat = src * (G - 1) + cand - (cand > src)
    assert np.array_equal(ii[flat], src) and np.array_equal(jj[flat], cand)
    r2 = {}

    def ols_res(t, X):
        r = np.empty_like(t)
        for f in range(5):
            te = fold == f
            mu, sd = X[~te].mean(0), X[~te].std(0); sd[sd == 0] = 1
            Z = np.column_stack([np.ones((~te).sum()), (X[~te] - mu) / sd])
            beta = np.linalg.lstsq(Z, t[~te], rcond=None)[0]
            r[te] = t[te] - np.column_stack([np.ones(te.sum()), (X[te] - mu) / sd]) @ beta
        return r
    for k in ["forward", "transpose", "sym_mean", "sym_max", "rank_conditioned", "order_share_F", "coexpr_abs_spearman"]:
        t = S[k][ii, jj]
        R = ols_res(t, X6 if k == "order_share_F" else XF)
        r2[k] = float(1 - np.var(R) / np.var(t))
        sc[f"resid_{k}"] = R[flat]
        if k == "forward":
            sc["resid6_forward"] = ols_res(t, X6)[flat]
    np.savez(VER / f"trrust_scores_{run}.npz", y=y, kk=kk, cand=cand, tfs=tfs, **sc)
    jdump({"run": run, "oos_r2_of_resid_model": r2, "seconds": time.time() - t0}, VER / f"trrust_scores_{run}_meta.json")
    print(run, "scores saved", len(sc), f"{time.time() - t0:.0f}s", r2)


def cmd_trrust_null(run, n_draws=1000, n_trades=3000):
    """Own boolean-matrix Curveball; saves the positive-pair index set of every null network."""
    t0 = time.time()
    ri = RUNS.index(run)
    _, _, _, M, tfs, G, nT, cand, kk, src, y = _trrust_base(run)
    B0 = M[tfs]
    rngc = np.random.default_rng(MYSEED + 200 + ri)
    P = int(y.sum())
    pos = np.empty((n_draws, P), dtype=np.int32)
    jac = []
    for d in range(n_draws):
        Bn = curveball_bool(B0, tfs, n_trades, rngc)
        assert (Bn.sum(1) == B0.sum(1)).all() and (Bn.sum(0) == B0.sum(0)).all()
        assert not Bn[np.arange(nT), tfs].any()
        pos[d] = np.flatnonzero(Bn[kk, cand])
        jac.append(1 - (Bn & B0).sum() / (Bn | B0).sum())
    np.save(VER / f"null_pos_{run}.npy", pos)
    jdump({"run": run, "n_draws": n_draws, "n_trades": n_trades, "mean_jaccard_distance": float(np.mean(jac)),
           "min_jaccard_distance": float(np.min(jac)), "n_identical": int(np.sum(np.array(jac) == 0)),
           "seconds": time.time() - t0}, VER / f"null_meta_{run}.json")
    print(run, "null", f"{time.time() - t0:.0f}s", np.mean(jac), np.min(jac))


def _U(y, kk, s, nT):
    """U[a, b] = sum over positives of TF a and negatives of TF b of 1[s_pos > s_neg] + 0.5 * 1[tie].
    Pooled AUROC of a TF bootstrap with counts w = w.U.w / ((w.npos)(w.nneg)). Exact, own method."""
    U = np.zeros((nT, nT))
    for b in range(nT):
        neg = np.sort(s[(kk == b) & ~y])
        lo = np.searchsorted(neg, s[y], "left")
        hi = np.searchsorted(neg, s[y], "right")
        U[:, b] = np.bincount(kk[y], weights=lo + 0.5 * (hi - lo), minlength=nT)
    npos = np.bincount(kk[y], minlength=nT).astype(float)
    nneg = np.bincount(kk[~y], minlength=nT).astype(float)
    return U, npos, nneg


def cmd_trrust_eval(run):
    from sklearn.metrics import roc_auc_score
    from scipy.stats import rankdata
    t0 = time.time()
    ri = RUNS.index(run)
    z = np.load(VER / f"trrust_scores_{run}.npz")
    y, kk = z["y"].astype(bool), z["kk"]
    nT = int(kk.max()) + 1
    sc = {k: z[k] for k in z.files if k not in ("y", "kk", "cand", "tfs")}
    pos = np.load(VER / f"null_pos_{run}.npy").astype(np.int64)
    P = int(y.sum()); N = len(y) - P
    W = np.stack([np.bincount(d, minlength=nT) for d in np.random.default_rng(MYSEED + 100 + ri).integers(0, nT, (NB, nT))]).astype(float)
    out = {"run": run, "n_tfs": nT, "n_pos": P, "n_neg": N, "scores": {}}
    boot = {}
    for k, s in sc.items():
        auc = float(roc_auc_score(y, s))
        U, npos, nneg = _U(y, kk, s, nT)
        assert abs(U.sum() / (npos.sum() * nneg.sum()) - auc) < 1e-9
        boot[k] = np.einsum("ia,ab,ib->i", W, U, W) / ((W @ npos) * (W @ nneg))
        r = rankdata(s)
        nv = (r[pos].sum(1) - P * (P + 1) / 2) / (P * N)
        sd = nv.std(ddof=1)
        out["scores"][k] = {"auroc": auc, "ci": np.percentile(boot[k], [2.5, 97.5]).tolist(),
                            "null_mean": float(nv.mean()), "null_sd": float(sd),
                            "z": float((auc - nv.mean()) / sd) if sd > 1e-12 else None,
                            "p": float((1 + (nv >= auc - 1e-12).sum()) / (1 + len(nv))) if sd > 1e-12 else None}
    diffs = {}
    pairs = [("sym_mean", "forward"), ("sym_max", "forward"), ("transpose", "forward"),
             ("sym_mean", "coexpr_abs_spearman"), ("sym_max", "coexpr_abs_spearman"), ("order_share_F", "forward"),
             ("resid_sym_mean", "resid_coexpr_abs_spearman"), ("resid_sym_max", "resid_coexpr_abs_spearman")]
    for g in ["gene_model_3feat_owngroups", "gene_variance"] + (["gene_variance_single_log"] if "gene_variance_single_log" in sc else []):
        pairs += [(g, k) for k in ["forward", "transpose", "sym_mean", "sym_max", "rank_conditioned", "order_share_F"]]
    for a, b in pairs:
        diffs[f"{a}_minus_{b}"] = {"diff": out["scores"][a]["auroc"] - out["scores"][b]["auroc"],
                                   "ci": np.percentile(boot[a] - boot[b], [2.5, 97.5]).tolist()}
    out["diffs"] = diffs
    out["_method"] = {"pooled": "sklearn roc_auc_score over all (evaluated TF, candidate) pairs",
                      "ci": f"bootstrap over TFs ({NB} reps, own seed); exact pooled AUROC of each resample from the "
                            "TF-by-TF U matrix (fitted models held fixed)",
                      "null": "own boolean Curveball networks (null_pos_<run>.npy); one-sided p = (1+#null>=obs)/(1+draws)",
                      "resid": "OLS float64 lstsq on standardised gene6 (+F, +indicator n_pair=0 for edges), own random "
                               "5-fold split over all ordered pairs; F residualised on gene6 only",
                      "gene_model": "logistic, 3 candidate features, 5 folds grouped by TF (own random TF->fold map)"}
    out["seconds"] = time.time() - t0
    jdump(out, VER / f"trrust_{run}.json")
    print(run, f"{out['seconds']:.0f}s")
    for k, v in out["scores"].items():
        print(f"  {k:30s} {v['auroc']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}] null {v['null_mean']:.3f} z "
              f"{'' if v['z'] is None else round(v['z'], 2)} p {v['p']}")
    for k, v in diffs.items():
        print(f"  {k:50s} {v['diff']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]")


def cmd_hgb(run, score):
    """HGB residual check for one score (own folds; same hyper-parameters as D1b)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import roc_auc_score
    from scipy.stats import rankdata
    t0 = time.time()
    ri = RUNS.index(run)
    S, pc, gf, M, tfs, G, nT, cand, kk, src, y = _trrust_base(run)
    ii, jj = np.nonzero(~np.eye(G, dtype=bool))
    gm, gv, gd = (gf[c].to_numpy(np.float64) for c in ["mean_expr", "variance", "dropout_rate"])
    X = np.column_stack([gm[ii], gv[ii], gd[ii], gm[jj], gv[jj], gd[jj]])
    if score != "order_share_F":
        X = np.column_stack([X, S["order_share_F"][ii, jj], (pc[ii, jj] == 0).astype(float)])
    t = S[score][ii, jj]
    fold = np.random.default_rng(MYSEED + 70 + ri).integers(0, 5, len(ii)).astype(np.int8)
    R = np.empty_like(t)
    for f in range(5):
        te = fold == f
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, random_state=1).fit(X[~te], t[~te])
        R[te] = t[te] - m.predict(X[te])
    flat = src * (G - 1) + cand - (cand > src)
    s = R[flat]
    auc = float(roc_auc_score(y, s))
    pos = np.load(VER / f"null_pos_{run}.npy").astype(np.int64)
    P = int(y.sum()); N = len(y) - P
    r = rankdata(s)
    nv = (r[pos].sum(1) - P * (P + 1) / 2) / (P * N)
    U, npos, nneg = _U(y, kk, s, nT)
    W = np.stack([np.bincount(d, minlength=nT) for d in np.random.default_rng(MYSEED + 100 + ri).integers(0, nT, (NB, nT))]).astype(float)
    boot = np.einsum("ia,ab,ib->i", W, U, W) / ((W @ npos) * (W @ nneg))
    o = {"run": run, "score": score, "hgb_resid_auroc": auc, "ci": np.percentile(boot, [2.5, 97.5]).tolist(),
         "null_mean": float(nv.mean()), "z": float((auc - nv.mean()) / nv.std(ddof=1)),
         "p": float((1 + (nv >= auc - 1e-12).sum()) / (1 + len(nv))), "seconds": time.time() - t0}
    jdump(o, VER / f"hgb_{run}_{score}.json")
    print(o)


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); acc = 1.0
    for rank in range(n, 0, -1):
        i = o[rank - 1]
        acc = min(acc, p[i] * n / rank); q[i] = acc
    return q


def cmd_bh():
    fam = ["forward", "transpose", "sym_mean", "sym_max", "rank_conditioned", "order_share_F"]
    out = {}
    tr = json.load(open(D1B / "trrust" / "trrust_variants_summary.json"))
    p = [tr[r]["scores"][k]["null"]["p_one_sided"] for r in RUNS for k in fam]
    q = bh(p)
    out["D1b_raw_null_q24_recomputed"] = {f"{r}|{k}": float(qq) for (r, k), qq in zip([(r, k) for r in RUNS for k in fam], q)}
    out["D1b_raw_null_q24_reported"] = {f"{r}|{k}": tr[r]["scores"][k]["null"]["q_bh_all_24_tests"] for r in RUNS for k in fam}
    rs = json.load(open(D1B / "resid" / "resid_summary.json"))
    keys = [(r, k, "ols_gene6" if k == "order_share_F" else "ols_gene6_F") for r in RUNS for k in fam]
    p = [rs[r]["scores"][k][nm]["null_p_one_sided"] for r, k, nm in keys]
    q = bh(p)
    out["D1b_resid_q24_recomputed"] = {f"{r}|{k}": float(qq) for (r, k, _), qq in zip(keys, q)}
    out["D1b_resid_q24_reported"] = {f"{r}|{k}": rs[r]["scores"][k][nm].get("null_q_bh_main_ols_family") for r, k, nm in keys}
    out["D1b_resid_null_mean_note"] = "D1b resid JSONs store z and p but not the null mean"
    # own p-values
    own = {}
    for r in RUNS:
        f = VER / f"trrust_{r}.json"
        if f.exists():
            own[r] = json.load(open(f))
    if len(own) == 4:
        p = [own[r]["scores"][k]["p"] for r in RUNS for k in fam]
        out["own_raw_q24"] = {f"{r}|{k}": float(qq) for (r, k), qq in zip([(r, k) for r in RUNS for k in fam], bh(p))}
        p = [own[r]["scores"][f"resid_{k}"]["p"] for r in RUNS for k in fam]
        out["own_resid_q24"] = {f"{r}|{k}": float(qq) for (r, k), qq in zip([(r, k) for r in RUNS for k in fam], bh(p))}
    jdump(out, VER / "bh_check.json")
    for k in ["D1b_raw_null_q24_recomputed", "D1b_resid_q24_recomputed", "own_raw_q24", "own_resid_q24"]:
        if k in out:
            print(k, {kk: round(v, 3) for kk, v in out[k].items() if "RPE1" in kk or "1B" in kk})
    print("max |recomputed - reported| raw:", max(abs(out["D1b_raw_null_q24_recomputed"][k] - out["D1b_raw_null_q24_reported"][k]) for k in out["D1b_raw_null_q24_reported"]))
    print("max |recomputed - reported| resid:", max(abs(out["D1b_resid_q24_recomputed"][k] - out["D1b_resid_q24_reported"][k]) for k in out["D1b_resid_q24_reported"]))


def cmd_figcheck():
    fig = pd.read_csv(D1B / "fig_attention_variants.csv")
    tab = pd.read_csv(D1B / "table_attention_variants.csv")
    kd = json.load(open(D1B / "knockdown" / "knockdown_variants_summary.json"))["runs"]
    tr = json.load(open(D1B / "trrust" / "trrust_variants_summary.json"))
    rs = json.load(open(D1B / "resid" / "resid_summary.json"))
    bad = []
    for _, r in fig.iterrows():
        if r.endpoint == "knockdown":
            o = kd[r.run]["scores"][r.score]
            ref = (o["mean_auroc"], *o["ci95"])
        elif r.measure == "pooled AUROC" and r.score != "gene_model_3feat":
            o = tr[r.run]["scores"][r.score]; ref = (o["pooled_auroc"], *o["ci95"])
        elif r.measure == "Curveball null z":
            ref = (tr[r.run]["scores"][r.score]["null"]["z"], np.nan, np.nan)
        elif r.measure == "Curveball null mean":
            ref = (tr[r.run]["scores"][r.score]["null"]["mean"], np.nan, np.nan)
        elif r.measure.startswith("pooled AUROC after residualising"):
            nm = r.measure.split("(")[1].rstrip(")")
            o = rs[r.run]["scores"][r.score][nm]; ref = (o["residualised_auroc"], *o["ci95"])
        else:
            g = json.load(open(D1B / "trrust" / f"gene_model_gaps_{r.run}.json"))
            ref = (g["gene_model_auroc"], *g["gene_model_ci95"])
        got = (r.value, r.ci_low, r.ci_high)
        if not all((np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-12 for a, b in zip(map(float, got), map(float, ref))):
            bad.append((r.run, r.score, r.measure))
    out = {"fig_rows": len(fig), "table_rows": len(tab), "fig_rows_not_matching_json": bad,
           "fig_measures": fig["measure"].value_counts().to_dict(),
           "fig_null_cols": {c: int(fig[c].isna().sum()) for c in fig.columns}}
    jdump(out, VER / "figcheck.json")
    print(json.dumps(out, indent=1))


def cmd_config():
    sys.path.insert(0, str(PROJ / "setup"))
    import platform, sklearn, scipy  # noqa: E401
    ins = [TRRUST_TSV, PROJ / "setup/token_dictionary.json", PROJ / "setup/dataset_loader.py",
           Path("<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/"
                "crispri_validation/data/gene_median_dictionary_gc104M.pkl")]
    for r in RUNS:
        ins += [p0(r) / f for f in ["attention_edges_layer_mean.npy", "attention_pair_counts.npy", "gene_features.csv",
                                    "hvg_gene_table.csv", "control_cells.csv", "spearman_edges.npy"]]
        ins += [OUT / "v2_eval" / "de_labels" / f"{r}.npz", D1B / "order" / f"order_counts_{r}.npy",
                D1B / "knockdown" / f"per_perturbation_{r}.csv"]
        if r != "217M_RPE1":
            ins.append(D1B / "order" / f"gene_variance_single_log_{r}.csv")
    ins += [D1B / "trrust" / "trrust_variants_summary.json", D1B / "resid" / "resid_summary.json",
            D1B / "fig_attention_variants.csv", D1B / "table_attention_variants.csv"]
    outs = sorted(p for p in VER.glob("*") if p.is_file() and not p.name.startswith("._") and p.name != "run_config.json")
    cfg = {"item": "independent verification of D1b", "device": "CPU only, no forward pass",
           "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
           "sklearn": sklearn.__version__, "pandas": pd.__version__,
           "seeds": {"own master": MYSEED, "knockdown boot": "MYSEED+run_index", "trrust boot": "MYSEED+100+run_index",
                     "curveball": "MYSEED+200+run_index", "resid folds": "MYSEED+70+run_index",
                     "gene-model TF groups": "MYSEED+50+run_index"},
           "inputs": [{"path": str(p), "sha256": sha(p)} for p in ins],
           "scripts": [{"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__).resolve())}],
           "outputs": [{"path": str(p), "sha256": sha(p)} for p in outs]}
    jdump(cfg, VER / "run_config.json")
    print("inputs", len(ins), "outputs", len(outs))


if __name__ == "__main__":
    a = sys.argv[1:]
    {"order": lambda: cmd_order(a[1]), "knockdown": cmd_knockdown,
     "trrust_scores": lambda: cmd_trrust_scores(a[1]),
     "trrust_null": lambda: cmd_trrust_null(a[1], *(int(x) for x in a[2:])),
     "trrust_eval": lambda: cmd_trrust_eval(a[1]), "bh": cmd_bh,
     "hgb": lambda: cmd_hgb(a[1], a[2]),
     "figcheck": cmd_figcheck, "config": cmd_config}[a[0]]()
