"""v3_attention_k562_verify -- second-way checks for item V3-6 (does not import the two v3 scripts).

  pairs [max_minutes]  MPS: end-to-end check of saved edges. For random gene pairs with few co-occurring
                       cells, re-run those cells through the deployed extractor (maxtoki_adapter
                       MaxTokiAttentionExtractor.forward_with_attention, output_attentions=True, CPU copy)
                       and rebuild E[a,b] at all 11 layers; compare with the saved edges. Also: attention rows
                       sum to 1, nothing above the diagonal, and one cell on CPU float32 vs MPS float32.
  counts               C and n_pair rebuilt from saved tokens with own code (position matrix)
  geneset              gene set + gene features from the stored X with float expm1 (no rounding)
  labels               knockdown labels with own Welch t + statsmodels BH
  scores               sklearn AUROCs; own bootstraps (seed 777); checkerboard null (own sampler)
  resid                OLS residuals with sklearn LinearRegression and a different random fold split
  config               run_config.json for outputs/v3_attention_k562 (sha256 of inputs, code, outputs)
Outputs: outputs/v3_attention_k562/verification/
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import json  # noqa: E402
import os  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJ = Path("<REPO_ROOT>/projects/maxtoki")
RUN = PROJ / "runs/attention-grn-217M"
OUT = RUN / "outputs"
OUT3 = OUT / "v3_attention_k562"
D_PREP, D_EDGE = OUT3 / "prep", OUT3 / "edges"
OV = OUT3 / "verification"
SETUP = PROJ / "setup"
sys.path.insert(0, str(SETUP))
TRRUST_TSV = Path("<DATA_ROOT>/biodyn-nmi-paper/"
                  "src/02_cssi_method/cssi_real_data/results/trrust_human.tsv")
SEED_V = 777
LAYER = 8


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save_json(obj, path: Path) -> None:
    def _d(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_d)
    os.replace(tmp, path)


def tf_pairs(gf: pd.DataFrame):
    """Own TRRUST parsing: pooled (TF, candidate) pairs, TFs with >= 3 targets in the gene set."""
    sym = [s.upper() for s in gf["symbol"]]
    s2i = {s: i for i, s in enumerate(sym)}
    G = len(sym)
    M = np.zeros((G, G), dtype=bool)
    with open(TRRUST_TSV) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            a, b = s2i.get(p[0].upper()), s2i.get(p[1].upper())
            if a is not None and b is not None and a != b:
                M[a, b] = True
    tfs = np.array(sorted(i for i in range(G) if M[i].sum() >= 3), dtype=np.int64)
    src = np.repeat(tfs, G - 1)
    cand = np.concatenate([np.delete(np.arange(G), t) for t in tfs])
    return M, tfs, src, cand, M[src, cand]


def auc_sk(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


# ======================================================================================== pairs (model)
def cmd_pairs(max_minutes: float = 7.0) -> None:
    import torch
    import hooks_v2 as hv2
    import inputs_v3 as I
    from maxtoki_adapter import MaxTokiAttentionExtractor
    t0 = time.time()
    OV.mkdir(parents=True, exist_ok=True)
    out_p = OV / "pairs_check.json"
    z = np.load(D_PREP / "cells_tokens.npz")
    toks, lens = z["tokens"], z["lens"]
    counts = np.load(D_PREP / "counts_int32.npy", mmap_mode="r")
    gf = pd.read_csv(D_PREP / "gene_features.csv")
    tid = gf["maxtoki_token_id"].to_numpy(np.int64)
    Eall = np.load(D_EDGE / "attention_edges_layer_mean.npy", mmap_mode="r")
    PC = np.load(D_EDGE / "attention_pair_counts.npy")
    C = np.load(D_EDGE / "order_counts.npy")
    rng = np.random.default_rng(SEED_V)
    cand = np.argwhere((PC >= 3) & (PC <= 6) & (C >= 1) & ~np.eye(len(gf), dtype=bool))
    pick = cand[rng.choice(len(cand), 6, replace=False)]
    # cells holding both genes, from the tokens
    pos_of = []
    cells_needed = set()
    for a, b in pick:
        cs = [i for i in range(len(lens)) if np.isin(tid[a], toks[i, :lens[i]]) and np.isin(tid[b], toks[i, :lens[i]])]
        assert len(cs) == PC[a, b], (a, b, len(cs), PC[a, b])
        pos_of.append(cs)
        cells_needed.update(cs)
    cells_needed = sorted(cells_needed)
    hv2.check_memory(3.0)
    chk = I.assert_encoding_batch([counts[i].astype(np.float64) for i in cells_needed],
                                  [toks[i, :lens[i]] for i in cells_needed], "k562", 2048, label="verify pairs")
    xt = MaxTokiAttentionExtractor(device="mps", dtype=torch.float32)
    sums = [np.zeros(11) for _ in pick]
    row_sum_dev, upper = 0.0, 0.0
    hm0 = None
    for i in cells_needed:
        if time.time() - t0 > max_minutes * 60:
            raise SystemExit("time budget reached before all cells; increase max_minutes")
        hv2.check_memory(3.0)
        seq = toks[i, :lens[i]].astype(np.int64)
        _, att = xt.forward_with_attention(torch.from_numpy(seq[None]))
        hm = np.stack([a[0].numpy().mean(0) for a in att])              # (L, T, T)
        del att
        row_sum_dev = max(row_sum_dev, float(np.abs(hm.sum(-1) - 1).max()))
        upper = max(upper, float(np.abs(np.triu(hm, 1)).max()))
        p = {int(t): k for k, t in enumerate(seq)}
        for j, ((a, b), cs) in enumerate(zip(pick, pos_of)):
            if i in cs:
                sums[j] += hm[:, p[int(tid[a])], p[int(tid[b])]]
        if hm0 is None:
            hm0 = hm
        del hm
    res = []
    for j, ((a, b), cs) in enumerate(zip(pick, pos_of)):
        e_re = sums[j] / len(cs)
        e_saved = np.array(Eall[:, a, b], dtype=np.float64)
        res.append({"query_gene": gf["symbol"][a], "key_gene": gf["symbol"][b], "n_pair": int(len(cs)),
                    "C": int(C[a, b]), "layer8_saved": float(e_saved[LAYER]), "layer8_rebuilt": float(e_re[LAYER]),
                    "max_abs_diff_all_layers": float(np.abs(e_re - e_saved).max()),
                    "max_rel_diff_all_layers": float(np.max(np.abs(e_re - e_saved) / np.maximum(np.abs(e_saved), 1e-12)))})
    # one cell, CPU float32 vs MPS float32 (device rounding size)
    i0 = cells_needed[0]
    xt_cpu = MaxTokiAttentionExtractor(device="cpu", dtype=torch.float32)
    torch.set_num_threads(4)
    _, att_c = xt_cpu.forward_with_attention(torch.from_numpy(toks[i0, :lens[i0]].astype(np.int64)[None]))
    hm_c = np.stack([a[0].numpy().mean(0) for a in att_c])
    dev_cpu_mps = float(np.abs(hm_c - hm0).max())
    out = {"pairs": res, "n_cells_run": len(cells_needed), "encoding_check": chk,
           "max_abs_diff_rebuilt_vs_saved_any_pair_any_layer": max(r["max_abs_diff_all_layers"] for r in res),
           "max_rel_diff_rebuilt_vs_saved": max(r["max_rel_diff_all_layers"] for r in res),
           "max_abs_row_sum_minus_1": row_sum_dev, "max_attention_above_diagonal": upper,
           "cpu_vs_mps_max_abs_diff_head_mean_attention_one_cell": dev_cpu_mps, "seconds": time.time() - t0}
    save_json(out, out_p)
    log(json.dumps(out, indent=1))


# ======================================================================================== counts
def cmd_counts() -> None:
    t0 = time.time()
    z = np.load(D_PREP / "cells_tokens.npz")
    toks, lens = z["tokens"], z["lens"]
    out = {}
    for tag, gfp, sfx in [("new", D_PREP / "gene_features.csv", ""),
                          ("deployed_genes", OUT / "phase0" / "gene_features.csv", "_deployed_genes")]:
        gf = pd.read_csv(gfp)
        tid = gf["maxtoki_token_id"].to_numpy(np.int64)
        G = len(tid)
        lut = np.full(20275, -1, np.int64); lut[tid] = np.arange(G)
        R = np.full((len(lens), G), np.nan)
        for i in range(len(lens)):
            u = lut[toks[i, :lens[i]]]
            k = np.flatnonzero(u >= 0)
            R[i, u[k]] = k
        pres = np.isfinite(R)
        PC = (pres.astype(np.int64).T @ pres.astype(np.int64))
        C = np.zeros((G, G), dtype=np.int64)
        Rq = np.where(pres, R, -np.inf)
        Rk = np.where(pres, R, np.inf)
        for a0 in range(0, G, 64):
            a1 = min(G, a0 + 64)
            C[a0:a1] = (Rk[:, None, :] < Rq[:, a0:a1, None]).sum(0)
        out[tag] = {"pair_counts_equal_saved": bool(np.array_equal(PC, np.load(D_EDGE / f"attention_pair_counts{sfx}.npy"))),
                    "order_counts_equal_saved": bool(np.array_equal(C, np.load(D_EDGE / f"order_counts{sfx}.npy")))}
    out["seconds"] = time.time() - t0
    save_json(out, OV / "counts_check.json")
    log(out)


# ======================================================================================== gene set
def cmd_geneset() -> None:
    import h5py
    import inputs_v3 as I
    from dataset_loader import K562_H5, SYM2ENS_PKL
    t0 = time.time()
    rows = pd.read_csv(OUT / "phase0" / "control_cells.csv")["cell_idx_global"].to_numpy(np.int64)
    with h5py.File(K562_H5, "r") as f:
        X = np.stack([f["X"][int(r), :] for r in rows]).astype(np.float64)
        syms = [s.decode() if isinstance(s, bytes) else s for s in f["var"]["gene_name_index"][:]]
    E = np.expm1(X)                                                    # float, no rounding
    L = np.log1p(E / E.sum(1, keepdims=True) * 1e4)
    v = L.var(0)
    with open(SYM2ENS_PKL, "rb") as fh:
        s2e = pickle.load(fh)
    vocab = set(I.get_tokenizer().gene_token_dict)
    ok = np.array([s2e.get(s) in vocab if s2e.get(s) is not None else False for s in syms])
    order = [g for g in np.argsort(-v, kind="stable") if ok[g]][:1500]
    sel = np.sort(np.array(order))
    gf = pd.read_csv(D_PREP / "gene_features.csv")
    out = {"same_gene_set": bool(np.array_equal(sel, gf["var_idx"].to_numpy())),
           "max_abs_diff_mean": float(np.abs(L[:, sel].mean(0) - gf["mean_expr"]).max()),
           "max_abs_diff_variance": float(np.abs(L[:, sel].var(0) - gf["variance"]).max()),
           "max_abs_diff_dropout": float(np.abs((X[:, sel] == 0).mean(0) - gf["dropout_rate"]).max()),
           "seconds": time.time() - t0}
    save_json(out, OV / "geneset_check.json")
    log(out)


# ======================================================================================== labels
def cmd_labels() -> None:
    """All candidate perturbations: own block reader, own Welch t, statsmodels BH; the kept set and every
    label must equal de_labels/v3_primary.npz."""
    import h5py
    from scipy.stats import t as tdist
    from statsmodels.stats.multitest import multipletests
    from dataset_loader import resolve, SYM2ENS_PKL
    t0 = time.time()
    d = np.load(OUT3 / "de_labels" / "v3_primary.npz")
    gf = pd.read_csv(D_PREP / "gene_features.csv")
    cols = gf["var_idx"].to_numpy(np.int64)
    s2i = {s.upper(): i for i, s in enumerate(gf["symbol"])}
    with open(SYM2ENS_PKL, "rb") as fh:
        s2e = pickle.load(fh)
    ds = resolve("k562")
    ctrl = d["control_rows"]
    coi_idx = np.where(ds.cell_of_interest_mask)[0]
    codes = ds.perturbation_codes[coi_idx]
    cats = ds.perturbation_categories
    perts = []
    for code in np.unique(codes):
        c = cats[code]
        if code in ds.control_category_codes:
            continue
        rows = coi_idx[codes == code]
        if len(rows) >= 30 and s2e.get(c) is not None and c.upper() in s2i:
            perts.append((c, s2i[c.upper()], rows))
    need = np.unique(np.concatenate([ctrl] + [r for _, _, r in perts]))
    Xn = np.empty((len(need), len(cols)))
    with h5py.File(ds.h5_path, "r") as f:
        X = f["X"]
        # contiguous runs of needed rows
        brk = np.flatnonzero(np.diff(need) > 50) + 1
        k = 0
        for seg in np.split(need, brk):
            blk = X[int(seg[0]):int(seg[-1]) + 1, :][seg - seg[0]][:, cols].astype(np.float64)
            Xn[k:k + len(seg)] = blk
            k += len(seg)
    e = np.expm1(Xn)
    L = np.log1p(e / e.sum(1, keepdims=True) * 1e4)
    where = {int(r): i for i, r in enumerate(need)}
    Cm = L[[where[int(r)] for r in ctrl]]
    m2, v2, n2 = Cm.mean(0), Cm.var(0, ddof=1), len(Cm)
    kept, n_dis, n_pos_mine = [], 0, 0
    saved = {str(s): i for i, s in enumerate(d["pert_symbol"])}
    for c, h, rows in perts:
        P = L[[where[int(r)] for r in rows]]
        m1, v1, n1 = P.mean(0), P.var(0, ddof=1), len(P)
        se2 = v1 / n1 + v2 / n2
        with np.errstate(invalid="ignore", divide="ignore"):
            tt = (m1 - m2) / np.sqrt(se2)
            df = se2 ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
            p = 2 * tdist.sf(np.abs(tt), df)
        p = np.where(np.isfinite(p), p, 1.0)
        q = multipletests(p, method="fdr_bh")[1]
        y = (np.abs(m1 - m2) >= 0.5) & (q < 0.05)
        y[h] = False
        if y.sum() >= 3 and (len(y) - 1 - y.sum()) >= 3:
            kept.append(c)
            n_pos_mine += int(y.sum())
            if c in saved:
                n_dis += int((y != d["labels"][saved[c]]).sum())
    out = {"n_candidate_perturbations": len(perts), "n_kept_mine": len(kept), "n_kept_saved": int(len(saved)),
           "kept_sets_equal": sorted(kept) == sorted(saved), "label_disagreements_on_kept": n_dis,
           "positives_mine": n_pos_mine, "positives_saved": int(d["labels"].sum()), "seconds": time.time() - t0}
    save_json(out, OV / "labels_check.json")
    log(out)


# ======================================================================================== scores
def _edge_scores():
    E = np.array(np.load(D_EDGE / "attention_edges_layer_mean.npy", mmap_mode="r")[LAYER], dtype=np.float64)
    np.fill_diagonal(E, 0)
    pc = np.load(D_EDGE / "attention_pair_counts.npy").astype(np.float64)
    C = np.load(D_EDGE / "order_counts.npy").astype(np.float64)
    F = np.divide(C, pc, out=np.zeros_like(C), where=pc > 0); np.fill_diagonal(F, 0)
    Ec = np.divide(E * pc, C, out=np.zeros_like(C), where=C > 0); np.fill_diagonal(Ec, 0)
    co = np.abs(np.load(D_PREP / "spearman_edges.npy").astype(np.float64)); np.fill_diagonal(co, 0)
    return {"forward": E, "transpose": E.T.copy(), "sym_mean": (E + E.T) / 2, "sym_max": np.maximum(E, E.T),
            "rank_conditioned": Ec, "order_share_F": F, "coexpr_abs_spearman": co}


def checkerboard(M: np.ndarray, forb_col: np.ndarray, n_swaps: int, rng) -> np.ndarray:
    M = M.copy()
    r, c = np.nonzero(M)
    edges = list(zip(r.tolist(), c.tolist()))
    done = 0
    while done < n_swaps:
        i, j = rng.integers(0, len(edges), 2)
        (a, x), (b, yy) = edges[i], edges[j]
        if a == b or x == yy or M[a, yy] or M[b, x] or yy == forb_col[a] or x == forb_col[b]:
            continue
        M[a, x] = M[b, yy] = False
        M[a, yy] = M[b, x] = True
        edges[i], edges[j] = (a, yy), (b, x)
        done += 1
    return M


def cmd_scores() -> None:
    from scipy.stats import rankdata
    t0 = time.time()
    S = _edge_scores()
    gf = pd.read_csv(D_PREP / "gene_features.csv")
    G = len(gf)
    S_gene = {"gene_variance": gf["variance"].to_numpy(np.float64)}
    out = {"knockdown": {}, "trrust": {}}
    # Endpoint A, sklearn per perturbation, own bootstrap
    d = np.load(OUT3 / "de_labels" / "v3_primary.npz")
    A = {k: [] for k in list(S) + list(S_gene)}
    for h, y in zip(d["pert_hvg_idx"], d["labels"]):
        v = np.arange(G) != h
        for k, M in S.items():
            A[k].append(auc_sk(y[v], M[h][v]))
        for k, g in S_gene.items():
            A[k].append(auc_sk(y[v], g[v]))
    A = {k: np.array(x) for k, x in A.items()}
    kd = json.load(open(OUT3 / "knockdown" / "knockdown_summary.json"))["variants"]["v3_primary"]["scores"]
    n = len(A["forward"])
    B = np.random.default_rng(SEED_V).integers(0, n, (2000, n))
    for k in A:
        o = {"mean": float(A[k].mean()), "abs_diff_vs_main": abs(float(A[k].mean()) - kd[k]["mean_auroc"])}
        for ref in ["forward", "gene_variance"]:
            if k != ref:
                dd = A[k][B].mean(1) - A[ref][B].mean(1)
                o[f"minus_{ref}_ci95_seed777"] = [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]
        out["knockdown"][k] = o
    # Endpoint B: own pairs, sklearn pooled AUROC, explicit TF bootstrap, checkerboard null
    M, tfs, src, cand, y = tf_pairs(gf)
    tr = json.load(open(OUT3 / "trrust" / "trrust_summary.json"))
    nT = len(tfs)
    k_of = np.repeat(np.arange(nT), G - 1)
    rng = np.random.default_rng(SEED_V + 1)
    boots = {k: [] for k in S}
    vals = {k: S[k][src, cand] for k in S}
    for _ in range(1000):
        dr = rng.integers(0, nT, nT)
        idx = np.concatenate([np.flatnonzero(k_of == t) for t in dr])
        for k in S:
            boots[k].append(auc_sk(y[idx], vals[k][idx]))
    # checkerboard null (own sampler), own-gene cell forbidden
    Mt = M[tfs]
    forb = tfs.copy()
    rngn = np.random.default_rng(SEED_V + 2)
    n_edges = int(Mt.sum())
    nulls = {k: [] for k in S}
    ranks = {k: rankdata(vals[k]) for k in S}
    n_bad = n_same = 0
    for _ in range(1000):
        Mn = checkerboard(Mt, forb, 20 * n_edges, rngn)
        n_bad += int(not (np.array_equal(Mn.sum(1), Mt.sum(1)) and np.array_equal(Mn.sum(0), Mt.sum(0))
                          and not Mn[np.arange(nT), forb].any()))
        n_same += int(np.array_equal(Mn, Mt))
        yn = np.concatenate([np.delete(Mn[t], tfs[t]) for t in range(nT)])
        P = int(yn.sum()); N = len(yn) - P
        for k in S:
            nulls[k].append((ranks[k][yn].sum() - P * (P + 1) / 2) / (P * N))
    for k in S:
        a = auc_sk(y, vals[k])
        nv = np.array(nulls[k])
        out["trrust"][k] = {"pooled_sklearn": a, "abs_diff_vs_main": abs(a - tr["scores"][k]["pooled_auroc"]),
                            "ci95_explicit_tf_bootstrap_seed778": [float(np.percentile(boots[k], 2.5)), float(np.percentile(boots[k], 97.5))],
                            "ci95_main": tr["scores"][k]["ci95"],
                            "checkerboard_z": float((a - nv.mean()) / nv.std(ddof=1)),
                            "checkerboard_p": float((1 + (nv >= a - 1e-12).sum()) / 1001),
                            "curveball_z_main": tr["scores"][k]["null"]["z"]}
    out["trrust_meta"] = {"n_tfs": int(nT), "n_pos": int(y.sum()), "checkerboard_draws_failing_checks": n_bad,
                          "checkerboard_draws_identical": n_same, "swaps_per_draw": 20 * n_edges}
    out["seconds"] = time.time() - t0
    save_json(out, OV / "scores_check.json")
    log(json.dumps(out, indent=1))


# ======================================================================================== resid
def cmd_resid() -> None:
    from sklearn.linear_model import LinearRegression
    t0 = time.time()
    S = _edge_scores()
    gf = pd.read_csv(D_PREP / "gene_features.csv")
    G = len(gf)
    M, tfs, src_e, cand, y = tf_pairs(gf)
    ii, jj = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    msk = ii != jj
    src, tgt = ii[msk], jj[msk]
    gm, gv, gd = (gf[c].to_numpy(np.float64) for c in ["mean_expr", "variance", "dropout_rate"])
    X6 = np.column_stack([gm[src], gv[src], gd[src], gm[tgt], gv[tgt], gd[tgt]])
    Fp = S["order_share_F"][src, tgt]
    flat = src_e * (G - 1) + cand - (cand > src_e)
    fold = np.random.default_rng(SEED_V).integers(0, 5, len(src))
    main = json.load(open(OUT3 / "resid" / "resid_summary.json"))["scores"]
    pos = np.load(OUT3 / "trrust" / "null_positions.npy").astype(np.int64)
    from scipy.stats import rankdata
    out = {}
    for k in ["forward", "transpose", "sym_mean", "sym_max", "order_share_F", "coexpr_abs_spearman"]:
        X = X6 if k in ("order_share_F", "coexpr_abs_spearman") else np.column_stack([X6, Fp])
        tgt_v = S[k][src, tgt]
        R = np.empty_like(tgt_v)
        for f in range(5):
            te = fold == f
            m = LinearRegression().fit(X[~te], tgt_v[~te])
            R[te] = tgt_v[te] - m.predict(X[te])
        s = R[flat]
        a = auc_sk(y, s)
        r = rankdata(s)
        P = pos.shape[1]; N = len(s) - P
        nv = (r[pos].sum(1) - P * (P + 1) / 2) / (P * N)
        nm = "ols_gene6" if k in ("order_share_F", "coexpr_abs_spearman") else "ols_gene6_F"
        out[k] = {"resid_auroc": a, "null_z": float((a - nv.mean()) / nv.std(ddof=1)),
                  "main_resid_auroc": main[k][nm]["residualised_auroc"], "main_null_z": main[k][nm]["null_z"]}
    out["seconds"] = time.time() - t0
    save_json(out, OV / "resid_check.json")
    log(json.dumps(out, indent=1))


# ======================================================================================== config
def cmd_config() -> None:
    import platform
    import h5py
    import scipy
    import sklearn
    import torch
    import transformers
    import inputs_v3 as I
    from dataset_loader import K562_H5, SYM2ENS_PKL
    from maxtoki_adapter import DEFAULT_GENE_MEDIAN_PKL
    t0 = time.time()
    cache_p = OV / "_sha256_cache.json"
    cache = json.load(open(cache_p)) if cache_p.exists() else {}

    def h(p: Path):
        p = Path(p)
        st = p.stat()
        key = f"{p}|{st.st_size}|{int(st.st_mtime)}"
        if key not in cache:
            cache[key] = I.sha256_file(p)
            save_json(cache, cache_p)
        return {"path": str(p), "bytes": st.st_size, "sha256": cache[key]}

    meta = json.load(open(D_PREP.parent / "extract" / "state_meta.json"))
    prep = json.load(open(D_PREP / "prep_check.json"))
    fin = json.load(open(D_EDGE / "finalize_check.json"))
    gf = pd.read_csv(D_PREP / "gene_features.csv")
    cells = pd.read_csv(D_PREP / "control_cells.csv")
    inputs = [h(p) for p in [SYM2ENS_PKL, DEFAULT_GENE_MEDIAN_PKL, TRRUST_TSV, SETUP / "token_dictionary.json",
                             SETUP / "MaxToki-217M-HF/model.safetensors", SETUP / "MaxToki-217M-HF/config.json",
                             OUT / "phase0/control_cells.csv", OUT / "phase0/hvg_gene_table.csv",
                             OUT / "phase0/gene_features.csv", OUT / "phase0/attention_pair_counts.npy",
                             OUT / "phase0/attention_edges_layer_mean.npy", OUT / "phase0/spearman_edges.npy",
                             OUT / "v2_eval/de_labels/217M_K562.npz",
                             OUT / "v2b_attention/order/order_counts_217M_K562.npy",
                             OUT / "v2b_attention/trrust/null_positions_217M_K562.npy",
                             OUT / "v2b_attention/table_attention_variants.csv",
                             OUT / "v2b_attention/trrust/trrust_variants_217M_RPE1.json",
                             OUT / "v2b_attention/order/order_check_217M_K562.json",
                             OUT / "v2b_attention/trrust/trrust_variants_217M_K562.json",
                             OUT / "v2b_attention/knockdown/per_perturbation_217M_K562.csv"]]
    code = [h(p) for p in [RUN / "scripts/v3_attention_k562.py", RUN / "scripts/v3_attention_k562_eval.py",
                           RUN / "scripts/v3_attention_k562_verify.py", SETUP / "inputs_v3.py", SETUP / "hooks_v2.py",
                           SETUP / "maxtoki_adapter.py", SETUP / "dataset_loader.py", RUN / "scripts/v2_common.py",
                           RUN / "scripts/v2_06_curveball.py"]]
    outs = [h(p) for p in sorted(OUT3.rglob("*")) if p.is_file() and not p.name.startswith("._")
            and p.name not in ("run_config.json", "_sha256_cache.json") and not p.name.endswith(".tmp")]
    cfg = {
        "item": "V3-6 attention-GRN 217M K562 with correct input encoding",
        "device": {"forward_passes": "mps (Apple Silicon), float32, eager attention", "analysis": "cpu",
                   "python": platform.python_version(), "platform": platform.platform(),
                   "packages": {"torch": torch.__version__, "transformers": transformers.__version__,
                                "numpy": np.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
                                "pandas": pd.__version__, "h5py": h5py.__version__},
                   "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")},
        "model": {"dir": str(SETUP / "MaxToki-217M-HF"), "dtype": "float32", "attn_implementation": "eager",
                  "layer_used": LAYER, "edge": "self_attn weights of model.model.layers[l] (= output_attentions[l]), "
                                                "mean over 8 heads, row = query gene; summed over cells holding both genes "
                                                "and divided by that pair count"},
        "seeds": {"cell_draw": "numpy default_rng(42).choice(K562 non-targeting, 2000) sorted (deployed)",
                  "de_controls": "default_rng(42).choice(K562 controls, 5000) sorted (deployed)",
                  "knockdown_bootstrap": "20261001+100", "trrust_bootstrap": "20261001+500",
                  "curveball": "20261001+600", "residual_bootstrap": "20261001+800", "kfold": 42, "hgb": 20261001,
                  "nll_bootstrap": 20261002, "verification": "777, 778, 779"},
        "cells": {"n": int(len(cells)), "h5ad_rows": cells["cell_idx_global"].tolist(),
                  "barcodes_sha256": I.array_sha256(np.array(cells["cell_barcode"].astype(str).tolist()))},
        "genes": {"n": int(len(gf)), "ensembl_ids": gf["ensembl_id"].tolist(), "rule": "top-1,500 variance of "
                  "log1p(counts / rowsum * 1e4) among var genes in the MaxToki vocabulary (single log)",
                  "n_shared_with_deployed": prep["gene_set"]["n_shared_with_deployed"]},
        "input_encoding": I.encoding_record("k562", rows=None, check_summary={
            "n_cells_checked_at_tokenisation": prep["tokens"]["n_cells"],
            "n_cells_checked_before_forward": fin["n_cells_encoding_checked_before_forward"],
            "all_pass": fin["all_encoding_checks_pass"],
            "max_order_violations": prep["tokens"]["max_order_violations"],
            "deployed_tokens_failing_check": prep["tokens"]["deployed_tokens_failing_check_encoding"]},
            counts_sha256=prep["counts_sha256"]),
        "tokens_sha256": prep["tokens_sha256"],
        "wall_time_per_chunk": meta["chunks"],
        "inputs": inputs, "code": code, "outputs": outs, "seconds_config": time.time() - t0}
    save_json(cfg, OUT3 / "run_config.json")
    log("written", len(inputs), "inputs", len(code), "code", len(outs), "outputs")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    {"pairs": lambda: cmd_pairs(float(a[1]) if len(a) > 1 else 7.0), "counts": cmd_counts, "geneset": cmd_geneset,
     "labels": cmd_labels, "scores": cmd_scores, "resid": cmd_resid, "config": cmd_config}[a[0]]()
