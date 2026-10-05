"""v3_attention_k562_eval -- the v2b attention evaluation, re-run on the correctly encoded 217M K562 run.

Inputs: outputs/v3_attention_k562/{prep,edges}/ (written by v3_attention_k562.py). CPU only.
Rules are those of V2_EVAL_REPORT.md and V2B_ATTENTION_REPORT.md; only the inputs change:
  * attention edges re-extracted from correctly encoded cells (counts / median order),
  * gene set, gene features, co-expression and knockdown labels on a single log.

Scores (layer 8, head mean, diagonal 0): forward E[P,T]; transpose E[T,P]; sym_mean; sym_max;
rank_conditioned Ec = E * n_pair / C (0 where C = 0); order_share_F = C / n_pair (no model).
Reference (outside the test family): |Spearman| co-expression on single-log values, same 2,000 cells.
Gene-level baselines: target variance (single log); 3-feature gene model (TRRUST).

Subcommands (in order)
  de                       knockdown labels: primary (new genes, single log), storedX sensitivity, and
                           single-log labels on the deployed genes; one read pass
  order                    order share F, decomposition of E, checks
  knockdown                Endpoint A
  trrust                   Endpoint B raw + Curveball null (v2_06 sampler, 1,000 x 5,000 trades)
  gene3                    3-feature gene model gaps
  resid ols | hgb [max_s]  residualisation (resumable); resid combine
  coexpr                   co-expression comparison (correlation, mutual residualisation, null of difference)
  layers                   per-layer supplement (point estimates; max-z-over-layers null)
  oldgenes                 new-encoding attention on the DEPLOYED gene set with the v2 labels (isolates the
                           encoding effect; same Curveball networks as v2b)
  table                    tidy tables, figure CSV, comparison CSV (v2b K562, v3 K562, v2b RPE1)
Seeds: v2 master 20261001 with the v2 offsets for run index 0 (K562): knockdown +100, TRRUST bootstrap
+500, Curveball +600, residual bootstrap +800; KFold seed 42.
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
from scipy import stats  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from v2_common import (PROJ, OUT, V2, auroc_rank, percentile_ci, bh, WeightedAUROC,  # noqa: E402
                       load_trrust, trrust_matrix, evaluated_tf_rows, tie_self_test, SYM2ENS_PKL)

OUT3 = OUT / "v3_attention_k562"
D_PREP, D_EDGE = OUT3 / "prep", OUT3 / "edges"
V2B = OUT / "v2b_attention"
SEED = 20261001
RI = 0                          # run index of 217M K562 in the v2 / v2b RUNS order
N_BOOT = 2000
N_DRAWS, N_TRADES = 1000, 5000
LAYER = 8
EDGE_SCORES = ["forward", "transpose", "sym_mean", "sym_max", "rank_conditioned"]
ALL_SCORES = EDGE_SCORES + ["order_share_F"]
LFC, QMAX, MIN_CELLS, MIN_POS, N_CTRL_DE, CTRL_SEED = 0.5, 0.05, 30, 3, 5000, 42


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
        raise TypeError(type(o))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_d)
    os.replace(tmp, path)


def zp(nv: np.ndarray, obs: float):
    sd = float(nv.std(ddof=1))
    if sd <= 1e-12:
        return None, None
    return float((obs - nv.mean()) / sd), float((1 + (nv >= obs - 1e-12).sum()) / (1 + len(nv)))


# ============================================================================================ contexts
class Ctx:
    """Everything one evaluation needs: gene table, scores, counts, labels."""

    def __init__(self, kind: str = "main"):
        self.kind = kind
        if kind == "main":
            self.gf = pd.read_csv(D_PREP / "gene_features.csv")
            sfx = ""
            self.coexpr_path = D_PREP / "spearman_edges.npy"
            self.label_path = OUT3 / "de_labels" / "v3_primary.npz"
        elif kind == "deployed_genes":
            self.gf = pd.read_csv(OUT / "phase0" / "gene_features.csv")       # deployed (double-log) features
            sfx = "_deployed_genes"
            self.coexpr_path = OUT / "phase0" / "spearman_edges.npy"          # deployed co-expression
            self.label_path = V2 / "de_labels" / "217M_K562.npz"              # v2 (double-log) labels
        else:
            raise ValueError(kind)
        self.G = len(self.gf)
        E = np.load(D_EDGE / f"attention_edges_layer_mean{sfx}.npy", mmap_mode="r")
        self.E_all = E
        self.E = np.array(E[LAYER], dtype=np.float64)
        np.fill_diagonal(self.E, 0.0)
        self.pc = np.load(D_EDGE / f"attention_pair_counts{sfx}.npy").astype(np.float64)
        self.C = np.load(D_EDGE / f"order_counts{sfx}.npy").astype(np.float64)

    def scores(self, E: np.ndarray | None = None) -> dict:
        E = self.E if E is None else E
        pc, C = self.pc, self.C
        F = np.where(pc > 0, C / np.maximum(pc, 1.0), 0.0)
        Ec = np.where(C > 0, E * pc / np.maximum(C, 1.0), 0.0)
        for M in (F, Ec):
            np.fill_diagonal(M, 0.0)
        return {"forward": E, "transpose": E.T.copy(), "sym_mean": 0.5 * (E + E.T),
                "sym_max": np.maximum(E, E.T), "rank_conditioned": Ec, "order_share_F": F}

    def coexpr(self) -> np.ndarray:
        sp = np.abs(np.load(self.coexpr_path).astype(np.float64))
        np.fill_diagonal(sp, 0.0)
        return sp

    def trrust_setup(self):
        sym = [s.upper() for s in self.gf["symbol"]]
        tr = load_trrust()
        Etr = trrust_matrix(sym, tr)
        tf_rows = evaluated_tf_rows(sym, tr, Etr)
        G = self.G
        k_of = np.concatenate([np.full(G - 1, k) for k in range(len(tf_rows))])
        cand = np.concatenate([np.delete(np.arange(G), r) for r in tf_rows])
        src = tf_rows[k_of]
        y = Etr[src, cand].astype(bool)
        return sym, Etr, tf_rows, k_of, cand, src, y


# ============================================================================================ de labels
def _read_block_rows(h5_path, rows_sorted: np.ndarray, cols: np.ndarray):
    """Stream the dense K562 X in row blocks. Returns stored X[:, cols] (float32), expm1 counts on cols
    (float64) and per-row count checks over ALL genes (whole-number multiples of the row's smallest value)."""
    import h5py
    n = len(rows_sorted)
    Xc = np.empty((n, len(cols)), dtype=np.float32)
    max_dev = np.zeros(n)
    n_fail = np.zeros(n, dtype=np.int64)
    with h5py.File(h5_path, "r") as f:
        X = f["X"]
        block = 8192
        k = 0
        lo, hi = int(rows_sorted[0]), int(rows_sorted[-1]) + 1
        for a in range((lo // block) * block, hi, block):
            b = min(a + block, X.shape[0])
            sel = rows_sorted[(rows_sorted >= a) & (rows_sorted < b)]
            if len(sel) == 0:
                continue
            arr = X[a:b, :][sel - a].astype(np.float64)
            e = np.expm1(arr)
            ee = np.where(e > 0, e, np.inf)
            unit = ee.min(1, keepdims=True)
            q = np.where(e > 0, e / unit, 0.0)
            dev = np.abs(q - np.round(q))
            tol = np.maximum(1e-3, 5e-6 * q)
            max_dev[k:k + len(sel)] = dev.max(1)
            n_fail[k:k + len(sel)] = (dev > tol).sum(1)
            Xc[k:k + len(sel)] = arr[:, cols].astype(np.float32)
            k += len(sel)
        assert k == n
    return Xc, max_dev, n_fail


def _de_one(Xl: np.ndarray, pos_of: dict, ctrl: np.ndarray, perts: list, G: int):
    C = Xl[[pos_of[int(r)] for r in ctrl]].astype(np.float64)
    mu_c = C.mean(0)
    keep_sym, keep_h, keep_n, labels, lfcs, qs = [], [], [], [], [], []
    dropped = 0
    for cat, h, rows in perts:
        P = Xl[[pos_of[int(r)] for r in rows]].astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            _, p = stats.ttest_ind(P, C, axis=0, equal_var=False)
        p = np.where(np.isfinite(p), p, 1.0)
        q = bh(p)
        lfc = P.mean(0) - mu_c
        y = (np.abs(lfc) >= LFC) & (q < QMAX)
        y[h] = False
        n_pos = int(y.sum())
        if n_pos < MIN_POS or (G - 1 - n_pos) < MIN_POS:
            dropped += 1
            continue
        keep_sym.append(cat); keep_h.append(h); keep_n.append(len(rows))
        labels.append(y); lfcs.append(lfc.astype(np.float32)); qs.append(q.astype(np.float32))
    return dict(pert_symbol=np.array(keep_sym), pert_hvg_idx=np.array(keep_h, np.int64),
                n_cells=np.array(keep_n, np.int64), labels=np.array(labels, dtype=bool).reshape(-1, G),
                lfc=np.array(lfcs, dtype=np.float32).reshape(-1, G), q_bh=np.array(qs, dtype=np.float32).reshape(-1, G),
                control_rows=ctrl), dropped


def cmd_de() -> None:
    sys.path.insert(0, str(PROJ / "setup"))
    from dataset_loader import resolve as load_ds
    t0 = time.time()
    outd = OUT3 / "de_labels"
    outd.mkdir(parents=True, exist_ok=True)
    ds = load_ds("k562")
    with open(SYM2ENS_PKL, "rb") as f:
        sym2ens = pickle.load(f)
    gf_new = pd.read_csv(D_PREP / "gene_features.csv")
    gf_dep = pd.read_csv(OUT / "phase0" / "gene_features.csv")
    sets = {"v3_primary": gf_new, "v3_storedX": gf_new, "deployed_genes_single_log": gf_dep}
    cols_union = np.union1d(gf_new["var_idx"].to_numpy(np.int64), gf_dep["var_idx"].to_numpy(np.int64))
    col_pos = {int(c): i for i, c in enumerate(cols_union)}
    coi, codes, cats, ctrl_codes = (ds.cell_of_interest_mask, ds.perturbation_codes,
                                    ds.perturbation_categories, ds.control_category_codes)
    ctrl = np.where(coi & np.isin(codes, list(ctrl_codes)))[0]
    n_ctrl_total = len(ctrl)
    ctrl = np.sort(np.random.default_rng(CTRL_SEED).choice(ctrl, N_CTRL_DE, replace=False))
    coi_idx = np.where(coi)[0]
    coi_codes = codes[coi]
    pert_rows = {}
    for code in np.unique(coi_codes):
        cat = cats[code]
        if code in ctrl_codes or cat.lower() == "control":
            continue
        rows = coi_idx[coi_codes == code]
        pert_rows[cat] = np.sort(rows)
    plan, skipped = {}, {}
    for name, gf in sets.items():
        s2i = {s.upper(): i for i, s in enumerate(gf["symbol"])}
        pl, sk = [], {"lt30": 0, "no_ens": 0, "not_in_geneset": 0}
        for cat, rows in pert_rows.items():
            if len(rows) < MIN_CELLS:
                sk["lt30"] += 1; continue
            if sym2ens.get(cat) is None:
                sk["no_ens"] += 1; continue
            h = s2i.get(cat.upper())
            if h is None:
                sk["not_in_geneset"] += 1; continue
            pl.append((cat, h, rows))
        plan[name], skipped[name] = pl, sk
    need = np.unique(np.concatenate([ctrl] + [p[2] for pl in plan.values() for p in pl]))
    t1 = time.time()
    Xc, max_dev, n_fail = _read_block_rows(ds.h5_path, need, cols_union)
    log(f"read {Xc.shape} in {time.time() - t1:.0f}s; count check: rows failing {int((n_fail > 0).sum())}, "
        f"max dev {max_dev.max():.2e}")
    pos_of = {int(r): i for i, r in enumerate(need)}
    meta = {"n_rows_read": int(len(need)), "count_check_rows_failing": int((n_fail > 0).sum()),
            "count_check_max_dev_from_whole_multiple": float(max_dev.max()),
            "n_control_cells_available": int(n_ctrl_total), "n_control_cells_used": int(len(ctrl)),
            "variants": {}}
    for name, gf in sets.items():
        cols = np.array([col_pos[int(c)] for c in gf["var_idx"]], dtype=np.int64)
        if name == "v3_storedX":
            Xl = Xc[:, cols].astype(np.float32)                       # stored log1p(CP10k), used as is
            norm = "stored X (log1p of CP10k over the full transcriptome), restricted to the 1,500 genes; no rescale"
        else:
            cnt = np.expm1(Xc[:, cols].astype(np.float64))            # proportional to counts
            rs = cnt.sum(1, keepdims=True); rs[rs == 0] = 1.0
            Xl = np.log1p(cnt / rs * 1e4).astype(np.float32)
            norm = "expm1(X) (= counts up to a per-cell factor), scaled to 1e4 over the 1,500 genes, log1p (single log; RPE1 rule)"
        res, dropped = _de_one(Xl, pos_of, ctrl, plan[name], len(gf))
        np.savez_compressed(outd / f"{name}.npz", **res)
        meta["variants"][name] = {"gene_set": "new (single-log variance)" if name != "deployed_genes_single_log" else "deployed",
                                  "normalisation": norm, "n_candidate_perturbations": len(plan[name]),
                                  "skipped_before_test": skipped[name], "dropped_lt3_positives_or_negatives": dropped,
                                  "n_perturbations_evaluated": int(res["labels"].shape[0]),
                                  "n_positive_pairs": int(res["labels"].sum()),
                                  "median_positives_per_perturbation": float(np.median(res["labels"].sum(1))) if len(res["labels"]) else None}
        log(name, meta["variants"][name])
    meta["rule"] = {"lfc_abs_min": LFC, "bh_q_max": QMAX, "min_cells": MIN_CELLS, "min_pos_and_neg": MIN_POS,
                    "n_ctrl_cap": N_CTRL_DE, "ctrl_seed": CTRL_SEED,
                    "test": "Welch t (scipy ttest_ind equal_var=False), BH over the 1,500 genes per perturbation"}
    meta["seconds"] = time.time() - t0
    save_json(meta, outd / "de_meta.json")


# ============================================================================================ order
def cmd_order() -> None:
    t0 = time.time()
    ctx = Ctx("main")
    E, pc, C, G = ctx.E, ctx.pc, ctx.C, ctx.G
    off = ~np.eye(G, dtype=bool)
    m = off & (pc > 0)
    S = ctx.scores()
    F, Ec = S["order_share_F"], S["rank_conditioned"]
    mc = m & (C > 0) & (E > 0)
    lE, lF, lEc = np.log(E[mc]), np.log(F[mc]), np.log(Ec[mc])
    r2 = 1 - np.var(lE - np.polyval(np.polyfit(lF, lE, 1), lF)) / np.var(lE)
    dec = {"spearman_E_vs_F_all_pairs_with_n_pair_gt0": float(spearmanr(E[m], F[m]).correlation),
           "spearman_E_vs_Ec_all_pairs": float(spearmanr(E[m], Ec[m]).correlation),
           "spearman_F_vs_Ec_all_pairs": float(spearmanr(F[m], Ec[m]).correlation),
           "r2_logE_on_logF_pairs_with_C_gt0": float(r2),
           "frac_n_pair_gt0_pairs_with_C_eq0": float(((C == 0) & m).sum() / m.sum()),
           "n_offdiag_pairs_with_n_pair_eq0": int((off & (pc == 0)).sum())}
    colF = (C.sum(0) / np.maximum(pc.sum(0), 1)).astype(np.float64)
    rho_EF, rho_Fcol, rho_EcF = [], [], []
    for p in range(G):
        mm = m[p]
        if mm.sum() > 10:
            rho_EF.append(spearmanr(E[p, mm], F[p, mm]).correlation)
            rho_Fcol.append(spearmanr(F[p, mm], colF[mm]).correlation)
            rho_EcF.append(spearmanr(Ec[p, mm], F[p, mm]).correlation)
    dec["median_within_row_spearman_E_vs_F"] = float(np.nanmedian(rho_EF))
    dec["median_within_row_spearman_F_vs_target_only_order_score"] = float(np.nanmedian(rho_Fcol))
    dec["median_within_row_spearman_Ec_vs_F"] = float(np.nanmedian(rho_EcF))
    # same numbers for the deployed edge (v2b K562), to compare
    v2b = json.load(open(V2B / "order" / "order_check_217M_K562.json"))
    out = {"decomposition_layer8": dec, "v2b_217M_K562_decomposition": v2b["decomposition_layer8"],
           "seconds": time.time() - t0}
    save_json(out, OUT3 / "order" / "order_check.json")
    log(json.dumps(out, indent=1))


# ============================================================================================ endpoint A
def knockdown_eval(ctx: Ctx, label_path: Path, extra_scores: dict | None = None, base: dict | None = None,
                   seed_off: int = 100):
    S = ctx.scores()
    S["coexpr_abs_spearman"] = ctx.coexpr()
    if extra_scores:
        S.update(extra_scores)
    base = base or {"gene_variance": ctx.gf["variance"].to_numpy(np.float64)}
    G = ctx.G
    d = np.load(label_path)
    syms, hs, Y = d["pert_symbol"], d["pert_hvg_idx"], d["labels"]
    names = list(S) + list(base)
    A = {k: [] for k in names}
    rho = []
    for h, y in zip(hs, Y):
        valid = np.ones(G, bool); valid[h] = False
        yy = y[valid]
        for k, M in S.items():
            A[k].append(auroc_rank(yy, M[h][valid]))
        for k, v in base.items():
            A[k].append(auroc_rank(yy, v[valid]))
        mm = valid & (ctx.pc[h] > 0)
        rho.append(spearmanr(S["forward"][h, mm], S["order_share_F"][h, mm]).correlation)
    A = {k: np.array(v) for k, v in A.items()}
    pp = pd.DataFrame({"pert_symbol": syms.astype(str), "pert_hvg_idx": hs,
                       "n_pos": [int(Y[i][np.arange(G) != h].sum()) for i, h in enumerate(hs)]})
    for k in names:
        pp[f"auc_{k}"] = A[k]
    n = len(hs)
    rng = np.random.default_rng(SEED + seed_off + RI)
    B = rng.integers(0, n, size=(N_BOOT, n))
    bm = {k: A[k][B].mean(1) for k in names}
    out = {"n_perturbations": int(n), "n_positive_pairs": int(pp["n_pos"].sum()),
           "median_positives_per_perturbation": float(pp["n_pos"].median()) if n else None,
           "median_within_row_spearman_forward_vs_F": float(np.nanmedian(rho)) if n else None, "scores": {}}
    bv = "gene_variance"
    for k in names:
        o = {"mean_auroc": float(A[k].mean()), "ci95": percentile_ci(bm[k]),
             "gap_variance_minus_score": float(A[bv].mean() - A[k].mean()),
             "gap_variance_minus_score_ci95": percentile_ci(bm[bv] - bm[k]),
             "n_perts_score_gt_variance": int((A[k] > A[bv]).sum())}
        if k != "forward":
            o["delta_vs_forward"] = float(A[k].mean() - A["forward"].mean())
            o["delta_vs_forward_ci95"] = percentile_ci(bm[k] - bm["forward"])
            o["wilcoxon_p_vs_forward"] = wilcox_p(A[k], A["forward"])
        if k != bv:
            o["wilcoxon_p_vs_variance"] = wilcox_p(A[k], A[bv])
        if k != "coexpr_abs_spearman":
            o["delta_vs_coexpr"] = float(A[k].mean() - A["coexpr_abs_spearman"].mean())
            o["delta_vs_coexpr_ci95"] = percentile_ci(bm[k] - bm["coexpr_abs_spearman"])
        out["scores"][k] = o
    return out, pp


def wilcox_p(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2 or np.allclose(a[m], b[m]):
        return float("nan")
    return float(stats.wilcoxon(a[m], b[m], zero_method="zsplit").pvalue)


def cmd_knockdown() -> None:
    t0 = time.time()
    ctx = Ctx("main")
    res = {"tie_self_test": tie_self_test(), "variants": {}}
    base = {"gene_variance": ctx.gf["variance"].to_numpy(np.float64),
            "gene_variance_storedX": ctx.gf["variance_storedX_log1p_cp10k"].to_numpy(np.float64)}
    for name in ["v3_primary", "v3_storedX"]:
        out, pp = knockdown_eval(ctx, OUT3 / "de_labels" / f"{name}.npz", base=base)
        (OUT3 / "knockdown").mkdir(parents=True, exist_ok=True)
        pp.to_csv(OUT3 / "knockdown" / f"per_perturbation_{name}.csv", index=False)
        # Wilcoxon family: variance vs each of the 6 scores, BH within this variant
        fam = ALL_SCORES
        p = np.array([out["scores"][k]["wilcoxon_p_vs_variance"] for k in fam])
        q = bh(np.where(np.isfinite(p), p, 1.0))
        for k, qq in zip(fam, q):
            out["scores"][k]["wilcoxon_q_vs_variance_bh6"] = float(qq)
        res["variants"][name] = out
        log(name, out["n_perturbations"], {k: (round(v["mean_auroc"], 4), [round(x, 4) for x in v["ci95"]])
                                           for k, v in out["scores"].items()})
    res["_method"] = {"endpoint": "A, knockdown response; labels = outputs/v3_attention_k562/de_labels/<variant>.npz",
                      "summary": "mean over perturbations of per-perturbation AUROC (tie-safe average ranks)",
                      "ci": "percentile bootstrap over perturbations, 2,000 reps, seed 20261001+100 (v2 K562 seed)",
                      "test": "paired Wilcoxon signed-rank, zero_method='zsplit', two-sided; BH over the 6 scores vs variance",
                      "seconds": time.time() - t0}
    save_json(res, OUT3 / "knockdown" / "knockdown_summary.json")


# ============================================================================================ endpoint B
def null_auroc(score: np.ndarray, pos: np.ndarray) -> np.ndarray:
    r = rankdata(score, method="average")
    P = pos.shape[1]
    N = len(score) - P
    return (r[pos].sum(1) - P * (P + 1) / 2.0) / (P * N)


def make_null(ctx: Ctx, seed: int):
    """Curveball networks (v2_06 sampler unchanged) + the v2_06 checks. Returns pooled positive positions."""
    from v2_06_curveball import curveball
    sym, Etr, tf_rows, k_of, cand, src, y = ctx.trrust_setup()
    G, nT = ctx.G, len(tf_rows)
    M = Etr[tf_rows].astype(np.int8)
    obs_rows = [set(np.flatnonzero(M[k]).tolist()) for k in range(nT)]
    obs_set = {(k, c) for k, r in enumerate(obs_rows) for c in r}
    rs0, cs0 = M.sum(1), M.sum(0)
    forb = tf_rows.copy()
    n_edges = int(M.sum())
    rngc = np.random.default_rng(seed)
    pos = np.empty((N_DRAWS, n_edges), dtype=np.int64)
    n_bad = n_ident = 0
    jac = []
    for d in range(N_DRAWS):
        rows = curveball(obs_rows, forb, N_TRADES, rngc)
        Mm = np.zeros_like(M)
        for k, r in enumerate(rows):
            Mm[k, list(r)] = 1
        if not (np.array_equal(Mm.sum(1), rs0) and np.array_equal(Mm.sum(0), cs0) and not Mm[np.arange(nT), forb].any()):
            n_bad += 1
        s = {(k, c) for k, r in enumerate(rows) for c in r}
        j = 1.0 - len(s & obs_set) / len(s | obs_set)
        jac.append(j); n_ident += int(j == 0)
        pp = []
        for k, cols in enumerate(rows):
            c = np.array(sorted(cols), dtype=np.int64)
            pp.append(k * (G - 1) + c - (c > tf_rows[k]))
        pos[d] = np.concatenate(pp)
    assert np.array_equal(np.sort(np.flatnonzero(y)), np.sort(np.concatenate(
        [k * (G - 1) + np.array(sorted(r)) - (np.array(sorted(r)) > tf_rows[k]) for k, r in enumerate(obs_rows)])))
    checks = {"n_draws": N_DRAWS, "n_trades_per_draw": N_TRADES, "seed": seed, "n_edges": n_edges,
              "draws_failing_margin_or_structural_zero_check": int(n_bad), "draws_identical_to_observed": int(n_ident),
              "mean_jaccard_distance_from_observed": float(np.mean(jac)),
              "min_jaccard_distance_from_observed": float(np.min(jac))}
    return pos, checks


def cmd_trrust() -> None:
    t0 = time.time()
    ctx = Ctx("main")
    S = ctx.scores()
    S["coexpr_abs_spearman"] = ctx.coexpr()
    base = {"gene_variance": ctx.gf["variance"].to_numpy(np.float64),
            "gene_variance_storedX": ctx.gf["variance_storedX_log1p_cp10k"].to_numpy(np.float64)}
    sym, Etr, tf_rows, k_of, cand, src, y = ctx.trrust_setup()
    nT = len(tf_rows)
    scores = {k: M[src, cand] for k, M in S.items()}
    for k, v in base.items():
        scores[k] = v[cand]
    pooled = {k: auroc_rank(y, s) for k, s in scores.items()}
    fast = {k: WeightedAUROC(y, s) for k, s in scores.items()}
    for k in scores:
        assert abs(fast[k]() - pooled[k]) < 1e-9
    rng = np.random.default_rng(SEED + 500 + RI)
    boot = {k: [] for k in scores}
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, nT, nT), minlength=nT)[k_of]
        for k in scores:
            boot[k].append(fast[k](w))
    boot = {k: np.array(v) for k, v in boot.items()}
    per_tf = []
    for k in range(nT):
        mk = k_of == k
        row = {"tf": sym[tf_rows[k]], "n_targets": int(y[mk].sum())}
        for name, s in scores.items():
            row[name] = auroc_rank(y[mk], s[mk])
        per_tf.append(row)
    per_tf = pd.DataFrame(per_tf)
    (OUT3 / "trrust").mkdir(parents=True, exist_ok=True)
    per_tf.to_csv(OUT3 / "trrust" / "per_tf.csv", index=False)
    pos, nchk = make_null(ctx, SEED + 600 + RI)
    np.save(OUT3 / "trrust" / "null_positions.npy", pos.astype(np.int32))
    nulls = {k: null_auroc(s, pos) for k, s in scores.items()}
    nchk["variance_auroc_null_max_abs_change"] = float(np.max(np.abs(nulls["gene_variance"] - pooled["gene_variance"])))
    out = {"n_tfs": int(nT), "tfs": [sym[r] for r in tf_rows], "n_trrust_edges_in_gene_set": int(Etr.sum()),
           "n_positive_pairs": int(y.sum()), "n_negative_pairs": int((~y).sum()), "null_checks": nchk, "scores": {}}
    for k in scores:
        z, p = zp(nulls[k], pooled[k])
        o = {"pooled_auroc": pooled[k], "ci95": percentile_ci(boot[k]),
             "per_tf_mean_auroc": float(np.nanmean(per_tf[k])),
             "gap_variance_minus_score": pooled["gene_variance"] - pooled[k],
             "gap_variance_minus_score_ci95": percentile_ci(boot["gene_variance"] - boot[k]),
             "n_tfs_score_gt_variance": int((per_tf[k] > per_tf["gene_variance"]).sum()),
             "null": {"mean": float(nulls[k].mean()), "sd": float(nulls[k].std(ddof=1)), "z": z, "p_one_sided": p,
                      "note": None if z is not None else "target-only score: unchanged by a null that keeps column sums"}}
        if k != "forward":
            o["delta_vs_forward"] = pooled[k] - pooled["forward"]
            o["delta_vs_forward_ci95"] = percentile_ci(boot[k] - boot["forward"])
        if k != "coexpr_abs_spearman":
            o["delta_vs_coexpr"] = pooled[k] - pooled["coexpr_abs_spearman"]
            o["delta_vs_coexpr_ci95"] = percentile_ci(boot[k] - boot["coexpr_abs_spearman"])
        out["scores"][k] = o
    # BH over the 6-score family (K562 v3); and over 12 tests together with the v2b RPE1 family
    p6 = np.array([out["scores"][k]["null"]["p_one_sided"] for k in ALL_SCORES])
    for k, q in zip(ALL_SCORES, bh(p6)):
        out["scores"][k]["null"]["q_bh_6_scores"] = float(q)
    rp = json.load(open(V2B / "trrust" / "trrust_variants_217M_RPE1.json"))
    p12 = np.r_[p6, [rp["scores"][k]["null"]["p_one_sided"] for k in ALL_SCORES]]
    q12 = bh(p12)
    for i, k in enumerate(ALL_SCORES):
        out["scores"][k]["null"]["q_bh_12_with_rpe1"] = float(q12[i])
    out["rpe1_q_bh_12_with_v3_k562"] = {k: float(q12[6 + i]) for i, k in enumerate(ALL_SCORES)}
    out["_method"] = {"pooled": "one AUROC over all (evaluated TF, candidate) pairs; tie-safe average ranks",
                      "ci": "percentile bootstrap over TFs, 2,000 reps, seed 20261001+500",
                      "null": "Curveball (v2_06 sampler), 1,000 draws x 5,000 trades, seed 20261001+600; "
                              "p = (1 + #null >= obs)/(1 + 1000), one-sided; z = (obs - mean)/SD"}
    out["seconds"] = time.time() - t0
    save_json(out, OUT3 / "trrust" / "trrust_summary.json")
    log("null checks", nchk)
    log({k: (round(v["pooled_auroc"], 4), [round(x, 4) for x in v["ci95"]], None if v["null"]["z"] is None
             else round(v["null"]["z"], 2)) for k, v in out["scores"].items()}, f"{out['seconds']:.0f}s")


def cmd_gene3() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    ctx = Ctx("main")
    S = ctx.scores()
    S["coexpr_abs_spearman"] = ctx.coexpr()
    sym, Etr, tf_rows, k_of, cand, src, y = ctx.trrust_setup()
    nT = len(tf_rows)
    gm, gv, gd = (ctx.gf[c].to_numpy(np.float64) for c in ["mean_expr", "variance", "dropout_rate"])
    X = np.column_stack([gm[cand], gv[cand], gd[cand]]).astype(np.float32)
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups=k_of):
        m = Pipeline([("sc", StandardScaler()),
                      ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))]).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    scores = {k: M[src, cand] for k, M in S.items()}
    scores["gene_model_3feat"] = oof
    pooled = {k: auroc_rank(y, v) for k, v in scores.items()}
    fast = {k: WeightedAUROC(y, v) for k, v in scores.items()}
    rng = np.random.default_rng(SEED + 500 + RI)
    boot = {k: [] for k in scores}
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, nT, nT), minlength=nT)[k_of]
        for k in scores:
            boot[k].append(fast[k](w))
    boot = {k: np.array(v) for k, v in boot.items()}
    # fold-assignment sensitivity: a random TF-to-fold map (seed 7)
    perm = np.random.default_rng(7).permutation(nT)
    oof2 = np.zeros(len(y))
    fold_of_tf = perm % 5
    for f in range(5):
        te = fold_of_tf[k_of] == f
        m = Pipeline([("sc", StandardScaler()),
                      ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))]).fit(X[~te], y[~te])
        oof2[te] = m.predict_proba(X[te])[:, 1]
    out = {"gene_model_auroc": pooled["gene_model_3feat"], "gene_model_ci95": percentile_ci(boot["gene_model_3feat"]),
           "gene_model_auroc_random_fold_map_seed7": auroc_rank(y, oof2),
           "gaps": {k: {"gap_gene_model_minus_score": pooled["gene_model_3feat"] - pooled[k],
                        "ci95": percentile_ci(boot["gene_model_3feat"] - boot[k])}
                    for k in scores if k != "gene_model_3feat"}}
    save_json(out, OUT3 / "trrust" / "gene_model_gaps.json")
    log(round(out["gene_model_auroc"], 4), out["gene_model_ci95"],
        {k: (round(v["gap_gene_model_minus_score"], 3), [round(x, 3) for x in v["ci95"]]) for k, v in out["gaps"].items()})


# ============================================================================================ residualisation
def _folds(n_pairs: int):
    from sklearn.model_selection import KFold
    return list(KFold(n_splits=5, shuffle=True, random_state=42).split(np.arange(n_pairs)))


def _ols_resid(target, X, folds):
    resid = np.empty_like(target)
    r2 = []
    for tr, te in folds:
        mu = X[tr].mean(0); sd = X[tr].std(0); sd[sd == 0] = 1.0
        Z = np.column_stack([np.ones(len(tr)), (X[tr] - mu) / sd])
        beta, *_ = np.linalg.lstsq(Z, target[tr], rcond=None)
        pt = Z @ beta
        r2.append(1 - ((target[tr] - pt) ** 2).sum() / (((target[tr] - target[tr].mean()) ** 2).sum() + 1e-300))
        resid[te] = target[te] - np.column_stack([np.ones(len(te)), (X[te] - mu) / sd]) @ beta
    return resid, float(np.mean(r2))


def _hgb_resid(target, X, folds):
    from sklearn.ensemble import HistGradientBoostingRegressor
    resid = np.empty_like(target)
    r2 = []
    for tr, te in folds:
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                          random_state=SEED).fit(X[tr], target[tr])
        pt = m.predict(X[tr])
        r2.append(1 - ((target[tr] - pt) ** 2).sum() / (((target[tr] - target[tr].mean()) ** 2).sum() + 1e-300))
        resid[te] = target[te] - m.predict(X[te])
    return resid, float(np.mean(r2))


def _resid_inputs(ctx: Ctx):
    S = ctx.scores()
    S["coexpr_abs_spearman"] = ctx.coexpr()
    sym, Etr, tf_rows, k_of, cand, src_e, y = ctx.trrust_setup()
    G = ctx.G
    ii, jj = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    mask = ii != jj
    src, tgt = ii[mask], jj[mask]
    gm, gv, gd = (ctx.gf[c].to_numpy(np.float64) for c in ["mean_expr", "variance", "dropout_rate"])
    gene6 = np.column_stack([gm[src], gv[src], gd[src], gm[tgt], gv[tgt], gd[tgt]])
    extra = np.column_stack([S["order_share_F"][src, tgt], (ctx.pc[src, tgt] == 0).astype(np.float64)])
    extra = extra[:, extra.std(0) > 0]          # drop a constant indicator (no pair with n_pair = 0)
    ols5 = np.column_stack([gm[src], gv[src], gm[tgt], gv[tgt], gd[tgt]])
    flat = src_e * (G - 1) + cand - (cand > src_e)
    return S, src, tgt, gene6, extra, ols5, flat, y, k_of, len(tf_rows)


def _score_resid(R, flat, y, k_of, draws, raw_boot, raw_auc, pos):
    s = R[flat]
    auc = auroc_rank(y, s)
    f = WeightedAUROC(y, s)
    boot = np.array([f(c[k_of]) for c in draws])
    nv = null_auroc(s, pos)
    z, p = zp(nv, auc)
    return {"residualised_auroc": auc, "ci95": percentile_ci(boot), "drop_from_raw": raw_auc - auc,
            "drop_ci95": percentile_ci(raw_boot - boot), "null_mean": float(nv.mean()), "null_z": z,
            "null_p_one_sided": p}


def cmd_resid(kind: str, max_seconds: float = 420.0) -> None:
    t0 = time.time()
    ctx = Ctx("main")
    S, src, tgt, gene6, extra, ols5, flat, y, k_of, nT = _resid_inputs(ctx)
    folds = _folds(len(src))
    rng = np.random.default_rng(SEED + 800 + RI)
    draws = [np.bincount(rng.integers(0, nT, nT), minlength=nT) for _ in range(N_BOOT)]
    pos = np.load(OUT3 / "trrust" / "null_positions.npy").astype(np.int64)
    od = OUT3 / "resid"
    od.mkdir(parents=True, exist_ok=True)
    if kind == "ols":
        res = {"scores": {}, "n_extra_columns": int(extra.shape[1])}
        for k in ALL_SCORES + ["coexpr_abs_spearman"]:
            target = S[k][src, tgt]
            raw = target[flat]
            raw_auc = auroc_rank(y, raw)
            fr = WeightedAUROC(y, raw)
            raw_boot = np.array([fr(c[k_of]) for c in draws])
            o = {"raw_auroc": raw_auc, "raw_ci95": percentile_ci(raw_boot)}
            sets = {"ols_gene6": gene6}
            if k != "order_share_F":
                sets["ols_gene6_F"] = np.column_stack([gene6, extra])
            if k == "forward":
                sets["ols5_deployed_features_f64"] = ols5
            for nm, X in sets.items():
                R, r2 = _ols_resid(target, X, folds)
                o[nm] = _score_resid(R, flat, y, k_of, draws, raw_boot, raw_auc, pos)
                o[nm]["r2_train_mean"] = r2
            res["scores"][k] = o
            log(k, {nm: round(v["residualised_auroc"], 4) for nm, v in o.items() if isinstance(v, dict) and "residualised_auroc" in v},
                f"{time.time() - t0:.0f}s")
        res["seconds"] = time.time() - t0
        save_json(res, od / "resid_ols.json")
    elif kind == "hgb":
        for k in ALL_SCORES:
            p = od / f"resid_hgb_{k}.json"
            if p.exists():
                continue
            if time.time() - t0 > max_seconds:
                log("time budget reached; re-run to continue"); return
            t1 = time.time()
            target = S[k][src, tgt]
            raw = target[flat]
            raw_auc = auroc_rank(y, raw)
            fr = WeightedAUROC(y, raw)
            raw_boot = np.array([fr(c[k_of]) for c in draws])
            X = gene6 if k == "order_share_F" else np.column_stack([gene6, extra])
            nm = "hgb_gene6" if k == "order_share_F" else "hgb_gene6_F"
            R, r2 = _hgb_resid(target, X, folds)
            o = _score_resid(R, flat, y, k_of, draws, raw_boot, raw_auc, pos)
            o.update({"features": nm, "r2_train_mean": r2, "raw_auroc": raw_auc, "seconds": time.time() - t1})
            save_json(o, p)
            log(k, nm, round(o["residualised_auroc"], 4), o["ci95"], f"{o['seconds']:.0f}s")
    elif kind == "combine":
        a = json.load(open(od / "resid_ols.json"))
        for k in ALL_SCORES:
            p = od / f"resid_hgb_{k}.json"
            if p.exists():
                h = json.load(open(p))
                a["scores"][k][h["features"]] = h
        for nm_edge, nm_F, tag in [("ols_gene6_F", "ols_gene6", "main_ols"), ("hgb_gene6_F", "hgb_gene6", "hgb")]:
            fam = [(k, nm_F if k == "order_share_F" else nm_edge) for k in ALL_SCORES
                   if (nm_F if k == "order_share_F" else nm_edge) in a["scores"][k]]
            q = bh(np.array([a["scores"][k][nm]["null_p_one_sided"] for k, nm in fam]))
            for (k, nm), qq in zip(fam, q):
                a["scores"][k][nm][f"null_q_bh_{tag}_family"] = float(qq)
                a["scores"][k][nm]["n_tests_in_bh_family"] = len(fam)
        a["_method"] = {
            "pairs": "all ordered off-diagonal pairs (1,500 x 1,499); 5-fold KFold(shuffle, random_state=42) cross-fitting",
            "ols_gene6": "OLS (float64 lstsq, standardised) on mean, variance, dropout of both genes (single-log features)",
            "ols_gene6_F": "ols_gene6 + order share F[P,T] (+ indicator n_pair = 0 when not constant)",
            "ols5_deployed_features_f64": "deployed feature set (source mean, var; target mean, var, dropout), float64",
            "hgb_gene6_F": "HistGradientBoostingRegressor(max_iter=200, lr=0.1, max_leaf_nodes=31) on gene6 + F",
            "ci": "percentile bootstrap over TFs, 2,000 reps, seed 20261001+800; fitted models held fixed",
            "null": "same Curveball networks as the trrust step; z and one-sided p of the residualised pooled AUROC"}
        save_json(a, od / "resid_summary.json")
        log({k: {nm: round(v["residualised_auroc"], 3) for nm, v in o.items() if isinstance(v, dict) and "residualised_auroc" in v}
             for k, o in a["scores"].items()})
    else:
        raise SystemExit("kind must be ols, hgb or combine")


# ============================================================================================ co-expression
def cmd_coexpr() -> None:
    t0 = time.time()
    ctx = Ctx("main")
    S = ctx.scores()
    co = ctx.coexpr()
    sym, Etr, tf_rows, k_of, cand, src_e, y = ctx.trrust_setup()
    G = ctx.G
    off = ~np.eye(G, dtype=bool)
    pos = np.load(OUT3 / "trrust" / "null_positions.npy").astype(np.int64)
    flat_rows, flat_cols = src_e, cand
    out = {"spearman_with_coexpr_all_offdiag_pairs": {}, "after_removing_coexpr": {}, "coexpr_after_removing": {},
           "null_of_difference": {}}
    cv = co[off]
    edges = np.quantile(cv, np.linspace(0, 1, 51))
    bins = np.clip(np.searchsorted(edges, co, side="right") - 1, 0, 49)

    def binned_resid(s, b):
        mean_b = np.bincount(b[off], weights=s[off], minlength=50) / np.maximum(np.bincount(b[off], minlength=50), 1)
        return s - mean_b[b]

    def lin_resid(s, x):
        A = np.column_stack([np.ones(off.sum()), x[off]])
        beta, *_ = np.linalg.lstsq(A, s[off], rcond=None)
        return s - (beta[0] + beta[1] * x)

    def score(M):
        v = M[flat_rows, flat_cols]
        a = auroc_rank(y, v)
        z, p = zp(null_auroc(v, pos), a)
        return {"auroc": a, "null_z": z, "null_p": p}

    co_scores = co[flat_rows, flat_cols]
    co_null = null_auroc(co_scores, pos)
    co_auc = auroc_rank(y, co_scores)
    for k in ["forward", "sym_mean", "sym_max", "transpose"]:
        out["spearman_with_coexpr_all_offdiag_pairs"][k] = float(spearmanr(S[k][off], cv).correlation)
        out["after_removing_coexpr"][k] = {"linear": score(lin_resid(S[k], co)), "bins50": score(binned_resid(S[k], bins))}
        vk = S[k][flat_rows, flat_cols]
        d_obs = auroc_rank(y, vk) - co_auc
        d_null = null_auroc(vk, pos) - co_null
        z, p = zp(d_null, d_obs)
        out["null_of_difference"][k] = {"auroc_minus_coexpr": d_obs, "null_mean": float(d_null.mean()), "z": z, "p": p}
    for k in ["sym_mean", "sym_max", "forward"]:
        eb = np.clip(np.searchsorted(np.quantile(S[k][off], np.linspace(0, 1, 51)), S[k], side="right") - 1, 0, 49)
        out["coexpr_after_removing"][k] = {"linear": score(lin_resid(co, S[k])), "bins50": score(binned_resid(co, eb))}
    out["coexpr_raw"] = {"auroc": co_auc, "null_z": zp(co_null, co_auc)[0]}
    out["seconds"] = time.time() - t0
    save_json(out, OUT3 / "coexpr" / "coexpr_comparison.json")
    log(json.dumps(out, indent=1))


# ============================================================================================ layers
def cmd_layers() -> None:
    t0 = time.time()
    ctx = Ctx("main")
    sym, Etr, tf_rows, k_of, cand, src, y = ctx.trrust_setup()
    pos = np.load(OUT3 / "trrust" / "null_positions.npy").astype(np.int64)
    d = np.load(OUT3 / "de_labels" / "v3_primary.npz")
    hs, Y = d["pert_hvg_idx"], d["labels"]
    G = ctx.G
    rows, zmat = [], {}
    for li in range(ctx.E_all.shape[0]):
        E = np.array(ctx.E_all[li], dtype=np.float64)
        np.fill_diagonal(E, 0.0)
        S = ctx.scores(E)
        for k in ["forward", "sym_mean"]:
            v = S[k][src, cand]
            a = auroc_rank(y, v)
            nv = null_auroc(v, pos)
            zmat.setdefault(k, []).append(((a - nv.mean()) / nv.std(ddof=1), (nv - nv.mean()) / nv.std(ddof=1)))
            kd = []
            for h, yy in zip(hs, Y):
                valid = np.ones(G, bool); valid[h] = False
                kd.append(auroc_rank(yy[valid], S[k][h][valid]))
            rows.append({"layer": li, "score": k, "trrust_pooled_auroc": a, "trrust_null_z": float((a - nv.mean()) / nv.std(ddof=1)),
                         "knockdown_mean_auroc": float(np.mean(kd)) if kd else None})
    df = pd.DataFrame(rows)
    (OUT3 / "layers").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT3 / "layers" / "per_layer.csv", index=False)
    maxz = {}
    for k, lst in zmat.items():
        zobs = np.array([a for a, _ in lst])
        znull = np.stack([b for _, b in lst]).max(0)
        maxz[k] = {"max_z_over_layers": float(zobs.max()), "best_layer": int(zobs.argmax()),
                   "p_layer_corrected": float((1 + (znull >= zobs.max()).sum()) / (1 + len(znull)))}
    save_json({"max_z_test": maxz, "seconds": time.time() - t0}, OUT3 / "layers" / "layers_summary.json")
    log(df.to_string(), maxz)


# ============================================================================================ deployed gene set
def cmd_oldgenes() -> None:
    """New-encoding attention, read on the deployed 1,500 genes, scored with exactly the v2b inputs
    (v2 double-log labels, deployed gene features, deployed co-expression, the v2b Curveball networks).
    The only difference from v2b K562 is the attention (and so C, n_pair, F), which now comes from
    correctly encoded cells. Paired differences new - old use the same perturbation / TF draws."""
    t0 = time.time()
    ctx = Ctx("deployed_genes")
    S_new = ctx.scores()
    # old (deployed) attention and order counts, exactly as v2b used them
    Eo = np.array(np.load(OUT / "phase0" / "attention_edges_layer_mean.npy", mmap_mode="r")[LAYER], dtype=np.float64)
    np.fill_diagonal(Eo, 0.0)
    pco = np.load(OUT / "phase0" / "attention_pair_counts.npy").astype(np.float64)
    Co = np.load(V2B / "order" / "order_counts_217M_K562.npy").astype(np.float64)
    Fo = np.where(pco > 0, Co / np.maximum(pco, 1.0), 0.0); np.fill_diagonal(Fo, 0.0)
    S_old = {"forward": Eo, "transpose": Eo.T.copy(), "sym_mean": 0.5 * (Eo + Eo.T), "sym_max": np.maximum(Eo, Eo.T),
             "order_share_F": Fo}
    keys = ["forward", "transpose", "sym_mean", "sym_max", "order_share_F"]
    G = ctx.G
    out = {"what": "v3 attention (correct encoding) on the deployed gene set vs the deployed attention; same labels, "
                   "features, TFs and null networks as v2b 217M K562", "knockdown": {}, "trrust": {}}
    # --- knockdown: v2 labels (double log) and single-log labels on the same genes
    for lab_name, lab_path in [("v2_labels_double_log", V2 / "de_labels" / "217M_K562.npz"),
                               ("single_log_labels", OUT3 / "de_labels" / "deployed_genes_single_log.npz")]:
        d = np.load(lab_path)
        hs, Y = d["pert_hvg_idx"], d["labels"]
        A = {f"{w}_{k}": [] for w in ("new", "old") for k in keys}
        for h, y in zip(hs, Y):
            valid = np.ones(G, bool); valid[h] = False
            for k in keys:
                A[f"new_{k}"].append(auroc_rank(y[valid], S_new[k][h][valid]))
                A[f"old_{k}"].append(auroc_rank(y[valid], S_old[k][h][valid]))
        A = {k: np.array(v) for k, v in A.items()}
        n = len(hs)
        rng = np.random.default_rng(SEED + 100 + RI)
        B = rng.integers(0, n, size=(N_BOOT, n))
        o = {"n_perturbations": int(n), "n_positive_pairs": int(Y.sum())}
        for k in keys:
            bn, bo = A[f"new_{k}"][B].mean(1), A[f"old_{k}"][B].mean(1)
            o[k] = {"new": float(A[f"new_{k}"].mean()), "new_ci95": percentile_ci(bn),
                    "old": float(A[f"old_{k}"].mean()), "old_ci95": percentile_ci(bo),
                    "new_minus_old": float(A[f"new_{k}"].mean() - A[f"old_{k}"].mean()),
                    "new_minus_old_ci95": percentile_ci(bn - bo)}
        out["knockdown"][lab_name] = o
    if (V2B / "knockdown" / "per_perturbation_217M_K562.csv").exists():
        v2pp = pd.read_csv(V2B / "knockdown" / "per_perturbation_217M_K562.csv")
        d = np.load(V2 / "de_labels" / "217M_K562.npz")
        old_fw = []
        for h, y in zip(d["pert_hvg_idx"], d["labels"]):
            valid = np.ones(G, bool); valid[h] = False
            old_fw.append(auroc_rank(y[valid], S_old["forward"][h][valid]))
        out["check_old_forward_equals_v2b_max_abs_diff"] = float(np.max(np.abs(np.array(old_fw) - v2pp["auc_forward"].to_numpy())))
    # --- TRRUST on the deployed genes, v2b null networks
    sym, Etr, tf_rows, k_of, cand, src, y = ctx.trrust_setup()
    nT = len(tf_rows)
    pos = np.load(V2B / "trrust" / "null_positions_217M_K562.npy").astype(np.int64)
    rng = np.random.default_rng(SEED + 500 + RI)
    draws = [np.bincount(rng.integers(0, nT, nT), minlength=nT)[k_of] for _ in range(N_BOOT)]
    for k in keys:
        vn, vo = S_new[k][src, cand], S_old[k][src, cand]
        an, ao = auroc_rank(y, vn), auroc_rank(y, vo)
        fn, fo = WeightedAUROC(y, vn), WeightedAUROC(y, vo)
        bn = np.array([fn(w) for w in draws]); bo = np.array([fo(w) for w in draws])
        zn, pn = zp(null_auroc(vn, pos), an)
        zo, po = zp(null_auroc(vo, pos), ao)
        out["trrust"][k] = {"new": an, "new_ci95": percentile_ci(bn), "new_null_z": zn, "new_null_p": pn,
                            "old": ao, "old_ci95": percentile_ci(bo), "old_null_z": zo, "old_null_p": po,
                            "new_minus_old": an - ao, "new_minus_old_ci95": percentile_ci(bn - bo)}
    v2t = json.load(open(V2B / "trrust" / "trrust_variants_217M_K562.json"))
    out["check_old_trrust_equals_v2b"] = {k: abs(out["trrust"][k]["old"] - v2t["scores"][k]["pooled_auroc"]) for k in keys}
    out["check_old_null_z_equals_v2b"] = {k: abs(out["trrust"][k]["old_null_z"] - v2t["scores"][k]["null"]["z"]) for k in keys}
    out["n_tfs"] = int(nT)
    # how similar are the two edges, and the two order shares?
    m = (~np.eye(G, dtype=bool)) & (ctx.pc > 0) & (pco > 0)
    out["spearman_new_vs_old_edge_all_pairs"] = float(spearmanr(S_new["forward"][m], Eo[m]).correlation)
    out["spearman_new_vs_old_F_all_pairs"] = float(spearmanr(S_new["order_share_F"][m], Fo[m]).correlation)
    out["seconds"] = time.time() - t0
    save_json(out, OUT3 / "oldgenes" / "deployed_genes_comparison.json")
    log(json.dumps(out, indent=1))


# ============================================================================================ tables
LABEL = {"forward": "forward attn[P,T] (deployed definition)", "transpose": "transpose attn[T,P]",
         "sym_mean": "symmetric mean", "sym_max": "symmetric max", "rank_conditioned": "rank-conditioned attn[P,T]",
         "order_share_F": "order share F (no model)", "gene_variance": "target variance (single log)",
         "gene_variance_storedX": "target variance (stored log1p CP10k)",
         "coexpr_abs_spearman": "|Spearman| co-expression (no model)",
         "gene_model_3feat": "3-feature gene model (logistic, TF-grouped CV)"}
KIND = {k: "attention edge" for k in EDGE_SCORES}
KIND.update({"order_share_F": "model-free order", "coexpr_abs_spearman": "model-free co-expression (reference)",
             "gene_variance": "gene-level baseline", "gene_variance_storedX": "gene-level baseline (sensitivity)",
             "gene_model_3feat": "gene-level baseline"})
RUN_NAME = "217M_K562_v3"
SCALE = "counts (correct encoding); single-log features and labels"


def cmd_table() -> None:
    kd = json.load(open(OUT3 / "knockdown" / "knockdown_summary.json"))["variants"]["v3_primary"]
    tr = json.load(open(OUT3 / "trrust" / "trrust_summary.json"))
    g3 = json.load(open(OUT3 / "trrust" / "gene_model_gaps.json"))
    rs = json.load(open(OUT3 / "resid" / "resid_summary.json"))
    tidy, fig = [], []
    for k, o in kd["scores"].items():
        tidy.append({"run": RUN_NAME, "input_scale": SCALE, "endpoint": "knockdown", "score": k, "score_label": LABEL[k],
                     "score_kind": KIND[k], "n_units": kd["n_perturbations"], "unit": "perturbation",
                     "auroc": o["mean_auroc"], "ci_low": o["ci95"][0], "ci_high": o["ci95"][1],
                     "gap_variance_minus_score": o["gap_variance_minus_score"],
                     "gap_ci_low": o["gap_variance_minus_score_ci95"][0], "gap_ci_high": o["gap_variance_minus_score_ci95"][1],
                     "delta_vs_forward": o.get("delta_vs_forward"),
                     "delta_ci_low": (o.get("delta_vs_forward_ci95") or [None, None])[0],
                     "delta_ci_high": (o.get("delta_vs_forward_ci95") or [None, None])[1],
                     "delta_vs_coexpr": o.get("delta_vs_coexpr"),
                     "delta_vs_coexpr_ci_low": (o.get("delta_vs_coexpr_ci95") or [None, None])[0],
                     "delta_vs_coexpr_ci_high": (o.get("delta_vs_coexpr_ci95") or [None, None])[1],
                     "wilcoxon_p_vs_forward": o.get("wilcoxon_p_vs_forward"),
                     "wilcoxon_p_vs_variance": o.get("wilcoxon_p_vs_variance"),
                     "n_units_score_gt_variance": o["n_perts_score_gt_variance"]})
        fig.append({"run": RUN_NAME, "input_scale": SCALE, "endpoint": "knockdown", "score": k, "score_label": LABEL[k],
                    "score_kind": KIND[k], "measure": "mean per-perturbation AUROC", "value": o["mean_auroc"],
                    "ci_low": o["ci95"][0], "ci_high": o["ci95"][1],
                    "ci_method": "percentile bootstrap over perturbations, 2,000 reps", "n_units": kd["n_perturbations"]})
    for k, o in tr["scores"].items():
        nl = o["null"]
        row = {"run": RUN_NAME, "input_scale": SCALE, "endpoint": "trrust", "score": k, "score_label": LABEL[k],
               "score_kind": KIND[k], "n_units": tr["n_tfs"], "unit": "TF", "auroc": o["pooled_auroc"],
               "ci_low": o["ci95"][0], "ci_high": o["ci95"][1], "gap_variance_minus_score": o["gap_variance_minus_score"],
               "gap_ci_low": o["gap_variance_minus_score_ci95"][0], "gap_ci_high": o["gap_variance_minus_score_ci95"][1],
               "delta_vs_forward": o.get("delta_vs_forward"),
               "delta_ci_low": (o.get("delta_vs_forward_ci95") or [None, None])[0],
               "delta_ci_high": (o.get("delta_vs_forward_ci95") or [None, None])[1],
               "delta_vs_coexpr": o.get("delta_vs_coexpr"),
               "delta_vs_coexpr_ci_low": (o.get("delta_vs_coexpr_ci95") or [None, None])[0],
               "delta_vs_coexpr_ci_high": (o.get("delta_vs_coexpr_ci95") or [None, None])[1],
               "n_units_score_gt_variance": o["n_tfs_score_gt_variance"],
               "null_mean": nl["mean"], "null_sd": nl["sd"], "null_z": nl["z"], "null_p": nl["p_one_sided"],
               "null_q_bh_6": nl.get("q_bh_6_scores"), "null_q_bh_12_with_rpe1": nl.get("q_bh_12_with_rpe1")}
        gg = g3["gaps"].get(k)
        if gg is not None:
            row["gap_gene_model_minus_score"] = gg["gap_gene_model_minus_score"]
            row["gap_gene_model_ci_low"], row["gap_gene_model_ci_high"] = gg["ci95"]
        if k in rs["scores"]:
            for nm, v in rs["scores"][k].items():
                if isinstance(v, dict) and "residualised_auroc" in v and nm != "ols5_deployed_features_f64":
                    row[f"resid_{nm}_auroc"] = v["residualised_auroc"]
                    row[f"resid_{nm}_ci_low"], row[f"resid_{nm}_ci_high"] = v["ci95"]
                    row[f"resid_{nm}_null_z"] = v["null_z"]
                    row[f"resid_{nm}_null_p"] = v["null_p_one_sided"]
                    qk = [kk for kk in v if kk.startswith("null_q_bh_")]
                    if qk:
                        row[f"resid_{nm}_null_q_bh"] = v[qk[0]]
                    fig.append({"run": RUN_NAME, "input_scale": SCALE, "endpoint": "trrust", "score": k,
                                "score_label": LABEL[k], "score_kind": KIND[k],
                                "measure": f"pooled AUROC after residualising ({nm})", "value": v["residualised_auroc"],
                                "ci_low": v["ci95"][0], "ci_high": v["ci95"][1],
                                "ci_method": "percentile bootstrap over TFs, 2,000 reps, fitted models held fixed",
                                "n_units": tr["n_tfs"]})
        tidy.append(row)
        fig.append({"run": RUN_NAME, "input_scale": SCALE, "endpoint": "trrust", "score": k, "score_label": LABEL[k],
                    "score_kind": KIND[k], "measure": "pooled AUROC", "value": o["pooled_auroc"], "ci_low": o["ci95"][0],
                    "ci_high": o["ci95"][1], "ci_method": "percentile bootstrap over TFs, 2,000 reps", "n_units": tr["n_tfs"]})
        if nl["z"] is not None:
            fig.append({"run": RUN_NAME, "input_scale": SCALE, "endpoint": "trrust", "score": k, "score_label": LABEL[k],
                        "score_kind": KIND[k], "measure": "Curveball null z", "value": nl["z"], "ci_low": None,
                        "ci_high": None, "ci_method": "none (z = (obs - null mean)/null SD, 1,000 draws)",
                        "n_units": tr["n_tfs"]})
    tidy.append({"run": RUN_NAME, "input_scale": SCALE, "endpoint": "trrust", "score": "gene_model_3feat",
                 "score_label": LABEL["gene_model_3feat"], "score_kind": KIND["gene_model_3feat"], "n_units": tr["n_tfs"],
                 "unit": "TF", "auroc": g3["gene_model_auroc"], "ci_low": g3["gene_model_ci95"][0],
                 "ci_high": g3["gene_model_ci95"][1]})
    fig.append({"run": RUN_NAME, "input_scale": SCALE, "endpoint": "trrust", "score": "gene_model_3feat",
                "score_label": LABEL["gene_model_3feat"], "score_kind": KIND["gene_model_3feat"], "measure": "pooled AUROC",
                "value": g3["gene_model_auroc"], "ci_low": g3["gene_model_ci95"][0], "ci_high": g3["gene_model_ci95"][1],
                "ci_method": "percentile bootstrap over TFs, 2,000 reps, fitted models held fixed", "n_units": tr["n_tfs"]})
    pd.DataFrame(tidy).to_csv(OUT3 / "table_attention_variants_v3_k562.csv", index=False)
    pd.DataFrame(fig).to_csv(OUT3 / "fig_attention_variants_v3_k562.csv", index=False)
    # comparison: v2b K562 (deployed, wrong encoding), v3 K562, v2b RPE1 (correct, unchanged)
    v2tab = pd.read_csv(V2B / "table_attention_variants.csv")
    comp = []
    for _, r in pd.DataFrame(tidy).iterrows():
        sc = r["score"]
        sc_v2 = "gene_variance_single_log" if sc == "gene_variance" else sc
        row = {"endpoint": r["endpoint"], "score": sc, "v3_K562_auroc": r["auroc"], "v3_K562_ci_low": r["ci_low"],
               "v3_K562_ci_high": r["ci_high"], "v3_K562_null_z": r.get("null_z")}
        for run, tag in [("217M_K562", "v2b_K562"), ("217M_RPE1", "v2b_RPE1")]:
            name = sc_v2 if run == "217M_K562" else ("gene_variance" if sc == "gene_variance" else sc)
            q = v2tab[(v2tab["run"] == run) & (v2tab["endpoint"] == r["endpoint"]) & (v2tab["score"] == name)]
            if len(q):
                row[f"{tag}_auroc"] = float(q["auroc"].iloc[0])
                row[f"{tag}_ci_low"], row[f"{tag}_ci_high"] = float(q["ci_low"].iloc[0]), float(q["ci_high"].iloc[0])
                if "null_z" in q:
                    row[f"{tag}_null_z"] = q["null_z"].iloc[0]
        comp.append(row)
    pd.DataFrame(comp).to_csv(OUT3 / "comparison_v2b_k562_v3_k562_v2b_rpe1.csv", index=False)
    log("rows", len(tidy), len(fig), len(comp))


# ============================================================================================ leave one TF out
def cmd_loo() -> None:
    """Leave-one-TF-out sensitivity of the TRRUST null z (raw scores), on the v3 gene set (Curveball networks
    of the trrust step) and on the deployed gene set (v2b networks, new-encoding attention)."""
    t0 = time.time()
    out = {}
    for kind, pos_path in [("main", OUT3 / "trrust" / "null_positions.npy"),
                           ("deployed_genes", V2B / "trrust" / "null_positions_217M_K562.npy")]:
        ctx = Ctx(kind)
        S = ctx.scores()
        S["coexpr_abs_spearman"] = ctx.coexpr()
        sym, Etr, tf_rows, k_of, cand, src, y = ctx.trrust_setup()
        pos = np.load(pos_path).astype(np.int64)
        nT = len(tf_rows)
        G = ctx.G
        res = {}
        for k in ["forward", "sym_mean", "sym_max", "coexpr_abs_spearman"]:
            v = S[k][src, cand]
            rows = []
            for drop in range(nT):
                keep = k_of != drop
                newidx = np.cumsum(keep) - 1                     # old flat index -> new flat index
                vv, yy = v[keep], y[keep]
                a = auroc_rank(yy, vv)
                # null: drop positives that belong to the dropped TF (row block), keep the rest of each draw
                blk = (pos // (G - 1)) != drop
                r = rankdata(vv, method="average")
                nulls = np.empty(len(pos))
                for d_ in range(len(pos)):
                    pp = newidx[pos[d_][blk[d_]]]
                    P = len(pp); N = len(vv) - P
                    nulls[d_] = (r[pp].sum() - P * (P + 1) / 2.0) / (P * N)
                z, p = zp(nulls, a)
                rows.append({"dropped_tf": sym[tf_rows[drop]], "auroc": a, "z": z, "p": p})
            df = pd.DataFrame(rows)
            res[k] = {"z_min": float(df["z"].min()), "z_max": float(df["z"].max()), "p_max": float(df["p"].max()),
                      "tf_at_z_min": str(df.loc[df["z"].idxmin(), "dropped_tf"]), "rows": rows}
        out[kind] = {"n_tfs": int(nT), "scores": res}
        log(kind, {k: (round(v["z_min"], 2), round(v["z_max"], 2), round(v["p_max"], 3), v["tf_at_z_min"]) for k, v in res.items()})
    out["seconds"] = time.time() - t0
    save_json(out, OUT3 / "trrust" / "leave_one_tf_out.json")


# ============================================================================================ deployed genes: co-expression
def cmd_oldgenes_coexpr() -> None:
    """Single-log |Spearman| co-expression on the DEPLOYED gene set (same 2,000 cells), scored on TRRUST with the
    v2b networks, so the deployed-gene attention result has a co-expression reference on the same scale."""
    t0 = time.time()
    ctx = Ctx("deployed_genes")
    cnt = np.load(D_PREP / "counts_int32.npy").astype(np.float64)
    L = np.log1p(cnt / cnt.sum(1, keepdims=True) * 1e4)[:, ctx.gf["var_idx"].to_numpy(np.int64)]
    R = np.apply_along_axis(rankdata, 0, L)
    R -= R.mean(0, keepdims=True)
    R /= (R.std(0, keepdims=True) + 1e-12)
    co = np.abs((R.T @ R) / L.shape[0])
    np.fill_diagonal(co, 0.0)
    sym, Etr, tf_rows, k_of, cand, src, y = ctx.trrust_setup()
    pos = np.load(V2B / "trrust" / "null_positions_217M_K562.npy").astype(np.int64)
    nT = len(tf_rows)
    rng = np.random.default_rng(SEED + 500 + RI)
    draws = [np.bincount(rng.integers(0, nT, nT), minlength=nT)[k_of] for _ in range(N_BOOT)]
    out = {}
    S = ctx.scores()
    for name, M in [("coexpr_single_log", co), ("coexpr_deployed_double_log", ctx.coexpr()),
                    ("forward_new", S["forward"]), ("sym_mean_new", S["sym_mean"])]:
        v = M[src, cand]
        a = auroc_rank(y, v)
        f = WeightedAUROC(y, v)
        b = np.array([f(w) for w in draws])
        z, p = zp(null_auroc(v, pos), a)
        out[name] = {"auroc": a, "ci95": percentile_ci(b), "null_z": z, "null_p": p}
    m = ~np.eye(ctx.G, dtype=bool)
    out["spearman_sym_mean_new_vs_coexpr_single_log_all_pairs"] = float(spearmanr(S["sym_mean"][m], co[m]).correlation)
    out["seconds"] = time.time() - t0
    save_json(out, OUT3 / "oldgenes" / "deployed_genes_coexpr_single_log.json")
    log(json.dumps(out, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    cmd = a[0]
    if cmd == "de":
        cmd_de()
    elif cmd == "order":
        cmd_order()
    elif cmd == "knockdown":
        cmd_knockdown()
    elif cmd == "trrust":
        cmd_trrust()
    elif cmd == "gene3":
        cmd_gene3()
    elif cmd == "resid":
        cmd_resid(a[1], float(a[2]) if len(a) > 2 else 420.0)
    elif cmd == "coexpr":
        cmd_coexpr()
    elif cmd == "layers":
        cmd_layers()
    elif cmd == "oldgenes":
        cmd_oldgenes()
    elif cmd == "table":
        cmd_table()
    elif cmd == "loo":
        cmd_loo()
    elif cmd == "oldgenes_coexpr":
        cmd_oldgenes_coexpr()
    else:
        raise SystemExit(__doc__)
