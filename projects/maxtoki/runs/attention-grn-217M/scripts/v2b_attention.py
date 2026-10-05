"""v2b_attention — direction and rank baselines for the attention-GRN edge (revision item D1b).

CPU only. No model forward pass. No model weights are loaded. The MaxToki tokeniser
(setup/maxtoki_adapter.py, a pure numpy rank encoder) is used only to rebuild the order
of the gene tokens in each control cell, exactly as Phase 0 built it.

Why this exists
---------------
MaxToki is a causal decoder over genes sorted by falling (median-normalised)
expression. The deployed edge is attn[P, T] with P (the regulator / perturbed gene) as
the QUERY and T (the candidate target) as the KEY, at layer 8, mean over heads. In a
cell where T comes after P, the causal mask forces attn[P, T] = 0. Phase 0 divided the
summed attention by ALL cells that contain both genes (n_pair), so

    E[P, T] = F[P, T] * Ec[P, T]

    F[P, T]  = C[P, T] / n_pair[P, T]    share of co-occurring cells in which T is
                                          ranked above P (model-free; "order share")
    Ec[P, T] = E[P, T] * n_pair / C      mean attention over the cells where the mask
                                          allows it ("rank-conditioned edge")
    C[P, T]  = number of control cells that contain both genes and put T before P.

Scores compared (all from the saved layer-8 head-mean tensor; diagonal set to 0)
  forward           E[P, T]                       deployed score
  transpose         E[T, P]
  sym_mean          (E[P, T] + E[T, P]) / 2
  sym_max           max(E[P, T], E[T, P])
  rank_conditioned  Ec[P, T]  (0 where C = 0)
  order_share_F     F[P, T]   (0 where n_pair = 0)   -- no model
Gene-level baselines: target variance as deployed (gene_features.csv) and, for the
three runs whose h5ad X holds log1p(CP10k) (217M K562, 217M Adamson, 1B K562),
target variance on a single log scale (variance of the stored values).
Reference pair score (outside the 6-score test family): |Spearman| co-expression in the
same control cells (deployed spearman_edges.npy).

Endpoints (rules exactly as in V2_EVAL_REPORT.md)
  A knockdown: per-perturbation AUROC on the v2 DE labels (outputs/v2_eval/de_labels),
    summary = mean over perturbations; 95% CI = percentile bootstrap over perturbations
    (2,000 reps, same seeds as v2_03); paired Wilcoxon (zsplit, two-sided).
  B TRRUST: pooled AUROC over evaluated TFs (>= 3 targets in the gene set); 95% CI =
    percentile bootstrap over TFs (2,000 reps, same seeds as v2_05); Curveball
    degree-preserving null regenerated with the v2_06 sampler and seeds (1,000 draws x
    5,000 trades; the same networks as v2); residualisation of every score on
    gene-level features (+ F) with 5-fold cross-fitting over all ordered pairs.

Subcommands
  order <run>                  token order -> C; checks              outputs/v2b_attention/order/
  knockdown                    Endpoint A, all runs                  outputs/v2b_attention/knockdown/
  trrust <run> | combine       Endpoint B raw scores + null          outputs/v2b_attention/trrust/
  gene3 <run>                  Endpoint B: v2 3-feature gene model as a baseline (gap CIs)
  resid <run> ols              residualisation, OLS float64          outputs/v2b_attention/resid/
  resid <run> hgb [max_s]      HGB check, resumable per score
  resid combine
  table                        tidy tables + fig_attention_variants.csv
  config [max_s]               run_config.json with sha256 of inputs / scripts / outputs
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True   # do not write .pyc files into existing folders

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from v2_common import (PROJ, RUNS, RUN_DIR, OUT, V2, N_BOOT, SEED, PRIMARY_LAYER,  # noqa: E402
                       TRRUST_TSV, SYM2ENS_PKL, phase_dir, load_gene_features,
                       load_attention_layer, load_trrust, trrust_matrix, evaluated_tf_rows,
                       auroc_rank, percentile_ci, bh, save_json, sha256, WeightedAUROC,
                       tie_self_test)

OUTB = OUT / "v2b_attention"
for sub in ["order", "knockdown", "trrust", "resid"]:
    (OUTB / sub).mkdir(parents=True, exist_ok=True)

MAX_LEN = 2048                      # deployed Phase-0 value for all four runs
LOG_RUNS = {"217M_K562", "217M_Adamson", "1B_K562"}   # X holds log1p(CP10k) (V2_EVAL V3)
EDGE_SCORES = ["forward", "transpose", "sym_mean", "sym_max", "rank_conditioned"]
ALL_SCORES = EDGE_SCORES + ["order_share_F"]
N_DRAWS = 1000                      # Curveball draws (as v2_06)
N_TRADES = 5000                     # trades per draw (as v2_06)


def input_scale(run: str) -> str:
    return "log1p(CP10k) fed as counts" if run in LOG_RUNS else "raw counts"


# ----------------------------------------------------------------------------- order
def _read_rows(h5_path: Path, rows: np.ndarray, n_genes: int) -> np.ndarray:
    import h5py
    X = np.empty((len(rows), n_genes), dtype=np.float32)
    with h5py.File(h5_path, "r") as f:
        Xd = f["X"]
        if isinstance(Xd, h5py.Dataset):
            for i in range(0, len(rows), 100):
                X[i:i + 100] = Xd[rows[i:i + 100], :]
        else:                                  # CSR group (Adamson)
            indptr = Xd["indptr"][:]
            X[:] = 0.0
            for i, r in enumerate(rows):
                a, b = int(indptr[r]), int(indptr[r + 1])
                X[i, Xd["indices"][a:b]] = Xd["data"][a:b].astype(np.float32)
    return X


def cmd_order(run: str) -> None:
    t0 = time.time()
    sys.path.insert(0, str(PROJ / "setup"))
    from dataset_loader import resolve as load_ds          # noqa: E402
    from maxtoki_adapter import MaxTokiTokenizer            # noqa: E402  (numpy tokeniser only)
    _, dskey, _ = RUNS[run]
    p0 = phase_dir("phase0", run)
    ds = load_ds(dskey)
    rows = pd.read_csv(p0 / "control_cells.csv")["cell_idx_global"].to_numpy(np.int64)
    assert np.all(np.diff(rows) > 0), "control rows must be sorted (as deployed)"
    hv = pd.read_csv(p0 / "hvg_gene_table.csv")
    hvg_var = hv["var_idx"].to_numpy(np.int64)
    G = len(hvg_var)
    assert list(hv["ensembl_id"]) == [ds.var_ensembl[i] for i in hvg_var], "gene table mismatch"
    X = _read_rows(ds.h5_path, rows, ds.n_genes_total)
    n = len(rows)
    t_read = time.time() - t0

    # input-scale facts and single-log gene variance
    nz = X[X > 0]
    facts = {"n_cells": int(n), "n_genes_in_file": int(ds.n_genes_total),
             "frac_nonzero_values_whole_numbers": float(np.mean(nz == np.round(nz))),
             "max_value": float(X.max())}
    Xh = X[:, hvg_var].astype(np.float64)
    if run in LOG_RUNS:
        var_single = Xh.var(0)                                     # stored values are log1p(CP10k)
        facts["median_cell_sum_expm1_X"] = float(np.median(np.expm1(X.astype(np.float64)).sum(1)))
    else:
        rs = X.astype(np.float64).sum(1, keepdims=True); rs[rs == 0] = 1.0
        var_single = np.log1p(X[:, hvg_var].astype(np.float64) / rs * 1e4).var(0)
    gf = load_gene_features(run)
    facts["spearman_single_log_var_vs_deployed_var"] = float(spearmanr(var_single, gf["variance"]).correlation)
    facts["max_abs_diff_single_log_var_vs_deployed_var"] = float(np.max(np.abs(var_single - gf["variance"].to_numpy())))
    pd.DataFrame({"hvg_idx": np.arange(G), "symbol": gf["symbol"], "variance_deployed": gf["variance"],
                  "variance_single_log": var_single}).to_csv(OUTB / "order" / f"gene_variance_single_log_{run}.csv", index=False)

    # rebuild the token order exactly as Phase 0 did
    tok = MaxTokiTokenizer()
    vi, vt, vm = tok.make_var_mapping(ds.var_ensembl)
    var_to_hvg = -np.ones(ds.n_genes_total, dtype=np.int64)
    var_to_hvg[hvg_var] = np.arange(G)
    mt = var_to_hvg[vi]
    R = np.full((n, G), np.inf, dtype=np.float32)      # sequence position of each HVG; inf = absent
    seq_lens, n_used = [], 0
    for ci in range(n):
        cell = tok.tokenize_cell(X[ci], vi, vt, vm, max_len=MAX_LEN)
        if cell is None:
            continue
        pos = cell.gene_positions
        hpp = np.full(len(pos), -1, dtype=np.int64)
        iv = pos >= 0
        hpp[iv] = mt[pos[iv]]
        hpi = np.flatnonzero(hpp >= 0)
        if len(hpi) < 2:
            continue
        R[ci, hpp[hpi]] = hpi.astype(np.float32)
        seq_lens.append(len(pos)); n_used += 1
    del X, Xh
    present = np.isfinite(R)
    P = present.astype(np.float32)
    PC = np.rint(P.T @ P).astype(np.int64)
    RA = np.where(present, R, -np.inf).astype(np.float32)   # query side: absent -> nothing is "before" it
    C = np.zeros((G, G), dtype=np.int64)                  # C[a, b] = #cells with both, b before a
    for a0 in range(0, G, 48):
        a1 = min(G, a0 + 48)
        C[a0:a1] = (R[:, None, :] < RA[:, a0:a1, None]).sum(0)
    np.fill_diagonal(C, 0)
    pc_saved = np.load(p0 / "attention_pair_counts.npy").astype(np.int64)
    off = ~np.eye(G, dtype=bool)
    checks = {
        "n_cells_tokenised_with_ge2_genes": int(n_used),
        "mean_seq_len": float(np.mean(seq_lens)),
        "pair_counts_reproduced_exactly": bool(np.array_equal(PC, pc_saved)),
        "n_pair_count_mismatches": int((PC != pc_saved).sum()),
        "C_plus_CT_equals_pair_counts_offdiag": bool(np.array_equal((C + C.T)[off], pc_saved[off])),
    }
    # zero pattern: with a causal mask, E[P,T] > 0 exactly when C[P,T] > 0 (for n_pair > 0)
    E = load_attention_layer(run).astype(np.float64)
    m = off & (pc_saved > 0)
    checks["layer8_pairs_C0_but_E_pos"] = int(((C == 0) & (E > 0) & m).sum())
    checks["layer8_pairs_Cpos_but_E0"] = int(((C > 0) & (E == 0) & m).sum())
    checks["n_offdiag_pairs_with_n_pair_gt0"] = int(m.sum())
    checks["n_offdiag_pairs_with_n_pair_eq0"] = int((off & (pc_saved == 0)).sum())
    checks["frac_n_pair_gt0_pairs_with_C_eq0"] = float(((C == 0) & m).sum() / m.sum())
    np.save(OUTB / "order" / f"order_counts_{run}.npy", C.astype(np.int32))

    # how much of the forward edge is the model-free order share
    F = np.where(pc_saved > 0, C / np.maximum(pc_saved, 1), 0.0)
    Ec = np.where(C > 0, E * pc_saved / np.maximum(C, 1), 0.0)
    mc = m & (C > 0) & (E > 0)
    lE, lF, lEc = np.log(E[mc]), np.log(F[mc]), np.log(Ec[mc])
    r2 = 1 - np.var(lE - np.polyval(np.polyfit(lF, lE, 1), lF)) / np.var(lE)
    dec = {
        "spearman_E_vs_F_all_pairs_with_n_pair_gt0": float(spearmanr(E[m], F[m]).correlation),
        "spearman_E_vs_Ec_all_pairs": float(spearmanr(E[m], Ec[m]).correlation),
        "spearman_F_vs_Ec_all_pairs": float(spearmanr(F[m], Ec[m]).correlation),
        "r2_logE_on_logF_pairs_with_C_gt0": float(r2),
        "var_share_logF": float(np.var(lF) / np.var(lE)),
        "var_share_logEc": float(np.var(lEc) / np.var(lE)),
        "var_share_2cov": float(2 * np.cov(lF, lEc)[0, 1] / np.var(lE)),
    }
    # within-row: F[P, .] versus a target-only score (weighted column mean of F)
    colF = (C.sum(0) / np.maximum(pc_saved.sum(0), 1)).astype(np.float64)   # share of cells where T is above its partner
    rho_EF, rho_Fcol = [], []
    for p in range(G):
        mm = m[p]
        if mm.sum() > 10:
            rho_EF.append(spearmanr(E[p, mm], F[p, mm]).correlation)
            rho_Fcol.append(spearmanr(F[p, mm], colF[mm]).correlation)
    dec["median_within_row_spearman_E_vs_F"] = float(np.nanmedian(rho_EF))
    dec["median_within_row_spearman_F_vs_target_only_order_score"] = float(np.nanmedian(rho_Fcol))
    out = {"run": run, "input_scale": input_scale(run), "facts": facts, "checks": checks,
           "decomposition_layer8": dec, "seconds_read": t_read, "seconds_total": time.time() - t0}
    save_json(out, OUTB / "order" / f"order_check_{run}.json")
    print(json.dumps(out, indent=1))


# ----------------------------------------------------------------------------- scores
def build_scores(run: str):
    E = load_attention_layer(run).astype(np.float64)
    np.fill_diagonal(E, 0.0)
    pc = np.load(phase_dir("phase0", run) / "attention_pair_counts.npy").astype(np.float64)
    C = np.load(OUTB / "order" / f"order_counts_{run}.npy").astype(np.float64)
    F = np.where(pc > 0, C / np.maximum(pc, 1.0), 0.0)
    Ec = np.where(C > 0, E * pc / np.maximum(C, 1.0), 0.0)
    for M in (F, Ec):
        np.fill_diagonal(M, 0.0)
    S = {"forward": E, "transpose": E.T.copy(), "sym_mean": 0.5 * (E + E.T),
         "sym_max": np.maximum(E, E.T), "rank_conditioned": Ec, "order_share_F": F}
    return S, pc, C


def ref_pair_scores(run: str) -> dict:
    """Reference pair score outside the test family: |Spearman| co-expression in the same
    Phase-0 control cells (deployed spearman_edges.npy; no model)."""
    sp = np.abs(np.load(phase_dir("phase0", run) / "spearman_edges.npy").astype(np.float64))
    np.fill_diagonal(sp, 0.0)
    return {"coexpr_abs_spearman": sp}


def gene_baselines(run: str) -> dict:
    gf = load_gene_features(run)
    b = {"gene_variance": gf["variance"].to_numpy(np.float64)}
    if run in LOG_RUNS:
        sl = pd.read_csv(OUTB / "order" / f"gene_variance_single_log_{run}.csv")
        b["gene_variance_single_log"] = sl["variance_single_log"].to_numpy(np.float64)
    return b


def wilcox_p(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if np.allclose(a[m], b[m]):
        return float("nan")
    return float(stats.wilcoxon(a[m], b[m], zero_method="zsplit").pvalue)


# ----------------------------------------------------------------------------- endpoint A
def cmd_knockdown() -> None:
    t0 = time.time()
    res = {"tie_self_test": tie_self_test(), "runs": {}}
    for ri, run in enumerate(RUNS):
        S, pc, C = build_scores(run)
        S = {**S, **ref_pair_scores(run)}
        base = gene_baselines(run)
        G = S["forward"].shape[0]
        d = np.load(V2 / "de_labels" / f"{run}.npz")
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
            mm = valid & (pc[h] > 0)
            rho.append(spearmanr(S["forward"][h, mm], S["order_share_F"][h, mm]).correlation)
        A = {k: np.array(v) for k, v in A.items()}
        pp = pd.DataFrame({"pert_symbol": syms.astype(str), "pert_hvg_idx": hs,
                           "n_pos": [int(Y[i][np.arange(G) != h].sum()) for i, h in enumerate(hs)]})
        for k in names:
            pp[f"auc_{k}"] = A[k]
        pp.to_csv(OUTB / "knockdown" / f"per_perturbation_{run}.csv", index=False)
        v2pp = pd.read_csv(V2 / "knockdown" / f"per_perturbation_{run}.csv")
        assert list(v2pp["pert_symbol"].astype(str)) == list(pp["pert_symbol"])
        check = {"max_abs_diff_forward_vs_v2_attention": float(np.max(np.abs(A["forward"] - v2pp["auc_attention"]))),
                 "max_abs_diff_variance_vs_v2": float(np.max(np.abs(A["gene_variance"] - v2pp["auc_variance"])))}
        n = len(hs)
        rng = np.random.default_rng(SEED + 100 + ri)               # same draws as v2_03
        B = rng.integers(0, n, size=(N_BOOT, n))
        bm = {k: A[k][B].mean(1) for k in names}
        out = {"n_perturbations": int(n), "n_positive_pairs": int(pp["n_pos"].sum()),
               "input_scale": input_scale(run), "checks_vs_v2": check,
               "median_within_row_spearman_forward_vs_F": float(np.nanmedian(rho)), "scores": {}}
        for k in names:
            o = {"mean_auroc": float(A[k].mean()), "ci95": percentile_ci(bm[k]),
                 "gap_variance_minus_score": float(A["gene_variance"].mean() - A[k].mean()),
                 "gap_variance_minus_score_ci95": percentile_ci(bm["gene_variance"] - bm[k]),
                 "n_perts_score_gt_variance": int((A[k] > A["gene_variance"]).sum())}
            if k != "forward":
                o["delta_vs_forward"] = float(A[k].mean() - A["forward"].mean())
                o["delta_vs_forward_ci95"] = percentile_ci(bm[k] - bm["forward"])
                o["wilcoxon_p_vs_forward"] = wilcox_p(A[k], A["forward"])
            if k != "gene_variance":
                o["wilcoxon_p_vs_variance"] = wilcox_p(A[k], A["gene_variance"])
            out["scores"][k] = o
        res["runs"][run] = out
        print(run, {k: (round(v["mean_auroc"], 4), [round(x, 4) for x in v["ci95"]]) for k, v in out["scores"].items()},
              check, flush=True)
    res["_method"] = {"endpoint": "A, knockdown response; labels = outputs/v2_eval/de_labels/<run>.npz",
                      "summary": "mean over perturbations of per-perturbation AUROC (tie-safe average ranks)",
                      "ci": "percentile bootstrap over perturbations, 2,000 reps, seed SEED+100+run_index (same draws as v2_03)",
                      "test": "paired Wilcoxon signed-rank, zero_method='zsplit', two-sided, no multiplicity correction",
                      "seconds": time.time() - t0}
    save_json(res, OUTB / "knockdown" / "knockdown_variants_summary.json")


# ----------------------------------------------------------------------------- endpoint B
def trrust_setup(run: str):
    gf = load_gene_features(run)
    G = len(gf)
    sym = [s.upper() for s in gf["symbol"]]
    trrust = load_trrust()
    Etr = trrust_matrix(sym, trrust)
    tf_rows = evaluated_tf_rows(sym, trrust, Etr)
    k_of = np.concatenate([np.full(G - 1, k) for k in range(len(tf_rows))])
    cand = np.concatenate([np.delete(np.arange(G), r) for r in tf_rows])
    src = tf_rows[k_of]
    y = Etr[src, cand].astype(bool)
    return gf, G, sym, Etr, tf_rows, k_of, cand, src, y


def null_auroc(score: np.ndarray, pos: np.ndarray) -> np.ndarray:
    r = rankdata(score, method="average")
    P = pos.shape[1]
    N = len(score) - P
    return (r[pos].sum(1) - P * (P + 1) / 2.0) / (P * N)


def cmd_trrust(run: str) -> None:
    from v2_06_curveball import curveball                     # the v2 sampler, unchanged
    t0 = time.time()
    ri = list(RUNS).index(run)
    S, pc, C = build_scores(run)
    S = {**S, **ref_pair_scores(run)}
    base = gene_baselines(run)
    gf, G, sym, Etr, tf_rows, k_of, cand, src, y = trrust_setup(run)
    nT = len(tf_rows)
    scores = {k: M[src, cand] for k, M in S.items()}
    for k, v in base.items():
        scores[k] = v[cand]
    pooled = {k: auroc_rank(y, s) for k, s in scores.items()}
    v2s = json.load(open(V2 / "trrust" / "trrust_summary.json"))[run]
    check = {"abs_diff_forward_vs_v2": abs(pooled["forward"] - v2s["pooled_auroc"]["attention"]),
             "abs_diff_variance_vs_v2": abs(pooled["gene_variance"] - v2s["pooled_auroc"]["variance"])}
    # TF bootstrap, same draws as v2_05
    fast = {k: WeightedAUROC(y, s) for k, s in scores.items()}
    rng = np.random.default_rng(SEED + 500 + ri)
    boot = {k: [] for k in scores}
    for _ in range(N_BOOT):
        draw = rng.integers(0, nT, nT)
        w = np.bincount(draw, minlength=nT)[k_of]
        for k in scores:
            boot[k].append(fast[k](w))
    boot = {k: np.array(v) for k, v in boot.items()}
    check["forward_ci_equals_v2"] = [round(x, 12) for x in percentile_ci(boot["forward"])] == \
        [round(x, 12) for x in v2s["pooled_ci95"]["attention"]]
    # per-TF AUROC (secondary view)
    per_tf = []
    for k in range(nT):
        mk = k_of == k
        row = {"tf": sym[tf_rows[k]], "n_targets": int(y[mk].sum())}
        for name, s in scores.items():
            row[name] = auroc_rank(y[mk], s[mk])
        per_tf.append(row)
    per_tf = pd.DataFrame(per_tf)
    per_tf.to_csv(OUTB / "trrust" / f"per_tf_{run}.csv", index=False)
    # Curveball null: identical networks to v2_06 (same sampler, same seed, same order of rng calls)
    M = Etr[tf_rows].astype(np.int8)
    obs_rows = [set(np.flatnonzero(M[k]).tolist()) for k in range(nT)]
    forb = tf_rows.copy()
    n_edges = int(M.sum())
    rngc = np.random.default_rng(SEED + 600 + ri)
    pos = np.empty((N_DRAWS, n_edges), dtype=np.int64)
    for d in range(N_DRAWS):
        rows = curveball(obs_rows, forb, N_TRADES, rngc)
        pp = []
        for k, cols in enumerate(rows):
            c = np.array(sorted(cols), dtype=np.int64)
            pp.append(k * (G - 1) + c - (c > tf_rows[k]))
        pos[d] = np.concatenate(pp)
    assert np.array_equal(np.sort(np.flatnonzero(y)),
                          np.sort(np.concatenate([k * (G - 1) + np.array(sorted(r)) - (np.array(sorted(r)) > tf_rows[k])
                                                  for k, r in enumerate(obs_rows)])))
    np.save(OUTB / "trrust" / f"null_positions_{run}.npy", pos.astype(np.int32))
    saved = np.load(V2 / "curveball" / f"null_curveball_{run}.npy")[:, 0]
    nulls = {k: null_auroc(s, pos) for k, s in scores.items()}
    check["max_abs_diff_forward_null_vs_v2"] = float(np.max(np.abs(nulls["forward"] - saved)))
    out = {"run": run, "input_scale": input_scale(run), "n_tfs": int(nT), "n_positive_pairs": int(y.sum()),
           "n_negative_pairs": int((~y).sum()), "checks_vs_v2": check, "scores": {}}
    for k in scores:
        o = {"pooled_auroc": pooled[k], "ci95": percentile_ci(boot[k]),
             "per_tf_mean_auroc": float(np.nanmean(per_tf[k])),
             "gap_variance_minus_score": pooled["gene_variance"] - pooled[k],
             "gap_variance_minus_score_ci95": percentile_ci(boot["gene_variance"] - boot[k]),
             "n_tfs_score_gt_variance": int((per_tf[k] > per_tf["gene_variance"]).sum())}
        if "gene_variance_single_log" in scores:
            o["gap_single_log_variance_minus_score"] = pooled["gene_variance_single_log"] - pooled[k]
            o["gap_single_log_variance_minus_score_ci95"] = percentile_ci(boot["gene_variance_single_log"] - boot[k])
        if k != "forward":
            o["delta_vs_forward"] = pooled[k] - pooled["forward"]
            o["delta_vs_forward_ci95"] = percentile_ci(boot[k] - boot["forward"])
        if k != "coexpr_abs_spearman":
            o["delta_vs_coexpr"] = pooled[k] - pooled["coexpr_abs_spearman"]
            o["delta_vs_coexpr_ci95"] = percentile_ci(boot[k] - boot["coexpr_abs_spearman"])
        nv = nulls[k]
        sd = float(nv.std(ddof=1))
        o["null"] = {"mean": float(nv.mean()), "sd": sd,
                     "z": float((pooled[k] - nv.mean()) / sd) if sd > 1e-12 else None,
                     "p_one_sided": float((1 + (nv >= pooled[k] - 1e-12).sum()) / (1 + len(nv))) if sd > 1e-12 else None,
                     "note": None if sd > 1e-12 else "target-only score: unchanged by a null that keeps column sums"}
        out["scores"][k] = o
    out["seconds"] = time.time() - t0
    save_json(out, OUTB / "trrust" / f"trrust_variants_{run}.json")
    print(run, check, {k: (round(v["pooled_auroc"], 4), [round(x, 4) for x in v["ci95"]],
                           None if v["null"]["z"] is None else round(v["null"]["z"], 2)) for k, v in out["scores"].items()},
          f"{out['seconds']:.0f}s", flush=True)


def cmd_trrust_combine() -> None:
    allr = {r: json.load(open(OUTB / "trrust" / f"trrust_variants_{r}.json")) for r in RUNS}
    runs = list(RUNS)
    fam = [(r, k) for r in runs for k in ALL_SCORES]
    p_all = np.array([allr[r]["scores"][k]["null"]["p_one_sided"] for r, k in fam])
    q_all = bh(p_all)
    for (r, k), q in zip(fam, q_all):
        allr[r]["scores"][k]["null"]["q_bh_all_24_tests"] = float(q)
    for k in ALL_SCORES + ["coexpr_abs_spearman"]:
        q = bh(np.array([allr[r]["scores"][k]["null"]["p_one_sided"] for r in runs]))
        for r, qq in zip(runs, q):
            allr[r]["scores"][k]["null"]["q_bh_across_4_runs"] = float(qq)
    allr["_method"] = {"pooled": "one AUROC over all (evaluated TF, candidate) pairs; tie-safe average ranks",
                       "ci": "percentile bootstrap over TFs, 2,000 reps, seed SEED+500+run_index (same draws as v2_05)",
                       "null": "Curveball, 1,000 draws x 5,000 trades, seed SEED+600+run_index, sampler from v2_06 "
                               "(same networks as v2); p = (1 + #null >= obs)/(1 + 1000), one-sided",
                       "bh": "q_bh_across_4_runs: per score over the 4 runs; q_bh_all_24_tests: 6 scores x 4 runs"}
    save_json(allr, OUTB / "trrust" / "trrust_variants_summary.json")
    for r in runs:
        print(r, {k: (round(v["pooled_auroc"], 3), round(v["null"]["z"], 2), v["null"].get("q_bh_all_24_tests"))
                  for k, v in allr[r]["scores"].items() if v["null"]["z"] is not None})


def cmd_gene3(run: str) -> None:
    """TRRUST: the v2 3-feature gene model (candidate mean, variance, dropout; logistic
    regression; GroupKFold(5) by TF; pooled out-of-fold predictions) as a baseline for
    every score, with the same TF-bootstrap draws (gap CIs; fitted models held fixed)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    ri = list(RUNS).index(run)
    S, pc, C = build_scores(run)
    S = {**S, **ref_pair_scores(run)}
    gf, G, sym, Etr, tf_rows, k_of, cand, src, y = trrust_setup(run)
    nT = len(tf_rows)
    gm, gv, gd = (gf[c].to_numpy(np.float64) for c in ["mean_expr", "variance", "dropout_rate"])
    X = np.column_stack([gm[cand], gv[cand], gd[cand]]).astype(np.float32)       # as v2_05
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups=k_of):
        m = Pipeline([("sc", StandardScaler()),
                      ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))]).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    scores = {k: M[src, cand] for k, M in S.items()}
    scores["gene_model_3feat"] = oof
    pooled = {k: auroc_rank(y, v) for k, v in scores.items()}
    v2s = json.load(open(V2 / "trrust" / "trrust_summary.json"))[run]
    fast = {k: WeightedAUROC(y, v) for k, v in scores.items()}
    rng = np.random.default_rng(SEED + 500 + ri)                  # same draws as v2_05 / trrust step
    boot = {k: [] for k in scores}
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, nT, nT), minlength=nT)[k_of]
        for k in scores:
            boot[k].append(fast[k](w))
    boot = {k: np.array(v) for k, v in boot.items()}
    out = {"run": run, "gene_model_auroc": pooled["gene_model_3feat"],
           "gene_model_ci95": percentile_ci(boot["gene_model_3feat"]),
           "check_abs_diff_vs_v2_gene3": abs(pooled["gene_model_3feat"] - v2s["pooled_auroc"]["gene3_logreg_oof"]),
           "gaps": {k: {"gap_gene_model_minus_score": pooled["gene_model_3feat"] - pooled[k],
                        "ci95": percentile_ci(boot["gene_model_3feat"] - boot[k])}
                    for k in scores if k != "gene_model_3feat"}}
    save_json(out, OUTB / "trrust" / f"gene_model_gaps_{run}.json")
    print(run, round(out["gene_model_auroc"], 4), "check", out["check_abs_diff_vs_v2_gene3"],
          {k: (round(v["gap_gene_model_minus_score"], 3), [round(x, 3) for x in v["ci95"]]) for k, v in out["gaps"].items()})


# ----------------------------------------------------------------------------- residualisation
def _folds(n_pairs: int):
    from sklearn.model_selection import KFold
    return list(KFold(n_splits=5, shuffle=True, random_state=42).split(np.arange(n_pairs)))


def _ols_resid(target: np.ndarray, X: np.ndarray, folds) -> tuple[np.ndarray, float]:
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


def _resid_inputs(run):
    S, pc, C = build_scores(run)
    S = {**S, **ref_pair_scores(run)}
    gf, G, sym, Etr, tf_rows, k_of, cand, src_e, y = trrust_setup(run)
    ii, jj = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    mask = ii != jj
    src, tgt = ii[mask], jj[mask]
    del ii, jj
    gm, gv, gd = (gf[c].to_numpy(np.float64) for c in ["mean_expr", "variance", "dropout_rate"])
    gene6 = np.column_stack([gm[src], gv[src], gd[src], gm[tgt], gv[tgt], gd[tgt]])
    extra = np.column_stack([S["order_share_F"][src, tgt], (pc[src, tgt] == 0).astype(np.float64)])
    ols5 = np.column_stack([gm[src], gv[src], gm[tgt], gv[tgt], gd[tgt]])   # deployed feature set
    flat = src_e * (G - 1) + cand - (cand > src_e)                           # pooled pairs in flat order
    return S, src, tgt, gene6, extra, ols5, flat, y, k_of, len(tf_rows)


def _score_resid(name, R, flat, y, k_of, draws, raw_boot, raw_auc, pos):
    s = R[flat]
    auc = auroc_rank(y, s)
    f = WeightedAUROC(y, s)
    boot = np.array([f(c[k_of]) for c in draws])
    nv = null_auroc(s, pos)
    return {"residualised_auroc": auc, "ci95": percentile_ci(boot),
            "drop_from_raw": raw_auc - auc, "drop_ci95": percentile_ci(raw_boot - boot),
            "null_z": float((auc - nv.mean()) / nv.std(ddof=1)),
            "null_p_one_sided": float((1 + (nv >= auc - 1e-12).sum()) / (1 + len(nv)))}


def cmd_resid(run: str, kind: str, max_seconds: float = 420.0) -> None:
    t0 = time.time()
    ri = list(RUNS).index(run)
    S, src, tgt, gene6, extra, ols5, flat, y, k_of, nT = _resid_inputs(run)
    folds = _folds(len(src))
    rng = np.random.default_rng(SEED + 800 + ri)                 # same draws as v2_07
    draws = [np.bincount(rng.integers(0, nT, nT), minlength=nT) for _ in range(N_BOOT)]
    pos = np.load(OUTB / "trrust" / f"null_positions_{run}.npy").astype(np.int64)
    if kind == "ols":
        res = {"run": run, "input_scale": input_scale(run), "scores": {}}
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
                R = np.empty(len(src))
                Rf, r2 = _ols_resid(target, X, folds)
                R[:] = Rf
                o[nm] = _score_resid(nm, R, flat, y, k_of, draws, raw_boot, raw_auc, pos)
                o[nm]["r2_train_mean"] = r2
            res["scores"][k] = o
            print(run, k, {nm: round(v["residualised_auroc"], 4) for nm, v in o.items() if isinstance(v, dict) and "residualised_auroc" in v},
                  f"{time.time() - t0:.0f}s", flush=True)
        v2r = json.load(open(V2 / "residualised" / f"resid_{run}.json"))
        res["check_forward_ols5_vs_v2_ols5_f64"] = abs(res["scores"]["forward"]["ols5_deployed_features_f64"]["residualised_auroc"]
                                                        - v2r["ols5_f64"]["residualised_auroc"])
        res["seconds"] = time.time() - t0
        save_json(res, OUTB / "resid" / f"resid_ols_{run}.json")
        print("check vs v2 ols5_f64:", res["check_forward_ols5_vs_v2_ols5_f64"])
    elif kind == "hgb":
        for k in ALL_SCORES:
            p = OUTB / "resid" / f"resid_hgb_{run}_{k}.json"
            if p.exists():
                continue
            if time.time() - t0 > max_seconds:
                print("time budget reached; re-run to continue"); return
            t1 = time.time()
            target = S[k][src, tgt]
            raw = target[flat]
            raw_auc = auroc_rank(y, raw)
            fr = WeightedAUROC(y, raw)
            raw_boot = np.array([fr(c[k_of]) for c in draws])
            X = gene6 if k == "order_share_F" else np.column_stack([gene6, extra])
            nm = "hgb_gene6" if k == "order_share_F" else "hgb_gene6_F"
            R, r2 = _hgb_resid(target, X, folds)
            o = _score_resid(nm, R, flat, y, k_of, draws, raw_boot, raw_auc, pos)
            o.update({"features": nm, "r2_train_mean": r2, "raw_auroc": raw_auc, "seconds": time.time() - t1})
            save_json(o, p)
            print(run, k, nm, round(o["residualised_auroc"], 4), o["ci95"], f"{o['seconds']:.0f}s", flush=True)
    else:
        raise SystemExit("kind must be ols or hgb")


def cmd_resid_combine() -> None:
    allr = {}
    for r in RUNS:
        a = json.load(open(OUTB / "resid" / f"resid_ols_{r}.json"))
        for k in ALL_SCORES:  # HGB check is run for the 6 family scores only
            p = OUTB / "resid" / f"resid_hgb_{r}_{k}.json"
            if p.exists():
                h = json.load(open(p))
                a["scores"][k][h["features"]] = h
        allr[r] = a
    allr["_method"] = {
        "pairs": "all ordered off-diagonal gene pairs (1,500 x 1,499); 5-fold KFold(shuffle, random_state=42) cross-fitting",
        "ols_gene6": "OLS (float64, lstsq, standardised) on mean, variance, dropout of both genes",
        "ols_gene6_F": "ols_gene6 + order share F[P,T] + indicator(n_pair = 0)",
        "ols5_deployed_features_f64": "deployed feature set (source mean, var; target mean, var, dropout), float64; forward only; must equal v2 ols5_f64",
        "hgb_gene6_F": "HistGradientBoostingRegressor(max_iter=200, lr=0.1, max_leaf_nodes=31) on gene6 + F + indicator",
        "ci": "percentile bootstrap over TFs, 2,000 reps, seed SEED+800+run_index (same draws as v2_07); fitted models held fixed",
        "null": "Curveball networks of the trrust step (same as v2); z and one-sided p of the residualised pooled AUROC",
        "gene features": "deployed gene_features.csv (in 3 runs computed after a second log; see V2_EVAL V3)"}
    # main residual per score (pre-stated): OLS on gene6 + F for attention edges, OLS on gene6 for F;
    # BH over the 24 null tests (6 scores x 4 runs), and separately for the HGB check
    for nm_edge, nm_F, tag in [("ols_gene6_F", "ols_gene6", "main_ols"), ("hgb_gene6_F", "hgb_gene6", "hgb")]:
        fam = [(r, k, nm_F if k == "order_share_F" else nm_edge) for r in RUNS for k in ALL_SCORES
               if (nm_F if k == "order_share_F" else nm_edge) in allr[r]["scores"][k]]
        q = bh(np.array([allr[r]["scores"][k][nm]["null_p_one_sided"] for r, k, nm in fam]))
        for (r, k, nm), qq in zip(fam, q):
            allr[r]["scores"][k][nm][f"null_q_bh_{tag}_family"] = float(qq)
            allr[r]["scores"][k][nm]["n_tests_in_bh_family"] = len(fam)
    save_json(allr, OUTB / "resid" / "resid_summary.json")
    for r in RUNS:
        print(r, {k: {nm: round(v["residualised_auroc"], 3) for nm, v in o.items() if isinstance(v, dict) and "residualised_auroc" in v}
                  for k, o in allr[r]["scores"].items()})


# ----------------------------------------------------------------------------- table / figure
LABEL = {"forward": "forward attn[P,T] (deployed)", "transpose": "transpose attn[T,P]",
         "sym_mean": "symmetric mean", "sym_max": "symmetric max",
         "rank_conditioned": "rank-conditioned attn[P,T]", "order_share_F": "order share F (no model)",
         "gene_variance": "target variance (deployed)", "gene_variance_single_log": "target variance (single log)",
         "coexpr_abs_spearman": "|Spearman| co-expression (no model)"}
KIND = {k: "attention edge" for k in EDGE_SCORES}
KIND.update({"order_share_F": "model-free order", "coexpr_abs_spearman": "model-free co-expression (reference)",
             "gene_variance": "gene-level baseline",
             "gene_variance_single_log": "gene-level baseline", "gene_model_3feat": "gene-level baseline"})
LABEL["gene_model_3feat"] = "3-feature gene model (logistic, TF-grouped CV)"


def cmd_table() -> None:
    kd = json.load(open(OUTB / "knockdown" / "knockdown_variants_summary.json"))["runs"]
    tr = json.load(open(OUTB / "trrust" / "trrust_variants_summary.json"))
    g3 = {r: json.load(open(OUTB / "trrust" / f"gene_model_gaps_{r}.json")) for r in RUNS}
    rs = json.load(open(OUTB / "resid" / "resid_summary.json"))
    tidy, fig = [], []
    for r in RUNS:
        for k, o in kd[r]["scores"].items():
            row = {"run": r, "input_scale": input_scale(r), "endpoint": "knockdown", "score": k, "score_label": LABEL[k],
                   "score_kind": KIND[k], "n_units": kd[r]["n_perturbations"], "unit": "perturbation",
                   "auroc": o["mean_auroc"], "ci_low": o["ci95"][0], "ci_high": o["ci95"][1],
                   "gap_variance_minus_score": o["gap_variance_minus_score"],
                   "gap_ci_low": o["gap_variance_minus_score_ci95"][0], "gap_ci_high": o["gap_variance_minus_score_ci95"][1],
                   "delta_vs_forward": o.get("delta_vs_forward"),
                   "delta_ci_low": (o.get("delta_vs_forward_ci95") or [None, None])[0],
                   "delta_ci_high": (o.get("delta_vs_forward_ci95") or [None, None])[1],
                   "wilcoxon_p_vs_forward": o.get("wilcoxon_p_vs_forward"),
                   "wilcoxon_p_vs_variance": o.get("wilcoxon_p_vs_variance"),
                   "n_units_score_gt_variance": o["n_perts_score_gt_variance"]}
            tidy.append(row)
            fig.append({"run": r, "input_scale": input_scale(r), "endpoint": "knockdown", "score": k,
                        "score_label": LABEL[k], "score_kind": KIND[k], "measure": "mean per-perturbation AUROC",
                        "value": o["mean_auroc"], "ci_low": o["ci95"][0], "ci_high": o["ci95"][1],
                        "ci_method": "percentile bootstrap over perturbations, 2,000 reps", "n_units": kd[r]["n_perturbations"]})
        for k, o in tr[r]["scores"].items():
            nl = o["null"]
            row = {"run": r, "input_scale": input_scale(r), "endpoint": "trrust", "score": k, "score_label": LABEL[k],
                   "score_kind": KIND[k], "n_units": tr[r]["n_tfs"], "unit": "TF",
                   "auroc": o["pooled_auroc"], "ci_low": o["ci95"][0], "ci_high": o["ci95"][1],
                   "gap_variance_minus_score": o["gap_variance_minus_score"],
                   "gap_ci_low": o["gap_variance_minus_score_ci95"][0], "gap_ci_high": o["gap_variance_minus_score_ci95"][1],
                   "gap_single_log_variance_minus_score": o.get("gap_single_log_variance_minus_score"),
                   "gap_single_log_ci_low": (o.get("gap_single_log_variance_minus_score_ci95") or [None, None])[0],
                   "gap_single_log_ci_high": (o.get("gap_single_log_variance_minus_score_ci95") or [None, None])[1],
                   "delta_vs_forward": o.get("delta_vs_forward"),
                   "delta_ci_low": (o.get("delta_vs_forward_ci95") or [None, None])[0],
                   "delta_ci_high": (o.get("delta_vs_forward_ci95") or [None, None])[1],
                   "n_units_score_gt_variance": o["n_tfs_score_gt_variance"],
                   "delta_vs_coexpr": o.get("delta_vs_coexpr"),
                   "delta_vs_coexpr_ci_low": (o.get("delta_vs_coexpr_ci95") or [None, None])[0],
                   "delta_vs_coexpr_ci_high": (o.get("delta_vs_coexpr_ci95") or [None, None])[1],
                   "null_mean": nl["mean"], "null_sd": nl["sd"], "null_z": nl["z"], "null_p": nl["p_one_sided"],
                   "null_q_bh_4runs": nl.get("q_bh_across_4_runs"), "null_q_bh_24": nl.get("q_bh_all_24_tests")}
            gg = g3[r]["gaps"].get(k)
            if gg is not None:
                row["gap_gene_model_minus_score"] = gg["gap_gene_model_minus_score"]
                row["gap_gene_model_ci_low"], row["gap_gene_model_ci_high"] = gg["ci95"]
            if k in rs[r]["scores"]:
                for nm, v in rs[r]["scores"][k].items():
                    if isinstance(v, dict) and "residualised_auroc" in v and nm != "ols5_deployed_features_f64":
                        row[f"resid_{nm}_auroc"] = v["residualised_auroc"]
                        row[f"resid_{nm}_ci_low"] = v["ci95"][0]
                        row[f"resid_{nm}_ci_high"] = v["ci95"][1]
                        row[f"resid_{nm}_null_z"] = v["null_z"]
                        row[f"resid_{nm}_null_p"] = v["null_p_one_sided"]
                        qk = [kk for kk in v if kk.startswith("null_q_bh_")]
                        if qk:
                            row[f"resid_{nm}_null_q_bh"] = v[qk[0]]
                        fig.append({"run": r, "input_scale": input_scale(r), "endpoint": "trrust", "score": k,
                                    "score_label": LABEL[k], "score_kind": KIND[k],
                                    "measure": f"pooled AUROC after residualising ({nm})",
                                    "value": v["residualised_auroc"], "ci_low": v["ci95"][0], "ci_high": v["ci95"][1],
                                    "ci_method": "percentile bootstrap over TFs, 2,000 reps, fitted models held fixed",
                                    "n_units": tr[r]["n_tfs"]})
            tidy.append(row)
            fig.append({"run": r, "input_scale": input_scale(r), "endpoint": "trrust", "score": k,
                        "score_label": LABEL[k], "score_kind": KIND[k], "measure": "pooled AUROC",
                        "value": o["pooled_auroc"], "ci_low": o["ci95"][0], "ci_high": o["ci95"][1],
                        "ci_method": "percentile bootstrap over TFs, 2,000 reps", "n_units": tr[r]["n_tfs"]})
            if nl["z"] is not None:
                fig.append({"run": r, "input_scale": input_scale(r), "endpoint": "trrust", "score": k,
                            "score_label": LABEL[k], "score_kind": KIND[k], "measure": "Curveball null z",
                            "value": nl["z"], "ci_low": None, "ci_high": None,
                            "ci_method": "none (z = (obs - null mean)/null SD, 1,000 draws)", "n_units": tr[r]["n_tfs"]})
                fig.append({"run": r, "input_scale": input_scale(r), "endpoint": "trrust", "score": k,
                            "score_label": LABEL[k], "score_kind": KIND[k], "measure": "Curveball null mean",
                            "value": nl["mean"], "ci_low": None, "ci_high": None,
                            "ci_method": "none", "n_units": tr[r]["n_tfs"]})
        gm3 = g3[r]
        tidy.append({"run": r, "input_scale": input_scale(r), "endpoint": "trrust", "score": "gene_model_3feat",
                     "score_label": LABEL["gene_model_3feat"], "score_kind": KIND["gene_model_3feat"],
                     "n_units": tr[r]["n_tfs"], "unit": "TF", "auroc": gm3["gene_model_auroc"],
                     "ci_low": gm3["gene_model_ci95"][0], "ci_high": gm3["gene_model_ci95"][1]})
        fig.append({"run": r, "input_scale": input_scale(r), "endpoint": "trrust", "score": "gene_model_3feat",
                    "score_label": LABEL["gene_model_3feat"], "score_kind": KIND["gene_model_3feat"],
                    "measure": "pooled AUROC", "value": gm3["gene_model_auroc"], "ci_low": gm3["gene_model_ci95"][0],
                    "ci_high": gm3["gene_model_ci95"][1], "ci_method": "percentile bootstrap over TFs, 2,000 reps, fitted models held fixed",
                    "n_units": tr[r]["n_tfs"]})
    pd.DataFrame(tidy).to_csv(OUTB / "table_attention_variants.csv", index=False)
    pd.DataFrame(fig).to_csv(OUTB / "fig_attention_variants.csv", index=False)
    print("rows:", len(tidy), len(fig))


# ----------------------------------------------------------------------------- run config
def cmd_config(max_seconds: float = 480.0) -> None:
    import platform
    t0 = time.time()
    cache_p = OUTB / "_sha256_cache.json"
    cache = json.load(open(cache_p)) if cache_p.exists() else {}

    def h(p: Path):
        st = p.stat()
        key = f"{p}|{st.st_size}|{int(st.st_mtime)}"
        if key not in cache:
            if time.time() - t0 > max_seconds:
                return None
            cache[key] = sha256(p)
            json.dump(cache, open(cache_p, "w"))
        return {"path": str(p), "bytes": st.st_size, "sha256": cache[key]}

    sys.path.insert(0, str(PROJ / "setup"))
    from dataset_loader import K562_H5, RPE1_H5, ADAMSON_H5   # noqa: E402
    from maxtoki_adapter import DEFAULT_GENE_MEDIAN_PKL        # noqa: E402
    raw = [K562_H5, RPE1_H5, ADAMSON_H5, TRRUST_TSV, SYM2ENS_PKL, DEFAULT_GENE_MEDIAN_PKL,
           PROJ / "setup/token_dictionary.json", PROJ / "setup/maxtoki_adapter.py", PROJ / "setup/dataset_loader.py"]
    inputs = [h(p) for p in raw]
    for r in RUNS:
        for fn in ["attention_edges_layer_mean.npy", "attention_pair_counts.npy", "gene_features.csv",
                   "hvg_gene_table.csv", "control_cells.csv"]:
            inputs.append(h(phase_dir("phase0", r) / fn))
        for rel in [f"de_labels/{r}.npz", f"knockdown/per_perturbation_{r}.csv", f"curveball/null_curveball_{r}.npy",
                    f"residualised/resid_{r}.json"]:
            inputs.append(h(V2 / rel))
    inputs.append(h(V2 / "trrust" / "trrust_summary.json"))
    scripts = [h(p) for p in [HERE / "v2b_attention.py", HERE / "v2_common.py", HERE / "v2_06_curveball.py"]]
    outs = [h(p) for p in sorted(OUTB.rglob("*")) if p.is_file() and not p.name.startswith("._")
            and p.name not in ("run_config.json", "_sha256_cache.json")]
    if any(x is None for x in inputs + scripts + outs):
        print("time budget reached; re-run to continue"); return
    import numpy, scipy, sklearn, h5py  # noqa: E401
    cfg = {"item": "D1b attention: direction and rank baselines",
           "device": "CPU only; no model forward pass; no model weights loaded (tokeniser only)",
           "python": platform.python_version(), "platform": platform.platform(),
           "packages": {"numpy": numpy.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
                        "pandas": pd.__version__, "h5py": h5py.__version__},
           "seeds": {"master (v2)": SEED, "knockdown_bootstrap": "SEED+100+run_index (= v2_03)",
                     "trrust_bootstrap": "SEED+500+run_index (= v2_05)", "curveball_null": "SEED+600+run_index (= v2_06)",
                     "residual_bootstrap": "SEED+800+run_index (= v2_07)", "residual_kfold": 42, "hgb": "SEED"},
           "run_index_order": list(RUNS),
           "parameters": {"layer": PRIMARY_LAYER, "max_len": MAX_LEN, "n_bootstrap": N_BOOT,
                          "curveball_draws": N_DRAWS, "curveball_trades": N_TRADES},
           "step_order": ["order <run> (x4)", "knockdown", "trrust <run> (x4); trrust combine",
                          "resid <run> ols (x4); resid <run> hgb (repeat until done); resid combine", "table", "config"],
           "inputs": inputs, "scripts": scripts, "outputs": outs}
    save_json(cfg, OUTB / "run_config.json")
    print("written", len(inputs), "inputs", len(scripts), "scripts", len(outs), "outputs", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    if a[0] == "order":
        cmd_order(a[1])
    elif a[0] == "knockdown":
        cmd_knockdown()
    elif a[0] == "gene3":
        cmd_gene3(a[1])
    elif a[0] == "trrust":
        cmd_trrust_combine() if a[1] == "combine" else cmd_trrust(a[1])
    elif a[0] == "resid":
        if a[1] == "combine":
            cmd_resid_combine()
        else:
            cmd_resid(a[1], a[2], float(a[3]) if len(a) > 3 else 420.0)
    elif a[0] == "table":
        cmd_table()
    elif a[0] == "config":
        cmd_config(float(a[1]) if len(a) > 1 else 480.0)
    else:
        raise SystemExit(__doc__)
