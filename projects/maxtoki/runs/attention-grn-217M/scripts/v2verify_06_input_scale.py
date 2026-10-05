"""Check what the deployed runs fed in as "expression": raw counts or log-normalised values.

Part 1 — value scale of the stored h5ad X for each run's Phase-0 control cells:
  fraction of non-zero values that are whole numbers; per-cell sum of expm1(X) (about
  10,000 when X = log1p(counts per 10k)); the smallest non-zero values.
Part 2 — model input. The deployed tokeniser (setup/maxtoki_adapter.py tokenize_cell)
  ranks genes by X / gene_median and keeps the top 2,046. On log values this order is not
  the order from counts. For each run's control cells we compare the deployed order (on X)
  with the order from expm1(X) (proportional to counts within a cell, so it gives the
  intended rank-value order): overlap of the kept gene sets, and Spearman correlation of
  the ranks of genes kept in both. RPE1 (raw counts) is the reference: expm1 is not
  applied there, so it is reported only in Part 1.
Part 3 — knockdown labels for 217M K562 under a single log: the stored values are
  already log1p(CP10k over the full transcriptome), so use them directly (no second
  scaling or log1p). Same Welch t / BH / |difference| >= 0.5 / >= 3 positives rule, same
  5,000 controls. Report n perturbations, positive pairs, attention and gene-variance
  mean AUROCs (variance recomputed on the single-log scale from the same 2,000 Phase-0
  control cells), and the gap with a perturbation bootstrap CI (2,000 reps, seed 51).
Reads only; CPU only. Output: outputs/v2_eval/verification/input_scale_check.json
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

PROJ = Path("<REPO_ROOT>/projects/maxtoki")
OUT = PROJ / "runs/attention-grn-217M/outputs"
VER = OUT / "v2_eval" / "verification"
sys.path.insert(0, str(PROJ / "setup"))
from dataset_loader import resolve as load_ds  # noqa: E402  (metadata + file path only)
from maxtoki_adapter import MaxTokiTokenizer  # noqa: E402  (vocabulary + medians only)

RUNS = {"217M_K562": ("", "k562"), "217M_RPE1": ("_rpe1", "rpe1"),
        "217M_Adamson": ("_adamson", "adamson"), "1B_K562": ("_k562_1b", "k562")}
SYM2ENS = Path("<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/crispri_validation/data/gene_name_id_dict_gc104M.pkl")


def read_rows(path, rows):
    rows = np.sort(np.asarray(rows))
    with h5py.File(path, "r") as f:
        X = f["X"]
        if isinstance(X, h5py.Dataset):
            out = np.vstack([X[rows[a:a + 500], :] for a in range(0, len(rows), 500)])
        else:
            import scipy.sparse as sp
            indptr = X["indptr"][:]
            n_cols = int(X.attrs["shape"][1])
            parts = []
            for r in rows:
                d = X["data"][indptr[r]:indptr[r + 1]]; ix = X["indices"][indptr[r]:indptr[r + 1]]
                v = np.zeros(n_cols, np.float32); v[ix] = d; parts.append(v)
            out = np.vstack(parts)
    return out.astype(np.float64)


def bh(p):
    n = len(p); o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1); return out


def main():
    tok = MaxTokiTokenizer()
    res = {"part1_value_scale": {}, "part2_token_order": {}}
    for run, (sfx, dsk) in RUNS.items():
        ds = load_ds(dsk)
        cc = pd.read_csv(OUT / f"phase0{sfx}/control_cells.csv")
        X = read_rows(ds.h5_path, cc["cell_idx_global"].to_numpy())
        nz = X[X > 0]
        res["part1_value_scale"][run] = {
            "file": str(ds.h5_path), "n_cells": int(len(X)), "n_genes_in_file": int(X.shape[1]),
            "frac_nonzero_values_whole_numbers": float(np.mean(np.abs(nz - np.round(nz)) < 1e-6)),
            "median_cell_sum_of_expm1_X": float(np.median(np.expm1(X).sum(1))) if nz.max() < 30 else None,
            "max_value": float(nz.max()), "min_nonzero_value": float(nz.min())}
        if run in ("217M_RPE1", "1B_K562"):
            continue  # RPE1 = raw counts reference; 1B uses the same K562 file as 217M K562
        var_idx, _, med = tok.make_var_mapping(list(ds.var_ensembl) if hasattr(ds, "var_ensembl") else
                                               [e.decode() if isinstance(e, bytes) else e for e in h5py.File(ds.h5_path, "r")["var"]["ensembl_id"][:]])
        ov, rho = [], []
        for x in X[:500]:
            e = x[var_idx]; m = e > 0
            gi = np.flatnonzero(m)
            o_dep = gi[np.argsort(-(e[m] / med[m]), kind="stable")][:2046]
            ec = np.expm1(e)
            o_cnt = gi[np.argsort(-(ec[m] / med[m]), kind="stable")][:2046]
            s_dep, s_cnt = set(o_dep.tolist()), set(o_cnt.tolist())
            ov.append(len(s_dep & s_cnt) / len(s_cnt))
            both = np.array(sorted(s_dep & s_cnt))
            r_dep = {g: i for i, g in enumerate(o_dep)}; r_cnt = {g: i for i, g in enumerate(o_cnt)}
            rho.append(spearmanr([r_dep[g] for g in both], [r_cnt[g] for g in both])[0])
        res["part2_token_order"][run] = {"n_cells_checked": 500,
                                         "mean_overlap_of_kept_genes": float(np.mean(ov)),
                                         "min_overlap_of_kept_genes": float(np.min(ov)),
                                         "mean_spearman_rank_order": float(np.mean(rho)),
                                         "min_spearman_rank_order": float(np.min(rho))}
        print(run, res["part1_value_scale"][run], res["part2_token_order"][run], flush=True)
    print(res["part1_value_scale"], flush=True)

    # Part 3: single-log knockdown labels, 217M K562
    gf = pd.read_csv(OUT / "phase0/gene_features.csv"); G = len(gf)
    cols = gf["var_idx"].to_numpy(np.int64)
    s2i = {s.upper(): i for i, s in enumerate(gf["symbol"])}
    sym2ens = pickle.load(open(SYM2ENS, "rb"))
    ds = load_ds("k562")
    coi, codes, cats, ctrl_codes = ds.cell_of_interest_mask, ds.perturbation_codes, ds.perturbation_categories, ds.control_category_codes
    ctrl = np.where(coi & np.isin(codes, list(ctrl_codes)))[0]
    ctrl = np.sort(np.random.default_rng(42).choice(ctrl, 5000, replace=False))
    coi_idx = np.where(coi)[0]; cc_ = codes[coi]
    perts = []
    for code in np.unique(cc_):
        cat = cats[code]
        if code in ctrl_codes or cat.lower() == "control":
            continue
        rows = coi_idx[cc_ == code]
        if len(rows) < 30 or sym2ens.get(cat) is None or cat.upper() not in s2i:
            continue
        perts.append((cat, s2i[cat.upper()], rows))
    C = read_rows(ds.h5_path, ctrl)[:, cols]
    A = np.array(np.load(OUT / "phase0/attention_edges_layer_mean.npy", mmap_mode="r")[8], dtype=np.float64)
    np.fill_diagonal(A, 0.0)
    pc = pd.read_csv(OUT / "phase0/control_cells.csv")["cell_idx_global"].to_numpy()
    var1 = read_rows(ds.h5_path, pc)[:, cols].var(0)          # single-log variance, same 2,000 cells
    var2 = gf["variance"].to_numpy(np.float64)                 # deployed (double-log) variance
    recs = []
    for cat, h, rows in perts:
        P = read_rows(ds.h5_path, rows)[:, cols]
        _, p = stats.ttest_ind(P, C, axis=0, equal_var=False)
        p = np.where(np.isfinite(p), p, 1.0)
        y = (np.abs(P.mean(0) - C.mean(0)) >= 0.5) & (bh(p) < 0.05)
        y[h] = False
        v = np.ones(G, bool); v[h] = False
        if y.sum() < 3 or (G - 1 - y.sum()) < 3:
            continue
        recs.append({"pert": cat, "n_pos": int(y.sum()), "att": roc_auc_score(y[v], A[h, v]),
                     "var_single_log": roc_auc_score(y[v], var1[v]), "var_deployed": roc_auc_score(y[v], var2[v])})
    df = pd.DataFrame(recs)
    gap = (df["var_single_log"] - df["att"]).to_numpy()
    rng = np.random.default_rng(51)
    B = rng.integers(0, len(df), (2000, len(df)))
    dep = pd.read_csv(OUT / "phase1/per_perturbation_auroc.csv")
    res["part3_single_log_labels_217M_K562"] = {
        "n_perturbations": int(len(df)), "n_positive_pairs": int(df["n_pos"].sum()),
        "attention_mean_auroc": float(df["att"].mean()),
        "variance_single_log_mean_auroc": float(df["var_single_log"].mean()),
        "variance_deployed_mean_auroc": float(df["var_deployed"].mean()),
        "gap_single_log_variance_minus_attention": float(gap.mean()),
        "gap_ci95_perturbation_bootstrap": [float(np.percentile(gap[B].mean(1), 2.5)), float(np.percentile(gap[B].mean(1), 97.5))],
        "n_perturbations_shared_with_deployed": int(len(set(df["pert"]) & set(dep["pert_symbol"]))),
        "deployed": {"n_perturbations": int(len(dep)), "n_positive_pairs": int(dep["n_de_positive"].sum())}}
    print(res["part3_single_log_labels_217M_K562"])
    json.dump(res, open(VER / "input_scale_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
