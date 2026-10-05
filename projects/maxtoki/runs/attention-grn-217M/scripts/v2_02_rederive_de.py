"""v2 step 2 — re-derive the knockdown-response labels from the raw h5ad files.

This is an independent re-implementation of the deployed DE rule
(phase1_trivial_baselines.py:113-203), written from the rule, not copied:

  * controls: control cells of the cell line; if > 5,000, a random 5,000
    (numpy default_rng(42).choice without replacement, sorted) — same draw as deployed.
  * a perturbation is kept if: not a control label; >= 30 cells; its symbol is in the
    Geneformer symbol->Ensembl dictionary; its symbol (upper case) is one of the
    run's 1,500 genes.
  * expression: the run's 1,500 genes only; each cell scaled to 10,000 over those
    1,500 genes, then log1p (the deployed code sums over the 1,500 genes, not the
    whole transcriptome).
  * test: Welch two-sample t-test per gene (scipy.stats.ttest_ind, equal_var=False,
    float64), perturbed cells vs controls; NaN p -> 1; BH over the 1,500 genes of
    that perturbation (own BH implementation).
  * positive ("responding gene"): |mean difference of log1p values| >= 0.5 AND BH q < 0.05;
    the perturbed gene itself is removed from the candidate list.
  * kept for evaluation only if >= 3 positives and >= 3 negatives.

Usage: python v2_02_rederive_de.py <run>   (run in v2_common.RUNS). Resumable:
skips a run whose output already exists.
"""
from __future__ import annotations

import pickle
import sys
import time

import h5py
import numpy as np
from scipy import stats

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from v2_common import PROJ, RUNS, V2, SYM2ENS_PKL, load_gene_features, bh, save_json  # noqa: E402

sys.path.insert(0, str(PROJ / "setup"))
from dataset_loader import resolve as load_ds  # noqa: E402  (metadata only)

LFC = 0.5
QMAX = 0.05
MIN_CELLS = 30
MIN_POS = 3
N_CTRL_DE = 5000
CTRL_SEED = 42

OUTD = V2 / "de_labels"
OUTD.mkdir(parents=True, exist_ok=True)


def read_rows(ds, rows_sorted: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Stream the h5ad in row blocks; return (len(rows), len(cols)) float32 (raw values)."""
    out = np.empty((len(rows_sorted), len(cols)), dtype=np.float32)
    with h5py.File(ds.h5_path, "r") as f:
        X = f["X"]
        if isinstance(X, h5py.Dataset):
            block = 969 * 8 if X.chunks else 8192
            lo, hi = int(rows_sorted[0]), int(rows_sorted[-1]) + 1
            start = (lo // block) * block
            k = 0
            for a in range(start, hi, block):
                b = min(a + block, X.shape[0])
                sel = rows_sorted[(rows_sorted >= a) & (rows_sorted < b)]
                if len(sel) == 0:
                    continue
                arr = X[a:b, :]
                out[k:k + len(sel)] = arr[sel - a][:, cols]
                k += len(sel)
            assert k == len(rows_sorted)
        else:  # CSR group (Adamson)
            import scipy.sparse as sp
            indptr = X["indptr"][:]
            n_cols = int(X.attrs["shape"][1])
            block = 5000
            k = 0
            for a in range(0, len(indptr) - 1, block):
                b = min(a + block, len(indptr) - 1)
                sel = rows_sorted[(rows_sorted >= a) & (rows_sorted < b)]
                if len(sel) == 0:
                    continue
                d = X["data"][indptr[a]:indptr[b]]
                ix = X["indices"][indptr[a]:indptr[b]]
                m = sp.csr_matrix((d, ix, indptr[a:b + 1] - indptr[a]), shape=(b - a, n_cols))
                out[k:k + len(sel)] = m[sel - a][:, cols].toarray()
                k += len(sel)
            assert k == len(rows_sorted)
    return out


def lognorm(x: np.ndarray) -> np.ndarray:
    """In place: row sums in float64, result stored float32 (memory)."""
    rs = x.sum(axis=1, dtype=np.float64, keepdims=True)
    rs[rs == 0] = 1.0
    for i in range(0, len(x), 20000):
        x[i:i + 20000] = np.log1p(x[i:i + 20000] / rs[i:i + 20000] * 1e4)
    return x


def main(run: str) -> None:
    out_path = OUTD / f"{run}.npz"
    if out_path.exists():
        print(f"{run}: exists, skip")
        return
    t0 = time.time()
    _, dskey, _ = RUNS[run]
    gf = load_gene_features(run)
    G = len(gf)
    cols = gf["var_idx"].to_numpy(np.int64)
    sym_u = [s.upper() for s in gf["symbol"]]
    s2i = {s: i for i, s in enumerate(sym_u)}
    with open(SYM2ENS_PKL, "rb") as f:
        sym2ens = pickle.load(f)

    ds = load_ds(dskey)
    coi = ds.cell_of_interest_mask
    codes = ds.perturbation_codes
    cats = ds.perturbation_categories
    ctrl_codes = ds.control_category_codes

    ctrl = np.where(coi & np.isin(codes, list(ctrl_codes)))[0]
    n_ctrl_total = len(ctrl)
    if len(ctrl) > N_CTRL_DE:
        ctrl = np.sort(np.random.default_rng(CTRL_SEED).choice(ctrl, N_CTRL_DE, replace=False))

    coi_idx = np.where(coi)[0]
    coi_codes = codes[coi]
    perts = []  # (symbol, hvg_idx, rows)
    n_skip = {"lt30": 0, "no_ens": 0, "not_in_geneset": 0}
    for code in np.unique(coi_codes):
        cat = cats[code]
        if code in ctrl_codes or cat.lower() == "control":
            continue
        rows = coi_idx[coi_codes == code]
        if len(rows) < MIN_CELLS:
            n_skip["lt30"] += 1
            continue
        if sym2ens.get(cat) is None:
            n_skip["no_ens"] += 1
            continue
        h = s2i.get(cat.upper())
        if h is None:
            n_skip["not_in_geneset"] += 1
            continue
        perts.append((cat, h, np.sort(rows)))
    print(f"{run}: controls used {len(ctrl)} of {n_ctrl_total}; candidate perturbations {len(perts)}; skipped {n_skip}")

    need = np.unique(np.concatenate([ctrl] + [p[2] for p in perts]))
    t1 = time.time()
    Xn = read_rows(ds, need, cols)
    print(f"  read {Xn.shape} in {time.time()-t1:.1f}s")
    pos_of = {int(r): i for i, r in enumerate(need)}
    Xl = lognorm(Xn)
    C = Xl[[pos_of[int(r)] for r in ctrl]].astype(np.float64)
    mu_c = C.mean(axis=0)

    keep_sym, keep_h, keep_n, labels, lfcs, qs = [], [], [], [], [], []
    dropped_few_pos = 0
    for cat, h, rows in perts:
        P = Xl[[pos_of[int(r)] for r in rows]].astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            _, p = stats.ttest_ind(P, C, axis=0, equal_var=False)
        p = np.where(np.isfinite(p), p, 1.0)
        q = bh(p)
        lfc = P.mean(axis=0) - mu_c
        y = (np.abs(lfc) >= LFC) & (q < QMAX)
        y[h] = False
        n_pos = int(y.sum())
        n_neg = int(G - 1 - n_pos)
        if n_pos < MIN_POS or n_neg < MIN_POS:
            dropped_few_pos += 1
            continue
        keep_sym.append(cat); keep_h.append(h); keep_n.append(len(rows))
        labels.append(y); lfcs.append(lfc.astype(np.float32)); qs.append(q.astype(np.float32))
    labels = np.array(labels, dtype=bool)
    np.savez_compressed(
        out_path,
        pert_symbol=np.array(keep_sym), pert_hvg_idx=np.array(keep_h, np.int64),
        n_cells=np.array(keep_n, np.int64), labels=labels,
        lfc=np.array(lfcs), q_bh=np.array(qs), control_rows=ctrl,
    )
    meta = {
        "run": run, "dataset": dskey, "n_control_cells_available": int(n_ctrl_total),
        "n_control_cells_used_for_DE": int(len(ctrl)),
        "n_candidate_perturbations": len(perts), "skipped_before_test": n_skip,
        "dropped_lt3_positives_or_negatives": dropped_few_pos,
        "n_perturbations_evaluated": int(labels.shape[0]),
        "n_positive_pairs": int(labels.sum()),
        "rule": {"lfc_abs_min": LFC, "bh_q_max": QMAX, "min_cells": MIN_CELLS,
                 "min_pos_and_neg": MIN_POS, "n_ctrl_cap": N_CTRL_DE, "ctrl_seed": CTRL_SEED,
                 "normalisation": "per cell, scale to 1e4 over the run's 1,500 genes, log1p",
                 "test": "Welch t (scipy ttest_ind equal_var=False), BH over 1,500 genes per perturbation"},
        "seconds": time.time() - t0,
    }
    save_json(meta, OUTD / f"{run}_meta.json")
    print(f"  kept {labels.shape[0]} perturbations, {int(labels.sum())} positive pairs ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    for r in (sys.argv[1:] or list(RUNS)):
        main(r)
