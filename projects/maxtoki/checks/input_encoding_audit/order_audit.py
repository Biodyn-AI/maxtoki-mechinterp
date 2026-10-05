"""Input-encoding audit: token order produced by each MaxToki run vs the correct rank-value order.

CPU only. No model is loaded. Uses setup/maxtoki_adapter.MaxTokiTokenizer (numpy rank encoder) only.
Reads single rows by index with h5py (never whole files).

Correct order (reference) = tokenize_cell on raw counts (or on CP10k = expm1 of log1p(CP10k));
the per-cell scale does not change the order, so counts/median and CP10k/median give the same order.

Writes order_metrics_cells.csv (one row per cell x dataset x reference), order_metrics_summary.csv,
saved_token_reproduction.json.
"""
import sys, os, json
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, h5py
from scipy.stats import spearmanr

PROJ = "<REPO_ROOT>/projects/maxtoki"
sys.path.insert(0, PROJ + "/setup")
from maxtoki_adapter import MaxTokiTokenizer  # noqa: E402
from dataset_loader import resolve as load_ds  # noqa: E402

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
RUNS = PROJ + "/runs"
B = "<DATA_ROOT>"
TSDIR = B + "/biodyn-work/single_cell_mechinterp/data/raw/"
TS_IMM = TSDIR + "tabula_sapiens_immune.h5ad"
TS_LUNG = TSDIR + "tabula_sapiens_lung.h5ad"
TS_SUB = TSDIR + "tabula_sapiens_immune_subset_20000.h5ad"
KRAS = TSDIR + "krasnow_lung_smartsq2.h5ad"
BIG = 10 ** 7

tok = MaxTokiTokenizer()


def read_row(node, i, n_genes):
    if isinstance(node, h5py.Dataset):
        return np.asarray(node[int(i), :], dtype=np.float64)
    ip = node["indptr"]
    a, b = int(ip[int(i)]), int(ip[int(i) + 1])
    v = np.zeros(n_genes, dtype=np.float64)
    v[np.asarray(node["indices"][a:b])] = np.asarray(node["data"][a:b], dtype=np.float64)
    return v


def order(vec, vm):
    """Full (untruncated) token order from the deployed tokenizer."""
    vi, vt, med = vm
    c = tok.tokenize_cell(vec.astype(np.float32), vi, vt, med, max_len=BIG)
    return np.array([], np.int64) if c is None else c.token_ids[1:-1]


def fed(vec, vm, max_len):
    vi, vt, med = vm
    c = tok.tokenize_cell(vec.astype(np.float32), vi, vt, med, max_len=max_len)
    return None if c is None else c.token_ids


def metrics(S, R, max_len, tok2med, expr_by_tok):
    n_fed = max_len - 2
    Sf, Rf = S[:n_fed], R[:n_fed]
    pS = {t: i for i, t in enumerate(S)}
    pR = {t: i for i, t in enumerate(R)}
    common = [t for t in R if t in pS]
    rho_full = spearmanr([pS[t] for t in common], [pR[t] for t in common]).correlation if len(common) > 2 else np.nan
    sSf, sRf = set(Sf.tolist()), set(Rf.tolist())
    common_fed = [t for t in Rf if t in sSf]
    pSf = {t: i for i, t in enumerate(Sf)}
    pRf = {t: i for i, t in enumerate(Rf)}
    rho_fed = spearmanr([pSf[t] for t in common_fed], [pRf[t] for t in common_fed]).correlation if len(common_fed) > 2 else np.nan
    k200 = min(200, len(R)); k2046 = min(2046, len(R))
    top200 = len(set(S[:200].tolist()) & set(R[:200].tolist())) / k200 if k200 else np.nan
    top2046 = len(set(S[:2046].tolist()) & set(R[:2046].tolist())) / k2046 if k2046 else np.nan
    kept = len(sSf & sRf) / len(Rf) if len(Rf) else np.nan
    L = max(len(Sf), len(Rf)); m = min(len(Sf), len(Rf))
    same_pos = float(np.sum(Sf[:m] == Rf[:m])) / L if L else np.nan
    # what does the script order track? 1/median vs in-cell expression (reference values)
    toks = list(S)
    inv_med = [-tok2med[t] for t in toks]           # rank by 1/median: small median first
    expr = [-expr_by_tok[t] for t in toks]
    pos = list(range(len(toks)))
    rho_invmed_script = spearmanr(pos, inv_med).correlation if len(toks) > 2 else np.nan
    rho_expr_script = spearmanr(pos, expr).correlation if len(toks) > 2 else np.nan
    toksR = list(R)
    rho_invmed_ref = spearmanr(range(len(toksR)), [-tok2med[t] for t in toksR]).correlation if len(toksR) > 2 else np.nan
    rho_expr_ref = spearmanr(range(len(toksR)), [-expr_by_tok[t] for t in toksR]).correlation if len(toksR) > 2 else np.nan
    return dict(n_genes_script=len(S), n_genes_ref=len(R), n_fed_script=len(Sf), n_fed_ref=len(Rf),
                truncated_ref=len(R) > n_fed,
                spearman_full=rho_full, spearman_fed=rho_fed, top200_overlap=top200, top2046_overlap=top2046,
                kept_set_overlap=kept, same_position_share=same_pos,
                rho_script_vs_inv_median=rho_invmed_script, rho_script_vs_expression=rho_expr_script,
                rho_ref_vs_inv_median=rho_invmed_ref, rho_ref_vs_expression=rho_expr_ref)


def var_map_from_ens(ens):
    return tok.make_var_mapping(ens)


def tok2med_map(vm):
    vi, vt, med = vm
    return {int(t): float(m) for t, m in zip(vt, med)}


rows_out = []
repro = {}


def run_dataset(name, path, rows, var_ens, max_len, script_fn_desc, refs, primary_rows, extra_info=None,
                saved_tokens=None, tech=None):
    """refs: dict ref_name -> ('X_expm1' | 'raw' | 'X_counts')."""
    vm = var_map_from_ens(var_ens)
    t2m = tok2med_map(vm)
    vi = vm[0]; vt = vm[1]
    with h5py.File(path, "r") as f:
        Xn = f["X"]
        n_genes = int(Xn.shape[1]) if isinstance(Xn, h5py.Dataset) else int(Xn.attrs["shape"][1])
        rawn = f["raw/X"] if "raw" in f and "X" in f["raw"] else None
        n_ok = 0; n_chk = 0
        for k, r in enumerate(rows):
            x = read_row(Xn, r, n_genes)
            S = order(x, vm)                         # what the script fed (X treated as counts)
            S_fed = fed(x, vm, max_len)
            if saved_tokens is not None and r in saved_tokens:
                n_chk += 1
                sv = np.asarray(saved_tokens[r]).astype(np.int64)
                sv = sv[sv != tok.PAD] if len(sv) > len(S_fed) else sv
                n_ok += int(len(sv) == len(S_fed) and np.array_equal(sv, S_fed))
            # double-log variant (the bug in the other project), for context only
            rs = x.sum(); xd = np.log1p(x / rs * 1e4) if rs > 0 else x
            D = order(xd, vm)
            for ref_name, kind in refs.items():
                if kind == "X_expm1":
                    xr = np.expm1(x)
                elif kind == "raw":
                    xr = read_row(rawn, r, n_genes)
                elif kind == "X_counts":
                    xr = x
                R = order(xr, vm)
                expr_by_tok = {int(t): float(v) for t, v in zip(vt, xr[vi])}
                m = metrics(S, R, max_len, t2m, expr_by_tok)
                md = metrics(D, R, max_len, t2m, expr_by_tok)
                row = dict(dataset=name, row=int(r), primary=bool(r in primary_rows), reference=ref_name,
                           max_len=max_len, tech=(tech[k] if tech is not None else ""), **m)
                row.update({f"dlog_{kk}": vv for kk, vv in md.items() if kk in
                            ("spearman_full", "top200_overlap", "top2046_overlap", "same_position_share")})
                rows_out.append(row)
        if saved_tokens is not None:
            repro[name] = {"n_checked": n_chk, "n_identical_to_saved": n_ok}
    print(name, "done", len(rows), flush=True)


rng0 = np.random.default_rng(0)


def pick(rows, n5=5, n50=50):
    rows = np.asarray(sorted(set(int(r) for r in rows)))
    p = np.sort(rng0.choice(rows, size=min(n5, len(rows)), replace=False))
    rest = np.setdiff1d(rows, p)
    s = np.sort(rng0.choice(rest, size=min(n50, len(rest)), replace=False))
    return p.tolist(), sorted(set(p.tolist()) | set(s.tolist()))


# ---------------- K562 controls (replogle_concat X = log1p CP10k; no raw counts in file)
ds = load_ds("k562")
z = np.load(RUNS + "/circuit-tracing-217M/outputs/v2_circuit/cells_tokens.npz")
offs = np.concatenate([[0], np.cumsum(z["lengths"])])
TF = z["tokens_flat"]
saved = {int(r): TF[offs[i]:offs[i + 1]] for i, r in enumerate(z["rows"])}
prim, allr = pick(list(saved))
run_dataset("K562_ctrl_replogle_concat", str(ds.h5_path), allr, ds.var_ensembl, 2048, "X as counts",
            {"expm1(X)=CP10k": "X_expm1"}, set(prim), saved_tokens=saved)

# ---------------- K562 perturbed cells used by v2_tf_specificity (same file)
import glob
saved_p = {}
for p in sorted(glob.glob(RUNS + "/sae-atlas-217M/outputs/v2_tf_specificity/cells/catalog_*.npz"))[:3]:
    c = np.load(p)
    CT, CO = c["tok"], c["offsets"]
    for i, r in enumerate(c["row"]):
        saved_p[int(r)] = CT[CO[i]:CO[i + 1]]
prim, allr = pick(list(saved_p), 5, 30)
run_dataset("K562_perturbed_replogle_concat", str(ds.h5_path), allr, ds.var_ensembl, 2048, "X as counts",
            {"expm1(X)=CP10k": "X_expm1"}, set(prim), saved_tokens=saved_p)

# ---------------- Adamson controls (X = log1p CP10k)
da = load_ds("adamson")
ctrl = np.where(da.cell_of_interest_mask & np.isin(da.perturbation_codes, list(da.control_category_codes)))[0]
ci = np.sort(np.random.default_rng(42).choice(ctrl, size=min(2000, len(ctrl)), replace=False))
prim, allr = pick(ci)
run_dataset("Adamson_ctrl", str(da.h5_path), allr, da.var_ensembl, 2048, "X as counts",
            {"expm1(X)=CP10k": "X_expm1"}, set(prim))

# ---------------- RPE1 controls (X = raw counts)
dr = load_ds("rpe1")
ctrl = np.where(dr.cell_of_interest_mask & np.isin(dr.perturbation_codes, list(dr.control_category_codes)))[0]
ci = np.sort(np.random.default_rng(42).choice(ctrl, size=2000, replace=False))
prim, allr = pick(ci)
run_dataset("RPE1_ctrl", str(dr.h5_path), allr, dr.var_ensembl, 2048, "X as counts",
            {"X=raw counts": "X_counts"}, set(prim))


def ts_var(path):
    with h5py.File(path, "r") as f:
        return [s.split(".")[0] for s in f["var"]["_index"][:].astype(str)]


def ts_obs(path, rows):
    with h5py.File(path, "r") as f:
        names = f["obs"]["_index"][:].astype(str) if "_index" in f["obs"] else None
    return ["SS2" if (names is not None and "_SS2_" in names[r]) else "10X" for r in rows]


TSREF = {"raw/X counts": "raw", "expm1(X)=CP10k decontX": "X_expm1"}

# ---------------- TS immune, max_len 2048 (steering cells; same file/config as spectral, topology-immune,
#                  SAE phase9/remaining TS cells)
zs = np.load(RUNS + "/exhaustive-mapping-217M/outputs/v2_steering/cells.npz", allow_pickle=True)
saved_s = {int(r): t for r, t in zip(zs["ts_rows"], zs["token_ids"]) if t is not None}
prim, allr = pick(list(saved_s))
run_dataset("TS_immune_L2048", TS_IMM, allr, ts_var(TS_IMM), 2048, "X as counts", TSREF, set(prim),
            saved_tokens=saved_s, tech=ts_obs(TS_IMM, allr))

# ---------------- TS immune, max_len 4096 (manifold internal panel)
zm = np.load(RUNS + "/manifold-discovery-217M/outputs/phase1/cells_internal.npz", mmap_mode="r")
om = pd.read_csv(RUNS + "/manifold-discovery-217M/outputs/phase1/cells_internal_obs.csv")
cand = np.random.default_rng(1).choice(len(om), size=55, replace=False)
TM, LM = zm["token_ids"], zm["seq_lens"]
saved_m = {int(om.cell_idx[i]): np.asarray(TM[i][: int(LM[i])]) for i in cand}
prim, allr = pick(list(saved_m))
run_dataset("TS_immune_L4096_manifold", TS_IMM, allr, ts_var(TS_IMM), 4096, "X as counts", TSREF, set(prim),
            saved_tokens=saved_m, tech=ts_obs(TS_IMM, allr))

# ---------------- TS lung, max_len 4096 (manifold lung_nonhema)
zl = np.load(RUNS + "/manifold-discovery-217M/outputs/phase1/cells_lung_nonhema.npz", mmap_mode="r")
ol = pd.read_csv(RUNS + "/manifold-discovery-217M/outputs/phase1/cells_lung_nonhema_obs.csv")
cand = np.random.default_rng(2).choice(len(ol), size=55, replace=False)
TL, LL = zl["token_ids"], zl["seq_lens"]
saved_l = {int(ol.cell_idx[i]): np.asarray(TL[i][: int(LL[i])]) for i in cand}
prim, allr = pick(list(saved_l))
run_dataset("TS_lung_L4096_manifold", TS_LUNG, allr, ts_var(TS_LUNG), 4096, "X as counts", TSREF, set(prim),
            saved_tokens=saved_l, tech=ts_obs(TS_LUNG, allr))

# ---------------- TS lung, max_len 2048 (topology lung domain: rng(42).choice(n, 1500))
with h5py.File(TS_LUNG, "r") as f:
    n = int(f["X"].attrs["shape"][0])
si = np.sort(np.random.default_rng(42).choice(n, size=1500, replace=False))
prim, allr = pick(si)
run_dataset("TS_lung_L2048_topology", TS_LUNG, allr, ts_var(TS_LUNG), 2048, "X as counts", TSREF, set(prim),
            tech=ts_obs(TS_LUNG, allr))

# ---------------- TS immune 20k subset, max_len 1024 (longevity stage 1)
so = pd.read_csv(RUNS + "/longevity-mechinterp-217M/outputs/stage1_20260505/sampled_obs.csv")
prim, allr = pick(so.obs_row.values)
run_dataset("TS_immune_sub20k_L1024_longevity", TS_SUB, allr, ts_var(TS_SUB), 1024, "X as counts", TSREF,
            set(prim), tech=ts_obs(TS_SUB, allr))

# ---------------- Krasnow lung Smart-seq2, max_len 2048 (topology external_lung)
with h5py.File(KRAS, "r") as f:
    n = int(f["X"].attrs["shape"][0])
si = np.sort(np.random.default_rng(42).choice(n, size=1500, replace=False))
prim, allr = pick(si)
run_dataset("Krasnow_lung_SS2_L2048_topology", KRAS, allr, ts_var(KRAS), 2048, "X as counts",
            {"raw/X counts": "raw", "expm1(X)=CPM": "X_expm1"}, set(prim))

df = pd.DataFrame(rows_out)
df.to_csv(OUT + "/order_metrics_cells.csv", index=False)
cols = ["spearman_full", "spearman_fed", "top200_overlap", "top2046_overlap", "kept_set_overlap",
        "same_position_share", "rho_script_vs_inv_median", "rho_script_vs_expression",
        "rho_ref_vs_inv_median", "rho_ref_vs_expression", "n_genes_ref",
        "dlog_spearman_full", "dlog_top200_overlap", "dlog_same_position_share"]
summ = []
for (dset, ref), g in df.groupby(["dataset", "reference"], sort=False):
    for lab, gg in [("primary5", g[g.primary]), ("all", g)]:
        d = {"dataset": dset, "reference": ref, "cells": lab, "n_cells": len(gg),
             "n_truncated_ref": int(gg.truncated_ref.sum())}
        for c in cols:
            d[c + "_mean"] = float(gg[c].mean()); d[c + "_min"] = float(gg[c].min())
        summ.append(d)
pd.DataFrame(summ).to_csv(OUT + "/order_metrics_summary.csv", index=False)
json.dump(repro, open(OUT + "/saved_token_reproduction.json", "w"), indent=1)
print(json.dumps(repro, indent=1))
