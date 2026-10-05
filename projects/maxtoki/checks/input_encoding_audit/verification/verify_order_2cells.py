# Independent order-agreement check: 2 cells per dataset, deployed path vs correct path.
import sys, json, pickle, glob
import numpy as np, h5py, pandas as pd
from scipy.stats import spearmanr
PROJ = "<REPO_ROOT>/projects/maxtoki"
sys.path.insert(0, PROJ + "/setup")
from maxtoki_adapter import MaxTokiTokenizer
from dataset_loader import K562_H5, RPE1_H5, ADAMSON_H5, SYM2ENS_PKL
B = "<DATA_ROOT>/biodyn-work/single_cell_mechinterp/data/raw"
TS_IMM, TS_SUB, TS_LUNG, KRAS = f"{B}/tabula_sapiens_immune.h5ad", f"{B}/tabula_sapiens_immune_subset_20000.h5ad", f"{B}/tabula_sapiens_lung.h5ad", f"{B}/krasnow_lung_smartsq2.h5ad"
R = PROJ + "/runs"
tok = MaxTokiTokenizer()
sym2ens = pickle.load(open(SYM2ENS_PKL, "rb"))

def dec(a): return [s.decode() if isinstance(s, bytes) else s for s in a]
def row(node, i, g):
    if isinstance(node, h5py.Dataset): return np.asarray(node[int(i)], dtype=np.float32)
    ip = node["indptr"]; a, b = int(ip[int(i)]), int(ip[int(i)+1])
    out = np.zeros(g, dtype=np.float32); out[np.asarray(node["indices"][a:b])] = np.asarray(node["data"][a:b], dtype=np.float32); return out
def var_ids(f, kind):
    v = f["var"]
    if kind == "k562": return [sym2ens.get(s) for s in dec(v["gene_name_index"][:])]
    if kind in ("rpe1", "adamson"): return dec(v["ensembl_id"][:])
    return [s.split(".")[0] for s in dec(v[v.attrs["_index"]][:])]

def genes(x, vi, vt, vm, L):
    c = tok.tokenize_cell(x, vi, vt, vm, max_len=L)
    return c.token_ids[1:-1]
def metrics(xdep, xref, vi, vt, vm, L):
    dep_full, ref_full = genes(xdep, vi, vt, vm, 10**9), genes(xref, vi, vt, vm, 10**9)
    common = np.intersect1d(dep_full, ref_full)
    pd_ = {t: k for k, t in enumerate(dep_full)}; pr = {t: k for k, t in enumerate(ref_full)}
    rho = spearmanr([pd_[t] for t in common], [pr[t] for t in common]).correlation
    K = L - 2
    dep_cut, ref_cut = dep_full[:K], ref_full[:K]
    n = min(len(dep_cut), len(ref_cut))
    same_pos = float(np.mean(dep_cut[:n] == ref_cut[:n]))
    top200 = len(set(dep_full[:200]) & set(ref_full[:200])) / 200
    kept = len(set(dep_cut) & set(ref_cut)) / len(ref_cut)
    # share of kept positions that are also kept in the ref cut, order within kept
    both = np.intersect1d(dep_cut, ref_cut)
    pdc = {t: k for k, t in enumerate(dep_cut)}; prc = {t: k for k, t in enumerate(ref_cut)}
    rho_fed = spearmanr([pdc[t] for t in both], [prc[t] for t in both]).correlation
    # 1/median ranking (the claimed driver)
    med = dict(zip(vt, vm))
    inv_med = np.argsort(np.argsort([med[t] for t in dep_full]))  # rank by median ascending = 1/median descending
    rho_dep_invmed = spearmanr(np.arange(len(dep_full)), inv_med).correlation
    rmed = np.argsort(np.argsort([med[t] for t in ref_full]))
    rho_ref_invmed = spearmanr(np.arange(len(ref_full)), rmed).correlation
    return dict(n_expr_dep=len(dep_full), n_expr_ref=len(ref_full), spearman_full=round(rho, 3), spearman_as_fed=round(rho_fed, 3),
                top200=round(top200, 3), kept_set=round(kept, 3), same_pos=round(same_pos, 4), ref_cut=bool(len(ref_full) > K),
                rho_fed_vs_invmedian=round(rho_dep_invmed, 3), rho_ref_vs_invmedian=round(rho_ref_invmed, 3))

out = []
def run(label, path, kind, rows, L, ref, saved=None):
    with h5py.File(path, "r") as f:
        g = (f["X"].shape[1] if isinstance(f["X"], h5py.Dataset) else int(f["X"].attrs["shape"][1]))
        vi, vt, vm = tok.make_var_mapping(var_ids(f, kind))
        for j, r in enumerate(rows):
            x = row(f["X"], r, g)
            if ref == "expm1X": xr = np.expm1(x.astype(np.float64)).astype(np.float32)
            elif ref == "X": xr = x
            else: xr = row(f["raw/X"], r, g)
            m = metrics(x, xr, vi, vt, vm, L)
            m.update(dataset=label, row=int(r), max_len=L, reference=ref)
            if ref == "raw/X" and "layers" in f and "decontXcounts" in f["layers"]:
                m["spearman_full_vs_decontX"] = metrics(x, row(f["layers/decontXcounts"], r, g), vi, vt, vm, L)["spearman_full"]
                m["top200_vs_decontX"] = metrics(x, row(f["layers/decontXcounts"], r, g), vi, vt, vm, L)["top200"]
            if saved is not None:
                dep = tok.tokenize_cell(x, vi, vt, vm, max_len=L).token_ids
                s = np.asarray(saved[j]); s = s[s != 0] if L == 4096 else s
                m["saved_tokens_identical"] = bool(len(dep) == len(s) and np.array_equal(dep, s))
            out.append(m); print(json.dumps(m), flush=True)

rng = np.random.default_rng(7)
# K562 controls: circuit v2 saved cells (hooks_v2 loader), 2,048
z = np.load(f"{R}/circuit-tracing-217M/outputs/v2_circuit/cells_tokens.npz")
off = np.concatenate([[0], np.cumsum(z["lengths"])]); pick = [3, 120]
run("K562 controls", K562_H5, "k562", z["rows"][pick], 2048, "expm1X", [z["tokens_flat"][off[i]:off[i+1]] for i in pick])
# K562 perturbed: v2 TF specificity catalog
cp = sorted(glob.glob(f"{R}/sae-atlas-217M/outputs/v2_tf_specificity/cells/catalog_*.npz"))[5]
z = np.load(cp); pick = [0, 10]
run("K562 perturbed", K562_H5, "k562", z["row"][pick], 2048, "expm1X", [z["tok"][z["offsets"][i]:z["offsets"][i+1]] for i in pick])
# Adamson controls and RPE1 controls: random control cells (runs sample controls with their own seeds)
with h5py.File(ADAMSON_H5, "r") as f:
    cats = dec(f["obs/condition/categories"][:]); codes = f["obs/condition/codes"][:]
ctrl = np.where(np.isin(codes, [i for i, c in enumerate(cats) if c.lower() in ("ctrl", "control")]))[0]
run("Adamson controls", ADAMSON_H5, "adamson", sorted(rng.choice(ctrl, 2, replace=False)), 2048, "expm1X")
with h5py.File(RPE1_H5, "r") as f:
    cats = dec(f["obs/perturbation/categories"][:]); codes = f["obs/perturbation/codes"][:]
ctrl = np.where(np.isin(codes, [i for i, c in enumerate(cats) if c.lower() == "control" or "non-targeting" in c.lower()]))[0]
run("RPE1 controls", RPE1_H5, "rpe1", sorted(rng.choice(ctrl, 2, replace=False)), 2048, "X")
# TS immune 2,048: v2 steering saved cells
z = np.load(f"{R}/exhaustive-mapping-217M/outputs/v2_steering/cells.npz", allow_pickle=True); pick = [7, 150]
run("TS immune 2048", TS_IMM, "ts", z["ts_rows"][pick], 2048, "raw/X", [z["token_ids"][i] for i in pick])
# TS immune 4,096: manifold internal saved cells
z = np.load(f"{R}/manifold-discovery-217M/outputs/phase1/cells_internal.npz"); obs = pd.read_csv(f"{R}/manifold-discovery-217M/outputs/phase1/cells_internal_obs.csv")
pick = [11, 5000]
run("TS immune 4096", TS_IMM, "ts", obs["cell_idx"].to_numpy()[pick], 4096, "raw/X", [z["token_ids"][i][:z["seq_lens"][i]] for i in pick])
# TS lung 4,096: lung_nonhema saved cells
z = np.load(f"{R}/manifold-discovery-217M/outputs/phase1/cells_lung_nonhema.npz"); obs = pd.read_csv(f"{R}/manifold-discovery-217M/outputs/phase1/cells_lung_nonhema_obs.csv")
pick = [2, 900]
run("TS lung 4096", TS_LUNG, "ts", obs["cell_idx"].to_numpy()[pick], 4096, "raw/X", [z["token_ids"][i][:z["seq_lens"][i]] for i in pick])
# TS lung 2,048 (topology): random rows
with h5py.File(TS_LUNG, "r") as f: n = int(f["X"].attrs["shape"][0])
run("TS lung 2048", TS_LUNG, "ts", sorted(rng.choice(n, 2, replace=False)), 2048, "raw/X")
# TS immune 20k, 1,024 (longevity): sampled_obs rows, one SS2 and one 10x
so = pd.read_csv(glob.glob(f"{R}/longevity-mechinterp-217M/outputs/stage1_*/sampled_obs.csv")[0])
tenx = so[so.obs_name.str.contains("10X")].obs_row.to_numpy(); ss2 = so[so.obs_name.str.contains("SS2")].obs_row.to_numpy()
run("TS immune 20k 1024", TS_SUB, "ts", [int(rng.choice(tenx)), int(rng.choice(ss2))], 1024, "raw/X")
# Krasnow 2,048: random rows
with h5py.File(KRAS, "r") as f: n = int(f["X"].attrs["shape"][0])
run("Krasnow 2048", KRAS, "ts", sorted(rng.choice(n, 2, replace=False)), 2048, "raw/X")
json.dump(out, open(sys.argv[1], "w"), indent=1, default=str)
