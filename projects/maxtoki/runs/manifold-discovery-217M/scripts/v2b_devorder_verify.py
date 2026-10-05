"""Independent check of item D8b (v2b_devorder_*): does MaxToki-217M carry a blood developmental ordering beyond
cell-type identity?

This script does NOT import v2b_devorder_common or any v2b_devorder_* code. It re-builds the features it needs
(MaxToki pooled drift via the run's phase5 build_pooled_drift; its own label lookups with new seeds; its own
rank-weighted token bag from the tokenised cells; its own HVG matrix from raw/X using scanpy's 'seurat' HVG code),
fits heads with the run's phase5 train_let (seed 42, 1 torch thread), and computes every gate with its own code.

Usage: python v2b_devorder_verify.py <part> [args]
  feat                      build features -> outputs/v2b_devorder/verify/feat_<rep>__<panel>.npy
  hvg                       build HVG features from raw counts (scanpy seurat flavour)
  fits <n_workers> <budget_s> [prefix]   resumable head-fit pool -> verify/fits.jsonl
  analyse                   tables, structured null, lung control, bootstraps, N60 -> verify/verify_results.json
  config                    verify/run_config.json (sha256 of inputs, scripts, outputs)
CPU only; no model forward pass.
"""
import sys
sys.dont_write_bytecode = True
import os
import contextlib, io, json, time, hashlib, traceback
from pathlib import Path
import numpy as np
import pandas as pd

RUN = Path("<REPO_ROOT>/projects/maxtoki/runs/manifold-discovery-217M")
ART = RUN / "artifacts"; PH1 = RUN / "outputs/phase1"; REP = RUN / "reports"
V2B = RUN / "outputs/v2b_devorder"
VO = V2B / "verify"; VO.mkdir(parents=True, exist_ok=True)
PANELS = ["internal", "external", "zeroshot", "lung_nonhema"]
RAW = Path("<DATA_ROOT>/biodyn-work/single_cell_mechinterp/data/raw")
H5 = {"internal": RAW / "tabula_sapiens_immune.h5ad", "external": RAW / "tabula_sapiens_immune.h5ad",
      "zeroshot": RAW / "tabula_sapiens_immune.h5ad", "lung_nonhema": RAW / "tabula_sapiens_lung.h5ad"}
CAP = {"zeroshot": 5000, "external": 12000}          # phase1bc MAX_CELLS_PER_PANEL
LOOK_SEEDS = [501, 502, 503]
REPS_BLOOD = ["maxtoki", "lkA501", "lkA502", "lkA503", "tokbag", "hvg"]
N60_RAND = [0, 1, 2, 3, 4, 5]

DAG = json.loads((RUN / "planning/h65_stage_dag.json").read_text())
S2B = {k: v["branch"] for k, v in DAG["stage_to_branch"].items()}
NODES = list(DAG["stage_to_branch"].keys())


def meta(p):
    return pd.read_csv(ART / f"anchors/anchor_meta_{p}.csv")


def branches(stages):
    return np.array([S2B.get(s, "_unk") for s in stages])


def stage_dist():
    from scipy.sparse.csgraph import shortest_path
    ix = {n: i for i, n in enumerate(NODES)}
    A = np.zeros((len(NODES), len(NODES)))
    for a, b in DAG["edges"]:
        A[ix[a], ix[b]] = A[ix[b], ix[a]] = 1
    S = shortest_path(A, unweighted=True, directed=False)
    S[~np.isfinite(S)] = 99
    return ix, S.astype(np.float32)


IX, SD = stage_dist()


def ruler(stages):
    s = np.array([IX[x] for x in stages])
    D = SD[np.ix_(s, s)].copy(); np.fill_diagonal(D, 0)
    return D


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ===================================================================================== features
def std_by_internal(F):
    mu = F["internal"].mean(0); sd = F["internal"].std(0) + 1e-6
    return {p: ((F[p] - mu) / sd).astype(np.float32) for p in F}


def pca64(F, seed=0):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=64, svd_solver="full", random_state=seed).fit(F["internal"])
    return std_by_internal({p: pca.transform(x) for p, x in F.items()})


def selected_cells(p):
    """Independent replay of phase1bc's subsample (default_rng(42), anchor order of anchors_<p>.csv)."""
    cells = pd.read_csv(PH1 / f"cells_{p}_obs.csv")
    anchors = pd.read_csv(PH1 / f"anchors_{p}.csv")
    n = len(cells); cap = CAP.get(p, n)
    if n <= cap:
        return np.arange(n), cells
    rng = np.random.default_rng(42)
    tgt = max(3, int(np.ceil(cap / max(1, len(anchors)))))
    aid = cells["anchor_id"].astype(str).to_numpy()
    keep = []
    for a in anchors["anchor_id"].astype(str):
        idx = np.flatnonzero(aid == a)
        if len(idx) > tgt:
            idx = rng.choice(idx, size=tgt, replace=False)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    return keep, cells.iloc[keep].reset_index(drop=True)


def part_feat():
    sys.path.insert(0, str(RUN / "scripts"))
    with contextlib.redirect_stdout(io.StringIO()):
        import phase5_let_anchor as P5
    info = {}
    # MaxToki pooled drift (run's own builder), internal mean/SD as phase7
    op_idx = json.loads((ART / "operators/operator_index.json").read_text())
    op = np.load(ART / "operators/pooled_drift_components.npz")
    raw = {p: P5.build_pooled_drift(np.load(ART / f"anchors/centroids_{p}.npy"), op["A_early"], op["A_mid"],
                                    op["A_late"], op_idx["block_partition"]) for p in PANELS}
    F = std_by_internal(raw)
    for p in PANELS:
        np.save(VO / f"feat_maxtoki__{p}.npy", F[p])
    # label lookups, new seeds: one N(0,1) 64-vector per exact cell_type string, + 0.5 N(0,1) noise per anchor
    M = {p: meta(p) for p in PANELS}
    labels = sorted(set(np.concatenate([M[p]["cell_type"].astype(str).to_numpy() for p in PANELS])))
    for s in LOOK_SEEDS:
        rng = np.random.default_rng(s)
        code = {c: rng.standard_normal(64) for c in labels}
        Fl = {}
        for i, p in enumerate(PANELS):
            r2 = np.random.default_rng([s, i])
            L = M[p]["cell_type"].astype(str).to_numpy()
            Fl[p] = np.stack([code[c] for c in L]) + 0.5 * r2.standard_normal((len(L), 64))
        Fl = std_by_internal(Fl)
        for p in PANELS:
            np.save(VO / f"feat_lkA{s}__{p}.npy", Fl[p])
    info["lookup_labels"] = len(labels)
    # token bag (own code): weight 1 - r/n for the gene at rank r of n (tokens between BOS and EOS)
    V = 0
    for p in PANELS:
        with np.load(PH1 / f"cells_{p}.npz") as z:
            V = max(V, int(z["token_ids"].max()) + 1)
    bags = {}
    for p in PANELS:
        keep, sub = selected_cells(p)
        m = M[p]; pos = {a: i for i, a in enumerate(m["anchor_id"].astype(str))}
        ai = np.array([pos[a] for a in sub["anchor_id"].astype(str)])
        cnt = np.bincount(ai, minlength=len(m)).astype(float)
        assert np.array_equal(cnt.astype(int), m["n_cells_centroided"].to_numpy().astype(int)), p
        with np.load(PH1 / f"cells_{p}.npz") as z:
            tok = z["token_ids"][keep]; L = z["seq_lens"][keep]
        flat = np.zeros(len(m) * V)
        for r in range(len(keep)):
            l = int(L[r])
            if l < 3:
                continue
            g = tok[r, 1:l - 1]; n = len(g)
            np.add.at(flat, ai[r] * V + g, 1.0 - np.arange(n) / n)
        bags[p] = np.log1p(flat.reshape(len(m), V) / cnt[:, None])
        del tok
    ok = bags["internal"].std(0) > 0
    Fb = pca64(std_by_internal({p: b[:, ok] for p, b in bags.items()}))
    for p in PANELS:
        np.save(VO / f"feat_tokbag__{p}.npy", Fb[p])
    info["tokbag_genes"] = int(ok.sum()); info["vocab"] = V
    # compare with the repair's token-bag and lookup files: anchor-anchor distance matrices (rotation/sign free)
    from scipy.spatial.distance import pdist
    cmp = {}
    for p in PANELS:
        a = np.load(V2B / f"features/tokenbag_pca64__{p}.npy"); b = Fb[p]
        cmp[f"tokenbag_{p}_dist_corr"] = float(np.corrcoef(pdist(a), pdist(b))[0, 1])
        cmp[f"tokenbag_{p}_max_abs_dist_diff"] = float(np.max(np.abs(pdist(a) - pdist(b))))
        a = np.load(V2B / f"features/maxtoki__{p}.npy")
        cmp[f"maxtoki_{p}_max_abs_diff"] = float(np.max(np.abs(a - F[p])))
    info["compare_with_repair_features"] = cmp
    (VO / "feat_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


def read_raw_rows(h5path, positions):
    import h5py
    order = np.argsort(positions)
    out = [None] * len(positions)
    with h5py.File(h5path, "r") as f:
        X = f["raw/X"]; ip = X["indptr"]; ind = X["indices"]; dat = X["data"]
        for o in order:
            a, b = int(ip[positions[o]]), int(ip[positions[o] + 1])
            out[o] = (ind[a:b], dat[a:b])
    return out


def part_hvg():
    """Raw counts -> per-cell CP10k -> log1p; HVG with scanpy flavor='seurat' (n_top_genes 2000) on internal cells;
    anchor mean of log1p CP10k on the HVGs; standardise; PCA-64 (internal fit)."""
    import scanpy as sc, anndata as ad, scipy.sparse as sp, h5py
    with h5py.File(H5["internal"], "r") as f:
        G = int(f["raw/X"].attrs["shape"][1])
    mats, ais = {}, {}
    for p in PANELS:
        keep, sub = selected_cells(p)
        pos = sub["cell_idx"].to_numpy().astype(np.int64)
        rows = read_raw_rows(H5[p], pos)
        indptr = np.r_[0, np.cumsum([len(r[0]) for r in rows])]
        X = sp.csr_matrix((np.concatenate([r[1] for r in rows]).astype(np.float64),
                           np.concatenate([r[0] for r in rows]), indptr), shape=(len(rows), G))
        tot = np.asarray(X.sum(1)).ravel(); tot[tot == 0] = 1
        X = sp.diags(1e4 / tot) @ X
        X.data = np.log1p(X.data)
        mats[p] = X.tocsr()
        m = meta(p); pos_a = {a: i for i, a in enumerate(m["anchor_id"].astype(str))}
        ais[p] = np.array([pos_a[a] for a in sub["anchor_id"].astype(str)])
        print(p, X.shape, flush=True)
    A = ad.AnnData(mats["internal"])
    sc.pp.highly_variable_genes(A, flavor="seurat", n_top_genes=2000)
    hv = np.flatnonzero(A.var["highly_variable"].to_numpy())
    F = {}
    for p in PANELS:
        m = meta(p); n = len(m)
        Agg = sp.csr_matrix((np.ones(len(ais[p])), (ais[p], np.arange(len(ais[p])))), shape=(n, len(ais[p])))
        cnt = np.asarray(Agg.sum(1)).ravel()
        F[p] = np.asarray((Agg @ mats[p][:, hv]).todense()) / cnt[:, None]
    Fh = pca64(std_by_internal(F))
    for p in PANELS:
        np.save(VO / f"feat_hvg__{p}.npy", Fh[p])
    theirs = np.load(V2B / "features/hvg_gene_index.npy")
    from scipy.spatial.distance import pdist
    info = {"n_hvg": int(len(hv)), "overlap_with_repair_hvg": int(len(set(hv) & set(theirs.tolist()))),
            "dist_corr_with_repair_hvg_pca64": {p: float(np.corrcoef(pdist(np.load(V2B / f"features/hvg_pca64__{p}.npy")),
                                                                       pdist(Fh[p]))[0, 1]) for p in PANELS}}
    (VO / "hvg_info.json").write_text(json.dumps(info, indent=1))
    print(info)


# ===================================================================================== fitting
_P5 = None


def P5():
    global _P5
    if _P5 is None:
        import torch
        torch.set_num_threads(1)
        sys.path.insert(0, str(RUN / "scripts"))
        with contextlib.redirect_stdout(io.StringIO()):
            import phase5_let_anchor as m
        _P5 = m
    return _P5


def fit(F, D):
    import torch
    with contextlib.redirect_stdout(io.StringIO()):
        head, z, _ = P5().train_let(np.ascontiguousarray(F, dtype=np.float32), D, label="verify", verbose=False)
    head.eval()
    return head, z


def apply(head, F):
    import torch
    with torch.no_grad():
        return head(torch.from_numpy(np.ascontiguousarray(F, dtype=np.float32)))[0].numpy()


def adist(z):
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    return np.arccos(np.clip(zn @ zn.T, -1 + 1e-7, 1 - 1e-7))


def sp_rho(x, y):
    """Spearman with average ranks (same value as scipy.stats.spearmanr for finite input)."""
    from scipy.stats import rankdata
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def pair_rho(Dh, D, ct=None):
    iu = np.triu_indices(len(D), 1)
    a = sp_rho(Dh[iu], D[iu])
    if ct is None:
        return a, float("nan")
    k = ct[iu[0]] != ct[iu[1]]
    return a, (sp_rho(Dh[iu][k], D[iu][k]) if k.sum() >= 2 else float("nan"))


def nmean(v):
    v = [x for x in v if x == x]
    return float(np.mean(v)) if v else float("nan")


def frozen_gates(f, z, D, donors, br, ct, with_trust=True):
    """phase7_external_validation.py:94-150 logic, own code, plus different-cell-type pair versions."""
    from sklearn.manifold import trustworthiness
    out = {"trust": float(trustworthiness(f, z, n_neighbors=15)) if with_trust else None}
    Dh = adist(z)
    out["global"], out["global_diff"] = pair_rho(Dh, D, ct)
    rng = np.random.default_rng(42); n = len(D); r = []
    for _ in range(20):
        idx = rng.permutation(n); t = np.sort(idx[:max(2, int(round(n * 0.2)))])
        r.append(pair_rho(Dh[np.ix_(t, t)], D[np.ix_(t, t)])[0])
    out["random"] = nmean(r)
    for name, lab in (("donor", donors), ("branch", br)):
        va, vd, per = [], [], {}
        for g in pd.unique(lab):
            if g == "_unk":
                continue
            t = np.flatnonzero(lab == g)
            if len(t) < 3:
                continue
            a, d = pair_rho(Dh[np.ix_(t, t)], D[np.ix_(t, t)], ct[t])
            va.append(a); vd.append(d); per[str(g)] = [a, d, int(len(t))]
        out[name] = nmean(va); out[name + "_diff"] = nmean(vd); out[name + "_per"] = per
    return out


def feat(rep, p):
    return np.load(VO / f"feat_{rep}__{p}.npy")


def n60_feat(arg):
    cen = np.load(ART / "anchors/centroids_internal.npy", mmap_mode="r")
    x = np.asarray(cen[:, 11, :], dtype=np.float32)
    if arg == "L10H6":
        A = np.load(ART / "operators/layer10_head6.npy").astype(np.float32)
    else:
        A = np.random.default_rng(77000 + int(arg[1:])).standard_normal((154, x.shape[1])).astype(np.float32)
    f = (x @ A.T).astype(np.float32)
    return ((f - f.mean(0)) / (f.std(0) + 1e-6)).astype(np.float32)      # phase9 lines 72-75


def run_task(key):
    t0 = time.time()
    try:
        return key, _task(key), time.time() - t0, None
    except Exception:
        return key, None, time.time() - t0, traceback.format_exc()


def _task(key):
    from sklearn.manifold import trustworthiness
    rep, part, arg = key.split("|")
    m = meta("internal"); D = np.load(ART / "anchors/d_target_internal.npy")
    ct = m["cell_type"].astype(str).to_numpy(); dn = m["donor_id"].astype(str).to_numpy()
    br = branches(m["hema_stage"].astype(str))
    F = n60_feat(rep[4:]) if rep.startswith("n60_") else feat(rep, "internal")
    n = len(F)
    if part == "full":
        head, z = fit(F, D)
        out = {"trust": float(trustworthiness(F, z, n_neighbors=15))}
        if not rep.startswith("n60_"):
            out["frozen"] = {}
            for p in ["external", "zeroshot", "lung_nonhema"]:
                mp = meta(p); fp = feat(rep, p); zp = apply(head, fp)
                Dp = np.load(ART / f"anchors/d_target_{p}.npy")
                g = frozen_gates(fp, zp, Dp, mp["donor_id"].astype(str).to_numpy(),
                                 branches(mp["hema_stage"].astype(str)), mp["cell_type"].astype(str).to_numpy())
                g["z"] = np.round(zp, 7).tolist()
                out["frozen"][p] = g
        return out
    if part == "rand":
        rng = np.random.default_rng(42)
        for _ in range(int(arg) + 1):
            idx = rng.permutation(n)
        nt = int(round(n * 0.2)); test = np.sort(idx[:nt]); train = np.sort(idx[nt:])
    else:
        lab = dn if part == "donor" else br
        test = np.flatnonzero(lab == arg); train = np.flatnonzero(lab != arg)
    head, _ = fit(F[train], D[np.ix_(train, train)])
    zt = apply(head, F[test])
    a, d = pair_rho(adist(zt), D[np.ix_(test, test)], ct[test])
    out = {"rho": a, "rho_diff": d, "n": int(len(test))}
    if part == "branch":
        out["test"] = test.tolist(); out["z"] = np.round(zt, 7).tolist()
    return out


def scorable(D, lab):
    out = []
    for g in pd.unique(lab):
        t = np.flatnonzero(lab == g)
        if len(t) < 3:
            continue
        iu = np.triu_indices(len(t), 1)
        if np.ptp(D[np.ix_(t, t)][iu]) > 0:
            out.append(str(g))
    return out


def task_list():
    m = meta("internal"); D = np.load(ART / "anchors/d_target_internal.npy")
    br = branches(m["hema_stage"].astype(str)); dn = m["donor_id"].astype(str).to_numpy()
    bg = scorable(D, br); dg = scorable(D, dn)
    T = []
    for r in REPS_BLOOD:
        T += [f"{r}|full|-"] + [f"{r}|branch|{g}" for g in bg]
    for r in ["lkA501", "lkA502", "lkA503"]:
        T += [f"{r}|rand|{i}" for i in range(10)] + [f"{r}|donor|{g}" for g in dg]
    for a in ["L10H6"] + [f"r{s}" for s in N60_RAND]:
        T += [f"n60_{a}|full|-"] + [f"n60_{a}|rand|{i}" for i in range(3)] + [f"n60_{a}|branch|{g}" for g in bg]
    return T


def load_fits():
    d = {}
    p = VO / "fits.jsonl"
    if p.exists():
        for l in p.read_text().splitlines():
            if l.strip():
                j = json.loads(l)
                if j["err"] is None:
                    d[j["key"]] = j["res"]
    return d


def _guard(a):
    key, dl = a
    if time.time() > dl:
        return key, None, 0, "SKIP"
    return run_task(key)


def _init():
    P5()


def part_fits(nw, budget, prefix=""):
    import multiprocessing as mp
    done = load_fits()
    todo = [t for t in task_list() if t not in done and t.startswith(prefix)]
    dl = time.time() + budget
    print("todo", len(todo), flush=True)
    with mp.get_context("spawn").Pool(nw, initializer=_init) as pool, open(VO / "fits.jsonl", "a") as fh:
        for key, res, sec, err in pool.imap_unordered(_guard, [(t, dl) for t in todo], chunksize=1):
            if err == "SKIP":
                continue
            fh.write(json.dumps({"key": key, "res": res, "sec": round(sec, 1), "err": err}) + "\n"); fh.flush()
            if err:
                print("ERR", key, err[-300:], flush=True)
    left = [t for t in task_list() if t not in load_fits()]
    print("left", len(left), flush=True)


# ===================================================================================== analysis
def part_analyse(which="all"):
    from scipy.stats import norm, rankdata
    d = load_fits()
    R = json.loads((VO / "verify_results.json").read_text()) if (VO / "verify_results.json").exists() else {}
    m = meta("internal"); ct = m["cell_type"].astype(str).to_numpy(); D = np.load(ART / "anchors/d_target_internal.npy")
    br = branches(m["hema_stage"].astype(str))
    if which in ("all", "tables"):
        T = {}
        for r in REPS_BLOOD:
            b = {k.split("|")[2]: v for k, v in d.items() if k.startswith(f"{r}|branch|")}
            row = {"internal_trust": d[f"{r}|full|-"]["trust"],
                   "internal_branch": nmean([v["rho"] for v in b.values()]),
                   "internal_branch_diff": nmean([v["rho_diff"] for v in b.values()]),
                   "internal_branch_per": {g: [v["rho"], v["rho_diff"], v["n"]] for g, v in b.items()}}
            rk = [k for k in d if k.startswith(f"{r}|rand|")]
            if rk:
                row["internal_random"] = nmean([d[k]["rho"] for k in rk])
                row["internal_donor"] = nmean([d[k]["rho"] for k in d if k.startswith(f"{r}|donor|")])
            for p, g in d[f"{r}|full|-"]["frozen"].items():
                row[p] = {k: v for k, v in g.items() if k != "z"}
            T[r] = row
        R["tables"] = T
        print(json.dumps({r: {k: (round(v, 4) if isinstance(v, float) else None) for k, v in T[r].items()
                              if isinstance(v, float)} for r in T}, indent=0))
        for r in T:
            e = T[r]["external"]; z = T[r]["zeroshot"]
            print(r, "ext", [round(e[k], 4) for k in ("trust", "random", "donor", "branch", "branch_diff", "global")],
                  "zs", [round(z[k], 4) for k in ("trust", "random", "donor", "branch", "branch_diff", "global")])
    if which in ("all", "null"):
        # eval-only structured null with held-out heads fixed; three class universes; T-lineage alone too
        held = {}
        for r in REPS_BLOOD:
            held[r] = {}
            for k, v in d.items():
                if k.startswith(f"{r}|branch|"):
                    t = np.array(v["test"]); Dh = adist(np.array(v["z"], dtype=np.float32)); iu = np.triu_indices(len(t), 1)
                    held[r][k.split("|")[2]] = (t, Dh[iu], iu, ct[t][iu[0]] != ct[t][iu[1]])
        acls = list(zip(m["cell_type"].astype(str), m["hema_stage"].astype(str)))
        def classes(panels):
            s = set()
            for p in panels:
                mm = meta(p); s |= set(zip(mm["cell_type"].astype(str), mm["hema_stage"].astype(str)))
            return sorted(s)
        def perm_map(cls, rng):
            byb = {}
            for c in cls:
                byb.setdefault(S2B[c[1]], []).append(c)
            out = {}
            for b in sorted(byb):
                cs = byb[b]; st = [c[1] for c in cs]; pi = rng.permutation(len(cs))
                for c, j in zip(cs, pi):
                    out[c] = st[j]
            return out
        def score(stages):
            Dn = ruler(stages); o = {}
            for r in REPS_BLOOD:
                va, vd, vt = [], [], np.nan
                for g, (t, x, iu, dif) in held[r].items():
                    y = Dn[np.ix_(t, t)][iu]
                    a = sp_rho(x, y); va.append(a)
                    vd.append(sp_rho(x[dif], y[dif]) if dif.sum() >= 2 else np.nan)
                    if g == "T_lineage":
                        vt = a
                o[r] = (nmean(va), nmean(vd), vt)
            return o
        obs = score(m["hema_stage"].astype(str).tolist())
        assert np.allclose(ruler(m["hema_stage"].astype(str).tolist()), D)
        NUL = {}
        for uname, panels in (("all_blood_classes", ["internal", "external", "zeroshot"]), ("internal_classes", ["internal"])):
            cls = classes(panels); vals = {r: [] for r in REPS_BLOOD}
            for dd in range(2000):
                mp_ = perm_map(cls, np.random.default_rng([31337, dd]))
                sc_ = score([mp_[c] for c in acls])
                for r in REPS_BLOOD:
                    vals[r].append(sc_[r])
            NUL[uname] = {}
            for r in REPS_BLOOD:
                a = np.array(vals[r], float); NUL[uname][r] = {}
                for j, nm_ in enumerate(("branch_all", "branch_diff", "T_lineage")):
                    nv = a[:, j]; nv = nv[~np.isnan(nv)]; ov = obs[r][j]
                    NUL[uname][r][nm_] = {"obs": ov, "null_mean": float(nv.mean()), "null_sd": float(nv.std(ddof=1)),
                                          "p": float((1 + (nv >= ov).sum()) / (1 + len(nv))), "n": int(len(nv))}
            print(uname, {r: {k: (round(v["obs"], 3), round(v["null_mean"], 3), round(v["p"], 3)) for k, v in NUL[uname][r].items()} for r in REPS_BLOOD})
        R["evalnull"] = NUL
    if which.startswith("lung"):
        # structured lung control: one random stage per lung CELL TYPE (uniform over the 34 stage nodes), frozen heads.
        # Gates as phase7 (random: 20 splits default_rng(42); donor; branch, skipping groups < 3 anchors / NaN).
        from scipy.stats import rankdata
        reps_sel = which.split(":")[1].split(",") if ":" in which else REPS_BLOOD
        ml = meta("lung_nonhema"); ctl = ml["cell_type"].astype(str).to_numpy(); dnl = ml["donor_id"].astype(str).to_numpy()
        uct = np.unique(ctl); n = len(ml)
        rng0 = np.random.default_rng(42); fixed = []
        for _ in range(20):
            idx = rng0.permutation(n); fixed.append(("r", np.sort(idx[:max(2, int(round(n * 0.2)))])))
        for g in pd.unique(dnl):
            t = np.flatnonzero(dnl == g)
            if len(t) >= 3:
                fixed.append(("d", t))
        def fast(xr, y):
            if np.ptp(y) == 0:
                return np.nan
            return float(np.corrcoef(xr, rankdata(y))[0, 1])
        L = R.get("lung_structured", {})
        draws = []
        for dd in range(2000):
            rng = np.random.default_rng([4242, dd])
            mp_ = dict(zip(uct, rng.choice(NODES, size=len(uct))))
            draws.append([mp_[c] for c in ctl])
        for r in reps_sel:
            z = np.array(d[f"{r}|full|-"]["frozen"]["lung_nonhema"]["z"], dtype=np.float32)
            trust = d[f"{r}|full|-"]["frozen"]["lung_nonhema"]["trust"]; Dh = adist(z)
            pre = []
            for kind, t in fixed:
                iu = np.triu_indices(len(t), 1); a = t[iu[0]]; b = t[iu[1]]
                pre.append((kind, a, b, rankdata(Dh[a, b]), np.ptp(Dh[a, b]) > 0))
            vals = []
            for st in draws:
                Dn = ruler(st); rr, dv = [], []
                for kind, a, b, xr, okx in pre:
                    v = fast(xr, Dn[a, b]) if okx else np.nan
                    (rr if kind == "r" else dv).append(v)
                brs = branches(st); bv = []
                for g in pd.unique(brs):
                    if g == "_unk":
                        continue
                    t = np.flatnonzero(brs == g)
                    if len(t) < 3:
                        continue
                    iu = np.triu_indices(len(t), 1); a = t[iu[0]]; b = t[iu[1]]
                    bv.append(sp_rho(Dh[a, b], Dn[a, b]))
                vals.append((nmean(rr), nmean(dv), nmean(bv)))
            a = np.array(vals, float)
            ok = np.nan_to_num(a, nan=-9) >= 0.20
            L[r] = {"trust": trust, "mean_random_donor_branch": np.nanmean(a, 0).tolist(), "pass_random": float(ok[:, 0].mean()),
                    "pass_donor": float(ok[:, 1].mean()), "pass_branch": float(ok[:, 2].mean()),
                    "pass_random_donor_branch": float(ok.all(1).mean()),
                    "branch_ge_0.346": float((np.nan_to_num(a[:, 2], nan=-9) >= 0.346443).mean()), "n_draws": len(draws)}
            print("lung", r, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in L[r].items()}, flush=True)
        R["lung_structured"] = L
    if which.startswith("boot:"):
        # donor cluster bootstrap by EXPLICIT expansion of the resample (pairs of two copies of one anchor dropped);
        # chunked: boot:<panel>:<b0>:<b1> appends replicates b0..b1-1 to verify/boot_<panel>.jsonl
        _, p, b0, b1 = which.split(":"); b0, b1 = int(b0), int(b1)
        mp = meta(p); ctp = mp["cell_type"].astype(str).to_numpy(); dnp = mp["donor_id"].astype(str).to_numpy()
        brp = branches(mp["hema_stage"].astype(str)); Dp = np.load(ART / f"anchors/d_target_{p}.npy")
        X = {r: adist(np.array(d[f"{r}|full|-"]["frozen"][p]["z"], dtype=np.float32)) for r in REPS_BLOOD}
        ud = sorted(set(dnp)); mem = {u: np.flatnonzero(dnp == u) for u in ud}
        def stat(idx):
            iu = np.triu_indices(len(idx), 1); a = idx[iu[0]]; b = idx[iu[1]]
            keep = a != b; a = a[keep]; b = b[keep]
            y = Dp[a, b]; dif = ctp[a] != ctp[b]; ry = rankdata(y); ryd = rankdata(y[dif])
            bra = brp[idx]; groups = []
            for g in pd.unique(bra):
                if g == "_unk":
                    continue
                mm = np.flatnonzero(bra == g)
                if len(mm) < 3:
                    continue
                iu2 = np.triu_indices(len(mm), 1); a2 = idx[mm[iu2[0]]]; b2 = idx[mm[iu2[1]]]; k2 = a2 != b2
                groups.append((a2[k2], b2[k2]))
            o = {}
            for r in REPS_BLOOD:
                x = X[r][a, b]
                o[(r, "global")] = float(np.corrcoef(rankdata(x), ry)[0, 1])
                o[(r, "global_diff")] = float(np.corrcoef(rankdata(x[dif]), ryd)[0, 1])
                o[(r, "branch")] = nmean([sp_rho(X[r][a2, b2], Dp[a2, b2]) for a2, b2 in groups])
            return o
        def contrasts(o):
            c = {}
            for mt in ("global", "global_diff", "branch"):
                lk = np.nanmean([o[(f"lkA{s}", mt)] for s in LOOK_SEEDS])
                c[f"{mt}|maxtoki"] = o[("maxtoki", mt)]
                c[f"{mt}|lookup_mean3"] = lk
                c[f"{mt}|maxtoki-lookup"] = o[("maxtoki", mt)] - lk
                c[f"{mt}|maxtoki-tokbag"] = o[("maxtoki", mt)] - o[("tokbag", mt)]
                c[f"{mt}|maxtoki-hvg"] = o[("maxtoki", mt)] - o[("hvg", mt)]
            return c
        fp = VO / f"boot_{p}.jsonl"
        with open(fp, "a") as fh:
            if b0 == 0:
                fh.write(json.dumps({"b": -1, "c": contrasts(stat(np.arange(len(mp))))}) + "\n")
            for b in range(b0, b1):
                rng = np.random.default_rng([5151, PANELS.index(p), b])
                pick = rng.choice(ud, size=len(ud), replace=True)
                fh.write(json.dumps({"b": b, "c": contrasts(stat(np.concatenate([mem[u] for u in pick])))}) + "\n")
                fh.flush()
        print("boot", p, b0, b1, "done", flush=True)
        return
    if which == "bootsum":
        B = {}
        for p in ("external", "zeroshot"):
            fp = VO / f"boot_{p}.jsonl"
            if not fp.exists():
                continue
            rows = {}
            for l in fp.read_text().splitlines():
                j = json.loads(l); rows[j["b"]] = j["c"]
            obs = rows.pop(-1); reps_ = list(rows.values())
            B[p] = {k: {"obs": v, "ci95": np.nanpercentile([x[k] for x in reps_], [2.5, 97.5]).tolist(),
                        "share_le_0": float(np.mean([x[k] <= 0 for x in reps_])), "n_boot": len(reps_)} for k, v in obs.items()}
            for k, v in B[p].items():
                print(p, k, round(v["obs"], 4), np.round(v["ci95"], 4), v["n_boot"])
        R["boot"] = B
    if which in ("all", "n60"):
        def comp(a):
            pre = f"n60_{a}|"
            if pre + "full|-" not in d:
                return None
            tr = d[pre + "full|-"]["trust"]
            rnd = nmean([d[k]["rho"] for k in d if k.startswith(pre + "rand|")])
            bra = nmean([d[k]["rho"] for k in d if k.startswith(pre + "branch|")])
            return {"trust": tr, "random": rnd, "branch": bra, "composite": 0.5 * tr + 0.25 * rnd + 0.25 * bra}
        N = {"L10H6": comp("L10H6"), "random_own_seeds": [comp(f"r{s}") for s in N60_RAND]}
        rep = json.loads((V2B / "v2b_n60_random_projection.json").read_text())
        pool = [x for x in rep["random_154_proj_of_centroid11_draws"]] + [x for x in N["random_own_seeds"] if x]
        h = N["L10H6"]; out = {}
        for k in ("trust", "random", "branch", "composite"):
            v = np.array([x[k] for x in pool]); mu, s = v.mean(), v.std(ddof=1)
            p1 = 1 - norm.cdf((h[k] - mu) / s)
            out[k] = {"L10H6": h[k], "pooled_random_mean": float(mu), "pooled_random_sd": float(s), "n": int(len(v)),
                      "share_ge": float((v >= h[k]).mean()), "P_best_of_88_normal": float(1 - (1 - p1) ** 88),
                      "P_best_of_8_normal": float(1 - (1 - p1) ** 8)}
        N["pooled_36"] = out
        sc_ = pd.read_csv(REP / "head_layer_screen.csv")
        l10 = sc_[sc_.layer == 10].sort_values("score", ascending=False)
        N["layer10_heads_composite"] = l10[["head", "score", "trustworthiness", "branch_holdout"]].round(4).values.tolist()
        N["n_real_heads_with_composite_ge_random_mean"] = int((sc_.score >= out["composite"]["pooled_random_mean"]).sum())
        R["n60"] = N
        print(json.dumps(N, indent=0, default=float)[:3000])
    (VO / "verify_results.json").write_text(json.dumps(R, indent=1, default=float))


def part_config():
    scripts = [RUN / "scripts/v2b_devorder_verify.py"]
    inputs = [ART / f"anchors/{x}_{p}.{e}" for p in PANELS for x, e in (("centroids", "npy"), ("d_target", "npy"), ("anchor_meta", "csv"))]
    inputs += [ART / "operators/pooled_drift_components.npz", ART / "operators/operator_index.json",
               ART / "operators/layer10_head6.npy", RUN / "planning/h65_stage_dag.json", REP / "head_layer_screen.csv",
               RUN / "scripts/phase5_let_anchor.py", RUN / "scripts/phase7_external_validation.py",
               RUN / "scripts/phase9_head_attribution.py", RUN / "scripts/phase1bc_hidden_states_and_centroids.py",
               V2B / "v2b_n60_random_projection.json", V2B / "features/hvg_gene_index.npy"]
    inputs += [PH1 / f"cells_{p}.npz" for p in PANELS] + [PH1 / f"cells_{p}_obs.csv" for p in PANELS] + \
              [PH1 / f"anchors_{p}.csv" for p in PANELS]
    inputs += sorted((V2B / "features").glob("tokenbag_pca64__*.npy")) + sorted((V2B / "features").glob("hvg_pca64__*.npy")) + \
              sorted((V2B / "features").glob("maxtoki__*.npy"))
    inputs += [PH1 / "cells_internal.npz", V2B / "table_gates_by_representation.csv", V2B / "v2b_tables.json",
               ]
    outs = sorted(f for f in VO.glob("*") if f.is_file() and not f.name.startswith("._") and f.name != "run_config.json")
    cfg = {"item": "independent check of D8b (v2b_devorder)", "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "inputs_sha256": {str(p): sha(p) for p in inputs}, "scripts_sha256": {str(p): sha(p) for p in scripts},
           "outputs_sha256": {str(p): sha(p) for p in outs},
           "raw_h5ad_not_hashed_bytes": {str(p): p.stat().st_size for p in set(H5.values())},
           "seeds": {"head": "phase5 train_let torch seed 42, 1 torch thread", "lookups": "codes default_rng(s), noise "
                     "default_rng([s, panel_index]), s in 501..503", "pca": "sklearn PCA svd_solver=full",
                     "eval-only null": "default_rng([31337, d]) d<2000", "lung structured": "default_rng([4242, d]) d<2000",
                     "donor bootstrap": "default_rng([5151, panel_index, b]) b<1000 (unit = donor; 13 external, 12 zero-shot)",
                     "lung deployed design": "default_rng([4343, d]) d<2000, one uniform stage per anchor",
                     "trust checks": "sklearn PCA svd_solver=full random_state=0; head seed 42; 1 torch thread",
                     "random projections": "default_rng(77000 + s) s in 0..5", "random holdout splits": "default_rng(42)"},
           "hardware": "CPU only, no model forward pass"}
    (VO / "run_config.json").write_text(json.dumps(cfg, indent=1))
    print(len(inputs), "inputs;", len(outs), "outputs")


# ===================================================================================== parallel donor bootstrap
# Same statistic and same seeds as part_analyse('boot:...') above (explicit expansion of the donor resample; pairs made
# of two copies of one anchor are dropped). Module-level so a process pool can run it. Appends to verify/boot_<p>.jsonl.
_BS = {}


def _boot_setup(p):
    d = load_fits()
    mp = meta(p)
    _BS.update(p=p, ct=mp["cell_type"].astype(str).to_numpy(), dn=mp["donor_id"].astype(str).to_numpy(),
               br=branches(mp["hema_stage"].astype(str)), D=np.load(ART / f"anchors/d_target_{p}.npy"),
               X={r: adist(np.array(d[f"{r}|full|-"]["frozen"][p]["z"], dtype=np.float32)) for r in REPS_BLOOD})
    _BS["ud"] = sorted(set(_BS["dn"])); _BS["mem"] = {u: np.flatnonzero(_BS["dn"] == u) for u in _BS["ud"]}


def _boot_stat(idx):
    from scipy.stats import rankdata
    Dp, ctp, brp, X = _BS["D"], _BS["ct"], _BS["br"], _BS["X"]
    iu = np.triu_indices(len(idx), 1); a = idx[iu[0]]; b = idx[iu[1]]
    keep = a != b; a = a[keep]; b = b[keep]
    y = Dp[a, b]; dif = ctp[a] != ctp[b]; ry = rankdata(y); ryd = rankdata(y[dif])
    bra = brp[idx]; groups = []
    for g in pd.unique(bra):
        if g == "_unk":
            continue
        mm = np.flatnonzero(bra == g)
        if len(mm) < 3:
            continue
        iu2 = np.triu_indices(len(mm), 1); a2 = idx[mm[iu2[0]]]; b2 = idx[mm[iu2[1]]]; k2 = a2 != b2
        groups.append((a2[k2], b2[k2]))
    o = {}
    for r in REPS_BLOOD:
        x = X[r][a, b]
        o[(r, "global")] = float(np.corrcoef(rankdata(x), ry)[0, 1])
        o[(r, "global_diff")] = float(np.corrcoef(rankdata(x[dif]), ryd)[0, 1])
        o[(r, "branch")] = nmean([sp_rho(X[r][a2, b2], Dp[a2, b2]) for a2, b2 in groups])
    c = {}
    for mt in ("global", "global_diff", "branch"):
        lk = np.nanmean([o[(f"lkA{s}", mt)] for s in LOOK_SEEDS])
        c[f"{mt}|maxtoki"] = o[("maxtoki", mt)]
        c[f"{mt}|lookup_mean3"] = lk
        c[f"{mt}|maxtoki-lookup"] = o[("maxtoki", mt)] - lk
        c[f"{mt}|maxtoki-tokbag"] = o[("maxtoki", mt)] - o[("tokbag", mt)]
        c[f"{mt}|maxtoki-hvg"] = o[("maxtoki", mt)] - o[("hvg", mt)]
    return c


def _boot_one(b):
    p = _BS["p"]
    rng = np.random.default_rng([5151, PANELS.index(p), b])
    pick = rng.choice(_BS["ud"], size=len(_BS["ud"]), replace=True)
    return b, _boot_stat(np.concatenate([_BS["mem"][u] for u in pick]))


def part_bootpar(p, b1, nw, budget):
    import multiprocessing as mp
    fp = VO / f"boot_{p}.jsonl"
    have = set()
    if fp.exists():
        for l in fp.read_text().splitlines():
            if l.strip():
                have.add(json.loads(l)["b"])
    todo = [b for b in range(b1) if b not in have]
    dl = time.time() + budget
    print(p, "todo", len(todo), flush=True)
    with mp.get_context("spawn").Pool(nw, initializer=_boot_setup, initargs=(p,)) as pool, open(fp, "a") as fh:
        it = pool.imap(_boot_one, todo, chunksize=2)
        for b, c in it:
            fh.write(json.dumps({"b": b, "c": c}) + "\n"); fh.flush()
            if time.time() > dl:
                pool.terminate()
                break
    n = len({json.loads(l)["b"] for l in fp.read_text().splitlines() if l.strip()})
    print(p, "have", n, flush=True)


# ===================================================================================== extra checks
def part_extra(which):
    """ruler: distinct ruler values between different (cell_type, stage) classes inside each branch.
    lungdeployed: the deployed lung design (one uniform random stage per ANCHOR, as phase1a_lung_nonhema_panel.py:62),
                  2000 draws default_rng([4343, d]), frozen heads; share of draws passing random, donor, branch.
    trust: is MaxToki's internal trust lead about input size or about whitening? Full-panel H65 fits on MaxToki
           PCA-64 with and without per-component standardisation (whitening), and PCA-256 without."""
    from scipy.stats import rankdata
    R = json.loads((VO / "verify_results.json").read_text())
    X = R.setdefault("extra", {})
    if which == "ruler":
        out = {}
        for p in ["internal", "external", "zeroshot"]:
            m = meta(p); st = m["hema_stage"].astype(str).to_numpy(); ctp = m["cell_type"].astype(str).to_numpy()
            Dp = np.load(ART / f"anchors/d_target_{p}.npy"); brp = branches(st); out[p] = {}
            for g in pd.unique(brp):
                t = np.flatnonzero(brp == g)
                iu = np.triu_indices(len(t), 1); a = t[iu[0]]; b = t[iu[1]]
                diffc = (ctp[a] != ctp[b]) | (st[a] != st[b])
                out[p][str(g)] = {"n": int(len(t)), "stages": sorted(set(st[t])),
                                  "values_between_classes": sorted(set(np.round(Dp[a, b][diffc], 3).tolist())),
                                  "values_same_class": sorted(set(np.round(Dp[a, b][~diffc], 3).tolist()))}
        X["ruler_structure"] = out
        print(json.dumps(out, indent=0)[:4000])
    if which.startswith("lungdeployed"):
        reps_sel = which.split(":")[1].split(",") if ":" in which else REPS_BLOOD
        d = load_fits()
        ml = meta("lung_nonhema"); dnl = ml["donor_id"].astype(str).to_numpy(); n = len(ml)
        rng0 = np.random.default_rng(42); fixed = []
        for _ in range(20):
            idx = rng0.permutation(n); fixed.append(("r", np.sort(idx[:max(2, int(round(n * 0.2)))])))
        for g in pd.unique(dnl):
            t = np.flatnonzero(dnl == g)
            if len(t) >= 3:
                fixed.append(("d", t))
        draws = [np.random.default_rng([4343, dd]).choice(NODES, size=n).tolist() for dd in range(2000)]
        out = X.setdefault("lung_deployed_design", {})
        for r in reps_sel:
            z = np.array(d[f"{r}|full|-"]["frozen"]["lung_nonhema"]["z"], dtype=np.float32); Dh = adist(z); vals = []
            for st in draws:
                Dn = ruler(st); rr, dv = [], []
                for kind, t in fixed:
                    (rr if kind == "r" else dv).append(pair_rho(Dh[np.ix_(t, t)], Dn[np.ix_(t, t)])[0])
                brs = branches(st); bv = []
                for g in pd.unique(brs):
                    if g == "_unk":
                        continue
                    t = np.flatnonzero(brs == g)
                    if len(t) >= 3:
                        bv.append(pair_rho(Dh[np.ix_(t, t)], Dn[np.ix_(t, t)])[0])
                vals.append((nmean(rr), nmean(dv), nmean(bv)))
            a = np.array(vals, float); ok = np.nan_to_num(a, nan=-9) >= 0.20
            out[r] = {"mean_random_donor_branch": np.nanmean(a, 0).tolist(), "sd_branch": float(np.nanstd(a[:, 2])),
                      "pass_random": float(ok[:, 0].mean()), "pass_donor": float(ok[:, 1].mean()),
                      "pass_branch": float(ok[:, 2].mean()), "pass_all3": float(ok.all(1).mean()),
                      "share_branch_le_obs_-0.1469": float((np.nan_to_num(a[:, 2], nan=9) <= -0.146871).mean())}
            print("lung deployed", r, {k: (np.round(v, 4).tolist() if isinstance(v, list) else round(v, 4)) for k, v in out[r].items()}, flush=True)
            (VO / "verify_results.json").write_text(json.dumps(R, indent=1, default=float))
    if which == "trust":
        from sklearn.decomposition import PCA
        from sklearn.manifold import trustworthiness
        D = np.load(ART / "anchors/d_target_internal.npy")
        F = feat("maxtoki", "internal").astype(np.float64)
        out = {}
        for k in (64, 256):
            S = PCA(n_components=k, svd_solver="full", random_state=0).fit_transform(F)
            for white in ((True, False) if k == 64 else (False,)):
                G = S / S.std(0) if white else S / S[:, 0].std()
                G = G.astype(np.float32)
                _, z = fit(G, D)
                out[f"pca{k}_{'whitened' if white else 'not_whitened'}"] = float(trustworthiness(G, z, n_neighbors=15))
                print(k, white, out, flush=True)
        X["trust_input_check"] = out
    if which == "trustbag":
        # token bag on internal anchors, rebuilt as in part_feat; standardised per gene; trust of the full-panel H65 fit
        # on (a) PCA-64 not whitened, (b) all genes (no PCA).
        from sklearn.decomposition import PCA
        from sklearn.manifold import trustworthiness
        D = np.load(ART / "anchors/d_target_internal.npy")
        m = meta("internal"); keep, sub = selected_cells("internal")
        pos = {a: i for i, a in enumerate(m["anchor_id"].astype(str))}
        ai = np.array([pos[a] for a in sub["anchor_id"].astype(str)]); cnt = np.bincount(ai, minlength=len(m)).astype(float)
        V = json.loads((VO / "feat_info.json").read_text())["vocab"]
        with np.load(PH1 / "cells_internal.npz") as z:
            tok = z["token_ids"][keep]; L = z["seq_lens"][keep]
        flat = np.zeros(len(m) * V)
        for r in range(len(keep)):
            l = int(L[r])
            if l < 3:
                continue
            g = tok[r, 1:l - 1]; n = len(g)
            np.add.at(flat, ai[r] * V + g, 1.0 - np.arange(n) / n)
        B = np.log1p(flat.reshape(len(m), V) / cnt[:, None]); B = B[:, B.std(0) > 0]
        B = (B - B.mean(0)) / (B.std(0) + 1e-6)
        S = PCA(n_components=64, svd_solver="full", random_state=0).fit_transform(B)
        chk = (S / S.std(0)); own = feat("tokbag", "internal")
        out = {"check_whitened_matches_saved_up_to_sign": float(np.max(np.abs(np.abs(chk) - np.abs(own))))}
        # an all-genes (19,601-d) fit was tried and did not finish inside the 9-minute limit; not reported
        for name, G in (("tokbag_pca64_not_whitened", S / S[:, 0].std()),):
            G = np.ascontiguousarray(G, dtype=np.float32); _, zz = fit(G, D)
            out[name] = float(trustworthiness(G, zz, n_neighbors=15)); print(out, flush=True)
        X["trust_tokbag_check"] = out
    (VO / "verify_results.json").write_text(json.dumps(R, indent=1, default=float))


if __name__ == "__main__":
    part = sys.argv[1]
    if part == "extra":
        part_extra(sys.argv[2])
    if part == "bootpar":
        part_bootpar(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]))
    if part == "feat":
        part_feat()
    elif part == "hvg":
        part_hvg()
    elif part == "fits":
        part_fits(int(sys.argv[2]), float(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else "")
    elif part == "analyse":
        part_analyse(sys.argv[2] if len(sys.argv) > 2 else "all")
    elif part == "config":
        part_config()
    elif part == "list":
        T = task_list(); dn = load_fits(); print(len(T), "tasks;", sum(t in dn for t in T), "done")
