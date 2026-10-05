"""V3-7 independent checks (second way). Does NOT import inputs_v3, v2b_devorder_* or v3_devorder_centroids code for
the quantities it checks. It uses only: the h5ad files, the token dictionary and gene-median files, the model
weights, the v3 outputs being checked, and the run's own LET trainer (phase5_let_anchor.train_let, the deployed gate
trainer) for head fits.

Usage (project venv python):
  v3_devorder_verify.py fresh anchors      MPS. Own tokeniser from raw/X; full LlamaForCausalLM forward with no
                                           padding; per-cell states and anchor centroids vs the stored v3 ones.
  v3_devorder_verify.py fresh nll          MPS. Next-gene loss (NLL, nats per gene token) of 45 random cells with
                                           the v3 tokens and with the deployed (log1p) tokens, by assay.
  v3_devorder_verify.py gates              CPU. Own pooled-drift features, own gate code, own donor bootstrap of the
                                           key contrasts, own eval-only structured null, own lung structured control.
  v3_devorder_verify.py ss2                CPU. Smart-seq2 sensitivity: centroids from 10x cells only.
  v3_devorder_verify.py config
Outputs: outputs/v3_devorder/verify/
"""
import sys
sys.dont_write_bytecode = True
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import contextlib
import hashlib
import io
import json
import math
import pickle
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("<REPO_ROOT>/projects/maxtoki")
RUN = PROJ / "runs/manifold-discovery-217M"
SCR = RUN / "scripts"
V3 = RUN / "outputs/v3_devorder"
VO = V3 / "verify"
VO.mkdir(parents=True, exist_ok=True)
PH1 = RUN / "outputs/phase1"
DEP_ART = RUN / "artifacts"
MODEL_DIR = PROJ / "setup/MaxToki-217M-HF"
TOKDICT = PROJ / "setup/token_dictionary.json"
MEDIANS = Path("<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/crispri_validation/"
               "data/gene_median_dictionary_gc104M.pkl")
RAW = Path("<DATA_ROOT>/biodyn-work/single_cell_mechinterp/data/raw")
H5 = {"ts_immune": RAW / "tabula_sapiens_immune.h5ad", "ts_lung": RAW / "tabula_sapiens_lung.h5ad"}
DS = {"internal": "ts_immune", "external": "ts_immune", "zeroshot": "ts_immune", "lung_nonhema": "ts_lung",
      "lung_control": "ts_lung"}
BLOOD = ["internal", "external", "zeroshot"]
CHUNK = 50
DAG = json.loads((RUN / "planning/h65_stage_dag.json").read_text())
S2B = {k: v["branch"] for k, v in DAG["stage_to_branch"].items()}
NODES = list(DAG["stage_to_branch"].keys())


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def wjson(name, obj):
    p = VO / name
    t = p.with_name(p.name + ".tmp")
    t.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    os.replace(t, p)


def plan(panel):
    return pd.read_csv(V3 / f"cells/plan_{panel}.csv")


def meta(panel):
    return pd.read_csv(V3 / f"artifacts/anchors/anchor_meta_{panel}.csv")


# ============================================================================================ own tokeniser
class OwnTok:
    def __init__(self, ds):
        import h5py
        full = json.loads(TOKDICT.read_text())
        tok = {k: int(v) for k, v in full.items() if k.startswith("ENSG") and int(v) < 20275}
        with open(MEDIANS, "rb") as f:
            med = pickle.load(f)
        with h5py.File(H5[ds], "r") as f:
            k = f["raw/var"].attrs.get("_index", "_index")
            k = k.decode() if isinstance(k, bytes) else k
            ens = [s.decode().split(".")[0] for s in f["raw/var"][k][:]]
        self.cols = np.array([i for i, e in enumerate(ens) if e in tok], dtype=np.int64)
        self.tid = np.array([tok[ens[i]] for i in self.cols], dtype=np.int64)
        self.med = np.array([float(med[ens[i]]) for i in self.cols], dtype=np.float64)
        self.ds = ds

    def row_counts(self, f, r):
        X = f["raw/X"]
        a, b = int(X["indptr"][r]), int(X["indptr"][r + 1])
        n = int(X.attrs["shape"][1])
        c = np.zeros(n)
        c[X["indices"][a:b]] = X["data"][a:b]
        return c

    def tokens(self, c, max_len=4096):
        assert np.all(c == np.round(c)), "raw/X not integer"
        v = c[self.cols] / c.sum() * 1e4 / self.med
        nz = np.nonzero(v > 0)[0]
        o = nz[np.argsort(-v[nz], kind="stable")][: max_len - 2]
        return np.concatenate([[2], self.tid[o], [3]]).astype(np.int64), v, o


def same_up_to_ties(own, stored, v_by_tok, rel=1e-5):
    if len(own) != len(stored):
        return False, -1
    d = np.nonzero(own != stored)[0]
    if d.size == 0:
        return True, 0
    a = np.array([v_by_tok[int(t)] for t in own[d]]); b = np.array([v_by_tok[int(t)] for t in stored[d]])
    return bool(np.all(np.abs(a - b) <= rel * np.maximum(a, b))), int(d.size)


def stored_cell(panel, sel_i):
    k = sel_i // CHUNK
    z = np.load(V3 / f"percell/{panel}/chunk_{k:04d}.npz")
    j = int(np.nonzero(z["sel_i"] == sel_i)[0][0])
    o = z["offsets"]
    return z["states"][j], z["tokens"][o[j]:o[j + 1]].astype(np.int64), json.loads(str(z["meta"])).get("pad_to")


# ============================================================================================ fresh (MPS)
def load_full_model():
    import torch
    from transformers import LlamaForCausalLM
    m = LlamaForCausalLM.from_pretrained(str(MODEL_DIR), torch_dtype=torch.float32, attn_implementation="eager")
    m.eval().to("mps")
    return m


def forward_full(model, ids):
    import torch
    L = len(ids)
    x = torch.from_numpy(np.asarray(ids, dtype=np.int64))[None].to("mps")
    with torch.no_grad():
        out = model(input_ids=x, output_hidden_states=True, use_cache=False, return_dict=True)
        hs = torch.stack([h[0] for h in out.hidden_states], 0)
        pooled = hs[:, 1:L - 1].mean(1).cpu().numpy().astype(np.float32)
        lp = torch.log_softmax(out.logits[0, :-2].float(), -1)
        tgt = x[0, 1:-1]
        nll = float(-lp.gather(1, tgt[:, None]).mean())
        top1 = float((out.logits[0, :-2].argmax(-1) == tgt).float().mean())
    if float(torch.mps.driver_allocated_memory()) > 4e9:
        torch.mps.empty_cache()
    return pooled, nll, top1


def part_fresh(which):
    import h5py
    import torch
    torch.set_num_threads(4)
    rng = np.random.default_rng(20261003)
    t0 = time.time()
    model = load_full_model()
    toks = {}
    res = {"which": which, "rng": "default_rng(20261003)"}
    if which == "anchors":
        picks = []
        mi = meta("internal")
        picks.append(("internal", str(mi.loc[mi["n_cells_centroided"].idxmin(), "anchor_id"])))
        me = meta("external"); picks.append(("external", str(me["anchor_id"].iloc[int(rng.integers(len(me)))])))
        ml = meta("lung_nonhema"); picks.append(("lung_nonhema", str(ml["anchor_id"].iloc[int(rng.integers(len(ml)))])))
        rows = []
        for panel, aid in picks:
            P = plan(panel); ds = DS[panel]
            if ds not in toks:
                toks[ds] = OwnTok(ds)
            T = toks[ds]
            cen_v3 = np.load(V3 / f"artifacts/anchors/centroids_{panel}.npy")
            ai = int(np.nonzero(meta(panel)["anchor_id"].astype(str).to_numpy() == aid)[0][0])
            sel = P.index[P["anchor_id"].astype(str) == aid].to_numpy()
            fresh = []
            with h5py.File(H5[ds], "r") as f:
                for s in sel:
                    c = T.row_counts(f, int(P["cell_idx"].iloc[s]))
                    own, v, o = T.tokens(c)
                    st, stok, pad = stored_cell(panel, int(s))
                    vb = {int(t): float(x) for t, x in zip(T.tid, v)}
                    ok, ndiff = same_up_to_ties(own, stok, vb)
                    pooled, nll, top1 = forward_full(model, stok)
                    fresh.append(pooled)
                    rows.append({"panel": panel, "anchor": aid, "sel_i": int(s), "assay": P["assay"].iloc[s],
                                 "tokens_equal_up_to_ties": ok, "n_positions_differ": ndiff, "len": len(stok),
                                 "stored_pad_to": pad, "state_max_abs_diff": float(np.abs(pooled - st).max()),
                                 "state_max_rel_diff": float((np.abs(pooled - st) / (np.abs(st) + 1e-3)).max()),
                                 "nll_v3": nll, "top1_v3": top1})
            c_fresh = np.mean(np.stack(fresh), 0)
            res[f"{panel}|{aid}"] = {"n_cells": int(len(sel)), "centroid_max_abs_diff": float(np.abs(c_fresh - cen_v3[ai]).max()),
                                     "centroid_scale_per_state": np.abs(cen_v3[ai]).max(1).round(3).tolist()}
            print(panel, aid, res[f"{panel}|{aid}"], flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(VO / "fresh_anchor_cells.csv", index=False)
        res["all_tokens_equal_up_to_ties"] = bool(df["tokens_equal_up_to_ties"].all())
        res["n_cells_tokens_identical"] = int((df["n_positions_differ"] == 0).sum())
        res["state_max_abs_diff_max"] = float(df["state_max_abs_diff"].max())
        res["state_max_abs_diff_by_pad"] = df.groupby(df["stored_pad_to"].astype(str))["state_max_abs_diff"].max().to_dict()
    else:
        rows = []
        for panel in BLOOD:
            P = plan(panel); ds = DS[panel]
            if ds not in toks:
                toks[ds] = OwnTok(ds)
            T = toks[ds]
            dz = np.load(PH1 / f"cells_{panel}.npz")
            pick = np.sort(rng.choice(len(P), 15, replace=False))
            with h5py.File(H5[ds], "r") as f:
                for s in pick:
                    c = T.row_counts(f, int(P["cell_idx"].iloc[s]))
                    own, v, o = T.tokens(c)
                    st, stok, pad = stored_cell(panel, int(s))
                    vb = {int(t): float(x) for t, x in zip(T.tid, v)}
                    ok, ndiff = same_up_to_ties(own, stok, vb)
                    p = int(P["cells_obs_pos"].iloc[s]); L = int(dz["seq_lens"][p])
                    dtok = dz["token_ids"][p, :L].astype(np.int64)
                    pv, nv, tv = forward_full(model, stok)
                    pd_, nd, td = forward_full(model, dtok)
                    rows.append({"panel": panel, "sel_i": int(s), "assay": P["assay"].iloc[s],
                                 "tokens_equal_up_to_ties": ok, "state_max_abs_diff": float(np.abs(pv - st).max()),
                                 "len_v3": len(stok), "len_deployed": L, "nll_v3": nv, "nll_deployed": nd,
                                 "top1_v3": tv, "top1_deployed": td})
                    print(rows[-1], flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(VO / "fresh_nll_cells.csv", index=False)
        df["ss2"] = df["assay"].astype(str).str.startswith("Smart")
        b = np.random.default_rng(7)
        for grp, g in df.groupby("ss2"):
            d = (g["nll_v3"] - g["nll_deployed"]).to_numpy()
            bs = [b.choice(d, len(d)).mean() for _ in range(5000)]
            res[f"ss2={grp}"] = {"n": int(len(g)), "nll_v3_mean": float(g["nll_v3"].mean()),
                                 "nll_deployed_mean": float(g["nll_deployed"].mean()),
                                 "diff_mean": float(d.mean()), "diff_ci95_cells": np.percentile(bs, [2.5, 97.5]).tolist(),
                                 "share_v3_lower": float(np.mean(d < 0)),
                                 "top1_v3_mean": float(g["top1_v3"].mean()), "top1_deployed_mean": float(g["top1_deployed"].mean())}
        res["all_tokens_equal_up_to_ties"] = bool(df["tokens_equal_up_to_ties"].all())
        res["state_max_abs_diff_max"] = float(df["state_max_abs_diff"].max())
        res["uniform_guess_nll"] = float(np.log(20275))
    res["seconds"] = time.time() - t0
    wjson(f"fresh_{which}.json", res)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}, default=str))


# ============================================================================================ own gate code
def P5():
    sys.path.insert(0, str(SCR))
    with contextlib.redirect_stdout(io.StringIO()):
        import phase5_let_anchor as p5
    return p5


def own_pooled_drift(cen, ops, part):
    blk = lambda layers: cen[:, [i + 1 for i in layers], :].astype(np.float64).mean(1)
    e, m, l = blk(part["early"]), blk(part["mid"]), blk(part["late"])
    ye, ym, yl = e @ ops["A_early"].astype(np.float64), m @ ops["A_mid"].astype(np.float64), l @ ops["A_late"].astype(np.float64)
    return np.concatenate([ye - ym, ym - yl], 1)


def build_feats(centroids: dict):
    ops = np.load(V3 / "artifacts/operators/pooled_drift_components.npz")
    part = json.loads((V3 / "artifacts/operators/operator_index.json").read_text())["block_partition"]
    raw = {p: own_pooled_drift(c, ops, part) for p, c in centroids.items()}
    mu = raw["internal"].mean(0); sd = raw["internal"].std(0) + 1e-6
    return {p: ((r - mu) / sd).astype(np.float32) for p, r in raw.items()}


def stage_dist():
    ix = {n: i for i, n in enumerate(NODES)}
    adj = {n: set() for n in NODES}
    for a, b in DAG["edges"]:
        adj[a].add(b); adj[b].add(a)
    S = np.full((len(NODES), len(NODES)), 99.0)
    for s in NODES:
        dist = {s: 0}; q = deque([s])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if w not in dist:
                    dist[w] = dist[u] + 1; q.append(w)
        for t, d in dist.items():
            S[ix[s], ix[t]] = d
    return ix, S


IX, SD = stage_dist()


def ruler(stages):
    s = np.array([IX[x] for x in stages]); D = SD[np.ix_(s, s)].copy(); np.fill_diagonal(D, 0); return D


def ranks(x):
    from scipy.stats import rankdata
    return rankdata(x)


def spear(x, y):
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def adist(z):
    z = np.asarray(z, np.float64); zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    return np.arccos(np.clip(zn @ zn.T, -1 + 1e-7, 1 - 1e-7))


def grp_mean(Dh, D, lab, ct=None, diff=False):
    vals = []
    for g in pd.unique(lab):
        if g == "_unk":
            continue
        m = np.nonzero(lab == g)[0]
        if len(m) < 3:
            continue
        iu = np.triu_indices(len(m), 1); a, b = m[iu[0]], m[iu[1]]
        k = np.ones(len(a), bool) if not diff else ct[a] != ct[b]
        if k.sum() < 2:
            continue
        r = spear(Dh[a[k], b[k]], D[a[k], b[k]])
        if r == r:
            vals.append(r)
    return float(np.mean(vals)) if vals else np.nan, len(vals)


def fit_head(F, D, threads=1, seed=None):
    import torch
    torch.set_num_threads(threads)
    p5 = P5()
    old = p5.SEED
    if seed is not None:
        p5.SEED = int(seed)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            head, z, _ = p5.train_let(np.ascontiguousarray(F, np.float32), D.astype(np.float32), label="v3verify",
                                      verbose=False)
    finally:
        p5.SEED = old
    sd = head.state_dict()
    return sd["W.weight"].numpy().astype(np.float64), sd["b"].numpy().astype(np.float64), z


def frozen(F, W, b, m, D):
    from sklearn.manifold import trustworthiness
    z = (F.astype(np.float64) - b) @ W.T
    Dh = adist(z)
    br = np.array([S2B.get(s, "_unk") for s in m["hema_stage"].astype(str)])
    ct = m["cell_type"].astype(str).to_numpy(); dn = m["donor_id"].astype(str).to_numpy()
    iu = np.triu_indices(len(D), 1)
    dif = ct[iu[0]] != ct[iu[1]]
    rng = np.random.default_rng(42); rv = []
    for _ in range(20):
        idx = np.sort(rng.permutation(len(D))[:max(2, int(round(len(D) * 0.2)))])
        ju = np.triu_indices(len(idx), 1)
        r_ = spear(Dh[np.ix_(idx, idx)][ju], D[np.ix_(idx, idx)][ju])
        if r_ == r_:
            rv.append(r_)
    return {"trust": float(trustworthiness(F, z.astype(np.float32), n_neighbors=15)),
            "random": float(np.mean(rv)) if rv else np.nan,
            "global": spear(Dh[iu], D[iu]), "global_diff_ct": spear(Dh[iu][dif], D[iu][dif]),
            "branch": grp_mean(Dh, D, br)[0], "branch_diff_ct": grp_mean(Dh, D, br, ct, True)[0],
            "donor": grp_mean(Dh, D, dn)[0]}, Dh


def pool_results():
    d = {}
    for l in (V3 / "pool/results.jsonl").read_text().splitlines():
        if l.strip():
            j = json.loads(l)
            if j.get("error") is None:
                d[j["key"]] = j["res"]
    return d


def part_gates():
    from sklearn.manifold import trustworthiness
    t0 = time.time()
    res = {}
    PAN = ["internal", "external", "zeroshot", "lung_nonhema"]
    cen = {p: np.load(V3 / f"artifacts/anchors/centroids_{p}.npy") for p in PAN}
    F = build_feats(cen)
    res["features_vs_v3_file_max_abs_diff"] = {p: float(np.abs(F[p] - np.load(V3 / f"features/maxtoki__{p}.npy")).max())
                                               for p in PAN}
    M = {p: meta(p) for p in PAN}
    D = {p: ruler(M[p]["hema_stage"].astype(str)) for p in PAN}
    res["ruler_equal_v3_file"] = {p: bool(np.array_equal(D[p].astype(np.float32), np.load(V3 / f"artifacts/anchors/d_target_{p}.npy")))
                                  for p in PAN}
    W, b, z = fit_head(F["internal"], D["internal"])
    pr = pool_results()
    full = pr["h65|maxtoki|pos|full|-"]
    res["internal_trust_own"] = float(trustworthiness(F["internal"], z, n_neighbors=15))
    res["internal_trust_pool"] = full["trust"]
    Dh = {}
    for p in ["external", "zeroshot", "lung_nonhema"]:
        g, Dh[p] = frozen(F[p], W, b, M[p], D[p])
        res[f"{p}_own"] = g
        res[f"{p}_pool"] = {k: full["frozen"][p][k] for k in g}
    # first lung control (lung immune cells, real stage labels; replaced in the run): v3 and deployed centroids,
    # same frozen head recipe; the deployed row must reproduce the v2 numbers (0.801 / 0.894 / 0.925 / 0.522 / 0.921)
    mlc = meta("lung_control"); Dlc = ruler(mlc["hema_stage"].astype(str))
    for tag, art in (("v3", V3 / "artifacts/anchors"), ("deployed", DEP_ART / "anchors")):
        cen_all = {p: np.load(art / f"centroids_{p}.npy") for p in PAN + ["lung_control"]}
        Fx = build_feats(cen_all)
        if tag == "v3":
            Wx, bx = W, b
        else:
            dep_head = np.load(V3.parent / "v2b_devorder/heads/h65_maxtoki_pos.npz")
            Wx, bx = dep_head["W"].astype(np.float64), dep_head["b"].astype(np.float64)
        res[f"lung_control_{tag}_own"] = frozen(Fx["lung_control"], Wx, bx, mlc, Dlc)[0]
    # internal branch holdout from the pool's held-out z, own Spearman, own eval-only structured null (new rng)
    mi = M["internal"]; ct = mi["cell_type"].astype(str).to_numpy()
    held = {}
    for k, v in pr.items():
        if k.startswith("h65|maxtoki|pos|branch|"):
            t = np.array(v["test_idx"]); held[k.split("|")[4]] = (t, adist(np.array(v["z"])))
    classes = sorted({(c, s) for p in BLOOD for c, s in zip(M[p]["cell_type"].astype(str), M[p]["hema_stage"].astype(str))})
    acls = list(zip(mi["cell_type"].astype(str), mi["hema_stage"].astype(str)))

    def heldscore(stages):
        Dx = ruler(stages); a, dfs = [], []
        for g, (t, X) in held.items():
            iu = np.triu_indices(len(t), 1); y = Dx[np.ix_(t, t)][iu]; x = X[iu]
            r = spear(x, y)
            if r == r:
                a.append(r)
            dif = ct[t][iu[0]] != ct[t][iu[1]]
            if dif.sum() >= 2:
                r2 = spear(x[dif], y[dif])
                if r2 == r2:
                    dfs.append(r2)
        return np.mean(a), (np.mean(dfs) if dfs else np.nan)
    obs = heldscore(mi["hema_stage"].astype(str).tolist())
    res["internal_branch_own"] = {"all": float(obs[0]), "diff_ct": float(obs[1])}
    by_branch = {}
    for c in classes:
        by_branch.setdefault(S2B[c[1]], []).append(c)
    nulls = []
    for d in range(2000):
        r = np.random.default_rng([31338, d]); mp_ = {}
        for br_, cs in sorted(by_branch.items()):
            st = [c[1] for c in cs]; pm = r.permutation(len(cs))
            for c, j in zip(cs, pm):
                mp_[c] = st[j]
        nulls.append(heldscore([mp_[c] for c in acls]))
    nulls = np.array(nulls)
    res["internal_branch_evalnull_own"] = {
        "observed": float(obs[0]), "null_mean": float(np.nanmean(nulls[:, 0])),
        "p_one_sided": float((1 + np.sum(nulls[:, 0] >= obs[0])) / (1 + len(nulls))),
        "diff_ct_observed": float(obs[1]), "diff_ct_null_mean": float(np.nanmean(nulls[:, 1])),
        "diff_ct_p": float((1 + np.nansum(nulls[:, 1] >= obs[1])) / (1 + np.sum(~np.isnan(nulls[:, 1])))),
        "rng": "default_rng([31338, d]) d<2000"}
    # donor bootstrap (explicit expansion, copies never paired) of global contrasts; head outputs recomputed from
    # the pool's saved head weights and the feature files (own z), lookups = mean of 5 seeds
    reps = ["maxtoki", "tokenbag_pca64", "hvg_pca64"] + [f"lookup_ct_s{k}" for k in range(5)] + \
        ["maxtoki_v2b", "tokenbag_v2b"]
    V2BO = V3.parent / "v2b_devorder"
    for p in ["external", "zeroshot"]:
        Z = {}
        for r in reps:
            base = V2BO if r.endswith("_v2b") else V3
            rr = r.replace("_v2b", "").replace("tokenbag", "tokenbag_pca64") if r.endswith("_v2b") else r
            hd = np.load(base / f"heads/h65_{rr}_pos.npz")
            f = np.load(base / f"features/{rr}__{p}.npy").astype(np.float64)
            Z[r] = adist((f - hd["b"]) @ hd["W"].T)
            zf = np.load(base / f"heads/z_h65_{rr}_pos_{p}.npy")
            assert np.abs(adist(zf) - Z[r]).max() < 1e-4
        m = M[p]; dn = m["donor_id"].astype(str).to_numpy(); ctp = m["cell_type"].astype(str).to_numpy()
        ud = np.unique(dn); mem = {u: np.nonzero(dn == u)[0] for u in ud}
        Dp = D[p]

        def stat(idx):
            n = len(idx); iu = np.triu_indices(n, 1)
            keep = idx[iu[0]] != idx[iu[1]]
            a, bb = idx[iu[0][keep]], idx[iu[1][keep]]
            y = ranks(Dp[a, bb]); out = {}
            for r in reps:
                out[r] = float(np.corrcoef(ranks(Z[r][a, bb]), y)[0, 1])
            lk = np.mean([out[f"lookup_ct_s{k}"] for k in range(5)])
            return {"maxtoki": out["maxtoki"], "minus_tokenbag": out["maxtoki"] - out["tokenbag_pca64"],
                    "minus_hvg": out["maxtoki"] - out["hvg_pca64"], "minus_lookup": out["maxtoki"] - lk,
                    "tokenbag": out["tokenbag_pca64"], "tokenbag_v2b": out["tokenbag_v2b"],
                    "maxtoki_v2b": out["maxtoki_v2b"],
                    "minus_tokenbag_v2b": out["maxtoki"] - out["tokenbag_v2b"],
                    "minus_maxtoki_v2b": out["maxtoki"] - out["maxtoki_v2b"]}
        o = stat(np.arange(len(m)))
        bs = []
        for bi in range(1000):
            r = np.random.default_rng([20261003, BLOOD.index(p), bi])
            pick = r.choice(ud, len(ud), replace=True)
            bs.append(stat(np.concatenate([mem[u] for u in pick])))
        res[f"boot_global_{p}_own"] = {k: {"observed": o[k], "ci95": np.percentile([x[k] for x in bs], [2.5, 97.5]).tolist()}
                                       for k in o}
        print(p, res[f"boot_global_{p}_own"], flush=True)
    # lung structured control: one random stage per lung cell type, own rng, frozen MaxToki head (own z)
    ml = M["lung_nonhema"]; ctl = ml["cell_type"].astype(str).to_numpy(); dnl = ml["donor_id"].astype(str).to_numpy()
    uct = np.unique(ctl); n = len(ml)
    rngs = np.random.default_rng(42); splits = []
    for _ in range(20):
        idx = rngs.permutation(n); splits.append(np.sort(idx[:max(2, int(round(n * 0.2)))]))
    lung = {}
    for r in ["maxtoki", "tokenbag_pca64"]:
        hd = np.load(V3 / f"heads/h65_{r}_pos.npz")
        X = adist((np.load(V3 / f"features/{r}__lung_nonhema.npy").astype(np.float64) - hd["b"]) @ hd["W"].T)
        passes = {"branch": 0, "random": 0, "donor": 0, "all3": 0}; bmean = []
        for d in range(2000):
            rr = np.random.default_rng([62620, d])
            mp_ = dict(zip(uct, rr.choice(NODES, size=len(uct))))
            st = [mp_[c] for c in ctl]; Dx = ruler(st)
            br = np.array([S2B.get(s, "_unk") for s in st])
            def rh(ii):
                iu = np.triu_indices(len(ii), 1); return spear(X[np.ix_(ii, ii)][iu], Dx[np.ix_(ii, ii)][iu])
            rnd = np.nanmean([rh(s) for s in splits])
            don = np.nanmean([rh(np.nonzero(dnl == g)[0]) for g in np.unique(dnl) if (dnl == g).sum() >= 3])
            bv = [rh(np.nonzero(br == g)[0]) for g in np.unique(br) if (br == g).sum() >= 3]
            bra = np.nanmean(bv) if len(bv) and not np.all(np.isnan(bv)) else np.nan
            bmean.append(bra)
            pb, pr_, pdn = bra >= 0.2, rnd >= 0.2, don >= 0.2
            passes["branch"] += pb; passes["random"] += pr_; passes["donor"] += pdn; passes["all3"] += (pb and pr_ and pdn)
        lung[r] = {k: v / 2000 for k, v in passes.items()}
        lung[r]["branch_mean"] = float(np.nanmean(bmean))
    res["lung_structured_own"] = {"rng": "default_rng([62620, d]) d<2000", **lung}
    print("lung", lung, flush=True)
    res["seconds"] = time.time() - t0
    wjson("verify_gates.json", res)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("boot")}, indent=1, default=str)[:6000])


# ============================================================================================ Smart-seq2 sensitivity
def part_ss2():
    """Centroids from 10x cells only. Anchors with no 10x cell are dropped from that panel. MaxToki pooled-drift
    features re-standardised on the kept internal anchors; H65 head re-fitted; frozen gates on external / zero-shot;
    internal branch holdout re-fitted per branch. Same for the token bag (v3 tokens, 10x cells only)."""
    t0 = time.time()
    PAN = ["internal", "external", "zeroshot"]
    out = {}
    cen10, keepA, tb = {}, {}, {}
    for p in PAN:
        P = plan(p); m = meta(p)
        st = np.zeros((len(P), 12, 1232), np.float32)
        for k in range(math.ceil(len(P) / CHUNK)):
            z = np.load(V3 / f"percell/{p}/chunk_{k:04d}.npz"); st[z["sel_i"]] = z["states"]
        is10 = P["assay"].astype(str).str.startswith("10x").to_numpy()
        a_col = P["anchor_id"].astype(str).to_numpy()
        ids = m["anchor_id"].astype(str).to_numpy()
        c = np.zeros((len(ids), 12, 1232), np.float32); n10 = np.zeros(len(ids), int)
        for ai, aid in enumerate(ids):
            mk = (a_col == aid) & is10
            n10[ai] = mk.sum()
            if n10[ai]:
                c[ai] = st[mk].mean(0)
        keepA[p] = n10 > 0
        cen10[p] = c[keepA[p]]
        out[f"{p}_anchors_kept"] = int(keepA[p].sum()); out[f"{p}_anchors_total"] = int(len(ids))
        out[f"{p}_cells_10x"] = int(is10.sum()); out[f"{p}_cells_total"] = int(len(P))
        # share of each anchor's cells that are Smart-seq2
        out[f"{p}_anchors_mixed"] = int(np.sum((n10 > 0) & (n10 < m["n_cells_centroided"].to_numpy())))
    F = build_feats(cen10)
    Mk = {p: meta(p)[keepA[p]].reset_index(drop=True) for p in PAN}
    D = {p: ruler(Mk[p]["hema_stage"].astype(str)) for p in PAN}
    from sklearn.manifold import trustworthiness
    cen_all = {p: np.load(V3 / f"artifacts/anchors/centroids_{p}.npy")[keepA[p]] for p in PAN}
    Fa = build_feats(cen_all)
    seeds = [42, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    per_seed = []
    for sd_ in seeds:
        row = {"seed": sd_}
        for tag, FF in (("tenx", F), ("all", Fa)):
            W, b, z = fit_head(FF["internal"], D["internal"], seed=sd_)
            row[f"{tag}_internal_trust"] = float(trustworthiness(FF["internal"], z, n_neighbors=15))
            for p in ["external", "zeroshot"]:
                g = frozen(FF[p], W, b, Mk[p], D[p])[0]
                for k, v in g.items():
                    row[f"{tag}_{p}_{k}"] = v
        per_seed.append(row)
    ps = pd.DataFrame(per_seed)
    ps.to_csv(VO / "ss2_per_seed.csv", index=False)
    out["per_seed_mean"] = {c: float(ps[c].mean()) for c in ps.columns if c != "seed"}
    out["per_seed_sd"] = {c: float(ps[c].std(ddof=1)) for c in ps.columns if c != "seed"}
    out["paired_diff_tenx_minus_all"] = {}
    for c in ps.columns:
        if c.startswith("tenx_"):
            d = ps[c] - ps["all_" + c[5:]]
            out["paired_diff_tenx_minus_all"][c[5:]] = {"mean": float(d.mean()), "sd": float(d.std(ddof=1)),
                                                       "min": float(d.min()), "max": float(d.max())}
    W, b, z = fit_head(F["internal"], D["internal"])
    out["internal_trust"] = float(trustworthiness(F["internal"], z, n_neighbors=15))
    for p in ["external", "zeroshot"]:
        out[p] = frozen(F[p], W, b, Mk[p], D[p])[0]
    # internal branch holdout
    br = np.array([S2B.get(s, "_unk") for s in Mk["internal"]["hema_stage"].astype(str)])
    ct = Mk["internal"]["cell_type"].astype(str).to_numpy()
    per = {}
    for g in pd.unique(br):
        te = np.nonzero(br == g)[0]
        if len(te) < 3:
            continue
        Dt = D["internal"][np.ix_(te, te)]; iu = np.triu_indices(len(te), 1)
        if np.ptp(Dt[iu]) == 0:
            continue
        tr = np.nonzero(br != g)[0]
        Wg, bg, _ = fit_head(F["internal"][tr], D["internal"][np.ix_(tr, tr)])
        Dh = adist((F["internal"][te].astype(np.float64) - bg) @ Wg.T)
        dif = ct[te][iu[0]] != ct[te][iu[1]]
        per[str(g)] = {"n": int(len(te)), "rho": spear(Dh[iu], Dt[iu]),
                       "rho_diff_ct": spear(Dh[iu][dif], Dt[iu][dif]) if dif.sum() >= 2 else np.nan}
    out["internal_branch_holdout"] = float(np.nanmean([v["rho"] for v in per.values()]))
    out["internal_branch_holdout_diff_ct"] = float(np.nanmean([v["rho_diff_ct"] for v in per.values()]))
    out["internal_branch_per_group"] = per
    # reference: same own code on all cells (v3 centroids), same anchors kept, so the two differ only by the SS2 cells
    Wa, ba, za = fit_head(Fa["internal"], D["internal"])
    out["ref_all_cells_same_anchors"] = {"internal_trust": float(trustworthiness(Fa["internal"], za, n_neighbors=15)),
                                         **{p: frozen(Fa[p], Wa, ba, Mk[p], D[p])[0] for p in ["external", "zeroshot"]}}
    out["seconds"] = time.time() - t0
    wjson("verify_ss2.json", out)
    print(json.dumps(out, indent=1, default=str)[:5000])


def part_config():
    import torch, transformers, sklearn, scipy
    files = sorted(VO.glob("*.json")) + sorted(VO.glob("*.csv"))
    ins = [V3 / "pool/results.jsonl"] + sorted((V3 / "artifacts/anchors").glob("*")) + sorted((V3 / "features").glob("*.npy")) + \
        sorted((V3 / "heads").glob("h65_*_pos.npz")) + [TOKDICT, MEDIANS, SCR / "phase5_let_anchor.py"]
    wjson("run_config.json", {
        "script": str(Path(__file__).resolve()), "script_sha256": sha(__file__),
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
                     "numpy": np.__version__, "sklearn": sklearn.__version__, "scipy": scipy.__version__},
        "device": "mps for 'fresh' (float32, eager, batch 1, no padding); cpu otherwise; LET heads 1 torch thread",
        "seeds": {"fresh": "default_rng(20261003)", "nll bootstrap": "default_rng(7)",
                  "evalnull": "default_rng([31338, d])", "donor bootstrap": "default_rng([20261003, panel, b])",
                  "lung": "default_rng([62620, d])", "LET": "torch seed 42 (phase5 train_let)"},
        "inputs_sha256": {str(p): sha(p) for p in ins if p.is_file() and not p.name.startswith("._")},
        "outputs_sha256": {str(p): sha(p) for p in files if p.name != "run_config.json" and not p.name.startswith("._")}})


if __name__ == "__main__":
    part = sys.argv[1]
    if part == "fresh":
        part_fresh(sys.argv[2])
    elif part == "gates":
        part_gates()
    elif part == "ss2":
        part_ss2()
    elif part == "config":
        part_config()
