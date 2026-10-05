"""Second independent check of item N01 (SAE feature annotation rates).

CPU only. No model forward pass. Written without reusing v2b_annotation_fdr.py or
v2b_annotation_fdr_verify.py.

What is new compared with the two earlier scripts:
- overlaps come from one sparse matrix product (feature x gene) @ (gene x term);
- p-values come from a lookup table P[n, K, x] = hypergeom.sf(x-1, N, K, n) (the deployed test);
- BH is done by rank counting (number of rejections = largest j with p_(j) * m / j < alpha),
  not by computing q-values (q-values are computed only to compare with the deployed file);
- a new frequency-matched null: systematic PPS sampling in random gene order. Each list of size n
  includes gene g with probability n * c_g / S exactly (c_g = count of g over the real lists,
  S = sum of c). So every gene's expected count equals its real count, unlike sequential weighted
  sampling, which flattens popular genes;
- a second, different null for context: a random relabelling of the gene universe applied to the
  real lists. It keeps which genes go together in a list and the shape of the popularity curve,
  but not which genes are popular;
- an extra BH family as a sensitivity check: BH within each feature, but only over tests with
  overlap >= 2 (a looser reading of "all terms tested for that feature").

Subcommands:
  prep                    build term sets, write terms_check.json and run_config.json
  real --layers 0-11      real catalogs, all rules, comparisons with the deployed files
  null --layers 5 --reps 20 --kind pps|perm
  summary
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
from scipy.stats import hypergeom

RUN = Path(__file__).resolve().parents[1]
OUTS = RUN / "outputs"
OUT = OUTS / "v2b_annotation_fdr_verify2"
BIOM = Path("<DATA_ROOT>/biodyn-nmi-paper")
REPLOGLE = BIOM / "src/02_cssi_method/crispri_validation/data/replogle_concat.h5ad"
GO = BIOM / "results/biological_impact/reference_edge_sets/go_bp_gene_sets.json"
KEGG = BIOM / "results/biological_impact/reference_edge_sets/kegg_gene_sets.json"
REACT = BIOM / "results/biological_impact/reference_edge_sets/reactome_gene_sets.json"
TRRUST = BIOM / "src/02_cssi_method/cssi_real_data/results/trrust_human.tsv"
ALPHA = 0.05
SEED_BASE = 77_000_000
KIND_OFF = {"pps": 100, "perm": 500, "uniform": 900}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def parse_layers(s: str):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


# ---------------------------------------------------------------- terms
def build_terms():
    with h5py.File(REPLOGLE, "r") as f:
        raw = f["var"]["gene_name_index"][:]
    var = [x.decode() if isinstance(x, bytes) else str(x) for x in raw]
    var_up = set(v.upper() for v in var)
    dbs = []
    for name, path in [("GO_BP", GO), ("KEGG", KEGG), ("Reactome", REACT)]:
        d = json.loads(path.read_text())
        for term, genes in d.items():
            dbs.append((f"{name}:{term}", set(g.upper() for g in genes)))
    tr = pd.read_csv(TRRUST, sep="\t", header=None, names=["tf", "target", "mode", "pmid"])
    tr["tf"] = tr["tf"].str.upper()
    tr["target"] = tr["target"].str.upper()
    for tf, grp in tr.groupby("tf"):
        dbs.append((f"TRRUST_TF:TRRUST_TF:{tf}", set(grp["target"].unique())))
    terms = []
    for k, v in dbs:
        inv = v & var_up
        if len(inv) >= 5:
            terms.append((k, sorted(inv)))
    var_hash = hashlib.sha256("\n".join(var).encode()).hexdigest()
    return terms, len(var), var_hash


def load_terms():
    d = json.loads((OUT / "terms_check.json").read_text())
    return d["terms"]


# ---------------------------------------------------------------- layer data
def load_layer(L, terms):
    cat = json.loads((OUTS / f"phase2/layer_{L:02d}/feature_catalog.json").read_text())
    names = json.loads((OUTS / f"phase0/layer_{L:02d}/gene_names.json").read_text())
    universe = sorted(set(g.upper() for g in names))
    gi = {g: i for i, g in enumerate(universe)}
    fids = np.array([c["feature_id"] for c in cat])
    lists = [[gi[g.upper()] for g in c["top20_genes"]] for c in cat]
    T = len(terms)
    K = np.array([len(t[1]) for t in terms], dtype=np.int64)  # term size inside the var list (deployed)
    rows, cols = [], []
    for j, (_, genes) in enumerate(terms):
        for g in genes:
            if g in gi:
                rows.append(gi[g]); cols.append(j)
    M = sp.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(len(universe), T))
    Ku = np.asarray(M.sum(axis=0)).ravel().astype(np.int64)  # term size inside the universe
    return dict(cat=cat, fids=fids, lists=lists, universe=universe, N=len(universe), M=M, K=K, Ku=Ku, T=T)


def overlap_matrix(lists, G, M):
    r = np.repeat(np.arange(len(lists)), [len(x) for x in lists])
    c = np.concatenate([np.asarray(x, dtype=np.int64) for x in lists])
    A = sp.csr_matrix((np.ones(len(c), dtype=np.int32), (r, c)), shape=(len(lists), G))
    if A.max() > 1:
        raise RuntimeError("duplicate gene inside a list")
    X = (A @ M).toarray().astype(np.int16)
    nvec = np.array([len(x) for x in lists], dtype=np.int64)
    return X, nvec


def pvals(X, nvec, Kvec, N):
    Ks, kidx = np.unique(Kvec, return_inverse=True)
    ns = np.arange(0, 21)
    xs = np.arange(0, 21)
    tab = np.ones((21, len(Ks), 21), dtype=np.float64)
    for n in np.unique(nvec):
        for a, k in enumerate(Ks):
            tab[n, a, :] = hypergeom.sf(xs - 1, N, k, n)
    tab[:, :, 0] = 1.0
    return tab[nvec[:, None], kidx[None, :], X]


# ---------------------------------------------------------------- BH helpers
def bh_n_reject(p_sorted, m):
    """p_sorted ascending 1-D. Number of BH rejections with q < ALPHA (strict, deployed convention)."""
    j = np.arange(1, len(p_sorted) + 1)
    ok = p_sorted * m / j < ALPHA
    return int(np.nonzero(ok)[0].max() + 1) if ok.any() else 0


def bh_q(p):
    """Standard BH q-values (same convention as statsmodels fdr_bh)."""
    p = np.asarray(p, dtype=np.float64)
    m = len(p)
    o = np.argsort(p, kind="mergesort")
    raw = p[o] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(raw[::-1])[::-1]
    q = np.minimum(q, 1.0)
    out = np.empty(m)
    out[o] = q
    return out


def per_feature_counts(P, mask=None):
    """Per-feature BH. Returns number of rejected terms per feature.
    mask=None: family = all T terms. Otherwise family = terms where mask is True."""
    F, T = P.shape
    if mask is None:
        S = np.sort(P, axis=1)
        j = np.arange(1, T + 1)[None, :]
        ok = S * T / j < ALPHA
        last = np.where(ok.any(axis=1), T - np.argmax(ok[:, ::-1], axis=1), 0)
        return last
    Q = np.where(mask, P, np.inf)
    S = np.sort(Q, axis=1)
    m = mask.sum(axis=1)[:, None].astype(np.float64)
    j = np.arange(1, T + 1)[None, :]
    with np.errstate(invalid="ignore"):
        ok = (S * np.maximum(m, 1) / j < ALPHA) & np.isfinite(S)
    last = np.where(ok.any(axis=1), T - np.argmax(ok[:, ::-1], axis=1), 0)
    return last


def global_reject_mask(P):
    F, T = P.shape
    m = F * T
    cand = P[P < ALPHA]
    if cand.size == 0:
        return np.zeros_like(P, dtype=bool)
    s = np.sort(cand)
    k = bh_n_reject(s, m)
    if k == 0:
        return np.zeros_like(P, dtype=bool)
    thr = s[k - 1]
    return P <= thr


def pooled_reject(P, keep):
    """BH over the kept entries only (deployed and bio-sae style). Returns boolean mask of rejections."""
    vals = P[keep]
    if vals.size == 0:
        return np.zeros_like(P, dtype=bool), vals.size
    s = np.sort(vals)
    k = bh_n_reject(s, vals.size)
    rej = np.zeros_like(P, dtype=bool)
    if k:
        rej = keep & (P <= s[k - 1])
    return rej, int(vals.size)


def score(P, X, Pu=None):
    """All rules on one catalog. Returns a dict of feature counts."""
    out = {}
    repA, _ = pooled_reject(P, (X >= 2) & (P < 0.1))
    out["A_deployed"] = int(repA.any(axis=1).sum())
    repA2, _ = pooled_reject(P, X >= 2)
    out["A2_biosae"] = int(repA2.any(axis=1).sum())
    nb = per_feature_counts(P)
    out["B_perfeature"] = int((nb > 0).sum())
    out["B_ge3"] = int((nb >= 3).sum())
    nb2 = per_feature_counts(P, mask=(X >= 2))
    out["B2_perfeature_ovl2"] = int((nb2 > 0).sum())
    gC = global_reject_mask(P)
    out["C_global"] = int(gC.any(axis=1).sum())
    if Pu is not None:
        out["Bu_perfeature"] = int((per_feature_counts(Pu) > 0).sum())
        out["Cu_global"] = int(global_reject_mask(Pu).any(axis=1).sum())
    return out, dict(repA=repA, nb=nb, gC=gC)


# ---------------------------------------------------------------- nulls
def null_pps(lists, G, rng):
    """Systematic PPS in a fresh random gene order for every list.
    Gene g enters a list of size n with probability n * c_g / S (exact), so E[count_g] = c_g.
    A gene is picked when the grid u, u+1, ..., u+n-1 falls in its slice of the cumulative sum.
    Calibration check at L5 (10 catalogs): mean null count vs real count, slope 1.0007, r = 0.986."""
    counts = np.bincount(np.concatenate([np.asarray(x) for x in lists]), minlength=G).astype(np.float64)
    S = counts.sum()
    out = []
    for x in lists:
        n = len(x)
        pi = n * counts / S
        if pi.max() >= 1:
            raise RuntimeError("inclusion probability >= 1")
        order = rng.permutation(G)
        cum = np.cumsum(pi[order])
        u = rng.random()
        pts = u + np.arange(n)
        pos = np.searchsorted(cum, pts, side="right")
        pos = np.minimum(pos, G - 1)
        sel = order[pos]
        if len(np.unique(sel)) != n:
            raise RuntimeError("pps produced a duplicate")
        out.append(sel.tolist())
    return out, counts


def null_uniform(lists, G, rng):
    return [rng.choice(G, size=len(x), replace=False).tolist() for x in lists], None


def null_perm(lists, G, rng):
    sigma = rng.permutation(G)
    return [sigma[np.asarray(x)].tolist() for x in lists], None


# ---------------------------------------------------------------- commands
def cmd_prep(_args):
    OUT.mkdir(parents=True, exist_ok=True)
    terms, n_var, var_hash = build_terms()
    (OUT / "terms_check.json").write_text(json.dumps({"n_var": n_var, "var_sha256": var_hash, "terms": terms}))
    # compare with the repair's terms.json
    cmp = {}
    tj = OUTS / "v2b_annotation_fdr/terms.json"
    if tj.exists():
        other = json.loads(tj.read_text())
        if isinstance(other, dict) and "terms" in other:
            other = other["terms"]
        try:
            if isinstance(other, list) and other and isinstance(other[0], dict):
                od = {t["term"]: set(t["genes"]) for t in other}
            elif isinstance(other, list):
                od = {t[0]: set(t[1]) for t in other}
            else:
                od = {k: set(v) for k, v in other.items()}
            md = {k: set(v) for k, v in terms}
            cmp = {"repair_n_terms": len(od), "mine_n_terms": len(md),
                   "same_names": set(od) == set(md),
                   "n_sets_differ": int(sum(od.get(k) != v for k, v in md.items()))}
        except Exception as e:  # noqa
            cmp = {"error": repr(e)}
    inputs = [GO, KEGG, REACT, TRRUST]
    for L in range(12):
        inputs += [OUTS / f"phase2/layer_{L:02d}/feature_catalog.json",
                   OUTS / f"phase0/layer_{L:02d}/gene_names.json",
                   OUTS / f"phase2/layer_{L:02d}/significant_enrichments.csv",
                   OUTS / f"phase2/layer_{L:02d}/annotation_summary.json"]
    inputs += [OUTS / "phase6/causal_patching.csv", RUN / "scripts/phase6_patching.py",
               OUTS / "v2b_annotation_fdr/summary.csv"]
    cfg = {
        "item": "N01 second independent check",
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha(Path(__file__).resolve()),
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "pandas": pd.__version__, "device": "cpu, no model forward pass",
        "alpha": ALPHA, "bh_convention": "reject when q < alpha (strict), as in the deployed code",
        "seed_rule": "null rep r of kind k at layer L: numpy default_rng(77000000 + 1000*L + offset[k] + r)",
        "kind_offset": KIND_OFF,
        "commands_run_in_order": [
            "prep", "real --layers 5", "real --layers 0-4", "real --layers 6-11",
            "null --kind pps --layers 5 --reps 20 (first 2 reps as a timing test, same seeds)",
            "null --kind pps --reps 10 for layers 0,11 / 1-3 / 4,6,7 / 8,9 / 10",
            "null --kind perm --layers 0,5,11 --reps 5", "null --kind uniform --layers 0,5,11 --reps 3",
            "real --layers 5 (re-run after fixing the phase-6 tie-break readout; counts unchanged)",
            "summary", "prep (last, to record the final script hash)"],
        "null_reps": {"pps": "20 at L5, 10 at every other layer", "perm": "5 at L0, L5, L11", "uniform": "3 at L0, L5, L11"},
        "n_terms": len(terms), "replogle_var_n": n_var, "replogle_var_sha256_newline_joined": var_hash,
        "terms_compare_with_repair": cmp,
        "interval_methods": {
            "null_range": "min-max and 2.5-97.5 percentile over null catalogs, unit = annotated features",
            "z": "(real count - null mean) / null SD (ddof=1) over null catalogs",
            "wilson": "Wilson 95% interval for the real rate, alive features as units",
        },
        "input_sha256": {str(p): sha(p) for p in inputs if p.exists()},
    }
    (OUT / "run_config.json").write_text(json.dumps(cfg, indent=1))
    print(len(terms), "terms;", cmp)


def wilson(k, n, z=1.959963984540054):
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def cmd_real(args):
    terms = load_terms()
    names = [t[0] for t in terms]
    (OUT / "real").mkdir(parents=True, exist_ok=True)
    for L in parse_layers(args.layers):
        t0 = time.time()
        d = load_layer(L, terms)
        X, nvec = overlap_matrix(d["lists"], d["N"], d["M"])
        P = pvals(X, nvec, d["K"], d["N"])
        Pu = pvals(X, nvec, d["Ku"], d["N"])
        res, aux = score(P, X, Pu)
        F, T = P.shape
        rec = {"layer": L, "n_alive": F, "N_universe": d["N"], "n_terms": T, "n_tests": F * T}
        rec.update(res)
        # deployed file comparison
        summ = json.loads((OUTS / f"phase2/layer_{L:02d}/annotation_summary.json").read_text())
        dep = pd.read_csv(OUTS / f"phase2/layer_{L:02d}/significant_enrichments.csv")
        rec["deployed_summary"] = summ
        keepA = (X >= 2) & (P < 0.1)
        rec["deployed_kept_tests"] = int(keepA.sum())
        fpos = {f: i for i, f in enumerate(d["fids"])}
        tpos = {t: j for j, t in enumerate(names)}
        mine = set(zip(*np.nonzero(aux["repA"])))
        theirs = set((fpos[f], tpos[t]) for f, t in zip(dep["feature_id"], dep["term"]))
        rec["rows_mine"] = len(mine); rec["rows_deployed"] = len(theirs)
        rec["rows_missing"] = len(theirs - mine); rec["rows_extra"] = len(mine - theirs)
        fi = dep["feature_id"].map(fpos).to_numpy(); tj = dep["term"].map(tpos).to_numpy()
        rec["max_abs_p_diff"] = float(np.max(np.abs(P[fi, tj] - dep["p_raw"].to_numpy())))
        rec["overlap_col_match"] = bool((X[fi, tj] == dep["overlap"].to_numpy()).all())
        rec["n_col_match"] = bool((nvec[fi] == dep["n"].to_numpy()).all())
        rec["K_col_match"] = bool((d["K"][tj] == dep["K"].to_numpy()).all())
        rec["N_col_match"] = bool((dep["N"] == d["N"]).all())
        qv = bh_q(P[keepA])
        qfull = np.full(P.shape, np.nan); qfull[keepA] = qv
        rec["max_abs_q_diff"] = float(np.max(np.abs(qfull[fi, tj] - dep["q_bh"].to_numpy())))
        rec["deployed_max_raw_p_sig"] = float(dep["p_raw"].max())
        rec["deployed_share_overlap2"] = float((dep["overlap"] == 2).mean())
        # per-feature BH surviving rows: overlap profile
        nb = aux["nb"]
        S_idx = np.argsort(P, axis=1, kind="mergesort")
        surv_rows = []
        for f in np.nonzero(nb)[0]:
            for j in S_idx[f, : nb[f]]:
                surv_rows.append((f, j))
        sr = np.array(surv_rows) if surv_rows else np.zeros((0, 2), int)
        if len(sr):
            ov = X[sr[:, 0], sr[:, 1]]
            rec["B_rows"] = int(len(sr))
            rec["B_rows_share_overlap2"] = float((ov == 2).mean())
            rec["B_rows_overlap2_full20"] = int(((ov == 2) & (nvec[sr[:, 0]] == 20)).sum())
            rec["B_rows_overlap2_short"] = int(((ov == 2) & (nvec[sr[:, 0]] < 20)).sum())
            rec["B_rows_overlap1"] = int((ov == 1).sum())
        lo, hi = wilson(rec["B_perfeature"], F)
        rec["B_rate"] = rec["B_perfeature"] / F
        rec["B_wilson"] = [lo, hi]
        rec["A_rate"] = rec["A_deployed"] / F
        rec["A2_rate"] = rec["A2_biosae"] / F
        rec["B2_rate"] = rec["B2_perfeature_ovl2"] / F
        rec["list_sizes_lt20"] = int((nvec < 20).sum())
        # global-BH features
        gC = aux["gC"]
        gl = []
        for f in np.nonzero(gC.any(axis=1))[0]:
            js = np.nonzero(gC[f])[0]
            jb = js[np.argmin(P[f, js])]
            gl.append({"feature_id": int(d["fids"][f]), "n_terms_global": int(len(js)),
                       "n_terms_perfeature": int(nb[f]), "min_p": float(P[f, jb]), "best_term": names[jb]})
        rec["global_features"] = gl
        # flags for comparison with repair's feature_flags
        flags = pd.DataFrame({"feature_id": d["fids"], "n_perfeature": nb,
                              "n_global": gC.sum(axis=1), "n_deployed": aux["repA"].sum(axis=1)})
        flags.to_csv(OUT / f"real/layer_{L:02d}_flags.csv", index=False)
        if L == 5:
            # phase 6 rule (phase6_patching.py:56-66) on the deployed file
            cnt = dep.groupby("feature_id").size().to_dict()
            rich = sorted([c for c in d["cat"] if cnt.get(c["feature_id"], 0) >= 3],
                          key=lambda c: -cnt.get(c["feature_id"], 0))
            picks = [c["feature_id"] for c in rich[:50]]
            used = pd.read_csv(OUTS / "phase6/causal_patching.csv")["feature_id"].tolist()
            fl = flags.set_index("feature_id")
            rec["phase6"] = {
                "picks_equal_used": int(len(set(picks) & set(used))), "n_used": len(used),
                "deployed_count_range": [int(min(cnt[f] for f in used)), int(max(cnt[f] for f in used))],
                "count_50th": int(cnt[picks[49]]), "count_51st": int(cnt[rich[50]["feature_id"]]) if len(rich) > 50 else None,
                "n_tied_at_boundary": int(sum(1 for v in cnt.values() if v == cnt[picks[49]])),
                "n_tied_inside_top50": int(sum(1 for f in picks if cnt[f] == cnt[picks[49]])),
                "annotated_perfeature": int((fl.loc[used, "n_perfeature"] > 0).sum()),
                "ge3_perfeature": int((fl.loc[used, "n_perfeature"] >= 3).sum()),
                "annotated_global": int((fl.loc[used, "n_global"] > 0).sum()),
                "global_ids": [int(f) for f in used if fl.loc[f, "n_global"] > 0],
            }
        rec["seconds"] = round(time.time() - t0, 1)
        (OUT / f"real/layer_{L:02d}.json").write_text(json.dumps(rec, indent=1))
        print(L, {k: rec[k] for k in ["A_deployed", "A2_biosae", "B_perfeature", "B2_perfeature_ovl2",
                                       "C_global", "Bu_perfeature", "Cu_global", "rows_missing",
                                       "rows_extra", "max_abs_p_diff", "max_abs_q_diff", "seconds"]}, flush=True)


def cmd_null(args):
    terms = load_terms()
    (OUT / "null").mkdir(parents=True, exist_ok=True)
    for L in parse_layers(args.layers):
        d = load_layer(L, terms)
        G = d["N"]
        path = OUT / f"null/layer_{L:02d}_{args.kind}.json"
        done = json.loads(path.read_text()) if path.exists() else {"reps": {}}
        real_counts = np.bincount(np.concatenate([np.asarray(x) for x in d["lists"]]), minlength=G)
        for r in range(args.reps):
            if str(r) in done["reps"]:
                continue
            t0 = time.time()
            rng = np.random.default_rng(SEED_BASE + 1000 * L + KIND_OFF[args.kind] + r)
            if args.kind == "pps":
                lists, _ = null_pps(d["lists"], G, rng)
            elif args.kind == "perm":
                lists, _ = null_perm(d["lists"], G, rng)
            else:
                lists, _ = null_uniform(d["lists"], G, rng)
            X, nvec = overlap_matrix(lists, G, d["M"])
            P = pvals(X, nvec, d["K"], d["N"])
            res, _ = score(P, X)
            nc = np.bincount(np.concatenate([np.asarray(x) for x in lists]), minlength=G)
            res["count_corr_with_real"] = float(np.corrcoef(nc, real_counts)[0, 1])
            res["sizes_equal"] = bool((nvec == np.array([len(x) for x in d["lists"]])).all())
            res["slots_same_as_real"] = float(np.mean([len(set(a) & set(b)) / len(b) for a, b in zip(lists, d["lists"])]))
            res["seconds"] = round(time.time() - t0, 1)
            done["reps"][str(r)] = res
            path.write_text(json.dumps(done, indent=1))
            print(L, args.kind, r, res, flush=True)


def cmd_summary(_args):
    rows = []
    rep = pd.read_csv(OUTS / "v2b_annotation_fdr/summary.csv").set_index("layer")
    for L in range(12):
        R = json.loads((OUT / f"real/layer_{L:02d}.json").read_text())
        row = {"layer": L, "n_alive": R["n_alive"]}
        for k in ["A_deployed", "A2_biosae", "B_perfeature", "B_ge3", "B2_perfeature_ovl2", "C_global",
                  "Bu_perfeature", "Cu_global"]:
            row[k] = R[k]
        row["deployed_file_n"] = R["deployed_summary"]["n_annotated"]
        row["rows_missing"] = R["rows_missing"]; row["rows_extra"] = R["rows_extra"]
        row["B_wilson_lo"], row["B_wilson_hi"] = R["B_wilson"]
        row["B_rows_share_overlap2"] = R.get("B_rows_share_overlap2")
        row["B_rows_overlap2_full20"] = R.get("B_rows_overlap2_full20")
        row["B_rows_overlap2_short"] = R.get("B_rows_overlap2_short")
        for kind in ["pps", "perm", "uniform"]:
            p = OUT / f"null/layer_{L:02d}_{kind}.json"
            if not p.exists():
                continue
            reps = json.loads(p.read_text())["reps"]
            row[f"{kind}_reps"] = len(reps)
            for k in ["A_deployed", "A2_biosae", "B_perfeature", "B_ge3", "B2_perfeature_ovl2", "C_global"]:
                v = np.array([reps[r][k] for r in reps], dtype=float)
                row[f"{kind}_{k}_mean"] = v.mean()
                row[f"{kind}_{k}_min"] = v.min(); row[f"{kind}_{k}_max"] = v.max()
                row[f"{kind}_{k}_sd"] = v.std(ddof=1) if len(v) > 1 else np.nan
                row[f"{kind}_{k}_n_ge_real"] = int((v >= R[k]).sum())
                row[f"{kind}_{k}_n_le_real"] = int((v <= R[k]).sum())
                sd = row[f"{kind}_{k}_sd"]
                row[f"{kind}_{k}_z"] = (R[k] - v.mean()) / sd if sd and sd > 0 else np.nan
            row[f"{kind}_count_corr_min"] = min(reps[r]["count_corr_with_real"] for r in reps)
            row[f"{kind}_slots_same_mean"] = float(np.mean([reps[r]["slots_same_as_real"] for r in reps]))
        # repair numbers for side-by-side
        for k_rep, k_me in [("A_n", "A_deployed"), ("A2_n", "A2_biosae"), ("B_n", "B_perfeature"),
                            ("B_ge3", "B_ge3"), ("C_n", "C_global"), ("Bu_n", "Bu_perfeature"), ("Cu_n", "Cu_global")]:
            row[f"repair_{k_rep}"] = int(rep.loc[L, k_rep])
            row[f"match_{k_me}"] = bool(int(rep.loc[L, k_rep]) == R[k_me])
        row["repair_null_freq_B_n_mean"] = rep.loc[L, "null_freq_B_n_mean"]
        row["repair_null_freq_A_mean_rate"] = rep.loc[L, "null_freq_A_mean"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary.csv", index=False)
    ex = {}
    if "pps_A_deployed_mean" in df:
        a = df["A_deployed"] / df["n_alive"]; b = df["pps_A_deployed_mean"] / df["n_alive"]
        ex["pearson_deployed_real_vs_pps_all12"] = float(np.corrcoef(a, b)[0, 1])
        ex["pearson_deployed_real_vs_pps_L1_L11"] = float(np.corrcoef(a[1:], b[1:])[0, 1])
        a = df["B_perfeature"] / df["n_alive"]; b = df["pps_B_perfeature_mean"] / df["n_alive"]
        ex["pearson_perfeature_real_vs_pps_all12"] = float(np.corrcoef(a, b)[0, 1])
        ex["pearson_perfeature_real_vs_pps_L1_L11"] = float(np.corrcoef(a[1:], b[1:])[0, 1])
        ex["global_total_real"] = int(df["C_global"].sum())
        ex["global_total_pps_mean"] = float(df["pps_C_global_mean"].sum())
        ex["chance_share_B_pps"] = (df["pps_B_perfeature_mean"] / df["B_perfeature"]).round(3).tolist()
        ex["deployed_real_minus_pps_points"] = ((df["A_deployed"] - df["pps_A_deployed_mean"]) / df["n_alive"] * 100).round(2).tolist()
        ex["biosae_real_minus_pps_points"] = ((df["A2_biosae"] - df["pps_A2_biosae_mean"]) / df["n_alive"] * 100).round(2).tolist()
    # repair's own correlation, recomputed without L0
    a = rep["A_rate"].to_numpy(); b = rep["null_freq_A_mean"].to_numpy()
    ex["repair_pearson_deployed_vs_freq_all12"] = float(np.corrcoef(a, b)[0, 1])
    ex["repair_pearson_deployed_vs_freq_L1_L11"] = float(np.corrcoef(a[1:], b[1:])[0, 1])
    a = rep["B_rate"].to_numpy(); b = rep["null_freq_B_mean"].to_numpy()
    ex["repair_pearson_perfeature_vs_freq_all12"] = float(np.corrcoef(a, b)[0, 1])
    ex["repair_pearson_perfeature_vs_freq_L1_L11"] = float(np.corrcoef(a[1:], b[1:])[0, 1])
    (OUT / "summary_extra.json").write_text(json.dumps(ex, indent=1))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 80)
    print(df.T.to_string())
    print(json.dumps(ex, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep")
    a = sub.add_parser("real"); a.add_argument("--layers", default="0-11")
    a = sub.add_parser("null"); a.add_argument("--layers", default="5"); a.add_argument("--reps", type=int, default=20)
    a.add_argument("--kind", choices=["pps", "perm", "uniform"], default="pps")
    sub.add_parser("summary")
    args = ap.parse_args()
    {"prep": cmd_prep, "real": cmd_real, "null": cmd_null, "summary": cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    main()
