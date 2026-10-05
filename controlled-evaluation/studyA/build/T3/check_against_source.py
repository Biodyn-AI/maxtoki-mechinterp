"""Check that keys/T3/reference.py reproduces the source report's per-TF tests (task_data/tf_tests.tsv, read only).
Runs the first part of reference.py (selection + observed statistics + exact nulls) on the package data and compares
K, M, T, p_rf, p_cm, p_cml (capped and uncapped) and E_M_cm for every TF x cut-off x database row.
Result on 2026-10-01: 546 of 546 rows match; largest absolute difference 2.2e-16."""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st
from scipy.special import gammaln

T0 = time.time()
PKG = Path("<EVAL_ROOT>/studyA/tasks/T3-paper")
D = PKG / "data"
OUT = None
NF = 4928
CUT = [0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.0]
DBS = ["ChIP", "TRRUST"]
COUNT_EDGES = [1, 2, 5, 10, 20, 50, 100, 200, 500]
MIN_CAND = {"ChIP": 20, "TRRUST": 5}
KEY = {}


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def bh(p):
    p = np.asarray(p, float); m = len(p)
    if m == 0:
        return p
    o = np.argsort(p, kind="mergesort")
    r = p[o] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty(m); out[o] = np.minimum(q, 1.0)
    return out


# ------------------------------------------------------------------ data
z = np.load(D / "cell_feature_means.npz")
rows, MG = z["rows"], z["mean_gene"]
man = pd.read_csv(D / "cell_manifest.csv", keep_default_na=False)
rix = {int(r): i for i, r in enumerate(rows)}


def mat(rr):
    return MG[[rix[int(r)] for r in rr]].astype(np.float64)


R = mat(man[man.group == "ref"].row.values)
P = mat(man[man.group == "pool"].row.values)
tfs = sorted(man[man.group == "kd"].tf.unique())
KD = {t: mat(man[(man.group == "kd") & (man.tf == t)].row.values) for t in tfs}
KEY["n_tfs"] = len(tfs)
KEY["n_cells"] = {g: int((man.group == g).sum()) for g in ["ref", "pool", "kd", "catalog"]}
KEY["gata1_n_kd_cells"] = len(KD["GATA1"])
KEY["median_over_features_of_mean_ref_value"] = float(np.median(R.mean(0)))

uni = pd.read_csv(D / "gene_universe.tsv", sep="\t", keep_default_na=False, na_values=[""])
genes = uni.gene.values; gidx = {g: i for i, g in enumerate(genes)}; NG = len(genes)
cnt = uni.detection_count.values.astype(int)
ln = uni.gene_length_bp.values.astype(float)
cbin = np.digitize(cnt, COUNT_EDGES)
lq = np.nanpercentile(ln, [33.3, 66.7])
lbin = np.where(np.isfinite(ln), np.digitize(np.nan_to_num(ln, nan=np.nanmedian(ln)), lq), 1)
cxl = cbin * 10 + lbin
KEY["length_tertile_cuts_bp"] = [float(x) for x in lq]

top = pd.read_csv(D / "feature_top20.tsv", sep="\t", keep_default_na=False)
top_idx = [np.array([], int)] * NF
for f, g in top.groupby("feature_id"):
    top_idx[f] = np.array([gidx[x] for x in g.sort_values("rank").gene if x != "<SPECIAL>"], int)
TOP = np.zeros((NF, NG), np.int8)
for f in range(NF):
    TOP[f, top_idx[f]] = 1
ntop = TOP.sum(1)

trr = pd.read_csv(D / "targets_trrust.tsv", sep="\t", keep_default_na=False)
dor = pd.read_csv(D / "targets_dorothea.tsv", sep="\t", keep_default_na=False)
dor["chip_flag"] = dor.chip_flag.astype(str).str.lower() == "true"


def gmask(targets):
    m = np.zeros(NG, bool); m[[gidx[x] for x in targets]] = True
    return m


sets = {"TRRUST": {t: gmask(g.target) for t, g in trr.groupby("tf")},
        "ChIP": {t: gmask(g.target) for t, g in dor[dor.chip_flag].groupby("tf")}}
S_G = sets["ChIP"]["GATA1"]
KEY["gata1_chip_targets_in_universe"] = int(S_G.sum())
KEY["gata1_trrust_targets_in_universe"] = int(sets["TRRUST"]["GATA1"].sum())
KEY["n_tfs_with_chip_set"] = int(sum(t in sets["ChIP"] for t in tfs))
KEY["n_tfs_with_trrust_set"] = int(sum(t in sets["TRRUST"] for t in tfs))
KEY["universe_median_detection_count"] = float(np.median(cnt))
KEY["gata1_chip_targets_median_detection_count"] = float(np.median(cnt[S_G]))
KEY["universe_median_length_bp"] = float(np.nanmedian(ln))
KEY["gata1_chip_targets_median_length_bp"] = float(np.nanmedian(ln[S_G]))
log("data loaded")

# ------------------------------------------------------------------ 1. selection
sel = {}
for t in tfs:
    A = KD[t]
    p = np.nan_to_num(st.mannwhitneyu(A, R, axis=0).pvalue, nan=1.0)
    sel[t] = {"p": p, "q": bh(p), "d": A.mean(0) - R.mean(0), "n": len(A)}


def resp(t, c):
    s = sel[t]
    return np.where((s["q"] < 0.05) & (np.abs(s["d"]) > c))[0]


KEY["n_tfs_with_responding_by_cutoff"] = {str(c): int(sum(len(resp(t, c)) > 0 for t in tfs)) for c in CUT}
KEY["gata1_K_by_cutoff"] = {str(c): int(len(resp("GATA1", c))) for c in CUT}
KEY["gata1_n_bh_sig_features"] = int((sel["GATA1"]["q"] < 0.05).sum())
KEY["gata1_responding_features_cut0.5"] = [int(x) for x in resp("GATA1", 0.5)]
KEY["tfs_with_responding_cut0.5"] = {t: int(len(resp(t, 0.5))) for t in tfs if len(resp(t, 0.5))}
log("selection done")

# ------------------------------------------------------------------ 2-3. statistic and nulls


def lchoose(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def capT(M):
    M = np.asarray(M)
    return np.where(M >= 2, np.minimum(M, 5), 0)


def p_rf(T, K, ov):
    """P(best of K random catalog features has overlap >= T); exact."""
    if T < 2 or K == 0:
        return 1.0
    N = len(ov); m = int((ov >= T).sum())
    if K > N - m:
        return 1.0
    return float(1 - math.exp(lchoose(N - m, K) - lchoose(N, K)))


class Swap:
    """Exact gene-swap null: feature f's overlap = sum over bins b of Hypergeom(N_b, S_b, n_fb)."""

    def __init__(self, key):
        self.key = key; self.bins = np.unique(key); self.bpos = {b: i for i, b in enumerate(self.bins)}
        self.Nb = np.array([(key == b).sum() for b in self.bins])
        self.comp = np.zeros((NF, len(self.bins)), np.int64)
        for f in range(NF):
            for b in key[top_idx[f]]:
                self.comp[f, self.bpos[b]] += 1

    def tables(self, S):
        Sb = np.array([S[self.key == b].sum() for b in self.bins]); k = np.arange(21)
        tab = np.zeros((len(self.bins), 21, 21))
        for i in range(len(self.bins)):
            for n in range(0, min(20, self.Nb[i]) + 1):
                tab[i, n] = st.hypergeom.pmf(k, self.Nb[i], Sb[i], n)
        return tab

    @staticmethod
    def conv(dist, pm):
        new = np.zeros_like(dist)
        for j in range(21):
            new[:, j:] += dist[:, j:j + 1] * pm[:, :21 - j]
        return new

    def pmf_all(self, S, tab=None):
        tab = self.tables(S) if tab is None else tab
        dist = np.zeros((NF, 21)); dist[:, 0] = 1
        for i in range(len(self.bins)):
            n = self.comp[:, i]
            if n.max():
                dist = self.conv(dist, tab[i, n])
        return dist

    def pmf_genes(self, gene_idx, tab):
        dist = np.zeros((1, 21)); dist[0, 0] = 1
        b, c = np.unique(self.key[gene_idx], return_counts=True)
        for bb, n in zip(b, c):
            dist = self.conv(dist, tab[self.bpos[bb], [n]])
        return dist[0]


def lcdf(pmf):
    c = np.concatenate([np.zeros((pmf.shape[0], 1)), np.cumsum(pmf, 1)], 1)
    return np.log(np.clip(c, 1e-300, 1.0))


def p_swap(T, lsum):
    return 1.0 if T < 2 else float(-np.expm1(lsum[T]))


SW = {"cm": Swap(cbin), "cml": Swap(cxl)}
LC, TAB = {}, {}


def get_lc(nk, db, t):
    k = (nk, db, t)
    if k not in LC:
        TAB[k] = SW[nk].tables(sets[db][t])
        pm = SW[nk].pmf_all(sets[db][t], TAB[k])
        LC[k] = (lcdf(pm), pm)
    return LC[k]


OV = {}


def get_ov(db, t):
    if (db, t) not in OV:
        OV[(db, t)] = TOP[:, sets[db][t]].sum(1).astype(int)
    return OV[(db, t)]


res = []
for t in tfs:
    for c in CUT:
        Rr = resp(t, c)
        for db in DBS:
            if t not in sets[db]:
                continue
            ov = get_ov(db, t)
            M = int(ov[Rr].max()) if len(Rr) else 0; T = int(capT(M))
            r = {"tf": t, "cutoff": c, "db": db, "K": len(Rr), "n_targets": int(sets[db][t].sum()), "M": M, "T": T,
                 "best_feature": int(Rr[np.argmax(ov[Rr])]) if len(Rr) else -1,
                 "p_rf": p_rf(T, len(Rr), ov), "p_rf_uncapped": p_rf(M, len(Rr), ov) if M >= 2 else 1.0}
            for nk in ["cm", "cml"]:
                lsum = get_lc(nk, db, t)[0][Rr].sum(0) if len(Rr) else np.zeros(22)
                r[f"p_{nk}"] = p_swap(T, lsum)
                r[f"p_{nk}_uncapped"] = p_swap(M, lsum) if M >= 2 else 1.0
                r[f"E_M_{nk}"] = float(sum(-np.expm1(lsum[k]) for k in range(1, 21))) if len(Rr) else 0.0
            res.append(r)
res = pd.DataFrame(res)
src = pd.read_csv("<REPO_ROOT>/projects/maxtoki/runs/sae-atlas-217M/outputs/v2_tf_specificity/task_data/tf_tests.tsv", sep="\t")
src = src[src.testable.astype(str) != "nan"]
m = res.merge(src, left_on=["tf", "cutoff", "db"], right_on=["tf", "cutoff", "db"], suffixes=("", "_src"))
print("rows ref", len(res), "src", len(src), "merged", len(m))
print("K equal", (m.K == m.K_src).mean(), "M equal", (m.M == m.M_obs).mean(), "T equal", (m["T"] == m.T_obs).mean())
for a, b in [("p_rf", "p_rf_src"), ("p_cm", "p_cm_src"), ("p_cml", "p_cml_src"), ("p_cm_uncapped", "p_cm_uncapped_src"), ("p_cml_uncapped", "p_cml_uncapped_src"), ("E_M_cm", "E_M_cm_src")]:
    d = (m[a] - m[b].astype(float)).abs()
    print(a, "max abs diff", float(d.max()), "n>1e-6", int((d > 1e-6).sum()))
bad = m[(m.p_cml - m.p_cml_src.astype(float)).abs() > 1e-6]
print(bad[["tf","cutoff","db","K","M","p_cml","p_cml_src"]].head(10))
