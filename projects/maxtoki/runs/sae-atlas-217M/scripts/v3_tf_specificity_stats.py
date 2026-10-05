"""V3-2: TF specificity of v3 layer-5 SAE features on correctly encoded K562 cells (CPU only).

Same design and code as v2_tf_specificity_stats.py (statistics unchanged). Inputs now come from
v3_tf_specificity_extract.py (v3 SAE, counts/median encoding) and v3_tf_specificity_taskdata.py
(top-20 catalog rebuilt from the v3 SAE). Changes from the v2 script:
  * paths -> outputs/v3_tf_specificity;
  * the swap-null check uses the v3 GATA1 responding features (the deployed feature ids 628,
    1334, 2006, 2610, 2627 mean nothing for the v3 SAE) and is checked by Monte Carlo only
    (the investigation prototype numbers belong to the old catalog);
  * resumable stages so that every call stays under --max-minutes: fake nulls (per group size),
    observed statistics + rank (one cache), power (per TF), calibration (per TF). A call that runs
    out of time exits with code 3; run it again.
  * power and calibration use one seed per TF (default_rng([seed, TF index])) so the result does
    not depend on where a call stopped.

1. Responding features (unit = cell): per cell, mean layer-5 SAE activation over the gene tokens;
   two-sided Mann-Whitney U, knockdown cells vs 400 reference control cells, per feature; BH
   across the 4,928 features; responding = q < 0.05 and |mean_kd - mean_ref| > cut-off.
   Cut-offs 0.5 (deployed, primary), 0.25, 0.1, 0.05, 0.02, 0.01, 0.
2. Statistic: M = best overlap of a responding feature's top-20 genes with the TF's target set;
   T = min(M, 5) if M >= 2 else 0 (deployed thresholds 2..5 as one statistic); p = P_null(T >= T_obs).
   Also the uncapped M.
3. Nulls: rf (random catalog features, exact), cm (gene swap matched on detection count, exact),
   cml (count x gene-length tertile, exact), fake (fake TFs from 800 pool control cells, whole
   selection re-run, 2,000 per size), fakeK (fake TF keeps its K most-changed features), okd (other
   knockdowns' responding features).
4. Rank of the true TF among all TFs with a target set (ChIP >= 20, TRRUST >= 5) by p_cm.
5. Power: plant k = 1..8 true targets into the top-20 of one random responding feature, 200 repeats.
6. BH across TFs per database, cut-off and null (20 primary TFs; all 87).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as st  # noqa: E402
from scipy.special import gammaln  # noqa: E402

PROJ = Path("<REPO_ROOT>/projects/maxtoki")
sys.path.insert(0, str(PROJ / "setup"))
import hooks_v2 as H  # noqa: E402

RUN = PROJ / "runs/sae-atlas-217M/outputs"
OUT = RUN / "v3_tf_specificity"
TD = OUT / "task_data"
CACHE = OUT / "stats_cache"
D_SAE = 4928
CUTOFFS = [0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.0]
PRIMARY_CUTOFF = 0.5
Q_FEAT = 0.05
THR = [2, 3, 4, 5]
B_FAKE = 2000
R_POWER = 200
K_PLANT = list(range(1, 9))
SEED = 20261003
MIN_CAND = {"ChIP": 20, "TRRUST": 5}
DBS = ["ChIP", "TRRUST"]


# ----------------------------------------------------------------------------- helpers (as v2)
def bh(p):
    p = np.asarray(p, float); m = len(p)
    o = np.argsort(p, kind="mergesort")
    r = p[o] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty(m); out[o] = np.minimum(q, 1.0)
    return out


class RefMW:
    """Two-sided Mann-Whitney U (asymptotic, continuity, tie correction for tied zeros) of a group
    vs a fixed reference; same formula as scipy (checked against scipy in the script)."""

    def __init__(self, R):
        self.n2 = R.shape[0]
        self.Rs = np.sort(R, axis=0)
        self.zero_ref = (R == 0).sum(0)
        self.mean_ref = R.mean(0)

    def contrib(self, A):
        out = np.empty(A.shape, np.float64)
        for f in range(A.shape[1]):
            lo = np.searchsorted(self.Rs[:, f], A[:, f], "left")
            hi = np.searchsorted(self.Rs[:, f], A[:, f], "right")
            out[:, f] = lo + 0.5 * (hi - lo)
        return out

    def test(self, U1, n1, zeros_a):
        n2 = self.n2; n = n1 + n2
        mu = n1 * n2 / 2.0
        t0 = (zeros_a + self.zero_ref).astype(np.float64)
        tie = (t0 ** 3 - t0) / (n * (n - 1))
        var = n1 * n2 / 12.0 * ((n + 1) - tie)
        U = np.maximum(U1, n1 * n2 - U1)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (U - mu - 0.5) / np.sqrt(var)
        p = np.clip(2 * st.norm.sf(z), 0, 1)
        p[~(var > 0)] = 1.0
        return p


class TieIndex:
    """Non-zero per-cell values that occur in >= 2 cells of the whole cell table, per feature.

    Needed in v3: with the correct encoding many cells start with the same gene(s), and a feature
    that fires only there gets the SAME per-cell mean in many cells (e.g. feature 3153: one value in
    2,812 of 9,200 cells). v2's fast test corrected the variance for tied zeros only. A tie inside any
    subset of cells is a value that occurs >= 2 times in the whole table, so this index covers every
    possible tie group of every test run here."""

    def __init__(self, ALL):
        pf, pv = [], []
        for f in range(ALL.shape[1]):
            v = ALL[:, f]; v = v[v != 0]
            u, c = np.unique(v, return_counts=True)
            m = c > 1
            pf.append(np.full(int(m.sum()), f, np.int64)); pv.append(u[m])
        self.pf = np.concatenate(pf); self.pv = np.concatenate(pv).astype(ALL.dtype)
        self.off = np.searchsorted(self.pf, np.arange(ALL.shape[1] + 1))
        self.feats = np.unique(self.pf)
        self.n_pairs = len(self.pf)

    def indicator(self, A):
        """(n, F) -> (n, n_pairs) bool: cell has that tied value for that feature."""
        out = np.zeros((A.shape[0], self.n_pairs), bool)
        for f in self.feats:
            a, b = self.off[f], self.off[f + 1]
            out[:, a:b] = A[:, f][:, None] == self.pv[a:b][None, :]
        return out

    def tie_sum(self, counts):
        """sum over tie groups of (t^3 - t), per feature, from the count of each tied value."""
        t = np.asarray(counts, np.float64)
        return np.bincount(self.pf, weights=t ** 3 - t, minlength=D_SAE)


class RefMWT(RefMW):
    """RefMW with the full tie correction (zeros AND tied non-zero values), as scipy."""

    def __init__(self, R, ties: TieIndex):
        super().__init__(R)
        self.ties = ties
        self.cnt_ref = ties.indicator(R).sum(0)

    def test_full(self, U1, n1, zeros_a, cnt_a):
        n2 = self.n2; n = n1 + n2
        mu = n1 * n2 / 2.0
        t0 = (zeros_a + self.zero_ref).astype(np.float64)
        tie = ((t0 ** 3 - t0) + self.ties.tie_sum(self.cnt_ref + cnt_a)) / (n * (n - 1))
        var = n1 * n2 / 12.0 * ((n + 1) - tie)
        U = np.maximum(U1, n1 * n2 - U1)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (U - mu - 0.5) / np.sqrt(var)
        p = np.clip(2 * st.norm.sf(z), 0, 1)
        p[~(var > 0)] = 1.0
        return p

    def test_group(self, A):
        return self.test_full(self.contrib(A).sum(0), len(A), (A == 0).sum(0), self.ties.indicator(A).sum(0))


def cap_T(M):
    M = np.asarray(M)
    return np.where(M >= 2, np.minimum(M, 5), 0)


def lchoose(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def p_randfeat(T, K, ov_all):
    if T < 2 or K == 0:
        return 1.0
    N = len(ov_all); m = int((ov_all >= T).sum())
    if K > N - m:
        return 1.0
    return float(1 - math.exp(lchoose(N - m, K) - lchoose(N, K)))


class SwapNull:
    """Exact gene-swap null for one binning: overlap of feature f with set S is a sum over bins of
    Hypergeom(N_b, S_b, n_fb)."""

    def __init__(self, key, top_idx):
        self.key = key
        self.bins = np.unique(key)
        self.bpos = {b: i for i, b in enumerate(self.bins)}
        self.Nb = np.array([(key == b).sum() for b in self.bins])
        self.comp = np.zeros((D_SAE, len(self.bins)), np.int64)
        for f, g in enumerate(top_idx):
            for b in key[g]:
                self.comp[f, self.bpos[b]] += 1

    def tables(self, smask):
        Sb = np.array([smask[self.key == b].sum() for b in self.bins])
        tab = np.zeros((len(self.bins), 21, 21))
        k = np.arange(21)
        for i in range(len(self.bins)):
            for n in range(0, min(20, self.Nb[i]) + 1):
                tab[i, n] = st.hypergeom.pmf(k, self.Nb[i], Sb[i], n)
        return tab

    @staticmethod
    def _conv(dist, pmf):
        new = np.zeros_like(dist)
        for j in range(21):
            new[:, j:] += dist[:, j:j + 1] * pmf[:, :21 - j]
        return new

    def pmf_all(self, smask, tab=None):
        tab = self.tables(smask) if tab is None else tab
        dist = np.zeros((D_SAE, 21)); dist[:, 0] = 1.0
        for i in range(len(self.bins)):
            n = self.comp[:, i]
            if n.max() == 0:
                continue
            dist = self._conv(dist, tab[i, n])
        return dist

    def pmf_one(self, gene_idx, tab):
        dist = np.zeros((1, 21)); dist[0, 0] = 1.0
        cnt = defaultdict(int)
        for b in self.key[gene_idx]:
            cnt[self.bpos[b]] += 1
        for i, n in cnt.items():
            dist = self._conv(dist, tab[i, [n]])
        return dist[0]


def log_cdf_lt(pmf):
    cdf = np.concatenate([np.zeros((pmf.shape[0], 1)), np.cumsum(pmf, 1)], 1)
    return np.log(np.clip(cdf, 1e-300, 1.0))


def p_swap(T, lsum_lt):
    if T < 2:
        return 1.0
    return float(-np.expm1(lsum_lt[T]))


def p_emp(Tobs, Tnull):
    if Tobs < 2:
        return 1.0
    return float((1 + (Tnull >= Tobs).sum()) / (1 + len(Tnull)))


# ----------------------------------------------------------------------------- data
def load_cells(out_dir=OUT):
    man = pd.read_csv(out_dir / "cell_manifest.csv", keep_default_na=False)
    meta = json.load(open(out_dir / "cell_manifest_meta.json"))
    prog = json.load(open(out_dir / "extract_progress.json"))
    rows, MG, NT, ZE = [], [], [], []
    for fpath in sorted(glob.glob(str(out_dir / "cells/summ_*.npz"))):
        z = np.load(fpath)
        rows.append(z["row"]); MG.append(z["mean_gene"]); NT.append(z["n_tokens"]); ZE.append(z["z_eos"].astype(np.float32))
    rows = np.concatenate(rows); MG = np.concatenate(MG); NT = np.concatenate(NT); ZE = np.concatenate(ZE)
    assert len(np.unique(rows)) == len(rows), "duplicate cells in summaries"
    return man, meta, prog, rows, MG, NT, ZE


def load_task(td=TD):
    uni = pd.read_csv(td / "gene_universe.tsv", sep="\t", keep_default_na=False)
    genes = uni.gene.values; gidx = {g: i for i, g in enumerate(genes)}; NG = len(genes)
    top = pd.read_csv(td / "feature_top20.tsv", sep="\t", keep_default_na=False)
    top_idx = [np.array([], int)] * D_SAE
    has_special = np.zeros(D_SAE, bool)
    for f, g in top.groupby("feature_id"):
        gl = list(g.sort_values("rank").gene)
        has_special[f] = "<SPECIAL>" in gl
        top_idx[f] = np.array([gidx[x] for x in gl if x != "<SPECIAL>"], int)
    trr = pd.read_csv(td / "targets_trrust.tsv", sep="\t", keep_default_na=False)
    dor = pd.read_csv(td / "targets_dorothea.tsv", sep="\t", keep_default_na=False)
    dor["chip_flag"] = dor.chip_flag.astype(str).str.lower() == "true"
    sets = {"TRRUST": {t: np.isin(np.arange(NG), [gidx[x] for x in g.target]) for t, g in trr.groupby("tf")},
            "ChIP": {t: np.isin(np.arange(NG), [gidx[x] for x in g.target]) for t, g in dor[dor.chip_flag].groupby("tf")}}
    TOP = np.zeros((D_SAE, NG), np.int8)
    for f in range(D_SAE):
        TOP[f, top_idx[f]] = 1
    return uni, genes, gidx, NG, top_idx, has_special, sets, TOP


def select_all(R, KD, tfs, ties: TieIndex | None = None):
    """ties=None: v2 test (tie correction for zeros only; used by the v2 replay).
    ties given: full tie correction (v3)."""
    mw = RefMW(R) if ties is None else RefMWT(R, ties)
    sel = {}
    for tf in tfs:
        A = KD[tf]
        if ties is None:
            p = mw.test(mw.contrib(A).sum(0), len(A), (A == 0).sum(0))
        else:
            p = mw.test_group(A)
        q = bh(p)
        dlt = A.mean(0) - mw.mean_ref
        sd = np.sqrt((A.var(0, ddof=1) + R.var(0, ddof=1)) / 2)
        d = np.where(sd > 0, dlt / np.where(sd > 0, sd, 1), 0.0)
        sel[tf] = {"p": p, "q": q, "delta": dlt, "d": d, "n": len(A)}
    return mw, sel


def responding(s, c):
    return np.where((s["q"] < Q_FEAT) & (np.abs(s["delta"]) > c))[0]


def out_of_time(t_start, max_min):
    return (time.time() - t_start) / 60 > max_min


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b-fake", type=int, default=B_FAKE)
    ap.add_argument("--r-power", type=int, default=R_POWER)
    ap.add_argument("--b-cal", type=int, default=250)
    ap.add_argument("--max-minutes", type=float, default=7.0)
    args = ap.parse_args()
    t_start = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    logp = CACHE / "log.json"
    log = json.loads(logp.read_text()) if logp.exists() else {}
    log.setdefault("calls", []).append({"start": time.strftime("%Y-%m-%dT%H:%M:%S")})

    man, meta, prog, rows, MG, NT, ZE = load_cells()
    rix = {int(r): i for i, r in enumerate(rows)}
    z_bos = np.load(OUT / "cells/z_bos.npy")
    failed = {r for r, _ in prog["failed"]}

    def mat(rr):
        return MG[[rix[int(r)] for r in rr]]

    ref_rows = man[man.group == "ref"].row.values
    pool_rows = man[man.group == "pool"].row.values
    R = mat(ref_rows); P = mat(pool_rows)
    tfs_all = meta["tfs_all"]; primary = meta["tfs_primary"]
    KD, tf_rows, incomplete = {}, {}, []
    for tf in tfs_all:
        sub = man[(man.group == "kd") & (man.tf == tf)]
        ok = sub.row.isin(rix) | sub.row.isin(failed)
        if not ok.all():
            incomplete.append(tf); continue
        rr = sub[sub.row.isin(rix)].row.values
        KD[tf] = mat(rr); tf_rows[tf] = rr
    tfs = [t for t in tfs_all if t in KD]
    log.update({"n_ref": len(ref_rows), "n_pool": len(pool_rows), "tfs_complete": len(tfs),
                "tfs_incomplete": incomplete, "n_failed_tokenize": len(failed)})
    print(f"cells: ref {len(R)}, pool {len(P)}, TFs complete {len(tfs)}/{len(tfs_all)}", flush=True)
    if incomplete:
        raise SystemExit(f"extraction incomplete for {incomplete}")

    ties = TieIndex(MG)
    if "nonzero_tied_values_total" not in log:
        allm = np.concatenate([R, P] + [KD[t] for t in tfs])
        dup = 0
        for f in range(D_SAE):
            v = allm[:, f]; v = v[v != 0]
            dup += len(v) - len(np.unique(v))
        log["nonzero_tied_values_total"] = int(dup)
        log["n_exact_zero_cell_feature_values"] = int((allm == 0).sum())
        log["tie_index"] = {"n_feature_value_pairs": int(ties.n_pairs), "n_features_with_nonzero_ties": int(len(ties.feats))}
        del allm

    uni, genes, gidx, NG, top_idx, has_special, sets, TOP = load_task()

    def overlaps(smask):
        return TOP[:, smask].sum(1).astype(np.int64)

    swap = {"cm": SwapNull(uni.count_bin.values.astype(int), top_idx),
            "cml": SwapNull(uni.count_x_length_bin.values.astype(int), top_idx)}

    # ---------------- selection ----------------
    mw, sel = select_all(R, KD, tfs, ties)
    if "fast_mw_check" not in log:
        from statsmodels.stats.multitest import multipletests
        chk = []
        for name, A in [("GATA1", KD.get("GATA1")),
                        ("fake95", P[np.random.default_rng(SEED + 1).choice(len(P), 95, replace=False)])]:
            p_fast = mw.test_group(A)
            p_zero_only = mw.test(mw.contrib(A).sum(0), len(A), (A == 0).sum(0))
            p_sc = np.nan_to_num(st.mannwhitneyu(A.astype(np.float64), R.astype(np.float64), axis=0).pvalue, nan=1.0)
            q_fast = bh(p_fast); q_sm = multipletests(p_sc, method="fdr_bh")[1]
            chk.append({"group": name, "max_abs_dp": float(np.abs(p_fast - p_sc).max()),
                        "max_abs_dp_v2_zero_ties_only": float(np.abs(p_zero_only - p_sc).max()),
                        "max_abs_dq": float(np.abs(q_fast - q_sm).max()),
                        "same_bh_set": bool(((q_fast < Q_FEAT) == (q_sm < Q_FEAT)).all())})
        log["fast_mw_check"] = chk
        print("fast MW check", chk, flush=True)
        assert all(c["max_abs_dp"] < 1e-5 and c["same_bh_set"] for c in chk), "fast MW disagrees with scipy"
    resp_sets = {(tf, c): responding(sel[tf], c) for tf in tfs for c in CUTOFFS}
    Kreal = {k: len(v) for k, v in resp_sets.items()}
    g05 = [int(f) for f in resp_sets[("GATA1", PRIMARY_CUTOFF)]]
    log["GATA1_responding_cut0.5"] = g05

    # token-pooled effect (deployed unit: all positions incl. <bos>/<eos>) vs cell-mean effect, GATA1
    if "GATA1_token_pooled_vs_cell_mean" not in log:
        def mean_all(rr):
            out = []
            for r in rr:
                i = rix[int(r)]; T = NT[i]
                out.append(((T - 2) * MG[i].astype(np.float64) + z_bos + ZE[i]) / T)
            return np.array(out)
        dall = mean_all(tf_rows["GATA1"]).mean(0) - mean_all(ref_rows).mean(0)
        log["GATA1_token_pooled_vs_cell_mean"] = {int(f): {"token_pooled": float(dall[f]),
                                                           "cell_mean": float(sel["GATA1"]["delta"][f])} for f in g05}

    tf_set = {(db, tf): sets[db].get(tf) for db in DBS for tf in tfs}
    ov_cache, lcdf_cache, tab_cache = {}, {}, {}

    def get_ov(db, tf):
        if (db, tf) not in ov_cache:
            ov_cache[(db, tf)] = overlaps(sets[db][tf])
        return ov_cache[(db, tf)]

    def get_lcdf(nk, db, tf):
        k = (nk, db, tf)
        if k not in lcdf_cache:
            tab = swap[nk].tables(sets[db][tf]); tab_cache[k] = tab
            lcdf_cache[k] = log_cdf_lt(swap[nk].pmf_all(sets[db][tf], tab))
        return lcdf_cache[k]

    # ---------------- check the exact swap null by Monte Carlo (v3 GATA1 features) ----------------
    if "swap_null_validation" not in log and "GATA1" in sets["ChIP"]:
        rmc = np.random.default_rng(SEED + 3)
        feats = g05 if g05 else list(np.argsort(sel["GATA1"]["q"])[:5])
        S = sets["ChIP"]["GATA1"]
        vv = {"features": [int(f) for f in feats]}
        for nk in ["cm", "cml"]:
            pm = swap[nk].pmf_all(S)
            lsum = log_cdf_lt(pm)[feats].sum(0)
            ex = {"E_overlap": [float((pm[f] * np.arange(21)).sum()) for f in feats],
                  "P_max_ge": {t: float(-np.expm1(lsum[t])) for t in range(1, 9)}}
            key = swap[nk].key
            pools = {b: np.where(key == b)[0] for b in np.unique(key)}
            NS = 10000; tot = np.zeros((len(feats), NS), int)
            for j, f in enumerate(feats):
                bc = defaultdict(int)
                for b in key[top_idx[f]]:
                    bc[b] += 1
                for b, nb in bc.items():
                    Pb = pools[b]
                    for s0 in range(0, NS, 1000):
                        s1 = min(NS, s0 + 1000)
                        ix = np.argsort(rmc.random((s1 - s0, len(Pb))), axis=1)[:, :nb]
                        tot[j, s0:s1] += S[Pb[ix]].sum(1)
            mx = tot.max(0)
            mc = {"E_overlap": [float(x) for x in tot.mean(1)],
                  "P_max_ge": {t: float((mx >= t).mean()) for t in range(1, 9)}, "n_sim": NS}
            gaps_p = [abs(ex["P_max_ge"][t] - mc["P_max_ge"][t]) for t in range(1, 9)]
            se = [math.sqrt(max(ex["P_max_ge"][t] * (1 - ex["P_max_ge"][t]), 1e-12) / NS) for t in range(1, 9)]
            vv[nk] = {"exact": ex, "MC": mc, "max_abs_gap_P": float(max(gaps_p)),
                      "max_gap_in_MC_SE": float(max(g / s for g, s in zip(gaps_p, se))),
                      "max_abs_gap_E": float(np.max(np.abs(np.array(ex["E_overlap"]) - np.array(mc["E_overlap"]))))}
        log["swap_null_validation"] = vv
        print("swap null validation (max gaps)", {nk: (vv[nk]["max_abs_gap_P"], vv[nk]["max_gap_in_MC_SE"]) for nk in ["cm", "cml"]}, flush=True)
    H.write_json(logp, log)

    # ---------------- fake TFs (selection-aware null), cached per group size ----------------
    cpool = mw.contrib(P)
    zpool = (P == 0)
    ipool = ties.indicator(P).astype(np.int32)
    sizes = sorted({sel[t]["n"] for t in tfs})
    by_size = defaultdict(list)
    for t in tfs:
        by_size[sel[t]["n"]].append(t)
    fakeM, fake_nresp, fake_feat_p = {}, {}, {}
    fixed_feats = np.random.default_rng(SEED + 5).choice(D_SAE, 5, replace=False)
    FDIR = OUT / "fake_null"; FDIR.mkdir(exist_ok=True)
    for n in sizes:
        tlist = by_size[n]
        fmeta = {"B": args.b_fake, "seed": SEED + 1000 + n, "size": n, "tfs": tlist,
                 "K": {f"{t}|{c}": Kreal[(t, c)] for t in tlist for c in CUTOFFS},
                 "fixed_feats": [int(x) for x in fixed_feats], "cutoffs": CUTOFFS, "q": Q_FEAT,
                 "sae": "v3 layer 5", "mw": "full tie correction"}
        cpath = FDIR / f"size_{n:03d}.npz"
        if cpath.exists():
            zc = np.load(cpath, allow_pickle=False)
            if json.loads(str(zc["meta"])) == fmeta:
                for t in tlist:
                    for c in CUTOFFS:
                        for db in DBS:
                            for v in ["fake", "fakeK"]:
                                fakeM[(t, c, db, v)] = zc[f"M|{t}|{c}|{db}|{v}"]
                for c in CUTOFFS:
                    fake_nresp[(n, c)] = zc[f"nresp|{c}"]
                fake_feat_p[n] = zc["fp"]
                continue
        if out_of_time(t_start, args.max_minutes):
            print(f"STOPPED (max minutes) before fake size {n}; run again", flush=True)
            sys.exit(3)
        Mf = {(t, c, db, v): np.zeros(args.b_fake, np.int16) for t in tlist for c in CUTOFFS for db in DBS
              for v in ["fake", "fakeK"]}
        nresp = {c: np.zeros(args.b_fake, np.int32) for c in CUTOFFS}
        fp = np.zeros((args.b_fake, 5))
        rs = np.random.default_rng(SEED + 1000 + n)
        for b in range(args.b_fake):
            ix = rs.choice(len(P), n, replace=False)
            p = mw.test_full(cpool[ix].sum(0), n, zpool[ix].sum(0), ipool[ix].sum(0)); q = bh(p)
            dl = np.abs(P[ix].mean(0) - mw.mean_ref)
            fp[b] = p[fixed_feats]
            ord_d = np.argsort(-dl, kind="stable")
            ord_p = np.lexsort((-dl, p))
            for c in CUTOFFS:
                Rb = np.where((q < Q_FEAT) & (dl > c))[0]
                nresp[c][b] = len(Rb)
                for t in tlist:
                    K = Kreal[(t, c)]
                    RK = (ord_d if c > 0 else ord_p)[:K]
                    for db in DBS:
                        if tf_set[(db, t)] is None:
                            continue
                        ov = get_ov(db, t)
                        if len(Rb):
                            Mf[(t, c, db, "fake")][b] = ov[Rb].max()
                        if K:
                            Mf[(t, c, db, "fakeK")][b] = ov[RK].max()
        fakeM.update(Mf)
        for c in CUTOFFS:
            fake_nresp[(n, c)] = nresp[c]
        fake_feat_p[n] = fp
        arrs = {f"M|{t}|{c}|{db}|{v}": Mf[(t, c, db, v)] for (t, c, db, v) in Mf}
        arrs.update({f"nresp|{c}": nresp[c] for c in CUTOFFS})
        tmp = FDIR / f"size_{n:03d}.tmp.npz"
        np.savez(tmp, meta=np.array(json.dumps(fmeta)), fp=fp, **arrs)
        tmp.replace(cpath)
        print(f"  fake size {n}: done ({time.time() - t_start:.0f}s)", flush=True)

    # ---------------- observed statistics, nulls, rank (one cache) ----------------
    rpath = CACHE / "res_observed.pkl"
    if rpath.exists():
        res = pickle.load(open(rpath, "rb"))
    else:
        res = []
        for tf in tfs:
            for c in CUTOFFS:
                Rr = resp_sets[(tf, c)]
                for db in DBS:
                    S = tf_set[(db, tf)]
                    row = {"tf": tf, "primary_tf": tf in primary, "cutoff": c, "db": db, "n_cells": sel[tf]["n"],
                           "K": len(Rr), "n_targets_in_universe": int(S.sum()) if S is not None else 0}
                    if S is None or S.sum() == 0:
                        row["testable"] = False; res.append(row); continue
                    ov = get_ov(db, tf)
                    M = int(ov[Rr].max()) if len(Rr) else 0
                    T = int(cap_T(M))
                    row.update({"M_obs": M, "T_obs": T, "testable": len(Rr) > 0,
                                "best_feature": int(Rr[np.argmax(ov[Rr])]) if len(Rr) else -1,
                                "n_feat_ge2": int((ov[Rr] >= 2).sum()) if len(Rr) else 0,
                                "p_rf": p_randfeat(T, len(Rr), ov)})
                    for nk in ["cm", "cml"]:
                        lc = get_lcdf(nk, db, tf)
                        lsum = lc[Rr].sum(0) if len(Rr) else np.zeros(22)
                        row[f"p_{nk}"] = p_swap(T, lsum)
                        row[f"p_{nk}_uncapped"] = p_swap(M, lsum) if M >= 2 else 1.0
                        row[f"E_M_{nk}"] = float(sum(-np.expm1(lsum[t]) for t in range(1, 21))) if len(Rr) else 0.0
                    res.append(row)
        res = pd.DataFrame(res)
        print(f"observed stats done ({time.time() - t_start:.0f}s)", flush=True)
        for i, r in res.iterrows():
            if not r.get("testable", False) or r.n_targets_in_universe == 0:
                continue
            t, c, db = r.tf, r.cutoff, r.db
            T = int(r.T_obs); M = int(r.M_obs)
            for v in ["fake", "fakeK"]:
                Mn = fakeM[(t, c, db, v)]
                res.loc[i, f"p_{v}"] = p_emp(T, cap_T(Mn))
                res.loc[i, f"p_{v}_uncapped"] = p_emp(M, Mn)
            ov = get_ov(db, t)
            others = [y for y in tfs if y != t and len(resp_sets[(y, c)]) > 0]
            Ty = np.array([cap_T(ov[resp_sets[(y, c)]].max()) for y in others], int)
            res.loc[i, "n_okd"] = len(others)
            res.loc[i, "okd_min_attainable_p"] = 1.0 / (1 + len(others))
            res.loc[i, "p_okd"] = p_emp(T, Ty) if len(others) else np.nan
        for col in ["p_fake", "p_fakeK", "p_okd", "p_cm", "p_cml", "p_rf"]:
            if col not in res:
                res[col] = np.nan
        res.loc[(res.n_targets_in_universe > 0) & (res.K == 0), ["p_fake", "p_cm", "p_cml", "p_rf"]] = 1.0
        # rank of the true TF among all TFs with a target set
        cand = {db: [y for y, s in sets[db].items() if s.sum() >= MIN_CAND[db]] for db in DBS}
        lcand = {}
        for db in DBS:
            for y in cand[db]:
                lcand[(db, y)] = log_cdf_lt(swap["cm"].pmf_all(sets[db][y]))[:, :7]
                if (db, y) not in ov_cache:
                    ov_cache[(db, y)] = overlaps(sets[db][y])
        for i, r in res.iterrows():
            if not r.get("testable", False):
                continue
            t, c, db = r.tf, r.cutoff, r.db
            Rr = resp_sets[(t, c)]
            ys = list(cand[db]) + ([t] if t not in cand[db] else [])
            pv = {}
            for y in ys:
                ov = ov_cache.get((db, y))
                if ov is None:
                    ov = ov_cache[(db, y)] = overlaps(sets[db][y])
                T = int(cap_T(ov[Rr].max()))
                lsum = lcand[(db, y)][Rr].sum(0) if (db, y) in lcand else get_lcdf("cm", db, y)[Rr].sum(0)
                pv[y] = p_swap(T, lsum)
            px = pv[t]; arr = np.array([pv[y] for y in ys if y != t])
            res.loc[i, "rank_cm"] = 1 + (arr < px - 1e-12).sum() + 0.5 * (np.abs(arr - px) <= 1e-12).sum()
            res.loc[i, "n_rank_cand"] = len(ys)
            res.loc[i, "rank_frac"] = res.loc[i, "rank_cm"] / len(ys)
            res.loc[i, "n_cand_p_lt_0.05"] = int((arr < 0.05).sum())
        log["n_rank_candidates"] = {db: len(cand[db]) for db in DBS}
        for scope, tset in [("primary", primary), ("all", tfs)]:
            for c in CUTOFFS:
                for db in DBS:
                    m = res.tf.isin(tset) & (res.cutoff == c) & (res.db == db) & (res.n_targets_in_universe > 0)
                    for col in ["p_fake", "p_cm", "p_cml", "p_okd", "p_fakeK"]:
                        v = res.loc[m, col].values.astype(float)
                        ok = np.isfinite(v)
                        q = np.full(len(v), np.nan)
                        if ok.any():
                            q[ok] = bh(v[ok])
                        res.loc[m, f"q_{col[2:]}_{scope}"] = q
        pickle.dump(res, open(rpath, "wb"))
        log["t_observed_rank_s"] = round(time.time() - t_start, 1)
        H.write_json(logp, log)
        print(f"rank + BH done ({time.time() - t_start:.0f}s)", flush=True)

    # ---------------- power by planting (cached per TF) ----------------
    PDIR = CACHE / "power"; PDIR.mkdir(exist_ok=True)
    for ti, t in enumerate(primary):
        pp = PDIR / f"{t}.pkl"
        if pp.exists() or t not in KD:
            continue
        if out_of_time(t_start, args.max_minutes):
            print(f"STOPPED (max minutes) before power {t}; run again", flush=True)
            H.write_json(logp, log); sys.exit(3)
        rp = np.random.default_rng([SEED + 7, ti])
        prow = []
        for c in CUTOFFS:
            Rr = resp_sets[(t, c)]
            if len(Rr) == 0:
                continue
            K = len(Rr)
            for db in DBS:
                S = tf_set[(db, t)]
                if S is None or S.sum() == 0:
                    continue
                ov = get_ov(db, t)
                lc = {nk: get_lcdf(nk, db, t) for nk in ["cm", "cml"]}
                lsum = {nk: lc[nk][Rr].sum(0) for nk in lc}
                tabs = {nk: tab_cache[(nk, db, t)] for nk in lc}
                Tn = {v: cap_T(fakeM[(t, c, db, v)]) for v in ["fake", "fakeK"]}
                others = [y for y in tfs if y != t and len(resp_sets[(y, c)]) > 0]
                Ty = np.array([cap_T(ov[resp_sets[(y, c)]].max()) for y in others], int)
                targets = np.where(S)[0]
                memo = {}
                for k in K_PLANT:
                    det = defaultdict(list); keff = []
                    for _ in range(args.r_power):
                        j = rp.integers(K); f = Rr[j]
                        g = top_idx[f]
                        nontarget_slots = [s for s in range(len(g)) if not S[g[s]]] + (["special"] if has_special[f] else [])
                        avail = np.setdiff1d(targets, g)
                        ke = min(k, len(nontarget_slots), len(avail))
                        keff.append(ke)
                        slots = rp.choice(len(nontarget_slots), ke, replace=False) if ke else []
                        new_t = rp.choice(avail, ke, replace=False) if ke else []
                        g2 = list(g); add = []
                        for sidx, nt in zip(slots, new_t):
                            s = nontarget_slots[sidx]
                            if s == "special":
                                add.append(nt)
                            else:
                                g2[s] = nt
                        g2 = np.array(g2 + add, int)
                        of = int(S[g2].sum())
                        M_others = int(np.delete(ov[Rr], j).max()) if K > 1 else 0
                        M2 = max(M_others, of); T2 = int(cap_T(M2))
                        det["rf"].append(p_randfeat(T2, K, ov) < 0.05)
                        for nk in ["cm", "cml"]:
                            key = (nk, tuple(sorted(swap[nk].key[g2])))
                            if key not in memo:
                                memo[key] = log_cdf_lt(swap[nk].pmf_one(g2, tabs[nk])[None, :])[0]
                            ls = lsum[nk] - lc[nk][f] + memo[key]
                            det[nk].append(p_swap(T2, ls) < 0.05)
                            det[nk + "_uncapped"].append((p_swap(M2, ls) if M2 >= 2 else 1.0) < 0.05)
                        for v in ["fake", "fakeK"]:
                            det[v].append(p_emp(T2, Tn[v]) < 0.05)
                        if len(others):
                            det["okd"].append(p_emp(T2, Ty) < 0.05)
                    row = {"tf": t, "cutoff": c, "db": db, "K": K, "k": k, "mean_k_planted": float(np.mean(keff)),
                           "n_targets_in_universe": int(S.sum())}
                    for kk, vv in det.items():
                        row[f"power_{kk}"] = float(np.mean(vv))
                    prow.append(row)
        pickle.dump(prow, open(pp, "wb"))
        print(f"  power {t} done ({time.time() - t_start:.0f}s)", flush=True)
    pw = pd.DataFrame([r for t in primary if (PDIR / f"{t}.pkl").exists() for r in pickle.load(open(PDIR / f"{t}.pkl", "rb"))])

    # ---------------- calibration: new fake TFs scored as if real (cached per TF) ----------------
    CDIR = CACHE / "calib"; CDIR.mkdir(exist_ok=True)
    cal_tfs = [t for t in primary if t in KD]
    for ti, t in enumerate(cal_tfs):
        cp_ = CDIR / f"{t}.pkl"
        if cp_.exists():
            continue
        if out_of_time(t_start, args.max_minutes):
            print(f"STOPPED (max minutes) before calibration {t}; run again", flush=True)
            H.write_json(logp, log); sys.exit(3)
        rc = np.random.default_rng([SEED + 555, ti])
        n = sel[t]["n"]
        recs = {"prand": defaultdict(list), "pcons": defaultdict(list), "nresp": defaultdict(list)}
        for b in range(args.b_cal):
            ix = rc.choice(len(P), n, replace=False)
            p = mw.test_full(cpool[ix].sum(0), n, zpool[ix].sum(0), ipool[ix].sum(0)); q = bh(p)
            dl = np.abs(P[ix].mean(0) - mw.mean_ref)
            ord_d = np.argsort(-dl, kind="stable"); ord_p = np.lexsort((-dl, p))
            for c in CUTOFFS:
                Rb = np.where((q < Q_FEAT) & (dl > c))[0]
                recs["nresp"][c].append(len(Rb))
                K = Kreal[(t, c)]
                for db in DBS:
                    if tf_set[(db, t)] is None or tf_set[(db, t)].sum() == 0:
                        continue
                    ov = get_ov(db, t)
                    for v, RR in [("fake", Rb), ("fakeK", (ord_d if c > 0 else ord_p)[:K])]:
                        if v == "fakeK" and K == 0:
                            continue
                        T = int(cap_T(ov[RR].max())) if len(RR) else 0
                        Tn = cap_T(fakeM[(t, c, db, v)])
                        gt = int((Tn > T).sum()); eq = int((Tn == T).sum())
                        recs["prand"][(c, db, v)].append((gt + rc.random() * (eq + 1)) / (len(Tn) + 1))
                        recs["pcons"][(c, db, v)].append(p_emp(T, Tn))
        pickle.dump({k: dict(v) for k, v in recs.items()}, open(cp_, "wb"))
        print(f"  calibration {t} done ({time.time() - t_start:.0f}s)", flush=True)
    prand, pcons, nresp_obs = defaultdict(list), defaultdict(list), defaultdict(list)
    for t in cal_tfs:
        d = pickle.load(open(CDIR / f"{t}.pkl", "rb"))
        for k, v in d["prand"].items():
            prand[k].extend(v)
        for k, v in d["pcons"].items():
            pcons[k].extend(v)
        for k, v in d["nresp"].items():
            nresp_obs[k].extend(v)
    cal = {"B_cal_per_tf": args.b_cal, "seed": f"default_rng([{SEED + 555}, TF index in primary list])"}
    for c in CUTOFFS:
        nr = np.array(nresp_obs[c])
        cal[f"cutoff_{c}"] = {"frac_fake_TFs_with_any_responding_feature": float((nr > 0).mean()),
                              "mean_n_responding_fake": float(nr.mean()), "n_fake_TFs": int(len(nr))}
        for db in DBS:
            for v in ["fake", "fakeK"]:
                pr_ = np.array(prand[(c, db, v)]); pc_ = np.array(pcons[(c, db, v)])
                if len(pr_) == 0:
                    continue
                ks = st.kstest(pr_, "uniform")
                cal[f"cutoff_{c}"][f"{db}_{v}"] = {"n": int(len(pr_)), "KS_D_randomized": float(ks.statistic),
                                                   "KS_p_randomized": float(ks.pvalue),
                                                   "frac_p_conservative_le_0.05": float((pc_ <= 0.05).mean()),
                                                   "frac_p_randomized_le_0.05": float((pr_ <= 0.05).mean())}
    ref_tf = "GATA1"
    fpp = fake_feat_p[sel[ref_tf]["n"]]
    cal["feature_level_MW_p_fake_TFs"] = {
        "size": int(sel[ref_tf]["n"]), "features": [int(x) for x in fixed_feats],
        "KS_p_per_feature": [float(st.kstest(fpp[:, j], "uniform").pvalue) for j in range(5)],
        "frac_p_lt_0.05_per_feature": [float((fpp[:, j] < 0.05).mean()) for j in range(5)]}
    cal["reference_tf_for_calibration"] = ref_tf

    # ---------------- knockdown efficiency (expm1(X) = CP10k, as v2; not a model input) ----------------
    import h5py
    kdeff = {}
    with h5py.File(meta["h5_path"], "r") as fh:
        var = [s.decode() if isinstance(s, bytes) else s for s in fh["var"]["gene_name_index"][:]]
        vup = {s.upper(): i for i, s in enumerate(var)}
        X = fh["X"]
        refX = np.stack([X[int(r), :] for r in ref_rows])
        for t in tfs:
            if t not in vup:
                kdeff[t] = None; continue
            col = vup[t]
            kx = np.array([X[int(r), col] for r in tf_rows[t]])
            rx = refX[:, col]
            kdeff[t] = {"mean_log_expr_kd": float(kx.mean()), "mean_log_expr_ref": float(rx.mean()),
                        "remaining_frac_linear": float(np.expm1(kx).mean() / max(np.expm1(rx).mean(), 1e-12)),
                        "frac_zero_kd": float((kx == 0).mean()), "frac_zero_ref": float((rx == 0).mean())}
    del refX

    # ---------------- old TRRUST >= 2 test on the v3 selection ----------------
    p8 = pd.read_csv(RUN / "phase8_true/perturbation_response.csv")
    tf48 = [t.upper() for t in p8[p8.is_tf].target]
    old = []
    for c in CUTOFFS:
        sub = res[(res.cutoff == c) & (res.db == "TRRUST")]
        for scope, tset in [("deployed48", tf48), ("primary20", primary), ("all87", tfs)]:
            s2 = sub[sub.tf.isin(tset)]
            old.append({"cutoff": c, "scope": scope, "n_tfs_in_scope_run": int(len(s2)),
                        "n_with_trrust_targets": int((s2.n_targets_in_universe > 0).sum()),
                        "n_with_responding": int((s2.K > 0).sum()),
                        "n_specific_old_rule_ge2": int(((s2.M_obs >= 2) & (s2.K > 0)).sum()),
                        "n_specific_and_p_rf_lt_0.05": int(((s2.M_obs >= 2) & (s2.p_rf < 0.05)).sum()),
                        "tfs_specific_old_rule": ";".join(sorted(s2[(s2.M_obs >= 2) & (s2.K > 0)].tf))})
    old = pd.DataFrame(old)

    # ---------------- write ----------------
    res.to_csv(OUT / "tf_results.csv", index=False)
    pw.to_csv(OUT / "power.csv", index=False)
    old.to_csv(OUT / "old_trrust_test.csv", index=False)
    H.write_json(OUT / "calibration.json", cal)
    H.write_json(OUT / "knockdown_efficiency.json", kdeff)
    fr = []
    for t in tfs:
        s = sel[t]
        for f in np.where(s["q"] < Q_FEAT)[0]:
            fr.append({"tf": t, "feature_id": int(f), "delta_mean": float(s["delta"][f]), "cohen_d": float(s["d"][f]),
                       "mwu_p": float(s["p"][f]), "bh_q": float(s["q"][f]), "n_kd_cells": s["n"],
                       **{f"resp_cut_{c}": bool(abs(s["delta"][f]) > c) for c in CUTOFFS}})
    pd.DataFrame(fr).to_csv(TD / "responding_features.tsv", sep="\t", index=False)
    summ_rows = []
    for t in tfs:
        r = {"tf": t, "primary": t in primary, "n_kd_cells": sel[t]["n"], "n_bh_sig": int((sel[t]["q"] < Q_FEAT).sum()),
             **{f"K_cut_{c}": len(resp_sets[(t, c)]) for c in CUTOFFS},
             "n_trrust_in_universe": int(tf_set[("TRRUST", t)].sum()) if tf_set[("TRRUST", t)] is not None else 0,
             "n_chip_in_universe": int(tf_set[("ChIP", t)].sum()) if tf_set[("ChIP", t)] is not None else 0}
        if kdeff.get(t):
            r["kd_remaining_frac_linear"] = kdeff[t]["remaining_frac_linear"]
        summ_rows.append(r)
    pd.DataFrame(summ_rows).to_csv(TD / "tf_summary.tsv", sep="\t", index=False)
    res.to_csv(TD / "tf_tests.tsv", sep="\t", index=False)
    np.savez_compressed(TD / "cell_feature_means.npz", rows=rows, mean_gene=MG.astype(np.float32), n_tokens=NT)

    log["calls"][-1]["wall_s"] = round(time.time() - t_start, 1)
    cfg = H.env_info("cpu")
    cfg.update({"script": __file__, "script_sha256": H.sha256_file(__file__),
                "extract_script_sha256": H.sha256_file(Path(__file__).parent / "v3_tf_specificity_extract.py"),
                "seeds": {"stats": SEED, "mw_check_fake": SEED + 1, "swap_mc": SEED + 3, "fixed_feats": SEED + 5,
                          "fake_per_size": f"{SEED}+1000+n", "power": f"default_rng([{SEED + 7}, TF index in primary])",
                          "calibration": f"default_rng([{SEED + 555}, TF index in primary])"},
                "B_fake": args.b_fake, "R_power": args.r_power, "B_cal_per_tf": args.b_cal, "cutoffs": CUTOFFS,
                "primary_cutoff": PRIMARY_CUTOFF, "q_feature": Q_FEAT, "thresholds": THR,
                "min_candidate_targets": MIN_CAND,
                "cell_ids": "cell_manifest.csv (dataset row + barcode), dataset " + meta["h5_path"],
                "feature_ids": "all 4,928 features of outputs/v3_sae/layer_05/sae_final.pt", "log": log})
    H.write_json(OUT / "run_config_stats.json", cfg)
    H.write_json(logp, log)
    print(f"done in {time.time() - t_start:.0f}s; DONE", flush=True)


if __name__ == "__main__":
    main()
