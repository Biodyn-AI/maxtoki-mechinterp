"""D3 (revision): TF specificity of layer-5 SAE features, redone with the fixed selection.

CPU only. Reads the per-cell summaries from v2_tf_specificity_extract.py and the task data
from v2_tf_specificity_taskdata.py.

1. Responding-feature selection (unit = cell)
   per cell: mean layer-5 SAE activation over the cell's gene tokens (input of block 5);
   two-sided Mann-Whitney U, knockdown cells vs 400 reference control cells, per feature;
   BH across the 4,928 features; responding = q < 0.05 and |mean_kd - mean_ref| > cut-off.
   Deployed cut-off 0.5 is the primary; sensitivity cut-offs 0.25, 0.1, 0.05, 0.02, 0.01, 0.
2. Deployed statistic: overlap of a responding feature's deployed top-20 genes with the TF's
   target set; "specific at threshold t" if any responding feature has overlap >= t, t in 2..5.
   With nested thresholds the threshold search reduces to T = min(M, 5) if M >= 2 else 0,
   M = best overlap over responding features. Every p-value below is P_null(T >= T_obs), which
   is the min-p over the four thresholds calibrated under that null. (Secondary: the uncapped M.)
3. Nulls
   rf    deployed random-feature null: K random catalog features (exact combinatorics)
   cm    gene-swap null, each top-20 gene swapped for a random gene with the same detection
         count bin (without replacement inside a bin within a feature), exact (hypergeometric
         convolution), features independent
   cml   same, bins = detection count x gene-length tertile
   fake  selection-aware null: fake TFs = random subsets of held-out control cells (pool, 800)
         of the knockdown group's size; the WHOLE selection is re-run against the same 400
         reference cells; B = 2,000 per size
   fakeK as fake, but the fake TF keeps its K most-changed features (K = the real TF's number
         of responding features; ordered by |delta| for cut-off > 0, by p for cut-off 0)
   okd   other-knockdown null: responding features of the other knockdowns (87 TFs, same rule,
         only those with >= 1 responding feature), scored against this TF's target set
4. Specificity: rank of the true TF among all TFs with target sets (ChIP >= 20 in-universe
   targets; TRRUST >= 5), by the cm p-value computed with each TF's set, same features.
5. Power: plant k = 1..8 true targets of the TF into the top-20 of a random responding feature
   (replacing random non-targets), 200 repeats, detection = p < 0.05 for each test.
6. BH across TFs on the fake (selection-aware) p-values, per database and cut-off.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st
from scipy.special import gammaln

PROJ = Path("<REPO_ROOT>/projects/maxtoki")
sys.path.insert(0, str(PROJ / "setup"))
import hooks_v2 as H  # noqa: E402

RUN = PROJ / "runs/sae-atlas-217M/outputs"
OUT = RUN / "v2_tf_specificity"
TD = OUT / "task_data"
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


# ----------------------------------------------------------------------------- helpers
def bh(p):
    p = np.asarray(p, float); m = len(p)
    o = np.argsort(p, kind="mergesort")
    r = p[o] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty(m); out[o] = np.minimum(q, 1.0)
    return out


class RefMW:
    """Two-sided Mann-Whitney U (asymptotic, continuity, tie correction) of a group vs a fixed
    reference, written so that many groups can be tested fast. Same formula as scipy 1.17
    (checked in the script). Ties: only exact zeros are tied in these data (checked)."""

    def __init__(self, R):
        self.n2 = R.shape[0]
        self.Rs = np.sort(R, axis=0)
        self.zero_ref = (R == 0).sum(0)
        self.mean_ref = R.mean(0)

    def contrib(self, A):
        """(n1, F): for each cell, #ref < x + 0.5 #ref == x, per feature."""
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


def cap_T(M):
    M = np.asarray(M)
    return np.where(M >= 2, np.minimum(M, 5), 0)


def lchoose(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def p_randfeat(T, K, ov_all):
    """Deployed random-feature null with threshold search: P(max over K random catalog features
    of overlap >= T). Exact. T = 0 -> 1."""
    if T < 2 or K == 0:
        return 1.0
    N = len(ov_all); m = int((ov_all >= T).sum())
    if K > N - m:
        return 1.0
    return float(1 - math.exp(lchoose(N - m, K) - lchoose(N, K)))


class SwapNull:
    """Exact gene-swap null for one binning. For a target set S, feature f's overlap is the sum
    over bins b of Hypergeom(N_b, S_b, n_fb) (n_fb = number of f's top-20 genes in bin b)."""

    def __init__(self, key, top_idx):
        self.key = key                                  # bin per universe gene
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
    """log P(O < t) for t = 0..21 (column t)."""
    cdf = np.concatenate([np.zeros((pmf.shape[0], 1)), np.cumsum(pmf, 1)], 1)
    return np.log(np.clip(cdf, 1e-300, 1.0))


def p_swap(T, lsum_lt):
    """P(M_null >= T) from sum over responding features of log P(O_f < t)."""
    if T < 2:
        return 1.0
    return float(-np.expm1(lsum_lt[T]))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b-fake", type=int, default=B_FAKE)
    ap.add_argument("--r-power", type=int, default=R_POWER)
    ap.add_argument("--power-tfs", default="primary", choices=["primary", "all"])
    ap.add_argument("--b-cal", type=int, default=250)
    ap.add_argument("--max-minutes", type=float, default=7.5)
    ap.add_argument("--only-fake", action="store_true")
    args = ap.parse_args()
    t_start = time.time()
    # every random step has its own seed (SEED + offset) so that skipping a step never changes another
    log = {}

    # ---------------- cells ----------------
    man = pd.read_csv(OUT / "cell_manifest.csv", keep_default_na=False)
    meta = json.load(open(OUT / "cell_manifest_meta.json"))
    prog = json.load(open(OUT / "extract_progress.json"))
    failed = {r for r, _ in prog["failed"]}
    rows, MG, NT = [], [], []
    for fpath in sorted(glob.glob(str(OUT / "cells/summ_*.npz"))):
        z = np.load(fpath)
        rows.append(z["row"]); MG.append(z["mean_gene"]); NT.append(z["n_tokens"])
    rows = np.concatenate(rows); MG = np.concatenate(MG); NT = np.concatenate(NT)
    assert len(np.unique(rows)) == len(rows), "duplicate cells in summaries"
    rix = {int(r): i for i, r in enumerate(rows)}
    z_bos = np.load(OUT / "cells/z_bos.npy")
    zeos = {}
    for fpath in sorted(glob.glob(str(OUT / "cells/summ_*.npz"))):
        z = np.load(fpath)
        for r, e in zip(z["row"], z["z_eos"]):
            zeos[int(r)] = e
    man["done"] = man.row.isin(rix)

    def mat(rr):
        return MG[[rix[int(r)] for r in rr]]

    ref_rows = man[man.group == "ref"].row.values
    pool_rows = man[man.group == "pool"].row.values
    assert man[man.group.isin(["ref", "pool"])].done.all()
    R = mat(ref_rows); P = mat(pool_rows)
    tfs_all = meta["tfs_all"]; primary = meta["tfs_primary"]
    KD, tf_rows, incomplete = {}, {}, []
    for tf in tfs_all:
        sub = man[(man.group == "kd") & (man.tf == tf)]
        ok = sub.done | sub.row.isin(failed)
        if not ok.all():
            incomplete.append(tf); continue
        rr = sub[sub.done].row.values
        KD[tf] = mat(rr); tf_rows[tf] = rr
    tfs = [t for t in tfs_all if t in KD]
    log["n_ref"] = len(ref_rows); log["n_pool"] = len(pool_rows)
    log["tfs_complete"] = len(tfs); log["tfs_incomplete"] = incomplete
    log["n_failed_tokenize"] = len(failed)
    print(f"cells: ref {len(R)}, pool {len(P)}, TFs complete {len(tfs)}/{len(tfs_all)}", flush=True)

    # ties other than zeros? (needed for the fast MW)
    allm = np.concatenate([R, P] + [KD[t] for t in tfs])
    dup = 0
    for f in range(D_SAE):
        v = allm[:, f]; v = v[v != 0]
        dup += len(v) - len(np.unique(v))
    log["nonzero_tied_values_total"] = int(dup)
    del allm

    # ---------------- task data ----------------
    uni = pd.read_csv(TD / "gene_universe.tsv", sep="\t", keep_default_na=False)
    genes = uni.gene.values; gidx = {g: i for i, g in enumerate(genes)}; NG = len(genes)
    top = pd.read_csv(TD / "feature_top20.tsv", sep="\t", keep_default_na=False)
    top_idx = [np.array([], int)] * D_SAE
    has_special = np.zeros(D_SAE, bool)
    for f, g in top.groupby("feature_id"):
        gl = list(g.sort_values("rank").gene)
        has_special[f] = "<SPECIAL>" in gl
        top_idx[f] = np.array([gidx[x] for x in gl if x != "<SPECIAL>"], int)
    n_slots = np.array([len(top_idx[f]) + has_special[f] for f in range(D_SAE)])
    trr = pd.read_csv(TD / "targets_trrust.tsv", sep="\t", keep_default_na=False)
    dor = pd.read_csv(TD / "targets_dorothea.tsv", sep="\t", keep_default_na=False)
    dor["chip_flag"] = dor.chip_flag.astype(str).str.lower() == "true"
    sets = {"TRRUST": {t: np.isin(np.arange(NG), [gidx[x] for x in g.target]) for t, g in trr.groupby("tf")},
            "ChIP": {t: np.isin(np.arange(NG), [gidx[x] for x in g.target]) for t, g in dor[dor.chip_flag].groupby("tf")}}
    TOP = np.zeros((D_SAE, NG), np.int8)
    for f in range(D_SAE):
        TOP[f, top_idx[f]] = 1

    def overlaps(smask):
        return TOP[:, smask].sum(1).astype(np.int64)

    cnt_key = uni.count_bin.values.astype(int)
    cxl_key = uni.count_x_length_bin.values.astype(int)
    swap = {"cm": SwapNull(cnt_key, top_idx), "cml": SwapNull(cxl_key, top_idx)}

    # ---------------- selection ----------------
    mw = RefMW(R)
    # check fast MW + BH against scipy/statsmodels
    from statsmodels.stats.multitest import multipletests
    chk = []
    for name, A in [("GATA1", KD.get("GATA1")), ("fake95", P[np.random.default_rng(SEED + 1).choice(len(P), 95, replace=False)])]:
        if A is None:
            continue
        U1 = mw.contrib(A).sum(0)
        p_fast = mw.test(U1, len(A), (A == 0).sum(0))
        # scipy on float64 input (on float32 input scipy loses ~1e-7 in p through float32 arithmetic)
        p_sc = np.nan_to_num(st.mannwhitneyu(A.astype(np.float64), R.astype(np.float64), axis=0).pvalue, nan=1.0)
        q_fast = bh(p_fast); q_sm = multipletests(p_sc, method="fdr_bh")[1]
        chk.append({"group": name, "max_abs_dp": float(np.abs(p_fast - p_sc).max()),
                    "max_abs_dlog10p": float(np.abs(np.log10(np.maximum(p_fast, 1e-300)) - np.log10(np.maximum(p_sc, 1e-300))).max()),
                    "max_abs_dq": float(np.abs(q_fast - q_sm).max()),
                    "same_bh_set": bool(((q_fast < Q_FEAT) == (q_sm < Q_FEAT)).all())})
    log["fast_mw_check"] = chk
    print(f"MW check done ({time.time()-t_start:.0f}s)", flush=True)
    print("fast MW check", chk, flush=True)
    # tolerance: the fast test corrects only for tied zeros; 63 of 4,928 features also have a few
    # tied non-zero values (<= 8), which changes p in the 7th digit at most
    assert all(c["max_abs_dp"] < 1e-5 and c["same_bh_set"] for c in chk), "fast MW disagrees with scipy"

    sel = {}
    for tf in tfs:
        A = KD[tf]
        U1 = mw.contrib(A).sum(0)
        p = mw.test(U1, len(A), (A == 0).sum(0)); q = bh(p)
        dlt = A.mean(0) - mw.mean_ref
        sd = np.sqrt((A.var(0, ddof=1) + R.var(0, ddof=1)) / 2)
        d = np.where(sd > 0, dlt / np.where(sd > 0, sd, 1), 0.0)
        sel[tf] = {"p": p, "q": q, "delta": dlt, "d": d, "n": len(A)}

    def responding(s, c):
        return np.where((s["q"] < Q_FEAT) & (np.abs(s["delta"]) > c))[0]

    # deployed-style token-pooled effect (all positions incl. <bos>/<eos>) for GATA1, for comparison
    if "GATA1" in KD:
        def mean_all(rr):
            out = []
            for r in rr:
                i = rix[int(r)]; T = NT[i]
                out.append(((T - 2) * MG[i] + z_bos + zeos[int(r)].astype(np.float64)) / T)
            return np.array(out)
        dall = mean_all(tf_rows["GATA1"]).mean(0) - mean_all(ref_rows).mean(0)
        log["GATA1_token_pooled_delta_deployed_features"] = {int(f): float(dall[f]) for f in [628, 1334, 2006, 2610, 2627]}
        log["GATA1_cell_mean_delta_deployed_features"] = {int(f): float(sel["GATA1"]["delta"][f]) for f in [628, 1334, 2006, 2610, 2627]}
        log["GATA1_q_deployed_features"] = {int(f): float(sel["GATA1"]["q"][f]) for f in [628, 1334, 2006, 2610, 2627]}

    # ---------------- per-set caches ----------------
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

    # validation of the exact swap null: reproduce the investigation prototype (null3) for the
    # deployed GATA1 features, and a Monte Carlo check
    if "GATA1" in sets["ChIP"] and not args.only_fake:
        rmc = np.random.default_rng(SEED + 3)
        dep5 = [628, 1334, 2006, 2610, 2627]
        vv = {}
        for nk in ["cm", "cml"]:
            pm = swap[nk].pmf_all(sets["ChIP"]["GATA1"])
            l5 = log_cdf_lt(pm)[dep5].sum(0)
            vv[nk] = {"E_overlap_2610": float((pm[2610] * np.arange(21)).sum()),
                      "P_2610_ge8": float(pm[2610, 8:].sum()),
                      "P_max5_ge8": float(-np.expm1(l5[8])), "P_max5_ge5": float(-np.expm1(l5[5]))}
            # Monte Carlo (without replacement inside a bin within a feature)
            key = swap[nk].key; S = sets["ChIP"]["GATA1"]
            pools = {b: np.where(key == b)[0] for b in np.unique(key)}
            NS = 10000; tot = np.zeros((len(dep5), NS), int)
            for j, f in enumerate(dep5):
                bc = defaultdict(int)
                for b in key[top_idx[f]]:
                    bc[b] += 1
                for b, nb in bc.items():
                    Pb = pools[b]
                    for s0 in range(0, NS, 1000):          # chunks keep memory small
                        s1 = min(NS, s0 + 1000)
                        ix = np.argsort(rmc.random((s1 - s0, len(Pb))), axis=1)[:, :nb]
                        tot[j, s0:s1] += S[Pb[ix]].sum(1)
            mx = tot.max(0)
            vv[nk]["MC"] = {"E_overlap_2610": float(tot[3].mean()), "P_2610_ge8": float((tot[3] >= 8).mean()),
                            "P_max5_ge8": float((mx >= 8).mean()), "P_max5_ge5": float((mx >= 5).mean()), "n_sim": NS}
        vv["investigation_null3"] = {"cm": {"E_overlap_2610": 4.44, "P_2610_ge8": 0.038, "P_max5_ge5": 0.47},
                                     "cml": {"E_overlap_2610": 5.12, "P_2610_ge8": 0.072, "P_max5_ge5": 0.64}}
        log["swap_null_validation"] = vv
        print(f"validation done ({time.time()-t_start:.0f}s)", flush=True)
        print("swap null validation", json.dumps(vv), flush=True)

    # ---------------- observed statistics ----------------
    res = []
    resp_sets = {}
    for tf in tfs:
        for c in CUTOFFS:
            Rr = responding(sel[tf], c); resp_sets[(tf, c)] = Rr
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
                for nk in ([] if args.only_fake else ["cm", "cml"]):
                    lc = get_lcdf(nk, db, tf)
                    lsum = lc[Rr].sum(0) if len(Rr) else np.zeros(22)
                    row[f"p_{nk}"] = p_swap(T, lsum)
                    row[f"p_{nk}_uncapped"] = p_swap(M, lsum) if M >= 2 else 1.0
                    row[f"E_M_{nk}"] = float(sum(-np.expm1(lsum[t]) for t in range(1, 21))) if len(Rr) else 0.0
                res.append(row)
    res = pd.DataFrame(res)
    print(f"observed stats done ({time.time()-t_start:.0f}s)", flush=True)
    log["t_observed_s"] = round(time.time() - t_start, 1)

    # ---------------- selection-aware null: fake TFs ----------------
    cpool = mw.contrib(P)                       # (800, F)
    zpool = (P == 0)
    sizes = sorted({sel[t]["n"] for t in tfs})
    by_size = defaultdict(list)
    for t in tfs:
        by_size[sel[t]["n"]].append(t)
    Kreal = {(t, c): len(resp_sets[(t, c)]) for t in tfs for c in CUTOFFS}
    fakeM = {}       # (tf, c, db, variant) -> array(B)
    fake_nresp = {}  # (size, c) -> array(B) number of responding features
    fixed_feats = np.random.default_rng(SEED + 5).choice(D_SAE, 5, replace=False)
    fake_feat_p = {}
    FDIR = OUT / "fake_null"; FDIR.mkdir(exist_ok=True)
    for n in sizes:
        tlist = by_size[n]
        fmeta = {"B": args.b_fake, "seed": SEED + 1000 + n, "size": n, "tfs": tlist,
                 "K": {f"{t}|{c}": Kreal[(t, c)] for t in tlist for c in CUTOFFS},
                 "fixed_feats": [int(x) for x in fixed_feats], "cutoffs": CUTOFFS, "q": Q_FEAT}
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
        if (time.time() - t_start) / 60 > args.max_minutes:
            print(f"STOPPED (max minutes) before fake size {n}; re-run to continue", flush=True)
            sys.exit(3)
        Mf = {(t, c, db, v): np.zeros(args.b_fake, np.int16) for t in tlist for c in CUTOFFS for db in DBS
              for v in ["fake", "fakeK"]}
        nresp = {c: np.zeros(args.b_fake, np.int32) for c in CUTOFFS}
        fp = np.zeros((args.b_fake, 5))
        rs = np.random.default_rng(SEED + 1000 + n)
        for b in range(args.b_fake):
            ix = rs.choice(len(P), n, replace=False)
            p = mw.test(cpool[ix].sum(0), n, zpool[ix].sum(0)); q = bh(p)
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
        print(f"  fake size {n}: done ({time.time()-t_start:.0f}s)", flush=True)
    if args.only_fake:
        print("fake nulls complete; stopping (--only-fake)", flush=True)
        return

    def p_emp(Tobs, Tnull):
        if Tobs < 2:
            return 1.0
        return float((1 + (Tnull >= Tobs).sum()) / (1 + len(Tnull)))

    # other-knockdown null
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
    # untestable rows: selection-aware p = 1 (no responding feature -> T = 0)
    for col in ["p_fake", "p_fakeK", "p_okd", "p_cm", "p_cml", "p_rf"]:
        if col not in res:
            res[col] = np.nan
    res.loc[(res.n_targets_in_universe > 0) & (res.K == 0), ["p_fake", "p_cm", "p_cml", "p_rf"]] = 1.0

    # ---------------- specificity rank among TFs ----------------
    cand = {db: [y for y, s in sets[db].items() if s.sum() >= MIN_CAND[db]] for db in DBS}
    lcand = {}
    t_rank = time.time()
    for db in DBS:
        for y in cand[db]:
            lcand[(db, y)] = log_cdf_lt(swap["cm"].pmf_all(sets[db][y]))[:, :7]
            if (db, y) not in ov_cache:
                ov_cache[(db, y)] = overlaps(sets[db][y])
    print(f"rank caches: {sum(len(v) for v in cand.values())} sets ({time.time()-t_rank:.0f}s)", flush=True)
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
            if (db, y) in lcand:
                lsum = lcand[(db, y)][Rr].sum(0)
            else:
                lsum = get_lcdf("cm", db, y)[Rr].sum(0)
            pv[y] = p_swap(T, lsum)
        px = pv[t]; arr = np.array([pv[y] for y in ys if y != t])
        res.loc[i, "rank_cm"] = 1 + (arr < px - 1e-12).sum() + 0.5 * (np.abs(arr - px) <= 1e-12).sum()
        res.loc[i, "n_rank_cand"] = len(ys)
        res.loc[i, "rank_frac"] = res.loc[i, "rank_cm"] / len(ys)
        res.loc[i, "n_cand_p_lt_0.05"] = int((arr < 0.05).sum())
    print(f"rank done ({time.time()-t_start:.0f}s)", flush=True)

    # ---------------- BH across TFs ----------------
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

    # ---------------- power by planting ----------------
    ptfs = primary if args.power_tfs == "primary" else tfs
    ptfs = [t for t in ptfs if t in KD]
    prow = []
    rp = np.random.default_rng(SEED + 7)
    for t in ptfs:
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
    pw = pd.DataFrame(prow)
    log["t_power_done_s"] = round(time.time() - t_start, 1)
    print(f"power done ({time.time()-t_start:.0f}s)", flush=True)

    # ---------------- calibration (fake TFs as if they were real) ----------------
    # B_CAL new fake TFs per primary TF (independent seed, same size as that TF's knockdown group,
    # drawn from the pool) go through the same selection code as a real knockdown and are scored
    # with that TF's target sets against the B_FAKE null draws. Under the null their p-values
    # must be uniform (randomised for ties) and P(p <= 0.05) <= 0.05 (conservative p).
    cal = {"B_cal_per_tf": args.b_cal}
    ref_tf = "GATA1" if "GATA1" in KD else tfs[0]
    n_ref_tf = sel[ref_tf]["n"]
    rc = np.random.default_rng(SEED + 555)
    prand, pcons, nresp_obs = defaultdict(list), defaultdict(list), defaultdict(list)
    cal_tfs = [t for t in primary if t in KD]
    for t in cal_tfs:
        n = sel[t]["n"]
        for b in range(args.b_cal):
            ix = rc.choice(len(P), n, replace=False)
            p = mw.test(cpool[ix].sum(0), n, zpool[ix].sum(0)); q = bh(p)
            dl = np.abs(P[ix].mean(0) - mw.mean_ref)
            ord_d = np.argsort(-dl, kind="stable"); ord_p = np.lexsort((-dl, p))
            for c in CUTOFFS:
                Rb = np.where((q < Q_FEAT) & (dl > c))[0]
                nresp_obs[c].append(len(Rb))
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
                        prand[(c, db, v)].append((gt + rc.random() * (eq + 1)) / (len(Tn) + 1))
                        pcons[(c, db, v)].append(p_emp(T, Tn))
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
    fpp = fake_feat_p[n_ref_tf]
    cal["feature_level_MW_p_fake_TFs"] = {
        "size": int(n_ref_tf), "features": [int(x) for x in fixed_feats],
        "KS_p_per_feature": [float(st.kstest(fpp[:, j], "uniform").pvalue) for j in range(5)],
        "frac_p_lt_0.05_per_feature": [float((fpp[:, j] < 0.05).mean()) for j in range(5)]}
    cal["reference_tf_for_calibration"] = ref_tf

    # ---------------- knockdown efficiency ----------------
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

    # ---------------- old TRRUST >=2 test on corrected selection ----------------
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
                        "n_specific_and_p_rf_lt_0.05": int(((s2.M_obs >= 2) & (s2.p_rf < 0.05)).sum())})
    old = pd.DataFrame(old)

    # ---------------- write ----------------
    res.to_csv(OUT / "tf_results.csv", index=False)
    pw.to_csv(OUT / "power.csv", index=False)
    old.to_csv(OUT / "old_trrust_test.csv", index=False)
    H.write_json(OUT / "calibration.json", cal)
    H.write_json(OUT / "knockdown_efficiency.json", kdeff)
    # responding features (task data)
    fr = []
    for t in tfs:
        s = sel[t]
        keep = np.where(s["q"] < Q_FEAT)[0]
        for f in keep:
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

    log["wall_s"] = round(time.time() - t_start, 1)
    cfg = H.env_info("cpu")
    cfg.update({"script": __file__, "script_sha256": H.sha256_file(__file__),
                "extract_script_sha256": H.sha256_file(Path(__file__).parent / "v2_tf_specificity_extract.py"),
                "seeds": {"stats": SEED, "fake_per_size": f"{SEED}+1000+n", "power": SEED + 7, "calibration": SEED + 99},
                "B_fake": args.b_fake, "R_power": args.r_power, "B_cal_per_tf": args.b_cal, "cutoffs": CUTOFFS, "primary_cutoff": PRIMARY_CUTOFF,
                "q_feature": Q_FEAT, "thresholds": THR, "min_candidate_targets": MIN_CAND,
                "cell_ids": "cell_manifest.csv (dataset row + barcode), dataset " + meta["h5_path"],
                "feature_ids": "all 4,928 layer-5 SAE features", "log": log})
    H.write_json(OUT / "run_config_stats.json", cfg)
    print(f"done in {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
