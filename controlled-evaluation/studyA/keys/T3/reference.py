"""Reference analysis for Study A task T3 (answer key; never copy into a task package).

Question: are the layer-5 SAE features that respond to a TF's knockdown specific to that TF's target genes?

Reads ONLY the package data (T3-paper/data; T3-contract/data is identical) and prints/saves every key number.
Run:  bin/python studyA/keys/T3/reference.py  [path/to/package]      (CPU, about 3 minutes, peak memory about 2 GB)

Method (follows the corrected analysis in the source report, V2_TF_SPECIFICITY_REPORT.md):
 1. Selection, unit = cell. Per TF: two-sided Mann-Whitney U per feature, knockdown cells vs the 400 'ref'
    control cells, on the per-cell feature value; BH across the 4,928 features; responding = q < 0.05 and
    |mean(kd) - mean(ref)| > cut-off. Cut-offs 0.5 (the source paper's value), 0.25, 0.1, 0.05, 0.02, 0.01, 0.
 2. Statistic: M = best overlap of a responding feature's top-20 genes with the TF's target set;
    T = min(M, 5) if M >= 2 else 0 (the source's thresholds 2..5 as one statistic). Also uncapped M.
    Target sets: 'ChIP' = DoRothEA edges with chip_flag; 'TRRUST' = TRRUST v2. Both restricted to the universe.
 3. Nulls: rf  = K random catalog features (exact);
           cm  = each top-20 gene swapped for a random universe gene in the same detection-count bin
                 (bins at 1,2,5,10,20,50,100,200,500; exact, hypergeometric convolution);
           cml = same with bins = count bin x gene-length tertile.
    Best-of-K: P(M_null >= T) = 1 - prod_f P(O_f < T).
 4. Specificity rank: same responding features, cm p-value computed with every TF set (ChIP: >= 20 targets in
    universe, 291 sets; TRRUST: >= 5, 109 sets); mid-rank of the true TF.
 5. BH across TFs (20 TFs with a ChIP set; all 87 TFs that have a set in that database), per cut-off.
 6. GATA1 details: effects with cell-bootstrap CIs; feature 2610; other-knockdown null; random control groups
    (selection re-run; the 'fake TF' null); K-matched random groups; planted-target power.
 7. Naive analyses (the traps): hypergeometric / Fisher tests against the whole universe; pooled union of top
    genes; GATA1 target enrichment in OTHER knockdowns' features; count-matched pooled test.
 8. Extra checks for the key: feature 2610's gene rarity and its overlap with all 291 ChIP sets; the count-matched
    null with coarse (decile) bins; a cut-off-free rank check (Spearman of |effect| with target overlap across
    features, true TF's set ranked among 291 sets, 20 TFs); Wilson CIs for the planted-target power; testable
    TFs per cut-off; the source paper's feature-level Fisher test read literally (responding vs other features x
    'top-20 overlaps the targets'), BH across TFs, and the rank of the true TF's set among all 291 ChIP sets for
    the same test (every setting), plus how rare the responding features' genes are.
"""
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
PKG = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "tasks" / "T3-paper"
D = PKG / "data"
OUT = Path(__file__).resolve().parent / "reference_output.json"
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
primary = [t for t in tfs if t in sets["ChIP"]]
log("observed statistics done")

# 5. BH across TFs
bhsum = {}
for c in CUT:
    for db in DBS:
        for scope, tset in [("chipTFs20", primary), ("all", tfs)]:
            s = res[(res.cutoff == c) & (res.db == db) & res.tf.isin(tset)]
            for col in ["p_cm", "p_cml", "p_cm_uncapped", "p_rf"]:
                q = bh(s[col].values)
                bhsum[f"{c}|{db}|{scope}|{col}"] = {"n_tfs": int(len(s)), "n_testable": int((s.K > 0).sum()),
                                                    "n_p_lt_0.05": int((s[col] < 0.05).sum()),
                                                    "n_q_lt_0.05": int((q < 0.05).sum()), "min_q": float(q.min())}
KEY["n_TF_settings_BH_significant_count_matched"] = int(sum(v["n_q_lt_0.05"] for k, v in bhsum.items() if k.endswith("|p_cm")))
KEY["n_TF_settings_BH_significant_count_length_matched"] = int(sum(v["n_q_lt_0.05"] for k, v in bhsum.items() if k.endswith("|p_cml")))
KEY["n_TF_settings_BH_significant_count_matched_uncapped"] = int(sum(v["n_q_lt_0.05"] for k, v in bhsum.items() if k.endswith("|p_cm_uncapped")))
KEY["bh_summary"] = bhsum
nom = res[(res.p_cm < 0.05) & (res.K > 0)]
KEY["nominal_p_cm_lt_0.05_rows"] = nom[["tf", "cutoff", "db", "K", "M", "p_cm", "p_cml", "p_rf"]].to_dict("records")
ce = res[(res.tf == "CEBPZ") & (res.db == "ChIP") & (res.cutoff == 0.01)].iloc[0]
s = res[(res.cutoff == 0.01) & (res.db == "ChIP") & res.tf.isin(primary)]
KEY["cebpz_chip_cut0.01"] = {"K": int(ce.K), "M": int(ce.M), "best_feature": int(ce.best_feature), "p_cm": float(ce.p_cm),
                             "q_cm_across_20_chip_TFs": float(bh(s.p_cm.values)[list(s.tf).index("CEBPZ")]),
                             "p_cml": float(ce.p_cml), "p_rf": float(ce.p_rf)}
g = res[(res.tf == "GATA1")].set_index(["db", "cutoff"])
KEY["gata1_tests"] = {f"{db}|{c}": {k: (float(v) if isinstance(v, (float, np.floating)) else int(v))
                                    for k, v in g.loc[(db, c)].items() if k not in ("tf",)} for db in DBS for c in CUT}
log("BH done")

# 4. specificity rank of the true TF among TF sets (cm p-value, same features)
cand = {db: [y for y, m in sets[db].items() if m.sum() >= MIN_CAND[db]] for db in DBS}
KEY["n_rank_candidates"] = {db: len(cand[db]) for db in DBS}


def rank_true(t, c, db):
    Rr = resp(t, c)
    if len(Rr) == 0:
        return None
    ys = list(cand[db]) + ([t] if t not in cand[db] else [])
    pv = {}
    for y in ys:
        T = int(capT(get_ov(db, y)[Rr].max()))
        pv[y] = p_swap(T, get_lc("cm", db, y)[0][Rr].sum(0))
    arr = np.array([pv[y] for y in ys if y != t]); px = pv[t]
    return {"rank": float(1 + (arr < px - 1e-12).sum() + 0.5 * (np.abs(arr - px) <= 1e-12).sum()),
            "n_candidates": len(ys), "p_cm_true": px, "n_other_sets_p_lt_0.05": int((arr < 0.05).sum())}


KEY["gata1_rank_cm"] = {f"ChIP|{c}": rank_true("GATA1", c, "ChIP") for c in CUT}
KEY["gata1_rank_cm"]["TRRUST|0.02"] = rank_true("GATA1", 0.02, "TRRUST")
KEY["cebpz_rank_cm_chip_0.01"] = rank_true("CEBPZ", 0.01, "ChIP")
log("rank done")

# 6. GATA1 details --------------------------------------------------------
A = KD["GATA1"]; rng = np.random.default_rng(20261001)
feats = sorted(set(resp("GATA1", 0.5)) | {2610})
nb = 1000
bd = np.zeros((nb, NF))
for b in range(nb):
    ia = rng.integers(0, len(A), len(A)); ir = rng.integers(0, len(R), len(R))
    bd[b] = A[ia].mean(0) - R[ir].mean(0)
qsig = sel["GATA1"]["q"] < 0.05
ovG = get_ov("ChIP", "GATA1")
KEY["gata1_effects_cut0.5"] = {
    int(f): {"delta": float(sel["GATA1"]["d"][f]), "ci95": [float(x) for x in np.percentile(bd[:, f], [2.5, 97.5])],
             "frac_boot_abs_delta_gt_0.5": float((np.abs(bd[:, f]) > 0.5).mean()), "q": float(sel["GATA1"]["q"][f]),
             "chip_overlap": int(ovG[f]), "trrust_overlap": int(get_ov("TRRUST", "GATA1")[f])} for f in feats}
bestM = np.array([ovG[np.where(qsig & (np.abs(bd[b]) > 0.5))[0]].max() for b in range(nb)])
KEY["gata1_boot_best_chip_overlap_cut0.5_counts"] = {int(k): int(v) for k, v in zip(*np.unique(bestM, return_counts=True))}
KEY["gata1_boot_note"] = "cell bootstrap (kd and ref resampled separately, 1000 resamples, seed 20261001); selection in a resample = q < 0.05 in the full data and |bootstrap delta| > 0.5"

# feature 2610: count-matched vs naive per-feature p
pm_cm = get_lc("cm", "ChIP", "GATA1")[1]; pm_cml = get_lc("cml", "ChIP", "GATA1")[1]
KEY["f2610_gata1_chip"] = {"overlap": int(ovG[2610]), "n_genes": int(ntop[2610]),
                           "E_overlap_count_matched": float((pm_cm[2610] * np.arange(21)).sum()),
                           "P_ge_8_count_matched": float(pm_cm[2610, 8:].sum()),
                           "P_ge_8_count_length_matched": float(pm_cml[2610, 8:].sum()),
                           "E_overlap_uniform_universe": float(ntop[2610] * S_G.sum() / NG),
                           "p_hypergeom_vs_universe": float(st.hypergeom.sf(ovG[2610] - 1, NG, int(S_G.sum()), int(ntop[2610]))),
                           "n_catalog_features_with_overlap_ge_8": int((ovG >= 8).sum())}
log("GATA1 bootstrap done")

# other-knockdown null for GATA1 (ChIP set), each cut-off
okd = {}
for c in CUT:
    Rg = resp("GATA1", c)
    if len(Rg) == 0:
        continue
    Tobs = int(capT(ovG[Rg].max()))
    others = [y for y in tfs if y != "GATA1" and len(resp(y, c))]
    Ty = np.array([int(capT(ovG[resp(y, c)].max())) for y in others])
    okd[str(c)] = {"T_obs": Tobs, "n_other_knockdowns": len(others),
                   "p_okd": float((1 + (Ty >= Tobs).sum()) / (1 + len(others))) if Tobs >= 2 else 1.0,
                   "n_others_T_ge_T_obs": int((Ty >= Tobs).sum())}
KEY["gata1_other_knockdown_null"] = okd

# random control groups of GATA1's size (pool vs ref), whole selection re-run ('fake TF' null)
rng2 = np.random.default_rng(20261002); B = 300; n = len(A)
nresp = {c: [] for c in CUT}; Tk = []
Kg = len(resp("GATA1", 0.5))
for b in range(B):
    ix = rng2.choice(len(P), n, replace=False); X = P[ix]
    p = np.nan_to_num(st.mannwhitneyu(X, R, axis=0).pvalue, nan=1.0); q = bh(p)
    dl = np.abs(X.mean(0) - R.mean(0))
    for c in CUT:
        nresp[c].append(int(((q < 0.05) & (dl > c)).sum()))
    topk = np.argsort(-dl, kind="stable")[:Kg]           # K-matched: keep the K most-changed features
    Tk.append(int(capT(ovG[topk].max())))
Tk = np.array(Tk); Tobs05 = int(capT(ovG[resp("GATA1", 0.5)].max()))
KEY["random_control_groups"] = {
    "B": B, "group_size": n,
    "frac_with_any_responding_feature": {str(c): float(np.mean(np.array(nresp[c]) > 0)) for c in CUT},
    "p_selection_aware_gata1_chip_cut0.5": float((1 + 0) / (1 + B)) if all(np.array(nresp[0.5]) == 0) else None,
    "p_K_matched_gata1_chip_cut0.5": float((1 + (Tk >= Tobs05).sum()) / (1 + B)), "T_obs": Tobs05,
    "K_matched_T_counts": {int(k): int(v) for k, v in zip(*np.unique(Tk, return_counts=True))}}
log("random control groups done")

# planted-target power, GATA1, cut-off 0.5, ChIP set, count-matched null (capped and uncapped)
rng3 = np.random.default_rng(20261003)
Rr = resp("GATA1", 0.5); lc_cm, _ = get_lc("cm", "ChIP", "GATA1"); tab = TAB[("cm", "ChIP", "GATA1")]
lsum0 = lc_cm[Rr].sum(0); targets = np.where(S_G)[0]; power = {}
for k in [2, 4, 8]:
    det = []; det_u = []
    for _ in range(200):
        j = rng3.integers(len(Rr)); f = Rr[j]; gl = top_idx[f].copy()
        slots = [i for i in range(len(gl)) if not S_G[gl[i]]]
        new = rng3.choice(np.setdiff1d(targets, gl), k, replace=False)
        for s_, nt in zip(rng3.choice(slots, k, replace=False), new):
            gl[s_] = nt
        of = int(S_G[gl].sum()); Mo = int(np.delete(ovG[Rr], j).max()) if len(Rr) > 1 else 0
        M2 = max(of, Mo); T2 = int(capT(M2))
        ls = lsum0 - lc_cm[f] + lcdf(SW["cm"].pmf_genes(gl, tab)[None, :])[0]
        det.append(p_swap(T2, ls) < 0.05); det_u.append((p_swap(M2, ls) if M2 >= 2 else 1.0) < 0.05)
    power[str(k)] = {"capped": float(np.mean(det)), "uncapped": float(np.mean(det_u))}
KEY["gata1_planted_power_cm_cut0.5"] = power
log("power done")

# 7. naive analyses (what the traps produce) --------------------------------
naive = {}
for c in CUT:
    for db in DBS:
        rows_ = []
        for t in tfs:
            if t not in sets[db]:
                continue
            Rr = resp(t, c); S = sets[db][t]; ns = int(S.sum())
            if len(Rr) == 0:
                rows_.append((t, 1.0, 1.0)); continue
            ov = get_ov(db, t)
            pf = st.hypergeom.sf(ov[Rr] - 1, NG, ns, ntop[Rr])            # per feature vs universe
            U = TOP[Rr].max(0).astype(bool)                                 # pooled union of top genes
            pp = st.hypergeom.sf(int((U & S).sum()) - 1, NG, ns, int(U.sum()))
            rows_.append((t, float(bh(pf).min()), float(pp)))
        arr = pd.DataFrame(rows_, columns=["tf", "minq_feature", "p_pooled"])
        naive[f"{c}|{db}"] = {"n_tfs": len(arr), "n_BH_across_TFs_feature_test": int((bh(arr.minq_feature.values) < 0.05).sum()),
                              "n_BH_across_TFs_pooled_test": int((bh(arr.p_pooled.values) < 0.05).sum()),
                              "tfs_pooled_BH": arr.tf[bh(arr.p_pooled.values) < 0.05].tolist()}
KEY["naive_universe_hypergeom"] = naive

cand_chip = cand["ChIP"]


def pooled(U, S, key=None):
    x = int((U & S).sum()); n = int(U.sum())
    if key is None:
        return x, n * S.sum() / NG, float(st.hypergeom.logsf(x - 1, NG, int(S.sum()), n))
    mu = var = 0.0
    for b in np.unique(key):
        inb = key == b; Nb = inb.sum(); Sb = (S & inb).sum(); nb_ = (U & inb).sum()
        if nb_ == 0:
            continue
        mu += nb_ * Sb / Nb
        if Nb > 1:
            var += nb_ * (Sb / Nb) * (1 - Sb / Nb) * (Nb - nb_) / (Nb - 1)
    return x, mu, float((x - mu) / np.sqrt(var)) if var > 0 else 0.0


gp = {}
for c in [0.25, 0.1, 0.05, 0.02]:
    U = TOP[resp("GATA1", c)].max(0).astype(bool)
    nv = {y: pooled(U, sets["ChIP"][y]) for y in cand_chip}
    cm = {y: pooled(U, sets["ChIP"][y], cbin) for y in cand_chip}
    arr_n = np.array([nv[y][2] for y in cand_chip if y != "GATA1"])
    arr_c = np.array([cm[y][2] for y in cand_chip if y != "GATA1"])
    gp[str(c)] = {"n_union_genes": int(U.sum()), "obs": nv["GATA1"][0], "exp_uniform": round(nv["GATA1"][1], 2),
                  "p_naive": float(np.exp(nv["GATA1"][2])), "rank_naive": int(1 + (arr_n < nv["GATA1"][2]).sum()),
                  "n_other_sets_naive_p_lt_0.05": int((np.exp(arr_n) < 0.05).sum()),
                  "exp_count_matched": round(cm["GATA1"][1], 2), "z_count_matched": round(cm["GATA1"][2], 2),
                  "rank_count_matched_z": int(1 + (arr_c > cm["GATA1"][2]).sum()),
                  "n_other_sets_z_gt_1.96": int((arr_c > 1.96).sum()), "n_candidates": len(cand_chip)}
KEY["gata1_pooled_union_chip"] = gp

# GATA1 ChIP-set enrichment in OTHER knockdowns' responding features (naive pooled test)
oth = {}
for c in [0.05, 0.02]:
    lst = []
    for t in tfs:
        if t == "GATA1" or len(resp(t, c)) < 5:
            continue
        U = TOP[resp(t, c)].max(0).astype(bool)
        lst.append((t, float(np.exp(pooled(U, S_G)[2]))))
    oth[str(c)] = {"n_other_knockdowns_with_ge5_features": len(lst),
                   "n_with_naive_p_lt_0.05_for_GATA1_set": int(sum(p < 0.05 for _, p in lst)),
                   "smallest": sorted(lst, key=lambda x: x[1])[:5]}
KEY["gata1_set_enrichment_in_other_knockdowns"] = oth
log("naive analyses done")

# 8. extra checks added for the key ---------------------------------------
# 8a. how rare are feature 2610's genes, and is 2610 rich in many TFs' ChIP targets (not only GATA1's)?
g2610 = top_idx[2610]
ov2610 = np.array([int(sets["ChIP"][y][g2610].sum()) for y in cand_chip])
hp2610 = np.array([st.hypergeom.sf(int(sets["ChIP"][y][g2610].sum()) - 1, NG, int(sets["ChIP"][y].sum()), len(g2610))
                   for y in cand_chip])
jG = cand_chip.index("GATA1")
KEY["f2610_profile"] = {
    "n_top_genes_detection_count_lt_10": int((cnt[g2610] < 10).sum()),
    "universe_share_detection_count_lt_10": float(np.mean(cnt < 10)),
    "all_top20_genes_share_detection_count_lt_10": float(np.mean(cnt[np.concatenate(top_idx)] < 10)),
    "median_length_bp_top_genes": float(np.nanmedian(ln[g2610])),
    "n_chip_sets_overlap_ge_5": int((ov2610 >= 5).sum()), "n_chip_sets_overlap_ge_8": int((ov2610 >= 8).sum()),
    "n_chip_sets_uniform_hypergeom_p_lt_0.05": int((hp2610 < 0.05).sum()),
    "gata1_rank_by_uniform_hypergeom_p": int(1 + (np.delete(hp2610, jG) < hp2610[jG]).sum()),
    "n_candidates": len(cand_chip)}

# 8b. sensitivity of the count-matched null to coarse bins (deciles of detection count instead of the fine edges)
dec = np.digitize(cnt, np.unique(np.percentile(cnt, np.arange(10, 100, 10))))
SWd = Swap(dec); pm_d = SWd.pmf_all(S_G); lc_d = lcdf(pm_d)
coarse = {"f2610_E_overlap_decile_bins": float((pm_d[2610] * np.arange(21)).sum()),
          "f2610_P_ge_8_decile_bins": float(pm_d[2610, 8:].sum())}
for c in [0.5, 0.25, 0.1]:
    Rr = resp("GATA1", c); M = int(ovG[Rr].max()); ls = lc_d[Rr].sum(0)
    coarse[f"gata1_cut{c}"] = {"K": int(len(Rr)), "M": M, "p_decile_bins_capped": p_swap(int(capT(M)), ls),
                               "p_decile_bins_uncapped": p_swap(M, ls) if M >= 2 else 1.0,
                               "p_fine_bins_uncapped": float(res[(res.tf == "GATA1") & (res.db == "ChIP") & (res.cutoff == c)].p_cm_uncapped.iloc[0])}
KEY["coarse_bin_sensitivity"] = coarse

# 8c. cut-off-free specificity check: for each of the 20 TFs with a ChIP set, Spearman correlation across features
#     (features with >= 10 listed genes) between |effect| and the overlap of the feature's top genes with a target set;
#     rank of the TF's own set among the 291 ChIP sets (1 = highest correlation)
okf = ntop >= 10
OVall = np.stack([TOP[:, sets["ChIP"][y]].sum(1) for y in cand_chip], 1).astype(float)[okf]
Yr = np.apply_along_axis(st.rankdata, 0, OVall); Yr = (Yr - Yr.mean(0)) / np.where(Yr.std(0) > 0, Yr.std(0), 1)
rk = []
for t in primary:
    x = st.rankdata(np.abs(sel[t]["d"])[okf]); x = (x - x.mean()) / x.std()
    rho = (x[:, None] * Yr).mean(0); j = cand_chip.index(t)
    rk.append({"tf": t, "rho_true": float(rho[j]), "rank": int(1 + (np.delete(rho, j) > rho[j]).sum()), "n": len(cand_chip)})
rkf = np.array([r["rank"] / r["n"] for r in rk]); rngb = np.random.default_rng(20261004)
bs = [rngb.choice(rkf, len(rkf)).mean() for _ in range(2000)]
KEY["cutoff_free_rank_check"] = {"per_tf": rk, "mean_rank_fraction": float(rkf.mean()),
                                 "ci95_bootstrap_over_tfs": [float(x) for x in np.percentile(bs, [2.5, 97.5])],
                                 "n_tfs_rank_in_top_half": int((rkf <= 0.5).sum()), "n_tfs_rank_in_top_5pct": int((rkf <= 0.05).sum()),
                                 "binomial_p_top_half": float(st.binomtest(int((rkf <= 0.5).sum()), len(rkf), 0.5, alternative="greater").pvalue),
                                 "note": "uncertainty: bootstrap over the 20 TFs (2,000 resamples, seed 20261004); 0.5 = no specificity"}

# 8d. Wilson CIs for the planted-target power (200 repeats each)
def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


for k_, v in KEY["gata1_planted_power_cm_cut0.5"].items():
    v["capped_wilson95"] = wilson(round(v["capped"] * 200), 200); v["uncapped_wilson95"] = wilson(round(v["uncapped"] * 200), 200)

# 8f. the source paper's feature-level test, read literally: one-sided Fisher exact test of responding vs other
#     catalog features x 'top-20 overlaps the TF's targets' (overlap >= 1, and >= 2); BH across TFs per setting
spec_fisher = {}
for c in CUT:
    for db in DBS:
        for mn in (1, 2):
            pv, names = [], []
            for t in tfs:
                if t not in sets[db]:
                    continue
                Rr = resp(t, c); names.append(t)
                if len(Rr) == 0:
                    pv.append(1.0); continue
                hit = get_ov(db, t) >= mn; isr = np.zeros(NF, bool); isr[Rr] = True
                a = int((isr & hit).sum()); b = int((isr & ~hit).sum()); cc = int((~isr & hit).sum()); d = int((~isr & ~hit).sum())
                pv.append(float(st.fisher_exact([[a, b], [cc, d]], alternative="greater")[1]))
            q = bh(pv)
            spec_fisher[f"{c}|{db}|overlap>={mn}"] = {"n_tfs": len(pv), "n_p_lt_0.05": int((np.array(pv) < 0.05).sum()),
                                                      "n_q_lt_0.05": int((q < 0.05).sum()),
                                                      "tfs_q_lt_0.05": [names[i] for i in np.where(q < 0.05)[0]],
                                                      "gata1_p": float(pv[names.index("GATA1")]) if "GATA1" in names else None}
KEY["spec_feature_fisher_test"] = spec_fisher

# 8f (cont.). for every ChIP setting where a TF passes BH with this Fisher test: the same test with every other ChIP set
#     (291 sets with >= 20 targets) on the same responding features; rank of the true TF (1 = smallest p)
OVc = np.stack([TOP[:, sets["ChIP"][y]].sum(1) for y in cand_chip], 1)
fr = {}
for k_, v in spec_fisher.items():
    c, db, mn = k_.split("|"); mn = int(mn[-1])
    if db != "ChIP":
        continue
    for t in v["tfs_q_lt_0.05"]:
        Rr = resp(t, float(c)); isr = np.zeros(NF, bool); isr[Rr] = True
        H = OVc >= mn; ps = []
        for j in range(len(cand_chip)):
            h = H[:, j]
            ps.append(st.fisher_exact([[int((isr & h).sum()), int((isr & ~h).sum())],
                                       [int((~isr & h).sum()), int((~isr & ~h).sum())]], alternative="greater")[1])
        ps = np.array(ps); j = cand_chip.index(t)
        fr[f"{k_}|{t}"] = {"K": int(len(Rr)), "p_true": float(ps[j]), "rank_true": int(1 + (np.delete(ps, j) < ps[j]).sum()),
                           "n_sets": len(cand_chip), "n_other_sets_p_lt_0.05": int((np.delete(ps, j) < 0.05).sum()),
                           "n_other_sets_p_le_p_true": int((np.delete(ps, j) <= ps[j]).sum())}
KEY["spec_feature_fisher_rank_among_tf_sets"] = fr

# 8g. the same Fisher test for every ChIP TF x cut-off x overlap rule with >= 1 responding feature: rank of the true TF's
#     set among the 291 ChIP sets (mid-rank for ties; 1 = smallest p); and how rare the responding features' genes are
#     (median over features of the median detection count of the feature's top genes)
medcnt = np.array([np.median(cnt[top_idx[f]]) if len(top_idx[f]) else np.nan for f in range(NF)])
allr = []
for t in primary:
    j = cand_chip.index(t)
    for c in CUT:
        Rr = resp(t, c); K = len(Rr)
        if K == 0:
            continue
        isr = np.zeros(NF, bool); isr[Rr] = True
        for mn in (1, 2):
            H = OVc >= mn; ps = st.hypergeom.sf((H & isr[:, None]).sum(0) - 1, NF, H.sum(0), K)
            o = np.delete(ps, j)
            allr.append({"tf": t, "cutoff": c, "min_overlap": mn, "K": K, "p_true": float(ps[j]),
                         "rank": float(1 + (o < ps[j]).sum() + 0.5 * (o == ps[j]).sum()),
                         "median_detection_of_responding_features": float(np.nanmedian(medcnt[Rr]))})
ar = pd.DataFrame(allr); fr_ = ar["rank"].values / len(cand_chip)
rngf = np.random.default_rng(20261005); tfl = ar.tf.unique()
bsf = [np.mean(np.concatenate([fr_[ar.tf.values == x] for x in rngf.choice(tfl, len(tfl))])) for _ in range(2000)]
KEY["spec_feature_fisher_rank_all_settings"] = {
    "n_settings": int(len(ar)), "n_tfs": int(len(tfl)), "mean_rank_fraction": float(fr_.mean()),
    "ci95_bootstrap_over_tfs": [float(x) for x in np.percentile(bsf, [2.5, 97.5])],
    "n_settings_true_tf_in_top_5pct": int((fr_ <= 0.05).sum()), "expected_by_chance_top_5pct": float(0.05 * len(ar)),
    "settings_in_top_5pct": ar[fr_ <= 0.05][["tf", "cutoff", "min_overlap", "K", "rank"]].to_dict("records"),
    "note": "Fisher one-sided, responding vs other catalog features x (top-20 overlap with the set >= min_overlap); 20 ChIP TFs, 7 cut-offs, min_overlap 1 and 2; uncertainty = bootstrap over TFs (2,000 resamples, seed 20261005)"}
KEY["responding_feature_rarity"] = {
    "all_features_median_of_median_detection": float(np.nanmedian(medcnt)),
    "gata1_by_cutoff": {str(c): float(np.nanmedian(medcnt[resp("GATA1", c)])) for c in CUT if len(resp("GATA1", c))},
    "max_by_cutoff": {str(c): float(np.nanmedian(medcnt[resp("MAX", c)])) for c in CUT if len(resp("MAX", c))},
    "note": "per feature: median detection_count of its listed top genes; then the median over the responding features"}

# 8e. testability: TFs whose target set has >= 2 genes in the universe (needed to ever reach overlap 2)
KEY["n_tfs_trrust_set_ge2_in_universe"] = int(sum(sets["TRRUST"][t].sum() >= 2 for t in tfs if t in sets["TRRUST"]))
KEY["n_testable_tfs_by_cutoff"] = {db: {str(c): int(((res.db == db) & (res.cutoff == c) & (res.K > 0)).sum()) for c in CUT} for db in DBS}
KEY["wall_seconds"] = round(time.time() - T0, 1)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


OUT.write_text(json.dumps(clean(KEY), indent=1))
print(json.dumps(clean({k: v for k, v in KEY.items() if k not in ("bh_summary", "naive_universe_hypergeom", "gata1_tests")}), indent=1))
log("written", OUT)
