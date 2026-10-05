"""Independent check of variants, |LFC| subsets, the threshold-family test and within-knockdown tests."""
import os, json, time
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import StratifiedTable

PKG = "<EVAL_ROOT>/studyA/tasks/T2-paper/data"
HERE = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()
P = pd.read_parquet(f"{HERE}/pairs_with_sumd.parquet")
Z = np.load(f"{PKG}/knockdown_lfc.npz", allow_pickle=False)
res = {}
y = P["lfc"].values < 0
p = P["predicted_decrease"].values
frac = (P["n_inhibitory_evidence"] / P["evidence"]).values
lfc = P["lfc"].values
genes = np.array(sorted(P["silenced_gene"].unique()))
g = pd.Categorical(P["silenced_gene"], categories=genes).codes
NG = len(genes)
targets = np.array(sorted(P["target_gene"].unique()))
t = pd.Categorical(P["target_gene"], categories=targets).codes
NT = len(targets)


def met(pred, obs):
    tp = np.sum(pred & obs); fn = np.sum(~pred & obs); fp = np.sum(pred & ~obs); tn = np.sum(~pred & ~obs)
    n = tp + fn + fp + tn
    acc = (tp + tn) / n; pobs = (tp + fn) / n; ppred = (tp + fp) / n
    r1 = tp / (tp + fn) if tp + fn else np.nan; r0 = tn / (tn + fp) if tn + fp else np.nan
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den > 0 else np.nan
    return dict(n=int(n), acc=acc, always_dec=pobs, ppred=ppred, ba=0.5 * (r1 + r0), mcc=mcc)


def gene_boot(pred, obs, gi, reps=2000, seed=31):
    gs = np.unique(gi)
    cm = {k: None for k in gs}
    tab = np.zeros((gi.max() + 1, 4))
    np.add.at(tab, (gi, 0), pred & obs); np.add.at(tab, (gi, 1), ~pred & obs)
    np.add.at(tab, (gi, 2), pred & ~obs); np.add.at(tab, (gi, 3), ~pred & ~obs)
    tab = tab[gs]
    rng = np.random.default_rng(seed)
    ba, mc, ac = [], [], []
    for _ in range(reps):
        c = tab[rng.integers(0, len(gs), len(gs))].sum(0)
        tp, fn, fp, tn = c
        ba.append(0.5 * (tp / (tp + fn) + tn / (tn + fp)) if (tp + fn) and (tn + fp) else np.nan)
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mc.append((tp * tn - fp * fn) / den if den > 0 else np.nan)
        ac.append((tp + tn) / c.sum())
    q = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    return dict(ba_ci=q(ba), mcc_ci=q(mc), acc_ci=q(ac))


V = {}
V["ties_as_decrease"] = met(frac >= 0.5, y) | gene_boot(frac >= 0.5, y, g)
V["sign_sum_d"] = met(P["sum_d"].values < 0, y) | gene_boot(P["sum_d"].values < 0, y, g)
un = (frac == 0) | (frac == 1)
V["unanimous"] = met(p[un], y[un]) | gene_boot(p[un], y[un], g[un])
V["unanimous"]["n_pairs"] = int(un.sum())
mx2 = P["max_abs_d"].values > 2
V["maxd_gt2"] = met(p[mx2], y[mx2]) | gene_boot(p[mx2], y[mx2], g[mx2])
V["maxd_gt2"]["n_genes"] = int(len(np.unique(g[mx2])))
for thr in [0.01, 0.05, 0.1, 0.25, 0.5]:
    s = np.abs(lfc) >= thr
    V[f"abs_lfc_ge_{thr}"] = met(p[s], y[s]) | gene_boot(p[s], y[s], g[s])
    V[f"abs_lfc_ge_{thr}"]["n_genes"] = int(len(np.unique(g[s])))
for ev in [5, 10, 20]:
    s = P["evidence"].values >= ev
    V[f"evidence_ge_{ev}"] = met(p[s], y[s]) | gene_boot(p[s], y[s], g[s])
mx1 = P["max_abs_d"].values > 1
V["maxd_gt1"] = met(p[mx1], y[mx1]) | gene_boot(p[mx1], y[mx1], g[mx1])
# stronger half of knockdowns by own LFC
sil = Z["silenced_genes"]; meas = list(Z["measured_genes"]); L = Z["lfc"]
own = np.array([L[i, meas.index(s_)] for i, s_ in enumerate(sil)])
own_by_gene = dict(zip(sil, own))
og = np.array([own_by_gene[x] for x in genes])
strong = og <= np.median(og)
s = strong[g]
V["stronger_half"] = met(p[s], y[s])
res["variants"] = V

# ---- |lfc| >= 0.5 subset, within-gene shuffle null and CMH ----
s5 = np.abs(lfc) >= 0.5
res["sub05_n"] = int(s5.sum()); res["sub05_genes"] = int(len(np.unique(g[s5])))


def mcc_of(pred, obs):
    return met(pred, obs)["mcc"]


def shuffle_within(pred, grp, rng):
    out = pred.copy()
    order = np.argsort(grp, kind="stable")
    gs = grp[order]
    bounds = np.flatnonzero(np.diff(gs)) + 1
    for blk in np.split(order, bounds):
        out[blk] = pred[rng.permutation(blk)]
    return out


rng = np.random.default_rng(777)
ps, ys, gs5 = p[s5], y[s5], g[s5]
obs_mcc = mcc_of(ps, ys)
null = np.array([mcc_of(shuffle_within(ps, gs5, rng), ys) for _ in range(2000)])
res["sub05_mcc"] = obs_mcc
res["sub05_null_mean"] = float(null.mean()); res["sub05_null_p"] = float((1 + np.sum(null >= obs_mcc)) / 2001)


def cmh(pred, obs, strata):
    tabs = []
    for k in np.unique(strata):
        m_ = strata == k
        a = np.sum(pred[m_] & obs[m_]); b = np.sum(pred[m_] & ~obs[m_])
        c = np.sum(~pred[m_] & obs[m_]); d = np.sum(~pred[m_] & ~obs[m_])
        tabs.append(np.array([[a, b], [c, d]], float))
    st = StratifiedTable(tabs)
    r = st.test_null_odds(correction=False)
    return float(st.oddsratio_pooled), float(r.pvalue)


res["sub05_cmh_by_gene"] = cmh(ps, ys, gs5)

# target's usual direction, leave the silenced gene out (majority over the other silenced genes)
dec_t = np.bincount(t, weights=y, minlength=NT); tot_t = np.bincount(t, minlength=NT)
dec_loo = dec_t[t] - y; tot_loo = tot_t[t] - 1
usual = np.where(tot_loo > 0, np.where(2 * dec_loo > tot_loo, 1, np.where(2 * dec_loo < tot_loo, 0, 2)), 2)
res["cmh_all_by_gene"] = cmh(p, y, g)
res["cmh_all_by_gene_x_usual"] = cmh(p, y, g * 3 + usual)
res["sub05_cmh_by_gene_x_usual"] = cmh(ps, ys, (g * 3 + usual)[s5])
s45 = np.abs(lfc) >= 0.45
res["sub045_cmh_by_gene_x_usual"] = cmh(p[s45], y[s45], (g * 3 + usual)[s45])
# target-direction (LOO by silenced gene) rule on the 0.5 subset
res["sub05_target_rule"] = met((usual == 1)[s5] | ((usual == 2)[s5] & True), ys)

# ---- threshold family ----
ths = [0.25, 0.4, 0.45, 0.5, 0.6, 0.75]
a = np.abs(lfc)
sfam = a >= ths[0]
bins = np.digitize(a[sfam], ths)  # 1..6
pf, yf, gf, af = p[sfam], y[sfam], g[sfam], a[sfam]
strata = gf * 10 + bins
masks = [af >= th for th in ths]
obs_v = np.array([mcc_of(pf[mk], yf[mk]) for mk in masks])
rng = np.random.default_rng(4242)
NV = []
for _ in range(2000):
    q = shuffle_within(pf, strata, rng)
    NV.append([mcc_of(q[mk], yf[mk]) for mk in masks])
NV = np.array(NV)
mu, sd = NV.mean(0), NV.std(0, ddof=1)
zo = (obs_v - mu) / sd
zn = (NV - mu) / sd
res["family_obs_mcc"] = obs_v.tolist(); res["family_null_mean"] = mu.tolist()
res["family_z"] = zo.tolist()
res["family_max_z"] = float(zo.max()); res["family_argmax"] = ths[int(zo.argmax())]
res["family_p_max"] = float((1 + np.sum(zn.max(1) >= zo.max())) / (1 + len(zn)))
res["family_single_p"] = [float((1 + np.sum(NV[:, j] >= obs_v[j])) / (1 + len(NV))) for j in range(len(ths))]

# ---- per-gene Spearman of frac vs -lfc ----
mean_other = (np.bincount(t, weights=lfc, minlength=NT)[t] - lfc) / np.maximum(tot_t[t] - 1, 1)
resid = lfc - mean_other
sp, sp_r = [], []
for k in range(NG):
    m_ = g == k
    if m_.sum() <= 10:
        continue
    r1 = stats.spearmanr(frac[m_], -lfc[m_]).statistic
    r2 = stats.spearmanr(frac[m_], -resid[m_]).statistic
    if np.isfinite(r1):
        sp.append(r1); sp_r.append(r2)
sp, sp_r = np.array(sp), np.array(sp_r)
res["spearman_n_genes"] = int(len(sp))
res["spearman_mean"] = float(np.mean(sp)); res["spearman_resid_mean"] = float(np.nanmean(sp_r))
rng = np.random.default_rng(8)
b1 = [np.mean(sp[rng.integers(0, len(sp), len(sp))]) for _ in range(2000)]
b2 = [np.nanmean(sp_r[rng.integers(0, len(sp_r), len(sp_r))]) for _ in range(2000)]
res["spearman_ci"] = [float(np.percentile(b1, 2.5)), float(np.percentile(b1, 97.5))]
res["spearman_resid_ci"] = [float(np.percentile(b2, 2.5)), float(np.percentile(b2, 97.5))]
res["seconds"] = round(time.time() - t0, 1)
print(json.dumps(res, indent=1, default=float))
