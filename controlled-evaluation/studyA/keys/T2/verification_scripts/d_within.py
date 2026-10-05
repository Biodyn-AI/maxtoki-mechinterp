"""CMH by hand, target usual direction from the full LFC matrix (other 226 silenced genes),
per-site variants (rebuild gene pairs from one source site's edges), per-gene Spearman after
removing the target's mean change in the other 226 knockdowns."""
import os, json, time
import numpy as np, pandas as pd
from scipy import stats

PKG = "<EVAL_ROOT>/studyA/tasks/T2-paper/data"
HERE = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()
P = pd.read_parquet(f"{HERE}/pairs_with_sumd.parquet")
Z = np.load(f"{PKG}/knockdown_lfc.npz", allow_pickle=False)
res = {}
y = P["lfc"].values < 0
p = P["predicted_decrease"].values
lfc = P["lfc"].values
frac = (P["n_inhibitory_evidence"] / P["evidence"]).values
sil = list(Z["silenced_genes"]); meas = list(Z["measured_genes"]); L = Z["lfc"].astype(np.float64)
si = {s: i for i, s in enumerate(sil)}; mi = {m: i for i, m in enumerate(meas)}
r = P["silenced_gene"].map(si).values; c = P["target_gene"].map(mi).values
NS = len(sil)
# usual direction of each measured gene over the other 226 silenced genes
dec_all = (L < 0).sum(0)                     # per measured gene, over 227
dec_other = dec_all[c] - (L[r, c] < 0)
usual = np.where(2 * dec_other > NS - 1, 1, np.where(2 * dec_other < NS - 1, 0, 2))
mean_other = (L.sum(0)[c] - L[r, c]) / (NS - 1)


def cmh(pred, obs, strata):
    a_obs = e = v = 0.0
    num = den = 0.0
    for k in np.unique(strata):
        m = strata == k
        n = m.sum()
        if n < 2:
            continue
        a = np.sum(pred[m] & obs[m]); b = np.sum(pred[m] & ~obs[m])
        cc = np.sum(~pred[m] & obs[m]); d = np.sum(~pred[m] & ~obs[m])
        r1, r2, c1, c2 = a + b, cc + d, a + cc, b + d
        a_obs += a; e += r1 * c1 / n
        v += r1 * r2 * c1 * c2 / (n * n * (n - 1))
        num += a * d / n; den += b * cc / n
    chi = (a_obs - e) ** 2 / v
    return dict(or_mh=float(num / den), p=float(stats.chi2.sf(chi, 1)), chi2=float(chi))


g = P["silenced_gene"].map(si).values
res["cmh_all_by_gene"] = cmh(p, y, g)
res["cmh_all_by_gene_x_usual"] = cmh(p, y, g * 3 + usual)
a = np.abs(lfc)
for th in [0.45, 0.5]:
    s = a >= th
    res[f"cmh_sub{th}_by_gene"] = cmh(p[s], y[s], g[s])
    res[f"cmh_sub{th}_by_gene_x_usual"] = cmh(p[s], y[s], (g * 3 + usual)[s])
# target-level Spearman: mean support fraction vs mean -lfc per target
T = pd.DataFrame({"t": P["target_gene"].values, "f": frac, "l": lfc}).groupby("t").mean()
res["target_level_spearman"] = float(stats.spearmanr(T["f"], -T["l"]).statistic)
# per-gene Spearman after removing the target's mean in the other 226 knockdowns
resid = lfc - mean_other
sp_r = []
for k in np.unique(g):
    m = g == k
    if m.sum() <= 10:
        continue
    v = stats.spearmanr(frac[m], -resid[m]).statistic
    if np.isfinite(stats.spearmanr(frac[m], -lfc[m]).statistic):
        sp_r.append(v)
sp_r = np.array(sp_r)
rng = np.random.default_rng(3)
bb = [np.nanmean(sp_r[rng.integers(0, len(sp_r), len(sp_r))]) for _ in range(2000)]
res["spearman_resid226"] = [float(np.nanmean(sp_r)), float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5)), int(len(sp_r))]

# ---- per-site variants: rebuild pairs from one source site's edges with the same rule ----
E = pd.read_csv(f"{PKG}/circuit_edges.csv")
FT = pd.read_csv(f"{PKG}/feature_top_genes.tsv", sep="\t")
top = {(l, f): v.split(",") for l, f, v in zip(FT["layer"], FT["feature"], FT["top10_genes"])}
genes = sorted({x for v in top.values() for x in v}); gi = {x: i for i, x in enumerate(genes)}; G = len(genes)
fid = {k: i for i, k in enumerate(top)}
M = np.array([[gi[x] for x in top[k]] for k in top], dtype=np.int64)
silset = set(sil); measset = set(meas)
lfc_lookup = lambda sg, tg: L[[si[x] for x in sg], [mi[x] for x in tg]]
site = {}
for s_layer in [0, 3, 6, 9]:
    Es = E[E["src_layer"] == s_layer]
    sidx = np.array([fid[k] for k in zip(Es["src_layer"], Es["src_feature"])])
    tidx = np.array([fid[k] for k in zip(Es["tgt_layer"], Es["tgt_feature"])])
    inh = (Es["sign"] == "inhibitory").values; ad = Es["cohens_d"].abs().values
    K, I, D = [], [], []
    for aa in range(10):
        for bb_ in range(10):
            sa, tb = M[sidx, aa], M[tidx, bb_]
            ok = sa != tb
            K.append(sa[ok] * G + tb[ok]); I.append(inh[ok]); D.append(ad[ok])
    K = np.concatenate(K); I = np.concatenate(I); D = np.concatenate(D)
    o = np.argsort(K, kind="stable"); K, I, D = K[o], I[o], D[o]
    uk, st = np.unique(K, return_index=True)
    ev = np.diff(np.append(st, len(K))); ni = np.add.reduceat(I.astype(np.int64), st); mx = np.maximum.reduceat(D, st)
    keep = (ev >= 2) | (mx > 2.0)
    uk, ev, ni = uk[keep], ev[keep], ni[keep]
    sg = np.array(genes)[uk // G]; tg = np.array(genes)[uk % G]
    ok = np.array([x in silset for x in sg]) & np.array([x in measset for x in tg])
    sg, tg, ev, ni = sg[ok], tg[ok], ev[ok], ni[ok]
    pr = ni / ev > 0.5
    ob = lfc_lookup(sg, tg) < 0
    tp = np.sum(pr & ob); fn = np.sum(~pr & ob); fp = np.sum(pr & ~ob); tn = np.sum(~pr & ~ob)
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    site[s_layer] = dict(n=int(len(pr)), n_genes=int(len(set(sg))), ppred=float(pr.mean()), pobs=float(ob.mean()),
                         acc=float((pr == ob).mean()), acc_minus_always=float((pr == ob).mean() - ob.mean()),
                         ba=float(0.5 * (tp / (tp + fn) + tn / (tn + fp))), mcc=float((tp * tn - fp * fn) / den) if den else None)
res["per_site_rebuild"] = site
# per-site by filtering pairs: not possible from the parquet alone (support is pooled over sites)
res["seconds"] = round(time.time() - t0, 1)
print(json.dumps(res, indent=1))
