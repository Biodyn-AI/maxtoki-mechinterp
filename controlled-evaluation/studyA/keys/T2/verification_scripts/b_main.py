"""Independent main metrics for T2: pooled metrics, clustered bootstraps, target-direction baseline,
per-gene summaries. Own seeds (not 42) except where the key defines the folds by seed 42."""
import os, json, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

PKG = "<EVAL_ROOT>/studyA/tasks/T2-paper/data"
HERE = os.path.dirname(os.path.abspath(__file__))
t0 = time.time()
P = pd.read_parquet(f"{HERE}/pairs_with_sumd.parquet")
E = pd.read_csv(f"{PKG}/circuit_edges.csv")
SF = pd.read_csv(f"{PKG}/source_features.tsv", sep="\t")
res = {}

y = (P["lfc"].values < 0)                # observed decrease
p = P["predicted_decrease"].values
genes = np.array(sorted(P["silenced_gene"].unique()))
g_idx = pd.Categorical(P["silenced_gene"], categories=genes).codes
NG = len(genes)


def conf_by_group(pred, obs, grp, ng):
    tp = np.bincount(grp, weights=(pred & obs), minlength=ng)
    fn = np.bincount(grp, weights=(~pred & obs), minlength=ng)
    fp = np.bincount(grp, weights=(pred & ~obs), minlength=ng)
    tn = np.bincount(grp, weights=(~pred & ~obs), minlength=ng)
    return np.stack([tp, fn, fp, tn], 1)


def metrics(c):
    tp, fn, fp, tn = [float(v) for v in c]
    n = tp + fn + fp + tn
    acc = (tp + tn) / n
    pobs = (tp + fn) / n; ppred = (tp + fp) / n
    always = max(pobs, 1 - pobs)
    rec1 = tp / (tp + fn) if tp + fn else np.nan
    rec0 = tn / (tn + fp) if tn + fp else np.nan
    ba = 0.5 * (rec1 + rec0)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den > 0 else np.nan
    chance = ppred * pobs + (1 - ppred) * (1 - pobs)
    kappa = (acc - chance) / (1 - chance) if chance < 1 else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    return dict(n=n, acc=acc, pobs=pobs, ppred=ppred, always_dec=pobs, best_const=always, ba=ba, mcc=mcc,
                chance=chance, kappa=kappa, prec_dec=prec, acc_minus_always_dec=acc - pobs,
                acc_minus_best_const=acc - always, ba_m_half=ba - 0.5, acc_minus_chance=acc - chance)


C = conf_by_group(p, y, g_idx, NG)
m = metrics(C.sum(0))
res["pooled"] = m
res["n_pairs"] = int(len(P)); res["n_genes"] = NG; res["n_targets"] = int(P["target_gene"].nunique())
res["pair_z_acc_vs_half"] = (m["acc"] - 0.5) / np.sqrt(0.25 / len(P))
npd = int(p.sum())
res["pair_z_prec_vs_half"] = (m["prec_dec"] - 0.5) / np.sqrt(0.25 / npd)
frac = (P["n_inhibitory_evidence"] / P["evidence"]).values
res["auroc_frac"] = float(roc_auc_score(y, frac))
# pair-level (naive) accuracy CI
res["pair_level_acc_ci"] = [m["acc"] - 1.96 * np.sqrt(m["acc"] * (1 - m["acc"]) / len(P)),
                            m["acc"] + 1.96 * np.sqrt(m["acc"] * (1 - m["acc"]) / len(P))]


def boot(Cg, groups_of_gene, ngroups, reps=2000, seed=2026, stats=("acc", "acc_minus_always_dec", "ba_m_half", "mcc", "acc_minus_chance")):
    """Cg: per-gene confusion; groups_of_gene: group id per gene; resample groups with replacement."""
    Cgrp = np.zeros((ngroups, Cg.shape[1]))
    np.add.at(Cgrp, groups_of_gene, Cg)
    rng = np.random.default_rng(seed)
    out = {s: [] for s in stats}
    for _ in range(reps):
        idx = rng.integers(0, ngroups, ngroups)
        mm = metrics(Cgrp[idx].sum(0))
        for s in stats:
            out[s].append(mm[s])
    return {s: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for s, v in out.items()}


res["boot_gene"] = boot(C, np.arange(NG), NG)

# source-feature groups and components
edge_src = set(zip(E["src_layer"], E["src_feature"]))
sf_sets = {g: set() for g in genes}
for l, f, tl in zip(SF["layer"], SF["feature"], SF["top10_genes"]):
    if (l, f) not in edge_src:
        continue
    for g in tl.split(","):
        if g in sf_sets:
            sf_sets[g].add((l, f))
keys = [tuple(sorted(sf_sets[g])) for g in genes]
uniq = {k: i for i, k in enumerate(sorted(set(keys)))}
grp = np.array([uniq[k] for k in keys])
res["n_sf_groups"] = len(uniq)
res["n_genes_without_edge_source"] = int(sum(1 for k in keys if len(k) == 0))
# components (union-find over genes sharing any source feature)
parent = list(range(NG))
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a
feat2genes = {}
for i, g in enumerate(genes):
    for f in sf_sets[g]:
        feat2genes.setdefault(f, []).append(i)
for f, lst in feat2genes.items():
    for j in lst[1:]:
        ra, rb = find(lst[0]), find(j)
        if ra != rb:
            parent[ra] = rb
roots = np.array([find(i) for i in range(NG)])
ur = {r: i for i, r in enumerate(sorted(set(roots)))}
comp = np.array([ur[r] for r in roots])
res["n_components"] = len(ur); res["largest_component"] = int(np.bincount(comp).max())
res["boot_sfgroup"] = boot(C, grp, len(uniq), seed=11)
res["boot_component"] = boot(C, comp, len(ur), seed=12)

# target-direction baseline
tg = P["target_gene"].values
targets = np.array(sorted(set(tg)))
t_idx = pd.Categorical(tg, categories=targets).codes
NT = len(targets)


def target_baseline(fold_of_gene):
    pred = np.zeros(len(P), bool)
    for k in np.unique(fold_of_gene):
        test = fold_of_gene[g_idx] == k
        train = ~test
        dec = np.bincount(t_idx[train], weights=y[train], minlength=NT)
        tot = np.bincount(t_idx[train], minlength=NT)
        glob = y[train].mean() > 0.5
        maj = np.where(tot > 0, np.where(dec * 2 > tot, True, np.where(dec * 2 < tot, False, glob)), glob)
        pred[test] = maj[t_idx[test]]
    return pred


# folds as the key defines them: default_rng(42) permutation of the sorted gene list, split in 5
perm = np.random.default_rng(42).permutation(NG)
fold42 = np.empty(NG, int)
for k, chunk in enumerate(np.array_split(perm, 5)):
    fold42[chunk] = k
tb42 = target_baseline(fold42)
res["target_baseline_seed42"] = metrics(conf_by_group(tb42, y, g_idx, NG).sum(0))
# other random fold splits
alt = []
for s in [1, 2, 3]:
    pm = np.random.default_rng(s).permutation(NG)
    fo = np.empty(NG, int)
    for k, chunk in enumerate(np.array_split(pm, 5)):
        fo[chunk] = k
    alt.append(metrics(conf_by_group(target_baseline(fo), y, g_idx, NG).sum(0))["acc"])
res["target_baseline_other_seeds_acc"] = alt
# leave-one-gene-out
tbl = target_baseline(np.arange(NG))
res["target_baseline_loo"] = metrics(conf_by_group(tbl, y, g_idx, NG).sum(0))
# circuit minus baseline, gene bootstrap
corr_c = np.bincount(g_idx, weights=(p == y), minlength=NG)
corr_b = np.bincount(g_idx, weights=(tb42 == y), minlength=NG)
cnt = np.bincount(g_idx, minlength=NG)
res["circuit_minus_baseline"] = float((corr_c.sum() - corr_b.sum()) / cnt.sum())
rng = np.random.default_rng(99)
bs = []
for _ in range(2000):
    i = rng.integers(0, NG, NG)
    bs.append((corr_c[i].sum() - corr_b[i].sum()) / cnt[i].sum())
res["circuit_minus_baseline_ci_gene"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
# per fold
pf = []
for k in range(5):
    sel = fold42[g_idx] == k
    pf.append([float((p[sel] == y[sel]).mean()), float(y[sel].mean()), float((tb42[sel] == y[sel]).mean())])
res["per_fold_circuit_always_baseline"] = pf

# per-gene summaries
pg = [metrics(C[i]) for i in range(NG)]
mccs = np.array([d["mcc"] for d in pg])
res["n_genes_mcc_defined"] = int(np.isfinite(mccs).sum())
res["per_gene_mean_mcc"] = float(np.nanmean(mccs))
ppred_g = np.array([d["ppred"] for d in pg])
res["n_one_sign_genes"] = int(((ppred_g == 0) | (ppred_g == 1)).sum())
amd = np.array([d["acc_minus_always_dec"] for d in pg])
res["per_gene_mean_acc_minus_always_dec"] = float(amd.mean())
res["per_gene_mean_acc"] = float(np.mean([d["acc"] for d in pg]))
rng = np.random.default_rng(5)
b1, b2 = [], []
for _ in range(2000):
    i = rng.integers(0, NG, NG)
    b1.append(np.nanmean(mccs[i])); b2.append(amd[i].mean())
res["per_gene_mean_mcc_ci"] = [float(np.percentile(b1, 2.5)), float(np.percentile(b1, 97.5))]
res["per_gene_mean_amd_ci"] = [float(np.percentile(b2, 2.5)), float(np.percentile(b2, 97.5))]
res["n_genes_beat_always_dec"] = int((amd > 0).sum())
res["n_genes_ba_gt_half"] = int(np.sum(np.array([d["ba"] for d in pg]) > 0.5))
res["seconds"] = round(time.time() - t0, 1)
print(json.dumps(res, indent=1, default=float))
