"""Reference computation for Study A, task T2 (CRISPRi direction from MaxToki-217M SAE circuits, K562).

Reads ONLY the package data (tasks/T2-paper/data/, byte-identical to tasks/T2-contract/data/).
CPU only. Results are merged into reference_output.json next to this file.

Usage (each section is a separate command; each takes well under 9 minutes):
    python reference.py main        # file facts, rebuild of the gene pairs from the edges, pooled
                                    # metrics, baselines, grouped bootstraps, per-gene summary,
                                    # |LFC| bands, pair-level tests, AUROC of the support fraction
    python reference.py variants    # analyst choices: tie rule, sign-of-summed-d rule, |LFC| filters,
                                    # support filters, one source site at a time, on-target knockdown;
                                    # |LFC| subsets with within-gene nulls; other target baselines
    python reference.py within      # association inside each knockdown (CMH, per-gene Spearman),
                                    # before and after the target gene's usual direction

Positive class everywhere = observed decrease (lfc < 0). Seeds follow the source evaluation
(bootstrap seed 42, 2,000 reps; folds of silenced genes from numpy default_rng(42)). With the same
seeds every number the source report gives is reproduced exactly from the package data.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "tasks" / "T2-paper" / "data"
OUT = HERE / "reference_output.json"
SEED = 42
NBOOT = 2000
SECTIONS = sys.argv[1:] or ["main"]


# ------------------------------------------------------------------ helpers
def r5(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [r5(v) for v in x]
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else float(round(x, 5))


def save(section, obj):
    allres = json.load(open(OUT)) if OUT.exists() else {}
    allres[section] = obj
    json.dump(allres, open(OUT, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def pci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def conf_counts(pred, obs):
    pred = np.asarray(pred, bool); obs = np.asarray(obs, bool)
    return int((pred & obs).sum()), int((~pred & obs).sum()), int((pred & ~obs).sum()), int((~pred & ~obs).sum())


def metrics_from_counts(tp, fn, fp, tn):
    """Positive = observed decrease. tp = pred dec & obs dec, fn = pred inc & obs dec,
    fp = pred dec & obs inc, tn = pred inc & obs inc. Works on arrays (bootstrap)."""
    tp, fn, fp, tn = (np.asarray(x, float) for x in (tp, fn, fp, tn))
    n = tp + fn + fp + tn
    p_obs = (tp + fn) / n
    p_pred = (tp + fp) / n
    acc = (tp + tn) / n
    with np.errstate(invalid="ignore", divide="ignore"):
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        bal = 0.5 * (tpr + tnr)
        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        chance = p_pred * p_obs + (1 - p_pred) * (1 - p_obs)
        kappa = (acc - chance) / (1 - chance)
        prec_dec = tp / (tp + fp)
        prec_inc = tn / (tn + fn)
    return dict(n=n, frac_obs_dec=p_obs, frac_pred_dec=p_pred, accuracy=acc, always_dec_acc=p_obs,
                always_inc_acc=1 - p_obs, best_constant_acc=np.maximum(p_obs, 1 - p_obs),
                acc_minus_best_constant=acc - np.maximum(p_obs, 1 - p_obs),
                recall_dec=tpr, recall_inc=tnr, balanced_acc=bal, mcc=mcc,
                chance_acc_given_marginals=chance, acc_minus_chance_given_marginals=acc - chance,
                cohen_kappa=kappa, precision_pred_dec=prec_dec, precision_pred_inc=prec_inc)


def scal(d):
    return {k: r5(v) for k, v in d.items()}


def per_gene_counts(df, pred):
    g = pd.DataFrame({"s": df["silenced_gene"].to_numpy(), "p": np.asarray(pred, bool), "o": df["obs_dec"].to_numpy()})
    ps = pd.DataFrame({
        "tp": (g.p & g.o).groupby(g.s).sum(), "fn": (~g.p & g.o).groupby(g.s).sum(),
        "fp": (g.p & ~g.o).groupby(g.s).sum(), "tn": (~g.p & ~g.o).groupby(g.s).sum()}).astype(int).sort_index()
    for k, v in metrics_from_counts(ps.tp, ps.fn, ps.fp, ps.tn).items():
        ps[k] = np.asarray(v)
    return ps


def gene_boot_ci(df, pred, seed=SEED, nboot=NBOOT):
    """Percentile bootstrap over silenced genes (with replacement); pooled metrics from summed counts."""
    ps = per_gene_counts(df, pred)
    C = ps[["tp", "fn", "fp", "tn"]].to_numpy(float)
    rng = np.random.default_rng(seed)
    W = np.stack([np.bincount(rng.integers(0, len(C), len(C)), minlength=len(C)) for _ in range(nboot)]).astype(float)
    bm = metrics_from_counts(*(W @ C).T)
    return dict(n_silenced_genes=int(len(C)),
                ci95_accuracy=r5(pci(bm["accuracy"])),
                ci95_acc_minus_best_constant=r5(pci(bm["acc_minus_best_constant"])),
                ci95_balanced_acc_minus_half=r5(pci(bm["balanced_acc"] - 0.5)),
                ci95_mcc=r5(pci(bm["mcc"])))


def summary(df, pred, ci=True):
    d = scal(metrics_from_counts(*conf_counts(pred, df["obs_dec"])))
    d["n_silenced_genes"] = int(df["silenced_gene"].nunique())
    if ci:
        d.update(gene_boot_ci(df, pred))
    return d


def within_gene_shuffle_null(s, seed, nperm=2000):
    """MCC of the method's prediction after shuffling it over target genes within each silenced gene.
    Keeps each silenced gene's share of predicted decreases and its observed labels; removes any
    pair-specific information."""
    rng = np.random.default_rng(seed)
    p = s.pred_dec.to_numpy(); o = s.obs_dec.to_numpy()
    idx = list(s.groupby("silenced_gene").indices.values())
    out = np.empty(nperm)
    for b in range(nperm):
        q = p.copy()
        for ii in idx:
            if len(ii) > 1:
                q[ii] = p[rng.permutation(ii)]
        out[b] = float(metrics_from_counts(*conf_counts(q, o))["mcc"])
    return out


def cmh_test(s):
    """Cochran-Mantel-Haenszel test of prediction x observation, stratified by silenced gene
    (continuity-corrected chi-square, 1 df) and the Mantel-Haenszel common odds ratio."""
    from scipy.stats import chi2
    A = EA = VA = num = den = 0.0
    for _, x in s.groupby("silenced_gene"):
        n = len(x)
        if n < 2:
            continue
        p = x.pred_dec.to_numpy(); o = x.obs_dec.to_numpy()
        a = float((p & o).sum()); b = float((p & ~o).sum()); c = float((~p & o).sum()); d = float((~p & ~o).sum())
        num += a * d / n; den += b * c / n
        A += a; EA += (a + b) * (a + c) / n
        VA += (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1))
    stat = (abs(A - EA) - 0.5) ** 2 / VA
    return (num / den if den > 0 else np.nan), float(chi2.sf(stat, 1))


def auroc(y, s):
    y = np.asarray(y, bool); n1 = int(y.sum()); n0 = len(y) - n1
    r = rankdata(s, method="average")
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


class WeightedAUROC:
    """AUROC under integer re-weighting of a fixed (y, s) set (bootstrap replication), tie-safe."""

    def __init__(self, y, s):
        order = np.argsort(s, kind="mergesort"); ss = np.asarray(s)[order]
        self.order = order; self.y = np.asarray(y, bool)[order]
        self.starts = np.r_[0, np.flatnonzero(ss[1:] != ss[:-1]) + 1]

    def __call__(self, w=None):
        w = np.ones(len(self.y)) if w is None else np.asarray(w, float)[self.order]
        wp = np.where(self.y, w, 0.0); wn = w - wp
        gp = np.add.reduceat(wp, self.starts); gn = np.add.reduceat(wn, self.starts)
        before = np.cumsum(gn) - gn
        return float((gp * (before + 0.5 * gn)).sum() / (gp.sum() * gn.sum()))


# ------------------------------------------------------------------ load (all sections)
t0 = time.time()
gp = pd.read_parquet(DATA / "gene_pairs.parquet")
edges = pd.read_csv(DATA / "circuit_edges.csv")
ftg = pd.read_csv(DATA / "feature_top_genes.tsv", sep="\t")
sf = pd.read_csv(DATA / "source_features.tsv", sep="\t")
sg = pd.read_csv(DATA / "silenced_genes.tsv", sep="\t")
Z = np.load(DATA / "knockdown_lfc.npz")
LFC = Z["lfc"]; MEAS = [str(g) for g in Z["measured_genes"]]; SIL = [str(g) for g in Z["silenced_genes"]]
assert SIL == sg.silenced_gene.tolist()
G2V = {g: i for i, g in enumerate(MEAS)}; SROW = {g: i for i, g in enumerate(SIL)}
TOP10 = {(int(l), int(f)): g.split(",") for l, f, g in zip(ftg.layer, ftg.feature, ftg.top10_genes)}
SRC_TOP10 = {(int(l), int(f)): g.split(",") for l, f, g in zip(sf.layer, sf.feature, sf.top10_genes)}
S = sorted(SIL)

df = gp.copy()
df["pred_dec"] = df["predicted_decrease"].astype(bool)
df["obs_dec"] = df["lfc"] < 0
df["frac_inh"] = df["n_inhibitory_evidence"] / df["evidence"]


def build_pairs(ed):
    """The method's rule (data/README.md step 8), vectorised. Returns one row per kept pair."""
    genes = sorted({g for v in TOP10.values() for g in v})
    g_idx = {g: i for i, g in enumerate(genes)}; NG = len(genes)
    keys = list(TOP10.keys()); k_row = {k: i for i, k in enumerate(keys)}
    T = np.array([[g_idx[g] for g in TOP10[k]] for k in keys], np.int64)
    s_idx = {s: i for i, s in enumerate(S)}
    E = np.zeros((len(S), NG), np.int64); NI = np.zeros((len(S), NG), np.int64)
    MX = np.zeros((len(S), NG)); SD = np.zeros((len(S), NG))
    tgt_row = np.array([k_row[(a, b)] for a, b in zip(ed.tgt_layer, ed.tgt_feature)], np.int64)
    inh_all = (ed["sign"].to_numpy() == "inhibitory").astype(np.int64)
    d_all = ed["cohens_d"].to_numpy(float)
    for (sl, sfe), idx in ed.groupby(["src_layer", "src_feature"]).indices.items():
        sgs = [g for g in TOP10[(int(sl), int(sfe))] if g in s_idx]
        if not sgs:
            continue
        flat = T[tgt_row[idx]].ravel()
        inh = np.repeat(inh_all[idx], 10); dd = np.repeat(d_all[idx], 10)
        cnt = np.bincount(flat, minlength=NG)
        ni = np.bincount(flat, weights=inh, minlength=NG).astype(np.int64)
        sd = np.bincount(flat, weights=dd, minlength=NG)
        mx = np.zeros(NG); np.maximum.at(mx, flat, np.abs(dd))
        for g in sgs:
            s = s_idx[g]
            E[s] += cnt; NI[s] += ni; SD[s] += sd; MX[s] = np.maximum(MX[s], mx)
    for g in S:                                     # a gene is never paired with itself
        if g in g_idx:
            E[s_idx[g], g_idx[g]] = 0; NI[s_idx[g], g_idx[g]] = 0; MX[s_idx[g], g_idx[g]] = 0; SD[s_idx[g], g_idx[g]] = 0
    keep = (E >= 2) | (MX > 2.0)
    si, gi = np.nonzero(keep)
    rec = pd.DataFrame({"silenced_gene": np.array(S)[si], "target_gene": np.array(genes)[gi],
                        "evidence": E[si, gi], "n_inhibitory_evidence": NI[si, gi],
                        "max_abs_d": MX[si, gi], "sum_d": SD[si, gi]})
    rec = rec[rec.target_gene.isin(G2V)].copy()
    rec["predicted_decrease"] = rec.n_inhibitory_evidence / rec.evidence > 0.5
    rec["lfc"] = LFC[rec.silenced_gene.map(SROW).to_numpy(), rec.target_gene.map(G2V).to_numpy()].astype(np.float64)
    rec["obs_dec"] = rec["lfc"] < 0
    return rec.sort_values(["silenced_gene", "target_gene"]).reset_index(drop=True)


def target_baseline(d):
    """Cross-fitted per-target direction: majority observed sign of the target gene across the
    silenced genes of the OTHER 4 folds (5 folds of silenced genes; tie/unseen -> training-fold
    global majority). Same folds and rule as the source evaluation."""
    rng = np.random.default_rng(SEED)
    folds = np.array_split(rng.permutation(np.asarray(sorted(d["silenced_gene"].unique()))), 5)
    fold_of = {s: i for i, f in enumerate(folds) for s in f}
    fo = d["silenced_gene"].map(fold_of).to_numpy()
    pred_t = np.zeros(len(d), bool); pred_g = np.zeros(len(d), bool); rows = []
    for k in range(5):
        tr = fo != k; te = ~tr
        glob = d.loc[tr, "obs_dec"].mean() > 0.5
        tmaj = d.loc[tr].groupby("target_gene")["obs_dec"].mean()
        t = d.loc[te, "target_gene"].map(tmaj)
        pred_t[te] = np.where(t.isna() | (t == 0.5), glob, t > 0.5)
        pred_g[te] = glob
        rows.append(dict(fold=k, n_silenced=int(d.loc[te, "silenced_gene"].nunique()), n_pairs=int(te.sum()),
                         circuit_acc=r5((d.loc[te, "pred_dec"] == d.loc[te, "obs_dec"]).mean()),
                         always_dec_acc=r5(d.loc[te, "obs_dec"].mean()),
                         target_baseline_acc=r5((pred_t[te] == d.loc[te, "obs_dec"].to_numpy()).mean())))
    return pred_t, pred_g, fo, rows


def within_gene_baseline(d):
    """Reference only: majority sign of the SAME knockdown on other target folds (uses its labels)."""
    trng = np.random.default_rng(SEED + 1)
    all_t = np.sort(d["target_gene"].unique())
    tfold = dict(zip(all_t, trng.permutation(np.arange(len(all_t)) % 5)))
    x = d[["silenced_gene", "target_gene", "obs_dec"]].copy(); x["tf"] = x.target_gene.map(tfold)
    cnt = x.groupby(["silenced_gene", "tf"])["obs_dec"].agg(["sum", "size"]).reset_index()
    tot = cnt.groupby("silenced_gene")[["sum", "size"]].sum()
    cnt = cnt.join(tot, on="silenced_gene", rsuffix="_tot")
    cnt["tr"] = (cnt["sum_tot"] - cnt["sum"]) / (cnt["size_tot"] - cnt["size"])
    m = x.merge(cnt[["silenced_gene", "tf", "tr"]], on=["silenced_gene", "tf"], how="left")
    return ((m["tr"] > 0.5) | (m["tr"] == 0.5)).to_numpy()


def source_groups(genes, edge_feats):
    """Group silenced genes by the source features (with >= 1 edge) whose top-10 list holds them;
    connected components join genes that share any such feature. Same procedure as the source."""
    member = {s: [] for s in genes}
    for (sl, f), tl in SRC_TOP10.items():
        if (sl, f) not in edge_feats:
            continue
        for g in tl:
            if g in member:
                member[g].append(f"L{sl}F{f}")
    memb = {s: tuple(sorted(set(v))) for s, v in member.items()}
    parent = {s: s for s in memb}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    byfeat = {}
    for s, fs in memb.items():
        for f in fs:
            byfeat.setdefault(f, []).append(s)
    for f, ss in byfeat.items():
        for s in ss[1:]:
            parent[find(s)] = find(ss[0])
    return (pd.Series({s: "|".join(v) for s, v in memb.items()}), pd.Series({s: find(s) for s in memb}),
            pd.Series({s: len(v) for s, v in memb.items()}))


def grouped_bootstrap(ps, groups, seed=SEED, nboot=NBOOT):
    """Resample groups of silenced genes with replacement; pooled metrics from summed counts."""
    gr = groups.reindex(ps.index).to_numpy()
    ug, gi = np.unique(gr, return_inverse=True)
    rng = np.random.default_rng(seed)
    W = np.stack([np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug)) for _ in range(nboot)]).astype(float)[:, gi]
    C = ps[["tp", "fn", "fp", "tn"]].to_numpy(float)
    bm = metrics_from_counts(*(W @ C).T)
    pooled = metrics_from_counts(*C.sum(0))
    tb = ps["tb_correct"].to_numpy(float); nn = ps["n"].to_numpy(float)

    def psm(col):
        v = ps[col].to_numpy(float); ok = ~np.isnan(v)
        return (W[:, ok] @ v[ok]) / W[:, ok].sum(1)
    out = {}

    def add(name, point, reps):
        out[name] = dict(point=r5(point), ci95=r5(pci(reps)))
    add("pooled_accuracy", pooled["accuracy"], bm["accuracy"])
    add("pooled_acc_minus_always_dec", pooled["accuracy"] - pooled["always_dec_acc"], bm["accuracy"] - bm["frac_obs_dec"])
    add("pooled_acc_minus_best_constant_rechosen_each_rep", pooled["acc_minus_best_constant"], bm["acc_minus_best_constant"])
    add("pooled_balanced_acc_minus_half", pooled["balanced_acc"] - 0.5, bm["balanced_acc"] - 0.5)
    add("pooled_mcc", pooled["mcc"], bm["mcc"])
    add("pooled_acc_minus_chance_given_marginals", pooled["acc_minus_chance_given_marginals"], bm["acc_minus_chance_given_marginals"])
    add("pooled_acc_minus_target_direction_baseline", pooled["accuracy"] - tb.sum() / nn.sum(), bm["accuracy"] - (W @ tb) / (W @ nn))
    add("per_gene_mean_accuracy", ps.accuracy.mean(), psm("accuracy"))
    add("per_gene_mean_acc_minus_always_dec", (ps.accuracy - ps.always_dec_acc).mean(), psm("accuracy") - psm("always_dec_acc"))
    add("per_gene_mean_balanced_acc_minus_half", ps.balanced_acc.mean() - 0.5, psm("balanced_acc") - 0.5)
    add("per_gene_mean_mcc", ps.mcc.mean(), psm("mcc"))
    return dict(n_groups=int(len(ug)), n_boot=nboot, seed=seed, stats=out)


# ================================================================== main
if "main" in SECTIONS:
    res: dict = {}
    # ---- 1. file facts
    res["file_facts"] = dict(
        n_edges=int(len(edges)), frac_edges_inhibitory=r5((edges.sign == "inhibitory").mean()),
        edges_by_source_site=edges.src_layer.value_counts().sort_index().to_dict(),
        n_source_features=int(len(sf)), n_source_features_with_edges=int(edges.groupby(["src_layer", "src_feature"]).ngroups),
        n_features_with_top_genes=int(len(ftg)),
        n_gene_pairs=int(len(df)), n_silenced_genes=int(df.silenced_gene.nunique()),
        n_target_genes=int(df.target_gene.nunique()), n_measured_genes=len(MEAS),
        n_lfc_exactly_zero=int((df.lfc == 0).sum()),
        n_ties_frac_half=int((df.frac_inh == 0.5).sum()), frac_ties=r5((df.frac_inh == 0.5).mean()),
        n_unanimous_inhibitory=int((df.frac_inh == 1).sum()), n_unanimous_excitatory=int((df.frac_inh == 0).sum()),
        crispri_cells_min_median_max=[int(sg.n_crispri_cells.min()), float(sg.n_crispri_cells.median()), int(sg.n_crispri_cells.max())],
        pairs_per_silenced_gene_min_median_max=[int(x) for x in df.groupby("silenced_gene").size().describe()[["min", "50%", "max"]]])

    # ---- 2. rebuild the gene pairs from the edges with the method's rule
    rb = build_pairs(edges)
    m = gp.merge(rb, on=["silenced_gene", "target_gene"], how="outer", suffixes=("_pkg", "_rb"), indicator=True)
    both = m[m._merge == "both"]
    res["rebuild_check"] = dict(
        n_pkg=int(len(gp)), n_rebuilt=int(len(rb)), n_only_pkg=int((m._merge == "left_only").sum()),
        n_only_rebuilt=int((m._merge == "right_only").sum()),
        evidence_equal=bool((both.evidence_pkg == both.evidence_rb).all()),
        n_inhibitory_equal=bool((both.n_inhibitory_evidence_pkg == both.n_inhibitory_evidence_rb).all()),
        max_abs_d_max_diff=r5((both.max_abs_d_pkg - both.max_abs_d_rb).abs().max()),
        prediction_equal=bool((both.predicted_decrease_pkg == both.predicted_decrease_rb).all()),
        lfc_equal=bool((both.lfc_pkg == both.lfc_rb).all()))

    # ---- 3. pooled metrics of the method's prediction
    tp, fn, fp, tn = conf_counts(df.pred_dec, df.obs_dec)
    pooled = scal(metrics_from_counts(tp, fn, fp, tn))
    pooled.update(tp_pred_dec_obs_dec=tp, fn_pred_inc_obs_dec=fn, fp_pred_dec_obs_inc=fp, tn_pred_inc_obs_inc=tn)
    res["pooled"] = pooled
    # the metric that ignores the prediction (fraction of pairs that go down)
    res["fraction_of_pairs_observed_decrease"] = r5(df.obs_dec.mean())

    # ---- 4. baselines
    pred_t, pred_g, fo, fold_rows = target_baseline(df)
    df["pred_target"] = pred_t
    pred_w = within_gene_baseline(df)
    res["baselines"] = {
        "always_decrease": summary(df, np.ones(len(df), bool), ci=False),
        "always_increase": summary(df, np.zeros(len(df), bool), ci=False),
        "global_majority_other_folds": summary(df, pred_g, ci=False),
        "target_direction_cross_fitted": summary(df, pred_t),
        "within_knockdown_majority_other_target_folds_reference_only": summary(df, pred_w, ci=False),
    }
    res["folds"] = fold_rows

    # ---- 5. per silenced gene + grouped bootstraps
    ps = per_gene_counts(df, df.pred_dec)
    ps["tb_correct"] = (df.pred_target == df.obs_dec).groupby(df.silenced_gene).sum().reindex(ps.index).to_numpy()
    ps["target_baseline_acc"] = ps.tb_correct / ps.n
    exact, comp, nfeat = source_groups(list(ps.index), set(zip(edges.src_layer, edges.src_feature)))
    ps["group"] = exact.reindex(ps.index).to_numpy(); ps["component"] = comp.reindex(ps.index).to_numpy()
    res["per_gene"] = dict(
        n_genes=int(len(ps)),
        mean_accuracy=r5(ps.accuracy.mean()), mean_always_dec_acc=r5(ps.always_dec_acc.mean()),
        mean_best_constant_acc=r5(ps.best_constant_acc.mean()), mean_balanced_acc=r5(ps.balanced_acc.mean()),
        mean_mcc=r5(ps.mcc.mean()), mean_frac_pred_dec=r5(ps.frac_pred_dec.mean()),
        median_frac_pred_dec=r5(ps.frac_pred_dec.median()), mean_frac_obs_dec=r5(ps.frac_obs_dec.mean()),
        frac_obs_dec_min_max=r5([ps.frac_obs_dec.min(), ps.frac_obs_dec.max()]),
        n_genes_frac_obs_dec_gt_0p6=int((ps.frac_obs_dec > 0.6).sum()), n_genes_frac_obs_dec_lt_0p4=int((ps.frac_obs_dec < 0.4).sum()),
        n_genes_beat_always_dec=int((ps.accuracy > ps.always_dec_acc).sum()),
        n_genes_beat_best_constant=int((ps.accuracy > ps.best_constant_acc).sum()),
        n_genes_balanced_acc_gt_half=int((ps.balanced_acc > 0.5).sum()), n_genes_mcc_gt_0=int((ps.mcc > 0).sum()),
        n_genes_mcc_undefined=int(ps.mcc.isna().sum()),
        n_genes_one_sign_for_all_targets=int(((ps.frac_pred_dec == 0) | (ps.frac_pred_dec == 1)).sum()),
        n_genes_beat_target_baseline=int((ps.accuracy > ps.target_baseline_acc).sum()),
        spearman_gene_pred_share_vs_obs_share=r5(spearmanr(ps.frac_pred_dec, ps.frac_obs_dec)[0]),
        spearman_gene_pred_share_vs_obs_share_p=r5(spearmanr(ps.frac_pred_dec, ps.frac_obs_dec)[1]),
        n_source_feature_groups=int(ps.group.nunique()), n_components=int(ps.component.nunique()),
        largest_component=int(ps.component.value_counts().iloc[0]),
        n_genes_in_more_than_one_source_feature=int((nfeat > 1).sum()))
    res["bootstrap_by_silenced_gene"] = grouped_bootstrap(ps, pd.Series(ps.index, index=ps.index))
    res["bootstrap_by_source_feature_group"] = grouped_bootstrap(ps, ps.group)
    res["bootstrap_by_component"] = grouped_bootstrap(ps, ps.component)

    # ---- 6. bands of |LFC| (source report section 6)
    bands = []
    a = df.lfc.abs()
    for lo, hi in [(0, 0.01), (0.01, 0.025), (0.025, 0.05), (0.05, 0.1), (0.1, 0.25), (0.25, np.inf)]:
        mm = ((a >= lo) & (a < hi)).to_numpy()
        d = summary(df[mm], df.pred_dec[mm]); d.update(abs_lfc_lo=lo, abs_lfc_hi=(None if hi == np.inf else hi))
        bands.append(d)
    res["bands_abs_lfc"] = bands

    # ---- 7. pair-level tests (treat the 698,624 pairs as independent; shown for contrast)
    n = len(df); acc = (df.pred_dec == df.obs_dec).mean()
    z = (acc - 0.5) / np.sqrt(0.25 / n)
    se = np.sqrt(acc * (1 - acc) / n)
    prec = tp / (tp + fp); zp = (prec - 0.5) / np.sqrt(0.25 / (tp + fp))
    res["pair_level_naive"] = dict(
        accuracy_vs_half_z=r5(z), accuracy_vs_half_p_two_sided=float(2 * norm.sf(abs(z))),
        accuracy_ci95_pairs_independent=r5([acc - 1.96 * se, acc + 1.96 * se]),
        precision_pred_dec=r5(prec), precision_pred_dec_vs_half_z=r5(zp),
        precision_pred_dec_vs_half_p_two_sided=float(2 * norm.sf(abs(zp))),
        frac_obs_dec_vs_half_z=r5((df.obs_dec.mean() - 0.5) / np.sqrt(0.25 / n)))
    pb = np.random.default_rng(SEED + 7); corr = (df.pred_dec == df.obs_dec).to_numpy()
    res["pair_level_naive"]["accuracy_ci95_pair_bootstrap"] = r5(pci([corr[pb.integers(0, n, n)].mean() for _ in range(200)]))

    # ---- 8. AUROC of the support fraction (continuous score), gene bootstrap
    y = df.obs_dec.to_numpy()
    sil_codes, sil_idx = np.unique(df.silenced_gene.to_numpy(), return_inverse=True)
    out_auc = {}
    for name, s in [("frac_inhibitory_evidence", df.frac_inh.to_numpy()),
                    ("signed_support_frac_times_max_abs_d", (df.frac_inh.to_numpy() - 0.5) * df.max_abs_d.to_numpy()),
                    ("target_baseline_binary", df.pred_target.to_numpy().astype(float))]:
        f = WeightedAUROC(y, s); rng = np.random.default_rng(SEED + 11); reps = []
        for _ in range(500):
            w = np.bincount(rng.integers(0, len(sil_codes), len(sil_codes)), minlength=len(sil_codes))[sil_idx]
            reps.append(f(w))
        per_gene = df.assign(_s=s).groupby("silenced_gene").apply(
            lambda g: auroc(g.obs_dec, g._s) if 0 < g.obs_dec.sum() < len(g) else np.nan, include_groups=False)
        out_auc[name] = dict(pooled_auroc=r5(f()), ci95_gene_bootstrap_500=r5(pci(reps)),
                             mean_per_gene_auroc=r5(per_gene.mean()))
    res["auroc"] = out_auc
    # magnitude: Spearman |LFC| vs max_abs_d (source paper's magnitude check), pooled
    res["magnitude_spearman_abs_lfc_vs_max_abs_d"] = r5(spearmanr(df.lfc.abs(), df.max_abs_d)[0])
    res["seconds"] = round(time.time() - t0, 1)
    save("main", res)
    print(json.dumps({k: res[k] for k in ["rebuild_check", "pooled", "fraction_of_pairs_observed_decrease"]}, indent=1))
    print(json.dumps(res["baselines"]["target_direction_cross_fitted"], indent=1))
    print(json.dumps(res["bootstrap_by_silenced_gene"]["stats"], indent=1))
    print("seconds", res["seconds"])


# ================================================================== variants (analyst choices)
if "variants" in SECTIONS:
    res = {}
    rb = build_pairs(edges)
    assert len(rb) == len(df) and (rb.silenced_gene.to_numpy() == df.silenced_gene.to_numpy()).all() \
        and (rb.target_gene.to_numpy() == df.target_gene.to_numpy()).all()
    df["sum_d"] = rb["sum_d"].to_numpy()
    a = df.lfc.abs()
    own = pd.Series({g: float(LFC[SROW[g], G2V[g]]) for g in SIL})
    res["on_target_lfc"] = dict(n_negative=int((own < 0).sum()), median=r5(own.median()),
                                quartiles=r5(own.quantile([0.25, 0.75]).tolist()),
                                n_below_minus_0p5=int((own < -0.5).sum()), n_below_minus_1=int((own < -1).sum()))
    strong = set(own[own <= own.median()].index)
    V = {}
    V["method_rule"] = summary(df, df.pred_dec)
    V["ties_as_decrease"] = summary(df, df.frac_inh >= 0.5)
    V["drop_ties"] = summary(df[df.frac_inh != 0.5], df.pred_dec[df.frac_inh != 0.5])
    V["sign_of_summed_d"] = summary(df, df.sum_d < 0)
    mk = (a >= 0.01).to_numpy()
    V["sign_of_summed_d_abs_lfc_ge_0p01"] = summary(df[mk], (df.sum_d < 0)[mk])
    V["method_rule_abs_lfc_ge_0p01"] = summary(df[mk], df.pred_dec[mk])
    for thr in [0.05, 0.1, 0.25, 0.5]:
        mk = (a >= thr).to_numpy()
        V[f"method_rule_abs_lfc_ge_{thr}"] = summary(df[mk], df.pred_dec[mk])
    mk = df.frac_inh.isin([0.0, 1.0]).to_numpy()
    V["unanimous_support_only"] = summary(df[mk], df.pred_dec[mk])
    for e in [5, 10, 20]:
        mk = (df.evidence >= e).to_numpy()
        V[f"evidence_ge_{e}"] = summary(df[mk], df.pred_dec[mk])
    mk = (df.max_abs_d > 2).to_numpy()
    V["max_abs_d_gt_2"] = summary(df[mk], df.pred_dec[mk])
    mk = (df.max_abs_d > 1).to_numpy()
    V["max_abs_d_gt_1"] = summary(df[mk], df.pred_dec[mk])
    mk = df.silenced_gene.isin(strong).to_numpy()
    V["strongest_half_on_target_knockdown"] = summary(df[mk], df.pred_dec[mk])
    for site in sorted(edges.src_layer.unique()):
        r = build_pairs(edges[edges.src_layer == site])
        if len(r):
            V[f"edges_from_source_site_{site}_only"] = summary(r, r.predicted_decrease)
    res["variants"] = V

    # ---- subsets by |LFC| threshold: is any excess pair-specific (within silenced gene)?
    pred_t, _, _, _ = target_baseline(df)
    df["pred_target"] = pred_t
    sub = {}
    for thr in [0.25, 0.4, 0.45, 0.5, 0.6, 0.75]:
        s = df[(a >= thr).to_numpy()]
        p = s.pred_dec.to_numpy(); o = s.obs_dec.to_numpy()
        obs_mcc = float(metrics_from_counts(*conf_counts(p, o))["mcc"])
        null = within_gene_shuffle_null(s, seed=SEED + 21)
        orr, cmh_p = cmh_test(s)
        g = s.groupby("silenced_gene").agg(n=("pred_dec", "size"), ps=("pred_dec", "mean"), os=("obs_dec", "mean"))
        pgm = per_gene_counts(s, s.pred_dec)
        sub[f"abs_lfc_ge_{thr}"] = dict(
            n=int(len(s)), n_silenced_genes=int(s.silenced_gene.nunique()),
            circuit=scal({k: v for k, v in metrics_from_counts(*conf_counts(p, o)).items()
                          if k in ["accuracy", "best_constant_acc", "balanced_acc", "mcc", "frac_pred_dec", "frac_obs_dec"]}),
            target_baseline=scal({k: v for k, v in metrics_from_counts(*conf_counts(s.pred_target, o)).items()
                                  if k in ["accuracy", "balanced_acc", "mcc"]}),
            within_gene_shuffle_null_mcc_mean=r5(np.nanmean(null)), within_gene_shuffle_null_mcc_sd=r5(np.nanstd(null)),
            within_gene_shuffle_p_one_sided=r5((1 + np.sum(null >= obs_mcc)) / (1 + len(null))),
            cmh_odds_ratio_within_gene=r5(orr), cmh_p_two_sided=r5(cmh_p),
            mean_per_gene_mcc=r5(pgm.mcc.mean()), n_genes_mcc_defined=int(pgm.mcc.notna().sum()),
            between_gene_spearman_pred_share_vs_obs_share=r5(spearmanr(g.ps, g.os)[0]),
            between_gene_spearman_p=r5(spearmanr(g.ps, g.os)[1]))
    res["subsets_abs_lfc"] = sub

    # ---- one test over all six thresholds: shuffle predictions within (silenced gene x |LFC| bin),
    # which keeps every threshold subset's per-gene predicted share; statistic = max over thresholds
    # of the MCC z-score against this null.
    thr = [0.25, 0.4, 0.45, 0.5, 0.6, 0.75]
    s = df[(a >= thr[0]).to_numpy()].reset_index(drop=True)
    sa = s.lfc.abs().to_numpy()
    b = np.digitize(sa, thr[1:])                                   # bin 0 = [0.25, 0.4), ..., 5 = >= 0.75
    gid = pd.factorize(s.silenced_gene.astype(str) + "|" + pd.Series(b).astype(str))[0]
    p = s.pred_dec.to_numpy(); o = s.obs_dec.to_numpy()
    masks = [sa >= t for t in thr]
    def mccs(q):
        return np.array([float(metrics_from_counts(*conf_counts(q[m], o[m]))["mcc"]) for m in masks])
    obs = mccs(p)
    pos = np.argsort(gid, kind="stable")
    rng = np.random.default_rng(SEED + 31); NP = 2000
    null = np.empty((NP, len(thr)))
    for k in range(NP):
        order = np.lexsort((rng.random(len(gid)), gid))
        q = np.empty_like(p); q[pos] = p[order]
        null[k] = mccs(q)
    mu, sd = np.nanmean(null, 0), np.nanstd(null, 0)
    zo = (obs - mu) / sd; zn = (null - mu) / sd
    res["threshold_family_test"] = dict(
        thresholds=thr, observed_mcc=r5(obs), null_mean_mcc=r5(mu), null_sd_mcc=r5(sd), z=r5(zo),
        per_threshold_p_one_sided=r5([(1 + np.sum(null[:, j] >= obs[j])) / (1 + NP) for j in range(len(thr))]),
        max_z_observed=r5(np.nanmax(zo)), max_z_p_one_sided=r5((1 + np.sum(np.nanmax(zn, 1) >= np.nanmax(zo))) / (1 + NP)),
        note="null shuffles the method's prediction within silenced gene x |LFC| bin; 2,000 draws")

    # ---- other definitions of the target-direction baseline (no circuit)
    grp = df.groupby("target_gene").obs_dec.agg(["sum", "size"])
    x = df.join(grp, on="target_gene")
    loo = (x["sum"] - x.obs_dec) / (x["size"] - 1)
    glob = df.obs_dec.mean() > 0.5
    D = (LFC < 0).astype(float)
    col = df.target_gene.map(G2V).to_numpy(); row = df.silenced_gene.map(SROW).to_numpy()
    loo_m = (D.sum(0)[col] - D[row, col]) / (D.shape[0] - 1)
    mean_m = (LFC.astype(float).sum(0)[col] - LFC[row, col]) / (LFC.shape[0] - 1)
    res["target_baseline_alternatives"] = {
        "cross_fitted_5_folds_pairs": summary(df, df.pred_target, ci=False),
        "leave_one_silenced_gene_out_pairs": summary(df, np.where((x["size"] - 1 == 0) | (loo == 0.5), glob, loo > 0.5), ci=False),
        "leave_one_silenced_gene_out_full_lfc_matrix": summary(df, np.where(loo_m == 0.5, glob, loo_m > 0.5), ci=False),
        "leave_one_out_mean_lfc_full_matrix": summary(df, mean_m < 0, ci=False)}

    # range of balanced accuracy / MCC / acc - best constant over all variants
    res["range_over_variants"] = {k: r5([min(v[k] for v in V.values() if v[k] is not None),
                                        max(v[k] for v in V.values() if v[k] is not None)])
                                  for k in ["accuracy", "balanced_acc", "mcc", "acc_minus_best_constant", "frac_pred_dec"]}
    res["variants_with_ci_balanced_acc_above_half"] = [k for k, v in V.items() if v["ci95_balanced_acc_minus_half"][0] > 0]
    res["variants_with_ci_mcc_above_0"] = [k for k, v in V.items() if v["ci95_mcc"][0] > 0]
    res["variants_with_accuracy_above_half"] = [k for k, v in V.items() if v["accuracy"] > 0.5]
    res["variants_with_accuracy_above_best_constant"] = [k for k, v in V.items() if v["acc_minus_best_constant"] > 0]
    res["seconds"] = round(time.time() - t0, 1)
    save("variants", res)
    print(json.dumps({k: {kk: v[kk] for kk in ["n", "frac_pred_dec", "frac_obs_dec", "accuracy", "balanced_acc", "mcc",
                                                "ci95_balanced_acc_minus_half", "ci95_mcc", "n_silenced_genes"]}
                      for k, v in V.items()}, indent=0))
    print(json.dumps({k: res[k] for k in res if k != "variants"}, indent=1))


# ================================================================== within (association inside each knockdown)
if "within" in SECTIONS:
    res = {}
    col = df.target_gene.map(G2V).to_numpy(); row = df.silenced_gene.map(SROW).to_numpy()
    Lf = LFC.astype(float); D = (Lf < 0).astype(float)
    # target gene's usual change over the OTHER 226 silenced genes (full LFC matrix, no circuit)
    df["tgt_loo_frac_dec"] = (D.sum(0)[col] - D[row, col]) / (D.shape[0] - 1)
    df["tgt_loo_mean_lfc"] = (Lf.sum(0)[col] - Lf[row, col]) / (Lf.shape[0] - 1)
    df["tb_loo"] = df.tgt_loo_frac_dec > 0.5
    df["neg_lfc"] = -df.lfc
    df["neg_resid"] = -(df.lfc - df.tgt_loo_mean_lfc)
    orr, p = cmh_test(df)
    s2 = df.assign(silenced_gene=df.silenced_gene + "|" + df.tb_loo.astype(str))
    orr2, p2 = cmh_test(s2)
    res["cmh_all_pairs"] = dict(
        by_silenced_gene=dict(odds_ratio=r5(orr), p_two_sided=r5(p)),
        by_silenced_gene_x_target_usual_direction=dict(odds_ratio=r5(orr2), p_two_sided=r5(p2)),
        note="pairs treated as independent inside each stratum; target usual direction = majority sign of the target over the other 226 silenced genes (full LFC matrix)")
    sub = {}
    for thr in [0.25, 0.45, 0.5]:
        s = df[(df.lfc.abs() >= thr).to_numpy()]
        o1, q1 = cmh_test(s)
        o2, q2 = cmh_test(s.assign(silenced_gene=s.silenced_gene + "|" + s.tb_loo.astype(str)))
        sub[f"abs_lfc_ge_{thr}"] = dict(by_silenced_gene=[r5(o1), r5(q1)], by_silenced_gene_x_target_usual_direction=[r5(o2), r5(q2)])
    res["cmh_subsets_odds_ratio_p"] = sub

    def per_gene_spearman(y):
        return df.groupby("silenced_gene").apply(
            lambda g: spearmanr(g.frac_inh, g[y])[0] if len(g) > 10 and g.frac_inh.nunique() > 1 else np.nan,
            include_groups=False).dropna()
    exact, comp, _ = source_groups(sorted(df.silenced_gene.unique()), set(zip(edges.src_layer, edges.src_feature)))

    def cboot(v, groups, seed):
        rng = np.random.default_rng(seed)
        ug, gi = np.unique(groups.reindex(v.index).to_numpy(), return_inverse=True)
        out = [(lambda w: (w * v.to_numpy()).sum() / w.sum())(np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[gi])
               for _ in range(NBOOT)]
        return r5(pci(out))
    for name, y in [("frac_inhibitory_vs_minus_lfc", "neg_lfc"),
                    ("frac_inhibitory_vs_minus_lfc_after_removing_target_usual_change", "neg_resid")]:
        v = per_gene_spearman(y)
        res[f"per_gene_spearman_{name}"] = dict(
            mean=r5(v.mean()), n_genes=int(len(v)), frac_positive=r5((v > 0).mean()),
            ci95_by_silenced_gene=cboot(v, pd.Series(v.index, index=v.index), SEED + 41),
            ci95_by_source_feature_group=cboot(v, exact, SEED + 41),
            ci95_by_component=cboot(v, comp, SEED + 41))
    t = df.groupby("target_gene").frac_inh.mean()
    tm = pd.Series(-Lf.mean(0), index=MEAS).reindex(t.index)
    rho = spearmanr(t.to_numpy(), tm.to_numpy())
    res["target_level_spearman_mean_frac_inhibitory_vs_mean_minus_lfc"] = dict(rho=r5(rho[0]), p=float(rho[1]), n_targets=int(len(t)))
    res["seconds"] = round(time.time() - t0, 1)
    save("within", res)
    print(json.dumps(res, indent=1))
