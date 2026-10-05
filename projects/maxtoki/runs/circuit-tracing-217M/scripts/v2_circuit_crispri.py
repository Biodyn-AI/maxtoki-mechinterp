"""v2 circuit tracing, part 4: CRISPRi directional evaluation with proper baselines (item D2).

Records are built EXACTLY as the corrected deployed metric
(scripts/audit_a2_groupkfold_crispri.py:51-162), for two edge sets:
  "v2"       outputs/v2_circuit/circuit_edges_v2.csv   (fixed hooks)
  "deployed" outputs/circuit_edges.csv                 (deployed, block-deleting hook)
Rule:
  * use every edge whose two endpoints have a top-gene list (feature_catalog.json);
  * pair each of the 10 top genes of the source feature with each of the 10 top
    genes of the target feature (upper case; self pairs skipped; duplicates in a
    list counted as in the script);
  * per gene pair: evidence = number of such (edge, gene, gene) triples,
    max_abs_d = max |Cohen's d|, n_inhib = number from "inhibitory" edges;
  * keep a pair if evidence >= 2 or max_abs_d > 2.0;
  * predicted decrease ("predicted_inhibitory") = n_inhib / evidence > 0.5 (tie -> increase);
  * keep a pair only if the source gene has >= 10 K562 CRISPRi cells and the
    target gene is a column of the expression matrix;
  * observed decrease ("actual_inhibitory") = LFC < 0 (LFC from v2_circuit_lfc.py,
    verified identical to the deployed records).
Metrics (positive class = observed decrease), on the identical records per edge set:
  class balance, accuracy, always-decrease / always-increase / majority-class
  accuracy, balanced accuracy, MCC, Cohen's kappa, accuracy expected from the two
  marginal rates, cross-fitted per-target direction baseline (5 folds of silenced
  genes), within-source baseline (5 target folds, uses the same knockdown's labels),
  per-silenced-gene averages, grouped bootstrap 95% CIs (2,000 reps):
    (a) resampling silenced genes, (b) resampling source-feature membership groups,
    (c) resampling connected components of silenced genes that share a source feature.
Writes outputs/v2_circuit/per_pair.parquet and outputs/v2_circuit/crispri/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJ / "setup"))
import hooks_v2 as H  # noqa: E402

RUN = PROJ / "runs/circuit-tracing-217M"
V2 = RUN / "outputs/v2_circuit"
OUT = V2 / "crispri"
PHASE2 = PROJ / "runs/sae-atlas-217M/outputs/phase2"
LFC_DIR = V2 / "lfc"
OLD_PP = RUN / "outputs/groupkfold_crispri/per_pair.parquet"
INV = PROJ / "verification/crispri/results.json"
EDGE_FILES = {"v2": V2 / "circuit_edges_v2.csv", "deployed": RUN / "outputs/circuit_edges.csv"}
SEED = 42
NBOOT = 2000


# ============================================================================ loading
def load_top10():
    top10, labelled = {}, set()
    for li in range(12):
        d = PHASE2 / f"layer_{li:02d}"
        for c in json.load(open(d / "feature_catalog.json")):
            top10[(li, int(c["feature_id"]))] = [g.upper() for g in c["top20_genes"][:10]]
        labelled |= {(li, int(f)) for f in pd.read_csv(d / "significant_enrichments.csv", usecols=["feature_id"]).feature_id.unique()}
    return top10, labelled


def load_lfc():
    src = pd.read_csv(LFC_DIR / "sources.csv")
    src = src[src.has_data].reset_index(drop=True)
    lfc = np.load(LFC_DIR / "lfc_matrix.npy")
    var_upper = (LFC_DIR / "var_symbols.txt").read_text().split("\n")
    gene_to_var = {s: i for i, s in enumerate(var_upper)}       # last occurrence wins, as deployed
    return src, lfc, gene_to_var


# ============================================================================ records
def build_records(edges, top10, S, lfc_rows, lfc, gene_to_var, edge_mask=None):
    """Vectorised version of audit_a2 lines 51-162 (checked against the deployed
    per_pair.parquet: same pairs, same predicted sign, same LFC)."""
    genes = sorted({g for v in top10.values() for g in v})
    g_idx = {g: i for i, g in enumerate(genes)}
    G = len(genes)
    keys = list(top10.keys()); k_row = {k: i for i, k in enumerate(keys)}
    T = np.full((len(keys), 10), -1, np.int64)
    for k, v in top10.items():
        T[k_row[k], :len(v)] = [g_idx[g] for g in v]
    s_idx = {s: i for i, s in enumerate(S)}
    ed = edges if edge_mask is None else edges[edge_mask]
    has = np.array([(a, b) in top10 and (c, d) in top10 for a, b, c, d in
                    zip(ed.src_layer, ed.src_feature, ed.tgt_layer, ed.tgt_feature)], bool)
    ed = ed[has]
    E = np.zeros((len(S), G), np.int64); NI = np.zeros((len(S), G), np.int64)
    MX = np.zeros((len(S), G)); SD = np.zeros((len(S), G))
    tgt_row = np.array([k_row[(a, b)] for a, b in zip(ed.tgt_layer, ed.tgt_feature)], np.int64)
    inh_all = (ed["sign"].to_numpy() == "inhibitory").astype(np.int64)
    d_all = ed["cohens_d"].to_numpy(float)
    for (sl, sf), idx in ed.groupby(["src_layer", "src_feature"]).indices.items():
        sgs = [g for g in top10.get((sl, sf), []) if g in s_idx]
        if not sgs:
            continue
        ids = T[tgt_row[idx]]
        ok = ids >= 0
        flat = ids[ok]
        inh = np.repeat(inh_all[idx][:, None], 10, 1)[ok]
        dd = np.repeat(d_all[idx][:, None], 10, 1)[ok]
        cnt = np.bincount(flat, minlength=G)
        ni = np.bincount(flat, weights=inh, minlength=G).astype(np.int64)
        sd = np.bincount(flat, weights=dd, minlength=G)
        mx = np.zeros(G); np.maximum.at(mx, flat, np.abs(dd))
        for sg in sgs:
            s = s_idx[sg]
            E[s] += cnt; NI[s] += ni; SD[s] += sd; MX[s] = np.maximum(MX[s], mx)
    for sg in S:
        if sg in g_idx:
            E[s_idx[sg], g_idx[sg]] = 0; NI[s_idx[sg], g_idx[sg]] = 0; MX[s_idx[sg], g_idx[sg]] = 0; SD[s_idx[sg], g_idx[sg]] = 0
    keep = (E >= 2) | (MX > 2.0)
    si, gi = np.nonzero(keep)
    rec = pd.DataFrame({"source": np.array(S)[si], "target": np.array(genes)[gi], "evidence": E[si, gi],
                        "n_inhibitory": NI[si, gi], "max_abs_d": MX[si, gi], "sum_d": SD[si, gi]})
    rec["frac_inhibitory"] = rec.n_inhibitory / rec.evidence
    rec["predicted_inhibitory"] = rec.frac_inhibitory > 0.5
    col = rec.target.map(gene_to_var)
    rec = rec[col.notna()].copy()
    rec["actual_lfc"] = lfc[rec.source.map(lfc_rows).to_numpy().astype(int), rec.target.map(gene_to_var).to_numpy().astype(int)].astype(np.float64)
    rec["actual_inhibitory"] = rec.actual_lfc < 0
    rec["correct"] = rec.predicted_inhibitory == rec.actual_inhibitory
    return rec.sort_values(["source", "target"]).reset_index(drop=True), int(has.sum())


# ============================================================================ metrics
def conf_counts(pred, obs):
    pred = np.asarray(pred, bool); obs = np.asarray(obs, bool)
    return int((pred & obs).sum()), int((~pred & obs).sum()), int((pred & ~obs).sum()), int((~pred & ~obs).sum())


def metrics_from_counts(tp, fn, fp, tn):
    """Same formulas as verification/crispri/analyze_crispri.py. Positive = observed decrease."""
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
    return dict(n=n, frac_obs_dec=p_obs, frac_obs_inc=1 - p_obs, frac_pred_dec=p_pred, frac_pred_inc=1 - p_pred,
                accuracy=acc, always_dec_acc=p_obs, always_inc_acc=1 - p_obs,
                majority_acc=np.maximum(p_obs, 1 - p_obs), acc_minus_best_constant=acc - np.maximum(p_obs, 1 - p_obs),
                recall_dec=tpr, recall_inc=tnr, balanced_acc=bal, mcc=mcc,
                chance_acc_given_marginals=chance, acc_minus_chance_given_marginals=acc - chance, cohen_kappa=kappa)


def scal(d):
    return {k: (float(v) if np.ndim(v) == 0 else v) for k, v in d.items()}


def per_source_table(df, pred_col="pred_dec"):
    g = df.assign(_p=df[pred_col].astype(bool), _o=df["obs_dec"].astype(bool))
    ps = pd.DataFrame({
        "tp": (g._p & g._o).groupby(g.source).sum(),
        "fn": (~g._p & g._o).groupby(g.source).sum(),
        "fp": (g._p & ~g._o).groupby(g.source).sum(),
        "tn": (~g._p & ~g._o).groupby(g.source).sum(),
    }).astype(int).sort_index()
    for k, v in metrics_from_counts(ps.tp, ps.fn, ps.fp, ps.tn).items():
        ps[k] = np.asarray(v)
    return ps


def add_target_baseline(df, folds):
    """Cross-fitted per-target direction: majority observed sign of the target
    gene across the silenced genes in the OTHER four folds (analyze_crispri.py 4a)."""
    fold_of_src = {s: i for i, f in enumerate(folds) for s in f}
    df["src_fold"] = df["source"].map(fold_of_src)
    df["pred_target_major"] = False
    df["pred_global_major"] = False
    rows = []
    for k in range(len(folds)):
        tr = df["src_fold"] != k; te = ~tr
        glob = df.loc[tr, "obs_dec"].mean() > 0.5
        tmaj = df.loc[tr].groupby("target")["obs_dec"].mean()
        t = df.loc[te, "target"].map(tmaj)
        df.loc[te, "pred_target_major"] = np.where(t.isna() | (t == 0.5), glob, t > 0.5)
        df.loc[te, "pred_global_major"] = glob
        sub = df.loc[te]
        rows.append(dict(fold=k, n_sources=int(sub.source.nunique()), n_pairs=int(len(sub)),
                         model_acc=float((sub.pred_dec == sub.obs_dec).mean()),
                         target_majority_acc=float((sub.pred_target_major == sub.obs_dec).mean()),
                         global_majority_acc=float((sub.pred_global_major == sub.obs_dec).mean()),
                         train_majority_is_decrease=bool(glob),
                         n_test_targets_unseen_in_train=int(t.isna().sum())))
    return rows


def add_within_source_baseline(df):
    trng = np.random.default_rng(SEED + 1)
    all_t = np.sort(df["target"].unique())
    tfold = dict(zip(all_t, trng.permutation(np.arange(len(all_t)) % 5)))
    df["tgt_fold"] = df["target"].map(tfold)
    cnt = df.groupby(["source", "tgt_fold"])["obs_dec"].agg(["sum", "size"]).reset_index()
    tot = cnt.groupby("source")[["sum", "size"]].sum()
    cnt = cnt.join(tot, on="source", rsuffix="_tot")
    cnt["train_frac_dec"] = (cnt["sum_tot"] - cnt["sum"]) / (cnt["size_tot"] - cnt["size"])
    m = df.merge(cnt[["source", "tgt_fold", "train_frac_dec"]], on=["source", "tgt_fold"], how="left")
    pred = (m["train_frac_dec"] > 0.5) | (m["train_frac_dec"] == 0.5)
    df["pred_source_major"] = pred.to_numpy()


def source_groups(sources, top10, feats, edge_feats=None):
    """Group silenced genes by the source features whose top-10 list holds them.
    Only source features with at least one edge in the scored edge set count
    (edge_feats); a feature without edges gives no prediction."""
    member = {s: [] for s in sources}
    for s_layer, fl in feats.items():
        for f in fl:
            if edge_feats is not None and (int(s_layer), int(f)) not in edge_feats:
                continue
            for g in top10[(int(s_layer), f)]:
                if g in member:
                    member[g].append(f"L{s_layer}F{f}")
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
    exact = pd.Series({s: "|".join(v) for s, v in memb.items()})
    comp = pd.Series({s: find(s) for s in memb})
    nfeat = pd.Series({s: len(v) for s, v in memb.items()})
    return exact, comp, nfeat


def bootstrap(df, ps, group_of_source, rng_seed, nboot=NBOOT):
    """Resample groups of silenced genes with replacement; pooled metrics from
    summed confusion counts; per-source means weight each source by its count."""
    groups = group_of_source.reindex(ps.index).to_numpy()
    ug, gi = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(rng_seed)
    Wg = np.stack([np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug)) for _ in range(nboot)]).astype(float)
    W = Wg[:, gi]                                                   # (B, S): weight of each source
    C = ps[["tp", "fn", "fp", "tn"]].to_numpy(float)
    BC = W @ C
    bm = metrics_from_counts(BC[:, 0], BC[:, 1], BC[:, 2], BC[:, 3])
    pobs = bm["frac_obs_dec"]
    tb = ps["_tb_correct"].to_numpy(float); nn = ps["n"].to_numpy(float)

    def psm(col):
        v = ps[col].to_numpy(float); ok = ~np.isnan(v)
        return (W[:, ok] @ v[ok]) / W[:, ok].sum(1)
    out = {}

    def add(name, point, reps):
        reps = np.asarray(reps, float)
        out[name] = dict(point=float(point), ci95=[float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))],
                         frac_reps_gt_0=float((reps > 0).mean()))
    pooled = metrics_from_counts(*C.sum(0))
    add("pooled_accuracy", pooled["accuracy"], bm["accuracy"])
    add("pooled_acc_minus_always_dec", pooled["accuracy"] - pooled["always_dec_acc"], bm["accuracy"] - pobs)
    add("pooled_acc_minus_best_constant_rechosen_each_rep", pooled["acc_minus_best_constant"], bm["acc_minus_best_constant"])
    add("pooled_balanced_acc_minus_0.5", pooled["balanced_acc"] - 0.5, bm["balanced_acc"] - 0.5)
    add("pooled_mcc", pooled["mcc"], bm["mcc"])
    add("pooled_acc_minus_chance_given_marginals", pooled["acc_minus_chance_given_marginals"], bm["acc_minus_chance_given_marginals"])
    tb_b = (W @ tb) / (W @ nn)
    add("pooled_acc_minus_target_direction_baseline", pooled["accuracy"] - tb.sum() / nn.sum(), bm["accuracy"] - tb_b)
    add("per_source_mean_accuracy", ps.accuracy.mean(), psm("accuracy"))
    add("per_source_mean_acc_minus_always_dec", (ps.accuracy - ps.always_dec_acc).mean(), psm("accuracy") - psm("always_dec_acc"))
    add("per_source_mean_balanced_acc_minus_0.5", ps.balanced_acc.mean() - 0.5, psm("balanced_acc") - 0.5)
    add("per_source_mean_mcc", ps.mcc.mean(), psm("mcc"))
    return dict(n_groups=int(len(ug)), n_boot=nboot, seed=rng_seed, stats=out)


def stratum_ci_by_silenced_gene(sub, nboot=NBOOT, seed=SEED):
    """Grouped bootstrap (silenced genes) of balanced accuracy - 0.5, MCC and
    accuracy - best constant inside one |LFC| band."""
    ps = per_source_table(sub)
    C = ps[["tp", "fn", "fp", "tn"]].to_numpy(float)
    rng = np.random.default_rng(seed)
    W = np.stack([np.bincount(rng.integers(0, len(C), len(C)), minlength=len(C)) for _ in range(nboot)]).astype(float)
    bm = metrics_from_counts(*(W @ C).T)
    ci = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]
    return dict(ci95_balanced_acc_minus_half=ci(bm["balanced_acc"] - 0.5), ci95_mcc=ci(bm["mcc"]),
                ci95_acc_minus_best_constant=ci(bm["acc_minus_best_constant"]))


def evaluate(rec, name, folds_order, top10, feats, edge_feats=None):
    df = rec.copy()
    df["pred_dec"] = df["predicted_inhibitory"].astype(bool)
    df["obs_dec"] = df["actual_inhibitory"].astype(bool)
    assert (df.obs_dec == (df.actual_lfc < 0)).all()
    res = {"edge_set": name}
    tp, fn, fp, tn = conf_counts(df.pred_dec, df.obs_dec)
    pooled = scal(metrics_from_counts(tp, fn, fp, tn))
    pooled.update(tp_pred_dec_obs_dec=tp, fn_pred_inc_obs_dec=fn, fp_pred_dec_obs_inc=fp, tn_pred_inc_obs_inc=tn,
                  n_sources=int(df.source.nunique()), n_targets=int(df.target.nunique()),
                  n_lfc_exactly_zero=int((df.actual_lfc == 0).sum()),
                  n_ties_frac_inhib_half=int((df.get("frac_inhibitory", pd.Series(dtype=float)) == 0.5).sum()))
    res["pooled"] = pooled
    # cross-fitted baselines
    rng = np.random.default_rng(SEED)
    src_perm = rng.permutation(np.asarray(folds_order))
    folds = np.array_split(src_perm, 5)
    res["folds"] = add_target_baseline(df, folds)
    add_within_source_baseline(df)
    base = []
    for bname, pred, note in [
            ("model (majority sign of SAE edges)", df.pred_dec, "uses no CRISPRi labels"),
            ("always decrease", np.ones(len(df), bool), "constant"),
            ("always increase", np.zeros(len(df), bool), "constant"),
            ("global majority, training source folds", df.pred_global_major, "5 folds of silenced genes"),
            ("per-target direction, training source folds", df.pred_target_major,
             "target gene's usual direction across the silenced genes of the other 4 folds"),
            ("per-source majority, other target folds", df.pred_source_major,
             "uses labels of the SAME knockdown; reference only")]:
        d = scal(metrics_from_counts(*conf_counts(pred, df.obs_dec)))
        d.update(predictor=bname, note=note)
        base.append(d)
    res["baselines"] = base
    # per source
    ps = per_source_table(df)
    tbc = (df.pred_target_major == df.obs_dec).groupby(df.source).sum().reindex(ps.index)
    ps["_tb_correct"] = tbc.to_numpy()
    ps["target_baseline_acc"] = ps["_tb_correct"] / ps["n"]
    exact, comp, nfeat = source_groups(list(ps.index), top10, feats, edge_feats)
    ps["source_group"] = exact.reindex(ps.index).to_numpy()
    ps["source_component"] = comp.reindex(ps.index).to_numpy()
    ps["n_source_features"] = nfeat.reindex(ps.index).to_numpy()
    res["per_source_summary"] = dict(
        n_sources=int(len(ps)),
        **{f"mean_{c}": float(ps[c].mean()) for c in ["accuracy", "always_dec_acc", "majority_acc", "balanced_acc", "mcc",
                                                      "frac_obs_dec", "frac_pred_dec", "target_baseline_acc"]},
        **{f"median_{c}": float(ps[c].median()) for c in ["accuracy", "balanced_acc", "mcc", "frac_pred_dec"]},
        n_sources_acc_gt_always_dec=int((ps.accuracy > ps.always_dec_acc).sum()),
        n_sources_balanced_acc_gt_half=int((ps.balanced_acc > 0.5).sum()),
        n_sources_mcc_gt_0=int((ps.mcc > 0).sum()),
        n_sources_mcc_nan=int(ps.mcc.isna().sum()),
        n_sources_model_beats_target_baseline=int((ps.accuracy > ps.target_baseline_acc).sum()),
        n_sources_all_pred_same_sign=int(((ps.frac_pred_dec == 0) | (ps.frac_pred_dec == 1)).sum()),
        n_exact_membership_groups=int(ps.source_group.nunique()),
        n_connected_components=int(ps.source_component.nunique()),
        largest_component=int(ps.source_component.value_counts().iloc[0]),
    )
    # bootstraps
    res["bootstrap_by_silenced_gene"] = bootstrap(df, ps, pd.Series(ps.index, index=ps.index), SEED)
    res["bootstrap_by_source_feature_group"] = bootstrap(df, ps, ps["source_group"], SEED)
    res["bootstrap_by_component"] = bootstrap(df, ps, ps["source_component"], SEED)
    # strata by |LFC|
    strata = []
    a = df.actual_lfc.abs()
    for lo, hi in [(0, 0.01), (0.01, 0.025), (0.025, 0.05), (0.05, 0.1), (0.1, 0.25), (0.25, np.inf)]:
        m = (a >= lo) & (a < hi)
        if m.sum() == 0:
            continue
        d = scal(metrics_from_counts(*conf_counts(df.pred_dec[m], df.obs_dec[m])))
        d.update(abs_lfc_lo=lo, abs_lfc_hi=hi, n_sources=int(df.source[m].nunique()))
        d.update(stratum_ci_by_silenced_gene(df[m]))
        strata.append(d)
    res["strata_by_abs_lfc"] = strata
    return res, df, ps


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    top10, labelled = load_top10()
    feats = json.loads((V2 / "source_features.json").read_text())
    src, lfc, gene_to_var = load_lfc()
    S = sorted(src.source)
    lfc_rows = dict(zip(src.source, src.matrix_row))
    results, checks = {}, {}

    # ---------------- deployed edges: rebuild and verify against the deployed records
    old_edges = pd.read_csv(EDGE_FILES["deployed"])
    rec_old, n_used_old = build_records(old_edges, top10, S, lfc_rows, lfc, gene_to_var)
    pp = pd.read_parquet(OLD_PP)
    m = pp.merge(rec_old, on=["source", "target"], how="outer", suffixes=("_pp", "_rb"), indicator=True)
    both = m[m._merge == "both"]
    checks["deployed_rebuild"] = dict(
        n_deployed_records=int(len(pp)), n_rebuilt=int(len(rec_old)),
        n_in_both=int(len(both)), n_only_deployed=int((m._merge == "left_only").sum()),
        n_only_rebuilt=int((m._merge == "right_only").sum()),
        pred_sign_matches=int((both.predicted_inhibitory_pp == both.predicted_inhibitory_rb).sum()),
        max_abs_lfc_diff=float((both.actual_lfc_pp - both.actual_lfc_rb).abs().max()),
        edges_used=n_used_old)
    checks["deployed_rebuild"]["pass"] = bool(checks["deployed_rebuild"]["n_only_deployed"] == 0
                                              and checks["deployed_rebuild"]["n_only_rebuilt"] == 0
                                              and checks["deployed_rebuild"]["pred_sign_matches"] == len(pp)
                                              and checks["deployed_rebuild"]["max_abs_lfc_diff"] == 0.0)
    # score the deployed records in their deployed row order (this fixes the 5 source folds exactly as deployed)
    res_old, df_old, ps_old = evaluate(pp.merge(rec_old[["source", "target", "frac_inhibitory"]], on=["source", "target"]),
                                       "deployed", pp["source"].unique(), top10, feats,
                                       set(zip(old_edges.src_layer, old_edges.src_feature)))
    results["deployed"] = res_old
    inv = json.loads(INV.read_text())
    checks["deployed_vs_investigation"] = dict(
        accuracy=[res_old["pooled"]["accuracy"], inv["pooled"]["accuracy"]],
        balanced_acc=[res_old["pooled"]["balanced_acc"], inv["pooled"]["balanced_acc"]],
        mcc=[res_old["pooled"]["mcc"], inv["pooled"]["mcc"]],
        target_baseline_acc=[[b for b in res_old["baselines"] if b["predictor"].startswith("per-target")][0]["accuracy"],
                             0.5915],
        acc_minus_always_dec_ci=[res_old["bootstrap_by_silenced_gene"]["stats"]["pooled_acc_minus_always_dec"]["ci95"],
                                 inv["grouped_bootstrap"]["stats"]["pooled_acc_minus_always_dec"]["ci95"]],
        bal_minus_half_ci=[res_old["bootstrap_by_silenced_gene"]["stats"]["pooled_balanced_acc_minus_0.5"]["ci95"],
                           inv["grouped_bootstrap"]["stats"]["pooled_balanced_acc_minus_0.5"]["ci95"]],
    )
    ps_old.to_csv(OUT / "per_source_metrics_deployed_edges.csv")

    # ---------------- v2 edges
    if not EDGE_FILES["v2"].exists():
        H.write_json(OUT / "results_deployed_only.json", dict(results=results, checks=checks))
        print(json.dumps(checks, indent=1, default=str)); print("v2 edges not built yet")
        return
    new_edges = pd.read_csv(EDGE_FILES["v2"])
    rec_new, n_used_new = build_records(new_edges, top10, S, lfc_rows, lfc, gene_to_var)
    checks["v2_records"] = dict(edges_total=int(len(new_edges)), edges_used=n_used_new, n_records=int(len(rec_new)),
                                n_sources=int(rec_new.source.nunique()))
    if len(rec_new) == 0:
        H.write_json(OUT / "results.json", dict(results=results, checks=checks))
        print("no v2 records"); return
    res_new, df_new, ps_new = evaluate(rec_new, "v2", sorted(rec_new.source.unique()), top10, feats,
                                       set(zip(new_edges.src_layer, new_edges.src_feature)))
    results["v2"] = res_new
    ps_new.to_csv(OUT / "per_source_metrics_v2_edges.csv")
    # sensitivity: v2 edges, original Phase 11 rule (both endpoints carry an enrichment label)
    lab_mask = np.array([(a, b) in labelled and (c, d) in labelled for a, b, c, d in
                         zip(new_edges.src_layer, new_edges.src_feature, new_edges.tgt_layer, new_edges.tgt_feature)])
    rec_lab, _ = build_records(new_edges, top10, S, lfc_rows, lfc, gene_to_var, edge_mask=lab_mask)
    if len(rec_lab):
        d = scal(metrics_from_counts(*conf_counts(rec_lab.predicted_inhibitory, rec_lab.actual_inhibitory)))
        results["v2_sensitivity_labelled_edges_only"] = dict(n_edges=int(lab_mask.sum()), **d)
    # sensitivity: ties predicted as decrease
    tie_dec = df_new.pred_dec | (df_new.frac_inhibitory == 0.5)
    results["v2_sensitivity_ties_as_decrease"] = scal(metrics_from_counts(*conf_counts(tie_dec, df_new.obs_dec)))
    # overlap of the two record sets
    mm = rec_new.merge(pp, on=["source", "target"], suffixes=("_v2", "_dep"))
    checks["record_overlap"] = dict(n_v2=int(len(rec_new)), n_deployed=int(len(pp)), n_shared=int(len(mm)),
                                    pred_sign_agreement_shared=float((mm.predicted_inhibitory_v2 == mm.predicted_inhibitory_dep).mean()) if len(mm) else None)
    if len(mm):
        obs = mm.actual_lfc_v2 < 0
        assert (mm.actual_lfc_v2 == mm.actual_lfc_dep).all()
        results["shared_records"] = dict(
            n=int(len(mm)), n_sources=int(mm.source.nunique()),
            v2=scal(metrics_from_counts(*conf_counts(mm.predicted_inhibitory_v2, obs))),
            deployed=scal(metrics_from_counts(*conf_counts(mm.predicted_inhibitory_dep, obs))))

    # ---------------- per_pair.parquet (v2) with fold and group columns
    grp = ps_new[["source_group", "source_component"]]
    out_pp = df_new.merge(grp, left_on="source", right_index=True, how="left")
    out_pp = out_pp[["source", "target", "evidence", "n_inhibitory", "frac_inhibitory", "max_abs_d", "sum_d",
                     "predicted_inhibitory", "actual_lfc", "actual_inhibitory", "correct", "src_fold",
                     "pred_target_major", "source_group", "source_component"]]
    out_pp = out_pp.rename(columns={"pred_target_major": "target_baseline_pred_inhibitory"})
    out_pp.to_parquet(V2 / "per_pair.parquet", index=False)

    # ---------------- comparison table
    rows = []
    for name, r in [("deployed edges (block-deleting hook)", res_old), ("v2 edges (fixed hooks)", res_new)]:
        p = r["pooled"]; bs = r["bootstrap_by_silenced_gene"]["stats"]; bg = r["bootstrap_by_source_feature_group"]["stats"]
        tb = [b for b in r["baselines"] if b["predictor"].startswith("per-target")][0]
        rows.append({"edge set": name, "pairs": int(p["n"]), "silenced genes": p["n_sources"],
                     "obs decrease": p["frac_obs_dec"], "pred decrease": p["frac_pred_dec"],
                     "accuracy": p["accuracy"], "always decrease": p["always_dec_acc"], "always increase": p["always_inc_acc"],
                     "majority class": p["majority_acc"], "balanced accuracy": p["balanced_acc"], "MCC": p["mcc"],
                     "chance from marginals": p["chance_acc_given_marginals"],
                     "target-direction baseline (cross-fitted)": tb["accuracy"],
                     "acc - best constant": p["acc_minus_best_constant"],
                     "CI acc - best constant (genes)": bs["pooled_acc_minus_best_constant_rechosen_each_rep"]["ci95"],
                     "CI acc - best constant (feature groups)": bg["pooled_acc_minus_best_constant_rechosen_each_rep"]["ci95"],
                     "CI balanced acc - 0.5 (genes)": bs["pooled_balanced_acc_minus_0.5"]["ci95"],
                     "CI balanced acc - 0.5 (feature groups)": bg["pooled_balanced_acc_minus_0.5"]["ci95"],
                     "CI MCC (genes)": bs["pooled_mcc"]["ci95"], "CI MCC (feature groups)": bg["pooled_mcc"]["ci95"],
                     "acc - target baseline": bs["pooled_acc_minus_target_direction_baseline"]["point"],
                     "CI acc - target baseline (genes)": bs["pooled_acc_minus_target_direction_baseline"]["ci95"]})
    pd.DataFrame(rows).to_csv(OUT / "comparison_table.csv", index=False)
    H.write_json(OUT / "results.json", dict(results=results, checks=checks))
    cfg = dict(script=str(Path(__file__)), script_sha256=H.sha256_file(__file__), device="cpu",
               seeds=dict(bootstrap=SEED, source_folds=SEED, within_source_target_folds=SEED + 1), n_boot=NBOOT,
               inputs={k: str(v) for k, v in EDGE_FILES.items()} | dict(lfc=str(LFC_DIR), deployed_records=str(OLD_PP)),
               numpy=np.__version__, pandas=pd.__version__, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
               wall_seconds=round(time.time() - t0, 1))
    H.write_json(OUT / "run_config.json", cfg)
    print(json.dumps(checks, indent=1, default=str))
    print(pd.DataFrame(rows).T.to_string())


if __name__ == "__main__":
    main()
