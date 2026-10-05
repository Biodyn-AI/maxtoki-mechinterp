"""Re-analysis of circuit-tracing CRISPRi directional accuracy on the IDENTICAL
corrected per-pair records (runs/circuit-tracing-217M/outputs/groupkfold_crispri/per_pair.parquet).

Read-only on the repo. Writes only to this script's folder.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("<REPO_ROOT>/projects/maxtoki")
PP = REPO / "runs/circuit-tracing-217M/outputs/groupkfold_crispri/per_pair.parquet"
OUT = Path(__file__).resolve().parent
SEED = 42
NBOOT = 2000

df = pd.read_parquet(PP)
df["obs_dec"] = df["actual_lfc"] < 0          # as in the script: LFC==0 counts as "not decrease"
df["obs_zero"] = df["actual_lfc"] == 0
df["pred_dec"] = df["predicted_inhibitory"].astype(bool)
assert (df["obs_dec"] == df["actual_inhibitory"]).all()
assert (df["correct"] == (df["pred_dec"] == df["obs_dec"])).all()


# ---------------------------------------------------------------- helpers
def conf_counts(pred, obs):
    pred = np.asarray(pred, bool); obs = np.asarray(obs, bool)
    tp = int((pred & obs).sum()); fn = int((~pred & obs).sum())
    fp = int((pred & ~obs).sum()); tn = int((~pred & ~obs).sum())
    return tp, fn, fp, tn


def metrics_from_counts(tp, fn, fp, tn):
    """positive class = observed decrease."""
    tp, fn, fp, tn = (np.asarray(x, float) for x in (tp, fn, fp, tn))
    n = tp + fn + fp + tn
    p_obs = (tp + fn) / n              # fraction observed decreases
    p_pred = (tp + fp) / n             # fraction predicted decreases
    acc = (tp + tn) / n
    with np.errstate(invalid="ignore", divide="ignore"):
        tpr = tp / (tp + fn)           # recall on decreases
        tnr = tn / (tn + fp)           # recall on increases
        bal = 0.5 * (tpr + tnr)
        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        chance = p_pred * p_obs + (1 - p_pred) * (1 - p_obs)   # expected acc if pred independent of obs
        kappa = (acc - chance) / (1 - chance)
    return dict(n=n, frac_obs_dec=p_obs, frac_obs_inc=1 - p_obs, frac_pred_dec=p_pred,
                frac_pred_inc=1 - p_pred, accuracy=acc, always_dec_acc=p_obs,
                always_inc_acc=1 - p_obs, majority_acc=np.maximum(p_obs, 1 - p_obs),
                acc_minus_best_constant=acc - np.maximum(p_obs, 1 - p_obs),
                recall_dec=tpr, recall_inc=tnr, balanced_acc=bal, mcc=mcc,
                chance_acc_given_marginals=chance, acc_minus_chance_given_marginals=acc - chance,
                cohen_kappa=kappa)


def scal(d):
    return {k: (float(v) if np.ndim(v) == 0 else v) for k, v in d.items()}


results = {}

# ---------------------------------------------------------------- 1. pooled
tp, fn, fp, tn = conf_counts(df["pred_dec"], df["obs_dec"])
pooled = scal(metrics_from_counts(tp, fn, fp, tn))
pooled.update(dict(tp_pred_dec_obs_dec=tp, fn_pred_inc_obs_dec=fn, fp_pred_dec_obs_inc=fp,
                   tn_pred_inc_obs_inc=tn, n_obs_lfc_exactly_zero=int(df["obs_zero"].sum()),
                   n_obs_lfc_positive=int((df["actual_lfc"] > 0).sum()),
                   n_sources=int(df["source"].nunique()), n_targets=int(df["target"].nunique())))
results["pooled"] = pooled

# zero-handling sensitivity: drop the LFC==0 pairs, or count them as decreases
m = ~df["obs_zero"]
results["pooled_drop_lfc_zero"] = scal(metrics_from_counts(*conf_counts(df.loc[m, "pred_dec"], df.loc[m, "obs_dec"])))
results["pooled_zero_as_decrease"] = scal(metrics_from_counts(*conf_counts(df["pred_dec"], df["actual_lfc"] <= 0)))

# ---------------------------------------------------------------- 2. per source
g = df.groupby("source", sort=True)
ps = pd.DataFrame({
    "tp": g.apply(lambda x: int((x.pred_dec & x.obs_dec).sum())),
    "fn": g.apply(lambda x: int((~x.pred_dec & x.obs_dec).sum())),
    "fp": g.apply(lambda x: int((x.pred_dec & ~x.obs_dec).sum())),
    "tn": g.apply(lambda x: int((~x.pred_dec & ~x.obs_dec).sum())),
})
pm = metrics_from_counts(ps.tp, ps.fn, ps.fp, ps.tn)
for k, v in pm.items():
    ps[k] = np.asarray(v)
ps["model_beats_always_dec"] = ps["accuracy"] > ps["always_dec_acc"]
ps["model_beats_own_majority"] = ps["accuracy"] > ps["majority_acc"]
ps.to_csv(OUT / "per_source_metrics.csv")

per_src_summary = {}
for col in ["n", "frac_obs_dec", "frac_pred_dec", "accuracy", "always_dec_acc", "always_inc_acc",
            "majority_acc", "acc_minus_best_constant", "balanced_acc", "mcc", "cohen_kappa",
            "chance_acc_given_marginals", "acc_minus_chance_given_marginals"]:
    s = ps[col].astype(float)
    per_src_summary[col] = dict(mean=float(s.mean()), median=float(s.median()),
                                min=float(s.min()), max=float(s.max()), n_nan=int(s.isna().sum()))
per_src_summary["n_sources_accuracy_gt_always_dec"] = int((ps.accuracy > ps.always_dec_acc).sum())
per_src_summary["n_sources_accuracy_gt_own_majority"] = int((ps.accuracy > ps.majority_acc).sum())
per_src_summary["n_sources_balanced_acc_gt_0.5"] = int((ps.balanced_acc > 0.5).sum())
per_src_summary["n_sources_mcc_gt_0"] = int((ps.mcc > 0).sum())
per_src_summary["n_sources_frac_obs_dec_gt_0.5"] = int((ps.frac_obs_dec > 0.5).sum())
per_src_summary["n_sources_all_pred_same_sign"] = int(((ps.frac_pred_dec == 0) | (ps.frac_pred_dec == 1)).sum())
per_src_summary["n_sources"] = int(len(ps))
# correlation: is per-source accuracy just the per-source fraction of decreases?
per_src_summary["pearson_accuracy_vs_frac_obs_dec"] = float(np.corrcoef(ps.accuracy, ps.frac_obs_dec)[0, 1])
results["per_source_summary"] = per_src_summary

# ---------------------------------------------------------------- 3. grouped bootstrap over sources
rng = np.random.default_rng(SEED)
S = len(ps)
C = ps[["tp", "fn", "fp", "tn"]].to_numpy(float)         # (S, 4)
W = np.stack([np.bincount(rng.integers(0, S, S), minlength=S) for _ in range(NBOOT)]).astype(float)  # (B, S)
BC = W @ C                                                 # pooled counts per replicate
bm = metrics_from_counts(BC[:, 0], BC[:, 1], BC[:, 2], BC[:, 3])
acc_b = bm["accuracy"]; pobs_b = bm["frac_obs_dec"]


def ci(x):
    x = np.asarray(x, float)
    return dict(point=None, ci95=[float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))],
                boot_mean=float(x.mean()), frac_reps_gt_0=float((x > 0).mean()))


def per_src_mean(col):
    v = ps[col].to_numpy(float)
    ok = ~np.isnan(v)
    return (W[:, ok] @ v[ok]) / W[:, ok].sum(1)


boot = {}
def add(name, point, reps):
    d = ci(reps); d["point"] = float(point); boot[name] = d

add("pooled_accuracy", pooled["accuracy"], acc_b)
add("pooled_always_dec_acc", pooled["always_dec_acc"], pobs_b)
add("pooled_acc_minus_always_dec", pooled["accuracy"] - pooled["always_dec_acc"], acc_b - pobs_b)
add("pooled_acc_minus_best_constant_rechosen_each_rep", pooled["acc_minus_best_constant"],
    acc_b - np.maximum(pobs_b, 1 - pobs_b))
add("pooled_balanced_acc_minus_0.5", pooled["balanced_acc"] - 0.5, bm["balanced_acc"] - 0.5)
add("pooled_mcc", pooled["mcc"], bm["mcc"])
add("pooled_acc_minus_chance_given_marginals", pooled["acc_minus_chance_given_marginals"],
    bm["acc_minus_chance_given_marginals"])
add("pooled_acc_minus_0.5", pooled["accuracy"] - 0.5, acc_b - 0.5)
psm = {c: per_src_mean(c) for c in ["accuracy", "always_dec_acc", "majority_acc", "balanced_acc", "mcc",
                                     "acc_minus_chance_given_marginals"]}
add("per_source_mean_accuracy", ps.accuracy.mean(), psm["accuracy"])
add("per_source_mean_acc_minus_0.5", ps.accuracy.mean() - 0.5, psm["accuracy"] - 0.5)
add("per_source_mean_acc_minus_always_dec", ps.accuracy.mean() - ps.always_dec_acc.mean(),
    psm["accuracy"] - psm["always_dec_acc"])
add("per_source_mean_acc_minus_own_majority_oracle", ps.accuracy.mean() - ps.majority_acc.mean(),
    psm["accuracy"] - psm["majority_acc"])
add("per_source_mean_balanced_acc_minus_0.5", ps.balanced_acc.mean() - 0.5, psm["balanced_acc"] - 0.5)
add("per_source_mean_mcc", ps.mcc.mean(), psm["mcc"])
add("per_source_mean_acc_minus_chance_given_marginals", ps.acc_minus_chance_given_marginals.mean(),
    psm["acc_minus_chance_given_marginals"])
results["grouped_bootstrap"] = dict(n_boot=NBOOT, unit="source (silenced gene), resampled with replacement",
                                    seed=SEED, stats=boot)

# ---------------------------------------------------------------- 4. cross-fitted baselines
# 4a. reproduce the script's 5 source folds exactly
unique_srcs = df["source"].unique()
frng = np.random.default_rng(SEED)
src_perm = frng.permutation(unique_srcs)
folds = np.array_split(src_perm, 5)
fold_of_src = {s: i for i, f in enumerate(folds) for s in f}
df["src_fold"] = df["source"].map(fold_of_src)
fold_rows = []
df["pred_global_major"] = False
df["pred_target_major"] = False
for k in range(5):
    tr = df["src_fold"] != k; te = ~tr
    glob = df.loc[tr, "obs_dec"].mean() > 0.5
    df.loc[te, "pred_global_major"] = glob
    tmaj = df.loc[tr].groupby("target")["obs_dec"].mean()
    t = df.loc[te, "target"].map(tmaj)
    pred_t = np.where(t.isna() | (t == 0.5), glob, t > 0.5)
    df.loc[te, "pred_target_major"] = pred_t
    sub = df.loc[te]
    fold_rows.append(dict(fold=k, n_sources=int(sub.source.nunique()), n_pairs=int(len(sub)),
                          model_acc=float(sub.correct.mean()),
                          train_majority_is_decrease=bool(glob),
                          global_majority_acc=float((sub.pred_global_major == sub.obs_dec).mean()),
                          target_majority_acc=float((sub.pred_target_major == sub.obs_dec).mean()),
                          frac_obs_dec=float(sub.obs_dec.mean())))
folds_df = pd.DataFrame(fold_rows)
folds_df.to_csv(OUT / "groupkfold_by_source_baselines.csv", index=False)

# 4b. per-source sign-matched baseline: within each source, 5 target folds
trng = np.random.default_rng(SEED + 1)
all_t = np.sort(df["target"].unique())
tfold = dict(zip(all_t, trng.permutation(np.arange(len(all_t)) % 5)))
df["tgt_fold"] = df["target"].map(tfold)
cnt = df.groupby(["source", "tgt_fold"])["obs_dec"].agg(["sum", "size"]).reset_index()
tot = cnt.groupby("source")[["sum", "size"]].sum()
cnt = cnt.join(tot, on="source", rsuffix="_tot")
cnt["train_frac_dec"] = (cnt["sum_tot"] - cnt["sum"]) / (cnt["size_tot"] - cnt["size"])
df = df.merge(cnt[["source", "tgt_fold", "train_frac_dec"]], on=["source", "tgt_fold"], how="left")
df["pred_source_major"] = df["train_frac_dec"] > 0.5
# ties: none expected; if exactly 0.5 predict decrease
df.loc[df["train_frac_dec"] == 0.5, "pred_source_major"] = True

base_rows = []
def base_row(name, pred, note):
    tp_, fn_, fp_, tn_ = conf_counts(df[pred] if isinstance(pred, str) else pred, df["obs_dec"])
    d = scal(metrics_from_counts(tp_, fn_, fp_, tn_)); d.update(predictor=name, note=note)
    base_rows.append(d)

base_row("model (majority sign of SAE edges)", "pred_dec", "uses no CRISPRi labels")
base_row("always decrease", np.ones(len(df), bool), "constant")
base_row("always increase", np.zeros(len(df), bool), "constant")
base_row("global majority from training source folds", "pred_global_major",
         "GroupKFold by source, the script's own 5 folds")
base_row("per-target majority from training source folds", "pred_target_major",
         "GroupKFold by source: target gene's usual direction across OTHER silenced genes")
base_row("per-source majority from the source's own other target folds", "pred_source_major",
         "within-source 5-fold over targets; uses labels of the SAME knockdown")
bdf = pd.DataFrame(base_rows)
cols = ["predictor", "accuracy", "balanced_acc", "mcc", "frac_pred_dec", "frac_obs_dec", "note"]
bdf[cols].to_csv(OUT / "pooled_baselines.csv", index=False)

# grouped bootstrap of model accuracy minus each cross-fitted baseline (predictions fixed, sources resampled)
def per_source_acc(predcol):
    return df.assign(_c=(df[predcol] == df["obs_dec"])).groupby("source")["_c"].agg(["sum", "size"]).reindex(ps.index)

mod = per_source_acc("pred_dec")
cf_boot = {}
for name, col in [("global_majority_cv", "pred_global_major"), ("target_majority_cv", "pred_target_major"),
                  ("source_majority_within_source_cv", "pred_source_major")]:
    b = per_source_acc(col)
    # pooled difference
    num_m = W @ mod["sum"].to_numpy(float); num_b = W @ b["sum"].to_numpy(float); den = W @ mod["size"].to_numpy(float)
    diff_pooled = num_m / den - num_b / den
    point_pooled = mod["sum"].sum() / mod["size"].sum() - b["sum"].sum() / b["size"].sum()
    # per-source mean difference
    d_src = (mod["sum"] / mod["size"] - b["sum"] / b["size"]).to_numpy(float)
    diff_src = (W @ d_src) / S
    cf_boot[name] = dict(
        baseline_pooled_acc=float(b["sum"].sum() / b["size"].sum()),
        baseline_per_source_mean_acc=float((b["sum"] / b["size"]).mean()),
        pooled_model_minus_baseline=dict(point=float(point_pooled), ci95=[float(np.percentile(diff_pooled, 2.5)), float(np.percentile(diff_pooled, 97.5))]),
        per_source_mean_model_minus_baseline=dict(point=float(d_src.mean()), ci95=[float(np.percentile(diff_src, 2.5)), float(np.percentile(diff_src, 97.5))]),
        n_sources_model_better=int((d_src > 0).sum()),
    )
results["cross_fitted_baselines"] = dict(folds=fold_rows, bootstrap=cf_boot)

# ---------------------------------------------------------------- 5. strata by |LFC|
df["abs_lfc"] = df["actual_lfc"].abs()
strata = []
for lo, hi in [(0, 0.01), (0.01, 0.025), (0.025, 0.05), (0.05, 0.1), (0.1, 0.25), (0.25, np.inf)]:
    mm = (df.abs_lfc >= lo) & (df.abs_lfc < hi)
    d = scal(metrics_from_counts(*conf_counts(df.loc[mm, "pred_dec"], df.loc[mm, "obs_dec"])))
    d.update(abs_lfc_lo=lo, abs_lfc_hi=hi)
    strata.append(d)
for thr in [0.1, 0.25, 0.5]:
    mm = df.abs_lfc >= thr
    d = scal(metrics_from_counts(*conf_counts(df.loc[mm, "pred_dec"], df.loc[mm, "obs_dec"])))
    d.update(abs_lfc_lo=thr, abs_lfc_hi=np.inf, n_sources=int(df.loc[mm, "source"].nunique()))
    strata.append(d)
sdf = pd.DataFrame(strata)
sdf.to_csv(OUT / "strata_by_abs_lfc.csv", index=False)
results["strata_by_abs_lfc"] = strata

# ---------------------------------------------------------------- 6. what drives the predicted sign
pt = df.groupby("target")["pred_dec"].mean()
results["prediction_structure"] = dict(
    n_targets=int(len(pt)),
    frac_targets_all_sources_pred_dec=float((pt == 1).mean()),
    frac_targets_all_sources_pred_inc=float((pt == 0).mean()),
    frac_targets_mixed=float(((pt > 0) & (pt < 1)).mean()),
)
# how much of the model's pred sign is explained by target alone vs source alone
def r2_of_groupmean(col, grp):
    y = df[col].astype(float); gm = df.groupby(grp)[col].transform("mean")
    return float(1 - ((y - gm) ** 2).sum() / ((y - y.mean()) ** 2).sum())
results["prediction_structure"].update(
    r2_pred_sign_by_target=r2_of_groupmean("pred_dec", "target"),
    r2_pred_sign_by_source=r2_of_groupmean("pred_dec", "source"),
    r2_obs_sign_by_target=r2_of_groupmean("obs_dec", "target"),
    r2_obs_sign_by_source=r2_of_groupmean("obs_dec", "source"),
)

(OUT / "results.json").write_text(json.dumps(results, indent=2, default=float))
print(json.dumps(results["pooled"], indent=1))
print(json.dumps(per_src_summary, indent=1))
print(json.dumps(boot, indent=1))
print(folds_df.to_string())
print(bdf[cols].to_string())
print(json.dumps(cf_boot, indent=1))
print(sdf[["abs_lfc_lo", "abs_lfc_hi", "n", "frac_obs_dec", "frac_pred_dec", "accuracy", "always_dec_acc", "balanced_acc", "mcc"]].to_string())
print(json.dumps(results["prediction_structure"], indent=1))
print("zero handling:", json.dumps({k: results[k]["accuracy"] for k in ["pooled_drop_lfc_zero", "pooled_zero_as_decrease"]}))
