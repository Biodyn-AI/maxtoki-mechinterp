"""v2 step 11: second-method checks on the intervals (no model, no re-fit; reads saved replicate files only).

 1. For every bootstrapped quantity: percentile CI (as reported), the 'basic' bootstrap CI
    [2*obs - q97.5, 2*obs - q2.5], the bootstrap mean minus the observed value (bias), and the Monte-Carlo
    precision of the percentile end points (SD of the 2.5% / 97.5% quantiles over 1,000 resamples of the
    replicate list, rng default_rng(99)).
 2. Cross-check with the independent investigation code (different script, different RNG streams):
    projects/maxtoki/verification/manifold_ci/02_bootstrap.json (donor bootstrap), 03_lotdo.json (leave-one-training-donor-out),
    05_internal_branch_*.json (per-branch internal holdout), 06_twolevel_summary.json (37-rep two-level bootstrap).
 3. Permutation statistic re-derived with scipy.stats.spearmanr for the first 300 external and all 10,000 lung
    permutations (the main script uses Pearson-of-average-ranks), compared value by value.
Writes outputs/v2_intervals/v2_11_interval_checks.json
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, "<REPO_ROOT>/projects/maxtoki/runs/manifold-discovery-217M/scripts")
import json
from pathlib import Path
import numpy as np, pandas as pd, torch
from v2_common import *

INV = Path("<REPO_ROOT>/projects/maxtoki/verification/manifold_ci")
J = lambda p: json.loads(Path(p).read_text())
rng_mc = np.random.default_rng(99)


def check(vals, obs, gate=None):
    v = np.asarray(vals, float); v = v[~np.isnan(v)]
    lo, hi = np.percentile(v, [2.5, 97.5])
    q = np.array([np.percentile(v[rng_mc.integers(0, len(v), len(v))], [2.5, 97.5]) for _ in range(1000)])
    out = {"observed": float(obs), "n_reps": int(len(v)), "boot_mean": float(v.mean()), "bias_mean_minus_obs": float(v.mean() - obs),
           "percentile_ci": [float(lo), float(hi)], "basic_ci": [float(2 * obs - hi), float(2 * obs - lo)],
           "mc_sd_of_endpoints": [float(q[:, 0].std(ddof=1)), float(q[:, 1].std(ddof=1))]}
    if gate is not None:
        out["frac_below_gate"] = float(np.mean(v < gate)); out["gate"] = gate
    return out


R = {"bootstrap_checks": {}, "cross_checks": {}, "permutation_statistic_check": {}}
rep = J(OUT / "v2_00_reproduce.json")
# ---- frozen-head donor bootstrap
for panel in ["external", "zeroshot", "lung_nonhema"]:
    df = pd.DataFrame([json.loads(l) for l in (OUT / f"frozen_boot/{panel}.jsonl").read_text().splitlines() if l.strip()])
    P = rep[panel]
    obs = {"trust": P["trust_masked_k15"], "branch": P["branch_within_mean"], "branch_aw": P["branch_within_anchor_weighted_mean"],
           "donor": P["donor_within_mean"], "global": P["global_spearman"]}
    for k, o in obs.items():
        R["bootstrap_checks"][f"{panel}|frozen_donor_boot|{k}"] = check(df[k], o, GATE_TRUST if k == "trust" else (None if k == "global" else GATE_CORR))
# ---- training-donor bootstrap with re-fit (300 reps)
bi = pd.read_csv(OUT / "v2_03_boot_internal_reps.csv")
rf = J(OUT / "v2_03_refit_summary.json"); R4 = rf["internal_reproduction_threads4"]["H65"]
for col, o, g in [("trust", R4["trust_in_sample"], GATE_TRUST), ("branch", R4["branch_holdout"]["plain_mean"], GATE_CORR),
                  ("branch_aw", R4["branch_holdout"]["anchor_weighted_mean"], GATE_CORR)]:
    R["bootstrap_checks"][f"internal|training_donor_boot_refit|{col}"] = check(bi[col], o, g)
for p in ["external", "zeroshot"]:
    for lvl in ["trainonly", "bothlevels"]:
        for k in ["trust", "branch"]:
            o = rep[p]["trust_masked_k15"] if k == "trust" else rep[p]["branch_within_mean"]
            R["bootstrap_checks"][f"{p}|{lvl}|{k}"] = check(bi[f"{p}_{lvl}_{k}"], o, GATE_TRUST if k == "trust" else GATE_CORR)
# ---- other orderings
oo = J(OUT / "v2_05_other_orderings.json")["bootstrap"]
for key, B in oo.items():
    df = pd.DataFrame([json.loads(l) for l in (OUT / f"orderings/boot_{key}.jsonl").read_text().splitlines() if l.strip()])
    for k in ["trust", "category", "donor"]:
        if k in df and k in B["observed"]:
            R["bootstrap_checks"][f"{key}|frozen_donor_boot|{k}"] = check(df[k], B["observed"][k], GATE_TRUST if k == "trust" else GATE_CORR)

# ---- cross-checks with the independent investigation scripts
inv2 = J(INV / "02_bootstrap.json"); fb = J(OUT / "v2_01_frozen_donor_bootstrap.json")
for p in ["external", "zeroshot"]:
    for k in ["trust", "branch"]:
        R["cross_checks"][f"{p}|donor_boot|{k}"] = {
            "v2 (rng [4242, code, rep], 2000 reps)": fb[p]["bootstrap"][k]["ci95_percentile"],
            "investigation (rng default_rng(2), 2000 reps)": inv2[p]["c_donor_cluster_bootstrap_2000"][k]["ci95_percentile"],
            "v2 lodo range": [fb[p]["jackknife"][k]["lodo_min"], fb[p]["jackknife"][k]["lodo_max"]],
            "investigation lodo t_ci95": inv2[p]["d_leave_one_donor_out"][k]["t_ci95"]}
lot = J(OUT / "v2_10_internal_lotdo.json"); inv3 = J(INV / "03_lotdo.json")
diffs = []
for u, v in lot["per_donor"].items():
    if u in inv3:
        for p in ["external", "zeroshot"]:
            for k in ["trust", "branch"]:
                diffs.append(abs(v[f"{p}_{k}"] - inv3[u][p][k]))
R["cross_checks"]["lotdo_external_zeroshot_max_abs_diff_vs_investigation"] = float(max(diffs)) if diffs else None
R["cross_checks"]["lotdo_n_compared"] = len(diffs)
inv5 = J(INV / "05_internal_branch_all_donors_H65.json")["all_donors_H65"]["per_branch"]
pb = pd.read_csv(OUT / "v2_03_internal_per_branch.csv")
pb = pb[(pb.ruler == "H65") & (pb.threads == 4)].set_index("branch")
R["cross_checks"]["internal_per_branch_max_abs_diff_vs_investigation"] = float(max(abs(pb.loc[b, "rho"] - inv5[b]["rho"]) for b in inv5
                                                                                   if inv5[b]["rho"] is not None))
R["cross_checks"]["internal_per_branch_branches_compared"] = sorted(b for b in inv5 if inv5[b]["rho"] is not None)
inv6 = J(INV / "06_twolevel_summary.json")
R["cross_checks"]["two_level"] = {
    k: {"v2 (300 reps) percentile": rf["bootstrap_training_donors"][f"{p}_{lvl2}_{m}"]["ci95_percentile"],
        "v2 sd": rf["bootstrap_training_donors"][f"{p}_{lvl2}_{m}"]["sd"],
        "investigation (37 reps) percentile": inv6[k]["ci95_percentile"], "investigation sd": inv6[k]["sd"]}
    for k, p, lvl2, m in [("external_trust_both", "external", "bothlevels", "trust"), ("external_branch_both", "external", "bothlevels", "branch"),
                          ("zeroshot_trust_both", "zeroshot", "bothlevels", "trust"), ("zeroshot_branch_both", "zeroshot", "bothlevels", "branch"),
                          ("external_branch_trainonly", "external", "trainonly", "branch")] if k in inv6}

# ---- permutation statistic re-derived with spearmanr
nodes, sidx, T = stage_distance_table()
FZ = Frozen()
head = LETHead(FZ.F_int.shape[1], LATENT_DIM)
head.load_state_dict(torch.load(ART / "heads/let_anchor_internal.pt", map_location="cpu")); head.eval()
PANEL_CODE = {"external": 1, "zeroshot": 2, "lung_nonhema": 3}
for panel, nchk in [("external", 300), ("lung_nonhema", 10000)]:
    f, d, m = FZ.features(panel); z = head_z(head, f); Dh = arccos_dist(z)
    iu = np.triu_indices(len(d), 1)
    stages = m["hema_stage"].astype(str).to_numpy()
    blocks = (m["donor_id"].astype(str) + "|" + m["tissue"].astype(str)).to_numpy()
    members = [np.where(blocks == g)[0] for g in pd.unique(blocks)]
    saved = np.load(OUT / f"perm_null_{panel}.npy")
    vals = []
    for b in range(nchk):
        rng = np.random.default_rng([777, PANEL_CODE[panel], b]); s = stages.copy()
        for ii in members:
            if len(ii) > 1:
                s[ii] = stages[rng.permutation(ii)]
        vals.append(float(spearmanr(Dh[iu], ruler_from_stages(s, sidx, T)[iu])[0]))
    vals = np.array(vals)
    obs = float(spearmanr(Dh[iu], d[iu])[0])
    R["permutation_statistic_check"][panel] = {
        "n_checked": nchk, "max_abs_diff_vs_saved_null": float(np.max(np.abs(vals - saved[:nchk]))),
        "observed_spearmanr": obs, "n_ge_obs_in_checked": int(np.sum(vals >= obs)),
        "p_two_sided_doubled_from_checked": float(min(1.0, 2 * min((1 + np.sum(vals >= obs)) / (nchk + 1), (1 + np.sum(vals <= obs)) / (nchk + 1))))}

(OUT / "v2_11_interval_checks.json").write_text(json.dumps(R, indent=2))
write_run_config("v2_11_interval_checks", [OUT / "v2_00_reproduce.json", OUT / "v2_01_frozen_donor_bootstrap.json",
                 OUT / "v2_03_boot_internal_reps.csv", OUT / "v2_03_refit_summary.json", OUT / "v2_05_other_orderings.json",
                 OUT / "v2_10_internal_lotdo.json", OUT / "perm_null_external.npy", OUT / "perm_null_lung_nonhema.npy",
                 INV / "02_bootstrap.json", INV / "03_lotdo.json", INV / "05_internal_branch_all_donors_H65.json",
                 INV / "06_twolevel_summary.json"] + CORE_INPUTS, {"mc_seed": 99})
for k, v in R["bootstrap_checks"].items():
    print(f"{k:55s} obs {v['observed']:+.4f} pct [{v['percentile_ci'][0]:+.4f},{v['percentile_ci'][1]:+.4f}] "
          f"basic [{v['basic_ci'][0]:+.4f},{v['basic_ci'][1]:+.4f}] bias {v['bias_mean_minus_obs']:+.4f} "
          f"mcSD {v['mc_sd_of_endpoints'][0]:.4f}/{v['mc_sd_of_endpoints'][1]:.4f}")
print(json.dumps(R["cross_checks"], indent=1)); print(json.dumps(R["permutation_statistic_check"], indent=1))
