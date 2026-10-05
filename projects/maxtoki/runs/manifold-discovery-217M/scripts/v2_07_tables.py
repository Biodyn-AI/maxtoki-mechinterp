"""v2 step 7: source-data CSVs for the manifold table and figure.

Reads only v2 outputs (v2_00 ... v2_06, v2_10, v2_11) and writes
  outputs/v2_intervals/table_gates.csv   one row per (ordering, panel, metric)
  outputs/v2_intervals/fig_gates.csv     H65 trust and branch per panel, with intervals, for the figure
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, "<REPO_ROOT>/projects/maxtoki/runs/manifold-discovery-217M/scripts")
import json
import numpy as np, pandas as pd
from v2_common import *

J = lambda n: json.loads((OUT / n).read_text())
rep = J("v2_00_reproduce.json"); fb = J("v2_01_frozen_donor_bootstrap.json"); rf = J("v2_03_refit_summary.json")
pf = J("v2_04_permutation_frozen.json"); pi = J("v2_04_permutation_internal.json"); oo = J("v2_05_other_orderings.json")
facts = J("v2_06_facts.json"); lot = J("v2_10_internal_lotdo.json"); chk = J("v2_11_interval_checks.json")["bootstrap_checks"]
cohort = pd.read_csv(OUT / "facts_cohorts.csv").set_index("panel")
rows = []
DONOR_BOOT = "donor cluster bootstrap (donors with replacement), head frozen, percentile"
TRAIN_BOOT = "bootstrap over the 11 training donors (with replacement), head re-fitted, percentile"
TWO_LEVEL = "two-level: training donors resampled + head re-fitted, then evaluation donors resampled, percentile"
TRAIN_ONLY = "training donors resampled + head re-fitted; evaluation panel fixed, percentile"


def add(ordering, panel, metric, value, definition, gate=None, ci=None, method=None, n_reps=None, unit=None,
        k=None, below=None, lodo=None, deployed=None, source=None, n_anchors=None, n_donors=None, note=None,
        group=None, check=None, lodo_unit=None):
    c = chk.get(check) if check else None
    rows.append({"ordering": ordering, "panel": panel, "metric": metric, "group": group, "value": value,
                 "boot_mean": c["boot_mean"] if c else None,
                 "basic_ci_low": c["basic_ci"][0] if c else None, "basic_ci_high": c["basic_ci"][1] if c else None,
                 "ci_low": ci[0] if ci else None, "ci_high": ci[1] if ci else None, "ci_method": method,
                 "n_reps": n_reps, "resampling_unit": unit, "gate": gate,
                 "passes_point": (None if gate is None or value is None or (isinstance(value, float) and np.isnan(value))
                                  else (value <= gate if metric.startswith("permutation") else value >= gate)),
                 "frac_resamples_below_gate": below, "lodo_min": lodo[0] if lodo else None,
                 "lodo_max": lodo[1] if lodo else None,
                 "lodo_unit": (lodo_unit or ("training donor (head re-fitted)" if panel == "internal" else "evaluation donor (head frozen)")) if lodo else None,
                 "deployed_value": deployed, "k_neighbours": k,
                 "n_anchors": n_anchors, "n_donors": n_donors, "definition": definition, "note": note,
                 "source_file": source})


def bcol(B, key, gate):
    b = B[key]; return b["ci95_percentile"], b.get(f"frac_below_{gate}")


# ------------------------------------------------------------- H65 internal
R4 = rf["internal_reproduction_threads4"]; BT = rf["bootstrap_training_donors"]; nb = BT["n_complete_reps"]
ni, ndi = int(cohort.loc["internal", "n_anchors"]), int(cohort.loc["internal", "n_donors"])
ci, bl = bcol(BT, "internal_trust", GATE_TRUST)
add("H65", "internal", "trustworthiness", R4["H65"]["trust_in_sample"], "in-sample; 2,464-d features vs 10-d head output; k=15",
    GATE_TRUST, ci, TRAIN_BOOT, nb, "training donor", 15, bl, [lot["ranges"]["trust"]["min"], lot["ranges"]["trust"]["max"]],
    deployed=0.811, source="v2_03_refit_summary.json; v2_10_internal_lotdo.json", n_anchors=ni, n_donors=ndi,
    check="internal|training_donor_boot_refit|trust", note="leave-one-training-donor-out lowest when dropping %s" % lot["ranges"]["trust"]["argmin"])
ci, bl = bcol(BT, "internal_branch_plain", GATE_CORR)
add("H65", "internal", "branch_holdout_plain_mean", R4["H65"]["branch_holdout"]["plain_mean"],
    "re-fit without each branch; Spearman on held-out pairs; plain mean over 6 branches", GATE_CORR, ci, TRAIN_BOOT, nb,
    "training donor", None, bl, [lot["ranges"]["branch_plain"]["min"], lot["ranges"]["branch_plain"]["max"]],
    deployed=0.370, source="v2_03_refit_summary.json; v2_10_internal_lotdo.json", n_anchors=ni, n_donors=ndi,
    check="internal|training_donor_boot_refit|branch", note="t interval over the 6 branch values: [%.3f, %.3f]" % tuple(R4["H65"]["branch_holdout"]["t_interval_95_over_branches"]))
ci, bl = bcol(BT, "internal_branch_anchor_weighted", GATE_CORR)
add("H65", "internal", "branch_holdout_anchor_weighted", R4["H65"]["branch_holdout"]["anchor_weighted_mean"],
    "same branch values weighted by anchors per branch", GATE_CORR, ci, TRAIN_BOOT, nb, "training donor", None, bl,
    [lot["ranges"]["branch_anchor_weighted"]["min"], lot["ranges"]["branch_anchor_weighted"]["max"]],
    source="v2_03_refit_summary.json; v2_10_internal_lotdo.json", n_anchors=ni, n_donors=ndi,
    check="internal|training_donor_boot_refit|branch_aw")
# per-branch internal holdout rows (re-fit without the branch); interval = training-donor bootstrap of that branch
pbi = pd.read_csv(OUT / "v2_03_internal_per_branch.csv")
for _, r in pbi[(pbi.threads == 4)].iterrows():
    bkey = f"per_branch_{r.branch}"
    bb = BT.get(bkey) if r.ruler == "H65" else None
    add("H65" if r.ruler == "H65" else "H65_null", "internal", "branch_holdout_one_branch", r.rho,
        "re-fit without this branch; Spearman over held-out pairs", GATE_CORR,
        bb["ci95_percentile"] if bb else None, TRAIN_BOOT if bb else None, bb["n_valid"] if bb else None,
        "training donor" if bb else None, None, bb.get("frac_below_0.2") if bb else None, group=r.branch,
        source="v2_03_internal_per_branch.csv; v2_03_refit_summary.json", n_anchors=int(r.n_anchors), n_donors=int(r.n_donors),
        note="%d stages (%s); %d pairs%s" % (r.n_stages, r.stages, r.n_pairs,
              "; branch not scorable in bootstrap replicates without donor TSP2 (constant ruler)" if bb and bb["n_nan"] > 0 else ""))
add("H65", "internal", "donor_holdout", R4["H65"]["donor_holdout_mean"], "re-fit without each donor; mean over 9 donors with >=3 anchors",
    GATE_CORR, deployed=0.716, source="v2_03_refit_summary.json", n_anchors=ni, n_donors=ndi, note="no interval computed")
add("H65", "internal", "random_holdout", rf["published"]["H65"]["random_holdout"], "10 random 80/20 splits with re-fit (deployed value, not re-run)",
    GATE_CORR, deployed=0.834, source="reports/quality_gates_let_anchor.json", n_anchors=ni, n_donors=ndi, note="not re-run, no interval")
add("H65", "internal", "permutation_p_two_sided", pi["global"]["p_two_sided_doubled"],
    "blocked permutation (stage labels within donor x tissue), head re-fitted per permutation, statistic = in-sample Spearman(latent, ruler)",
    0.001, None, None, pi["global"]["B"], "permutation", None, None, source="v2_04_permutation_internal.json", n_anchors=ni, n_donors=ndi,
    note="observed %.5f; null max %.5f; %d of %d null values >= observed" % (pi["global"]["observed"], pi["global"]["null_max"],
                                                                            pi["global"]["n_null_ge_obs"], pi["global"]["B"]))
add("H65", "internal", "permutation_p_two_sided_trust_statistic", pi["trust_same_permutations"]["p_two_sided_doubled"],
    "same permutations and re-fits as above, statistic = in-sample trustworthiness (secondary; the spec names no statistic)",
    0.001, None, None, pi["trust_same_permutations"]["B"], "permutation", source="v2_04_permutation_internal.json", n_anchors=ni,
    n_donors=ndi, note="observed %.4f; null mean %.4f; null max %.4f" % (pi["trust_same_permutations"]["observed"],
                                                                      pi["trust_same_permutations"]["null_mean"], pi["trust_same_permutations"]["null_max"]))
add("H65_null", "internal", "trustworthiness", R4["null"]["trust_in_sample"], "within-branch shuffled ruler", GATE_TRUST,
    deployed=0.799, source="v2_03_refit_summary.json", n_anchors=ni)
add("H65_null", "internal", "branch_holdout_plain_mean", R4["null"]["branch_holdout"]["plain_mean"], "within-branch shuffled ruler",
    GATE_CORR, deployed=-0.001, source="v2_03_refit_summary.json", n_anchors=ni)
sn = rf["optimiser_noise_seeds_1_to_10"]
add("H65", "internal", "trustworthiness_seed_range", None, "same fit with torch seeds 1..10", GATE_TRUST,
    [sn["H65"]["trust_min"], sn["H65"]["trust_max"]], "min-max over 10 seeds (optimiser noise, not sampling)", 10, "seed",
    source="v2_03_refit_summary.json", note="null ruler seeds: %.3f-%.3f" % (sn["null"]["trust_min"], sn["null"]["trust_max"]))

# ------------------------------------------------------------- H65 frozen panels
for panel, lab in [("external", "external"), ("zeroshot", "zero-shot"), ("lung_nonhema", "lung control")]:
    P = rep[panel]; FB = fb[panel]; nr = FB["n_reps"]; na = P["n_anchors"]; nd = P["n_donors"]
    order = "H65" if panel != "lung_nonhema" else "lung_control_random_labels"
    for key, metric, val, gate, dep, dfn in [
        ("trust", "trustworthiness", P["trust_sklearn_k15"], GATE_TRUST, P["published"]["trustworthiness"], "frozen head; 2,464-d features vs 10-d head output; k=15"),
        ("branch", "within_branch_corr_plain_mean", P["branch_within_mean"], GATE_CORR, P["published"]["branch_holdout"], "frozen head; mean within-branch Spearman; nothing held out"),
        ("branch_aw", "within_branch_corr_anchor_weighted", P["branch_within_anchor_weighted_mean"], GATE_CORR, None, "same, weighted by anchors per branch"),
        ("donor", "within_donor_corr", P["donor_within_mean"], GATE_CORR, P["published"]["donor_holdout"], "frozen head; mean within-donor Spearman"),
        ("global", "global_corr", P["global_spearman"], None, P["published"]["global_correlation"], "frozen head; Spearman over all pairs")]:
        ci, bl = FB["bootstrap"][key]["ci95_percentile"], FB["bootstrap"][key].get(f"frac_below_{gate}")
        jk = FB["jackknife"][key]
        add(order, lab, metric, val, dfn, gate, ci, DONOR_BOOT, nr, "evaluation donor", 15 if key == "trust" else None, bl,
            [jk["lodo_min"], jk["lodo_max"]], dep, "v2_01_frozen_donor_bootstrap.json", na, nd,
            note="lowest when dropping %s" % jk["lodo_argmin"], check=f"{panel}|frozen_donor_boot|{key}")
    add(order, lab, "random_holdout", P["random_holdout_frozen"], "frozen head; 20 random 20% subsets", GATE_CORR,
        deployed=P["published"]["random_holdout"], source="v2_00_reproduce.json", n_anchors=na, n_donors=nd, note="no interval")
    pp = pf[panel]
    add(order, lab, "permutation_p_two_sided", pp["p_two_sided_doubled"],
        "blocked permutation (stage labels within donor x tissue), frozen head, statistic = Spearman(latent, ruler)",
        0.001, None, None, pp["B"], "permutation", source="v2_04_permutation_frozen.json", n_anchors=na, n_donors=nd,
        note="observed %.4f; null mean %.4f; null max %.4f; blocks %d" % (pp["observed"], pp["null_mean"], pp["null_max"], pp["n_blocks"]))
    if panel in ("external", "zeroshot"):
        for k in ["trust", "branch"]:
            g = GATE_TRUST if k == "trust" else GATE_CORR
            b = BT[f"{panel}_bothlevels_{k}"]
            add("H65", lab, ("trustworthiness" if k == "trust" else "within_branch_corr_plain_mean") + "_two_level",
                P["trust_sklearn_k15"] if k == "trust" else P["branch_within_mean"], "as above", g,
                b["ci95_percentile"], TWO_LEVEL, nb, "training donor + evaluation donor", None, b.get(f"frac_below_{g}"),
                source="v2_03_refit_summary.json", n_anchors=na, n_donors=nd, check=f"{panel}|bothlevels|{k}")
            b = BT[f"{panel}_trainonly_{k}"]; rk = f"{panel}_{'trust' if k == 'trust' else 'branch'}"
            add("H65", lab, ("trustworthiness" if k == "trust" else "within_branch_corr_plain_mean") + "_training_donors_only",
                P["trust_sklearn_k15"] if k == "trust" else P["branch_within_mean"], "as above; only the training donors resampled", g,
                b["ci95_percentile"], TRAIN_ONLY, nb, "training donor", None, b.get(f"frac_below_{g}"),
                [lot["ranges"][rk]["min"], lot["ranges"][rk]["max"]], source="v2_03_refit_summary.json; v2_10_internal_lotdo.json",
                n_anchors=na, n_donors=nd, check=f"{panel}|trainonly|{k}", lodo_unit="training donor (head re-fitted)",
                note="leave-one-training-donor-out lowest when dropping %s" % lot["ranges"][rk]["argmin"])

pbf = pd.read_csv(OUT / "v2_00_per_branch.csv")
for _, r in pbf.iterrows():
    lab = {"external": "external", "zeroshot": "zero-shot", "lung_nonhema": "lung control"}[r.panel]
    add("H65" if r.panel != "lung_nonhema" else "lung_control_random_labels", lab, "within_branch_corr_one_branch",
        None if pd.isna(r.rho) else r.rho, "frozen head; Spearman over within-branch pairs; nothing held out", GATE_CORR,
        group=r.branch, source="v2_00_per_branch.csv", n_anchors=int(r.n_anchors), n_donors=int(r.n_donors),
        note="%d stages (%s); %d pairs%s" % (r.n_stages, r.stages, r.n_pairs, "; constant ruler, not scored" if pd.isna(r.rho) else ""))
# first lung control (lung-resident immune cells, real stage labels) - replaced during the run
L1 = lot["lung_control_v1_frozen"]
for metric, key, dkey, g in [("trustworthiness", "trust_k15", "trustworthiness", GATE_TRUST), ("random_holdout", "random_holdout", "random_holdout", GATE_CORR),
                             ("within_donor_corr", "donor_within_mean", "donor_holdout", GATE_CORR),
                             ("within_branch_corr_plain_mean", "branch_within_mean", "branch_holdout", GATE_CORR),
                             ("global_corr", "global_spearman", "global_correlation", None)]:
    add("lung_control_v1_immune_real_labels", "lung control v1 (replaced)", metric, L1[key], "frozen head; recomputed from saved centroids", g,
        deployed=L1["deployed"][dkey], source="v2_10_internal_lotdo.json", n_anchors=L1["n_anchors"], n_donors=L1["n_donors"],
        note="first negative control: lung-resident immune cells with their real stage labels; passed all four gates; replaced by lung_nonhema")

# ------------------------------------------------------------- H38 / H95 / H103
fit = oo["fit"]; OB = oo["bootstrap"]
for name, lab in [("H38_lite", "H38_LITE"), ("H95", "H95")]:
    add(lab, "internal", "trustworthiness", fit[name]["internal"]["trust_in_sample_k15"], "in-sample, kept anchors, k=15",
        GATE_TRUST, source="v2_05_other_orderings.json", n_anchors=fit[name]["internal"]["n_anchors_kept"], k=15,
        deployed=fit["deployed"]["H95_internal_trust"] if name == "H95" else fit["deployed"]["H38_internal"]["positive"]["trustworthiness"])
    for panel, plab in [("external", "external"), ("zeroshot", "zero-shot")]:
        E = fit[name][panel]; key = f"{name}_{panel}"; B = OB.get(key, {})
        for m, val, g in [("trustworthiness", E["trustworthiness"], GATE_TRUST), ("random_holdout", E["random_holdout"], GATE_CORR),
                          ("within_donor_corr", E["donor_holdout"], GATE_CORR), ("within_category_corr", E["category_holdout"], GATE_CORR),
                          ("global_corr", E["global_correlation"], None)]:
            bk = {"trustworthiness": "trust", "within_donor_corr": "donor", "within_category_corr": "category"}.get(m)
            ci = B[bk]["ci95_percentile"] if bk and bk in B else None
            bl = B[bk].get(f"frac_below_{g}") if bk and bk in B else None
            add(lab, plab, m, val, "frozen head (phase15_validate_candidate.evaluate_frozen)", g, ci,
                DONOR_BOOT if ci else None, B.get("n_reps") if ci else None, "evaluation donor" if ci else None, 15 if m == "trustworthiness" else None,
                bl, deployed=(json.loads((REP / ("external_validation_H38_lite.json" if (name, panel) == ("H38_lite", "external") else
                              "zeroshot_H38_lite.json" if (name, panel) == ("H38_lite", "zeroshot") else
                              "zeroshot_H95.json" if (name, panel) == ("H95", "zeroshot") else "external_validation_h95.json")).read_text())
                              .get("trustworthiness") if m == "trustworthiness" else None),
                source="v2_05_other_orderings.json", n_anchors=E.get("n_anchors_kept"), n_donors=E.get("n_donors"),
                check=f"{key}|frozen_donor_boot|{bk}" if ci else None,
                lodo=(B["lodo_trust_range"] if m == "trustworthiness" else B.get("lodo_category_range") if m == "within_category_corr" else None) if B else None)
H = fit["H103"]
add("H103", "internal", "trustworthiness", H["internal"]["trust_in_sample_k15_kept"], "in-sample on 45 kept B-lineage anchors, k=15",
    GATE_TRUST, deployed=0.860, source="v2_05_other_orderings.json", n_anchors=45, k=15,
    note="ruler has 2 values only: 'b cell' (depth 1) vs 'plasma cell' (depth 5)")
add("H103", "internal", "trustworthiness_all_anchors", H["internal"]["trust_k15_all_internal_anchors"], "H103 head applied to all 290 internal anchors, k=15",
    GATE_TRUST, source="v2_05_other_orderings.json", n_anchors=290, k=15)
for panel, plab in [("external", "external"), ("zeroshot", "zero-shot")]:
    E = H[panel]; B = OB[f"H103_{panel}_all"]
    add("H103", plab, "trustworthiness_kept_k15", E["trust_kept_k15"], "kept B-lineage anchors, k=15 (deployed definition for external)", GATE_TRUST,
        deployed=0.877 if panel == "external" else None, source="v2_05_other_orderings.json", n_anchors=E["n_anchors_kept"],
        n_donors=E["n_donors_kept"], k=15, note=None if panel == "external" else "not computable: 18 anchors < 2k")
    add("H103", plab, "trustworthiness_kept_k5", E["trust_kept_k5"], "kept B-lineage anchors, k=5", GATE_TRUST,
        deployed=0.857 if panel == "zeroshot" else None, source="v2_05_other_orderings.json", n_anchors=E["n_anchors_kept"],
        n_donors=E["n_donors_kept"], k=5)
    add("H103", plab, "trustworthiness_all_anchors_k15", E["trust_all_anchors_k15"], "H103 head applied to ALL panel anchors, k=15",
        GATE_TRUST, B["trust"]["ci95_percentile"], DONOR_BOOT, B["n_reps"], "evaluation donor", 15, B["trust"].get("frac_below_0.8"),
        B["lodo_trust_range"], source="v2_05_other_orderings.json", n_anchors=E["n_all_anchors"], n_donors=E["n_donors_all"],
        check=f"H103_{panel}_all|frozen_donor_boot|trust")
    for m, val in [("global_corr_kept", E["global_spearman_kept"]), ("random_holdout_kept", E["random_holdout_kept"]),
                   ("within_donor_corr_kept", E["donor_holdout_kept"]), ("within_depth_corr_kept", E["depth_holdout_kept"])]:
        add("H103", plab, m, val, "kept B-lineage anchors, frozen head", None if m.startswith("global") else GATE_CORR,
            source="v2_05_other_orderings.json", n_anchors=E["n_anchors_kept"], n_donors=E["n_donors_kept"],
            note="NaN: each depth group has a constant ruler" if m.startswith("within_depth") else None)

tab = pd.DataFrame(rows)
tab.to_csv(OUT / "table_gates.csv", index=False)

# ------------------------------------------------------------- figure source data (H65 trust + branch per panel)
fig = []
for _, r in tab[(tab.ordering.isin(["H65", "lung_control_random_labels"])) &
                (tab.metric.isin(["trustworthiness", "branch_holdout_plain_mean", "within_branch_corr_plain_mean",
                                  "branch_holdout_anchor_weighted", "within_branch_corr_anchor_weighted",
                                  "trustworthiness_two_level", "within_branch_corr_plain_mean_two_level",
                                  "trustworthiness_training_donors_only", "within_branch_corr_plain_mean_training_donors_only"]))].iterrows():
    kind = "trustworthiness" if r.metric.startswith("trust") else "branch"
    endpoint = ("re-fit branch holdout (a branch is withheld)" if r.metric.startswith("branch_holdout")
                else "frozen-head within-branch correlation (nothing withheld)" if kind == "branch"
                else "trustworthiness (model features vs head output; ruler not used)")
    fig.append({"panel": r.panel, "quantity": kind, "weighting": "anchor-weighted" if "anchor_weighted" in r.metric else "plain mean",
                "interval_type": "two-level" if "two_level" in r.metric else ("training-donor bootstrap" if (r.panel == "internal" or "training_donors_only" in r.metric) else "evaluation-donor bootstrap"),
                "value": r.value, "ci_low": r.ci_low, "ci_high": r.ci_high, "basic_ci_low": r.basic_ci_low,
                "basic_ci_high": r.basic_ci_high, "boot_mean": r.boot_mean, "lodo_min": r.lodo_min, "lodo_max": r.lodo_max,
                "lodo_unit": r.lodo_unit, "frac_resamples_below_gate": r.frac_resamples_below_gate, "gate": r.gate, "endpoint": endpoint,
                "n_reps": r.n_reps, "n_anchors": r.n_anchors, "n_donors": r.n_donors, "deployed_value": r.deployed_value})
pd.DataFrame(fig).to_csv(OUT / "fig_gates.csv", index=False)
write_run_config("v2_07_tables", [OUT / n for n in ["v2_00_reproduce.json", "v2_01_frozen_donor_bootstrap.json",
                 "v2_03_refit_summary.json", "v2_04_permutation_frozen.json", "v2_04_permutation_internal.json",
                 "v2_05_other_orderings.json", "v2_06_facts.json", "facts_cohorts.csv", "v2_10_internal_lotdo.json",
                 "v2_11_interval_checks.json", "v2_00_per_branch.csv", "v2_03_internal_per_branch.csv"]], {})
print(tab[["ordering", "panel", "metric", "value", "ci_low", "ci_high", "passes_point", "frac_resamples_below_gate"]].to_string())
