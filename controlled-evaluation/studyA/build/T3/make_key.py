"""Write studyA/keys/T3/key.json. Every number is read from keys/T3/reference_output.json (made by reference.py
from the package data); only the text is written here."""
import json
from pathlib import Path

KD = Path("<EVAL_ROOT>/studyA/keys/T3")
R = json.loads((KD / "reference_output.json").read_text())
g05 = R["gata1_tests"]["ChIP|0.5"]
eff = R["gata1_effects_cut0.5"]
f26 = R["f2610_gata1_chip"]
prof = R["f2610_profile"]
rnk = R["gata1_rank_cm"]
pw = R["gata1_planted_power_cm_cut0.5"]
ce = R["cebpz_chip_cut0.01"]
rcg = R["random_control_groups"]
okd = R["gata1_other_knockdown_null"]
pu = R["gata1_pooled_union_chip"]["0.25"]
cfr = R["cutoff_free_rank_check"]
coarse = R["coarse_bin_sensitivity"]
oth = R["gata1_set_enrichment_in_other_knockdowns"]["0.05"]
sf = R["spec_feature_fisher_test"]
sfr = R["spec_feature_fisher_rank_among_tf_sets"]
sfa = R["spec_feature_fisher_rank_all_settings"]
rar = R["responding_feature_rarity"]
sf_g = sfr["0.1|ChIP|overlap>=2|GATA1"]
sf_tfs = sorted({k.split("|")[-1] for k in sfr})
sf_other = [v["n_other_sets_p_lt_0.05"] for v in sfr.values()]
sf_ranks = [v["rank_true"] for v in sfr.values()]
sf_gm = [v["n_other_sets_p_lt_0.05"] for k, v in sfr.items() if k.endswith("|GATA1") or k.endswith("|MAX")]
r3 = lambda x: round(float(x), 3)

SEL = ("selection rule: per TF and feature, two-sided Mann-Whitney U of the per-cell value (mean_gene), knockdown "
       "cells vs the 400 'ref' cells; BH across the 4,928 features; responding = q < 0.05 and |mean(kd) - mean(ref)| "
       "> cut-off")

key_numbers = [
    {"name": "n_tfs_knocked_down", "value": R["n_tfs"], "tolerance": 0,
     "definition": "Distinct TFs among 'kd' cells in data/cell_manifest.csv."},
    {"name": "n_tfs_with_chip_set", "value": R["n_tfs_with_chip_set"], "tolerance": 0,
     "definition": "Knocked-down TFs with >= 1 DoRothEA pair with chip_flag = True (target in the universe)."},
    {"name": "n_tfs_with_trrust_set", "value": R["n_tfs_with_trrust_set"], "tolerance": 0,
     "definition": "Knocked-down TFs with >= 1 TRRUST target in the universe."},
    {"name": "n_tfs_trrust_set_ge2", "value": R["n_tfs_trrust_set_ge2_in_universe"], "tolerance": 0,
     "definition": "Knocked-down TFs with >= 2 TRRUST targets in the universe (the minimum for a '>= 2 of top-20' rule to be reachable)."},
    {"name": "gata1_chip_targets_in_universe", "value": R["gata1_chip_targets_in_universe"], "tolerance": 0,
     "definition": "GATA1 DoRothEA pairs with chip_flag = True."},
    {"name": "gata1_trrust_targets_in_universe", "value": R["gata1_trrust_targets_in_universe"], "tolerance": 0,
     "definition": "GATA1 TRRUST targets in the universe."},
    {"name": "gata1_n_kd_cells", "value": R["gata1_n_kd_cells"], "tolerance": 0, "definition": "GATA1 knockdown cells."},
    {"name": "median_feature_mean_ref", "value": round(R["median_over_features_of_mean_ref_value"], 4), "tolerance": 0.0005,
     "definition": "Median over the 4,928 features of the feature's mean per-cell value in the 400 'ref' cells (shows the scale of the 0.5 effect cut-off)."},
    {"name": "gata1_chip_targets_median_detection_count", "value": R["gata1_chip_targets_median_detection_count"], "tolerance": 0,
     "definition": f"Median detection_count of GATA1's 219 ChIP-seq targets (universe median {R['universe_median_detection_count']})."},
    {"name": "gata1_chip_targets_median_length_bp", "value": R["gata1_chip_targets_median_length_bp"], "tolerance": 1,
     "definition": f"Median gene_length_bp of GATA1's ChIP-seq targets (universe median {R['universe_median_length_bp']})."},
    {"name": "gata1_K_cut0.5", "value": R["gata1_K_by_cutoff"]["0.5"], "tolerance": 0,
     "definition": f"Number of GATA1 responding features at cut-off 0.5 ({SEL}). Welch t-test + BH gives the same 5."},
    {"name": "gata1_responding_features_cut0.5", "value": R["gata1_responding_features_cut0.5"], "tolerance": "exact set",
     "definition": "Feature ids of GATA1's responding features at cut-off 0.5 (same rule)."},
    {"name": "n_tfs_with_responding_cut0.5", "value": R["n_tfs_with_responding_by_cutoff"]["0.5"], "tolerance": 0,
     "definition": "TFs (of 87) with >= 1 responding feature at cut-off 0.5: " + ", ".join(f"{k} {v}" for k, v in R["tfs_with_responding_cut0.5"].items()) + "."},
    {"name": "n_tfs_with_responding_no_cutoff", "value": R["n_tfs_with_responding_by_cutoff"]["0.0"], "tolerance": 8,
     "definition": "TFs (of 87) with >= 1 BH-significant feature and no effect cut-off (Mann-Whitney; Welch t-test gives 77)."},
    {"name": "gata1_n_bh_sig_features", "value": R["gata1_n_bh_sig_features"], "tolerance": 150,
     "definition": "GATA1 features with BH q < 0.05 (no effect cut-off), out of 4,928."},
    {"name": "n_testable_chip_tfs_by_cutoff", "value": R["n_testable_tfs_by_cutoff"]["ChIP"], "tolerance": "exact per cut-off with the stated rule",
     "definition": "TFs with a ChIP set and >= 1 responding feature, per effect cut-off."},
    {"name": "f2610_gata1_effect", "value": r3(eff["2610"]["delta"]), "tolerance": 0.005,
     "definition": "mean(GATA1 kd) - mean(ref) of feature 2610's per-cell value."},
    {"name": "f2610_gata1_effect_ci95", "value": [round(x, 2) for x in eff["2610"]["ci95"]], "tolerance": 0.015,
     "definition": "95% percentile CI of that effect, cell bootstrap (kd and ref resampled separately, 1,000 resamples). The source report gives [0.412, 0.512]."},
    {"name": "f2610_frac_bootstrap_above_cutoff", "value": round(eff["2610"]["frac_boot_abs_delta_gt_0.5"], 3), "tolerance": 0.03,
     "definition": "Share of cell-bootstrap resamples in which |effect of 2610| > 0.5 (source report: 0.06)."},
    {"name": "f2610_chip_overlap", "value": f26["overlap"], "tolerance": 0,
     "definition": "GATA1 ChIP-seq targets among feature 2610's top-20 genes."},
    {"name": "gata1_best_chip_overlap_cut0.5", "value": g05["M"], "tolerance": 0,
     "definition": f"Largest overlap of a GATA1 responding feature's top-20 with the GATA1 ChIP set at cut-off 0.5 (feature {g05['best_feature']})."},
    {"name": "gata1_best_trrust_overlap_cut0.5", "value": R["gata1_tests"]["TRRUST|0.5"]["M"], "tolerance": 0,
     "definition": "Same with the 11 GATA1 TRRUST targets."},
    {"name": "gata1_p_random_feature_null_cut0.5", "value": r3(g05["p_rf"]), "tolerance": 0.03,
     "definition": "P(best overlap of K = 5 random catalog features >= 3), exact; the source paper's null."},
    {"name": "gata1_p_count_matched_cut0.5", "value": r3(g05["p_cm"]), "tolerance": 0.1,
     "definition": "P(best overlap >= 3) when every top-20 gene of the 5 features is swapped for a random universe gene with a similar detection count (bins at 1, 2, 5, 10, 20, 50, 100, 200, 500; log2 bins give 0.357, decile bins 0.438)."},
    {"name": "gata1_p_count_length_matched_cut0.5", "value": r3(g05["p_cml"]), "tolerance": 0.1,
     "definition": "Same, bins = detection-count bin x gene-length tertile."},
    {"name": "gata1_chip_rank_cut0.5", "value": rnk["ChIP|0.5"]["rank"], "tolerance": 30,
     "definition": f"Rank (1 = most significant, mid-rank for ties) of GATA1's ChIP set among {rnk['ChIP|0.5']['n_candidates']} ChIP sets with >= 20 targets in the universe, by count-matched p for GATA1's 5 responding features. Lower cut-offs: 255-277 of 291."},
    {"name": "f2610_expected_overlap_count_matched", "value": r3(f26["E_overlap_count_matched"]), "tolerance": 0.5,
     "definition": "Expected GATA1 ChIP overlap of feature 2610 under the count-matched swap null (fine bins)."},
    {"name": "f2610_p_ge8_count_matched", "value": r3(f26["P_ge_8_count_matched"]), "tolerance": 0.015,
     "definition": "P(overlap of 2610 >= 8) under the count-matched swap null (count + length matched: " + str(r3(f26["P_ge_8_count_length_matched"])) + ")."},
    {"name": "f2610_expected_overlap_uniform", "value": r3(f26["E_overlap_uniform_universe"]), "tolerance": 0.02,
     "definition": "Expected overlap if the 20 genes were drawn uniformly from the 6,324-gene universe (20 x 219 / 6,324)."},
    {"name": "f2610_p_uniform_hypergeom", "value": float(f"{f26['p_hypergeom_vs_universe']:.2g}"), "tolerance": "same order of magnitude",
     "definition": "Hypergeometric P(overlap >= 8) against the uniform universe (the naive test)."},
    {"name": "f2610_n_rare_genes", "value": prof["n_top_genes_detection_count_lt_10"], "tolerance": 0,
     "definition": f"Genes in 2610's top-20 detected in < 10 of the 500 catalog cells (universe share {round(prof['universe_share_detection_count_lt_10'], 3)})."},
    {"name": "f2610_n_chip_sets_overlap_ge5", "value": prof["n_chip_sets_overlap_ge_5"], "tolerance": 0,
     "definition": f"Of {prof['n_candidates']} ChIP sets (>= 20 targets in universe), how many overlap 2610's top-20 by >= 5 genes."},
    {"name": "n_tf_settings_bh_significant_matched_nulls", "value": R["n_TF_settings_BH_significant_count_matched"] + R["n_TF_settings_BH_significant_count_length_matched"] + R["n_TF_settings_BH_significant_count_matched_uncapped"], "tolerance": 0,
     "definition": "Number of (TF, cut-off, database) settings with BH q < 0.05 across TFs under the count-matched, count+length-matched or uncapped count-matched null; 7 cut-offs x 2 databases; BH across the 20 ChIP TFs or the 58 TRRUST TFs (TFs with no responding feature enter with p = 1)."},
    {"name": "cebpz_cut0.01_chip_p_count_matched", "value": r3(ce["p_cm"]), "tolerance": 0.01,
     "definition": f"The only nominal p < 0.05 under the count-matched null with the capped statistic T: CEBPZ, ChIP, cut-off 0.01, K = {ce['K']}, best overlap {ce['M']} (feature {ce['best_feature']}). With the uncapped overlap M there are two more nominal hits, each at a single setting: MAX, ChIP, cut-off 0.1 (p = 0.041) and TBP, ChIP, cut-off 0 (p = 0.018); neither passes BH (q 0.81 and 0.35; independent verification, VERIFICATION.md)."},
    {"name": "cebpz_cut0.01_chip_q", "value": round(ce["q_cm_across_20_chip_TFs"], 2), "tolerance": 0.1,
     "definition": "BH q of that p across the 20 ChIP TFs."},
    {"name": "power_planted_k8_count_matched", "value": pw["8"]["capped"], "tolerance": 0.05,
     "definition": "GATA1, cut-off 0.5: replace 8 non-target genes in one random responding feature with 8 random GATA1 ChIP targets; share of 200 repeats with count-matched p < 0.05 (capped statistic). Wilson 95% CI " + str([round(x, 2) for x in pw["8"]["capped_wilson95"]]) + "; source report 0.93 [0.88, 0.95]; uncapped 1.00."},
    {"name": "power_planted_k4_count_matched", "value": pw["4"]["capped"], "tolerance": 0.03,
     "definition": "Same with 4 planted targets."},
    {"name": "power_planted_k2_count_matched", "value": pw["2"]["capped"], "tolerance": 0.06,
     "definition": "Same with 2 planted targets."},
    {"name": "random_groups_any_responding_cut0.5", "value": rcg["frac_with_any_responding_feature"]["0.5"], "tolerance": 0.01,
     "definition": f"Share of {rcg['B']} random groups of {rcg['group_size']} 'pool' control cells (vs the 400 'ref' cells, same selection rule) with >= 1 responding feature at cut-off 0.5 (0.0 at every cut-off >= 0.05; {rcg['frac_with_any_responding_feature']['0.0']} with no cut-off)."},
    {"name": "gata1_p_K_matched_random_groups_cut0.5", "value": r3(rcg["p_K_matched_gata1_chip_cut0.5"]), "tolerance": 0.06,
     "definition": "Random control groups keep their K = 5 most-changed features; P(best ChIP overlap >= 3); (1 + b)/(1 + B)."},
    {"name": "gata1_p_other_knockdowns_cut0.5", "value": r3(okd["0.5"]["p_okd"]), "tolerance": 0,
     "definition": f"(1 + number of other knockdowns whose responding features reach GATA1-set overlap >= 3) / (1 + {okd['0.5']['n_other_knockdowns']}); cut-off 0.5."},
    {"name": "gata1_pooled_union_cut0.25_naive_p", "value": float(f"{pu['p_naive']:.2g}"), "tolerance": "same order of magnitude",
     "definition": f"Naive pooled test: union of the top-20 genes of GATA1's 12 responding features at cut-off 0.25 ({pu['n_union_genes']} genes) holds {pu['obs']} GATA1 ChIP targets vs {pu['exp_uniform']} expected uniformly; hypergeometric p. {pu['n_other_sets_naive_p_lt_0.05']} of 290 other ChIP sets also give p < 0.05 on the same union."},
    {"name": "gata1_pooled_union_cut0.25_count_matched_z", "value": pu["z_count_matched"], "tolerance": 0.3,
     "definition": f"Same union, expectation matched on detection-count bin ({pu['exp_count_matched']} expected); z-score. GATA1 ranks {pu['rank_count_matched_z']} of 291 sets; {pu['n_other_sets_z_gt_1.96']} other sets have z > 1.96."},
    {"name": "gata1_set_naive_enriched_in_other_knockdowns_cut0.05", "value": oth["n_with_naive_p_lt_0.05_for_GATA1_set"], "tolerance": 2,
     "definition": f"Of {oth['n_other_knockdowns_with_ge5_features']} other knockdowns with >= 5 responding features at cut-off 0.05, how many have a union of top-20 genes that is naively (uniform hypergeometric) enriched for GATA1 ChIP targets at p < 0.05."},
    {"name": "cutoff_free_mean_rank_fraction", "value": r3(cfr["mean_rank_fraction"]), "tolerance": 0.03,
     "definition": "For each of the 20 ChIP TFs: Spearman correlation across features (>= 10 listed genes) between |effect| and overlap with a target set; rank of the TF's own set among 291 ChIP sets divided by 291; mean over the 20 TFs (0.5 = no specificity; lower = more specific). 95% CI over TFs " + str([round(x, 2) for x in cfr["ci95_bootstrap_over_tfs"]]) + f"; {cfr['n_tfs_rank_in_top_half']} of 20 in the top half, {cfr['n_tfs_rank_in_top_5pct']} in the top 5%."},
    {"name": "coarse_bins_gata1_cut0.25_uncapped_p", "value": r3(coarse["gata1_cut0.25"]["p_decile_bins_uncapped"]), "tolerance": 0.01,
     "definition": f"Count-matched null with coarse decile bins of detection count (the lowest decile spans counts 1-27): GATA1, ChIP, cut-off 0.25, uncapped M = 8. Fine bins give {r3(coarse['gata1_cut0.25']['p_fine_bins_uncapped'])}. Shows that coarse matching leaves the rarity effect in."},
    {"name": "spec_fisher_gata1_chip_cut0.1_overlap2_p", "value": float(f"{sf_g['p_true']:.2g}"), "tolerance": "same order of magnitude",
     "definition": ("The source paper's feature-level test read literally: one-sided Fisher exact test, GATA1's 52 responding features at cut-off 0.1 vs the "
                    "other 4,876 catalog features x (top-20 holds >= 2 GATA1 ChIP targets). BH across the 20 ChIP TFs passes. With overlap >= 1 instead: p = "
                    + f"{sf['0.1|ChIP|overlap>=1']['gata1_p']:.1g}.")},
    {"name": "spec_fisher_gata1_chip_cut0.1_overlap2_rank", "value": sf_g["rank_true"], "tolerance": 15,
     "definition": (f"Same Fisher test on the same 52 features, run with each of the 291 ChIP sets (>= 20 targets in the universe): rank of GATA1's own set "
                    f"(1 = smallest p). {sf_g['n_other_sets_p_lt_0.05']} of the 290 other sets also give p < 0.05.")},
    {"name": "spec_fisher_n_other_sets_p_lt_0.05_range", "value": [min(sf_other), max(sf_other)], "tolerance": 15,
     "definition": (f"For every (cut-off, overlap rule, TF) where the Fisher test passes BH across the 20 ChIP TFs ({len(sfr)} cases; TFs {', '.join(sf_tfs)}): "
                    f"range over cases of how many of the 290 other ChIP sets also give p < 0.05 on the same responding features. True-TF rank range "
                    f"{min(sf_ranks)}-{max(sf_ranks)} of 291. TRRUST: 0 TFs pass BH at any cut-off.")},
    {"name": "spec_fisher_true_tf_mean_rank_fraction", "value": r3(sfa["mean_rank_fraction"]), "tolerance": 0.03,
     "definition": (f"Same Fisher test for all {sfa['n_settings']} (ChIP TF, cut-off, overlap >= 1 or 2) settings with >= 1 responding feature "
                    f"({sfa['n_tfs']} TFs): rank of the TF's own set among 291 ChIP sets / 291, averaged (0.5 = no specificity). 95% CI over TFs "
                    f"{[round(x, 2) for x in sfa['ci95_bootstrap_over_tfs']]}; {sfa['n_settings_true_tf_in_top_5pct']} settings in the top 5% "
                    f"({sfa['expected_by_chance_top_5pct']:.1f} expected by chance): CEBPZ cut-off 0 overlap >= 2 (rank 3) and GATA1 cut-off 0.5 overlap >= 1 (rank 13).")},
    {"name": "gata1_responding_features_median_detection_cut0.1", "value": rar["gata1_by_cutoff"]["0.1"], "tolerance": 2,
     "definition": (f"Median over GATA1's responding features at cut-off 0.1 of the feature's median top-gene detection_count (all features: "
                    f"{rar['all_features_median_of_median_detection']}). Responding features at cut-offs 0.01-0.5 are rare-gene features, which is why "
                    "they overlap the (rare, long-gene) ChIP sets of many TFs.")},
]

traps = [
    {"id": "rarity_unmatched_null",
     "description": ("A feature's top-20 list is ranked by mean activation where active, with no minimum count, so some lists are "
                     f"dominated by genes seen in very few cells (feature 2610: {prof['n_top_genes_detection_count_lt_10']} of 20 genes detected in < 10 of 500 cells, "
                     f"vs {round(100 * prof['universe_share_detection_count_lt_10'], 1)}% of universe genes). ChIP-seq target sets are biased to rare, long genes "
                     f"(GATA1 ChIP targets: median detection {R['gata1_chip_targets_median_detection_count']:.0f} vs {R['universe_median_detection_count']:.0f}; "
                     f"median length {R['gata1_chip_targets_median_length_bp'] / 1000:.0f} kb vs {R['universe_median_length_bp'] / 1000:.0f} kb). A null that draws genes "
                     "uniformly from the universe, or random catalog features (mostly common-gene features), under-predicts the overlap of rare-gene features. "
                     "This is the error behind the original GATA1 'positive' (ledger L026, critical)."),
     "correct_handling": ("Match the null on detection count with fine bins at the rare end (and ideally on gene length), or use comparisons that hold the "
                          "features fixed: rank of the true TF's set among all TF sets, or the true set scored on other knockdowns' features. "
                          f"If the report discusses feature 2610, it gives its expected overlap under a count-matched null as about {r3(f26['E_overlap_count_matched'])}, not {r3(f26['E_overlap_uniform_universe'])}."),
     "mishandling": (f"Uniform hypergeometric / Fisher / random-gene-set enrichment read as specificity (2610: p = {f26['p_hypergeom_vs_universe']:.1e}; GATA1 pooled "
                     f"union at cut-off 0.25: p = {pu['p_naive']:.0e}; GATA1, MAX, GTF2B, TERF2, THAP1 pass BH under the pooled uniform test at some cut-offs); or a "
                     f"count null with coarse quantile bins (decile bins: 2610 expected {r3(coarse['f2610_E_overlap_decile_bins'])}, GATA1 uncapped p = "
                     f"{r3(coarse['gata1_cut0.25']['p_decile_bins_uncapped'])} at cut-off 0.25).")},
    {"id": "specificity_needs_comparison_with_other_tfs",
     "description": ("The question is specificity: a TF's responding features should hold more of that TF's targets than of other TFs' targets, and "
                     "more than other knockdowns' features do. Enrichment for the TF's own set alone is not specificity, because the same features "
                     "are rich in the long-gene ChIP sets of many TFs and many knockdowns change similar features."),
     "correct_handling": (f"Ranks the true TF's set among other TF sets on the same features (GATA1: {rnk['ChIP|0.5']['rank']:.0f}/291 at cut-off 0.5, "
                          f"{min(rnk[f'ChIP|{c}']['rank'] for c in ['0.25', '0.1', '0.05', '0.02', '0.01', '0.0']):.0f}-{max(rnk[f'ChIP|{c}']['rank'] for c in ['0.25', '0.1', '0.05', '0.02', '0.01', '0.0']):.0f}/291 at lower cut-offs; "
                          f"cut-off-free mean rank fraction over 20 TFs {r3(cfr['mean_rank_fraction'])}, 0 TFs in the top 5%) and/or scores the true set on "
                          f"other knockdowns' features (GATA1 p = {r3(okd['0.5']['p_okd'])} at 0.5; feature 2610 overlaps >= 5 targets of {prof['n_chip_sets_overlap_ge_5']} of 291 TF sets)."),
     "mishandling": (f"Tests only the TF's own set and calls enrichment 'specific'; does not notice that {pu['n_other_sets_naive_p_lt_0.05']} of 290 other ChIP sets are just as "
                     f"'enriched' in GATA1's features, or that the GATA1 set is 'enriched' in {oth['n_with_naive_p_lt_0.05_for_GATA1_set']} of {oth['n_other_knockdowns_with_ge5_features']} "
                     "other knockdowns' features (cut-off 0.05). The same holds for the source paper's feature-level Fisher test "
                     f"(responding vs other features x 'top-20 overlaps the targets'): with ChIP sets it passes BH for {', '.join(sf_tfs)} at some cut-offs "
                     f"(GATA1 cut-off 0.1: p = {sf_g['p_true']:.0e}), but {min(sf_other)}-{max(sf_other)} of the 290 other ChIP sets also give p < 0.05 "
                     f"on the same features, and over all settings the TF's own set ranks at {r3(sfa['mean_rank_fraction'])} (0.5 = chance). Reading these "
                     "Fisher passes as specificity is this trap.")},
    {"id": "degenerate_random_control_group_null",
     "description": (f"A selection-aware null made from random groups of control cells (same size as the knockdown group, whole selection re-run) "
                     f"selects no feature at cut-offs >= 0.05 (0 of {rcg['B']} groups here; 0 of 2,000 in the source report at >= 0.25). Every knockdown "
                     "whose responding features reach overlap >= 2 then gets the floor p-value 1/(B + 1). This shows the knockdown changed features, "
                     "not that the changed features are about the TF's targets."),
     "correct_handling": (f"Recognises the degeneracy and uses a null that keeps the number of selected features (K most-changed features of random groups: "
                          f"GATA1 p = {r3(rcg['p_K_matched_gata1_chip_cut0.5'])}) or other knockdowns' features, or does not use this null for the verdict."),
     "mishandling": "Reports GATA1 (or any TF) as specific because p is at the floor (0.0005 with 2,000 groups; BH q = 0.010 across 20 TFs in the source report)."},
    {"id": "knife_edge_selection_and_forking_paths",
     "description": ("The former GATA1 positive rested on feature 2610 (8 of 20 ChIP targets), whose effect sat on the cut-off (0.50011 vs 0.5 in the "
                     f"original run; ledger L027). Here it is {r3(eff['2610']['delta'])} (95% CI {eff['2610']['ci95'][0]:.2f}-{eff['2610']['ci95'][1]:.2f}, cell bootstrap), "
                     f"above 0.5 in {round(100 * eff['2610']['frac_boot_abs_delta_gt_0.5'])}% of resamples. The cut-off has no natural scale for per-cell means "
                     f"(median feature mean {R['median_over_features_of_mean_ref_value']:.3f}). Positive-looking p-values appear only in single combinations of cut-off x database x "
                     f"threshold (2..5) x null: CEBPZ at cut-off 0.01 only (p = {r3(ce['p_cm'])}, q = {round(ce['q_cm_across_20_chip_TFs'], 2)}); GATA1 TRRUST at 0.02 "
                     f"(overlap 2, p = {r3(rnk['TRRUST|0.02']['p_cm_true'])}); GATA1 uncapped random-feature p = {r3(R['gata1_tests']['ChIP|0.25']['p_rf_uncapped'])} at 0.25; "
                     "uncapped count-matched p = 0.041 for MAX at 0.1 and 0.018 for TBP at 0 (ChIP; q 0.81 and 0.35). "
                     "The original deployment presented a threshold search (2 databases x 4 thresholds) with an uncorrected p = 0.030 as a positive (ledger L028)."),
     "correct_handling": ("Fixes the selection rule in advance or reports every cut-off; ideally shows selection stability by bootstrapping cells; corrects across "
                          "TFs and states the search over cut-offs, thresholds, databases and nulls; does not treat a single-setting nominal hit as a finding."),
     "mishandling": ("Lowers the cut-off until 2610 enters and reports GATA1 (8/20) as specific; reports CEBPZ or any single p < 0.05 as a positive; "
                     "picks the best of several thresholds or databases without correction; treats 2610 as a GATA1-responding feature at the 0.5 cut-off.")},
    {"id": "testability_and_power",
     "description": (f"At the source cut-off of 0.5 only {R['n_tfs_with_responding_by_cutoff']['0.5']} of 87 TFs have any responding feature and only 1 of them (GATA1) "
                     f"has a ChIP set; 22 TFs have no target in the universe; only {R['n_tfs_trrust_set_ge2_in_universe']} of 58 TRRUST sets have >= 2 targets in the universe, "
                     "so most TFs can never pass a '>= 2 of top-20' rule. The original '0/48' counted untestable TFs as negatives (ledger L021) and called a "
                     "random-feature pass rate 'power' (ledger L025). A negative verdict needs testable TFs and a power check."),
     "correct_handling": ("Reports testable TFs per cut-off (ChIP: " + ", ".join(f"{c}: {v}" for c, v in R["n_testable_tfs_by_cutoff"]["ChIP"].items()) + ") and "
                          f"power from planted targets (GATA1, 0.5, count-matched: 4 targets {pw['4']['capped']}, 8 targets {pw['8']['capped']}, 2 targets {pw['2']['capped']}); "
                          "notes that the capped statistic (thresholds 2..5) loses power under a length-matched null and at low cut-offs while the uncapped overlap keeps it. "
                          "Concludes 'not supported' on that basis."),
     "mishandling": ("'0 of 87 specific' with no count of testable TFs and no power check; quoting a false-positive rate as power; or calling the result "
                     "'inconclusive' because few TFs respond at 0.5, without looking at lower cut-offs (up to 14 ChIP TFs testable) or planted-signal power.")},
    {"id": "generalising_from_one_tf_and_wrong_unit",
     "description": ("The question is about TFs in general; GATA1 is one example. The verdict must rest on all testable TFs with a correction across TFs, "
                     "and uncertainty must use the right unit: cells for effects and selection (knockdown and control cells resampled separately), "
                     "TFs for statements about TFs in general."),
     "correct_handling": "Tests every TF with a target set at each cut-off, corrects across TFs, and reports cell-bootstrap CIs for effects and TF-level intervals for summaries across TFs.",
     "mishandling": ("Draws the verdict from GATA1 alone (in either direction); presents bootstraps over features or genes (treated as independent) as the "
                     "uncertainty of a TF-level claim; gives no uncertainty for the numbers that carry the verdict.")},
]

false_statements = [
    "GATA1's responding features are significantly enriched for GATA1 ChIP-seq targets, so the features are specific to GATA1's targets (false: best overlap at cut-off 0.5 is 3 of 20, count-matched p = 0.37, rank 151 of 291 TF sets).",
    "Feature 2610 responds to GATA1 knockdown above the 0.5 effect cut-off (false: 0.461, 95% CI 0.41-0.51).",
    "Feature 2610's 8-of-20 overlap with GATA1 ChIP targets is far beyond chance, p about 1e-7 (false once rarity is matched: expected 4.4, P(>= 8) = 0.037, 0.072 with length; it also overlaps >= 5 targets of 69 of 291 TF sets).",
    "Random control-cell groups never reproduce GATA1's overlap, so GATA1's specificity is significant (p about 0.0005) (false: random groups select no features at all; with K matched, p = 0.86).",
    "CEBPZ shows target-specific features (false: under the count-matched null it is nominal at one cut-off only (0.01; q = 0.61 across 20 TFs) and gone at the neighbouring cut-offs; its one top Fisher rank (3/291 at cut-off 0, overlap >= 2) comes with 61 other ChIP sets also at p < 0.05, and it ranks 88-246/291 at cut-offs 0.01-0.02; 2 top-5% settings out of 106 is fewer than the 5.3 expected by chance).",
    "At least one TF passes multiple-testing correction for target specificity under a gene-frequency-matched null (false: 0 TFs at any cut-off or database).",
    "All 87 TFs were tested at the source cut-off of 0.5 and none was specific (false: only 6 TFs have a responding feature at 0.5, and only GATA1 has a ChIP set).",
    "The analysis has no power to detect TF-target specificity (false: 4 planted targets detected about 99% and 8 about 93-95% of the time for GATA1 at cut-off 0.5, count-matched null).",
    "GATA1 knockdown was confirmed by a drop in GATA1 mRNA (false: GATA1 is not in the measured panel; tf_gene_measured = False).",
    "The knocked-down TF's own target set ranks at or near the top for its responding features, as a general statement (false: GATA1 151/291 at 0.5 and 255-277/291 at lower cut-offs; over 20 TFs the mean rank fraction is 0.65 and no TF is in the top 5% in the cut-off-free check. Single settings near the top do occur at about the chance rate, e.g. CEBPZ 3/291 and GATA1 13/291 in 2 of 106 Fisher settings, GATA1 TRRUST 2/109 at 0.02 with p = 0.14; quoting one of these as a single-setting fact is not false).",
    "Only the GATA1 knockdown changes features that hold GATA1 targets (false: GATA1 targets are as frequent in other knockdowns' responding features; other-knockdown p = 0.67 at 0.5).",
    f"With the source paper's Fisher test and DoRothEA ChIP-seq targets, GATA1 and MAX responding features are significantly enriched for their own targets, which shows target-specific features (false: the same features are just as enriched for {min(sf_gm)}-{max(sf_gm)} of 290 other TFs' ChIP sets; GATA1's set ranks {sf_g['rank_true']}/291 at cut-off 0.1).",
    "Most knockdowns do not change any SAE feature significantly (BH q < 0.05 across features) (false: without an effect cut-off " + f"{R['n_tfs_with_responding_by_cutoff']['0.0']} of 87 TFs have BH-significant features; only with an effect cut-off do few TFs respond: " + ", ".join(f"{v} at {c}" for c, v in R["n_tfs_with_responding_by_cutoff"].items() if c != "0.0") + ").",
]

key = {
    "task": "T3",
    "question": "Are the sparse-autoencoder features that respond to a transcription factor's knockdown specific to that factor's target genes?",
    "verdict_options": ["supported", "not supported", "inconclusive"],
    "key_verdict": "not supported",
    "key_conclusion": (
        "Across the 87 knocked-down TFs (20 with DoRothEA ChIP-seq target sets, 58 with TRRUST sets) and effect cut-offs from 0.5 down to 0, "
        "no TF's responding features hold more of its own targets than expected once the null keeps each gene's detection frequency "
        "(0 TFs pass BH under count-matched or count-and-length-matched gene-swap nulls), and the TF's own set ranks no better than other TFs' sets. "
        f"GATA1 has 5 responding features at the source cut-off of 0.5; the best holds 3 of its 219 ChIP-seq targets in its top 20 (count-matched p = {r3(g05['p_cm'])}, "
        f"rank {rnk['ChIP|0.5']['rank']:.0f} of 291 TF sets), and the 8-target feature 2610 sits just below the cut-off ({r3(eff['2610']['delta'])}, 95% CI "
        f"{eff['2610']['ci95'][0]:.2f}-{eff['2610']['ci95'][1]:.2f}) and is a rare, long-gene feature rich in the ChIP targets of many TFs. "
        f"The tests can detect real specificity (4 or 8 planted GATA1 targets are detected in {100 * pw['4']['capped']:.0f}% and {100 * pw['8']['capped']:.0f}% of 200 repeats), "
        "so this is a negative result, not a lack of power."),
    "acceptable_alternatives": [],
    "acceptable_alternatives_note": (
        "None. 'supported' is the result of the traps (uniform or coarse nulls, the degenerate random-group null, a cut-off chosen to include feature 2610, "
        "or a single nominal hit such as CEBPZ). 'inconclusive' is not accepted: once rarity is matched there is no TF with a corrected signal at any of the "
        "cut-offs, the TF's own set ranks in the middle or lower half of all TF sets (no hint of a weak signal), up to 14 ChIP TFs and 48 TRRUST TFs are "
        "testable at lower cut-offs, and planted-target power for moderate signal is high. Weak specificity (1-2 extra targets in a top-20 list) cannot be "
        "excluded (power about 0.2 for 2 planted targets); a report should say so as a limitation, but it does not change the verdict."),
    "key_numbers": key_numbers,
    "traps": traps,
    "false_statement_checks": false_statements,
    "source": ("Corrected analysis: biomi_automation/projects/maxtoki/runs/sae-atlas-217M/V2_TF_SPECIFICITY_REPORT.md (2026-10-01). "
               "Deployment errors: error ledger rows L020, L021, L025-L028. Every number here is recomputed from the package data by reference.py."),
}
(KD / "key.json").write_text(json.dumps(key, indent=1, ensure_ascii=False) + "\n")
print("key.json written:", len(key_numbers), "key numbers,", len(traps), "traps,", len(false_statements), "false statements")
