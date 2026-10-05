"""Write studyA/keys/T2/key.json. Every key number is read from keys/T2/reference_output.json
(the key's own recomputation from the package data). The numbers that the source report also gives
(V2_CIRCUIT_REPORT.md, section 0 and 6) were checked to match exactly by check_against_source.py
(128 of 128 quantities). The prose cites the same values, rounded."""
import json
from pathlib import Path

K = Path("<EVAL_ROOT>/studyA/keys/T2")
r = json.loads((K / "reference_output.json").read_text())
M, V, W = r["main"], r["variants"], r["within"]
P = M["pooled"]; TB = M["baselines"]["target_direction_cross_fitted"]
BG = M["bootstrap_by_silenced_gene"]["stats"]; BF = M["bootstrap_by_source_feature_group"]["stats"]
BC = M["bootstrap_by_component"]["stats"]
VV = V["variants"]; SUB = V["subsets_abs_lfc"]; FAM = V["threshold_family_test"]

PAIRS = ("All 698,624 rows of data/gene_pairs.parquet (227 silenced genes); prediction = predicted_decrease "
         "(the method's rule); observed decrease = lfc < 0 (no lfc is exactly 0); positive class = decrease.")
GENE_CI = "Percentile bootstrap over the 227 silenced genes (pooled metric from summed counts), 2,000 resamples."

nums = []


def add(name, value, tol, definition):
    nums.append({"name": name, "value": value, "tolerance": tol, "definition": definition})


ff = M["file_facts"]
add("n_gene_pairs", ff["n_gene_pairs"], 0, "Rows in data/gene_pairs.parquet.")
add("n_silenced_genes", ff["n_silenced_genes"], 0, "Distinct silenced_gene values (= rows of silenced_genes.tsv).")
add("n_target_genes", ff["n_target_genes"], 0, "Distinct target_gene values.")
add("n_tied_pairs", ff["n_ties_frac_half"], 0,
    "Pairs with n_inhibitory_evidence / evidence exactly 0.5 (10.3%); the method's rule predicts 'increase' for them.")
add("frac_observed_decrease", P["frac_obs_dec"], 0.0005,
    PAIRS + " Share of pairs with lfc < 0. Equals the accuracy of always predicting 'decrease', the best constant rule.")
add("frac_predicted_decrease", P["frac_pred_dec"], 0.0005, PAIRS + " Share of pairs with predicted_decrease = True.")
add("accuracy", P["accuracy"], 0.0005, PAIRS + " Share of pairs where predicted_decrease equals (lfc < 0).")
add("accuracy_ci_low", BG["pooled_accuracy"]["ci95"][0], 0.004,
    GENE_CI + f" Upper bound {BG['pooled_accuracy']['ci95'][1]}. A pair-level interval "
    f"({M['pair_level_naive']['accuracy_ci95_pairs_independent']}) ignores the clustering and is far too narrow.")
add("accuracy_ci_high", BG["pooled_accuracy"]["ci95"][1], 0.004, GENE_CI)
add("accuracy_minus_always_decrease", BG["pooled_acc_minus_always_dec"]["point"], 0.001,
    "Accuracy minus the share of observed decreases (always-'decrease' rule).")
add("accuracy_minus_always_decrease_ci_low", BG["pooled_acc_minus_always_dec"]["ci95"][0], 0.006,
    GENE_CI + f" Upper bound {BG['pooled_acc_minus_always_dec']['ci95'][1]}. Resampling source-feature groups "
    f"(119): {BF['pooled_acc_minus_always_dec']['ci95']}; connected components (50): {BC['pooled_acc_minus_always_dec']['ci95']}.")
add("accuracy_minus_always_decrease_ci_high", BG["pooled_acc_minus_always_dec"]["ci95"][1], 0.006, GENE_CI)
add("balanced_accuracy", P["balanced_acc"], 0.0005,
    PAIRS + " Mean of recall on decreases and recall on increases.")
add("balanced_accuracy_minus_half_ci_low", BG["pooled_balanced_acc_minus_half"]["ci95"][0], 0.003,
    GENE_CI + f" Upper bound {BG['pooled_balanced_acc_minus_half']['ci95'][1]}. By source-feature group "
    f"{BF['pooled_balanced_acc_minus_half']['ci95']}; by component {BC['pooled_balanced_acc_minus_half']['ci95']}.")
add("balanced_accuracy_minus_half_ci_high", BG["pooled_balanced_acc_minus_half"]["ci95"][1], 0.003, GENE_CI)
add("mcc", P["mcc"], 0.0005, PAIRS + " Matthews correlation coefficient from the pooled 2x2 table.")
add("mcc_ci_low", BG["pooled_mcc"]["ci95"][0], 0.004,
    GENE_CI + f" Upper bound {BG['pooled_mcc']['ci95'][1]}. By source-feature group {BF['pooled_mcc']['ci95']}; "
    f"by component {BC['pooled_mcc']['ci95']}.")
add("mcc_ci_high", BG["pooled_mcc"]["ci95"][1], 0.004, GENE_CI)
add("cohen_kappa", P["cohen_kappa"], 0.0005, PAIRS + " Cohen's kappa of prediction vs observation.")
add("chance_accuracy_from_marginals", P["chance_acc_given_marginals"], 0.0005,
    "p_pred x p_obs + (1 - p_pred) x (1 - p_obs): the accuracy expected if predictions were independent of the "
    f"observations with the same two class rates. Accuracy minus this: {P['acc_minus_chance_given_marginals']} "
    f"(gene-bootstrap CI {BG['pooled_acc_minus_chance_given_marginals']['ci95']}).")
add("precision_of_decrease_predictions", P["precision_pred_dec"], 0.0005,
    "Share of pairs predicted 'decrease' that decrease. Below the base rate of decreases "
    f"({P['frac_obs_dec']}); its pair-level z against 0.5 ({M['pair_level_naive']['precision_pred_dec_vs_half_z']}) "
    "is the wrong comparison.")
add("fraction_decrease_ignoring_prediction", M["fraction_of_pairs_observed_decrease"], 0.0005,
    "Share of all pairs with lfc < 0, whatever the prediction. This is what the original deployed code reported "
    "as 'directional accuracy' (54.61% with its edges); it does not measure the prediction at all.")
add("target_direction_baseline_accuracy", TB["accuracy"], 0.01,
    "No model: for each pair, predict the target gene's majority observed sign among the silenced genes of the "
    "other 4 folds (5 folds of silenced genes, numpy default_rng(42) permutation of the sorted gene list; tie or "
    "unseen target -> training-fold majority, 'decrease'). Balanced accuracy "
    f"{TB['balanced_acc']}, MCC {TB['mcc']}. Leave-one-silenced-gene-out versions give 0.588-0.592 (balanced "
    "0.576-0.580).")
add("circuit_minus_target_baseline", BG["pooled_acc_minus_target_direction_baseline"]["point"], 0.01,
    f"Circuit accuracy minus target-direction baseline accuracy. {GENE_CI} with the baseline fixed: "
    f"{BG['pooled_acc_minus_target_direction_baseline']['ci95']}; by component "
    f"{BC['pooled_acc_minus_target_direction_baseline']['ci95']}. In all 5 folds the circuit (0.494-0.506) is below "
    "both always-'decrease' (0.537-0.561) and the baseline (0.575-0.592).")
au = M["auroc"]["frac_inhibitory_evidence"]
add("auroc_inhibitory_support_fraction", au["pooled_auroc"], 0.001,
    "AUROC of n_inhibitory_evidence / evidence for observed decrease, pooled over all pairs (average ranks for "
    f"ties). Silenced-gene bootstrap (500) {au['ci95_gene_bootstrap_500']}; mean per-gene AUROC {au['mean_per_gene_auroc']}.")
add("per_gene_mean_mcc", BG["per_gene_mean_mcc"]["point"], 0.001,
    "Mean over silenced genes of the per-gene MCC (213 genes; 14 genes get one sign for every target, so MCC is "
    f"undefined and they are skipped). Gene-bootstrap CI {BG['per_gene_mean_mcc']['ci95']}.")
add("per_gene_mean_accuracy_minus_always_decrease", BG["per_gene_mean_acc_minus_always_dec"]["point"], 0.001,
    f"Mean over the 227 silenced genes of (gene accuracy - gene share of decreases). CI {BG['per_gene_mean_acc_minus_always_dec']['ci95']}. "
    f"Mean per-gene accuracy {BG['per_gene_mean_accuracy']['point']}.")
add("n_genes_one_sign_for_all_targets", M["per_gene"]["n_genes_one_sign_for_all_targets"], 0,
    "Silenced genes for which the method predicts the same direction for every target.")
add("pair_level_z_accuracy_vs_half", M["pair_level_naive"]["accuracy_vs_half_z"], 0.05,
    "Shown for contrast only: (accuracy - 0.5) / sqrt(0.25 / 698,624), treating pairs as independent "
    f"(p = {M['pair_level_naive']['accuracy_vs_half_p_two_sided']:.1e}). The gene-resampled accuracy CI includes 0.5.")
for key, label in [("ties_as_decrease", "Ties (support fraction exactly 0.5) predicted 'decrease'"),
                   ("sign_of_summed_d", "Prediction = sum of signed Cohen's d over the supporting triples < 0 "
                                        "(rebuilt from circuit_edges.csv and feature_top_genes.tsv), the source paper's rule"),
                   ("unanimous_support_only", "Only pairs whose support is all inhibitory or all excitatory (79,269 pairs)"),
                   ("max_abs_d_gt_2", "Only pairs with max_abs_d > 2 (29,373 pairs, 158 silenced genes)")]:
    x = VV[key]
    add(f"variant_{key}_accuracy", x["accuracy"], 0.001,
        f"{label}. Predicted-decrease share {x['frac_pred_dec']}, observed {x['frac_obs_dec']}; accuracy gene-bootstrap CI "
        f"{x['ci95_accuracy']}; balanced accuracy {x['balanced_acc']} (minus 0.5: CI {x['ci95_balanced_acc_minus_half']}); "
        f"MCC {x['mcc']} (CI {x['ci95_mcc']}).")
b25 = [b for b in M["bands_abs_lfc"] if b["abs_lfc_lo"] == 0.25][0]
add("band_abs_lfc_ge_0p25_balanced_accuracy", b25["balanced_acc"], 0.001,
    f"Pairs with |lfc| >= 0.25 ({int(b25['n'])} pairs, {b25['n_silenced_genes']} silenced genes, 1.4%); the largest-change band "
    f"of the source report. Balanced accuracy minus 0.5 CI {b25['ci95_balanced_acc_minus_half']}; MCC {b25['mcc']} "
    f"(CI {b25['ci95_mcc']}); accuracy {b25['accuracy']} vs always-'decrease' {b25['frac_obs_dec']}.")
s5 = SUB["abs_lfc_ge_0.5"]
add("subset_abs_lfc_ge_0p5_mcc", s5["circuit"]["mcc"], 0.003,
    f"Post-hoc subset |lfc| >= 0.5: {s5['n']} pairs, {s5['n_silenced_genes']} silenced genes. Balanced accuracy "
    f"{s5['circuit']['balanced_acc']} (minus 0.5: gene-bootstrap CI {VV['method_rule_abs_lfc_ge_0.5']['ci95_balanced_acc_minus_half']}); "
    f"MCC gene-bootstrap CI {VV['method_rule_abs_lfc_ge_0.5']['ci95_mcc']}; accuracy {s5['circuit']['accuracy']} vs "
    f"always-'decrease' {s5['circuit']['best_constant_acc']}. Null that shuffles predictions within each silenced gene: "
    f"mean MCC {s5['within_gene_shuffle_null_mcc_mean']}, one-sided p = {s5['within_gene_shuffle_p_one_sided']}; "
    f"CMH stratified by silenced gene p = {s5['cmh_p_two_sided']}. Target-direction baseline on the same pairs: "
    f"MCC {s5['target_baseline']['mcc']}, accuracy {s5['target_baseline']['accuracy']}. CMH odds ratio by silenced gene "
    f"{W['cmh_subsets_odds_ratio_p']['abs_lfc_ge_0.5']['by_silenced_gene'][0]}; stratified also by the target's usual "
    f"direction {W['cmh_subsets_odds_ratio_p']['abs_lfc_ge_0.5']['by_silenced_gene_x_target_usual_direction'][0]} "
    f"(p = {W['cmh_subsets_odds_ratio_p']['abs_lfc_ge_0.5']['by_silenced_gene_x_target_usual_direction'][1]}). The odds "
    "ratio does not move toward 1, so the target's usual direction does not explain the excess; the within-knockdown "
    "tests are weak because only 14 knockdowns (273 pairs) have both outcomes and both predictions here, and 11 "
    "strata (137 pairs) after the extra split. At |lfc| >= 0.45 (755 pairs) the CMH by silenced gene gives odds ratio "
    f"{W['cmh_subsets_odds_ratio_p']['abs_lfc_ge_0.45']['by_silenced_gene'][0]}, p = "
    f"{W['cmh_subsets_odds_ratio_p']['abs_lfc_ge_0.45']['by_silenced_gene'][1]} (one threshold, not corrected).")
add("threshold_family_max_z_p", FAM["max_z_p_one_sided"], 0.04,
    "One test over the |lfc| thresholds 0.25, 0.4, 0.45, 0.5, 0.6, 0.75: predictions shuffled within silenced gene x "
    f"|lfc| bin (2,000 draws), statistic = max over thresholds of the MCC z-score. Max z = {FAM['max_z_observed']} "
    f"(at 0.45; single-threshold p there = {FAM['per_threshold_p_one_sided'][2]}).")
cm = W["cmh_all_pairs"]
add("cmh_by_silenced_gene_odds_ratio", cm["by_silenced_gene"]["odds_ratio"], 0.002,
    "Mantel-Haenszel odds ratio of prediction x observation over all pairs, stratified by silenced gene "
    f"(pairs treated as independent inside each stratum), p = {cm['by_silenced_gene']['p_two_sided']}. Stratified also by "
    "the target gene's usual direction (majority sign over the other 226 silenced genes): odds ratio "
    f"{cm['by_silenced_gene_x_target_usual_direction']['odds_ratio']}, p = {cm['by_silenced_gene_x_target_usual_direction']['p_two_sided']}.")
sp = W["per_gene_spearman_frac_inhibitory_vs_minus_lfc"]
sr = W["per_gene_spearman_frac_inhibitory_vs_minus_lfc_after_removing_target_usual_change"]
add("per_gene_spearman_support_vs_minus_lfc", sp["mean"], 0.002,
    f"Mean over {sp['n_genes']} silenced genes (> 10 pairs) of Spearman(support fraction, -lfc). CI by silenced gene "
    f"{sp['ci95_by_silenced_gene']}, by source-feature group {sp['ci95_by_source_feature_group']}, by component "
    f"{sp['ci95_by_component']}. After subtracting each target's mean lfc over the other 226 silenced genes: "
    f"{sr['mean']} (CI by gene {sr['ci95_by_silenced_gene']}). Target-level Spearman of mean support fraction vs mean "
    f"-lfc: {W['target_level_spearman_mean_frac_inhibitory_vs_mean_minus_lfc']['rho']}.")
ot = V["on_target_lfc"]
add("on_target_lfc_median", ot["median"], 0.005,
    f"Median over the 227 silenced genes of their own lfc (row g, column g of knockdown_lfc.npz). All {ot['n_negative']} "
    f"are negative; {ot['n_below_minus_0p5']} are below -0.5. The stronger half of knockdowns gives balanced accuracy "
    f"{VV['strongest_half_on_target_knockdown']['balanced_acc']}.")
rg = V["range_over_variants"]
add("variants_balanced_accuracy_max", rg["balanced_acc"][1], 0.003,
    f"Over 21 analyst variants (tie rule, sign of summed d, |lfc| >= 0.01/0.05/0.1/0.25/0.5, support filters, one source "
    f"site at a time, stronger knockdowns): balanced accuracy {rg['balanced_acc']}, MCC {rg['mcc']}, accuracy minus "
    f"always-'decrease' {rg['acc_minus_best_constant']} (never above 0). The maximum is the |lfc| >= 0.5 subset; it is the "
    "only variant whose balanced-accuracy or MCC CI excludes chance.")
add("frac_edges_inhibitory", ff["frac_edges_inhibitory"], 0.0005, "Share of the 272,905 edges with sign = inhibitory.")

key = {
    "task": "T2",
    "question": ("Do the SAE feature circuits of MaxToki-217M predict the direction in which a gene's expression "
                 "changes when another gene is silenced by CRISPRi in K562 cells?"),
    "key_verdict": "not supported",
    "key_conclusion": (
        "The circuit's direction predictions carry no detectable information. Over 698,624 (silenced gene, target gene) "
        "pairs from 227 knockdowns, accuracy is 49.8% (95% CI 49.2-50.4%, resampling silenced genes), 4.6 points below "
        "always predicting 'decrease' (54.3%; difference CI -5.8 to -3.3 points). Balanced accuracy is 49.7% "
        "(CI for minus 50%: -0.8 to +0.2 points) and MCC is -0.006 (CI -0.017 to +0.004); the AUROC of the inhibitory "
        "support fraction is 0.498. A rule with no model, the target gene's usual direction in other knockdowns, gets "
        "58.7%, 8.9 points above the circuit. The tie rule, the sign of summed d, support filters and single source "
        "sites do not change this (21 variants: accuracy never above always-'decrease'). The only nominal positive is a "
        "post-hoc subset of strongly changed pairs (|LFC| >= 0.5: 416 pairs, 59 genes, MCC 0.17): its excess is mostly "
        "between knockdowns, is not significant within knockdowns (CMH p = 0.22), does not survive a test over the "
        "thresholds tried (p = 0.085), and is far below the model-free rule on the same pairs (MCC 0.59). Over all "
        "pairs, the small within-knockdown association (CMH odds ratio 1.011) disappears once the target gene's usual "
        "direction is accounted for (0.998)."),
    "acceptable_alternatives": [
        {
            "verdict": "inconclusive",
            "counts_as_correct_only_if": (
                "the report (1) shows that on all pairs the circuit has no skill by a measure that is not fooled by the "
                "54/46 class split (balanced accuracy, MCC, kappa, AUROC, or accuracy compared with always-'decrease' or "
                "with chance from both class rates), with uncertainty that resamples silenced genes or groups of them, "
                "AND (2) does not present accuracy against 50%, the precision of 'decrease' predictions, the share of "
                "pairs that decrease, or pair-level p-values as evidence of skill, AND (3) bases the hedge on a specific, "
                "computed limit: a small excess in strongly changed pairs that the report itself shows is fragile "
                "(post-hoc threshold, few genes, or not significant within knockdowns), or the noisy ground truth "
                "(most |LFC| below 0.05, no significance filter) while acknowledging that the circuit is also at chance "
                "on the strongly changed pairs or below a model-free direction rule."),
            "reason": (
                "The test is well powered for the full set (CIs within about 1 point of chance), so 'not supported' is "
                "the key. But strongly changed pairs are few (1.4% have |LFC| >= 0.25) and a small, non-robust "
                "within-knockdown excess appears at some thresholds (best single threshold p = 0.01-0.03 before correction). "
                "An analyst who finds no skill overall but declines to rule out a small effect on large changes has "
                "reached the same scientific picture. 'supported' is never correct for this task.")
        }
    ],
    "key_numbers": nums,
    "traps": [
        {
            "id": "class_imbalance_vs_50",
            "description": (
                "54.3% of pairs go down after the knockdown, so 50% is not the chance level for accuracy. Always "
                "saying 'decrease' scores 54.3%; predictions independent of the outcome with the circuit's 50.9% "
                "'decrease' rate score 50.1%. The source paper itself compares its 56.4% with '50% chance'."),
            "correct_handling": (
                "Compares accuracy with always-'decrease' / the majority class (49.8% vs 54.3%) or with chance from both "
                "class rates (50.1%), or uses balanced accuracy, MCC, kappa or AUROC (49.7%, -0.006, -0.006, 0.498); "
                "reports the observed class split."),
            "mishandling": (
                "Judges accuracy only against 50%: e.g. calls a subset 'above chance' because its accuracy exceeds 50% "
                "(all-inhibitory-or-all-excitatory support 51.8% [51.0, 52.6]; max |d| > 2: 55.7%; ties as 'decrease' "
                "50.5%), or calls the circuit 'significantly worse than chance / anti-predictive' because 49.8% is below "
                "50%.")
        },
        {
            "id": "constant_sign_predictor",
            "description": (
                "Accuracy rewards a rule that says 'decrease' more often than the data do, with no skill at all. In the "
                "original deployment the circuit said 'decrease' for 90% of pairs and scored 53.5%, above 50% only "
                "because of this. In this package the overall 'decrease' share is 50.9%, but it is skewed in subsets "
                "(max |d| > 2: 85%; source site 9 only: 79%; unanimous support: 66%; ties counted as 'decrease': 61%) and "
                "14 silenced genes get one sign for every target (MCC undefined)."),
            "correct_handling": (
                "Judges every set it scores with a measure that does not reward a constant rule (balanced accuracy, "
                "MCC, kappa, chance from both class rates), or reports raw accuracy only next to the predicted- and "
                "observed-'decrease' shares and the always-'decrease' score; treats one-sign genes as uninformative."),
            "mishandling": (
                "Reports a gain in raw accuracy for a subset or rule variant without its predicted share (for example "
                "'pairs with |d| > 2 reach 55.7%'), or counts per-gene accuracy of one-sign genes as skill.")
        },
        {
            "id": "resampling_unit",
            "description": (
                "The 698,624 pairs are not independent. They come from 227 knockdowns: all pairs of one knockdown share "
                "its LFC profile and its source features, and knockdowns share source features (119 groups of identical "
                "source-feature sets, 50 connected components, the largest with 88 genes). Pair-level tests are far too "
                "narrow: accuracy vs 0.5 z = -4.0 (p = 6e-5); precision of 'decrease' vs 0.5 z = 47.7; a CMH test "
                "stratified by knockdown p = 0.043. The source paper's p-value over pairs (p < 1e-92 for its magnitude "
                "correlation; the specification quotes p < 1e-32) has the same problem."),
            "correct_handling": (
                "Resamples silenced genes (accuracy CI 49.2-50.4%; MCC CI -0.017 to +0.004) or source-feature groups or "
                "components, and says what was resampled; does not rest a claim on pair-level p-values."),
            "mishandling": (
                "Uses a binomial test, chi-square, pair bootstrap or DeLong test on the pairs as if independent and draws "
                "a conclusion from it (a positive claim, or 'significantly below chance'); or reports no uncertainty.")
        },
        {
            "id": "metric_ignores_prediction",
            "description": (
                "The original deployed code counted every pair whose target went down as 'correct', whatever the "
                "prediction, and reported the result (54.61%) as directional accuracy. On this package the same "
                "calculation gives 54.3%. A related slip is to report the share of predicted-'decrease' pairs that go "
                "down (precision 54.0%) as the accuracy."),
            "correct_handling": (
                "Defines accuracy as agreement between predicted and observed sign over all pairs, both classes "
                "(49.8%), and would get a different number if the predictions were flipped or shuffled."),
            "mishandling": (
                "Reports about 54% 'directional accuracy' from the share of pairs that decrease or from the precision of "
                "'decrease' predictions, and treats it as skill.")
        },
        {
            "id": "target_direction_baseline",
            "description": (
                "Direction is partly predictable with no model: some target genes go down in most knockdowns. Predicting "
                "each target's usual sign in other knockdowns (cross-fitted by silenced gene) gives 58.7% (balanced "
                "57.5%, MCC 0.157); the circuit is 8.9 points lower [8.0, 10.0]. The circuit's within-knockdown "
                "association over all pairs (CMH odds ratio 1.011, pair-level p = 0.043; per-gene Spearman of support fraction vs -LFC "
                "0.0096, CI by gene 0.0003-0.019) is target-gene information: it is gone after accounting for the target's "
                "usual direction (odds ratio 0.998, p = 0.66; Spearman -0.002 [-0.012, +0.007])."),
            "correct_handling": (
                "Compares the circuit with a model-free target-direction rule (or controls for target-gene effects) and "
                "reports that the circuit falls below it; checks any small positive against target identity."),
            "mishandling": (
                "Reads a small within-knockdown association (for example the CMH p = 0.04 or a per-gene Spearman CI above "
                "0 when resampling genes) as knockdown-specific prediction without accounting for target-gene effects; "
                "or says the direction of change cannot be predicted from these data at all (pure noise), when a rule "
                "with no model gets 58.7%.")
        },
        {
            "id": "post_hoc_subsets",
            "description": (
                "Many reasonable choices move the number: tie rule, the source paper's sign-of-summed-d rule, |LFC| "
                "filters, support filters, one source site, stronger knockdowns. Over 21 variants balanced accuracy runs "
                "0.494-0.595 and MCC -0.011 to 0.166; accuracy never beats always-'decrease'. One subset has a CI above "
                "chance: |LFC| >= 0.5 (416 pairs, 0.06%, 59 genes; balanced 0.595, MCC 0.166 [0.067, 0.283]). Most of "
                "it is between knockdowns (a within-knockdown shuffle null already gives MCC 0.119; p = 0.11; CMH "
                "by knockdown odds ratio 1.56, p = 0.22, from only 14 informative knockdowns); a test over the "
                "thresholds tried gives p = 0.085; the target-direction rule on the same pairs reaches MCC 0.595. "
                "Adding the target's usual direction as a stratum does not shrink the odds ratio (1.80, p = 0.35), so "
                "this subset is not explained by target identity; it is simply too small to test within knockdowns. "
                "At |LFC| >= 0.45 the CMH by knockdown is nominally significant (odds ratio 1.63, p = 0.037) before "
                "correction for the thresholds tried."),
            "correct_handling": (
                "Fixes the main analysis before looking, or reports all variants it tried; treats a subset positive as "
                "exploratory and checks it within knockdowns, for the thresholds tried, and against the target-direction "
                "rule; does not build the verdict on it."),
            "mishandling": (
                "Reports the strongest subset (for example |LFC| >= 0.5 or >= 0.45, with a CI above chance) as evidence "
                "that the circuits predict direction for strong effects, without checking within knockdowns, for "
                "multiple thresholds, or against the target-direction rule.")
        }
    ],
    "false_statement_checks": [
        "The circuit predicts the direction of change better than chance. (False: accuracy 49.8% [49.2, 50.4], below "
        "always-'decrease' 54.3%; balanced accuracy 49.7%, MCC -0.006 [-0.017, +0.004], AUROC 0.498.)",
        "The circuit beats always predicting 'decrease'. (False: -4.6 points [-5.8, -3.3]; no variant of 21 does.)",
        "The circuit's directional accuracy is about 54%. (False: 54.3% is the share of pairs that decrease, which ignores "
        "the prediction; 54.0% is the precision of 'decrease' predictions, below the 54.3% base rate. Accuracy is 49.8%.)",
        "The circuit is significantly worse than chance (anti-predictive). (False: only against 50% with pairs treated as "
        "independent, z = -4.0. Against chance from both class rates the gap is -0.3 points [-0.8, +0.2]; MCC CI "
        "includes 0; the gene-resampled accuracy CI includes 50%.)",
        "With about 700,000 pairs even a small difference is highly significant. (False: pairs cluster in 227 "
        "knockdowns and 50 groups of knockdowns that share source features; resampling those gives accuracy CIs "
        "5 to 8 times wider than pair-level ones.)",
        "The direction of a gene's change after a knockdown cannot be predicted from these data. (False: the target "
        "gene's usual direction in other knockdowns gives 58.7%, MCC 0.157.)",
        "The circuits predict direction for strongly changed genes (for example |LFC| >= 0.5). (Not supported: post-hoc "
        "subset of 416 pairs and 59 genes; mostly between knockdowns, within-knockdown p = 0.11-0.22; threshold "
        "family p = 0.085; the target rule reaches MCC 0.595 on the same pairs. A report may correctly say that the "
        "subset's odds ratio stays near 1.5-1.8 after stratifying by the target's usual direction; that is not a "
        "false statement.)",
        "Using the sign of the summed Cohen's d (the source paper's rule) instead of the majority rule gives skill. "
        "(False: accuracy 50.2%, balanced 0.496, MCC -0.007 [-0.017, +0.003].)",
        "Pairs with stronger circuit support (more evidence, larger |d|, unanimous edges) show skill. (False: balanced "
        "accuracy 0.494-0.510, every CI includes 0.5; raw accuracy gains come from saying 'decrease' more often, 85% of "
        "the time for max |d| > 2.)",
        "The null result is because the knockdowns did not work. (False: all 227 silenced genes have a negative own "
        "LFC, median -0.93; the stronger half of knockdowns gives balanced accuracy 0.500.)",
        "The circuit's support fraction carries knockdown-specific information about direction. (Not supported: the "
        "small within-knockdown association, CMH odds ratio 1.011, p = 0.043 with pairs treated as independent, becomes "
        "0.998, p = 0.66, once the target gene's usual direction is accounted for.)"
    ],
    "reference": {
        "script": "reference.py (sections: main, variants, within)",
        "output": "reference_output.json",
        "notes": "NOTES.md",
        "source_agreement": (
            "128 of 128 quantities checked against the source evaluation behind V2_CIRCUIT_REPORT.md (pooled counts and "
            "metrics, target-direction baseline, per-gene summary, all three grouped bootstraps with their CIs, the "
            "|LFC| bands, the tie sensitivity, fold-by-fold accuracies) match exactly with the same seeds. The gene pairs "
            "rebuilt from circuit_edges.csv + feature_top_genes.tsv with the method's rule equal gene_pairs.parquet "
            "row for row (see build/T2/check_against_source.json).")
    },
    "verification": "Independently re-derived and amended on 2026-10-01; see VERIFICATION.md."
}
(K / "key.json").write_text(json.dumps(key, indent=1, ensure_ascii=False) + "\n")
print(len(nums), "key numbers;", len(key["traps"]), "traps;", len(key["false_statement_checks"]), "false-statement checks")
