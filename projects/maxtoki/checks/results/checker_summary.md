## spec_lint (8 deployed specs)

| spec | template | S1 | S2 | S3 pin | S4 refs | S5 params | S6 val block | S7 slots | refs (required form / exist as written / total) |
|---|---|---|---|---|---|---|---|---|---|
| attention-grn-extraction-and-evaluation.md | REJECT | FAIL | PASS | TEXT_MATCH_ONLY_NO_GIT | FAIL | PASS | FAIL | FAIL | 0 / 22 / 22 |
| residual-stream-spectral-geometry.md | REJECT | FAIL | PASS | TEXT_MATCH_ONLY_NO_GIT | FAIL | PASS | FAIL | FAIL | 0 / 13 / 28 |
| topology-geometry-141-hypotheses.md | REJECT | FAIL | PASS | TEXT_MATCH_ONLY_NO_GIT | FAIL | PASS | FAIL | FAIL | 0 / 18 / 22 |
| manifold-discovery-extraction-compactification.md | REJECT | FAIL | PASS | NONE_WITH_REASON | FAIL | PASS | FAIL | FAIL | 0 / 7 / 7 |
| longevity-mechinterp-donor-aware.md | ACCEPT | PASS | PASS | VERIFIED_GIT_HEAD | FAIL | PASS | FAIL | FAIL | 0 / 15 / 46 |
| 01-sae-atlas.md | REJECT | FAIL | PASS | TEXT_MATCH_ONLY_NO_GIT | FAIL | FAIL | FAIL | FAIL | 0 / 45 / 52 |
| 02-causal-circuit-tracing.md | REJECT | FAIL | PASS | TEXT_MATCH_ONLY_NO_GIT | FAIL | PASS | FAIL | FAIL | 5 / 27 / 27 |
| 03-exhaustive-mapping-and-steering.md | REJECT | FAIL | PASS | TEXT_MATCH_ONLY_NO_GIT | FAIL | PASS | FAIL | FAIL | 0 / 18 / 26 |

## number_trace

| scope | document | numbers | traced (share, Wilson 95% CI) | mean chance rate (bootstrap 95% CI) | excess over chance (bootstrap 95% CI) | weak traces | untraced |
|---|---|---|---|---|---|---|---|
| run_summary | attention-grn-217M-FINAL_SUMMARY.md | 102 | 100 (0.98, 0.93–0.99) | 0.95 (0.92–0.98) | 0.03 (0.01–0.05) | 98 | 2 |
| run_summary | FINAL_SUMMARY.md | 93 | 92 (0.99, 0.94–1.00) | 0.95 (0.92–0.97) | 0.04 (0.02–0.06) | 90 | 1 |
| run_summary | spectral-geometry-217M-FINAL_SUMMARY.md | 200 | 199 (0.99, 0.97–1.00) | 0.91 (0.87–0.94) | 0.09 (0.06–0.12) | 185 | 1 |
| run_summary | FINAL_SUMMARY.md | 116 | 116 (1.00, 0.97–1.00) | 0.96 (0.93–0.98) | 0.04 (0.02–0.06) | 114 | 0 |
| run_summary | topology-141-217M-FINAL_SUMMARY.md | 55 | 55 (1.00, 0.93–1.00) | 0.93 (0.87–0.98) | 0.07 (0.02–0.13) | 52 | 0 |
| run_summary | FINAL_SUMMARY.md | 95 | 93 (0.98, 0.93–0.99) | 0.88 (0.84–0.93) | 0.10 (0.06–0.13) | 89 | 2 |
| run_summary | manifold-discovery-217M-FINAL_SUMMARY.md | 215 | 211 (0.98, 0.95–0.99) | 0.74 (0.71–0.76) | 0.25 (0.22–0.27) | 194 | 4 |
| run_summary | FINAL_SUMMARY.md | 263 | 259 (0.98, 0.96–0.99) | 0.73 (0.70–0.75) | 0.26 (0.24–0.28) | 239 | 4 |
| run_summary | FINAL_SUMMARY.md | 35 | 22 (0.63, 0.46–0.77) | 0.23 (0.17–0.29) | 0.40 (0.21–0.58) | 3 | 13 |
| run_summary | sae-atlas-217M-FINAL_SUMMARY.md | 141 | 139 (0.99, 0.95–1.00) | 0.96 (0.93–0.98) | 0.03 (0.00–0.05) | 134 | 2 |
| run_summary | sae-atlas-217M-FINAL_SUMMARY_12layer.md | 71 | 70 (0.99, 0.92–1.00) | 0.97 (0.92–1.00) | 0.02 (-0.01–0.06) | 68 | 1 |
| run_summary | circuit-tracing-217M-FINAL_SUMMARY.md | 112 | 96 (0.86, 0.78–0.91) | 0.58 (0.50–0.66) | 0.28 (0.20–0.36) | 68 | 16 |
| run_summary | exhaustive-mapping-217M-FINAL_SUMMARY.md | 149 | 135 (0.91, 0.85–0.94) | 0.53 (0.47–0.59) | 0.38 (0.31–0.44) | 80 | 14 |
| paper_vs_union_of_8_runs (found in any run) | main.tex | 124 | 123 (0.99, 0.96–1.00) | 0.97 (0.95–1.00) | 0.02 (-0.00–0.04) | 121 | 1 |
| paper_chain (written in run R's FINAL_SUMMARY and found in run R's outputs) | main.tex | 124 | 119 (0.96, 0.91–0.98) | 0.50 (0.43–0.56) | 0.46 (0.39–0.53) | 0 | 5 |
| paper_vs_numbers_written_in_FINAL_SUMMARY_files (transcription check) | main.tex | 124 | 120 (0.97, 0.92–0.99) | 0.50 (0.44–0.57) | 0.47 (0.40–0.53) | 0 | 4 |
| paper_vs_extended_corpus (runs + audits + summaries + setup json/csv/log/txt) | main.tex | 124 | 123 (0.99, 0.96–1.00) | 0.98 (0.95–1.00) | 0.02 (-0.00–0.04) | 0 | 1 |
| paper_vs_union_excluding_spec_numbers | main.tex | 48 | 47 (0.98, 0.89–1.00) | 0.94 (0.86–1.00) | 0.04 (-0.00–0.10) | 45 | 1 |
| paper_class:decimal_1dp | main.tex | 13 | 13 (1.00, 0.77–1.00) | 1.00 (1.00–1.00) | 0.00 (0.00–0.00) | 13 | 0 |
| paper_class:decimal_2+dp | main.tex | 63 | 63 (1.00, 0.94–1.00) | 0.99 (0.96–1.00) | 0.01 (0.00–0.04) | 62 | 0 |
| paper_class:integer>=10 | main.tex | 48 | 47 (0.98, 0.89–1.00) | 0.95 (0.89–1.00) | 0.03 (-0.00–0.08) | 46 | 1 |
| run_summaries_class:decimal_1dp | (all FINAL_SUMMARY files) | 203 | 193 (0.95, 0.91–0.97) | 0.89 (0.86–0.92) | 0.06 (0.03–0.09) | 182 | 10 |
| run_summaries_class:decimal_2+dp | (all FINAL_SUMMARY files) | 900 | 881 (0.98, 0.97–0.99) | 0.76 (0.74–0.78) | 0.22 (0.20–0.24) | 783 | 19 |
| run_summaries_class:integer>=10 | (all FINAL_SUMMARY files) | 525 | 497 (0.95, 0.92–0.96) | 0.85 (0.82–0.87) | 0.10 (0.08–0.12) | 442 | 28 |
| run_summaries_class:sci | (all FINAL_SUMMARY files) | 19 | 16 (0.84, 0.62–0.94) | 0.45 (0.29–0.62) | 0.39 (0.23–0.55) | 7 | 3 |

## run_manifest_check: flagged parameter deviations

| run | spec parameter | value(s) used | spec default | spec range | status | reason written down? |
|---|---|---|---|---|---|---|
| attention-grn-217M | n_ctrl | 200;2000 | 2000 | {500, 2000, 10000} | OUT_OF_RANGE | no |
| attention-grn-217M | hvg | 1500 | {2000, 3309} | {1000, 2000, 5000} | OUT_OF_RANGE | no |
| attention-grn-217M | intervention_cells_fidelity | 2000;300 | 2000 | not machine-readable | DIFFERS_NO_RANGE_GIVEN | no |
| attention-grn-217M | degree_null_n_curveball | 50 | 200 | not machine-readable | DIFFERS_NO_RANGE_GIVEN | projects/maxtoki/runs/attention-grn-217M/README.md:32 |
| spectral-geometry-217M | autoloop_iterations | 8 | 63 | 40–80 | OUT_OF_RANGE | no |
| topology-141-217M | null_label_permutation_replicates | 100;50;8 | 100–200 | 100–1000 | OUT_OF_RANGE | no |
| topology-141-217M | null_rewiring_replicates | 12;8 | 24 | ≥ 24 | OUT_OF_RANGE | no |
| sae-atlas-217M | n_cells_training | 500 | 2000 | not machine-readable | DIFFERS_NO_RANGE_GIVEN | no |
| sae-atlas-217M | n_cells_training | 500 | 3000 | not machine-readable | DIFFERS_NO_RANGE_GIVEN | no |
| sae-atlas-217M | causal_patching_cells | 50 | 200 | not machine-readable | DIFFERS_NO_RANGE_GIVEN | no |
| exhaustive-mapping-217M | n_cells_per_condition | 50 | 200 | 100–500 | OUT_OF_RANGE | no |

## run_manifest_check: gate objects not simply consistent

| run | file | object | status | declared gates with no value | within 0.005 of threshold |
|---|---|---|---|---|---|
| attention-grn-217M | phase12_verdict.json | criteria.C3_residualized_retains_signal | CONSISTENT; IMPLAUSIBLE_FRACTION |  |  |
| attention-grn-217M | phase12_verdict.json | criteria.C4_top_heads_causal | AMBIGUOUS_NOT_RECOMPUTED |  |  |
| manifold-discovery-217M | external_validation_lung_control.json | (root) | CONSISTENT; NEGATIVE_CONTROL_PASSED |  | trustworthiness=0.8013 vs 0.8 |
| manifold-discovery-217M | h38_lite_quality_gates.json | positive | PASS_WITH_UNCOMPUTED_GATES | blocked_permutation_p_max;n_permutations_min |  |
| manifold-discovery-217M | quality_gates_let_anchor.json | positive | PASS_WITH_UNCOMPUTED_GATES | blocked_permutation_p_max;n_permutations_min |  |
| manifold-discovery-217M | zeroshot_h103.json | (root) | PROTOCOL_K_DIFFERS(['trustworthiness_k5'] vs k=15) |  |  |
| manifold-discovery-217M | zeroshot_transfer_anchor_head.json | (root) | CONSISTENT; GATE_SET_FOR_PANEL_UNSTATED | blocked_permutation_p_max;n_permutations_min |  |
