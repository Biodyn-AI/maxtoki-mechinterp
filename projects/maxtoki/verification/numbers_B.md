# numbers_B — number check for paper-plos-one/main.tex, lines 1043–1348

Paper: `<REPO_ROOT>/projects/maxtoki/paper-plos-one/main.tex`
Figure data (TikZ): `<REPO_ROOT>/projects/maxtoki/paper-biosystems/main.tex` lines 1013–1062 (Fig 2), 1178–1221 (Fig 3), 1262–1307 (Fig 4).

Path short-hands used in the table:
- `S/` = `<REPO_ROOT>/projects/maxtoki/summaries/`
- `R/` = `<REPO_ROOT>/projects/maxtoki/runs/`
- `A/` = `<REPO_ROOT>/projects/maxtoki/audits/`
- `P/` = `<REPO_ROOT>/pipelines/`

Status key: MATCH / MISMATCH / NOT FOUND / AMBIGUOUS / DIFFERENT ENDPOINT (the number exists but the source measured something else than the paper says).

"Verified" means I read the file. "Computed" means I ran a short script on a run file (read-only). "Inferred" means my reading, not a file statement.

---

## 1. The most important problems (short list)

1. **Fig 2 and lines 1135–1138 describe the wrong ground truth.** The bars are per-perturbation AUROC against genes called differentially expressed (DE) in the same CRISPR screen. They are not "curated regulator–target relationships". The curated (TRRUST) AUROCs are different numbers (0.483 / 0.605 / 0.520 / 0.555). Verified in `R/attention-grn-217M/scripts/phase1_trivial_baselines.py:1-14, 215-256`.
2. **"Ten-times-larger" 1B model is wrong.** 1B / 217M = about 4.6x. The source summary itself says "5x" (`S/attention-grn-217M-FINAL_SUMMARY.md:16`). The 1B run also used 200 control cells, not 2,000.
3. **"Residualizing collapses attention to chance" is false for the 1B run.** The 1B residual AUROC is 0.543 and the run's own verdict file marks criterion C3 as PASS (78% of above-chance signal kept). `R/attention-grn-217M/outputs/phase12_verdict_k562_1b.json`. The final summary table wrongly lists C3 = FAIL for 1B.
4. **The CRISPRi direction result has no real signal above base rates** (computed). The circuit calls 89.6% of pairs "inhibitory", and 54.6% of real log-fold-changes are negative. A predictor that always says "inhibitory" scores 54.61% (the same number as the old "buggy" metric). Shuffling predictions within each silenced gene gives 53.59%. The paper's 53.42% per-gene mean is 0.13 points *below* its own base-rate expectation (53.55%; 95% CI of the difference −0.28 to +0.02 points). So "the interval excludes 50%, so the effect is real" uses the wrong null.
5. **The "five-fold cross-validation" for CRISPRi is not a cross-validation.** Nothing is trained. The folds only split genes into 5 groups and report accuracy per group. The 53.42% is the per-gene mean with a bootstrap CI, not a CV result. The 5-fold mean in the JSON is 53.47% (the summary and audit say 53.55%, which is also wrong).
6. **"16 internal features over 60 genes, 7.7 kB in total" is wrong on two counts.** The "60 genes" are 60 coordinates of the attention-head space and 60 coordinates of the residual stream, not genes. The 7.7 kB counts only the sparse operator; the trained read-out head (8,973-byte file) is extra, and the whole 217M model must still run.
7. **"Factor of 2,400 from a 154-dimensional dense form" is wrong.** 2,352x is from the 2,464-dim full-drift operator (18.2 MB). From the 154-dim dense single-head operator (0.76 MB) it is about 98x.
8. **Trustworthiness does not measure what the paper says.** The code computes `sklearn.manifold.trustworthiness(features, z, k=15)`: it checks whether the 10-D read-out keeps the neighbours of the model's own feature space. The biological ruler is not used. It is also computed on anchors (centroids of up to 50 cells from one donor × tissue × cell type), not on cells.
9. **External and zero-shot "branch-holdout" is not a holdout.** The head is frozen after training on all internal branches. The external/zero-shot numbers (0.346, 0.317) are within-branch correlations, with no branch withheld from fitting.
10. **The lung negative control fails all four gates, not three.** Trust is 0.7996, below 0.80. So the claims that trustworthiness "clears the gate even on the lung control" and "passes a manifold that is clearly wrong" are wrong. The first lung control (lung immune cells, 51 anchors) passed all four gates and was rebuilt; the paper does not say this.
11. **Table 4 has three "n.t." cells that were tested.** H38 external = 0.883, H38 zero-shot = 0.787 (a FAIL), H95 zero-shot = 0.819. The paper copied a stale summary. H103 zero-shot 0.857 used k=5 on 18 anchors from 9 donors, not the 12-donor panel.
12. **"Pre-registered catalogue of 141 candidate orderings" does not exist.** The 141 number belongs to the topology pipeline. H92–H112 were created during the run. H95 was first judged INCONCLUSIVE; the "three-gate fallback" was added after sweep 1.
13. **Phase-11 numbers use a different endpoint.** "Four strongest axes of variation" are really the top-4 factors by ablation impact (factors 1, 4, 11, 9), not the four largest singular values. "Branch-holdout 0.973 → 0.123" is branch *balanced accuracy* of a frozen linear probe.
14. **The triplet denominator is wrong.** 4 triplets were tested. 2,980 is the number of target features per triplet, not the number of triplets.
15. **The SAE 1.01x test is not about TF targets.** It measures logit change on each feature's own top-20 genes vs other genes after zeroing the feature.
16. **Fig 4 TRRUST bars do not match the source**: source 0.003, 0.0, 0.0, 0.0; figure 0.003, 0.001, 0.001, 0.001. Also these TRRUST values are not p-values for any observed result (0 of 5 features passed).

---

## 2. Full table

| Paper line | Claim text (short) | Value in paper | Source file:line | Value in source | Status |
|---|---|---|---|---|---|
| 1046–1049 | Model encodes identity geometry "recoverable, externally generalizable, locally compressible" | framing | `S/manifold-discovery-217M-FINAL_SUMMARY.md:15-18` | same framing | MATCH (framing; see rows below for the weak parts) |
| 1061–1063 | "Five methodologically distinct probes agree" | 5 probes | rows below | 5 probes exist | MATCH |
| 1077–1079 | Attention vs degree-preserving (curveball) null: z ≈ 0 in all four runs | z ≈ 0 | `S/attention-grn-217M-FINAL_SUMMARY.md:107`; `S/attention-grn-217M-k562-report.md:75`; `S/attention-grn-217M-rpe1-report.md` Phase 3b; `S/attention-grn-217M-adamson-report.md:74`; `S/attention-grn-1B-k562-report.md:74` | −0.14 / +0.17 / +0.16 / −0.04 | MATCH. Note: "sits at chance" is loose. In RPE1 the observed TRRUST AUROC is 0.6045; z≈0 means it equals the degree-preserving null, not 0.5. Curveball used 50 iterations; the spec asks for 200 (`P/attention-grn-extraction-and-evaluation.md:79, 213`). |
| 1079–1082 | Four runs: K562, RPE1, Adamson, 1B | 4 runs | `S/attention-grn-217M-FINAL_SUMMARY.md:7-12` | 4 runs | MATCH |
| 1082 | "ten-times-larger MaxToki-1B" | 10x | `S/attention-grn-217M-FINAL_SUMMARY.md:16` ("5x model-size increment"); `main.tex:182-183` macros = 217M and 1B | 1B/217M ≈ 4.6x | MISMATCH |
| 1082–1084 | Gene-variance "classifier" beats attention by 6–20 AUROC points in every run | 6–20 pts | `S/attention-grn-217M-k562-report.md:32`; rpe1 report:31; adamson report:31; 1B report:31 | +0.0837, +0.1632, +0.2042, +0.0562 | MATCH on values. DIFFERENT ENDPOINT on meaning: AUROC is against DE-called targets of each perturbation, and "classifier" is just the target gene's variance used as a score (`phase1_trivial_baselines.py:233-247`), no fitting. |
| 1085 | p ≤ 1e-4 after multiple-testing correction | ≤1e-4 | `R/attention-grn-217M/outputs/phase12_verdict*.json` | 4.03e-10, 1.51e-51, 1.38e-10, 9.47e-05 | MATCH (the summary misquotes RPE1 as 2e-52 and Adamson as 5e-11: `S/attention-grn-217M-FINAL_SUMMARY.md:32-33`) |
| 1086–1088 | After removing gene-level info, attention ranks pairs "no better than chance" | chance, all runs | verdict JSONs, C3 block | K562 0.4866; RPE1 0.4922; Adamson 0.5202; **1B 0.5434, C3 PASS (78.4% of above-chance signal kept)** | MISMATCH for 1B. (`S/attention-grn-217M-FINAL_SUMMARY.md:12` wrongly lists 1B C3 = FAIL.) |
| 1088–1090 | What attention adds over variance is "indistinguishable from noise" | — | 1B report Phase 3c: `S/attention-grn-1B-k562-report.md:79-81` | Phase 2 gain ≤ +0.002 (matches); but 1B propensity-matched gain Δattn = +0.0457 | AMBIGUOUS / overstated (contrary 1B evidence not mentioned) |
| 1095–1096 | Direction correct 53.48% over 1,503,408 pairs | 53.48%, 1,503,408 | `R/circuit-tracing-217M/outputs/groupkfold_crispri/summary.json` | 0.534761, n_pairs 1,503,408, 248 silenced genes | MATCH on value. See base-rate row below. |
| 1096–1099 | "Averaging within each silenced gene first, and holding out whole silenced genes in five-fold CV so no gene is in training and test, gives 53.42% (95% CI [52.60, 54.16])" | 53.42% [52.60, 54.16] | same JSON; script `R/circuit-tracing-217M/scripts/audit_a2_groupkfold_crispri.py:184-212` | per-gene mean 0.534187, bootstrap CI [0.52605, 0.54165] over 248 genes; 5-fold mean 0.534730 | DIFFERENT ENDPOINT. 53.42% is the per-gene mean with a gene bootstrap. The 5-fold number is 53.47%. No model is trained, so "training and test" does not apply. (Summary/audit say 5-fold = 53.55% and "966 source genes": both wrong vs JSON: `S/circuit-tracing-217M-FINAL_SUMMARY.md:103`, `A/audit-20260507-completion-v2.md:73, 78`.) |
| 1099–1101 | "The interval excludes 50%, so the effect is real" | real effect | computed from `R/circuit-tracing-217M/outputs/groupkfold_crispri/per_pair.parquet` | 89.6% of predictions "inhibitory"; 54.6% of actual LFC < 0; always-"inhibitory" = 54.61%; within-gene shuffle = 53.59% ± 0.02%; per-gene base-rate expectation 53.55%; observed − expected = −0.13 pts (95% CI −0.28 to +0.02) | MISMATCH (wrong null; no signal above base rates). Computed by me. |
| 1101 | "about three percentage points above a coin flip" | ~3 pts | same | 3.4–3.5 pts | MATCH (but see row above) |
| 1106–1108 | Silencing a feature: median specificity 1.01x, "no preference for the factor's known targets" | 1.01x | `S/sae-atlas-217M-FINAL_SUMMARY.md:241-245`; `A/bootstrap_cis_summary.json:2-6`; script `R/sae-atlas-217M/scripts/phase6_patching.py:1-5, 105-172` | 1.0149 (CI [1.011, 1.019]); ratio = logit change on the feature's own top-20 genes vs other genes | DIFFERENT ENDPOINT (not TF targets; it is the feature's own top genes) |
| 1109 | 2.36x for Geneformer in source study | 2.36x | `S/sae-atlas-217M-FINAL_SUMMARY.md:230, 242` | 2.36x | MATCH |
| 1109–1110 | 0 of 48 factors specific under TRRUST | 0/48 | `R/sae-atlas-217M/outputs/phase8t_chance_baseline/summary.json`; `S/sae-atlas-217M-FINAL_SUMMARY.md:110` | observed 0 of 48; only 3 of 48 TFs had any responding feature (CDC5L, GATA1, MED1) | MATCH on value; key context omitted |
| 1110–1113 | "This particular headline did not survive the audit" | — | `A/audit_a2_groupkfold_status.md` §2; `S/sae-atlas-217M-FINAL_SUMMARY.md:242` | 1.01x survived (tight CI); only 0/48 was re-qualified | AMBIGUOUS wording |
| 1116 | "Of the 141 screened topological hypotheses" | 141 screened | `R/topology-141-217M/FINAL_SUMMARY.md:34-38`; `S/topology-141-217M-FINAL_SUMMARY.md:89` | Only the "headline backbone" plus some extensions were run; several phases skipped | MISMATCH / overstated |
| 1117 | H123 is "the strongest candidate" | strongest | `R/topology-141-217M/FINAL_SUMMARY.md:17, 64` | "paper's single strongest" (source paper, not MaxToki) | AMBIGUOUS |
| 1119–1120 | H123 clears its null in at most 1 of 12 layers in any domain | ≤1/12 | `R/topology-141-217M/outputs/phase78/h123_signed_motif_degree_strata.csv` | immune 1/12 (layer 9, margin +0.002); lung 0/12; ext-lung 0/12 | MATCH |
| 1120–1122 | Strictest CV (regulators and targets held out): immune "accuracy" 0.505 → 0.462 | 0.505 → 0.462 | `R/topology-141-217M/outputs/groupkfold_h123/summary.json` | mean AUROC 0.5051 → 0.4616 (12 layers, 76 test pairs per layer) | MATCH on values. It is AUROC, not accuracy. With 76 test pairs, 0.462 is not clearly below chance. |
| 1126–1127 | Zero superadditive cases "out of 2,980 tested" | 0 / 2,980 | `R/exhaustive-mapping-217M/outputs/experiment2_v2/combinatorial_summary.json` (and `experiment2/`) | 4 triplets; each measured on ~2,977–2,980 target features; 0 superadditive | DIFFERENT ENDPOINT (denominator is target features per triplet, not triplets) |
| 1127–1128 | Three-way redundancy ratio 0.190 vs 0.59 in source | 0.190 / 0.59 | same JSON; `S/exhaustive-mapping-217M-FINAL_SUMMARY.md:47` | 0.1896 (v2), 0.1897 (v1); source 0.59 | MATCH. Note: all 4 unrelated triplets give the same ratio to 4 decimals (std 0.0001). This suggests the metric does not depend on which features are chosen, which weakens the "redundancy" reading (inferred). |
| 1135–1138 | Variance baseline "ranks true regulator–target pairs better than attention" | — | `phase1_trivial_baselines.py:1-14` | ground truth = DE targets of each perturbation in the same screen | DIFFERENT ENDPOINT |
| Fig 2 caption 1145–1147 | Blue bars = attention AUROC for "curated regulator–target relationships" | 0.512 / 0.603 / 0.508 / 0.540 | TikZ `paper-biosystems/main.tex:1035-1036`; `phase12_verdict*.json` C1 | 0.5119 / 0.6033 / 0.5081 / 0.5398 = per-perturbation AUROC vs DE targets. TRRUST (curated) AUROCs are 0.4829 / 0.6045 / 0.5195 / 0.5554 | DIFFERENT ENDPOINT (values match the DE-based metric, not curated pairs) |
| Fig 2 bars | Orange = gene-variance baseline | 0.596 / 0.766 / 0.712 / 0.596 | verdict JSONs | 0.5957 / 0.7664 / 0.7123 / 0.5961 | MATCH |
| Fig 2 caption 1149 | Baseline wins by 5.6 to 20.4 points | 5.6–20.4 | per-run reports | 5.62 (1B) to 20.42 (Adamson) | MATCH |
| Fig 2 caption 1150 | BH-corrected Wilcoxon p ≤ 1e-4 | ≤1e-4 | verdict JSONs | max 9.47e-05 | MATCH |
| Fig 2 caption 1150–1152 | "Scaling the model tenfold narrows the gap" | 10x | `S/attention-grn-1B-k562-report.md:8`; `S/attention-grn-217M-FINAL_SUMMARY.md:12, 63` | ~4.6x; gap 0.084 → 0.056; 1B run used 200 cells (217M used 2,000) and 155 vs 174 perturbations | MISMATCH (factor), plus unstated sample-size confound |
| Fig 2 caption 1152–1154 | Incremental gain at most +0.002 in any run | ≤ +0.002 | verdict JSONs C2; reports Phase 2 | cross-pert: +0.0007, −0.0003, −0.0001, +0.0020 (other splits up to +0.0021; gene+both +0.0023) | MATCH |
| Fig 2 caption 1154 | "pre-registered 0.005 threshold" | 0.005 | `R/attention-grn-217M/scripts/phase12_verdict.py:72-83`; spec `P/attention-grn-extraction-and-evaluation.md:720-723` | 0.005 is a constant in the run script. The spec's criterion is "lower 95% CI bound > 0"; 0.005 appears only as a power target (spec line 190). Spec C3 needs ≥ 80% retained; script uses 50%. | AMBIGUOUS (not found as a pre-registered threshold) |
| Fig 2 caption 1154–1155 | Residualizing "collapses it to chance" | chance | verdict JSONs C3 | 1B 0.5434 (OLS), C3 PASS | MISMATCH for 1B |
| 1164–1166 | Operator: "16 internal features read out over 60 genes, 7.7 kB in total" | 16 × 60 genes, 7.7 kB | `R/manifold-discovery-217M/scripts/phase10_compaction_chain.py:148-165`; `reports/compaction_chain.json`; `reports/factor_gene_loadings.csv`; `artifacts/heads/compact/hard_sparse_16f_60g.pt` | 16 SVD factors × (60 of 154 head-dim + 60 of 1,232 residual-dim coordinates); 7,744 bytes = operator only; head file 8,973 bytes extra; needs full 217M forward pass | MISMATCH ("genes" and "in total") |
| 1166–1168 | Entry H65 = hematopoietic differentiation sequence | H65 | `S/manifold-discovery-217M-FINAL_SUMMARY.md:34` | H65 = 34-stage DAG | MATCH. Note: internal panel has only 6 of 290 anchors at progenitor stages (HSC 2, MPP 2, CMP 1, MEP 1) — computed from `artifacts/anchors/anchor_meta_internal.csv`. "From stem cell to mature" is thin at the stem end. |
| 1170 | "clears all four quality gates fixed before the run" | 4 gates | `R/manifold-discovery-217M/reports/quality_gates_spec.json`; `quality_gates_let_anchor.json` | 5 gates were fixed (incl. blocked-permutation p ≤ 0.001, ≥ 2,000 perms). The permutation gate was never computed (no script computes it). | AMBIGUOUS / NOT FOUND (5th gate) |
| 1171–1174 | Trustworthiness = "do cells that neighbour inside the model also neighbour biologically?"; 0.811 vs null 0.799 | 0.811 / 0.799 | `reports/quality_gates_let_anchor.json`; `scripts/phase5_let_anchor.py:236, 258` | 0.8109 / 0.7985; computed as `trustworthiness(features, z, k=15)` on 290 anchors | Values MATCH. DIFFERENT ENDPOINT on definition: model-feature space vs 10-D read-out; no biological ruler; anchors not cells. |
| 1174 | Null = "cells shuffled within their own developmental branch" | — | `planning/research_plan.md:27-28` | stage labels of anchors permuted within branch | MATCH (minor wording: anchors) |
| 1175–1178 | Branch-holdout 0.370 vs null −0.001 | 0.370 / −0.001 | `quality_gates_let_anchor.json` | 0.3704 / −0.0010 | MATCH |
| 1178–1179 | Random holdout 0.834, donor holdout 0.716 | 0.834 / 0.716 | same | 0.8341 / 0.7159 | MATCH (holdout unit = anchors) |
| 1183–1185 | Null passes random and donor gates (0.799, 0.707) | 0.799 / 0.707 | same | 0.7986 / 0.7071 | MATCH. Note: `planning/research_plan.md:29` says the null "must FAIL all four gates"; it failed only two. |
| 1186–1188 | Trust margin narrow, 0.811 vs 0.799 | — | same | 0.8109 vs 0.7985 | MATCH |
| 1190–1192 | External: 13 donors, no shared cells; trust 0.896 | 13, 0.896 | `reports/external_validation_external.json` | 600 anchors, 13 donors, 46 tissues; trust 0.8957 | MATCH |
| 1192 | 95% CI [0.884, 0.899] | [0.884, 0.899] | `reports/external_validation_external_bootstrap.json` | [0.8839, 0.8991]; method = 200 draws of 80% of anchors without replacement; subsample mean 0.8916 | DIFFERENT ENDPOINT (subsampling interval, not a bootstrap CI; anchor-level, not donor-level as audit A3 asked, `A/audit-20260507.md:387`) |
| 1192–1193 | External branch-holdout 0.346 | 0.346 | same JSON; `scripts/phase7_external_validation.py:1-6, 64, 102-154` | 0.3464 = within-branch Spearman under a frozen head trained on all internal branches | DIFFERENT ENDPOINT (no branch withheld) |
| 1193–1194 | Zero-shot, no refitting, 12 donors from different tissues: 0.827 / 0.317 | 0.827 / 0.317 | `reports/zeroshot_transfer_anchor_head.json`; `scripts/phase8_zeroshot_transfer.py:70-130` | 160 anchors, 12 donors, 45 tissues; trust 0.8267; branch 0.3168 (frozen within-branch correlation) | Values MATCH; branch value is DIFFERENT ENDPOINT. External was also no-refit, so the contrast in the sentence is misleading. |
| 1195–1197 | Lung control, 50 anchors, random labels, fails 3 of 4 gates (−0.016, 0.013, −0.147) | 3 of 4 | `reports/external_validation_lung_nonhema.json` | 50 anchors, 4 donors; trust 0.79956 (< 0.80), rand −0.0164, donor 0.0134, branch −0.1469 | Values MATCH. MISMATCH on count: fails 4 of 4. Not disclosed: first lung control passed all 4 gates (`reports/external_validation_lung_control.json`: "BAD: negative control passed") and was rebuilt. |
| 1197–1199 | Compress "from a 154-dimensional dense form down to 7.7 kB, a factor of 2,400" | 2,400x | `reports/compaction_chain.json` | full drift 2,464-dim 18.214 MB → 7.744 kB = 2,352x; 154-dim dense 0.759 MB → 7.744 kB = 98x | MISMATCH |
| 1199 | Compression "costs no measurable quality" | — | same | vs full drift: trust 0.811→0.909, branch 0.370→0.433; vs 154-dim dense: branch 0.467→0.433. Internal panel only; compact operators never run on external/zero-shot. | AMBIGUOUS / overstated |
| 1199–1201 | "Something another laboratory could actually ship and reuse" | — | `S/manifold-discovery-217M-FINAL_SUMMARY.md:159-178` | Phase 12 LITE: extracted z10 is not better than PCA-10, SVD-10 or avg-pool on any endpoint | Overstated |
| Tab 4 caption 1209–1211 | "pre-registered catalogue of 141 candidate biological orderings" | 141 | `R/manifold-discovery-217M/planning/research_plan.md`; `STATUS.md`; `reports/hypothesis_registry*.csv` | No 141-entry catalogue. 141 is the topology pipeline. H92–H112 were made during the run (2026-05-05/07). | NOT FOUND |
| Tab 4 caption 1216–1217 | Trust pass mark 0.80, pre-registered | 0.80 | `reports/quality_gates_spec.json` | 0.80 | MATCH (definition problem as above) |
| Tab 4 caption 1224–1226 | External = 13 donors; zero-shot = 12 donors, all rows | 13 / 12 | `reports/external_validation_h103.json`; `reports/zeroshot_h103.json` | H103 external 106 anchors, 12 donors; H103 zero-shot 18 anchors, 9 donors, k=5 | DIFFERENT ENDPOINT for H103 cells |
| Tab 4 caption 1228–1231 | H38, H95, H103 are "category-based", checked by a "three-gate fallback" | — | `reports/h38_lite_quality_gates.json`; `reports/hypothesis_registry.csv`, `hypothesis_registry_3gate.csv`, `hypothesis_registry_sweep2.csv` | H38 LITE used 4 gates (category holdout 0.293). H103 is an ordinal depth ruler with only 2 distinct depths in 45 anchors (depth holdout NaN). H95 original verdict INCONCLUSIVE; 3-gate rule added afterwards (file times 17:37 → 18:27, 2026-05-05). | MISMATCH (reasons and pre-registration) |
| Tab 4, H65 row | 0.811, 0.370, 0.896, 0.346, 0.827 | as listed | reports above | 0.8109, 0.3704, 0.8957, 0.3464, 0.8267 | MATCH (0.346 is frozen within-branch, see above) |
| Tab 4, H38 internal | 0.814 | 0.814 | `reports/h38_lite_quality_gates.json` | 0.8135 (a hand-made 7-category proxy, not OmniPath) | MATCH |
| Tab 4, H38 external | n.t. | not tested | `reports/external_validation_H38_lite.json`; `R/manifold-discovery-217M/FINAL_SUMMARY.md` §16 | trust 0.883 (4/4 gates) | MISMATCH |
| Tab 4, H38 zero-shot | n.t. | not tested | `reports/zeroshot_H38_lite.json` | trust 0.787, FAIL | MISMATCH (a failure is hidden) |
| Tab 4, H95 internal / external | 0.800 / 0.888 | — | `reports/hypothesis_registry_3gate.csv`; `reports/external_validation_h95.json` | 0.8005 / 0.8885 | MATCH |
| Tab 4, H95 zero-shot | n.t. | not tested | `reports/zeroshot_H95.json` | trust 0.819 (3-gate pass) | MISMATCH |
| Tab 4, H103 internal / external | 0.860 / 0.877 | — | `reports/hypothesis_registry_sweep2.csv`; `reports/external_validation_h103.json` | 0.8601 / 0.8772 | MATCH |
| Tab 4, H103 zero-shot | 0.857 | 0.857 | `reports/zeroshot_h103.json` | 0.8567 with k=5 (not the fixed k=15), 18 anchors, 9 donors | DIFFERENT ENDPOINT |
| Tab 4, lung row | 0.800, −0.147 | — | `reports/external_validation_lung_nonhema.json` | 0.79956, −0.1469 | MATCH (rounding) |
| Tab 4 note 1265–1266 | Lung control "is an internal-panel construct only" | — | same JSON (`"frozen_from": "internal_anchor"`, panel lung_nonhema); `S/manifold-discovery-217M-FINAL_SUMMARY.md:80-90` (Phase 7) | separate lung panel scored with the frozen head | MISMATCH |
| Fig 3 bars | 0.811/0.896/0.827/0.800 and 0.370/0.346/0.317/−0.147 | — | TikZ `paper-biosystems/main.tex:1200-1201`; reports above | same | MATCH (mixed endpoints: 0.370 is a true holdout; the others are frozen within-branch values) |
| Fig 3 caption 1272–1276 | Trust clears 0.80 internal, rises external; lung at boundary; lung −0.147 vs hema 0.31–0.37 | — | reports above | 0.811, 0.896, 0.7996; −0.147 vs 0.317–0.370 | MATCH |
| Fig 3 caption 1277–1278 | Trustworthiness alone "passes" a clearly wrong manifold | passes | `reports/external_validation_lung_nonhema.json` | 0.79956 < 0.80, fails | MISMATCH |
| 1284–1285 | Head that "contributes most" = head 6 of layer 10 | L10H6 | `reports/head_layer_screen_summary.json`; `scripts/phase9_head_attribution.py:109-125` | L10H6 is best single-head read-out by composite (trust + random + branch); no contribution/ablation test | AMBIGUOUS (ranking rule differs from wording) |
| 1285–1288 | "Four strongest axes of variation account for only 18.3% of the total effect of ablating it" | 18.3% | `reports/factor_ablation.json`; `reports/factor_gene_loadings.csv`; `scripts/phase11_factor_ablation.py:1-14, 237-298` | top4_pct_of_total_impact 0.1825; the 4 are factors 1, 4, 11, 9 ranked by leave-one-out impact (singular values 1.117, 1.041, 1.004, 1.014 — not the top 4) | DIFFERENT ENDPOINT (value matches) |
| 1288 | scGPT source figure 66.2% | 66.2% | `S/manifold-discovery-217M-FINAL_SUMMARY.md:135` | 66.2% | MATCH |
| 1289–1291 | "Branch-holdout collapses from 0.973 to 0.123" | 0.973 → 0.123 | `reports/factor_ablation.json` | branch_balanced_acc 0.9728 → 0.1234 (frozen linear probe) | DIFFERENT ENDPOINT |
| 1292–1295 | Architectural reading: trajectory pre-training spreads the manifold | — | none | no scGPT run in this project; one model only | Speculative (hedged with "appears") |
| 1297–1298 | Trustworthiness "clears the gate even on the lung control" | clears | `reports/external_validation_lung_nonhema.json` | 0.79956 | MISMATCH |
| 1305–1307 | Original test: 0/48 TFs; specific = ≥ 2 of top-20 genes in TRRUST targets | 0/48, ≥2 | `R/sae-atlas-217M/scripts/remaining_phases.py:454`; `outputs/phase8t_chance_baseline/summary.json` | ≥2 of top-20; 0 of 48 | MATCH |
| 1308 | Audit pattern P6 flagged it as underpowered | P6 | `A/audit-20260507.md:192-213, 392` | P6 flagged "no positive-control TF"; A8 asked for GATA1/MYC/TAL1 | MATCH (approximately; the power analysis came later, A8 v1) |
| 1309–1310 | GATA1 57 TRRUST targets, TAL1 10 | 57 / 10 | `R/sae-atlas-217M/outputs/phase8t_positive_tf_rerun/run.log:2` | GATA1=57, MYC=100, TAL1=10 | MATCH |
| 1310–1311 | Threshold near-impossible under random draws | — | `S/sae-atlas-217M-FINAL_SUMMARY.md:151-165` | P(≥2 overlap, K=5) = 0.004 for GATA1; TAL1: 0 of 4,928 features | MATCH |
| 1311–1312 | "0%" measured database narrowness, "rather than the model" | — | `outputs/phase8t_chance_baseline/summary.json` | 45 of 48 TFs had zero responding features (detection 4/100 perturbations) | Overstated (most of the 0/48 is non-detection, not TRRUST size) |
| 1313–1315 | Re-run with feature IDs; DoRothEA ChIP-seq; thresholds 2–5 | — | `R/sae-atlas-217M/scripts/audit_a8_rerun_phase8t_targeted.py:52, 246-296` | as stated; GATA1 used 50 of 95 cells | MATCH |
| 1315–1317 | Feature 2610: 8 of top-20 in the 470-target set | 2610, 8, 470 | `outputs/phase8t_positive_tf_rerun/specificity_table.csv`; run.log:3 | best_feature_id 2610, best_overlap 8; GATA1 ChIP targets 470 | MATCH |
| 1317–1318 | p_null = 0.030 over 1,000 random-feature draws (Phipson & Smyth cited) | 0.030 | `outputs/phase8t_positive_tf_rerun/specificity_with_null.csv`; script lines 274-296 | 0.03, N_NULL = 1000; computed as plain fraction b/m (Phipson–Smyth would give 31/1001 = 0.031); null draws from all features, not from perturbation-responsive ones | MATCH on value; method differs from the citation |
| Fig 4 bars (TRRUST) | 0.003, 0.001, 0.001, 0.001 | as listed | TikZ `paper-biosystems/main.tex:1285`; `specificity_with_null.csv` | 0.003, 0.0, 0.0, 0.0; observed n_specific = 0 at every threshold | MISMATCH (values) and DIFFERENT ENDPOINT (these are chance pass rates, not p-values of an observed result) |
| Fig 4 bars (ChIP-seq) | 0.602, 0.252, 0.096, 0.030 | as listed | TikZ line 1286; `specificity_with_null.csv` | 0.602, 0.252, 0.096, 0.03 | MATCH |
| Fig 4 caption | TRRUST 57, ChIP 470; p = 0.030 at ≥5 for n = 1 TF; Bonferroni α/4 = 0.0125 | — | as above | as above; arithmetic correct | MATCH |
| 1335–1338 | Single TF, single line, single ChIP-seq dataset, post-hoc threshold; MYC and TAL1 not perturbed in K562 | — | `S/sae-atlas-217M-FINAL_SUMMARY.md:123, 143`; `phase8t_positive_tf_rerun/summary.json` | MYC, TAL1: 0 cells in the local K562 file. ChIP set = DoRothEA ChIP-seq TSV (an aggregated resource). | AMBIGUOUS ("not perturbed in K562" is broader than what was checked; the K562 h5ad is not on disk now, so I could not re-check) |
| 1341 | p = 0.030 uncorrected | 0.030 | as above | 0.03 | MATCH |

No hardware or run-time claims appear in lines 1043–1348.

---

## 3. Other problems found in the source files (not paper numbers, but they affect trust in the sources)

- `S/attention-grn-217M-FINAL_SUMMARY.md:12` says 1B C3 = FAIL; the JSON says PASS.
- `S/attention-grn-217M-FINAL_SUMMARY.md:36` says no run's attention beats any trivial baseline. In RPE1, attention beats mean expression and 1−dropout, both significant (`phase12_verdict_rpe1.json`).
- `S/manifold-discovery-217M-FINAL_SUMMARY.md` (the copy the paper used) and `R/manifold-discovery-217M/FINAL_SUMMARY.md` have split apart. The runs copy has the H38/H95 back-fill results (§16–17); the summaries copy does not. The paper's "n.t." cells come from the stale copy.
- `S/manifold-discovery-217M-FINAL_SUMMARY.md:136` gives top-10 factors ≈ 33%; the JSON gives 37.6%.
- `S/manifold-discovery-217M-FINAL_SUMMARY.md:282` says H103 zero-shot used 7 donors; the JSON says 9.
- Verdict JSONs for RPE1 and Adamson carry the K562 scope text ("Replogle K562 non-targeting controls only"), and the 1B report header says "MaxToki-217M-HF". These are template errors.
- The loader and summary call Adamson a CRISPRa screen (`setup/dataset_loader.py:96`). From general knowledge, Adamson et al. 2016 used CRISPRi. Not verified from a file; outside this line range.
- In the GATA1 re-test, "responding" features are chosen by a Mann–Whitney test over all token positions of 50 cells against control tokens. So the test unit is tokens, not cells (inferred from script lines 211-232). All five p-values are 0.0.

---

## 4. What I did not check

- Lines outside 1043–1348 (for example Methods text that may define the metrics differently).
- The source papers' numbers (2.36x, 0.59, 66.2%). I only checked that the project summaries report them.
- Whether MYC and TAL1 are in the Replogle K562 file. The file path in `setup/dataset_loader.py:21` is not on disk now.
- I did not re-run any model. All computed numbers come from `per_pair.parquet` and `anchor_meta_internal.csv`.

## 5. Plain-words summary

Most numbers in this part of the paper are copied correctly. The problems are in what the numbers are said to mean. The attention figure uses screen-derived targets but says "curated" pairs. The 1B model is about 4.6 times bigger, not 10. The CRISPRi "real effect" disappears once you compare against how often each answer occurs by chance. The manifold section calls head-space coordinates "genes", counts only part of the file size, uses the wrong starting point for the 2,400x claim, and describes trustworthiness as a biology check when the code does not use biology. The lung control fails all four gates, not three. Table 4 hides three tests that were run, one of which failed.
