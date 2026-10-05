# numbers_C — number-provenance audit, PLOS ONE paper lines 1348–1725

Paper: `<REPO_ROOT>/projects/maxtoki/paper-plos-one/main.tex` (file has 1,724 lines; range covers
Results: steering, cross-architecture, cross-layer, spectral geometry, CKA, "audit framework in practice"; Discussion; Limitations; Conclusion).
Figure data were checked against the TikZ in `<REPO_ROOT>/projects/maxtoki/paper-biosystems/main.tex`.

Status codes: **MATCH** / **MISMATCH** / **NOT FOUND** / **AMBIGUOUS** / **DIFFERENT ENDPOINT** (the number exists, but the source computed a
different thing, unit, denominator, resampling scheme or hardware). "Verified" = I read the file or ran a small check. "Inference" = my reasoning, not in any file.

No model forward pass was run. Only small numpy checks (CCA chance level, cell-sample overlap, cosine of saved signatures).

---

## 1. Source key (all paths absolute)

Prefix `MT` = `<REPO_ROOT>/projects/maxtoki`

| Key | Absolute path |
|---|---|
| S1 | MT/summaries/exhaustive-mapping-217M-FINAL_SUMMARY.md |
| S2 | MT/runs/exhaustive-mapping-217M/outputs/experiment3/steering_summary.json |
| S3 | MT/runs/exhaustive-mapping-217M/scripts/experiment3_rerun.py |
| S3b | MT/runs/exhaustive-mapping-217M/outputs/experiment3/state_signatures.npz and MT/runs/exhaustive-mapping-217M/outputs/experiment3_rerun.log |
| S4 | MT/paper-biosystems/main.tex (TikZ figure data, lines 1358–1396, 1424–1469, 1512–1660) |
| S5 | MT/runs/topology-141-217M/outputs/phase14_cross_model/phase14_summary.json and cross_model_alignment.csv |
| S6 | MT/runs/topology-141-217M/outputs/phase4_scgpt/phase4_summary.json and cross_model_alignment_scgpt.csv |
| S7 | MT/runs/spectral-geometry-217M/outputs/phase9b/bootstrap_pearson.json |
| S8 | MT/runs/spectral-geometry-217M/outputs/phase9b/summary.json; script MT/runs/spectral-geometry-217M/scripts/phase9b_cross_model.py |
| S9 | MT/runs/topology-141-217M/scripts/phase14_cross_model_cca.py |
| S10 | MT/runs/topology-141-217M/outputs/phase11_autoloop/iter_04_cross_layer_consistency/rows.json |
| S11 | MT/runs/topology-141-217M/outputs/phase11_autoloop/iter_04_replication/summary.json |
| S12 | MT/runs/spectral-geometry-217M/outputs/phase1/per_layer_metrics.csv |
| S13 | MT/runs/spectral-geometry-217M/outputs/phase1/null_feature_shuffle.json |
| S14 | MT/runs/spectral-geometry-217M/outputs/phase1/run_config.json |
| S15 | MT/runs/spectral-geometry-217M/scripts/phase1_svd.py |
| S16 | MT/runs/spectral-geometry-217M/outputs/phase8/cka_per_layer.csv and summary.json |
| S17 | MT/runs/spectral-geometry-217M/scripts/phase8_stability.py (and phase0_extract.py) |
| S18 | <REPO_ROOT>/pipelines/residual-stream-spectral-geometry.md |
| S19 | MT/README.md |
| S20 | MT/audits/audit-20260507.md |
| S21 | MT/audits/audit-20260507-completion.md |
| S22 | MT/audits/audit-20260507-completion-v2.md |
| S23 | MT/runs/circuit-tracing-217M/outputs/groupkfold_crispri/summary.json |
| S24 | MT/runs/circuit-tracing-217M/scripts/remaining_phases.py |
| S25 | MT/runs/circuit-tracing-217M/scripts/audit_a2_groupkfold_crispri.py |
| S26 | MT/audits/audit_a2_groupkfold_status.md |
| S27 | MT/runs/sae-atlas-217M/outputs/phase8t_positive_tf_rerun/specificity_with_null.csv |
| S28 | MT/runs/sae-atlas-217M/outputs/phase8t_positive_tf_case_study/summary.json |
| S29 | MT/runs/sae-atlas-217M/outputs/phase8t_chance_baseline/summary.json |
| S30 | MT/runs/manifold-discovery-217M/reports/external_validation_external_bootstrap.json |
| S31 | MT/audits/bootstrap_cis_summary.json and MT/audits/audit_a3_bootstrap_cis.py |
| S32 | MT/summaries/attention-grn-217M-k562-report.md (dated Apr 17) and MT/summaries/attention-grn-217M-FINAL_SUMMARY.md |
| S33 | MT/summaries/manifold-discovery-217M-FINAL_SUMMARY.md |
| S34 | MT/runs/manifold-discovery-217M/reports/hypothesis_registry.csv, hypothesis_registry_sweep2.csv, hypothesis_registry_sweep3.csv |
| S35 | <REPO_ROOT>/repos/topology-biomechinterp2/iterations/iter_0045/brainstormer_hypothesis_roadmap.md |
| S36 | <REPO_ROOT>/pipelines/topology-geometry-141-hypotheses.md |
| S37 | MT/setup/token_dictionary.json + <DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/crispri_validation/data/gene_name_id_dict_gc104M.pkl |
| S38 | MT/requirements.txt (transformers==5.5.4); MT/.venv/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:410–425; .../transformers/utils/output_capturing.py:201–262 |
| S39 | MT/summaries/spectral-geometry-217M-FINAL_SUMMARY.md |
| S40 | MT/runs/topology-141-217M/FINAL_SUMMARY.md and MT/runs/topology-141-217M/EXTENDED_FINDINGS.md |
| S41 | MT/setup/maxtoki_adapter.py:255–279 |

---

## 2. Three problems that matter most (read these first)

### 2.1 The steering result measures a skipped transformer block, not a feature edit (verified in code and output)

- S3 line 256: `h_li = out_clean.hidden_states[li]`. In HF Llama, `hidden_states[li]` is the **input** to block `li`.
- S3 lines 262–273: the patched state `h_li + delta` is written as the **output** of `model.layers[hl]`, with `hl = li` for li = 0, 3, 6, 9 (S3 lines 147–155).
- So for L0, L3, L6 and L9 the hook throws away everything block `li` computed. The "edit" is: delete block `li`, then add a small SAE delta.
- For L11: `hidden_states[11]` is the output **after the final RMSNorm** (S38: `hidden_states[-1]` is tied to `last_hidden_state`, which is `self.norm(...)`). The script writes this normed tensor back as the pre-norm output of block 10, so the final norm is applied twice.
- The output data show exactly this. Within each layer, the three different features (some with positive, some with negative pseudotime correlation) give the same Δs to 3 decimals, the same top genes, and the same result at α=2 and α=5 (S2 `features[*].alpha_results`):
  - L0: Δs 0.0958 / 0.0962 / 0.0968 (F40, F1657, F4797); top genes SOD3, KITLG, APOE for all three.
  - L6: −0.0079 / −0.0079 / −0.0079; top genes C1QB, CPA3, HBE1 for all three.
  - L11: 0.0044 / 0.0043 / 0.0057; top gene FTL for all three; |Δlogit| ≈ 1.07 for all three.
- The source summary even reports this as an "incidental finding" (α-saturation, S1 lines 189–201). It is a symptom of the bug, not a property of the model. The audit rated exhaustive-mapping P1 "clean" and P9 "correct" and praised the α-saturation diagnostic (S20 lines 87, 131, 251, 273). So the audit missed it.
- Consequence: every steering number in lines 1349–1401 (Δs values, 5x, 20x, L6 "inversion", L0 genes SOD3/KITLG, L6 C1QB, L11 FTL, "non-monotonic mid-layer basin") is a block-removal / double-norm effect. It is not evidence about what the SAE features do.

### 2.2 Cross-model "internal gene geometry" is only the input embedding table (verified)

- Topology Phase 14 and Phase 4 use `tap = static` (S5/S6 CSV column `tap`; S9 lines 150–180): the token embedding rows of each model, PCA to 30 dims.
- Spectral Phase 9b (the 0.382 and its CI) uses MaxToki `embed_tokens` vs Geneformer `embeddings.word_embeddings` (S8 script lines 5–10, 72–106).
- The paper's figure caption (line 1420) says "internal gene geometry". The text (lines 1444–1446) then concludes "input embeddings carry most of the gene-pair geometric load", which cannot be tested when only input embeddings were compared.
- The CCA number has no null in the source (S9 computes nulls only for Pearson and top-1). My simulation (random 30-dim vs 30-dim data, exact CCA, mean of top-10 canonical correlations): n=350 gives 0.429 (SD 0.009, p95 0.444); n=380 gives 0.411 (SD 0.008, p95 0.427). The scGPT values 0.401 / 0.438 / 0.406 (S6) sit at this chance level. So scGPT CCA "0.40" means no alignment at all, and Geneformer's 0.78 is about 0.35 above chance, not 0.78 above zero. (Inference: sklearn's NIPALS CCA may differ slightly from exact CCA.)

### 2.3 Several "audit in practice" and Discussion statements have no source, or conflict with the source

- The directional-accuracy bug was not found by a P3 phrase search. The first audit accepted 54.6% (S20 lines 86, 130, 203, 317). The A2 status note (S26 lines 62–90, 18:08) still called it robust. The bug was found while re-implementing Phase 11 for A2, a P4 (GroupKFold) action (S25 lines 1–6).
- "H115 and H118 were retired after two negatives each, freeing compute for H95 and H103": H115/H118 appear in no MaxToki run file. They exist only in the prior pinned repo S35 (iter_0045): H115 "retire_now", but H118 "prioritize_hardening" (not retired; it is the ancestor of H123). H95/H103 came from fixed 6-candidate sweeps (S34), not from a retirement loop.
- "Audit consumed roughly three days of agent compute": not in any file. File times put the whole audit plus both remediation cycles on 2026-05-07, from 16:20 (first audit-day output, S6) to 21:10 (S22). That is about 5 hours. (Inference from file modification times.)
- "~50 GPU-hours": hardware was a MacBook Pro, Apple Silicon, 32 GB, MPS backend (S32 FINAL_SUMMARY line 127). The ~50 h is summed wall-clock, about half of it the manifold run.
- "15 hours of human supervision": no source file mentions it.

---

## 3. Full claim table

Line numbers are paper lines. "Value in source" is the exact value found.

### 3.1 Trajectory steering (lines 1348–1401)

| # | Paper line | Claim text (short) | Value in paper | Source file:line | Value in source | Status | Notes |
|---|---|---|---|---|---|---|---|
| 1 | 1349–1351 | Exhaustive mapping supplies the project's "first causal result" | first causal result | S1 lines 3–50; S20 line 131 | Exp 1 (ablation-based "causal edges") and Stage-2 circuit tracing and SAE Phase-6 patching are also interventions | AMBIGUOUS | Wording overstates "first". |
| 2 | 1351–1353 | Immune cells early in the "inferred developmental ordering", from Tabula Sapiens | TS immune cells, early | S3 lines 41, 109–131; S2 `n_cells`, `n_early_cells` | 200 random TS immune cells; "pseudotime" = PC1 of log1p expression; early = bottom quartile = 50 cells | MATCH (data) / wording overstated | S1 line 132 admits PC1 is a proxy, not diffusion pseudotime. "Developmental ordering" overstates a first principal component. |
| 3 | 1353–1356 | "We artificially add activity along selected internal features" | feature steering | S3 lines 147–155, 256, 262–273; S38 | Hook replaces block li output with block li input + SAE delta (block skipped); L11 applies final RMSNorm twice | DIFFERENT ENDPOINT | See §2.1. Verified by code; the identical per-feature outputs confirm it. |
| 4 | 1358–1366 | Δs = cos(z′,g_late) − cos(z′,g_early) − baseline; z′ is "the edited representation"; g are "directions corresponding to late and early developmental states" | representation space | S3 lines 188, 194–195, 296–302 | z′ = steered **logits** averaged over positions; g_early/g_late = mean clean logits of bottom/top PC1 quartile | DIFFERENT ENDPOINT (minor) | Logit space, not representation space. cos(g_early, g_late) = 0.881 (S3b log line 16; I recomputed 0.8810 from the npz). S1 line 159 wrongly says ≈0.999 (source-internal error; not used by paper). |
| 5 | 1366–1368, Fig | L0 clearly positive; L6 reverses sign | L0 +0.0962; L6 −0.0079 (Fig) | S2 `per_layer_summary_alpha5` | L0 0.096226; L6 −0.007897 | MATCH (numbers) / DIFFERENT ENDPOINT (meaning) | Numbers match. They measure block removal (§2.1). |
| 6 | Fig (S4 l.1382) | L3 +0.0178, L9 +0.0141, L11 +0.0048 | as listed | S2 | L3 0.017829; L9 0.014076; L11 0.004817 | MATCH | TikZ values rounded correctly. |
| 7 | 1370–1372 | L0 push ≈5x the strongest deeper-layer effect (L3) | ≈5x | S2 (derived) | 0.096226 / 0.017829 = 5.40 | MATCH | |
| 8 | 1372; Fig caption 1383 | L0 ≈20x the weakest (L11) | ≈20x | S2 (derived) | 0.096226 / 0.004817 = 19.98 | MATCH | |
| 9 | 1372–1373 | "the L6 inversion is robust" | robust | S2; S1 lines 94, 197 | No null, no CI. frac_mat 0.00 in 3 features × 50 cells, but the 3 features give identical results (one effective intervention). |Δs| 0.0079 is smaller than L9's 0.0141 | MISMATCH (overstated) | The paper calls L9 (0.0141) "near a noise floor" but calls the smaller L6 (0.0079) "robust". Internally inconsistent. |
| 10 | 1373–1375; Fig caption 1385–1386 | No permutation null bounded "for L9 and L11" | null missing for L9/L11 only | S2, S3 | No null was computed for **any** layer, including L0 and L6 | MISMATCH | Wording implies L0/L6 had one. L3 (+0.0178, larger than L9) is not classified at all. |
| 11 | 1376–1378 | Top-upregulated genes: L0 SOD3, KITLG; L6 C1QB; L11 FTL | genes | S2 `top_upregulated` | L0: SOD3, KITLG, APOE; L6: C1QB, HBE1/CPA3; L11: FTL, MT-CO3, MT-CO2/MT-ND1 | MATCH (names) / DIFFERENT ENDPOINT | Same genes for all 3 features per layer, so they describe the block removal, not the feature. The labels ("progenitor/stemness markers", "terminal erythroid") are interpretation; FTL co-appears with mitochondrial genes. |
| 12 | Fig caption 1387–1391 | Unlike Geneformer's monotone L0→L17 progression; MaxToki "non-monotonic mid-layer basin" | qualitative | S1 lines 100–101 | Stated in source summary | MATCH (to summary) / DIFFERENT ENDPOINT | The Geneformer pattern is the cited paper's claim, not re-run. The MaxToki "basin" rests on the bugged numbers. |
| 13 | 1396–1401 | Steering shows the model "can be moved along a differentiation axis it clearly represents" | qualitative | §2.1 | Not supported once the bug is taken into account | MISMATCH (overstated) | |

### 3.2 Cross-architecture scope (lines 1403–1434)

| # | Paper line | Claim text | Value in paper | Source file:line | Value in source | Status | Notes |
|---|---|---|---|---|---|---|---|
| 14 | 1404–1406 | Alignment with Geneformer "(same tokenizer, same family) replicates strongly" | replicates strongly | S5 `verdict`; S40 EXTENDED_FINDINGS line 17 | "Layer-1 PARTIALLY REPLICATES"; top-1 38–47% vs source-paper 72% | MISMATCH (overstated) | Tokenizer shared: MATCH (S39 line 247, 1500/1500 token ids). "Same family": Geneformer is a BERT encoder, MaxToki a Llama decoder (S8 `interpretation`). |
| 15 | 1406 | Geneformer CCA r = 0.78 | 0.78 | S5 | 0.7831 (lung), 0.7757 (immune), 0.7662 (ext-lung); mean 0.775 | MATCH | No CCA null in source (S9). Chance for this set-up ≈0.41–0.43 (my simulation, §2.2). |
| 16 | 1406–1408 | Pairwise gene-similarity Pearson 0.382, 95% CI [0.380, 0.384] | 0.382 [0.380, 0.384] | S7 | 0.38198; CI [0.37974, 0.38434]; n_bootstrap 1000 | MATCH (values) / DIFFERENT ENDPOINT | (a) Resampling is **80% subsampling without replacement, not a bootstrap**, and the CI is not rescaled; for a mean-like statistic that makes the SD about 2x too small (inference: variance ratio (n/m)(1−m/n) = 0.25). (b) This number comes from the spectral pipeline on 1,500 HVGs. The CCA (0.78), the scGPT numbers and the top-1 numbers come from the topology pipeline on 350–382 genes. The like-for-like topology Pearson for Geneformer is 0.387–0.400 (S5). |
| 17 | 1408 | scGPT CCA r = 0.40 | 0.40 | S6 | 0.4011, 0.4382, 0.4057 (mean 0.415) | MATCH (approx.) | Range 0.40–0.44. At chance level for 30-dim CCA with n≈350–380 (§2.2, inference from simulation). |
| 18 | 1409–1410 | scGPT pairwise Pearson ≈0, slightly negative on immune, z = −3.0 | ≈0; z −3.0 | S6 | lung −0.0084 (z −1.46); immune −0.0188 (z −2.95); ext-lung −0.0047 (z −0.83) | MATCH | |
| 19 | 1410–1411 | Top-1 retrieval 5–8% (scGPT) vs 38–47% (Geneformer) | 5–8%; 38–47% | S6; S5 | scGPT 4.97%, 6.57%, 7.63%; Geneformer 38.2%, 46.6%, 45.0% | MATCH | Procrustes rotation is fit and scored on the same genes; the null shuffles labels without refitting (S9 lines 181–209), so z-values are inflated (inference; z not quoted in paper). |
| 20 | 1411–1414 | "the failure isolates cleanly to architectural-family scope" | isolates cleanly | Paper's own caption 1431–1432; S6/S9 | scGPT differs in tokenizer, architecture and training data; only input embedding tables compared | MISMATCH (overstated) | Contradicts the paper's own figure caption ("without isolating which axis is responsible"). |
| 21 | 1419–1421 | Caption: "MaxToki's internal gene geometry aligns well with Geneformer" | internal geometry | S5/S6 `tap=static`; S8 script lines 5–10 | Static input-embedding rows only | DIFFERENT ENDPOINT | See §2.2. |
| 22 | 1422–1423 | "Three independent measures agree" | independent | S9 | All three come from the same two embedding tables on the same genes | AMBIGUOUS (overstated) | Not independent measurements. |
| 23 | 1427–1428 | Top-1 error bars = "range across cell-type panels" | cell-type panels | S5/S6 | Panels are tissue domains: lung, immune, external_lung | MISMATCH (minor) | TikZ: 0.425 ± 0.045 and 0.065 ± 0.015 match the ranges (S4 l.1440–1447). |
| 24 | Fig (S4) | Gene-pair bar: Geneformer 0.382, scGPT drawn at 0.0 | 0.382 / 0.0 | S7; S6 | 0.382 from spectral pipeline; scGPT −0.005 to −0.019 from topology pipeline | DIFFERENT ENDPOINT | Bars mix two pipelines and two gene sets. |

### 3.3 Cross-layer stability (lines 1437–1446)

| # | Paper line | Claim text | Value in paper | Source file:line | Value in source | Status | Notes |
|---|---|---|---|---|---|---|---|
| 25 | 1439 | Cross-layer Pearson 0.604–0.678 for H123 motif feature | 0.604–0.678 | S10 | lung 0.6542, immune 0.6040, ext-lung 0.6779 | MATCH | S10 report.json decision = "INCONCLUSIVE". |
| 26 | 1440 | 0.561 for Euclidean distance | 0.561 | S11 `aggregate_means_across_3_domains` | 0.56099 (per domain 0.478, 0.614, 0.591) | MATCH | Mean, while #25 is a range; mixed presentation. |
| 27 | 1440–1441 | 0.275 for triangle-defect | 0.275 | S11 | 0.27468 (per domain 0.187, 0.409, 0.228) | MATCH | S11 `verdict` text says "All three feature types are above 0.6" — false inside the source; S22 F11 corrects it. |
| 28 | 1441 | ~0.65 shares ~40% of variance | ~40% | arithmetic | 0.65² = 0.42 | MATCH | |
| 29 | 1444–1446 | Within-layer stability and cross-model agreement "express the same underlying fact: input embeddings carry most of the gene-pair geometric load" | qualitative | S5/S6/S8 | Cross-model work only used input embeddings; no test of "most of the load" | NOT FOUND (interpretation) | Partly circular (§2.2). |

### 3.4 Spectral geometry and CKA (lines 1448–1506)

| # | Paper line | Claim text | Value in paper | Source file:line | Value in source | Status | Notes |
|---|---|---|---|---|---|---|---|
| 30 | 1452–1453; Fig 1479 | Effective rank 561 at L0 → 94 at L11 | 561 → 94 | S12 | 560.809 → 93.643 | MATCH | L0 = token-embedding output (S41, S17 header). L11 = `hidden_states[-1]` = **after the final RMSNorm** (S38). The last step 172.1 → 93.6 crosses that norm. |
| 31 | 1453–1455 | Not strictly monotonic; shallow rebound across layers 7–9 | rebound 7–9 | S12 | L6 169.4, L7 174.4, L8 201.7, L9 201.9, L10 172.1 | MATCH | Also a small rebound L1 332.6 → L2 338.1, not mentioned. Rebound is +19% over L6. |
| 32 | 1456; Fig 1480 | ρ = −0.895, p = 8×10⁻⁵ | −0.895; 8e-5 | S14 | ρ −0.89510; p 8.3666e-05 | MATCH | |
| 33 | 1456–1457; Fig 1482 | Feature-shuffle null at L11 gives ER 573 | 573 | S13 | 573.081 | MATCH | |
| 34 | 1457; Fig 1483 | 6.1x ratio | 6.1x | S13 | 6.1198 | MATCH | |
| 35 | 1457–1459; Fig 1476–1486 | Ratio "confirms the compression is a property of the trained model"; "rules out" an architectural explanation | learned vs architectural | S15 lines 134–145 | Null permutes each hidden dimension independently across genes. Code comment calls it a "sparsity-artefact control". No untrained / random-init model was run. | DIFFERENT ENDPOINT | The shuffle tests "is there correlation between dimensions", not "was it learned". Overstated. |
| 36 | 1460–1462; Fig 1495 | Three **disjoint** 2,000-cell samples, seeds 42/43/44 | disjoint | S17 phase8 lines 100–102; S16 summary | Independent random draws from 592,317 cells; not forced disjoint | MISMATCH (minor) | My recomputation of the same draws: 6, 8 and 8 shared cells per pair (inference assumes n_obs = 592,317 as stated in S39 line 169). |
| 37 | 1463–1464; Fig 1497 | CKA 1.000 → 0.991, Δ −0.009 | 1.000→0.991 | S16 | L0 1.0 (all 3 pairs), L11 0.9914, delta 0.0086 | MATCH (numbers) / DIFFERENT ENDPOINT at L0 | MaxToki L0 is the context-free token embedding, so its per-gene mean is identical in every sample: CKA = 1 by construction. scGPT's layer 0 (0.979) includes expression-value input, so the L0 comparison is not like-for-like. |
| 38 | 1465–1466; Fig 1498 | scGPT 0.979 → 0.779, Δ −0.20 | 0.979→0.779 | S18 line 328; S16 summary `paper_scgpt_*` | 0.979 → 0.779, across random cell samples, same model | MATCH | Same design (cell samples). Not re-run here; it is the source paper's number. |
| 39 | 1469 | MaxToki training scale ~175M cells | ~175M | S19 line 14 | "175M human single-cell transcriptomes" | MATCH | |
| 40 | 1469–1472 | Re-averaging changes the per-gene embedding by <1% in CKA even at the final layer | <1% | S16 | 1 − 0.9914 = 0.86% | MATCH | The explanation "converged enough at this training scale" is not tested. S16 summary.json `verdict` says the opposite ("qualitatively reproduces the paper's CKA pattern") — source-internal conflict. |
| 41 | Fig (S4) | CKA points at L0, 1, 5, 7, 9, 11 | 1.000, 0.9997, 0.9991, 0.9966, 0.9928, 0.9914 | S16 | same | MATCH | Only 6 of 12 layers plotted. L10 (0.9931) > L9 (0.9928). |
| 42 | 1466–1468; Fig 1500–1504 | The scGPT "early conserved, late input-specific" pattern is "architecture-and-training-regime-dependent, not universal" | qualitative | S39 line 243 | Stated in summary | MATCH (to summary) / overstated | One model vs one other; L0 comparison is by-construction (#37). |

### 3.5 The audit framework in practice (lines 1510–1595)

| # | Paper line | Claim text | Value in paper | Source file:line | Value in source | Status | Notes |
|---|---|---|---|---|---|---|---|
| 43 | 1514–1517 | Two remediation cycles raised fully addressed share from 61% to 75% across 80 cells (8 × 10) | 61% → 75% | S20 lines 45–48; S22 lines 187–190 | v0 49/23/8 (61/29/10%); v1 68/26/6%; v2 60/18/2 (75/22/3%) | MATCH (as stated) | No re-scored per-cell matrix exists for v1 or v2 (searched MT; only the v2 text). S21 line 84 uses a different pattern-level score ("6/10 clean, was 5"), which also conflicts with S20 line 474 (5 clean / 3 partial / 2 weak). S22 line 183 says "0 untouched" vs "2 unaddressed". Both cycles on 2026-05-07 (S21 18:09, S22 21:10). |
| 44 | 1517–1519 | "Three audit-detected failures shifted headline numbers materially" | three | S22 F7, F8, F10 | P3 (54.61→53.48) and P1/P6 (GATA1) shifted headlines; the P8 example (manifold CI) did not shift any number | AMBIGUOUS | Other headline shifts existed (H123 0.505→0.462, 41x→14.6–41x; S21 lines 26, 36) but are not the three cited. |
| 45 | Fig caption 1527–1530 | 49 of 80 before; 60 of 80 after; only 2 cells untouched; residual 3% | 49/80, 60/80, 2 | S20 line 46; S22 line 190 | 49; 60; 2 unaddressed (2.5%) | MATCH | TikZ (S4) bars 61/68/75, 29/26/22, 10/6/3 match S22. |
| 46 | Fig caption 1530–1532 | Residual = external data, multi-day reimplementation, cross-model re-extraction | 3 reasons | S22 lines 159–172, 192–195 | S22 lists **4** deferred items; item 2 = multi-seed Phase-0 re-extraction | MATCH (to S22 line 192–195) / incomplete | Three reasons for two cells. |
| 47 | 1540–1548 | Bug: code counted how often observed change was negative, never compared to predicted | description | S24 line 327 | `if (actual_lfc < 0): n_correct_direction += 1` | MATCH | Verified in code. |
| 48 | 1541–1543 | "The audit agent searched the run summaries for 'directional accuracy' and then read the code behind it" (P3) | discovery route | S20 lines 86, 130, 203, 317; S26 lines 62–90; S25 lines 1–6 | First audit accepted 54.6% ("itself a null comparison"); A2 status called it robust; bug found while re-implementing Phase 11 for A2 (a P4 GroupKFold action) | NOT FOUND / MISMATCH | The discovery story and the P3 label are not in the sources. |
| 49 | 1549–1550 | Corrected 53.48% overall | 53.48% | S23 | 0.534761 | MATCH | Computed on 1,503,408 pairs; original was on 1,458,016 pairs (S31 / phase11 json). Different denominator. |
| 50 | 1550 | 53.42% per-source mean, 95% CI [52.60%, 54.16%] | 53.42 [52.60, 54.16] | S23; S25 lines 205–213 | 0.534187; CI [0.52605, 0.54165]; 1,000 bootstrap draws over 248 source genes | MATCH | S22 line 72 says "966 unique source genes"; S23 says 248. Source-internal conflict (paper does not quote it). |
| 51 | 1551 | Original 54.61% | 54.61% | S23 `comparison_vs_original_phase11`; S31 | 0.5461 (796,159 / 1,458,016) | MATCH | |
| 52 | 1555–1558 | Original ≥2-of-top-20 TRRUST gave p_null = 0.003 for GATA1 ("we re-confirmed") | 0.003 | S27 row GATA1,TRRUST,2 | 0.003 | MATCH | Earlier power run gave 0.004 (S28 `power_at_gata1_k=5.trrust`). |
| 53 | 1561–1564 | ChIP-seq direct targets at thresholds {2,3,4,5} inverted the verdict at the strict threshold | ≥5 inverted | S27 | ChIP ≥2: 2/5, p 0.602; ≥3: 1/5, p 0.252; ≥4: 1/5, p 0.096; ≥5: 1/5, p 0.030 | MATCH | All ChIP positives driven by one feature (2610, overlap 8). |
| 54 | 1567–1568 | "Six of the eight pipelines reported point estimates without bootstrap intervals on first pass" | six | S20 lines 245–258, 387 | P8 table: 7 "partial", 1 "clean"; A3 lists 5 headline numbers | NOT FOUND | |
| 55 | 1568–1570 | Audit prescribed 1,000 non-parametric bootstrap iterations on the unit of inference for each headline number | 1,000 bootstrap | S20 line 387; S31; S23; S7; S30 | A3 gives no iteration count. Done: 10,000 feature bootstrap (SAE); Wilson analytic CI then 1,000 per-source bootstrap (CRISPRi); 80% subsampling ×1,000 (spectral); 80% anchor subsampling ×200 (manifold); 10,000 (topology) | MISMATCH | Two of five are subsampling, not bootstrap; one started as an analytic Wilson interval. |
| 56 | 1570–1571 | None of the intervals overturned a qualitative conclusion | none | S22 line 213 | "No previous finding overturned by v2" | MATCH | |
| 57 | 1573 | Manifold branch-holdout 0.346, CI [0.270, 0.480] | 0.346 [0.270, 0.480] | S30 | observed 0.34644; CI [0.26976, 0.47979]; bootstrap mean 0.369 | MATCH (values) / DIFFERENT ENDPOINT | 80% anchor subsample **without replacement**, **200** iterations, unit = anchors (not donors). Not the "1,000 non-parametric bootstrap on the unit of inference" the paper describes. |
| 58 | 1576–1577 | Audit re-derived the cross-architectural-family scope qualifier (P7) | P7 | S20 lines 66–72, 230, 343–355, 389 | CT3 / A5 | MATCH | Based on the Phase-4 scGPT run done on audit day (S6, 16:20). |
| 59 | 1577–1578 | Cross-layer framing (P9, against "barely changes") | P9 | S22 F11 lines 104–121; S40 EXTENDED_FINDINGS line 26 | Reframing exists (F11, residual #9) | AMBIGUOUS | Pattern label "P9" not in source; the source tied it to residual #9 / replication. |
| 60 | 1578–1581 | H123 lung result sensitive to cell sample; verdict stable | qualitative | S22 F12 lines 129–147 | seed 42 mean AUROC 0.413 vs seed 43 0.494; mean |Δ| 0.081; 0/12 layers at both | MATCH | |
| 61 | 1581–1583 | Audit added explicit BH correction across the attention pipeline's tests (P8) | audit contribution | S32 k562-report line 30 (Apr 17); S20 lines 84, 248 | BH-corrected Wilcoxon already present before the audit; audit rated it "clean — exemplary" | MISMATCH | The audit did not add it. |
| 62 | 1585–1588 | Would have reported "no TF out of 48 shows specificity" and "54.61%" | 48; 54.61% | S29; S23 | 0/48; 0.5461 | MATCH | |
| 63 | 1588–1592 | Errors "in opposite directions"; "a single systematic bias cannot produce errors in both directions" is "the strongest evidence" | reasoning | — | — | NOT FOUND (overstated) | A bias toward confident headline claims can produce both. The 54.61→53.48 change is 1.1 points. |
| 64 | 1592–1595 | Headlines the audit did not flag are not formally validated | caveat | S20 lines 87, 131, 251, 273 | Confirmed by a concrete case: the steering hook bug (§2.1) was rated "clean"/"correct" | MATCH (caveat is real) | This is a live example, not just a theoretical limit. |

### 3.6 Discussion (lines 1598–1689)

| # | Paper line | Claim text | Value in paper | Source file:line | Value in source | Status | Notes |
|---|---|---|---|---|---|---|---|
| 65 | 1601–1604 | Contract "held without exception"; agent never re-read source paper | qualitative | — | No log or record of this | NOT FOUND | |
| 66 | 1605–1609 | Mandatory "positive control" slot in the topology pipeline forced a chance baseline the manual version lacked | template slot | CLAUDE.md template (10 sections, no positive-control slot); S36 (no "positive control" text); S20 line 206; S21 line 10 | Audit rated topology P6 "weak — lacks a positive control"; it was added later by audit action A1 | MISMATCH | Credit goes to the audit, not the template. |
| 67 | 1609–1612 | Manifold sweep over 12 candidate orderings spanned 24 hours of agent compute | 12 candidates, 24 h | S33 line 7, line 235; S34 `elapsed_s` | 24 h = main H65 run (2026-05-03 to 05-04). The 12-candidate sweep ran 2026-05-05; summed elapsed 1.46 h + 0.24 h = 1.70 h | DIFFERENT ENDPOINT | Two different things joined into one claim. |
| 68 | 1613–1617 | H115 and H118 retired after two negatives each, freeing compute for H95 and H103 | H115/H118 | S35 lines 16, 22; S34 | H115/H118 not in any MaxToki run. In the prior repo, H115 = "retire_now"; H118 = "prioritize_hardening" (not retired) | MISMATCH | H95/H103 came from fixed 6-candidate batches. |
| 69 | 1617–1619 | H95 and H103 passed their quality gates (3-gate fallback) | passed | S33 lines 244, 254, 261; S34 sweep2 verdict | H95 POSITIVE (3-gate); H103 POSITIVE_3GATE_FALLBACK | MATCH | |
| 70 | 1621–1622 | Five probes behind the regulatory-logic negative | five | S20 lines 315–320 | five listed | MATCH | |
| 71 | 1623–1633 | Tokenizer has no entries for lncRNAs, miRNAs, isoforms | excluded | S37 (checked) | 20,271 ENSG tokens; 0 MIR*, 3 LINC*; MALAT1, NEAT1, XIST, H19, MEG3, MIR155HG absent | MATCH | Verified by counting. |
| 72 | 1643–1644 | Confident-but-wrong results "on three of eight pipelines before correction" | three | S22 | Not stated; circuit-tracing and sae-atlas are clear; third unclear | AMBIGUOUS | |
| 73 | 1648 | Audit consumed "roughly three days of agent compute" | ~3 days | S20, S21, S22 file times; S6 time | All audit work on 2026-05-07, 16:20–21:10 (about 5 h) | NOT FOUND / likely MISMATCH | Inference from file modification times. |
| 74 | 1649–1650 | Eight pipelines required ~50 hours (about two days) of agent compute | ~50 h | S39 lines 181, 227; S32 FINAL line 125; sae-atlas FINAL line 272; circuit FINAL line 33; S1 line 165; S33 line 7; S34; topology logs; longevity FINAL line 7 | Sum of reported wall-clock ≈ 48–49 h (manifold ~24 h + sweeps ~2.6 h; exhaustive 6.2 h; attention 5.5 h; circuit 4 h; sae 2.5 h; spectral 1.7 h; longevity 1.5 h; topology ≥0.6 h) | MATCH (approx.) | Wall-clock on a laptop, not GPU time. |
| 75 | 1650–1652 | Audit "cost somewhat more than the analysis it checks" | audit > analysis | #73, #74 | ~5 h vs ~49 h by timestamps | MISMATCH (inference) | Also conflicts with "cheap" in line 1657 and 1709. |
| 76 | 1661–1665 | Catalogue "ports cleanly" between SCFMs with the transformers interface | qualitative | — | Only MaxToki was run | NOT FOUND (interpretation) | |
| 77 | 1676–1682 | "Three audit categories were left open": scVI/Palantir/CellTypist; AIDA; H123 on scGPT | three | S22 lines 159–172 | Four deferred: + multi-seed Phase-0 re-extraction for sae-atlas / circuit-tracing / exhaustive-mapping; H123 replication named as "scGPT or Geneformer V2-316M" | MISMATCH (omission) | |
| 78 | 1683–1685 | GATA1: n=1 TF, K562, post-hoc threshold, p = 0.030 uncorrected, "marginal under Bonferroni" | p 0.030; marginal | S27 | p 0.030 at ≥5; 4 thresholds (×2 databases) tested | MATCH (p) / MISMATCH (wording) | Bonferroni over 4 thresholds gives 0.12; over 8 tests 0.24. That is not significant, not "marginal". |
| 79 | 1686–1689 | No non-agent baseline was run | none | — | consistent with sources | MATCH | |

### 3.7 Conclusion (lines 1692–1710)

| # | Paper line | Claim text | Value in paper | Source file:line | Value in source | Status | Notes |
|---|---|---|---|---|---|---|---|
| 80 | 1697–1698 | Approximately 15 hours of human supervision | 15 h | searched MT (excluding paper folders) | No mention anywhere | NOT FOUND | Appears only in paper drafts (main.tex lines 224, 306, 575, 645, 661, 898). |
| 81 | 1698–1699 | ~50 **GPU-hours** of agent compute | GPU-hours | S32 FINAL line 127; #74 | MacBook Pro, Apple Silicon, 32 GB, MPS, float32; totals are wall-clock | DIFFERENT ENDPOINT (unit/hardware) | Discussion (line 1649) says "hours of agent compute"; Conclusion upgrades it to "GPU-hours". |
| 82 | 1700–1702 | "Six interrelated findings" | six | — | Not defined in this range | AMBIGUOUS | |
| 83 | 1707–1710 | Audit "caught errors its own authors had missed, in two directions at once"; "cheap enough" | qualitative | #48, #64, #75 | The two errors were real; but the audit missed the steering bug; its cost statement conflicts with line 1650–1652 | AMBIGUOUS (overstated) | |

---

## 4. List of all non-MATCH rows (for the caller)

**DIFFERENT ENDPOINT**
- #3, #5, #11, #12 (steering): numbers measure a skipped block (L0–L9) or double RMSNorm (L11), not a feature edit.
- #4: Δs is in logit space, not "representation".
- #16: CI for 0.382 is 80% subsampling (not bootstrap, not rescaled); 0.382 is from a different pipeline and gene set than the CCA / scGPT numbers.
- #21, #24: "internal gene geometry" is input-embedding tables only; figure mixes two pipelines.
- #35: feature-shuffle null cannot separate learned from architectural.
- #37: L0 CKA = 1 by construction (static token embedding); not comparable to scGPT's 0.979.
- #57: manifold CI uses 80% anchor subsampling ×200, not 1,000 bootstrap on donors.
- #67: 24 h was the main manifold run, not the 12-candidate sweep (1.7 h).
- #81: "GPU-hours" is wall-clock on an Apple-Silicon laptop.

**MISMATCH**
- #9 L6 "robust" (no null; smaller than L9 which is called noise); #10 no null for any layer; #13 steering interpretation.
- #14 "replicates strongly" vs source "PARTIALLY REPLICATES"; #20 "isolates cleanly" (contradicts own caption); #23 panels are tissues, not cell types.
- #36 samples not disjoint (6–8 shared cells per pair).
- #48 discovery route of the directional-accuracy bug; #55 "1,000 non-parametric bootstrap" prescription; #61 BH correction pre-dated the audit.
- #66 template has no positive-control slot; the audit added it.
- #68 H115/H118 story (from a different prior repo; H118 was not retired).
- #75 audit cost vs analysis cost; #77 three vs four open items; #78 "marginal under Bonferroni" (0.12–0.24).

**NOT FOUND**
- #29 input embeddings carry "most" of the load; #48 phrase-search narrative; #54 "six of eight"; #63 "single bias cannot produce both"; #65 contract held / no paper re-reading; #73 three days of audit compute; #76 ports cleanly; #80 15 hours of supervision.

**AMBIGUOUS**
- #1 "first causal result"; #22 "independent measures"; #44 "three failures shifted headlines"; #59 P9 label; #72 "three of eight pipelines"; #82 "six findings"; #83 conclusion wording.

---

## 5. Source-internal inconsistencies found on the way (not quoted by the paper, but relevant)

- S1 line 159 says cos(g_early, g_late) ≈ 0.999; the log (S3b line 16) and the npz give 0.881.
- S1 lines 136–138 say `hidden_states[11]` is the residual "before RMSNorm"; in the installed transformers it is after the final norm (S38).
- S16 summary.json `verdict` says MaxToki "qualitatively reproduces the paper's CKA pattern"; S39 line 229 says it "does not replicate".
- S11 `verdict` says all three feature types are above 0.6; triangle-defect is 0.275.
- S22 line 72 says 966 source genes; S23 says 248.
- S31 fallback text says 1,000 permutations for Phase 9b; S8 says n_perm 500.
- S40 FINAL_SUMMARY line 453 says MaxToki and Geneformer share "autoregressive training"; Geneformer is a masked (BERT-style) model.

---

## 6. Plain-words summary

Most raw numbers in lines 1348–1725 copy correctly from the run files. About 45 of 83 checked items match.

The biggest problem is the steering section. The code replaces a whole transformer block's output with that block's input. So the "feature steering" results really measure what happens when a block is deleted. Three different features give the same answer to three decimals. The audit did not catch this.

The cross-model section compares only the input embedding tables, not internal geometry. Its scGPT CCA of 0.40 is at the chance level for this set-up (about 0.41–0.43). One CI uses subsampling, not a bootstrap.

Several Discussion and audit claims have no source or conflict with it: the H115/H118 story, the three days of audit compute (files show about five hours on one day), "GPU-hours" (it was a laptop), the P3 discovery story, the BH correction, the positive-control template slot, and "15 hours of supervision".

Not done: I did not re-run any model, so I did not measure how large the true feature-steering effect is once the hook is fixed.
