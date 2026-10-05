## TF specificity of layer-5 SAE features (MaxToki-217M, K562 knockdowns)

This analysis asks whether knocking down a transcription factor (TF) changes SAE features whose top genes are that TF's known targets. For each TF it selects the layer-5 SAE features that respond to the knockdown. It then tests whether the top-20 genes of those features overlap the TF's target set more than matched chance.

### Data

- **Model.** MaxToki-217M, Hugging Face safetensors, float32, Apple MPS, eager attention, batch size 1, max_len 2,048. torch 2.11.0, transformers 5.5.4. The forward pass stops at the input of block 5 (`hidden_states[5]`).
- **SAE.** The layer-5 SAE of section 02 (`projects/maxtoki/runs/sae-atlas-217M/outputs/v3_sae/layer_05/sae_final.pt`).
- **Cells** (Replogle K562 Perturb-seq, `replogle_concat.h5ad`), 9,200 in total:
  - 400 reference controls and 800 pool controls (for fake TFs), drawn from the 10,691 K562 non-targeting cells with seed 20261001;
  - 7,500 knockdown cells for 87 TFs, at most 100 per TF (seed 20261002); 34 to 100 cells per TF; GATA1 has 95;
  - 500 catalog control cells (seed 42), the SAE training cells of section 02.
- **Input.** Counts rebuilt and encoded with `inputs_v3` (section 00). Two cells (rows 282850 and 401789) have expm1(X) values that are already whole numbers with smallest value 1; `inputs_v3.counts_accept_unit_one` reads them as counts.
- **TFs.** 87 knocked-down TFs. 20 primary TFs have DoRothEA ChIP-seq target sets: ATF4, BRF2, CEBPZ, E2F6, GATA1, GTF2B, HINFP, MAX, MYBL2, SETDB1, SP2, SRF, STAT5A, TBP, TERF1, TERF2, TFDP1, THAP1, THAP11, ZNF24. TRRUST targets are used as a second database.
- **Gene universe.** 6,328 genes seen in the 500 catalog cells. GATA1 targets in the universe: 221 ChIP-seq, 11 TRRUST.
- **Knockdown strength** (expm1(X) values as a label, not a model input). For the 80 TFs whose gene is in the 6,546-gene panel, the TF's mean expression in knockdown cells is at most 13% of the reference level (median 0%). GATA1 and 6 other TF genes are not in the panel.

### Method

1. **Per-cell feature values.** For each cell, the layer-5 SAE code is computed at every position. The unit of analysis is the mean of each feature over the gene positions of the cell (`<bos>` and `<eos>` excluded).
2. **Top-20 catalog.** Encode every position of the 500 catalog cells, including `<bos>`/`<eos>` (named `<SPECIAL>`). For each feature and gene, take the mean activation over positions where the feature is active. The top-20 genes are the 20 genes with the highest mean (no minimum count). All 4,928 features have 20 genes; 47 lists contain `<SPECIAL>`. 1.1% of top-20 slots hold a gene seen in 5 or fewer cells.
3. **Selection of responding features.** For each TF: two-sided Mann–Whitney U test of the knockdown cells against the 400 reference cells, per feature, with the full tie correction. BH over the 4,928 features. A feature responds if q < 0.05 and |mean difference| > cut-off. Cut-offs: 0.5 (the value of the deployed analysis), 0.25, 0.1, 0.05, 0.02, 0.01 and 0. K = number of responding features. The full tie correction matters here: 78% of the 9,200 cells start with the same gene (RANBP1), so a feature that fires only near the start of the sequence has the same per-cell mean in many cells.
4. **Statistic.** M = the best overlap between the top-20 of any responding feature and the TF's target set. T = min(M, 5) if M ≥ 2, else 0. T is the statistic of the deployed analysis (thresholds 2 to 5). p = P_null(T ≥ T_obs). The uncapped M is also tested.
5. **Nulls.**
   - rf: K random catalog features (exact). This is the null of the deployed analysis.
   - cm (count-matched): each top-20 gene swapped for a random gene from the same detection-count bin (exact, by hypergeometric convolution).
   - cml: bins of detection count × gene-length tertile (cuts at 21,392 and 56,262 bp).
   - fake: random subsets of the 800 pool cells of the knockdown size, with the whole selection run again (2,000 per size).
   - fakeK: as fake, but the fake TF keeps its K most-changed features.
   - okd: the responding features of every other knockdown with K ≥ 1.
6. **Rank.** The cm p-value of the same features against the target set of each of 291 ChIP-seq TFs (≥ 20 targets) or 109 TRRUST TFs (≥ 5 targets). The rank of the true TF is reported.
7. **Multiple testing.** BH across TFs, separately per database, cut-off and null (over the 20 primary TFs and over all 87). No correction across cut-offs or nulls.
8. **Power.** Plant k = 1 to 8 true targets into the top-20 of one random responding feature. 200 repeats per (TF, cut-off, database, k), for the 20 primary TFs.
9. **GATA1 bootstrap.** Knockdown and reference cells are resampled separately with replacement (1,000 resamples, seed 20263027). The selection is run again in each resample with scipy's Mann–Whitney test.

### Checks

| Check | Result | Pass |
|---|---|---|
| Encoding check before every forward pass | `inputs_v3.assert_encoding_batch` on every cell of all 368 batches of 25: 9,200 / 9,200 pass, 0 order violations (1,212 positions are exact ties in another order, which is allowed) | yes |
| Negative control for the check | Tokens ranked from the stored log1p values fail the check in 100 / 100 sample cells | yes |
| Catalog cells = SAE training cells | 1,019,996 / 1,019,996 positions identical | yes |
| Live codes vs stored SAE codes (catalog cells) | identical at all 1,019,996 positions (same 32 features, value difference 0.0); the catalogs built from both are identical for all 4,928 features | yes |
| `<bos>` code | the same in every cell (largest difference 0.0 over 9,200 cells) | yes |
| Second model path (12 cells: 6 GATA1, 6 reference) | `LlamaForCausalLM(output_hidden_states=True).hidden_states[5]` + `TopKSAE.encode`, no hooks: per-cell means equal the stored ones within 2.4 × 10⁻⁷ (values up to 2.6); tokens identical | yes |
| Mann–Whitney code vs scipy (float64) | largest difference in p 0.0 (GATA1 and a 95-cell fake group); same BH sets | yes |
| Exact swap null vs Monte Carlo (GATA1's 7 features, 10,000 draws) | P(best ≥ t), t = 1 to 8: largest gap 0.005 (cm) and 0.007 (cml), at most 1.8 Monte Carlo SE; expected overlaps within 0.010 | yes |
| GATA1 K and best overlap from the written tables (by gene name) | all 14 (cut-off × database) rows equal `tf_results.csv` | yes |
| Selection false-positive rate (1,000 fresh random splits of the 1,200 controls, 400 vs 95) | P(≥ 1 responding feature): 0.032 at cut-off 0 (95% Clopper–Pearson 0.022 to 0.045); 0.002 at 0.01; 0.001 at 0.02 and 0.05; 0 at 0.1 and above. Single-feature p-values uniform (KS p 0.03 to 0.45 for 5 pre-chosen features; 0.36 for a random feature per split; share p < 0.05: 0.031 to 0.062) | yes |
| Fake-TF p-values uniform | 1,000 new fake TFs per primary TF (20,000), scored against the stored nulls; 28 settings: KS p median 0.56, min 0.040 (2 of 28 below 0.05; 1.4 expected); share of p ≤ 0.05 at most 1.0%; two-sample chi-square p < 0.05 in 5 of 132 per-TF tests | yes |
| fakeK null with a new seed (7 TFs with K > 0 at cut-off 0.25) | chi-square p 0.12 to 0.98; overlaps counted by a second implementation (by gene name) | yes |
| Planted signal (k = 8 ChIP targets, GATA1, cut-off 0.5) | detected with probability 1.00 [0.98, 1.00] by the uncapped count-matched test | yes |

Shared-pool effect. Fake TFs all share the same 400 reference cells and 800 pool cells. So single-feature fake-TF p-values are not uniform across fake TFs (KS p down to 5 × 10⁻²³ for one feature; 5.0% of fake TFs have ≥ 1 BH-significant feature at cut-off 0). The pool itself does not differ from the reference (0 BH-significant features; 7 features with p < 0.001 against 4.9 expected; 331 with p < 0.05 against 246). With fresh random splits the test is calibrated (row above).

### Results

**Main findings.**
- No TF has SAE features that are specific to its targets. Over 87 TFs, 7 cut-offs and both databases, no TF passes BH (q < 0.05 across TFs) under the cm, cml, fakeK or okd null.
- GATA1 changes 7 features at cut-off 0.5. The best one (feature 3843) has 4 of its top-20 genes among the 221 GATA1 ChIP-seq targets. This is close to chance for these genes (cm p = 0.32; expected best overlap 3.1). GATA1 ranks 82nd of 291 TFs for these features.
- The deployed analysis reported one GATA1 feature with 8 of its top-20 genes among GATA1 ChIP-seq targets (p = 0.030 against random features). In this SAE, no feature of the 4,928 has more than 5 GATA1 ChIP-seq targets in its top 20 (10 features have 5), and none of those 10 responds to GATA1 at cut-off 0.5.

**Responding features per primary TF** (`task_data/tf_summary.tsv`):

| TF | Cells | BH q < 0.05 (no cut-off) | K at 0.5 | 0.25 | 0.1 | 0.05 | 0.02 | 0.01 |
|---|---|---|---|---|---|---|---|---|
| GATA1 | 95 | 4,097 | **7** | 13 | 21 | 46 | 249 | 543 |
| MAX | 100 | 2,888 | 1 | 3 | 10 | 20 | 95 | 304 |
| THAP1 | 100 | 1,492 | 0 | 1 | 3 | 6 | 18 | 93 |
| HINFP | 100 | 1,435 | 0 | 1 | 3 | 6 | 21 | 87 |
| TFDP1 | 100 | 725 | 0 | 0 | 0 | 4 | 4 | 36 |
| CEBPZ | 100 | 680 | 0 | 0 | 2 | 2 | 7 | 24 |
| GTF2B | 34 | 510 | 0 | 1 | 2 | 7 | 24 | 90 |
| MYBL2 | 81 | 266 | 0 | 0 | 4 | 5 | 7 | 17 |
| TERF2 | 47 | 250 | 1 | 1 | 1 | 3 | 7 | 21 |
| TBP | 100 | 186 | 0 | 1 | 1 | 4 | 8 | 20 |
| 10 others | 49–100 | 0–22 | 0 | 0 | 0 | 0–1 | 0–2 | 0–5 |

- Per-cell feature means are small (median over features of the reference mean 0.012).
- At cut-off 0.5, 17 of 87 TFs have at least one responding feature: AATF, ATF5, BDP1, CDC5L, ECD, GATA1 (7), GTF2A1, MAX, MED1 (2), PDCD11 (2), PHB2 (2), RUVBL1, SNIP1, TAF1, TAF5, TERF2, TSG101 (2). With no effect cut-off, 74 of 87 do.

**GATA1 responding features at cut-off 0.5** (n = 95 knockdown and 400 reference cells; effect = mean difference of per-cell feature means, knockdown minus reference; interval: percentile cell bootstrap, knockdown and reference resampled separately, 1,000 resamples):

| Feature | Effect [95% CI] | Cohen's d | Selected in bootstraps | GATA1 ChIP targets in top-20 |
|---|---|---|---|---|
| 4905 | 1.695 [1.568, 1.804] | 3.50 | 100% | 1 (GNA12) |
| 2665 | 0.756 [0.588, 0.917] | 1.19 | 100% | 1 (SUCO) |
| 1725 | 0.738 [0.687, 0.781] | 3.43 | 100% | 2 (FBXW11, ACER3) |
| 4264 | 0.702 [0.634, 0.767] | 2.71 | 100% | 1 (RNF220) |
| 3763 | 0.597 [0.460, 0.741] | 1.13 | 92% | 1 (DIS3L2) |
| **3843** | 0.561 [0.496, 0.626] | 2.37 | 97% | **4** (MBD5, FRYL, CDKAL1, TBL1XR1) |
| 892 | 0.506 [0.434, 0.579] | 1.99 | 56% | 0 |

- Across bootstraps, K at 0.5 has median 7 (95% range 5 to 7). Only one other feature is ever selected. The best ChIP overlap is 4 in 96.5% of resamples and 2 in 3.5%. It never reaches 5.
- The four ChIP targets of feature 3843 are seen in 2, 16, 23 and 100 of the 500 catalog cells. Two are rare genes. This is why the count-matched null expects a best overlap of 3.1.

**GATA1 tests at every cut-off, ChIP-seq targets** (`summary_tables.md`, Table B):

| Cut-off | K | M | T | p rf | p cm | p cml | E[M] under cm | p fake | p fakeK | p okd (other knockdowns) | Rank / 291 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 7 | 4 | 4 | 0.055 | **0.318** | 0.431 | 3.11 | 0.0005 | 0.189 | 0.059 (16) | 82 |
| 0.25 | 13 | 4 | 4 | 0.101 | 0.471 | 0.546 | 3.54 | 0.0005 | 0.554 | 0.088 (33) | 176 |
| 0.1 | 21 | 4 | 4 | 0.158 | 0.554 | 0.628 | 3.71 | 0.0005 | 0.696 | 0.128 (46) | 178 |
| 0.05 | 46 | 4 | 4 | 0.314 | 0.656 | 0.720 | 3.89 | 0.0005 | 0.865 | 0.151 (52) | 190 |
| 0.02 | 249 | 5 | 5 | 0.405 | 0.573 | 0.786 | 4.86 | 0.0005 | 0.802 | 0.105 (56) | 122 |
| 0.01 | 543 | 5 | 5 | 0.689 | 0.667 | 0.852 | 5.01 | 0.0005 | 0.945 | 0.422 (63) | 138 |
| 0 | 4,097 | 5 | 5 | 1.000 | 0.943 | 0.979 | 5.42 | 0.0015 | 1.000 | 0.568 (73) | 177 |

- With TRRUST targets, GATA1's best overlap is 1 of 11 at every cut-off, so T = 0 and every p = 1.
- The uncapped cm p equals the capped one at every cut-off (for example 0.318 at 0.5).
- fakeK at 0.5: p = 0.189, Monte Carlo SE 0.009 (2,000 draws).
- The fake null is degenerate. Random control groups of these sizes never select a feature at cut-off 0.25 or above (0 of 5,000 fake TFs at 0.5). So every knockdown with T ≥ 2 gets the floor p of 0.0005 (BH q 0.005 across the 20 primary TFs). This shows that the knockdown changed something, not that the features know the TF's targets.
- The okd p of 0.059 at 0.5 is the smallest value possible with 16 other knockdowns, so it cannot reach 0.05.
- MAX at 0.5 (1 feature, best ChIP overlap 2): p cm 0.33, rank 73 of 291. TERF2 at 0.5 (1 feature): ChIP overlap 1, T = 0.

**All 20 primary TFs, by cut-off** (`summary_tables.md`, Table A; "BH" = number of TFs with q < 0.05 across TFs):

| Cut-off | Database | Testable TFs | Median K | T ≥ 2 | BH fake | BH cm | BH cml | BH fakeK | BH okd | True TF in top 5% of ranks |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | ChIP | 3 | 1 | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| 0.25 | ChIP | 7 | 1 | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| 0.1 | ChIP | 9 | 3 | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| 0.05 | ChIP | 11 | 5 | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| 0.02 | ChIP | 12 | 8 | 9 | 9 | 0 | 0 | 0 | 0 | 0 |
| 0.01 | ChIP | 13 | 24 | 10 | 10 | 0 | 0 | 0 | 0 | 0 |
| 0 | ChIP | 16 | 258 | 14 | 14 | 0 | 0 | 0 | 0 | 0 |
| 0.5 to 0.01 | TRRUST | 2 to 8 | – | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0 | TRRUST | 11 | 266 | 1 (TFDP1) | 1 | 0 | 0 | 0 | 0 | 1 |

Over all 87 TFs (Table A2 of `summary_tables.md`), 0 TFs pass BH under cm, cml, fakeK or okd at any cut-off.

**Nominal hits** (p < 0.05 under a matched null; none survives BH):

| TF | Database | Cut-off | K | Best overlap | p cm | p fakeK | p okd | Smallest q |
|---|---|---|---|---|---|---|---|---|
| BDP1 | TRRUST | 0 | 1,084 | 2 | 0.009 | 0.27 | 0.36 | q cm 0.52 (87 TFs) |
| SETDB1 | ChIP | 0 | 22 | 4 | 0.036 (cml 0.033) | 0.046 | 0.54 | q fakeK 0.44 |
| TBP | ChIP | 0.05 | 4 | 3 | 0.31 | 0.017 | 0.40 | q fakeK 0.19 |

Each hit appears at one cut-off and under one or two nulls only. With 7 cut-offs × 2 databases × 4 matched nulls × 87 TFs and no correction across cut-offs, a few such p-values are expected.

**Power** (planted targets; mean detection probability at alpha 0.05; one TF: 95% Wilson interval over 200 repeats; several TFs: 95% interval from a bootstrap over TFs; `summary_tables.md` Table D, `power_summary.json`):

| Cut-off | Database | k planted | TFs | rf | cm | cm uncapped | cml | cml uncapped | fakeK |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | ChIP | 4 | GATA1 | 0.88 [0.82, 0.91] | 0.00 [0.00, 0.02] | 0.29 [0.23, 0.36] | 0.00 | 0.29 | 0.88 |
| 0.5 | ChIP | 8 | GATA1 | 1.00 [0.98, 1.00] | 0.00 [0.00, 0.02] | 1.00 [0.98, 1.00] | 0.00 | 1.00 | 1.00 |
| 0.5 | TRRUST | 2 | GATA1 | 1.00 [0.98, 1.00] | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.5 | ChIP | 8 | 3 (GATA1, MAX, TERF2) | 1.00 | 0.67 [0.00, 1.00] | 1.00 [1.00, 1.00] | 0.66 | 1.00 | 1.00 |
| 0.1 | ChIP | 8 | 9 | 1.00 [1.00, 1.00] | 0.89 [0.67, 1.00] | 1.00 [1.00, 1.00] | 0.86 [0.63, 1.00] | 1.00 | 1.00 |
| 0 | ChIP | 8 | 16 | 0.50 [0.25, 0.75] | 0.56 [0.31, 0.81] | 1.00 [1.00, 1.00] | 0.56 | 0.93 [0.82, 1.00] | 0.50 |

- The cap at 5 removes the power of the capped tests for GATA1. Under the count-matched null, P(best of 7 GATA1 features ≥ 5) = 0.099 (exact; Monte Carlo 0.099). The capped T is at most 5, so its p can never go below 0.099. The uncapped test finds 8 planted targets every time (P(best ≥ 8) = 0.0006 under the null).
- The observed GATA1 result is not hidden by the cap: the observed best overlap is 4, and the uncapped p equals the capped p (0.32).
- TRRUST: 2 planted targets are found every time when the TF has at least 2 TRRUST targets in the universe. TFs with fewer can never pass (mean power 0.75 to 0.88 at lower cut-offs).
- The okd null has no power at 0.5 (16 other knockdowns; smallest possible p 0.059).

**The TRRUST rule of the deployed analysis** (≥ 2 TRRUST targets in the top-20 of a responding feature; `old_trrust_test.csv`):

| Cut-off | 48 TFs of the deployed analysis: with responding features / pass | 20 primary TFs | All 87 TFs |
|---|---|---|---|
| 0.5 | 10 / 0 | 3 / 0 | 17 / 0 |
| 0.25 | 17 / 0 | 7 / 0 | 34 / 0 |
| 0.1 | 25 / 0 | 9 / 0 | 47 / 0 |
| 0.02 | 30 / 0 | 12 / 0 | 57 / 0 |
| 0 | 41 / 2 (BDP1, ERCC2) | 16 / 1 (TFDP1) | 74 / 3 |

Every pass has p rf ≥ 0.05. GATA1 meets this rule at no cut-off (best TRRUST overlap 1).

### Verification

No separate verification by a second agent is recorded for this analysis. The report itself describes these cross-checks (Checks table): a second model path without the hook code for 12 cells (`scripts/v3_tf_specificity_verify.py`, `checks/verify.json`); the GATA1 numbers re-derived from the written tables by gene name; the exact nulls compared with Monte Carlo; the Mann–Whitney code compared with scipy; and the calibration checks (`scripts/v3_tf_specificity_calib_check.py`, `calibration.json`, `calibration_extra.json`), including a second, independent overlap count for the fakeK null. All agreed.

### Limits

- One cell line (K562), one layer (5) and one SAE training seed. Whether another seed gives the same GATA1 features was not tested.
- Two target databases only (DoRothEA ChIP-seq for 20 TFs, TRRUST).
- The fake-TF null cannot measure specificity, because control groups select no feature at cut-offs ≥ 0.25. It is reported next to the K-matched and other-knockdown nulls.
- Fake TFs share one reference group and one pool. Their p-values are calibrated given that pool, but they are not independent draws from the population.
- No correction across the 7 cut-offs or across the nulls. BH is applied only across TFs.
- The rank test is descriptive: target sets of different TFs differ in size and gene mix.
- Power was simulated for the 20 primary TFs only.
- GATA1 and six other TF genes are not in the 6,546-gene panel, so their knockdown strength cannot be checked.
- The top-20 definition (mean activation when active, no minimum count) is kept, so rare genes can enter top-20 lists. Only the matched nulls (cm, cml) correct for this.
- Gene length is the genomic span from a gene-position table that is not part of the release; the lengths used are in the `gene_length_bp` column of `projects/maxtoki/runs/sae-atlas-217M/outputs/v3_tf_specificity/task_data/gene_universe.tsv`; 54 of 6,328 genes have none and are put in the middle tertile.

### Files

- Scripts (`projects/maxtoki/runs/sae-atlas-217M/scripts/`): `v3_tf_specificity_extract.py` (cell manifest, encoding check, forward passes to block 5, per-cell SAE summaries), `v3_tf_specificity_taskdata.py` (gene universe, target sets, top-20 catalog), `v3_tf_specificity_stats.py` (selection, nulls, rank, power, BH, calibration), `v3_tf_specificity_calib_check.py`, `v3_tf_specificity_verify.py`, `v3_tf_specificity_summary.py` (GATA1 bootstrap and detail, tables, `run_config.json`). Shared code: `projects/maxtoki/setup/inputs_v3.py`, `projects/maxtoki/setup/hooks_v2.py`, `projects/maxtoki/setup/topk_sae.py`.
- Outputs (`projects/maxtoki/runs/sae-atlas-217M/outputs/v3_tf_specificity/`): `tf_results.csv` (all TFs × cut-offs × databases, all p and q), `summary.json`, `summary_tables.md`, `summary_counts.csv`, `power.csv`, `power_summary.json`, `old_trrust_test.csv`, `gata1_bootstrap.json`, `gata1_detail.json`, `knockdown_efficiency.json`, `calibration.json`, `calibration_extra.json`, `checks/verify.json`, `prepare.json`, `cell_manifest.csv`, `cell_manifest_meta.json`, `cells/` (per-cell summaries and the token ids fed to the model), `fake_null/`, `stats_cache/`.
- Reusable data: `.../outputs/v3_tf_specificity/task_data/` with a `README.md` that defines every file (gene universe, top-20 catalog, target sets, responding features with signed effects, p and q for all 87 TFs, per-TF summaries, test results, cell manifest, per-cell feature means for all 9,200 cells in `cell_feature_means.npz`).
- Provenance: `run_config.json` (device, versions, seeds, cell IDs, feature IDs, input sha256, code sha256, wall time per chunk, encoding-check result), `run_config_extract.json`, `run_config_stats.json`.
- Run time: 9,200 forward passes at 0.21 s per cell, about 34 minutes in total (MPS).
