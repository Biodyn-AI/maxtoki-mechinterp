## Feature circuits and the CRISPRi direction test (MaxToki-217M, K562)

This analysis builds a map of "edges" between SAE features. An edge means: removing a source feature at one layer consistently changes a target feature at a later layer. It then asks whether the sign of these edges predicts which way a gene moves when its source gene is silenced by CRISPRi.

### Data

- **Model.** MaxToki-217M, Hugging Face safetensors, float32, Apple MPS. torch 2.11.0, transformers 5.5.4.
- **SAEs.** The 12 SAEs of section 02.
- **Cells for tracing.** 200 K562 non-targeting control cells from `replogle_concat.h5ad`, drawn with numpy `default_rng(42)` from the 10,691 control rows and sorted (`inputs_v3.load_k562_control_cells(200, pool=200, seed=42)`). Encoded from counts (section 00). 195 of 200 cells are cut at 2,046 genes; tokens per cell 1,434–2,048 (median 2,048).
- **Cells for knockdown labels.** All K562 knockdown cells of `replogle_concat.h5ad` (6,546 genes in the file) and 3,000 K562 non-targeting control cells drawn with `default_rng(42)` from the 10,691.
- **Gene-set databases for feature selection.** GO Biological Process, KEGG, Reactome and TRRUST TF target sets, each cut to the K562 gene panel and kept if they have ≥ 5 genes (2,377 terms).

### Method

**Part 1: choosing source features.** The rule of the deployed analysis picks the features whose top genes are most enriched in gene-set databases.

1. Top-20 catalog per feature and layer: the mean activation over the positions where the feature is active, over all 1,019,996 positions of the 500 SAE training cells (including `<bos>`/`<eos>` as `<SPECIAL>`); the 20 genes with the highest mean.
2. Enrichment: for each feature and term with ≥ 2 shared genes, a hypergeometric p (N = 6,329 distinct names, K = term size, n = list size). Rows with p < 0.1 are BH-adjusted together per layer; significant = q < 0.05.
3. Score = sum of −log10 p over the feature's significant rows. The 30 features with the largest score are taken at each of layers 0, 3, 6 and 9 (120 source features).

**Part 2: tracing edges.** Each source feature is removed in each cell, and the change of every later SAE feature is measured.

1. Edit: `hooks_v2.Ablate(s, [f])`, which adds −z_f(x) · W_dec[:, f] to the live input of block s at every position, including `<bos>`/`<eos>` (section 00).
2. Read-out: the captured live input of every later site t = s + 1 to 11 (t = 11 is the input of `lm_head`), encoded with that layer's SAE. For each cell and each of the 4,928 target features: Δz = mean over positions of (z with the edit − z without it).
3. Per (source, target) pair over the 200 cells: Cohen's d = mean(Δz) / sqrt(var(Δz, ddof = 1) + 10⁻¹²); consistency = max(number of cells with Δz > 0, n − that number) / n.
4. Edge if |d| > 0.5 and consistency > 0.7. "Inhibitory" if mean Δz < 0 (removing the source lowers the target). About 3.8 million (feature, target) pairs are tested, with no multiple-testing correction (the rule of the deployed analysis).
5. Exact speed-ups: the forward pass starts at block s from the stored clean input of block s; a source feature that is zero at every position gives Δz = 0 without a forward pass.

**Part 3: knockdown log-fold-changes (LFC).** One log only.

1. n_ig = round(expm1(X_ig) / u_i), with u_i = the smallest non-zero expm1(X) in cell i. Every cell was checked to be whole-number multiples of u_i (largest deviation 4.7 × 10⁻⁴; 3 of 50,729 knockdown cells already had u_i = 1). This gives integer counts up to one factor per cell.
2. v_ig = log1p(10,000 · n_ig / Σ_g' n_ig'), summing over the 6,546 genes in the file.
3. LFC(s, g) = mean of v_ig over the K562 cells whose guide targets s − mean of v_ig over the 3,000 control cells.
4. A silenced gene is used only with ≥ 10 cells. LFCs were computed for 1,543 candidate genes (a superset of the top-10 genes of the source features); 394 have ≥ 10 cells.
5. Observed decrease = LFC < 0. No LFC in the records is exactly 0.

**Part 4: CRISPRi direction test.** Each edge says "removing source feature A lowers (or raises) target feature B". The test turns this into gene-level predictions.

1. Records (rule in `projects/maxtoki/runs/circuit-tracing-217M/scripts/audit_a2_groupkfold_crispri.py:51-162`): pair each top-10 gene of an edge's source feature with each top-10 gene of its target feature (no self pairs). Per gene pair: evidence = number of such triples, n_inhibitory, and max |d|. Keep the pair if evidence ≥ 2 or max |d| > 2.
2. Predicted decrease = n_inhibitory / evidence > 0.5 (a tie counts as "increase").
3. Keep the record if the source gene has a K562 guide with ≥ 10 cells and the target gene is measured.
4. Metrics: accuracy, balanced accuracy, Matthews correlation coefficient (MCC), Cohen's kappa. Baselines: always "decrease" (the majority class), always "increase", the accuracy expected from the two class rates alone, and a target-direction baseline that uses no model. The target-direction baseline predicts each target gene's usual direction in the other 4 of 5 folds of silenced genes (seed 42).
5. Intervals: 95% percentile grouped bootstrap, 2,000 resamples (seed 42). Three ways to group records: (a) silenced genes (n = 248); (b) source-feature groups, i.e. silenced genes that sit in the same set of source features with ≥ 1 edge (n = 150); (c) connected components of silenced genes that share any source feature (n = 42; the largest has 140 genes). The best constant is chosen again in each resample. The target-direction baseline is fitted once on the 5 folds and not refitted inside resamples.
6. Exploratory follow-up: the same metrics in 6 bands of |LFC|.

### Checks

**Source features.**
- Fed the catalogs of the deployed analysis, the enrichment code rebuilds all 12 of its enrichment tables (same rows in the same order, p within 8.5 × 10⁻¹³ relative), and the selection code rebuilds its 120 source features exactly.
- The layer-5 catalog built here equals the catalog built independently in section 03 (4,928 of 4,928 ordered lists identical).
- No tie at the cut-off: score of the 30th vs 31st feature 138.41 vs 137.70 (L0), 113.06 vs 112.98 (L3), 90.58 vs 89.25 (L6), 87.69 vs 86.60 (L9).

**Input encoding.**
- At load time all 200 cells pass `inputs_v3.check_encoding` (counts are whole-number multiples of the cell's smallest value; largest deviation 3.7 × 10⁻⁴).
- Before every chunk, the next ≤ 12 cells are checked again with `assert_encoding_batch` before any forward pass: 27 of 27 chunks pass (317 cell checks, 0 order violations; a few exact ties in either order, which the check allows). The spot check (below) checked its cells the same way (5 of 5 chunks, 200 cells).

**Tracing.**

| Check | Result | Pass |
|---|---|---|
| Zero edit (`hooks_v2.ZeroDelta(s)`, same partial forward, 4 source layers × 200 cells) | largest \|Δz\| = 0.0 in every cell; 0 edges | yes |
| Edges at the very next layer (s → s + 1) | L0→L1 37,382; L3→L4 2,360; L6→L7 2,598; L9→L10 4,138 | yes |
| Partial forward vs full forward (2 cells × 8 features) | largest difference 0.0 in hidden states and in Δz | yes |
| 3 edges recomputed from scratch | see the spot-check table below | yes |
| Edges recomputed a second way (two-pass numpy, no running mean) | same 318,387 edges, all signs equal, largest \|Δd\| 4.7 × 10⁻⁷ | yes |

**Spot check** (`scripts/v3_circuit_spotcheck.py`). Separate code: cells re-drawn from the h5ad file, counts rebuilt with its own code, tokenised with `tokenize_cell`. The edit is a plain forward hook that adds decode(z with f set to 0) − decode(z) (CPU SAE) to the output of the module that produces `hidden_states[s]`. The read-out is `output_hidden_states[t]`. Same rows and tokens as the trace. Three edges drawn with seed 20261001:

| Edge | d (trace) | d (from scratch) | Consistency (trace / scratch) | Largest per-cell difference in Δz |
|---|---|---|---|---|
| L0 F199 → L1 F580, excitatory | 0.6294 | 0.62937 | 0.95 / 0.95 | 4.4 × 10⁻⁹ |
| L9 F2421 → L10 F2835, inhibitory | −0.6047 | −0.60471 | 0.985 / 0.98 | 1.3 × 10⁻⁸ |
| L6 F4315 → L11 F973, excitatory | 0.9334 | 0.93343 | 0.96 / 0.96 | 3.7 × 10⁻⁸ |

The d differences (< 5 × 10⁻⁵) come from 4-decimal rounding in the CSV. In the second edge, one cell has a Δz within 1.3 × 10⁻⁸ of zero, and its sign differs between MPS and CPU; this moves consistency from 0.985 to 0.98.

**Knockdown labels.**
- Without rounding (expm1(X) used directly), the LFC changes by at most 7.6 × 10⁻⁸.
- A second implementation (cells found with `dataset_loader`, controls drawn again, own count code) gives the same LFC for 5 random silenced genes (largest difference 1.1 × 10⁻¹⁴) and the same cell counts.
- On-target knockdown: all 248 silenced genes have their own LFC < 0 (median −0.67; 69% below −0.5).

### Results

**Main findings.**
- The edge map has 318,387 edges; 57.9% are inhibitory. Most come from layer-0 sources.
- The circuit has no detectable skill at predicting which way a gene moves after silencing. Balanced accuracy is 50.19% and MCC is +0.0037; both intervals include chance under all three groupings.
- The circuit is below always saying "decrease" (by 2.2 points) and 8.3 points below a target-direction rule that uses no model (58.54%).

**Source features** (rank order; `source_features.json`; scores, best terms and top-10 genes in `annotation/selection_detail.csv`):
- L0: 3635, 3083, 1495, 4594, 3421, 73, 199, 948, 1213, 1216, 1116, 1715, 3605, 4802, 734, 603, 1762, 968, 136, 2591, 1990, 3810, 3582, 4714, 4321, 1883, 661, 1087, 2785, 3765
- L3: 3050, 1370, 1335, 1952, 2982, 3262, 4036, 150, 3662, 4912, 3708, 2737, 1949, 3020, 4029, 693, 3043, 4361, 1863, 2050, 3295, 2256, 2217, 3150, 778, 1891, 4162, 1885, 3788, 4801
- L6: 3559, 271, 2942, 4456, 3396, 925, 4212, 4255, 1371, 2337, 4818, 3937, 3421, 1422, 4315, 1508, 3798, 2720, 3018, 2123, 840, 3397, 991, 4029, 262, 1645, 922, 3654, 376, 2700
- L9: 1468, 1107, 1873, 2007, 4645, 1826, 271, 630, 1294, 3199, 1879, 3683, 1652, 4255, 1089, 3493, 198, 3971, 2551, 2421, 1386, 4880, 1863, 2753, 1137, 2680, 4170, 813, 3664, 56
- Rank 1 per layer: L0 F3635 (score 481; best term Reactome G1/S transition; top genes include ST13, FKBP4, STIP1, TCP1), L3 F3050 (352; KEGG proteasome), L6 F3559 (372; KEGG proteasome), L9 F1468 (382; GO BP proton transmembrane transport).
- Features with at least one significant enrichment (of 4,928): L0 1,979 of 2,497 alive; L3 3,463; L6 3,202; L9 3,146.

**Edges by layer pair** (`edges_per_layer_pair.csv`):

| Source → target | Edges | Share inhibitory |
|---|---|---|
| L0 → L1 | 37,382 | 0.581 |
| L0 → L2 | 34,741 | 0.594 |
| L0 → L3 | 33,911 | 0.584 |
| L0 → L4 | 24,369 | 0.583 |
| L0 → L5 | 22,319 | 0.594 |
| L0 → L6 | 23,143 | 0.574 |
| L0 → L7 | 20,697 | 0.564 |
| L0 → L8 | 19,963 | 0.556 |
| L0 → L9 | 19,353 | 0.541 |
| L0 → L10 | 18,544 | 0.521 |
| L0 → L11 | 17,413 | 0.568 |
| L3 → L4 | 2,360 | 0.728 |
| L3 → L5 | 2,207 | 0.689 |
| L3 → L6 | 2,447 | 0.518 |
| L3 → L7 | 2,253 | 0.543 |
| L3 → L8 | 2,149 | 0.560 |
| L3 → L9 | 2,137 | 0.566 |
| L3 → L10 | 2,287 | 0.531 |
| L3 → L11 | 2,438 | 0.541 |
| L6 → L7 | 2,598 | 0.639 |
| L6 → L8 | 3,134 | 0.682 |
| L6 → L9 | 3,695 | 0.684 |
| L6 → L10 | 4,302 | 0.718 |
| L6 → L11 | 4,863 | 0.531 |
| L9 → L10 | 4,138 | 0.683 |
| L9 → L11 | 5,544 | 0.582 |
| **Total** | **318,387** | **0.579** |

- By source layer: L0 271,835 edges (57.3% inhibitory); L3 18,278 (58.4%); L6 18,592 (64.5%); L9 9,682 (62.5%). Edges at the next layer: 46,478.
- Edges per source feature: median 567, mean 2,653, max 48,035. Every source feature has at least one edge. Median per layer: L0 7,947; L3 560; L6 553; L9 281.
- |d| of edges at the 10th / 50th / 90th / 99th percentile: 0.52 / 0.66 / 1.53 / 3.50. 18.4% of edges have |d| > 1.
- The feature with the most edges is L0 F1883 (48,035 edges, 15% of all edges). In the 500 SAE training cells it fires at 100% of `<bos>`/`<eos>` positions and at 1.5% of gene positions. It is the only source feature that fires on most special-token positions.
- The source features are active in almost every cell: only 202 of 24,000 (cell, feature) cases have no active position.
- A symmetric consistency rule (max(#Δz > 0, #Δz < 0) / n > 0.7) gives 316,461 edges. Only 1,926 edges (0.6%) depend on how Δz = 0 is counted.

**CRISPRi records.** All 318,387 edges have top-gene lists at both ends. Result: 796,304 records, 248 silenced genes (31–570 cells each; median 110; 31,539 cells in total), 6,328 target genes. 77,976 records (9.8%) are ties and are predicted "increase".

**Main table** (`crispri/results.json`; interval: percentile grouped bootstrap by silenced gene, n = 248, 2,000 resamples):

| Quantity | Value |
|---|---|
| Observed decrease / increase | 0.5248 / 0.4752 |
| Predicted decrease / increase | 0.5211 / 0.4789 |
| Confusion (pred dec & obs dec, pred inc & obs dec, pred dec & obs inc, pred inc & obs inc) | 218,513 / 199,396 / 196,450 / 181,945 |
| **Accuracy** | **0.5029** [0.4987, 0.5070] |
| Always "decrease" (majority class) | 0.5248 |
| Always "increase" | 0.4752 |
| Recall on decreases / on increases | 0.523 / 0.481 |
| **Balanced accuracy** | **0.5019** |
| **MCC** | **+0.0037** |
| Cohen's kappa | +0.0037 |
| Accuracy expected from the two class rates | 0.5010 |
| Majority class learned on other folds of silenced genes | 0.5248 |
| **Target-direction baseline** (no model) | **0.5854** (balanced 0.582, MCC 0.165) |
| Majority within the same silenced gene, other target folds (uses that knockdown's own labels; reference only) | 0.5719 |

**Intervals under three groupings** (95% percentile grouped bootstrap, 2,000 resamples):

| Statistic | Point | By silenced gene (n = 248) | By feature group (n = 150) | By component (n = 42) |
|---|---|---|---|---|
| Accuracy − best constant | −0.0219 | [−0.0369, −0.0062] | [−0.0447, −0.0001] | [−0.0387, +0.0063] |
| Balanced accuracy − 0.5 | +0.0019 | [−0.0018, +0.0057] | [−0.0026, +0.0068] | [−0.0006, +0.0130] |
| MCC | +0.0037 | [−0.0037, +0.0115] | [−0.0052, +0.0138] | [−0.0013, +0.0261] |
| Accuracy − chance from class rates | +0.0018 | [−0.0018, +0.0057] | – | – |
| Accuracy − target-direction baseline | −0.0826 | [−0.0918, −0.0730] | – | – |

- Balanced accuracy and MCC are at chance under every grouping.
- Raw accuracy is above 50% only because both the circuit and the data lean a little towards "decrease".
- The circuit is below always-decrease (the interval is below 0 by silenced gene and by feature group; by component it includes 0).
- Fold by fold (5 folds of silenced genes, seed 42): circuit 0.495–0.511; learned majority 0.516–0.536; target-direction baseline 0.566–0.594. The circuit is below the baseline in every fold.

**Per silenced gene** (mean over the 248 genes; interval: percentile bootstrap over silenced genes, 2,000 resamples):

| Quantity | Value |
|---|---|
| Mean accuracy | 0.5029 [0.4983, 0.5070] |
| Mean accuracy − always decrease | −0.0167 [−0.0283, −0.0049] |
| Mean balanced accuracy − 0.5 | +0.0010 [−0.0009, +0.0029] |
| Mean MCC | +0.0020 [−0.0020, +0.0061] |
| Mean (median) predicted-decrease share | 0.535 (0.512) |
| Genes where the circuit beats always-decrease | 81 |
| Genes with balanced accuracy > 0.5 / MCC > 0 | 126 / 126 (2 genes get one sign for all targets) |
| Genes where the circuit beats the target-direction baseline | 19 |

**Sensitivity analyses** (same edges):

| Variant | Accuracy | Balanced accuracy | MCC |
|---|---|---|---|
| Ties predicted as "decrease" | 0.5058 | 0.4999 | −0.0001 |
| Only edges whose two ends carry an enrichment label (217,212 edges, 713,995 records) | 0.5029 | 0.5021 | +0.0042 |
| Sign of the summed d instead of the majority sign | 0.5043 | 0.5007 | +0.0013 |
| Labels from the file's own full-library total (single log; agree with the main labels in 99.3% of records, r = 0.9995) | 0.5035 | 0.5023 | +0.0046 [−0.0024, +0.0119] (by silenced gene) |

With the file-total labels, always-decrease gives 0.5292 and the target-direction baseline 0.5866.

**By size of the observed change (exploratory).** The bands were planned. The Bonferroni and component intervals were added after looking at these numbers. Intervals: percentile grouped bootstrap by silenced gene (n = 248, 2,000 resamples); Bonferroni = adjusted for 6 bands; component = grouped by the 42 components.

| \|LFC\| band | Records | Observed decrease | Accuracy | Always-decrease | Balanced − 0.5 [95% CI] | Bonferroni CI | Component CI | Target baseline (accuracy / balanced) |
|---|---|---|---|---|---|---|---|---|
| < 0.01 | 144,824 | 0.506 | 0.497 | 0.506 | −0.0031 [−0.0056, −0.0006] | [−0.0064, +0.0002] | [−0.0045, +0.0032] | 0.513 / 0.513 |
| 0.01–0.025 | 190,600 | 0.519 | 0.499 | 0.519 | −0.0013 [−0.0047, +0.0019] | [−0.0059, +0.0033] | [−0.0044, +0.0020] | 0.542 / 0.540 |
| 0.025–0.05 | 210,107 | 0.528 | 0.501 | 0.528 | −0.0003 [−0.0041, +0.0035] | [−0.0054, +0.0049] | [−0.0027, +0.0095] | 0.587 / 0.583 |
| 0.05–0.1 | 165,582 | 0.533 | 0.505 | 0.533 | +0.0034 [−0.0028, +0.0094] | [−0.0048, +0.0118] | [−0.0010, +0.0210] | 0.638 / 0.631 |
| 0.1–0.25 | 75,098 | 0.537 | 0.520 | 0.537 | **+0.0181** [+0.0063, +0.0297] | **[+0.0019, +0.0351]** | [+0.0044, +0.0456] | 0.696 / 0.684 |
| ≥ 0.25 | 10,093 (179 genes) | 0.613 | 0.534 | 0.613 | +0.0284 [+0.0032, +0.0537] | [−0.0053, +0.0655] | [−0.0106, +0.0547] | 0.727 / 0.700 |

- The 0.1–0.25 band (9.4% of records) shows a small positive result: balanced accuracy 51.8%, MCC +0.036. It survives the Bonferroni and component intervals. The ≥ 0.25 band survives neither.
- This is weak. In the same records the circuit is still below always-decrease (0.520 vs 0.537), and the target-direction baseline reaches 68.4% balanced accuracy, 17 points more.
- Treat it as a hint for a later planned test, not as a finding.

### Verification

No separate verification by a second agent is recorded for this analysis. The report itself describes these second-way checks (`outputs/v3_circuit/verify/verify.json`, script `scripts/v3_circuit_verify.py`; all pass):

| Number | First way | Second way | Agreement |
|---|---|---|---|
| Edge set | running mean and variance over cells | two-pass numpy in two halves | identical 318,387 edges, all signs; largest \|Δd\| 4.7 × 10⁻⁷ |
| Records | vectorised bincount | pandas merge and groupby | identical 796,304 records (evidence, n_inhibitory, prediction, LFC) |
| Accuracy, balanced accuracy, MCC | confusion-count formulas | scikit-learn | equal to machine precision |
| Target-direction baseline | pandas groupby | plain loop | 0.585446 both |
| Interval, balanced accuracy − 0.5 (by gene) | [−0.0018, +0.0057] (seed 42) | own loop, seed 7: [−0.0019, +0.0057] | within Monte Carlo error |
| Interval, MCC (by gene) | [−0.0037, +0.0115] | [−0.0039, +0.0114] | within Monte Carlo error |
| LFC, 5 silenced genes | main script | separate code | largest difference 1.1 × 10⁻¹⁴; same cell counts |

The spot check of 3 edges with fully separate code (Checks) covers all 200 cells.

### Limits

- **No null for edge counts.** Random source features or random directions of matched size were not run. So "318,387 edges" is a count under the edge rule, not evidence that the edges mean something.
- **The selection rule was not tested.** It picks features by enrichment strength, and annotation rates of this kind are close to chance in this project. Other rules would give other source features.
- **One SAE seed**, one cell type (K562) and four source layers.
- **Design choices of the deployed analysis were kept**: edits at every position including `<bos>`/`<eos>` (L0 F1883 gives 15% of all edges); no correction for about 3.8 million (feature, target) tests; a consistency rule that counts Δz = 0 against "positive" (only 0.6% of edges depend on it).
- **CRISPRi limits.** No significance filter on the LFC; K562 only; the target-direction baseline is not refitted inside resamples; records share genes, which the grouped bootstraps handle only in part. The band analysis was not planned.
- **Downstream uses of the edge list** (pointwise mutual information, knowledge graph, disease mapping, edge-density audit) were not run.
- Cells 0–18 and the source-feature selection were run in a separate session with the same scripts (hashes recorded per chunk). They are covered by the per-cell file checks, the spot check (all 200 cells) and the second-way edge computation, but those 19 cells were not traced a second time.
- Other jobs shared the machine, so per-cell timings (38–100 s, mean 57 s) are not clean benchmarks. The exact zero-edit result and the spot-check agreement show that results were not affected.

### Files

- Scripts (`projects/maxtoki/runs/circuit-tracing-217M/scripts/`): `v3_circuit_annotate.py` (catalogs, enrichments, source-feature selection, checks), `v3_circuit_trace.py` (trace; resumable; `--check-partial`), `v3_circuit_aggregate.py` (per-cell vectors to edges), `v3_circuit_spotcheck.py`, `v3_circuit_lfc.py`, `v3_circuit_crispri.py` (records, metrics, baselines, bootstraps; metric code from `v2_circuit_crispri.py`; record rule from `audit_a2_groupkfold_crispri.py`), `v3_circuit_bands.py`, `v3_circuit_verify.py`, `v3_circuit_finalize.py`. Shared code: `projects/maxtoki/setup/inputs_v3.py`, `projects/maxtoki/setup/hooks_v2.py`.
- Outputs (`projects/maxtoki/runs/circuit-tracing-217M/outputs/v3_circuit/`):
  - edges: `circuit_edges_v3.csv` (src_layer, src_feature, tgt_layer, tgt_feature, cohens_d, consistency, sign, mean_delta, n_pos, n_neg, n_zero, consistency_sym, passes_symmetric_rule), `edge_stats_main.npz`, `edges_per_layer_pair.csv`, `edges_per_source_feature.csv`, `aggregate_summary.json`;
  - source features: `source_features.json`, `annotation/` (catalogs, enrichments, `selection_detail.csv`, `checks.json`, `summary.json`, `run_config.json`);
  - trace: `cells_tokens.npz`, `cells_encoding.json`, `trace_cells/` (200 files), `trace_progress.json` (wall time, memory and encoding check per chunk), `combos_main.csv`, `combos_ctrl.csv`, `checks/partial_forward_check.json`, `spotcheck/`;
  - labels: `lfc/` (`lfc_panel.npy` main labels, `lfc_file.npy` file-total labels, `sources.csv` with cell counts and on-target LFC, `control_rows.npy`, `checks.json`, `run_config.json`);
  - CRISPRi: `crispri/` (`results.json`, `per_source_metrics_v3_edges.csv`, `bands_followup.json`, `run_config.json`, `run_config_bands.json`), `per_pair.parquet` (796,304 records);
  - checks and provenance: `verify/verify.json`, `verify/run_config.json`, `run_config.json` (device, versions, seeds, the 200 dataset rows and token counts, the 120 source features, SAE and code sha256, encoding-check result per chunk, wall time per chunk).
- Run time: 27 chunks of ≤ 8 min, 3.23 h in total (MPS).
