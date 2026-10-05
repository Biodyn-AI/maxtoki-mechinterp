# V3 circuit report — circuit tracing and the CRISPRi direction test with v3 SAEs and correctly encoded inputs (item V3-3)

Date: 2026-10-02. Model: MaxToki-217M (HF safetensors, float32, Apple MPS). torch 2.11.0, transformers 5.5.4.
SAEs: the v3 SAEs (`runs/sae-atlas-217M/outputs/v3_sae/`, retrained on correctly encoded inputs; `V3_SAE_REPORT.md`).
Inputs: `setup/inputs_v3.py` (sha256 `019e3834…`, unchanged). Edits: `setup/hooks_v2.py` (sha256 `1ba6fda5…`, unchanged).
Every number below comes from a file listed in section 11. CI = 95% percentile interval from a grouped bootstrap
(2,000 resamples, seed 42) unless stated otherwise. "pts" = percentage points.

## 0. Results first

**Edges (same 200 cells, same selection rule, same edge rule as the deployed and v2 runs)**

| | Deployed run (wrong inputs, block-deleting hook) | v2 (wrong inputs, fixed hooks) | **v3 (right inputs, v3 SAEs, fixed hooks)** |
|---|---:|---:|---:|
| Total edges | 2,144,011 | 272,905 | **318,387** |
| Share "inhibitory" (removing the source lowers the target) | 88.2% | 58.3% | **57.9%** |
| Edges at the next layer (s → s+1), 4 source layers | 0 | 37,952 | **46,478** |
| Median \|d\| of an edge | 1.30 | 0.65 | **0.66** |
| Source features with no edge (of 120) | – | 18 | **0** |

- The edge map looks much like v2. The share of inhibitory edges is the same (58%). There are 17% more edges.
- The big change is at the deep source layers. Layer 6 sources now give 18,592 edges (v2: 2,070) and layer 9
  sources 9,682 (v2: 453). Every source feature now has at least one edge.
- Layer-0 sources are mostly the same features as before (23 of 30 have an old decoder direction with |cos| ≥ 0.9).
  Layers 3, 6 and 9 are new features (no source feature has an old match ≥ 0.9).

**CRISPRi direction test (positive class = target gene goes down after silencing), identical records**

| | v2 report (v2 edges, double-log labels) | **v3 (v3 edges, single-log labels)** |
|---|---:|---:|
| Records (silenced gene × target gene) | 698,624 | **796,304** |
| Silenced genes | 227 | **248** |
| Observed decrease | 54.31% | **52.48%** |
| Predicted decrease | 50.89% | **52.11%** |
| Accuracy [CI, silenced genes] | 49.76% | **50.29% [49.87, 50.70]** |
| Always "decrease" (= majority class, = best constant) | 54.31% | **52.48%** |
| Always "increase" | 45.69% | **47.52%** |
| Accuracy minus best constant [CI, silenced genes] | −4.55 pts [−5.79, −3.34] | **−2.19 pts [−3.69, −0.62]** |
| Balanced accuracy | 49.68% | **50.19%** |
| Balanced accuracy minus 0.5 [CI, silenced genes] | −0.32 pts [−0.83, +0.20] | **+0.19 pts [−0.18, +0.57]** |
| MCC [CI, silenced genes] | −0.0063 [−0.0166, +0.0041] | **+0.0037 [−0.0037, +0.0115]** |
| Accuracy expected from the two class rates alone | 50.08% | **50.10%** |
| Target-direction baseline (cross-fitted, no model) | 58.70% | **58.54%** |
| Accuracy minus target-direction baseline [CI, silenced genes] | −8.94 pts [−9.98, −7.99] | **−8.26 pts [−9.18, −7.30]** |

- **The verdict does not change: the circuit has no detectable skill at predicting which way a gene moves.**
  Balanced accuracy is 50.2% and MCC is +0.004; both CIs include chance under all three groupings (section 7).
- The circuit is still worse than always saying "decrease" (by 2.2 pts) and much worse than a rule that uses no
  model at all: "does this target usually go down when other genes are silenced?" (58.5%, 8.3 pts better).
- Fixing the labels alone does not change this. Re-scoring the v2 edges with the corrected single-log labels gives
  balanced accuracy 49.5% and MCC −0.009 (section 7.5).
- One exploratory hint: in pairs with a mid-sized change (|LFC| 0.1–0.25, 9.4% of pairs) balanced accuracy is
  51.8%, and its CI stays above 0.5 after a Bonferroni correction for 6 bands. This was not planned. It is far below
  the no-model baseline in the same pairs (68.4%), and it is absent when v2 edges are scored with the same labels
  (section 7.4).

## 1. What I ran, and what stayed the same as v2

| Setting | v2 (`v2_circuit_trace.py`) | v3 (`v3_circuit_trace.py`) |
|---|---|---|
| Cells | 200 K562 non-targeting cells, `default_rng(42).choice`, sorted (deployed `circuit_trace.py:100-123`) | **same 200 dataset rows** (asserted against the v2 file) |
| Input encoding | `tokenize_cell(X)` with X = log1p(CP10k) (**wrong**) | counts n = round(expm1(X) / u), u = smallest non-zero expm1(X) of the cell; ranked by n / Σn × 10⁴ / Geneformer gene median; first 2,046 genes; `<bos>`/`<eos>` (**right**, `inputs_v3`) |
| SAEs | deployed (`sae-atlas-217M/outputs/phase1`) | **v3** (`sae-atlas-217M/outputs/v3_sae`) |
| Source layers | 0, 3, 6, 9 | same |
| Source features | deployed IDs (30 per layer) | **deployed rule re-applied to the v3 SAEs** (section 2) |
| Edit | `hooks_v2.Ablate(s, [f])`: delta = −z_f(x) · W_dec[:, f] added to the input of block s (= `hidden_states[s]`), every position incl. `<bos>`/`<eos>` | same |
| Read-out | hooks_v2 captures of the input of site t = s+1..11; mean over positions of z_ablated − z_clean | same |
| Cohen's d, consistency, edge rule, sign | d = mean / sqrt(var(ddof=1) + 1e-12) over 200 cells; consistency = max(#Δz>0, n−#Δz>0)/n; edge if \|d\| > 0.5 and consistency > 0.7; "inhibitory" if mean Δz < 0 | same (same code) |
| Exact speed-ups | partial forward from block s; inactive source feature → exact zeros | same (partial-forward check: max difference 0.0, section 4) |
| Deployed-hook controls | legacy_zero, legacy_real | dropped (they tested the old hook, which is not used) |

How different the inputs are on these 200 cells (old vs new order, before the cut): Spearman 0.840 (min 0.723),
first-200 overlap 0.562, kept-gene overlap 0.916, genes at the same position 0.09%. 195 of 200 cells are cut at
2,046 genes (tokens per cell 1,434–2,048, median 2,048). This matches the audit (K562 controls 0.84 / 0.55 / 0.92).

Run: 27 chunks of ≤ 8 min, 3.23 h wall time in total, 38–100 s per cell (mean 57 s; the machine was shared with
other jobs, so these are not clean timings). Free memory at chunk start ≥ 5.67 GB, lowest during a chunk 4.45 GB;
MPS driver memory at the end of a chunk ≤ 3.63 GB. The first two chunks (cells 0–18) ran in an earlier session of
this same item with the same script (sha256 `28a1bf3b…`, recorded per chunk).

## 2. Source features: the deployed rule on the v3 SAEs

**The rule** (deployed `circuit_trace.py:133-145` and `sae-atlas-217M/scripts/full_12layer_pipeline.py:81-98, 194-264`),
re-implemented in `scripts/v3_circuit_annotate.py`:

1. **Catalog.** For each SAE feature and each gene: the mean activation over the token positions where the feature is
   active, over all 1,019,996 positions of the 500 SAE training cells (v3 tokens, incl. `<bos>`/`<eos>` as
   "<SPECIAL>"). Top-20 genes = the 20 genes with the highest mean.
2. **Enrichment.** Terms from GO BP, KEGG, Reactome and TRRUST TF target sets, each cut to the K562 panel and kept if
   ≥ 5 genes (2,377 terms). For each feature and term with ≥ 2 shared genes: hypergeometric p (N = 6,329 distinct
   names, K = term size, n = list size). Rows with p < 0.1 are BH-corrected together per layer; significant = q < 0.05.
3. **Selection.** Score = sum of −log10 p over the feature's significant rows. Take the 30 features with the largest
   score at each of layers 0, 3, 6 and 9.

**Checks (all pass, `outputs/v3_circuit/annotation/checks.json`).**
- The enrichment code fed the deployed catalogs rebuilds all 12 deployed `significant_enrichments.csv` files: same
  rows in the same order, same overlap / K / n / N, p within 8.5 × 10⁻¹³ (relative).
- The selection code fed the deployed enrichments rebuilds the 120 deployed / v2 source features exactly.
- The v3 layer-5 catalog equals the one built independently by item V3-2 (4,928 of 4,928 ordered lists identical).
- No tie at the cut-off: score of the 30th vs 31st feature 138.41 vs 137.70 (L0), 113.06 vs 112.98 (L3),
  90.58 vs 89.25 (L6), 87.69 vs 86.60 (L9).

Annotated features at the source layers (v3 / deployed): L0 1,979 of 2,497 alive / 2,016 of 2,498; L3 3,463 / 3,364;
L6 3,202 / 3,371; L9 3,146 / 3,442 (of 4,928).

**The 120 selected features** (by rank; `outputs/v3_circuit/source_features.json`, details with score, best term and
top-10 genes in `annotation/selection_detail.csv`):
- L0: 3635, 3083, 1495, 4594, 3421, 73, 199, 948, 1213, 1216, 1116, 1715, 3605, 4802, 734, 603, 1762, 968, 136, 2591,
  1990, 3810, 3582, 4714, 4321, 1883, 661, 1087, 2785, 3765
- L3: 3050, 1370, 1335, 1952, 2982, 3262, 4036, 150, 3662, 4912, 3708, 2737, 1949, 3020, 4029, 693, 3043, 4361, 1863,
  2050, 3295, 2256, 2217, 3150, 778, 1891, 4162, 1885, 3788, 4801
- L6: 3559, 271, 2942, 4456, 3396, 925, 4212, 4255, 1371, 2337, 4818, 3937, 3421, 1422, 4315, 1508, 3798, 2720, 3018,
  2123, 840, 3397, 991, 4029, 262, 1645, 922, 3654, 376, 2700
- L9: 1468, 1107, 1873, 2007, 4645, 1826, 271, 630, 1294, 3199, 1879, 3683, 1652, 4255, 1089, 3493, 198, 3971, 2551,
  2421, 1386, 4880, 1863, 2753, 1137, 2680, 4170, 813, 3664, 56

Examples (rank 1 per layer): L0 F3635 (score 481; best term Reactome G1/S transition; ST13, FKBP4, STIP1, TCP1 …),
L3 F3050 (352; KEGG proteasome), L6 F3559 (372; KEGG proteasome), L9 F1468 (382; GO BP proton transmembrane transport).

**Are these the old features?** (`outputs/v3_circuit/source_feature_match.csv`; best |cos| between the v3 decoder
direction and any deployed decoder direction at the same layer)

| Layer | Median best \|cos\| | Best match ≥ 0.9 | Same number as a v2 source feature |
|---|---:|---:|---:|
| 0 | 0.92 | 23 of 30 | 16 of 30 (their own old direction: \|cos\| 0.78–0.96) |
| 3 | 0.35 | 0 | 2 (\|cos\| ≤ 0.34, chance) |
| 6 | 0.28 | 0 | 0 |
| 9 | 0.34 | 0 | 1 |

Layer 0 is the token embedding, so its SAE barely changed (`V3_SAE_REPORT.md` 3.3) and about half of the layer-0
sources are the same features as in v2. At layers 3, 6 and 9 all sources are new features.

Caveat: the rule picks features by how strongly their top genes hit gene-set databases. Earlier work found that this
kind of annotation rate is close to chance (`sae-atlas-217M/V2B_ANNOTATION_FDR_REPORT.md`). I applied the rule because
the task asks for the deployed rule. I did not test other rules.

## 3. Input encoding checks

- At load time all 200 cells passed `inputs_v3.check_encoding` (token order = counts / median order; counts are
  whole-number multiples of the cell's smallest value, largest deviation 3.7 × 10⁻⁴).
- Before every chunk, the next ≤ 12 cells were checked again with `assert_encoding_batch` before any forward pass:
  27 of 27 chunks pass (317 cell checks, 0 order violations; a few exact ties placed in either order, which the check
  allows). The spot check checked its cells the same way (5 of 5 chunks, 200 cells).
- The old tokens of the same rows (`deployed_tokenize_X`) rebuild the saved v2 tokens exactly, so v3 and v2 are paired
  cell by cell.

## 4. Sanity checks (all pass)

| Check | Result | Pass |
|---|---|---|
| Zero edit (`hooks_v2.ZeroDelta(s)`, same partial forward, 4 source layers × 200 cells) | max \|Δz\| = 0.0 in every cell; 0 edges | yes |
| Edges at s → s+1 | L0→L1 37,382; L3→L4 2,360; L6→L7 2,598; L9→L10 4,138 | yes |
| Partial forward vs full forward (2 cells × 8 features) | max difference 0.0 in hidden states and in Δz | yes |
| 3 edges recomputed from scratch (`v3_circuit_spotcheck.py`) | see below | yes |
| Edges recomputed a second way (two-pass numpy, no running mean) | same 318,387 edges, all signs equal, max \|Δd\| 4.7 × 10⁻⁷ | yes |

**Spot check.** Separate code: cells re-drawn from the h5ad, counts rebuilt with its own code (round(expm1(X) / u)),
tokenised with `tokenize_cell`; edit = a plain forward hook that ADDS decode(z with f zeroed) − decode(z) (CPU v3 SAE) to
the output of the module that produces `hidden_states[s]`; read-out = `output_hidden_states[t]`. Same rows and same
tokens as the trace: yes. Edges drawn with seed 20261001 from L0→L1, L9→L10 and L6→L11 (all three pairs had edges).

| Edge | d (trace) | d (from scratch) | Consistency (trace / scratch) | Max per-cell \|ΔΔz\| |
|---|---:|---:|---|---:|
| L0 F199 → L1 F580, excitatory | 0.6294 | 0.62937 | 0.95 / 0.95 | 4.4 × 10⁻⁹ |
| L9 F2421 → L10 F2835, inhibitory | −0.6047 | −0.60471 | 0.985 / 0.98 | 1.3 × 10⁻⁸ |
| L6 F4315 → L11 F973, excitatory | 0.9334 | 0.93343 | 0.96 / 0.96 | 3.7 × 10⁻⁸ |

The d differences (< 5 × 10⁻⁵) are the 4-decimal rounding of the CSV. In the second edge one cell has a Δz within
1.3 × 10⁻⁸ of zero, and its sign differs between MPS and CPU; that moves consistency from 0.985 to 0.98. The
recomputation covers all 200 cells, including cells 0–18 from the earlier session.

## 5. Edges: v3 vs v2 vs deployed

| Source → target | v3 edges | v3 share inhibitory | v2 edges | v2 share inhibitory | Deployed edges | Deployed share inhibitory |
|---|---:|---:|---:|---:|---:|---:|
| L0 → L1 | 37,382 | 0.581 | 34,539 | 0.579 | 0 | – |
| L0 → L2 | 34,741 | 0.594 | 31,578 | 0.592 | 146,332 | 0.980 |
| L0 → L3 | 33,911 | 0.584 | 26,811 | 0.596 | 142,019 | 0.984 |
| L0 → L4 | 24,369 | 0.583 | 15,439 | 0.642 | 139,162 | 0.985 |
| L0 → L5 | 22,319 | 0.594 | 16,858 | 0.611 | 132,216 | 0.990 |
| L0 → L6 | 23,143 | 0.574 | 21,746 | 0.575 | 118,418 | 0.986 |
| L0 → L7 | 20,697 | 0.564 | 23,693 | 0.560 | 113,720 | 0.987 |
| L0 → L8 | 19,963 | 0.556 | 25,480 | 0.550 | 93,845 | 0.987 |
| L0 → L9 | 19,353 | 0.541 | 23,418 | 0.555 | 98,722 | 0.989 |
| L0 → L10 | 18,544 | 0.521 | 14,162 | 0.580 | 108,583 | 0.986 |
| L0 → L11 | 17,413 | 0.568 | 10,301 | 0.591 | 96,577 | 0.990 |
| L3 → L4 | 2,360 | 0.728 | 2,349 | 0.816 | 0 | – |
| L3 → L5 | 2,207 | 0.689 | 3,469 | 0.628 | 103,673 | 0.633 |
| L3 → L6 | 2,447 | 0.518 | 3,694 | 0.576 | 95,596 | 0.802 |
| L3 → L7 | 2,253 | 0.543 | 4,962 | 0.541 | 90,816 | 0.827 |
| L3 → L8 | 2,149 | 0.560 | 5,373 | 0.524 | 75,283 | 0.790 |
| L3 → L9 | 2,137 | 0.566 | 4,023 | 0.533 | 82,210 | 0.825 |
| L3 → L10 | 2,287 | 0.531 | 1,686 | 0.536 | 91,162 | 0.892 |
| L3 → L11 | 2,438 | 0.541 | 801 | 0.519 | 81,305 | 0.909 |
| L6 → L7 | 2,598 | 0.639 | 774 | 0.689 | 0 | – |
| L6 → L8 | 3,134 | 0.682 | 594 | 0.704 | 71,274 | 0.548 |
| L6 → L9 | 3,695 | 0.684 | 377 | 0.761 | 71,116 | 0.521 |
| L6 → L10 | 4,302 | 0.718 | 211 | 0.777 | 62,448 | 0.636 |
| L6 → L11 | 4,863 | 0.531 | 114 | 0.798 | 50,591 | 0.689 |
| L9 → L10 | 4,138 | 0.683 | 290 | 0.845 | 0 | – |
| L9 → L11 | 5,544 | 0.582 | 163 | 0.859 | 78,943 | 0.849 |
| **Total** | **318,387** | **0.579** | **272,905** | **0.583** | **2,144,011** | **0.882** |

By source layer (v3 / v2): L0 271,835 / 244,025 (inhibitory 57.3% / 58.2%); L3 18,278 / 26,357 (58.4% / 57.6%);
L6 18,592 / 2,070 (64.5% / 72.1%); L9 9,682 / 453 (62.5% / 85.0%).

Other facts:
- Edges per source feature: median 567, mean 2,653, max 48,035; no feature without an edge (v2: 18). Median per
  layer: L0 7,947; L3 560; L6 553; L9 281 (v2: 6,593; 479; 45; 6).
- \|d\| of v3 edges at the 10th / 50th / 90th / 99th percentile: 0.52 / 0.66 / 1.53 / 3.50 (v2: 0.52 / 0.65 / 1.43 /
  3.52). 18.4% of edges have \|d\| > 1 (v2: 16.4%).
- The feature with the most edges is again L0 F1883 (48,035 edges; 15% of all v3 edges; v2: 43,356). In the 500 SAE
  training cells it fires at 100% of `<bos>`/`<eos>` positions and at 1.5% of gene positions. It is the only source
  feature that fires on most special-token positions. The deployed design edits every position, so I kept that.
- The v3 source features are active in almost every cell: only 202 of 24,000 (cell, feature) cases had no active
  position (v2: 2,682).
- Symmetric consistency rule (max(#Δz>0, #Δz<0)/n > 0.7): 316,461 edges; only 1,926 v3 edges (0.6%) depend on how
  Δz = 0 is counted.
- v3 and v2 edges cannot be matched one by one, because the SAE features are different (section 2).

## 6. Knockdown log-fold-changes, single log

**Formula** (`scripts/v3_circuit_lfc.py`):
- n_ig = round(expm1(X_ig) / u_i), with u_i = the smallest non-zero expm1(X) in cell i. Every cell was checked to be
  whole-number multiples of u_i (largest deviation 4.7 × 10⁻⁴; 3 of 50,729 knockdown cells already had u_i = 1).
  This gives integer counts up to one factor per cell.
- v_ig = log1p(10,000 · n_ig / Σ_g' n_ig'), summing over the 6,546 genes in the file. **This is the one and only log.**
- LFC(s, g) = mean of v_ig over the K562 cells whose guide targets s − mean of v_ig over the control cells.
- Control cells: 3,000 drawn with `default_rng(42)` from the 10,691 K562 non-targeting cells (the same rows as v2).
- Knockdown cells: all K562 cells of the guide; a silenced gene is used only with ≥ 10 cells. Of 1,543 candidate genes
  (top-10 genes of the v3 and the v2 source features), 394 have ≥ 10 cells. **The 248 silenced genes in the v3 records
  have 31–570 cells each (median 110; 31,539 in total).**
- Observed decrease = LFC < 0. No LFC in the records is exactly 0.

**Checks** (`outputs/v3_circuit/lfc/checks.json`, `verify/verify.json`):
- The old double-log value, recomputed from the same arrays with the v2 code, equals the v2 LFC exactly (max
  difference 0.0) for all 248 v2 sources, with the same cell counts and the same control rows. So the cells are the same.
- Without rounding (expm1(X) used directly) the single-log LFC changes by at most 7.6 × 10⁻⁸.
- A second implementation (cells found with `dataset_loader`, controls redrawn, own count code) gives the same LFC for
  5 random silenced genes (max difference 1.1 × 10⁻¹⁴) and the same cell counts.
- On-target knockdown: all 248 silenced genes have their own LFC < 0 (median −0.67; 69% below −0.5).

**How much the labels moved.** Over all 394 × 6,546 values, the new and old LFC agree in sign for 92.8% (Pearson
r = 0.915). On the v3 records the agreement is 92.6% (r = 0.903). So about 7% of the labels flip. A single log with the
file's own full-library total (stored X) agrees with the primary label in 99.3% of records (r = 0.9995).

## 7. CRISPRi direction test on identical records

Record rule (the method's rule, `audit_a2_groupkfold_crispri.py:51-162`; the v2 code is imported unchanged, sha256
`e7658d82…`): pair each top-10 gene of an edge's source feature with each top-10 gene of its target feature (no self
pairs); per gene pair, evidence = number of such triples, n_inhibitory, max |d|; keep if evidence ≥ 2 or max |d| > 2;
predicted decrease = n_inhibitory / evidence > 0.5 (a tie counts as "increase"); keep if the source gene has a K562
guide with ≥ 10 cells and the target gene is measured. Top-10 lists come from the v3 catalogs (section 2).

All 318,387 edges have top-gene lists at both ends. Result: **796,304 records, 248 silenced genes, 6,328 target genes.**
77,976 records (9.8%) are ties.

### 7.1 Main table (v3 edges, single-log labels)

| Quantity | Value |
|---|---|
| Observed decrease / increase | 0.5248 / 0.4752 |
| Predicted decrease / increase | 0.5211 / 0.4789 |
| Confusion (pred dec & obs dec, pred inc & obs dec, pred dec & obs inc, pred inc & obs inc) | 218,513 / 199,396 / 196,450 / 181,945 |
| **Accuracy** | **0.5029** [0.4987, 0.5070] |
| Always decrease (= majority class) | 0.5248 |
| Always increase | 0.4752 |
| Recall on decreases / on increases | 0.523 / 0.481 |
| **Balanced accuracy** | **0.5019** |
| **MCC** | **+0.0037** |
| Cohen's kappa | +0.0037 |
| Accuracy expected from the two class rates | 0.5010 |
| Global majority learned on other source folds | 0.5248 |
| **Target-direction baseline** (target's usual direction in the other 4 folds of silenced genes) | **0.5854** (balanced 0.582, MCC 0.165) |
| Within-source majority, other target folds (uses the same knockdown's labels; reference only) | 0.5719 |

**Grouped bootstrap CIs.** Three ways to group: (a) silenced genes (248); (b) source-feature groups: silenced genes
that sit in the same set of source features (with ≥ 1 edge) — 150 groups; (c) connected components of silenced genes
that share any source feature — 42 (largest 140 genes). The best constant is re-chosen in each resample. The
target-direction baseline is cross-fitted once on 5 fixed folds and not refit inside resamples.

| Statistic | Point | CI by silenced gene | CI by feature group | CI by component |
|---|---:|---|---|---|
| Accuracy − best constant | −0.0219 | [−0.0369, −0.0062] | [−0.0447, −0.0001] | [−0.0387, +0.0063] |
| Balanced accuracy − 0.5 | +0.0019 | [−0.0018, +0.0057] | [−0.0026, +0.0068] | [−0.0006, +0.0130] |
| MCC | +0.0037 | [−0.0037, +0.0115] | [−0.0052, +0.0138] | [−0.0013, +0.0261] |
| Accuracy − chance from class rates | +0.0018 | [−0.0018, +0.0057] | – | – |
| Accuracy − target-direction baseline | −0.0826 | [−0.0918, −0.0730] | – | – |

What this means: balanced accuracy and MCC are at chance in every grouping. Raw accuracy is above 50% only because
both the circuit and the data lean a little towards "decrease". The circuit is below always-decrease (the CI is below
0 by silenced gene and by feature group; by component it includes 0) and 8.3 points below the no-model baseline.

Fold by fold (5 folds of silenced genes, seed 42): circuit 0.495–0.511; global majority 0.516–0.536; target-direction
baseline 0.566–0.594. The circuit is below the baseline in every fold.

### 7.2 Per silenced gene (mean over genes; CI by silenced gene)

| Quantity | v3 (248 genes) | v2 (227 genes) |
|---|---:|---:|
| Mean accuracy | 0.5029 [0.4983, 0.5070] | 0.5127 [0.5016, 0.5247] |
| Mean accuracy − always decrease | −0.0167 [−0.0283, −0.0049] | −0.0345 [−0.0431, −0.0259] |
| Mean balanced accuracy − 0.5 | +0.0010 [−0.0009, +0.0029] | +0.0011 [−0.0019, +0.0042] |
| Mean MCC | +0.0020 [−0.0020, +0.0061] | +0.0047 [−0.0041, +0.0139] |
| Mean predicted-decrease share (median) | 0.535 (0.512) | 0.626 (0.613) |
| Genes where the circuit beats always-decrease | 81 | 63 |
| Genes with balanced accuracy > 0.5 / MCC > 0 | 126 / 126 (2 genes get one sign for all targets) | 110 / 110 |
| Genes where the circuit beats the target-direction baseline | 19 | 29 |

### 7.3 Sensitivity (v3 edges)

- Ties predicted as "decrease": accuracy 0.5058, balanced 0.4999, MCC −0.0001.
- Only edges whose two ends carry an enrichment label (the original Phase 11 rule): 217,212 edges, 713,995 records;
  accuracy 0.5029, balanced 0.5021, MCC +0.0042.
- Sign of the summed d instead of the majority sign: accuracy 0.5043, balanced 0.5007, MCC +0.0013.
- File labels (single log, full-library total): accuracy 0.5035, balanced 0.5023, MCC +0.0046 [−0.0024, +0.0119];
  always-decrease 0.5292; target baseline 0.5866.
- Old double-log labels on the same v3 records: accuracy 0.5022, balanced 0.5009, MCC +0.0019 [−0.0047, +0.0087];
  always-decrease 0.5292; target baseline 0.5871.

### 7.4 By size of the observed change (exploratory follow-up)

Bands of |LFC| (single-log units). CIs by silenced gene; Bonferroni = corrected for 6 bands; components = 42 groups.
Target baseline = cross-fitted target-direction baseline on the same records.

| \|LFC\| band | Records | Obs. decrease | Accuracy | Always-dec | Balanced − 0.5 [95% CI] | Bonferroni CI | Component CI | Target baseline (acc / balanced) |
|---|---:|---:|---:|---:|---|---|---|---|
| < 0.01 | 144,824 | 0.506 | 0.497 | 0.506 | −0.0031 [−0.0056, −0.0006] | [−0.0064, +0.0002] | [−0.0045, +0.0032] | 0.513 / 0.513 |
| 0.01–0.025 | 190,600 | 0.519 | 0.499 | 0.519 | −0.0013 [−0.0047, +0.0019] | [−0.0059, +0.0033] | [−0.0044, +0.0020] | 0.542 / 0.540 |
| 0.025–0.05 | 210,107 | 0.528 | 0.501 | 0.528 | −0.0003 [−0.0041, +0.0035] | [−0.0054, +0.0049] | [−0.0027, +0.0095] | 0.587 / 0.583 |
| 0.05–0.1 | 165,582 | 0.533 | 0.505 | 0.533 | +0.0034 [−0.0028, +0.0094] | [−0.0048, +0.0118] | [−0.0010, +0.0210] | 0.638 / 0.631 |
| 0.1–0.25 | 75,098 | 0.537 | 0.520 | 0.537 | **+0.0181** [+0.0063, +0.0297] | **[+0.0019, +0.0351]** | [+0.0044, +0.0456] | 0.696 / 0.684 |
| ≥ 0.25 | 10,093 (179 genes) | 0.613 | 0.534 | 0.613 | +0.0284 [+0.0032, +0.0537] | [−0.0053, +0.0655] | [−0.0106, +0.0547] | 0.727 / 0.700 |

- The 0.1–0.25 band shows a small positive (balanced accuracy 51.8%, MCC +0.036) that survives the Bonferroni and the
  component CIs. The ≥ 0.25 band does not survive either.
- This is weak. In the same pairs, the circuit is still below always-decrease (0.520 vs 0.537), and the no-model
  baseline reaches 68.4% balanced accuracy, 17 points more.
- It comes from the v3 edges, not from the label fix: v2 edges scored with the same single-log labels give +0.0009
  [−0.0128, +0.0158] in this band. v3 edges with the old double-log labels give +0.0086 [−0.0002, +0.0168].
- The bands were in the v2 design, but the Bonferroni and component CIs were added after I saw these numbers. Treat
  this as a hint for a later planned test, not as a finding.

### 7.5 What the label fix alone does (v2 edges re-scored)

The v2 records were rebuilt from the v2 edges and the deployed catalogs. With the old double-log labels they equal
`outputs/v2_circuit/per_pair.parquet` exactly (698,624 of 698,624 records, every predicted sign, LFC difference 0.0).
With the corrected single-log labels:
- 6.7% of the labels flip. Observed decrease goes from 54.31% to 54.81%.
- Accuracy 0.4962 [0.4894, 0.5025]; always-decrease 0.5481; accuracy − best constant −5.18 pts [−6.95, −3.44].
- Balanced accuracy 0.4953 (−0.47 pts [−1.05, +0.11]); MCC −0.0093 [−0.0209, +0.0021].
- Target-direction baseline 0.5916; accuracy − baseline −9.54 pts [−10.87, −8.28].

So with correct labels the v2 circuit is still at chance.

### 7.6 Overlap of the record sets

v3 and v2 records share 300,220 gene pairs and 96 silenced genes (v3 248, v2 227). On the shared pairs the two circuits
predict the same sign 60.4% of the time. The v3 silenced genes overlap the deployed set in 102 of 248 genes.

## 8. What changed compared with the deployed and the v2 results

| | Deployed | v2 | v3 |
|---|---|---|---|
| Inputs | wrong order (log1p ranked as counts) | wrong order | **right order** |
| SAEs | deployed | deployed | **v3** |
| Hooks | block-deleting bug | fixed | fixed |
| CRISPRi labels | double log | double log | **single log** |
| Edges | 2,144,011 (88.2% inhibitory) | 272,905 (58.3%) | **318,387 (57.9%)** |
| Accuracy / always-decrease | 53.48% / 54.61% | 49.76% / 54.31% | **50.29% / 52.48%** |
| Balanced accuracy / MCC | 49.83% / −0.0055 | 49.68% / −0.0063 | **50.19% / +0.0037** |
| Target-direction baseline | 59.15% | 58.70% | **58.54%** |

- **Edges.** The edge map with correct inputs and new SAEs looks like the v2 map in its overall shape: ~58% inhibitory,
  most edges from layer-0 sources, L0 F1883 (the `<bos>`/`<eos>` feature) on top. It has 17% more edges, and the deep
  source layers (6 and 9) now reach many more targets. Any claim built on the deployed 2.1 M edges stays retracted
  (V2 report). Claims built on v2 edge counts per layer should use the v3 numbers.
- **CRISPRi direction.** The verdict is the same in all three runs: no directional skill. The v3 numbers sit a little
  above 50% where v2 sat a little below, but every CI for balanced accuracy and MCC includes chance, and the circuit
  loses to both always-decrease and the no-model target baseline.
- **Labels.** The double log flipped about 7% of the direction labels. Fixing it does not change any conclusion.

## 9. Second-way checks (`outputs/v3_circuit/verify/verify.json`: all pass)

| Number | First way | Second way | Agreement |
|---|---|---|---|
| Edge set | running mean / variance over cells | two-pass numpy in two halves | identical 318,387 edges, all signs; max \|Δd\| 4.7 × 10⁻⁷ |
| Records | vectorised bincount (`v2_circuit_crispri.build_records`) | pandas merge + groupby | identical 796,304 records (evidence, n_inhibitory, prediction, LFC) |
| Same second-way record code on v2 edges | – | vs v2 `per_pair.parquet` | identical 698,624 records |
| Accuracy / balanced accuracy / MCC | confusion-count formulas | scikit-learn | equal to machine precision |
| Target-direction baseline | pandas groupby | plain loop | 0.585446 both |
| CI balanced − 0.5 (genes) | [−0.0018, +0.0057] (seed 42) | own loop, seed 7: [−0.0019, +0.0057] | within Monte-Carlo error |
| CI MCC (genes) | [−0.0037, +0.0115] | [−0.0039, +0.0114] | within Monte-Carlo error |
| LFC, 5 silenced genes | main script | separate code | max difference 1.1 × 10⁻¹⁴; same cell counts |

## 10. What I could NOT do, and limits

- **No null for edge counts.** I did not run random source features or random directions of matched size. So
  "318,387 edges" is a count under the deployed rule, not evidence that the edges mean something.
- **Downstream analyses not redone.** PMI, knowledge graph, disease mapping and the edge-density audit (A4) read the
  edge list. I did not re-run them.
- **The selection rule itself was not tested.** It picks features by enrichment strength, which earlier work found is
  near chance as an annotation measure. Different rules would give different source features.
- **One SAE seed.** Which features exist depends on the SAE training run (`V3_SAE_REPORT.md` has one seed).
- **Kept deployed design choices**, including ones I would not choose: edits at every position including
  `<bos>`/`<eos>` (L0 F1883 gives 15% of all edges); no correction for ~3.8 million (feature, target) tests; the
  consistency rule that counts Δz = 0 against "positive" (only 0.6% of edges depend on it).
- **CRISPRi limits.** No significance filter on LFC; K562 only; the target-direction baseline is not refit inside
  bootstrap resamples; records share genes, which the grouped bootstraps handle only in part. The band follow-up
  (7.4) was not planned.
- **Earlier session.** Cells 0–18 and the source-feature selection were run by an earlier session of this item with the
  same scripts (hashes recorded). I checked them through the per-cell file checks, the spot check (which covers all
  200 cells) and the second-way edge computation. I did not re-run those 19 cells.
- **One bug fixed before any output was written.** The first LFC run stopped with an error (an array name clashed
  with an argument of `np.savez`). No part file had been written. I renamed the array and re-ran from the start.
- **Shared machine.** Other jobs ran at the same time, so per-cell timings (38–100 s) are not clean benchmarks. The
  exact zero-edit result and the spot-check agreement show the results were not affected.
- Importing `v2_circuit_crispri.py` created new bytecode cache files in `scripts/__pycache__/`. No existing file
  under `runs/` or `setup/` was changed.

## 11. Files

Scripts (`runs/circuit-tracing-217M/scripts/`):
- `v3_circuit_annotate.py` — v3 catalogs, enrichments and the source-feature selection (+ checks against deployed files).
- `v3_circuit_trace.py` — the trace (resumable, `--max-minutes`; `--check-partial`).
- `v3_circuit_aggregate.py` — per-cell vectors → edges, per-layer-pair table, controls.
- `v3_circuit_spotcheck.py` — 3 edges recomputed from scratch.
- `v3_circuit_lfc.py` — single-log knockdown LFCs (+ file-total and double-log versions for comparison).
- `v3_circuit_crispri.py` — records, metrics, baselines, bootstraps (analyses A–D); metric code imported unchanged
  from `v2_circuit_crispri.py`.
- `v3_circuit_bands.py` — |LFC| band follow-up.
- `v3_circuit_verify.py` — second-way checks.
- `v3_circuit_finalize.py` — `run_config.json` files and the source-feature match table.

Outputs (`runs/circuit-tracing-217M/outputs/v3_circuit/`):
- `circuit_edges_v3.csv` (columns as v2: src_layer, src_feature, tgt_layer, tgt_feature, cohens_d, consistency, sign,
  mean_delta, n_pos, n_neg, n_zero, consistency_sym, passes_symmetric_rule), `edge_stats_main.npz`,
  `edges_per_layer_pair.csv` (v3, v2 and deployed counts), `edges_per_source_feature.csv`, `aggregate_summary.json`.
- `source_features.json`, `source_feature_match.csv`, `annotation/` (catalogs, enrichments, `selection_detail.csv`,
  `checks.json`, `summary.json`, `run_config.json`).
- `cells_tokens.npz` (rows, v3 tokens, old tokens for comparison), `cells_encoding.json`, `trace_cells/` (200 files),
  `trace_progress.json` (wall time, memory and encoding check per chunk), `combos_main.csv`, `combos_ctrl.csv`,
  `checks/partial_forward_check.json`.
- `spotcheck/` (`spotcheck_result.json`, `edges_picked.json`, `progress.json`, `run_config.json`).
- `lfc/` (`lfc_panel.npy` primary, `lfc_file.npy`, `lfc_v2style.npy`, `sources.csv` with cell counts and on-target
  LFC, `control_rows.npy`, `checks.json`, `run_config.json`).
- `crispri/` (`results.json`, `comparison_table.csv`, `per_source_metrics_v3_edges.csv`, `bands_followup.json`,
  `run_config.json`, `run_config_bands.json`).
- `per_pair.parquet` — 796,304 records (columns as the v2 file, plus `lfc_file` and `lfc_v2_double_log`).
- `verify/verify.json`, `verify/run_config.json`.
- `run_config.json` — device, versions, seeds, the 200 dataset rows and token counts, the 120 source features, SAE and
  code sha256, encoding-check result per chunk, wall time per chunk.

## Plain-words summary

The earlier circuit runs gave the model genes in the wrong order, and their SAE features came from SAEs trained on
those wrong inputs. I redid the circuit tracing with the right gene order and the retrained SAEs. I picked the 120
source features with the same rule as before, applied to the new SAEs. At layer 0 about half of them are the same
features as before; at layers 3, 6 and 9 they are all new. I used the same 200 cells and the same edge rule.

The new run finds 318,387 edges, 58% of them "inhibitory". That is close to the v2 run (272,905 edges, 58%). The main
difference is that sources in the deeper layers now reach many more targets. All checks pass: an empty edit gives no
edges, edges appear at the very next layer, three edges recomputed with separate code match, and a second way of
computing the edges gives the same list.

I also fixed the knockdown labels, which had been logged twice. About 7% of the "up or down" labels changed. Then I
rebuilt the CRISPRi test with the same rule. The circuit says "goes down" for 52% of the gene pairs and is right
50.3% of the time. Always saying "goes down" is right 52.5% of the time, and a simple rule that only looks at each
target gene's usual direction is right 58.5% of the time. Balanced accuracy is 50.2% and MCC is +0.004, both at
chance. So, as before, the circuit cannot tell which way a gene will move after silencing. There is a small, unplanned
hint of signal in pairs with mid-sized changes, but it is far weaker than the simple rule and needs a planned test.
