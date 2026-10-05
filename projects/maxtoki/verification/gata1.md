# GATA1 / SAE-atlas TF-specificity test: what was done, what is wrong, what can be re-run

KEY = gata1. All paths are absolute. "Verified" = I read the file or ran the numbers. "Inferred" = my reading, not checked directly.

## 0. Bottom line (numbers first)

1. **The GATA1 positive does not hold up under a fair null.** Feature 2610's top-20 gene list is made of rarely detected, very long genes. The median gene in its top-20 appears 6.5 times in the 1,019,996 control token positions. The median across all 4,928 features is 119.5. Only 3 features are this "rare". DoRothEA ChIP-seq target lists are packed with exactly these genes: 26% of genes longer than 140 kb are GATA1 ChIP targets, against 0.13% of genes shorter than 16 kb.
   - If each of 2610's 20 genes is swapped for a random gene with the same detection count, the expected overlap with the GATA1 ChIP set is **4.44** (not ~0.7). P(overlap >= 8) = **0.038**. If the swap also matches gene length, the expected overlap is **5.12** and P(overlap >= 8) = **0.072**.
   - Under this matched null, P(at least one of the 5 features reaches the paper's >=5 threshold) = **0.47** (count-matched) or **0.64** (count and length matched). The paper's random-feature null gives 0.031.
2. **Feature 2610 is not GATA1-specific.** I kept the 5 GATA1 features fixed and swapped in the ChIP set of each of the 291 DoRothEA ChIP TFs with >=20 targets in the gene set. Feature 2610 overlaps >=5 targets for **69 of 291 TFs** and >=8 for 3 TFs (FOXA1 8, FOXP1 8, GATA1 8). Under the count-matched null, GATA1 ranks **46th of 291** (58th with count and length matched). ZBTB7A, SMAD1, FOXJ2 and TFAP2A all show a stronger excess.
3. **Reviewer 2 is right about "power", and the audit script mislabels it.** The file `phase8t_positive_tf_case_study/summary.json` calls its numbers "power", but they are false-positive rates for random features. A real planted-signal test (section 6) shows:
   - The TRRUST test is **not** low-powered for TRRUST-type signal. With 2 planted TRRUST targets, power is 0.9997 at a false-positive rate of 0.003.
   - The TRRUST test has **zero** power if the real signal is ChIP-type. Power stays at 0.003–0.004 for any k, because the GATA1 TRRUST and ChIP lists share only 1 gene.
4. **On the Bonferroni point.** The 4 thresholds are nested cuts of one number, the best overlap. So a correct adjustment for searching over thresholds gives p = **0.0316**, not 0.12. Bonferroni is too strict here. But this does not rescue the result, because the null itself is wrong (point 1).
5. **The step that picks "responding features" has code problems** (section 3). The main ones:
   - The SAE is applied to the wrong layer (off by one).
   - The control group is only about 5 cells.
   - The test runs on token positions, not cells.
   - Feature 2610 passed the effect cut-off by 0.0001 (effect = 0.50011, cut-off 0.5).
6. **The test cannot be run on all 48 TFs from saved files.** The responding-feature IDs exist for GATA1 only. The 0/48 run used an older SAE, which was overwritten on Apr 19. Only 8 of the 48 TFs have DoRothEA ChIP targets. Rebuilding everything needs MaxToki forward passes (costs in section 7). I did not run any.

## 1. Where things are (verified)

| Item | Path |
|---|---|
| Paper text (PLOS ONE) | `<REPO_ROOT>/projects/maxtoki/paper-plos-one/main.tex` lines 1301–1348 (section), 1554–1564 (P1/P6), 1683–1685 (limitations) |
| Figure data (TikZ) | `<REPO_ROOT>/projects/maxtoki/paper-biosystems/main.tex` lines 1291–1292 |
| Original 0/48 run (Phase 8t) | script `<REPO_ROOT>/projects/maxtoki/runs/sae-atlas-217M/scripts/remaining_phases.py` lines 340–482; output `.../outputs/phase8_true/{perturbation_response.csv,summary.json}`; log `.../outputs/remaining_phases.log` |
| A4 chance baseline (TRRUST) | `.../scripts/audit_a4_phase8t_chance_baseline.py`; `.../outputs/phase8t_chance_baseline/{summary.json,null_distribution.csv}` |
| A8 "power" (really a null false-positive rate) | `.../scripts/audit_a8_positive_tf_case_study.py`; `.../outputs/phase8t_positive_tf_case_study/summary.json` |
| **A8 re-run (source of GATA1 / 2610 / 0.030)** | `.../scripts/audit_a8_rerun_phase8t_targeted.py`; `.../outputs/phase8t_positive_tf_rerun/{per_tf_response.json,specificity_table.csv,specificity_with_null.csv,summary.json,run.log}` |
| Feature top-20 catalog (L5, all 4,928 features) | `.../outputs/phase2/layer_05/feature_catalog.json` (the same file exists for all 12 layers) |
| Gene universe (token-to-gene for 500 control cells, L5) | `.../outputs/phase0/layer_05/gene_names.json` (the raw activations `.npy` were deleted) |
| SAE checkpoint L5 | `.../outputs/phase1/layer_05/sae_final.pt` (trained Apr 19 by `full_12layer_pipeline.py`) |
| DoRothEA ChIP-seq file (the "470 set") | `<DATA_ROOT>/biodyn-work/single_cell_mechinterp/external/networks/dorothea_chipseq_human.tsv` |
| DoRothEA all levels | `<DATA_ROOT>/biodyn-work/single_cell_mechinterp/external/networks/dorothea_human.tsv` |
| TRRUST | `<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv` |
| Perturb-seq data | `<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/crispri_validation/data/replogle_concat.h5ad` (30 GB; X = 643,413 × 6,546, dense float32) |
| Audit write-up | `<REPO_ROOT>/projects/maxtoki/audits/audit-20260507-completion-v2.md` lines 38–66 (F7) |

(`...` = `<REPO_ROOT>/projects/maxtoki/runs/sae-atlas-217M`)

## 2. Exactly how the test works (verified from code)

**Perturbation data.** Replogle K562 cells from `replogle_concat.h5ad`, with cell_line == "k562" (188,590 cells).
- There are 1,383 targeted genes, and all have >= 20 cells.
- There are 10,691 non-targeting (NT) control cells.
- GATA1 has 95 cells. The re-run used 50 of them, drawn with `rng.choice` and seed 42.

**Model and layer.** MaxToki-217M has 11 blocks, so HF `hidden_states` has 12 entries (index 0 = embeddings).
- The L5 SAE and the top-20 catalog come from `hidden[5]` (`full_12layer_pipeline.py:141`).
- The A8 re-run feeds **`hidden[PROBE_LAYER + 1]` = `hidden[6]`** into this SAE, for both perturbed and control cells (`audit_a8_rerun_phase8t_targeted.py:139, 201`).
- So the SAE runs one layer later than the layer it was trained on. **This is an off-by-one bug.**
- The original 0/48 run was worse. Its controls were `hidden[5]`, from the Apr 17 phase0 file `layer_05_activations.npy` (`remaining_phases.py:390`). Its perturbed cells were `hidden[6]` (`remaining_phases.py:418`). So it compared two different layers.

**Tokenisation.** Rank-value encoding, `max_len=2048`, <bos> and <eos> included. There are ~2,040 tokens per cell (408,082 rows / 200 control cells, from `run.log`).

**Selection of "responding features"** (`audit_a8_rerun_phase8t_targeted.py:217–233`). This is the same rule as `remaining_phases.py:433–447`.
- Perturbed values: SAE codes for **all token positions** of the 50 GATA1 cells (~102k rows).
- Control values: `h_ctrl[:10000]`, the **first 10,000 token rows** of the 200 NT cells. At ~2,040 tokens per cell, that is about **5 cells** (inferred from the row counts).
- Test: two-sided Mann–Whitney U on token values, per feature, for all 4,928 features. No multiple-testing correction.
- Rule: `p < 0.05 and |mean_pert − mean_ctrl| > 0.5`.
- All 5 GATA1 features have p = 0.0, so the p filter does nothing. Only the effect cut-off matters.
- Effects: 628: 0.792, 1334: 1.872, 2006: 1.655, **2610: 0.50011**, 2627: 0.882 (`per_tf_response.json`). Feature 2610 passes the 0.5 cut-off by 0.0001.
- The sign of the change was not saved (the code stores `abs`).
- Number of responding features: GATA1 = 5 in the re-run (50 cells). MYC had 0 cells. TAL1 was not in the list. In the original 0/48 run (20 cells, older SAE), only CDC5L = 1, GATA1 = 5 and MED1 = 3 had any. The other 45 TFs had 0 (`phase8_true/perturbation_response.csv`).

**Top-20 genes per feature** (`full_12layer_pipeline.py:195–236`; same logic in `phases2_to_8.py:150–160`).
- The SAE encodes all positions of 500 NT control cells (1,019,996 positions).
- For each feature and gene, the code takes the mean activation over the positions where the feature is active (> 0).
- The 20 genes with the highest mean are the top-20. **There is no minimum count.**
- So a gene seen once with one high activation can enter the top-20. Four of 2610's top-20 genes are seen exactly once (MBD5, IMMP2L, RABGAP1L, ALDOA).
- 4,807 of 4,928 features have a full 20 genes. The other 121 have fewer.

**Gene universe.** 6,324 genes appear in the 500 control cells. The data matrix has only 6,546 genes, a limited gene panel (my inference: probably HVG or essential-screen genes). In this universe:
- GATA1 ChIP targets: **219 of 470**.
- GATA1 TRRUST targets: **11 of 57**.
- The two lists share **1** gene.

The paper says "57 TRRUST targets" and "470-element ChIP-seq set". The sizes that actually matter are 11 and 219.

**Target sets.**
- TRRUST: all modes, no filter.
- The "ChIP-seq direct-target set" is `dorothea_chipseq_human.tsv`. By `MECH.md:346` in `<DATA_ROOT>/biodyn-work/single_cell_mechinterp/`, this is DoRothEA from OmniPath, levels A–D, filtered by the DoRothEA ChIP-seq evidence flag.
- For GATA1 the 470 edges are **425 level D, 42 level C, 3 level B, 0 level A**. Most are single-evidence, lowest-confidence edges. They are not "direct targets" in a curated sense.
- Across all 296 TFs in this file, set sizes run from 4 to 491 (median 441; 207 TFs have >= 400). This fits a per-TF cap of about 500 ChIP targets (inferred).

**Null used for p = 0.030** (`audit_a8_rerun_phase8t_targeted.py:273–296`).
- Draw K = 5 features uniformly from all 4,928 catalog features, 1,000 times.
- Report the fraction of draws in which at least one feature has >= thr of its top-20 in the target set.
- 30/1,000 = 0.030. The paper cites Phipson–Smyth, but the code uses b/m, not (b+1)/(m+1). The exact value is 0.03107 (my calculation).
- The null does not repeat the selection step. It does not match on any feature property. It does not account for the threshold search.

**The two "p = 0.003" numbers are not p-values of an observation.**
- For TRRUST, GATA1 failed every threshold. The 0.003 is the chance that random features would pass.
- The figure plots this TRRUST false-positive rate next to the ChIP p-value as if they were the same kind of number.
- The figure also shows TRRUST values of 0.001 at thresholds 3–5. The CSV says 0.0 (`specificity_with_null.csv`, figure data at `paper-biosystems/main.tex:1291`).

## 3. Other problems found (verified unless marked)

1. **Off-by-one layer** (section 2). The top-20 catalog describes `hidden[5]` features. Selection ran on `hidden[6]`. Fixing this could change which features respond, and 2610 could drop out.
2. **About 5 control cells** as the reference, against 50 perturbed cells. Cell-to-cell differences (depth, number of genes) can by themselves push a mean difference over 0.5.
3. **Token-level test.** About 100k vs 10k token values, so every p-value is ~0. The unit should be the cell.
4. **Knife-edge selection.** 2610's effect is 0.50011 against a 0.5 cut-off.
5. **The SAE used for 0/48 is gone.** Phase 8t ran on Apr 17 (log `remaining_phases.log`). `full_12layer.log` lines 126–139 show the L5 SAE was retrained on Apr 19 at the same path. So the 0/48 run cannot be reproduced with its original SAE. The A4 baseline also mixed the Apr 17 counts with the Apr 19 catalog.
6. **How the 48 TFs were chosen.** They are the first 48 TRRUST TFs in category order with >= 20 K562 cells (`remaining_phases.py:376–384`). That is alphabetical, AATF to PDCD11. They were not chosen for relevance to K562.
7. **"Power" is mislabelled** in `audit_a8_positive_tf_case_study.py:86–104` and in `sae-atlas-217M-FINAL_SUMMARY.md` line ~168 ("≈60% power at K=5"). Both are false-positive rates for random features. This is exactly the error Reviewer 2 points out.

## 4. Which TFs can be tested (verified; file `gata1/tf48_target_inventory.csv`)

Of the 48 TFs in `phase8_true`:
- Any DoRothEA target (any level): **19**.
- DoRothEA A/B targets: **12**.
- DoRothEA ChIP-seq targets: **8** (ATF4 156 in-universe, CEBPZ 222, E2F6 172, GATA1 219, GTF2B 200, HINFP 89, MAX 188, MYBL2 130).
- >= 2 TRRUST targets in the gene universe: **21**.

Across all 1,383 K562 perturbations:
- TRRUST TFs: 80.
- DoRothEA TFs (any level): 34.
- TRRUST or DoRothEA: 87.
- DoRothEA ChIP TFs: 20 (ATF4, BRF2, CEBPZ, E2F6, GATA1, GTF2B, HINFP, MAX, MYBL2, SETDB1, SP2, SRF, STAT5A, TBP, TERF1, TERF2, TFDP1, THAP1, THAP11, ZNF24). All have >= 34 cells and 89–287 targets in the universe (`gata1/k562_chip_tfs.csv`).
- MYC has 0 K562 cells. TAL1 is not perturbed.

Top-20 lists exist for **all** 4,928 L5 features, and for all 12 layers (`phase2/layer_XX/feature_catalog.json`). So the overlap step can be re-run for any TF right away.

Responding-feature IDs exist **only for GATA1**. The other TFs need new forward passes.

## 5. Prototype of the selection-aware null (run; no forward passes)

Scripts and outputs are in `<AGENT_TMP>/<SESSION_DIR>/d422dc33-6d6f-4b2b-bbf7-f274d8a82fae/scratchpad/understand/gata1/` (`null1.py`, `null2.py`, `null3.py` and their JSON/CSV files).

**What this prototype does and does not do.** It keeps the 5 selected features fixed, and it handles the threshold search and the gene-rarity confound. It does **not** re-run the feature selection for fake TFs. That needs MaxToki forward passes (section 7).

**A. Exact random-feature null, same as the paper but exact** (`null1_results.json`).
- GATA1 ChIP overlap across all 4,928 features: 0: 2638, 1: 1462, 2: 544, 3: 195, 4: 58, 5: 21, 6: 6, 7: 3, 8: 1. The only feature at 8 is 2610 itself.
- P(max over 5 random features >= t): t=2: 0.602, 3: 0.257, 4: 0.087, **5: 0.0311**, 6: 0.0101, 7: 0.0041, **8: 0.00101**.
- TRRUST: >= 2 gives 0.00304, and >= 3 is impossible (0 features).
- **Threshold search, adjusted by min-p over the nested thresholds {2,3,4,5}: p = 0.0316** (200k simulations). Nested thresholds cost almost nothing, so Bonferroni's 0.12 is too strict.
- Using the observed best overlap (8) directly gives p = 0.001 under this null. This null is invalid, though (see B).

**B. The confound: gene rarity and gene length** (`null2_results.json`).
- GATA1 ChIP membership by detection count in the 500 control cells: 47% for genes seen once, 20–28% for genes seen 2–10 times, 7.9% for 10–50, 2.0% for 50–200, 1.2% for > 200. Spearman rho = −0.17 (p = 4e−42).
- By gene length (quantile bins, from `<REPO_ROOT>/projects/biotensor/data/genemanifold/gene_pos.json`, span = end − start): 0–16 kb 0.13%, 16–36 kb 0.19%, 36–75 kb 0.45%, 75–140 kb 4.8%, 140–1,288 kb **25.8%**. Spearman rho = +0.27 (p = 2e−107).
- Feature 2610's top-20 median length is 191 kb, against 36 kb for the universe and 203 kb for GATA1 ChIP targets. Its 8 overlap genes: MBD5 (496 kb, seen 1×), IMMP2L (900 kb, 1×), RABGAP1L (836 kb, 1×), EVL (173 kb, 2×), STAG1 (416 kb, 6×), NFIA (598 kb, 7×), ARID1B (435 kb, 9×), STXBP5 (188 kb, 40×).
- Median detection count of the top-20 genes: 2610 = 6.5; the other four features = 72.5–97.0; all features: 5th percentile 55.5, median 119.5. **Only 3 of 4,928 features are as rare as 2610.** Their mean GATA1 ChIP overlap is 5.67.

**C. Gene-swap null that keeps each gene's rarity (and length)** (`null3_exact_geneswap.json`; 100k draws, no repeats within a feature).

| Null | Expected overlap for 2610 | P(2610 >= 8) | P(best of 5 >= 5) |
|---|---|---|---|
| Random features (paper) | 0.72 (all-feature mean) | 0.0002 | 0.031 |
| Matched on detection count | 4.44 | **0.038** | **0.47** |
| Matched on count and length | 5.12 | **0.072** | **0.64** |

The other four features have expected overlaps of 0.70–1.09 under the matched nulls. They are ordinary.

**D. TF-label swap: keep the 5 features, use other TFs' ChIP sets** (`tf_swap_null.csv`, `tf_swap_2610_count_only.csv`, `tf_swap_2610_count_x_length.csv`; 291 TFs with >= 20 in-universe targets).
- Raw overlap with feature 2610: >= 5 for 69/291 TFs, >= 8 for 3 TFs (FOXA1, FOXP1, GATA1). My inference is that FOXA1 is not expressed in K562.
- Under the paper's random-feature null, GATA1 ties for first with about 10 TFs at the floor (fraction of TFs with p <= GATA1 = 0.034). Hypergeometric rank for 2610: 4th. Under the paper's own min-p over thresholds 2–5, GATA1 ranks 98th (34% of TFs are as good or better).
- Under the count-matched gene-swap null: GATA1's p = 0.043, **rank 46/291**, rank 14 by excess. With count and length matched: p = 0.079, **rank 58/291**, rank 32 by excess. (These TF-swap runs sample with replacement inside a bin, so they are a close approximation. The exact values for GATA1 are in C.)
- Reading: after controlling for gene rarity and length, 2610 carries a small excess of about 3 ChIP genes. That excess is not specific to GATA1.

## 6. Power simulation with a planted signal (run; `power.py`, `power_results.json`)

**Design.**
- Draw 5 random features. In one of them, replace k of its top-20 genes with k distinct genes from a "truth" target set.
- Detection means the best of 5 reaches the critical value at alpha = 0.05 under the random-feature null: TRRUST >= 2 (p = 0.003) and ChIP >= 5 (p = 0.031).
- 20,000 simulations per cell.
- "recall r" means each planted true target is in the evaluation database with probability r. Otherwise it is swapped for a gene not in the database.

| Truth → evaluated with | k=0 | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| TRRUST → TRRUST (r=1) | 0.003 | 0.034 | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ChIP → ChIP (r=1) | 0.030 | 0.038 | 0.068 | 0.153 | 0.419 | **0.999** | 0.999 | 0.999 | 1.000 |
| ChIP → ChIP (r=0.5) | 0.033 | 0.034 | 0.043 | 0.059 | 0.091 | 0.153 | 0.250 | 0.482 | 0.693 |
| ChIP → ChIP (r=0.25) | 0.031 | 0.035 | 0.037 | 0.038 | 0.044 | 0.053 | 0.061 | 0.104 | 0.159 |
| **ChIP → TRRUST** | 0.004 | 0.003 | 0.004 | 0.004 | 0.004 | 0.004 | 0.004 | 0.004 | 0.003 |
| TRRUST → ChIP | 0.032 | 0.032 | 0.031 | 0.030 | 0.030 | 0.028 | 0.027 | 0.027 | 0.030 |

**What this means.**
- The original TRRUST test was not "underpowered" in general. It detects 2-of-20 TRRUST targets almost every time.
- It has no power when the true signal is made of genes TRRUST does not list. For GATA1 that is nearly all ChIP genes, because the lists share 1 gene in the universe.
- The paper's claim of "low power from a low null probability" is wrong in its reasoning. The correct statement is that the two databases disagree almost completely.

**Not done.**
- A power run with a rarity-matched critical value (the ChIP critical value would rise for rare-gene features).
- Planting into the actual 5 GATA1 features instead of random features.

## 7. Feasibility and cost of the three requested analyses

Timing basis, verified from `run.log`: MaxToki-217M on MPS, about 0.8 s per cell (50 GATA1 cells in 39.9 s) and about 1 s per control cell (200 in 192.9 s). One Mann–Whitney call at these sizes takes 3.4 ms on CPU (I timed it on synthetic sparse data), so 4,928 features take about 17 s per TF.

First fix the selection step, or any null inherits its bugs:
- Use `hidden[5]`.
- Use all control cells, or a large pool, with a cell-level (pseudobulk) test.
- Store the effect sign.
- Report selection with a stability check (bootstrap over cells).

**(1) Selection-aware null.** Feasible.
- Option A: fake TFs from NT cells.
  - Encode about 1,000 NT cells once: about 17 min on MPS.
  - Store sparse TopK codes. With k = 32 per token, that is about 65k non-zeros per cell, about 0.3 GB for 1,000 cells. Also store per-cell per-feature sums for exact pooled means.
  - Each null replicate samples 50 fake "perturbed" cells, re-runs the selection (the mean cut-off first, then Mann–Whitney only for candidates), takes the best overlap with the GATA1 ChIP set over the selected features, and takes the min-p over thresholds {2,3,4,5}.
  - 1,000 replicates: minutes on CPU after encoding. Total about 30 min.
- Option B: permuted perturbation labels, using other real perturbations as fake TFs.
  - 200 random K562 perturbations × 50 cells: about 2.2 h of forward passes plus about 1 h of Mann–Whitney. All 1,383 would take about 15 h.
  - This gives the most direct null: do features that respond to a random knockdown overlap GATA1 ChIP targets as much?
- In both options, also use the rarity/length-matched overlap statistic from 5C. Otherwise the rare-gene confound comes back.

**(2) Power simulation.** Done here in seconds (section 6). Extending it to rarity-matched critical values and to the real 5 features takes under 5 min of CPU.

**(3) All testable TFs with BH.**
- The 20 DoRothEA ChIP TFs perturbed in K562: about 14 min of forward passes plus about 6 min of statistics. About 25 min total.
- All 87 TRRUST ∪ DoRothEA TFs: about 1.5 h.
- Apply BH across TFs (and across the two databases). Count TFs with 0 responding features as "not tested", not as negatives.
- A cleaner specificity test is a 20 × 20 matrix: responding features of TF i against the ChIP set of TF j. Compare the diagonal with the off-diagonal.
- Only 8 of the original 48 TFs have ChIP sets, so "all 48" is not possible with ChIP. With TRRUST, 21 of 48 have >= 2 targets in the universe.

## 8. What I did NOT do

- No MaxToki forward passes. So no re-selection of responding features, no fix of the layer bug, and no test of whether 2610 still responds.
- No check of the Mann–Whitney step on real SAE codes. The 17 s per TF figure comes from synthetic data.
- Gene length comes from `biotensor/data/genemanifold/gene_pos.json` as genomic span. I did not use a transcript-length annotation. 54 of 6,324 universe genes have no length.
- The TF-swap tables sample with replacement inside rarity bins, so they are approximate. The GATA1-only numbers in 5C are exact (sampled without replacement).
- I did not check GATA1 knockdown efficiency in these 95 cells, or whether GATA1 target genes actually change in the perturbed cells.
- No repo files were changed.

## 9. Suggested reply to the reviewers (draft wording)

- Accept Reviewer 2's power point. Replace the "low power" wording with the planted-signal result: the TRRUST test detects TRRUST-type enrichment almost always (2/20 → 0.9997), but cannot see ChIP-type enrichment (≤ 0.004), because the two GATA1 lists share 1 gene.
- Note that threshold search over nested cut-offs gives p = 0.032, not 0.12. Then say this does not matter, because a null matched for gene rarity and length gives p = 0.04–0.07 for 2610, and GATA1 ranks 46–58 of 291 TFs for the same feature. **Withdraw the "positive reversal" claim.** State that TF-specificity is untested at adequate rigor, not present and not absent.
- Or run (1) Option A plus (3) on the 20 ChIP TFs, about 1 h in total, after fixing the layer and control bugs, and report whatever comes out.

## 10. Plain-words summary

The GATA1 "positive" comes from one SAE feature (2610) whose top-20 list is full of long genes that are rarely seen in the data. The GATA1 ChIP-seq list is full of the same kind of genes. When the null keeps that property, 2610's overlap of 8 is only mildly unusual (p about 0.04–0.07). The same feature matches the ChIP lists of dozens of unrelated TFs just as well or better. Reviewer 2 is right that the paper confused a low false-positive rate with low power. A real power test shows the TRRUST test works for TRRUST-type signal but cannot see ChIP-type signal. The feature-selection code also has bugs: wrong layer, about 5 control cells, token-level statistics, and a borderline cut-off. The fair test (fake-TF null, all 20 ChIP TFs, BH) is feasible in about 1 hour of MaxToki forward passes after those bugs are fixed. I did not run it.
