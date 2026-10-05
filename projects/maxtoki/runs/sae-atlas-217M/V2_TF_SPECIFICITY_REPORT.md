# V2 TF-specificity report: layer-5 SAE features vs TF target sets, redone (item D3)

Date: 2026-10-01. Model: MaxToki-217M (HF safetensors, float32, Apple MPS). torch 2.11.0, transformers 5.5.4.
Every number below comes from a file written by the scripts in section 9. "Cell bootstrap" means cells are
resampled with replacement, knockdown and control groups separately.

## 0. Results first

1. **The deployed GATA1 positive does not survive the fixed selection.**
   - With the SAE read at the right layer and 400 real control cells, GATA1 still has 5 responding features at
     the deployed cut-off (0.5). Four are the same as before (628, 1334, 2006, 2627). Feature **2610 drops out**.
     Its effect is 0.461 (95% CI 0.412 to 0.512, cell bootstrap, 1,000 resamples), below the 0.5 cut-off. It
     is selected in only 6% of bootstrap resamples. Feature 3167 comes in instead.
   - The best GATA1 ChIP-seq overlap among the 5 features is now **3 of 20** (feature 3167), not 8.
     That gives p = 0.26 under the deployed random-feature null and p = 0.37 under a null matched on gene
     detection count (0.56 when matched on count and gene length). GATA1 ranks **151st of 291** TFs for the
     same features. The TRRUST overlap is 0.
2. **No TF shows target-specific features under a fair null.** I tested 87 knocked-down TFs (20 with ChIP-seq
   target sets) at 7 effect cut-offs. No TF passes BH (q < 0.05 across TFs) under the count-matched null, the
   count-and-length-matched null, or the other-knockdown null, at any cut-off. One single nominal hit exists
   (CEBPZ, cut-off 0.01, section 5.3). It appears at one cut-off only.
3. **The requested selection-aware null (fake TFs from control cells) cannot test specificity.**
   Random groups of control cells never produce a responding feature at cut-offs 0.25 and above (0 of 2,000
   per group size). So any real knockdown whose changed features hold 2 or more targets gets the floor p-value,
   1/2,001 = 0.0005. For GATA1 at the deployed cut-off, BH across the 20 TFs gives q = 0.010. This only shows
   that the knockdown changes some features. It says nothing about whether those features are about the TF's
   targets. I added two selection-aware nulls that can measure this (section 3). Both give GATA1 p = 0.86
   (K-matched fake TFs) and p = 0.67 (other knockdowns; only 5 other knockdowns had a responding feature).
4. **The deployed cut-off of 0.5 is almost never reached at the correct layer.** Per-cell feature means are
   small (median 0.008 across features). At 0.5, only 6 of 87 TFs have any responding feature (CDC5L 1,
   GATA1 5, MED1 2, PHB2 1, RUVBL1 1, TAF1 2), and GATA1 is the only one of the 20 primary TFs.
   With no effect cut-off (BH only), 70 of 87 TFs have responding features (GATA1: 4,206 of 4,928).
5. **The old TRRUST "≥ 2 of top-20" test, on the fixed selection:** 0 of 48 deployed TFs pass at cut-off 0.5
   (3 have responding features). This is the same verdict as the deployed "0/48". At cut-offs ≤ 0.02 GATA1
   meets the old rule (2 TRRUST targets), but its p-value is 0.15 or higher under every null.
6. **Power (planted targets).** For GATA1 at the deployed cut-off, planting 8 GATA1 ChIP targets into one
   responding feature is detected with probability **0.93** (95% Wilson CI 0.88 to 0.95, 200 repeats) under
   the count-matched null, and 1.00 under the deployed null. Planting 2 TRRUST targets is detected with
   probability 1.00. But the deployed statistic stops at 5 targets (thresholds 2..5). Under the count-and-length
   null this cap kills the power (0.01 at k = 8). Without the cap the power is 1.00. When many features
   respond (low cut-offs), the capped statistic loses power too (0.43 to 0.75 at k = 8); the uncapped one keeps
   0.89 to 0.96.

All named sanity checks pass (section 4): the fake-TF p-values are close to uniform (KS on 20,000 fake TFs,
28 settings: median KS p 0.45, smallest 0.007; at most 1.3% of conservative p-values ≤ 0.05), the planted
signal at k = 8 reaches 0.93 power for the ChIP set under the count-matched null, and GATA1 is in the set.

---

## 1. What was wrong in the deployed test, and what I changed

The deployed test is in `scripts/audit_a8_rerun_phase8t_targeted.py` (GATA1) and `scripts/remaining_phases.py`
Phase 8t (the "0/48"). The investigation (`verification/gata1.md`) and the hook report
(`runs/exhaustive-mapping-217M/V2_HOOKS_REPORT.md`) found these selection bugs. Each is fixed here.

| Problem in the deployed code | Fix in this run |
|---|---|
| Layer-5 SAE applied to `hidden_states[6]` (one layer late; VE 0.665 instead of 0.897) | SAE applied to the input of block 5 = `hidden_states[5]`, the tensor it was trained on (`hooks_v2.site_module(model, 5)`). The forward pass stops there. I checked that the captured tensor equals `hidden_states[5]` (max abs difference 0.0, 3 cells). |
| Control = first 10,000 token rows, about 5 cells | 400 reference control cells, sampled at random (seed 20261001) from the 10,191 K562 non-targeting cells that were not used to build the catalog |
| Test on token positions (~100,000 vs 10,000 values; every p ≈ 0) | Unit = cell. Per cell: the feature's mean activation over the cell's gene tokens (`<bos>`, `<eos>` excluded). Two-sided Mann-Whitney U, knockdown cells vs reference cells |
| No multiple-testing correction across features | BH across the 4,928 features; responding = q < 0.05 and abs(mean difference) > cut-off |
| Only |effect| stored | Signed effect, Cohen's d, p and q stored for every (TF, feature) with q < 0.05 |
| 20 or 50 knockdown cells; GATA1, MYC, TAL1 only, or 48 alphabetical TFs | Up to 100 knockdown cells per TF (seed 20261002 + TF index); all 87 K562-knocked-down TFs with TRRUST or DoRothEA targets; the 20 with ChIP-seq targets are the primary set |
| Null = random catalog features only; p = b/m | Five nulls (section 3); all Monte-Carlo p-values use (1 + b)/(1 + B) |

Kept unchanged, as asked: the effect cut-off 0.5 (primary), the top-20 catalog and its definition, and the
statistic (overlap of a responding feature's top-20 genes with the TF's target set, thresholds 2..5).

## 2. Data

- **Cells** (Replogle K562, `replogle_concat.h5ad`; manifest with dataset row, barcode and gem group in
  `outputs/v2_tf_specificity/cell_manifest.csv`): 400 reference controls, 800 held-out controls (the pool for
  fake TFs), 7,500 knockdown cells (87 TFs, 34 to 100 cells each; GATA1 95), and the 500 catalog control cells
  (seed 42, used only to rebuild the catalog). The three control groups do not overlap.
  All 9,200 cells tokenised (0 failures).
- **Primary TFs (20, DoRothEA ChIP-seq targets):** ATF4, BRF2, CEBPZ, E2F6, GATA1, GTF2B, HINFP, MAX, MYBL2,
  SETDB1, SP2, SRF, STAT5A, TBP, TERF1, TERF2, TFDP1, THAP1, THAP11, ZNF24.
- **Gene universe:** 6,324 genes seen in the 500 catalog cells. GATA1 targets in the universe: 219 ChIP-seq,
  11 TRRUST.
- **Knockdown check:** for the 80 TFs whose gene is in the 6,546-gene panel, the TF's mean expression in
  knockdown cells is at most 30% of the reference level (median 0%). Seven TF genes are not in the panel,
  including GATA1, so their knockdown cannot be checked here.
- **Forward passes:** 7 chunks of about 7.5 minutes each, 52 minutes in total, 0.29 to 0.43 s per cell
  (MPS, batch 1, eager attention, max_len 2,048). Memory was ≥ 3 GB free before each model load.

## 3. Methods

**Statistic.** For a TF and a database (ChIP-seq or TRRUST), M = the best overlap of a responding feature's
top-20 genes with the target set. The deployed rule calls the TF "specific at threshold t" if M ≥ t, for
t = 2, 3, 4, 5. These four cuts are nested, so the search over them equals one statistic:
T = min(M, 5) if M ≥ 2, else 0. Every p-value below is P_null(T ≥ T_obs). This is the min-p over the four
thresholds, calibrated under that null. I also report the uncapped M (thresholds 2..20).

**Nulls.**

| Code | Null | How |
|---|---|---|
| rf | Deployed random-feature null | K random catalog features (K = number of responding features), exact formula |
| cm | Gene swap, matched on detection count | each top-20 gene swapped for a random gene from the same count bin (bins at 1, 2, 5, 10, 20, 50, 100, 200, 500 cells), without replacement inside a bin; computed exactly by convolving hypergeometric distributions |
| cml | Gene swap, matched on count and gene length | same, bins = count bin x length tertile (cuts 21,375 and 56,220 bp) |
| fake | Selection-aware, as specified | fake TF = random subset of the 800 pool cells, same size as the knockdown group; the whole selection re-run against the same 400 reference cells; T from the fake TF's responding features and the real TF's target set; 2,000 fake TFs per group size |
| fakeK | Selection-aware, K-matched (added) | as fake, but the fake TF keeps its K most-changed features (K = the real TF's count; ordered by abs(mean difference) for cut-off > 0, by p for cut-off 0). Added because "fake" is degenerate |
| okd | Other knockdowns (added) | the responding features of every other knockdown (of the 87) that has ≥ 1 responding feature at that cut-off, scored against this TF's target set. Smallest possible p = 1/(1 + number of such knockdowns) |

**Specificity rank.** For the same responding features, I computed the cm p-value with the target set of every
TF that has one (ChIP: 291 TFs with ≥ 20 targets in the universe; TRRUST: 109 TFs with ≥ 5). The rank of the
true TF among them is reported (mid-rank for ties; only meaningful when T_obs ≥ 2).

**Multiplicity.** BH across TFs, separately for each database, cut-off and null: across the 20 primary TFs and
across all 87. TFs with no responding feature enter the fake, cm and cml corrections with p = 1. For fakeK
and okd, only TFs with ≥ 1 responding feature enter. I did not correct across the 7 cut-offs or across nulls.

**Power.** For each primary TF with ≥ 1 responding feature: pick one responding feature at random, replace
k random non-target genes in its top-20 with k random true targets (k = 1..8), recompute T, and test it under
each null at alpha 0.05. 200 repeats per (TF, cut-off, database, k). The cm and cml nulls are recomputed for
the edited feature (planted targets change its gene mix).

## 4. Sanity checks (all pass)

| Check | Result | Pass |
|---|---|---|
| Tokenisation matches the deployed catalog cells | 1,019,996 of 1,019,996 positions identical | yes |
| Rebuilt top-20 catalog (our forward passes, SAE at `hidden_states[5]`) matches the deployed catalog | 4,927 of 4,928 lists identical in genes and order; the last differs by 1 gene (Jaccard 0.905) | yes |
| Fast Mann-Whitney (used for all fake TFs) vs scipy (float64) | max abs p difference 7.4e-8 (GATA1) and 8.2e-7 (a fake TF); same BH selections | yes |
| Exact gene-swap null vs the investigation prototype and vs Monte Carlo (deployed GATA1 features) | count-matched: E[overlap of 2610] 4.444 (prototype 4.44), P(2610 ≥ 8) 0.0366 (0.038), P(best of 5 ≥ 5) 0.472 (0.47); count+length: 5.117 / 0.0719 / 0.642 (5.12 / 0.072 / 0.64); Monte Carlo (10,000 draws) agrees within its error (largest gaps: 0.007 in a probability, 0.023 in the expected overlap) | yes |
| **Fake-TF p-values uniform (KS)** | 1,000 new fake TFs per primary TF (20,000 total), same selection code, scored against the stored 2,000-draw nulls; 28 settings (7 cut-offs x 2 databases x fake/fakeK): KS p median 0.45, min 0.007; share of conservative p ≤ 0.05 at most 0.013; two-sample chi-square (new vs stored null draws) p < 0.05 in 5 of 135 per-TF tests (about the 5% expected by chance) | yes |
| At the deployed cut-off (fake TF KS) | ChIP fake 0.31 (n = 20,000), ChIP fakeK 0.91 (n = 1,000), TRRUST fake 0.95, TRRUST fakeK 0.13 | yes |
| **Planted signal, k = 8, ChIP set, matched null** | GATA1 at cut-off 0.5: 0.93 [0.88, 0.95] (count-matched); 1.00 uncapped | yes |
| **GATA1 in the set** | yes (95 knockdown cells, 219 ChIP and 11 TRRUST targets in the universe) | yes |
| Selection false-positive rate, fresh random splits of the 1,200 control cells (1,000 splits, 400 vs 95) | P(≥ 1 responding feature): 0.039 at cut-off 0; 0.004 at 0.01; 0.002 at 0.02; 0 at 0.05 and above. Single-feature p-values uniform (KS p 0.32 to 0.56; share p < 0.05: 0.043 to 0.056) | yes |

Two things I found while checking, and resolved:
- With the fixed 400 reference cells and the fixed 800-cell pool, the per-feature p-values of fake TFs were
  not uniform across fake TFs (KS p down to 4e-38 for one feature), and 10.9% of fake TFs had ≥ 1 BH feature at
  cut-off 0. Cause: all fake TFs share one pool and one reference, so a chance pool-vs-reference difference is
  inherited by every fake TF. With fresh random splits (row above) the test is calibrated (0.039). The pool
  itself does not differ from the reference (0 BH-significant features; 8 features with p < 0.001 against 4.9
  expected).
- In the first calibration run (250 fake TFs per TF), one of 28 KS tests gave p = 1e-4 (cut-off 0.25, ChIP,
  fakeK). Re-drawing 2,000 fake TFs for the three TFs involved reproduced the stored null (chi-square
  p = 0.45, 0.47, 0.79), and the larger 1,000-per-TF run passes (KS p 0.23). I found no code cause. I read it as
  Monte-Carlo noise in a 2,000-draw null where T takes only 3 to 4 values.

## 5. Results

### 5.1 Selection

Number of responding features per primary TF (K) by cut-off. Full table: `task_data/tf_summary.tsv`.

| TF | cells | BH q < 0.05 | K at 0.5 | 0.25 | 0.1 | 0.05 | 0.02 | 0.01 |
|---|---|---|---|---|---|---|---|---|
| GATA1 | 95 | 4,206 | **5** | 12 | 52 | 119 | 254 | 405 |
| MAX | 100 | 3,160 | 0 | 1 | 16 | 62 | 186 | 291 |
| HINFP | 100 | 1,477 | 0 | 0 | 3 | 9 | 58 | 142 |
| THAP1 | 100 | 1,307 | 0 | 0 | 1 | 7 | 55 | 157 |
| TFDP1 | 100 | 720 | 0 | 0 | 1 | 4 | 36 | 107 |
| GTF2B | 34 | 477 | 0 | 0 | 2 | 14 | 57 | 102 |
| CEBPZ | 100 | 418 | 0 | 0 | 0 | 0 | 4 | 7 |
| MYBL2 | 81 | 417 | 0 | 0 | 2 | 5 | 10 | 29 |
| TERF2 | 47 | 162 | 0 | 1 | 1 | 3 | 7 | 16 |
| TBP | 100 | 63 | 0 | 0 | 0 | 0 | 0 | 1 |
| 10 others | 49–100 | 0–11 | 0 | 0 | 0 | 0 | 0 | 0 |

**GATA1 at the deployed cut-off** (`gata1_bootstrap.json`; effect = mean difference of per-cell feature
means, knockdown minus reference; CI = cell bootstrap, 1,000 resamples, seed 20263027):

| Feature | Effect [95% CI] | Selected in bootstraps | ChIP targets in top-20 | Deployed run |
|---|---|---|---|---|
| 1334 | 1.611 [1.374, 1.852] | 100% | 1 | selected (1.872) |
| 2006 | 1.336 [1.213, 1.451] | 100% | 1 | selected (1.655) |
| 3167 | 0.855 [0.738, 0.966] | 100% | 3 | not selected |
| 2627 | 0.756 [0.663, 0.845] | 100% | 1 | selected (0.882) |
| 628 | 0.688 [0.589, 0.789] | 100% | 2 | selected (0.792) |
| 2610 | 0.461 [0.412, 0.512] | 6% | 8 | selected (0.50011) |

The best ChIP overlap at cut-off 0.5 is 3 in 94% of bootstraps and 8 (via 2610) in 6%. The token-pooled
effect (the deployed definition, all positions) is the same as the cell-mean effect to 3 decimals (2610:
0.4615 vs 0.4612), so the drop of 2610 comes from the layer and control fixes, not from the change of unit.
I did not separate the layer fix from the control fix.

### 5.2 TF tests at the deployed cut-off (0.5)

Only GATA1 is testable among the 20 primary TFs (ChIP set). Among all 87, five more TFs have responding
features (CDC5L, MED1, PHB2, RUVBL1, TAF1); none has a ChIP-seq set, and their TRRUST sets have 1 to 2 genes
in the universe. Their TRRUST overlap is 0.

| GATA1, cut-off 0.5 | ChIP-seq | TRRUST |
|---|---|---|
| K / best overlap M / T | 5 / 3 / 3 | 5 / 0 / 0 |
| rf (deployed null) | 0.257 | 1 |
| cm (count-matched) | 0.371 (expected M 2.31) | 1 |
| cml (count + length) | 0.564 | 1 |
| fake (as specified) | 0.0005 (floor; BH q 0.010) | 1 |
| fakeK | 0.864 | 1 |
| okd (5 other knockdowns; smallest possible p 0.17) | 0.667 | 1 |
| rank among TFs | 151 / 291 | – (T = 0) |

### 5.3 Sensitivity to the effect cut-off (20 primary TFs)

| cut-off | db | testable TFs | median K | T ≥ 2 | BH fake | BH cm | BH cml | BH fakeK | BH okd | true TF in top 5% |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | ChIP | 1 | 5 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| 0.25 | ChIP | 3 | 1 | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| 0.1 | ChIP | 8 | 2 | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| 0.05 | ChIP | 8 | 8 | 5 | 5 | 0 | 0 | 0 | 0 | 0 |
| 0.02 | ChIP | 9 | 55 | 7 | 7 | 0 | 0 | 0 | 0 | 0 |
| 0.01 | ChIP | 10 | 104 | 9 | 9 | 0 | 0 | 1 | 0 | 0 |
| 0 | ChIP | 14 | 418 | 12 | 12 | 0 | 0 | 0 | 0 | 0 |
| 0.5 to 0.05 | TRRUST | 1 to 5 | – | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.02 to 0 | TRRUST | 6 to 10 | – | 1 (GATA1) | 1 | 0 | 0 | 0 | 0 | 1 at 0.02 |

"BH fake" counts every TF with T ≥ 2, because the fake null is degenerate (section 0, point 3).
The all-87 table gives the same picture (`summary_tables.md`, Table A2): 0 TFs pass BH under cm, cml or okd
at any cut-off.

**The one nominal hit:** CEBPZ, ChIP, cut-off 0.01. 7 responding features with small effects (0.010 to 0.032),
best overlap 4 (feature 2740). p = 0.027 (rf), 0.030 (cm; BH q 0.61), 0.036 (cml), 0.002 (fakeK; BH q 0.020
across the 10 testable TFs), 0.55 (okd), rank 15 of 291. At the next cut-offs it disappears (0.02: best overlap
1; 0: p_cm 0.19). There is no correction across the 7 cut-offs. I do not read it as a finding.

GATA1 across cut-offs (ChIP): once feature 2610 enters (cut-off ≤ 0.25), M = 8 but the expected M under the
count-matched null rises to 5.6 to 6.6. Uncapped p = 0.09 to 0.15, capped p = 0.79 to 1.0, rank 255 to 277 of
291. GATA1 TRRUST at cut-off 0.02: M = 2, p_cm 0.14, rank 2 of 109 (the only "top 5%" case).

### 5.4 Power (planted targets)

Mean detection probability at alpha 0.05. One TF at cut-off 0.5 (GATA1): 95% Wilson CI over 200 repeats.
Several TFs at 0.1: 95% CI from a bootstrap over TFs. Full table: `summary_tables.md` Table D, `power_summary.json`.

| cut-off | db | k | TFs | rf | cm | cm uncapped | cml | cml uncapped | fakeK |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | ChIP | 2 | 1 | 0.19 | 0.18 | 0.18 | 0.02 | 0.02 | 0.00 |
| 0.5 | ChIP | 4 | 1 | 1.00 | 0.99 | 1.00 | 0.03 | 0.38 | 0.00 |
| 0.5 | ChIP | 8 | 1 | 1.00 [0.98, 1.00] | **0.93 [0.88, 0.95]** | 1.00 | 0.01 [0.00, 0.03] | 1.00 | 0.00 |
| 0.5 | TRRUST | 2 | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| 0.1 | ChIP | 8 | 8 | 0.88 [0.62, 1.00] | 0.75 [0.38, 1.00] | 0.96 [0.89, 1.00] | 0.64 [0.28, 0.89] | 0.96 [0.89, 1.00] | 0.75 |
| 0 | ChIP | 8 | 14 | 0.43 [0.21, 0.71] | 0.43 [0.14, 0.71] | 0.93 [0.81, 1.00] | 0.43 | 0.93 | 0.50 |

What this means:
- At the deployed cut-off, the count-matched test detects 4 or more planted ChIP targets almost always.
- The cap at 5 is the weak point. Planted ChIP targets are rare, long genes. Under the length-matched null, a
  feature full of such genes is expected to hold several ChIP targets by chance, so P(≥ 5) stays above 0.05.
  Only the uncapped count (8 of 20) separates signal from that background.
- When many features respond, the chance best overlap already reaches 5, so the capped test cannot see more.
  For several TFs the real data already sit at T = 5, so planting adds nothing.
- The TRRUST test detects 2 planted TRRUST targets every time (1.00), as the investigation found. TFs with
  fewer than 2 TRRUST targets in the universe can never pass it (this is why mean TRRUST power stays at 0.7 to
  0.86 at lower cut-offs).
- The fake null has power 1.00 for any T ≥ 2 and the okd null has no power at cut-off 0.5 (5 other knockdowns).
  Neither can be read as a specificity test there.

### 5.5 Old TRRUST "≥ 2 of top-20" test on the fixed selection (`old_trrust_test.csv`)

| cut-off | deployed 48 TFs: testable / pass | 20 primary | all 87 |
|---|---|---|---|
| 0.5 | 3 / 0 | 1 / 0 | 6 / 0 |
| 0.25 | 5 / 0 | 3 / 0 | 13 / 0 |
| 0.1 | 17 / 0 | 8 / 0 | 36 / 0 |
| 0.02 | 25 / 1 | 9 / 1 | 49 / 1 |
| 0 | 39 / 1 | 14 / 1 | 70 / 1 |

The one pass is GATA1 (2 TRRUST targets), with p_rf ≥ 0.147 in every case. At 0.5 the three testable deployed
TFs are CDC5L (1 feature), GATA1 (5) and MED1 (2). The deployed run found the same three (1, 5 and 3).

## 6. What changed versus the deployed result

| Deployed claim | Fixed run |
|---|---|
| GATA1 responding features 628, 1334, 2006, 2610, 2627 | 628, 1334, 2006, 2627 and 3167; 2610 out (effect 0.461 < 0.5) |
| GATA1 ChIP-seq overlap 8 of 20 (2610), p = 0.030 vs random features | best overlap 3 of 20 (3167); p = 0.26 (same null), 0.37 (count-matched), 0.56 (count + length), 0.86 (K-matched fake TFs); rank 151/291 |
| "0/48 TFs specific" (TRRUST ≥ 2) | 0/48 at cut-off 0.5 (3 testable); the same three TFs respond |
| "≈ 60% power" (actually a random-feature false-positive rate) | real power: count-matched 0.93 at k = 8 (ChIP, GATA1, cut-off 0.5); TRRUST 1.00 at k = 2; the length-matched capped test has almost no power for ChIP-type signal |
| Positive TF-specificity case | No TF passes BH under any null that controls gene rarity, at any of 7 cut-offs |

## 7. What I could NOT do, and limits

- **The fake-TF null as specified cannot measure specificity.** Control-cell groups select no features, so its
  p-value only says the knockdown changed something. I report it, and I added the K-matched and
  other-knockdown nulls to answer the specificity question.
- **The fake TFs share one reference group and one pool.** Their p-values are calibrated given that pool
  (section 4), but they are not independent draws from the population. I did not re-draw the reference group
  for every fake TF.
- I did not separate the effect of the layer fix from the effect of the control fix on feature 2610. That would
  need SAE codes at `hidden_states[6]`, which I did not compute.
- GATA1 and six other TF genes are not in the 6,546-gene panel, so I could not check their knockdown strength.
- No correction across the 7 cut-offs or across the 6 nulls. BH is only across TFs.
- The rank test is descriptive. Target sets of different TFs are not exchangeable (they differ in size and
  gene mix), so the rank is not a p-value.
- Power was simulated for the 20 primary TFs only, not for all 87.
- Gene length is the genomic span from `biotensor/data/genemanifold/gene_pos.json`, not transcript length.
  54 of 6,324 genes have no length (put in the middle tertile, as in the prototype).
- The deployed top-20 definition (mean activation when active, no minimum count) is kept, as asked. Its
  rare-gene problem stays in the statistic; only the matched nulls handle it.
- The original "0/48" SAE (Apr 17) no longer exists, so the original run itself cannot be reproduced.
- `run_config_stats.json` lists a calibration seed (SEED + 99) that the code does not use; the code uses
  SEED + 555. This is noted in `run_config.json`.

## 8. How to reuse the data

`outputs/v2_tf_specificity/task_data/` has everything needed for a later controlled experiment, with a
`README.md` that defines every file and column: the gene universe with detection counts, lengths and bins;
top-20 lists for all 4,928 layer-5 features (deployed lists plus our rebuilt activation values); TRRUST and
DoRothEA targets in the universe (with confidence levels and ChIP flags); responding features with signed
effects, p, q and cut-off flags for all 87 TFs; per-TF summaries and all test results; the cell manifest; and
the per-cell feature means for all 9,200 cells (`cell_feature_means.npz`, 151 MB).

## 9. Files

Scripts (`runs/sae-atlas-217M/scripts/`):
- `v2_tf_specificity_extract.py`: cell manifest, MaxToki forward passes to block 5, per-cell SAE summaries (MPS, chunked).
- `v2_tf_specificity_taskdata.py`: gene universe, target sets, token check, catalog rebuild.
- `v2_tf_specificity_stats.py`: selection, all nulls, rank, power, BH, calibration (CPU; fake nulls cached in `fake_null/`).
- `v2_tf_specificity_calib_check.py`: extra calibration checks A to D.
- `v2_tf_specificity_summary.py`: GATA1 bootstrap, CIs, summary tables, merged `run_config.json`, task-data README.

Outputs (`runs/sae-atlas-217M/outputs/v2_tf_specificity/`):
- `tf_results.csv` (all tests, TF x cut-off x database), `power.csv`, `power_summary.json`, `old_trrust_test.csv`,
  `summary.json`, `summary_tables.md`, `summary_counts.csv`, `gata1_bootstrap.json`, `knockdown_efficiency.json`,
  `calibration.json`, `calibration_extra.json`
- `run_config.json` (merged provenance: device, versions, seeds, cell IDs, wall time per chunk, sha256 of every
  script, SAE and hook library), plus `run_config_extract.json`, `run_config_stats.json`
- `cell_manifest.csv`, `cell_manifest_meta.json`, `extract_progress.json`, `cells/` (per-cell summaries and
  catalog-cell sparse codes), `fake_null/` (stored fake-TF null draws)
- `task_data/` (section 8)

## Plain-words summary

The old test read the SAE one layer too late, compared against about 5 control cells, and counted token
positions instead of cells. I fixed all three. I ran 9,200 cells through MaxToki: 400 control cells, 800 more
control cells for fake TFs, and up to 100 knockdown cells for each of 87 TFs.

With the fix, GATA1 still changes 5 features at the old cut-off. But the feature behind the old GATA1 result
(2610) is no longer one of them. Its change is 0.46, just under the 0.5 cut-off. The best GATA1 feature now
holds 3 GATA1 ChIP-seq targets in its top 20 genes. That is what you expect by chance once you account for
which genes are rare (p = 0.37), and GATA1 sits in the middle of 291 TFs for these features.

Across all 87 TFs and seven cut-offs, no TF shows features that are specific to its targets after correcting
for rare genes and for testing many TFs. The requested fake-TF test does reject for every knockdown that
changes anything, because random control groups change nothing. So it shows the knockdown had an effect, not
that the features know the TF's targets.

The tests can see a real signal when there is one: 8 planted GATA1 targets are found 93% of the time. But the
old statistic stops counting at 5 targets. With a gene-length-matched null, that cap makes ChIP-type signal
almost invisible. Counting past 5 fixes it.
