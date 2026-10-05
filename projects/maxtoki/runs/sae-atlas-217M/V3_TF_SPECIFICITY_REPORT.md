# V3-2 TF-specificity report: v3 layer-5 SAE features vs TF target sets, on correctly encoded K562 cells

Date: 2026-10-02. Model: MaxToki-217M (HF safetensors, float32, Apple MPS, eager attention).
torch 2.11.0, transformers 5.5.4. SAE: the v3 layer-5 SAE (`outputs/v3_sae/layer_05/sae_final.pt`,
retrained on correctly encoded inputs, see `V3_SAE_REPORT.md`). Inputs: counts / Geneformer gene median,
from `setup/inputs_v3.py`. Same design, same cells and same statistics code as
`V2_TF_SPECIFICITY_REPORT.md`. Every number below comes from a file in
`outputs/v3_tf_specificity/` (section 9). "Cell bootstrap" means cells are resampled with replacement,
knockdown and control groups separately.

## 0. Results first

1. **Still no TF-specific features.** I tested 87 knocked-down TFs (20 with ChIP-seq target sets) at 7
   effect cut-offs. **No TF passes BH (q < 0.05 across TFs)** under the count-matched gene-swap null, the
   count-and-length-matched null, the K-matched fake-TF null or the other-knockdown null, at any cut-off.
   This is the same verdict as v2. Three single nominal hits (p < 0.05) exist; none survives BH
   (smallest q = 0.19; section 5.4).
2. **GATA1 at the deployed cut-off (0.5).** GATA1 changes **7 features** (v2: 5). The best one (feature
   3843) has **4 of its top-20 genes** among the 221 GATA1 ChIP-seq targets in the gene universe.
   - Count-matched null: p = **0.32** (expected best overlap 3.1). Count-and-length null: p = **0.43**.
   - K-matched fake TFs: p = **0.19** (Monte-Carlo SE 0.009, 2,000 draws). Other knockdowns: p = 0.059,
     which is the smallest value possible with 16 other knockdowns.
   - Deployed random-feature null: p = 0.055. GATA1 ranks **82nd of 291** TFs for these 7 features.
   - TRRUST: best overlap 1 of 11 targets, so the deployed "≥ 2 of top-20" rule fails.
   - The cell bootstrap gives the same best overlap (4) in 96.5% of 1,000 resamples and 2 in 3.5%.
     It never reaches 5 or more.
3. **The old GATA1 feature (2610, 8 ChIP targets of 20) has no counterpart in the v3 SAE.** No v3
   feature has more than 5 GATA1 ChIP targets in its top-20 (10 features have 5; none has 6 or more).
   The v3 feature whose decoder direction is closest to old 2610 has |cos| 0.19 (a random direction gives
   0.11). It shares 2 of 20 top genes. Only one old GATA1 feature has a clear v3 match: old 3167 ↔
   v3 3763 (|cos| 0.75, 6 shared top genes). Both respond to GATA1 knockdown.
4. **More TFs now pass the 0.5 effect cut-off, but it is still rare.** 17 of 87 TFs have at least one
   responding feature at 0.5 (v2: 6). Among the 20 primary TFs, 3 do (GATA1 7, MAX 1, TERF2 1; v2: GATA1
   only). With no effect cut-off (BH only), 74 of 87 TFs have responding features (v2: 70). The number of
   BH-significant features per TF agrees closely with v2 (Spearman 0.96 over 87 TFs).
5. **The deployed statistic (capped at 5 targets) now has no power for GATA1.** With 7 GATA1 features, a
   best overlap of 5 or more happens by chance with probability 0.099 under the count-matched null. So the
   capped test can never give p < 0.05. Planting 8 GATA1 ChIP targets into one GATA1 feature is detected
   with probability **0.00** (95% Wilson CI 0.00 to 0.02, 200 repeats) by the capped count-matched test,
   and **1.00** (0.98 to 1.00) by the uncapped one. In v2 the capped test still had power 0.93. Planting 2
   TRRUST targets is detected every time (1.00). The observed GATA1 negative does not come from the cap:
   the observed best overlap is 4, below the cap.
6. **The old "≥ 2 TRRUST targets in a top-20" test**: 0 of the deployed 48 TFs pass at cut-off 0.5
   (10 have responding features, 8 of them have TRRUST targets in the universe). Same verdict as the
   deployed "0/48" and v2.

All named checks pass (section 4). The inputs passed the encoding check on all 9,200 cells before any
forward pass. The live SAE codes equal the stored v3 codes exactly. The statistics code reproduces every
v2 number when fed the v2 inputs. The fake-TF p-values are calibrated. The planted signal is found by the
uncapped tests.

---

## 1. What was done, and what changed from v2

The audit (`checks/INPUT_ENCODING_AUDIT.md`) found that v2 fed MaxToki genes ranked by log1p(CP10k) /
median instead of counts / median. v2 also used the deployed SAE, which was trained on those wrong inputs.
This run repeats the v2 design with two changes and nothing else on purpose:

| Item | v2 | v3 (this run) |
|---|---|---|
| Model input | `tokenize_cell(X)`, X = log1p(CP10k) — wrong order | counts = round(expm1(X) / unit) (`inputs_v3`), ranked by counts / median; checked on every cell before its forward pass |
| SAE | deployed layer-5 SAE (`outputs/phase1`) | v3 layer-5 SAE (`outputs/v3_sae`), same site: input of block 5 = `hidden_states[5]` |
| Top-20 catalog | deployed catalog (`phase2/layer_05/feature_catalog.json`) | rebuilt from the v3 SAE with the deployed definition (section 3) |
| Cells | 9,200 (400 ref, 800 pool, 7,500 knockdown, 500 catalog) | **the same 9,200 cells** (manifest rebuilt with the v2 seeds; identical rows, groups and barcodes) |
| Mann-Whitney tie correction | tied zeros only | all ties (zeros and tied non-zero values); see below |

Why the tie correction changed. With the correct encoding, 78% of the 9,200 cells start with the same
gene (RANBP1, `ENSG00000099901`; old encoding: 14% of the catalog cells shared their most common first
gene). A feature that fires only near the start of the sequence then gets exactly the same per-cell mean
in many cells. For example, feature 3153 has one value in 2,812 cells. The v2 fast test ignored such
ties in its variance. On a 95-cell fake group this gave p-values up to 0.0066 away from scipy. I added
the full tie correction (16,594 tied feature-value pairs over 4,573 features). It now equals scipy
exactly (difference 0.0).

Everything else is unchanged: per-cell unit, two-sided Mann-Whitney U vs 400 reference cells, BH across
4,928 features, responding = q < 0.05 and |mean difference| > cut-off, cut-offs 0.5 (deployed) to 0, the
statistic (best overlap of a responding feature's top-20 with the target set, thresholds 2..5 = one
capped statistic T), the five nulls, the TF rank, the power simulation and BH across TFs.

## 2. Data

- **Cells** (Replogle K562, `replogle_concat.h5ad`): 400 reference controls, 800 pool controls (for fake
  TFs), 7,500 knockdown cells (87 TFs, 34 to 100 cells each; GATA1 95) and the 500 catalog control cells
  (= the v3 SAE training cells). `cell_manifest.csv` has dataset row, barcode and gem group. It equals the
  v2 manifest row for row.
- **Primary TFs (20, DoRothEA ChIP-seq targets):** ATF4, BRF2, CEBPZ, E2F6, GATA1, GTF2B, HINFP, MAX,
  MYBL2, SETDB1, SP2, SRF, STAT5A, TBP, TERF1, TERF2, TFDP1, THAP1, THAP11, ZNF24.
- **Gene universe:** 6,328 genes seen in the 500 catalog cells with the correct encoding (v2: 6,324; all
  v2 genes are kept, 4 are new). Detection counts agree with v2 at Spearman 0.89. GATA1 targets in the
  universe: 221 ChIP-seq (v2: 219), 11 TRRUST.
- **Knockdown check** (expm1(X) = CP10k, a label, not a model input): for the 80 TFs whose gene is in the
  6,546-gene panel, the TF's mean expression in knockdown cells is at most 13% of the reference level
  (median 0%). GATA1 and 6 other TF genes are not in the panel.
- **Forward passes:** 9,200 cells, 0.21 s per cell, 6 calls of at most 7.6 minutes, about 34 minutes in total
  (MPS, batch 1, max_len 2,048, forward pass stopped at block 5). Available memory was at least 9 GB at
  the start of each call and never fell below 3 GB.
- **Two edge-case cells.** Rows 282850 and 401789 have expm1(X) values that are already whole numbers
  (smallest value exactly 1). `inputs_v3.CountReader.counts` rejects them, because it expects "whole
  multiples of a non-integer unit". They are valid counts with factor 1. I **added** a function,
  `inputs_v3.counts_accept_unit_one`, at the end of `setup/inputs_v3.py`. It returns the same result as
  `CountReader.counts` for every other row and accepts these two. No existing function was changed. The
  first extraction call stopped on row 282850 before any forward pass of that batch (400 cells were done
  by then, with identical code otherwise).

## 3. Methods (as v2 unless stated)

**Top-20 catalog of the v3 SAE (deployed definition, `full_12layer_pipeline.py:197-235`).** Encode every
token position of the 500 catalog cells (incl. `<bos>`/`<eos>` = `<SPECIAL>`). For each feature and gene,
take the mean activation over positions where the feature is active. The top-20 genes are the 20 highest
means (no minimum count). Built from `outputs/v3_sae/codes/layer_05_topk.npz` and rebuilt from our own
forward passes (identical, section 4). All 4,928 features have 20 genes; 47 lists contain `<SPECIAL>`.
The rare-gene problem of this definition is a little smaller than in the deployed catalog: 1.1% of
top-20 slots hold a gene seen in 5 or fewer cells (deployed: 1.3%).

**Statistic.** M = best overlap of a responding feature's top-20 with the target set. T = min(M, 5) if
M ≥ 2, else 0. p = P_null(T ≥ T_obs). Also the uncapped M.

**Nulls** (all as v2): rf = K random catalog features (exact); cm = each top-20 gene swapped for a random
gene from the same detection-count bin (exact, hypergeometric convolution); cml = bins of count × gene
length tertile (cuts 21,392 and 56,262 bp); fake = random pool subsets of the knockdown size, whole
selection re-run (2,000 per size); fakeK = fake TF keeps its K most-changed features; okd = responding
features of every other knockdown with K ≥ 1.

**Rank:** cm p-value of the same features with the target set of each of 291 ChIP (≥ 20 targets) or 109
TRRUST (≥ 5) TFs; rank of the true TF. **Multiplicity:** BH across TFs per database, cut-off and null
(20 primary; all 87). No correction across cut-offs or nulls. **Power:** plant k = 1..8 true targets
into the top-20 of one random responding feature, 200 repeats per (TF, cut-off, database, k), 20
primary TFs; one seed per TF.

## 4. Checks (all pass)

| Check | Result | Pass |
|---|---|---|
| Same cells as v2 | 9,200 rows, groups, TFs and barcodes identical; catalog rows = v3 SAE training cells | yes |
| **Encoding check before every forward pass** | `inputs_v3.assert_encoding_batch` on every cell of all 368 batches of 25: 9,200 / 9,200 pass, 0 order violations (1,212 positions are exact ties put in another order, which is allowed) | yes |
| The check can tell old from new | old (v2) tokens of 100 sample cells fail it 100 / 100. Old vs new order on those cells: Spearman 0.84, top-200 overlap 0.57, kept genes 0.94 (audit: 0.84 / 0.55 / 0.92) | yes |
| Catalog-cell tokens = v3 SAE tokens | 1,019,996 / 1,019,996 positions identical | yes |
| **Live codes vs stored v3 codes** (catalog cells) | identical at all 1,019,996 positions (same 32 features, value difference 0.0); the two catalogs are identical for all 4,928 features | yes |
| `<bos>` code the same in every cell | max difference 0.0 over 9,200 cells | yes |
| **Second model path** (12 cells: 6 GATA1, 6 reference) | `LlamaForCausalLM(output_hidden_states=True).hidden_states[5]` + `TopKSAE.encode`, no hooks_v2: per-cell means equal the stored ones within 2.4e-7 (values up to 2.6); tokens identical | yes |
| **v3 stats code on v2 inputs** | reproduces all 236 testable v2 rows: K, M, T, best feature identical; p_rf, p_cm, p_cml, p_cm uncapped, E[M], p_fake, p_fakeK within 2.2e-16 | yes |
| Mann-Whitney vs scipy (float64) | full tie correction: max difference 0.0 in p (GATA1 and a 95-cell fake group); same BH sets | yes |
| Exact swap null vs Monte Carlo (GATA1's 7 features, 10,000 draws) | P(best ≥ t), t = 1..8: largest gap 0.005 (cm) and 0.007 (cml), at most 1.8 Monte-Carlo SE; expected overlaps within 0.010 | yes |
| GATA1 K and best overlap re-derived from the written tables (by gene name) | all 14 (cut-off × database) rows equal `tf_results.csv` | yes |
| **Selection false-positive rate**, 1,000 fresh random splits of the 1,200 controls (400 vs 95) | P(≥ 1 responding feature): 0.032 at cut-off 0 (95% Clopper-Pearson 0.022 to 0.045); 0.002 at 0.01; 0.001 at 0.02 and 0.05; 0 at 0.1 and above. Single-feature p-values uniform (KS p 0.03 to 0.45 for 5 fixed features; 0.36 for a random feature per split; share p < 0.05: 0.031 to 0.062) | yes |
| **Fake-TF p-values uniform** | 1,000 new fake TFs per primary TF (20,000), scored against the stored nulls; 28 settings: KS p median 0.56, min 0.040 (2 of 28 below 0.05; 1.4 expected); share of conservative p ≤ 0.05 at most 1.0%; two-sample chi-square p < 0.05 in 5 of 132 per-TF tests | yes |
| fakeK null reproduced with a new seed (7 TFs with K > 0 at cut-off 0.25) | chi-square p 0.12 to 0.98; the overlap here is counted by a second, independent implementation (by gene name) | yes |
| **Planted signal, k = 8, ChIP, matched null** | GATA1 at 0.5: 1.00 [0.98, 1.00] with the uncapped count-matched test (the capped test: 0.00, section 5.5) | yes (uncapped) |
| **GATA1 in the set** | yes: 95 knockdown cells, 221 ChIP and 11 TRRUST targets in the universe | yes |

As in v2, the fake-TF p-values of single features are not uniform across fake TFs that share the fixed
400-cell reference and 800-cell pool (KS p down to 5e-23 for one feature; 5.0% of fake TFs have ≥ 1 BH
feature at cut-off 0, v2: 10.9%). The pool itself does not differ from the reference (0 BH features; 7
features with p < 0.001 against 4.9 expected; 331 with p < 0.05 against 246). With fresh random splits
(row above) the test is calibrated. So this is the shared-pool effect v2 described, not a broken test.

## 5. Results

### 5.1 Selection

Responding features per primary TF (K). Full table: `task_data/tf_summary.tsv`; v2 numbers from
`../v2_tf_specificity/task_data/tf_summary.tsv`.

| TF | cells | BH q < 0.05 (v2) | K at 0.5 (v2) | 0.25 | 0.1 (v2) | 0.05 | 0.02 | 0.01 (v2) |
|---|---|---|---|---|---|---|---|---|
| GATA1 | 95 | 4,097 (4,206) | **7** (5) | 13 | 21 (52) | 46 | 249 | 543 (405) |
| MAX | 100 | 2,888 (3,160) | 1 (0) | 3 | 10 (16) | 20 | 95 | 304 (291) |
| THAP1 | 100 | 1,492 (1,307) | 0 (0) | 1 | 3 (1) | 6 | 18 | 93 (157) |
| HINFP | 100 | 1,435 (1,477) | 0 (0) | 1 | 3 (3) | 6 | 21 | 87 (142) |
| TFDP1 | 100 | 725 (720) | 0 (0) | 0 | 0 (1) | 4 | 4 | 36 (107) |
| CEBPZ | 100 | 680 (418) | 0 (0) | 0 | 2 (0) | 2 | 7 | 24 (7) |
| GTF2B | 34 | 510 (477) | 0 (0) | 1 | 2 (2) | 7 | 24 | 90 (102) |
| MYBL2 | 81 | 266 (417) | 0 (0) | 0 | 4 (2) | 5 | 7 | 17 (29) |
| TERF2 | 47 | 250 (162) | 1 (0) | 1 | 1 (1) | 3 | 7 | 21 (16) |
| TBP | 100 | 186 (63) | 0 (0) | 1 | 1 (0) | 4 | 8 | 20 (1) |
| 10 others | 49–100 | 0–22 (0–11) | 0 | 0 | 0 | 0–1 | 0–2 | 0–5 |

Per-cell feature means are small, as in v2 (median over features of the reference mean 0.012; v2: 0.008).
At 0.5, the 17 TFs with responding features are AATF, ATF5, BDP1, CDC5L, ECD, GATA1 (7), GTF2A1, MAX,
MED1 (2), PDCD11 (2), PHB2 (2), RUVBL1, SNIP1, TAF1, TAF5, TERF2, TSG101 (2) (v2: CDC5L, GATA1, MED1,
PHB2, RUVBL1, TAF1).

### 5.2 GATA1 detail

Responding features at cut-off 0.5 (`gata1_bootstrap.json`, `gata1_detail.json`). Effect = mean
difference of per-cell feature means, knockdown minus reference. CI = cell bootstrap, 1,000 resamples
(seed 20263027; selection re-run in each resample with scipy's Mann-Whitney, which handles the ties from
duplicated cells).

| v3 feature | Effect [95% CI] | Cohen's d | Selected in bootstraps | ChIP targets in top-20 | Closest old feature (|cos|) |
|---|---|---|---|---|---|
| 4905 | 1.695 [1.568, 1.804] | 3.50 | 100% | 1 (GNA12) | 2288 (0.19) |
| 2665 | 0.756 [0.588, 0.917] | 1.19 | 100% | 1 (SUCO) | 3167 (0.45) |
| 1725 | 0.738 [0.687, 0.781] | 3.43 | 100% | 2 (FBXW11, ACER3) | 4022 (0.18) |
| 4264 | 0.702 [0.634, 0.767] | 2.71 | 100% | 1 (RNF220) | 4264 (0.25) |
| 3763 | 0.597 [0.460, 0.741] | 1.13 | 92% | 1 (DIS3L2) | **3167 (0.75)** |
| **3843** | 0.561 [0.496, 0.626] | 2.37 | 97% | **4** (MBD5, FRYL, CDKAL1, TBL1XR1) | 3843 (0.28) |
| 892 | 0.506 [0.434, 0.579] | 1.99 | 56% | 0 | 2700 (0.28) |

- Bootstrap: K at 0.5 is 7 in the median (95% range 5 to 7). Only one other feature is ever selected at
  0.5. The best ChIP overlap is 4 in 96.5% of resamples and 2 in 3.5%.
- The four ChIP targets of feature 3843 are seen in 2, 16, 23 and 100 of the 500 catalog cells. So two of
  them are rare genes, which is why the count-matched null expects a best overlap of 3.1 for these 7
  features.
- At lower cut-offs the best GATA1 ChIP overlap is 4 or 5 (features 3347, 2464, 1455). Each time it is
  close to what the count-matched null expects for that many features (3.5 to 5.4), p_cm 0.47 to 0.94.
- **Old features.** Closest v3 feature (by decoder direction) to each old GATA1 feature: 628 → 4905
  (|cos| 0.13), 1334 → 241 (0.11), 2006 → 4427 (0.22), **2610 → 3843 (0.19)**, 2627 → 1016 (0.15),
  3167 → 3763 (0.75). Random directions give a median best match of 0.11. So only old 3167 survives as
  a feature. Old 2610 (8 of 20 ChIP targets) has no counterpart: its best decoder match shares 2 of 20 top
  genes, and its best gene-list match (v3 2935) shares 3 genes and does not respond to GATA1 (effect
  −0.027). Note: both SAE sets start from the same seed-42 weights, so a small cosine between features
  with the same number (4264 ↔ 4264, 3843 ↔ 3843) can come from the shared start (`V3_SAE_REPORT.md`
  3.3).
- **No v3 feature looks like old 2610.** Over all 4,928 v3 features, the most GATA1 ChIP targets in a
  top-20 is 5 (10 features). None of those 10 features passes the 0.5 cut-off for GATA1.

### 5.3 TF tests at the deployed cut-off (0.5)

Among the 20 primary TFs, GATA1, MAX and TERF2 are testable with the ChIP set; GATA1 and MAX with TRRUST.

| GATA1, cut-off 0.5 | ChIP-seq (v2) | TRRUST (v2) |
|---|---|---|
| K / best overlap M / T | 7 / 4 / 4 (5 / 3 / 3) | 7 / 1 / 0 (5 / 0 / 0) |
| rf (deployed null) | 0.055 (0.257) | 1 (1) |
| cm (count-matched) | **0.318**, expected M 3.11 (0.371) | 1 (1) |
| cml (count + length) | 0.431 (0.564) | 1 (1) |
| fake (as specified) | 0.0005 = floor; BH q 0.005 across the 20 primary TFs (v2: 0.0005, q 0.010) | 1 (1) |
| fakeK | 0.189 ± 0.009 Monte-Carlo SE (0.864) | 1 (1) |
| okd (16 other knockdowns; smallest possible p 0.059) | 0.059 (0.667, 5 knockdowns) | 1 (1) |
| rank among TFs | 82 / 291 (151 / 291) | – |

MAX (1 feature, best ChIP overlap 2): p_cm 0.33, rank 73/291. TERF2 (1 feature): ChIP overlap 1, T = 0.
The fake null is still degenerate: random control groups of these sizes never select a feature at 0.25
or above (0 of 5,000 fake TFs at 0.5), so every knockdown with T ≥ 2 gets the floor p. It shows the
knockdown changed something, not that the features know the TF's targets. The okd p of 0.059 is the
floor: no other knockdown's responding features reach 4 GATA1 ChIP targets at this cut-off. With 16
knockdowns this cannot reach 0.05, and it does not hold at lower cut-offs (0.088 to 0.57).

### 5.4 Sensitivity to the effect cut-off (20 primary TFs)

Full tables: `summary_tables.md` (Tables A, A2, B, C).

| cut-off | db | testable TFs | median K | T ≥ 2 | BH fake | BH cm | BH cml | BH fakeK | BH okd | true TF in top 5% |
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

All 87 TFs (Table A2): 0 TFs pass BH under cm, cml, fakeK or okd at any cut-off. In v2, CEBPZ passed BH
under fakeK only (q 0.020 at cut-off 0.01); it does not in v3 (cut-off 0.01: K = 24, best overlap 3,
p_cm 0.53).

**The three nominal hits** (p < 0.05 under a fair null; none survives BH):
- BDP1, TRRUST, cut-off 0 (K = 1,084): best overlap 2, p_cm 0.009, BH q 0.52 (87 TFs); p_fakeK 0.27,
  p_okd 0.36.
- SETDB1, ChIP, cut-off 0 (K = 22): best overlap 4, p_cm 0.036, p_cml 0.033, p_fakeK 0.046; q_cm 0.66,
  q_fakeK 0.44; p_okd 0.54.
- TBP, ChIP, cut-off 0.05 (K = 4): best overlap 3, p_fakeK 0.017 (q 0.19), but p_cm 0.31, p_okd 0.40.
Each appears at one cut-off and under one or two nulls only. With 7 cut-offs × 2 databases × 4 fair
nulls × 87 TFs and no correction across cut-offs, a few such p-values are expected. I do not read them
as findings.

### 5.5 Power (planted targets)

Mean detection probability at alpha 0.05. One TF: 95% Wilson CI over 200 repeats. Several TFs: 95% CI
from a bootstrap over TFs. Full table: `summary_tables.md` Table D, `power_summary.json`.

| cut-off | db | k | TFs | rf | cm | cm uncapped | cml | cml uncapped | fakeK |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | ChIP | 4 | GATA1 | 0.88 [0.82, 0.91] | **0.00 [0.00, 0.02]** | 0.29 [0.23, 0.36] | 0.00 | 0.29 | 0.88 |
| 0.5 | ChIP | 8 | GATA1 | 1.00 [0.98, 1.00] | **0.00 [0.00, 0.02]** | **1.00 [0.98, 1.00]** | 0.00 | 1.00 | 1.00 |
| 0.5 | TRRUST | 2 | GATA1 | 1.00 [0.98, 1.00] | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.5 | ChIP | 8 | 3 (GATA1, MAX, TERF2) | 1.00 | 0.67 [0.00, 1.00] | 1.00 [1.00, 1.00] | 0.66 | 1.00 | 1.00 |
| 0.1 | ChIP | 8 | 9 | 1.00 [1.00, 1.00] | 0.89 [0.67, 1.00] | 1.00 [1.00, 1.00] | 0.86 [0.63, 1.00] | 1.00 | 1.00 |
| 0 | ChIP | 8 | 16 | 0.50 [0.25, 0.75] | 0.56 [0.31, 0.81] | 1.00 [1.00, 1.00] | 0.56 | 0.93 [0.82, 1.00] | 0.50 |

What this means:
- **The cap at 5 kills the capped tests for GATA1.** Under the count-matched null, P(best of 7 GATA1
  features ≥ 5) = 0.099 (exact; Monte Carlo 0.099). A capped T can be at most 5, so its p can never go
  below 0.099. In v2 (5 features, other gene mix) this probability was 0.026 and the capped test had
  power 0.93. The uncapped test finds 8 planted targets every time (P(best ≥ 8) = 0.0006).
- The observed GATA1 result is not hidden by the cap: the observed best overlap is 4, and the uncapped
  p equals the capped p (0.32).
- At lower cut-offs the capped tests lose power for the same reason (0.56 at cut-off 0); the uncapped ones
  keep 0.93 to 1.00 at k = 8.
- TRRUST: 2 planted targets are found every time when the TF has at least 2 TRRUST targets in the
  universe. TFs with fewer can never pass (mean power 0.75 to 0.88 at lower cut-offs).
- The okd null has no power at 0.5 (16 other knockdowns, floor 0.059 > 0.05).

### 5.6 Old TRRUST "≥ 2 of top-20" test on the v3 selection (`old_trrust_test.csv`)

| cut-off | deployed 48 TFs: with responding features / pass | 20 primary | all 87 |
|---|---|---|---|
| 0.5 | 10 / 0 (v2: 3 / 0) | 3 / 0 | 17 / 0 |
| 0.25 | 17 / 0 | 7 / 0 | 34 / 0 |
| 0.1 | 25 / 0 | 9 / 0 | 47 / 0 |
| 0.02 | 30 / 0 (v2: 25 / 1, GATA1) | 12 / 0 | 57 / 0 |
| 0 | 41 / 2 (BDP1, ERCC2) (v2: 39 / 1) | 16 / 1 (TFDP1) | 74 / 3 |

Every pass has p_rf ≥ 0.05 (column `n_specific_and_p_rf_lt_0.05` = 0 in all rows). GATA1 no longer meets
the old rule at any cut-off (best TRRUST overlap 1; v2: 2 at cut-offs ≤ 0.02).

## 6. What changed versus the deployed result and v2

| Claim | Deployed | v2 (wrong encoding, old SAE) | v3 (this run) |
|---|---|---|---|
| GATA1 responding features at 0.5 | 628, 1334, 2006, 2610, 2627 | 628, 1334, 2006, 2627, 3167 (2610 out at 0.461) | 7 new features: 4905, 2665, 1725, 4264, 3763, 3843, 892 (feature numbers do not carry over) |
| Best GATA1 ChIP overlap at 0.5 | 8 of 20 (2610) | 3 of 20 (3167) | 4 of 20 (3843) |
| Its p-value | 0.030 vs random features | 0.26 rf, 0.37 cm, 0.56 cml, 0.86 fakeK | 0.055 rf, **0.32 cm**, 0.43 cml, 0.19 fakeK, 0.059 okd (floor) |
| GATA1 rank among 291 TFs | – | 151 | 82 |
| A feature with ≥ 8 GATA1 ChIP targets exists | yes (2610) | yes (2610, outside the 0.5 selection) | **no** (max 5 in any of 4,928 features) |
| TFs passing BH under a fair null | – | 0 (cm, cml, okd); CEBPZ under fakeK only | **0 under all four** |
| Old TRRUST "≥ 2" rule, deployed 48, cut-off 0.5 | 0 / 48 | 0 (3 testable) | 0 (10 with responding features) |
| Power of the capped count-matched test, GATA1, 8 planted ChIP targets | "≈ 60%" (a false-positive rate) | 0.93 | **0.00**; uncapped 1.00 |
| TFs with ≥ 1 responding feature at 0.5 (of 87) | – | 6 | 17 |

In short: the correct inputs and the retrained SAE change which features respond and their numbers, but
not the verdict. The one positive claim in the deployed run (GATA1 feature 2610) was already gone in v2,
and in v3 there is not even a feature like it.

I cannot say how much of the change comes from the input fix and how much from the SAE retrain. Both
changed together. The old SAE fits the new activations badly at layer 5 (FVE 0.29, `V3_SAE_REPORT.md`),
and running the new SAE on wrongly encoded inputs would mean a forward pass on wrong inputs, which this
workflow forbids.

## 7. What I could NOT do, and limits

- **The fake-TF null as specified still cannot measure specificity** (control groups select nothing at
  cut-offs ≥ 0.25). I report it with the K-matched and other-knockdown nulls, as in v2.
- **Fake TFs share one reference group and one pool.** Their p-values are calibrated given that pool
  (section 4), but they are not independent draws from the population.
- No correction across the 7 cut-offs or across the nulls. BH is only across TFs.
- The rank test is descriptive (target sets of different TFs differ in size and gene mix).
- Power was simulated for the 20 primary TFs only.
- One SAE training seed (from V3-1), layer 5 only. I did not test whether another seed gives the same
  GATA1 features.
- GATA1 and six other TF genes are not in the 6,546-gene panel, so their knockdown strength cannot be
  checked here.
- The deployed top-20 definition (mean activation when active, no minimum count) is kept, so its
  rare-gene problem stays in the statistic. Only the matched nulls handle it.
- I could not separate the effect of the input fix from the effect of the SAE retrain (section 6).
- The decoder-cosine bridge to old features is only a guide: both SAE sets share their starting weights.
- Gene length is the genomic span from `biotensor/data/genemanifold/gene_pos.json` (54 of 6,328 genes have
  none; put in the middle tertile).

## 8. How to reuse the data

`outputs/v3_tf_specificity/task_data/` has the same layout as the v2 task data, with a `README.md` that
defines every file: gene universe (detection counts, lengths, bins), the v3 top-20 catalog, target sets
in the universe, responding features with signed effects, p and q for all 87 TFs, per-TF summaries, all
test results, the cell manifest and the per-cell feature means for all 9,200 cells
(`cell_feature_means.npz`). The token ids fed to the model are stored per cell in `cells/summ_*.npz`.

## 9. Files

Scripts (`runs/sae-atlas-217M/scripts/`):
- `v3_tf_specificity_extract.py`: `prepare` (manifest = v2 manifest, order check, negative control) and
  `extract` (encoding check, MaxToki forward passes to block 5, v3 SAE per-cell summaries; MPS, chunked).
- `v3_tf_specificity_taskdata.py`: gene universe, target sets, v3 top-20 catalog from stored and live codes.
- `v3_tf_specificity_stats.py`: selection (full tie correction), all nulls, rank, power, BH, calibration
  (CPU, resumable; caches in `stats_cache/`, fake nulls in `fake_null/`).
- `v3_tf_specificity_calib_check.py`: calibration checks A to D.
- `v3_tf_specificity_replay_v2.py`: the v3 stats code on the v2 inputs.
- `v3_tf_specificity_verify.py`: second model path for 12 cells; GATA1 numbers from the written tables.
- `v3_tf_specificity_summary.py`: GATA1 bootstrap and detail, old-feature bridge, tables, v2 comparison,
  merged `run_config.json`, task-data README.
- Shared code: `setup/inputs_v3.py` (one function added: `counts_accept_unit_one`), `setup/hooks_v2.py`
  and `setup/topk_sae.py` (unchanged).

Outputs (`runs/sae-atlas-217M/outputs/v3_tf_specificity/`, 896 MB, of which 696 MB per-cell files):
- `tf_results.csv`, `power.csv`, `power_summary.json`, `old_trrust_test.csv`, `summary.json`,
  `summary_tables.md`, `summary_counts.csv`, `gata1_bootstrap.json`, `gata1_detail.json`, `v2_vs_v3.json`,
  `knockdown_efficiency.json`, `calibration.json`, `calibration_extra.json`
- `checks/replay_v2.json`, `checks/verify.json`, `prepare.json`
- `run_config.json` (merged: device, versions, seeds, cell IDs, feature IDs, input sha256, code sha256,
  wall time per chunk, encoding-check result), `run_config_extract.json`, `run_config_stats.json`
- `cell_manifest.csv`, `cell_manifest_meta.json`, `extract_progress.json`, `cells/`, `fake_null/`,
  `stats_cache/`, `task_data/`

## Plain-words summary

v2 gave MaxToki genes in the wrong order, and its SAE was trained on that wrong order. I redid the
TF-specificity test with the right gene order and the retrained SAE, on the same 9,200 cells and with the
same statistics. Every cell passed the order check before it went into the model.

The answer does not change. No TF has SAE features that are specific to its target genes once rare genes
and testing many TFs are taken into account. This holds for all 87 TFs, all 7 cut-offs and all four fair
nulls.

GATA1 knockdown now changes 7 features. The best one holds 4 of GATA1's ChIP-seq targets in its top 20
genes. That is about what chance gives for these genes (p = 0.32), and GATA1 sits at rank 82 of 291 TFs.
The old GATA1 feature with 8 targets has no match in the new SAE. No new feature holds more than 5 GATA1
targets.

The checks pass: the model inputs are right, two ways of getting the SAE codes agree exactly, the
statistics code gives back every v2 number on v2 data, and the p-values behave correctly under the null.
One warning for later work: the old statistic stops counting at 5 targets. For GATA1 in v3, that cap
means it could not detect even 8 planted targets. Counting past 5 detects them every time.
