## Spectral geometry: effective rank and cross-sample CKA by layer (MaxToki-217M, Tabula Sapiens immune)

This analysis asks two questions about the per-gene vectors inside MaxToki-217M. First: how many directions do these vectors use at each layer (effective rank), and does the fall with depth come from training or from the architecture? A model with the same architecture and random weights answers the second part. Second: how much does the gene geometry of a layer change when each gene's average is taken over a different set of cells (cross-sample CKA)?

### Data

- **Model.** MaxToki-217M, Hugging Face safetensors, float32, eager attention, Apple MPS. torch 2.11.0, transformers 5.5.4, numpy 1.26.4. The model has 11 decoder blocks and a hidden size of 1,232.
- **Random-initialised models.** Three models with the same configuration (`LlamaConfig` read from the checkpoint). For seed s = 0, 1, 2: `torch.manual_seed(s)`, then `LlamaForCausalLM(config)`. This is the transformers default initialisation: weights normal(0, 0.02), norm weights 1. The state-dict sha256 of each model is recorded and checked again in every job chunk.
- **Cells.** Tabula Sapiens immune (`tabula_sapiens_immune.h5ad`, 592,317 cells). Counts come from `raw/X` (integer counts, before ambient-RNA removal). Three samples of 2,000 cells:
  - A: `numpy.random.default_rng(42).choice(592317, 2000, replace=False)`. These are the cells of the deployed analysis (cell types match in 2,000 of 2,000).
  - B and C: draws with `default_rng(43)` and `default_rng(44)`. Any cell that also occurs in an earlier sample was replaced by a new random cell that no other sample uses (`default_rng(20261002)` for B, `default_rng(20261003)` for C). 6 cells were replaced in B and 16 in C.
  - No cell is shared between A, B and C.
  - Smart-seq2 cells: 73 in A, 80 in B, 80 in C. The rest are droplet (UMI) cells. All cells were kept.
- **Genes.** The 1,500 genes of the deployed analysis (`projects/maxtoki/runs/spectral-geometry-217M/outputs/phase0/gene_features.csv`). Applying its rule to the same cells gives the same 1,500 genes. The rule: genes in the MaxToki vocabulary with a TRRUST, STRING or lineage-marker annotation, non-zero variance and expression in more than 0.5% of sample-A cells (3,073 genes), ranked by variance across sample-A cells; the top 1,500 are kept. The variance was computed on log1p(CP10k) of the atlas's stored `X`, which is already log-transformed.
- **Second gene panel (sensitivity only).** The same rule applied to log1p(CP10k) of the raw counts. It shares 1,426 of its 1,500 genes with the main panel.
- All 1,500 genes are seen in every sample. A gene's vector averages only the cells in which it is among the 2,046 kept genes: a median of 493.5 cells in A, 482 in B and 485 in C (range 38–1,854). In a 1,000-cell half of a sample the median is 240–247.

### Method

1. **Input encoding** (section 00). For each cell: counts / cell total × 10,000 / Geneformer gc104M gene median; rank from high to low; keep the first 2,046 genes; add `<bos>` and `<eos>` (at most 2,048 tokens). Every cell is checked before its forward pass (Checks).
2. **Forward pass.** One cell per sequence, batch 1. `output_hidden_states=True` gives 12 states. L0 is the token embedding. L1 to L10 are the outputs of blocks 0 to 9. L11 is the final RMSNorm applied to the output of block 10.
3. **Per-gene vectors.** At each layer, each panel gene's vector is the mean of its hidden state over all cells in which it is among the kept genes. Sums are stored per block of 200 cells, so the mean over any set of blocks can be formed exactly.
4. **Jobs.** Trained model on A, B and C. Random seeds 0, 1 and 2 on A. Random seed 0 also on B and C (for the cross-sample CKA control). This is 8 × 2,000 = 16,000 forward passes.
5. **Effective rank (ER).** Take the 1,500 × 1,232 matrix of per-gene vectors at one layer and centre its columns. With singular values σ_i, let p_i = σ_i² / Σ σ_j². ER = exp(−Σ p_i log p_i). ER is a number of dimensions; here it can be at most 1,232. The metric code (`_effective_rank`, `_svd_and_metrics`) is copied from `projects/maxtoki/runs/spectral-geometry-217M/scripts/phase1_svd.py`.
6. **Other measures from the same code.**
   - Participation ratio: (Σ σ_i²)² / Σ σ_i⁴.
   - Share of variance on the top direction: σ_1² / Σ σ_i².
   - TwoNN local intrinsic dimension (Facco et al., 2017): 500 genes drawn with a fixed seed; for each, μ = (distance to its 2nd-nearest gene) / (distance to its nearest gene); the slope of −log(1 − F(μ)) against log μ over the lowest 90% of μ. TwoNN describes the dimension of the structure near each gene, not the global spread.
7. **Feature-shuffle null.** Shuffle each of the 1,232 columns across genes, independently (one numpy stream), then compute ER. Computed at every layer.
8. **Compression summaries.** ER(L1) / ER(L11); share of ER lost from L1 to L11 = 1 − ER(L11) / ER(L1); the same from L0; Spearman ρ of ER with layer index.
9. **Cross-sample CKA.** Linear CKA (Kornblith et al., 2019) between the per-gene matrices X and Y of two samples at the same layer, rows matched by gene, columns centred: CKA = ‖XᵀY‖²_F / (‖XᵀX‖_F ‖YᵀY‖_F). 1 means the same geometry up to rotation and overall scale. The value reported is the mean over the three pairs (A–B, A–C, B–C), for L1 to L11. L0 is left out: a gene's L0 vector is its embedding row, which is the same in every sample, so CKA is 1 by construction. Done for the trained model and random seed 0. Code: `linear_cka`, copied from `projects/maxtoki/runs/spectral-geometry-217M/scripts/phase8_stability.py`.
10. **Split-half CKA.** Inside each sample: per-gene means over blocks 0–4 against blocks 5–9 (1,000 against 1,000 cells). Done for all 8 jobs. It gives a CKA control for random seeds 1 and 2 without more forward passes, and it shows how 1 − CKA changes when each gene's average uses about half as many cells (in sample A, a median of 246–247 instead of 493.5).
11. **Intervals.** Delete-one-group jackknife. SE = sqrt((g − 1)/g × Σ (θ_i − θ̄)²); 95% CI = estimate ± t(0.975, g − 1) × SE.
    - Over cells: each sample is split into 10 random blocks of 200 cells (`default_rng(20261005 + k)`, k = sample index). For the three-sample CKA, block j is deleted from all three samples at once. g = 10.
    - Over genes (CKA only): 30 random groups of 50 genes (`default_rng(20261007)`). g = 30.
    - The ER intervals use the eigenvalue route (Verification, first row). The intervals are centred on the estimates and are not bias-corrected.

### Checks

| Check | Result |
|---|---|
| Encoding check before every forward pass (`tokenize_counts(check=True)` and `assert_encoding_batch` on every 50-cell batch; the tokens must also equal the tokens saved when the samples were prepared) | 16,000 of 16,000 cells in 320 batches pass |
| Counts are whole numbers (every input row of `raw/X`) | 6,000 of 6,000 rows |
| Metric code identical to the code it was copied from (`phase1_svd.py`, `phase8_stability.py`; abstract syntax tree comparison) | 5 of 5 functions identical |
| Samples disjoint; A equals the stated draw | pairwise overlap 0, 0, 0; A identical |
| L0 per-gene mean equals the model's embedding row (L0 does not depend on the input) | largest relative difference 1.6 × 10⁻⁶ |
| Model health: next-gene loss (nats per gene token) | trained, UMI cells: 3.28 in A, B and C; trained, Smart-seq2 cells: 7.5–7.7; random models: 10.16. A uniform guess over the 20,275-token vocabulary gives 9.92. |
| Memory | free memory never below 3.8 GB (guard at 3 GB) |

**Why jackknife intervals, not bootstrap.** Percentile bootstraps were also computed (`analysis/cka.json`). Both are shifted for this design:
- Drawing genes with replacement puts identical rows into both matrices, which pushes CKA up. For the trained model at L11 the estimate 0.9962 sits at the lower edge of the gene-bootstrap interval [0.9962, 0.9967].
- Drawing 200-cell blocks with replacement leaves about 63% distinct cells. The averages get noisier and CKA goes down: trained L11 [0.9903, 0.9943], random [0.918, 0.948], both below their estimates (0.9962 and 0.9647).

The jackknife never copies a gene or a cell. Its width agrees with two other estimates (Verification). For the same reason no cell bootstrap was used for ER.

### Results

**Main findings.**
- The trained model's ER is 561 at L0 and 328 at L1. It then falls, not quite steadily (it rises at L2, L5 and L8), to 227 at L11 [95% CI 224.8–229.5]. The two other samples give 225.6 and 226.3.
- Three random-weight models (seeds 0–2, on sample A) fall from 816–817 (L0) to 642–645 (L1) and 390–421 (L11).
- From L1 to L11 the random models lose 34.6–39.6% of their ER; the trained model loses 30.8–31.3%. Every random run loses more than every trained sample. The 95% cell-block jackknife intervals do not overlap (trained at most 31.8%, random at least 32.5%). These intervals cover cell sampling only, not variation between seeds. So the fall of ER from L1 to L11 is a property of the architecture with these inputs and the default initialisation. Training adds extra drops at the first block and at the final norm (see "Where the trained and random models differ").
- Training lowers ER at every layer, from the embedding table on (561 against 816–817). Its first block removes 41.5% of ER (random: 21.0–21.3%).
- Local intrinsic dimension (TwoNN) falls from 102 to 5.3 in the trained model. In the random models it falls from 204–207 to 107–120.
- The feature-shuffle null stays far above the real ER in every model and layer (at L11: 639–646 against 226–227 in the trained model; 809–812 against 390–421 in the random models). It cannot tell learned from architectural compression.
- Cross-sample CKA, trained: 0.9997 (L1) to 0.9962 (L11). Random seed 0: 0.9938 to 0.9647. With 1,000-cell halves instead of 2,000-cell samples (a median of about 246 instead of 493.5 cells behind each gene's average in sample A), 1 − CKA at L11 nearly doubles in the trained model (0.0038 to 0.0073) and rises ×1.7 in random seed 0 (0.035 to 0.059). So for the trained model most of the gap from 1 is sampling noise in the per-gene averages.

**ER per layer** (`analysis/per_layer_metrics.csv`, `analysis/per_layer_metrics_optional.csv`; 1,500 genes, all cells; rounded to whole dimensions):

| Model / sample | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Trained, A | 561 | 328 | 355 | 343 | 302 | 304 | 275 | 261 | 273 | 267 | 266 | 227 |
| Trained, B | 561 | 328 | 355 | 343 | 302 | 303 | 275 | 261 | 273 | 267 | 266 | 226 |
| Trained, C | 561 | 328 | 355 | 343 | 302 | 304 | 276 | 262 | 273 | 267 | 266 | 226 |
| Random seed 0, A | 816 | 642 | 649 | 624 | 589 | 556 | 518 | 482 | 452 | 419 | 394 | 398 |
| Random seed 1, A | 817 | 645 | 650 | 625 | 583 | 546 | 506 | 464 | 431 | 403 | 383 | 390 |
| Random seed 2, A | 817 | 645 | 652 | 632 | 605 | 570 | 538 | 504 | 470 | 443 | 413 | 421 |
| Random seed 0, B | 816 | 643 | 650 | 626 | 593 | 562 | 526 | 490 | 461 | 428 | 404 | 411 |
| Random seed 0, C | 816 | 642 | 650 | 625 | 592 | 560 | 525 | 488 | 459 | 425 | 400 | 406 |

Jackknife SE over cells (dimensions): trained 0.1 at L1, 0.1–0.6 at L2–L10, 0.8–1.0 at L11; random 0.2–0.3 at L1, rising to 4.3–6.1 at L11. L0 does not depend on the cells (SE 0). The spread across the three disjoint samples is of the same size: trained L11 225.6–227.2 (SD 0.8); random seed 0 L11 397.6–410.6 (SD 6.6).

**Compression summary** (`analysis/intervals.json`, `analysis/summary.json`; 95% jackknife CI over 10 blocks of 200 cells):

| Model / sample | ER L0 | ER L1 | ER L11 [95% CI] | L1/L11 [95% CI] | Share lost L1 → L11 [95% CI] | L0/L11 [95% CI] | Spearman ρ, L0–L11 | Feature-shuffle ER at L11 |
|---|---|---|---|---|---|---|---|---|
| Trained, A | 560.8 | 328.1 | 227.2 [224.8, 229.5] | 1.44 [1.43, 1.46] | 30.8% [30.0, 31.5] | 2.47 [2.44, 2.49] | −0.930 | 645.6 |
| Trained, B | 560.8 | 328.2 | 225.6 [223.9, 227.3] | 1.45 [1.44, 1.47] | 31.3% [30.7, 31.8] | 2.49 [2.47, 2.50] | −0.930 | 639.0 |
| Trained, C | 560.8 | 328.1 | 226.3 [224.3, 228.2] | 1.45 [1.44, 1.46] | 31.0% [30.4, 31.6] | 2.48 [2.46, 2.50] | −0.930 | 641.2 |
| Random seed 0, A | 815.9 | 642.1 | 397.6 [386.7, 408.6] | 1.61 [1.57, 1.66] | 38.1% [36.4, 39.8] | 2.05 [2.00, 2.11] | −0.986 | 810.3 |
| Random seed 1, A | 816.8 | 644.9 | 389.8 [376.0, 403.5] | 1.65 [1.60, 1.71] | 39.6% [37.4, 41.7] | 2.10 [2.02, 2.17] | −0.986 | 809.3 |
| Random seed 2, A | 816.8 | 644.6 | 421.4 [408.2, 434.6] | 1.53 [1.48, 1.58] | 34.6% [32.5, 36.7] | 1.94 [1.88, 2.00] | −0.986 | 812.1 |
| Random seed 0, B | 815.9 | 642.9 | 410.6 [400.9, 420.3] | 1.57 [1.53, 1.60] | 36.1% [34.7, 37.6] | 1.99 [1.94, 2.03] | −0.986 | 810.9 |
| Random seed 0, C | 815.9 | 642.4 | 405.7 [394.3, 417.1] | 1.58 [1.54, 1.63] | 36.8% [35.1, 38.6] | 2.01 [1.95, 2.07] | −0.986 | 810.1 |

Spearman ρ is the rank correlation of ER with the layer index (12 layers). p = 1 × 10⁻⁵ for the trained model and 4 × 10⁻⁹ for the random models. From L1 to L11 only: ρ = −0.909 (trained) and −0.982 (random).

**Where the trained and random models differ** (ER, sample A for the random seeds; all from `analysis/per_layer_metrics.csv`):

| Step | Trained (A / B / C) | Random (seeds 0–2 on A; seed 0 on B, C) |
|---|---|---|
| L0 → L1 (first block) | −41.5% in all three | −21.0% to −21.3% |
| L1 → L11 | −30.8% to −31.3% | −34.6% to −39.6% |
| L10 → L11 (the step that applies the final norm) | −14.6% to −15.1% | +1.0% to +2.1% |
| L0 → L11 | −59.5% to −59.8% | −48.4% to −52.3% |

- Measured from L0, the trained model loses more (59.5–59.8% against 48.4–52.3%). The whole difference sits in the first block.
- After L1 the random models lose more.
- The learned norm weights are one possible reason for the trained model's last-step drop. This was not tested.

**Other measures** (same matrices and code; `analysis/per_layer_metrics.csv`, `analysis/per_layer_metrics_optional.csv`):

| Measure | Trained (A / B / C) | Random (seeds 0–2 on A; seed 0 on B, C) |
|---|---|---|
| TwoNN local dimension, L0 → L11 | 102 → 5.3 / 5.3 / 5.3 | 204–207 → 107–120 |
| TwoNN, L1 to L11 range | 82 at L1 falling steadily to 5.3 (sample A) | 48–121, mostly about 100 |
| Participation ratio, L1 → L11 | 243 → 119 / 116 / 117 | 487–489 → 70–93 |
| Share of variance on the top direction, L1 → L11 | 1.8% → 4.0% | 0.5% → 7.9–10.1% |

- Random-weight models put more and more variance on a few directions with depth. Their participation ratio falls more than the trained model's, and one direction grows to about 10% of the variance.
- But their genes stay locally high-dimensional (TwoNN about 100).
- The trained model is the opposite: its global spread falls less, but its genes sit on a locally low-dimensional structure (TwoNN about 5).
- These are observations. Why they differ was not tested.

**Feature-shuffle null** (ER after shuffling each column across genes):

| Model | L1 to L10 | L11 |
|---|---|---|
| Trained | 780–791 (sample A) | 639.0–645.6 (A, B, C) |
| Random | 809–817 (seeds 0–2, sample A) | 809.3–812.1 (all five random jobs) |

The null is far above the real ER in both the trained and the random models, at every layer. It is not near the maximum of 1,232 either. So it cannot separate compression learned in training from compression that the architecture produces with random weights. That needs the random-weight models.

**Sensitivity** (ER at L1 / L6 / L11; `analysis/report_tables.md`, Table 3):

| Setting | Trained, A | Random seed 0, A |
|---|---|---|
| Main (1,500 genes, all cells) | 328.1 / 275.2 / 227.2 | 642.1 / 518.4 / 397.6 |
| UMI cells only (73 Smart-seq2 cells dropped) | 328.0 / 274.2 / 227.4 | 641.6 / 514.9 / 392.5 |
| Second gene panel (picked on log1p(CP10k) of raw counts) | 332.4 / 278.3 / 228.5 | 641.6 / 504.8 / 379.2 |

Neither setting changes the picture. Over the three settings, L1/L11 is 1.44–1.45 for the trained model (samples A, B, C) and 1.53–1.73 for random seeds 0–2.

**Cross-sample CKA, three disjoint 2,000-cell samples** (mean of the three pairs; `analysis/intervals.json`, `cka3_jackknife`; 95% jackknife CI over 30 groups of 50 genes and over 10 cell blocks):

| Layer | Trained | [genes] | [cells] | Random seed 0 | [genes] | [cells] |
|---|---|---|---|---|---|---|
| L1 | 0.9997 | [0.9997, 0.9998] | [0.9997, 0.9998] | 0.9938 | [0.9936, 0.9941] | [0.9936, 0.9941] |
| L2 | 0.9997 | [0.9996, 0.9997] | [0.9996, 0.9997] | 0.9880 | [0.9871, 0.9889] | [0.9871, 0.9889] |
| L3 | 0.9993 | [0.9993, 0.9994] | [0.9992, 0.9994] | 0.9804 | [0.9785, 0.9823] | [0.9780, 0.9828] |
| L4 | 0.9991 | [0.9990, 0.9991] | [0.9989, 0.9992] | 0.9741 | [0.9714, 0.9769] | [0.9702, 0.9781] |
| L5 | 0.9988 | [0.9987, 0.9989] | [0.9987, 0.9990] | 0.9696 | [0.9662, 0.9731] | [0.9645, 0.9748] |
| L6 | 0.9981 | [0.9980, 0.9982] | [0.9979, 0.9982] | 0.9673 | [0.9631, 0.9715] | [0.9615, 0.9732] |
| L7 | 0.9972 | [0.9970, 0.9975] | [0.9969, 0.9976] | 0.9666 | [0.9619, 0.9714] | [0.9602, 0.9730] |
| L8 | 0.9974 | [0.9972, 0.9976] | [0.9972, 0.9977] | 0.9654 | [0.9603, 0.9706] | [0.9585, 0.9723] |
| L9 | 0.9978 | [0.9976, 0.9980] | [0.9976, 0.9980] | 0.9655 | [0.9601, 0.9709] | [0.9589, 0.9722] |
| L10 | 0.9976 | [0.9974, 0.9978] | [0.9973, 0.9978] | 0.9655 | [0.9600, 0.9710] | [0.9589, 0.9722] |
| L11 | **0.9962** | [0.9959, 0.9964] | [0.9958, 0.9966] | **0.9647** | [0.9598, 0.9697] | [0.9588, 0.9706] |

- From L1 to L11, CKA drops by 0.0036 in the trained model and by 0.029 in the random model.
- The intervals describe CKA of 2,000-cell samples, not a noise-free CKA. At L11 the cell-block jackknife bias is −0.0037 (trained) and −0.031 (random seed 0), 22 and 12 times the SE. A negative bias means that averages over more cells would give a CKA closer to 1.
- The three pairs agree: trained L11 0.99605, 0.99612, 0.99642; random 0.96263, 0.96530, 0.96631.
- The trained model's per-gene averages are clearly more stable across cell samples than the random model's, at every layer from L1.

**Split-half CKA inside one sample** (1,000 against 1,000 cells; `analysis/intervals.json`, `cka_half_jackknife`; 95% jackknife CI over genes and over cell blocks):

| Job | L1 | L6 | L11 [genes] [cells] |
|---|---|---|---|
| Trained, A | 0.9995 | 0.9963 | 0.9927 [0.9922, 0.9932] [0.9919, 0.9936] |
| Trained, B | 0.9995 | 0.9963 | 0.9930 [0.9925, 0.9935] [0.9917, 0.9943] |
| Trained, C | 0.9995 | 0.9961 | 0.9924 [0.9919, 0.9929] [0.9906, 0.9942] |
| Random seed 0, A | 0.9882 | 0.9462 | 0.9414 [0.9345, 0.9483] [0.9315, 0.9513] |
| Random seed 1, A | 0.9880 | 0.9475 | 0.9435 [0.9366, 0.9503] [0.9324, 0.9545] |
| Random seed 2, A | 0.9881 | 0.9386 | 0.9300 [0.9224, 0.9376] [0.9152, 0.9449] |
| Random seed 0, B | 0.9881 | 0.9467 | 0.9385 [0.9300, 0.9470] [0.9315, 0.9455] |
| Random seed 0, C | 0.9879 | 0.9425 | 0.9341 [0.9258, 0.9425] [0.9174, 0.9509] |

- All three random seeds are far less stable than the trained model at every layer. Seed 0 is typical of the three.

**What the CKA gap from 1 measures.** If the gap came only from sampling noise in the per-gene averages, halving the number of cells behind each average would double 1 − CKA.

| Model | 1 − CKA at L11, 2,000-cell samples (mean of the three pairs; median 482–494 cells per gene) | 1 − CKA at L11, 1,000-cell halves of sample A (median 246–247 cells per gene) | Ratio |
|---|---|---|---|
| Trained | 0.0038 | 0.0073 | 1.9 |
| Random seed 0 | 0.035 | 0.059 | 1.7 |

The halves of samples B and C give 0.0070 and 0.0076 (trained) and 0.061 and 0.066 (random seed 0). So for the trained model almost all of the gap from 1 is noise from which cells were averaged. For random seed 0 the ratio is lower (1.7), so noise is a smaller share of its gap. These CKA values measure how much a gene's average vector moves when it is computed from a different random set of cells of the same atlas. They are not a comparison between models or between tissues.

### Verification

No independent analysis of this run is recorded. The main numbers were computed a second way (`outputs/v3_spectral/verify/verify.json`; all pass):

| Number | First way | Second way | Agreement |
|---|---|---|---|
| ER, all 12 layers (trained A, random seed 0 A, and the stored deployed vectors) | singular values (`_svd_and_metrics`, the copied metric code) | eigenvalues of the float64 covariance | equal to 3 decimals |
| Three-sample CKA, L1, L6, L11 (trained and random seed 0) | `linear_cka` (feature space) | HSIC on centred gram matrices, float64 | differences ≤ 4 × 10⁻⁶ |
| Tokens of 120 random cells | `inputs_v3` | own tokeniser, float64 | 115 identical; 5 differ only inside exact ties |
| Stored sums, all 8 jobs | sum of per-cell sums | sum of per-gene block sums | largest relative difference 3.9 × 10⁻⁹ |
| Hidden states, two cells per model | stored per-cell sums | recomputed from scratch on MPS and on CPU; `output_hidden_states` against hook captures (`hooks_v2.ResidualEditor`) at all 12 sites | captures identical (difference 0.0); stored sums reproduced within 7 × 10⁻⁷ relative; CPU against MPS within 2.6 × 10⁻⁶ relative |
| Width of the CKA interval, trained L11 | gene-jackknife SE 0.00012; cell-jackknife SE 0.00017 | gene-bootstrap SD 0.00013; SD of the three pairwise CKAs 0.0002 | same size |

Applied to the stored per-gene vectors of the deployed analysis, whose inputs were ranked from log values, the same metric code returns that analysis's published values exactly.

### Limits

- **The deployed input order was not run through the model here.** The gap between the deployed last-layer ER and the value here is put down to the input order because cells, genes and metric code are the same. It was not shown by a direct run. The deployed analysis did not record its torch and transformers versions.
- **One random-initialisation scheme.** Only the transformers default (normal(0, 0.02)) was used. Other schemes, or a model with shuffled trained weights, were not tried. The architecture conclusion holds for this scheme and these inputs.
- **Random-weight CKA across samples uses one seed** (seed 0 on A, B and C). Seeds 1 and 2 were run on sample A only; their split-half CKA agrees with seed 0.
- **One atlas, one gene panel.** Tabula Sapiens immune cells only. The panel is weighted towards annotated genes. A second panel picked on raw counts gives the same ER, but other tissues and other gene sets were not tested.
- **Per-gene averages.** ER and CKA describe genes' average vectors over cells, not single-cell states. CKA here mostly measures averaging noise. No other model's cross-sample CKA was computed, so these values are not compared with any other model or with CKA values from studies that compare fine-tuned models.
- **TwoNN has no interval** beyond the spread across samples and seeds (trained 5.3 in all three samples; random 107–120). It is a noisy estimator in high dimension. Read it as large against small, not as exact values. Why the trained and random models differ in TwoNN and participation ratio was not tested.
- **Intervals are not bias-corrected.** The jackknife bias estimate for ER at L11 is +0.78 dimensions for the trained model and +11.9 to +13.1 for the random models. A positive value means that averages over more cells would give a somewhat lower ER. Correcting for it would lower the random models' L11 ER and make their fall from L1 larger, so the comparison would not change.
- **Smart-seq2 cells** (73, 80 and 80) were kept. The model predicts them much worse than UMI cells (loss 7.5–7.7 against 3.28 nats; a uniform guess gives 9.92). Dropping them changes the trained ER at L11 by 0.3 in sample A (0.4 in B, 0.5 in C).
- **Counts** are `raw/X`, before ambient-RNA removal. The `decontXcounts` layer was not tried.
- **Other analyses built on these vectors were not run with these inputs:** gene-set enrichment of singular vectors (STRING, TF–target), cell-type-specific compression, the germinal-centre–plasma-cell angle, BATF/BCL6, and principal angles between samples. The per-gene vectors needed for them are saved for samples A, B and C.
- **One context per sequence.** Each sequence holds one cell (at most 2,046 genes). MaxToki's multi-cell trajectory input was not used. MaxToki-1B was not tested.

### Files

- Scripts (`projects/maxtoki/runs/spectral-geometry-217M/scripts/`): `v3_spectral.py` (stages `prepare`, `extract` (resumable), `analyze --part metrics|cka|cka_half|summary`, `verify --part code|samples|tokens|sums|l0|metrics2|forward`), `v3_spectral_intervals.py` (parts `er`, `cka`, `half` (jackknife), `tables`, `index`). Metric code copied from `phase1_svd.py` and `phase8_stability.py` in the same folder. Shared code: `projects/maxtoki/setup/inputs_v3.py`, `projects/maxtoki/setup/hooks_v2.py`, `projects/maxtoki/setup/maxtoki_adapter.py`.
- Outputs (`projects/maxtoki/runs/spectral-geometry-217M/outputs/v3_spectral/`):
  - `run_config.json`: devices, versions, seeds, sample hashes, wall time per job, code sha256.
  - `prepare/`: `cells_A.npz`, `cells_B.npz`, `cells_C.npz` (rows, tokens, blocks, assay), `cells.csv`, `panel.json` (both gene panels), `run_config.json` (cell rows, gene IDs, counts and token sha256, encoding-check result).
  - `extract/<job>/` for jobs `trained_A`, `trained_B`, `trained_C`, `rand0_A`, `rand1_A`, `rand2_A`, `rand0_B`, `rand0_C`: `acc_sums.npy` (block sums), `acc_counts.npy`, `percell_sums.npy`, `cells.csv` (per-cell loss), `layer_gene_embeddings_panelD.npy` (12 × 1,500 × 1,232 per-gene vectors), `gene_counts_panelD.npy`, `state.json`, `run_config.json` (model, initialisation seed and hash, environment, encoding check, chunk times).
  - `analysis/`: `per_layer_metrics.csv` and `per_layer_metrics_optional.csv` (ER, participation ratio, top-direction share, TwoNN and shuffle-null ER per job and layer), `intervals.json` (jackknife intervals: `er_jackknife_cells`, `cka3_jackknife`, `cka_half_jackknife`, `contrast`), `cka.json` (CKA point estimates per pair and the bootstraps not used for intervals), `cka_half.json`, `summary.json`, `report_tables.md`, `er_jackknife_reps_<job>.npy`, `run_config.json`.
  - `verify/verify.json`: the second-way checks.
  - `code_versions/`: exact copies of `v3_spectral.py` for each script hash recorded by an extraction job.
- Run time: 10,134 s of MPS (2.8 h) for the 16,000 forward passes, in 30 chunks; about 1 h of CPU analysis.
- To reproduce, from `projects/maxtoki`: `OMP_NUM_THREADS=4 .venv/bin/python runs/spectral-geometry-217M/scripts/v3_spectral.py prepare`, then `extract --max-minutes 7` until it prints `ALL_DONE True` (add `--jobs rand0_B,rand0_C` for the seed-0 runs on B and C), then the `analyze`, `v3_spectral_intervals.py` and `verify` parts.
