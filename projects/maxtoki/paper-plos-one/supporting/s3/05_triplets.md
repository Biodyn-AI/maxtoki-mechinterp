## Feature triplets: do three SAE features interact? (MaxToki-217M, K562)

This analysis removes three SAE features (one each at layers 0, 5 and 9) alone, in pairs and all together. It asks whether removing all three does more or less than the sum of the parts. The non-additive part is the three-way interaction term. It is measured on the layer-11 SAE codes and on the logits.

### Data

- **Model.** MaxToki-217M, Hugging Face safetensors, float32, Apple MPS. torch 2.11.0, transformers 5.5.4. CPU threads capped at 4.
- **SAEs.** The SAEs of section 02 at L0, L5 and L9 (edits) and L11 (read-out).
- **Cells.** 20 K562 non-targeting control cells of `replogle_concat.h5ad`: rows 289300, 292833, 293501, 294821, 296499, 296999, 297671, 298043, 298575, 304014, 304434, 306407, 308960, 310091, 310869, 314183, 315807, 316745, 317964, 322713. They are the first 20 cells that tokenize from a pool of 100 drawn with numpy `default_rng(42)` and sorted (`inputs_v3.load_k562_control_cells(20, pool=100, seed=42)`). Barcodes are in `cells.json`.
- **Input.** Counts rebuilt and encoded as in section 00 (largest deviation from a whole-number multiple 1.3 × 10⁻⁴). All 20 cells are cut at 2,046 genes, so every cell has 2,048 tokens.

### Method

1. **Feature choice (made before any ablation).** At each of L0, L5 and L9:
   - candidates = SAE features active (z > 0) in ≥ 1% of all tokens of the 20 clean cells (all positions): L0 1,750, L5 460, L9 618;
   - the log10 activation frequency of the candidates is cut into 8 equal-count bins (octiles);
   - one feature is drawn at random from each bin with numpy `default_rng(20261001 + layer)`.
   The choice was written to `selection.json` at 13:40:16, before the first ablation (13:41:51). Chosen frequencies are 1.0% to 2.1%. Every chosen feature is active in all 20 cells, and none fires at `<bos>`. The "annotated" label below comes from the enrichment annotation of section 04. It was added after the choice and is a description only.

   | Layer | Feature | Octile | Frequency in 20 cells | Frequency in SAE training cells | Active positions per cell | Single effect eff_X (× 10⁻⁴) | Annotated (post hoc) | Top term |
   |---|---|---|---|---|---|---|---|---|
   | L0 | 763 | 0 | 0.0113 | 0.0109 | 23.1 | 1.78 | yes | GO negative regulation of programmed cell death |
   | L0 | 3035 | 1 | 0.0119 | 0.0121 | 24.4 | 3.00 | yes | Reactome transcriptional regulation by TP53 |
   | L0 | 2518 | 2 | 0.0137 | 0.0132 | 28.1 | 1.54 | yes | GO histone H3-K4 methylation |
   | L0 | 3778 | 3 | 0.0147 | 0.0137 | 30.1 | 2.64 | no | – |
   | L0 | 859 | 4 | 0.0158 | 0.0159 | 32.4 | 2.60 | yes | TRRUST E2F6 |
   | L0 | 293 | 5 | 0.0172 | 0.0173 | 35.2 | 2.76 | yes | Reactome synthesis of active ubiquitin |
   | L0 | 1910 | 6 | 0.0195 | 0.0191 | 40.0 | 2.92 | yes | GO tRNA modification |
   | L0 | 3275 | 7 | 0.0208 | 0.0207 | 42.6 | 6.36 | yes | KEGG Cushing syndrome |
   | L5 | 3812 | 0 | 0.0104 | 0.0098 | 21.2 | 0.85 | yes | Reactome metabolism of lipids |
   | L5 | 4369 | 1 | 0.0108 | 0.0094 | 22.1 | 0.87 | no | – |
   | L5 | 1812 | 2 | 0.0113 | 0.0126 | 23.1 | 1.50 | yes | GO regulation of endopeptidase activity |
   | L5 | 1550 | 3 | 0.0122 | 0.0127 | 24.9 | 1.10 | yes | GO intracellular protein transport |
   | L5 | 3142 | 4 | 0.0134 | 0.0149 | 27.4 | 2.07 | no | – |
   | L5 | 4903 | 5 | 0.0145 | 0.0163 | 29.8 | 2.40 | no | – |
   | L5 | 4821 | 6 | 0.0168 | 0.0199 | 34.4 | 1.49 | yes | Reactome transport of small molecules |
   | L5 | 1053 | 7 | 0.0213 | 0.0214 | 43.7 | 1.36 | no | – |
   | L9 | 1154 | 0 | 0.0102 | 0.0103 | 20.8 | 0.57 | no | – |
   | L9 | 4302 | 1 | 0.0106 | 0.0119 | 21.6 | 0.63 | yes | GO negative regulation of programmed cell death |
   | L9 | 1535 | 2 | 0.0114 | 0.0135 | 23.4 | 0.59 | no | – |
   | L9 | 235 | 3 | 0.0117 | 0.0114 | 24.1 | 0.67 | yes | GO hemopoiesis |
   | L9 | 494 | 4 | 0.0125 | 0.0131 | 25.6 | 0.72 | no | – |
   | L9 | 110 | 5 | 0.0138 | 0.0142 | 28.2 | 0.78 | yes | KEGG ribosome biogenesis in eukaryotes |
   | L9 | 2772 | 6 | 0.0154 | 0.0156 | 31.5 | 0.97 | no | – |
   | L9 | 1628 | 7 | 0.0166 | 0.0141 | 34.0 | 0.73 | yes | GO NADH dehydrogenase complex assembly |

   eff_X = mean over the 4,928 L11 targets of |cell-mean Δz| for that feature removed alone.

2. **Noise-floor features.** Two features per layer that should do nothing. L0: 1581 and 3049; L9: 1995 and 3537; all four are exactly silent in the 20 cells. No L5 feature is silent in these cells, so the 2 rarest L5 features were used: 3437 (active at 4 positions in 3 cells) and 4481 (40 positions, 2 per cell in all 20 cells). This rule was set before any ablation. Noise-floor triplets: 8 with all three null (2 × 2 × 2) and 24 with one null feature (8 each with null A, B or C; the null-B set uses 3437).
3. **Conditions.** 512 real triplets (8 × 8 × 8) and 32 noise-floor triplets. Each triplet needs the singles A, B, C, the pairs AB, AC, BC and the triple ABC. In total 826 distinct conditions × 20 cells = 16,520 ablation runs, plus 20 clean runs.
4. **Edits.** `hooks_v2.Ablate(l, [f])` removes z_f · W_dec[:, f] from the live input of block l at every position (section 00). Edits combine: the L5 code is computed after the L0 edit, and so on. Conditions without an L0 edit start at block 5 or 9 from the stored clean input.
5. **Read-outs per condition and cell.**
   - L11 SAE code change: Δz = mean over the 2,048 positions of (z with edits − z clean), 4,928 values.
   - Logit change: mean over positions, centred over the vocabulary, 20,275 values.
   - KL(clean ‖ edited) of the next-token distribution, mean over positions.
   Position means are computed as sums over the position axis.
6. **Effects.** e_X[t] = mean over cells of Δz[t] under condition X. Three-way term I_ABC = e_ABC − e_AB − e_AC − e_BC + e_A + e_B + e_C, computed per cell first. Pairwise term I_XY = d_XY − d_X − d_Y.
7. **Tests.**
   - Per target: one-sample t-test of the per-cell I_ABC over the 20 cells (df 19), and a bootstrap p as a second p-value. BH at q < 0.05 within each triplet (4,928 tests) and across all 512 × 4,928 = 2,523,136 tests.
   - Per triplet (aggregate): T = Σ_t |mean over cells of I_ABC[t]|. Null = 999 random per-cell sign flips (seed 20261002). BH across the 512 triplets.
   - Count null: the same 999 sign flips applied to all triplets at once (each flip ranked inside each triplet's own values). This gives a null for the number of triplets at p < 0.05 that keeps the dependence between triplets (they share cells and conditions).
   - Per feature: a pooled sign-flip test over each feature's 64 triplets, with BH over the 24 features.
8. **Size of the interaction.** Noise-adjusted energy: ‖true mean‖² is estimated as ‖mean‖² − Σ_t var_t / n. Energy share = noise-adjusted energy of I_ABC / noise-adjusted energy of the joint effect e_ABC. ρ3 = ‖e_ABC‖² / ‖e_A + e_B + e_C‖², noise-adjusted. ρ2 is the same for pairs. Intervals: 95% basic bootstrap, resampling cells (n = 20), 1,000 multinomial cell-weight vectors (seed 20261001) shared by all triplets and targets. The basic interval pivots on the unadjusted value, because the bootstrap of the adjusted energy centres on the unadjusted one.
9. **Power.** A planted interaction is added to d_ABC of 64 seeded triplets (seed 20261003), at 5%, 10%, 20%, 50% and 100% of each cell's own joint effect, and as a constant at 2%, 5% and 10% of the cell-mean joint effect.
10. **Redundancy ratios** (formulas of the deployed analysis). With eff_X = mean over targets of |e_X[t]|: pairwise AB = eff_AB / (eff_A + eff_B) (likewise AC, BC); three-way = eff_ABC / (eff_A + eff_B + eff_C); marginal C given AB = (eff_ABC − eff_AB) / eff_C. Each is also computed for an exactly additive joint effect (e_ABC set to e_A + e_B + e_C).
11. **Superadditivity rules.** Rule of the deployed analysis: |e_ABC| > 1.1 × (|e_A| + |e_B| + |e_C|), over targets with |e_ABC| > 0.01. Rule of the pipeline specification: |d_ABC| > |d_A| + |d_B| + |d_C|, over targets with |d| > 0.5 in any condition, where d is Cohen's d_z over the 20 cells; also in bands of the ratio of |d|.
12. **Per-position read-out check.** 4 cells × 4 seeded triplets (seed 20261021) × 8 full forward passes. Per-position energy share of I_ABC in the joint change at four read-outs.

### Checks

| Check | What it tests | Result | Pass |
|---|---|---|---|
| Encoding | `inputs_v3.check_encoding` on every cell right before its clean forward pass, in every model stage | 20 / 20 cells pass in every stage; 0 order violations; 2 positions with exact ties ordered differently (allowed). Tokens ranked from the stored log1p values fail 20 / 20 | yes |
| Zero edit | `ZeroDelta` at L0, L5, L9 and all three vs clean, 2 cells (MPS); cell 0 again with CPU float64 sums | largest logit difference 0.0; largest Δz 0.0; KL 0.0 | yes |
| Clean repeat | two clean passes | 0.0 | yes |
| Shortened path = full path | 2 conditions per condition type, 2 cells | largest difference 0.0 (Δz and logits) | yes |
| float32 vs float64 position mean | 1 triple condition, 2 cells | largest difference 8.0 × 10⁻⁹ | yes |
| AB vs B | cell-mean Δz of AB vs B, all 64 (A, B) pairs | 0 of 64 equal; smallest largest-difference 0.020 (median 0.049) | yes |
| ABC vs C | same, all 512 triplets | 0 of 512 equal; smallest 0.019 (median 0.049) | yes |
| Silent features | triplets with an exactly silent L0 or L9 feature (8 all-null, 8 null-A, 8 null-C) | I_ABC = 0.0 exactly in every cell and target | yes |
| Near-silent L5 feature | 8 null-B triplets (L5 3437, 4 active positions in 3 cells) | I ≠ 0 only in those cells: largest cell-mean \|I\| 2.4 × 10⁻⁴; 0 of 8 with any interval excluding 0; 1 of 8 at aggregate p < 0.05 | yes |
| Paired conditions | a condition with a silent null feature must equal the same condition without it (separate passes, often separate chunks) | 2,144 of 2,144 bit-identical | yes |
| Results vary across triplets | SD of the three-way ratio across triplets | 0.042 | yes |
| Stored values | 2 random conditions in each of the 220 groups (440) recomputed from scratch, full model path, CPU float64 sums | largest difference Δz 1.1 × 10⁻⁷, logits 2.5 × 10⁻⁸; KL within 0.15%; 220 of 220 stored clean fingerprints correct | yes |
| Read-out vs stored | position-mean Δz of ABC, 16 cell-triplet pairs | largest difference 1.2 × 10⁻⁸ | yes |

### Results

**Main findings.**
- There is no three-way interaction of a size that matters. The noise-adjusted share of the joint effect's energy that is three-way interaction is 0.0000028 (median over 512 triplets; 95% interval of the median [−0.00092, +0.00059]). So it is at most about 0.06%.
- The tests had power: a planted interaction of 5% of each cell's joint effect was found in 64 of 64 triplets.
- Two small effects are visible. One L5 feature shows a borderline excess of nominal three-way hits (found after looking; at most 0.03% of the energy). Two L5 → L9 feature pairs are 3–4% sub-additive.
- The edits are small (2.2% of the L11 residual norm). So the result says the model is smooth for small edits, not that its features never combine.

**Is there a three-way interaction?**

| Test | Result |
|---|---|
| Triplets with ≥ 1 target whose 95% interval of I_ABC excludes 0 (no adjustment) | 512 of 512 (488 targets per triplet on average) |
| Same count when each cell's I gets a random sign (no true interaction) | 488 per triplet (ratio 1.004) |
| Triplets with ≥ 1 target at BH q < 0.05 within the triplet (t-test) | **4 of 512** (up to 25.6 expected by chance) |
| Triplets with ≥ 1 target at BH q < 0.05 across all 2,523,136 tests | **0 of 512** |
| Same two counts with bootstrap p-values | 0 and 0 |
| Smallest t-test p over all tests | 6.7 × 10⁻⁶ |
| Targets with p < 0.05 (no adjustment) per triplet, mean | 110 (5% of the moving targets would be 246) |
| Logits: triplets with any vocabulary logit at BH q < 0.05 (within) | 7 of 512 |
| Aggregate sign-flip test, p < 0.05 | **44 of 512** |
| Count null for that number | mean 25.1, SD 8.4, 95th percentile 40; **p = 0.029** |
| Aggregate sign-flip test, BH across 512 | 0 of 512 |
| T_obs / T_null median (median over triplets) | 1.001 (5–95%: 0.984–1.022) |
| All interaction orders (e_ABC − e_A − e_B − e_C): p < 0.05 / count null / BH | 28 / mean 25.1, SD 13.3, p = 0.38 / 0 |

The 4 per-target hits (one target in one triplet each):

| Triplet (L0, L5, L9) | Target | p | q within | q across all | Mean I | \|I\| / \|e_ABC\| at that target |
|---|---|---|---|---|---|---|
| 2518, 4369, 110 | 4731 | 9.9 × 10⁻⁶ | 0.049 | 0.64 | 1.1 × 10⁻⁸ | 0.00012 |
| 293, 4903, 1154 | 1404 | 7.6 × 10⁻⁶ | 0.037 | 0.64 | −3.4 × 10⁻⁸ | 0.00023 |
| 1910, 1812, 1154 | 1535 | 9.5 × 10⁻⁶ | 0.047 | 0.64 | −9.7 × 10⁻⁹ | 0.00007 |
| 1910, 4903, 235 | 4858 | 6.7 × 10⁻⁶ | 0.033 | 0.64 | −1.5 × 10⁻⁸ | 0.00094 |

- These are not evidence of a three-way interaction. BH within a triplet allows a 5% chance of one false hit per triplet, so up to 25.6 of 512 triplets would show one by chance; 4 do.
- Their size (about 10⁻⁸) is below the float32 rounding of the stored values (the recomputation differs by up to 1.1 × 10⁻⁷).
- The t-test finds them because the L11 read-out has little noise: tiny, consistent shifts in all 20 cells give a small p even at 10⁻⁸.

The excess of nominal aggregate hits (44 of 512) against the count null (p = 0.029):

| Pooled over a feature's 64 triplets | L5 4369 | L9 1535 | All 512 triplets |
|---|---|---|---|
| Triplets at p < 0.05 | 15 of 64 | 10 of 64 | 44 of 512 |
| Pooled sign-flip p (same flips for all its triplets) | **0.002** | 0.008 | 0.181 |
| BH q over the 24 features | **0.048** | 0.096 | – |
| Pooled noise-adjusted energy share of I_ABC [95% basic CI] | 0.000010 [−0.00054, +0.00028] | 0.000014 [−0.00074, +0.00040] | 0.0000021 [−0.00061, +0.00034] |
| Pooled ρ3 [95% basic CI] | 0.9998 [0.9960, 1.0035] | 1.0058 [0.9985, 1.0131] | – |

- The largest per-feature count (15, L5 4369) is above what the null gives for the most-hit of 24 features (null mean 7.9, 95th percentile 12; p = 0.018).
- One feature passes BH over 24, just (q = 0.048). The others have q ≥ 0.096.
- Even for 4369, the three-way term is at most 0.03% of the joint effect's energy (upper interval bound).
- This was found after looking at the results. It is borderline and tiny.

A limit of the aggregate test: with 999 sign flips the smallest possible p is 0.001. BH over 512 triplets can then reject only if at least 11 triplets sit at that floor (over 192 pairs: at least 4). So "0 of 512 after BH" in the aggregate test is partly built into the design. The count null and the pooled tests do not have this limit.

**How big is the interaction?** (medians over the 512 triplets; 95% basic bootstrap interval of the median, cells resampled, n = 20, 1,000 resamples):

| Quantity | L11 SAE codes | Logits |
|---|---|---|
| Noise-adjusted energy share of I_ABC in the joint effect | **0.0000028** [−0.00092, +0.00059] | about 10⁻⁸ or less (see note) |
| Same, all interaction orders | 0.0000078 [−0.0041, +0.0027] | – |
| Triplets whose own interval of the I_ABC share is above 0 | 0 of 512 | – |
| Noise-adjusted ρ3 = ‖e_ABC‖² / ‖e_A + e_B + e_C‖² | **1.0037** [0.9977, 1.0112] | 1.00055 [1.00047, 1.00067] |
| ρ3 without the noise adjustment | 0.998 | 1.00057 |
| Triplets with a ρ3 interval above 1 / below 1 | 28 / 11 | – |
| Noise-adjusted pairwise ρ2 (mean of AB, AC, BC) | 1.0020 [0.9993, 1.0063] | – |
| Per-cell I_ABC energy share (median over cells, then triplets) | 0.0091 | 2.1 × 10⁻¹⁰ |

- On the logits, the unadjusted share is 1.6 × 10⁻⁸ and the noise-adjusted median is −9 × 10⁻¹³. At this size the values are float32 rounding, and the interval is not meaningful.
- On the logits, ρ3 = 1.00055 excludes 1 by 0.055%. That is the pairwise part (below); it is negligible.
- 28 triplets have a ρ3 interval above 1 and 11 below. If the intervals were exact, about 13 per side would be expected. With 20 cells the basic intervals are likely somewhat too narrow.
- KL(clean ‖ edited) per position, median over conditions: singles 2.0 × 10⁻⁴, pairs 5.2 × 10⁻⁴, triples 7.5 × 10⁻⁴ nats. It grows roughly with the number of edits (1 : 2.6 : 3.8).
- For the joint edit ABC, a median of 274 of 4,928 targets per triplet pass BH.

**Power** (planted interaction added to d_ABC of 64 seeded triplets):

| Planted interaction | Aggregate sign-flip test: p < 0.05 / BH within 64 | Per-target t-test: triplets with ≥ 1 BH target (within / across 64) |
|---|---|---|
| 5% of each cell's own joint effect | 64 / 64 | 64 / 64 |
| 10% | 64 / 64 | 64 / 64 |
| 20% | 64 / 64 | 64 / 64 |
| 50% | 64 / 64 | 64 / 64 |
| 100% | 64 / 64 | 64 / 64 |

- A planted interaction that is the same in every cell is found even more easily: 2% of the cell-mean joint effect (0.04% of its energy) gives 64 of 64 by the aggregate test. Its energy-share interval is above 0 in 5 of 64 at 2%, 57 of 64 at 5% and 64 of 64 at 10%.
- The L11 SAE read-out is dense: in a typical cell 4,768 of its 4,928 features are active somewhere. So a small edit moves most targets in every cell, and the change is consistent across cells. The median number of targets with a BH-significant single effect per triplet is 415 (A), 284 (B) and 1,502 (C).

**Pairwise interactions** (192 pairs: 64 each of AB, AC, BC):

| Quantity | Result |
|---|---|
| Pairs at aggregate sign-flip p < 0.05 | 17 of 192 (count null mean 9.4, SD 3.8; p = 0.048) |
| Pairs passing BH (aggregate) | 0 of 192 (BH needs ≥ 4 pairs at the p floor) |
| Pairs with ≥ 1 target at BH q < 0.05 (within pair) | **29 of 192** (up to 9.6 expected by chance) |
| Pairs whose interval of the I_XY energy share is above 0 | 0 of 192 |
| Pairs with a ρ2 interval below 1 / above 1 | 5 / 12 |
| Median noise-adjusted ρ2 | 1.0012 |
| \|I_XY\| share of \|e_XY\| (observed / sign-flip noise) | 0.0807 / 0.0811 |

Two pairs stand out. Both are an L5 feature followed by an L9 feature (ρ2 intervals: 95% basic bootstrap over cells, n = 20, 1,000 resamples):

| Pair (L5 → L9) | Targets at BH q < 0.05 | ρ2 [95% CI] | Aggregate p | cos(I_XY, e of the L9 feature) | L9 feature active positions per cell: alone → after the L5 edit |
|---|---|---|---|---|---|
| 3812 → 110 | 1,537 | 0.961 [0.939, 0.980] | 0.003 | −0.38 | 28.3 → 28.0 (changes in 7 of 20 cells) |
| 1812 → 1628 | 623 | 0.966 [0.944, 0.993] | 0.001 | −0.40 | 34.1 → 32.2 (changes in 15 of 20 cells) |

- Removing both features changes the L11 codes by 3.4–3.9% less energy than the two single removals added up.
- The interaction term itself is small (energy share 0.0013 and 0.0007; intervals include 0). The shortfall comes from its overlap with the additive part: it points partly against the L9 feature's own effect.
- Reading: the L5 feature helps drive the L9 feature. Once the L5 feature is removed, less of the L9 feature is left to remove. For 1812 → 1628 the L9 feature is active at fewer positions after the L5 edit. For 3812 → 110 the count barely changes, so the effect is probably on activation values, which were not recorded.
- The other 27 pairs with a per-target BH hit have 1–35 hits each. At those targets |I_XY| is 0.008–3.2% of the joint effect (median per pair). 24 of the 27 have a ρ2 interval that includes 1.
- 12 pairs have a ρ2 interval above 1 (the joint effect is 0.5–2.2% larger in energy than the sum); 6 of them involve L0 feature 763. 5 pairs have an interval below 1 (the two above, plus three with no per-target hit). If the intervals were exact, about 5 per side would be expected. These are weak signs, not findings.

**Where the per-cell non-additivity of the L11 codes comes from** (4 cells × 4 triplets; triplets (763, 4369, 110), (3035, 4903, 110), (1910, 3812, 110), (1910, 4369, 2772); per-position energy share of I_ABC in the joint change, median (range) over 16 cell-triplet pairs):

| Read-out | Per position | Position mean |
|---|---|---|
| L11 residual stream (input of `lm_head`, dense) | 5.0 × 10⁻⁹ (1.8 × 10⁻⁹ to 7.0 × 10⁻⁴) | 8.7 × 10⁻¹¹ |
| L11 SAE pre-activation (dense, linear in h) | 8.2 × 10⁻⁹ (2.8 × 10⁻⁹ to 6.3 × 10⁻⁴) | 1.4 × 10⁻¹⁰ |
| Logits (centred) | 7.1 × 10⁻⁹ (2.6 × 10⁻⁹ to 7.6 × 10⁻⁴) | 1.1 × 10⁻¹⁰ |
| **L11 SAE code (TopK 32)** | **0.046** (0.033 to 0.074) | 0.0068 |

- 99.99% of the code-level I_ABC energy sits on entries where the top-32 set differs between conditions. 90% of the joint code change itself sits on such entries.
- Under ABC, 24% of positions change their top-32 set; 0.31 features are swapped per position.
- The joint change of the L11 residual is 2.2% of its norm (median; 1.6–4.1%).
- So the model responds additively. The apparent non-additivity of the codes comes from features entering or leaving the top 32.

**Superadditivity:**

| Rule | Pool | Superadditive | Other classes |
|---|---|---|---|
| Deployed analysis: \|e_ABC\| > 1.1 × (\|e_A\| + \|e_B\| + \|e_C\|), targets with \|e_ABC\| > 0.01 | 6,738 targets pooled over 512 triplets | 3 (0.04%), in 3 of 512 triplets | – |
| Pipeline specification, strict: \|d_ABC\| > \|d_A\| + \|d_B\| + \|d_C\|, targets with \|d\| > 0.5 in any condition | 1,528,321 targets (60.6% of 2,523,136) | 5,407 (0.35%), in 512 of 512 triplets | – |
| Pipeline specification, banded (ratio of \|d\|) | same | > 1.1×: 3,344 (0.22%) | 0.9–1.1×: 7,168 (0.47%); < 0.9×: 1,517,809 (99.3%) |

- These rules have no valid null, so the counts are not evidence for or against synergy.
- The inclusion rule keeps 60.6% of targets because effects are consistent across cells (pure noise would keep 23.5%).
- "Sub-additive" is the usual outcome when |d_ABC| is compared with a sum of three |d|: with the raw-mean rule at 0.9×, an exactly additive joint effect is already called sub-additive for 44.7% of included targets (observed 56.7%).

**Redundancy ratios** (distribution across the 512 triplets; interval: 95% basic bootstrap of the median, cells resampled, n = 20, 1,000 resamples):

| Ratio | Median [95% CI of median] | 5%–95% across triplets | SD across triplets | If exactly additive (median) |
|---|---|---|---|---|
| Pairwise AB | 0.797 [0.790, 0.806] | 0.729–0.876 | 0.044 | – |
| Pairwise AC | 0.888 [0.873, 0.889] | 0.853–0.937 | 0.024 | – |
| Pairwise BC | 0.864 [0.838, 0.856] | 0.840–0.896 | 0.018 | – |
| Pairwise mean | **0.851** [0.836, 0.850] | 0.818–0.895 | 0.022 | 0.865 |
| Three-way | **0.743** [0.730, 0.746] | 0.686–0.838 | 0.042 | 0.767 |
| Marginal C given AB | 0.440 [0.415, 0.450] | 0.328–0.533 | 0.060 | – |
| Three-way, logits | 0.758 [0.751, 0.770] | 0.670–0.900 | 0.065 | – |

- For a few ratios (pairwise BC, pairwise mean) the noise bias is about as large as the spread, so the basic interval does not contain the plain estimate.
- All 512 triplets have a three-way interval below 1. That does not mean redundancy: the formula gives 0.767 for an exactly additive joint effect, and 0.758 on the logits, where the response is additive to about 10⁻⁸. The ratios fall below 1 because effects of opposite sign cancel across targets and noise inflates |mean|.
- Observed minus additive three-way ratio: −0.023 (interval of the median [−0.019, −0.010]). This gap is noise: e_ABC is one noisy condition, while e_A + e_B + e_C sums three, so its |mean| is inflated more. The noise-adjusted ρ3 removes it (1.0037).
- For context, the deployed analysis reported a three-way ratio of 0.1896 and a pairwise mean of 0.3235 from 4 triplets.
- Breakdowns: by frequency half (octiles 0–3 vs 4–7 at each layer), the median three-way ratio is 0.715–0.779 and the median I_ABC energy share is −0.00002 to +0.00001 in all 8 groups. By the post-hoc annotation label it is 0.734–0.761 and −0.00001 to +0.00002. No group has any target passing BH across all tests.

### Verification

No separate verification by a second agent is recorded for this analysis. The report itself describes a cross-check with separately written code (`v3_triplets.py crosscheck`, CPU float64): conditions read straight from the cell files, `scipy.stats.ttest_1samp`, BH as a step-up rule, the noise-adjusted energy from a pairwise-product (U-statistic) formula, and a new sign-flip seed.

| Number | Main analysis | Cross-check |
|---|---|---|
| Triplets with any BH target within / across | 4 / 0 of 512 | 4 / 0 of 512 |
| Smallest p | 6.708 × 10⁻⁶ | 6.708 × 10⁻⁶ |
| Median ρ3 | 1.0037108 | 1.0037108 (largest relative difference over triplets 9 × 10⁻¹⁶) |
| Median I_ABC energy share | 2.7829 × 10⁻⁶ | 2.7829 × 10⁻⁶ (largest difference 2 × 10⁻¹⁸) |
| Aggregate sign-flip p < 0.05 / BH (new seed 20261099) | 44 / 0 | 44 / 0 |
| Interval of the median energy share | basic [−0.00092, +0.00059] | new seed, percentile [0.00062, 0.00223]; as a basic interval around the unadjusted median 0.00123: [−0.00100, +0.00061] |

The percentile interval sits around the unadjusted value, as expected (Method, step 8); turned into a basic interval it matches. The count-null stage recomputed all 512 triplet and 192 pair sign-flip p-values: identical to the main analysis (largest difference 0.0). The 440 conditions recomputed from scratch (Checks) also match.

### Limits

- **20 cells**, not the 200 of the pipeline specification. Cells were traded for triplets.
- **Small edits only.** Each edit removes one feature active in 1.0–2.1% of tokens. The joint change of the L11 residual is 2.2% of its norm. A smooth network is close to linear here. Larger edits were not tested.
- **One read-out layer** (L11 codes and logits), one cell type (K562), one SAE seed.
- **No annotation was used to choose features.** The labels are post hoc and descriptive. Annotation rates in this project are near chance, so they carry little biological meaning.
- **The L5 noise floor is near-null, not null.** The exact-zero checks hold for the L0 and L9 null features only.
- **The L5 4369 excess is post hoc and borderline** (q = 0.048 over 24 features). It was not tested on new cells.
- **The pairwise mechanism is inferred, not measured.** It rests on active-position counts and on the cosine with the L9 feature's single effect. L9 activation values under the L5 edit were not recorded.
- **The 4 per-target hits were not recomputed in float64.** Their size (10⁻⁸) is below the stored float32 rounding.
- **No valid null for the superadditivity rules.**
- **The basic bootstrap intervals with 20 cells are likely somewhat too narrow** (28 + 11 triplets have a ρ3 interval that excludes 1, against about 26 if the intervals were exact).
- **The main script file was edited once during the `run` stage.** Chunks 1–5 (74 of 220 groups: cells 0–5 and groups 0–7 of cell 6) ran with `v3_triplets.py` sha256 73ada252…; chunks 6–17 and all later stages ran with 94e31c03…. The text of the first version was not kept. The checks that cover both parts all pass: the recomputation covers all 220 groups (440 conditions, largest difference 1.1 × 10⁻⁷), 2,144 of 2,144 paired conditions are bit-identical, 220 of 220 clean fingerprints are correct, and the read-out of cells 0–3 (all from the first part) matches the stored values to 1.2 × 10⁻⁸.
- Another job ran on the machine from about 13:45 to 15:55. It slowed the grid from about 14 s to 38 s per group of conditions. It did not change any result (every determinism check gave 0.0).

### Files

- Scripts (`projects/maxtoki/runs/exhaustive-mapping-217M/scripts/`): `v3_triplets.py` (stages `select`, `verify`, `run` (resumable), `recheck`, `readout`, `analyze`, `power`, `crosscheck`, `compare`), `v3_triplets_followup.py` (stages `countnull`, `pairs`, `perfeature`; no model). Shared code: `projects/maxtoki/setup/inputs_v3.py`, `projects/maxtoki/setup/hooks_v2.py`. SAEs: `projects/maxtoki/runs/sae-atlas-217M/outputs/v3_sae/layer_{00,05,09,11}/sae_final.pt`.
- Outputs (`projects/maxtoki/runs/exhaustive-mapping-217M/outputs/v3_triplets/`): `run_config.json` (device, versions, seeds, cell rows and barcodes, counts and token sha256, input-encoding record and check results, features and null features, SAE, hook and input sha256, wall time, free memory and script sha256 per chunk), `selection.json` (rule, bins, choice, post-hoc labels), `cells.json`, `cells_tokens.npz`, `cells_counts.npz`, `clean_frequencies.npz`, `verify.json`, `recheck/cell*.json`, `readout/cell*.json`, `readout_summary.json`, `cells/cell{ci}_g{gi}.npz` (raw per-cell results), `summary.json`, `per_triplet.json`, `per_pair.json`, `interaction_terms_real.npz`, `power.json`, `crosscheck.json`, `bootstrap_cell_weights.npy`, `signflip_signs.npy`, `analyze_cache/`.
- Follow-up outputs (`.../outputs/v3_triplets_followup/`): `countnull.json`, `pairs.json`, `perfeature.json`, `run_config.json`.
- Run time: model stages on MPS: select 0.3 min, verify 0.7 min, run 113.2 min in 17 chunks of ≤ 7.3 min (16,520 ablation runs), recheck 14.7 min, readout 2.6 min. CPU stages: analyze 8.0 min, power 0.3 min, crosscheck 0.2 min; countnull 0.3, pairs 0.6, perfeature 0.9 min. Lowest free memory seen 3.54 GB (guard at 3 GB).
