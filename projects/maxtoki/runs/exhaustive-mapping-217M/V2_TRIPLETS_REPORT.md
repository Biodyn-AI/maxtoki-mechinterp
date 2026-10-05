# V2 triplets report — combinatorial ablation redone (revision item D5)

Date: 2026-10-01. Model: MaxToki-217M (HF safetensors, float32, Apple MPS). torch 2.11.0, transformers 5.5.4.
Hooks: `setup/hooks_v2.py` (edits on the input of block l, SAE code from the live tensor, edits stack).
Every number below comes from the files listed in section 8.

## 0. Results first

1. **512 triplets were ablated, not 4.** The design is a full 8 × 8 × 8 grid of (L0, L5, L9) SAE features on
   20 K562 control cells. Interventions: 24 single, 192 pair and 512 triple conditions (728), plus 98 conditions
   for 32 noise-floor triplets. That is 826 conditions × 20 cells = 16,520 ablation runs. Measurements: each run
   is read out at 4,928 L11 SAE features and 20,275 logits. So 512 triplets give 512 × 4,928 = 2,523,136
   triplet-by-target interaction terms. The deployed "2,980" was a count of targets for one triplet.
2. **No three-way interaction (I_ABC) was found.**
   - Per target: **0 of 512 triplets** have any L11 target with BH q < 0.05 (t-test over cells), either within the
     triplet (4,928 tests) or across all 2,523,136 tests. Bootstrap p-values give the same 0 of 512.
     The smallest p over all 2.5 M tests is 8.4e-4.
   - Per triplet (all targets together, sign-flip test): 18 of 512 triplets have p < 0.05 (25.6 expected by
     chance) and 0 of 512 survive BH. The observed statistic is 0.999 × its null median (5–95% over triplets:
     0.981–1.017).
   - Size: the noise-corrected share of the joint effect's energy that is three-way interaction is
     **−0.0000** (median over triplets; 95% CI of the median [−0.0145, +0.0094]; cells resampled, 1,000 reps).
     No single triplet has a CI above 0.
   - The test had power. A planted interaction of 10% of each cell's own joint effect is detected in
     64 of 64 triplets (5%: 47 of 64).
3. **The model's response to these three edits is additive.** Inside each cell, the three-way term is about
   1e-9 of the energy of the joint change in the logits (median over 512 triplets), and about 2.5e-8 of the
   energy of the joint change of the L11 residual stream at each position (read-out check, 16 cell-triplet pairs).
   The L11 SAE codes look less additive inside a cell (28% per position) only because the L11 SAE is a TopK read-out:
   99.99% of that non-additivity sits on features that enter or leave the top 32 between conditions.
   It has no consistent sign across cells, so it averages out.
4. **Superadditivity.** Deployed 1.1× rule: 7 of the 2,049 targets with |e_ABC| > 0.01, pooled over all 512
   triplets (0.34%), in 7 of 512 triplets. Spec rule (Cohen's d): 10,347 of 583,779 included targets (1.8%) in
   all 512 triplets. Neither rule has a valid null. The spec's inclusion rule (|d| > 0.5 in any of 7 conditions)
   keeps 23.1% of targets, close to the ~23.5% that pure noise would give with 20 cells. The valid tests in point 2
   find nothing.
5. **The deployed "redundancy ratios" do not measure redundancy.** With the deployed formula the median
   three-way ratio is 0.639 (95% CI of the median [0.633, 0.646]) and the median pairwise mean is 0.766
   ([0.755, 0.769]). But the same formula gives 0.706 and 0.811 if the joint effect is set to exactly the sum of
   the single effects, and 0.678 on the logits, where the response is additive to 1e-9. The ratio is below 1
   because effects of opposite sign cancel across targets and because noise inflates |mean|. The noise-corrected
   energy ratio ‖e_ABC‖² / ‖e_A + e_B + e_C‖² is **1.006** [0.978, 1.051]: additive. Pairwise: 1.003 [0.985, 1.027].
6. **All sanity checks pass** (section 3): zero edit = clean (0.0); AB ≠ B in 64 of 64 pairs; ABC ≠ C in 512 of 512;
   never-active triplets give exactly 0; results now vary across triplets (three-way ratio SD 0.039 vs 0.0001
   deployed); 440 stored conditions recomputed from scratch match to 3.1e-8.

What this means for the paper: "2,980 tested triplets", "zero superadditive cases" and "three-way redundancy
0.190" are all replaced. The corrected result is: across 512 triplets, removing three single SAE features at
L0, L5 and L9 has an additive effect on the model. There is no synergy and no redundancy beyond noise.
These edits are small (the joint change of the L11 residual is 1.8% of its norm), so additivity is what a smooth
network should show. The result says little about how the model combines larger signals.

---

## 1. What was wrong in the deployed run

From `verification/triplets.md` and `V2_HOOKS_REPORT.md`:
- Only 4 triplets were ablated. "2,980" was the number of L11 targets with |mean change| > 0.01 for one triplet.
- The hook replaced the output of block l with `hidden_states[l] + delta`. That deleted block l.
- Each hook returned a fixed tensor from the clean run, so the deepest hook erased the others: AB = B, AC = C,
  BC = C, ABC = C. "Zero superadditive" was then certain by arithmetic, and the 0.190 ratio was
  eff_C / (eff_A + eff_B + eff_C).
- `hooks_v2.py` fixes both. This run uses it unchanged.

## 2. What I ran

### 2.1 Cells
- 20 K562 non-targeting control cells from the Replogle file
  `<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/crispri_validation/data/replogle_concat.h5ad`.
- Same pool as `experiment2_rerun.py`: 100 control cells drawn with numpy `default_rng(42)`, sorted rows, the first
  20 that tokenize (`H.load_k562_control_cells(20, pool=100, seed=42)`).
- Rows: 289300, 292833, 293501, 294821, 296499, 296999, 297671, 298043, 298575, 304014, 304434, 306407, 308960,
  310091, 310869, 314183, 315807, 316745, 317964, 322713. Barcodes are in `outputs/v2_triplets/cells.json`.
- Every cell is 2,048 tokens (BOS, 2,046 ranked genes, EOS), single-cell input as in the deployed run.

### 2.2 Features (chosen before any ablation; seed 20261001 + layer)
- Rule: candidates are features active (z > 0) in ≥ 1% of all tokens of the 20 cells (clean pass, all positions).
  "Annotated" means the feature has at least one row in the sae-atlas `phase2/layer_XX/significant_enrichments.csv`.
  The log10 frequency of the candidates is split into 4 quartile bins. One annotated and one unannotated feature is
  drawn at random from each bin.
- Candidates: L0 1,733 (1,536 annotated), L5 617 (284), L9 1,046 (622).
- None of the 24 chosen features fires at `<bos>` in any cell.

| Layer | Feature | Annotated | Freq. in 20 cells | Freq. in atlas | Top term (if annotated) |
|---|---|---|---|---|---|
| L0 | 810 | yes | 0.0116 | 0.0116 | Reactome Phase II conjugation |
| L0 | 3147 | no | 0.0127 | 0.0124 | – |
| L0 | 2137 | yes | 0.0139 | 0.0141 | TRRUST MEN1 |
| L0 | 3526 | no | 0.0147 | 0.0150 | – |
| L0 | 1107 | yes | 0.0182 | 0.0174 | Reactome G1/S transition |
| L0 | 194 | no | 0.0182 | 0.0176 | – |
| L0 | 1947 | yes | 0.0214 | 0.0215 | Reactome MAP2K and MAPK activation |
| L0 | 2810 | no | 0.0225 | 0.0234 | – |
| L5 | 3783 | yes | 0.0105 | 0.0099 | GO spliceosomal snRNP assembly |
| L5 | 3811 | no | 0.0110 | 0.0116 | – |
| L5 | 2451 | yes | 0.0139 | 0.0138 | Reactome Metabolism |
| L5 | 1679 | no | 0.0119 | 0.0341 | – |
| L5 | 2307 | yes | 0.0203 | 0.0171 | GO positive regulation of intracellular signal transduction |
| L5 | 4904 | no | 0.0161 | 0.0160 | – |
| L5 | 4341 | yes | 0.0407 | 0.0376 | Reactome COPII-mediated vesicle transport |
| L5 | 735 | no | 0.0286 | 0.0312 | – |
| L9 | 776 | yes | 0.0102 | 0.0096 | GO myeloid cell differentiation |
| L9 | 4233 | no | 0.0107 | 0.0123 | – |
| L9 | 1606 | yes | 0.0119 | 0.0111 | GO protein transport |
| L9 | 234 | no | 0.0135 | 0.0122 | – |
| L9 | 504 | yes | 0.0184 | 0.0192 | Reactome membrane trafficking |
| L9 | 121 | no | 0.0196 | 0.0160 | – |
| L9 | 2868 | yes | 0.0274 | 0.0258 | GO cellular response to oxygen-containing compound |
| L9 | 1474 | no | 0.0218 | 0.0127 | – |

- The ≥ 1% rule leaves a narrow range: chosen frequencies are 1.0% to 4.1%.
- Never-active features for the noise floor (zero activations in the 20 clean cells): L0 1574, 3057;
  L5 2618, 4501; L9 1151, 3329. Noise-floor triplets: 8 all-null (2 × 2 × 2) and 24 with one null feature
  (8 each with null A, null B, null C).

### 2.3 Conditions and read-outs
- Ablate(l, f) removes z_f · W_dec[:, f] from the live input of block l at every position (hooks_v2).
  Conditions with several edits stack: the L5 code is computed after the L0 edit, and so on.
- Speed-up: conditions without an L0 edit start the forward pass at block 5 or 9 from a cached live input.
  The `verify` stage checked that this gives the same result as the full pass (difference 0.0).
- Read-outs per condition and cell:
  - L11 SAE code change, mean over the 2,048 positions: Δz = mean_pos(z_ablated − z_clean), 4,928 values
    (same read-out as deployed). The L11 SAE reads the input of `lm_head` (after the final norm).
  - Logit change, mean over positions, centred over the vocabulary (20,275 values).
  - KL(clean ‖ ablated) of the next-token distribution, mean over positions.
- Effects: e_X[t] = mean over cells of Δz[t]. Möbius term: I_ABC = e_ABC − e_AB − e_AC − e_BC + e_A + e_B + e_C
  (spec, Experiment 2). It is computed per cell first, then averaged.

### 2.4 Statistics
- Resampling unit: the cell (n = 20). Bootstrap: 1,000 multinomial cell-weight vectors (seed 20261001), shared by
  all triplets and targets.
- Per-target I_ABC: percentile 95% CI of the mean; one-sample t-test (df 19); bootstrap p (2 × smaller tail, +1).
  BH at q < 0.05 within each triplet (4,928 tests) and across all 512 × 4,928 tests.
- Per-triplet aggregate test: T = Σ_t |mean over cells of I_ABC[t]|; null = 999 random per-cell sign flips of the
  whole target vector (seed 20261002). BH across the 512 triplets.
- Noise-corrected energy: ‖true mean‖² is estimated as ‖mean‖² − Σ_t var_t / n. This removes the part of ‖mean‖²
  that comes from cell-to-cell noise. Ratios of these energies get basic (bias-corrected) bootstrap CIs.
- Deployed-style ratios: eff_X = mean over targets of |e_X|; pairwise XY = eff_XY / (eff_X + eff_Y);
  three-way = eff_ABC / (eff_A + eff_B + eff_C); marginal C|AB = (eff_ABC − eff_AB) / eff_C. These ratios are
  biased by noise. I give basic bootstrap CIs. For a few of them the bias is about as large as the spread, so the
  CI does not contain the plain estimate.

## 3. Sanity checks (all pass)

| Check | What it tests | Result | Pass |
|---|---|---|---|
| Zero-delta | `ZeroDelta` at L0, L5, L9 and all three vs clean, 2 cells (MPS), repeated on cell 0 with CPU float64 | max abs logit diff **0.0**; max abs Δz **0.0**; KL 0.0 | yes |
| Clean repeat | two clean passes | 0.0 | yes |
| Fast path = full path | 2 conditions per condition type, 2 cells | max abs diff 0.0 (Δz and logits) | yes |
| AB vs B | cell-mean Δz of AB vs B, all 64 (A, B) pairs | 0 of 64 equal; smallest max-abs difference 0.0041 (median 0.0168). Deployed: identical | yes |
| ABC vs C | same, all 512 triplets | 0 of 512 equal; smallest 0.0040 (median 0.0172). Deployed: identical | yes |
| Never-active triplets | 8 all-null triplets | I_ABC = **0.0 exactly** in every cell and target; 0 of 8 with any CI excluding 0 | yes |
| One null feature | 24 triplets | null A, null B: 0.0 exactly. Null C: max |I| 0.0020 in one cell, because the L9 null feature became active at 1 position after upstream edits. 0 of 24 with any CI excluding 0 | yes |
| Twin conditions | a condition with an inactive null feature must equal the same condition without it (run in separate passes, often separate chunks) | 2,398 of 2,398 bit-identical | yes |
| Results vary across triplets | SD of the three-way ratio across triplets | **0.039** (deployed 0.0001). Pair mean SD 0.027. The deployed identity three-way = eff_C / Σeff no longer holds (correlation −0.28, max difference 0.64) | yes |
| Single effects depend on the feature | eff_X (mean |e| over targets) per feature | L0 1.8e-4 to 4.4e-4; L5 0.7e-4 to 4.0e-4; L9 0.8e-4 to 2.2e-4 | yes |
| Stored values | 2 random conditions per stored group (440 in all) recomputed from scratch, full model path, CPU float64 sums | max abs diff Δz 3.1e-8, logits 1.9e-8 | yes |
| Read-out check vs stored | position-mean Δz of ABC, 16 cell-triplet pairs | max abs diff 8.8e-9 | yes |

**An MPS problem found on the way.** A full-tensor `.sum()` or `.mean()` over 10–41 M elements, called inside the
forward-pass loop, sometimes returned a wrong value: 15 of 220 stored clean "fingerprints" and 9 of 440 rechecked
mean |Δlogit| values (error up to 69%). The tensors themselves were bit-identical on repeat. The same calls on a
stand-alone random tensor were right 900 of 900 times. All reported effects use sums over the 2,048 positions
(dim 0). Those matched a CPU float64 recomputation in 440 of 440 conditions. Mean |Δlogit| is not used anywhere.
Other agents who reduce large MPS tensors to one number should check against the CPU.

## 4. Results

### 4.1 Is there a three-way interaction? (per target and per triplet)

| Test | Result |
|---|---|
| Triplets with ≥ 1 target whose 95% CI of I_ABC excludes 0 (uncorrected) | 512 of 512 (563 targets per triplet on average, range 471–665) |
| Same count when each cell's interaction vector gets a random sign (no true interaction) | 563 per triplet (ratio observed / null 0.9995) |
| Triplets with ≥ 1 target at BH q < 0.05 within the triplet (t-test) | **0 of 512** (0%; 95% Clopper–Pearson upper bound 0.72%) |
| Triplets with ≥ 1 target at BH q < 0.05 across all 2,523,136 tests (t-test) | **0 of 512** |
| Same two counts with bootstrap p-values | 0 of 512 and 0 of 512 |
| Targets with uncorrected t-test p < 0.05 | 36.8 per triplet (5% of the ~4,640 targets that move would be 232) |
| Logits: triplets with any vocabulary logit at BH q < 0.05 (within triplet) | 0 of 512 |
| Aggregate sign-flip test, p < 0.05 uncorrected | 18 of 512 (25.6 expected) |
| Aggregate sign-flip test, BH across 512 triplets | **0 of 512** |
| Same test on all interaction orders (e_ABC − e_A − e_B − e_C) | 16 of 512 uncorrected, 0 after BH |

- The uncorrected "CI excludes 0" count is pure noise. The sign-flip null gives the same count. A 1,000-rep
  percentile bootstrap over only 20 cells gives intervals that are too narrow.
- The t-test is conservative here. Per-cell Δz values are spiky: for most targets only a few cells move. A target
  that moves in one cell only always has |t| = 1.

**Power.** What the tests can find, from planting an interaction into d_ABC of 64 random triplets:

| Planted interaction | Aggregate sign-flip test (p < 0.05 / BH) | Per-target t-test, triplets with ≥ 1 BH target (within / across 64) |
|---|---|---|
| 5% of each cell's own joint effect | 47 / 46 of 64 | 27 / 20 of 64 |
| 10% | 64 / 64 | 39 / 38 |
| 20% | 64 / 64 | 48 / 50 |
| 50% | 64 / 64 | 56 / 59 |
| 100% | 64 / 64 | 56 / 63 |

- So the aggregate test would find a consistent three-way interaction as small as 10% of the joint effect
  (under 1% of its energy) in every triplet. The per-target test is much weaker. Even the joint effect itself passes
  BH at only 17 of 4,928 targets per triplet (median; mean 36; single effects: A 7.5, B 1.5, C 2).
- A planted interaction that is the same in every cell (no extra spread) is found more easily: 5% of the cell-mean
  joint effect gives 64 of 64 by both tests. That is an optimistic upper bound.

### 4.2 How big is the interaction?

Medians over the 512 triplets, with 95% bootstrap CIs of the median (cells resampled):

| Quantity | L11 SAE codes | Logits |
|---|---|---|
| Noise-corrected energy share of I_ABC in the joint effect | **−0.0000** [−0.0145, +0.0094] | **0.0000** [0.0000, 0.0000] |
| Same, all interaction orders (e_ABC − e_A − e_B − e_C) | −0.0001 [−0.044, +0.027] | – |
| Triplets whose own CI of the I_ABC share is above 0 | 0 of 512 | – |
| Noise-corrected ρ3 = ‖e_ABC‖² / ‖e_A + e_B + e_C‖² | **1.006** [0.978, 1.051]; 7 triplets CI > 1, 1 CI < 1 | 1.0006 [1.0003, 1.0007] |
| Noise-corrected pairwise ρ2 (mean of AB, AC, BC) | 1.003 [0.985, 1.027] | – |
| Per-cell I_ABC energy share (median over cells, then triplets) | 0.106 | **9.4e-10** |

- Without the noise correction, ρ3 is 0.944. The correction moves it to 1.006. So the apparent shortfall of the
  joint effect is noise, not redundancy.
- On the logits ρ3 = 1.0006 excludes 1, but by 0.06%. That is the pairwise interaction part, and it is negligible.
- Effect sizes are small. KL(clean ‖ ablated) per position, median over conditions: singles 1.8e-4, pairs 3.9e-4,
  triples 6.2e-4 nats. They grow roughly in proportion to the number of edits (1 : 2.2 : 3.5).

### 4.3 Where does the per-cell non-additivity of the L11 codes come from? (read-out check)

Script stage `readout`: 4 cells (the first 4 of the 20) × 4 seeded triplets × 8 full forward passes.
Per-position energy share of I_ABC in the joint change, median (range) over 16 cell-triplet pairs:

| Read-out | Per position | Position mean |
|---|---|---|
| L11 residual stream (input of `lm_head`, dense) | 2.5e-8 (1.1e-8 to 1.9e-3) | 7.8e-10 |
| L11 SAE pre-activation W_enc(h − μ) + b (dense, linear in h) | 3.8e-8 (1.5e-8 to 2.0e-3) | 9.8e-10 |
| Logits (centred) | 2.7e-8 (8.6e-9 to 2.2e-3) | 9.6e-10 |
| **L11 SAE code (TopK 32)** | **0.28** (0.16 to 0.36) | 0.14 (0.08 to 0.17) |

- Share of the code-level I_ABC energy that sits on entries where the TopK active set differs between conditions:
  **99.99%**. Share of the joint code change itself on such entries: 98.9%.
- Under ABC, 32% of positions change their top-32 set (0.38 features swapped per position on average).
- The joint change of the L11 residual is 1.8% of its norm (median).
- **Plain reading.** The model itself responds additively: at the residual stream the three-way term is about
  1e-8 of the joint change, as expected for small edits. The L11 SAE turns small smooth changes into features
  jumping in or out of the top 32. That makes each cell's code change look non-additive, in a direction that differs
  from cell to cell. It is a property of the read-out, not of the model. It also means the L11 code read-out
  itself is noisy: almost all of any code change is top-32 membership switching.

### 4.4 Superadditivity

| Rule | Pool | Superadditive | Other classes |
|---|---|---|---|
| Deployed: |e_ABC| > 1.1 × (|e_A| + |e_B| + |e_C|), targets with |e_ABC| > 0.01 | 2,049 targets pooled over 512 triplets (about 4 per triplet) | **7 (0.34%)**, in 7 of 512 triplets | – |
| Spec, strict: |d_ABC| > |d_A| + |d_B| + |d_C|, targets with |d| > 0.5 in any condition | 583,779 targets (23.1% of 2,523,136) | **10,347 (1.8%)**, in 512 of 512 triplets | – |
| Spec, banded (ratio of |d|) | same | > 1.1×: 6,060 (1.0%) | 0.9–1.1×: 12,840 (2.2%); < 0.9×: 564,879 (96.8%) |

- d is Cohen's d_z over the 20 cells (mean / SD of the per-cell change). The spec asked for 200 cells; this run has 20.
- These counts are not evidence for or against synergy:
  - The inclusion rule keeps 23.1% of targets. Pure noise would give about 23.5% (|d| > 0.5 with n = 20 is |t| > 2.24,
    p = 0.0375 per condition, 7 conditions).
  - |d_ABC| is one noisy number and |d_A| + |d_B| + |d_C| is the sum of three, so "sub-additive" is the usual outcome
    even with no interaction. With the deployed raw-mean rule at 0.9×, an exactly additive joint effect would already
    be called sub-additive for 49.6% of included targets (observed 68.7%).
  - I tried a sign-flip null for these rules (`v2_triplets_followup.py classnull`). It is not valid: I_c and the
    additive part share the same condition terms with opposite signs, so the null ABC* is noisier than the real ABC.
    It gave more superadditive calls than observed (spec strict 12,618 vs 10,347; deployed 28.9 vs 7). I do not use it.
- The valid tests (section 4.1) find no synergy.
- The deployed run reported "0 of ~2,980" per triplet. Now only about 4 targets per triplet pass |e_ABC| > 0.01,
  because the fixed edits are far smaller than deleting a block.

### 4.5 Redundancy ratios (deployed formula)

Distribution across the 512 triplets; CI = basic bootstrap 95% CI of the median (cells resampled):

| Ratio | Median [95% CI of median] | 5%–95% across triplets | SD across triplets | If exactly additive (median) | Deployed (4 triplets) |
|---|---|---|---|---|---|
| Pairwise AB | 0.731 [0.730, 0.742] | 0.681–0.823 | 0.045 | – | 0.165 |
| Pairwise AC | 0.783 [0.765, 0.786] | 0.739–0.847 | 0.037 | – | 0.219 |
| Pairwise BC | 0.769 [0.739, 0.764] | 0.729–0.846 | 0.035 | – | 0.586 |
| Pairwise mean | **0.766** [0.755, 0.769] | 0.724–0.812 | 0.027 | 0.811 | 0.3235 |
| Three-way | **0.639** [0.633, 0.646] | 0.586–0.710 | 0.039 | 0.706 | 0.1896 |
| Marginal C given AB | 0.253 [0.225, 0.279] | 0.139–0.472 | 0.105 | – | 0.294 |
| Three-way, logits | 0.678 [0.659, 0.688] | 0.571–0.822 | 0.075 | – | – |

- Means across triplets: three-way 0.641 (CI [0.635, 0.649]); pairwise mean 0.767 ([0.754, 0.769]).
- All 512 triplets have a three-way CI below 1. But "below 1" does not mean redundant. The same formula gives 0.706
  for an exactly additive joint effect and 0.678 on the logits, where the response is additive to 1e-9. Two things
  push it below 1: effects of opposite sign cancel across targets, and noise inflates each |mean|.
- Observed minus additive three-way ratio: −0.067 (CI of the median [−0.055, −0.033]). This gap is noise: e_ABC is
  one noisy condition, e_A + e_B + e_C sums three, so the denominator is inflated more. The noise-corrected energy
  ratio, which removes this, is 1.006 [0.978, 1.051] (section 4.2).
- No difference between annotated and unannotated features. Median three-way ratio by pattern (L0, L5, L9;
  a = annotated, u = unannotated): aaa 0.640, aau 0.649, aua 0.636, auu 0.647, uaa 0.633, uau 0.639, uua 0.620,
  uuu 0.630 (64 triplets each). None has any BH target.

### 4.6 Pairwise interactions (context)

- 192 pairs (64 each of AB, AC, BC). Pairwise term I_XY = e_XY − e_X − e_Y.
- 0 of 192 pairs pass the aggregate sign-flip test after BH. 0 of 192 have any target at BH q < 0.05.
- Observed |I_XY| share of |e_XY|: median 0.285. Sign-flip noise gives 0.285.

## 5. What changed versus the deployed result

| Item | Deployed (`experiment2_v2/combinatorial_summary.json`) | Now |
|---|---|---|
| Triplets ablated | 4 (and 4 in v1, sharing L0 F0 and L5 F0) | **512** (+ 32 noise-floor) |
| Feature choice | greedy, top of a shared-pathway list | seeded, 4 annotated + 4 unannotated per layer, spread over frequency |
| Cells | 50 | 20 |
| Hook | deleted block l; deepest hook erased the others | edit on the input of block l, live code, edits stack |
| AB vs B, ABC vs C | identical | differ in 64/64 and 512/512 |
| "2,980" | targets of one triplet with |e_ABC| > 0.01 | about 4 targets per triplet pass that cut (2,049 over 512 triplets) |
| Interaction term I_ABC | not computed | computed per cell and target; 0 of 512 triplets with any BH target; energy share −0.0000 [−0.0145, 0.0094] |
| Superadditive (1.1× rule) | 0 of ~2,980 (forced by AB = B, ABC = C) | 7 of 2,049 pooled (0.34%); not above noise |
| Pairwise mean ratio | 0.3235 (SD 0.0) | 0.766 (SD 0.027 across triplets); not a redundancy measure |
| Three-way ratio | 0.1896 (SD 0.0001) | 0.639 (SD 0.039); not a redundancy measure |
| Noise-corrected joint vs additive energy | – | 1.006 [0.978, 1.051]: additive |

## 6. What I could NOT do, and limits

- **20 cells, not the spec's 200.** The grid design traded cells for triplets. Per-target tests are weak with 20 cells
  (section 4.1, power table). The per-triplet aggregate test and the energy shares are the strong evidence.
- **Small edits only.** Each edit removes one SAE feature active in 1–4% of tokens. The joint change of the L11
  residual is about 1.8% of its norm. In this regime a smooth network is close to linear, so additivity is expected.
  I did not test larger edits (several features per layer, steering, or whole-direction ablations).
- **One read-out layer.** I read out at L11 (codes and logits) only, as deployed. I did not read out at
  intermediate layers (for example L9 after an L0 + L5 edit).
- **No valid null for the two classification rules.** The sign-flip null I tried is biased (section 4.4). The cosine
  between the per-cell interaction and the additive part (`align.json`) is negative by construction for the same
  reason (pure independent noise would give about −0.93). I do not use either as evidence.
- **Read-out check is small.** 4 cells × 4 triplets. The pattern was the same in all 16 pairs.
- The chosen features cover a narrow frequency range (1.0–4.1%), because of the ≥ 1% rule. Rarer features were
  not tested; they would give smaller effects.
- Edits apply at every position, including `<bos>`/`<eos>`, as in the deployed code. None of the 24 features fires
  at `<bos>` in these cells.
- Script changes during the run: edits between run chunks touched only the analyze stage and bookkeeping
  (recorded in `run_config.json`, with the script hash per chunk). The reported numbers come from the final
  `analyze` call, made with the final script (sha256 9c915b8a…). The follow-up `classnull` stage ran before I added
  the `align` stage to the same file; nothing else changed.
- Other unrelated jobs ran on the machine (CPU). They slowed some chunks but did not change results (every
  determinism check gave 0.0).

## 7. Run cost

- Ablation grid: 16,520 condition runs in 69.3 min over 10 chunks (≤ 7.5 min each; 15.9 s median per group of
  conditions). Select 19 s, verify 46 s, recheck 10.4 min, final analyze 3.9 min. Follow-up: 1.1 + 2.7 + 2.1 min.
- Memory guard at 3 GB free (`hooks_v2.check_memory`) before every group. Peak memory footprint
  (`/usr/bin/time -l`): run 4.05 GB, verify 5.35 GB, analyze 4.47 GB.

## 8. Files

Scripts:
- `runs/exhaustive-mapping-217M/scripts/v2_triplets.py` — stages `select`, `verify`, `run` (resumable),
  `recheck`, `analyze`.
- `runs/exhaustive-mapping-217M/scripts/v2_triplets_followup.py` — stages `classnull`, `align` (no model) and
  `readout` (model).

Outputs, `runs/exhaustive-mapping-217M/outputs/v2_triplets/`:
- `run_config.json` (device, versions, seeds, cell rows and barcodes, features, wall time and script hash per chunk,
  notes), `selection.json` (feature choice and the rule), `cells.json`, `cells_tokens.npz`, `clean_frequencies.npz`.
- `verify.json`, `recheck/cell*.json`.
- `cells/cell{ci}_g{gi}.npz` — raw per-cell results: condition keys, Δz (4,928), Δlogit (20,275), KL, live n_active.
- `summary.json` (all summaries in this report), `per_triplet.json`, `per_pair.json`,
  `interaction_terms_real.npz` (per triplet × target: mean I_ABC, CI, t-test p, BH q across),
  `bootstrap_cell_weights.npy`, `signflip_signs.npy`.

Outputs, `runs/exhaustive-mapping-217M/outputs/v2_triplets_followup/`:
- `classnull.json`, `align.json`, `readout/cell*.json`, `readout_summary.json`, `run_config.json`.

## Plain-words summary

The old test removed features in groups of three, but its code was broken. It deleted whole layers, and the last
edit wiped out the others, so "no synergy" was guaranteed. It also tested only 4 groups; the famous "2,980" was a
count of measured features, not of groups. I redid it with the fixed code on 512 groups of three features
(8 from each of three layers, half with known biology, half without) in 20 cells. All the checks pass: an empty edit
changes nothing, combined edits really combine, silent features do nothing, and results now differ from group to
group. Removing three features together does what you get by adding up the three single removals. No group shows a
three-way effect that survives correction for multiple tests, and the test was strong enough to find one that is
10% of the joint effect. Inside the model the effects add up almost perfectly (the leftover is about one part in
100 million). The last-layer feature dictionary makes the read-out look jumpy, but that is the dictionary, not the
model. The old "redundancy ratios" fall below 1 even for perfectly additive effects, so they never measured
redundancy. Corrected for noise, the joint effect is 1.006 times the summed effect: additive. These edits are
small, so this says the model is smooth for small changes, not that its features never work together.
