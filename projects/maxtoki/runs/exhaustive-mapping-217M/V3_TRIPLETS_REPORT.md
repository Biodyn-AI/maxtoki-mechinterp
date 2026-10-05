# V3 triplets report — the triplet grid redone with v3 SAEs and correctly encoded inputs (item V3-4)

Date: 2026-10-02. Model: MaxToki-217M (HF safetensors, float32, Apple MPS). torch 2.11.0, transformers 5.5.4.
Hooks: `setup/hooks_v2.py`, unchanged (edit on the input of block l, SAE code from the live tensor, edits stack).
Inputs: `setup/inputs_v3.py` (counts / gene median order). SAEs: `runs/sae-atlas-217M/outputs/v3_sae` (item V3-1).
Design: the same as `V2_TRIPLETS_REPORT.md`, except for the feature rule (section 2.2).
Every number below comes from the files in section 9.

## 0. Results first

1. **The grid ran in full on correctly encoded inputs.** 512 real triplets (8 × 8 × 8 features at L0, L5, L9) and
   32 noise-floor triplets on 20 K562 control cells: 826 conditions × 20 cells = 16,520 ablation runs.
   Each run is read out at 4,928 L11 SAE features and 20,275 logits. All 20 cells passed the encoding check before
   every forward pass. The tokens v2 used fail the same check (20 of 20). None of the 20 v3 token sequences equals
   its v2 sequence.
2. **Still no three-way interaction (I_ABC) of a size that matters.**
   - Per target (t-test over cells, BH): **0 of 512** triplets have any L11 target at q < 0.05 across all
     2,523,136 tests (smallest p 6.7e-6). Within each triplet (4,928 tests): 4 of 512, one target each. Up to 25.6
     such triplets are expected by chance. The 4 hits are about 1e-8 in size, 0.007–0.09% of the joint effect at
     that target, and at the float32 rounding level (section 4.1).
   - Size: the noise-corrected share of the joint effect's energy that is three-way interaction is
     **0.0000028** (median over 512 triplets; 95% CI of the median [−0.00092, +0.00059]; cells resampled,
     1,000 reps). So at most about 0.06% of the energy. The v2 interval was 15 times wider ([−0.0145, +0.0094]).
     No triplet has its own CI above 0.
   - Joint vs additive energy: ρ3 = ‖e_ABC‖² / ‖e_A + e_B + e_C‖², noise-corrected, = **1.0037** [0.9977, 1.0112].
   - The tests had power. A planted interaction of 5% of each cell's own joint effect is found in 64 of 64 triplets
     by both the aggregate test and the per-target test (v2: 47 and 27 of 64).
3. **With more power, two small things now show that v2 could not see.**
   - **A weak excess of nominal hits.** The aggregate sign-flip test gives 44 of 512 triplets at p < 0.05.
     The triplets share cells, so I built a null that keeps that dependence: mean 25.1, SD 8.4; p = 0.029.
     The excess sits mostly on one L5 feature (4369: 15 of its 64 triplets; pooled test p = 0.002, BH q = 0.048
     over the 24 features). Its size is tiny: pooled energy share 0.00001, 95% CI [−0.00054, +0.00028].
     This was found after looking, and it is borderline.
   - **Two L5 → L9 feature pairs are sub-additive.** Removing L5 feature 3812 and L9 feature 110 together changes
     the L11 codes by 3.9% less energy than the sum of the two single removals (ρ2 = 0.961 [0.939, 0.980];
     1,537 targets at BH q < 0.05). Pair 1812 → 1628: 3.4% less (ρ2 = 0.966 [0.944, 0.993]; 623 targets).
     The interaction points partly against the L9 feature's own effect (cosine −0.38 and −0.40). That fits a
     simple reading: the L5 feature helps drive the L9 feature, so once it is gone there is less of the L9 feature
     left to remove. In v2, no pair had any target at BH q < 0.05.
4. **The model itself responds additively.** At the L11 residual stream, the three-way term is 5e-9 of the
   per-position energy of the joint change (median, 16 cell-triplet pairs; v2 2.5e-8). On the logits it is 7e-9.
   The L11 SAE codes look less additive per position (4.6%; v2 28%), and 99.99% of that sits on features that
   enter or leave the top 32. Each joint edit moves the L11 residual by 2.2% of its norm (median; v2 1.8%).
5. **The deployed "redundancy ratios" still do not measure redundancy.** Three-way ratio 0.743 (95% CI of the
   median [0.730, 0.746]); pairwise mean 0.851 ([0.836, 0.850]). With the joint effect set to exactly the sum of
   the singles, the same formula gives 0.767 and 0.865. Both are below 1 for the reasons v2 gave (opposite signs
   cancel across targets; noise inflates |mean|).
6. **Superadditivity (deployed 1.1× rule):** 3 of 6,738 targets with |e_ABC| > 0.01 (0.04%), in 3 of 512
   triplets. The deployed run reported 0 of ~2,980 for one triplet; v2 found 7 of 2,049.
7. **All sanity checks pass** (section 3). Zero edit = clean (0.0). AB ≠ B in 64 of 64 pairs and ABC ≠ C in 512 of
   512. Triplets with an exactly silent feature give I = 0.0 exactly. 440 stored conditions recomputed from
   scratch match to 1.1e-7.
8. **The analysis code is the v2 code.** Run on the stored v2 grid, it gives back 623 of 627 v2 summary numbers
   exactly and the other 4 to 2.6e-6 (relative). All 23 v2 power numbers come back exactly. So every v2 → v3
   change comes from the inputs and the SAEs, not from the analysis.

What this means for the paper: the v2 conclusion holds on correctly encoded inputs with retrained SAEs, and the
bound is now 15 times tighter. Removing three single SAE features at L0, L5 and L9 has an additive effect on the
model. The three-way term is at most about 0.06% of the joint effect's energy. One new, modest result: some
L5 → L9 feature pairs are partly redundant (3–4% of the joint energy), as expected when one feature feeds the
other. The edits are still small (2.2% of the residual norm), so this says the model is smooth for small edits.

---

## 1. Why a v3

- `V2_TRIPLETS_REPORT.md` fixed the hooks, but two things under it were wrong:
  - **Inputs.** `hooks_v2.load_k562_control_cells` ranked genes by log1p(CP10k) / median instead of counts /
    median (`checks/INPUT_ENCODING_AUDIT.md`). For these 20 cells the old and correct gene orders agree at
    Spearman 0.838 (0.764–0.874). Only 56% of the first 200 genes are the same (50–64%). 92% of the kept genes are
    the same (89–98%). Almost no gene is at its right position (0.07%). All 20 cells are cut at 2,046 genes, so the
    wrong order also changed which genes were kept.
  - **SAEs.** The deployed SAEs were trained on those wrong inputs. They do not fit correct inputs at deep layers
    (`runs/sae-atlas-217M/V3_SAE_REPORT.md`). This run uses the v3 SAEs at L0, L5, L9 (edits) and L11 (read-out).
- The hooks, the 20 cell rows, the grid design, the read-outs and the statistics are the same as v2.

## 2. What I ran

### 2.1 Cells and input encoding

- The same 20 K562 non-targeting control cells as v2: rows 289300, 292833, 293501, 294821, 296499, 296999,
  297671, 298043, 298575, 304014, 304434, 306407, 308960, 310091, 310869, 314183, 315807, 316745, 317964, 322713
  of `replogle_concat.h5ad` (pool of 100 drawn with numpy `default_rng(42)`, sorted, first 20 that tokenize).
  Barcodes are in `outputs/v3_triplets/cells.json`.
- Counts: the file has no raw counts. `X` is log1p(CP10k). `inputs_v3` takes expm1(X), checks each row is a
  whole-number multiple of its smallest value (max deviation 1.3e-4), and divides by that value. This is exactly
  proportional to counts, so the order is the counts / median order.
- Tokens: counts / row total × 1e4 / Geneformer gc104M gene median, ranked high to low, first 2,046 genes,
  `<bos>` … `<eos>`. Every cell is 2,048 tokens.
- **Encoding check.** `inputs_v3.check_encoding` runs on every cell right before its clean forward pass in every
  model stage (select, verify, run, recheck, readout). It checks that the token order equals the counts / median
  order, that no left-out gene ranks above a kept one, and that the input is count-like (log values cannot pass).
  Result: 20 of 20 pass, 0 order violations, 2 positions where exact ties are ordered differently from a
  stable sort (allowed). The stored counts and tokens are checked against their sha256 at the start of each stage.

### 2.2 Features (fixed before any ablation)

The v2 rule (4 annotated + 4 unannotated per layer) needs annotations made with the v3 SAEs. Those did not exist
when the features had to be fixed, so the item asked for a frequency rule instead:

- Candidates: v3 SAE features active (z > 0) in ≥ 1% of all tokens of the 20 clean cells (all positions).
- The log10 frequency of the candidates is cut into 8 equal-count bins (octiles). One feature is drawn at random
  from each bin with numpy `default_rng(20261001 + layer)`.
- Logged in `selection.json` at 13:40:16, before the first ablation (13:41:51).
- Candidates: L0 1,750; L5 460; L9 618 (v2: 1,733; 617; 1,046). Chosen frequencies 1.0% to 2.1% (v2: 1.0% to 4.1%).
- Every chosen feature is active in all 20 cells and none fires at `<bos>`.
- None of the 24 is a v2 feature: the best |cos| between a chosen v3 decoder column and the 8 v2 features of the
  same layer is ≤ 0.18. At L0 the chosen features look like deployed L0 features (best |cos| 0.86–0.95). At L5
  and L9 they do not (0.15–0.64). This matches V3_SAE_REPORT.
- The "annotated" column is a label from the v3 annotation of item V3-3
  (`circuit-tracing-217M/outputs/v3_circuit/annotation`), added after the choice. It is a description only.

| Layer | Feature | Octile | Freq. in 20 cells | Freq. in v3 SAE training | Active positions per cell | Single effect eff_X (×1e-4) | Annotated (post hoc) | Top term |
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

"Single effect eff_X" = mean over the 4,928 L11 targets of |cell-mean Δz| for that feature alone.

**Noise-floor features.** v2 used 2 features per layer with zero activations in the 20 cells. The v3 L5 SAE has
no such feature: all 4,928 fire somewhere in these cells. The first `select` call stopped there. Before any
ablation I added a fallback: use the 2 rarest L5 features instead. The octile rule was not changed.
- L0: 1581, 3049 (exactly silent). L9: 1995, 3537 (exactly silent).
- L5: 3437 (active at 4 positions in 3 cells) and 4481 (40 positions, 2 per cell in all 20 cells).
- Noise-floor triplets as v2: 8 all-null (2 × 2 × 2) and 24 with one null feature (8 each with null A, B, C;
  the null-B set uses 3437).

### 2.3 Conditions and read-outs (as v2)

- Ablate(l, f) removes z_f · W_dec[:, f] from the live input of block l at every position. Edits stack: the L5
  code is computed after the L0 edit, and so on.
- Conditions without an L0 edit start at block 5 or 9 from a cached live input. `verify` showed this gives the
  same numbers as the full pass (difference 0.0).
- Read-outs per condition and cell: L11 SAE code change Δz = mean over the 2,048 positions of (z_ablated − z_clean)
  (4,928 values); logit change, mean over positions, centred over the vocabulary (20,275 values);
  KL(clean ‖ ablated) of the next-token distribution, mean over positions. Position means are sums over dim 0
  (a full-tensor MPS sum was unreliable in v2).
- Effects: e_X[t] = mean over cells of Δz[t]. Three-way term I_ABC = e_ABC − e_AB − e_AC − e_BC + e_A + e_B + e_C,
  computed per cell first.

### 2.4 Statistics (as v2, plus three follow-ups)

- Resampling unit: the cell (n = 20). Bootstrap: 1,000 multinomial cell-weight vectors (seed 20261001), shared by
  all triplets and targets.
- Per-target I_ABC: one-sample t-test (df 19) and bootstrap p; BH at q < 0.05 within each triplet (4,928 tests)
  and across all 512 × 4,928 tests.
- Per-triplet aggregate test: T = Σ_t |mean over cells of I_ABC[t]|; null = 999 random per-cell sign flips
  (seed 20261002); BH across the 512 triplets.
- Noise-corrected energy: ‖true mean‖² estimated as ‖mean‖² − Σ_t var_t / n. Ratios of these get basic bootstrap
  CIs that pivot on the uncorrected value (the bootstrap of the corrected energy centres on the uncorrected one).
- New follow-ups (`scripts/v3_triplets_followup.py`, no model):
  - `countnull`: a null for the number of triplets with p < 0.05 that keeps the dependence between triplets
    (the same 999 sign flips applied to all triplets at once; each flip ranked inside each triplet's own values).
  - `perfeature`: for each of the 24 features, a pooled sign-flip test over its 64 triplets (BH over 24), and the
    pooled energy share and ρ3 with basic CIs.
  - `pairs`: all 192 pairwise terms I_XY = d_XY − d_X − d_Y with energy share, ρ2 and their CIs, and two
    mechanism read-outs for the later-layer feature.

## 3. Sanity checks (all pass)

| Check | What it tests | Result | Pass |
|---|---|---|---|
| Encoding | `inputs_v3.check_encoding` before every clean forward pass, every model stage | 20 of 20 cells pass, every stage; v2 tokens fail 20 of 20 | yes |
| Zero edit | `ZeroDelta` at L0, L5, L9 and all three vs clean, 2 cells (MPS); cell 0 again with CPU float64 sums | max abs logit diff **0.0**; max abs Δz **0.0**; KL 0.0 | yes |
| Clean repeat | two clean passes | 0.0 | yes |
| Fast path = full path | 2 conditions per condition type, 2 cells | max abs diff 0.0 (Δz and logits) | yes |
| float32 vs float64 position mean | 1 triple condition, 2 cells | max abs diff 8.0e-9 | yes |
| AB vs B | cell-mean Δz of AB vs B, all 64 (A, B) pairs | 0 of 64 equal; smallest max-abs difference 0.020 (median 0.049) | yes |
| ABC vs C | same, all 512 triplets | 0 of 512 equal; smallest 0.019 (median 0.049) | yes |
| Silent features | triplets with an exactly silent L0 or L9 feature (8 all-null, 8 null-A, 8 null-C) | I_ABC = **0.0 exactly** in every cell and target | yes |
| Near-silent L5 feature | 8 null-B triplets (L5 3437, 4 active positions in 3 cells) | I ≠ 0 only in those cells: max cell-mean \|I\| 2.4e-4; 0 of 8 with any CI excluding 0; 1 of 8 at aggregate p < 0.05 | yes |
| Twin conditions | a condition with a silent null feature must equal the same condition without it (separate passes, often separate chunks) | 2,144 of 2,144 bit-identical | yes |
| Results vary across triplets | SD of the three-way ratio across triplets | **0.042** (deployed 0.0001); correlation with the deployed artefact eff_C / Σeff is −0.49 | yes |
| Stored values | 2 random conditions in each of the 220 groups (440) recomputed from scratch, full model path, CPU float64 sums | max abs diff Δz 1.1e-7, logits 2.5e-8; KL within 0.15%; 220 of 220 stored clean fingerprints correct | yes |
| Read-out vs stored | position-mean Δz of ABC, 16 cell-triplet pairs (readout stage) | max abs diff 1.2e-8 | yes |

## 4. Results

### 4.1 Is there a three-way interaction?

| Test | v3 | v2 |
|---|---|---|
| Triplets with ≥ 1 target whose 95% CI of I_ABC excludes 0 (uncorrected) | 512 of 512 (488 per triplet) | 512 of 512 (563) |
| Same count when each cell's I gets a random sign (no true interaction) | 488 per triplet (ratio 1.004) | 563 (0.9995) |
| Triplets with ≥ 1 target at BH q < 0.05 within the triplet (t-test) | **4 of 512** (≤ 25.6 expected by chance) | 0 of 512 |
| Triplets with ≥ 1 target at BH q < 0.05 across all 2,523,136 tests | **0 of 512** | 0 of 512 |
| Same two counts with bootstrap p-values | 0 and 0 | 0 and 0 |
| Smallest t-test p over all tests | 6.7e-6 | 8.4e-4 |
| Targets with uncorrected p < 0.05 per triplet (mean) | 110 (5% of moving targets would be 246) | 36.8 (232) |
| Logits: triplets with any vocabulary logit at BH q < 0.05 (within) | 7 of 512 | 0 of 512 |
| Aggregate sign-flip test, p < 0.05 | **44 of 512** | 18 of 512 |
| … dependence-aware null for that count | mean 25.1, SD 8.4, 95th pct 40; **p = 0.029** | – |
| Aggregate sign-flip test, BH across 512 | 0 of 512 | 0 of 512 |
| T_obs / T_null median (median over triplets) | 1.001 (5–95%: 0.984–1.022) | 0.999 |
| All interaction orders (e_ABC − e_A − e_B − e_C): p < 0.05 / null count / BH | 28 / mean 25.1, SD 13.3, p = 0.38 / 0 | 16 / – / 0 |

**The 4 per-target hits.** Each is one target in one triplet:

| Triplet (L0, L5, L9) | Target | p | q within | q across all | Mean I | \|I\| / \|e_ABC\| at that target |
|---|---|---|---|---|---|---|
| 2518, 4369, 110 | 4731 | 9.9e-6 | 0.049 | 0.64 | 1.1e-8 | 0.00012 |
| 293, 4903, 1154 | 1404 | 7.6e-6 | 0.037 | 0.64 | −3.4e-8 | 0.00023 |
| 1910, 1812, 1154 | 1535 | 9.5e-6 | 0.047 | 0.64 | −9.7e-9 | 0.00007 |
| 1910, 4903, 235 | 4858 | 6.7e-6 | 0.033 | 0.64 | −1.5e-8 | 0.00094 |

- These are not evidence of a three-way interaction. BH within a triplet allows a 5% chance of a false hit per
  triplet when nothing is there. So up to 25.6 of 512 triplets would show one by chance; 4 do.
- Their size (1e-8) is below the float32 rounding of the stored values (the recheck differs by up to 1.1e-7).
  I did not re-run them in float64, because at 0.01–0.1% of the joint effect they do not matter either way.
- The t-test finds them now because the v3 read-out is much less noisy (section 4.3). Tiny, consistent shifts in
  all 20 cells give a small p even when they are 1e-8.

**The excess of nominal aggregate hits (44 of 512).** A binomial count would call 44 extreme. But triplets share
cells and conditions, so their p-values move together. The dependence-aware null (same sign flips for all
triplets) has mean 25.1 and SD 8.4, and 44 or more happens with p = 0.029. So there is a weak excess.

| Pooled over a feature's 64 triplets | L5 4369 | L9 1535 | All 512 triplets |
|---|---|---|---|
| Triplets at p < 0.05 | 15 of 64 | 10 of 64 | 44 of 512 |
| Pooled sign-flip p (same flips for all its triplets) | **0.002** | 0.008 | 0.181 |
| BH q over the 24 features | **0.048** | 0.096 | – |
| Pooled noise-corrected energy share of I_ABC | 0.000010 [−0.00054, +0.00028] | 0.000014 [−0.00074, +0.00040] | 0.0000021 [−0.00061, +0.00034] |
| Pooled ρ3 | 0.9998 [0.9960, 1.0035] | 1.0058 [0.9985, 1.0131] | – |

- The largest per-feature count (15, L5 4369) is above what the null gives for the most-hit of 24 features
  (null mean 7.9, 95th percentile 12; p = 0.018).
- One feature passes BH over 24, just (q = 0.048). The others have q ≥ 0.096.
- Even for 4369, the three-way term is at most 0.03% of the joint effect's energy (upper CI bound).
- This was found after looking at the results. It is borderline and it is tiny. I would not report it as a finding
  without new cells.

**A limit of the aggregate test (also true in v2).** With 999 sign flips the smallest possible p is 0.001.
BH over 512 triplets can then only reject if at least 11 triplets sit at that floor (over 192 pairs: at least 4).
So "0 of 512 after BH" in the aggregate test is partly built into the design. The dependence-aware count null and
the pooled tests above do not have this limit.

### 4.2 How big is the interaction?

Medians over the 512 triplets, with 95% basic bootstrap CIs of the median (cells resampled):

| Quantity | v3 L11 SAE codes | v3 logits | v2 L11 codes |
|---|---|---|---|
| Noise-corrected energy share of I_ABC in the joint effect | **0.0000028** [−0.00092, +0.00059] | about 1e-8 or less (see note) | −0.0000 [−0.0145, +0.0094] |
| Same, all interaction orders | 0.0000078 [−0.0041, +0.0027] | – | −0.0001 [−0.044, +0.027] |
| Triplets whose own CI of the I_ABC share is above 0 | 0 of 512 | – | 0 of 512 |
| Noise-corrected ρ3 = ‖e_ABC‖² / ‖e_A + e_B + e_C‖² | **1.0037** [0.9977, 1.0112] | 1.00055 [1.00047, 1.00067] | 1.006 [0.978, 1.051] |
| ρ3 without the noise correction | 0.998 | 1.00057 | 0.944 |
| Triplets with ρ3 CI above 1 / below 1 | 28 / 11 | – | 7 / 1 |
| Noise-corrected pairwise ρ2 (mean of AB, AC, BC) | 1.0020 [0.9993, 1.0063] | – | 1.003 [0.985, 1.027] |
| Per-cell I_ABC energy share (median over cells, then triplets) | 0.0091 | 2.1e-10 | 0.106 |

- Note on the logits: the uncorrected share is 1.6e-8 and the noise-corrected median is −9e-13. At this size the
  values are float32 rounding, and the bootstrap interval is not meaningful.
- On the logits ρ3 = 1.00055 excludes 1, by 0.055%. That is the pairwise part (section 4.4); it is negligible.
  v2 gave the same number (1.0006).
- 28 triplets have a ρ3 CI above 1 and 11 below. If the CIs were exact, about 13 per side would be expected.
  With 20 cells, the basic bootstrap CIs are likely somewhat too narrow, so I do not read much into this.
- Effect sizes are small. KL(clean ‖ ablated) per position, median over conditions: singles 2.0e-4, pairs 5.2e-4,
  triples 7.5e-4 nats (v2: 1.8e-4, 3.9e-4, 6.2e-4). They grow roughly with the number of edits (1 : 2.6 : 3.8).
- Joint vs single significance: for the joint edit ABC, a median of 274 of 4,928 targets per triplet pass BH
  (v2: 17).

### 4.3 Power

Planted interaction added to d_ABC of 64 seeded triplets (seed 20261003):

| Planted interaction | Aggregate sign-flip test (p < 0.05 / BH within 64) | Per-target t-test: triplets with ≥ 1 BH target (within / across 64) | v2, same columns |
|---|---|---|---|
| 5% of each cell's own joint effect | 64 / 64 | 64 / 64 | 47 / 46; 27 / 20 |
| 10% | 64 / 64 | 64 / 64 | 64 / 64; 39 / 38 |
| 20% | 64 / 64 | 64 / 64 | 64 / 64; 48 / 50 |
| 50% | 64 / 64 | 64 / 64 | 64 / 64; 56 / 59 |
| 100% | 64 / 64 | 64 / 64 | 64 / 64; 56 / 63 |

- A planted interaction that is the same in every cell is found even more easily: 2% of the cell-mean joint effect
  (0.04% of its energy) gives 64 of 64 by the aggregate test; its energy-share CI is above 0 in 5 of 64 at 2%,
  57 of 64 at 5% and 64 of 64 at 10%.
- **Why v3 has more power.** The v3 L11 SAE reads out more evenly. In a typical cell 4,768 of its 4,928 features
  are active somewhere (v2: 3,500). So a small edit moves most targets in every cell, and the change is consistent
  across cells. The median number of targets with a BH-significant single effect per triplet rose from 1.5–7.5
  (v2: A 7.5, B 1.5, C 2) to 284–1,502 (v3: A 415, B 284, C 1,502).

### 4.4 Pairwise interactions

192 pairs (64 each of AB, AC, BC). I_XY = d_XY − d_X − d_Y.

| Quantity | v3 | v2 |
|---|---|---|
| Pairs at aggregate sign-flip p < 0.05 | 17 of 192 (dependence-aware null mean 9.4, SD 3.8; p = 0.048) | – |
| Pairs passing BH (aggregate) | 0 of 192 (BH needs ≥ 4 pairs at the p floor) | 0 of 192 |
| Pairs with ≥ 1 target at BH q < 0.05 (within pair) | **29 of 192** (≤ 9.6 expected by chance) | 0 of 192 |
| Pairs whose CI of the I_XY energy share is above 0 | 0 of 192 | – |
| Pairs with ρ2 CI below 1 / above 1 | 5 / 12 | – |
| Median noise-corrected ρ2 | 1.0012 | 1.003 |
| \|I_XY\| share of \|e_XY\| (observed / sign-flip noise) | 0.0807 / 0.0811 | 0.285 / 0.285 |

Two pairs stand out. Both are an L5 feature followed by an L9 feature:

| Pair (L5 → L9) | Targets at BH q < 0.05 | ρ2 [95% CI] | Aggregate p | cos(I_XY, e_L9 feature) | L9 feature active positions per cell: alone → after the L5 edit |
|---|---|---|---|---|---|
| 3812 → 110 | 1,537 | 0.961 [0.939, 0.980] | 0.003 | −0.38 | 28.3 → 28.0 (changes in 7 of 20 cells) |
| 1812 → 1628 | 623 | 0.966 [0.944, 0.993] | 0.001 | −0.40 | 34.1 → 32.2 (changes in 15 of 20 cells) |

- The joint removal changes the L11 codes by 3.4–3.9% less energy than the two single removals added up.
- The interaction term itself is small (energy share 0.0013 and 0.0007, CIs include 0). The shortfall comes from
  its overlap with the additive part: it points partly against the L9 feature's own effect.
- Reading: the L5 feature helps drive the L9 feature. Once the L5 feature is removed, there is less of the L9
  feature left to remove. For 1812 → 1628 the L9 feature is active at fewer positions after the L5 edit. For
  3812 → 110 the count barely changes, so the effect is probably on activation values, which I did not record.
- The other 27 pairs with a per-target BH hit have 1–35 hits each. At those targets |I_XY| is 0.008–3.2% of the
  joint effect (median per pair). 24 of the 27 have a ρ2 CI that includes 1.
- 12 pairs have a ρ2 CI above 1 (the joint effect is 0.5–2.2% larger in energy than the sum); 6 of them involve
  L0 feature 763. 5 pairs have a CI below 1 (the two above, plus three with no per-target hit). If the CIs were
  exact, about 5 per side would be expected. With 20 cells the basic CIs are likely somewhat too narrow, and no
  pair passes the aggregate test after BH. I count these as weak signs, not findings.

### 4.5 Where does the per-cell non-additivity of the L11 codes come from? (read-out check)

Stage `readout`: the first 4 cells × 4 seeded triplets (seed 20261021; triplets (763, 4369, 110), (3035, 4903, 110),
(1910, 3812, 110), (1910, 4369, 2772)) × 8 full forward passes. Per-position energy share of I_ABC in the joint
change, median (range) over 16 cell-triplet pairs:

| Read-out | v3 per position | v3 position mean | v2 per position |
|---|---|---|---|
| L11 residual stream (input of `lm_head`, dense) | 5.0e-9 (1.8e-9 to 7.0e-4) | 8.7e-11 | 2.5e-8 |
| L11 SAE pre-activation (dense, linear in h) | 8.2e-9 (2.8e-9 to 6.3e-4) | 1.4e-10 | 3.8e-8 |
| Logits (centred) | 7.1e-9 (2.6e-9 to 7.6e-4) | 1.1e-10 | 2.7e-8 |
| **L11 SAE code (TopK 32)** | **0.046** (0.033 to 0.074) | 0.0068 | 0.28 |

- Share of the code-level I_ABC energy on entries where the top-32 set differs between conditions: **99.99%**
  (v2 99.99%). Share of the joint code change itself on such entries: 90% (v2 98.9%).
- Under ABC, 24% of positions change their top-32 set (v2 32%); 0.31 features swapped per position (v2 0.38).
- The joint change of the L11 residual is 2.2% of its norm (median; 1.6–4.1%).
- Plain reading: as in v2, the model responds additively. The v3 L11 SAE turns the same smooth change into
  fewer top-32 swaps than the deployed one, so its codes look more additive.

### 4.6 Superadditivity

| Rule | Pool | Superadditive | Other classes |
|---|---|---|---|
| Deployed: \|e_ABC\| > 1.1 × (\|e_A\| + \|e_B\| + \|e_C\|), targets with \|e_ABC\| > 0.01 | 6,738 targets pooled over 512 triplets (v2: 2,049) | **3 (0.04%)**, in 3 of 512 triplets (v2: 7, 0.34%) | – |
| Spec, strict: \|d_ABC\| > \|d_A\| + \|d_B\| + \|d_C\|, targets with \|d\| > 0.5 in any condition | 1,528,321 targets (60.6% of 2,523,136; v2 23.1%) | **5,407 (0.35%)**, in 512 of 512 triplets (v2: 1.8%) | – |
| Spec, banded (ratio of \|d\|) | same | > 1.1×: 3,344 (0.22%) | 0.9–1.1×: 7,168 (0.47%); < 0.9×: 1,517,809 (99.3%) |

- d is Cohen's d_z over 20 cells. The spec asked for 200 cells.
- As in v2, these rules have no valid null, so the counts are not evidence for or against synergy.
  The inclusion rule now keeps 60.6% of targets because effects are consistent across cells (pure noise would keep
  23.5%). "Sub-additive" is the usual outcome for |d_ABC| vs a sum of three |d|: with the deployed raw-mean rule at
  0.9×, an exactly additive joint effect is already called sub-additive for 44.7% of included targets (observed
  56.7%).
- I did not repeat the v2 sign-flip null for these rules, because v2 showed it is biased.

### 4.7 Redundancy ratios (deployed formula)

Distribution across the 512 triplets; CI = basic bootstrap 95% CI of the median (cells resampled):

| Ratio | Median [95% CI of median] | 5%–95% across triplets | SD across triplets | If exactly additive (median) | v2 median | Deployed (4 triplets) |
|---|---|---|---|---|---|---|
| Pairwise AB | 0.797 [0.790, 0.806] | 0.729–0.876 | 0.044 | – | 0.731 | 0.165 |
| Pairwise AC | 0.888 [0.873, 0.889] | 0.853–0.937 | 0.024 | – | 0.783 | 0.219 |
| Pairwise BC | 0.864 [0.838, 0.856] | 0.840–0.896 | 0.018 | – | 0.769 | 0.586 |
| Pairwise mean | **0.851** [0.836, 0.850] | 0.818–0.895 | 0.022 | 0.865 | 0.766 | 0.3235 |
| Three-way | **0.743** [0.730, 0.746] | 0.686–0.838 | 0.042 | 0.767 | 0.639 | 0.1896 |
| Marginal C given AB | 0.440 [0.415, 0.450] | 0.328–0.533 | 0.060 | – | 0.253 | 0.294 |
| Three-way, logits | 0.758 [0.751, 0.770] | 0.670–0.900 | 0.065 | – | 0.678 | – |

- For a few ratios the noise bias is about as large as the spread, so the basic CI does not contain the plain
  estimate (pairwise BC, pairwise mean), as in v2.
- All 512 triplets have a three-way CI below 1. That does not mean redundancy: the formula gives 0.767 for an
  exactly additive joint effect and 0.758 on the logits, where the response is additive to about 1e-8.
- Observed minus additive three-way ratio: −0.023 (CI of the median [−0.019, −0.010]; v2 −0.067). This gap is
  noise (e_ABC is one noisy condition; e_A + e_B + e_C sums three, so its |mean| is inflated more). The
  noise-corrected ρ3 removes it: 1.0037.
- Ratios moved up from v2 because the v3 read-out is less noisy. They are still not a measure of redundancy.
- Breakdowns: by frequency half (octiles 0–3 vs 4–7 at each layer) the median three-way ratio is 0.715–0.779 and the
  median I_ABC energy share is −0.00002 to +0.00001 in all 8 groups. By the post-hoc annotation label it is
  0.734–0.761 and −0.00001 to +0.00002. No group has any target passing BH across all tests.

## 5. What changed versus the deployed result and v2

| Item | Deployed | v2 (fixed hooks, wrong inputs, deployed SAEs) | v3 (fixed hooks, right inputs, v3 SAEs) |
|---|---|---|---|
| Input order | log1p(CP10k) / median | same (wrong) | counts / median; checked on every cell |
| SAEs | deployed | deployed | v3 (retrained on right inputs) |
| Triplets | 4 | 512 + 32 noise floor | 512 + 32 noise floor |
| Feature rule | greedy, shared-pathway list | 4 annotated + 4 unannotated per layer, frequency bins | 8 per layer, one per frequency octile |
| Cells | 50 | 20 | same 20 rows |
| AB vs B, ABC vs C | identical (hook bug) | differ 64/64, 512/512 | differ 64/64, 512/512 |
| Triplets with a target at BH q < 0.05 across all tests | not computed | 0 of 512 | **0 of 512** |
| Same, within triplet | – | 0 of 512 | 4 of 512 (one target each, ~1e-8; ≤ 25.6 expected) |
| Aggregate sign-flip p < 0.05 | – | 18 of 512 | 44 of 512 (dependence-aware p = 0.029; one L5 feature, q = 0.048) |
| I_ABC energy share | – | −0.0000 [−0.0145, +0.0094] | **0.0000028 [−0.00092, +0.00059]** |
| ρ3 (noise-corrected) | – | 1.006 [0.978, 1.051] | **1.0037 [0.9977, 1.0112]** |
| Power, 5% planted (aggregate / per-target) | – | 47 / 27 of 64 | 64 / 64 of 64 |
| Pairs with any BH target | – | 0 of 192 | 29 of 192; two L5 → L9 pairs 3–4% sub-additive |
| Superadditive (1.1× rule) | 0 of ~2,980 (forced) | 7 of 2,049 | 3 of 6,738 |
| Pairwise mean ratio | 0.3235 (SD 0.0) | 0.766 | 0.851 (if additive 0.865) |
| Three-way ratio | 0.1896 (SD 0.0001) | 0.639 | 0.743 (if additive 0.767) |
| Residual-stream I share per position | – | 2.5e-8 | 5.0e-9 |
| L11 code I share per position | – | 0.28 | 0.046 |
| \|Δh11\| / \|h11\| for ABC | – | 1.8% | 2.2% |

Why things changed:
- The conclusion did not change. The numbers got tighter because the v3 L11 SAE read-out is less noisy across
  cells (section 4.3).
- The extra power makes very small effects visible: a borderline three-way excess on one L5 feature (≤ 0.03% of
  the energy) and two modest L5 → L9 pairwise redundancies (3–4%).
- The analysis code did not change (section 6.2). Feature IDs differ between v2 and v3 (different SAEs), so only
  design-level numbers can be compared.

## 6. Verification: key numbers derived a second way

### 6.1 Independent code on the v3 data (`crosscheck`)

Separately written code: conditions read straight from the cell files into a dictionary, `scipy.stats.ttest_1samp`,
BH as a step-up rejection rule, the noise-corrected energy from the pairwise-product (U-statistic) formula, and a
new sign-flip seed.

| Number | analyze | crosscheck |
|---|---|---|
| Triplets with any BH target within / across | 4 / 0 of 512 | 4 / 0 of 512 |
| Smallest p | 6.708e-6 | 6.708e-6 |
| Median ρ3 | 1.0037108 | 1.0037108 (max relative difference over triplets 9e-16) |
| Median I_ABC energy share | 2.7829e-6 | 2.7829e-6 (max difference 2e-18) |
| Aggregate sign-flip p < 0.05 / BH (new seed 20261099) | 44 / 0 | 44 / 0 |
| CI of the median energy share | basic [−0.00092, +0.00059] | new seed, percentile [0.00062, 0.00223]; as a basic interval around the uncorrected median 0.00123: [−0.00100, +0.00061] |

The percentile interval sits around the uncorrected value, as expected (section 2.4); turned into a basic
interval it matches the analyze interval.
The `countnull` stage recomputed all 512 triplet and 192 pair sign-flip p-values: identical to analyze (max
difference 0.0).

### 6.2 The v3 analysis code on the v2 data (`replay`)

The v3 `analyze` and `power` code, run on the stored v2 grid, gives back the v2 results:
- 627 numbers of `v2_triplets/summary.json` compared: 623 identical, 4 differ by at most 2.6e-6 (relative; float32
  bootstrap sums in a different order). Only the v2 annotation breakdown is missing (v3 uses the frequency rule).
- 23 of 23 power numbers in `v2_triplets_followup/align.json` and `classnull.json`: identical.

So the v2 → v3 differences are due to the inputs and the SAEs.

## 7. What I could NOT do, and limits

- **20 cells, not the spec's 200.** Same trade-off as v2 (cells for triplets).
- **Small edits only.** Each edit removes one feature active in 1.0–2.1% of tokens. The joint change of the L11
  residual is 2.2% of its norm. A smooth network is close to linear here. I did not test larger edits.
- **One read-out layer** (L11 codes and logits), as deployed and v2.
- **No v3 annotation was used to choose features.** It did not exist when the features were fixed. The label in
  section 2.2 is post hoc and descriptive. Annotation rates in this project are near chance (v2b annotation FDR),
  so I would not read biology into it.
- **The L5 noise floor is near-null, not null.** The v3 L5 SAE has no feature silent in these 20 cells. The
  exact-zero checks hold for the L0 and L9 null features only.
- **The L5 4369 excess is post hoc and borderline** (q = 0.048 over 24 features). I did not test it on new cells.
- **The pairwise mechanism is inferred, not measured.** I used active-position counts and the cosine with the L9
  feature's single effect. I did not record the L9 activation values under the L5 edit.
- **The 4 per-target hits were not re-run in float64.** Their size (1e-8) is below the stored float32 rounding.
- **No valid null for the two superadditivity rules** (as v2).
- **The basic bootstrap CIs with 20 cells are likely somewhat too narrow** (28 + 11 triplets have a ρ3 CI that
  excludes 1, against about 26 if the CIs were exact).
- **The main script changed once during the `run` stage.** Chunks 1–5 (74 of 220 groups: cells 0–5 and cell 6
  groups 0–7) ran with `v3_triplets.py` sha256 73ada252…; the file was then edited while later stages were being
  written, and chunks 6–17 and all later stages ran with 94e31c03…. The old text was not kept, so I cannot diff
  it. The checks that cover both parts all pass: the recheck covers all 220 groups (440 conditions, max
  difference 1.1e-7), 2,144 of 2,144 twin conditions are bit-identical, 220 of 220 clean fingerprints are
  correct, and the read-out of cells 0–3 (all from the first part) matches the stored values to 1.2e-8.
- **Another project's job ran on the machine** from about 13:45 to 15:55. It slowed the grid from ~14 s to ~38 s
  per group of conditions. It did not change any result (every determinism check gave 0.0).

## 8. Run cost

- Model stages (MPS): select 0.3 min, verify 0.7 min, run 113.2 min in 17 chunks of ≤ 7.3 min (16,520 ablation
  runs), recheck 14.7 min in 2 chunks, readout 2.6 min.
- CPU stages: analyze 8.0 min (2 calls), power 0.3 min, crosscheck 0.2 min, compare < 0.1 min; follow-up countnull
  0.3, pairs 0.6, perfeature 0.9, replay 6.2 min.
- Memory: guard at 3 GB free before every group of conditions; the lowest free memory seen was 3.54 GB.
  analyze peak footprint 4.42 GB (`/usr/bin/time -l`). Model stages were not measured with `/usr/bin/time`.
- CPU threads capped at 4 (`OMP_NUM_THREADS=4`, `torch.set_num_threads(4)`).

## 9. Files

Scripts (`runs/exhaustive-mapping-217M/scripts/`):
- `v3_triplets.py` — stages `select`, `verify`, `run` (resumable), `recheck`, `readout`, `analyze`, `power`,
  `crosscheck`, `compare`.
- `v3_triplets_followup.py` — stages `countnull`, `pairs`, `perfeature`, `replay` (no model).

Shared code used: `setup/inputs_v3.py` (sha256 019e3834…, not changed), `setup/hooks_v2.py` (sha256 1ba6fda5…,
not changed). SAEs: `runs/sae-atlas-217M/outputs/v3_sae/layer_{00,05,09,11}/sae_final.pt` (sha256 in run_config).

Outputs, `runs/exhaustive-mapping-217M/outputs/v3_triplets/`:
- `run_config.json` (device, versions, seeds, cell rows and barcodes, counts and token sha256, input-encoding record
  and check results, features and null features, SAE / hooks / inputs sha256, wall time, free memory and script
  sha256 per chunk, notes), `selection.json` (rule, bins, choice, post-hoc labels), `cells.json`,
  `cells_tokens.npz`, `cells_counts.npz`, `clean_frequencies.npz`.
- `verify.json`, `recheck/cell*.json`, `readout/cell*.json`, `readout_summary.json`.
- `cells/cell{ci}_g{gi}.npz` — raw per-cell results (condition keys, Δz 4,928, Δlogit 20,275, KL, live n_active).
- `summary.json`, `per_triplet.json`, `per_pair.json`, `interaction_terms_real.npz`, `power.json`,
  `crosscheck.json`, `compare.json`, `bootstrap_cell_weights.npy`, `signflip_signs.npy`, `analyze_cache/`.

Outputs, `runs/exhaustive-mapping-217M/outputs/v3_triplets_followup/`:
- `countnull.json`, `pairs.json`, `perfeature.json`, `v2_replay_compare.json`, `v2_replay/` (the v3 analysis code
  run on v2 data; for the code check only), `run_config.json`.

## Plain-words summary

The v2 test of "do three features work together?" fed the model genes in the wrong order and used feature
dictionaries trained on that wrong input. I redid it with the right gene order and the retrained dictionaries.
Every cell passed the order check before every model run. Same design: 512 groups of three features (one each
from layers 0, 5 and 9), 20 cells, each removed alone, in pairs and all together. All checks pass: an empty edit
changes nothing, combined edits really combine, silent features do nothing, and stored results match a fresh
recomputation. The answer is the same as before, only sharper. Removing three features together does what you get
by adding up the single removals. The three-way part is at most about 0.06% of the total change, and the test
would have found a planted three-way effect of 5% of the joint change in every group tested. Because the new read-out is much less noisy, two
small things now show up. One layer-5 feature shows a tiny, borderline three-way effect (at most 0.03%), found
after looking. And two layer-5 / layer-9 feature pairs overlap: removing both does 3–4% less than the two single
removals added up, as if the first feature feeds the second. The old "redundancy ratios" still fall below 1 even
for perfectly additive effects, so they still measure nothing. I also ran the new analysis code on the old data and
got the old numbers back, so the changes come from the corrected inputs and dictionaries, not from the code. These
edits are small, so the result says the model is smooth for small changes, not that its features never combine.
