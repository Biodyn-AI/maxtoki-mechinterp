## Steering along an erythroid maturity score (MaxToki-217M, Tabula Sapiens bone marrow)

This analysis asks whether single SAE features act as maturity switches. Steering here means: while the model reads a cell, scale the part of the hidden state that one SAE feature writes, at every gene position, and measure how far the model's output moves along a maturity score. A feature is a switch only if it moves cells further, in its predicted direction, than random features of similar activity. The deployment reported steering effects and gene lists at five layers; its hook deleted a transformer block (at layer 11 it applied the final normalisation twice), and with a zero edit it gives the same table (main text). Everything below (differentiation series, cells, read-out, features, null, edits and tests) was written to `design.json` at 08:53:02, before the first forward pass at 08:53:38, and was not changed. The design fixed how each p value is computed, but it set no pass threshold. Analyses added after the first look at the results are marked "post hoc".

Here, moving a cell means moving the model's output score for that cell. No change in the cell's real state was measured.

### Data

- **Model.** MaxToki-217M, Hugging Face safetensors, float32, Apple MPS, one cell per sequence. torch 2.11.0, transformers 5.5.4.
- **SAEs.** The SAEs of section 02 (trained on 500 correctly encoded K562 control cells; 4,928 features; TopK 32), at layers 0, 3, 6, 9 and 11. The site of layer l is the input of block l (`hidden_states[l]`); layer 11 is the input of `lm_head` (section 00).
- **Cells.** Tabula Sapiens immune (`tabula_sapiens_immune.h5ad`), tissue = bone marrow, assay = 10x 3' v3 only. Counts from `raw/X`, encoded as in section 00 (at most 2,048 tokens).
- **Differentiation series.** The erythroid branch of blood development, with stage numbers from the `cell_type` annotation: 0 hematopoietic stem cell (HSC), 1 common myeloid progenitor (CMP), 2 erythroid progenitor cell (EryP), 3 erythrocyte. Stage numbers are ordinal labels, not measured time. Left out: "hematopoietic precursor cell" (its free annotations mix granulocytes, monocytes and other cells), Smart-seq2 cells (read counts, which MaxToki-217M handles badly) and 10x 5' v2 cells (so that one assay covers every stage).
- **Cell groups** (numpy `default_rng(20261003)`; groups drawn in a fixed order; 442 cells in total):

| Group | Donors | HSC | CMP | EryP | Erythrocyte | Used for |
|---|---|---:|---:|---:|---:|---|
| Selection | TSP14, TSP21, TSP25 (up to 30 per stage and donor) | 76 | 90 | 76 | 90 | maturity score, span, choice of candidates and null features |
| Steered | TSP2 (47), TSP27 (3) | 50 | – | – | – | steering |
| Axis check | TSP2, TSP27 | the 50 steered HSCs | 30 | 30 | 0 | held-out check of the score (clean passes only) |

- The steering donors are disjoint from the selection donors. TSP27 has only 9 HSCs, so 47 of the 50 steered HSCs come from TSP2. TSP2 and TSP27 have no erythrocytes, so the held-out check uses EryP as the late group.
- Median tokens per cell: 2,048 in every group except selection erythrocytes (1,796).

### Method

1. **Read-out.** For each cell, m = the next-gene logits averaged over all positions (20,275 tokens). g_early = mean m of the 76 selection HSCs; g_late = mean m of the 90 selection erythrocytes. Maturity score s(m) = cos(m, g_late) − cos(m, g_early). The effect of an edit is Δs = s(steered) − s(clean) for the same cell, both computed by the same code path. Δs has no unit.
2. **Span.** S = mean s of selection erythrocytes − mean s of selection HSCs = 0.198. Effects are also given as a percentage of S. A shift of 100% equals the full gap between the mean HSC and mean erythrocyte scores in the selection donors. (The steered HSCs come from other donors and start higher, at a mean s of −0.077 against −0.104.)
3. **Second read-out.** Gap share = (m_steered − m_clean) · (g_late − g_early) / |g_late − g_early|²: the share of the straight line from g_early to g_late that the edit moves the cell.
4. **Axis check (rule set in advance).** On axis-check cells: AUROC of s for EryP against HSC ≥ 0.80, and Spearman(s, stage) > 0 over HSC, CMP and EryP. Steering would not have run if this failed.
5. **Feature activity.** Per cell, a_f = mean over gene positions (not `<bos>`/`<eos>`) of the SAE code z_f. A feature is alive if z_f > 0 at ≥ 1 gene position in ≥ 50% of selection HSCs. Alive features: L0 2,481; L3 4,912; L6 4,370; L9 3,348; L11 4,190 (of 4,928).
6. **Candidates.** ρ_f = mean over the 3 selection donors of the within-donor Spearman correlation between a_f and stage (0–3). The candidates are the 3 alive features with the largest |ρ_f| at each layer (ties: lower feature ID). 15 candidates in total.
7. **Random-feature null.** 20 other alive features per layer, drawn at random without replacement (`default_rng(20261003 + layer)`): 7, 7 and 6 from the 50 alive features closest to each candidate in log10(mean a_f over selection HSCs). Mean activity sets the size of the edit, so null edits are about as large as candidate edits in selection HSCs.
8. **Edit.** `hooks_v2.Steer(layer, f, α, mode="scale")` at gene positions only: added vector = (α − 1) · z_f(x) · W_dec[:, f], with z computed from the live input at the hook. α = 0 removes the feature, α = 2 doubles it, α = 5 multiplies it by 5; α = 1 adds nothing. Predicted sign: for α = 2 and 5, sign(Δs) = sign(ρ_f); for α = 0, −sign(ρ_f). Edit size = mean L2 norm of the added vector per gene position.
9. **Positive controls.** `hooks_v2.AddVector` at gene positions, with v = mean gene-position hidden state at the site over selection erythrocytes minus that over selection HSCs. Two sizes: the full v ("full control"), and v scaled to the planned α = 5 edit size of the layer's candidates, 4 × the mean over the 3 candidates of mean a_f in selection HSCs ("size-matched control").
10. **Runs.** 50 cells × 5 layers × (23 features × 3 α + 2 controls) = 17,750 steered passes, plus 442 clean passes. Exact speed-up: an edit at site l changes nothing before block l, so each steered pass runs only blocks l to 10, starting from the stored clean input of block l, and computes the position-mean logits as `lm_head`(mean of the normed hidden states) (`lm_head` is linear with no bias). Every row is compared with a clean reference computed by the same path.
11. **Statistics.** 95% intervals: percentile bootstrap over the 50 steered cells, 10,000 resamples, seed 20261003 (the same resamples for every row), unless stated otherwise. Directional p per candidate and α = (1 + number of null features with an effect at least as large in the predicted direction) / (1 + 20); the smallest possible p is 1/21 = 0.048. Size p uses |Δs|. The design fixed these p values but no pass threshold. Below, a candidate is called a hit if its directional p is 0.048 (no null feature of its layer as large in the predicted direction) at any α; only L9 F2903 reaches this, and it does so at every α. Layer test: the 3 candidates against all C(23, 3) = 1,771 sets of 3 of the layer's 23 tested features (each feature signed by its own ρ). Pooled tests: (a) the candidates' mean within-layer directional rank, summed over the 5 layers, with labels permuted within layers (100,000 permutations); this is the permutation test that `design.json` names. (b) The mean over the 5 layers of the within-layer Spearman(ρ_f, Δs_f) over 23 features, also with labels permuted within layers (100,000 permutations), with one-sided and two-sided p. `design.json` also names a Spearman over all 115 features pooled; that number is not reported.
12. **Post-hoc analyses** (added after the first look at the results; marked "post hoc" below): the gap-share comparison with the null; the per-unit-of-edit comparison (mean Δs / mean edit size); a donor split; comparison with all 100 null features pooled over layers; the cosine between each mean logit change and g_late − g_early; the number of erythroid genes among the 10 most-raised genes (14-gene list: AHSP, ALAS2, HBM, GYPA, SLC4A1, HBG1, HBG2, HBZ, HBD, GYPB, KLF1, HBA1, HBA2, HBB); and gene descriptions of the candidates in 32 selection cells (8 per stage, all from donor TSP14). "Most-raised genes" = largest mean increase of the position-mean logit over the 50 cells at α = 5.

### Checks

| Check | What it tests | Result | Pass |
|---|---|---|---|
| Encoding, all cells | token order = counts / median order; counts are count-like (section 00) | 442 / 442 pass; 0 order violations (38 positions with exact ties, allowed) | yes |
| Encoding, negative control | the input order of the deployed analysis (stored values ranked as if they were counts) must fail | fails for 442 / 442 | yes |
| Encoding, before each forward pass | counts re-read from `raw/X`, sha256 equal to the prepared counts, check run again | all clean, steered and verify cells pass | yes |
| Axis check (rule set in advance) | held-out donors: AUROC(EryP vs HSC) ≥ 0.80 and Spearman(s, stage) > 0 | 0.991 and 0.844 | yes |
| Speed-up = full forward pass | 75 rows (cells 332–334; candidates, null features, controls) against `hooks_v2.run_with_edits` | largest Δs difference 2.2 × 10⁻⁹; largest logit difference 1.9 × 10⁻⁶ | yes |
| Zero edit | α = 1 and `ZeroDelta`, speed-up path and full path | logits and Δs change by 0.0 exactly (18 rows) | yes |
| Silent feature | a feature with z = 0 at every position | change 0.0 exactly (15 rows) | yes |
| CPU vs MPS | cell 332, L9 F2903, α = 5 | Δs difference 7.0 × 10⁻¹⁰ | yes |
| Silent rows in the main run | every (cell, feature, α) row where the feature is inactive | 807 rows; largest \|Δs\| 0.0 and largest logit change 0.0 | yes |
| Clean reference vs full pass | speed-up clean logits vs full clean logits, all cells and layers | largest difference 1.9 × 10⁻⁶ | yes |
| Edit applied when active | edit size > 0 in every row where the feature is active | yes | yes |
| Memory | free memory checked before each cell; stop below 3 GB | lowest free memory 5.8 GB; no chunk stopped | yes |

### Results

**The maturity score.**

| Quantity | Value |
|---|---|
| cos(g_early, g_late) | 0.894 (the two signatures are close, so s moves in small numbers) |
| Selection cells, mean s (SD 0.03–0.05 per stage) | HSC −0.104; CMP −0.009; EryP +0.063; erythrocyte +0.094 |
| Span S | 0.198 |
| Within-donor Spearman(s, stage), selection donors | TSP14 0.86; TSP21 0.73; TSP25 0.89 |
| Held-out donors, mean s | HSC −0.077; CMP +0.001; EryP +0.079 |
| Held-out AUROC (0.5 = chance) | EryP vs HSC 0.991; CMP vs HSC 0.917; EryP vs CMP 0.887 |
| Held-out Spearman(s, stage), HSC/CMP/EryP | 0.844 [0.782, 0.885] (percentile bootstrap over the 110 cells, 2,000 resamples) |
| Within-stage Spearman(s, number of tokens), selection cells | HSC −0.42; CMP −0.41; EryP +0.05; erythrocyte +0.27 |

AUROC is the probability that a random EryP cell scores above a random HSC.

**How well the K562-trained SAEs fit these cells.** Fraction of variance explained (FVE), all positions:

| Layer | Steered HSCs | Selection cells | Held-out K562 tokens (section 02) |
|---|---:|---:|---:|
| L0 | 0.46 | 0.50 | 0.700 |
| L3 | 0.59 | 0.62 | 0.912 |
| L6 | 0.37 | 0.39 | 0.847 |
| L9 | 0.34 | 0.37 | 0.859 |
| L11 | 0.43 | 0.42 | 0.871 |

A large part of what these bone-marrow cells carry is not in any single feature.

**Positive controls: the read-out can move.** Δs at α = 5 for the null; controls as described in Method step 9.

| Layer | Null Δs: mean (SD) | Null range | Null mean \|Δs\| (% of span) | Candidates mean \|Δs\| (% of span) | Size-matched control, % of span [95% CI] | Candidates / size-matched control | Edit size: candidates / null / size-matched control | Full control, % of span [95% CI] | Full control gap share |
|---|---|---|---|---|---|---|---|---|---|
| L0 | +1.0e-4 (3.0e-4) | [−4.9e-4, +6.9e-4] | 0.12% | 0.095% | 2.91% [2.59, 3.26] | 0.033 | 0.0045 / 0.0046 / 0.0046 | 23.7% [21.6, 25.8] | 0.26 |
| L3 | +2.8e-4 (7.2e-4) | [−2.2e-4, +2.4e-3] | 0.18% | 0.043% | 0.63% [0.57, 0.70] | 0.068 | 0.094 / 0.105 / 0.100 | 73.4% [70.3, 76.3] | 0.71 |
| L6 | +9.9e-6 (2.6e-5) | [−2.5e-5, +1.1e-4] | 0.008% | 0.004% | 0.09% [0.09, 0.10] | 0.040 | 0.031 / 0.031 / 0.021 | 73.1% [70.3, 75.9] | 0.71 |
| L9 | −7.2e-7 (1.1e-3) | [−2.9e-3, +1.9e-3] | 0.33% | 1.65% | 3.90% [3.77, 4.03] | 0.42 | 2.80 / 1.24 / 1.43 | 89.3% [87.4, 91.1] | 0.85 |
| L11 | −6.9e-5 (2.5e-4) | [−7.1e-4, +3.6e-4] | 0.075% | 0.039% | 1.36% [1.30, 1.42] | 0.029 | 0.45 / 0.74 / 0.63 | 109.5% [103.3, 115.6] | 1.00 |

- The full control moved 50 of 50 HSCs toward the erythrocyte signature at every layer, by 24% (L0) to 109% (L11) of the span on average (means over the 50 cells; single cells vary around these means). Its edits are far larger than the feature edits (24.9 against 3.44 per gene position for L9 F2903 at L9). At L11 the gap share is 1.000 and the cosine between the logit change and g_late − g_early is 0.9999996. This must hold, because `lm_head` is linear, so the L11 result checks the read-out code, not the model.
- At the planned size of the candidate edits, the same direction moved cells 0.09–3.9% of the span. Except at L9, the candidates reached 3–7% of what this direction reached at about the same edit size. At L9 the candidates' actual edits (mean 2.80 per gene position) were about twice the planned size (1.43).
- The full control raised AHSP, HBM, ALAS2, SLC4A1 and GYPA at every layer; from L3 on, all five are among its 10 most-raised genes (at L0, SLC4A1 and GYPA are not). The size-matched control has all five in its top 10 only at L9 and L11. At L0 it raised mostly other genes (LCN2, HBG2, RNASE1). Five of these genes (AHSP, ALAS2, HBM, SLC4A1, GYPA) are among the 15 genes highest in g_late − g_early itself (AHSP, ALAS2, HBM, LGALS3, SELENBP1, SLC4A1, MYL4, GYPA, KLF1, CTSE, HBZ, HBQ1, RNASE1, HEPACAM2, CD36).

**Candidates against the random-feature null.** Δs = mean over the 50 steered HSCs. ρ = maturity correlation in selection cells. p values are against the 20 random features of the same layer (smallest possible 0.048).

| Layer | Feature | ρ | Cells active | Δs α = 0 | Δs α = 2 | Δs α = 5 [95% CI] | α = 5 as % of span [95% CI] | Directional p (α = 0 / 2 / 5) | Size p (α = 5) |
|---|---|---|---|---|---|---|---|---|---|
| L0 | 1976 | +0.73 | 50/50 | −4.7e-5 | +5.1e-5 | +1.9e-4 [+1.1e-4, +2.7e-4] | +0.094 [+0.055, +0.134] | 0.38 / 0.38 / 0.43 | 0.57 |
| L0 | 1883 | +0.72 | 50/50 | +1.6e-5 | +1.6e-5 | +2.7e-4 [+1.3e-4, +4.3e-4] | +0.138 [+0.066, +0.216] | 0.81 / 0.62 / 0.29 | 0.38 |
| L0 | 2449 | −0.68 | 50/50 | −1.1e-6 | +2.8e-6 | +1.0e-4 [−1.0e-5, +2.2e-4] | +0.052 [−0.005, +0.110] | 0.38 / 0.33 / 0.62 | 0.62 |
| L3 | 1133 | −0.81 | 50/50 | +1.3e-5 | +6.1e-6 | +1.5e-4 [+8.5e-5, +2.2e-4] | +0.077 [+0.043, +0.112] | 0.38 / 0.52 / 0.81 | 0.29 |
| L3 | 980 | −0.72 | 50/50 | +1.0e-5 | −1.2e-5 | −8.2e-5 [−9.8e-5, −6.5e-5] | −0.041 [−0.050, −0.033] | 0.38 / 0.33 / 0.19 | 0.57 |
| L3 | 1029 | −0.72 | 50/50 | +1.4e-5 | −9.5e-6 | −1.9e-5 [−2.7e-5, −1.0e-5] | −0.010 [−0.014, −0.005] | 0.38 / 0.38 / 0.38 | 0.90 |
| L6 | 3965 | +0.77 | 50/50 | −3.4e-6 | +2.9e-6 | +1.4e-5 [+9.9e-6, +2.0e-5] | +0.007 [+0.005, +0.010] | 0.33 / 0.38 / 0.38 | 0.43 |
| L6 | 1976 | +0.70 | 41/50 | +2.0e-6 | −1.3e-6 | −8.3e-7 [−5.0e-6, +5.2e-6] | −0.0004 [−0.003, +0.003] | 0.81 / 0.81 / 0.76 | 1.00 |
| L6 | 646 | +0.69 | 36/50 | −1.2e-6 | +1.4e-6 | +7.0e-6 [+4.1e-6, +1.1e-5] | +0.004 [+0.002, +0.005] | 0.43 / 0.43 / 0.43 | 0.57 |
| **L9** | **2903** | **+0.84** | **42/50** | **−1.7e-3** | **+1.7e-3** | **+6.9e-3 [+3.8e-3, +1.0e-2]** | **+3.47 [+1.94, +5.18]** | **0.048 / 0.048 / 0.048** | **0.048** |
| L9 | 878 | −0.73 | 50/50 | −9.4e-5 | +4.0e-4 | +2.9e-3 [+1.9e-3, +3.9e-3] | +1.47 [+0.95, +1.95] | 0.81 / 0.95 / 1.00 | 0.048 |
| L9 | 3003 | +0.70 | 42/50 | +1.9e-6 | −1.6e-6 | −4.8e-6 [−6.6e-6, −3.2e-6] | −0.002 [−0.003, −0.002] | 0.57 / 0.57 / 0.57 | 0.86 |
| L11 | 2767 | −0.75 | 50/50 | −8.0e-6 | +8.1e-6 | +3.3e-5 [+2.6e-5, +4.1e-5] | +0.017 [+0.013, +0.021] | 0.76 / 0.76 / 0.76 | 0.57 |
| L11 | 4372 | +0.75 | 39/50 | −5.5e-6 | +5.5e-6 | +2.2e-5 [+1.7e-5, +2.9e-5] | +0.011 [+0.008, +0.014] | 0.33 / 0.33 / 0.33 | 0.67 |
| L11 | 2835 | −0.75 | 42/50 | +4.4e-5 | −4.4e-5 | −1.7e-4 [−2.2e-4, −1.3e-4] | −0.088 [−0.111, −0.067] | 0.24 / 0.24 / 0.24 | 0.29 |

- On the score set in advance, 14 of the 15 candidates do not move cells more than random features in the predicted direction (directional p 0.19–1.00 at every α). At α = 5 their effects are 0.0004% to 1.5% of the span in size.
- Most of their intervals exclude zero. That is not evidence of steering. The edit is deterministic, so it moves every cell a little, and random features move cells by similar amounts. The one exception in size is L9 F878 (size p 0.048; below).
- The response is close to linear in α − 1 for 5 of the 15 candidates (L9 F2903, the three L11 candidates and L0 F1976): Δs(α = 0) / Δs(α = 2) is −0.91 to −1.01 and Δs(α = 5) / Δs(α = 2) is 3.63 to 4.10 (L9 F2903: −1.00 and 4.02). For the other ten (at L0, L3, L6 and L9) the ratios are irregular (α = 5 to α = 2 from 0.62 to 36.4). The three α values are still close to one test repeated, because they use the same cells and features and, within each layer, rank the 23 features in almost the same order.
- The null features were matched on activity in selection HSCs. In the steered HSCs, the candidates' edits were 0.03 to 4.0 times the null mean (L9 F2903: 2.8 times; L9 F3003: 0.03 times). The per-unit-of-edit comparison below controls for this.

**The one hit: L9 feature 2903.**

- *What it is* (post hoc). Active at ≥ 1 gene position in 97% of selection HSCs, but at only 11% of gene positions in HSCs, 33% in CMPs, 99% in EryP and 97% in erythrocytes (32 selection cells, 8 per stage, all from donor TSP14; these shares are not a general figure for HSCs: in the steered HSCs it is active in 42 of 50 cells, at a mean of 382 gene positions per cell over all 50, of at most 2,046). It fires on 7,657 different genes, each a tiny share of its total activity (largest: ANK1, TFRC, BLVRB; 0.08–0.09% each). So it marks the cell's erythroid state, not a gene. ρ = +0.84 (+0.87, +0.80 and +0.84 in the three selection donors).
- *Effect.* α = 5: Δs = +0.0069 [+0.0038, +0.0103], 3.5% [1.9%, 5.2%] of the span. α = 2: +0.86% [+0.47%, +1.30%]. α = 0 (removed): −0.86% [−1.31%, −0.47%]. Gap share at α = 5: 2.6% [1.4%, 3.9%] of the line from g_early to g_late. Both read-outs agree in sign at every α.
- *Against the null.* At every α it beats all 20 random L9 features, in the predicted direction and in size (p = 0.048, the floor). z against the L9 null = +6.4 at α = 5 (this assumes a roughly normal null; the 20 null values span −0.0029 to +0.0019). Per unit of edit size (post hoc) it also beats all 20 (p = 0.048), so its larger edit does not explain the result.
- *Correction for 15 candidates.* p = 0.048 alone does not survive: about 0.7 of 15 candidates would reach the floor by chance (15 / 21). Post hoc, against all 100 null features pooled over the five layers, none is as large (p = 1/101 = 0.0099; × 15 candidates = 0.15), and its effect is 2.4 times the largest of them. So it is a clear outlier, but the formal p is limited by the size of the null.
- *Cells.* It moves 42 of 50 HSCs in the predicted direction. These are exactly the 42 cells where it is active; in the other 8 the edit is zero. Donor split (post hoc): TSP2 +0.0066 [+0.0035, +0.0102] (n = 47); TSP27 +0.0108 [+0.0039, +0.0175] (n = 3, too few to count as a replication).
- *Size in context.* The size-matched control moves cells 3.9% of the span with an edit of 1.43 per gene position, less than half of feature 2903's edit. Feature 2903 moves them 3.5% with an edit of 3.44. Per unit of edit (post hoc) it does 37% of what the control direction does (Δs 0.0020 against 0.0054 per unit). The full erythrocyte-minus-HSC difference at L9 has size 24.9 per gene position.
- *Genes.* Its most-raised genes at α = 5 are AHSP, ALAS2, HBM, GYPA, HBG1, SLC4A1, HBG2 and GYPB. 9 of its top 10 are on the 14-gene erythroid list, against 0.2 on average for the 20 random L9 features (post hoc). All six of AHSP, ALAS2, HBM, GYPA, HBG1 and SLC4A1 are among the 10 genes most raised by the size-matched control at L9. Five of them (not HBG1) are among the 15 genes highest in g_late − g_early. No small set of genes defines the score, which is a cosine over all 20,275 logits. But these are the genes that any push along this axis raises, so they restate the score rather than add biology.

**L9 feature 878: a large effect whose direction depends on the read-out.**

- An HSC-high feature (ρ = −0.73; genes with the largest share of its activity: RUNX1, ETV6, FKBP5, RNF220, CDK6). Active in 50 of 50 HSCs, at about 500 gene positions per cell.
- On the score set in advance, amplifying it moves HSCs *toward* the erythrocyte signature (α = 5: +0.0029 [+0.0019, +0.0039], +1.5% of the span). That is the wrong direction (directional p = 1.00). In size it beats all 20 random L9 features (size p = 0.048), is 4.5 times their mean |Δs| (1.47% against 0.33% of the span), and equals the largest of all 100 null effects (ratio 1.00).
- On the gap share (post hoc), it moves cells toward HSC, as predicted (α = 5: −3.3% [−3.8%, −2.9%]; beats all 20 null features at every α). The cosine between its logit change and g_late − g_early is −0.16.
- So the edit moves the logits back along the HSC-to-erythrocyte line, but it also changes the logit vector in other ways that raise the cosine score. Its direction depends on the read-out. It cannot be called a maturity switch either way.

**Floor hits on the post-hoc read-outs.** On the score set in advance, only L9 F2903 reaches the floor. On the post-hoc read-outs (candidates_table.csv), more candidates do:

- Gap share: L9 F2903 and L9 F878 at every α; L6 F3965 at α = 2 and 5 (its gap share is tiny, +0.0002 at α = 5).
- Gap share per unit of edit, α = 5: L9 F2903, L6 F646 and L11 F2835.
- Δs per unit of edit, α = 5: L9 F2903 only.
- With 15 candidates, 3 strengths and 2 read-outs, a few results at the floor are expected by chance. "14 of 15" refers to the score set in advance.

**Layer-level and pooled tests.**

| Layer | Candidates vs all 1,771 sets of 3: directional p (α = 0 / 2 / 5) | Size p (α = 0 / 2 / 5) | Spearman(ρ_f, Δs_f) over 23 features (α = 0 / 2 / 5) |
|---|---|---|---|
| L0 | 0.64 / 0.55 / 0.47 | 0.90 / 0.87 / 0.62 | −0.14 / +0.31 / +0.33 |
| L3 | 0.27 / 0.33 / 0.45 | 0.86 / 0.96 / 0.65 | −0.50 / +0.40 / +0.34 |
| L6 | 0.27 / 0.24 / 0.16 | 0.58 / 0.68 / 0.74 | +0.15 / −0.15 / −0.15 |
| L9 | 0.09 / 0.12 / 0.14 | 0.08 / 0.05 / 0.02 | −0.14 / +0.10 / +0.08 |
| L11 | 0.47 / 0.46 / 0.46 | 0.61 / 0.61 / 0.61 | −0.29 / +0.29 / +0.29 |

- At no layer does the candidate set push in the predicted direction more than random sets of 3. At L9 the candidates are larger than random sets (size p 0.02 at α = 5), but they point in different directions (2903 as predicted, 878 against).
- Pooled over layers, the test named in `design.json`: the candidates' summed directional rank is no higher than chance (p 0.29 / 0.36 / 0.24 at α = 0 / 2 / 5).
- Pooled over layers, mean within-layer Spearman(ρ_f, Δs_f) over 23 features per layer (predicted sign negative for α = 0, positive for α = 2 and 5): −0.18 / +0.19 / +0.18; one-sided permutation p 0.027 / 0.022 / 0.031; two-sided 0.055 / 0.045 / 0.063. The Spearman over all 115 features pooled is not reported.
- Most of the 115 features in this test are null features with |ρ| up to 0.66. So the trend says "features that track maturity push a little more along the axis". It does not say that the chosen candidates are switches. It is weak, and the three α values are close to one test repeated (same cells and features; within each layer they rank the features in almost the same order).

**Genes of the other candidates.** The candidates that do not beat the null raise mixed, unrelated genes (for example L3 F1133: RYR2, GRM5, PLCB1, RGS7, DOCK4; L11 F2767: CNBP, DARS1, MAGED1). L0 F1883 has 5 erythroid genes in its top 10 (HBA1, HBG2, AHSP, HBZ, HBA2), but its Δs does not beat the null (directional p 0.29 at α = 5). So erythroid genes in a list are not enough to call a feature a switch.

### Verification

No separate verification by a second agent is recorded for this analysis. The key numbers were computed a second way with separate code (`scripts/v3_steering_crosscheck.py`, output `outputs/v3_steering/crosscheck.json`; all pass), in addition to the forward-pass checks above (`scripts/v3_steering_verify.py`, `outputs/v3_steering/verify/verify.json`):

| Number | First way | Second way | Agreement |
|---|---|---|---|
| Δs of all 2,750 candidate and control rows | summary code | recomputed from the stored float32 logits | largest difference 4.4 × 10⁻¹⁶ |
| Δs of all 17,750 rows | summary code | recomputed from the stored dot products with the signatures | largest difference 1.4 × 10⁻¹⁴ |
| Bootstrap intervals (45 candidate × α rows) | own resampling loop | `scipy.stats.bootstrap` (percentile) | means identical; interval ends within 2.3% of the interval width |
| Candidate choice | selection stage | separate code from the stored clean files | same alive counts and the same 3 candidates at every layer; ρ equal within 10⁻¹⁵ |
| Held-out AUROC (EryP vs HSC) | own code | scikit-learn | 0.99133 both |
| 45 empirical p values | summary code | recomputed | identical (difference ≤ 1.1 × 10⁻¹⁶) |
| Δs, speed-up path | partial forward from block l | full forward with `hooks_v2.run_with_edits` (75 rows) | largest difference 2.2 × 10⁻⁹ |
| Δs, device | MPS | CPU (cell 332, L9 F2903, α = 5) | difference 7.0 × 10⁻¹⁰ |
| 42 cells moved by L9 F2903 | summary | per-cell files (`steer/cell_*.npz`): cells with Δs > 0 at α = 5 and cells with the feature active | 42 and 42, the same cells; mean 0.006875 |

### Limits

- **The SAEs fit these cells poorly** (FVE 0.34–0.59 on the steered HSCs, against 0.70–0.91 on held-out K562 tokens at the same layers). No SAE was trained on bone-marrow cells. So the negative result is "none of the 15 tested features of these K562-trained SAEs acts as a maturity switch beyond what a feature that marks erythroid cells would do", not "no feature is a switch" and not "the model has no such direction". Of the 2,481 to 4,912 alive features per layer, only the 3 candidates were tested as switches; 20 more per layer were steered only as the null. The positive control shows that a direction that moves cells does exist.
- **One donor in practice.** 47 of the 50 steered HSCs come from TSP2. The held-out axis check uses TSP2 and TSP27, which have no erythrocytes, so EryP stands in for the late stage there.
- **20 null features per layer** cap every per-feature p at 0.048. With 15 candidates, one feature at the floor is expected by chance. The case for L9 F2903 rests on its size against all 100 null features (2.4 times the largest), which is a post-hoc comparison.
- **Not tested:** other lineages (myeloid, lymphoid); steering erythroid cells back toward HSC; combinations of features; α above 5; layers other than 0, 3, 6, 9 and 11; features other than the 3 candidates per layer; read-outs other than the position-mean logits (for example hidden-state probes); MaxToki-1B. The deployed "pseudotime" axis was not used, because it separates two lineages rather than ordering maturity (main text).
- **The read-out is the model's position-mean next-gene logits.** It stands in for cell state; it is not a measured state. Stages are annotation labels, not time.
- **Length.** Selection erythrocytes have shorter sequences (median 1,796 tokens; 2,048 in the other groups), and within HSCs and CMPs the score correlates with length (−0.42, −0.41). So the late signature carries some length information. This cannot cause a steering effect, because edits do not change tokens, and the steered HSCs nearly all have 2,048 tokens.
- Other jobs shared the machine, so wall times are not clean timings (clean stage 530 s for 442 cells; steering 5,943 s for 50 cells in 16 chunks; verify stage 102 s). Free memory never fell below 5.8 GB.

### Files

- Scripts (`projects/maxtoki/runs/exhaustive-mapping-217M/scripts/`): `v3_steering.py` (stages design, prep, clean, select, steer), `v3_steering_verify.py`, `v3_steering_summarize.py`, `v3_steering_crosscheck.py`, `v3_steering_describe.py`, `v3_steering_extra.py`. Shared code: `projects/maxtoki/setup/inputs_v3.py`, `projects/maxtoki/setup/hooks_v2.py`. SAEs: `projects/maxtoki/runs/sae-atlas-217M/outputs/v3_sae/`.
- Outputs (`projects/maxtoki/runs/exhaustive-mapping-217M/outputs/v3_steering/`):
  - design and inputs: `design.json` (rules, written before any forward pass), `prep.json`, `cells.npz` (rows, donors, stages, tokens, counts sha256), `clean/cell_*.npz` (442 files);
  - selection: `selection.json` (axis, SAE fit, candidates, null features, norms of the control vectors), `signatures.npz` (g_early, g_late and the control vectors);
  - steering: `steer/cell_*.npz` (50 files; per row Δs, gap share, edit size, active positions, logit changes);
  - summaries: `summary.json` (all main numbers), `candidates_table.csv`, `null_table.csv`, `controls_table.csv`;
  - checks and post-hoc analyses: `verify/verify.json`, `crosscheck.json`, `describe.json`, `extra.json`;
  - provenance: `run_config.json` (device, versions, seeds, cell IDs, feature IDs, input sha256, code sha256, per-chunk wall time, free memory and encoding-check counts). `comparison.json` holds the deployed per-layer values for reference.
