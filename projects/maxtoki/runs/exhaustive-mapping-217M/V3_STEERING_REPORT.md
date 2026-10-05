# V3 steering report — maturity steering with v3 SAEs, correct inputs and a defined maturity axis (item V3-8)

Date: 2026-10-03. Model: MaxToki-217M (HF safetensors, float32, Apple MPS). torch 2.11.0, transformers 5.5.4.
Hooks: `setup/hooks_v2.py`, unchanged (edit on the input of block l, SAE code from the live tensor).
Inputs: `setup/inputs_v3.py` (Tabula Sapiens `raw/X` integer counts → counts / Geneformer gene median → ranked, no log).
SAEs: `runs/sae-atlas-217M/outputs/v3_sae` (retrained on correctly encoded K562 inputs, item V3-1).
Every number below comes from the files in section 9. Δs has no unit (it is a difference of cosine differences).
All 95% CIs are cell-level percentile bootstraps over the 50 steered cells (10,000 resamples, seed 20261003),
unless the text says otherwise.

## 0. Results first

1. **The design ran in full on correctly encoded inputs.** 50 hematopoietic stem cells (HSCs) × 5 layers ×
   (3 candidate features + 20 random features) × 3 strengths (α = 0, 2, 5), plus 2 positive controls per layer:
   17,750 steered forward passes (partial passes from the edited layer on; checked against full passes), plus
   442 clean passes. Every cell passed the encoding check right before its
   forward passes. The deployed tokenisation path (stored `X` ranked as if it were counts) fails the same check
   for 442 of 442 cells.
2. **The maturity axis is real and checked on held-out donors.** The axis is the erythroid branch of blood
   development, with stage labels from the cell-type annotation: HSC → common myeloid progenitor (CMP) →
   erythroid progenitor (EryP) → erythrocyte (bone marrow, 10x 3' v3 only). The score
   s = cos(mean logits, g_late) − cos(mean logits, g_early) separates EryP from HSC cells of donors never used to
   build it with AUROC **0.991** (AUROC = chance that a random EryP cell scores above a random HSC; 0.5 = chance).
   Spearman(s, stage) = **0.84** [0.78, 0.88]. The full distance from HSC to erythrocyte on this score
   ("the span") is S = **0.198**.
3. **14 of the 15 maturity-correlated features do not steer more than random features.** Their α = 5 effects are
   between 0.0004% and 1.5% of the span (in size). Their directional p values against 20 random features of matched activity
   are 0.19–1.00. Most of their CIs exclude zero, but that only means the edit always changes the output a little.
   Random features change it just as much.
4. **One feature passes: L9 feature 2903.** It is a dense "erythroid state" feature (active at 99% of gene
   positions in EryP and erythrocyte cells, 11% in HSCs). Amplifying it 5× in HSCs moves them toward the
   erythrocyte signature by Δs = **+0.0069** [+0.0038, +0.0103] = **3.5%** [1.9%, 5.2%] of the span.
   Removing it (α = 0) moves them back by 0.86% [0.47%, 1.31%]. It beats all 20 random L9 features at every α
   (p = 1/21 = 0.048, the smallest possible), also per unit of edit size. Its effect is 2.4 times the largest
   effect of all 100 random features across all layers (post hoc). It moves cells in 42 of 50 HSCs: exactly the
   42 where it is active.
5. **That one hit is small and expected.** The ideal direction (mean erythrocyte residual minus mean HSC residual),
   scaled to a similar edit size, moves cells 3.9% of the span with less than half the edit. Per unit of edit,
   feature 2903 does 37% of what that direction does. Its top raised genes (AHSP, ALAS2, HBM, GYPA, HBG1,
   SLC4A1) are the genes that any push along this axis raises. So the gene list restates the readout. It is not
   new biology.
6. **No layer-level or pooled test finds a set of maturity switches.** The 3 chosen candidates do not rank above
   random sets of 3 features in the predicted direction at any layer (p 0.09–0.64 over layers and α). Across layers, features more
   correlated with maturity push slightly more along the axis (mean within-layer Spearman +0.18 to +0.19,
   one-sided permutation p 0.022–0.031, two-sided 0.045–0.063). That is weak, and the three α values are close to
   one test repeated (the response is nearly linear in α − 1).
7. **All sanity checks pass** (section 3). Zero edits and edits of silent features change nothing (0.0 exactly).
   The fast partial forward pass equals the full forward pass (Δs difference ≤ 2.2e-9). CPU equals MPS (7e-10).
   Every key number was recomputed a second way (section 3.2).
8. **Versus the deployed result:** the published steering table (L0 +0.096, "5×/20×", "L6 inversion",
   KITLG/SOD3/C1QB/FTL genes) came from a hook that deleted a block (V2_HOOKS_REPORT.md). On a defined axis with
   correct inputs, fixed hooks and v3 SAEs, the candidates at L0, L3, L6 and L11 do nothing that random features
   do not also do. The only effect sits at L9, and its genes are erythroid genes, not the deployed lists.

---

## 1. Why a v3

- **Deployed Experiment 3** (`scripts/experiment3_rerun.py`): its hook replaced the output of block l with the
  clean input of block l. That deleted block l (L0–L9) or applied the final norm twice (L11). A zero edit gives
  the published table (V2_HOOKS_REPORT.md, section 5).
- **Its "pseudotime"** was PC1 of 200 random Tabula Sapiens immune cells of many cell types, built on
  double-logged expression (INPUT_ENCODING_AUDIT.md, section 6). That is not a maturity axis. The early and late
  groups were cell-type groups, and the "biologically coherent genes" were genes moved by the deleted block.
- **v2** (`scripts/v2_steering.py`) fixed the hooks but kept the PC1 axis, fed log1p(CP10k) as counts and used the
  deployed SAEs. It stopped after 34 of 115 work units, and no summary was computed.
- **v3** (this item) fixes all four: correct input order, retrained SAEs, a defined differentiation series with
  stage labels, and a random-feature null fixed before any forward pass.

## 2. Design (written to `design.json` at 08:53, before any forward pass)

**2.1 Differentiation series.** Tabula Sapiens immune, tissue = bone marrow, assay = 10x 3' v3.
Stage 0 hematopoietic stem cell, 1 common myeloid progenitor, 2 erythroid progenitor cell, 3 erythrocyte.
Stage numbers are ordinal labels from the annotation, not measured time. Left out: "hematopoietic precursor cell"
(its free annotations mix granulocytes, monocytes and other cells); Smart-seq2 cells (read counts, which
MaxToki-217M handles badly, V3_INPUTS_REPORT.md); 10x 5' v2 cells (so that one assay covers every stage).

**2.2 Cells** (numpy `default_rng(20261003)`).
- *Selection cells* (donors TSP14, TSP21, TSP25; up to 30 per stage and donor; 332 cells: 76 HSC, 90 CMP,
  76 EryP, 90 erythrocytes). Used for the signatures, the span and the feature choice.
- *Steered cells:* 50 HSCs from donors TSP2 and TSP27 (donor-disjoint from selection). In practice 47 come from
  TSP2 and 3 from TSP27 (TSP27 has only 9 HSCs).
- *Axis-check cells:* 30 CMP and 30 EryP cells from TSP2/TSP27, plus the 50 steered HSCs (clean passes only).
  TSP2/TSP27 have no erythrocytes, so the held-out check uses EryP as the late group.

**2.3 Readout.** m = logits averaged over all positions (as deployed). g_early = mean m of selection HSCs,
g_late = mean m of selection erythrocytes. s(m) = cos(m, g_late) − cos(m, g_early).
Δs = s(steered) − s(clean) for the same cell, computed by the same code path. Second readout:
gap share = (m_steered − m_clean)·(g_late − g_early) / |g_late − g_early|², the share of the straight line from
g_early to g_late that the edit moves. Span S = mean s(erythrocytes) − mean s(HSCs) on selection cells.

**2.4 Features.** v3 SAEs at L0, L3, L6, L9, L11 (site l = input of block l; L11 = input of `lm_head`).
Per cell, a_f = mean over gene positions (not `<bos>`/`<eos>`) of the SAE code z_f.
Alive = active at ≥ 1 gene position in ≥ 50% of selection HSCs.
ρ_f = mean over the 3 selection donors of the within-donor Spearman correlation between a_f and stage (0–3).
Candidates = the 3 alive features with the largest |ρ_f|.
Null = 20 other alive features per layer, drawn at random (`rng(20261003 + layer)`) from the 50 alive features
closest to each candidate in log10(mean a_f in selection HSCs) (7, 7, 6 per candidate). So nulls make edits of
about the same size in selection HSCs.

**2.5 Edit.** `hooks_v2.Steer(layer, f, α, mode="scale", positions=gene positions)`:
added vector = (α − 1) · z_f(x) · W_dec[:, f] at each gene position. α = 0 removes the feature, α = 1 adds nothing,
α = 2 doubles it, α = 5 multiplies it by 5. Predicted sign: amplify (α = 2, 5) → sign(Δs) = sign(ρ_f);
remove (α = 0) → −sign(ρ_f).

**2.6 Positive controls.** `hooks_v2.AddVector` at gene positions with v = mean gene-position residual of selection
erythrocytes minus selection HSCs at that site. Two sizes: full v ("full control"), and v scaled to the planned
α = 5 edit size of the candidates ("size-matched control", 4 × mean candidate activity in selection HSCs).

**2.7 Statistics.** Per candidate and α: directional p = (1 + number of nulls with an effect at least as large in
the predicted direction) / 21, so the smallest possible p is 0.048. Size p uses |Δs|. Layer test: the 3 candidates
against all 1,771 sets of 3 out of the 23 features. Pooled test: Spearman(ρ_f, Δs_f) over 23 features per layer,
averaged over layers, with labels permuted within layers (100,000 permutations), and the candidates' summed rank.

**2.8 Speed shortcut (checked).** An edit at site l does not change anything before block l. So each steered
pass reruns only blocks l..10 from the cached clean input of block l, and computes the position-mean logits as
`lm_head(mean of normed hidden)` (`lm_head` is linear with no bias). Every row is compared with a clean
reference computed by the same path. Section 3 checks this against full forward passes.

## 3. Checks

### 3.1 Sanity checks (all pass)

| Check | What it tests | Result | Pass |
|---|---|---|---|
| Encoding, prep | token order = counts / median order, counts are count-like, all 442 cells | 442 / 442 pass; 0 order violations (38 positions with exact ties, allowed) | yes |
| Encoding power | the deployed path (stored `X` as counts) must fail | fails for 442 / 442; top-200 overlap with the correct order 0.59 (0.38–0.99) | yes |
| Encoding, before each forward pass | counts re-read from `raw/X`, sha256 equal to prep, check re-run | 442 clean + 50 steer + verify cells, all pass | yes |
| Axis check (fixed rule) | held-out donors: AUROC(EryP vs HSC) ≥ 0.80 and Spearman(s, stage) > 0 | 0.991 and 0.844 | yes |
| V1 shortcut = full pass | 75 rows (cell 332–334; candidates, nulls, control) vs `hooks_v2.run_with_edits` | max Δs difference 2.2e-9; max logit difference 1.9e-6 | yes |
| V2 zero edit | α = 1 and `ZeroDelta`, shortcut and full | logits and Δs change 0.0 exactly (18 rows) | yes |
| V3 silent feature | a feature with z = 0 at every position | change 0.0 exactly (15 rows) | yes |
| V4 CPU vs MPS | cell 332, L9 F2903, α = 5 | Δs difference 7.0e-10 | yes |
| Silent rows in the main run | every (cell, feature, α) row where the feature is inactive | 807 rows; max |Δs| 0.0 and max logit change 0.0 | yes |
| Clean reference vs full pass | shortcut clean logits vs full clean logits, all cells and layers | max difference 1.9e-6 | yes |
| Edit applied when active | edit size > 0 in every active row | yes | yes |
| Memory | free memory checked before each cell; stop below 3 GB | lowest free memory 5.8 GB; no chunk stopped | yes |

### 3.2 Key numbers re-derived a second way (`crosscheck.json`, `v3_steering_crosscheck.py`)

- X1: Δs of all 2,750 candidate/control rows recomputed from the stored float32 logits: max difference 4.4e-16.
- X2: Δs of all 17,750 rows recomputed from stored dot products with the signatures: max difference 1.4e-14.
- X3: bootstrap CIs recomputed with `scipy.stats.bootstrap` (percentile): means identical; CI ends within 2.3% of
  the CI width.
- X4: candidate choice recomputed from the clean files with separate code: same alive counts and same 3 candidates
  at every layer; ρ equal to 1e-15.
- X5: axis AUROC with `sklearn`: 0.99133 (same).
- X6: all 45 empirical p values recomputed: identical.

## 4. Results

### 4.1 The axis

- The two signatures are close: cos(g_early, g_late) = 0.894. That is why s moves in small numbers.
- Selection cells, mean s by stage: HSC −0.104, CMP −0.009, EryP +0.063, erythrocyte +0.094 (SD 0.03–0.05).
  Span S = 0.198. Within-donor Spearman(s, stage): 0.86, 0.73, 0.89.
- Held-out donors (TSP2/TSP27): mean s HSC −0.077, CMP +0.001, EryP +0.079. AUROC CMP vs HSC 0.917,
  EryP vs HSC 0.991, EryP vs CMP 0.887. Spearman(s, stage) 0.844 [0.782, 0.885] (2,000 cell resamples).
- Length: erythrocytes have shorter sequences (median 1,796 tokens; the other stages 2,048). Within HSCs and CMPs,
  s correlates with length (−0.42, −0.41). The steered HSCs nearly all have 2,048 tokens, and edits do not change
  tokens, so length cannot cause a steering effect. But the late signature carries some length information.
- **The SAEs fit these cells much worse than K562.** Fraction of variance explained (FVE) on the steered HSCs
  (all positions): L0 0.46, L3 0.59, L6 0.37, L9 0.34, L11 0.43. On held-out K562 tokens the same SAEs give
  0.70, 0.91, 0.85, 0.86, 0.87 (V3_SAE_REPORT.md). The SAEs were trained on K562 cells only. So a large part of
  what these bone-marrow cells carry is not in any single feature.

### 4.2 Positive controls: the readout can move

| Layer | Null Δs α=5: mean (SD) | Null range α=5 | Null mean \|Δs\| as % of span | Candidates mean \|Δs\| α=5 (% of span) | Size-matched control Δs (% of span) [95% CI] | Candidates / control | Mean edit size per gene position: candidates / nulls / control | Full control Δs (% of span) [95% CI] | Full control gap share |
|---|---|---|---|---|---|---|---|---|---|
| L0 | +1.0e-04 (3.0e-04) | [−4.9e-04, +6.9e-04] | 0.12% | 1.9e-04 (0.095%) | 5.8e-03 (2.91%) [5.1e-03, 6.5e-03] | 0.033 | 0.0045 / 0.0046 / 0.0046 | 0.047 (23.7%) [0.043, 0.051] | 0.26 |
| L3 | +2.8e-04 (7.2e-04) | [−2.2e-04, +2.4e-03] | 0.18% | 8.5e-05 (0.043%) | 1.3e-03 (0.63%) [1.1e-03, 1.4e-03] | 0.068 | 0.094 / 0.105 / 0.100 | 0.145 (73.4%) [0.139, 0.151] | 0.71 |
| L6 | +9.9e-06 (2.6e-05) | [−2.5e-05, +1.1e-04] | 0.008% | 7.3e-06 (0.004%) | 1.9e-04 (0.09%) [1.7e-04, 2.0e-04] | 0.040 | 0.031 / 0.031 / 0.021 | 0.145 (73.1%) [0.139, 0.150] | 0.71 |
| L9 | −7.2e-07 (1.1e-03) | [−2.9e-03, +1.9e-03] | 0.33% | 3.3e-03 (1.65%) | 7.7e-03 (3.90%) [7.5e-03, 8.0e-03] | 0.42 | 2.80 / 1.24 / 1.43 | 0.177 (89.3%) [0.173, 0.181] | 0.85 |
| L11 | −6.9e-05 (2.5e-04) | [−7.1e-04, +3.6e-04] | 0.075% | 7.7e-05 (0.039%) | 2.7e-03 (1.36%) [2.6e-03, 2.8e-03] | 0.029 | 0.45 / 0.74 / 0.63 | 0.217 (109.5%) [0.205, 0.229] | 1.00 |

- Adding the full erythrocyte-minus-HSC residual difference moves HSCs 24% (L0) to 109% (L11) of the span, in
  50 of 50 cells at every layer. At L11 the gap share is 1.000 and the cosine between the logit change and
  g_late − g_early is 0.9999996. This must hold, because `lm_head` is linear. It shows the readout code is right.
- At the size of the candidate edits, the same direction moves cells 0.09–3.9% of the span. So the candidates
  (except L9) reach only 3–7% of what the best direction of the same size reaches.
- The control raises AHSP, HBM, ALAS2, SLC4A1, GYPA at every layer. These are the top genes of
  g_late − g_early itself (AHSP, ALAS2, HBM, LGALS3, SELENBP1, SLC4A1, MYL4, GYPA, KLF1).

### 4.3 Candidates against the random-feature null

Δs per candidate (mean over 50 HSCs). ρ = maturity correlation in selection cells. p values: against the 20
random features of the same layer (smallest possible 0.048).

| Layer | Feature | ρ | Cells active | Δs α=0 [95% CI] | Δs α=2 [95% CI] | Δs α=5 [95% CI] | α=5 as % of span [CI] | Directional p (α=0/2/5) | Size p (α=5) |
|---|---|---|---|---|---|---|---|---|---|
| L0 | 1976 | +0.73 | 50/50 | −4.7e-05 [−7.7e-05, −1.9e-05] | +5.1e-05 [+2.6e-05, +7.8e-05] | +1.9e-04 [+1.1e-04, +2.7e-04] | +0.094 [+0.055, +0.134] | 0.38 / 0.38 / 0.43 | 0.57 |
| L0 | 1883 | +0.72 | 50/50 | +1.6e-05 [−1.4e-05, +4.6e-05] | +1.6e-05 [−1.6e-05, +4.9e-05] | +2.7e-04 [+1.3e-04, +4.3e-04] | +0.138 [+0.066, +0.216] | 0.81 / 0.62 / 0.29 | 0.38 |
| L0 | 2449 | −0.68 | 50/50 | −1.1e-06 [−3.2e-05, +3.0e-05] | +2.8e-06 [−3.0e-05, +3.5e-05] | +1.0e-04 [−1.0e-05, +2.2e-04] | +0.052 [−0.005, +0.110] | 0.38 / 0.33 / 0.62 | 0.62 |
| L3 | 1133 | −0.81 | 50/50 | +1.3e-05 [+2.6e-06, +2.3e-05] | +6.1e-06 [−7.0e-06, +1.9e-05] | +1.5e-04 [+8.5e-05, +2.2e-04] | +0.077 [+0.043, +0.112] | 0.38 / 0.52 / 0.81 | 0.29 |
| L3 | 980 | −0.72 | 50/50 | +1.0e-05 [+7.6e-06, +1.2e-05] | −1.2e-05 [−1.5e-05, −9.4e-06] | −8.2e-05 [−9.8e-05, −6.5e-05] | −0.041 [−0.050, −0.033] | 0.38 / 0.33 / 0.19 | 0.57 |
| L3 | 1029 | −0.72 | 50/50 | +1.4e-05 [+1.1e-05, +1.6e-05] | −9.5e-06 [−1.2e-05, −7.4e-06] | −1.9e-05 [−2.7e-05, −1.0e-05] | −0.010 [−0.014, −0.005] | 0.38 / 0.38 / 0.38 | 0.90 |
| L6 | 3965 | +0.77 | 50/50 | −3.4e-06 [−4.3e-06, −2.7e-06] | +2.9e-06 [+2.1e-06, +3.9e-06] | +1.4e-05 [+9.9e-06, +2.0e-05] | +0.007 [+0.005, +0.010] | 0.33 / 0.38 / 0.38 | 0.43 |
| L6 | 1976 | +0.70 | 41/50 | +2.0e-06 [+1.3e-06, +2.8e-06] | −1.3e-06 [−2.1e-06, −4.2e-07] | −8.3e-07 [−5.0e-06, +5.2e-06] | −0.000 [−0.003, +0.003] | 0.81 / 0.81 / 0.76 | 1.00 |
| L6 | 646 | +0.69 | 36/50 | −1.2e-06 [−1.9e-06, −5.9e-07] | +1.4e-06 [+6.5e-07, +2.2e-06] | +7.0e-06 [+4.1e-06, +1.1e-05] | +0.004 [+0.002, +0.005] | 0.43 / 0.43 / 0.43 | 0.57 |
| **L9** | **2903** | **+0.84** | **42/50** | **−1.7e-03 [−2.6e-03, −9.2e-04]** | **+1.7e-03 [+9.3e-04, +2.6e-03]** | **+6.9e-03 [+3.8e-03, +1.0e-02]** | **+3.47 [+1.94, +5.18]** | **0.05 / 0.05 / 0.05** | **0.05** |
| L9 | 878 | −0.73 | 50/50 | −9.4e-05 [−3.3e-04, +1.8e-04] | +4.0e-04 [+1.3e-04, +6.4e-04] | +2.9e-03 [+1.9e-03, +3.9e-03] | +1.47 [+0.95, +1.95] | 0.81 / 0.95 / 1.00 | 0.05 |
| L9 | 3003 | +0.70 | 42/50 | +1.9e-06 [+1.4e-06, +2.6e-06] | −1.6e-06 [−2.1e-06, −1.1e-06] | −4.8e-06 [−6.6e-06, −3.2e-06] | −0.002 [−0.003, −0.002] | 0.57 / 0.57 / 0.57 | 0.86 |
| L11 | 2767 | −0.75 | 50/50 | −8.0e-06 [−9.7e-06, −6.3e-06] | +8.1e-06 [+6.4e-06, +9.9e-06] | +3.3e-05 [+2.6e-05, +4.1e-05] | +0.017 [+0.013, +0.021] | 0.76 / 0.76 / 0.76 | 0.57 |
| L11 | 4372 | +0.75 | 39/50 | −5.5e-06 [−7.0e-06, −4.1e-06] | +5.5e-06 [+4.2e-06, +7.1e-06] | +2.2e-05 [+1.7e-05, +2.9e-05] | +0.011 [+0.008, +0.014] | 0.33 / 0.33 / 0.33 | 0.67 |
| L11 | 2835 | −0.75 | 42/50 | +4.4e-05 [+3.4e-05, +5.6e-05] | −4.4e-05 [−5.6e-05, −3.3e-05] | −1.7e-04 [−2.2e-04, −1.3e-04] | −0.088 [−0.111, −0.067] | 0.24 / 0.24 / 0.24 | 0.29 |

What this shows:
- Most CIs exclude zero. That is not evidence of steering. The edit is deterministic, so it shifts every cell a
  little, and random features shift cells by the same amounts (null column in 4.2).
- At L0–L6 and L11 the effects are 0.0002–0.14% of the span. The deployed table claimed effects in the tens of
  percent of its own axis.
- The response is close to linear in α − 1 for most features: Δs(α = 0) ≈ −Δs(α = 2) and Δs(α = 5) ≈ 4 × Δs(α = 2)
  (L9 F2903: −1.00 and 4.02; L11: −0.98 to −1.01 and 3.96–4.10). So the three α values are not three independent
  tests. At L0 the ratios are irregular (3.6, 17, 36) because the effects there are tiny.
- The 20 null features at each layer were matched on activity in selection HSCs. In the steered HSCs the
  candidates' actual edits were 0.03–4.0 times the null mean (L9 F2903: 2.8 times; L9 F3003: 0.03 times). Section 4.4 controls for this.

### 4.4 The hit: L9 feature 2903

- **What it is.** Active at ≥ 1 gene position in 97% of selection HSCs, but at only 11% of HSC gene positions,
  33% of CMP, 99% of EryP and 97% of erythrocyte gene positions. It fires on 7,657 different genes, each a tiny
  share of its total (largest: ANK1, TFRC, BLVRB, SPTA1, PRDX2, CA2, FECH, 0.07–0.09% each). So it marks the
  cell's erythroid state, not a gene. ρ = +0.84 (+0.87, +0.80, +0.84 in the three selection donors).
- **Effect.** α = 5: Δs +0.0069 [+0.0038, +0.0103], 3.5% [1.9%, 5.2%] of the span. α = 2: 0.86% [0.47%, 1.30%].
  α = 0 (removed): −0.86% [−1.31%, −0.47%]. Gap share at α = 5: 2.6% [1.4%, 3.9%] of the line from g_early to
  g_late. Both readouts agree in sign at every α.
- **Against the null.** At every α it beats all 20 random L9 features, in the predicted direction and in size
  (p = 0.048, the floor). z against the L9 null = +6.4 at α = 5 (assumes the null is roughly normal; the 20 null
  values span −0.0029 to +0.0019). Per unit of edit size it also beats all 20 (p = 0.048), so the larger edit does
  not explain it.
- **Correction for 15 candidates.** On its own, p = 0.048 does not survive: about 0.7 of 15 candidates would hit
  the floor by chance. Post hoc, against all 100 null features pooled over the 5 layers, none is as large
  (p = 1/101 = 0.0099; × 15 candidates = 0.15), and its effect is 2.4 times the largest of them. So it is a clear
  outlier, but the formal p is limited by the 20-feature null.
- **Cells.** It moves 42 of 50 HSCs in the predicted direction. Those are exactly the 42 cells where it is active;
  in the other 8 the edit is zero. Donor split (post hoc): TSP2 +0.0066 [+0.0035, +0.0102] (n = 47);
  TSP27 +0.0108 [+0.0039, +0.0175] (n = 3, too few to count as a replication).
- **Size in context.** The size-matched control direction moves cells 3.9% of the span with an edit of 1.43 per
  gene position. Feature 2903 moves them 3.5% with 3.44. Per unit of edit, it does 37% of what the best direction
  does (Δs 0.0020 vs 0.0054 per unit). The full erythrocyte-minus-HSC difference at L9 has size 24.9 per gene
  position.
- **Genes.** At α = 5 its top raised genes are AHSP, ALAS2, HBM, GYPA, HBG1, SLC4A1, HBG2, GYPB: 9 of its top 10
  are on a 14-gene erythroid list, against 0.2 on average for the 20 random L9 features (post hoc). The positive
  control raises the same genes. They are what "toward erythrocyte" means on this readout.

### 4.5 L9 feature 878: a large effect whose direction depends on the readout

- An HSC-high feature (ρ = −0.73; largest genes RUNX1, ETV6, FKBP5, CDK6, MEIS1, FLT3). Active in 50 of 50 HSCs
  at about 500 gene positions.
- Pre-registered readout: amplifying it moves HSCs **toward** the erythrocyte signature (α = 5: +0.0029
  [+0.0019, +0.0039], +1.5% of the span). That is the wrong direction (directional p = 1.00), although its size
  is as large as the largest of all 100 null effects.
- Linear readouts: the gap share moves toward HSC, as predicted (α = 5: −3.3% [−3.8%, −2.9%]; beats all 20 nulls).
  The cosine between its logit change and g_late − g_early is −0.16.
- So the edit pushes the logits back along the HSC-to-erythrocyte line, but it also changes the logit vector in
  other ways that raise the cosine score. It cannot be called a maturity switch. The gap-share comparison was
  added after the first look at the data.

### 4.6 Layer-level and pooled tests

| Layer | Candidates vs all 1,771 sets of 3: directional p (α = 0 / 2 / 5) | Size p (α = 0 / 2 / 5) | Spearman(ρ_f, Δs_f) over 23 features (α = 0 / 2 / 5) |
|---|---|---|---|
| L0 | 0.64 / 0.55 / 0.47 | 0.90 / 0.87 / 0.62 | −0.14 / +0.31 / +0.33 |
| L3 | 0.27 / 0.33 / 0.45 | 0.86 / 0.96 / 0.65 | −0.50 / +0.40 / +0.34 |
| L6 | 0.27 / 0.24 / 0.16 | 0.58 / 0.68 / 0.74 | +0.15 / −0.15 / −0.15 |
| L9 | 0.09 / 0.12 / 0.14 | 0.08 / 0.05 / 0.02 | −0.14 / +0.10 / +0.08 |
| L11 | 0.47 / 0.46 / 0.46 | 0.61 / 0.61 / 0.61 | −0.29 / +0.29 / +0.29 |

- No layer's candidate set pushes in the predicted direction more than random sets of 3. At L9 the candidates are
  larger than random sets (size p 0.02 at α = 5), but they point in different directions (2903 right, 878 wrong).
- Pooled over layers (predicted sign: negative for α = 0, positive for α = 2, 5): mean within-layer
  Spearman(ρ_f, Δs_f) = −0.18 / +0.19 / +0.18; one-sided p 0.027 / 0.022 / 0.031; two-sided 0.055 / 0.045 / 0.063.
  Candidate rank sum vs random: p 0.29 / 0.36 / 0.24. Most of the 115 features in this test are null features with
  |ρ| up to 0.66, so this says "features that track maturity push a little more along the axis", not "the chosen
  candidates are switches".

### 4.7 Gene lists (the deployed "biologically coherent genes")

- The deployed lists (L0 KITLG, SOD3, APOE; L3 C1QTNF3-AMACR; L6 C1QB, HBE1, CPA3; L9 PAQR3; L11 FTL, MT-CO3)
  were produced by the block-deleting hook with a zero edit (V2_HOOKS_REPORT.md). Only two of them appear in any
  v3 candidate's top 10 raised genes: MT-CO1 (L0 F2449, rank 7) and MT-CO3 (L11 F2835, rank 9), both mitochondrial
  genes, and neither feature beats the null.
- In v3, the candidates that do not beat the null raise unrelated mixed genes (for example L3 F1133: RYR2, GRM5,
  PLCB1, RGS7, DOCK4; L11 F2767: CNBP, DARS1, MAGED1). There is no story to tell from them.
- L0 F1883 has 5 erythroid genes in its top 10 (HBA1, HBG2, AHSP, HBZ, HBA2), but its Δs does
  not beat the null (p 0.29). So erythroid genes in a list are not enough to call a feature a switch.

## 5. What changed versus the deployed result and v2

| | Deployed Experiment 3 | v2 (unfinished) | v3 (this report) |
|---|---|---|---|
| Hook | deletes block l (L0–L9); double norm at L11 | hooks_v2 (fixed) | hooks_v2 (fixed) |
| Input order | log1p(CP10k) ranked as counts | same (wrong) | raw counts / median (checked per cell) |
| SAEs | deployed (trained on mis-encoded K562) | deployed | v3 (retrained, correct inputs) |
| Axis | PC1 of 200 mixed immune cells (double-log) | same PC1 | HSC → CMP → EryP → erythrocyte labels; checked on held-out donors |
| Features | 3 per layer from the deployed switch list | same 3 + nulls | 3 per layer by within-donor ρ on selection cells; 20 matched nulls |
| Null | none | planned, 19 of 100 null units run | 20 per layer, all run |
| L0 effect, α = 5 | +0.096 (published) = zero-edit value +0.096 | −0.0009, +0.0003, −0.0008 (old axis) | +0.0002, +0.0003, +0.0001 (0.05–0.14% of span), inside the null |
| L9 effect, α = 5 | +0.014 (= zero-edit value) | −0.0002, −0.00005, −0.00003 | +0.0069 (F2903, beats null), +0.0029 (F878, wrong direction), −0.000005 |
| Genes | KITLG/SOD3, C1QB, PAQR3, FTL (hook artefact) | not computed | erythroid genes for the one hit only, same as the positive control |

- The deployed values and the v3 values are on different axes, so their ratio is not meaningful. The comparison
  that matters is already made in V2_HOOKS_REPORT.md: the deployed values are the hook, not the features.
- v2 measured effects of about 5e-6 to 3e-3 on the old PC1 axis, with null features of the same size, and stopped
  before any summary. v3 completes that design on correct inputs and finds the same picture for 14 of 15
  candidates.
- The v3 feature IDs cannot be matched to the deployed IDs (L0 F40, L3 F4071, L6 F4138, L9 F1014, L11 F3924),
  because the SAEs were retrained.

## 6. Post-hoc additions (labelled; not in `design.json`)

Added after the first look at the results:
- the gap-share comparison with the null, and the per-unit-of-edit comparison (sections 4.3–4.5);
- `v3_steering_extra.py` → `extra.json`: donor split, comparison with all 100 nulls pooled, the cosine between
  each logit change and g_late − g_early, and the erythroid-gene count in top-10 lists;
- the gene descriptions of the candidates (`v3_steering_describe.py` → `describe.json`, 32 selection cells).

## 7. What I could not do, and limits

- **The SAEs fit these cells poorly** (FVE 0.34–0.59 on the steered HSCs vs 0.70–0.91 on K562). No bone-marrow SAE
  was trained. So the negative result is "no single feature of a K562-trained SAE is a maturity switch", not "the
  model has no such direction". The positive control shows that a direction that moves cells does exist.
- **One donor in practice.** 47 of the 50 steered HSCs come from TSP2. TSP27 has only 9 HSCs (3 drawn).
  The held-out axis check uses TSP2/TSP27, which have no erythrocytes, so EryP stands in for the late stage there.
- **20 nulls per layer** cap every per-feature p at 0.048. With 15 candidates, one feature at the floor is
  expected by chance. The hit's case rests on its size against all 100 nulls (2.4 times the largest), which is a
  post-hoc comparison.
- **One lineage and one direction.** I steered HSCs toward erythrocytes only. I did not steer erythroid cells
  back, did not test other lineages (myeloid, lymphoid), did not combine features, and did not use α above 5.
- **The readout is the model's position-mean next-gene logits.** It is a stand-in for cell state, not a measured
  state. Stages are annotation labels, not time.
- Length differs between stages (erythrocytes are shorter), so the late signature carries some length signal.
  It cannot create a steering effect, because edits do not change tokens.
- I did not rerun the deployed PC1 axis with v3 inputs; it is not a maturity axis, so there was nothing to gain.
- Other projects used the machine at the same time, so the wall times are not clean timings
  (clean stage 530 s for 442 cells; steer stage 5,943 s for 50 cells in 16 chunks; verify 102 s).
  Free memory never fell below 5.8 GB.

## 8. What this means for the paper

The deployed steering claims (L0 push toward maturity, 5×/20× ratios, L6 inversion, coherent gene programs)
do not survive. On a defined erythroid maturity axis, with correct inputs, fixed hooks and retrained SAEs,
14 of 15 maturity-correlated features move cells no more than random features. One dense erythroid-state feature
at L9 does move HSCs toward the erythrocyte signature, by 3.5% [1.9%, 5.2%] of the HSC-to-erythrocyte distance
at 5× amplification. That is a small, real causal effect, and it is the effect one would expect from a feature
that marks erythroid cells. It is weaker per unit of edit than the plain mean-difference direction.

## 9. Files

- Scripts (`runs/exhaustive-mapping-217M/scripts/`): `v3_steering.py` (stages design, prep, clean, select,
  steer), `v3_steering_verify.py`, `v3_steering_summarize.py`, `v3_steering_crosscheck.py`,
  `v3_steering_describe.py`, `v3_steering_extra.py`.
- Outputs (`runs/exhaustive-mapping-217M/outputs/v3_steering/`): `design.json` (rules, written first),
  `prep.json`, `cells.npz` (rows, donors, stages, tokens, counts sha256), `clean/cell_*.npz` (442),
  `selection.json` (axis, SAE fit, candidates, nulls), `signatures.npz`, `steer/cell_*.npz` (50),
  `verify/verify.json`, `summary.json`, `candidates_table.csv`, `null_table.csv`, `controls_table.csv`,
  `comparison.json`, `crosscheck.json`, `describe.json`, `extra.json`, `run_config.json` (device, versions,
  seeds, cell IDs, feature IDs, input sha256, code sha256, per-chunk wall time and encoding-check counts).
- This report: `runs/exhaustive-mapping-217M/V3_STEERING_REPORT.md`.

## Plain-words summary

The old steering result was produced by a bug that deleted a layer of the model, on inputs in the wrong gene
order, along a "maturity" axis that was really a mix of cell types. I redid it properly. I took one real
developmental path in bone marrow (stem cell → myeloid progenitor → red-cell progenitor → red blood cell), with
labels from the data. I built a "how red-cell-like does the model's output look" score from one set of donors,
and checked it on other donors (it separates the stages very well, AUROC 0.99). Then I took 50 stem cells from the
held-out donors and turned up or switched off 3 features per layer that track maturity, plus 20 random features
per layer for comparison. 14 of the 15 maturity features moved the cells no more than random features did, and
by tiny amounts (at most about 1.5% of the stem-cell-to-red-cell distance). One feature at layer 9, which is
simply "this cell looks like a red-cell precursor", moved stem cells 3.5% of the way toward red cells when turned
up 5 times. That beats every random feature, but it is small, it comes from one donor in practice, and a plain
"red cell minus stem cell" direction does better for the same size of push. The old gene lists do not come back.
