# MaxToki attention-GRN pipeline — final extended summary

## Final verdict: **FAIL on all four runs**

Four independent runs on the attention-GRN pipeline:

| Run | Model | Dataset | Modality | N_ctrl | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|---|---|
| k562_217M | MaxToki-217M | Replogle K562 | CRISPRi | 2000 | FAIL | FAIL | FAIL | FAIL |
| rpe1      | MaxToki-217M | Replogle RPE1 | CRISPRi | 2000 | FAIL | FAIL | FAIL | — |
| adamson   | MaxToki-217M | Adamson K562  | CRISPRa | 2000 | FAIL | FAIL | PASS* | — |
| k562_1b   | MaxToki-1B   | Replogle K562 | CRISPRi | 200  | FAIL | FAIL | FAIL | — |

*Adamson C3 trivially passes only because baseline TRRUST AUROC is already 0.52 (near chance), so "fraction of above-chance signal retained after residualization" is vacuously satisfied. Not a meaningful pass.

The paper's core claim — "attention in single-cell foundation models captures co-expression structure, not regulatory signal" — **replicates in two independent CRISPRi contexts (K562, RPE1), one CRISPRa context (Adamson), and across a 5× model-size increment (217M → 1B)**.

## Key metrics side-by-side

| Run | attn primary AUROC (per-pert) | ΔAUROC gene+attn | TRRUST baseline | TRRUST residual |
|---|---|---|---|---|
| k562_217M | 0.5119 | +0.0007 | 0.4829 | 0.4866 |
| rpe1      | 0.6033 | −0.0003 | 0.6045 | 0.4922 |
| adamson   | 0.5081 | −0.0001 | 0.5195 | 0.5202 |
| k562_1b   | 0.5398 | +0.0020 | 0.5554 | 0.5434 |

**Gene-variance mean AUROC** (the dominant trivial baseline in every case):

| Run | Gene variance | Gene variance − attention | p(BH) |
|---|---|---|---|
| k562_217M | 0.596 | +0.084 | 4e-10 |
| rpe1      | 0.766 | +0.163 | 2e-52 |
| adamson   | 0.712 | +0.204 | 5e-11 |
| k562_1b   | 0.596 | +0.056 | 1e-4 |

In every run, **a 1-feature gene-level univariate classifier (gene variance) crushes the attention-derived GRN** by 6–20 AUROC points at BH-corrected Wilcoxon p ≤ 1e-4. No run's attention beats any of the three trivial baselines (variance, mean expression, 1-dropout). TF out-degree underperforms attention (sanity-check negative control passes everywhere).

## Specific findings from the extended runs

### Phase 0b — value-weighted edges (K562, 217M)

Tests the "signal hides in value/FFN pathway" hypothesis. VW cosine per-layer TRRUST AUROC peaks at L9 (0.588), L10 (0.587). At primary L8, VW per-pert AUROC = 0.513 vs raw attention 0.532 — **VW underperforms raw attention** (Δ=−0.019, p=0.012). Combined with the raw-attention null, this rules out both attention-pattern and value-pathway explanations.

Interesting architectural asymmetry: **raw attention peaks at shallow layers (L1: 0.553), VW peaks at late layers (L9–L10: ≈0.59).** No equivalent pattern reported for Geneformer in the source paper — likely specific to causal Llama decoders.

### CSSI — cell-state stratified interpretability (K562, 217M, K=10)

Paper's **constructive remedy**. KMeans clustering on HVG expression produced highly skewed clusters [814, 594, 310, 261, plus 6 near-empty (1–14 cells)] — homogeneous NT controls don't stratify well.

Results vs baseline (attention L8 = 0.512 per-pert, 0.483 TRRUST):
| Aggregation | per-pert AUROC | TRRUST AUROC | vs baseline |
|---|---|---|---|
| baseline attention | 0.5119 | 0.4829 | — |
| CSSI-max          | 0.5000 | 0.4452 | Δ=−0.012, p=4e-4 (worse) |
| CSSI-mean         | 0.5039 | 0.4308 | Δ=−0.008, p=0.02 (worse) |
| CSSI-SCENIC       | 0.5123 | 0.4786 | Δ=+0.0003, p=0.35 (tie) |
| oracle_cluster_max | — | 0.5017 | even best-case upper bound is at chance |

**CSSI fails to rescue attention signal on homogeneous NT controls** — as the paper predicted for scGPT/Geneformer. Two of three aggregations are significantly *worse* than baseline.

### Scale — 1B vs 217M on K562

1B has ~2.7× the raw attention per-pert AUROC improvement over 217M (0.540 vs 0.512). But 1B's gene-variance baseline is **identical** (0.596, data property). The gap shrinks to Δ=+0.056 (vs 217M's +0.084), suggesting 1B captures *slightly* more regulatory-relevant structure — but still not enough to flip the verdict.

Incremental ΔAUROC scales 2.8× (+0.002 at 1B vs +0.0007 at 217M), but remains well below the 0.005 significance threshold. **Scale helps marginally; it does not rescue the pipeline.**

## Layer profiles (per-pert AUROC across layers, attention L-mean)

### MaxToki-217M (K562)
```
L0 0.539  L1 0.553  L2 0.546  L3 0.511  L4 0.526  L5 0.529
L6 0.524  L7 0.519  L8 0.512  L9 0.503  L10 0.537
```
Shallow-layer peak (L1), NOT the late-layer "regulatory" pattern of Geneformer V2-316M.

### MaxToki-217M (RPE1)
```
L0 0.682  L1 0.707  L2 0.684  L3 0.605  L4 0.645  L5 0.602
L6 0.602  L7 0.557  L8 0.603  L9 0.531  L10 0.621
```
Same shape: L1 peak. **Depth-wise attention hierarchy is not the Geneformer pattern** — it is specific to bidirectional BERT-style encoders.

## Per-head causal ablation (K562 217M Phase 4)

Six conditions × 300 cells × TRRUST-AUROC at primary layer. All deltas tiny and ordered randomly:

| Condition | TRRUST AUROC (L8) | Δ vs baseline |
|---|---|---|
| baseline | 0.4399 | — |
| top5_trrust | 0.4347 | −0.005 |
| random5_A | 0.4544 | +0.015 |
| random5_B | 0.4344 | −0.005 |
| random5_C | 0.4578 | +0.018 |
| entropy_matched_5 | 0.4533 | +0.014 |

**Ablating top-5 TRRUST heads does nothing different from ablating random heads** — the same causal null the paper reports for scGPT and Geneformer. "Regulatory" heads are not a causal bottleneck.

## What changes with scale or context — and what doesn't

**Moves with context (CRISPRi vs CRISPRa, K562 vs RPE1):**
- Raw attention AUROC: 0.51 ↔ 0.60 (range)
- TRRUST baseline AUROC: 0.48 ↔ 0.60
- Gene-variance AUROC: 0.60 ↔ 0.77
- Whatever moves the "signal" also moves the "null" in lockstep.

**Does NOT move:**
- Curveball null z-score (attention vs degree-preserving random): −0.14 / +0.17 / +0.16 / −0.04 — all at null in every run.
- Incremental ΔAUROC (gene+attn vs gene-only): ≤ +0.002 in every run, never the 0.005 threshold.
- Residualized AUROC collapses to chance after OLS cross-fit in every run.

**The invariant finding:** attention information is a subset of the information in gene-level univariate features. No context, no scale, no post-hoc remedy rescues it.

## Compute budget consumed (cumulative)

| Stage | Runtime |
|---|---|
| Phase 0 K562 217M (N=2000) | 54 min |
| Phase 0b K562 VW edges (N=2000) | 17 min |
| Phase 4 K562 ablation (300×6) | 23 min |
| Phase 0 RPE1 217M (N=2000) | 67 min |
| Phase 0 Adamson 217M (N=2000) | 27 min |
| Phase 6 CSSI K=10 (N=2000) | 35 min |
| Phase 0 K562 1B (N=200) | 95 min |
| All downstream analyses (Phase 1, 2, 3, 12 × 4 runs) | ~20 min |
| **Total** | **~5.5 h compute** |

Hardware: MacBook Pro (Apple Silicon, 32 GB unified memory), MPS backend, float32.

## Artefacts in outputs/

- `phase0{,_rpe1,_adamson,_k562_1b}/` — attention tensors, Spearman baselines, gene features, HVG tables
- `phase0b/` — value-weighted edges (K562)
- `phase1-4{,_rpe1,_adamson,_k562_1b}/` — per-phase analysis CSVs and JSONs
- `phase6_cssi/` — CSSI per-cluster edges and aggregations
- `phase12_verdict{,_rpe1,_adamson,_k562_1b}.json` — structured verdict per run
- `run_report{,_rpe1,_adamson,_k562_1b}.md` — human-readable per-run reports
- `cross_dataset_report.md` — K562 vs RPE1 comparison
- `FINAL_SUMMARY.md` — this file

## Cross-reference — Stage 3 (exhaustive mapping + trajectory steering)

This pipeline arrived at a clear negative verdict: MaxToki's attention does not encode causal regulatory logic — it captures co-expression, and every attempt to extract TF→target edges from attention fails under degree-preserving or gene-variance nulls (curveball z ≈ 0, Phase-11 CRISPRi 54.6% ≈ chance).

Stage 3 (`summaries/exhaustive-mapping-217M-FINAL_SUMMARY.md`) provides the first **affirmative positive** mechanistic-interpretability finding on MaxToki:

- **Trajectory steering at the final residual (L11):** amplifying each of 3 top-switch-feature SAE activations in early-pseudotime Tabula Sapiens immune cells pushes cell-state logits toward the mature signature in ~78–82% of cells (α ∈ {2, 5}). Top-upregulated gene under steering is **FTL** (ferritin light chain — a terminal erythroid marker), which is biologically coherent with "push toward maturity."
- **Layer-dependent directionality:** L0 features uniformly push toward maturity (frac_mat = 0.98, Δs = +0.096 — the largest effect size across all layers), L3 and L9 also toward (frac_mat = 1.00), L11 toward (0.79), while L6 uniformly *counter-differentiates* (frac_mat = 0.00, Δs = −0.008). Unlike Geneformer's monotone L0→L17 progression in the paper, MaxToki has a mid-layer counter-differentiation basin.
- **Zero higher-order synergy** replicates at third-order combinatorial ablation: 0 of 2,980+ target features are superadditive in any of the tested triplets — so the absence of causal regulatory logic found here is not an artifact of missing-by-attention; no gate of any order recovers it.

The cell-state-control finding is **not** about TF→target regulation (this pipeline's null stands) — it is about coarse trajectory direction. SCFMs can encode a latent "progenitor→mature" axis amplifiable via activation addition, even when they cannot encode the fine-grained regulatory edges that would make them useful as in-silico perturbation platforms. The two findings are consistent: MaxToki represents cell *identity* along a differentiation axis, not the *regulatory mechanism* that moves a cell along that axis.
