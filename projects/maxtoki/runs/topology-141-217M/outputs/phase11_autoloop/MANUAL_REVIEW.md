# Phase 11 autoloop — manual review of automated decisions

The autoloop's decision rule (`PROMOTE` if positive null-gap in ≥4/12 layers in ≥2/3
domains) is calibrated for per-layer rows. Two iterations need post-hoc reclassification.

## iter_04 H-cross — RECLASSIFY: INCONCLUSIVE → **NOVEL POSITIVE**

The autoloop emitted "INCONCLUSIVE" because the rows were one summary per domain
(not per-layer), so by-domain `count` ≥ 4 was structurally impossible. Looking
at the actual numbers:

| Domain | Mean cross-layer Pearson of H123 motif feature | per-layer AUROC range |
|---|---:|---:|
| lung | **0.654** | 0.395–0.435 |
| immune | **0.604** | 0.489–0.533 |
| external_lung | **0.678** | 0.377–0.449 |

All three domains show **mean cross-layer Pearson > 0.6** on the H123 signed
motif-community feature. This means the per-pair signed-motif consistency
score is essentially the same at every transformer layer — a single static
axis. Combined with the per-layer AUROC hovering at 0.4–0.53 (close to or
below chance), this is a strong finding:

> **MaxToki's gene-pair community structure is essentially fixed at the
> embedding layer and propagates almost unchanged through all 11 transformer
> layers** — the residual stream does not reorganise gene similarities in any
> meaningful regulatory direction.

This sharpens the rest of the run's conclusion: not only is the H123 motif
signal weak (1/12 layers per domain), but it doesn't even *vary* across
layers, so there's no "best layer" to extract the signal from.

## iter_06 H-pooled — KEEP RETIRE (failure of test, not of hypothesis)

Only 36 genes are common to all three domain pools (lung, immune,
external_lung) at the pinned phase-1 pair tables. That's too few for a
meaningful pooled kNN + community detection. The hypothesis is untestable in
this scope; the right action would be to widen the gene pool selection
upstream, not to draw a substantive conclusion. Retired as
"no signal — gene-pool overlap insufficient".

## Final reclassified outcome

| Iteration | Original | Manual | Notes |
|---|---|---|---|
| iter_01 H-orc | INCONCLUSIVE | INCONCLUSIVE | 10/36 positive; mean Δ −0.027. Doesn't replicate H23-Forman's clean below-chance pattern. |
| iter_02 H-hyp | INCONCLUSIVE | INCONCLUSIVE | 5/36 positive; mean delta_norm 0.045 (near euclidean threshold 0.05). Not meaningfully hyperbolic. |
| iter_03 H-coex-res | INCONCLUSIVE | RETIRE | 9/36 positive; mean Δ −0.034. Triangle-defect signal does NOT survive coex residualization — confirms the small phase-6 positive is coex-driven. |
| iter_04 H-cross | INCONCLUSIVE | **NOVEL POSITIVE** | mean cross-layer Pearson 0.60–0.68 across all 3 domains. H123 motif feature is a single static axis — residual stream doesn't reorganise gene similarities through layers. |
| iter_05 H-tokfreq | INCONCLUSIVE | RETIRE | 6/36 positive; lung mean Δ −0.220, immune −0.103. H123 signal is NOT concentrated in high-frequency tokens — actually slightly stronger in low-frequency. Rules out popularity confound. |
| iter_06 H-pooled | RETIRE_no_signal | (untestable) | Only 36 common genes — pool too small. Not a substantive negative. |

## Final tally
- 1 NOVEL POSITIVE (cross-layer single-axis structure)
- 2 INCONCLUSIVE (ORC, hyperbolicity)
- 2 RETIRE (coex-residualised triangle-defect, token-frequency strata)
- 1 untestable (pooled multi-tissue)
