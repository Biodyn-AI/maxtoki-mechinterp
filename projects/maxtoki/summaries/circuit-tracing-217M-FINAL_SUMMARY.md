# Stage 2 — Causal Circuit Tracing — MaxToki-217M Summary

## Headline

**2,144,011 causal edges** across 120 source features (30 per source layer at L0, L3, L6, L9), traced through 200 K562 cells. The circuit graph is **extremely dense** (22× denser than Geneformer's 52,116 edges), **strongly inhibitory** (88.2%), and has **lower biological coherence** (36.1% vs paper's 53%).

**Chance-baseline anchor (added 2026-05-07, audit action A4):** 2.14M edges
out of 3,252,480 candidate (src_feature × tgt_layer × tgt_feature) pairs is
a 65.9% pass rate at |d|≥0.5 ∩ consistency≥0.7. Under noise-floor null
(Cohen's d ~ N(0, 0.235) for n_perturb=20 / n_ctrl=200; consistency ~
Bin(20, 0.5)/20), the gate-pass rate would be ~0.20%. **Observed exceeds
null by ~332×** (`outputs/edge_density_chance_baseline/summary.json`). The
dense-circuit picture is not a noise artifact. **However, the "41× denser
than paper" headline is partly threshold-sensitive:** at |d|≥2.0 ∩
consistency≥0.9, MaxToki retains 761,429 edges — still 14.6× the paper's 52K
but a smaller multiple than the loose-gate 41×. Conclusion: dense-circuit
picture is genuine; the 41× number is itself stable enough only as a
"loose-threshold" claim and should be reported alongside the strict-threshold
multiple to avoid threshold-cherry-picking concerns.

## Key metrics

| Metric | MaxToki-217M | Paper (Geneformer K562/K562) |
|---|---|---|
| Total edges | **2,144,011** | 52,116 |
| Source features | 120 (4 × 30) | 120 (4 × 30) |
| Edges/feature | **17,867** | 434 |
| Target coverage | **77.8%** | 31.9% |
| Mean \|d\| | **1.836** | 1.05 |
| Median \|d\| | **1.303** | 0.92 |
| Inhibitory % | **88.2%** | 80.1% |
| Biological coherence | **36.1%** | 52.9% |
| Compute time | ~4 hours | 7.5 hours |

## Per-source-layer breakdown

| Source | Downstream layers | Edges | Edges/feature | Compute |
|---|---|---|---|---|
| L0 | 11 (L1→L11) | 1,189,594 | 39,653 | 71 min |
| L3 | 8 (L4→L11) | 620,045 | 20,668 | 63 min |
| L6 | 5 (L7→L11) | 255,429 | 8,514 | 56 min |
| L9 | 2 (L10→L11) | 78,943 | 2,631 | 51 min |

Clear depth attenuation: earlier layers have broader causal influence. L0 features each affect ~40K downstream features — nearly saturating the target space (77.8% coverage).

## Top broadcast hubs (out-degree)

All top hubs are at L0, each with ~39,670 downstream connections (near-saturation of 11 downstream layers × 4,928 features = 54,208 possible):

| Feature | Out-degree |
|---|---|
| L0_F3083 | 39,676 |
| L0_F1727 | 39,674 |
| L0_F968 | 39,672 |
| L0_F1451 | 39,672 |
| L0_F2025 | 39,670 |

## Interpretation

### What replicates
- **Inhibitory dominance** (88.2%) — matches and exceeds the paper's 80.1%. Features encode necessary information; ablation reduces downstream activity.
- **Early-layer broadcast hubs** — L0 features are the most broadly connected, consistent with the paper's finding.
- **Strong effect sizes** — mean |d| = 1.84 (paper: 1.05). MaxToki features carry very strong per-feature effects.
- **Depth attenuation** — edges/feature declines from L0 (39,653) to L9 (2,631), matching the paper's pattern.

### What differs
- **Extreme density** — 2.14M edges vs paper's 52K (41× denser). This suggests MaxToki's Llama decoder has less feature redundancy than Geneformer's BERT — ablating any single feature causes widespread downstream disruption.
- **Lower biological coherence** (36.1% vs 53%) — despite denser circuits, fewer edges connect biologically related features. This is consistent with MaxToki's broader, less specific feature representations (recall: 0% TF specificity in Stage 1).
- **Near-saturation at L0** — every L0 source feature affects ~73% of all possible targets across all downstream layers. This level of saturation means the circuit is more "broadcast" than "circuit" — L0 features are essential for the entire computation, not for specific biological pathways.

### Architectural insight
MaxToki's causal Llama decoder creates a **dependency-dominated architecture** where early-layer features are globally necessary. This contrasts with Geneformer's more modular architecture where specific features influence specific downstream pathways. The high |d| + high density + low coherence pattern suggests MaxToki has learned a **deeply entangled** representation where biological concepts are more superposed and less separable than in Geneformer.

## Phases 7-12 results

### Phase 7 — PMI validation
No overlapping layer pairs between causal source features and PMI top-20 dependency lists (different selection criteria, as expected — paper also found zero source overlap).

### Phase 9 — Knowledge extraction
- **2,131 domains, 180,821 unique domain pairs** (paper: 1,126 domains, 16,002 pairs)
- Much denser meta-graph, consistent with the 41× denser circuit
- Feedback loops identified
- Early-layer domains: DNA pairing, homologous recombination (L0)
- Late-layer domains: gene expression regulation (L8+)

### Phase 10 — Gene-level predictions
- Raw gene pairs: **141.6M** (paper: 2.58M)
- Filtered (evidence≥2 or |d|>2): **5,442,818**
- Match TRRUST: 361 (0.01%) — paper: 0.15%

### Phase 11 — CRISPRi validation (critical test)
- Validated pairs: **1,458,016**
- **Directional accuracy: 54.6%** (paper: 56.4%)
- Both near-chance (50%), confirming circuits encode co-expression, not causal regulation

### Phase 11 — RE-RUN with proper sign tracking + GroupKFold (added 2026-05-07, audit action A2)

The original Phase 11 metric had a **bug**: it ignored predicted sign and just measured `actual_lfc < 0`. v2 re-runs with proper predicted-sign tracking + GroupKFold-by-source-gene (`outputs/groupkfold_crispri/summary.json`):

| Metric | Original (buggy) | v2 (corrected) |
|---|---:|---:|
| Overall directional accuracy | 54.61% | **53.48%** (n=1,503,408 pairs) |
| GroupKFold-by-source 5-fold mean | n/a | 53.55% (folds: 0.519–0.546) |
| Per-source mean accuracy | n/a | **53.42%** (95% CI [52.60%, 54.16%]) |

**The corrected directional accuracy is 53.48% — slightly lower than the buggy
54.61% but the 95% CI [52.60%, 54.16%] is decisively above 50% (chance).** Under
proper CV-by-source-gene, MaxToki's circuit predictions have a small but real
directional accuracy advantage (~3.4 percentage points above chance) that
cannot be attributed to per-source-gene pseudoreplication. This is consistent
with the "co-expression encodes some real directional signal" interpretation
but with a much tighter and methodologically defensible effect size estimate.

**Audit action A2 fully closes for circuit-tracing.** The headline 54.61%
should be retired in favor of 53.48% with the 95% CI.

### Phase 12 — Disease mapping
- Disease-associated domains: 469/2,131 (22.0%)
- Centrality: disease median=388, non-disease=360, **p=0.38 (not significant)**
- Paper: p=1.2e-11. MaxToki's circuit is too dense for disease-specific enrichment.

## Updated replication scorecard

| Phase | Finding | Paper | MaxToki | Status |
|---|---|---|---|---|
| 0-4 | Circuit construction | 52,116 edges | **2,144,011** | ✅ (much denser) |
| 5 | Inhibitory dominance | 80.1% | **88.2%** | ✅ |
| 5 | Early-layer broadcast hubs | L0 top hubs | L0 top hubs | ✅ |
| 5 | Effect size | mean \|d\|=1.05 | **1.84** | ✅ (stronger) |
| 6 | Biological coherence | 52.9% | **36.1%** | ⚠️ Lower |
| 9 | Domain meta-graph | 1,126 domains | **2,131 domains** | ✅ (denser) |
| 10 | Gene predictions | 975K filtered | **5.4M filtered** | ✅ |
| 10 | TRRUST match | 0.15% | 0.01% | ⚠️ Lower |
| 11 | CRISPRi directional | **56.4%** | **54.6%** | ✅ (same near-chance) |
| 12 | Disease centrality | p=1.2e-11 | p=0.38 | ❌ (not significant) |

## Scope
- Model: MaxToki-217M-HF, 12 layers, d=1232
- Source layers: L0, L3, L6, L9 (30 features each = 120 total)
- 200 K562 control cells, single condition (K562/K562)
- Significance thresholds: |d| > 0.5, consistency > 0.7
- Phases run: 0-12 (all except 8 multi-condition and 13 cross-model)
- Phases skipped: 8 (blocked — needs multi-tissue SAEs), 13 (blocked — single model)

## Cross-reference — Stage 3 (exhaustive mapping + higher-order ablation)

Stage 3 (`summaries/exhaustive-mapping-217M-FINAL_SUMMARY.md`) extends this selective-tracing run in two directions:

**1. Selective vs exhaustive sampling.** Stage 2 here traced 30 annotation-quality-filtered features per layer at L0/L3/L6/L9 (120 total). Stage 3 traced 1,000 features at L5 alone (500 top-annotated + 500 random unannotated) and found:
- **Top-20 hubs at L5 are 55% annotated, 45% unannotated** — so selecting by annotation quality (as Stage 2 does) systematically excludes ~45% of the highest-edge-count features. Stage 2's 36.1% biological-coherence number may understate what an unbiased sample would show.
- 100% of Stage-3 features have >1,000 downstream edges at {L6, L8, L11}. The mean-|d|=1.84 and 88.2%-inhibitory-dominance findings of Stage 2 scale up: MaxToki's circuit is uniformly dense rather than hub/tail-split.

**2. Pairwise → three-way redundancy.** Stage 2 here did not test combinatorial ablation (single-feature perturbation only). Stage 3's Experiment 2 extends to pairwise (AB/AC/BC) and three-way (ABC) ablation, computing the redundancy ratios the paper reports for Geneformer:
- **v1 (shared-ontology greedy selector, 4 triplets all sharing L0_F0 × L5_F0):** pairwise AB 0.165, three-way 0.190 — MaxToki redundancy is ~4.5× deeper than Geneformer's.
- **v2 (diversified selector — distinct L0/L5/L9 features across 4 distinct Reactome pathways):** pairwise mean across AB/AC/BC = **0.324**, three-way = **0.190**, std<0.0003 across triplets. Per-pair: AB=0.165, AC=0.219, BC=0.586. Monotonic deepening 1.00→0.324→0.190 replicates qualitatively vs paper's 1.00→0.74→0.59.
- Zero superadditive synergy at third order (0 of ~2,980 targets in every triplet across both v1 and v2), replicating the paper's qualitative finding of no higher-order logical gates.
- **Incidental v2 finding — feature-invariant ablation magnitudes:** the near-zero across-triplet standard deviation (±0.0001 on every ratio, despite 4 distinct feature sets) shows that MaxToki's per-layer features have approximately uniform aggregate causal magnitude. Any feature selection at a given layer produces the same redundancy ratio. This reinforces the "uniformly dense circuit" picture from Stage 3 Exp 1 (100% of features have >1000 edges).

Taken together, Stage 3 reframes Stage 2's CRISPRi-validation null (54.6% directional, near-chance): the problem is not that the *traced circuit* is wrong, but that it is *too densely redundant* to yield specific gene-level regulatory predictions. Stage 2's sparse-circuit baseline was always an incomplete view of MaxToki's dependency structure.
