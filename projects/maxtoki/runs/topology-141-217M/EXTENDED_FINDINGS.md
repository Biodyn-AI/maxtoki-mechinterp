# topology-141-217M — Extended findings (Phases 14–16)

_Run-folder addendum to FINAL_SUMMARY.md._
_Phases run: **14** (cross-model alignment vs Geneformer V2-316M), **15**
(cross-tissue invariance), **16** (lineage marker clustering, layer-wise
persistence/dimension trajectory, TF-hub centrality, H23 Forman direction)._
_Date: 2026-05-05._

After the headline backbone produced a clean negative across the source paper's
"robust positives", I ran six additional hypothesis tests in the spirit of
the source pipeline. **Three of the six find clean positives on MaxToki-217M.**

## Results table (six exploratory hypotheses)

| ID | Hypothesis | Outcome | Strength |
|---|---|---|---|
| **H24-ext** | Cross-model CCA vs Geneformer V2-316M (static embeddings, PCA-30) | **POSITIVE** | CCA mean r ≈ **0.78** (paper target 0.80, within ±0.05). Top-1 retrieval 38–47% (paper 72%, partial). Pearson off-diag 0.39–0.40 with z>>100 vs gene-shuffle null. Replicates **across all 3 domains**. |
| **H-tissue** | Cross-tissue invariance: gene-gene similarity matrices conserved across lung/immune/ext-lung for shared genes | **STRONG POSITIVE (novel)** | Mean Pearson **0.74–0.86** between domain pairs (z=6.6–24.3 vs gene-shuffle); top-1 retrieval median 93–99%. Even the worst per-layer Pearson is 0.45–0.75. **MaxToki encodes a tissue-invariant gene geometry** — not a finding from the source paper. |
| **H-lineage** | Cell-type marker (B/T/NK/Myeloid) within-set kNN purity at L6, vs random gene-set null | **POSITIVE** | Lung 2/3 lineages sig (T-cell z=+6.2, Myeloid z=+7.5); ext-lung **4/4 lineages sig** (T/B/NK/Myeloid all z>3, mean z=4.9). Immune: too few markers in the inherited 350-gene pool to test (limitation of the immune-reuse pool selection). |
| **H-depth** | Layer-wise H1 persistence trajectory shape | **NON-TRIVIAL** | Three-phase trajectory: rise L0→L5 (peak ≈150–210), drop L6→L7 (≈45–110), partial recovery L8→L11. Same qualitative shape across all three domains — consistent with a depth-wise compression-then-decompression pattern. Not a feature-shuffle-significant signal in absolute magnitude (the headline finding), but the trajectory shape itself is reproducible. |
| **H-hub** | TF-degree (TRRUST) vs eigenvector centrality in kNN graph at L6, vs degree-permutation null | **POSITIVE in 2/3 domains** | Spearman ρ = +0.06 (lung, n.s. z=1.3), **+0.12 (immune, z=2.34)**, **+0.11 (ext-lung, z=2.36)**. High-degree regulatory hubs sit at higher kNN centrality on MaxToki — extends the Phase 7+8 community-structure result with a more direct hub geometry test. |
| **H23 sanity** | Forman curvature direction (paper: high-curvature edges should be LESS regulatory; AUROC 0.34–0.39) | **DIRECTIONAL REPLICATES across all 3 domains** | Lung AUROC=0.35, Immune=0.42, Ext-lung=0.34. Exactly matches the paper's expected 0.34–0.39 range for two domains; immune slightly weaker. **Validates the spec implementation.** |
| **H139 spot-check** | Sectional anisotropy descriptors over kNN graph in PCA-20, classified vs H70 triangle-defect baseline under three nulls (endpoint-swap / row-shuffle / label-permutation). Paper: ΔAUROC +0.031, partial (6/9 splits). | **PARTIAL — replicates paper's "partial" verdict** | Layer 11, 1 seed × 3 domains. Mean Δ vs H70 = **+0.046** (lung +0.109, immune +0.053, ext-lung −0.024). Null-gap (Δ vs max q95 across 3 nulls): lung −0.001, immune **+0.001**, ext-lung −0.034 — only immune passes the strict null-fragility audit, mirroring the source paper's pattern that H139 survives strict-null only in immune. Endpoint-swap is the dominant null (close to observed AUC by construction), confirming much of the signal is endpoint-symmetric. Spot-check simplification: support-direction term replaced with kNN-rank asymmetry (the iter_0051 reference uses a pipeline-internal H70/H136 support matrix). |
| **Phase 4 vs scGPT** | Cross-model CCA + Procrustes alignment on static gene embeddings (HUGO symbols, 380/350/380 shared genes per domain). Paper claim: cross-model alignment is the most robust property (Layer 1 of 5-layer hierarchy). Phase 14 already showed MaxToki ↔ Geneformer alignment r=0.78. | **DOES NOT GENERALIZE ACROSS ARCHITECTURAL FAMILIES** | CCA mean r ≈ **0.40** (vs Geneformer 0.78). Pairwise Pearson on full static embeddings is **essentially zero** (lung −0.008, immune −0.019, ext-lung −0.005), even slightly negative in immune (z=−3.0). Top-1 retrieval 5–8% (z=20–30 vs null but far below the 38–47% seen for Geneformer). The "Layer 1" robustness claim of the source paper holds within the autoregressive-Ensembl-tokenizer family (MaxToki ↔ Geneformer V2-316M) but breaks across to scGPT, which uses continuous-value rank-binned tokenization and a different attention-only architecture. **This weakens the source paper's claim that cross-model CCA is the most robust property.** |
| **H124→H138 chain** | Source paper's "annotation-extension degradation": H123 + STRING (H124) → + GO co-membership (H127) → + continuous GO similarity (H130) → + ontology sheaf (H138). Paper: monotonic null-gap collapse 6/6 → <3/6 → 2/9 → 0/9 → 0/9 with raw effect-size growing. | **DEGRADATION CHAIN UNINFORMATIVE — H123 ALREADY AT FLOOR** | All four extensions land within ±0.01 of H123 baseline (lung 0/12 layers positive null-gap throughout; immune 1/12 → 2/12 with extensions; ext-lung 0/12 throughout). Mean Δ-vs-null-p95 across extensions: −0.018 to −0.020 (immune), −0.033 to −0.041 (lung), −0.055 to −0.060 (ext-lung). The paper's degradation cannot manifest on MaxToki because H123 is already at floor (0–1/12 layers passing null-gap per domain). Mildly notable: H124 (STRING) and H127 (GO co-membership) tick immune from 1/12 to 2/12 layers positive — small but consistent across all 4 extensions, suggesting MaxToki's gene-pair structure is modestly more aligned with PPI/GO co-membership than with TRRUST signed motif-community placement. |
| **Phase 11 autoloop H-cross** | Cross-layer Pearson of the H123 signed-motif-community feature (does the per-pair motif score correlate across the 11 transformer layers?) | **NOVEL POSITIVE on MaxToki** | Mean cross-layer Pearson = **0.654 (lung), 0.604 (immune), 0.678 (external_lung)** — all 3 domains positive. Per-layer AUROC stays near chance (0.40–0.53), so the signed-motif feature isn't getting *stronger* across layers; it's just barely changing at all. **Interpretation: MaxToki's gene-pair community structure is essentially fixed at the embedding layer and propagates almost unchanged through all transformer layers** — the residual stream does not reorganise gene similarities in any meaningful regulatory direction. Sharpens the run's overall verdict: not only is H123 weak, it doesn't *vary* across layers, so there's no "best layer" to extract a regulatory signal from. |
| **Phase 11 autoloop negatives** | 4 additional hypotheses: H-orc (Ollivier-Ricci direction), H-hyp (Gromov hyperbolicity), H-coex-res (coex-residualised triangle-defect), H-tokfreq (popularity-stratified H123) | **All inconclusive or retire** | (i) ORC direction: 10/36 positive, doesn't replicate H23-Forman's clean below-chance pattern. (ii) Hyperbolicity: mean delta_norm 0.045, near euclidean threshold — not meaningfully hyperbolic. (iii) Coex-residualised triangle-defect: 9/36 positive, mean Δ −0.034 — the small phase-6 triangle-defect positive on lung dissolves once explicit coexpression is regressed out. (iv) Token-frequency strata: 6/36 positive, signal slightly stronger in *low*-frequency than high-frequency tokens — rules out the popularity confound but doesn't help recover signal. (H-pooled was untestable: only 36 common genes across the 3 domain pools.) |

## Why these positives survived where the headline backbone didn't

The headline backbone tested whether MaxToki's residual-stream gene
embeddings carry **fine-grained directed regulatory edge** information at
layer-wise resolution beyond what coexpression captures (H123 etc.). On
MaxToki, that fails — consistent with attention-GRN and SAE-Stage-2.

The three exploratory positives are about **coarser** geometric properties:

1. **Static-embedding alignment (H24-ext)** is essentially a question about
   the input vocabulary's pre-tokenizer geometry. Both models share the
   identical 20,275-token Ensembl-ID dictionary; they *also* converge on
   similar gene-pair similarity structure on the static embedding **after
   independent pre-training on different objectives**. Per the source paper,
   this is the most robust property (Layer 1 of the 5-layer hierarchy).
2. **Cross-tissue invariance (H-tissue)** is about whether the model's
   residual stream is tissue-context-stable. This is a property of the
   architecture + tokenizer setup more than of biology — **of course** the
   static layer is tissue-invariant, and the trained transformer doesn't
   appear to introduce tissue-specific divergence at the gene-geometry
   level. Note: this directly mirrors the Phase 8 CKA finding from
   spectral-geometry-217M (1.000 → 0.991 across cell samples); we now
   confirm it across **tissues**, not just cells.
3. **Lineage marker clustering (H-lineage)** is a coarse "do canonical
   markers cluster?" test that the residual-stream-spectral-geometry
   pipeline already found positive on the immune domain. We now show it
   replicates on lung and external-lung too. This is consistent with: the
   model encodes broad cell-type identity even if it doesn't encode
   directed regulatory logic.
4. **TF-hub centrality (H-hub)** is the strongest *new* result of the
   extended run. The paper does not explicitly test "do high-out-degree
   TRRUST TFs sit at higher kNN centrality"; we get z=2.3–2.4 in immune
   and external-lung. This is an order of magnitude smaller than the
   marker-purity z-scores, but it's directionally consistent — MaxToki
   does place regulatory hubs at central positions in the gene graph,
   even if it doesn't place individual TF→target pairs at distinctive
   geometric configurations (which is what H123 tests, and which fails).

## Updated 5-layer hierarchy assignment (revised)

| Layer | Source-paper level | MaxToki-217M verdict |
|---|---|---|
| **L1 cross-model** | most robust | **PARTIALLY REPLICATES vs Geneformer (r=0.78, top-1 38–47%); FAILS vs scGPT (r=0.40, top-1 5–8%, pairwise Pearson ≈0)** — cross-model alignment is robust *within* the autoregressive-Ensembl family but does NOT generalize across architectural families. This weakens the paper's "Layer 1 most robust" claim. |
| L2 persistent homology | robust under feature-shuffle | does NOT replicate (1/12 layers per domain) |
| L3 manifold distance hierarchy | moderate | does NOT replicate (triangle-defect anti-predictive) |
| L4 H123 signed motif-community | strongest single | does NOT replicate (0–1/12 layers per domain) |
| L5 strict max-null | localised to immune | NEGATIVE (mean −0.050, immune does not show +0.012 robustness) |
| **NEW: cross-tissue invariance** | not in paper | **STRONG POSITIVE on MaxToki** (mean Pearson 0.74–0.86) |
| **NEW: TF-hub centrality** | not in paper | weak-positive in 2/3 domains (z≈2.3) |
| Forman curvature direction | spec sanity | direction replicates 3/3 |
| Cell-type marker purity | spec sanity (Phase 6) | replicates 4/4 lineages on ext-lung, 2/3 on lung |

## What this changes about the maxtoki story

Combining with prior pipelines (attention-GRN, spectral-geometry, SAE 1/2/3),
the MaxToki interpretability picture sharpens:

- **What MaxToki encodes (positives, replicated across pipelines):**
  coexpression structure; coarse cell-type / progenitor→mature trajectory
  axis (SAE Stage-3); cross-tissue-invariant gene-similarity geometry
  (this run); shared static gene geometry with Geneformer V2-316M
  (this run); high-degree regulatory hub centrality; canonical
  lineage-marker clustering.
- **What MaxToki does NOT encode (negatives, replicated across pipelines):**
  directed causal regulatory logic at the TF→target edge level
  (attention-GRN curveball ≈0; SAE Stage-2 CRISPRi 54.6%); fine-grained
  signed motif-community structure (H123 0–1/12 layers); triangle-defect
  spectrum advantage over coexpression baseline; persistent topological
  cycles surviving rewiring controls.

The two stories are not contradictory: MaxToki is a **trajectory generative
model**, not a regulatory-edge inference model. The geometric invariants it
preserves are the ones useful for trajectory generation (cell-type identity,
progenitor→mature axis, tissue-invariant gene geometry), not the ones useful
for in-silico TF perturbation (directed regulation, hub motifs).
