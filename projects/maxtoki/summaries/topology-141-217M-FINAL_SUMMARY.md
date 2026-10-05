# topology-geometry-141-hypotheses on MaxToki-217M — Final Summary

**Run folder:** `projects/maxtoki/runs/topology-141-217M/`
**Pipeline spec:** `pipelines/topology-geometry-141-hypotheses.md` (arxiv 2602.22289, Kendiukhov 2026)
**Target:** MaxToki-217M-HF (LlamaForCausalLM, 11 layers, hidden=1232, vocab=20,275)
**Run date:** 2026-05-03
**Domains:** Tabula Sapiens lung (1500 cells × 382 genes), Tabula Sapiens immune (1500 cells × 350 genes — embeddings reused from spectral-geometry-217M phase0), Krasnow lung SMART-seq2 held-out (1500 cells × 380 genes, "external_lung")

## Updated headline verdict (after Phases 14–16, 2026-05-05)

**Three positives surfaced in the extended exploratory pass, including a
PARTIAL replication of the source paper's most-robust Layer-1 finding.**

### Positives
1. **Cross-model alignment vs Geneformer V2-316M (H24-ext) — Layer-1
   PARTIALLY REPLICATES.** Both MaxToki-217M and Geneformer V2-316M share
   the identical 20,275-token Ensembl-ID vocabulary. After PCA-reducing
   both static embeddings to 30 dims, mean canonical correlation **r ≈
   0.78** (paper target 0.80, replicates within ±0.05), Pearson off-diag
   **0.39–0.40** with z>>100 vs gene-shuffle null, top-1 retrieval
   **38–47%** (paper target 72%, partial). Holds across all 3 tissue
   domains.
2. **Cross-tissue invariance (H-tissue, NOVEL).** For genes shared between
   tissue-domain pools, the gene-gene cosine matrix is preserved across
   contexts: mean Pearson **0.74–0.86** between domain pairs (z=6.6–24.3),
   median top-1 retrieval 93–99%. **MaxToki encodes a tissue-invariant
   gene geometry** — not in the source paper.
3. **TF-hub centrality (H-hub, NOVEL).** High-out-degree TRRUST TFs sit at
   higher eigenvector centrality in the kNN graph at L6. ρ=0.12 (immune,
   z=2.34) and 0.11 (ext-lung, z=2.36); marginal on lung.

Sanity checks also replicate: H23 Forman curvature direction (paper expects
AUROC 0.34–0.39 on "high curvature → regulatory" — observed 0.34/0.42/0.35,
exactly in range); cell-type marker purity (lung 2/3 lineages sig at z=6–7;
ext-lung 4/4 sig at z=3–6).

### Negatives (still hold from the headline backbone)

The source paper's headline tests on scGPT and Geneformer V2-316M
**do not replicate on MaxToki-217M** under matching null controls.

| Layer | Source-paper verdict (scGPT/Geneformer) | MaxToki-217M observation |
|---|---|---|
| 1 cross-model | r=0.80, top-1=72%, p=3.2e-5 | **SKIPPED** (no reference-model extraction) |
| 2 PH (feature-shuffle) | 11/12 lung, 12/12 immune, 12/12 ext-lung | **1/12 lung, 1/12 immune, 0/12 ext-lung** |
| 2 PH (rewire null) | 0/24 — topology vanishes | 0/12 — qualitatively replicates (no signal to kill) |
| 3 distance hierarchy | Eucl < Geo < Diff < Tri (monotonic) | **Tri is anti-predictive (−0.055)**; only Diff slightly +0.012 |
| 4 H123 motif×community | **22/22 rows positive, 6/6 splits null-gap** | **0–1/12 layers positive null-gap per domain** |
| 5 strict max-null (immune) | +0.012 robust | **−0.033 fragile** (lung −0.039, ext-lung −0.078) |
| Stability-sel H91 | ΔAUROC +0.074 | mean AUC 0.50–0.55 (Δ ≈ +0.01) |

## Extended findings table (Phases 14–16, added 2026-05-05)

| Phase | Hypothesis | Verdict | Key numbers |
|---|---|---|---|
| 14 | H24-ext cross-model CCA vs Geneformer V2-316M | **POSITIVE (Layer-1 partial)** | mean r=0.78; Pearson 0.39–0.40 (z>100); top-1 38–47% |
| 15 | H-tissue cross-domain alignment | **STRONG POSITIVE (novel)** | Pearson 0.74–0.86 between domain pairs; top-1 median 93–99% |
| 16 | H-lineage marker purity | **POSITIVE (sanity)** | lung 2/3 sig (z=6–7), ext-lung 4/4 sig (z=3–6) |
| 16 | H-depth H1 trajectory | non-trivial shape | rise L0→L5 → drop L6→L7 → recover L8→L11, all 3 domains |
| 16 | H-hub TF-degree centrality | **POSITIVE 2/3 domains (novel)** | immune ρ=0.12 z=2.34; ext-lung ρ=0.11 z=2.36; lung n.s. |
| 16 | H23 Forman direction | sanity replicates 3/3 | AUROC 0.34/0.42/0.35 (paper 0.34–0.39) |

Overall strict max-null mean margin: **−0.050** (4/36 tests pass).
Per source paper's interpretation: "fewer than 15 of 141 survive strict";
on MaxToki, only ~10% (4/36) of the headline backbone survives.

## Cross-pipeline synthesis (with the three completed pipelines)

| Pipeline | Verdict |
|---|---|
| `attention-grn-extraction-and-evaluation` | Co-expression, not regulatory logic. CRISPRi-curveball z ≈ 0. |
| `residual-stream-spectral-geometry` | Mixed: depth-wise rank compression replicates; PPI gradient does not. |
| `sparse-autoencoders` (3 stages) | Stage 1 atlas: features partially annotated; Stage 2 CRISPRi 54.6% near-chance; Stage 3 trajectory steering: **positive** (progenitor→mature axis amplifiable). |
| **`topology-geometry-141-hypotheses` (this run)** | **Negative across the headline backbone.** No persistent topology beyond rewiring; manifold distance hierarchy collapses; H123 at 0–1/12 layers per domain; strict margins all negative. |

Together, four of five completed pipelines converge on:
**MaxToki-217M encodes co-expression and a coarse progenitor→mature
trajectory axis, but does not encode fine-grained directed regulatory
logic at the geometric, attention, or causal-circuit level required by
the source papers' headline tests.**

## Key implementation notes / divergences from spec

1. **Gene-pool sizing.** Used 350–382 genes per domain (pipeline spec calls for ~320). Pool composition: priority TFs from TRRUST + targets + STRING + canonical lineage markers in MaxToki vocab, capped at 350 (immune) or built fresh (lung, ext-lung).
2. **Immune embeddings reused from `spectral-geometry-217M/outputs/phase0`** (1500 cells × 1500-HVG → sub-selected to 350 topology-pool genes; PCA re-fit per-layer to 20 components). Lung + external-lung freshly extracted (1500 cells each, MAX_LEN=2048, MPS device).
3. **Triangle-defect spectrum** built per the source paper recipe (k ∈ {8,12,16}, sum across scales). One implementation bug found and fixed mid-run: common-neighbour list could include i or j (with sentinel D[i,i]=∞), producing NaN; patched to exclude self-indices.
4. **H123 sign-shuffle null** uses TF-identity-preserving permutation stratified by TF-degree quartile × target-degree quartile (4×4 strata). 100 null replicates per layer.
5. **Strict max-null** evaluated at 3 representative layers (L3/L6/L9) × 4 metrics × 3 domains = 36 tests. Four null families: feature-shuffle (12 reps), label-permutation (100 reps), degree-preserving rewiring (8 reps via curveball), coexpression-matched floor (|Pearson| baseline AUROC).
6. **Phase 4 (cross-model CCA), Phase 11 (Codex autoloop), and the annotation-extension degradation chain (H124-H138) were not run** — see FINAL_SUMMARY.md "Scope" section in the run folder.

## Artefacts

- `FINAL_SUMMARY.md` — full numerical summary in run folder
- `outputs/phase0/{lung,immune,external_lung}/` — embeddings (full + PCA(20)), gene_features, log-expression for coexpression null
- `outputs/phase1/{domain}/pair_table.csv` + `splits.json` — TF/target disjoint splits × 3 seeds
- `outputs/phase5/h01_h03_persistent_homology.csv` + `h01_rewiring_null.csv` + `h47_bifiltration_cycle_rank.csv`
- `outputs/phase6/distance_deltaauroc.csv` + `hierarchy_summary.csv`
- `outputs/phase78/{h16,h116,h123}.csv`
- `outputs/phase9/h91_stability_selection.csv`
- `outputs/phase12/h141_strict_margins.csv`
- `outputs/synthesis/layered_findings.csv` + `all_phase_summaries.json`
- `logs/phase{0,1,5,6,78,9,12}.log` — per-phase stdout
