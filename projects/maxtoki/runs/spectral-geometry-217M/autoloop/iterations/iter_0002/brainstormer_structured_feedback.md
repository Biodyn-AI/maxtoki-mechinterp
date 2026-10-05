# Brainstormer Structured Feedback — iter_0002

**Reviewing:** executor_iteration_report.md, executor_hypothesis_screen.json, executor_next_steps.md

---

## What worked

**H03 is the headline finding of the run.** The kNN precision analysis with a hypergeometric null is the right test for local pair-level co-localization, and the L0 z=23 result is genuinely large. Correctly identifying that the SVD-based SV3 analysis (z=3.12) missed this because SVD axes are dominated by variance structure is a sharp methodological observation — include this in any downstream paper-comparison text.

**Cross-hypothesis synthesis is correct.** The three lenses (H01 topological persistence, H02 local intrinsic dimensionality, H03 graph cohesion) converging on the same story (L0 rich → deep compression → non-monotone depth structure) is appropriately highlighted. The L7-L9 non-monotonicity is flagged consistently across H02 and H03; this is a real MaxToki-specific structural property worth pursuing.

**The Fiedler value decay (0.166 → 0.008, 21×) is a strong complementary result to H03's precision@k.** It tells a compatible story (late-layer gene graph barely connected, consistent with H01 cluster convergence). Good to report both.

---

## Problems

**H01 artefact does not support biological follow-up without a rerun.** The `h01_persistent_topology.csv` records per-layer summary statistics (persistence_ratio, n_long, max_lifetime_ratio) but not per-gene component assignments. The next_steps document correctly identifies GO BP annotation of persistent components as a follow-up, but this requires rerunning the union-find while tracking gene labels. The current artefact is a dead end for that analysis. The executor should have stored the component membership matrix at cost of a few extra lines.

**H02 significance is fragile.** q=0.015 at a single layer (L6) with borderline values at L5, L7, L11 is a weak result. The executor should have tested this with (a) a mixed-effects model across all layers simultaneously, not per-layer Mann-Whitney followed by BH, and (b) confound regression against mean_expr and dropout_rate before reporting it as a biological finding. The LID stratification by functional group is interesting but is not yet a clean claim.

**H03 null is underpowered for the Fiedler comparison.** Using 8 ER random graphs as the null for Fiedler value z-score is not adequate — the executor correctly caveatted that kNN graphs are structurally different from ER graphs, making the z-score non-interpretable. This null should either be dropped or replaced with a degree-preserving random graph null (configuration model) that preserves the kNN degree sequence.

**The "L0 is geometrically rich" framing is partly an artifact of tokenization.** MaxToki L0 embeddings are trained static gene token embeddings (not contextual). The strong PPI encoding at L0 (z=23) may simply reflect that the tokenizer training injected biological co-occurrence priors directly into the static vocabulary embeddings — i.e., this is a tokenizer-level finding, not a transformer inference finding. The executor does not distinguish between "what MaxToki's token embeddings encode" (frozen vocabulary, pre-transformer) and "what the transformer forward pass does." This distinction is critical for interpretation and should be made explicit in the next report.

**No per-cell analysis was attempted.** All three experiments use per-gene mean embeddings averaged over 2000 cells. This was appropriate for speed, but it means every single result in this iteration is conditional on the mean-embedding assumption. Cell-type-specific geometry (the most biologically interesting question) remains completely untested. The executor acknowledged this but treated it as a mild limitation rather than a major gap in the analysis.

---

## Recommendations for iter_0003 executor

1. When building kNN adjacency in any experiment, always save the full per-pair per-layer adjacency matrix (not just per-layer summary statistics). Downstream analyses will need it.
2. Any new LID result must be accompanied by the confound regression (mean_expr, dropout_rate) before being reported as significant.
3. Replace the ER null for Fiedler values with a configuration-model null (preserve kNN degree sequence, randomize which genes are connected). `networkx.random_degree_sequence_graph` or `igraph.Erdos_Renyi` with `degree` argument.
4. When reporting L0 findings, explicitly label them as "token embedding" findings and distinguish from "transformer inference" findings. The two can be separated by comparing L0 results to Geneformer static (same tokenizer but different training — Geneformer static z=5.03 for PPI; MaxToki L0 z=23 — the gap is likely training-regime-specific).
