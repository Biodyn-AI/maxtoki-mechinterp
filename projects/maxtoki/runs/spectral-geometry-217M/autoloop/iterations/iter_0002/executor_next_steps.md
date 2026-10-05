# Candidate Follow-up Experiments — iter_0002

## Priority 1 (highest upside)

- **STRING-persistent kNN pairs across layers (H03 extension):** Identify which STRING-700 gene pairs remain kNN neighbours at ALL layers vs only at L0 vs only at late layers. Map to STRING interaction categories (coexpression, experimental, textmining, combined). Does the "always-neighbours" set correspond to high-confidence experimental interactions? Does the "L0-only" set correspond to sequence/coexpression correlates? This directly characterises what MaxToki's token embeddings encode vs what contextual processing retains.

- **Re-extract per-cell hidden states for 200 cells per cell type (2000 cells total), stratified by cell type.** Compute cell-type centroid trajectories in full residual-stream space (not just per-gene means). Test: do cell-type centroids converge (similar to differentiation trajectory) or diverge across MaxToki layers? This directly tests the "MaxToki encodes pseudotime" hypothesis. Estimated compute: ~30 min.

## Priority 2 (medium upside)

- **Biological annotation of persistent H0 components (H01 extension):** At each layer, identify which genes form the long-lived topological components (lifetime > 0.10). Run GO BP enrichment on each persistent cluster. Does cluster identity change across layers in a biologically interpretable way?

- **LID vs dropout_rate / mean_expr regression (H02 extension):** Test whether per-gene LID at L0 correlates with mean expression or dropout rate (from gene_features.csv). This would distinguish a "technical noise" explanation from a "functional annotation" explanation for the TF/target LID difference.

- **H1 (1-cycle) persistent topology:** If `pip install ripser` succeeds in the environment, run proper Vietoris-Rips H0+H1 persistence. H1 cycles in the gene embedding space would indicate loop-like regulatory topology (not just clustering). Currently blocked by missing library.

## Priority 3 (quick sanity checks)

- **Non-monotone LID depth profile characterisation:** The LID has a notable increase at L7–L9 relative to L6 and L10 (means: L6=6.8, L7=7.4, L8=7.7, L9=6.6, L10=5.4). Test whether this is reproducible with k=10 and k=30 LID estimation. If consistent, it represents a real non-monotone LID depth transition — potentially linked to MaxToki's trajectory-encoding phase.

- **STRING kNN cohesion within confidence strata (H03 complement):** Confirmed that STRING-700 has higher z than STRING-900 at most layers (L0: 23.0 vs 21.2 is similar, but at other layers STRING-900 is often higher). Run the same kNN precision analysis for STRING interaction subcategories (coexpression vs combined) to test which biological relationship underlies the L0 PPI encoding.
