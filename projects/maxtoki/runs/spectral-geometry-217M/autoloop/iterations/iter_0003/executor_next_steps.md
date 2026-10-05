# iter_0003 — Candidate Follow-ups (iter_0004)

## Priority 1 (HIGH): Annotate the 15 always+early STRING-persistent pairs (H3-BC follow-up)

The 14 "early" persistent pairs (93% in 900+ tier) and the 1 "always" pair (all 12 layers) are the most structurally invariant STRING interactions in MaxToki-217M embeddings. Retrieve per-channel STRING scores (experimental, coexpression, textmining) for these specific pairs from the STRING v12.0 full score file and compare to "never" pairs. If "always"+"early" pairs are dominated by experimental evidence channels while "never" pairs are predominantly coexpression/textmining, this would confirm the original H3-BC hypothesis at the channel level.

## Priority 2 (HIGH): H3-G result interpretation + pseudotime axis test (pending H3-G completion)

Once H3-G centroid trajectory results are in:
- If related pairs converge: test whether the convergence is lineage-ordered (Kendall τ with HSC→progenitor→effector ordering). Test whether the converging direction projects onto the leading PCA axis of centroids at L11.
- If all pairs compress equally: run the "lineage-agnostic" analysis — does the L11 centroid distance matrix cluster cells by gross lineage (lymphoid/myeloid) without intra-lineage resolution?

## Priority 3 (MEDIUM): OLS confound correction for LID — verify at other layers (H3-DEL follow-up)

The Sub-E OLS showed TF_only coef is NOT significant at L6 after expression/dropout control, but target_only IS significant. Test whether this pattern holds at L0, L3, and L11. If target_only LID elevation is robustly above-confound at multiple layers, this is a genuine biological signal about target gene neighborhood geometry, not just a technical artifact.

## Priority 4 (MEDIUM): Persistent H0 component biological annotation (H01 follow-up)

H01 (iter_0002) found fewer long-lived H0 components in real vs shuffle at all layers (the real embedding compresses). Now that we have layer_count per STRING pair (H3-BC), annotate the 130 non-"never" pairs: are they enriched in specific GO BP terms or TRRUST TF-target pairs? Use the h3bc_pair_categories.csv with GO annotation to test whether persistent pairs are co-functional modules.

## Priority 5 (LOW/MOONSHOT): Install ripser and run H1 persistent topology

The `ripser` package was not available in iter_0002. Try `pip install ripser` and test whether 1-cycles (loops) exist in the gene-embedding filtration at any layer. Predict: at L0 (PPI-encoding), circular structures may reflect feedback regulatory loops; at L11 (compressed), 1-cycles should collapse. This is the original H01's deferred test.
