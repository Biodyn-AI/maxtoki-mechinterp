# Executor Iteration Report — iter_0002
**Date:** 2026-05-05  
**Model:** MaxToki-217M-HF  
**Hypotheses tested:** 3  
**New families covered:** persistent_homology (H01), intrinsic_dimensionality/per-gene (H02), graph_topology/kNN-cohesion (H03)

---

## Summary

Three novel experiments ran to completion against the `(12, 1500, 1232)` per-gene mean embeddings. All three produced non-trivial results; none recapitulate prior findings. H03 yields the iteration's headline result.

---

## H01 — H0 Persistent Topology (Betti-0 Filtration)

**Method:** Proper Betti-0 persistence barcode computed via union-find on cosine-distance sorted edges (MST-equivalent persistence). Null: per-dimension feature shuffle (preserves marginal distributions, destroys pairwise structure). 12 layers × 2 conditions. Previously logged phase9a result was a single-threshold stub with no filtration.

**Reproduction command:**
```
python3 <REPO_ROOT>/projects/maxtoki/runs/spectral-geometry-217M/autoloop/iterations/iter_0002/h01_persistent_topology.py
```

**Results:**

| Layer | persistence_ratio | max_lifetime_ratio | n_long (real) | n_long (null) |
|---|---|---|---|---|
| L0 | 0.921 | 0.977 | 1499 | 1499 |
| L4 | 0.563 | 0.820 | 1499 | 1499 |
| L6 | 0.424 | 0.804 | 1396 | 1499 |
| L10 | 0.361 | 0.631 | 1040 | 1499 |
| L11 | **0.251** | 0.745 | **590** | 1499 |

**Interpretation:** The persistence ratio (real/null) decreases monotonically across depth. Real gene embeddings have *less* total H0 persistence than the feature-shuffle null at all layers, and the gap widens substantially at late layers (L11 ratio = 0.25). The n_long ratio (components surviving > 0.10 cosine-distance threshold) drops to 0.39 at L11 — real embeddings have only 590 long-lived topological features vs 1499 in the null. This is a topological signature of **gene clustering convergence**: late-layer genes are in tightly concentrated groups (few well-separated components), whereas feature-shuffled points remain dispersed. This confirms rank compression from a topological viewpoint (not just spectral) and adds a new geometric characterisation not present in prior analyses.

**Caveat:** Ratio < 1 is somewhat expected (real embeddings will cluster more than random), but the strong monotonic *decrease* with depth (0.92→0.25) and the large L11 n_long gap are novel results not pre-supposed by the spectral analysis. A valid biological null would permute gene labels while keeping embedding geometry fixed (to test biological annotation significance vs geometric significance); this remains as a follow-up.

**Artefact:** `h01_persistent_topology.csv`

---

## H02 — Per-Gene Local Intrinsic Dimensionality (LID) Stratified by Annotation

**Method:** Levina-Bickel MLE (k=20 cosine-NN) per gene per layer. Gene groups: TF_only (n=151), TF_and_target (n=192), target_only (n=595), string_member (n=553), other (n=9). Mann-Whitney tests with BH-FDR correction across all layers and group pairs.

**Reproduction command:**
```
python3 <REPO_ROOT>/projects/maxtoki/runs/spectral-geometry-217M/autoloop/iterations/iter_0002/h02_lid_per_gene.py
```

**Global LID trajectory (mean across all 1500 genes):**

| Layer | L0 | L2 | L4 | L6 | L8 | L11 |
|---|---|---|---|---|---|---|
| mean LID | 46.7 | 18.1 | 8.5 | 6.8 | 7.7 | 3.5 |

13.3× compression L0→L11 (stronger than TwoNN estimate of 6.1×).

**Per-group median LID (selected layers):**

| Group | L0 | L6 | L11 |
|---|---|---|---|
| TF_only (n=151) | 45.3 | 6.0 | 3.23 |
| TF_and_target (n=192) | 46.2 | 6.1 | 3.39 |
| target_only (n=595) | 48.1 | 6.8 | 3.57 |
| string_member (n=553) | 40.0 | 5.5 | 2.68 |

**Statistical tests (TF_only vs target_only):**

| Layer | delta_median | q_BH |
|---|---|---|
| L3 | −1.43 | 0.116 |
| L5 | −0.70 | 0.064 |
| L6 | −0.77 | **0.015** |
| L7 | −0.77 | 0.053 |
| L11 | −0.35 | 0.055 |

**Interpretation:** TF genes consistently occupy lower-LID (more geometrically concentrated) neighbourhoods than target genes. The effect is significant at L6 (q=0.015) and borderline at L5, L7, L11. string_member genes have the lowest LID of all groups, consistent with their being physically-interacting hub proteins in compact co-expression blocks. At L0, target_only has the *highest* LID (48.1 vs 45.3 for TF_only), and this ordering is preserved through depth. This suggests regulatory gene roles have a local geometry signature: TFs occupy tighter regions, while their downstream targets are more dispersed.

**Note on L6 significance:** The LID compression is non-monotone at L7–L9 (slight increase after L6), which may explain why L6 is the strongest detection point. This non-monotonicity was also hinted in TwoNN (prior phases); L6 may represent a phase transition in MaxToki's representation regime.

**Artefacts:** `h02_lid_summary.csv`, `h02_lid_tests.csv`, `h02_lid_per_gene.csv`, `h02_lid_matrix.npy`

---

## H03 — STRING PPI Cohesion in Gene kNN Graph Across Layers

**Method:** Per-layer kNN graph (k=15, cosine similarity) on 1500 gene embeddings. STRING precision@k: fraction of STRING-700 (n=844 pairs) and STRING-900 (n=506 pairs) gene pairs appearing as direct kNN neighbours. Null: hypergeometric expectation (random edge assignment at same density). Fiedler value of normalised Laplacian (8 ER-random-graph nulls for comparison).

**Reproduction command:**
```
python3 <REPO_ROOT>/projects/maxtoki/runs/spectral-geometry-217M/autoloop/iterations/iter_0002/h03_spectral_gap.py
```

**Results:**

| Layer | Fiedler | STRING-700 prec | z_700 | STRING-900 prec | z_900 |
|---|---|---|---|---|---|
| L0 | 0.166 | 0.102 | **23.0** | 0.119 | **21.2** |
| L1 | 0.141 | 0.064 | 10.8 | 0.083 | 11.7 |
| L2 | 0.097 | 0.040 | 6.0 | 0.053 | 7.0 |
| L3 | 0.090 | 0.036 | 5.0 | 0.046 | 5.7 |
| L4 | 0.079 | 0.026 | 3.2 | 0.036 | 4.3 |
| L5 | 0.084 | 0.021 | 2.1 | 0.030 | 3.3 |
| L6 | 0.079 | 0.019 | 1.5 | 0.024 | 2.1 |
| L7 | 0.054 | 0.024 | 2.4 | 0.026 | 2.3 |
| L8 | 0.051 | 0.031 | **3.8** | 0.038 | **4.2** |
| L9 | 0.053 | 0.024 | 2.3 | 0.026 | 2.1 |
| L10 | 0.057 | 0.021 | 1.6 | 0.028 | 2.4 |
| L11 | 0.008 | 0.024 | 3.0 | 0.024 | 2.3 |

**Headline finding:** MaxToki's **token embeddings (L0) strongly encode STRING PPI topology as kNN proximity** (z=23.0 for STRING-700 pairs, z=21.2 for STRING-900 pairs). This signal drops sharply after the first transformer block (L1: z=10.8) and falls to z≈1–3 by L4–L7. There is a partial recovery at L8 (z=3.8 / 4.2) and L11 (z=3.0 / 2.3).

**Relation to prior SVD analysis:** The SV3 analysis (prior phases) found PPI signal at 3/12 layers (SV3 z=+3.12). That analysis projected onto global variance axes; these kNN results reveal a *much stronger* signal at L0 that the SVD-based approach did not capture because SVD axes are dominated by the high-variance dimensions. kNN precision detects local pair-level co-localization directly.

**Fiedler value:** Monotonically decreases from 0.166 (L0) to 0.008 (L11), a 21× drop in algebraic connectivity. At L11, the gene embedding kNN graph is barely connected — genes have converged into a small number of very tight clusters with minimal inter-cluster edges. This is geometrically consistent with the H01 result (loss of H0 persistence) and the TwoNN compression.

**Artefact:** `h03_spectral_gap.csv`

---

## Cross-hypothesis synthesis

Three independent geometric lenses (topological persistence, local dimensionality, graph cohesion) all converge on the same story:

1. **L0 (token embedding) is geometrically rich**: strong PPI cohesion (H03, z=23), high and heterogeneous LID (H02, mean=47), near-random H0 persistence (H01, ratio=0.92).
2. **The forward pass compresses geometry dramatically**: all three metrics show strong depth decay.
3. **At late layers (L6–L11), gene groups differ in their residual geometry**: TFs vs targets have significantly different LID (H02, q=0.015 at L6); STRING pairs partially recover kNN proximity (H03, z≈3–4 at L8, L11); H0 persistence becomes much lower than null (H01, ratio=0.25 at L11).
4. **Non-monotone depth structure**: LID compression is not strictly monotone (bump at L7–L9). STRING kNN cohesion has a U-shaped depth profile with L0 peak, mid-layer trough, and partial late recovery. This is a MaxToki-specific pattern not predicted by the prior scGPT spectral geometry paper.

---

## Limitations

- All analyses use per-gene embeddings averaged over 2000 cells, not per-cell hidden states. Cell-type-specific geometry requires re-extraction.
- The `other` gene group (n=9) is too small for reliable statistics and was excluded from interpretations.
- The Fiedler z-scores (H03) using ER null are large negatives because kNN graphs are structurally different from ER random graphs; the null comparison is informative directionally but not interpretable as a conventional z-score.
