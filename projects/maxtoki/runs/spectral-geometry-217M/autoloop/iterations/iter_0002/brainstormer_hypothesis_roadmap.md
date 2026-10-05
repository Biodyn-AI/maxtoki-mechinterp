# Brainstormer Hypothesis Roadmap — iter_0002 → iter_0003

**Date:** 2026-05-05  
**Iteration reviewed:** iter_0002  
**Previous artefacts consumed:** executor_iteration_report.md, executor_hypothesis_screen.json, FINAL_SUMMARY.md, executor_next_steps.md

---

## Retire / Deprioritize

The following directions have accumulated ≥ 2 negative or fully-inconclusive outcomes with adequate controls. Do not allocate executor cycles to these without a major methodological change.

| Direction | Outcome history | Decision | Reason |
|---|---|---|---|
| TF-vs-target regulatory edge-level AUROC (SV5-SV7) | phase5b: ρ=+0.35 (opposite); phase5 co-expression residualization: SV5-SV7 collapses to null | **retire_now** | Two attempts with different controls both null. No residual signal after co-expression correction. |
| Repression/activation asymmetry | phase5c: mean repression AUROC 0.500, activation 0.500 | **retire_now** | Completely null. Paper's +0.021 delta not recoverable. |
| STRING confidence gradient ρ=1.000 replication | phase4: Δ700→900 = −0.16 (wrong direction); H03: L0 z_700=23 ≈ z_900=21.2 (confidence tiers inseparable at L0) | **retire_now** | MaxToki encodes PPI binarily, not with graded confidence. Confirmed from two angles. |
| TRRUST pairs beyond STRING (non-PPI regulatory geometry) | phase4ext: z=−0.34, 0/12 significant | **retire_now** | No signal. The PPI geometric signal is STRING-specific, not general to regulatory databases. |
| BCL6 metabolic isolation / B-cell attractor dynamics | phase7: 0/20 metabolic neighbors, GC-plasma angle converges (opposite paper direction) | **retire_now** | Gene-set too narrow AND the paper's attractor geometry is architecture-specific (bidir encoder vs causal decoder). Not rescuable without scGPT-compatible bidirectional architecture. |
| TF-TF vs TF-target SVD probe (SV2-SV4 and SV5-SV7) | phase5ext: TF-TF=0.501, TF-target=0.525 (SV2-SV4); both 0.500 (SV5-SV7) | **retire_now for SVD** | SVD axes cannot distinguish TF clustering. The signal has migrated to LID-based geometry (H02); pursue there instead. |
| FFL intermediacy geometry | phase9c: anti-intermediacy z=−1.93, 6/12 layers | **rescue_once_with_major_change** | Paper found null; we find anti-intermediacy. Rescue attempt: stratify by signed FFL type (+/+ vs +/−) before any further test. Do not rerun unsigned. |

---

## New Hypothesis Portfolio

### H-A — Biological annotation of L11 persistent H0 topological clusters

**Hypothesis:** The 590 long-lived topological components at L11 (from H01) are not arbitrary — they correspond to biologically coherent gene modules (GO terms, pathways, co-regulatory programs).

**Test design:** Rerun H01 union-find with gene label tracking (store per-gene component assignment at L11, threshold = lifetime > 0.10 cosine distance). Cluster the 590 long-lived components by shared annotations. Run Fisher's exact test + BH-FDR against GO BP, Reactome, and TRRUST TF-target co-annotation. Use `h01_persistent_topology.py` as base; modify to output `gene → component_id` at each layer.

**Expected signal if true:** Clusters enriched for specific pathways (e.g., one cluster = immune effector genes, another = metabolic hub genes). The most persistent component at L11 should contain STRING high-confidence hubs.

**Null/control:** Permute gene labels within the same component size distribution; test whether enrichment z-scores exceed permuted expectation.

**Value:** high | **Cost:** low (reuses H01 infrastructure, adds ~30 lines)

---

### H-B — STRING-persistent kNN pairs: always / L0-only / late-only classification

**Hypothesis:** The STRING pairs that are kNN neighbors at L0 (z=23) are a heterogeneous mix. The subset that remain neighbors across ALL 12 layers corresponds to experimentally-confirmed direct physical interactions; the L0-only subset corresponds to coexpression / textmining associations that the forward pass discards.

**Test design:** For each of the STRING-700 gene pairs, record whether they are direct kNN-15 neighbors at each of 12 layers (boolean, 12-dim vector per pair). Classify pairs as: `always` (12/12 layers), `early-persistent` (L0-L3, dropping after), `late-persistent` (surviving at L8–L11), `L0-only` (present only at L0), `never`. Cross-tabulate pair categories with STRING channel weights (coexpression, experimental, textmining, combined). Test whether `always` pairs are enriched for high experimental weight vs coexpression weight (Mann-Whitney).

**Input artefacts:** `outputs/phase0/layer_gene_embeddings.npy`, STRING-700 pair list from phase4 artefacts. **New computation required:** rerun kNN per layer, tracking per-pair adjacency (N_pairs × 12 boolean matrix, small).

**Expected signal if true:** `always` set enriched for experimental interaction channel (physical binding); `L0-only` set enriched for coexpression/textmining. Extends H03's headline finding into mechanistic interpretation.

**Null/control:** Hypergeometric random-pair baseline for each category; pair category assignments under gene-label shuffle.

**Value:** high | **Cost:** low

---

### H-C — Per-gene PPI proximity decay rate vs STRING confidence

**Hypothesis:** STRING pairs decay from kNN proximity at different rates depending on their confidence tier. High-confidence (900+) pairs decay more slowly than low-confidence (700-900) pairs — i.e., MaxToki's forward pass preferentially preserves confident PPI signal even if it cannot discriminate confidence at a single layer (which phase4 showed).

**Test design:** For each STRING-700 pair, compute a decay score = number of layers where the pair is a kNN-15 neighbor (0-12). Split by confidence tier (700-900 vs 900+). Mann-Whitney test on decay score distribution. Also test by interaction channel.

**Expected signal if true:** Median decay score (900+) > median decay score (700-900). This would rescue the "confidence gradient" finding at the layer-trajectory level even though it is absent at any single layer.

**Null/control:** Shuffle confidence tier labels; compare real group difference to shuffled distribution.

**Value:** high | **Cost:** low (uses H-B pair adjacency matrix as input)

---

### H-D — STRING hub degree predicts LID at L11 (Spearman correlation)

**Hypothesis:** Per-gene STRING-700 degree (number of STRING-700 partners among the 1500 HVGs) is the primary predictor of LID at L11. TF/target group differences in LID are partially mediated by degree, since TFs tend to be high-degree nodes.

**Test design:** (1) Compute per-gene STRING-700 degree from existing STRING pair list. (2) Spearman(degree, LID_L11) across all 1500 genes. (3) Partial correlation: TF_group → LID_L11 controlling for degree. If the TF/target LID difference (H02, q=0.015 at L6) collapses after degree control, degree is the true driver.

**Input artefacts:** `h02_lid_per_gene.csv`, `gene_features.csv`, STRING-700 pair list.

**Expected signal if true:** Spearman(degree, LID_L11) ≈ −0.3 to −0.5 (higher degree = lower LID). TF/target partial correlation attenuates but does not vanish (functional role contributes beyond degree).

**Null/control:** Permute gene labels on degree vector; test against null Spearman.

**Value:** high | **Cost:** very low (< 30 lines, existing data)

---

### H-E — LID confound regression: mean expression and dropout rate

**Hypothesis:** The TF/target LID difference at L6 (q=0.015, H02) is confounded by technical gene properties: TFs tend to be more highly expressed and have lower dropout rates, which could independently drive LID stratification in the embedding space.

**Test design:** Load `gene_features.csv` (mean expression, dropout rate per gene). Compute Spearman(mean_expr, LID_L0), Spearman(dropout_rate, LID_L6), Spearman(mean_expr, LID_L11). Fit OLS: LID_L6 ~ group + mean_expr + dropout_rate. Report whether group coefficient (TF vs target) remains significant after controlling for technical covariates.

**Input artefacts:** `h02_lid_per_gene.csv`, `outputs/phase0/gene_features.csv`.

**Expected signal if true:** Partial effect of TF group remains significant (i.e., it is a genuine functional signature, not a noise confound). If it vanishes, the H02 result is a technical artifact.

**Null/control:** OLS null = expression/dropout alone predicts LID with no group coefficient needed.

**Value:** high (confound control — necessary before claiming H02 as biological result) | **Cost:** very low

---

### H-F — Cross-architecture conserved PPI pairs (MaxToki L0 ∩ Geneformer static)

**Hypothesis:** The STRING pairs that are kNN neighbors in BOTH MaxToki-217M L0 AND Geneformer-V2-316M static embeddings represent the most strongly convergent, architecture-independent PPI encoding — likely corresponding to well-studied, high-confidence physical complexes.

**Test design:** Build kNN-15 graph on Geneformer static embeddings (already loaded in phase9b artefacts). For each STRING-700 pair, record adjacency in Geneformer static. Intersect with MaxToki L0 adjacency (from H-B). Classify pairs as: both-models (most conserved), MaxToki-only, Geneformer-only, neither. Run GO/TRRUST enrichment on the both-models set. Hypothesis: both-models set is enriched for high experimental confidence and known stable complexes (ribosome subunits, proteasome, etc.).

**Input artefacts:** `outputs/phase9b/` (Geneformer static embeddings), H-B adjacency matrix for MaxToki L0.

**Expected signal if true:** both-models set has higher STRING experimental weight than MaxToki-only pairs. GO enrichment reveals canonical stable complexes (ribosome, proteasome, ATP synthase).

**Null/control:** Hypergeometric baseline for intersection size; gene-label shuffle on Geneformer embeddings to test whether intersection is above chance.

**Value:** high | **Cost:** low (both embeddings exist)

---

### H-G — Cell-type centroid trajectory across layers (pseudotime encoding test)

**Hypothesis:** MaxToki's forward pass (depth) encodes differentiation pseudotime. Per-cell-type centroids in the residual stream trace a biologically ordered trajectory across layers, with more differentiated cell types (e.g., plasma cells) separating from progenitor-like types (e.g., naive B cells) as depth increases.

**Test design:** Re-extract per-cell hidden states from MaxToki-217M-HF for 200 cells each from 5 cell types: naive B, memory B, T CD4, T CD8, NK, monocyte. Compute per-cell-type centroid in residual stream (1232-dim) at each of 12 layers. Track: (1) pairwise centroid cosine distances across layers; (2) whether related types (naive B vs memory B) converge or diverge with depth; (3) whether a UMAP of centroids reveals a pseudotime-like ordering at late vs early layers.

**Input artefacts:** MaxToki-217M-HF weights (already used in phase0), Tabula Sapiens immune lineage cell metadata. **New computation:** 30-40 min for 1000+ cells inference.

**Expected signal if true:** Late-layer centroid distances between related cell types are smaller (convergence toward cell-type identity), while unrelated types remain separated. Or: a clear ordering (HSC → progenitor → effector) emerges in late layers.

**Null/control:** Random-cell label shuffle within each extraction; bootstrap centroid stability across 3 random samples of 200 cells per type.

**Value:** high | **Cost:** high (new extraction, ~40 min)

---

### H-H — Non-monotone LID depth profile robustness (k=10, k=30)

**Hypothesis:** The L7-L9 LID bump (L6 mean=6.8, L7=7.4, L8=7.7, L9=6.6, L10=5.4) is a real geometric phase transition, not an artifact of k=20 nearest-neighbor estimation. It reflects a MaxToki-specific depth regime — possibly where trajectory-level context is being integrated.

**Test design:** Rerun LID (Levina-Bickel MLE) at k=10 and k=30. Report per-layer mean LID and compare to k=20 baseline. If the L7-L9 bump persists across k values, it is real. Additionally, compute per-group (TF/target/string_member) LID at k=10, k=30 and test whether significance at L6 (q=0.015 at k=20) holds.

**Input artefacts:** `outputs/phase0/layer_gene_embeddings.npy`, `gene_features.csv`. **New computation:** ~5 min.

**Expected signal if true:** L7-L9 bump visible at k=10 and k=30, confirming it is a structural property of the embedding, not an estimator artifact.

**Null/control:** Monotone smooth interpolation as the null shape; test whether bump exceeds interpolation by > 1 SD.

**Value:** medium | **Cost:** very low

---

### H-I — Signed FFL geometry: activating vs repressive FFLs

**Hypothesis:** The anti-intermediacy signal in FFL geometry (phase9c, z=−1.93) is sign-specific. Co-activating FFLs (+→+→+) show different geometric behavior than mixed or repressive FFLs (containing at least one − edge). MaxToki may encode the DIRECTION of regulation even if it cannot encode regulatory edges generally.

**Test design:** Split the 1152 TRRUST FFLs by sign: (a) all-positive edges (TF1 →+ TF2 →+ Target); (b) mixed-sign (at least one −). Rerun phase9c cosine distance analysis separately for each class. Compare anti-intermediacy z-scores between classes.

**Input artefacts:** TRRUST regulatory sign data (from TRRUST v2, already loaded in phase5 infrastructure), FFL list from phase9c.

**Expected signal if true:** All-positive FFLs show weaker anti-intermediacy (or even positive intermediacy), while mixed-sign FFLs show strong anti-intermediacy. This would rescue the "signed regulation" hypothesis from the SV5-SV7 null.

**Null/control:** Permuted FFL sign labels; compare group difference to sign-permutation baseline.

**Value:** medium | **Cost:** low

---

### H-J — L8 PPI recovery: attention head attribution

**Hypothesis:** The partial recovery of STRING PPI kNN cohesion at L8 (z=3.8 vs z=1.5 at L6) is driven by specific attention heads at L8 that preferentially attend to STRING protein-interaction neighbors.

**Test design:** Extract attention weights from MaxToki-217M-HF at layer 8, averaged over the same 2000 cells. For each attention head, compute: STRING-pair attention enrichment = mean attention weight on STRING-700 pairs vs random gene pairs (z-score against gene-label-shuffle null). Identify head(s) with z > 2. Compare to L6 and L11 attention head STRING enrichment.

**Input artefacts:** MaxToki-217M-HF (requires `output_attentions=True` extraction, same pipeline as phase0 but saving attention weights). **New computation:** ~20 min (single pass over 2000 cells).

**Expected signal if true:** 1-3 specific L8 heads show STRING-700 pair attention enrichment (z > 2), while L6 heads do not. This would identify the mechanistic origin of the L8 recovery — an attention head that "looks up" protein-interaction partners.

**Null/control:** Gene-label shuffle on attention weight matrices; compare observed STRING pair enrichment to shuffled distribution.

**Value:** high | **Cost:** medium (new extraction, attention weights)

---

### H-K — Layer transition sharpness via residual displacement profile

**Hypothesis:** The residual stream does not change uniformly across depth. Specific layer transitions (L0→L1, L5→L6, L7→L8) correspond to abrupt geometric displacements — candidates for "phase transitions" where representation regime shifts.

**Test design:** For each gene, compute per-layer displacement vector: δ_l = h_{l+1} − h_l. Compute mean |δ_l|_2 and mean cosine similarity between h_l and h_{l+1} across all genes. Plot displacement magnitude and directional coherence across layers. Test whether L0→L1, L7→L8 displacements are larger than adjacent transitions.

**Input artefacts:** `outputs/phase0/layer_gene_embeddings.npy` (12, 1500, 1232). **New computation:** < 5 min.

**Expected signal if true:** Spike in displacement magnitude at L0→L1 (expected from H03 PPI drop) and at L7→L8 (LID non-monotonicity). Low displacement at L3→L4 through L6→L7 (stable compression phase).

**Null/control:** Layer-permuted displacement vectors; test whether real profile is non-uniform beyond chance.

**Value:** medium | **Cost:** very low

---

### H-L — Cross-model per-gene LID comparison: MaxToki L0 vs Geneformer static

**Hypothesis:** Geneformer V2-316M static embeddings, despite different architecture, show the same TF/target LID asymmetry as MaxToki L0 — i.e., the lower LID of TF genes is a convergent property of gene embedding in single-cell foundation models, not MaxToki-specific.

**Test design:** Compute per-gene Levina-Bickel LID (k=20, cosine) on Geneformer-V2-316M static embeddings for the same 1500 HVGs. Compare group-level median LID (TF_only, target_only, string_member) to MaxToki L0 values. Test TF_only vs target_only Mann-Whitney. If MaxToki-Geneformer LID ordering is the same (TF < target), this is a convergent cross-architecture finding.

**Input artefacts:** `outputs/phase9b/` (Geneformer static embeddings for 1500 HVGs). **New computation:** < 5 min.

**Expected signal if true:** Geneformer also shows TF_only LID < target_only LID. The TF geometric compactness is a biological property encoded convergently.

**Null/control:** Gene-group label shuffle on Geneformer embeddings; compare observed group difference to null.

**Value:** high | **Cost:** very low

---

## Top 3 for Immediate Execution (iter_0003)

### Slot 1 — High-Probability Discovery Candidate

**H-B: STRING-persistent kNN pairs across layers**

This is the most direct extension of H03's headline finding (L0 z=23). The pair-level decomposition will reveal what MaxToki's token embeddings encode vs what contextual processing retains. Expected to produce clean, interpretable results (experimental pairs persist vs coexpression pairs are discarded). Low compute, high biological payoff.

Combine with H-C (decay rate vs confidence tier) in the same script — they share the pair adjacency matrix, so both are computed in one pass.

### Slot 2 — High-Risk / High-Reward Candidate

**H-G: Cell-type centroid trajectory (pseudotime encoding test)**

If MaxToki's depth-wise compression encodes a pseudotime coordinate rather than generic rank reduction, this will appear as a biologically ordered centroid trajectory. This would reframe ALL prior results (the "non-monotone at L7-L9" structure would become interpretable as a pseudotime regime boundary). Failure is informative too: if centroids move randomly across layers, the trajectory-encoding hypothesis is dead and we should treat MaxToki as a bidirectional-equivalent encoder for gene geometry purposes.

### Slot 3 — Cheap Broad-Screen Candidate

**H-D + H-E + H-L combined (hub degree, LID confound regression, cross-model LID)**

Three ultra-cheap analyses using already-available artefacts:
- H-D: Does STRING degree predict L11 LID? (< 20 lines, existing data)
- H-E: Does TF/target LID survive mean_expr + dropout control? (< 30 lines, existing data + gene_features.csv)
- H-L: Do Geneformer static embeddings show the same TF/target LID asymmetry? (< 30 lines, Geneformer static from phase9b)

These can run as a single script in under 15 minutes. They either confirm H02 as a clean biological finding or retire it as a technical confound.

---

## Secondary priority (iter_0004 candidates)

- H-A (L11 cluster GO annotation) — requires H01 rerun with gene tracking; do after H-B establishes the pair persistence map.
- H-H (non-monotone LID robustness) — quick sanity check, can run any time.
- H-J (L8 attention head attribution) — high value but requires new extraction; schedule after H-G.
- H-K (layer transition sharpness) — very cheap, can add to any executor run as a freebie.
- H-I (signed FFL geometry) — medium value, do after understanding H-G pseudotime structure.
- H-F (cross-architecture conserved pairs) — do after H-B pair adjacency map is built.
