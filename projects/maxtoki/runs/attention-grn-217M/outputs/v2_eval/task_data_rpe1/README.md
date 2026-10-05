# Task data: MaxToki-217M attention on RPE1 control cells

This folder holds inputs only. It contains no evaluation results.

## Files

| File | What it is |
|---|---|
| `attention_layer8_headmean.npy` | float32 array, shape (1500, 1500). Attention score for every ordered gene pair. Row i = query gene, column j = key gene. Gene order = `genes.tsv`. |
| `pair_cell_counts.npy` | int32 array, shape (1500, 1500). Number of cells in which both genes were present as tokens. The attention score is an average over these cells. |
| `genes.tsv` | The 1,500 genes: `index` (row/column in the arrays), gene `symbol`, `ensembl_id`, `maxtoki_token_id`. |
| `gene_stats_control_cells.tsv` | Per-gene statistics over the same 2,000 control cells: `mean` and `variance` of log1p(CP10k) expression, and `dropout_rate` (fraction of cells with zero count). |
| `trrust_edges_in_gene_set.tsv` | TRRUST v2 (human) regulator-target rows where both the TF and the target are among the 1,500 genes. Columns: `tf`, `target`, `mode`, `pmid` as in the source file, plus `tf_index`, `target_index`. Rows are copied as they are in the source file: one pair can appear on several rows (different mode or PMID), and a TF can be listed as its own target. |
| `MANIFEST.sha256` | sha256 of every file above. |

## How the data were made

1. **Cells.** Replogle et al. 2022 genome-wide Perturb-seq screen in RPE1 cells
   (`ReplogleWeissman2022_rpe1.h5ad`). 2,000 cells were drawn at random (numpy
   `default_rng(42)`, without replacement) from the 11,485 cells labelled `control`
   (non-targeting guides). No perturbed cells are used in this folder.
2. **Normalisation for gene statistics and gene choice.** Each cell was scaled to
   10,000 total counts over all 8,749 measured genes, then log1p. Mean and variance
   (population variance, ddof = 0) are over the 2,000 cells. Dropout rate uses the raw counts.
3. **Gene set.** Genes present in the MaxToki vocabulary (Ensembl IDs) were ranked by
   variance of log1p(CP10k) over the 2,000 cells; the top 1,500 were kept, then put in
   the order of the original h5ad columns.
4. **Model input.** Each cell was tokenised with MaxToki's rank-value encoding: every
   gene with a non-zero count (in the vocabulary, all measured genes, not only the 1,500)
   was divided by that gene's median (Geneformer `gene_median_dictionary_gc104M.pkl`),
   genes were sorted from highest to lowest normalised value, the list was cut at 2,046
   genes, and `<bos>` / `<eos>` tokens were added (mean length about 2,039 tokens).
5. **Model.** MaxToki-217M (Hugging Face safetensors, `LlamaForCausalLM`, 11 layers,
   8 attention heads, hidden size 1,232), float32, eager attention, one forward pass per
   cell. The model is causal: each token attends only to itself and to earlier tokens.
   Tokens are ordered by normalised expression, so in a given cell the attention from
   gene i to gene j can be non-zero only if gene j came at or before gene i.
6. **Aggregation.** Only layer index 8 (0-based, the 9th of 11 layers) is included here.
   For each head h and each cell, the attention weight from the token of gene i (query)
   to the token of gene j (key) was added to a running sum, for every pair of the 1,500
   genes that were both present in that cell. Each sum was divided by
   `pair_cell_counts[i, j]`. Pairs never seen together have score 0. The 8 heads were
   then averaged with equal weight. The diagonal holds each gene's attention to its own
   token; it is kept as stored.

The exact code is `runs/attention-grn-217M/scripts/phase0_rpe1.py` of the source project
(layer 8 of `attention_edges_layer_mean.npy`). The TRRUST file is the one used by the
source project (`trrust_human.tsv`, 9,396 rows).
