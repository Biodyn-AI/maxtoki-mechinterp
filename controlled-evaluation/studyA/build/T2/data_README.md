# Data: MaxToki-217M SAE circuit edges and K562 CRISPRi knockdown responses

All paths below are relative to this `data/` folder. No model is needed to use these files.
Gene symbols are upper case in every file.

## Files

| File | What it is |
|---|---|
| `circuit_edges.csv` | 272,905 circuit edges, one row per (source feature, target feature). Columns: `src_layer`, `src_feature` (site and feature id of the source feature), `tgt_layer`, `tgt_feature` (site and feature id of the target feature), `cohens_d`, `consistency`, `sign` (`inhibitory` or `excitatory`). `cohens_d` and `consistency` are rounded to 4 decimals. |
| `source_features.tsv` | The 120 source features. Columns: `layer` (site), `feature` (feature id), `top10_genes` (comma-separated, highest first). |
| `feature_top_genes.tsv` | The top-10 genes of every feature that appears in `circuit_edges.csv`, as source or target (46,705 features). Columns: `layer`, `feature`, `top10_genes` (comma-separated, highest first). |
| `silenced_genes.tsv` | The 227 genes silenced by CRISPRi that appear in `gene_pairs.parquet`. Columns: `silenced_gene`, `n_crispri_cells` (number of K562 cells carrying a guide against the gene). |
| `knockdown_lfc.npz` | NumPy archive. `lfc`: float32, shape (227, 6546), the log-fold change of every measured gene (column) after silencing each gene (row). `silenced_genes`: (227,) row order, same order as `silenced_genes.tsv`. `measured_genes`: (6546,) column order. `n_crispri_cells`: (227,) cells per silenced gene. `control_mean`: (6546,) mean log1p(counts per 10,000) of each measured gene in the control cells. Load with `numpy.load`. |
| `gene_pairs.parquet` | 698,624 (silenced gene, target gene) pairs made from the circuit edges by the method's rule (step 8 below), with the observed change. Columns are listed below. |
| `MANIFEST.sha256` | sha256 of every file above. |

### `gene_pairs.parquet` columns

| Column | Meaning |
|---|---|
| `silenced_gene` | The gene silenced by CRISPRi. In the circuit, it is a top gene of the source feature. |
| `target_gene` | The gene whose change is measured. In the circuit, it is a top gene of the target feature. |
| `evidence` | Number of (edge, source gene, target gene) combinations that support this pair. |
| `n_inhibitory_evidence` | How many of those come from `inhibitory` edges. |
| `max_abs_d` | Largest \|Cohen's d\| among them (values as in `circuit_edges.csv`). |
| `predicted_decrease` | The method's prediction: `n_inhibitory_evidence / evidence > 0.5`. True = silencing `silenced_gene` lowers `target_gene`; False = it raises `target_gene`. |
| `lfc` | Observed log-fold change of `target_gene` after silencing `silenced_gene` (same value as in `knockdown_lfc.npz`). |

Rows are sorted by `silenced_gene`, then `target_gene`. There are 6,324 distinct target genes.

## How the data were made

1. **Model.** MaxToki-217M (Hugging Face safetensors, `LlamaForCausalLM`, 11 layers, 8 attention
   heads, hidden size 1,232), float32. A cell is tokenised with rank-value encoding: every gene
   with a non-zero count is divided by that gene's median in a large reference corpus, genes are
   sorted from highest to lowest value, the list is cut at 2,046 genes, and `<bos>` / `<eos>` tokens
   are added (at most 2,048 tokens). Each cell is given to the model on its own.
2. **Sparse autoencoders (SAEs).** Twelve TopK SAEs, one per site, each with 4,928 features and
   k = 32 (each token position keeps its 32 largest feature activations). Site l (0 to 10) is the
   residual stream at the input of block l. Site 11 is the final normalised hidden state, the input
   of the output head. Each SAE was trained on the residual-stream vectors at all token positions of
   500 K562 non-targeting control cells from the Replogle et al. 2022 screen.
3. **Top genes of a feature.** Over the same 500 cells, for each feature and each gene, the
   feature's mean activation at the gene's token positions (positions where the feature is active).
   Genes are ranked by this mean. The first 10 are given here. The entry `<SPECIAL>` stands for the
   `<bos>` / `<eos>` tokens; it is not a gene.
4. **Source features.** At each of the sites 0, 3, 6 and 9, the 30 features with the strongest
   gene-set enrichment of their top-20 genes (largest sum of -log10 p over their significant GO
   Biological Process, KEGG, Reactome and TRRUST terms). 120 source features in all.
5. **Cells for tracing.** 200 K562 non-targeting control cells from the same screen, drawn at random
   (numpy `default_rng(42)`, without replacement).
6. **Removing a source feature.** For each cell and each source feature f at site s: the edit is the
   change in the SAE reconstruction when f's activation is set to zero (that is, minus f's activation
   times f's decoder vector). It was added to the input of block s at every token position, including
   `<bos>` and `<eos>`, and the model was run on from block s. At every later site t (s < t <= 11),
   every SAE feature's activation was computed with and without the edit. The change (with edit minus
   without) was averaged over the token positions of the cell.
7. **Edges.** For each (source feature, target feature) pair, over the 200 cells:
   `cohens_d` = mean change / SD of the change (SD with ddof = 1);
   `consistency` = max(n_pos, 200 - n_pos) / 200, where n_pos is the number of cells with a change
   greater than 0. A pair is an edge if |d| > 0.5 and consistency > 0.7 (applied before rounding).
   `sign` = `inhibitory` if the mean change is negative (removing the source feature lowers the
   target feature), `excitatory` otherwise. 18 of the 120 source features have no edge.
8. **Gene pairs (the method's rule).** The method turns edges into gene-level predictions as follows.
   For each edge, every top-10 gene of the source feature is paired with every top-10 gene of the
   target feature (a gene is never paired with itself). For each (source gene, target gene) pair:
   `evidence` = number of such (edge, source gene, target gene) combinations;
   `n_inhibitory_evidence` = how many of them come from inhibitory edges;
   `max_abs_d` = the largest |d| among them. A pair is kept if `evidence` >= 2 or `max_abs_d` > 2.0.
   The prediction is `predicted_decrease` = `n_inhibitory_evidence / evidence` > 0.5 (exactly 0.5
   gives False). The method reads True as "silencing the source gene lowers the target gene" and
   False as "silencing the source gene raises the target gene". Only pairs whose source gene is in
   `silenced_genes.tsv` and whose target gene is measured are kept.
9. **Knockdown responses.** K562 cells from the Replogle et al. 2022 CRISPRi Perturb-seq screen
   (a combined file with 6,546 measured genes). Each cell was scaled to 10,000 total counts over the
   measured genes, then log1p. Control cells: 3,000 K562 non-targeting cells drawn at random (numpy
   `default_rng(42)`, without replacement). For a silenced gene g and a measured gene t:
   `lfc[g, t]` = mean over the K562 cells carrying a guide against g, minus mean over the control
   cells. No significance test and no minimum effect size were applied. The silenced genes are the
   top-10 genes of source features that have a CRISPRi guide with at least 10 K562 cells and that
   appear in at least one gene pair (227 genes, 32 to 575 cells each). Every silenced gene is also a
   measured gene, so its own `lfc` is in its row of `knockdown_lfc.npz`.
