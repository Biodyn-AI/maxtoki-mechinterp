# Task brief

## 1. Question

Do MaxToki-217M's attention maps encode regulator-to-target relationships? Use the
TRRUST curated network as ground truth.

Give exactly one verdict:

- **supported**
- **not supported**
- **inconclusive**

## 2. Background

**The model.** MaxToki-217M is a transformer model of single-cell gene expression
(Gómez Ortega & Theodoris, bioRxiv 2026). It uses the Llama decoder architecture:
11 layers, 8 attention heads per layer, hidden size 1,232, about 217 million parameters.
It was trained on about 175 million human single-cell transcriptomes, first on single
cells and then on sequences of cell states. A cell is given to the model as a sequence
of gene tokens. Each gene's count is divided by that gene's median across a large
reference corpus, and genes are put in order from highest to lowest value (rank-value
encoding). The model is causal: each token attends only to itself and to tokens that
come before it. Here each cell was given to the model on its own.

**The cells.** 2,000 unperturbed control cells (non-targeting guides) of the RPE1 cell
line, drawn at random from the Replogle et al. 2022 genome-wide Perturb-seq screen.

**The genes.** 1,500 genes: the most variable genes in these 2,000 cells (variance of
log1p of counts per 10,000) among the genes in the model's vocabulary.

**The attention scores.** Every cell was run through the model once. For layer index 8
(0-based; the 9th of 11 layers), the attention weight from the token of gene i (query)
to the token of gene j (key) was averaged over the cells in which both genes were
present, and then over the 8 heads. Attention weights are softmax outputs, so each lies
between 0 and 1. Only this layer is provided. It was chosen before any evaluation, by
its relative depth in the network.

**The reference network.** TRRUST v2 (human; Han et al. 2018) is a database of
transcription factor (TF) to target-gene relationships, curated from the literature.
Only rows where both the TF and the target are among the 1,500 genes are provided.

## 3. Materials

All paths are relative to this package folder.

| Path | What it is |
|---|---|
| `data/attention_layer8_headmean.npy` | float32, shape (1500, 1500). Attention score for every ordered gene pair. Row i = query gene, column j = key gene. Unitless (mean softmax weight). |
| `data/pair_cell_counts.npy` | int32, shape (1500, 1500). Number of cells (out of 2,000) in which both genes were present as tokens; the attention score is an average over these cells. |
| `data/genes.tsv` | 1,500 rows: `index` (row/column in the arrays), `symbol`, `ensembl_id`, `maxtoki_token_id`. |
| `data/gene_stats_control_cells.tsv` | 1,500 rows: per-gene `mean` and `variance` of log1p(counts per 10,000), and `dropout_rate` (fraction of the 2,000 cells with a zero count). |
| `data/trrust_edges_in_gene_set.tsv` | TRRUST rows with both genes in the gene set: `tf`, `target`, `mode`, `pmid`, `tf_index`, `target_index`. |
| `data/README.md` | How each data file was made, step by step. |
| `data/MANIFEST.sha256` | sha256 of the data files. |
| `methods/source_method_paper.pdf` | The source paper for this kind of analysis. It applies the analysis to other single-cell models. |

A method specification for this kind of analysis is provided in contract/SPEC.md; follow it.

## 4. Deliverable

Return two things.

**(a) `report`**: a Markdown report with these sections, in this order:

1. **Question**
2. **Data**
3. **Methods**
4. **Results**: give every estimate with its uncertainty (for example a 95% confidence
   interval) and say what the unit of resampling was.
5. **Verdict**: one of supported / not supported / inconclusive, then 2-4 sentences of
   justification.
6. **Limitations**

**(b) `results`**: a JSON object with these fields:

- `verdict`: one of `"supported"`, `"not supported"`, `"inconclusive"`.
- `estimates`: a list of objects, each with `name`, `value`, `ci_low`, `ci_high`,
  `method_of_uncertainty`.
- `analysis_script`: a complete, self-contained Python script. When it is run from the
  package root with
  `<EVAL_ROOT>/bin/python`, it must
  recompute every number reported in `estimates` from `data/` and print them as JSON.

## 5. Limits

- CPU only. Python with numpy, pandas, scipy and scikit-learn is available at
  `<EVAL_ROOT>/bin/python`.
- Each command must finish within 9 minutes.
- Aim to finish within about 2 hours of work.
- Do not write files. Run Python inline.
- Read only inside this package folder.
