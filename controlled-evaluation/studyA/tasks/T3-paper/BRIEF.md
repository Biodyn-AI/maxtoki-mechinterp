# Task brief

## 1. Question

**Are the sparse-autoencoder features that respond to a transcription factor's knockdown specific to that
factor's target genes?**

Answer with exactly one verdict:

- `supported`
- `not supported`
- `inconclusive`

The question is about transcription factors (TFs) in general, across all knocked-down TFs in the data. GATA1
is a well-studied example that you may wish to look at.

## 2. Background

**Model.** MaxToki-217M is a single-cell foundation model: a decoder-only transformer (Llama architecture,
11 blocks, hidden size 1,232). A cell is given to the model as a sequence of gene tokens ordered by the
gene's expression rank in that cell (highest first), between a `<bos>` and an `<eos>` token, at most 2,048
tokens.

**Sparse autoencoder (SAE).** A TopK SAE (4,928 features, k = 32) was trained on the model's residual-stream
activations entering decoder block 5, using 500 K562 non-targeting control cells. Each feature has a list
of its top-20 genes, built from those 500 cells.

**Cells.** K562 cells from the Replogle et al. (2022) genome-scale CRISPRi Perturb-seq screen: 7,500
knockdown cells for 87 TFs (up to 100 cells per TF), and 1,700 non-targeting control cells in three
disjoint random groups (`ref` 400, `pool` 800, and the 500 `catalog` cells used for the SAE and the top-20
lists). Every cell was run through the model and encoded with the SAE. For each cell and feature you get
one number: the feature's mean activation over the cell's gene tokens.

**Target genes.** TRRUST v2 and DoRothEA (all confidence levels, with a flag for DoRothEA's ChIP-seq-supported
pairs), restricted to the 6,324 genes that can appear in a top-20 list.

**Source method.** `methods/source_method_paper.pdf` is the paper that introduced this kind of SAE atlas and
the perturbation-response analysis, applied there to other single-cell foundation models.

## 3. Files

All data are in `data/`. `data/README.md` defines every file and column. In short:

| file | content |
|---|---|
| `data/cell_feature_means.npz` | `mean_gene` (9,200 cells x 4,928 features, float32): per-cell feature values; `rows` (cell ids); `n_tokens` |
| `data/cell_manifest.csv` | one row per cell: id, group (`ref` / `pool` / `kd` / `catalog`), knocked-down TF, barcode, GEM group, UMI count |
| `data/feature_top20.tsv` | top-20 genes of each feature, with the activation values used to rank them |
| `data/feature_info.tsv` | per-feature summary of the top-20 lists |
| `data/gene_universe.tsv` | the 6,324 genes, with detection count in the 500 catalog cells, gene length and chromosome |
| `data/targets_trrust.tsv` | TRRUST v2 TF-target pairs (target in the gene universe) |
| `data/targets_dorothea.tsv` | DoRothEA TF-target pairs (target in the gene universe), with confidence level and ChIP-seq flag |
| `data/tf_info.tsv` | per TF: number of knockdown cells and how much of the TF's own mRNA remains, where measured |

## 4. What to deliver

Return two things.

**(a) A report in Markdown** with these sections, in this order:

1. **Question**
2. **Data** (what you used)
3. **Methods**
4. **Results**: give every estimate with its uncertainty (for example a 95% confidence interval) and say
   what the unit of resampling or replication is.
5. **Verdict**: one of `supported` / `not supported` / `inconclusive`, plus 2 to 4 sentences of
   justification.
6. **Limitations**

**(b) A results object** (JSON) with these fields:

- `verdict`: one of `supported`, `not supported`, `inconclusive`
- `estimates`: a list of objects `{name, value, ci_low, ci_high, method_of_uncertainty}`
- `analysis_script`: a complete, self-contained Python script. Run from the package root with
  `<EVAL_ROOT>/bin/python`, it must recompute every number you
  report from the files in `data/` and print them as JSON.

## 5. Limits

- Use Python: `<EVAL_ROOT>/bin/python` (numpy, pandas, scipy,
  scikit-learn, statsmodels and torch are installed). CPU only.
- Each command must finish within 9 minutes.
- Aim to finish within about 2 hours of work.
- Do not write files. Run Python inline (for example `python - <<'EOF' ... EOF`).
- Read only inside this package folder.
