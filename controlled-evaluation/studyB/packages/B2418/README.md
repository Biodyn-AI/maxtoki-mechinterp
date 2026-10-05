# Cell-embedding geometry in three single-cell foundation models (fetal gut epithelium)

## Question

The same cells are passed through three frozen single-cell foundation models. How compact is the
cloud of cell embeddings each model produces? "Compact" is measured four ways:

| measure | what it computes | more compact = |
|---|---|---|
| participation ratio (PR) | (sum of covariance eigenvalues)^2 / sum of squared eigenvalues; an effective number of dimensions | lower |
| PC1 share | fraction of total variance on the first principal component | higher |
| d90 | number of principal components needed to reach 90% of the variance | lower |
| norm CV | standard deviation / mean of the per-cell L2 norm | lower |

All four use the full covariance spectrum of the mean-centred embeddings. The same statistics are
also computed on 10 random halves of the cells (seed 0) to show how stable they are.

## Data

- Cells: 4,269 human fetal gut epithelial cells from the CELLxGENE "Developing Human Gut" dataset
  (6-11 post-conception weeks). Seven cell types (stem cell, progenitor cell, enterocyte, colon
  epithelial cell, intestine goblet cell, enteroendocrine cell, epithelial cell), at most 800 cells
  per type (random subsample, seed 0). Raw UMI counts, 25,770 genes (human gene symbols).
- A diffusion pseudotime rooted in the stem/progenitor cells is stored with each cell
  (field name `palantir_pseudotime`, kept for format compatibility). The geometry analysis does not
  use it.
- Models (frozen, public weights):
  - scGPT whole-human (12 layers, width 512)
  - Geneformer V2-316M (18 layers, width 1152)
  - STATE SE-600M (16 layers, width 2048)
- Embedding per cell: the output of transformer layer 11 (0-indexed), mean-pooled over the
  positions of the cell's expressed genes. The same 4,269 cells, in the same order, for every model.

## Files

| path | contents |
|---|---|
| `code/geometry.py` | the analysis (numpy only) |
| `code/extraction/extract_scgpt.py` | makes `data/embeddings/scgpt_gut.npz` |
| `code/extraction/extract_geneformer.py` | makes `data/embeddings/geneformer_gut.npz` |
| `code/extraction/extract_state.py`, `state_loader.py` | make `data/embeddings/state_gut.npz` |
| `data/embeddings/<model>_gut.npz` | `emb` (4269, width) float32; `pseudotime` (4269,); `clusters` (4269,) cell type; `cell_idx` (4269,) row in the count matrix |
| `data/raw/gut_epithelium_first200.h5ad` | the first 200 rows of the 4,269-cell count matrix, same layout as the full file |
| `models/scgpt_whole_human/` | `args.json` and `vocab.json` shipped with the scGPT checkpoint |
| `models/geneformer_v2_316m/config.json` | Hugging Face config of Geneformer V2-316M |
| `models/state_se600m/config.yaml` | config shipped with STATE SE-600M |
| `outputs/` | results of `code/geometry.py` (see below) |
| `SUMMARY.md` | the analyst's write-up of the results |

`state_gut.npz` stores the cell types as category codes `0`-`6` (alphabetical order of the names
above); the other two files store the names. The geometry analysis does not use the labels.

## How to run

From the package root:

```
python code/geometry.py
```

CPU only (the script uses two BLAS threads), about 20-60 seconds, about 0.5 GB of memory. It writes:

- `outputs/geometry.json`: all statistics per model, the random-half ranges, and the model order
  for each measure;
- `outputs/geometry_table.tsv`: the same numbers as a table;
- `outputs/run_log.txt`: the printed table.

## Extraction step (not runnable here)

The embeddings in `data/embeddings/` were made by running the three scripts in `code/extraction/`
on the full 4,269-cell count matrix (`data/raw/gut_epithelium_counts.h5ad`, 139 MB, not included):

```
python code/extraction/extract_scgpt.py
python code/extraction/extract_geneformer.py
python code/extraction/extract_state.py
```

These need model weights that are not in this package: the scGPT whole-human `best_model.pt`
(about 205 MB) plus the scGPT source tree; the Geneformer V2-316M weights (Hugging Face hub,
`ctheodoris/Geneformer`) plus its token, median and name-to-id dictionaries; and the STATE SE-600M
`model.safetensors` and `protein_embeddings.pt` (about 3.3 GB) plus the `arc-state` package. The
200-cell file in `data/raw/` shows the exact input format the scripts read (CSR counts in `X`,
gene symbols in `var/index`, cell type codes in `obs/clusters` with names in
`obs/__categories/clusters`). Rows 0-199 of each `.npz` correspond to its 200 cells.
