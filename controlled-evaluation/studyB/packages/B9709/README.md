# Cell-cycle phase decodability: STATE-SE layer 11 next to expression

## Question

Does the layer-11 cell embedding of STATE-SE (the SE-600M state-embedding model) hold the cell-cycle
phase in a form that a linear probe can read? The gene expression of the cells is scored with the same
probe, as a comparator.

## Cells

- Source dataset: a pooled CRISPRi Perturb-seq dataset of essential-gene knockdowns in four human cell
  lines (K562, RPE1, HepG2, Jurkat). 643,413 cells x 6,546 genes. Values are stored as log1p(CP10k),
  with no raw counts. The file itself (about 30 GB) is not included.
- The analysis uses 3,000 non-targeting control cells. They are drawn at random from the 39,165
  non-targeting cells with `numpy.random.RandomState(42)` and sorted by source row
  (`select_controls()` in `code/cc_common.py`). The draw does not filter on cell line, so the cells
  come from all four lines.

## Phase coordinate

- Markers: the Tirosh et al. 2016 S (43) and G2M (54) gene lists; 37 and 50 of them are in the gene list.
- Each marker is z-scored across the scored cells. The S score and the G2M score are the mean z-scores.
  A cell is called G1 if both scores are below 0, otherwise S or G2M by the larger score.
- phi = atan2 of the first two principal components of the z-scored markers. It is oriented so that phi
  increases G1 -> S -> G2M, and rotated so that the G1 mean sits at 0.
- Phase is scored separately for each representation, on that representation's cells
  (`load_representation()` and `run_one()` in `code/decode_phase.py`).

## Probe

- Each representation is centred, reduced to 20 principal components and whitened.
- 5-fold ridge regression (alpha = 10) predicts (cos phi, sin phi). The predicted angle is atan2 of the two
  predictions.
- Score: circular R^2 = 1 - mean(1 - cos(error)) / mean(1 - cos(phi - circular mean of phi)).
  1 is perfect. 0 is no better than giving every cell the mean phase.
- A kNN regressor (k = 15) on the same 20 components is reported next to it.

## Data

| file | contents |
|---|---|
| `data/cells.npz` | CRISPRi target gene (`target_categories`, `target_codes`) and cell line (`line_categories`, `line_codes`) for all 643,413 source rows, in source-row order |
| `data/markers.npz` | stored log1p(CP10k) expression of the 87 marker genes found (`genes`) for all 643,413 source rows (`rows`, in source-row order), as `X` (643,413 x 87) |
| `data/expression_controls.npz` | stored log1p(CP10k) expression of the 3,000 selected cells, all 6,546 genes, as a CSR matrix (`data`, `indices`, `indptr`, `shape`), plus `genes` and `cell_idx`. Written by `code/extraction/build_inputs.py` |
| `data/state_se_L11.npz` | STATE-SE layer-11 embeddings: `emb` (3,000 x 2,048, float32) and `cell_idx`. `code/extraction/build_inputs.py` writes the 3,000 selected cells to an h5ad, and `code/extraction/extract_state.py` runs the model on it |

STATE-SE details: layer 11 of 16, d = 2,048. The cell vector is the mean of the layer-11 residual stream
over the non-CLS positions of expressed genes. 6,369 of the 6,546 genes are in the model's
protein-embedding vocabulary. STATE-SE expects raw counts. The source holds only log1p(CP10k) values, so
the model was given log1p values.

## How to run

From the package root:

```
python code/decode_phase.py
```

CPU only, about 15 seconds on an idle machine, about 1 GB of memory (numpy 1.26.4, scipy, scikit-learn 1.8.0). It writes
`outputs/decodability.json` and `outputs/run_log.txt`. The scripts in `code/extraction/` are kept for
reference. They need the full source dataset and the STATE-SE model, which are not included.

## Outputs

- `outputs/decodability.json`: per representation, the number of cells, input dimension, PCA dimension,
  markers found, discrete phase counts, linear circ-R^2 and kNN circ-R^2.
- `outputs/run_log.txt`: the printed log of the run.
- `SUMMARY.md`: the write-up of the results.
