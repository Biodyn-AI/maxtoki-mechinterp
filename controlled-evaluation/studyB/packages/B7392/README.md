# Cell-cycle phase transfer from K562 to RPE1: C2S-Scale layer 21 vs its input encoding

## Question

C2S-Scale-Gemma-2-2B is a text language model (Gemma-2, 2B parameters) trained on "cell sentences".
A cell sentence is a list of a cell's most highly expressed gene names, in descending order of expression.
So the model sees which genes are present and their order, but not their expression values.

This analysis asks one question. A readout of cell-cycle phase is fit on K562 cells and then applied,
unchanged, to RPE1 cells. Does the model's internal state (layer 21) carry the phase across cell lines
better than, worse than, or as well as the expression information the model is given?

## Data (`data/`)

| file | content |
|---|---|
| `k562_expression.npz`, `k562_genes.txt` | 3,000 K562 cells x 6,546 genes. Non-targeting control cells from the Replogle et al. 2022 Perturb-seq screen. Values are log1p of per-cell normalised counts. Stored as a compressed sparse row matrix (`data`, `indices`, `indptr`, `shape`); gene symbols one per line, in column order. |
| `rpe1_expression.npz`, `rpe1_genes.txt` | 3,000 RPE1 cells x 8,749 genes. Non-targeting control cells from the same screen series. Same format and normalisation. |
| `act_k562/layer_21_activations.npy` | 3,000 x 2,304 float32. Residual stream after transformer block 21, at the last token of each cell's prompt ("cell-summary" activations). |
| `act_k562/row_cell_ids.npy` | Row i of the activations belongs to this row of the K562 expression matrix. |
| `act_k562/manifest.json` | Settings of the extraction run. |
| `act_rpe1/...` | The same three files for RPE1. |

The activations were made earlier on a GPU with `code/extract_activations.py` (cell sentences built by
`code/cell_sentences.py`), run once per cell line on the expression matrix in `data/` (stored as `.h5ad` at the
time), with `--layers 21 --max-cells 3000 --max-genes 512 --unit cell-summary --store-dtype float32`.
That step needs the model weights and a GPU. It is not part of the run below.

## Method (`code/transfer.py`)

1. **Phase label.** Each cell gets a cell-cycle phase angle theta from `code/cc_phase.py`: PCA on the
   Tirosh S and G2/M marker genes, angle in the PC1-PC2 plane. The angle is oriented the same way in both
   cell lines (S before G2/M, S peak placed at 45 degrees). It is computed separately in each cell line.
2. **Shared genes.** Genes present in both cell lines (matched by upper-case symbol): 6,544.
3. **Four arms**, each a frozen transfer: standardise features on K562, fit ridge regression
   (alpha = 1000) from features to (cos theta, sin theta) on K562, apply the same scaler and ridge to RPE1,
   and take the predicted angle as atan2 of the two outputs.
   - `expr_full`: all shared genes, expression values.
   - `expr_512_mag`: each cell's top-512 genes, expression values; all other genes 0.
   - `expr_512_rank`: each cell's top-512 genes, rank only (512 for the top gene down to 1); all other
     genes 0. This is meant to carry the same information as the model's 512-gene cell sentence.
   - `model`: layer-21 cell-summary activations.
   Two reference arms: `constant` (every cell gets the mean RPE1 phase) and `random` (uniform angles).
4. **Metrics** on the 3,000 RPE1 cells. R_diff = |mean(exp(i(pred - true)))|: 1 is perfect up to a fixed
   rotation, 0 is no information. Median angular error after the best single rotation, in degrees.
5. **Paired bootstrap.** 5,000 draws of 3,000 RPE1 cells with replacement (seed 0). In each draw every
   arm is scored on the same cells. For each contrast the result is the mean difference in R_diff,
   a 95% percentile interval, and the share of draws above 0.
6. **Verdict rule** on `model - expr_512_rank`: interval above 0 means the model adds information,
   interval below 0 means it loses information, interval containing 0 means parity.

## How to run

From the package root:

```
python code/transfer.py
```

Needs numpy, scipy and scikit-learn. CPU only. About 1 to 2 minutes and up to 2.5 GB of memory.
It writes `outputs/transfer_results.json`. The console output of the run is in `outputs/run_log.txt`.

## Files

- `code/transfer.py` - the analysis.
- `code/cc_phase.py` - cell-cycle phase angle.
- `code/data_io.py` - loads the expression matrices.
- `code/cell_sentences.py`, `code/extract_activations.py` - how the model input and activations were made (not run here).
- `outputs/` - results of running `code/transfer.py` on `data/`.
- `SUMMARY.md` - the analyst's write-up of the results.
