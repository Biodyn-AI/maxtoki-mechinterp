# Branch-point curvature in foundation-model cell embeddings: significance test

## What this analysis does

This is one step of a project that compares how single-cell foundation models lay out
differentiation trajectories. For each model and tissue, the question is whether the
cell embedding relates to developmental time in a straight-line way, or whether there is
a curved (nonlinear) part on top.

The curvature statistic for one model x tissue cell is

    curvature = CV R2[degree-2 polynomial kernel ridge] - CV R2[linear kernel ridge]

Both ridges predict a cell's pseudotime from its full embedding (no PCA). A positive value
means the curved decoder predicts pseudotime better than the straight-line one.

This step tests whether each curvature value is significant.

- **Decoders.** Embedding columns are standardised. Linear kernel K = X X^T / d + 1;
  polynomial kernel = K^2 (degree 2, same gamma = 1/d and bias). Kernel ridge with
  5-fold cross-validation (shuffled, seed 0). For each decoder the ridge penalty is
  picked from an 8-value grid (1e-3 to 1e4) as the one with the highest CV R2.
- **Null.** 200 synthetic targets per cell. Each is a random linear function of the
  standardised embedding plus Gaussian noise. The noise is scaled so that the target's
  linear R2 matches the real linear R2 of that cell. So the true relation is linear by
  construction, at the same strength as the real data. The same curvature statistic is
  computed on each target. p = (number of null values >= real + 1) / 201.
- **Multiple testing.** Benjamini-Hochberg across the 6 cells.
- **Bootstrap.** 1,000 resamples of cells from the out-of-fold predictions of the two
  decoders (decoders are not refit), giving a 95% percentile interval for the curvature.

## Data

`data/<model>_<tissue>.npz`, 6 files. Each holds 1,500 cells, a random subset (seed 0)
of the cells in the full extraction.

- `emb` - float32 (1500, d). Per-cell mean over gene tokens of the residual stream after
  transformer layer 11.
  - `geneformer`: Geneformer V2-316M, d = 1152.
  - `scgpt`: scGPT whole-human, d = 512, input values 51-bin quantile-binned as in the
    model's pretraining.
- `pseudotime` - float64 (1500,). Scanpy diffusion pseudotime (dpt), computed on the
  expression data before embedding.
- `cell_type` - string (1500,). Author cell-type label (not used by the code).

Tissues:

- `lung` - human lung airway epithelium (Tabula Sapiens). Root: basal cells.
- `gut` - human fetal intestinal epithelium (Developing Human Gut, 6-11 weeks
  post-conception). Root: stem and progenitor cells.
- `pancreas` - mouse pancreas endocrinogenesis, E15.5 (Bastidas-Ponce et al. 2019).
  Root: ductal cells. Mouse gene symbols were mapped to human by uppercasing.

## Code

- `code/curvature_significance.py` - decoders, null, bootstrap, BH correction, output.

## How to run

From the package root:

    python code/curvature_significance.py

It needs numpy and scikit-learn. It runs on CPU in about 1-3 minutes with one thread and
uses about 0.4 GB of memory.

## Outputs

- `outputs/curvature_significance.json` - settings, and for each cell: number of cells,
  embedding size, linear and polynomial CV R2, curvature, selected penalties, null
  summary (mean, SD, 95th percentile, z, p, BH q) and bootstrap interval.
- `outputs/results_table.tsv` - the same numbers as one row per cell.

`SUMMARY.md` is the analyst's write-up of the results.
