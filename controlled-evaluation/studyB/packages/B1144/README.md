# Cell-cycle phase readout: transfer from K562 to RPE1

## What this analysis does

This is one step of a project that studies how the text-based single-cell model
C2S-Scale-Gemma-2-2B (a Gemma-2 language model trained on "cell sentences") represents
the cell cycle. Earlier steps, not part of this package, showed that a cell's
cell-cycle phase can be read out from the model's activations within one dataset.

This step asks whether that readout carries over to a second cell line without
refitting:

    fit a phase readout on K562 cells  ->  apply it, unchanged, to RPE1 cells

The same is done for a readout built on plain gene expression, so each readout has a
point of reference.

- **Phase label (ground truth).** For each dataset separately, take the Tirosh/Scanpy
  S-phase and G2/M marker genes, run PCA on them, and use the angle in the PC1-PC2
  plane as the cell's phase (the tricycle/Revelio construction). The sign and origin of
  this angle are arbitrary after PCA, so they are set from the markers: S must come
  before G2/M, and the S peak is placed at 45 degrees. This gives both datasets the same
  phase convention. Code: `code/cc_phase.py`.
- **Readouts.** Each is a ridge regression (alpha = 1000, features standardised) from
  the features to (cos phase, sin phase); the predicted phase is atan2 of the two
  outputs. Both are fit on K562 only.
  - *model*: C2S-Scale-Gemma-2-2B activations at layer 21 (2,304 numbers per cell).
  - *expression*: log-normalised expression of the 6,544 genes present in both datasets.
- **Scores.** Each readout is scored on held-out K562 cells (5-fold cross-validation,
  "within") and on all RPE1 cells with the K562 fit frozen ("transfer").
  Retention = transfer / within. The script then applies a pass/fail rule to decide
  whether each readout transferred. The statistics and the rule are defined in
  `code/transfer_test.py`.

RPE1 was chosen as the target because it is a single cell type (so phase cannot be
mixed up with cell identity) and 93 of the 94 marker genes are in its panel. Both cell
lines come from the same lab and sequencing platform, so this tests transfer across
cell lines, not across batches or platforms.

## Data

- `data/k562/k562_controls.h5ad` - 3,000 K562 cells, 6,546 genes. Non-targeting
  control cells from the Replogle et al. Perturb-seq screen. `X` holds log-normalised
  expression (sparse, float32); gene symbols are the `var` index.
- `data/rpe1/rpe1_controls.h5ad` - 3,000 RPE1 non-targeting control cells from the
  same screen series, 8,749 genes, same format.
- `data/<cell line>/activations/layer_21_activations.npy` - float32 array of shape
  (3000, 2304). For each cell, the model was given a cell sentence (the cell's expressed
  genes in order of decreasing expression, top 512, with the standard C2S prompt), and the
  residual stream after transformer block 21 was taken at the last token.
  `row_cell_ids.npy` gives, for each row, the matching row of the h5ad file.
  `manifest.json` records the extraction settings.

## Code

- `code/cc_phase.py` - marker gene sets, module scores and the oriented phase angle.
- `code/transfer_test.py` - builds both readouts, scores them and writes the results.

## How to run

From the package root:

    python code/transfer_test.py

This writes `outputs/transfer_rpe1.json`. To write somewhere else, add
`--out <file.json>`. It needs numpy, scipy, scikit-learn and anndata, runs on CPU in
one to three minutes and uses about 1.5 GB of memory.

## Outputs

- `outputs/transfer_rpe1.json` - gene and cell counts, and for each readout
  (`model`, `expression`) the within and transfer scores, the median angular errors
  in degrees and the retention, plus the pass/fail flags and the verdict string.
- `outputs/run_log.txt` - console output of the run that produced the JSON.

`SUMMARY.md` is the analyst's write-up of the results.
