# Cell-cycle phase in scGPT and Geneformer embeddings, and in expression (K562 control cells)

## Question

Each cell in a growing population sits somewhere on the cell cycle. That position can be written as an angle
(the phase). For two single-cell foundation models and for plain expression, this analysis asks:

1. How well can the phase be read out of the cell's representation, with a linear readout and with a
   k-nearest-neighbour (kNN) readout?
2. Does the loop that the cells trace in the representation fit a flat circle (a loop lying in one linear
   2-plane), or does it leave that plane more than a flat circle with the same noise would?

## Data (`data/`)

Cells: 3,000 non-targeting control cells of the Replogle et al. 2022 K562 CRISPRi Perturb-seq data (no gene
knocked down). Each file holds the cells' representation (`emb`) plus the phase label of the same cells.

| file | representation | cells | width |
|---|---|---|---|
| `scgpt_k562.npz` | scGPT (whole-human checkpoint), layer 11 of 12, mean over the cell's gene tokens. Input: the cell's expressed genes, top 1,200 by expression, values put into 51 quantile bins (scGPT's own binned input). | 3,000 | 512 |
| `geneformer_k562.npz` | Geneformer V2-316M, layer 11 of 18, mean over the cell's gene tokens (rank-based input). | 2,000 | 1,152 |
| `expr_k562.npz` | No model. log(1 + counts per 10,000) of all 6,546 genes in the source file, float32. Also holds `genes`. | 3,000 | 6,546 |

The 2,000 Geneformer cells are a subset of the 3,000 (match them with `cell_idx`, the cell's row in the
source data). The scGPT and expression files hold the same 3,000 cells in the same order. The embeddings were
computed once, earlier; the model weights are not in this package.

Fields in every file: `emb` (cells x width, float32), `phi` (phase angle in radians, in [-pi, pi)),
`s_score`, `g2m_score`, `cc_phase` ("G1", "S" or "G2M"), `cell_idx`.

**How the phase label is made.** The phase is defined from marker-gene expression only, not from any model
(see `code/cc_common.py`, `phase_label`). S score and G2M score are the mean z-scored expression of the
Tirosh et al. 2016 S-phase genes (37 of 43 found) and G2/M genes (50 of 54 found). The discrete call is G1 if
both scores are below 0, else S or G2M by the larger score. The angle `phi` is atan2 of the first two principal
components of the z-scored S + G2M marker genes, turned so that it runs G1 -> S -> G2M and rotated so that the
G1 cells sit at 0. The Geneformer file's label was computed on its own 2,000 cells (its z-scores and PCA use
only those cells). `code/phase_check.py` recomputes every stored label from `data/expr_k562.npz`.

## Method (`code/cc_geometry.py`)

For one representation:

1. Centre it, keep the top 20 principal components and scale each to unit variance (`prep`).
2. Circular R2 of the phase, 5-fold cross-validated (`circ_r2`). The readout predicts (cos phi, sin phi):
   linear = ridge regression (alpha 10); kNN = k-nearest-neighbour regression (k = 15). The predicted phase
   is atan2 of the two predictions. circ-R2 = 1 - mean(1 - cos(error)) / mean(1 - cos(phi - circular mean)).
   1 is perfect; 0 is no better than always guessing the mean phase.
3. Local direction of the loop in each of 12 equal phase sectors (`loop_tangents`): ridge regression of the
   phase offset from the sector centre on the 20 components; the unit coefficient vector is the direction in
   which the phase grows there.
4. Shape of these 12 directions (`turning_stats`): total turning (sum of angles between neighbouring sectors,
   all the way round), planarity (share of their energy in the best-fit 2-plane) and the mean out-of-plane
   angle (how far the sector directions leave that 2-plane).
5. Flat-circle null (`flat_null`), 20 draws: a synthetic representation of the same width whose phase signal
   lies exactly in one random 2-plane, plus Gaussian noise set by binary search so that its linear circ-R2
   matches the real one. The flat-circle hypothesis is rejected only if the real out-of-plane angle is above
   the 95th percentile of the null draws. One-sided p = (number of null draws >= real + 1) / 21.

`code/cc_summary.py` puts the three results in one table, and also re-scores the circ-R2 of all three
representations on the 2,000 cells they share (Geneformer file's label as the target).

## How to run

From the package root, with the interpreter given to you (CPU only):

```
python code/phase_check.py                        # recompute the phase labels (under 30 s)
python code/cc_geometry.py scgpt      --out DIR   # about 1-1.5 min
python code/cc_geometry.py geneformer --out DIR   # about 1-2.5 min
python code/cc_geometry.py expr       --out DIR   # about 8-12 min in total; split it, see below
python code/cc_summary.py             --out DIR   # needs the three JSON files in DIR; about 1 min
```

Times were measured on a busy machine with one BLAS thread (the default set at the top of `cc_geometry.py`).

`DIR` defaults to `outputs/`. Use any writable folder if the package is read-only. Each null draw is saved in
`DIR/null_draws_<model>.json` as soon as it is done, and a later call reuses the draws already there. So a long
run can be split, for example `python code/cc_geometry.py expr --out DIR --max-new-draws 5`, repeated until all
20 draws exist. Draw `k` uses seed `100 + k`, so a split run gives the same numbers as one call. The decodability
numbers (step 2) are printed within the first minute of each call.

Memory: about 1.1 GB at peak (expr), 0.8 GB or less otherwise. Python packages: numpy, scikit-learn.

## Outputs (`outputs/`)

| file | content |
|---|---|
| `cc_geometry_<model>.json` | circ-R2 (linear, kNN), cells per sector, total turning, planarity, out-of-plane angle, null means and SDs, the 20 null out-of-plane angles, p, rejected or not |
| `cc_geometry_<model>.log` | the same, as printed text |
| `null_draws_<model>.json` | turning statistics of each null draw, by seed |
| `cc_summary.json`, `cc_summary.log` | the three-way table and the shared-cell decodability |
| `phase_check.log` | output of `code/phase_check.py` |

`SUMMARY.md` is the analyst's write-up of these results.
