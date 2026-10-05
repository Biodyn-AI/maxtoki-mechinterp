# Predicting the genetic-interaction term in Norman 2019 double perturbations

## What the analysis asks

Norman et al. (2019, Science) used CRISPR activation in K562 cells to switch on single genes and pairs
of genes, and measured the result by single-cell RNA-seq. For a pair A_B and a measured gene h, the
genetic-interaction term is

    I_h = dAB_h - (dA_h + dB_h)

where dX_h is the mean log-normalised expression of gene h in cells carrying perturbation X, minus
its mean in control cells. If the double were exactly the sum of the two singles, I would be zero.

The analysis asks how much of I can be predicted from the two single responses alone (dA_h, dB_h),
using ridge regression with a few simple feature sets. Each (pair, gene) is one example. Whole pairs
are held out in 5-fold cross-validation. Performance is the per-pair correlation between predicted
and measured I across genes, averaged over pairs, with a bootstrap 95% interval over pairs. It is
also reported as a share of a ceiling. The ceiling is the split-half reliability of the I profile,
Spearman-Brown corrected to the full data, then square-rooted.

Feature sets, in increasing order of what they may use:

| name | features |
|---|---|
| additive | none: predicts I = 0 |
| sum-only | dA + dB |
| saturation | dA + dB, dA*dB, min, max, abs(dA)+abs(dB), sign agreement, (dA-dB)^2 |
| saturation+gene | the saturation features plus a per-gene mean of I over the training pairs |

## Data

`data/norman2019_subset.npz` is a subset of the scPerturb release of the Norman 2019 data
(file `NormanWeissman2019_filtered.h5ad`, 111,445 cells, 33,694 genes). It holds:

| key | shape | meaning |
|---|---|---|
| `counts` | 63,360 x 1,000, uint16 | raw UMI counts |
| `totals` | 63,360, float64 | each cell's total UMI count over all 33,694 genes |
| `labels` | 63,360, str | `control`, a single gene name (e.g. `CEBPA`), or a pair `A_B` |
| `genes` | 1,000, str | gene symbols for the columns of `counts` |

How the subset was drawn:

- Genes: the 1,000 genes with the highest mean log1p(CP10k) expression over all 111,445 cells.
- Cells: a random draw with a fixed seed of up to 400 cells per single perturbation, up to 200
  cells per double perturbation, and 2,000 control cells. Groups smaller than the cap keep all
  their cells.
- `totals` come from the full transcriptome, so CP10k normalisation matches normalising the full
  file.

The subset has 105 single perturbations (113 to 400 cells each) and 131 double perturbations. The
code keeps the 115 doubles with at least 120 cells, and both of their singles are present.

## Code

`code/learn_interaction.py` does the whole analysis:

1. Normalise counts to CP10k with `totals`, then log1p.
2. Compute the control mean, each single's response dA, and each double's response dAB.
3. Build I for each kept pair, and the split-half ceiling.
4. Fit the four models with 5-fold cross-validation over pairs, and score them.

## How to run

From the package root (CPU only, about 10 seconds, under 1 GB of memory; needs numpy and
scikit-learn):

    python code/learn_interaction.py

Defaults: `--data data/norman2019_subset.npz --out outputs/results.json --top-genes 1000
--min-single 60 --min-double 120 --folds 5`.

## Outputs

- `outputs/results.json`: for each model, `corr` (mean per-pair correlation), `lo`/`hi` (bootstrap
  95% interval), `var_explained` (pooled), `n` (pairs scored), `frac_of_ceiling`; and under
  `_ceiling`: `split_half`, `full` (Spearman-Brown), `ceiling` (its square root), and counts.
- `outputs/run_log.txt`: the printed log of the run that produced `results.json`.

`SUMMARY.md` gives the results and conclusions.
