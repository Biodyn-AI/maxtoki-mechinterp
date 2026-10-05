# Cell-cycle phase label: K562 vs RPE1

## What this analysis does

Every cell in two cell lines gets a continuous cell-cycle phase angle (0-360 degrees). The question is
whether the phase label means the same thing in both lines. This is needed before RPE1 can be used as the
target cell line for testing whether a phase readout learned on K562 cells works on a different cell line.

The check is done gene by gene. The gene list is a published set of cell-cycle genes with known peak
phases (Whitfield et al. 2002, synchronised HeLa cells). For each of these genes, the script finds the
phase at which the gene is most expressed, separately in K562 and in RPE1. If the two phase labels mean
the same thing, a gene's peak phase should be about the same in both lines.

## Data (`data/`)

| file | cells | genes | content |
|---|---|---|---|
| `k562_controls.h5ad` | 3,000 | 6,546 | K562 non-targeting control cells (Replogle et al. 2022 Perturb-seq) |
| `rpe1_controls.h5ad` | 3,000 | 8,749 | RPE1 non-targeting control cells (Replogle et al. 2022 Perturb-seq) |

- `X` is a sparse float32 matrix of log(1 + counts per 10,000). In the K562 file the scaling to 10,000 was
  done before the gene panel was cut to 6,546 genes, so per-cell totals of expm1(X) are 7,400-9,600. In the
  RPE1 file they are exactly 10,000.
- Gene names are HGNC symbols. 6,544 genes are in both files.
- No cell annotations are included.

## Code (`code/`)

- `cc_phase.py`
  - `S_GENES`, `G2M_GENES`: the Tirosh et al. 2016 S-phase and G2/M marker lists (as used by Scanpy).
  - `score_genes`: Scanpy-style module score (mean of the set minus the mean of expression-matched control
    genes drawn from the whole panel, seed 0).
  - `phase_angle`: places each cell on a circle. It z-scores the marker genes present in the panel, runs
    PCA (SVD), and takes the angle of each cell in the PC1-PC2 plane.
  - `line_checks`: per-line checks: phase concentration R (0 = cells spread all around the circle, 1 = all
    at one angle), and the angles where the S score and the G2M score peak (circular mean of the phase over
    the 200 highest-scoring cells).
- `compare_lines.py`: runs the whole analysis and writes the outputs.

## Method

1. For each line separately: `phase_angle`, then `line_checks`.
2. Peak phase of a gene in one line: weights w = expression minus the gene's mean expression, clipped at 0.
   Peak = angle of sum(w * exp(i * phase)) / sum(w), in degrees.
3. Genes present in both lines are compared with the circular correlation coefficient (Jammalamadaka and
   SenGupta) and with the median and mean absolute angular difference (0-180 degrees).

## How to run

From the package root:

```
python code/compare_lines.py            # writes to outputs/
python code/compare_lines.py --out DIR  # writes to another folder
```

Needs numpy, scipy and anndata. CPU only, about 20 seconds, about 0.6 GB of memory. The result is
deterministic.

## Outputs (`outputs/`)

- `phase_agreement.json`: for each line, the info returned by `phase_angle` plus the `line_checks`
  values; then the comparison: genes compared, genes not compared, circular correlation, median and mean
  absolute difference.
- `gene_peak_phase.csv`: one row per compared gene: peak phase in K562, in RPE1, and the absolute
  difference.
- `run_log.txt`: the printed log.

`SUMMARY.md` is the analyst's write-up of the result.
