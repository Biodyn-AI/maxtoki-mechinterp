# DoRothEA sign vs CRISPRi knockdown response

## Question

DoRothEA labels many of its transcription factor (TF) to target edges as activating or inhibiting.
This analysis asks whether that label predicts the direction in which the target gene moves when the
TF is knocked down by CRISPRi.

The prediction is simple. Knocking down an activator should lower its target (dE < 0). Knocking down a
repressor should raise it (dE > 0).

## Data (`data/`)

| file | content |
|---|---|
| `k562_tf_knockdown_response.csv.gz` | K562 genome-wide CRISPRi Perturb-seq (Replogle et al. 2022, Cell), "normalized bulk" release. One row per guide-level pseudo-bulk profile (`gene_transcript` = `<library id>_<gene>_<promoter>_<Ensembl id>`). Values are the mean over the profile's cells of each gene's expression z-normalized against non-targeting control cells (the release's own normalization). 672 profiles x 2,746 genes. |
| `rpe1_tf_knockdown_response.csv.gz` | Same for the RPE1 screen of the same study. 74 profiles x 1,301 genes. |
| `k562_guide_info.csv`, `rpe1_guide_info.csv` | Number of cells behind each profile (`num_cells_filtered`), from the same release. |
| `dorothea_omnipath.tsv.gz` | The DoRothEA TF-target table as served by OmniPath (downloaded August 2026), unchanged. `is_stimulation` / `is_inhibition` give the sign; `dorothea_level` gives the confidence level (A highest, E lowest). |

How the two response files were cut from the full release:

- Rows: every profile whose knocked-down gene is the source of at least one signed DoRothEA edge
  (an edge marked as stimulation only, or as inhibition only).
- Columns: the signed targets of those TFs, plus the TFs themselves, among the genes measured in the
  release. Genes with a non-finite value in any profile of the full release were removed first
  (K562: 8,248 to 8,175 genes; RPE1: 8,749 to 8,747). Where a gene name occurs twice in the release,
  the later column is kept.
- Values are copied unchanged (float32).

## Method (`code/sign_test.py`)

1. Profiles that knock down the same gene are averaged, giving one response vector per TF. A TF's
   value in its own gene is set to missing.
2. Each signed DoRothEA edge (TF to target, self-edges dropped; edges marked both ways or neither way
   dropped) is scored if the TF was knocked down and the target was measured with a non-zero response.
3. Edges are split into three strata by response size: all edges, top 50% of |dE|, top 10% of |dE|.
4. Per stratum the script reports:
   - the 2x2 table of database class (activation / inhibition) by observed direction (up / down),
     the odds ratio OR = odds(up given inhibition) / odds(up given activation), and a two-sided
     Fisher exact p;
   - two sign-shuffle nulls for the OR (2,000 shuffles each, one-sided p): signs permuted across all
     edges of the stratum, and signs permuted only among the edges of the same TF;
   - a 95% interval for the OR from resampling TFs with replacement (2,000 resamples);
   - balanced accuracy of the database sign (mean of the accuracy on activation edges and on
     inhibition edges) with a bootstrap 95% interval, next to two baselines: always predict "down",
     and a random sign. The balanced accuracy over the observed up / down classes is also shown.
5. For the top 10% stratum, results are broken down by DoRothEA confidence level (levels with at
   least 30 edges).

Random seed 20260801. The run is deterministic.

## How to run

From the package root:

```
python code/sign_test.py
```

Options: `--data` (default `data`), `--out` (default `outputs`), `--lines` (default `k562,rpe1`).
Needs numpy, pandas and scipy. CPU only; about 1 minute and under 0.5 GB of memory.

## Outputs (`outputs/`)

| file | content |
|---|---|
| `run_log.txt` | Everything the script prints: data sizes, the balanced-accuracy table, the 2x2 / odds-ratio table with nulls and intervals, and the level breakdown, for both lines. |
| `sign_k562.json`, `sign_rpe1.json` | All numbers in the log, per stratum, in machine-readable form. |
| `edges_k562.csv`, `edges_rpe1.csv` | One row per scored edge: TF, target, database sign (+1 activation, -1 inhibition), DoRothEA level, measured response dE. |

`SUMMARY.md` gives the analyst's write-up of the results.
