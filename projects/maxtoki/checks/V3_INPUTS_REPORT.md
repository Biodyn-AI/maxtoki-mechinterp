# V3-0: correct-encoding loader for MaxToki inputs

Date: 2026-10-01. Code: `setup/inputs_v3.py` (new shared module) and `setup/test_inputs_v3.py`
(tests). Outputs: `checks/v3_inputs/`. No existing file was changed.

## 0. Results first

1. **The new loader gives the order MaxToki expects, for every dataset any run used.**
   530 real cells (10 dataset × context groups, the audit's own cells) pass every check.
   Each token sequence is the counts / gene-median order of its cell, cut at `max_len − 2`.
2. **The check catches the old error.** The deployed tokens (`tokenize_cell` on stored `X`)
   fail the new check in 475 of 475 non-RPE1 cells (45 to 1,870 out-of-order neighbour
   pairs per cell; median 593). The stored log1p `X` rows fail the "looks like counts" test in
   475 of 475 cells. RPE1, which was already correct, passes both (55 of 55).
3. **The numbers match the audit.** On the audit's 5 primary cells per group, my Spearman and
   top-200 numbers between the old and the new order equal the audit's per-cell numbers
   (largest difference 5 × 10⁻¹⁰). On the extended set (up to 55 cells) the largest
   difference is 1.8 × 10⁻⁶ for Spearman and 0.001 for the share measures. These come from
   float32 rounding in the audit's reference (section 4).
4. **The K562 drop-in loader picks the same cells as before.** For the three calls the v2
   scripts made, it returns the same rows as `hooks_v2.load_k562_control_cells`. The
   200-cell call returns the exact 200 rows saved by `v2_circuit_trace`. The old loader
   rebuilds the saved v2 tokens in 200 of 200 cells. The new tokens differ from the old ones
   in 200 of 200 cells. So v3 re-runs can be paired cell by cell with the v2 runs.
5. **The model runs on the new inputs.** 50 forward passes (MPS, float32, contexts 1,024–4,096)
   gave finite outputs. A zero edit at all 12 hook sites (hooks_v2) left the logits exactly
   unchanged (max difference 0.0).
6. **New finding: Smart-seq2 cells look foreign to the model under the correct encoding.**
   Next-gene loss (NLL, nats per gene token) is **3.35** [95% CI 3.20–3.53] in 42 UMI cells
   (10x, Perturb-seq) and **8.11** [7.88–8.38] in 8 Smart-seq2 cells. A uniform guess over the
   whole vocabulary would score 9.92. Top-1 accuracy is 0.185 vs 0.010. The two groups do not
   overlap (lowest Smart-seq2 7.66, highest UMI 4.75; Mann–Whitney *p* = 3.7 × 10⁻⁹). This
   matters for the topology "external lung" domain (Krasnow, all Smart-seq2) and for the few
   Smart-seq2 cells in Tabula Sapiens panels. I did not test why (section 7).

## 1. What was built

`setup/inputs_v3.py` (import it; later agents may add functions at the end, never change these):

| Function | What it does |
|---|---|
| `CountReader(name)` / `read_counts(name, rows)` | Reads rows one at a time with h5py and returns count vectors aligned to `var`. Checks every row. |
| `tokenize_counts(counts, var_map, max_len)` | counts / row total × 10,000, then `maxtoki_adapter.tokenize_cell` (÷ median, rank high→low, keep `max_len − 2`, add `<bos>`=2 / `<eos>`=3). Runs `check_encoding` and raises `EncodingError` on failure. |
| `check_encoding(cell_counts, tokens, var_map, max_len)` | Checks a token sequence against the cell's counts (list below). Returns a dict with `"pass"`. |
| `assert_encoding_batch(counts, cells, var_map, max_len)` | Runs `check_encoding` on a batch; raises on any failure. Call it before every forward pass. |
| `tokenize_rows(name, rows, max_len)` | Correctly encoded cells for any dataset and rows. |
| `load_k562_control_cells(n_cells, pool=100, seed=42, max_len=2048)` | Drop-in for `hooks_v2.load_k562_control_cells`: same selection, same return values, correct input. |
| `deployed_tokenize_X(name, rows, max_len)` | The OLD wrong path, kept only for comparisons. Never feed it to the model. |
| `full_order`, `order_agreement` | Untruncated orders and the audit's agreement measures. |
| `read_obs_column(name, col, rows)` | One obs column, e.g. `"assay"` to find Smart-seq2 cells. |
| `encoding_record`, `file_fingerprint`, `array_sha256` | Blocks for `run_config.json`. |

Count source per file (dataset names accepted by every function):

| Name | File | Count source |
|---|---|---|
| `k562` (alias `replogle_concat`) | `replogle_concat.h5ad` | `round(expm1(X) / unit)` |
| `adamson` | `adamson/perturb_processed_symbols.h5ad` | `round(expm1(X) / unit)` |
| `rpe1` | `ReplogleWeissman2022_rpe1.h5ad` | `X` (integer counts) |
| `ts_immune`, `ts_immune_sub20k`, `ts_lung`, `ts_kidney` | Tabula Sapiens h5ad files | `raw/X` (integer counts); option `source="decontX"` |
| `krasnow` | `krasnow_lung_smartsq2.h5ad` | `raw/X` (integer read counts) |

For K562 and Adamson, `unit` is the smallest non-zero `expm1(X)` value in the cell. The loader
first checks that every value is a whole-number multiple of it. Then it divides and rounds.
This gives integer counts up to one per-cell factor, so the order is the counts order. It also
removes the float32 noise in `X` (relative error about 3 × 10⁻⁷). Without the rounding, that
noise swaps a few genes whose values are almost tied (4 of 55 K562 control cells; see C4b).
`expm1_mode="float"` returns plain `expm1(X)` if someone needs it.

What `check_encoding` checks:
1. `<bos>` first, `<eos>` last, no other special token, no repeated gene, every token a gene of
   this dataset with count > 0.
2. counts / median does not increase along the sequence. Values within 10⁻⁶ of each other
   (relative) count as tied and may come in either order. That is float32 rounding.
3. The length is min(expressed vocab genes, `max_len − 2`), and no gene left out has a larger
   value than a kept gene.
4. The counts look like counts: whole numbers, or whole-number multiples of the smallest value
   (tolerance max(10⁻³, 5 × 10⁻⁶ × multiple)). Log values fail this, so the same wrong input
   cannot pass as both counts and tokens.
5. Agreement with an independent stable-sort reference: positions that differ must hold tied
   values.

## 2. Checks and results

Cells: the audit's cells (`checks/input_encoding_audit/order_metrics_cells.csv`), 5 primary +
up to 50 more per group, same rows. 530 cells in total. All checks passed.

| Check | Result |
|---|---|
| C1 count test on the v3 source | 530 / 530 pass. RPE1, TS, Krasnow: exact integers. K562: largest distance from a whole multiple 2.2 × 10⁻⁴ (multiples up to 775). Adamson: 1.5 × 10⁻⁴ (up to 930). |
| C1, second way (K562) | The implied cell total 10,000 / unit is a whole number in all 90 K562 cells (within 0.0034). It is 0.978–0.991 of the file's `UMI_count`. So one unit is one UMI, and the CP10k total was about 1.3% below `UMI_count`. Adamson's implied totals are not whole numbers, so there I can only say the values are proportional to integers. Order is not affected. |
| C2 stored `X` fails the count test | 475 / 475 non-RPE1 cells fail (as they should). RPE1 `X` passes 55 / 55. |
| C3 v3 tokens pass `check_encoding` | 530 / 530. |
| C4 equal to `tokenize_cell` on raw integer counts | 528 / 530 identical; the other 2 (RPE1) differ only inside exact ties. |
| C4b K562/Adamson: tie-equivalent to `tokenize_cell(expm1(X))` | 145 / 145 (137 identical; 8 differ only inside float32 near-ties). |
| C5 deployed tokens fail `check_encoding` | 475 / 475 non-RPE1 cells fail. RPE1 deployed tokens pass 55 / 55. |
| C6 deployed path rebuilds the tokens saved by the runs | 255 / 255 (K562 controls 55, K562 perturbed 35, TS immune 2,048 55, TS immune 4,096 55, TS lung 4,096 55). |
| C7 agreement numbers equal the audit's | 530 / 530 within tolerance (Spearman ≤ 10⁻⁵, shares ≤ 10⁻³); 5 primary cells per group equal to ≤ 5 × 10⁻¹⁰. |
| L1 K562 drop-in loader | Calls (5, 100, 42), (20, 100, 42), (200, 200, 42): same rows as the old loader; 200 rows equal the saved `v2_circuit` rows; old loader = saved tokens 200 / 200; new tokens pass the check 225 / 225; identical to old 0 / 225. |
| L2 TS kidney (no MaxToki run uses it) | 5 / 5 random rows load and pass. |
| M1 model smoke test | 50 / 50 forward passes finite; zero edit at all 12 sites changes logits by 0.0. |

## 3. How different the correct input is from what the runs fed

Spearman = rank correlation of gene positions over all expressed genes (1 = same order).
Top-200 = share of the first 200 genes that are the same genes. Extended set: mean
[bootstrap 95% CI over cells, 10,000 resamples, seed 20261001].

| Group (runs) | 5 cells: Spearman mean (min) | 5 cells: top-200 mean (min) | Audit 5-cell row | Extended n | Spearman [95% CI] | Top-200 [95% CI] |
|---|---|---|---|---|---|---|
| K562 controls, 2,048 | 0.837 (0.782) | 0.549 (0.485) | 0.84 (0.78) / 0.55 (0.48) | 55 | 0.834 [0.826, 0.842] | 0.560 [0.552, 0.569] |
| K562 perturbed, 2,048 | 0.844 (0.830) | 0.531 (0.470) | 0.84 (0.83) / 0.53 (0.47) | 35 | 0.839 [0.831, 0.846] | 0.559 [0.548, 0.569] |
| Adamson controls, 2,048 | 0.787 (0.703) | 0.700 (0.660) | 0.79 (0.70) / 0.70 (0.66) | 55 | 0.809 [0.794, 0.823] | 0.702 [0.695, 0.708] |
| RPE1 controls, 2,048 | 1.000 | 1.000 | 1.00 / 1.00 | 55 | 1.000 | 1.000 |
| TS immune, 2,048 | 0.893 (0.854) | 0.653 (0.585) | 0.89 (0.85) / 0.65 (0.58) | 55 | 0.858 [0.840, 0.876] | 0.624 [0.602, 0.647] |
| TS immune, 4,096 | 0.869 (0.809) | 0.658 (0.570) | 0.87 (0.81) / 0.66 (0.57) | 55 | 0.873 [0.858, 0.888] | 0.634 [0.608, 0.662] |
| TS lung, 4,096 | 0.874 (0.809) | 0.591 (0.420) | 0.87 (0.81) / 0.59 (0.42) | 55 | 0.863 [0.841, 0.884] | 0.590 [0.562, 0.617] |
| TS lung, 2,048 | 0.876 (0.865) | 0.586 (0.540) | 0.88 (0.87) / 0.59 (0.54) | 55 | 0.873 [0.851, 0.891] | 0.594 [0.575, 0.611] |
| TS immune 20k, 1,024 | 0.880 (0.833) | 0.646 (0.555) | 0.88 (0.83) / 0.65 (0.56) | 55 | 0.850 [0.820, 0.872] | 0.608 [0.587, 0.628] |
| Krasnow (Smart-seq2), 2,048 | 0.766 (0.711) | 0.501 (0.370) | 0.77 (0.71) / 0.50 (0.37) | 55 | 0.758 [0.747, 0.769] | 0.495 [0.471, 0.518] |

What this means: the fed order was related to the right one (Spearman 0.76–0.89) but 30–50%
of the top 200 genes were wrong. Every 5-cell number equals the audit's. Kept-set overlap
(share of correctly kept genes the run also kept) on the extended set: K562 0.92, Adamson
1.00, TS 0.85–0.99, Krasnow 0.95. Almost no gene sat at its right position (0.001–0.012),
except RPE1 (0.9999).

## 4. Why a few extended cells differ from the audit by a tiny amount

The audit built its reference with `tokenize_cell` on float32 values. v3 uses float64 integer
counts. Genes whose counts / median differ by less than about 10⁻⁶ can swap places under
float32 rounding. This moved Spearman by at most 1.8 × 10⁻⁶ and changed at most 2 of 2,046
positions (RPE1, where the deployed path used float32 `X`). It does not change any reported
number at the precision used.

## 5. Model check on the new inputs (baseline for later runs)

Only correctly encoded inputs were fed; each batch passed `assert_encoding_batch` first.
NLL = mean negative log-likelihood of the next gene token, in nats (lower = more predictable).
5 primary cells per group; mean [bootstrap 95% CI over cells].

| Group | NLL | NLL, first 200 positions | Top-1 accuracy |
|---|---|---|---|
| K562 controls | 3.08 [2.96, 3.23] | 4.96 | 0.207 |
| K562 perturbed | 3.09 [2.97, 3.20] | 4.91 | 0.203 |
| Adamson controls | 4.62 [4.51, 4.70] | 6.14 | 0.083 |
| RPE1 controls | 3.00 [2.92, 3.06] | 4.70 | 0.211 |
| TS immune 2,048 (1 of 5 Smart-seq2) | 4.20 [3.19, 5.95] | 5.68 | 0.154 |
| TS immune 4,096 (1 of 5 Smart-seq2) | 3.93 [2.86, 5.88] | 5.70 | 0.182 |
| TS lung 4,096 (1 of 5 Smart-seq2) | 4.02 [2.84, 6.13] | 5.85 | 0.176 |
| TS lung 2,048 | 3.34 [3.13, 3.52] | 5.14 | 0.182 |
| TS immune 20k, 1,024 | 3.64 [3.60, 3.68] | 4.70 | 0.163 |
| Krasnow (all Smart-seq2) | 8.25 [7.98, 8.61] | 8.12 | 0.009 |
| **All UMI cells (n = 42)** | **3.35 [3.20, 3.53]** | 5.08 | 0.185 [0.170, 0.199] |
| **All Smart-seq2 cells (n = 8)** | **8.11 [7.88, 8.38]** | 8.29 | 0.010 [0.008, 0.012] |

Reading it:
- In UMI cells the model predicts the next gene well above chance (uniform guess = 9.92 nats).
- In every Smart-seq2 cell it is close to guessing. Smart-seq2 `raw/X` holds read counts.
  Read counts grow with gene length; UMI counts do not. That is one possible reason. I did not
  test it.
- Adamson is worse than K562 (4.62 vs 3.08). The Adamson file keeps only 3,769 vocab genes, so
  many genes the model expects are absent. That is one possible reason; not tested.
- Peak memory of the model part: 1.1 GB resident (`/usr/bin/time -l`). Wall time: 53 s.
  Running the model part twice gave identical NLL in every cell.

## 6. What changed compared with the deployed and v2 runs

- Deployed and v2 runs (except RPE1) fed `tokenize_cell(X)` with `X` = log1p values. I rebuilt
  that path (`deployed_tokenize_X`) and it matches every saved token set (255 + 200 cells).
- v3 feeds counts. Same cells, same tokenizer, same median dictionary, same `max_len`; only the
  values that are ranked change. The changes in order are the section 3 numbers.
- The K562 cell selection is unchanged, so v3 and v2 results can be compared cell by cell.
- New: every v3 input is checked before use, and `run_config.json` can record the check result
  (`encoding_record`).

## 7. What was not done

- **No forward pass on wrong inputs.** The workflow rules forbid feeding inputs that fail the
  check. So I did not measure whether the model finds the old order or the new order more
  predictable. That would be a direct, model-based test that counts / median is what it
  expects. It costs about 2 minutes of MPS if it is allowed.
- **Smart-seq2 cause not tested.** The gene-length explanation is a guess. I did not test
  whether the deployed (log) encoding was also near chance on these cells, for the same reason
  as above.
- **raw/X vs decontX in Tabula Sapiens.** The task says `raw/X`. On the 25 TS primary cells the
  decontX order agrees with the raw/X order at Spearman ≥ 0.92 (mean 0.98–0.999), but top-200
  overlap drops to 0.74 and kept-set overlap to 0.69 in single cells. `source="decontX"` is
  there if a run wants it; the choice should be stated in each report.
- No run script was changed. Later agents must switch to these loaders themselves.
  Longevity reads `adata.X` through the upstream repo; it needs its own swap (`tokenize_rows`
  covers the data).
- The 1B model was not run. It uses the same tokenizer, so the inputs are the same.
- K562 and Adamson files keep only 6,546 and 4,888 genes. The correct order is the order of
  those genes. Genes not in the file cannot be recovered.
- TS kidney: loader tested on 5 cells; no audit numbers exist to compare.

## 8. How to use it

```python
import sys; sys.path.insert(0, "<REPO_ROOT>/projects/maxtoki/setup")
import numpy as np, inputs_v3 as I
cells, rows, h5 = I.load_k562_control_cells(200, pool=200, seed=42)          # was hooks_v2.load_k562_control_cells
cells, kept, info, counts = I.tokenize_rows("ts_immune", rows_ts, 2048, return_counts=True)
chk = I.assert_encoding_batch(counts, cells, "ts_immune", 2048, label="batch 0")  # before each forward pass
run_config["input_encoding"] = I.encoding_record("ts_immune", rows=kept, check_summary=chk,
                                                 counts_sha256=I.array_sha256(np.stack(counts)))
assay = I.read_obs_column("ts_immune", "assay", kept)                       # flag Smart-seq2 cells
```

Files:
- `setup/inputs_v3.py`, `setup/test_inputs_v3.py`
- `checks/v3_inputs/cells_cpu.csv` (per cell, all checks and agreement numbers),
  `summary_cpu.csv`, `summary.json` (means, SD, CIs), `cpu_results.json` (loader checks,
  provenance), `decontx_vs_raw.csv`, `model/*.json`, `model_cells.csv`, `run_config.json`
  (device, versions, seeds, cell IDs, file fingerprints, sha256 of the count rows used, code
  sha256, wall time per chunk, encoding check result).
- Re-run: `OMP_NUM_THREADS=4 .venv/bin/python setup/test_inputs_v3.py --part cpu`, then
  `--part model --device mps`, then `--part summary` (about 2 minutes in total).

## Plain-words summary

I wrote one shared loader that gives MaxToki the gene order it was built for: genes ranked by
raw count divided by the gene's typical level, with no log. For K562 and Adamson, which store
only log values, the loader undoes the log and checks that the result is whole-number counts.
For Tabula Sapiens and Krasnow it reads the stored raw counts. RPE1 was already right.

Every cell the loader returns is checked: the token order must match the counts order, and the
input must look like counts. The old inputs fail this check in every affected cell, so the
check works. My numbers for how far the old order was from the right one match the audit
exactly on the same cells. The new K562 loader picks the same cells as the old one, so new and
old results can be compared cell by cell.

The model runs fine on the new inputs. One surprise: for Smart-seq2 cells (all of the Krasnow
lung data and a few Tabula Sapiens cells) the model can barely predict the next gene, while for
10x and Perturb-seq cells it does well. Any re-run that uses Smart-seq2 cells should treat them
separately.
