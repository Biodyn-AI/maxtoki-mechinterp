# K562 -> RPE1 zero-shot transfer of the cell-cycle phase readout

Run: C2S-Scale-Gemma-2-2B, layer 21. Source 3,000 K562 non-targeting controls, target
3,000 RPE1 non-targeting controls, 6,544 shared genes for the expression readout.
Output: `outputs/transfer_rpe1.json`, log in `outputs/run_log.txt`.

## Result

| readout | within (K562, 5-fold CV) | transfer (RPE1, frozen) | retention |
|---|---|---|---|
| model (C2S layer 21) | 0.897 (13 deg) | -0.814 (24 deg) | -0.91 |
| expression (6,544 shared genes) | 0.919 (11 deg) | -0.796 (16 deg) | -0.87 |

Scores are circular correlations between predicted and true phase. Median angular
error is in brackets.

- **Both readouts work at home.** On held-out K562 cells both reach a circular
  correlation of about 0.9.
- **Both fail on RPE1, and in the same way.** The frozen readouts give strongly
  negative circular correlations (-0.81 and -0.80). Retention is -0.91 and -0.87.
- **Neither passes the validity rule.** No winner is declared, and the model is not
  compared with expression.
- **The phase convention is not the cause.** Orientation is consistent across the two
  datasets: the S peak sits at 45 deg in both, and G2/M follows it by 109 deg (K562)
  and 97 deg (RPE1). A mirrored phase axis is ruled out.

## Interpretation

1. **No valid transfer.** A phase readout fit on K562 does not recover RPE1 phase,
   either from the model's activations or from expression.
2. **The RPE1 phase label is the problem, not the readouts.** The model and expression
   fail by almost the same amount, and both work well on K562. What they share is the
   target label. In RPE1 the marker-PCA angle does not measure cell-cycle position the
   way it does in K562, so RPE1 is not a valid target for this variable.
3. **The run says nothing about generalisation.** It cannot tell whether the model's
   cell-cycle representation carries over to other cell lines better or worse than
   expression.

## Limits

- RPE1 comes from the same lab and platform as K562. Even a working transfer would
  test a change of cell line, not of batch or platform.
- The two readouts do not get the same input. The model sees the top 512 genes of each
  cell as a ranked list; the expression readout sees all 6,544 shared genes with their
  values. Their scores should not be read as a head-to-head comparison.

## Next

- Pick a different target dataset, and check its phase label against the source
  before running transfer.
