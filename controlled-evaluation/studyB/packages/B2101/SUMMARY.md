# K562 -> RPE1 zero-shot transfer of the cell-cycle phase readout

Run: C2S-Scale-Gemma-2-2B, layer 21. Source 3,000 K562 non-targeting controls, target
3,000 RPE1 non-targeting controls, 6,544 shared genes for the expression readout.
Output: `outputs/transfer_rpe1.json`, log in `outputs/run_log.txt`.

## Result

| readout | within (K562, 5-fold CV) | transfer (RPE1, frozen) | retention |
|---|---|---|---|
| model (C2S layer 21) | 0.886 (13 deg) | 0.789 (23 deg) | 0.89 |
| expression (6,544 shared genes) | 0.913 (11 deg) | 0.878 (16 deg) | 0.96 |
| constant predictor (RPE1 mean phase) | - | 0.043 (86 deg) | - |
| uniform random predictor | - | 0.017 (88 deg) | - |

Scores are R_diff between predicted and true phase (1 = perfect up to a rotation,
0 = no association). Median angular error after the best rotation is in brackets.

- **Both readouts work at home.** On held-out K562 cells both reach an R_diff of about
  0.9.
- **Both transfer to RPE1.** The frozen readouts reach 0.789 and 0.878, against 0.043
  for a constant predictor and 0.017 for a random one. Median error is 23 and 16 deg,
  against 86-88 deg for chance. Retention is 0.89 and 0.96.
- **Both pass the validity rule.** The best rotation is small (-4 and -2 deg), so the
  frozen readouts land on the right part of the circle with no adjustment on RPE1.
- **The phase convention is consistent.** The S peak sits at 45 deg in both datasets,
  and G2/M follows it by 109 deg (K562) and 97 deg (RPE1).

## Interpretation

1. **Valid zero-shot transfer.** A phase readout fit on K562 recovers RPE1 phase,
   both from the model's activations and from expression.
2. **The RPE1 phase label behaves as a cell-cycle label.** A readout trained on a
   different cell line recovers it to within about 20 deg, with no RPE1 labels used.
3. **The model's layer-21 phase signal is not specific to K562.** It carries over to
   a near-diploid epithelial line with a different karyotype, without refitting.

## Limits

- RPE1 comes from the same lab and platform as K562. This tests a change of cell
  line, not of batch or platform.
- The two readouts do not get the same input. The model sees the top 512 genes of each
  cell as a ranked list; the expression readout sees all 6,544 shared genes with their
  values. Their scores should not be read as a head-to-head comparison.

## Next

- Repeat on a target from a different platform, with the same chance floors.
