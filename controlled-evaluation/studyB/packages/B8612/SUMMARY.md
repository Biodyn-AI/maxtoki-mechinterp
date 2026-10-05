# RPE1 vs K562: does the cell-cycle phase label agree?

**Purpose.** RPE1 non-targeting controls are the planned target for the K562 -> RPE1 phase-transfer test.
Before RPE1's phase can serve as ground truth, it must mean the same thing as K562's phase. Test: the peak
phase of each Whitfield 2002 cell-cycle gene, computed in each line, compared gene by gene.

## Each line on its own

| | K562 | RPE1 |
|---|---|---|
| cell-cycle marker genes used | 87 | 93 |
| PC1+PC2 share of variance | 0.306 | 0.349 |
| phase concentration R | 0.079 | 0.043 |
| S peak / G2M peak | 90.9 / 342.1 deg | 280.1 / 17.2 deg |
| S to G2M separation | 108.8 deg | 97.1 deg |

In both lines the cells spread all around the circle (R near 0), and the S and G2M scores peak about a
quarter turn apart. Both lines give a usable cell-cycle circle.

## K562 vs RPE1, gene by gene (43 genes present in both panels)

| statistic | value |
|---|---|
| circular correlation of per-gene peak phase | **-0.983** |
| median absolute difference | 63 deg |
| mean absolute difference | 87 deg |

CCNE1, CCNE2 and UBE2C are not in the K562 panel and were left out.

## Conclusions

1. The per-gene peak phases are almost perfectly anti-correlated (-0.983). The gene order around the circle
   in RPE1 is the mirror image of the order in K562. The median gene sits 63 deg away from its K562
   position.
2. Each line on its own shows a clear cycle, so this is not a missing cycle in RPE1. The RPE1 phase label
   runs in the opposite direction to the K562 one and does not correspond to it.
3. RPE1's phase label is therefore not valid as a transfer target. A readout fit on K562 phase and scored on
   RPE1 would be judged against a reversed ruler, so its score would say nothing about the readout.
4. Next step: do not run the K562 -> RPE1 transfer on this label. Find a target line whose phase agrees with
   K562 gene by gene before running any transfer.
