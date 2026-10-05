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
| S peak / G2M peak | 45.0 / 153.8 deg | 45.0 / 142.1 deg |
| S to G2M separation | 108.8 deg | 97.1 deg |

In both lines the cells spread all around the circle (R near 0), and the S and G2M scores peak about a
quarter turn apart. Both lines give a usable cell-cycle circle.

## K562 vs RPE1, gene by gene (43 genes present in both panels)

| statistic | value |
|---|---|
| circular correlation of per-gene peak phase | **+0.983** |
| median absolute difference | 15 deg |
| mean absolute difference | 16 deg |

CCNE1, CCNE2 and UBE2C are not in the K562 panel and were left out.

## Conclusions

1. The per-gene peak phases are almost perfectly correlated (+0.983). The gene order around the circle in
   RPE1 is the same as the order in K562. The median gene sits 15 deg away from its K562 position.
2. Each line on its own shows a clear cycle, and the two cycles line up gene by gene. The RPE1 phase label
   runs in the same direction as the K562 one and corresponds to it.
3. RPE1's phase label is therefore valid as a transfer target. A readout fit on K562 phase and scored on
   RPE1 is judged against the same ruler, so its score will say something about the readout.
4. Next step: run the K562 -> RPE1 transfer on this label. RPE1 comes from the same lab and platform as
   K562, so this tests transfer across cell biology, not across batch or platform.
