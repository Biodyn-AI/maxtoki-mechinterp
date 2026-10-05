# Summary: cell-cycle phase in STATE-SE layer 11

## Result

STATE-SE layer 11 does not hold a readable cell-cycle phase. On 3,000 non-targeting control cells the
linear circ-R^2 is -0.017 (kNN -0.097). The expression of the same cells gives 0.929 (kNN 0.858).

| representation | cells | input dim | linear circ-R^2 | kNN circ-R^2 |
|---|---|---|---|---|
| expression, log1p(CP10k) | 3,000 | 6,546 | **0.929** | 0.858 |
| STATE-SE layer 11 | 3,000 | 2,048 | **-0.017** | -0.097 |

Probe: 20 whitened principal components, 5-fold ridge on (cos phi, sin phi). 0 means no better than giving
every cell the mean phase.

## What it means

1. The cell-cycle loop is clearly in the input. A linear probe on 20 components of expression reads the
   phase almost perfectly (0.929).
2. STATE-SE loses it. Both probes are at or below 0, so the layer-11 embedding tells us nothing about where
   a cell is in the cycle. The kNN probe does no better than the linear one, so a nonlinear read-out does
   not find it either.
3. So STATE is different from the expression it reads. Its layer-11 cell vector does not keep cell-cycle
   state. A likely reason is the design: genes enter as protein-embedding tokens, and the cell vector is a
   mean over all expressed genes. That mean may wash out a program carried by about 90 marker genes.
4. In practice, STATE-SE layer 11 should not be used where a readable cell-cycle phase is needed, such as
   phase decoding or phase-aware comparisons between cells.

## Caveats

- STATE-SE expects raw counts. The source holds only log1p(CP10k) values, so the model was given log1p
  input. This is off its training distribution and could weaken the signal.
- phi is a linear projection of marker expression, and the markers are among the expression genes. So the
  probe favours expression. The expression number is an upper reference, not an equal rival.
- The 3,000 controls are pooled from four cell lines (K562, RPE1, HepG2, Jurkat).
- One layer (11 of 16), one probe size (20 components), one cross-validation seed.
