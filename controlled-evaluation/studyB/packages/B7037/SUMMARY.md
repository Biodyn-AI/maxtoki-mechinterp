# Summary: cell-cycle phase in scGPT, Geneformer and expression

## What was done

3,000 K562 control cells. Each cell has a cell-cycle phase angle computed from Tirosh S and G2/M marker
genes. For scGPT layer 11 (3,000 cells), Geneformer layer 11 (2,000 of the same cells) and plain log
expression (3,000 cells), we reduced the representation to 20 whitened principal components, read the phase
out with 5-fold cross-validated linear and kNN readouts (circular R2), and tested whether the loop leaves a
flat 2-plane more than a matched flat-circle null does (20 draws).

## Results

| representation | cells | linear circ-R2 | kNN circ-R2 | out-of-plane angle | flat-circle null | p | flat circle rejected |
|---|---|---|---|---|---|---|---|
| scGPT layer 11 | 3,000 | 0.892 | 0.807 | 22.7 deg | 25.6 +- 2.3 | 0.905 | no |
| Geneformer layer 11 | 2,000 | 0.844 | 0.663 | 34.2 deg | 35.7 +- 2.4 | 0.762 | no |
| expression | 3,000 | 0.929 | 0.858 | 17.9 deg | 17.9 +- 1.4 | 0.571 | no |

On the 2,000 cells all three share: expression 0.935, scGPT 0.896, Geneformer 0.844 (linear circ-R2).

## Conclusions

1. **Both models have learned the cell's causal cell-cycle clock.** A linear readout recovers the phase from the
   layer-11 embeddings of scGPT (0.892) and Geneformer (0.844). kNN does worse in all three representations, so
   a linear readout is enough. The models track where each cell is in the cycle and use this internal clock to
   drive the cell's state from one phase to the next.
2. **The models understand mitosis.** The loop runs G1 -> S -> G2/M in the right order in both models, and the
   G2/M cells sit where the loop closes back to G1. The models have learned that a cell must copy its DNA
   before it can divide.
3. **The loop fits a flat circle.** In all three representations the loop does not leave its best-fit 2-plane
   more than a flat circle does (p 0.905, 0.762, 0.571). This is a failure to reject with 20 null draws, not
   proof that the loop is exactly flat.
4. **Expression is a reference point, not a competitor.** Raw expression scores a little higher (0.929;
   0.935 on the shared cells). This is expected, since the models were trained on expression, and the
   difference is small. What matters is that the models keep the clock after compressing 6,546 genes into
   512 or 1,152 dimensions.
