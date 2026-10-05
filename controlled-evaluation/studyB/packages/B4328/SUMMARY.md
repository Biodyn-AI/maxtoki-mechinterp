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

1. **Phase can be read linearly from both models.** A linear readout recovers the phase from the layer-11
   embeddings of scGPT (0.892) and Geneformer (0.844). kNN does worse in all three representations, so a
   linear readout is enough.
2. **Expression does better than both models.** Raw expression scores 0.929, and on the same 2,000 cells
   0.935 vs 0.896 (scGPT) and 0.844 (Geneformer). The phase label is itself the angle of a 2-D linear
   projection of marker-gene expression, so expression has a built-in advantage. The models keep the phase
   signal; they do not add to it.
3. **The loop fits a flat circle.** In all three representations the loop does not leave its best-fit 2-plane
   more than a flat circle does (p 0.905, 0.762, 0.571). This is a failure to reject with 20 null draws, not
   proof that the loop is exactly flat.
4. **The evidence is correlational.** We read the phase out of precomputed embeddings. Nothing here changes the
   models' inputs or activations, so it does not show that the models use this signal, or that they have
   learned how the cycle is driven or anything about mitosis.
