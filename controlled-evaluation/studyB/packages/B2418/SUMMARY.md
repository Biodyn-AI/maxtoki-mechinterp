# Summary: how compact is each model's cell manifold? (fetal gut epithelium)

## What was done

The same 4,269 fetal gut epithelial cells were embedded by three frozen models: scGPT whole-human,
Geneformer V2-316M and STATE SE-600M. Each cell's embedding is the layer-11 output, mean-pooled
over the cell's expressed genes (`code/extraction/`). `code/geometry.py` then measures four things
on each model's cloud of cells: participation ratio (PR, an effective number of dimensions), the
share of variance on PC1, d90 (PCs needed for 90% of the variance) and the coefficient of variation
of per-cell norms. It repeats the measures on 10 random halves of the cells.

## Results (`outputs/run_log.txt`)

| model | width | PR | PC1 share | d90 | norm CV |
|---|---|---|---|---|---|
| scGPT | 512 | 8.11 | 29.2% | **19** | **0.003** |
| Geneformer | 1152 | **2.73** | **57.6%** | 20 | 0.019 |
| STATE | 2048 | 56.5 | 11.3% | 775 | 0.028 |

The random halves agree closely: PR 7.91-8.26 (scGPT), 2.69-2.77 (Geneformer), 53.6-57.7 (STATE);
PC1 share 28.5-29.8% / 57.0-58.1% / 11.1-11.6%.

## Conclusions

1. **Geneformer has the most compact cell manifold by PR and PC1 share.** One direction holds 58%
   of the variance (PR 2.7), but with a long tail: it needs 20 PCs to reach 90%.
2. **scGPT spreads its variance more evenly but is just as low-dimensional.** PR 8.1 and PC1 29%,
   yet it reaches 90% of the variance with 19 PCs, the same as Geneformer (the random halves
   overlap: 18-19 vs 19-20). Per-cell norms are almost constant (CV 0.3%).
3. **STATE is by far the most diffuse.** PR 56 and 775 PCs for 90% of the variance; no direction
   holds more than about 11%.
4. **Which of scGPT and Geneformer is "more compact" depends on the measure; STATE is last on
   every measure and in every random half.** It is not a sampling effect.

## Caveats

- Layer 11 is the last layer of scGPT (12 layers) but a middle layer of Geneformer (18) and STATE
  (16). Compactness can change with depth.
- The widths differ (512 / 1152 / 2048). PR and d90 can grow with width. This cannot explain the
  20-fold PR gap between Geneformer and STATE, but it may affect the smaller gaps.
- Not tested: other tissues, other layers, and the models' own cell tokens (CLS) instead of the
  gene mean-pool.
