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
| scGPT | 512 | **1.82** | **72.4%** | **3** | **0.003** |
| Geneformer | 1152 | 2.73 | 57.6% | 20 | 0.019 |
| STATE | 2048 | 56.5 | 11.3% | 775 | 0.028 |

The random halves agree closely: PR 1.81-1.83 (scGPT), 2.69-2.77 (Geneformer), 53.6-57.7 (STATE);
PC1 share 72.2-72.6% / 57.0-58.1% / 11.1-11.6%.

## Conclusions

1. **scGPT has the most compact cell manifold of the three, on all four measures.** Its gut cells
   lie close to one axis: PC1 holds 72% of the variance and 3 PCs hold 90%. Per-cell norms are
   almost constant (CV 0.3%), so cells differ by direction, not by size.
2. **Geneformer is second.** It also has one dominant direction (PC1 58%, PR 2.7), but with a
   long tail: it needs 20 PCs to reach 90%.
3. **STATE is by far the most diffuse.** PR 56 and 775 PCs for 90% of the variance; no direction
   holds more than about 11%.
4. **The order scGPT > Geneformer > STATE is the same on every measure and in every random half.**
   It is not a sampling effect.

## Caveats

- Layer 11 is the last layer of scGPT (12 layers) but a middle layer of Geneformer (18) and STATE
  (16). Compactness can change with depth.
- The widths differ (512 / 1152 / 2048). PR and d90 can grow with width. This cannot explain the
  30-fold PR gap between scGPT and STATE, but it may affect the smaller gaps.
- Not tested: other tissues, other layers, and the models' own cell tokens (CLS) instead of the
  gene mean-pool.
