# Summary: branch-point curvature significance

## Result

Curvature is significant in **6 of 6** model x tissue cells. p = 0.005 in every cell,
which is the smallest value 200 null draws can give. BH q = 0.005 in every cell.

| model | tissue | species | linear R2 | curvature | null mean | z | p | q | bootstrap 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| geneformer | lung | human | 0.935 | +0.0180 | -0.0175 | +4.85 | 0.005 | 0.005 | [+0.0146, +0.0216] |
| scgpt | lung | human | 0.966 | +0.0076 | -0.0119 | +4.68 | 0.005 | 0.005 | [+0.0051, +0.0108] |
| geneformer | gut | human | 0.759 | +0.0407 | -0.0143 | +3.99 | 0.005 | 0.005 | [+0.0300, +0.0513] |
| scgpt | gut | human | 0.861 | +0.0174 | -0.0153 | +4.90 | 0.005 | 0.005 | [+0.0083, +0.0270] |
| geneformer | pancreas | mouse | 0.946 | +0.0021 | -0.0273 | +2.45 | 0.005 | 0.005 | [+0.0000, +0.0042] |
| scgpt | pancreas | mouse | 0.973 | +0.0024 | -0.0120 | +3.49 | 0.005 | 0.005 | [+0.0012, +0.0036] |

## What this means

1. **The curvature is real and general.** Both models show it in all three lineages. It is
   not a quirk of one model or one tissue. Each null target has the same linear strength
   as the real data, so the test asks for curvature beyond a linear relation of that
   strength. Every cell passes, with z from +2.5 to +4.9.
2. **The effect is small.** Curvature is +0.002 to +0.041 R2 on top of a linear R2 of
   0.76 to 0.97. So the embeddings are mostly linear in developmental time, with a small
   curved part near the branch points.
3. **Human tissues curve more than mouse.** Human lung and gut give +0.008 to +0.041.
   Mouse pancreas gives +0.002 in both models. The two models agree on this order, so it
   looks like a property of the trajectory's species, not of the model or the lineage.
4. **Bootstrap agrees.** The 95% interval is above zero in five cells. For Geneformer
   pancreas it touches zero (lower bound +0.0000), but the null test still rejects there.

## Caveats

- The bootstrap holds the fitted decoders fixed. It measures sampling noise in the R2
  difference, not noise in the fit.
- One embedding layer (layer 11) and 1,500 cells per tissue were used. Other layers were
  not tested.
- There is only one mouse tissue. The species reading should be checked on a second
  mouse trajectory.
