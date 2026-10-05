# Summary: branch-point curvature significance

## Result

Curvature is significant in **1 of 6** model x tissue cells: Geneformer gut (p = 0.005,
BH q = 0.030, z = +5.73). The other five cells have q >= 0.34.

| model | tissue | species | linear R2 | curvature | null mean | z | p | q | bootstrap 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| geneformer | lung | human | 0.935 | +0.0180 | +0.0069 | +1.07 | 0.124 | 0.338 | [+0.0146, +0.0216] |
| scgpt | lung | human | 0.966 | +0.0076 | +0.0164 | -0.65 | 0.771 | 0.771 | [+0.0051, +0.0108] |
| geneformer | gut | human | 0.759 | +0.0407 | +0.0006 | +5.73 | 0.005 | 0.030 | [+0.0300, +0.0513] |
| scgpt | gut | human | 0.861 | +0.0174 | +0.0083 | +0.96 | 0.169 | 0.338 | [+0.0083, +0.0270] |
| geneformer | pancreas | mouse | 0.946 | +0.0021 | +0.0037 | -0.22 | 0.622 | 0.771 | [+0.0000, +0.0042] |
| scgpt | pancreas | mouse | 0.973 | +0.0024 | +0.0082 | -0.68 | 0.741 | 0.771 | [+0.0012, +0.0036] |

## What this means

1. **The curvature is not general.** Only one cell clears the null. In three cells
   (scGPT lung and both pancreas cells) the real curvature is below the null mean, so
   shuffled labels give more apparent curvature than the real ones.
2. **The polynomial decoder gains R2 by chance.** The null mean is positive in all six
   cells (+0.0006 to +0.0164). Each cell has its own noise floor, so raw curvature values
   cannot be compared across cells.
3. **No species conclusion.** Human lung and gut give +0.008 to +0.041 and mouse pancreas
   +0.002, but only human gut is significant. That cell also has the lowest linear R2
   (0.759), so the most room for a curved part. All mouse cells have linear R2 >= 0.946.
   Species and headroom cannot be told apart with this panel.
4. **Bootstrap does not settle it.** The 95% interval is above zero in five cells. It
   shows the R2 difference is stable across resamples, not that it beats chance.

## Caveats

- The bootstrap holds the fitted decoders fixed. It measures sampling noise in the R2
  difference, not noise in the fit.
- Under the permuted labels the linear R2 is much lower than on the real labels. So the
  null is not run at the same headroom as the real data.
- One embedding layer (layer 11) and 1,500 cells per tissue were used. Other layers were
  not tested.
