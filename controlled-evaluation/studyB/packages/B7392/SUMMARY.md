# Summary: K562 -> RPE1 phase transfer, C2S-Scale layer 21 vs its input encoding

Ridge readout of cell-cycle phase fit on 3,000 K562 cells, applied unchanged to 3,000 RPE1 cells.
C2S-Scale-Gemma-2-2B layer 21, 6,544 shared genes, 5,000-draw paired bootstrap over RPE1 cells.

## Results

| arm | R_diff | median error |
|---|---|---|
| expr_full (6,544 genes, values) | 0.878 | 16 deg |
| expr_512_mag (top-512, values) | 0.820 | 17 deg |
| expr_512_rank (top-512, rank only - the model's input) | 0.792 | 20 deg |
| **model, layer 21** | **0.789** | 23 deg |
| constant / random | 0.043 / 0.017 | 86 / 88 deg |

| paired contrast | delta R_diff | 95% CI | P(>0) |
|---|---|---|---|
| **model - expr_512_rank** | **-0.0022** | **[-0.0144, +0.0105]** | 0.360 |
| model - expr_full | -0.0889 | [-0.1000, -0.0781] | 0.000 |
| expr_full - expr_512_rank (tokenisation loss) | +0.0868 | [+0.0743, +0.0992] | 1.000 |
| expr_full - expr_512_mag (keeping only 512 genes) | +0.0582 | [+0.0473, +0.0694] | 1.000 |
| expr_512_mag - expr_512_rank (dropping values) | +0.0285 | [+0.0195, +0.0378] | 1.000 |

## Conclusions

1. **The model is at parity with its input.** Given the same information a cell sentence carries
   (top-512 genes, rank only), layer 21 transfers phase equally well: -0.0022, and the 95% CI spans 0.
   So the model neither adds nor loses cell-cycle information. It carries what it is given through
   21 layers and passes it on to a new cell line.
2. **The loss to raw expression is the tokenizer, not the model.** The model is 0.089 below all-gene
   expression. But the input format alone costs 0.087 before the first layer: 0.058 from keeping only
   512 genes and 0.029 from dropping expression values. That is almost the whole gap.
3. **Practical reading.** Comparing an SCFM with raw expression charges it for its tokenizer. The fair
   baseline is expression in the model's own input encoding, and against that baseline C2S comes out even.

## Caveats

- Parity is not proof of zero effect: the CI allows a gain of up to 0.0105 or a loss of up to 0.0144.
- The two metrics lean the same way. The model is slightly worse on R_diff and on median error
  (23 vs 20 deg).
- RPE1 comes from the same lab and platform as K562. This tests transfer across cell lines, not
  across batch or platform.
- One model, one layer, one readout (ridge, alpha = 1000).
