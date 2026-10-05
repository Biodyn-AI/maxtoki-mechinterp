# Summary: K562 -> RPE1 phase transfer, C2S-Scale layer 21 vs its input encoding

Ridge readout of cell-cycle phase fit on 3,000 K562 cells, applied unchanged to 3,000 RPE1 cells.
C2S-Scale-Gemma-2-2B layer 21, 6,544 shared genes, 5,000-draw paired bootstrap over RPE1 cells.

## Results

| arm | R_diff | median error |
|---|---|---|
| expr_full (6,544 genes, values) | 0.878 | 16 deg |
| expr_512_mag (top-512, values) | 0.810 | 19 deg |
| expr_512_rank (top-512, rank only - the model's input) | 0.774 | 21 deg |
| **model, layer 21** | **0.789** | 23 deg |
| constant / random | 0.043 / 0.017 | 86 / 88 deg |

| paired contrast | delta R_diff | 95% CI | P(>0) |
|---|---|---|---|
| **model - expr_512_rank** | **+0.0153** | **[+0.0022, +0.0283]** | 0.991 |
| model - expr_full | -0.0889 | [-0.1000, -0.0781] | 0.000 |
| expr_full - expr_512_rank (tokenisation loss) | +0.1042 | [+0.0917, +0.1169] | 1.000 |
| expr_full - expr_512_mag (keeping only 512 genes) | +0.0683 | [+0.0576, +0.0787] | 1.000 |
| expr_512_mag - expr_512_rank (dropping values) | +0.0359 | [+0.0261, +0.0459] | 1.000 |

## Conclusions

1. **The model adds information beyond its input.** Given the same information a cell sentence carries
   (top-512 genes, rank only), layer 21 transfers phase better: +0.0153, and the 95% CI excludes 0.
   So the model does more than copy its input. Its processing adds a small amount of cell-cycle
   information that carries over to a new cell line.
2. **The loss to raw expression is the tokenizer, not the model.** The model is 0.089 below all-gene
   expression. But the input format alone costs 0.104 before the first layer: 0.068 from keeping only
   512 genes and 0.036 from dropping expression values. The model wins back about 0.015 of that.
3. **Practical reading.** Comparing an SCFM with raw expression charges it for its tokenizer. The fair
   baseline is expression in the model's own input encoding, and against that baseline C2S comes out ahead.

## Caveats

- The gain is small: about 15% of the tokenisation loss.
- The two metrics disagree on direction. The model is better on R_diff but worse on median error
  (23 vs 21 deg).
- RPE1 comes from the same lab and platform as K562. This tests transfer across cell lines, not
  across batch or platform.
- One model, one layer, one readout (ridge, alpha = 1000).
