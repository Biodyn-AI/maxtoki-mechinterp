# SAE Atlas — MaxToki-217M (Full 12-Layer) Final Summary

## Model
MaxToki-217M-HF — Llama decoder, 12 layer states (post-embed + 11 transformer blocks), d=1232, 4928 SAE features/layer.

## Training Summary (all 12 layers)

| Layer | VarExpl | Alive | Dead | Ann Rate | Modules |
|---|---|---|---|---|---|
| L0  | 69.3% | 2498 | 2430 | **80.7%** | 10 |
| L1  | **90.4%** | 4928 | 0 | 74.8% | 10 |
| L2  | **93.5%** | 4928 | 0 | 71.1% | 10 |
| L3  | 93.2% | 4928 | 0 | 68.3% | 10 |
| L4  | 92.4% | 4928 | 0 | 66.1% | 10 |
| L5  | 88.9% | 4928 | 0 | 67.8% | 10 |
| L6  | 84.6% | 4876 | 52 | 68.4% | 10 |
| L7  | 80.3% | 4895 | 33 | 69.6% | 10 |
| L8  | 77.2% | 4884 | 44 | 71.5% | 10 |
| L9  | 74.5% | 4918 | 10 | 69.8% | 10 |
| L10 | 75.7% | 4915 | 13 | 64.1% | 10 |
| L11 | 82.8% | 4813 | 115 | 61.7% | 10 |

**Totals:** 56,702 alive features, 120 modules, ~69% mean annotation rate.

## Annotation Profile (U-shaped)
80.7% (L0) → 66.1% (L4 trough) → 71.5% (L8 recovery) → 61.7% (L11 terminal decline)

## Cross-Layer Persistence (NEW — differs from paper)

Adjacent persistence (decoder cosine > 0.7):
- L0→L1: 0.5%, L1→L2: 3.3%, L2→L3: 1.6%, L3→L4: 1.1%
- L4→L5: 4.6%, L5→L6: 7.2%, **L6→L7: 21.5%** (peak)
- L7→L8: 16.0%, L8→L9: 18.0%, L9→L10: 9.8%, L10→L11: 6.5%

**MaxToki has dramatically higher cross-layer persistence than Geneformer** (21.5% peak vs paper's 2.5% max). Features persist across depth, especially L6-L9.

Long-range from L0: peaks at L0→L8 (13.3%), still 6.3% at L0→L11 (paper: 0% by L12).

## Information Highways (all 11 pairs)

All pairs at **99.6–100%** highway rate (features with ≥1 target at PMI > 3).
Mean max PMI: 4.26 (L0→L1) to 6.77 (L6→L7).
Paper: 97.4–99.8%. Replicates.

## Key Findings

1. **Structural SAE quality replicates** (69-94% VarExpl, 10 modules/layer, 99.9% novel)
2. **Annotation rate shows U-shaped profile** matching the paper
3. **Cross-layer persistence is MUCH higher** than Geneformer — architectural difference
4. **Information highways at 99.6-100%** — replicates
5. **0% TF regulatory specificity** — central negative replicates
6. **Causal patching median 1.01×** — features not causally specific (non-replication)

## Interactive Atlas
https://biodyn-ai.github.io/maxtoki-atlas/
