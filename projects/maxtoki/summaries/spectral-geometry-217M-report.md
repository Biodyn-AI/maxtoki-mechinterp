# Residual-stream spectral geometry — MaxToki-217M run report

## Phase 0 — extraction

- model: `MaxToki-217M-HF`
- dataset: `tabula_sapiens_immune`
- n_cells: `2000`
- n_hvg: `1500`
- max_len: `2048`
- device: `mps`
- n_layer_states: `12`
- hidden_size: `1232`
- mean_seq_len: `1783.629`
- mean_fwd_seconds: `0.39222485113143923`
- total_phase_seconds: `907.3946018218994`
- n_genes_with_observations: `1500`

## Phase 1 — global spectral metrics

| Layer | Effective rank | Participation ratio | SV1 var frac | TwoNN |
|---|---|---|---|---|
| 0 | 560.81 | 411.66 | 0.007 | 102.00 |
| 1 | 332.59 | 230.39 | 0.025 | 51.83 |
| 2 | 338.15 | 205.45 | 0.032 | 47.03 |
| 3 | 297.56 | 146.04 | 0.045 | 27.99 |
| 4 | 231.70 | 93.79 | 0.051 | 20.78 |
| 5 | 212.06 | 78.60 | 0.057 | 17.47 |
| 6 | 169.44 | 52.90 | 0.069 | 10.28 |
| 7 | 174.41 | 51.41 | 0.076 | 10.49 |
| 8 | 201.74 | 64.14 | 0.083 | 10.40 |
| 9 | 201.91 | 71.80 | 0.075 | 8.34 |
| 10 | 172.11 | 54.07 | 0.105 | 6.35 |
| 11 | 93.64 | 24.48 | 0.173 | 6.70 |

- depth vs effective rank: ρ=-0.895 p=8.37e-05
- depth vs SV1 variance fraction: ρ=+0.979 p=3.09e-08
- feature-shuffle null on final layer: ER observed 93.64, shuffled 573.08 (ratio 6.12×)

## Phase 3 — SV1 vs GO Cellular Component

| GO CC term | Best layer | z | observed | null mean | p_bh |
|---|---|---|---|---|---|
| ER_lumen (GO:0005788) | L8 | -0.06 | 0.062 | 0.065 | 1.00e+00 |
| cytosol (GO:0005829) | L0 | +1.27 | 0.079 | 0.069 | 6.71e-01 |
| endoplasmic_reticulum (GO:0005783) | L7 | +1.28 | 0.096 | 0.070 | 6.93e-01 |
| extracellular_space (GO:0005615) | L2 | +5.21 | 0.181 | 0.068 | 1.92e-02 |
| mitochondrial_matrix (GO:0005759) | L0 | +0.09 | 0.070 | 0.066 | 1.00e+00 |
| mitochondrion (GO:0005739) | L6 | +1.88 | 0.103 | 0.069 | 3.32e-01 |
| nucleus (GO:0005634) | L7 | +1.06 | 0.075 | 0.069 | 6.93e-01 |
| plasma_membrane (GO:0005886) | L11 | +4.67 | 0.123 | 0.069 | 1.92e-02 |

## Phase 4 — SV2–SV4 vs STRING PPI (pairs_700)

| Axis | Mean z across layers | Significant layers |
|---|---|---|
| SV3 | +3.12 | 3 / 12 |
| SV2 | +2.62 | 1 / 12 |
| SV7 | +1.98 | 2 / 12 |
| SV4 | +1.78 | 0 / 12 |
| SV6 | +1.72 | 0 / 12 |
| SV5 | +1.24 | 1 / 12 |
| SV1 | +0.36 | 0 / 12 |
- SV2 2-point gradient: z_mean(STRING≥700)=+2.62, z_mean(STRING≥900)=+2.46

## Phase 5 — TF vs target classifier + edge-level AUROC

- Joint SV2-SV7 AUROC mean across 12 layers: 0.5667  max: 0.6197 at L11
- Permutation-null p min: 0.0099  max: 0.7822
- SV5-SV7 edge-level AUROC per layer: [0.4899545060136075, 0.5071485884216232, 0.5175876989222365, 0.5031048220840435, 0.4874439310915914, 0.4871239987848745, 0.4971433282377694, 0.4925963701860129, 0.5049418666417056, 0.508533047564466, 0.5141921239083396, 0.5173246179475636]
- Edge-AUROC vs depth: ρ=+0.350 p=2.65e-01

## Phase 6 — Cell-type marker clustering

- Mean AUROC (within-type vs across-type) across layers: 0.6183  max: 0.7787 at L0

## Phase 7 — B-cell attractor dynamics

- rank_PAX5_vs_depth: ρ=+0.881 p=1.53e-04
- rank_BATF_vs_depth: ρ=+0.371 p=2.35e-01
- dist_BATF_to_PAX5_vs_depth: ρ=+0.979 p=3.09e-08
- rank_BACH2_vs_depth: ρ=-0.524 p=8.00e-02
- dist_BACH2_to_PAX5_vs_depth: ρ=+0.888 p=1.14e-04
- rank_BCL6_vs_depth: ρ=+0.524 p=8.00e-02
- dist_BCL6_to_PAX5_vs_depth: ρ=+0.937 p=6.99e-06
- rank_PRDM1_vs_depth: ρ=-0.860 p=3.32e-04
- dist_PRDM1_to_PAX5_vs_depth: ρ=+0.951 p=2.04e-06
- rank_IRF4_vs_depth: ρ=+0.559 p=5.86e-02
- dist_IRF4_to_PAX5_vs_depth: ρ=+0.937 p=6.99e-06
- rank_IRF8_vs_depth: ρ=+0.333 p=2.91e-01
- dist_IRF8_to_PAX5_vs_depth: ρ=+0.888 p=1.14e-04
- gc_plasma_angle_vs_depth: ρ=-0.909 p=4.19e-05
- twonn_B cell_vs_depth: ρ=-0.867 p=2.60e-04
- twonn_T cell_vs_depth: ρ=-0.895 p=8.37e-05
- twonn_Myeloid_vs_depth: ρ=+nan p=nan
- PAX5 rank L0→L11: 12 → 146
- BATF rank L0→L11: 698 → 423
- BACH2 rank L0→L11: 1365 → 671
- GC-plasma angle L0→L11: 71.7° → 46.4°

## Phase 9 — Negative-control findings

- 9d: GO BP SV2 BH-significant: 0/675 (expect near 0)
- 9e: raw ER↔AUROC ρ=-0.503 (p=9.52e-02)
- 9e: partial ER↔AUROC|layer ρ=+0.189 (p=5.57e-01)