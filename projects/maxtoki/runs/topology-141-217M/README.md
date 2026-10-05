# topology-141-217M — MaxToki-217M-HF on `topology-geometry-141-hypotheses` pipeline

Source spec: `pipelines/topology-geometry-141-hypotheses.md` (arxiv 2602.22289).
Target model: **MaxToki-217M-HF** (LlamaForCausalLM, 11 transformer layers, hidden_size=1232).
Run created: 2026-05-03.

## Scoped execution plan

The full source paper drives 141 hypotheses through 53 iterations of a Codex-5.3-xhigh
autonomous loop (Phase 11). That is infeasible for a single run; instead this folder executes
the **headline backbone** of the pipeline end-to-end on MaxToki-217M:

| Phase | Status | What we run |
|---|---|---|
| 0 | included | per-layer per-gene embedding extraction on **lung, immune, external-lung** (Krasnow held-out) |
| 1+2+3 | included | TF/target disjoint splits (3 seeds × 2 regimes), four null families, ΔAUROC harness |
| 4 cross-model CCA | **included** | done as **Phase 14** vs Geneformer V2-316M (CCA r=0.78) and **Phase 4** vs scGPT static embeddings (CCA r=0.40 — alignment is much weaker across architectural families) |
| 5 | included | persistent homology (H01/H03), bifiltration cycle-rank (H47), rewiring null |
| 6 | included | manifold distance hierarchy: Euclidean, geodesic (H13), diffusion (H16), triangle-defect (H69/H70) |
| 7+8 | included | Louvain communities, H116 TRRUST sign-motif, **H123 signed motif-community hardening** (the strongest finding) |
| 9 | included | H91 stability-selected descriptors (combined classifier) |
| 10 | included (spot-check) | H23 Forman curvature directional check (in `phase16_extended/`) + H139 sectional anisotropy spot-check (`phase10/`); the broader 70+ negatives are not exhaustively re-screened |
| 11 autoloop | **substitute run** | 6-hypothesis Claude-as-brainstormer batch with explicit retirement (`outputs/phase11_autoloop/`). 1 novel positive (cross-layer single-axis H123), 2 inconclusive, 2 retire, 1 untestable. Not a faithful Codex-5.3-xhigh replication. |
| 12 | included | H141 strict max-null audit |
| 13 | included | 5-layer hierarchy synthesis + FINAL_SUMMARY.md |

The annotation-extension degradation chain (H124/H127/H130/H138) was added 2026-05-07
in `outputs/phase8_extensions/`. As predicted, the degradation chain on MaxToki is
uninformative — H123 is already at floor (0-1/12 layers per domain), so adding STRING /
GO / continuous-GO / ontology-sheaf features cannot further erode null-gap robustness.
All four extensions land within ±0.01 of H123 baseline.

## Data sources

- **Lung** — `tabula_sapiens_lung.h5ad` (65,847 cells × 60,606 genes; SMART-seq2 / 10x mix)
- **Immune** — `tabula_sapiens_immune.h5ad` (existing spectral-geometry phase0 reused if compatible; otherwise re-extracted)
- **External-lung** — `krasnow_lung_smartsq2.h5ad` (9,409 cells × 53,514 genes; SMART-seq2; not used for tuning) — held-out role per Phase 0

## Reference databases

- TRRUST v2 signed regulatory motifs (`biodyn-nmi-paper/.../trrust_human.tsv`)
- STRING v11 PPI edges (`biodyn-nmi-paper/.../string_ppi_edges.json`)
- Gene Ontology BP / KEGG / Reactome (`biodyn-nmi-paper/.../{go_bp,kegg,reactome}_gene_sets.json`) — used in Phase 7/8/9 only

## Layout

```
topology-141-217M/
├── README.md                  ← this file
├── scripts/                   ← phase-by-phase runners
├── outputs/
│   ├── phase0/                ← embeddings + gene_features per (model, domain)
│   ├── phase1/                ← gene_pool_splits per (domain, regime, seed)
│   ├── phase5/                ← persistent homology + bifiltration cycle-rank
│   ├── phase6/                ← manifold-distance ΔAUROC tables
│   ├── phase4_scgpt/          ← Phase 4 cross-model CCA vs scGPT static embeddings
│   ├── phase78/               ← H16, H116, H123
│   ├── phase8_extensions/     ← H124/H127/H130/H138 annotation chain
│   ├── phase9/                ← H91 stability-selection
│   ├── phase10/               ← H139 sectional-anisotropy spot-check
│   ├── phase11_autoloop/      ← 6-hypothesis Claude-substitute autoloop
│   ├── phase12/               ← strict max-null audit
│   └── synthesis/             ← layered findings + final figures
├── logs/                      ← per-phase stdout
└── FINAL_SUMMARY.md           ← cross-pipeline synthesis (written at end)
```
