# longevity-mechinterp-donor-aware on MaxToki-217M

Run of [`pipelines/longevity-mechinterp-donor-aware.md`](../../../../pipelines/longevity-mechinterp-donor-aware.md)
against MaxToki-217M-HF (LlamaForCausalLM, 11 layers, hidden_size 1232).

## Scope decision (read first)

The source paper's primary discovery cohorts are **AIDA phase 1 v1 / v2** plus
external stress-test sets (Allen aging, Yazar, Tabula Sapiens immune). Of these,
only the Tabula Sapiens immune sets are present locally
(`<DATA_ROOT>/biodyn-work/single_cell_mechinterp/data/raw/`).

Per the pipeline's own §11 ("Porting to a new target model"):

> Run the pipeline at small scale first: `tabula_sapiens_immune_subset_20000.h5ad`
> → Stages 1–5 only. If Stage 1 balanced accuracy is < 0.5 + 0.05 above
> permutation null, stop — the model may not encode age at all, and the rest of
> the pipeline is moot.

We follow that recipe exactly. AIDA-scale deep-dive (Stages 6–11) is **deferred**
until at least one of those datasets is downloaded — the strict gate is meant
for ≥424-donor cohorts and the pipeline explicitly notes "the strict gate fails
below 30–50 donors". Tabula Sapiens immune subset has **24 donors**, so it is a
prototype cohort, not a deep-dive cohort. Source-paper Table 2 outcome is also
clear: even on AIDA v2 with full strict gates passing, composition-matched
re-runs fail 0/4 — the realistic upside is informative (likely-negative) signal,
not a positive headline claim.

The MethylGPT branch (Stage 12) does not apply to MaxToki: MaxToki is a single-
cell transcriptomics model, not a methylation model.

## Layer choice

MaxToki-217M has 11 transformer blocks. Prior MaxToki pipeline runs (spectral
geometry, attention-GRN, SAE atlas, manifold discovery) standardised on **L5**
as the canonical mid-stack residual tap. We do the same here so the resulting
representations are directly comparable to those prior runs, and use the
naming convention `maxtoki217m_layer_05` to match `<model>_layer_<NN>`.

## Run layout

```
runs/longevity-mechinterp-217M/
├── README.md                ← this file
├── scripts/
│   ├── maxtoki_runtime.py   ← MaxTokiContextualRuntime (extract_representations)
│   ├── run_stage1.py        ← donor-held-out probe + manifold + permutation null
│   ├── run_stage2.py        ← top-k geometry + within-cell-type controls (TBD if S1 passes)
│   └── run_stage3_sae.py    ← SAE pilot + donor-aware feature scoring (TBD if S1 passes)
├── outputs/
│   ├── stage0_smoke_test/   ← gene-coverage check + per-cell forward-pass timing
│   ├── stage1_<date>/       ← probe CSV, manifold CSV, permutation-null CSV, report.md
│   └── ...
└── notes/                   ← deviations, observations
```

## Parameter values (matching pipeline §8 defaults; deviations noted)

| Parameter | Pipeline default | This run | Why |
|---|---|---|---|
| `model_layer` | varies per model | `maxtoki217m_layer_05` | matches prior MaxToki pipelines (L5 canonical) |
| `max_genes_per_cell` | 1200 (scGPT) / 2048 (Geneformer) | **1024** | as-run value in `scripts/run_stage1.py` (`MAX_GENES_PER_CELL = 1024`); mean observed seq_len 998 → effectively no truncation |
| `n_splits` | 5 | 5 | |
| `n_bins` (age) | 3 | 3 | Tabula Sapiens immune has 19 age levels (22–74y) — quantile bins of 3 |
| `min_per_label` | 50 | 50 | |
| `n_perm` (S1 null) | ≥100 | 100 | |
| `target_n_cells` | n/a (Stage 1 cell budget) | 8000 | Subset file is 20k cells — 8k stays well within MPS memory while keeping ≥330 cells/donor avg |
| `max_cells_per_donor` | n/a | 600 | balanced subsample cap |
| `seed` | 42 | 42 | |

## Status

| Stage | Status | Headline |
|---|---|---|
| 0 — setup + smoke test | ✅ done | scripts wired; per-cell forward 0.21s on MPS, mean seq_len ≈998 tokens |
| 1 — frozen probes + manifold | ❌ FAIL (G1) | balanced acc **0.2755** (MaxToki L5) vs null mean **0.3320**, p=1.0 — observed is *below* null. HVG-PCA-50 baseline 0.1722. See [`outputs/stage1_20260505/report.md`](outputs/stage1_20260505/report.md). |
| 2 — manifold robustness | ⛔ skipped | gated on S1; S1 failed |
| 3 — SAE pilot | ⛔ skipped | gated on S1; S1 failed |
| 4 — cross-model convergence | ⛔ skipped | gated on S1; S1 failed |
| 5 — interventions | ⛔ skipped | gated on S1; S1 failed |
| 6–11 — deep-dive | **deferred** | requires AIDA v1/v2 (not local) — unaffected by the prototype S1 verdict |
| 12 — MethylGPT | **not applicable** | MaxToki is transcriptomics, not methylation |

See [`FINAL_SUMMARY.md`](FINAL_SUMMARY.md) for the full verdict and what it
does and does not imply about MaxToki + longevity.
