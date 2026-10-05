# FINAL SUMMARY — longevity-mechinterp-donor-aware on MaxToki-217M

**Run:** `projects/maxtoki/runs/longevity-mechinterp-217M/`
**Pipeline:** [`pipelines/longevity-mechinterp-donor-aware.md`](../../../pipelines/longevity-mechinterp-donor-aware.md)
**Cohort:** Tabula Sapiens immune subset (20k cells subsampled to 2,974 across 17 donors, 38 cell types, 3 quantile age bins)
**Layer:** `maxtoki217m_layer_05` (canonical mid-stack residual, matching prior MaxToki pipelines)
**Wall clock:** ~92 min for Stage 1 representation extraction + probe + null
**Verdict:** ❌ **G1 FAIL — pipeline halts at Stage 1 per §11 item 4. Stages 2–5 not run.**

---

## Headline numbers

| Quantity                               | Value |
|----------------------------------------|------:|
| MaxToki L5 balanced accuracy (5-fold, donor-held-out) | **0.2755** ± 0.0346 |
| HVG-PCA-50 baseline balanced accuracy  | 0.1722 ± 0.0370 |
| Permutation null mean (n=100)          | **0.3320** ± 0.0103 |
| Null p95                               | 0.3487 |
| Right-tail *p* (observed > null)       | **1.0000** |
| Observed − null mean                   | **−0.0565** (5.65 pp **below** null) |
| PC1 ↔ age correlation                  | 0.0384 |
| Age-label silhouette                   | −0.0057 |

The probe never beats the permutation null. The age axis is not a dominant
geometric direction in the L5 representation.

## G1 gate

Pipeline §11 item 4:

> Run the pipeline at small scale first: `tabula_sapiens_immune_subset_20000.h5ad`
> → Stages 1–5 only. **If Stage 1 balanced accuracy is < 0.5 + 0.05 above
> permutation null, stop** — the model may not encode age at all, and the rest
> of the pipeline is moot.

Observed (0.2755) is **below** the null mean (0.3320), let alone 0.05 above it.
The gate fails decisively. Per §11, Stages 2–5 are not run on this cohort.

## What this does and does not imply

**It does imply** that on a 17-donor immune-only cohort with a 22–74y age range,
MaxToki-217M L5 representations do not separate cells by age in a way a frozen
linear/MLP probe can pick up across held-out donors. MaxToki L5 *does* beat the
trivial HVG-PCA-50 baseline (0.2755 vs 0.1722, +10.3 pp) — i.e. it is encoding
*something* beyond raw expression PCA — but that "something" is not the age
signal.

**It does NOT imply** any of the following:

1. **That MaxToki cannot do longevity.** The source paper's own Table 2 shows
   that *even on AIDA v2 with all strict gates passing*, composition-matched
   re-runs fail 0/4 — the pipeline is built assuming the headline result is
   likely-negative on causally rigorous controls. A small-cohort prototype
   failing the loose G1 gate is fully consistent with that prior.
2. **That the AIDA-scale deep-dive (Stages 6–11) is moot.** The strict null
   gate in this pipeline is built for ≥424-donor discovery cohorts. The pipeline
   itself notes the gate fails below 30–50 donors; this run had 17. The
   prototype's job was to confirm the gate fails at this scale before
   committing AIDA-scale compute, *not* to be a stand-in for the deep dive.
3. **That layer 5 is the wrong choice.** L5 is the canonical mid-stack tap used
   by every other MaxToki pipeline (`spectral-geometry-217M`,
   `attention-grn-217M`, `sae-atlas-217M`, `manifold-discovery-217M`,
   `topology-141-217M`); the result is comparable to those by design. A
   layer sweep (L0–L10) is the natural follow-up if and when AIDA cohorts are
   available.

## Why "below null" rather than "near null"

The permutation null sits *above* both observed representations (MaxToki and
HVG-PCA). With 17 donors and a strongly imbalanced age distribution
(1713 / 661 / 600 cells across the three quantile bins), a permuted classifier
that reverts to majority-class prediction on donor-held-out splits achieves
higher *balanced* accuracy than a learned classifier that overfits donor
identity on real labels. This is the documented small-cohort failure mode of
balanced accuracy under donor-held-out CV — exactly the regime the §11 item 4
gate is designed to detect and reject.

## Recommended next steps (when AIDA becomes available)

1. Pull AIDA phase 1 v1 / v2 (the source paper's primary discovery cohorts,
   ≥424 donors each) into `data/raw/`.
2. Re-run Stage 1 on AIDA v2 with `target_n_cells = 50000` and the existing
   `MAX_GENES_PER_CELL = 1024`. With ≥424 donors the donor-held-out null
   should drop into the 0.34–0.36 band and a real signal (if present) should
   exceed 0.55 cleanly.
3. Only if Stage 1 passes G1 on AIDA, run the full Stages 2–11 deep dive. The
   composition-matched re-run (§9 of the pipeline) is the load-bearing test;
   plan compute for that, not for the headline AUROC.
4. Stages 6–11's strict gates already use the same thresholds across all
   models — do **not** retune for MaxToki (the pipeline explicitly forbids
   this in §11 item 6).

## Run-time issues encountered (resolved)

- **Original failure:** the script crashed after all compute finished, while
  writing `report.md`, because `pandas.DataFrame.to_markdown()` requires the
  `tabulate` package which isn't installed in `projects/maxtoki/.venv`. All
  numerical artefacts (`probe_aggregate.csv`, `manifold_diagnostics.csv`,
  `permutation_null.csv`, `extraction_stats.json`, `maxtoki_layer_reps.npy`)
  were written successfully; only the markdown report was missing. The report
  has now been reconstructed manually from the CSVs (no recompute).
- **Resume cost:** zero. Compute artefacts on disk are authoritative.
- **Future-proofing:** if Stages 2–5 are run later, either install `tabulate`
  in the venv or replace the `to_markdown(...)` calls in the run scripts with
  manual table rendering.

## Cross-references

- Sibling MaxToki runs (same model, same L5 tap):
  `spectral-geometry-217M/`, `attention-grn-217M/`, `sae-atlas-217M/`,
  `manifold-discovery-217M/`, `topology-141-217M/`,
  `circuit-tracing-217M/`, `exhaustive-mapping-217M/`.
- Pipeline cross-reference: the negative verdict on attention-derived causal
  regulatory logic noted in `pipelines/attention-grn.md` and the SAE
  mega-pipeline is **not** invalidated by this result — those are different
  questions on different cohorts and different probes.
- Source paper artefacts for scGPT/Geneformer (Stage 4 reference):
  `repos/longevity-mechinterp/.../outputs/stage4_*/` — to be reused when an
  AIDA-scale Stage 1 passes G1 and Stage 4 cross-model convergence is computed.

## Status as of 2026-05-07

- Stage 0–1: complete. G1 fail recorded.
- Stages 2–5: not started (correctly halted by gate).
- Stages 6–11: **deferred** pending AIDA cohort availability, *unaffected* by
  this prototype run.
- Stage 12 (MethylGPT): N/A — MaxToki is single-cell transcriptomics.

## Data-availability blocker (audit residual exposure #7, 2026-05-07)

The audit at `projects/maxtoki/audits/audit-20260507.md` lists this run's
deferred AIDA-scale stages as residual exposure item #7. Status:

**Required data NOT present locally:**
- AIDA phase 1 v1 (~424 donors, primary discovery cohort)
- AIDA phase 1 v2 (broader donor pool, primary discovery cohort)
- Allen aging plasma cohort
- Yazar cohort

**Local data inventory** (`<DATA_ROOT>/biodyn-work/single_cell_mechinterp/data/raw/`):
- Tabula Sapiens immune subset (24 donors) — used for the G1 prototype here
- Tabula Sapiens immune full (varies)

**What's blocked.** Stages 6–11 (donor-aware probes at scale, full SAE,
cross-model pathway match, intervention, strict gate, reweighting,
composition-matched forward pass, multi-seed panel) all require ≥424-donor
cohorts. The strict null gate is documented to fail below 30–50 donors
(pipeline §11). The 17-donor TS prototype passes only the G1 *gate-feasibility
check*, not the substantive deep-dive analysis.

**What would unblock:**
1. Download AIDA phase 1 v1 from CELLxGENE / Sanger Cellular Genetics Programme
   (~70 GB; check current accession at https://aida-phase1.org/data).
2. Run `pipelines/longevity-mechinterp-donor-aware.md` Stage 1 against AIDA v2
   with `target_n_cells = 50000` and existing `MAX_GENES_PER_CELL = 1024`.
3. Only proceed to Stages 2–11 if Stage 1 passes G1 on AIDA (BA > 0.55 ± 0.03
   above null mean).

**Audit decision:** retain as residual exposure #7. Local replication is not
feasible without AIDA data acquisition. The prototype run's G1 fail is
*reported as the prototype it is*, not as a MaxToki deficit.
