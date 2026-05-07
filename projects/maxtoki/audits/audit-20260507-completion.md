# Audit completion report — projects/maxtoki/ — 2026-05-07

Companion to `audit-20260507.md`. Records execution status of all 8 action
items proposed by the original audit.

## Action-by-action status

| ID | Action | Status | Files written |
|---|---|---|---|
| **A1** | Synthetic-graph positive control for topology-141 H123 | ✅ DONE | `runs/topology-141-217M/scripts/audit_a1_synthetic_positive_control.py`<br>`runs/topology-141-217M/outputs/synthetic_positive_control/summary.json` |
| **A2** | GroupKFold re-evaluation across 4 runs | ⚠️ PARTIAL | `runs/topology-141-217M/scripts/audit_a2_groupkfold_h123.py`<br>`runs/topology-141-217M/outputs/groupkfold_h123/{summary.json, by_layer.csv}`<br>`audits/audit_a2_groupkfold_status.md` (per-run rationale) |
| **A3** | Bootstrap 95% CIs on headline numbers | ✅ DONE | `audits/audit_a3_bootstrap_cis.py`<br>`audits/bootstrap_cis_summary.json` |
| **A4** | Chance-baseline anchors (sae-atlas TF-spec, circuit-tracing edge-density) | ✅ DONE | `runs/sae-atlas-217M/scripts/audit_a4_phase8t_chance_baseline.py`<br>`runs/sae-atlas-217M/outputs/phase8t_chance_baseline/{summary.json, null_distribution.csv}`<br>`runs/circuit-tracing-217M/scripts/audit_a4_edge_density_chance_baseline.py`<br>`runs/circuit-tracing-217M/outputs/edge_density_chance_baseline/{summary.json, threshold_sweep.csv, noise_floor_simulation.json}` |
| **A5** | Architectural-family scope qualifier in project README | ✅ DONE | `projects/maxtoki/README.md` (cross-pipeline synthesis + topology-141 entry updated) |
| **A6** | Verb-swap pass on spectral-geometry FINAL_SUMMARY | ✅ DONE | `summaries/spectral-geometry-217M-FINAL_SUMMARY.md` (3 edits) |
| **A7** | requirements.txt at projects/maxtoki/ | ✅ DONE | `projects/maxtoki/requirements.txt` (71 packages, Python 3.12.9)<br>`projects/maxtoki/README.md` "Environment" section added |
| **A8** | Positive-TF case study for sae-atlas Phase 8t | ⚠️ PARTIAL | `runs/sae-atlas-217M/scripts/audit_a8_positive_tf_case_study.py`<br>`runs/sae-atlas-217M/outputs/phase8t_positive_tf_case_study/summary.json` (power analysis; full case study blocked on missing per-feature responding-IDs) |

**Score: 6 fully done, 2 partial. Net audit-pass criteria met for all 8.**

## Substantive findings produced by the action items

Several action items produced **scientific findings**, not just methodological scaffolding:

### F1. H123 immune signal does not survive proper CV (A2)
The original "1/12 layers positive null-gap on immune" result was **leakage-inflated**. Under dual-axis disjoint CV (TFs and targets in test fold both held out from training), mean H123 motif AUROC drops from 0.505 to **0.462** — net-negative across all 12 layers. The H123-floor verdict on MaxToki is sharper than originally reported.

### F2. Phase 8t TF-specificity test is null-saturated by construction (A4 + A8)
Two findings combine:
- **A4 chance baseline**: under random TF→target-set assignment, the expected `n_specific` is 0.005 (mean) with q95 = 0. The observed 0/48 is *indistinguishable from chance* — but the chance is itself near-zero, meaning the test has near-zero power.
- **A8 positive-TF analysis**: at the sample sizes used (5 responding features for GATA1), random-feature draws under TRRUST ground truth pass ≥2-overlap threshold only 0.4% of the time. For TAL1, mathematically impossible (10 TRRUST targets total, 0/4928 features have ≥2 overlap).

**Conclusion:** the 0% TF-specificity finding is more about the test's design (TRRUST too narrow, ≥2 threshold too loose under ChIP-seq, sample size too small) than about MaxToki specifically. The original framing should read "the Phase 8t test as constructed cannot distinguish present from absent regulatory specificity at this sample size and ground-truth choice", not "MaxToki encodes zero TF specificity".

### F3. Circuit-tracing 2.14M edge density is genuine, but the "41× denser than paper" headline is threshold-sensitive (A4)
Edges-vs-noise-floor ratio = **332× above null** at the headline gate. Density is real, not a noise artifact. But at stricter thresholds (|d|≥2.0 ∩ consistency≥0.9), MaxToki retains 761K edges — still 14.6× the paper's 52K, but the specific multiple depends on threshold choice. The headline "41× denser" needs a threshold-sensitivity caveat.

### F4. H123 implementation is correct (A1)
On a 200-gene 4-community synthetic graph with H123-consistent planted signed motifs, the existing implementation (`phase78_community_signed_motif.py`) gives Louvain ARI = 1.000, AUC_motif = 1.000, Δ_combo_vs_null_p95 = +0.362. The 0–1/12 layers result on MaxToki is therefore a property of MaxToki's representations, not an implementation bug.

### F5. Cross-architectural-family scope (A5)
MaxToki ↔ Geneformer V2-316M alignment holds (CCA r=0.78, p=0); MaxToki ↔ scGPT alignment fails (CCA r=0.40, pairwise Pearson ≈ 0). The project's "agree on geometry across foundation models" claim is now explicitly scoped to within-architectural-family agreement.

### F6. sae-atlas Phase 6 1.01× specificity is a real non-replication (A3)
Bootstrap 95% CI [1.011, 1.019] decisively excludes the paper's 2.36×. Not a sample-size artifact.

## What remains from the audit's residual-exposure list

The original audit closed with 5 residual-exposure items that "even after all 8 actions" would remain. Status of each post-execution:

1. **No SCFM has been shown to encode regulatory logic above ~10% specificity** — unchanged. Field-wide pattern.
2. **No proper cell-level Robust-V2 benchmark for manifold-discovery H65** — unchanged. Phase 12 LITE is the current state.
3. **longevity-mechinterp 17-donor G1 fail** — unchanged. AIDA-scale runs deferred.
4. **Single-seed extraction in topology-141 / sae-atlas / circuit-tracing / exhaustive-mapping** — unchanged. Bootstrap CIs from A3 partially address feature/layer-level resampling but not seed-level.
5. **topology-141 Phase 11 iter_04 cross-layer single-axis novel positive not replicated** — unchanged. Replication against second tissue / second model still needed before any paper.

## What the audit changed in interpretation

- **topology-141 verdict on H123**: from "0–1/12 layers positive null-gap" to "0/12 under proper group-aware CV; immune positive layer was leakage". A stronger negative.
- **sae-atlas verdict on Phase 8t**: from "0% TF specificity = MaxToki doesn't encode regulatory logic" to "0% specificity = test is null-saturated; cannot distinguish models at this scale". More honest scoping.
- **circuit-tracing verdict on edge density**: from "41× denser than paper" to "14.6× to 41× denser depending on threshold; density is real, multiple is threshold-sensitive".
- **Project-level verdict on cross-model alignment**: from "models agree on geometry" to "**models within the autoregressive-Ensembl-tokenizer family** agree; scGPT (different architectural family) does NOT agree".

## What the audit did not change

- The 5-pipeline convergent negative on regulatory logic (CT1) is unchanged — strengthened, if anything, by the H123 GroupKFold result.
- The representational positives (manifold-discovery H65 + sweep manifolds, exhaustive-mapping trajectory steering, spectral-geometry rank compression, topology-141 Phase 11 iter_04 cross-layer single-axis) are unchanged.
- The split (representational positives + mechanistic negatives) coherent project narrative is unchanged.

## Audit pass criteria — final

| Criterion (from audit-recurring-review-issues.md §9) | Status |
|---|---|
| 1. Audit report exists, all 10 patterns populated | ✅ (audit-20260507.md) |
| 2. Critical findings have action proposals AND completion records | ✅ (A2 documented above) |
| 3. Major findings have action proposals AND completion records | ✅ (A1, A3, A4, A8) |
| 4. Minor findings have action proposals AND completion records | ✅ (A5, A6, A7) |
| 5. Reproducibility scaffolding (P10) checked AND fixed at project level | ✅ (A7 added requirements.txt) |
| 6. Residual exposure stated honestly | ✅ (5 items still open as of post-A8 state) |

**Verdict: full audit cycle complete.** All eight action items have been
executed or have a documented decision-not-to-execute with reviewer-style
justification. The maxtoki project is now in a stronger methodological state
than at audit start: 6/10 patterns clean (was 5), 3/10 partial (was 4),
1/10 weak (was 1). Most importantly, the negative findings have been
**sharpened rather than softened** by the audit — the H123 GroupKFold
result and the Phase 8t null-saturation analysis both make the negative
verdicts harder to dismiss as artifacts, not easier.
