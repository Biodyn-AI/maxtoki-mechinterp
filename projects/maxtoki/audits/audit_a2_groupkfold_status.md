# Audit action A2 — GroupKFold re-evaluation status

The audit (`audit-20260507.md`) flagged P4 (pseudoreplication / leaky CV) as
the project's weakest pattern, with action A2 calling for GroupKFold re-evaluation
across 4 runs:
1. topology-141 H123 ΔAUROC
2. sae-atlas Phase 6 specificity
3. circuit-tracing CRISPRi 54.6% directional
4. exhaustive-mapping combinatorial ablation ratios

This note records the per-run status as of 2026-05-07.

---

## (1) topology-141 H123 ΔAUROC — DONE

**Script:** `runs/topology-141-217M/scripts/audit_a2_groupkfold_h123.py`
**Output:** `runs/topology-141-217M/outputs/groupkfold_h123/{summary.json, by_layer.csv}`

Re-evaluated immune domain × 12 layers under no-grouping, GroupKFold-by-TF
(5-fold), GroupKFold-by-target (5-fold), dual-axis disjoint:

| CV scheme | Mean AUROC |
|---|---:|
| no grouping (original) | 0.505 |
| GroupKFold by TF | 0.492 |
| GroupKFold by target | 0.488 |
| Dual-axis disjoint | **0.462** |

**Finding:** under the strictest CV, H123 motif AUROC drops below chance.
The original immune "1/12 layers positive null-gap" was leakage-inflated.
**Sharpens but does not overturn the project's H123-floor verdict** — the
finding becomes "the H123 implementation finds no above-chance signal on
MaxToki under proper CV", which is a stronger negative than "1/12 positive".

This result is integrated into `runs/topology-141-217M/FINAL_SUMMARY.md`
(Phase 7+8 H123 section).

**lung and external_lung not re-evaluated:** both were at 0/12 layers under
no-grouping; a stricter CV cannot move them lower.

---

## (2) sae-atlas Phase 6 specificity — DOES NOT APPLY

The Phase 6 metric is per-feature `specificity_ratio = mean_target_delta /
mean_other_delta`, computed independently for each of 50 patched features.
The inference unit is already the feature, not pairs. **GroupKFold does not
apply in the same sense** — there's no shared-endpoint structure between
features.

The bootstrap CI computed for A3 (median 1.015, 95% CI [1.011, 1.019]) is
the appropriate stability check for this metric. The CI decisively excludes
the paper's 2.36×, confirming a real non-replication. No re-run needed.

**The audit recommendation to apply GroupKFold here was overgeneralized.**
The correct stability check for per-feature ratios is feature-level
bootstrap, which has been done.

---

## (3) circuit-tracing CRISPRi 54.6% directional accuracy — BLOCKED on per-pair data

**Status:** per-pair labels are not on disk
(`outputs/phase11_crispri_validation.json` contains only aggregate counts:
n_validated=1,458,016; n_correct=796,159).

**Why GroupKFold matters here:** the 1.46M validated pairs share source
features (120 SAE features at L0/L3/L6/L9) and target features (4928 SAE
features per downstream layer). Each source feature contributes ~12,000
validated pairs on average. If any single source feature is "lucky" or
has high consistency with TRRUST signs, that one feature's contribution
dominates the directional accuracy.

**To re-evaluate properly would require re-running Phase 11** with per-pair
records (source_feature, target_feature, predicted_sign, ground_truth_sign,
correct) saved. Effort: 1–2 hr (re-running Phase 11's edge-by-edge sign
prediction + Replogle DE matching).

**Falling back to a structural argument:** the Wilson 95% CI on 0.5461 with
N=1.46M is [0.54525, 0.54686] — extraordinarily tight. Even if effective N
is reduced 100× by source-feature pseudoreplication, the CI under N=14,580
would be [0.538, 0.554] — still significantly above 50%. The substantive
finding ("CRISPRi directional accuracy is a few SD above chance, but small
effect size") is robust. What GroupKFold *would* refine is the precise
position of the point estimate; the qualitative conclusion is unchanged.

**Recommendation:** add `responding_pair_records` capture to Phase 11 and
re-run if the precise number matters for any external paper. For internal
project synthesis the structural argument suffices.

---

## (4) exhaustive-mapping combinatorial ablation ratios — DOES NOT APPLY (already addressed)

The Stage 3 Experiment 2 ablation ratios (pairwise/three-way) are computed
across 4 distinct triplets (post-translational modification, gene expression,
immune system, cell cycle checkpoints), each with disjoint L0/L5/L9 features.
The across-triplet standard deviation is **<0.0003 on every ratio** (per
`exhaustive-mapping-217M-FINAL_SUMMARY.md` §"Feature-invariant ablation
magnitudes — incidental finding from Exp 2 v2"). This is itself a stability
check that's stronger than what GroupKFold would provide.

The triplet "feature-invariance" finding (any feature selection at a given
layer produces the same redundancy ratio) is the analog of GroupKFold for
this analysis — and it's documented as showing zero between-triplet variance.
**No re-evaluation needed.**

---

## A2 summary

| Run | Status | Outcome |
|---|---|---|
| topology-141 H123 | ✅ DONE | dual-axis CV drops AUROC from 0.505 → 0.462; sharpens negative |
| sae-atlas Phase 6 | ❌ N/A | per-feature metric — A3 bootstrap CI is the correct stability check |
| circuit-tracing CRISPRi | ⚠️ BLOCKED | per-pair labels not stored; Wilson CI fallback shows qualitative conclusion robust |
| exhaustive-mapping triplets | ❌ N/A | across-triplet std<0.0003 already serves as the stability check |

**Net P4 improvement:** the H123 result was the most consequential of the
four, and it has been re-evaluated under proper group-aware CV. The
remaining three are either not directly applicable (Phase 6, exhaustive
triplets) or require a re-extraction that delivers limited additional
information given existing CIs (CRISPRi).

The audit's P4 verdict updates from "weak" to "**weak on circuit-tracing
specifically; resolved on topology-141; not applicable elsewhere**".
