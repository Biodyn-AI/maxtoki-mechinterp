"""Audit action A8 — positive-TF case study with ChIP-seq direct targets.

Evaluate whether the Phase 8t test would discriminate ChIP-validated
direct-target overlap from random gene-set overlap, for the canonical
positive-control TFs GATA1, MYC, TAL1 in K562.

**Blocker.** Phase 8t (`outputs/phase8_true/perturbation_response.csv`)
recorded only the *count* of responding features per TF, not the feature
*IDs*. Of the three target TFs, only GATA1 was among the 100 perturbations
tested (5 responding features). MYC and TAL1 were not in the selected
target set. Without the responding-feature IDs, the original specificity
test (≥2 of top-20 overlap with ChIP target set) cannot be re-run.

**What this script does instead** (a feasibility / power analysis):

  1. Loads dorothea ChIP-seq direct targets for GATA1, MYC, TAL1.
  2. For each TF, loads the L5 SAE feature catalog (4928 features × top-20
     genes each).
  3. For varying K = number of "responding" features (1..20), simulates 1000
     random K-feature draws and computes the fraction of draws where ≥1
     feature's top-20 overlaps the TF's ChIP targets by ≥2.
  4. Reports the sensitivity-as-function-of-K curve. This characterises the
     test's *power* against the planted-signal scenario "K features really
     do correspond to this TF's regulation".

The output is **not** the positive-control the audit originally requested
(which would require re-running Phase 8t with feature IDs saved). It IS the
power calibration that the existing-data audit can produce; it tells us at
what K count the Phase 8t test would, in principle, detect specificity if
the signal were really there.

Recommended next step: re-run Phase 8t with `responding_feature_ids` saved
per TF, then re-evaluate specificity using the ChIP-seq targets directly.
A patch in `remaining_phases.py` adds 2 lines (save `responding` list per
target row to a JSON sidecar). Effort: 30 min plus re-extraction time
(~30 min for 3 TFs with the existing tokenizer infrastructure).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[3]
RUN = PROJ / "runs/sae-atlas-217M"
OUT = RUN / "outputs/phase8t_positive_tf_case_study"
OUT.mkdir(parents=True, exist_ok=True)

CHIPSEQ_TSV = Path("<DATA_ROOT>/biodyn-work/single_cell_mechinterp/external/networks/dorothea_chipseq_human.tsv")
PROBE_LAYER = 5
TARGET_TFS = ["GATA1", "MYC", "TAL1"]
K_VALUES = [1, 2, 3, 5, 7, 10, 15, 20]
N_SIM = 1000
OVERLAP_THRESHOLD = 2
SEED = 42

print("[A8] Loading dorothea ChIP-seq direct targets...")
chip = pd.read_csv(CHIPSEQ_TSV, sep="\t")
chip["source"] = chip["source"].str.upper()
chip["target"] = chip["target"].str.upper()
print(f"  ChIP-seq edges: {len(chip):,}; unique sources: {chip['source'].nunique()}")

chip_targets = {tf: set(chip[chip["source"] == tf]["target"]) for tf in TARGET_TFS}
for tf, tgts in chip_targets.items():
    print(f"  {tf}: {len(tgts)} ChIP-seq direct targets")

print("\n[A8] Loading L5 SAE feature catalog...")
catalog_path = RUN / f"outputs/phase2/layer_{PROBE_LAYER:02d}/feature_catalog.json"
with open(catalog_path) as f:
    catalog = json.load(f)
feat_top20 = {c["feature_id"]: set(g.upper() for g in c["top20_genes"]) for c in catalog}
print(f"  Loaded {len(feat_top20)} features × top-20 genes each")

# How many features have ≥2 top-20 overlap with each TF's ChIP targets?
print("\n[A8] How many of the 4928 L5 features have ≥2 top-20 overlap with each TF's ChIP-seq targets?")
n_feat_specific_per_tf = {}
for tf in TARGET_TFS:
    tgts = chip_targets[tf]
    n_specific = sum(1 for fi, top20 in feat_top20.items()
                     if len(top20 & tgts) >= OVERLAP_THRESHOLD)
    n_feat_specific_per_tf[tf] = n_specific
    print(f"  {tf}: {n_specific}/{len(feat_top20)} features have ≥{OVERLAP_THRESHOLD} ChIP-seq target overlap "
          f"({100 * n_specific / len(feat_top20):.1f}%)")

# Power simulation: for varying K, what's P(any of K random features overlaps ≥2)?
print("\n[A8] Power simulation: P(detected as specific | K responding features)")
rng = np.random.default_rng(SEED)
all_features = list(feat_top20.keys())
power_curve = {}
for tf in TARGET_TFS:
    tgts = chip_targets[tf]
    power_per_K = {}
    for K in K_VALUES:
        if K > len(all_features):
            continue
        n_detected = 0
        for _ in range(N_SIM):
            samp = rng.choice(all_features, size=K, replace=False)
            if any(len(feat_top20[fi] & tgts) >= OVERLAP_THRESHOLD for fi in samp):
                n_detected += 1
        power_per_K[K] = n_detected / N_SIM
    power_curve[tf] = power_per_K
    print(f"  {tf}: " + ", ".join(f"K={K}→{p:.3f}" for K, p in power_per_K.items()))

# What did Phase 8t actually observe for GATA1?
print("\n[A8] Phase 8t observed for GATA1: 5 responding features, 0 specific (under TRRUST).")
phase8t_gata1 = pd.read_csv(RUN / "outputs/phase8_true/perturbation_response.csv").query("target == 'GATA1'")
if len(phase8t_gata1):
    n_responding_gata1 = int(phase8t_gata1["n_responding"].iloc[0])
    chance_at_5 = power_curve["GATA1"].get(5, None)
    print(f"  GATA1 had {n_responding_gata1} responding features.")
    print(f"  Chance of detection under random 5-feature draw vs ChIP-seq: "
          f"{chance_at_5:.3f}")

# Comparison: what was the chance under TRRUST (the actual test)?
TRRUST_TSV = Path("<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv")
trrust = pd.read_csv(TRRUST_TSV, sep="\t", header=None, names=["tf", "target", "mode", "pmid"])
trrust["tf"] = trrust["tf"].str.upper()
trrust["target"] = trrust["target"].str.upper()
trrust_targets = {tf: set(trrust[trrust["tf"] == tf]["target"]) for tf in TARGET_TFS}
print("\n[A8] Comparison: TRRUST target sets")
for tf in TARGET_TFS:
    if tf in trrust_targets:
        tgts = trrust_targets[tf]
        n_specific = sum(1 for top20 in feat_top20.values()
                         if len(top20 & tgts) >= OVERLAP_THRESHOLD)
        print(f"  {tf}: {len(tgts)} TRRUST targets; {n_specific}/4928 features have ≥{OVERLAP_THRESHOLD} TRRUST-target overlap")

# Power comparison ChIP-seq vs TRRUST at GATA1's K=5
print("\n[A8] Power comparison at GATA1 K=5 (ChIP-seq vs TRRUST):")
gata1_chip_tgts = chip_targets["GATA1"]
gata1_trrust_tgts = trrust_targets.get("GATA1", set())
n_det_chip = 0; n_det_trrust = 0
for _ in range(N_SIM):
    samp = rng.choice(all_features, size=5, replace=False)
    if any(len(feat_top20[fi] & gata1_chip_tgts) >= OVERLAP_THRESHOLD for fi in samp):
        n_det_chip += 1
    if any(len(feat_top20[fi] & gata1_trrust_tgts) >= OVERLAP_THRESHOLD for fi in samp):
        n_det_trrust += 1
print(f"  ChIP-seq (n={len(gata1_chip_tgts)} targets):  P(any random K=5 feature ≥2 overlap) = {n_det_chip/N_SIM:.3f}")
print(f"  TRRUST    (n={len(gata1_trrust_tgts)} targets):  P(any random K=5 feature ≥2 overlap) = {n_det_trrust/N_SIM:.3f}")

summary = {
    "scope": "GATA1, MYC, TAL1 in K562 — power analysis of the Phase 8t test under ChIP-seq direct-target ground truth",
    "method": "feasibility / power simulation; original Phase 8t responding-feature IDs were not saved",
    "blocker_for_full_case_study": (
        "Phase 8t recorded only counts (n_responding) per TF, not feature IDs. "
        "MYC and TAL1 were not in the 100 selected perturbation targets. "
        "Re-running with feature-ID capture is required for the actual case study; "
        "this script characterises the test's *power* given current Phase 8t scope."
    ),
    "chip_seq_target_set_sizes": {tf: len(chip_targets[tf]) for tf in TARGET_TFS},
    "trrust_target_set_sizes": {tf: len(trrust_targets.get(tf, set())) for tf in TARGET_TFS},
    "n_features_with_>=2_chip_overlap": n_feat_specific_per_tf,
    "n_features_with_>=2_trrust_overlap": {
        tf: int(sum(1 for top20 in feat_top20.values()
                    if len(top20 & trrust_targets.get(tf, set())) >= OVERLAP_THRESHOLD))
        for tf in TARGET_TFS
    },
    "power_curve_chipseq": power_curve,
    "power_at_gata1_k=5": {
        "chipseq": n_det_chip / N_SIM,
        "trrust": n_det_trrust / N_SIM,
    },
    "interpretation": (
        f"At GATA1's actual responding-feature count (K=5), the test's power against "
        f"random 5-feature draws is "
        f"{n_det_chip/N_SIM:.3f} under ChIP-seq targets and "
        f"{n_det_trrust/N_SIM:.3f} under TRRUST targets. "
        "The ChIP-seq target set is much larger than TRRUST for GATA1, so the test "
        "has higher power against ChIP-seq ground truth — but the chance of a random "
        "feature draw passing the threshold is also higher, so a positive ChIP-seq "
        "result wouldn't necessarily be more meaningful than a positive TRRUST result. "
        "What matters is whether GATA1's ACTUAL responding features show overlap "
        "ABOVE this random baseline. Without the responding-feature IDs, we can't "
        "answer that question. Recommend re-running Phase 8t with feature-ID capture."
    ),
    "recommended_followup": (
        "Patch `remaining_phases.py` Phase 8t loop (line ~462) to save responding "
        "feature IDs alongside counts: change `pert_results.append({...})` to "
        "include `responding_feature_ids: responding[:]`. Re-run on GATA1, MYC, TAL1 "
        "(adding MYC and TAL1 to the target list at line ~370). Then evaluate "
        "specificity under ChIP-seq targets and compare to a 1000-resample random-"
        "feature null."
    ),
}
out_path = OUT / "summary.json"
out_path.write_text(json.dumps(summary, indent=2))
print(f"\n[A8] Wrote {out_path}")
