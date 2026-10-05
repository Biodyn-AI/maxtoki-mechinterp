"""Audit action A4 — chance baseline for the Phase 8t TF-specificity 0/48 finding.

The Phase 8t (true CRISPRi) test calls a TF "specific" if at least one
significantly-responding SAE feature has ≥2 of its top-20 genes overlapping
that TF's TRRUST target set. The headline finding is **0/48 TFs specific**.

The audit asks: anchor 0/48 to a chance baseline. What's the probability of
0/48 under a TF-target-relabeling null (where each TF's "known targets" set
is replaced by a random other TF's target set, keeping the responding-feature
top-20 sets fixed)?

This isolates "are the responding features actually aligned with the TF's
known targets?" from "are the responding features aligned with any random
gene-set of similar size and composition?"

Outputs: outputs/phase8t_chance_baseline/{summary.json,null_distribution.csv}
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[3]
RUN = PROJ / "runs/sae-atlas-217M"
OUT = RUN / "outputs/phase8t_chance_baseline"
OUT.mkdir(parents=True, exist_ok=True)

TRRUST_TSV = Path("<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv")
PROBE_LAYER = 5
N_PERM = 1000
SEED = 42
OVERLAP_THRESHOLD = 2  # paper convention: ≥2 of top-20 overlap

print("[A4-phase8t] Loading observed Phase 8t results + TRRUST + feature catalog...")

# Observed
obs_df = pd.read_csv(RUN / "outputs/phase8_true/perturbation_response.csv")
print(f"  Loaded {len(obs_df)} target rows ({obs_df['is_tf'].sum()} TFs)")
print(f"  Detected: {obs_df['detected'].sum()}; TF-detected: {(obs_df['is_tf'] & obs_df['detected']).sum()}; specific: {obs_df['is_specific'].sum()}")

# TRRUST per-TF target set
trrust = pd.read_csv(TRRUST_TSV, sep="\t", header=None,
                     names=["tf", "target", "mode", "pmid"])
trrust["tf"] = trrust["tf"].str.upper()
trrust["target"] = trrust["target"].str.upper()
trrust_targets: dict[str, set[str]] = {}
for _, r in trrust.iterrows():
    trrust_targets.setdefault(r["tf"], set()).add(r["target"])
print(f"  TRRUST: {len(trrust_targets)} TFs with target sets, "
      f"median {int(np.median([len(v) for v in trrust_targets.values()]))} targets each")

# Feature catalog: top-20 genes per feature
cat_path = RUN / f"outputs/phase2/layer_{PROBE_LAYER:02d}/feature_catalog.json"
with open(cat_path) as f:
    catalog = json.load(f)
feat_top20 = {c["feature_id"]: set(g.upper() for g in c["top20_genes"]) for c in catalog}
print(f"  Loaded feature catalog: {len(feat_top20)} features at L{PROBE_LAYER}")

# We only have aggregate per-target counts in the CSV (n_responding), not the
# specific responding-feature ids. So we model the chance baseline at the
# aggregate level: for each detected TF, what's the probability that *any*
# responding feature's top-20 overlaps a random TRRUST target set by ≥2?
#
# Under the null, we assign each detected TF a random other TF's target set
# and ask whether ≥2 overlap occurs. We don't have the specific feature IDs
# per detected TF, so we approximate by: for each detected TF with N_resp
# responding features, sample N_resp random features from feat_top20 and
# ask whether any of their top-20 sets overlap a random TF's targets by ≥2.

detected_tfs = obs_df[(obs_df["is_tf"]) & (obs_df["detected"])].copy()
print(f"\n[A4-phase8t] Detected TFs: {list(detected_tfs['target'])}; n_responding: {detected_tfs['n_responding'].tolist()}")

if len(detected_tfs) == 0:
    print("  No detected TFs — chance baseline is 0/N_total trivially")
    summary = {
        "observed_n_tf_specific": 0,
        "n_tfs_tested": 48,
        "n_tfs_detected": 0,
        "chance_baseline": "trivially 0 (no detection)",
        "interpretation": "Specificity rate is 0 by construction when no TF is detected",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
else:
    rng = np.random.default_rng(SEED)
    all_features = list(feat_top20.keys())
    all_tfs = list(trrust_targets.keys())
    n_perm_specific = []

    for perm_i in range(N_PERM):
        n_specific_this_perm = 0
        for _, row in detected_tfs.iterrows():
            n_resp = int(row["n_responding"])
            # Sample n_resp random features
            samp_feats = rng.choice(all_features, size=n_resp, replace=False)
            # Sample a random TF and use its target set
            random_tf = rng.choice(all_tfs)
            random_targets = trrust_targets[random_tf]
            for fi in samp_feats:
                if len(feat_top20.get(fi, set()) & random_targets) >= OVERLAP_THRESHOLD:
                    n_specific_this_perm += 1
                    break
        n_perm_specific.append(n_specific_this_perm)

    n_perm_specific = np.array(n_perm_specific)
    obs_n_specific = int(obs_df["is_specific"].sum())

    # Empirical p-value: P(null ≤ observed)
    p_left = float((n_perm_specific <= obs_n_specific).mean())
    p_right = float((n_perm_specific >= obs_n_specific).mean())

    summary = {
        "observed_n_tf_specific": obs_n_specific,
        "n_tfs_tested": int(obs_df["is_tf"].sum()),
        "n_tfs_detected": int(len(detected_tfs)),
        "detected_tfs": detected_tfs["target"].tolist(),
        "n_responding_per_detected_tf": detected_tfs["n_responding"].astype(int).tolist(),
        "chance_baseline_method": "random TF-target-set assignment, sampled feature top-20s",
        "n_perm": N_PERM,
        "null_n_specific_mean": float(n_perm_specific.mean()),
        "null_n_specific_median": float(np.median(n_perm_specific)),
        "null_n_specific_std": float(n_perm_specific.std()),
        "null_n_specific_q05": float(np.percentile(n_perm_specific, 5)),
        "null_n_specific_q25": float(np.percentile(n_perm_specific, 25)),
        "null_n_specific_q75": float(np.percentile(n_perm_specific, 75)),
        "null_n_specific_q95": float(np.percentile(n_perm_specific, 95)),
        "p_observed_le_null": p_left,
        "p_observed_ge_null": p_right,
        "interpretation": (
            f"Under random TF-target-set assignment, n_tf_specific has null "
            f"mean {n_perm_specific.mean():.2f} (q5-q95: "
            f"{np.percentile(n_perm_specific,5):.0f}-{np.percentile(n_perm_specific,95):.0f}). "
            f"Observed = {obs_n_specific}. The 0/48 finding is "
            + ("INDISTINGUISHABLE from chance" if p_right > 0.05 and p_left > 0.05 else
               ("SIGNIFICANTLY BELOW chance" if p_left < 0.05 else "SIGNIFICANTLY ABOVE chance"))
            + " — meaning the TF→target alignment of responding features is "
            + ("not detectable above what random gene-set overlap produces" if p_right > 0.05 else "structurally distinct from random")
            + "."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame({"n_specific_under_null": n_perm_specific}).to_csv(
        OUT / "null_distribution.csv", index=False)

print(f"\n[A4-phase8t] Wrote {OUT}/summary.json")
print(json.dumps(summary, indent=2))
