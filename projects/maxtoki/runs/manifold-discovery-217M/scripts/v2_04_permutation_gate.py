"""v2 step 4: the fifth pre-registered gate (blocked-permutation p), as frozen in reports/quality_gates_spec.json:
    blocked_permutation_p_max = 0.001, n_permutations_min = 2000,
    permutation_blocking = "by_donor_and_tissue", p_value_method = "two_sided".
The spec does not name the test statistic. We use the Spearman correlation between the head's pairwise
arc-cos latent distances and the H65 ruler over all anchor pairs (the run's 'global_correlation'; the source
paper attaches its blocked p to the analogous 'geodesic-biological correlation').
Null: the hema_stage labels are permuted among the anchors of each donor x tissue block; the ruler is rebuilt
from the permuted labels on the 34-stage DAG; the statistic is recomputed.

  frozen panels (external, zero-shot, lung_nonhema): the deployed head is fixed (it never saw these panels),
      so only the ruler changes. B = 10,000 permutations (rng default_rng([777, panel_code, b])).
  internal panel: the head was fitted to the internal ruler, so each permutation RE-FITS the head
      (v2_02_refit_pool.py, 'perm' tasks, B = 2,000); this script only summarises those rows.

Two-sided p (primary): p = min(1, 2 * min(p_hi, p_lo)), p_hi = (1 + #{T_b >= T_obs}) / (B + 1),
p_lo = (1 + #{T_b <= T_obs}) / (B + 1). Also reported: centred |T - mean(null)| version.
usage: python v2_04_permutation_gate.py frozen | internal
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, "<REPO_ROOT>/projects/maxtoki/runs/manifold-discovery-217M/scripts")
import json, time
import numpy as np, pandas as pd, torch
from scipy.stats import rankdata
from v2_common import *

SPEC = json.loads((REP / "quality_gates_spec.json").read_text())
PERM_SEED = 777
PANEL_CODE = {"external": 1, "zeroshot": 2, "lung_nonhema": 3}


def pvals(obs, null):
    null = np.asarray(null, float); B = len(null)
    p_hi = (1 + np.sum(null >= obs)) / (B + 1); p_lo = (1 + np.sum(null <= obs)) / (B + 1)
    c = null.mean()
    p_abs = (1 + np.sum(np.abs(null - c) >= abs(obs - c))) / (B + 1)
    return {"B": int(B), "observed": float(obs), "null_mean": float(c), "null_sd": float(null.std(ddof=1)),
            "null_min": float(null.min()), "null_max": float(null.max()),
            "null_q999": float(np.quantile(null, 0.999)) if B >= 1000 else None,
            "z_vs_null": float((obs - c) / null.std(ddof=1)),
            "n_null_ge_obs": int(np.sum(null >= obs)), "n_null_le_obs": int(np.sum(null <= obs)),
            "p_one_sided_upper": float(p_hi), "p_two_sided_doubled": float(min(1.0, 2 * min(p_hi, p_lo))),
            "p_two_sided_centred": float(p_abs),
            "min_attainable_p_two_sided_doubled": float(min(1.0, 2 / (B + 1))),
            "passes_gate_p_le_0.001": bool(min(1.0, 2 * min(p_hi, p_lo)) <= SPEC["gates"]["blocked_permutation_p_max"]
                                           and B >= SPEC["gates"]["n_permutations_min"])}


def frozen():
    nodes, sidx, T = stage_distance_table()
    FZ = Frozen()
    head = LETHead(FZ.F_int.shape[1], LATENT_DIM)
    head.load_state_dict(torch.load(ART / "heads/let_anchor_internal.pt", map_location="cpu")); head.eval()
    out = {}
    for panel in ["external", "zeroshot", "lung_nonhema"]:
        t0 = time.time()
        f, d, m = FZ.features(panel)
        z = head_z(head, f)
        stages = m["hema_stage"].astype(str).to_numpy()
        D0 = ruler_from_stages(stages, sidx, T)
        assert np.allclose(D0, d), f"{panel}: rebuilt ruler differs from saved d_target"
        iu = np.triu_indices(len(d), 1)
        rx = rankdata(arccos_dist(z)[iu]); rx = (rx - rx.mean()) / rx.std()
        def stat(D):
            ry = rankdata(D[iu]); ry = (ry - ry.mean()) / ry.std()
            return float(np.mean(rx * ry))
        obs = stat(D0)
        blocks = (m["donor_id"].astype(str) + "|" + m["tissue"].astype(str)).to_numpy()
        ub = pd.unique(blocks); members = [np.where(blocks == g)[0] for g in ub]
        B = 10000; null = np.empty(B)
        for b in range(B):
            rng = np.random.default_rng([PERM_SEED, PANEL_CODE[panel], b])
            s = stages.copy()
            for ii in members:
                if len(ii) > 1:
                    s[ii] = stages[rng.permutation(ii)]
            null[b] = stat(ruler_from_stages(s, sidx, T))
        sizes = pd.Series(blocks).value_counts()
        R = pvals(obs, null)
        R.update({"n_anchors": int(len(d)), "n_blocks": int(len(ub)), "n_singleton_blocks": int((sizes == 1).sum()),
                  "largest_block": int(sizes.max()), "head": "frozen deployed Phase-5 head (saved weights)",
                  "p_first_2000": pvals(obs, null[:2000]), "seconds": time.time() - t0})
        np.save(OUT / f"perm_null_{panel}.npy", null)
        out[panel] = R
        print(panel, json.dumps({k: R[k] for k in ["observed", "null_mean", "null_sd", "null_max", "p_two_sided_doubled",
                                                   "passes_gate_p_le_0.001", "n_blocks"]}), f"{time.time()-t0:.0f}s", flush=True)
    (OUT / "v2_04_permutation_frozen.json").write_text(json.dumps(out, indent=2))
    write_run_config("v2_04_permutation_frozen", CORE_INPUTS, {"perm_seed": PERM_SEED, "B": 10000,
                                                                "statistic": "Spearman(arc-cos latent, H65 ruler), all pairs"})


def internal():
    rows = [json.loads(l) for l in (OUT / "refit_pool/results.jsonl").read_text().splitlines() if l.strip()]
    obs_row = [r for r in rows if r["id"] == "repro_full|H65"][0]
    perm = sorted([r for r in rows if r["kind"] == "perm"], key=lambda r: r["b"])
    null = np.array([r["stat"] for r in perm]); null_trust = np.array([r["trust"] for r in perm])
    R = {"statistic": "in-sample Spearman(arc-cos latent, ruler), head re-fitted on each permuted ruler",
         "global": pvals(obs_row["global_rho"], null),
         "trust_same_permutations": pvals(obs_row["trust"], null_trust),
         "n_labels_changed_median": float(np.median([r["n_changed"] for r in perm])),
         "perm_indices_done": [int(perm[0]["b"]), int(perm[-1]["b"])] if perm else None}
    # sensitivity: the same H65 fit with torch seeds 1..10 (optimiser noise) compared with this (seed-42) null
    seeds = sorted([r for r in rows if r["kind"] == "seed" and r["ruler"] == "H65"], key=lambda r: r["seed"])
    R["seed_sensitivity"] = [{"seed": r["seed"], "observed_global": r["global_rho"],
                              "n_null_ge": int(np.sum(null >= r["global_rho"])),
                              "p_two_sided_doubled_vs_seed42_null": pvals(r["global_rho"], null)["p_two_sided_doubled"]}
                             for r in seeds]
    R["margin_observed_minus_null_max"] = float(obs_row["global_rho"] - null.max())
    null_row = [r for r in rows if r["id"] == "repro_full|null"]
    if null_row:
        R["within_branch_null_ruler_in_sample_global"] = null_row[0]["global_rho"]
        R["within_branch_null_ruler_trust"] = null_row[0]["trust"]
    (OUT / "v2_04_permutation_internal.json").write_text(json.dumps(R, indent=2))
    write_run_config("v2_04_permutation_internal", [OUT / "refit_pool/results.jsonl"] + CORE_INPUTS,
                     {"perm_seed": PERM_SEED, "B": len(null)})
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: v[kk] for kk in ["B", "observed", "null_mean", "null_max", "p_two_sided_doubled", "passes_gate_p_le_0.001"]}) for k, v in R.items()}, indent=1))


if __name__ == "__main__":
    {"frozen": frozen, "internal": internal}[sys.argv[1]]()
