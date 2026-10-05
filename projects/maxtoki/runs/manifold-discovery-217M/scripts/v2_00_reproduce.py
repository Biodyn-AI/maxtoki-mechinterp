"""v2 step 0: reproduce every deployed H65 number from saved artefacts, and break the
branch-holdout into its per-branch parts (item 3).

 - re-fits the Phase-5 head (same seed/settings) and checks it equals artifacts/heads/let_anchor_internal.pt
 - also re-fits with torch.set_num_threads(1) (the setting the bootstrap workers use) and records the difference
 - frozen-head gates on external / zero-shot / lung_nonhema: trust (sklearn and our masked version),
   random, donor, branch (per-branch detail), global Spearman
 - per-branch rows (anchors, stages, donors, pairs, rho) and plain / anchor-weighted / pair-weighted means
 - the INTERNAL refit numbers (branch- and donor-holdout, null) are computed as tasks in v2_02_refit_pool.py
   and summarised in v2_03_aggregate_refit.py
Outputs: outputs/v2_intervals/v2_00_reproduce.json, v2_00_per_branch.csv, z_<panel>_frozen.npy
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, "<REPO_ROOT>/projects/maxtoki/runs/manifold-discovery-217M/scripts")
import json, time
import numpy as np, pandas as pd, torch
from sklearn.manifold import trustworthiness
from v2_common import *

t_all = time.time()
FZ = Frozen()
res = {"threads_default": torch.get_num_threads()}

# ---------------------------------------------------------------- head re-fit check
t0 = time.time(); head = FZ.fit_head(); t_fit = time.time() - t0
saved = torch.load(ART / "heads/let_anchor_internal.pt", map_location="cpu")
res["head_fit_seconds_default_threads"] = t_fit
res["refit_vs_saved_maxabs_default_threads"] = {k: float((v - saved[k]).abs().max()) for k, v in head.state_dict().items()}
torch.set_num_threads(1)
t0 = time.time(); head1 = train_let(FZ.F_int, FZ.d_int); t_fit1 = time.time() - t0
res["head_fit_seconds_1_thread"] = t_fit1
res["refit_vs_saved_maxabs_1_thread"] = {k: float((v - saved[k]).abs().max()) for k, v in head1.state_dict().items()}
torch.set_num_threads(res["threads_default"])
print(json.dumps({k: res[k] for k in res if k.startswith(("head_fit", "refit"))}, indent=1), flush=True)
# use the SAVED weights for all frozen-head scoring (they are the deployed head)
head.load_state_dict(saved); head.eval()

pub = {"external": json.loads((REP / "external_validation_external.json").read_text()),
       "zeroshot": json.loads((REP / "zeroshot_transfer_anchor_head.json").read_text()),
       "lung_nonhema": json.loads((REP / "external_validation_lung_nonhema.json").read_text())}

rows_branch = []
for panel in ["external", "zeroshot", "lung_nonhema"]:
    f, d, m = FZ.features(panel)
    z = head_z(head, f)
    np.save(OUT / f"z_{panel}_frozen.npy", z)
    n = len(d); origin = np.arange(n)
    Dh = arccos_dist(z)
    br = branch_labels(m); don = m["donor_id"].astype(str).to_numpy()
    t_sk = float(trustworthiness(f, z, n_neighbors=15))
    t_my = trust_masked(euclid(f), euclid(z), origin, 15)
    b_mean, nb, b_rows = group_spearman(Dh, d, br, origin, detail=True)
    dn_mean, nd, dn_rows = group_spearman(Dh, d, don, origin, skip=(), detail=True)
    g = global_spearman(Dh, d, origin)
    r = random_holdout_frozen(z, d)
    scored = [x for x in b_rows if x["rho"] is not None]
    w = np.array([x["n_anchors"] for x in scored], float); v = np.array([x["rho"] for x in scored])
    wp = np.array([x["n_pairs"] for x in scored], float)
    res[panel] = {
        "n_anchors": int(n), "n_donors": int(len(set(don))), "n_tissues": int(m["tissue"].nunique()),
        "trust_sklearn_k15": t_sk, "trust_masked_k15": t_my,
        "random_holdout_frozen": r, "donor_within_mean": dn_mean, "n_donor_groups_scored": nd,
        "branch_within_mean": b_mean, "n_branches_scored": nb,
        "branch_within_anchor_weighted_mean": float((w * v).sum() / w.sum()),
        "branch_within_pair_weighted_mean": float((wp * v).sum() / wp.sum()),
        "global_spearman": g,
        "published": {k: pub[panel][k] for k in ["trustworthiness", "random_holdout", "donor_holdout",
                                                  "branch_holdout", "global_correlation"]},
        "per_donor": dn_rows,
    }
    for x in b_rows:
        stages = sorted(set(m["hema_stage"][br == x["group"]]))
        rows_branch.append({"panel": panel, "ruler": "H65", "mode": "frozen_head_within_branch",
                            "branch": x["group"], "n_anchors": x["n_anchors"], "n_stages": len(stages),
                            "stages": "|".join(stages),
                            "n_donors": int(len(set(don[br == x["group"]]))), "n_pairs": x["n_pairs"],
                            "rho": x["rho"]})
    print(panel, json.dumps({k: v for k, v in res[panel].items() if k != "per_donor"}, indent=1), flush=True)

# internal refit parts (branch/donor holdout, null) are computed as tasks in v2_02_refit_pool.py

qg = json.loads((REP / "quality_gates_let_anchor.json").read_text())
res["published_internal"] = qg
pd.DataFrame(rows_branch).to_csv(OUT / "v2_00_per_branch.csv", index=False)
res["elapsed_s"] = time.time() - t_all
(OUT / "v2_00_reproduce.json").write_text(json.dumps(res, indent=2))
write_run_config("v2_00_reproduce", CORE_INPUTS, {"seed_head": SEED, "notes": "exact reproduction + per-branch breakdown"})
print("done", f"{time.time()-t_all:.0f}s")
