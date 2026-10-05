"""Step 5: internal branch-holdout 0.370 (and null -0.001) broken down per held-out branch,
plus the same with the dominant training donor TSP2 (187 of 290 anchors) removed.
Replicates grouped_holdout_corr in phase5_let_anchor.py:167-199 (refit per held-out branch)."""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, "<AGENT_TMP>/<SESSION_DIR>/d422dc33-6d6f-4b2b-bbf7-f274d8a82fae/scratchpad/understand/manifold_ci")
import json, time
import numpy as np, pandas as pd, torch
torch.set_num_threads(4)
from scipy.stats import spearmanr, t as tdist
tq = lambda k: float(tdist.ppf(0.975, k - 1))
from common import *

A_e, A_m, A_l, part = load_operators()
c, d, m = load_panel("internal")
d_null = np.load(ART / "anchors/d_target_internal_null_shuffled.npy")
F = build_pooled_drift(c, A_e, A_m, A_l, part)
F = (F - F.mean(0)) / (F.std(0) + 1e-6)   # standardised once on the full panel, as in Phase 5
br = branch_labels(m); don = m["donor_id"].astype(str).to_numpy()


def per_branch(keep, D):
    out = {}
    for g in pd.unique(br[keep]):
        test = keep[br[keep] == g]; train = keep[br[keep] != g]
        if len(test) < 3:
            continue
        head = train_let(F[train], D[np.ix_(train, train)])
        with torch.no_grad():
            z = head(torch.from_numpy(F[test]).float())[0].numpy()
        Dt = D[np.ix_(test, test)]; iu = np.triu_indices(len(test), 1)
        rho, _ = spearmanr(arccos_dist(z)[iu], Dt[iu])
        out[str(g)] = {"n_test": int(len(test)), "n_stages": int(m["hema_stage"].to_numpy()[test].__len__() and len(set(m["hema_stage"].to_numpy()[test]))),
                       "rho": None if np.isnan(rho) else float(rho)}
    vals = [v["rho"] for v in out.values() if v["rho"] is not None]
    return {"per_branch": out, "mean": float(np.mean(vals)), "n_scored": len(vals),
            "sd_across_branches": float(np.std(vals, ddof=1)),
            "t_ci95_over_branches": [float(np.mean(vals) - tq(len(vals)) * np.std(vals, ddof=1) / np.sqrt(len(vals))),
                                     float(np.mean(vals) + tq(len(vals)) * np.std(vals, ddof=1) / np.sqrt(len(vals)))]}


t0 = time.time(); R = {}
allidx = np.arange(len(d)); noTSP2 = np.where(don != "TSP2")[0]
name, lab = sys.argv[1], sys.argv[2]          # all_donors|without_TSP2 , H65|null
keep = {"all_donors": allidx, "without_TSP2": noTSP2}[name]
D = {"H65": d, "null": d_null}[lab]
R[f"{name}_{lab}"] = per_branch(keep, D)
print(name, lab, json.dumps(R[f"{name}_{lab}"]), f"{time.time()-t0:.0f}s", flush=True)
(OUT / f"05_internal_branch_{name}_{lab}.json").write_text(json.dumps(R, indent=2))
