"""v2 step 1: donor-level cluster bootstrap and leave-one-donor-out for the FROZEN-head H65 numbers
(external, zero-shot, lung_nonhema). The head is the deployed Phase-5 head (saved weights); only the
evaluation panel is resampled. Uncertainty from which 11 training donors fitted the head is NOT in this
script (see v2_02_refit_pool.py 'boot_full', two-level).

Resampling unit: donor. Each replicate draws the panel's donors with replacement (same number of donors),
and takes all anchors of each drawn donor. rng = numpy default_rng([4242, panel_code, r]).
Metrics per replicate (copies of one anchor are never paired and never count as neighbours):
  trust      trustworthiness k=15 between standardised 2,464-d pooled-drift features and the 10-d head output
  branch     mean over branches (>=3 anchors, non-constant ruler) of within-branch Spearman(arc-cos latent, ruler)
  branch_aw  the same, weighted by the number of anchors in each branch
  donor      mean over drawn donors (each draw is its own group) of within-donor Spearman
  global     Spearman over all pairs
Interval: percentile 2.5/97.5 of the replicates. Also: leave-one-donor-out values, jackknife SE, t interval.

Resumable: per-replicate rows are appended to outputs/v2_intervals/frozen_boot/<panel>.jsonl.
usage: python v2_01_frozen_donor_bootstrap.py <panel> <n_reps> <budget_seconds>
       python v2_01_frozen_donor_bootstrap.py summarize
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, "<REPO_ROOT>/projects/maxtoki/runs/manifold-discovery-217M/scripts")
import json, time
import numpy as np, pandas as pd, torch
from scipy.stats import t as tdist
from v2_common import *
from v2_common import summarize as summarize_stats

BDIR = OUT / "frozen_boot"; BDIR.mkdir(parents=True, exist_ok=True)
PANEL_CODE = {"external": 1, "zeroshot": 2, "lung_nonhema": 3}
BOOT_SEED = 4242
torch.set_num_threads(2)


def load(panel):
    FZ = Frozen()
    head = LETHead(FZ.F_int.shape[1], LATENT_DIM)
    head.load_state_dict(torch.load(ART / "heads/let_anchor_internal.pt", map_location="cpu")); head.eval()
    f, d, m = FZ.features(panel)
    z = head_z(head, f)
    return dict(DX=euclid(f), DZ=euclid(z), Dh=arccos_dist(z), d=d, br=branch_labels(m),
                don=m["donor_id"].astype(str).to_numpy(), n=len(d))


def metrics(P, idx, donor_group):
    DX = P["DX"][np.ix_(idx, idx)]; DZ = P["DZ"][np.ix_(idx, idx)]
    Dh = P["Dh"][np.ix_(idx, idx)]; Dt = P["d"][np.ix_(idx, idx)]
    t = trust_masked(DX, DZ, idx) if len(idx) > 2 * 15 else float("nan")  # sklearn needs k < n/2
    b, nb, rows = group_spearman(Dh, Dt, P["br"][idx], idx, detail=True)
    sc = [r for r in rows if r["rho"] is not None]
    baw = float(np.average([r["rho"] for r in sc], weights=[r["n_anchors"] for r in sc])) if sc else float("nan")
    dn, nd = group_spearman(Dh, Dt, donor_group, idx, skip=())
    g = global_spearman(Dh, Dt, idx)
    return {"trust": t, "branch": b, "branch_aw": baw, "n_branches": nb, "donor": dn, "n_donor_groups": nd,
            "global": g, "n_anchors": int(len(idx))}


def run(panel, n_reps, budget):
    T0 = time.time()
    P = load(panel)
    fn = BDIR / f"{panel}.jsonl"
    done = set()
    if fn.exists():
        done = {json.loads(l)["rep"] for l in fn.read_text().splitlines() if l.strip()}
    ud = np.array(sorted(set(P["don"])))
    mem = {u: np.where(P["don"] == u)[0] for u in ud}
    with open(fn, "a") as fh:
        for r in range(n_reps):
            if r in done:
                continue
            if time.time() - T0 > budget:
                break
            rng = np.random.default_rng([BOOT_SEED, PANEL_CODE[panel], r])
            pick = rng.choice(ud, size=len(ud), replace=True)
            idx = np.concatenate([mem[u] for u in pick])
            grp = np.concatenate([np.full(len(mem[u]), f"{u}#{k}") for k, u in enumerate(pick)])
            row = {"rep": r, **metrics(P, idx, grp)}
            fh.write(json.dumps(row) + "\n"); fh.flush()
    n_done = len({json.loads(l)["rep"] for l in fn.read_text().splitlines() if l.strip()})
    print(panel, "replicates done:", n_done, f"{time.time()-T0:.0f}s", flush=True)


def summarize():
    out = {}
    for panel in ["external", "zeroshot", "lung_nonhema"]:
        fn = BDIR / f"{panel}.jsonl"
        if not fn.exists():
            continue
        P = load(panel)
        allidx = np.arange(P["n"])
        obs = metrics(P, allidx, P["don"])
        rows = [json.loads(l) for l in fn.read_text().splitlines() if l.strip()]
        df = pd.DataFrame(rows).sort_values("rep")
        R = {"n_anchors": P["n"], "n_donors": int(len(set(P["don"]))), "observed": obs, "n_reps": int(len(df)),
             "resampling": "donor cluster bootstrap, donors drawn with replacement, all anchors of a drawn donor kept; "
                           "head frozen; percentile 2.5/97.5",
             "rng": f"numpy default_rng([{BOOT_SEED}, {PANEL_CODE[panel]}, rep])",
             "resample_size_range": [int(df.n_anchors.min()), int(df.n_anchors.max())],
             "n_branches_scored_range": [int(df.n_branches.min()), int(df.n_branches.max())]}
        gates = {"trust": GATE_TRUST, "branch": GATE_CORR, "branch_aw": GATE_CORR, "donor": GATE_CORR, "global": None}
        R["bootstrap"] = {k: summarize_arr(df[k].to_numpy(), gates[k]) for k in ["trust", "branch", "branch_aw", "donor", "global"]}
        # leave-one-donor-out
        ud = np.array(sorted(set(P["don"])))
        lodo = {}
        for u in ud:
            idx = np.where(P["don"] != u)[0]
            lodo[u] = {"n_anchors_left": int(len(idx)), **metrics(P, idx, P["don"][idx])}
        R["leave_one_donor_out"] = lodo
        G = len(ud); q = float(tdist.ppf(0.975, G - 1))
        jk = {}
        for k in ["trust", "branch", "branch_aw", "donor", "global"]:
            v = np.array([lodo[u][k] for u in ud], float); v = v[~np.isnan(v)]
            se = float(np.sqrt((len(v) - 1) / len(v) * ((v - v.mean()) ** 2).sum()))
            jk[k] = {"lodo_min": float(v.min()), "lodo_max": float(v.max()),
                     "lodo_argmin": str(ud[int(np.nanargmin([lodo[u][k] for u in ud]))]),
                     "lodo_argmax": str(ud[int(np.nanargmax([lodo[u][k] for u in ud]))]),
                     "jackknife_se": se, "t_ci95": [obs[k] - q * se, obs[k] + q * se], "df": len(v) - 1}
        R["jackknife"] = jk
        out[panel] = R
        print(panel, json.dumps({k: R["bootstrap"][k]["ci95_percentile"] for k in R["bootstrap"]}), flush=True)
    (OUT / "v2_01_frozen_donor_bootstrap.json").write_text(json.dumps(out, indent=2))
    write_run_config("v2_01_frozen_donor_bootstrap", CORE_INPUTS,
                     {"boot_seed": BOOT_SEED, "panel_codes": PANEL_CODE, "k_trust": 15})


def summarize_arr(a, gate):
    return summarize_stats(a, gate)


if __name__ == "__main__":
    if sys.argv[1] == "summarize":
        summarize()
    else:
        run(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]))
