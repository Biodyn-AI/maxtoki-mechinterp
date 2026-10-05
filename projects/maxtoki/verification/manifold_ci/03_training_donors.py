"""Step 3: add the uncertainty that comes from WHICH 11 internal (training) donors fitted the head.

mode 'lotdo'  : leave-one-training-donor-out. Refit head without donor u (11 refits), evaluate
                on the full external and zero-shot panels.
mode 'twolevel': two-level donor bootstrap. Each rep resamples the 11 internal donors with
                replacement, refits the head (same seed/hyperparameters as Phase 5), then
                resamples the external (and zero-shot) donors with replacement. Copies of one
                anchor are never paired in the metrics.
usage: python 03_training_donors.py lotdo
       python 03_training_donors.py twolevel <rng_seed> <n_reps>
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, "<AGENT_TMP>/<SESSION_DIR>/d422dc33-6d6f-4b2b-bbf7-f274d8a82fae/scratchpad/understand/manifold_ci")
import json, time
import numpy as np, torch
torch.set_num_threads(4)
from common import *

mode = sys.argv[1]
A_e, A_m, A_l, part = load_operators()
c_int, d_int, m_int = load_panel("internal")
F_int_raw = build_pooled_drift(c_int, A_e, A_m, A_l, part)
don_int = m_int["donor_id"].astype(str).to_numpy()
ud_int = np.array(sorted(set(don_int)))

panels = {}
for p in ["external", "zeroshot"]:
    c, d, m = load_panel(p)
    panels[p] = dict(F_raw=build_pooled_drift(c, A_e, A_m, A_l, part), d=d,
                     br=branch_labels(m), don=m["donor_id"].astype(str).to_numpy())


def fit_head(idx_int):
    f = F_int_raw[idx_int]
    mu = f.mean(0); sd = f.std(0) + 1e-6
    head = train_let((f - mu) / sd, d_int[np.ix_(idx_int, idx_int)])
    return head, mu, sd


def eval_panel(p, head, mu, sd, idx=None):
    P = panels[p]
    f = (P["F_raw"] - mu) / sd
    with torch.no_grad():
        z = head(torch.from_numpy(f).float())[0].numpy()
    if idx is None:
        idx = np.arange(len(P["d"]))
    fx, zx = f[idx], z[idx]
    t = trust_masked(euclid(fx), euclid(zx), idx)
    b, _ = group_spearman(arccos_dist(zx), P["d"][np.ix_(idx, idx)], P["br"][idx], idx)
    return t, b


if mode == "lotdo":
    out = {}
    t0 = time.time()
    head, mu, sd = fit_head(np.arange(len(don_int)))
    out["_all_11_donors"] = {p: dict(zip(["trust", "branch"], eval_panel(p, head, mu, sd))) for p in panels}
    print("all", out["_all_11_donors"], f"{time.time()-t0:.1f}s", flush=True)
    for u in ud_int:
        idx = np.where(don_int != u)[0]
        head, mu, sd = fit_head(idx)
        out[u] = {"n_train_anchors": int(len(idx))}
        out[u].update({p: dict(zip(["trust", "branch"], eval_panel(p, head, mu, sd))) for p in panels})
        print(u, out[u], f"{time.time()-t0:.1f}s", flush=True)
    (OUT / "03_lotdo.json").write_text(json.dumps(out, indent=2))

elif mode == "twolevel":
    seed, reps = int(sys.argv[2]), int(sys.argv[3])
    rng = np.random.default_rng(seed)
    mem_int = {u: np.where(don_int == u)[0] for u in ud_int}
    rows = []
    t0 = time.time()
    for r in range(reps):
        pick = rng.choice(ud_int, size=len(ud_int), replace=True)
        idx_int = np.concatenate([mem_int[u] for u in pick])
        head, mu, sd = fit_head(idx_int)
        row = {"rep": r, "n_train_anchors": int(len(idx_int)), "TSP2_copies": int((pick == "TSP2").sum())}
        for p in panels:
            don = panels[p]["don"]; ud = np.array(sorted(set(don)))
            mem = {u: np.where(don == u)[0] for u in ud}
            t_fix, b_fix = eval_panel(p, head, mu, sd)          # training donors resampled only
            pk = rng.choice(ud, size=len(ud), replace=True)
            idx = np.concatenate([mem[u] for u in pk])
            t_bt, b_bt = eval_panel(p, head, mu, sd, idx)       # both levels resampled
            row.update({f"{p}_trust_trainonly": t_fix, f"{p}_branch_trainonly": b_fix,
                        f"{p}_trust_both": t_bt, f"{p}_branch_both": b_bt})
        rows.append(row)
        print(json.dumps(row), f"{time.time()-t0:.1f}s", flush=True)
    (OUT / f"03_twolevel_seed{seed}.json").write_text(json.dumps(rows, indent=1))
