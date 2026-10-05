"""v2 step 10: leave-one-TRAINING-donor-out (LOTDO) for the H65 internal numbers, with the head RE-FITTED.

For each of the 11 internal (training) donors u:
  - drop all anchors of u; re-standardise the 2,464-d pooled-drift features on the remaining rows
    (as the training-donor bootstrap in v2_02_refit_pool.py does);
  - 'lotdo_full|u'  : fit the LET head (exact Phase-5 trainer, seed 42) on the remaining anchors;
                      record in-sample trust (k=15) and in-sample global Spearman; then score the FROZEN
                      re-fitted head on the full external and zero-shot panels (trust, within-branch mean, global);
  - 'lotdo_branch|u|g': branch-holdout inside the remaining anchors: re-fit without branch g, Spearman on the
                      held-out pairs (branches with < 3 anchors or a constant ruler are skipped, as in Phase 5).
Also (main process, no re-fit): the deployed head scored on the FIRST lung control panel ('lung_control',
lung-resident immune cells with real stage labels, 51 anchors), to check the deployed numbers.

Resumable: one JSON line per task in outputs/v2_intervals/lotdo/results.jsonl.
usage: python v2_10_internal_lotdo.py run <n_workers> <budget_seconds>
       python v2_10_internal_lotdo.py summarize
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, "<REPO_ROOT>/projects/maxtoki/runs/manifold-discovery-217M/scripts")
import json, time
import multiprocessing as mp
import numpy as np, pandas as pd, torch
from scipy.stats import t as tdist
from sklearn.manifold import trustworthiness
from v2_common import *
import v2_02_refit_pool as P

LDIR = OUT / "lotdo"; LDIR.mkdir(parents=True, exist_ok=True)
RES = LDIR / "results.jsonl"


def init_worker():
    P.setup()


def prep(u):
    G = P.G
    keep = np.where(G["don"] != u)[0]
    f_raw = G["F_raw"][keep]; mu = f_raw.mean(0); sd = f_raw.std(0) + 1e-6
    Fk = ((f_raw - mu) / sd).astype(np.float32)
    Dk = G["d"][np.ix_(keep, keep)]
    return keep, Fk, Dk, mu, sd


def run_task(task):
    torch.set_num_threads(1)
    t0 = time.time(); G = P.G
    u = task["donor"]
    keep, Fk, Dk, mu, sd = prep(u)
    out = {"id": task["id"], "kind": task["kind"], "donor": u, "n_train": int(len(keep))}
    if task["kind"] == "lotdo_full":
        head = train_let(Fk, Dk)
        z = head_z(head, Fk); o = np.arange(len(keep))
        out.update(trust=trust_masked(euclid(Fk), euclid(z), o), global_rho=P.pair_rho(arccos_dist(z), Dk, o))
        for p in ["external", "zeroshot"]:
            out[p] = P.eval_panel(p, head, mu, sd)
    else:
        brk = G["br"][keep]
        test = np.where(brk == task["group"])[0]; train = np.where(brk != task["group"])[0]
        head = train_let(Fk[train], Dk[np.ix_(train, train)])
        z = head_z(head, Fk[test])
        out.update(group=task["group"], n_test=int(len(test)),
                   rho=P.pair_rho(arccos_dist(z), Dk[np.ix_(test, test)], test))
    out["seconds"] = time.time() - t0
    return out


def build_tasks():
    G = P.G; tasks = []
    for u in G["ud"]:
        tasks.append({"id": f"lotdo_full|{u}", "kind": "lotdo_full", "donor": str(u)})
        keep = np.where(G["don"] != u)[0]; brk = G["br"][keep]
        for g in sorted(set(brk)):
            test = np.where(brk == g)[0]
            if len(test) < 3 or P.constant_ruler(keep[test], G["d"]):
                continue
            tasks.append({"id": f"lotdo_branch|{u}|{g}", "kind": "lotdo_branch", "donor": str(u), "group": str(g)})
    return tasks


def run(n_workers, budget):
    T0 = time.time()
    P.setup()
    tasks = build_tasks()
    done = {json.loads(l)["id"] for l in RES.read_text().splitlines() if l.strip()} if RES.exists() else set()
    todo = [t for t in tasks if t["id"] not in done]
    print(f"tasks {len(tasks)} done {len(done)} todo {len(todo)}", flush=True)
    if not todo:
        return
    ctx = mp.get_context("spawn")
    n_new = 0
    with ctx.Pool(n_workers, initializer=init_worker) as pool, open(RES, "a") as fh:
        pending = []; it = iter(todo); exhausted = False
        while True:
            while not exhausted and len(pending) < n_workers and time.time() - T0 < budget:
                try:
                    t = next(it)
                except StopIteration:
                    exhausted = True; break
                pending.append(pool.apply_async(run_task, (t,)))
            if not pending:
                break
            still = []
            for a in pending:
                if a.ready():
                    fh.write(json.dumps(a.get()) + "\n"); fh.flush(); n_new += 1
                else:
                    still.append(a)
            pending = still
            if time.time() - T0 > budget + 75:
                pool.terminate(); break
            if pending:
                pending[0].wait(0.5)
    print(f"finished {n_new} tasks in {time.time()-T0:.0f}s; total {len(done)+n_new}/{len(tasks)}", flush=True)


def summarize():
    P.setup(); G = P.G
    rows = [json.loads(l) for l in RES.read_text().splitlines() if l.strip()]
    byid = {r["id"]: r for r in rows}
    tasks = build_tasks()
    missing = [t["id"] for t in tasks if t["id"] not in byid]
    rf = json.loads((OUT / "v2_03_refit_summary.json").read_text())["internal_reproduction_threads4"]["H65"]
    full_obs = {"trust": rf["trust_in_sample"], "branch_plain": rf["branch_holdout"]["plain_mean"],
                "branch_anchor_weighted": rf["branch_holdout"]["anchor_weighted_mean"]}
    rep = json.loads((OUT / "v2_00_reproduce.json").read_text())
    full_obs.update({"external_trust": rep["external"]["trust_sklearn_k15"], "external_branch": rep["external"]["branch_within_mean"],
                     "zeroshot_trust": rep["zeroshot"]["trust_sklearn_k15"], "zeroshot_branch": rep["zeroshot"]["branch_within_mean"]})
    per = {}
    for u in G["ud"]:
        f = byid.get(f"lotdo_full|{u}")
        bb = [r for r in rows if r["kind"] == "lotdo_branch" and r["donor"] == u]
        sc = [b for b in bb if b["rho"] is not None]
        v = np.array([b["rho"] for b in sc], float); w = np.array([b["n_test"] for b in sc], float)
        per[str(u)] = {
            "n_anchors_dropped": int((G["don"] == u).sum()), "n_train": f["n_train"] if f else None,
            "trust": f["trust"] if f else None, "global_in_sample": f["global_rho"] if f else None,
            "branch_plain": float(v.mean()) if len(v) else None,
            "branch_anchor_weighted": float((w * v).sum() / w.sum()) if len(v) else None,
            "n_branches_scored": int(len(v)),
            "per_branch": {b["group"]: {"n_test": b["n_test"], "rho": b["rho"]} for b in bb},
            "external_trust": f["external"]["trust"] if f else None, "external_branch": f["external"]["branch"] if f else None,
            "zeroshot_trust": f["zeroshot"]["trust"] if f else None, "zeroshot_branch": f["zeroshot"]["branch"] if f else None}
    out = {"n_tasks": len(tasks), "n_missing": len(missing), "missing": missing,
           "resampling": "leave one of the 11 training donors out; features re-standardised on the remaining anchors; "
                         "head re-fitted (seed 42); external / zero-shot scored with the re-fitted head, panels not resampled",
           "observed_all_11_donors": full_obs, "per_donor": per, "ranges": {}}
    for k in ["trust", "branch_plain", "branch_anchor_weighted", "external_trust", "external_branch", "zeroshot_trust", "zeroshot_branch"]:
        vals = {u: per[u][k] for u in per if per[u][k] is not None}
        if not vals:
            continue
        a = np.array(list(vals.values()), float); n = len(a)
        se = float(np.sqrt((n - 1) / n * ((a - a.mean()) ** 2).sum())); q = float(tdist.ppf(0.975, n - 1))
        gate = GATE_TRUST if "trust" in k else GATE_CORR
        out["ranges"][k] = {"min": float(a.min()), "argmin": min(vals, key=vals.get), "max": float(a.max()),
                            "argmax": max(vals, key=vals.get), "n_donors": n, "jackknife_se": se,
                            "t_ci95_around_observed": [full_obs[k] - q * se, full_obs[k] + q * se],
                            "n_below_gate": int((a < gate).sum()), "gate": gate}
    # first lung control (lung-resident immune cells, real stage labels): deployed head, frozen
    FZ = Frozen()
    head = LETHead(FZ.F_int.shape[1], LATENT_DIM)
    head.load_state_dict(torch.load(ART / "heads/let_anchor_internal.pt", map_location="cpu")); head.eval()
    f, d, m = FZ.features("lung_control")
    z = head_z(head, f); o = np.arange(len(d)); Dh = arccos_dist(z)
    dep = json.loads((REP / "external_validation_lung_control.json").read_text())
    br, nb = group_spearman(Dh, d, branch_labels(m), o)
    dn, nd = group_spearman(Dh, d, m["donor_id"].astype(str).to_numpy(), o, skip=())
    out["lung_control_v1_frozen"] = {
        "n_anchors": int(len(d)), "n_donors": int(m["donor_id"].nunique()),
        "trust_k15": float(trustworthiness(f, z, n_neighbors=15)), "random_holdout": random_holdout_frozen(z, d),
        "donor_within_mean": dn, "branch_within_mean": br, "n_branches_scored": nb,
        "global_spearman": global_spearman(Dh, d, o),
        "deployed": {k: dep[k] for k in ["trustworthiness", "random_holdout", "donor_holdout", "branch_holdout", "global_correlation",
                                         "all_gates_pass", "diagnostic"]}}
    (OUT / "v2_10_internal_lotdo.json").write_text(json.dumps(out, indent=2, default=float))
    write_run_config("v2_10_internal_lotdo", CORE_INPUTS + [ART / "anchors/centroids_lung_control.npy",
                     ART / "anchors/d_target_lung_control.npy", ART / "anchors/anchor_meta_lung_control.csv",
                     REP / "external_validation_lung_control.json", RES, OUT / "v2_03_refit_summary.json",
                     OUT / "v2_00_reproduce.json"], {"seed_head": SEED})
    print(json.dumps(out["ranges"], indent=1)); print(json.dumps(out["lung_control_v1_frozen"], indent=1))
    print("missing", len(missing))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]), float(sys.argv[3]))
    else:
        summarize()
