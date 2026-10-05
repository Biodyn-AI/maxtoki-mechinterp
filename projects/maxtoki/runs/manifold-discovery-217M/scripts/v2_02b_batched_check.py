"""v2 step 2b: check the batched trainer (v2_common.train_let_batched) against the one-at-a-time exact trainer.

Members re-fitted in batches and compared, value by value, with the exact results already in
refit_pool/results.jsonl (written by v2_02_refit_pool.py with the exact trainer):
  - the H65 head on all 290 internal anchors (trust, in-sample global Spearman) and its 6 branch-holdouts
  - the first permutations ('perm' tasks)
  - the first bootstrap replicates ('boot_full' trust, 'boot_branch' rho)
Also records the time per member.
usage: python v2_02b_batched_check.py <threads> <n_perm> <n_boot_reps>
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, "<REPO_ROOT>/projects/maxtoki/runs/manifold-discovery-217M/scripts")
import json, time
import numpy as np, torch
import v2_02_refit_pool as P
from v2_common import *

threads, n_perm, n_boot = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
P.setup(); torch.set_num_threads(threads)
G = P.G
rows = {json.loads(l)["id"]: json.loads(l) for l in (OUT / "refit_pool/results.jsonl").read_text().splitlines() if l.strip()}

members = []   # (id, features, ruler, eval-callable)
F, d = G["F"], G["d"]; o = np.arange(len(F))
members.append(("repro_full|H65", F, d, lambda prm, F=F, d=d: {
    "trust": trust_masked(euclid(F), euclid(z_from_params(prm, F)), o),
    "global_rho": P.pair_rho(arccos_dist(z_from_params(prm, F)), d, o)}))
for g in ["T_lineage", "B_lineage", "monocyte", "granulocyte", "erythroid", "stem"]:
    test = np.where(G["br"] == g)[0]; train = np.where(G["br"] != g)[0]
    members.append((f"repro_branch|H65|{g}", F[train], d[np.ix_(train, train)],
                    lambda prm, test=test: {"rho": P.pair_rho(arccos_dist(z_from_params(prm, F[test])), d[np.ix_(test, test)], test)}))
for bb in range(n_perm):
    s = P.perm_stages(bb); D = ruler_from_stages(s, G["idx"], G["T"])
    members.append((f"perm|{bb}", F, D, lambda prm, D=D: {
        "stat": P.pair_rho(arccos_dist(z_from_params(prm, F)), D, o),
        "trust": trust_masked(euclid(F), euclid(z_from_params(prm, F)), o)}))
for r in range(n_boot):
    pick, idx = P.boot_indices(r)
    f_raw = G["F_raw"][idx]; mu = f_raw.mean(0); sd = f_raw.std(0) + 1e-6
    Fb = ((f_raw - mu) / sd).astype(np.float32); Db = d[np.ix_(idx, idx)]
    members.append((f"boot_full|{r}", Fb, Db, lambda prm, Fb=Fb, Db=Db, idx=idx: {
        "trust": trust_masked(euclid(Fb), euclid(z_from_params(prm, Fb)), idx),
        "global_rho": P.pair_rho(arccos_dist(z_from_params(prm, Fb)), Db, idx)}))
    brb = G["br"][idx]
    for g in sorted(set(brb)):
        tid = f"boot_branch|{r}|{g}"
        if tid not in rows:
            continue
        test = np.where(brb == g)[0]; train = np.where(brb != g)[0]
        members.append((tid, Fb[train], Db[np.ix_(train, train)],
                        lambda prm, Fb=Fb, Db=Db, test=test, idx=idx: {
                            "rho": P.pair_rho(arccos_dist(z_from_params(prm, Fb[test])), Db[np.ix_(test, test)], idx[test])}))

members = [m for m in members if m[0] in rows]
print("members", len(members), flush=True)
t0 = time.time()
params = train_let_batched([m[1] for m in members], [m[2] for m in members])
secs = time.time() - t0
cmp = []
for (tid, _, _, ev), prm in zip(members, params):
    got = ev(prm); ex = rows[tid]
    for k, v in got.items():
        cmp.append({"id": tid, "metric": k, "exact": ex[k], "batched": v,
                    "diff": None if (v is None or ex[k] is None) else v - ex[k]})
out = {"threads": threads, "n_members": len(members), "seconds_total": secs, "seconds_per_member": secs / len(members),
       "comparisons": cmp}
for k in ["trust", "global_rho", "rho", "stat"]:
    dd = np.array([c["diff"] for c in cmp if c["metric"] == k and c["diff"] is not None])
    if len(dd):
        out[f"absdiff_{k}"] = {"n": int(len(dd)), "median": float(np.median(np.abs(dd))), "max": float(np.abs(dd).max()),
                               "mean_signed": float(dd.mean())}
(OUT / f"v2_02b_batched_check_t{threads}.json").write_text(json.dumps(out, indent=2))
print(json.dumps({k: v for k, v in out.items() if k != "comparisons"}, indent=1))
