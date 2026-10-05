"""v2 step 2: everything that needs the LET head to be RE-FITTED, run as a resumable task pool.

One task = one head fit (exact phase5 trainer, seed 42, 1500 Adam steps; see v2_common.train_let).
Results are appended, one JSON line per task, to outputs/v2_intervals/refit_pool/results.jsonl;
a re-run skips finished task ids, so the work can be done in chunks of < 9 minutes.

Task kinds (in submission order):
  repro_full        H65 / null head on all 290 internal anchors: in-sample trust (k=15) + in-sample global Spearman
  repro_branch      internal branch-holdout, one task per held-out branch (H65 and null ruler); branches whose
                    held-out ruler is constant are not fitted (they give NaN and are dropped, exactly as phase5 does)
  repro_donor       internal donor-holdout, one task per held-out donor (H65)
  seed              H65 and null full-panel fit with torch seeds 1..10 instead of 42 (optimiser-noise check)
  boot_full / boot_branch
                    bootstrap over the 11 TRAINING donors (resample donors with replacement, rng seed [20261001, r]).
                    Features re-standardised on the resampled rows; head re-fitted. boot_full records in-sample
                    trust + global Spearman, then scores the frozen head on the external and zero-shot panels, both on
                    the full panels (training-donor uncertainty only) and on donor-resampled panels (two levels,
                    rng seed [20261001, r, 7]). boot_branch = branch-holdout for one held-out branch of replicate r.
                    Copies of one anchor are never paired.
  perm              fifth gate: stage labels permuted within donor x tissue blocks (rng seed [777, b]), ruler rebuilt
                    on the H65 stage DAG, head RE-FITTED on the permuted ruler, statistic = in-sample Spearman between
                    head arc-cos distances and the (permuted) ruler; in-sample trust also recorded.

Every task records the torch thread count it used ('threads').

usage: python v2_02_refit_pool.py <n_workers> <budget_seconds> [n_boot] [n_perm]
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, "<REPO_ROOT>/projects/maxtoki/runs/manifold-discovery-217M/scripts")
import json, os, time
import multiprocessing as mp
import numpy as np, pandas as pd, torch
from v2_common import *

POOL_DIR = OUT / "refit_pool"; POOL_DIR.mkdir(parents=True, exist_ok=True)
RES = POOL_DIR / "results.jsonl"
BOOT_SEED = 20261001
PERM_SEED = 777
G = {}


def setup():
    """Load data (in the main process and in every worker)."""
    torch.set_num_threads(1)
    FZ = Frozen()
    G["F_raw"] = FZ.F_int_raw; G["F"] = FZ.F_int; G["d"] = FZ.d_int; G["m"] = FZ.m_int
    G["d_null"] = np.load(ART / "anchors/d_target_internal_null_shuffled.npy")
    G["br"] = branch_labels(FZ.m_int); G["don"] = FZ.m_int["donor_id"].astype(str).to_numpy()
    G["stages"] = FZ.m_int["hema_stage"].astype(str).to_numpy()
    G["blocks"] = (FZ.m_int["donor_id"].astype(str) + "|" + FZ.m_int["tissue"].astype(str)).to_numpy()
    G["ud"] = np.array(sorted(set(G["don"])))
    G["nodes"], G["idx"], G["T"] = stage_distance_table()
    ev = {}
    for p in ["external", "zeroshot"]:
        c, d, m = load_panel(p)
        ev[p] = dict(F_raw=build_pooled_drift(np.asarray(c), FZ.A_e, FZ.A_m, FZ.A_l, FZ.part), d=d,
                     br=branch_labels(m), don=m["donor_id"].astype(str).to_numpy())
    G["ev"] = ev


def boot_indices(r):
    rng = np.random.default_rng([BOOT_SEED, r])
    pick = rng.choice(G["ud"], size=len(G["ud"]), replace=True)
    idx = np.concatenate([np.where(G["don"] == u)[0] for u in pick])
    return pick, idx


def perm_stages(b):
    rng = np.random.default_rng([PERM_SEED, b])
    s = G["stages"].copy()
    for g in pd.unique(G["blocks"]):
        ii = np.where(G["blocks"] == g)[0]
        if len(ii) > 1:
            s[ii] = G["stages"][rng.permutation(ii)]
    return s


def pair_rho(Dh, Dt, origin):
    iu = np.triu_indices(len(origin), 1)
    keep = origin[iu[0]] != origin[iu[1]]
    x = Dh[iu][keep]; y = Dt[iu][keep]
    if len(x) < 2 or np.ptp(y) == 0 or np.ptp(x) == 0:
        return None
    return float(spearmanr(x, y)[0])


def eval_panel(p, head, mu, sd, idx=None):
    P = G["ev"][p]
    f = ((P["F_raw"] - mu) / sd).astype(np.float32)
    z = head_z(head, f)
    if idx is None:
        idx = np.arange(len(P["d"]))
    fx, zx = f[idx], z[idx]
    t = trust_masked(euclid(fx), euclid(zx), idx)
    Dh = arccos_dist(zx); Dt = P["d"][np.ix_(idx, idx)]
    b, nb = group_spearman(Dh, Dt, P["br"][idx], idx)
    gl = global_spearman(Dh, Dt, idx)
    return {"trust": t, "branch": b, "n_branches": nb, "global": gl, "n_anchors": int(len(idx))}


def run_task(task):
    kind = task["kind"]; t0 = time.time()
    torch.set_num_threads(int(task.get("threads", 1)))
    F, d = G["F"], G["d"]
    out = {"id": task["id"], "kind": kind}
    if kind in ("repro_full", "seed"):
        D = d if task["ruler"] == "H65" else G["d_null"]
        head = train_let_seed(F, D, task.get("seed", SEED))
        z = head_z(head, F); o = np.arange(len(F))
        out.update(ruler=task["ruler"], seed=task.get("seed", SEED),
                   trust=trust_masked(euclid(F), euclid(z), o), global_rho=pair_rho(arccos_dist(z), D, o))
    elif kind == "repro_branch" or kind == "repro_donor":
        D = d if task["ruler"] == "H65" else G["d_null"]
        grp = G["br"] if kind == "repro_branch" else G["don"]
        test = np.where(grp == task["group"])[0]; train = np.where(grp != task["group"])[0]
        head = train_let(F[train], D[np.ix_(train, train)])
        z = head_z(head, F[test])
        out.update(ruler=task["ruler"], group=task["group"], n_test=int(len(test)),
                   rho=pair_rho(arccos_dist(z), D[np.ix_(test, test)], test))
    elif kind in ("boot_full", "boot_branch"):
        r = task["rep"]; pick, idx = boot_indices(r)
        f_raw = G["F_raw"][idx]; mu = f_raw.mean(0); sd = f_raw.std(0) + 1e-6
        Fb = ((f_raw - mu) / sd).astype(np.float32); Db = d[np.ix_(idx, idx)]
        if kind == "boot_full":
            head = train_let(Fb, Db)
            z = head_z(head, Fb)
            out.update(rep=r, n_train=int(len(idx)), n_TSP2=int((pick == "TSP2").sum()),
                       trust=trust_masked(euclid(Fb), euclid(z), idx),
                       global_rho=pair_rho(arccos_dist(z), Db, idx))
            rng_e = np.random.default_rng([BOOT_SEED, r, 7])
            for p in ["external", "zeroshot"]:
                out[f"{p}_trainonly"] = eval_panel(p, head, mu, sd)
                don = G["ev"][p]["don"]; ud = np.array(sorted(set(don)))
                pk = rng_e.choice(ud, size=len(ud), replace=True)
                eidx = np.concatenate([np.where(don == u)[0] for u in pk])
                out[f"{p}_bothlevels"] = eval_panel(p, head, mu, sd, eidx)
        else:
            brb = G["br"][idx]
            test = np.where(brb == task["group"])[0]; train = np.where(brb != task["group"])[0]
            head = train_let(Fb[train], Db[np.ix_(train, train)])
            z = head_z(head, Fb[test])
            out.update(rep=r, group=task["group"], n_test=int(len(test)),
                       n_test_distinct=int(len(set(idx[test]))),
                       rho=pair_rho(arccos_dist(z), Db[np.ix_(test, test)], idx[test]))
    elif kind == "perm":
        s = perm_stages(task["b"])
        D = ruler_from_stages(s, G["idx"], G["T"])
        head = train_let(F, D)
        z = head_z(head, F); o = np.arange(len(F))
        out.update(b=task["b"], n_changed=int((s != G["stages"]).sum()),
                   stat=pair_rho(arccos_dist(z), D, o), trust=trust_masked(euclid(F), euclid(z), o))
    out["seconds"] = time.time() - t0
    out["threads"] = int(task.get("threads", 1))
    return out


def train_let_seed(features, d_target, seed):
    """train_let with a different torch seed (identical otherwise)."""
    if seed == SEED:
        return train_let(features, d_target)
    torch.manual_seed(seed)
    n, d_in = features.shape
    init_beta = max(1.0, float(d_target.max()) / (np.pi / 2.0 + 1e-6))
    head = LETHead(d_in, LATENT_DIM, init_beta=init_beta)
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    x = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
    D = torch.from_numpy(np.ascontiguousarray(d_target, dtype=np.float32))
    for _ in range(EPOCHS):
        opt.zero_grad()
        z, recon = head(x)
        loss = let_loss(z, x, recon, D, beta=head.beta)
        loss.backward(); opt.step()
    head.eval()
    return head


def constant_ruler(test_idx_orig, D):
    """True if all held-out pairs (copies of one anchor excluded) share one ruler value."""
    iu = np.triu_indices(len(test_idx_orig), 1)
    keep = test_idx_orig[iu[0]] != test_idx_orig[iu[1]]
    vals = D[np.ix_(test_idx_orig, test_idx_orig)][iu][keep]
    return len(vals) < 2 or np.ptp(vals) == 0


def build_tasks(n_boot, n_perm):
    tasks = []
    for lab in ["H65", "null"]:
        tasks.append({"id": f"repro_full|{lab}", "kind": "repro_full", "ruler": lab})
    for lab in ["H65", "null"]:
        D = G["d"] if lab == "H65" else G["d_null"]
        for g in pd.unique(G["br"]):
            test = np.where(G["br"] == g)[0]
            if len(test) < 3 or constant_ruler(test, D):
                continue
            tasks.append({"id": f"repro_branch|{lab}|{g}", "kind": "repro_branch", "ruler": lab, "group": str(g)})
    for u in G["ud"]:
        test = np.where(G["don"] == u)[0]
        if len(test) < 3 or constant_ruler(test, G["d"]):
            continue
        tasks.append({"id": f"repro_donor|H65|{u}", "kind": "repro_donor", "ruler": "H65", "group": str(u)})
    # the same reproduction tasks again with 4 torch threads (the setting under which the investigation
    # reproduced the published null branch-holdout exactly); ids prefixed 't4|'
    rep4 = [dict(t, id="t4|" + t["id"], threads=4) for t in tasks if t["kind"].startswith("repro")]
    tasks = rep4 + tasks
    for lab in ["H65", "null"]:
        for s in range(1, 11):
            tasks.append({"id": f"seed|{lab}|{s}", "kind": "seed", "ruler": lab, "seed": s})
    boot, perm = [], []
    for r in range(n_boot):
        pick, idx = boot_indices(r)
        bt = [{"id": f"boot_full|{r}", "kind": "boot_full", "rep": r}]
        brb = G["br"][idx]
        for g in sorted(set(brb)):
            test = np.where(brb == g)[0]
            if len(test) < 3 or constant_ruler(idx[test], G["d"]):
                continue
            bt.append({"id": f"boot_branch|{r}|{g}", "kind": "boot_branch", "rep": r, "group": str(g)})
        boot.append(bt)
    for b in range(n_perm):
        perm.append({"id": f"perm|{b}", "kind": "perm", "b": b})
    # interleave: after each bootstrap replicate's tasks, add the same number of permutation tasks
    pi = 0
    for bt in boot:
        tasks.extend(bt)
        k = len(bt)
        tasks.extend(perm[pi:pi + k]); pi += k
    tasks.extend(perm[pi:])
    return tasks


def init_worker():
    setup()


if __name__ == "__main__":
    n_workers = int(sys.argv[1]); budget = float(sys.argv[2])
    n_boot = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    n_perm = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
    T0 = time.time()
    setup()
    tasks = build_tasks(n_boot, n_perm)
    done = set()
    if RES.exists():
        for line in RES.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    todo = [t for t in tasks if t["id"] not in done]
    (POOL_DIR / "task_list.json").write_text(json.dumps({"n_tasks": len(tasks), "n_boot": n_boot, "n_perm": n_perm,
                                                         "boot_seed": BOOT_SEED, "perm_seed": PERM_SEED}, indent=1))
    print(f"tasks total {len(tasks)}  done {len(done)}  todo {len(todo)}", flush=True)
    if not todo:
        sys.exit(0)
    ctx = mp.get_context("spawn")
    n_new = 0
    with ctx.Pool(n_workers, initializer=init_worker) as pool, open(RES, "a") as fh:
        pending = []
        it = iter(todo)
        exhausted = False
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
                    r = a.get()
                    fh.write(json.dumps(r) + "\n"); fh.flush(); n_new += 1
                else:
                    still.append(a)
            pending = still
            if time.time() - T0 > budget + 75:  # hard stop; unfinished tasks are redone next run
                pool.terminate(); break
            if pending:
                pending[0].wait(0.5)  # block briefly on the oldest task instead of busy-looping
    print(f"finished {n_new} tasks in {time.time()-T0:.0f}s; total done {len(done)+n_new}/{len(tasks)}", flush=True)
    print("worker peak RSS (MB) from ru_maxrss of children:",
          round(__import__('resource').getrusage(__import__('resource').RUSAGE_CHILDREN).ru_maxrss / 1e6, 1), flush=True)
