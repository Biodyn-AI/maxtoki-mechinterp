"""v2b step 2: every LET-head fit as a resumable task pool (one task = one head fit + its evaluation).

Usage:
  python v2b_devorder_02_pool.py list [prefix]            # count pending / done tasks
  python v2b_devorder_02_pool.py run <n_workers> <start_budget_s> [prefix ...]
      starts new tasks only during the first <start_budget_s> seconds, then waits for running ones and exits.
      Results are appended to outputs/v2b_devorder/pool/results.jsonl (key, result, seconds).

Task keys: family|rep|variant|part|arg
  family  h65 | h38 | h95 | h103 | n60
  variant pos      the deployed ruler
          runnull  the run's own null ruler (H65: d_target_internal_null_shuffled.npy, within-branch shuffle;
                   H38/H95/H103: the global label shuffle of phase13 / phase3a / sweep2, default_rng(42))
          perm<d>  structured null: H65 - the (cell_type, stage) class -> stage map permuted among classes of the
                   same branch, default_rng([2026, d]); H38/H95 - the cell_type -> category-set map permuted among
                   cell types that have a category, default_rng([3838 or 9595, d])
  part    full     fit on all internal anchors: in-sample trust; head applied frozen to external / zero-shot
                   (and lung_nonhema for H65 pos) with the phase7/8 (H65) or phase15 (H38/H95) gate code
          rand i   i-th random 80/20 split of phase5.random_holdout_corr (rng default_rng(42) replayed)
          donor g  phase5.grouped_holdout_corr, donor g held out
          branch g / cat g   the same with branch (H65) or primary category / depth group (H38/H95/H103)
  Every holdout value is computed on all held-out pairs (the run's number) and on held-out pairs whose two
  anchors have DIFFERENT Tabula Sapiens cell_type strings ('diff_ct').
  n60 tasks: n60|<L10H6 or rand>|<centroid index>_<seed or head>|full/rand/branch|arg  (phase9 scoring).

Threads: torch 1 thread per worker. LET trainer = phase5_let_anchor.train_let (seed 42), imported read-only.
"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, "<REPO_ROOT>/projects/maxtoki/runs/manifold-discovery-217M/scripts")
import json, os, time, traceback
import multiprocessing as mp
import numpy as np, pandas as pd

POOL = None  # set in main

H65_POS_REPS = ["maxtoki", "lookup_ct_s0", "tokenbag_pca64", "hvg_pca64", "lookup_ct_s1", "lookup_ct_s2",
                "lookup_ct_s3", "lookup_ct_s4", "lookup_cls_s0", "lookup_cls_s1", "lookup_cls_s2", "maxtoki_pca64"]
H65_NULL_REPS = ["maxtoki", "lookup_ct_s0", "tokenbag_pca64", "hvg_pca64"]
CAT_POS_REPS = ["maxtoki", "lookup_ct_s0", "tokenbag_pca64", "hvg_pca64", "lookup_ct_s1", "lookup_ct_s2"]
CAT_NULL_REPS = ["maxtoki", "lookup_ct_s0"]
PERM_REPS = ["maxtoki", "lookup_ct_s0", "tokenbag_pca64", "hvg_pca64"]
CAT_PERM_REPS = ["maxtoki", "lookup_ct_s0", "tokenbag_pca64", "hvg_pca64"]
N_PERM_FULL = 40
N_PERM_BRANCH = 20
N_PERM_CAT = 30
N60_RAND11 = 30
N60_RAND1 = 10


# =====================================================================================================
# worker side
# =====================================================================================================
_C = {}


def C():
    """Per-process cache of everything a task needs."""
    if _C:
        return _C
    import torch
    torch.set_num_threads(1)
    import v2b_devorder_common as V
    import phase13_h38_lite as P13
    import phase3a_manifold_sweep as S1
    import phase3a_sweep2_ordinal as S2
    import phase15_validate_candidate as P15
    _C.update(V=V, P13=P13, S1=S1, S2=S2, P15=P15)
    _C["meta"] = {p: V.meta(p) for p in V.PANELS}
    _C["classes"] = V.h65_classes()
    _C["feat"] = {}
    return _C


def feat(rep, panel):
    c = C()
    k = (rep, panel)
    if k not in c["feat"]:
        c["feat"][k] = c["V"].load_feat(rep, panel)
    return c["feat"][k]


# ---------------------------------------------------------------- rulers per family / variant / panel
def h65_stages(variant, panel):
    c = C(); V = c["V"]; m = c["meta"][panel]
    if variant.startswith("perm"):
        d = int(variant[4:])
        mp_ = V.permuted_stage_map(c["classes"], np.random.default_rng([2026, d]))
        return [mp_[x] for x in V.anchor_classes(m)]
    return m["hema_stage"].astype(str).tolist()


def h65_ruler(variant, panel):
    c = C(); V = c["V"]
    if variant == "pos":
        return np.load(V.ART / f"anchors/d_target_{panel}.npy")
    if variant == "runnull":
        return np.load(V.ART / f"anchors/d_target_{panel}_null_shuffled.npy")
    return V.h65_ruler(h65_stages(variant, panel))


def cat_perm_map(family, d):
    """cell_type (lower-case) -> category set, permuted among cell types that have a non-empty set."""
    c = C()
    catf = cat_fn(family, None)
    cts = sorted({x.strip().lower() for p in ("internal", "external", "zeroshot")
                  for x in c["meta"][p]["cell_type"].astype(str)})
    cts = [x for x in cts if catf(x)]
    sets = [frozenset(catf(x)) for x in cts]
    rng = np.random.default_rng([3838 if family == "h38" else 9595, d])
    perm = rng.permutation(len(cts))
    return {x: set(sets[j]) for x, j in zip(cts, perm)}


def cat_fn(family, variant):
    c = C()
    base = {"h38": c["P13"].categorize, "h95": c["S1"].cat_h95}[family]
    if variant is None or not variant.startswith("perm"):
        return base
    mp_ = cat_perm_map(family, int(variant[4:]))
    return lambda ct: mp_.get(ct.strip().lower(), set())


def cat_setup(family, variant, panel="internal"):
    """Returns keep mask, ruler D (kept x kept), group labels (kept), using the run's code paths:
    H38 phase13 (build_h38_ruler, primary = sorted(categories)[0]); H95 phase3a_manifold_sweep.evaluate_candidate;
    H103 phase3a_sweep2_ordinal.evaluate. runnull = the run's global shuffle with default_rng(42)."""
    c = C(); m = c["meta"][panel]
    cts = m["cell_type"].astype(str).tolist()
    if family == "h103":
        S2 = c["S2"]
        D_full, depths = S2.build_ordinal_ruler(cts, S2.H103_DEPTH)
        keep = depths >= 0
        kk = np.where(keep)[0]
        D = D_full[np.ix_(kk, kk)]
        dk = depths[keep]
        groups = dk.astype(int).astype(str)
        if variant == "runnull":
            perm = np.random.default_rng(42).permutation(len(dk))
            ds = dk[perm]
            D = np.abs(ds[None, :] - ds[:, None]).astype(np.float32)
            groups = ds.astype(int).astype(str)
        return keep, D, groups
    catf = cat_fn(family, variant)
    keep = np.array([bool(cat_fn(family, None)(x)) for x in cts])
    ctk = [x for x, k in zip(cts, keep) if k]
    if family == "h38":
        all_cats = c["P13"].ALL_CATEGORIES
    else:
        all_cats = sorted(set().union(*[cat_fn(family, None)(x) for x in ctk]))
    if variant == "runnull":
        perm = np.random.default_rng(42).permutation(len(ctk))
        ctk = [ctk[p] for p in perm]
    D, _ = c["S1"].build_ruler(ctk, catf, all_cats)
    groups = np.array([sorted(catf(x))[0] if catf(x) else "_unk" for x in ctk])
    return keep, D, groups


# ---------------------------------------------------------------- one task
def holdout_eval(F, D, test, train, ct):
    V = C()["V"]
    head, _ = V.fit(F[train], D[np.ix_(train, train)])
    zt = V.z_from(V.head_params(head), F[test])
    a, d = V.pair_rhos(V.arccos_dist(zt), D[np.ix_(test, test)], ct[test])
    return {"n": int(len(test)), "rho": a, "rho_diff_ct": d}, zt


def rand_split(n, i, frac=0.2):
    rng = np.random.default_rng(42)
    for _ in range(i + 1):
        idx = rng.permutation(n)
    n_test = int(round(n * frac))
    return np.array(sorted(idx[:n_test])), np.array(sorted(idx[n_test:]))


def run_task(key):
    t0 = time.time()
    try:
        res = _run_task(key)
        return key, res, time.time() - t0, None
    except Exception:
        return key, None, time.time() - t0, traceback.format_exc()


def _run_task(key):
    c = C(); V = c["V"]
    from sklearn.manifold import trustworthiness
    fam, rep, variant, part, arg = key.split("|")
    if fam == "n60":
        return n60_task(rep, variant, part, arg)
    m = c["meta"]["internal"]
    ct_all = m["cell_type"].astype(str).to_numpy()
    donors_all = m["donor_id"].astype(str).to_numpy()
    if fam == "h65":
        F = feat(rep, "internal"); D = h65_ruler(variant, "internal")
        ct = ct_all; donors = donors_all
        groups = V.branches_of(m["hema_stage"].astype(str))     # true branches (phase5 uses them for the null too)
    else:
        keep, D, groups = cat_setup(fam, variant)
        F = feat(rep, "internal")[keep]; ct = ct_all[keep]; donors = donors_all[keep]
    n = len(F)
    if part == "full":
        head, z = V.fit(F, D)
        P = V.head_params(head)
        out = {"trust": float(trustworthiness(F, z, n_neighbors=15)), "n": int(n)}
        out["frozen"] = frozen_panels(fam, rep, variant, P) if variant != "runnull" else {}
        if variant in ("pos", "runnull"):
            np.savez(V.HEADS / f"{fam}_{rep}_{variant}.npz", W=P["W"], b=P["b"], log_beta=P["log_beta"], z_internal=z)
        return out
    if part == "rand":
        test, train = rand_split(n, int(arg))
    else:
        lab = donors if part == "donor" else groups
        test = np.where(lab == arg)[0]; train = np.where(lab != arg)[0]
    r, zt = holdout_eval(F, D, test, train, ct)
    if part in ("branch", "cat") and variant in ("pos", "runnull"):
        r["test_idx"] = test.tolist(); r["z"] = np.round(zt, 7).tolist()
    return r


def frozen_panels(fam, rep, variant, P):
    """Apply the internal head unchanged to the other panels."""
    c = C(); V = c["V"]
    out = {}
    if fam == "h65":
        panels = ["external", "zeroshot"] + (["lung_nonhema"] if variant == "pos" else [])
        for p in panels:
            m = c["meta"][p]; f = feat(rep, p); z = V.z_from(P, f)
            D = h65_ruler(variant, p) if p != "lung_nonhema" else np.load(V.ART / "anchors/d_target_lung_nonhema.npy")
            br = V.branches_of(m["hema_stage"].astype(str))
            g = V.frozen_gates_h65(f, z, D, m["donor_id"].astype(str).to_numpy(), br, m["cell_type"].astype(str).to_numpy())
            out[p] = g
            if variant in ("pos", "runnull"):
                np.save(V.HEADS / f"z_{fam}_{rep}_{variant}_{p}.npy", z)
        return out
    for p in ["external", "zeroshot"]:
        m = c["meta"][p]
        keep, D, groups = cat_setup(fam, "pos" if variant == "runnull" else variant, p)
        f = feat(rep, p)[keep]; z = V.z_from(P, f)
        ct = m["cell_type"].astype(str).to_numpy()[keep]; dn = m["donor_id"].astype(str).to_numpy()[keep]
        g = V.frozen_gates_cat(f, z, D, dn, groups, ct)
        g["n_kept"] = int(keep.sum())
        if fam == "h103":
            from sklearn.manifold import trustworthiness
            for k in (5, 8, 15):
                if k < len(f) / 2:
                    g[f"trust_k{k}"] = float(trustworthiness(f, z, n_neighbors=k))
        out[p] = g
        if variant == "pos":
            np.save(V.HEADS / f"z_{fam}_{rep}_{variant}_{p}.npy", z)
    return out


# ---------------------------------------------------------------- N60: phase9 single-head scoring
def n60_features(variant, arg0):
    V = C()["V"]
    cen = np.load(V.ART / "anchors/centroids_internal.npy", mmap_mode="r")
    idx, s = arg0.split("_")
    x = np.asarray(cen[:, int(idx), :], dtype=np.float32)
    if variant == "head":
        A = np.load(V.ART / f"operators/layer{int(idx)-1:02d}_head{int(s)}.npy").astype(np.float32)
    else:
        A = np.random.default_rng([6060, int(idx), int(s)]).standard_normal((154, x.shape[1])).astype(np.float32)
    f = (x @ A.T).astype(np.float32)
    return ((f - f.mean(axis=0)) / (f.std(axis=0) + 1e-6)).astype(np.float32)   # phase9 lines 72-75


def n60_task(variant, arg0, part, arg):
    c = C(); V = c["V"]
    from sklearn.manifold import trustworthiness
    m = c["meta"]["internal"]
    F = n60_features(variant, arg0)
    D = np.load(V.ART / "anchors/d_target_internal.npy")
    ct = m["cell_type"].astype(str).to_numpy()
    br = V.branches_of(m["hema_stage"].astype(str))
    if part == "full":
        _, z = V.fit(F, D)
        return {"trust": float(trustworthiness(F, z, n_neighbors=15))}
    if part == "rand":
        test, train = rand_split(len(F), int(arg))
    else:
        test = np.where(br == arg)[0]; train = np.where(br != arg)[0]
    r, _ = holdout_eval(F, D, test, train, ct)
    return r


# =====================================================================================================
# task list
# =====================================================================================================
def scorable_groups(D, lab, n_min=3):
    out = []
    for g in pd.unique(lab):
        t = np.where(lab == g)[0]
        if len(t) < n_min:
            continue
        Dt = D[np.ix_(t, t)]; iu = np.triu_indices(len(t), 1)
        if len(iu[0]) and np.ptp(Dt[iu]) > 0:
            out.append(str(g))
    return out


def internal_parts(fam, variant, n_rand):
    c = C(); V = c["V"]; m = c["meta"]["internal"]
    donors = m["donor_id"].astype(str).to_numpy()
    if fam == "h65":
        D = h65_ruler(variant, "internal"); groups = V.branches_of(m["hema_stage"].astype(str)); dn = donors
        gname = "branch"
    else:
        keep, D, groups = cat_setup(fam, variant); dn = donors[keep]; gname = "cat"
    parts = [("full", "-")] + [("rand", str(i)) for i in range(n_rand)]
    parts += [("donor", g) for g in scorable_groups(D, dn)]
    parts += [(gname, g) for g in scorable_groups(D, groups)]
    return parts


def all_tasks():
    T = []
    def add(fam, rep, variant, parts):
        T.extend(f"{fam}|{rep}|{variant}|{p}|{a}" for p, a in parts)
    pos65 = internal_parts("h65", "pos", 10)
    for r in H65_POS_REPS:
        add("h65", r, "pos", pos65)
    nul65 = internal_parts("h65", "runnull", 10)
    for r in H65_NULL_REPS:
        add("h65", r, "runnull", nul65)
    for fam, nr in (("h38", 10), ("h95", 5), ("h103", 5)):
        pp = internal_parts(fam, "pos", nr); pn = internal_parts(fam, "runnull", nr)
        for r in CAT_POS_REPS:
            add(fam, r, "pos", pp)
        for r in CAT_NULL_REPS:
            add(fam, r, "runnull", pn)
    # N60: L10H6 reproduction + random projections (phase9: full + 3 random splits + branch holdout)
    m = C()["meta"]["internal"]; V = C()["V"]
    br = V.branches_of(m["hema_stage"].astype(str)); D = np.load(V.ART / "anchors/d_target_internal.npy")
    n60_parts = [("full", "-")] + [("rand", str(i)) for i in range(3)] + [("branch", g) for g in scorable_groups(D, br)]
    add("n60", "head", "11_6", n60_parts)
    for s in range(N60_RAND11):
        add("n60", "rand", f"11_{s}", n60_parts)
    for s in range(N60_RAND1):
        add("n60", "rand", f"1_{s}", n60_parts)
    # structured nulls (refit)
    for d in range(N_PERM_FULL):
        for r in PERM_REPS:
            add("h65", r, f"perm{d}", [("full", "-")])
            if d < N_PERM_BRANCH:
                Dp = h65_ruler(f"perm{d}", "internal")
                add("h65", r, f"perm{d}", [("branch", g) for g in scorable_groups(Dp, br)])
    for d in range(N_PERM_CAT):
        for fam in ("h38", "h95"):
            for r in CAT_PERM_REPS:
                add(fam, r, f"perm{d}", [("full", "-")])
    return T


# =====================================================================================================
# driver
# =====================================================================================================
def done_keys(path):
    if not path.exists():
        return set()
    out = set()
    for l in path.read_text().splitlines():
        if l.strip():
            j = json.loads(l)
            if j.get("error") is None:
                out.add(j["key"])
    return out


def _init():
    import torch
    torch.set_num_threads(1)
    C()


def _guarded(args):
    key, deadline = args
    if time.time() > deadline:
        return key, None, 0.0, "SKIPPED_DEADLINE"
    return run_task(key)


def main():
    import v2b_devorder_common as V
    pool_dir = V.OUT / "pool"; pool_dir.mkdir(exist_ok=True)
    res_path = pool_dir / "results.jsonl"
    cmd = sys.argv[1]
    tasks = all_tasks()
    (pool_dir / "task_list.json").write_text(json.dumps(tasks))
    done = done_keys(res_path)
    if cmd == "list":
        pre = sys.argv[2:] or [""]
        sel = [t for t in tasks if any(t.startswith(p) for p in pre)]
        print("tasks", len(sel), "done", sum(t in done for t in sel), "pending", sum(t not in done for t in sel))
        return
    nw = int(sys.argv[2]); budget = float(sys.argv[3]); pre = sys.argv[4:] or [""]
    todo = [t for t in tasks if t not in done and any(t.startswith(p) for p in pre)]
    deadline = time.time() + budget
    print(f"pending {len(todo)} (selected), workers {nw}, start budget {budget:.0f}s", flush=True)
    n_ok = 0; T0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(nw, initializer=_init) as pool, open(res_path, "a") as fh:
        for key, res, sec, err in pool.imap_unordered(_guarded, [(t, deadline) for t in todo], chunksize=1):
            if err == "SKIPPED_DEADLINE":
                continue
            fh.write(json.dumps({"key": key, "res": res, "sec": round(sec, 2), "error": err}, default=float) + "\n")
            fh.flush()
            n_ok += err is None
            if err:
                print("ERROR", key, err[-400:], flush=True)
    left = len([t for t in tasks if t not in done_keys(res_path)])
    print(f"finished {n_ok} tasks in {time.time()-T0:.0f}s; tasks left overall: {left}", flush=True)


if __name__ == "__main__":
    main()
