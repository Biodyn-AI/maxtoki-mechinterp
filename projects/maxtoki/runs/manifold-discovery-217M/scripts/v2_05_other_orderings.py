"""v2 step 5: fill the H38 / H95 / H103 cells of the manifold table with re-computed values and donor-level intervals.

 'fit' mode
   - H38 LITE and H95: head trained on the internal kept anchors exactly as scripts/phase15_validate_candidate.py
     (same ruler builders, imported read-only), then the FROZEN head is scored on the external and zero-shot panels
     with phase15's own evaluate_frozen (trust k=15 on the kept anchors; random / donor / category holdouts).
   - H103: head trained on the 45 internal B-lineage anchors as scripts/phase7_h103_external.py. Scored on
       (a) the kept B-lineage anchors, as deployed (external 106 anchors k=15; zero-shot 18 anchors k=5, the deployed
           value; k=8 is the largest k sklearn allows for 18 points),
       (b) ALL anchors of each panel at k=15 (trust needs no ruler, so the whole 160-anchor zero-shot panel can be used).
   - In-sample internal trust is recomputed for all three (reproduces the registry values).
   Saves per (ordering, panel) the arrays needed for the bootstrap to outputs/v2_intervals/orderings/.
 'boot <key> <n_reps> <budget_s>'  donor cluster bootstrap (frozen head, donors with replacement,
     rng default_rng([5353, key_code, r])), trust + within-category/depth mean + donor mean; resumable.
 'summarize'  writes v2_05_other_orderings.json
"""
import sys; sys.dont_write_bytecode = True
SCR = "<REPO_ROOT>/projects/maxtoki/runs/manifold-discovery-217M/scripts"
sys.path.insert(0, SCR)
import json, math, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.manifold import trustworthiness
from v2_common import *
from v2_common import summarize as summarize_stats

ODIR = OUT / "orderings"; ODIR.mkdir(parents=True, exist_ok=True)
BOOT_SEED = 5353


def fit_mode():
    import phase15_validate_candidate as P15            # read-only import (no bytecode written)
    from phase3a_sweep2_ordinal import H103_DEPTH, build_ordinal_ruler as ord_ruler_sweep2
    gates = json.loads((REP / "quality_gates_spec.json").read_text())["gates"]
    FZ = Frozen()
    f_int = FZ.F_int; m_int = FZ.m_int
    res = {}
    panels = {}
    for p in ["external", "zeroshot"]:
        c, d, m = load_panel(p)
        panels[p] = (np.asarray(c), m)

    # ---------------- H38 LITE and H95 via phase15 code path
    for name in ["H38_lite", "H95"]:
        t0 = time.time()
        spec = P15.CAND_REGISTRY[name]
        D_int, prim_int, cats_int, keep_int = P15.build_ruler_for_panel(spec, m_int)
        f_k = f_int[keep_int]
        head = train_let(f_k, D_int)
        z_in = head_z(head, f_k)
        R = {"internal": {"n_anchors_kept": int(keep_int.sum()), "trust_in_sample_k15": float(trustworthiness(f_k, z_in, n_neighbors=15))}}
        for p in ["external", "zeroshot"]:
            cen, meta = panels[p]
            R[p] = P15.evaluate_frozen(head, FZ.mu, FZ.sd, FZ.A_e, FZ.A_m, FZ.A_l, FZ.part, cen, meta, spec, p, gates)
            # arrays for the bootstrap
            f = ((build_pooled_drift(cen, FZ.A_e, FZ.A_m, FZ.A_l, FZ.part) - FZ.mu) / FZ.sd).astype(np.float32)
            D, prim, cats, keep = P15.build_ruler_for_panel(spec, meta)
            fk = f[keep]; zk = head_z(head, fk)
            np.savez(ODIR / f"{name}_{p}.npz", f=fk, z=zk, D=D, groups=np.array(prim).astype(str),
                     donors=meta.loc[keep, "donor_id"].astype(str).to_numpy())
        R["seconds"] = time.time() - t0
        res[name] = R
        print(name, json.dumps({k: (v.get("trustworthiness", v.get("trust_in_sample_k15")) if isinstance(v, dict) else v) for k, v in R.items()}), flush=True)

    # ---------------- H103 as phase7_h103_external.py
    t0 = time.time()
    D_full, depths_int = ord_ruler_sweep2(m_int["cell_type"].astype(str).tolist(), H103_DEPTH)
    keep_int = depths_int >= 0
    ki = np.where(keep_int)[0]
    D_int = D_full[np.ix_(ki, ki)]
    head = train_let(f_int[keep_int], D_int)
    R = {"internal": {"n_anchors_kept": int(keep_int.sum()),
                      "trust_in_sample_k15_kept": float(trustworthiness(f_int[keep_int], head_z(head, f_int[keep_int]), n_neighbors=15)),
                      "trust_k15_all_internal_anchors": float(trustworthiness(f_int, head_z(head, f_int), n_neighbors=15)),
                      "n_all": int(len(f_int))}}
    for p in ["external", "zeroshot"]:
        cen, meta = panels[p]
        f = ((build_pooled_drift(cen, FZ.A_e, FZ.A_m, FZ.A_l, FZ.part) - FZ.mu) / FZ.sd).astype(np.float32)
        Dp, dep = ord_ruler_sweep2(meta["cell_type"].astype(str).tolist(), H103_DEPTH)
        keep = dep >= 0; kk = np.where(keep)[0]
        D = Dp[np.ix_(kk, kk)]
        fk = f[keep]; zk = head_z(head, fk); z_all = head_z(head, f)
        donors_k = meta.loc[keep, "donor_id"].astype(str).to_numpy()
        n_k = int(keep.sum())
        Dh = arccos_dist(zk); o = np.arange(n_k)
        groups = dep[keep].astype(int).astype(str)
        # phase7_h103-style holdouts on kept anchors (groups with constant ruler skipped)
        rnd = []
        rng = np.random.default_rng(SEED)
        for _ in range(20):
            idx = rng.permutation(n_k); n_test = max(2, int(round(n_k * 0.2))); test = sorted(idx[:n_test])
            Dt = D[np.ix_(test, test)]; iu = np.triu_indices(len(test), 1)
            if len(iu[0]) == 0 or Dt[iu].std() < 1e-6:
                continue
            rho = spearmanr(arccos_dist(zk[test])[iu], Dt[iu])[0]
            if not np.isnan(rho):
                rnd.append(float(rho))
        don_mean, n_don = group_spearman(Dh, D, donors_k, o, skip=())
        dep_mean, n_dep = group_spearman(Dh, D, groups, o, skip=())
        entry = {"n_anchors_kept": n_k, "n_donors_kept": int(len(set(donors_k))),
                 "n_distinct_depths_kept": int(len(set(groups))),
                 "trust_kept_k15": float(trustworthiness(fk, zk, n_neighbors=15)) if n_k > 30 else None,
                 "trust_kept_k5": float(trustworthiness(fk, zk, n_neighbors=5)) if n_k > 10 else None,
                 "trust_kept_k8": float(trustworthiness(fk, zk, n_neighbors=8)) if n_k > 16 else None,
                 "trust_all_anchors_k15": float(trustworthiness(f, z_all, n_neighbors=15)),
                 "n_all_anchors": int(len(f)), "n_donors_all": int(meta["donor_id"].nunique()),
                 "global_spearman_kept": global_spearman(Dh, D, o),
                 "random_holdout_kept": float(np.mean(rnd)) if rnd else None,
                 "donor_holdout_kept": don_mean, "n_donor_groups": n_don,
                 "depth_holdout_kept": dep_mean, "n_depth_groups": n_dep}
        R[p] = entry
        np.savez(ODIR / f"H103_{p}_all.npz", f=f, z=z_all, donors=meta["donor_id"].astype(str).to_numpy())
        np.savez(ODIR / f"H103_{p}_kept.npz", f=fk, z=zk, D=D, groups=groups, donors=donors_k)
    R["seconds"] = time.time() - t0
    res["H103"] = R
    print("H103", json.dumps(R, indent=1), flush=True)
    res["deployed"] = {k: json.loads((REP / f).read_text()) for k, f in [
        ("H38_internal", "h38_lite_quality_gates.json"), ("H38_external", "external_validation_H38_lite.json"),
        ("H38_zeroshot", "zeroshot_H38_lite.json"), ("H95_external_inline", "external_validation_h95.json"),
        ("H95_zeroshot", "zeroshot_H95.json"), ("H103_external", "external_validation_h103.json"),
        ("H103_zeroshot_inline", "zeroshot_h103.json")]}
    reg3 = pd.read_csv(REP / "hypothesis_registry_3gate.csv").set_index("name")
    reg2 = pd.read_csv(REP / "hypothesis_registry_sweep2.csv").set_index("name")
    res["deployed"]["H95_internal_trust"] = float(reg3.loc["H95_effector_modality", "trust_pos"])
    res["deployed"]["H103_internal_trust"] = float(reg2.loc["H103_B_cell_maturation", "positive_trust"])
    (OUT / "v2_05_other_orderings_fit.json").write_text(json.dumps(res, indent=2, default=float))
    write_run_config("v2_05_other_orderings_fit", list(CORE_INPUTS) + [Path(SCR) / n for n in [
        "phase15_validate_candidate.py", "phase14_sweep3.py", "phase13_h38_lite.py", "phase3a_manifold_sweep.py",
        "phase3a_sweep2_ordinal.py", "phase7_h103_external.py"]], {"seed_head": SEED})


BOOT_KEYS = {"H38_lite_external": 1, "H38_lite_zeroshot": 2, "H95_external": 3, "H95_zeroshot": 4,
             "H103_external_all": 5, "H103_zeroshot_all": 6}


def load_key(key):
    a = np.load(ODIR / f"{key}.npz", allow_pickle=True)   # our own files; donor ids were saved as object arrays
    donors = a["donors"].astype(str)
    if key.startswith("H103"):
        return dict(DX=euclid(a["f"]), DZ=euclid(a["z"]), Dh=None, D=None, groups=None, donors=donors)
    return dict(DX=euclid(a["f"]), DZ=euclid(a["z"]), Dh=arccos_dist(a["z"]), D=a["D"], groups=a["groups"].astype(str), donors=donors)


def metr(P, idx, dgroup):
    out = {"trust": trust_masked(P["DX"][np.ix_(idx, idx)], P["DZ"][np.ix_(idx, idx)], idx) if len(idx) > 30 else float("nan"),
           "n_anchors": int(len(idx))}
    if P["D"] is not None:
        Dh = P["Dh"][np.ix_(idx, idx)]; Dt = P["D"][np.ix_(idx, idx)]
        out["category"], out["n_groups"] = group_spearman(Dh, Dt, P["groups"][idx], idx, skip=())
        out["donor"], out["n_donor_groups"] = group_spearman(Dh, Dt, dgroup, idx, skip=())
    return out


def boot_mode(key, n_reps, budget):
    T0 = time.time(); P = load_key(key)
    fn = ODIR / f"boot_{key}.jsonl"
    done = {json.loads(l)["rep"] for l in fn.read_text().splitlines() if l.strip()} if fn.exists() else set()
    ud = np.array(sorted(set(P["donors"]))); mem = {u: np.where(P["donors"] == u)[0] for u in ud}
    with open(fn, "a") as fh:
        for r in range(n_reps):
            if r in done:
                continue
            if time.time() - T0 > budget:
                break
            rng = np.random.default_rng([BOOT_SEED, BOOT_KEYS[key], r])
            pick = rng.choice(ud, size=len(ud), replace=True)
            idx = np.concatenate([mem[u] for u in pick])
            grp = np.concatenate([np.full(len(mem[u]), f"{u}#{k}") for k, u in enumerate(pick)])
            fh.write(json.dumps({"rep": r, **metr(P, idx, grp)}) + "\n"); fh.flush()
    print(key, "done", len({json.loads(l)["rep"] for l in fn.read_text().splitlines() if l.strip()}), f"{time.time()-T0:.0f}s")


def summarize_mode():
    fit = json.loads((OUT / "v2_05_other_orderings_fit.json").read_text())
    out = {"fit": fit, "bootstrap": {}}
    for key in BOOT_KEYS:
        fn = ODIR / f"boot_{key}.jsonl"
        if not fn.exists():
            continue
        df = pd.DataFrame([json.loads(l) for l in fn.read_text().splitlines() if l.strip()])
        P = load_key(key); allidx = np.arange(len(P["donors"]))
        R = {"n_reps": int(len(df)), "observed": metr(P, allidx, P["donors"]),
             "resampling": "donor cluster bootstrap, head frozen, percentile 2.5/97.5",
             "rng": f"default_rng([{BOOT_SEED}, {BOOT_KEYS[key]}, rep])",
             "trust": summarize_stats(df["trust"].to_numpy(), GATE_TRUST)}
        for k in ["category", "donor"]:
            if k in df:
                R[k] = summarize_stats(df[k].to_numpy(), GATE_CORR)
        lodo = {}
        for u in sorted(set(P["donors"])):
            idx = np.where(P["donors"] != u)[0]
            lodo[u] = metr(P, idx, P["donors"][idx])
        R["lodo_trust_range"] = [float(np.nanmin([v["trust"] for v in lodo.values()])), float(np.nanmax([v["trust"] for v in lodo.values()]))]
        if "category" in df:
            R["lodo_category_range"] = [float(np.nanmin([v["category"] for v in lodo.values()])), float(np.nanmax([v["category"] for v in lodo.values()]))]
        out["bootstrap"][key] = R
        print(key, R["observed"], R["trust"]["ci95_percentile"], R.get("category", {}).get("ci95_percentile"))
    (OUT / "v2_05_other_orderings.json").write_text(json.dumps(out, indent=2, default=float))
    write_run_config("v2_05_other_orderings_boot", [OUT / "v2_05_other_orderings_fit.json"] +
                     sorted(ODIR.glob("*.npz")), {"boot_seed": BOOT_SEED, "keys": BOOT_KEYS})


if __name__ == "__main__":
    if sys.argv[1] == "fit":
        fit_mode()
    elif sys.argv[1] == "boot":
        boot_mode(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]))
    else:
        summarize_mode()
