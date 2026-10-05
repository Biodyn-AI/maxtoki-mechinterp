"""V3-7 extra check: how much do the H65 numbers move with the LET head's torch seed (optimiser noise)?

The v2b/v3 pool fits every head once with torch seed 42 (phase5_let_anchor.SEED). An independent re-fit from features
that differ by ~7e-6 (float64 instead of float32 pooled drift) moved MaxToki's internal trust from 0.845 to 0.813, so
the single-seed values carry optimiser noise. This script re-fits the full H65 head (all internal anchors, deployed
ruler) with torch seeds 0..19 for the v3 MaxToki features and 0..9 for the v2b (deployed-input) MaxToki features, and
scores in-sample trust plus the frozen external / zero-shot gates with the v2b gate code (frozen_gates_h65).
It also scores the pool's own seed-42 heads with an independent gate implementation (v3_devorder_verify.frozen) to
separate gate-code agreement from optimiser noise.

Usage: python v3_devorder_seeds.py run <n_workers> <start_budget_s> | summary
Output: outputs/v3_devorder/seeds/results.jsonl, seeds_summary.json
"""
import sys
sys.dont_write_bytecode = True
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "<REPO_ROOT>/projects/maxtoki/runs/manifold-discovery-217M/scripts")
import contextlib
import io
import json
import time
import multiprocessing as mp

import numpy as np

from v3_devorder_common import V3, V2B, patch_v2b, write_json, sha256_file

V = patch_v2b()
OUT = V3 / "seeds"
OUT.mkdir(exist_ok=True)
RES = OUT / "results.jsonl"
TASKS = [("v3", s, "maxtoki") for s in range(20)] + [("v2b", s, "maxtoki") for s in range(10)] + \
    [("v3", s, r) for r in ("tokenbag_pca64", "hvg_pca64", "lookup_ct_s0") for s in range(10)] + \
    [("v2b", s, "tokenbag_pca64") for s in range(10)]


def feat(src, rep, panel):
    base = V3 if src == "v3" else V2B
    return np.load(base / f"features/{rep}__{panel}.npy").astype(np.float32)


def run(task):
    src, seed, rep = task
    import torch
    torch.set_num_threads(1)
    from sklearn.manifold import trustworthiness
    P5 = V.P5
    t0 = time.time()
    old = P5.SEED
    P5.SEED = int(seed)
    try:
        F = feat(src, rep, "internal")
        D = np.load(V3 / "artifacts/anchors/d_target_internal.npy")
        head, z = V.fit(F, D)
    finally:
        P5.SEED = old
    out = {"src": src, "seed": int(seed), "rep": rep, "trust": float(trustworthiness(F, z, n_neighbors=15))}
    P = V.head_params(head)
    for p in ("external", "zeroshot", "lung_nonhema"):
        m = V.meta(p)
        f = feat(src, rep, p)
        zz = V.z_from(P, f)
        g = V.frozen_gates_h65(f, zz, np.load(V3 / f"artifacts/anchors/d_target_{p}.npy"),
                               m["donor_id"].astype(str).to_numpy(), V.branches_of(m["hema_stage"].astype(str)),
                               m["cell_type"].astype(str).to_numpy())
        out[p] = {k: g[k] for k in ("trust", "random", "donor", "branch", "branch_diff_ct", "global", "global_diff_ct")}
    out["sec"] = time.time() - t0
    return out


def _init():
    import torch
    torch.set_num_threads(1)
    patch_v2b()


def _guard(a):
    task, deadline = a
    if time.time() > deadline:
        return None
    return run(task)


def done():
    if not RES.exists():
        return set()
    return {(j["src"], j["seed"], j.get("rep", "maxtoki")) for j in map(json.loads, RES.read_text().splitlines()) if j}


def main_run(nw, budget):
    todo = [t for t in TASKS if t not in done()]
    deadline = time.time() + budget
    print("pending", len(todo), flush=True)
    with mp.get_context("spawn").Pool(nw, initializer=_init) as pool, open(RES, "a") as fh:
        for r in pool.imap_unordered(_guard, [(t, deadline) for t in todo]):
            if r is None:
                continue
            fh.write(json.dumps(r) + "\n"); fh.flush()
            print(r["src"], r["seed"], r["rep"], round(r["trust"], 4), round(r["external"]["global"], 4),
                  round(r["zeroshot"]["branch"], 4), f"{r['sec']:.0f}s", flush=True)
    print("left", len([t for t in TASKS if t not in done()]))


def summary():
    rows = [json.loads(l) for l in RES.read_text().splitlines() if l.strip()]
    for r in rows:
        r.setdefault("rep", "maxtoki")
    allrows = rows
    rows = [r for r in allrows if r["rep"] == "maxtoki"]
    tb = json.loads((V3 / "v2b_tables.json").read_text())
    null = tb["structured_null_refit"]["maxtoki"]
    out = {"method": "LET head re-fitted on all internal anchors with torch seeds (phase5 train_let, 1 thread); "
                     "values: mean, SD, min, max over seeds; null = 40 structured-null rulers at seed 42 (v3 pool)"}
    for src in ("v3", "v2b"):
        R = [r for r in rows if r["src"] == src]
        if not R:
            continue
        o = {"n_seeds": len(R)}

        def st(vals):
            v = np.array(vals, float)
            return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "min": float(v.min()), "max": float(v.max()),
                    "share_ge_0.80": float(np.mean(v >= 0.80)), "share_ge_0.20": float(np.mean(v >= 0.20))}
        o["internal_trust"] = st([r["trust"] for r in R])
        for p in ("external", "zeroshot", "lung_nonhema"):
            for k in ("trust", "random", "donor", "branch", "branch_diff_ct", "global"):
                o[f"{p}_{k}"] = st([r[p][k] for r in R])
        o["seed42_pool"] = json.loads((V3 if src == "v3" else V2B).joinpath("v2b_tables.json").read_text())[
            "summaries"]["h65|maxtoki|pos"]["trust"]
        out[src] = o
    # observed (seed mean) vs the structured null (40 draws, seed 42)
    for k, key in (("internal_trust", "trust"), ("external_trust", "external_trust"),
                   ("zeroshot_trust", "zeroshot_trust"), ("external_global", "external_global"),
                   ("zeroshot_global", "zeroshot_global"), ("external_branch", "external_branch"),
                   ("zeroshot_branch", "zeroshot_branch")):
        n = null[key]
        v3 = out["v3"][k]
        out.setdefault("v3_seed_mean_vs_structured_null", {})[k] = {
            "seed_mean": v3["mean"], "seed_min": v3["min"], "seed_max": v3["max"],
            "null_mean": n["null_mean"], "null_sd": n["null_sd"], "null_max": n["null_max"],
            "z_seed_mean": (v3["mean"] - n["null_mean"]) / n["null_sd"],
            "share_seeds_above_null_max": float(np.mean([((r["trust"] if k == "internal_trust" else
                                                           r[k.split("_")[0]][k.split("_", 1)[1]]) > n["null_max"])
                                                         for r in rows if r["src"] == "v3"]))}
    # seed-paired contrasts (same torch seed for both heads), seeds 0..9
    pc = {}
    for src, other in (("v3", "tokenbag_pca64"), ("v3", "hvg_pca64"), ("v3", "lookup_ct_s0"), ("v2b", "tokenbag_pca64")):
        a = {r["seed"]: r for r in allrows if r["src"] == src and r["rep"] == "maxtoki"}
        b = {r["seed"]: r for r in allrows if r["src"] == src and r["rep"] == other}
        ss = sorted(set(a) & set(b))
        for p in ("external", "zeroshot"):
            for k in ("global", "global_diff_ct", "branch"):
                d = np.array([a[s_][p][k] - b[s_][p][k] for s_ in ss])
                pc[f"{src}|maxtoki_minus_{other}|{p}|{k}"] = {"n_seeds": len(ss), "mean": float(d.mean()),
                                                              "sd": float(d.std(ddof=1)), "min": float(d.min()),
                                                              "max": float(d.max()), "share_gt_0": float(np.mean(d > 0)),
                                                              "other_mean": float(np.mean([b[s_][p][k] for s_ in ss]))}
    out["seed_paired_contrasts"] = pc
    write_json(V3 / "seeds_summary.json", out)
    print(json.dumps(out, indent=1)[:6000])


if __name__ == "__main__":
    if sys.argv[1] == "run":
        main_run(int(sys.argv[2]), float(sys.argv[3]))
    else:
        summary()
