"""v2 step 3: summarise the re-fit tasks in outputs/v2_intervals/refit_pool/results.jsonl.

 - internal reproduction (1 and 4 torch threads): trust, branch-holdout per branch, donor-holdout per donor;
   plain mean, anchor-weighted mean, pair-weighted mean, t interval over the 6 branch values
 - optimiser-noise check: torch seeds 1..10 (H65 and null)
 - bootstrap over the 11 TRAINING donors with head re-fit (only complete replicates are used):
   trust, branch-holdout (plain and anchor-weighted), and the two-level external / zero-shot numbers
Writes outputs/v2_intervals/v2_03_refit_summary.json and v2_03_internal_per_branch.csv,
v2_03_boot_internal_reps.csv.
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, "<REPO_ROOT>/projects/maxtoki/runs/manifold-discovery-217M/scripts")
import json
import numpy as np, pandas as pd
from scipy.stats import t as tdist
from v2_common import *
from v2_common import summarize as S

rows = [json.loads(l) for l in (OUT / "refit_pool/results.jsonl").read_text().splitlines() if l.strip()]
for r in rows:
    r.setdefault("threads", 1)  # rows from the first chunk were written before the field existed; that chunk used 1 thread
byid = {r["id"]: r for r in rows}
m = pd.read_csv(ART / "anchors/anchor_meta_internal.csv")
br = branch_labels(m); don = m["donor_id"].astype(str).to_numpy(); stages = m["hema_stage"].astype(str).to_numpy()
qg = json.loads((REP / "quality_gates_let_anchor.json").read_text())
out = {"published": {"H65": qg["positive"], "null": qg["null_shuffled"]}}


def branch_table(prefix, lab):
    rr = [r for r in rows if r["id"].startswith(f"{prefix}repro_branch|{lab}|")]
    tab = []
    for r in rr:
        sel = br == r["group"]
        iu = int(sel.sum() * (sel.sum() - 1) / 2)
        tab.append({"ruler": lab, "threads": r.get("threads", 1), "branch": r["group"], "n_anchors": int(sel.sum()),
                    "n_stages": int(len(set(stages[sel]))), "stages": "|".join(sorted(set(stages[sel]))),
                    "n_donors": int(len(set(don[sel]))), "n_pairs": iu, "rho": r["rho"]})
    return tab


def means(tab):
    v = np.array([t["rho"] for t in tab], float); w = np.array([t["n_anchors"] for t in tab], float)
    wp = np.array([t["n_pairs"] for t in tab], float)
    q = float(tdist.ppf(0.975, len(v) - 1)); se = float(v.std(ddof=1) / np.sqrt(len(v)))
    return {"plain_mean": float(v.mean()), "anchor_weighted_mean": float((w * v).sum() / w.sum()),
            "pair_weighted_mean": float((wp * v).sum() / wp.sum()), "n_branches": int(len(v)),
            "t_interval_95_over_branches": [float(v.mean() - q * se), float(v.mean() + q * se)],
            "largest3_branches_mean": float(np.mean([t["rho"] for t in sorted(tab, key=lambda t: -t["n_anchors"])[:3]])),
            "smallest3_branches_mean": float(np.mean([t["rho"] for t in sorted(tab, key=lambda t: -t["n_anchors"])[3:]]))}


allrows = []
for prefix, th in [("", 1), ("t4|", 4)]:
    R = {}
    for lab in ["H65", "null"]:
        full = byid.get(f"{prefix}repro_full|{lab}")
        tab = branch_table(prefix, lab)
        allrows += tab
        R[lab] = {"trust_in_sample": full["trust"] if full else None,
                  "global_in_sample": full["global_rho"] if full else None,
                  "branch_holdout": means(tab) if len(tab) >= 2 else None}
    dn = [r for r in rows if r["id"].startswith(f"{prefix}repro_donor|H65|")]
    R["H65"]["donor_holdout_per_donor"] = {r["group"]: {"n_anchors": int((don == r["group"]).sum()), "rho": r["rho"]} for r in dn}
    v = [r["rho"] for r in dn if r["rho"] is not None]
    R["H65"]["donor_holdout_mean"] = float(np.mean(v)) if v else None
    w = [int((don == r["group"]).sum()) for r in dn if r["rho"] is not None]
    R["H65"]["donor_holdout_anchor_weighted_mean"] = float(np.average(v, weights=w)) if v else None
    out[f"internal_reproduction_threads{th}"] = R
pd.DataFrame(allrows).to_csv(OUT / "v2_03_internal_per_branch.csv", index=False)

# ---------------------------------------------------------------- optimiser noise (seeds)
sd = {}
for lab in ["H65", "null"]:
    rr = [r for r in rows if r["id"].startswith(f"seed|{lab}|")]
    if rr:
        tr = np.array([r["trust"] for r in rr]); gl = np.array([r["global_rho"] for r in rr])
        sd[lab] = {"n_seeds": len(rr), "trust_min": float(tr.min()), "trust_max": float(tr.max()), "trust_mean": float(tr.mean()),
                   "trust_sd": float(tr.std(ddof=1)), "global_min": float(gl.min()), "global_max": float(gl.max()),
                   "frac_trust_below_0.80": float(np.mean(tr < 0.80)), "values_trust": tr.tolist()}
if "H65" in sd and "null" in sd:
    sd["H65_minus_null_trust_mean"] = sd["H65"]["trust_mean"] - sd["null"]["trust_mean"]
out["optimiser_noise_seeds_1_to_10"] = sd

# ---------------------------------------------------------------- bootstrap over training donors
tasks_meta = json.loads((OUT / "refit_pool/task_list.json").read_text())
sys.path.insert(0, str(RUN / "scripts"))
import v2_02_refit_pool as P
P.setup()
need = {}
for t in P.build_tasks(tasks_meta["n_boot"], 0):
    if t["kind"] in ("boot_full", "boot_branch"):
        need.setdefault(t["rep"], []).append(t["id"])
reps = []
for r, ids in sorted(need.items()):
    if not all(i in byid for i in ids):
        continue
    full = byid[f"boot_full|{r}"]
    bb = [byid[i] for i in ids if i.startswith("boot_branch")]
    sc = [b for b in bb if b["rho"] is not None]
    v = np.array([b["rho"] for b in sc]); w = np.array([b["n_test"] for b in sc], float)
    row = {"rep": r, "n_train": full["n_train"], "n_TSP2_copies": full["n_TSP2"], "trust": full["trust"],
           "global": full["global_rho"], "branch": float(v.mean()) if len(v) else np.nan,
           "branch_aw": float((w * v).sum() / w.sum()) if len(v) else np.nan, "n_branches": int(len(v)),
           **{f"rho_{b['group']}": b["rho"] for b in bb}}
    for p in ["external", "zeroshot"]:
        for lvl in ["trainonly", "bothlevels"]:
            for k in ["trust", "branch", "global"]:
                row[f"{p}_{lvl}_{k}"] = full[f"{p}_{lvl}"][k]
    reps.append(row)
df = pd.DataFrame(reps)
df.to_csv(OUT / "v2_03_boot_internal_reps.csv", index=False)
B = {"n_complete_reps": int(len(df)), "resampling": "11 training donors drawn with replacement (rng default_rng([20261001, rep])); "
     "head re-fitted on the resampled rows (features re-standardised on them); trust in-sample with copies never "
     "neighbours; branch-holdout re-fits one head per held-out branch; percentile 2.5/97.5"}
if len(df):
    B["internal_trust"] = S(df["trust"], GATE_TRUST)
    B["internal_branch_plain"] = S(df["branch"], GATE_CORR)
    B["internal_branch_anchor_weighted"] = S(df["branch_aw"], GATE_CORR)
    B["internal_global"] = S(df["global"])
    B["n_branches_scored_range"] = [int(df.n_branches.min()), int(df.n_branches.max())]
    B["n_train_range"] = [int(df.n_train.min()), int(df.n_train.max())]
    B["frac_reps_without_TSP2"] = float(np.mean(df.n_TSP2_copies == 0))
    for c in [c for c in df.columns if c.startswith("rho_")]:
        B[f"per_branch_{c[4:]}"] = S(df[c].astype(float), GATE_CORR) if df[c].notna().sum() > 2 else None
    for p in ["external", "zeroshot"]:
        for lvl in ["trainonly", "bothlevels"]:
            for k in ["trust", "branch"]:
                B[f"{p}_{lvl}_{k}"] = S(df[f"{p}_{lvl}_{k}"].astype(float), GATE_TRUST if k == "trust" else GATE_CORR)
out["bootstrap_training_donors"] = B
(OUT / "v2_03_refit_summary.json").write_text(json.dumps(out, indent=2, default=float))
write_run_config("v2_03_aggregate_refit", [OUT / "refit_pool/results.jsonl"] + CORE_INPUTS, {"boot_seed": 20261001})
for k in ["internal_reproduction_threads1", "internal_reproduction_threads4"]:
    print(k, json.dumps({lab: {kk: vv for kk, vv in out[k][lab].items() if kk != "donor_holdout_per_donor"} for lab in ["H65", "null"]}, indent=1, default=float))
print(json.dumps({k: v for k, v in sd.items() if not isinstance(v, dict)}, indent=1))
print(json.dumps({k: (v["ci95_percentile"] if isinstance(v, dict) and "ci95_percentile" in v else v) for k, v in B.items()}, indent=1, default=float))
