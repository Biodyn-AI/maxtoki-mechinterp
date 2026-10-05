"""V3-7 step 4: the v2b tables, structured nulls, lung structured control, donor bootstraps and random-projection
control, re-run unchanged on the v3 pool results.

The analysis code is scripts/v2b_devorder_03_analyze.py, imported unchanged after patch_v2b() (all its paths then point
at outputs/v3_devorder/). The same arguments as the v2b run are used:
  tables | evalnull | lung 6161 | lung 6262 | boot_frozen <panel> 2000 noglobal | boot_global <panel> |
  boot_internal | n60 | n60dim
Extra parts (v3 only):
  compare   deployed / v2b / v3 side by side for every key number -> v3_compare.json, table_v3_vs_v2b.csv
  config    run_config.json for the whole v3 analysis (sha256 of inputs, scripts, outputs; seeds; threads)
v2b writes files named v2b_*.json; in v3 they keep that name inside outputs/v3_devorder/ (the code is unchanged), and
the compare part reads both folders.
"""
import sys
sys.dont_write_bytecode = True
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "<REPO_ROOT>/projects/maxtoki/runs/manifold-discovery-217M/scripts")
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from v3_devorder_common import (V3, V2B, V3ANCH, V3FEAT, V3HEADS, CELLS, PERCELL, PH1, DEP_ART, RUN, SCR, SETUP,
                                patch_v2b, assert_paths_v3, sha256_file, write_json)

V = patch_v2b()
import v2b_devorder_03_analyze as A   # noqa: E402
assert_paths_v3(A)
assert A.RES == V3 / "pool" / "results.jsonl"


def jload(p):
    return json.loads(Path(p).read_text())


def part_compare():
    """Key numbers: deployed (from the run reports, as reproduced exactly by v2b), v2b (deployed inputs), v3."""
    out = {}
    tb = {"v2b": jload(V2B / "v2b_tables.json"), "v3": jload(V3 / "v2b_tables.json")}
    rows = []

    def add(name, unit, get):
        r = {"number": name, "unit": unit}
        for k in ("v2b", "v3"):
            try:
                r[k] = float(get(k))
            except Exception:
                r[k] = float("nan")
        r["change"] = r["v3"] - r["v2b"]
        rows.append(r)
    S = lambda k, key: tb[k]["summaries"][key]
    for rep in ["maxtoki", "maxtoki_pca64", "tokenbag_pca64", "hvg_pca64"]:
        add(f"H65 internal trust | {rep}", "trustworthiness k=15", lambda k: S(k, f"h65|{rep}|pos")["trust"])
        add(f"H65 internal random holdout | {rep}", "Spearman", lambda k: S(k, f"h65|{rep}|pos")["random"])
        add(f"H65 internal donor holdout | {rep}", "Spearman", lambda k: S(k, f"h65|{rep}|pos")["donor"])
        add(f"H65 internal branch holdout | {rep}", "Spearman, plain mean over branches",
            lambda k: S(k, f"h65|{rep}|pos")["branch"])
        add(f"H65 internal branch holdout, diff cell type | {rep}", "Spearman",
            lambda k: S(k, f"h65|{rep}|pos")["branch_diff_ct"])
        for p in ("external", "zeroshot", "lung_nonhema"):
            for m in ("trust", "random", "donor", "branch", "branch_diff_ct", "global", "global_diff_ct"):
                add(f"H65 {p} frozen {m} | {rep}", "Spearman (trust: trustworthiness)",
                    lambda k: S(k, f"h65|{rep}|pos")["frozen"][p][m])
    for fam, g in (("h38", "cat"), ("h95", "cat"), ("h103", "cat")):
        for m in ("trust", "random", "donor", g):
            add(f"{fam.upper()} internal {m} | maxtoki", "", lambda k: S(k, f"{fam}|maxtoki|pos")[m])
        for p in ("external", "zeroshot"):
            for m in ("trust", "random", "donor", "category"):
                add(f"{fam.upper()} {p} frozen {m} | maxtoki", "", lambda k: S(k, f"{fam}|maxtoki|pos")["frozen"][p][m])
    add("H65 run null internal trust | maxtoki", "", lambda k: S(k, "h65|maxtoki|runnull")["trust"])
    add("H65 run null internal branch | maxtoki", "", lambda k: S(k, "h65|maxtoki|runnull")["branch"])
    for rep in ["maxtoki", "lookup_ct_s0", "tokenbag_pca64", "hvg_pca64"]:
        for key in ("internal_branch", "external_branch", "external_global", "zeroshot_branch", "zeroshot_global",
                    "trust", "external_branch_diff_ct"):
            add(f"structured null p | {rep} | {key}", "one-sided p, refit null",
                lambda k: tb[k]["structured_null_refit"][rep][key]["p_one_sided"])
            add(f"structured null mean | {rep} | {key}", "", lambda k: tb[k]["structured_null_refit"][rep][key]["null_mean"])
    for f, keys in (("v2b_boot_external_global.json", ["global|maxtoki_minus_lookup", "global|maxtoki_minus_tokenbag",
                                                       "global|maxtoki_minus_hvg", "global|maxtoki",
                                                       "global_diff_ct|maxtoki_minus_tokenbag"]),
                    ("v2b_boot_zeroshot_global.json", ["global|maxtoki_minus_lookup", "global|maxtoki_minus_tokenbag",
                                                       "global|maxtoki_minus_hvg", "global|maxtoki",
                                                       "global_diff_ct|maxtoki_minus_tokenbag"]),
                    ("v2b_boot_external.json", ["branch|maxtoki_minus_lookup", "branch|maxtoki_minus_tokenbag",
                                                "branch|maxtoki_minus_hvg", "branch|maxtoki", "branch_diff_ct|maxtoki"]),
                    ("v2b_boot_zeroshot.json", ["branch|maxtoki_minus_lookup", "branch|maxtoki_minus_tokenbag",
                                                "branch|maxtoki_minus_hvg", "branch|maxtoki"]),
                    ("v2b_boot_internal_branch.json", ["branch|maxtoki_minus_lookup", "branch|maxtoki_minus_tokenbag",
                                                       "branch|maxtoki_minus_hvg", "branch|maxtoki"])):
        for key in keys:
            for part in ("observed", "lo", "hi"):
                def g(k, f=f, key=key, part=part):
                    d = jload((V2B if k == "v2b" else V3) / f)["results"][key]
                    return d["observed"] if part == "observed" else d["ci95"][0 if part == "lo" else 1]
                add(f"{f.replace('v2b_', '').replace('.json', '')} | {key} | {part}", "donor bootstrap 95% percentile", g)
    for f in ("v2b_evalnull_internal_branch.json",):
        for rep in ("maxtoki", "tokenbag_pca64", "hvg_pca64", "lookup_ct_s0"):
            for m in ("branch_all_pairs", "branch_diff_ct"):
                for part in ("observed", "null_mean", "p_one_sided"):
                    add(f"evalnull | {rep} | {m} | {part}", "", lambda k, rep=rep, m=m, part=part:
                        jload((V2B if k == "v2b" else V3) / f)["results"][rep][m][part])
    for tag in ("6161", "6262"):
        for rep in ("maxtoki", "lookup_ct_s0", "tokenbag_pca64", "hvg_pca64"):
            for m in ("branch", "random", "donor", "random_donor_branch"):
                def g(k, tag=tag, rep=rep, m=m):
                    d = jload((V2B if k == "v2b" else V3) / f"v2b_lung_control_{tag}.json")["results"]
                    return list(d.values())[0][rep]["pass_rate"][m]
                add(f"lung {tag} pass rate | {rep} | {m}", "share of 2,000 label draws", g)
    for k_ in ("trust", "random", "branch", "branch_diff_ct", "composite"):
        add(f"N60 L10H6 {k_}", "", lambda k, k_=k_: jload((V2B if k == "v2b" else V3) / "v2b_n60_random_projection.json")
            ["L10H6_recomputed"][k_])
        add(f"N60 random-proj (index 11) mean {k_}", "", lambda k, k_=k_: jload(
            (V2B if k == "v2b" else V3) / "v2b_n60_random_projection.json")["random_154_proj_of_centroid11"][k_]["mean"])
        add(f"N60 share random >= L10H6 {k_}", "", lambda k, k_=k_: jload(
            (V2B if k == "v2b" else V3) / "v2b_n60_random_projection.json")["random_154_proj_of_centroid11"][k_]
            ["share_random_ge_L10H6"])
    df = pd.DataFrame(rows)
    df.to_csv(V3 / "table_v3_vs_v2b.csv", index=False)
    write_json(V3 / "v3_compare.json", {"rows": rows})
    print(df.to_string(max_rows=400, max_colwidth=80))


def part_config():
    files = [A.RES, V3 / "pool" / "task_list.json", V3 / "features_manifest.json"] + \
        sorted(V3HEADS.glob("*")) + sorted(V3FEAT.glob("*.npy")) + sorted(V3.glob("v2b_*.json")) + \
        sorted(V3.glob("table_*.csv")) + sorted(V3.glob("v3_*.json")) + sorted((V3 / "thread_check").glob("*.jsonl")) + \
        [V3 / "seeds_summary.json", V3 / "seeds" / "results.jsonl"] + sorted((V3 / "verify").glob("*"))
    scripts = sorted(f for f in SCR.glob("v3_devorder_*.py") if not f.name.startswith("._")) + \
        sorted(f for f in SCR.glob("v2b_devorder_0*.py") if not f.name.startswith("._")) + [SCR / "v2b_devorder_common.py"]
    inputs = sorted(V3ANCH.glob("*")) + sorted((V3 / "artifacts/operators").glob("*")) + \
        [RUN / "planning/h65_stage_dag.json", RUN / "reports/quality_gates_spec.json", RUN / "reports/head_layer_screen.csv"]
    inputs += [SCR / f for f in ["phase5_let_anchor.py", "phase7_external_validation.py", "phase8_zeroshot_transfer.py",
                                 "phase9_head_attribution.py", "phase13_h38_lite.py", "phase3a_manifold_sweep.py",
                                 "phase3a_sweep2_ordinal.py", "phase15_validate_candidate.py"]]
    import torch, sklearn, scipy
    cfg = {"item": "V3-7 developmental ordering beyond cell-type identity, correct input encoding",
           "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
                        "sklearn": sklearn.__version__, "scipy": scipy.__version__, "pandas": pd.__version__},
           "device": "cpu (LET heads, statistics); centroids from MPS, see run_config_centroids.json",
           "torch_threads": "1 per worker (as v2b); thread check at 1 and 4",
           "inputs_sha256": {str(p): sha256_file(p) for p in inputs if p.is_file() and not p.name.startswith("._")},
           "scripts_sha256": {str(p): sha256_file(p) for p in scripts},
           "outputs_sha256": {str(p): sha256_file(p) for p in files if p.is_file() and not p.name.startswith("._")},
           "seeds": {"LET head": "torch seed 42 inside phase5 train_let",
                     "lookup_ct": "codes default_rng([7001,k]) k=0..4; noise default_rng([7001,k,panel])",
                     "lookup_cls": "codes default_rng([7002,k]) k=0..2; noise default_rng([7002,k,panel])",
                     "pca": "sklearn PCA random_state 42", "random holdout splits": "default_rng(42) as phase5",
                     "H65 structured null (refit)": "default_rng([2026, d]) d=0..39 (branch holdout d=0..19)",
                     "H65 structured null (eval-only)": "default_rng([2027, d]) d=0..1999",
                     "H38/H95 structured null": "default_rng([3838|9595, d]) d=0..29",
                     "lung per-anchor / per-cell-type": "default_rng([6161, d]) / default_rng([6262, d]) d=0..1999",
                     "donor bootstrap": "default_rng([8080, panel_index, b]) b=0..1999 (internal panel index 9)",
                     "random projections": "default_rng([6060, centroid_index, seed])"},
           "same_as_v2b": "task list, task code, gate code, seeds and arguments are those of the v2b run; only the "
                          "MaxToki centroids (hence maxtoki, maxtoki_pca64, N60) and the token bag (v3 tokens) changed"}
    write_json(V3 / "run_config.json", cfg)
    print("run_config.json:", len(cfg["inputs_sha256"]), "inputs,", len(cfg["outputs_sha256"]), "outputs,",
          len(scripts), "scripts")


if __name__ == "__main__":
    part = sys.argv[1]
    t0 = time.time()
    if part == "tables":
        A.part_tables()
    elif part == "evalnull":
        A.part_evalnull()
    elif part == "lung":
        A.part_lung(sys.argv[2])
    elif part == "boot_frozen":
        wg = True
        if len(sys.argv) > 4:
            wg = {"noglobal": False, "globalonly": "only"}.get(sys.argv[4], True)
        A.part_boot_frozen(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2000, with_global=wg)
    elif part == "boot_global":
        A.part_boot_global(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2000)
    elif part == "boot_internal":
        A.part_boot_internal()
    elif part == "n60":
        A.part_n60()
    elif part == "n60dim":
        A.part_n60dim()
    elif part == "compare":
        part_compare()
    elif part == "config":
        part_config()
    print(part, "done", f"{time.time()-t0:.0f}s")
