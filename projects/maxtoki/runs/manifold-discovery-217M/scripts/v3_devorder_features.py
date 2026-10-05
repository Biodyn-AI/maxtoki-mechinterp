"""V3-7 step 2: every anchor representation scored by the v2b gates, rebuilt on the v3 (correct-encoding) inputs.

Usage: python v3_devorder_features.py <part>
  maxtoki   v2b_devorder_01_features.part_maxtoki, run unchanged on the v3 centroids (patched paths):
            maxtoki (2,464-number pooled drift, internal mean/SD) and maxtoki_pca64.
  lookup    v2b part_lookup, unchanged (labels only); checked to be identical to the v2b files.
  tokenbag  the v2b token-bag recipe (weight 1 - r/n per gene token at rank r of n; anchor mean; log1p; genes with
            SD > 0 on internal; standardise; PCA-64 on internal) on the v3 tokens, i.e. the rank order the model now
            reads. The same code run on the deployed tokens must reproduce the v2b file exactly (checked here).
  hvg       the v2b HVG feature never used the model input (raw/X counts of the same cells). It is copied from v2b
            and its sha256 checked; the copy is recorded in run_config.
  manifest  features_manifest.json (sha256 + shape of every feature file)
"""
import sys
sys.dont_write_bytecode = True
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, "<REPO_ROOT>/projects/maxtoki/runs/manifold-discovery-217M/scripts")
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from v3_devorder_common import (V3, V3FEAT, V3ANCH, CELLS, PH1, V2B, SCR, patch_v2b, assert_paths_v3, sha256_file,
                                write_json, selected_cells)

V = patch_v2b()
import v2b_devorder_01_features as F1   # noqa: E402  (imported after the patch: its copied paths are the v3 ones)
assert_paths_v3(F1)
assert F1.FEAT == V3FEAT and F1.ART == V3.parent / "v3_devorder" / "artifacts"

PANELS = V.PANELS            # internal, external, zeroshot, lung_nonhema (the v2b panels)


def tokenbag_from(tok_of_panel):
    """v2b_devorder_01_features.part_tokenbag maths; tok_of_panel(p) -> (token_ids (n_sel, L), seq_lens (n_sel,),
    anchor_id per selected cell)."""
    data = {p: tok_of_panel(p) for p in PANELS}
    Vv = max(int(d[0].max()) + 1 for d in data.values())
    bags = {}
    for p in PANELS:
        tok, L, a_col = data[p]
        m = V.meta(p)
        idx_of = {a: i for i, a in enumerate(m["anchor_id"].astype(str))}
        bag = np.zeros((len(m), Vv)); cnt = np.zeros(len(m))
        for r in range(len(L)):
            l = int(L[r]); ai = idx_of[a_col[r]]; cnt[ai] += 1
            if l < 3:
                continue
            g = tok[r, 1:l - 1]; n = len(g)
            np.add.at(bag[ai], g, 1.0 - np.arange(n) / max(n, 1))
        bags[p] = np.log1p(bag / cnt[:, None])
    ok = bags["internal"].std(0) > 0
    Fs = F1.standardise_all({p: b[:, ok] for p, b in bags.items()})
    S, evr = F1.pca_rep(Fs)
    return S, {"vocab": Vv, "genes_kept": int(ok.sum()), "pca64_evr_sum": float(np.sum(evr))}


def deployed_tokens(p):
    keep, sub = selected_cells(p)
    z = np.load(PH1 / f"cells_{p}.npz")
    return z["token_ids"][keep], z["seq_lens"][keep], sub["anchor_id"].astype(str).to_numpy()


def v3_tokens(p):
    z = np.load(CELLS / f"tokens_{p}.npz")
    plan = pd.read_csv(CELLS / f"plan_{p}.csv")
    assert np.array_equal(z["rows"], plan["cell_idx"].to_numpy())
    return z["token_ids"], z["seq_lens"], plan["anchor_id"].astype(str).to_numpy()


def part_tokenbag():
    # 1) the recipe on the deployed tokens must give the v2b file exactly
    S_dep, info_dep = tokenbag_from(deployed_tokens)
    repro = {p: float(np.abs(S_dep[p] - np.load(V2B / f"features/tokenbag_pca64__{p}.npy")).max()) for p in PANELS}
    print("deployed-token recipe vs v2b file, max abs diff:", repro, flush=True)
    if max(repro.values()) > 1e-5:
        raise RuntimeError(f"token-bag recipe does not reproduce v2b: {repro}")
    # 2) the same recipe on the v3 tokens
    S, info = tokenbag_from(v3_tokens)
    F1.save("tokenbag_pca64", S)
    return {"tokenbag_v3": info, "tokenbag_deployed_recipe_check": {"max_abs_diff_vs_v2b": repro, **info_dep}}


def part_lookup():
    info = F1.part_lookup()
    same = {}
    for f in sorted(V3FEAT.glob("lookup_*.npy")):
        same[f.name] = sha256_file(f) == sha256_file(V2B / "features" / f.name)
    print("lookup identical to v2b:", all(same.values()), flush=True)
    if not all(same.values()):
        raise RuntimeError(f"lookup features differ from v2b: {same}")
    return {**info, "identical_to_v2b": same}


def part_hvg():
    out = {}
    for f in sorted((V2B / "features").glob("hvg_*.npy")):
        if f.name.startswith("._"):
            continue
        dst = V3FEAT / f.name
        shutil.copyfile(f, dst)
        a, b = sha256_file(f), sha256_file(dst)
        if a != b:
            raise RuntimeError(f"copy of {f.name} differs")
        out[f.name] = a
    return {"hvg_copied_from_v2b_sha256": out,
            "reason": "built from raw/X counts of the same cells (v2b_devorder_01_features.part_hvg); it never used the "
                      "model input, so the encoding fix does not change it"}


def part_maxtoki():
    info = F1.part_maxtoki()
    return info


def part_manifest():
    files = sorted(f for f in V3FEAT.glob("*.npy") if not f.name.startswith("._"))
    man = {f.name: {"sha256": sha256_file(f), "shape": list(np.load(f, mmap_mode="r").shape)} for f in files}
    write_json(V3 / "features_manifest.json", man)
    print(len(man), "feature files")


if __name__ == "__main__":
    import torch
    part = sys.argv[1]
    t0 = time.time()
    info = {"maxtoki": part_maxtoki, "lookup": part_lookup, "tokenbag": part_tokenbag, "hvg": part_hvg,
            "manifest": part_manifest}[part]()
    if part != "manifest":
        inputs = [V3ANCH / f"anchor_meta_{p}.csv" for p in PANELS]
        if part == "maxtoki":
            inputs += [V3ANCH / f"centroids_{p}.npy" for p in PANELS] + [
                V3.joinpath("artifacts/operators/pooled_drift_components.npz"),
                V3.joinpath("artifacts/operators/operator_index.json")]
        if part == "tokenbag":
            inputs += [CELLS / f"tokens_{p}.npz" for p in PANELS] + [CELLS / f"plan_{p}.csv" for p in PANELS] + \
                      [PH1 / f"cells_{p}.npz" for p in PANELS] + [V2B / f"features/tokenbag_pca64__{p}.npy" for p in PANELS]
        write_json(V3 / f"run_config_features_{part}.json", {
            "script": str(Path(__file__).resolve()), "script_sha256": sha256_file(__file__),
            "imports": {"v2b_devorder_01_features.py": sha256_file(SCR / "v2b_devorder_01_features.py"),
                        "v2b_devorder_common.py": sha256_file(SCR / "v2b_devorder_common.py"),
                        "v3_devorder_common.py": sha256_file(SCR / "v3_devorder_common.py")},
            "written_at": time.strftime("%Y-%m-%d %H:%M:%S"), "python": sys.version.split()[0],
            "numpy": np.__version__, "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
            "inputs_sha256": {str(p): sha256_file(p) for p in inputs},
            "seeds": {"pca": "sklearn PCA random_state 42", "lookup_ct": "default_rng([7001,k]) / ([7001,k,panel])",
                      "lookup_cls": "default_rng([7002,k]) / ([7002,k,panel])"},
            "info": info, "seconds": time.time() - t0})
    print(part, "done", f"{time.time()-t0:.0f}s", json.dumps(info, default=str)[:1500] if part != "manifest" else "")
