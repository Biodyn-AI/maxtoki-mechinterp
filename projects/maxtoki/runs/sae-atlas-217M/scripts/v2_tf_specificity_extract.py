"""D3 (revision): per-cell layer-5 SAE feature summaries for the TF-specificity re-run.

What this script does
---------------------
1. Builds a fixed cell manifest (seeds recorded) from the Replogle K562 data:
   * ``ref``     : 400 K562 non-targeting (NT) control cells = the reference group
                   for the responding-feature test.
   * ``pool``    : 800 other NT cells = held-out pool for "fake TFs".
   * ``kd``      : up to 100 knockdown cells for each TF knocked down in K562 that has
                   TRRUST or DoRothEA targets (87 TFs). The 20 with DoRothEA ChIP-seq
                   targets are the primary set and are processed first.
   * ``catalog`` : the 500 NT cells that full_12layer_pipeline.py used to build the
                   deployed top-20 catalog (seed 42). Used only to rebuild and check the
                   catalog. ``ref`` and ``pool`` never contain these cells.
2. For each cell: rank-value tokenisation (max_len 2048, same as the deployed runs),
   forward pass up to the INPUT of block 5 (= hidden_states[5], the tensor the layer-5
   SAE was trained on; hooks_v2 site convention), stop the forward pass there, encode
   with the layer-5 SAE, and save:
     mean_gene[f]  = mean of z_f over the cell's gene tokens (all positions except
                     <bos> and <eos>)                 -> the unit of inference
     nact_gene[f]  = number of gene tokens with z_f > 0
     z_eos[f]      = code at <eos> (float16); the <bos> code is the same for every
                     cell (causal model) and is stored once.
   For ``catalog`` cells it also stores the sparse codes of every position (top-32
   indices and values) and the token ids, to rebuild the top-20 catalog.
3. Chunked and resumable: ``--max-minutes`` bounds one call; progress is saved after
   every ``--save-every`` cells; a new call continues where the last one stopped.

Memory guard: before loading the model the script needs >= 3.0 GB available
(free + inactive pages, hooks_v2.available_memory_gb). While running, it stops
cleanly if available memory drops below ``--run-floor-gb`` (default 1.5 GB; the
model job itself holds ~1.7 GB of what vm_stat counts, so a 3 GB in-run floor
would stop every run immediately).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

PROJ = Path("<REPO_ROOT>/projects/maxtoki")
sys.path.insert(0, str(PROJ / "setup"))
import hooks_v2 as H  # noqa: E402
from maxtoki_adapter import MaxTokiTokenizer  # noqa: E402
from dataset_loader import resolve as load_ds, SYM2ENS_PKL  # noqa: E402

OUT = PROJ / "runs/sae-atlas-217M/outputs/v2_tf_specificity"
CELLDIR = OUT / "cells"
MANIFEST = OUT / "cell_manifest.csv"
PROGRESS = OUT / "extract_progress.json"
RUNCFG = OUT / "run_config_extract.json"

TRRUST_TSV = Path("<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv")
DOROTHEA_TSV = Path("<DATA_ROOT>/biodyn-work/single_cell_mechinterp/external/networks/dorothea_human.tsv")
CHIP_TSV = Path("<DATA_ROOT>/biodyn-work/single_cell_mechinterp/external/networks/dorothea_chipseq_human.tsv")

LAYER = 5
MAX_LEN = 2048
SEED_CATALOG = 42          # replicates full_12layer_pipeline.py (N_CTRL = 500)
N_CATALOG = 500
SEED_CTRL = 20261001       # ref + pool draw
N_REF = 400
N_POOL = 800
SEED_KD = 20261002         # per-TF seed = SEED_KD + index of TF in sorted list of 87
N_KD_MAX = 100


def _dec(a):
    return [s.decode() if isinstance(s, bytes) else s for s in a]


def build_manifest() -> pd.DataFrame:
    ds = load_ds("k562")
    with h5py.File(ds.h5_path, "r") as f:
        obs = f["obs"]
        barcode = _dec(obs["cell_barcode"][:]) if not isinstance(obs["cell_barcode"], h5py.Group) else None
        if barcode is None:
            g = obs["cell_barcode"]
            cats = _dec(g["categories"][:]); codes = g["codes"][:]
            barcode = [cats[c] for c in codes]
        gem = obs["gem_group"]
        if isinstance(gem, h5py.Group):
            gcats = _dec(gem["categories"][:]); gem_group = np.array([gcats[c] for c in gem["codes"][:]])
        else:
            gem_group = gem[:]
        umi = obs["UMI_count"][:]
    k562 = ds.cell_of_interest_mask
    codes = ds.perturbation_codes
    cats = ds.perturbation_categories
    ctrl_all = np.where(k562 & np.isin(codes, list(ds.control_category_codes)))[0]

    # catalog cells: exactly as full_12layer_pipeline.py
    rng = np.random.default_rng(SEED_CATALOG)
    cat_idx = np.sort(rng.choice(ctrl_all, size=N_CATALOG, replace=False))

    # ref + pool: disjoint from catalog cells
    rest = np.setdiff1d(ctrl_all, cat_idx)
    rng2 = np.random.default_rng(SEED_CTRL)
    perm = rng2.permutation(rest)
    ref_idx = np.sort(perm[:N_REF])
    pool_idx = np.sort(perm[N_REF:N_REF + N_POOL])

    # TF lists
    tr = pd.read_csv(TRRUST_TSV, sep="\t", header=None, names=["tf", "target", "mode", "pmid"])
    dor = pd.read_csv(DOROTHEA_TSV, sep="\t")
    chip = pd.read_csv(CHIP_TSV, sep="\t")
    tr_tfs = set(tr.tf.str.upper()); dor_tfs = set(dor.source.str.upper()); chip_tfs = set(chip.source.str.upper())
    k562_codes = codes[k562]
    present = pd.Series(k562_codes).value_counts()
    pert = {cats[c].upper(): int(n) for c, n in present.items()
            if c not in ds.control_category_codes}
    tfs = sorted(g for g in pert if g in (tr_tfs | dor_tfs))
    primary = sorted(g for g in tfs if g in chip_tfs)

    rows = []
    for i in cat_idx:
        rows.append((int(i), "catalog", "", 3))
    for i in ref_idx:
        rows.append((int(i), "ref", "", 0))
    for i in pool_idx:
        rows.append((int(i), "pool", "", 1))
    k562_rows = np.where(k562)[0]
    cat_upper = [c.upper() for c in cats]
    for ti, tf in enumerate(tfs):
        code = cat_upper.index(tf)
        pool_tf = k562_rows[codes[k562_rows] == code]
        rng_tf = np.random.default_rng(SEED_KD + ti)
        n_use = min(N_KD_MAX, len(pool_tf))
        pick = np.sort(rng_tf.choice(pool_tf, size=n_use, replace=False))
        prio = 2 if tf in primary else 4
        for i in pick:
            rows.append((int(i), "kd", tf, prio))
    man = pd.DataFrame(rows, columns=["row", "group", "tf", "priority"])
    man["barcode"] = [barcode[i] for i in man.row]
    man["gem_group"] = [gem_group[i] for i in man.row]
    man["UMI_count"] = [float(umi[i]) for i in man.row]
    man["is_primary_tf"] = man.tf.isin(primary)
    man = man.sort_values(["priority", "tf", "row"], kind="stable").reset_index(drop=True)
    meta = {
        "h5_path": str(ds.h5_path),
        "n_k562_nt_cells": int(len(ctrl_all)),
        "seed_catalog": SEED_CATALOG, "n_catalog": N_CATALOG,
        "seed_ctrl": SEED_CTRL, "n_ref": N_REF, "n_pool": N_POOL,
        "seed_kd": SEED_KD, "n_kd_max": N_KD_MAX,
        "tfs_all": tfs, "tfs_primary": primary,
        "k562_cells_per_tf": {t: pert[t] for t in tfs},
        "priority_order": "0 ref, 1 pool, 2 primary kd, 3 catalog, 4 secondary kd",
    }
    return man, meta


class _Stop(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=7.0)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--max-priority", type=int, default=4)
    ap.add_argument("--run-floor-gb", type=float, default=1.5)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    CELLDIR.mkdir(parents=True, exist_ok=True)

    if not MANIFEST.exists():
        man, meta = build_manifest()
        man.to_csv(MANIFEST, index=False)
        H.write_json(OUT / "cell_manifest_meta.json", meta)
        print(f"manifest: {len(man)} cells; groups {man.group.value_counts().to_dict()}", flush=True)
    man = pd.read_csv(MANIFEST, keep_default_na=False)
    prog = json.loads(PROGRESS.read_text()) if PROGRESS.exists() else {"done": [], "failed": [], "chunks": []}
    done = set(prog["done"]) | set(r for r, _ in prog["failed"])
    todo = man[(~man.row.isin(done)) & (man.priority <= args.max_priority)]
    # a cell can appear in two groups only if a KD cell were also NT (impossible); rows are unique
    assert man.row.is_unique, "manifest rows must be unique"
    print(f"todo {len(todo)} of {len(man)} cells (max priority {args.max_priority})", flush=True)
    if len(todo) == 0:
        print("nothing to do"); return

    gb = H.check_memory(3.0)
    print(f"available memory before model load: {gb:.2f} GB", flush=True)

    ds = load_ds("k562")
    with open(SYM2ENS_PKL, "rb") as fh:
        sym2ens = pickle.load(fh)
    tok = MaxTokiTokenizer()
    vi, vt, vm = tok.make_var_mapping([sym2ens.get(s) for s in ds.var_symbols])

    xt = H.load_model(args.device)
    model = xt.model
    sae = H.load_saes([LAYER], device=args.device)[LAYER]
    site = H.site_module(model, LAYER)
    store = {}

    def pre_hook(mod, a):
        store["x"] = a[0].detach()
        raise _Stop()

    handle = site.register_forward_pre_hook(pre_hook)
    bos_path = CELLDIR / "z_bos.npy"

    buf = {"row": [], "n_tokens": [], "mean_gene": [], "nact_gene": [], "z_eos": []}
    cat_buf = {"row": [], "offsets": [0], "tok": [], "idx": [], "val": []}
    n_new = 0
    t_cells = []

    def flush(tag):
        nonlocal buf, cat_buf
        stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time()*1000)%1000:03d}"
        if buf["row"]:
            np.savez(CELLDIR / f"summ_{stamp}_{tag}.npz",
                     row=np.array(buf["row"], np.int64), n_tokens=np.array(buf["n_tokens"], np.int32),
                     mean_gene=np.stack(buf["mean_gene"]).astype(np.float32),
                     nact_gene=np.stack(buf["nact_gene"]).astype(np.uint16),
                     z_eos=np.stack(buf["z_eos"]).astype(np.float16))
        if cat_buf["row"]:
            np.savez(CELLDIR / f"catalog_{stamp}_{tag}.npz",
                     row=np.array(cat_buf["row"], np.int64), offsets=np.array(cat_buf["offsets"], np.int64),
                     tok=np.concatenate(cat_buf["tok"]).astype(np.int32),
                     idx=np.concatenate(cat_buf["idx"]).astype(np.int16),
                     val=np.concatenate(cat_buf["val"]).astype(np.float32))
        prog["done"].extend(buf["row"])
        H.write_json(PROGRESS, prog)
        buf = {"row": [], "n_tokens": [], "mean_gene": [], "nact_gene": [], "z_eos": []}
        cat_buf = {"row": [], "offsets": [0], "tok": [], "idx": [], "val": []}

    stop_reason = "finished"
    with h5py.File(ds.h5_path, "r") as f:
        X = f["X"]
        for k, rec in enumerate(todo.itertuples(index=False)):
            if (time.time() - t_start) / 60 > args.max_minutes:
                stop_reason = "max_minutes"; break
            if k % 20 == 0:
                gbk = H.available_memory_gb()
                if gbk == gbk and gbk < args.run_floor_gb:
                    stop_reason = f"memory_guard ({gbk:.2f} GB)"; break
            t0 = time.time()
            x = X[int(rec.row), :].astype(np.float32)
            cell = tok.tokenize_cell(x, vi, vt, vm, max_len=MAX_LEN)
            if cell is None:
                prog["failed"].append([int(rec.row), "tokenize_none"]); continue
            ids = cell.token_ids
            assert ids[0] == tok.BOS and ids[-1] == tok.EOS
            T = len(ids)
            try:
                with torch.no_grad():
                    model(torch.from_numpy(ids[None, :]).to(args.device), use_cache=False)
            except _Stop:
                pass
            with torch.no_grad():
                z = sae.encode(store["x"][0])            # (T, 4928)
                zg = z[1:T - 1]
                mean_gene = zg.mean(0).cpu().numpy()
                nact = (zg > 0).sum(0).cpu().numpy()
                z_eos = z[T - 1].cpu().numpy()
                if not bos_path.exists():
                    np.save(bos_path, z[0].cpu().numpy().astype(np.float32))
                if rec.group == "catalog":
                    v, ix = torch.topk(z, 32, dim=1)
                    cat_buf["row"].append(int(rec.row)); cat_buf["tok"].append(ids.astype(np.int32))
                    cat_buf["idx"].append(ix.cpu().numpy().astype(np.int16).reshape(-1))
                    cat_buf["val"].append(v.cpu().numpy().astype(np.float32).reshape(-1))
                    cat_buf["offsets"].append(cat_buf["offsets"][-1] + T)
            del z, zg, store["x"]
            buf["row"].append(int(rec.row)); buf["n_tokens"].append(T)
            buf["mean_gene"].append(mean_gene); buf["nact_gene"].append(nact); buf["z_eos"].append(z_eos)
            n_new += 1
            t_cells.append(time.time() - t0)
            if n_new % 25 == 0:
                H.free_device_cache()
            if len(buf["row"]) >= args.save_every:
                flush("p")
                print(f"  saved; {n_new} new cells, {np.mean(t_cells):.2f} s/cell, "
                      f"{(time.time()-t_start)/60:.1f} min", flush=True)
    flush("p")
    handle.remove()
    wall = time.time() - t_start
    prog["chunks"].append({"start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t_start)),
                           "wall_s": round(wall, 1), "n_cells": n_new,
                           "s_per_cell": round(float(np.mean(t_cells)), 3) if t_cells else None,
                           "stop_reason": stop_reason})
    H.write_json(PROGRESS, prog)
    cfg = H.env_info(args.device)
    cfg.update({
        "script": __file__, "script_sha256": H.sha256_file(__file__),
        "sae_path": str(sae.path), "sae_sha256": H.sha256_file(sae.path),
        "layer_site": "input of model.model.layers[5] = hidden_states[5] (hooks_v2.site_module)",
        "max_len": MAX_LEN, "seeds": {"catalog": SEED_CATALOG, "ctrl": SEED_CTRL, "kd": SEED_KD},
        "n_ref": N_REF, "n_pool": N_POOL, "n_kd_max": N_KD_MAX,
        "memory_guard": {"before_model_load_min_gb": 3.0, "in_run_floor_gb": args.run_floor_gb},
        "cell_ids": "cell_manifest.csv (dataset row index + barcode) in this folder",
        "dataset_path": str(ds.h5_path),
        "chunks": prog["chunks"],
        "n_done": len(prog["done"]), "n_failed": len(prog["failed"]),
    })
    H.write_json(RUNCFG, cfg)
    remaining = int((~man.row.isin(set(prog["done"]) | set(r for r, _ in prog["failed"]))).sum())
    print(f"chunk done: {n_new} cells in {wall/60:.1f} min ({stop_reason}); remaining {remaining}", flush=True)


if __name__ == "__main__":
    main()
