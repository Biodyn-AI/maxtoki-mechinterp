"""V3-2: per-cell layer-5 SAE feature summaries for the TF-specificity re-run, with the
v3 SAE (outputs/v3_sae/layer_05) and CORRECTLY encoded K562 inputs (setup/inputs_v3.py).

Same design as v2_tf_specificity_extract.py. Two changes only:
  1. input encoding: counts = round(expm1(X) / unit) (inputs_v3.CountReader("k562")), ranked by
     counts / gene median (inputs_v3.tokenize_counts, checked per cell). v2 ranked log1p(CP10k).
  2. SAE: the v3 layer-5 SAE (retrained on correctly encoded inputs), read at the input of
     block 5 (= hidden_states[5], hooks_v2.site_module(model, 5)).

Sub-commands
------------
  prepare  CPU. Builds the cell manifest with the v2 seeds and checks it equals the v2 manifest
           (same 9,200 cells). Old-vs-new order agreement on a sample of knockdown cells, and a
           negative control: the old (v2) tokens must FAIL the encoding check.
  extract  MPS. For every cell: counts -> tokens (checked) -> every batch of 25 cells passes
           inputs_v3.assert_encoding_batch on ALL its cells -> forward pass to the input of block
           5 (batch size 1, eager attention, max_len 2,048; the forward pass stops there) ->
           v3 layer-5 SAE codes -> saves
             mean_gene[f] = mean of z_f over gene tokens (<bos>, <eos> excluded)  (unit of inference)
             nact_gene[f] = number of gene tokens with z_f > 0
             z_eos[f]     = code at <eos>; the <bos> code is the same for every cell (checked)
             tokens       = the token ids fed to the model
           For the 500 catalog cells it also saves the top-32 codes at every position.
           Resumable; --max-minutes bounds one call. Memory: needs >= 3 GB available (vm_stat
           free + inactive) before the model loads and every 20 cells, else it stops cleanly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)

PROJ = Path("<REPO_ROOT>/projects/maxtoki")
sys.path.insert(0, str(PROJ / "setup"))
import hooks_v2 as H  # noqa: E402
import inputs_v3 as iv3  # noqa: E402
from dataset_loader import resolve as load_ds  # noqa: E402

RUN = PROJ / "runs/sae-atlas-217M"
OUT = RUN / "outputs/v3_tf_specificity"
CELLDIR = OUT / "cells"
MANIFEST = OUT / "cell_manifest.csv"
PROGRESS = OUT / "extract_progress.json"
RUNCFG = OUT / "run_config_extract.json"
V2OUT = RUN / "outputs/v2_tf_specificity"
SAE_DIR_V3 = RUN / "outputs/v3_sae"

TRRUST_TSV = Path("<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv")
DOROTHEA_TSV = Path("<DATA_ROOT>/biodyn-work/single_cell_mechinterp/external/networks/dorothea_human.tsv")
CHIP_TSV = Path("<DATA_ROOT>/biodyn-work/single_cell_mechinterp/external/networks/dorothea_chipseq_human.tsv")

LAYER = 5
MAX_LEN = 2048
SEED_CATALOG = 42          # = full_12layer_pipeline.py / v3_sae.py (N_CTRL = 500)
N_CATALOG = 500
SEED_CTRL = 20261001       # ref + pool draw (same as v2)
N_REF = 400
N_POOL = 800
SEED_KD = 20261002         # per-TF seed = SEED_KD + index of TF in sorted list (same as v2)
N_KD_MAX = 100
CHECK_BATCH = 25
MIN_GB = 3.0


def _dec(a):
    return [s.decode() if isinstance(s, bytes) else s for s in a]


def build_manifest():
    """Identical to v2_tf_specificity_extract.build_manifest (same seeds, same rules)."""
    ds = load_ds("k562")
    with h5py.File(ds.h5_path, "r") as f:
        obs = f["obs"]
        if isinstance(obs["cell_barcode"], h5py.Group):
            g = obs["cell_barcode"]
            cats = _dec(g["categories"][:]); codes_ = g["codes"][:]
            barcode = [cats[c] for c in codes_]
        else:
            barcode = _dec(obs["cell_barcode"][:])
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
    rng = np.random.default_rng(SEED_CATALOG)
    cat_idx = np.sort(rng.choice(ctrl_all, size=N_CATALOG, replace=False))
    rest = np.setdiff1d(ctrl_all, cat_idx)
    rng2 = np.random.default_rng(SEED_CTRL)
    perm = rng2.permutation(rest)
    ref_idx = np.sort(perm[:N_REF])
    pool_idx = np.sort(perm[N_REF:N_REF + N_POOL])
    tr = pd.read_csv(TRRUST_TSV, sep="\t", header=None, names=["tf", "target", "mode", "pmid"])
    dor = pd.read_csv(DOROTHEA_TSV, sep="\t")
    chip = pd.read_csv(CHIP_TSV, sep="\t")
    tr_tfs = set(tr.tf.str.upper()); dor_tfs = set(dor.source.str.upper()); chip_tfs = set(chip.source.str.upper())
    present = pd.Series(codes[k562]).value_counts()
    pert = {cats[c].upper(): int(n) for c, n in present.items() if c not in ds.control_category_codes}
    tfs = sorted(g for g in pert if g in (tr_tfs | dor_tfs))
    primary = sorted(g for g in tfs if g in chip_tfs)
    rows = [(int(i), "catalog", "", 3) for i in cat_idx]
    rows += [(int(i), "ref", "", 0) for i in ref_idx]
    rows += [(int(i), "pool", "", 1) for i in pool_idx]
    k562_rows = np.where(k562)[0]
    cat_upper = [c.upper() for c in cats]
    for ti, tf in enumerate(tfs):
        code = cat_upper.index(tf)
        pool_tf = k562_rows[codes[k562_rows] == code]
        rng_tf = np.random.default_rng(SEED_KD + ti)
        pick = np.sort(rng_tf.choice(pool_tf, size=min(N_KD_MAX, len(pool_tf)), replace=False))
        prio = 2 if tf in primary else 4
        rows += [(int(i), "kd", tf, prio) for i in pick]
    man = pd.DataFrame(rows, columns=["row", "group", "tf", "priority"])
    man["barcode"] = [barcode[i] for i in man.row]
    man["gem_group"] = [gem_group[i] for i in man.row]
    man["UMI_count"] = [float(umi[i]) for i in man.row]
    man["is_primary_tf"] = man.tf.isin(primary)
    man = man.sort_values(["priority", "tf", "row"], kind="stable").reset_index(drop=True)
    meta = {"h5_path": str(ds.h5_path), "n_k562_nt_cells": int(len(ctrl_all)),
            "seed_catalog": SEED_CATALOG, "n_catalog": N_CATALOG, "seed_ctrl": SEED_CTRL, "n_ref": N_REF,
            "n_pool": N_POOL, "seed_kd": SEED_KD, "n_kd_max": N_KD_MAX, "tfs_all": tfs, "tfs_primary": primary,
            "k562_cells_per_tf": {t: pert[t] for t in tfs},
            "priority_order": "0 ref, 1 pool, 2 primary kd, 3 catalog, 4 secondary kd"}
    return man, meta


# =============================================================================
def cmd_prepare(args):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    man, meta = build_manifest()
    v2 = pd.read_csv(V2OUT / "cell_manifest.csv", keep_default_na=False)
    v2meta = json.load(open(V2OUT / "cell_manifest_meta.json"))
    same = {
        "n_cells_v3": int(len(man)), "n_cells_v2": int(len(v2)),
        "same_rows_groups_tfs_in_order": bool(len(man) == len(v2) and
                                              (man.row.values == v2.row.values).all() and
                                              (man.group.values == v2.group.values).all() and
                                              (man.tf.values == v2.tf.values).all()),
        "same_barcodes": bool(len(man) == len(v2) and (man.barcode.values == v2.barcode.values).all()),
        "same_tf_lists": bool(meta["tfs_all"] == v2meta["tfs_all"] and meta["tfs_primary"] == v2meta["tfs_primary"]),
    }
    if not all(v for k, v in same.items() if k.startswith("same")):
        raise SystemExit(f"manifest differs from v2: {same}")
    # catalog rows must equal the v3 SAE training cells
    v3c = np.load(SAE_DIR_V3 / "cells.npz")
    same["catalog_rows_equal_v3_sae_cells"] = bool(np.array_equal(np.sort(man[man.group == "catalog"].row.values),
                                                                  v3c["rows"]))
    if not same["catalog_rows_equal_v3_sae_cells"]:
        raise SystemExit("catalog rows differ from outputs/v3_sae/cells.npz")
    man.to_csv(MANIFEST, index=False)
    H.write_json(OUT / "cell_manifest_meta.json", meta)

    # old-vs-new order agreement + negative control on a sample (25 ref, 25 GATA1, 50 random kd cells)
    vm = iv3.get_var_map("k562")
    tok = iv3.get_tokenizer()
    rs = np.random.default_rng(SEED_CTRL + 77)
    samp = list(man[man.group == "ref"].row.values[:25]) + list(man[(man.tf == "GATA1")].row.values[:25])
    kd_other = man[(man.group == "kd") & (man.tf != "GATA1")].row.values
    samp += list(rs.choice(kd_other, 50, replace=False))
    agree, old_fail, new_pass = [], 0, 0
    with iv3.CountReader("k562") as r:
        for row in samp:
            c = r.counts(int(row))
            new = iv3.tokenize_counts(c, vm, MAX_LEN, check=True)
            new_pass += 1
            x_old = r.stored_X(int(row)).astype(np.float32)
            old = tok.tokenize_cell(x_old, vm.var_indices, vm.token_ids, vm.medians, max_len=MAX_LEN)
            old_fail += int(not iv3.check_encoding(c, old, vm, MAX_LEN)["pass"])
            o_old = tok.tokenize_cell(x_old, vm.var_indices, vm.token_ids, vm.medians, max_len=10 ** 9).token_ids[1:-1]
            agree.append(iv3.order_agreement(o_old, iv3.full_order(c, vm), MAX_LEN))
    ag = pd.DataFrame(agree).astype(float)
    prep = {"manifest_vs_v2": same, "n_cells": int(len(man)),
            "groups": man.group.value_counts().to_dict(),
            "order_sample": {"n_cells": len(samp), "rows": [int(x) for x in samp],
                             "new_tokens_pass_check": new_pass, "old_v2_tokens_fail_check": old_fail,
                             "old_vs_new_agreement_mean": ag.mean(numeric_only=True).round(4).to_dict(),
                             "old_vs_new_agreement_min": ag.min(numeric_only=True).round(4).to_dict()},
            "wall_s": round(time.time() - t0, 1)}
    H.write_json(OUT / "prepare.json", prep)
    print(json.dumps(prep["manifest_vs_v2"]), flush=True)
    print(json.dumps(prep["order_sample"]["old_vs_new_agreement_mean"]), "old fail", old_fail, "of", len(samp), flush=True)


class _Stop(Exception):
    pass


def cmd_extract(args):
    t_start = time.time()
    CELLDIR.mkdir(parents=True, exist_ok=True)
    if not MANIFEST.exists():
        raise SystemExit("run `prepare` first")
    man = pd.read_csv(MANIFEST, keep_default_na=False)
    assert man.row.is_unique
    prog = json.loads(PROGRESS.read_text()) if PROGRESS.exists() else {
        "done": [], "failed": [], "chunks": [], "enc_batches": 0, "enc_cells_checked": 0,
        "enc_tie_reordered": 0, "enc_max_order_violations": 0, "bos_max_abs_diff": 0.0}
    done = set(prog["done"]) | set(r for r, _ in prog["failed"])
    todo = man[(~man.row.isin(done)) & (man.priority <= args.max_priority)]
    print(f"todo {len(todo)} of {len(man)} cells", flush=True)
    if len(todo) == 0:
        print("nothing to do; DONE"); return
    gb = H.check_memory(MIN_GB)
    print(f"available memory before model load: {gb:.2f} GB", flush=True)

    vm = iv3.get_var_map("k562")
    xt = H.load_model(args.device)
    model = xt.model
    sae = H.load_saes([LAYER], device=args.device, sae_dir=SAE_DIR_V3)[LAYER]
    site = H.site_module(model, LAYER)
    store = {}

    def pre_hook(mod, a):
        store["x"] = a[0].detach()
        raise _Stop()

    handle = site.register_forward_pre_hook(pre_hook)
    bos_path = CELLDIR / "z_bos.npy"
    z_bos_ref = np.load(bos_path) if bos_path.exists() else None

    def new_buf():
        return {"row": [], "n_tokens": [], "mean_gene": [], "nact_gene": [], "z_eos": [], "tok": [],
                "counts_sha": [], "tok_sha": []}

    buf = new_buf()
    cat_buf = {"row": [], "offsets": [0], "tok": [], "idx": [], "val": []}
    n_new = 0
    t_cells = []

    def flush(tag):
        nonlocal buf, cat_buf
        stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
        if buf["row"]:
            lens = np.array(buf["n_tokens"], np.int64)
            np.savez(CELLDIR / f"summ_{stamp}_{tag}.npz",
                     row=np.array(buf["row"], np.int64), n_tokens=lens.astype(np.int32),
                     mean_gene=np.stack(buf["mean_gene"]).astype(np.float32),
                     nact_gene=np.stack(buf["nact_gene"]).astype(np.uint16),
                     z_eos=np.stack(buf["z_eos"]).astype(np.float32),
                     tok_offsets=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
                     tokens=np.concatenate(buf["tok"]).astype(np.int32),
                     counts_sha256=np.array(buf["counts_sha"]), tokens_sha256=np.array(buf["tok_sha"]))
        if cat_buf["row"]:
            np.savez(CELLDIR / f"catalog_{stamp}_{tag}.npz",
                     row=np.array(cat_buf["row"], np.int64), offsets=np.array(cat_buf["offsets"], np.int64),
                     tok=np.concatenate(cat_buf["tok"]).astype(np.int32),
                     idx=np.concatenate(cat_buf["idx"]).astype(np.int16),
                     val=np.concatenate(cat_buf["val"]).astype(np.float32))
        prog["done"].extend(buf["row"])
        H.write_json(PROGRESS, prog)
        buf = new_buf()
        cat_buf = {"row": [], "offsets": [0], "tok": [], "idx": [], "val": []}

    stop_reason = "finished"
    recs = list(todo.itertuples(index=False))
    with iv3.CountReader("k562") as reader:
        for b0 in range(0, len(recs), CHECK_BATCH):
            if (time.time() - t_start) / 60 > args.max_minutes:
                stop_reason = "max_minutes"; break
            gbk = H.available_memory_gb()
            if gbk == gbk and gbk < MIN_GB:
                stop_reason = f"memory_guard ({gbk:.2f} GB)"; break
            batch = recs[b0:b0 + CHECK_BATCH]
            # ---- counts -> tokens (each checked) -> batch check BEFORE any forward pass ----
            cnts, cells, keep = [], [], []
            for rec in batch:
                c, cchk = iv3.counts_accept_unit_one(reader, int(rec.row), return_check=True)
                if cchk.get("accepted_as"):
                    prog.setdefault("rows_accepted_unit_one", [])
                    if int(rec.row) not in prog["rows_accepted_unit_one"]:
                        prog["rows_accepted_unit_one"].append(int(rec.row))
                cell = iv3.tokenize_counts(c, vm, MAX_LEN, check=True)
                if cell is None:
                    prog["failed"].append([int(rec.row), "tokenize_none"]); continue
                cnts.append(c); cells.append(cell); keep.append(rec)
            if not cells:
                continue
            enc = iv3.assert_encoding_batch(cnts, cells, vm, MAX_LEN, n_first=None,
                                            label=f"batch@{b0}")   # raises EncodingError on failure
            prog["enc_batches"] += 1
            prog["enc_cells_checked"] += enc["n_checked"]
            prog["enc_tie_reordered"] += enc["n_positions_tie_reordered"]
            prog["enc_max_order_violations"] = max(prog["enc_max_order_violations"], enc["max_order_violations"])
            for rec, c, cell in zip(keep, cnts, cells):
                t0 = time.time()
                ids = cell.token_ids
                assert ids[0] == iv3.BOS and ids[-1] == iv3.EOS
                T = len(ids)
                try:
                    with torch.no_grad():
                        model(torch.from_numpy(ids[None, :].astype(np.int64)).to(args.device), use_cache=False)
                except _Stop:
                    pass
                with torch.no_grad():
                    z = sae.encode(store["x"][0])             # (T, 4928)
                    zg = z[1:T - 1]
                    mean_gene = zg.mean(0).cpu().numpy()
                    nact = (zg > 0).sum(0).cpu().numpy()
                    z_eos = z[T - 1].cpu().numpy()
                    zb = z[0].cpu().numpy().astype(np.float32)
                    if z_bos_ref is None:
                        np.save(bos_path, zb); z_bos_ref = zb
                    else:
                        prog["bos_max_abs_diff"] = max(prog["bos_max_abs_diff"], float(np.abs(zb - z_bos_ref).max()))
                    if rec.group == "catalog":
                        v, ix = torch.topk(z, 32, dim=1)
                        cat_buf["row"].append(int(rec.row)); cat_buf["tok"].append(ids.astype(np.int32))
                        cat_buf["idx"].append(ix.cpu().numpy().astype(np.int16).reshape(-1))
                        cat_buf["val"].append(v.cpu().numpy().astype(np.float32).reshape(-1))
                        cat_buf["offsets"].append(cat_buf["offsets"][-1] + T)
                del z, zg
                store.pop("x", None)
                buf["row"].append(int(rec.row)); buf["n_tokens"].append(T)
                buf["mean_gene"].append(mean_gene); buf["nact_gene"].append(nact); buf["z_eos"].append(z_eos)
                buf["tok"].append(ids.astype(np.int32))
                buf["counts_sha"].append(iv3.array_sha256(c)[:16]); buf["tok_sha"].append(iv3.array_sha256(ids.astype(np.int32))[:16])
                n_new += 1
                t_cells.append(time.time() - t0)
            if n_new % 100 == 0:
                H.free_device_cache()
            if len(buf["row"]) >= args.save_every:
                flush("p")
                print(f"  saved; {n_new} new cells, {np.mean(t_cells):.3f} s/cell, "
                      f"{(time.time() - t_start) / 60:.1f} min, mem {H.available_memory_gb():.1f} GB", flush=True)
    flush("p")
    handle.remove()
    wall = time.time() - t_start
    prog["chunks"].append({"start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t_start)),
                           "wall_s": round(wall, 1), "n_cells": n_new,
                           "s_per_cell": round(float(np.mean(t_cells)), 3) if t_cells else None,
                           "mem_gb_at_start": round(gb, 2), "stop_reason": stop_reason})
    H.write_json(PROGRESS, prog)
    cfg = H.env_info(args.device)
    cfg.update({
        "script": __file__, "script_sha256": H.sha256_file(__file__),
        "sae_path": str(sae.path), "sae_sha256": H.sha256_file(sae.path),
        "layer_site": "input of model.model.layers[5] = hidden_states[5] (hooks_v2.site_module)",
        "max_len": MAX_LEN, "batch_size_forward": 1, "attention": "eager",
        "seeds": {"catalog": SEED_CATALOG, "ctrl": SEED_CTRL, "kd": SEED_KD},
        "n_ref": N_REF, "n_pool": N_POOL, "n_kd_max": N_KD_MAX,
        "memory_guard_gb": MIN_GB, "threads": {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                                                "torch": torch.get_num_threads()},
        "input_encoding": iv3.encoding_record("k562"),
        "encoding_check_result": {"batches": prog["enc_batches"], "cells_checked": prog["enc_cells_checked"],
                                  "cells_done": len(prog["done"]), "all_pass": True,
                                  "positions_tie_reordered": prog["enc_tie_reordered"],
                                  "max_order_violations": prog["enc_max_order_violations"],
                                  "rule": "every cell of every 25-cell batch passes inputs_v3.assert_encoding_batch "
                                          "before its forward pass; a failure raises EncodingError and aborts"},
        "bos_code_max_abs_diff_across_cells": prog["bos_max_abs_diff"],
        "rows_accepted_unit_one": prog.get("rows_accepted_unit_one", []),
        "unit_one_note": ("expm1(X) of these rows is already whole numbers (smallest value 1); "
                          "inputs_v3.CountReader.counts rejects them, inputs_v3.counts_accept_unit_one "
                          "(added by this item) accepts them as counts with factor 1"),
        "cell_ids": "cell_manifest.csv (dataset row index + barcode) in this folder",
        "per_cell_input_sha256": "cells/summ_*.npz: counts_sha256, tokens_sha256 (first 16 hex chars) per cell",
        "chunks": prog["chunks"], "n_done": len(prog["done"]), "n_failed": len(prog["failed"]),
    })
    H.write_json(RUNCFG, cfg)
    remaining = int((~man.row.isin(set(prog["done"]) | set(r for r, _ in prog["failed"]))).sum())
    print(f"chunk done: {n_new} cells in {wall / 60:.1f} min ({stop_reason}); remaining {remaining}"
          + ("; DONE" if remaining == 0 else ""), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "extract"])
    ap.add_argument("--max-minutes", type=float, default=7.0)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--max-priority", type=int, default=4)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    {"prepare": cmd_prepare, "extract": cmd_extract}[args.cmd](args)


if __name__ == "__main__":
    main()
