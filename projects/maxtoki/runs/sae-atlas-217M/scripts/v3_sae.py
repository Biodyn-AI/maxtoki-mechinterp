"""v3_sae -- retrain the 12 MaxToki-217M TopK SAEs on CORRECTLY encoded K562 inputs.

Why
---
checks/INPUT_ENCODING_AUDIT.md: the deployed SAEs (outputs/phase1/layer_XX) were trained on
hidden states of 500 K562 control cells whose tokens were ranked from log1p(CP10k) / median
instead of counts / median. This script repeats the deployed protocol exactly
(scripts/full_12layer_pipeline.py Phase 0 + Phase 1, setup/topk_sae.py:train_sae) with ONE change:
the input encoding comes from setup/inputs_v3.py (counts = round(expm1(X) / unit), checked).

Deployed protocol (reproduced)
------------------------------
  cells   : dataset_loader.resolve("k562"); K562 non-targeting cells; rng = default_rng(42);
            rows = sort(rng.choice(ctrl_all, 500, replace=False)); every cell that tokenizes (500).
  tokens  : max_len 2048 (<bos> + 2,046 genes + <eos>); batch size 1 (no padding).
  sites   : hidden_states[l], l = 0..11, every position incl. <bos>/<eos>; float32; MPS.
  split   : 1,019,996 positions < n_train + n_eval = 1.1M, so train_sae's else-branch:
            train = first int(0.9 N) positions (in cell order), eval = the rest (held out).
  SAE     : TopKSAE(d_model 1232, d_sae 4928, k 32); Adam lr 3e-4; batch 4096; 4 epochs;
            DataLoader(shuffle=True); torch.manual_seed(42); decoder columns renormalised each step.
Deliberate differences (none change the maths):
  * the train rows are read once into one float32 array (4.5 GB); the deployed code made two
    extra copies (peak ~14 GB; this machine is shared). Batches are gathered from that array by
    index. The DataLoader is the same object type over an index dataset, so the shuffle order uses
    the global torch RNG exactly as before. Under memory pressure from other jobs, layers 4-11
    were instead trained from a byte-checked (sha256) copy of the train rows on the internal SSD
    (--cache-dir), read row by row with pread (no page cache, no large anonymous array). Layer 0 epoch 0 was
    gathered from a memory map of the X6 file. Same rows and order in every case.
  * mu = train mean computed in float64 and cast to float32 (deployed: float32 torch mean).
  * the held-out evaluation runs in chunks of 8,192 rows instead of one 102,000-row batch.
  * training checkpoints each epoch (model, optimiser and torch RNG state) so it can resume.

Sub-commands (each is resumable; run repeatedly until it says DONE)
-------------------------------------------------------------------
  prepare            CPU: cells, v3 + deployed tokens, encoding checks, deployed-token reproduction.
  extract            MPS: hidden_states[0..11] for the v3 tokens -> activations/layer_XX.npy.
  train              MPS: one SAE per layer -> layer_XX/sae_final.pt (+ results.json, training_log.json).
  evaluate           MPS: v3 and deployed SAEs on the held-out v3 tokens (+ all-token dead counts).
  hooks_check        MPS: hooks_v2.load_sae on the v3 folder; live captures vs stored activations.
  summarize          CPU: bootstrap CIs and tables -> eval/summary.json, eval/table.csv.
  cleanup            delete activations/layer_XX.npy if their total exceeds 20 GB (recorded).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch

torch.set_num_threads(4)

PROJ = Path(__file__).resolve().parents[3]
RUN = PROJ / "runs/sae-atlas-217M"
OUT = RUN / "outputs/v3_sae"
ACT_DIR = OUT / "activations"
EVAL_ACT_DIR = OUT / "activations_eval"
CODES_DIR = OUT / "codes"
EVAL_DIR = OUT / "eval"
LOG_DIR = OUT / "logs"
DEPLOYED_SAE_DIR = RUN / "outputs/phase1"
DEPLOYED_GENE_NAMES = RUN / "outputs/phase0/layer_00/gene_names.json"
sys.path.insert(0, str(PROJ / "setup"))

import inputs_v3 as iv3  # noqa: E402
import hooks_v2 as hv2  # noqa: E402
from topk_sae import TopKSAE  # noqa: E402

# ---- deployed constants (full_12layer_pipeline.py:44-49, :170-176; topk_sae.train_sae) ----
SEED = 42
N_CTRL = 500
MAX_LEN = 2048
N_LAYERS = 12
HIDDEN = 1232
D_SAE = 4 * HIDDEN  # 4928
K = 32
LR = 3e-4
BATCH = 4096
EPOCHS = 4
N_TRAIN = 1_000_000
N_EVAL = 100_000
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MIN_FREE_GB = 3.0
BOOT_SEED = 20261001
N_BOOT = 2000
DELETE_THRESHOLD_GB = 20.0
CACHE_DIR = None  # set by --cache-dir (train only): a fast internal-SSD copy of one layer's train rows


# =============================================================================
# helpers
# =============================================================================
def log(msg: str):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def mem_gb() -> float:
    return hv2.available_memory_gb()


def guard(where: str):
    gb = mem_gb()
    if gb == gb and gb < MIN_FREE_GB:
        log(f"STOP: available memory {gb:.2f} GB < {MIN_FREE_GB} GB at {where}; nothing half-written kept")
        raise hv2.MemoryGuardError(f"{gb:.2f} GB at {where}")
    return gb


def read_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text())


def update_run_config(section: str, obj: dict):
    path = OUT / "run_config.json"
    cfg = read_json(path, {}) or {}
    cfg.setdefault("pipeline", "sae-atlas-217M / v3_sae")
    cfg.setdefault("item", "V3-1: retrain the 12 layer SAEs on correctly encoded inputs")
    cfg[section] = obj
    cfg["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    hv2.write_json(path, cfg)


def code_hashes() -> dict:
    files = {
        "v3_sae.py": Path(__file__).resolve(),
        "inputs_v3.py": PROJ / "setup/inputs_v3.py",
        "hooks_v2.py": PROJ / "setup/hooks_v2.py",
        "topk_sae.py": PROJ / "setup/topk_sae.py",
        "maxtoki_adapter.py": PROJ / "setup/maxtoki_adapter.py",
        "dataset_loader.py": PROJ / "setup/dataset_loader.py",
        "deployed full_12layer_pipeline.py": RUN / "scripts/full_12layer_pipeline.py",
    }
    return {k: iv3.sha256_file(v) for k, v in files.items()}


def append_chunk_record(step: str, rec: dict):
    path = OUT / "chunks.json"
    d = read_json(path, {}) or {}
    d.setdefault(step, []).append(rec)
    hv2.write_json(path, d)


def load_cells():
    z = np.load(OUT / "cells.npz")
    return {k: z[k] for k in z.files}


def layout():
    lay = read_json(OUT / "layout.json")
    if lay is None:
        raise SystemExit("run `prepare` first")
    return lay


def npy_header_len(path) -> tuple[int, tuple]:
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, version)
        return f.tell(), shape


def act_path(layer: int) -> Path:
    return ACT_DIR / f"layer_{layer:02d}.npy"


def open_acts(layer: int):
    return np.load(act_path(layer), mmap_mode="r")


def cell_of_position(offsets: np.ndarray, pos: np.ndarray) -> np.ndarray:
    return np.searchsorted(offsets, pos, side="right") - 1


# =============================================================================
# prepare (CPU)
# =============================================================================
def cmd_prepare(args):
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    from dataset_loader import resolve as load_ds
    import h5py

    # --- same cell selection as full_12layer_pipeline.py:59-62 ---
    ds = load_ds("k562")
    rng = np.random.default_rng(SEED)
    ctrl_all = np.where(ds.cell_of_interest_mask &
                        np.isin(ds.perturbation_codes, list(ds.control_category_codes)))[0]
    ctrl_idx = np.sort(rng.choice(ctrl_all, size=N_CTRL, replace=False))
    log(f"selected {len(ctrl_idx)} K562 control rows from {len(ctrl_all)} (rows {ctrl_idx[:3]}...)")

    vm = iv3.get_var_map("k562")
    tok = iv3.get_tokenizer()
    with h5py.File(ds.h5_path, "r") as f:
        var_symbols = [s.decode() if isinstance(s, bytes) else s for s in f["var"]["gene_name_index"][:]]
    token_to_gene = {int(t): var_symbols[int(v)].upper() for v, t in zip(vm.var_indices, vm.token_ids)}

    # --- v3 tokens (checked) and deployed tokens (old path, for comparison only) ---
    cells_v3, cells_dep, counts_all, row_checks, agree = [], [], [], [], []
    with iv3.CountReader("k562") as r:
        for row in ctrl_idx:
            c, chk = r.counts(int(row), return_check=True)
            cell = iv3.tokenize_counts(c, vm, MAX_LEN, check=True)
            x_old = r.stored_X(int(row)).astype(np.float32)
            dep = tok.tokenize_cell(x_old, vm.var_indices, vm.token_ids, vm.medians, max_len=MAX_LEN)
            if cell is None or dep is None:
                raise SystemExit(f"row {row} does not tokenize (deployed kept all 500)")
            cells_v3.append(cell.token_ids.astype(np.int32))
            cells_dep.append(dep.token_ids.astype(np.int32))
            counts_all.append(c)
            row_checks.append(chk)
            # order agreement on full (uncut) orders, audit measures
            o_new = iv3.full_order(c, vm)
            o_old = tok.tokenize_cell(x_old, vm.var_indices, vm.token_ids, vm.medians,
                                      max_len=10 ** 9).token_ids[1:-1]
            agree.append(iv3.order_agreement(o_old, o_new, MAX_LEN))
    counts_all = np.stack(counts_all)

    # --- encoding check, explicit, on every cell ---
    enc = iv3.assert_encoding_batch(counts_all, cells_v3, vm, MAX_LEN, n_first=None, label="v3 500 cells")
    # the old tokens must FAIL the check (shows the check has power on these cells)
    n_old_fail = sum(not iv3.check_encoding(counts_all[i], cells_dep[i], vm, MAX_LEN)["pass"]
                     for i in range(len(cells_dep)))

    lens_v3 = np.array([len(t) for t in cells_v3])
    lens_dep = np.array([len(t) for t in cells_dep])
    if not np.array_equal(lens_v3, lens_dep):
        raise SystemExit("token lengths differ between encodings (should be identical)")
    offsets = np.concatenate([[0], np.cumsum(lens_v3)]).astype(np.int64)
    n_total = int(offsets[-1])
    tokens_v3 = np.concatenate(cells_v3)
    tokens_dep = np.concatenate(cells_dep)

    # --- the deployed gene_names.json must equal the old-path tokens of these rows ---
    dep_names = json.loads(DEPLOYED_GENE_NAMES.read_text())
    my_dep_names = [token_to_gene.get(int(t), "<special>") for t in tokens_dep]
    dep_repro = {
        "file": str(DEPLOYED_GENE_NAMES),
        "n_positions_deployed": len(dep_names),
        "n_positions_rebuilt": len(my_dep_names),
        "identical": dep_names == my_dep_names,
        "n_mismatch": int(sum(a != b for a, b in zip(dep_names, my_dep_names))) if len(dep_names) == len(my_dep_names) else None,
    }
    if not dep_repro["identical"]:
        raise SystemExit(f"deployed token reproduction failed: {dep_repro}")
    del dep_names
    names_v3 = [token_to_gene.get(int(t), "<special>") for t in tokens_v3]
    (OUT / "gene_names_v3.json").write_text(json.dumps(names_v3))

    # --- split, exactly as topk_sae.train_sae ---
    if n_total > N_TRAIN + N_EVAL:
        raise SystemExit("n_total > 1.1M: train_sae would use the random-permutation branch; not handled")
    split = int(0.9 * n_total)
    c_first_eval = int(cell_of_position(offsets, np.array([split]))[0])
    eval_cells = list(range(c_first_eval, N_CTRL))
    same_pos = np.mean([np.mean(a[1:-1] == b[1:-1]) for a, b in zip(cells_v3, cells_dep)])

    np.savez(OUT / "cells.npz", rows=ctrl_idx.astype(np.int64), offsets=offsets,
             tokens_v3=tokens_v3, tokens_deployed=tokens_dep)
    lay = {"n_cells": N_CTRL, "n_total": n_total, "split": split, "n_train": split,
           "n_eval": n_total - split, "first_eval_cell": c_first_eval,
           "first_eval_cell_positions_in_eval": int(offsets[c_first_eval + 1] - split),
           "first_eval_cell_length": int(lens_v3[c_first_eval]),
           "n_eval_cells": len(eval_cells), "n_fully_held_out_cells": len(eval_cells) - 1,
           "mean_tokens_per_cell": float(lens_v3.mean()),
           "n_cells_cut_at_2046_genes": int(np.sum(lens_v3 == MAX_LEN))}
    hv2.write_json(OUT / "layout.json", lay)

    def summ(key):
        v = np.array([a[key] for a in agree], dtype=float)
        return {"mean": float(np.nanmean(v)), "min": float(np.nanmin(v)), "max": float(np.nanmax(v))}
    agree_summary = {k: summ(k) for k in ("spearman_full", "spearman_fed", "top200_overlap",
                                          "top2046_overlap", "kept_set_overlap", "same_position_share")}
    prep = {
        "cells": {"rule": "dataset_loader.resolve('k562'); K562 non-targeting; default_rng(42).choice(500) sorted "
                          "(full_12layer_pipeline.py:59-62)",
                  "n_control_pool": int(len(ctrl_all)), "rows": [int(r) for r in ctrl_idx]},
        "layout": lay,
        "encoding_check": enc,
        "count_checks": {"n": len(row_checks), "all_expected_ok": all(c["expected_ok"] for c in row_checks),
                         "kinds": sorted({c["kind"] for c in row_checks}),
                         "max_dev": float(max(c["max_dev"] for c in row_checks))},
        "old_tokens_fail_check": f"{n_old_fail} of {len(cells_dep)}",
        "deployed_token_reproduction": dep_repro,
        "order_agreement_old_vs_v3_500_cells": agree_summary,
        "same_position_share_cut_sequences": float(same_pos),
        "counts_sha256": iv3.array_sha256(counts_all),
        "tokens_v3_sha256": iv3.array_sha256(tokens_v3),
        "tokens_deployed_sha256": iv3.array_sha256(tokens_dep),
        "seconds": time.time() - t0,
    }
    hv2.write_json(OUT / "prepare.json", prep)
    update_run_config("prepare", {
        "env": hv2.env_info("cpu"),
        "encoding": iv3.encoding_record("k562", rows=ctrl_idx, check_summary=enc,
                                        counts_sha256=prep["counts_sha256"]),
        "seeds": {"cell_sampling": SEED, "sae_init_and_shuffle": SEED, "bootstrap": BOOT_SEED},
        "code_sha256": code_hashes(),
        "summary": {k: prep[k] for k in ("layout", "count_checks", "old_tokens_fail_check",
                                          "deployed_token_reproduction", "tokens_v3_sha256",
                                          "tokens_deployed_sha256", "counts_sha256")},
        "wall_seconds": prep["seconds"],
    })
    log(f"prepare DONE: {n_total:,} positions, split {split:,}; eval cells {c_first_eval}..{N_CTRL-1}; "
        f"deployed tokens reproduced: {dep_repro['identical']}; old tokens fail check {n_old_fail}/500; "
        f"Spearman old vs v3 {agree_summary['spearman_full']['mean']:.3f}")


# =============================================================================
# extract (MPS)
# =============================================================================
def _write_header(path: Path, n_rows: int) -> int:
    hdr = {"descr": np.lib.format.dtype_to_descr(np.dtype("<f4")), "fortran_order": False,
           "shape": (int(n_rows), HIDDEN)}
    with open(path, "wb") as f:
        np.lib.format.write_array_header_1_0(f, hdr)
        return f.tell()


def cmd_extract(args):
    t_start = time.time()
    ACT_DIR.mkdir(parents=True, exist_ok=True)
    cells = load_cells()
    lay = layout()
    offsets, toks, rows = cells["offsets"], cells["tokens_v3"], cells["rows"]
    n_total = lay["n_total"]
    prog_path = OUT / "extract_progress.json"
    prog = read_json(prog_path, {"n_done": 0, "header_len": None})
    n_done = int(prog["n_done"])
    if n_done >= N_CTRL:
        log("extract already DONE")
        return
    # files: header + appended rows; truncate to the last committed cell on resume
    hlen = None
    for l in range(N_LAYERS):
        p = act_path(l)
        if n_done == 0 or not p.exists():
            if n_done != 0:
                raise SystemExit(f"{p} missing but progress says {n_done} cells done; delete progress to restart")
            h = _write_header(p, n_total)
        else:
            h, shape = npy_header_len(p)
            if tuple(shape) != (n_total, HIDDEN):
                raise SystemExit(f"{p} has shape {shape}")
        hlen = h if hlen is None else hlen
        if h != hlen:
            raise SystemExit("header lengths differ")
        want = h + int(offsets[n_done]) * HIDDEN * 4
        size = p.stat().st_size
        if size < want:
            raise SystemExit(f"{p}: size {size} < committed {want}; inconsistent, restart extraction")
        if size > want:
            os.truncate(p, want)
    vm = iv3.get_var_map("k562")
    guard("extract start")
    xt = hv2.load_model(DEVICE)
    model_info = {"n_layers": xt.n_layers, "hidden_size": xt.hidden_size}
    assert xt.hidden_size == HIDDEN and xt.n_layers + 1 == N_LAYERS
    fds = [os.open(act_path(l), os.O_WRONLY | os.O_APPEND) for l in range(N_LAYERS)]
    n_start = n_done
    batch = 25
    checks = []
    try:
        reader = iv3.CountReader("k562")
        reader.__enter__()
        while n_done < N_CTRL:
            if (time.time() - t_start) / 60 > args.max_minutes:
                break
            gb = guard(f"extract cell {n_done}")
            b_end = min(N_CTRL, n_done + batch)
            # encoding check on every cell of this batch, before its forward passes
            cnts = [reader.counts(int(rows[i])) for i in range(n_done, b_end)]
            seqs = [toks[offsets[i]:offsets[i + 1]] for i in range(n_done, b_end)]
            chk = iv3.assert_encoding_batch(cnts, seqs, vm, MAX_LEN, n_first=None,
                                            label=f"extract cells {n_done}-{b_end - 1}")
            checks.append({"cells": [n_done, b_end - 1], "all_pass": chk["all_pass"],
                           "n_checked": chk["n_checked"], "mem_gb": round(gb, 2)})
            for ci in range(n_done, b_end):
                ids = torch.from_numpy(seqs[ci - n_done].astype(np.int64)[None, :])
                _, hidden = xt.forward_with_hidden_states(ids)
                assert len(hidden) == N_LAYERS
                for l in range(N_LAYERS):
                    a = hidden[l][0].numpy().astype(np.float32, copy=False)
                    assert a.shape == (len(seqs[ci - n_done]), HIDDEN)
                    if not np.isfinite(a).all():
                        raise SystemExit(f"non-finite activations cell {ci} layer {l}")
                    os.write(fds[l], np.ascontiguousarray(a).tobytes())
                del hidden
                hv2.free_device_cache()
            for fd in fds:
                os.fsync(fd)
            n_done = b_end
            hv2.write_json(prog_path, {"n_done": n_done, "header_len": hlen})
            log(f"extract: {n_done}/{N_CTRL} cells  ({(time.time() - t_start) / 60:.1f} min, mem {gb:.1f} GB)")
        reader.__exit__()
    finally:
        for fd in fds:
            os.close(fd)
    rec = {"step": "extract", "cells": [n_start, n_done - 1], "wall_seconds": time.time() - t_start,
           "encoding_checks": checks, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    append_chunk_record("extract", rec)
    if n_done >= N_CTRL:
        sizes = {f"layer_{l:02d}": act_path(l).stat().st_size for l in range(N_LAYERS)}
        expect = hlen + n_total * HIDDEN * 4
        bad = {k: v for k, v in sizes.items() if v != expect}
        if bad:
            raise SystemExit(f"final sizes wrong: {bad} (expected {expect})")
        update_run_config("extract", {
            "env": hv2.env_info(DEVICE), "model": model_info,
            "site": "hidden_states[l] from output_hidden_states (as deployed full_12layer_pipeline.py:140-141)",
            "n_positions": n_total, "file_bytes_each": expect,
            "total_GB": round(expect * N_LAYERS / 1e9, 2),
            "encoding_check": "inputs_v3.assert_encoding_batch on every cell of every 25-cell batch before "
                              "its forward passes (see chunks.json)",
            "chunks": read_json(OUT / "chunks.json")["extract"]})
        log("extract DONE")
    else:
        log(f"extract paused at {n_done}/{N_CTRL}; run again")


# =============================================================================
# train (MPS), reproduces topk_sae.train_sae
# =============================================================================
def _load_rows(path: Path, start: int, stop: int) -> np.ndarray:
    """Rows [start, stop) of a float32 (N, HIDDEN) .npy file, read sequentially into RAM.
    F_NOCACHE keeps the read out of the page cache (so the data are not held twice)."""
    import fcntl
    hlen, shape = npy_header_len(path)
    assert shape[1] == HIDDEN and 0 <= start < stop <= shape[0]
    out = np.empty((stop - start, HIDDEN), dtype=np.float32)
    buf = memoryview(out.reshape(-1).view(np.uint8))
    with open(path, "rb", buffering=0) as f:
        try:
            fcntl.fcntl(f.fileno(), getattr(fcntl, "F_NOCACHE", 48), 1)
        except OSError:
            pass
        f.seek(hlen + start * HIDDEN * 4)
        got = 0
        while got < len(buf):
            r = f.readinto(buf[got:got + (64 << 20)])
            if not r:
                raise IOError(f"short read in {path}")
            got += r
    return out


def _cache_train_rows(layer: int, split: int) -> np.ndarray:
    """Copy rows [0, split) of layer_XX.npy to CACHE_DIR (raw float32) once; return a read-only memmap.
    The copy is checked byte for byte against the source (sha256) and deleted after training."""
    import fcntl
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dst = CACHE_DIR / f"layer_{layer:02d}_train.f32"
    nbytes = split * HIDDEN * 4
    src = act_path(layer)
    hlen, _ = npy_header_len(src)
    if not (dst.exists() and dst.stat().st_size == nbytes and (CACHE_DIR / f"{dst.name}.ok").exists()):
        h_src = hashlib.sha256()
        with open(src, "rb", buffering=0) as fi, open(dst, "wb") as fo:
            try:
                fcntl.fcntl(fi.fileno(), getattr(fcntl, "F_NOCACHE", 48), 1)
            except OSError:
                pass
            fi.seek(hlen)
            left = nbytes
            while left:
                b = fi.read(min(left, 64 << 20))
                if not b:
                    raise IOError("short read")
                h_src.update(b)
                fo.write(b)
                left -= len(b)
        h_dst = hashlib.sha256()
        with open(dst, "rb") as fi:
            for b in iter(lambda: fi.read(64 << 20), b""):
                h_dst.update(b)
        if h_src.hexdigest() != h_dst.hexdigest():
            raise IOError("cache copy differs from source")
        (CACHE_DIR / f"{dst.name}.ok").write_text(h_src.hexdigest())
    return RowFile(dst, split)


class RowFile:
    """Read-only float32 (n, HIDDEN) raw file. Reads with pread and F_NOCACHE (no page cache, no
    anonymous memory beyond the rows asked for). Slices read one block; index arrays are read
    row by row by 16 threads (rows sorted for locality, then put back in the asked order)."""

    def __init__(self, path: Path, n: int):
        import fcntl
        from concurrent.futures import ThreadPoolExecutor
        self.path, self.shape, self.row = Path(path), (int(n), HIDDEN), HIDDEN * 4
        self.fd = os.open(self.path, os.O_RDONLY)
        try:
            fcntl.fcntl(self.fd, getattr(fcntl, "F_NOCACHE", 48), 1)
        except OSError:
            pass
        self.pool = ThreadPoolExecutor(16)

    def _rows(self, rows) -> bytes:
        return b"".join(os.pread(self.fd, self.row, int(r) * self.row) for r in rows)

    def __getitem__(self, key):
        if isinstance(key, slice):
            a, b, _ = key.indices(self.shape[0])
            buf = os.pread(self.fd, (b - a) * self.row, a * self.row)
            if len(buf) != (b - a) * self.row:
                raise IOError("short read")
            return np.frombuffer(buf, dtype=np.float32).reshape(b - a, HIDDEN)
        idx = np.asarray(key)
        order = np.argsort(idx, kind="stable")
        parts = list(self.pool.map(self._rows, np.array_split(idx[order], 32)))
        srt = np.frombuffer(b"".join(parts), dtype=np.float32).reshape(len(idx), HIDDEN)
        out = np.empty_like(srt)
        out[order] = srt
        return out

    def close(self):
        self.pool.shutdown()
        os.close(self.fd)


def _train_mean(acts, split) -> np.ndarray:
    s = np.zeros(HIDDEN, dtype=np.float64)
    for a in range(0, split, 65536):
        s += acts[a:min(split, a + 65536)].astype(np.float64).sum(axis=0)
    return (s / split).astype(np.float32)


@torch.no_grad()
def eval_like_deployed(model, X_ev_mm, mu, seed: int) -> dict:
    """topk_sae.train_sae evaluation block, in chunks (same numbers up to summation order)."""
    dev = next(model.parameters()).device
    mu_dev = mu.to(dev)
    sse, sst, n_el = 0.0, 0.0, 0
    alive = torch.zeros(model.d_sae, dtype=torch.bool)
    l0 = 0
    for a in range(0, X_ev_mm.shape[0], 8192):
        xb = torch.from_numpy(np.array(X_ev_mm[a:a + 8192], dtype=np.float32)).to(dev)
        x_hat, h = model(xb, mu_dev)
        sse += float(((xb - x_hat) ** 2).sum(dim=1).cpu().double().sum().item())
        sst += float(((xb - mu_dev) ** 2).sum(dim=1).cpu().double().sum().item())
        n_el += xb.numel()
        # counts on CPU: on MPS, bool .sum() sometimes returns wrong totals (seen: 57.5 "active"
        # features per row with k = 32); the deployed code also counted on CPU (topk_sae.py)
        act = h.cpu() > 0
        alive |= act.any(dim=0)
        l0 += int(act.sum().item())
    var_explained = 1.0 - (sse / n_el) / max(sst / n_el, 1e-12)
    n_alive = int(alive.sum().item())
    W = model.W_dec.weight.detach().cpu().numpy()
    rng_p = np.random.default_rng(seed + 999)
    pair_cos = []
    for _ in range(min(500, model.d_sae * (model.d_sae - 1) // 2)):
        i, j = rng_p.choice(model.d_sae, 2, replace=False)
        pair_cos.append(float(np.abs((W[:, i] * W[:, j]).sum())))
    return {"variance_explained": float(var_explained), "n_alive": n_alive, "n_dead": model.d_sae - n_alive,
            "dead_rate": float((model.d_sae - n_alive) / model.d_sae),
            "mean_abs_decoder_cosine": float(np.mean(pair_cos)),
            "mean_l0_eval": l0 / X_ev_mm.shape[0]}


def train_one(layer: int, max_minutes: float, t_call: float) -> str:
    from torch.utils.data import DataLoader, TensorDataset
    out_dir = OUT / f"layer_{layer:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "results.json").exists() and (out_dir / "sae_final.pt").exists():
        r = read_json(out_dir / "results.json")
        if not r.get("counts_on_cpu"):
            # first version counted active features with an MPS bool sum (unreliable): recount
            sae = hv2.load_sae(layer, DEVICE, OUT)
            ev = eval_like_deployed(sae.sae, open_acts(layer)[layout()["split"]:], sae.mu.cpu(), SEED)
            old = {k: r.get(k) for k in ("variance_explained", "n_alive", "n_dead", "mean_l0_eval")}
            r.update({k: ev[k] for k in ("variance_explained", "n_alive", "n_dead", "dead_rate", "mean_l0_eval")})
            r["counts_on_cpu"] = True
            r["recount_note"] = f"recounted on CPU after an MPS bool-sum bug; first values {old}"
            hv2.write_json(out_dir / "results.json", r)
            log(f"  L{layer}: recounted {old} -> FVE {ev['variance_explained']:.4f} dead {ev['n_dead']} "
                f"L0 {ev['mean_l0_eval']:.3f}")
        return "done"
    prog = read_json(OUT / "extract_progress.json", {"n_done": 0})
    if prog["n_done"] < N_CTRL:
        raise SystemExit("extraction not finished")
    t_layer = time.time()
    acts = open_acts(layer)
    n_total = acts.shape[0]
    assert acts.shape == (layout()["n_total"], HIDDEN) and acts.dtype == np.float32
    # ---- train_sae, line by line ----
    rng = np.random.default_rng(SEED)  # noqa: F841  (unused in the else-branch, as deployed)
    torch.manual_seed(SEED)
    if n_total > N_TRAIN + N_EVAL:
        raise SystemExit("random-permutation branch not expected")
    split = int(0.9 * n_total)
    n_train, n_eval = split, n_total - split
    X_eval_mm = acts[split:]
    ckpt_path = out_dir / "train_ckpt.pt"
    guard(f"train L{layer} start")
    t_mu = time.time()
    if CACHE_DIR is not None:
        # sha256-checked copy on the internal SSD, read with pread (no page cache, small memory)
        X_train = _cache_train_rows(layer, split)
    else:
        X_train = _load_rows(act_path(layer), 0, split)  # in RAM, 4.5 GB
    guard(f"train L{layer} after loading")
    mu = torch.from_numpy(_train_mean(X_train, split))
    t_mu = time.time() - t_mu
    model = TopKSAE(HIDDEN, D_SAE, K).to(DEVICE)
    mu_dev = mu.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loader = DataLoader(TensorDataset(torch.arange(n_train)), batch_size=BATCH, shuffle=True, drop_last=False)
    log_rows, step, start_epoch, t_train_prev = [], 0, 0, 0.0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        torch.set_rng_state(ck["torch_rng"])
        log_rows, step, start_epoch, t_train_prev = ck["log"], ck["step"], ck["epoch"] + 1, ck["train_seconds"]
        if not torch.equal(ck["mu"], mu):
            raise SystemExit("mu differs from checkpoint")
        log(f"  L{layer}: resumed after epoch {ck['epoch']}")
    t0 = time.time() - t_train_prev
    from concurrent.futures import ThreadPoolExecutor
    prefetch = ThreadPoolExecutor(1)
    for epoch in range(start_epoch, EPOCHS):
        guard(f"train L{layer} epoch {epoch}")
        if (time.time() - t_call) / 60 > max_minutes - 2.5:
            prefetch.shutdown()
            if isinstance(X_train, RowFile):
                X_train.close()
            return "paused"
        model.train()
        epoch_loss, n_batches, t_io = 0.0, 0, 0.0
        # the shuffle is drawn when the iterator is created, so listing the batches first
        # uses the torch RNG exactly as iterating them one by one
        batches = [ib.numpy() for (ib,) in loader]
        nxt = prefetch.submit(X_train.__getitem__, batches[0])
        for bi in range(len(batches)):
            ti = time.time()
            xb_np = nxt.result()
            if bi + 1 < len(batches):
                nxt = prefetch.submit(X_train.__getitem__, batches[bi + 1])
            t_io += time.time() - ti
            xb = torch.from_numpy(xb_np).to(DEVICE)
            x_hat, h = model(xb, mu_dev)
            loss = ((xb - x_hat) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model._normalise_decoder()
            epoch_loss += loss.item()
            n_batches += 1
            step += 1
        log_rows.append({"epoch": epoch, "loss": epoch_loss / max(n_batches, 1), "steps": step,
                         "seconds": time.time() - t0})
        log(f"  L{layer} epoch {epoch}: loss={log_rows[-1]['loss']:.6f} steps={step} "
            f"time={log_rows[-1]['seconds']:.1f}s (batch reads {t_io:.1f}s)")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "torch_rng": torch.get_rng_state(), "log": log_rows, "step": step, "epoch": epoch,
                    "train_seconds": time.time() - t0, "mu": mu}, ckpt_path)
    model.eval()
    ev = eval_like_deployed(model, X_eval_mm, mu, SEED)
    results = {"d_model": HIDDEN, "d_sae": D_SAE, "k": K, "n_train": int(n_train), "n_eval": int(n_eval),
               "n_epochs": EPOCHS, "lr": LR, "batch_size": BATCH,
               "variance_explained": ev["variance_explained"], "n_alive": ev["n_alive"],
               "n_dead": ev["n_dead"], "dead_rate": ev["dead_rate"],
               "mean_abs_decoder_cosine": ev["mean_abs_decoder_cosine"],
               "total_train_seconds": float(time.time() - t0), "training_log": log_rows,
               # v3 additions (deployed results.json has the keys above only)
               "mean_l0_eval": ev["mean_l0_eval"], "counts_on_cpu": True,
               "input_encoding": "v3: counts = round(expm1(X)/unit) via setup/inputs_v3.py (checked)",
               "train_mean_seconds": t_mu, "device": DEVICE}
    torch.save({"W_enc_weight": model.W_enc.weight.detach().cpu(),
                "W_enc_bias": model.W_enc.bias.detach().cpu(),
                "W_dec_weight": model.W_dec.weight.detach().cpu(),
                "mu": mu.cpu(),
                "config": {"d_model": HIDDEN, "d_sae": D_SAE, "k": K}}, out_dir / "sae_final.pt")
    hv2.write_json(out_dir / "training_log.json", log_rows)
    hv2.write_json(out_dir / "results.json", results)
    if ckpt_path.exists():
        ckpt_path.unlink()
    prefetch.shutdown()
    if isinstance(X_train, RowFile):
        X_train.close()
    del X_train
    if CACHE_DIR is not None:
        (CACHE_DIR / f"layer_{layer:02d}_train.f32").unlink(missing_ok=True)
        (CACHE_DIR / f"layer_{layer:02d}_train.f32.ok").unlink(missing_ok=True)
    append_chunk_record("train", {"layer": layer, "wall_seconds": time.time() - t_layer,
                                  "train_rows_source": "pread of a sha256-checked copy on the internal SSD" if CACHE_DIR else "RAM",
                                  "resumed_from_epoch": start_epoch,
                                  "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
    log(f"  L{layer} DONE: FVE={ev['variance_explained']:.4f} dead={ev['n_dead']} "
        f"L0={ev['mean_l0_eval']:.2f} ({time.time() - t_layer:.0f}s)")
    return "done"


def cmd_train(args):
    t_call = time.time()
    layers = range(N_LAYERS) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    for l in layers:
        if (time.time() - t_call) / 60 > args.max_minutes - 2.5:
            log("time budget used; run again")
            return
        st = train_one(l, args.max_minutes, t_call)
        if st == "paused":
            log(f"paused in layer {l}; run again")
            return
        hv2.free_device_cache()
    done = all((OUT / f"layer_{l:02d}/results.json").exists() for l in range(N_LAYERS))
    if done:
        update_run_config("train", {
            "env": hv2.env_info(DEVICE),
            "hyperparameters": {"d_model": HIDDEN, "d_sae": D_SAE, "k": K, "lr": LR, "batch_size": BATCH,
                                "n_epochs": EPOCHS, "n_train_cap": N_TRAIN, "n_eval_cap": N_EVAL,
                                "seed": SEED, "optimizer": "Adam", "loss": "MSE",
                                "decoder": "unit-norm columns after each step"},
            "split": {k: layout()[k] for k in ("n_total", "split", "n_train", "n_eval")},
            "chunks": read_json(OUT / "chunks.json").get("train", []),
            "per_layer": {f"layer_{l:02d}": read_json(OUT / f"layer_{l:02d}/results.json") for l in range(N_LAYERS)},
        })
        log("train DONE (all 12 layers)")


# =============================================================================
# evaluate (MPS): per-cell sufficient statistics on held-out tokens + all-token pass
# =============================================================================
@torch.no_grad()
def _encode_topk(sae: hv2.LoadedSAE, xb: torch.Tensor):
    pre = sae.sae.W_enc(xb - sae.mu)
    vals, idx = pre.topk(sae.sae.k, dim=-1)
    vals = vals.clamp(min=0)
    h = torch.zeros_like(pre)
    h.scatter_(1, idx, vals)
    x_hat = sae.sae.W_dec(h) + sae.mu
    return vals, idx, x_hat


def _per_cell_stats(sae: hv2.LoadedSAE, X: np.ndarray, cell_ids: np.ndarray, n_cells_eval: int) -> dict:
    """Sufficient statistics per held-out cell (X rows already restricted to held-out positions)."""
    dev = sae.mu.device
    out = {"n": np.zeros(n_cells_eval, np.int64), "sse": np.zeros(n_cells_eval),
           "sst_own": np.zeros(n_cells_eval), "l0": np.zeros(n_cells_eval, np.int64),
           "feat_count": np.zeros((n_cells_eval, D_SAE), np.int64)}
    for a in range(0, X.shape[0], 8192):
        b = min(X.shape[0], a + 8192)
        xb = torch.from_numpy(np.ascontiguousarray(X[a:b], dtype=np.float32)).to(dev)
        vals, idx, x_hat = _encode_topk(sae, xb)
        se = ((xb - x_hat) ** 2).sum(dim=1).cpu().double().numpy()
        so = ((xb - sae.mu) ** 2).sum(dim=1).cpu().double().numpy()
        act_np = vals.cpu().numpy() > 0          # counted on CPU (MPS bool sums are unreliable)
        idx_np = idx.cpu().numpy()
        l0 = act_np.sum(axis=1)
        cid = cell_ids[a:b]
        np.add.at(out["n"], cid, 1)
        np.add.at(out["sse"], cid, se)
        np.add.at(out["sst_own"], cid, so)
        np.add.at(out["l0"], cid, l0)
        for c in np.unique(cid):
            m = cid == c
            out["feat_count"][c] += np.bincount(idx_np[m][act_np[m]], minlength=D_SAE)
    return out


def cmd_evaluate(args):
    t_call = time.time()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_ACT_DIR.mkdir(parents=True, exist_ok=True)
    CODES_DIR.mkdir(parents=True, exist_ok=True)
    lay = layout()
    cells = load_cells()
    offsets = cells["offsets"]
    split, n_total = lay["split"], lay["n_total"]
    pos_eval = np.arange(split, n_total)
    cell_eval_abs = cell_of_position(offsets, pos_eval)
    c0 = lay["first_eval_cell"]
    cell_eval = (cell_eval_abs - c0).astype(np.int64)
    n_ce = N_CTRL - c0
    for l in range(N_LAYERS):
        res_path = EVAL_DIR / f"layer_{l:02d}.npz"
        if res_path.exists():
            continue
        if (time.time() - t_call) / 60 > args.max_minutes - 2.0:
            log("time budget used; run again")
            return
        if not (OUT / f"layer_{l:02d}/sae_final.pt").exists():
            raise SystemExit(f"layer {l} not trained")
        guard(f"evaluate L{l}")
        t_l = time.time()
        acts = open_acts(l)
        # kept copy of the held-out rows (small; the full file may be deleted later)
        ev_path = EVAL_ACT_DIR / f"layer_{l:02d}.npy"
        X_ev = np.ascontiguousarray(acts[split:])
        if not ev_path.exists():
            np.save(ev_path, X_ev)
        saes = {"v3": hv2.load_sae(l, DEVICE, OUT), "deployed": hv2.load_sae(l, DEVICE, DEPLOYED_SAE_DIR)}
        stats = {}
        for name, sae in saes.items():
            st = _per_cell_stats(sae, X_ev, cell_eval, n_ce)
            for k, v in st.items():
                stats[f"{name}__{k}"] = v
        # per-cell sums for the common (held-out mean) baseline
        sum_x = np.zeros((n_ce, HIDDEN))
        sum_x2 = np.zeros(n_ce)
        for c in range(n_ce):
            m = cell_eval == c
            xc = X_ev[m].astype(np.float64)
            sum_x[c] = xc.sum(axis=0)
            sum_x2[c] = float((xc ** 2).sum())
        # all-token pass: dead features over all 1,019,996 positions, in-sample FVE, v3 codes
        h = hashlib.sha256()
        all_counts = {n: np.zeros(D_SAE, np.int64) for n in saes}
        train_sse = {n: 0.0 for n in saes}
        train_sst_own = {n: 0.0 for n in saes}
        l0_all = {n: 0 for n in saes}
        codes_idx = np.zeros((n_total, K), np.int16)
        codes_val = np.zeros((n_total, K), np.float32)
        for a in range(0, n_total, 16384):
            b = min(n_total, a + 16384)
            xa = np.ascontiguousarray(acts[a:b], dtype=np.float32)
            h.update(xa.tobytes())
            for name, sae in saes.items():
                xb = torch.from_numpy(xa).to(sae.mu.device)
                vals, idx, x_hat = _encode_topk(sae, xb)
                vals_np, idx_np = vals.cpu().numpy(), idx.cpu().numpy()
                act_np = vals_np > 0                 # counted on CPU
                all_counts[name] += np.bincount(idx_np[act_np], minlength=D_SAE)
                l0_all[name] += int(act_np.sum())
                if a < split:
                    bb = min(b, split) - a
                    train_sse[name] += float(((xb[:bb] - x_hat[:bb]) ** 2).sum(dim=1).cpu().double().sum().item())
                    train_sst_own[name] += float(((xb[:bb] - sae.mu) ** 2).sum(dim=1).cpu().double().sum().item())
                if name == "v3":
                    codes_idx[a:b] = idx_np.astype(np.int16)
                    codes_val[a:b] = vals_np
        np.savez(CODES_DIR / f"layer_{l:02d}_topk.npz", idx=codes_idx, val=codes_val)
        del codes_idx, codes_val
        np.savez(res_path, sum_x=sum_x, sum_x2=sum_x2, cell_abs=np.arange(c0, N_CTRL),
                 **stats,
                 **{f"{n}__all_feat_count": all_counts[n] for n in saes},
                 **{f"{n}__train_sse": np.array(train_sse[n]) for n in saes},
                 **{f"{n}__train_sst_own": np.array(train_sst_own[n]) for n in saes},
                 **{f"{n}__l0_all": np.array(l0_all[n]) for n in saes})
        meta = read_json(EVAL_DIR / "meta.json", {}) or {}
        meta[f"layer_{l:02d}"] = {"activations_data_sha256": h.hexdigest(),
                                  "eval_rows_sha256": iv3.array_sha256(X_ev),
                                  "v3_sae_sha256": iv3.sha256_file(OUT / f"layer_{l:02d}/sae_final.pt"),
                                  "deployed_sae_sha256": iv3.sha256_file(DEPLOYED_SAE_DIR / f"layer_{l:02d}/sae_final.pt"),
                                  "seconds": time.time() - t_l}
        hv2.write_json(EVAL_DIR / "meta.json", meta)
        del saes, X_ev
        hv2.free_device_cache()
        log(f"evaluate L{l} done ({time.time() - t_l:.0f}s)")
    update_run_config("evaluate", {"env": hv2.env_info(DEVICE), "meta": read_json(EVAL_DIR / "meta.json"),
                                   "held_out": {"positions": [split, n_total], "cells": [c0, N_CTRL - 1]}})
    log("evaluate DONE")


# =============================================================================
# hooks_check (MPS): load the v3 SAEs through hooks_v2; live captures == stored activations
# =============================================================================
def cmd_hooks_check(args):
    t0 = time.time()
    lay = layout()
    cells = load_cells()
    offsets, toks, rows = cells["offsets"], cells["tokens_v3"], cells["rows"]
    split = lay["split"]
    c0 = lay["first_eval_cell"]
    pick = [c0 + 1, (c0 + N_CTRL) // 2, N_CTRL - 1]  # fully held-out cells
    vm = iv3.get_var_map("k562")
    cnts, _ = iv3.read_counts("k562", [int(rows[c]) for c in pick])
    seqs = [toks[offsets[c]:offsets[c + 1]] for c in pick]
    enc = iv3.assert_encoding_batch(cnts, seqs, vm, MAX_LEN, label="hooks_check")
    guard("hooks_check")
    xt = hv2.load_model(DEVICE)
    model = xt.model
    saes = hv2.load_saes(range(N_LAYERS), DEVICE, OUT)
    ev = {l: np.load(EVAL_DIR / f"layer_{l:02d}.npz") for l in range(N_LAYERS)}
    out = {"cells": pick, "encoding_check": enc, "per_layer": {}}
    worst = 0.0
    for ci, c in enumerate(pick):
        ids = torch.from_numpy(seqs[ci].astype(np.int64)[None, :])
        _, ed = hv2.run_with_edits(model, ids, saes, edits=(), capture=range(N_LAYERS), return_logits=False)
        for l in range(N_LAYERS):
            live = ed.hidden(l, "pre").numpy()
            stored = np.load(EVAL_ACT_DIR / f"layer_{l:02d}.npy", mmap_mode="r")[
                offsets[c] - split: offsets[c + 1] - split]
            d = float(np.abs(live - stored).max())
            scale = float(np.abs(stored).max())
            z = ed.codes(l, "pre", sae=saes[l]).numpy()  # dense codes on CPU
            x_hat = z @ saes[l].W_dec.cpu().numpy().T + saes[l].mu.cpu().numpy()
            sse_live = float(((live.astype(np.float64) - x_hat) ** 2).sum())
            sse_eval = float(ev[l]["v3__sse"][c - c0])
            rec = out["per_layer"].setdefault(f"layer_{l:02d}", [])
            rec.append({"cell": int(c), "max_abs_diff_live_vs_stored": d, "max_abs_stored": scale,
                        "sse_live": sse_live, "sse_eval": sse_eval,
                        "sse_rel_diff": abs(sse_live - sse_eval) / max(sse_eval, 1e-12),
                        "l0_live": float((z > 0).sum(axis=1).mean())})
            worst = max(worst, d / max(scale, 1e-12))
        del ed
        hv2.free_device_cache()
    # Ablate every feature at layer 5 on one cell: post must equal pre - (x_hat - mu)
    l = 5
    ids = torch.from_numpy(seqs[0].astype(np.int64)[None, :])
    _, ed = hv2.run_with_edits(model, ids, saes, edits=[hv2.Ablate(l, features=list(range(D_SAE)))],
                               capture=[l], return_logits=False)
    pre, post = ed.hidden(l, "pre"), ed.hidden(l, "post")
    with torch.no_grad():
        _, _, x_hat = _encode_topk(saes[l], pre.to(DEVICE))
        expect = (pre.to(DEVICE) - (x_hat - saes[l].mu)).cpu()
    out["ablate_all_layer5"] = {"max_abs_diff": float((post - expect).abs().max()),
                                "max_abs_pre": float(pre.abs().max())}
    out["max_rel_diff_live_vs_stored"] = worst
    max_sse_rel = max(r["sse_rel_diff"] for v in out["per_layer"].values() for r in v)
    out["max_sse_rel_diff"] = max_sse_rel
    out["pass"] = bool(worst < 1e-4 and max_sse_rel < 1e-3 and out["ablate_all_layer5"]["max_abs_diff"] < 1e-3)
    out["seconds"] = time.time() - t0
    hv2.write_json(OUT / "hooks_check.json", out)
    update_run_config("hooks_check", {"pass": out["pass"], "max_rel_diff_live_vs_stored": worst,
                                      "max_sse_rel_diff": max_sse_rel,
                                      "ablate_all_layer5": out["ablate_all_layer5"],
                                      "encoding_check": enc, "wall_seconds": out["seconds"]})
    log(f"hooks_check: pass={out['pass']} live-vs-stored rel {worst:.2e}; SSE rel {max_sse_rel:.2e}; "
        f"ablate {out['ablate_all_layer5']['max_abs_diff']:.2e}")


# =============================================================================
# summarize (CPU): bootstrap over held-out cells
# =============================================================================
def cmd_summarize(args):
    lay = layout()
    rng = np.random.default_rng(BOOT_SEED)
    n_ce = N_CTRL - lay["first_eval_cell"]
    boot_idx = rng.integers(0, n_ce, size=(N_BOOT, n_ce))
    rows_out, summ = [], {"method": f"cluster bootstrap over the {n_ce} held-out cells (the first is only "
                                    f"partly held out), {N_BOOT} resamples, seed {BOOT_SEED}, 95% percentile CI",
                          "layers": {}}

    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    for l in range(N_LAYERS):
        z = np.load(EVAL_DIR / f"layer_{l:02d}.npz")
        n, sx, sx2 = z["v3__n"], z["sum_x"], z["sum_x2"]
        if not np.array_equal(n, z["deployed__n"]):
            raise SystemExit("token counts per cell differ between SAEs")
        # vectorised bootstrap via cell weights (common baseline: the resample's own mean)
        W = np.zeros((N_BOOT, n_ce))
        for b in range(N_BOOT):
            W[b] = np.bincount(boot_idx[b], minlength=n_ce)
        N_b = W @ n
        SX_b = W @ sx
        SST_b = W @ sx2 - (SX_b ** 2).sum(1) / N_b
        res = {}
        point = {}
        for name in ("v3", "deployed"):
            sse, sst_own, l0 = z[f"{name}__sse"], z[f"{name}__sst_own"], z[f"{name}__l0"]
            fc = z[f"{name}__feat_count"]
            N = n.sum()
            SST = sx2.sum() - (sx.sum(0) ** 2).sum() / N
            point[name] = {
                "fve_heldout_mean_baseline": float(1 - sse.sum() / SST),
                "fve_own_mu_baseline": float(1 - sse.sum() / sst_own.sum()),
                "mean_l0": float(l0.sum() / N),
                "dead_heldout": int(np.sum(fc.sum(0) == 0)),
                "dead_all_tokens": int(np.sum(z[f"{name}__all_feat_count"] == 0)),
                "fve_train_tokens_own_mu": float(1 - z[f"{name}__train_sse"] / z[f"{name}__train_sst_own"]),
                "mean_l0_all_tokens": float(z[f"{name}__l0_all"] / lay["n_total"]),
            }
            b_fve = 1 - (W @ sse) / SST_b
            b_own = 1 - (W @ sse) / (W @ sst_own)
            b_l0 = (W @ l0) / N_b
            res[name] = {"fve": b_fve, "own": b_own, "l0": b_l0}
            point[name]["fve_heldout_mean_baseline_ci"] = ci(b_fve)
            point[name]["fve_own_mu_baseline_ci"] = ci(b_own)
            point[name]["mean_l0_ci"] = ci(b_l0)
        diff = {"fve_v3_minus_deployed": float(point["v3"]["fve_heldout_mean_baseline"] -
                                               point["deployed"]["fve_heldout_mean_baseline"]),
                "fve_v3_minus_deployed_ci": ci(res["v3"]["fve"] - res["deployed"]["fve"]),
                "l0_v3_minus_deployed": float(point["v3"]["mean_l0"] - point["deployed"]["mean_l0"]),
                "l0_v3_minus_deployed_ci": ci(res["v3"]["l0"] - res["deployed"]["l0"])}
        v3r = read_json(OUT / f"layer_{l:02d}/results.json")
        depr = read_json(DEPLOYED_SAE_DIR / f"layer_{l:02d}/results.json")
        summ["layers"][f"layer_{l:02d}"] = {"v3": point["v3"], "deployed_on_v3_tokens": point["deployed"],
                                            "paired": diff,
                                            "v3_results_json": {k: v3r[k] for k in ("variance_explained", "n_dead",
                                                                                    "mean_l0_eval")},
                                            "deployed_results_json_old_tokens": {k: depr[k] for k in (
                                                "variance_explained", "n_dead")},
                                            "check_fve_two_ways_absdiff": abs(
                                                v3r["variance_explained"] - point["v3"]["fve_own_mu_baseline"]),
                                            "check_l0_two_ways_absdiff": abs(
                                                v3r["mean_l0_eval"] - point["v3"]["mean_l0"]),
                                            "check_dead_two_ways_equal": v3r["n_dead"] == point["v3"]["dead_heldout"]}
        P, D = point["v3"], point["deployed"]
        rows_out.append({
            "layer": l,
            "v3_fve": P["fve_heldout_mean_baseline"], "v3_fve_lo": P["fve_heldout_mean_baseline_ci"][0],
            "v3_fve_hi": P["fve_heldout_mean_baseline_ci"][1],
            "v3_dead_heldout": P["dead_heldout"], "v3_dead_all": P["dead_all_tokens"],
            "v3_l0": P["mean_l0"], "v3_l0_lo": P["mean_l0_ci"][0], "v3_l0_hi": P["mean_l0_ci"][1],
            "dep_fve": D["fve_heldout_mean_baseline"], "dep_fve_lo": D["fve_heldout_mean_baseline_ci"][0],
            "dep_fve_hi": D["fve_heldout_mean_baseline_ci"][1],
            "dep_dead_heldout": D["dead_heldout"], "dep_dead_all": D["dead_all_tokens"],
            "dep_l0": D["mean_l0"], "dep_l0_lo": D["mean_l0_ci"][0], "dep_l0_hi": D["mean_l0_ci"][1],
            "fve_diff": diff["fve_v3_minus_deployed"], "fve_diff_lo": diff["fve_v3_minus_deployed_ci"][0],
            "fve_diff_hi": diff["fve_v3_minus_deployed_ci"][1],
            "v3_fve_own_mu": P["fve_own_mu_baseline"], "dep_fve_own_mu": D["fve_own_mu_baseline"],
            "v3_fve_train": P["fve_train_tokens_own_mu"], "dep_fve_old_tokens_reported": depr["variance_explained"],
            "dep_dead_old_tokens_reported": depr["n_dead"],
        })
    import csv
    with open(EVAL_DIR / "table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    hv2.write_json(EVAL_DIR / "summary.json", summ)
    update_run_config("summarize", {"bootstrap": summ["method"], "table": str(EVAL_DIR / "table.csv")})
    for r in rows_out:
        log(f"L{r['layer']:2d}  v3 FVE {r['v3_fve']:.4f} [{r['v3_fve_lo']:.4f},{r['v3_fve_hi']:.4f}] "
            f"dead {r['v3_dead_heldout']}/{r['v3_dead_all']} L0 {r['v3_l0']:.2f} | deployed FVE {r['dep_fve']:.4f} "
            f"dead {r['dep_dead_heldout']}/{r['dep_dead_all']} L0 {r['dep_l0']:.2f} | diff {r['fve_diff']:+.4f} "
            f"[{r['fve_diff_lo']:+.4f},{r['fve_diff_hi']:+.4f}]")


# =============================================================================
# cleanup
# =============================================================================
def cmd_cleanup(args):
    files = [act_path(l) for l in range(N_LAYERS) if act_path(l).exists()]
    total = sum(p.stat().st_size for p in files) / 1e9
    need = [OUT / f"layer_{l:02d}/sae_final.pt" for l in range(N_LAYERS)] + \
           [EVAL_DIR / f"layer_{l:02d}.npz" for l in range(N_LAYERS)] + \
           [EVAL_ACT_DIR / f"layer_{l:02d}.npy" for l in range(N_LAYERS)] + \
           [CODES_DIR / f"layer_{l:02d}_topk.npz" for l in range(N_LAYERS)] + [OUT / "hooks_check.json"]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        raise SystemExit(f"not deleting: missing outputs {missing[:3]}...")
    rec = {"activation_files": [str(p) for p in files], "total_GB": round(total, 2),
           "threshold_GB": DELETE_THRESHOLD_GB}
    if total > DELETE_THRESHOLD_GB and args.yes:
        for p in files:
            p.unlink()
            ad = p.parent / ("._" + p.name)
            if ad.exists():
                ad.unlink()
        rec["deleted"] = True
        rec["note"] = ("full per-layer training activations deleted (total > 20 GB). Kept: held-out rows "
                       "(activations_eval/), top-32 v3 SAE codes for all positions (codes/), tokens (cells.npz), "
                       "data sha256 per layer (eval/meta.json).")
    else:
        rec["deleted"] = False
    update_run_config("cleanup", rec)
    log(f"cleanup: {rec}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "extract", "train", "evaluate", "hooks_check", "summarize",
                                    "cleanup"])
    ap.add_argument("--max-minutes", type=float, default=7.5)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()
    global CACHE_DIR
    CACHE_DIR = Path(args.cache_dir) if args.cache_dir else None
    {"prepare": cmd_prepare, "extract": cmd_extract, "train": cmd_train, "evaluate": cmd_evaluate,
     "hooks_check": cmd_hooks_check, "summarize": cmd_summarize, "cleanup": cmd_cleanup}[args.cmd](args)


if __name__ == "__main__":
    main()
