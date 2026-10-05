"""V3-7 step 1: MaxToki-217M anchor centroids with the CORRECT input encoding (MPS, resumable).

Same cells, anchors, layers and pooling as the deployed phase1bc_hidden_states_and_centroids.py. The only change is
the model input: genes are ranked by raw/X integer counts / Geneformer gene median (setup/inputs_v3.py), not by the
stored X (= log1p(CP10k) of decontX counts) / median.

Usage (run from anywhere; use the project venv python):
  v3_devorder_centroids.py plan                         CPU. Cell lists (phase1bc subsample replayed), obs-label check,
                                                        assay, deployed-token rebuild check. Writes cells/plan_<panel>.csv
  v3_devorder_centroids.py extract --max-minutes 7.5    MPS. Resumable in chunks of 50 cells; repeat until it prints
                                                        ALL_DONE True. Stops cleanly if free memory < 3 GB.
  v3_devorder_centroids.py status
  v3_devorder_centroids.py finalize                     CPU. Centroids, anchor_meta, rulers, operators, run_config.json

Per cell (exactly as phase1bc lines 160-176): one forward pass of the cell alone (batch 1, attention mask of ones,
float32, eager attention), all 12 hidden states (index 0 = embedding output, 1..10 = block outputs, 11 = last block
after the final RMSNorm), mean over gene-token positions 1..L-2 (BOS and EOS excluded). The base LlamaModel is
called (no lm_head); its hidden states are identical to LlamaForCausalLM's (checked: max difference 0.0).
Centroid = float32 mean of the per-cell vectors of the anchor's cells.

Encoding check (rule 6): every cell is tokenised with inputs_v3.tokenize_counts(check=True) and every chunk passes
inputs_v3.assert_encoding_batch on ALL its cells before any forward pass; failure raises EncodingError (abort).
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
import argparse
import json
import math
import platform
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "<REPO_ROOT>/projects/maxtoki/runs/manifold-discovery-217M/scripts")
from v3_devorder_common import *  # noqa: E402,F401

import inputs_v3 as I  # noqa: E402
import hooks_v2 as H   # noqa: E402

CHUNK = 50
MIN_FREE_GB = 3.0
MPS_CACHE_LIMIT = 4.0e9
PROC_LIMIT_GB = 7.5          # RSS + MPS driver peak of this process; stop the call above this
PAD_TO = 256                 # length bucketing for the forward pass (see pooled_states)
SCRIPT = Path(__file__).resolve()


def rss_now_gb() -> float:
    import subprocess
    try:
        return int(subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True,
                                  timeout=10).stdout.strip()) * 1024 / 1e9
    except Exception:
        return float("nan")


def chunk_path(panel, k):
    return PERCELL / panel / f"chunk_{k:04d}.npz"


def plan_path(panel):
    return CELLS / f"plan_{panel}.csv"


# =============================================================================================================
# plan
# =============================================================================================================
def read_obs_index(ds):
    import h5py
    with h5py.File(I.DATASETS[ds]["path"], "r") as f:
        k = f["obs"].attrs.get("_index", "_index")
        k = k.decode() if isinstance(k, bytes) else k
        return np.array([s.decode() if isinstance(s, bytes) else s for s in f["obs"][k][:]], dtype=object)


def stage_plan(args):
    t0 = time.time()
    out = {"written_at": time.strftime("%Y-%m-%d %H:%M:%S"), "panels": {}}
    obs_idx = {}
    for panel in PANELS_ALL:
        ds = DATASET[panel]
        keep, sub = selected_cells(panel)            # raises if the replay differs from the deployed subsample
        if ds not in obs_idx:
            obs_idx[ds] = read_obs_index(ds)
        rows = sub["cell_idx"].to_numpy().astype(np.int64)
        lab = obs_idx[ds][rows]
        n_lab_mismatch = int(np.sum(lab != sub["obs_label"].astype(str).to_numpy()))
        if n_lab_mismatch:
            raise RuntimeError(f"{panel}: {n_lab_mismatch} obs labels differ at cell_idx rows")
        assay = I.read_obs_column(ds, "assay", rows)
        dz = np.load(PH1 / f"cells_{panel}.npz")
        dep_len = dz["seq_lens"][keep]
        # the deployed tokens are tokenize_cell(stored X row): rebuild the first 5 selected cells (row mapping check)
        dep_rebuild = I.deployed_tokenize_X(ds, rows[:5], MAX_LEN)
        n_same = 0
        for j, c in enumerate(dep_rebuild):
            L = int(dep_len[j])
            n_same += int(c is not None and np.array_equal(c.token_ids, dz["token_ids"][keep[j], :L].astype(np.int64)))
        if n_same != 5:
            raise RuntimeError(f"{panel}: deployed tokens not rebuilt from X at the planned rows ({n_same}/5)")
        plan = sub.copy()
        plan.insert(0, "sel_i", np.arange(len(sub)))
        plan.insert(1, "cells_obs_pos", keep)
        plan["assay"] = assay
        plan["deployed_seq_len"] = dep_len
        plan.to_csv(plan_path(panel), index=False)
        out["panels"][panel] = {
            "dataset": ds, "n_cells": int(len(plan)), "n_anchors": int(plan["anchor_id"].nunique()),
            "n_chunks": int(math.ceil(len(plan) / CHUNK)), "obs_label_mismatch": n_lab_mismatch,
            "deployed_tokens_rebuilt_from_X_first5": n_same,
            "assay_counts": {str(k): int(v) for k, v in pd.Series(assay).value_counts().items()},
            "anchors_with_any_smartseq": int(plan.groupby("anchor_id")["assay"].apply(
                lambda s: s.astype(str).str.startswith("Smart").any()).sum()),
            "deployed_seq_len_median": float(np.median(dep_len))}
        print(panel, out["panels"][panel], flush=True)
    out["seconds"] = time.time() - t0
    write_json(CELLS / "plan.json", out)
    print("plan done", f"{time.time()-t0:.0f}s")


def load_plans():
    return {p: pd.read_csv(plan_path(p)) for p in PANELS_ALL}


# =============================================================================================================
# extract
# =============================================================================================================
def load_base_model(device):
    import torch
    m = H.load_model(device)                    # MaxTokiAttentionExtractor, eager attention, float32
    model = m.model
    model.eval()
    return model, model.model, {"attn_implementation": model.config._attn_implementation,
                                "dtype": str(next(model.parameters()).dtype)}


def pooled_states(base, token_ids, device, pad_to: int | None = PAD_TO):
    """phase1bc lines 160-176 for one cell; returns (12, 1232) float32.
    pad_to: right-pad the sequence with <pad> (attention mask 0) up to a multiple of pad_to (max 4,096). In a causal
    model the real positions never see later positions, so their hidden states are unchanged (checked: max abs
    difference 3.8e-6, float32 noise); it only stops MPS from keeping compiled kernels for every new length, which
    made the process grow by ~6 MB per cell. Chunks written before this option existed have no padding."""
    import torch
    L = len(token_ids)
    if L < 3:
        return np.zeros((N_STATES, HIDDEN), dtype=np.float32)
    T = L if not pad_to else min(MAX_LEN, int(math.ceil(L / pad_to) * pad_to))
    ids = torch.zeros((1, T), dtype=torch.long)
    ids[0, :L] = torch.from_numpy(np.asarray(token_ids, dtype=np.int64))
    mask = torch.zeros((1, T), dtype=torch.long)
    mask[0, :L] = 1
    ids = ids.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        out = base(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False, return_dict=True)
        hs = torch.stack([h.squeeze(0) for h in out.hidden_states], dim=0)
        if hs.shape[0] != N_STATES:
            raise RuntimeError(f"expected {N_STATES} hidden states, got {hs.shape[0]}")
        pooled = hs[:, 1:L - 1, :].mean(dim=1) if L >= 4 else hs[:, 1:L, :].mean(dim=1)
        r = pooled.cpu().numpy().astype(np.float32)
    del out, hs, pooled, ids, mask
    return r


def save_npz_atomic(path: Path, **arrs):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrs)
    os.replace(tmp, path)


def stage_extract(args):
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    t_call = time.time()
    budget = float(args.max_minutes) * 60.0
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device != "mps":
        raise RuntimeError("MPS not available")
    plans = load_plans()
    panels = args.panels.split(",") if args.panels else PANELS_ALL
    model = base = None
    minfo = None
    stopped = "all_done"
    n_done_call = 0
    est_chunk = 60.0
    last_mps_peak = 0.0
    for panel in panels:
        P = plans[panel]
        n = len(P)
        nch = math.ceil(n / CHUNK)
        pending = [k for k in range(nch) if not chunk_path(panel, k).exists()]
        if not pending:
            continue
        ds = DATASET[panel]
        vm = I.get_var_map(ds)
        dz = np.load(PH1 / f"cells_{panel}.npz")
        dep_tok, dep_len = dz["token_ids"], dz["seq_lens"]
        with I.CountReader(ds) as reader:
            for k in pending:
                el = time.time() - t_call
                if el + est_chunk * 1.15 > budget:
                    stopped = "time"
                    break
                try:
                    free_gb = H.check_memory(MIN_FREE_GB)
                except H.MemoryGuardError as e:
                    print(f"[extract] {panel} chunk {k}: {e}", flush=True)
                    stopped = "memory"
                    break
                if model is not None:
                    proc_gb = rss_now_gb() + max(float(torch.mps.driver_allocated_memory()), last_mps_peak) / 1e9
                    if proc_gb > PROC_LIMIT_GB:
                        # MPS keeps compiled kernels for every new sequence length (~5 MB each, CPU side), so the
                        # process grows during a call; stop and let the next call start a fresh process.
                        print(f"[extract] process memory {proc_gb:.2f} GB > {PROC_LIMIT_GB} GB; stopping this call",
                              flush=True)
                        stopped = "process_memory"
                        break
                if model is None:
                    model, base, minfo = load_base_model(device)
                tc = time.time()
                idx = list(range(k * CHUNK, min(n, (k + 1) * CHUNK)))
                rows = P["cell_idx"].to_numpy()[idx].astype(np.int64)
                # ---- read counts, tokenise (each cell checked), check the whole batch before any forward pass ----
                counts, cells, chk_rows = [], [], []
                for r in rows:
                    c, chk = reader.counts(int(r), return_check=True)
                    cell = I.tokenize_counts(c, vm, MAX_LEN, check=True)
                    if cell is None:
                        raise I.EncodingError(f"{panel} row {r}: no expressed vocab gene")
                    counts.append(c)
                    cells.append(cell)
                    chk_rows.append(chk)
                enc = I.assert_encoding_batch(counts, cells, vm, MAX_LEN, label=f"{panel} chunk {k}")
                t_tok = time.time() - tc
                # ---- forward passes ----
                states = np.zeros((len(idx), N_STATES, HIDDEN), dtype=np.float32)
                sec = np.zeros(len(idx))
                mps_peak = 0.0
                for j, cell in enumerate(cells):
                    tf = time.time()
                    states[j] = pooled_states(base, cell.token_ids, device)
                    sec[j] = time.time() - tf
                    drv = float(torch.mps.driver_allocated_memory())
                    mps_peak = max(mps_peak, drv)
                    if drv > MPS_CACHE_LIMIT:
                        torch.mps.empty_cache()
                if not np.all(np.isfinite(states)):
                    raise RuntimeError(f"{panel} chunk {k}: non-finite hidden states")
                # ---- how different is the input from the deployed one (cut sequences, both 4,096 tokens) ----
                agree = []
                for j, cell in enumerate(cells):
                    p = int(P["cells_obs_pos"].iloc[idx[j]])
                    dg = dep_tok[p, 1:int(dep_len[p]) - 1].astype(np.int64)
                    vg = cell.token_ids[1:-1]
                    a = I.order_agreement(dg, vg, MAX_LEN)
                    agree.append([a["spearman_full"], a["top200_overlap"], a["kept_set_overlap"],
                                  a["same_position_share"], len(vg), len(dg)])
                agree = np.array(agree, dtype=np.float64)
                toks = [c.token_ids.astype(np.int32) for c in cells]
                offs = np.concatenate([[0], np.cumsum([len(t) for t in toks])]).astype(np.int64)
                rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9   # peak, bytes on macOS
                rss_now = rss_now_gb()
                save_npz_atomic(
                    chunk_path(panel, k), states=states, sel_i=np.array(idx, dtype=np.int64), rows=rows,
                    tokens=np.concatenate(toks), offsets=offs, sec_forward=sec,
                    agree=agree,   # spearman (common genes), top-200 overlap, kept-set overlap, same position, len v3, len deployed
                    count_total=np.array([float(c.sum()) for c in counts]),
                    meta=np.array(json.dumps({
                        "panel": panel, "chunk": k, "dataset": ds, "n_cells": len(idx),
                        "seconds_chunk": time.time() - tc, "seconds_tokenise_and_check": t_tok,
                        "free_gb_before": free_gb, "mps_driver_peak_gb": mps_peak / 1e9, "rss_peak_gb": rss_gb,
                        "rss_now_gb": rss_now,
                        "encoding_check": enc, "count_checks_all_expected": all(c["expected_ok"] for c in chk_rows),
                        "count_kind": sorted({c["kind"] for c in chk_rows}), "model": minfo, "device": device,
                        "pad_to": PAD_TO,
                        "torch_threads": torch.get_num_threads(), "written_at": time.strftime("%Y-%m-%d %H:%M:%S")})))
                n_done_call += 1
                last_mps_peak = mps_peak
                est_chunk = max(20.0, time.time() - tc)
                print(f"[extract] {panel} chunk {k + 1}/{nch} cells {idx[0]}-{idx[-1]}  {time.time() - tc:.0f}s "
                      f"(tok+check {t_tok:.1f}s, fwd {sec.sum():.0f}s)  free {free_gb:.1f} GB  mps_peak {mps_peak / 1e9:.2f} GB "
                      f"rss peak/now {rss_gb:.2f}/{rss_now:.2f} GB  top200 {np.nanmean(agree[:, 1]):.3f}", flush=True)
        del dep_tok, dz
        if stopped != "all_done":
            break
    st = status_dict()
    print(f"[extract] this call: {n_done_call} chunks in {time.time() - t_call:.0f}s; stopped: {stopped}")
    for p, v in st["panels"].items():
        print(f"  {p}: {v['done']}/{v['n_chunks']} chunks")
    print("ALL_DONE", st["all_done"])


def status_dict():
    plans = {p: pd.read_csv(plan_path(p)) for p in PANELS_ALL}
    out = {"panels": {}}
    left_cells = 0
    secs = []
    for p in PANELS_ALL:
        n = len(plans[p]); nch = math.ceil(n / CHUNK)
        done = [k for k in range(nch) if chunk_path(p, k).exists()]
        out["panels"][p] = {"n_chunks": nch, "done": len(done), "n_cells": n}
        left_cells += n - sum(min(CHUNK, n - k * CHUNK) for k in done)
    out["all_done"] = all(v["done"] == v["n_chunks"] for v in out["panels"].values())
    out["cells_left"] = left_cells
    return out


def stage_status(args):
    st = status_dict()
    print(json.dumps(st, indent=1))


# =============================================================================================================
# finalize
# =============================================================================================================
def load_chunks(panel, n):
    nch = math.ceil(n / CHUNK)
    states = np.zeros((n, N_STATES, HIDDEN), dtype=np.float32)
    toks, lens, agree, sec, metas, totals = [None] * n, np.zeros(n, np.int64), np.zeros((n, 6)), np.zeros(n), [], np.zeros(n)
    for k in range(nch):
        z = np.load(chunk_path(panel, k))
        si = z["sel_i"]
        states[si] = z["states"]
        o = z["offsets"]
        for j, s in enumerate(si):
            toks[s] = z["tokens"][o[j]:o[j + 1]]
            lens[s] = o[j + 1] - o[j]
        agree[si] = z["agree"]
        sec[si] = z["sec_forward"]
        totals[si] = z["count_total"]
        metas.append(json.loads(str(z["meta"])))
    return states, toks, lens, agree, sec, totals, metas


def stage_finalize(args):
    import shutil
    from collections import deque
    t0 = time.time()
    st = status_dict()
    if not st["all_done"]:
        raise RuntimeError(f"extraction not finished: {st}")
    plans = load_plans()
    rep = {"panels": {}}
    chunk_walls = {}
    for panel in PANELS_ALL:
        P = plans[panel]
        n = len(P)
        states, toks, lens, agree, sec, totals, metas = load_chunks(panel, n)
        chunk_walls[panel] = [round(m["seconds_chunk"], 1) for m in metas]
        anchors = pd.read_csv(PH1 / f"anchors_{panel}.csv")
        anchor_ids = anchors["anchor_id"].astype(str).to_numpy()
        a_col = P["anchor_id"].astype(str).to_numpy()
        cen = np.zeros((len(anchor_ids), N_STATES, HIDDEN), dtype=np.float32)
        npa = np.zeros(len(anchor_ids), dtype=np.int32)
        for ai, aid in enumerate(anchor_ids):           # phase1bc lines 183-193
            mask = a_col == aid
            npa[ai] = int(mask.sum())
            if npa[ai]:
                cen[ai] = states[mask].mean(axis=0)
        np.save(V3ANCH / f"centroids_{panel}.npy", cen)
        meta = anchors.copy()
        meta["n_cells_centroided"] = npa
        dep_meta = pd.read_csv(DEP_ART / f"anchors/anchor_meta_{panel}.csv")
        if not meta.astype(str).equals(dep_meta.astype(str)):
            raise RuntimeError(f"{panel}: anchor_meta differs from the deployed one")
        meta.to_csv(V3ANCH / f"anchor_meta_{panel}.csv", index=False)
        # padded token array of the selected cells (for the token-bag feature)
        T = np.zeros((n, MAX_LEN), dtype=np.int32)
        for i, t in enumerate(toks):
            T[i, :len(t)] = t
        np.savez_compressed(CELLS / f"tokens_{panel}.npz", token_ids=T, seq_lens=lens.astype(np.int32),
                            rows=P["cell_idx"].to_numpy().astype(np.int64), sel_i=np.arange(n))
        # rulers depend only on labels: copy the deployed files after checking them against a rebuild
        dep_cen = np.load(DEP_ART / f"anchors/centroids_{panel}.npy")
        info = {"n_cells": n, "n_anchors": int(len(anchor_ids)),
                "cells_per_anchor": [int(npa.min()), int(np.median(npa)), int(npa.max())],
                "seq_len_v3_median": float(np.median(lens)), "seq_len_deployed_median": float(np.median(agree[:, 5] + 2)),
                "share_cut_at_4094_v3": float(np.mean(agree[:, 4] >= MAX_LEN - 2)),
                "share_cut_at_4094_deployed": float(np.mean(agree[:, 5] >= MAX_LEN - 2)),
                "order_vs_deployed_mean": {"spearman_common_genes": float(np.nanmean(agree[:, 0])),
                                           "top200_overlap": float(np.nanmean(agree[:, 1])),
                                           "kept_set_overlap": float(np.nanmean(agree[:, 2])),
                                           "same_position_share": float(np.nanmean(agree[:, 3]))},
                "forward_seconds_mean": float(sec.mean()), "forward_seconds_total": float(sec.sum()),
                "encoding_checks": {"chunks": len(metas), "all_pass": all(m["encoding_check"]["all_pass"] for m in metas),
                                    "cells_checked": int(sum(m["encoding_check"]["n_checked"] for m in metas)),
                                    "max_order_violations": int(max(m["encoding_check"]["max_order_violations"] for m in metas)),
                                    "count_checks_all_expected": all(m["count_checks_all_expected"] for m in metas),
                                    "count_kinds": sorted({k for m in metas for k in m["count_kind"]})},
                "mps_driver_peak_gb_max": float(max(m["mps_driver_peak_gb"] for m in metas)),
                "rss_peak_gb_max": float(max(m["rss_peak_gb"] for m in metas)),
                "free_gb_before_min": float(min(m["free_gb_before"] for m in metas))}
        # centroid change vs deployed, per hidden state
        cs, rel, dc = [], [], []
        for s in range(N_STATES):
            a = cen[:, s, :].astype(np.float64); b = dep_cen[:, s, :].astype(np.float64)
            cs.append(float(np.mean(np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12))))
            rel.append(float(np.mean(np.linalg.norm(a - b, axis=1) / (np.linalg.norm(b, axis=1) + 1e-12))))
            from scipy.stats import spearmanr
            from scipy.spatial.distance import pdist
            dc.append(float(spearmanr(pdist(a), pdist(b))[0]))
        info["v3_vs_deployed_centroids"] = {"mean_cosine_per_state": cs, "mean_relative_change_per_state": rel,
                                            "anchor_distance_spearman_per_state": dc}
        rep["panels"][panel] = info
        print(panel, json.dumps({k: v for k, v in info.items() if k != "v3_vs_deployed_centroids"}), flush=True)
    # rulers
    import importlib
    dag = json.loads((RUN / "planning/h65_stage_dag.json").read_text())
    nodes = list(dag["stage_to_branch"].keys())
    adj = {n_: set() for n_ in nodes}
    for a, b in dag["edges"]:
        adj[a].add(b); adj[b].add(a)

    def build_ruler(stage_labels):            # phase1bc build_ruler (same rule: shortest path, 99 if disconnected)
        cache = {}
        n_ = len(stage_labels)
        D = np.zeros((n_, n_), dtype=np.float32)
        for i, si in enumerate(stage_labels):
            if si not in cache:
                dist = {si: 0}; q = deque([si])
                while q:
                    u = q.popleft()
                    for v in adj.get(u, ()):
                        if v not in dist:
                            dist[v] = dist[u] + 1; q.append(v)
                cache[si] = dist
            for j in range(i + 1, n_):
                sj = stage_labels[j]
                D[i, j] = D[j, i] = 99 if sj not in cache[si] else float(cache[si][sj])
        return D
    s2b = {k: v["branch"] for k, v in dag["stage_to_branch"].items()}
    ruler_check = {}
    for panel in ["internal", "external", "zeroshot", "lung_nonhema"]:
        m = pd.read_csv(V3ANCH / f"anchor_meta_{panel}.csv")
        lab = m["hema_stage"].astype(str).tolist()
        D = build_ruler(lab)
        rng = np.random.default_rng(42)                         # phase1bc lines 203-221
        la = np.array(lab); sh = la.copy()
        bpa = np.array([s2b.get(s, "_unknown") for s in lab])
        for br in np.unique(bpa):
            ii = np.where(bpa == br)[0]
            sh[ii] = rng.permutation(la[ii])
        Dn = build_ruler(sh.tolist())
        dep = np.load(DEP_ART / f"anchors/d_target_{panel}.npy")
        depn = np.load(DEP_ART / f"anchors/d_target_{panel}_null_shuffled.npy")
        ok = bool(np.array_equal(D, dep)) and bool(np.array_equal(Dn, depn))
        ruler_check[panel] = ok
        if not ok:
            raise RuntimeError(f"{panel}: rebuilt ruler differs from deployed")
        np.save(V3ANCH / f"d_target_{panel}.npy", D)
        np.save(V3ANCH / f"d_target_{panel}_null_shuffled.npy", Dn)
    # operators read by the v2b code (weights only; not affected by the input encoding): copy and check
    ops = ["pooled_drift_components.npz", "operator_index.json", "layer10_head6.npy"]
    op_sha = {}
    for f in ops:
        shutil.copyfile(DEP_ART / "operators" / f, V3OPS / f)
        a, b = sha256_file(DEP_ART / "operators" / f), sha256_file(V3OPS / f)
        if a != b:
            raise RuntimeError(f"operator copy {f} differs")
        op_sha[f] = a
    rep["rulers_equal_deployed_rebuild"] = ruler_check
    rep["operators_copied_sha256"] = op_sha
    rep["chunk_wall_seconds"] = chunk_walls
    write_json(V3 / "v3_centroids_summary.json", rep)
    write_run_config_centroids(rep)
    print("finalize done", f"{time.time() - t0:.0f}s")


def write_run_config_centroids(rep):
    import torch
    import transformers
    import sklearn, scipy, h5py
    plans = load_plans()
    inputs = []
    for p in PANELS_ALL:
        inputs += [PH1 / f"cells_{p}_obs.csv", PH1 / f"anchors_{p}.csv", PH1 / f"cells_{p}.npz",
                   DEP_ART / f"anchors/anchor_meta_{p}.csv", DEP_ART / f"anchors/centroids_{p}.npy"]
    inputs += [RUN / "planning/h65_stage_dag.json", MODEL_DIR / "config.json", SETUP / "token_dictionary.json"]
    inputs += sorted(MODEL_DIR.glob("*.safetensors"))
    outputs = sorted(V3ANCH.glob("*.npy")) + sorted(V3ANCH.glob("*.csv")) + sorted(V3OPS.glob("*")) + \
        sorted(CELLS.glob("*")) + [V3 / "v3_centroids_summary.json"]
    enc = {}
    for p in PANELS_ALL:
        ds = DATASET[p]
        rows = plans[p]["cell_idx"].to_numpy().astype(np.int64)
        enc[p] = I.encoding_record(ds, rows=rows, check_summary=rep["panels"][p]["encoding_checks"])
    cfg = {"item": "V3-7 manifold hidden-state centroids with the correct input encoding",
           "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "env": H.env_info("mps"), "python": platform.python_version(),
           "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__,
                        "pandas": pd.__version__, "sklearn": sklearn.__version__, "scipy": scipy.__version__,
                        "h5py": h5py.__version__},
           "device": "mps", "dtype": "float32", "attn_implementation": "eager", "batch": 1,
           "torch_threads": int(os.environ.get("OMP_NUM_THREADS", "4")),
           "seeds": {"cell subsample": "default_rng(42), phase1bc lines 124-144 replayed",
                     "null ruler": "default_rng(42) within-branch shuffle, phase1bc lines 203-221"},
           "pooling": "mean over positions 1..L-2 of each of the 12 hidden states (phase1bc lines 170-174)",
           "layers": "hidden_states 0..11 of LlamaModel (0 = embeddings, 11 = after final RMSNorm)",
           "cells": {p: {"rows": plans[p]["cell_idx"].astype(int).tolist(),
                         "obs_label": plans[p]["obs_label"].astype(str).tolist()} for p in PANELS_ALL},
           "input_encoding": enc,
           "inputs_sha256": {str(p): sha256_file(p) for p in inputs},
           "code_sha256": {str(p): sha256_file(p) for p in [SCRIPT, SCR / "v3_devorder_common.py",
                                                            SETUP / "inputs_v3.py", SETUP / "hooks_v2.py",
                                                            SETUP / "maxtoki_adapter.py"]},
           "outputs_sha256": {str(p): sha256_file(p) for p in outputs if p.is_file() and not p.name.startswith("._")},
           "wall_seconds_per_chunk": rep["chunk_wall_seconds"]}
    write_json(V3 / "run_config_centroids.json", cfg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["plan", "extract", "status", "finalize"])
    ap.add_argument("--max-minutes", type=float, default=7.5)
    ap.add_argument("--panels", default="")
    a = ap.parse_args()
    {"plan": stage_plan, "extract": stage_extract, "status": stage_status, "finalize": stage_finalize}[a.stage](a)
