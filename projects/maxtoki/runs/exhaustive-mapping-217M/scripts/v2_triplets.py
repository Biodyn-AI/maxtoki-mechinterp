"""v2 triplets (revision item D5): three-way SAE-feature ablation, redone with fixed hooks.

What the deployed Experiment 2 did wrong (see verification/triplets.md):
  * only 4 triplets were ablated; "2,980" was a count of downstream L11 targets, not triplets;
  * its hooks deleted a whole block (block offset) and the deepest hook erased the others
    (AB == B, ABC == C), so "zero synergy" and the 0.190 ratio were fixed by arithmetic.

This script:
  * uses setup/hooks_v2.py (forward pre-hooks at the SAE site, SAE code from the LIVE tensor,
    edits stack);
  * runs a full 8 x 8 x 8 grid of (L0, L5, L9) SAE features = 512 triplets, with the
    24 single and 192 pair conditions shared (728 ablation conditions per cell);
  * adds noise-floor triplets built with features that are never active in these cells
    (8 all-null triplets + 24 mixed triplets with one null feature);
  * uses 20 K562 non-targeting control cells (same pool and seed as experiment2_rerun.py);
  * reads out the L11 SAE code (per-cell mean over tokens, as deployed) and a logit summary.

Speed-up (checked numerically by the `verify` stage): conditions that do not edit L0 start
the forward pass from the cached live input of block 5 or block 9 instead of from the
embeddings. The edits themselves are always hooks_v2 pre-hooks.

Stages (run in this order):
  select   : clean pass on the 20 cells, activation frequencies, feature choice (seeded).
  verify   : zero-delta check and fast-path == full-path check on 2 cells.
  run      : the ablation grid; resumable; call repeatedly with --max-minutes 8 until DONE.
  analyze  : Moebius interaction terms, bootstrap CIs, BH, redundancy ratios, sanity checks.

Usage:
  PY=projects/maxtoki/.venv/bin/python
  $PY runs/exhaustive-mapping-217M/scripts/v2_triplets.py select
  $PY runs/exhaustive-mapping-217M/scripts/v2_triplets.py verify
  $PY runs/exhaustive-mapping-217M/scripts/v2_triplets.py run --max-minutes 8   (repeat)
  $PY runs/exhaustive-mapping-217M/scripts/v2_triplets.py analyze
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJ / "setup"))
import hooks_v2 as H  # noqa: E402

RUN = PROJ / "runs/exhaustive-mapping-217M"
OUT = RUN / "outputs/v2_triplets"
CELL_DIR = OUT / "cells"
ANN_DIR = PROJ / "runs/sae-atlas-217M/outputs/phase2"
SCRIPT_PATH = Path(__file__).resolve()

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SRC_LAYERS = (0, 5, 9)
TGT = 11
N_CELLS = 20
CELL_POOL_SEED = 42          # same pool as experiment2_rerun.py (seed 42, 100-cell pool)
SEL_SEED = 20261001          # feature selection (rng seeded with SEL_SEED + layer)
BOOT_SEED = 20261001
N_BOOT = 1000
N_PER_LAYER = 8              # 4 annotated + 4 unannotated
N_NULL = 2                   # never-active features per layer (noise floor)
MIN_FREQ = 0.01              # every chosen feature active in >= 1% of tokens of the 20 cells
MIN_FREE_GB = 3.0
NONE = -1


# =============================================================================
# small helpers
# =============================================================================
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_json(p):
    with open(p) as fh:
        return json.load(fh)


def update_run_config(section: str, entry):
    """Append ``entry`` to run_config[section] (list) or set it (dict)."""
    p = OUT / "run_config.json"
    cfg = load_json(p) if p.exists() else {}
    if isinstance(entry, dict) and section in ("env", "selection", "cells", "seeds", "verify"):
        cfg[section] = entry
    else:
        if isinstance(entry, dict):
            entry = dict(entry, script_sha256=H.sha256_file(SCRIPT_PATH))
        cfg.setdefault(section, []).append(entry)
    cfg["script"] = str(SCRIPT_PATH)
    cfg["script_sha256"] = H.sha256_file(SCRIPT_PATH)
    H.write_json(p, cfg)


def load_cells_cached():
    p = OUT / "cells_tokens.npz"
    d = np.load(p, allow_pickle=False)
    meta = load_json(OUT / "cells.json")
    toks = [d[f"cell_{i:02d}"] for i in range(len(meta["rows"]))]
    return toks, meta


def free():
    H.free_device_cache()


# =============================================================================
# forward machinery
# =============================================================================
class Runner:
    """Holds the model, SAEs, and the per-cell clean caches."""

    def __init__(self, device=DEVICE):
        self.device = device
        log(f"loading model on {device}")
        self.xt = H.load_model(device)
        self.model = self.xt.model
        self.nb = H.n_blocks(self.model)
        self.saes = H.load_saes(list(SRC_LAYERS) + [TGT], device=device)
        self.layer_kwargs = None

    # -- kwargs of the decoder layers (mask, rotary embeddings) ----------------
    def _kw_hook(self):
        store = {}

        def pre(mod, args, kwargs):
            store["kw"] = dict(kwargs)
            return None
        h = self.model.model.layers[0].register_forward_pre_hook(pre, with_kwargs=True)
        return store, h

    @torch.no_grad()
    def clean(self, ids):
        """Clean full pass. Captures live inputs of sites 0, 5, 9, 11 on device."""
        store, h = self._kw_hook()
        try:
            with H.ResidualEditor(self.model, self.saes, edits=[], capture=[0, 5, 9, TGT],
                                  capture_device=self.device) as ed:
                out = self.model(ids, use_cache=False, return_dict=True)
        finally:
            h.remove()
        self.layer_kwargs = store["kw"]
        logits = out.logits[0].float()
        del out
        c = {
            "h0": ed.captured[0]["pre"], "h5": ed.captured[5]["pre"], "h9": ed.captured[9]["pre"],
            "h11": ed.captured[TGT]["pre"],
        }
        c["z11"] = self.saes[TGT].encode(c["h11"][0])                 # (T, d_sae) device
        c["logits"] = logits                                           # (T, V) device
        c["logp"] = torch.log_softmax(logits, dim=-1)
        c["p"] = c["logp"].exp()
        return c

    @torch.no_grad()
    def forward_from(self, start, h_in, ids):
        """Run blocks start..10, the final norm and lm_head. start == 0 uses the full model."""
        if start == 0:
            out = self.model(ids, use_cache=False, return_dict=True)
            lg = out.logits
            del out
            return lg
        h = h_in
        for l in range(start, self.nb):
            h = self.model.model.layers[l](h, **self.layer_kwargs)
        h = self.model.model.norm(h)
        return self.model.lm_head(h)

    @torch.no_grad()
    def run_condition(self, ids, start, h_in, edits, capture, clean):
        """One ablation condition. Returns (measurement dict, editor)."""
        with H.ResidualEditor(self.model, self.saes, edits=edits, capture=capture,
                              capture_device=self.device) as ed:
            lg = self.forward_from(start, h_in, ids)
        meas = self.measure(lg[0].float(), ed.captured[TGT]["pre"][0], clean)
        n_act = {}
        for l, info in ed.edit_log.items():
            for f, n in info["n_active"].items():
                n_act[(l, f)] = n
        meas["n_active"] = n_act
        del lg
        return meas, ed

    @torch.no_grad()
    def measure(self, logits, h11, clean):
        T = logits.shape[0]
        z = self.saes[TGT].encode(h11)
        dz = (z - clean["z11"]).sum(0) / T                              # (d_sae,)
        dl = logits - clean["logits"]
        dlog = dl.sum(0) / T                                             # (V,)
        mabs = dl.abs().mean()
        logp = torch.log_softmax(logits, dim=-1)
        kl = (clean["p"] * (clean["logp"] - logp)).sum(-1).mean()
        res = {
            "dz": dz.to("cpu", torch.float32).numpy(),
            "dlog": dlog.to("cpu", torch.float32).numpy(),
            "mabs": float(mabs.item()),
            "kl": float(kl.item()),
        }
        del z, dl, logp
        return res


def ablate_edits(a, b, c, start):
    """hooks_v2 edits for condition (a, b, c); only layers >= start (earlier ones are cached)."""
    edits = []
    if a != NONE and start <= 0:
        edits.append(H.Ablate(0, [int(a)]))
    if b != NONE and start <= 5:
        edits.append(H.Ablate(5, [int(b)]))
    if c != NONE and start <= 9:
        edits.append(H.Ablate(9, [int(c)]))
    return edits


# =============================================================================
# design
# =============================================================================
def build_design(sel):
    """All conditions (a, b, c) with NONE = -1, and all triplets."""
    A, B, C = sel["features"]["0"], sel["features"]["5"], sel["features"]["9"]
    NA, NB, NC = sel["null_features"]["0"], sel["null_features"]["5"], sel["null_features"]["9"]
    triplets = []      # dicts: a, b, c, kind
    for a, b, c in itertools.product(A, B, C):
        triplets.append({"a": a, "b": b, "c": c, "kind": "real"})
    for a, b, c in itertools.product(NA, NB, NC):
        triplets.append({"a": a, "b": b, "c": c, "kind": "null_all"})
    for i in range(N_PER_LAYER):
        triplets.append({"a": A[i], "b": B[i], "c": NC[0], "kind": "null_C"})
        triplets.append({"a": A[i], "b": NB[0], "c": C[i], "kind": "null_B"})
        triplets.append({"a": NA[0], "b": B[i], "c": C[i], "kind": "null_A"})
    conds = set()
    for t in triplets:
        a, b, c = t["a"], t["b"], t["c"]
        for keep in itertools.product([0, 1], repeat=3):
            if sum(keep) == 0:
                continue
            conds.add((a if keep[0] else NONE, b if keep[1] else NONE, c if keep[2] else NONE))
    # prefixes needed by the fast path: (a, b, NONE) for every (a, b, c) with b != NONE
    for (a, b, c) in list(conds):
        if b != NONE:
            conds.add((a, b, NONE))
        if a != NONE:
            conds.add((a, NONE, NONE))
    conds = sorted(conds)
    a_groups = [NONE] + list(A) + list(NA)
    return triplets, conds, a_groups


# =============================================================================
# stage: select
# =============================================================================
def stage_select(args):
    OUT.mkdir(parents=True, exist_ok=True)
    CELL_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    free_gb = H.check_memory(MIN_FREE_GB)
    log(f"free memory {free_gb:.1f} GB")
    cells, rows, h5 = H.load_k562_control_cells(N_CELLS, pool=100, seed=CELL_POOL_SEED, max_len=2048)
    assert len(cells) == N_CELLS, len(cells)
    import h5py
    with h5py.File(h5, "r") as f:
        obs = f["obs"]
        barcodes = None
        for key in ("_index", "cell_barcode", "barcode", "index"):
            if key in obs:
                ds = obs[key]
                try:
                    barcodes = [ds[int(r)] for r in rows]
                    barcodes = [x.decode() if isinstance(x, bytes) else str(x) for x in barcodes]
                    break
                except Exception:
                    barcodes = None
    np.savez(OUT / "cells_tokens.npz", **{f"cell_{i:02d}": c.token_ids for i, c in enumerate(cells)})
    cell_meta = {
        "dataset_path": h5, "rows": rows, "barcodes": barcodes,
        "n_tokens": [int(len(c.token_ids)) for c in cells],
        "pool": "H.load_k562_control_cells(n_cells=20, pool=100, seed=42, max_len=2048): "
                "K562 non-targeting controls, 100-cell pool drawn with numpy default_rng(42), "
                "sorted rows, first 20 that tokenize (same pool as experiment2_rerun.py)",
    }
    H.write_json(OUT / "cells.json", cell_meta)

    R = Runner()
    d_sae = R.saes[0].W_dec.shape[1]
    counts = {l: np.zeros(d_sae, dtype=np.int64) for l in (0, 5, 9, 11)}
    bos_counts = {l: np.zeros(d_sae, dtype=np.int64) for l in (0, 5, 9)}
    total = 0
    clean_z11_mean = []
    for i, c in enumerate(cells):
        H.check_memory(MIN_FREE_GB)
        ids = torch.from_numpy(c.token_ids[None, :]).to(R.device)
        cl = R.clean(ids)
        for l, key in ((0, "h0"), (5, "h5"), (9, "h9")):
            z = R.saes[l].encode(cl[key][0])
            counts[l] += (z > 0).sum(0).cpu().numpy()
            bos_counts[l] += (z[0] > 0).cpu().numpy().astype(np.int64)
            del z
        counts[11] += (cl["z11"] > 0).sum(0).cpu().numpy()
        clean_z11_mean.append(cl["z11"].mean(0).cpu().numpy())
        total += ids.shape[1]
        del cl
        free()
        log(f"clean cell {i} ({ids.shape[1]} tokens)")
    freq = {l: counts[l] / total for l in counts}
    np.savez(OUT / "clean_frequencies.npz", total_tokens=total,
             **{f"freq_L{l}": freq[l] for l in freq},
             **{f"bos_cells_L{l}": bos_counts[l] for l in bos_counts},
             clean_z11_cellmean=np.stack(clean_z11_mean))

    import pandas as pd
    selection = {"features": {}, "null_features": {}, "details": {}, "rule": (
        "Per layer: candidates = features active (z>0) in >= 1% of all tokens of the 20 cells "
        "(clean pass, all positions incl. <bos>/<eos>). Annotated = feature has >= 1 row in "
        "sae-atlas phase2 significant_enrichments.csv. Candidate log10(frequency) split into 4 "
        "quartile bins (over annotated + unannotated candidates together); one annotated and "
        "one unannotated feature drawn at random from each bin (numpy default_rng(SEL_SEED + "
        "layer)). Order: [ann bin0, unann bin0, ann bin1, unann bin1, ...]. Null features: "
        f"{N_NULL} drawn at random from features with zero activations in the 20 cells.")}
    for l in SRC_LAYERS:
        rng = np.random.default_rng(SEL_SEED + l)
        enr = pd.read_csv(ANN_DIR / f"layer_{l:02d}/significant_enrichments.csv")
        ann = set(int(x) for x in enr["feature_id"].unique())
        catalog = {int(d["feature_id"]): d for d in load_json(ANN_DIR / f"layer_{l:02d}/feature_catalog.json")}
        fr = freq[l]
        cand = np.where(fr >= MIN_FREQ)[0]
        is_ann = np.array([int(f) in ann for f in cand])
        logf = np.log10(fr[cand])
        edges = np.quantile(logf, [0, 0.25, 0.5, 0.75, 1.0])
        bins = np.clip(np.searchsorted(edges[1:-1], logf, side="right"), 0, 3)
        chosen, det = [], []
        for bi in range(4):
            for want_ann in (True, False):
                order = sorted(range(4), key=lambda x: (abs(x - bi), x))
                pick = None
                for bj in order:
                    pool = cand[(bins == bj) & (is_ann == want_ann)]
                    pool = np.array([p for p in pool if int(p) not in chosen])
                    if len(pool):
                        pick = int(rng.choice(pool))
                        used_bin = bj
                        break
                assert pick is not None
                chosen.append(pick)
                terms = enr[enr["feature_id"] == pick].sort_values("q_bh")["term"].tolist()[:3]
                det.append({
                    "feature": pick, "annotated": want_ann, "freq_bin_target": bi, "freq_bin_used": used_bin,
                    "freq_20cells": float(fr[pick]), "freq_atlas": (float(catalog[pick]["activation_frequency"]) if pick in catalog else None),
                    "active_at_bos_cells": int(bos_counts[l][pick]), "top_terms": terms,
                    "n_terms": int((enr["feature_id"] == pick).sum()),
                })
        zero = np.where(counts[l] == 0)[0]
        nulls = [int(x) for x in rng.choice(zero, size=N_NULL, replace=False)]
        selection["features"][str(l)] = chosen
        selection["null_features"][str(l)] = nulls
        selection["details"][str(l)] = {
            "n_candidates": int(len(cand)), "n_candidates_annotated": int(is_ann.sum()),
            "n_alive_20cells": int((counts[l] > 0).sum()), "n_zero_20cells": int(len(zero)),
            "log10_freq_bin_edges": [float(x) for x in edges],
            "candidates_per_bin": {str(b): {"annotated": int(((bins == b) & is_ann).sum()),
                                            "unannotated": int(((bins == b) & ~is_ann).sum())}
                                   for b in range(4)},
            "chosen": det,
            "null": [{"feature": n, "freq_atlas": (float(catalog[n]["activation_frequency"]) if n in catalog else None),
                      "in_atlas_catalog": n in catalog} for n in nulls],
        }
        log(f"L{l}: chosen {chosen} nulls {nulls}")
    H.write_json(OUT / "selection.json", selection)
    update_run_config("env", H.env_info(DEVICE))
    update_run_config("seeds", {"cell_pool_seed": CELL_POOL_SEED, "selection_seed_base": SEL_SEED,
                                "bootstrap_seed": BOOT_SEED, "n_boot": N_BOOT,
                                "torch_manual_seed": 0})
    update_run_config("cells", cell_meta)
    update_run_config("selection", {"features": selection["features"],
                                    "null_features": selection["null_features"]})
    update_run_config("chunks", {"stage": "select", "start": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                 "wall_seconds": round(time.time() - t0, 1), "free_gb": free_gb})
    log(f"select done in {time.time() - t0:.0f}s")


# =============================================================================
# stage: verify
# =============================================================================
def stage_verify(args):
    torch.manual_seed(0)
    t0 = time.time()
    free_gb = H.check_memory(MIN_FREE_GB)
    sel = load_json(OUT / "selection.json")
    toks, meta = load_cells_cached()
    triplets, conds, a_groups = build_design(sel)
    R = Runner()
    rng = np.random.default_rng(7)
    res = {"zero_delta": [], "fast_vs_full": [], "float64_check": [], "clean_repeat": []}
    for ci in range(2):
        ids = torch.from_numpy(toks[ci][None, :]).to(R.device)
        cl = R.clean(ids)
        cl2 = R.clean(ids)
        res["clean_repeat"].append({
            "cell": ci, "max_abs_logits": float((cl["logits"] - cl2["logits"]).abs().max().item()),
            "max_abs_z11": float((cl["z11"] - cl2["z11"]).abs().max().item())})
        del cl2
        # (1) zero-delta: ZeroDelta at 0, 5, 9 (each and all) via the full model
        for layers in ([0], [5], [9], [0, 5, 9]):
            with H.ResidualEditor(R.model, R.saes, edits=[H.ZeroDelta(l) for l in layers],
                                  capture=[TGT], capture_device=R.device) as ed:
                lg = R.forward_from(0, None, ids)
            m = R.measure(lg[0].float(), ed.captured[TGT]["pre"][0], cl)
            res["zero_delta"].append({
                "cell": ci, "layers": layers,
                "max_abs_logits": float((lg[0].float() - cl["logits"]).abs().max().item()),
                "max_abs_dz": float(np.abs(m["dz"]).max()), "kl": m["kl"]})
            del lg
        # (2) fast path vs full path (all edits applied from the embeddings)
        sample_types = {}
        for cnd in conds:
            key = tuple(int(x != NONE) for x in cnd)
            sample_types.setdefault(key, []).append(cnd)
        picks = []
        for key, lst in sorted(sample_types.items()):
            idx = rng.choice(len(lst), size=min(2, len(lst)), replace=False)
            picks += [lst[i] for i in idx]
        # clean caches for the fast path
        for (a, b, c) in picks:
            # full path
            edits_full = ablate_edits(a, b, c, 0)
            m_full, _ = R.run_condition(ids, 0, None, edits_full, [TGT], cl)
            # fast path, as the run stage does it
            if a != NONE:
                mA, edA = R.run_condition(ids, 0, None, ablate_edits(a, NONE, NONE, 0), [5, 9, TGT], cl)
                h5, h9 = edA.captured[5]["pre"], edA.captured[9]["pre"]
            else:
                h5, h9 = cl["h5"], cl["h9"]
            if b != NONE:
                mB, edB = R.run_condition(ids, 5, h5, ablate_edits(a, b, NONE, 5), [9, TGT], cl)
                h9 = edB.captured[9]["pre"]
                if c == NONE:
                    m_fast = mB
            elif c == NONE:
                m_fast = mA
            if c != NONE:
                m_fast, _ = R.run_condition(ids, 9, h9, ablate_edits(a, b, c, 9), [TGT], cl)
            res["fast_vs_full"].append({
                "cell": ci, "cond": [int(a), int(b), int(c)],
                "max_abs_dz_diff": float(np.abs(m_fast["dz"] - m_full["dz"]).max()),
                "max_abs_dlog_diff": float(np.abs(m_fast["dlog"] - m_full["dlog"]).max()),
                "max_abs_dz": float(np.abs(m_full["dz"]).max()),
                "kl_full": m_full["kl"], "kl_fast": m_fast["kl"]})
            free()
        # (3) float32-on-device token mean vs float64 on CPU, one ABC condition
        a, b, c = [x for x in picks if all(v != NONE for v in x)][0]
        with H.ResidualEditor(R.model, R.saes, edits=ablate_edits(a, b, c, 0), capture=[TGT],
                              capture_device=R.device) as ed:
            lg = R.forward_from(0, None, ids)
        z = R.saes[TGT].encode(ed.captured[TGT]["pre"][0]).cpu().double()
        z0 = cl["z11"].cpu().double()
        dz64 = (z - z0).mean(0).numpy()
        m = R.measure(lg[0].float(), ed.captured[TGT]["pre"][0], cl)
        res["float64_check"].append({"cell": ci, "cond": [int(a), int(b), int(c)],
                                     "max_abs_diff": float(np.abs(m["dz"] - dz64).max()),
                                     "max_abs_dz": float(np.abs(dz64).max()),
                                     "median_abs_nonzero_dz": float(np.median(np.abs(dz64[dz64 != 0])))})
        del lg, cl
        free()
        log(f"verify cell {ci} done")
    zd = max(r["max_abs_logits"] for r in res["zero_delta"])
    zz = max(r["max_abs_dz"] for r in res["zero_delta"])
    fv = max(r["max_abs_dz_diff"] for r in res["fast_vs_full"])
    fl = max(r["max_abs_dlog_diff"] for r in res["fast_vs_full"])
    f64 = max(r["max_abs_diff"] for r in res["float64_check"])
    res["verdict"] = {
        "zero_delta_max_abs_logits": zd, "zero_delta_max_abs_dz": zz,
        "zero_delta_pass(<1e-5)": bool(zd < 1e-5 and zz < 1e-5),
        "fast_vs_full_max_abs_dz": fv, "fast_vs_full_max_abs_dlog": fl,
        "fast_vs_full_pass(<1e-5)": bool(fv < 1e-5 and fl < 1e-5),
        "float32_mean_vs_float64_max_abs": f64, "float32_pass(<1e-6)": bool(f64 < 1e-6),
        "clean_repeat_max_abs": max(r["max_abs_logits"] for r in res["clean_repeat"]),
    }
    res["wall_seconds"] = round(time.time() - t0, 1)
    H.write_json(OUT / "verify.json", res)
    update_run_config("verify", res["verdict"])
    update_run_config("chunks", {"stage": "verify", "start": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                 "wall_seconds": res["wall_seconds"], "free_gb": free_gb})
    log(json.dumps(res["verdict"], indent=1))


# =============================================================================
# stage: run
# =============================================================================
def group_path(ci, gi):
    return CELL_DIR / f"cell{ci:02d}_g{gi:02d}.npz"


def stage_run(args):
    torch.manual_seed(0)
    t_start = time.time()
    budget = args.max_minutes * 60.0
    sel = load_json(OUT / "selection.json")
    toks, meta = load_cells_cached()
    triplets, conds, a_groups = build_design(sel)
    cond_set = set(conds)
    todo = [(ci, gi) for ci in range(len(toks)) for gi in range(len(a_groups))
            if not group_path(ci, gi).exists()]
    if not todo:
        log("DONE (all groups exist)")
        return
    log(f"{len(todo)} groups left of {len(toks) * len(a_groups)}; {len(conds)} conditions per cell")
    try:
        free_gb = H.check_memory(MIN_FREE_GB)
    except H.MemoryGuardError as e:
        log(f"STOP: {e}")
        return
    R = Runner()
    done_groups, durations = [], []
    cur_ci, cl = None, None
    status = "PARTIAL"
    try:
        for (ci, gi) in todo:
            est = max(durations) if durations else 60.0
            if time.time() - t_start + est > budget:
                log("time budget reached; stopping cleanly")
                break
            H.check_memory(MIN_FREE_GB)
            tg = time.time()
            ids = torch.from_numpy(toks[ci][None, :]).to(R.device)
            if cur_ci != ci:
                cl = None
                free()
                cl = R.clean(ids)
                cur_ci = ci
                # fingerprint computed on CPU in float64: a full-tensor .sum() on MPS was found to return
                # wrong values now and then for these 10-40M element tensors (see stage 'recheck')
                clean_sig = np.array([float(cl["z11"].cpu().double().sum()), float(cl["logits"].cpu().double().sum())])
            a = a_groups[gi]
            recs = {}                                   # cond -> meas
            # level A
            if a == NONE:
                h5a, h9a = cl["h5"], cl["h9"]
            else:
                m, ed = R.run_condition(ids, 0, None, ablate_edits(a, NONE, NONE, 0), [5, 9, TGT], cl)
                recs[(a, NONE, NONE)] = m
                h5a, h9a = ed.captured[5]["pre"], ed.captured[9]["pre"]
                del ed
            # (a, NONE, c)
            for (x, y, z) in conds:
                if x == a and y == NONE and z != NONE:
                    m, ed = R.run_condition(ids, 9, h9a, ablate_edits(x, y, z, 9), [TGT], cl)
                    recs[(x, y, z)] = m
                    del ed
            # (a, b, *)
            bs = sorted({y for (x, y, z) in conds if x == a and y != NONE})
            for b in bs:
                m, ed = R.run_condition(ids, 5, h5a, ablate_edits(a, b, NONE, 5), [9, TGT], cl)
                recs[(a, b, NONE)] = m
                h9ab = ed.captured[9]["pre"]
                del ed
                for (x, y, z) in conds:
                    if x == a and y == b and z != NONE:
                        m, ed = R.run_condition(ids, 9, h9ab, ablate_edits(x, y, z, 9), [TGT], cl)
                        recs[(x, y, z)] = m
                        del ed
                del h9ab
            del h5a, h9a
            keys = sorted(recs)
            assert all(k in cond_set for k in keys), "unexpected condition"
            # live n_active of each edited feature: the L0 count comes from the (a,-,-) run, the L5
            # count from the (a,b,-) run (its live context), the L9 count from the run itself.
            na = np.full((len(keys), 3), -1, dtype=np.int64)
            for i, (x, y, z) in enumerate(keys):
                if x != NONE:
                    na[i, 0] = recs[(x, NONE, NONE)]["n_active"].get((0, int(x)), -2)
                if y != NONE:
                    na[i, 1] = recs[(x, y, NONE)]["n_active"].get((5, int(y)), -2)
                if z != NONE:
                    na[i, 2] = recs[(x, y, z)]["n_active"].get((9, int(z)), -2)
            np.savez(group_path(ci, gi),
                     conds=np.array(keys, dtype=np.int64),
                     dz=np.stack([recs[k]["dz"] for k in keys]).astype(np.float32),
                     dlog=np.stack([recs[k]["dlog"] for k in keys]).astype(np.float32),
                     mabs=np.array([recs[k]["mabs"] for k in keys]),
                     kl=np.array([recs[k]["kl"] for k in keys]),
                     n_active=na, clean_sig=clean_sig, a_group=a)
            durations.append(time.time() - tg)
            done_groups.append([ci, gi, len(keys), round(durations[-1], 1)])
            log(f"cell {ci} group {gi} (a={a}): {len(keys)} conditions in {durations[-1]:.1f}s")
            del recs
            free()
        else:
            status = "DONE"
    except H.MemoryGuardError as e:
        log(f"STOP (memory guard): {e}")
    left = sum(1 for ci in range(len(toks)) for gi in range(len(a_groups)) if not group_path(ci, gi).exists())
    if left == 0:
        status = "DONE"
    update_run_config("chunks", {"stage": "run", "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t_start)),
                                 "wall_seconds": round(time.time() - t_start, 1), "free_gb_at_start": free_gb,
                                 "groups_done": done_groups, "groups_left": left})
    log(f"{status}: {left} groups left")


# =============================================================================
# stage: recheck  (independent recomputation with CPU float64 reductions)
# =============================================================================
@torch.no_grad()
def measure_cpu64(logits, h11, clean64, sae11):
    """Reference measurement: SAE encode on the device, every reduction on the CPU in float64."""
    z = sae11.encode(h11).cpu().double()
    lg = logits.cpu().double()
    dz = (z - clean64["z"]).mean(0).numpy()
    dl = lg - clean64["lg"]
    logp = torch.log_softmax(lg, dim=-1)
    kl = float((clean64["p"] * (clean64["logp"] - logp)).sum(-1).mean())
    return {"dz": dz, "dlog": dl.mean(0).numpy(), "mabs": float(dl.abs().mean()), "kl": kl,
            "max_abs_dlogit": float(dl.abs().max())}


def stage_recheck(args):
    """For every cell and every stored group, recompute 2 random conditions from scratch (full model
    path, all edits applied from the embeddings) with CPU float64 reductions, and compare with the
    stored values. Also repeats the zero-delta check with CPU-side comparisons."""
    torch.manual_seed(0)
    t_start = time.time()
    budget = args.max_minutes * 60.0
    rdir = OUT / "recheck"
    rdir.mkdir(parents=True, exist_ok=True)
    sel = load_json(OUT / "selection.json")
    toks, meta = load_cells_cached()
    triplets, conds, a_groups = build_design(sel)
    todo = [ci for ci in range(len(toks)) if not (rdir / f"cell{ci:02d}.json").exists()]
    if not todo:
        log("DONE (recheck complete)")
        return
    free_gb = H.check_memory(MIN_FREE_GB)
    R = Runner()
    durations, done = [], []
    try:
        for ci in todo:
            est = max(durations) if durations else 90.0
            if time.time() - t_start + est > budget:
                log("time budget reached; stopping cleanly")
                break
            H.check_memory(MIN_FREE_GB)
            tc = time.time()
            ids = torch.from_numpy(toks[ci][None, :]).to(R.device)
            cl = R.clean(ids)
            c64 = {"z": cl["z11"].cpu().double(), "lg": cl["logits"].cpu().double()}
            c64["logp"] = torch.log_softmax(c64["lg"], dim=-1)
            c64["p"] = c64["logp"].exp()
            out = {"cell": ci, "row": meta["rows"][ci],
                   "clean_fingerprint_cpu64": [float(c64["z"].sum()), float(c64["lg"].sum())],
                   "stored_fingerprints": [], "checks": []}
            rng = np.random.default_rng(SEL_SEED + 1000 + ci)
            if ci == 0:
                zres = []
                for layers in ([0], [5], [9], [0, 5, 9]):
                    with H.ResidualEditor(R.model, R.saes, edits=[H.ZeroDelta(l) for l in layers],
                                          capture=[TGT], capture_device=R.device) as ed:
                        lg = R.forward_from(0, None, ids)
                    m = measure_cpu64(lg[0], ed.captured[TGT]["pre"][0], c64, R.saes[TGT])
                    zres.append({"layers": layers, "max_abs_dlogit": m["max_abs_dlogit"],
                                 "max_abs_dz": float(np.abs(m["dz"]).max()), "kl": m["kl"]})
                    del lg
                out["zero_delta_cpu64"] = zres
            for gi in range(len(a_groups)):
                g = np.load(group_path(ci, gi))
                out["stored_fingerprints"].append([float(x) for x in g["clean_sig"]])
                keys = [tuple(int(v) for v in k) for k in g["conds"]]
                for r in rng.choice(len(keys), size=min(2, len(keys)), replace=False):
                    a, b, c = keys[r]
                    with H.ResidualEditor(R.model, R.saes, edits=ablate_edits(a, b, c, 0), capture=[TGT],
                                          capture_device=R.device) as ed:
                        lg = R.forward_from(0, None, ids)
                    m = measure_cpu64(lg[0], ed.captured[TGT]["pre"][0], c64, R.saes[TGT])
                    del lg
                    out["checks"].append({
                        "group": gi, "cond": [a, b, c],
                        "max_abs_dz_diff": float(np.abs(m["dz"] - g["dz"][r]).max()),
                        "max_abs_dz": float(np.abs(m["dz"]).max()),
                        "max_abs_dlog_diff": float(np.abs(m["dlog"] - g["dlog"][r]).max()),
                        "max_abs_dlog": float(np.abs(m["dlog"]).max()),
                        "mabs_stored": float(g["mabs"][r]), "mabs_cpu64": m["mabs"],
                        "kl_stored": float(g["kl"][r]), "kl_cpu64": m["kl"]})
                    free()
            H.write_json(rdir / f"cell{ci:02d}.json", out)
            del cl, c64
            free()
            durations.append(time.time() - tc)
            done.append([ci, round(durations[-1], 1)])
            log(f"recheck cell {ci}: {len(out['checks'])} conditions in {durations[-1]:.0f}s; "
                f"max dz diff {max(c['max_abs_dz_diff'] for c in out['checks']):.2e}, "
                f"max dlog diff {max(c['max_abs_dlog_diff'] for c in out['checks']):.2e}")
    except H.MemoryGuardError as e:
        log(f"STOP (memory guard): {e}")
    left = sum(1 for ci in range(len(toks)) if not (rdir / f"cell{ci:02d}.json").exists())
    update_run_config("chunks", {"stage": "recheck", "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t_start)),
                                 "wall_seconds": round(time.time() - t_start, 1), "free_gb_at_start": free_gb,
                                 "cells_done": done, "cells_left": left})
    log(("DONE" if left == 0 else "PARTIAL") + f": {left} cells left")


# =============================================================================
# stage: analyze
# =============================================================================
def bh(p):
    """Benjamini-Hochberg adjusted q-values for a 1-D array."""
    p = np.asarray(p, dtype=np.float64)
    n = p.size
    order = np.argsort(p)
    q = np.empty(n)
    ranked = p[order] * n / np.arange(1, n + 1)
    q[order] = np.minimum.accumulate(ranked[::-1])[::-1]
    return np.minimum(q, 1.0)


def ttest_p(x):
    """Two-sided one-sample t-test p-values along axis 0 (cells). Zero-variance -> p = 1
    if the mean is 0, else p = 0 (constant non-zero effect in every cell)."""
    from scipy import stats
    n = x.shape[0]
    m = x.mean(0)
    sd = x.std(0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = m / (sd / np.sqrt(n))
    p = 2 * stats.t.sf(np.abs(t), df=n - 1)
    p = np.where(sd > 0, p, np.where(m == 0, 1.0, 0.0))
    return p


def summarize(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    return {"n": int(x.size), "mean": float(x.mean()), "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
            "min": float(x.min()), "p05": float(np.percentile(x, 5)), "p25": float(np.percentile(x, 25)),
            "median": float(np.median(x)), "p75": float(np.percentile(x, 75)),
            "p95": float(np.percentile(x, 95)), "max": float(x.max())}


def ci95(v):
    """Percentile interval (used only for unbiased linear summaries such as means of I_ABC)."""
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def ci_basic(point, boot, plugin=None):
    """Basic (pivot) bootstrap interval: [point - (q97.5 - plugin), point - (q2.5 - plugin)].
    ``plugin`` is the value of the statistic on the original sample computed the way the bootstrap
    replicates are computed (the 'truth' of the bootstrap world). For a plug-in statistic plugin =
    point; for a noise-corrected statistic it is the uncorrected plug-in value. This corrects the
    first-order bias that the percentile interval keeps (several ratios here are biased by noise)."""
    plugin = point if plugin is None else plugin
    q = ci95(boot)
    return [float(point - (q[1] - plugin)), float(point - (q[0] - plugin))]


def stage_analyze(args):
    t0 = time.time()
    sel = load_json(OUT / "selection.json")
    toks, meta = load_cells_cached()
    triplets, conds, a_groups = build_design(sel)
    nC = args.n_cells or len(toks)
    out_dir = Path(args.out_dir) if args.out_dir else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    cidx = {c: i for i, c in enumerate(conds)}
    K = len(conds)
    d_sae, V = 4928, 20275
    D = np.zeros((nC, K, d_sae), dtype=np.float32)
    LG = np.zeros((nC, K, V), dtype=np.float32)
    MABS = np.zeros((nC, K)); KL = np.zeros((nC, K)); NA = np.full((nC, K, 3), -3, dtype=np.int64)
    seen = np.zeros((nC, K), dtype=bool)
    clean_sigs = {}
    for ci in range(nC):
        for gi in range(len(a_groups)):
            p = group_path(ci, gi)
            if not p.exists():
                raise SystemExit(f"missing {p}; run stage 'run' until DONE")
            g = np.load(p)
            clean_sigs.setdefault(ci, []).append(g["clean_sig"])
            for r, k in enumerate(map(tuple, g["conds"])):
                j = cidx[k]
                assert not seen[ci, j], f"duplicate {k}"
                seen[ci, j] = True
                D[ci, j] = g["dz"][r]; LG[ci, j] = g["dlog"][r]
                MABS[ci, j] = g["mabs"][r]; KL[ci, j] = g["kl"][r]; NA[ci, j] = g["n_active"][r]
    assert seen.all(), "some conditions missing"
    log(f"loaded {nC} cells x {K} conditions")
    # --- data-integrity checks ------------------------------------------------------------
    # (1) recheck stage: 2 random conditions per stored group recomputed from scratch with CPU
    #     float64 reductions.
    integrity = {}
    rdir = OUT / "recheck"
    rfiles = sorted(rdir.glob("cell*.json"))
    if rfiles:
        rc = [load_json(f) for f in rfiles]
        chk = [c for d in rc for c in d["checks"]]
        fp_bad = sum(1 for d in rc for sfp in d["stored_fingerprints"]
                     if any(abs(sfp[i] - d["clean_fingerprint_cpu64"][i]) > 1e-5 * abs(d["clean_fingerprint_cpu64"][i])
                            for i in (0, 1)))
        mabs_rel = [abs(c["mabs_stored"] - c["mabs_cpu64"]) / max(c["mabs_cpu64"], 1e-30) for c in chk]
        kl_rel = [abs(c["kl_stored"] - c["kl_cpu64"]) / max(c["kl_cpu64"], 1e-30) for c in chk if c["kl_cpu64"] > 0]
        integrity["recheck"] = {
            "cells": len(rc), "conditions_recomputed": len(chk),
            "max_abs_dz_diff": max(c["max_abs_dz_diff"] for c in chk),
            "max_abs_dlog_diff": max(c["max_abs_dlog_diff"] for c in chk),
            "kl_max_rel_diff": max(kl_rel) if kl_rel else None,
            "mean_abs_dlogit_stored_wrong(rel>1e-3)": int(sum(x > 1e-3 for x in mabs_rel)),
            "mean_abs_dlogit_max_rel_err": max(mabs_rel),
            "stored_clean_fingerprints_wrong(rel>1e-5)": fp_bad,
            "stored_clean_fingerprints_total": sum(len(d["stored_fingerprints"]) for d in rc),
            "zero_delta_cpu64": rc[0].get("zero_delta_cpu64"),
            "note": ("stored dz/dlog (dim-0 sums on MPS) match CPU float64; the full-tensor MPS .sum()/.mean() "
                     "used for the clean fingerprint and for mean|dlogit| was sometimes wrong, so mean|dlogit| "
                     "is not used anywhere and the fingerprint is replaced by the recheck"),
        }
    # (2) twin conditions: a condition that contains a never-active feature which stayed inactive in
    #     that cell must equal (bit for bit) the same condition without that feature. Both were
    #     computed in separate forward passes, often in separate chunks.
    nullset = {l: set(sel["null_features"][str(l)]) for l in SRC_LAYERS}
    n_twin = n_twin_equal = 0
    twin_max = 0.0
    for j, k in enumerate(conds):
        for pos, l in enumerate(SRC_LAYERS):
            if k[pos] != NONE and k[pos] in nullset[l]:
                tw = list(k); tw[pos] = NONE; tw = tuple(tw)
                if tw == (NONE, NONE, NONE) or tw not in cidx:
                    continue
                jt = cidx[tw]
                for ci in range(nC):
                    if NA[ci, j, pos] == 0:
                        n_twin += 1
                        eq = np.array_equal(D[ci, j], D[ci, jt]) and np.array_equal(LG[ci, j], LG[ci, jt])
                        n_twin_equal += int(eq)
                        twin_max = max(twin_max, float(np.abs(D[ci, j] - D[ci, jt]).max()),
                                       float(np.abs(LG[ci, j] - LG[ci, jt]).max()))
    integrity["twin_conditions"] = {"pairs_compared": n_twin, "pairs_bitwise_equal": n_twin_equal,
                                    "max_abs_diff": twin_max}
    log(f"integrity: {json.dumps(integrity, default=str)[:600]}")
    # logit read-out: position-mean delta-logit, centred over the vocabulary (a shift shared by all
    # tokens does not change the softmax, so it is removed before any logit summary)
    LG -= LG.mean(axis=2, keepdims=True)

    def ix(a, b, c):
        return cidx[(a, b, c)]

    rng = np.random.default_rng(BOOT_SEED)
    Wb = rng.multinomial(nC, np.full(nC, 1.0 / nC), size=N_BOOT).astype(np.float32) / nC   # (B, nC)
    np.save(out_dir / "bootstrap_cell_weights.npy", Wb)
    dev = args.device or DEVICE
    Wt = torch.from_numpy(Wb).to(dev)
    # sign-flip null (cells independent; under H0 a cell's interaction vector is as likely to be
    # +I as -I): one random sign per cell, applied to the whole target vector of that cell
    N_FLIP = 999
    frng = np.random.default_rng(BOOT_SEED + 1)
    Sf = frng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(N_FLIP, nC))
    np.save(out_dir / "signflip_signs.npy", Sf)
    St = torch.from_numpy(Sf / nC).to(dev)                         # (N_FLIP, nC) -> mean of flipped cells
    S5 = torch.from_numpy(Sf[:5]).to(dev)                          # 5 flips for the CI-exclusion null

    def tpct(srt, q):
        """numpy-style 'linear' percentile of a tensor already sorted along dim 0."""
        pos = q / 100.0 * (srt.shape[0] - 1)
        lo_i = int(np.floor(pos)); fr = pos - lo_i
        hi_i = min(lo_i + 1, srt.shape[0] - 1)
        return srt[lo_i] + fr * (srt[hi_i] - srt[lo_i])

    def energy_np(X):
        """Unbiased estimate of ||true cell-mean vector||^2 from per-cell rows X (n, T):
        ||mean||^2 - sum_t var_t / n (var with ddof=1)."""
        n = X.shape[0]
        return float((X.mean(0) ** 2).sum() - X.var(0, ddof=1).sum() / n)

    def energy_boot(Xt):
        """Same estimator under each bootstrap weight vector (rows of Wt). Xt: (n, T) tensor."""
        n = Xt.shape[0]
        m = Wt @ Xt                                   # (B, T) weighted means
        m2 = Wt @ (Xt * Xt)
        var = (m2 - m * m).clamp_min(0) * n / (n - 1)
        return ((m * m).sum(1) - var.sum(1) / n).cpu().numpy()

    A, B, C = sel["features"]["0"], sel["features"]["5"], sel["features"]["9"]
    real = [t for t in triplets if t["kind"] == "real"]

    # ------------------------------------------------------------------ per condition
    effD = D.mean(0, dtype=np.float64)                       # (K, d_sae) mean over cells
    effL = LG.mean(0, dtype=np.float64)
    eff_abs = np.abs(effD).mean(1)                          # deployed "eff_X" (mean over targets of |e|)
    eff_abs_L = np.abs(effL).mean(1)
    # bootstrap of eff_X (mean over targets of |mean over resampled cells|)
    boot_eff = np.zeros((K, N_BOOT), dtype=np.float32)
    boot_eff_L = np.zeros((K, N_BOOT), dtype=np.float32)
    with torch.no_grad():
        for j0 in range(0, K, 16):
            js = slice(j0, min(K, j0 + 16))
            x = torch.from_numpy(np.ascontiguousarray(D[:, js])).to(dev).reshape(nC, -1)
            boot_eff[js] = (Wt @ x).reshape(N_BOOT, -1, d_sae).abs().mean(2).T.cpu().numpy()
            for j1 in range(js.start, js.stop, 4):
                jj = slice(j1, min(js.stop, j1 + 4))
                y = torch.from_numpy(np.ascontiguousarray(LG[:, jj])).to(dev).reshape(nC, -1)
                boot_eff_L[jj] = (Wt @ y).reshape(N_BOOT, -1, V).abs().mean(2).T.cpu().numpy()
            del x, y
    free()
    log("bootstrap of condition effects done")

    def ratios(e):
        mx = np.maximum
        r_ab = e["AB"] / mx(e["A"] + e["B"], 1e-30)
        r_ac = e["AC"] / mx(e["A"] + e["C"], 1e-30)
        r_bc = e["BC"] / mx(e["B"] + e["C"], 1e-30)
        three = e["ABC"] / mx(e["A"] + e["B"] + e["C"], 1e-30)
        marg = (e["ABC"] - e["AB"]) / mx(e["C"], 1e-30)
        return r_ab, r_ac, r_bc, (r_ab + r_ac + r_bc) / 3, three, marg

    # ------------------------------------------------------------------ per triplet
    rows = []
    n_trip = len(triplets)
    p_all_real = []                      # for BH across triplets
    I_store_mean = np.zeros((n_trip, d_sae), dtype=np.float32)
    lo_store = np.zeros((n_trip, d_sae), dtype=np.float32)
    hi_store = np.zeros((n_trip, d_sae), dtype=np.float32)
    p_store = np.zeros((n_trip, d_sae), dtype=np.float64)
    pboot_store = np.zeros((n_trip, d_sae), dtype=np.float64)
    boot_ratio = {k: np.zeros((len(real), N_BOOT), dtype=np.float32) for k in
                  ("AB", "AC", "BC", "pair_mean", "three", "marg", "Ishare", "three_L", "pair_mean_L", "Ishare_L",
                   "three_add", "three_minus_add", "pair_add", "pair_minus_add",
                   "rho3", "eshare_I", "eshare_tot", "rho3_L", "eshare_I_L", "rho2")}
    for ti, t in enumerate(triplets):
        a, b, c = t["a"], t["b"], t["c"]
        E = {"A": (a, NONE, NONE), "B": (NONE, b, NONE), "C": (NONE, NONE, c),
             "AB": (a, b, NONE), "AC": (a, NONE, c), "BC": (NONE, b, c), "ABC": (a, b, c)}
        J = {k: ix(*v) for k, v in E.items()}
        Dc = {k: D[:, J[k]].astype(np.float64) for k in J}            # (nC, d_sae)
        Lc = {k: LG[:, J[k]].astype(np.float64) for k in J}
        I = Dc["ABC"] - Dc["AB"] - Dc["AC"] - Dc["BC"] + Dc["A"] + Dc["B"] + Dc["C"]     # per cell
        IL = Lc["ABC"] - Lc["AB"] - Lc["AC"] - Lc["BC"] + Lc["A"] + Lc["B"] + Lc["C"]
        Im = I.mean(0)
        with torch.no_grad():
            bI_t = Wt @ torch.from_numpy(I.astype(np.float32)).to(dev)          # (B, d_sae)
            srt, _ = torch.sort(bI_t, dim=0)
            lo = tpct(srt, 2.5).cpu().numpy().astype(np.float64)
            hi = tpct(srt, 97.5).cpu().numpy().astype(np.float64)
            nle = (bI_t <= 0).sum(0).cpu().numpy(); nge = (bI_t >= 0).sum(0).cpu().numpy()
            del srt
        alive = np.abs(I).max(0) > 0
        excl = alive & ((lo > 0) | (hi < 0))
        pboot = np.minimum(1.0, 2 * np.minimum(nle + 1, nge + 1) / (N_BOOT + 1))
        pboot = np.where(alive, pboot, 1.0)
        p = ttest_p(I)
        q_within = bh(p)
        IL_m = IL.mean(0)
        pL = ttest_p(IL)
        qL_within = bh(pL)
        I_store_mean[ti], lo_store[ti], hi_store[ti], p_store[ti], pboot_store[ti] = Im, lo, hi, p, pboot
        with torch.no_grad():
            It = torch.from_numpy(I.astype(np.float32)).to(dev)
            T_obs = float(It.mean(0).abs().sum().item())
            T_null = (St @ It).abs().sum(1).cpu().numpy()
            p_agg = float((1 + (T_null >= T_obs).sum()) / (N_FLIP + 1)) if T_obs > 0 else 1.0
            n_excl_null = []
            alive_t = torch.from_numpy(alive).to(dev)
            for f in range(S5.shape[0]):
                bf = Wt @ (It * S5[f][:, None])
                sf, _ = torch.sort(bf, dim=0)
                lf, hf = tpct(sf, 2.5), tpct(sf, 97.5)
                n_excl_null.append(int((alive_t & ((lf > 0) | (hf < 0))).sum().item()))
                del bf, sf
            del It
        # interaction share: sum_t |I_t| / sum_t |e_ABC_t|
        eABC = Dc["ABC"].mean(0)
        share = np.abs(Im).sum() / max(np.abs(eABC).sum(), 1e-30)
        share_L = np.abs(IL_m).sum() / max(np.abs(Lc["ABC"].mean(0)).sum(), 1e-30)
        # pairwise interaction terms (context)
        IAB = (Dc["AB"] - Dc["A"] - Dc["B"]).mean(0)
        IAC = (Dc["AC"] - Dc["A"] - Dc["C"]).mean(0)
        IBC = (Dc["BC"] - Dc["B"] - Dc["C"]).mean(0)
        pair_share = [np.abs(x).sum() / max(np.abs(Dc[k].mean(0)).sum(), 1e-30)
                      for x, k in ((IAB, "AB"), (IAC, "AC"), (IBC, "BC"))]
        # deployed redundancy ratios (eff_X = mean over targets of |e_X|)
        ef = {k: float(eff_abs[J[k]]) for k in J}
        efL = {k: float(eff_abs_L[J[k]]) for k in J}
        r_ab, r_ac, r_bc, r_pm, r_three, r_marg = (float(x) for x in ratios(ef))
        rL = [float(x) for x in ratios(efL)]
        # spec classification (Cohen's d_z per condition, |d|>0.5 in any condition)
        dd = {}
        for k in J:
            m = Dc[k].mean(0); sd_ = Dc[k].std(0, ddof=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                dd[k] = np.where(sd_ > 0, m / sd_, 0.0)
        incl = np.zeros(d_sae, dtype=bool)
        for k in dd:
            incl |= np.abs(dd[k]) > 0.5
        sum_d = np.abs(dd["A"]) + np.abs(dd["B"]) + np.abs(dd["C"])
        with np.errstate(divide="ignore", invalid="ignore"):
            rd = np.where(sum_d > 0, np.abs(dd["ABC"]) / sum_d, np.inf)
        spec_super_strict = int((incl & (np.abs(dd["ABC"]) > sum_d)).sum())
        spec_super_band = int((incl & (rd > 1.1)).sum())
        spec_add = int((incl & (rd >= 0.9) & (rd <= 1.1)).sum())
        spec_sub = int((incl & (rd < 0.9)).sum())
        # deployed rule (raw means, |e_ABC| > 0.01, 1.1x)
        eA, eB, eC = (np.abs(Dc[k].mean(0)) for k in ("A", "B", "C"))
        dep_tg = np.abs(eABC) > 0.01
        dep_super = int((dep_tg & (np.abs(eABC) > 1.1 * (eA + eB + eC))).sum())
        # sanity: AB vs B etc. (max abs difference of the cell-mean effect)
        san = {
            "AB_vs_B": float(np.abs(Dc["AB"].mean(0) - Dc["B"].mean(0)).max()),
            "AC_vs_C": float(np.abs(Dc["AC"].mean(0) - Dc["C"].mean(0)).max()),
            "BC_vs_C": float(np.abs(Dc["BC"].mean(0) - Dc["C"].mean(0)).max()),
            "ABC_vs_C": float(np.abs(eABC - Dc["C"].mean(0)).max()),
            "AB_vs_B_percell": float(np.abs(Dc["AB"] - Dc["B"]).max()),
        }
        # live n_active of each edited feature in ABC (b and c in their live context)
        na_abc = NA[:, J["ABC"]]
        rows.append({
            "triplet": ti, "kind": t["kind"], "a": a, "b": b, "c": c,
            "n_targets_alive_any": int((np.abs(I).max(0) > 0).sum()),
            "I_max_abs": float(np.abs(Im).max()), "I_sum_abs": float(np.abs(Im).sum()),
            "I_max_abs_percell": float(np.abs(I).max()),
            "n_ci_excl0": int(excl.sum()), "n_q05_within_t": int((q_within < 0.05).sum()),
            "n_ci_excl0_signflip_null": n_excl_null, "n_p05_uncorrected_t": int((p < 0.05).sum()),
            "agg_T_obs": T_obs, "agg_T_null_median": float(np.median(T_null)), "agg_p_signflip": p_agg,
            "share_I_noise_signflip": float(np.median(T_null)) / max(float(np.abs(eABC).sum()), 1e-30),
            "frac_sub_obs_spec_included": float(np.mean(np.abs(eABC[incl]) < 0.9 * (eA + eB + eC)[incl])) if incl.any() else None,
            "frac_sub_additive_spec_included": float(np.mean(np.abs((Dc["A"] + Dc["B"] + Dc["C"]).mean(0)[incl]) < 0.9 * (eA + eB + eC)[incl])) if incl.any() else None,
            "n_q05_within_boot": int((bh(pboot) < 0.05).sum()),
            "min_p_t": float(p.min()), "min_p_boot": float(pboot.min()),
            "share_I_over_ABC": float(share), "share_I_over_ABC_logits": float(share_L),
            "pair_shares": [float(x) for x in pair_share],
            "logit_n_q05_within_t": int((qL_within < 0.05).sum()),
            "eff": ef, "eff_logits": efL,
            "r_AB": r_ab, "r_AC": r_ac, "r_BC": r_bc, "r_pair_mean": r_pm, "r_three": r_three,
            "marg_C_given_AB": r_marg,
            "r_three_logits": rL[4], "r_pair_mean_logits": rL[3],
            "legacy_identity_eC_over_sum": ef["C"] / max(ef["A"] + ef["B"] + ef["C"], 1e-30),
            "spec_n_included": int(incl.sum()), "spec_super_strict": spec_super_strict,
            "spec_super_band": spec_super_band, "spec_additive": spec_add, "spec_sub": spec_sub,
            "dep_n_targets": int(dep_tg.sum()), "dep_super": dep_super,
            "kl": {k: float(KL[:, J[k]].mean()) for k in J},
            "sanity": san,
            "n_active_ABC_cellmean": [float(na_abc[:, 0][na_abc[:, 0] >= 0].mean()) if (na_abc[:, 0] >= 0).any() else None,
                                      float(na_abc[:, 1].mean()), float(na_abc[:, 2].mean())],
        })
        if t["kind"] == "real":
            ri = len(p_all_real)
            p_all_real.append(p)
            # bootstrap of ratios
            be = {k: boot_eff[J[k]] for k in J}
            beL = {k: boot_eff_L[J[k]] for k in J}
            r = ratios(be); rl = ratios(beL)
            boot_ratio["AB"][ri], boot_ratio["AC"][ri], boot_ratio["BC"][ri] = r[0], r[1], r[2]
            boot_ratio["pair_mean"][ri], boot_ratio["three"][ri], boot_ratio["marg"][ri] = r[3], r[4], r[5]
            boot_ratio["three_L"][ri], boot_ratio["pair_mean_L"][ri] = rl[4], rl[3]
            with torch.no_grad():
                f32 = lambda x: torch.from_numpy(x.astype(np.float32)).to(dev)  # noqa: E731
                bABC = (Wt @ f32(Dc["ABC"])).abs().sum(1)
                boot_ratio["Ishare"][ri] = (bI_t.abs().sum(1) / bABC.clamp_min(1e-30)).cpu().numpy()
                bIL = (Wt @ f32(IL)).abs().sum(1)
                bLABC = (Wt @ f32(Lc["ABC"])).abs().sum(1)
                boot_ratio["Ishare_L"][ri] = (bIL / bLABC.clamp_min(1e-30)).cpu().numpy()
                bx = {k: Wt @ f32(Dc[k]) for k in ("A", "B", "C", "AB", "AC", "BC", "ABC")}
                mabs_ = {k: v.abs().mean(1) for k, v in bx.items()}
                den3 = mabs_["A"] + mabs_["B"] + mabs_["C"]
                r_add3 = (bx["A"] + bx["B"] + bx["C"]).abs().mean(1) / den3
                r_obs3 = mabs_["ABC"] / den3
                pr_add = sum((bx[x] + bx[y]).abs().mean(1) / (mabs_[x] + mabs_[y]) for x, y in (("A", "B"), ("A", "C"), ("B", "C"))) / 3
                pr_obs = sum(mabs_[x + y] / (mabs_[x] + mabs_[y]) for x, y in (("A", "B"), ("A", "C"), ("B", "C"))) / 3
                boot_ratio["three_add"][ri] = r_add3.cpu().numpy()
                boot_ratio["three_minus_add"][ri] = (r_obs3 - r_add3).cpu().numpy()
                boot_ratio["pair_add"][ri] = pr_add.cpu().numpy()
                boot_ratio["pair_minus_add"][ri] = (pr_obs - pr_add).cpu().numpy()
                del bx
            eA_s, eB_s, eC_s = (Dc[k].mean(0) for k in ("A", "B", "C"))
            den3p = np.abs(eA_s).mean() + np.abs(eB_s).mean() + np.abs(eC_s).mean()
            rows[-1]["r_three_additive"] = float(np.abs(eA_s + eB_s + eC_s).mean() / den3p)
            pa = []
            for x, y in (("A", "B"), ("A", "C"), ("B", "C")):
                ex, ey = Dc[x].mean(0), Dc[y].mean(0)
                pa.append(np.abs(ex + ey).mean() / (np.abs(ex).mean() + np.abs(ey).mean()))
            rows[-1]["r_pair_mean_additive"] = float(np.mean(pa))
            rows[-1]["ci_three_minus_additive"] = ci_basic(rows[-1]["r_three"] - rows[-1]["r_three_additive"],
                                                           boot_ratio["three_minus_add"][ri])
            # noise-corrected energies (L2): rho3 = E[ABC] / E[A+B+C]; interaction energy shares
            Xadd = Dc["A"] + Dc["B"] + Dc["C"]
            Xtot = Dc["ABC"] - Xadd                                   # all interaction orders
            Ladd = Lc["A"] + Lc["B"] + Lc["C"]
            en = {"ABC": energy_np(Dc["ABC"]), "add": energy_np(Xadd), "I": energy_np(I), "tot": energy_np(Xtot),
                  "L_ABC": energy_np(Lc["ABC"]), "L_add": energy_np(Ladd), "L_I": energy_np(IL)}
            e2 = []
            for x, y in (("A", "B"), ("A", "C"), ("B", "C")):
                e2.append(energy_np(Dc[x + y]) / max(energy_np(Dc[x] + Dc[y]), 1e-30))
            rows[-1]["energy"] = en
            # per-cell non-additivity: ||I_c||^2 / ||d_ABC,c||^2 and the same for all orders, median over cells
            nz = (Dc["ABC"] ** 2).sum(1) > 0
            rows[-1]["percell_share_I_median"] = float(np.median((I[nz] ** 2).sum(1) / (Dc["ABC"][nz] ** 2).sum(1))) if nz.any() else None
            rows[-1]["percell_share_tot_median"] = float(np.median((Xtot[nz] ** 2).sum(1) / (Dc["ABC"][nz] ** 2).sum(1))) if nz.any() else None
            nzl = (Lc["ABC"] ** 2).sum(1) > 0
            rows[-1]["percell_share_I_logits_median"] = float(np.median((IL[nzl] ** 2).sum(1) / (Lc["ABC"][nzl] ** 2).sum(1))) if nzl.any() else None
            pl = lambda X: float((X.mean(0) ** 2).sum())  # noqa: E731  plug-in (uncorrected) energy
            rows[-1]["rho3_plugin"] = pl(Dc["ABC"]) / max(pl(Xadd), 1e-30)
            rows[-1]["eshare_I_plugin"] = pl(I) / max(pl(Dc["ABC"]), 1e-30)
            rows[-1]["eshare_tot_plugin"] = pl(Xtot) / max(pl(Dc["ABC"]), 1e-30)
            rows[-1]["rho3_logits_plugin"] = pl(Lc["ABC"]) / max(pl(Ladd), 1e-30)
            rows[-1]["eshare_I_logits_plugin"] = pl(IL) / max(pl(Lc["ABC"]), 1e-30)
            rows[-1]["rho2_mean_plugin"] = float(np.mean([pl(Dc[x + y]) / max(pl(Dc[x] + Dc[y]), 1e-30)
                                                          for x, y in (("A", "B"), ("A", "C"), ("B", "C"))]))
            rows[-1]["rho3"] = en["ABC"] / max(en["add"], 1e-30)
            rows[-1]["eshare_I"] = en["I"] / max(en["ABC"], 1e-30)
            rows[-1]["eshare_tot"] = en["tot"] / max(en["ABC"], 1e-30)
            rows[-1]["rho3_logits"] = en["L_ABC"] / max(en["L_add"], 1e-30)
            rows[-1]["eshare_I_logits"] = en["L_I"] / max(en["L_ABC"], 1e-30)
            rows[-1]["rho2_mean"] = float(np.mean(e2))
            with torch.no_grad():
                b_abc = energy_boot(f32(Dc["ABC"])); b_add = energy_boot(f32(Xadd))
                b_I = energy_boot(f32(I)); b_tot = energy_boot(f32(Xtot))
                bl_abc = energy_boot(f32(Lc["ABC"])); bl_add = energy_boot(f32(Ladd)); bl_I = energy_boot(f32(IL))
                b2 = np.mean([energy_boot(f32(Dc[x + y])) / np.maximum(energy_boot(f32(Dc[x] + Dc[y])), 1e-30)
                              for x, y in (("A", "B"), ("A", "C"), ("B", "C"))], axis=0)
            with torch.no_grad():
                Xt_ = f32(Xtot)
                Ttot = float(Xt_.mean(0).abs().sum().item())
                Ttn = (St @ Xt_).abs().sum(1).cpu().numpy()
                del Xt_
            rows[-1]["agg_p_signflip_tot"] = float((1 + (Ttn >= Ttot).sum()) / (N_FLIP + 1)) if Ttot > 0 else 1.0
            rows[-1]["tot_T_obs_over_null"] = Ttot / max(float(np.median(Ttn)), 1e-30)
            boot_ratio["rho3"][ri] = b_abc / np.maximum(b_add, 1e-30)
            boot_ratio["eshare_I"][ri] = b_I / np.maximum(b_abc, 1e-30)
            boot_ratio["eshare_tot"][ri] = b_tot / np.maximum(b_abc, 1e-30)
            boot_ratio["rho3_L"][ri] = bl_abc / np.maximum(bl_add, 1e-30)
            boot_ratio["eshare_I_L"][ri] = bl_I / np.maximum(bl_abc, 1e-30)
            boot_ratio["rho2"][ri] = b2
            rr_ = rows[-1]
            rr_["ci_rho3"] = ci_basic(rr_["rho3"], boot_ratio["rho3"][ri], rr_["rho3_plugin"])
            rr_["ci_eshare_I"] = ci_basic(rr_["eshare_I"], boot_ratio["eshare_I"][ri], rr_["eshare_I_plugin"])
            rr_["ci_eshare_tot"] = ci_basic(rr_["eshare_tot"], boot_ratio["eshare_tot"][ri], rr_["eshare_tot_plugin"])
            rr_["ci_r_three"] = ci_basic(rr_["r_three"], boot_ratio["three"][ri])
            rr_["ci_r_pair_mean"] = ci_basic(rr_["r_pair_mean"], boot_ratio["pair_mean"][ri])
            rr_["ci_share_I"] = ci_basic(rr_["share_I_over_ABC"], boot_ratio["Ishare"][ri])
        if ti % 64 == 0:
            log(f"triplet {ti}/{n_trip}")
        del bI_t
    # BH across all real triplets x targets
    P = np.concatenate(p_all_real)
    Q = bh(P).reshape(len(real), d_sae)
    real_idx = [i for i, t in enumerate(triplets) if t["kind"] == "real"]
    for k, ti in enumerate(real_idx):
        rows[ti]["n_q05_across_t"] = int((Q[k] < 0.05).sum())
    PB = np.concatenate([pboot_store[i] for i in real_idx])
    QB = bh(PB).reshape(len(real), d_sae)
    for k, ti in enumerate(real_idx):
        rows[ti]["n_q05_across_boot"] = int((QB[k] < 0.05).sum())

    R_ = [rows[i] for i in real_idx]
    N_ = [r for r in rows if r["kind"] != "real"]
    q_agg = bh([r["agg_p_signflip"] for r in R_])
    for r, q in zip(R_, q_agg):
        r["agg_q_signflip_across_triplets"] = float(q)
    q_tot = bh([r["agg_p_signflip_tot"] for r in R_])
    for r, q in zip(R_, q_tot):
        r["agg_q_signflip_tot_across_triplets"] = float(q)

    # ---------------------------------------------------------------- planted-interaction power check
    # Add a consistent interaction kappa * (cell-mean joint effect) to every cell's d_ABC of 64 random real
    # triplets, then re-run the two triplet-level tests. Shows what size of three-way interaction these
    # 20 cells could have detected.
    prng = np.random.default_rng(BOOT_SEED + 2)
    pick = sorted(prng.choice(len(real_idx), size=64, replace=False).tolist())
    power = {}
    for kappa in (0.02, 0.05, 0.1, 0.2, 0.3):
        det_p, det_e, det_bh = [], [], []
        for k in pick:
            t = triplets[real_idx[k]]
            a, b, c = t["a"], t["b"], t["c"]
            Jk = {"A": ix(a, NONE, NONE), "B": ix(NONE, b, NONE), "C": ix(NONE, NONE, c), "AB": ix(a, b, NONE),
                  "AC": ix(a, NONE, c), "BC": ix(NONE, b, c), "ABC": ix(a, b, c)}
            Dk = {q: D[:, j].astype(np.float64) for q, j in Jk.items()}
            plant = kappa * Dk["ABC"].mean(0)[None, :]
            Ik = Dk["ABC"] + plant - Dk["AB"] - Dk["AC"] - Dk["BC"] + Dk["A"] + Dk["B"] + Dk["C"]
            ABCk = Dk["ABC"] + plant
            with torch.no_grad():
                It_ = torch.from_numpy(Ik.astype(np.float32)).to(dev)
                To = float(It_.mean(0).abs().sum().item())
                Tn = (St @ It_).abs().sum(1).cpu().numpy()
                bI_ = energy_boot(It_); bA_ = energy_boot(torch.from_numpy(ABCk.astype(np.float32)).to(dev))
                del It_
            pk = float((1 + (Tn >= To).sum()) / (N_FLIP + 1))
            pt_ = energy_np(Ik) / max(energy_np(ABCk), 1e-30)
            pl_ = float((Ik.mean(0) ** 2).sum()) / max(float((ABCk.mean(0) ** 2).sum()), 1e-30)
            ci_ = ci_basic(pt_, bI_ / np.maximum(bA_, 1e-30), pl_)
            det_p.append(pk); det_e.append(ci_[0] > 0)
        qk = bh(det_p)
        power[str(kappa)] = {
            "planted_energy_share_of_joint_effect": float(kappa ** 2 / (1 + kappa) ** 2),
            "n_triplets": len(pick),
            "detected_signflip_p05": int(sum(x < 0.05 for x in det_p)),
            "detected_signflip_BH05_within_64": int(sum(x < 0.05 for x in qk)),
            "detected_energy_share_ci_above_0": int(sum(det_e)),
        }
    log(f"power: {json.dumps(power)}")

    # ---------------------------------------------------------------- pairwise interactions (context)
    pair_rows = []
    for (X, Y, lx, ly) in (("A", "B", 0, 1), ("A", "C", 0, 2), ("B", "C", 1, 2)):
        fx = A if X == "A" else B
        fy = B if Y == "B" else C
        for u in fx:
            for v in fy:
                kx = [NONE, NONE, NONE]; kx[lx] = u
                ky = [NONE, NONE, NONE]; ky[ly] = v
                kxy = list(kx); kxy[ly] = v
                dX, dY, dXY = (D[:, ix(*k)].astype(np.float64) for k in (kx, ky, kxy))
                I2 = dXY - dX - dY
                I2m = I2.mean(0)
                with torch.no_grad():
                    It2 = torch.from_numpy(I2.astype(np.float32)).to(dev)
                    T2 = float(It2.mean(0).abs().sum().item())
                    T2n = (St @ It2).abs().sum(1).cpu().numpy()
                    del It2
                p2 = ttest_p(I2)
                exy = dXY.mean(0)
                pair_rows.append({
                    "pair": X + Y, "x": int(u), "y": int(v),
                    "share_I2": float(np.abs(I2m).sum() / max(np.abs(exy).sum(), 1e-30)),
                    "share_I2_noise_signflip": float(np.median(T2n) / max(np.abs(exy).sum(), 1e-30)),
                    "agg_p_signflip": float((1 + (T2n >= T2).sum()) / (N_FLIP + 1)) if T2 > 0 else 1.0,
                    "n_q05_within_t": int((bh(p2) < 0.05).sum()),
                    "r_pair": float(np.abs(exy).mean() / (np.abs(dX.mean(0)).mean() + np.abs(dY.mean(0)).mean())),
                    "r_pair_additive": float(np.abs(dX.mean(0) + dY.mean(0)).mean()
                                             / (np.abs(dX.mean(0)).mean() + np.abs(dY.mean(0)).mean())),
                })
    q2 = bh([r["agg_p_signflip"] for r in pair_rows])
    for r, q in zip(pair_rows, q2):
        r["agg_q_signflip_across_pairs"] = float(q)
    pair_summary = {
        "n_pairs": len(pair_rows),
        "pairs_agg_signflip_BH05": int(sum(r["agg_q_signflip_across_pairs"] < 0.05 for r in pair_rows)),
        "pairs_any_target_BH_within_t": int(sum(r["n_q05_within_t"] > 0 for r in pair_rows)),
        "share_I2_median": float(np.median([r["share_I2"] for r in pair_rows])),
        "share_I2_noise_median": float(np.median([r["share_I2_noise_signflip"] for r in pair_rows])),
        "by_pair_type": {pt: {"n": len(rr), "agg_BH05": int(sum(r["agg_q_signflip_across_pairs"] < 0.05 for r in rr)),
                              "share_I2_median": float(np.median([r["share_I2"] for r in rr])),
                              "share_I2_noise_median": float(np.median([r["share_I2_noise_signflip"] for r in rr])),
                              "r_pair_median": float(np.median([r["r_pair"] for r in rr])),
                              "r_pair_additive_median": float(np.median([r["r_pair_additive"] for r in rr]))}
                         for pt in ("AB", "AC", "BC") for rr in [[r for r in pair_rows if r["pair"] == pt]]},
    }
    log("pairwise interactions done")

    def frac_any(key, rr):
        return int(sum(1 for r in rr if r[key] > 0)), len(rr)

    # median ratios across triplets, bootstrap CI of the median (cells resampled jointly)
    def med_ci(key, point, plugin=None):
        """Median across triplets with a basic (pivot) bootstrap CI over cells; the cells are resampled
        jointly for all triplets. ``plugin``: per-triplet uncorrected values for noise-corrected stats."""
        bmed = np.median(boot_ratio[key], axis=0)
        pt = float(np.median(point))
        pl_ = float(np.median(plugin)) if plugin is not None else pt
        return {"median": pt, "ci95_of_median": ci_basic(pt, bmed, pl_),
                "ci95_percentile_of_median(biased; for reference)": ci95(bmed),
                "median_plugin_uncorrected": pl_ if plugin is not None else None,
                "dist": summarize(point)}
    ratio_summary = {
        "pairwise_AB": med_ci("AB", [r["r_AB"] for r in R_]),
        "pairwise_AC": med_ci("AC", [r["r_AC"] for r in R_]),
        "pairwise_BC": med_ci("BC", [r["r_BC"] for r in R_]),
        "pairwise_mean": med_ci("pair_mean", [r["r_pair_mean"] for r in R_]),
        "three_way": med_ci("three", [r["r_three"] for r in R_]),
        "marginal_C_given_AB": med_ci("marg", [r["marg_C_given_AB"] for r in R_]),
        "interaction_share": med_ci("Ishare", [r["share_I_over_ABC"] for r in R_]),
        "three_way_logits": med_ci("three_L", [r["r_three_logits"] for r in R_]),
        "pairwise_mean_logits": med_ci("pair_mean_L", [r["r_pair_mean_logits"] for r in R_]),
        "interaction_share_logits": med_ci("Ishare_L", [r["share_I_over_ABC_logits"] for r in R_]),
        "n_triplets_three_way_ci_above_1": int(sum(1 for r in R_ if r["ci_r_three"][0] > 1)),
        "n_triplets_three_way_ci_below_1": int(sum(1 for r in R_ if r["ci_r_three"][1] < 1)),
        "n_triplets_pair_mean_ci_below_1": int(sum(1 for r in R_ if r["ci_r_pair_mean"][1] < 1)),
    }
    ratio_summary["three_way_if_additive"] = med_ci("three_add", [r["r_three_additive"] for r in R_])
    ratio_summary["three_way_minus_additive"] = med_ci("three_minus_add",
                                                       [r["r_three"] - r["r_three_additive"] for r in R_])
    ratio_summary["pairwise_mean_if_additive"] = med_ci("pair_add", [r["r_pair_mean_additive"] for r in R_])
    ratio_summary["pairwise_mean_minus_additive"] = med_ci("pair_minus_add",
                                                           [r["r_pair_mean"] - r["r_pair_mean_additive"] for r in R_])
    ratio_summary["noise_corrected_rho3=E[ABC]/E[A+B+C]"] = med_ci("rho3", [r["rho3"] for r in R_],
                                                                    [r["rho3_plugin"] for r in R_])
    ratio_summary["noise_corrected_rho2_pair_mean"] = med_ci("rho2", [r["rho2_mean"] for r in R_],
                                                             [r["rho2_mean_plugin"] for r in R_])
    ratio_summary["noise_corrected_energy_share_I_ABC"] = med_ci("eshare_I", [r["eshare_I"] for r in R_],
                                                                 [r["eshare_I_plugin"] for r in R_])
    ratio_summary["noise_corrected_energy_share_all_interactions"] = med_ci(
        "eshare_tot", [r["eshare_tot"] for r in R_], [r["eshare_tot_plugin"] for r in R_])
    ratio_summary["noise_corrected_rho3_logits"] = med_ci("rho3_L", [r["rho3_logits"] for r in R_],
                                                          [r["rho3_logits_plugin"] for r in R_])
    ratio_summary["noise_corrected_energy_share_I_ABC_logits"] = med_ci(
        "eshare_I_L", [r["eshare_I_logits"] for r in R_], [r["eshare_I_logits_plugin"] for r in R_])
    ratio_summary["n_triplets_rho3_ci_below_1"] = int(sum(1 for r in R_ if r["ci_rho3"][1] < 1))
    ratio_summary["n_triplets_rho3_ci_above_1"] = int(sum(1 for r in R_ if r["ci_rho3"][0] > 1))
    ratio_summary["n_triplets_eshare_I_ci_above_0"] = int(sum(1 for r in R_ if r["ci_eshare_I"][0] > 0))
    ratio_summary["n_triplets_eshare_tot_ci_above_0"] = int(sum(1 for r in R_ if r["ci_eshare_tot"][0] > 0))
    ratio_summary["n_triplets_three_minus_additive_ci_below_0"] = int(sum(1 for r in R_ if r["ci_three_minus_additive"][1] < 0))
    ratio_summary["n_triplets_three_minus_additive_ci_above_0"] = int(sum(1 for r in R_ if r["ci_three_minus_additive"][0] > 0))
    # bootstrap of the across-triplet MEAN ratio as well
    for key in ("pair_mean", "three"):
        ratio_summary[f"{key}_mean_across_triplets_ci95_basic"] = ci_basic(
            float(np.mean([r["r_" + ("pair_mean" if key == "pair_mean" else "three")] for r in R_])),
            boot_ratio[key].mean(0))

    # spec / deployed classification pooled
    tot_incl = sum(r["spec_n_included"] for r in R_)
    cls = {
        "spec_targets_included(|d|>0.5 any condition)": tot_incl,
        "spec_superadditive_strict(|d_ABC|>sum)": sum(r["spec_super_strict"] for r in R_),
        "spec_superadditive_band(>1.1x)": sum(r["spec_super_band"] for r in R_),
        "spec_additive(0.9-1.1x)": sum(r["spec_additive"] for r in R_),
        "spec_subadditive(<0.9x)": sum(r["spec_sub"] for r in R_),
        "deployed_targets(|e_ABC|>0.01)": sum(r["dep_n_targets"] for r in R_),
        "deployed_superadditive(>1.1x)": sum(r["dep_super"] for r in R_),
        "n_triplets_with_any_spec_super_strict": frac_any("spec_super_strict", R_)[0],
        "n_triplets_with_any_deployed_super": frac_any("dep_super", R_)[0],
    }
    for k in list(cls):
        pass
    cls["frac_spec_super_strict"] = cls["spec_superadditive_strict(|d_ABC|>sum)"] / max(tot_incl, 1)
    w = np.array([r["spec_n_included"] for r in R_], dtype=float)
    cls["raw_mean_subadditive(<0.9x)_frac_of_spec_included_observed"] = float(np.average(
        [r["frac_sub_obs_spec_included"] or 0 for r in R_], weights=w))
    cls["raw_mean_subadditive(<0.9x)_frac_if_exactly_additive"] = float(np.average(
        [r["frac_sub_additive_spec_included"] or 0 for r in R_], weights=w))
    cls["frac_deployed_super"] = cls["deployed_superadditive(>1.1x)"] / max(cls["deployed_targets(|e_ABC|>0.01)"], 1)

    # significance of I_ABC
    sig = {
        "n_real_triplets": len(R_),
        "triplets_any_target_ci_excl0(uncorrected)": frac_any("n_ci_excl0", R_),
        "triplets_any_target_BH_within_triplet_t": frac_any("n_q05_within_t", R_),
        "triplets_any_target_BH_across_triplets_t": frac_any("n_q05_across_t", R_),
        "triplets_any_target_BH_within_triplet_boot": frac_any("n_q05_within_boot", R_),
        "triplets_any_target_BH_across_triplets_boot": frac_any("n_q05_across_boot", R_),
        "targets_ci_excl0_total": int(sum(r["n_ci_excl0"] for r in R_)),
        "targets_BH_across_t_total": int(sum(r["n_q05_across_t"] for r in R_)),
        "targets_BH_within_t_total": int(sum(r["n_q05_within_t"] for r in R_)),
        "targets_tested_total": len(R_) * d_sae,
        "n_ci_excl0_per_triplet": summarize([r["n_ci_excl0"] for r in R_]),
        "n_BH_across_t_per_triplet": summarize([r["n_q05_across_t"] for r in R_]),
        "expected_ci_excl0_per_triplet_if_null(5% of alive targets)": float(np.mean([0.05 * r["n_targets_alive_any"] for r in R_])),
        "logits_triplets_any_vocab_BH_within_t": frac_any("logit_n_q05_within_t", R_),
        "ci_excl0_per_triplet_signflip_null": summarize([np.mean(r["n_ci_excl0_signflip_null"]) for r in R_]),
        "ci_excl0_observed_over_null_median_ratio": float(np.median([r["n_ci_excl0"] / max(np.mean(r["n_ci_excl0_signflip_null"]), 1e-9) for r in R_])),
        "n_p05_uncorrected_t_per_triplet": summarize([r["n_p05_uncorrected_t"] for r in R_]),
        "expected_p05_uncorrected_t_if_null": float(np.mean([0.05 * r["n_targets_alive_any"] for r in R_])),
        "aggregate_signflip_test": {
            "statistic": "T = sum over 4,928 L11 targets of |mean over cells of I_ABC|; null = 999 random per-cell sign flips",
            "n_triplets_p05_uncorrected": int(sum(r["agg_p_signflip"] < 0.05 for r in R_)),
            "n_triplets_BH05_across_triplets": int(sum(r["agg_q_signflip_across_triplets"] < 0.05 for r in R_)),
            "p_distribution": summarize([r["agg_p_signflip"] for r in R_]),
            "T_obs_over_T_null_median": summarize([r["agg_T_obs"] / max(r["agg_T_null_median"], 1e-30) for r in R_]),
        },
        "aggregate_signflip_test_all_orders(d_ABC - d_A - d_B - d_C)": {
            "n_triplets_p05_uncorrected": int(sum(r["agg_p_signflip_tot"] < 0.05 for r in R_)),
            "n_triplets_BH05_across_triplets": int(sum(r["agg_q_signflip_tot_across_triplets"] < 0.05 for r in R_)),
            "T_obs_over_T_null_median": summarize([r["tot_T_obs_over_null"] for r in R_]),
        },
        "percell_nonadditivity(median over triplets of the median over cells)": {
            "I_ABC_energy_share": float(np.median([r["percell_share_I_median"] for r in R_])),
            "all_orders_energy_share": float(np.median([r["percell_share_tot_median"] for r in R_])),
            "I_ABC_energy_share_logits": float(np.median([r["percell_share_I_logits_median"] for r in R_])),
        },
        "interaction_share_observed_median": float(np.median([r["share_I_over_ABC"] for r in R_])),
        "interaction_share_signflip_noise_median": float(np.median([r["share_I_noise_signflip"] for r in R_])),
    }

    # noise-floor triplets
    noise = {}
    for kind in ("null_all", "null_A", "null_B", "null_C"):
        rr = [r for r in N_ if r["kind"] == kind]
        noise[kind] = {
            "n_triplets": len(rr),
            "max_abs_I_cellmean": max(r["I_max_abs"] for r in rr),
            "max_abs_I_percell": max(r["I_max_abs_percell"] for r in rr),
            "triplets_any_ci_excl0": frac_any("n_ci_excl0", rr),
            "triplets_any_BH_within_t": frac_any("n_q05_within_t", rr),
            "null_feature_live_n_active_max": None,
        }
    # does a null feature become active under upstream edits?
    nullset = {l: set(sel["null_features"][str(l)]) for l in SRC_LAYERS}
    mx = {}
    for j, k in enumerate(conds):
        for pos, l in enumerate(SRC_LAYERS):
            if k[pos] != NONE and k[pos] in nullset[l]:
                v = NA[:, j, pos]
                v = v[v >= 0]
                if v.size:
                    mx[l] = max(mx.get(l, 0), int(v.max()))
    noise["null_feature_live_n_active_max_by_layer"] = {str(l): mx.get(l) for l in SRC_LAYERS}

    # sanity checks
    real_pairs_AB = [(a, b) for a in A for b in B]
    ab_vs_b = [float(np.abs(effD[ix(a, b, NONE)] - effD[ix(NONE, b, NONE)]).max()) for a, b in real_pairs_AB]
    ab_vs_a = [float(np.abs(effD[ix(a, b, NONE)] - effD[ix(a, NONE, NONE)]).max()) for a, b in real_pairs_AB]
    abc_vs_c = [r["sanity"]["ABC_vs_C"] for r in R_]
    legacy = np.array([r["legacy_identity_eC_over_sum"] for r in R_])
    three = np.array([r["r_three"] for r in R_])
    single_eff = {f"L{l}": {str(f): float(eff_abs[ix(*(f if p == q else NONE for q in range(3)))])
                            for f in sel["features"][str(l)]}
                  for p, l in enumerate(SRC_LAYERS)}
    sanity = {
        "verify_stage": load_json(OUT / "verify.json")["verdict"] if (OUT / "verify.json").exists() else None,
        "data_integrity": integrity,
        "AB_vs_B_maxabs_over_targets": {"min_over_64_pairs": min(ab_vs_b), "median": float(np.median(ab_vs_b)),
                                        "n_pairs_equal(<1e-7)": int(sum(x < 1e-7 for x in ab_vs_b))},
        "AB_vs_A_maxabs_over_targets": {"min_over_64_pairs": min(ab_vs_a), "median": float(np.median(ab_vs_a))},
        "ABC_vs_C_maxabs_over_targets": {"min_over_512": min(abc_vs_c), "median": float(np.median(abc_vs_c)),
                                         "n_equal(<1e-7)": int(sum(x < 1e-7 for x in abc_vs_c))},
        "three_way_ratio_std_across_triplets": float(three.std(ddof=1)),
        "old_run_three_way_std": 0.0001,
        "corr_three_way_vs_legacy_identity": float(np.corrcoef(three, legacy)[0, 1]),
        "max_abs_three_way_minus_legacy_identity": float(np.abs(three - legacy).max()),
        "single_effect_eff_by_feature": single_eff,
        "single_effect_eff_std_by_layer": {k: float(np.std(list(v.values()), ddof=1)) for k, v in single_eff.items()},
    }

    counts = {
        "interventions": {
            "triplets_real": len(R_), "triplets_noise_floor": len(N_),
            "conditions_real_grid": {"single": 24, "pair": 192, "triple": 512, "total": 728},
            "conditions_total_incl_noise_floor": K,
            "cells": nC, "ablated_forward_passes": K * nC, "clean_forward_passes": nC,
        },
        "measurements": {
            "L11_targets_per_condition": d_sae, "logit_targets_per_condition": V,
            "triplet_x_L11_target_interaction_terms(real)": len(R_) * d_sae,
        },
    }
    # per-feature-type breakdown (annotated vs unannotated in each layer)
    det = sel["details"]
    ann_of = {(l, d["feature"]): d["annotated"] for l in SRC_LAYERS for d in det[str(l)]["chosen"]}
    by_ann = {}
    for r in R_:
        key = "".join("a" if ann_of[(l, f)] else "u" for l, f in zip(SRC_LAYERS, (r["a"], r["b"], r["c"])))
        by_ann.setdefault(key, []).append(r)
    ann_break = {k: {"n": len(v), "median_three_way": float(np.median([r["r_three"] for r in v])),
                     "median_share_I": float(np.median([r["share_I_over_ABC"] for r in v])),
                     "frac_any_BH_across": float(np.mean([r["n_q05_across_t"] > 0 for r in v]))}
                 for k, v in sorted(by_ann.items())}

    kl_by_order = {}
    for order in (1, 2, 3):
        js = [j for j, k in enumerate(conds) if sum(x != NONE for x in k) == order
              and all(x == NONE or x in sel["features"][str(l)] for x, l in zip(k, SRC_LAYERS))]
        kl_by_order[str(order)] = {"n_conditions": len(js), "kl_cellmean": summarize(KL[:, js].mean(0))}
    summary = {"counts": counts, "significance_I_ABC": sig, "redundancy_ratios": ratio_summary,
               "pairwise_interactions": pair_summary, "logit_effect_size_by_order": kl_by_order,
               "planted_interaction_power": power,
               "superadditivity": cls, "noise_floor": noise, "sanity": sanity,
               "by_annotation_pattern(L0,L5,L9; a=annotated,u=unannotated)": ann_break,
               "bootstrap": {"unit": "cell", "n_boot": N_BOOT, "seed": BOOT_SEED,
                             "method": "multinomial cell weights shared across all triplets and targets; percentile CIs for per-target I_ABC means (unbiased, linear); basic (pivot) CIs for every ratio and energy summary"},
               "tests": "per-target one-sample t-test over 20 cells (df=19) on per-cell I_ABC; BH q<0.05 "
                        "within triplet (4,928 targets) and across all 512 x 4,928; percentile bootstrap CI "
                        "(1,000 reps over cells) for the uncorrected 'CI excludes 0' count; bootstrap p "
                        "(2*min tail, +1 smoothing) as a second p-value",
               "deployed_v2_for_comparison": {"n_triplets": 4, "pair_mean": 0.3235, "three_way": 0.1896,
                                              "three_way_std": 0.0001, "superadditive": "0 of ~2,980 targets"},
               "wall_seconds": round(time.time() - t0, 1)}
    H.write_json(out_dir / "summary.json", summary)
    H.write_json(out_dir / "per_triplet.json", rows)
    H.write_json(out_dir / "per_pair.json", pair_rows)
    np.savez_compressed(out_dir / "interaction_terms_real.npz",
                        triplets=np.array([[r["a"], r["b"], r["c"]] for r in R_]),
                        I_mean=I_store_mean[real_idx], ci_lo=lo_store[real_idx], ci_hi=hi_store[real_idx],
                        p_t=p_store[real_idx], q_across_t=Q)
    if out_dir == OUT:
        update_run_config("chunks", {"stage": "analyze", "start": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                     "wall_seconds": round(time.time() - t0, 1)})
    log(json.dumps({"significance": sig, "ratios_median": {k: v["median"] if isinstance(v, dict) else v
                                                           for k, v in ratio_summary.items()}}, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "verify", "run", "recheck", "analyze"])
    ap.add_argument("--max-minutes", type=float, default=8.0)
    ap.add_argument("--n-cells", type=int, default=0, help="analyze: use only the first N cells (testing)")
    ap.add_argument("--device", default="", help="analyze: torch device for the bootstrap (default mps)")
    ap.add_argument("--out-dir", default="", help="analyze: write outputs here instead of outputs/v2_triplets")
    args = ap.parse_args()
    {"select": stage_select, "verify": stage_verify, "run": stage_run, "recheck": stage_recheck,
     "analyze": stage_analyze}[args.stage](args)


if __name__ == "__main__":
    main()
