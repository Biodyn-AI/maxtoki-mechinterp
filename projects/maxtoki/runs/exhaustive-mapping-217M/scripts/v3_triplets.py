"""v3 triplets (item V3-4): the 8 x 8 x 8 triplet grid redone with the v3 SAEs and correctly encoded inputs.

Why a v3
--------
v2 (scripts/v2_triplets.py, V2_TRIPLETS_REPORT.md) fixed the hooks but used
  * inputs ranked from log1p(CP10k) instead of counts (checks/INPUT_ENCODING_AUDIT.md), and
  * the deployed SAEs, which were trained on those wrong inputs (V3_SAE_REPORT.md).
This script repeats the v2 design with
  * cells from setup/inputs_v3.load_k562_control_cells (same 20 rows as v2, right gene order);
    every cell passes inputs_v3.check_encoding before every forward pass (abort otherwise);
  * the v3 SAEs (runs/sae-atlas-217M/outputs/v3_sae) at L0, L5, L9 (edits) and L11 (read-out);
  * setup/hooks_v2.py for every edit (unchanged).

Feature choice (fixed before any ablation; written to selection.json and run_config.json by `select`):
  per layer, candidates = v3 SAE features active (z > 0) in >= 1% of all tokens of the 20 clean cells;
  the log10 activation frequency of the candidates is cut into 8 equal-count bins (octiles); one
  feature is drawn at random from each bin (numpy default_rng(20261001 + layer)). There are no v3
  annotations used for the choice (the v2 rule "4 annotated + 4 unannotated" is replaced). A post-hoc
  label from the v3 annotation built by item V3-3 is recorded but not used to choose.
  Null features (noise floor): 2 per layer drawn at random from features with zero activations in
  the 20 clean cells.

Stages (run in this order; every model stage is resumable or short):
  select     : encoding checks, clean pass, frequencies, feature choice.          (model)
  verify     : zero-delta, clean repeat, fast path == full path, float32 vs float64. (model)
  run        : the ablation grid; resumable; repeat with --max-minutes until DONE. (model)
  recheck    : 2 stored conditions per group recomputed from scratch, CPU float64.  (model)
  readout    : per-position non-additivity of h11 / SAE pre-activation / code / logits. (model)
  analyze    : interaction terms, bootstrap, BH, power, ratios; resumable.           (no model)
  power      : per-target power, proportional-plant aggregate power, BH effect counts. (no model)
  crosscheck : key numbers re-derived with independent code (CPU float64).        (no model)
  compare    : side-by-side numbers v3 vs v2 vs deployed.                          (no model)

Usage (from projects/maxtoki):
  OMP_NUM_THREADS=4 .venv/bin/python runs/exhaustive-mapping-217M/scripts/v3_triplets.py <stage> [--max-minutes 7]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import torch

torch.set_num_threads(4)

PROJ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJ / "setup"))
import hooks_v2 as H  # noqa: E402
import inputs_v3 as I  # noqa: E402

RUN = PROJ / "runs/exhaustive-mapping-217M"
OUT = RUN / "outputs/v3_triplets"
CELL_DIR = OUT / "cells"
SRC = OUT                    # read directory for analyze/power/crosscheck (= OUT except in the v2 replay test)
V2 = RUN / "outputs/v2_triplets"
V2F = RUN / "outputs/v2_triplets_followup"
SAE_DIR = PROJ / "runs/sae-atlas-217M/outputs/v3_sae"
DEP_SAE_DIR = PROJ / "runs/sae-atlas-217M/outputs/phase1"
ANN_V3_DIR = PROJ / "runs/circuit-tracing-217M/outputs/v3_circuit/annotation"
SCRIPT_PATH = Path(__file__).resolve()

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DATASET = "k562"
MAX_LEN = 2048
SRC_LAYERS = (0, 5, 9)
TGT = 11
N_CELLS = 20
CELL_POOL = 100
CELL_POOL_SEED = 42          # same pool as v2 / experiment2_rerun.py
SEL_SEED = 20261001          # feature selection: default_rng(SEL_SEED + layer)
BOOT_SEED = 20261001
N_BOOT = 1000
N_FLIP = 999
N_PER_LAYER = 8
N_BINS = 8                   # one feature per octile of log10 frequency
N_NULL = 2
MIN_FREQ = 0.01
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
    """Set run_config[section] (single-entry sections) or append ``entry`` to run_config[section]."""
    p = OUT / "run_config.json"
    cfg = load_json(p) if p.exists() else {}
    single = ("env", "selection", "cells", "seeds", "verify", "input_encoding", "design", "inputs")
    if section in single:
        cfg[section] = entry
    else:
        if isinstance(entry, dict):
            entry = dict(entry, script_sha256=H.sha256_file(SCRIPT_PATH))
        cfg.setdefault(section, []).append(entry)
    cfg["item"] = "V3-4: triplet grid with v3 SAEs and correctly encoded inputs"
    cfg["script"] = str(SCRIPT_PATH)
    cfg["script_sha256"] = H.sha256_file(SCRIPT_PATH)
    H.write_json(p, cfg)


def free():
    H.free_device_cache()


def tokens_sha(toks):
    return I.array_sha256(np.concatenate([np.asarray(t, dtype=np.int64) for t in toks]))


def load_cells_checked(stage: str):
    """Stored tokens + counts, checked against the hashes logged by `select`, then the inputs_v3
    encoding check on every cell. Raises I.EncodingError (abort) on any failure."""
    meta = load_json(OUT / "cells.json")
    d = np.load(OUT / "cells_tokens.npz", allow_pickle=False)
    toks = [d[f"cell_{i:02d}"] for i in range(len(meta["rows"]))]
    counts = np.load(OUT / "cells_counts.npz", allow_pickle=False)["counts"]
    if I.array_sha256(counts) != meta["counts_sha256"]:
        raise I.EncodingError(f"{stage}: stored counts differ from the counts logged at selection")
    if tokens_sha(toks) != meta["tokens_sha256"]:
        raise I.EncodingError(f"{stage}: stored tokens differ from the tokens logged at selection")
    chk = I.assert_encoding_batch(counts, toks, DATASET, MAX_LEN, label=f"{stage}: all {len(toks)} cells")
    return toks, counts, meta, chk


def check_cell(counts_row, tok, label):
    """Encoding check right before a forward pass on one cell (batch size 1)."""
    r = I.check_encoding(counts_row, tok, DATASET, MAX_LEN)
    if not r["pass"]:
        raise I.EncodingError(f"{label}: input encoding check failed: {r['problems']}")
    return r


# =============================================================================
# forward machinery (as v2, with the v3 SAEs)
# =============================================================================
class Runner:
    def __init__(self, device=DEVICE):
        self.device = device
        log(f"loading model on {device}")
        torch.manual_seed(0)
        self.xt = H.load_model(device)
        self.model = self.xt.model
        self.nb = H.n_blocks(self.model)
        self.saes = H.load_saes(list(SRC_LAYERS) + [TGT], device=device, sae_dir=SAE_DIR)
        for l, s in self.saes.items():
            assert Path(s.path).parent.parent == SAE_DIR, s.path
        self.layer_kwargs = None

    def _kw_hook(self):
        store = {}

        def pre(mod, args, kwargs):
            store["kw"] = dict(kwargs)
            return None
        h = self.model.model.layers[0].register_forward_pre_hook(pre, with_kwargs=True)
        return store, h

    @torch.no_grad()
    def clean(self, ids):
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
        c = {"h0": ed.captured[0]["pre"], "h5": ed.captured[5]["pre"], "h9": ed.captured[9]["pre"],
             "h11": ed.captured[TGT]["pre"]}
        c["z11"] = self.saes[TGT].encode(c["h11"][0])
        c["logits"] = logits
        c["logp"] = torch.log_softmax(logits, dim=-1)
        c["p"] = c["logp"].exp()
        return c

    @torch.no_grad()
    def forward_from(self, start, h_in, ids):
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
        """Position means are dim-0 sums (a full-tensor MPS .sum() was found unreliable in v2)."""
        T = logits.shape[0]
        z = self.saes[TGT].encode(h11)
        dz = (z - clean["z11"]).sum(0) / T
        dl = logits - clean["logits"]
        dlog = dl.sum(0) / T
        logp = torch.log_softmax(logits, dim=-1)
        kl = (clean["p"] * (clean["logp"] - logp)).sum(-1).sum(0) / T
        res = {"dz": dz.to("cpu", torch.float32).numpy(), "dlog": dlog.to("cpu", torch.float32).numpy(),
               "kl": float(kl.item())}
        del z, dl, logp
        return res


def ablate_edits(a, b, c, start):
    edits = []
    if a != NONE and start <= 0:
        edits.append(H.Ablate(0, [int(a)]))
    if b != NONE and start <= 5:
        edits.append(H.Ablate(5, [int(b)]))
    if c != NONE and start <= 9:
        edits.append(H.Ablate(9, [int(c)]))
    return edits


# =============================================================================
# design (identical to v2)
# =============================================================================
def build_design(sel):
    A, B, C = sel["features"]["0"], sel["features"]["5"], sel["features"]["9"]
    NA, NB, NC = sel["null_features"]["0"], sel["null_features"]["5"], sel["null_features"]["9"]
    triplets = []
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
def read_barcodes(h5, rows):
    import h5py
    with h5py.File(h5, "r") as f:
        obs = f["obs"]
        for key in ("_index", "cell_barcode", "barcode", "index"):
            if key in obs:
                try:
                    ds = obs[key]
                    return [(lambda x: x.decode() if isinstance(x, bytes) else str(x))(ds[int(r)]) for r in rows]
                except Exception:
                    pass
    return None


def stage_select(args):
    OUT.mkdir(parents=True, exist_ok=True)
    CELL_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    free_gb = H.check_memory(MIN_FREE_GB)
    log(f"free memory {free_gb:.1f} GB")
    # ---------------------------------------------------------------- cells and encoding checks
    cells, rows, h5, info = I.load_k562_control_cells(N_CELLS, pool=CELL_POOL, seed=CELL_POOL_SEED,
                                                      max_len=MAX_LEN, return_info=True)
    assert len(cells) == N_CELLS, len(cells)
    counts, row_checks = I.read_counts(DATASET, rows)
    toks = [c.token_ids.astype(np.int64) for c in cells]
    chk = I.assert_encoding_batch(counts, toks, DATASET, MAX_LEN, label="select: all 20 cells")
    v2meta = load_json(V2 / "cells.json")
    same_rows_as_v2 = rows == v2meta["rows"]
    if not same_rows_as_v2:
        raise SystemExit(f"rows differ from v2: {rows} vs {v2meta['rows']}")
    v2t = np.load(V2 / "cells_tokens.npz")
    old = [v2t[f"cell_{i:02d}"].astype(np.int64) for i in range(N_CELLS)]
    rebuilt_old = I.deployed_tokenize_X(DATASET, rows, MAX_LEN)
    old_rebuilt_equal = [bool(np.array_equal(o, r.token_ids)) for o, r in zip(old, rebuilt_old)]
    old_fail = [not I.check_encoding(counts[i], old[i], DATASET, MAX_LEN)["pass"] for i in range(N_CELLS)]
    new_eq_old = [bool(np.array_equal(old[i], toks[i])) for i in range(N_CELLS)]
    vm = I.get_var_map(DATASET)
    old_full = I.deployed_tokenize_X(DATASET, rows, 10 ** 9)
    agree = []
    for i in range(N_CELLS):
        new_order = I.full_order(counts[i], vm)
        old_order = old_full[i].token_ids[1:-1]
        agree.append(I.order_agreement(old_order, new_order, MAX_LEN))
    agree_summary = {k: {"mean": float(np.mean([a[k] for a in agree])), "min": float(np.min([a[k] for a in agree])),
                         "max": float(np.max([a[k] for a in agree]))}
                     for k in ("spearman_full", "top200_overlap", "kept_set_overlap", "same_position_share")}
    agree_summary["n_cells_cut_at_2046"] = int(sum(a["truncated_b"] for a in agree))
    barcodes = read_barcodes(h5, rows)
    np.savez(OUT / "cells_tokens.npz", **{f"cell_{i:02d}": t for i, t in enumerate(toks)})
    np.savez(OUT / "cells_counts.npz", counts=counts, rows=np.array(rows))
    cell_meta = {
        "dataset": DATASET, "dataset_path": h5, "rows": rows, "barcodes": barcodes,
        "n_tokens": [int(len(t)) for t in toks],
        "pool": "inputs_v3.load_k562_control_cells(n_cells=20, pool=100, seed=42, max_len=2048): K562 "
                "non-targeting controls, 100-cell pool drawn with numpy default_rng(42), sorted rows, first 20 "
                "that tokenize (same rows as v2_triplets)",
        "counts_sha256": I.array_sha256(counts), "tokens_sha256": tokens_sha(toks),
        "same_rows_as_v2": same_rows_as_v2,
        "v2_tokens_rebuilt_by_deployed_path": f"{sum(old_rebuilt_equal)} of {N_CELLS}",
        "v2_tokens_fail_encoding_check": f"{sum(old_fail)} of {N_CELLS}",
        "v3_tokens_identical_to_v2": f"{sum(new_eq_old)} of {N_CELLS}",
        "order_agreement_v2_vs_v3(full order, inputs_v3.order_agreement)": agree_summary,
        "order_agreement_per_cell": agree,
    }
    H.write_json(OUT / "cells.json", cell_meta)
    enc = I.encoding_record(DATASET, rows=rows, check_summary=chk, counts_sha256=cell_meta["counts_sha256"])
    enc["count_row_checks_all_pass"] = all(c["expected_ok"] for c in row_checks)
    enc["count_row_unit_max_dev"] = max(c["max_dev"] for c in row_checks)
    enc["loader_info"] = info
    update_run_config("input_encoding", enc)
    log(f"cells ok: rows == v2 {same_rows_as_v2}; v2 tokens fail check {sum(old_fail)}/20; "
        f"Spearman old vs new {agree_summary['spearman_full']['mean']:.3f}")

    # ---------------------------------------------------------------- clean pass with v3 SAEs
    R = Runner()
    d_sae = R.saes[0].W_dec.shape[1]
    cnt = {l: np.zeros(d_sae, dtype=np.int64) for l in (0, 5, 9, 11)}
    bos_cnt = {l: np.zeros(d_sae, dtype=np.int64) for l in (0, 5, 9)}
    cells_active = {l: np.zeros(d_sae, dtype=np.int64) for l in (0, 5, 9)}
    total = 0
    z11_cellmean = []
    for i, t in enumerate(toks):
        H.check_memory(MIN_FREE_GB)
        check_cell(counts[i], t, f"select clean cell {i}")
        ids = torch.from_numpy(t[None, :]).to(R.device)
        cl = R.clean(ids)
        for l, key in ((0, "h0"), (5, "h5"), (9, "h9")):
            z = R.saes[l].encode(cl[key][0])
            act = (z > 0)
            cnt[l] += act.sum(0).cpu().numpy()
            bos_cnt[l] += act[0].cpu().numpy().astype(np.int64)
            cells_active[l] += act.any(0).cpu().numpy().astype(np.int64)
            del z, act
        cnt[11] += (cl["z11"] > 0).sum(0).cpu().numpy()
        z11_cellmean.append(cl["z11"].cpu().double().mean(0).numpy())
        total += ids.shape[1]
        del cl
        free()
        log(f"clean cell {i} ({ids.shape[1]} tokens)")
    freq = {l: cnt[l] / total for l in cnt}
    np.savez(OUT / "clean_frequencies.npz", total_tokens=total, **{f"freq_L{l}": freq[l] for l in freq},
             **{f"bos_cells_L{l}": bos_cnt[l] for l in bos_cnt},
             **{f"n_cells_active_L{l}": cells_active[l] for l in cells_active},
             clean_z11_cellmean=np.stack(z11_cellmean))

    # ---------------------------------------------------------------- feature choice
    selection = {"features": {}, "null_features": {}, "details": {}, "rule": (
        "Per layer: candidates = v3 SAE features active (z>0) in >= 1% of all tokens of the 20 clean cells "
        "(all positions incl. <bos>/<eos>). Candidate log10(frequency) cut into 8 equal-count bins (octiles, "
        "numpy quantile); one feature drawn at random from each bin with numpy default_rng(20261001 + layer), "
        "bins in order 0 (rarest) .. 7 (commonest). If a bin had no unused candidate the nearest bin would be "
        "used (logged). Null features: 2 drawn with the same generator from features with zero activations "
        "in the 20 clean cells; if a layer has fewer than 2 such features (v3 L5: none), its 2 rarest features "
        "(fewest active positions, rarest first) are used as near-null features instead. No annotation is used "
        "for the choice; the v3 annotation label (item V3-3, circuit-tracing-217M/outputs/v3_circuit/annotation) "
        "is recorded afterwards for description only."),
        "rule_amendment": ("The first select call (2026-10-02 13:38) stopped before choosing because L5 had no "
                           "feature with zero activations. The near-null fallback was added before any ablation "
                           "was run. The octile rule for the 24 real features was not changed."),
        "selected_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed_base": SEL_SEED}
    import pandas as pd
    v2sel = load_json(V2 / "selection.json")
    for l in SRC_LAYERS:
        rng = np.random.default_rng(SEL_SEED + l)
        fr = freq[l]
        cand = np.where(fr >= MIN_FREQ)[0]
        logf = np.log10(fr[cand])
        edges = np.quantile(logf, np.linspace(0, 1, N_BINS + 1))
        bins = np.clip(np.searchsorted(edges[1:-1], logf, side="right"), 0, N_BINS - 1)
        chosen, det = [], []
        for bi in range(N_BINS):
            order = sorted(range(N_BINS), key=lambda x: (abs(x - bi), x))
            pick, used = None, None
            for bj in order:
                pool = [int(p) for p in cand[bins == bj] if int(p) not in chosen]
                if pool:
                    pick = int(rng.choice(np.array(pool)))
                    used = bj
                    break
            assert pick is not None
            chosen.append(pick)
            det.append({"feature": pick, "freq_bin_target": bi, "freq_bin_used": used,
                        "freq_20cells": float(fr[pick]), "n_cells_active_of_20": int(cells_active[l][pick]),
                        "active_at_bos_cells": int(bos_cnt[l][pick])})
        zero = np.where(cnt[l] == 0)[0]
        if len(zero) >= N_NULL:
            nulls = [int(x) for x in rng.choice(zero, size=N_NULL, replace=False)]
            null_kind = "exact: zero activations in the 20 clean cells"
        else:
            # v3 L5 SAE: every feature fires somewhere in these cells. Use the N_NULL rarest features
            # (fewest active positions; ties broken by feature id), rarest first.
            order = np.lexsort((np.arange(d_sae), cnt[l]))
            nulls = [int(x) for x in order[:N_NULL]]
            null_kind = (f"near-null: no feature has zero activations at L{l}; the {N_NULL} rarest features are used "
                         f"(active positions {[int(cnt[l][x]) for x in nulls]} of {total})")
        log(f"L{l} nulls: {null_kind}")
        selection["features"][str(l)] = chosen
        selection["null_features"][str(l)] = nulls
        selection["details"][str(l)] = {
            "n_candidates": int(len(cand)), "n_alive_20cells": int((cnt[l] > 0).sum()),
            "n_zero_20cells": int(len(zero)),
            "log10_freq_bin_edges": [float(x) for x in edges],
            "candidates_per_bin": {str(b): int((bins == b).sum()) for b in range(N_BINS)},
            "chosen": det, "null_kind": null_kind,
            "null": [{"feature": n, "active_positions_20cells": int(cnt[l][n]),
                      "n_cells_active_of_20": int(cells_active[l][n])} for n in nulls]}
        log(f"L{l}: {len(cand)} candidates; chosen {chosen}; nulls {nulls}")
    # selection is fixed from here on; write it before the post-hoc descriptions
    H.write_json(OUT / "selection.json", selection)
    update_run_config("selection", {"features": selection["features"], "null_features": selection["null_features"],
                                    "null_kind": {l: selection["details"][l]["null_kind"] for l in selection["details"]},
                                    "rule": selection["rule"], "rule_amendment": selection["rule_amendment"],
                                    "selected_at": selection["selected_at"]})

    # ---------------------------------------------------------------- post-hoc descriptions (not used to choose)
    dep = H.load_saes(SRC_LAYERS, device="cpu", sae_dir=DEP_SAE_DIR)
    v3cpu = H.load_saes(SRC_LAYERS, device="cpu", sae_dir=SAE_DIR)
    for l in SRC_LAYERS:
        enr = pd.read_csv(ANN_V3_DIR / f"layer_{l:02d}/significant_enrichments.csv")
        codes = np.load(SAE_DIR / f"codes/layer_{l:02d}_topk.npz")
        idx, val = codes["idx"], codes["val"]
        atlas_cnt = np.bincount(idx[val > 0].astype(np.int64), minlength=d_sae)
        atlas_freq = atlas_cnt / idx.shape[0]
        Wv = v3cpu[l].W_dec.detach().numpy()
        Wd = dep[l].W_dec.detach().numpy()
        Wv2 = Wd[:, v2sel["features"][str(l)]]
        for d in selection["details"][str(l)]["chosen"] + selection["details"][str(l)]["null"]:
            f = d["feature"]
            terms = enr[enr["feature_id"] == f].sort_values("q_bh")["term"].tolist()
            d["posthoc_v3_annotated"] = bool(len(terms) > 0)
            d["posthoc_v3_top_terms"] = terms[:3]
            d["posthoc_v3_n_terms"] = int(len(terms))
            d["freq_v3_sae_training_500cells"] = float(atlas_freq[f])
            cs = np.abs(Wv[:, f] @ Wd)
            d["best_abs_cos_to_deployed_decoder"] = float(cs.max())
            d["best_deployed_match_id"] = int(cs.argmax())
            d["best_abs_cos_to_v2_chosen_features_same_layer"] = float(np.abs(Wv[:, f] @ Wv2).max())
        del idx, val, codes
    selection["posthoc_note"] = ("posthoc_* fields, freq_v3_sae_training_500cells and the cosine matches were "
                                 "added after the selection was written; they were not used to choose.")
    H.write_json(OUT / "selection.json", selection)
    update_run_config("env", H.env_info(DEVICE))
    update_run_config("seeds", {"cell_pool_seed": CELL_POOL_SEED, "selection_seed_base": SEL_SEED,
                                "bootstrap_seed": BOOT_SEED, "n_boot": N_BOOT, "signflip_seed": BOOT_SEED + 1,
                                "n_signflip": N_FLIP, "power_pick_seed": BOOT_SEED + 2,
                                "readout_pick_seed": SEL_SEED + 20, "crosscheck_signflip_seed": 20261099,
                                "torch_manual_seed": 0})
    update_run_config("cells", {k: cell_meta[k] for k in ("dataset_path", "rows", "barcodes", "n_tokens", "pool",
                                                          "counts_sha256", "tokens_sha256")})
    update_run_config("inputs", {"sae_dir": str(SAE_DIR),
                                 "sae_sha256": {str(l): H.sha256_file(SAE_DIR / f"layer_{l:02d}/sae_final.pt")
                                                for l in list(SRC_LAYERS) + [TGT]},
                                 "hooks_v2_sha256": H.sha256_file(Path(H.__file__)),
                                 "inputs_v3_sha256": I.sha256_file(Path(I.__file__)),
                                 "posthoc_annotation_dir": str(ANN_V3_DIR)})
    update_run_config("chunks", {"stage": "select", "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
                                 "wall_seconds": round(time.time() - t0, 1), "free_gb": free_gb,
                                 "encoding_check": chk})
    log(f"select done in {time.time() - t0:.0f}s")


# =============================================================================
# stage: verify
# =============================================================================
def stage_verify(args):
    torch.manual_seed(0)
    t0 = time.time()
    free_gb = H.check_memory(MIN_FREE_GB)
    sel = load_json(OUT / "selection.json")
    toks, counts, meta, chk = load_cells_checked("verify")
    triplets, conds, a_groups = build_design(sel)
    R = Runner()
    rng = np.random.default_rng(7)
    res = {"zero_delta": [], "fast_vs_full": [], "float64_check": [], "clean_repeat": []}
    for ci in range(2):
        check_cell(counts[ci], toks[ci], f"verify cell {ci}")
        ids = torch.from_numpy(toks[ci][None, :]).to(R.device)
        cl = R.clean(ids)
        cl2 = R.clean(ids)
        res["clean_repeat"].append({
            "cell": ci, "max_abs_logits": float((cl["logits"] - cl2["logits"]).abs().max().item()),
            "max_abs_z11": float((cl["z11"] - cl2["z11"]).abs().max().item())})
        del cl2
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
        sample_types = {}
        for cnd in conds:
            key = tuple(int(x != NONE) for x in cnd)
            sample_types.setdefault(key, []).append(cnd)
        picks = []
        for key, lst in sorted(sample_types.items()):
            idx = rng.choice(len(lst), size=min(2, len(lst)), replace=False)
            picks += [lst[i] for i in idx]
        for (a, b, c) in picks:
            m_full, _ = R.run_condition(ids, 0, None, ablate_edits(a, b, c, 0), [TGT], cl)
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
                "max_abs_dz": float(np.abs(m_full["dz"]).max()), "kl_full": m_full["kl"], "kl_fast": m_fast["kl"]})
            free()
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
                                     "max_abs_dz": float(np.abs(dz64).max())})
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
        "encoding_check": chk,
    }
    res["wall_seconds"] = round(time.time() - t0, 1)
    H.write_json(OUT / "verify.json", res)
    update_run_config("verify", res["verdict"])
    update_run_config("chunks", {"stage": "verify", "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
                                 "wall_seconds": res["wall_seconds"], "free_gb": free_gb, "encoding_check": chk})
    log(json.dumps(res["verdict"], indent=1, default=str))


# =============================================================================
# stage: run
# =============================================================================
def group_path(ci, gi):
    return SRC / "cells" / f"cell{ci:02d}_g{gi:02d}.npz"


def stage_run(args):
    torch.manual_seed(0)
    t_start = time.time()
    budget = args.max_minutes * 60.0
    sel = load_json(OUT / "selection.json")
    toks, counts, meta, chk = load_cells_checked("run")
    triplets, conds, a_groups = build_design(sel)
    cond_set = set(conds)
    todo = [(ci, gi) for ci in range(len(toks)) for gi in range(len(a_groups)) if not group_path(ci, gi).exists()]
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
    min_free = free_gb
    try:
        for (ci, gi) in todo:
            est = max(durations) if durations else 60.0
            if time.time() - t_start + est > budget:
                log("time budget reached; stopping cleanly")
                break
            min_free = min(min_free, H.check_memory(MIN_FREE_GB))
            tg = time.time()
            ids = torch.from_numpy(toks[ci][None, :]).to(R.device)
            if cur_ci != ci:
                cl = None
                free()
                check_cell(counts[ci], toks[ci], f"run cell {ci}")
                cl = R.clean(ids)
                cur_ci = ci
                clean_sig = np.array([float(cl["z11"].cpu().double().sum()), float(cl["logits"].cpu().double().sum())])
            a = a_groups[gi]
            recs = {}
            if a == NONE:
                h5a, h9a = cl["h5"], cl["h9"]
            else:
                m, ed = R.run_condition(ids, 0, None, ablate_edits(a, NONE, NONE, 0), [5, 9, TGT], cl)
                recs[(a, NONE, NONE)] = m
                h5a, h9a = ed.captured[5]["pre"], ed.captured[9]["pre"]
                del ed
            for (x, y, z) in conds:
                if x == a and y == NONE and z != NONE:
                    m, ed = R.run_condition(ids, 9, h9a, ablate_edits(x, y, z, 9), [TGT], cl)
                    recs[(x, y, z)] = m
                    del ed
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
            na = np.full((len(keys), 3), -1, dtype=np.int64)
            for i, (x, y, z) in enumerate(keys):
                if x != NONE:
                    na[i, 0] = recs[(x, NONE, NONE)]["n_active"].get((0, int(x)), -2)
                if y != NONE:
                    na[i, 1] = recs[(x, y, NONE)]["n_active"].get((5, int(y)), -2)
                if z != NONE:
                    na[i, 2] = recs[(x, y, z)]["n_active"].get((9, int(z)), -2)
            np.savez(group_path(ci, gi), conds=np.array(keys, dtype=np.int64),
                     dz=np.stack([recs[k]["dz"] for k in keys]).astype(np.float32),
                     dlog=np.stack([recs[k]["dlog"] for k in keys]).astype(np.float32),
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
                                 "min_free_gb_seen": min_free, "groups_done": done_groups, "groups_left": left,
                                 "encoding_check": chk})
    log(f"{status}: {left} groups left")


# =============================================================================
# stage: recheck (independent recomputation, full model path, CPU float64 reductions)
# =============================================================================
@torch.no_grad()
def measure_cpu64(logits, h11, clean64, sae11):
    z = sae11.encode(h11).cpu().double()
    lg = logits.cpu().double()
    dz = (z - clean64["z"]).mean(0).numpy()
    dl = lg - clean64["lg"]
    logp = torch.log_softmax(lg, dim=-1)
    kl = float((clean64["p"] * (clean64["logp"] - logp)).sum(-1).mean())
    return {"dz": dz, "dlog": dl.mean(0).numpy(), "kl": kl, "max_abs_dlogit": float(dl.abs().max())}


def stage_recheck(args):
    torch.manual_seed(0)
    t_start = time.time()
    budget = args.max_minutes * 60.0
    rdir = OUT / "recheck"
    rdir.mkdir(parents=True, exist_ok=True)
    sel = load_json(OUT / "selection.json")
    toks, counts, meta, chk = load_cells_checked("recheck")
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
            check_cell(counts[ci], toks[ci], f"recheck cell {ci}")
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
                                 "cells_done": done, "cells_left": left, "encoding_check": chk})
    log(("DONE" if left == 0 else "PARTIAL") + f": {left} cells left")


# =============================================================================
# stage: readout (per-position non-additivity; as v2_triplets_followup readout)
# =============================================================================
READ_CELLS = 4
READ_TRIPLETS = 4


@torch.no_grad()
def stage_readout(args):
    t_start = time.time()
    rdir = OUT / "readout"
    rdir.mkdir(parents=True, exist_ok=True)
    sel = load_json(OUT / "selection.json")
    toks, counts, meta, chk = load_cells_checked("readout")
    triplets, conds, a_groups = build_design(sel)
    real = [t for t in triplets if t["kind"] == "real"]
    rng = np.random.default_rng(SEL_SEED + 20)
    picks = sorted(rng.choice(len(real), size=READ_TRIPLETS, replace=False).tolist())
    todo = [ci for ci in range(READ_CELLS) if not (rdir / f"cell{ci:02d}.json").exists()]
    if not todo:
        log("DONE (readout complete)")
        summarize_readout(rdir)
        return
    free_gb = H.check_memory(MIN_FREE_GB)
    torch.manual_seed(0)
    xt = H.load_model(DEVICE)
    model = xt.model
    saes = H.load_saes([0, 5, 9, TGT], device=DEVICE, sae_dir=SAE_DIR)
    s11 = saes[TGT]
    Wlm = model.lm_head.weight
    assert model.lm_head.bias is None
    budget = args.max_minutes * 60
    durations, done = [], []
    for ci in todo:
        est = max(durations) if durations else 120.0
        if time.time() - t_start + est > budget:
            log("time budget reached; stopping cleanly")
            break
        try:
            H.check_memory(MIN_FREE_GB)
        except H.MemoryGuardError as e:
            log(f"STOP (memory guard): {e}")
            break
        tc = time.time()
        check_cell(counts[ci], toks[ci], f"readout cell {ci}")
        ids = torch.from_numpy(toks[ci][None, :]).to(DEVICE)

        def run(edits):
            with H.ResidualEditor(model, saes, edits=edits, capture=[TGT], capture_device=DEVICE) as ed:
                o = model(ids, use_cache=False, return_dict=True)
                del o
            return ed.captured[TGT]["pre"][0]

        h0 = run([])
        u0 = s11.sae.W_enc(h0 - s11.mu)
        z0 = s11.encode(h0)
        act0 = (z0 > 0).cpu()
        out = {"cell": ci, "row": meta["rows"][ci], "triplets": []}
        g_cache = {}
        for pk in picks:
            t = real[pk]
            a, b, c = t["a"], t["b"], t["c"]
            names = {"A": (a, NONE, NONE), "B": (NONE, b, NONE), "C": (NONE, NONE, c), "AB": (a, b, NONE),
                     "AC": (a, NONE, c), "BC": (NONE, b, c), "ABC": (a, b, c)}
            dh, du, dz, act = {}, {}, {}, {}
            for k, (x, y, w) in names.items():
                h = run(ablate_edits(x, y, w, 0))
                dh[k] = (h - h0).cpu().double()
                u = s11.sae.W_enc(h - s11.mu)
                du[k] = (u - u0).cpu().double()
                z = s11.encode(h)
                dz[k] = (z - z0).cpu().double()
                act[k] = (z > 0).cpu()
                del h, u, z
            sgn = {"ABC": 1, "AB": -1, "AC": -1, "BC": -1, "A": 1, "B": 1, "C": 1}
            rec = {"triplet_index_in_real": pk, "a": a, "b": b, "c": c}
            for nm, dd in (("h11", dh), ("sae_preact", du), ("sae_code", dz)):
                Iv = sum(sgn[k] * dd[k] for k in sgn)
                tot = dd["ABC"] - dd["A"] - dd["B"] - dd["C"]
                den = max(float((dd["ABC"] ** 2).sum()), 1e-300)
                denm = max(float((dd["ABC"].mean(0) ** 2).sum()), 1e-300)
                rec[nm] = {"per_position_energy_share_I": float((Iv ** 2).sum() / den),
                           "per_position_energy_share_all_orders": float((tot ** 2).sum() / den),
                           "posmean_energy_share_I": float((Iv.mean(0) ** 2).sum() / denm),
                           "posmean_energy_share_all_orders": float((tot.mean(0) ** 2).sum() / denm)}
                if nm == "h11":
                    Il = (Iv.float().to(DEVICE) @ Wlm.T).cpu().double()
                    Al = (dd["ABC"].float().to(DEVICE) @ Wlm.T).cpu().double()
                    Il = Il - Il.mean(1, keepdim=True)
                    Al = Al - Al.mean(1, keepdim=True)
                    rec["logits_centered"] = {
                        "per_position_energy_share_I": float((Il ** 2).sum() / max(float((Al ** 2).sum()), 1e-300)),
                        "posmean_energy_share_I": float((Il.mean(0) ** 2).sum() / max(float((Al.mean(0) ** 2).sum()), 1e-300))}
                    del Il, Al
                if nm == "sae_code":
                    stable = act["A"] & act["B"] & act["C"] & act["AB"] & act["AC"] & act["BC"] & act["ABC"] & act0
                    union = act["A"] | act["B"] | act["C"] | act["AB"] | act["AC"] | act["BC"] | act["ABC"] | act0
                    switching = union & ~stable
                    Ie = Iv ** 2
                    rec[nm]["share_of_I_energy_on_switching_entries"] = float(Ie[switching].sum() / max(float(Ie.sum()), 1e-300))
                    rec[nm]["share_of_dABC_energy_on_switching_entries"] = float(
                        (dd["ABC"] ** 2)[switching].sum() / max(float((dd["ABC"] ** 2).sum()), 1e-300))
                    rec[nm]["positions_with_topk_set_change_vs_clean_ABC"] = float(((act["ABC"] != act0).any(1)).float().mean())
                    rec[nm]["mean_features_switched_per_position_ABC"] = float((act["ABC"] != act0).sum(1).float().mean() / 2)
                    gi = a_groups.index(a)
                    if gi not in g_cache:
                        g = np.load(group_path(ci, gi))
                        g_cache[gi] = {tuple(int(v) for v in k): g["dz"][r] for r, k in enumerate(g["conds"])}
                    stored = g_cache[gi][(a, b, c)]
                    rec[nm]["max_abs_diff_vs_stored_posmean_dz_ABC"] = float(np.abs(dd["ABC"].mean(0).numpy() - stored).max())
                    rec[nm]["max_abs_stored_posmean_dz_ABC"] = float(np.abs(stored).max())
                del Iv, tot
            rec["rel_size_dh11_ABC_over_h11"] = float(dh["ABC"].norm() / h0.cpu().double().norm())
            out["triplets"].append(rec)
            del dh, du, dz, act
            free()
        H.write_json(rdir / f"cell{ci:02d}.json", out)
        del h0, u0, z0, act0
        free()
        durations.append(time.time() - tc)
        done.append([ci, round(durations[-1], 1)])
        log(f"readout cell {ci} done in {durations[-1]:.0f}s")
    left = sum(1 for ci in range(READ_CELLS) if not (rdir / f"cell{ci:02d}.json").exists())
    update_run_config("chunks", {"stage": "readout", "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t_start)),
                                 "wall_seconds": round(time.time() - t_start, 1), "free_gb_at_start": free_gb,
                                 "cells_done": done, "cells_left": left, "encoding_check": chk,
                                 "triplets_read": [[real[p]["a"], real[p]["b"], real[p]["c"]] for p in picks]})
    if left == 0:
        summarize_readout(rdir)
    log(("DONE" if left == 0 else "PARTIAL") + f": {left} cells left")


def summarize_readout(rdir):
    recs = []
    for f in sorted(rdir.glob("cell*.json")):
        d = load_json(f)
        for r in d["triplets"]:
            recs.append(dict(r, cell=d["cell"]))
    s = {"n_cell_triplet_pairs": len(recs)}
    for nm in ("h11", "sae_preact", "sae_code", "logits_centered"):
        s[nm] = {}
        for key in recs[0][nm]:
            v = np.array([r[nm][key] for r in recs], dtype=float)
            s[nm][key] = {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}
    s["rel_size_dh11_ABC_over_h11"] = {"median": float(np.median([r["rel_size_dh11_ABC_over_h11"] for r in recs])),
                                       "min": float(np.min([r["rel_size_dh11_ABC_over_h11"] for r in recs])),
                                       "max": float(np.max([r["rel_size_dh11_ABC_over_h11"] for r in recs]))}
    H.write_json(OUT / "readout_summary.json", s)
    log(json.dumps(s, indent=1))


# =============================================================================
# analysis helpers
# =============================================================================
def bh(p):
    p = np.asarray(p, dtype=np.float64).ravel()
    n = p.size
    order = np.argsort(p)
    q = np.empty(n)
    ranked = p[order] * n / np.arange(1, n + 1)
    q[order] = np.minimum.accumulate(ranked[::-1])[::-1]
    return np.minimum(q, 1.0)


def ttest_p(x):
    from scipy import stats
    n = x.shape[0]
    m = x.mean(0)
    sd = x.std(0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = m / (sd / np.sqrt(n))
    p = 2 * stats.t.sf(np.abs(t), df=n - 1)
    return np.where(sd > 0, p, np.where(m == 0, 1.0, 0.0))


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
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def ci_basic(point, boot, plugin=None):
    """Basic (pivot) bootstrap interval [point - (q97.5 - plugin), point - (q2.5 - plugin)]."""
    plugin = point if plugin is None else plugin
    q = ci95(boot)
    return [float(point - (q[1] - plugin)), float(point - (q[0] - plugin))]


def load_grid(nC, conds, a_groups, with_logits=True):
    cidx = {c: i for i, c in enumerate(conds)}
    K = len(conds)
    D = np.zeros((nC, K, 4928), dtype=np.float32)
    LG = np.zeros((nC, K, 20275), dtype=np.float32) if with_logits else None
    KL = np.zeros((nC, K))
    NA = np.full((nC, K, 3), -3, dtype=np.int64)
    seen = np.zeros((nC, K), dtype=bool)
    for ci in range(nC):
        for gi in range(len(a_groups)):
            p = group_path(ci, gi)
            if not p.exists():
                raise SystemExit(f"missing {p}; run stage 'run' until DONE")
            g = np.load(p)
            for r, k in enumerate(map(tuple, g["conds"])):
                j = cidx[k]
                assert not seen[ci, j], f"duplicate {k}"
                seen[ci, j] = True
                D[ci, j] = g["dz"][r]
                if with_logits:
                    LG[ci, j] = g["dlog"][r]
                KL[ci, j] = g["kl"][r]
                NA[ci, j] = g["n_active"][r]
    assert seen.all(), "some conditions missing"
    return D, LG, KL, NA, cidx


# =============================================================================
# stage: analyze (resumable: per-triplet state is saved under analyze_cache/)
# =============================================================================
def stage_analyze(args):
    t0 = time.time()
    budget = args.max_minutes * 60.0
    sel = load_json(SRC / "selection.json")
    meta = load_json(SRC / "cells.json")
    triplets, conds, a_groups = build_design(sel)
    nC = len(meta["rows"])
    cache = OUT / "analyze_cache"
    cache.mkdir(exist_ok=True)
    dev = args.device or "cpu"
    K = len(conds)
    d_sae, V = 4928, 20275
    D, LG, KL, NA, cidx = load_grid(nC, conds, a_groups)
    log(f"loaded {nC} cells x {K} conditions ({time.time() - t0:.0f}s)")

    def ix(a, b, c):
        return cidx[(a, b, c)]

    # ---------------------------------------------------------------- integrity
    integrity = {}
    rfiles = sorted((SRC / "recheck").glob("cell*.json"))
    if rfiles:
        rc = [load_json(f) for f in rfiles]
        chk = [c for d in rc for c in d["checks"]]
        fp_bad = sum(1 for d in rc for sfp in d["stored_fingerprints"]
                     if any(abs(sfp[i] - d["clean_fingerprint_cpu64"][i]) > 1e-5 * abs(d["clean_fingerprint_cpu64"][i])
                            for i in (0, 1)))
        kl_rel = [abs(c["kl_stored"] - c["kl_cpu64"]) / max(c["kl_cpu64"], 1e-30) for c in chk if c["kl_cpu64"] > 0]
        integrity["recheck"] = {
            "cells": len(rc), "conditions_recomputed": len(chk),
            "max_abs_dz_diff": max(c["max_abs_dz_diff"] for c in chk),
            "max_abs_dlog_diff": max(c["max_abs_dlog_diff"] for c in chk),
            "kl_max_rel_diff": max(kl_rel) if kl_rel else None,
            "stored_clean_fingerprints_wrong(rel>1e-5)": fp_bad,
            "stored_clean_fingerprints_total": sum(len(d["stored_fingerprints"]) for d in rc),
            "zero_delta_cpu64": rc[0].get("zero_delta_cpu64")}
    nullset = {l: set(sel["null_features"][str(l)]) for l in SRC_LAYERS}
    n_twin = n_twin_equal = 0
    twin_max = 0.0
    for j, k in enumerate(conds):
        for pos, l in enumerate(SRC_LAYERS):
            if k[pos] != NONE and k[pos] in nullset[l]:
                tw = list(k)
                tw[pos] = NONE
                tw = tuple(tw)
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
    LG -= LG.mean(axis=2, keepdims=True)

    rng = np.random.default_rng(BOOT_SEED)
    Wb = rng.multinomial(nC, np.full(nC, 1.0 / nC), size=N_BOOT).astype(np.float32) / nC
    np.save(OUT / "bootstrap_cell_weights.npy", Wb)
    Wt = torch.from_numpy(Wb).to(dev)
    frng = np.random.default_rng(BOOT_SEED + 1)
    Sf = frng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(N_FLIP, nC))
    np.save(OUT / "signflip_signs.npy", Sf)
    St = torch.from_numpy(Sf / nC).to(dev)
    S5 = torch.from_numpy(Sf[:5]).to(dev)

    def tpct(srt, q):
        pos = q / 100.0 * (srt.shape[0] - 1)
        lo_i = int(np.floor(pos))
        fr = pos - lo_i
        hi_i = min(lo_i + 1, srt.shape[0] - 1)
        return srt[lo_i] + fr * (srt[hi_i] - srt[lo_i])

    def energy_np(X):
        n = X.shape[0]
        return float((X.mean(0) ** 2).sum() - X.var(0, ddof=1).sum() / n)

    def energy_boot(Xt):
        n = Xt.shape[0]
        m = Wt @ Xt
        m2 = Wt @ (Xt * Xt)
        var = (m2 - m * m).clamp_min(0) * n / (n - 1)
        return ((m * m).sum(1) - var.sum(1) / n).cpu().numpy()

    def f32(x):
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(dev)

    A, B, C = sel["features"]["0"], sel["features"]["5"], sel["features"]["9"]
    real = [t for t in triplets if t["kind"] == "real"]
    real_pos = {}
    for i, t in enumerate(triplets):
        if t["kind"] == "real":
            real_pos[i] = len(real_pos)
    n_trip = len(triplets)

    # ---------------------------------------------------------------- per-condition effects (cached)
    effD = D.mean(0, dtype=np.float64)
    effL = LG.mean(0, dtype=np.float64)
    eff_abs = np.abs(effD).mean(1)
    eff_abs_L = np.abs(effL).mean(1)
    pe = cache / "boot_eff.npz"
    if pe.exists():
        z_ = np.load(pe)
        boot_eff, boot_eff_L = z_["boot_eff"], z_["boot_eff_L"]
    else:
        boot_eff = np.zeros((K, N_BOOT), dtype=np.float32)
        boot_eff_L = np.zeros((K, N_BOOT), dtype=np.float32)
        with torch.no_grad():
            for j0 in range(0, K, 16):
                js = slice(j0, min(K, j0 + 16))
                x = f32(D[:, js]).reshape(nC, -1)
                boot_eff[js] = (Wt @ x).reshape(N_BOOT, -1, d_sae).abs().mean(2).T.cpu().numpy()
                for j1 in range(js.start, js.stop, 4):
                    jj = slice(j1, min(js.stop, j1 + 4))
                    y = f32(LG[:, jj]).reshape(nC, -1)
                    boot_eff_L[jj] = (Wt @ y).reshape(N_BOOT, -1, V).abs().mean(2).T.cpu().numpy()
                del x, y
        np.savez(pe, boot_eff=boot_eff, boot_eff_L=boot_eff_L)
        log(f"bootstrap of condition effects done ({time.time() - t0:.0f}s)")

    def ratios(e):
        mx = np.maximum
        r_ab = e["AB"] / mx(e["A"] + e["B"], 1e-30)
        r_ac = e["AC"] / mx(e["A"] + e["C"], 1e-30)
        r_bc = e["BC"] / mx(e["B"] + e["C"], 1e-30)
        three = e["ABC"] / mx(e["A"] + e["B"] + e["C"], 1e-30)
        marg = (e["ABC"] - e["AB"]) / mx(e["C"], 1e-30)
        return r_ab, r_ac, r_bc, (r_ab + r_ac + r_bc) / 3, three, marg

    # ---------------------------------------------------------------- per-triplet loop (resumable)
    ps = cache / "state.pkl"
    if ps.exists():
        with open(ps, "rb") as fh:
            st = pickle.load(fh)
        log(f"resuming analyze at triplet {st['next']}")
    else:
        st = {"next": 0, "rows": [],
              "I_mean": np.zeros((n_trip, d_sae), dtype=np.float32), "lo": np.zeros((n_trip, d_sae), dtype=np.float32),
              "hi": np.zeros((n_trip, d_sae), dtype=np.float32), "p": np.ones((n_trip, d_sae), dtype=np.float64),
              "pboot": np.ones((n_trip, d_sae), dtype=np.float64),
              "boot": {k: np.zeros((len(real), N_BOOT), dtype=np.float32) for k in
                       ("AB", "AC", "BC", "pair_mean", "three", "marg", "Ishare", "three_L", "pair_mean_L", "Ishare_L",
                        "three_add", "three_minus_add", "pair_add", "pair_minus_add",
                        "rho3", "eshare_I", "eshare_tot", "rho3_L", "eshare_I_L", "rho2")}}

    def save_state():
        tmp = ps.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(st, fh, protocol=4)
        tmp.replace(ps)

    boot_ratio = st["boot"]
    t_loop = time.time()
    per_trip = []
    for ti in range(st["next"], n_trip):
        est = max(per_trip) * 70 if per_trip else 60.0
        if time.time() - t0 + est > budget and ti % 64 == 0 and ti > st["next"]:
            save_state()
            update_run_config("chunks", {"stage": "analyze", "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
                                         "wall_seconds": round(time.time() - t0, 1), "triplets_done": ti,
                                         "device": dev})
            log(f"PARTIAL: analyze stopped at triplet {ti}/{n_trip}; run again")
            return
        tt = time.time()
        t = triplets[ti]
        a, b, c = t["a"], t["b"], t["c"]
        E = {"A": (a, NONE, NONE), "B": (NONE, b, NONE), "C": (NONE, NONE, c),
             "AB": (a, b, NONE), "AC": (a, NONE, c), "BC": (NONE, b, c), "ABC": (a, b, c)}
        J = {k: ix(*v) for k, v in E.items()}
        Dc = {k: D[:, J[k]].astype(np.float64) for k in J}
        Lc = {k: LG[:, J[k]].astype(np.float64) for k in J}
        Iv = Dc["ABC"] - Dc["AB"] - Dc["AC"] - Dc["BC"] + Dc["A"] + Dc["B"] + Dc["C"]
        IL = Lc["ABC"] - Lc["AB"] - Lc["AC"] - Lc["BC"] + Lc["A"] + Lc["B"] + Lc["C"]
        Im = Iv.mean(0)
        with torch.no_grad():
            bI_t = Wt @ f32(Iv)
            srt, _ = torch.sort(bI_t, dim=0)
            lo = tpct(srt, 2.5).cpu().numpy().astype(np.float64)
            hi = tpct(srt, 97.5).cpu().numpy().astype(np.float64)
            nle = (bI_t <= 0).sum(0).cpu().numpy()
            nge = (bI_t >= 0).sum(0).cpu().numpy()
            del srt
        alive = np.abs(Iv).max(0) > 0
        excl = alive & ((lo > 0) | (hi < 0))
        pboot = np.minimum(1.0, 2 * np.minimum(nle + 1, nge + 1) / (N_BOOT + 1))
        pboot = np.where(alive, pboot, 1.0)
        p = ttest_p(Iv)
        q_within = bh(p)
        IL_m = IL.mean(0)
        qL_within = bh(ttest_p(IL))
        st["I_mean"][ti], st["lo"][ti], st["hi"][ti], st["p"][ti], st["pboot"][ti] = Im, lo, hi, p, pboot
        with torch.no_grad():
            It = f32(Iv)
            T_obs = float(np.abs(Im).sum())
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
        eABC = Dc["ABC"].mean(0)
        share = np.abs(Im).sum() / max(np.abs(eABC).sum(), 1e-30)
        share_L = np.abs(IL_m).sum() / max(np.abs(Lc["ABC"].mean(0)).sum(), 1e-30)
        IAB = (Dc["AB"] - Dc["A"] - Dc["B"]).mean(0)
        IAC = (Dc["AC"] - Dc["A"] - Dc["C"]).mean(0)
        IBC = (Dc["BC"] - Dc["B"] - Dc["C"]).mean(0)
        pair_share = [np.abs(x).sum() / max(np.abs(Dc[k].mean(0)).sum(), 1e-30)
                      for x, k in ((IAB, "AB"), (IAC, "AC"), (IBC, "BC"))]
        ef = {k: float(eff_abs[J[k]]) for k in J}
        efL = {k: float(eff_abs_L[J[k]]) for k in J}
        r_ab, r_ac, r_bc, r_pm, r_three, r_marg = (float(x) for x in ratios(ef))
        rL = [float(x) for x in ratios(efL)]
        dd = {}
        for k in J:
            m = Dc[k].mean(0)
            sd_ = Dc[k].std(0, ddof=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                dd[k] = np.where(sd_ > 0, m / sd_, 0.0)
        incl = np.zeros(d_sae, dtype=bool)
        for k in dd:
            incl |= np.abs(dd[k]) > 0.5
        sum_d = np.abs(dd["A"]) + np.abs(dd["B"]) + np.abs(dd["C"])
        with np.errstate(divide="ignore", invalid="ignore"):
            rd = np.where(sum_d > 0, np.abs(dd["ABC"]) / sum_d, np.inf)
        eA, eB, eC = (np.abs(Dc[k].mean(0)) for k in ("A", "B", "C"))
        dep_tg = np.abs(eABC) > 0.01
        san = {"AB_vs_B": float(np.abs(Dc["AB"].mean(0) - Dc["B"].mean(0)).max()),
               "AC_vs_C": float(np.abs(Dc["AC"].mean(0) - Dc["C"].mean(0)).max()),
               "BC_vs_C": float(np.abs(Dc["BC"].mean(0) - Dc["C"].mean(0)).max()),
               "ABC_vs_C": float(np.abs(eABC - Dc["C"].mean(0)).max())}
        na_abc = NA[:, J["ABC"]]
        row = {
            "triplet": ti, "kind": t["kind"], "a": a, "b": b, "c": c,
            "n_targets_alive_any": int(alive.sum()),
            "I_max_abs": float(np.abs(Im).max()), "I_sum_abs": float(np.abs(Im).sum()),
            "I_max_abs_percell": float(np.abs(Iv).max()),
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
            "marg_C_given_AB": r_marg, "r_three_logits": rL[4], "r_pair_mean_logits": rL[3],
            "legacy_identity_eC_over_sum": ef["C"] / max(ef["A"] + ef["B"] + ef["C"], 1e-30),
            "spec_n_included": int(incl.sum()), "spec_super_strict": int((incl & (np.abs(dd["ABC"]) > sum_d)).sum()),
            "spec_super_band": int((incl & (rd > 1.1)).sum()),
            "spec_additive": int((incl & (rd >= 0.9) & (rd <= 1.1)).sum()), "spec_sub": int((incl & (rd < 0.9)).sum()),
            "dep_n_targets": int(dep_tg.sum()), "dep_super": int((dep_tg & (np.abs(eABC) > 1.1 * (eA + eB + eC))).sum()),
            "kl": {k: float(KL[:, J[k]].mean()) for k in J},
            "sanity": san,
            "n_active_ABC_cellmean": [float(na_abc[:, 0][na_abc[:, 0] >= 0].mean()) if (na_abc[:, 0] >= 0).any() else None,
                                      float(na_abc[:, 1].mean()), float(na_abc[:, 2].mean())],
        }
        if t["kind"] == "real":
            ri = real_pos[ti]
            be = {k: boot_eff[J[k]] for k in J}
            beL = {k: boot_eff_L[J[k]] for k in J}
            r = ratios(be)
            rl = ratios(beL)
            boot_ratio["AB"][ri], boot_ratio["AC"][ri], boot_ratio["BC"][ri] = r[0], r[1], r[2]
            boot_ratio["pair_mean"][ri], boot_ratio["three"][ri], boot_ratio["marg"][ri] = r[3], r[4], r[5]
            boot_ratio["three_L"][ri], boot_ratio["pair_mean_L"][ri] = rl[4], rl[3]
            with torch.no_grad():
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
            row["r_three_additive"] = float(np.abs(eA_s + eB_s + eC_s).mean() / den3p)
            pa = []
            for x, y in (("A", "B"), ("A", "C"), ("B", "C")):
                ex, ey = Dc[x].mean(0), Dc[y].mean(0)
                pa.append(np.abs(ex + ey).mean() / (np.abs(ex).mean() + np.abs(ey).mean()))
            row["r_pair_mean_additive"] = float(np.mean(pa))
            row["ci_three_minus_additive"] = ci_basic(row["r_three"] - row["r_three_additive"], boot_ratio["three_minus_add"][ri])
            Xadd = Dc["A"] + Dc["B"] + Dc["C"]
            Xtot = Dc["ABC"] - Xadd
            Ladd = Lc["A"] + Lc["B"] + Lc["C"]
            en = {"ABC": energy_np(Dc["ABC"]), "add": energy_np(Xadd), "I": energy_np(Iv), "tot": energy_np(Xtot),
                  "L_ABC": energy_np(Lc["ABC"]), "L_add": energy_np(Ladd), "L_I": energy_np(IL)}
            e2 = [energy_np(Dc[x + y]) / max(energy_np(Dc[x] + Dc[y]), 1e-30)
                  for x, y in (("A", "B"), ("A", "C"), ("B", "C"))]
            row["energy"] = en
            nz = (Dc["ABC"] ** 2).sum(1) > 0
            row["percell_share_I_median"] = float(np.median((Iv[nz] ** 2).sum(1) / (Dc["ABC"][nz] ** 2).sum(1))) if nz.any() else None
            row["percell_share_tot_median"] = float(np.median((Xtot[nz] ** 2).sum(1) / (Dc["ABC"][nz] ** 2).sum(1))) if nz.any() else None
            nzl = (Lc["ABC"] ** 2).sum(1) > 0
            row["percell_share_I_logits_median"] = float(np.median((IL[nzl] ** 2).sum(1) / (Lc["ABC"][nzl] ** 2).sum(1))) if nzl.any() else None

            def pl(X):
                return float((X.mean(0) ** 2).sum())
            row["rho3_plugin"] = pl(Dc["ABC"]) / max(pl(Xadd), 1e-30)
            row["eshare_I_plugin"] = pl(Iv) / max(pl(Dc["ABC"]), 1e-30)
            row["eshare_tot_plugin"] = pl(Xtot) / max(pl(Dc["ABC"]), 1e-30)
            row["rho3_logits_plugin"] = pl(Lc["ABC"]) / max(pl(Ladd), 1e-30)
            row["eshare_I_logits_plugin"] = pl(IL) / max(pl(Lc["ABC"]), 1e-30)
            row["rho2_mean_plugin"] = float(np.mean([pl(Dc[x + y]) / max(pl(Dc[x] + Dc[y]), 1e-30)
                                                     for x, y in (("A", "B"), ("A", "C"), ("B", "C"))]))
            row["rho3"] = en["ABC"] / max(en["add"], 1e-30)
            row["eshare_I"] = en["I"] / max(en["ABC"], 1e-30)
            row["eshare_tot"] = en["tot"] / max(en["ABC"], 1e-30)
            row["rho3_logits"] = en["L_ABC"] / max(en["L_add"], 1e-30)
            row["eshare_I_logits"] = en["L_I"] / max(en["L_ABC"], 1e-30)
            row["rho2_mean"] = float(np.mean(e2))
            with torch.no_grad():
                b_abc = energy_boot(f32(Dc["ABC"]))
                b_add = energy_boot(f32(Xadd))
                b_I = energy_boot(f32(Iv))
                b_tot = energy_boot(f32(Xtot))
                bl_abc = energy_boot(f32(Lc["ABC"]))
                bl_add = energy_boot(f32(Ladd))
                bl_I = energy_boot(f32(IL))
                b2 = np.mean([energy_boot(f32(Dc[x + y])) / np.maximum(energy_boot(f32(Dc[x] + Dc[y])), 1e-30)
                              for x, y in (("A", "B"), ("A", "C"), ("B", "C"))], axis=0)
                Xt_ = f32(Xtot)
                Ttot = float(np.abs(Xtot.mean(0)).sum())
                Ttn = (St @ Xt_).abs().sum(1).cpu().numpy()
                del Xt_
            row["agg_p_signflip_tot"] = float((1 + (Ttn >= Ttot).sum()) / (N_FLIP + 1)) if Ttot > 0 else 1.0
            row["tot_T_obs_over_null"] = Ttot / max(float(np.median(Ttn)), 1e-30)
            boot_ratio["rho3"][ri] = b_abc / np.maximum(b_add, 1e-30)
            boot_ratio["eshare_I"][ri] = b_I / np.maximum(b_abc, 1e-30)
            boot_ratio["eshare_tot"][ri] = b_tot / np.maximum(b_abc, 1e-30)
            boot_ratio["rho3_L"][ri] = bl_abc / np.maximum(bl_add, 1e-30)
            boot_ratio["eshare_I_L"][ri] = bl_I / np.maximum(bl_abc, 1e-30)
            boot_ratio["rho2"][ri] = b2
            row["ci_rho3"] = ci_basic(row["rho3"], boot_ratio["rho3"][ri], row["rho3_plugin"])
            row["ci_eshare_I"] = ci_basic(row["eshare_I"], boot_ratio["eshare_I"][ri], row["eshare_I_plugin"])
            row["ci_eshare_tot"] = ci_basic(row["eshare_tot"], boot_ratio["eshare_tot"][ri], row["eshare_tot_plugin"])
            row["ci_r_three"] = ci_basic(row["r_three"], boot_ratio["three"][ri])
            row["ci_r_pair_mean"] = ci_basic(row["r_pair_mean"], boot_ratio["pair_mean"][ri])
            row["ci_share_I"] = ci_basic(row["share_I_over_ABC"], boot_ratio["Ishare"][ri])
        st["rows"].append(row)
        del bI_t
        st["next"] = ti + 1
        per_trip.append(time.time() - tt)
        if ti % 64 == 0:
            log(f"triplet {ti}/{n_trip} ({np.mean(per_trip):.2f}s per triplet)")
            save_state()
    save_state()
    rows = st["rows"]
    assert len(rows) == n_trip and [r["triplet"] for r in rows] == list(range(n_trip))
    log(f"per-triplet loop finished ({time.time() - t_loop:.0f}s this call)")
    if time.time() - t0 > budget - 90:
        update_run_config("chunks", {"stage": "analyze", "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
                                     "wall_seconds": round(time.time() - t0, 1), "triplets_done": n_trip, "device": dev})
        log("PARTIAL: loop done, summaries left for the next call; run again")
        return

    real_idx = [i for i, t in enumerate(triplets) if t["kind"] == "real"]
    P = st["p"][real_idx]
    Q = bh(P).reshape(len(real), d_sae)
    for k, ti in enumerate(real_idx):
        rows[ti]["n_q05_across_t"] = int((Q[k] < 0.05).sum())
    QB = bh(st["pboot"][real_idx]).reshape(len(real), d_sae)
    for k, ti in enumerate(real_idx):
        rows[ti]["n_q05_across_boot"] = int((QB[k] < 0.05).sum())
    R_ = [rows[i] for i in real_idx]
    N_ = [r for r in rows if r["kind"] != "real"]
    for r, q in zip(R_, bh([r["agg_p_signflip"] for r in R_])):
        r["agg_q_signflip_across_triplets"] = float(q)
    for r, q in zip(R_, bh([r["agg_p_signflip_tot"] for r in R_])):
        r["agg_q_signflip_tot_across_triplets"] = float(q)

    # ---------------------------------------------------------------- planted interaction (constant), as v2
    prng = np.random.default_rng(BOOT_SEED + 2)
    pick = sorted(prng.choice(len(real_idx), size=64, replace=False).tolist())
    power = {}
    for kappa in (0.02, 0.05, 0.1, 0.2, 0.3):
        det_p, det_e = [], []
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
                It_ = f32(Ik)
                To = float(np.abs(Ik.mean(0)).sum())
                Tn = (St @ It_).abs().sum(1).cpu().numpy()
                bI_ = energy_boot(It_)
                bA_ = energy_boot(f32(ABCk))
                del It_
            pk = float((1 + (Tn >= To).sum()) / (N_FLIP + 1))
            pt_ = energy_np(Ik) / max(energy_np(ABCk), 1e-30)
            pl_ = float((Ik.mean(0) ** 2).sum()) / max(float((ABCk.mean(0) ** 2).sum()), 1e-30)
            ci_ = ci_basic(pt_, bI_ / np.maximum(bA_, 1e-30), pl_)
            det_p.append(pk)
            det_e.append(ci_[0] > 0)
        qk = bh(det_p)
        power[str(kappa)] = {"planted_energy_share_of_joint_effect": float(kappa ** 2 / (1 + kappa) ** 2),
                             "n_triplets": len(pick),
                             "detected_signflip_p05": int(sum(x < 0.05 for x in det_p)),
                             "detected_signflip_BH05_within_64": int(sum(x < 0.05 for x in qk)),
                             "detected_energy_share_ci_above_0": int(sum(det_e))}
    log(f"power: {json.dumps(power)}")

    # ---------------------------------------------------------------- pairwise interactions (context)
    pair_rows = []
    for (X, Y, lx, ly) in (("A", "B", 0, 1), ("A", "C", 0, 2), ("B", "C", 1, 2)):
        fx = A if X == "A" else B
        fy = B if Y == "B" else C
        for u in fx:
            for v in fy:
                kx = [NONE, NONE, NONE]
                kx[lx] = u
                ky = [NONE, NONE, NONE]
                ky[ly] = v
                kxy = list(kx)
                kxy[ly] = v
                dX, dY, dXY = (D[:, ix(*k)].astype(np.float64) for k in (kx, ky, kxy))
                I2 = dXY - dX - dY
                I2m = I2.mean(0)
                with torch.no_grad():
                    T2 = float(np.abs(I2m).sum())
                    T2n = (St @ f32(I2)).abs().sum(1).cpu().numpy()
                exy = dXY.mean(0)
                pair_rows.append({
                    "pair": X + Y, "x": int(u), "y": int(v),
                    "share_I2": float(np.abs(I2m).sum() / max(np.abs(exy).sum(), 1e-30)),
                    "share_I2_noise_signflip": float(np.median(T2n) / max(np.abs(exy).sum(), 1e-30)),
                    "agg_p_signflip": float((1 + (T2n >= T2).sum()) / (N_FLIP + 1)) if T2 > 0 else 1.0,
                    "n_q05_within_t": int((bh(ttest_p(I2)) < 0.05).sum()),
                    "rho2_noise_corrected": energy_np(dXY) / max(energy_np(dX + dY), 1e-30),
                    "r_pair": float(np.abs(exy).mean() / (np.abs(dX.mean(0)).mean() + np.abs(dY.mean(0)).mean())),
                    "r_pair_additive": float(np.abs(dX.mean(0) + dY.mean(0)).mean()
                                             / (np.abs(dX.mean(0)).mean() + np.abs(dY.mean(0)).mean()))})
    for r, q in zip(pair_rows, bh([r["agg_p_signflip"] for r in pair_rows])):
        r["agg_q_signflip_across_pairs"] = float(q)
    pair_summary = {
        "n_pairs": len(pair_rows),
        "pairs_agg_signflip_p05_uncorrected": int(sum(r["agg_p_signflip"] < 0.05 for r in pair_rows)),
        "pairs_agg_signflip_BH05": int(sum(r["agg_q_signflip_across_pairs"] < 0.05 for r in pair_rows)),
        "pairs_any_target_BH_within_t": int(sum(r["n_q05_within_t"] > 0 for r in pair_rows)),
        "share_I2_median": float(np.median([r["share_I2"] for r in pair_rows])),
        "share_I2_noise_median": float(np.median([r["share_I2_noise_signflip"] for r in pair_rows])),
        "rho2_noise_corrected_median": float(np.median([r["rho2_noise_corrected"] for r in pair_rows])),
        "by_pair_type": {pt: {"n": len(rr), "agg_BH05": int(sum(r["agg_q_signflip_across_pairs"] < 0.05 for r in rr)),
                              "share_I2_median": float(np.median([r["share_I2"] for r in rr])),
                              "share_I2_noise_median": float(np.median([r["share_I2_noise_signflip"] for r in rr])),
                              "r_pair_median": float(np.median([r["r_pair"] for r in rr])),
                              "r_pair_additive_median": float(np.median([r["r_pair_additive"] for r in rr]))}
                         for pt in ("AB", "AC", "BC") for rr in [[r for r in pair_rows if r["pair"] == pt]]}}
    log("pairwise interactions done")

    def frac_any(key, rr):
        return int(sum(1 for r in rr if r[key] > 0)), len(rr)

    def med_ci(key, point, plugin=None):
        bmed = np.median(boot_ratio[key], axis=0)
        pt = float(np.median(point))
        pl_ = float(np.median(plugin)) if plugin is not None else pt
        return {"median": pt, "ci95_of_median": ci_basic(pt, bmed, pl_),
                "ci95_percentile_of_median(biased; for reference)": ci95(bmed),
                "median_plugin_uncorrected": pl_ if plugin is not None else None, "dist": summarize(point)}
    rs = {
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
    rs["three_way_if_additive"] = med_ci("three_add", [r["r_three_additive"] for r in R_])
    rs["three_way_minus_additive"] = med_ci("three_minus_add", [r["r_three"] - r["r_three_additive"] for r in R_])
    rs["pairwise_mean_if_additive"] = med_ci("pair_add", [r["r_pair_mean_additive"] for r in R_])
    rs["pairwise_mean_minus_additive"] = med_ci("pair_minus_add", [r["r_pair_mean"] - r["r_pair_mean_additive"] for r in R_])
    rs["noise_corrected_rho3=E[ABC]/E[A+B+C]"] = med_ci("rho3", [r["rho3"] for r in R_], [r["rho3_plugin"] for r in R_])
    rs["noise_corrected_rho2_pair_mean"] = med_ci("rho2", [r["rho2_mean"] for r in R_], [r["rho2_mean_plugin"] for r in R_])
    rs["noise_corrected_energy_share_I_ABC"] = med_ci("eshare_I", [r["eshare_I"] for r in R_], [r["eshare_I_plugin"] for r in R_])
    rs["noise_corrected_energy_share_all_interactions"] = med_ci("eshare_tot", [r["eshare_tot"] for r in R_],
                                                                 [r["eshare_tot_plugin"] for r in R_])
    rs["noise_corrected_rho3_logits"] = med_ci("rho3_L", [r["rho3_logits"] for r in R_], [r["rho3_logits_plugin"] for r in R_])
    rs["noise_corrected_energy_share_I_ABC_logits"] = med_ci("eshare_I_L", [r["eshare_I_logits"] for r in R_],
                                                             [r["eshare_I_logits_plugin"] for r in R_])
    rs["n_triplets_rho3_ci_below_1"] = int(sum(1 for r in R_ if r["ci_rho3"][1] < 1))
    rs["n_triplets_rho3_ci_above_1"] = int(sum(1 for r in R_ if r["ci_rho3"][0] > 1))
    rs["n_triplets_eshare_I_ci_above_0"] = int(sum(1 for r in R_ if r["ci_eshare_I"][0] > 0))
    rs["n_triplets_eshare_tot_ci_above_0"] = int(sum(1 for r in R_ if r["ci_eshare_tot"][0] > 0))
    rs["n_triplets_three_minus_additive_ci_below_0"] = int(sum(1 for r in R_ if r["ci_three_minus_additive"][1] < 0))
    rs["n_triplets_three_minus_additive_ci_above_0"] = int(sum(1 for r in R_ if r["ci_three_minus_additive"][0] > 0))
    rs["pair_mean_mean_across_triplets_ci95_basic"] = ci_basic(float(np.mean([r["r_pair_mean"] for r in R_])),
                                                               boot_ratio["pair_mean"].mean(0))
    rs["three_mean_across_triplets_ci95_basic"] = ci_basic(float(np.mean([r["r_three"] for r in R_])),
                                                           boot_ratio["three"].mean(0))
    rs["pair_mean_mean_across_triplets"] = float(np.mean([r["r_pair_mean"] for r in R_]))
    rs["three_mean_across_triplets"] = float(np.mean([r["r_three"] for r in R_]))

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
        "n_triplets_with_any_deployed_target": frac_any("dep_n_targets", R_)[0],
    }
    cls["frac_spec_included_of_all_targets"] = tot_incl / (len(R_) * d_sae)
    from scipy import stats as _st
    p_d = 2 * _st.t.sf(0.5 * np.sqrt(nC), df=nC - 1)
    cls["frac_included_expected_from_pure_noise(|d|>0.5 in any of 7 conditions, independent)"] = float(1 - (1 - p_d) ** 7)
    cls["frac_spec_super_strict"] = cls["spec_superadditive_strict(|d_ABC|>sum)"] / max(tot_incl, 1)
    w = np.array([r["spec_n_included"] for r in R_], dtype=float)
    cls["raw_mean_subadditive(<0.9x)_frac_of_spec_included_observed"] = float(np.average(
        [r["frac_sub_obs_spec_included"] or 0 for r in R_], weights=w))
    cls["raw_mean_subadditive(<0.9x)_frac_if_exactly_additive"] = float(np.average(
        [r["frac_sub_additive_spec_included"] or 0 for r in R_], weights=w))
    cls["frac_deployed_super"] = cls["deployed_superadditive(>1.1x)"] / max(cls["deployed_targets(|e_ABC|>0.01)"], 1)

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
        "min_p_t_over_all_tests": float(min(r["min_p_t"] for r in R_)),
        "n_targets_alive_per_triplet": summarize([r["n_targets_alive_any"] for r in R_]),
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
            "expected_p05_if_null": 0.05 * len(R_),
            "n_triplets_BH05_across_triplets": int(sum(r["agg_q_signflip_across_triplets"] < 0.05 for r in R_)),
            "p_distribution": summarize([r["agg_p_signflip"] for r in R_]),
            "T_obs_over_T_null_median": summarize([r["agg_T_obs"] / max(r["agg_T_null_median"], 1e-30) for r in R_])},
        "aggregate_signflip_test_all_orders(d_ABC - d_A - d_B - d_C)": {
            "n_triplets_p05_uncorrected": int(sum(r["agg_p_signflip_tot"] < 0.05 for r in R_)),
            "n_triplets_BH05_across_triplets": int(sum(r["agg_q_signflip_tot_across_triplets"] < 0.05 for r in R_)),
            "T_obs_over_T_null_median": summarize([r["tot_T_obs_over_null"] for r in R_])},
        "percell_nonadditivity(median over triplets of the median over cells)": {
            "I_ABC_energy_share": float(np.median([r["percell_share_I_median"] for r in R_])),
            "all_orders_energy_share": float(np.median([r["percell_share_tot_median"] for r in R_])),
            "I_ABC_energy_share_logits": float(np.median([r["percell_share_I_logits_median"] for r in R_]))},
        "interaction_share_observed_median": float(np.median([r["share_I_over_ABC"] for r in R_])),
        "interaction_share_signflip_noise_median": float(np.median([r["share_I_noise_signflip"] for r in R_])),
    }
    noise = {}
    for kind in ("null_all", "null_A", "null_B", "null_C"):
        rr = [r for r in N_ if r["kind"] == kind]
        noise[kind] = {"n_triplets": len(rr), "max_abs_I_cellmean": max(r["I_max_abs"] for r in rr),
                       "max_abs_I_percell": max(r["I_max_abs_percell"] for r in rr),
                       "triplets_any_ci_excl0": frac_any("n_ci_excl0", rr),
                       "triplets_any_BH_within_t": frac_any("n_q05_within_t", rr),
                       "triplets_agg_signflip_p05": int(sum(r["agg_p_signflip"] < 0.05 for r in rr))}
    mx = {}
    for j, k in enumerate(conds):
        for pos, l in enumerate(SRC_LAYERS):
            if k[pos] != NONE and k[pos] in nullset[l]:
                v = NA[:, j, pos]
                v = v[v >= 0]
                if v.size:
                    mx[l] = max(mx.get(l, 0), int(v.max()))
    noise["null_feature_live_n_active_max_by_layer"] = {str(l): mx.get(l) for l in SRC_LAYERS}

    ab_vs_b = [float(np.abs(effD[ix(a, b, NONE)] - effD[ix(NONE, b, NONE)]).max()) for a in A for b in B]
    ab_vs_a = [float(np.abs(effD[ix(a, b, NONE)] - effD[ix(a, NONE, NONE)]).max()) for a in A for b in B]
    abc_vs_c = [r["sanity"]["ABC_vs_C"] for r in R_]
    legacy = np.array([r["legacy_identity_eC_over_sum"] for r in R_])
    three = np.array([r["r_three"] for r in R_])
    single_eff = {f"L{l}": {str(f): float(eff_abs[ix(*(f if p == q else NONE for q in range(3)))])
                            for f in sel["features"][str(l)]} for p, l in enumerate(SRC_LAYERS)}
    # live n_active of each chosen feature in its single condition (positions per cell)
    nact_single = {}
    for p, l in enumerate(SRC_LAYERS):
        for f in sel["features"][str(l)]:
            key = tuple(f if p == q else NONE for q in range(3))
            nact_single[f"L{l}_{f}"] = float(NA[:, ix(*key), p].mean())
    sanity = {
        "verify_stage": load_json(SRC / "verify.json")["verdict"] if (SRC / "verify.json").exists() else None,
        "data_integrity": integrity,
        "AB_vs_B_maxabs_over_targets": {"min_over_64_pairs": min(ab_vs_b), "median": float(np.median(ab_vs_b)),
                                        "n_pairs_equal(<1e-7)": int(sum(x < 1e-7 for x in ab_vs_b))},
        "AB_vs_A_maxabs_over_targets": {"min_over_64_pairs": min(ab_vs_a), "median": float(np.median(ab_vs_a))},
        "ABC_vs_C_maxabs_over_targets": {"min_over_512": min(abc_vs_c), "median": float(np.median(abc_vs_c)),
                                         "n_equal(<1e-7)": int(sum(x < 1e-7 for x in abc_vs_c))},
        "three_way_ratio_std_across_triplets": float(three.std(ddof=1)),
        "pair_mean_ratio_std_across_triplets": float(np.std([r["r_pair_mean"] for r in R_], ddof=1)),
        "corr_three_way_vs_legacy_identity": float(np.corrcoef(three, legacy)[0, 1]),
        "max_abs_three_way_minus_legacy_identity": float(np.abs(three - legacy).max()),
        "single_effect_eff_by_feature": single_eff,
        "single_effect_eff_range_by_layer": {k: [min(v.values()), max(v.values())] for k, v in single_eff.items()},
        "live_n_active_positions_per_cell_single_condition": nact_single,
    }
    counts = {
        "interventions": {"triplets_real": len(R_), "triplets_noise_floor": len(N_),
                          "conditions_real_grid": {"single": 24, "pair": 192, "triple": 512, "total": 728},
                          "conditions_total_incl_noise_floor": K, "cells": nC,
                          "ablated_forward_passes": K * nC, "clean_forward_passes": nC},
        "measurements": {"L11_targets_per_condition": d_sae, "logit_targets_per_condition": V,
                         "triplet_x_L11_target_interaction_terms(real)": len(R_) * d_sae}}
    # breakdown by frequency half (bins 0-3 = rarer 'l', bins 4-7 = commoner 'h') and by the post-hoc v3 annotation
    det = sel["details"]
    bin_of = {(l, d["feature"]): d["freq_bin_used"] for l in SRC_LAYERS for d in det[str(l)]["chosen"]}
    ann_of = {(l, d["feature"]): d.get("posthoc_v3_annotated") for l in SRC_LAYERS for d in det[str(l)]["chosen"]}

    def breakdown(keyf):
        groups = {}
        for r in R_:
            groups.setdefault(keyf(r), []).append(r)
        return {k: {"n": len(v), "median_rho3": float(np.median([r["rho3"] for r in v])),
                    "median_eshare_I": float(np.median([r["eshare_I"] for r in v])),
                    "median_three_way": float(np.median([r["r_three"] for r in v])),
                    "n_agg_signflip_p05": int(sum(r["agg_p_signflip"] < 0.05 for r in v)),
                    "frac_any_BH_across": float(np.mean([r["n_q05_across_t"] > 0 for r in v]))}
                for k, v in sorted(groups.items())}
    by_freq = breakdown(lambda r: "".join("l" if bin_of[(l, f)] < 4 else "h"
                                          for l, f in zip(SRC_LAYERS, (r["a"], r["b"], r["c"]))))
    by_ann = breakdown(lambda r: "".join("a" if ann_of[(l, f)] else "u"
                                         for l, f in zip(SRC_LAYERS, (r["a"], r["b"], r["c"]))))
    kl_by_order = {}
    for order in (1, 2, 3):
        js = [j for j, k in enumerate(conds) if sum(x != NONE for x in k) == order
              and all(x == NONE or x in sel["features"][str(l)] for x, l in zip(k, SRC_LAYERS))]
        kl_by_order[str(order)] = {"n_conditions": len(js), "kl_cellmean": summarize(KL[:, js].mean(0))}
    summary = {"counts": counts, "significance_I_ABC": sig, "redundancy_ratios": rs,
               "pairwise_interactions": pair_summary, "logit_effect_size_by_order": kl_by_order,
               "planted_interaction_power": power, "superadditivity": cls, "noise_floor": noise, "sanity": sanity,
               "by_frequency_half(L0,L5,L9; l=octile bins 0-3, h=bins 4-7)": by_freq,
               "by_posthoc_v3_annotation(L0,L5,L9; a=annotated, u=not; NOT used for selection)": by_ann,
               "bootstrap": {"unit": "cell", "n_boot": N_BOOT, "seed": BOOT_SEED, "device": dev,
                             "method": "multinomial cell weights shared across all triplets and targets; percentile CIs "
                                       "for per-target I_ABC means; basic (pivot) CIs for every ratio and energy summary"},
               "tests": "per-target one-sample t-test over 20 cells (df=19) on per-cell I_ABC; BH q<0.05 within "
                        "triplet (4,928 targets) and across all 512 x 4,928; bootstrap p as a second p-value; "
                        "aggregate sign-flip test per triplet (999 flips), BH across 512 triplets",
               "deployed_for_comparison": {"n_triplets": 4, "pair_mean": 0.3235, "three_way": 0.1896,
                                           "three_way_std": 0.0001, "superadditive": "0 of ~2,980 targets"},
               "wall_seconds_last_call": round(time.time() - t0, 1)}
    H.write_json(OUT / "summary.json", summary)
    H.write_json(OUT / "per_triplet.json", rows)
    H.write_json(OUT / "per_pair.json", pair_rows)
    np.savez_compressed(OUT / "interaction_terms_real.npz",
                        triplets=np.array([[r["a"], r["b"], r["c"]] for r in R_]),
                        I_mean=st["I_mean"][real_idx], ci_lo=st["lo"][real_idx], ci_hi=st["hi"][real_idx],
                        p_t=st["p"][real_idx], q_across_t=Q)
    update_run_config("chunks", {"stage": "analyze", "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
                                 "wall_seconds": round(time.time() - t0, 1), "triplets_done": n_trip,
                                 "device": dev, "status": "DONE"})
    log(json.dumps({"significance": {k: sig[k] for k in ("triplets_any_target_BH_within_triplet_t",
                                                         "triplets_any_target_BH_across_triplets_t",
                                                         "aggregate_signflip_test")},
                    "eshare_I": rs["noise_corrected_energy_share_I_ABC"]["median"],
                    "rho3": rs["noise_corrected_rho3=E[ABC]/E[A+B+C]"]["median"]}, indent=1, default=str))
    log("DONE analyze")


# =============================================================================
# stage: power (no model) -- per-target power and proportional-plant aggregate power (v2 follow-up)
# =============================================================================
def stage_power(args):
    t0 = time.time()
    sel = load_json(SRC / "selection.json")
    meta = load_json(SRC / "cells.json")
    nC = len(meta["rows"])
    triplets, conds, a_groups = build_design(sel)
    D, _, _, _, cidx = load_grid(nC, conds, a_groups, with_logits=False)
    real = [t for t in triplets if t["kind"] == "real"]
    Sf = np.load(OUT / "signflip_signs.npy")
    prng = np.random.default_rng(BOOT_SEED + 2)
    pick = sorted(prng.choice(len(real), size=64, replace=False).tolist())

    def get(t):
        a, b, c = t["a"], t["b"], t["c"]
        J = {"A": (a, NONE, NONE), "B": (NONE, b, NONE), "C": (NONE, NONE, c), "AB": (a, b, NONE),
             "AC": (a, NONE, c), "BC": (NONE, b, c), "ABC": (a, b, c)}
        return {q: D[:, cidx[v]].astype(np.float64) for q, v in J.items()}
    per_target = {}
    for mode in ("constant", "proportional"):
        for kappa in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
            P = []
            for k in pick:
                Dc = get(real[k])
                Iv = Dc["ABC"] - Dc["AB"] - Dc["AC"] - Dc["BC"] + Dc["A"] + Dc["B"] + Dc["C"]
                Iv = Iv + (kappa * Dc["ABC"].mean(0)[None, :] if mode == "constant" else kappa * Dc["ABC"])
                P.append(ttest_p(Iv))
            P = np.stack(P)
            qw = np.stack([bh(p) for p in P])
            qa = bh(P).reshape(P.shape)
            per_target[f"{mode}_kappa{kappa}"] = {"triplets_any_target_BH_within": int((qw < 0.05).any(1).sum()),
                                                  "triplets_any_target_BH_across_64": int((qa < 0.05).any(1).sum()),
                                                  "targets_BH_across_64_total": int((qa < 0.05).sum()),
                                                  "n_triplets": len(pick)}
    log("per-target power done")
    agg = {}
    for kappa in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
        ps = []
        for k in pick:
            Dc = get(real[k])
            Iv = Dc["ABC"] - Dc["AB"] - Dc["AC"] - Dc["BC"] + Dc["A"] + Dc["B"] + Dc["C"] + kappa * Dc["ABC"]
            To = np.abs(Iv.mean(0)).sum()
            Tn = np.abs((Sf / nC) @ Iv).sum(1)
            ps.append(float((1 + (Tn >= To).sum()) / (len(Tn) + 1)))
        agg[str(kappa)] = {"detected_p05": int(sum(p < 0.05 for p in ps)),
                           "detected_BH05_within_64": int(sum(q < 0.05 for q in bh(ps))), "n_triplets": len(pick)}
    log("aggregate power done")
    n_sig = {k: [] for k in ("A", "B", "C", "ABC", "I")}
    for t in real:
        Dc = get(t)
        Iv = Dc["ABC"] - Dc["AB"] - Dc["AC"] - Dc["BC"] + Dc["A"] + Dc["B"] + Dc["C"]
        for k in ("A", "B", "C", "ABC"):
            n_sig[k].append(int((bh(ttest_p(Dc[k])) < 0.05).sum()))
        n_sig["I"].append(int((bh(ttest_p(Iv)) < 0.05).sum()))
    res = {"per_target_power": per_target,
           "per_target_power_note": ("constant: kappa x (cell-mean joint effect) added to every cell; proportional: "
                                     "kappa x (that cell's own joint effect) added to each cell. t-test over 20 cells, "
                                     "BH at q<0.05. Same 64 triplets as the analyze power check."),
           "aggregate_signflip_power_proportional_plant": agg,
           "targets_with_BH_significant_effect_per_triplet(t-test, q<0.05 within triplet)":
               {k: summarize(v) for k, v in n_sig.items()},
           "wall_seconds": round(time.time() - t0, 1)}
    H.write_json(OUT / "power.json", res)
    update_run_config("chunks", {"stage": "power", "end": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                 "wall_seconds": res["wall_seconds"]})
    log(json.dumps(res, indent=1, default=str)[:3000])


# =============================================================================
# stage: crosscheck (independent code, CPU float64)
# =============================================================================
def stage_crosscheck(args):
    """Re-derive the key numbers with separately written code:
      * conditions read straight from the cell files into a dict (no shared loader);
      * per-target p from scipy.stats.ttest_1samp; BH as a step-up rejection rule (not q-values);
      * noise-corrected energy from the pairwise-product (U-statistic) formula
            E = (|sum_i x_i|^2 - sum_i |x_i|^2) / (n (n-1)),
        which equals |mean|^2 - sum_t var_t / n but is computed differently;
      * the aggregate sign-flip test with a new random seed (Monte Carlo agreement expected, not identity);
      * a percentile bootstrap of the median energy share with a new seed."""
    from scipy import stats
    t0 = time.time()
    sel = load_json(SRC / "selection.json")
    meta = load_json(SRC / "cells.json")
    nC = len(meta["rows"])
    A, B, C = sel["features"]["0"], sel["features"]["5"], sel["features"]["9"]
    data = [dict() for _ in range(nC)]
    for f in sorted((SRC / "cells").glob("cell*_g*.npz")):
        ci = int(f.name[4:6])
        g = np.load(f)
        for r, k in enumerate(g["conds"]):
            data[ci][tuple(int(v) for v in k)] = g["dz"][r].astype(np.float64)
    n_conds = {len(d) for d in data}

    def stack(key):
        return np.stack([data[ci][key] for ci in range(nC)])

    def reject_bh(p, q=0.05):
        p = np.asarray(p).ravel()
        m = p.size
        s = np.sort(p)
        ok = np.nonzero(s <= q * np.arange(1, m + 1) / m)[0]
        if ok.size == 0:
            return np.zeros(m, dtype=bool)
        return p <= s[ok.max()]

    def energy_u(X):
        S = X.sum(0)
        return float(((S ** 2).sum() - (X ** 2).sum()) / (X.shape[0] * (X.shape[0] - 1)))

    rng = np.random.default_rng(20261099)
    signs = rng.choice([-1.0, 1.0], size=(999, nC))
    rows = []
    all_p = []
    for a in A:
        for b in B:
            for c in C:
                dA, dB, dC = stack((a, NONE, NONE)), stack((NONE, b, NONE)), stack((NONE, NONE, c))
                dAB, dAC, dBC, dABC = stack((a, b, NONE)), stack((a, NONE, c)), stack((NONE, b, c)), stack((a, b, c))
                Iv = dABC - dAB - dAC - dBC + dA + dB + dC
                with np.errstate(invalid="ignore", divide="ignore"):
                    p = stats.ttest_1samp(Iv, 0.0, axis=0).pvalue
                m = Iv.mean(0)
                p = np.where(np.isnan(p), np.where(m == 0, 1.0, 0.0), p)
                all_p.append(p)
                e_abc, e_add, e_I = energy_u(dABC), energy_u(dA + dB + dC), energy_u(Iv)
                Tobs = np.abs(m).sum()
                Tn = np.abs(signs @ Iv / nC).sum(1)
                # Gram matrices: the U-statistic energy of a resampled set with cell counts w is
                # (w' G w - w . diag G) / (n (n-1))
                rows.append({"a": a, "b": b, "c": c, "rho3": e_abc / e_add, "eshare_I": e_I / e_abc,
                             "any_bh_within": bool(reject_bh(p).any()),
                             "p_signflip": float((1 + (Tn >= Tobs).sum()) / 1000.0),
                             "_GI": Iv @ Iv.T, "_GABC": dABC @ dABC.T})
    P = np.concatenate(all_p)
    rej_across = reject_bh(P).reshape(len(rows), -1)
    q_sf = reject_bh([r["p_signflip"] for r in rows])
    # percentile bootstrap of the median eshare_I with a new seed (cells resampled)
    brng = np.random.default_rng(20261098)
    W = np.stack([np.bincount(brng.integers(0, nC, nC), minlength=nC) for _ in range(1000)]).astype(np.float64)
    GI = np.stack([r["_GI"] for r in rows])
    GA = np.stack([r["_GABC"] for r in rows])

    def eboot(G):
        quad = np.einsum("bi,tij,bj->bt", W, G, W)
        diag = W @ np.stack([np.diag(g) for g in G]).T
        return (quad - diag) / (nC * (nC - 1))
    bmed = np.median(eboot(GI) / eboot(GA), axis=1)
    # the same Gram-matrix formula with w = 1 must give the point estimates
    ones = np.ones(nC)
    gram_point = [float((ones @ gi @ ones - np.trace(gi)) / (ones @ ga @ ones - np.trace(ga))) for gi, ga in zip(GI, GA)]
    gram_vs_u = max(abs(gp - r["eshare_I"]) for gp, r in zip(gram_point, rows))
    an = load_json(OUT / "summary.json")
    pt = load_json(OUT / "per_triplet.json")
    pt_real = [r for r in pt if r["kind"] == "real"]
    assert [(r["a"], r["b"], r["c"]) for r in pt_real] == [(r["a"], r["b"], r["c"]) for r in rows]
    d_rho3 = max(abs(r["rho3"] - q["rho3"]) / max(abs(q["rho3"]), 1e-30) for r, q in zip(rows, pt_real))
    d_esh = max(abs(r["eshare_I"] - q["eshare_I"]) for r, q in zip(rows, pt_real))
    res = {
        "n_cells": nC, "conditions_per_cell": sorted(n_conds), "n_real_triplets": len(rows),
        "triplets_any_target_BH_within(scipy t-test, step-up BH)": int(sum(r["any_bh_within"] for r in rows)),
        "triplets_any_target_BH_across_all": int(rej_across.any(1).sum()),
        "targets_BH_across_all": int(rej_across.sum()),
        "min_p_over_all_tests": float(P.min()),
        "median_rho3_u_statistic": float(np.median([r["rho3"] for r in rows])),
        "median_eshare_I_u_statistic": float(np.median([r["eshare_I"] for r in rows])),
        "max_rel_diff_rho3_vs_analyze": float(d_rho3),
        "max_abs_diff_eshare_I_vs_analyze": float(d_esh),
        "signflip_new_seed": {"n_p05": int(sum(r["p_signflip"] < 0.05 for r in rows)),
                              "n_BH05": int(np.sum(q_sf))},
        "eshare_I_median_percentile_ci95_new_seed(1000 reps, Gram-matrix formula)": ci95(bmed),
        "gram_formula_vs_u_statistic_max_abs_diff": float(gram_vs_u),
        "analyze_values": {
            "triplets_any_target_BH_within": an["significance_I_ABC"]["triplets_any_target_BH_within_triplet_t"],
            "triplets_any_target_BH_across": an["significance_I_ABC"]["triplets_any_target_BH_across_triplets_t"],
            "min_p": an["significance_I_ABC"]["min_p_t_over_all_tests"],
            "rho3_median": an["redundancy_ratios"]["noise_corrected_rho3=E[ABC]/E[A+B+C]"]["median"],
            "eshare_I_median": an["redundancy_ratios"]["noise_corrected_energy_share_I_ABC"]["median"],
            "eshare_I_ci": an["redundancy_ratios"]["noise_corrected_energy_share_I_ABC"]["ci95_of_median"],
            "signflip_p05": an["significance_I_ABC"]["aggregate_signflip_test"]["n_triplets_p05_uncorrected"],
            "signflip_BH05": an["significance_I_ABC"]["aggregate_signflip_test"]["n_triplets_BH05_across_triplets"]},
        "wall_seconds": round(time.time() - t0, 1)}
    H.write_json(OUT / "crosscheck.json", res)
    update_run_config("chunks", {"stage": "crosscheck", "end": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                 "wall_seconds": res["wall_seconds"]})
    log(json.dumps(res, indent=1, default=str))


# =============================================================================
# stage: compare (v3 vs v2 vs deployed)
# =============================================================================
def stage_compare(args):
    v3 = load_json(OUT / "summary.json")
    v2 = load_json(V2 / "summary.json")
    p3 = load_json(OUT / "power.json")
    v2c = load_json(V2F / "classnull.json")
    v2a = load_json(V2F / "align.json")
    r3 = load_json(OUT / "readout_summary.json")
    r2 = load_json(V2F / "readout_summary.json")

    def g(d, *ks):
        for k in ks:
            d = d[k]
        return d
    rows = []

    def add(name, *path, src3=v3, src2=v2):
        try:
            a2 = g(src2, *path)
        except Exception:
            a2 = None
        try:
            a3 = g(src3, *path)
        except Exception:
            a3 = None
        rows.append({"quantity": name, "v2": a2, "v3": a3})
    S = "significance_I_ABC"
    RR = "redundancy_ratios"
    add("triplets with any target BH within (t)", S, "triplets_any_target_BH_within_triplet_t")
    add("triplets with any target BH across (t)", S, "triplets_any_target_BH_across_triplets_t")
    add("aggregate sign-flip p<0.05 (of 512)", S, "aggregate_signflip_test", "n_triplets_p05_uncorrected")
    add("aggregate sign-flip BH", S, "aggregate_signflip_test", "n_triplets_BH05_across_triplets")
    add("T_obs / T_null median", S, "aggregate_signflip_test", "T_obs_over_T_null_median", "median")
    add("all-orders sign-flip p<0.05", S, "aggregate_signflip_test_all_orders(d_ABC - d_A - d_B - d_C)", "n_triplets_p05_uncorrected")
    add("all-orders sign-flip BH", S, "aggregate_signflip_test_all_orders(d_ABC - d_A - d_B - d_C)", "n_triplets_BH05_across_triplets")
    add("energy share I_ABC median", RR, "noise_corrected_energy_share_I_ABC", "median")
    add("energy share I_ABC CI", RR, "noise_corrected_energy_share_I_ABC", "ci95_of_median")
    add("energy share all orders median", RR, "noise_corrected_energy_share_all_interactions", "median")
    add("energy share all orders CI", RR, "noise_corrected_energy_share_all_interactions", "ci95_of_median")
    add("triplets eshare_I CI above 0", RR, "n_triplets_eshare_I_ci_above_0")
    add("rho3 median", RR, "noise_corrected_rho3=E[ABC]/E[A+B+C]", "median")
    add("rho3 CI", RR, "noise_corrected_rho3=E[ABC]/E[A+B+C]", "ci95_of_median")
    add("rho3 uncorrected median", RR, "noise_corrected_rho3=E[ABC]/E[A+B+C]", "median_plugin_uncorrected")
    add("rho3 triplets CI>1", RR, "n_triplets_rho3_ci_above_1")
    add("rho3 triplets CI<1", RR, "n_triplets_rho3_ci_below_1")
    add("rho2 median", RR, "noise_corrected_rho2_pair_mean", "median")
    add("rho2 CI", RR, "noise_corrected_rho2_pair_mean", "ci95_of_median")
    add("rho3 logits median", RR, "noise_corrected_rho3_logits", "median")
    add("rho3 logits CI", RR, "noise_corrected_rho3_logits", "ci95_of_median")
    add("eshare_I logits median", RR, "noise_corrected_energy_share_I_ABC_logits", "median")
    add("per-cell I share (codes)", S, "percell_nonadditivity(median over triplets of the median over cells)", "I_ABC_energy_share")
    add("per-cell I share (logits)", S, "percell_nonadditivity(median over triplets of the median over cells)", "I_ABC_energy_share_logits")
    add("three-way ratio median", RR, "three_way", "median")
    add("three-way ratio CI", RR, "three_way", "ci95_of_median")
    add("three-way ratio if additive", RR, "three_way_if_additive", "median")
    add("pair mean ratio median", RR, "pairwise_mean", "median")
    add("pair mean ratio CI", RR, "pairwise_mean", "ci95_of_median")
    add("pair mean if additive", RR, "pairwise_mean_if_additive", "median")
    add("three-way logits", RR, "three_way_logits", "median")
    add("marginal C|AB", RR, "marginal_C_given_AB", "median")
    add("three-way SD across triplets", "sanity", "three_way_ratio_std_across_triplets")
    add("deployed rule targets", "superadditivity", "deployed_targets(|e_ABC|>0.01)")
    add("deployed rule superadditive", "superadditivity", "deployed_superadditive(>1.1x)")
    add("spec included", "superadditivity", "spec_targets_included(|d|>0.5 any condition)")
    add("spec strict superadditive", "superadditivity", "spec_superadditive_strict(|d_ABC|>sum)")
    add("pairs agg BH", "pairwise_interactions", "pairs_agg_signflip_BH05")
    add("pairs any target BH", "pairwise_interactions", "pairs_any_target_BH_within_t")
    add("pair share I2 median", "pairwise_interactions", "share_I2_median")
    add("pair share I2 noise", "pairwise_interactions", "share_I2_noise_median")
    for o in ("1", "2", "3"):
        add(f"KL order {o} median", "logit_effect_size_by_order", o, "kl_cellmean", "median")
    for kappa in ("0.05", "0.1", "0.2"):
        add(f"power constant plant {kappa} (signflip p<.05)", "planted_interaction_power", kappa, "detected_signflip_p05")
    for kappa in ("0.05", "0.1", "0.2", "0.5", "1.0"):
        rows.append({"quantity": f"power proportional {kappa}: aggregate p<.05 / BH",
                     "v2": [v2a["aggregate_signflip_power_proportional_plant"][kappa]["detected_p05"],
                            v2a["aggregate_signflip_power_proportional_plant"][kappa]["detected_BH05_within_64"]],
                     "v3": [p3["aggregate_signflip_power_proportional_plant"][kappa]["detected_p05"],
                            p3["aggregate_signflip_power_proportional_plant"][kappa]["detected_BH05_within_64"]]})
        k = f"proportional_kappa{kappa}"
        rows.append({"quantity": f"power proportional {kappa}: per-target triplets BH within / across",
                     "v2": [v2c["per_target_power"][k]["triplets_any_target_BH_within"],
                            v2c["per_target_power"][k]["triplets_any_target_BH_across_64"]],
                     "v3": [p3["per_target_power"][k]["triplets_any_target_BH_within"],
                            p3["per_target_power"][k]["triplets_any_target_BH_across_64"]]})
    key = "targets_with_BH_significant_effect_per_triplet(t-test, q<0.05 within triplet)"
    for k in ("A", "B", "C", "ABC", "I"):
        rows.append({"quantity": f"targets with BH effect per triplet, {k} (median/mean)",
                     "v2": [v2a[key][k]["median"], v2a[key][k]["mean"]], "v3": [p3[key][k]["median"], p3[key][k]["mean"]]})
    for nm in ("h11", "sae_preact", "sae_code", "logits_centered"):
        rows.append({"quantity": f"readout per-position I share, {nm} (median)",
                     "v2": r2[nm]["per_position_energy_share_I"]["median"],
                     "v3": r3[nm]["per_position_energy_share_I"]["median"]})
    rows.append({"quantity": "readout share of code I energy on top-k switching entries",
                 "v2": r2["sae_code"]["share_of_I_energy_on_switching_entries"]["median"],
                 "v3": r3["sae_code"]["share_of_I_energy_on_switching_entries"]["median"]})
    rows.append({"quantity": "readout positions with top-k set change under ABC",
                 "v2": r2["sae_code"]["positions_with_topk_set_change_vs_clean_ABC"]["median"],
                 "v3": r3["sae_code"]["positions_with_topk_set_change_vs_clean_ABC"]["median"]})
    rows.append({"quantity": "|dh11 ABC| / |h11| (median)", "v2": r2["rel_size_dh11_ABC_over_h11"]["median"],
                 "v3": r3["rel_size_dh11_ABC_over_h11"]["median"]})
    rows.append({"quantity": "single effect eff range by layer", "v2": None,
                 "v3": v3["sanity"]["single_effect_eff_range_by_layer"]})
    v2_single = v2["sanity"]["single_effect_eff_by_feature"]
    rows[-1]["v2"] = {k: [min(v.values()), max(v.values())] for k, v in v2_single.items()}
    out = {"rows": rows, "deployed": v3["deployed_for_comparison"],
           "note": "v2 = deployed SAEs + wrongly encoded inputs + fixed hooks; v3 = v3 SAEs + right inputs + same hooks. "
                   "Feature IDs differ between v2 and v3 (different SAEs); only design-level numbers are comparable."}
    H.write_json(OUT / "compare.json", out)
    for r in rows:
        print(f"{r['quantity']:<70} v2={json.dumps(r['v2'], default=str)[:60]:<62} v3={json.dumps(r['v3'], default=str)[:60]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "verify", "run", "recheck", "readout", "analyze", "power",
                                      "crosscheck", "compare"])
    ap.add_argument("--max-minutes", type=float, default=7.0)
    ap.add_argument("--device", default="", help="analyze: torch device for the bootstrap (default cpu)")
    args = ap.parse_args()
    {"select": stage_select, "verify": stage_verify, "run": stage_run, "recheck": stage_recheck,
     "readout": stage_readout, "analyze": stage_analyze, "power": stage_power, "crosscheck": stage_crosscheck,
     "compare": stage_compare}[args.stage](args)


if __name__ == "__main__":
    main()
