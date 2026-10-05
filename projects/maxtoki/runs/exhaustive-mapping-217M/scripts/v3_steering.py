"""v3 trajectory steering (item V3-8): v3 SAEs, correct inputs, and a defined maturity axis.

Why a v3
--------
* Deployed Experiment 3 (scripts/experiment3_rerun.py): its hook deleted block l (L0-L9) or applied the
  final norm twice (L11). The published table is reproduced by a zero edit (V2_HOOKS_REPORT.md).
* Its "pseudotime" was PC1 of 200 random Tabula Sapiens immune cells of 28 cell types (cell type explains
  94.5% of it; ledger R18), and the gene stories built on it were wrong (R19).
* v2 (scripts/v2_steering.py) fixed the hooks but kept the PC1 axis, fed log1p(CP10k) as counts
  (checks/INPUT_ENCODING_AUDIT.md) and used the deployed SAEs, which do not fit correctly encoded inputs
  (V3_SAE_REPORT.md). It stopped after 34 of 115 work units.

This script
-----------
* Inputs: Tabula Sapiens immune, raw/X integer counts, via setup/inputs_v3.py. Every cell passes
  inputs_v3.assert_encoding_batch right before its forward passes (abort otherwise).
* SAEs: runs/sae-atlas-217M/outputs/v3_sae (retrained on correctly encoded K562 inputs).
* Edits: setup/hooks_v2.py only (Steer mode="scale"; AddVector for the positive control).
* Maturity axis: one differentiation series with known stage labels from the cell-type annotation,
  bone marrow, 10x 3' v3 only. Fixed in design.json before any forward pass (stage `design`).
* Selection cells (donors TSP14, TSP21, TSP25) give the signatures g_early / g_late and the feature
  choice. Steered cells are 50 HSCs from other donors (TSP2, TSP27). So nothing about the steered
  cells is used to build the axis or to choose features.
* Speed: an edit at layer l changes nothing before block l (V2_HOOKS_REPORT test ii). So each steered
  row reruns only blocks l..10 from the cached clean input of block l, with hooks_v2 placing the edit.
  The mean over positions of the logits is computed as lm_head(mean of normed hidden) (lm_head is
  linear, no bias) for l <= 10; for l = 11 the edited lm_head input goes through lm_head in full.
  Every row is compared with a clean reference computed by the same path, and stage `verify` checks
  the shortcut against the full forward pass with hooks_v2.run_with_edits.

Stages (run in this order; model stages are resumable; repeat until they print DONE):
  design     write design.json (rules, groups, thresholds)                  (no model)
  prep       choose cells, tokenize from raw/X, encoding checks             (no model)
  clean      one clean pass per cell: mean logits, SAE stats at 5 layers    (model)
  select     signatures, axis check, feature choice, null draw, controls   (no model)
  steer      50 HSCs x 5 layers x (3 + 20 features x 3 alphas + 2 controls) (model)
  (verify, summarize, crosscheck: scripts/v3_steering_verify.py, v3_steering_summarize.py)
Usage (from projects/maxtoki):
  OMP_NUM_THREADS=4 .venv/bin/python runs/exhaustive-mapping-217M/scripts/v3_steering.py <stage> [--max-minutes 7.5]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import torch
import torch.nn.functional as Fnn

torch.set_num_threads(4)

PROJ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJ / "setup"))
import hooks_v2 as H  # noqa: E402
import inputs_v3 as I  # noqa: E402

RUN = PROJ / "runs/exhaustive-mapping-217M"
OUT = RUN / "outputs/v3_steering"
CLEAN_DIR = OUT / "clean"
STEER_DIR = OUT / "steer"
VERIFY_DIR = OUT / "verify"
SAE_DIR = PROJ / "runs/sae-atlas-217M/outputs/v3_sae"
DEP3 = RUN / "outputs/experiment3"
V2S = RUN / "outputs/v2_steering"
V2Z = RUN / "outputs/v2_zero_edit_control"
SCRIPT_PATH = Path(__file__).resolve()

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DATASET = "ts_immune"
MAX_LEN = 2048
LAYERS = [0, 3, 6, 9, 11]
D_SAE = 4928
VOCAB = 20275
TISSUE = "bone marrow"
ASSAY = "10x 3' v3"
STAGES = {0: "hematopoietic stem cell", 1: "common myeloid progenitor",
          2: "erythroid progenitor cell", 3: "erythrocyte"}
STAGE_SHORT = {0: "HSC", 1: "CMP", 2: "EryP", 3: "Ery"}
SEL_DONORS = ["TSP14", "TSP21", "TSP25"]
STEER_DONORS = ["TSP2", "TSP27"]
N_SEL_PER_STAGE_DONOR = 30
N_STEER = 50
N_VAL_PER_STAGE = 30          # CMP and EryP cells from the steering donors (axis check only)
ALIVE_MIN_FRAC = 0.5          # alive = active at >= 1 gene position in >= 50% of selection HSC cells
N_CAND = 3
N_NULL = 20
NULL_K_NEAREST = 50
NULL_SPLIT = [7, 7, 6]
ALPHAS = [0.0, 2.0, 5.0]
SEED = 20261003               # cell sampling
NULL_SEED = 20261003          # null draw: default_rng(NULL_SEED + layer)
BOOT_SEED = 20261003
N_BOOT = 10000
MIN_FREE_GB = 3.0
AXIS_MIN_AUROC = 0.80         # axis check on steering-donor cells (HSC vs EryP)
SIG_NAMES = ["g_early", "g_late", "g_cmp", "g_eryp", "g_all"]


def log(msg):
    print(msg, flush=True)


def load_json(p, default=None):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default


def cosd(a, b) -> float:
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b / (na * nb))


def update_run_config(key=None, value=None, chunk=None):
    p = OUT / "run_config.json"
    cfg = load_json(p, {"chunks": []})
    cfg.update(H.env_info(DEVICE))
    cfg.update({"script": str(SCRIPT_PATH), "script_sha256_latest": H.sha256_file(SCRIPT_PATH),
                "hooks_v2": str(PROJ / "setup/hooks_v2.py"),
                "inputs_v3": str(PROJ / "setup/inputs_v3.py"),
                "inputs_v3_sha256": H.sha256_file(PROJ / "setup/inputs_v3.py"),
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                "seeds": {"cell_sampling": SEED, "null_draw": f"{NULL_SEED} + layer", "bootstrap": BOOT_SEED,
                          "torch_numpy_global": SEED}})
    if key is not None:
        cfg[key] = value
    if chunk is not None:
        cfg["chunks"].append(chunk)
    H.write_json(p, cfg)


# ============================================================================= design
def stage_design():
    p = OUT / "design.json"
    if p.exists():
        log(f"design.json exists (written {load_json(p)['written']}); not rewritten")
        return
    d = {
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": "Written before any forward pass of this item. Later stages read these values; they are not changed.",
        "differentiation_series": {
            "tissue": TISSUE, "assay": ASSAY, "dataset": DATASET,
            "stages": {str(k): v for k, v in STAGES.items()},
            "why": ("Erythroid branch of haematopoiesis: hematopoietic stem cell -> common myeloid progenitor "
                    "(upstream of the megakaryocyte-erythroid progenitor) -> erythroid progenitor cell -> "
                    "erythrocyte. All four labels exist in bone marrow 10x 3' v3 cells of several donors. "
                    "Stage numbers are ordinal labels from the cell-type annotation, not measured time. "
                    "'hematopoietic precursor cell' is left out because its free annotations mix granulocyte, "
                    "monocyte and other cells. Smart-seq2 cells are left out (read counts; MaxToki-217M handles "
                    "them badly, checks/V3_INPUTS_REPORT.md). 10x 5' v2 cells are left out so that one assay "
                    "covers every stage."),
            "early": STAGES[0], "late": STAGES[3],
        },
        "cells": {
            "selection": f"donors {SEL_DONORS}; up to {N_SEL_PER_STAGE_DONOR} random cells per (stage, donor); "
                         "used for the signatures, the axis and the feature choice",
            "steered": f"{N_STEER} random HSCs from donors {STEER_DONORS} (donor-disjoint from selection)",
            "axis_validation": f"{N_VAL_PER_STAGE} random CMP and {N_VAL_PER_STAGE} random EryP cells from donors "
                               f"{STEER_DONORS}, plus the {N_STEER} steered HSCs (clean passes only)",
            "sampling": f"numpy default_rng({SEED}); groups drawn in a fixed order; row indices sorted",
        },
        "input_encoding": "raw/X integer counts -> counts / total * 1e4 / Geneformer gene median, ranked, first "
                          f"{MAX_LEN - 2} genes, <bos>/<eos> (inputs_v3.tokenize_counts); "
                          "assert_encoding_batch on every cell right before its forward passes",
        "readout": {
            "m": "logits averaged over all positions of the sequence (as deployed)",
            "signatures": "g_early = mean m of selection HSCs; g_late = mean m of selection erythrocytes "
                          "(also g_cmp, g_eryp, g_all for re-scoring)",
            "score": "s(m) = cos(m, g_late) - cos(m, g_early)",
            "delta_s": "s(m_steered) - s(m_clean) for the same cell (clean = same computation path, no edit)",
            "gap_share": "(m_steered - m_clean) . (g_late - g_early) / |g_late - g_early|^2 "
                         "(share of the early-to-late distance moved along that line)",
            "natural_span": "S = mean s over selection erythrocytes - mean s over selection HSCs; "
                            "delta_s is also given as a percentage of S",
        },
        "axis_check": f"on steering-donor cells (never used to build g): AUROC of s for EryP vs HSC >= "
                      f"{AXIS_MIN_AUROC} and Spearman(s, stage) > 0 over HSC/CMP/EryP; steering is not run if it fails",
        "features": {
            "layers": LAYERS, "sae_dir": str(SAE_DIR),
            "per_cell_activation": "a_f(cell) = mean over gene positions (not <bos>/<eos>) of the v3 SAE code z_f",
            "alive": f"z_f > 0 at >= 1 gene position in >= {ALIVE_MIN_FRAC:.0%} of selection HSC cells",
            "maturity_correlation": "rho_f = mean over the 3 selection donors of the within-donor Spearman "
                                    "correlation between a_f and the stage number (0-3); constant -> 0",
            "candidates": f"the {N_CAND} alive features with the largest |rho_f| (ties: lower feature id)",
            "null": f"{N_NULL} alive features per layer, not candidates, drawn at random (rng(NULL_SEED + layer)) "
                    f"without replacement, {NULL_SPLIT} from the {NULL_K_NEAREST} alive features nearest to each "
                    "candidate in log10(mean a_f over selection HSCs) (= mean edit size per gene position)",
            "predicted_sign": "amplify (alpha 2, 5): sign(delta_s) = sign(rho_f); remove (alpha 0): -sign(rho_f)",
        },
        "edit": "hooks_v2.Steer(layer, f, alpha, mode='scale', positions=gene positions): "
                "delta = (alpha - 1) z_f(x) W_dec[:, f] at every gene position (alpha 0 removes the feature; "
                "alpha 1 adds nothing)",
        "alphas": ALPHAS,
        "positive_control": "hooks_v2.AddVector(layer, v, positions=gene positions) with v = mean gene-position "
                            "residual at the site over selection erythrocytes minus over selection HSCs; "
                            "two sizes: full v, and v rescaled to the mean alpha = 5 edit size of the 3 candidates "
                            "(4 x mean over candidates of mean a_f over selection HSCs)",
        "statistics": {
            "ci": f"per-cell percentile bootstrap of the mean over the {N_STEER} steered cells, {N_BOOT} resamples, "
                  f"seed {BOOT_SEED} (same resamples for every unit)",
            "empirical_p": "per candidate and alpha: directional p = (1 + #null features with predicted-sign "
                           "delta_s >= the candidate's) / (1 + 20); size p uses |delta_s|; the smallest possible "
                           "p is 1/21 = 0.048",
            "layer_test": "3 candidates vs all C(23,3) = 1,771 subsets of the 23 features (directional stat uses "
                          "each feature's own sign of rho)",
            "pooled": "Spearman(rho_f, delta_s_f) over 23 features per layer and over all 115; stratified "
                      "permutation (labels permuted within layer, 100,000 draws) for the sum over layers of the "
                      "candidates' mean directional rank",
        },
    }
    H.write_json(p, d)
    update_run_config("design_sha256", H.sha256_file(p))
    log(f"wrote {p}")


# ============================================================================= prep
def obs_frame():
    import pandas as pd
    cols = {}
    for c in ["cell_type", "tissue", "assay", "donor_id"]:
        cols[c] = I.read_obs_column(DATASET, c)
    import h5py
    with h5py.File(I.DATASETS[DATASET]["path"], "r") as f:
        k = f["obs"].attrs.get("_index", "_index")
        k = k.decode() if isinstance(k, bytes) else k
        names = np.array([s.decode() if isinstance(s, bytes) else s for s in f["obs"][k][:]], dtype=object)
    cols["obs_name"] = names
    return pd.DataFrame(cols)


def stage_prep():
    if (OUT / "cells.npz").exists():
        log("cells.npz exists")
        return
    if not (OUT / "design.json").exists():
        raise SystemExit("run stage design first")
    t0 = time.time()
    df = obs_frame()
    base = (df.tissue == TISSUE) & (df.assay == ASSAY)
    rng = np.random.default_rng(SEED)
    groups = []   # (group, stage, donor, rows)
    for s in range(4):
        for d in SEL_DONORS:
            pool = np.where(base & (df.cell_type == STAGES[s]) & (df.donor_id == d))[0]
            k = min(N_SEL_PER_STAGE_DONOR, len(pool))
            rows = np.sort(rng.choice(pool, size=k, replace=False)) if k else np.zeros(0, int)
            groups.append(("selection", s, d, rows))
    pool = np.where(base & (df.cell_type == STAGES[0]) & df.donor_id.isin(STEER_DONORS))[0]
    groups.append(("steer", 0, "+".join(STEER_DONORS), np.sort(rng.choice(pool, size=N_STEER, replace=False))))
    for s in (1, 2):
        pool = np.where(base & (df.cell_type == STAGES[s]) & df.donor_id.isin(STEER_DONORS))[0]
        groups.append(("validation", s, "+".join(STEER_DONORS),
                       np.sort(rng.choice(pool, size=min(N_VAL_PER_STAGE, len(pool)), replace=False))))
    rows_all = np.concatenate([g[3] for g in groups]).astype(np.int64)
    if len(np.unique(rows_all)) != len(rows_all):
        raise SystemExit("a cell was drawn twice")
    grp = np.concatenate([[g[0]] * len(g[3]) for g in groups]).astype(object)
    stage = np.concatenate([[g[1]] * len(g[3]) for g in groups]).astype(np.int64)
    cells, kept, info, counts = I.tokenize_rows(DATASET, rows_all, MAX_LEN, return_counts=True)
    if kept != [int(r) for r in rows_all]:
        raise SystemExit(f"{len(rows_all) - len(kept)} cells did not tokenize; stop and look")
    vm = I.get_var_map(DATASET)
    chk = I.assert_encoding_batch(counts, cells, vm, MAX_LEN, label="prep: all cells")
    # power of the check: the deployed path (tokenize_cell on stored X) must fail it
    dep = I.deployed_tokenize_X(DATASET, rows_all, MAX_LEN)
    dep_fail = sum(not I.check_encoding(c, t, vm, MAX_LEN)["pass"] for c, t in zip(counts, dep))
    agree = [I.order_agreement(I.full_order(c, vm), I.full_order(c, vm), MAX_LEN)["spearman_full"]
             for c in counts[:2]]
    old_vs_new = []
    for c, t in zip(counts, dep):
        old = np.asarray(t.token_ids[1:-1])
        new = I.full_order(c, vm)
        old_vs_new.append(len(set(old[:200].tolist()) & set(new[:200].tolist())) / min(200, len(new)))
    real = df.iloc[rows_all]
    toks = np.empty(len(cells), dtype=object)
    for i, c in enumerate(cells):
        toks[i] = np.asarray(c.token_ids, dtype=np.int64)
    cnt_sha = I.array_sha256(np.stack(counts).astype(np.float64))
    np.savez(OUT / "cells.npz", rows=rows_all, group=grp, stage=stage,
             donor=real.donor_id.to_numpy().astype(object), cell_type=real.cell_type.to_numpy().astype(object),
             obs_name=real.obs_name.to_numpy().astype(object), token_ids=toks,
             n_tokens=np.array([len(t) for t in toks], dtype=np.int64),
             counts_sha256=np.array([I.array_sha256(c) for c in counts], dtype=object))
    prep = {"n_cells": len(rows_all),
            "groups": [{"group": g[0], "stage": g[1], "stage_label": STAGES[g[1]], "donor": g[2], "n": int(len(g[3]))}
                       for g in groups],
            "available": {f"{STAGES[s]}|{d}": int((base & (df.cell_type == STAGES[s]) & (df.donor_id == d)).sum())
                          for s in range(4) for d in SEL_DONORS + STEER_DONORS},
            "encoding_check": chk, "tokenize_info": info,
            "deployed_path_fails_check": f"{dep_fail} of {len(dep)}",
            "deployed_vs_correct_top200_overlap_mean": float(np.mean(old_vs_new)),
            "deployed_vs_correct_top200_overlap_range": [float(np.min(old_vs_new)), float(np.max(old_vs_new))],
            "self_agreement_sanity": agree,
            "counts_sha256_all": cnt_sha,
            "n_tokens_by_group_stage": {f"{g}|{STAGE_SHORT[s]}": [int(np.median(
                [len(toks[i]) for i in range(len(toks)) if grp[i] == g and stage[i] == s])),
                int(sum(1 for i in range(len(toks)) if grp[i] == g and stage[i] == s))]
                for g in ("selection", "steer", "validation") for s in range(4)
                if any(grp[i] == g and stage[i] == s for i in range(len(toks)))},
            "wall_s": round(time.time() - t0, 1)}
    H.write_json(OUT / "prep.json", prep)
    update_run_config("inputs", I.encoding_record(DATASET, rows=rows_all, check_summary=chk, counts_sha256=cnt_sha))
    update_run_config("cells", {"rows": [int(r) for r in rows_all], "obs_names": [str(x) for x in real.obs_name],
                                "group": [str(x) for x in grp], "stage": [int(x) for x in stage],
                                "donor": [str(x) for x in real.donor_id]})
    log(json.dumps({k: v for k, v in prep.items() if k not in ("available", "tokenize_info")}, indent=1, default=str))


def load_cells():
    c = np.load(OUT / "cells.npz", allow_pickle=True)
    return {k: c[k] for k in c.files}


# ============================================================================= model helpers
class Model:
    def __init__(self):
        H.check_memory(MIN_FREE_GB)
        xt = H.load_model(DEVICE)
        self.model = xt.model
        self.model.eval()
        self.saes = H.load_saes(LAYERS, device=DEVICE, sae_dir=SAE_DIR)
        for l, s in self.saes.items():
            assert Path(s.path).parent.parent == SAE_DIR, s.path
        self.W = self.model.lm_head.weight          # (V, D)
        assert self.model.lm_head.bias is None
        self.nb = int(self.model.config.num_hidden_layers)
        self.reader = I.CountReader(DATASET).__enter__()
        self.vm = I.get_var_map(DATASET)

    def check_cell(self, cells, i, label):
        """Encoding check right before the forward passes of this cell (re-reads raw/X)."""
        c = self.reader.counts(int(cells["rows"][i]))
        if I.array_sha256(c) != cells["counts_sha256"][i]:
            raise I.EncodingError(f"{label}: counts of row {cells['rows'][i]} changed since prep")
        return I.assert_encoding_batch([c], [cells["token_ids"][i]], self.vm, MAX_LEN, label=label)

    @torch.no_grad()
    def clean_full(self, ids):
        """Full clean pass. Returns mean logits (float64 CPU), captured site inputs, layer kwargs."""
        kw = {}

        def grab(module, args, kwargs):
            kw.update(kwargs)
            return None
        h = self.model.model.layers[0].register_forward_pre_hook(grab, with_kwargs=True)
        try:
            with H.ResidualEditor(self.model, self.saes, edits=[], capture=LAYERS) as ed:
                out = self.model(ids, use_cache=False, return_dict=True)
        finally:
            h.remove()
        ml = out.logits[0].mean(0).float().cpu().double().numpy()
        del out
        caps = {l: ed.captured[l]["pre"] for l in LAYERS}
        return ml, caps, kw

    @torch.no_grad()
    def partial(self, layer, h_in, kw, edits):
        """Mean-over-positions logits when the forward pass starts at the input of `layer`
        (h_in = clean input of that site, (1, T, D) on device), with hooks_v2 edits."""
        with H.ResidualEditor(self.model, self.saes, edits=edits) as ed:
            if layer < self.nb:
                h = h_in
                for k in range(layer, self.nb):
                    h = self.model.model.layers[k](h, **kw)
                hn = self.model.model.norm(h)
                ml = Fnn.linear(hn[0].mean(0), self.W)
            else:
                ml = self.model.lm_head(h_in)[0].mean(0)
        return ml.float().cpu().double().numpy(), ed.edit_log.get(layer, {})


def gene_mask(T):
    m = np.ones(T, dtype=bool)
    m[0] = False
    m[T - 1] = False
    return m


# ============================================================================= clean
def stage_clean(max_min):
    t_start = time.time()
    cells = load_cells()
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    n = len(cells["rows"])
    todo = [i for i in range(n) if not (CLEAN_DIR / f"cell_{i:03d}.npz").exists()]
    chunk = {"stage": "clean", "start": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "script_sha256": H.sha256_file(SCRIPT_PATH), "free_gb_at_start": round(H.available_memory_gb(), 2),
             "cells_done": [], "encoding_checks_passed": 0}
    if not todo:
        log("clean: all cells done")
        return True
    M = Model()
    est = 3.0
    done = False
    try:
        for i in todo:
            if (time.time() - t_start) + est * 1.5 > max_min * 60:
                break
            t0 = time.time()
            chunk["min_free_gb"] = min(chunk.get("min_free_gb", 99), H.check_memory(MIN_FREE_GB))
            M.check_cell(cells, i, f"clean cell {i}")
            chunk["encoding_checks_passed"] += 1
            tok = cells["token_ids"][i]
            T = len(tok)
            ids = torch.from_numpy(tok[None, :]).to(DEVICE)
            ml, caps, kw = M.clean_full(ids)
            gm = torch.from_numpy(gene_mask(T)).to(DEVICE)
            rec = {"mean_logits": ml.astype(np.float32), "T": T}
            for l in LAYERS:
                x = caps[l][0].to(DEVICE)                        # (T, D)
                s = M.saes[l]
                z = s.encode(x)
                xh = z @ s.W_dec.T + s.mu
                err = ((x - xh) ** 2).sum(1)                     # (T,)
                zg = z[gm]
                rec[f"meanz_L{l}"] = zg.mean(0).float().cpu().numpy()
                rec[f"nact_L{l}"] = (zg > 0).sum(0).int().cpu().numpy()
                rec[f"meanres_L{l}"] = x[gm].mean(0).float().cpu().numpy()
                for nm, msk in (("all", torch.ones(T, dtype=torch.bool, device=DEVICE)), ("gene", gm)):
                    xs = x[msk].cpu().double()
                    rec[f"sse_{nm}_L{l}"] = float(err[msk].cpu().double().sum())
                    rec[f"sumx_{nm}_L{l}"] = xs.sum(0).cpu().numpy()
                    rec[f"sumsq_{nm}_L{l}"] = float((xs ** 2).sum().cpu())
                    rec[f"n_{nm}_L{l}"] = int(msk.sum().cpu())
                del x, z, xh, err, zg
            tmp = CLEAN_DIR / f"cell_{i:03d}.tmp.npz"
            np.savez(tmp, **rec)
            tmp.replace(CLEAN_DIR / f"cell_{i:03d}.npz")
            chunk["cells_done"].append(i)
            del caps, kw, ids
            if len(chunk["cells_done"]) % 10 == 0:
                H.free_device_cache()
                log(f"clean cell {i} T={T} {time.time() - t0:.1f}s elapsed {(time.time() - t_start) / 60:.1f} min")
            est = time.time() - t0
        done = len([i for i in range(n) if not (CLEAN_DIR / f"cell_{i:03d}.npz").exists()]) == 0
    except H.MemoryGuardError as e:
        chunk["stopped"] = str(e)
        log(f"memory guard: {e}")
    finally:
        chunk["wall_seconds"] = round(time.time() - t_start, 1)
        chunk["end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        update_run_config(chunk=chunk)
    log("clean: DONE" if done else f"clean: {len(chunk['cells_done'])} cells this chunk; run again")
    return done


def load_clean(idx):
    return [dict(np.load(CLEAN_DIR / f"cell_{i:03d}.npz")) for i in idx]


# ============================================================================= select
def spearman_cols(A, y):
    """Spearman of every column of A (n, F) with y (n,), average ranks; constant column -> 0."""
    from scipy.stats import rankdata
    R = rankdata(A, axis=0)
    ry = rankdata(y)
    Rc = R - R.mean(0)
    yc = ry - ry.mean()
    den = np.sqrt((Rc ** 2).sum(0) * (yc ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 0, (Rc * yc[:, None]).sum(0) / np.where(den > 0, den, 1.0), 0.0)
    return r


def auroc(pos, neg):
    """AUROC with average ranks for ties (Mann-Whitney)."""
    from scipy.stats import rankdata
    x = np.concatenate([pos, neg])
    r = rankdata(x)
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def score(m, gL, gE):
    return cosd(m, gL) - cosd(m, gE)


def stage_select():
    if (OUT / "selection.json").exists():
        log("selection.json exists")
        return True
    from scipy.stats import spearmanr
    cells = load_cells()
    n = len(cells["rows"])
    missing = [i for i in range(n) if not (CLEAN_DIR / f"cell_{i:03d}.npz").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} clean cells missing; run clean")
    grp, stage, donor = cells["group"], cells["stage"], cells["donor"]
    sel = np.where(grp == "selection")[0]
    C = load_clean(range(n))
    ML = np.stack([c["mean_logits"] for c in C]).astype(np.float64)
    sig = {"g_early": ML[sel][stage[sel] == 0].mean(0), "g_late": ML[sel][stage[sel] == 3].mean(0),
           "g_cmp": ML[sel][stage[sel] == 1].mean(0), "g_eryp": ML[sel][stage[sel] == 2].mean(0),
           "g_all": ML[sel].mean(0)}
    gE, gL = sig["g_early"], sig["g_late"]
    s_all = np.array([score(ML[i], gL, gE) for i in range(n)])
    # ---- axis description and check
    axis = {"cos_g_early_g_late": cosd(gE, gL), "norm_g_late_minus_g_early": float(np.linalg.norm(gL - gE)),
            "selection_mean_s_by_stage": {STAGE_SHORT[s]: float(s_all[sel][stage[sel] == s].mean()) for s in range(4)},
            "selection_sd_s_by_stage": {STAGE_SHORT[s]: float(s_all[sel][stage[sel] == s].std(ddof=1)) for s in range(4)}}
    S = axis["selection_mean_s_by_stage"]["Ery"] - axis["selection_mean_s_by_stage"]["HSC"]
    axis["natural_span_S"] = float(S)
    sd = np.where(grp != "selection")[0]
    s_h = s_all[sd][stage[sd] == 0]; s_c = s_all[sd][stage[sd] == 1]; s_e = s_all[sd][stage[sd] == 2]
    rho_v = spearmanr(stage[sd], s_all[sd]).statistic
    rng = np.random.default_rng(BOOT_SEED)
    bs = []
    for _ in range(2000):
        j = rng.integers(0, len(sd), len(sd))
        bs.append(spearmanr(stage[sd][j], s_all[sd][j]).statistic)
    axis["steering_donors"] = {
        "mean_s": {"HSC": float(s_h.mean()), "CMP": float(s_c.mean()), "EryP": float(s_e.mean())},
        "auroc_EryP_vs_HSC": auroc(s_e, s_h), "auroc_CMP_vs_HSC": auroc(s_c, s_h), "auroc_EryP_vs_CMP": auroc(s_e, s_c),
        "spearman_s_stage": float(rho_v), "spearman_ci95_cell_bootstrap_2000": [float(np.percentile(bs, 2.5)),
                                                                               float(np.percentile(bs, 97.5))],
        "n": {"HSC": int(len(s_h)), "CMP": int(len(s_c)), "EryP": int(len(s_e))}}
    # selection cells: s vs stage within donor, and length confound
    T_all = cells["n_tokens"]
    axis["selection_within_donor_spearman_s_stage"] = {
        d: float(spearmanr(stage[sel][donor[sel] == d], s_all[sel][donor[sel] == d]).statistic) for d in SEL_DONORS}
    axis["within_stage_spearman_s_vs_n_tokens"] = {
        STAGE_SHORT[s]: float(spearmanr(T_all[sel][stage[sel] == s], s_all[sel][stage[sel] == s]).statistic)
        for s in range(4)}
    axis["spearman_s_vs_n_tokens_steering_donor_cells"] = float(spearmanr(T_all[sd], s_all[sd]).statistic)
    axis["check_pass"] = bool(axis["steering_donors"]["auroc_EryP_vs_HSC"] >= AXIS_MIN_AUROC and rho_v > 0)
    log(json.dumps(axis, indent=1))
    # ---- SAE fit on these cells
    fit = {}
    for l in LAYERS:
        out = {}
        for nm in ("all", "gene"):
            for gname, idx in (("selection", sel), ("steered", np.where(grp == "steer")[0])):
                sse = sum(C[i][f"sse_{nm}_L{l}"] for i in idx)
                N = sum(C[i][f"n_{nm}_L{l}"] for i in idx)
                sx = sum(C[i][f"sumx_{nm}_L{l}"] for i in idx)
                sq = sum(C[i][f"sumsq_{nm}_L{l}"] for i in idx)
                sst = sq - float(sx @ sx) / N
                out[f"fve_{nm}_{gname}"] = float(1 - sse / sst)
        fit[f"L{l}"] = out
    log(f"SAE FVE on these cells: {json.dumps(fit)}")
    # ---- features
    sel_hsc = sel[stage[sel] == 0]
    steer_idx = np.where(grp == "steer")[0]
    out_layers = {}
    pc = {}
    for l in LAYERS:
        A = np.stack([C[i][f"meanz_L{l}"] for i in sel]).astype(np.float64)          # (n_sel, F)
        act_hsc = np.stack([C[i][f"nact_L{l}"] for i in sel_hsc]) > 0                # (n_hsc, F)
        frac_hsc = act_hsc.mean(0)
        alive = frac_hsc >= ALIVE_MIN_FRAC
        rho_d = np.stack([spearman_cols(A[donor[sel] == d], stage[sel][donor[sel] == d]) for d in SEL_DONORS])
        rho = rho_d.mean(0)
        rho_pooled = spearman_cols(A, stage[sel])
        mean_a_hsc = np.stack([C[i][f"meanz_L{l}"] for i in sel_hsc]).astype(np.float64).mean(0)
        alive_ids = np.where(alive)[0]
        order = alive_ids[np.lexsort((alive_ids, -np.abs(rho[alive_ids])))]
        cand = [int(f) for f in order[:N_CAND]]
        # steered cells: activity (reported, not used)
        act_st = np.stack([C[i][f"nact_L{l}"] for i in steer_idx]) > 0
        a_st = np.stack([C[i][f"meanz_L{l}"] for i in steer_idx]).astype(np.float64)
        pool = np.array([f for f in alive_ids if f not in cand])
        rng = np.random.default_rng(NULL_SEED + l)
        chosen, nulls = set(), []
        for c, k in zip(cand, NULL_SPLIT):
            cpool = np.array([f for f in pool if f not in chosen])
            dist = np.abs(np.log10(mean_a_hsc[cpool]) - np.log10(mean_a_hsc[c]))
            near = cpool[np.argsort(dist, kind="stable")[:NULL_K_NEAREST]]
            for f in rng.choice(near, size=k, replace=False):
                chosen.add(int(f))
                nulls.append({"feature": int(f), "matched_to": c})

        def frow(f):
            return {"feature": int(f), "rho": float(rho[f]), "rho_by_donor": [float(x) for x in rho_d[:, f]],
                    "rho_pooled": float(rho_pooled[f]), "frac_active_sel_hsc": float(frac_hsc[f]),
                    "mean_a_sel_hsc": float(mean_a_hsc[f]),
                    "mean_a_by_stage_sel": [float(A[stage[sel] == s, f].mean()) for s in range(4)],
                    "frac_active_steered": float(act_st[:, f].mean()), "mean_a_steered": float(a_st[:, f].mean())}
        cands = [frow(f) for f in cand]
        for c in cands:
            c["predicted_sign_amplify"] = int(np.sign(c["rho"]))
        nl = []
        for nrec in nulls:
            r = frow(nrec["feature"])
            r["matched_to"] = nrec["matched_to"]
            r["size_ratio_to_matched"] = float(mean_a_hsc[nrec["feature"]] / mean_a_hsc[nrec["matched_to"]])
            nl.append(r)
        # positive control vectors
        Rm = {s: np.stack([C[i][f"meanres_L{l}"] for i in sel[stage[sel] == s]]).astype(np.float64).mean(0)
              for s in (0, 3)}
        v = Rm[3] - Rm[0]
        target = 4.0 * float(np.mean([mean_a_hsc[f] for f in cand]))
        pc[f"v_full_L{l}"] = v.astype(np.float32)
        pc[f"v_matched_L{l}"] = (v / np.linalg.norm(v) * target).astype(np.float32)
        out_layers[str(l)] = {
            "n_alive": int(alive.sum()), "n_features": D_SAE,
            "candidates": cands, "nulls": nl,
            "rank_of_candidates_by_abs_rho_among_alive": [int(np.where(order == f)[0][0]) for f in cand],
            "abs_rho_quantiles_alive": [float(np.quantile(np.abs(rho[alive_ids]), q)) for q in (0.5, 0.9, 0.99)],
            "max_abs_rho_all_features": float(np.max(np.abs(rho))),
            "positive_control": {"norm_v_full": float(np.linalg.norm(v)), "norm_v_matched": target,
                                 "cos_v_with_mean_hsc_residual": cosd(v, Rm[0])},
        }
        log(f"L{l}: alive {int(alive.sum())}; candidates "
            + ", ".join(f"F{c['feature']} rho {c['rho']:+.3f} a {c['mean_a_sel_hsc']:.3f} steered-active {c['frac_active_steered']:.2f}"
                        for c in cands)
            + f"; null a range {min(x['mean_a_sel_hsc'] for x in nl):.3f}-{max(x['mean_a_sel_hsc'] for x in nl):.3f}"
            + f"; |v| {np.linalg.norm(v):.2f} matched {target:.3f}")
    np.savez(OUT / "signatures.npz", **sig, s_clean_all=s_all, **pc)
    selj = {"axis": axis, "sae_fit_on_these_cells": fit, "layers": out_layers,
            "check_axis_pass": axis["check_pass"]}
    H.write_json(OUT / "selection.json", selj)
    update_run_config("features", {str(l): {"candidates": [c["feature"] for c in out_layers[str(l)]["candidates"]],
                                            "nulls": [x["feature"] for x in out_layers[str(l)]["nulls"]]}
                                   for l in LAYERS})
    if not axis["check_pass"]:
        log("AXIS CHECK FAILED: steering will not run")
        return False
    return True


# ============================================================================= steer
def unit_rows(selj, l):
    L = selj["layers"][str(l)]
    rows = []
    for c in L["candidates"]:
        for a in ALPHAS:
            rows.append(("cand", c["feature"], a))
    for nn in L["nulls"]:
        for a in ALPHAS:
            rows.append(("null", nn["feature"], a))
    rows.append(("pc_full", -1, 1.0))
    rows.append(("pc_matched", -1, 1.0))
    return rows


def metrics(m_st, m_ref, G):
    """Per-row numbers from steered and clean mean logits (float64)."""
    gE, gL = G["g_early"], G["g_late"]
    dv = m_st - m_ref
    gd = gL - gE
    return {"delta_s": score(m_st, gL, gE) - score(m_ref, gL, gE),
            "gap_share": float(dv @ gd / (gd @ gd)),
            "mean_dlogit": float(dv.mean()), "meanabs_dlogit": float(np.abs(dv).mean()),
            "maxabs_dlogit": float(np.abs(dv).max())}


def stage_steer(max_min):
    t_start = time.time()
    selj = load_json(OUT / "selection.json")
    if selj is None or not selj["check_axis_pass"]:
        raise SystemExit("selection missing or axis check failed")
    cells = load_cells()
    G = {k: v.astype(np.float64) for k, v in np.load(OUT / "signatures.npz").items()}
    Gmat = np.stack([G[k] for k in SIG_NAMES])           # (K, V)
    steer_idx = [int(i) for i in np.where(cells["group"] == "steer")[0]]
    STEER_DIR.mkdir(parents=True, exist_ok=True)
    todo = [i for i in steer_idx if not (STEER_DIR / f"cell_{i:03d}.npz").exists()]
    chunk = {"stage": "steer", "start": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "script_sha256": H.sha256_file(SCRIPT_PATH), "free_gb_at_start": round(H.available_memory_gb(), 2),
             "cells_done": [], "encoding_checks_passed": 0, "wall_s_per_cell": []}
    if not todo:
        log("steer: DONE (all cells)")
        return True
    M = Model()
    est = load_json(OUT / "steer_est.json", {"s": 150.0})["s"]
    try:
        for i in todo:
            if (time.time() - t_start) + est * 1.1 > max_min * 60:
                break
            t0 = time.time()
            chunk["min_free_gb"] = min(chunk.get("min_free_gb", 99), H.check_memory(MIN_FREE_GB))
            M.check_cell(cells, i, f"steer cell {i}")
            chunk["encoding_checks_passed"] += 1
            tok = cells["token_ids"][i]
            T = len(tok)
            gmask = gene_mask(T)
            ids = torch.from_numpy(tok[None, :]).to(DEVICE)
            m_full, caps, kw = M.clean_full(ids)
            rec = {"m_full": m_full.astype(np.float32), "T": T}
            for l in LAYERS:
                h_in = caps[l].to(DEVICE)
                m_ref, _ = M.partial(l, h_in, kw, [])
                z = M.saes[l].encode(h_in[0])
                zg = z[torch.from_numpy(gmask).to(DEVICE)]
                rows = unit_rows(selj, l)
                nr = len(rows)
                R = {k: np.zeros(nr) for k in ("delta_s", "gap_share", "mean_dlogit", "meanabs_dlogit",
                                               "maxabs_dlogit", "edit_norm_gene", "n_active_gene", "sum_z_gene")}
                dots = np.zeros((nr, len(SIG_NAMES)))
                sq = np.zeros(nr)
                dvec16 = np.zeros((nr, VOCAB), dtype=np.float16)
                keep32 = {}
                for r, (kind, f, a) in enumerate(rows):
                    if kind in ("cand", "null"):
                        edits = [H.Steer(l, feature=int(f), alpha=float(a), positions=gmask)]
                        R["n_active_gene"][r] = float((zg[:, int(f)] > 0).sum().item())
                        R["sum_z_gene"][r] = float(zg[:, int(f)].sum().item())
                    else:
                        v = torch.from_numpy(G[("v_full_L" if kind == "pc_full" else "v_matched_L") + str(l)]
                                             .astype(np.float32)).to(DEVICE)
                        edits = [H.AddVector(l, vector=v, positions=gmask)]
                    m_st, elog = M.partial(l, h_in, kw, edits)
                    for k, val in metrics(m_st, m_ref, G).items():
                        R[k][r] = val
                    R["edit_norm_gene"][r] = elog.get("delta_norm", 0.0) * T / gmask.sum()
                    dots[r] = Gmat @ m_st
                    sq[r] = m_st @ m_st
                    dvec16[r] = (m_st - m_ref).astype(np.float16)
                    if kind in ("cand", "pc_full", "pc_matched"):
                        keep32[r] = m_st.astype(np.float32)
                rec[f"L{l}_m_ref"] = m_ref.astype(np.float32)
                rec[f"L{l}_ref_vs_full_maxabs"] = float(np.abs(m_ref - m_full).max())
                rec[f"L{l}_ref_vs_full_ds"] = float(score(m_ref, G["g_late"], G["g_early"])
                                                    - score(m_full, G["g_late"], G["g_early"]))
                rec[f"L{l}_ref_dots"] = Gmat @ m_ref
                rec[f"L{l}_ref_sq"] = float(m_ref @ m_ref)
                for k, val in R.items():
                    rec[f"L{l}_{k}"] = val
                rec[f"L{l}_dots"] = dots
                rec[f"L{l}_sq"] = sq
                rec[f"L{l}_dvec16"] = dvec16
                rec[f"L{l}_keep32_rows"] = np.array(sorted(keep32), dtype=np.int64)
                rec[f"L{l}_keep32"] = np.stack([keep32[r] for r in sorted(keep32)])
                del h_in, z, zg
            tmp = STEER_DIR / f"cell_{i:03d}.tmp.npz"
            np.savez(tmp, **rec)
            tmp.replace(STEER_DIR / f"cell_{i:03d}.npz")
            del caps, kw, ids
            H.free_device_cache()
            dt = time.time() - t0
            est = dt
            H.write_json(OUT / "steer_est.json", {"s": est})
            chunk["cells_done"].append(i)
            chunk["wall_s_per_cell"].append(round(dt, 1))
            ds5 = {l: float(np.mean([rec[f'L{l}_delta_s'][r] for r, (k, f, a) in enumerate(unit_rows(selj, l))
                                     if k == 'cand' and a == 5.0])) for l in LAYERS}
            log(f"steer cell {i} T={T} {dt:.0f}s  cand ds(a5) " + " ".join(f"L{l} {v:+.5f}" for l, v in ds5.items())
                + "  ref-vs-full max " + " ".join(f"{rec[f'L{l}_ref_vs_full_maxabs']:.1e}" for l in LAYERS))
    except H.MemoryGuardError as e:
        chunk["stopped"] = str(e)
        log(f"memory guard: {e}")
    finally:
        chunk["wall_seconds"] = round(time.time() - t_start, 1)
        chunk["end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        update_run_config(chunk=chunk)
    left = [i for i in steer_idx if not (STEER_DIR / f"cell_{i:03d}.npz").exists()]
    log("steer: DONE" if not left else f"steer: {len(steer_idx) - len(left)} of {len(steer_idx)} cells done; run again")
    return not left


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["design", "prep", "clean", "select", "steer"])
    ap.add_argument("--max-minutes", type=float, default=7.5)
    args = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "design":
        stage_design()
    elif args.stage == "prep":
        stage_prep()
    elif args.stage == "clean":
        stage_clean(args.max_minutes)
    elif args.stage == "select":
        stage_select()
    elif args.stage == "steer":
        stage_steer(args.max_minutes)


if __name__ == "__main__":
    main()
