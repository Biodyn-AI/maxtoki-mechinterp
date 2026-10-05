"""v3 spectral geometry (item V3-5): per-layer effective rank and cross-sample CKA, redone with
correctly encoded inputs, with a random-initialised MaxToki-217M control.

Why a v3
--------
The deployed spectral run (scripts/phase0_extract.py -> phase1_svd.py, phase8_stability.py)
fed MaxToki genes ranked by log1p(CP10k) / median instead of counts / median
(checks/INPUT_ENCODING_AUDIT.md). Its headline numbers were
  * effective rank 561 (L0) -> 94 (L11), feature-shuffle null 573 at L11;
  * linear CKA across three 2,000-cell samples 1.000 (L0) -> 0.991 (L11).
Two further problems (REVISION_PLAN E36/E37): the three samples were not disjoint (6-8 shared
cells per pair) and a feature-shuffle null cannot tell learned from architectural compression.

What this script does
---------------------
  * Same dataset (Tabula Sapiens immune), same 1,500-gene panel (phase0/gene_features.csv),
    same context (2,048 tokens), same per-gene averaging, same metric code (copied verbatim
    from phase1_svd.py and phase8_stability.py; the verify stage checks the copies).
  * Inputs from setup/inputs_v3.py: raw/X integer counts -> counts / gene median order. Every
    cell is re-tokenised and passes inputs_v3.check_encoding before its forward pass
    (assert_encoding_batch on every 50-cell batch); the tokens must also equal the tokens
    saved by `prepare`.
  * Cells: sample A = the deployed Phase 0 cells (default_rng(42).choice, 2,000).
    Samples B and C = the deployed Phase 8 draws (seeds 43, 44) with every cell that is already
    in an earlier sample replaced by a fresh draw (default_rng(20261002 / 20261003)) from rows
    used by none of the deployed samples. So A, B, C are truly disjoint.
  * Models: the trained checkpoint, and three random-initialised MaxToki-217M models
    (same config, transformers default init: normal(0, 0.02) weights, RMSNorm weights 1),
    torch.manual_seed(0 / 1 / 2) before construction, run on sample A. The CKA control for the
    random-init models is a split-half CKA (sample A blocks 0-4 vs 5-9), computed the same way
    for the trained model (analyze --part cka_half).
  * Each sample is split into 10 random blocks of 200 cells (for a cell-level bootstrap), and
    Smart-seq2 cells are also summed separately (UMI-only sensitivity check).
  * A second gene panel ("panel S") is chosen with the deployed rule but on log1p(CP10k) of
    the raw counts (the deployed panel was chosen on a double log). Sensitivity check only.

Stages
------
  prepare : CPU. Cells, panels, tokens, order agreement with the deployed tokens.
  extract : MPS. Resumable. Run repeatedly with --max-minutes until it prints ALL_DONE True.
  analyze : CPU. --part metrics | cka | erboot | summary.
  verify  : CPU (+ a few MPS/CPU forward passes). Second-way checks.

Usage (from projects/maxtoki):
  OMP_NUM_THREADS=4 .venv/bin/python runs/spectral-geometry-217M/scripts/v3_spectral.py <stage> [...]

Layer convention: hidden_states[0] = token embedding; hidden_states[l] (1..10) = output of
block l-1; hidden_states[11] = final RMSNorm(output of block 10) (setup/hooks_v2.py docstring).
No edits are made here, so output_hidden_states is safe; verify checks it against
hooks_v2.ResidualEditor captures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
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

RUN = PROJ / "runs/spectral-geometry-217M"
OUT = RUN / "outputs/v3_spectral"
PREP = OUT / "prepare"
EXT = OUT / "extract"
ANA = OUT / "analysis"
VER = OUT / "verify"
DEP0 = RUN / "outputs/phase0"
DEP1 = RUN / "outputs/phase1"
DEP8 = RUN / "outputs/phase8"
SCRIPT_PATH = Path(__file__).resolve()
MODEL_DIR = PROJ / "setup/MaxToki-217M-HF"

BIOM_ROOT = Path("<DATA_ROOT>")
SYM2ENS = BIOM_ROOT / "biodyn-nmi-paper/src/02_cssi_method/crispri_validation/data/gene_name_id_dict_gc104M.pkl"
TRRUST_TSV = BIOM_ROOT / "biodyn-nmi-paper/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv"
STRING_JSON = BIOM_ROOT / "biodyn-nmi-paper/results/biological_impact/reference_edge_sets/string_ppi_edges.json"

DATASET = "ts_immune"
MAX_LEN = 2048
N_CELLS = 2000
N_HVG = 1500
N_BLOCKS = 10
SS2_SLOT = N_BLOCKS            # accumulator index for the Smart-seq2-only sums
N_SLOTS = N_BLOCKS + 1
N_STATES = 12                  # hidden_states entries (embedding + 11 blocks)
HIDDEN = 1232
BATCH = 50                     # cells per encoding-check batch
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MIN_FREE_GB = 3.0

SAMPLES = ("A", "B", "C")
DEPLOYED_SAMPLE_SEEDS = {"A": 42, "B": 43, "C": 44}
REFILL_SEEDS = {"B": 20261002, "C": 20261003}
BLOCK_SEED = 20261005          # block id = default_rng(BLOCK_SEED + k).permutation(n) % 10
RANDOM_INIT_SEEDS = (0, 1, 2)
PHASE1_SEED = 42               # deployed phase1_svd.py SEED
BOOT_SEED = 20261006
N_BOOT_GENE = 500
N_BOOT_CELL = 200

# priority order: core jobs first. The MPS forward pass runs at ~1.1 s/cell on this machine now (deployed:
# 0.39 s/cell), so the random-init model is run on sample A only; its CKA control is a split-half CKA
# (blocks 0-4 vs 5-9 of sample A, 1,000 vs 1,000 cells), computed the same way for the trained model.
# rand0_B / rand0_C stay available as OPTIONAL_JOBS (--jobs rand0_B,rand0_C) but are not run by default.
JOBS = ["trained_A", "rand0_A", "rand1_A", "rand2_A", "trained_B", "trained_C"]
OPTIONAL_JOBS = ["rand0_B", "rand0_C"]

MARKERS = [
    "CD19", "CD79A", "MS4A1", "BLK", "VPREB3", "FCRL1", "PAX5", "BATF", "BACH2", "BCL6",
    "PRDM1", "IRF4", "IRF8", "CD3D", "CD3E", "CD3G", "CD4", "CD8A", "CD8B", "LCK", "ZAP70",
    "TCF7", "LEF1", "TBX21", "GATA3", "NCAM1", "KLRD1", "KLRF1", "NKG7", "GNLY", "CD68",
    "CD163", "MRC1", "MSR1", "MERTK", "CD14", "LYZ", "VCAN", "S100A8", "S100A9",
    "CLEC9A", "CLEC10A", "CD1C", "FCER1A", "ELANE", "MPO", "AZU1", "DEFA3", "CSF3R",
    "NAMPT", "GLUL", "PFKFB3", "HK1", "HK2", "PKM", "LDHA", "MTOR", "STAT3",
]  # copied from phase0_extract.py


# =============================================================================
# Metric code copied VERBATIM from the deployed scripts (verify checks the copies)
# =============================================================================
# ---- from scripts/phase1_svd.py ----
def _two_nn(X, seed=0):
    """Facco 2017 TwoNN intrinsic dimensionality."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n < 4:
        return float("nan")
    from scipy.spatial.distance import cdist
    idx = rng.choice(n, size=min(n, 500), replace=False)
    D = cdist(X[idx], X)
    D.sort(axis=1)
    r1 = D[:, 1]
    r2 = D[:, 2]
    ok = (r1 > 1e-9) & (r2 > r1)
    mu = r2[ok] / r1[ok]
    mu.sort()
    # CDF fit
    F = (np.arange(1, len(mu) + 1)) / len(mu)
    mask = F < 0.9
    if mask.sum() < 10:
        return float("nan")
    x = np.log(mu[mask])
    y = -np.log(1.0 - F[mask] + 1e-12)
    d, _ = np.polyfit(x, y, 1)
    return float(d)


def _effective_rank(sigmas):
    p = sigmas ** 2
    p = p / max(p.sum(), 1e-12)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def _participation_ratio(sigmas):
    s2 = sigmas ** 2
    return float(s2.sum() ** 2 / max((s2 ** 2).sum(), 1e-12))


def _svd_and_metrics(M, seed):
    """M: (n_genes, hidden). Mean-centre columns, SVD, compute dimensionality metrics."""
    mu = M.mean(axis=0, keepdims=True)
    Mc = M - mu
    U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    out = {
        "n_genes": int(M.shape[0]),
        "effective_rank": _effective_rank(S),
        "participation_ratio": _participation_ratio(S),
        "sv1_variance_fraction": float(S[0] ** 2 / (S ** 2).sum()),
        "svd_sigmas": S.tolist(),
    }
    out["twonn_intrinsic_dim"] = _two_nn(Mc, seed=seed)
    return out, U, S, Vt


# ---- from scripts/phase8_stability.py ----
def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two (n, d) matrices. Returns scalar in [0, 1]."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    num = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    den = (np.linalg.norm(X.T @ X, ord="fro") * np.linalg.norm(Y.T @ Y, ord="fro"))
    return float(num / den) if den > 0 else 0.0


def feature_shuffle_er(M: np.ndarray, rng_seed: int, svd_seed: int) -> float:
    """phase1_svd.py feature-shuffle null, same steps: shuffle every column with one
    default_rng(rng_seed) stream (column order), then _svd_and_metrics(seed=svd_seed)."""
    rng = np.random.default_rng(rng_seed)
    M_shuf = M.copy()
    for j in range(M.shape[1]):
        rng.shuffle(M_shuf[:, j])
    m, _, _, _ = _svd_and_metrics(M_shuf, seed=svd_seed)
    return m["effective_rank"]


# =============================================================================
# Helpers
# =============================================================================
def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def sha256_file(p) -> str:
    return I.sha256_file(p)


def write_json(path, obj):
    H.write_json(path, obj)


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path) as fh:
        return json.load(fh)


def atomic_save_npy(path: Path, arr: np.ndarray):
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def update_run_config(folder: Path, section: str, entry, append: bool = False):
    p = folder / "run_config.json"
    rc = read_json(p, {}) or {}
    if append:
        rc.setdefault(section, []).append(entry)
    else:
        rc[section] = entry
    write_json(p, rc)


def code_record() -> dict:
    return {
        "script": str(SCRIPT_PATH), "script_sha256": sha256_file(SCRIPT_PATH),
        "inputs_v3_sha256": sha256_file(PROJ / "setup/inputs_v3.py"),
        "hooks_v2_sha256": sha256_file(PROJ / "setup/hooks_v2.py"),
        "maxtoki_adapter_sha256": sha256_file(PROJ / "setup/maxtoki_adapter.py"),
        "deployed_phase0_extract_sha256": sha256_file(RUN / "scripts/phase0_extract.py"),
        "deployed_phase1_svd_sha256": sha256_file(RUN / "scripts/phase1_svd.py"),
        "deployed_phase8_stability_sha256": sha256_file(RUN / "scripts/phase8_stability.py"),
    }


def peak_rss_gb() -> float:
    # macOS: ru_maxrss in bytes
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def _dec(a):
    return [s.decode() if isinstance(s, bytes) else s for s in a]


def read_var_column(column: str) -> np.ndarray:
    import h5py
    with h5py.File(I.DATASETS[DATASET]["path"], "r") as f:
        node = f["var"][column]
        if isinstance(node, h5py.Group) and "categories" in node:
            cats = np.array(_dec(node["categories"][:]), dtype=object)
            codes = node["codes"][:]
            return np.where(codes >= 0, cats[np.maximum(codes, 0)], None)
        vals = node[:]
        return np.array(_dec(vals), dtype=object) if vals.dtype.kind in "SO" else vals


def read_csr_rows(node_path: str, rows):
    """Rows of a CSR group in the h5ad as a scipy CSR matrix (values as stored, float32)."""
    import h5py
    import scipy.sparse as sp
    with h5py.File(I.DATASETS[DATASET]["path"], "r") as f:
        node = f[node_path]
        ip, ind, dat = node["indptr"], node["indices"], node["data"]
        n_cols = int(node.attrs["shape"][1])
        indptr, inds, vals = [0], [], []
        for r in rows:
            a, b = int(ip[int(r)]), int(ip[int(r) + 1])
            inds.append(np.asarray(ind[a:b]))
            vals.append(np.asarray(dat[a:b], dtype=np.float32))
            indptr.append(indptr[-1] + (b - a))
    return sp.csr_matrix((np.concatenate(vals), np.concatenate(inds), np.array(indptr)),
                         shape=(len(rows), n_cols))


def n_total_cells() -> int:
    import h5py
    with h5py.File(I.DATASETS[DATASET]["path"], "r") as f:
        X = f["X"]
        return int(X.attrs["shape"][0]) if isinstance(X, h5py.Group) else int(X.shape[0])


# =============================================================================
# Panel selection (the deployed phase0_extract.py rule, re-implemented line by line)
# =============================================================================
def priority_symbols() -> set:
    import pandas as pd
    trrust = pd.read_csv(TRRUST_TSV, sep="\t", header=None, names=["tf", "target", "mode", "pmid"])
    ps = set(trrust["tf"].str.upper()) | set(trrust["target"].str.upper())
    with open(STRING_JSON) as fh:
        string_data = json.load(fh)
    for pair_list in string_data.values():
        for a, b in pair_list:
            ps.add(a.upper())
            ps.add(b.upper())
    ps |= set(s.upper() for s in MARKERS)
    return ps


def select_panel(X_dense32: np.ndarray, var_ens_clean, var_feature_name, prio: set, n_hvg: int = N_HVG):
    """phase0_extract.py lines 'Normalize' .. 'hvg_indices_np', unchanged except that X is given.
    X_dense32: (n_cells, n_vars) float32, the values that phase0 normalises (rs, *1e4, log1p).
    Works in place on X_dense32 (same float32 arithmetic as the deployed expression)."""
    tok = I.get_tokenizer()
    X = X_dense32
    rs = X.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    X /= rs
    X *= 1e4
    np.log1p(X, out=X)            # X is now X_log
    eligible = np.array([e in tok.gene_token_dict for e in var_ens_clean], dtype=bool)
    priority_mask = np.array([(str(var_feature_name[i]).upper() in prio) for i in range(len(var_ens_clean))],
                             dtype=bool)
    gene_var_all = X.var(axis=0)
    gene_nz_pct = (X > 0).mean(axis=0)
    pool_mask = priority_mask & eligible & (gene_var_all > 0) & (gene_nz_pct > 0.005)
    pool_idx = np.where(pool_mask)[0]
    pool_sorted = pool_idx[np.argsort(-gene_var_all[pool_idx])]
    priority_taken = pool_sorted[: min(n_hvg, len(pool_sorted))]
    remaining_need = n_hvg - len(priority_taken)
    if remaining_need > 0:
        rest_mask = ~pool_mask & eligible & (gene_var_all > 0)
        rest_idx = np.where(rest_mask)[0]
        rest_sorted = rest_idx[np.argsort(-gene_var_all[rest_idx])][:remaining_need]
        hvg = np.sort(np.concatenate([priority_taken, rest_sorted])).astype(np.int64)
    else:
        hvg = np.sort(priority_taken).astype(np.int64)
    info = {"n_priority_pool": int(len(pool_idx)), "n_priority_taken": int(len(priority_taken)),
            "n_eligible": int(eligible.sum()), "n_priority_mask": int(priority_mask.sum())}
    return hvg, info


# =============================================================================
# Stage: prepare
# =============================================================================
def stage_prepare(args):
    import pandas as pd
    t0 = time.time()
    PREP.mkdir(parents=True, exist_ok=True)
    free_gb = H.check_memory(MIN_FREE_GB)
    N = n_total_cells()
    print(f"[prepare] {DATASET}: {N} cells; free memory {free_gb:.1f} GB")

    # ---- cells ----
    dep = {s: np.sort(np.random.default_rng(DEPLOYED_SAMPLE_SEEDS[s]).choice(N, size=N_CELLS, replace=False))
           for s in SAMPLES}
    used_deployed = set(np.concatenate(list(dep.values())).tolist())
    overlaps_dep = {f"{a}{b}": int(len(set(dep[a].tolist()) & set(dep[b].tolist())))
                    for a, b in (("A", "B"), ("A", "C"), ("B", "C"))}
    samples = {"A": dep["A"]}
    taken = set(dep["A"].tolist())
    refill_log = {}
    for s in ("B", "C"):
        keep = [int(r) for r in dep[s] if int(r) not in taken]
        n_need = N_CELLS - len(keep)
        rng = np.random.default_rng(REFILL_SEEDS[s])
        new = []
        while len(new) < n_need:
            r = int(rng.integers(0, N))
            if r in used_deployed or r in taken or r in new:
                continue
            new.append(r)
        samples[s] = np.sort(np.array(keep + new, dtype=np.int64))
        taken |= set(samples[s].tolist())
        refill_log[s] = {"deployed_seed": DEPLOYED_SAMPLE_SEEDS[s], "kept_from_deployed": len(keep),
                         "replaced": n_need, "refill_seed": REFILL_SEEDS[s], "new_rows": sorted(new)}
    for a, b in (("A", "B"), ("A", "C"), ("B", "C")):
        assert not (set(samples[a].tolist()) & set(samples[b].tolist())), "samples not disjoint"
    assert all(len(samples[s]) == N_CELLS for s in SAMPLES)
    print(f"[prepare] deployed overlaps {overlaps_dep}; v3 samples disjoint; refills "
          f"{ {s: refill_log[s]['replaced'] for s in refill_log} }")

    # sample A must be the deployed Phase 0 cells
    meta = pd.read_csv(DEP0 / "cell_metadata.csv")
    ct_A = I.read_obs_column(DATASET, "cell_type", samples["A"])
    ct_match = float(np.mean(np.array(ct_A, dtype=str) == meta["cell_type"].astype(str).to_numpy()))
    assert ct_match == 1.0, f"sample A cell types differ from deployed cell_metadata.csv ({ct_match})"

    # ---- panels ----
    var_idx_str = None
    import h5py
    with h5py.File(I.DATASETS[DATASET]["path"], "r") as f:
        k = f["var"].attrs.get("_index", "_index")
        k = k.decode() if isinstance(k, bytes) else k
        var_idx_str = _dec(f["var"][k][:])
    var_ens_clean = np.array([e.split(".")[0] for e in var_idx_str])
    var_feature_name = read_var_column("feature_name")
    prio = priority_symbols()
    gf = pd.read_csv(DEP0 / "gene_features.csv")
    panel_D = gf["var_idx"].astype(int).to_numpy()
    assert np.all(np.diff(panel_D) > 0) and len(panel_D) == N_HVG
    assert all(var_ens_clean[panel_D] == gf["ensembl_id"].astype(str).to_numpy())

    # reproduce the deployed selection (stored X, double log) as a check of the rule
    H.check_memory(MIN_FREE_GB)
    Xs = read_csr_rows("X", samples["A"]).toarray().astype(np.float32)
    rep_D, info_D = select_panel(Xs, var_ens_clean, var_feature_name, prio)
    del Xs
    rep_match = int(len(set(rep_D.tolist()) & set(panel_D.tolist())))
    print(f"[prepare] deployed panel reproduced from stored X: {rep_match}/{N_HVG} genes")
    # panel S: same rule on log1p(CP10k) of raw counts (single log)
    H.check_memory(MIN_FREE_GB)
    Xr = read_csr_rows("raw/X", samples["A"]).toarray().astype(np.float32)
    panel_S, info_S = select_panel(Xr, var_ens_clean, var_feature_name, prio)
    del Xr
    overlap_DS = int(len(set(panel_S.tolist()) & set(panel_D.tolist())))
    print(f"[prepare] panel S (raw counts, single log): overlap with deployed panel {overlap_DS}/{N_HVG}")
    panel_U = np.array(sorted(set(panel_D.tolist()) | set(panel_S.tolist())), dtype=np.int64)
    u_of_var = {int(v): i for i, v in enumerate(panel_U)}
    D_in_U = np.array([u_of_var[int(v)] for v in panel_D], dtype=np.int64)
    S_in_U = np.array([u_of_var[int(v)] for v in panel_S], dtype=np.int64)
    tok = I.get_tokenizer()
    panel = {
        "panel_D_var_idx": panel_D.tolist(), "panel_S_var_idx": panel_S.tolist(),
        "panel_U_var_idx": panel_U.tolist(), "D_in_U": D_in_U.tolist(), "S_in_U": S_in_U.tolist(),
        "U_ensembl": var_ens_clean[panel_U].tolist(),
        "U_symbol": [str(var_feature_name[i]) for i in panel_U],
        "U_token_id": [int(tok.gene_token_dict[e]) for e in var_ens_clean[panel_U]],
        "panel_D_source": str(DEP0 / "gene_features.csv"),
        "panel_D_reproduced_from_stored_X": rep_match, "panel_D_rule_info": info_D,
        "panel_S_rule": "phase0_extract.py rule, values = log1p(raw/X / row sum * 1e4) of sample A",
        "panel_S_rule_info": info_S, "overlap_D_S": overlap_DS,
    }
    write_json(PREP / "panel.json", panel)

    # ---- tokens ----
    vm = I.get_var_map(DATASET)
    var_to_U = -np.ones(vm.n_vars, dtype=np.int64)
    var_to_U[panel_U] = np.arange(len(panel_U))
    D_mask_U = np.zeros(len(panel_U), dtype=bool)
    D_mask_U[D_in_U] = True
    deployed_counts = np.load(DEP0 / "gene_counts.npy")
    rows_csv = []
    enc_summary = {}
    for k, s in enumerate(SAMPLES):
        H.check_memory(MIN_FREE_GB)
        rows = samples[s]
        assay = I.read_obs_column(DATASET, "assay", rows)
        ctype = I.read_obs_column(DATASET, "cell_type", rows)
        block = np.random.default_rng(BLOCK_SEED + k).permutation(len(rows)) % N_BLOCKS
        dep_cells = I.deployed_tokenize_X(DATASET, rows, MAX_LEN)
        toks, offs, kept_mask, n_checked = [], [0], [], 0
        hsh = hashlib.sha256()
        gene_obs_v3 = np.zeros(len(panel_U), dtype=np.int64)
        gene_obs_dep = np.zeros(len(panel_U), dtype=np.int64)
        with I.CountReader(DATASET) as r:
            for j, row in enumerate(rows):
                c, chk = r.counts(int(row), return_check=True)
                nz = np.nonzero(c)[0]
                hsh.update(np.int64(row).tobytes()); hsh.update(nz.astype(np.int64).tobytes())
                hsh.update(c[nz].astype(np.float64).tobytes())
                cell = I.tokenize_counts(c, vm, MAX_LEN, check=True)   # raises EncodingError on failure
                n_checked += 1
                dc = dep_cells[j]
                rec = {"sample": s, "pos": j, "row": int(row), "assay": assay[j], "cell_type": ctype[j],
                       "block": int(block[j]), "is_ss2": bool(assay[j] == "Smart-seq2")}
                if cell is None:
                    kept_mask.append(False)
                    rec.update({"tokenized": False})
                    rows_csv.append(rec)
                    offs.append(offs[-1])
                    continue
                kept_mask.append(True)
                t = cell.token_ids.astype(np.int32)
                toks.append(t)
                offs.append(offs[-1] + len(t))
                gp = cell.gene_positions[1:-1]
                u = var_to_U[vm.var_indices[gp]]
                u = u[u >= 0]
                gene_obs_v3[u] += 1
                rec.update({"tokenized": True, "n_tokens_v3": int(len(t)),
                            "n_panelD_v3": int(D_mask_U[u].sum()), "n_panelU_v3": int(len(u))})
                if dc is not None:
                    gpd = dc.gene_positions[1:-1]
                    ud = var_to_U[vm.var_indices[gpd]]
                    ud = ud[ud >= 0]
                    gene_obs_dep[ud] += 1
                    rec.update({"n_tokens_deployed": int(len(dc.token_ids)), "n_panelD_deployed": int(D_mask_U[ud].sum()),
                                "tokens_equal_deployed": bool(len(dc.token_ids) == len(t) and np.array_equal(dc.token_ids, t))})
                    if s == "A" or j % 10 == 0:   # order agreement (all of A, every 10th cell of B, C)
                        full_v3 = I.full_order(c, vm)
                        xr = r.stored_X(int(row)).astype(np.float32)
                        full_dep = tok.tokenize_cell(xr, vm.var_indices, vm.token_ids, vm.medians, max_len=10 ** 9)
                        ag = I.order_agreement(full_dep.token_ids[1:-1], full_v3, MAX_LEN)
                        rec.update({k2: ag[k2] for k2 in ("spearman_full", "top200_overlap", "kept_set_overlap",
                                                          "same_position_share")})
                rows_csv.append(rec)
        tokens = np.concatenate(toks).astype(np.int32)
        np.savez(PREP / f"cells_{s}.npz", rows=rows, kept=np.array(kept_mask), tokens=tokens,
                 offsets=np.array(offs, dtype=np.int64), block=block.astype(np.int64),
                 is_ss2=np.array([a == "Smart-seq2" for a in assay]),
                 assay=np.array([str(a) for a in assay]), cell_type=np.array([str(c) for c in ctype]))
        enc_summary[s] = {"n_rows": int(len(rows)), "n_tokenized": int(sum(kept_mask)),
                          "n_check_encoding_pass": int(sum(kept_mask)), "n_count_checks": n_checked,
                          "counts_sha256": hsh.hexdigest(), "tokens_sha256": I.array_sha256(tokens),
                          "n_ss2": int(np.sum([a == "Smart-seq2" for a in assay])),
                          "panelD_genes_observed_v3": int((gene_obs_v3[D_in_U] > 0).sum()),
                          "panelD_genes_observed_deployed": int((gene_obs_dep[D_in_U] > 0).sum()),
                          "panelD_obs_per_gene_v3_median": float(np.median(gene_obs_v3[D_in_U])),
                          "panelD_obs_per_gene_deployed_median": float(np.median(gene_obs_dep[D_in_U]))}
        if s == "A":
            enc_summary[s]["deployed_gene_counts_reproduced"] = bool(np.array_equal(gene_obs_dep[D_in_U], deployed_counts))
            np.save(PREP / "gene_obs_A_v3_U.npy", gene_obs_v3)
            np.save(PREP / "gene_obs_A_deployed_U.npy", gene_obs_dep)
        print(f"[prepare] sample {s}: {enc_summary[s]}")
    pd.DataFrame(rows_csv).to_csv(PREP / "cells.csv", index=False)

    rc = {
        "stage": "prepare", "env": H.env_info("cpu"), "code": code_record(),
        "seeds": {"deployed_sample_seeds": DEPLOYED_SAMPLE_SEEDS, "refill_seeds": REFILL_SEEDS,
                  "block_seed": BLOCK_SEED, "n_blocks": N_BLOCKS},
        "cells": {s: [int(x) for x in samples[s]] for s in SAMPLES},
        "deployed_sample_overlaps": overlaps_dep, "refill": refill_log,
        "sample_A_equals_deployed_phase0": {"cell_type_match": ct_match},
        "feature_ids": {"panel_D_ensembl": var_ens_clean[panel_D].tolist(),
                        "panel_S_ensembl": var_ens_clean[panel_S].tolist()},
        "input_encoding": {s: I.encoding_record(DATASET, rows=None, check_summary=enc_summary[s],
                                                counts_sha256=enc_summary[s]["counts_sha256"]) for s in SAMPLES},
        "wall_seconds": round(time.time() - t0, 1), "peak_rss_gb": round(peak_rss_gb(), 2),
    }
    write_json(PREP / "run_config.json", rc)
    print(f"[prepare] done in {time.time() - t0:.0f}s; peak RSS {peak_rss_gb():.2f} GB")


# =============================================================================
# Models
# =============================================================================
def state_dict_sha256(model) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().to("cpu").contiguous().numpy().tobytes())
    return h.hexdigest()


def build_model(kind: str, device: str):
    """kind = 'trained' or 'rand<seed>'. Returns (model, info)."""
    from transformers import LlamaConfig, LlamaForCausalLM
    if kind == "trained":
        model = H.load_model(device).model
        model.eval()
        info = {"kind": "trained", "model_dir": str(MODEL_DIR),
                "attn_implementation": model.config._attn_implementation}
        return model, info
    seed = int(kind[len("rand"):])
    cfg = LlamaConfig.from_pretrained(str(MODEL_DIR))
    cfg._attn_implementation = "eager"
    torch.manual_seed(seed)
    model = LlamaForCausalLM(cfg)          # transformers default init (normal(0, initializer_range))
    model.eval()
    sd = model.state_dict()
    info = {"kind": "random_init", "init_seed": seed, "init": "transformers default (LlamaForCausalLM(config))",
            "initializer_range": float(cfg.initializer_range),
            "attn_implementation": model.config._attn_implementation,
            "state_dict_sha256_cpu": state_dict_sha256(model),
            "param_std": {k: float(sd[k].std()) for k in ("model.embed_tokens.weight",
                                                          "model.layers.0.self_attn.q_proj.weight",
                                                          "model.layers.10.mlp.down_proj.weight")}}
    model.to(device)
    return model, info


def forward_states(model, token_ids: np.ndarray, positions: np.ndarray, device: str):
    """One cell. Returns (sub (12, k, H) float32 numpy at `positions`, nll, top1)."""
    with torch.no_grad():
        ids = torch.from_numpy(token_ids.astype(np.int64)[None, :]).to(device)
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False, return_dict=True)
        hs = torch.stack(out.hidden_states, 0)[:, 0]          # (12, T, H)
        pos_t = torch.from_numpy(positions.astype(np.int64)).to(device)
        sub = hs.index_select(1, pos_t).float().cpu().numpy()
        logits = out.logits[0, :-2].float()                   # predicts tokens 1..T-2 (genes)
        tgt = ids[0, 1:-1]
        lp = torch.log_softmax(logits, -1)
        nll = float(-lp.gather(1, tgt[:, None]).mean())
        top1 = float((logits.argmax(-1) == tgt).float().mean())
    return sub, nll, top1


# =============================================================================
# Stage: extract
# =============================================================================
def load_prep():
    panel = read_json(PREP / "panel.json")
    cells = {s: dict(np.load(PREP / f"cells_{s}.npz")) for s in SAMPLES}
    return panel, cells


def job_paths(job: str) -> dict:
    d = EXT / job
    return {"dir": d, "acc": d / "acc_sums.npy", "cnt": d / "acc_counts.npy", "percell": d / "percell_sums.npy",
            "state": d / "state.json", "cells": d / "cells.csv"}


def stage_extract(args):
    import pandas as pd
    t_call = time.time()
    budget = float(args.max_minutes) * 60.0
    panel, cells_all = load_prep()
    vm = I.get_var_map(DATASET)
    panel_U = np.array(panel["panel_U_var_idx"], dtype=np.int64)
    G = len(panel_U)
    var_to_U = -np.ones(vm.n_vars, dtype=np.int64)
    var_to_U[panel_U] = np.arange(G)
    jobs = args.jobs.split(",") if args.jobs else JOBS
    model, model_kind, model_info = None, None, None
    stopped_reason = "all_done"
    for job in jobs:
        kind, s = job.rsplit("_", 1)
        P = job_paths(job)
        P["dir"].mkdir(parents=True, exist_ok=True)
        st = read_json(P["state"], None) or {"job": job, "next": 0, "done": False, "chunks": [],
                                              "batches_checked": 0, "cells_checked": 0}
        if st["done"]:
            continue
        if time.time() - t_call > budget - 60:
            stopped_reason = "time"
            break
        C = cells_all[s]
        rows = C["rows"]
        kept = C["kept"].astype(bool)
        offs = C["offsets"]
        n = len(rows)
        free_gb = H.check_memory(MIN_FREE_GB)
        # model (reuse within a call when consecutive jobs share it)
        if model_kind != kind:
            if model is not None:
                del model
                H.free_device_cache()
            model, model_info = build_model(kind, DEVICE)
            model_kind = kind
            if kind != "trained":
                ref = st.get("model_info", {}).get("state_dict_sha256_cpu")
                if ref is not None and ref != model_info["state_dict_sha256_cpu"]:
                    raise RuntimeError(f"{job}: random-init model differs from the one used in earlier chunks")
        st["model_info"] = model_info
        # accumulators
        if P["acc"].exists():
            acc = np.load(P["acc"])
            cnt = np.load(P["cnt"])
            percell = np.load(P["percell"])
            cell_recs = pd.read_csv(P["cells"]).to_dict("records") if P["cells"].exists() else []
        else:
            acc = np.zeros((N_SLOTS, N_STATES, G, HIDDEN), dtype=np.float32)
            cnt = np.zeros((N_SLOTS, G), dtype=np.int64)
            percell = np.zeros((n, N_STATES, HIDDEN), dtype=np.float32)
            cell_recs = []
        start_i = int(st["next"])
        t_chunk = time.time()
        min_free = free_gb
        mps_peak = 0.0
        stop_now = False
        i = start_i
        with I.CountReader(DATASET) as reader:
            while i < n:
                if time.time() - t_call > budget:
                    stopped_reason = "time"
                    stop_now = True
                    break
                try:
                    min_free = min(min_free, H.check_memory(MIN_FREE_GB))
                except H.MemoryGuardError as e:
                    print(f"[extract] {job}: {e}")
                    stopped_reason = "memory"
                    stop_now = True
                    break
                i1 = min(n, i + BATCH)
                # ---- encoding check for the whole batch, before any forward pass ----
                b_counts, b_cells, b_idx = [], [], []
                for j in range(i, i1):
                    if not kept[j]:
                        continue
                    c = reader.counts(int(rows[j]))
                    cell = I.tokenize_counts(c, vm, MAX_LEN, check=True)
                    saved = C["tokens"][offs[j]:offs[j + 1]]
                    if cell is None or not np.array_equal(cell.token_ids.astype(np.int32), saved):
                        raise I.EncodingError(f"{job} row {rows[j]}: tokens differ from prepare/cells_{s}.npz")
                    b_counts.append(c)
                    b_cells.append(cell)
                    b_idx.append(j)
                if b_cells:
                    I.assert_encoding_batch(b_counts, b_cells, vm, MAX_LEN, label=f"{job} cells {i}-{i1 - 1}")
                    st["batches_checked"] += 1
                    st["cells_checked"] += len(b_cells)
                # ---- forward passes ----
                for j, cell in zip(b_idx, b_cells):
                    gp = cell.gene_positions
                    seqpos = np.nonzero(gp >= 0)[0]
                    u = var_to_U[vm.var_indices[gp[seqpos]]]
                    keep = u >= 0
                    seqpos, u = seqpos[keep], u[keep]
                    tf = time.time()
                    sub, nll, top1 = forward_states(model, cell.token_ids, seqpos, DEVICE)
                    blk = int(C["block"][j])
                    acc[blk][:, u, :] += sub
                    cnt[blk, u] += 1
                    if bool(C["is_ss2"][j]):
                        acc[SS2_SLOT][:, u, :] += sub
                        cnt[SS2_SLOT, u] += 1
                    percell[j] = sub.sum(axis=1, dtype=np.float64).astype(np.float32)
                    cell_recs.append({"pos": int(j), "row": int(rows[j]), "block": blk, "is_ss2": bool(C["is_ss2"][j]),
                                      "n_tokens": int(len(cell.token_ids)), "n_panelU": int(len(u)),
                                      "nll": nll, "top1": top1, "fwd_s": round(time.time() - tf, 4)})
                if DEVICE == "mps":
                    try:
                        mps_peak = max(mps_peak, torch.mps.driver_allocated_memory() / 1e9)
                    except Exception:
                        pass
                i = i1
                if (i // BATCH) % 4 == 0:
                    H.free_device_cache()
        # ---- checkpoint ----
        atomic_save_npy(P["acc"], acc)
        atomic_save_npy(P["cnt"], cnt)
        atomic_save_npy(P["percell"], percell)
        pd.DataFrame(cell_recs).to_csv(P["cells"], index=False)
        st["next"] = int(i)
        st["done"] = bool(i >= n)
        st["chunks"].append({"start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t_chunk)),
                             "cells": [start_i, int(i)], "wall_s": round(time.time() - t_chunk, 1),
                             "min_free_gb": round(min_free, 2), "peak_rss_gb": round(peak_rss_gb(), 2),
                             "mps_driver_peak_gb": round(mps_peak, 2), "stopped": stopped_reason if stop_now else ""})
        write_json(P["state"], st)
        print(f"[extract] {job}: cells {start_i}->{i}/{n} in {time.time() - t_chunk:.0f}s "
              f"(min free {min_free:.1f} GB, RSS peak {peak_rss_gb():.2f} GB, MPS {mps_peak:.2f} GB)")
        if st["done"]:
            finalize_job(job, panel, cells_all, st, acc, cnt)
        del acc, cnt, percell
        if stop_now:
            break
    all_done = all((read_json(job_paths(j)["state"], {}) or {}).get("done", False) for j in JOBS)
    print(f"STATUS stopped={stopped_reason} ALL_DONE {all_done} call_wall_s={time.time() - t_call:.0f}")
    if stopped_reason == "memory":
        sys.exit(3)


def finalize_job(job, panel, cells_all, st, acc, cnt):
    """Panel-D per-gene means (deployed layout) + run_config.json for the job folder."""
    P = job_paths(job)
    kind, s = job.rsplit("_", 1)
    D_in_U = np.array(panel["D_in_U"], dtype=np.int64)
    tot = acc[:N_BLOCKS].sum(axis=0)
    c = cnt[:N_BLOCKS].sum(axis=0).astype(np.float32)
    denom = c.copy()
    denom[denom == 0] = np.nan
    emb = np.nan_to_num(tot / denom[None, :, None], nan=0.0).astype(np.float32)
    np.save(P["dir"] / "layer_gene_embeddings_panelD.npy", emb[:, D_in_U])
    np.save(P["dir"] / "gene_counts_panelD.npy", cnt[:N_BLOCKS].sum(axis=0)[D_in_U].astype(np.int32))
    prc = read_json(PREP / "run_config.json")
    C = cells_all[s]
    rc = {
        "stage": "extract", "job": job, "model": st["model_info"], "env": H.env_info(DEVICE),
        "code": code_record(), "model_safetensors_sha256": sha256_file(MODEL_DIR / "model.safetensors"),
        "dataset": DATASET, "max_len": MAX_LEN, "sample": s,
        "cells": [int(r) for r, k in zip(C["rows"], C["kept"]) if k],
        "feature_ids": {"panel_D_ensembl": prc["feature_ids"]["panel_D_ensembl"],
                        "panel_U_ensembl": panel["U_ensembl"]},
        "seeds": {"block_seed": BLOCK_SEED, "random_init_seed": st["model_info"].get("init_seed")},
        "input_encoding": {**prc["input_encoding"][s],
                           "per_forward_check": f"tokenize_counts(check=True) + assert_encoding_batch on every "
                                                f"{BATCH}-cell batch; {st['cells_checked']} cells in "
                                                f"{st['batches_checked']} batches passed; tokens equal prepare/cells_{s}.npz"},
        "chunks": st["chunks"], "wall_seconds_total": round(sum(ch["wall_s"] for ch in st["chunks"]), 1),
        "outputs": {"acc_sums.npy": "(11, 12, G_U, 1232) float32: slots 0-9 = sums over the cells of each random "
                                    "block, slot 10 = sums over Smart-seq2 cells only",
                    "acc_counts.npy": "(11, G_U) cell counts per slot and gene",
                    "percell_sums.npy": "(n, 12, 1232) per-cell sum over its panel-U positions (checksum)",
                    "layer_gene_embeddings_panelD.npy": "(12, 1500, 1232) per-gene means over all cells, deployed layout"},
    }
    write_json(P["dir"] / "run_config.json", rc)


# =============================================================================
# Analysis helpers
# =============================================================================
def job_done(job) -> bool:
    return bool((read_json(job_paths(job)["state"], {}) or {}).get("done", False))


def load_job(job):
    P = job_paths(job)
    acc = np.load(P["acc"], mmap_mode="r")
    cnt = np.load(P["cnt"])
    return acc, cnt


def means_from(acc, cnt, slots_weights: np.ndarray, layer: int | None = None, genes: np.ndarray | None = None):
    """Per-gene means from slot sums. slots_weights: (N_SLOTS,) multiplicities (can be negative for
    'all minus SS2'). Returns (means float32 (L or 1, g, H), counts (g,))."""
    w = np.asarray(slots_weights, dtype=np.float64)
    gi = slice(None) if genes is None else genes
    c = (w[:, None] * cnt[:, gi]).sum(axis=0)
    if layer is None:
        tot = np.zeros((N_STATES,) + (cnt[:, gi].shape[1], HIDDEN), dtype=np.float32)
        for sl in np.nonzero(w)[0]:
            tot += np.float32(w[sl]) * np.asarray(acc[sl][:, gi])
    else:
        tot = np.zeros((1, cnt[:, gi].shape[1], HIDDEN), dtype=np.float32)
        for sl in np.nonzero(w)[0]:
            tot[0] += np.float32(w[sl]) * np.asarray(acc[sl][layer][gi])
    denom = c.astype(np.float32)
    denom[denom == 0] = np.nan
    m = np.nan_to_num(tot / denom[None, :, None], nan=0.0).astype(np.float32)
    return m, c


W_ALL = np.r_[np.ones(N_BLOCKS), 0.0]
W_UMI = np.r_[np.ones(N_BLOCKS), -1.0]


def layer_metrics(E: np.ndarray, counts: np.ndarray, label: str, seed0: int = PHASE1_SEED,
                  with_null: bool = True, null_all_layers: bool = False) -> list:
    """phase1_svd.py per-layer metrics on the genes with count > 0 (deployed rule)."""
    nz = counts > 0
    rows = []
    for li in range(E.shape[0]):
        M = E[li][nz]
        m, _, S, _ = _svd_and_metrics(M, seed=seed0 + li)
        rec = {"label": label, "layer": li, "n_genes": m["n_genes"], "effective_rank": m["effective_rank"],
               "participation_ratio": m["participation_ratio"], "sv1_variance_fraction": m["sv1_variance_fraction"],
               "twonn_intrinsic_dim": m["twonn_intrinsic_dim"]}
        if with_null and (null_all_layers or li == E.shape[0] - 1):
            rec["feature_shuffle_er"] = feature_shuffle_er(np.asarray(M).copy(), seed0 + 777, seed0 + 999)
        rows.append(rec)
    return rows


def ci95(x) -> list:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


# =============================================================================
# Stage: analyze
# =============================================================================
def stage_analyze(args):
    import pandas as pd
    ANA.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    H.check_memory(MIN_FREE_GB)
    panel = read_json(PREP / "panel.json")
    D_in_U = np.array(panel["D_in_U"], dtype=np.int64)
    S_in_U = np.array(panel["S_in_U"], dtype=np.int64)
    part = args.part
    if part == "metrics":
        out_csv = ANA / "per_layer_metrics.csv"
        done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
        have = set(done["label"]) if len(done) else set()
        rows = []
        todo = []
        # deployed embeddings, same code
        todo.append(("deployed_A_phase0", lambda: (np.load(DEP0 / "layer_gene_embeddings.npy"),
                                                   np.load(DEP0 / "gene_counts.npy")), True))
        for sd in (43, 44):
            todo.append((f"deployed_seed{sd}_phase8", lambda sd=sd: (np.load(DEP8 / f"embeddings_seed{sd}.npy"),
                                                                     np.ones(N_HVG, dtype=np.int64)), False))
        for job in JOBS:
            if not job_done(job):
                continue
            todo.append((f"v3_{job}_panelD_all", lambda job=job: _job_means(job, W_ALL, D_in_U), True))
            todo.append((f"v3_{job}_panelD_umi", lambda job=job: _job_means(job, W_UMI, D_in_U), False))
            if job.endswith("_A"):
                todo.append((f"v3_{job}_panelS_all", lambda job=job: _job_means(job, W_ALL, S_in_U), False))
        for label, loader, null_all in todo:
            if label in have:
                continue
            if time.time() - t0 > float(args.max_minutes) * 60:
                print("[analyze metrics] time budget reached; run again")
                break
            H.check_memory(MIN_FREE_GB)
            E, c = loader()
            r = layer_metrics(E, c, label, null_all_layers=null_all)
            rows.extend(r)
            print(f"  {label}: ER " + " ".join(f"{x['effective_rank']:.1f}" for x in r)
                  + f" | shuffle L11 {r[-1].get('feature_shuffle_er', float('nan')):.1f}")
            del E
            pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_csv(out_csv, index=False)
        update_run_config(ANA, "metrics", {"time": now(), "code": code_record(), "labels_done":
                                           sorted(set(pd.read_csv(out_csv)["label"]))})
    elif part == "cka":
        stage_cka(args, t0, D_in_U)
    elif part == "erboot":
        stage_erboot(args, t0, D_in_U)
    elif part == "cka_half":
        stage_cka_half(args, t0, D_in_U)
    elif part == "summary":
        stage_summary()
    print(f"[analyze {part}] {time.time() - t0:.0f}s; peak RSS {peak_rss_gb():.2f} GB")


CKA_SETS = {"deployed": None, "v3_trained": ("trained_A", "trained_B", "trained_C"),
            "v3_rand0": ("rand0_A", "rand0_B", "rand0_C")}
PAIRS = ((0, 1), (0, 2), (1, 2))


def _centered_weighted(K, w):
    n = w.sum()
    a = K @ w / n
    c = float(w @ a / n)
    C = K - a[:, None]
    C -= a[None, :]
    C += c
    return C


def _cka3_weighted(Ks, w):
    """CKA for the 3 pairs from uncentered float64 grams, rows weighted by multiplicities w."""
    Cs = [_centered_weighted(K, w) for K in Ks]
    ww = np.outer(w, w)
    At = [ww * C for C in Cs]
    self_ = [float(np.vdot(At[k], Cs[k])) for k in range(3)]
    return [float(np.vdot(At[a], Cs[b]) / np.sqrt(self_[a] * self_[b])) for a, b in PAIRS]


def _cka_set_inputs(name, D_in_U):
    """Returns (get_layer(li) -> [3 float32 (g, H) matrices], keep mask, labels, loaded jobs or None)."""
    jobs = CKA_SETS[name]
    if jobs is None:
        Es = [np.load(DEP0 / "layer_gene_embeddings.npy", mmap_mode="r"),
              np.load(DEP8 / "embeddings_seed43.npy", mmap_mode="r"),
              np.load(DEP8 / "embeddings_seed44.npy", mmap_mode="r")]
        keep = np.ones(N_HVG, dtype=bool)
        return (lambda li: [np.asarray(E[li])[keep] for E in Es]), keep, ["42", "43", "44"], None
    loaded = [load_job(j) for j in jobs]
    cnts = [c[:N_BLOCKS].sum(axis=0)[D_in_U] for _, c in loaded]
    keep = np.all([c > 0 for c in cnts], axis=0)
    return (lambda li: [means_from(a, c, W_ALL, layer=li, genes=D_in_U)[0][0][keep] for a, c in loaded]), \
        keep, ["A", "B", "C"], loaded


def stage_cka(args, t0, D_in_U):
    """Per-layer linear CKA (deployed code) for the 3 sample pairs + gene bootstrap (resumable per layer)
    + cell-block bootstrap for the v3 sets (resumable per layer)."""
    budget = float(args.max_minutes) * 60 - 45
    res = read_json(ANA / "cka.json", {}) or {}
    for name, jobs in CKA_SETS.items():
        if jobs is not None and not all(job_done(j) for j in jobs):
            print(f"  {name}: jobs not finished; skipped")
            continue
        R = res.setdefault(name, {"per_layer": {}, "n_boot_gene": N_BOOT_GENE,
                                  "method": "linear CKA (phase8_stability.linear_cka) on per-gene mean vectors, "
                                            "panel D (genes seen in all 3 samples); gene bootstrap = genes "
                                            "resampled with replacement (multiplicity weights on the gram "
                                            "matrices), percentile 95% CI"})
        if R.get("gene_boot_complete"):
            continue
        H.check_memory(MIN_FREE_GB)
        get_layer, keep, labels, _ = _cka_set_inputs(name, D_in_U)
        R["genes_used"] = int(keep.sum())
        for li in range(N_STATES):
            if str(li) in R["per_layer"]:
                continue
            if time.time() - t0 > budget:
                write_json(ANA / "cka.json", res)
                print("[cka] time budget reached; run again")
                return
            Ms = get_layer(li)
            ck = [linear_cka(Ms[a], Ms[b]) for a, b in PAIRS]
            Ks = [np.asarray(M, dtype=np.float64) @ np.asarray(M, dtype=np.float64).T for M in Ms]
            ng = Ks[0].shape[0]
            ck64 = _cka3_weighted(Ks, np.ones(ng))
            rec = {"layer": li, "n_genes": int(ng),
                   **{f"cka_{labels[a]}_{labels[b]}": ck[k] for k, (a, b) in enumerate(PAIRS)},
                   "cka_mean": float(np.mean(ck)), "cka_mean_gram_float64": float(np.mean(ck64))}
            if li >= 1:
                rng = np.random.default_rng(BOOT_SEED + 100 * li)
                boots = np.zeros((N_BOOT_GENE, 3))
                for bi in range(N_BOOT_GENE):
                    w = np.bincount(rng.integers(0, ng, ng), minlength=ng).astype(np.float64)
                    boots[bi] = _cka3_weighted(Ks, w)
                rec["gene_boot_ci_mean"] = ci95(boots.mean(axis=1))
                rec["gene_boot_sd_mean"] = float(boots.mean(axis=1).std(ddof=1))
                for k, (a, b) in enumerate(PAIRS):
                    rec[f"gene_boot_ci_{labels[a]}_{labels[b]}"] = ci95(boots[:, k])
            R["per_layer"][str(li)] = rec
            write_json(ANA / "cka.json", res)
            print(f"  {name} L{li}: CKA mean {rec['cka_mean']:.5f} pairs {np.round(ck, 5)} "
                  f"gene-boot CI {np.round(rec.get('gene_boot_ci_mean', [1, 1]), 5)} ({time.time() - t0:.0f}s)")
        R["gene_boot_complete"] = True
        write_json(ANA / "cka.json", res)
    # ---- cell-block bootstrap (v3 sets only) ----
    for name in ("v3_trained", "v3_rand0"):
        jobs = CKA_SETS[name]
        if name not in res or not res[name].get("gene_boot_complete") or res[name].get("cell_boot_complete"):
            continue
        H.check_memory(MIN_FREE_GB)
        R = res[name]
        loaded = [load_job(j) for j in jobs]
        keep = np.all([c[:N_BLOCKS].sum(axis=0)[D_in_U] > 0 for _, c in loaded], axis=0)
        rngc = np.random.default_rng(BOOT_SEED + 1)
        Wd = np.array([[np.bincount(rngc.integers(0, N_BLOCKS, N_BLOCKS), minlength=N_BLOCKS) for _ in range(3)]
                       for _ in range(N_BOOT_CELL)], dtype=np.float32)       # (n_boot, 3, 10)
        bpath = ANA / f"cka_cellboot_{name}.npy"
        boot = np.load(bpath) if bpath.exists() else np.full((N_BOOT_CELL, N_STATES, 3), np.nan)
        for li in range(1, N_STATES):
            if np.all(np.isfinite(boot[:, li])):
                continue
            if time.time() - t0 > budget - 120:
                np.save(bpath, boot)
                print("[cka cell-boot] time budget reached; run again")
                return
            tl = time.time()
            blocks = [np.stack([np.asarray(a[b][li][D_in_U], dtype=np.float32) for b in range(N_BLOCKS)])
                      for a, _ in loaded]                                     # 3 x (10, G, H)
            bc = [c[:N_BLOCKS][:, D_in_U].astype(np.float32) for _, c in loaded]
            for bi in range(N_BOOT_CELL):
                Ms, oks = [], []
                for k in range(3):
                    w = Wd[bi, k]
                    cs = w @ bc[k]
                    tot = np.tensordot(w, blocks[k], axes=(0, 0))
                    ok = cs > 0
                    M = np.zeros_like(tot)
                    M[ok] = tot[ok] / cs[ok][:, None]
                    Ms.append(M)
                    oks.append(ok)
                ok_all = keep & oks[0] & oks[1] & oks[2]
                Ks = [(M[ok_all] @ M[ok_all].T).astype(np.float64) for M in Ms]
                boot[bi, li] = _cka3_weighted(Ks, np.ones(int(ok_all.sum())))
            del blocks
            np.save(bpath, boot)
            R["per_layer"][str(li)]["cell_boot_ci_mean"] = ci95(boot[:, li].mean(axis=1))
            R["per_layer"][str(li)]["cell_boot_sd_mean"] = float(boot[:, li].mean(axis=1).std(ddof=1))
            write_json(ANA / "cka.json", res)
            print(f"  {name} cell-boot L{li}: mean CKA CI {np.round(ci95(boot[:, li].mean(axis=1)), 5)} "
                  f"({time.time() - tl:.0f}s)")
        R["cell_boot_complete"] = True
        R["n_boot_cell"] = N_BOOT_CELL
        R["cell_boot_method"] = ("each sample's cells are in 10 random blocks of 200; blocks drawn with replacement, "
                                 "independently per sample; per-gene means recomputed (float32 grams); percentile "
                                 "95% CI of the 3-pair mean CKA")
        write_json(ANA / "cka.json", res)
    update_run_config(ANA, "cka", {"time": now(), "code": code_record(), "boot_seed": BOOT_SEED,
                                   "n_boot_gene": N_BOOT_GENE, "n_boot_cell": N_BOOT_CELL})


N_BOOT_HALF = 200
HALVES = (np.arange(0, 5), np.arange(5, 10))


def stage_cka_half(args, t0, D_in_U):
    """Split-half CKA inside one 2,000-cell sample: per-gene means over blocks 0-4 vs blocks 5-9
    (two disjoint random halves of ~1,000 cells). Same for the trained model (A, B, C) and the
    random-init models (A). Linear CKA (deployed code) + gene bootstrap (N_BOOT_HALF reps)."""
    budget = float(args.max_minutes) * 60 - 45
    res = read_json(ANA / "cka_half.json", {}) or {}
    for job in ("trained_A", "rand0_A", "rand1_A", "rand2_A", "trained_B", "trained_C"):
        if not job_done(job) or (job in res and res[job].get("complete")):
            continue
        H.check_memory(MIN_FREE_GB)
        acc, cnt = load_job(job)
        R = res.setdefault(job, {"per_layer": {}, "n_boot_gene": N_BOOT_HALF,
                                 "method": "per-gene means over sample blocks 0-4 vs 5-9 (disjoint halves); "
                                           "linear CKA (phase8_stability.linear_cka); gene bootstrap percentile 95% CI"})
        wh = []
        for hb in HALVES:
            w = np.zeros(N_SLOTS)
            w[hb] = 1.0
            wh.append(w)
        c_h = [(w[:, None] * cnt).sum(axis=0)[D_in_U] for w in wh]
        keep = (c_h[0] > 0) & (c_h[1] > 0)
        R["genes_used"] = int(keep.sum())
        import pandas as pd
        blk = pd.read_csv(job_paths(job)["cells"])["block"].to_numpy()
        R["cells_per_half"] = [int(np.isin(blk, hb).sum()) for hb in HALVES]
        for li in range(1, N_STATES):
            if str(li) in R["per_layer"]:
                continue
            if time.time() - t0 > budget:
                write_json(ANA / "cka_half.json", res)
                print("[cka_half] time budget reached; run again")
                return
            Ms = [means_from(acc, cnt, w, layer=li, genes=D_in_U)[0][0][keep] for w in wh]
            ck = linear_cka(Ms[0], Ms[1])
            Ks = [np.asarray(M, dtype=np.float64) @ np.asarray(M, dtype=np.float64).T for M in Ms]
            ng = Ks[0].shape[0]
            rng = np.random.default_rng(BOOT_SEED + 500 + li)
            boots = np.zeros(N_BOOT_HALF)
            for bi in range(N_BOOT_HALF):
                w = np.bincount(rng.integers(0, ng, ng), minlength=ng).astype(np.float64)
                Ca = _centered_weighted(Ks[0], w)
                Cb = _centered_weighted(Ks[1], w)
                ww_cache = np.outer(w, w)
                At = ww_cache * Ca
                boots[bi] = float(np.vdot(At, Cb) / np.sqrt(np.vdot(At, Ca) * np.vdot(ww_cache * Cb, Cb)))
            R["per_layer"][str(li)] = {"layer": li, "cka": ck, "gene_boot_ci": ci95(boots),
                                       "gene_boot_sd": float(boots.std(ddof=1))}
            write_json(ANA / "cka_half.json", res)
            print(f"  {job} half-split L{li}: CKA {ck:.5f} CI {np.round(ci95(boots), 5)} ({time.time() - t0:.0f}s)")
        R["complete"] = True
        write_json(ANA / "cka_half.json", res)
    update_run_config(ANA, "cka_half", {"time": now(), "code": code_record(), "boot_seed": BOOT_SEED + 500,
                                        "n_boot": N_BOOT_HALF})


def stage_erboot(args, t0, D_in_U):
    """Cell-block bootstrap of effective rank at L1, L6, L11 (eigenvalue route, float64)."""
    budget = float(args.max_minutes) * 60 - 45
    res = read_json(ANA / "er_cellboot.json", {}) or {}
    layers = [1, 6, 11]
    for job in ("trained_A", "rand0_A", "rand1_A", "rand2_A", "trained_B", "trained_C"):
        if job in res or not job_done(job):
            continue
        if time.time() - t0 > budget - 200:
            print("[analyze erboot] time budget reached; run again")
            break
        H.check_memory(MIN_FREE_GB)
        acc, cnt = load_job(job)
        rngb = np.random.default_rng(BOOT_SEED + 2)
        Wb = np.array([np.bincount(rngb.integers(0, N_BLOCKS, N_BLOCKS), minlength=N_BLOCKS)
                       for _ in range(N_BOOT_CELL)], dtype=np.float64)
        bc = cnt[:N_BLOCKS][:, D_in_U].astype(np.float64)
        ers = np.full((N_BOOT_CELL, len(layers)), np.nan)
        for k, li in enumerate(layers):
            blocks = np.stack([np.asarray(acc[b][li][D_in_U], dtype=np.float64) for b in range(N_BLOCKS)])
            for bi, w in enumerate(Wb):
                cs = w @ bc
                ok = cs > 0
                tot = np.tensordot(w, blocks, axes=(0, 0))
                M = tot[ok] / cs[ok][:, None]
                Mc = M - M.mean(axis=0, keepdims=True)
                ev = np.clip(np.linalg.eigvalsh(Mc.T @ Mc), 0, None)
                p = ev / ev.sum()
                p = p[p > 0]
                ers[bi, k] = float(np.exp(-(p * np.log(p)).sum()))
            del blocks
            print(f"  {job} L{li}: ER cell-boot CI {np.round(ci95(ers[:, k]), 1)} ({time.time() - t0:.0f}s)")
        res[job] = {"layers": layers, "er_ci": {str(li): ci95(ers[:, k]) for k, li in enumerate(layers)},
                    "er_boot_mean": {str(li): float(np.nanmean(ers[:, k])) for k, li in enumerate(layers)},
                    "ratio_L1_over_L11_ci": ci95(ers[:, 0] / ers[:, 2]),
                    "ratio_L1_over_L11_boot_mean": float(np.nanmean(ers[:, 0] / ers[:, 2])),
                    "n_boot": N_BOOT_CELL,
                    "method": "10 random blocks of 200 cells resampled with replacement; per-gene means recomputed; "
                              "ER from eigenvalues of the float64 covariance (same quantity as the SVD route); "
                              "percentile 95% CI"}
        np.save(ANA / f"er_cellboot_{job}.npy", ers)
        write_json(ANA / "er_cellboot.json", res)
    update_run_config(ANA, "erboot", {"time": now(), "code": code_record(), "boot_seed": BOOT_SEED + 2})


def _job_means(job, w, genes):
    acc, cnt = load_job(job)
    m, c = means_from(acc, cnt, w, genes=genes)
    return m, c


def stage_summary():
    import pandas as pd
    from scipy.stats import spearmanr
    met = pd.read_csv(ANA / "per_layer_metrics.csv")
    cka = read_json(ANA / "cka.json", {}) or {}
    erb = read_json(ANA / "er_cellboot.json", {}) or {}
    summ = {"time": now()}

    def er_curve(label):
        d = met[met["label"] == label].sort_values("layer")
        return d["effective_rank"].to_numpy(), d

    out = {}
    for label in sorted(met["label"].unique()):
        er, d = er_curve(label)
        rho, p = spearmanr(np.arange(len(er)), er)
        rho1, p1 = spearmanr(np.arange(1, len(er)), er[1:])
        out[label] = {"er": [round(float(x), 2) for x in er],
                      "er_L0": float(er[0]), "er_L1": float(er[1]), "er_L11": float(er[-1]),
                      "ratio_L0_over_L11": float(er[0] / er[-1]), "ratio_L1_over_L11": float(er[1] / er[-1]),
                      "spearman_depth_L0_L11": [float(rho), float(p)], "spearman_depth_L1_L11": [float(rho1), float(p1)],
                      "feature_shuffle_er_L11": float(d["feature_shuffle_er"].iloc[-1])
                      if "feature_shuffle_er" in d and np.isfinite(d["feature_shuffle_er"].iloc[-1]) else None,
                      "twonn": [round(float(x), 2) for x in d["twonn_intrinsic_dim"]],
                      "pr": [round(float(x), 2) for x in d["participation_ratio"]]}
    summ["per_label"] = out
    # random-init spread at sample A
    rand = [f"v3_rand{s}_A_panelD_all" for s in RANDOM_INIT_SEEDS if f"v3_rand{s}_A_panelD_all" in out]
    if rand:
        R = np.array([out[l]["er"] for l in rand])
        summ["random_init_A"] = {"labels": rand, "er_mean": R.mean(axis=0).round(2).tolist(),
                                 "er_min": R.min(axis=0).round(2).tolist(), "er_max": R.max(axis=0).round(2).tolist(),
                                 "ratio_L1_over_L11": [out[l]["ratio_L1_over_L11"] for l in rand]}
    tr = [f"v3_trained_{s}_panelD_all" for s in SAMPLES if f"v3_trained_{s}_panelD_all" in out]
    if tr:
        T = np.array([out[l]["er"] for l in tr])
        summ["trained_disjoint_samples"] = {"labels": tr, "er_mean": T.mean(axis=0).round(2).tolist(),
                                            "er_min": T.min(axis=0).round(2).tolist(),
                                            "er_max": T.max(axis=0).round(2).tolist()}
    summ["cka"] = {k: [{kk: v[kk] for kk in v if kk in ("layer", "cka_mean", "gene_boot_ci_mean", "cell_boot_ci_mean")}
                       for _, v in sorted(cka[k]["per_layer"].items(), key=lambda t: int(t[0]))] for k in cka}
    summ["er_cellboot"] = erb
    write_json(ANA / "summary.json", summ)
    print(json.dumps({k: (v if k != "per_label" else {l: {kk: v[l][kk] for kk in ("er_L0", "er_L1", "er_L11",
                      "ratio_L1_over_L11", "feature_shuffle_er_L11")} for l in v}) for k, v in summ.items()
                      if k in ("per_label", "random_init_A", "trained_disjoint_samples")}, indent=1))


# =============================================================================
# Stage: verify (second-way checks)
# =============================================================================
def stage_verify(args):
    import ast

    import pandas as pd
    VER.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    H.check_memory(MIN_FREE_GB)
    res = read_json(VER / "verify.json", {}) or {}
    panel = read_json(PREP / "panel.json")
    D_in_U = np.array(panel["D_in_U"], dtype=np.int64)
    part = args.part

    if part in ("code", "all"):
        # V1: copied metric code equals the deployed source
        def fsrc(path, name):
            tree = ast.parse(Path(path).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == name:
                    return ast.dump(node)
            return None
        mine = ast.parse(SCRIPT_PATH.read_text())
        mine_f = {n.name: ast.dump(n) for n in ast.walk(mine) if isinstance(n, ast.FunctionDef)}
        same = {}
        for fn in ("_two_nn", "_effective_rank", "_participation_ratio", "_svd_and_metrics"):
            same[fn] = fsrc(RUN / "scripts/phase1_svd.py", fn) == mine_f.get(fn)
        same["linear_cka"] = fsrc(RUN / "scripts/phase8_stability.py", "linear_cka") == mine_f.get("linear_cka")
        res["V1_metric_code_identical_to_deployed"] = same
        print("V1", same)

    if part in ("samples", "all"):
        prc = read_json(PREP / "run_config.json")
        cells = {s: set(prc["cells"][s]) for s in SAMPLES}
        N = n_total_cells()
        A_dep = set(np.random.default_rng(42).choice(N, size=N_CELLS, replace=False).tolist())
        res["V2_samples"] = {"sizes": {s: len(cells[s]) for s in SAMPLES},
                             "pairwise_overlap": {f"{a}{b}": len(cells[a] & cells[b]) for a, b in (("A", "B"), ("A", "C"), ("B", "C"))},
                             "A_equals_deployed_phase0_draw": cells["A"] == A_dep}
        print("V2", res["V2_samples"])

    if part in ("tokens", "all"):
        # V3: independent re-tokenisation (own code: raw/X counts / median, float64 stable sort)
        import h5py
        vm = I.get_var_map(DATASET)
        med = vm.medians.astype(np.float64)
        rng = np.random.default_rng(BOOT_SEED + 3)
        n_ok, n_tot, n_tie_only = 0, 0, 0
        with h5py.File(I.DATASETS[DATASET]["path"], "r") as f:
            node = f["raw"]["X"]
            ip, ind, dat = node["indptr"], node["indices"], node["data"]
            for s in SAMPLES:
                C = dict(np.load(PREP / f"cells_{s}.npz"))
                for j in rng.choice(np.nonzero(C["kept"])[0], size=40, replace=False):
                    r = int(C["rows"][j])
                    a, b = int(ip[r]), int(ip[r + 1])
                    x = np.zeros(vm.n_vars)
                    x[np.asarray(ind[a:b])] = np.asarray(dat[a:b], dtype=np.float64)
                    assert np.all(x == np.round(x))
                    v = x[vm.var_indices] / med
                    expr = np.nonzero(v > 0)[0]
                    order = expr[np.argsort(-v[expr], kind="stable")][: MAX_LEN - 2]
                    mine_t = np.r_[2, vm.token_ids[order], 3]
                    saved = C["tokens"][C["offsets"][j]:C["offsets"][j + 1]]
                    n_tot += 1
                    if np.array_equal(mine_t, saved):
                        n_ok += 1
                    elif len(mine_t) == len(saved):
                        diff = np.nonzero(mine_t != saved)[0]
                        vals_m = v[[vm.tok2col[int(t)] for t in mine_t[diff]]]
                        vals_s = v[[vm.tok2col[int(t)] for t in saved[diff]]]
                        if np.allclose(vals_m, vals_s, rtol=1e-6, atol=0):
                            n_tie_only += 1
        res["V3_independent_tokenisation"] = {"cells": n_tot, "identical": n_ok, "differ_only_in_ties": n_tie_only}
        print("V3", res["V3_independent_tokenisation"])

    if part in ("sums", "all"):
        # V4: bookkeeping — sum over cells of per-cell sums == sum over genes of block sums, per layer
        out = {}
        for job in JOBS + OPTIONAL_JOBS:
            if not job_done(job):
                continue
            acc, cnt = load_job(job)
            pc = np.load(job_paths(job)["percell"], mmap_mode="r")
            tot_cells = np.asarray(pc, dtype=np.float64).sum(axis=0)                     # (12, H)
            tot_genes = np.zeros((N_STATES, HIDDEN))
            for b in range(N_BLOCKS):
                tot_genes += np.asarray(acc[b], dtype=np.float64).sum(axis=1)
            rel = float(np.abs(tot_cells - tot_genes).max() / np.abs(tot_genes).max())
            cells_csv = pd.read_csv(job_paths(job)["cells"])
            out[job] = {"max_rel_diff": rel, "n_cells": int(len(cells_csv)),
                        "count_total_equals_panel_positions": int(cnt[:N_BLOCKS].sum()) == int(cells_csv["n_panelU"].sum()),
                        "nll_mean_umi": float(cells_csv.loc[~cells_csv["is_ss2"], "nll"].mean()),
                        "nll_mean_ss2": float(cells_csv.loc[cells_csv["is_ss2"], "nll"].mean()) if cells_csv["is_ss2"].any() else None,
                        "top1_mean_umi": float(cells_csv.loc[~cells_csv["is_ss2"], "top1"].mean())}
            print("V4", job, out[job])
        res["V4_accumulation_bookkeeping"] = out

    if part in ("l0", "all"):
        # V5: layer 0 per-gene mean == the model's embedding row (encoding-independent check)
        from safetensors.numpy import load_file
        emb_tr = load_file(str(MODEL_DIR / "model.safetensors"))["model.embed_tokens.weight"]
        tokU = np.array(panel["U_token_id"], dtype=np.int64)[D_in_U]
        out = {}
        for job in JOBS:
            if not job_done(job):
                continue
            E0 = np.load(job_paths(job)["dir"] / "layer_gene_embeddings_panelD.npy", mmap_mode="r")[0]
            if job.startswith("trained"):
                ref = emb_tr[tokU]
            else:
                from transformers import LlamaConfig, LlamaForCausalLM
                cfg = LlamaConfig.from_pretrained(str(MODEL_DIR))
                cfg._attn_implementation = "eager"
                torch.manual_seed(int(job.split("_")[0][4:]))
                m = LlamaForCausalLM(cfg)
                ref = m.model.embed_tokens.weight.detach().numpy()[tokU]
                del m
            out[job] = float(np.abs(np.asarray(E0) - ref).max() / np.abs(ref).max())
            print("V5", job, out[job])
        dep0 = np.load(DEP0 / "layer_gene_embeddings.npy", mmap_mode="r")[0]
        out["deployed_phase0"] = float(np.abs(np.asarray(dep0) - emb_tr[tokU]).max() / np.abs(emb_tr[tokU]).max())
        res["V5_layer0_equals_embedding_rel_maxdiff"] = out

    if part in ("metrics2", "all"):
        # V6: ER by a second route (eigenvalues of the float64 covariance) and CKA via HSIC on grams
        out = {}
        labels = {"deployed_A_phase0": (np.load(DEP0 / "layer_gene_embeddings.npy", mmap_mode="r"), None)}
        for job in ("trained_A", "rand0_A"):
            if job_done(job):
                labels[f"v3_{job}"] = (np.load(job_paths(job)["dir"] / "layer_gene_embeddings_panelD.npy", mmap_mode="r"),
                                       np.load(job_paths(job)["dir"] / "gene_counts_panelD.npy"))
        for lab, (E, c) in labels.items():
            ers = []
            for li in range(N_STATES):
                M = np.asarray(E[li], dtype=np.float64)
                if c is not None:
                    M = M[c > 0]
                Mc = M - M.mean(axis=0)
                ev = np.clip(np.linalg.eigvalsh(Mc.T @ Mc), 0, None)
                p = ev / ev.sum()
                p = p[p > 0]
                ers.append(float(np.exp(-(p * np.log(p)).sum())))
            out[lab] = [round(x, 3) for x in ers]
            print("V6 ER(eig)", lab, out[lab])
        res["V6_er_second_route_eigvalsh_float64"] = out
        # CKA second route: HSIC with explicit centering matrix on grams, float64
        ck = {}
        for name, jobs in (("v3_trained", ("trained_A", "trained_B", "trained_C")),
                           ("v3_rand0", ("rand0_A", "rand0_B", "rand0_C"))):
            if not all(job_done(j) for j in jobs):
                continue
            Es = [np.load(job_paths(j)["dir"] / "layer_gene_embeddings_panelD.npy", mmap_mode="r") for j in jobs]
            cs = [np.load(job_paths(j)["dir"] / "gene_counts_panelD.npy") for j in jobs]
            keep = np.all([c > 0 for c in cs], axis=0)
            n = int(keep.sum())
            Hc = np.eye(n) - 1.0 / n
            vals = []
            for li in (1, 6, 11):
                Ks = [Hc @ (np.asarray(E[li], dtype=np.float64)[keep] @ np.asarray(E[li], dtype=np.float64)[keep].T) @ Hc
                      for E in Es]
                hs = lambda a, b: float(np.sum(a * b))
                pair = [hs(Ks[a], Ks[b]) / np.sqrt(hs(Ks[a], Ks[a]) * hs(Ks[b], Ks[b])) for a, b in ((0, 1), (0, 2), (1, 2))]
                vals.append({"layer": li, "cka_mean_hsic": float(np.mean(pair))})
            ck[name] = vals
            print("V6 CKA(HSIC)", name, vals)
        res["V6_cka_second_route_hsic"] = ck

    if part in ("forward", "all"):
        # V7: (a) output_hidden_states == hooks_v2.ResidualEditor captures; (b) MPS == CPU; (c) per-cell
        # checksum of 2 cells re-computed from scratch equals the stored per-cell sums. Trained and rand0.
        out = {}
        panel_U = np.array(panel["panel_U_var_idx"], dtype=np.int64)
        vm = I.get_var_map(DATASET)
        var_to_U = -np.ones(vm.n_vars, dtype=np.int64)
        var_to_U[panel_U] = np.arange(len(panel_U))
        C = dict(np.load(PREP / "cells_A.npz"))
        for job in ("trained_A", "rand0_A"):
            if not job_done(job):
                continue
            kind = job.split("_")[0]
            pc = np.load(job_paths(job)["percell"], mmap_mode="r")
            recs = []
            subs = {}
            for dev in (DEVICE, "cpu"):
                model, info = build_model(kind, dev)
                for j in (int(np.nonzero(C["kept"])[0][0]), int(np.nonzero(C["kept"])[0][-1])):
                    c = I.read_counts(DATASET, [int(C["rows"][j])])[0][0]
                    cell = I.tokenize_counts(c, vm, MAX_LEN, check=True)
                    I.assert_encoding_batch([c], [cell], vm, MAX_LEN, label=f"verify {job} {j}")
                    gp = cell.gene_positions
                    seqpos = np.nonzero(gp >= 0)[0]
                    u = var_to_U[vm.var_indices[gp[seqpos]]]
                    seqpos = seqpos[u >= 0]
                    sub, nll, _ = forward_states(model, cell.token_ids, seqpos, dev)
                    ids = torch.from_numpy(cell.token_ids[None, :].astype(np.int64)).to(dev)
                    ed = H.ResidualEditor(model, saes=None, edits=[], capture=list(range(12)))
                    with torch.no_grad(), ed:
                        model(input_ids=ids, use_cache=False)
                    cap = np.stack([ed.captured[l]["pre"][0].float().cpu().numpy() for l in range(12)])[:, seqpos]
                    chk = sub.sum(axis=1, dtype=np.float64)
                    recs.append({"device": dev, "cell_pos": j,
                                 "hidden_vs_hooks_capture_maxabs": float(np.abs(cap - sub).max()),
                                 "percell_sum_rel_diff_vs_stored": float(np.abs(chk - pc[j]).max() / np.abs(pc[j]).max()),
                                 "nll": nll})
                    subs[(dev, j)] = sub
                    if dev == "cpu" and (DEVICE, j) in subs and DEVICE != "cpu":
                        ref = subs[(DEVICE, j)]
                        recs[-1]["cpu_vs_mps_rel_maxdiff_per_layer"] = [
                            float(np.abs(sub[l] - ref[l]).max() / np.abs(ref[l]).max()) for l in range(N_STATES)]
                del model
                H.free_device_cache()
            out[job] = recs
            print("V7", job, recs)
        res["V7_forward_rechecks"] = out

    res["time"] = now()
    write_json(VER / "verify.json", res)
    update_run_config(VER, f"verify_{part}", {"time": now(), "code": code_record(), "env": H.env_info(DEVICE),
                                              "wall_s": round(time.time() - t0, 1)}, append=False)
    print(f"[verify {part}] {time.time() - t0:.0f}s")


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "extract", "analyze", "verify"])
    ap.add_argument("--max-minutes", type=float, default=7.0)
    ap.add_argument("--jobs", default="")
    ap.add_argument("--part", default="all")
    args = ap.parse_args()
    {"prepare": stage_prepare, "extract": stage_extract, "analyze": stage_analyze, "verify": stage_verify}[args.stage](args)


if __name__ == "__main__":
    main()
