"""inputs_v3 -- correctly encoded MaxToki inputs (v3).

Why this file exists
--------------------
checks/INPUT_ENCODING_AUDIT.md found that every MaxToki run except RPE1 passed the
stored ``X`` row straight to ``MaxTokiTokenizer.tokenize_cell``. In those files ``X``
holds log1p(CP10k) (Krasnow: log1p(CPM)), not counts. So the model saw genes ranked
by log1p(CP10k) / median instead of counts / median. The audit measured Spearman
0.77-0.89 and top-200 overlap 0.50-0.70 between the fed order and the right order.

What MaxToki expects (rank-value encoding inherited from Geneformer)
--------------------------------------------------------------------
    value(g) = counts(g) / total_counts * 10,000 / median(g)
    keep genes with value > 0, sort high -> low, keep the first max_len - 2,
    wrap with <bos> (2) ... <eos> (3).
``median(g)`` is the Geneformer gc104M gene median (setup/maxtoki_adapter.py,
DEFAULT_GENE_MEDIAN_PKL). The per-cell factor 10,000 / total does not change the
order inside a cell, so any per-cell constant multiple of counts gives the same order.

Correct count source per file (audit sections 1 and V1)
------------------------------------------------------
  k562 (replogle_concat), adamson : expm1(X)  (X = log1p(CP10k); expm1(X) is a whole-number
                                    multiple of integer counts; no raw counts in these files).
                                    By default the loader returns round(expm1(X) / unit), where
                                    unit = the smallest non-zero expm1(X) in the cell (= 1 count).
                                    These are the integer counts (up to one per-cell factor) with
                                    the float32 rounding noise of X removed (relative ~3e-7, which
                                    otherwise swaps a few near-tied genes). expm1_mode="float"
                                    returns expm1(X) itself.
  rpe1 (ReplogleWeissman2022_rpe1): X         (already integer counts)
  ts_immune, ts_immune_sub20k,
  ts_lung, ts_kidney, krasnow     : raw/X     (integer counts; raw/var == var, checked)
                                    (TS alternative: layers/decontXcounts, source="decontX")

Every loader here checks its rows before returning them:
  * integer counts: every non-zero value is a whole number;
  * expm1(X): every non-zero value is a whole-number multiple of the smallest non-zero
    value in that cell (tolerance max(1e-3, 5e-6 * multiple); log1p values fail this).
``check_encoding`` then checks that a token sequence is exactly the counts / median
order of that cell (descending, top max_len - 2 kept; values equal within 1e-6 relative,
i.e. float32 rounding, may come in either order) and
that the counts it was given look like counts. Loaders run it on every cell they return
and raise ``EncodingError`` if anything fails. Later scripts must also call
``assert_encoding_batch`` (or use these loaders) before any forward pass.

Public API (stable; later agents may ADD functions at the end, never change these)
---------------------------------------------------------------------------------
  DATASETS, EncodingError, VarMap
  get_tokenizer(), var_ensembl(name), get_var_map(name)
  check_count_like(row)                          -> dict
  CountReader(name, source=None, expm1_mode="integer")  (context manager: .counts(row), .stored_X(row))
  read_counts(name, rows, source=None)           -> (float64 array (n, n_vars), list[dict])
  tokenize_counts(counts, var_map, max_len)      -> TokenizedCell | None   (checked)
  check_encoding(cell_counts, tokens, var_map, max_len=None) -> dict ("pass" key)
  assert_encoding_batch(counts_rows, token_seqs, var_map, max_len, n_first=None) -> dict
  tokenize_rows(name, rows, max_len, source=None) -> (cells, kept_rows, info)
  load_k562_control_cells(n_cells, pool=100, seed=42, max_len=2048)
                     -> (cells, rows, h5_path)   drop-in for hooks_v2.load_k562_control_cells
  deployed_tokenize_X(name, rows, max_len)       -> the OLD (wrong) path, for comparison only
  full_order(counts, var_map)                    -> untruncated gene-token order
  order_agreement(order_a, order_b, max_len)     -> audit-style agreement measures
  read_obs_column(name, column, rows=None)       -> obs values (categoricals decoded), e.g. "assay"
  encoding_record(...), file_fingerprint(path), array_sha256(a), sha256_file(path)

Note for full-length (Smart-seq2/3) cells: raw/X holds READ counts, which grow with gene
length. With the correct encoding MaxToki-217M predicts the next gene in these cells far worse
than in UMI (10x, Perturb-seq) cells (checks/V3_INPUTS_REPORT.md). Use read_obs_column(name,
"assay") to find them.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np

SETUP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SETUP_DIR.parent
if str(SETUP_DIR) not in sys.path:
    sys.path.insert(0, str(SETUP_DIR))

from dataset_loader import ADAMSON_H5, K562_H5, RPE1_H5, SYM2ENS_PKL  # noqa: E402
from maxtoki_adapter import DEFAULT_GENE_MEDIAN_PKL, DEFAULT_TOKEN_DICT, MaxTokiTokenizer, TokenizedCell  # noqa: E402

TS_DIR = Path("<DATA_ROOT>/biodyn-work/single_cell_mechinterp/data/raw")

BOS, EOS, PAD, MASK = MaxTokiTokenizer.BOS, MaxTokiTokenizer.EOS, MaxTokiTokenizer.PAD, MaxTokiTokenizer.MASK
TARGET_SUM = 1e4

# var_kind: how to get Ensembl ids aligned to var.
#   "symbol"   : var/gene_name_index symbols -> Ensembl via the Geneformer gc104M dict
#   "ensembl"  : var/ensembl_id
#   "index"    : var/_index (Ensembl ids; any ".version" suffix is dropped)
DATASETS = {
    "k562":             dict(path=K562_H5, source="expm1_X", var_kind="symbol",
                             note="replogle_concat; X = log1p(CP10k); also HEPG2/Jurkat/RPE1 rows"),
    "adamson":          dict(path=ADAMSON_H5, source="expm1_X", var_kind="ensembl",
                             note="Adamson 2016; X = log1p(CP10k) (normalised before gene subsetting)"),
    "rpe1":             dict(path=RPE1_H5, source="X_counts", var_kind="ensembl",
                             note="ReplogleWeissman2022_rpe1; X = integer counts"),
    "ts_immune":        dict(path=TS_DIR / "tabula_sapiens_immune.h5ad", source="raw_X", var_kind="index",
                             note="X = log1p(decontXcounts CP10k); raw/X = integer counts"),
    "ts_immune_sub20k": dict(path=TS_DIR / "tabula_sapiens_immune_subset_20000.h5ad", source="raw_X",
                             var_kind="index", note="as ts_immune"),
    "ts_lung":          dict(path=TS_DIR / "tabula_sapiens_lung.h5ad", source="raw_X", var_kind="index",
                             note="as ts_immune"),
    "ts_kidney":        dict(path=TS_DIR / "tabula_sapiens_kidney.h5ad", source="raw_X", var_kind="index",
                             note="as ts_immune (used only by a Geneformer script so far)"),
    "krasnow":          dict(path=TS_DIR / "krasnow_lung_smartsq2.h5ad", source="raw_X", var_kind="index",
                             note="Smart-seq2; X = log1p(CPM); raw/X = integer read counts"),
}
ALIASES = {"replogle_concat": "k562", "ts_immune_subset_20000": "ts_immune_sub20k"}

SOURCE_DESCRIPTION = {
    "expm1_X": "expm1(X) (X = log1p(CP10k)), checked per row to be whole-number multiples of the cell's "
               "smallest non-zero value; by default divided by that value and rounded (= integer counts up "
               "to one per-cell factor; expm1_mode='integer')",
    "X_counts": "X (integer counts), checked per row",
    "raw_X": "raw/X (integer counts, same genes and order as var), checked per row",
    "decontX": "layers/decontXcounts (integer counts after ambient-RNA removal), checked per row",
}
# Sources allowed per file type (the default is DATASETS[name]["source"]).
_ALLOWED_SOURCES = {
    "expm1_X": {"expm1_X"},
    "X_counts": {"X_counts"},
    "raw_X": {"raw_X", "decontX", "expm1_X"},
}

# Tolerances
INT_TOL_REL = 1e-6          # integer test: |v - round(v)| <= 1e-6 * max(1, v)
MULT_TOL_ABS = 1e-3         # whole-multiple test: |q - round(q)| <= max(1e-3, 5e-6 * q)
MULT_TOL_REL = 5e-6
MULT_MAX_Q = 1e5            # above this the multiple test has no power -> fail
ORDER_REL_TOL = 1e-6        # descending-order test on counts / median: values within 1 ppm
                            # count as tied (float32 rounding); log-order errors are far larger


class EncodingError(RuntimeError):
    """Raised when an input is not correctly rank-value encoded."""


def _resolve(name: str) -> str:
    n = ALIASES.get(name.lower(), name.lower())
    if n not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(DATASETS)}")
    return n


def _dec(a):
    return [s.decode() if isinstance(s, bytes) else s for s in a]


# =============================================================================
# Tokenizer, gene ids, var map
# =============================================================================
@lru_cache(maxsize=1)
def get_tokenizer() -> MaxTokiTokenizer:
    return MaxTokiTokenizer()


def var_ensembl(name: str) -> list:
    """Ensembl id (or None) for every var column of the dataset, in var order.
    For files with raw/X, also checks that raw/var lists the same genes in the same order."""
    name = _resolve(name)
    spec = DATASETS[name]
    with h5py.File(spec["path"], "r") as f:
        v = f["var"]
        if spec["var_kind"] == "symbol":
            with open(SYM2ENS_PKL, "rb") as fh:
                sym2ens = pickle.load(fh)
            ens = [sym2ens.get(s) for s in _dec(v["gene_name_index"][:])]
        elif spec["var_kind"] == "ensembl":
            ens = _dec(v["ensembl_id"][:])
        else:
            k = v.attrs.get("_index", "_index")
            k = k.decode() if isinstance(k, bytes) else k
            idx = _dec(v[k][:])
            ens = [s.split(".")[0] if isinstance(s, str) else None for s in idx]
            if "raw" in f and "var" in f["raw"]:
                rv = f["raw"]["var"]
                rk = rv.attrs.get("_index", "_index")
                rk = rk.decode() if isinstance(rk, bytes) else rk
                if _dec(rv[rk][:]) != idx:
                    raise EncodingError(f"{name}: raw/var genes differ from var genes")
    return ens


@dataclass
class VarMap:
    """Gene columns of one dataset that are in the MaxToki vocabulary."""
    dataset: str
    var_indices: np.ndarray   # int64 (n_vocab_genes,) columns of the count row
    token_ids: np.ndarray     # int64 (n_vocab_genes,)
    medians: np.ndarray       # float32 (n_vocab_genes,), exactly what tokenize_cell uses
    n_vars: int
    tok2col: dict = field(repr=False, default_factory=dict)   # token id -> position in var_indices

    def __post_init__(self):
        if not self.tok2col:
            self.tok2col = {int(t): i for i, t in enumerate(self.token_ids)}
        if len(self.tok2col) != len(self.token_ids):
            raise EncodingError(f"{self.dataset}: several var columns map to the same token")


@lru_cache(maxsize=None)
def get_var_map(name: str) -> VarMap:
    name = _resolve(name)
    ens = var_ensembl(name)
    vi, vt, vm = get_tokenizer().make_var_mapping(ens)
    return VarMap(dataset=name, var_indices=vi, token_ids=vt, medians=vm, n_vars=len(ens))


# =============================================================================
# Count checks
# =============================================================================
def check_count_like(row) -> dict:
    """Do the non-zero values look like counts (or one per-cell multiple of counts)?

    integer          : all non-zero values are whole numbers (>= 0).
    integer_multiple : value / smallest value is a whole number for every gene
                       (what expm1(log1p(CP10k)) gives). log1p values fail this.
    Returns {"pass", "kind", "n_nonzero", "unit", "implied_total", "max_dev", "q_max", ...}.
    """
    v = np.asarray(row, dtype=np.float64).ravel()
    nz = v[v != 0]
    out = {"n_nonzero": int(nz.size)}
    if nz.size == 0:
        return {**out, "pass": False, "kind": "empty", "reason": "no non-zero values"}
    if np.any(nz < 0) or not np.all(np.isfinite(nz)):
        return {**out, "pass": False, "kind": "invalid", "reason": "negative or non-finite values"}
    dev_int = np.abs(nz - np.round(nz))
    out["frac_integer"] = float(np.mean(dev_int <= INT_TOL_REL * np.maximum(1.0, nz)))
    if out["frac_integer"] == 1.0:
        return {**out, "pass": True, "kind": "integer", "unit": 1.0, "max_dev": float(dev_int.max()),
                "q_max": float(nz.max()), "total": float(nz.sum()), "weak": bool(nz.size < 10)}
    s = float(nz.min())
    q = nz / s
    dev = np.abs(q - np.round(q))
    tol = np.maximum(MULT_TOL_ABS, MULT_TOL_REL * q)
    ok = bool(np.all(dev <= tol)) and float(q.max()) <= MULT_MAX_Q
    return {**out, "pass": ok, "kind": "integer_multiple", "unit": s,
            "implied_total": float(TARGET_SUM / s), "max_dev": float(dev.max()),
            "max_rel_dev": float((dev / q).max()), "q_max": float(q.max()),
            "n_fail": int(np.sum(dev > tol)), "sum_multiples": float(np.round(q).sum()),
            "weak": bool(nz.size < 10),
            "reason": "" if ok else "values are not whole-number multiples of the smallest value"}


# =============================================================================
# Reading rows
# =============================================================================
def _read_row(node, i: int, n_cols: int) -> np.ndarray:
    """One row as float64, from a dense dataset or a CSR group (as stored, no transform)."""
    if isinstance(node, h5py.Dataset):
        return np.asarray(node[int(i), :], dtype=np.float32).astype(np.float64)
    ip = node["indptr"]
    a, b = int(ip[int(i)]), int(ip[int(i) + 1])
    out = np.zeros(n_cols, dtype=np.float64)
    out[np.asarray(node["indices"][a:b])] = np.asarray(node["data"][a:b], dtype=np.float32)
    return out


def _n_cols(node) -> int:
    if isinstance(node, h5py.Dataset):
        return int(node.shape[1])
    return int(node.attrs["shape"][1])


def _n_rows(node) -> int:
    if isinstance(node, h5py.Dataset):
        return int(node.shape[0])
    return int(node.attrs["shape"][0])


class CountReader:
    """Open one dataset and read correctly encoded count rows.

        with CountReader("ts_immune") as r:
            c = r.counts(12345)        # float64 (n_vars,), checked
            x = r.stored_X(12345)      # the stored X row (log1p values), no transform
    """

    def __init__(self, name: str, source: str | None = None, check: bool = True,
                 expm1_mode: str = "integer"):
        self.name = _resolve(name)
        if expm1_mode not in ("integer", "float"):
            raise ValueError("expm1_mode must be 'integer' or 'float'")
        self.expm1_mode = expm1_mode
        spec = DATASETS[self.name]
        self.path = Path(spec["path"])
        self.source = source or spec["source"]
        if self.source not in _ALLOWED_SOURCES[spec["source"]]:
            raise ValueError(f"source {self.source!r} not valid for {self.name} "
                             f"(allowed {_ALLOWED_SOURCES[spec['source']]})")
        self.check = check
        self.f = None

    def __enter__(self):
        self.f = h5py.File(self.path, "r")
        X = self.f["X"]
        self.n_vars = _n_cols(X)
        self.n_cells = _n_rows(X)
        if self.source == "raw_X":
            self.node = self.f["raw"]["X"]
        elif self.source == "decontX":
            self.node = self.f["layers"]["decontXcounts"]
        else:
            self.node = X
        if _n_cols(self.node) != self.n_vars or _n_rows(self.node) != self.n_cells:
            raise EncodingError(f"{self.name}: {self.source} shape differs from X")
        return self

    def __exit__(self, *exc):
        if self.f is not None:
            self.f.close()
        self.f = None
        return False

    def stored_X(self, row: int) -> np.ndarray:
        """The stored X row exactly as the deployed scripts read it (float32 values)."""
        return _read_row(self.f["X"], row, self.n_vars)

    def counts(self, row: int, return_check: bool = False):
        x = _read_row(self.node, row, self.n_vars)
        if self.source == "expm1_X":
            if x.max() > 30:
                raise EncodingError(f"{self.name} row {row}: X max {x.max():.1f} > 30; not log1p values")
            x = np.expm1(x)
        chk = check_count_like(x)
        if self.source == "expm1_X":
            ok = chk["pass"] and chk["kind"] == "integer_multiple"
        else:
            ok = chk["pass"] and chk["kind"] == "integer"
        chk = {**chk, "row": int(row), "source": self.source, "expected_ok": bool(ok)}
        if self.check and not ok:
            raise EncodingError(f"{self.name} row {row}: {self.source} values are not count-like: {chk}")
        if self.source == "expm1_X":
            chk["expm1_mode"] = self.expm1_mode
            if ok and self.expm1_mode == "integer":
                # whole-number multiples of the cell's unit = integer counts up to one per-cell factor
                x = np.round(x / chk["unit"])
        return (x, chk) if return_check else x


def read_counts(name: str, rows: Sequence[int], source: str | None = None, check: bool = True,
                expm1_mode: str = "integer"):
    """Counts for ``rows`` (in the given order). Returns (float64 (n, n_vars), list of row checks)."""
    out, checks = [], []
    with CountReader(name, source, check, expm1_mode) as r:
        for i in rows:
            x, c = r.counts(int(i), return_check=True)
            out.append(x)
            checks.append(c)
    return (np.stack(out) if out else np.zeros((0, 0))), checks


# =============================================================================
# Tokenising and the order check
# =============================================================================
def tokenize_counts(counts, var_map: VarMap, max_len: int, *, total: float | None = None,
                    check: bool = True, target_sum: float = TARGET_SUM) -> TokenizedCell | None:
    """MaxToki rank-value encoding of one cell from its counts.

    value = counts / total * target_sum, then setup/maxtoki_adapter.tokenize_cell divides by
    the gene median, sorts high -> low, keeps max_len - 2 genes and adds <bos>/<eos>.
    ``total`` defaults to the sum of the row (all genes in the file). Any per-cell constant
    gives the same order. With check=True the result must pass ``check_encoding``."""
    counts = np.asarray(counts, dtype=np.float64).ravel()
    tot = float(counts.sum()) if total is None else float(total)
    if not tot > 0:
        return None
    x = counts / tot * float(target_sum)
    cell = get_tokenizer().tokenize_cell(x, var_map.var_indices, var_map.token_ids,
                                         var_map.medians, max_len=int(max_len))
    if cell is None:
        return None
    if check:
        res = check_encoding(counts, cell, var_map, max_len)
        if not res["pass"]:
            raise EncodingError(f"{var_map.dataset}: token order check failed: "
                                f"{ {k: res[k] for k in ('problems', 'n_order_violations', 'count_like')} }")
    return cell


def full_order(counts, var_map: VarMap) -> np.ndarray:
    """Untruncated gene-token order (no <bos>/<eos>) of tokenize_counts."""
    c = tokenize_counts(counts, var_map, max_len=10 ** 9, check=False)
    return np.zeros(0, np.int64) if c is None else c.token_ids[1:-1]


def _strip(tokens) -> np.ndarray:
    t = np.asarray(getattr(tokens, "token_ids", tokens), dtype=np.int64).ravel()
    # drop trailing padding (saved 4,096-token arrays are padded with 0 after <eos>)
    if t.size and t[-1] == PAD:
        nz = np.nonzero(t != PAD)[0]
        t = t[: nz[-1] + 1] if nz.size else t[:0]
    return t


def check_encoding(cell_counts, tokens, var_map: VarMap | str, max_len: int | None = None, *,
                   require_count_like: bool = True, rel_tol: float = ORDER_REL_TOL) -> dict:
    """Is ``tokens`` exactly the MaxToki counts / median order of ``cell_counts``?

    cell_counts : the cell's count row over ALL var columns (counts or a per-cell multiple).
    tokens      : token ids (or a TokenizedCell), <bos> genes... <eos>, trailing PAD allowed.
    Checks: (1) <bos>/<eos> at the ends, no other special token, no repeated gene, every token
    is a vocab gene of this dataset with count > 0; (2) counts/median is non-increasing along
    the sequence (relative tolerance ``rel_tol``); (3) length = min(#expressed vocab genes,
    max_len - 2) (if max_len is None: no gene left out has a larger value than the last kept
    one); (4) the counts are count-like (check_count_like), so log values cannot pass.
    Also reports agreement with an independent stable-sort reference: positions that differ
    must hold equal values (exact ties). Returns a dict with "pass"."""
    vm = get_var_map(var_map) if isinstance(var_map, str) else var_map
    c_all = np.asarray(cell_counts, dtype=np.float64).ravel()
    problems: list[str] = []
    t = _strip(tokens)
    if t.size < 2 or t[0] != BOS or t[-1] != EOS:
        problems.append("sequence does not start with <bos> and end with <eos>")
    genes = t[1:-1] if t.size >= 2 else np.zeros(0, np.int64)
    if np.any(np.isin(genes, [BOS, EOS, PAD, MASK])):
        problems.append("special token inside the gene sequence")
    if len(np.unique(genes)) != len(genes):
        problems.append("repeated gene token")
    cols = np.array([vm.tok2col.get(int(g), -1) for g in genes], dtype=np.int64)
    if np.any(cols < 0):
        problems.append(f"{int(np.sum(cols < 0))} tokens are not vocab genes of {vm.dataset}")
        cols = cols[cols >= 0]
    med = vm.medians.astype(np.float64)
    c_vocab = c_all[vm.var_indices]
    val_all = np.where(c_vocab > 0, c_vocab / med, 0.0)
    v = val_all[cols]
    if np.any(c_vocab[cols] <= 0):
        problems.append("token for a gene with zero count")
    viol = v[1:] > v[:-1] * (1.0 + rel_tol)
    n_viol = int(viol.sum())
    expressed = np.nonzero(c_vocab > 0)[0]
    n_expr = int(expressed.size)
    if max_len is not None:
        exp_len = min(n_expr, int(max_len) - 2)
        if len(genes) != exp_len:
            problems.append(f"length {len(genes)} != expected {exp_len}")
    left_out = np.setdiff1d(expressed, cols, assume_unique=False)
    n_left_bigger = 0
    if left_out.size and v.size:
        n_left_bigger = int(np.sum(val_all[left_out] > v.min() * (1.0 + rel_tol)))
    if n_left_bigger:
        problems.append(f"{n_left_bigger} genes left out have a larger value than a kept gene")
    # independent reference: stable sort of counts / median in float64
    ref_cols = expressed[np.argsort(-val_all[expressed], kind="stable")][: len(cols)]
    m = min(len(ref_cols), len(cols))
    same = cols[:m] == ref_cols[:m]
    diff_pos = np.nonzero(~same)[0]
    ties_only = bool(np.all(np.abs(val_all[cols[diff_pos]] - val_all[ref_cols[diff_pos]])
                            <= rel_tol * np.maximum(val_all[ref_cols[diff_pos]], 1e-300))) if diff_pos.size else True
    cl = check_count_like(c_all)
    if require_count_like and not cl["pass"]:
        problems.append(f"counts are not count-like ({cl.get('kind')}): {cl.get('reason', '')}")
    if n_viol:
        problems.append(f"{n_viol} adjacent pairs out of counts/median order")
    if not ties_only:
        problems.append("differs from the stable-sort reference at positions that are not ties")
    return {
        "pass": not problems,
        "problems": problems,
        "n_genes": int(len(genes)),
        "n_expressed_vocab": n_expr,
        "n_order_violations": n_viol,
        "n_left_out_bigger": n_left_bigger,
        "same_position_as_reference": float(np.mean(same)) if m else 1.0,
        "n_positions_tie_reordered": int(diff_pos.size),
        "count_like": {k: cl.get(k) for k in ("pass", "kind", "n_nonzero", "max_dev", "q_max")},
    }


def assert_encoding_batch(counts_rows, token_seqs, var_map: VarMap | str, max_len: int,
                          n_first: int | None = None, label: str = "") -> dict:
    """Run check_encoding on the first ``n_first`` cells (all if None); raise EncodingError on
    any failure. Call this on every batch before a forward pass. Returns a summary dict."""
    n = len(token_seqs) if n_first is None else min(int(n_first), len(token_seqs))
    worst_viol, n_tie = 0, 0
    for i in range(n):
        r = check_encoding(counts_rows[i], token_seqs[i], var_map, max_len)
        if not r["pass"]:
            raise EncodingError(f"{label} cell {i}: input encoding check failed: {r['problems']}")
        worst_viol = max(worst_viol, r["n_order_violations"])
        n_tie += r["n_positions_tie_reordered"]
    return {"label": label, "n_checked": n, "n_cells": len(token_seqs), "all_pass": True,
            "max_order_violations": worst_viol, "n_positions_tie_reordered": n_tie,
            "check": "inputs_v3.check_encoding (token order == counts/median order; counts count-like)"}


def tokenize_rows(name: str, rows: Iterable[int], max_len: int, *, source: str | None = None,
                  return_counts: bool = False, expm1_mode: str = "integer"):
    """Correctly encoded cells for dataset ``name`` and the given row indices (order kept).
    Rows with no expressed vocab gene are dropped. Every cell passes check_encoding.
    Returns (cells, kept_rows, info) or (cells, kept_rows, info, counts_list)."""
    name = _resolve(name)
    vm = get_var_map(name)
    cells, kept, row_checks, counts_kept = [], [], [], []
    with CountReader(name, source, expm1_mode=expm1_mode) as r:
        for i in rows:
            c, chk = r.counts(int(i), return_check=True)
            row_checks.append(chk)
            cell = tokenize_counts(c, vm, max_len, check=True)
            if cell is None:
                continue
            cells.append(cell)
            kept.append(int(i))
            if return_counts:
                counts_kept.append(c)
        src = r.source
    info = {"dataset": name, "path": str(DATASETS[name]["path"]), "source": src,
            "source_description": SOURCE_DESCRIPTION[src], "max_len": int(max_len),
            "expm1_mode": expm1_mode if src == "expm1_X" else None,
            "n_rows_requested": len(row_checks), "n_cells": len(cells),
            "count_checks_all_pass": all(c["expected_ok"] for c in row_checks),
            "encoding_check": f"check_encoding passed on all {len(cells)} cells"}
    if return_counts:
        return cells, kept, info, counts_kept
    return cells, kept, info


def load_k562_control_cells(n_cells: int, pool: int = 100, seed: int = 42, max_len: int = 2048,
                            *, return_info: bool = False):
    """Drop-in replacement for hooks_v2.load_k562_control_cells with the CORRECT encoding.

    Same cell selection: K562 non-targeting cells of replogle_concat, rng(seed).choice of
    ``pool`` cells, sorted row indices, first ``n_cells`` that tokenize (the set of cells that
    tokenize is the same under both encodings, so the same rows come back). The input is
    round(expm1(X) / unit) (expm1(X) checked to be whole-number multiples of the cell's
    smallest value) instead of X (log1p(CP10k)).
    Every returned cell passes check_encoding. Returns (list[TokenizedCell], list[int] rows,
    h5 path) like the old function; with return_info=True a 4th item (dict) is added."""
    from dataset_loader import resolve as load_ds
    ds = load_ds("k562")
    rng = np.random.default_rng(seed)
    ctrl_all = np.where(ds.cell_of_interest_mask &
                        np.isin(ds.perturbation_codes, list(ds.control_category_codes)))[0]
    ctrl_idx = np.sort(rng.choice(ctrl_all, size=pool, replace=False))
    need = ctrl_idx[: min(pool, max(n_cells * 2, n_cells + 5))]
    vm = get_var_map("k562")
    cells, rows, checks = [], [], []
    with CountReader("k562") as r:
        for row in need:
            c, chk = r.counts(int(row), return_check=True)
            cell = tokenize_counts(c, vm, max_len, check=True)
            if cell is not None:
                cells.append(cell)
                rows.append(int(row))
                checks.append(chk)
            if len(cells) == n_cells:
                break
    if not return_info:
        return cells, rows, str(ds.h5_path)
    info = {"dataset": "k562", "path": str(ds.h5_path), "source": "expm1_X",
            "source_description": SOURCE_DESCRIPTION["expm1_X"], "selection":
            f"rng({seed}).choice(K562 non-targeting, {pool}) sorted, first {n_cells} that tokenize",
            "rows": rows, "max_len": int(max_len),
            "count_unit_max_dev": max((c["max_dev"] for c in checks), default=None),
            "encoding_check": f"check_encoding passed on all {len(cells)} cells"}
    return cells, rows, str(ds.h5_path), info


def deployed_tokenize_X(name: str, rows: Iterable[int], max_len: int):
    """OLD, WRONG path kept only for comparisons: tokenize_cell on the stored X row with no
    transform, exactly as every deployed / v2 script did. Never feed this to the model.
    Returns list of TokenizedCell | None (one per row)."""
    name = _resolve(name)
    vm = get_var_map(name)
    tok = get_tokenizer()
    out = []
    with CountReader(name, check=False) as r:
        for i in rows:
            x = r.stored_X(int(i)).astype(np.float32)
            out.append(tok.tokenize_cell(x, vm.var_indices, vm.token_ids, vm.medians, max_len=int(max_len)))
    return out


def read_obs_column(name: str, column: str, rows: Sequence[int] | None = None) -> np.ndarray:
    """One obs column of the dataset (categoricals decoded to their labels), for ``rows`` or all."""
    name = _resolve(name)
    with h5py.File(DATASETS[name]["path"], "r") as f:
        node = f["obs"][column]
        if isinstance(node, h5py.Group) and "categories" in node:
            cats = np.array(_dec(node["categories"][:]), dtype=object)
            codes = node["codes"][:] if rows is None else np.array([node["codes"][int(r)] for r in rows])
            out = np.where(codes >= 0, cats[np.maximum(codes, 0)], None)
        else:
            vals = node[:] if rows is None else np.array([node[int(r)] for r in rows])
            out = np.array(_dec(vals), dtype=object) if vals.dtype.kind in "SO" else vals
    return out


# =============================================================================
# Order comparison (same measures as checks/input_encoding_audit/order_audit.py)
# =============================================================================
def order_agreement(order_a, order_b, max_len: int) -> dict:
    """Agreement between two untruncated gene orders (no <bos>/<eos>). ``order_b`` is the
    reference. Measures as in the audit: Spearman of positions over shared genes (full and
    inside both cut sequences), top-200 / top-2,046 overlap, kept-set overlap and the share of
    token positions holding the same gene (over the cut sequences)."""
    from scipy.stats import spearmanr
    S = np.asarray(order_a, dtype=np.int64)
    R = np.asarray(order_b, dtype=np.int64)
    k = int(max_len) - 2
    Sf, Rf = S[:k], R[:k]
    pS = {int(t): i for i, t in enumerate(S)}
    pR = {int(t): i for i, t in enumerate(R)}
    common = [int(t) for t in R if int(t) in pS]
    rho = spearmanr([pS[t] for t in common], [pR[t] for t in common]).correlation if len(common) > 2 else np.nan
    sSf = set(Sf.tolist())
    pSf = {int(t): i for i, t in enumerate(Sf)}
    pRf = {int(t): i for i, t in enumerate(Rf)}
    cf = [int(t) for t in Rf if int(t) in sSf]
    rho_fed = spearmanr([pSf[t] for t in cf], [pRf[t] for t in cf]).correlation if len(cf) > 2 else np.nan
    k200, k2046 = min(200, len(R)), min(2046, len(R))
    L, m = max(len(Sf), len(Rf)), min(len(Sf), len(Rf))
    return {
        "n_a": int(len(S)), "n_b": int(len(R)), "truncated_b": bool(len(R) > k),
        "spearman_full": float(rho), "spearman_fed": float(rho_fed),
        "top200_overlap": len(set(S[:200].tolist()) & set(R[:200].tolist())) / k200 if k200 else np.nan,
        "top2046_overlap": len(set(S[:2046].tolist()) & set(R[:2046].tolist())) / k2046 if k2046 else np.nan,
        "kept_set_overlap": len(sSf & set(Rf.tolist())) / len(Rf) if len(Rf) else np.nan,
        "same_position_share": float(np.sum(Sf[:m] == Rf[:m])) / L if L else np.nan,
    }


# =============================================================================
# Provenance helpers
# =============================================================================
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_sha256(a) -> str:
    a = np.ascontiguousarray(np.asarray(a))
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def file_fingerprint(path, block: int = 16 << 20) -> dict:
    """Size, mtime and sha256 of the first and last 16 MB (the data files are 0.2-30 GB,
    too large to hash in full every run). Pair it with array_sha256 of the rows used."""
    p = Path(path)
    st = p.stat()
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        h.update(fh.read(block))
        if st.st_size > block:
            fh.seek(max(block, st.st_size - block))
            h.update(fh.read(block))
    return {"path": str(p), "size": int(st.st_size), "mtime": time.strftime("%Y-%m-%dT%H:%M:%S",
            time.localtime(st.st_mtime)), "sha256_head_tail_16MB": h.hexdigest()}


def encoding_record(name: str, rows: Sequence[int] | None = None, source: str | None = None,
                    check_summary: dict | None = None, counts_sha256: str | None = None) -> dict:
    """Block for run_config.json describing the input encoding of a run."""
    name = _resolve(name)
    src = source or DATASETS[name]["source"]
    rec = {
        "inputs_module": str(Path(__file__).resolve()),
        "inputs_v3_sha256": sha256_file(__file__),
        "dataset": name,
        "file": file_fingerprint(DATASETS[name]["path"]),
        "count_source": src,
        "count_source_description": SOURCE_DESCRIPTION[src],
        "encoding": "counts / row total * 1e4 / Geneformer gc104M gene median, ranked high->low, "
                    "first max_len-2 genes, <bos>=2 ... <eos>=3 (maxtoki_adapter.tokenize_cell)",
        "gene_median_pkl": str(DEFAULT_GENE_MEDIAN_PKL),
        "token_dictionary": str(DEFAULT_TOKEN_DICT),
    }
    if rows is not None:
        rec["rows"] = [int(r) for r in rows]
    if counts_sha256 is not None:
        rec["counts_sha256"] = counts_sha256
    if check_summary is not None:
        rec["encoding_check"] = check_summary
    return rec


# =============================================================================
# Added by item V3-2 (TF specificity), 2026-10-02. Additive only: nothing above was changed.
# =============================================================================
def counts_accept_unit_one(reader: CountReader, row: int, return_check: bool = False):
    """Same as ``reader.counts(row)``, with one extra accepted case for an expm1(X) source.

    ``CountReader.counts`` expects expm1(X) rows to be check_count_like kind "integer_multiple".
    A few K562 rows have expm1(X) values that are already whole numbers (kind "integer"; the
    smallest non-zero value is exactly 1, so the per-cell factor is 1). Whole numbers are
    whole-number multiples of 1, so these rows are counts too and give the correct order, but
    ``CountReader.counts`` rejects them. Seen in 2 of 9,200 K562 cells (rows 282850 and 401789).
    For such a row this returns round(expm1(X)) (expm1_mode "integer") or expm1(X) ("float") and
    adds ``"accepted_as": "integer_unit_1"`` to the check dict. Every other row: exactly
    ``reader.counts(row)`` (same values, same errors)."""
    if reader.source != "expm1_X":
        return reader.counts(row, return_check=return_check)
    saved = reader.check
    reader.check = False
    try:
        x, chk = reader.counts(row, return_check=True)
    finally:
        reader.check = saved
    if not chk["expected_ok"]:
        if chk["pass"] and chk["kind"] == "integer" and float(chk.get("unit", 0.0)) == 1.0:
            if reader.expm1_mode == "integer":
                x = np.round(x)
            chk = {**chk, "expected_ok": True, "accepted_as": "integer_unit_1"}
        elif saved:
            raise EncodingError(f"{reader.name} row {row}: {reader.source} values are not count-like: {chk}")
    return (x, chk) if return_check else x
