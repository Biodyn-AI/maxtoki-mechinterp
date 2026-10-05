"""Shared helpers for the v2 re-evaluation of the attention-GRN runs (revision item D1).

CPU only. No model forward pass. Reads the deployed Phase-0 / Phase-1 / Phase-2 /
Phase-3 artefacts and the raw perturbation h5ad files. Never writes outside
outputs/v2_eval/.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

PROJ = Path("<REPO_ROOT>/projects/maxtoki")
RUN_DIR = PROJ / "runs/attention-grn-217M"
OUT = RUN_DIR / "outputs"
V2 = OUT / "v2_eval"
V2.mkdir(parents=True, exist_ok=True)

TRRUST_TSV = Path("<DATA_ROOT>/biodyn-nmi-paper/"
                  "src/02_cssi_method/cssi_real_data/results/trrust_human.tsv")
SYM2ENS_PKL = Path("<DATA_ROOT>/biodyn-nmi-paper/"
                   "src/02_cssi_method/crispri_validation/data/gene_name_id_dict_gc104M.pkl")

PRIMARY_LAYER = 8          # 0-indexed; used by every deployed run (phase1/2/3 default "8")
N_BOOT = 2000
SEED = 20261001            # master seed for all v2 resampling

# run name -> (output suffix, dataset key for setup/dataset_loader.py, model folder)
RUNS = {
    "217M_K562":    ("",         "k562",    "MaxToki-217M-HF"),
    "217M_RPE1":    ("_rpe1",    "rpe1",    "MaxToki-217M-HF"),
    "217M_Adamson": ("_adamson", "adamson", "MaxToki-217M-HF"),
    "1B_K562":      ("_k562_1b", "k562",    "MaxToki-1B-HF"),
}


def sha256(path: Path, block: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def phase_dir(phase: str, run: str) -> Path:
    return OUT / f"{phase}{RUNS[run][0]}"


def load_gene_features(run: str) -> pd.DataFrame:
    return pd.read_csv(phase_dir("phase0", run) / "gene_features.csv")


def load_attention_layer(run: str, layer: int = PRIMARY_LAYER) -> np.ndarray:
    """One (G, G) slice of attention_edges_layer_mean.npy via mmap (row = query gene,
    column = key gene). Returned as a float32 copy with the diagonal left as stored."""
    a = np.load(phase_dir("phase0", run) / "attention_edges_layer_mean.npy", mmap_mode="r")
    out = np.array(a[layer], dtype=np.float32)
    del a
    return out


def n_layers_stored(run: str) -> int:
    a = np.load(phase_dir("phase0", run) / "attention_edges_layer_mean.npy", mmap_mode="r")
    n = int(a.shape[0])
    del a
    return n


def load_trrust() -> pd.DataFrame:
    t = pd.read_csv(TRRUST_TSV, sep="\t", header=None, names=["tf", "target", "mode", "pmid"])
    t["tf"] = t["tf"].str.upper()
    t["target"] = t["target"].str.upper()
    return t


def trrust_matrix(symbols_upper: list[str], trrust: pd.DataFrame) -> np.ndarray:
    """Deployed rule (phase3_residualization.py:98-103): E[i, j] = 1 if TRRUST has
    TF=gene i -> target=gene j, both in the gene set, i != j."""
    s2i = {s: i for i, s in enumerate(symbols_upper)}
    G = len(symbols_upper)
    E = np.zeros((G, G), dtype=np.int8)
    for tf, tg in zip(trrust["tf"], trrust["target"]):
        i = s2i.get(tf)
        j = s2i.get(tg)
        if i is not None and j is not None and i != j:
            E[i, j] = 1
    return E


def evaluated_tf_rows(symbols_upper: list[str], trrust: pd.DataFrame, E: np.ndarray) -> np.ndarray:
    """Deployed rule (phase3_residualization.py:108-113): TF in gene set with >= 3
    TRRUST targets in the gene set. Returned in ascending gene-index order."""
    s2i = {s: i for i, s in enumerate(symbols_upper)}
    rows = []
    for tf in set(trrust["tf"]) & set(symbols_upper):
        i = s2i[tf]
        if E[i].sum() >= 3:
            rows.append(i)
    return np.array(sorted(rows), dtype=np.int64)


# ---------------------------------------------------------------- AUROC helpers
def auroc_rank(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUROC with average ranks for ties (tie-safe)."""
    y = np.asarray(y).astype(bool)
    n1 = int(y.sum())
    n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s, method="average")
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def tie_self_test() -> dict:
    """AUROC must be exactly 0.5 for a constant score and match sklearn on tied data."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    s_const = np.zeros(5000)
    s_tied = rng.integers(0, 9, 5000).astype(float) + 0.3 * y
    return {
        "constant_score_auroc": auroc_rank(y, s_const),
        "tied_score_rank_auroc": auroc_rank(y, s_tied),
        "tied_score_sklearn_auroc": float(roc_auc_score(y, s_tied)),
    }


def percentile_ci(x: np.ndarray, lo: float = 2.5, hi: float = 97.5) -> list[float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, lo)), float(np.percentile(x, hi))]


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (own implementation)."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


def save_json(obj, path: Path) -> None:
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=_default)


class WeightedAUROC:
    """Fast AUROC for many re-weightings of one fixed (y, score) set.

    Sorts once; ties are handled by grouping equal scores (a tied positive/negative pair
    counts 1/2), so the result equals the average-rank AUROC on the data replicated by
    integer weights (a bootstrap resample). O(n) per call."""

    def __init__(self, y: np.ndarray, s: np.ndarray):
        order = np.argsort(s, kind="mergesort")
        ss = np.asarray(s)[order]
        self.order = order
        self.y = np.asarray(y).astype(bool)[order]
        starts = np.r_[0, np.flatnonzero(ss[1:] != ss[:-1]) + 1]
        self.starts = starts

    def __call__(self, w: np.ndarray | None = None) -> float:
        if w is None:
            w = np.ones(len(self.y))
        else:
            w = np.asarray(w, dtype=float)[self.order]
        wp = np.where(self.y, w, 0.0)
        wn = w - wp
        gp = np.add.reduceat(wp, self.starts)
        gn = np.add.reduceat(wn, self.starts)
        before = np.cumsum(gn) - gn
        P, N = gp.sum(), gn.sum()
        if P == 0 or N == 0:
            return float("nan")
        return float((gp * (before + 0.5 * gn)).sum() / (P * N))
