"""v2 cross-model alignment (revision item D10) -- shared helpers.

This module copies the deployed metric code from
  runs/topology-141-217M/scripts/phase14_cross_model_cca.py:69-126
  runs/topology-141-217M/scripts/phase4_scgpt_cross_model.py:66-109
so that every re-analysis uses the SAME settings as the deployed run:
  * each model's static (input) gene-embedding rows, centred, PCA to 30 dims
    (sklearn PCA, random_state=42), fitted separately per model;
  * sklearn CCA(n_components=10, max_iter=200) on the two 30-dim tables,
    score = mean of the 10 in-sample canonical correlations;
  * gene-pair Pearson/Spearman on the FULL static rows (cosine matrices, upper triangle);
  * top-1 retrieval = orthogonal Procrustes MaxToki-PCA -> other-PCA, fitted and scored
    on the same genes, candidates = all genes in the panel.

Nothing here runs a model forward pass. Only embedding tables are read.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning

PROJ = Path("<REPO_ROOT>/projects/maxtoki")
RUN = PROJ / "runs/topology-141-217M"
OUT = RUN / "outputs/v2_crossmodel"
PREP = OUT / "embeddings_subset.npz"

GENEFORMER_DIR = Path(
    "<HF_CACHE>/hub/models--ctheodoris--Geneformer"
    "/snapshots/05fcbeb8a27d49e0a7a4349152202ee2c1cbfd28/Geneformer-V2-316M"
)
GENEFORMER_TOKEN_DICT = GENEFORMER_DIR.parent / "geneformer/token_dictionary_gc104M.pkl"
MAXTOKI_ST = PROJ / "setup/MaxToki-217M-HF/model.safetensors"
MAXTOKI_TOKDICT = PROJ / "setup/token_dictionary.json"
SCGPT_DIR = Path("<DATA_ROOT>/biodyn-work/"
                 "subproject_53_scgpt_gpl_replication/embeddings")
SCGPT_PT = SCGPT_DIR / "scgpt_whole_human_gene_embeddings.pt"
SCGPT_NPZ = SCGPT_DIR / "scgpt_gene_embeddings.npz"
SCGPT_CKPT = Path("<DATA_ROOT>/biodyn-work/single_cell_mechinterp/"
                  "external/scGPT_checkpoints/whole-human/best_model.pt")

DOMAINS = ["lung", "immune", "external_lung"]
DOMAIN_SOURCE = {
    "lung": "Tabula Sapiens lung (tabula_sapiens_lung.h5ad), 1,500 sampled cells",
    "immune": "Tabula Sapiens immune (tabula_sapiens_immune.h5ad), reused spectral-geometry phase0 cells",
    "external_lung": "Krasnow lung Smart-seq2 (krasnow_lung_smartsq2.h5ad), 1,500 sampled cells",
}
# model keys used in the prepared npz
MODELS = {
    "gf": "Geneformer V2-316M bert.embeddings.word_embeddings (static input table)",
    "scgpt_fixed": "scGPT whole-human encoder.embedding + enc_norm LayerNorm ('normed_embeddings'), "
                   "rows looked up by the vocab INDEX VALUE (correct)",
    "scgpt_deployed": "same scGPT table, rows looked up by dict POSITION as in phase4_scgpt_cross_model.py:119-121 "
                      "(deployed; wrong gene for almost every row)",
}
TOP_K_CCA = 10
PCA_DIM_FOR_ALIGN = 30


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def normalise(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def d_align_for(n: int) -> int:
    return min(PCA_DIM_FOR_ALIGN, max(2, n // 4))


def pca_reduce(X: np.ndarray, d: int) -> np.ndarray:
    """Deployed PCA step (phase14:169-172)."""
    return PCA(n_components=d, random_state=42).fit_transform(X - X.mean(0))


def cca_mean_r(A: np.ndarray, B: np.ndarray, k: int = TOP_K_CCA) -> tuple[float, list, int]:
    """Deployed CCA score (phase14:75-96). Returns (mean r, list of r, n_convergence_warnings)."""
    n = min(A.shape[0], B.shape[0])
    A = A[:n]; B = B[:n]
    k = min(k, A.shape[1], B.shape[1], n - 1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        cca = CCA(n_components=k, max_iter=200)
        try:
            Ac, Bc = cca.fit_transform(A, B)
        except Exception:
            return float("nan"), [], 0
        nconv = sum(1 for x in w if issubclass(x.category, ConvergenceWarning))
    rs = []
    for i in range(k):
        a, b = Ac[:, i], Bc[:, i]
        if a.std() == 0 or b.std() == 0:
            continue
        rs.append(float(np.corrcoef(a, b)[0, 1]))
    if not rs:
        return float("nan"), [], nconv
    return float(np.mean(rs)), rs, nconv


def procrustes_R(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    d = min(A.shape[1], B.shape[1])
    M = A[:, :d].T @ B[:, :d]
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt


def top1_from(A: np.ndarray, B: np.ndarray, R: np.ndarray, ids_q=None, ids_c=None) -> float:
    """Fraction of query rows of A whose nearest candidate row of B (cosine after
    rotating A by R) is the same gene. ids_q / ids_c give original gene ids so that a
    gene duplicated by the bootstrap counts as a match to any copy of itself."""
    d = R.shape[0]
    sim = normalise(A[:, :d].astype(np.float64) @ R) @ normalise(B[:, :d].astype(np.float64)).T
    j = np.argmax(sim, axis=1)
    if ids_q is None:
        ids_q = np.arange(A.shape[0]); ids_c = np.arange(B.shape[0])
    return float((ids_c[j] == ids_q).mean())


def top1_insample(A: np.ndarray, B: np.ndarray, ids=None) -> float:
    """Deployed top-1 (phase14:108-126): rotation fitted and scored on the same genes."""
    A = A.astype(np.float64); B = B.astype(np.float64)
    R = procrustes_R(A, B)
    return top1_from(A, B, R, ids, ids)


def cos_matrix(X: np.ndarray) -> np.ndarray:
    Xn = normalise(X.astype(np.float64))
    return Xn @ Xn.T


def offdiag_pearson(Sa: np.ndarray, Sb: np.ndarray, idx: np.ndarray | None = None) -> float:
    """Upper-triangle Pearson of two gene-gene cosine matrices. With a bootstrap index
    vector idx, pairs made of two copies of the same gene are dropped (their cosine is 1
    by construction and is not a gene pair)."""
    if idx is None:
        iu = np.triu_indices(Sa.shape[0], 1)
        return float(np.corrcoef(Sa[iu], Sb[iu])[0, 1])
    n = len(idx)
    iu = np.triu_indices(n, 1)
    gi = idx[iu[0]]; gj = idx[iu[1]]
    keep = gi != gj
    a = Sa[gi[keep], gj[keep]]; b = Sb[gi[keep], gj[keep]]
    return float(np.corrcoef(a, b)[0, 1])


def pct_ci(v, lo=2.5, hi=97.5):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, lo)), float(np.percentile(v, hi))]


def summarise_null(obs: float, null: np.ndarray) -> dict:
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    return {
        "n": int(len(null)),
        "mean": float(null.mean()),
        "sd": float(null.std(ddof=1)),
        "p95": float(np.percentile(null, 95)),
        "p99": float(np.percentile(null, 99)),
        "max": float(null.max()),
        "p_empirical_one_sided": float((1 + (null >= obs).sum()) / (1 + len(null))),
        "z": float((obs - null.mean()) / null.std(ddof=1)) if null.std(ddof=1) > 0 else float("nan"),
        "excess_over_mean": float(obs - null.mean()),
    }


def load_prepared():
    z = np.load(PREP, allow_pickle=False)
    return z


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)
