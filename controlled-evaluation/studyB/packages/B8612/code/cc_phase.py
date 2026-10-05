"""cc_phase - continuous cell-cycle phase angle per cell.

The cell cycle is a circle: G1 -> S -> G2M -> G1. To place cells on that circle we use the tricycle/Revelio
construction: PCA on cell-cycle marker genes only; the leading two PCs span an ellipse and the angle around it
is the continuous cell-cycle position.

  theta(cell) = atan2(PC2, PC1)   over the cell-cycle-gene expression submatrix

Check (in expression space): the S-score and G2M-score should peak roughly a quarter turn apart, the
signature of a genuine cycle.
"""
from __future__ import annotations
import numpy as np

# Tirosh/Scanpy canonical sets (the standard ones; intersected with the panel at runtime)
S_GENES = ["MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6", "CDCA7", "DTL",
           "PRIM1", "UHRF1", "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2",
           "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51", "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1",
           "BLM", "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8"]
G2M_GENES = ["HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2", "NUF2", "CKS1B",
             "MKI67", "TMPO", "CENPF", "TACC3", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11",
             "ANP32E", "TUBB4B", "GTSE1", "KIF20B", "HJURP", "CDCA3", "CDC20", "TTK", "CDC25C", "KIF2C",
             "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1",
             "ANLN", "LBR", "CKAP5", "CENPE", "CTCF", "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA"]


def _dense(X):
    import scipy.sparse as sp
    return X.toarray() if sp.issparse(X) else np.asarray(X)


def score_genes(adata, genes, n_bins=25, n_ctrl=50, seed=0):
    """Scanpy-style module score: mean(set) - mean(expression-binned control set)."""
    rng = np.random.default_rng(seed)
    var = np.char.upper(np.asarray(adata.var_names).astype(str))
    pos = {s: i for i, s in enumerate(var)}
    idx = np.array([pos[g] for g in genes if g in pos])
    if len(idx) == 0:
        return np.zeros(adata.n_obs), idx
    X = _dense(adata.X)
    mean_all = X.mean(0)
    order = np.argsort(mean_all)
    ranks = np.empty(len(mean_all), int); ranks[order] = np.arange(len(mean_all))
    bins = (ranks / len(ranks) * n_bins).astype(int)
    ctrl = []
    for b in np.unique(bins[idx]):
        pool = np.where(bins == b)[0]
        k = int((bins[idx] == b).sum()) * n_ctrl
        ctrl.append(rng.choice(pool, min(k, len(pool)), replace=False))
    ctrl = np.concatenate(ctrl) if ctrl else np.array([], int)
    return X[:, idx].mean(1) - (X[:, ctrl].mean(1) if len(ctrl) else 0.0), idx


def phase_angle(adata, seed=0):
    """Continuous cell-cycle position: angle in the PC1-PC2 plane of cell-cycle-gene expression.
    Returns (theta in [0,2pi), s_score, g2m_score, info)."""
    var = np.char.upper(np.asarray(adata.var_names).astype(str))
    pos = {s: i for i, s in enumerate(var)}
    cc = [g for g in (S_GENES + G2M_GENES) if g in pos]
    idx = np.array([pos[g] for g in cc])
    X = _dense(adata.X)[:, idx].astype(np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    U, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    P = U[:, :2] * S[:2]
    theta = np.mod(np.arctan2(P[:, 1], P[:, 0]), 2 * np.pi)
    s_score, _ = score_genes(adata, S_GENES, seed=seed)
    g2m_score, _ = score_genes(adata, G2M_GENES, seed=seed)
    var_ratio = float((S[:2] ** 2).sum() / (S ** 2).sum())
    return theta, s_score, g2m_score, dict(n_cc_genes=len(cc), pc12_var_ratio=var_ratio)


def circ_mean(a):
    return float(np.mod(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a))), 2 * np.pi))


def line_checks(theta, s_score, g2m_score, top=200):
    """Phase concentration R (1 = all cells at one angle, 0 = spread around the circle) and the angles at
    which the S and G2M scores peak (circular mean of theta over the `top` highest-scoring cells)."""
    out = dict(phase_R=round(float(np.abs(np.mean(np.exp(1j * theta)))), 3))
    out["s_peak_deg"] = round(np.rad2deg(circ_mean(theta[np.argsort(-s_score)[:top]])), 1)
    out["g2m_peak_deg"] = round(np.rad2deg(circ_mean(theta[np.argsort(-g2m_score)[:top]])), 1)
    d = abs(out["s_peak_deg"] - out["g2m_peak_deg"]) % 360
    out["s_g2m_separation_deg"] = round(min(d, 360 - d), 1)
    return out
