"""Shared pieces for the cell-cycle decodability analysis: marker lists, cell selection, phase, PCA prep.

PHASE COORDINATE. Tirosh et al. 2016 S / G2M marker z-scores -> PCA(2) -> phi = atan2, then ORIENTED so phi
increases G1 -> S -> G2M and rotated to put G1 at phi = 0. Note: phi is (up to radial normalisation) a linear
projection of z-scored marker expression, so it structurally advantages linear probes, and most of all a probe
on expression itself (the markers are among its input genes).
"""
import os
import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PKG, "data")
OUTPUTS = os.path.join(PKG, "outputs")

N_CELLS, SEED = 3000, 42
DIM = 20
CONTROL_LABELS = ["non-targeting", "Non-targeting", "non_targeting"]

# Tirosh et al. 2016 cell-cycle markers.
S_GENES = ["MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6", "CDCA7", "DTL",
           "PRIM1", "UHRF1", "MLF1IP", "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76", "SLBP",
           "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51", "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN",
           "DSCC1", "BLM", "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8"]
G2M_GENES = ["HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2", "NUF2", "CKS1B",
             "MKI67", "TMPO", "CENPF", "TACC3", "FAM64A", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1",
             "KIF11", "ANP32E", "TUBB4B", "GTSE1", "KIF20B", "HJURP", "CDCA3", "HN1", "CDC20", "TTK", "CDC25C",
             "KIF2C", "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR", "AURKA",
             "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF", "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA"]


# ----------------------------------------------------------------------------- circular primitives
def wrap(a):
    """Map angles into (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def circ_mean(a):
    a = np.asarray(a)
    return float(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))


# ----------------------------------------------------------------------------- cells
def load_cells():
    """Per-row labels of the source dataset (643,413 rows): CRISPRi target gene and cell line."""
    z = np.load(os.path.join(DATA, "cells.npz"), allow_pickle=False)
    target = z["target_categories"][z["target_codes"]]
    line = z["line_categories"][z["line_codes"]]
    return target, line


def select_controls(n_cells=N_CELLS, seed=SEED):
    """Source rows of the non-targeting control cells used throughout (random draw, sorted)."""
    target, _ = load_cells()
    ctrl = np.where(np.isin(target, CONTROL_LABELS))[0]
    rs = np.random.RandomState(seed)
    return np.sort(rs.choice(ctrl, min(n_cells, len(ctrl)), replace=False))


def load_marker_rows(rows):
    """Stored log1p(CP10k) expression of the marker genes for the given source rows (in that order)."""
    z = np.load(os.path.join(DATA, "markers.npz"), allow_pickle=False)
    pos = {int(r): i for i, r in enumerate(z["rows"])}
    missing = [int(r) for r in rows if int(r) not in pos]
    assert not missing, f"{len(missing)} requested rows are not in markers.npz"
    X = z["X"][[pos[int(r)] for r in rows]].astype(np.float64)
    return X, np.char.upper(z["genes"].astype(str))


# ----------------------------------------------------------------------------- phase
def _score(Xlog, gene_upper, gene_set):
    idx = [np.where(gene_upper == g)[0][0] for g in gene_set if g in set(gene_upper)]
    if not idx:
        return np.zeros(Xlog.shape[0]), 0
    sub = Xlog[:, idx]
    z = (sub - sub.mean(0, keepdims=True)) / (sub.std(0, keepdims=True) + 1e-8)
    return z.mean(1), len(idx)


def phase_angle(Xlog, gene_upper, cc_phase):
    """phi in [-pi, pi): PCA(2) of z-scored S+G2M marker expression -> atan2, then ORIENTED and ROTATED.

    atan2 sets neither the handedness nor the origin, so we pin it by the discrete call: require the cyclic
    order G1 -> S -> G2M to be the direction of INCREASING phi, then rotate G1 to phi = 0.
    """
    idx = [np.where(gene_upper == g)[0][0] for g in (S_GENES + G2M_GENES) if g in set(gene_upper)]
    Z = Xlog[:, idx]
    Z = (Z - Z.mean(0, keepdims=True)) / (Z.std(0, keepdims=True) + 1e-8)
    from sklearn.decomposition import PCA
    P = PCA(2, random_state=0).fit_transform(Z - Z.mean(0))
    phi = np.arctan2(P[:, 1], P[:, 0])

    m = {p: circ_mean(phi[cc_phase == p]) for p in ("G1", "S", "G2M") if (cc_phase == p).sum() > 10}
    if {"G1", "S", "G2M"} <= set(m):
        fwd = wrap(m["S"] - m["G1"]) > 0 and wrap(m["G2M"] - m["S"]) > 0
        if not fwd:
            phi = -phi
            m = {p: circ_mean(phi[cc_phase == p]) for p in ("G1", "S", "G2M")}
        phi = wrap(phi - m["G1"])
    return phi


def phase_for_rows(rows):
    """Cell-cycle scores, discrete phase and phi for the given source rows, scored within that set of cells."""
    Xlog, gu = load_marker_rows(rows)
    s_score, ns = _score(Xlog, gu, S_GENES)
    g2m_score, ng = _score(Xlog, gu, G2M_GENES)
    cc_phase = np.where(np.maximum(s_score, g2m_score) < 0, "G1",
                        np.where(s_score >= g2m_score, "S", "G2M"))
    phi = phase_angle(Xlog, gu, cc_phase)
    return dict(phi=phi, cc_phase=cc_phase, s_score=s_score, g2m_score=g2m_score, n_s=ns, n_g2m=ng)


# ----------------------------------------------------------------------------- representation prep
def prep(X, d=DIM, seed=0):
    """center -> PCA(d) -> whiten."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    Xc = X - X.mean(0)
    Xr = PCA(min(d, Xc.shape[1], Xc.shape[0] - 1), random_state=seed).fit_transform(Xc)
    return StandardScaler().fit_transform(Xr)
