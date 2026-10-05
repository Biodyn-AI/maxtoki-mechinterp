"""Shared helpers for the cell-cycle loop analysis: circular statistics, the embedding preprocessing,
and the cell-cycle phase label.

PHASE LABEL. The phase angle phi of each cell is computed from that cell's own expression of cell-cycle
marker genes (Tirosh et al. 2016 S-phase and G2/M lists below). Steps (phase_label):
  1. S score and G2M score = mean z-scored expression of the S genes / of the G2M genes.
     Discrete call: G1 if both scores < 0, else S if S score >= G2M score, else G2M.
  2. z-score all S + G2M marker genes, PCA to 2 components, phi = atan2(PC2, PC1).
  3. Orient phi so that it increases G1 -> S -> G2M, then rotate it so the G1 cells sit at phi = 0.
So phi is a function of marker-gene expression only. It is the target that every representation is scored
against. The phi stored in data/*.npz was made this way; code/phase_check.py recomputes it from
data/expr_k562.npz.
"""
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
DIM = 20

# Tirosh et al. 2016 cell-cycle marker lists.
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
    """Map angles into [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def circ_mean(a):
    a = np.asarray(a)
    return float(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def ang(u, v):
    """Unsigned angle between two vectors, in degrees (0-180)."""
    return float(np.degrees(np.arccos(np.clip(unit(u) @ unit(v), -1.0, 1.0))))


# ----------------------------------------------------------------------------- phase label
def _score(Xlog, gene_upper, gene_set):
    idx = [np.where(gene_upper == g)[0][0] for g in gene_set if g in set(gene_upper)]
    if not idx:
        return np.zeros(Xlog.shape[0]), 0
    sub = Xlog[:, idx]
    z = (sub - sub.mean(0, keepdims=True)) / (sub.std(0, keepdims=True) + 1e-8)
    return z.mean(1), len(idx)


def phase_angle(Xlog, gene_upper, cc_phase):
    """phi in [-pi, pi): PCA(2) of z-scored S + G2M marker expression -> atan2, then oriented and rotated."""
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


def phase_label(Xlog, genes):
    """Xlog: cells x genes, log(1 + counts per 10,000). Returns s_score, g2m_score, cc_phase, phi, n markers."""
    Xlog = np.asarray(Xlog, dtype=np.float64)
    gu = np.char.upper(np.asarray(genes).astype(str))
    s_score, ns = _score(Xlog, gu, S_GENES)
    g2m_score, ng = _score(Xlog, gu, G2M_GENES)
    cc_phase = np.where(np.maximum(s_score, g2m_score) < 0, "G1",
                        np.where(s_score >= g2m_score, "S", "G2M"))
    phi = phase_angle(Xlog, gu, cc_phase)
    return dict(s_score=s_score, g2m_score=g2m_score, cc_phase=cc_phase, phi=phi, n_s=ns, n_g2m=ng)


# ----------------------------------------------------------------------------- representation prep
def prep(X, d=DIM, seed=0):
    """Center -> PCA(d) -> standardise each component (whitening)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    Xc = X - X.mean(0)
    Xr = PCA(min(d, Xc.shape[1], Xc.shape[0] - 1), random_state=seed).fit_transform(Xc)
    return StandardScaler().fit_transform(Xr)


def emb_path(model):
    return os.path.join(DATA, f"{model}_k562.npz")
