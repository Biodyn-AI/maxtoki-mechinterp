"""Recompute the phase label of every data file from the expression matrix and compare with the stored one.

data/expr_k562.npz holds log(1 + counts per 10,000) for all 6,546 genes of the 3,000 cells. The scGPT file has
the same 3,000 cells in the same order. The Geneformer file has 2,000 of these cells; its label was computed on
those 2,000 cells alone (the z-scores and the PCA use only them), so it is recomputed the same way here.

Run from the package root:  python code/phase_check.py
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from cc_common import phase_label, emb_path, wrap  # noqa: E402

ex = np.load(emb_path("expr"), allow_pickle=False)
X, genes, cells = ex["emb"], ex["genes"], ex["cell_idx"]
row = {c: i for i, c in enumerate(cells)}

for model in ("expr", "scgpt", "geneformer"):
    z = np.load(emb_path(model), allow_pickle=False)
    idx = np.array([row[c] for c in z["cell_idx"]])
    lab = phase_label(X[idx], genes)
    d_phi = np.abs(wrap(lab["phi"] - z["phi"]))
    print(f"{model:>10}: n = {len(idx)} | markers found S {lab['n_s']}, G2M {lab['n_g2m']} | "
          f"max |phi diff| = {d_phi.max():.2e} rad | "
          f"max |S score diff| = {np.abs(lab['s_score'] - z['s_score']).max():.2e} | "
          f"max |G2M score diff| = {np.abs(lab['g2m_score'] - z['g2m_score']).max():.2e} | "
          f"same discrete phase: {np.mean(lab['cc_phase'] == z['cc_phase']):.4f}")
    for p in ("G1", "S", "G2M"):
        m = z["cc_phase"] == p
        mu = np.degrees(np.arctan2(np.sin(z["phi"][m]).mean(), np.cos(z["phi"][m]).mean()))
        print(f"            {p:>3}: {m.sum():4d} cells, circular mean phi = {mu:+7.1f} deg")
