"""Linear circular decodability of cell-cycle phase from per-cell representations.

For each representation: load the per-cell matrix and its cell_idx, score the cell-cycle phase phi on those
cells from the marker genes, reduce the representation to 20 whitened PCs, and predict (cos phi, sin phi)
with 5-fold ridge regression. The score is the circular R^2:

    circ-R2 = 1 - mean(1 - cos(phi_pred - phi)) / mean(1 - cos(phi - circmean(phi)))

1 = perfect, 0 = no better than predicting the circular mean for every cell. A kNN regressor (k = 15) on the
same PCs is reported next to it.

Run (from the package root):  python code/decode_phase.py
Out:  outputs/decodability.json, outputs/run_log.txt
"""
import os, sys, json
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "3")
sys.dont_write_bytecode = True
from collections import Counter
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from cc_common import (DATA, OUTPUTS, DIM, S_GENES, G2M_GENES, circ_mean, phase_for_rows, prep,  # noqa: E402
                       select_controls)

SEED = 0

REPRESENTATIONS = {
    "expression":   dict(file="expression_controls.npz", cell_idx="source_row"),
    "state_se_L11": dict(file="state_se_L11.npz", cell_idx="input_row"),
}

LOG = []


def log(msg=""):
    print(msg, flush=True)
    LOG.append(msg)


def circ_r2(Xz, phi, model="linear", seed=SEED):
    """Circular R^2 for predicting phi from Xz. 1 - E[1-cos(err)] / E[1-cos(phi - circmean)]."""
    Y = np.column_stack([np.cos(phi), np.sin(phi)])
    P = np.zeros_like(Y)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(Xz):
        m = Ridge(alpha=10.0) if model == "linear" else KNeighborsRegressor(15)
        P[te] = m.fit(Xz[tr], Y[tr]).predict(Xz[te])
    err = 1.0 - np.cos(np.arctan2(P[:, 1], P[:, 0]) - phi)
    base = float(np.mean(1.0 - np.cos(phi - circ_mean(phi))))
    return float(1.0 - err.mean() / (base + 1e-12))


def load_representation(name):
    spec = REPRESENTATIONS[name]
    z = np.load(os.path.join(DATA, spec["file"]), allow_pickle=False)
    if "emb" in z.files:
        emb = z["emb"].astype(np.float64)
    else:
        emb = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])).toarray()
        emb = emb.astype(np.float64)
    rows = z["cell_idx"].astype(int)
    if spec["cell_idx"] == "input_row":
        rows = select_controls()[rows]
    assert len(rows) == emb.shape[0]
    return emb, rows


def run_one(name):
    emb, rows = load_representation(name)
    ph = phase_for_rows(rows)
    phi, cc_phase = ph["phi"], ph["cc_phase"]

    log(f"[{name}] {emb.shape[0]} cells x {emb.shape[1]}-d | S markers {ph['n_s']}/{len(S_GENES)}, "
        f"G2M {ph['n_g2m']}/{len(G2M_GENES)} | phase {dict(sorted(Counter(cc_phase.tolist()).items()))}")
    for p in ("G1", "S", "G2M"):
        m = cc_phase == p
        if m.sum():
            log(f"    {p:>4}: n={int(m.sum()):4d}  circ-mean phi = {np.degrees(circ_mean(phi[m])):+7.1f} deg")

    Xz = prep(emb, DIM)
    lin = circ_r2(Xz, phi, "linear")
    knn = circ_r2(Xz, phi, "knn")
    log(f"    linear circ-R2 = {lin:.3f} | kNN circ-R2 = {knn:.3f}   ({DIM} whitened PCs, 5-fold)")
    return dict(n_cells=int(emb.shape[0]), emb_dim=int(emb.shape[1]), pca_dim=DIM,
                n_s_markers=int(ph["n_s"]), n_g2m_markers=int(ph["n_g2m"]),
                phase_counts={k: int(v) for k, v in sorted(Counter(cc_phase.tolist()).items())},
                linear_circ_r2=lin, knn_circ_r2=knn)


def main():
    sel = select_controls()
    log(f"[cells] {len(sel)} non-targeting control cells selected (RandomState 42, sorted)")
    out = {}
    for name in REPRESENTATIONS:
        out[name] = run_one(name)
    e, s = out["expression"]["linear_circ_r2"], out["state_se_L11"]["linear_circ_r2"]
    log("")
    log(f"===== linear circ-R2: expression {e:.3f} | STATE-SE L11 {s:.3f} | STATE / expression = {s / e:.2f} =====")
    os.makedirs(OUTPUTS, exist_ok=True)
    json.dump(out, open(os.path.join(OUTPUTS, "decodability.json"), "w"), indent=1)
    with open(os.path.join(OUTPUTS, "run_log.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")
    log(f"[done] -> outputs/decodability.json, outputs/run_log.txt")


if __name__ == "__main__":
    main()
