"""Build the two per-cell inputs for the selected control cells (reference; not runnable from the package).

Needs the full source dataset (SOURCE_H5AD: 643,413 cells x 6,546 genes, about 30 GB; not included) plus
anndata and h5py. Writes:
  data/expression_controls.npz            stored log1p(CP10k) expression of the selected cells (CSR) + cell_idx
  controls_for_state.h5ad            the same cells, for code/extraction/extract_state.py

The source file holds log1p(CP10k) values only (no raw counts), so that is also what STATE-SE receives.
"""
import os, sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, h5py, anndata as ad, pandas as pd
from scipy import sparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cc_common import DATA, select_controls  # noqa: E402

SOURCE_H5AD = os.environ.get("SOURCE_H5AD", "crispri_pooled.h5ad")
STATE_INPUT = os.environ.get("STATE_INPUT", "controls_for_state.h5ad")


def _dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in a])


sel = select_controls()
with h5py.File(SOURCE_H5AD, "r") as f:
    genes = _dec(f["var"]["gene_name_index"][:]).astype(str)
    X = np.stack([np.asarray(f["X"][int(i), :], dtype=np.float32) for i in sel])

A = sparse.csr_matrix(X)
np.savez_compressed(os.path.join(DATA, "expression_controls.npz"), data=A.data, indices=A.indices,
                    indptr=A.indptr, shape=np.array(A.shape), genes=genes, cell_idx=sel)

adata = ad.AnnData(X=A, obs=pd.DataFrame(index=[str(i) for i in sel]),
                   var=pd.DataFrame(index=pd.Index(genes, name="index")))
adata.write_h5ad(STATE_INPUT)
print(f"wrote expression_controls.npz {A.shape} and {STATE_INPUT} {adata.shape}")
