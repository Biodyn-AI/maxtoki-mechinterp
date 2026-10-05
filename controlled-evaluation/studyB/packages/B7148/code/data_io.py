"""Load the expression matrices in data/ (compressed sparse rows + one gene name per line)."""
from __future__ import annotations
import os
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class Cells:
    X: sp.csr_matrix          # cells x genes, log1p of per-cell normalised counts
    var_names: np.ndarray     # gene symbols, one per column of X

    @property
    def n_obs(self) -> int:
        return self.X.shape[0]

    @property
    def n_vars(self) -> int:
        return self.X.shape[1]


def load_cells(data_dir: str, name: str) -> Cells:
    z = np.load(os.path.join(data_dir, f"{name}_expression.npz"))
    X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(int(s) for s in z["shape"]))
    with open(os.path.join(data_dir, f"{name}_genes.txt")) as f:
        genes = np.array(f.read().splitlines())
    assert len(genes) == X.shape[1]
    return Cells(X=X, var_names=genes)
