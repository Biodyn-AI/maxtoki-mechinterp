"""Per-dataset loader for the attention-GRN pipeline.

Abstracts differences between the Replogle concat (K562/HEPG2/jurkat/RPE1)
and the standalone ReplogleWeissman2022_rpe1 file so Phases 0-4 can be run
on either without code duplication.

Selected via env var DATASET ∈ {"k562", "rpe1"}.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

BIOM_ROOT = Path("<DATA_ROOT>/biodyn-nmi-paper")
SYM2ENS_PKL = BIOM_ROOT / "src/02_cssi_method/crispri_validation/data/gene_name_id_dict_gc104M.pkl"

K562_H5 = BIOM_ROOT / "src/02_cssi_method/crispri_validation/data/replogle_concat.h5ad"
RPE1_H5 = BIOM_ROOT / "results/round13_non_k562_replication/data/ReplogleWeissman2022_rpe1.h5ad"
ADAMSON_H5 = Path(
    "<DATA_ROOT>/biodyn-work/single_cell_mechinterp/"
    "data/perturb/adamson/perturb_processed_symbols.h5ad"
)


@dataclass
class DatasetHandle:
    """Abstracts dataset-specific metadata.

    Attributes:
        name: short identifier ("k562" / "rpe1")
        h5_path: path to the h5ad file
        n_cells_total, n_genes_total
        var_symbols: list[str] — gene symbol per var column (best effort, may be Ensembl if no symbol)
        var_ensembl: list[str|None] — Ensembl ID per var column
        cell_of_interest_mask: bool array over all cells (True = in the cell line of interest)
        perturbation_codes_for_coi: int32 array, only for cells in cell_of_interest_mask
        perturbation_categories: list[str] — parallel to codes
        control_category_codes: set[int] — codes that identify non-targeting / control
    """
    name: str
    h5_path: Path
    n_cells_total: int
    n_genes_total: int
    var_symbols: list
    var_ensembl: list
    cell_of_interest_mask: np.ndarray  # bool, length n_cells_total
    perturbation_codes: np.ndarray     # int32, length n_cells_total
    perturbation_categories: list
    control_category_codes: set


def load_hvg_matrix(ds: "DatasetHandle", var_idx: np.ndarray, progress: bool = True) -> np.ndarray:
    """Dataset-backend-agnostic loader for the full HVG expression matrix.

    Handles both dense-h5py and CSR-sparse h5ad storage. Returns a dense
    (n_cells, n_hvg) float32 array.
    """
    import anndata as ad
    import scipy.sparse as sp
    adata = ad.read_h5ad(str(ds.h5_path), backed="r")
    try:
        X = adata.X
        n_cells = ds.n_cells_total
        n_hvg = len(var_idx)
        out = np.empty((n_cells, n_hvg), dtype=np.float32)
        chunk = 10000
        for i in range(0, n_cells, chunk):
            j = min(i + chunk, n_cells)
            X_sub = X[i:j]
            if sp.issparse(X_sub) or hasattr(X_sub, "toarray"):
                arr = X_sub.toarray()
            else:
                arr = np.asarray(X_sub)
            out[i:j] = arr[:, var_idx].astype(np.float32)
    finally:
        adata.file.close()
    return out


def resolve(name: str) -> DatasetHandle:
    name = name.lower()
    if name == "k562":
        return _load_k562_concat()
    if name == "rpe1":
        return _load_rpe1()
    if name == "adamson":
        return _load_adamson()
    raise ValueError(f"unknown dataset: {name!r}")


def _load_adamson() -> DatasetHandle:
    """Adamson 2016 K562 CRISPRa perturb-seq.

    Stored as CSR sparse. Condition labels have form ``GENE+ctrl`` (or
    ``ctrl`` for the non-targeting category). We strip ``+ctrl`` to recover
    the perturbed gene symbol.
    """
    with h5py.File(ADAMSON_H5, "r") as f:
        shape = f["X"].attrs["shape"]
        n_cells, n_genes = int(shape[0]), int(shape[1])
        var_ensembl = [
            s.decode() if isinstance(s, bytes) else s
            for s in f["var"]["ensembl_id"][:]
        ]
        var_symbols = [
            s.decode() if isinstance(s, bytes) else s
            for s in f["var"]["gene_name"][:]
        ]
        cell_of_interest_mask = np.ones(n_cells, dtype=bool)  # all K562
        pg_cats_raw = [
            s.decode() if isinstance(s, bytes) else s
            for s in f["obs"]["condition"]["categories"][:]
        ]
        pg_codes = f["obs"]["condition"]["codes"][:]

    # Normalise the condition labels:  "GENE+ctrl" -> "GENE", "ctrl" -> "control"
    pg_cats = []
    for c in pg_cats_raw:
        c2 = c
        if c2.endswith("+ctrl"):
            c2 = c2[:-5]
        if c2.lower() == "ctrl":
            c2 = "control"
        pg_cats.append(c2)

    control_codes = {
        i for i, g in enumerate(pg_cats_raw)
        if g.lower() in ("ctrl", "control")
    }

    return DatasetHandle(
        name="adamson",
        h5_path=ADAMSON_H5,
        n_cells_total=n_cells,
        n_genes_total=n_genes,
        var_symbols=var_symbols,
        var_ensembl=var_ensembl,
        cell_of_interest_mask=cell_of_interest_mask,
        perturbation_codes=pg_codes,
        perturbation_categories=pg_cats,
        control_category_codes=control_codes,
    )


def _load_k562_concat() -> DatasetHandle:
    with h5py.File(K562_H5, "r") as f:
        n_cells, n_genes = f["X"].shape
        var_symbols = [
            s.decode() if isinstance(s, bytes) else s
            for s in f["var"]["gene_name_index"][:]
        ]
        cl_cats = [
            (s.decode() if isinstance(s, bytes) else s).lower()
            for s in f["obs"]["cell_line"]["categories"][:]
        ]
        cl_codes = f["obs"]["cell_line"]["codes"][:]
        k562_mask = (cl_codes == cl_cats.index("k562"))

        pg_cats_b = f["obs"]["gene"]["categories"][:]
        pg_cats = [s.decode() if isinstance(s, bytes) else s for s in pg_cats_b]
        pg_codes = f["obs"]["gene"]["codes"][:]

    # symbol -> ensembl via Geneformer dict
    with open(SYM2ENS_PKL, "rb") as f:
        sym2ens = pickle.load(f)
    var_ensembl = [sym2ens.get(s) for s in var_symbols]

    control_codes = {
        i for i, g in enumerate(pg_cats)
        if "non-targeting" in g.lower() or "nontargeting" in g.lower()
    }

    return DatasetHandle(
        name="k562",
        h5_path=K562_H5,
        n_cells_total=int(n_cells),
        n_genes_total=int(n_genes),
        var_symbols=var_symbols,
        var_ensembl=var_ensembl,
        cell_of_interest_mask=k562_mask,
        perturbation_codes=pg_codes,
        perturbation_categories=pg_cats,
        control_category_codes=control_codes,
    )


def _load_rpe1() -> DatasetHandle:
    with h5py.File(RPE1_H5, "r") as f:
        n_cells, n_genes = f["X"].shape
        # var has both ensembl_id and gene_name columns directly
        var_ensembl = [
            s.decode() if isinstance(s, bytes) else s
            for s in f["var"]["ensembl_id"][:]
        ]
        var_symbols = [
            s.decode() if isinstance(s, bytes) else s
            for s in f["var"]["gene_name"][:]
        ]
        # No cell_line column; all cells are RPE1
        cell_of_interest_mask = np.ones(n_cells, dtype=bool)

        pg_cats_b = f["obs"]["perturbation"]["categories"][:]
        pg_cats = [s.decode() if isinstance(s, bytes) else s for s in pg_cats_b]
        pg_codes = f["obs"]["perturbation"]["codes"][:]

    # "control" is the label used in this file
    control_codes = {
        i for i, g in enumerate(pg_cats)
        if g.lower() == "control"
        or "non-targeting" in g.lower()
        or "nontargeting" in g.lower()
    }

    return DatasetHandle(
        name="rpe1",
        h5_path=RPE1_H5,
        n_cells_total=int(n_cells),
        n_genes_total=int(n_genes),
        var_symbols=var_symbols,
        var_ensembl=var_ensembl,
        cell_of_interest_mask=cell_of_interest_mask,
        perturbation_codes=pg_codes,
        perturbation_categories=pg_cats,
        control_category_codes=control_codes,
    )


def perturbation_ensembl_for_code(ds: DatasetHandle, code: int, sym2ens: dict) -> str | None:
    """Resolve the perturbation category at `code` to an Ensembl ID.

    For RPE1 the perturbation label is already a gene symbol (same semantics as
    K562). Use the sym2ens dict for both datasets to get the Ensembl ID.
    """
    cat = ds.perturbation_categories[code]
    if cat.lower() == "control":
        return None
    if "non-targeting" in cat.lower() or "nontargeting" in cat.lower():
        return None
    return sym2ens.get(cat)
