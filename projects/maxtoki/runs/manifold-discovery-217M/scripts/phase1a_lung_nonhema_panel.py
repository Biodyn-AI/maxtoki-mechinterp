"""Phase 1A — build a proper lung_nonhema negative control panel.

The original `lung_control` panel was constructed by filtering Tabula Sapiens lung
cells to those whose `cell_type` mapped to a hematopoietic stage. That selected
*lung-resident immune cells* (lymphocytes, macrophages, etc.) — which still belong
to the H65 manifold. Phase 7 confirmed: that "negative control" passed all gates,
which is biologically correct (lung-resident immune cells are H65-shaped) but
methodologically an invalid negative control.

This script builds the actual non-hematopoietic negative control: TS lung cells
where `cell_type` does NOT map to any hematopoietic stage (e.g. epithelial,
endothelial, stromal, fibroblast, club cell, type II pneumocyte). For these
cells we *assign* synthetic stage labels uniformly drawn from the H65 stage
DAG so the same H65 ruler can be applied — but the ruler is meaningless for
non-hematopoietic cells, so the manifold should fail trustworthiness +
branch-holdout when projected through the frozen Phase-5 head.

Outputs (under runs/manifold-discovery-217M/outputs/phase1/):
  cells_lung_nonhema.npz, cells_lung_nonhema_obs.csv, anchors_lung_nonhema.csv
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "setup"))
from maxtoki_adapter import MaxTokiTokenizer

DATA_RAW = Path("<DATA_ROOT>/biodyn-work/single_cell_mechinterp/data/raw")
TS_LUNG = DATA_RAW / "tabula_sapiens_lung.h5ad"
OUT = RUN / "outputs/phase1"
OUT.mkdir(parents=True, exist_ok=True)

STAGE_DAG = json.loads((RUN / "planning/h65_stage_dag.json").read_text())
CT_TO_STAGE: dict[str, str | None] = {
    k: v for k, v in STAGE_DAG["cell_type_to_stage"].items() if not k.startswith("$$")
}
STAGE_NODES = list(STAGE_DAG["stage_to_branch"].keys())

MIN_CELLS_PER_ANCHOR = 5
MAX_CELLS_PER_ANCHOR = 30  # smaller than internal — this is just a control
TARGET_ANCHORS = 50

SEED = 42
rng = np.random.default_rng(SEED)


def is_hema(row) -> bool:
    fa = str(row.get("free_annotation", "") or "").strip().lower()
    ct = str(row.get("cell_type", "") or "").strip().lower()
    if CT_TO_STAGE.get(fa) is not None: return True
    if CT_TO_STAGE.get(ct) is not None: return True
    return False


def main():
    print("=" * 70)
    print("PHASE 1A — lung_nonhema panel construction (proper negative control)")
    print("=" * 70)
    t_phase = time.time()

    print("\n[1] Loading TS lung...")
    a = ad.read_h5ad(str(TS_LUNG), backed="r")
    print(f"  shape: {a.shape}")

    obs = a.obs.reset_index(drop=False).rename(columns={"index": "obs_label"})
    obs["__pos"] = np.arange(len(obs), dtype=np.int64)
    obs["is_hema"] = obs.apply(is_hema, axis=1)
    nonhema = obs[~obs["is_hema"]].reset_index(drop=True)
    print(f"  non-hematopoietic cells: {len(nonhema)} (cell types: {nonhema['cell_type'].nunique()})")

    # Build anchor groups by (donor × tissue × cell_type). Assign random H65 stages
    # uniformly so the ruler can be computed; the result will be a stage label that
    # is uncorrelated with the actual cell biology — that is the point.
    g = nonhema.groupby(["donor_id", "tissue", "cell_type"], observed=True)
    sizes = g.size().rename("n").reset_index()
    sizes = sizes[sizes["n"] >= MIN_CELLS_PER_ANCHOR].reset_index(drop=True)
    print(f"  viable groups: {len(sizes)}")

    # Sort by cell count desc; pick top TARGET_ANCHORS
    sizes = sizes.sort_values("n", ascending=False).head(TARGET_ANCHORS).reset_index(drop=True)
    sizes["anchor_id"] = [f"lung_nonhema_{i:04d}" for i in range(len(sizes))]
    sizes["hema_stage"] = rng.choice(STAGE_NODES, size=len(sizes))

    print(f"\n[2] Selected {len(sizes)} non-hematopoietic anchors. Sample:")
    print(sizes.head(10)[["anchor_id", "donor_id", "tissue", "cell_type", "n", "hema_stage"]].to_string(index=False))

    # Gather cells per anchor (capped)
    rows = []
    for _, ar in sizes.iterrows():
        key = (ar["donor_id"], ar["tissue"], ar["cell_type"])
        try:
            grp = g.get_group(key)
        except KeyError:
            continue
        positions = grp["__pos"].to_numpy()
        if len(positions) > MAX_CELLS_PER_ANCHOR:
            positions = rng.choice(positions, size=MAX_CELLS_PER_ANCHOR, replace=False)
        labels = grp.set_index("__pos").loc[positions, "obs_label"].to_numpy()
        for pos, lab in zip(positions, labels):
            rows.append({
                "anchor_id": ar["anchor_id"],
                "obs_label": str(lab),
                "cell_idx": int(pos),
                "donor_id": ar["donor_id"],
                "tissue": ar["tissue"],
                "cell_type": ar["cell_type"],
                "hema_stage": ar["hema_stage"],
            })
    cells_df = pd.DataFrame(rows)
    print(f"\n[3] Gathered {len(cells_df)} cells across {len(sizes)} anchors.")

    # Tokenize
    print("\n[4] Tokenizing cells via MaxTokiTokenizer (chunks of 1000)...")
    tokenizer = MaxTokiTokenizer()
    var_ens = a.var_names.to_numpy().astype(str)
    var_ens_clean = np.array([e.split(".")[0] for e in var_ens])
    var_indices_full, var_tokens_full, var_medians_full = tokenizer.make_var_mapping(
        var_ens_clean.tolist()
    )
    L_cap = tokenizer.model_input_size
    n_cells = len(cells_df)
    token_ids = np.full((n_cells, L_cap), tokenizer.PAD, dtype=np.int32)
    gene_positions = np.full((n_cells, L_cap), -1, dtype=np.int32)
    attn_mask = np.zeros((n_cells, L_cap), dtype=np.int8)
    seq_lens = np.zeros(n_cells, dtype=np.int32)
    cell_positions = cells_df["cell_idx"].to_numpy()
    chunk = 1000
    import scipy.sparse as sp
    for s in range(0, n_cells, chunk):
        e = min(s + chunk, n_cells)
        sub = a[cell_positions[s:e]].X
        if sp.issparse(sub):
            sub = sub.toarray()
        sub = np.asarray(sub, dtype=np.float32)
        for j, expr_row in enumerate(sub):
            cell = tokenizer.tokenize_cell(
                expr_row, var_indices_full, var_tokens_full, var_medians_full, max_len=L_cap
            )
            i = s + j
            if cell is None:
                token_ids[i, 0] = tokenizer.BOS
                token_ids[i, 1] = tokenizer.EOS
                attn_mask[i, :2] = 1
                seq_lens[i] = 2
                continue
            L = len(cell.token_ids)
            token_ids[i, :L] = cell.token_ids
            gene_positions[i, :L] = cell.gene_positions
            attn_mask[i, :L] = 1
            seq_lens[i] = L
        print(f"    chunk {e}/{n_cells} done")

    np.savez_compressed(
        OUT / "cells_lung_nonhema.npz",
        token_ids=token_ids, gene_positions=gene_positions,
        attn_mask=attn_mask, seq_lens=seq_lens,
    )
    cells_df.to_csv(OUT / "cells_lung_nonhema_obs.csv", index=False)
    sizes.to_csv(OUT / "anchors_lung_nonhema.csv", index=False)
    print(f"\nDone in {time.time()-t_phase:.1f}s. Tokens at {OUT/'cells_lung_nonhema.npz'}.")


if __name__ == "__main__":
    main()
