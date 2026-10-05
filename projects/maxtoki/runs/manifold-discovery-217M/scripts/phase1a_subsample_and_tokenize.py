"""Phase 1A — derive hematopoietic stages, subsample anchor panels, tokenize.

Reads Tabula Sapiens immune (and lung for negative control), derives a hematopoietic
stage label per cell from a curated cell_type → stage mapping, builds four anchor
panels (internal H65, strict non-overlap external H65, multi-donor zero-shot, lung
negative control), tokenizes via MaxTokiTokenizer, and persists per-cell token
arrays to disk so that Phase 1B can run MaxToki forward passes.

Outputs (under runs/manifold-discovery-217M/outputs/phase1/):
  cells_internal.npz          tokens, gene_positions, attn_mask, obs_table for internal panel
  cells_external.npz          same, external strict non-overlap panel
  cells_zeroshot.npz          same, multi-donor zero-shot panel
  cells_lung_control.npz      same, lung negative-control panel
  panel_summary.json          counts + composition + donor non-overlap verification

Design choices: see ../planning/research_plan.md and ../planning/h65_stage_dag.json.
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
TS_IMMUNE = DATA_RAW / "tabula_sapiens_immune.h5ad"
TS_LUNG = DATA_RAW / "tabula_sapiens_lung.h5ad"

OUT = RUN / "outputs/phase1"
OUT.mkdir(parents=True, exist_ok=True)

STAGE_DAG = json.loads((RUN / "planning/h65_stage_dag.json").read_text())
CT_TO_STAGE: dict[str, str | None] = {
    k: v for k, v in STAGE_DAG["cell_type_to_stage"].items() if not k.startswith("$$")
}

# Per-anchor cell threshold and per-panel target sizes.
# Source paper: 298 internal, 616 external, 165 zero-shot. We target the same order.
MIN_CELLS_PER_ANCHOR = 5
MAX_CELLS_PER_ANCHOR = 50  # cap so a few huge groups don't dominate centroids
TARGET_INTERNAL_ANCHORS = 290
TARGET_EXTERNAL_ANCHORS = 600
TARGET_ZEROSHOT_ANCHORS = 160
TARGET_LUNG_CONTROL_ANCHORS = 200

SEED = 42
rng = np.random.default_rng(SEED)


def derive_stage(row) -> str | None:
    fa = str(row.get("free_annotation", "") or "").strip().lower()
    ct = str(row.get("cell_type", "") or "").strip().lower()
    return CT_TO_STAGE.get(fa) or CT_TO_STAGE.get(ct)


def build_anchor_table(adata, panel_label: str) -> pd.DataFrame:
    obs = adata.obs.copy()
    obs["hema_stage"] = obs.apply(derive_stage, axis=1)
    obs["__cell_idx"] = np.arange(len(obs))
    obs = obs[obs["hema_stage"].notna()].copy()
    if len(obs) == 0:
        print(f"  [{panel_label}] no cells with valid hematopoietic stage")
        return obs
    print(f"  [{panel_label}] cells with valid hematopoietic stage: {len(obs)}")
    g = obs.groupby(["donor_id", "tissue", "cell_type", "hema_stage"], observed=True)
    sizes = g.size().rename("n").reset_index()
    sizes = sizes[sizes["n"] >= MIN_CELLS_PER_ANCHOR].reset_index(drop=True)
    sizes["anchor_id"] = [f"{panel_label}_{i:04d}" for i in range(len(sizes))]
    print(f"  [{panel_label}] viable (donor × tissue × cell_type × stage) anchors: {len(sizes)}")
    return sizes


def split_donors(donors: list[str], frac_internal: float = 0.45) -> tuple[list[str], list[str]]:
    """Deterministic donor split. Internal panel gets `frac_internal` of donors; external gets rest."""
    sd = sorted(set(donors))
    n_int = max(1, int(round(len(sd) * frac_internal)))
    rng_local = np.random.default_rng(SEED)
    perm = rng_local.permutation(len(sd))
    int_idx = set(perm[:n_int].tolist())
    int_donors = [d for i, d in enumerate(sd) if i in int_idx]
    ext_donors = [d for d in sd if d not in int_donors]
    return int_donors, ext_donors


def select_panel(
    anchors: pd.DataFrame,
    donors_keep: list[str],
    target_n_anchors: int,
    panel_label: str,
) -> pd.DataFrame:
    sub = anchors[anchors["donor_id"].isin(donors_keep)].copy()
    if len(sub) > target_n_anchors:
        # Keep largest anchors first (by cell count), then sample to hit target.
        sub = sub.sort_values("n", ascending=False).reset_index(drop=True)
        sub = sub.iloc[:target_n_anchors].reset_index(drop=True)
    print(f"  [{panel_label}] selected anchors: {len(sub)} (cells: {int(sub['n'].sum())})")
    return sub


def gather_cells(
    adata, obs_full: pd.DataFrame, anchor_df: pd.DataFrame, panel_label: str
) -> pd.DataFrame:
    """Return per-cell rows attached to selected anchors, capped at MAX_CELLS_PER_ANCHOR.

    Tracks integer position-in-adata as `cell_idx` from the start (positional, not by
    pandas index label, since adata.obs may have non-integer string IDs).
    """
    obs_with_stage = obs_full.reset_index(drop=False).rename(columns={"index": "obs_label"})
    obs_with_stage["__pos"] = np.arange(len(obs_with_stage), dtype=np.int64)
    obs_with_stage["hema_stage"] = obs_with_stage.apply(derive_stage, axis=1)
    obs_with_stage = obs_with_stage[obs_with_stage["hema_stage"].notna()].reset_index(drop=True)
    g = obs_with_stage.groupby(
        ["donor_id", "tissue", "cell_type", "hema_stage"], observed=True
    )
    rows = []
    for _, ar in anchor_df.iterrows():
        key = (ar["donor_id"], ar["tissue"], ar["cell_type"], ar["hema_stage"])
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
    df = pd.DataFrame(rows)
    print(f"  [{panel_label}] cells gathered: {len(df)}")
    return df


def tokenize_cells(adata, cells_df: pd.DataFrame, tokenizer: MaxTokiTokenizer):
    """Tokenize using the adapter's tokenize_cell. Loads expression rows in chunks.

    Returns dict of padded int32 arrays:
      token_ids       (n_cells, L_cap)
      gene_positions  (n_cells, L_cap) — position into MaxToki vocab (var_idx); -1 for BOS/EOS/PAD
      attn_mask       (n_cells, L_cap)
      seq_lens        (n_cells,)
    """
    import scipy.sparse as sp

    var_ens = adata.var_names.to_numpy().astype(str)
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
    t0 = time.time()
    for s in range(0, n_cells, chunk):
        e = min(s + chunk, n_cells)
        rows_idx = cell_positions[s:e]
        # Pull a slice of the backed h5ad — anndata.AnnData[idx] returns a view; .X
        # then materialises the sub-matrix.
        sub = adata[rows_idx].X
        if sp.issparse(sub):
            sub = sub.toarray()
        sub = np.asarray(sub, dtype=np.float32)
        for j, expr_row in enumerate(sub):
            cell = tokenizer.tokenize_cell(
                expr_row, var_indices_full, var_tokens_full, var_medians_full, max_len=L_cap
            )
            i = s + j
            if cell is None:
                # No expressed genes; record bos/eos only
                token_ids[i, 0] = tokenizer.BOS
                token_ids[i, 1] = tokenizer.EOS
                attn_mask[i, :2] = 1
                seq_lens[i] = 2
                continue
            L = len(cell.token_ids)
            token_ids[i, :L] = cell.token_ids
            # gene_positions in the adapter are positions into the kept-HVG-set
            # (i.e., index into var_indices_full); store directly.
            gene_positions[i, :L] = cell.gene_positions
            attn_mask[i, :L] = 1
            seq_lens[i] = L
        elapsed = time.time() - t0
        eta = elapsed / e * (n_cells - e)
        print(f"    tokenized {e}/{n_cells}  elapsed {elapsed:.1f}s  eta {eta:.1f}s")
    return {
        "token_ids": token_ids,
        "gene_positions": gene_positions,
        "attn_mask": attn_mask,
        "seq_lens": seq_lens,
    }


def main():
    print("=" * 70)
    print("PHASE 1A — derive stages, subsample, tokenize")
    print("=" * 70)
    t_phase = time.time()

    print("\n[1] Loading Tabula Sapiens immune (backed='r')...")
    t0 = time.time()
    a_imm = ad.read_h5ad(str(TS_IMMUNE), backed="r")
    print(f"  immune adata: {a_imm.shape}  ({time.time()-t0:.1f}s)")

    print("\n[2] Building all viable hematopoietic anchors over the full TS immune...")
    all_anchors = build_anchor_table(a_imm, "ts_immune_all")
    if len(all_anchors) == 0:
        raise RuntimeError("No viable hematopoietic anchors derivable. Check stage mapping.")

    print("\n[3] Splitting donors → internal vs external strict non-overlap...")
    int_donors, ext_donors = split_donors(all_anchors["donor_id"].astype(str).unique().tolist())
    print(f"  internal donors: {len(int_donors)}  external donors: {len(ext_donors)}")

    panels = {}
    panels["internal"] = select_panel(all_anchors, int_donors, TARGET_INTERNAL_ANCHORS, "internal")
    panels["external"] = select_panel(all_anchors, ext_donors, TARGET_EXTERNAL_ANCHORS, "external")
    # Multi-donor zero-shot uses a slice of external donors but a different anchor sample
    rng_zs = np.random.default_rng(SEED + 1)
    zs_donors = sorted(ext_donors)
    panels["zeroshot"] = select_panel(
        all_anchors[~all_anchors["anchor_id"].isin(panels["external"]["anchor_id"])],
        zs_donors,
        TARGET_ZEROSHOT_ANCHORS,
        "zeroshot",
    )

    print("\n[4] Loading TS lung for negative control...")
    a_lung = ad.read_h5ad(str(TS_LUNG), backed="r")
    print(f"  lung adata: {a_lung.shape}")
    lung_anchors_all = build_anchor_table(a_lung, "ts_lung_all")
    lung_donors = sorted(lung_anchors_all["donor_id"].astype(str).unique().tolist())
    panels["lung_control"] = select_panel(
        lung_anchors_all, lung_donors, TARGET_LUNG_CONTROL_ANCHORS, "lung_control"
    )

    print("\n[5] Verifying donor non-overlap (internal vs external + zeroshot)...")
    int_d = set(panels["internal"]["donor_id"].astype(str))
    ext_d = set(panels["external"]["donor_id"].astype(str))
    zs_d = set(panels["zeroshot"]["donor_id"].astype(str))
    if int_d & ext_d:
        raise RuntimeError(f"Donor overlap internal ∩ external: {int_d & ext_d}")
    if int_d & zs_d:
        raise RuntimeError(f"Donor overlap internal ∩ zeroshot: {int_d & zs_d}")
    print(f"  ok. |int|={len(int_d)} |ext|={len(ext_d)} |zs|={len(zs_d)} no overlap")

    print("\n[6] Tokenizing each panel...")
    tokenizer = MaxTokiTokenizer()
    panel_to_adata = {
        "internal": a_imm,
        "external": a_imm,
        "zeroshot": a_imm,
        "lung_control": a_lung,
    }
    summary = {"panels": {}, "donors_internal": sorted(int_d), "donors_external": sorted(ext_d)}
    for label, anchor_df in panels.items():
        if len(anchor_df) == 0:
            print(f"  skipping {label}: 0 anchors")
            continue
        print(f"\n  [{label}] gathering cells...")
        adata = panel_to_adata[label]
        cells_df = gather_cells(adata, adata.obs, anchor_df, label)
        print(f"  [{label}] tokenizing {len(cells_df)} cells...")
        tok = tokenize_cells(adata, cells_df, tokenizer)
        # Persist
        out_path = OUT / f"cells_{label}.npz"
        np.savez_compressed(
            out_path,
            token_ids=tok["token_ids"],
            gene_positions=tok["gene_positions"],
            attn_mask=tok["attn_mask"],
            seq_lens=tok["seq_lens"],
        )
        cells_df.to_csv(OUT / f"cells_{label}_obs.csv", index=False)
        anchor_df.to_csv(OUT / f"anchors_{label}.csv", index=False)
        summary["panels"][label] = {
            "n_anchors": int(len(anchor_df)),
            "n_cells": int(len(cells_df)),
            "n_donors": int(anchor_df["donor_id"].nunique()),
            "n_tissues": int(anchor_df["tissue"].nunique()),
            "n_cell_types": int(anchor_df["cell_type"].nunique()),
            "n_stages": int(anchor_df["hema_stage"].nunique()),
            "max_seq_len": int(tok["seq_lens"].max()),
            "median_seq_len": int(np.median(tok["seq_lens"])),
            "tokens_path": str(out_path.relative_to(RUN)),
        }
        print(f"  [{label}] persisted → {out_path.relative_to(RUN)}")

    (OUT / "panel_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nTotal phase 1A time: {time.time()-t_phase:.1f}s")
    print(f"Summary written to {(OUT/'panel_summary.json').relative_to(RUN)}")


if __name__ == "__main__":
    main()
