"""Stage 1 — frozen contextual probes + manifold diagnostics for MaxToki-217M.

Mirrors the source pipeline's Stage 1 (donor-held-out probes + manifold +
permutation null) but with the MaxToki representation as the model under test
and HVG-PCA as the trivial baseline.

Imports the upstream Stage-1 helpers (sample selection, age-label derivation,
group-split scoring, manifold metrics, permutation null) directly from
repos/longevity-mechinterp/implementation/scripts/run_stage1_longevity_mechinterp.py
so the methodology is bit-for-bit identical to the scGPT/Geneformer runs.

Outputs:
    outputs/stage1_<date>/
      ├── probe_aggregate.csv        balanced_accuracy / macro_f1 per representation
      ├── manifold_diagnostics.csv   participation_ratio / pc1_age_corr / silhouette
      ├── permutation_null.csv       null distribution for the best representation
      ├── sampled_obs.csv            per-cell metadata (donor, age, cell type)
      ├── extraction_stats.json      per-cell forward-pass timings
      └── report.md                  human-readable summary + G1 verdict
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# --- Wire up upstream Stage 1 helpers -------------------------------------------------
RUN_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = RUN_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

UPSTREAM_SCRIPTS = Path(
    "<REPO_ROOT>/repos/longevity-mechinterp/"
    "implementation/scripts"
)
sys.path.insert(0, str(UPSTREAM_SCRIPTS))

from run_stage1_longevity_mechinterp import (  # type: ignore  # noqa: E402
    _balanced_subsample_by_donor,
    _baseline_expression_representation,
    _build_sample_selection,
    _evaluate_representations,
    _load_subset_anndata,
    _permutation_null_for_best,
    _select_dataset_columns,
)

from maxtoki_runtime import MaxTokiContextualRuntime  # type: ignore  # noqa: E402


# --- Run config -----------------------------------------------------------------------
DATASET_PATH = Path(
    "<DATA_ROOT>/biodyn-work/single_cell_mechinterp/"
    "data/raw/tabula_sapiens_immune_subset_20000.h5ad"
)
DATASET_ID = "tabula_sapiens_immune_subset_20000"
PROJECT_ROOT = Path("<REPO_ROOT>/projects/maxtoki")
SETUP_DIR = PROJECT_ROOT / "setup"

TARGET_N_CELLS = 3000
MAX_CELLS_PER_DONOR = 200
MIN_CELLS_PER_DONOR = 30
MIN_CELLS_PER_AGE_LABEL = 50
AGE_BINS = 3
N_SPLITS = 5
N_PERM = 100
SEED = 42

MAX_GENES_PER_CELL = 1024  # MaxToki ctx 4096; 1024 is plenty for single-cell prompts
LAYER_INDICES = (5,)  # canonical L5 mid-stack residual

OUT_ROOT = RUN_ROOT / "outputs"
TODAY = datetime.now().strftime("%Y%m%d")
OUT_DIR = OUT_ROOT / f"stage1_{TODAY}"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    print(f"[stage1] writing to {OUT_DIR}")

    # 1. Pick obs columns + balanced donor subsample (uses upstream helpers).
    cols = _select_dataset_columns(DATASET_PATH)
    print(f"[stage1] columns: age={cols.age_col} donor={cols.donor_col} ct={cols.celltype_col}")

    selection = _build_sample_selection(
        path=DATASET_PATH,
        cols=cols,
        max_cells=TARGET_N_CELLS,
        max_cells_per_donor=MAX_CELLS_PER_DONOR,
        min_cells_per_donor=MIN_CELLS_PER_DONOR,
        min_cells_per_age_label=MIN_CELLS_PER_AGE_LABEL,
        age_bins=AGE_BINS,
        seed=SEED,
    )
    meta = selection.metadata.copy()
    print(
        f"[stage1] sampled {meta.shape[0]} cells across {meta['donor_id'].nunique()} donors, "
        f"age labels: {dict(meta['age_label'].value_counts())}"
    )
    print(f"[stage1] cell types ({meta['cell_type'].nunique()}): "
          f"{meta['cell_type'].value_counts().head(8).to_dict()}")
    meta.to_csv(OUT_DIR / "sampled_obs.csv", index=False)

    # 2. Load the cells.
    adata_sub = _load_subset_anndata(DATASET_PATH, selection.obs_index)
    print(f"[stage1] loaded subset adata: {adata_sub.shape}")

    # 3. HVG-PCA baseline (50 components).
    print(f"[stage1] computing HVG-PCA baseline (n_components=50)")
    t0 = time.time()
    hvg_pca = _baseline_expression_representation(adata_sub, n_components=50, seed=SEED)
    print(f"[stage1] HVG-PCA shape={hvg_pca.shape}  ({time.time()-t0:.1f}s)")

    # 4. MaxToki contextual representation.
    print(f"[stage1] extracting MaxToki layer{list(LAYER_INDICES)} representations")
    runtime = MaxTokiContextualRuntime(
        model_dir=SETUP_DIR / "MaxToki-217M-HF",
        model_tag="maxtoki217m",
    )
    cache_path = OUT_DIR / "maxtoki_layer_reps.npy"
    reps_maxtoki, ext_stats = runtime.extract_representations(
        adata_sub,
        max_genes_per_cell=MAX_GENES_PER_CELL,
        layer_indices=LAYER_INDICES,
        cache_path=cache_path,
    )
    # Drop NaN rows (cells where tokenization yielded zero gene tokens).
    arr_any = next(iter(reps_maxtoki.values()))
    finite_mask = np.isfinite(arr_any).all(axis=1)
    n_drop = int((~finite_mask).sum())
    print(f"[stage1] dropped {n_drop} cells with NaN representations")
    if n_drop > 0:
        meta = meta.loc[finite_mask].reset_index(drop=True)
        hvg_pca = hvg_pca[finite_mask.values]
        reps_maxtoki = {k: v[finite_mask.values] for k, v in reps_maxtoki.items()}

    print(f"[stage1] final n_cells: {meta.shape[0]}, donors: {meta['donor_id'].nunique()}")

    # 5. Build the rep dict {name: ndarray}.
    reps = dict(reps_maxtoki)
    reps["hvg_pca_50"] = hvg_pca

    # 6. Run probes + manifold metrics.
    print(f"[stage1] evaluating representations...")
    probe_df, manifold_df = _evaluate_representations(
        reps=reps, meta=meta, n_splits=N_SPLITS, seed=SEED
    )
    probe_df["dataset_id"] = DATASET_ID
    manifold_df["dataset_id"] = DATASET_ID
    probe_df.to_csv(OUT_DIR / "probe_aggregate.csv", index=False)
    manifold_df.to_csv(OUT_DIR / "manifold_diagnostics.csv", index=False)
    print("\n[stage1] probe aggregate:")
    print(probe_df.to_string(index=False))
    print("\n[stage1] manifold diagnostics:")
    print(manifold_df.to_string(index=False))

    # 7. Permutation null on the best representation.
    print(f"[stage1] permutation null (n_perm={N_PERM}) on best rep...")
    null_df = _permutation_null_for_best(
        reps=reps, probe_df=probe_df, meta=meta, n_splits=N_SPLITS, n_perm=N_PERM, seed=SEED
    )
    if not null_df.empty:
        null_df.to_csv(OUT_DIR / "permutation_null.csv", index=False)
        print("\n[stage1] permutation null:")
        print(null_df.to_string(index=False))

    # 8. Stats + verdict.
    best_row = probe_df.iloc[0].to_dict()
    best_rep = best_row["representation"]
    best_ba = float(best_row["balanced_accuracy_mean"])
    null_ba = (
        float(null_df.iloc[0]["null_p95"]) if not null_df.empty and "null_p95" in null_df.columns else None
    )
    null_p = (
        float(null_df.iloc[0]["p_value_right_tail"])
        if not null_df.empty and "p_value_right_tail" in null_df.columns
        else None
    )
    null_mean = (
        float(null_df.iloc[0]["null_mean"]) if not null_df.empty and "null_mean" in null_df.columns else None
    )
    null_std = (
        float(null_df.iloc[0]["null_std"]) if not null_df.empty and "null_std" in null_df.columns else None
    )

    hvg_row = probe_df[probe_df["representation"] == "hvg_pca_50"]
    hvg_ba = float(hvg_row.iloc[0]["balanced_accuracy_mean"]) if not hvg_row.empty else float("nan")

    g1_pass_vs_null = (null_p is not None) and (null_p < 0.05)
    g1_pass_vs_baseline = best_rep != "hvg_pca_50" and best_ba > hvg_ba
    g1_pass = bool(g1_pass_vs_null and g1_pass_vs_baseline)

    extraction_summary = {
        "dataset_id": DATASET_ID,
        "n_cells_in_pre_extract": int(meta.shape[0] + n_drop),
        "n_cells_in_final": int(meta.shape[0]),
        "n_donors_final": int(meta["donor_id"].nunique()),
        "n_cell_types_final": int(meta["cell_type"].nunique()),
        "max_genes_per_cell": MAX_GENES_PER_CELL,
        "layer_indices": list(LAYER_INDICES),
        "model": "MaxToki-217M-HF",
        "extraction_stats": {
            "n_cells_in": ext_stats.n_cells_in,
            "n_cells_out": ext_stats.n_cells_out,
            "n_skipped_zero_expr": ext_stats.n_skipped_zero_expr,
            "mean_seq_len": ext_stats.mean_seq_len,
            "mean_fwd_seconds": ext_stats.mean_fwd_seconds,
        },
        "best_representation": best_rep,
        "best_balanced_accuracy_mean": best_ba,
        "hvg_pca_balanced_accuracy_mean": hvg_ba,
        "null_mean": null_mean,
        "null_std": null_std,
        "null_p95": null_ba,
        "null_p_value": null_p,
        "g1_pass_vs_null": g1_pass_vs_null,
        "g1_pass_vs_baseline": g1_pass_vs_baseline,
        "g1_pass": g1_pass,
        "total_seconds": float(time.time() - t_start),
        "seed": SEED,
    }
    with open(OUT_DIR / "extraction_stats.json", "w") as f:
        json.dump(extraction_summary, f, indent=2)

    # 9. Markdown report.
    md = []
    md.append(f"# Stage 1 — frozen contextual probes + manifold diagnostics ({TODAY})")
    md.append("")
    md.append(f"- **Model:** MaxToki-217M-HF (LlamaForCausalLM, hidden_size 1232, 11 blocks)")
    md.append(f"- **Layer probed:** {[f'maxtoki217m_layer_{li:02d}' for li in LAYER_INDICES]}")
    md.append(f"- **Dataset:** `{DATASET_ID}` (Tabula Sapiens immune subset 20k)")
    md.append(f"- **Sample:** {meta.shape[0]} cells, {meta['donor_id'].nunique()} donors, "
              f"{meta['cell_type'].nunique()} cell types")
    md.append(f"- **Age bins:** {AGE_BINS} (quantile); labels = {dict(meta['age_label'].value_counts())}")
    md.append(f"- **Donor-held-out splits:** {N_SPLITS}-fold GroupShuffleSplit (test_size=0.25)")
    md.append(f"- **Permutation null:** n_perm={N_PERM}")
    md.append("")
    md.append(f"## Probe aggregate (sorted by balanced accuracy)")
    md.append("")
    md.append(probe_df.to_markdown(index=False, floatfmt=".4f"))
    md.append("")
    md.append(f"## Manifold diagnostics")
    md.append("")
    md.append(manifold_df.to_markdown(index=False, floatfmt=".4f"))
    md.append("")
    md.append(f"## Permutation null (best rep = `{best_rep}`)")
    md.append("")
    if not null_df.empty:
        md.append(null_df.to_markdown(index=False, floatfmt=".4f"))
    else:
        md.append("_no null computed_")
    md.append("")
    md.append(f"## G1 verdict")
    md.append("")
    md.append(f"- best balanced accuracy: **{best_ba:.4f}** ({best_rep})")
    md.append(f"- HVG-PCA-50 baseline: **{hvg_ba:.4f}**")
    if null_p is not None:
        md.append(
            f"- permutation null: mean={null_mean:.4f}±{null_std:.4f}, "
            f"p95={null_ba:.4f}, p-value={null_p:.4f}"
        )
    md.append(f"- **G1 (vs null):** {'✅ PASS' if g1_pass_vs_null else '❌ FAIL'}")
    md.append(f"- **G1 (vs HVG-PCA baseline):** {'✅ PASS' if g1_pass_vs_baseline else '❌ FAIL'}")
    md.append(f"- **G1 overall:** {'✅ PASS — proceed to Stage 2' if g1_pass else '❌ FAIL — stop per pipeline §11.4'}")
    md.append("")
    md.append(f"## Extraction stats")
    md.append("")
    md.append(f"- forward-pass mean: {ext_stats.mean_fwd_seconds:.2f}s/cell")
    md.append(f"- mean cell seq_len: {ext_stats.mean_seq_len:.0f} gene tokens")
    md.append(f"- total wall-clock: {extraction_summary['total_seconds']:.0f}s")
    (OUT_DIR / "report.md").write_text("\n".join(md))
    print(f"\n[stage1] wrote report → {OUT_DIR / 'report.md'}")
    print(f"\n[stage1] G1 verdict: {'PASS' if g1_pass else 'FAIL'} "
          f"(vs null: {g1_pass_vs_null}, vs baseline: {g1_pass_vs_baseline})")


if __name__ == "__main__":
    main()
