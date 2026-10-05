"""Stage 2 — manifold robustness gate for MaxToki-217M.

Reads Stage 1 outputs (probe_aggregate.csv, manifold_diagnostics.csv,
sampled_obs.csv) and runs:

  - top-k representation geometry merge (probe + manifold metrics for the
    best representations)
  - within-cell-type expression-baseline controls: for each cell type with
    enough cells/donors/age classes, compute HVG-PCA and re-run the donor-
    held-out probe. If age signal disappears once we condition on cell type,
    Stage 1 geometry is composition-driven.

The within-cell-type pass uses tighter `min_donors_per_celltype=8` (vs upstream
default 20) because Tabula Sapiens immune subset 20k has only 17 donors after
balanced-subsample filtering — the upstream default would skip every cell type.
This is a noted deviation from the pipeline's §8 default, justified by the
porting recipe in §11 (small-scale prototype).

Outputs:
    outputs/stage2_<date>/
      ├── topk_representation_geometry.csv
      ├── within_celltype_expr_controls.csv
      ├── stage2_summary.json
      └── report.md
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "scripts"))
sys.path.insert(
    0,
    "<REPO_ROOT>/repos/longevity-mechinterp/"
    "implementation/scripts",
)

from run_stage2_manifold_robustness import (  # type: ignore  # noqa: E402
    _topk_representation_geometry,
    _within_celltype_expr_controls,
)


DATASET_ID = "tabula_sapiens_immune_subset_20000"
DATASET_PATH = Path(
    "<DATA_ROOT>/biodyn-work/single_cell_mechinterp/"
    "data/raw/tabula_sapiens_immune_subset_20000.h5ad"
)

# Within-cell-type filters — relaxed for 17-donor cohort.
TOP_K = 3
EXPR_COMPONENTS = 64
GROUP_SPLITS = 5
MIN_CELLS_PER_CELLTYPE = 80
MIN_DONORS_PER_CELLTYPE = 8  # default 20 → 8 (Tabula Sapiens immune subset has only 17 donors)
MIN_AGE_CLASSES_PER_CELLTYPE = 3
MAX_CELLTYPES = 12
G2_PC1_ABS_THRESHOLD = 0.10
G2_SIL_THRESHOLD = 0.00
SEED = 42


def _resolve_stage1_dir() -> Path:
    candidates = sorted((RUN_ROOT / "outputs").glob("stage1_*"))
    candidates = [c for c in candidates if c.is_dir()]
    if not candidates:
        raise FileNotFoundError("No Stage 1 output directories found")
    return candidates[-1]


def main():
    today = datetime.now().strftime("%Y%m%d")
    stage1_dir = _resolve_stage1_dir()
    stage2_dir = RUN_ROOT / "outputs" / f"stage2_{today}"
    stage2_dir.mkdir(parents=True, exist_ok=True)
    print(f"[stage2] reading Stage 1 from {stage1_dir}")
    print(f"[stage2] writing Stage 2 to {stage2_dir}")

    probe_df = pd.read_csv(stage1_dir / "probe_aggregate.csv")
    manifold_df = pd.read_csv(stage1_dir / "manifold_diagnostics.csv")
    meta = pd.read_csv(stage1_dir / "sampled_obs.csv")
    print(f"[stage2] inputs: probe rows={len(probe_df)}, manifold rows={len(manifold_df)}, "
          f"sample rows={len(meta)}")

    topk_df = _topk_representation_geometry(probe_df=probe_df, manifold_df=manifold_df, top_k=TOP_K)
    if not topk_df.empty:
        topk_df.insert(0, "dataset_id", DATASET_ID)
        topk_df.to_csv(stage2_dir / "topk_representation_geometry.csv", index=False)
        print("[stage2] top-k geometry:")
        print(topk_df.to_string(index=False))

    print(f"[stage2] within-cell-type controls (max {MAX_CELLTYPES} cell types)...")
    within_df = _within_celltype_expr_controls(
        dataset_path=DATASET_PATH,
        sampled_meta=meta,
        expr_components=EXPR_COMPONENTS,
        group_splits=GROUP_SPLITS,
        min_cells_per_celltype=MIN_CELLS_PER_CELLTYPE,
        min_donors_per_celltype=MIN_DONORS_PER_CELLTYPE,
        min_age_classes_per_celltype=MIN_AGE_CLASSES_PER_CELLTYPE,
        max_celltypes_per_dataset=MAX_CELLTYPES,
        seed=SEED,
    )
    if not within_df.empty:
        within_df.insert(0, "dataset_id", DATASET_ID)
        within_df.to_csv(stage2_dir / "within_celltype_expr_controls.csv", index=False)
        ok = within_df[within_df["status"] == "ok"]
        print(f"[stage2] within-cell-type ok rows: {len(ok)}/{len(within_df)}")
        if not ok.empty:
            print(ok[["cell_type", "n_cells", "n_donors", "n_age_classes",
                      "balanced_accuracy_mean", "pc1_age_corr", "age_label_silhouette"]]
                  .to_string(index=False))

    # G2 verdict
    topk_max_pc1 = float(topk_df["pc1_age_corr_abs"].max()) if not topk_df.empty else float("nan")
    topk_max_sil = float(topk_df["age_label_silhouette"].max()) if not topk_df.empty else float("nan")
    g2_geometry_pass = bool(
        np.isfinite(topk_max_pc1) and np.isfinite(topk_max_sil)
        and topk_max_pc1 >= G2_PC1_ABS_THRESHOLD
        and topk_max_sil >= G2_SIL_THRESHOLD
    )

    summary = {
        "dataset_id": DATASET_ID,
        "stage1_dir": str(stage1_dir),
        "best_representation": str(probe_df.iloc[0]["representation"]) if not probe_df.empty else "",
        "best_balanced_accuracy_mean": float(probe_df.iloc[0]["balanced_accuracy_mean"])
        if not probe_df.empty else float("nan"),
        "topk_max_pc1_abs_corr": topk_max_pc1,
        "topk_max_silhouette": topk_max_sil,
        "g2_geometry_pass": g2_geometry_pass,
        "min_donors_per_celltype": MIN_DONORS_PER_CELLTYPE,
        "n_within_celltype_tests_ok": int((within_df["status"] == "ok").sum())
        if not within_df.empty else 0,
        "n_within_celltype_tests_skipped": int((within_df["status"] == "skipped").sum())
        if not within_df.empty else 0,
        "within_celltype_best_bacc": (
            float(within_df.loc[within_df["status"] == "ok", "balanced_accuracy_mean"].max())
            if not within_df.empty and (within_df["status"] == "ok").any()
            else float("nan")
        ),
    }
    (stage2_dir / "stage2_summary.json").write_text(json.dumps(summary, indent=2))

    md = []
    md.append(f"# Stage 2 — manifold robustness gate ({today})")
    md.append("")
    md.append(f"- **Stage 1 source:** `{stage1_dir.name}`")
    md.append(f"- **Best representation:** `{summary['best_representation']}`  "
              f"(balanced_accuracy_mean = {summary['best_balanced_accuracy_mean']:.4f})")
    md.append(f"- **Top-k max |pc1_age_corr|:** {topk_max_pc1:.4f}")
    md.append(f"- **Top-k max silhouette:** {topk_max_sil:.4f}")
    md.append(f"- **G2 geometry gate:** {'✅ PASS' if g2_geometry_pass else '❌ FAIL'}  "
              f"(thresholds: |pc1_corr| ≥ {G2_PC1_ABS_THRESHOLD}, silhouette ≥ {G2_SIL_THRESHOLD})")
    md.append("")
    md.append(f"## Top-k representation geometry")
    md.append("")
    if not topk_df.empty:
        md.append(topk_df[["representation", "balanced_accuracy_mean", "pc1_age_corr",
                           "pc1_age_corr_abs", "age_label_silhouette",
                           "participation_ratio"]].to_markdown(index=False, floatfmt=".4f"))
    md.append("")
    md.append(f"## Within-cell-type expression-baseline controls")
    md.append("")
    md.append(f"_Tighter `min_donors_per_celltype = {MIN_DONORS_PER_CELLTYPE}` (vs upstream default 20) "
              f"because Tabula Sapiens immune subset has only 17 donors._")
    md.append("")
    if not within_df.empty:
        ok = within_df[within_df["status"] == "ok"]
        skipped = within_df[within_df["status"] == "skipped"]
        if not ok.empty:
            md.append("### Cell types with successful within-cell-type probe")
            md.append("")
            md.append(ok[["cell_type", "n_cells", "n_donors", "n_age_classes",
                          "balanced_accuracy_mean", "balanced_accuracy_std",
                          "pc1_age_corr", "age_label_silhouette",
                          "participation_ratio"]].to_markdown(index=False, floatfmt=".4f"))
            md.append("")
        if not skipped.empty:
            md.append("### Cell types skipped")
            md.append("")
            md.append(skipped[["cell_type", "n_cells", "n_donors", "n_age_classes", "skip_reason"]]
                      .to_markdown(index=False))
            md.append("")
    md.append("## Interpretation")
    md.append("")
    if g2_geometry_pass:
        md.append("- Top-k representation geometry passes G2: at least one of the best representations "
                  "has a non-trivial PC1↔age correlation and age-label silhouette.")
    else:
        md.append("- Top-k representation geometry **fails G2**: none of the best representations "
                  "have a non-trivial PC1↔age correlation. Whatever donor-held-out probe accuracy "
                  "we measured in Stage 1 may be carried by within-rep features that do not align "
                  "with the leading PC. This is consistent with source-paper Stage-2 outcome (0/5 "
                  "datasets passed the geometry gate on scGPT/Geneformer).")
    n_ok = summary["n_within_celltype_tests_ok"]
    if n_ok > 0:
        md.append(f"- Within-cell-type expression-baseline probes succeeded for {n_ok} cell types. "
                  "If these match or exceed the global Stage 1 balanced accuracy, age signal is "
                  "intrinsic (not composition-driven). If they collapse to chance (~0.33 for 3 "
                  "balanced bins), Stage 1 was carrying composition information rather than "
                  "intra-celltype age program.")
    md.append("")
    (stage2_dir / "report.md").write_text("\n".join(md))
    print(f"[stage2] wrote report → {stage2_dir / 'report.md'}")


if __name__ == "__main__":
    main()
