"""Build the Study A task T3 package data (TF specificity of layer-5 SAE features, MaxToki-217M, K562).

Reads the task data of the MaxToki deployment (read only) and writes neutral, package-relative copies to
studyA/tasks/T3-paper/data/. The contract package is a copy of the paper package plus contract/SPEC.md
(made by the shell step listed in keys/T3/NOTES.md).

Changes made to the source files:
  gene_universe.tsv   keep gene, detection_count, gene_length_bp, chr (drop the precomputed bin columns)
  feature_top20.tsv   keep feature_id, rank, gene; rename mean_act_when_active_rebuilt -> mean_act_when_active,
                      n_active_positions_rebuilt -> n_active_positions; drop gene_detection_count, in_rebuilt_top20
  feature_info.tsv    keep feature_id, n_genes_listed, has_special_token, n_active_positions (total)
  targets_*.tsv       unchanged
  cell_manifest.csv   drop priority and is_primary_tf
  cell_feature_means.npz  unchanged (rows, mean_gene, n_tokens)
  tf_info.tsv         new: tf, n_kd_cells, tf_gene_measured, kd_remaining_frac (from tf_summary.tsv)
Not copied: responding_features.tsv, tf_summary.tsv, tf_tests.tsv, catalog_check.json, README.md (rewritten).
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("<REPO_ROOT>/projects/maxtoki/runs/sae-atlas-217M/outputs/v2_tf_specificity/task_data")
DST = Path("<EVAL_ROOT>/studyA/tasks/T3-paper/data")
DST.mkdir(parents=True, exist_ok=True)

uni = pd.read_csv(SRC / "gene_universe.tsv", sep="\t", keep_default_na=False, na_values=[""])
uni["chr"] = uni["chr"].fillna("")
uni[["gene", "detection_count", "gene_length_bp", "chr"]].to_csv(DST / "gene_universe.tsv", sep="\t", index=False)

top = pd.read_csv(SRC / "feature_top20.tsv", sep="\t", keep_default_na=False)
top = top.rename(columns={"mean_act_when_active_rebuilt": "mean_act_when_active",
                          "n_active_positions_rebuilt": "n_active_positions"})
top[["feature_id", "rank", "gene", "mean_act_when_active", "n_active_positions"]].to_csv(
    DST / "feature_top20.tsv", sep="\t", index=False)

fi = pd.read_csv(SRC / "feature_info.tsv", sep="\t", keep_default_na=False)
fi = pd.DataFrame({"feature_id": fi.feature_id, "n_genes_listed": fi.n_top20,
                   "has_special_token": fi.n_special_in_top20.astype(int) > 0,
                   "n_active_positions": fi.n_active_positions_rebuilt})
fi.to_csv(DST / "feature_info.tsv", sep="\t", index=False)

for f in ["targets_trrust.tsv", "targets_dorothea.tsv"]:
    shutil.copyfile(SRC / f, DST / f)

man = pd.read_csv(SRC / "cell_manifest.csv", keep_default_na=False)
man[["row", "group", "tf", "barcode", "gem_group", "UMI_count"]].to_csv(DST / "cell_manifest.csv", index=False)

shutil.copyfile(SRC / "cell_feature_means.npz", DST / "cell_feature_means.npz")

ts = pd.read_csv(SRC / "tf_summary.tsv", sep="\t")
ti = pd.DataFrame({"tf": ts.tf, "n_kd_cells": ts.n_kd_cells,
                   "tf_gene_measured": ts.kd_remaining_frac_linear.notna(),
                   "kd_remaining_frac": ts.kd_remaining_frac_linear.round(4)})
ti.to_csv(DST / "tf_info.tsv", sep="\t", index=False)

# consistency checks
z = np.load(DST / "cell_feature_means.npz")
assert z["mean_gene"].shape == (9200, 4928)
assert set(z["rows"]) == set(man.row)
assert len(ti) == 87 and set(ti.tf) == set(man[man.group == "kd"].tf)
print("built", sorted(p.name for p in DST.iterdir()))
