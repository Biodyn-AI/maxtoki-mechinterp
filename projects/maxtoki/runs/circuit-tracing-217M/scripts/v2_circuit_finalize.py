"""v2 circuit tracing, part 6: run_config.json for outputs/v2_circuit/ and an old-vs-new edge comparison (item D2).

Writes outputs/v2_circuit/run_config.json and outputs/v2_circuit/compare_summary.json.
CPU only; no model forward pass.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_circuit_trace as T  # noqa: E402

H = T.H
OUT = T.OUT
SCRIPTS = Path(__file__).resolve().parent


def compare():
    new = pd.read_csv(OUT / "circuit_edges_v2.csv")
    old = pd.read_csv(T.OLD_EDGES)
    key = ["src_layer", "src_feature", "tgt_layer", "tgt_feature"]
    res = {}
    per_src = []
    for s in T.SOURCE_LAYERS:
        n = new[new.src_layer == s]; o = old[old.src_layer == s]
        per_src.append(dict(src_layer=s, n_edges_v2=int(len(n)), n_edges_deployed=int(len(o)),
                            share_inhibitory_v2=float((n.sign == "inhibitory").mean()) if len(n) else None,
                            share_inhibitory_deployed=float((o.sign == "inhibitory").mean()) if len(o) else None,
                            edges_per_feature_v2_median=float(n.groupby("src_feature").size().reindex(
                                T.select_source_features()[s], fill_value=0).median())))
    res["per_source_layer"] = per_src
    # chance overlap: the deployed run has no edges at t = s + 1, so compare on t >= s + 2 only
    n2 = new[new.tgt_layer >= new.src_layer + 2]
    o2 = old[old.tgt_layer >= old.src_layer + 2]
    n_combos = sum((T.N_SITES - (s + 2)) * T.N_FEATURES_PER_LAYER for s in T.SOURCE_LAYERS)
    possible = n_combos * T.D_SAE
    shared = n2.merge(o2, on=key, suffixes=("_new", "_old"))
    res["overlap_t_ge_s_plus_2"] = dict(
        n_possible_pairs=int(possible), n_v2=int(len(n2)), n_deployed=int(len(o2)),
        deployed_density=len(o2) / possible,
        frac_v2_edges_also_deployed=len(shared) / max(1, len(n2)),
        expected_if_independent=len(o2) / possible,
        n_shared=int(len(shared)),
        sign_agreement_shared=float((shared.sign_new == shared.sign_old).mean()),
        pearson_d_shared=float(np.corrcoef(shared.cohens_d_new, shared.cohens_d_old)[0, 1]),
        spearman_d_shared=float(shared.cohens_d_new.rank().corr(shared.cohens_d_old.rank())))
    res["abs_d_v2_quantiles"] = {str(q): float(new.cohens_d.abs().quantile(q)) for q in [0.1, 0.5, 0.9, 0.99]}
    res["abs_d_deployed_quantiles"] = {str(q): float(old.cohens_d.abs().quantile(q)) for q in [0.1, 0.5, 0.9, 0.99]}
    res["frac_abs_d_gt_1"] = dict(v2=float((new.cohens_d.abs() > 1).mean()), deployed=float((old.cohens_d.abs() > 1).mean()))
    res["frac_abs_d_gt_2"] = dict(v2=float((new.cohens_d.abs() > 2).mean()), deployed=float((old.cohens_d.abs() > 2).mean()))
    return res


def main():
    t0 = time.time()
    cmp_ = compare()
    H.write_json(OUT / "compare_summary.json", cmp_)
    z = np.load(OUT / "cells_tokens.npz")
    prog = json.loads((OUT / "trace_progress.json").read_text())
    scripts = sorted(p for p in SCRIPTS.glob("v2_circuit_*.py"))
    cfg = dict(
        item="D2: Stage-2 circuit tracing with fixed hooks + CRISPRi directional evaluation",
        env=H.env_info(T.DEVICE),
        model_dir=str(T.PROJ / "setup/MaxToki-217M-HF"), model_dtype="float32",
        sae_dir=str(T.PROJ / "runs/sae-atlas-217M/outputs/phase1"),
        seeds=dict(cell_selection=T.SEED, torch_manual_seed=0, spotcheck_edge_pick=20261001,
                   resume_check_cell_pick=20261001, crispri_bootstrap=42, crispri_source_folds=42,
                   crispri_within_source_target_folds=43, lfc_control_sample=42),
        cells=dict(dataset=str(z["h5_path"]), n_cells=int(len(z["rows"])), dataset_rows=[int(r) for r in z["rows"]],
                   n_tokens=[int(x) for x in z["lengths"]], selection="numpy default_rng(42).choice of K562 "
                   "non-targeting rows, 200, sorted (circuit_trace.py:100-123); max_len 2048"),
        source_features=json.loads((OUT / "source_features.json").read_text()),
        edge_rule=dict(abs_d_gt=0.5, consistency_gt=0.7,
                       d="mean/sqrt(var(ddof=1)+1e-12) over 200 cells",
                       consistency="max(#cells dz>0, n - #cells dz>0)/n"),
        trace_chunks=[dict(start=c["start"], wall_seconds=c["wall_seconds"], cells=c["cells_processed"],
                           seconds_per_cell=c["seconds_per_cell"], status=c["status"],
                           trace_script_sha256=c.get("trace_script_sha256",
                                                     "2973ef9c56207f011ba300164cc9932a585d38bf1f01e6d5b8e44b0c766108e0"
                                                     if c["start"] >= "2026-10-01T04:42" else "not recorded (cells 0-4, "
                                                     "before the exact skip of inactive features was added)"))
                      for c in prog["chunks"]],
        total_trace_hours=round(sum(c["wall_seconds"] for c in prog["chunks"]) / 3600, 2),
        script_sha256={p.name: H.sha256_file(p) for p in scripts},
        notes=["Cells 0-170 were traced by an earlier session (interrupted when the app quit); cells 171-199 in this "
               "session. checks/resume_check.json: two finished cells recomputed, 'main' identical (max abs diff 0.0).",
               "Trace script edits during the run: (1) after cell 4, exact skip of source features with z=0 at every "
               "position (checked on cells 0-4); (2) after cell 170, install_capture_hooks_first (only affects the "
               "legacy zero-edit control; repaired for the 31 earlier first-of-chunk cells by v2_circuit_legacy_fix.py)."],
        wall_seconds=round(time.time() - t0, 1),
    )
    H.write_json(OUT / "run_config.json", cfg)
    print(json.dumps(cmp_, indent=1))
    print("total trace hours", cfg["total_trace_hours"])


if __name__ == "__main__":
    main()
