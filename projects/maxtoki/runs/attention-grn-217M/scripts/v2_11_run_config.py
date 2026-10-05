"""v2 step 11 — run_config.json: inputs (sha256), scripts (sha256), outputs (sha256), seeds,
software versions, device. Hashes are cached by (path, size, mtime) so the step can be
re-run in pieces: python v2_11_run_config.py [max_seconds]
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v2_common import (PROJ, RUNS, V2, RUN_DIR, TRRUST_TSV, SYM2ENS_PKL, SEED, N_BOOT,  # noqa: E402
                       PRIMARY_LAYER, phase_dir, sha256, save_json)

CACHE = V2 / "_sha256_cache.json"
RAW = [
    Path("<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/crispri_validation/data/replogle_concat.h5ad"),
    Path("<DATA_ROOT>/biodyn-nmi-paper/results/round13_non_k562_replication/data/ReplogleWeissman2022_rpe1.h5ad"),
    Path("<DATA_ROOT>/biodyn-work/single_cell_mechinterp/data/perturb/adamson/perturb_processed_symbols.h5ad"),
    TRRUST_TSV, SYM2ENS_PKL,
    PROJ / "setup/dataset_loader.py", PROJ / "setup/token_dictionary.json",
    PROJ / "setup/MaxToki-217M-HF/config.json", PROJ / "setup/MaxToki-1B-HF/config.json",
    PROJ / "setup/MaxToki-217M-HF/model.safetensors", PROJ / "setup/MaxToki-1B-HF/model.safetensors",
]
PER_RUN = {
    "phase0": ["gene_features.csv", "attention_edges_layer_mean.npy", "spearman_edges.npy",
               "control_cells.csv", "run_config.json", "attention_pair_counts.npy"],
    "phase1": ["per_perturbation_auroc.csv", "wilcoxon_baseline_vs_edges.csv",
               "attention_per_layer_auroc.csv", "run_config.json"],
    "phase2": ["pair_dataset.csv", "delta_auroc_summary.csv", "run_config.json"],
    "phase3": ["full_pair_dataset.csv", "residualization_results.csv", "null_results.json", "run_config.json"],
}


def main(max_seconds: float):
    t0 = time.time()
    cache = json.load(open(CACHE)) if CACHE.exists() else {}

    def h(p: Path):
        st = p.stat()
        key = f"{p}|{st.st_size}|{int(st.st_mtime)}"
        if key not in cache:
            if time.time() - t0 > max_seconds:
                return None
            cache[key] = sha256(p)
            json.dump(cache, open(CACHE, "w"))
        return {"path": str(p), "bytes": st.st_size, "sha256": cache[key]}

    inputs = [h(p) for p in RAW]
    for run in RUNS:
        for ph, files in PER_RUN.items():
            for fn in files:
                inputs.append(h(phase_dir(ph, run) / fn))
    scripts = [h(p) for p in sorted((RUN_DIR / "scripts").glob("v2_*.py"))]
    outs = [h(p) for p in sorted(V2.rglob("*")) if p.is_file() and not p.name.startswith("._")
            and p.name not in ("run_config.json", CACHE.name)]
    if any(x is None for x in inputs + scripts + outs):
        print("time budget reached; re-run to continue"); return
    import numpy, scipy, sklearn, pandas, h5py  # noqa: E401
    cfg = {
        "item": "D1 attention-GRN v2 re-evaluation",
        "device": "CPU only; no model forward pass (safetensors read for headers only)",
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {"numpy": numpy.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
                     "pandas": pandas.__version__, "h5py": h5py.__version__},
        "seeds": {"master": SEED, "knockdown_bootstrap": "SEED+100+run_index", "incremental_bootstrap": "SEED+400+run_index",
                  "trrust_bootstrap": "SEED+500+run_index", "curveball_null": "SEED+600+run_index",
                  "curveball_mixing": "SEED+650+run_index", "checkerboard_null": "SEED+700+run_index",
                  "residualisation_bootstrap": "SEED+800+run_index", "layer_bootstrap": "SEED+900+run_index",
                  "DE_control_subsample": 42, "residualisation_kfold": 42},
        "run_index_order": list(RUNS),
        "parameters": {"primary_layer": PRIMARY_LAYER, "n_bootstrap": N_BOOT, "curveball_draws": 1000,
                       "curveball_trades_per_draw": 5000, "checkerboard_draws": 1000,
                       "checkerboard_accepted_swaps_per_edge": 30},
        "step_order": ["v2_01_facts.py (after v2_02)", "v2_02_rederive_de.py", "v2_03_knockdown.py",
                       "v2_04_incremental.py", "v2_05_trrust.py", "v2_06_curveball.py <run> ... ; combine",
                       "v2_07_residualise.py <run> ... ; combine", "v2_08_layers.py <run> ... ; combine",
                       "v2_09_figdata.py", "v2_10_task_data_rpe1.py", "v2_11_run_config.py"],
        "inputs": inputs, "scripts": scripts, "outputs": outs,
        "note": "outputs/v2_eval/de_labels/*.npz are derived intermediates (re-derived DE labels).",
    }
    save_json(cfg, V2 / "run_config.json")
    print("written", len(inputs), "inputs", len(scripts), "scripts", len(outs), "outputs", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 480)
