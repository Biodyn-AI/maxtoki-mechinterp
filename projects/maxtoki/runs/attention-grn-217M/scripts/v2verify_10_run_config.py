"""run_config.json for the verification pass (v2verify_*.py): inputs with sha256, scripts,
outputs, seeds, software. The 30 GB K562 h5ad hash is taken from the v2 hash cache
(outputs/v2_eval/_sha256_cache.json) when its (path, size, mtime) key matches; every
other file is hashed here. Output: outputs/v2_eval/verification/run_config.json
"""
import hashlib, json, platform, sys
from pathlib import Path

PROJ = Path("<REPO_ROOT>/projects/maxtoki")
RUN = PROJ / "runs/attention-grn-217M"
OUT = RUN / "outputs"
VER = OUT / "v2_eval/verification"
BIG = Path("<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/crispri_validation/data/replogle_concat.h5ad")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def entry(p):
    st = p.stat()
    if p == BIG:
        cache = json.load(open(OUT / "v2_eval/_sha256_cache.json"))
        key = f"{p}|{st.st_size}|{int(st.st_mtime)}"
        return {"path": str(p), "bytes": st.st_size, "sha256": cache[key], "note": "from v2 hash cache (path|size|mtime match)"}
    return {"path": str(p), "bytes": st.st_size, "sha256": sha(p)}


inputs = [BIG,
          Path("<DATA_ROOT>/biodyn-nmi-paper/results/round13_non_k562_replication/data/ReplogleWeissman2022_rpe1.h5ad"),
          Path("<DATA_ROOT>/biodyn-work/single_cell_mechinterp/data/perturb/adamson/perturb_processed_symbols.h5ad"),
          Path("<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv"),
          Path("<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/crispri_validation/data/gene_name_id_dict_gc104M.pkl"),
          PROJ / "setup/dataset_loader.py", PROJ / "setup/maxtoki_adapter.py",
          PROJ / "setup/MaxToki-217M-HF/config.json", PROJ / "setup/MaxToki-1B-HF/config.json",
          PROJ / "setup/MaxToki-217M-HF/model.safetensors", PROJ / "setup/MaxToki-1B-HF/model.safetensors",
          RUN / "scripts/phase3_residualization.py", OUT / "phase2_k562_1b/pair_dataset.csv"]
for sfx in ["", "_rpe1", "_adamson", "_k562_1b"]:
    for fn in ["gene_features.csv", "attention_edges_layer_mean.npy", "spearman_edges.npy", "control_cells.csv", "run_config.json"]:
        inputs.append(OUT / f"phase0{sfx}/{fn}")
    inputs += [OUT / f"phase1{sfx}/per_perturbation_auroc.csv", OUT / f"phase1{sfx}/run_config.json",
               OUT / f"phase3{sfx}/full_pair_dataset.csv"]
agent = ["knockdown/knockdown_summary.json", "trrust/trrust_summary.json", "curveball/curveball_summary.json",
         "layers/layers_summary.json", "_sha256_cache.json"]
agent += [f"knockdown/per_perturbation_{r}.csv" for r in ["217M_K562", "217M_RPE1", "217M_Adamson", "1B_K562"]]
agent += [f"curveball/null_curveball_{r}.npy" for r in ["217M_K562", "217M_RPE1", "217M_Adamson", "1B_K562"]]
inputs += [OUT / "v2_eval" / a for a in agent]
import numpy, scipy, sklearn, pandas, h5py  # noqa: E401
cfg = {"what": "independent verification of revision item D1 (attention-GRN v2), CPU only, no model forward pass",
       "python": platform.python_version(), "platform": platform.platform(),
       "packages": {"numpy": numpy.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
                    "pandas": pandas.__version__, "h5py": h5py.__version__},
       "seeds": {"endpointA_bootstrap": "7+run_index", "trrust_bootstrap": "11+run_index", "own_curveball_null": "23+run_index",
                 "incremental_bootstrap": 31, "incremental_refit_splits": "41..60", "single_log_knockdown_bootstrap": 51,
                 "single_log_trrust_bootstrap": "61+run_index", "deployed_null_rerun": "42..91 (deployed)",
                 "ols_kfold": 42},
       "run_index_order": ["217M_K562", "217M_RPE1", "217M_Adamson", "1B_K562"],
       "inputs": [entry(p) for p in inputs],
       "scripts": [entry(p) for p in sorted((RUN / "scripts").glob("v2verify_*.py"))],
       "outputs": [entry(p) for p in sorted(VER.glob("*.json")) if p.name != "run_config.json" and not p.name.startswith("._")]}
json.dump(cfg, open(VER / "run_config.json", "w"), indent=1)
print(len(cfg["inputs"]), "inputs", len(cfg["scripts"]), "scripts", len(cfg["outputs"]), "outputs")
