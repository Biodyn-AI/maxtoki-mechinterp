"""v2 step 1 — run facts: model sizes, cells, layers (read from files only, no model run).

Parameter counts come from the safetensors headers (tensor shapes only; weights are not
loaded). Cells and layers come from the deployed Phase-0 / Phase-1 / Phase-3 configs and
control_cells.csv files, and from the scripts' defaults.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v2_common import PROJ, RUNS, V2, PRIMARY_LAYER, phase_dir, n_layers_stored, save_json  # noqa: E402


def param_count(model_dir: Path) -> dict:
    p = model_dir / "model.safetensors"
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    tot = emb = head = 0
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        c = int(np.prod(v["shape"]))
        tot += c
        if "embed_tokens" in k:
            emb += c
        if k.startswith("lm_head"):
            head += c
    cfg = json.load(open(model_dir / "config.json"))
    return {"total_parameters": tot, "embedding_parameters": emb, "lm_head_parameters": head,
            "non_embedding_non_head_parameters": tot - emb - head,
            "n_layers": cfg["num_hidden_layers"], "n_heads": cfg["num_attention_heads"],
            "n_kv_heads": cfg.get("num_key_value_heads"), "hidden_size": cfg["hidden_size"]}


def main():
    models = {m: param_count(PROJ / "setup" / m) for m in ["MaxToki-217M-HF", "MaxToki-1B-HF"]}
    a, b = models["MaxToki-217M-HF"], models["MaxToki-1B-HF"]
    facts = {"models": models,
             "ratio_1B_over_217M_total": b["total_parameters"] / a["total_parameters"],
             "ratio_1B_over_217M_non_embedding": b["non_embedding_non_head_parameters"] / a["non_embedding_non_head_parameters"],
             "runs": {}}
    for run, (sfx, ds, mdir) in RUNS.items():
        c0 = json.load(open(phase_dir("phase0", run) / "run_config.json"))
        c1 = json.load(open(phase_dir("phase1", run) / "run_config.json"))
        c3 = json.load(open(phase_dir("phase3", run) / "run_config.json"))
        cc = pd.read_csv(phase_dir("phase0", run) / "control_cells.csv")
        L = n_layers_stored(run)
        de_meta = json.load(open(V2 / "de_labels" / f"{run}_meta.json"))
        per_layer = pd.read_csv(phase_dir("phase1", run) / "attention_per_layer_auroc.csv")
        best = per_layer.loc[per_layer["mean_auroc"].idxmax()]
        facts["runs"][run] = {
            "model_folder": mdir, "dataset": ds,
            "phase0_model_name_written_in_config": c0.get("model"),
            "stored_attention_layers": L, "config_n_layers": models[mdir]["n_layers"],
            "config_n_heads": models[mdir]["n_heads"],
            "control_cells_for_attention_and_gene_features": int(len(cc)),
            "phase0_config_n_ctrl": c0.get("n_ctrl"),
            "mean_tokens_per_cell": c0.get("mean_seq_len"),
            "control_cells_for_DE_calls": de_meta["n_control_cells_used_for_DE"],
            "control_cells_available": de_meta["n_control_cells_available"],
            "primary_layer_index_0based": c1["primary_layer"],
            "phase3_primary_layer": c3["primary_layer"],
            "primary_layer_depth_fraction_index_over_n_layers": c1["primary_layer"] / L,
            "knockdown_endpoint_best_layer_in_deployed_profile": int(best["layer"]),
            "knockdown_endpoint_best_layer_mean_auroc": float(best["mean_auroc"]),
            "knockdown_endpoint_primary_layer_mean_auroc": float(per_layer.loc[per_layer.layer == c1["primary_layer"], "mean_auroc"].iloc[0]),
            "deployed_curveball_draws": c3.get("curveball_n"),
        }
    save_json(facts, V2 / "run_facts.json")
    print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main()
