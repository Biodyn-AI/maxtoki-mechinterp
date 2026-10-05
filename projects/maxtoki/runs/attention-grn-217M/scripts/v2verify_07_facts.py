"""Independent check of model sizes and run facts (no weights loaded).

Parameter counts: read each model.safetensors header (JSON) and multiply shapes; also
check config.json (layers, heads). Run facts: Phase-0 run_config.json n_ctrl, number of
stored attention layers, Phase-1 primary layer, gene-set overlap between 217M K562 and 1B K562.
Output: outputs/v2_eval/verification/facts_check.json
"""
import json, struct
from pathlib import Path
import numpy as np, pandas as pd
PROJ = Path("<REPO_ROOT>/projects/maxtoki")
OUT = PROJ / "runs/attention-grn-217M/outputs"
res = {}
for m in ["MaxToki-217M-HF", "MaxToki-1B-HF"]:
    with open(PROJ / "setup" / m / "model.safetensors", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n))
    tot = sum(int(np.prod(v["shape"])) for k, v in h.items() if k != "__metadata__")
    tied = any(k.startswith("lm_head") for k in h)
    cfg = json.load(open(PROJ / "setup" / m / "config.json"))
    res[m] = {"params": tot, "has_lm_head_tensor": tied, "layers": cfg["num_hidden_layers"],
              "heads": cfg["num_attention_heads"], "tie_word_embeddings": cfg.get("tie_word_embeddings")}
res["ratio"] = res["MaxToki-1B-HF"]["params"] / res["MaxToki-217M-HF"]["params"]
for run, sfx in {"217M_K562": "", "217M_RPE1": "_rpe1", "217M_Adamson": "_adamson", "1B_K562": "_k562_1b"}.items():
    c0 = json.load(open(OUT / f"phase0{sfx}/run_config.json")); c1 = json.load(open(OUT / f"phase1{sfx}/run_config.json"))
    L = np.load(OUT / f"phase0{sfx}/attention_edges_layer_mean.npy", mmap_mode="r").shape[0]
    res[run] = {"n_ctrl": c0["n_ctrl"], "stored_layers": int(L), "primary_layer": c1["primary_layer"],
                "control_cells_rows": int(len(pd.read_csv(OUT / f"phase0{sfx}/control_cells.csv")))}
a = set(pd.read_csv(OUT / "phase0/gene_features.csv")["ensembl_id"]); b = set(pd.read_csv(OUT / "phase0_k562_1b/gene_features.csv")["ensembl_id"])
res["gene_overlap_217M_K562_vs_1B_K562"] = len(a & b)
json.dump(res, open(OUT / "v2_eval/verification/facts_check.json", "w"), indent=1)
print(json.dumps(res, indent=1))
