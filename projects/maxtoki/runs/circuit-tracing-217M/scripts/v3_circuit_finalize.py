"""v3 circuit tracing, part 6: run_config.json for outputs/v3_circuit/ and a source-feature comparison (item V3-3).

Writes outputs/v3_circuit/run_config.json (device, versions, seeds, cell rows, feature IDs, input and
code sha256, wall time per chunk, encoding-check results) and outputs/v3_circuit/source_feature_match.csv
(for each v3 source feature: best |cos| between its decoder direction and any deployed decoder direction
at the same layer, and whether that best match was a v2 / deployed source feature). CPU only.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v3_circuit_trace as T  # noqa: E402

H = T.H
I = T.I
OUT = T.OUT
DEP_SAE = T.PROJ / "runs/sae-atlas-217M/outputs/phase1"
V2 = T.RUN / "outputs/v2_circuit"


def wdec(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    W = ck["W_dec_weight"].double()                      # (d_model, d_sae)
    return W / W.norm(dim=0, keepdim=True).clamp_min(1e-12)


def feature_match():
    v3f = json.loads((OUT / "source_features.json").read_text())
    v2f = json.loads((V2 / "source_features.json").read_text())
    det = pd.read_csv(OUT / "annotation/selection_detail.csv")
    rows = []
    for s in T.SOURCE_LAYERS:
        Wn = wdec(T.V3SAE / f"layer_{s:02d}/sae_final.pt")
        Wo = wdec(DEP_SAE / f"layer_{s:02d}/sae_final.pt")
        v2set = set(int(x) for x in v2f[str(s)])
        for f in v3f[str(s)]:
            c = (Wo.T @ Wn[:, f]).abs()
            j = int(c.argmax())
            rows.append(dict(src_layer=s, v3_feature=int(f), best_deployed_feature=j, best_abs_cos=float(c[j]),
                             best_match_is_v2_source=j in v2set, same_number_is_v2_source=int(f) in v2set,
                             abs_cos_same_number=float(c[int(f)])))
    df = pd.DataFrame(rows).merge(det.rename(columns={"feature_id": "v3_feature"}), on=["src_layer", "v3_feature"])
    df.to_csv(OUT / "source_feature_match.csv", index=False)
    summ = {str(s): dict(n_same_number_as_v2_source=int(df[df.src_layer == s].same_number_is_v2_source.sum()),
                         n_best_match_abs_cos_ge_0p9=int((df[df.src_layer == s].best_abs_cos >= 0.9).sum()),
                         n_best_match_ge_0p9_and_v2_source=int(((df.src_layer == s) & (df.best_abs_cos >= 0.9) & df.best_match_is_v2_source).sum()),
                         median_best_abs_cos=float(df[df.src_layer == s].best_abs_cos.median())) for s in T.SOURCE_LAYERS}
    return summ


def main():
    t0 = time.time()
    fm = feature_match()
    z = np.load(OUT / "cells_tokens.npz")
    prog = json.loads((OUT / "trace_progress.json").read_text())
    enc = json.loads((OUT / "cells_encoding.json").read_text())
    scripts = sorted(HERE.glob("v3_circuit_*.py")) + [HERE / "v2_circuit_crispri.py"]
    sp = OUT / "spotcheck/progress.json"
    cfg = dict(
        item="V3-3: Stage-2 circuit tracing with v3 SAEs and correctly encoded inputs + CRISPRi direction test",
        env=H.env_info(T.DEVICE),
        model_dir=str(T.PROJ / "setup/MaxToki-217M-HF"), model_dtype="float32",
        sae_dir=str(T.V3SAE),
        sae_sha256={l: H.sha256_file(T.V3SAE / f"layer_{l:02d}/sae_final.pt") for l in range(12)},
        seeds=dict(cell_selection=T.SEED, torch_manual_seed=0, spotcheck_edge_pick=20261001, crispri_bootstrap=42,
                   crispri_source_folds=42, crispri_within_source_target_folds=43, lfc_control_sample=42,
                   verify_bootstrap=7, verify_lfc_gene_pick=20261002),
        cells=dict(dataset=str(z["h5_path"]), n_cells=int(len(z["rows"])), dataset_rows=[int(r) for r in z["rows"]],
                   n_tokens=[int(x) for x in z["lengths"]],
                   selection="numpy default_rng(42).choice of K562 non-targeting rows, 200, sorted (circuit_trace.py:100-123); "
                             "max_len 2048; same rows as the deployed and v2 runs",
                   counts_sha256=enc["counts_sha256"], tokens_sha256=enc["tokens_sha256"]),
        input_encoding=dict(
            rule="counts n = round(expm1(X) / u) (u = smallest non-zero expm1(X) of the cell; checked whole-number multiples), "
                 "ranked by n / sum(n) * 1e4 / Geneformer gene median, first 2,046 genes, <bos>/<eos> (inputs_v3)",
            load_time=enc["load_info"]["encoding_check"],
            per_chunk=[c["encoding_check"] for c in prog["chunks"]],
            all_chunks_pass=all(c["encoding_check"]["all_pass"] for c in prog["chunks"]),
            spotcheck_chunks=[c.get("encoding_check") for c in json.loads(sp.read_text())["chunks"]] if sp.exists() else None,
            old_vs_new_order=enc["old_vs_new_order"], record=enc["encoding"]),
        source_features=json.loads((OUT / "source_features.json").read_text()),
        source_feature_rule="per layer 0/3/6/9: the 30 features with the largest sum of -log10 p over their BH-significant "
                            "(q < 0.05) GO_BP/KEGG/Reactome/TRRUST enrichments of the top-20 genes (circuit_trace.py:133-145), "
                            "re-applied to the v3 SAE catalogs (v3_circuit_annotate.py)",
        source_feature_match_to_deployed=fm,
        edge_rule=dict(abs_d_gt=0.5, consistency_gt=0.7, d="mean/sqrt(var(ddof=1)+1e-12) over 200 cells",
                       consistency="max(#dz>0, n-#dz>0)/n", sign="inhibitory if mean dz < 0"),
        edit="hooks_v2.Ablate(s, [f]): delta = -z_f(x) W_dec[:, f] added to the input of block s (= hidden_states[s]), "
             "every position incl. <bos>/<eos>; read-out = hooks_v2 captures of the input of site t",
        trace_chunks=prog["chunks"],
        trace_wall_hours=round(sum(c["wall_seconds"] for c in prog["chunks"]) / 3600, 2),
        spotcheck_chunks=json.loads(sp.read_text())["chunks"] if sp.exists() else None,
        hooks_v2_sha256=H.sha256_file(T.PROJ / "setup/hooks_v2.py"),
        inputs_v3_sha256=H.sha256_file(T.PROJ / "setup/inputs_v3.py"),
        code_sha256={p.name: H.sha256_file(p) for p in scripts},
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), wall_seconds=round(time.time() - t0, 1))
    H.write_json(OUT / "run_config.json", cfg)
    # small run_config files for the sub-folders that do not write their own
    sc = json.loads((OUT / "spotcheck/spotcheck_result.json").read_text()) if (OUT / "spotcheck/spotcheck_result.json").exists() else {}
    H.write_json(OUT / "spotcheck/run_config.json", dict(
        script=str(HERE / "v3_circuit_spotcheck.py"), script_sha256=H.sha256_file(HERE / "v3_circuit_spotcheck.py"),
        env=sc.get("env"), device=T.DEVICE, seeds=sc.get("seeds"), sae_dir=str(T.V3SAE), dataset_rows=sc.get("dataset_rows"),
        edges=json.loads((OUT / "spotcheck/edges_picked.json").read_text()),
        chunks=json.loads((OUT / "spotcheck/progress.json").read_text())["chunks"],
        input_encoding="counts rebuilt independently (round(expm1(X)/u)), tokenised with tokenize_cell; inputs_v3.assert_encoding_batch "
                       "passed on every cell before its forward pass (see chunks)",
        cells_redrawn_sha256=H.sha256_file(OUT / "spotcheck/cells_redrawn.npz")))
    vr = json.loads((OUT / "verify/verify.json").read_text()) if (OUT / "verify/verify.json").exists() else {}
    H.write_json(OUT / "verify/run_config.json", dict(
        script=str(HERE / "v3_circuit_verify.py"), script_sha256=H.sha256_file(HERE / "v3_circuit_verify.py"), device="cpu",
        seeds=dict(bootstrap=7, lfc_gene_pick=20261002, control_sample=42), all_pass=vr.get("all_pass"),
        wall_seconds=vr.get("wall_seconds"),
        inputs=dict(edges_sha256=H.sha256_file(OUT / "circuit_edges_v3.csv"), per_pair_sha256=H.sha256_file(OUT / "per_pair.parquet"))))
    H.write_json(OUT / "crispri/run_config_bands.json", dict(
        script=str(HERE / "v3_circuit_bands.py"), script_sha256=H.sha256_file(HERE / "v3_circuit_bands.py"), device="cpu",
        seed=42, n_boot=2000, per_pair_sha256=H.sha256_file(OUT / "per_pair.parquet")))
    print(json.dumps(fm, indent=1))
    print("trace wall hours", cfg["trace_wall_hours"], "chunks", len(prog["chunks"]), "encoding all pass", cfg["input_encoding"]["all_chunks_pass"])


if __name__ == "__main__":
    main()
