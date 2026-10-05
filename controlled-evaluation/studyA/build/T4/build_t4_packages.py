"""Build the T4 task data (cross-model gene-embedding alignment) for Study A.

Writes the shared data/ folder into both package folders:
  studyA/tasks/T4-paper/data/   and   studyA/tasks/T4-contract/data/
The two data folders are byte-identical (checked at the end).

Sources (read only, never modified):
  MaxToki-217M HF checkpoint       model.embed_tokens.weight
  MaxToki token dictionary          setup/token_dictionary.json
  Geneformer V2-316M checkpoint    bert.embeddings.word_embeddings.weight
  Geneformer token dictionary       token_dictionary_gc104M.pkl  (written as JSON)
  Geneformer gene name dictionary   gene_name_id_dict_gc104M.pkl (written as CSV)
  scGPT whole-human checkpoint      best_model.pt: encoder.embedding.weight, encoder.enc_norm.{weight,bias}
  scGPT vocab.json, args.json       copied byte for byte
  Topology panels                   runs/topology-141-217M/outputs/phase0/<panel>/gene_features.csv
                                    (only the Ensembl id column is kept, same row order)
CPU only. No forward pass.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open

EVAL = Path("<EVAL_ROOT>")
PROJ = Path("<REPO_ROOT>/projects/maxtoki")
RUN = PROJ / "runs/topology-141-217M"
GF_SNAP = Path("<HF_CACHE>/hub/models--ctheodoris--Geneformer"
               "/snapshots/05fcbeb8a27d49e0a7a4349152202ee2c1cbfd28")
SRC = {
    "maxtoki_st": PROJ / "setup/MaxToki-217M-HF/model.safetensors",
    "maxtoki_cfg": PROJ / "setup/MaxToki-217M-HF/config.json",
    "maxtoki_tok": PROJ / "setup/token_dictionary.json",
    "gf_st": GF_SNAP / "Geneformer-V2-316M/model.safetensors",
    "gf_cfg": GF_SNAP / "Geneformer-V2-316M/config.json",
    "gf_tok": GF_SNAP / "geneformer/token_dictionary_gc104M.pkl",
    "gf_names": GF_SNAP / "geneformer/gene_name_id_dict_gc104M.pkl",
    "scgpt_ckpt": Path("<DATA_ROOT>/biodyn-work/single_cell_mechinterp/"
                       "external/scGPT_checkpoints/whole-human/best_model.pt"),
    "scgpt_vocab": Path("<DATA_ROOT>/biodyn-work/single_cell_mechinterp/"
                        "external/scGPT_checkpoints/whole-human/vocab.json"),
    "scgpt_args": Path("<DATA_ROOT>/biodyn-work/single_cell_mechinterp/"
                       "external/scGPT_checkpoints/whole-human/args.json"),
    "scgpt_pt_deployed": Path("<DATA_ROOT>/biodyn-work/"
                              "subproject_53_scgpt_gpl_replication/embeddings/scgpt_whole_human_gene_embeddings.pt"),
}
PANELS = ["lung", "immune", "external_lung"]
OUT_PRIMARY = EVAL / "studyA/tasks/T4-paper/data"
OUT_COPY = EVAL / "studyA/tasks/T4-contract/data"
BUILD = EVAL / "studyA/build/T4"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def load_key(path: Path, key: str) -> np.ndarray:
    with safe_open(str(path), framework="pt") as f:
        t = f.get_tensor(key)
    return t.float().numpy() if t.dtype != torch.float32 else t.numpy()


def main():
    log = {"sources": {}, "outputs": {}, "checks": {}}
    for k, p in SRC.items():
        log["sources"][k] = {"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size}

    out = OUT_PRIMARY
    if out.exists():
        shutil.rmtree(out)
    for sub in ["maxtoki_217m", "geneformer_v2_316m", "scgpt_whole_human", "gene_ids", "panels"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    # ---------------- MaxToki
    with safe_open(str(SRC["maxtoki_st"]), framework="pt") as f:
        mt_dtype = str(f.get_tensor("model.embed_tokens.weight").dtype)
    mt = load_key(SRC["maxtoki_st"], "model.embed_tokens.weight")
    log["checks"]["maxtoki_dtype_in_checkpoint"] = mt_dtype
    np.save(out / "maxtoki_217m/embed_tokens.npy", np.ascontiguousarray(mt, dtype=np.float32))
    shutil.copyfile(SRC["maxtoki_tok"], out / "maxtoki_217m/token_dictionary.json")
    shutil.copyfile(SRC["maxtoki_cfg"], out / "maxtoki_217m/config.json")
    mt_tok = json.loads(SRC["maxtoki_tok"].read_text())

    # ---------------- Geneformer
    with safe_open(str(SRC["gf_st"]), framework="pt") as f:
        gf_dtype = str(f.get_tensor("bert.embeddings.word_embeddings.weight").dtype)
    gf = load_key(SRC["gf_st"], "bert.embeddings.word_embeddings.weight")
    log["checks"]["geneformer_dtype_in_checkpoint"] = gf_dtype
    np.save(out / "geneformer_v2_316m/word_embeddings.npy", np.ascontiguousarray(gf, dtype=np.float32))
    with open(SRC["gf_tok"], "rb") as f:
        gf_tok = pickle.load(f)
    gf_tok = {str(k): int(v) for k, v in gf_tok.items()}
    (out / "geneformer_v2_316m/token_dictionary.json").write_text(json.dumps(gf_tok, indent=0))
    shutil.copyfile(SRC["gf_cfg"], out / "geneformer_v2_316m/config.json")

    # ---------------- scGPT (checkpoint tensors, rows in the checkpoint's own order)
    st = torch.load(SRC["scgpt_ckpt"], map_location="cpu", weights_only=False)
    if isinstance(st, dict) and not any(k.startswith("encoder.") for k in st):
        for cand in ["model_state_dict", "state_dict", "model"]:
            if cand in st:
                st = st[cand]
                break
    W = st["encoder.embedding.weight"].float().numpy()
    ln_w = st["encoder.enc_norm.weight"].float().numpy()
    ln_b = st["encoder.enc_norm.bias"].float().numpy()
    log["checks"]["scgpt_ckpt_keys_used"] = ["encoder.embedding.weight", "encoder.enc_norm.weight", "encoder.enc_norm.bias"]
    np.save(out / "scgpt_whole_human/encoder_embedding_weight.npy", np.ascontiguousarray(W, dtype=np.float32))
    np.savez(out / "scgpt_whole_human/encoder_enc_norm.npz", weight=ln_w.astype(np.float32),
             bias=ln_b.astype(np.float32), eps=np.array(1e-5, dtype=np.float64))
    shutil.copyfile(SRC["scgpt_vocab"], out / "scgpt_whole_human/vocab.json")
    shutil.copyfile(SRC["scgpt_args"], out / "scgpt_whole_human/args.json")
    del st

    # cross-check vs the table the deployment read (normed_embeddings in the .pt extract)
    dep = torch.load(SRC["scgpt_pt_deployed"], map_location="cpu", weights_only=False)
    normed = torch.nn.functional.layer_norm(torch.from_numpy(W), (W.shape[1],), torch.from_numpy(ln_w),
                                            torch.from_numpy(ln_b), eps=1e-5).numpy()
    log["checks"]["scgpt_raw_vs_deployed_extract_max_abs_diff"] = float(np.abs(W - dep["embeddings"].float().numpy()).max())
    log["checks"]["scgpt_layernorm_vs_deployed_normed_max_abs_diff"] = float(
        np.abs(normed - dep["normed_embeddings"].float().numpy()).max())
    vocab = json.loads(SRC["scgpt_vocab"].read_text())
    log["checks"]["scgpt_vocab_json_equals_deployed_pt_vocab"] = bool(
        {k: int(v) for k, v in dep["vocab"].items()} == {k: int(v) for k, v in vocab.items()})
    log["checks"]["scgpt_vocab_json_key_position_equals_id"] = int(
        sum(1 for i, k in enumerate(vocab) if int(vocab[k]) == i))
    del dep

    # ---------------- Ensembl <-> symbol table (Geneformer release dictionary, one-to-one)
    with open(SRC["gf_names"], "rb") as f:
        names = pickle.load(f)
    tab = pd.DataFrame({"symbol": list(names.keys()), "ensembl_id": list(names.values())})
    log["checks"]["ensembl_symbol_rows"] = int(len(tab))
    log["checks"]["ensembl_symbol_unique_symbols"] = int(tab.symbol.nunique())
    log["checks"]["ensembl_symbol_unique_ensembl"] = int(tab.ensembl_id.nunique())
    tab[["ensembl_id", "symbol"]].to_csv(out / "gene_ids/ensembl_symbol.csv", index=False)

    # ---------------- panels (Ensembl id only, deployed row order)
    sym_of = dict(zip(tab.ensembl_id, tab.symbol))
    for p in PANELS:
        g = pd.read_csv(RUN / "outputs/phase0" / p / "gene_features.csv")
        ens = g["ensembl_id"].astype(str)
        # checks: the mapping table reproduces the deployed symbols; all genes have rows everywhere
        log["checks"][f"panel_{p}"] = {
            "n": int(len(g)),
            "symbol_from_table_equals_deployed_symbol": int(sum(sym_of.get(e) == s for e, s in zip(ens, g["symbol"]))),
            "maxtoki_token_equals_deployed": int(sum(mt_tok.get(e) == t for e, t in zip(ens, g["maxtoki_token_id"]))),
            "in_geneformer_dict": int(sum(e in gf_tok for e in ens)),
            "symbol_in_scgpt_vocab_exact_case": int(sum(sym_of.get(e) in vocab for e in ens)),
        }
        pd.DataFrame({"ensembl_id": ens}).to_csv(out / "panels" / f"{p}.csv", index=False)

    # ---------------- copy to the second package and check identity
    if OUT_COPY.exists():
        shutil.rmtree(OUT_COPY)
    shutil.copytree(out, OUT_COPY)
    for f in sorted(out.rglob("*")):
        if f.is_file() and not f.name.startswith("._"):
            rel = f.relative_to(out)
            h1 = sha(f); h2 = sha(OUT_COPY / rel)
            assert h1 == h2, rel
            log["outputs"][str(rel)] = {"sha256": h1, "bytes": f.stat().st_size}
    (BUILD / "build_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log["checks"], indent=2))


if __name__ == "__main__":
    main()
