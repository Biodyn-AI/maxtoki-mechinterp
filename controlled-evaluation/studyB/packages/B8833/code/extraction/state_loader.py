"""Load STATE SE-600M from the HF model folder (config.yaml + model.safetensors + protein_embeddings.pt)
WITHOUT the 11 GB Lightning .ckpt, and expose the per-gene residual stream.

Construction mirrors state.emb.train.trainer.main exactly (the same StateEmbeddingModel(...) call), then
loads the safetensors weights (strict=False) and sets pe_embedding from the local protein embeddings. We
override the cluster embedding path in cfg with the local protein_embeddings.pt so get_embedding_cfg works.

The residual stream = output of `model.transformer_encoder` : (batch, seq_len, d_model=2048). Position 0 is
the CLS/cell embedding (overwritten in _compute_embedding_for_batch); positions 1.. are gene tokens whose
identity is `GENES[batch_sentences[i, pos]]`, GENES = list(protein_embeds.keys()).
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from safetensors.torch import load_file

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "state_se600m"


def load_protein_embeds():
    """protein_embeddings.pt -> OrderedDict {gene_symbol: 5120-vector}. Returns (dict, gene_list, matrix)."""
    pe = torch.load(MODEL_DIR / "protein_embeddings.pt", weights_only=False, map_location="cpu")
    assert isinstance(pe, dict), type(pe)
    genes = list(pe.keys())
    mat = torch.vstack([pe[g].float() for g in genes])   # (n_genes, 5120)
    return pe, genes, mat


def build_cfg():
    cfg = OmegaConf.load(MODEL_DIR / "config.yaml")
    # point the "current" embedding entry at the local protein_embeddings.pt
    cur = cfg.embeddings.current
    cfg.embeddings[cur].all_embeddings = str(MODEL_DIR / "protein_embeddings.pt")
    return cfg


def load_state_se(device="cpu", dtype=torch.float32):
    from state.emb.nn.model import StateEmbeddingModel

    cfg = build_cfg()
    emb = cfg.embeddings[cfg.embeddings.current]
    pe_dict, genes, pe_mat = load_protein_embeds()

    model = StateEmbeddingModel(
        token_dim=int(emb.size),          # 5120
        d_model=int(cfg.model.emsize),    # 2048
        nhead=int(cfg.model.nhead),       # 16
        d_hid=int(cfg.model.d_hid),       # 2048
        nlayers=int(cfg.model.nlayers),   # 16
        output_dim=int(cfg.model.output_dim),  # 2048
        dropout=0.0,
        emb_size=int(emb.size),
        collater=None,
        cfg=cfg,
    )
    sd = load_file(str(MODEL_DIR / "model.safetensors"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # pe_embedding is set from the protein matrix (not in the safetensors state dict)
    model.pe_embedding = nn.Embedding.from_pretrained(pe_mat)
    model.eval().to(device=device, dtype=dtype)
    model.pe_embedding.to(device=device, dtype=dtype)
    info = dict(missing=list(missing), unexpected=list(unexpected), n_genes=len(genes),
                d_model=int(cfg.model.emsize), nlayers=int(cfg.model.nlayers))
    return model, cfg, genes, info


if __name__ == "__main__":
    model, cfg, genes, info = load_state_se()
    print("loaded STATE SE-600M")
    print("  d_model:", info["d_model"], "nlayers:", info["nlayers"], "n_genes:", info["n_genes"])
    print("  missing keys:", len(info["missing"]), "-> sample:", info["missing"][:6])
    print("  unexpected keys:", len(info["unexpected"]), "-> sample:", info["unexpected"][:6])
    print("  pe_embedding:", tuple(model.pe_embedding.weight.shape))
    nparams = sum(p.numel() for p in model.parameters())
    print(f"  params: {nparams/1e6:.1f}M")
    print("  transformer_encoder layers:", len(model.transformer_encoder.blocks)
          if hasattr(model.transformer_encoder, "blocks") else "?")
    print("  sample genes:", genes[:8])
