#!/usr/bin/env python
"""How data/attention_scores.npz was made (ESM-2 650M attention at every cysteine pair).

This step needs the ESM-2 650M weights (facebook/esm2_t33_650M_UR50D, revision
08e4846e537177426273712802403f7ba8261b6c) and several GB of memory. The shipped scores
were produced with it once, in bfloat16 with eager attention, and are the input to
code/disulfide_attention.py. You do not need to run it to reproduce the analysis.

For each protein and each of the 33 x 20 = 660 heads:

    A  = attention over the residue block only (<cls>/<eos> dropped)    (L x L)
    S  = (A + A^T) / 2                                                   symmetrise
    C  = S - outer(S.sum(1), S.sum(0)) / S.sum()                        APC

and the cysteine-pair entries C[i, j] are read off. Row 660 is the unweighted mean of
the 660 heads' C[i, j].

    python code/extract_attention.py --out data/attention_scores_rebuilt.npz
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODEL = "facebook/esm2_t33_650M_UR50D"
REVISION = "08e4846e537177426273712802403f7ba8261b6c"


def pair_scores_one_layer(A: np.ndarray, ii: np.ndarray, jj: np.ndarray) -> np.ndarray:
    """A: (n_head, L, L) attention of one layer over residues. Returns (n_head, n_pairs)."""
    Sym = 0.5 * (A + np.transpose(A, (0, 2, 1)))
    tot = Sym.sum(axis=(1, 2), keepdims=True)
    tot[tot == 0] = 1.0
    r = Sym.sum(2, keepdims=True)
    c = Sym.sum(1, keepdims=True)
    C = Sym - (r * c) / tot
    return C[:, ii, jj]


def main():
    import torch
    from transformers import AutoTokenizer, EsmModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA / "attention_scores_rebuilt.npz"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    with open(DATA / "proteins.tsv") as f:
        seqs = {r["accession"]: r["sequence"] for r in csv.DictReader(f, delimiter="\t")}
    dtype = dict(float32=torch.float32, bfloat16=torch.bfloat16)[args.dtype]
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = EsmModel.from_pretrained(MODEL, revision=REVISION, dtype=dtype,
                                     add_pooling_layer=False,
                                     attn_implementation="eager").eval().to(args.device)
    n_layers = model.config.num_hidden_layers
    n_head = model.config.num_attention_heads

    out = {}
    for split in ("select", "test"):
        z = np.load(DATA / f"pairs_{split}.npz", allow_pickle=False)
        acc, pi, pj = z["protein"], z["pos_i"], z["pos_j"]
        S = np.zeros((n_layers * n_head + 1, acc.size), dtype=np.float32)
        _, first = np.unique(acc, return_index=True)
        for p in acc[np.sort(first)]:
            seq = seqs[str(p)]
            sel = np.flatnonzero(acc == p)
            ii, jj = pi[sel], pj[sel]
            enc = {k: v.to(args.device) for k, v in
                   tok(seq, return_tensors="pt", add_special_tokens=True).items()}
            with torch.no_grad():
                att = model(**enc, output_attentions=True).attentions
            L = len(seq)
            acc_sum = np.zeros(sel.size, dtype=np.float64)
            for li in range(n_layers):
                A = att[li][0, :, 1:L + 1, 1:L + 1].float().cpu().numpy()
                v = pair_scores_one_layer(A, ii, jj)
                S[li * n_head:(li + 1) * n_head, sel] = v
                acc_sum += v.sum(0)
            S[-1, sel] = acc_sum / (n_layers * n_head)
        out[f"S_{split}"] = S
        print(f"{split}: {S.shape}")
    np.savez_compressed(args.out, n_layers=np.array(n_layers), n_head=np.array(n_head), **out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
