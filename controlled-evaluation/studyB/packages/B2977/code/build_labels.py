#!/usr/bin/env python
"""Build the per-residue concept label matrix from Swiss-Prot FT lines.

Ground truth for scoring SAE features. CPU-only.

Row order is `load_corpus()` order -- the same order the activation extractor writes
residues in.

Run from the package root:

    python code/build_labels.py              # writes to outputs/
    python code/build_labels.py --out DIR    # writes to DIR instead
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "code"))

from dataset import (ALL_FT_KEYS, FT_KEY_TO_GROUP, build_label_matrix,  # noqa: E402
                     concept_summary, load_corpus, parse_features)

CORPUS = PKG / "data/corpus/swissprot_5k.jsonl"
DAT = PKG / "data/uniprot/uniprot_sprot_subset.dat.gz"
MIN_PREVALENCE = 1e-3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=PKG / "outputs")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    proteins = load_corpus(CORPUS)
    accs = {p.acc for p in proteins}
    n_res = sum(len(p) for p in proteins)
    print(f"corpus: {len(proteins):,} proteins / {n_res:,} residues")

    t0 = time.time()
    print(f"parsing {DAT.name} (restricted to {len(accs):,} accessions)...", flush=True)
    feats = parse_features(DAT, accessions=accs)
    print(f"  parsed FT tables for {len(feats):,} accessions in {time.time()-t0:.0f}s")

    matched = len(accs & set(feats))
    print(f"  {matched:,}/{len(accs):,} corpus accessions carry >=1 FT annotation")
    if matched < 0.5 * len(accs):
        print("ABORT: <50% of corpus matched the flat file -- accession parsing is "
              "probably broken, not the biology.", file=sys.stderr)
        return 1

    labels, keys, prot_index = build_label_matrix(proteins, feats, keys=ALL_FT_KEYS)
    assert labels.shape[0] == n_res, f"row mismatch {labels.shape[0]} != {n_res}"

    summary = concept_summary(labels, keys)
    np.savez_compressed(out / "residue_labels.npz", labels=labels,
                        keys=np.array(keys), prot_index=prot_index)
    (out / "concept_summary.json").write_text(json.dumps({
        "n_residues": int(n_res), "n_proteins": len(proteins),
        "n_accessions_with_ft": matched, "concepts": summary,
    }, indent=2))

    print(f"\n{'concept':<12} {'group':<20} {'positives':>10} {'prevalence':>11}")
    for r in summary:
        if r["n_positive_residues"] == 0:
            continue
        print(f"{r['key']:<12} {FT_KEY_TO_GROUP.get(r['key'],'other'):<20} "
              f"{r['n_positive_residues']:>10,} {r['prevalence']:>11.5f}")
    usable = [r["key"] for r in summary if r["prevalence"] >= MIN_PREVALENCE]
    print(f"\n{len(usable)} concepts at prevalence >= {MIN_PREVALENCE:g} "
          f"(the scoring floor): {', '.join(usable)}")
    print("artefacts -> outputs/" if out == PKG / "outputs" else f"artefacts -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
