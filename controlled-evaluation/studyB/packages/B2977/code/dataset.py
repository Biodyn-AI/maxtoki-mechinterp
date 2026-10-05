"""Swiss-Prot sequence corpus + residue-level annotation parsing.

Residue-label extraction (`parse_features`, `build_label_matrix`) pulls the UniProtKB
`FT` feature table out of the Swiss-Prot flat file into a per-residue label matrix. This
is the ground truth that sparse-autoencoder (SAE) features of the protein language model
are scored against.

Here a *token* is one residue of one protein, and the annotation is a property of that
(protein, position) pair. So scoring is a per-residue binary classification problem,
scored with precision/recall/F1 as InterPLM does.
"""
from __future__ import annotations

import gzip
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# UniProtKB FT feature keys we treat as residue-level concepts.
# ---------------------------------------------------------------------------
# Grouped by the kind of biology they encode. Each is a candidate "concept" that an
# SAE feature might be monosemantic for. Keys are the literal FT line tokens in the
# Swiss-Prot flat file (uniprot_sprot.dat).
FT_CONCEPT_GROUPS: dict[str, list[str]] = {
    # --- functional sites: the highest-value targets, sparse and mechanistic ---
    "catalytic": ["ACT_SITE"],
    "binding": ["BINDING", "CA_BIND", "ZN_FING", "DNA_BIND", "NP_BIND", "METAL"],
    "site_other": ["SITE"],
    # --- structural / topological ---
    "secondary_structure": ["HELIX", "STRAND", "TURN"],
    "topology": ["TRANSMEM", "INTRAMEM", "TOPO_DOM"],
    "targeting": ["SIGNAL", "TRANSIT", "PROPEP"],
    # --- post-translational modification ---
    "ptm": ["MOD_RES", "LIPID", "CARBOHYD", "DISULFID", "CROSSLNK"],
    # --- architecture ---
    "domain": ["DOMAIN", "REPEAT", "COILED", "MOTIF", "COMPBIAS", "REGION"],
    # --- variation (useful as a negative control: PLMs should NOT track these) ---
    "variation": ["VARIANT", "MUTAGEN", "CONFLICT"],
}

FT_KEY_TO_GROUP: dict[str, str] = {
    k: g for g, keys in FT_CONCEPT_GROUPS.items() for k in keys
}
ALL_FT_KEYS: list[str] = sorted(FT_KEY_TO_GROUP)


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

@dataclass
class Protein:
    acc: str
    name: str
    seq: str
    organism: str = ""

    def __len__(self) -> int:
        return len(self.seq)


def load_corpus(path: Path) -> list[Protein]:
    out = []
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            out.append(Protein(d["acc"], d["name"], d["seq"], d.get("organism", "")))
    return out


# ---------------------------------------------------------------------------
# Residue-level annotation from the Swiss-Prot flat file
# ---------------------------------------------------------------------------

# FT lines in the modern (2019+) flat-file format:
#   FT   BINDING         63
#   FT                   /ligand="Zn(2+)"
#   FT   HELIX           4..17
_FT_RE = re.compile(r"^FT   ([A-Z_]+)\s+(?:([<>?]?\d+)(?:\.\.([<>?]?\d+))?)")
_NUM_RE = re.compile(r"\d+")


def parse_features(
    dat_path: Path,
    accessions: Optional[set[str]] = None,
    keys: Optional[list[str]] = None,
) -> dict[str, dict[str, list[tuple[int, int]]]]:
    """Parse residue-level FT ranges from a Swiss-Prot flat file (`.dat` or `.dat.gz`).

    Returns `{accession: {ft_key: [(start, end), ...]}}` with **1-based inclusive**
    UniProt coordinates. Fuzzy bounds (`<12`, `?`) are resolved to their numeric part;
    entries with no resolvable position are skipped.

    `accessions` restricts parsing to a subset (pass your corpus accessions).
    """
    keys = set(keys or ALL_FT_KEYS)
    out: dict[str, dict[str, list[tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
    opener = gzip.open if str(dat_path).endswith(".gz") else open

    cur_acc: Optional[str] = None
    keep = False
    with opener(dat_path, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("AC   "):
                if cur_acc is None:
                    cur_acc = line[5:].split(";")[0].strip()
                    keep = accessions is None or cur_acc in accessions
            elif line.startswith("//"):
                cur_acc, keep = None, False
            elif keep and line.startswith("FT   "):
                m = _FT_RE.match(line.rstrip("\n"))
                if not m:
                    continue
                key, start_s, end_s = m.group(1), m.group(2), m.group(3)
                if key not in keys or start_s is None:
                    continue
                sm = _NUM_RE.search(start_s)
                if not sm:
                    continue
                start = int(sm.group())
                if end_s:
                    em = _NUM_RE.search(end_s)
                    end = int(em.group()) if em else start
                else:
                    end = start
                if end < start:
                    start, end = end, start
                out[cur_acc][key].append((start, end))
    return {a: dict(d) for a, d in out.items()}


def build_label_matrix(
    proteins: list[Protein],
    features: dict[str, dict[str, list[tuple[int, int]]]],
    keys: Optional[list[str]] = None,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Flatten per-protein FT ranges into a per-residue boolean label matrix.

    Returns:
      labels      (n_residues, n_keys) bool -- aligned to the concatenation of
                  `proteins` in order, one row per residue
      keys        the column order
      prot_index  (n_residues,) int32 -- which protein each row came from

    Row order MUST match the order the activation extractor emits residues in, or
    every downstream number is silently wrong. Both use `load_corpus` order.
    """
    keys = list(keys or ALL_FT_KEYS)
    key_pos = {k: i for i, k in enumerate(keys)}
    total = sum(len(p) for p in proteins)
    labels = np.zeros((total, len(keys)), dtype=bool)
    prot_index = np.zeros(total, dtype=np.int32)

    offset = 0
    for pi, p in enumerate(proteins):
        L = len(p)
        prot_index[offset:offset + L] = pi
        for key, ranges in features.get(p.acc, {}).items():
            col = key_pos.get(key)
            if col is None:
                continue
            for start, end in ranges:
                # UniProt is 1-based inclusive -> python slice on the protein block
                s = offset + max(start - 1, 0)
                e = offset + min(end, L)
                if e > s:
                    labels[s:e, col] = True
        offset += L
    return labels, keys, prot_index


def concept_summary(labels: np.ndarray, keys: list[str]) -> list[dict]:
    """Per-concept residue counts and prevalence -- needed to sanity-check that a
    concept is frequent enough to score (a concept with 30 positive residues will
    produce noise-driven F1 values)."""
    n = labels.shape[0]
    rows = []
    for i, k in enumerate(keys):
        pos = int(labels[:, i].sum())
        rows.append({
            "key": k,
            "group": FT_KEY_TO_GROUP.get(k, "other"),
            "n_positive_residues": pos,
            "prevalence": pos / n if n else 0.0,
        })
    rows.sort(key=lambda r: -r["n_positive_residues"])
    return rows
