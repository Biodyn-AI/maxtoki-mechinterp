# Residue concept labels from Swiss-Prot feature tables

## What this analysis does

This is one stage of a project that trains sparse autoencoders (SAEs) on the per-residue
activations of the protein language model ESM-2. The project asks whether single SAE
features line up with known residue-level properties of proteins ("concepts").

This stage builds the concepts. For every residue of every protein in the corpus, it
records which UniProtKB feature-table (`FT`) keys cover that residue. The result is a
boolean matrix with one row per residue and one column per FT key. There are 31 keys,
in nine groups: catalytic, binding, other sites, secondary structure, topology,
targeting, post-translational modification (PTM), domain architecture, and sequence
variation. The stage also writes a per-concept summary: the number of positive residues
and the prevalence (positive residues / all residues).

Downstream (not part of this package), each SAE feature is scored against each concept
as a per-residue binary classification (precision, recall, F1). Only concepts with
prevalence of at least 1e-3 are scored, because rarer concepts give noisy F1 values.
The variation group (VARIANT, MUTAGEN, CONFLICT) is used there as a negative control.
Row i of the label matrix must match row i of the activation cache; both follow the
corpus order.

## Data

- `data/corpus/swissprot_5k.jsonl` - 5,000 Swiss-Prot proteins, one JSON object per
  line with `acc` (primary accession), `name` (entry name), `seq` (amino-acid sequence)
  and `organism`. Proteins were sampled uniformly from Swiss-Prot entries of length 30
  to 1,022 residues (1,022 is the ESM-2 context limit). 1,606,104 residues in total.
- `data/uniprot/uniprot_sprot_subset.dat.gz` - the Swiss-Prot flat file
  (`uniprot_sprot.dat`, downloaded July 2026) cut down to the 5,000 corpus entries.
  Only the `ID`, `AC` and `FT` lines and the `//` entry terminators are kept; all other
  line types were removed to save space. FT lines use the current flat-file format, for
  example `FT   HELIX           4..17`, with qualifier lines such as `/evidence=...`
  under them.

## Code

- `code/dataset.py` - corpus loading (`load_corpus`), FT parsing (`parse_features`),
  the per-residue label matrix (`build_label_matrix`) and the per-concept summary
  (`concept_summary`).
- `code/build_labels.py` - runs the stage end to end.

## How to run

From the package root:

    python code/build_labels.py

This writes to `outputs/`. To write somewhere else, add `--out <dir>`. It needs only
numpy, runs on CPU in a few seconds and uses about 150 MB of memory.

## Outputs

- `outputs/residue_labels.npz` - `labels` (1,606,104 x 31, bool), `keys` (the column
  names) and `prot_index` (int32, the corpus index of the protein each row belongs to).
- `outputs/concept_summary.json` - `n_residues`, `n_proteins`, `n_accessions_with_ft`,
  and for each concept its `key`, `group`, `n_positive_residues` and `prevalence`,
  sorted by count.
- `outputs/build_log.txt` - console output of the run that produced these files.

`SUMMARY.md` is the analyst's write-up of the results.
