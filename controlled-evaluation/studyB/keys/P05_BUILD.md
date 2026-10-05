# P05 build record (protein-lm-sae residue concept labels)

- Flawed package: `studyB/packages/B2977` (key `keys/B2977.json`)
- Clean package: `studyB/packages/B4481` (key `keys/B4481.json`)
- Build scripts: `studyB/build/P05/` (extract.py, subset.py, verify_full.py, verify_packages.py)
- Built 2026-10-01. Python: `maxtoki-framework-eval/bin/python` (numpy 1.26.4). CPU only.

## Sources (read only, nothing in biomi_automation was changed)

- Code: `biomi_automation/projects/protein-lm-sae/setup/dataset.py` (functions `load_corpus`,
  `parse_features`, `build_label_matrix` with the span fill at line 260, `concept_summary`)
  and `run/02_build_labels.py`.
- Data: `data/corpus/swissprot_20k.jsonl` (20,000 proteins, 6,519,320 residues) and
  `data/uniprot/uniprot_sprot.dat.gz` (575,503 entries).
- Reference outputs: `data/labels/residue_labels.npz` + `concept_summary.json` (span-filled)
  and `data/labels/residue_labels_bondfix.npz` + `concept_summary_bondfix.json` (bond ends only).
  The builder for the bondfix files is not on disk. I rewrote it from the description in
  `setup/align.py` (mark only the two bond residues for DISULFID and CROSSLNK).
- To avoid writing `__pycache__` into the source repo, dataset.py was copied to the scratchpad
  and imported from there with bytecode writing off.

## Step 1 - extract the flat-file lines and check the full corpus

`extract.py` streamed `gzip -dc uniprot_sprot.dat.gz` and kept the ID, AC, FT and `//` lines
for the 20,000 corpus accessions (all 20,000 found; 1 min). `verify_full.py` ran the
original `parse_features` + `build_label_matrix` on this excerpt:

| check (20,000 proteins) | result |
|---|---|
| n_residues / accessions with FT | 6,519,320 / 16,399 - both match `concept_summary.json` |
| span-filled matrix vs `residue_labels.npz` | identical, bit for bit (labels, keys, prot_index) |
| span-filled counts vs `concept_summary.json` | identical for all 31 keys |
| bond-ends matrix vs `residue_labels_bondfix.npz` | identical, bit for bit |
| bond-ends counts vs `concept_summary_bondfix.json` | identical for all 31 keys |
| DISULFID span / bond ends | 120,551 (7.19% Cys) / 7,889 (100% Cys) |
| CROSSLNK span / bond ends | 3,312 (20.6% Lys) / 739 (74.7% Lys) |
| CARBOHYD (control) | 3,489 both (95.7% Asn) |
| concepts >= 1e-3 | 22 both |

So my rebuilt bond-ends rule reproduces the original fixed labels exactly.

Note on the pair description: it says CROSSLNK "falls below the 1e-3 floor" after the fix.
In the full corpus CROSSLNK is below the floor in BOTH versions (5.1e-4 span, 1.1e-4 bond
ends), and DISULFID stays above it in both (1.85e-2 vs 1.21e-3). The error changes what
DISULFID means and how big it is, not which concepts are scored.

## Step 2 - shrink to 5,000 proteins

`subset.py`: uniform random 5,000 of the 20,000 (numpy default_rng(0), one draw, order kept),
1,606,104 residues, lengths 30-1,020. The flat-file excerpt was cut to those 5,000 entries
(ID/AC/FT/`//` lines only, qualifier lines kept) and gzipped: 621 KB. Corpus jsonl 2.2 MB.
Packages are 3.0 MB each.

## Step 3 - what changed versus the source code

Both packages (identical files except `code/dataset.py` and `SUMMARY.md`):
- `code/dataset.py`: copied from `setup/dataset.py`. Removed unused `iter_fasta`,
  `build_corpus` and `aa_composition` (no FASTA ships, and no composition table should ship).
  Removed the module-docstring lines about the single-cell pipelines and release stats.
  `parse_features` docstring trimmed. The span fill line is unchanged (flawed package line 174).
- `code/build_labels.py`: from `run/02_build_labels.py`. Paths made package-relative,
  `--out` option added, the activation-metadata cross-check removed (no activations ship),
  and the final line now lists the concepts above the floor.
- README.md: identical in both packages (sha256 checked). No outcome statements.
- No comment in either package mentions the error or the fix. The clean package adds only
  `BOND_KEYS = ("DISULFID", "CROSSLNK")` (line 52) and a 5-line branch in
  `build_label_matrix` (lines 171-175) that marks the start and end positions only.
- `align.py` (which holds the warning) was not shipped.

## Step 4 - verification of the packages

Both packages were run from their root with `bin/python code/build_labels.py` (1.7 s,
130 MB peak), then re-run with `--out` to a scratch folder: arrays, JSON and the log (minus
the timing line) are identical. `verify_packages.py` results:

| check (5,000 proteins) | flawed B2977 | clean B4481 |
|---|---|---|
| matrix == matching rows of the original matrix | `residue_labels.npz`: identical | `residue_labels_bondfix.npz`: identical |
| columns that differ between packages | DISULFID, CROSSLNK only | |
| DISULFID positives / prevalence / rank | 33,467 / 0.02084 / 5 | 2,154 / 0.00134 / 19 |
| DISULFID cysteine fraction (background 1.33%) | 7.13% | 100% |
| DISULFID runs, mean run length | 586, 57.1 | 2,070, 1.0 |
| CROSSLNK positives / prevalence / Lys fraction | 993 / 6.2e-4 / 19.0% | 178 / 1.1e-4 / 82.0% |
| CARBOHYD (control) | 794, 93.95% Asn | 794, 93.95% Asn |
| concepts >= 1e-3 | 21 (same set) | 21 (same set) |

In the 5,000 subset MUTAGEN falls under the floor (6.9e-4), so 21 concepts are scored
instead of the full corpus's 22; this is the same in both packages. The flat-file excerpt
holds 1,101 DISULFID records (1,072 ranged, 29 single position) and 173 CROSSLNK records
(16 ranged, 157 single position). No bond position lies outside its protein. Two DISULFID
records have an unknown partner (`4..?`, `20..?`); both versions label only the known residue.

## SUMMARY.md

Written in the voice of the original analyst (style taken from the project README and paper:
"N concepts clear the 1e-3 floor", legacy keys empty, variation group as negative control).
The two summaries differ only where the numbers differ: the table order, the below-floor
list, the PTM bullet ("2.1%, fifth overall, more common than HELIX or BINDING" vs "0.13%,
19th overall, just above the floor") and conclusion 2 ("mainly through DISULFID, a large and
well-populated target" vs "through DISULFID and MOD_RES, two small concepts of similar size").
597 vs 598 words.

## Limits

- The opaque ids were drawn with `secrets`; B2977 = flawed, B4481 = clean (assignment arbitrary).
- AppleDouble `._*` files (com.apple.provenance only) are created by macOS on this exFAT drive
  in every folder; they carry no content.
- The downstream SAE scoring is not in the packages, so the consequence of the error on F1
  (the label means "inside a disulfide loop" rather than "bonded cysteine") is described in
  the key but cannot be checked by a reviewer inside the package.

## Changed after independent verification (2026-10-01)

See `P05_VERIFICATION.md`. The flawed SUMMARY.md PTM bullet no longer says "more common
than HELIX or BINDING" (now "at 2.1% of residues and fifth overall"; 591 words). Key facts
about fuzzy and isoform positions were corrected (77 of 40,063 concept records, not ~108).
