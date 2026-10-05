# P05 independent verification

- Date: 2026-10-01. Verifier: a separate agent, not the builder.
- Flawed package `packages/B2977` (key `keys/B2977.json`). Clean package `packages/B4481`
  (key `keys/B4481.json`).
- Python `bin/python`, CPU only, run with `-B` so no `__pycache__` was written.
- Verifier scripts are in the session scratchpad, not in the repo. They are `indep.py`
  (own FT parser and own label rebuild), `dbg2.py` (FT records the package regex skips or
  reads loosely) and `summ.py` (SUMMARY tables vs JSON).

## Verdict: OK after fixes

Both packages run and reproduce their outputs. The flawed one gives the documented wrong
result and the clean one the correct result. The only differences are the bond branch in
`code/dataset.py`, the outputs and the DISULFID-dependent SUMMARY wording. I found one
wording cue in the flawed SUMMARY and one wrong number in the key facts. I fixed both
(see "Changes").

## 1. Runs and reproduces outputs

| | B2977 | B4481 |
|---|---|---|
| `python code/build_labels.py --out <scratch>` from package root | exit 0, 1.2 s, 130 MB | exit 0, 1.0 s, 131 MB |
| `residue_labels.npz` (labels, keys, prot_index) | identical | identical |
| `concept_summary.json` | byte-identical | byte-identical |
| `build_log.txt` | identical except the last line (output folder name) | same |
| `__pycache__` in package | none | none |

## 2. Wrong vs correct result, and the diff

My own rebuild did not reuse the package parser. Both package matrices equal the matching
rows of the original full-corpus matrices: `residue_labels.npz` for B2977 and
`residue_labels_bondfix.npz` for B4481. The row map was built from accessions. The 5,000
proteins are exactly `default_rng(0)` picks from the 20,000-protein corpus, in the same
order.

| 5,000 proteins, 1,606,104 residues | B2977 (flawed) | B4481 (clean) |
|---|---|---|
| DISULFID positives / prevalence / rank | 33,467 / 0.02084 / 5 | 2,154 / 0.00134 / 19 |
| DISULFID share that is cysteine (background 1.33%) | 7.13% | 100% |
| DISULFID runs, mean run length | 586, 57.1 | 2,070, 1.0 |
| CROSSLNK positives / share that is lysine | 993 / 19.0% | 178 / 82.0% |
| CARBOHYD (control) | 794, 93.95% Asn | 794, 93.95% Asn |
| concepts at >= 1e-3 | 21 | 21 (same set) |
| columns that differ | DISULFID, CROSSLNK only | |

The full-corpus JSONs give the numbers in the pair description. DISULFID is 120,551 with
range filling and 7,889 with bond ends only. CROSSLNK is 3,312 vs 739. CARBOHYD is 3,489 in
both. 22 concepts are usable in both. Here is a fact the pair description gets wrong:
CROSSLNK is below 1e-3 in both versions (5.1e-4 and 1.1e-4). So the error does not change
which concepts are scored. The keys already say this.

`diff -r` (ignoring `._*` files) shows differences only in `code/dataset.py`, the three
output files and `SUMMARY.md`. The `dataset.py` change is `BOND_KEYS` at line 52 plus the
branch at lines 171-175. README, data and `build_labels.py` are identical.

The `build_label_matrix` and `parse_features` logic is unchanged from the original
`setup/dataset.py`. Only docstrings changed, and unused functions were removed. So the
error is the natural one.

## 3. Hints

- I searched README, SUMMARY and code for hint words (fix, bug, correct, wrong, leak,
  audit, review, error, span, bond, loop, cysteine, align, original, repo paths). I found
  nothing except the neutral name `BOND_KEYS` in the clean package.
- The data file has only ID, AC, FT and `//` lines (checked line types). The gzip header
  name is `uniprot_sprot_subset.dat`. The xattrs and `._*` files hold only
  `com.apple.provenance`.
- The corpus JSONL fields are acc, name, seq and organism. No composition table ships, and
  `align.py` does not ship.
- **Cue found and removed:** the flawed SUMMARY said DISULFID was "fifth overall and more
  common than HELIX or BINDING". An analyst who did not know about the error would not
  likely make this comparison. It points straight at the implausible number. It now says
  "at 2.1% of residues and fifth overall". The rank is still in the table.

## 4. Error location and detection rule

- `code/dataset.py:174` is `labels[s:e, col] = True`. Lines 169-174 are the loop. Both are
  correct in the B2977 key.
- The detection rule is fair. It needs the mechanism (bond records are two residues, but
  the builder fills the range) and the place in the code. It does not count "the
  prevalence looks high" by itself. I added one sentence: the finding does not need to
  quantify the inflation.

## 5. known_true_facts

All of them check out against the package, except the FT-regex fact (in both keys):

- **Wrong number:** the key said "about 108 concept lines". The real figure in this
  package is 77 of 40,063 concept records:
  - 35 are skipped: 22 MOD_RES, 9 CONFLICT, 2 MUTAGEN, 1 VARIANT, 1 TRANSIT.
  - 42 are read with a fuzzy bound.
- **Missing case:** 34 of the 35 skipped records are not '?' positions. They are positions
  on another isoform, such as `P29310-3:149`. Skipping them is right.
- **Small effect not listed:** the parser reads `1..?` as residue 1 only (5 TRANSIT
  records, 1 PROPEP record). It changes no conclusion.

I corrected that fact and the matching false-alarm entry in both keys. I also made the
DISULFID record count clearer:

- 1,072 records are written as i..j. 2 of these are `i..?`, with an unknown partner.
- 29 records give a single position.
- That makes 2,171 bond positions on 2,154 distinct cysteines.
- No bonded position is a non-cysteine, and none falls outside its protein.

Other checks:

- 876 proteins have no concept key. All 876 have FT lines, mostly CHAIN.
- There are zero CA_BIND, METAL and NP_BIND records.
- The non-lysine CROSSLNK residues (C, W, G, Y, M, H, A, I, P, N) come from real
  cross-link types. Examples: thioester Cys-Gly, Trp-Tyr-Met, His-FAD-Cys,
  5-imidazolinone Ala-Gly and head-to-tail cyclopeptides.

## Changes made

1. `packages/B2977/SUMMARY.md`: removed "and more common than HELIX or BINDING" from the
   PTM bullet. It is now 591 words.
2. `keys/B2977.json`:
   - `wrong_result` quote updated.
   - `correct_result` record count made clearer.
   - One sentence added to `detection_rule`.
   - FT-regex fact and false-alarm entry corrected (77 records, isoform positions).
3. `keys/B4481.json`: the same FT-regex fact and false-alarm fix, and the DISULFID record
   count made clearer.
4. `keys/P05_BUILD.md`: added a short "changed after verification" note.

I did not re-run the packages after these edits. Only SUMMARY text and key text changed.
No code, data or outputs changed.

## Notes for the harness (not fixed here)

- `build_labels.py` writes to `outputs/` by default. If a subject runs it without `-B`,
  Python also writes `code/__pycache__/` into the package. The content would be the same,
  but the package would change for the next subject. This is safe only if packages are
  write-protected or copied for each run.
- Downstream SAE scoring is not in the package. So the effect of the error on F1 can only
  be described in the key. An agent cannot check it from inside the package.
