# Disulfide-bonded cysteine pairs and ESM-2 attention heads

## What this analysis does

A disulfide bond joins two cysteines. In a protein with several cysteines, only some pairs are bonded.
The question is whether the attention maps of the protein language model ESM-2 (650M parameters,
33 layers x 20 heads = 660 heads) can tell the bonded pairs apart from the other cysteine pairs of the
same protein.

Every pair of cysteines (i < j) inside one protein is a candidate. Each head gives each candidate a
score from its attention map. A score is judged by average precision (AP): how well it ranks the bonded
pairs above the non-bonded pairs. The no-skill AP is the base rate (the fraction of pairs that are
bonded). The best head is chosen on one set of proteins (select) and its AP is reported on a different
set of proteins (test). The mean of all 660 heads and the sequence separation of the pair are scored
the same way, as baselines.

## Data (`data/`)

Proteins come from a corpus of 18,176 reviewed UniProtKB/Swiss-Prot entries, split by protein
(before this analysis) into disjoint groups. This analysis uses the select group and the test group.

Bond labels come from the UniProt `FT DISULFID i..j` records of each entry (1-based, converted to
0-based positions). Records with a single position (bonds to another chain) are not used. Both ends of
every record are cysteines in the sequence. A pair without a record is labelled not bonded.

A protein is kept if it has at least 2 cysteines, at least 1 annotated intrachain bond, at most 1,022
residues (the ESM-2 input limit) and at most 40 cysteines. Two test proteins were left out by the last
rule (64 and 41 cysteines). No other protein was left out by the length or cysteine-count rules.

| split | proteins | candidate pairs | bonded pairs |
|---|---|---|---|
| select | 113 | 4,419 | 313 |
| test | 123 | 7,463 | 366 |

| file | content |
|---|---|
| `pairs_select.npz`, `pairs_test.npz` | one row per candidate pair: `protein` (UniProt accession), `protein_length`, `pos_i`, `pos_j` (0-based, i < j), `bonded` (bool). Rows are grouped by protein. |
| `attention_scores.npz` | `S_select` (661 x 4,419) and `S_test` (661 x 7,463), float32; `n_layers` = 33, `n_head` = 20. Row `layer * 20 + head` is that head's score for each pair (column k = row k of the pairs file). Row 660 is the mean of the 660 head rows. |
| `proteins.tsv` | one row per protein: accession, split, length, number of cysteines, number of annotated intrachain bonds, sequence. |

### How the attention scores were made

`code/extract_attention.py` holds the code. Each protein was run on its own through
`facebook/esm2_t33_650M_UR50D` (revision `08e4846e537177426273712802403f7ba8261b6c`), bfloat16, eager
attention, once, on an Apple-silicon GPU. For each head: the attention map over the residues (the
`<cls>` and `<eos>` tokens dropped) is made symmetric, S = (A + A^T) / 2, then corrected with the
average product correction (APC), C = S - outer(row sums, column sums) / total. The score of pair
(i, j) is C[i, j]. You do not need to run this step; it needs the model weights and several GB of memory.

## Code (`code/`)

- `disulfide_attention.py`: the analysis. Metric helpers (AP with ties handled as groups, precision at
  k averaged over tied orders, within-protein label permutation, protein-level bootstrap), the two
  scorers (attention: 661 candidates; sequence separation: |i - j| and |i - j| / length, both signs,
  4 candidates), and the report.
- `extract_attention.py`: how `attention_scores.npz` was made (see above).

## Method

1. For each candidate of a scorer, AP on the select pairs and on the test pairs.
2. The candidate with the highest select AP is chosen. Its test AP is the reported number.
3. The highest test AP of any candidate is also reported, next to the expected highest test AP under a
   null (bond labels shuffled within each test protein, so each protein keeps its number of bonds; 256
   candidates sampled, best-of-N expectation over 200 draws).
4. 95% interval for the chosen candidate's test AP: bootstrap over test proteins (200 resamples).
5. Per-protein view on the test proteins for the chosen candidate: mean per-protein AP, precision of
   the top-ranked pair (P@1), precision in the top k pairs where k = that protein's bond count.
6. Attention only: per-layer summary and the 10 heads with the highest select AP.

Heads are named `L{layer}H{head}`, both counted from 0 (layer 32 is the last layer).

## How to run

From the package root:

```
python code/disulfide_attention.py              # writes to outputs/
python code/disulfide_attention.py --out DIR    # writes to another folder
python code/disulfide_attention.py --selftest   # checks the metric helpers (needs scikit-learn)
```

Needs numpy. CPU only, about 10 seconds, under 0.2 GB of memory. The result is deterministic (seed 42).

## Outputs (`outputs/`)

- `results.json`: data counts and base rates; for each scorer the chosen candidate, its select and test
  AP, the bootstrap interval, the test maximum and its null expectation, the number of candidates with
  test AP at least 2x the base rate, and the per-protein view; the mean of all heads as a single
  candidate without selection; the per-layer summary; the top 10 heads by select AP.
- `head_ap.csv`: select and test AP for all 661 attention candidates.
- `run_log.txt`: the printed log.

`SUMMARY.md` is the analyst's write-up of the result.
