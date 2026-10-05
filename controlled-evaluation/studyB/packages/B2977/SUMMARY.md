# Results summary - residue concept labels

## Headline

- Corpus: 5,000 Swiss-Prot proteins, 1,606,104 residues. 4,124 of the 5,000 proteins
  (82%) carry at least one of the 31 concept keys.
- Label matrix: 1,606,104 x 31 booleans in `outputs/residue_labels.npz`, rows in corpus
  order.
- **21 of the 31 concepts clear the 1e-3 prevalence floor** and go forward to SAE
  scoring. Prevalence runs from 7.3e-5 (LIPID) to 0.140 (DOMAIN).
- Three keys (CA_BIND, METAL, NP_BIND) have no positive residues. These are legacy keys
  that the current flat-file format no longer uses, so they are empty, not missing.

## Concepts that will be scored

| # | concept | group | positive residues | prevalence |
|---|---|---|---:|---:|
| 1 | DOMAIN | domain | 224,224 | 0.13961 |
| 2 | REGION | domain | 135,838 | 0.08458 |
| 3 | TOPO_DOM | topology | 80,723 | 0.05026 |
| 4 | TRANSMEM | topology | 68,839 | 0.04286 |
| 5 | DISULFID | ptm | 33,467 | 0.02084 |
| 6 | HELIX | secondary_structure | 31,592 | 0.01967 |
| 7 | COMPBIAS | domain | 31,524 | 0.01963 |
| 8 | REPEAT | domain | 22,554 | 0.01404 |
| 9 | BINDING | binding | 19,010 | 0.01184 |
| 10 | STRAND | secondary_structure | 18,530 | 0.01154 |
| 11 | COILED | domain | 10,953 | 0.00682 |
| 12 | SIGNAL | targeting | 9,302 | 0.00579 |
| 13 | ZN_FING | binding | 6,187 | 0.00385 |
| 14 | DNA_BIND | binding | 5,220 | 0.00325 |
| 15 | PROPEP | targeting | 4,913 | 0.00306 |
| 16 | MOTIF | domain | 3,262 | 0.00203 |
| 17 | VARIANT | variation | 3,243 | 0.00202 |
| 18 | TRANSIT | targeting | 2,946 | 0.00183 |
| 19 | TURN | secondary_structure | 2,757 | 0.00172 |
| 20 | CONFLICT | variation | 1,860 | 0.00116 |
| 21 | MOD_RES | ptm | 1,802 | 0.00112 |

Below the floor (not scored): ACT_SITE 1,534 (9.6e-4), MUTAGEN 1,109 (6.9e-4),
CROSSLNK 993 (6.2e-4), CARBOHYD 794 (4.9e-4), SITE 577 (3.6e-4), INTRAMEM 263
(1.6e-4), LIPID 117 (7.3e-5).

## By group

- **Domain architecture** is the bulk of the labelled residues. All six keys clear the
  floor. DOMAIN alone covers 14% of residues.
- **Topology:** TOPO_DOM (5.0%) and TRANSMEM (4.3%) are scored. INTRAMEM is too rare.
- **Secondary structure:** HELIX, STRAND and TURN are all scored.
- **Targeting:** SIGNAL, PROPEP and TRANSIT are all scored.
- **Binding:** BINDING, ZN_FING and DNA_BIND are scored.
- **PTM:** two keys are scored. DISULFID (disulfide bonds) is the largest PTM concept
  at 2.1% of residues and fifth overall. MOD_RES is just above the floor (0.11%).
  CROSSLNK, CARBOHYD and LIPID fall below it.
- **Catalytic and other sites:** ACT_SITE sits just under the floor (9.6e-4) and SITE
  is well under it, so neither is scored. Catalytic residues are therefore not tested.
- **Variation (negative control):** VARIANT and CONFLICT are scored; MUTAGEN is not.

## Conclusions

1. The label set is usable: 21 concepts have enough positives for stable per-residue
   F1. The smallest scored concept, MOD_RES, still has 1,802 positive residues.
2. Every group except catalytic/sites contributes at least one scored concept. PTM
   biology enters the scoring mainly through DISULFID, a large and well-populated
   target.
3. The negative control is in place, with two of the three variation keys scored.

Next step: score SAE features from each layer against these 21 concepts.
