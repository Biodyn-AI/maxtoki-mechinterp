# Best single SAE feature per UniProt residue concept (ESM-2 650M, layer 16)

## What this analysis asks

Does a sparse autoencoder (SAE) trained on the layer-16 residual stream of the protein
language model ESM-2 650M contain single features that pick out UniProt residue concepts,
such as signal peptides, transmembrane segments or bonded cysteines?

For each concept, the analysis finds the best single SAE feature, measures how well it
predicts the concept's residues (F1), and compares that F1 with the F1 a random predictor
would get.

## Model and SAE

- Protein model: ESM-2 650M (`facebook/esm2_t33_650M_UR50D`), residual stream after
  layer 16 (1,280 dimensions per residue).
- SAE: TopK SAE with 5,120 features and k = 32. For each residue it keeps the 32 largest
  encoder pre-activations (set to 0 if negative) and sets every other feature to 0.
  It was trained for 8 epochs on 2.5M residues from 14,540 Swiss-Prot proteins. On
  held-out proteins it explains 74.6% of the variance of the layer-16 activations.
- None of the 600 proteins in this package were used to train the SAE.
- The SAE weights are not included. The package holds the SAE codes it produced.

## Data (`data/`)

600 Swiss-Prot proteins (lengths 35 to 1,012 residues), drawn at random from proteins
that were not used to train the SAE. 197,279 residues in total.

| file | contents |
|---|---|
| `sae_codes.npz` | `feature` (197,279 x 32, int16): indices of the 32 active SAE features for each residue. `value` (197,279 x 32, float16): their activations. `n_features` = 5120. All features not listed have activation 0. |
| `residue_labels.npz` | `labels` (197,279 x 31, bool): one column per UniProt feature key. `concepts` (31): the key names. `protein` (197,279, int16): protein id 0-599. `position` (197,279, int16): 0-based residue position in its protein. Rows are in the same order as `sae_codes.npz`. |
| `proteins.tsv` | protein id, UniProt accession, entry name, organism, length. |

Labels come from the Swiss-Prot flat file (FT lines). A residue is positive for a key when
it lies inside an annotated span of that key. DISULFID and CROSSLNK entries name bonded
residues, so for these two keys only the bonded residues are positive, not the stretch
between them.

Four keys have no positive residues in these 600 proteins: CA_BIND, METAL and NP_BIND
(older UniProt keys that are no longer used) and INTRAMEM.

**Negative control.** VARIANT marks positions where natural sequence variants have been
reported. That reflects which proteins people have studied, not a property of the residue,
so a model feature should not detect it.

## Method (`code/score_concepts.py`)

1. **Protein split.** The 600 proteins are split 50/50 at random (seed 42) into split A
   and split B. No protein contributes residues to both splits.
2. **Concepts scored.** A concept is scored if its prevalence over all 197,279 residues is
   at least 0.001 and it has at least 10 positive residues in each split.
3. **Features scored.** A feature is scored if it is nonzero on at least 50 residues of
   split A.
4. **Threshold.** For each feature and concept, 20 candidate thresholds are taken at the
   0.00, 0.05, ..., 0.95 quantiles of the feature's nonzero activations in split A. A
   residue is predicted positive when the activation is above the threshold. The threshold
   with the highest F1 on split A is kept.
5. **Scoring on split B.** At that threshold, F1, precision, recall and activation rate
   (fraction of split-B residues predicted positive) are computed on split B.
6. **Random baseline.** A predictor that fires at random on a fraction `a` of residues,
   for a concept with prevalence `p`, has expected F1 = 2ap / (a + p). The margin is F1
   minus this baseline, with `a` and `p` measured on split B.
7. **Best feature.** For each concept, the best feature is reported. A concept "passes"
   if its best feature has F1 >= 0.5 (the InterPLM headline threshold) and margin >= 0.1.

## How to run

From this folder:

```
python code/score_concepts.py
```

Needs numpy and scipy. CPU only. It takes under a minute and under 0.5 GB of memory.
The results are deterministic and overwrite the files in `outputs/`.

## Outputs (`outputs/`)

- `best_feature_per_concept.csv` and `.json`: one row per scored concept, sorted by F1.
  Columns: `concept`; `prevalence` (over all residues); `n_pos_a`, `n_pos_b` (positive
  residues per split); `feature` (index of the best feature); `f1`, `precision`, `recall`
  (split B); `threshold`; `activation_rate` (split B); `random_f1`; `f1_margin`;
  `f1_a` (the feature's F1 on split A at the same threshold); `passes`.
- `summary.json`: residue and feature counts, number of concepts passing, mean and median
  best F1, mean margin, and the number of concepts with margin >= 0.1.

The analyst's write-up of the results is in `SUMMARY.md`.
