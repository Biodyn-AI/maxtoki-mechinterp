# Results summary: single SAE features for UniProt residue concepts

**Question.** Does the layer-16 TopK SAE on ESM-2 650M contain single features that pick
out UniProt residue concepts?

**Setup.** 600 Swiss-Prot proteins not used to train the SAE (197,279 residues), split
50/50 by protein. 5,035 of the 5,120 features fire often enough to be scored, and 21
concepts are common enough to be scored. For each concept, both the activation threshold
and the best feature are chosen by F1 on split A, and F1 is then measured on split B, so
the reported F1 comes from proteins that were not used to choose either. Each F1 is
compared with the F1 a random predictor would get at the same firing rate.

**Main result.** 2 of 21 concepts have a single feature with F1 >= 0.5 that beats its
random baseline by at least 0.1:

| concept | feature | F1 (split B) | random F1 |
|---|---|---|---|
| SIGNAL | 2936 | 0.833 | 0.004 |
| DISULFID | 1925 | 0.721 | 0.001 |

Across all 21 concepts, the mean best F1 is 0.162 (median 0.031), and the mean margin
over random is +0.143. For 8 of 21 concepts the best feature is at least 0.1 above
random. For 10 concepts (REPEAT, COILED, DNA_BIND, HELIX, MOTIF, MUTAGEN, PROPEP, STRAND,
TURN, VARIANT) the feature chosen on split A scores no better than random on split B
(F1 below 0.004, margin 0 or below).

**Negative control.** VARIANT marks where natural variants have been reported, which is
not a property of the residue. Its best feature scores F1 0.000 on split B, as a
negative control should.

**Interpretation.** The layer-16 SAE has clear single-feature detectors for two concepts:
N-terminal signal peptides (SIGNAL, F1 0.833) and bonded cysteines (DISULFID, F1 0.721).
Six more concepts (COMPBIAS, TRANSMEM, ZN_FING, TRANSIT, REGION, BINDING) have a best
feature well above random, but with F1 0.14 to 0.31, so it covers only part of the
concept. For the other 13 concepts, the best feature is less than 0.1 above random on
split B. In this sample, single features that generalise across proteins exist for a few
concepts, not for most. The SIGNAL and DISULFID features are the candidates for
causal tests (ablation or steering).

**Caveats.** Swiss-Prot annotation is incomplete, so unlabelled residues are not all true
negatives and F1 is a lower bound. Only one layer and one SAE were scored. Concepts below
1% prevalence have only 66 to 1,275 positive residues in a split, so their F1 values are
less precise than those of common concepts. A concept near 0 here may still have a useful
feature that this sample could not identify.
