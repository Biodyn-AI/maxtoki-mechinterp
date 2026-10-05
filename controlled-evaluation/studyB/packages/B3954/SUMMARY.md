# Results summary: single SAE features for UniProt residue concepts

**Question.** Does the layer-16 TopK SAE on ESM-2 650M contain single features that pick
out UniProt residue concepts?

**Setup.** 600 Swiss-Prot proteins not used to train the SAE (197,279 residues), split
50/50 by protein. 5,035 of the 5,120 features fire often enough to be scored, and 21
concepts are common enough to be scored. For each feature and concept, the activation
threshold is tuned on split A and F1 is measured on split B, so the reported F1 comes
from proteins that were not used to tune it. Each F1 is compared with the F1 a random
predictor would get at the same firing rate.

**Main result.** 6 of 21 concepts have a single feature with F1 >= 0.5 that beats its
random baseline by at least 0.1:

| concept | feature | F1 (split B) | random F1 |
|---|---|---|---|
| SIGNAL | 2936 | 0.833 | 0.004 |
| DISULFID | 1925 | 0.721 | 0.001 |
| REPEAT | 4672 | 0.626 | 0.013 |
| COILED | 904 | 0.519 | 0.008 |
| DNA_BIND | 3159 | 0.517 | 0.004 |
| ZN_FING | 624 | 0.501 | 0.003 |

Across all 21 concepts, the mean best F1 is 0.360 (median 0.300), and the mean margin
over random is +0.342. For 19 of 21 concepts the best feature is at least 0.1 above
random. Only DOMAIN (margin +0.092) and TURN (+0.082) fall short of that.

**Negative control.** VARIANT marks where natural variants have been reported, which is
not a property of the residue. Its best feature does not pass the gate (F1 0.358).

**Interpretation.** The layer-16 SAE has monosemantic features for a range of residue
concepts. They cover N-terminal signal peptides (SIGNAL), bonded cysteines (DISULFID),
repeats and coiled coils (REPEAT, COILED), and DNA-binding regions and zinc fingers
(DNA_BIND, ZN_FING). Most of these concepts are rare (0.1% to 1.3% of residues), so a
feature with F1 above 0.5 picks out a small, specific set of residues, not a broad
region. These six features are the candidates for causal tests (ablation or steering).

**Caveats.** Swiss-Prot annotation is incomplete, so unlabelled residues are not all true
negatives and F1 is a lower bound. Only one layer and one SAE were scored. Concepts below
1% prevalence have only 66 to 1,275 positive residues in a split, so their F1 values are
less precise than those of common concepts.
