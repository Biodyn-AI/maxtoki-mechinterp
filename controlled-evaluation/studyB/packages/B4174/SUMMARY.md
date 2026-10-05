# Disulfide-bonded cysteine pairs in ESM-2 attention

**Purpose.** Can ESM-2 (650M) tell which cysteine pairs in a protein are disulfide-bonded? Every
cysteine pair in 236 Swiss-Prot proteins with at least one annotated intrachain bond is scored by each
of the 660 attention heads (symmetrised, APC-corrected attention). The best head is chosen on the
select proteins and scored on the test proteins, against two baselines: sequence separation and the
mean of all heads.

## Data

| | select | test |
|---|---|---|
| proteins | 113 | 123 |
| candidate cysteine pairs | 4,419 | 7,463 |
| bonded pairs | 313 | 366 |
| base rate (no-skill AP) | 0.071 | 0.049 |

## Results on the test proteins

| scorer | candidate | test AP (95% CI) | x base rate | P@1 per protein |
|---|---|---|---|---|
| no skill | - | 0.049 | 1.0 | - |
| sequence separation | shorter first | 0.113 (0.098-0.134) | 2.3 | 0.207 |
| mean of all 660 heads | no selection | 0.141 (0.114-0.177) | 2.9 | 0.317 |
| best head, chosen on select | L32H13 | **0.718** (0.664-0.782) | 14.6 | 0.805 |

L32H13 had select AP 0.742. The highest test AP of any candidate is 0.745 (L29H18); the expected
highest of 661 candidates under the within-protein null is 0.087. The five heads with the highest
select AP are all in late layers: L32H13, L29H18 (test 0.745), L31H17 (0.639), L30H1 (0.595) and
L23H6 (0.527). 192 of the 661 candidates reach at least 2x the base rate on the test proteins.

## Conclusions

1. Attention ranks bonded cysteine pairs far above chance. Head L32H13 puts a bonded pair at the top
   in 80% of test proteins, with an AP 14.6 times the base rate. That is far above sequence
   separation (0.113) and the mean head (0.141), and it held from select to test proteins
   (0.742 -> 0.718).
2. Other late-layer heads also score high on the test proteins (L29H18 0.745, L31H17 0.639,
   L30H1 0.595), so the signal is spread over a group of heads.
3. This is a correlational readout. We compared attention weights with annotations; no head was
   ablated or patched. So it does not show that the model uses these heads to work out bonding or
   that it has learned bonding chemistry. Attention follows residue contacts in general (Rao et al.
   2020), and bonded cysteines are in contact. We did not test whether these heads prefer bonded
   pairs over other close pairs.
4. Next step: ablate L32H13 and the other top heads and measure what changes, and score the same
   heads on non-cysteine contacts from structures.
