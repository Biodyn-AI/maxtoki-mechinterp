# Summary: does the DoRothEA sign predict the direction of the CRISPRi response?

## Question

DoRothEA marks some TF-target edges as activating or inhibiting. If the label is right, knocking down
an activator should lower its target, and knocking down a repressor should raise it. We tested this on
the Replogle et al. 2022 CRISPRi Perturb-seq screens in K562 (genome-wide) and RPE1.

## What was scored

- K562: 8,358 signed edges with a measured response (429 TFs; 6,903 activation and 1,455 inhibition
  edges).
- RPE1: 1,866 signed edges (58 TFs; 1,622 activation and 244 inhibition edges).
- Three strata by response size: all edges, top 50% of |dE|, top 10% of |dE|. The top 10% means
  |dE| >= 0.166 in K562 and >= 0.453 in RPE1 (z units).

## Results

Main statistic: the odds ratio (OR) of "target goes up" for inhibition edges versus activation edges.
OR = 1 means no association. OR > 1 means the database sign and the response agree.

| line | stratum | edges (TFs) | inhibition edges | frac. up, activation | frac. up, inhibition | OR | Fisher p | within-TF shuffle: null median, p | TF-resampled 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| K562 | all | 8,358 (429) | 1,455 | 0.501 | 0.498 | 0.99 | 0.89 | 1.04, 0.88 | 0.87-1.13 |
| K562 | top 50% | 4,179 (333) | 678 | 0.502 | 0.524 | 1.09 | 0.31 | 1.14, 0.73 | 0.90-1.31 |
| K562 | top 10% | 836 (151) | 114 | 0.504 | 0.702 | 2.31 | 7.7e-5 | 1.76, 0.065 | 1.51-3.57 |
| RPE1 | all | 1,866 (58) | 244 | 0.417 | 0.500 | 1.40 | 0.015 | 0.97, 0.0035 | 0.79-2.42 |
| RPE1 | top 50% | 933 (46) | 130 | 0.362 | 0.431 | 1.33 | 0.14 | 0.91, 0.018 | 0.56-2.57 |
| RPE1 | top 10% | 187 (26) | 24 | 0.374 | 0.708 | 4.06 | 0.0033 | 1.47, 0.0025 | 0.71-14.25 |

Other numbers, top 10% stratum:

- Shuffling the signs across all edges gives a null OR near 1.0 (p = 0.0005 in K562, 0.0025 in RPE1).
- Balanced accuracy of the database sign (chance 0.5): 0.599 [0.553, 0.645] in K562 and
  0.667 [0.564, 0.763] in RPE1. "Always predict down" scores 0.397 and 0.459; a random sign scores
  0.453 and 0.406. Balanced over the observed up / down classes instead, the database sign scores
  0.547 and 0.577.
- DoRothEA level A edges: balanced accuracy 0.629 in K562 (267 edges, 20 of them inhibition, 60 TFs)
  and 0.863 in RPE1 (69 edges, 7 inhibition, 15 TFs). Level D: 0.562 and 0.564.
- In RPE1, MYC supplies 82 of the 187 top-10% edges and 9 of the 24 inhibition edges.

All numbers are in `outputs/run_log.txt` and `outputs/sign_*.json`; per-edge values are in
`outputs/edges_*.csv`.

## Interpretation

1. The database sign agrees with the direction of the strongest responses, but only modestly. In the
   top 10%, inhibition targets go up 70% of the time in both lines, against 50% (K562) and 37% (RPE1)
   for activation targets: OR 2.31 and 4.06.
2. Below the top decile the agreement fades. Across all edges K562 shows none (OR 0.99). RPE1 shows a
   weak one (OR 1.40), but its TF-resampled interval (0.79-2.42) includes 1.
3. The edges are grouped by TF, and this matters. In K562, shuffling signs only within each TF already
   gives a median OR of 1.76; the observed 2.31 beats it with p = 0.065. So most of the K562 agreement
   comes from differences between TFs (TFs with more inhibition labels have more targets going up),
   and edge-by-edge information is not clearly shown. RPE1 passes the within-TF test (p = 0.0025), but
   rests on 26 TFs and 24 inhibition edges, with MYC supplying 44% of the edges; the TF-resampled
   interval is 0.71-14.25.
4. A knockdown measures the total effect on a target: direct regulation plus everything downstream.
   Agreement in sign cannot show that an edge is direct, or which edges are real.
5. Level A edges score higher than level D, but on only 20 and 7 inhibition edges.
6. Both screens come from one lab and one CRISPRi Perturb-seq platform: one platform in two cell
   lines, not an independent replication.

## Conclusion

DoRothEA's activation/inhibition labels are consistent with the direction of the strongest CRISPRi
responses: a modest positive association in the top 10% of responses in two cell lines (OR 2.3 and
4.1), none across all K562 edges, and fragile once edges are grouped by TF. The test concerns total
knockdown effects on one platform. It does not show that DoRothEA edges are direct regulatory
interactions.
