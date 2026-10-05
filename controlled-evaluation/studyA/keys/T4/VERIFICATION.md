# T4 verification log

Date: 2026-10-01. CPU only. One independent pass over the T4 packages and the T4 answer key.
I did not reuse `reference.py`. All my numbers come from my own code (in `verification/`), which
reads only the package `data/` folder.

## Verdict

**OK after fixes.** The packages are clean and the data are correct. All 57 key numbers match my
own numbers within their tolerances. The key verdict follows from the evidence. I changed only
wording in five text fields of `key.json`. No key number and no package file was changed.

One study-level problem is still open (not a package or key problem). `protocol/PREREGISTRATION.md`
section 3 describes T4 as a developmental-ordering task. These packages test cross-model alignment.
The pre-registration is frozen, so I did not edit it. It needs a dated amendment before Study A runs.

## 1. The two packages differ only where they should

- `diff -r` (ignoring macOS `._` files) shows only two differences:
  - `T4-contract/BRIEF.md` has one extra sentence: "A method specification for this kind of
    analysis is provided in contract/SPEC.md; follow it." This is the same sentence T1 and T3 use.
  - `T4-contract/contract/SPEC.md` exists only in the contract package.
- All data files and the PDF are byte-identical across the two packages.
- `SPEC.md` is byte-identical to `pipelines/topology-geometry-141-hypotheses.md` (sha256 `43731ef3…`,
  file date 2026-04-15 20:58).
- `methods/source_method_paper.pdf` is byte-identical to `references/2602.22289_Kendiukhov_topology_141_hypotheses.pdf`.

**Spec choice is right.** The topology-141 spec's Phase 4 describes the cross-model CCA, Procrustes
top-1 and distance-correlation tests that the deployed scripts ran. The spectral-geometry spec has
only a short raw-cosine check (its Phase 9b) with no CCA or Procrustes.

## 2. No hints in the packages

- I read `BRIEF.md` (both) and `data/README.md` in full.
- I searched both for: fix, bug, v2, retract, correct, wrong, leak, confound, baseline, null, chance,
  artefact, artifact, expect, should find, GATA1, audit, review, error, deploy, mistake, trap,
  enumerate, position, order, verify, check, independent, overlap, any decimal number, and repo
  paths (`biomi`, `projects/`, `runs/`, `repos/`).
- Every hit is neutral:
  - "V2" is part of the model name (Geneformer V2-316M) or TRRUST v2.
  - "chance" appears only in the verdict option text.
  - "position" refers to Llama and BERT position vectors.
  - "order" refers to the report sections, the rank-value encoding or the panel row order.
- No outcome statement, no number from the source report, and no repo path appears in the brief or the README.
- The brief names the Python interpreter path. That path is required by the protocol and is the same in T1 and T3.
- `SPEC.md` is the treatment and is verbatim, as the protocol requires. It contains "Audit" in its
  title, "bug" in two pitfalls, `repos/…` paths, and the source paper's own expected values
  (r = 0.80, top-1 72%). These all predate the deployment. None mentions MaxToki, the scGPT
  vocabulary or this analysis's outcome. I left it unchanged.
- The PDF is the original arXiv paper. It does not mention MaxToki.
- The 52 `._*` files hold only `com.apple.provenance` ("This resource fork intentionally left blank").

## 3. Data files load and match their descriptions

Checked by `verification/check_data.py` (output `check_data.json`):

| Check | Result |
|---|---|
| `MANIFEST.sha256` | lists every data file; 0 hash mismatches |
| Table shapes / dtype | MaxToki (20275, 1232), Geneformer (20275, 1152), scGPT (60697, 512); all float32, all finite |
| Tables vs original checkpoints | max difference 0.0 for all three tables and both LayerNorm vectors; `eps` = 1e-5 |
| MaxToki dictionary | 23,277 entries = 20,271 Ensembl + 6 special + 3,000 numeric (-1500 to 1499); every id ≥ 20,275 is a numeric token, `<boq>` or `<eoq>`, as the README says |
| Geneformer dictionary | 20,275 entries = 20,271 Ensembl + 4 special; equals the release `.pkl`; same id as MaxToki for all 20,271 genes |
| scGPT `vocab.json` | 60,697 entries; ids are a permutation of 0…60,696; key position equals id for only 1 key; no case collisions |
| `ensembl_symbol.csv` | 63,675 rows; one-to-one; equals the release `.pkl` |
| Panels | 382 / 350 / 380 unique genes; same genes and order as the deployed `gene_features.csv`; all in MaxToki and Geneformer; all symbols found in scGPT `vocab.json` (exact case); symbols equal the deployed symbols for all 1,112 rows |
| Panel overlaps | lung-immune 75, lung-external 168, immune-external 77, all three 36, union 828 |
| Panel README text | matches `phase0_extract.py` (priority TF 80 > target 150 > STRING fill to 350 > markers; immune reused 2,000 cells, top 1,500 HVGs, capped at 350) |

Cross-check against the source run's own table (`outputs/v2_crossmodel/embeddings_subset.npz`):
my scGPT rows by token id match its "fixed" rows (max diff 9e-7). My rows by key position match its
"deployed" rows (max diff 8e-7). Under the key-position rule, 0 of 382 / 350 / 380 genes get their own row.
So a subject can reproduce both the right and the deployed lookup from the package alone.

## 4. Key numbers re-derived with my own code

Code: `verification/verify_metrics.py`. Choices:
- Exact SVD PCA and closed-form CCA (QR + SVD), plus sklearn's PCA(random_state=42) and CCA(10) for the deployed setting.
- My own fold splits and seeds.
- Pearson bootstrap with weights c_g·c_h, which drops pairs made of two copies of one gene. This matched direct indexing to 3e-16.
- In-sample stages used SVD PCA. Held-out and OOB stages used the top-30 eigenvectors (`scipy.linalg.eigh`) for speed. These give the same subspace, so the metrics are unchanged.

**Result: 57 of 57 key numbers are within tolerance** (`verification/compare.txt`). The largest gaps:

| Key number | Key | Mine | Tolerance |
|---|---|---|---|
| Held-out CCA, Geneformer, immune | 0.558 | 0.573 | 0.03 |
| Held-out CCA, scGPT, immune | 0.481 | 0.495 | 0.03 |
| Held-out CCA, Geneformer, lung | 0.604 | 0.615 | 0.03 |
| Held-out top-1, scGPT, lung | 0.059 | 0.049 | 0.03 |
| Held-out top-1, Geneformer, immune | 0.158 | 0.169 | 0.03 |

All in-sample numbers (Pearson, CCA, top-1, chance levels, wrong-lookup values) agree to 0.002 or better.
The exact-PCA and raw-scGPT variants also fall inside every tolerance. For example, exact in-sample
top-1 for Geneformer on immune is 0.514 against the key's 0.466 ± 0.06.

Paired Geneformer minus scGPT, my values:

| Panel | Pearson diff [95% CI] | Held-out CCA diff, OOB [95% CI] | Held-out top-1 diff, OOB [95% CI] |
|---|---|---|---|
| lung | 0.132 [0.117, 0.147] | 0.112 [0.052, 0.172] (500 draws) | 9.2 pts [2.8, 15.9] |
| immune | 0.137 [0.121, 0.153] | 0.106 [0.047, 0.174] (450 draws) | 8.9 pts [1.6, 16.1] |
| external lung | 0.104 [0.089, 0.119] | 0.104 [0.048, 0.158] (250 draws) | 6.8 pts [0.9, 12.7] |

Chance levels, my values:
- Pearson: mean 0.000, SD 0.004.
- In-sample CCA: 0.411 / 0.429 / 0.412. This is the same for the scGPT table.
- In-sample top-1 with the rotation refitted: 6.7 / 8.0 / 7.2%. Without refitting (the deployed null) it is 0.3%.
- Held-out CCA: about 0.00 (95th percentile 0.03-0.05). Held-out top-1: 0.2-0.3%.
- A with-replacement bootstrap of in-sample CCA (Geneformer lung) gives 0.910 [0.891, 0.929]. That interval does not contain the observed 0.783. This confirms the trap.

Extras, also matching the key: union of panels Pearson 0.373 vs 0.238. On 2,000 random shared genes
(my own draw) Pearson is 0.430 vs 0.219; the key's NOTES give 0.427 vs 0.218.
Sanity pairs (cosine with the token-id rule vs the key-position rule): CD3D-CD3E 0.62 vs 0.27;
HBA1-HBB 0.57 vs -0.05; RPL3-RPL5 0.47 vs 0.10. These match the key.

## 5. The verdict follows from the evidence

- scGPT is far above chance on every metric and panel. Example: Pearson 0.26-0.28 against chance 0.00 (SD 0.004).
- Geneformer is above scGPT on every metric. The paired CIs exclude 0 for Pearson and held-out CCA on all three panels.
- This matches the corrected source report and its verification notes ("Geneformer aligns with
  MaxToki better than scGPT does on every measure"; "scGPT also agrees with MaxToki clearly above chance").
- I also checked whether another reasonable metric would give "about as well". It would not (`verification/robust.json`, `robust_metrics.py`):

| Metric (lung / immune / external lung) | Geneformer | scGPT | Chance |
|---|---|---|---|
| Spearman of cosine matrices | 0.35 / 0.37 / 0.35 | 0.23 / 0.24 / 0.25 | about 0 |
| Linear CKA (full vectors) | 0.73 / 0.72 / 0.71 | 0.53 / 0.56 / 0.56 | 0.39-0.42 |
| In-sample CCA with PCA-20 (the spec's PCA size) | 0.69 / 0.66 / 0.66 | 0.61 / 0.62 / 0.61 | 0.30-0.31 |
| Top-10 neighbour overlap | 0.21 / 0.19 / 0.18 | 0.16 / 0.16 / 0.16 | 0.03 |

The gap is smallest for local neighbour overlap. On external lung it is 0.019, with a paired per-gene
CI of [0.004, 0.034]. That CI is still above 0. So the key's "no alternative verdict" is defensible.
I added this to the key's note for graders.

## 6. Traps are real; handling text made fairer

All six traps are real and their numbers check out:
- the scGPT row lookup;
- the in-sample CCA chance level of about 0.41;
- the in-sample top-1 null, which must refit the rotation;
- the resampling unit and interval validity;
- the panel overlap;
- the scope and attribution limits.

I changed wording in four places where a grader could have marked a sound deliverable down.

## Changes made

All changes were made in `studyA/build/T4/make_key.py` (originals saved in my scratchpad). Then
`make_key.py` was re-run. That re-run changed only these five fields of `key.json`; the 57 key
numbers, the verdict, the traps' ids and descriptions, and the false-statement checks are unchanged.

1. `acceptable_alternatives_note`: added the metric-robustness result from section 5.
2. Trap `insample_top1_null`, `correct_handling`: "Geneformer 15-18%" became "about 14-19%".
   The held-out values are 14.0-18.7% in my run and 14.6-17.9% in the source report.
3. Trap `resampling_unit_and_interval_validity`, `mishandling`: before, it listed "judging the gap only
   by whether two unpaired intervals overlap" as a mishandling. That would penalise a correct, conservative
   check: the unpaired Pearson intervals do not overlap at all. Now it lists "no uncertainty at all for the gap" and
   "calling the gap uncertain only because two unpaired intervals overlap". It also says that
   non-overlapping unpaired intervals are not a mishandling.
4. Trap `panels_not_independent`, `correct_handling`: added that a report which never treats the panels
   as independent replications and never combines their p-values counts as handled.
5. Trap `scope_and_attribution`, `correct_handling`: the requirement is now "do not attribute the gap to one
   difference". Saying that the design cannot tell which difference explains the gap is now marked as best,
   not required.

New `key.json` sha256: `e69a98329f762e18d69b763cc7a648d8d9a80382246aea4276734cc657e71c92`
(was `107bc090…0317dbb`). I did not write it to `protocol/FREEZE_LOG.md`.

Added: `verification/` (my scripts and their JSON outputs). `NOTES.md`, `reference.py`,
`reference_output.json`, `parts/` and both packages are unchanged.

## Open issues (not fixed)

1. **Pre-registration mismatch** (see Verdict). This needs an amendment before Study A runs.
2. **Package folder names.** The folders are named `…/maxtoki-framework-eval/studyA/tasks/T4-paper` and
   `…/T4-contract`. A subject told to read "this package folder" sees its arm and the study name in the path.
   This is the same for T1 and T3. The harness could copy each package to a neutral path before a run.
3. **`._` files.** These exist because the drive is exFAT. They are harmless but visible in a listing.
4. **Draw counts.** I used fewer draws than the key in some places:
   - OOB: 450 draws for immune and 250 for external lung, against the key's 500.
   - Held-out chance: 30 permutations, against the key's 200.
   - The reason: the machine load average was 80-170, and each command had to finish in 9 minutes.
   - My intervals agree with the key's to about 0.01.
5. **Not done.** I did not run a trial subject, any layer-wise test, or any check of the source run's
   earlier phase outputs beyond `embeddings_subset.npz`.

## Summary in plain words

- The two packages differ only by the spec file and one sentence. Nothing in them gives away the answer.
- The data are exact copies of the model tables and dictionaries. They match the README.
- My own code reproduced all 57 key numbers within tolerance.
- The verdict holds: scGPT is aligned with MaxToki well above chance, and clearly less than Geneformer.
  This is true on every metric I tried, not just the ones the key uses.
- I reworded four trap descriptions and one key note, so that careful but different handling is not marked as wrong.
- The pre-registration still describes a different T4. That must be fixed at the protocol level.
