# T3 verification (independent check of the packages and the key)

Date: 2026-10-01. CPU only. I used my own code, not `reference.py`. My scripts are in
`verification_scripts/`. They read only the package data (`tasks/T3-paper/data/`). Run
`sel.py` first (it writes `sel.npz` to the working folder, about 45 s); the other scripts read it.

**Verdict: OK after fixes.** The packages needed no change. The key verdict and all 51 key
numbers stand. I made four wording fixes in the key (traps and false-statement checks) and
added one note to `NOTES.md`. All changes are listed in section 7.

## 1. The two packages differ only where they should

`diff -rq -x '._*' T3-paper T3-contract` shows two differences only:

- `BRIEF.md`: one added sentence, "A method specification for this kind of analysis is
  provided in contract/SPEC.md; follow it."
- `contract/` exists only in T3-contract.

Other checks:
- All 9 data files and the PDF are byte-identical between the two packages (`cmp`).
- `methods/source_method_paper.pdf` is byte-identical to
  `references/2603.02952_Kendiukhov_SAE_atlas.pdf`.
- `contract/SPEC.md` equals `pipelines/sparse-autoencoders/01-sae-atlas.md` (file date
  2026-04-15 20:05) except for local paths. I read the full `diff`: every changed line is a path
  (`../../repos/bio-sae/...` to a repo-relative path, the PDF path to `methods/...`, and
  `../attention-grn-...md` to a bare file name). No other text changed.
- No symlinks. No key or build files inside either package.

## 2. Hints and outcome statements

I searched `BRIEF.md` and `data/README.md` for: fix, bug, v2, retract, correct, wrong, leak,
confound, baseline, null, chance, artefact, artifact, expect, should, result, rarity, rare,
audit, review, error, deploy, experiment, version, original, investigat, ledger, specific,
prior, known, 2610, 3167, report numbers (0.6, 0.37, 0.86), and paths (`biomi`, `/Volumes`,
`../`, `runs/`, `repos/`). I also read both files in full.

- No outcome statements, no result numbers, no audit or review language.
- "v2" appears only in "TRRUST v2", the database name. "specific" appears only in the question.
  "results" appears only in "results object" (the deliverable).
- The only absolute path is the Python interpreter, which the protocol requires.

Kept as is (none states or implies the answer):
1. **The spec carries the deployment's own claims**, for example "6.2% (3/48) TRRUST TF
   specificity" for Geneformer and GATA1 as a "gained" TF in a Geneformer multi-tissue run. It
   is the deployed version, which the protocol says the contract arm gets. It has nothing on
   MaxToki results.
2. **The data give gene detection counts, gene lengths and `n_active_positions`,** and the
   README says the top-20 lists use "no minimum number of positions". These make the rarity
   trap findable. They are plain, needed descriptions of the data, and they are the same in
   both arms. The `chr` column is also there and is not relevant.
3. **Order of knockdown cells.** In `cell_manifest.csv` the 20 TFs with ChIP-seq sets come first.
   This can be worked out from the target files anyway. Reordering would also need the `.npz`
   rows reordered, so I left it.
4. **`._*` files.** There are 13 in T3-paper and 15 in T3-contract. Each holds only the macOS
   `com.apple.provenance` tag ("This resource fork intentionally left blank"), with no path or
   content. I tested removing the tag with `xattr -d`. The command returns 0 but the tag stays,
   so they cannot be removed on this exFAT drive. T4 has the same files. T1 has none.
5. **Interpreter path (harness level, not in the package).** `bin/python` is a wrapper for a
   virtual environment under `biomi_automation/projects/maxtoki/.venv`. Python warnings print
   that path (I saw one from scipy). A subject could learn the repository location this way.
   I cannot fix this inside the package. The protocol's transcript scan for reads outside the
   package covers what a subject does with it.

## 3. Data load and match their descriptions

All checks pass (`check_data.py`):
- `cell_feature_means.npz`: exactly `rows` (9,200, int64), `mean_gene` (9,200 x 4,928, float32),
  `n_tokens` (9,200, int32). Values 0 to 4.17, no NaN. Token counts 720 to 2,048. Catalog cells
  hold 1,019,996 tokens in total, as the README says.
- `cell_manifest.csv`: 9,200 rows, same order as `rows`. Groups: kd 7,500, pool 800, catalog
  500, ref 400. 87 TFs with 34 to 100 cells. `tf` is empty for all control cells. Barcodes are
  unique, and the control groups do not overlap.
- `feature_top20.tsv`: 97,492 rows, 4,928 features, 4,807 with 20 entries and 121 with fewer,
  `<SPECIAL>` in 59 features. Ranks follow decreasing `mean_act_when_active`. All genes are in
  the universe.
- `feature_info.tsv`: `n_genes_listed` and `has_special_token` match the top-20 file. For every
  feature, the sum of the per-gene `n_active_positions` is at most the feature's total.
- `gene_universe.tsv`: 6,324 unique upper-case genes. Detection 1 to 500. 54 genes have no
  length and no chromosome. All 6,324 occur in some top-20 list.
- Target files: TRRUST 2,151 rows, 482 TFs. DoRothEA 106,815 rows, 459 TFs, 296 with ChIP-seq
  pairs. All targets are in the universe, with no duplicate pairs. Of the 87 knocked-down TFs:
  58 have a TRRUST set, 30 a DoRothEA set, 20 a ChIP-seq set, and 22 have none. All match the README.
- `tf_info.tsv`: `n_kd_cells` matches the manifest. 7 TF genes are not measured (BRF2, FOXL2,
  GATA1, HINFP, HOXC10, MZF1, STAT5A).

## 4. Key numbers, re-derived with my own code

My implementation: Mann-Whitney U (scipy, float64, two-sided) for each TF and feature, then BH
across features. Exact gene-swap nulls by convolving hypergeometric distributions. Exact
random-feature null. Fisher tests with scipy. My own seeds for the Monte-Carlo parts.

| key number | key | mine | within tolerance |
|---|---|---|---|
| n_tfs_knocked_down / with ChIP set / with TRRUST set / TRRUST >= 2 | 87 / 20 / 58 / 30 | 87 / 20 / 58 / 30 | yes |
| GATA1 ChIP / TRRUST targets, kd cells | 219 / 11 / 95 | 219 / 11 / 95 | yes |
| median feature mean, ref | 0.008 | 0.00795 | yes |
| GATA1 ChIP targets median detection / length | 34 / 203,066 | 34 / 203,066 (universe 133 / 35,664.5) | yes |
| GATA1 K and features at 0.5 | 5: 628, 1334, 2006, 2627, 3167 | same | yes |
| TFs responding at 0.5 / no cut-off | 6 / 70 | 6 (CDC5L 1, GATA1 5, MED1 2, PHB2 1, RUVBL1 1, TAF1 2) / 70 | yes |
| GATA1 BH-significant features | 4,206 | 4,206 | yes |
| testable ChIP TFs by cut-off | 1, 3, 8, 8, 9, 10, 14 | 1, 3, 8, 8, 9, 10, 14 | yes |
| 2610 effect, 95% CI, share > 0.5 | 0.461, [0.41, 0.51], 0.072 | 0.461, [0.412, 0.512], 0.058 | yes |
| 2610 ChIP overlap; GATA1 best ChIP / TRRUST overlap at 0.5 | 8; 3 / 0 | 8; 3 / 0 | yes |
| GATA1 p: random features / count-matched / count+length | 0.257 / 0.371 / 0.564 | 0.257 / 0.371 / 0.564 | yes |
| GATA1 ChIP rank at 0.5 | 151 | 151 (255 to 275 at 0.25 to 0.05) | yes |
| 2610 expected overlap (count-matched), P(>= 8), uniform E, hypergeometric p | 4.444, 0.037, 0.693, 1.6e-7 | 4.456, 0.037, 0.693, 1.6e-7 | yes |
| 2610 rare genes; ChIP sets with overlap >= 5 | 14; 69 | 14; 69 | yes |
| settings passing BH under matched nulls | 0 | 0 | yes |
| CEBPZ (ChIP, 0.01) p and q | 0.03, 0.61 | 0.030, 0.607 (rank 15/291) | yes |
| planted power k = 8 / 4 / 2 | 0.95 / 0.985 / 0.20 | 0.965 / 0.975 / 0.225 | yes |
| random groups with a responding feature at 0.5 | 0.0 | 0.0 (also 0 at 0.25, 0.1, 0.05; 0.09 with no cut-off) | yes |
| K-matched random groups, GATA1 p | 0.86 | 0.844 | yes |
| other-knockdown p, GATA1 at 0.5 | 0.667 | 0.667 (3 of 5) | yes |
| pooled union at 0.25: naive p; count-matched z | 9.1e-10; 2.37 | 9.1e-10 (219 other sets p < 0.05); 2.36 (rank 70) | yes |
| GATA1 set enriched in other knockdowns at 0.05 | 14 of 26 | 14 of 26 | yes |
| cut-off-free mean rank fraction | 0.652 | 0.652 (CI 0.52 to 0.78; 6 in top half; 0 in top 5%) | yes |
| coarse decile bins: GATA1 uncapped p at 0.25 | 0.015 | 0.0146 (2610 expected 2.462) | yes |
| Fisher test, GATA1 ChIP 0.1, overlap >= 2: p, rank | 5.6e-9, 22 | 5.6e-9, 22 (overlap >= 1: 0.0007) | yes |
| Fisher passes: other sets with p < 0.05 | 61 to 226 | 61 to 226; 16 cases; CEBPZ, GATA1, GTF2B, MAX, TERF2, THAP1; ranks 3 to 147 | yes |
| Fisher mean rank fraction over 106 settings | 0.57 | 0.570 (CI over 14 TFs 0.48 to 0.67); 2 in top 5% | yes |
| responding-feature median detection, GATA1 at 0.1 | 47 (all 119.5) | 47 (all 119.5) | yes |

All 51 numbers agree within their tolerances. The exact parts agree to 3 decimals. The
Monte-Carlo parts differ only by random draws.

Other claims I checked: Welch t-test + BH gives the same 5 GATA1 features and 77 TFs. The pooled
uniform test passes BH for exactly GATA1, GTF2B, MAX, TERF2 and THAP1. TRRUST: the Fisher test
passes BH for 0 TFs. GATA1 TRRUST at 0.02: p = 0.141, rank 2 of 109. GATA1 uncapped
random-feature p at 0.25: 0.002. CEBPZ Fisher ranks are 88.5 to 246 at cut-offs 0.01 and 0.02.

Two small differences:
- **Bins.** I put detection count 500 into the 200 to 500 bin. The reference gives the 5 genes
  with count 500 their own bin (`np.digitize`). This is why my expected overlap for 2610 is
  4.456 and not 4.444, and why 106 other sets have z > 1.96 for the pooled union, not 108. Both
  are well inside the tolerances. The key's bin description ("bins at 1, 2, ..., 500") fits both
  readings.
- **Length tertiles.** The cuts 21,375 and 56,220 bp are not the tertiles of the 6,324-gene
  universe (21,431 and 56,119). A subject using universe tertiles would get slightly different
  count+length p-values. The tolerance of 0.1 covers this.

## 5. The verdict follows from the evidence

`not supported` is right, and it agrees with the source report ("No TF shows target-specific
features under a fair null").
- No TF passes BH under a null that keeps gene detection count, at any of 7 cut-offs, in
  either database.
- The TF's own set is not ranked above other TFs' sets. The mean rank fraction is 0.65 in the
  cut-off-free check and 0.57 in the Fisher check, where 0.5 means chance.
- Power is high for moderate signal. 4 planted GATA1 targets are found about 98% of the time,
  and 8 about 96%.
- Up to 14 ChIP-seq TFs and 48 TRRUST TFs can be tested at lower cut-offs.
- Only weak signal (2 planted targets, power about 0.2) cannot be ruled out. The key already
  asks reports to name this as a limitation.

`inconclusive` is the only other verdict a careful analyst might argue for. The reasons would
be that GATA1 knockdown strength cannot be checked and that weak signal cannot be ruled out.
Neither changes the picture: no setting shows corrected enrichment, and the own-set ranks sit at
or below chance. I agree with the key not accepting it.

## 6. Traps

All six traps are real in these data. I reproduced the numbers each one rests on:
- rarity: uniform p 1.6e-7 compared with count-matched 0.037 for 2610.
- comparison with other TFs: 219 of 290 other sets are also "enriched".
- degenerate random-group null: 0 of 300 groups select a feature at cut-offs of 0.05 or more.
- knife-edge selection: 2610 is above 0.5 in 6 to 7% of bootstraps.
- testability: 1 testable ChIP-seq TF at 0.5.
- one-TF generalisation: the question is about all TFs.

The handling text was fair except for the points I fixed below.

## 7. Changes made

All key changes were made in `build/T3/make_key.py`. I then regenerated `key.json`. A
`diff` of the old and new `key.json` shows only these five text changes. No number, verdict,
trap id or count changed: still 51 numbers, 6 traps and 13 false-statement checks.

1. **`cebpz_cut0.01_chip_p_count_matched` definition.** It said CEBPZ is "the only nominal
   p < 0.05 under the count-matched null". That holds only for the capped statistic. With the
   uncapped overlap there are two more single-setting hits: MAX (ChIP, cut-off 0.1, p = 0.041,
   q = 0.81) and TBP (ChIP, cut-off 0, p = 0.018, q = 0.35). `reference_output.json` counts them
   (`n_p_lt_0.05` = 1 at those settings) but the key did not name them. A grader could otherwise
   mark a true statement about them as false. The definition now names them.
2. **Trap `knife_edge_selection_and_forking_paths`.** I added the same two uncapped hits to its
   list of single-setting p-values. "shows selection stability by bootstrapping cells" now
   reads "ideally shows ...". The core of this trap is not treating a single setting as a
   finding. A bootstrap of selection is good practice, but missing it should not fail the trap
   on its own.
3. **Trap `rarity_unmatched_null`, correct handling.** It required reports to give feature
   2610's expected overlap (4.444). 2610 is not a responding feature at the source cut-off, so
   a correct analysis may never discuss it. It now reads "If the report discusses feature 2610,
   it gives its expected overlap ... as about 4.444, not 0.693."
4. **False-statement check on CEBPZ.** The reason given was "one cut-off only". That is true for
   the count-matched null, but the Fisher test also gives CEBPZ its best rank (3/291) at a
   different cut-off (0). The reason now covers both, with the chance rate (2 top-5% settings
   in 106 against 5.3 expected).
5. **False-statement check on own-set rank.** It said "no TF is in the top 5%". That is true for
   the cut-off-free check but not for single settings: CEBPZ 3/291 and GATA1 13/291 in the
   Fisher test, and GATA1 TRRUST 2/109. The check now says it is about the general statement,
   and that quoting one such setting as a fact is not false.
6. **`NOTES.md`.** Added a grader note on the two uncapped nominal hits (MAX, TBP).
7. **Added `verification_scripts/`** (my code) and this file.

Packages: no changes.

## 8. Not done

- I did not recompute the count+length-matched power, the uncapped power or power at other
  cut-offs. The key quotes these only in text taken from the source report.
- I did not run my selection for random control groups at cut-offs 0.02 and 0.01. I ran 0.5,
  0.25, 0.1, 0.05 and no cut-off.
- I could not remove the `._*` files (section 2, point 4), and I could not hide the venv path that
  the interpreter's warnings print (section 2, point 5).

## Plain-words summary

The two packages are the same apart from the spec and one sentence. They contain no hints of
the answer. The data files match their README. I rebuilt all 51 key numbers with my own code,
and every one is within its tolerance. The verdict `not supported` is right and matches the
source report. I changed five pieces of key text so that graders do not mark true statements
as false and do not ask for things a correct report need not contain. No number or verdict
changed.
