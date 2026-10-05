# T1 verification (independent check of the packages and the key)

Date: 2026-10-01. CPU only. I used my own code, not `reference.py`. My scripts are in
`verification_scripts/` (`common.py`, `main.py`, `null_curveball.py`, `resid.py`,
`boot.py`, `incr.py`, `hgb_spread.py`). They read only the package data.

**Verdict: OK after fixes.** The packages needed no change. The key needed four
wording fixes (traps and number notes). The verdict and every key number stand.

## 1. The two packages differ only where they should

`diff -r T1-paper T1-contract` shows two differences only:

- `BRIEF.md`: one added sentence, "A method specification for this kind of analysis is
  provided in contract/SPEC.md; follow it." It sits in the same place as in T3 and T4.
- `contract/` exists only in T1-contract.

Other checks:
- `contract/SPEC.md` is byte-identical to the deployed spec (file date 2026-04-15).
- Both `methods/source_method_paper.pdf` files are byte-identical to the source PDF.
- All five data files are byte-identical to the source task-data folder.
- `MANIFEST.sha256` verifies in both packages (6 of 6 files).
- No hidden files (`._*`, `.DS_Store`) in either package.

## 2. Hints and outcome statements

I searched `BRIEF.md` and `data/README.md` for: fixed, bug, v2, retract, correct, wrong,
leak, confound, baseline, null, chance, artefact, artifact, expected, should find, audit,
review, error, verify, answer, trap, degree, popularity, residual, co-expression,
evaluation paths (`projects/`, `runs/`, `scripts/`, `biomi`, `studyA`), and AUROC-like
numbers (0.5x, 0.6x, 0.7x). I also read both files in full.

- No outcome statements and no result numbers.
- "v2" appears only in "TRRUST v2", the database name.
- "chosen before any evaluation" (layer 8) is true. The source report says layer 8 was
  fixed in advance by a depth rule.
- The README is a rewrite of the source README with package-relative paths. The
  original file name of the h5ad and the pointer to the source scripts were removed.
- The PDF has no mention of MaxToki. The spec has no mention of MaxToki or RPE1 results
  for this model.

Kept as is, but worth knowing (none is a hint of the answer to this task):

1. **The spec, copied verbatim as required, carries a prior toward "fail".** Its step
   15 calls "fail" "the expected outcome based on scGPT, Geneformer V1/V2, scVI, and
   C2S-Pythia". It also has old repository paths and the word "audit" in its own sense.
   This is part of the contract treatment as deployed. The Study A analysis should say
   that the contract arm received this prior, not only a method.
2. **The data README says TRRUST has duplicate rows and self-listed TFs.** This is a
   plain description of the file and came from the source README. It makes the
   pair-set trap easier. It is the same in both arms.
3. **Paths.** The brief names the interpreter under `maxtoki-framework-eval/`, and the
   package folders are named `T1-paper` / `T1-contract`. Subjects see their own folder
   path anyway, so the interpreter path adds nothing new. The folder name tells a
   subject its arm; that is a harness choice, not something I can fix in the package.

## 3. Data load and match their descriptions

- Attention: float32, 1500 x 1500, range 0 to 0.166, no NaN.
- Pair counts: int32, 1500 x 1500, symmetric, maximum 1,978 (of 2,000 cells).
  Off-diagonal counts never exceed either gene's diagonal count.
- Attention is never non-zero where the pair count is 0 (6 such cells).
- `genes.tsv` and `gene_stats_control_cells.tsv`: 1,500 rows, same order and symbols,
  no duplicate symbols, Ensembl IDs or token IDs.
- Gene stats: variance 0.22 to 1.22, mean 0.27 to 3.18, dropout 0.0015 to 0.771.
  A gene is a token in fewer cells than it is detected in (about 66% on average),
  because each cell is cut at 2,046 tokens. This fits "present as tokens".
- TRRUST: 192 rows, 175 unique pairs, 3 self-rows (2 unique self-pairs). Every
  `tf_index` and `target_index` matches the symbol in `genes.tsv` (upper case).
- Diagonal (self-attention) mean is 162 times the off-diagonal mean.
- 58% of the variance of the off-diagonal attention matrix is the key gene's column
  mean. So attention is mostly a property of the key gene.

## 4. Key numbers re-derived with my own code

Different seeds and a different null sampler (one long Curveball chain, 20,000 burn-in
trades, then 1,000 draws 1,000 trades apart). All within the key's tolerance.

| Key number | Key | Mine |
|---|---|---|
| rows / unique non-self edges | 192 / 173 | 192 / 173 |
| TFs / positives / negatives | 16 / 138 / 23,846 | same |
| positives on distinct targets; targets per TF | 87; 4-21 | 87; 4-21 |
| attention, TF as query (pooled) | 0.6045 | 0.6045 |
| TF-bootstrap CI | 0.549-0.655 | 0.547-0.654 |
| per-TF mean | 0.5867 | 0.5867 |
| TF as key / symmetrised mean / max | 0.546 / 0.638 / 0.647 | same |
| variance | 0.6864 | 0.6864 |
| gene model (logistic, GroupKFold by TF) | 0.7494 | 0.7494 (random TF splits 0.737-0.756) |
| variance - attention [CI] | 0.082 [0.028, 0.145] | 0.082 [0.027, 0.150] |
| gene model - attention [CI] | 0.145 [0.055, 0.238] | 0.145 [0.056, 0.239] |
| degree null mean (SD) | 0.574 (0.017) | 0.574 (0.017) |
| degree null z / p (pooled) | 1.82 / 0.046 | 1.80 / 0.036 |
| degree null z, per-TF statistic | 0.75 | 0.79 (p 0.21) |
| label permutation z | 4.12 | 4.23 |
| row-only null (TF degree only) | mean 0.501, z 4.20 | mean 0.502, z 4.21 |
| expression proximity, null z | 1.85 | 1.86 |
| co-detection lift, null z | 3.09 | 2.99 (per-TF 2.97) |
| symmetrised attention, null z | 3.26 (per-TF 2.3) | 3.15 (per-TF 2.1) |
| residualised OLS [CI] | 0.492 [0.404, 0.568] | 0.492 [0.402, 0.568] |
| residualised boosted trees [CI] | 0.559 [0.488, 0.617] | 0.564 [0.496, 0.624] |
| residualised symmetrised (OLS, symmetric features) | 0.551 | 0.551 |
| incremental, logistic, folds by TF [CI] | +0.013 [0.0005, 0.025] | +0.013 [0.001, 0.028] |
| incremental, logistic, folds by gene | +0.009 [-0.002, 0.019] | +0.009 [-0.001, 0.021] |
| two-way TF x gene bootstrap, attention | 0.509-0.699 | 0.502-0.696 |
| self-pairs scored / duplicate rows counted | 0.610 / 0.624 | 0.610 / 0.624 |
| TF threshold >= 1 / >= 5 | 42 TFs 0.595 / 11 TFs 0.609 | same |

Null checks: every draw kept all row and column sums and never put an edge on a TF's
own gene. No draw equalled the real network (Jaccard distance 0.81 minimum, 0.88 mean).

Two results differ from the key in a way that matters for grading (fixed below):

- **Pair bootstrap vs TF bootstrap.** For attention they give almost the same CI here
  (pair 0.550-0.655; TF 0.546-0.652). Resampling candidate genes is what widens it
  (genes only 0.546-0.669; two-way 0.502-0.696).
- **Boosted-tree incremental value moves with settings and seed.** With folds by TF it
  is always negative (-0.015 to -0.037 over 2 settings x 3 seeds). With folds by gene it
  ranges from -0.013 to +0.046; my default-settings run gave +0.037 [+0.008, +0.064].
  There the boosted gene-only model is weak (0.64-0.68), and gene + attention
  (0.66-0.71) stays below the logistic gene-only model (0.75).

## 5. The verdict follows from the evidence

"not supported" is right, and it agrees with the corrected source report (layer-8
TRRUST result negative; V4: co-expression beats the degree null as well as attention).

- Attention (0.605) is above 0.5 but below variance alone (0.686) and the gene model
  (0.749).
- A null that keeps each TF's and each target's edge count gives 0.574. That is about
  70% of attention's lead over 0.5.
- What is left is borderline (z 1.8, p 0.04-0.05 one-sided; per-TF z 0.8) and goes to
  0.492 after OLS residualisation.
- Scores with no model (co-detection lift, expression proximity) beat the same null
  about as strongly as any attention variant.
- Under the spec's own rule, criterion (a) fails (attention does not beat every
  gene-level baseline), so the spec's verdict is "fail" whatever (d) would show.

"inconclusive" under the key's three conditions is fair: the pre-specified score has a
nominal p below 0.05 against the right null in this single-run task, and symmetrised
attention reaches z about 3. "supported" is not defensible.

## 6. Traps: real, and the descriptions are now fair

All six traps are real in these data (numbers in section 4). Changes to `key.json`:

1. **`resampling_unit` (rewritten).** Before: it said the effective sample size is about
   16 TFs and listed a pair bootstrap as a mishandling. That is not accurate here: the
   pair bootstrap gives the same attention CI as the TF bootstrap; candidate genes are
   the dependence that matters. Now: the description gives the pair, TF and two-way
   intervals; correct handling accepts TF, gene or two-way resampling, or pair
   resampling with the dependence stated; mishandling is ignoring the dependence and
   leaning on pair-level precision for a strong claim, or giving no uncertainty.
2. **`degree_preserving_null` mishandling.** Added "a null that keeps only each TF's
   number of targets and draws targets uniformly" (mean 0.50, z 4.2). The spec
   describes Curveball as preserving "each TF's out-degree", so contract-arm subjects
   may build exactly this null. Before, graders had no line for it.
3. **`forking_paths`, `incremental_auroc_logreg`, false-statement check on incremental
   value.** Added the boosted-tree spread (folds by TF -0.015 to -0.037; folds by gene
   -0.013 to +0.046) and the rule to judge incremental claims against the strongest
   gene-only model. Without this, a subject who reported a positive boosted-tree gain
   with folds by gene could be marked as stating a falsehood when the number is real.
4. **`gene_model_auroc_pooled` note.** "Other gene-only models give 0.68-0.80" changed
   to "about 0.64-0.82" (boosted trees with folds by gene give 0.64-0.68).
5. Added `"verification"` field pointing here. `NOTES.md`: two short notes added
   (incremental value; resampling unit) and this file added to the file list.

Hashes after the changes: `key.json` 9edc7ed1a0a69c5336438f5aee4f519dbe99771340f1452692a90139fa3402b9,
`NOTES.md` 77adaa6cf53786aa7d76cd868f3acde244d721409de69f8b068ee1f03a2aa2ca.
Before: `key.json` 6d059e2f9209482dec826cd32ecef1f26ad0a7da7df746c5d97c49d645e02c4a.

## Not done

- I did not write the key hash into `protocol/FREEZE_LOG.md`. That should happen when
  the key is frozen, after this check.
- I did not compute Spearman co-expression. The package has no expression matrix. The
  key quotes the source value (z 3.1).
- I did not run `reference.py` or a second independent null sampler (checkerboard).
- I did not re-check the CIs for fitted models with refitting. Like the key, they hold
  the fitted models fixed.

## Plain-words summary

The two packages are clean and match each other except for the spec. The data load and
match their README. My own code reproduces every key number within tolerance. The
verdict "not supported" follows from the data and agrees with the corrected report. I
fixed the key in four places so graders do not mark correct work as wrong: the pair
bootstrap is not worse than the TF bootstrap here, a TF-degree-only null needed its own
line, and the boosted-tree incremental value is unstable and needed its full range.
