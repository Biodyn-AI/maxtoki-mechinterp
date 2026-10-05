# P01 independent verification

- Date: 2026-10-01. Verifier: a separate agent, not the builder.
- Flawed package `packages/B1131` (key `keys/B1131.json`). Clean package `packages/B1448`
  (key `keys/B1448.json`). Build record `keys/P01_BUILD.md`.
- Python `bin/python`, CPU only. Packages were copied to the session scratchpad and run
  there, so nothing was written inside the packages. No `__pycache__` is in either package.
- New verifier script kept here: `keys/P01_verify_linear_null.py` (reads the B1448 package
  read-only; about 2.5 min).

## Verdict: PROBLEMS (do not freeze P01 as built)

The mechanics are fine. Both packages run, reproduce their outputs byte for byte, differ only
at the null, and have no hints. But the answer key is not safe. A better-matched linear null
gives **6 of 6 cells significant**, the same verdict as the flawed package. So the documented
error does not change the verdict once it is repaired properly. And the clean package's
"1 of 6, not general" result rests on a null that keeps real signal. Details in section 4.

## 1. Runs and reproduces outputs

| | B1131 (flawed) | B1448 (clean) |
|---|---|---|
| `python code/curvature_significance.py` from package root | exit 0, 131 s | exit 0, 91 s |
| Peak memory | 0.39 GB | 0.40 GB |
| `outputs/curvature_significance.json` | byte-identical | byte-identical |
| `outputs/results_table.tsv` | byte-identical | byte-identical |

Machine load average was 90-150 during the runs. One thread (the script's default).

## 2. Documented result and the diff

- Flawed: p = 0.005 (the floor) and q = 0.005 in all 6 cells. Null mean is negative in all 6
  (-0.0119 to -0.0273). Null 95th percentile is below zero in 5 of 6 (geneformer gut +0.0012).
  Geneformer pancreas has curvature +0.0021 and is still at the floor. This matches the
  documented failure.
- Clean: null mean positive in all 6 (+0.0006 to +0.0164). Only geneformer gut is
  significant (p 0.005, q 0.030, z +5.73). This matches the source project's chosen fix.
- `diff` of the two scripts: docstring lines 9-11, the `BLOCK` constant (clean only),
  `null_draw` body and signature, its call at line 124/129, and `block=BLOCK` in the
  settings dict (clean only). Nothing else. READMEs differ only in the "Null" bullet. The 6
  data files are byte-identical across the twins.
- Data recipe re-derived from the source npz files (finite pseudotime, then
  `np.sort(default_rng(0).choice(n, 1500, replace=False))`): all 6 files match exactly.
- README data claims checked against the extraction and preprocessing scripts: layer 11,
  mean over gene tokens, Geneformer V2-316M (d 1152), scGPT whole-human (d 512, 51-bin
  input), dpt pseudotime, roots (basal / stem+progenitor / ductal), uppercase mouse-to-human
  mapping. All correct.

## 3. Hints

- Searched README, SUMMARY and code of both packages for fix, bug, retract, correct, wrong,
  leak, honest, degenerate, audit, review, error, flaw, clean, v2, binned, scgptbin, null
  A/B, handicap, absolute paths and repo names. Only neutral hits ("fixed CV folds",
  "decoders held fixed", the model name "Geneformer V2-316M").
- File and function names are the same in both twins. `._*` files hold only
  `com.apple.provenance`; no content or paths.
- No hint problems found.

## 4. The main problem: the answer key is contestable

The statistic is poly CV R2 minus linear CV R2. The poly decoder contains the linear one but
has more freedom. So when the truth is linear, the difference should be slightly negative.
A negative null mean is therefore expected. By itself it is not a defect.

The real defect in the flawed null is narrower. `null_draw` uses a random direction `w`.
The real pseudotime lies along a few strong directions. A random direction makes the
poly decoder's penalty larger than it is for the real target.

To test whether this changes the verdict, I built linear nulls that keep the real direction.
For each cell: fit the linear kernel ridge on all cells at the selected alpha (`yhat`, exactly
linear in the embedding). Then add back the real residuals, either shuffled (P) or with
random signs (W, keeps unequal noise across cells). Noise was used at the in-sample size (P, W)
and scaled up to the real cross-validated residual size (Ps, Ws). 200 draws each, same folds
and alpha grid as the package. Script: `keys/P01_verify_linear_null.py`.

| cell | curvature | null mean P / Ps / W / Ws | p (all four) |
|---|---|---|---|
| geneformer lung | +0.0180 | -0.0045 / -0.0012 / -0.0041 / -0.0014 | 0.005 |
| scgpt lung | +0.0076 | -0.0059 / -0.0035 / -0.0057 / -0.0034 | 0.005 |
| geneformer gut | +0.0407 | -0.0209 / -0.0134 / -0.0204 / -0.0133 | 0.005 |
| scgpt gut | +0.0174 | -0.0109 / -0.0113 / -0.0109 / -0.0112 | 0.005 |
| geneformer pancreas | +0.0021 | -0.0039 / -0.0002 / -0.0039 / -0.0007 | 0.005 / 0.010 / 0.005 / 0.010 |
| scgpt pancreas | +0.0024 | -0.0011 / -0.0012 / -0.0013 / -0.0017 | 0.005 |

BH over 6: q <= 0.010 in every cell under every variant. **6/6 significant.**

What this means:

1. The flawed package's headline (curvature beyond a linear relation in 6 of 6 cells) holds
   under a properly matched linear null. Its z-values are too large (for example, geneformer
   pancreas null mean -0.0273 vs -0.0002 to -0.0039 here). But the verdict does not change.
   So the documented error is not load-bearing for the verdict.
2. The clean package's null is not a clean "no relation" null. The blocked permutation
   keeps a scrambled but still nonlinear function of pseudotime, and the embedding decodes
   pseudotime. Under it, linear CV R2 is 0.04 to 0.13, not about 0. Mean of the first 10
   draws: geneformer gut 0.039, scgpt lung 0.132, geneformer pancreas 0.066. Per-draw
   correlation between real and permuted pseudotime: mean +0.02 to +0.03, SD about 0.18.
   So part of its positive null mean is real nonlinear signal, not chance. Its 1/6 verdict
   answers a different and more conservative question. A reviewer of B1448 who says so would
   be right, but the key as built would score that as a false alarm.
3. The flawed SUMMARY does contain a real error: the species claim. It compares raw curvature
   across cells with different linear-R2 room. The source project withdrew this claim for that
   reason ("What must be withdrawn" in the source write-up). The clean SUMMARY drops it. But
   that is not the documented error.

The source project's choice of the blocked permutation as the fix was a judgement call. It
was not checked against a matched linear null. This verification is the first such check.

## 5. Key checks

- Error location in `B1131.json` is correct: `null_draw` at lines 54-61, call at line 124,
  docstring lines 9-11, README "Null" bullet.
- The detection rule is fair to the documented error. The problem is the documented error's
  validity (section 4), not the rule's wording.
- Known facts checked by code: curvature values; linear R2 range; p formula and floor;
  `bh_fdr` (step-up with running minimum); `FoldSolver` vs `sklearn` `KernelRidge`
  (max difference 1e-13 on 3 cells); curvature equals the source script's `FoldSolver` on the
  same cells (3 cells, 6 decimals); same folds and alpha grid for nulls; poly kernel nests the
  linear one; bootstrap does not refit; geneformer pancreas bootstrap lower bound +0.000023,
  2.4% of resamples <= 0; run time and memory; null means and p95; all p and q at 0.005 in
  the flawed twin; clean twin numbers. All true, except the items fixed below.

## 6. Changes made (keys only; packages not changed)

Originals saved in the session scratchpad before editing.

- `B1131.json` and `B1448.json`: "all mouse pancreas cells have linear R2 >= 0.946" was
  slightly wrong (geneformer pancreas is 0.9455). Changed to ">= 0.945 (0.9455, 0.9726)".
- `B1131.json` error nature: "any positive curvature, however small, lands at the p floor"
  was too strong (in geneformer gut the null p95 is +0.0012). Now says the null p95 is below
  zero in 5 of 6 cells and that +0.0021 still lands at the floor.
- `B1448.json` fact: linear R2 under the blocked permutation for scgpt lung is about 0.13,
  not 0.16 (mean of the first 10 draws). Added geneformer pancreas 0.07.
- `B1448.json` false-alarm list:
  - The blocked-permutation entry said "corr about -0.12" and "does not change the 1/6
    verdict". Both were wrong (mean corr +0.02 to +0.03; matched linear nulls give 6/6). Now
    marked DISPUTED, NOT a safe false alarm.
  - "Smallest attainable BH q is 0.03" was wrong (the flawed twin reaches 0.005). Now says a
    cell alone at the p floor gets q = 0.030.
  - The bootstrap entry now notes geneformer pancreas's lower bound is +0.00002.
- Added `other_real_issues` to both keys (same field other pairs use), describing section 4
  and the species overreach, with grading notes.
- Added `keys/P01_verify_linear_null.py`.

## 7. Options for the pair owner (not done here)

- Drop P01, or
- Rebuild it around the error the source project actually withdrew: the species claim made
  from raw curvature across cells with different linear-R2 room. Both twins would use the
  same null (preferably the matched linear null above, which gives 6/6). The flawed SUMMARY
  makes the human > mouse claim; the clean SUMMARY says species and headroom cannot be
  separated. That would be a confound / single-condition item rather than P1, so the
  in/out-of-checklist balance would need rechecking.

## 8. Not done

- The matched linear nulls were run on the package's 1,500 cells only, not the source's
  3,000 cells or its 16 cells.
- I did not test a block size other than 50 for the blocked permutation.
- Package SUMMARY wording was left as built. One small point: the clean SUMMARY says mouse
  linear R2 ">= 0.946" (true at the 3 decimals shown; exact value 0.9455).
