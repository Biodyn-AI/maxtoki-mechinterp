# P11 build record (DoRothEA sign vs CRISPRi knockdown response; planted P9)

| id | status | twin |
|---|---|---|
| B4668 | flawed | B8104 |
| B8104 | clean | B4668 |

The ids were drawn at random in the first build session. The id-to-status mapping and the coin flip are in
`studyB/build/P11/ids.txt` (`flip=0`). Neither id matches any other package or key. Built 2026-10-01 with
`maxtoki-framework-eval/bin/python`, CPU only. Build scripts are in `studyB/build/P11/`, outside every package.
The two packages differ only in `SUMMARY.md`. Every other file has the same md5 in both.

## Sources (read only; nothing in biomi_automation was changed)

- Code: `biomi_automation/projects/biotensor/codebase/route_a/scgpt/runpod_scale/sign_test.py` (171 lines). Its
  helpers `perturb_residual.load_replogle` and `sign_test.load_signed` were imported for the source re-run, with
  bytecode writing off.
- Data: `runs/grn_benchmark/replogle/K562_gwps_normalized_bulk_01.h5ad` (11,258 x 8,248) and
  `rpe1_normalized_bulk_01.h5ad` (2,679 x 8,749); `data/netdb/omnipath_dorothea_levels.tsv`.
- Results: `runs/grn_benchmark/sign_k562.json` and `sign_rpe1.json`.
- The source has no write-up to copy. Both SUMMARY.md texts were written in the analyst's voice. The flawed one
  carries the planted overreach.

## What was built

- `make_data.py` makes the package data:
  - `dorothea_omnipath.tsv.gz` is the OmniPath file, gzipped. It decompresses to the same md5 as the source.
  - `<line>_tf_knockdown_response.csv.gz` keeps every guide-level profile whose gene is a signed DoRothEA
    source (672 K562 rows, 74 RPE1 rows). It keeps the signed targets of those TFs plus the TFs themselves,
    among genes that are finite in every row of the full release (2,746 and 1,301 columns). Values are float32
    and round-trip exactly.
  - `<line>_guide_info.csv` holds the cell count for each profile.
  - Each package's data is 13 MB.
- `code/sign_test.py` is a standalone rewrite of the source script with package-relative paths.
  - Kept: the signed-edge loader, guide averaging, the own-gene NaN, the three |dE| strata, the
    balanced-accuracy table with the always-down and random baselines, and the level breakdown.
  - Added, as the build notes asked: the balancing-free 2x2 table with its OR and Fisher p, two sign-shuffle
    nulls (across edges and within TF), a TF-resampling interval, the balanced accuracy over the observed
    classes, and a 0.5 Haldane correction for tables with a zero cell. The first trial run (`run1/`) gave
    `OR nan` for two level rows, so the Haldane correction was added (`run2/`).
  - Dropped: the co-expression arm, which needs the prep matrix; the repo imports; and the docstring history
    (rank-1 nuisance, FFN probe).
- README.md is the same in both packages and has no outcome statements.
- SUMMARY.md has the same title, the same "What was scored" and "Results" sections, and the same numbers in
  both packages. Only the Interpretation and Conclusion sections differ. The flawed one is 749 words. The clean
  one is 780.
  - Flawed: the sign "predicts" the response. CRISPRi "confirms that DoRothEA edges are direct causal
    regulatory interactions". The two lines are presented as independent support. The network is "a validated
    wiring diagram" and "ground truth".
  - Clean: the agreement is modest and top-decile only. It is absent across all K562 edges and fragile once
    edges are grouped by TF. A knockdown measures total effects. Both lines come from one platform.
  - Neither summary claims the inhibition-only pattern, because it failed on four other datasets in the source
    project.

## Verification

| check | result |
|---|---|
| edge table vs the source logic re-run on the full h5ad files (`reproduce_orig.py`, `verify_vs_source.py`) | same 8,358 / 1,866 edges, same order, signs and levels; max dE difference 5e-9 |
| point estimates vs the source JSON | DATABASE sign and always-down: balanced accuracy, per-class accuracy and n per stratum equal to 1e-12 |
| headline numbers | top-10% OR 2.31 (K562) and 4.06 (RPE1); all edges 0.99 and 1.40. These match the build notes |
| bootstrap intervals and the random baseline | small differences from the source (for example K562 top 10% [0.5529, 0.6450] vs [0.5545, 0.6450]). The random stream differs because the co-expression arm was dropped and the nulls were added |
| `[prior]` fraction negative | 0.4913 / 0.5456 on the packaged matrix vs 0.4865 / 0.5387 on the full release. It is descriptive only |
| OR function | equals the inverse of scipy's Fisher odds ratio (2.3142 for K562 top 10%) |
| within-TF shuffle | each group keeps its own label mix (500 random trials) |
| extra checks (`extra_checks.py`, log in `extra_checks.log`) | own-gene knockdown is negative for 97.4% / 98.0% of measurable TFs. Keeping only TFs with own dE < -0.2 gives OR 4.05 / 7.26. Without MYC the OR is 2.22 / 1.64. Leave-one-TF-out ranges are 2.18-2.50 and 1.64-5.41 |
| outputs re-run (after the restart) from each package root into a scratch folder | all 5 files byte-identical, in both packages. Took 35-75 s with a 0.25 GB peak |
| hint scan | no fix/bug/v2/retract/correct/wrong/leak/audit/review/error/flaw/clean words and no repo paths. The only hit is `index=False` |

Flawed is wrong and clean is right, with the same numbers in both (planted P9). The flawed claims are wrong for
three reasons:
- The package holds only knockdown responses, which are total effects. It has no binding data, no time course
  and no direct-vs-indirect control.
- The outputs show agreement only in the top 10%, with none across all K562 edges.
- K562 does not pass the within-TF shuffle (p = 0.065).

Every clean statement is backed by `run_log.txt`, `sign_*.json` or the extra checks above.

## Limits

- The original source run was not repeated with its co-expression arm. Only the sign arm was compared.
- The first session ended before the keys were written. This session found the packages complete, re-checked
  them as above, and then wrote the keys and this record. It changed nothing inside the packages.
- macOS writes AppleDouble `._*` files on this exFAT drive. They hold metadata only, and none are present
  inside the P11 packages.
