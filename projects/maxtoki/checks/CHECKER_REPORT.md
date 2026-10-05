# Deterministic checks on the MaxToki deployment — checker report

Date: 2026-10-01. Scope: the 8 deployed pipeline specs and the 8 run folders under
`projects/maxtoki/runs/`, as they stood at the end of the deployment (files dated on or before
2026-05-07; later `v2_*` correction files are excluded). CPU only; no model was run.
Every number below is in `results/checker_summary.json` (made by `summarize_results.py`) or in the
per-check result files named next to it. How to run and what each check means: `README.md`.
The requirement-by-requirement table is `REQUIREMENTS_TABLE.md` / `requirements_table.csv`.

---

## 1. Results first

1. **A literal template check rejects 7 of the 8 deployed specs.** Only the longevity spec has all ten
   sections in order. The other seven have no "Code references" section, so their later headings are
   numbered one lower than the template. All eight were executed. (`spec_lint.py` S1; same result from
   an independent string check and from the investigation's prototype linter.)
2. **Code pointers almost never use the required form.** 5 of 230 pointers in the 8 specs are written as
   `repos/<name>/path:LINE`. 165 resolve as written, 53 only by path suffix or file name, 2 not at all,
   10 are wildcards. 174 point into clones that have no `.git`, so their content cannot be tied to the
   pinned commit. 4 named functions are not defined in the file the attention spec cites for them
   (`run_incremental_value`, `run_residualization`, `permutation_test_contrast`, `bootstrap_stability`).
3. **Pins:** 6 of 8 specs have a hash that matches `PINNED_COMMIT.txt` but the clone has no `.git`;
   1 (longevity) is verified against git HEAD; 1 (manifold) states that no repository exists.
4. **No spec has a machine-readable validation block, and none has labelled slots for all four of
   null model, trivial baseline, positive control and scope.** Labelled slots exist in 6 / 1 / 1 / 2 of
   the 8 specs. 80 of 122 validation items carry a numeric threshold, but 7 of 8 Validation sections
   are written as replication targets for the source model, not as pass/fail gates for MaxToki.
5. **The runs did not use the spec's parameter values in 10 parameters (11 table rows) across 5 runs.**
   Only 66 of 258 parameter rows could be linked to what a run recorded. Among those, examples:
   Curveball null 50 draws vs 200 in the spec; label-permutation null 8 and 50 vs 100–1,000; rewiring
   null 8 and 12 vs ≥ 24; 50 cells per ablation condition vs 100–500; SAE training on 500 cells vs
   2,000; loop capped at 8 iterations vs 40–80. A reason is written down for 1 of the 11 (Curveball).
   Two of these need care: `n_ctrl` differs only in the 1B side run (`phase0_k562_1b`; the 217M runs used
   the default 2,000), and the second `n_cells_training` row (Tabula Sapiens) is linked to the K562
   control count, although the run's Tabula Sapiens step also used 500 cells (`phase9.log`).
6. **Two recorded gate verdicts rest on a gate that was never computed.** The frozen manifold gate file
   requires all five gates on the internal panel and four on the external panel. On the internal panel,
   2 result objects (LET anchor, H38-lite) record "pass" while the permutation gate (p ≤ 0.001 with
   ≥ 2,000 permutations) has no value anywhere in the run. The external-panel verdicts need only the four
   computed gates. The first lung negative control (`lung_control`, marked `expected_to_fail`) passed;
   the run itself noticed that this panel held lung immune cells and rebuilt it as `lung_nonhema`,
   which failed as expected (manifold `FINAL_SUMMARY.md` lines 39–40). The zero-shot panel is not named
   by the frozen rules, so which gates apply there is a human decision. H103's zero-shot
   trustworthiness used k = 5 while the frozen protocol says k = 15 (the file says why: 18 anchors).
   In the attention K562 verdict, the value stored as a "fraction of signal retained" is −13,421.
7. **Number tracing finds almost everything, but so does chance.** Counting each number once per run,
   1,132 of 1,186 distinct FINAL_SUMMARY numbers are found in their run's outputs within rounding
   (95.4%, Wilson 95% CI 94.1–96.5%). Random numbers of the same format are found 78% of the time.
   So tracing adds 18 percentage points over chance. Numbers from one run share one corpus, so the
   interval resamples the 8 runs: chance 65–89%, excess 8–27 points (2,000 resamples). 980 of the
   1,132 traces are WEAK (that number's own chance rate is ≥ 0.5). (Pooled over all 13 files without
   removing the 461 repeats between two copies of a summary: 1,587 of 1,647, chance 80%, excess 16.)
   For the paper checked against all 8 runs, chance is 97%, so "found in the outputs" says nothing.
   The useful paper check is the chain paper → run summary → same run's outputs: 119 of 124 numbers,
   chance 50%, excess 0.46 (95% CI 0.39–0.53, bootstrap over numbers).
8. **Presence is not correctness.** All 27 paper number mentions (23 distinct values) that the manual
   number audits judged wrong in meaning (10 MISMATCH, 17 DIFFERENT ENDPOINT) are "found" in the run
   outputs. The one number the manual audits marked NOT FOUND ("15 hours", 4 mentions) is also
   "found" — by chance.

Plain meaning: code can catch missing sections, bad pointers, unpinned code, parameter changes, and
gate flags that do not follow from the numbers. It cannot tell whether a null, a baseline or a
number's meaning is right. Those need an agent or a person reading the work.

---

## 2. What was built

`projects/maxtoki/checks/`:

| file | role |
|---|---|
| `spec_lint.py` | spec checks S1–S7 (sections, non-empty, pin, code pointers, parameter table, validation block, labelled slots) |
| `number_trace.py` | trace every number in FINAL_SUMMARY files and the paper to run outputs, with a chance-match rate |
| `run_manifest_check.py` | rebuild what each run used; compare with the spec (M1); recompute gates (M2); phases (M3); reproducibility records (M4); interval/scope/wording pointers (M5) |
| `param_map.json` | hand-curated links from 58 spec parameter rows to where a run records the value used (judgment, with a reason per link) |
| `verify/*.py` | five second-way checks (section 6) |
| `summarize_results.py` | builds `results/checker_summary.json/.md` and `results/repair_queue.csv` |
| `requirements_table.py` | builds `REQUIREMENTS_TABLE.md` and `requirements_table.csv` |
| `run_all.sh`, `README.md` | how to run; what each check does and does not establish |

Every main script writes a `*_run_config.json` with the sha256 of each input file (files over 200 MB:
size and date only), the tool's own hash, Python version and seed 20261001.
With `--strict`, each main script exits 1 on its blocking rules, so it can be used as a pre-run or
pre-summary gate. Nothing calls it yet.

---

## 3. Spec lint (8 deployed specs)

| spec | template | S1 | S2 | S3 pin | S4 refs | S5 params | S6 val block | S7 slots | refs: required form / as written / total |
|---|---|---|---|---|---|---|---|---|---|
| attention-grn-extraction-and-evaluation.md | REJECT | FAIL | PASS | TEXT_MATCH_ONLY_NO_GIT | FAIL | PASS | FAIL | FAIL | 0 / 22 / 22 |
| residual-stream-spectral-geometry.md | REJECT | FAIL | PASS | TEXT_MATCH_ONLY_NO_GIT | FAIL | PASS | FAIL | FAIL | 0 / 13 / 28 |
| topology-geometry-141-hypotheses.md | REJECT | FAIL | PASS | TEXT_MATCH_ONLY_NO_GIT | FAIL | PASS | FAIL | FAIL | 0 / 18 / 22 |
| manifold-discovery-extraction-compactification.md | REJECT | FAIL | PASS | NONE_WITH_REASON | FAIL | PASS | FAIL | FAIL | 0 / 7 / 7 |
| longevity-mechinterp-donor-aware.md | ACCEPT | PASS | PASS | VERIFIED_GIT_HEAD | FAIL | PASS | FAIL | FAIL | 0 / 15 / 46 |
| 01-sae-atlas.md | REJECT | FAIL | PASS | TEXT_MATCH_ONLY_NO_GIT | FAIL | FAIL | FAIL | FAIL | 0 / 45 / 52 |
| 02-causal-circuit-tracing.md | REJECT | FAIL | PASS | TEXT_MATCH_ONLY_NO_GIT | FAIL | PASS | FAIL | FAIL | 5 / 27 / 27 |
| 03-exhaustive-mapping-and-steering.md | REJECT | FAIL | PASS | TEXT_MATCH_ONLY_NO_GIT | FAIL | PASS | FAIL | FAIL | 0 / 18 / 26 |

Notes (details in `results/spec_lint_coderefs.csv`, `spec_lint_params.csv`, `spec_lint_validation_items.csv`):
- The 2 pointers that do not resolve: `run_claude_topology_autoloop.py` (topology spec; the file exists
  only in a different pinned clone, `topology-biomechinterp1`) and
  `../../repos/bio-sae-circuits/results/circuit_analysis.json` (03 spec).
- Many spectral and topology pointers are partial paths such as `iter_0010/run_iter0010_screen.py`;
  they resolve only as a suffix of `iterations/iter_0010/...`.
- The longevity spec's 13 line pointers are all in range and all 31 named functions sit at the cited
  line. The 02 spec's 5 `repos/...:LINE` pointers and 14 function-at-line checks all pass.
- S5 fails for `01-sae-atlas.md` because `n_cells_training` appears twice. In the 02 spec, cells such
  as `**|d| > 0.5**` contain unescaped pipe characters, which break the table's columns.
- 258 parameter rows; 177 defaults and 80 ranges parse as numbers.
- The audit spec itself (not one of the 8) passes S1 but cites 3 paths that do not exist, including the
  `iter_0009/run_iter0009_screen.py` example.

---

## 4. Number tracing

Per FINAL_SUMMARY file (unit = distinct number in the file; traced share with Wilson 95% CI; chance
rate and excess with percentile bootstrap 95% CI over numbers, 2,000 resamples):

| run / file | numbers | traced (share, CI) | chance (CI) | excess (CI) | weak | untraced |
|---|---|---|---|---|---|---|
| attention-grn — summaries/ | 102 | 100 (0.98, 0.93–0.99) | 0.95 (0.92–0.98) | 0.03 (0.01–0.05) | 98 | 2 |
| attention-grn — outputs/ | 93 | 92 (0.99, 0.94–1.00) | 0.95 (0.92–0.97) | 0.04 (0.02–0.06) | 90 | 1 |
| spectral — summaries/ | 200 | 199 (0.99, 0.97–1.00) | 0.91 (0.87–0.94) | 0.09 (0.06–0.12) | 185 | 1 |
| spectral — outputs/ | 116 | 116 (1.00, 0.97–1.00) | 0.96 (0.93–0.98) | 0.04 (0.02–0.06) | 114 | 0 |
| topology — summaries/ | 55 | 55 (1.00, 0.93–1.00) | 0.93 (0.87–0.98) | 0.07 (0.02–0.13) | 52 | 0 |
| topology — run folder | 95 | 93 (0.98, 0.93–0.99) | 0.88 (0.84–0.93) | 0.10 (0.06–0.13) | 89 | 2 |
| manifold — summaries/ | 215 | 211 (0.98, 0.95–0.99) | 0.74 (0.71–0.76) | 0.25 (0.22–0.27) | 194 | 4 |
| manifold — run folder | 263 | 259 (0.98, 0.96–0.99) | 0.73 (0.70–0.75) | 0.26 (0.24–0.28) | 239 | 4 |
| longevity — run folder | 35 | 22 (0.63, 0.46–0.77) | 0.23 (0.17–0.29) | 0.40 (0.21–0.58) | 3 | 13 |
| sae-atlas — summaries/ | 141 | 139 (0.99, 0.95–1.00) | 0.96 (0.93–0.98) | 0.03 (0.00–0.05) | 134 | 2 |
| sae-atlas — 12-layer | 71 | 70 (0.99, 0.92–1.00) | 0.97 (0.92–1.00) | 0.02 (−0.01–0.06) | 68 | 1 |
| circuit-tracing | 112 | 96 (0.86, 0.78–0.91) | 0.58 (0.50–0.66) | 0.28 (0.20–0.36) | 68 | 16 |
| exhaustive-mapping | 149 | 135 (0.91, 0.85–0.94) | 0.53 (0.47–0.59) | 0.38 (0.31–0.44) | 80 | 14 |

What this means:
- Where a run wrote millions of values (attention: 4.05 M distinct values; sae-atlas: 0.41 M), almost any
  2- or 3-decimal number is present somewhere. There, "traced" is close to worthless (excess 0.02–0.04).
- Pooled by format: 2+-decimal numbers are found 98% of the time vs 76% by chance; 1-decimal numbers
  95% vs 89%; integers ≥ 10 95% vs 85%; 10^k numbers 84% vs 45%.
- The 33 untraced numbers that are not also in the spec (32 distinct; "217" is in two copies of one
  summary; list: `results/repair_queue.csv`, check `number_trace`) are mostly: model names (217),
  numbers from another run (2,980 and the triplet ratios 0.165 / 0.190 / 0.219 / 0.324 quoted in the
  circuit summary; 2,144,011 quoted in the exhaustive
  summary), numbers computed in the text (17,867 edges per feature; 54,208 = 11 × 4,928; 10.3%), outside
  facts (175 M cells, 70 GB), and planned bands (0.34–0.36, 0.55). None of these is traceable to the
  run's own outputs as written.

Paper (`main.tex` body, 124 distinct numbers after exclusions):

| check | found | chance (CI) | excess (CI) |
|---|---|---|---|
| in the union of the 8 runs' outputs | 123 | 0.97 (0.95–1.00) | 0.02 (−0.00–0.04) |
| chain: in run R's FINAL_SUMMARY and in run R's outputs | 119 | 0.50 (0.43–0.56) | 0.46 (0.39–0.53) |
| written in any FINAL_SUMMARY (transcription) | 120 | 0.50 (0.44–0.57) | 0.47 (0.40–0.53) |
| union + audits/ + summaries/ + setup/ data files | 123 | 0.98 (0.95–1.00) | 0.02 (−0.00–0.04) |

The 5 paper numbers without a chain: 175 (pre-training cells; in a summary, not in outputs), 4,096
(context length), 20.4 (AUROC-point gap), 0.0125 (α/4), 0.678 (cross-layer Pearson upper bound).
The last four are found in run outputs (where chance is high) but are written in no FINAL_SUMMARY.

---

## 5. Run manifest

**M1 parameters.** 258 spec rows: 66 linked to run evidence (55 through `param_map.json`, 11 by exact
name); 191 with no evidence found; 1 suppressed as a different parameter. Status of linked rows: 42 match the default, 6 in range but not
the default, 6 out of range, 5 differ where the spec gives no range, 6 are source-model descriptors
(not judged), 1 not machine-readable.

| run | spec parameter | used | spec default | spec range | status | reason written? |
|---|---|---|---|---|---|---|
| attention-grn | n_ctrl | 200 (1B run); 2000 | 2000 | {500, 2000, 10000} | OUT_OF_RANGE | no |
| attention-grn | hvg | 1500 | {2000, 3309} | {1000, 2000, 5000} | OUT_OF_RANGE | no |
| attention-grn | intervention_cells_fidelity | 300 (phase 4); 2000 | 2000 | not machine-readable | DIFFERS | no |
| attention-grn | degree_null_n_curveball | 50 | 200 | not machine-readable | DIFFERS | yes (README.md:32) |
| spectral | autoloop_iterations | 8 | 63 | 40–80 | OUT_OF_RANGE | no |
| topology | null_label_permutation_replicates | 8; 50; 100 | 100–200 | 100–1,000 | OUT_OF_RANGE | no |
| topology | null_rewiring_replicates | 8; 12 | 24 | ≥ 24 | OUT_OF_RANGE | no |
| sae-atlas | n_cells_training (two spec rows) | 500 | 2000 / 3000 | not machine-readable | DIFFERS | no |
| sae-atlas | causal_patching_cells | 50 | 200 | not machine-readable | DIFFERS | no |
| exhaustive-mapping | n_cells_per_condition | 50 | 200 | 100–500 | OUT_OF_RANGE | no |

Longevity's 10 automatic matches all agree with the spec, but 7 of them exist only in the Stage-3
script, and the run summary says Stages 2–5 were not run (the run stopped at gate G1).

**M2 gates** (`results/run_manifest_gates.csv`): 18 gate objects. 11 consistent with nothing else to
note. 2 manifold internal-panel objects PASS_WITH_UNCOMPUTED_GATES (permutation gate required on the
internal panel, never computed). 1 consistent but NEGATIVE_CONTROL_PASSED (the first `lung_control`
panel; trustworthiness 0.8013 vs 0.80; superseded by `lung_nonhema`, which failed as expected).
1 consistent but GATE_SET_FOR_PANEL_UNSTATED (zero-shot panel; the frozen rules name only the internal
and external panels). 1 PROTOCOL_K_DIFFERS (H103, k = 5 vs 15). 1 IMPLAUSIBLE_FRACTION (attention K562
C3, −13,421; its baseline AUROC is below 0.5, so the "fraction" divides by a floor of about 10⁻⁶).
1 not recomputed (attention C4: three candidate value keys).

**M3 phases.** Only the attention run wrote a machine-readable `phases_run` list; it ran 6 of the 14
phases the spec names (0b and 5–11 not run). The other 7 runs have no such record.

**M4 reproducibility.** 75 of 75 scripts that use randomness set a seed-like value. The project
`requirements.txt` pins every package with `==`, but it was written on 2026-05-07 17:52, on the audit
day, after the runs. The checkpoint is loaded from a local folder that the README fills from the Hugging
Face `main` branch; no `revision=` and no recorded weight hash. 5 of 8 runs record a device (mps, cpu).
0 of 8 runs have a run manifest or a machine-readable decision/retirement log; 1 run
(topology) has a MANUAL_REVIEW file that overrode coded decisions. All 15 script paths cited in
summaries resolve.

**M5 pointers.** 25 lines in the 13 FINAL_SUMMARY files mention an interval; 6 name a method; 4 name a
method and a resampling unit. 8 of 13 files have a scope/limitation heading. 64 distinct lines (76 term hits) contain a word
from the audit's P9 list (e.g. "causal", "confirmed", "mechanistic").

---

## 6. Verification (second ways)

| what | second way | result |
|---|---|---|
| S1 missing sections | plain string check of `## ` lines; prototype linter from the investigation | 10 of 10 specs agree with both |
| validation-item counts | prototype linter | identical (122 items, 80 with thresholds, 33 "if broken" rules) |
| traced / untraced | different tokeniser + string formatting (Python rounding) instead of interval test | 1,605 of 1,605 numbers agree |
| chance rate | exact share over all grid points instead of 50 random points | mean 0.806 exact vs 0.807 sampled; r = 0.989 over 1,633 numbers; mean absolute difference 0.022 |
| flagged parameter values | raw-text regex on the cited line / JSON file | 35 of 35 values confirmed |
| tracer vs human-style judgment | join with manual audits `numbers_A/B/C.md` | 130 number mentions joined (22 not joinable; 149 manual rows have no number in the value column); MATCH 96 of 98 traced; MISMATCH 10 of 10 and DIFFERENT ENDPOINT 17 of 17 also traced |

The last row is the key limit: a presence check agrees with a careful reader on "the number exists"
and says nothing about "the number means what the text says".

---

## 7. How this differs from what the deployment and the paper said

| deployed claim | what the checks show |
|---|---|
| "The template is enforced: missing or out-of-order sections are rejected before execution." | No code read the specs. A literal check rejects 7 of 8. |
| Quoted "Step 4": Curveball null, 1000 iterations, seed 42, reject at z ≥ 3 | The spec's parameter table asks for 200 Curveball draws (`spec_lint_params.csv`); the run used 50 (M1). The investigation found that the quoted step text is not in the spec. |
| "A named, mandatory slot" for null model, trivial baseline, positive control, scope | No spec has all four labelled; positive control is labelled in 1 of 8 (longevity), not in topology. |
| "Four pre-registered gates" | The frozen gate file declares five gates for the internal panel and four for the external panel. The fifth (permutation) was never computed, yet 2 internal-panel objects record PASS (M2). |
| Checkpoints pinned | No revision pin and no weight hash in the project files; download from `main`. |
| Audit as "machine-applicable checks" | The P1–P10 verdicts were free text by one agent. With this package, parts of P1 (null size), P8 (interval words), P9 (word list) and most of P10 have code; P2–P7 remain judgment. |
| Human spot-checks traced numbers to artefacts | No record exists (investigation). A scripted trace is possible, but in 4 of 8 runs chance alone finds 88–97% of same-format numbers. |
| "2-consecutive-negatives retirement rule" | No run has a machine-readable retirement log; this package does not implement the rule. |

---

## 8. What I could not do, and limits

- **Null and baseline correctness are not checked by code.** The Curveball null that did not rewire
  (investigation `attn_endpoints.md`) would pass every check here except the null-size check. Catching
  it needs an implementation test written into the spec's validation block and data the run saves
  (e.g. how many null draws differ from the input). None of the runs saved that.
- **Most parameter rows are unlinked** (191 of 258). The links that exist are agent judgment
  (`param_map.json`). Exact-name matching alone found 11.
- **Snapshot by file date.** Deployed files are dated ≤ 2026-05-07 and new work is in `v2_*`; this split
  is clean here but would not survive a copy that resets dates.
- **Binary arrays are not read** (`.npy`, `.npz`, `.h5ad`), so numbers that exist only there are untraced.
- **The chance model** assumes a "random number" is uniform on the precision grid in [0.5x, 1.5x]. A
  different window would give somewhat different rates; the exact-grid check agrees with the sampled one.
- **The paper's prose claims about the framework are not parsed.** Section 7 is my reading, using the
  check outputs.
- The cross-check with the manual audits joined 130 manual number mentions; 22 could not be joined
  (values the extractor drops, such as integers below 10, or written differently in the paper), and
  149 manual rows have no number in their value column.
- I did not check the public GitHub repository, did not run any model, and did not edit any spec, run,
  summary, audit or paper file.

---

## 9. Plain-words summary

I built three checkers and ran them on the eight deployed specs and runs. The spec checker shows that
seven of the eight specs would have been rejected by the paper's own template rule, that code pointers
almost never use the required form, and that most pinned clones cannot be tied to their commit. The run
checker shows ten parameters where a run used a different value than its spec (mostly smaller nulls and
fewer cells), with a reason written down only once, and two internal-panel gate verdicts that say
"pass" although one required gate was never computed. The number tracer shows that most summary numbers can be found in the
outputs, but random numbers of the same shape are found almost as often, so this is weak evidence; the
paper-to-summary-to-output chain is more useful. Most importantly, every number a careful reader judged
wrong in meaning still passes the presence check. Code can make missing pieces and silent deviations
visible. It cannot decide whether an analysis is right; that stays with an agent's reading and a human's
decision.

---

## Verification notes (independent check by a second agent, 2026-10-01)

Sections 1–8 above were corrected where this check found errors. Each change is listed here.
The code changes are in `verification_review/code_changes.diff`. My own re-derivation is
`verification_review/independent_recount.py` → `verification_review/results.json`, with the sha256 of
every input in `verification_review/run_config.json` (seed 20261001, CPU only, no model run).

### What I did

- Re-ran the whole package from a copy in a scratch folder, with every value cache rebuilt from the raw
  run files. All outputs of `spec_lint.py`, `number_trace.py`, `run_manifest_check.py` and the five
  `verify/` scripts were byte-identical to the delivered ones. The input hashes recorded in the three
  `*_run_config.json` files (1,611 + 19 + 118 files) still matched the files on disk.
- Re-derived the key numbers with my own code (V1–V5 in `independent_recount.py`).
- Read the flagged values in the scripts, logs and JSON files by hand.
- Tested `--strict`: all three main scripts exit 1 on the deployment, as the README says.

### Confirmed without change

- S1: 7 of 8 specs fail (my own parse of the `## ` headings). Code pointers: 230 total, 5 in the
  required form, 165 resolve as written, 53 by suffix or name, 2 do not resolve, 10 wildcards, 174 in
  clones without `.git`, 4 functions not in the cited file (they exist in other files of the same clone).
- Pins: 6 of the 8 specs point to clones with no `.git`; longevity's HEAD equals its pin.
- Deviation values: Curveball 50 (`phase3_residualization.py`), label-permutation N_NULL 8 and 50,
  rewiring N_REWIRE 12 and 8, SAE patching N_CELLS 50 (`phase6.log`: "cells: 50"), phase-4 `n_ctrl` 300,
  exhaustive 50 cells per condition, spectral loop "iters 2..9" (8, `autoloop_master_log.md`).
- C3 −13,421 is (0.4866 − 0.5) divided by a floor of 10⁻⁶ (`phase3_residualization.py:206-207`).
- Snapshot rule: outside `v2_` paths, only 4 files under runs/, setup/, summaries/ and audits/ are dated
  after 2026-05-07 (two `V2_*.md` reports, `setup/hooks_v2.py`, `setup/test_hooks_v2.py`). All four are
  excluded by date. No post-deployment file leaks into the evidence.

### Errors found and fixed

1. **Wrong unit for the pooled tracing numbers.** The headline "1,587 of 1,647 ... unit = distinct
   number" counted 461 repeats: the same number written in two copies of one run's summary. The
   interval also treated numbers as independent, but numbers from one run share one corpus.
   Fix: `summarize_results.py` now adds `number_trace.distinct_per_run` with a bootstrap over the 8 runs.
   New headline: 1,132 of 1,186 distinct numbers traced (95.4%, Wilson 94.1–96.5%); chance 0.78
   (bootstrap over runs 0.65–0.89); excess 0.18 (bootstrap over runs 0.08–0.27). The bootstrap over
   numbers (0.16–0.19) is too narrow. My recount (V3) gives the same point values.
2. **The gate check ignored the frozen panel rule.** `quality_gates_spec.json` says a branch must pass
   "ALL five gates on the internal panel" and "ALL four gates on the strict non-overlap external panel".
   The checker asked the external-panel objects for the permutation gate too. Fix in
   `run_manifest_check.py`: required gates now follow the panel; a panel the rules do not name (the
   zero-shot panel) gets GATE_SET_FOR_PANEL_UNSTATED, which does not block. PASS_WITH_UNCOMPUTED_GATES
   falls from 5 objects to 2 (LET anchor and H38-lite, both internal). My recount (V5) agrees.
   I also corrected the report's framing of `lung_control`: the run itself found that this first lung
   panel held lung immune cells and replaced it by `lung_nonhema`, which failed as expected. The repair
   queue falls from 96 to 95 items (M2: 8 → 7).
3. **The manual-audit cross-check misread one file and missed repeats.** `numbers_C.md` has an extra
   first column ("#") and a last "Notes" column. The script used columns 0, 2 and the last one for every
   table, so no `numbers_C` row was ever joined. It also joined a manual row only when the number's
   FIRST mention in the paper was within ±3 lines. Fix in `verify/crosscheck_manual_audit.py`: columns
   come from each table's header; any mention line counts. Joined mentions: 59 → 130 (22 not joinable).
   Judged wrong in meaning: 18 → 27 mentions (23 distinct values); all are still "found". NOT FOUND is
   still one number ("15 hours", 4 mentions), also "found" by chance.
4. **The paper extractor dropped a number followed by the word "in".** The rule that blanks LaTeX
   lengths (`0.2in`, `4pt`) allowed a space, so "0.59 / in the source study" (main.tex:1127) was blanked.
   Fix in `numtext.py`: the unit must be attached. Paper numbers 123 → 124; union 123/124; chain
   119/124 (chance 0.50, excess 0.46, 95% CI 0.39–0.53); transcription 120/124. No summary number changed.
5. **"76 lines" with audit P9 words** were 76 term hits on 64 distinct lines. The summary now records
   both; the report and the requirements table quote 64.
6. Smaller text fixes: the `n_ctrl` deviation is only in the 1B side run; the Tabula Sapiens
   `n_cells_training` row is linked to the K562 constant (the run's Tabula Sapiens step also used 500
   cells); the C3 denominator is a 10⁻⁶ floor; H103's k = 5 has its reason written in the file.

Files changed: `numtext.py`, `run_manifest_check.py`, `summarize_results.py`, `requirements_table.py`,
`verify/crosscheck_manual_audit.py`, `README.md`, this report, and the regenerated outputs
(`REQUIREMENTS_TABLE.md`, `requirements_table.csv`, and in `results/`: `number_trace_*`,
`run_manifest_gates.csv`, `run_manifest.json`, `run_manifest_stdout.txt`, `crosscheck_manual_audit.*`,
`checker_summary.*`, `repair_queue.csv`, `*_run_config.json`). No file under runs/, setup/,
summaries/, audits/ or pipelines/ was touched.

### Remaining problems (not fixed)

- **The chance rate depends on the null.** A second null (V4): numbers written in OTHER runs'
  summaries, searched in this run's outputs, re-weighted to this run's mix of number types. It gives a
  pooled chance of 0.68 (bootstrap over runs 0.56–0.80) instead of 0.78, and an excess of 0.27
  (0.16–0.38) instead of 0.18. It is lower in 6 of 8 runs (topology 0.60 vs 0.89; manifold 0.52 vs 0.73),
  about equal in exhaustive-mapping, and higher in circuit-tracing (0.78 vs 0.58). Under both nulls,
  "found in the outputs" is weak evidence in the large runs. The exact size of the excess is not settled.
- **"Trivial baseline" slot (S7) uses the literal phrase.** Counting any heading or bold label that
  contains "baseline", 4 of 8 specs have one (attention, topology, manifold, 03), not 1. The headline
  "no spec has all four slots" still holds, because only longevity labels a positive control and it
  has no null label.
- **Parameter links stay judgment.** `param_map.json` is keyed by name, so duplicate spec rows share one
  link. The reason finder counts "paper uses" as a reason; for Curveball it points to README.md:32,
  which only notes the change; the reason ("Scoped down for compute budget") is at README.md:87. It does
  not read FINAL_SUMMARY files. The rewiring-null spec value "24 (across 6 refinement iterations)" is a
  total over iterations, so comparing one phase's 12 or 8 with it is a judgment call.
- The seed check (M4) only finds a seed statement in the same script as a random call. The
  requirements.txt date is a last-modified time, not a creation time.

### Plain-words summary

The checker package runs and gives the same files when re-run from scratch. Most of its numbers are
right. I fixed five errors. The pooled tracing figure counted many numbers twice and gave too narrow
an interval. The gate check asked external-panel results for a gate the frozen rules only require on
the internal panel, so "5 verdicts without a required gate" is really 2. The manual-audit cross-check
skipped one of the three audit files and most repeated numbers; with them fixed, 27 of 27 mentions that
a reader judged wrong still pass the presence check. One paper number was lost by a LaTeX rule. One
count mixed term hits with lines. What stays open: how often a number is "found" by chance depends on
which random numbers you compare with (0.68 or 0.78), and the parameter links remain an agent's reading.
