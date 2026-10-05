# MaxToki checker package (deterministic checks)

This folder holds code that checks the pipeline specs and the eight deployed MaxToki runs
without any model, GPU or agent. Every check reads files and writes only to `results/`.
It never edits `pipelines/`, `runs/`, `setup/`, `summaries/` or `audits/`.

Nothing in this folder existed during the deployment (April-May 2026). It was built on
2026-10-01 to measure what a deterministic layer can and cannot catch.

## How to run

```bash
cd projects/maxtoki/checks
./run_all.sh                     # 3 minutes (warm cache) to 7 minutes (cold disk), CPU only
# or one at a time:
../.venv/bin/python spec_lint.py            [--strict]
../.venv/bin/python number_trace.py         [--strict] [--runs RUN ...] [--skip-paper] [--rebuild-cache]
../.venv/bin/python run_manifest_check.py   [--strict]
```

Without `--strict` each script is a report and exits 0. With `--strict` it exits 1 when a blocking
rule fails (listed below). No pipeline calls these scripts yet. Wiring them in as a pre-run step and a
pre-summary step is a decision for a human.

Python: `projects/maxtoki/.venv` (standard library + numpy only). Peak memory measured for
`number_trace.py`: about 0.46 GB. `verify/verify_number_trace.py` builds string sets for up to 4 million
values per run and takes about 4 minutes; its memory was not measured.
Seed: `20261001` for every random step (`common.SEED`).

## What is checked (the deployment snapshot)

- Specs: the eight deployed specs (`pipelines/attention-grn-extraction-and-evaluation.md`,
  `residual-stream-spectral-geometry.md`, `topology-geometry-141-hypotheses.md`,
  `manifold-discovery-extraction-compactification.md`, `longevity-mechinterp-donor-aware.md`,
  `sparse-autoencoders/01-…`, `02-…`, `03-…`), plus the audit spec and `04-atlas-deployment.md`
  for comparison (labelled "not deployed").
- Runs: the eight folders under `projects/maxtoki/runs/`.
- **Snapshot rule.** Only files with a modification time before 2026-05-08 00:00 and with no `v2_`
  folder or file prefix count as run evidence. Every deployed file is dated on or before 2026-05-07;
  the correction work started on 2026-10-01 writes into `v2_*` paths. So the deployed claims are
  checked against the deployed outputs. (`common.in_deployment_snapshot`.) File dates can be wrong
  after a copy; this rule is a heuristic that happens to split this project cleanly.
- Evidence file types: `.json .csv .tsv .log .txt .jsonl`. Markdown, HTML and Python files never
  count as evidence, so prose (or a number typed into a script) cannot vouch for itself.
  Binary arrays (`.npy`, `.npz`, `.pt`, `.h5ad`) are not read.

## spec_lint.py — lint of the pipeline specs

| id | What it checks | Blocking under `--strict` |
|---|---|---|
| S1 | The ten template sections (Overview, Source, Inputs, Outputs, Dependencies, Methodology, Code references, Parameters, Validation, Known pitfalls) are level-2 headings, in that order, numbered 1-10; extra sections only after 10. | yes |
| S2 | Each present section has >= 200 characters and no TBD/TODO marker. | no |
| S3 | Source section has a 40-hex commit that is in `repos/<name>/PINNED_COMMIT.txt`; if the clone has `.git`, the commit exists and equals HEAD. "No repository / N/A" text is a WAIVER for a human. | FAIL only |
| S4 | Every code pointer: form (`repos/<name>/path:LINE` is the required form; other forms counted), file exists (as written, by unique path suffix, or by unique file name inside the pinned clone), LINE within the file, and a named function/class after the pointer is defined in that file (and within ±3 lines of LINE for the first one). | a path that does not resolve |
| S5 | Parameters table: present, has a range column, unique names; counts defaults and ranges that parse as numbers. | no |
| S6 | Validation section: items split and classed (numeric threshold / number only / no number), items anchored to the source model, "if X then broken" rules; PASS only if a machine-readable block (fenced YAML with `validation:`) exists. | no |
| S7 | Labelled slots (a heading, or a line that starts with a short bold label) for null model, trivial baseline, positive control and scope statement. Plain mentions are counted separately. | no |

**Does NOT establish:** that any section is correct or enough; that a clone without `.git` holds the
pinned commit (line checks run on the working tree); that a labelled slot contains a real control;
that the executor read or followed the spec. The number parser for table cells is conservative:
a cell it cannot read is reported "not machine-readable", never guessed.

Outputs: `spec_lint.json`, `spec_lint_summary.csv`, `spec_lint_coderefs.csv`, `spec_lint_params.csv`,
`spec_lint_validation_items.csv`, `spec_lint_run_config.json`.

## number_trace.py — are the written numbers in the run outputs?

1. Numbers are pulled from each run's FINAL_SUMMARY file(s) (`summaries/<run>-FINAL_SUMMARY*.md` and
   any `FINAL_SUMMARY*.md` inside the run folder) and from the paper body (`paper-plos-one/main.tex`,
   between `\begin{document}` and the bibliography). Identifiers (L6, H65, P1, MaxToki-217M), label
   numbers (Fig 2, Table 3, Phase 4, layer 6), dates, hashes, arXiv ids, code spans and integers
   below 10 are excluded and listed with the reason (`numtext.py`).
2. A number written as x with d decimals is **traced** if some artefact value v satisfies
   |v − x| ≤ 0.5·10^−d (scaled for 10^k and M/K suffixes). For numbers written with `%`, 100·v also
   counts. Integers must match exactly; integers with trailing zeros get an extra "relaxed" column.
3. Run summaries are searched in that run's outputs. Paper numbers are searched four ways: in the
   union of the 8 runs; as a **chain** (the number is written in run R's FINAL_SUMMARY and found in
   run R's outputs); in the numbers written in any FINAL_SUMMARY (a transcription check); and in an
   extended set that adds `audits/`, `summaries/` and `setup/` data files.
4. **Chance matches.** Each number is replaced by 50 random numbers with the same written precision
   in [0.5x, 1.5x] (never x itself); the share of them that are also "found" is that number's chance
   rate. A traced number with chance rate ≥ 0.5 is flagged WEAK.
5. Intervals: traced share — Wilson 95% interval, unit = distinct number in a document. Mean chance
   rate and "excess" (traced − chance, paired per number) — percentile bootstrap 95%, 2,000
   resamples of numbers, seed 20261001. Pooled over runs, `summarize_results.py` counts each number
   once per run (two copies of the same summary are not double-counted) and adds a run-level
   bootstrap (2,000 resamples of the 8 runs), because numbers from one run share one corpus.

`--strict`: exit 1 if any FINAL_SUMMARY number that is not also written in the spec is untraced.

**Does NOT establish:** that a traced number is correct, belongs to the claim it sits in, or came from
the file where it was found (a match can be a coincidence — that is what the chance rate measures);
that an untraced number is wrong (it may be computed in memory, stored in a binary array, quoted from
another run, or quoted from the source paper — the `also_in_spec` column marks numbers that also
appear in the spec). It never checks meaning: a number from the wrong endpoint is still "found".

Outputs: `number_trace_summary.csv`, `number_trace_numbers.csv`, `number_trace_paper_numbers.csv`,
`number_trace.json`, `number_trace_run_config.json`, and a value cache in `results/cache/`.

## run_manifest_check.py — what each run actually used

The deployed runs wrote no run manifest, so this check rebuilds one from scripts (Python `ast`:
assignments, environment-variable defaults, keyword arguments, argparse defaults; assignments inside
an `if` block are listed but never judged) and from numeric settings in output JSON.

| id | What it checks | Blocking under `--strict` |
|---|---|---|
| M1 | For each spec Parameters row: value(s) used vs the parsed default and range → MATCH_DEFAULT, IN_RANGE_NOT_DEFAULT, OUT_OF_RANGE, DIFFERS_NO_RANGE_GIVEN, SPEC_NOT_MACHINE_READABLE, DESCRIPTOR_NOT_JUDGED, NO_EVIDENCE_FOUND. Recorded JSON values outrank script constants. For a deviation, looks for a line in the run's scripts/README naming both values and a reason word. | a deviation with no written reason |
| M2 | Gate recomputation: JSON objects with a pass flag are re-evaluated from their own numbers against thresholds in the same file or the run's frozen gates file. Flags INCONSISTENT, PASS_WITH_UNCOMPUTED_GATES (a gate required for that object's panel has no value), NEGATIVE_CONTROL_PASSED (`expected_to_fail: true` but passed), PROTOCOL_K_DIFFERS (metric key encodes a different k than the frozen protocol), IMPLAUSIBLE_FRACTION, boundary values within 0.005. Required gates follow the frozen file's panel rule where it states one (manifold: five on the internal panel, four on the external panel); a panel the rule does not name gets GATE_SET_FOR_PANEL_UNSTATED (pointer for a human). | INCONSISTENT, PASS_WITH_UNCOMPUTED_GATES, NEGATIVE_CONTROL_PASSED, IMPLAUSIBLE_FRACTION |
| M3 | Phases named in the spec vs a machine-readable `phases_run` list, if the run wrote one. | no |
| M4 | Reproducibility records: seed set in scripts that use randomness; environment file and pinning; device recorded; checkpoint loads with a revision pin, download refs, recorded weight hash; script paths cited in summaries resolve; timing fields; decision/retirement logs; manual reclassification files; run manifest. | no |
| M5 | Pointers for a reader: lines with an interval and whether they name a method and a resampling unit; scope/limitation headings; causal/mechanistic words from the audit checklist's P9 list. | no |

**The link between a spec row and a run value is partly judgment.** `param_map.json` lists, for
58 spec rows, where the used value is recorded and why that link is right; it also marks rows as
`descriptor` (a property of the source model, e.g. Geneformer layer numbers, not judged) or
`not_same_parameter` (suppresses a wrong exact-name match). It was written by an agent on
2026-10-01 by reading the scripts. Rows with no curated entry use exact-name matching only.
A run that writes a manifest with the spec's own parameter names removes this judgment.

**Does NOT establish:** that a matched constant was the value in force when a reported number was
produced (scripts can change after a run); that unlinked rows (most of them) were followed; that an
in-range value is adequate; that a gate threshold is sensible; that a flagged word is wrong in
context; that a null or baseline is valid.

Outputs: `run_manifest_params.csv`, `run_manifest_gates.csv`, `run_manifest_phases.csv`,
`run_manifest_repro.csv`, `run_manifest_ci_scope.csv`, `run_manifest_wording.csv`, `run_manifest.json`,
`run_manifest_run_config.json`.

## Verification (second ways) — `verify/`

| script | What it re-derives | How it differs from the main check |
|---|---|---|
| `verify_spec_lint.py` | Missing sections; validation-item counts | Plain string tests on `## ` lines; also compares with the prototype linter from the investigation (`projects/maxtoki/verification/contract/`). |
| `verify_number_trace.py` | Traced / untraced for every plain decimal and integer in the run summaries | Different tokeniser; formats each artefact value to the written decimals as a string (Python rounding) instead of the interval test. |
| `verify_chance_rate.py` | Chance rate per number | Exact share over all grid points in [0.5x, 1.5x] instead of 50 random ones. |
| `crosscheck_manual_audit.py` | Tracer result for paper numbers the manual audits (`numbers_A/B/C.md`) judged | Compares with human-style judgments of meaning (MATCH / MISMATCH / DIFFERENT ENDPOINT / NOT FOUND). Column positions are read from each table's header; a manual number joins to the tracer's number at any line where that number occurs in the paper. |
| `verify_manifest_deviations.py` | Every value behind a flagged parameter deviation | Raw-text regex on the cited script line or JSON file instead of `ast`/`json`. |

## Other files

- `summarize_results.py` → `results/checker_summary.json`, `results/checker_summary.md` (all numbers
  quoted in `CHECKER_REPORT.md`) and `results/repair_queue.csv` (one row per deterministic failure,
  with the owner: agent, human, or both).
- `requirements_table.py` → `REQUIREMENTS_TABLE.md` and `requirements_table.csv`: every framework
  requirement, what was in place in the deployment, and how it is checked now (code / agent
  judgment / human decision).
- `common.py`, `spec_parse.py`, `numtext.py`, `corpus.py`, `run_evidence.py`: shared helpers.
- Each main script writes `<name>_run_config.json`: tool hash, time, Python version, seed and the
  sha256 of every input file (files over 200 MB: size and date only, stated in the record).

`verification_review/` holds the independent re-derivation written by the verifying agent
(`independent_recount.py` -> `results.json`, `run_config.json`); run it after `run_all.sh`.

Not part of this package: `DEPLOYMENT_FACTS.md`, `deployment_facts.json` and `d12_deployment_facts/`
in this folder were written by a separate task (the hardware/compute ledger).

## Known limits

- The runs are a moving target: other work is adding `v2_*` files. The snapshot rule excludes them.
- The number parser for spec cells and the number extractor for prose are heuristics. Their
  outputs list every parse, so a reader can see each decision.
- Chance rates are high in large output folders (0.9+ for 2-decimal numbers), so "traced" is weak
  evidence there. The informative outputs are the untraced list, the WEAK flag and the chain check.
- The checks do not read the paper's prose claims about the framework (e.g. "the template is
  enforced"). Comparing those claims with these results is a human task.
