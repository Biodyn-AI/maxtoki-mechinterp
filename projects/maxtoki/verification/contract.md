# KEY=contract — Is the "executable contract" enforced? What is automatic, what is judgment, what is human?

Reviewer 2 asked: which requirements are checked by code, which by agent judgment, how breaks are
found, how fixes are triggered, and which decisions need a human. This report answers from the files.

Everything below marked **[verified]** I read in a file or computed with a script. Anything marked
**[inferred]** is my reading of the evidence.

---

## 1. Results first

1. **No spec checker exists.** [verified] I searched 1,633 script files in the repository (all `.py`, `.js`,
   `.sh`, `.ts`, `.yml`, `.toml` outside `repos/`, `node_modules`, venvs). None parses a pipeline spec or checks
   its sections. There are no Claude Code hooks, no slash commands, and no CI for specs. The top-level
   folder is not even a git repository, so there are no git hooks either.
2. **A literal checker would have rejected 7 of the 8 deployed specs.** [verified] I wrote one now
   (`contract/spec_lint.py`). Seven of the eight specs used on MaxToki have no "Code references" section.
   Only the longevity spec passes. All eight were executed anyway. So the paper's sentence
   *"The template is enforced: missing or out-of-order sections are rejected before execution"*
   (`paper-plos-one/main.tex:587-588`) is false.
3. **The "2-consecutive-negatives retirement rule" is not in any code that ran on the two loops the paper names.**
   [verified]
   - The only code with a "2 consecutive" rule is the spectral-geometry loop driver. It counts executor
     *crashes or missing files*, not negative results. It fired on network errors and stopped the loop after 4 iterations.
   - The topology-141 "loop" is one Python script with 6 hard-coded hypotheses and a **3**-in-a-row stop rule.
     The rule never fired. A later manual review changed 4 of the 6 automatic decisions.
   - The manifold-discovery run used Claude Code's `/loop` wake-ups and a hand-kept `STATUS.md`. It has no retirement code.
   - The paper's example, *"H115 and H118 were retired after two negatives each"* (`main.tex:1615-1617`),
     cannot be traced: **H115 and H118 appear in no run artefact.** They appear only in the paper texts.
4. **The validation sections are not usable as gates on a new model.** [verified] 7 of 8 Validation sections say
   the run "must reproduce the paper's sanity-check signatures" on the *source* model (scGPT or Geneformer)
   "before any new-model claim is trusted". No MaxToki run did this pre-flight. Instead the source numbers
   were reused as things to compare MaxToki against.
5. **Several other "contract" claims in the paper do not match the files.** [verified] The quoted "Step 4"
   null-model rule is not in the spec. The run used a smaller null than the spec asked for. The "mandatory
   positive-control slot" does not exist. The audit spec calls itself "reading + judgment, not running".
6. **A deterministic layer is cheap to build.** [verified] My prototype spec linter runs in under 1 second.
   My prototype run-artefact checker runs in about 2 minutes over all 8 runs. Section 7 gives a full design.

---

## 2. What I searched and what I found

| Place searched | What is there | Checks specs or runs? |
|---|---|---|
| `<REPO_ROOT>/.claude/settings.local.json` | 4 allow-listed Bash commands (pandoc, `python3 check.py`). No hooks. | No |
| `~/.claude/settings.json` | env var, model, plugins. No hooks. `~/.claude/hooks` and `~/.claude/commands` do not exist. | No |
| Git hooks | Top-level folder is not a git repo (`git rev-parse` fails). Sub-project `.git/hooks` only in other projects. | No |
| CI | Only `projects/maxtoki/runs/sae-atlas-217M/atlas/.github/workflows/deploy.yml` (GitHub Pages build of the web atlas: `npm ci`, `npm run build`). | No |
| `projects/maxtoki/paper-plos-one/build/check_plos.py` | Checks the **manuscript** against PLOS formatting rules (template class, line numbers, no graphics). | No (paper only) |
| `projects/maxtoki/paper-biosystems/docx-build/check.py` | Checks a .docx against the LaTeX and PDF. | No (paper only) |
| `projects/maxtoki/audits/audit_a3_bootstrap_cis.py` | Remediation script: bootstrap CIs for 3 headline numbers (2 skipped). | Recomputes numbers; does not check specs |
| `projects/maxtoki/runs/spectral-geometry-217M/autoloop/run_maxtoki_autoloop.py` | The only real executor/brainstormer loop driver. See §3.1. | Checks that output files exist |
| `projects/maxtoki/runs/topology-141-217M/scripts/phase11_autoloop.py` | Scripted batch of 6 hypotheses with a coded decision rule. See §3.2. | Codes one decision rule |
| `projects/maxtoki/runs/manifold-discovery-217M/STATUS.md` | Tracker read by `/loop` wake-ups (line 3). `iterations/` is empty. | No |
| `prompts/reviewer-1-*.md`, `reviewer-2-*.md`, `brainstormer-ideas.md` | Role prompts for reviewing papers. The audit report says they were out of scope (`audits/audit-20260507.md:455-458`). | No |
| `projects/biotensor/codebase/route_genemanifold/loop_cycle_workflow.js` | Later project (dated 2026-07-20), **not part of this paper**. Even there, gate pass/fail is read out of JSON by an LLM call with a schema (lines ~463-478), and the scope ban is prompt text. | Partly |

Rules about the template exist only as prose:
- `CLAUDE.md:45` and `AGENTS.md:21`: "must include these ten sections in order ... check all ten are present and substantive" — an instruction to an agent.
- `README.md:33`: "should include these sections" — weaker wording.
- `CLAUDE.md:58`: code must be cited as `repos/<name>/path/file.py:LINE`.
- `README.md:29` describes the audit pipeline as "Methodology document only — no scripts."

---

## 3. The autonomous loops: what the code actually does

### 3.1 Spectral-geometry loop — the only real two-agent loop [verified]

File: `projects/maxtoki/runs/spectral-geometry-217M/autoloop/run_maxtoki_autoloop.py`
(note: the paper says the loop was used for topology-141 and manifold-discovery, not this pipeline).

What is deterministic (code):
- Spawns `claude --model sonnet --print --permission-mode bypassPermissions --max-budget-usd 5` for each role (lines 113-120), 30-minute timeout (line 47).
- `_validate_executor_output` (lines 145-172) checks only: report file exists; screen JSON exists and parses; `hypotheses` list is non-empty; at least one csv/json/npy file exists. It does not look at content, numbers, nulls, or controls.
- Stop rule (lines 234-263): stop after **2 consecutive executor failures**, where "failure" = the file check above fails. Also stops on a `runtime/STOP` file or max iterations.
- `_all_prior_hypotheses` (lines 86-108) pastes a list of earlier hypotheses and their `retired` flag into the next prompt. **The code never acts on `retired`.**

What is agent judgment:
- Retirement itself. The executor prompt says: "if a direction ... has ≥ 2 negative/inconclusive outcomes with adequate controls, mark it as `retired` in the JSON" (`autoloop/prompts/executor_prompt.md:39`). The brainstormer prompt asks it to mark `retire_now` (`prompts/brainstormer_prompt.md:30`).
- "Adequate controls", "near-identical method", "negative" are all left to the model.

What actually happened (`autoloop/autoloop_master_log.md:2-9`, `runtime/*.log`):
- iter_0002: executor rc=0, valid; brainstormer rc=0.
- iter_0003: executor **timed out (rc=124) but still passed validation**; brainstormer rc=1 — log ends with `API Error: Unable to connect to API (UNKNOWN_CERTIFICATE_VERIFICATION_ERROR)`.
- iter_0004 and iter_0005: executor rc=1 — log ends with `API Error: Unable to connect to API (ECONNRESET)`. Two in a row → `ABORT after 2 consecutive failures`.
- All 6 hypotheses in the two screen JSONs have `"retired": false`. Nothing was retired by the loop. The brainstormer's roadmap (iter_0002) did write `retire_now` for 6 directions — as prose in a markdown table.

So the one "2 consecutive" rule in code is a crash guard. It fired on network errors.

### 3.2 Topology-141 "autoloop" [verified]

File: `projects/maxtoki/runs/topology-141-217M/scripts/phase11_autoloop.py`.
- Docstring (lines 1-20): "autoloop substitute ... runs a curated batch of 6 additional hypotheses". The 6 are hard-coded in `ITERATIONS` (from line 499). No LLM is called.
- Coded decision rule (lines 95-113): PROMOTE if positive null-gap in ≥4 layers in ≥2 domains; INCONCLUSIVE if any domain positive; else RETIRE.
- Stop rule (lines 517-540): **3** consecutive RETIRE, not 2. It never fired.
- Code output (`outputs/phase11_autoloop/autoloop_summary.json`): 0 PROMOTE, 5 INCONCLUSIVE, 1 RETIRE_no_signal.
- Then `outputs/phase11_autoloop/MANUAL_REVIEW.md:42-51` reclassified: 2 INCONCLUSIVE → RETIRE, 1 INCONCLUSIVE → "NOVEL POSITIVE", 1 RETIRE → "untestable". The FINAL_SUMMARY reports the manual labels (`runs/topology-141-217M/FINAL_SUMMARY.md:468-478`).
- So 4 of 6 final decisions were changed by judgment after the coded rule ran. [inferred: the review was written by the agent, since no human sign-off is recorded.]

### 3.3 Manifold-discovery [verified]

- `runs/manifold-discovery-217M/STATUS.md:3`: "Each `/loop` wake-up reads this, picks the next unfinished sub-task, executes it, then updates this file." `planning/research_plan.md:72`: "Driven by the `/loop` skill."
- `iterations/` is empty. There is no retirement code.
- Hypothesis IDs in scripts, STATUS and FINAL_SUMMARY range from H38 to H112. **H115 and H118 appear nowhere under `runs/`, `summaries/` or `audits/`.** They appear only in `paper/`, `paper-jbi/`, `paper-cbac/`, `paper-deanon/`, `paper-biosystems/`, `paper-plos-one/` main.tex files and in `paper-biosystems/README.md:150`.

### 3.4 What the paper says vs what the code does

| Paper claim | File evidence |
|---|---|
| Loop used for topology-141 and manifold-discovery (`main.tex:635-642`) | Real two-agent loop exists only for spectral-geometry. Topology-141 = scripted batch. Manifold = `/loop` + STATUS.md. |
| "a hypothesis family that fails twice in a row is retired" | Only coded "2 in a row" rule is a crash guard. Topology rule is 3 in a row, and was overridden. |
| "H115 and H118 were retired after two negatives each" (`main.tex:1615-1617`) | No artefact mentions H115 or H118. |
| Public verifiability | The public repo (per the task context) has no `runs/`, so none of this loop evidence is public. |

---

## 4. Spec lint results (the ten-section template)

Script: `contract/spec_lint.py`. Output: `contract/spec_lint_results.json`, `contract/spec_lint_summary.csv`,
`contract/validation_items.csv`. It only reads files.

It checks level-2 headings against the ten names in `CLAUDE.md:45-56`, checks order, checks numbering,
counts validation items and parameter rows, and checks `repos/...` code citations.

| Spec (8 deployed + 2 others) | Result | Sections present | Missing | Extra | Heading numbers match? |
|---|---|---|---|---|---|
| attention-grn-extraction-and-evaluation.md | **FAIL** | 9/10 | Code references | 10. Quick-start | No (Parameters=7, Validation=8, Pitfalls=9) |
| residual-stream-spectral-geometry.md | **FAIL** | 9/10 | Code references | 10. Quick-start | No |
| topology-geometry-141-hypotheses.md | **FAIL** | 9/10 | Code references | 10. Quick-start | No |
| manifold-discovery-extraction-compactification.md | **FAIL** | 9/10 | Code references | 10. Quick-start | No |
| longevity-mechinterp-donor-aware.md | PASS | 10/10 | — | 11. Porting | Yes |
| sparse-autoencoders/01-sae-atlas.md | **FAIL** | 9/10 | Code references | 10. Quick-start | No |
| sparse-autoencoders/02-causal-circuit-tracing.md | **FAIL** | 9/10 | Code references | 10. Quick-start | No |
| sparse-autoencoders/03-exhaustive-mapping-and-steering.md | **FAIL** | 9/10 | Code references | 10. Quick-start | No |
| audit-recurring-review-issues.md (not one of the 8) | PASS | 10/10 | — | — | Yes |
| sparse-autoencoders/04-atlas-deployment.md (not one of the 8) | **FAIL** | 5/10 | Outputs, Dependencies, Methodology, Code references, Parameters | 3 | No |

Notes:
- All present sections are in the right order. The failure is the missing section, not order.
- In the 7 failing specs, code pointers are scattered inside Methodology as `**Code reference:**` lines.
  So the content partly exists. But a checker that "rejects missing sections" would still reject them.
- `manifold-discovery` §2 has no commit hash, because the source paper has no repo. The "pinned repo commit hash" slot is unfilled for a valid reason. A checker needs an explicit "NONE: reason" value for this.

### Code citations (CLAUDE.md:58 requires `repos/<name>/path:LINE`)

| | 8 deployed specs total |
|---|---|
| Citations in `repos/<name>/...` form | 86 |
| ... of which carry a `:LINE` | **5** (all in 02-causal-circuit-tracing.md: lines 131, 140, 446, 480, 507) |
| Bare paths like `` `src/...py` `` or `` `iter_0009/...py` `` (no repo named) | 67 |
| `repos/` paths that do not exist in the local clone | 1 in 03 (`repos/bio-sae-circuits/results/circuit_analysis.json`, a data path) |

The audit spec itself cites `repos/topology-biomechinterp2/iterations/iter_0009/run_iter0009_screen.py`
(`pipelines/audit-recurring-review-issues.md:684`). That folder holds only `executor_prompt.md`.
This is exactly the audit's own P10 pattern ("cites a script path that doesn't exist at the pinned commit").

### Pins

Spec §2 hashes all match the text in `repos/<name>/PINNED_COMMIT.txt`. But 6 of 10 cloned repos have **no `.git`**
(`bio-sae`, `bio-sae-circuits`, `biomechinterp-framework`, `sae-biological-map`, `topology-biomechinterp1`,
`topology-biomechinterp2`). For those, nothing can prove the files on disk are that commit. Only
`longevity-mechinterp` (spec-cited) has a matching git HEAD (`5a61464...`).

---

## 5. Validation sections: how many criteria can a machine check?

Method (heuristic, in `spec_lint.py`): split each Validation section into top-level items. An item is a
"numeric threshold" if it has a number plus a comparator, a range, or a tolerance (≥, ≤, <, >, ±, "a–b",
"within", "tolerance", "at least"...). Labels like "Phase 3", "L15", "H123" are removed first.
Per-item labels are in `contract/validation_items.csv`.

| Spec | Items | Numeric threshold | Number, no threshold | No number | Has "if X then broken" rule | Parameter rows | Range column? |
|---|---|---|---|---|---|---|---|
| attention-grn | 13 | 12 | 1 | 0 | 2 | 36 | yes |
| spectral-geometry | 22 | 15 | 7 | 0 | 5 | 38 | yes |
| topology-141 | 13 | 10 | 3 | 0 | 6 | 30 | no |
| manifold-discovery | 16 | 3 | 13 | 0 | 3 | 39 | no |
| longevity | 14 | 5 | 3 | 6 | 2 | 30 | yes |
| 01-sae-atlas | 15 | 14 | 1 | 0 | 5 | 36 | yes |
| 02-circuit-tracing | 15 | 10 | 5 | 0 | 6 | 22 | no |
| 03-exhaustive-mapping | 14 | 11 | 3 | 0 | 4 | 30 | yes |
| **Total (8 specs)** | **122** | **80 (66%)** | **36** | **6** | **33** | **261** | 5 of 8 |

So about two thirds of the validation items *could* be turned into code checks. But what they check matters:

- **7 of 8 Validation sections are source-model replication targets.** Their first line says so, e.g.
  - attention-grn (line 777): "must reproduce the paper's sanity-check signatures on the reference datasets before any new-model claim is trusted. If any signature deviates beyond the tolerance, the adapter or the environment is wrong."
  - spectral (line 500): "... on scGPT + Tabula Sapiens immune ..."
  - 01-sae-atlas (line 623): "... on Geneformer V2-316M + K562 ..."
  - 03 (line 614): "... on Geneformer V2-316M ..."
  - Only longevity uses mostly model-agnostic sanity rules (e.g. `frac_active` between 0.05–0.5; `l1_prop_distance_to_target` < 0.05).
- **No MaxToki run did this pre-flight.** I found no run file that mentions the "sanity signatures" or a source-model reproduction step. Instead the numbers became MaxToki comparison targets. Example: `summaries/spectral-geometry-217M-FINAL_SUMMARY.md:19-37` shows "replicated" / "NOT replicated" rows against the source numbers.
- **Taken literally, the contract would have declared runs broken.** The spectral summary has several "NOT replicated" rows. The spec says a deviation means "the adapter or the environment is wrong". Nobody applied that rule. That was a judgment call (a correct one, since a new model should differ).
- The truly portable checks are the "if X then the implementation is broken" rules on nulls and controls. Examples: topology item 3 (rewiring null must give 0/24 significant), spectral item 4 (feature-shuffle null ER must not be near the observed value), attention item 8 (MLP ablation at the wrong layer must move AUROC), 03 item 13 (|Δs| > 0.1 means α is misapplied). The heuristic finds 33 items with such a rule. Many of those still quote source-model magnitudes.

**Plain reading:** the Validation sections were written to check a *reproduction* of the source paper.
They were not written as pass/fail gates for a *new* model. That is why no one could enforce them.

---

## 6. Other paper statements about the contract that the files do not support

1. **The quoted "Step 4 (null model)"** (`main.tex:603-610`): "Curveball ... (1000 iterations, seed=42) ... Reject the null at z≥3.0 AND p_BH≤10⁻⁴."
   - The spec has no such step. The real text is Phase 3 step 3 (`pipelines/attention-grn-extraction-and-evaluation.md:213`): "Generate **200** permuted GRNs using the curveball algorithm ... 1,000 additional permutations under the label-shuffling null." It lists the source paper's result (z = 3.63). It has **no rejection threshold** and no seed. Parameter table line 757: `degree_null_n_curveball` = **200**.
   - The MaxToki run used **50**: `runs/attention-grn-217M/scripts/phase3_residualization.py:9-10` ("n=50 here, vs n=200 in the source paper — scoped down for compute budget") and line 54. The outputs record `"curveball_null_iters": 50`.
   - So the paper's point that "the agent has no latitude over what counts as the null" (`main.tex:617-621`) is contradicted: the agent changed the null size and wrote down why.
2. **Another parameter deviation** (same kind): topology spec `null_label_permutation_replicates` = 100–200, valid range 100–1,000 (`pipelines/topology-geometry-141-hypotheses.md:514`). `runs/topology-141-217M/scripts/phase10_h139_sectional_anisotropy.py:57` uses `N_NULL = 8` for its label-permutation null (lines 234-237).
3. **"each is given a named, mandatory slot with non-empty content"** for null model, trivial baseline, positive control, scope qualifier (`main.tex:594-600`). The ten-section template has no such slots.
4. **"the mandatory 'positive control' slot in the topological-screening pipeline forced the agent to construct an explicit chance baseline"** (`main.tex:1605-1609`). The phrase "positive control" occurs **0 times** in `pipelines/topology-geometry-141-hypotheses.md`. The topology positive control was added **after** the run as audit action A1 (`runs/topology-141-217M/FINAL_SUMMARY.md:319`, "added 2026-05-07, audit action A1"). The audit had flagged "no run has a synthetic-graph positive control" (`audits/audit-20260507.md:33`).
5. **Audit as "machine-applicable checks"** (abstract, `main.tex:222`) and "a diagnostic signature (how to detect it mechanically)" (`main.tex:679`). The audit spec says the pass is "reading + judgment, not running" (`pipelines/audit-recurring-review-issues.md:662`). Its workflow is "Read ... Spot-check ... Record verdict" (lines 620-626). The audit report says it "stops at the markdown layer for most claims" and ~30 load-bearing claims were not checked against JSON or scripts (`audits/audit-20260507.md:438-445`). Some diagnostic signatures *could* be mechanical (P10 path checks, P1 grep for plain `np.random.permutation`), but no code implements them.
6. **Human decisions are not logged.** The paper describes directional, methodological and gating decisions (`main.tex:645-660`). The only recorded human decision I found in run files is `runs/manifold-discovery-217M/STATUS.md:123` ("user confirmed 217M-only ..."). There is no decisions log.
7. Side note [verified, but only file times]: the audit report and both remediation reports were last written between 17:38 and 21:10 on 2026-05-07, and all `audit_*.py` scripts between 17:55 and 20:34 the same day. File times do not prove duration. But no artefact supports "two-to-three days of agent compute" for the audit.

---

## 7. What the paper can truthfully say about enforcement

Suggested wording, all traceable to files:

> Pipelines are written as markdown specifications that follow a ten-section template. The template is a
> convention, stated in the repository's agent instructions (CLAUDE.md, AGENTS.md); it was **not checked
> by code** in this deployment. A post-hoc linter shows that 7 of the 8 specifications lack a separate
> code-references section (code pointers are inline within Methodology). Specifications are therefore
> detailed prompts plus reference values, not machine-enforced contracts.
>
> Deterministic checks in this deployment were limited to: (i) in one loop driver, a check that each
> executor turn produced a report, a parseable hypothesis JSON and at least one data file, with the loop
> stopping after two consecutive failed turns; and (ii) in one scripted hypothesis batch, a coded
> promote/inconclusive/retire rule, whose outputs were then manually reclassified. Everything else —
> whether a method followed the spec, whether parameters were changed, whether a result is negative,
> when to retire a hypothesis family, and all ten audit verdicts — was agent judgment, with human
> spot-checks. Parameter deviations from the specifications occurred (e.g. 50 instead of 200 Curveball
> permutations) and were documented by the agent in code comments but not detected by any check.

And the paper must drop or fix: the "rejected before execution" sentence; the invented "Step 4" quote;
the "mandatory positive-control slot"; the H115/H118 retirement example (unless the artefacts are found);
"machine-applicable checks" for the audit (say "a checklist applied by an agent").

---

## 8. A deterministic checker that could be built now

Three layers. Each check has a fixed input, a fixed rule, and an exit code. Judgment is kept, but only
where a rule cannot decide, and it is logged.

### Layer A — spec linter (before execution; blocks the run)

Prototype: `contract/spec_lint.py` (reads only; < 1 s for 10 specs).

| Check | Rule | On fail |
|---|---|---|
| A1 Sections | All ten `##` headings present, in order, numbered 1-10. Extra sections only after 10. | Block |
| A2 Non-empty | Each section ≥ N characters (e.g. 200) and not "TBD". | Block |
| A3 Source pin | §2 has a 40-hex hash equal to `repos/<name>/PINNED_COMMIT.txt`, and equal to `git rev-parse HEAD` in that clone; or an explicit `pin: NONE (reason)`. | Block |
| A4 Code refs | Every code pointer is `repos/<name>/path:LINE`; file exists; LINE ≤ file length. | Block (or warn during migration) |
| A5 Parameters | §8 is a table with columns `name`, `default`, `valid_range`, `phase`. Names are unique snake_case. | Block |
| A6 Validation block | §9 contains a fenced machine-readable block (below). Every item has `kind`, `artefact`, `field`, `op`, `value`, `on_fail`. | Block |
| A7 Pitfall links | Each numbered pitfall that has a test names a check id from A6. | Warn |

Proposed §9 format (YAML inside the markdown):

```yaml
validation:
  - id: rewiring_null_sanity
    kind: implementation        # implementation | source_replication | new_model_gate
    artefact: outputs/phase5/rewiring_null_summary.json
    field: n_significant
    op: "=="
    value: 0
    on_fail: block              # block | flag_for_review | info
  - id: er_collapse_source
    kind: source_replication    # only run when target == source model
    artefact: outputs/phase1/effective_rank.json
    field: er_layer_last
    op: within
    value: [1.4, 1.8]
    on_fail: flag_for_review
```

The `kind` field fixes the problem in §5: source-replication checks run only on the source model;
implementation checks (nulls, positive controls, synthetic recovery) run on every model and can block.

### Layer B — run manifest and parameter conformance (at launch and at the end)

| Check | Rule | On fail |
|---|---|---|
| B1 Manifest | Each run writes `run_manifest.json`: spec path + spec file hash, pinned commit, git/venv lock hash, seeds, every parameter value used, phases run. | Block summary |
| B2 Parameter conformance | Each value in the manifest is compared with §8 `valid_range`. Out of range → `deviation` record needing a reason field. | Flag; the reason string must exist (a human approves if the deviation touches a null or a gate) |
| B3 Phase coverage | Phases run vs phases in §6. Missing phases listed in the summary as "not run". | Flag |

This would have caught: Curveball 50 vs 200 (`phase3_residualization.py:54` vs spec line 757), label permutations 8 vs 100–1,000 (`phase10_h139...py:57` vs spec line 514), and attention-grn running only phases 0a,1,2,3,4,12 (`runs/attention-grn-217M/README.md:3-4`).

### Layer C — artefact checks (after the run; before any summary is accepted)

Prototype: `contract/run_check.py` (reads only; ~113 s wall time for all 8 runs).

| Check | Rule | Prototype result on MaxToki |
|---|---|---|
| C1 Validation block | Evaluate each §9 item against its artefact field. Exit non-zero on any `block` failure. | Not possible today: no spec has a machine-readable block |
| C2 Number trace | Each decimal number in FINAL_SUMMARY that is not a quoted spec value must match a value in the run's json/csv/tsv/log/txt artefacts at the same rounding. Markdown is excluded so prose cannot vouch for itself. | Traced: attention 46/47, spectral 65/65, topology 17/17, manifold 144/144, longevity 12/17, sae-atlas 25/25, circuit-tracing 38/50, exhaustive 64/67. 22 numbers had no match (e.g. circuit-tracing `53.55`, `0.586`, `0.324`; longevity `0.34`, `0.36`, `5.65`). Some are cross-run quotes. **Caveat:** 2-decimal numbers can match by chance in large CSVs, so "traced" is weak evidence; "untraced" is a concrete to-do list. |
| C3 Path resolution | Every `*.py` path cited in a summary resolves. | 10 cited paths, 0 missing (most summaries cite no script paths at all) |
| C4 Pin resolution | Spec hash = `PINNED_COMMIT.txt` = git HEAD. | Text matches for all 7 hashed specs; HEAD verifiable for only 1 of them (6 clones have no `.git`) |
| C5 Gate computation | Gates (z ≥ 3, margin > 0.02, trustworthiness ≥ 0.80, etc.) are computed by code from artefacts, not read by an LLM. | Not done today (see the biotensor workflow, lines ~463-478, where an LLM extracts survivors) |

### Layer D — loop control in code

| Check | Rule |
|---|---|
| D1 Retirement | The driver, not the model, retires a family: read each `executor_hypothesis_screen.json`, group by `family` + `lineage`, and retire when the last 2 `decision` values are `negative`/`inconclusive` **and** the linked implementation checks (C1 `kind: implementation`) passed. Write the retirement to `retired.json`. Refuse to run a retired family unless a human override exists. |
| D2 Crash vs negative | Keep the crash guard separate from the retirement rule. A timeout (rc=124) must fail validation, not pass it. |
| D3 Reclassification | Any change to a coded decision (like `MANUAL_REVIEW.md`) must be written to `decisions.jsonl` with who, why, old value, new value. |

### Who decides what (the table the reviewer asked for)

| Requirement | Deterministic (code) | Agent judgment | Human |
|---|---|---|---|
| Spec has ten sections, pins, code refs | A1-A4 | — | Approves waivers (e.g. `pin: NONE`) |
| Parameters within spec ranges | B2 | Writes the reason for a deviation | Approves deviations that change a null or gate |
| Implementation sanity (nulls, positive controls) | C1 `kind: implementation` | — | — |
| Numbers in the summary come from artefacts | C2, C3 | Explains untraced numbers | Spot-checks a random sample |
| Gate pass/fail | C5 | — | Sets gate thresholds before the run |
| Retire a hypothesis family | D1 | Proposes rescue | Approves overrides |
| Is a result scientifically meaningful, is the claim wording right, are confounds handled | Not decidable by rule | Audit patterns P1-P10 | Final sign-off before writing |
| Expensive runs (SAE training, re-extraction) | — | Estimates cost | Go / no-go, logged in `decisions.jsonl` |

**How noncompliance is detected:** exit codes from A, B, C. **How remediation is triggered:** a failed
check writes a `remediation.jsonl` entry that becomes the next executor task; a summary cannot be marked
final while any `block` item is open. **What still needs a human:** thresholds, waivers, deviations to
nulls or gates, overrides of coded decisions, and whether a claim is scientifically correct. Section
checks do not make science correct; they only make the missing pieces visible.

---

## 9. What I did not do

- I did not open the public GitHub repo; I relied on the task's description of its contents.
- I did not check every parameter of every spec against every script. I report two deviations I found by grep. There are likely more.
- The validation-item classifier is a heuristic. I spot-read the item list (`contract/validation_items.csv`) and the counts look right for the research specs; the per-item labels are in that file for anyone to audit.
- The number-trace check (C2) can give false matches for short numbers. I did not try to resolve the 22 untraced numbers beyond two greps (e.g. circuit-tracing's `0.586`/`0.324` come from exhaustive-mapping outputs, i.e. cross-run quotes).
- I did not run any model or GPU job.

## Files produced (scratch only)

- `<AGENT_TMP>/<SESSION_DIR>/d422dc33-6d6f-4b2b-bbf7-f274d8a82fae/scratchpad/understand/contract/spec_lint.py`
- `.../contract/spec_lint_results.json`, `.../contract/spec_lint_summary.csv`, `.../contract/validation_items.csv`
- `<AGENT_TMP>/<SESSION_DIR>/d422dc33-6d6f-4b2b-bbf7-f274d8a82fae/scratchpad/understand/contract/run_check.py`
- `.../contract/run_check_results.json`
- `.../contract/all_scripts.txt` (the 1,633 files searched)

## Plain-words summary

The paper says the spec template is enforced and bad specs are rejected. No code does that. If it had,
7 of the 8 specs would have been rejected. The loop's "retire after two failures" rule is, in code, a
guard against crashes; it stopped the only real loop after network errors. The retirement example in the
paper (H115, H118) is in no data file. Several other statements about the "contract" (the quoted null
step, the positive-control slot, "machine-applicable" audit) do not match the files. The honest claim is:
the specs are detailed prompts with reference numbers; checks were done by the agent and by a human
reading. A real deterministic layer is small and I prototyped two parts of it.
