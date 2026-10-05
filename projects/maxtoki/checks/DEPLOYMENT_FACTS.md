# MaxToki deployment facts (items D11 and D12)

This file is a ledger of checked facts about how the April–May 2026 MaxToki-217M deployment actually ran.
Each fact names the file it comes from. Every number here was computed by a saved script from saved files.

- Machine-readable version: `projects/maxtoki/checks/deployment_facts.json`
- Scripts, inputs, outputs, hashes: `projects/maxtoki/checks/d12_deployment_facts/` (`scripts/s1…s7`, `out/`, `inputs/`, `run_config.json`)
- Paths below are relative to `projects/maxtoki/` unless they start with `pipelines/`.
- All times are local machine time (CEST, UTC+2). Files written after 2026-05-08 06:00 are later revision work and are left out.
- No model was run. Everything was done on the CPU by reading files.

---

## 1. Results first

| Question | Answer (from the files) | Main evidence |
|---|---|---|
| Device | Apple MPS on one laptop for every model forward pass, including the 1B model and SAE training. Some analysis ran on the same laptop's CPU. No log, config or script run used CUDA. | `run_config.json` files (`"device": "mps"`), `runs/sae-atlas-217M/outputs/full_12layer.log` line 2 |
| Dates | First artefact 2026-04-15 23:59:54. Last artefact 2026-05-07 21:10:46. Span 21.9 days. Files were written on 11 of those days. | `out/timeline.json` |
| Compute time, whole deployment | 47.75 h of log windows, with overlaps counted once (53.79 h if each pipeline is summed on its own). A second method gives 49.8–53.6 h. Scripts themselves printed or saved 29.6 h (a lower bound). This is laptop time, not GPU-hours. | `out/timeline.json` |
| Audit | The audit spec, the audit sweep and both repair rounds were all done on 2026-05-07 in one session. Audit request to final report: 3.67 h (4.35 h if the spec writing is included). | `out/audit_timing.json` |
| Human supervision time | Not logged anywhere. What is logged: 73 typed prompts during the deployment (97 counting spec writing and paper drafting). | `out/human_prompts.json` |
| Agent model | Not recorded in any run file. The only record of a model name in the runs is the spectral-geometry autoloop: all 6 of its agent calls used the `sonnet` alias. The "Claude Opus 4.7" claim comes only from an agent-written statement. | `runs/spectral-geometry-217M/autoloop/runtime/driver.log` |
| Model input | Single cells only, in every pipeline. No multi-cell trajectories. No time or query tokens. | `setup/maxtoki_adapter.py:72-75`, 27 scripts, 5 token files |
| Topology-141 | Only the "headline backbone" ran. 21 of 141 numbered hypotheses have code. The strict max-null covered 3 of 12 layer states. The dual-axis CV covered one hypothesis (H123) in one domain (immune). | `out/pipeline_facts.json` → `topology` |
| H115 / H118 | Not in any MaxToki run, summary, audit, setup or pipeline file. Outside the paper drafts, the only other place found (search of the MaxToki `runs/`, `summaries/`, `audits/`, `setup/`, plus `pipelines/`, `repos/`, `references/`, `prompts/`; this revision's own notes left out) is the pinned reference repo of the earlier topology study (`repos/topology-biomechinterp2/iterations/iter_0044`–`0053`), which is not a MaxToki run. *(Corrected in verification; see the end of this file.)* | `out/pipeline_facts.json` → `h115_h118_search`; `verification/out/v4_facts.json` |
| MaxToki checkpoint | Hugging Face `theodoris-lab/MaxToki` revision `21aa7b7f844c146c16fb79ffb3aa75d9b07a3d77`. All 6 local files match it byte for byte. This pin was never written down; it is reconstructed from the download date. The same bytes also sit at earlier and later revisions (e7041eb7 to today's `main`), so the hashes alone do not single out 21aa7b7f. | `out/provenance.json`; `verification/out/v2_hashes.json` |
| Adamson data | It is a knock-down (CRISPRi) dataset. The project calls it CRISPRa, which is wrong. 86 perturbations in the file; 57 were evaluated. | `out/adamson_modality.json` |

---

## 2. Whole deployment

### 2.1 Dates

- First artefact: `setup/token_dictionary.json`, 2026-04-15 23:59:54. First file in a run folder: `runs/attention-grn-217M/scripts/phase0_extract.py`, 2026-04-16 00:12:57.
- Last artefact: `audits/audit-20260507-completion-v2.md`, 2026-05-07 21:10:46.
- Span: 21.88 days.
- Days with any file write (11): Apr 15, 16, 17, 18, 19, 20, 23 and May 3, 4, 5, 7.
- Distinct clock-hours with a file write: 66.
- No files were written from Apr 24 to May 2, or on May 6.

### 2.2 Compute hours, measured three ways

| Method | What it measures | Total, overlaps counted once | Sum over pipelines |
|---|---|---|---|
| A. Log windows | For each `.log` file: from its creation to its last write. Merged per pipeline and across pipelines. | **47.75 h** | 53.79 h |
| B. Activity sessions | Log windows plus every single file write, merged when two events are closer than G minutes. | 49.76 h (G=15), **50.65 h (G=30)**, 53.59 h (G=60) | 56.77 / 58.01 / 61.46 h |
| C. Logged durations | Durations the scripts printed or saved (curated list, each read from a file). | — | **29.61 h** (lower bound) |

What these numbers mean:
- A and B agree within about 3–6 h. So "about 48–51 hours of laptop wall-clock" is a fair summary.
- C is lower because many steps did not print a duration.
- None of these is GPU time. The machine has no discrete GPU. MPS is the Apple-silicon GPU backend in PyTorch, and part of the time was CPU work.
- None of these measures agent thinking time. A job with no log and no file write is invisible.
- Pipelines overlapped for 3.40 h of the 47.75 h (2 at once: 1.46 h; 3 at once: 1.24 h; 4 at once: 0.70 h). The first time 4 ran together was 2026-05-05 16:18 (longevity, manifold, spectral autoloop, topology).
- The paper's Table 3 hours sum to 47.5 h. That total is close to method A. But the per-pipeline numbers differ (section 3).

### 2.3 Hardware

- Every log or config that names a device says `mps`. No run file mentions CUDA, A100 or a remote machine.
- Three scripts have a CUDA branch that never ran on this Mac: `runs/sae-atlas-217M/atlas/scripts/extract_and_enrich_missing_layers.py`, `runs/manifold-discovery-217M/scripts/phase1bc_hidden_states_and_centroids.py`, `runs/longevity-mechinterp-217M/scripts/run_stage3_sae.py`. The shared loader `setup/maxtoki_adapter.py:34-39` picks CUDA, then MPS, then CPU.
- The run files only say "MacBook Pro (Apple Silicon, 32 GB unified memory)" (`README.md`). The machine that holds the files today is a Mac14,9 with an Apple M2 Pro and 32 GB (from `sysctl`). It is probably the same machine, but no file proves it.

---

## 3. Per pipeline

### 3.1 Summary table

Hours are method A (log windows). "First session" = the first block of work on that pipeline; "later" = separate later sessions (extensions and audit repairs). "Work blocks" = method B with a 3-hour gap, which shows day-level blocks. "Logged" = method C.

| Pipeline | File-write dates | Device | Spec phases | Log h (first / later) | Work blocks h | Logged h | Paper Table 3 |
|---|---|---|---|---|---|---|---|
| spectral-geometry | Apr 16–17; May 3, 5, 7 | MPS | 11 | 3.63 (0.80 / 2.82) | 3.74 | 2.60 | 0.5 |
| attention-grn (4 runs) | Apr 16 (docs Apr 17, 23) | MPS, 1B included | 14 | 8.83 (8.83 / 0) | 9.63 | 4.44 | 5.5 |
| topology-141 | May 3; May 5; May 7 | MPS extraction, CPU analysis | 14 | 2.59 (1.26 / 1.33) | 7.05 | 0.88 | 4+ |
| sae-atlas | Apr 17–19; May 3; May 7 | MPS, incl. SAE training | 14 | 10.10 (9.77 / 0.33) | 15.87 | 1.86 | 2.5 |
| circuit-tracing | Apr 19 22:25 – Apr 20; May 7 | MPS (script default) | 14 | 4.17 (4.17 / 0) | 7.40 | 4.02 | 4 |
| exhaustive-mapping | Apr 20; Apr 23 | MPS (script default) | 14 | 7.72 (6.18 / 1.54) | 9.08 | 5.77 | 5.5+ |
| manifold-discovery | May 3–4; May 5; May 7 | MPS forward passes, CPU heads | 14 | 15.21 (11.92 / 3.29) | 20.69 | 9.87 | 24 |
| longevity | May 5 (docs May 7) | MPS (runtime default) | 13 stages | 1.54 (1.54 / 0) | 1.59 | 0.17 | 1.5 |

Notes on the table:
- Work blocks include audit repair work on 2026-05-07 (circuit 3.22 h, sae-atlas 1.59 h, part of topology's 4.81 h block).
- Logged durations: spectral = Phase 0 (907 s) + Phase 8 (2,702 s) + autoloop agent turns (5,746 s); attention-grn = the five Phase-0/0b extractions; topology = two Phase-0 extractions + the audit re-extraction (894 s); sae-atlas = the 12-layer run only (6,681 s); circuit = four source layers (14,473 s); exhaustive = experiment 1 (331.2 min) + experiment-2 rerun (15 min); manifold = 8 "done in" entries in `STATUS.md` (35,520 s); longevity = Stage-1 extraction only (624 s).
- Paper Table 3 values (0.5, 5.5, 4+, 2.5, 4, 5.5+, 24, 1.5) were read from `paper-plos-one/main.tex` lines 930–1013.

### 3.2 Agent model and sessions

| Pipeline | Sessions (history ids) | Human prompts | Agent model recorded in run files |
|---|---|---|---|
| spectral-geometry | 965ec1c8, 8fb71b84 | 8 | Autoloop only: `sonnet` alias, 6 of 6 agent calls (4 executor, 2 brainstormer), 2026-05-05 16:15–17:51. Exact Sonnet version not recorded. Main sessions: not recorded. |
| attention-grn | 965ec1c8 | 4 | not recorded |
| topology-141 | 3af9282c, 2a9284b9 | 5 | not recorded; Phase 11 "autoloop" called no LLM |
| sae-atlas | 965ec1c8, 8fb71b84 | 17 (10 about the web atlas) | not recorded |
| circuit-tracing | 965ec1c8 | 4 | not recorded |
| exhaustive-mapping | 965ec1c8, 8fb71b84 | 7 | not recorded |
| manifold-discovery | 8be46b0f, 5137b56d | 6 | not recorded |
| longevity | b9b1ea22, 58d03bad | 3 | not recorded |

- The only source for "Claude Opus 4.7" is `paper/LLM_USAGE_STATEMENT.txt:15`. The agent wrote it on 2026-05-08 01:19. The PLOS paper repeats it (`paper-plos-one/main.tex:1020`).
- It cannot be checked. The April–May session transcripts are not on disk. `~/.claude/history.jsonl` stores typed prompts only, with no model field.

### 3.3 Phases run, and which were extensions

Phase evidence was found by script: output and log names, and `[Phase N]` markers printed inside logs. "Base" = first session. "Added" = later days of the same session. "Later" = a separate later session.

**spectral-geometry** (spec: Phases 0–10)
- Base, Apr 16 19:05 – Apr 17 10:36: Phases 0, 1, 3, 4, 5 (5b, 5c), 6, 7 (7b), 9 (9a, 9c, 9d, 9e). Phase 2 is the shared co-pole test inside `phases_345.py`.
- Later, May 3: Phase 9b (cross-model vs Geneformer) and Phase 8 (stability over 3 cell samples, not the specified fine-tuning seeds).
- Later, May 5: Phase 10 (autonomous loop), as 4 iterations with Sonnet. It stopped after 2 executor failures, both API connection errors.
- Audit, May 7: bootstrap of the Phase-9b Pearson value.
- The pipeline summary still says "Phase 10 … NOT run" (`summaries/spectral-geometry-217M-FINAL_SUMMARY.md`, section "Phase 2 / 10 — status"). The May 5 loop is not reported there.

**attention-grn** (spec: Phases 0a, 0b, 1–12)
- Base, Apr 16 00:14–02:00 (K562, 217M): Phases 0a, 1, 2, 3, 4, 12.
- Added the same day: Phase 0b (11:14), Phase 6 CSSI (13:59), and re-runs of Phases 0a, 1, 2, 3, 12 for RPE1, Adamson and the 1B model (until 18:46).
- Not run as specified: Phases 5, 7, 8, 9, 10, 11 (`runs/attention-grn-217M/README.md`). The RPE1, Adamson and 1B re-runs cover part of Phase 5 (cross-context) and Phase 9 (multi-model).
- Reduced: Curveball null 50 permutations (spec 200); Phase 4 with 6 conditions (spec 13); the 1B run used 200 control cells (217M runs: 2,000).
- No later compute. Only a summary edit on Apr 23.

**topology-141** (spec: Phases 0–13) — see section 6.

**sae-atlas** (spec: Phases 0–13)
- Base, Apr 17 15:53–23:20, on 3 layers (0, 5, 11): Phases 0, 1, 2, 3, 4, 5, 7, 8, 8t, 10, 11.
- Added in the same session: Phase 6 and Phase 9 (Apr 18); a full 12-layer re-run plus Phases 12 and 13 (Apr 19). The 12-layer run overwrote the Apr 17 Phase 0/1/2/5 outputs, including the layer-5 SAE that Phase 8t had used.
- Later, May 3: web atlas data and deployment (spec `04-atlas-deployment.md`, not one of the eight).
- Audit, May 7: Phase 8t null, power case study, and a targeted GATA1 re-run on MPS.
- The Phase-0 activation files were deleted after training.

**circuit-tracing** (spec: Phases 0–13)
- Base, Apr 19 22:24 – Apr 20 02:27: Phases 0, 1, 5, 6 (Phases 2–4 run inside the tracing loop). Source layers L0/L3/L6/L9, 30 features each, 200 cells.
- Added, Apr 20 11:05–11:13: Phases 7, 9, 10, 11, 12.
- Not run: Phase 8 (four experimental conditions), Phase 13 (cross-model comparison).
- Audit, May 7: edge-density null (17:56) and the sign-corrected Phase-11 re-run (20:19–21:09). The re-run used saved edges; it ran no model.

**exhaustive-mapping** (spec: Phases 0–13 in 3 experiments)
- Base, Apr 20 14:18–20:39: Experiment 1 (Phases 0–5; 1,000 of ~4,928 L5 features), Experiment 2 (Phases 6–8; 4 triplets), Experiment 3 (Phases 9–12, steering).
- Later, Apr 23 19:36–22:19 (new session 8fb71b84): Experiment-2 re-run with new triplets, Experiment-3 re-run (L11 fix), Phase 13 synthesis.

**manifold-discovery** (spec: Phases 0–13)
- Base, May 3 19:41 – May 4 08:31 (12.83 h work block): Phases 0, 1 (1a, 1b), 2, 4–11, and 12 and 13 at "LITE" scope, on the pre-set H65 ordering.
- Later, May 5 16:09–19:11: Phase 3a sweeps #1 and #2 (12 candidate orderings), external and zero-shot checks for H95 and H103.
- Later, May 7 11:46–12:54 and 15:57–19:38: sweep #3 (6 candidates, "Phase 14"), Phases 15–17 (not spec phases), and audit work.
- Not run: full Phase-12 cell-level benchmark; Phase 3b; the fifth quality gate (permutation p) was never computed.
- The paper's "24 hours" does not match one block. The base block is 12.83 h. All four blocks sum to 20.69 h. Log windows sum to 15.21 h.

**longevity** (spec: Stages 0–12)
- Base, May 5 16:16–17:50: Stages 0 and 1. Gate G1 failed.
- Not run: Stages 2–5 (gated on Stage 1), 6–11 (need AIDA data), 12 (not applicable).
- May 7: documents only (README 11:40, FINAL_SUMMARY 18:50). No compute.

### 3.4 Tokens per cell and input format

All pipelines fed the model **one cell per sequence**: `<bos>` + that cell's genes ranked by median-normalised expression + `<eos>` (`setup/maxtoki_adapter.py`, `tokenize_cell`). 27 scripts build their model input this way, and none builds a multi-cell sequence.

MaxToki's trajectory format needs time tokens and query tokens. In `setup/token_dictionary.json` these are 3,000 time tokens plus `<boq>`/`<eoq>`, with ids 20,275–23,276. The HF checkpoints have a 20,275-entry vocabulary, so these ids do not exist there. The adapter keeps only gene ids below 20,275 (`setup/maxtoki_adapter.py:72-75`). No run script uses time or query tokens. The five manifold token files contain no id above 20,239.

| Pipeline | Max sequence (tokens) | Max gene tokens | Recorded mean sequence length (incl. `<bos>`/`<eos>`) |
|---|---|---|---|
| spectral-geometry | 2,048 | 2,046 | 1,783.6 (2,000 TS immune cells) |
| attention-grn | 2,048 | 2,046 | K562 2,039.0; RPE1 2,038.9; Adamson 991.1; 1B K562 2,040.4 |
| topology-141 | 2,048 | 2,046 | not recorded for lung / external lung (1,500 cells each); immune reuses the spectral embeddings (1,783.6) |
| sae-atlas | 2,048 | 2,046 | 2,040.0 (1,019,996 positions / 500 K562 cells); 1,773.9 (TS immune, Phase 9) |
| circuit-tracing | 2,048 | 2,046 | not recorded (200 K562 cells) |
| exhaustive-mapping | 2,048 | 2,046 | not recorded |
| manifold-discovery | 4,096 | 4,094 | panel means 2,190.3 (internal), 2,344.0 (external), 2,261.1 (zero-shot), 2,395.8 (lung control), 2,901.5 (lung non-hema); 7.8%–25.6% of cells hit the 4,096 cap |
| longevity | 1,024 | 1,022 | 998.2 (2,974 cells) |

So a single cell used about 1,000–2,900 tokens on average. It did not "occupy a single position".

---

## 4. Audit (2026-05-07, session 2a9284b9)

| Step | Start (human prompt) | End (file written) | Duration |
|---|---|---|---|
| Audit spec written | 16:49:46 | 16:57:38 `pipelines/audit-recurring-review-issues.md` | 7.9 min |
| Audit sweep (80 verdicts) | 17:30:17 | 17:38:25 `audits/audit-20260507.md` | 8.1 min |
| Repair round 1 | 17:50:00 | 18:09:25 `audits/audit-20260507-completion.md` | 19.4 min (13 repair files, 17:55–18:08) |
| Human asks if all is addressed | 18:47:31 | — | — |
| Repair round 2 | 18:48:59 | 21:10:46 `audits/audit-20260507-completion-v2.md` | 141.8 min |
| **Total** | 17:30:17 (apply request) | 21:10:46 | **3.67 h** (4.35 h from the spec request) |

- All audit files are dated 2026-05-07. No audit file exists on any other day.
- Three repair steps ran MaxToki on MPS: the lung re-extraction with seed 43 (893.5 s), the targeted GATA1 re-run (19:10–19:30), and the cell-level "lite" benchmark (19:25–19:38).
- The same session ran the topology-141 extensions earlier that day (11:38–16:37).
- Human prompts in this session on May 7: 11. Five were about the audit. Short, neutral gists:
  - 11:33 asks the agent to find the parallel sessions of the last days
  - 11:35 picks one of the options the agent offered
  - 11:38 asks to check whether the topology-141 run is complete
  - 11:40 approves the agent's proposed completion work
  - 15:55 asks to run the items the topology run had left out of scope
  - 16:49 asks for a new audit pipeline built from earlier peer-review comments
  - 17:30 asks to apply that audit to all MaxToki results (summaries folder)
  - 17:50 asks to fix all issues the audit found
  - 18:47 asks whether all audit issues are now addressed
  - 18:48 asks to address the remaining issues
  - 23:31 pasted text (content not stored); starts the paper work

---

## 5. Human supervision: what is recorded

- **Total supervision time was not logged.** No stopwatch, time sheet or activity log exists. `~/.claude/history.jsonl` records when a prompt was typed. It does not record reading, checking or thinking time. So "about 15 hours" can be neither confirmed nor refuted.
- Prompts in the 10 MaxToki-related sessions (Apr 15 18:31 – May 8 01:30): **97**.
  - 14 were spec writing before the runs (session 17b19da0, plus the longevity spec prompt).
  - 10 were paper drafting (May 7 23:31 onward).
  - **73 were deployment prompts.** By type: 21 "continue/proceed", 9 run requests, 9 "is anything remaining?", 11 session admin (find a past session, log in), 10 about the web atlas, 5 audit, 3 model setup, 2 documentation, 1 stop decision, 1 scope decision, 1 status check.
- Three other sessions in the same folder in that window were left out: a different model (AIDO.Cell), an unrelated journal correspondence, and a download of another model.

Deployment prompts per day (MaxToki sessions only; first–last prompt time is shown only to place the prompts in the day, and is **not** a measure of supervision time):

| Day | Deployment prompts | First–last prompt | Sessions |
|---|---|---|---|
| Apr 15 | 8 (plus 14 spec-writing) | 23:18–23:57 | 17b19da0, 965ec1c8 |
| Apr 16 | 3 | 10:52–18:57 | 965ec1c8 |
| Apr 17 | 8 | 10:15–21:35 | 965ec1c8 |
| Apr 18 | 1 | 14:40 | 965ec1c8 |
| Apr 19 | 11 | 13:10–22:20 | 965ec1c8 |
| Apr 20 | 4 | 11:03–14:15 | 965ec1c8 |
| Apr 23 | 10 | 19:27–20:45 | 8fb71b84 |
| Apr 24 | 1 | 13:49 | 8fb71b84 |
| May 3 | 7 | 18:55–19:51 | 8fb71b84, 3af9282c, 8be46b0f |
| May 5 | 5 | 16:07–16:10 | 8fb71b84, 3af9282c, 8be46b0f, b9b1ea22 |
| May 7 | 15 (plus 4 paper) | 11:33–18:48 | 2a9284b9, 58d03bad, 5137b56d |
| **Total** | **73** | | |

On May 5 the person sent 5 prompts to four different sessions within 3 minutes (16:07–16:10). Those four sessions then ran in parallel.

---

## 6. Topology-141 (item D11)

- **Scope.** The run README says it executed "the **headline backbone**" (`runs/topology-141-217M/README.md`). The run summary says "The 130+ 'explorer' hypotheses beyond the headline backbone … are therefore not re-screened" (`runs/topology-141-217M/FINAL_SUMMARY.md`, section "Scope").
- **Hypotheses with code.** 23 numbered ids appear in the topology scripts. Two of them (H30, H136) are only named as sources, not tested. So **21 of the 141 numbered hypotheses have code**: H01, H03, H13, H16, H17, H20, H23, H24, H32, H47, H69, H70, H91, H116, H123, H124, H127, H130, H138, H139, H141. The Phase-11 batch added 6 unnumbered ones (H-orc, H-hyp, H-coex-res, H-cross, H-tokfreq, H-pooled).
- **When each phase ran.**
  - Base run, 2026-05-03 19:24–20:41: Phases 0, 1–3 (one script), 5, 6, 7+8, 9, 12, 13, on lung, external lung and immune. The immune domain reused the spectral-geometry Phase-0 embeddings. Lung and external lung were new extractions (1,500 cells each, max length 2,048, MPS, seed 42; 1,182 s and 1,100 s).
  - Later, May 5 16:10–17:08: Phase 14 (Geneformer CCA, the Geneformer half of spec Phase 4), Phase 15, Phase 16. Phases 14–16 are not spec phases.
  - Later, May 7: Phase 10 spot-checks (H23, H139; 11:44), Phase 8 extensions H124–H138 (16:01–16:14), Phase 4 vs scGPT (16:14–16:20), Phase 11 substitute batch (16:24–16:36).
  - Audit repairs, May 7 18:01–20:49: synthetic positive control, GroupKFold for H123, Louvain seed check, lung re-extraction with seed 43.
- **Strict max-null (H141).** Layers **3, 6 and 9 only**, out of 12 layer states (`runs/topology-141-217M/scripts/phase12_strict_max_null.py:174`). 4 metrics × 3 domains × 3 layers = 36 tests. 4 of 36 pass.
- **Dual-axis cross-validation.** Only for **H123**, only in the **immune** domain, at all 12 layers, 5 folds, **76 test pairs per layer**. Dual-axis AUROC ranged 0.448–0.476. It was written at 18:07 on May 7, during audit repair round 1. It was a one-off audit repair for H123, not a step applied to a set of surviving candidates. (The base run did hold out one axis at a time: Phase 1 built TF-disjoint and target-disjoint splits, and Phase 9 used them for H91. See the verification notes.)
- **Phase 11.** One Python script with 6 hard-coded hypotheses and no LLM call. Coded outcome: 0 promote, 5 inconclusive, 1 retire. Its stop rule (3 consecutive retires) never fired. A later manual review changed 4 of the 6 labels (`outputs/phase11_autoloop/MANUAL_REVIEW.md`).

---

## 7. Loop mechanics that existed

| Pipeline | What existed | Stop rule in code | Retirement in code | What happened |
|---|---|---|---|---|
| spectral-geometry | A real two-agent loop: a driver script starts separate `claude --model sonnet --print` calls for executor and brainstormer (`autoloop/run_maxtoki_autoloop.py`) | Stop after **2 consecutive executor failures**, where failure = missing report file, missing or empty hypothesis JSON, or no data file (line 258). Also a STOP file and a max iteration. | None. The driver passes `retired` flags into the next prompt but never acts on them. | 4 iterations on May 5. Stopped after 2 failures, both "Unable to connect to API". 0 of 6 hypotheses marked retired. |
| topology-141 | A scripted batch (`scripts/phase11_autoloop.py`), no LLM | 3 consecutive RETIRE | coded promote / inconclusive / retire rule | Rule never fired; 4 of 6 labels changed by manual review |
| manifold-discovery | Claude Code `/loop` wake-ups reading a hand-kept `STATUS.md` (line 3) | none in code | none (0 of 23 scripts mention "retire") | `iterations/` is empty. `STATUS.md` says "Loop ending" on May 5; a new human prompt restarted work on May 7. |
| the other 5 | no loop | — | — | The human asked "is anything remaining?" and "proceed" (section 5) |

- The only "2 in a row" rule in code is the spectral crash guard. It counts failed turns, not negative results.
- **H115 and H118** do not appear in `runs/`, `summaries/`, `audits/`, `setup/` or `pipelines/` (0 hits). They appear in the paper drafts (`paper-plos-one/main.tex:1616` and older drafts). They also appear in the pinned reference repo `repos/topology-biomechinterp2/iterations/iter_0044`–`0053`, which is the earlier topology study's own autonomous run, not a MaxToki run. There, `iter_0045/brainstormer_hypothesis_roadmap.md` lines 16 and 22 mark H115 `retire_now` and H118 `prioritize_hardening` (not retired). Manifold hypothesis ids in `STATUS.md` and `FINAL_SUMMARY.md` run from H38 to H112.

---

## 8. Checkpoints

| Model | What was used | Revision | Check |
|---|---|---|---|
| MaxToki-217M-HF | `setup/MaxToki-217M-HF/` (downloaded 2026-04-16 00:00 from `resolve/main`) | `21aa7b7f844c146c16fb79ffb3aa75d9b07a3d77` (HF, 2026-04-02) was `main` at download time | `model.safetensors` 867,797,760 B, sha256 `9da1fbf8cc489486d158d4653a14e3f44ac49c75a8e1480566eab0e9f99cf898` = HF LFS id at 21aa7b7f. `config.json` and `generation_config.json` git-blob ids match. |
| MaxToki-1B-HF | `setup/MaxToki-1B-HF/` (2026-04-16 14:01–14:03) | same | `model.safetensors` 4,196,167,128 B, sha256 `96c6c8f9d61732c9bedfc37a9cfd937359e6ce0bfa860526646d101815a242e6` = HF LFS id. Configs match. |
| Geneformer V2-316M | HF cache snapshot `05fcbeb8a27d49e0a7a4349152202ee2c1cbfd28` (HF lastModified 2026-02-06), named in 4 scripts | 05fcbeb8 | `model.safetensors` 1,265,455,076 B, sha256 `965ceccea81953d362081ef3843560a0e4fef88d396c28017881f1e94b1246f3` = HF LFS id at 05fcbeb8. The local cache now points `main` to `04c2b2e8…`, but that snapshot and today's HF `main` serve the same V2-316M weights (same sha256). So "latest" still gives the same weights today; pinning 05fcbeb8 is still wise because `main` can change. *(Corrected in verification.)* |
| scGPT whole-human | Only a derived gene-embedding table: `…/subproject_53_scgpt_gpl_replication/embeddings/scgpt_whole_human_gene_embeddings.pt` (249,944,756 B, sha256 `b41289b862126ccd8db166e2688e08bcbe4149c87a9a177f8409b4723bf33f4b`), used by `runs/topology-141-217M/scripts/phase4_scgpt_cross_model.py` | not on HF | Source checkpoint `…/scGPT_checkpoints/whole-human/best_model.pt`, 205,385,258 B, sha256 `6cb5d451ab5c4b33eb673adbe4fddc61d2389df1b89b7651a9fe2e557572b922`; its `args.json` names `cellxgene_census_human-May23-08-36-2023` |

- The three later MaxToki commits (2026-04-23) only upload config files "for tracking". The weight files are the same at today's `main`.
- The 21aa7b7f pin is reconstructed after the fact from dates and hashes. No run file recorded a revision. The hashes fit more than one revision: the 217M files are identical at 1523f7c2 (2026-03-28), e7041eb7 (2026-03-31), 21aa7b7f and today's `main`; the 1B files at e7041eb7, 21aa7b7f and `main`. 21aa7b7f is picked because it was `main` on the download day. Any of these revisions gives the same weights.
- The scGPT gene-embedding table was made from the whole-human `best_model.pt` by `…/subproject_53_scgpt_gpl_replication/scripts/phase0_extract_embeddings.py` (lines 19 and 196).
- Gene medians and the symbol-to-Ensembl map came from Geneformer's gc104M dictionaries (`gene_median_dictionary_gc104M.pkl`, 1,512,661 B; `gene_name_id_dict_gc104M.pkl`, 1,660,882 B). MaxToki's own medians were not used.

---

## 9. Datasets

| File (size) | Public source and version | Used by |
|---|---|---|
| `replogle_concat.h5ad` (30,010,775,846 B; 643,413 cells × 6,546 genes; HepG2 96,616, Jurkat 184,470, K562 188,590, RPE1 173,737 cells) | **Byte-identical** (sha256 `a7992a61…3d56`) to `replogle_concat.h5ad` in Hugging Face dataset `arcinstitute/State-Replogle-Filtered` (commits dated 2025-12-06). Only the K562 part was used: 1,383 perturbed genes, 10,691 non-targeting cells. | attention-grn, sae-atlas, circuit-tracing, exhaustive-mapping |
| `ReplogleWeissman2022_rpe1.h5ad` (1,236,886,900 B; 247,914 × 8,749) | **Byte-identical** (md5 `cc7f1ec5…80fa`) to the scPerturb Zenodo file (records 7041849, 7278143, 7416068, 10044268, 13350497). 2,393 perturbations + control. Its modality field says only "CRISPR". | attention-grn (RPE1 run) |
| `adamson/perturb_processed_symbols.h5ad` (601,696,516 B; 68,603 × 4,888) | GEARS-processed Adamson 2016 data (inferred from its `uns` keys and `GENE+ctrl` labels). The `_symbols` file name suggests a local variant; byte identity could not be checked. | attention-grn (Adamson run) |
| `tabula_sapiens_immune.h5ad` (19,773,999,714 B; 592,317 × 60,606) | CELLxGENE dataset version `e62b0182-368f-4875-8ac1-f49e3d5eda60`, collection `e5f58829-1a66-40b5-a624-9046778e74f5`, schema 7.0.0 | spectral, topology, sae-atlas, exhaustive, manifold |
| `tabula_sapiens_immune_subset_20000.h5ad` (1,721,702,998 B; 20,000 cells) | local subsample of the same dataset version | sae-atlas, longevity |
| `tabula_sapiens_lung.h5ad` (3,196,847,019 B; 65,847 × 60,606) | CELLxGENE `40f8b1a3-9f76-4ac4-8761-32078555ed4e`, schema 7.0.0 | topology, sae-atlas, manifold |
| `tabula_sapiens_kidney.h5ad` (449,970,582 B; 11,376 cells) | CELLxGENE `65ca6e36-73b0-4c88-b0f3-7b23b48844ad` | one sae-atlas web-atlas script |
| `krasnow_lung_smartsq2.h5ad` (186,674,175 B; 9,409 × 53,514) | CELLxGENE `c88e0403-da93-40f4-99b5-f5fdeb81a82c` (Krasnow Lab Human Lung Cell Atlas, Smart-seq2), collection `5d445965-…` | topology (external lung) |
| TRRUST `trrust_human.tsv` (297,659 B); DoRothEA `dorothea_chipseq_human.tsv` (1,640,587 B); STRING `string_ppi_edges.json` (91,141 B); Reactome / KEGG / GO BP gene-set JSONs; `gene2go_all.pkl` | no version recorded in any file | various (see `deployment_facts.json`) |

Full sha256 values for all 15 data files and 14 model files are in `d12_deployment_facts/run_config.json`.

### 9.1 Adamson: CRISPRi or CRISPRa?

- The project calls it **CRISPRa** (`setup/dataset_loader.py` docstring; run table in `summaries/attention-grn-217M-FINAL_SUMMARY.md`).
- The data say it is a **knock-down**. For each perturbed gene, I compared its mean expression in its own perturbed cells with control cells (24,263 control cells).
  - 85 of 85 measured target genes go down. Fraction 1.00 (exact 95% CI 0.958–1.000).
  - Median log2 fold change −2.93 (95% CI −3.16 to −2.75; bootstrap over perturbed genes, 10,000 resamples, seed 42). 78 of 85 fall by more than 2-fold.
  - Control check: 20 random expressed genes per perturbation, in the same cells, go down only 28.8% of the time (1,700 pairs; median log2 fold change +0.21).
- This matches CRISPR interference, which is the design of the original Adamson et al. (2016) study. The "CRISPRa" label is wrong.
- The wrong label starts in the pipeline spec itself: `pipelines/attention-grn-extraction-and-evaluation.md:20` lists "Adamson et al. 2016 (K562 CRISPRa)". The attention-grn summary then reads a "CRISPRi vs CRISPRa" context effect into it (`summaries/attention-grn-217M-FINAL_SUMMARY.md:100`).
- Size: 86 perturbations in the file (plus control). The attention-grn Adamson run evaluated **57** of them. The other runs evaluated 174 (K562, 217M), 325 (RPE1) and 155 (K562, 1B).

---

## 10. What differs from the claims in `paper-plos-one/main.tex`

| Claim (line) | What the files show |
|---|---|
| "a single A100-80GB" (901–902) | MPS on an Apple-silicon laptop for every run; no CUDA anywhere in the run files |
| SAE training "ran on CPU" (902–903) | SAE training ran on MPS (`full_12layer.log` line 2; `setup/topk_sae.py` default `"mps"`) |
| "50 GPU-hours" (224–225, 661–662, 1698–1699) | 47.75 h of laptop log windows (49.8–53.6 h by a second method); MPS plus CPU; not GPU-hours |
| Table 3 hours (934–1013) | Differ per pipeline: spectral 0.5 vs 3.63 log h; sae-atlas 2.5 vs 10.10; topology 4+ vs 2.59; manifold 24 vs 15.21 (12.83 h base block). Extensions exist but are unmarked for spectral, sae-atlas and manifold. |
| Audit took "two-to-three days" (685) / "roughly three days" (1648) | 3.67 h on one afternoon (4.35 h including the spec) |
| "approximately 15 hours of human supervision" (224, 575, 661, 1697) | Not logged. 73 deployment prompts are logged. No time record exists. |
| Claude Opus 4.7 executed all eight pipelines (1020–1031) | Not recorded in any run file. The spectral autoloop used `sonnet`. |
| Loop used for topology-141 and manifold, with a 2-consecutive-negatives retirement rule (564–568, 635–641) | The only LLM loop ran in spectral-geometry (not mentioned in the paper). Topology used a scripted batch with a 3-in-a-row rule. Manifold used `/loop` plus a hand-kept tracker with no retirement code. The only "2 in a row" rule counts crashes. |
| H115 and H118 "were retired after two negatives each" (1614–1617) | Neither id appears in any MaxToki run file. They come from the earlier topology study's own run (`repos/topology-biomechinterp2`), where H115 was marked `retire_now` and H118 was not retired |
| "a whole cell occupies a single position" (852–853) | Each cell was its own sequence of about 1,000–2,900 gene tokens on average (up to 4,096) |
| Glossary: Adamson is one of the screens where "thousands of genes are silenced" (466–468) | Adamson has 86 perturbations (57 evaluated). It is a knock-down, but the project's own files label it CRISPRa. |
| "141 … hypotheses … each screened at every layer" and CV "for the surviving candidates" (954–957) | Backbone only (21 numbered hypotheses with code); strict max-null at 3 of 12 layer states; dual-axis CV for H123 in immune only |
| Checkpoints used "at the exact revisions pinned in the Dependencies section" (`data-and-code-availability.txt`) | No spec pins a checkpoint. The MaxToki revision is reconstructed (21aa7b7f) and verified by hash. |

---

## 11. How this was checked, and limits

**Two ways where possible.**
- Compute time: log windows (A) and activity sessions (B) agree within 3–6 h. Logged durations (C) are a lower bound. My per-pipeline log-window hours equal the earlier `hardware.md` figures (3.63, 8.83, 2.59, 10.10, 4.17, 7.72, 15.21, 1.54 h).
- Deployment span: file times (Apr 15 23:59 → May 7 21:10) and prompt times (Apr 15 23:18 → May 7 18:48) agree.
- Audit: prompt times and file times give the same story. My total (3.67 h) equals the earlier `audit_history.md` figure (3 h 40 min).
- Prompts: my count for the 9 non-spec sessions is 83, the same as `audit_history.md`.
- Checkpoints: sha256 of the weights and git-blob ids of the configs, at two HF revisions (21aa7b7f and today's `main`).
- Adamson: an exact binomial interval and a bootstrap, plus a random-gene control.

**Intervals.** The only intervals are for the Adamson test. Resampling unit: the perturbed gene. Method: percentile bootstrap, 10,000 resamples, seed 42; and an exact Clopper–Pearson interval for the fraction (the bootstrap is degenerate at 85 of 85). Time figures are counts from files; they have no sampling error. Instead I report how they move when the merging gap changes (15, 30, 60 min).

**What I could not do.**
- I could not identify the agent model for any main session. Transcripts are gone, and `history.jsonl` has no model field.
- I could not measure human time. Only prompt counts exist.
- I could not find the exact Sonnet version behind the `sonnet` alias.
- I could not confirm the Adamson file byte-for-byte against a public copy. I found no version for TRRUST, DoRothEA, STRING, Reactome, KEGG or GO.
- I did not test the Replogle K562 and RPE1 files for knock-down from expression. Only Adamson was tested.
- Mean tokens per cell were not recorded for topology (new domains), circuit-tracing and exhaustive-mapping. I did not recompute them, because that needs the exact sampled cells.
- File times cannot show agent thinking time. Log windows can include idle time. Some jobs (autoloop agent turns, the seed-43 re-extraction) wrote no log of their own.
- The phase labels "base / added / later" use session windows that I set from the prompt times and file times (listed in `deployment_facts.json` → `pipelines.*` and in `out/pipeline_facts.json` → `session_windows`).
- Other agents were writing new `v2_*` files into `runs/` while this ran. Those files are after the cutoff and are excluded; the excluded count changes over time.

---

## Plain-words summary

All the model work ran on one Apple laptop, using its built-in GPU through MPS, plus its CPU. No A100 or other NVIDIA GPU was used. SAE training also ran on MPS. The deployment ran from 15 April to 7 May 2026, with files written on 11 days. Logged jobs took about 48 hours of laptop time; a second way of counting gives about 50 hours. These are not GPU-hours. The audit, including both repair rounds, took under four hours on the afternoon of 7 May, not days. Nobody logged how long the human spent. We only know that 73 prompts were typed during the runs. No run file records which Claude model did the work, except the one real agent loop, which used Sonnet. Every pipeline fed MaxToki one cell at a time, never multi-cell trajectories. Topology-141 ran only a small backbone of its 141 hypotheses. Its strictest test covered 3 layers, and its strongest cross-validation covered one hypothesis in one tissue. H115 and H118 are in no MaxToki file; they come from an earlier, different topology run. The MaxToki weights match Hugging Face revision 21aa7b7f exactly (and the same bytes are still on `main`). The two Replogle files match public copies exactly. The Adamson data are a knock-down screen with 86 perturbations, not an activation screen as the project's files say.

---

## Verification notes (independent re-check, 2026-10-01)

A second agent re-checked this ledger. It used its own code, on the CPU only, with no model run. Scripts, outputs and input hashes: `checks/d12_deployment_facts/verification/` (`v1_timing.py` … `v5_patch_json.py`, `out/`, `inputs/`, `run_config.json`). The additions to `deployment_facts.json` are the keys named `verification_2026_10_01`, `adamson_verification_2026_10_01` and `verification` (written by `v5_patch_json.py`; run it again if `s7_assemble.py` is ever re-run).

**Verdict: OK after fixes.** The main numbers reproduce exactly. Three statements were wrong or too strong and are fixed above.

### What was re-derived, and how

| Fact | Original | Re-check | How the re-check differs |
|---|---|---|---|
| Log windows, overlaps counted once | 47.75 h | 47.75 h | Every second covered by a log window is marked in a set, then counted (no interval sorting or merging) |
| Log windows, summed per pipeline | 53.79 h | 53.79 h | same; all 8 per-pipeline values match (3.63, 8.83, 2.59, 10.10, 4.17, 7.72, 15.21, 1.54 h) |
| Hours with 2 or more pipelines running | 3.40 h; first 4 at once 2026-05-05 16:18 | 3.40 h; 16:18:28 | counted per second |
| Work blocks (3-hour gap) | e.g. manifold 20.69 h, sae-atlas 15.87 h | same, within 0.01 h | time-ordered scan instead of interval merge |
| Span, active days, clock-hours | 21.88 d, 11 days, 66 | 21.883 d, 11, 66 | `os.scandir` walk; the 3-file difference in file count is 3 symlinks the original followed (no effect) |
| Audit steps | 7.9 / 8.1 / 19.4 / 141.8 min; 3.67 h | same; 3.675 h | prompt times read straight from `history.jsonl` |
| Prompts in the 10 MaxToki sessions | 97 (73 deployment) | 97; per-session counts match | I also re-read every prompt and agree with the hand-assigned categories |
| All model and data hashes | as in `run_config.json` | all 10 re-hashed files equal | fresh sha256 (and md5 for the RPE1 file) |
| Public matches | Replogle = HF `arcinstitute/State-Replogle-Filtered`; RPE1 = Zenodo | both confirmed | metadata fetched again from HF and Zenodo record 13350497 |
| Adamson knock-down | 85 of 85 down; median log2FC −2.93 (95% CI −3.16 to −2.75) | 85 of 85 down (exact 95% CI 0.958–1.000); median −2.931 (CI −3.164 to −2.745 with seed 42 and with seed 2026) | added a rank statistic: in all 85 the target is lower in its own cells than in controls (Mann–Whitney AUC < 0.5; median AUC 0.232, 95% CI 0.186–0.281). Among the 82 targets seen in at least 5% of control cells, 82 go down (exact 95% CI 0.956–1.000). The one unmeasured target is TIMM23. Resampling unit: the perturbed gene; percentile bootstrap, 10,000 resamples. |
| Tokens per cell, manifold token files, token dictionary | as in §3.4 | same values | read straight from run configs, logs and `.npz` files |
| Topology strict max-null, dual-axis CV | layers 3/6/9, 36 tests, 4 pass; H123, immune, 76 pairs, AUROC 0.448–0.476 | same | read straight from the CSV and summary JSON |
| Spectral autoloop | 6 calls, all `--model sonnet`; stopped after 2 executor failures | same | read straight from `driver.log` |

### Corrections made in this file and in `deployment_facts.json`

1. **H115 and H118 (sections 1, 7, 10, plain-words summary).** The ledger said they "appear only in paper drafts". That is wrong. A search of `repos/` finds them in 39 files of `repos/topology-biomechinterp2/iterations/iter_0044`–`0053`. That repo is the earlier topology study's own autonomous run, pinned as the reference for the topology pipeline. In `iter_0045/brainstormer_hypothesis_roadmap.md`, H115 is marked `retire_now` and H118 `prioritize_hardening` (not retired). The main point stands: no MaxToki run contains them. But the paper's example most likely comes from that earlier run, and even there H118 was not retired.
2. **Geneformer "latest would not reproduce" (section 8).** Wrong for the weights. Today's HF `main` and the local `04c2b2e8` snapshot serve the same `Geneformer-V2-316M/model.safetensors` (sha256 `965ceccea819…`) as `05fcbeb8`. Pinning `05fcbeb8` is still good practice.
3. **MaxToki revision (sections 1 and 8).** The hashes match 21aa7b7f, but they do not pick it out. The 217M files are identical at 1523f7c2, e7041eb7, 21aa7b7f and today's `main`; the 1B files at e7041eb7, 21aa7b7f and `main`. The date of download is what picks 21aa7b7f. For reproducing the weights this makes no difference.

### Additions (facts that were missing)

- **Layer count.** The 217M checkpoint has 11 transformer layers (`setup/MaxToki-217M-HF/config.json`, `num_hidden_layers` = 11). The "12 layer states" are the embedding output plus these 11 layers. The strict max-null used 3 of the 12.
- **Held-out splits in the topology base run.** Phase 1 built TF-disjoint and target-disjoint splits (3 seeds; `scripts/phase123_splits_nulls.py:118-155`). Phase 9 (H91) used them one axis at a time (`scripts/phase9_stability_selection.py:150`). Its docstring says "dual disjoint", but the code never holds out TFs and targets together. The only test that does is the audit GroupKFold for H123. I reworded the section-6 sentence "No hypothesis had 'survived' to be tested this way", which was a judgement, not a file fact.
- **Spectral autoloop calls.** Of the 6 Sonnet calls, 2 finished normally (iter_0002). One hit the 1,800 s timeout but left valid files (iter_0003 executor). Three failed with API connection errors (iter_0003 brainstormer, iter_0004 and iter_0005 executors). The driver line `CLAUDE_MODEL = ... "sonnet"  # cheaper than opus for an iterative loop` (`run_maxtoki_autoloop.py:46`) hints that the agent that wrote it treated Opus as the default. It names no version, so it does not confirm "Opus 4.7".
- **Where "Opus 4.7" came from.** No typed prompt from 2026-04-15 to 2026-05-08 contains "Opus", "Sonnet", "4.7", "A100" or "GPU-hour". The prompt that asked for the LLM usage statement (2026-05-08 01:18) holds no pasted text. So the model name was written by the agent, as the ledger says. `~/.claude.json` keeps only the most recent session's model use for this folder (a much later session), so it says nothing about April–May.
- **Origin of the CRISPRa label.** It is already in the pipeline spec (`pipelines/attention-grn-extraction-and-evaluation.md:20`). The attention-grn summary builds a "CRISPRi vs CRISPRa" context effect on it (`summaries/attention-grn-217M-FINAL_SUMMARY.md:100`).
- **scGPT provenance.** The embedding table was made from the whole-human `best_model.pt` by `subproject_53_scgpt_gpl_replication/scripts/phase0_extract_embeddings.py` (lines 19 and 196).

### Caveats on method (not errors, but worth knowing)

- Log windows run from a log file's creation to its last write. On this exFAT disk a log that is rewritten keeps its first creation time, so a window can include idle time. Example: `phase0_k562_1b.log` spans 2.62 h, but its `run_config.json` records 5,736 s (1.59 h) for the whole phase. Only 4 of 113 logs print full date-times; in all 4 the printed times match the file times (the autoloop driver log is in UTC, 2 hours behind local time).
- The "logged durations" total (29.6 h) includes 1.6 h of autoloop agent turns. That is LLM agent time, not laptop compute.
- `history.jsonl` grows with every new prompt, so its sha256 in any run_config is a snapshot. The April–May rows do not change.
- Other agents keep adding `v2_*` files to `runs/`. At my run 237 files were after the cutoff (181 in the original run). None of them was created before the cutoff, so no deployment file was changed.

### Not re-checked

- I did not re-derive every per-phase "base / added / later" label in section 3.3. I spot-checked spectral Phase 8, attention-grn Phase 4 (6 conditions) and its 50-permutation Curveball null, and the topology splits.
- I did not re-hash the Tabula Sapiens, Krasnow or reference-database files. I re-read the CELLxGENE dataset and collection ids from each file's `uns` and they match.
- I could not find the agent model for the main sessions either. Transcripts for those sessions are not on disk.

### Plain-words summary of the verification

The ledger holds up. The compute hours, dates, audit times, prompt counts, file hashes and the Adamson knock-down test all came out the same when I redid them with different code. Three things were wrong or too strong, and I fixed them. First, H115 and H118 are not only in the paper: they come from an earlier topology study's own run, where H118 was not even retired. Second, today's Geneformer download still gives the same weights, so "latest would not reproduce" was wrong. Third, the MaxToki weights are the same at several Hugging Face revisions, so the hashes confirm the weights but not the exact revision. I also added a few missing facts: the model has 11 layers (12 layer states), the topology base run held out one axis at a time, and 3 of the 6 Sonnet loop calls failed with network errors.
