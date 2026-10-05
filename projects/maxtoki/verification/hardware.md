# Hardware and compute: what the MaxToki paper says vs what the run files show

KEY = hardware. Reviewer 2: "The manuscript specifies A100 execution, while public run reports list Apple MPS execution."

All times below are local machine time (CEST, UTC+2). I checked this: the autoloop driver log writes UTC
(`2026-05-05T14:15:34+00:00`), and the matching local file times are 16:15.

## 1. Verdict (short)

- **The reviewer is right. No MaxToki run used an A100 or any NVIDIA GPU.** Every model forward pass in every
  pipeline ran on **Apple MPS** (the Apple-silicon GPU backend in PyTorch) on one laptop. That laptop is recorded as
  "MacBook Pro (Apple Silicon, 32 GB unified memory)". Some analysis steps ran on the same laptop's CPU.
- **No remote GPU was used.** There is no RunPod, CUDA, ssh, scp, rsync, SLURM or cloud trace in any run file.
  RunPod/A100 use in this repo starts in **August 2026**, in a different project (biotensor). That is three months
  after the MaxToki runs (April 15 to May 7, 2026).
- **"SAE training ran on CPU" is also wrong.** SAE training defaulted to MPS. The 12-layer SAE log says `device=mps`.
- **"~50 GPU-hours" is roughly the right number of hours but the wrong unit.** It is the sum of the per-pipeline
  runtimes in Table 3 (47.5 h). Those runtimes were measured on the laptop (MPS plus CPU). The log files agree:
  47.75 h when overlapping runs are counted once. So "about 50 hours of compute on one Apple-silicon laptop" is
  supported. "GPU-hours" and "A100" are not.
- **"Three weeks of wall-clock" is supported.** The first run file is from 2026-04-15 23:59 and the last is from
  2026-05-07 21:10. That is 21.9 days. Files were written on only 11 of those days.
- **"Two-to-three days of agent compute" for the audit is not supported.** All six audit files were written on
  **2026-05-07 between 17:38 and 21:10**. The audit catalogue itself was saved at 16:57 that day. The audit report
  says "No re-runs were performed." The file evidence fits a few hours in one day, not 48–72 h.
- **"15 hours of human supervision" cannot be checked.** No run file records human time. The session transcripts
  from April–May 2026 are no longer on disk. The paper itself says no stopwatch was kept.
- **Where "A100" came from:** it first appears in `paper/main.tex` (ICML draft, saved 2026-05-08 01:33) and in
  `paper/LLM_USAGE_STATEMENT.txt` (2026-05-08 01:19). It is in **no** run file, summary or audit. Likely source
  (my inference, not proven): the reviewer prompt `prompts/reviewer-2-research-quality.md:92` contains the sentence
  "Apple Silicon M2 Max is different from NVIDIA A100". The drafting agent read such prompts. It then copied every
  later version forward.

## 2. What the PLOS ONE paper claims (verified by reading)

File: `<REPO_ROOT>/projects/maxtoki/paper-plos-one/main.tex`

| Line(s) | Claim |
|---|---|
| 224–225 | "approximately 15 hours of human supervision driving 50 GPU-hours of agent compute" (abstract) |
| 574–575 | human supervisor "roughly 15 hours of attention in total" (Fig. 1 caption) |
| 660–663 | "15 hours of supervision over three weeks of wall-clock against ~50 GPU-hours of agent compute" |
| 684–685 | audit "executes end-to-end on a project folder in two-to-three days of agent compute" |
| 895–903 | Table 3 caption: "Agent hours" is "unattended end-to-end wall-clock time ... **not** billed GPU time". Then: "All runs used a single A100-80GB graphics processor for model forward passes; sparse-autoencoder training and statistical analysis ran on CPU." |
| 934–1013 | Table 3 agent hours: spectral 0.5, attention-grn 5.5, topology 4+, sae-atlas 2.5, circuit 4, exhaustive 5.5+, manifold 24, longevity 1.5 (sum 47.5) |
| 1611 | manifold sweep "spanned 24 hours of agent compute" |
| 1648–1650 | "the audit consumed roughly three days of agent compute, set against the ~50 hours — about two days — of agent compute the eight pipelines themselves required" |
| 1697–1699 | "Approximately 15 hours of high-level human supervision ... directed ~50 GPU-hours of agent compute" |

Also `<REPO_ROOT>/projects/maxtoki/paper-plos-one/data-and-code-availability.txt:1`:
"the hardware used for each run (a single NVIDIA A100-80GB graphics processor for model forward passes; CPU for
sparse-autoencoder training and statistical analysis) are recorded in the per-run log files under
projects/maxtoki/runs/ in the code repository". Three problems: (a) the logs say MPS, not A100; (b) the logs say
SAE training was on MPS; (c) `runs/` is **not** in the public repository (commit 89da276).

Internal contradiction: the Table 3 caption says agent hours are "not billed GPU time", but the abstract and
conclusion call their sum "50 GPU-hours".

## 3. Device actually used, per pipeline (verified by reading scripts and logs)

Base path: `<REPO_ROOT>/projects/maxtoki/`

Every model-running script picks the device the same way:
`DEVICE = os.environ.get("...", "mps" if torch.backends.mps.is_available() else "cpu")`.
Only three files even mention CUDA, and only as a fallback branch that never ran on this Mac:
`setup/maxtoki_adapter.py:35-38`, `runs/longevity-mechinterp-217M/scripts/run_stage3_sae.py:78-81`,
`runs/sae-atlas-217M/atlas/scripts/extract_and_enrich_missing_layers.py:480-485`.

| Pipeline | Device in scripts | Device in logs / reports (evidence) |
|---|---|---|
| spectral-geometry | MPS: `runs/spectral-geometry-217M/scripts/phase0_extract.py:38`, `phase8_stability.py:44` | `outputs/phase0.log:2` "device=mps"; `outputs/phase8.log:2` "device=mps"; `outputs/run_report.md:10` "device: `mps`"; `outputs/phase0/run_config.json:9`; `outputs/FINAL_SUMMARY.md:172` "float32 on MPS"; autoloop `iterations/iter_0003/h3g_run.log:3` "Device: mps" |
| attention-grn (4 runs incl. 1B) | MPS: `runs/attention-grn-217M/scripts/phase0_extract.py:51`, `phase0_any.py:41`, `phase0_rpe1.py:39`, `phase0b_value_weighted.py:55`, `phase4_causal_ablation.py:46`, `phase6_cssi.py:48` | `README.md:12` "Device: MPS (Apple Silicon), float32, eager attention"; `outputs/FINAL_SUMMARY.md:127` "Hardware: MacBook Pro (Apple Silicon, 32 GB unified memory), MPS backend, float32"; `outputs/phase0.log:3`, `phase0b.log:2,8`, `phase4.log:29`, `phase0_rpe1.log:3,22`, `phase0_adamson.log:2,20`, `phase6_cssi.log:2,10` all "mps". **The 1B model also ran on MPS:** `outputs/phase0_k562_1b.log:20` "Loading MaxToki on mps (model_dir=.../MaxToki-1B-HF)"; `outputs/phase0_k562_1b/run_config.json:8` "device": "mps" |
| topology-141 | MPS: `runs/topology-141-217M/scripts/phase0_extract.py:74` | `logs/phase0_lung_extlung.log:4` "loading MaxToki-217M on mps"; `outputs/phase0/phase0_summary.json:10,22`; `outputs/phase0/lung/run_config.json:9`; `outputs/seed_stability_phase0/seed43/phase0_summary.json:10`; `FINAL_SUMMARY.md:81,93`. Immune embeddings were reused from spectral-geometry (MPS). |
| sae-atlas | MPS for extraction **and SAE training**: `runs/sae-atlas-217M/scripts/phase1_train_saes.py:16` (`SAE_DEVICE` default mps); `full_12layer_pipeline.py:43,176`; `setup/topk_sae.py:70` default `device="mps"`; also `phase0_extract_positions.py:38`, `phase6_patching.py:37`, `phase9_multitissue.py:43`, `remaining_phases.py:52`, `audit_a8_rerun_phase8t_targeted.py:49` | `outputs/phase0.log:8` "Loading MaxToki-217M on mps"; `outputs/full_12layer.log:2` "FULL 12-LAYER SAE PIPELINE N_ctrl=500 device=mps" (this run trains all 12 SAEs); `outputs/phase6.log:6` "device: mps"; `outputs/phase8t_positive_tf_rerun/run.log:1` "device=mps"; `.../summary.json:3` |
| circuit-tracing | MPS: `runs/circuit-tracing-217M/scripts/circuit_trace.py:37` | log does not print the device; script has MPS cache calls at lines 226-228, 270-272 |
| exhaustive-mapping | MPS: `runs/exhaustive-mapping-217M/scripts/experiment1_exhaustive.py:34`, `experiments_2_3.py:40`, `experiment2_rerun.py:43`, `experiment3_rerun.py:42` | logs do not print the device. Summary `summaries/exhaustive-mapping-217M-FINAL_SUMMARY.md:165` says it ran "on same hardware class" as the source paper's "M2 Max" (the M2 Max figure is the **source paper's** Geneformer budget, not this run) |
| manifold-discovery | MPS for forward passes: `runs/manifold-discovery-217M/scripts/phase1bc_hidden_states_and_centroids.py:43`; **CPU** for LET heads: `phase5_let_anchor.py:44` `DEVICE = "cpu"  # ... CPU avoids MPS contention` | `outputs/phase1b_internal.log:2,7`, `phase1b_external_v3.log:2,8`, `phase1b_zeroshot_v3.log:2,8`, `phase1b_lung_control_v3.log:2,7`, `phase1b_lung_nonhema.log:2,7` all "mps"; `artifacts/anchors/pos_pool_meta.json:11,23,35,47,59`. `STATUS.md:107` phase 9 "88 single-head LET-10D, CPU"; `STATUS.md:102` phase 10 "~30-45 min on CPU"; `STATUS.md:115,118` "rate 0.73 s/cell on MPS" |
| longevity-mechinterp | MPS: `runs/longevity-mechinterp-217M/scripts/maxtoki_runtime.py:92` comment "MPS does not benefit from padding-batching" | `README.md:67` "8k stays well within MPS memory"; `README.md:75` "per-cell forward 0.21s on MPS"; `outputs/stage1_run.log` "mean_fwd=0.21s" |

Other laptop signs (verified):
- `runs/manifold-discovery-217M/STATUS.md:99` "memory pressure: only 56MB free, system load 5+".
- `STATUS.md:100` first attempt "died silently ... (likely macOS App Nap / sleep). Re-launched under `caffeinate -i`".
- Logs show `<CONDA_ROOT>/...` Python paths (e.g. `runs/topology-141-217M/logs/phase12_lung.log:35`).
- `projects/maxtoki/README.md:68` "Hardware used across runs: MacBook Pro (Apple Silicon, 32 GB unified memory), MPS backend, float32".
- The public repo (checked with `gh api` at commit 89da276) has
  `projects/maxtoki/summaries/attention-grn-217M-FINAL_SUMMARY.md:127` "Hardware: Apple Silicon laptop (32 GB unified memory), MPS backend, float32",
  `.../attention-grn-217M-k562-report.md:16` "device: `mps`", and `.../spectral-geometry-217M-FINAL_SUMMARY.md:176` "float32 on MPS".
  The same public repo's `projects/maxtoki/paper/main.tex:670` says "a single A100-80GB". So the public repo contradicts itself. This is what the reviewer saw.

The exact chip is **not** recorded in any run file. The machine that holds the files now is a MacBook Pro
Mac14,9, Apple M2 Pro, 32 GB (from `system_profiler`). This matches "MacBook Pro, 32 GB" in the summaries. I infer,
but cannot prove, that it is the same machine.

Remote-GPU search (verified): `grep -i` for `a100|runpod|nvidia|cuda:0|H100|vast.ai|ssh|scp|rsync|slurm|sbatch|colab|gcloud`
over `runs/`, `summaries/`, `audits/`, `setup/` finds nothing except the three CUDA fallback branches above and
false hits on the word "draws". `projects/biotensor/RUNPOD.md:4` says "Written 2026-08-14 after running the whole
scGPT/Geneformer evaluation on one A100" (file created 2026-08-14). That is the first A100 use I can find, and it is not MaxToki.

## 4. Timing evidence per pipeline

Three kinds of evidence:
- **Logged duration**: a time the script printed (tqdm bars, "done in Ns", "elapsed").
- **Log window**: from the log file's creation time to its last write, merged across logs so overlaps count once.
- **File window**: first to last file write in the run folder, per day.

Scripts I wrote to get these (in scratchpad): `hw/timeline.py`, `hw/perday.py`, `hw/logtimes.py`, `hw/union.py`;
raw per-log output `hw/logtimes.jsonl`.

| Pipeline | Paper "agent hours" | Source of paper number (summary) | Logged durations | Log window (h) | File-write windows (local time) |
|---|---|---|---|---|---|
| spectral-geometry | 0.5 | `summaries/spectral-geometry-217M-FINAL_SUMMARY.md:181` "~30 min total" | phase0 forward pass 13.1 min (tqdm); phase8 stability 45.0 min (tqdm, 05-03); autoloop executors 925 s + 1800 s (timeout) + 913 s + 884 s, brainstormers 447 s + 777 s (`autoloop/runtime/driver.log`) | 3.63 | 04-16 19:05–19:44; 04-17 10:28–10:36; 05-03 19:16–20:29; 05-05 16:13–17:51; 05-07 18:53–18:54 |
| attention-grn (×4) | 5.5 | `summaries/attention-grn-217M-FINAL_SUMMARY.md:113-125` "~5.5 h compute" | forward passes: K562 54.6, VW 17.7, ablation 22.4, RPE1 70.6, Adamson 27.3, CSSI 36.5, **1B K562 94.6 min (N=200 cells)** = 323.7 min = 5.4 h | 8.83 | all on 04-16, 00:12–18:46 |
| topology-141 | 4+ | **no source found.** No summary or run file gives "4 h" | phase 0 lung + external lung `wall_seconds` 1182.4 + 1100.0 = 38 min (`runs/topology-141-217M/FINAL_SUMMARY.md:83,95`); seed-43 re-extraction "~15 min wall clock" (`FINAL_SUMMARY.md:289`) | 2.59 | 05-03 19:24–20:41; 05-05 16:52–17:08 (logs from 16:10); 05-07 11:44–20:50 |
| sae-atlas | 2.5 | `summaries/sae-atlas-217M-FINAL_SUMMARY.md:272` "~2.5h" (3-layer pass only) | 3-layer: phase0 extraction 4.1 min, SAE training ~97 s/layer; **12-layer run: sum of "Lx DONE in" = 6,681 s = 1.86 h, device=mps** (not in the 2.5 h) | 10.10 | 04-17 15:53–23:20; 04-18 14:42–16:00; 04-19 13:13–22:14; 05-03 18:59–19:02 (web atlas data); 05-07 17:55–19:30 |
| circuit-tracing | 4 | `summaries/circuit-tracing-217M-FINAL_SUMMARY.md:33` "~4 hours" | L0 4270 s + L3 3752 s + L6 3380 s + L9 3071 s = 14,473 s = **4.02 h** (`outputs/circuit_trace.log`) | 4.17 | 04-19 22:25; 04-20 02:26–11:13; 05-07 17:56–21:09 (audit add-ons) |
| exhaustive-mapping | 5.5+ | `summaries/exhaustive-mapping-217M-FINAL_SUMMARY.md:19,165` "5.5 hours", "~5.5 hr exp1 + ~40 min exp2 + ~4 min exp3" | exp1 "331.2min elapsed" = **5.52 h** (`outputs/experiment1.log`); exp2_3 log 20:01–20:39; exp2 rerun "wall-clock: 15.0 min" | 7.72 | 04-20 14:19–20:39; 04-23 19:36–22:19 |
| manifold-discovery | 24 | `summaries/manifold-discovery-217M-FINAL_SUMMARY.md:7` "Run period: 2026-05-03 to 2026-05-04 (~24 hours wall time)" | from `STATUS.md`: 1B-internal 10,985 s; 1B-zeroshot v3 4,445 s; 1B-lung_control v3 1,574 s; 1B-external 12,789 s (3.5 h); phase 9 (CPU) 4,672 s; phase 1A 900 s | 15.21 | 05-03 19:41 → 05-04 08:31 (**12.8 h**, not 24 h); 05-05 16:09–19:11 (3.0 h); 05-07 11:46–19:38 (7.9 h). The three sessions together are ~23.7 h. So "~24 h" only fits if the 05-05 and 05-07 extension sessions are counted, but the summary says "05-03 to 05-04". |
| longevity-mechinterp | 1.5 | `runs/longevity-mechinterp-217M/FINAL_SUMMARY.md:7` "~92 min for Stage 1" | extraction of 2,974 cells "elapsed=624s" at 0.21 s/cell (`outputs/stage1_run.log`) | 1.54 | 05-05 16:16–17:50; 05-07 11:40–18:50 (docs only) |
| **Total** | **47.5** | | | **53.8** summed per pipeline; **47.75** when overlaps across pipelines count once | |

What this means:
- Five of the eight "agent hours" (spectral, attention-grn, circuit, exhaustive, longevity) match the logged
  first-pass compute times. They are **laptop MPS/CPU runtimes**. They are not "unattended end-to-end wall-clock"
  of the agent, as the Table 3 caption says. They leave out re-runs and extensions (e.g. the 1.86 h 12-layer SAE run,
  spectral phase 8 and the autoloop).
- "topology 4+" has no source in any file. The logged evidence is 2.6 h of log windows.
- "manifold 24" is stated in the manifold summary, but the base run (05-03 to 05-04) spans 12.8 h of file writes.
- Several pipelines ran **at the same time** on the one laptop (e.g. 05-03 evening: spectral phase 8, topology,
  manifold and the SAE web atlas all wrote files between 18:59 and 20:41). So per-pipeline hours overlap.

## 5. Whole-deployment timeline (file-modification times; verified)

| Date | What was written |
|---|---|
| 2026-04-15 19:04–23:40 | pipeline specs in `pipelines/` (e.g. attention-grn 19:04, spectral 19:24, topology 20:58, manifold 21:52, longevity 23:40); `setup/token_dictionary.json` 23:59 |
| 2026-04-16 | setup adapter + HF checkpoints (00:00–19:01); attention-grn all 4 runs (00:12–18:46); spectral-geometry first pass (19:05–19:44) |
| 2026-04-17 | spectral remaining (10:24–10:36); first summaries (10:16); sae-atlas 3-layer (15:53–23:20); `setup/topk_sae.py` 15:54 |
| 2026-04-18 | sae-atlas phases 6 and 9 (14:42–16:00) |
| 2026-04-19 | sae-atlas 12-layer + cross-layer (13:13–22:14); circuit-tracing start (22:24) |
| 2026-04-20 | circuit-tracing (to 02:27; remaining 11:05–11:13); exhaustive-mapping exp1–3 (14:18–20:39) |
| 2026-04-23 | exhaustive-mapping reruns (19:36–22:19); attention-grn summary update (20:31) |
| 2026-04-24 → 05-02 | **no files written** (9-day gap) |
| 2026-05-03 | sae-atlas web atlas data (18:59–19:02); spectral phase 8/9b (19:16–20:29); topology base run (19:24–20:41); manifold start (19:41) |
| 2026-05-04 | manifold base run continues to 08:31 |
| 2026-05-05 | manifold sweep (16:09–19:11); longevity (16:16–17:50); spectral autoloop with **Claude Sonnet** (16:15–17:51); topology phases 14–16 (16:10–17:08) |
| 2026-05-06 | **no files written** |
| 2026-05-07 | extensions from 11:40 (manifold 14–17, topology 10/11/8ext/scGPT); audit catalogue `pipelines/audit-recurring-review-issues.md` 16:57; audit report 17:38; completion 18:09; completion-v2 21:10; audit re-runs 17:55–21:09 |
| 2026-05-07 23:40 → 05-08 01:34 | first paper draft `paper/` (README 23:40, LLM statement 01:19, main.tex 01:33) — A100 and 50 GPU-hours appear here |

- First run artefact: 2026-04-15 23:59. Last run artefact: 2026-05-07 21:10. Span: 21.9 days.
- Active days (any file write in runs/summaries/audits/setup): 11. Distinct clock-hours with any write: 67.
- Caveat: `runs/sae-atlas-217M/atlas/.gitignore` shows mtime 2026-04-15 19:45 but was created 2026-05-03; the web
  atlas files were copied with old modification times. I used creation time to rule it out as the first artefact.

## 6. Where each claim first appeared (verified by grep over all paper folders)

| Claim | First appearance | Traceable to a run file? |
|---|---|---|
| "A100-80GB" | `paper/main.tex:883` (saved 2026-05-08 01:33), then copied to `paper-deanon/main.tex:883`, `paper-jbi/main.tex:1133`, `paper-cbac/main.tex:1097`, `paper-biosystems/main.tex:785,1861`, `paper-plos-one/main.tex:901` and `data-and-code-availability.txt:1` | **No.** Zero hits in runs/, summaries/, audits/, setup/. Only other hit in the repo before August: `prompts/reviewer-2-research-quality.md:92` (created 2026-04-15) "Apple Silicon M2 Max is different from NVIDIA A100." Likely source (inference). |
| "CPU for SAE training" | `paper/main.tex:883-884`; `paper/LLM_USAGE_STATEMENT.txt:49-50` "auxiliary CPU work (SAE training, statistical resampling)" | **No.** Scripts and `full_12layer.log:2` show MPS. |
| "~50 GPU-hours" | `paper/main.tex:259-260, 866`; `paper/LLM_USAGE_STATEMENT.txt:47-49` "approximately 50 GPU-hours of MaxToki forward passes" | The **number** matches the sum of the Table 3 hours (47.5) and the merged log windows (47.75 h). The word **GPU** (as in datacenter GPU) is not supported. |
| Table 3 agent hours | `paper/main.tex:359-370` (column "Hr.", caption "Wall clock is agent-side end-to-end") | Yes for 7 of 8: taken from per-pipeline summaries (see §4). "topology 4+" has no source. |
| "15 hours of human supervision" | `paper/main.tex:54, 259`; `paper/LLM_USAGE_STATEMENT.txt:47-48` | **No record anywhere.** The paper admits no stopwatch was kept (`paper-plos-one/main.tex:657`). April–May transcripts are not in `~/.claude/projects/` (only September files remain). |
| "three weeks of wall-clock" | `paper/main.tex:259` | **Yes.** 2026-04-15 → 2026-05-07 = 21.9 days. |
| audit "1–2 days" / "three days" / "6% overhead" | `paper/main.tex:272` says "1--2 days of agent time"; `paper/main.tex:827-828` says "approximately three days ... roughly 6% audit overhead" (3 days vs 50 h is ~144%, not 6%, so the first draft contradicts itself). Later versions say "two-to-three days" and "roughly three days". | **No.** All audit files are from 2026-05-07 17:38–21:10; catalogue saved 16:57 the same day; report says "No re-runs were performed" (`audits/audit-20260507.md`, header). |
| "manifold sweep spanned 24 hours" | from `summaries/manifold-discovery-217M-FINAL_SUMMARY.md:7` | Partly. Base run file window is 12.8 h. Three sessions together ~23.7 h. |

## 7. Side findings (not asked, but related)

1. **Model used by the agent.** The paper's AI declaration (`paper-plos-one/main.tex`, "Declaration of generative AI")
   (line 1019 onward) says Claude Opus 4.7 ran all eight pipelines. But the spectral-geometry autoloop ran **Claude Sonnet**:
   `runs/spectral-geometry-217M/autoloop/runtime/driver.log:1` "autoloop start ... model=sonnet", and each spawn line
   "claude --model sonnet --print ... --max-budget-usd 5".
2. **Autoloop stop rule.** The same log ends "aborting: 2 consecutive executor failures" (`driver.log`, last line).
   Two executor runs failed to write their report files. The paper describes the rule as retiring a hypothesis after
   "2 consecutive negatives". Here the rule fired on crashes, not on negative results.
3. **Label bug in the 1B log.** `runs/attention-grn-217M/outputs/phase0_k562_1b.log:2` header says "MaxToki-217M",
   but line 20 loads `MaxToki-1B-HF`. Only the header is wrong.
4. **Data statement.** `paper-plos-one/data-and-code-availability.txt:1` says per-run logs are "under
   projects/maxtoki/runs/ in the code repository". The public repo (commit 89da276) has no `runs/`.
5. **Laptop speed.** The 1B model took 94.6 min for 200 cells (~28 s per cell) on MPS
   (`phase0_k562_1b.log`). This speed fits a laptop, not an A100.

## 8. What the paper can truthfully say

Suggested wording (every part below is backed by a file):

> All model forward passes ran on a single Apple-silicon laptop (MacBook Pro, 32 GB unified memory) using the PyTorch
> MPS backend in float32. Sparse-autoencoder training also ran on MPS. Some analyses (for example the per-head
> probes in manifold discovery) ran on the same machine's CPU. No cloud or datacenter GPU was used.
> The per-pipeline runtimes in Table 3 are compute times recorded in each run's logs and summaries. They cover the
> base run of each pipeline, not later extensions or re-runs. Together they sum to about 48 hours of laptop compute.
> Several pipelines ran at the same time on this machine. The deployment spanned 22 days (15 April – 7 May 2026),
> with run files written on 11 of those days. The audit and its two remediation passes were carried out in one day
> (7 May 2026).

What to drop or change:
- Drop "A100-80GB" everywhere (main text, table caption, data statement, and the public `paper/main.tex`).
- Replace "50 GPU-hours" with "about 50 hours of compute on one Apple-silicon laptop".
- Replace "SAE training ... ran on CPU" with "SAE training ran on MPS".
- Replace "two-to-three days" / "roughly three days" for the audit with what the files show (one day; the audit
  report and both remediation passes were written between 16:57 and 21:10 on 7 May 2026). Drop the claim that the
  audit cost more than the analysis.
- Change the Table 3 caption: the hours are logged compute times, not "unattended end-to-end wall-clock".
- Fix or drop "topology 4+" (no source; logs show ~2.6 h). Say "manifold ~13 h base run, ~24 h with two later
  extension sessions".
- Keep "15 hours of human supervision" only as the author's own estimate. Say it was not logged.
- Add that the spectral-geometry autoloop used Claude Sonnet, if the AI declaration is kept.
- Either publish `runs/` (or at least the `*.log`, `run_config.json` and `run_report.md` files) or remove the claim
  that run logs are in the public repository.

## 9. Method notes and limits

- The disk is exFAT (`mount`: `/dev/disk4s2 on <LOCAL_VOLUME> (exfat ...)`). It keeps creation and
  modification times. For most files, creation time equals modification time, so the files were written in one go
  (or rewritten). Folder times show 2026-08-12 15:52, probably from a later reorganisation; file times were kept.
- File times show when files were written, not when the agent was thinking. Agent time without file writes is
  not visible. Log windows can include idle time.
- I did not read any file over 1.5 GB. I ran no model or GPU job.
- The public repo was checked with read-only `gh api` calls at `ref=89da276`.

Report path: <AGENT_TMP>/<SESSION_DIR>/d422dc33-6d6f-4b2b-bbf7-f274d8a82fae/scratchpad/understand/hardware.md
