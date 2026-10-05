# numbers_A — number-provenance audit of the PLOS ONE paper, lines 1–1043

Paper: `<REPO_ROOT>/projects/maxtoki/paper-plos-one/main.tex`
Scope: title, abstract, introduction, glossary (Table in lines 338–487), background, methods,
Table 1 (audit checklist), Table 3 (pipelines, agent hours), AI declaration.

Path shorthand used below:
- `MT/` = `<REPO_ROOT>/projects/maxtoki/`
- `PL/` = `<REPO_ROOT>/pipelines/`

How I checked: I read every summary in `MT/summaries/`, every audit file in `MT/audits/`, the
run-folder summaries/READMEs, selected run JSON/CSV/logs, the relevant scripts, the 217M
`config.json`, and the pipeline specs. I ran no model code. "Verified" means I read it in a file.
"Inferred" means I reasoned from file contents or file dates.

---

## 1. The most important problems (numbers first)

1. **Hardware is wrong.** The paper says every forward pass ran on "a single A100-80GB" (L901–902)
   and that SAE training ran on CPU (L902–903). Every run record says Apple-silicon MPS on a
   MacBook Pro (32 GB). No run log or report mentions A100 or CUDA. The 12-layer SAE training log
   says `device=mps`.
2. **"50 GPU-hours" is not GPU time.** The eight Table 3 "agent hours" sum to 47.5 h. The Table 3
   caption itself says these hours are "not billed GPU time". The machine had no discrete GPU.
   The abstract (L224–225) and methods (L662) call the same quantity "50 GPU-hours".
3. **"15 hours of human supervision" has no source** in summaries, runs or audits. It appears only
   in earlier paper drafts (`MT/paper/main.tex:54,259`, `MT/paper-biosystems/main.tex:69`).
4. **The quoted pipeline "Step 4" (L605–609) is not in the spec or the code.** Spec: curveball
   n = 200. Code actually run: curveball n = 50 in all four runs. There is no "z ≥ 3.0 AND
   p_BH ≤ 1e-4" rule and no "gene-variance-only logistic regression (Step 3)" anchor.
5. **"2,980 feature triplets" (L988–989) is wrong.** The run tested 4 triplets. 2,980 is the
   number of downstream target features with an effect in each triplet.
6. **Manifold row of Table 3 misdescribes the sweep.** H65 was not one of the 12 sweep candidates.
   H38 LITE also passed all four gates. H95 and H103 were called positive under a 3-gate fallback
   added during the sweep. Only H65 was compressed. The frozen gate file lists five gates
   (including a blocked-permutation p ≤ 0.001 with ≥ 2,000 permutations), not four; the
   permutation gate is never reported.
7. **Topology row overstates scope.** Only a "headline backbone" of the 141 hypotheses was run.
   The run summary says the 130+ explorer hypotheses were not re-screened. The strict max-null
   was run at 3 layers (L3/L6/L9), not every layer.
8. **Off-by-one hook in circuit tracing, exhaustive mapping and steering (verified by reading
   code; not re-run).** In `circuit_trace.py`, `experiment1_exhaustive.py`, `experiment2_rerun.py`
   and `experiment3_rerun.py`, the patched value is built from `hidden_states[s]` (the INPUT of
   block s) and then written over the OUTPUT of `model.layers[s]`. That silently deletes block
   s's own computation for every "feature" intervention at L0/L3/L5/L6/L9. So the 2.1 M and
   4.97 M "feature-to-feature edges", the triplet ratios, and the L0/L6 steering results measure
   "remove block s + add a small SAE delta", not "intervene on one feature". The SAE-atlas
   Phase 6 script uses the correct offset (`hidden_states[PROBE_LAYER + 1]` with a hook on
   `layers[PROBE_LAYER]`), which shows the other scripts use a different, wrong convention.
   Several reported oddities fit this: all top L0 hubs have almost the same out-degree
   (39,670–39,676); all 1,000 Exp-1 features have > 1,000 edges; ablation ratios are identical
   across triplets (std < 0.0003); at L0 all three features push toward maturity whatever their
   sign; α = 2 and α = 5 give the same effect.
9. **The audit timing is wrong.** Paper: the audit runs "in two-to-three days of agent compute"
   (L684–685). The audit spec says it is sized for 1–2 days. The file times show the spec, the
   audit, v1 remediation and v2 remediation were all written on 2026-05-07 between 16:57 and 21:10.
10. **Review sets are misattributed.** The six review sets behind the audit (spec lines 12–18)
    include three papers that are not among the seven "prior manual studies cited above"
    (intelligence-genes, longevity, BCB2026 arbitration). Three cited studies (spectral,
    manifold, exhaustive) contributed no review.
11. **Figures are not "plotted directly from the run artefacts by reproducible code"** (L1040–1041).
    They are TikZ plots with hand-typed coordinates in `MT/paper-biosystems/main.tex`, compiled by
    `MT/paper-plos-one/build/make_figures.sh`, which reads no run file.
12. **"A whole cell occupies a single position in the sequence" (L852–853) is contradicted** by
    the paper's own glossary (one gene = one token) and by every run: mean tokens per single cell
    were 991 (Adamson), 2,038.9 (RPE1), ≈998 (longevity).

---

## 2. Claim-by-claim table

Status key: MATCH / MISMATCH / NOT FOUND / AMBIGUOUS / DIFFERENT ENDPOINT.
"DIFFERENT ENDPOINT" = the number exists in the source, but the source measured something other
than what the paper says it measured (or a different unit/denominator/scheme/hardware).

### 2.1 Abstract (L210–239)

| Paper line | Claim text | Value in paper | Source file:line | Value in source | Status |
|---|---|---|---|---|---|
| 221 | "ten-pattern audit framework" | 10 | `PL/audit-recurring-review-issues.md:131` | "Ten patterns recur across the six reviews" | MATCH |
| 222–223 | "We deploy eight pipelines" | 8 | `MT/audits/audit-20260507.md:17-20` | 8 runs audited | MATCH |
| 224 | "approximately 15 hours of human supervision" | ~15 h | none in `MT/summaries`, `MT/runs`, `MT/audits`; only earlier drafts `MT/paper/main.tex:54,259` | — | NOT FOUND |
| 224–225 | "50 GPU-hours of agent compute" | 50 GPU-h | Table 3 hours (L934–1013) sum to 47.5 h; caption L898–899 says these are "not billed GPU time"; `MT/README.md:68` hardware = MacBook Pro Apple Silicon, MPS | 47.5 h of mixed compute/wall-clock on MPS | DIFFERENT ENDPOINT |
| 225–227 | audit "corrected two headline results that had erred in opposite directions" | 2, opposite | `MT/audits/audit-20260507-completion-v2.md:38-66` (GATA1), `:68-85` (CRISPRi 54.61→53.48); `MT/summaries/circuit-tracing-217M-FINAL_SUMMARY.md:93-94` vs `:106-112` | GATA1: negative→"some specificity". CRISPRi: point estimate went down, but verdict went from "near-chance" to "decisively above 50%". Both verdicts moved toward more regulatory signal. | DIFFERENT ENDPOINT (opposite only for the point estimate) |
| 228–229 | manifold "transfers to disjoint donors" | — | `MT/summaries/manifold-discovery-217M-FINAL_SUMMARY.md:23,37,96` | 13 disjoint donors external; 12 zero-shot | MATCH |
| 230 | "compresses 2,400×" | 2,400× | `MT/runs/manifold-discovery-217M/reports/compaction_chain.csv` rows full_drift (18.213888 MB) and hard_sparse (0.007744 MB); summary `:123` | 2,352× (summary rounds to 2,400×) | MATCH (rounded) |
| 230 | "7.7 KB sparse operator" | 7.7 KB | same CSV, `hard_sparse_16f_60g` | 0.007744 MB | MATCH |
| 230–232 | "Five methodologically distinct probes ... converge on a negative" | 5 | `MT/audits/audit-20260507.md:313-325` (CT1) | 5 probes listed. But post-audit: GATA1 now "encodes some TF→target specificity" (`completion-v2.md:62-66`), CRISPRi "decisively above 50%" (`circuit...SUMMARY.md:106-112`), and circuit/exhaustive rest on the off-by-one hook (§3) | DIFFERENT ENDPOINT (overstated) |
| 235–236 | "Activation steering is causally effective, strongest at the input layer" | L0 strongest | `MT/summaries/exhaustive-mapping-217M-FINAL_SUMMARY.md:92-96,193-199`; `MT/runs/exhaustive-mapping-217M/outputs/experiment3/steering_summary.json` | Δs L0 +0.0962 (largest); L3 +0.0178; L9 +0.0141; L11 +0.0048. By mean abs logit change, L11 (1.07) > L0 (0.42). No random-feature control. Hook overwrites block output (§3). | DIFFERENT ENDPOINT |
| 236 | "counter-differentiation inversion at mid-stack layer L6" | L6 | same summary `:94` | frac_mat 0.00, Δs −0.0079 (3 features, 50 cells) | DIFFERENT ENDPOINT (number matches; measurement confounded by hook, §3) |
| 238–239 | "All pipeline execution and audit were performed by AI agents" | — | `MT/summaries/spectral-geometry-217M-FINAL_SUMMARY.md:353` ("pending explicit user authorization"); `MT/runs/topology-141-217M/outputs/phase11_autoloop/MANUAL_REVIEW.md` (unattributed "manual" reclassification) | not recorded who did what | AMBIGUOUS |

### 2.2 Introduction (L247–336)

| Paper line | Claim text | Value in paper | Source file:line | Value in source | Status |
|---|---|---|---|---|---|
| 296–299 | eight pipelines "developed in our prior ... studies" [7 citations] | 8 pipelines, 7 cites | `PL/longevity-mechinterp-donor-aware.md` §2 | longevity pipeline's source paper is not among the 7 cites | AMBIGUOUS (minor) |
| 302–304 | "recurring reviewer objections it accumulated are exactly what the audit framework encodes" | — | `PL/audit-recurring-review-issues.md:12-18` | 3 of 6 review sets are on papers not in the cited list | MISMATCH |
| 306 | "roughly 15 hours of human supervision" | ~15 h | none | — | NOT FOUND |
| 316–319 | "the audit shifted three load-bearing verdicts, including two ... in opposite directions" | 3 | `MT/audits/audit-20260507-completion.md:57-62` (4 interpretation shifts); `completion-v2.md:38-85,123-147` (F7, F8, F12 more) | ≥ 6 recorded shifts; "three" is a selection | AMBIGUOUS |
| 324–326 | "six interrelated findings ... not directed regulatory mechanism" | 6 | paper Results subsections L1060–1448 (6) ; `completion-v2.md:213-218` | 6 matches; but audit v2 re-scopes the negative to "at least partial encoding for GATA1" | MATCH (count); wording overstated |

### 2.3 Glossary table (L338–487)

| Paper line | Claim text | Value in paper | Source file:line | Value in source | Status |
|---|---|---|---|---|---|
| 363–365 | each gene one token; "20,275-entry vocabulary" | 20,275 | `MT/setup/MaxToki-217M-HF/config.json` (`vocab_size`); `MT/setup/maxtoki_adapter.py:71-75` | 20275. (token_dictionary.json has 23,277 entries; 3,002 ids ≥ 20,275 are outside the HF vocab) | MATCH |
| 367–368 | "L0 is the input embedding and L11 the final layer of ... 11-layer stack" | 11 layers, L0–L11 | config `num_hidden_layers: 11`; `MT/summaries/sae-atlas-217M-FINAL_SUMMARY_12layer.md:4` | 12 states = embedding + 11 blocks | MATCH |
| 368–369 | "Layer 10, head 6" as an example head | L10H6 | `MT/summaries/manifold-discovery-217M-FINAL_SUMMARY.md:45-46,105` | head index uses block numbers 0–10; under the glossary's L0 = embedding convention this block writes state L11 | AMBIGUOUS (two index conventions) |
| 466–469 | Replogle K562, RPE1, Adamson: "thousands of genes are silenced one at a time" | thousands, silenced | `MT/summaries/attention-grn-217M-FINAL_SUMMARY.md:11`; `MT/summaries/attention-grn-217M-adamson-report.md:27` | project labels Adamson "CRISPRa"; Adamson evaluation n = 57 perturbations | MISMATCH (for Adamson) |

### 2.4 Background (L490–542)

| Paper line | Claim text | Value in paper | Source file:line | Value in source | Status |
|---|---|---|---|---|---|
| 523–525 | TopK SAEs "roughly 5,000 features per layer" | ~5,000 | `MT/summaries/sae-atlas-217M-FINAL_SUMMARY_12layer.md:4` | 4,928 per layer | MATCH |
| 525–526 | "causal circuits of roughly 2.1 million feature-to-feature edges" | 2.1 M | `MT/summaries/circuit-tracing-217M-FINAL_SUMMARY.md:5,25` | 2,144,011 edges; but produced by the off-by-one hook (`MT/runs/circuit-tracing-217M/scripts/circuit_trace.py:178,201,208`) | DIFFERENT ENDPOINT (number matches) |

### 2.5 Materials and methods (L545–695)

| Paper line | Claim text | Value in paper | Source file:line | Value in source | Status |
|---|---|---|---|---|---|
| 564–568 | Fig 1: autonomous-loop agent "alternates proposing and testing until a stopping rule fires" for two pipelines | 2 pipelines | `MT/runs/topology-141-217M/README.md:9-12,23`; `.../outputs/phase11_autoloop/autoloop_summary.json` (`n_iterations: 6`); `MT/runs/manifold-discovery-217M/STATUS.md:3` | topology: Codex loop skipped; a 6-hypothesis batch, each tested once, then manually reclassified. manifold: a `/loop` task tracker, not a proposer/tester alternation | DIFFERENT ENDPOINT |
| 574–575 | Fig 1: "roughly 15 hours of attention" | ~15 h | none | — | NOT FOUND |
| 583 | template section "source (citation plus pinned commit hash)" | pinned hash | `PL/manifold-discovery-extraction-compactification.md` §2 | "no public repository ... no reference implementation"; no hash | MISMATCH (1 of 8 specs) |
| 587–588 | "The template is enforced: missing or out-of-order sections are rejected before execution" | enforced | `## ` headings of each spec in `PL/` | 7 of 8 deployed research specs have no top-level "Code references" section and add "Quick-start" as §10 (only `longevity-...md` matches the list); no validator script found | MISMATCH |
| 605–606 | quoted Step 4: "Curveball ... null permutation (1000 iterations, seed=42)" | 1,000 | spec `PL/attention-grn-extraction-and-evaluation.md:79,213,757`; code `MT/runs/attention-grn-217M/scripts/phase3_residualization.py:9-10,54,57`; `MT/runs/attention-grn-217M/outputs/phase12_verdict*.json` (`curveball_null_iters`) | spec: curveball n = 200 (label shuffle n = 1,000). Run: n = 50 in all 4 runs; seed 42 is in the code only | MISMATCH |
| 606–608 | "Anchor the chance baseline against the gene-variance-only logistic regression (Step 3)" | — | spec Phase 1, `PL/attention-grn-...md:146-166` | trivial baselines are univariate AUROCs; no such step | MISMATCH |
| 608–609 | "Reject the null at z ≥ 3.0 AND p_BH ≤ 10^-4" | z ≥ 3, p ≤ 1e-4 | spec `:723` ("z-score > 3"); verdict code `MT/runs/attention-grn-217M/scripts/phase12_verdict.py:43-128` (criteria C1–C4, no z rule); `MT/audits/audit-20260507.md:106` | no p_BH ≤ 1e-4 rule anywhere; "BH-q ≤ 1e-4" is an observed result in the audit; executed verdict has no z test | MISMATCH |
| 612–617 | Curveball "leaves every gene with exactly the number of partners" | — | spec `:213` | spec: preserves each TF's out-degree | MATCH (spec) |
| 635–637 | loop used for "the topology-141 pipeline and the manifold-discovery sweep" | 2 | as row for L564 | topology loop was replaced by a single 6-hypothesis batch | AMBIGUOUS |
| 640–641 | "2-consecutive-negatives retirement rule" adopted | 2 negatives | `PL/topology-geometry-141-hypotheses.md:420,535`; `PL/residual-stream-spectral-geometry.md:418`; `.../phase11_autoloop/MANUAL_REVIEW.md` | rule is in the specs; in the MaxToki topology batch, hypotheses were retired after one test; manifold sweeps had no retirement | DIFFERENT ENDPOINT (spec rule, not executed rule) |
| 642–643 | persistent file-based memory across conversations; multi-day runs | — | `MT/runs/manifold-discovery-217M/STATUS.md:3-21` | STATUS.md loop tracker, 2026-05-03 → 05-07 | MATCH |
| 651 | gating on "multi-hour SAE training runs and full residual-stream re-extractions" | — | `MT/summaries/spectral-geometry-217M-FINAL_SUMMARY.md:353` | "Deferred pending explicit user authorization" | MATCH (partial evidence) |
| 656–657 | supervisor did not write code, debug runs or write summaries | — | none | not recorded | NOT FOUND |
| 661 | "over three weeks of wall-clock" | 3 weeks | file dates: `MT/runs/attention-grn-217M/outputs/phase0.log` (Apr 16 01:09); `MT/audits/audit-20260507-completion-v2.md` (May 7 21:10) | 21 days | MATCH (inferred from file dates) |
| 662 | "~50 GPU-hours of agent compute" | 50 | as abstract row | 47.5 h, not GPU | DIFFERENT ENDPOINT |
| 663–667 | "Two of the three audit-detected failures were first noticed during this human spot-checking" | 2 of 3 | `MT/audits/*.md` | audit docs attribute detection to actions A2/A4/A8 (`completion-v2.md:38-85`); no human spot-check record | NOT FOUND |
| 675–677 | "ten recurring ... objections from six independent peer-reviewer comment sets accumulated by the prior manual studies cited above" | 10, 6 | `PL/audit-recurring-review-issues.md:12-18,40-46` | 10 and 6 match; but the six reviews are of: attention/perturbation, SAE/circuits, topology, intelligence-genes, longevity, BCB2026 arbitration. Three are not cited studies; three cited studies have no review | MISMATCH (attribution) |
| 679–681 | each pattern has signature, prescription, citation to reviews | — | spec, e.g. `:513` ("Cited by") | present | MATCH |
| 684–685 | audit "executes end-to-end on a project folder in two-to-three days of agent compute" | 2–3 days | spec `:25` ("doable in 1–2 days"); file mtimes: spec 16:57, `audit-20260507.md` 17:38, `completion.md` 18:09, `completion-v2.md` 21:10, all 2026-05-07 | same afternoon/evening | MISMATCH |
| 685–686 | "per-pattern 8×10 verdict matrix" | 80 cells | `MT/audits/audit-20260507.md:45-48` | "8 runs × 10 patterns = 80 cells" | MATCH |
| 686–687 | "plus an action-proposal table" | — | `audit-20260507.md:381-392` | A1–A8 | MATCH |
| 692–695 | "the checklist caught errors its authors had missed — including two that pointed in opposite directions" | 2 | `MT/audits/audit_a2_groupkfold_status.md:62-91`; `circuit...SUMMARY.md:96-99` | CRISPRi sign bug was found during the A2 GroupKFold re-run, after the first audit had judged a re-run unnecessary ("structural argument suffices") | AMBIGUOUS |

### 2.6 Table 1 — audit checklist (L697–829)

| Paper line | Claim text | Value in paper | Source file:line | Value in source | Status |
|---|---|---|---|---|---|
| 708–711 | patterns "distilled from six independent peer-review comment sets on the earlier manual studies" [7 cites] | 6 | `PL/audit-recurring-review-issues.md:12-18` | see L675 row | MISMATCH (attribution) |
| 739–826 | pattern names P1–P10 | 10 | spec §6 | same ten patterns | MATCH |
| 808 | P8 repair: "Non-parametric bootstrap (1,000 iterations) computed at the unit of inference" | 1,000 | spec `:506` ("bootstrap (1000 resamples ...)") ; executed: `MT/audits/audit_a3_bootstrap_cis.py:29` (N_BOOT = 10,000), `MT/summaries/manifold-discovery-217M-FINAL_SUMMARY.md:72` (80% subsample without replacement × 200), `completion-v2.md:89-90` (80% subsample × 1,000), `MT/audits/bootstrap_cis_summary.json` (Wilson CI for CRISPRi) | spec says 1,000; executed resampling used 10,000, 200, 1,000, subsampling without replacement, and Wilson CIs | MATCH (to spec); executed scheme differs |

### 2.7 Deployment target (L832–885)

| Paper line | Claim text | Value in paper | Source file:line | Value in source | Status |
|---|---|---|---|---|---|
| 839 | "most recent major SCFM release (March 2026)" | March 2026 | `MT/README.md:8` | bioRxiv 10.64898/2026.03.30.715396 | MATCH |
| 840 | "from the Theodoris/NVIDIA group" | — | `MT/README.md:9-11` | Theodoris lab; code repo NVIDIA-Digital-Bio | MATCH |
| 840–842 | "~1T gene tokens across ~175M ... transcriptomes ... birth through 90+ years" | 1T, 175M, 90+ | `MT/README.md:14` | same (secondary; not checked against the MaxToki PDF) | MATCH |
| 848–851 | autoregressive Llama, two-stage training | — | config `architectures: LlamaForCausalLM`; `MT/README.md:14` | same | MATCH |
| 852–853 | "a whole cell occupies a single position in the sequence" | 1 position per cell | `MT/summaries/attention-grn-217M-adamson-report.md:12`, `...-rpe1-report.md:12`; `MT/runs/longevity-mechinterp-217M/README.md:75`; glossary L363–364 | mean tokens per single cell 991.08, 2,038.9, ≈998; one gene = one token (`MT/README.md:60` has the same wrong framing) | MISMATCH |
| 855 | Geneformer convention "≤ 2,048 genes per cell" | 2,048 | `MT/README.md:60` | same | MATCH |
| 862 | "shares its 20,275-token Ensembl-ID gene tokenizer exactly with Geneformer V2-316M" | exact | `MT/summaries/spectral-geometry-217M-FINAL_SUMMARY.md:247`; `MT/summaries/topology-141-217M-FINAL_SUMMARY.md:17` | "identical"; checked on 1,500/1,500 HVG token ids only | MATCH (check covered 1,500 tokens) |
| 863–865 | MaxToki–Geneformer is "within-family"; scGPT exposes a "cross-architectural-family" qualifier | — | `spectral...SUMMARY.md:247` (Geneformer = "BERT-style encoder"); `MT/README.md:131` and `audit-20260507.md:66-72` call Geneformer "autoregressive" | Geneformer is encoder-only (paper L845 says so too). MaxToki and Geneformer differ in architecture and align; the shared factor is the tokenizer | MISMATCH (label) |
| 869–871 | 11 layers, hidden 1232, 8 heads/layer, vocab 20,275, context 4,096 | 11/1232/8/20275/4096 | `MT/setup/MaxToki-217M-HF/config.json` | 11 / 1232 / 8 / 20275 / 4096 | MATCH |
| 880–881 | glue "lives under projects/maxtoki/setup/" | — | local `MT/setup/` | exists locally; per task context, public repo 89da276 has no `setup/` or `runs/` (not verified by me) | MATCH (local only) |
| 882–885 | 7 of 8 give results; ageing probe "returned no result clearing its pre-registered gate" | 7/8 | `MT/runs/longevity-mechinterp-217M/FINAL_SUMMARY.md:8` | G1 FAIL, Stages 2–5 not run | MATCH |

### 2.8 Table 3 — pipelines, agent hours (L887–1016)

| Paper line | Claim text | Value in paper | Source file:line | Value in source | Status |
|---|---|---|---|---|---|
| 893–895 | typewriter ids are the published spec identifiers, "so each row can be traced to a runnable file" | ids | `PL/` file names | real names: `residual-stream-spectral-geometry.md`, `attention-grn-extraction-and-evaluation.md`, `topology-geometry-141-hypotheses.md`, `sparse-autoencoders/01-…/02-causal-circuit-tracing.md/03-exhaustive-mapping-and-steering.md`, etc. `sae-circuit-tracing` and `sae-exhaustive-mapping` match no file or run folder | MISMATCH (minor) |
| 895–897 | "Agent hours is unattended end-to-end wall-clock time" | wall-clock | `MT/summaries/attention-grn-217M-FINAL_SUMMARY.md:113-125` ("compute"), `circuit...SUMMARY.md:33` ("Compute time"), `MT/summaries/sae-atlas-217M-FINAL_SUMMARY.md:272` ("Total compute"), manifold `:7` ("wall time") | most figures are summed phase runtimes, not agent wall-clock | DIFFERENT ENDPOINT |
| 899–901 | "+" marks pipelines with extension phases; figure is a lower bound | only topology, exhaustive marked | spectral Phase 8 (+~73 min, `spectral...SUMMARY.md:227`); SAE 12-layer run (+6,681 s = 1.86 h, `MT/runs/sae-atlas-217M/outputs/full_12layer.log`); manifold sweeps 2026-05-05 and 05-07 (`STATUS.md:5-21`) | these extensions are unmarked | MISMATCH |
| 901–902 | "All runs used a single A100-80GB graphics processor for model forward passes" | A100-80GB | `MT/README.md:68`; `attention-grn-217M-FINAL_SUMMARY.md:127`; `device: mps` in `MT/summaries/attention-grn-217M-k562-report.md:16`, `-rpe1-report.md:15`, `-adamson-report.md:15`, `attention-grn-1B-k562-report.md:15`, `spectral-geometry-217M-report.md:10`; `exhaustive...SUMMARY.md:165` ("M2 Max ... same hardware class"); no A100/CUDA in any run log | MacBook Pro Apple Silicon, 32 GB, MPS | MISMATCH |
| 902–903 | "sparse-autoencoder training and statistical analysis ran on CPU" | CPU | `MT/runs/sae-atlas-217M/outputs/full_12layer.log:2` (`device=mps`, includes SAE training); `MT/runs/sae-atlas-217M/scripts/phase1_train_saes.py:16` (default mps) | SAE training on MPS; stats on CPU plausible (manifold `:161` "~100s on CPU") | MISMATCH (SAE part) |
| 929–930 | spectral: SVD "at each of the 12 layers" | 12 | `sae-atlas-217M-FINAL_SUMMARY_12layer.md:4`; paper L870 | 12 hidden-state taps = embedding + 11 layers | AMBIGUOUS (paper says 11 layers elsewhere) |
| 930–933 | spectral: effective dims, subcellular location, PPI, cell-sample change | — | `spectral...SUMMARY.md:44-48,67,79-81,225-243` | ER compression 6.1×; SV1 GO CC 10/96; SV3 STRING z +3.12; CKA over 3 cell samples | MATCH |
| 934 | spectral agent hours | 0.5 | `spectral...SUMMARY.md:181` | "~30 min" (first pass only; Phase 8 added ~73 min) | MATCH (base); undercount |
| 937 | attention-grn "(4 runs)" | 4 | `attention-grn-217M-FINAL_SUMMARY.md:7-12` | 4 runs | MATCH |
| 941–943 | degree-preserving null + one-feature expression-variance baseline | — | `attention-grn-217M-FINAL_SUMMARY.md:27-36,107` | curveball z −0.14/+0.17/+0.16/−0.04 (n = 50); gene variance beats attention by 6–20 AUROC points | MATCH |
| 943–946 | K562, RPE1, Adamson, 1B replication | — | same `:9-12` | 1B run used 200 cells vs 2,000 | MATCH |
| 947 | attention-grn hours | 5.5 | same `:125` | "~5.5 h compute" (phase sum 338 min = 5.6 h) | MATCH (compute time) |
| 954–955 | "141 pre-registered topological hypotheses ..., each screened at every layer against a hierarchy of null models" | 141, every layer | `MT/runs/topology-141-217M/FINAL_SUMMARY.md:33-50`; `MT/summaries/topology-141-217M-FINAL_SUMMARY.md:88` | "headline backbone" only; "130+ explorer hypotheses ... not re-screened"; strict max-null at 3 layers × 4 metrics × 3 domains = 36 tests | MISMATCH |
| 955–957 | "for the surviving candidates, under cross-validation that holds out regulators and targets simultaneously" | — | `MT/audits/audit_a2_groupkfold_status.md:15-40` | dual-axis CV applied only to H123 immune (not a survivor; AUROC 0.505→0.462) | DIFFERENT ENDPOINT |
| 958 | topology hours | 4+ | no total in any topology doc; `MT/runs/topology-141-217M/logs/` mtimes: backbone 2026-05-03 ~19:30–20:40, extensions May 5 and May 7 | no stated figure | NOT FOUND |
| 965–966 | SAE: "Approximately 5,000 sparse features ... per layer across 12 layers" | ~5,000 × 12 | `sae-atlas-217M-FINAL_SUMMARY_12layer.md:4,23` | 4,928 × 12 taps | MATCH |
| 966–967 | annotated "by the genes and cell types that switch it on" | — | `MT/README.md:104`; `sae-atlas-217M-FINAL_SUMMARY.md:215-217` | GO/KEGG/Reactome/STRING/TRRUST; 95.9% enriched for ≥1 of 32 TS immune cell types | MATCH |
| 967–968 | tested for TF-target specificity | — | `sae-atlas-217M-FINAL_SUMMARY.md:105-110` | Phase 8t 0/48 | MATCH |
| 969 | SAE hours | 2.5 | `sae-atlas-217M-FINAL_SUMMARY.md:272` | ~2.5 h base; 12-layer run adds 1.86 h | MATCH (base); undercount |
| 976 | circuit tracing "2.1 million feature-to-feature edges" | 2.1 M | `circuit...SUMMARY.md:5,25` | 2,144,011 | MATCH (number) |
| 976–977 | edges "built by intervening on one feature and measuring the effect on others" | — | `MT/runs/circuit-tracing-217M/scripts/circuit_trace.py:178,201,208` | patch = `hidden_states[src]` + delta written to OUTPUT of `layers[src]`, i.e. block `src` is also deleted | DIFFERENT ENDPOINT (code reading) |
| 978–980 | CRISPRi check "whether the predicted direction of change matches the observed direction" | — | `circuit...SUMMARY.md:93-104` | original metric ignored predicted sign (54.61%); only the audit re-run does this (53.48%) | MATCH (post-audit re-run only) |
| 981 | circuit hours | 4 | `circuit...SUMMARY.md:33,39-42` | ~4 h (71+63+56+51 min) | MATCH |
| 988 | "4.97 million edges mapped exhaustively" | 4.97 M | `exhaustive...SUMMARY.md:6,12,15,130` | 4,970,096; 1,000 of ~4,928 L5 features, 3 downstream layers; same hook issue (`experiment1_exhaustive.py:145,159,164`) | DIFFERENT ENDPOINT ("exhaustively" overstated; hook) |
| 988–989 | "combinatorial ablation of 2,980 feature triplets" | 2,980 triplets | `MT/runs/exhaustive-mapping-217M/outputs/experiment2/combinatorial_summary.json` and `experiment2_v2/...json`; `exhaustive...SUMMARY.md:39,48,131` | `n_triplets: 4`; 2,980 = `n_targets_with_effect` per triplet | MISMATCH |
| 989–991 | steering: "feature activity is added at a chosen layer" and shift measured | — | `MT/runs/exhaustive-mapping-217M/scripts/experiment3_rerun.py:147-155,256,263-273` | z·α scaling, patched value from `hidden_states[li]` written to OUTPUT of `layers[li]` (block li deleted for li < 11) | DIFFERENT ENDPOINT |
| 992 | exhaustive hours | 5.5+ | `exhaustive...SUMMARY.md:19,165` | 5.5 h + 40 min + 4 min | MATCH |
| 997–998 | "does it survive on donors the model was never shown?" | — | `manifold...SUMMARY.md:37` | donors held out from the LET head; nothing shows they were absent from MaxToki pre-training | DIFFERENT ENDPOINT (overstated; inference) |
| 999 | "A sweep over 12 candidate biological orderings" | 12 | `manifold...SUMMARY.md:233-235`; run-folder `MT/runs/manifold-discovery-217M/FINAL_SUMMARY.md:269-288` | 12 in sweeps #1+#2; sweep #3 adds 6 (18 total) | MATCH (for the two-sweep version) |
| 999–1000 | "each scored on four pre-registered quality gates" | 4 | `MT/runs/manifold-discovery-217M/reports/quality_gates_spec.json` (gates; "must pass ALL five gates"); `reports/quality_gates_let_anchor.json`; summary `:235` | frozen spec has 5 gates incl. blocked-permutation p ≤ 0.001, ≥ 2,000 perms, never reported; sweeps used a 3-gate fallback added 2026-05-05 | MISMATCH |
| 1000–1001 | "and then compressed to the smallest operator that preserves the ordering" | each | summary `:269` | "External validation, zero-shot transfer, compaction, factor ablation are NOT done for the sweep manifolds" | MISMATCH |
| 1001–1002 | "The hematopoietic ordering (H65) is the one that passed all four" | H65 only | summary `:34,218-225,244,254,261` | H65 was the pre-set primary branch, not a sweep candidate; H38 LITE also passed all four; H95/H103 "POSITIVE (3-gate)" | MISMATCH |
| 1003 | manifold hours | 24 | summary `:7` | "~24 hours wall time" (05-03→05-04); later sweeps not counted, no "+" | MATCH (base); undercount |
| 1009–1010 | ageing probe: donors as unit; permutation null | — | `MT/runs/longevity-mechinterp-217M/FINAL_SUMMARY.md:16-18`; `audit-20260507.md:159` | donor-held-out 5-fold; n = 100 permutations over donors | MATCH |
| 1011–1012 | no result clearing its gate | — | same `:8,38` | G1 FAIL (0.2755 vs null 0.3320) | MATCH |
| 1013 | ageing hours | 1.5 | same `:7` | ~92 min | MATCH |

### 2.9 AI declaration (L1019–1041)

| Paper line | Claim text | Value in paper | Source file:line | Value in source | Status |
|---|---|---|---|---|---|
| 1020 | agent was "Claude Opus 4.7" | Opus 4.7 | none in runs/summaries/audits; only `MT/paper/LLM_USAGE_STATEMENT.txt` | runs mention "Claude" but no model version | NOT FOUND |
| 1030–1032 | all eight pipelines executed end-to-end by the same agent | — | none | not recorded | NOT FOUND |
| 1038–1039 | no other-provider models for load-bearing work | — | `MT/runs/topology-141-217M/FINAL_SUMMARY.md:45-50` | Codex loop skipped; consistent | MATCH (consistent, not provable) |
| 1039–1041 | "AI tools were not used to create or alter any figure ...; all figures are plotted directly from the run artefacts by reproducible code" | — | `MT/paper-plos-one/build/make_figures.sh:12-28`; `MT/paper-biosystems/main.tex:1204-1205` (e.g.) and 20 `\addplot ... coordinates` blocks; paper L1034 (manuscript drafted by the agent) | figures are TikZ with hand-typed numbers inside the agent-drafted LaTeX; no script reads run files | MISMATCH |

---

## 3. The off-by-one hook, in detail (verified by reading code, not by running it)

Transformers `hidden_states` convention (checked in the project venv,
`MT/.venv/lib/python3.12/site-packages/transformers/utils/output_capturing.py:108-113,255-262`,
transformers 5.5.4 per `MT/requirements.txt:66`): `hidden_states[0]` = embeddings = input to
block 0; `hidden_states[k]` = output of block k−1; the last entry is the final-normed state.
Older 4.x versions use the same convention (input to each block, appended before the block runs).

The four scripts below take `h = hidden_states[s]` (input of block s), add an SAE delta, and then
use a forward hook on `model.layers[s]` to REPLACE that block's output with `h + delta`:

- `MT/runs/circuit-tracing-217M/scripts/circuit_trace.py:178` (clean states), `:201` (patch), `:208` (hook on `layers[src_l]`), source layers 0/3/6/9 (`:41`).
- `MT/runs/exhaustive-mapping-217M/scripts/experiment1_exhaustive.py:145,159,164` (source layer 5).
- `MT/runs/exhaustive-mapping-217M/scripts/experiment2_rerun.py:208,230` (triplet layers 0/5/9).
- `MT/runs/exhaustive-mapping-217M/scripts/experiment3_rerun.py:256,263,273`; `hook_layer_for` at `:147-155` only fixes li = 11.

Effect: block s's attention and MLP output is thrown away in every "feature" intervention at
s ∈ {0, 3, 5, 6, 9}. The measured "edges", "redundancy ratios" and "steering" are therefore
dominated by removing a whole transformer block. The SAE-atlas Phase 6 script does it correctly
(`MT/runs/sae-atlas-217M/scripts/phase6_patching.py:127,141`: `hidden_states[PROBE_LAYER + 1]`
with a hook on `layers[PROBE_LAYER]`).

Reported patterns that fit block removal: L0 hubs 39,670–39,676 edges each
(`circuit...SUMMARY.md:52-56`); 100% of 1,000 features have > 1,000 edges
(`exhaustive...SUMMARY.md:18`); ratios identical across triplets (std < 0.0003,
`exhaustive...SUMMARY.md:177-185`); at L0 all three steered features push toward maturity
whatever their correlation sign (`:64-66,92`); α = 2 and α = 5 give the same result (`:191-199`).
This is an inference about cause; the code fact itself is verified.

Extra problem in the combinatorial ablation (`experiment2_rerun.py:204-230`): when two or three
layers are ablated together, each hook writes a patch built from the CLEAN state of that layer.
So a later hook (e.g. at L5) overwrites everything an earlier ablation (e.g. at L0) changed. The
joint effect of A+B can then never exceed the effect of the later layer alone. This alone could
produce the "deep redundancy" (pairwise ratio 0.165, three-way 0.190) and "zero synergy" results,
independent of the model. Inference from the code; not re-run.

Paper claims in lines 1–1043 affected: abstract L235–236 (steering), background L525–526
(2.1 M edges), Table 3 L976–977, L988–991.

---

## 4. Wording that overstates the evidence (lines 1–1043)

- L227–230 "MaxToki-217M encodes recoverable, externally generalizable cell-identity geometry":
  the anchor-level benchmark found the extracted head does not beat PCA-10 / SVD-10 on raw
  expression (`manifold...SUMMARY.md:172-178`), and the 192-cell benchmark found no feature above
  chance (`:195-203`). The audit itself asked for "extraction-quality scope, not
  benchmark-superiority scope" (`audit-20260507.md:412-419`). "External" = held-out Tabula Sapiens
  donors from the same atlas.
- L230–232 "Five ... probes converge on a negative": after the audit, 2 of the 5 were re-scoped
  toward a small positive, and 2 rest on the hook bug.
- L235 "Activation steering is causally effective": no random-feature control; direction-
  independent at L0; saturates by α = 2; confounded by block removal.
- L225–227, L316–319, L692–695 "erred in opposite directions ... hard to explain as a single
  systematic bias": true for the CRISPRi point estimate only; both verdict changes moved the
  same way (toward some regulatory signal).
- L324–326 "not directed regulatory mechanism": the audit v2 says the claim is "now scoped to
  ... TRRUST-narrow ground truth ... ChIP-seq + strict threshold finds at least partial encoding"
  (`completion-v2.md:213-218`).
- L587–588 "The template is enforced": no enforcement exists and the deployed specs do not follow
  the stated section list.
- L988 "mapped exhaustively" / L985 "Taken to saturation": 1,000 of ~4,928 features at one source
  layer, 3 downstream layers.
- L997–998 "donors the model was never shown": only never shown to the probe head.
- L954–955 "141 ... hypotheses ... each screened at every layer": backbone only.
- L1040–1041 "plotted directly from the run artefacts by reproducible code": hand-typed TikZ data.

---

## 5. What I did not check

- I did not re-run any analysis or model (not allowed). The hook bug is from code reading.
- I did not check the public GitHub repo at 89da276; statements about it come from the task context.
- I did not check MaxToki facts (1T tokens, 175M cells, dates) against the MaxToki PDF, only
  against `MT/README.md`.
- I did not check figure coordinates (Figs 2–9) against run files; their captions fall outside
  lines 1–1043 except Fig 1.
- I did not check external-literature claims (AI Scientist, MLE-bench, etc.).

## 6. Plain-words summary

Most model numbers are right: 11 layers, 1232 hidden size, 8 heads, 20,275 vocabulary, 4,096
context, 4,928 SAE features, 2.14 M and 4.97 M edges, 2,400× and 7.7 KB, the 8×10 audit matrix.
The problems are in how the work is described. The hardware is wrong (MPS laptop, not an A100).
"50 GPU-hours" is really 47.5 hours of mixed runtime. "15 hours of supervision" has no record.
The quoted spec step was never in the spec and the run used 50 curveball permutations, not 1,000.
"2,980 triplets" is really 4 triplets. The manifold and topology rows claim more than was run.
The audit took one afternoon, not 2–3 days. The figures hold hand-typed numbers. Most serious:
the circuit-tracing, exhaustive-mapping and steering scripts overwrite the wrong block output,
so their "feature" effects mostly measure deleting a whole transformer block.
