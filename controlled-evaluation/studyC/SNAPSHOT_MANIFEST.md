# Study C pre-audit snapshot: manifest

This manifest is for the study team. It is kept outside the snapshot folder so blind auditors do not see it.

## 1. Numbers first

- Snapshot folder: `<EVAL_ROOT>/studyC/snapshot/maxtoki`
- 1,862 files, 221.7 MB (limit about 300 MB). Built 2026-10-01T01:27:30. Cutoff: files last changed before 2026-05-07T16:57 local time (the audit checklist was written at 16:57 on 2026-05-07).
- Copied as they were: 1,806 files. Rebuilt with audit text removed: 6 files. Listed but not copied: 343 files (10030.9 MB). Left out because they changed at or after the cutoff: 269 files.
- Leak scan: 1,708 text files. No hit for audit files, audit actions, paper folders, the evaluation workspace, the corrected CRISPRi numbers, repair outputs, re-run names, hook-bug wording, sign-bug wording or A100. All strong-term hits are pre-audit wording (section 7).
- macOS wrote 2,065 '._*' sidecar files (exFAT stores the com.apple.provenance attribute that way); the build deleted them.

## 2. What is in the snapshot

Counts below are the 1,806 files copied unchanged. The snapshot also holds the 6 rebuilt summaries (section 3) and 50 `_large_files_listing.tsv` files, which gives the total of 1,862.

| area | files | size |
|---|---:|---:|
| runs/sae-atlas-217M | 138 | 73.5 MB |
| runs/attention-grn-217M | 117 | 63.7 MB |
| runs/manifold-discovery-217M | 227 | 44.6 MB |
| runs/exhaustive-mapping-217M | 1032 | 20.8 MB |
| runs/topology-141-217M | 121 | 10.0 MB |
| runs/spectral-geometry-217M | 113 | 2.9 MB |
| reference_data | 6 | 2.5 MB |
| summaries | 13 | 2.0 MB |
| pipelines | 9 | 0.6 MB |
| setup | 8 | 0.6 MB |
| runs/longevity-mechinterp-217M | 13 | 0.4 MB |
| runs/circuit-tracing-217M | 9 | 0.1 MB |

- `runs/<8 runs>/`: scripts, logs, configs, run-level READMEs, STATUS, EXTENDED_FINDINGS, MANUAL_REVIEW, and outputs under 5 MB, plus nine 5-20 MB files chosen because they hold per-pair or per-anchor data behind headline numbers (attention-grn pair datasets, manifold internal and zero-shot centroids, steering state signatures, one copy of the SAE gene-name list).
- `summaries/`: the pre-audit files as they were, plus 4 rebuilt files (section 3).
- `pipelines/`: the 8 deployed specs and `sparse-autoencoders/README.md` (the SAE composition guide). All are dated 2026-04-15 and were not changed later. Left out: `audit-recurring-review-issues.md` (the audit checklist) and `sparse-autoencoders/04-atlas-deployment.md` (web atlas, not one of the 8).
- **Additions beyond the task list** (remove if not wanted): `setup/` (the adapter, dataset loader and TopK-SAE code that every run imports, the token dictionary, and the two model config.json files; no weights) and `reference_data/biomechinterp/...` (6 small public reference files that the run scripts read from absolute paths: TRRUST, STRING pairs, GO BP / Reactome / KEGG gene sets, the Geneformer symbol-to-Ensembl dictionary; the folder mirrors the original path under `<DATA_ROOT>/`). DoRothEA files were not added, because only the post-audit repair used them.
- `_large_files_listing.tsv` in each folder that had large or duplicate files: name, bytes, mtime, shape or columns/rows, and why it was not copied.

## 3. Rebuilt files (audit-added text removed)

File times of rebuilt files are set to 2026-05-07 16:56 (the nominal snapshot time). Their real history is below.

- `runs/topology-141-217M/FINAL_SUMMARY.md` (source last changed 2026-05-07T20:50:15). Method: strip every '### ...' section whose heading says 'added 2026-05-07' (6 audit-marked + 4 other May-7 sections).
  Detail: `{"removed_line_ranges": [[261, 277], [278, 286], [287, 304], [305, 318], [319, 328], [329, 336], [430, 442], [443, 454], [455, 466], [467, 480]]}`
- `runs/longevity-mechinterp-217M/FINAL_SUMMARY.md` (source last changed 2026-05-07T18:50:28). Method: strip the '## Data-availability blocker (audit residual exposure #7 ...)' section.
  Detail: `{"removed_line_ranges": [[130, 162]]}`
- `summaries/circuit-tracing-217M-FINAL_SUMMARY.md` (source last changed 2026-05-07T21:10:10). Method: strip the A4 paragraph and the A2 re-run section.
  Detail: `{"removed_line_ranges": [[7, 20], [82, 102]]}`
- `summaries/sae-atlas-217M-FINAL_SUMMARY.md` (source last changed 2026-05-07T19:31:01). Method: strip the A8 v2/A8 v1/A4 sections and the inline A3 CI sentence.
  Detail: `{"removed_line_ranges": [[114, 204]], "removed_inline": ["**Bootstrap 95% CI (added 2026-05-07, audit action A3): [1.011, 1.019]** over 10..."]}`
- `summaries/manifold-discovery-217M-FINAL_SUMMARY.md` (source last changed 2026-05-07T19:39:22). Method: strip the A3 bootstrap block and the residual-#6 section.
  Detail: `{"removed_line_ranges": [[72, 77], [173, 197]], "removed_single_lines": [72]}`
- `summaries/spectral-geometry-217M-FINAL_SUMMARY.md` (source last changed 2026-05-07T18:54:38). Method: revert the 3 A6 wording edits using the 2026-04-17 copy; strip the inline A3 CI.
  Detail: `{"reverted_wording_edits": [{"current_lines": [97, 106], "restored_from_old_lines": [97, 103]}, {"current_lines": [140, 141], "restored_from_old_lines": [137, 138]}, {"current_lines": [145, 146], "restored_from_old_lines": [142, 142]}], "removed_inline": ["[bootstrap 95% CI 0.380, 0.384]"]}`

Confidence, file by file:
- **circuit-tracing summary: high.** The two removed blocks are marked '(added 2026-05-07, audit action A4/A2)'. The rest keeps the pre-audit text (54.6% 'near-chance').
- **sae-atlas summary: high.** Removed the A8 v2, A8 v1 and A4 sections (one contiguous block) and one inline A3 sentence. A clean pre-audit sibling also exists and is included as-is (`sae-atlas-217M-FINAL_SUMMARY_12layer.md`, 2026-04-19).
- **manifold summary (summaries/ copy): high.** After removing the A3 block and the residual-#6 section, it differs from the pre-audit runs/ copy (2026-05-07 16:41) only by the backfill sections that were never copied into summaries/. The stale 'n/t' rows are therefore genuinely pre-audit.
- **spectral summary: high.** The audit made exactly 3 wording edits (A6) and one inline CI (A3). A diff against the older 2026-04-17 copy shows those 3 edits as the only changed passages in the pre-Phase-8 text; they were restored from that copy. All other differences are Phase 8/9b additions from 2026-05-03.
- **longevity run summary: medium-high.** Removed the '## Data-availability blocker (audit residual exposure #7 ...)' section at the end. The rest reads as the pre-audit write-up (it refers to the report rebuilt at 11:40 that day). No older copy exists, so small in-place edits elsewhere cannot be ruled out; the audit completion files list only this addition for longevity.
- **topology run summary: medium.** Every '### ...' section whose heading contains 'added 2026-05-07' was removed (this is broader than the literal marker '(added 2026-05-07': only 1 of the 4 non-audit headings below contains that exact string): 6 audit sections (GroupKFold, Louvain seeds, Phase-0 seed, cross-layer replication, synthetic control, bootstrap CIs) and 4 other May-7 sections (H139, Phase 4 vs scGPT, Phase 8 chain, Phase 11 autoloop). Those 4 were most likely written before the audit (the Phase 11 section still has the pre-audit 'NOVEL POSITIVE' framing, none of the 4 mentions the audit, and the pre-audit manifold summary of 16:41 shows the executor itself used '(added 2026-05-07)' markers before the audit), and their content is in the snapshot anyway via `runs/topology-141-217M/EXTENDED_FINDINGS.md` and `outputs/phase11_autoloop/MANUAL_REVIEW.md` (both 16:37). Verifier's view: the evidence favours the version that keeps them. A version that keeps those 4 sections is at `<EVAL_ROOT>/studyC/build/alternatives/topology-141-217M_FINAL_SUMMARY.keep_non_audit_may7_sections.md` if the team prefers it.

## 4. Left out on purpose

- `README.md`: edited in place 2026-05-07 18:50 (A5/A7/residual #5); A5 rewrote parts of the topology row and the cross-pipeline synthesis with no copy of the old wording, so the pre-audit text cannot be reconstructed with confidence.
- `requirements.txt`: created 2026-05-07 17:52 by audit action A7 (did not exist before the audit).
  A partial rebuild of the README (Environment section, 'Cross-pipeline scope qualifier' and 'Field-wide regulatory-specificity context' paragraphs removed; source lines 63-135 range, 9 lines) is at `<EVAL_ROOT>/studyC/build/alternatives/README.partial_reconstruction.md`. It still contains A5 wording in the topology row and the cross-pipeline synthesis bullets. It is **not** in the snapshot.
- `audits/` (all six files), every `paper*/` folder, the public repo copy, `review_plans/`, `prompts/` (reviewer role prompts) and the audit checklist spec.
- 47 files changed on 2026-05-07 at or after 16:57: the audit repair scripts and their outputs (13 of them have 'audit' in the name), plus the edited summaries handled in section 3.
- 220 files created or changed after 2026-05-07 (the current re-runs: `v2_*` scripts and outputs, `setup/hooks_v2.py`, `setup/test_hooks_v2.py`, `attention-grn-217M/outputs/v2_eval/`, and similar). Dates seen: {'': 2, '2026-05-07': 47, '2026-10-01': 220}.
- The web atlas build (`runs/sae-atlas-217M/atlas/`, `atlas-data/`: 179 files, 338.4 MB, not counting node_modules, which was skipped entirely), except `atlas-data/cross_layer_graph.json`, which a circuit-tracing script reads.
- Model weights and SAE checkpoints (listed with sizes), raw datasets (h5ad files, 0.6-30 GB each; not listed).

## 5. Listed, not copied

| reason | files | size |
|---|---:|---:|
| web atlas build/data (not an analysis output) | 179 | 338.4 MB |
| per-head operator export (not copied to keep the snapshot small) | 88 | 66.8 MB |
| larger than 20 MB (not copied) | 51 | 9386.2 MB |
| 5-20 MB (not copied to keep the snapshot small) | 14 | 135.6 MB |
| identical copy of phase0/layer_05/gene_names.json (not copied) | 11 | 103.8 MB |

Key files an auditor cannot open (they can see them in the listings): `circuit-tracing-217M/outputs/circuit_edges.csv` (79.9 MB), `attention-grn-217M/outputs/phase0*/attention_edges_layer_mean.npy` and per-head arrays, `manifold-discovery-217M/artifacts/anchors/centroids_external.npy` (35.5 MB), `manifold-discovery-217M/outputs/phase1/cells_*.npz`, the 13 SAE checkpoints (48.6 MB each), spectral `layer_gene_embeddings.npy` files. So auditors can read all code and small outputs but cannot re-run the heaviest re-analyses (edge counts by layer, donor bootstrap on the external panel, curveball with attention scores) from the snapshot alone.

## 6. What blind auditors can and cannot find from this snapshot

- Can find (evidence is in the snapshot): the Phase 11 sign bug (code + JSON), the triplet overwrite (code + ratios), the block-deleting hooks (code; Exp-1 per-feature files show 0 edges at L6), steering invariance (steering JSONs), the curveball non-mixing (code + TRRUST file + gene lists), the 0/48 detection problem, the manifold gate/null/zero-shot issues (reports + anchor metadata), the spectral CKA/disjoint issues, spec deviations.
- Cannot find: errors that exist only in the audit files, the repair outputs or the paper (for example the corrected-CRISPRi base-rate error, the GATA1 matched-null problem, the 80% subsampling CIs). Those rows in the error ledger have `present_in_pre_audit_state` = no and should not be scored in Study C.
- Cannot fully find: errors whose only pre-audit trace was the project README (it is left out).
- Note: the deployed circuit-tracing spec itself warns (pitfall 6) that HF `output_hidden_states` indexing is off by one. This is genuine pre-audit content and stays.

## 7. Leak scan (build/leak_scan.py -> build/leak_scan_results.json)

| term group | hits | judgement |
|---|---:|---|
| added 2026-05-07 | 6 | pre-audit May-7 work: runs/manifold-discovery-217M/FINAL_SUMMARY.md (16:41) and runs/topology-141-217M/README.md (16:38); no audit content |
| audit action / residual | 0 | none |
| audits/ folder | 0 | none |
| audit spec file | 0 | none |
| reviewer | 13 | spec text about the executor-reviewer loop and a brainstormer prompt ('not a conservative reviewer'); pre-audit |
| revision | 5 | commit subjects of pinned source repos ('Add revision analyses', 'BMC Genomics revision') and 'selection revision' in the spectral summary; pre-audit |
| investigation | 5 | spec text ('focused investigation'); pre-audit |
| paper folders | 0 | none |
| framework-eval / studyC | 0 | none |
| 54.61 / 53.48 / 53.42 | 132 | numeric coincidences inside CSV/log values (e.g. 0.53422433); no CRISPRi result |
| repair outputs | 0 | none |
| V2_ / v2 re-run names | 43 | all are 'SV2_SV4'-style singular-vector names in spectral outputs; no re-run file |
| hook artefact wording | 0 | none |
| sign bug wording | 0 | none |
| off-by-one | 1 | deployed spec 02-causal-circuit-tracing.md pitfall 6 (2026-04-15), a genuine part of the contract |
| A100 | 0 | none |
| spot-check | 7 | the topology H139 'spot-check' phase name; pre-audit |
| Opus / Claude version | 0 | none |

Weak terms (innocent uses are common; every hit is in the results file):

| term | hits | files | judgement |
|---|---:|---:|---|
| audit (any) | 89 | 33 | pipeline vocabulary only: 'strict max-null audit' (H141), 'boundary-condition audits', 'Spec §8 validation audit' (spectral, 2026-05-03); no reference to the 2026-05-07 audit |
| bug (word) | 13 | 8 | spec pitfalls ('a common bug is ...'), a fixed implementation bug in the pre-audit topology summary, the exhaustive summary's own 'not a code bug' claim (pre-audit), STATUS notes |
| 2026-05-07 (date) | 15 | 4 | pre-audit May-7 work (manifold STATUS/FINAL_SUMMARY, topology README, longevity status line) |
| GATA1 | 109 | 17 | gene names in perturbation lists and gene tables |
| ChIP | 20 | 4 | spec reference lists (DoRothEA as a database); no ChIP-seq result |
| corrected | 19 | 9 | 'BH-corrected' and similar |
| overwrite | 5 | 2 | spec text on how to apply the ablation delta |
| power | 8 | 4 | spec text and one manifold sentence |

## 8. Suggestions for running Study C

- Point auditors at `snapshot/maxtoki/` only, from a workspace with no project instruction file or memory index; block reads outside it and scan transcripts afterwards.
- Do not give the original audit checklist as is: it names this project's cases (GATA1 ChIP-seq, H123 positive control, '54.6% near-chance'). Use a neutral rewrite of P1-P10 for the checklist arm.
- Score only ledger rows with `present_in_pre_audit_state` = yes or partly.

## 9. Files

- Build script: `<EVAL_ROOT>/studyC/build/build_snapshot.py`; inventory: `<EVAL_ROOT>/studyC/build/snapshot_inventory.json`; per-file hashes: `<EVAL_ROOT>/studyC/build/snapshot_files_sha256.tsv` (1,862 rows).
- Leak scan: `<EVAL_ROOT>/studyC/build/leak_scan.py` -> `<EVAL_ROOT>/studyC/build/leak_scan_results.json`.
- Alternatives (not in the snapshot): `<EVAL_ROOT>/studyC/build/alternatives`.

## Plain-words summary

The snapshot is a 221.7 MB copy of the MaxToki project as it stood just before the audit began. It has every run's code, logs, configs and small results, the pipeline specs, and the run summaries. Six summaries that the audit edited were rebuilt by cutting out the audit's additions; for five of them we are confident, for the topology one we cut a little more than needed. The project README could not be rebuilt and is left out. Audit files, paper folders, repair outputs and the new re-run files are all left out. A word scan found no leak of the audit, the later checks or the known answers; the hits are ordinary pipeline words.
