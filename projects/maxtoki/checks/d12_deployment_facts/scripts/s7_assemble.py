"""D11 + D12 step 7: assemble checks/deployment_facts.json and run_config.json.

Reads only the out/*.json files written by s1-s6, the manuscript (to quote the
claims being checked, read-only), and machine info from sysctl.
Curated items (phase coverage notes, claim list) carry an evidence path each.
"""
import datetime
import glob
import hashlib
import json
import os
import platform
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import MT, REPO, OUT, INP, HERE, PIPELINES, iso, sha256_file, union, total_hours

L = lambda n: json.load(open(os.path.join(OUT, n)))
tl, hp, pv, am, pf, au = (L("timeline.json"), L("human_prompts.json"), L("provenance.json"),
                          L("adamson_modality.json"), L("pipeline_facts.json"), L("audit_timing.json"))
CHECKS = os.path.join(MT, "checks")
R = "projects/maxtoki/"

# ------------------------------------------------------------ Table 3 hours in the manuscript (read-only)
tex = open(MT + "/paper-plos-one/main.tex").read().splitlines()
t3 = [l.strip().rstrip("\\").strip() for l in tex[929:1013] if re.fullmatch(r"\s*[0-9.]+\+?\s*\\\\\s*", l)]
T3 = dict(zip(PIPELINES, t3))
assert len(t3) == 8, t3

# ------------------------------------------------------------ base vs later log hours
ts = lambda s: datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp()
logs = tl["logs"]
split = {}
for p in PIPELINES:
    w = pf["session_windows"][p]
    base = [ts(w["base"][0]), ts((w["same_session"] or w["base"])[1])]
    parts = {"first_session": [], "later_sessions": []}
    for l in logs:
        if l["pipeline"] != p:
            continue
        a, b = ts(l["birth"]), ts(l["mtime"])
        parts["first_session" if base[0] <= a <= base[1] else "later_sessions"].append((a, b))
    split[p] = {k: round(total_hours(union(v)), 2) for k, v in parts.items()}

# ------------------------------------------------------------ sessions per pipeline (from prompts)
tag2p = {"spectral-geometry": "spectral-geometry-217M", "attention-grn": "attention-grn-217M", "topology-141": "topology-141-217M",
         "sae-atlas": "sae-atlas-217M", "circuit-tracing": "circuit-tracing-217M", "exhaustive-mapping": "exhaustive-mapping-217M",
         "manifold-discovery": "manifold-discovery-217M", "longevity": "longevity-mechinterp-217M"}
sess = {p: sorted({x["session"] for x in hp["prompts"] if tag2p.get(x["pipeline"]) == p and x["category"] != "spec_authoring"}) for p in PIPELINES}
nprompts = {p: sum(1 for x in hp["prompts"] if tag2p.get(x["pipeline"]) == p and x["category"] not in ("spec_authoring", "paper")) for p in PIPELINES}

# ------------------------------------------------------------ curated coverage notes (each with evidence)
COVER = {
    "spectral-geometry-217M": dict(
        spec_phases_run="all 11 spec phases have some run evidence; Phase 2 is the shared co-pole test inside phases_345.py; Phase 8 (cell-sample stability, not fine-tuning seeds) and Phase 9b were run on 2026-05-03; Phase 10 (autonomous loop) ran only as a 4-iteration Sonnet autoloop on 2026-05-05 that aborted",
        not_run_or_reduced=["Phase 8 as specified (3 fine-tuning seeds): replaced by 3 cell samples",
                            "Phase 10: 4 iterations (spec: 40-80), aborted after 2 API connection failures; the pipeline summary still says 'Phase 10 ... NOT run'"],
        evidence=["runs/spectral-geometry-217M/outputs/phases_345.log", "runs/spectral-geometry-217M/outputs/phase8.log",
                  "runs/spectral-geometry-217M/autoloop/runtime/driver.log", "summaries/spectral-geometry-217M-FINAL_SUMMARY.md (section 'Phase 2 / 10 — status')"]),
    "attention-grn-217M": dict(
        spec_phases_run="0a, 1, 2, 3, 4, 12 first (K562 217M); then 0b and 6 (CSSI) the same day; phases 0a, 1, 2, 3, 12 repeated for RPE1, Adamson and the 1B model",
        not_run_or_reduced=["Phases 5, 7, 8, 9, 10, 11 not run as specified (README: '5-11 ... skipped'); the RPE1/Adamson/1B re-runs cover part of phase 5 (cross-context) and phase 9 (multi-model)",
                            "Phase 3 Curveball null: 50 permutations (spec 200)", "Phase 4: 6 conditions (spec 13)",
                            "1B run: 200 control cells (217M runs: 2,000)"],
        evidence=["runs/attention-grn-217M/README.md (table 'Phases run')", "runs/attention-grn-217M/scripts/phase3_residualization.py:9-10",
                  "runs/attention-grn-217M/outputs/phase0_k562_1b/run_config.json"]),
    "topology-141-217M": dict(
        spec_phases_run="base run 2026-05-03: phases 0, 1-3 (one script), 5, 6, 7+8, 9, 12, 13 on 3 domains; later sessions: 14-16 (not spec phases) on 2026-05-05; phase 10 spot-check, phase 8 extensions, phase 4 vs scGPT and phase 11 substitute batch on 2026-05-07",
        not_run_or_reduced=["only the 'headline backbone' of the 141 hypotheses; the 130+ explorer hypotheses were not re-screened",
                            "Phase 11: a scripted 6-hypothesis batch (no LLM, no Codex), not an autonomous loop",
                            "Phase 12 strict max-null on layers 3, 6, 9 only (of 12 layer states)",
                            "Phase 10: 2 spot-checks (H23 Forman direction, H139) instead of the 70+ remaining hypotheses",
                            "immune domain reuses the spectral-geometry Phase-0 embeddings (2,000 TS immune cells)"],
        evidence=["runs/topology-141-217M/README.md", "runs/topology-141-217M/FINAL_SUMMARY.md (section 'Scope')",
                  "runs/topology-141-217M/scripts/phase12_strict_max_null.py:174", "runs/topology-141-217M/outputs/phase0/immune/run_config.json"]),
    "sae-atlas-217M": dict(
        spec_phases_run="all 14 spec phases have run evidence; base run 2026-04-17 on 3 layers (0, 5, 11); phases 6 and 9 on 04-18; full 12-layer re-run plus phases 12-13 on 04-19 (the re-run overwrote the 3-layer phase0/1/2/5/12 outputs); web atlas deployed 05-03",
        not_run_or_reduced=["phase0 activations and cell-id files were deleted after training (cannot rerun SAE training without re-extraction)",
                            "the 04-17 layer-5 SAE used by Phase 8t was overwritten by the 04-19 run"],
        evidence=["runs/sae-atlas-217M/outputs/phases2_to_8.log", "runs/sae-atlas-217M/outputs/remaining_phases.log",
                  "runs/sae-atlas-217M/outputs/full_12layer.log", "runs/sae-atlas-217M/outputs/phase6.log"]),
    "circuit-tracing-217M": dict(
        spec_phases_run="phases 0, 1, (2-4 inside the tracing loop), 5, 6 overnight 04-19/20 at source layers L0/L3/L6/L9 (30 features each, 200 cells); phases 7, 9, 10, 11, 12 on 04-20 morning",
        not_run_or_reduced=["Phase 8 (four experimental conditions) not run", "Phase 13 (cross-model comparison) not run as a phase; a summary was written"],
        evidence=["runs/circuit-tracing-217M/outputs/circuit_trace.log", "runs/circuit-tracing-217M/outputs/remaining_phases.log"]),
    "exhaustive-mapping-217M": dict(
        spec_phases_run="Experiment 1 (phases 0-5; 1,000 features at L5), Experiment 2 (phases 6-8; 4 triplets) and Experiment 3 (phases 9-12) on 04-20; Experiment 2 and 3 re-run and phase 13 synthesis on 04-23",
        not_run_or_reduced=["Experiment 1: 1,000 of ~4,928 L5 features (spec: all ~4,065 active)", "Experiment 2: 4 triplets (spec: 8)"],
        evidence=["runs/exhaustive-mapping-217M/outputs/experiment1.log", "runs/exhaustive-mapping-217M/outputs/experiment2_rerun.log",
                  "runs/exhaustive-mapping-217M/outputs/experiment3_rerun.log", "summaries/exhaustive-mapping-217M-FINAL_SUMMARY.md (section 'Scope reductions')"]),
    "manifold-discovery-217M": dict(
        spec_phases_run="base run 05-03 19:41 to 05-04 08:31: phases 0, 1, 2, 4-13 (12 and 13 at 'LITE' scope) on the pre-set H65 ordering; phase 3a sweeps #1-#2 (12 candidates) on 05-05; sweep #3 (6 candidates) and phases 15-17 (not spec phases) on 05-07",
        not_run_or_reduced=["Phase 12 full cell-level benchmark (scVI/Palantir) deferred; only anchor-level LITE", "Phase 13 H38 at LITE scope",
                            "Phase 3b (author-led closure) has no artefact", "no retirement code; the quality-gate file lists 5 gates, the permutation gate was never computed"],
        evidence=["runs/manifold-discovery-217M/STATUS.md", "runs/manifold-discovery-217M/reports/quality_gates_spec.json"]),
    "longevity-mechinterp-217M": dict(
        spec_phases_run="stages 0 (setup) and 1 (frozen probes) on 05-05; gate G1 failed",
        not_run_or_reduced=["stages 2-5 skipped (gated on stage 1)", "stages 6-11 deferred (need AIDA data)", "stage 12 not applicable"],
        evidence=["runs/longevity-mechinterp-217M/README.md (Status table)", "runs/longevity-mechinterp-217M/FINAL_SUMMARY.md"]),
}

DEVICE_NOTE = {
    "spectral-geometry-217M": "MPS (phase0.log, phase8.log, run_config.json; autoloop h3g_run.log 'Device: mps')",
    "attention-grn-217M": "MPS for all 4 runs including the 1B model (run_config.json device=mps in all phase0 runs)",
    "topology-141-217M": "MPS for lung/external-lung extraction (run_config.json); immune embeddings reused from spectral-geometry; later phases CPU analysis",
    "sae-atlas-217M": "MPS for extraction AND SAE training (full_12layer.log 'device=mps'; setup/topk_sae.py default device='mps')",
    "circuit-tracing-217M": "logs print no device; script default MPS (circuit_trace.py:37)",
    "exhaustive-mapping-217M": "logs print no device; script default MPS (experiment1_exhaustive.py:34, experiment2_rerun.py:43, experiment3_rerun.py:42)",
    "manifold-discovery-217M": "MPS for forward passes (phase1b logs); CPU for LET heads, head scan, compaction, probes (7 scripts set DEVICE='cpu')",
    "longevity-mechinterp-217M": "log prints no device; runtime picks cuda>mps>cpu (setup/maxtoki_adapter.py:34-39), no CUDA on this machine; README says MPS",
}

TOKENS = {
    "spectral-geometry-217M": dict(max_len=2048, max_gene_tokens=2046, mean_seq_len=pf["token_budget"]["recorded_sequence_lengths"]["spectral-geometry-217M"], cells="2,000 TS immune cells"),
    "attention-grn-217M": dict(max_len=2048, max_gene_tokens=2046, mean_seq_len=pf["token_budget"]["recorded_sequence_lengths"]["attention-grn-217M"], cells="2,000 control cells per 217M run; 200 for 1B"),
    "topology-141-217M": dict(max_len=2048, max_gene_tokens=2046, mean_seq_len="not recorded (lung, external lung 1,500 cells each); immune reused from spectral-geometry (mean 1,783.6)", cells="1,500 per new domain"),
    "sae-atlas-217M": dict(max_len=2048, max_gene_tokens=2046, mean_seq_len=pf["token_budget"]["recorded_sequence_lengths"]["sae-atlas-217M"], cells="500 K562 controls; 500 TS immune (phase 9)"),
    "circuit-tracing-217M": dict(max_len=2048, max_gene_tokens=2046, mean_seq_len="not recorded", cells="200 K562 controls (circuit_trace.log 'cell 200/200')"),
    "exhaustive-mapping-217M": dict(max_len=2048, max_gene_tokens=2046, mean_seq_len="not recorded", cells="K562 controls (exp 1-2); TS immune (exp 3)"),
    "manifold-discovery-217M": dict(max_len=4096, max_gene_tokens=4094, mean_seq_len=pf["token_budget"]["recorded_sequence_lengths"]["manifold-discovery-217M"], cells="panels of 1,500-30,000 TS cells"),
    "longevity-mechinterp-217M": dict(max_len=1024, max_gene_tokens=1022, mean_seq_len=pf["token_budget"]["recorded_sequence_lengths"]["longevity-mechinterp-217M"], cells="2,974 TS immune-subset cells"),
}

pipelines = {}
for p in PIPELINES:
    pipelines[p] = dict(
        spec=pf["spec_phases"][p]["spec"], spec_phase_count=pf["spec_phases"][p]["n"] if p != "exhaustive-mapping-217M" else 14,
        phase_evidence=pf["phase_evidence"][p],
        coverage=COVER[p],
        file_write_days=tl["per_pipeline_files"][p]["days"],
        first_file=tl["per_pipeline_files"][p]["first"], last_file=tl["per_pipeline_files"][p]["last"],
        compute_hours=dict(
            log_windows=tl["A_log_windows"]["per_pipeline"][p]["hours"],
            log_windows_first_session=split[p]["first_session"], log_windows_later_sessions=split[p]["later_sessions"],
            activity_sessions_gap30=tl["B_activity_sessions"]["gap_30min"]["per_pipeline"][p]["hours"],
            logged_durations_lower_bound=tl["C_logged_durations"]["per_pipeline_hours"][p],
            logged_duration_items={k: v["seconds"] for k, v in tl["C_logged_durations"]["items"][p].items()},
            log_window_list=tl["A_log_windows"]["per_pipeline"][p]["windows"],
            work_blocks_gap3h=tl["work_blocks_gap3h"][p],
            work_blocks_gap3h_total=round(sum(x["hours"] for x in tl["work_blocks_gap3h"][p]), 2),
            paper_table3_agent_hours=T3[p]),
        device=DEVICE_NOTE[p], device_counts=pf["devices"][p],
        agent_model=dict(
            recorded_in_run_files=("Claude 'sonnet' alias for all 6 autoloop executor/brainstormer calls on 2026-05-05 (exact version not recorded); main sessions not recorded"
                                   if p == "spectral-geometry-217M" else "not recorded"),
            claimed="Claude Opus 4.7 (paper/LLM_USAGE_STATEMENT.txt:15, agent-written 2026-05-08 01:19; paper-plos-one/main.tex:1020)",
            sessions=sess[p], n_human_prompts=nprompts[p]),
        token_budget=TOKENS[p],
        input_format="single cells: <bos> + that cell's genes ranked by median-normalised expression + <eos>; no multi-cell trajectories, no time or query tokens",
    )

machine = dict(hw_model=subprocess.run(["sysctl", "-n", "hw.model"], capture_output=True, text=True).stdout.strip(),
               cpu=subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
               mem_gb=int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip()) / 2**30,
               note="machine that holds the files on 2026-10-01; run files record only 'MacBook Pro (Apple Silicon, 32 GB)' and device 'mps'; same machine is likely but not proven")

facts = dict(
    meta=dict(generated=datetime.datetime.now().isoformat(timespec="seconds"), timezone="local machine time, CEST (UTC+2) in April-May 2026",
              scripts=sorted(os.path.relpath(s, MT) for s in glob.glob(HERE + "/scripts/*.py")),
              report=R + "checks/DEPLOYMENT_FACTS.md", run_config=R + "checks/d12_deployment_facts/run_config.json",
              cutoff_for_deployment_files="2026-05-08 06:00 (later files are revision work and are excluded)"),
    whole_deployment=dict(
        first_artefact=tl["deployment"]["first_artefact"], first_run_folder_artefact=tl["deployment"]["first_run_folder_artefact"],
        last_artefact=tl["deployment"]["last_artefact"], span_days=tl["deployment"]["span_days"],
        active_days=tl["deployment"]["active_days"], n_active_days=tl["deployment"]["n_active_days"],
        n_distinct_clock_hours_with_a_write=tl["deployment"]["n_distinct_clock_hours_with_a_write"],
        compute_hours=dict(
            log_windows_overlap_aware=tl["A_log_windows"]["total"]["overlap_aware_h"],
            log_windows_sum_over_pipelines=tl["A_log_windows"]["total"]["sum_of_pipelines_h"],
            activity_sessions_overlap_aware={k: v["overlap_aware_h"] for k, v in tl["B_activity_sessions"].items()},
            logged_durations_lower_bound_sum=round(sum(tl["C_logged_durations"]["per_pipeline_hours"].values()), 2),
            paper_table3_sum=round(sum(float(x.rstrip("+")) for x in t3), 1),
            concurrency=tl["A_log_windows"]["concurrency"],
            what_it_is="laptop wall-clock time of logged jobs (MPS forward passes plus CPU analysis); not GPU-hours, not agent thinking time"),
        hardware=dict(device_in_every_log_or_config_that_names_one="mps", cuda_or_a100_in_any_run_log=False, machine_now=machine),
    ),
    pipelines=pipelines,
    audit=au,
    human_supervision=dict(
        source="~/.claude/history.jsonl (typed prompts only)",
        n_prompts_maxtoki_sessions_apr15_may8=hp["n_prompts_maxtoki_sessions"],
        n_deployment_prompts_excl_spec_authoring_and_paper=hp["n_deployment_prompts_excl_spec_and_paper"],
        by_category=hp["by_category"], per_day=hp["per_day"], by_pipeline=hp["by_pipeline"],
        excluded_sessions_same_folder=hp["excluded_sessions_same_folder"],
        supervision_time=hp["supervision_time"],
        paper_claim="about 15 hours of human supervision (paper-plos-one/main.tex:224, 575, 661, 1697): no record supports or refutes it"),
    topology_141=pf["topology"],
    loops=pf["loops"],
    h115_h118=pf["h115_h118_search"],
    checkpoints=dict(maxtoki_revision=pv["maxtoki_revision"],
                     maxtoki_files={k: v for k, v in pv["models"].items() if k.startswith("MaxToki-")},
                     geneformer=dict(pv["geneformer"], files={k: v for k, v in pv["models"].items() if k.startswith("Geneformer-V2")},
                                     referenced_by=pv["geneformer_snapshot_referenced_by"]),
                     scgpt=dict(checkpoint=pv["models"]["scGPT whole-human best_model.pt"], args_json=pv["models"]["scGPT whole-human args.json"],
                                args_save_dir=pv["scgpt_args_save_dir"], derived_embeddings=pv["models"]["scGPT derived gene embeddings (.pt)"],
                                referenced_by=pv["scgpt_embedding_file_referenced_by"],
                                note="scGPT is not on HF; the run used only a gene-embedding table derived from the whole-human checkpoint"),
                     tokenizer_and_medians={k: v for k, v in pv["models"].items() if k.startswith(("MaxToki token", "Geneformer gene"))}),
    datasets=dict(files=pv["datasets"], perturbation_counts=L("perturbation_counts.json"), dataset_loader_names_by_pipeline=pv["dataset_loader_names_by_pipeline"],
                  rpe1_obs_perturbation_type=pv["rpe1_obs_perturbation_type_categories"],
                  replogle_concat_public_source="huggingface.co/datasets/arcinstitute/State-Replogle-Filtered (file replogle_concat.h5ad; sha256 identical; HF commits dated 2025-12-06)",
                  rpe1_public_source="Zenodo scPerturb RNA h5ad records 7041849 / 7278143 / 7416068 / 10044268 / 13350497 (md5 identical)",
                  adamson=dict(am, public_source="GEARS-processed Adamson 2016 data (inferred from uns keys and 'GENE+ctrl' labels); local file name has '_symbols'; byte identity not checkable")),
)
path = os.path.join(CHECKS, "deployment_facts.json")
json.dump(facts, open(path, "w"), indent=2, default=str)
print(path)

# ------------------------------------------------------------ run_config.json
inputs = {}
for l in logs:
    inputs[l["log"]] = l["sha256"]
for f in sorted(glob.glob(INP + "/*.json")):
    inputs[os.path.relpath(f, MT)] = sha256_file(f)
for rel in ["paper-plos-one/main.tex", "paper/LLM_USAGE_STATEMENT.txt", "setup/maxtoki_adapter.py", "setup/dataset_loader.py", "setup/topk_sae.py",
            "runs/spectral-geometry-217M/autoloop/run_maxtoki_autoloop.py", "runs/topology-141-217M/scripts/phase11_autoloop.py",
            "runs/topology-141-217M/scripts/phase12_strict_max_null.py", "runs/topology-141-217M/outputs/phase12/h141_strict_margins.csv",
            "runs/topology-141-217M/outputs/groupkfold_h123/summary.json", "runs/topology-141-217M/outputs/phase11_autoloop/autoloop_summary.json",
            "runs/topology-141-217M/outputs/phase11_autoloop/MANUAL_REVIEW.md", "runs/manifold-discovery-217M/STATUS.md",
            "runs/manifold-discovery-217M/outputs/phase1/panel_summary.json", "runs/attention-grn-217M/outputs/phase1_adamson/per_perturbation_auroc.csv",
            "runs/longevity-mechinterp-217M/outputs/stage1_20260505/extraction_stats.json", "setup/token_dictionary.json",
            "summaries/spectral-geometry-217M-FINAL_SUMMARY.md", "audits/audit-20260507.md", "audits/audit-20260507-completion.md",
            "audits/audit-20260507-completion-v2.md"] + [f"runs/manifold-discovery-217M/outputs/phase1/cells_{x}.npz" for x in ("internal", "external", "zeroshot", "lung_control", "lung_nonhema")]:
    inputs[rel] = sha256_file(os.path.join(MT, rel))
inputs["../../pipelines/audit-recurring-review-issues.md"] = sha256_file(REPO + "/pipelines/audit-recurring-review-issues.md")
for p, s in pf["spec_phases"].items():
    inputs["../../" + s["spec"]] = s["sha256"]
inputs[os.path.expanduser("~/.claude/history.jsonl")] = hp["source"]["sha256"]
big = {}
for k, v in pv["models"].items():
    big[v["path"]] = v["sha256"]
for k, v in pv["datasets"].items():
    big[v["path"]] = v["sha256"]
rc = dict(
    item="D11 + D12 deployment facts ledger", created=datetime.datetime.now().isoformat(timespec="seconds"),
    python=sys.version.split()[0], platform=platform.platform(),
    packages={m: __import__(m).__version__ for m in ("numpy", "scipy", "h5py")},
    seeds=dict(adamson_null_and_bootstrap=42),
    run_order=["s1_timeline.py", "s2_history.py", "s3_provenance.py hash-small", "s3_provenance.py hash-big", "s3_provenance.py report",
               "s4_adamson_modality.py", "s4b_perturbation_counts.py", "s5_pipeline_facts.py", "s6_audit.py", "s7_assemble.py"],
    scripts={os.path.relpath(s, MT): sha256_file(s) for s in sorted(glob.glob(HERE + "/scripts/*.py"))},
    outputs={os.path.relpath(f, MT): sha256_file(f) for f in sorted(glob.glob(OUT + "/*")) if not f.endswith("sha256_cache.json")},
    inputs_sha256=inputs, weights_and_datasets_sha256=big,
    file_metadata_snapshot=dict(path="checks/d12_deployment_facts/out/file_index.csv", sha256=sha256_file(OUT + "/file_index.csv"),
                                note="birth/modification times of every file under runs/ summaries/ audits/ setup/ (exFAT keeps both)"),
    network_reads="public Hugging Face and Zenodo metadata APIs, saved under inputs/ on 2026-10-01",
    no_model_forward_pass=True, device="CPU only",
)
json.dump(rc, open(os.path.join(HERE, "run_config.json"), "w"), indent=2)
print(os.path.join(HERE, "run_config.json"))
print(json.dumps(dict(T3=T3, split=split, sess=sess, nprompts=nprompts), indent=1))
