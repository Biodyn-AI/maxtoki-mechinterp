"""Add the independent verification results to checks/deployment_facts.json
and write verification/run_config.json.

Idempotent: it only sets keys (running it twice gives the same file).
If s7_assemble.py is ever re-run, run this script again afterwards.
Inputs: verification/out/v1_timing.json, v2_hashes.json, v3_adamson.json, v4_facts.json
"""
import datetime
import hashlib
import json
import os
import platform

MT = "<REPO_ROOT>/projects/maxtoki"
V = os.path.join(MT, "checks/d12_deployment_facts/verification")
FACTS = os.path.join(MT, "checks/deployment_facts.json")
L = lambda n: json.load(open(os.path.join(V, "out", n)))
v1, v2, v3, v4 = L("v1_timing.json"), L("v2_hashes.json"), L("v3_adamson.json"), L("v4_facts.json")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()

d = json.load(open(FACTS))
GF_MAIN = {e["path"]: e for e in json.load(open(os.path.join(V, "inputs/hf_geneformer_tree_main_V2-316M.json")))}["Geneformer-V2-316M/model.safetensors"]["lfs"]["oid"]

# ---------------------------------------------------------------- corrections in place
revs = v2["maxtoki_revisions_with_identical_files"]
d["checkpoints"]["maxtoki_revision"]["verification_2026_10_01"] = dict(
    same_bytes_at_revisions={r: {sub: (all(x.values()) if isinstance(x, dict) else x) for sub, x in v.items()} for r, v in revs.items()},
    note=("The hash match does not single out 21aa7b7f. The 217M files are byte-identical at 1523f7c2 (2026-03-28), "
          "e7041eb7 (2026-03-31), 21aa7b7f and today's main; the 1B files at e7041eb7, 21aa7b7f and main (the 1B folder "
          "does not exist at 1523f7c2). 21aa7b7f is chosen because it was main when the files were downloaded "
          "(2026-04-16). Any of these revisions gives the same weights."),
    evidence="checks/d12_deployment_facts/verification/out/v2_hashes.json; verification/inputs/hf_maxtoki_tree_*.json")
d["checkpoints"]["geneformer"]["verification_2026_10_01"] = dict(
    hf_main_V2_316M_model_safetensors_lfs_sha256=GF_MAIN,
    same_as_05fcbeb8=GF_MAIN == v2["files"]["Geneformer-V2-316M/model.safetensors"]["sha256"],
    note=("CORRECTION: today's HF main (and the local 04c2b2e8 snapshot) serve the same Geneformer-V2-316M "
          "model.safetensors (sha256 965ceccea819...) as 05fcbeb8. The ledger's earlier line \"'latest' would not "
          "reproduce\" was wrong for the weights. Pinning 05fcbeb8 is still advised because main can change."),
    evidence="checks/d12_deployment_facts/verification/inputs/hf_geneformer_tree_main_V2-316M.json")
d["h115_h118"]["verification_2026_10_01"] = dict(
    hits_outside_maxtoki=v4["h115_h118_counts"],
    example_files=v4["h115_h118_files"]["repos"][:6],
    prior_repo_iter0045_lines=v4.get("prior_repo_iter0045_lines"),
    note=("CORRECTION: H115 and H118 are in no MaxToki file, but they are not 'only in paper drafts'. They are "
          "hypothesis ids of the earlier topology study's own autonomous run, in the pinned reference repo "
          "repos/topology-biomechinterp2/iterations/iter_0044-0053. There, iter_0045 marks H115 'retire_now' and "
          "H118 'prioritize_hardening' (not retired). So the paper's example comes from a different, earlier run."),
    evidence="checks/d12_deployment_facts/verification/out/v4_facts.json")
d["topology_141"]["verification_2026_10_01"] = dict(
    layer_states=("12 layer states = embedding output + 11 transformer layers (setup/MaxToki-217M-HF/config.json "
                  "num_hidden_layers = 11); the strict max-null used 3 of them (3, 6, 9)."),
    single_axis_splits_in_base_run=("The base run did make held-out splits: Phase 1 built TF-disjoint and "
                                    "target-disjoint splits (3 seeds), and Phase 9 (H91) used them one axis at a time "
                                    "(scripts/phase9_stability_selection.py:150). Its docstring says 'dual disjoint', "
                                    "but the code never holds out both at once. The only both-at-once test is the "
                                    "audit GroupKFold for H123."),
    strict_max_null_recheck=v4["strict_max_null"], dual_axis_recheck=v4["dual_axis"])
d["loops"]["spectral-geometry-217M"]["verification_2026_10_01"] = dict(
    call_outcomes=v4["autoloop"]["calls"], api_errors=v4["autoloop"]["api_errors"],
    note=("Of the 6 Sonnet calls, 2 finished normally (iter_0002), 1 hit the 1,800 s timeout but left valid files "
          "(iter_0003 executor), and 3 failed with API connection errors (iter_0003 brainstormer, iter_0004 and "
          "iter_0005 executors). The driver comment at run_maxtoki_autoloop.py:46 reads 'cheaper than opus for an "
          "iterative loop'. That hints the authoring agent treated Opus as the default, but it names no version and "
          "is not a record of the model used."))
d["datasets"]["adamson_verification_2026_10_01"] = dict(
    recheck=dict(n_measured=v3["n_measured"], log2fc=v3["log2fc"], auc=v3["auc"], detection=v3["detection"],
                 bootstrap=v3["bootstrap"], expressed_targets=v3["expressed_targets_ctrl_detection_ge_5pct"],
                 not_measured=v3["not_measured"], n_double_perturbations=v3["n_double_perturbations"]),
    label_origin=("The CRISPRa label is already in the pipeline spec's reference list "
                  "(pipelines/attention-grn-extraction-and-evaluation.md:20, 'Adamson et al. 2016 (K562 CRISPRa)'). "
                  "The attention-grn summary builds on it ('Moves with context (CRISPRi vs CRISPRa ...)', "
                  "summaries/attention-grn-217M-FINAL_SUMMARY.md:100)."),
    evidence="checks/d12_deployment_facts/verification/out/v3_adamson.json")

# ---------------------------------------------------------------- verification block
d["verification"] = dict(
    date="2026-10-01", by="independent verifier (second agent), CPU only, no model forward pass",
    verdict="OK after fixes",
    scripts=[f"checks/d12_deployment_facts/verification/{s}" for s in
             ("v1_timing.py", "v2_hashes.py", "v3_adamson.py", "v4_facts.py", "v5_patch_json.py")],
    reproduced=dict(
        log_windows_overlap_aware_h=v1["log_windows"]["overlap_aware_h"],
        log_windows_sum_h=v1["log_windows"]["sum_over_pipelines_h"],
        per_pipeline_log_h=v1["log_windows"]["per_pipeline_h"],
        overlap_h=v1["log_windows"]["hours_with_2_or_more"],
        work_blocks_gap3h_h=v1["log_windows"]["work_blocks_gap3h_h"],
        span_days=v1["whole"]["span_days"], n_active_days=v1["whole"]["n_active_days"],
        audit_minutes=dict(spec=v1["audit"]["spec_min"], sweep=v1["audit"]["sweep_min"],
                           round1=v1["audit"]["round1_min"], round2=v1["audit"]["round2_min"]),
        audit_total_h=v1["audit"]["total_from_apply_h"], n_prompts_maxtoki_sessions=v1["prompts"]["n_maxtoki_sessions"],
        all_file_hashes_equal_original=all(x["equals_original_agent_hash"] for x in v2["files"].values()),
        replogle_concat_matches_hf=v2["replogle_concat_matches_state_replogle_main"],
        rpe1_matches_zenodo=v2["rpe1_zenodo_13350497"]["match"],
        adamson_85_of_85_down=v3["log2fc"]["n_down"] == 85, adamson_median_log2fc=v3["log2fc"]["median"],
    ),
    corrections=["h115_h118: they are in the pinned prior topology repo, not only in paper drafts",
                 "checkpoints.geneformer: HF main still serves the same V2-316M weights; 'latest would not reproduce' was wrong",
                 "checkpoints.maxtoki_revision: the hash match fits revisions 1523f7c2/e7041eb7 to main, not only 21aa7b7f"],
    additions=["topology_141: 12 layer states = embedding + 11 layers; base run had single-axis held-out splits",
               "loops.spectral: per-call outcomes; 3 of 6 calls failed with API errors",
               "datasets.adamson: second statistic (AUC), expressed-target subset 82/82, label origin in the spec"],
)
json.dump(d, open(FACTS, "w"), indent=2, default=str)

# ---------------------------------------------------------------- run_config
H = os.path.expanduser("~/.claude/history.jsonl")
inputs = {
    "~/.claude/history.jsonl (read only; it grows as new prompts are typed, so this hash is a snapshot)": v1["prompts"]["history_sha256"],
    "checks/d12_deployment_facts/out/provenance.json": sha(os.path.join(MT, "checks/d12_deployment_facts/out/provenance.json")),
    "runs/topology-141-217M/outputs/phase12/h141_strict_margins.csv": sha(MT + "/runs/topology-141-217M/outputs/phase12/h141_strict_margins.csv"),
    "runs/topology-141-217M/outputs/groupkfold_h123/summary.json": sha(MT + "/runs/topology-141-217M/outputs/groupkfold_h123/summary.json"),
    "runs/spectral-geometry-217M/autoloop/runtime/driver.log": sha(MT + "/runs/spectral-geometry-217M/autoloop/runtime/driver.log"),
    "runs/attention-grn-217M/outputs/phase1_adamson/per_perturbation_auroc.csv": v3["project_csv_sha256"],
}
inputs.update({f"verification/inputs/{k}": v for k, v in v2["inputs_sha256"].items()})
inputs.update({f"weights/data: {k}": x["sha256"] for k, x in v2["files"].items()})
rc = dict(
    item="Verification of D11 + D12 deployment facts ledger",
    created=datetime.datetime.now().isoformat(timespec="seconds"),
    python=platform.python_version(), platform=platform.platform(),
    seeds=dict(adamson_bootstrap=[42, 2026]),
    run_order=["v1_timing.py", "v2_hashes.py", "v3_adamson.py", "v4_facts.py", "v5_patch_json.py"],
    scripts={s: sha(os.path.join(V, s)) for s in ("v1_timing.py", "v2_hashes.py", "v3_adamson.py", "v4_facts.py", "v5_patch_json.py")},
    outputs={o: sha(os.path.join(V, "out", o)) for o in sorted(os.listdir(os.path.join(V, "out"))) if not o.startswith("._")},
    inputs_sha256=inputs,
    network_reads="public Hugging Face model/dataset tree and commit APIs and Zenodo record 13350497, fetched 2026-10-01 into verification/inputs/",
    no_model_forward_pass=True, device="CPU only",
)
json.dump(rc, open(os.path.join(V, "run_config.json"), "w"), indent=1)
print("patched", FACTS)
print(json.dumps(d["verification"]["reproduced"], indent=1))
