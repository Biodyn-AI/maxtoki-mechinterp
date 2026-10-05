"""Compare the T2 key reference (computed from the package data only) with the source evaluation
(runs/circuit-tracing-217M/outputs/v2_circuit/crispri/results.json, the numbers of V2_CIRCUIT_REPORT.md).
Writes build/T2/check_against_source.json. Read only on the source side."""
import json
from pathlib import Path

SRC = Path("<REPO_ROOT>/projects/maxtoki/runs/circuit-tracing-217M/"
           "outputs/v2_circuit/crispri/results.json")
REF = Path("<EVAL_ROOT>/studyA/keys/T2/reference_output.json")
OUT = Path(__file__).resolve().parent / "check_against_source.json"

s = json.load(open(SRC))["results"]["v2"]
r = json.load(open(REF))
m = r["main"]
rows = []


def add(name, src, ref, tol=6e-5):
    if isinstance(src, (list, tuple)):
        ok = all(abs(a - b) <= tol for a, b in zip(src, ref))
    else:
        ok = abs(src - ref) <= tol
    rows.append(dict(name=name, source=src, reference=ref, match=bool(ok)))


p = s["pooled"]
for k in ["frac_obs_dec", "frac_pred_dec", "accuracy", "balanced_acc", "mcc", "cohen_kappa", "chance_acc_given_marginals"]:
    add(f"pooled.{k}", p[k], m["pooled"][k])
for a, b in [("tp_pred_dec_obs_dec", "tp_pred_dec_obs_dec"), ("fn_pred_inc_obs_dec", "fn_pred_inc_obs_dec"),
             ("fp_pred_dec_obs_inc", "fp_pred_dec_obs_inc"), ("tn_pred_inc_obs_inc", "tn_pred_inc_obs_inc")]:
    add(f"pooled.{a}", p[a], m["pooled"][b], 0)
add("pooled.n_ties", p["n_ties_frac_inhib_half"], m["file_facts"]["n_ties_frac_half"], 0)
tb = [b for b in s["baselines"] if b["predictor"].startswith("per-target")][0]
add("target_baseline.accuracy", tb["accuracy"], m["baselines"]["target_direction_cross_fitted"]["accuracy"])
add("target_baseline.balanced_acc", tb["balanced_acc"], m["baselines"]["target_direction_cross_fitted"]["balanced_acc"])
add("target_baseline.mcc", tb["mcc"], m["baselines"]["target_direction_cross_fitted"]["mcc"])
wb = [b for b in s["baselines"] if b["predictor"].startswith("per-source")][0]
add("within_knockdown_baseline.accuracy", wb["accuracy"],
    m["baselines"]["within_knockdown_majority_other_target_folds_reference_only"]["accuracy"])
name_map = {"pooled_balanced_acc_minus_0.5": "pooled_balanced_acc_minus_half",
            "per_source_mean_accuracy": "per_gene_mean_accuracy",
            "per_source_mean_acc_minus_always_dec": "per_gene_mean_acc_minus_always_dec",
            "per_source_mean_balanced_acc_minus_0.5": "per_gene_mean_balanced_acc_minus_half",
            "per_source_mean_mcc": "per_gene_mean_mcc"}
for unit_s, unit_r in [("bootstrap_by_silenced_gene", "bootstrap_by_silenced_gene"),
                       ("bootstrap_by_source_feature_group", "bootstrap_by_source_feature_group"),
                       ("bootstrap_by_component", "bootstrap_by_component")]:
    add(f"{unit_s}.n_groups", s[unit_s]["n_groups"], m[unit_r]["n_groups"], 0)
    for k, v in s[unit_s]["stats"].items():
        kr = name_map.get(k, k)
        add(f"{unit_s}.{k}.point", v["point"], m[unit_r]["stats"][kr]["point"])
        add(f"{unit_s}.{k}.ci95", v["ci95"], m[unit_r]["stats"][kr]["ci95"])
ps = s["per_source_summary"]
add("per_gene.n_beat_always_dec", ps["n_sources_acc_gt_always_dec"], m["per_gene"]["n_genes_beat_always_dec"], 0)
add("per_gene.n_one_sign", ps["n_sources_all_pred_same_sign"], m["per_gene"]["n_genes_one_sign_for_all_targets"], 0)
add("per_gene.n_beat_target_baseline", ps["n_sources_model_beats_target_baseline"], m["per_gene"]["n_genes_beat_target_baseline"], 0)
add("per_gene.n_groups", ps["n_exact_membership_groups"], m["per_gene"]["n_source_feature_groups"], 0)
add("per_gene.n_components", ps["n_connected_components"], m["per_gene"]["n_components"], 0)
add("per_gene.largest_component", ps["largest_component"], m["per_gene"]["largest_component"], 0)
for i, st in enumerate(s["strata_by_abs_lfc"]):
    b = m["bands_abs_lfc"][i]
    add(f"band{i}.n", st["n"], b["n"], 0)
    add(f"band{i}.balanced_acc", st["balanced_acc"], b["balanced_acc"])
    add(f"band{i}.ci_bal", st["ci95_balanced_acc_minus_half"], b["ci95_balanced_acc_minus_half"])
    add(f"band{i}.ci_mcc", st["ci95_mcc"], b["ci95_mcc"])
ties = json.load(open(SRC))["results"]["v2_sensitivity_ties_as_decrease"]
v = r["variants"]["variants"]["ties_as_decrease"]
for k in ["accuracy", "balanced_acc", "mcc"]:
    add(f"ties_as_decrease.{k}", ties[k], v[k])
for fr in s["folds"]:
    f = r["main"]["folds"][fr["fold"]]
    add(f"fold{fr['fold']}.circuit_acc", fr["model_acc"], f["circuit_acc"])
    add(f"fold{fr['fold']}.target_acc", fr["target_majority_acc"], f["target_baseline_acc"])

n_ok = sum(x["match"] for x in rows)
json.dump(dict(n_checked=len(rows), n_match=n_ok, rows=rows), open(OUT, "w"), indent=1)
print(f"{n_ok} / {len(rows)} match")
for x in rows:
    if not x["match"]:
        print("MISMATCH", x)
