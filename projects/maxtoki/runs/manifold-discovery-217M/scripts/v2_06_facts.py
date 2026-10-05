"""v2 step 6: FACTS — exact implemented definitions and counts behind the manifold claims, computed from saved files.

Writes outputs/v2_intervals/v2_06_facts.json, facts_cohorts.csv, facts_orderings.csv.
No model is run. Every number is read from or computed on files under runs/manifold-discovery-217M/.
"""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, "<REPO_ROOT>/projects/maxtoki/runs/manifold-discovery-217M/scripts")
import json, os, time
import numpy as np, pandas as pd
from v2_common import *

F = {}
dag = stage_dag(); s2b = {k: v["branch"] for k, v in dag["stage_to_branch"].items()}

# ------------------------------------------------------------------ cohorts
metas = {p: pd.read_csv(ART / f"anchors/anchor_meta_{p}.csv") for p in ["internal", "external", "zeroshot", "lung_nonhema", "lung_control"]}
cells = {}
for p in ["internal", "external", "zeroshot", "lung_nonhema", "lung_control"]:
    fn = RUN / f"outputs/phase1/cells_{p}_obs.csv"
    cells[p] = pd.read_csv(fn, usecols=["anchor_id", "obs_label", "donor_id"])
coh_rows = []
for p, m in metas.items():
    br = m["hema_stage"].map(s2b).fillna("_unk")
    coh_rows.append({
        "panel": p, "n_anchors": len(m), "n_donors": m["donor_id"].nunique(), "n_tissues": m["tissue"].nunique(),
        "n_cell_types": m["cell_type"].nunique(), "n_stage_labels": m["hema_stage"].nunique(),
        "n_cells_used_in_centroids": int(m["n_cells_centroided"].sum()),
        "n_cells_gathered_phase1a": int(len(cells[p])),
        "cells_per_anchor_min": int(m["n_cells_centroided"].min()), "cells_per_anchor_median": float(m["n_cells_centroided"].median()),
        "cells_per_anchor_max": int(m["n_cells_centroided"].max()),
        "group_size_in_atlas_min": int(m["n"].min()), "group_size_in_atlas_max": int(m["n"].max()),
        "anchors_per_donor": json.dumps(m["donor_id"].value_counts().to_dict()),
        "anchors_per_branch": json.dumps(br.value_counts().to_dict()),
        "top_donor_share": float(m["donor_id"].value_counts().iloc[0] / len(m)),
        "anchors_at_progenitor_stages_HSC_MPP_CMP_CLP_GMP_MEP": int(m["hema_stage"].isin(["HSC", "MPP", "CMP", "CLP", "GMP", "MEP"]).sum()),
        "tissues": "|".join(sorted(m["tissue"].astype(str).unique())),
    })
coh = pd.DataFrame(coh_rows); coh.to_csv(OUT / "facts_cohorts.csv", index=False)
don = {p: set(m["donor_id"].astype(str)) for p, m in metas.items()}
tis = {p: set(m["tissue"].astype(str)) for p, m in metas.items()}
aid = {p: set(m["anchor_id"].astype(str)) for p, m in metas.items()}
cel = {p: set(c["obs_label"].astype(str)) for p, c in cells.items()}
key4 = {p: set(zip(m.donor_id, m.tissue, m.cell_type)) for p, m in metas.items()}
F["overlap"] = {
    "internal_external_donors": sorted(don["internal"] & don["external"]),
    "internal_zeroshot_donors": sorted(don["internal"] & don["zeroshot"]),
    "zeroshot_donors_subset_of_external": don["zeroshot"] <= don["external"],
    "external_donors_not_in_zeroshot": sorted(don["external"] - don["zeroshot"]),
    "zeroshot_external_shared_donors": int(len(don["zeroshot"] & don["external"])),
    "zeroshot_external_shared_anchor_ids": int(len(aid["zeroshot"] & aid["external"])),
    "zeroshot_external_shared_cells_gathered": int(len(cel["zeroshot"] & cel["external"])),
    "internal_external_shared_cells_gathered": int(len(cel["internal"] & cel["external"])),
    "internal_zeroshot_shared_cells_gathered": int(len(cel["internal"] & cel["zeroshot"])),
    "zeroshot_tissues": int(len(tis["zeroshot"])), "zeroshot_tissues_also_in_external": int(len(tis["zeroshot"] & tis["external"])),
    "zeroshot_anchors_sharing_donor_tissue_celltype_with_an_external_anchor": int(sum(
        (r.donor_id, r.tissue, r.cell_type) in key4["external"] for r in metas["zeroshot"].itertuples())),
    "lung_nonhema_donors": sorted(don["lung_nonhema"]),
    "lung_nonhema_donors_in_internal_training": sorted(don["lung_nonhema"] & don["internal"]),
    "lung_nonhema_donors_in_external": sorted(don["lung_nonhema"] & don["external"]),
    "lung_nonhema_stage_labels": "drawn uniformly at random from the 34 DAG stages (phase1a_lung_nonhema_panel.py:91)",
    "lung_control_v1_note": "first lung control = lung-resident IMMUNE cells (51 anchors); it passed all four gates "
                            "(reports/external_validation_lung_control.json) and was replaced by lung_nonhema",
}
F["anchor_definition"] = ("anchor = all cells sharing donor x tissue x cell_type x hema_stage, >= 5 cells in the atlas "
                          "(phase1a_subsample_and_tokenize.py:65-79); the LARGEST groups are kept first (lines 94-106); "
                          "<= 50 cells per anchor gathered (line 49); for the forward pass external was cut to 20 and "
                          "zero-shot to 32 cells per anchor (phase1bc_hidden_states_and_centroids.py:50-54, 122-142). "
                          "Centroid = mean over the anchor's cells of the per-cell hidden state; per-cell hidden state = "
                          "mean over gene-token positions (BOS/EOS excluded) at each of the 12 hidden-state entries.")

# ------------------------------------------------------------------ trustworthiness definition (numbers + code refs)
F["trustworthiness"] = {
    "call": "sklearn.manifold.trustworthiness(X=features, X_embedded=z, n_neighbors=15), Euclidean metric",
    "code": ["phase5_let_anchor.py:236", "phase7_external_validation.py:94", "phase8_zeroshot_transfer.py:76",
             "phase15_validate_candidate.py:148"],
    "X": "2,464-d pooled-drift features of each anchor centroid, standardised with the INTERNAL panel mean/std",
    "X_embedded": "the 10-d LET head output z for the same anchors",
    "uses_biological_ruler": False,
    "unit": "anchors (centroids), not cells",
    "internal": "in-sample: the head was fitted on the same 290 anchors",
    "external_zeroshot_lung": "frozen head; no re-fit",
    "k": 15,
    "plain": "Of each anchor's 15 nearest anchors in the 10-d head output, how many were also near in the 2,464-d model "
             "feature space (penalised by how far down the feature-space ranking they sit). It asks whether the small "
             "read-out keeps the model's own neighbourhoods. It does not compare the model with biology.",
}

# ------------------------------------------------------------------ holdout definitions
F["branch_metrics"] = {
    "internal_branch_holdout": "for each branch with >= 3 anchors: re-fit the head WITHOUT that branch, project the "
                               "held-out anchors, Spearman(arc-cos latent distance, ruler distance) over held-out pairs; "
                               "plain mean over branches whose ruler is not constant (phase5_let_anchor.py:167-199). "
                               "6 branches scored (T, B, monocyte, granulocyte, erythroid, stem).",
    "external_zeroshot_lung_within_branch": "head frozen (trained on ALL internal branches); for each branch with >= 3 "
                               "anchors, Spearman(arc-cos latent, ruler) over within-branch pairs; plain mean "
                               "(phase7_external_validation.py:132-150, phase8_zeroshot_transfer.py:120-139). Nothing is "
                               "held out. Branches with one stage give a constant ruler and are dropped.",
    "donor_metric_internal": "refit leaving one donor out (donors with >= 3 anchors)",
    "donor_metric_frozen": "mean within-donor Spearman under the frozen head",
    "random_metric_internal": "10 random 80/20 anchor splits with refit (n_iters=10; comment says 20), phase5:240",
    "random_metric_frozen": "20 random 20% anchor subsets, frozen head, no refit (phase7:101-114)",
    "branch_labels": "branch = stage_to_branch of planning/h65_stage_dag.json",
}

# ------------------------------------------------------------------ compaction
npz = np.load(ART / "operators/pooled_drift_components.npz")
A_bytes = int(npz["A_early"].nbytes + npz["A_mid"].nbytes + npz["A_late"].nbytes)
A_top = np.load(ART / "operators/layer10_head6.npy")
cc = json.loads((REP / "compaction_chain.json").read_text())
cc = {r["label"]: r for r in cc}
n_fac, n_sel = 16, 60
eff_entries = n_fac * (n_sel + n_sel) + n_fac
F["compaction"] = {
    "full_drift_operator": {"what": "A_early, A_mid, A_late: each the mean over a layer block of the full o_proj "
                                    "matrix transposed (1232 x 1232), applied to block-mean hidden states; feature = "
                                    "concat(y_early - y_mid, y_mid - y_late) = 2,464-d",
                            "shape_each": list(npz["A_early"].shape), "dtype": str(npz["A_early"].dtype),
                            "bytes_total": A_bytes, "mb_in_report": cc["full_drift_reference"]["mb_size"]},
    "single_head_L10H6": {"what": "A = o_proj(layer 10)[:, 6*154:7*154].T, shape (154, 1232); feature = h @ A.T (154-d), "
                                  "where h = centroid entry 11 = the LAST hidden state",
                          "shape": list(A_top.shape), "bytes": int(A_top.nbytes),
                          "mb_in_report": cc["compact_top1_L10H6"]["mb_size"]},
    "hard_sparse_16f_60g": {
        "what": "keep the top 16 singular triplets (u_k, s_k, v_k) of A_L10H6; in each, keep the 60 largest-|value| "
                "entries of u_k (60 of 154 head-dimension coordinates) and of v_k (60 of 1,232 residual-stream "
                "coordinates); A_sparse = sum_k s_k u_k v_k^T (phase10_compaction_chain.py:148-165)",
        "stored_values_counted": f"{n_fac} x ({n_sel} + {n_sel}) + {n_fac} singular values = {eff_entries} float32 numbers",
        "bytes_counted": eff_entries * 4, "mb_in_report": cc["hard_sparse_16f_60g"]["mb_size"],
        "not_counted": {
            "indices_of_the_kept_entries": f"{n_fac * 2 * n_sel} indices = {n_fac * 2 * n_sel * 2} bytes as int16, "
                                           f"{n_fac * 2 * n_sel * 4} bytes as int32",
            "LET_head_on_154_d_features": "W 10x154 + b 154 + log_beta = 1,695 float32 = 6,780 bytes; saved file "
                                          f"artifacts/heads/compact/hard_sparse_16f_60g.pt = "
                                          f"{os.path.getsize(ART / 'heads/compact/hard_sparse_16f_60g.pt')} bytes",
            "feature_standardisation": "154 means + 154 SDs from the internal panel (1,232 bytes); computed at run "
                                       "time, not saved anywhere",
            "model": "a full MaxToki-217M forward pass through all 11 layers for every cell, mean over gene tokens, "
                     "then averaging cells into anchors",
        },
        "the_60_are_genes": False,
    },
    "ratios": {"full_drift_bytes_over_hard_sparse_bytes": A_bytes / (eff_entries * 4),
               "single_head_bytes_over_hard_sparse_bytes": A_top.nbytes / (eff_entries * 4),
               "reported_as": "2,400x 'from a 154-dimensional dense form' (paper) - the 2,352x is from the 2,464-d "
                              "pooled operator, the 154-d single-head operator gives 98x"},
    "quality_numbers": {k: {kk: cc[k][kk] for kk in ["trustworthiness", "random_holdout", "donor_holdout", "branch_holdout"]}
                        for k in cc},
    "evaluation_scope": "internal panel only, head re-fitted per variant, in-sample trust; no external or zero-shot "
                        "test of any compact operator; no interval; L10H6 was chosen on the same internal panel",
    "head_choice": "L10H6 = top of reports/head_layer_screen.csv by score = 0.5*trust + 0.25*random + 0.25*branch "
                   "(phase9_head_attribution.py:98-105), all on the internal panel; not an ablation or contribution test",
    "last_hidden_state_note": "in transformers 4.37.1 (the anaconda env that ran Phase 1B) the last hidden_states "
                              "entry is taken AFTER the final RMSNorm (modeling_llama.py:1087-1091)",
}

# ------------------------------------------------------------------ factor ablation (0.973 -> 0.123, 18.3%)
fa = json.loads((REP / "factor_ablation.json").read_text())
pe = pd.read_csv(REP / "factor_ablation_per_endpoint.csv")
U, S, Vt = np.linalg.svd(A_top.astype(np.float32), full_matrices=False)
order_sv = np.argsort(-S)
sv_rank = {int(k): int(np.where(order_sv == k)[0][0]) + 1 for k in fa["core_factors"]}
tot = pe["pooled_drop"].sum()
top4 = pe.sort_values("pooled_drop", ascending=False).head(4)
by_sv4 = pe[pe["factor"].isin([0, 1, 2, 3])]
branch_classes = sorted(set(metas["internal"]["hema_stage"].map(s2b).dropna()) - {"_unk"})
F["factor_ablation"] = {
    "operator": "rank-64 SVD truncation of A_L10H6",
    "probe_setup": "LET head fitted on the rank-64 features of the 290 internal anchors; five linear probes fitted on "
                   "the head output of the SAME anchors (800 Adam steps, no holdout); every ablation is scored on the "
                   "same 290 anchors with head and probes frozen (phase11_factor_ablation.py:150-193)",
    "pooled_drop_definition": "sum over 7 endpoint metrics (branch bal.acc, branch macro-F1, stage bal.acc, stage "
                              "macro-F1, CD4/CD8 AUROC, mono/macro AUROC, pseudotime Spearman) of max(0, intact - "
                              "ablated) when ONE factor is removed (phase11:229); docstring says 'sum of squared drops' "
                              "but the code sums plain drops",
    "four_strongest_axes": {"factors": fa["core_factors"],
                            "how_chosen": "top 4 by leave-one-factor-out pooled drop (not by singular value)",
                            "singular_values": {int(k): float(S[k]) for k in fa["core_factors"]},
                            "rank_by_singular_value": sv_rank},
    "share_18_3": {"value_json": fa["top4_pct_of_total_impact"],
                   "recomputed_from_csv": float(top4["pooled_drop"].sum() / tot),
                   "share_of_top4_by_singular_value_factors_0_1_2_3": float(by_sv4["pooled_drop"].sum() / tot),
                   "meaning": "sum of the single-factor pooled drops of the 4 highest-impact factors divided by the sum "
                              "over all 64 factors"},
    "share_66_2": "the source paper's figure for scGPT (pipeline spec manifold-discovery-extraction-compactification.md "
                  "lines 26 and 424); not recomputed in this project; same definition is assumed, not verified",
    "0.973_to_0.123": {"intact_branch_balanced_acc": fa["intact_metrics"]["branch_balanced_acc"],
                       "core_only_branch_balanced_acc": fa["core_metrics"]["branch_balanced_acc"],
                       "what": "branch BALANCED ACCURACY of the frozen linear branch probe when the operator keeps ONLY "
                               "the 4 core factors (the other 60 zeroed) vs all 64; in-sample; not branch-holdout",
                       "n_branch_classes": len(branch_classes), "branch_classes": branch_classes,
                       "chance_balanced_accuracy": 1.0 / len(branch_classes)},
}

# ------------------------------------------------------------------ orderings: which sweep, which gates, when
def mt(p):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(p)))

orows = []
qg = json.loads((REP / "quality_gates_let_anchor.json").read_text())
orows.append({"ordering": "H65", "where": "Phase 5 (pre-chosen from source paper, not a sweep winner)", "report": "quality_gates_let_anchor.json",
              "report_time": mt(REP / "quality_gates_let_anchor.json"), "ruler": "shortest path on 34-stage DAG",
              "gates_applied": "4 numeric gates (trust, random, donor, branch) + paired within-branch-shuffle null; permutation gate NOT computed",
              "internal_trust": qg["positive"]["trustworthiness"], "verdict_recorded": qg["verdict"]})
h38 = json.loads((REP / "h38_lite_quality_gates.json").read_text())
orows.append({"ordering": "H38_LITE", "where": "Phase 13 LITE", "report": "h38_lite_quality_gates.json",
              "report_time": mt(REP / "h38_lite_quality_gates.json"), "ruler": "Hamming on 7 hand-made signalling categories (not OmniPath)",
              "gates_applied": "4 gates (category holdout as 4th) + global-shuffle null", "internal_trust": h38["positive"]["trustworthiness"],
              "verdict_recorded": h38["verdict"]})
for fn, sweep, rule in [("hypothesis_registry.csv", "sweep 1 (Hamming)", "4 gates; category holdout NaN for all -> all INCONCLUSIVE"),
                        ("hypothesis_registry_sweep2.csv", "sweep 2 (ordinal)", "4 gates with 3-gate fallback BUILT IN (fallback does not require the null to fail)"),
                        ("hypothesis_registry_sweep3.csv", "sweep 3 (Hamming + ordinal)", "4 gates / 3-gate fallback / DIRECTIONAL tag; verdicts re-classified by hand after the run (STATUS.md:84)")]:
    reg = pd.read_csv(REP / fn)
    for r in reg.itertuples():
        orows.append({"ordering": r.name.split("_")[0], "full_name": r.name, "where": sweep, "report": fn, "report_time": mt(REP / fn),
                      "ruler": {"sweep 1 (Hamming)": "Hamming on cell-type categories",
                                "sweep 2 (ordinal)": "ordinal depth, L1 distance"}.get(sweep, str(getattr(r, "ruler_kind", ""))),
                      "gates_applied": rule, "internal_trust": r.positive_trust, "verdict_recorded": r.verdict,
                      "n_anchors": r.n_anchors_kept})
reg3 = pd.read_csv(REP / "hypothesis_registry_3gate.csv")
for r in reg3.itertuples():
    orows.append({"ordering": r.name.split("_")[0], "full_name": r.name, "where": "sweep 1 RE-EVALUATED with the new 3-gate fallback",
                  "report": "hypothesis_registry_3gate.csv", "report_time": mt(REP / "hypothesis_registry_3gate.csv"),
                  "ruler": "Hamming on categories", "gates_applied": "3-gate fallback (trust, random, donor; null must fail) added after sweep 1 "
                  f"(phase3a_revaluate_3gate.py written {mt(RUN / 'scripts/phase3a_revaluate_3gate.py')})",
                  "internal_trust": r.trust_pos, "verdict_recorded": f"{r.original_verdict} -> {r.revalued_verdict}", "n_anchors": r.n_anchors_kept})
p16 = json.loads((REP / "phase16_graph_dag_rescue.json").read_text())
orows.append({"ordering": "H107_graph", "where": "Phase 16 rescue of H107", "report": "phase16_graph_dag_rescue.json",
              "report_time": mt(REP / "phase16_graph_dag_rescue.json"), "ruler": "tree shortest path", "gates_applied": "4 gates + null",
              "internal_trust": p16["positive_trust"], "verdict_recorded": p16["verdict"]})
orow = pd.DataFrame(orows); orow.to_csv(OUT / "facts_orderings.csv", index=False)
F["orderings_summary"] = {
    "n_sweep_candidates": int(sum(1 for r in orows if r["where"].startswith("sweep ") and "RE-EVALUATED" not in r["where"])),
    "sweep_candidates": sorted({r["ordering"] for r in orows if r["where"].startswith("sweep ")}),
    "catalogue_of_141": "not found in the run; 141 is the topology pipeline's hypothesis count",
    "H115_H118": "no record in the run",
    "three_gate_fallback_timeline": {
        "sweep1_registry_written": mt(REP / "hypothesis_registry.csv"),
        "fallback_script_written": mt(RUN / "scripts/phase3a_revaluate_3gate.py"),
        "sweep1_reevaluated": mt(REP / "hypothesis_registry_3gate.csv"),
        "sweep2_script_written": mt(RUN / "scripts/phase3a_sweep2_ordinal.py"),
        "sweep2_registry_written": mt(REP / "hypothesis_registry_sweep2.csv"),
        "H103_external": mt(REP / "external_validation_h103.json"),
        "H103_zeroshot_no_script": mt(REP / "zeroshot_h103.json"),
        "H95_external_no_script": mt(REP / "external_validation_h95.json"),
        "sweep3_registry": mt(REP / "hypothesis_registry_sweep3.csv"),
        "sweep3_script_last_edit_after_registry": mt(RUN / "scripts/phase14_sweep3.py"),
        "H38_H95_backfill": mt(REP / "zeroshot_H38_lite.json"),
    },
    "H103_ruler_in_practice": None,
}
# H103: which depths actually occur
sys.path.insert(0, str(RUN / "scripts"))
from phase3a_sweep2_ordinal import H103_DEPTH  # read-only
dep_counts = {}
for p in ["internal", "external", "zeroshot"]:
    ct = metas[p]["cell_type"].astype(str).str.strip().str.lower()
    dd = ct.map(H103_DEPTH).dropna().astype(int)
    dep_counts[p] = {"depth_counts": {str(k): int(v) for k, v in dd.value_counts().sort_index().items()},
                     "cell_types": sorted(set(ct[ct.isin(H103_DEPTH.keys())]))}
F["orderings_summary"]["H103_ruler_in_practice"] = dep_counts
F["permutation_gate_in_run"] = "quality_gates_spec.json fixes blocked_permutation_p_max 0.001 with >= 2000 permutations, " \
                               "blocked by donor and tissue, two-sided; no script in scripts/ computes it"

(OUT / "v2_06_facts.json").write_text(json.dumps(F, indent=2, default=str))
write_run_config("v2_06_facts", CORE_INPUTS + [REP / "compaction_chain.json", REP / "factor_ablation.json",
                 REP / "factor_ablation_per_endpoint.csv", ART / "operators/layer10_head6.npy",
                 REP / "hypothesis_registry.csv", REP / "hypothesis_registry_3gate.csv", REP / "hypothesis_registry_sweep2.csv",
                 REP / "hypothesis_registry_sweep3.csv", REP / "h38_lite_quality_gates.json",
                 REP / "phase16_graph_dag_rescue.json"] + [RUN / f"outputs/phase1/cells_{p}_obs.csv" for p in cells], {})
print(json.dumps({k: F[k] for k in ["overlap", "ratios"] if k in F}, indent=1, default=str))
print(json.dumps(F["compaction"]["ratios"], indent=1)); print(json.dumps(F["factor_ablation"], indent=1, default=str))
print(json.dumps(F["orderings_summary"], indent=1, default=str))
print(coh[["panel", "n_anchors", "n_donors", "n_tissues", "n_cells_used_in_centroids", "top_donor_share", "anchors_at_progenitor_stages_HSC_MPP_CMP_CLP_GMP_MEP"]].to_string())
