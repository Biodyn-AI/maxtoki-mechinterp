"""v2 cross-model alignment (revision item D10), step 4: collect all job outputs into
outputs/v2_crossmodel/summary.json and the figure source-data file outputs/v2_crossmodel/fig_crossmodel.csv.
Reads only files written by v2_crossmodel_prepare.py, v2_crossmodel_stats.py, v2_crossmodel_paired.py
and runs/spectral-geometry-217M/scripts/v2_crossmodel_pearson_ci.py. No computation beyond collection."""
from __future__ import annotations

import os
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import csv
import json

import v2_crossmodel_common as C

J = C.OUT / "jobs"
SPEC = C.PROJ / "runs/spectral-geometry-217M/outputs/v2_crossmodel/pearson_1500hvg_ci.json"
LABEL = {"gf": "MaxToki-217M vs Geneformer V2-316M",
         "scgpt_fixed": "MaxToki-217M vs scGPT whole-human (corrected gene lookup)",
         "scgpt_deployed": "MaxToki-217M vs scGPT whole-human (deployed lookup; wrong genes)"}
REPR = {"gf": "static input tables: MaxToki model.embed_tokens.weight vs Geneformer bert.embeddings.word_embeddings.weight",
        "scgpt_fixed": "static input tables: MaxToki model.embed_tokens.weight vs scGPT encoder.embedding.weight + encoder.enc_norm",
        "scgpt_deployed": "static input tables: MaxToki model.embed_tokens.weight vs scGPT encoder.embedding.weight + encoder.enc_norm (rows misassigned)"}
PANEL_KIND = "tissue/dataset gene panel (not a cell-type panel); static tables do not depend on cells, so the panel only sets the gene list"


def load(model, dom, part):
    f = J / f"{model}__{dom}__{part}.json"
    return json.loads(f.read_text()) if f.exists() else None


def main():
    prep = json.loads((C.OUT / "prepare_checks.json").read_text())
    paired = json.loads((C.OUT / "paired_differences.json").read_text())
    spec = json.loads(SPEC.read_text())
    summary = {"representation": {k: REPR[k] for k in REPR}, "panels": {}, "paired_gf_minus_scgpt": {},
               "panel_gene_overlap": prep["panel_gene_overlap"], "spectral_1500hvg": spec,
               "scgpt_lookup_bug": {d: {"n_genes": prep["domains"][d]["n_scgpt"],
                                        "rows_correct_under_deployed_lookup": prep["domains"][d]["scgpt_deployed_row_equals_fixed_row"],
                                        "examples": prep["domains"][d]["scgpt_deployed_examples_symbol_to_gene_actually_read"]}
                                    for d in C.DOMAINS},
               "scgpt_sanity_pairs": prep["scgpt_sanity_pairs"],
               "reproduction_of_deployed_max_abs_diff": max(v["max_abs_diff_vs_deployed"] for v in prep["reproduction_of_deployed"].values())}
    rows = []

    def add(model, panel, n, metric, definition, value, ci=None, ci_method="", chance_mean=None, chance_p95=None,
            chance_method="", deployed=None, source=""):
        rows.append({"comparison": LABEL.get(model, model), "model_key": model, "panel": panel,
                     "panel_kind": PANEL_KIND if panel in C.DOMAINS else "1,500 HVGs, Tabula Sapiens immune (spectral-geometry phase0)",
                     "representation": REPR.get(model, ""), "n_genes": n, "metric": metric, "metric_definition": definition,
                     "value": value, "ci_low": ci[0] if ci else "", "ci_high": ci[1] if ci else "", "ci_method": ci_method,
                     "chance_mean": "" if chance_mean is None else chance_mean,
                     "chance_p95": "" if chance_p95 is None else chance_p95, "chance_method": chance_method,
                     "excess_over_chance": "" if chance_mean is None or value is None else value - chance_mean,
                     "deployed_value": "" if deployed is None else deployed, "source_file": source})

    for model in ("gf", "scgpt_fixed", "scgpt_deployed"):
        for dom in C.DOMAINS:
            obs = load(model, dom, "obs"); perm = load(model, dom, "perm"); boot = load(model, dom, "boot")
            bg = load(model, dom, "bootgene"); oob = load(model, dom, "oob"); cv = load(model, dom, "cv")
            cvp = load(model, dom, "cvperm"); pc = load(model, dom, "permcheck")
            dep = prep["reproduction_of_deployed"].get(f"{dom}/{model}")
            n = obs["n_genes"]
            p = {"n_genes": n, "observed": obs, "chance_permutation": {k: perm[k] for k in
                 ("cca_null", "top1_null_refit", "top1_null_deployed_style_no_refit", "pearson_null")},
                 "n_perm": perm["n_perm"], "perm_seed": perm["seed"],
                 "permcheck_max_abs_diff_cca": pc["max_abs_diff"]}
            if boot:
                p["gene_bootstrap"] = {k: boot[k] for k in ("n_boot", "seed", "mean_unique_genes_per_resample",
                                                            "cca", "top1", "pearson", "cca_null", "top1_null")}
            if bg:
                p["gene_bootstrap_wholegene_chance"] = {k: bg[k] for k in ("cca_null_gene", "top1_null_gene", "cca_excess",
                                                                           "top1_excess", "resample_check_max_abs_diff")}
            if oob:
                p["oob_bootstrap_heldout"] = oob
            if cv:
                p["cv_heldout"] = {k: cv[k] for k in cv if k.startswith("cv_") or k.startswith("top1_")}
            if cvp:
                p["cv_heldout_chance"] = {k: cvp[k] for k in cvp if k.endswith("_null")}
                p["cv_heldout_chance"]["n_perm"] = cvp["n_perm"]
            summary["panels"][f"{model}/{dom}"] = p
            src = f"outputs/v2_crossmodel/jobs/{model}__{dom}__*.json"
            cn, tn, pn = perm["cca_null"], perm["top1_null_refit"], perm["pearson_null"]
            add(model, dom, n, "cca_insample_mean10",
                "deployed metric: mean of 10 in-sample canonical r, sklearn CCA on PCA-30 of each table",
                obs["cca_mean_r"], None,
                "no valid interval: a gene bootstrap with replacement inflates in-sample CCA (copies of genes); see report",
                cn["mean"], cn["p95"], f"{perm['n_perm']} permutations of gene correspondence, same n/dims/settings",
                dep["deployed_cca_mean_r"] if dep else None, src)
            add(model, dom, n, "pairwise_cosine_pearson",
                "Pearson of the two gene-gene cosine matrices (upper triangle), full static rows",
                obs["pairwise_pearson"], boot["pearson"]["ci95"] if boot else None,
                f"gene bootstrap with replacement, {boot['n_boot']} draws, copy pairs dropped, percentile" if boot else "",
                pn["mean"], pn["p95"], f"{perm['n_perm']} permutations of gene labels",
                dep["deployed_pairwise_pearson"] if dep else None, src)
            add(model, dom, n, "top1_insample",
                "deployed metric: Procrustes on PCA-30 fitted and scored on the same genes; candidates = all n genes",
                obs["top1"], None,
                "no valid interval: gene bootstrap inflates in-sample top-1 (copies); see report",
                tn["mean"], tn["p95"], f"{perm['n_perm']} permutations with the rotation REFITTED (deployed null did not refit: "
                                       f"mean {perm['top1_null_deployed_style_no_refit']['mean']:.4f})",
                dep["deployed_top1"] if dep else None, src)
            if cv:
                add(model, dom, n, "cca_heldout_mean10_5fold",
                    "mean of 10 canonical r measured on held-out genes; PCA/CCA fitted on the other 80%",
                    cv["cv_cca_mean_r"], cv["cv_cca_mean_r_range"], "min-max over 10 random 5-fold splits (spread, not a CI)",
                    cvp["cv_cca_mean_r_null"]["mean"] if cvp else None, cvp["cv_cca_mean_r_null"]["p95"] if cvp else None,
                    f"{cvp['n_perm']} permutations, same 5-fold procedure" if cvp else "", None, src)
                add(model, dom, n, "top1_heldout_5fold",
                    "held-out gene's nearest neighbour among all n genes after a rotation fitted on the other 80%",
                    cv["cv_top1"], cv["cv_top1_range"], "min-max over 10 random 5-fold splits (spread, not a CI)",
                    cvp["cv_top1_null"]["mean"] if cvp else None, cvp["cv_top1_null"]["p95"] if cvp else None,
                    f"{cvp['n_perm']} permutations, same 5-fold procedure (1/n = {1 / n:.4f})" if cvp else "", None, src)
            if oob:
                add(model, dom, n, "cca_heldout_mean10_oob",
                    "mean of 10 canonical r on out-of-bag genes; fitted on a gene bootstrap resample",
                    oob["oob_cca_mean_r"]["mean"], oob["oob_cca_mean_r"]["ci95"],
                    f"gene bootstrap with replacement, {oob['n_boot']} draws, out-of-bag scoring, percentile",
                    cvp["cv_cca_mean_r_null"]["mean"] if cvp else None, cvp["cv_cca_mean_r_null"]["p95"] if cvp else None,
                    "held-out chance from the 5-fold permutations", None, src)
                add(model, dom, n, "top1_heldout_oob",
                    "out-of-bag gene's nearest neighbour among all n genes; rotation fitted on a gene bootstrap resample",
                    oob["oob_top1"]["mean"], oob["oob_top1"]["ci95"],
                    f"gene bootstrap with replacement, {oob['n_boot']} draws, out-of-bag scoring, percentile",
                    cvp["cv_top1_null"]["mean"] if cvp else None, cvp["cv_top1_null"]["p95"] if cvp else None,
                    "held-out chance from the 5-fold permutations", None, src)

    for dom in C.DOMAINS:
        pd_ = paired[f"panel_{dom}"]
        summary["paired_gf_minus_scgpt"][dom] = pd_
        add("gf_minus_scgpt_fixed", dom, pd_["n_genes"], "pairwise_cosine_pearson_difference",
            "Geneformer minus scGPT (corrected), same genes, same resamples", pd_["pearson_gf_minus_scgpt"]["observed"],
            pd_["pearson_gf_minus_scgpt"]["ci95"], f"paired gene bootstrap, {pd_['pearson_gf_minus_scgpt']['n_boot']} draws, percentile",
            source="outputs/v2_crossmodel/paired_differences.json")
        add("gf_minus_scgpt_fixed", dom, pd_["n_genes"], "cca_heldout_oob_difference",
            "Geneformer minus scGPT (corrected), out-of-bag held-out canonical r, same resamples",
            pd_["heldout_cca_gf_minus_scgpt"]["mean"], pd_["heldout_cca_gf_minus_scgpt"]["ci95"],
            f"paired gene bootstrap, {pd_['heldout_cca_gf_minus_scgpt']['n_boot']} draws, out-of-bag, percentile",
            source="outputs/v2_crossmodel/paired_differences.json")
        add("gf_minus_scgpt_fixed", dom, pd_["n_genes"], "top1_heldout_oob_difference",
            "Geneformer minus scGPT (corrected), out-of-bag held-out top-1, same resamples",
            pd_["heldout_top1_gf_minus_scgpt"]["mean"], pd_["heldout_top1_gf_minus_scgpt"]["ci95"],
            f"paired gene bootstrap, {pd_['heldout_top1_gf_minus_scgpt']['n_boot']} draws, out-of-bag, percentile",
            source="outputs/v2_crossmodel/paired_differences.json")
    h = paired["hvg1500_pearson_gf_minus_scgpt"]
    summary["paired_gf_minus_scgpt"]["hvg1500"] = h

    # spectral 1,500-HVG rows
    ssrc = "runs/spectral-geometry-217M/outputs/v2_crossmodel/pearson_1500hvg_ci.json"
    B = spec["B_gene_bootstrap_with_replacement"]; ch = spec["chance_permutation"]
    add("gf", "hvg1500", spec["n_genes"], "pairwise_cosine_pearson",
        "Pearson of the two gene-gene cosine matrices (upper triangle); the paper's 0.382", spec["observed_pearson"],
        B["ci95"], f"gene bootstrap with replacement, {B['n_boot']} draws, copy pairs dropped, percentile",
        ch["mean"], ch["p95"], f"{ch['n_perm']} permutations of gene labels", 0.382, ssrc)
    for tag, key in (("scgpt_fixed", "fixed_lookup"), ("scgpt_deployed", "deployed_dict_position_lookup")):
        s = spec["scgpt_same_1500_hvg"][key]
        add(tag, "hvg1500", spec["scgpt_same_1500_hvg"]["n_genes"], "pairwise_cosine_pearson",
            "Pearson of the two gene-gene cosine matrices (upper triangle)", s["observed_pearson"], s["boot_ci95"],
            f"gene bootstrap with replacement, {s['n_boot']} draws, copy pairs dropped, percentile",
            s["chance_mean"], s["chance_p95"], f"{s['n_perm']} permutations of gene labels", None, ssrc)
    add("gf_minus_scgpt_fixed", "hvg1500", h["n_genes"], "pairwise_cosine_pearson_difference",
        "Geneformer minus scGPT (corrected), same genes, same resamples", h["observed_diff"], h["ci95"],
        f"paired gene bootstrap, {h['n_boot']} draws, percentile", source="outputs/v2_crossmodel/paired_differences.json")

    summary["source_verdict_wording"] = {
        "runs/topology-141-217M/outputs/phase14_cross_model/phase14_summary.json (verdict, all 3 panels)":
            "Layer-1 PARTIALLY REPLICATES (pearson z>3)",
        "runs/topology-141-217M/EXTENDED_FINDINGS.md:17 (H24-ext)": "POSITIVE ... Top-1 retrieval 38-47% (paper 72%, partial)",
        "runs/topology-141-217M/EXTENDED_FINDINGS.md:71 (L1 cross-model)":
            "PARTIALLY REPLICATES vs Geneformer (r=0.78, top-1 38-47%); FAILS vs scGPT (r=0.40, top-1 5-8%, pairwise Pearson ~0)",
        "runs/topology-141-217M/EXTENDED_FINDINGS.md:24 and FINAL_SUMMARY.md:453 (Phase 4 vs scGPT)":
            "DOES NOT GENERALIZE ACROSS ARCHITECTURAL FAMILIES",
        "runs/topology-141-217M/FINAL_SUMMARY.md:39,61": "Phase 4 cross-model CCA: SKIPPED (stale; written before the later Phase 4/14 additions)",
        "summaries/spectral-geometry-217M-FINAL_SUMMARY.md:251": "0.382 [0.380, 0.384] vs source paper 0.825: 'weaker'",
        "runs/spectral-geometry-217M/outputs/phase9b/summary.json": "alignment significantly above shuffled null",
    }
    C.write_json(C.OUT / "summary.json", summary)
    with open(C.OUT / "fig_crossmodel.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {len(rows)} rows to fig_crossmodel.csv")

    # complete run_config.json: scripts, seed rules, output hashes
    rc_f = C.OUT / "run_config.json"
    rc = json.loads(rc_f.read_text())
    rc["scripts"] = ["scripts/v2_crossmodel_common.py", "scripts/v2_crossmodel_prepare.py",
                     "scripts/v2_crossmodel_stats.py", "scripts/v2_crossmodel_paired.py",
                     "scripts/v2_crossmodel_summarize.py",
                     "../spectral-geometry-217M/scripts/v2_crossmodel_pearson_ci.py"]
    rc["seeds"] = {"stats_jobs": "20261001 + 1000*model_index(gf=0, scgpt_fixed=1, scgpt_deployed=2) + "
                                 "100*domain_index(lung=0, immune=1, external_lung=2) + part_index(perm=1, permcheck=2, "
                                 "boot=3, cv=4, cvperm=5, bootgene=6); oob reuses the boot resamples; exact seed stored in each job JSON",
                   "paired": "20261201 + domain_index; 1,500-HVG paired Pearson 20261211",
                   "pca_random_state": 42}
    rc["counts"] = {"permutations_chance": 1000, "gene_bootstrap": 1000, "heldout_cv_repeats": 10,
                    "heldout_cv_permutations": 200, "paired_pearson_bootstrap": 2000, "paired_oob_bootstrap": 500}
    rc["outputs"] = {}
    for name in ("summary.json", "fig_crossmodel.csv", "paired_differences.json", "prepare_checks.json",
                 "embeddings_subset.npz"):
        rc["outputs"][name] = C.sha256_file(C.OUT / name)
    rc["outputs"]["spectral pearson_1500hvg_ci.json"] = C.sha256_file(SPEC)
    C.write_json(rc_f, rc)


if __name__ == "__main__":
    main()
