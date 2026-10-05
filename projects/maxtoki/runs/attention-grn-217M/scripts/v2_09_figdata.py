"""v2 step 9 — source-data CSVs for a two-panel figure (knockdown endpoint | TRRUST endpoint).

Columns: run, method, auroc, ci_low, ci_high, n
  fig_knockdown.csv: auroc = mean per-perturbation AUROC; CI = percentile bootstrap over
                     perturbations (2,000 reps); n = number of perturbations.
  fig_trrust.csv:    auroc = pooled AUROC over all (TF, candidate) pairs; CI = percentile
                     bootstrap over TFs (2,000 reps); n = number of evaluated TFs.
                     Row 'degree_preserving_null_mean': auroc = mean of 1,000 Curveball null
                     draws, ci = 2.5th and 97.5th percentiles of those draws (not a CI).
                     Residualised OLS row = float64 fit (corrected in verification,
                     2026-10-01); the float32 replica of the deployed value is the row
                     'attention_L8_residualised_ols_float32_deployed_replica' (do not plot it).
All values are read from the v2 JSON outputs; nothing is typed by hand.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v2_common import RUNS, V2  # noqa: E402


def main():
    kd = json.load(open(V2 / "knockdown" / "knockdown_summary.json"))["runs"]
    rows = []
    for r in RUNS:
        k = kd[r]
        rows.append({"run": r, "method": "attention_L8", "auroc": k["attention_mean_auroc"],
                     "ci_low": k["attention_ci95"][0], "ci_high": k["attention_ci95"][1], "n": k["n_perturbations"]})
        rows.append({"run": r, "method": "gene_variance", "auroc": k["variance_mean_auroc"],
                     "ci_low": k["variance_ci95"][0], "ci_high": k["variance_ci95"][1], "n": k["n_perturbations"]})
    pd.DataFrame(rows).to_csv(V2 / "fig_knockdown.csv", index=False)

    tr = json.load(open(V2 / "trrust" / "trrust_summary.json"))
    cb = json.load(open(V2 / "curveball" / "curveball_summary.json"))
    rs = json.load(open(V2 / "residualised" / "residualised_summary.json"))
    rows = []
    for r in RUNS:
        t = tr[r]; n = t["n_tfs_evaluated"]
        for key, name in [("attention", "attention_L8"), ("variance", "gene_variance"),
                          ("gene3_logreg_oof", "gene_model_3feat")]:
            rows.append({"run": r, "method": name, "auroc": t["pooled_auroc"][key],
                         "ci_low": t["pooled_ci95"][key][0], "ci_high": t["pooled_ci95"][key][1], "n": n})
        # verification 2026-10-01: the OLS row uses the float64 fit (ols5_f64); the float32
        # replica of the deployed number is kept as a separate, clearly named row.
        for key, name in [("ols5_f64", "attention_L8_residualised_ols"), ("hgb6", "attention_L8_residualised_hgb"),
                          ("ols5", "attention_L8_residualised_ols_float32_deployed_replica")]:
            rows.append({"run": r, "method": name, "auroc": rs[r][key]["residualised_auroc"],
                         "ci_low": rs[r][key]["ci95"][0], "ci_high": rs[r][key]["ci95"][1], "n": n})
        c = cb[r]["curveball"]
        rows.append({"run": r, "method": "degree_preserving_null_mean", "auroc": c["null_mean"],
                     "ci_low": c["null_q025_q975"][0], "ci_high": c["null_q025_q975"][1], "n": n})
    pd.DataFrame(rows).to_csv(V2 / "fig_trrust.csv", index=False)
    print(pd.read_csv(V2 / "fig_knockdown.csv").round(4).to_string(index=False))
    print(pd.read_csv(V2 / "fig_trrust.csv").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
