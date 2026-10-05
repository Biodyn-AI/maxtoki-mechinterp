"""Independent verification of the v2 knockdown-response endpoint (Endpoint A).

Written without importing v2_common. Labels come from the DEPLOYED Phase-3 pair file
(outputs/phase3*/full_pair_dataset.csv: pert_symbol, pert_hvg, target_hvg, is_de), which
the deployed code wrote with its own copy of the DE rule. Attention comes straight from
phase0*/attention_edges_layer_mean.npy[8] (diagonal zeroed), variance from
phase0*/gene_features.csv. AUROC = sklearn roc_auc_score per perturbation.

Intervals, two routes:
  route 1: percentile bootstrap over perturbations, 2,000 reps, seed 7 (different seed
           from v2), resampling perturbations with replacement;
  route 2: normal interval mean +/- 1.96 * SD / sqrt(n) over perturbations.
Also: paired Wilcoxon (zsplit) variance vs attention on the per-perturbation AUROCs.
Output: outputs/v2_eval/verification/endpointA_check.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

OUT = Path("<REPO_ROOT>/projects/maxtoki/runs/attention-grn-217M/outputs")
VER = OUT / "v2_eval" / "verification"
VER.mkdir(parents=True, exist_ok=True)
RUNS = {"217M_K562": "", "217M_RPE1": "_rpe1", "217M_Adamson": "_adamson", "1B_K562": "_k562_1b"}
SEED = 7
NB = 2000


def main():
    res = {}
    agent = json.load(open(OUT / "v2_eval/knockdown/knockdown_summary.json"))["runs"]
    for ri, (run, sfx) in enumerate(RUNS.items()):
        gf = pd.read_csv(OUT / f"phase0{sfx}/gene_features.csv")
        G = len(gf)
        A = np.array(np.load(OUT / f"phase0{sfx}/attention_edges_layer_mean.npy", mmap_mode="r")[8], dtype=np.float64)
        np.fill_diagonal(A, 0.0)
        var = gf["variance"].to_numpy(np.float64)
        fp = pd.read_csv(OUT / f"phase3{sfx}/full_pair_dataset.csv",
                         usecols=["pert_symbol", "pert_hvg", "target_hvg", "is_de"])
        rows = []
        for sym, d in fp.groupby("pert_symbol", sort=True):
            h = int(d["pert_hvg"].iloc[0])
            t = d["target_hvg"].to_numpy()
            y = d["is_de"].to_numpy().astype(int)
            assert h not in set(t.tolist()) and len(t) == G - 1
            rows.append({"pert": sym, "n_pos": int(y.sum()),
                         "att": roc_auc_score(y, A[h, t]), "var": roc_auc_score(y, var[t])})
        df = pd.DataFrame(rows)
        n = len(df)
        att, vv = df["att"].to_numpy(), df["var"].to_numpy()
        gap = vv - att
        rng = np.random.default_rng(SEED + ri)
        B = rng.integers(0, n, (NB, n))
        pct = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
        normal = lambda x: [float(x.mean() - 1.96 * x.std(ddof=1) / np.sqrt(n)),
                            float(x.mean() + 1.96 * x.std(ddof=1) / np.sqrt(n))]
        # compare with the agent's per-perturbation file
        ag = pd.read_csv(OUT / f"v2_eval/knockdown/per_perturbation_{run}.csv").set_index("pert_symbol")
        m = df.set_index("pert").join(ag, how="inner", rsuffix="_agent")
        res[run] = {
            "n_perturbations": n, "n_positive_pairs": int(df["n_pos"].sum()),
            "attention_mean": float(att.mean()), "variance_mean": float(vv.mean()), "gap": float(gap.mean()),
            "boot_ci_attention": pct(att[B].mean(1)), "boot_ci_variance": pct(vv[B].mean(1)),
            "boot_ci_gap": pct(gap[B].mean(1)),
            "normal_ci_gap": normal(gap),
            "wilcoxon_p_raw": float(stats.wilcoxon(vv, att, zero_method="zsplit").pvalue),
            "n_var_gt_att": int((vv > att).sum()),
            "max_abs_diff_vs_agent_per_pert_att": float((m["att"] - m["auc_attention"]).abs().max()),
            "max_abs_diff_vs_agent_per_pert_var": float((m["var"] - m["auc_variance"]).abs().max()),
            "n_matched_to_agent": int(len(m)),
            "agent_reported": {k: agent[run][k] for k in ["n_perturbations", "n_positive_pairs",
                                                          "attention_mean_auroc", "attention_ci95",
                                                          "variance_mean_auroc", "variance_ci95",
                                                          "gap_variance_minus_attention", "gap_ci95"]},
        }
        print(run, json.dumps(res[run], indent=None)[:900], flush=True)
    res["_method"] = {"labels": "deployed phase3 full_pair_dataset.csv is_de", "auroc": "sklearn roc_auc_score",
                      "bootstrap": "percentile, over perturbations, 2000 reps, seed 7+run_index",
                      "normal": "mean +/- 1.96 SD/sqrt(n) over perturbations"}
    json.dump(res, open(VER / "endpointA_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
