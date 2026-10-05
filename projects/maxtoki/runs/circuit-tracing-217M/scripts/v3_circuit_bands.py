"""v3 circuit tracing, part 4b: follow-up of the |LFC| bands (item V3-3). CPU only. Exploratory.

The main CRISPRi run (v3_circuit_crispri.py) reports balanced accuracy and MCC inside 6 bands of |LFC|,
with 95% CIs by silenced gene that are not corrected for looking at 6 bands. With v3 edges and
single-log labels the two largest-change bands have CIs above 0. This script asks how robust that is:
  * Bonferroni CIs for 6 bands (percentiles 0.417 / 99.583) by silenced gene;
  * 95% CIs by connected component of silenced genes (shared source features);
  * the cross-fitted target-direction baseline and always-decrease inside each band (same records);
  * the same bands with the file LFC (sensitivity labels).
Bands use the |LFC| of the label being scored. Writes outputs/v3_circuit/crispri/bands_followup.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v2_circuit_crispri as C2  # noqa: E402

H = C2.H
V3 = C2.PROJ / "runs/circuit-tracing-217M/outputs/v3_circuit"
BANDS = [(0, 0.01), (0.01, 0.025), (0.025, 0.05), (0.05, 0.1), (0.1, 0.25), (0.25, np.inf)]
NB = 2000


def boot(sub, pred_col, group_col, seed=42):
    g = sub.assign(_p=sub[pred_col].astype(bool), _o=sub.obs_dec)
    C = pd.DataFrame({"tp": g._p & g._o, "fn": ~g._p & g._o, "fp": g._p & ~g._o, "tn": ~g._p & ~g._o,
                      "grp": g[group_col].to_numpy()}).groupby("grp").sum().to_numpy(float)
    rng = np.random.default_rng(seed)
    W = np.stack([np.bincount(rng.integers(0, len(C), len(C)), minlength=len(C)) for _ in range(NB)]).astype(float)
    m = C2.metrics_from_counts(*(W @ C).T)
    return np.asarray(m["balanced_acc"]) - 0.5, np.asarray(m["mcc"]), len(C)


def bands(pp, lfc_col, label):
    df = pp.copy()
    df["obs_dec"] = df[lfc_col] < 0
    out = []
    a = df[lfc_col].abs()
    for lo, hi in BANDS:
        sub = df[(a >= lo) & (a < hi)]
        mm = C2.scal(C2.metrics_from_counts(*C2.conf_counts(sub.predicted_inhibitory, sub.obs_dec)))
        tb = C2.scal(C2.metrics_from_counts(*C2.conf_counts(sub.target_baseline_pred_inhibitory, sub.obs_dec)))
        bg, mg, ng = boot(sub, "predicted_inhibitory", "source")
        bc, mc_, nc = boot(sub, "predicted_inhibitory", "source_component")
        q = lambda x, lo_, hi_: [float(np.nanpercentile(x, lo_)), float(np.nanpercentile(x, hi_))]
        out.append(dict(labels=label, abs_lfc_lo=lo, abs_lfc_hi=hi, n_pairs=int(len(sub)), n_silenced_genes=int(sub.source.nunique()),
                        n_components=nc, frac_obs_dec=mm["frac_obs_dec"], accuracy=mm["accuracy"], always_dec=mm["always_dec_acc"],
                        balanced_acc_minus_half=mm["balanced_acc"] - 0.5, mcc=mm["mcc"],
                        ci95_bal_genes=q(bg, 2.5, 97.5), ci_bonferroni6_bal_genes=q(bg, 100 * 0.025 / 6, 100 - 100 * 0.025 / 6),
                        ci95_bal_components=q(bc, 2.5, 97.5),
                        ci95_mcc_genes=q(mg, 2.5, 97.5), ci_bonferroni6_mcc_genes=q(mg, 100 * 0.025 / 6, 100 - 100 * 0.025 / 6),
                        ci95_mcc_components=q(mc_, 2.5, 97.5),
                        target_baseline_accuracy=tb["accuracy"], target_baseline_balanced_acc=tb["balanced_acc"], target_baseline_mcc=tb["mcc"]))
    return out


def main():
    t0 = time.time()
    pp = pd.read_parquet(V3 / "per_pair.parquet")
    res = dict(primary=bands(pp, "actual_lfc", "single-log panel LFC (primary)"),
               file_labels=bands(pp, "lfc_file", "file log1p(CP10k) LFC"),
               note="exploratory; bands were part of the v2 design, the follow-up CIs were added after seeing the v3 numbers; "
                    "the target-direction baseline is the cross-fitted one from the primary run (fit on the primary labels), "
                    "so for the file labels it is only approximate",
               script_sha256=H.sha256_file(__file__), seed=42, n_boot=NB, wall_seconds=None)
    res["wall_seconds"] = round(time.time() - t0, 1)
    H.write_json(V3 / "crispri/bands_followup.json", res)
    for r in res["primary"] + res["file_labels"]:
        print(r["labels"][:6], r["abs_lfc_lo"], r["n_pairs"], r["n_silenced_genes"], round(r["frac_obs_dec"], 3), round(r["accuracy"], 3),
              round(r["always_dec"], 3), "bal", round(r["balanced_acc_minus_half"], 4), [round(x, 4) for x in r["ci95_bal_genes"]],
              "bonf", [round(x, 4) for x in r["ci_bonferroni6_bal_genes"]], "comp", [round(x, 4) for x in r["ci95_bal_components"]],
              "mcc", round(r["mcc"], 4), "tb acc", round(r["target_baseline_accuracy"], 3), "tb bal", round(r["target_baseline_balanced_acc"], 3))


if __name__ == "__main__":
    main()
