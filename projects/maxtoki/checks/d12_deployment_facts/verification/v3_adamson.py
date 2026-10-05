"""Independent re-check of the Adamson knock-down test (verifier's own code).

Three statistics per perturbed gene G (condition "G+ctrl"), all computed on
the expression of G itself, perturbed cells vs control cells:
  1. log2 ratio of means (same statistic as the original s4 script);
  2. Mann-Whitney AUC = P(value in a perturbed cell > value in a control cell),
     ties counted half (AUC < 0.5 means G is lower in its own cells);
  3. fraction of cells with G detected (> 0), perturbed minus control.
Intervals: percentile bootstrap over perturbed genes (the resampling unit),
10,000 resamples, seeds 42 and 2026; exact Clopper-Pearson for fractions.
Output: verification/out/v3_adamson.json
"""
import csv
import hashlib
import json
import os

import h5py
import numpy as np
import scipy.sparse as sp
from scipy.stats import beta, rankdata

MT = "<REPO_ROOT>/projects/maxtoki"
V = os.path.join(MT, "checks/d12_deployment_facts/verification")
P = "<DATA_ROOT>/biodyn-work/single_cell_mechinterp/data/perturb/adamson/perturb_processed_symbols.h5ad"
PROJ = MT + "/runs/attention-grn-217M/outputs/phase1_adamson/per_perturbation_auroc.csv"


def dec(a):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


with h5py.File(P, "r") as f:
    shape = tuple(int(x) for x in f["X"].attrs["shape"])
    X = sp.csr_matrix((f["X/data"][:], f["X/indices"][:], f["X/indptr"][:]), shape=shape)
    gn = f["var/gene_name"]
    genes = dec(gn[:]) if isinstance(gn, h5py.Dataset) else [dec(gn["categories"][:])[i] for i in gn["codes"][:]]
    cond_cats = dec(f["obs/condition/categories"][:])
    cond = np.array(cond_cats)[f["obs/condition/codes"][:]]
    x_max = float(X.data.max())
    x_is_integer = bool(np.all(np.equal(np.mod(X.data[:100000], 1), 0)))

col = {g: i for i, g in enumerate(genes)}
targets = sorted({c.split("+")[0] for c in cond_cats if c != "ctrl"})
double = [c for c in cond_cats if c != "ctrl" and not c.endswith("+ctrl")]
measured = [t for t in targets if t in col]
sub = X[:, [col[t] for t in measured]].toarray()
del X
is_ctrl = cond == "ctrl"

rows = []
for k, t in enumerate(measured):
    pert = cond == f"{t}+ctrl"
    a, c = sub[pert, k], sub[is_ctrl, k]
    lfc = float(np.log2((a.mean() + 1e-3) / (c.mean() + 1e-3)))
    r = rankdata(np.concatenate([a, c]))
    auc = float((r[: len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(c)))
    rows.append(dict(target=t, n_pert=int(pert.sum()), log2fc=lfc, auc=auc,
                     det_pert=float((a > 0).mean()), det_ctrl=float((c > 0).mean())))

lfc = np.array([r["log2fc"] for r in rows])
auc = np.array([r["auc"] for r in rows])
ddet = np.array([r["det_pert"] - r["det_ctrl"] for r in rows])


def cp(k, n):
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [round(lo, 4), round(hi, 4)]


boot = {}
for seed in (42, 2026):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(lfc), size=(10000, len(lfc)))
    boot[f"seed_{seed}"] = dict(
        median_log2fc_ci95=[round(float(x), 3) for x in np.percentile(np.median(lfc[idx], 1), [2.5, 97.5])],
        median_auc_ci95=[round(float(x), 3) for x in np.percentile(np.median(auc[idx], 1), [2.5, 97.5])])

proj = [r["pert_symbol"] for r in csv.DictReader(open(PROJ))]
out = dict(
    input_sha256=hashlib.sha256(open(P, "rb").read()).hexdigest(),
    project_csv_sha256=hashlib.sha256(open(PROJ, "rb").read()).hexdigest(),
    shape=shape, x_max=x_max, x_first_100k_values_all_integer=x_is_integer,
    n_conditions=len(cond_cats), n_perturbed_genes=len(targets), n_double_perturbations=len(double),
    n_control_cells=int(is_ctrl.sum()), n_measured=len(measured),
    not_measured=[t for t in targets if t not in col],
    log2fc=dict(n_down=int((lfc < 0).sum()), frac_down_cp95=cp(int((lfc < 0).sum()), len(lfc)),
                median=round(float(np.median(lfc)), 3), n_below_minus1=int((lfc < -1).sum())),
    auc=dict(n_below_half=int((auc < 0.5).sum()), frac_below_half_cp95=cp(int((auc < 0.5).sum()), len(auc)),
             median=round(float(np.median(auc)), 3)),
    detection=dict(n_lower_in_pert=int((ddet < 0).sum()), median_diff=round(float(np.median(ddet)), 3)),
    bootstrap=boot, resampling_unit="perturbed gene",
    n_project_perturbations=len(proj), n_project_perturbations_unique=len(set(proj)),
    project_subset_log2fc_all_negative=all(r["log2fc"] < 0 for r in rows if r["target"] in set(proj)),
    weakest_5=sorted(rows, key=lambda r: -r["log2fc"])[:5],
    expressed_targets_ctrl_detection_ge_5pct=dict(
        n=int(sum(r["det_ctrl"] >= 0.05 for r in rows)),
        n_down=int(sum(r["det_ctrl"] >= 0.05 and r["log2fc"] < 0 for r in rows)),
        n_auc_below_half=int(sum(r["det_ctrl"] >= 0.05 and r["auc"] < 0.5 for r in rows)),
        frac_down_cp95=cp(int(sum(r["det_ctrl"] >= 0.05 and r["log2fc"] < 0 for r in rows)),
                          int(sum(r["det_ctrl"] >= 0.05 for r in rows)))),
    per_target=rows,
)
json.dump(out, open(os.path.join(V, "out/v3_adamson.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("weakest_5", "per_target")}, indent=1))
for r in out["weakest_5"]:
    print(r)
