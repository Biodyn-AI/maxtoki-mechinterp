"""D12 step 4: is the 'Adamson' dataset CRISPRi (knock-down) or CRISPRa (activation)?

The project labels it CRISPRa (setup/dataset_loader.py docstring; attention-grn
summary). The data file carries no modality field. Test from the data:
for each perturbed gene G (condition "G+ctrl") that is measured, compare the
mean expression of G in cells carrying that perturbation with control cells.
CRISPRi -> G goes down in its own perturbed cells. CRISPRa -> G goes up.

Null: the same statistic for 20 random measured non-target genes per
perturbation (same cells), to show what "no targeting" looks like.
Interval: bootstrap over perturbations (resampling unit = perturbed gene),
10,000 resamples, seed 42, percentile 95% CI.
Also: how many perturbations exist in the file and how many the project used
(rows of runs/attention-grn-217M/outputs/phase1_adamson/per_perturbation_auroc.csv).
Outputs: out/adamson_modality.json, out/adamson_target_knockdown.csv
"""
import csv
import json
import os
import sys

import h5py
import numpy as np
import scipy.sparse as sp
from scipy.stats import beta

sys.path.insert(0, os.path.dirname(__file__))
from common import MT, OUT, dump, sha256_file

P = "<DATA_ROOT>/biodyn-work/single_cell_mechinterp/data/perturb/adamson/perturb_processed_symbols.h5ad"
PROJ_CSV = MT + "/runs/attention-grn-217M/outputs/phase1_adamson/per_perturbation_auroc.csv"
SEED = 42
rng = np.random.default_rng(SEED)

with h5py.File(P, "r") as f:
    X = sp.csr_matrix((f["X/data"][:], f["X/indices"][:], f["X/indptr"][:]),
                      shape=tuple(f["X"].attrs["shape"]))
    genes = [g.decode() for g in f["var/gene_name"][:]] if not isinstance(f["var/gene_name"], h5py.Group) else \
        [f["var/gene_name/categories"][i].decode() for i in f["var/gene_name/codes"][:]]
    cats = [c.decode() for c in f["obs/condition/categories"][:]]
    codes = f["obs/condition/codes"][:]
    ct = f["obs/cell_type/categories"][:]
    cell_type = [c.decode() for c in ct]

Xc = X.tocsc()
gidx = {g: i for i, g in enumerate(genes)}
ctrl = np.where(codes == cats.index("ctrl"))[0]
ctrl_mean = np.asarray(Xc[ctrl].mean(axis=0)).ravel()
eps = 1e-3

rows, null_lfc = [], []
for ci, c in enumerate(cats):
    if c == "ctrl":
        continue
    tgt = c.replace("+ctrl", "")
    cells = np.where(codes == ci)[0]
    if tgt not in gidx:
        rows.append(dict(target=tgt, n_cells=len(cells), measured=False))
        continue
    j = gidx[tgt]
    pm = float(Xc[cells, j].mean())
    cm = float(ctrl_mean[j])
    lfc = float(np.log2((pm + eps) / (cm + eps)))
    # null: 20 random measured genes with control mean > 0.1 (expressed), same cells
    cand = np.where(ctrl_mean > 0.1)[0]
    cand = cand[cand != j]
    pick = rng.choice(cand, 20, replace=False)
    pmr = np.asarray(Xc[cells][:, pick].mean(axis=0)).ravel()
    nl = np.log2((pmr + eps) / (ctrl_mean[pick] + eps))
    null_lfc += nl.tolist()
    rows.append(dict(target=tgt, n_cells=len(cells), measured=True, mean_pert=pm, mean_ctrl=cm, log2fc=lfc,
                     null_median_log2fc=float(np.median(nl))))

meas = [r for r in rows if r["measured"]]
lfc = np.array([r["log2fc"] for r in meas])
down = (lfc < 0).astype(float)
boot = rng.choice(len(lfc), (10000, len(lfc)), replace=True)
frac_ci = np.percentile(down[boot].mean(1), [2.5, 97.5]).tolist()
med_ci = np.percentile(np.median(lfc[boot], axis=1), [2.5, 97.5]).tolist()
null_lfc = np.array(null_lfc)

proj = [r["pert_symbol"] for r in csv.DictReader(open(PROJ_CSV))]
file_targets = {r["target"] for r in rows}

with open(os.path.join(OUT, "adamson_target_knockdown.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["target", "n_cells", "measured", "mean_pert", "mean_ctrl", "log2fc", "null_median_log2fc"])
    w.writeheader()
    for r in rows:
        w.writerow(r)

res = dict(
    input=dict(path=P, sha256=sha256_file(P), project_csv=PROJ_CSV, project_csv_sha256=sha256_file(PROJ_CSV)),
    cell_type_field=cell_type,
    n_conditions_in_file=len(cats), n_perturbations_in_file=len(cats) - 1,
    n_control_cells=int(len(ctrl)), n_cells_total=int(X.shape[0]),
    n_targets_measured=len(meas),
    n_targets_down=int(down.sum()),
    frac_targets_down=float(down.mean()), frac_targets_down_ci95_bootstrap=frac_ci,
    frac_targets_down_ci95_clopper_pearson=[float(beta.ppf(0.025, down.sum(), len(down) - down.sum() + 1)) if down.sum() > 0 else 0.0,
                                            float(beta.ppf(0.975, down.sum() + 1, len(down) - down.sum())) if down.sum() < len(down) else 1.0],
    median_target_log2fc=float(np.median(lfc)), median_target_log2fc_ci95=med_ci,
    n_targets_log2fc_below_minus1=int((lfc < -1).sum()),
    null_random_genes=dict(n=int(len(null_lfc)), frac_down=float((null_lfc < 0).mean()),
                           median_log2fc=float(np.median(null_lfc))),
    interval_method="bootstrap over perturbed genes, 10,000 resamples, seed 42, percentile 95%; plus exact Clopper-Pearson 95% interval for the fraction (the bootstrap is degenerate at 85/85)",
    project_label="CRISPRa (setup/dataset_loader.py:96 docstring; summaries/attention-grn-217M-FINAL_SUMMARY.md run table)",
    n_perturbations_used_by_project=len(proj),
    project_perturbations_all_in_file=all(p in file_targets for p in proj),
)
res["verdict"] = ("knock-down (CRISPRi-like): the perturbed gene falls in its own cells"
                  if res["frac_targets_down"] > 0.8 else
                  "activation (CRISPRa-like)" if res["frac_targets_down"] < 0.2 else "unclear")
print(dump("adamson_modality.json", res))
print(json.dumps({k: v for k, v in res.items() if k != "input"}, indent=1))
