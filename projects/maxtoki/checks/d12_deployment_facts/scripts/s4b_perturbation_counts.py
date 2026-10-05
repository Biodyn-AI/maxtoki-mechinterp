"""D12 step 4b: number of perturbed genes in each perturbation file and how many
the attention-grn runs evaluated (rows of per_perturbation_auroc.csv).
Reads obs category codes only (no expression matrix).
Outputs: out/perturbation_counts.json
"""
import csv
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import MT, dump

B = "<DATA_ROOT>"


def cats(g):
    return [c.decode() if isinstance(c, bytes) else c for c in g["categories"][:]]


res = {}
with h5py.File(B + "/biodyn-nmi-paper/src/02_cssi_method/crispri_validation/data/replogle_concat.h5ad", "r") as f:
    cl = cats(f["obs/cell_line"]); clc = f["obs/cell_line/codes"][:]
    gn = cats(f["obs/gene"]); gc = f["obs/gene/codes"][:]
    k = clc == cl.index("k562")
    labels = {gn[i] for i in np.unique(gc[k])}
    res["replogle_concat_k562"] = dict(n_cells=int(k.sum()), n_gene_labels=len(labels),
                                       n_perturbed_genes=len(labels - {"non-targeting"}),
                                       n_nontargeting_cells=int((gc[k] == gn.index("non-targeting")).sum()))
with h5py.File(B + "/biodyn-nmi-paper/results/round13_non_k562_replication/data/ReplogleWeissman2022_rpe1.h5ad", "r") as f:
    pc = cats(f["obs/perturbation"])
    res["replogle_rpe1"] = dict(n_cells=int(f["obs/perturbation/codes"].shape[0]), n_perturbation_labels=len(pc),
                                control_like_labels=[c for c in pc if "control" in c.lower() or "non" in c.lower()][:5])
for run, sub in [("k562_217M", "phase1"), ("rpe1", "phase1_rpe1"), ("adamson", "phase1_adamson"), ("k562_1b", "phase1_k562_1b")]:
    p = f"{MT}/runs/attention-grn-217M/outputs/{sub}/per_perturbation_auroc.csv"
    res[f"attention_grn_{run}_perturbations_evaluated"] = sum(1 for _ in csv.DictReader(open(p)))
print(dump("perturbation_counts.json", res))
print(res)
