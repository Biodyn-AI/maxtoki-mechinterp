"""Re-run the deployed Curveball null exactly as shipped, by executing the function source
taken from scripts/phase3_residualization.py (not a re-typed copy), with the deployed
seeds (42 + trial, 50 trials, n_iter = 5 x number of TRRUST edges in the gene set), and
count draws identical to the observed matrix (whole matrix and evaluated TF rows) and
self-loops created. Output: outputs/v2_eval/verification/deployed_null_check.json
"""
import ast, csv, json
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path("<REPO_ROOT>/projects/maxtoki/runs/attention-grn-217M/outputs")
SRC = OUT.parent / "scripts/phase3_residualization.py"
TRRUST = Path("<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv")
tree = ast.parse(SRC.read_text())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "curveball_permute")
ns = {"np": np}
exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SRC), "exec"), ns)
curveball_permute = ns["curveball_permute"]
pairs = [(r[0].upper(), r[1].upper()) for r in csv.reader(open(TRRUST), delimiter="\t") if len(r) >= 2]
res = {}
for run, sfx in {"217M_K562": "", "217M_RPE1": "_rpe1", "217M_Adamson": "_adamson", "1B_K562": "_k562_1b"}.items():
    gf = pd.read_csv(OUT / f"phase0{sfx}/gene_features.csv"); G = len(gf)
    idx = {s.upper(): i for i, s in enumerate(gf["symbol"])}
    E = np.zeros((G, G), dtype=np.int8)
    for a, b in pairs:
        if a in idx and b in idx and a != b:
            E[idx[a], idx[b]] = 1
    tfs = np.array(sorted({idx[a] for a, _ in pairs if a in idx and E[idx[a]].sum() >= 3}))
    whole = rows = loops = 0
    for t in range(50):
        P = curveball_permute(E, n_iter=5 * int(E.sum()), seed=42 + t)
        whole += int(np.array_equal(P, E)); rows += int(np.array_equal(P[tfs], E[tfs])); loops += int(np.trace(P))
    res[run] = {"n_edges": int(E.sum()), "n_rows_with_edges": int((E.sum(1) > 0).sum()),
                "identical_whole": whole, "identical_evaluated_rows": rows, "self_loops_total": loops}
    print(run, res[run], flush=True)
json.dump(res, open(OUT / "v2_eval/verification/deployed_null_check.json", "w"), indent=1)
