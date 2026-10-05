"""How independent are the 248 silenced genes? Genes that sit in the top-10 list of the
same SAE source feature(s) inherit the same predictions. Group them and re-bootstrap."""
import json
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path("<REPO_ROOT>/projects/maxtoki")
ps = pd.read_csv("per_source_metrics.csv", index_col=0)
ed = pd.read_csv(REPO / "runs/circuit-tracing-217M/outputs/circuit_edges.csv", usecols=["src_layer", "src_feature"]).drop_duplicates()
top10 = {}
for li in range(12):
    for c in json.load(open(REPO / f"runs/sae-atlas-217M/outputs/phase2/layer_{li:02d}/feature_catalog.json")):
        top10[(li, c["feature_id"])] = [g.upper() for g in c["top20_genes"][:10]]
member = {s: [] for s in ps.index}
for sl, sf in zip(ed.src_layer, ed.src_feature):
    for g in top10.get((sl, sf), []):
        if g in member:
            member[g].append(f"L{sl}F{sf}")
memb = {s: tuple(sorted(set(v))) for s, v in member.items()}
nfeat = pd.Series({s: len(v) for s, v in memb.items()})
# connected components: genes linked if they share any source feature
parent = {s: s for s in memb}
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
byfeat = {}
for s, fs in memb.items():
    for f in fs:
        byfeat.setdefault(f, []).append(s)
for f, ss in byfeat.items():
    for s in ss[1:]:
        parent[find(s)] = find(ss[0])
comp = pd.Series({s: find(s) for s in memb})
exact = pd.Series({s: "|".join(v) for s, v in memb.items()})
ps["exact_group"] = exact; ps["component"] = comp
out = dict(n_sources=int(len(ps)), n_source_features_total=int(len(ed)),
           n_source_features_touching_the_248=int(len(byfeat)),
           source_features_per_gene_quantiles={str(q): float(v) for q, v in nfeat.quantile([0, .25, .5, .75, 1]).items()},
           n_distinct_exact_membership_sets=int(exact.nunique()),
           n_connected_components=int(comp.nunique()),
           largest_component_size=int(comp.value_counts().iloc[0]))
# cluster bootstrap over exact-membership groups
rng = np.random.default_rng(42)
C = ps[["tp", "fn", "fp", "tn"]].to_numpy(float)
groups = ps["exact_group"].to_numpy(); ug = np.unique(groups); gi = np.searchsorted(ug, groups)
Gc = np.zeros((len(ug), 4)); np.add.at(Gc, gi, C)
diffs = []; bas = []
for _ in range(2000):
    w = np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug)).astype(float)
    tp, fn, fp, tn = w @ Gc
    n = tp + fn + fp + tn
    diffs.append((tp + tn) / n - (tp + fn) / n); bas.append(0.5 * (tp / (tp + fn) + tn / (tn + fp)) - 0.5)
out["cluster_bootstrap_exact_groups"] = dict(
    n_groups=int(len(ug)),
    pooled_acc_minus_always_dec_ci95=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
    pooled_balanced_acc_minus_half_ci95=[float(np.percentile(bas, 2.5)), float(np.percentile(bas, 97.5))])
ps[["exact_group", "component"]].to_csv("source_groups.csv")
Path("source_clusters.json").write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2))
