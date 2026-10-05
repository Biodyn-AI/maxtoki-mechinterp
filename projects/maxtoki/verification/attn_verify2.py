import numpy as np, pandas as pd, json, importlib.util
from pathlib import Path
from sklearn.metrics import roc_auc_score
OUT = Path("<REPO_ROOT>/projects/maxtoki/runs/attention-grn-217M/outputs")
spec = importlib.util.spec_from_file_location("m", "attn_trrust_baselines.py")
RUNS = {"217M_K562": "", "217M_RPE1": "_rpe1", "217M_Adamson": "_adamson", "1B_K562": "_k562_1b"}
TRRUST = "<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv"
tr = pd.read_csv(TRRUST, sep="\t", header=None, names=["tf","target","mode","pmid"])
tr["tf"]=tr.tf.str.upper(); tr["target"]=tr.target.str.upper()
def curveball_permute(mat, n_iter, seed):
    rng = np.random.default_rng(seed)
    rows = [set(np.where(r)[0]) for r in mat.astype(bool)]
    G = len(rows)
    for _ in range(n_iter):
        i, j = rng.choice(G, 2, replace=False)
        a, b = rows[i], rows[j]
        inter = a & b; sym = (a | b) - inter
        if len(sym) < 2: continue
        sym_list = list(sym); rng.shuffle(sym_list)
        k = len(a) - len(inter)
        rows[i] = inter | set(sym_list[:k]); rows[j] = inter | set(sym_list[k:])
    out = np.zeros_like(mat)
    for r_idx, cols in enumerate(rows):
        for c in cols: out[r_idx, c] = 1
    return out
for name, sfx in RUNS.items():
    pp = pd.read_csv(OUT/f"phase1{sfx}"/"per_perturbation_auroc.csv")
    print(name, "per-pert: n_perts", len(pp), "sum_pos", int(pp.n_de_positive.sum()), "sum_neg", int(pp.n_de_negative.sum()),
          "median_pos", float(pp.n_de_positive.median()),
          "mean attn %.4f var %.4f gap %.4f" % (pp.auc_attention_primary.mean(), pp.auc_gene_variance.mean(),
                                                  pp.auc_gene_variance.mean()-pp.auc_attention_primary.mean()),
          "n_var>attn", int((pp.auc_gene_variance>pp.auc_attention_primary).sum()))
    gf = pd.read_csv(OUT/f"phase0{sfx}"/"gene_features.csv"); G=len(gf)
    sym=[s.upper() for s in gf.symbol]; s2i={s:i for i,s in enumerate(sym)}
    A = np.array(np.load(OUT/f"phase0{sfx}"/"attention_edges_layer_mean.npy", mmap_mode="r")[8], dtype=np.float32)
    np.fill_diagonal(A,0)
    E=np.zeros((G,G),np.int8)
    for a,b in zip(tr.tf,tr.target):
        i=s2i.get(a); j=s2i.get(b)
        if i is not None and j is not None and i!=j: E[i,j]=1
    tfm=np.zeros(G,bool)
    for t in set(tr.tf)&set(sym):
        if E[s2i[t]].sum()>=3: tfm[s2i[t]]=True
    rows=np.where(tfm)[0]
    def auc(L):
        s=[];l=[]
        for r in rows:
            m=np.ones(G,bool); m[r]=False; s.append(A[r][m]); l.append(L[r][m])
        return roc_auc_score(np.concatenate(l), np.concatenate(s))
    n_iter=5*int(E.sum()); vals=[]; unchanged=0
    for t in range(50):
        P=curveball_permute(E,n_iter,42+t); unchanged+=int((P==E).all()); vals.append(auc(P))
    obs=auc(E); vals=np.array(vals)
    print("   verbatim curveball 50: obs %.4f null mean %.4f std %.4f z %.2f ; trials with NO change: %d/50" % (obs, vals.mean(), vals.std(), (obs-vals.mean())/vals.std(), unchanged))
