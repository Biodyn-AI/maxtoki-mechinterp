# TRRUST-endpoint incremental value: logreg gene-only vs gene+attn, GroupKFold by TF (mirrors phase2 settings)
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
OUT = Path("<REPO_ROOT>/projects/maxtoki/runs/attention-grn-217M/outputs")
RUNS = {"217M_K562": "", "217M_RPE1": "_rpe1", "217M_Adamson": "_adamson", "1B_K562": "_k562_1b"}
tr = pd.read_csv("<DATA_ROOT>/biodyn-nmi-paper/src/02_cssi_method/cssi_real_data/results/trrust_human.tsv", sep="\t", header=None, names=["tf","target","mode","pmid"])
tr["tf"]=tr.tf.str.upper(); tr["target"]=tr.target.str.upper()
for name,sfx in RUNS.items():
    gf=pd.read_csv(OUT/f"phase0{sfx}"/"gene_features.csv"); G=len(gf)
    sym=[s.upper() for s in gf.symbol]; s2i={s:i for i,s in enumerate(sym)}
    A=np.array(np.load(OUT/f"phase0{sfx}"/"attention_edges_layer_mean.npy",mmap_mode="r")[8],dtype=np.float32); np.fill_diagonal(A,0)
    E=np.zeros((G,G),np.int8)
    for a,b in zip(tr.tf,tr.target):
        i=s2i.get(a); j=s2i.get(b)
        if i is not None and j is not None and i!=j: E[i,j]=1
    rows=[s2i[t] for t in set(tr.tf)&set(sym) if E[s2i[t]].sum()>=3]
    X=[];y=[];g=[]
    for r in rows:
        m=np.ones(G,bool); m[r]=False; idx=np.where(m)[0]
        X.append(np.column_stack([gf.mean_expr.values[idx],gf.variance.values[idx],gf.dropout_rate.values[idx],A[r,idx]]))
        y.append(E[r,idx]); g.append(np.full(len(idx),r))
    X=np.vstack(X).astype(np.float32); y=np.concatenate(y); g=np.concatenate(g)
    res={}
    for fs,cols in [("gene_only",[0,1,2]),("gene_plus_attn",[0,1,2,3]),("attn_only",[3])]:
        aucs=[]
        for trn,te in GroupKFold(n_splits=5).split(X,y,groups=g):
            if y[trn].sum()==0 or y[te].sum()==0: continue
            mdl=Pipeline([("sc",StandardScaler()),("lr",LogisticRegression(max_iter=2000,class_weight="balanced"))]).fit(X[trn][:,cols],y[trn])
            aucs.append(roc_auc_score(y[te],mdl.predict_proba(X[te][:,cols])[:,1]))
        res[fs]=float(np.mean(aucs))
    print(name, {k:round(v,4) for k,v in res.items()}, "delta_attn=%.4f"%(res["gene_plus_attn"]-res["gene_only"]))
