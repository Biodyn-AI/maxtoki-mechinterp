import numpy as np, pandas as pd, time
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
P="<EVAL_ROOT>/studyA/tasks/T3-paper/data/"
z=np.load(P+"cell_feature_means.npz"); X=z["mean_gene"].astype(np.float64); nt=z["n_tokens"]
m=pd.read_csv(P+"cell_manifest.csv")
print("catalog tokens sum", nt[m.group.values=="catalog"].sum(), "minus bos/eos", nt[m.group.values=="catalog"].sum()-1000)
ref=X[m.group.values=="ref"]
tfs=sorted(m[m.group=="kd"].tf.unique())
P_=np.zeros((len(tfs),X.shape[1])); E=np.zeros_like(P_); Q=np.zeros_like(P_)
t=time.time()
for i,tf in enumerate(tfs):
    kd=X[(m.group.values=="kd")&(m.tf.values==tf)]
    p=mannwhitneyu(kd,ref,axis=0,alternative="two-sided").pvalue
    p=np.where(np.isnan(p),1.0,p)
    P_[i]=p; E[i]=kd.mean(0)-ref.mean(0); Q[i]=multipletests(p,method="fdr_bh")[1]
print("time", time.time()-t)
np.savez("sel.npz",P=P_,E=E,Q=Q,tfs=np.array(tfs))
print("median feature mean ref", np.median(ref.mean(0)))
