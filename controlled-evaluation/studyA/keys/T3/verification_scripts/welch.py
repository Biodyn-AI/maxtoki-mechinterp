import numpy as np, pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from lib import TRR, P
z=np.load(P+"cell_feature_means.npz"); X=z["mean_gene"].astype(np.float64)
m=pd.read_csv(P+"cell_manifest.csv"); ref=X[m.group.values=="ref"]
tfs=sorted(m[m.group=="kd"].tf.unique())
print("TRRUST >=2", sum(1 for t in tfs if t in TRR and TRR[t].sum()>=2))
anyc=0
for t in tfs:
    kd=X[(m.group.values=="kd")&(m.tf.values==t)]
    p=ttest_ind(kd,ref,axis=0,equal_var=False).pvalue; p=np.where(np.isnan(p),1,p)
    q=multipletests(p,method="fdr_bh")[1]; anyc+=(q<0.05).any()
    if t=="GATA1": print("GATA1 welch 0.5", np.where((q<0.05)&(np.abs(kd.mean(0)-ref.mean(0))>0.5))[0].tolist())
print("welch any BH", anyc)
