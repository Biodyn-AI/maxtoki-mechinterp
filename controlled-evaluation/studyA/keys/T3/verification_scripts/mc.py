import numpy as np, pandas as pd, time
from lib import *
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
rng=np.random.default_rng(777)
z=np.load(P+"cell_feature_means.npz"); X=z["mean_gene"].astype(np.float64)
m=pd.read_csv(P+"cell_manifest.csv")
ref=X[m.group.values=="ref"]; pool=X[m.group.values=="pool"]
gk=X[(m.group.values=="kd")&(m.tf.values=="GATA1")]
# bootstrap 2610
e=[]; 
for b in range(1000):
    a=gk[rng.integers(0,len(gk),len(gk)),2610].mean()-ref[rng.integers(0,len(ref),len(ref)),2610].mean(); e.append(a)
e=np.array(e); print("2610 effect", gk[:,2610].mean()-ref[:,2610].mean(), "CI", np.percentile(e,[2.5,97.5]), "frac>0.5", (np.abs(e)>0.5).mean())
# random groups
cb=count_bins(); gset=CHIP["GATA1"]; ov=overlaps(gset)
refm=ref.mean(0)
anyresp={c:0 for c in [0.5,0.25,0.1,0.05,0.0]}; bK=0; B=300
t=time.time()
for b in range(B):
    idx=rng.choice(len(pool),95,replace=False); grp=pool[idx]
    p=mannwhitneyu(grp,ref,axis=0).pvalue; p=np.where(np.isnan(p),1,p); q=multipletests(p,method="fdr_bh")[1]
    eff=grp.mean(0)-refm
    for c in anyresp: anyresp[c]+=int(((q<0.05)&(np.abs(eff)>c)).any())
    top5=np.argsort(-np.abs(eff))[:5]
    bK+=int(ov[top5].max()>=3)
print("time",time.time()-t)
print("random groups any responding", {c:v/B for c,v in anyresp.items()}, "K-matched p", (1+bK)/(1+B))
