import numpy as np, pandas as pd
from lib import *
from scipy.stats import hypergeom, norm, rankdata
from statsmodels.stats.proportion import proportion_confint
s=np.load("sel.npz"); Q=s["Q"]; E=s["E"]; tfs=list(s["tfs"])
cb=count_bins(); gset=CHIP["GATA1"]; ov=overlaps(gset); g=tfs.index("GATA1")
resp=lambda i,c: np.where((Q[i]<0.05)&(np.abs(E[i])>c))[0]
# okd at 0.5
others=[t for t in tfs if t!="GATA1" and len(resp(tfs.index(t),0.5))>0]
b=sum(ov[resp(tfs.index(t),0.5)].max()>=3 for t in others)
print("okd others", others, "b", b, "p", (1+b)/(1+len(others)))
# power
rng=np.random.default_rng(99)
R5=resp(g,0.5); tgt=np.where(gset)[0]
for k in (2,4,8):
    hits=0
    for r in range(200):
        f=rng.choice(R5); lst=lists[f].copy()
        non=np.where(~gset[lst])[0]; pos=rng.choice(non,k,replace=False)
        cand=np.setdiff1d(tgt,lst); lst[pos]=rng.choice(cand,k,replace=False)
        ovs=[gset[lists[x]].sum() if x!=f else gset[lst].sum() for x in R5]
        T=Tcap(max(ovs))
        if T==0: continue
        pr=1.0
        for x in R5:
            d=dist_feature(lst if x==f else lists[x],gset,cb); pr*=d[:T].sum() if T<len(d) else 1.0
        hits+= (1-pr)<0.05
    print("power k",k,hits/200, proportion_confint(hits,200,method="wilson"))
# pooled union at 0.25
R=resp(g,0.25); U=np.unique(np.concatenate([lists[f] for f in R])); n=len(U)
big=[t for t,v in CHIP.items() if v.sum()>=20]
def naive(setv): return hypergeom.sf(setv[U].sum()-1,NG,setv.sum(),n)
def zcm(setv):
    obs=setv[U].sum(); mu=0; var=0
    for bb in np.unique(cb[U]):
        nb=(cb[U]==bb).sum(); Nb=(cb==bb).sum(); Tb=(setv&(cb==bb)).sum()
        mu+=nb*Tb/Nb; var+=hypergeom.var(Nb,Tb,nb)
    return (obs-mu)/np.sqrt(var), mu
print("union genes", n, "GATA1 targets", gset[U].sum(), "expected unif", n*gset.sum()/NG, "naive p", naive(gset))
print("other sets naive p<0.05", sum(naive(CHIP[t])<0.05 for t in big if t!="GATA1"))
zs={t:zcm(CHIP[t])[0] for t in big}; print("GATA1 z", zcm(gset), "rank", rankdata(-np.array(list(zs.values())))[big.index("GATA1")], "others z>1.96", sum(v>1.96 for t,v in zs.items() if t!="GATA1"))
# GATA1 set enriched in other knockdowns' features at 0.05
cnt=0; tot=0
for t in tfs:
    if t=="GATA1": continue
    Rr=resp(tfs.index(t),0.05)
    if len(Rr)>=5:
        tot+=1; UU=np.unique(np.concatenate([lists[f] for f in Rr]))
        cnt+= hypergeom.sf(gset[UU].sum()-1,NG,gset.sum(),len(UU))<0.05
print("other kd >=5 feats at 0.05", tot, "naive enriched for GATA1", cnt)
# coarse decile bins
dec=np.quantile(det,np.linspace(0,1,11)); cbd=np.clip(np.searchsorted(dec[1:-1],det,side="right"),0,9)
print("decile bin 0 range", det[cbd==0].min(), det[cbd==0].max())
d=dist_feature(lists[2610],gset,cbd); print("2610 E decile", (np.arange(len(d))*d).sum())
M=ov[R].max(); pr=1.0
for f in R:
    d=dist_feature(lists[f],gset,cbd); pr*=d[:M].sum()
print("GATA1 0.25 uncapped decile p", 1-pr, "M", M)
# median detection of responding features
fmed=np.array([np.median(det[l]) if len(l) else np.nan for l in lists])
print("GATA1 0.1 resp median of feature medians", np.median(fmed[resp(g,0.1)]), "all", np.nanmedian(fmed))
