import numpy as np
from lib import *
from scipy.stats import rankdata
s=np.load("sel.npz"); Q=s["Q"]; E=s["E"]; tfs=list(s["tfs"])
cb=count_bins()
big=[t for t,v in CHIP.items() if v.sum()>=20]
bigT=[t for t,v in TRR.items() if v.sum()>=5]; print("trrust sets >=5", len(bigT))
def pset(R,setv,cap=True):
    ov=np.array([setv[lists[f]].sum() for f in R]); M=int(ov.max()); T=Tcap(M) if cap else M
    if T<=0: return 1.0
    pr=1.0
    for f in R:
        d=dist_feature(lists[f],setv,cb); pr*=d[:T].sum() if T<len(d) else 1.0
    return 1-pr
def rank_of(tf,c,S,names):
    i=tfs.index(tf); R=np.where((Q[i]<0.05)&(np.abs(E[i])>c))[0]
    if len(R)>300: R=R  # all
    ps=np.array([pset(R,S[t]) for t in names])
    r=rankdata(ps)  # average ranks, 1 = smallest p
    return r[names.index(tf)], len(R), ps[names.index(tf)]
for c in [0.5,0.25,0.1,0.05]:
    print("GATA1 chip rank", c, rank_of("GATA1",c,CHIP,big))
print("GATA1 trrust rank 0.02", rank_of("GATA1",0.02,TRR,bigT))
print("CEBPZ chip rank 0.01", rank_of("CEBPZ",0.01,CHIP,big))
