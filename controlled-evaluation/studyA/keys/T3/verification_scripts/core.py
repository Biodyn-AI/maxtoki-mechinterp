import numpy as np
from lib import *
from statsmodels.stats.multitest import multipletests
s=np.load("sel.npz"); Q=s["Q"]; E=s["E"]; tfs=list(s["tfs"])
cb=count_bins(); lt=len_tert(); cbl=cb*3+lt
print("count bin sizes", np.bincount(cb), "n det==500", (det==500).sum())
print("length tertile cuts check: quantiles", np.nanquantile(L,[1/3,2/3]))
g=tfs.index("GATA1")
R5=np.where((Q[g]<0.05)&(np.abs(E[g])>0.5))[0]
ov=overlaps(CHIP["GATA1"]); M=ov[R5].max(); print("GATA1 cut0.5 overlaps", dict(zip(R5.tolist(),ov[R5].tolist())))
T=Tcap(M)
print("p_rf", p_rf(len(R5),ov,T), "p_cm", p_max_ge(R5,CHIP["GATA1"],T,cb), "p_cml", p_max_ge(R5,CHIP["GATA1"],T,cbl))
ovt=overlaps(TRR["GATA1"]); print("TRRUST best", ovt[R5].max(), "n chip", CHIP["GATA1"].sum(), "n trr", TRR["GATA1"].sum())
# 2610
f=2610; l=lists[f]
print("2610 overlap", ov[f], "len list", len(l))
d=dist_feature(l,CHIP["GATA1"],cb); print("E cm", (np.arange(len(d))*d).sum(), "P>=8", d[8:].sum())
d2=dist_feature(l,CHIP["GATA1"],cbl); print("E cml", (np.arange(len(d2))*d2).sum(), "P>=8 cml", d2[8:].sum())
nset=CHIP["GATA1"].sum(); print("uniform E", 20*nset/NG, "hypergeom p", hypergeom.sf(7,NG,nset,20))
print("rare <10 in 2610", (det[l]<10).sum(), "universe share <10", (det<10).mean())
big=[t for t,v in CHIP.items() if v.sum()>=20]; print("chip sets >=20", len(big))
print("sets overlapping 2610 >=5", sum(CHIP[t][l].sum()>=5 for t in big))
gm=CHIP["GATA1"]; print("GATA1 chip median det", np.median(det[gm]), "universe", np.median(det), "median len", np.nanmedian(L[gm]), "universe", np.nanmedian(L))
