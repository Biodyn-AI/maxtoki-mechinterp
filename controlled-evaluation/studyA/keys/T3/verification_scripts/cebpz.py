import numpy as np
from lib import *
from scipy.stats import fisher_exact, rankdata
s=np.load("sel.npz"); Q=s["Q"]; E=s["E"]; tfs=list(s["tfs"])
big=[t for t,v in CHIP.items() if v.sum()>=20]; OV={t:overlaps(CHIP[t]) for t in big}
def fp(R,ov,thr):
    sel=np.zeros(NF,bool); sel[R]=True; hit=ov>=thr
    return fisher_exact([[(sel&hit).sum(),(sel&~hit).sum()],[(~sel&hit).sum(),(~sel&~hit).sum()]],alternative="greater")[1]
i=tfs.index("CEBPZ")
for c in [0.02,0.01,0.0]:
    R=np.where((Q[i]<0.05)&(np.abs(E[i])>c))[0]
    for thr in (1,2):
        ps=np.array([fp(R,OV[u],thr) for u in big]); print(c,thr,len(R),"p",ps[big.index("CEBPZ")],"rank",rankdata(ps)[big.index("CEBPZ")])
