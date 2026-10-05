import numpy as np
from lib import *
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests
s=np.load("sel.npz"); Q=s["Q"]; E=s["E"]; tfs=list(s["tfs"])
cb=count_bins()
cuts=[0.5,0.25,0.1,0.05,0.02,0.01,0.0]
chip20=[t for t in tfs if t in CHIP]
pas=set()
for c in cuts:
    ps=[]
    for t in chip20:
        R=np.where((Q[tfs.index(t)]<0.05)&(np.abs(E[tfs.index(t)])>c))[0]
        if len(R)==0: ps.append(1.0); continue
        U=np.unique(np.concatenate([lists[f] for f in R])); sv=CHIP[t]
        ps.append(hypergeom.sf(sv[U].sum()-1,NG,sv.sum(),len(U)))
    q=multipletests(ps,method="fdr_bh")[1]
    p_=[(t,c) for t,qq in zip(chip20,q) if qq<0.05]; pas|=set(t for t,_ in p_); print(c,p_)
print("pooled uniform BH passes TFs", sorted(pas))
# uncapped cm nominal p<0.05 anywhere; uncapped rf GATA1 0.25
for db,S in (("chip",CHIP),("trrust",TRR)):
    for t in [x for x in tfs if x in S]:
        ov=overlaps(S[t])
        for c in cuts:
            R=np.where((Q[tfs.index(t)]<0.05)&(np.abs(E[tfs.index(t)])>c))[0]
            if len(R)==0 or len(R)>600: continue
            M=int(ov[R].max())
            if M<1: continue
            pr=1.0
            for f in R:
                d=dist_feature(lists[f],S[t],cb); pr*=d[:M].sum() if M<len(d) else 1.0
            if 1-pr<0.05: print("uncapped cm p<0.05", db,t,c,len(R),M,1-pr)
g=tfs.index("GATA1"); ov=overlaps(CHIP["GATA1"]); R=np.where((Q[g]<0.05)&(np.abs(E[g])>0.25))[0]
print("GATA1 0.25 uncapped rf", p_rf(len(R),ov,8))
