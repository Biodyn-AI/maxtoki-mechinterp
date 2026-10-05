import numpy as np, json
from lib import *
from statsmodels.stats.multitest import multipletests
from scipy.stats import rankdata
s=np.load("sel.npz"); Q=s["Q"]; E=s["E"]; tfs=list(s["tfs"])
cb=count_bins(); cbl=cb*3+len_tert()
cuts=[0.5,0.25,0.1,0.05,0.02,0.01,0.0]
# cache per-feature overlap distributions lazily
cache={}
def dist(f,tf,db,bins,key):
    k=(f,tf,db,key)
    if k not in cache: cache[k]=dist_feature(lists[f],(CHIP if db=="chip" else TRR)[tf],bins)
    return cache[k]
def pmax(feats,tf,db,m,bins,key):
    if m<=0: return 1.0
    pr=1.0
    for f in feats:
        d=dist(f,tf,db,bins,key); pr*=d[:m].sum() if m<len(d) else 1.0
    return 1-pr
ovc={}
def ovl(tf,db):
    if (tf,db) not in ovc: ovc[(tf,db)]=overlaps((CHIP if db=="chip" else TRR)[tf])
    return ovc[(tf,db)]
rows=[]
nsig=0
for db,S in (("chip",CHIP),("trrust",TRR)):
    tl=[t for t in tfs if t in S]
    for c in cuts:
        res={}
        for t in tl:
            i=tfs.index(t); R=np.where((Q[i]<0.05)&(np.abs(E[i])>c))[0]
            if len(R)==0: res[t]=dict(K=0,M=0,prf=1,pcm=1,pcml=1,pcmu=1); continue
            ov=ovl(t,db); M=int(ov[R].max()); T=Tcap(M)
            res[t]=dict(K=len(R),M=M,prf=p_rf(len(R),ov,T),pcm=pmax(R,t,db,T,cb,"cm"),pcml=pmax(R,t,db,T,cbl,"cml"),pcmu=pmax(R,t,db,M,cb,"cm") if M>=1 else 1.0)
        for nm in ("pcm","pcml","pcmu"):
            q=multipletests([res[t][nm] for t in tl],method="fdr_bh")[1]
            nsig+=(q<0.05).sum()
            for t,qq in zip(tl,q): res[t]["q_"+nm]=qq
        for t in tl:
            r=res[t]; 
            if r["pcm"]<0.05 or (db=="chip" and t=="GATA1") or (t=="CEBPZ" and db=="chip"):
                rows.append((db,c,t,r["K"],r["M"],round(r["prf"],4),round(r["pcm"],4),round(r["q_pcm"],3),round(r["pcml"],4),round(r["pcmu"],4)))
print("n BH-significant settings under cm/cml/cm-uncapped:", nsig)
for r in rows: print(r)
