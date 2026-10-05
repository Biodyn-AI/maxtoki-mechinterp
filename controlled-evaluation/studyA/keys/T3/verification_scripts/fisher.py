import numpy as np
from lib import *
from scipy.stats import fisher_exact, spearmanr, rankdata
from statsmodels.stats.multitest import multipletests
s=np.load("sel.npz"); Q=s["Q"]; E=s["E"]; tfs=list(s["tfs"])
big=[t for t,v in CHIP.items() if v.sum()>=20]
chip20=[t for t in tfs if t in CHIP]; print("chip kd TFs", len(chip20), "sizes", {t:int(CHIP[t].sum()) for t in chip20})
OV={t:overlaps(CHIP[t]) for t in big}
for t in chip20:
    if t not in OV: OV[t]=overlaps(CHIP[t])
cuts=[0.5,0.25,0.1,0.05,0.02,0.01,0.0]
def fp(R,ov,thr):
    sel=np.zeros(NF,bool); sel[R]=True; hit=ov>=thr
    a=(sel&hit).sum(); b=(sel&~hit).sum(); c=(~sel&hit).sum(); d=(~sel&~hit).sum()
    return fisher_exact([[a,b],[c,d]],alternative="greater")[1]
passing=[]; allset=[]
for c in cuts:
    for thr in (1,2):
        ps={}; Rs={}
        for t in chip20:
            R=np.where((Q[tfs.index(t)]<0.05)&(np.abs(E[tfs.index(t)])>c))[0]; Rs[t]=R
            ps[t]=fp(R,OV[t],thr) if len(R)>0 else 1.0
        q=multipletests([ps[t] for t in chip20],method="fdr_bh")[1]
        for t,qq in zip(chip20,q):
            R=Rs[t]
            if len(R)==0: continue
            pall=np.array([fp(R,OV[u],thr) for u in big])
            rk=rankdata(pall)[big.index(t)]
            nother=int(sum(pall[j]<0.05 for j,u in enumerate(big) if u!=t))
            allset.append((t,c,thr,rk/len(big),rk))
            if qq<0.05: passing.append((t,c,thr,ps[t],qq,rk,nother))
print("passing cases", len(passing), sorted(set(p[0] for p in passing)))
for p in passing: print(p)
print("n other range", min(p[6] for p in passing), max(p[6] for p in passing), "rank range", min(p[5] for p in passing), max(p[5] for p in passing))
fr=np.array([a[3] for a in allset]); print("settings", len(allset), "mean rank frac", fr.mean(), "top5%", sum(a[4]<=0.05*291 for a in allset), [a for a in allset if a[4]<=0.05*291])
rng=np.random.default_rng(1); T=sorted(set(a[0] for a in allset)); bs=[]
for b in range(2000):
    pick=rng.choice(T,len(T)); bs.append(np.mean([a[3] for tt in pick for a in allset if a[0]==tt]))
print("CI over TFs", np.percentile(bs,[2.5,97.5]), "n TFs", len(T))
g=tfs.index("GATA1"); R=np.where((Q[g]<0.05)&(np.abs(E[g])>0.1))[0]
print("GATA1 0.1 >=2 p", fp(R,OV["GATA1"],2), ">=1 p", fp(R,OV["GATA1"],1))
# TRRUST Fisher
trtf=[t for t in tfs if t in TRR]; npass=0
OT={t:overlaps(TRR[t]) for t in trtf}
for c in cuts:
    for thr in (1,2):
        ps=[]
        for t in trtf:
            R=np.where((Q[tfs.index(t)]<0.05)&(np.abs(E[tfs.index(t)])>c))[0]
            ps.append(fp(R,OT[t],thr) if len(R)>0 else 1.0)
        npass+=(multipletests(ps,method="fdr_bh")[1]<0.05).sum()
print("TRRUST Fisher BH passes", npass)
# cut-off-free rank
nl=np.array([len(l) for l in lists]); keep=nl>=10
fr2=[]
for t in chip20:
    ae=np.abs(E[tfs.index(t)])[keep]
    rh={u:spearmanr(ae,OV[u][keep])[0] for u in big}
    vals=np.array([rh[u] for u in big]); vals=np.nan_to_num(vals,nan=-9)
    rk=rankdata(-vals)[big.index(t)]; fr2.append(rk/len(big))
fr2=np.array(fr2); bs=[rng.choice(fr2,len(fr2)).mean() for _ in range(2000)]
print("cutoff-free mean rank frac", fr2.mean(), np.percentile(bs,[2.5,97.5]), "top half", (fr2<=0.5).sum(), "top5%", (fr2<=0.05).sum())
