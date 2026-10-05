# Independent degree-preserving null (single Curveball chain, thinned), plus contrast nulls.
# Run from this folder: <python> null_curveball.py   (about 15 s)
from common import *
import random
tfs,M=edges(); K=len(tfs)
att=pairmat(tfs,lambda tf:A[tf]); attk=pairmat(tfs,lambda tf:A[:,tf]); sym=(att+attk)/2
prox=pairmat(tfs,lambda tf:-np.abs(mean[tf]-mean)); dc=np.diag(C); lift=pairmat(tfs,lambda tf:C[tf]*2000/(dc[tf]*dc))
scores={'att':att,'sym':sym,'prox':prox,'lift':lift}
mask=~np.isnan(att); n1=M.sum(); n0=mask.sum()-n1; deg=M.sum(1)
R={};Rrow={}
for k_,Sm in scores.items():
    r=np.zeros_like(Sm); r[mask]=rankdata(Sm[mask]); R[k_]=r
    rr=np.zeros_like(Sm)
    for i in range(K): rr[i,mask[i]]=rankdata(Sm[i,mask[i]])
    Rrow[k_]=rr
def stats(Mx):
    out={}
    for k_ in scores:
        out[k_+'_pooled']=(R[k_][Mx].sum()-n1*(n1+1)/2)/(n1*n0)
        out[k_+'_pertf']=np.mean([(Rrow[k_][i][Mx[i]].sum()-deg[i]*(deg[i]+1)/2)/(deg[i]*(N-1-deg[i])) for i in range(K)])
    return out
obs=stats(M); random.seed(99)
rows=[set(np.flatnonzero(M[i]).tolist()) for i in range(K)]; own=list(tfs)
def trade():
    a,b=random.sample(range(K),2); ra,rb=rows[a],rows[b]; ao=ra-rb; bo=rb-ra
    if not ao and not bo: return
    sa={x for x in ao if x==own[b]}; sb={x for x in bo if x==own[a]}
    pool=list((ao-sa)|(bo-sb)); random.shuffle(pool); na=len(ao)-len(sa)
    rows[a]=(ra&rb)|sa|set(pool[:na]); rows[b]=(ra&rb)|sb|set(pool[na:])
for _ in range(20000): trade()
draws=[]; jac=[]; col0=M.sum(0); ok=True; E0=set((i,j) for i in range(K) for j in np.flatnonzero(M[i]))
for d in range(1000):
    for _ in range(1000): trade()
    Mx=np.zeros_like(M)
    for i in range(K): Mx[i,list(rows[i])]=True
    ok&=bool((Mx.sum(1)==deg).all() and (Mx.sum(0)==col0).all() and not any(Mx[i,own[i]] for i in range(K)))
    E=set((i,j) for i in range(K) for j in rows[i]); jac.append(1-len(E&E0)/len(E|E0)); draws.append(stats(Mx))
print('margins+forbidden ok',ok,'jaccard min/mean',round(min(jac),3),round(np.mean(jac),3))
for k_ in obs:
    v=np.array([d[k_] for d in draws]); print(k_,'obs %.5f null %.5f sd %.5f z %.3f p %.4f'%(obs[k_],v.mean(),v.std(ddof=1),(obs[k_]-v.mean())/v.std(ddof=1),(1+(v>=obs[k_]).sum())/(1+len(v))))
rng=np.random.default_rng(5); r=R['att'][mask]
v=np.array([(r[rng.permutation(mask.sum())[:n1]].sum()-n1*(n1+1)/2)/(n1*n0) for _ in range(2000)])
print('label permutation null mean %.4f sd %.4f z %.3f'%(v.mean(),v.std(ddof=1),(obs['att_pooled']-v.mean())/v.std(ddof=1)))
v=[]
for _ in range(2000):
    Mx=np.zeros_like(M)
    for i in range(K): Mx[i,rng.choice(np.flatnonzero(mask[i]),deg[i],replace=False)]=True
    v.append((R['att'][Mx].sum()-n1*(n1+1)/2)/(n1*n0))
v=np.array(v); print('row-only null mean %.4f sd %.4f z %.3f'%(v.mean(),v.std(ddof=1),(obs['att_pooled']-v.mean())/v.std(ddof=1)))
