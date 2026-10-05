# Residualised attention (OLS float64 on 5 features; boosted trees on 6). About 30 s.
from common import *
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor
tfs,M=edges(); K=len(tfs); I,J=np.where(~np.eye(N,dtype=bool)); a=A[I,J]
def resid(X,model,seed=3):
    r=np.zeros(len(a))
    for tr,te in KFold(5,shuffle=True,random_state=seed).split(X):
        if model=='ols':
            b=np.linalg.lstsq(np.column_stack([np.ones(len(tr)),X[tr]]),a[tr],rcond=None)[0]; r[te]=a[te]-np.column_stack([np.ones(len(te)),X[te]])@b
        else: r[te]=a[te]-HistGradientBoostingRegressor(random_state=0).fit(X[tr],a[tr]).predict(X[te])
    Rm=np.full((N,N),np.nan); Rm[I,J]=r; return Rm
rng=np.random.default_rng(11)
def tfboot(Sm):
    rows=[(Sm[k][~np.isnan(Sm[k])],M[k][~np.isnan(Sm[k])]) for k in range(K)]; v=[]
    for b in range(2000):
        idx=rng.integers(0,K,K); v.append(auc(np.concatenate([rows[k][1] for k in idx]),np.concatenate([rows[k][0] for k in idx])))
    return np.percentile(v,[2.5,97.5]).round(4)
for nm,X,mod in [('ols5',np.column_stack([mean[I],var[I],mean[J],var[J],drop[J]]),'ols'),('hgb6',np.column_stack([mean[I],var[I],drop[I],mean[J],var[J],drop[J]]),'hgb')]:
    Sm=pairmat(tfs,lambda tf,R=resid(X,mod):R[tf]); print(nm,'resid pooled %.4f'%pooled(M,Sm),'CI',tfboot(Sm))
