import sys, time
from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
tfs,M=edges(); K=len(tfs)
att=pairmat(tfs,lambda tf:A[tf]); m=~np.isnan(att); y=M[m]
tfid=np.repeat(np.arange(K),N-1)
cand=np.concatenate([np.flatnonzero(m[k]) for k in range(K)])
Xg=np.column_stack([mean[cand],var[cand],drop[cand]]); Xa=np.column_stack([Xg,att[m]])
def oof(X,groups,model):
    o=np.zeros(len(y))
    for tr,te in GroupKFold(5).split(X,y,groups):
        if model=='lr': c=make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced',max_iter=2000))
        else: c=HistGradientBoostingClassifier(random_state=0,class_weight='balanced')
        c.fit(X[tr],y[tr]); o[te]=c.predict_proba(X[te])[:,1]
    return o
rng=np.random.default_rng(13)
rows=[np.flatnonzero(tfid==k) for k in range(K)]
def fast_auc_boot(o1,o0,B=1000):
    v=[]
    for b in range(B):
        idx=np.concatenate([rows[k] for k in rng.integers(0,K,K)]); v.append(auc(y[idx],o1[idx])-auc(y[idx],o0[idx]))
    return np.percentile(v,[2.5,97.5]).round(4)
mods=sys.argv[1].split(',')
for grp_name,groups in [('byTF',tfid),('byGene',cand)]:
    for mod in mods:
        t0=time.time(); o0=oof(Xg,groups,mod); o1=oof(Xa,groups,mod)
        print(grp_name,mod,'gene-only %.4f  +att %.4f  delta %.4f'%(auc(y,o0),auc(y,o1),auc(y,o1)-auc(y,o0)),'CI',fast_auc_boot(o1,o0),'%.0fs'%(time.time()-t0),flush=True)
