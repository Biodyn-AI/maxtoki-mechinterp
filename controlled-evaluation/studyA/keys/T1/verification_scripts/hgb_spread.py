# Boosted-tree incremental value across settings and seeds, folds by TF and by gene. About 2 min.
from common import *
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
tfs,M=edges(); K=len(tfs); att=pairmat(tfs,lambda tf:A[tf]); m=~np.isnan(att); y=M[m]
tfid=np.repeat(np.arange(K),N-1); cand=np.concatenate([np.flatnonzero(m[k]) for k in range(K)])
Xg=np.column_stack([mean[cand],var[cand],drop[cand]]); Xa=np.column_stack([Xg,att[m]])
for gname,groups in [('byTF',tfid),('byGene',cand)]:
    for sname,kw in [('default',{}),('lr.05_leaf15',dict(max_iter=100,learning_rate=0.05,max_leaf_nodes=15))]:
        res=[]
        for seed in [0,1,2]:
            o0=np.zeros(len(y)); o1=np.zeros(len(y))
            for tr,te in GroupKFold(5).split(Xg,y,groups):
                for X,o in [(Xg,o0),(Xa,o1)]: o[te]=HistGradientBoostingClassifier(class_weight='balanced',random_state=seed,**kw).fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
            res.append((auc(y,o0),auc(y,o1)))
        print(gname,sname,' '.join('g=%.3f +a=%.3f d=%+.3f'%(a,b,b-a) for a,b in res),flush=True)
