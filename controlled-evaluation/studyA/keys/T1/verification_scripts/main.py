# Pair set, AUROCs of all scores, logistic gene model, TF bootstrap, analyst choices. About 2 min.
from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
tfs,M=edges()
print('nTF',len(tfs),'pos',M.sum(),'targets per TF',M.sum(1).min(),M.sum(1).max(),'distinct targets',M.any(0).sum())
att=pairmat(tfs,lambda tf:A[tf]); m=~np.isnan(att); print('neg',(~M[m]).sum())
attk=pairmat(tfs,lambda tf:A[:,tf]); sym=(att+attk)/2; mx=np.fmax(att,attk)
vs=pairmat(tfs,lambda tf:var); mn=pairmat(tfs,lambda tf:mean); det=pairmat(tfs,lambda tf:1-drop)
prox=pairmat(tfs,lambda tf:-np.abs(mean[tf]-mean)); dc=np.diag(C); lift=pairmat(tfs,lambda tf:C[tf]*2000/(dc[tf]*dc))
res={}
for nm,Sm in [('att_q',att),('att_k',attk),('sym',sym),('max',mx),('var',vs),('mean',mn),('detect',det),('prox',prox),('lift',lift)]:
    res[nm]=(pooled(M,Sm),per_tf(M,Sm).mean()); print(nm,'pooled %.5f  perTF %.5f'%res[nm])
y=M[m]; grp=np.repeat(np.arange(len(tfs)),N-1); X=np.column_stack([mn[m],vs[m],1-det[m]]); oof=np.zeros(len(y))
for tr,te in GroupKFold(5).split(X,y,grp):
    oof[te]=make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced',max_iter=1000)).fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
print('gene model pooled OOF',auc(y,oof)); np.save('oof_gene_model.npy',oof)
rng=np.random.default_rng(2024); rows={k:(att[k][m[k]],vs[k][m[k]],oof[grp==k],M[k][m[k]]) for k in range(len(tfs))}; ba=[];bv=[];bg=[]
for b in range(2000):
    idx=rng.integers(0,len(tfs),len(tfs)); yy=np.concatenate([rows[k][3] for k in idx])
    ba.append(auc(yy,np.concatenate([rows[k][0] for k in idx]))); bv.append(auc(yy,np.concatenate([rows[k][1] for k in idx]))); bg.append(auc(yy,np.concatenate([rows[k][2] for k in idx])))
ba,bv,bg=map(np.array,(ba,bv,bg)); q=lambda x:np.percentile(x,[2.5,97.5]).round(4)
print('att CI',q(ba),'var CI',q(bv),'gene model CI',q(bg),'var-att CI',q(bv-ba),'gm-att CI',q(bg-ba))
for mt in [1,2,3,4,5]:
    tf2,M2=edges(min_t=mt); print('min_t',mt,'nTF',len(tf2),'pos',M2.sum(),'auc %.4f'%pooled(M2,pairmat(tf2,lambda tf:A[tf])))
