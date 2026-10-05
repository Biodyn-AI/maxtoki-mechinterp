# Pair, TF, candidate-gene and two-way (TF x gene) bootstraps; Hanley-McNeil. Needs oof_gene_model.npy from main.py. About 1 min.
from common import *
from scipy.stats import norm
tfs,M=edges(); K=len(tfs); att=pairmat(tfs,lambda tf:A[tf]); m=~np.isnan(att); y=M[m]; s=att[m]
tfid=np.repeat(np.arange(K),N-1); cand=np.concatenate([np.flatnonzero(m[k]) for k in range(K)]); oofg=np.load('oof_gene_model.npy')
P={k:np.unique(x,return_inverse=True) for k,x in [('att',s),('var',var[cand]),('gm',oofg)]}; yf=y.astype(float)
def wauc(key,w):
    u,inv=P[key]; L=len(u); wp=np.bincount(inv,weights=w*yf,minlength=L); wn=np.bincount(inv,weights=w*(1-yf),minlength=L)
    cn=np.cumsum(wn)-wn; return (wp*(cn+0.5*wn)).sum()/(wp.sum()*wn.sum())
rng=np.random.default_rng(31); out={'pair':[],'tf':[],'gene':[],'twoway':[]}
for b in range(2000):
    ws={'pair':rng.multinomial(len(y),np.full(len(y),1/len(y))).astype(float),'tf':np.bincount(rng.integers(0,K,K),minlength=K)[tfid].astype(float),'gene':np.bincount(rng.integers(0,N,N),minlength=N)[cand].astype(float)}
    ws['twoway']=ws['tf']*ws['gene']
    for k,w in ws.items():
        a=wauc('att',w); out[k].append((a,wauc('var',w)-a,wauc('gm',w)-a))
for k,v in out.items():
    v=np.array(v); print(k,'att',np.percentile(v[:,0],[2.5,97.5]).round(3),'var-att',np.percentile(v[:,1],[2.5,97.5]).round(3),'gm-att',np.percentile(v[:,2],[2.5,97.5]).round(3))
A_=auc(y,s); n1=y.sum(); n0=len(y)-n1; Q1=A_/(2-A_); Q2=2*A_**2/(1+A_)
se=np.sqrt((A_*(1-A_)+(n1-1)*(Q1-A_**2)+(n0-1)*(Q2-A_**2))/(n1*n0)); print('Hanley-McNeil se %.4f z %.2f p2 %.1e'%(se,(A_-.5)/se,2*(1-norm.cdf((A_-.5)/se))))
