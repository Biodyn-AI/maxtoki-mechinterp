import numpy as np, pandas as pd
from scipy import stats
PKG = "<EVAL_ROOT>/studyA/tasks/T2-paper/data"
P = pd.read_parquet(f"{PKG}/gene_pairs.parquet"); Z = np.load(f"{PKG}/knockdown_lfc.npz")
sil=list(Z["silenced_genes"]); meas=list(Z["measured_genes"]); L=Z["lfc"].astype(float)
r=P.silenced_gene.map({s:i for i,s in enumerate(sil)}).values; c=P.target_gene.map({m:i for i,m in enumerate(meas)}).values
D=(L<0).astype(float); usual=((D.sum(0)[c]-D[r,c])/(len(sil)-1))>0.5
y=P.lfc.values<0; p=P.predicted_decrease.values
def cmh(pred,obs,strata,corr):
    A=EA=VA=num=den=0.0; used=0; kept=0
    for k in np.unique(strata):
        m=strata==k; n=m.sum()
        if n<2: continue
        a=np.sum(pred[m]&obs[m]); b=np.sum(pred[m]&~obs[m]); cc=np.sum(~pred[m]&obs[m]); d=np.sum(~pred[m]&~obs[m])
        v=(a+b)*(cc+d)*(a+cc)*(b+d)/(n*n*(n-1))
        if v>0: used+=1; kept+=n
        num+=a*d/n; den+=b*cc/n; A+=a; EA+=(a+b)*(a+cc)/n; VA+=v
    st=((abs(A-EA)-(0.5 if corr else 0))**2)/VA
    return round(num/den,3), round(float(stats.chi2.sf(st,1)),4), used, kept
g=P.silenced_gene.astype('category').cat.codes.values
for th in [0.45,0.5]:
    s=np.abs(P.lfc.values)>=th
    print(th, int(s.sum()), "by gene corr/uncorr:", cmh(p[s],y[s],g[s],True), cmh(p[s],y[s],g[s],False))
    print(th, "by gene x usual corr/uncorr:", cmh(p[s],y[s],(g*2+usual)[s],True), cmh(p[s],y[s],(g*2+usual)[s],False))
    # logistic check: obs ~ pred + usual, cluster-robust by silenced gene
    import statsmodels.api as sm
    X=sm.add_constant(np.column_stack([p[s].astype(float),usual[s].astype(float)]))
    fit=sm.Logit(y[s].astype(float),X).fit(disp=0,cov_type='cluster',cov_kwds={'groups':g[s]})
    print(th, "logit obs~pred+usual, cluster by gene: coef pred %.3f se %.3f p %.3f; coef usual %.3f p %.4f"%(fit.params[1],fit.bse[1],fit.pvalues[1],fit.params[2],fit.pvalues[2]))
