import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
rng = np.random.default_rng(1)
def auroc_argsort(y, s, kind="quicksort"):
    r = np.argsort(np.argsort(s, kind=kind), kind=kind) + 1.0
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
n = 684
for nv in (2, 9, 22, 684):
    s = rng.integers(0, nv, n).astype(float) if nv < 684 else rng.normal(size=n)
    y = np.zeros(n, bool); y[-171:] = True     # positives appended after negatives
    print("1. %3d values | stable argsort-argsort %.4f | quicksort %.4f | sklearn %.4f"
          % (nv, auroc_argsort(y, s, "stable"), auroc_argsort(y, s), roc_auc_score(y, s)))
for nn, alpha in ((5, 1e3), (9, 1e3), (9, 1.0), (12, 1e3)):
    vals = []
    for rep in range(300):
        X = rng.normal(size=(nn, 50)); t = rng.normal(size=nn)
        P = np.zeros(nn)
        for tr, te in KFold(min(5, nn), shuffle=True, random_state=rep).split(X):
            P[te] = Ridge(alpha=alpha).fit(X[tr], t[tr]).predict(X[te])
        r = spearmanr(P, t).statistic
        vals.append(r)
    v = np.array(vals)
    print("2. n=%2d alpha=%g: mean signed rho %.3f, mean |rho| %.3f" % (nn, alpha, v.mean(), np.abs(v).mean()))
def circ_corr(a, b):
    am = np.angle(np.exp(1j * a).mean()); bm = np.angle(np.exp(1j * b).mean())
    sa, sb = np.sin(a - am), np.sin(b - bm)
    return (sa * sb).sum() / np.sqrt((sa ** 2).sum() * (sb ** 2).sum())
out = []
for rep in range(200):
    true = rng.uniform(0, 2*np.pi, 400)                       # near-uniform truth (R small)
    pred = np.angle(0.6*np.exp(1j*true) + 0.9)                # good but concentrated readout
    out.append((circ_corr(pred, true), abs(np.exp(1j*(pred-true)).mean()), abs(np.exp(1j*true).mean()), abs(np.exp(1j*pred).mean())))
o = np.array(out)
print("3. near-uniform truth, concentrated readout: circ_corr mean %.3f (min %.3f, share<0 %.2f), R_diff %.3f, R_true %.3f, R_pred %.3f"
      % (o[:,0].mean(), o[:,0].min(), (o[:,0] < 0).mean(), o[:,1].mean(), o[:,2].mean(), o[:,3].mean()))
