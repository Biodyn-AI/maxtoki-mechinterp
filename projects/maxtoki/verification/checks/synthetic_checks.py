"""Tiny synthetic checks that three documented bug mechanisms reproduce (seconds, CPU)."""
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
rng = np.random.default_rng(0)

# 1. argsort(argsort) AUROC on a 9-valued score with label-correlated ORDER (e.g. rows sorted by label)
def auroc_argsort(y, s):
    r = np.argsort(np.argsort(s)) + 1.0
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
n = 684
y = np.zeros(n, bool); y[:171] = True            # positives listed first, as a pair table often is
s = rng.integers(0, 9, n).astype(float)          # 9 distinct values, no signal
# stable sort breaks ties by position -> order correlates with label
print("1. tie AUROC: argsort-argsort %.4f vs sklearn %.4f" % (auroc_argsort(y, s), roc_auc_score(y, s)))
y2 = y[::-1].copy()
print("   (positives listed LAST) argsort-argsort %.4f vs sklearn %.4f" % (auroc_argsort(y2, s), roc_auc_score(y2, s)))

# 2. abs(Spearman) of KFold(min(5,n)) ridge predictions on pure noise at n=10
vals = []
for rep in range(200):
    nn = 10
    X = rng.normal(size=(nn, 50)); t = rng.normal(size=nn)
    P = np.zeros(nn)
    for tr, te in KFold(min(5, nn), shuffle=True, random_state=rep).split(X):
        P[te] = Ridge(alpha=10.0).fit(X[tr], t[tr]).predict(X[te])
    r = spearmanr(P, t).statistic
    vals.append((r, abs(r)))
v = np.array(vals)
print("2. noise probe n=10: mean signed rho %.3f, mean |rho| %.3f, share |rho|>0.5: %.2f" % (v[:,0].mean(), v[:,1].mean(), (v[:,1] > 0.5).mean()))

# 3. Jammalamadaka circ_corr when one variable is uniform-by-construction
def circ_corr(a, b):
    am = np.angle(np.exp(1j * a).mean()); bm = np.angle(np.exp(1j * b).mean())
    sa, sb = np.sin(a - am), np.sin(b - bm)
    return (sa * sb).sum() / np.sqrt((sa ** 2).sum() * (sb ** 2).sum())
true = np.repeat(np.arange(12) * 2 * np.pi / 12, 2)            # 12 bins x 2 cells, R ~ 1e-16
pred = true + rng.normal(0, 0.3, true.size)                    # a GOOD readout
print("3. uniform-grid truth: R_true=%.1e circ_corr=%.3f R_diff=%.3f" % (abs(np.exp(1j*true).mean()), circ_corr(pred, true), abs(np.exp(1j*(pred-true)).mean())))
pred_bad = rng.uniform(0, 2*np.pi, true.size)
print("   random readout:    circ_corr=%.3f R_diff=%.3f" % (circ_corr(pred_bad, true), abs(np.exp(1j*(pred_bad-true)).mean())))
