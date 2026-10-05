import json, numpy as np, pandas as pd
from pathlib import Path
exec(open('reconstruct_original.py').read().split("res = {}")[0])  # reuse builders (read-only)
corr = build(np.ones(len(ed), bool))
m = pp.merge(corr[["source","target","frac_inhib","evidence"]], on=["source","target"])
obs = (m.actual_lfc < 0).to_numpy()
def sc(pred, mask=None):
    if mask is None: mask = np.ones(len(pred), bool)
    p, o = pred[mask], obs[mask]
    tp=(p&o).sum(); tn=(~p&~o).sum(); fp=(p&~o).sum(); fn=(~p&o).sum()
    return dict(n=int(mask.sum()), accuracy=float((tp+tn)/len(p)), always_dec=float(o.mean()), frac_pred_dec=float(p.mean()),
                balanced_acc=float(0.5*(tp/(tp+fn)+tn/(tn+fp))))
tie = (m.frac_inhib == 0.5).to_numpy()
out = dict(
  as_run_ties_predicted_increase=sc(m.predicted_inhibitory.to_numpy()),
  ties_predicted_decrease=sc((m.frac_inhib >= 0.5).to_numpy()),
  ties_dropped=sc(m.predicted_inhibitory.to_numpy(), ~tie),
  ties_only_obs_frac_dec=float(obs[tie].mean()), n_ties=int(tie.sum()),
  evidence_quantiles={str(q): float(v) for q, v in m.evidence.quantile([0.1,0.25,0.5,0.75,0.9]).items()},
  frac_inhib_quantiles={str(q): float(v) for q, v in m.frac_inhib.quantile([0.05,0.1,0.25,0.5]).items()},
)
Path('tie_sensitivity.json').write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2))
