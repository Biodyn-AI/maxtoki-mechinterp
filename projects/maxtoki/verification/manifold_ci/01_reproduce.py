"""Step 1: reproduce the published frozen-head numbers from saved artefacts, and time one head fit."""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, "<AGENT_TMP>/<SESSION_DIR>/d422dc33-6d6f-4b2b-bbf7-f274d8a82fae/scratchpad/understand/manifold_ci")
import json, time
import numpy as np, torch
from sklearn.manifold import trustworthiness
from common import *

A_e, A_m, A_l, part = load_operators()
c_int, d_int, m_int = load_panel("internal")
f_int_raw = build_pooled_drift(c_int, A_e, A_m, A_l, part)
mu = f_int_raw.mean(0); sd = f_int_raw.std(0) + 1e-6
f_int = (f_int_raw - mu) / sd

t0 = time.time()
head = train_let(f_int, d_int)
print(f"head fit time: {time.time()-t0:.2f}s")

# compare with saved Phase-5 weights
sd_saved = torch.load(ART / "heads/let_anchor_internal.pt", map_location="cpu")
for k, v in head.state_dict().items():
    print(f"  saved-vs-refit max|diff| {k}: {float((v - sd_saved[k]).abs().max()):.3e}")

out = {}
for panel, rep in [("external", "external_validation_external.json"),
                   ("zeroshot", "zeroshot_transfer_anchor_head.json"),
                   ("lung_nonhema", "external_validation_lung_nonhema.json")]:
    c, d, m = load_panel(panel)
    f = (build_pooled_drift(c, A_e, A_m, A_l, part) - mu) / sd
    with torch.no_grad():
        z = head(torch.from_numpy(f).float())[0].numpy()
    t_sk = trustworthiness(f, z, n_neighbors=15)
    origin = np.arange(len(d))
    t_my = trust_masked(euclid(f), euclid(z), origin, 15)
    Dh = arccos_dist(z)
    br, nb = group_spearman(Dh, d, branch_labels(m), origin)
    dn, nd = group_spearman(Dh, d, m["donor_id"].astype(str).to_numpy(), origin)
    pub = json.loads((REP / rep).read_text())
    out[panel] = dict(trust_sklearn=t_sk, trust_custom=t_my, branch=br, n_branches=nb, donor=dn, n_donor_groups=nd,
                      published_trust=pub["trustworthiness"], published_branch=pub["branch_holdout"],
                      published_donor=pub["donor_holdout"])
    print(panel, json.dumps(out[panel], indent=1))
    np.save(OUT / f"z_{panel}_frozen.npy", z)
(OUT / "01_reproduce.json").write_text(json.dumps(out, indent=2))
