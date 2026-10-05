"""Parse the two-level bootstrap logs (one JSON row per line, elapsed time appended) and summarise."""
import json, glob
import numpy as np
OUT = "<AGENT_TMP>/<SESSION_DIR>/d422dc33-6d6f-4b2b-bbf7-f274d8a82fae/scratchpad/understand/manifold_ci"
rows = []
for f in sorted(glob.glob(OUT + "/tl*.log")):
    for line in open(f):
        line = line.strip()
        if line.startswith("{"):
            rows.append(json.loads(line[: line.rindex("}") + 1]))
res = {"n_reps": len(rows),
       "n_reps_without_TSP2": int(sum(r["TSP2_copies"] == 0 for r in rows)),
       "train_anchor_range": [min(r["n_train_anchors"] for r in rows), max(r["n_train_anchors"] for r in rows)]}
for key in ["external_trust_trainonly", "external_branch_trainonly", "external_trust_both", "external_branch_both",
            "zeroshot_trust_trainonly", "zeroshot_branch_trainonly", "zeroshot_trust_both", "zeroshot_branch_both"]:
    v = np.array([r[key] for r in rows], float); v = v[~np.isnan(v)]
    thr = 0.80 if "trust" in key else 0.20
    res[key] = {"ci95_percentile": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                "mean": float(v.mean()), "sd": float(v.std(ddof=1)), "min": float(v.min()), "max": float(v.max()),
                f"frac_below_{thr}": float(np.mean(v < thr)), "n": int(len(v))}
print(json.dumps(res, indent=1))
open(OUT + "/06_twolevel_summary.json", "w").write(json.dumps(res, indent=2))
