"""Independent re-read of the per-pipeline facts in the D12 ledger
(verifier's own code; reads run files directly, not out/pipeline_facts.json).

Checks: recorded mean sequence lengths; manifold token files; token dictionary
and HF vocab; topology strict max-null scope; dual-axis CV scope; spectral
autoloop calls, models and failures; topology Phase-11 outcome; H115/H118
search widened to the pinned reference repos under repos/.
Output: verification/out/v4_facts.json
"""
import csv
import glob
import json
import os
import re

import numpy as np

MT = "<REPO_ROOT>/projects/maxtoki"
REPO = "<REPO_ROOT>"
V = os.path.join(MT, "checks/d12_deployment_facts/verification")
R = MT + "/runs/"
j = lambda p: json.load(open(p))

out = {}
# ------------------------------------------------ recorded sequence lengths
out["mean_seq_len"] = {
    "spectral phase0": j(R + "spectral-geometry-217M/outputs/phase0/run_config.json").get("mean_seq_len"),
    "attn K562 217M": j(R + "attention-grn-217M/outputs/phase0/run_config.json").get("mean_seq_len"),
    "attn RPE1": j(R + "attention-grn-217M/outputs/phase0_rpe1/run_config.json").get("mean_seq_len"),
    "attn Adamson": j(R + "attention-grn-217M/outputs/phase0_adamson/run_config.json").get("mean_seq_len"),
    "attn K562 1B": j(R + "attention-grn-217M/outputs/phase0_k562_1b/run_config.json").get("mean_seq_len"),
    "sae 12-layer (positions/500)": int(re.search(r"total positions: ([\d,]+)", open(R + "sae-atlas-217M/outputs/full_12layer.log").read()).group(1).replace(",", "")) / 500,
}
lj = open(R + "longevity-mechinterp-217M/outputs/stage1_20260505/extraction_stats.json").read()
out["mean_seq_len"]["longevity stage1"] = float(re.search(r'"mean_seq_len":\s*([\d.]+)', lj).group(1))
out["max_len_in_run_configs"] = {os.path.relpath(p, MT): j(p).get("max_len") for p in
                                 glob.glob(R + "*/outputs/**/run_config.json", recursive=True)
                                 if "/._" not in p and "v2_" not in p and "max_len" in j(p)}
man = {}
for p in sorted(glob.glob(R + "manifold-discovery-217M/outputs/phase1/cells_*.npz")):
    if "/._" in p:
        continue
    z = np.load(p)
    L, T = z["seq_lens"], z["token_ids"]
    man[os.path.basename(p)] = dict(n=int(len(L)), mean=round(float(L.mean()), 1), frac_4096=round(float((L >= 4096).mean()), 4),
                                    max_id=int(T.max()))
out["manifold_npz"] = man
td = j(MT + "/setup/token_dictionary.json")
ng = {k: v for k, v in td.items() if not k.startswith("ENSG")}
out["token_dictionary"] = dict(n=len(td), n_non_gene=len(ng), min_non_gene_id_ge_20275=min(v for v in ng.values() if v >= 20275),
                               max_id=max(td.values()), hf_vocab=j(MT + "/setup/MaxToki-217M-HF/config.json")["vocab_size"],
                               n_numeric_keys=sum(1 for k in ng if re.fullmatch(r"-?\d+", k)))

# ------------------------------------------------ topology scope
rows = list(csv.DictReader(open(R + "topology-141-217M/outputs/phase12/h141_strict_margins.csv")))
out["strict_max_null"] = dict(n=len(rows), layers=sorted({r["layer"] for r in rows}, key=int),
                              n_pass=sum(r["passes_strict"] in ("1", "True", "true") for r in rows))
g = j(R + "topology-141-217M/outputs/groupkfold_h123/summary.json")
out["dual_axis"] = dict(scope=g["scope"], layers=len(g["by_layer"]), domains=sorted({r["domain"] for r in g["by_layer"]}),
                        pairs=sorted({r["n_test_pairs_dual"] for r in g["by_layer"]}),
                        auc=[min(r["auc_dual_axis_disjoint"] for r in g["by_layer"]), max(r["auc_dual_axis_disjoint"] for r in g["by_layer"])])
imm = j(R + "topology-141-217M/outputs/phase0/immune/run_config.json")
out["immune_layer_states"] = imm.get("n_layer_states")
out["hf_num_hidden_layers_217M"] = j(MT + "/setup/MaxToki-217M-HF/config.json").get("num_hidden_layers")

# ------------------------------------------------ spectral autoloop
dl = open(R + "spectral-geometry-217M/autoloop/runtime/driver.log").read().splitlines()
calls = [l for l in dl if " done " in l]
out["autoloop"] = dict(n_spawn=sum("spawning" in l for l in dl), n_model_sonnet=sum("--model sonnet" in l for l in dl),
                       other_model_flags=sorted({m for l in dl for m in re.findall(r"--model (\S+)", l)} - {"sonnet"}),
                       calls=[re.sub(r"^\[[^\]]+\]\s*", "", l)[:90] for l in calls], last=dl[-1][-45:],
                       api_errors={os.path.basename(p): [x for x in open(p).read().splitlines() if "API Error" in x]
                                   for p in sorted(glob.glob(R + "spectral-geometry-217M/autoloop/runtime/iter_*.log"))
                                   if "API Error" in open(p).read()})
a = j(R + "topology-141-217M/outputs/phase11_autoloop/autoloop_summary.json")
out["topology_phase11"] = {k: a[k] for k in ("n_iterations", "n_promote", "n_inconclusive", "n_retire")}


# ------------------------------------------------ H115 / H118, widened search
def hits(top, exts=(".md", ".py", ".json", ".csv", ".tex", ".txt", ".log", ".tsv")):
    found = []
    for dp, dn, fn in os.walk(top):
        dn[:] = [d for d in dn if d not in (".git", "node_modules", "__pycache__", ".venv") and d != "revision"]
        for f in fn:
            if f.startswith("._") or not f.endswith(exts):
                continue
            p = os.path.join(dp, f)
            try:
                if os.path.getsize(p) > 20e6:
                    continue
                t = open(p, errors="ignore").read()
            except OSError:
                continue
            if re.search(r"\bH11[58]\b", t):
                found.append(os.path.relpath(p, REPO))
    return found


h = {}
for sub in ["projects/maxtoki/runs", "projects/maxtoki/summaries", "projects/maxtoki/audits", "projects/maxtoki/setup",
            "pipelines", "repos", "references", "prompts"]:
    top = os.path.join(REPO, sub)
    if os.path.isdir(top):
        h[sub] = hits(top)
out["h115_h118_files"] = {k: v[:12] for k, v in h.items()}
out["h115_h118_counts"] = {k: len(v) for k, v in h.items()}
rm = REPO + "/repos/topology-biomechinterp2/iterations/iter_0045/brainstormer_hypothesis_roadmap.md"
if os.path.exists(rm):
    out["prior_repo_iter0045_lines"] = [l.strip()[:90] for l in open(rm).read().splitlines() if re.search(r"`H11[58]`", l)]

json.dump(out, open(os.path.join(V, "out/v4_facts.json"), "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str)[:7000])
