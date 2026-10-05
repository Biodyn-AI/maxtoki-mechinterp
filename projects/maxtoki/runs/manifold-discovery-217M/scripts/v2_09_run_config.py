"""v2 step 9: one combined run_config.json for all v2 work (inputs with sha256, seeds, script hashes, outputs)."""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, "<REPO_ROOT>/projects/maxtoki/runs/manifold-discovery-217M/scripts")
import json, time
from v2_common import *

parts = {p.stem.replace("run_config_", ""): json.loads(p.read_text()) for p in sorted(OUT.glob("run_config_*.json"))}
inputs = {}
for v in parts.values():
    inputs.update(v.get("inputs", {}))
scripts = sorted((RUN / "scripts").glob("v2_*.py"))
outputs = sorted([p for p in OUT.rglob("*") if p.is_file() and not p.name.startswith("._") and "task_data" not in p.parts
                  and p.name != "run_config.json"])
cfg = {
    "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
    "interpreter": sys.executable,
    "seeds": {"LET head (as deployed)": 42, "optimiser-noise seeds": list(range(1, 11)),
              "training-donor bootstrap": "numpy default_rng([20261001, rep]); evaluation level default_rng([20261001, rep, 7])",
              "internal permutation (re-fit)": "numpy default_rng([777, b]), b = 0..1999",
              "frozen-panel permutation": "numpy default_rng([777, panel_code, b]), b = 0..9999; codes external 1, zeroshot 2, lung 3",
              "frozen-panel donor bootstrap": "numpy default_rng([4242, panel_code, rep]), rep = 0..1999",
              "other orderings donor bootstrap": "numpy default_rng([5353, key_code, rep]), rep = 0..1999",
              "leave-one-training-donor-out (v2_10)": "no random numbers; head seed 42",
              "Monte-Carlo precision of interval end points (v2_11)": "numpy default_rng(99), 1,000 resamples of the replicate list"},
    "not_used_for_any_reported_number": ["v2_common.train_let_batched (checked in v2_02b_batched_check.py; differs from the exact "
                                         "trainer by up to 0.012 in trust, so it was not used)"],
    "torch_threads": "re-fit tasks 1 thread per worker; reproduction tasks also at 4 threads ('t4|' ids)",
    "inputs_sha256": inputs,
    "scripts_sha256": {str(p): sha256(p) for p in scripts},
    "outputs_sha256": {str(p): sha256(p) for p in outputs},
    "per_script_configs": sorted(parts),
}
(OUT / "run_config.json").write_text(json.dumps(cfg, indent=2))
print("inputs", len(inputs), "scripts", len(scripts), "outputs", len(outputs))
