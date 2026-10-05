"""v2b step 2b: does the torch thread count explain the small H38 / H95 reproduction gaps?
Re-runs selected pool tasks (MaxToki, deployed rulers) with 1, 4 and 6 torch threads, one process per call.
Usage: python v2b_devorder_02b_threads.py <threads> <task key> [<task key> ...]
Writes outputs/v2b_devorder/thread_check/results.jsonl."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, "<REPO_ROOT>/projects/maxtoki/runs/manifold-discovery-217M/scripts")
import json, time
import torch
nt = int(sys.argv[1]); torch.set_num_threads(nt)
import v2b_devorder_02_pool as POOL
POOL.C()
torch.set_num_threads(nt)
out = POOL.C()["V"].OUT / "thread_check" / "results.jsonl"
with open(out, "a") as fh:
    for key in sys.argv[2:]:
        t0 = time.time()
        k, res, sec, err = POOL.run_task(key)
        if res and "frozen" in res:
            res = {"trust": res["trust"]}
        if res:
            res.pop("z", None); res.pop("test_idx", None)
        fh.write(json.dumps({"threads": nt, "key": k, "res": res, "error": err, "sec": round(sec, 1)}, default=float) + "\n")
        print(nt, k, res, err, round(sec, 1), flush=True)
