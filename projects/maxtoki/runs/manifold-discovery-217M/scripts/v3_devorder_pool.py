"""V3-7 step 3: every LET-head fit of the v2b analysis, re-run on the v3 features (resumable task pool).

The task list, the task code and the gate code are those of scripts/v2b_devorder_02_pool.py, imported unchanged
after patch_v2b() has pointed its paths at outputs/v3_devorder/. Results: outputs/v3_devorder/pool/results.jsonl.

Usage:
  python v3_devorder_pool.py list [prefix ...]
  python v3_devorder_pool.py run <n_workers> <start_budget_s> [prefix ...]
      starts tasks only during the first <start_budget_s> seconds, then waits for running ones and exits.
  python v3_devorder_pool.py threads <n_threads> <key> [<key> ...]     (the v2b 02b thread-count check)
Workers: torch 1 thread each (as v2b). Every worker process applies the same patch in its initializer before it
touches any v2b module.
"""
import sys
sys.dont_write_bytecode = True
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "<REPO_ROOT>/projects/maxtoki/runs/manifold-discovery-217M/scripts")
import json
import time
import multiprocessing as mp

from v3_devorder_common import V3, patch_v2b, assert_paths_v3

V = patch_v2b()
import v2b_devorder_02_pool as POOL   # noqa: E402

RES = V3 / "pool" / "results.jsonl"


def v3_init():
    import torch
    torch.set_num_threads(1)
    V_ = patch_v2b()
    c = POOL.C()
    assert c["V"] is V_ and str(c["V"].OUT).endswith("v3_devorder") and str(c["V"].ART).endswith("v3_devorder/artifacts")


def done_keys():
    return POOL.done_keys(RES)


def main():
    (V3 / "pool").mkdir(exist_ok=True)
    cmd = sys.argv[1]
    v3_init()
    if cmd == "threads":
        import torch
        nt = int(sys.argv[2]); torch.set_num_threads(nt)
        out = V3 / "thread_check"; out.mkdir(exist_ok=True)
        with open(out / "results.jsonl", "a") as fh:
            for key in sys.argv[3:]:
                k, res, sec, err = POOL.run_task(key)
                if res and "frozen" in res:
                    res = {"trust": res["trust"]}
                if res:
                    res.pop("z", None); res.pop("test_idx", None)
                fh.write(json.dumps({"threads": nt, "key": k, "res": res, "error": err, "sec": round(sec, 1)},
                                    default=float) + "\n")
                print(nt, k, res, err, round(sec, 1), flush=True)
        return
    tasks = POOL.all_tasks()
    (V3 / "pool" / "task_list.json").write_text(json.dumps(tasks))
    done = done_keys()
    if cmd == "list":
        pre = sys.argv[2:] or [""]
        sel = [t for t in tasks if any(t.startswith(p) for p in pre)]
        print("tasks", len(sel), "done", sum(t in done for t in sel), "pending", sum(t not in done for t in sel))
        return
    nw = int(sys.argv[2]); budget = float(sys.argv[3]); pre = sys.argv[4:] or [""]
    todo = [t for t in tasks if t not in done and any(t.startswith(p) for p in pre)]
    deadline = time.time() + budget
    print(f"pending {len(todo)} (selected), workers {nw}, start budget {budget:.0f}s", flush=True)
    n_ok = 0; T0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(nw, initializer=v3_init) as pool, open(RES, "a") as fh:
        for key, res, sec, err in pool.imap_unordered(POOL._guarded, [(t, deadline) for t in todo], chunksize=1):
            if err == "SKIPPED_DEADLINE":
                continue
            fh.write(json.dumps({"key": key, "res": res, "sec": round(sec, 2), "error": err}, default=float) + "\n")
            fh.flush()
            n_ok += err is None
            if err:
                print("ERROR", key, err[-400:], flush=True)
    left = len([t for t in tasks if t not in done_keys()])
    print(f"finished {n_ok} tasks in {time.time()-T0:.0f}s; tasks left overall: {left}", flush=True)


if __name__ == "__main__":
    main()
