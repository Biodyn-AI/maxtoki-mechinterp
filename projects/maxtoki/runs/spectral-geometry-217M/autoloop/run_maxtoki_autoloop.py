"""Streamlined MaxToki spectral-geometry autoloop runner.

Modelled after repos/topology-biomechinterp1/loop/run_claude_topology_autoloop.py
but stripped of the LaTeX-paper validation gate. Each iteration:

1. Run the executor: spawn Claude CLI with the executor prompt + the prior
   iteration's brainstormer brief + a list of all prior hypotheses.
   Executor must write executor_iteration_report.md + executor_hypothesis_screen.json
   + at least one machine-readable artefact in iter_NNNN/.
2. Validate executor output (basic schema check; no LaTeX gate).
3. Run the brainstormer: spawn Claude CLI with the brainstormer prompt + the
   executor's outputs. Writes brainstormer_*.md.
4. Loop.

Branch retirement: each iteration's hypothesis-screen JSON includes a
`retired` flag + `lineage`; the master log is appended automatically.

Stop conditions:
- max_iterations reached
- runtime/STOP file exists
- executor returns non-zero or no hypothesis-screen JSON for 2 consecutive iterations
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ITERATIONS = ROOT / "iterations"
PROMPTS = ROOT / "prompts"
RUNTIME = ROOT / "runtime"
RUNTIME.mkdir(exist_ok=True)
STOP_FILE = RUNTIME / "STOP"
MASTER_LOG = ROOT / "autoloop_master_log.md"
PROJECT_ROOT = Path("<REPO_ROOT>/projects/maxtoki")

CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "<CLAUDE_CLI>")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "sonnet")  # cheaper than opus for an iterative loop
CLAUDE_TIMEOUT_SECONDS = int(os.environ.get("CLAUDE_TIMEOUT", "1800"))  # 30 min per call
SLEEP_BETWEEN = int(os.environ.get("SLEEP_BETWEEN", "10"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _next_iter_number() -> int:
    if not ITERATIONS.exists():
        return 1
    nums = [
        int(d.name.split("_")[1])
        for d in ITERATIONS.iterdir()
        if d.is_dir() and d.name.startswith("iter_") and d.name.split("_")[1].isdigit()
    ]
    return (max(nums) + 1) if nums else 1


def _iter_dir(n: int) -> Path:
    return ITERATIONS / f"iter_{n:04d}"


def _prior_iter_dir(n: int) -> Path | None:
    if n <= 1:
        return None
    return _iter_dir(n - 1)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _append_log(line: str) -> None:
    MASTER_LOG.parent.mkdir(parents=True, exist_ok=True)
    with MASTER_LOG.open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def _all_prior_hypotheses(current_iter: int, max_scan: int = 10) -> str:
    """Compact summary of last N iterations' hypotheses for retirement-policy enforcement."""
    out_lines: list[str] = []
    scanned = 0
    for n in range(current_iter - 1, 0, -1):
        if scanned >= max_scan:
            break
        d = _iter_dir(n)
        screen = d / "executor_hypothesis_screen.json"
        if not screen.exists():
            continue
        try:
            payload = json.loads(_read(screen))
            for h in payload.get("hypotheses", []):
                out_lines.append(
                    f"  {d.name} {h.get('id', '?')} | family={h.get('family', '?')} | "
                    f"decision={h.get('decision', '?')} | direction={h.get('result_direction', '?')} | "
                    f"retired={h.get('retired', False)}"
                )
        except Exception as exc:
            out_lines.append(f"  {d.name} (parse error: {exc})")
        scanned += 1
    return "\n".join(out_lines) if out_lines else "  (no prior iterations)"


def _spawn_claude(prompt_text: str, cwd: Path, label: str, log_path: Path) -> int:
    """Run claude CLI as a non-interactive subprocess. Returns exit code."""
    cmd = [
        CLAUDE_BIN,
        "--model", CLAUDE_MODEL,
        "--print",
        "--permission-mode", "bypassPermissions",
        "--allowed-tools", "Read", "Write", "Edit", "Bash", "Glob", "Grep",
        "--max-budget-usd", "5",
    ]
    print(f"[{_utc_now()}] [{label}] spawning: {' '.join(cmd)}  (cwd={cwd})", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(f"=== {label} prompt ===\n{prompt_text}\n\n=== response ===\n")
        logf.flush()
        try:
            proc = subprocess.run(
                cmd,
                input=prompt_text,
                cwd=str(cwd),
                stdout=logf,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=CLAUDE_TIMEOUT_SECONDS,
            )
            return proc.returncode
        except subprocess.TimeoutExpired:
            logf.write(f"\n[TIMEOUT after {CLAUDE_TIMEOUT_SECONDS}s]\n")
            return 124
        except Exception as exc:
            logf.write(f"\n[EXCEPTION] {type(exc).__name__}: {exc}\n")
            return 1


def _validate_executor_output(iter_dir: Path) -> tuple[bool, list[str]]:
    failures: list[str] = []
    report = iter_dir / "executor_iteration_report.md"
    screen = iter_dir / "executor_hypothesis_screen.json"
    if not report.exists():
        failures.append("missing executor_iteration_report.md")
    if not screen.exists():
        failures.append("missing executor_hypothesis_screen.json")
    else:
        try:
            payload = json.loads(_read(screen))
            hyps = payload.get("hypotheses", [])
            if not isinstance(hyps, list) or len(hyps) == 0:
                failures.append("hypothesis_screen has empty 'hypotheses'")
        except Exception as exc:
            failures.append(f"hypothesis_screen JSON invalid: {exc}")
    # at least one machine-readable artefact (other than the screen JSON)
    has_artefact = False
    for p in iter_dir.iterdir():
        if p.is_file() and p.suffix.lower() in {".csv", ".tsv", ".npy", ".npz", ".pkl", ".pt", ".parquet"}:
            has_artefact = True
            break
        if p.is_file() and p.suffix.lower() == ".json" and p.name != "executor_hypothesis_screen.json":
            has_artefact = True
            break
    if not has_artefact:
        failures.append("no machine-readable artefact (csv/json/npy/...) in iter_dir")
    return (len(failures) == 0, failures)


def _build_executor_prompt(iter_n: int, iter_dir: Path) -> str:
    base = _read(PROMPTS / "executor_prompt.md")
    prior_brief = ""
    prev = _prior_iter_dir(iter_n)
    if prev is not None:
        brief = prev / "brainstormer_next_iteration_brief.md"
        if brief.exists():
            prior_brief = f"\n\n## Brainstormer brief (prior iteration {prev.name})\n\n{_read(brief)}"
    prior_hyps = _all_prior_hypotheses(iter_n)
    addendum = (
        f"\n\n---\n\n"
        f"## This iteration\n\n"
        f"- iteration name: `iter_{iter_n:04d}`\n"
        f"- write all artefacts to: `{iter_dir}`\n"
        f"- replace `iter_XXXX` in your hypothesis-screen JSON with `iter_{iter_n:04d}`\n"
        f"- python: use the system `python` (anaconda base, has torch+transformers+safetensors)\n\n"
        f"## Recent prior hypotheses (for retirement-policy enforcement)\n\n"
        f"{prior_hyps}\n"
        f"{prior_brief}\n"
    )
    return base + addendum


def _build_brainstormer_prompt(iter_n: int, iter_dir: Path) -> str:
    base = _read(PROMPTS / "brainstormer_prompt.md")
    base = (
        base.replace("{{ITERATION_DIR}}", str(iter_dir))
            .replace("{{PROJECT_ROOT}}", str(PROJECT_ROOT))
            .replace("{{ITERATIONS_ROOT}}", str(ITERATIONS))
    )
    addendum = (
        f"\n\n---\n\n"
        f"## This iteration\n\n"
        f"- iteration: `iter_{iter_n:04d}`\n"
        f"- iter_dir: `{iter_dir}`\n"
        f"- write your three brainstormer_*.md files into iter_dir.\n"
    )
    return base + addendum


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--start-iter", type=int, default=None,
                        help="resume from this iteration number (defaults to next free)")
    args = parser.parse_args()

    if not Path(CLAUDE_BIN).exists():
        print(f"FATAL: claude binary not found at {CLAUDE_BIN}", flush=True)
        return 2
    if STOP_FILE.exists():
        STOP_FILE.unlink()

    start = args.start_iter or _next_iter_number()
    end = start + args.max_iterations - 1
    print(f"[{_utc_now()}] autoloop start  range=iter_{start:04d}..iter_{end:04d}  "
          f"model={CLAUDE_MODEL}  timeout={CLAUDE_TIMEOUT_SECONDS}s", flush=True)
    _append_log(f"\n## autoloop run started {_utc_now()}  iters {start}..{end}  model={CLAUDE_MODEL}\n")

    consecutive_failures = 0

    for n in range(start, end + 1):
        if STOP_FILE.exists():
            print(f"[{_utc_now()}] STOP file present → halting before iter_{n:04d}", flush=True)
            break
        iter_dir = _iter_dir(n)
        iter_dir.mkdir(parents=True, exist_ok=True)
        iter_log = RUNTIME / f"iter_{n:04d}.log"

        # ---- executor ----
        prompt_e = _build_executor_prompt(n, iter_dir)
        (iter_dir / "executor_prompt.md").write_text(prompt_e, encoding="utf-8")
        t0 = time.time()
        rc_e = _spawn_claude(prompt_e, iter_dir, f"executor iter_{n:04d}", iter_log)
        dt_e = time.time() - t0
        ok_e, fails_e = _validate_executor_output(iter_dir)
        line = (f"[{_utc_now()}] iter_{n:04d} executor done  rc={rc_e}  dt={dt_e:.0f}s  "
                f"valid={ok_e}  fails={fails_e}")
        print(line, flush=True)
        _append_log("- " + line)

        if not ok_e:
            consecutive_failures += 1
            if consecutive_failures >= 2:
                print(f"[{_utc_now()}] aborting: {consecutive_failures} consecutive executor failures", flush=True)
                _append_log(f"- [{_utc_now()}] ABORT after {consecutive_failures} consecutive failures")
                return 3
        else:
            consecutive_failures = 0

        # ---- brainstormer (only if executor succeeded; otherwise skip) ----
        if ok_e:
            prompt_b = _build_brainstormer_prompt(n, iter_dir)
            (iter_dir / "brainstormer_prompt.md").write_text(prompt_b, encoding="utf-8")
            iter_log_b = RUNTIME / f"iter_{n:04d}_brainstormer.log"
            t0 = time.time()
            rc_b = _spawn_claude(prompt_b, iter_dir, f"brainstormer iter_{n:04d}", iter_log_b)
            dt_b = time.time() - t0
            line_b = (f"[{_utc_now()}] iter_{n:04d} brainstormer done  rc={rc_b}  dt={dt_b:.0f}s")
            print(line_b, flush=True)
            _append_log("- " + line_b)

        time.sleep(SLEEP_BETWEEN)

    print(f"[{_utc_now()}] autoloop complete", flush=True)
    _append_log(f"## autoloop run finished {_utc_now()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
