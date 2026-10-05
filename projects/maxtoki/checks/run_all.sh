#!/usr/bin/env bash
# Run every check in order. CPU only, reads the repository, writes only checks/results/.
# Total wall time on the project laptop: 3 minutes (warm cache) to 7 minutes (cold disk).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-$HERE/../.venv/bin/python}"
cd "$HERE"
"$PY" spec_lint.py
"$PY" number_trace.py
"$PY" run_manifest_check.py
"$PY" verify/verify_spec_lint.py
"$PY" verify/verify_chance_rate.py
"$PY" verify/crosscheck_manual_audit.py
"$PY" verify/verify_manifest_deviations.py
"$PY" verify/verify_number_trace.py
"$PY" summarize_results.py > /dev/null
"$PY" requirements_table.py
echo "done: see results/checker_summary.md and results/repair_queue.csv"
