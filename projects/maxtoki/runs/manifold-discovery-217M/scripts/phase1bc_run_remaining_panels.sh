#!/usr/bin/env bash
# Sequentially run Phase 1B+C for the three non-internal panels. Each panel
# blocks on the previous so a single GPU isn't oversubscribed.
set -e
set -o pipefail

cd "$(dirname "$0")/../../../../.."
PY=<CONDA_ROOT>/bin/python
SCRIPT=projects/maxtoki/runs/manifold-discovery-217M/scripts/phase1bc_hidden_states_and_centroids.py
LOGDIR=projects/maxtoki/runs/manifold-discovery-217M/outputs

for panel in zeroshot lung_control external; do
    echo "=== panel: $panel  start: $(date -Iseconds) ==="
    $PY -u "$SCRIPT" "$panel" 2>&1 | tee "$LOGDIR/phase1b_$panel.log"
    echo "=== panel: $panel  end:   $(date -Iseconds) ==="
done
