#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/mickaelbegon/Documents/Kevin/cocofest-pedalage
PY=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python
OUT="$ROOT/local-results/isokinetic-prescribed-unit-1p3-fho-bo64-20260930/fho-continuation"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# FHO_9 is the last accepted prefix.  Each invocation solves three local RHOs
# and one larger FHO.  ``set -e`` makes the service stop at the first rejected
# numerical or operational-audit stage; it never skips an uncertified bridge.
for start_cycles in $(seq 9 3 75); do
  taskset -c 0-11 "$PY" "$ROOT/scripts/run_isokinetic_fho_operational_continuation.py" \
    --output-dir "$OUT" --relative-energy-limit 0.001 --start-cycles "$start_cycles"
done
