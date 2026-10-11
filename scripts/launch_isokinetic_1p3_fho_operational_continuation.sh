#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/mickaelbegon/Documents/Kevin/cocofest-pedalage
PY=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python
OUT="$ROOT/local-results/isokinetic-prescribed-unit-1p3-fho-bo64-20260930/fho-continuation"

# CPU affinity is also imposed by the user systemd unit.  It is repeated here
# so child numerical processes cannot escape CPU 0--11.
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec taskset -c 0-11 "$PY" "$ROOT/scripts/run_isokinetic_fho_operational_continuation.py" \
  --output-dir "$OUT" --relative-energy-limit 0.001 "$@"
