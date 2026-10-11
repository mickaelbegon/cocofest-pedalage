#!/usr/bin/env bash
# Experimental autonomous K5-I1 run: propagate the candidate's own solution.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python}"
HSL="${HSL:-/home/mickaelbegon/miniforge3/envs/cocofest-rho32/opt/libhsl/v2025.7.21/lib/libhsl.so}"
MODE="${1:?expected baseline, local, or compiled}"
OUT="${2:?expected output directory}"
WINDOWS="${3:-100}"
CACHE="${4:-$OUT/cache}"
case "$MODE" in baseline|local|compiled) ;; *) exit 2 ;; esac
mkdir -p "$OUT"
OUT="$(realpath "$OUT")"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-cocofest}"
unset DING_FROZEN_INPUT_DIR DING_FROZEN_RETURN_RECORDED DING_FROZEN_CALL_OFFSET

ARGS=(
  --solvers ipopt --objective fatigue --ipopt-profile scientific-radau5
  --ipopt-use-sx --mechanical-formulation reduced --formulation isokinetic
  --isokinetic-kinematics prescribed --isokinetic-omega -6.283185307179586
  --energy-equivalent-torque 0.2 --load-torque-min -3 --load-torque-max 3
  --ipopt-enforce-start-constraints --cycles-per-window 1
  --stimulations-per-cycle 30 --n-windows "$WINDOWS" --n-threads 1
  --ipopt-linear-solver ma57 --ipopt-hsl-library "$HSL"
  --nlp-tolerance 1e-6 --primal-feasibility-threshold 1e-5
  --reduced-cycling-profile benchmark-seed/reduced-cycling-fourier12.npz
  --state-scaling full --compact-rho-output --rho-pulse-width-transfer-mode repeat
  --rho-pulse-width-extrapolation-factor 1.0
  --reduced-internal-crank-velocity-guard off
  --no-optional-nlp-periodic-ipopt-hot-start
  --output-json "$OUT/result.json"
)
if [[ "$MODE" == compiled ]]; then
  if [[ -n "${DING_PAIR_SIDE:-}" ]]; then
    "$PYTHON" scripts/benchmark_ding_local_pair_worker.py \
      --barrier-dir "${DING_PAIR_BARRIER_DIR:?missing pair barrier directory}" \
      --side "$DING_PAIR_SIDE" -- \
      --callbacks hessian --cache-dir "$CACHE" local -- "${ARGS[@]}"
  else
    "$PYTHON" scripts/benchmark_ding_local_compiled.py \
      --callbacks hessian --cache-dir "$CACHE" local -- "${ARGS[@]}"
  fi
else
  "$PYTHON" scripts/benchmark_ding_radau5_local_ab.py "$MODE" -- "${ARGS[@]}"
fi
