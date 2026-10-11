#!/usr/bin/env bash
# Reproducible unilateral K5-I1 plus local Ding, with identical frozen IPOPT calls.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python}"
HSL="${HSL:-/home/mickaelbegon/miniforge3/envs/cocofest-rho32/opt/libhsl/v2025.7.21/lib/libhsl.so}"
OUT="${1:-$ROOT/local-results/prescribed-ding-local-ab}"
WINDOWS="${2:-100}"
mkdir -p "$OUT/baseline" "$OUT/local"
OUT="$(realpath "$OUT")"
cd "$ROOT"

export PYTHONPATH="$ROOT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-cocofest}"

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
)

if [[ ! -f "$OUT/baseline/result.json" ]]; then
  "$PYTHON" scripts/benchmark_ding_radau5_local_ab.py baseline -- \
    "${ARGS[@]}" --output-json "$OUT/baseline/result.json" \
    >"$OUT/baseline/solver.log" 2>&1
fi

if [[ ! -f "$OUT/local/result.json" ]]; then
  DING_FROZEN_INPUT_DIR="$OUT/baseline/symbolic-audit" \
  DING_FROZEN_RETURN_RECORDED=1 \
  "$PYTHON" scripts/benchmark_ding_radau5_local_ab.py local -- \
    "${ARGS[@]}" --output-json "$OUT/local/result.json" \
    >"$OUT/local/solver.log" 2>&1
fi

"$PYTHON" scripts/summarize_prescribed_ding_local_ab.py "$OUT"
