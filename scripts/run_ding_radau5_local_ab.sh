#!/usr/bin/env bash
# Experimental local-state A/B with identical current-code settings.
set -euo pipefail
source .github/scripts/benchmark_env.sh rho32
windows="${1:-1}"
mode="${2:-local}"
destination="${3:-ding-local-radau5-ab-20260913/$mode$windows}"
mkdir -p "$destination"
python scripts/benchmark_ding_radau5_local_ab.py "$mode" -- \
  --solvers ipopt --objective fatigue --ipopt-profile scientific-radau5 \
  --ipopt-enforce-start-constraints --cycles-per-window 1 \
  --stimulations-per-cycle 30 --n-windows "$windows" --n-threads 1 \
  --signed-crank-torque 0.1 --nlp-tolerance 1e-6 --primal-feasibility-threshold 1e-5 \
  --common-initial-solution fatrop-linear-solver-150-20260912/resistance-0p10Nm/common-seed-radau5-ma57-current/common-reduced.npz \
  --adopt-common-initial-solution-warmup-cycles --ipopt-linear-solver ma57 \
  --ipopt-disable-historical-initial-guess \
  --reduced-cycling-profile benchmark-seed/reduced-cycling-fourier12.npz \
  --state-scaling full --first-node-wheel-q-slack 0 --terminal-wheel-q-slack 0.002 \
  --formulation dynamic --compact-rho-output --rho-pulse-width-transfer-mode repeat \
  --rho-pulse-width-extrapolation-factor 1.0 --reduced-internal-crank-velocity-guard off \
  --no-optional-nlp-periodic-ipopt-hot-start --output-json "$destination/result.json" \
  2>&1 | tee "$destination/solver.log"
