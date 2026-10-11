#!/usr/bin/env bash
# First certified three-cycle bridge for the prescribed isokinetic PW² FHO.
set -euo pipefail

ROOT=/home/mickaelbegon/Documents/Kevin/cocofest-pedalage
PY=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python
HSL=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/opt/libhsl/v2025.7.21/lib/libhsl.so
OUT="$ROOT/local-results/isokinetic-prescribed-unit-1p3-pw2-20261001/fho-pw"
export PYTHONPATH="$ROOT" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT/rho-seed-0003" "$OUT/fho-0003"

# Construct the initial FHO seed solely from the first three unit-RHO PW² windows.
taskset -c 0-11 "$PY" "$ROOT/examples/fes_multibody/cycling/cycling_fes_solver_comparison.py" \
  --solvers ipopt --objective control --objective-shape quadratic \
  --formulation isokinetic --mechanical-formulation reduced \
  --isokinetic-kinematics prescribed --isokinetic-omega -6.283185307179586 \
  --energy-equivalent-torque 1.3 --load-torque-min -4 --load-torque-max 4 \
  --stimulations-per-cycle 30 --ipopt-profile periodic_collocation \
  --ipopt-collocation-method radau --ipopt-collocation-degree 5 --ipopt-use-sx \
  --ipopt-linear-solver ma57 --warmup-ipopt-linear-solver ma57 --ipopt-hsl-library "$HSL" \
  --ipopt-max-iter 4000 --nlp-tolerance 1e-8 --primal-feasibility-threshold 1e-5 \
  --ipopt-enforce-start-constraints --cycles-per-window 1 --n-windows 3 --n-threads 12 \
  --compact-rho-output --retry-failed-rho-without-advance \
  --receding-horizon-solution-output "$OUT/rho-seed-0003/trajectory.npz" \
  --output-json "$OUT/rho-seed-0003/result.json" >"$OUT/rho-seed-0003/driver.log" 2>&1

# FHO deliberately stays MX/interpreted; it starts from exactly those three RHO cycles.
exec taskset -c 0-11 "$PY" "$ROOT/examples/fes_multibody/cycling/cycling_fes_solver_comparison.py" \
  --solvers ipopt --objective control --objective-shape quadratic \
  --formulation isokinetic --mechanical-formulation reduced \
  --isokinetic-kinematics prescribed --isokinetic-omega -6.283185307179586 \
  --energy-equivalent-torque 1.3 --load-torque-min -4 --load-torque-max 4 \
  --stimulations-per-cycle 30 --ipopt-profile periodic_collocation \
  --ipopt-collocation-method radau --ipopt-collocation-degree 5 --ipopt-no-use-sx \
  --ipopt-linear-solver ma57 --warmup-ipopt-linear-solver ma57 --ipopt-hsl-library "$HSL" \
  --ipopt-max-iter 4000 --nlp-tolerance 1e-8 --primal-feasibility-threshold 1e-5 \
  --ipopt-enforce-start-constraints --ipopt-disable-standard-warmup \
  --adopt-common-initial-solution-warmup-cycles --optional-nlp-periodic-ipopt-hot-start \
  --initial-guess-diagnostics --single-shot --cycles-per-window 3 --n-windows 3 --n-threads 12 \
  --common-initial-solution "$OUT/rho-seed-0003/trajectory.npz" \
  --common-initial-solution-output "$OUT/fho-0003/solution.npz" \
  --output-json "$OUT/fho-0003/result.json" >"$OUT/fho-0003/driver.log" 2>&1
