#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/mickaelbegon/Documents/Kevin/cocofest-pedalage
PY=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python
HSL=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/opt/libhsl/v2025.7.21/lib/libhsl.so
OUT="$ROOT/local-results/isokinetic-prescribed-unit-1p3-fho-bo64-20260930"
export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT/rho-seed" "$OUT/fho"

# Recreate and export exactly the known 79-cycle certified RHO prefix.  This
# archive, rather than the one-cycle replay checkpoint, is the FHO seed.
taskset -c 0-11 "$PY" "$ROOT/examples/fes_multibody/cycling/cycling_fes_solver_comparison.py" \
  --solvers ipopt --objective fatigue --objective-shape quadratic \
  --formulation isokinetic --mechanical-formulation reduced \
  --isokinetic-kinematics prescribed --isokinetic-omega -6.283185307179586 \
  --energy-equivalent-torque 1.3 --load-torque-min -4 --load-torque-max 4 \
  --stimulations-per-cycle 30 --ipopt-profile periodic_collocation \
  --ipopt-collocation-method radau --ipopt-collocation-degree 5 --ipopt-use-sx \
  --ipopt-linear-solver ma57 --warmup-ipopt-linear-solver ma57 --ipopt-hsl-library "$HSL" \
  --ipopt-max-iter 4000 --nlp-tolerance 1e-8 --primal-feasibility-threshold 1e-5 \
  --ipopt-enforce-start-constraints --cycles-per-window 1 --n-windows 79 --n-threads 12 \
  --compact-rho-output --retry-failed-rho-without-advance \
  --receding-horizon-solution-output "$OUT/rho-seed/validated-rho-trajectory.npz" \
  --rho-replay-checkpoint-output "$OUT/rho-seed/last-certified-rho-replay.npz" \
  --output-json "$OUT/rho-seed/result.json" >"$OUT/rho-seed/driver.log" 2>&1

# FHO stays MX/interpreted with exact IPOPT Hessian and MA57.  It only starts
# after the multi-cycle RHO seed passed its own trajectory/seam contract.
taskset -c 0-11 "$PY" "$ROOT/examples/fes_multibody/cycling/cycling_fes_solver_comparison.py" \
  --solvers ipopt --objective fatigue --objective-shape quadratic \
  --formulation isokinetic --mechanical-formulation reduced \
  --isokinetic-kinematics prescribed --isokinetic-omega -6.283185307179586 \
  --energy-equivalent-torque 1.3 --load-torque-min -4 --load-torque-max 4 \
  --stimulations-per-cycle 30 --ipopt-profile periodic_collocation \
  --ipopt-collocation-method radau --ipopt-collocation-degree 5 --ipopt-no-use-sx \
  --ipopt-linear-solver ma57 --warmup-ipopt-linear-solver ma57 --ipopt-hsl-library "$HSL" \
  --ipopt-max-iter 4000 --nlp-tolerance 1e-8 --primal-feasibility-threshold 1e-5 \
  --ipopt-enforce-start-constraints --ipopt-disable-standard-warmup \
  --adopt-common-initial-solution-warmup-cycles --optional-nlp-periodic-ipopt-hot-start \
  --initial-guess-diagnostics --single-shot --cycles-per-window 79 --n-windows 79 --n-threads 12 \
  --common-initial-solution "$OUT/rho-seed/validated-rho-trajectory.npz" \
  --common-initial-solution-output "$OUT/fho/solution.npz" \
  --output-json "$OUT/fho/result.json" >"$OUT/fho/driver.log" 2>&1
