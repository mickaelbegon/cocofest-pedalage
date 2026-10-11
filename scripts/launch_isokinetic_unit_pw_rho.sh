#!/usr/bin/env bash
# Certified unit-RHO baseline for the pulse-width-squared campaign.
set -euo pipefail

ROOT=/home/mickaelbegon/Documents/Kevin/cocofest-pedalage
PY=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python
HSL=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/opt/libhsl/v2025.7.21/lib/libhsl.so
OUT="$ROOT/local-results/isokinetic-prescribed-unit-1p3-pw2-20261001/rho-unit-pw"

export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT"

# The control objective returns one normalized direct PW value per muscle.
# With --objective-shape quadratic, IPOPT minimizes sum_i (PW_i/PWmax_i)^2.
exec taskset -c 12 "$PY" "$ROOT/examples/fes_multibody/cycling/cycling_fes_solver_comparison.py" \
  --solvers ipopt --objective control --objective-shape quadratic \
  --formulation isokinetic --mechanical-formulation reduced \
  --isokinetic-kinematics prescribed --isokinetic-omega -6.283185307179586 \
  --energy-equivalent-torque 1.3 --load-torque-min -4 --load-torque-max 4 \
  --stimulations-per-cycle 30 --ipopt-profile periodic_collocation \
  --ipopt-collocation-method radau --ipopt-collocation-degree 5 --ipopt-use-sx \
  --ipopt-linear-solver ma57 --warmup-ipopt-linear-solver ma57 --ipopt-hsl-library "$HSL" \
  --ipopt-max-iter 4000 --nlp-tolerance 1e-8 --primal-feasibility-threshold 1e-5 \
  --ipopt-enforce-start-constraints --cycles-per-window 1 --n-windows 160 --n-threads 1 \
  --compact-rho-output --retry-failed-rho-without-advance \
  --receding-horizon-solution-output "$OUT/validated-rho-trajectory.npz" \
  --allow-partial-receding-horizon-solution-output \
  --rho-replay-checkpoint-output "$OUT/last-certified-rho-replay.npz" \
  --output-json "$OUT/result.json"
