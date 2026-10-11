#!/usr/bin/env bash
# Solve the full 73-cycle fixed-weight RHO-BO PW² candidate as one FHO.
#
# This is a candidate-generating step.  A strict DOP853 energy audit remains
# the scientific certificate; the exported solution is retained even if the
# already-observed sub-0.1% cross-integrator discrepancy rejects that gate.
set -euo pipefail

ROOT=/home/mickaelbegon/Documents/Kevin/cocofest-pedalage
PY=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python
HSL=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/opt/libhsl/v2025.7.21/lib/libhsl.so
BASE="$ROOT/local-results/isokinetic-prescribed-unit-1p3-pw2-20261001"
OUT="$BASE/fho-pw-bo-step1"
SEED="$BASE/analysis/replay-rho-bo-pw/trajectory.npz"

export PYTHONPATH="$ROOT" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT/fho-0073"

# The four weights are the BO optimum, normalized to geometric mean one, in
# Delt_ant, Delt_post, Biceps, Triceps order.  The FHO uses precisely the same
# fixed per-muscle PW² objective as its RHO-BO seed.
taskset -c 0-11 "$PY" "$ROOT/examples/fes_multibody/cycling/cycling_fes_solver_comparison.py" \
  --solvers ipopt --objective control --objective-shape quadratic \
  --parametric-control-weights \
  --control-weight-values 1.6197008272329414 0.6423432578716108 0.8614265710555464 1.1157830567058247 \
  --formulation isokinetic --mechanical-formulation reduced \
  --isokinetic-kinematics prescribed --isokinetic-omega -6.283185307179586 \
  --energy-equivalent-torque 1.3 --load-torque-min -4 --load-torque-max 4 \
  --stimulations-per-cycle 30 --ipopt-profile periodic_collocation \
  --ipopt-collocation-method radau --ipopt-collocation-degree 5 --ipopt-no-use-sx \
  --ipopt-linear-solver ma57 --warmup-ipopt-linear-solver ma57 --ipopt-hsl-library "$HSL" \
  --ipopt-max-iter 4000 --nlp-tolerance 1e-8 --primal-feasibility-threshold 1e-5 \
  --ipopt-enforce-start-constraints --ipopt-disable-standard-warmup \
  --adopt-common-initial-solution-warmup-cycles --optional-nlp-periodic-ipopt-hot-start \
  --initial-guess-diagnostics --allow-primal-feasible-common-initial-solution-output \
  --single-shot --cycles-per-window 73 --n-windows 73 --n-threads 12 \
  --common-initial-solution "$SEED" \
  --common-initial-solution-output "$OUT/fho-0073/solution.npz" \
  --output-json "$OUT/fho-0073/result.json" >"$OUT/fho-0073/driver.log" 2>&1

# A result within 0.1 % of the independent DOP853 energy check can be used as
# an *operational* continuation seed.  This sidecar never alters the strict
# `success` value produced by the solver.
"$PY" "$ROOT/scripts/run_isokinetic_fho_operational_continuation.py" \
  --output-dir "$OUT" --relative-energy-limit 0.001 \
  --audit-only-result "$OUT/fho-0073/result.json" \
  --audit-output "$OUT/fho-0073/operational-audit.json"
