#!/usr/bin/env bash
# Re-optimise the whole fixed-weight RHO-BO fatigue trajectory as one FHO.
#
# Unlike the earlier fatigue bridge, its 82-cycle initial guess is the full
# RHO-BO trajectory, with the identical fixed fatigue weights.  This cleanly
# tests whether the BO seed improves FHO convergence rather than comparing
# different objectives.
set -euo pipefail

ROOT=/home/mickaelbegon/Documents/Kevin/cocofest-pedalage
PY=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python
HSL=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/opt/libhsl/v2025.7.21/lib/libhsl.so
BASE="$ROOT/local-results/isokinetic-prescribed-unit-1p3-fho-bo64-20260930"
SEED="$ROOT/local-results/isokinetic-prescribed-unit-1p3-pw2-20261001/analysis/replay-rho-bo-min-fatigue/trajectory.npz"
OUT="$BASE/fho-from-rho-bo-fatigue"

export PYTHONPATH="$ROOT" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT/fho-0082"

exec taskset -c 0-11 "$PY" "$ROOT/examples/fes_multibody/cycling/cycling_fes_solver_comparison.py" \
  --solvers ipopt --objective fatigue --objective-shape quadratic \
  --parametric-fatigue-weights \
  --fatigue-weight-values 0.16344816093170542 0.7727057265931767 0.21106237001190037 0.9427821707470022 \
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
  --single-shot --cycles-per-window 82 --n-windows 82 --n-threads 12 \
  --common-initial-solution "$SEED" \
  --common-initial-solution-output "$OUT/fho-0082/solution.npz" \
  --output-json "$OUT/fho-0082/result.json" >"$OUT/fho-0082/driver.log" 2>&1

# Record the explicit 0.1 % operational decision alongside the strict output.
"$PY" "$ROOT/scripts/run_isokinetic_fho_operational_continuation.py" \
  --output-dir "$OUT" --relative-energy-limit 0.001 \
  --audit-only-result "$OUT/fho-0082/result.json" \
  --audit-output "$OUT/fho-0082/operational-audit.json"
