#!/usr/bin/env bash
# Start a scientifically separate FHO campaign.  Unlike the historical
# aggregate-work campaigns, it fixes E_prod at every crank-turn boundary.
set -euo pipefail

ROOT=/home/mickaelbegon/Documents/Kevin/cocofest-pedalage
PY=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python
HSL=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/opt/libhsl/v2025.7.21/lib/libhsl.so
OUT="$ROOT/local-results/isokinetic-prescribed-unit-1p3-fho-per-cycle-radau7-20261002/fho-continuation"
RHO79="$ROOT/local-results/isokinetic-prescribed-unit-1p3-fho-bo64-20260930/rho-seed/validated-rho-trajectory.npz"

export PYTHONPATH="$ROOT" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT"
mkdir -p "$OUT" "$OUT/attempts" "$OUT/attempt-logs" "$(dirname "$OUT")/rho-seed"
cp --update=none "$RHO79" "$(dirname "$OUT")/rho-seed/validated-rho-trajectory.npz"

# Build FHO_3 directly from RHO_1..3.  Those RHO cycles provide the initial
# primal trajectory only; FHO_3 itself is solved with work W at each boundary.
"$PY" -c 'import importlib.util, pathlib, sys; p=pathlib.Path(".github/scripts/run_full_horizon_benchmark.py"); s=importlib.util.spec_from_file_location("fho_runner",p); m=importlib.util.module_from_spec(s); sys.modules[s.name]=m; s.loader.exec_module(m); m.write_rho_seed_prefix(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), 3)' \
  "$RHO79" "$OUT/rho-prefix-0003.npz"

taskset -c 0-11 "$PY" "$ROOT/examples/fes_multibody/cycling/cycling_fes_solver_comparison.py" \
  --solvers ipopt --objective fatigue --objective-shape quadratic \
  --formulation isokinetic --mechanical-formulation reduced \
  --isokinetic-kinematics prescribed --enforce-isokinetic-work-per-cycle \
  --isokinetic-omega -6.283185307179586 --energy-equivalent-torque 1.3 \
  --load-torque-min -4 --load-torque-max 4 --stimulations-per-cycle 30 \
  --ipopt-profile periodic_collocation --ipopt-collocation-method radau \
  --ipopt-collocation-degree 7 --ipopt-no-use-sx --ipopt-linear-solver ma57 \
  --warmup-ipopt-linear-solver ma57 --ipopt-hsl-library "$HSL" \
  --ipopt-max-iter 4000 --nlp-tolerance 1e-8 --primal-feasibility-threshold 1e-5 \
  --ipopt-enforce-start-constraints --ipopt-disable-standard-warmup \
  --adopt-common-initial-solution-warmup-cycles --optional-nlp-periodic-ipopt-hot-start \
  --initial-guess-diagnostics --single-shot --cycles-per-window 3 --n-windows 3 \
  --n-threads 12 --common-initial-solution "$OUT/rho-prefix-0003.npz" \
  --common-initial-solution-output "$OUT/fho-0003-solution.npz" \
  --output-json "$OUT/fho-0003-result.json" >"$OUT/fho-0003-driver.log" 2>&1

current=3
while [ "$current" -lt 180 ]; do
  advanced=false
  for increment in 3 2 1; do
    attempt="$OUT/attempts/from-${current}-plus${increment}-$(date -u +%Y%m%dT%H%M%SZ)"
    if taskset -c 0-11 "$PY" "$ROOT/scripts/run_isokinetic_fho_operational_continuation.py" \
      --output-dir "$OUT" --start-cycles "$current" --extension-cycles "$increment" \
      --attempt-label "plus${increment}" --attempt-dir "$attempt" \
      --relative-energy-limit 0.001 --collocation-degree 7 \
      --allow-rho-failure-fho-bridge \
      >"$OUT/attempt-logs/$(basename "$attempt").log" 2>&1; then
      current=$((current + increment))
      advanced=true
      break
    fi
  done
  if [ "$advanced" != true ]; then
    printf 'STOP: no accepted +3, +2, or +1 extension from FHO_%04d.\n' "$current" \
      >"$OUT/STOP-after-fho-${current}.txt"
    exit 0
  fi
done
