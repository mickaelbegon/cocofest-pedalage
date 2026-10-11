#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/mickaelbegon/Documents/Kevin/cocofest-pedalage
PY=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python
HSL=/home/mickaelbegon/miniforge3/envs/cocofest-rho32/opt/libhsl/v2025.7.21/lib/libhsl.so
OUT="$ROOT/local-results/isokinetic-prescribed-unit-1p3-fho-bo64-20260930/fho-continuation"
RHO79="$ROOT/local-results/isokinetic-prescribed-unit-1p3-fho-bo64-20260930/rho-seed/validated-rho-trajectory.npz"
export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT"
mkdir -p "$OUT"

# The continuation explicitly starts at FHO_3, not at a monolithic FHO_79.
# Use the certified RHO_1..3 prefix only as the initial primal trajectory.
"$PY" -c 'import importlib.util, pathlib, sys; p=pathlib.Path(".github/scripts/run_full_horizon_benchmark.py"); s=importlib.util.spec_from_file_location("fho_runner",p); m=importlib.util.module_from_spec(s); sys.modules[s.name]=m; s.loader.exec_module(m); m.write_rho_seed_prefix(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), 3)' \
  "$RHO79" "$OUT/rho-prefix-0003.npz"

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
  --initial-guess-diagnostics --single-shot --cycles-per-window 3 --n-windows 3 --n-threads 12 \
  --common-initial-solution "$OUT/rho-prefix-0003.npz" \
  --common-initial-solution-output "$OUT/fho-0003-solution.npz" \
  --output-json "$OUT/fho-0003-result.json" >"$OUT/driver.log" 2>&1
