#!/usr/bin/env bash
# Paired sequential ACADOS replay, each process limited to eight mapping threads.
set -euo pipefail
workspace=/home/mickaelbegon/Documents/Kevin/cocofest-pedalage
runtime=/home/mickaelbegon/miniforge3/envs/cocofest-rho32
windows="${1:-8}"
pilot_dir="${2:?Supply a new absolute output directory under the workspace.}"
existing_seed="${3:-}"
if ! [[ "$windows" =~ ^[1-9][0-9]*$ ]]; then
  echo "WINDOWS must be a positive integer" >&2
  exit 2
fi
case "$pilot_dir" in "$workspace"/*) ;; *) echo "Output must be inside the workspace" >&2; exit 2;; esac
if [[ -e "$pilot_dir" ]]; then
  echo "Refusing to overwrite an existing pilot directory: $pilot_dir" >&2
  exit 2
fi
export PYTHONPATH="$workspace${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$runtime/bin:$PATH"
export LD_LIBRARY_PATH="$runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ACADOS_SOURCE_DIR="$runtime"
export TERA_PATH="$runtime/bin/t_renderer"
export IPOPT_HSL_LIBRARY="$runtime/opt/libhsl/v2025.7.21/lib/libhsl.so"
export MPLCONFIGDIR="$pilot_dir/matplotlib" MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export OMP_THREAD_LIMIT=1 OMP_DYNAMIC=FALSE NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
mkdir -p "$pilot_dir/ipopt-cycle1/codegen" "$pilot_dir/nominal/codegen" "$pilot_dir/budget30/codegen"
common=(
  --benchmark-profile scientific-radau5 --objective fatigue --ipopt-use-sx
  --ipopt-linear-solver ma57 --warmup-ipopt-linear-solver ma57
  --cycles-per-window 1 --stimulations-per-cycle 50 --n-threads 8
  --signed-crank-torque 0.1 --terminal-wheel-q-slack 0.002
  --mechanical-formulation reduced --formulation dynamic
  --pulse-width-max-step-us 100 --reduced-internal-crank-velocity-guard on
  --compact-rho-output
)
cd "$pilot_dir/ipopt-cycle1/codegen"
seed_file="$pilot_dir/ipopt-cycle1/common-cycle1.npz"
if [[ -n "$existing_seed" ]]; then
  test -f "$existing_seed"
  seed_file="$existing_seed"
else
"$runtime/bin/python" "$workspace/examples/fes_multibody/cycling/cycling_fes_solver_comparison.py" \
  "${common[@]}" --solvers ipopt --n-windows 1 --ipopt-max-iter 2000 \
  --ipopt-ode-solver collocation --ipopt-collocation-degree 5 --ipopt-collocation-method radau \
  --codegen-tag "budget-pilot-seed-$windows" \
  --common-initial-solution-output "$pilot_dir/ipopt-cycle1/common-cycle1.npz" \
  --output-json "$pilot_dir/ipopt-cycle1/result.json" \
  > "$pilot_dir/ipopt-cycle1/solver.log" 2>&1
fi
test -f "$seed_file"
for variant in nominal budget30; do
  budget_options=()
  if [[ "$variant" == budget30 ]]; then
    budget_options=(--acados-rho-initial-iteration-budget 30)
  fi
  cd "$pilot_dir/$variant/codegen"
  "$runtime/bin/python" "$workspace/examples/fes_multibody/cycling/cycling_fes_solver_comparison.py" \
    "${common[@]}" --solvers acados --n-windows "$windows" \
    --acados-dir "$runtime" --experimental-reduced-acados \
    --acados-nlp-solver-type SQP --acados-integrator-type IRK \
    --acados-sim-stages 4 --acados-sim-steps 5 --acados-max-iter 100 \
    --codegen-tag "budget-pilot-$variant-$windows" \
    --common-initial-solution "$seed_file" \
    --adopt-common-initial-solution-warmup-cycles \
    --common-initial-solution-recenter-first-node-bounds --acados-disable-standard-ipopt-warmup \
    --disable-acados-assisted-hot-start --disable-periodic-fes-warmup-projection \
    --disable-full-dynamics-phase-one --disable-periodic-ipopt-refinement \
    --acados-ipopt-recovery --acados-ipopt-fallback-advance \
    --acados-ipopt-recovery-collocation-degree 5 --acados-ipopt-recovery-max-iterations 2000 \
    --retry-failed-rho-without-advance --max-consecutive-failing 1 \
    --initial-guess-diagnostics --acados-diagnostics --exact-initial-nlp-audit \
    --primal-feasibility-threshold 1e-5 \
    --output-json "$pilot_dir/$variant/result.json" \
    "${budget_options[@]}" > "$pilot_dir/$variant/solver.log" 2>&1
done
