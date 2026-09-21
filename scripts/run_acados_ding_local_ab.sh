#!/usr/bin/env bash
# Native ACADOS full/local A/B, same scientific input and isolated codegen.
set -euo pipefail
source .github/scripts/benchmark_env.sh rho32
export ACADOS_SOURCE_DIR="$CONDA_PREFIX"
windows="${1:-1}"
variant="${2:-local}"
destination="${3:-acados-ding-local-integration-20260913/$variant$windows}"
mkdir -p "$destination"
destination="$(realpath "$destination")"
extra=()
case "$variant" in local) extra=(--acados-ding-local-reduction);; baseline) ;; *) exit 2;; esac
extra+=("${@:4}")
python scripts/benchmark_acados_ding_local.py "$destination/native-map-audit.json" -- \
  --solvers acados --objective fatigue --benchmark-profile scientific-radau5 \
  --ipopt-use-sx --ipopt-linear-solver ma57 --warmup-ipopt-linear-solver ma57 \
  --cycles-per-window 1 --stimulations-per-cycle "${ACADOS_DING_STIMULATIONS:-50}" --n-windows "$windows" --n-threads 1 \
  --signed-crank-torque 0.1 --terminal-wheel-q-slack 0.002 \
  --mechanical-formulation reduced --formulation dynamic --state-scaling full \
  --reduced-internal-crank-velocity-guard "${ACADOS_DING_GUARD:-on}" --compact-rho-output \
  --common-initial-solution "${ACADOS_DING_SEED:-diagnostics-acados-cycle1-20260912/attempt35-50free-ipopt-guard-auto/common-cycle1.npz}" \
  --adopt-common-initial-solution-warmup-cycles --common-initial-solution-recenter-first-node-bounds \
  --reduced-cycling-profile benchmark-seed/reduced-cycling-fourier12.npz \
  --acados-dir "$CONDA_PREFIX" --experimental-reduced-acados \
  --acados-nlp-solver-type SQP --acados-integrator-type IRK \
  --acados-sim-stages 4 --acados-sim-steps 5 --acados-max-iter "${ACADOS_DING_MAX_ITER:-100}" \
  --codegen-tag "ding-local-$variant-$windows" \
  --acados-disable-standard-ipopt-warmup --disable-acados-assisted-hot-start \
  --disable-periodic-fes-warmup-projection --disable-full-dynamics-phase-one \
  --disable-periodic-ipopt-refinement --retry-failed-rho-without-advance --max-consecutive-failing 1 \
  --initial-guess-diagnostics --acados-diagnostics --exact-initial-nlp-audit \
  --primal-feasibility-threshold 1e-5 --output-json "$destination/result.json" \
  "${extra[@]}" 2>&1 | tee "$destination/solver.log"
