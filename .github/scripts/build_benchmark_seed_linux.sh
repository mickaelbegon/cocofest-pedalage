#!/usr/bin/env bash
set -euo pipefail

# Builds the three seed artefacts every cycling benchmark depends on:
#   benchmark-seed/common-reduced.npz
#   benchmark-seed/common-full.npz
#   benchmark-seed/reduced-cycling-fourier12.npz
#
# This is section 10 of docs/cycling_solver_benchmark/linux_32core_setup.md,
# packaged so it can be re-run without pasting a shell function. Run it in the
# IPOPT/FATROP/ACADOS environment (cocofest-rho32) with the numerical libraries
# pinned to a single thread.

workspace="${GITHUB_WORKSPACE:-${COCOFEST_ROOT:-$(git rev-parse --show-toplevel)}}"
cd "$workspace"

export GITHUB_WORKSPACE="$workspace"
export PYTHONPATH="$workspace${PYTHONPATH:+:$PYTHONPATH}"
if [[ -z "${BENCHMARK_THREADS:-}" ]]; then
  BENCHMARK_THREADS="$(python -c '
import sys
sys.path.insert(0, sys.argv[1])
from run_benchmarks import default_worker_threads
print(default_worker_threads())
' "$workspace/.github/scripts" 2>/dev/null || true)"
  if ! [[ "$BENCHMARK_THREADS" =~ ^[1-9][0-9]*$ ]]; then
    BENCHMARK_THREADS="$(nproc)"
  fi
fi
export BENCHMARK_THREADS
export OMP_NUM_THREADS=1
export OMP_THREAD_LIMIT=1
export OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate cocofest-rho32 before building the seed." >&2
  exit 1
fi
command -v jq >/dev/null || { echo "jq is required." >&2; exit 1; }

mkdir -p benchmark-seed-result benchmark-seed

prepare_seed() {
  local mechanics="$1"
  local mechanics_options=()
  if [[ "$mechanics" == reduced ]]; then
    mechanics_options+=(--mechanical-formulation reduced)
  else
    mechanics_options+=(
      --common-initial-solution "$workspace/benchmark-seed-result/common-reduced.npz"
    )
  fi

  echo "=== building '$mechanics' seed with BENCHMARK_THREADS=$BENCHMARK_THREADS ==="
  python examples/fes_multibody/cycling/cycling_fes_solver_comparison.py \
    --solvers ipopt \
    --objective fatigue \
    --ipopt-profile periodic_collocation \
    --ipopt-use-sx \
    --ipopt-enforce-start-constraints \
    --cycles-per-window 1 \
    --stimulations-per-cycle 30 \
    --n-windows 1 \
    --n-threads "$BENCHMARK_THREADS" \
    --crank-assistance 0.00 \
    --standard-warmup-seed .github/benchmark-seeds/legacy-resistive-0p22-warmup.npz \
    --legacy-standard-warmup-seed-signed-torque 0.22 \
    --standard-warmup-seed-continuation \
    --warmup-ipopt-linear-solver mumps \
    --ipopt-linear-solver mumps \
    --ipopt-max-iter 2000 \
    --ipopt-disable-historical-initial-guess \
    --reduced-cycling-profile "$workspace/benchmark-seed-result/reduced-cycling-fourier12.npz" \
    --state-scaling full \
    --first-node-wheel-q-slack 0 \
    --terminal-wheel-q-slack 0.002 \
    --compact-rho-output \
    --print-traces \
    --common-initial-solution-output "$workspace/benchmark-seed-result/common-${mechanics}.npz" \
    --output-json "$workspace/benchmark-seed-result/seed-check-${mechanics}.json" \
    "${mechanics_options[@]}"

  jq -e '.results[0] | (.success == true and .attempted_windows == 1)' \
    "benchmark-seed-result/seed-check-${mechanics}.json" >/dev/null
  echo "=== '$mechanics' seed certified ==="
}

prepare_seed reduced
prepare_seed full

cp benchmark-seed-result/common-reduced.npz benchmark-seed/
cp benchmark-seed-result/common-full.npz benchmark-seed/
cp benchmark-seed-result/reduced-cycling-fourier12.npz benchmark-seed/

for artefact in common-reduced.npz common-full.npz reduced-cycling-fourier12.npz; do
  test -f "benchmark-seed/$artefact" || { echo "Missing benchmark-seed/$artefact" >&2; exit 1; }
done
echo "Seed ready:"
ls -la benchmark-seed/
