#!/usr/bin/env bash
set -euo pipefail

# Endurance benchmark for a dedicated Ryzen 9 5950X host.  Every case runs
# sequentially so wall times are not contaminated by another solver.  Positive
# signed torque opposes the nominal negative crank velocity and is therefore a
# physical resistance in the cycling convention used by Cocofest.

workspace="${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}"
output_root="${OUTPUT_ROOT:-$workspace/benchmark-results/ryzen5950x-resistance-sweep}"
max_rhos="${MAX_RHOS:-2000}"
rho_threads="${RHO_THREADS:-16}"
resistances_nm="${RESISTANCES_NM:-0.10 0.15 0.20}"
strategies="${STRATEGIES:-ipopt madnlp acados-ipopt}"
force_rerun="${FORCE_RERUN:-false}"

export GITHUB_WORKSPACE="$workspace"
export PYTHONPATH="$workspace${PYTHONPATH:+:$PYTHONPATH}"
export BENCHMARK_THREADS="$rho_threads"
export BENCHMARK_CYCLES_PER_WINDOW=1
export BENCHMARK_Q_SLACK="${BENCHMARK_Q_SLACK:-0.002}"
export BENCHMARK_MAX_ITER="${BENCHMARK_MAX_ITER:-2000}"
export BENCHMARK_CYCLES="$max_rhos"
export MADNLP_FAST_MAX_ITERATIONS="${MADNLP_FAST_MAX_ITERATIONS:-73}"
export MADNLP_FAST_MAX_WALL_TIME="${MADNLP_FAST_MAX_WALL_TIME:-20}"
export MADNLP_FIRST_MAX_ITERATIONS="${MADNLP_FIRST_MAX_ITERATIONS:-2000}"
export HIGH_ACCURACY_TRACE_MAX_CYCLES="${HIGH_ACCURACY_TRACE_MAX_CYCLES:-30}"
export HIGH_ACCURACY_TRACE_CYCLE_MILESTONES="${HIGH_ACCURACY_TRACE_CYCLE_MILESTONES:-100,300,600,1000,1500,2000}"

# Keep numerical-library parallelism at one thread.  --n-threads controls the
# stage/map parallelism; nested BLAS/OpenMP teams would oversubscribe the 16
# physical cores and invalidate comparisons, especially at RHO recoveries.
export OMP_NUM_THREADS="${NUMERIC_THREADS:-1}"
export OMP_THREAD_LIMIT="${NUMERIC_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${NUMERIC_THREADS:-1}"
export MKL_NUM_THREADS="${NUMERIC_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMERIC_THREADS:-1}"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-1}"

if ! [[ "$max_rhos" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_RHOS must be a strictly positive integer, got '$max_rhos'." >&2
  exit 2
fi
if ! [[ "$rho_threads" =~ ^[1-9][0-9]*$ ]]; then
  echo "RHO_THREADS must be a strictly positive integer, got '$rho_threads'." >&2
  exit 2
fi
if [[ "$force_rerun" != "true" && "$force_rerun" != "false" ]]; then
  echo "FORCE_RERUN must be true or false, got '$force_rerun'." >&2
  exit 2
fi
if [[ ! -f "$workspace/benchmark-seed/common-reduced.npz" ]]; then
  echo "Missing benchmark-seed/common-reduced.npz; follow linux_32core_setup.md first." >&2
  exit 2
fi

mkdir -p "$output_root"
configuration_tag="$(printf '%s' "$strategies" | tr ' /' '--')"

physical_cores="$(lscpu -p=CORE 2>/dev/null | awk -F, '!/^#/ {seen[$1]=1} END {print length(seen)}')"
logical_cpus="$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
if [[ -n "$physical_cores" && "$rho_threads" -gt "$physical_cores" ]]; then
  echo "WARNING: RHO_THREADS=$rho_threads exceeds $physical_cores physical cores ($logical_cpus logical CPUs)."
  echo "Compare 16 and 30 on a short calibration before using 30 for timing claims."
fi

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_sha=$(git -C "$workspace" rev-parse HEAD)"
  echo "hostname=$(hostname)"
  echo "max_rhos=$max_rhos"
  echo "rho_threads=$rho_threads"
  echo "numeric_threads=$OMP_NUM_THREADS"
  echo "resistances_nm=$resistances_nm"
  echo "strategies=$strategies"
  command -v lscpu >/dev/null 2>&1 && lscpu
} > "$output_root/host-and-run-configuration-${configuration_tag}.txt"

result_is_complete() {
  local result="$1"
  [[ -f "$result" ]] || return 1
  jq -e '
    .results[0] as $r |
    ($r.error == null) and
    (($r.validated_cycles == $r.requested_cycles) or
     ($r.fatigue_endurance_outcome.accepted == true) or
     (($r.first_failed_rho != null) and
      (([$r.maximum_consecutive_failures // 0,
         $r.maximum_consecutive_uncertified_attempts // 0,
         $r.maximum_consecutive_window_failures // 0] | max) >= 2)))
  ' "$result" >/dev/null
}

run_nlp_case() {
  local resistance="$1"
  local tag="$2"
  local solver="$3"
  local resistance_root="$output_root/$tag"
  local case_slug="${solver}-r5-fatigue-endurance"
  local result="$resistance_root/${case_slug}-reduced/result.json"
  if [[ "$force_rerun" == "false" ]] && result_is_complete "$result"; then
    echo "Skipping completed case: $tag/$solver"
    return
  fi
  mkdir -p "$resistance_root"
  export BENCHMARK_ASSISTANCE="signed:+$resistance"
  bash "$workspace/.github/scripts/run_cycling_benchmark_case.sh" \
    "$case_slug" "$solver" reduced mumps collocation "$resistance_root" \
    "$max_rhos" true sx none 5 scientific-radau5 auto true
}

run_acados_reference_seed() {
  local resistance="$1"
  local resistance_root="$2"
  local native_seed="$resistance_root/acados-reference-reduced/native-seed.npz"
  local result="$resistance_root/acados-reference-reduced/result.json"
  if [[ "$force_rerun" == "false" && -f "$native_seed" ]] && result_is_complete "$result"; then
    return
  fi
  local case_dir="$resistance_root/acados-reference-reduced"
  local codegen_dir
  mkdir -p "$case_dir"
  codegen_dir="$(mktemp -d "$case_dir/codegen.XXXXXX")"
  pushd "$codegen_dir" >/dev/null
  python "$workspace/examples/fes_multibody/cycling/cycling_fes_solver_comparison.py" \
    --solvers acados \
    --objective fatigue \
    --ipopt-profile periodic_collocation \
    --ipopt-use-sx \
    --ipopt-enforce-start-constraints \
    --cycles-per-window 1 \
    --stimulations-per-cycle 30 \
    --n-windows 1 \
    --n-threads "$rho_threads" \
    --crank-assistance "signed:+$resistance" \
    --primal-feasibility-threshold 1e-5 \
    --max-consecutive-failing 2 \
    --retry-failed-rho-without-advance \
    --standard-warmup-seed "$workspace/.github/benchmark-seeds/legacy-resistive-0p22-warmup.npz" \
    --legacy-standard-warmup-seed-signed-torque 0.22 \
    --standard-warmup-seed-continuation \
    --common-initial-solution "$workspace/benchmark-seed/common-reduced.npz" \
    --common-initial-solution-output "$native_seed" \
    --warmup-ipopt-linear-solver mumps \
    --ipopt-linear-solver mumps \
    --ipopt-disable-historical-initial-guess \
    --reduced-cycling-profile "$workspace/benchmark-seed/reduced-cycling-fourier12.npz" \
    --state-scaling full \
    --first-node-wheel-q-slack 0 \
    --terminal-wheel-q-slack "$BENCHMARK_Q_SLACK" \
    --mechanical-formulation reduced \
    --experimental-reduced-acados \
    --acados-dir "${ACADOS_SOURCE_DIR:-${CONDA_PREFIX:?CONDA_PREFIX is required for ACADOS}}" \
    --acados-check-reuse-possible \
    --acados-max-iter 100 \
    --acados-nlp-solver-type SQP \
    --acados-integrator-type IRK \
    --acados-collocation-type GAUSS_LEGENDRE \
    --acados-sim-stages 4 \
    --acados-sim-steps 5 \
    --acados-newton-iter 5 \
    --acados-stationarity-tolerance 5e-3 \
    --acados-control-homotopy-release-final-radius \
    --compact-rho-output \
    --print-traces \
    --codegen-tag "ryzen-${tag}-reference" \
    --output-json "$result" \
    2>&1 | tee "$case_dir/solver.log"
  popd >/dev/null
  result_is_complete "$result"
  [[ -f "$native_seed" ]]
}

run_acados_ipopt_case() {
  local resistance="$1"
  local tag="$2"
  local resistance_root="$output_root/$tag"
  local case_dir="$resistance_root/acados-ipopt-r5-reduced"
  local result="$case_dir/result.json"
  if [[ "$force_rerun" == "false" ]] && result_is_complete "$result"; then
    echo "Skipping completed case: $tag/acados-ipopt"
    return
  fi
  run_acados_reference_seed "$resistance" "$resistance_root"
  local codegen_dir
  mkdir -p "$case_dir"
  codegen_dir="$(mktemp -d "$case_dir/codegen.XXXXXX")"
  pushd "$codegen_dir" >/dev/null
  python "$workspace/examples/fes_multibody/cycling/cycling_fes_solver_comparison.py" \
    --solvers acados \
    --objective fatigue \
    --ipopt-profile periodic_collocation \
    --ipopt-use-sx \
    --ipopt-enforce-start-constraints \
    --cycles-per-window 1 \
    --stimulations-per-cycle 30 \
    --n-windows "$max_rhos" \
    --n-threads "$rho_threads" \
    --crank-assistance "signed:+$resistance" \
    --primal-feasibility-threshold 1e-5 \
    --max-consecutive-failing 2 \
    --retry-failed-rho-without-advance \
    --standard-warmup-seed "$workspace/.github/benchmark-seeds/legacy-resistive-0p22-warmup.npz" \
    --legacy-standard-warmup-seed-signed-torque 0.22 \
    --standard-warmup-seed-continuation \
    --common-initial-solution "$resistance_root/acados-reference-reduced/native-seed.npz" \
    --warmup-ipopt-linear-solver mumps \
    --ipopt-linear-solver mumps \
    --ipopt-disable-historical-initial-guess \
    --reduced-cycling-profile "$workspace/benchmark-seed/reduced-cycling-fourier12.npz" \
    --state-scaling full \
    --first-node-wheel-q-slack 0 \
    --terminal-wheel-q-slack "$BENCHMARK_Q_SLACK" \
    --mechanical-formulation reduced \
    --experimental-reduced-acados \
    --terminal-wheel-qdot-bound-margin 0.3 \
    --acados-terminal-wheel-qdot-homotopy-margins 3,2.5,2,1.5,1,0.5,0.3 \
    --acados-dir "${ACADOS_SOURCE_DIR:-${CONDA_PREFIX:?CONDA_PREFIX is required for ACADOS}}" \
    --acados-check-reuse-possible \
    --acados-max-iter 100 \
    --acados-nlp-solver-type SQP \
    --acados-integrator-type IRK \
    --acados-collocation-type GAUSS_LEGENDRE \
    --acados-sim-stages 4 \
    --acados-sim-steps 5 \
    --acados-newton-iter 5 \
    --acados-stationarity-tolerance 5e-3 \
    --acados-control-homotopy-release-final-radius \
    --acados-ipopt-recovery \
    --acados-ipopt-fallback-advance \
    --acados-ipopt-recovery-max-iterations "$BENCHMARK_MAX_ITER" \
    --acados-ipopt-recovery-collocation-degree 5 \
    --acados-ipopt-recovery-irk-seed-audit \
    --compact-rho-output \
    --print-traces \
    --codegen-tag "ryzen-${tag}-acados-ipopt" \
    --output-json "$result" \
    2>&1 | tee "$case_dir/solver.log"
  popd >/dev/null
  result_is_complete "$result"
}

for resistance in $resistances_nm; do
  if ! [[ "$resistance" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Resistance must be a non-negative decimal magnitude, got '$resistance'." >&2
    exit 2
  fi
  resistance_tag="resistance-$(printf '%s' "$resistance" | tr '.' 'p')Nm"
  echo "=== $resistance_tag ==="
  for strategy in $strategies; do
    case "$strategy" in
      ipopt) run_nlp_case "$resistance" "$resistance_tag" ipopt ;;
      madnlp) run_nlp_case "$resistance" "$resistance_tag" madnlp ;;
      acados-ipopt) run_acados_ipopt_case "$resistance" "$resistance_tag" ;;
      *) echo "Unknown strategy '$strategy'." >&2; exit 2 ;;
    esac
  done
done

summary_csv="$output_root/endurance-summary.csv"
echo "resistance_nm,strategy,n_threads,validated_cycles,attempted_rho,requested_cycles,endurance_outcome,first_failed_rho,total_solver_iterations,hot_wall_median_s,hot_wall_p90_s,solver_time_s,rho_loop_wall_time_s,rho_loop_wall_per_validated_cycle_s,recovery_attempt_count,recovery_wall_time_s,fallback_advanced_count,compiled_library_build_count,runtime_bounds_changed,compiled_source_reused,objective,executed_fatigue,fatigue_auc,min_capacity,biceps_capacity,delt_ant_capacity,delt_post_capacity,triceps_capacity" > "$summary_csv"
while IFS= read -r result; do
  relative="${result#"$output_root/"}"
  resistance="$(printf '%s' "$relative" | cut -d/ -f1 | sed -e 's/^resistance-//' -e 's/Nm$//' -e 's/p/./g')"
  case_dir_name="$(printf '%s' "$relative" | cut -d/ -f2)"
  case "$case_dir_name" in
    ipopt-*) strategy=ipopt ;;
    madnlp-*) strategy=madnlp-ipopt ;;
    acados-ipopt-*) strategy=acados-ipopt ;;
    *) strategy="$case_dir_name" ;;
  esac
  jq -r --arg resistance "$resistance" --arg strategy "$strategy" \
    --argjson n_threads "$rho_threads" '
    .results[0] |
    [
      $resistance,
      $strategy,
      $n_threads,
      .validated_cycles,
      .attempted_windows,
      .requested_cycles,
      .fatigue_endurance_outcome.label,
      .first_failed_rho,
      ([.solver_attempt_accounting.attempts[]?.iterations // 0] | add // null),
      .hot_wall_time_median_s,
      .hot_wall_time_p90_s,
      .solver_time_s,
      .execution_timing.rho_solve_loop_wall_time_s,
      (if .validated_cycles > 0 then
         (.execution_timing.rho_solve_loop_wall_time_s / .validated_cycles)
       else null end),
      ((.nlp_ipopt_recovery.attempt_count // 0) +
       (.acados_ipopt_recovery.attempt_count // 0)),
      (([.nlp_ipopt_recovery_summaries[]?.wall_time_s // 0] | add // 0) +
       (.acados_ipopt_recovery.recovery_wall_time_s // 0)),
      ((.nlp_ipopt_recovery.fallback_advanced_count // 0) + (.acados_ipopt_recovery.fallback_advanced_count // 0)),
      .compiled_nlp_reuse.compiled_library_build_count,
      .compiled_nlp_reuse.runtime_bounds_changed,
      .compiled_nlp_reuse.compiled_source_reused,
      .objective,
      .executed_fatigue_objective,
      .fatigue_auc_cycles,
      .min_A_capacity_ratio,
      ([.muscle_fatigue[]? | select(.muscle == "Biceps") | .final_capacity_ratio][0] // null),
      ([.muscle_fatigue[]? | select(.muscle == "Delt_ant") | .final_capacity_ratio][0] // null),
      ([.muscle_fatigue[]? | select(.muscle == "Delt_post") | .final_capacity_ratio][0] // null),
      ([.muscle_fatigue[]? | select(.muscle == "Triceps") | .final_capacity_ratio][0] // null)
    ] | @csv
  ' "$result" >> "$summary_csv"
done < <(find "$output_root" -mindepth 3 -maxdepth 3 -name result.json -type f ! -path '*/acados-reference-reduced/*' | sort)

echo "Benchmark complete: $summary_csv"
