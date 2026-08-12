#!/usr/bin/env bash
set -euo pipefail

common_seed_path="$1"
assisted_hot_start="$2"
crank_assistance="$3"
terminal_wheel_q_slack="$4"
solver_max_iterations="$5"
fast_bound_margin="$6"
dropout_rho_list="$7"
followup_rhos="$8"

if [[ "$crank_assistance" != "signed:+0.15" ]]; then
  echo "acados_dropout requires signed:+0.15 N.m so it reuses the certified endurance problem." >&2
  exit 2
fi

IFS=',' read -r -a dropout_rhos <<< "$dropout_rho_list"
previous_rho=0
for rho in "${dropout_rhos[@]}"; do
  if [[ ! "$rho" =~ ^[0-9]+$ ]] || (( rho < 2 || rho <= previous_rho )); then
    echo "acados_dropout_rhos must be unique, increasing integers >= 2." >&2
    exit 2
  fi
  previous_rho="$rho"
done
if [[ ! "$followup_rhos" =~ ^[0-9]+$ ]] || (( followup_rhos < 1 )); then
  echo "acados_dropout_followup_rhos must be a positive integer." >&2
  exit 2
fi
reduced_rhos=$((previous_rho + followup_rhos))

assisted_hot_start_options=(--disable-acados-assisted-hot-start)
if [[ "$assisted_hot_start" == "true" ]]; then
  assisted_hot_start_options=(
    --acados-assisted-hot-start
    --acados-control-homotopy-release-final-radius
  )
fi

run_acados_dropout_case() {
  local case_name="$1"
  local iteration_cap="$2"
  local case_dir="acados-ipopt-hybrid-results/$case_name"
  local cap_options=()
  mkdir -p "$case_dir"
  if [[ "$iteration_cap" != "nominal" ]]; then
    cap_options=(
      --acados-forced-iteration-cap-rhos "$dropout_rho_list"
      --acados-forced-iteration-cap "$iteration_cap"
    )
  fi
  python examples/fes_multibody/cycling/cycling_fes_solver_comparison.py \
    --solvers acados \
    --objective fatigue \
    --cycles-per-window 1 \
    --stimulations-per-cycle 30 \
    --n-windows "$reduced_rhos" \
    --n-threads "$BENCHMARK_THREADS" \
    --crank-assistance "$crank_assistance" \
    --first-node-wheel-q-slack 0 \
    --terminal-wheel-q-slack "$terminal_wheel_q_slack" \
    --mechanical-formulation reduced \
    --experimental-reduced-acados \
    --compact-rho-output \
    --common-initial-solution "$common_seed_path" \
    --common-initial-solution-recenter-first-node-bounds \
    --adopt-common-initial-solution-warmup-cycles \
    "${assisted_hot_start_options[@]}" \
    --acados-disable-standard-ipopt-warmup \
    --periodic-ipopt-refinement-ode-solver collocation \
    --periodic-ipopt-refinement-collocation-degree 5 \
    --periodic-ipopt-refinement-collocation-method radau \
    --periodic-ipopt-refinement-iterations "$solver_max_iterations" \
    --warmup-ipopt-linear-solver mumps \
    --retry-failed-rho-without-advance \
    --max-consecutive-failing 2 \
    --acados-integrator-type IRK \
    --acados-sim-stages 4 \
    --acados-sim-steps 5 \
    --acados-wheel-qdot-fast-bound-margin "$fast_bound_margin" \
    --acados-initial-irk-rollout \
    --acados-max-iter 100 \
    --acados-store-iterates \
    --acados-ipopt-recovery \
    --acados-ipopt-recovery-max-iterations "$solver_max_iterations" \
    --acados-ipopt-recovery-collocation-degree 5 \
    --acados-ipopt-fallback-advance \
    --ipopt-linear-solver mumps \
    --validate-integrator-maps \
    --high-accuracy-trace-max-cycles 30 \
    --high-accuracy-trace-cycle-milestones "$dropout_rho_list" \
    --receding-horizon-solution-output "$case_dir/validated-prefix.npz" \
    --allow-partial-receding-horizon-solution-output \
    --output-json "$case_dir/result.json" \
    "${cap_options[@]}" \
    2>&1 | tee "$case_dir/solver.log"
}

run_acados_dropout_case nominal-100 nominal
run_acados_dropout_case cap-10 10
run_acados_dropout_case cap-20 20
run_acados_dropout_case cap-30 30
for cap in 10 20 30; do
  jq -e \
    --argjson requested "$reduced_rhos" \
    --argjson cap "$cap" \
    --arg rhos "$dropout_rho_list" '
    ($rhos | split(",") | map(tonumber)) as $targets |
    .configurations.acados.acados_forced_iteration_cap_rhos == $targets and
    .configurations.acados.acados_forced_iteration_cap == $cap and
    .results[0].error == null and
    .results[0].requested_cycles == $requested and
    (.results[0].acados_forced_iteration_cap_summaries | length) == ($targets | length) and
    all(.results[0].acados_forced_iteration_cap_summaries[];
        .target_rho as $rho |
        ($targets | index($rho)) != null and
        .iteration_cap == $cap and
        .nominal_budget_restored == true)
  ' "acados-ipopt-hybrid-results/cap-$cap/result.json"
done
jq -s '
  map({
    case: (if .configurations.acados.acados_forced_iteration_cap == null
           then "nominal-100"
           else "cap-\(.configurations.acados.acados_forced_iteration_cap)"
           end),
    validated_cycles: .results[0].validated_cycles,
    executed_fatigue_objective: .results[0].executed_fatigue_objective,
    fatigue_auc_cycles: .results[0].fatigue_auc_cycles,
    min_A_capacity_ratio: .results[0].min_A_capacity_ratio,
    forced_caps: .results[0].acados_forced_iteration_cap_summaries,
    fallback_count: .results[0].acados_ipopt_recovery.fallback_advanced_count,
    recovery_wall_time_s: .results[0].acados_ipopt_recovery.recovery_wall_time_s,
    end_to_end_wall_time_s: .results[0].end_to_end_wall_time_s,
    rho_pipeline_wall_time_per_cycle_s: .results[0].rho_pipeline_wall_time_per_cycle_s
  })
' acados-ipopt-hybrid-results/{nominal-100,cap-10,cap-20,cap-30}/result.json \
  > acados-ipopt-hybrid-results/dropout-summary.json
