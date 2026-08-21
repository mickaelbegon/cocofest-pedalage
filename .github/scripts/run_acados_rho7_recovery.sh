#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 6 ]]; then
  echo "usage: $0 SEED TORQUE TERMINAL_Q_SLACK IPOPT_MAX_ITER QDOT_MARGIN THREADS" >&2
  exit 2
fi

seed_path="$1"
resistive_torque="$2"
terminal_q_slack="$3"
ipopt_max_iter="$4"
qdot_margin="$5"
threads="$6"

if [[ "$resistive_torque" != "signed:+0.15" ]]; then
  echo "acados_rho7_recovery requires signed:+0.15 N.m." >&2
  exit 2
fi
if [[ "$qdot_margin" != "2.6" ]]; then
  echo "acados_rho7_recovery must reproduce the reference 2.6 rad/s guard." >&2
  exit 2
fi

root_dir=acados-ipopt-hybrid-results
baseline_dir="$root_dir/rho7-baseline"
recovery_dir="$root_dir/rho7-r5-recertification"
reference_dir="$root_dir/rho7-ipopt-r5-reference"
each_window_dir="$root_dir/rho7-r5-each-window-radau-iia"
mkdir -p "$baseline_dir" "$recovery_dir" "$reference_dir" "$each_window_dir"

common_options=(
  --solvers acados
  --objective fatigue
  --cycles-per-window 1
  --stimulations-per-cycle 30
  --n-threads "$threads"
  --crank-assistance "$resistive_torque"
  --first-node-wheel-q-slack 0
  --terminal-wheel-q-slack "$terminal_q_slack"
  --primal-feasibility-threshold 1e-5
  --mechanical-formulation reduced
  --experimental-reduced-acados
  --compact-rho-output
  --common-initial-solution-recenter-first-node-bounds
  --adopt-common-initial-solution-warmup-cycles
  --disable-acados-assisted-hot-start
  --acados-disable-standard-ipopt-warmup
  --disable-periodic-fes-warmup-projection
  --retry-failed-rho-without-advance
  --max-consecutive-failing 2
  --acados-integrator-type IRK
  --acados-sim-stages 4
  --acados-sim-steps 5
  --acados-wheel-qdot-fast-bound-margin "$qdot_margin"
  --acados-cyclical-transfer-mode extrapolate
  --acados-transfer-irk-rollout
  --acados-transfer-bound-homotopy
  --acados-transfer-bound-homotopy-fractions 0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1
  --acados-transfer-bound-homotopy-padding 0.05
  --acados-transfer-bound-homotopy-iterations 40
  --acados-transfer-bound-homotopy-tolerance 1e-4
  --acados-transfer-bound-homotopy-solver-tolerance 1e-4
  --acados-transfer-bound-homotopy-min-fraction-step 0.001953125
  --acados-transfer-bound-homotopy-max-refinements 16
  --acados-max-iter 100
  --acados-stationarity-tolerance 5e-3
  --acados-control-homotopy-radii 1e-6,1e-5
  --acados-control-homotopy-tolerance 2e-2
  --acados-control-homotopy-window-growth 10
  --acados-control-homotopy-window-max-radius 1e-5
  --acados-transfer-pulse-width-trust-radius 1e-5
  --ipopt-linear-solver mumps
)

# First reproduce the chain that exposed the resistant RHO-7 failure. The
# checkpoint is written only after six independently certified shifts, before
# the target-RHO-7 solve. A failed terminal can therefore never enter it.
python examples/fes_multibody/cycling/cycling_fes_solver_comparison.py \
  "${common_options[@]}" \
  --n-windows 7 \
  --common-initial-solution "$seed_path" \
  --rho-prepared-checkpoint-output-template \
  "$baseline_dir/prepared-after-{completed_windows}-for-{target_rho}.npz" \
  --rho-prepared-checkpoint-windows 6 \
  --receding-horizon-solution-output "$baseline_dir/validated-prefix.npz" \
  --allow-partial-receding-horizon-solution-output \
  --output-json "$baseline_dir/result.json" \
  2>&1 | tee "$baseline_dir/solver.log"

checkpoint="$baseline_dir/prepared-after-6-for-7.npz"
test -s "$checkpoint"
jq -e '
  .configurations.acados.crank_torque_role == "resistive" and
  .configurations.acados.constant_crank_torque == 0.15 and
  .results[0].validated_cycles >= 6 and
  any(.results[0].rho_prepared_checkpoints[];
      .completed_windows == 6 and .target_rho == 7)
' "$baseline_dir/result.json"

# The prepared checkpoint above is retained for audit, but its ACADOS rollout
# has already been projected onto moving bounds. Rebuild the recovery seed
# causally from the last certified cycle: preserve its terminal state as the
# new first node, repeat the phase-aligned PW/state profile, and shift theta by
# the measured signed revolution. This never consumes the failed RHO-7 primal.
repeat_checkpoint="$baseline_dir/repeat-after-6-for-7.npz"
python examples/fes_multibody/cycling/extract_rho_checkpoint.py \
  --source "$baseline_dir/validated-prefix.npz" \
  --output "$repeat_checkpoint" \
  --completed-windows 6 \
  --cycles-per-window 1 \
  --repeat-last-certified
test -s "$repeat_checkpoint"

# The repeat above is retained as the negative-control seed.  For recovery,
# continue every state by the measured boundary-to-boundary increment of the
# last certified cycle.  This preserves the phase-aligned within-cycle
# increments and removes the artificial terminal-to-old-node-1 discontinuity.
extrapolated_checkpoint="$baseline_dir/extrapolated-after-6-for-7.npz"
python examples/fes_multibody/cycling/extract_rho_checkpoint.py \
  --source "$baseline_dir/validated-prefix.npz" \
  --output "$extrapolated_checkpoint" \
  --completed-windows 6 \
  --cycles-per-window 1 \
  --extrapolate-last-certified
test -s "$extrapolated_checkpoint"

# Solver-neutral control experiment: solve the replay checkpoint directly
# with reduced IPOPT/Radau-5, before any ACADOS bound/target transfer.  This
# separates a genuinely infeasible replay formulation from corruption in the
# cross-solver recovery bridge.  Its outcome is diagnostic and must not skip
# the subsequent ACADOS recertification experiment.
python examples/fes_multibody/cycling/cycling_fes_solver_comparison.py \
  --solvers ipopt \
  --objective fatigue \
  --ipopt-profile periodic_collocation \
  --ipopt-use-sx \
  --ipopt-ode-solver collocation \
  --ipopt-collocation-degree 5 \
  --ipopt-collocation-method radau \
  --ipopt-max-iter "$ipopt_max_iter" \
  --ipopt-linear-solver mumps \
  --ipopt-disable-historical-initial-guess \
  --n-windows 1 \
  --cycles-per-window 1 \
  --stimulations-per-cycle 30 \
  --n-threads "$threads" \
  --crank-assistance "$resistive_torque" \
  --mechanical-formulation reduced \
  --wheel-qdot-bound-margin "$qdot_margin" \
  --first-node-wheel-q-slack 0 \
  --terminal-wheel-q-slack "$terminal_q_slack" \
  --state-scaling full \
  --common-initial-solution "$extrapolated_checkpoint" \
  --common-initial-solution-recenter-first-node-bounds \
  --adopt-common-initial-solution-warmup-cycles \
  --compact-rho-output \
  --output-json "$reference_dir/result.json" \
  2>&1 | tee "$reference_dir/solver.log"
test -s "$reference_dir/result.json"

# Determine whether the loss of recursive feasibility is accumulated by the
# ACADOS IRK transfer. R5 now refines every shifted primal before the next
# ACADOS solve. The previous ablation showed that a successful R5 refinement
# was immediately destroyed by the subsequent IRK rollout (omega reached
# +8.56 rad/s). Keep the R5 primal intact here; ACADOS itself remains the
# mandatory native recertifier. The first preserved-primal run reached two
# certified RHO but then stalled with a 1.12e-3 IRK dynamics residual while
# the R5 primal itself had inf_pr near 1e-9. Widening the PW trust region from
# +/-10 to +/-50 us did not advance farther and increased the residual to
# 2.95e-2. Restore +/-10 us and instead align the one-step ACADOS IRK tableau
# with the SX/Radau-5 collocation seed: Radau IIA, five stages, one integration
# step per shooting interval. This isolates transcription mismatch without
# changing the OCP or using the unsupported Bioptim SX+IRK bridge.
refined_common_options=()
for option in "${common_options[@]}"; do
  case "$option" in
    --acados-transfer-irk-rollout|--acados-transfer-bound-homotopy)
      ;;
    *)
      refined_common_options+=("$option")
      ;;
  esac
done
python examples/fes_multibody/cycling/cycling_fes_solver_comparison.py \
  "${refined_common_options[@]}" \
  --n-windows 7 \
  --common-initial-solution "$seed_path" \
  --periodic-ipopt-refinement \
  --periodic-ipopt-refinement-each-window \
  --periodic-ipopt-refinement-iterations "$ipopt_max_iter" \
  --periodic-ipopt-refinement-use-sx \
  --periodic-ipopt-refinement-ode-solver collocation \
  --periodic-ipopt-refinement-collocation-degree 5 \
  --periodic-ipopt-refinement-collocation-method radau \
  --acados-reset-solver-before-solve \
  --acados-collocation-type GAUSS_RADAU_IIA \
  --acados-sim-stages 5 \
  --acados-sim-steps 1 \
  --receding-horizon-solution-output "$each_window_dir/validated-prefix.npz" \
  --allow-partial-receding-horizon-solution-output \
  --output-json "$each_window_dir/result.json" \
  2>&1 | tee "$each_window_dir/solver.log"
test -s "$each_window_dir/result.json"
jq '{
  validated_cycles: .results[0].validated_cycles,
  outcome: .results[0].fatigue_endurance_outcome.label,
  refinements: .results[0].inter_window_refinement_summaries
}' "$each_window_dir/result.json" > "$each_window_dir/summary.json"

# Force IPOPT/Radau-5 on the frozen checkpoint, inject only its certified
# primal, reset ACADOS memory, then require ACADOS itself to advance target
# RHO 1 of the replay (the physical RHO 7). Fallback advance is intentionally
# absent: this is a recertification experiment, not the production hybrid.
python examples/fes_multibody/cycling/cycling_fes_solver_comparison.py \
  "${common_options[@]}" \
  --n-windows 2 \
  --common-initial-solution "$extrapolated_checkpoint" \
  --disable-periodic-ipopt-refinement \
  --warmup-ipopt-linear-solver mumps \
  --acados-ipopt-recovery \
  --acados-ipopt-recovery-max-iterations "$ipopt_max_iter" \
  --acados-ipopt-recovery-collocation-degree 5 \
  --acados-ipopt-recovery-seed-collocation-degree 3 \
  --acados-ipopt-recovery-seed-max-iterations "$ipopt_max_iter" \
  --acados-ipopt-recovery-irk-seed-audit \
  --acados-ipopt-recovery-force-first-rho \
  --ipopt-linear-solver mumps \
  --receding-horizon-solution-output "$recovery_dir/validated-prefix.npz" \
  --allow-partial-receding-horizon-solution-output \
  --output-json "$recovery_dir/result.json" \
  2>&1 | tee "$recovery_dir/solver.log"

jq -e '
  .configurations.acados.crank_torque_role == "resistive" and
  .configurations.acados.constant_crank_torque == 0.15 and
  .results[0].validated_cycles >= 1 and
  .results[0].acados_ipopt_recovery.fallback_advance_enabled == false and
  any(.results[0].acados_ipopt_recovery_summaries[];
      .forced_for_ci == true and
      .recovery_role == "seed_only" and
      .collocation_degree == 3 and
      .seed_source ==
        "last_certified_prepared_rho_primal_projected_to_current_bounds" and
      .recovery_seed_audit.finite == true and
      .fallback_advanced == false) and
  all(.results[0].acados_ipopt_recovery_summaries[];
      if .recovery_role == "certifying_fallback_candidate"
      then .collocation_degree == 5
      else true
      end) and
  any(.results[0].acados_ipopt_recovery_summaries[];
      .quality == "converged" and
      .feasibility.passes_tolerance == true and
      .seed_injected == true and
      .fallback_advanced == false) and
  any(.results[0].solver_attempt_accounting.attempts[];
      .target_rho == 1 and .advanced == true)
' "$recovery_dir/result.json"
