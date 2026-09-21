#!/usr/bin/env python3
"""Run the existing dynamic IPOPT RHO benchmark with a PACE cost adapter.

Example: python scripts/run_rho_pace_benchmark.py --pace-journal run/pace.jsonl
--pace-config run/pace-config.json -- [cycling_fes_solver_comparison options]

The config has optional ``policy`` (RhoPaceConfig fields), ``initial_weights``
(mapping by model muscle name), and mandatory ``initial_weight_basis``. Omit
initial_weights for a uniform start; a physiological calibration must provide
strictly positive weights explicitly and record its provenance in the basis.
The standard benchmark result and this sidecar journal together describe the
arm. The benchmark's unweighted fatigue metrics remain comparison metrics.
"""

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
os.environ.setdefault("MPLBACKEND", "Agg")


def validate_benchmark_arguments(args, config):
    """Refuse modes whose objective integration is not verified by this adapter."""
    requirements = {
        "solvers": ("ipopt",), "single_shot": False, "objective": "fatigue",
        "objective_shape": "quadratic", "formulation": "dynamic",
        "mechanical_formulation": "reduced", "cycles_per_window": 1,
        "ipopt_c_compile": False, "compact_rho_output": True,
    }
    for name, expected in requirements.items():
        if getattr(args, name, None) != expected:
            raise ValueError(f"RHO-PACE requires {name}={expected!r}")
    if not 1 <= args.n_windows <= config.max_cycles:
        raise ValueError("n_windows must be within the PACE campaign cycle limit")
    torque = args.resistive_torque
    if torque is None or not 0 < torque < float("inf"):
        raise ValueError("An explicit finite positive --signed-crank-torque is required")
    return torque


def validate_weight_configuration(declared, config):
    basis = declared.get("initial_weight_basis")
    if not isinstance(basis, str) or not basis.strip():
        raise ValueError("Physiological weights require nonempty initial_weight_basis provenance")
    if not config.adaptation_enabled:
        weights = declared.get("initial_weights")
        if not isinstance(weights, dict) or not weights:
            raise ValueError("RHO-Physio requires explicit named initial_weights")
        for name, weight in weights.items():
            if (not isinstance(name, str) or not name.strip() or isinstance(weight, bool)
                    or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0):
                raise ValueError("RHO-Physio requires finite positive weights for named muscles")


class _NumericalMuscleSnapshot(SimpleNamespace):
    def post_stimulation_amplitude(self):
        return self.calcium_amplitude


class _NumericalSolutionSnapshot:
    def __init__(self, states, controls):
        self.states, self.controls = states, controls

    def decision_states(self, **_kwargs):
        return self.states

    def decision_controls(self, **_kwargs):
        return self.controls


def _evaluate_pace_snapshot(payload):
    """Spawn entry point: no live Solution, model, controller or OCP crosses processes."""
    from cocofest.optimization.rho_pace import RhoPaceConfig
    return _predictive_moment_proposal(
        solution=_NumericalSolutionSnapshot(payload["states"], payload["controls"]),
        models=tuple(_NumericalMuscleSnapshot(**model) for model in payload["models"]),
        parsed=SimpleNamespace(**payload["parsed"]),
        controller=SimpleNamespace(config=RhoPaceConfig(**payload["config"]), weights=payload["weights"]),
        cycle_index=payload["cycle_index"], journal_path=payload["journal_path"],
        archive_path=payload["archive_path"],
    )


def _predictive_moment_proposal(*, solution, models, parsed, controller, cycle_index, journal_path,
                                archive_path=None):
    """Evaluate the bounded slow rollout from one certified, in-memory RHO.

    The temporary archive is solely an adapter boundary: it gives the already
    validated RHO-to-rollout code its usual immutable input format.  It is
    removed before the next fast OCP is started.  No FHO solution, terminal
    value, or future RHO trajectory enters the proposal.
    """
    from dataclasses import asdict
    from hashlib import sha256
    from time import monotonic

    import numpy as np
    from cocofest.optimization.endurance_weight_supervisor import (
        EnduranceWeightSupervisor, SupervisorSnapshot, WeightSupervisorConfig,
    )
    from cocofest.optimization.rho_adaptive_moment_policy import build_rho_adaptive_moment_policy
    from cocofest.optimization.rho_rollout_adapter import select_certified_rho_cycle
    from cocofest.optimization.batched_weighted_cycle_prediction import BatchedWeightedCyclePredictor
    from cocofest.optimization.weighted_cycle_prediction import WeightedCyclePredictor

    if archive_path is None:
        handle = tempfile.NamedTemporaryFile(
            prefix=f".predictive-rho-{cycle_index}-", suffix=".npz", delete=False,
            dir=Path(journal_path).parent,
        )
        source = Path(handle.name)
        handle.close()
    else:
        source = Path(archive_path)
    try:
        # Only fields consumed by the archive adapter are declared here.  The
        # comparison CLI namespace intentionally exposes ``resistive_torque``
        # rather than the lower-level ``constant_crank_torque`` field used by
        # the generic seed serializer.
        metadata = {
            "model_formulation": "periodic_node",
            "calcium_forcing_formulation": "exact_exponential_periodic_node",
            "mechanical_formulation": parsed.mechanical_formulation,
            "formulation": parsed.formulation,
            "stimulations_per_cycle": int(parsed.stimulations_per_cycle),
            "producer_collocation_degree": int(parsed.ipopt_collocation_degree),
            "producer_collocation_method": str(parsed.ipopt_collocation_method),
            "pulse_width_maximum_s": 0.0006,
            "producer_solver": "ipopt",
            "signed_crank_torque_nm": float(parsed.resistive_torque),
            "producer_mode": "receding_horizon_concatenation",
            "cycles_per_window": 1,
            "producer_cycles_per_window": 1,
            "producer_requested_cycles": 1,
            "cycle_duration_s": 1.0,
            "slow_projection_source": "current_certified_rho_only",
        }
        if isinstance(solution, _NumericalSolutionSnapshot):
            states, controls = solution.states, solution.controls
        else:
            from bioptim import SolutionMerge
            states = solution.decision_states(to_merge=SolutionMerge.NODES)
            controls = solution.decision_controls(to_merge=SolutionMerge.NODES)
        np.savez(source, **{f"states__{name}": value for name, value in states.items()},
                 **{f"controls__{name}": value for name, value in controls.items()},
                 metadata__json=np.asarray(json.dumps(metadata, sort_keys=True)))
        task = build_rho_adaptive_moment_policy(
            source, parsed.reduced_cycling_profile, cycle_index=0, cycle_period=1.0,
            muscle_models=models,
        )
        cycle, _ = select_certified_rho_cycle(source, cycle_index=0, cycle_period=1.0)
        source.unlink(missing_ok=True)
        state_names = ("Cn", "F", "A", "Tau1", "Km")
        terminal = tuple(
            tuple(float(cycle.states[f"{component}_{name}"][cycle.end_column])
                  for component in state_names)
            for name in task.muscle_names
        )
        predictor = WeightedCyclePredictor(
            task.intervals, task.parameters,
            substeps=controller.config.projection_substeps,
            reference_regularization=1e-3,
        )
        context = {
            "source": "current_certified_rho_only",
            "cycle_index": int(cycle_index),
            "horizon_cycles": controller.config.projection_horizon_cycles,
            "stimulations_per_cycle": int(parsed.stimulations_per_cycle),
            "resistance_nm": float(parsed.resistive_torque),
        }
        snapshot = SupervisorSnapshot(
            task_id="online-rho-signed-moment",
            context_token=sha256(json.dumps(context, sort_keys=True).encode()).hexdigest(),
            cycle_index=int(cycle_index), start_time_s=float(cycle_index), created_at_s=monotonic(),
            horizon_cycles=controller.config.projection_horizon_cycles,
            muscle_names=task.muscle_names, state_component_names=state_names,
            start_state=terminal, incumbent_weights=controller.weights,
        )
        supervisor = EnduranceWeightSupervisor(WeightSupervisorConfig(
            min_weight=controller.config.min_relative_weight,
            max_weight=controller.config.max_relative_weight,
            adjustment_factor=controller.config.projection_adjustment_factor,
            muscle_count=len(task.parameters),
            selection_mode=("guarded_fatigue" if controller.config.projection_fatigue_guard
                            and controller.config.projection_tracking_mode == "projected_capacity" else
                            "full_horizon_deficit" if controller.config.projection_tracking_mode == "projected_capacity"
                            else "exact_prefix"),
            max_candidate_log_step=(controller.config.max_log_step if controller.config.projection_fatigue_guard else None),
            minimum_relative_improvement=controller.config.projection_minimum_relative_improvement,
            minimum_absolute_improvement=controller.config.projection_minimum_absolute_improvement,
        ))

        rollout_audits = []
        batched_results = None
        result_by_weights = None

        def evaluate(weights, context):
            nonlocal batched_results, result_by_weights
            if batched_results is None:
                declared_candidates = supervisor.candidates(snapshot)
                batched_results = BatchedWeightedCyclePredictor(predictor).rollout_many(
                    np.asarray(context.start_state),
                    [candidate.weights for candidate in declared_candidates],
                    horizon_cycles=context.horizon_cycles,
                    tracking_mode=controller.config.projection_tracking_mode,
                    score_block_cycles=controller.config.update_every_cycles,
                )
                result_by_weights = {
                    tuple(candidate.weights): result
                    for candidate, result in zip(declared_candidates, batched_results, strict=True)
                }
                rollout_audits.extend({
                    "weights": list(candidate.weights), "status": result.status,
                    "completed_cycles": result.completed_cycles,
                    "completed_intervals": result.completed_intervals,
                    "first_failure": result.first_failure, "metadata": result.metadata,
                } for candidate, result in zip(declared_candidates, batched_results, strict=True))
            return result_by_weights[tuple(weights)]

        proposal = supervisor.evaluate(
            snapshot, evaluate, budget_seconds=controller.config.effective_projection_budget_seconds,
        )
        evidence = proposal.chosen_evidence
        audit = {
            "strategy": ("predictive_moment_guarded_fatigue_v3" if controller.config.projection_fatigue_guard
                         else "predictive_moment_long_capacity_v2"),
            "selection_config": asdict(supervisor.config),
            "source": context["source"], "source_cycle_index": int(cycle_index),
            "horizon_cycles": context["horizon_cycles"],
            "update_every_cycles": controller.config.update_every_cycles,
            "tracking_mode": controller.config.projection_tracking_mode,
            "candidate_evaluation_backend": "numpy_batch_sequential_phases",
            "candidate_rollouts": rollout_audits,
            "candidate_evaluations": [asdict(item) for item in proposal.evaluations],
            "candidate_count_declared": len(supervisor.candidates(snapshot)),
            "candidate_count_evaluated": len(proposal.evaluations),
            "selection_basis": proposal.selection_basis,
            "has_actionable_evidence": proposal.has_actionable_evidence,
            "selected_candidate": (None if proposal.chosen_candidate is None
                                   else asdict(proposal.chosen_candidate)),
            "selected_evidence": (None if evidence is None else {
                "completed": evidence.completed, "feasible": evidence.feasible,
                "completed_duration": evidence.completed_duration,
                "minimum_signed_margin": evidence.minimum_signed_margin,
                "status": evidence.status,
                "full_horizon_normalized_deficit": evidence.full_horizon_normalized_deficit,
                "terminal_normalized_reserve": evidence.terminal_normalized_reserve,
                "full_horizon_mean_squared_fatigue": evidence.full_horizon_mean_squared_fatigue,
                "first_block_mean_squared_fatigue": evidence.first_block_mean_squared_fatigue,
                "terminal_minimum_capacity": evidence.terminal_minimum_capacity,
            }),
            "elapsed_s": proposal.elapsed_s, "budget_seconds": proposal.budget_seconds,
            "budget_exhausted": proposal.budget_exhausted,
        }
        return (proposal.weights if proposal.has_actionable_evidence else None), audit
    finally:
        source.unlink(missing_ok=True)


def main(argv=None, *, adaptation_enabled=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pace-journal", type=Path, required=True)
    parser.add_argument("--pace-config", type=Path, required=True)
    parser.add_argument("benchmark_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    # The benchmark changes cwd to its example directory. Bind all PACE
    # paths to the caller's cwd before entering it, just as it does for its
    # own output paths; otherwise a successful run hides the sidecar there.
    caller_directory = Path.cwd()
    args.pace_config = args.pace_config.expanduser().resolve()
    args.pace_journal = args.pace_journal.expanduser().resolve()
    benchmark_argv = args.benchmark_args
    if benchmark_argv[:1] == ["--"]:
        benchmark_argv = benchmark_argv[1:]

    from bioptim import SolutionMerge
    from cocofest.optimization.fes_nmpc_multibody import FesNmpcMsk
    from cocofest.optimization.rho_pace import (
        RhoPaceConfig, RhoPaceController, update_bioptim_fatigue_cost,
    )
    from cocofest.optimization.rho_pace_async import AsyncPaceWorker
    from examples.fes_multibody.cycling import cycling_fes_solver_comparison as benchmark
    from examples.fes_multibody.cycling.cycling_pulse_width_mhe_acados_periodic import (
        _rho_solution_is_certified,
    )

    declared = json.loads(args.pace_config.read_text(encoding="utf-8"))
    unexpected = set(declared) - {"policy", "initial_weights", "initial_weight_basis", "calibration"}
    if unexpected:
        raise ValueError(f"Unknown PACE configuration fields: {sorted(unexpected)}")
    if "calibration" in declared and not isinstance(declared["calibration"], dict):
        raise ValueError("PACE calibration evidence must be an object when supplied")
    policy = dict(declared.get("policy", {}))
    if adaptation_enabled is not None:
        policy["adaptation_enabled"] = adaptation_enabled
    config = RhoPaceConfig(**policy)
    validate_weight_configuration(declared, config)
    parsed = benchmark.build_cli().parse_args(benchmark_argv)
    # The benchmark changes into its example directory before calling the
    # cyclic callback.  Snapshot construction starts from that callback, so retain
    # the caller-resolved reduced profile rather than a now-invalid relative
    # CLI spelling.
    if parsed.reduced_cycling_profile is not None:
        reduced_profile = Path(parsed.reduced_cycling_profile).expanduser()
        parsed.reduced_cycling_profile = (
            reduced_profile if reduced_profile.is_absolute()
            else (caller_directory / reduced_profile).resolve()
        )
    torque = validate_benchmark_arguments(parsed, config)
    original = FesNmpcMsk.solve_fes_nmpc
    controllers = []

    def solve_with_pace(nmpc, update_functions, *positional, **kwargs):
        if controllers:
            raise RuntimeError("Multiple RHO sessions require separate PACE journals")
        models = nmpc.nlp[0].model.muscles_dynamics_model
        names = tuple(model.muscle_name for model in models)
        weights = declared.get("initial_weights", dict.fromkeys(names, 1.0))
        if set(weights) != set(names):
            raise ValueError("initial_weights must match actual OCP muscle names exactly")
        parameter_names = ("a_scale", "alpha_a", "alpha_tau1", "alpha_km", "tau_fat",
                           "tau1_rest", "km_rest", "tauc", "tau2", "pd0", "pdt")
        parameters = {m.muscle_name: {key: float(getattr(m, key)) for key in parameter_names}
                      for m in models}
        controller = RhoPaceController(
            names, [weights[name] for name in names], signed_crank_torque_nm=torque,
            parameters=parameters, initial_weight_basis=declared["initial_weight_basis"],
            config=config, journal_path=args.pace_journal,
        )
        controller._record({"event": "benchmark_arguments", "arguments": benchmark_argv,
                            "resolved_benchmark_arguments": json.loads(json.dumps(vars(parsed), default=str)),
                            "policy": asdict(config), "initial_seed_certification":
                            "delegated_to_original_benchmark", "initial_weight_calibration":
                            declared.get("calibration"), "comparison_metrics":
                            "original_unweighted_fatigue_metrics"})
        controllers.append(controller)
        worker = (AsyncPaceWorker(
            _evaluate_pace_snapshot, horizon_cycles=config.projection_horizon_cycles,
            budget_seconds=config.effective_projection_budget_seconds,
            max_age_cycles=config.update_every_cycles,
            cpu_ids=config.projection_worker_cpu_ids,
        ) if config.adaptation_enabled and config.adaptation_strategy == "predictive_moment"
             and config.projection_async else None)
        archive_path = None

        def callback(ocp, cycle_index, solution, **extra):
            nonlocal archive_path
            continue_solving = update_functions(ocp, cycle_index, solution, **extra)
            if solution is not None:
                solution = getattr(solution, "_cocofest_fallback_solution", None) or solution
                feasibility = getattr(solution, "_cocofest_feasibility_summary", None)
                controller._record({"event": "completed_window", "cycle_index": int(cycle_index) - 1,
                                    "solver_status": int(solution.status),
                                    "physical_tolerance_passed": bool(
                                        feasibility and feasibility.get("passes_tolerance", False)),
                                    "certified": _rho_solution_is_certified(solution.status, feasibility)})
            if not continue_solving or cycle_index >= config.max_cycles:
                controller._record({"event": "stopped", "cycle_index": cycle_index,
                                    "reason": "benchmark_stopped" if not continue_solving
                                    else "maximum_cycles_reached"})
                return False
            if solution is not None:
                certified = _rho_solution_is_certified(
                    solution.status, getattr(solution, "_cocofest_feasibility_summary", None))
                states = solution.decision_states(to_merge=SolutionMerge.NODES)
                ratios = [float(states[f"A_{m.muscle_name}"][0, -1]) / m.a_scale for m in models]
            else:
                # First solve has no completed source; original seed validation
                # and all original initial bounds are retained.
                certified = True
                ratios = [float(ocp.nlp[0].x_bounds[f"A_{m.muscle_name}"].min[0, 0]) / m.a_scale
                          for m in models]
            predictive_weights = None
            predictive_audit = None
            if worker is not None:
                predictive_weights, predictive_audit = worker.poll(
                    cycle_index=int(cycle_index), incumbent_weights=controller.weights,
                )
                if predictive_audit is not None:
                    controller._record({"event": "projection_completed", **predictive_audit})
                    if archive_path is not None:
                        archive_path.unlink(missing_ok=True)
                        archive_path = None
            if (
                solution is not None and certified and controller.connected and config.adaptation_enabled
                and config.adaptation_strategy == "predictive_moment"
                and int(cycle_index) % config.update_every_cycles == 0
                and worker is None
            ):
                try:
                    predictive_weights, predictive_audit = _predictive_moment_proposal(
                        solution=solution, models=models, parsed=parsed, controller=controller,
                        cycle_index=int(cycle_index), journal_path=args.pace_journal,
                    )
                except Exception as error:
                    # Predictive failure must hold the last certified cost; it
                    # must never fall back silently to the capacity heuristic.
                    predictive_audit = {
                        "strategy": "predictive_moment_v1", "source_cycle_index": int(cycle_index),
                        "has_actionable_evidence": False,
                        "error": f"{type(error).__name__}: {error}",
                    }
            event = controller.boundary(
                int(cycle_index), ratios, certified=certified,
                signed_crank_torque_nm=torque,
                apply_weights=lambda values: update_bioptim_fatigue_cost(ocp, values),
                predictive_weights=predictive_weights, predictive_audit=predictive_audit,
            )
            if event["status"] == "refused":
                # A stopped arm must not silently become baseline RHO.
                return False
            if (worker is not None and solution is not None and certified and controller.connected
                    and int(cycle_index) % config.update_every_cycles == 0 and not worker.busy):
                import numpy as np
                from uuid import uuid4
                archive_path = args.pace_journal.parent / f".predictive-rho-{uuid4().hex}.npz"
                payload = {
                    "states": {name: np.asarray(value).copy() for name, value in states.items()},
                    "controls": {name: np.asarray(value).copy() for name, value in
                                 solution.decision_controls(to_merge=SolutionMerge.NODES).items()},
                    "models": [{**parameters[m.muscle_name], "muscle_name": m.muscle_name,
                                "calcium_amplitude": float(m.post_stimulation_amplitude())} for m in models],
                    "parsed": vars(parsed).copy(), "weights": controller.weights,
                    "config": {**asdict(config), "projection_horizon_cycles": worker.horizon_cycles,
                               "projection_budget_seconds": worker.budget_seconds},
                    "cycle_index": int(cycle_index), "journal_path": str(args.pace_journal),
                    "archive_path": str(archive_path),
                }
                try:
                    worker.submit(payload, cycle_index=int(cycle_index), incumbent_weights=controller.weights)
                except Exception as error:
                    controller._record({"event": "projection_submit_failed", "cycle_index": int(cycle_index),
                                        "error": f"{type(error).__name__}: {error}"})
                else:
                    controller._record({"event": "projection_submitted", "cycle_index": int(cycle_index),
                                        "asynchronous": True, "horizon_cycles": worker.horizon_cycles,
                                        "budget_seconds": worker.budget_seconds,
                                        "worker_cpu_ids": worker.cpu_ids})
            return True

        try:
            return original(nmpc, callback, *positional, **kwargs)
        finally:
            if worker is not None:
                worker.close()
            if archive_path is not None:
                archive_path.unlink(missing_ok=True)

    FesNmpcMsk.solve_fes_nmpc = solve_with_pace
    try:
        # Execute the already imported module, avoiding a second __main__
        # module identity. The public parser and main have identical keys.
        from examples.fes_multibody.cycling.cycling_pulse_width_mhe import MyCyclicNMPC
        if MyCyclicNMPC.solve_fes_nmpc is not solve_with_pace:
            raise RuntimeError("Cycling NMPC does not inherit the PACE integration")
        benchmark.main(**vars(parsed))
        if (len(controllers) != 1 or not controllers[0].connected
                or not args.pace_journal.is_file() or args.pace_journal.stat().st_size == 0):
            raise RuntimeError("RHO benchmark did not produce a connected PACE cost and journal")
        for controller in controllers:
            controller._record({"event": "launcher_completed", "ocp_cost_connected": controller.connected,
                                "physical_outcome": "see_original_benchmark_result"})
    except BaseException as error:
        for controller in controllers:
            controller._record({"event": "launcher_failed", "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        os.chdir(caller_directory)
        FesNmpcMsk.solve_fes_nmpc = original


if __name__ == "__main__":
    main()
