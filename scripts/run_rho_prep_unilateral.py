#!/usr/bin/env python3
"""Pilot a two-regime, patient-state-conditioned unilateral RHO policy.

The first regime uses a fixed-weight RHO-BO incumbent.  Once a certified
cycle crosses a capacity threshold, a single Physio-U work/fatigability update
sets the second-regime weights.  The numerical fatigue weights are parameters
of the compiled NLP, so the update must not rebuild its objective graph.

This is a research pilot, not a clinical controller or a physiological
failure detector.  A terminal IPOPT failure remains indeterminate unless the
standard frozen-state counterfactual supplies its own certificate.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
from types import MethodType

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_BO_WEIGHTS = {
    "Delt_ant": 0.05122998807382126,
    "Delt_post": 0.204711211458898,
    "Biceps": 0.13112644328025347,
    "Triceps": 0.954972066693792,
}


def _positive_weights(weights, muscle_names):
    values = np.asarray([weights[name] for name in muscle_names], dtype=float)
    if values.shape != (len(muscle_names),) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Every named muscle must have a finite, positive weight.")
    return values


def _geometric_mean_one(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Weights must be a nonempty finite positive vector.")
    return np.exp(np.log(values) - np.mean(np.log(values)))


def _scheduled_update_due(completed_cycles, update_every_cycles):
    """Whether a state-triggered update may be evaluated on this cycle."""
    if update_every_cycles < 1:
        raise ValueError("update_every_cycles must be at least 1")
    return completed_cycles % update_every_cycles == 0


def install_ipopt_ding_local_reduction(audit_directory):
    """Use the audited Radau-5 local Ding adapter in the direct RHO driver.

    This process-local hook leaves Bioptim/Cocofest sources untouched and
    restores the original solver factories after ``solve_case`` returns.
    """
    from scripts.benchmark_ding_radau5_local_ab import LocalSolver
    import bioptim.interfaces.interface_utils as interface_utils
    import bioptim.interfaces.ipopt_interface as ipopt_interface

    original_nlpsol = interface_utils.nlpsol
    original_generic_solve = ipopt_interface.generic_solve
    active = {}
    counter = {"value": 0}
    output = Path(audit_directory)

    def generic_solve(interface, expand_during_shake_tree=False):
        if getattr(interface.opts, "c_compile", False) or getattr(interface.opts, "function_transform", False):
            raise ValueError("Local Ding reduction requires interpreted SX IPOPT evaluators.")
        active["interface"] = interface
        return original_generic_solve(interface, expand_during_shake_tree)

    def nlpsol(name, plugin, nlp, options):
        if plugin != "ipopt":
            raise ValueError(f"Local Ding reduction expected IPOPT, received {plugin!r}.")
        if "interface" not in active:
            raise RuntimeError("Bioptim did not expose the active IPOPT interface before nlpsol creation.")
        solver = LocalSolver(
            original_nlpsol, name, plugin, nlp, options, active["interface"],
            "condensed", output, counter["value"],
        )
        counter["value"] += 1
        return solver

    previous_journal = os.environ.get("DING_ONLINE_JOURNAL")
    os.environ["DING_ONLINE_JOURNAL"] = "1"
    ipopt_interface.generic_solve = generic_solve
    interface_utils.nlpsol = nlpsol

    def restore():
        ipopt_interface.generic_solve = original_generic_solve
        interface_utils.nlpsol = original_nlpsol
        if previous_journal is None:
            os.environ.pop("DING_ONLINE_JOURNAL", None)
        else:
            os.environ["DING_ONLINE_JOURNAL"] = previous_journal

    return restore


def _capacity_ratios(solution, muscle_models):
    from bioptim import SolutionMerge

    states = solution.decision_states(to_merge=SolutionMerge.NODES)
    ratios = {
        model.muscle_name: float(np.asarray(states[f"A_{model.muscle_name}"]).reshape(-1)[-1] / model.a_scale)
        for model in muscle_models
    }
    if not ratios or not all(math.isfinite(value) and value > 0 for value in ratios.values()):
        raise ValueError("Certified terminal capacities must be finite and positive.")
    return states, ratios


def install_two_regime_policy(runtime, *, switch_min_capacity_ratio: float,
                              update_config: dict, log_events: list[dict], progress_path: Path):
    """Switch once on a certified capacity state without rebuilding the NLP."""
    nmpc = runtime["nmpc"]
    args = runtime["args"]
    binding = getattr(nmpc, "fatigue_weight_binding", None)
    if binding is None:
        raise ValueError("RHO-PREP requires parametric_fatigue_weights=true.")
    models = tuple(nmpc.nlp[0].model.muscles_dynamics_model)
    muscle_names = tuple(model.muscle_name for model in models)
    initial_weights = _geometric_mean_one(binding.weights)
    binding.update(nmpc, initial_weights)
    original_advance = nmpc.advance_window
    state = {
        "switched": False,
        "threshold_reached": False,
        "completed_cycles": 0,
        "compiled_nlp": id(nmpc.nlp[0]),
    }

    def write_progress():
        payload = {
            "kind": "rho_prep_unilateral_pilot_progress",
            "policy": "rho_prep_two_regime_physio_update_work_v1_experimental",
            "completed_cycles": state["completed_cycles"],
            "switch_occurred": state["switched"],
            "events": log_events,
            "compiled_nlp_reused": id(nmpc.nlp[0]) == state["compiled_nlp"],
        }
        temporary = progress_path.with_suffix(progress_path.suffix + ".tmp")
        temporary.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(progress_path)

    def advance_and_maybe_switch(self, solution, *advance_args, **advance_kwargs):
        result = original_advance(solution, *advance_args, **advance_kwargs)
        if solution is None:
            return result
        feasibility = getattr(solution, "_cocofest_feasibility_summary", {})
        from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as driver
        certified = driver._rho_solution_is_certified(solution.status, feasibility)
        if not certified:
            return result
        state["completed_cycles"] += 1
        if state["switched"]:
            return result
        states, capacity_ratios = _capacity_ratios(solution, models)
        minimum_ratio = min(capacity_ratios.values())
        if minimum_ratio <= switch_min_capacity_ratio:
            state["threshold_reached"] = True
        if not state["threshold_reached"]:
            return result

        # Latch the state trigger, but build the PW-max envelope only on the
        # configured cadence. A crossing at cycle 392 with a 20-cycle cadence
        # must wait for cycle 400 instead of consuming the only transition.
        if not _scheduled_update_due(
            state["completed_cycles"], update_config["update_every_cycles"],
        ):
            return result

        from cocofest.optimization.physio_update import (
            PhysioUpdateConfig,
            build_isokinetic_max_pw_envelope_from_discrete_cycle,
            propose_physio_update,
        )

        config = PhysioUpdateConfig(**update_config)
        envelope_states = dict(states)
        theta_source = "theta"
        if "theta" not in envelope_states:
            model = nmpc.nlp[0].model
            if not getattr(model, "uses_prescribed_kinematics", False):
                raise ValueError(f"Cannot recover crank angle from certified state keys {sorted(envelope_states)}.")
            reference = next(iter(envelope_states.values()))
            column_count = np.asarray(reference).reshape(-1).size
            stimulation_count = int(args.stimulations_per_cycle)
            if (column_count - 1) % stimulation_count:
                raise ValueError("Certified state nodes do not partition into the prescribed stimulation phases.")
            nodes_per_phase = (column_count - 1) // stimulation_count
            phase_duration = (2.0 * np.pi / abs(float(args.isokinetic_omega))) / stimulation_count
            phase_angles = float(model.isokinetic_theta0) + float(args.isokinetic_omega) * phase_duration * np.arange(
                stimulation_count + 1, dtype=float,
            )
            theta_values = np.interp(
                np.arange(column_count, dtype=float),
                np.arange(stimulation_count + 1, dtype=float) * nodes_per_phase,
                phase_angles,
            )
            envelope_states["theta"] = theta_values
            theta_source = "prescribed_theta0_plus_omega_t_at_phase_endpoints"
        live = build_isokinetic_max_pw_envelope_from_discrete_cycle(
            states=envelope_states,
            muscle_models=models,
            reduced_dynamics=nmpc.nlp[0].model.reduced_dynamics,
            stimulations_per_cycle=int(args.stimulations_per_cycle),
            angular_velocity_rad_s=float(args.isokinetic_omega),
            activate_force_length=bool(getattr(nmpc.nlp[0].model, "activate_force_length_relationship", True)),
            activate_force_velocity=bool(getattr(nmpc.nlp[0].model, "activate_force_velocity_relationship", True)),
            activate_passive_force=bool(getattr(nmpc.nlp[0].model, "activate_passive_force_relationship", True)),
        )
        proposal = propose_physio_update(
            completed_cycles=state["completed_cycles"],
            certified=True,
            incumbent_weights=binding.weights,
            **{key: live[key] for key in (
                "current_capacity", "rest_capacity", "alpha_a", "tau_fat", "reference_force",
                "available_positive_power", "phase_durations", "required_active_work",
            )},
            config=config,
        )
        proposal_weights = np.asarray(proposal["weights"], dtype=float)
        before = binding.weights.copy()
        if proposal.get("status") == "proposed":
            update = binding.update(nmpc, proposal_weights)
            state["switched"] = True
        else:
            update = {"ocp_cost_updated": False, "reason": proposal.get("reason", "held")}
            # A due update explicitly held (for example inside the deadband) is
            # the one deliberate transition attempt for this pilot.
            state["switched"] = True
        if id(nmpc.nlp[0]) != state["compiled_nlp"]:
            raise RuntimeError("The fatigue-weight transition rebuilt the compiled RHO NLP.")
        log_events.append({
            "event": "state_triggered_regime_transition",
            "completed_cycles": state["completed_cycles"],
            "capacity_ratios": capacity_ratios,
            "minimum_capacity_ratio": minimum_ratio,
            "threshold": switch_min_capacity_ratio,
            "muscle_order": muscle_names,
            "weights_before": before.tolist(),
            "weights_after": binding.weights.tolist(),
            "proposal": _jsonable(proposal),
            "theta_source": theta_source,
            "parameter_update": update,
            "compiled_nlp_reused": True,
            "endurance_improvement_validated": False,
        })
        write_progress()
        print(
            "rho_prep_transition: "
            f"cycle={state['completed_cycles']} min_capacity={minimum_ratio:.4f} "
            f"status={proposal.get('status')} compiled_nlp_reused=True",
            flush=True,
        )
        return result

    nmpc.advance_window = MethodType(advance_and_maybe_switch, nmpc)
    log_events.append({
        "event": "policy_initialized",
        "policy": "rho_prep_two_regime_physio_update_work_v1_experimental",
        "muscle_order": muscle_names,
        "initial_weights": binding.weights.tolist(),
        "switch_trigger": "min_certified_A_over_A_rest_below_threshold",
        "switch_min_capacity_ratio": switch_min_capacity_ratio,
        "max_regime_transitions": 1,
        "objective_graph_rebuild_required": False,
        "uses_fho_data": False,
        "endurance_improvement_validated": False,
    })
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    write_progress()


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def main(argv=None):
    from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as driver

    argv = list(sys.argv[1:] if argv is None else argv)
    if any(token in {"-h", "--help"} for token in argv):
        print(__doc__)
        print("\nThis command accepts the cycling RHO options plus:")
        print("  --rho-prep-switch-min-capacity-ratio FLOAT (default: 0.55)")
        print("  --rho-prep-physio-update-every-cycles INT (default: 20)")
        print("  --rho-prep-smoothing FLOAT (default: 0.2)")
        print("  --rho-prep-max-log-step FLOAT (default: log(1.1))")
        print("  --rho-prep-deadband-log FLOAT (default: log(1.01))")
        print("  --rho-prep-min-relative-weight FLOAT (default: 0.05)")
        print("  --rho-prep-max-relative-weight FLOAT (default: 20)")
        print("  --rho-prep-full-ding (disable the default local Ding reduction)")
        return 0
    parser = driver.build_argument_parser()
    parser.add_argument("--rho-prep-switch-min-capacity-ratio", type=float, default=0.55)
    parser.add_argument("--rho-prep-physio-update-every-cycles", type=int, default=20)
    parser.add_argument("--rho-prep-smoothing", type=float, default=0.2)
    parser.add_argument("--rho-prep-max-log-step", type=float, default=math.log(1.1))
    parser.add_argument("--rho-prep-deadband-log", type=float, default=math.log(1.01))
    parser.add_argument("--rho-prep-min-relative-weight", type=float, default=0.05)
    parser.add_argument("--rho-prep-max-relative-weight", type=float, default=20.0)
    parser.add_argument("--rho-prep-output-json", type=Path, required=True)
    parser.add_argument(
        "--rho-prep-full-ding", action="store_true",
        help="Use all Ding states as NLP variables instead of the default audited local reduction.",
    )
    parser.add_argument("--rho-prep-verbose", action="store_true",
                        help="Print detailed IPOPT diagnostics for every RHO window.")
    args = parser.parse_args(argv)
    if not math.isfinite(args.rho_prep_switch_min_capacity_ratio) or not 0 < args.rho_prep_switch_min_capacity_ratio < 1:
        parser.error("--rho-prep-switch-min-capacity-ratio must be in (0, 1).")
    if args.solver not in ("ipopt", "acados") or args.formulation != "isokinetic" or not args.use_sx:
        parser.error("The RHO-PREP pilot requires IPOPT or ACADOS SX isokinetic RHO.")
    if not args.rho_prep_full_ding:
        reduction_requirements = (
            args.mechanical_formulation == "reduced",
            args.ode_solver == "collocation",
            args.collocation_method == "radau",
            args.collocation_degree == 5,
            args.stimulations_per_cycle == 30,
            args.cycles_per_window == 1,
            args.pulse_width_max_step_us is None,
            args.pulse_width_slew_weight == 0,
        )
        if not all(reduction_requirements):
            parser.error(
                "The default isokinetic Ding reduction requires reduced mechanics, "
                "one-cycle IPOPT SX Radau-5 windows at 30 Hz, and no PW slew constraint. "
                "Use --rho-prep-full-ding only for a full-state comparison."
            )
    args.parametric_fatigue_weights = True
    args.fatigue_weight_values = [DEFAULT_BO_WEIGHTS[name] for name in DEFAULT_BO_WEIGHTS]
    update_config = {
        "update_every_cycles": args.rho_prep_physio_update_every_cycles,
        "smoothing": args.rho_prep_smoothing,
        "max_log_step": args.rho_prep_max_log_step,
        "deadband_log": args.rho_prep_deadband_log,
        "min_relative_weight": args.rho_prep_min_relative_weight,
        "max_relative_weight": args.rho_prep_max_relative_weight,
    }
    events = []
    runtime_state = {}
    receipt_path = Path(args.rho_prep_output_json).resolve()
    progress_path = receipt_path.with_name(receipt_path.stem + ".progress.json")

    def on_runtime_built(runtime):
        runtime_state["compiled_nlp_id"] = id(runtime["nmpc"].nlp[0])
        install_two_regime_policy(
            runtime,
            switch_min_capacity_ratio=args.rho_prep_switch_min_capacity_ratio,
            update_config=update_config,
            log_events=events,
            progress_path=progress_path,
        )

    ding_audit_directory = receipt_path.parent / "ding-local-audit"
    restore_local_reduction = None
    if not args.rho_prep_full_ding:
        restore_local_reduction = install_ipopt_ding_local_reduction(ding_audit_directory)
    try:
        result = driver.solve_case(args, echo=args.rho_prep_verbose, on_runtime_built=on_runtime_built)
    finally:
        if restore_local_reduction is not None:
            restore_local_reduction()
    receipt = {
        "schema_version": 1,
        "kind": "rho_prep_unilateral_pilot",
        "policy": "rho_prep_two_regime_physio_update_work_v1_experimental",
        "resistance_nm": float(args.energy_equivalent_torque),
        "target_cycles_for_20_minutes_at_1_hz": 1200,
        "requested_cycles": int(args.n_windows),
        "switch_min_capacity_ratio": args.rho_prep_switch_min_capacity_ratio,
        "physio_update_config": update_config,
        "ding_local_reduction": {
            # ACADOS runs the full Ding state model.  The process-local
            # Radau-5 adapter is reserved for its IPOPT recovery solves.
            "enabled": not args.rho_prep_full_ding and args.solver == "ipopt",
            "ipopt_recovery_enabled": bool(
                args.solver == "acados" and getattr(args, "acados_ipopt_recovery", False)
                and not args.rho_prep_full_ding
            ),
            "method": "exact_discrete_radau5_reconstruction_of_Cn_Tau1_Km_with_F_A_retained",
            "audit_directory": str(ding_audit_directory)
            if not args.rho_prep_full_ding and args.solver == "ipopt"
            else None,
        },
        "events": events,
        "switch_occurred": any(event.get("event") == "state_triggered_regime_transition" for event in events),
        "compiled_nlp_reused": True,
        "uses_fho_data": False,
        "result_summary": {
            key: result.get(key) for key in (
                "covered_cycles", "validated_cycles", "stop", "success", "end_to_end_wall_time_s",
                "wall_time_s", "solver_time_s", "wall_time_per_cycle_s", "min_A_capacity_ratio",
            )
        },
        "endurance_improvement_validated": False,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    result_path = receipt_path.with_name(receipt_path.stem + ".result.json")
    result_path.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt["result_path"] = str(result_path)
    receipt_path.write_text(json.dumps(_jsonable(receipt), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"rho_prep_receipt: {receipt_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
