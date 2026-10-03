"""Process-isolated, cycle-synchronous bilateral isokinetic RHO-PACE.

Each spawned child owns one complete persistent ``solve_fes_nmpc`` session.
Only compact numerical reports and next-cycle commands cross the pipes; no
CasADi/IPOPT/MA57 object is shared. A two-phase rendezvous makes both local
cost/target updates finish before either child starts the next window.
"""
from __future__ import annotations

from dataclasses import asdict
from copy import copy
import json
import math
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import time
import traceback
from hashlib import sha256
from types import MethodType
from typing import Any, Mapping

from cocofest.optimization.independent_arm_rho_pace import (
    BilateralArmPaceConfig,
    BilateralArmPaceController,
    IndependentArmResistancePace,
    IndependentArmRhoPaceConfig,
)


ARMS = ("right", "left")
PULSE_WIDTH_NUMERICAL_REPLAY_TOLERANCE_FLOOR_S = 5e-12


def _requested_restart_checkpoint_cycles(payload: Mapping[str, Any]) -> frozenset[int]:
    """Read explicit prepared-window exports without silently checkpointing runs."""
    raw = payload.get("prepared_restart_checkpoint_cycles", ())
    if raw is None:
        return frozenset()
    if not isinstance(raw, (list, tuple)):
        raise ValueError("prepared_restart_checkpoint_cycles must be a list of positive cycle indices")
    if any(type(value) is not int or value < 1 for value in raw):
        raise ValueError("prepared_restart_checkpoint_cycles must contain positive integers")
    if len(set(raw)) != len(raw):
        raise ValueError("prepared_restart_checkpoint_cycles must not repeat a cycle")
    return frozenset(raw)


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    """Publish a small receipt only after both worker exports are available."""
    from tempfile import NamedTemporaryFile

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix="." + path.name,
                                suffix=".tmp", delete=False) as stream:
            json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _publish_prepared_pair_export(root: Path, completed_cycles: int,
                                  prepared: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    """Join worker exports, without ever claiming a fresh-worker restart.

    ``prepared-export.json`` is intentionally *not* the final ``receipt.json``
    consumed by the causal continuation validator.  It becomes a receipt only
    after a new solver process restores both archives and reproduces each
    prepared-problem digest.
    """
    reports = {side: prepared[side].get("checkpoint_export") for side in ARMS}
    if not any(reports.values()):
        return None
    if not all(isinstance(report, Mapping) and report.get("status") == "exported"
                   for report in reports.values()):
        raise RuntimeError("A bilateral prepared checkpoint was requested but only one arm exported it")
    directory = root / "checkpoints" / f"cycle-{completed_cycles}"
    configuration = root / "configuration.json"
    arms = {}
    for side, report in reports.items():
        archive = Path(report["primal_path"]).resolve()
        try:
            archive_relative = archive.relative_to(directory)
        except ValueError as error:  # pragma: no cover - worker contract guard
            raise RuntimeError("Worker checkpoint escaped the bilateral checkpoint directory") from error
        arms[side] = {**dict(report), "primal_path": str(archive_relative)}
    document = {
        "schema_version": 1,
        "checkpoint_kind": "prepared_bilateral_shifted_primal_export",
        "completed_cycles": completed_cycles,
        "configuration_path": "../../configuration.json",
        "configuration_sha256": sha256(configuration.read_bytes()).hexdigest(),
        "arms": arms,
        "serialization_roundtrip_exact": True,
        "fresh_worker_replay_verified": False,
        "exact_bilateral_restart": False,
        "next_required_validation": (
            "Rebuild each arm in a fresh worker, restore its primal, active bounds and fixed parameters, "
            "then require identical prepared_problem_sha256 before publishing receipt.json."
        ),
    }
    _atomic_json(directory / "prepared-export.json", document)
    return {"status": "exported_pending_fresh_worker_replay",
            "path": str((directory / "prepared-export.json").resolve())}


def _pace_vr_snapshot_from_solution(*, ocp, solution, states, models, names, args,
                                    side, source_cycle, target_work_j, config, incumbent_weights):
    """Only numerical snapshot extraction runs on the RHO process boundary."""
    import numpy as np
    from bioptim import SolutionMerge
    from cocofest.optimization.adaptive_moment_rollout import DingPulseWidthParameters, MomentTrackingInterval
    from cocofest.optimization.mechanical_reserve_calibration import clip_numerical_pulse_width_bound_violations
    from cocofest.optimization.pace_vr import PaceVrConfig, PaceVrSupervisor, create_pace_vr_snapshot

    phase_count = int(args.stimulations_per_cycle)
    omega = float(args.isokinetic_omega)
    duration = 2 * math.pi / abs(omega)
    theta_end = float(np.asarray(states["theta"]).reshape(-1)[-1])
    terminal = np.asarray([[float(np.asarray(states[f"{key}_{name}"]).reshape(-1)[-1])
                            for key in ("Cn", "F", "A", "Tau1", "Km")] for name in names])
    parameters = tuple(DingPulseWidthParameters.from_model(model, pulse_width_max=.0006) for model in models)
    controls = solution.decision_controls(to_merge=SolutionMerge.NODES)
    widths = np.asarray([np.asarray(controls[f"last_pulse_width_{name}"]).reshape(-1) for name in names])
    tolerance, tolerance_audit = _pulse_width_replay_tolerance_from_nlp(
        ocp=ocp, muscle_names=names, nlp_tolerance=args.nlp_tolerance)
    widths, clip_audit = clip_numerical_pulse_width_bound_violations(
        widths, parameters, tolerance_s=tolerance, certified_source=True)
    model = ocp.nlp[0].model
    dynamics = model.reduced_dynamics
    intervals, nonmuscle, lower_power, upper_power = [], [], [], []
    for phase in range(phase_count):
        theta = theta_end + omega * duration * (phase + .5) / phase_count
        fl, fv, passive = map(lambda x: np.asarray(x, float), dynamics.muscle_relationships(theta, omega))
        gain = ((fl if getattr(model, "activate_force_length_relationship", True) else np.ones_like(fl))
                * (fv if getattr(model, "activate_force_velocity_relationship", True) else np.ones_like(fv))
                + (passive if getattr(model, "activate_passive_force_relationship", True) else np.zeros_like(passive)))
        coefficients = dynamics.coefficient_values(theta)
        intervals.append(MomentTrackingInterval(duration / phase_count,
            tuple(float(m.post_stimulation_amplitude()) for m in models), tuple(gain),
            tuple(coefficients["muscle_effectiveness"]), (0.,) * len(names)))
        nonmuscle.append(-omega * (coefficients["projected_gravity"]
                                  + coefficients["projected_velocity_quadratic"] * omega**2))
        factor = -omega * coefficients["external_torque_effectiveness"]
        bounds = [factor * float(args.load_torque_min), factor * float(args.load_torque_max)]
        lower_power.append(min(bounds))
        upper_power.append(max(bounds))
    supervisor = PaceVrSupervisor(intervals=intervals, pulse_width_parameters=parameters,
        angular_velocity_rad_s=omega, required_work_j=target_work_j, nonmuscle_power_w=nonmuscle,
        power_lower_w=lower_power, power_upper_w=upper_power,
        config=PaceVrConfig(horizon_cycles=config["horizon_cycles"], **config["core"]))
    snapshot = create_pace_vr_snapshot(supervisor, terminal, widths, request_id=f"{side}:{source_cycle}",
        source_cycle=source_cycle, deadline_seconds=config["deadline_seconds"], certified=True,
        weights=incumbent_weights,
        fit_reference=("predicted_next_terminal" if config.get("application_mode")
                       == "terminal_reserve_target_experimental" else "source_terminal"))
    return asdict(snapshot), {"pulse_width_bound_audit": clip_audit,
                              "pulse_width_tolerance_audit": tolerance_audit,
                              "geometry_quadrature": "interval_midpoint",
                              "power_bounds_source": "RHO_load_torque_bounds_times_minus_omega_bext"}


def _apply_pace_vr_weight_decision(ocp, pace, decision, *, certified, current_cycle):
    """Apply a checked numerical parameter update, never reconstruct objectives.

    The local finite-difference trust region belongs to the *slow projection*
    that generated ``decision["weights"]``.  It is not an admissibility
    region for the subsequently observed RHO terminal state: the latter is
    deliberately not inserted into the RHO NLP (``terminal_value_in_nlp`` is
    false).  Rejecting a decision because the state has evolved since its
    snapshot therefore disconnects every asynchronous proposal in practice.
    Provenance, age and deadline are checked by :class:`BilateralPaceVrAsync`;
    here we retain only the boundary and numerical-parameter safety checks.
    """
    import numpy as np
    from cocofest.optimization.pace_vr_async import APPLICATION_MODE

    if certified is not True or decision.get("applied_cycle") != current_cycle:
        raise ValueError("PACE-VR application requires its intended certified boundary.")
    if decision.get("application_mode") != APPLICATION_MODE or decision.get("deadline_met") is not True:
        raise ValueError("PACE-VR decision has no valid explicit weight-adapter contract.")
    binding = getattr(ocp, "fatigue_weight_binding", None)
    if binding is None:
        raise ValueError("PACE-VR cannot update weights without the fixed-parameter binding.")
    weights = np.asarray(decision.get("weights"), dtype=float).reshape(-1)
    if weights.size != len(pace.weights) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("PACE-VR decision must provide one finite, strictly positive weight per muscle.")
    nlp = ocp.nlp[0]
    compiled = getattr(getattr(ocp, "ocp_solver", None), "shaked_ocp_solver", None)
    before = list(pace.weights)
    receipt = binding.update(ocp, weights.tolist())
    if ocp.nlp[0] is not nlp or getattr(getattr(ocp, "ocp_solver", None), "shaked_ocp_solver", None) is not compiled:
        raise RuntimeError("PACE-VR numerical weight update changed the compiled NLP.")
    pace.weights = tuple(weights.tolist())
    return {"status": "applied", "source_cycle": decision["source_cycle"], "applied_cycle": current_cycle,
            "application_lag_cycles": current_cycle - decision["source_cycle"], "weights_before": before,
            "weights_after": list(pace.weights), "parameter_update": receipt,
            "application_mode": APPLICATION_MODE, "terminal_value_in_nlp": False,
            "compiled_nlp_reused": True}


def _apply_pace_rt_decision(ocp, pace, decision, *, certified, current_cycle,
                            reserve_priority=False):
    """Install a PACE reserve-target model as numerical RHO parameters."""
    from cocofest.optimization.pace_rt_ocp import PaceRtObjectiveBinding, PACE_RT_MODE

    if certified is not True or decision.get("applied_cycle") != current_cycle:
        raise ValueError("PACE-RT application requires its intended certified boundary.")
    if decision.get("application_mode") != PACE_RT_MODE or decision.get("deadline_met") is not True:
        raise ValueError("PACE-RT decision has no valid terminal-target contract.")
    binding = getattr(ocp, "task_reserve_binding", None)
    if not isinstance(binding, PaceRtObjectiveBinding):
        raise ValueError("PACE-RT requires its fixed terminal reserve-target binding.")
    nlp = ocp.nlp[0]
    compiled = getattr(getattr(ocp, "ocp_solver", None), "shaked_ocp_solver", None)
    receipt = binding.update_from_rollout(
        ocp, local_fit=decision.get("local_fit"), source_completed_cycles=int(decision["source_cycle"]),
        completed_cycles=int(current_cycle), proximal_weight=float(decision.get("terminal_proximal_weight", .03)),
        shortage_weight=float(decision.get("terminal_shortage_weight", 1.)),
        constraint_activation=0. if reserve_priority else None,
    )
    if ocp.nlp[0] is not nlp or getattr(getattr(ocp, "ocp_solver", None), "shaked_ocp_solver", None) is not compiled:
        raise RuntimeError("PACE-RT numerical terminal update changed the compiled NLP.")
    return {"status": "applied", "source_cycle": decision["source_cycle"], "applied_cycle": current_cycle,
            "application_lag_cycles": current_cycle - decision["source_cycle"],
            "parameter_update": receipt, "application_mode": PACE_RT_MODE,
            "terminal_value_in_nlp": True, "compiled_nlp_reused": True,
            "weights_unchanged": list(pace.weights)}


def _observe_pace_rt_terminal(ocp, states, models, *, completed_cycles):
    """Audit the terminal point against the active PACE-RT local domain.

    The domain is deliberately an audit/hold guard, never a hard RHO
    constraint.  A compact local model may be useful inside its announced
    box, but it must not remain active for another RHO after the solved state
    has left that box.  The caller can retain a freshly arrived replacement
    decision, which is centred on a new predicted terminal state.
    """
    import numpy as np
    from cocofest.optimization.pace_rt_ocp import PaceRtObjectiveBinding

    binding = getattr(ocp, "task_reserve_binding", None)
    if not isinstance(binding, PaceRtObjectiveBinding):
        return None
    coordinates = [
        float(np.asarray(states[f"A_{model.muscle_name}"], dtype=float).reshape(-1)[-1]) / model.a_scale
        for model in models
    ]
    return binding.validate_terminal_point(coordinates, completed_cycles=int(completed_cycles))


def _reserve_update_due(*, certified: bool, has_solution: bool, physical_cycle: int) -> bool:
    return bool(has_solution and certified and
                (physical_cycle == 1 or physical_cycle > 0 and physical_cycle % 20 == 0))


def _pulse_width_replay_tolerance_from_nlp(*, ocp, muscle_names, nlp_tolerance):
    """Return per-muscle physical replay tolerance from NLP scaling.

    The solver tolerance applies to the normalized decision variable, whereas
    the Ding calibration replays pulse widths in seconds. Therefore each
    threshold is ``max(nlp_tolerance * u_scaling, 5 ps)``. This is a numeric
    conversion only, never a relaxation of the physical Ding bounds.
    """
    import numpy as np

    if (isinstance(nlp_tolerance, (bool, np.bool_)) or not np.isfinite(nlp_tolerance)
            or nlp_tolerance < 0.0):
        raise ValueError("NLP tolerance must be a finite nonnegative scalar.")
    names = tuple(str(name) for name in muscle_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("A nonempty unique muscle order is required for PW replay tolerance.")
    try:
        scaling_map = ocp.nlp[0].u_scaling
    except (AttributeError, IndexError, TypeError) as error:
        raise ValueError("PW replay tolerance requires the first NLP control scaling.") from error
    scaling_s = []
    for name in names:
        key = f"last_pulse_width_{name}"
        try:
            scale = np.asarray(scaling_map[key].scaling, dtype=float).reshape(-1)
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"PW replay tolerance is missing a valid scaling for {key}.") from error
        if scale.shape != (1,) or not np.isfinite(scale[0]) or scale[0] <= 0.0:
            raise ValueError(f"PW replay tolerance requires positive scalar scaling for {key}.")
        scaling_s.append(float(scale[0]))
    scaling_s = np.asarray(scaling_s)
    tolerance_s = np.maximum(float(nlp_tolerance) * scaling_s,
                             PULSE_WIDTH_NUMERICAL_REPLAY_TOLERANCE_FLOOR_S)
    audit = {
        "units": "s",
        "derivation": "max(nlp_tolerance * u_scaling_s, floor_s)",
        "nlp_tolerance_normalized": float(nlp_tolerance),
        "u_scaling_s_by_muscle": scaling_s.tolist(),
        "floor_s": PULSE_WIDTH_NUMERICAL_REPLAY_TOLERANCE_FLOOR_S,
        "effective_tolerance_s_by_muscle": tolerance_s.tolist(),
    }
    return tolerance_s, audit


def _certified_mechanical_reserve_profile(*, states, models, reduced_dynamics,
                                          phase_count, omega, target_work_j,
                                          binding, source_cycle, calibration_policy="simultaneous_pwmax_v1",
                                          pulse_widths=None, pulse_width_numerical_tolerance_s=None,
                                          pulse_width_numerical_tolerance_audit=None):
    """Reconstruct an auditable phase-endpoint proxy from one certified RHO."""
    import numpy as np
    from cocofest.optimization.adaptive_moment_rollout import DingPulseWidthParameters, MomentTrackingInterval
    from cocofest.optimization.mechanical_reserve_calibration import (
        calibrate_isokinetic_ding_margin_model, calibrate_isokinetic_ding_pulse_width_force_map,
        clip_numerical_pulse_width_bound_violations,
    )

    names = tuple(str(model.muscle_name) for model in models)
    if len(names) != binding.muscle_count or len(set(names)) != len(names):
        raise ValueError("Certified reserve muscle order does not match the binding.")
    keys = ("theta", *(f"{component}_{name}" for name in names
                       for component in ("Cn", "F", "A", "Tau1", "Km")))
    vectors = {}
    for key in keys:
        if key not in states:
            raise ValueError(f"Certified reserve profile is missing {key}.")
        vector = np.asarray(states[key], dtype=float).reshape(-1)
        if not vector.size or not np.all(np.isfinite(vector)):
            raise ValueError(f"Certified reserve profile contains invalid {key}.")
        vectors[key] = vector
    count = vectors["theta"].size
    if any(vector.size != count for vector in vectors.values()) or (count - 1) % phase_count:
        raise ValueError("Certified reserve collocation nodes do not partition into phases.")
    stride = (count - 1) // phase_count
    if stride < 1 or binding.interval_count != phase_count:
        raise ValueError("Certified reserve profile/binding phase count differs.")
    if not np.isclose(vectors["theta"][-1] - vectors["theta"][0],
                      np.sign(omega) * 2.0 * math.pi, rtol=0, atol=1e-8):
        raise ValueError("Certified reserve profile does not span one signed pedal turn.")
    endpoints = np.arange(1, phase_count + 1) * stride
    forces = np.asarray([vectors[f"F_{name}"][endpoints] for name in names])
    if np.any(forces < 0):
        raise ValueError("Certified reserve forces must be nonnegative.")
    terminal = np.asarray([[vectors[f"{key}_{name}"][-1]
                            for key in ("Cn", "F", "A", "Tau1", "Km")]
                           for name in names])
    duration = 2.0 * math.pi / abs(omega)
    if not np.isclose(sum(binding.durations), duration, rtol=1e-9):
        raise ValueError("Reserve binding durations do not span the certified cycle.")
    intervals, nonmuscle = [], []
    model_parameters = []
    for model in models:
        model_parameters.append(DingPulseWidthParameters.from_model(model, pulse_width_max=.0006))
    for endpoint in endpoints:
        theta = float(vectors["theta"][endpoint])
        fl, fv, passive = reduced_dynamics.muscle_relationships(theta, omega)
        gains = np.asarray(fl) * np.asarray(fv) + np.asarray(passive)
        coefficients = reduced_dynamics.coefficient_values(theta)
        moment = np.asarray(coefficients["muscle_effectiveness"], dtype=float)
        if gains.shape != (len(names),) or moment.shape != (len(names),):
            raise ValueError("Certified reserve geometry/model dimension mismatch.")
        nonmuscle.append(-omega * (coefficients["projected_gravity"]
                                   + coefficients["projected_velocity_quadratic"] * omega**2))
        intervals.append(MomentTrackingInterval(
            duration=duration / phase_count,
            calcium_amplitudes=tuple(float(model.post_stimulation_amplitude()) for model in models),
            mechanical_gains=tuple(float(value) for value in gains),
            moment_coefficients=tuple(float(value) for value in moment),
            target_moments=(0.0,) * len(names),
        ))
    required_power = float(target_work_j) / duration
    calibration = calibrate_isokinetic_ding_margin_model(
        terminal_states=terminal, intervals=intervals,
        pulse_width_parameters=model_parameters, angular_velocity_rad_s=omega,
        required_power_w=np.full(phase_count, required_power),
        required_work_j=target_work_j,
        power_scale_w=max(1.0, abs(required_power)),
        nonmuscle_power_w=nonmuscle,
        calibration_policy=calibration_policy,
    )
    pulse_width_force_model = None
    pulse_width_numerical_bound_audit = None
    if (getattr(binding, "candidate_force_coupling", False)
            or getattr(binding, "local_pulse_width_cost", False)):
        if pulse_widths is None:
            raise ValueError("PW reserve tangent requires certified direct PW controls.")
        if pulse_width_numerical_tolerance_s is None:
            raise ValueError("PW reserve tangent requires an NLP-scaled numerical replay tolerance.")
        pulse_widths, pulse_width_numerical_bound_audit = clip_numerical_pulse_width_bound_violations(
            pulse_widths, model_parameters, tolerance_s=pulse_width_numerical_tolerance_s,
            certified_source=True,
        )
        if pulse_width_numerical_tolerance_audit is not None:
            if not isinstance(pulse_width_numerical_tolerance_audit, Mapping):
                raise ValueError("PW replay tolerance audit must be a mapping.")
            pulse_width_numerical_bound_audit.update(pulse_width_numerical_tolerance_audit)
        pulse_width_force_model = calibrate_isokinetic_ding_pulse_width_force_map(
            terminal_states=terminal, intervals=intervals, pulse_width_parameters=model_parameters,
            pulse_widths=pulse_widths, mechanics_mode="isokinetic",
        )
        forces = pulse_width_force_model.reference_forces
    digest = sha256(np.ascontiguousarray(forces, dtype=np.float64).tobytes()).hexdigest()
    margin_digest = sha256(np.ascontiguousarray(
        np.concatenate((calibration.margin_model.reference_states.ravel(),
                        calibration.margin_model.reference_margins,
                        calibration.margin_model.state_jacobian.ravel())),
        dtype=np.float64).tobytes()).hexdigest()
    receipt = {
        "status": "updated", "source_cycle": int(source_cycle),
        "application_cycle": int(source_cycle) + 1,
        "calibration_kind": calibration.calibration_kind,
        "calibration_policy": getattr(calibration, "calibration_policy", calibration_policy),
        "margin_aggregation": getattr(calibration, "margin_aggregation", None),
        "selected_pulse_widths_s": (
            calibration.selected_pulse_widths.tolist()
            if getattr(calibration, "selected_pulse_widths", None) is not None else None
        ),
        "attainable_work_certified": False,
        "endurance_prediction": False,
        "source_state_sampling": "certified_phase_endpoints_piecewise_constant",
        "source_state_columns": count, "nodes_per_phase": stride,
        "muscle_names": list(names), "force_sha256": digest,
        "margin_sha256": margin_digest,
        "reduced_model_sha256": getattr(reduced_dynamics, "source_model_sha256", None),
        "force_shape": list(forces.shape), "omega_rad_s": float(omega),
        "cycle_duration_s": duration, "target_work_j": float(target_work_j),
        "candidate_force_coupled_to_pw": bool(getattr(binding, "candidate_force_coupling", False)),
        "pw_force_tangent_available": pulse_width_force_model is not None,
        "pulse_width_numerical_bound_audit": pulse_width_numerical_bound_audit,
        "local_pulse_width_cost": bool(getattr(binding, "local_pulse_width_cost", False)),
        "terminal_slow_states": terminal[:, 2:].tolist(),
    }
    return forces, calibration.margin_model, receipt, pulse_width_force_model


class _FrozenRhoRetry:
    """Retry a prepared physical window before Bioptim can advance it.

    The independent-arm callback owns preparation, so it must also own the
    checkpoint. The driver's separate callback and its captured checkpoint do
    not run in this coordinator. ``max_attempts`` includes the nominal solve.
    """

    def __init__(self, program, driver, *, tolerance, max_attempts, recovery_args=None,
                 reserve_priority_epsilon=None, solver=None, secondary_fatigue_weights=None):
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        self.program, self.driver = program, driver
        self.tolerance, self.max_attempts = tolerance, max_attempts
        self.recovery_args = recovery_args
        if reserve_priority_epsilon is not None:
            if (solver is None or not math.isfinite(reserve_priority_epsilon)
                    or reserve_priority_epsilon < 0):
                raise ValueError("Reserve priority requires a solver and finite nonnegative epsilon")
        self.reserve_priority_epsilon = reserve_priority_epsilon
        self.solver = solver
        self.secondary_fatigue_weights = secondary_fatigue_weights
        self.stages = ["frozen_primal_reset_duals"]
        if getattr(recovery_args, "nlp_ipopt_recovery_ma57_tuned", False):
            self.stages.append("ma57_tuned")
        if getattr(recovery_args, "ipopt_failed_rho_pw_micro_retry", False):
            self.stages.append("direct_pw_micro_perturbation")
        if len(self.stages) > 1 and max_attempts < len(self.stages) + 1:
            raise ValueError("The explicit retry ladder requires max_consecutive_failing >= "
                             f"{len(self.stages) + 1} (nominal solve plus enabled stages)")
        self.completed = self.failed_attempts = 0
        self.checkpoint = None
        # Bypass solve_case's instance-level retry closure: its private
        # checkpoint is refreshed by a different callback. The class method
        # retains normal FES/stimulation transfers and before-advance audits.
        self.native_advance = type(program).advance_window.__get__(program, type(program))
        self.native_export = type(program).export_data.__get__(program, type(program)) if reserve_priority_epsilon is not None else None
        program.advance_window = MethodType(self._advance, program)
        if self.native_export is not None:
            program.export_data = MethodType(self._export_data, program)

    def _problem_arrays(self):
        nlp = self.program.nlp[0]
        for prefix, container in (
            ("x_bounds", nlp.x_bounds), ("u_bounds", nlp.u_bounds),
            ("parameter_bounds", getattr(self.program, "parameter_bounds", {})),
        ):
            for key in container.keys():
                for field in ("min", "max"):
                    yield f"{prefix}:{key}:{field}", getattr(container[key], field)

    def capture(self):
        """Freeze after the coordinator installs this cycle's work and weights."""
        import numpy as np
        from cocofest.optimization.receding_horizon_initial_guess import initial_guess_signature

        self.checkpoint = self.driver.snapshot_initial_guess(self.program)
        self.signature = initial_guess_signature(self.checkpoint)
        self.frozen_bounds = {key: np.array(values, copy=True) for key, values in self._problem_arrays()}
        self.frozen_parameters = {
            key: np.array(self.program.parameter_init[key].init, copy=True)
            for key in getattr(self.program, "parameter_init", {}).keys()
        }
        self.frozen_cycle_index = getattr(self.program, "absolute_wheel_q_cycle_index", None)

    def _assert_frozen_problem(self):
        import numpy as np
        current = dict(self._problem_arrays())
        if current.keys() != self.frozen_bounds.keys() or any(
            not np.array_equal(current[key], values) for key, values in self.frozen_bounds.items()
        ):
            raise RuntimeError("The failed RHO changed frozen physical bounds; retry refused")
        if getattr(self.program, "absolute_wheel_q_cycle_index", None) != self.frozen_cycle_index:
            raise RuntimeError("The failed RHO advanced its physical cycle; retry refused")

    def _clear_duals(self):
        reset = self.driver.apply_nlp_dual_warm_start(self.program, None, solver_name="ipopt", mode="off")
        interface = getattr(self.program, "ocp_solver", None)
        reset["multipliers_cleared"] = bool(interface is not None
            and getattr(interface, "lam_x", None) is None and getattr(interface, "lam_g", None) is None)
        if not reset["multipliers_cleared"]:
            raise RuntimeError("Failed to clear IPOPT multipliers before the frozen RHO retry")
        return reset

    def _restore(self):
        from cocofest.optimization.receding_horizon_initial_guess import initial_guess_signature

        self._assert_frozen_problem()
        self.driver._restore_initial_guess_snapshot(self.program, self.checkpoint)
        for key, values in self.frozen_parameters.items():
            self.program.parameter_init[key].init[:, :] = values
        restored = initial_guess_signature(self.driver.snapshot_initial_guess(self.program))
        if restored != self.signature:
            raise RuntimeError("The frozen RHO primal was not restored exactly")
        return {"strategy": "frozen_primal_reset_duals", "prepared_primal_signature": self.signature,
                "restored_primal_signature": restored, "bounds_unchanged": True, "dual_reset": self._clear_duals()}

    def _prepare_retry(self, stage, failed_solution):
        from cocofest.optimization.receding_horizon_initial_guess import initial_guess_signature

        audit = self._restore()
        audit["strategy"] = stage
        if stage == "ma57_tuned":
            args = self.recovery_args
            options = self.driver._ipopt_recovery_advanced_options(args)
            if getattr(args, "ipopt_hsl_library", None):
                options["hsllib"] = args.ipopt_hsl_library
            self.driver.reset_cached_nlp_solver_for_option_change(self.program)
            try:
                _, recovery = self.driver.run_periodic_nlp_recovery(
                    self.program, self.program, recovery_solver="ipopt",
                    max_iterations=args.nlp_ipopt_recovery_max_iterations,
                    tolerance=self.tolerance, linear_solver="ma57", failed_target_solution=failed_solution,
                    target_solver="ipopt", mechanical_formulation="reduced",
                    seed_source="independent_arm_frozen_prepared_primal",
                    ipopt_advanced_options=options, echo=False)
                audit["recovery"] = recovery
                audit["recovery_options"] = options
                if not recovery.get("accepted"):
                    self._restore()
            finally:
                # CasADi caches numerical options in its NLP capsule. Clear
                # the tuned capsule so the following native call rebuilds
                # with its original nominal solver object and tolerances.
                self.driver.reset_cached_nlp_solver_for_option_change(self.program)
                self._clear_duals()
        elif stage == "direct_pw_micro_perturbation":
            try:
                audit["perturbation"] = self.driver.apply_failed_rho_micro_pulse_width_perturbation(
                    self.program, self.checkpoint, retry_index=0)
            except (ValueError, RuntimeError) as error:
                audit["perturbation"] = {"applied": False,
                    "reason": "micro_pw_invariant_refused", "error": f"{type(error).__name__}: {error}"}
            if not audit["perturbation"].get("applied"):
                audit["retry_refused"] = audit["perturbation"].get("reason", "no_safe_pw_change")
                self._restore()
        self._assert_frozen_problem()
        audit["retry_primal_signature"] = initial_guess_signature(self.driver.snapshot_initial_guess(self.program))
        audit["requires_nominal_certification"] = True
        return audit

    def _run_terminal_feasibility_probe(self):
        """Diagnose feasibility without letting a diagnostic advance the RHO."""
        args = self.recovery_args
        if not getattr(args, "ipopt_frozen_rho_feasibility_probe", False):
            return None
        self._restore()
        probe = self.driver.run_frozen_rho_zero_objective_feasibility_probe(
            self.program,
            max_iterations=args.nlp_ipopt_recovery_max_iterations,
            tolerance=self.tolerance,
            linear_solver="ma57",
            echo=False,
        )
        # The probe must leave the exact prepared problem ready for the
        # coordinator's terminal report, irrespective of its solver outcome.
        self._restore()
        return probe

    def _run_terminal_counterfactual_probes(self):
        """Run reversible ordinary-objective tests at a terminal frozen RHO."""
        import numpy as np
        from cocofest.optimization.independent_arm_backends import set_terminal_eprod_target

        args = self.recovery_args
        requested_reliefs = tuple(getattr(args, "ipopt_frozen_rho_work_relief_fraction", ()) or ())
        pw_relief = float(getattr(args, "ipopt_frozen_rho_pw_upper_relief_fraction", 0.0) or 0.0)
        fatigue_rest = bool(getattr(args, "ipopt_frozen_rho_fatigue_rest_probe", False))
        if not (requested_reliefs or pw_relief or fatigue_rest):
            return {}
        if any(not 0.0 < float(value) < 1.0 for value in requested_reliefs):
            raise ValueError("Every frozen-RHO work relief fraction must lie in (0, 1).")
        if not 0.0 <= pw_relief <= 1.0:
            raise ValueError("The frozen-RHO PW upper relief fraction must lie in [0, 1].")

        def solve(kind, metadata):
            self._clear_duals()
            return self.driver.run_frozen_rho_nominal_objective_probe(
                self.program, kind=kind,
                max_iterations=args.nlp_ipopt_recovery_max_iterations,
                tolerance=self.tolerance, linear_solver="ma57", metadata=metadata)

        probes = {}
        nlp = self.program.nlp[0]
        target = float(nlp.x_bounds["E_prod"].min[0, 2])
        for fraction in requested_reliefs:
            fraction = float(fraction)
            relieved = target * (1.0 - fraction)
            self._restore()
            set_terminal_eprod_target(self.program, relieved)
            try:
                probes.setdefault("work_relief", []).append(solve(
                    "nominal_objective_work_relief",
                    {"work_target_original_j": target, "work_target_relaxed_j": relieved,
                     "work_relief_fraction": fraction, "physical_rho_advanced": False}))
            finally:
                set_terminal_eprod_target(self.program, target)
                self._restore()

        if pw_relief:
            self._restore()
            original = {key: (np.array(nlp.u_bounds[key].min, copy=True),
                              np.array(nlp.u_bounds[key].max, copy=True))
                        for key in nlp.u_bounds.keys() if key.startswith("last_pulse_width_")}
            if not original:
                probes["pw_upper_relief"] = {"kind": "nominal_objective_pw_upper_relief",
                    "available": False, "reason": "no_direct_pulse_width_controls"}
            else:
                for key, (lower, upper) in original.items():
                    nlp.u_bounds[key].max[:, :] = upper + pw_relief * (upper - lower)
                try:
                    probes["pw_upper_relief"] = solve("nominal_objective_pw_upper_relief", {
                        "pw_upper_relief_fraction": pw_relief,
                        "physical_rho_advanced": False,
                        "counterfactual": "relaxes_declared_direct_pw_bounds_only",
                    })
                finally:
                    for key, (lower, upper) in original.items():
                        nlp.u_bounds[key].min[:, :] = lower
                        nlp.u_bounds[key].max[:, :] = upper
                    self._restore()

        if fatigue_rest:
            self._restore()
            original = {key: (np.array(nlp.x_bounds[key].min, copy=True),
                              np.array(nlp.x_bounds[key].max, copy=True))
                        for key in nlp.x_bounds.keys() if key.startswith(("A_", "Tau1_", "Km_"))}
            rest = {}
            for model in nlp.model.muscles_dynamics_model:
                rest.update({f"A_{model.muscle_name}": float(model.a_scale),
                             f"Tau1_{model.muscle_name}": float(model.tau1_rest),
                             f"Km_{model.muscle_name}": float(model.km_rest)})
            missing = set(rest) - set(original)
            if missing:
                probes["fatigue_rest"] = {"kind": "nominal_objective_fatigue_rest",
                    "available": False, "reason": f"missing_fatigue_states:{sorted(missing)}"}
            else:
                for key, value in rest.items():
                    nlp.x_bounds[key].min[:, 0] = value
                    nlp.x_bounds[key].max[:, 0] = value
                    nlp.x_init[key].init[:, :] = value
                try:
                    probes["fatigue_rest"] = solve("nominal_objective_fatigue_rest", {
                        "physical_rho_advanced": False,
                        "counterfactual": "A_Tau1_Km_rest__F_Cn_kinematics_work_and_pw_retained",
                        "restored_states": sorted(rest),
                    })
                finally:
                    for key, (lower, upper) in original.items():
                        nlp.x_bounds[key].min[:, :] = lower
                        nlp.x_bounds[key].max[:, :] = upper
                    self._restore()
        return probes

    def _export_data(self, program, solution):
        """Replace a certified primary Solution before Bioptim exports its cycle.

        Bioptim exports before ``advance_window``. Swapping the Solution's
        contents here makes export, physical transfer, and the next callback
        all observe the same secondary trajectory.
        """
        import numpy as np
        from bioptim import SolutionMerge
        from cocofest.optimization.pace_rt_ocp import PaceRtObjectiveBinding

        binding = getattr(program, "task_reserve_binding", None)
        if (self.reserve_priority_epsilon is None or not isinstance(binding, PaceRtObjectiveBinding)
                or binding.last_model is None or binding.values[0] == 0):
            return self.native_export(solution)
        if not binding.enforce_target_constraint or self.checkpoint is None:
            raise RuntimeError("Reserve priority requires a compiled target constraint and frozen RHO")
        primary_feasibility = self.driver._solution_feasibility_summary(solution, self.tolerance)
        if not self.driver._rho_solution_is_certified(solution.status, primary_feasibility):
            return self.native_export(solution)
        primary_stats = self.driver.snapshot_nlp_solver_stats(program)
        original_values = binding.values.copy()
        original_model = dict(binding.last_model)
        original_count = binding.update_count
        fatigue_binding = getattr(program, "fatigue_weight_binding", None)
        if fatigue_binding is None or self.secondary_fatigue_weights is None:
            raise RuntimeError("Reserve priority requires the compiled fatigue-weight binding")
        original_fatigue_weights = fatigue_binding.weights.copy()
        original_fatigue_count = fatigue_binding.update_count
        if np.any(original_fatigue_weights != 0):
            raise RuntimeError("Reserve priority primary solve has nonzero fatigue weights")
        original_nlp = program.nlp[0]
        original_compiled = getattr(getattr(program, "ocp_solver", None), "shaked_ocp_solver", None)
        audit = {
            "mode": "experimental_pace_rt_two_solve_priority",
            "primary_objective": "pace_rt_shortage_plus_terminal_proximity",
            "physical_cycle": self.completed + 1,
            "prepared_primal_signature": self.signature,
            "pace_rt_graph_signature_sha256": binding.graph_signature_sha256,
            "pace_rt_model_sha256": binding.model_sha256,
            "pace_rt_task_context_sha256": binding.task_context_sha256,
            "pace_rt_fit_source_completed_cycles": binding.source_completed_cycles,
            "primary_fatigue_weights": original_fatigue_weights.tolist(),
            "primary": {"status": int(solution.status), "feasibility": primary_feasibility,
                        "solver_stats": primary_stats,
                        "objective_cost": float(solution.cost),
                        "solver_time_s": float(solution.real_time_to_optimize)},
            "epsilon_predicted_value_units": float(self.reserve_priority_epsilon),
            "physical_rho_advanced_between_stages": False,
        }
        secondary = None
        try:
            states = solution.decision_states(to_merge=SolutionMerge.NODES)
            point = [float(np.asarray(states[c.state_key], dtype=float)[c.index, -1] - c.offset) / c.scale
                     for c in binding.coordinates]
            observed = binding.validate_terminal_point(point, completed_cycles=self.completed + 1)
            if observed["terminal_trust_validated"] is not True:
                raise ValueError("Primary terminal point left the PACE-RT fit domain")
            value = float(observed["predicted_terminal_value"])
            target = value + self.reserve_priority_epsilon
            if not math.isfinite(target):
                raise ValueError("Secondary reserve target is nonfinite")
            audit.update({"primary_predicted_terminal_value": value,
                          "secondary_target_predicted_value": target,
                          "primary_terminal_audit": observed,
                          "target_rule": "primary_predicted_terminal_value_plus_epsilon"})
            audit["stage_transition"] = binding.activate_fatigue_secondary_stage(
                program, reserve_target=target)
            audit["secondary_fatigue_weights"] = list(self.secondary_fatigue_weights)
            audit["fatigue_transition"] = fatigue_binding.update(program, self.secondary_fatigue_weights)
            self._assert_frozen_physical_problem()
            if program.nlp[0] is not original_nlp or getattr(
                    getattr(program, "ocp_solver", None), "shaked_ocp_solver", None) is not original_compiled:
                raise RuntimeError("Reserve priority changed the compiled NLP")
            secondary, second_audit = self.driver.solve_frozen_rho_secondary_stage(
                program, solver=self.solver, tolerance=self.tolerance)
            second_audit["objective_cost"] = float(secondary.cost)
            audit["secondary"] = second_audit
            if not second_audit["certified"]:
                raise RuntimeError("Secondary RHO did not pass nominal certification")
            second_states = secondary.decision_states(to_merge=SolutionMerge.NODES)
            second_point = [float(np.asarray(second_states[c.state_key], dtype=float)[c.index, -1] - c.offset) / c.scale
                            for c in binding.coordinates]
            second_value = binding.validate_terminal_point(
                second_point, completed_cycles=self.completed + 1)["predicted_terminal_value"]
            audit["secondary_predicted_terminal_value"] = float(second_value)
            if not math.isfinite(second_value) or second_value > target + self.tolerance:
                raise RuntimeError("Secondary RHO violated its predicted-value target")
            self._assert_frozen_physical_problem()
            audit["compiled_nlp_reused"] = True
            audit["status"] = "committed_secondary"
        except Exception as error:
            audit.update({"status": "secondary_failed_rolled_back",
                          "error": f"{type(error).__name__}: {error}"})
            solution._cocofest_priority_failed = True
        finally:
            # Restore the primary numerical objective for the next physical
            # RHO. On failure, also restore its exact prepared primal/duals.
            binding._write(program, original_values)
            binding.last_model = original_model
            fatigue_binding.update(program, original_fatigue_weights)
            if secondary is None or audit["status"] != "committed_secondary":
                binding.update_count = original_count
                fatigue_binding.update_count = original_fatigue_count
                self._restore()
            else:
                self._assert_frozen_problem()
        if audit["status"] == "committed_secondary":
            # RecedingHorizonOptimization retains this object as `sol` after
            # export; replacing its contents is needed for the next callback
            # and final aggregate Solution to use the committed trajectory.
            solution.__dict__.clear()
            solution.__dict__.update(secondary.__dict__)
        solution._cocofest_priority_protocol = audit
        return self.native_export(solution)

    def _assert_frozen_physical_problem(self):
        import numpy as np
        current = dict(self._problem_arrays())
        physical = (key for key in self.frozen_bounds if not key.startswith("parameter_bounds:"))
        if any(not np.array_equal(current[key], self.frozen_bounds[key]) for key in physical):
            raise RuntimeError("Reserve priority changed frozen physical bounds")
        if getattr(self.program, "absolute_wheel_q_cycle_index", None) != self.frozen_cycle_index:
            raise RuntimeError("Reserve priority advanced the physical cycle between stages")

    def _advance(self, program, solution, *args, **kwargs):
        if self.checkpoint is None:
            raise RuntimeError("No prepared checkpoint exists for the physical RHO")
        feasibility = self.driver._solution_feasibility_summary(solution, self.tolerance)
        certified = (not getattr(solution, "_cocofest_priority_failed", False)
                     and self.driver._rho_solution_is_certified(solution.status, feasibility))
        # An auxiliary tuned solve replaces live solver stats. Keep the
        # nominal attempt's evidence before entering any recovery helper.
        solution._cocofest_nominal_solver_stats = self.driver.snapshot_nlp_solver_stats(program)
        solution._cocofest_target_rho = self.completed + 1
        solution._cocofest_advanced_physical_rho = False
        solution._cocofest_feasibility_summary = feasibility
        solution._cocofest_attempt_in_physical_rho = self.failed_attempts + 1
        program._cocofest_retry_same_rho_pending = False
        if not certified:
            if getattr(solution, "_cocofest_priority_failed", False):
                # A second-stage failure is terminal for this frozen window:
                # another nominal solve would use a different first-stage
                # witness and obscure the two-stage audit.
                self.failed_attempts += 1
                return None
            self.failed_attempts += 1
            stage_index = self.failed_attempts - 1
            if self.failed_attempts < self.max_attempts and stage_index < len(self.stages):
                audit = self._prepare_retry(self.stages[stage_index], solution)
                solution._cocofest_frozen_retry = audit
                program._cocofest_retry_same_rho_pending = "retry_refused" not in audit
            if not program._cocofest_retry_same_rho_pending:
                solution._cocofest_zero_objective_feasibility_probe = self._run_terminal_feasibility_probe()
                solution._cocofest_counterfactual_probes = self._run_terminal_counterfactual_probes()
            # Even the terminal failed attempt must not modify the FES state.
            return None
        result = self.native_advance(solution, *args, **kwargs)
        solution._cocofest_advanced_physical_rho = True
        self.completed += 1
        self.failed_attempts = 0
        return result


def _worker_solver_affinity(payload: Mapping[str, Any], side: str) -> tuple[int, ...] | None:
    """Return the post-initialization CPU set requested for one solver worker.

    The process runner deliberately delays this affinity until its OCP and
    optional native callback cache have been constructed.  Pinning the parent
    with ``taskset`` instead also pins the two simultaneous cold starts (and
    GCC on a cold cache), which can make a warm benchmark appear to hang.
    """
    requested = payload.get("solver_cpu_affinity")
    if requested is None:
        return None
    if isinstance(requested, Mapping):
        requested = requested.get(side)
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 0:
        raise ValueError("solver_cpu_affinity must be a non-negative CPU integer or an arm-to-CPU mapping.")
    return (requested,)


def _apply_worker_solver_affinity(payload: Mapping[str, Any], side: str) -> tuple[int, ...] | None:
    """Pin only the ready-to-solve child and return its effective CPU set."""
    requested = _worker_solver_affinity(payload, side)
    if requested is None:
        return None
    if not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("solver_cpu_affinity requires os.sched_setaffinity on this platform.")
    available = os.sched_getaffinity(0)
    missing = set(requested) - available
    if missing:
        raise ValueError(f"Requested solver CPU(s) are unavailable to the worker: {sorted(missing)}")
    os.sched_setaffinity(0, requested)
    return tuple(sorted(os.sched_getaffinity(0)))


def _resolve_hsl_library(payload: Mapping[str, Any]) -> str | None:
    """Locate the optional MA57 library owned by the selected runtime.

    The GUI launches the coordinator in a clean process.  Merely selecting
    ``ma57`` is insufficient in such a process: IPOPT also needs the HSL
    shared object.  Prefer an explicit request/environment override, then the
    runtime prefix supplied by the public CLI.
    """
    explicit = payload.get("ipopt_hsl_library") or os.environ.get("IPOPT_HSL_LIBRARY")
    if explicit:
        candidate = Path(str(explicit)).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())
    prefix = payload.get("runtime_prefix") or os.environ.get("CONDA_PREFIX")
    if not prefix:
        return None
    candidates = sorted(Path(str(prefix)).expanduser().glob("opt/libhsl/*/lib/libhsl.so"))
    return str(candidates[-1].resolve()) if candidates else None


def _can_preserve_static_unit_objective(pace: BilateralArmPaceController) -> bool:
    """Whether the built-in fatigue objective needs no symbolic replacement.

    The unilateral OCP is initially built with the unit-weight fatigue cost.
    C callback compilation is safe only when this exact objective stays in
    place.  A feedback controller, or non-unit relative weights, must retain
    the existing symbolic-update path instead.
    """
    return (
        not pace.config.adaptation_enabled
        and all(math.isclose(weight, 1.0, rel_tol=0.0, abs_tol=1e-12) for weight in pace.weights)
    )


def _can_preserve_compiled_objective(pace: BilateralArmPaceController, runtime: Mapping[str, Any]) -> bool:
    """A fixed parameter binding also keeps the compiled graph intact."""
    return _can_preserve_static_unit_objective(pace) or getattr(
        runtime.get("nmpc"), "fatigue_weight_binding", None
    ) is not None


def _timing_summary(values):
    """Linear-interpolated quantiles, matching numpy's default percentile."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None}
    def quantile(fraction):
        index = (len(ordered) - 1) * fraction
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (index - lower) * (ordered[upper] - ordered[lower])
    return {"count": len(ordered), "mean": sum(ordered) / len(ordered),
            "median": quantile(.5), "p90": quantile(.9), "p95": quantile(.95),
            "min": ordered[0], "max": ordered[-1], "sum": sum(ordered)}


def _receive_pair(connections, *, kind, completed_cycles, timeout_seconds, processes=None):
    pending = dict(connections)
    messages = {}
    deadline = time.monotonic() + timeout_seconds
    while pending:
        ready = wait(list(pending.values()), timeout=min(1.0, max(0., deadline - time.monotonic())))
        if not ready:
            if processes is not None:
                exited = [side for side in pending if not processes[side].is_alive()]
                if exited:
                    codes = {side: processes[side].exitcode for side in exited}
                    raise RuntimeError(
                        f"Worker process exited before {kind} at cycle {completed_cycles}: {codes}"
                    )
            if time.monotonic() < deadline:
                continue
            raise TimeoutError(f"Timed out awaiting {kind} at cycle {completed_cycles}: {sorted(pending)}")
        for connection in ready:
            side = next(side for side, candidate in pending.items() if candidate is connection)
            try:
                message = connection.recv()
            except EOFError as error:
                raise RuntimeError(f"{side} worker exited before {kind}") from error
            if message.get("kind") == "error":
                raise RuntimeError(f"{side} worker: {message.get('error')}\n{message.get('traceback', '')}")
            if message.get("kind") != kind or message.get("completed_cycles") != completed_cycles:
                raise RuntimeError(f"Unexpected {side} worker message: {message}")
            messages[side] = message
            del pending[side]
    return messages


def _driver_arguments(payload, side):
    """Prepare the same reduced Radau-5/MA57 conditions as the first benchmark."""
    from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as driver
    extra = payload.get(f"{side}_driver_arguments", [])
    if not isinstance(extra, list) or any(not isinstance(value, str) for value in extra):
        raise ValueError(f"{side}_driver_arguments must be a list of CLI strings")
    argv = [
        "--solver", "ipopt", "--mechanical-formulation", "reduced",
        "--formulation", "isokinetic", "--ode-solver", "collocation",
        "--collocation-degree", "5", "--collocation-method", "radau",
        "--ipopt-linear-solver", "ma57", "--use-sx", "--cycles-per-window", "1",
        "--n-windows", str(payload["cycles"]), "--stimulations-per-cycle",
        str(payload.get("stimulations_per_cycle", 30)), "--n-threads", "1",
        "--compact-rho-output", "--energy-equivalent-torque",
        str(payload[f"{side}_equivalent_mean_torque_nm"]),
        "--disable-historical-ipopt-initial-guess",
    ]
    hsl_library = _resolve_hsl_library(payload)
    if hsl_library:
        argv.extend(("--ipopt-hsl-library", hsl_library))
    if payload.get("parametric_fatigue_weights", False):
        argv.append("--parametric-fatigue-weights")
        fatigue_values = payload.get("fatigue_weight_values")
        if fatigue_values is not None:
            if (not isinstance(fatigue_values, (list, tuple)) or not fatigue_values
                    or any(isinstance(value, bool) or not isinstance(value, (int, float))
                           or not math.isfinite(float(value)) or not 0 <= float(value) <= 1
                           for value in fatigue_values)):
                raise ValueError("fatigue_weight_values must be a nonempty finite [0, 1] list.")
            argv.extend(("--fatigue-weight-values", *[str(float(value)) for value in fatigue_values]))
    elif payload.get("fatigue_weight_values") is not None:
        raise ValueError("fatigue_weight_values requires parametric_fatigue_weights=true.")
    if float(payload.get("experimental_max_pw_work_weight", 0.)) > 0:
        argv.extend(("--experimental-max-pw-work-weight", str(payload["experimental_max_pw_work_weight"]),
                     "--experimental-max-pw-work-substeps", str(payload.get("experimental_max_pw_work_substeps", 16)),
                     "--experimental-max-pw-work-policy", payload.get("experimental_max_pw_work_policy", "all_intervals_pw_max"),
                     "--experimental-max-pw-work-gradient-filter", payload.get("experimental_max_pw_work_gradient_filter", "full")))
    if float(payload.get("experimental_mechanical_reserve_weight", 0.0)) > 0.0:
        argv.extend(("--experimental-mechanical-reserve-weight",
                     str(payload["experimental_mechanical_reserve_weight"])))
        horizons = payload.get("experimental_mechanical_reserve_horizons")
        if horizons is not None:
            if not isinstance(horizons, (list, tuple)):
                raise ValueError("experimental_mechanical_reserve_horizons must be a list of positive integers.")
            argv.extend(("--experimental-mechanical-reserve-horizons", *map(str, horizons)))
        policy = payload.get("experimental_mechanical_reserve_calibration_policy")
        if policy is not None:
            if not isinstance(policy, str):
                raise ValueError("experimental_mechanical_reserve_calibration_policy must be a string.")
            argv.extend(("--experimental-mechanical-reserve-calibration-policy", policy))
        if payload.get("experimental_mechanical_reserve_pw_force_coupling", False):
            argv.append("--experimental-mechanical-reserve-pw-force-coupling")
        if payload.get("experimental_mechanical_reserve_local_pw_cost", False):
            argv.append("--experimental-mechanical-reserve-local-pw-cost")
            argv.extend(("--experimental-mechanical-reserve-local-pw-trust-us", str(
                payload.get("experimental_mechanical_reserve_local_pw_trust_us", 25.0)
            )))
    args = driver.build_argument_parser().parse_args([*argv, *extra])
    # This internal configuration is consumed during unilateral RHO graph
    # construction, after the exact model and task context are available.  It
    # is deliberately not serialized through the public CLI.
    args.experimental_pace_rt_config = (
        dict(payload["experimental_pace_vr"])
        if payload.get("experimental_pace_vr", {}).get("application_mode")
        == "terminal_reserve_target_experimental" else None
    )
    if args.solver != "ipopt" or args.formulation != "isokinetic" or args.cycles_per_window != 1:
        raise ValueError("The bilateral process runner requires IPOPT, isokinetic, one-cycle RHO windows")
    if (args.nlp_ipopt_recovery_ma57_tuned or args.ipopt_failed_rho_pw_micro_retry) and not args.retry_failed_rho_without_advance:
        raise ValueError("Independent-arm recovery stages require --retry-failed-rho-without-advance")
    counterfactual_requested = (
        args.ipopt_frozen_rho_feasibility_probe
        or bool(args.ipopt_frozen_rho_work_relief_fraction)
        or bool(args.ipopt_frozen_rho_pw_upper_relief_fraction)
        or args.ipopt_frozen_rho_fatigue_rest_probe
    )
    if counterfactual_requested and not args.retry_failed_rho_without_advance:
        raise ValueError("Independent-arm feasibility probes require --retry-failed-rho-without-advance")
    if args.retry_failed_rho_without_advance:
        secondary = (
            "ipopt_madnlp_recovery", "nlp_failed_rho_phase_one_recovery", "nlp_ipopt_fallback_advance",
        )
        unsupported_ipopt = args.nlp_ipopt_recovery and not args.nlp_ipopt_recovery_ma57_tuned
        if any(getattr(args, option, False) for option in secondary) or unsupported_ipopt:
            raise ValueError("Independent-arm frozen retries currently support the same IPOPT backend only; "
                             "secondary-solver and Phase-I recovery need a separately validated adapter")
        driver.validate_tuned_ma57_recovery_options(args)
        if args.nlp_ipopt_recovery_ma57_tuned and args.nlp_ipopt_recovery_collocation_degree != args.collocation_degree:
            raise ValueError("Independent-arm tuned recovery must keep the exact collocation degree")
    # The GUI/parent owns the common speed; arm-specific overrides cannot make
    # the two physical cycle durations disagree.
    args.isokinetic_omega = float(payload.get("omega_rad_s", -2 * math.pi))
    return args


def _configured_payload_model(payload: Mapping[str, Any], side: str | None = None):
    """Resolve an optional common or side-specific Ding configuration.

    The process runner historically always built the repository-default
    muscles. A bilateral endurance result with a declared parameter variant
    must instead patch both spawned arm factories before their NLP graphs are
    constructed. Returning the resolved document also gives the worker a
    compact, JSON-safe receipt to retain beside its numerical result.
    """
    if side is not None and side not in ARMS:
        raise ValueError(f"Unknown independent-arm side {side!r}")
    declared = payload.get(f"{side}_model_config") if side is not None else None
    if declared is None:
        declared = payload.get("model_config")
    if declared is None:
        return None
    if not isinstance(declared, str) or not declared.strip():
        raise ValueError("model_config must be a nonempty path when supplied")
    path = Path(declared).expanduser().resolve(strict=True)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"model_config is not valid JSON: {path}") from error
    from cocofest.optimization.configured_cycling_model import resolve_model_config
    return path, resolve_model_config(document)


def _arm_worker(connection, side, payload, output_root):
    """One child; every callback is part of the same native RHO solve loop."""
    root = Path(output_root) / side
    root.mkdir(parents=True, exist_ok=True)
    # Capture native IPOPT's file-descriptor output as well as Python prints.
    with (root / "solver.log").open("w", encoding="utf-8") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            import numpy as np
            from bioptim import MultiCyclicCycleSolutions, SolutionMerge
            from cocofest.optimization.independent_arm_backends import set_terminal_eprod_target
            from cocofest.optimization.rho_pace import update_bioptim_fatigue_cost
            from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as driver

            args = _driver_arguments(payload, side)
            build_args = copy(args)
            # The local controller runs restoration against the exact live
            # NLP. Avoid building the driver's unused second recovery OCP or
            # arming its separately captured checkpoint closure.
            build_args.nlp_ipopt_recovery = False
            build_args.nlp_ipopt_recovery_ma57_tuned = False
            configured_model = _configured_payload_model(payload, side)
            model_builds = []
            if configured_model is None:
                runtime = driver.build_unilateral_runtime(build_args, echo=False)
                configured_receipt = None
            else:
                model_path, model_config = configured_model
                from cocofest.optimization.configured_cycling_model import configured_model_factories
                with configured_model_factories(model_config, model_builds):
                    runtime = driver.build_unilateral_runtime(build_args, echo=False)
                configured_receipt = {
                    "model_config_path": str(model_path),
                    "case_id": model_config["case_id"],
                    "muscle_parameter_fingerprint": model_config["muscle_parameter_fingerprint"],
                    "configured_muscle_parameters": model_config["muscles"],
                    "model_builds": model_builds,
                }
            nmpc, solver = runtime["nmpc"], runtime["solver"]
            effective_affinity = _apply_worker_solver_affinity(payload, side)
            if effective_affinity is not None:
                print(f"Solver worker affinity: {effective_affinity}", flush=True)
            models = nmpc.nlp[0].model.muscles_dynamics_model
            names = tuple(model.muscle_name for model in models)
            declared = dict(payload.get("muscle_pace", {}))
            basis = declared.pop("initial_weight_basis", "uniform independent-arm baseline; no FHO data")
            initial = declared.pop(f"{side}_initial_weights", dict.fromkeys(names, 1.))
            declared.pop(f"{'left' if side == 'right' else 'right'}_initial_weights", None)
            if not isinstance(initial, Mapping) or set(initial) != set(names):
                raise ValueError(f"{side}_initial_weights must identify the four model muscles exactly")
            pace = BilateralArmPaceController(
                side, names, [initial[name] for name in names], config=BilateralArmPaceConfig(**declared),
                equivalent_mean_torque_nm=payload[f"{side}_equivalent_mean_torque_nm"],
                initial_weight_basis=basis, journal_path=root / "weights.jsonl")
            from cocofest.optimization.pace_vr_async import validate_pace_vr_configuration
            vr_config = validate_pace_vr_configuration(payload)
            reserve_priority = bool(vr_config and vr_config.get("lexicographic_reserve_priority", False))
            if reserve_priority and (args.solver != "ipopt" or not args.use_sx
                                     or args.formulation != "isokinetic"):
                raise ValueError("PACE-RT reserve priority requires IPOPT SX isokinetic RHO")
            if reserve_priority and getattr(nmpc, "fatigue_weight_binding", None) is None:
                raise ValueError("PACE-RT reserve priority requires compiled fatigue weights")
            vr_events = []
            reserve_binding = getattr(nmpc, "mechanical_reserve_binding", None)
            reserve_events = []
            if reserve_binding is not None:
                if getattr(nmpc, "fatigue_weight_binding", None) is not None:
                    raise ValueError("Reserve and fatigue ParameterList bindings cannot coexist.")
                if not _can_preserve_static_unit_objective(pace):
                    raise ValueError("Experimental reserve requires fixed unit fatigue weights and disabled adaptation.")
                if args.solver != "ipopt" or not args.use_sx or args.formulation != "isokinetic":
                    raise ValueError("Experimental reserve worker requires IPOPT SX isokinetic RHO.")
                if args.nlp_ipopt_recovery_ma57_tuned:
                    raise ValueError("Experimental reserve excludes tuned recovery that rebuilds the compiled solver.")
                pace.connected = True
            if driver.nlp_c_compile_enabled(args):
                if not _can_preserve_compiled_objective(pace, runtime):
                    raise ValueError(
                        "IPOPT C compilation requires fixed unit fatigue weights or "
                        "parametric_fatigue_weights=true for the compiled PACE path."
                    )
                # The factory installs either the unit-weight objective or its
                # fixed-parameter equivalent. By marking it connected, the
                # initial RHO boundary audits it but does not reconstruct the
                # objective. Terminal work and parameter equality bounds remain
                # numerical updates.
                pace.connected = True
            # ``records`` deliberately contains physical (committed) RHO
            # windows only.  A native retry can produce several solver
            # attempts for one physical cycle; putting those attempts in this
            # list would make the endpoint look longer than it is.
            records = []
            recovery_attempts = []
            requested_checkpoint_cycles = _requested_restart_checkpoint_cycles(payload)
            frozen_retry = (_FrozenRhoRetry(nmpc, driver,
                tolerance=driver._window_feasibility_tolerance(args),
                max_attempts=(args.max_consecutive_failing
                              if args.retry_failed_rho_without_advance else 1), recovery_args=args,
                reserve_priority_epsilon=(vr_config["lexicographic_reserve_epsilon"]
                                          if reserve_priority else None), solver=solver,
                secondary_fatigue_weights=(tuple(pace.weights) if reserve_priority else None))
                if args.retry_failed_rho_without_advance or reserve_priority else None)

            def callback(ocp, cycle_index, solution):
                completed = int(cycle_index)
                if solution is None:
                    ratios = [float(ocp.nlp[0].x_bounds[f"A_{model.muscle_name}"].min[0, 0]) / model.a_scale
                              for model in models]
                    metrics = {"certified": True, "initial_seed": True}
                else:
                    states = solution.decision_states(to_merge=SolutionMerge.NODES)
                    ratios = [float(states[f"A_{model.muscle_name}"][0, -1]) / model.a_scale for model in models]
                    feasibility = driver._solution_feasibility_summary(solution, driver._window_feasibility_tolerance(args))
                    certified = driver._rho_solution_is_certified(solution.status, feasibility)
                    eprod = np.asarray(states["E_prod"], dtype=float).reshape(-1)
                    raw_cost = float(solution.cost)
                    solver_stats = getattr(solution, "_cocofest_nominal_solver_stats", None)
                    if solver_stats is None:
                        solver_stats = driver.snapshot_nlp_solver_stats(ocp)
                    native_status = solver_stats.get("return_status") or driver._native_solver_status(ocp)
                    if getattr(solution, "_cocofest_priority_failed", False):
                        certified = False
                    metrics = {"certified": certified, "success": certified, "status": int(solution.status),
                               "native_solver_status": native_status,
                               "solver_stats": solver_stats,
                               "feasibility": feasibility,
                               # A compact native RHO result may not expose a
                               # finite aggregate cost even though the window
                               # status and feasibility certification are
                               # valid. JSON output is deliberately strict:
                               # preserve that distinction as ``null`` rather
                               # than making the worker fail after completion.
                               "cost": raw_cost if math.isfinite(raw_cost) else None,
                               "solver_time_s": float(solution.real_time_to_optimize),
                               "achieved_work_j_per_cycle": float(eprod[-1] - eprod[0]),
                               "weights_used": list(pace.weights),
                               "equivalent_mean_torque_nm": pace.equivalent_mean_torque_nm,
                               "target_work_j_per_cycle": 2 * math.pi * pace.equivalent_mean_torque_nm}
                    active_fatigue_binding = getattr(ocp, "fatigue_weight_binding", None)
                    if active_fatigue_binding is not None:
                        # The controller's policy weights and the numerical
                        # fatigue-objective parameters are normally identical.
                        # Keep both in the audit because a zero-fatigue PACE-RT
                        # ablation deliberately freezes the latter at zero.
                        metrics["fatigue_objective_weights"] = list(active_fatigue_binding.weights)
                    if reserve_binding is not None:
                        metrics["mechanical_reserve_solver"] = reserve_binding.observe_solver(ocp)
                        from cocofest.optimization.mechanical_reserve_projection_ocp import (
                            ACTIVATION_PARAMETER_KEY, FORCE_PARAMETER_KEY, MARGIN_PARAMETER_KEY,
                        )
                        slow = np.asarray([
                            [float(states[f"{key}_{name}"][0, -1])
                             for key in ("A", "Tau1", "Km")]
                            for name in names
                        ])
                        vectors = reserve_binding._vectors()
                        reserve_outputs = reserve_binding.function(
                            slow, vectors[FORCE_PARAMETER_KEY], vectors[MARGIN_PARAMETER_KEY],
                            vectors[ACTIVATION_PARAMETER_KEY])
                        metrics["mechanical_reserve_cost"] = {
                            "weighted_cost": float(reserve_outputs[0]),
                            "unweighted_proxy_penalty": float(reserve_outputs[1]),
                            "hard_minimum_margin": float(reserve_outputs[2]),
                            "activation": reserve_binding.activation,
                            "attainable_work_certified": False,
                        }
                        metrics["mechanical_reserve_gradient_audit"] = reserve_binding.sensitivity_audit(slow)
                    if getattr(ocp, "max_pw_work_binding", None) is not None:
                        from cocofest.optimization.max_pw_work_capacity_ocp import max_pw_work_boundary
                        capacity_audit = max_pw_work_boundary(ocp, solution, certified=metrics["certified"])
                        metrics["max_pw_work_capacity"] = capacity_audit
                        if not capacity_audit["accepted"]:
                            metrics["certified"] = metrics["success"] = False
                            metrics["hold_reason"] = capacity_audit.get("reason", "capacity_domain_invalid")
                    metrics["attempt_in_physical_rho"] = int(getattr(solution, "_cocofest_attempt_in_physical_rho", 1))
                    if hasattr(solution, "_cocofest_frozen_retry"):
                        metrics["retry_preparation"] = solution._cocofest_frozen_retry
                    if hasattr(solution, "_cocofest_zero_objective_feasibility_probe"):
                        metrics["zero_objective_feasibility_probe"] = solution._cocofest_zero_objective_feasibility_probe
                    if hasattr(solution, "_cocofest_counterfactual_probes"):
                        metrics["counterfactual_probes"] = solution._cocofest_counterfactual_probes
                    if hasattr(solution, "_cocofest_priority_protocol"):
                        metrics["pace_rt_priority_protocol"] = solution._cocofest_priority_protocol
                metrics.update({"capacity_ratios": dict(zip(names, ratios)),
                                "minimum_capacity_ratio": min(ratios)})
                # The driver tags the returned solution with the *physical*
                # RHO it represents.  ``cycle_index`` is an attempt counter
                # when retry-failed-rho-without-advance is enabled.
                physical_completed = (
                    int(getattr(solution, "_cocofest_target_rho", completed))
                    if solution is not None else completed
                )
                if solution is not None:
                    # This is intentionally post-solve.  The local PACE-RT
                    # model shapes the RHO objective but is not a physical
                    # feasibility constraint; only the solved endpoint tells
                    # us whether its announced trust region was respected.
                    terminal_audit = _observe_pace_rt_terminal(
                        ocp, states, models, completed_cycles=physical_completed)
                    if terminal_audit is not None:
                        metrics["pace_rt_terminal"] = terminal_audit
                if (solution is not None and physical_completed == 1
                        and payload.get("resistance_pace", {}).get("initial_split_policy")
                        == "capacity_fatigability_after_first_cycle"):
                    # This first certified RHO is a reference measurement, not
                    # a future-horizon solve.  Its full Ding state is retained
                    # only long enough to construct an auditable, one-cycle
                    # PW-max *opportunity* and force-induced damage measure.
                    try:
                        from cocofest.optimization.bilateral_initial_split import capacity_fatigability_measurement
                        from cocofest.optimization.physio_update import build_isokinetic_max_pw_envelope_from_discrete_cycle
                        envelope = build_isokinetic_max_pw_envelope_from_discrete_cycle(
                            states=states, muscle_models=models,
                            reduced_dynamics=ocp.nlp[0].model.reduced_dynamics,
                            stimulations_per_cycle=int(args.stimulations_per_cycle),
                            angular_velocity_rad_s=float(args.isokinetic_omega),
                            activate_force_length=bool(getattr(
                                ocp.nlp[0].model, "activate_force_length_relationship", True)),
                            activate_force_velocity=bool(getattr(
                                ocp.nlp[0].model, "activate_force_velocity_relationship", True)),
                            activate_passive_force=bool(getattr(
                                ocp.nlp[0].model, "activate_passive_force_relationship", True)),
                        )
                        metrics["capacity_fatigability_initial_split"] = _jsonable(
                            capacity_fatigability_measurement(
                                certified=metrics["certified"],
                                reference_work_j=metrics["achieved_work_j_per_cycle"],
                                **{key: envelope[key] for key in (
                                    "available_positive_power", "phase_durations", "current_capacity",
                                    "rest_capacity", "alpha_a", "tau_fat", "reference_force",
                                )},
                            )
                        )
                    except (TypeError, ValueError, KeyError) as error:
                        # The reference RHO remains valid; missing diagnostic
                        # data must hold the manual allocation, never abort it.
                        metrics["capacity_fatigability_initial_split"] = {
                            "policy": "capacity_fatigability_reference_v1", "status": "held",
                            "reason": "measurement_build_failed", "detail": str(error),
                        }
                retry_same_rho = bool(getattr(ocp, "_cocofest_retry_same_rho_pending", False))
                if solution is not None:
                    with (root / "attempts.jsonl").open("a", encoding="utf-8") as journal:
                        journal.write(json.dumps({"attempt_index": completed,
                            "physical_cycle": physical_completed, **metrics}, allow_nan=False) + "\n")
                if solution is not None and not metrics["certified"] and retry_same_rho:
                    # The native driver has kept the prepared primal, bounds
                    # and physical state frozen.  Do not expose a boundary to
                    # the coordinator: the other arm remains blocked at its
                    # current boundary while this arm retries the same RHO.
                    recovery_attempts.append({
                        "attempt_index": completed,
                        "physical_cycle": physical_completed,
                        **metrics,
                    })
                    return True
                if vr_config is not None and solution is not None and metrics["certified"]:
                    metrics["pace_vr_terminal_slow"] = [
                        [float(np.asarray(states[f"{key}_{name}"]).reshape(-1)[-1]) for key in ("A", "Tau1", "Km")]
                        for name in names]
                    if physical_completed == 1 or physical_completed % vr_config["update_every_cycles"] == 0:
                        snapshot_started = time.perf_counter()
                        try:
                            snapshot, snapshot_audit = _pace_vr_snapshot_from_solution(
                                ocp=ocp, solution=solution, states=states, models=models, names=names, args=args,
                                side=side, source_cycle=physical_completed,
                                target_work_j=2 * math.pi * pace.equivalent_mean_torque_nm,
                                config=vr_config, incumbent_weights=pace.weights)
                            metrics["pace_vr_snapshot"] = snapshot
                            metrics["pace_vr_snapshot_audit"] = snapshot_audit
                        except (ValueError, TypeError, KeyError, RuntimeError) as error:
                            metrics["pace_vr_snapshot_audit"] = {"status": "held", "reason": str(error)}
                        metrics["pace_vr_snapshot_time_s"] = time.perf_counter() - snapshot_started
                if solution is not None:
                    records.append({"cycle": physical_completed, **metrics})
                connection.send({"kind": "boundary", "completed_cycles": physical_completed, "metrics": metrics})
                command = connection.recv()
                if command.get("kind") == "stop":
                    return False
                if command.get("kind") != "prepare" or command.get("completed_cycles") != physical_completed:
                    raise RuntimeError("Invalid coordinator prepare command")
                terminal_audit = metrics.get("pace_rt_terminal")
                incoming_decision = command.get("pace_vr_decision")
                incoming_pace_rt = bool(
                    isinstance(incoming_decision, dict)
                    and incoming_decision.get("application_mode")
                    == "terminal_reserve_target_experimental"
                )
                if (isinstance(terminal_audit, dict) and terminal_audit.get("active")
                        and terminal_audit.get("terminal_trust_validated") is not True
                        and not incoming_pace_rt):
                    # Holding the previous PACE-RT parameters after an
                    # extrapolated endpoint would make the next RHO depend on
                    # an unsupported local model.  This numeric deactivation
                    # keeps the compiled NLP intact and leaves its fatigue
                    # objective available.  A freshly accepted PACE-RT fit is
                    # allowed to replace the old one below.
                    from cocofest.optimization.pace_rt_ocp import PaceRtObjectiveBinding
                    binding = getattr(ocp, "task_reserve_binding", None)
                    if isinstance(binding, PaceRtObjectiveBinding):
                        receipt = binding.deactivate(ocp)
                        guard_event = {
                            "status": "deactivated", "kind": "terminal_trust_guard",
                            "source_cycle": binding.source_completed_cycles,
                            "observed_cycle": physical_completed,
                            "terminal_audit": terminal_audit,
                            "parameter_update": receipt,
                            "compiled_nlp_reused": True,
                        }
                        vr_events.append(guard_event)
                        metrics["pace_rt_terminal"]["action"] = "deactivated_before_next_rho"
                        with (root / "pace_vr.jsonl").open("a", encoding="utf-8") as journal:
                            journal.write(json.dumps(guard_event, allow_nan=False) + "\n")
                started = time.perf_counter()
                torque = float(command["equivalent_mean_torque_nm"])
                fatigue_weight_binding = getattr(ocp, "fatigue_weight_binding", None)
                fixed_fatigue_values = payload.get("fatigue_weight_values")
                if fixed_fatigue_values is not None:
                    fixed_fatigue_values = np.asarray(fixed_fatigue_values, dtype=float).reshape(-1)
                    if (fatigue_weight_binding is None or fixed_fatigue_values.shape != fatigue_weight_binding.weights.shape
                            or not np.allclose(fatigue_weight_binding.weights, fixed_fatigue_values, atol=0., rtol=0.)):
                        raise RuntimeError("Configured fixed fatigue ablation was not installed in the compiled NLP.")

                    def apply_weights(_values):
                        # Boundary zero establishes the arm policy's bookkeeping.
                        # It must not overwrite an explicitly fixed numerical
                        # objective (notably the no-fatigue ablation).
                        return {"ocp_cost_updated": True, "fixed_fatigue_objective": True,
                                "weights": fixed_fatigue_values.tolist(),
                                "objective_graph_rebuild_required": False}
                else:
                    apply_weights = (
                        (lambda values: fatigue_weight_binding.update(ocp, values))
                        if fatigue_weight_binding is not None
                        else (lambda values: update_bioptim_fatigue_cost(ocp, values))
                    )
                physio_update_inputs = None
                if (pace.connected and pace.config.adaptation_enabled
                        and pace.config.adaptation_strategy in {"physio_update", "mechanical_sensitivity"}
                        and physical_completed % pace.config.update_every_cycles == 0):
                    # Build only at a due certified boundary. The adapter
                    # uses Cn/F/A/Tau1/Km, PW limits and isokinetic geometry;
                    # it refuses the unsafe A-only capacity shortcut.
                    from cocofest.optimization.physio_update import (
                        build_isokinetic_max_pw_envelope_from_discrete_cycle,
                    )
                    live_envelope = build_isokinetic_max_pw_envelope_from_discrete_cycle(
                        states=states, muscle_models=models,
                        reduced_dynamics=ocp.nlp[0].model.reduced_dynamics,
                        stimulations_per_cycle=int(args.stimulations_per_cycle),
                        angular_velocity_rad_s=float(args.isokinetic_omega),
                        activate_force_length=bool(getattr(
                            ocp.nlp[0].model, "activate_force_length_relationship", True)),
                        activate_force_velocity=bool(getattr(
                            ocp.nlp[0].model, "activate_force_velocity_relationship", True)),
                        activate_passive_force=bool(getattr(
                            ocp.nlp[0].model, "activate_passive_force_relationship", True)),
                    )
                    physio_update_inputs = {
                        key: live_envelope[key] for key in (
                            "current_capacity", "rest_capacity", "alpha_a", "tau_fat",
                            "reference_force", "available_positive_power", "phase_durations",
                            "required_active_work",
                        )
                    }
                previous_torque = pace.equivalent_mean_torque_nm
                event = pace.boundary(physical_completed, ratios, certified=metrics["certified"],
                    equivalent_mean_torque_nm=torque,
                    apply_weights=apply_weights, physio_update_inputs=physio_update_inputs)
                if event["status"] in {"refused", "fatal"}:
                    raise RuntimeError(f"Local weight update refused: {event}")
                if vr_config is not None and command.get("pace_vr_decision") is not None:
                    from cocofest.optimization.pace_rt_ocp import PACE_RT_MODE
                    apply = (_apply_pace_rt_decision
                             if command["pace_vr_decision"].get("application_mode") == PACE_RT_MODE
                             else _apply_pace_vr_weight_decision)
                    apply_options = ({"reserve_priority": reserve_priority}
                                     if apply is _apply_pace_rt_decision else {})
                    vr_event = apply(ocp, pace, command["pace_vr_decision"], certified=metrics["certified"],
                                     current_cycle=physical_completed, **apply_options)
                    vr_events.append(vr_event)
                    with (root / "pace_vr.jsonl").open("a", encoding="utf-8") as journal:
                        journal.write(json.dumps(vr_event, allow_nan=False) + "\n")
                    event["pace_vr"] = vr_event
                if reserve_binding is not None and solution is not None:
                    due = _reserve_update_due(certified=metrics["certified"], has_solution=True,
                                              physical_cycle=physical_completed)
                    if due:
                        try:
                            if not np.isclose(torque, previous_torque, atol=1e-12, rtol=0):
                                raise ValueError("Work target changed from calibrated reserve load.")
                            pulse_widths = None
                            if (reserve_binding.candidate_force_coupling
                                    or reserve_binding.local_pulse_width_cost):
                                controls = solution.decision_controls(to_merge=SolutionMerge.NODES)
                                pulse_widths = np.asarray([
                                    np.asarray(controls[f"last_pulse_width_{name}"], dtype=float).reshape(-1)
                                    for name in names
                                ])
                                if pulse_widths.shape != (len(names), int(args.stimulations_per_cycle)):
                                    raise ValueError("Certified direct PW controls do not span one reserve cycle.")
                                (pulse_width_numerical_tolerance_s,
                                 pulse_width_numerical_tolerance_audit) = _pulse_width_replay_tolerance_from_nlp(
                                     ocp=ocp, muscle_names=names, nlp_tolerance=args.nlp_tolerance,
                                 )
                            else:
                                pulse_width_numerical_tolerance_s = None
                                pulse_width_numerical_tolerance_audit = None
                            forces, margin_model, receipt, pulse_width_force_model = _certified_mechanical_reserve_profile(
                                states=states, models=models,
                                reduced_dynamics=ocp.nlp[0].model.reduced_dynamics,
                                phase_count=int(args.stimulations_per_cycle),
                                omega=float(args.isokinetic_omega),
                                target_work_j=2.0 * math.pi * torque,
                                binding=reserve_binding, source_cycle=physical_completed,
                                calibration_policy=args.experimental_mechanical_reserve_calibration_policy,
                                pulse_widths=pulse_widths,
                                pulse_width_numerical_tolerance_s=pulse_width_numerical_tolerance_s,
                                pulse_width_numerical_tolerance_audit=pulse_width_numerical_tolerance_audit,
                            )
                            local_terms = None
                            if reserve_binding.local_pulse_width_cost:
                                local_terms = reserve_binding.local_pulse_width_cost_terms(
                                    initial_states=np.asarray(receipt["terminal_slow_states"], dtype=float),
                                    forces=forces, margin_model=margin_model,
                                    pulse_width_force_model=pulse_width_force_model,
                                )
                                receipt["local_pw_gradient_l2"] = float(np.linalg.norm(local_terms["gradient"]))
                                receipt["local_pw_force_gradient_l2"] = float(np.linalg.norm(local_terms["force_gradient"]))
                                receipt["local_pw_trust_us"] = float(
                                    reserve_binding.local_pulse_width_trust_s * 1e6)
                            receipt["parameter_update"] = reserve_binding.update(
                                ocp, forces=forces, margin_model=margin_model,
                                pulse_width_force_model=pulse_width_force_model,
                                local_pulse_width_terms=local_terms)
                            receipt["source_solver_status"] = metrics.get("native_solver_status")
                            receipt["source_feasibility"] = metrics.get("feasibility")
                        except (ValueError, TypeError, KeyError, RuntimeError) as error:
                            receipt = {"status": "held", "source_cycle": physical_completed,
                                       "reason": "certified_profile_reconstruction_unavailable",
                                       "detail": f"{type(error).__name__}: {error}",
                                       "active_previous_profile": bool(reserve_binding.activation)}
                        reserve_events.append(receipt)
                    else:
                        reserve_events.append({"status": "held", "source_cycle": physical_completed,
                                               "reason": "uncertified_source" if not metrics["certified"] else "twenty_cycle_cadence_not_due",
                                               "active_previous_profile": bool(reserve_binding.activation)})
                    metrics["mechanical_reserve"] = reserve_events[-1]
                # ``advance_window`` runs after every solve and overwrites the
                # active bounds from its shifted state.  Consequently a target
                # changed at a block boundary must be reasserted *before every
                # subsequent solve*, not only on the boundary where its value
                # changes.  This is a numeric bound update, not a rebuild.
                set_terminal_eprod_target(ocp, 2 * math.pi * torque)
                if frozen_retry is not None:
                    frozen_retry.capture()
                checkpoint_export = None
                if physical_completed in requested_checkpoint_cycles:
                    # This is the shifted *next* window, after all numerical
                    # updates selected at the current certified boundary.
                    from cocofest.simulation.rho_restart_checkpoint import export_prepared_checkpoint

                    configured_model_path = (
                        Path(configured_receipt["model_config_path"])
                        if configured_receipt is not None else None
                    )
                    checkpoint_export = export_prepared_checkpoint(
                        root.parent / "checkpoints" / f"cycle-{physical_completed}" / f"{side}.npz",
                        ocp, completed_cycles=physical_completed,
                        model_path=configured_model_path,
                    )
                connection.send({"kind": "prepared", "completed_cycles": physical_completed,
                                 "weight_event": event, "checkpoint_export": checkpoint_export,
                                 "prepare_time_s": time.perf_counter() - started})
                command = connection.recv()
                if command.get("kind") == "stop":
                    return False
                if command != {"kind": "solve", "completed_cycles": physical_completed}:
                    raise RuntimeError("Invalid coordinator solve command")
                # The native RHO already shifted bounds/initial guesses before
                # this callback. Keep its active model and drop historical graph
                # references not needed by compact output.
                ocp.all_models.clear()
                return True

            nmpc.solve_fes_nmpc(callback, solver=solver, total_cycles=payload["cycles"],
                external_force=runtime["external_force"], cycle_solutions=MultiCyclicCycleSolutions.ALL_CYCLES,
                get_all_iterations=False, cyclic_options={"states": {}},
                # The local controller publishes its terminal failure before
                # returning False. A native attempt limit would stop earlier
                # and send `finished` while the parent awaits `boundary`.
                max_consecutive_failing=math.inf if frozen_retry is not None else args.max_consecutive_failing,
                compact_solution_output=True)
            result = {"side": side, "pid": os.getpid(), "cycles": records,
                      "solver_cpu_affinity": list(effective_affinity) if effective_affinity is not None else None,
                      "configured_model": configured_receipt,
                      "validated_cycles": sum(item["certified"] for item in records),
                      "success": len(records) == payload["cycles"] and all(item["certified"] for item in records),
                      "recovery_attempts": recovery_attempts,
                      "weights_audit": pace.events,
                      "fatigue_weight_binding": (
                          nmpc.fatigue_weight_binding.summary()
                          if getattr(nmpc, "fatigue_weight_binding", None) is not None
                          else None
                      ),
                      "mechanical_reserve_binding": reserve_binding.summary() if reserve_binding is not None else None,
                      "max_pw_work_binding": (nmpc.max_pw_work_binding.summary()
                          if getattr(nmpc, "max_pw_work_binding", None) is not None else None),
                      "mechanical_reserve_events": reserve_events,
                      "pace_vr_events": vr_events,
                      "timing_s": _timing_summary(item["solver_time_s"] for item in records),
                      "warm_timing_s": _timing_summary(item["solver_time_s"] for item in records[2:])}
            (root / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            connection.send({"kind": "finished", "completed_cycles": len(records), "result": result})
        except BaseException as error:
            failure = {"kind": "error", "error": f"{type(error).__name__}: {error}",
                       "traceback": traceback.format_exc()}
            (root / "error.json").write_text(json.dumps(failure, indent=2) + "\n")
            connection.send(failure)
        finally:
            connection.close()


class IndependentArmProcessCoordinator:
    """Build the two NLPs inside spawned workers and rendezvous every cycle."""

    def __init__(self, payload, *, worker_target=None):
        self.payload = dict(payload)
        self._worker_target = worker_target or _arm_worker
        from cocofest.optimization.pace_vr_async import validate_pace_vr_configuration
        self.pace_vr_config = validate_pace_vr_configuration(self.payload)
        _requested_restart_checkpoint_cycles(self.payload)
        if self.payload.get("solver", "ipopt") != "ipopt":
            raise ValueError("Process RHO-PACE currently supports IPOPT/MA57; ACADOS cost updates remain unvalidated")
        if self.payload.get("formulation", "isokinetic") != "isokinetic" or self.payload.get("cycles_per_window", 1) != 1:
            raise ValueError("Process RHO-PACE requires isokinetic one-cycle windows")
        if self.payload.get("parallel", True) is not True:
            raise ValueError("Process RHO-PACE runs the arms in parallel; parallel must be true")
        if any(self.payload.get(f"{side}_runner_config") is not None for side in ARMS):
            raise ValueError("Use explicit right_driver_arguments/left_driver_arguments for process RHO-PACE; opaque runner configs are unsupported")
        # Fail before spawning workers if a variant is malformed or missing.
        # The workers re-resolve it independently because spawned processes do
        # not share Python state with their parent.
        _configured_payload_model(self.payload, "right")
        _configured_payload_model(self.payload, "left")
        omega = float(self.payload.get("omega_rad_s", -2 * math.pi))
        if not math.isfinite(omega) or omega >= 0:
            raise ValueError("omega_rad_s must be finite and negative")
        # Validate the shape before workers are spawned.  Availability itself
        # is checked in each child because schedulers may give them a narrower
        # CPU set than the parent.
        requested_affinity = self.payload.get("solver_cpu_affinity")
        if requested_affinity is not None:
            if isinstance(requested_affinity, Mapping):
                unknown = set(requested_affinity) - set(ARMS)
                if unknown:
                    raise ValueError(f"Unknown solver_cpu_affinity arm(s): {sorted(unknown)}")
                for side in ARMS:
                    _worker_solver_affinity(self.payload, side)
            else:
                _worker_solver_affinity(self.payload, "right")

    def run_with_resistance_pace_to_directory(self, output_root, pace, *, cycles=1):
        if type(cycles) is not int or cycles < 1:
            raise ValueError("cycles must be a positive integer")
        if self.pace_vr_config is not None and (pace.config.capacity_feedback or pace.config.initial_split_policy != "manual"):
            raise ValueError("Experimental PACE-VR requires fixed manual arm work targets.")
        if float(self.payload.get("experimental_mechanical_reserve_weight", 0.0)) > 0.0:
            if pace.config.capacity_feedback or pace.config.initial_split_policy != "manual":
                raise ValueError("Experimental reserve requires a fixed manual arm load split.")
            local = self.payload.get("muscle_pace", {})
            if local.get("adaptation_enabled", True) is not False:
                raise ValueError("Experimental reserve requires disabled muscle-weight adaptation.")
            for side in ARMS:
                values = local.get(f"{side}_initial_weights", {})
                if values and any(float(value) != 1.0 for value in values.values()):
                    raise ValueError("Experimental reserve requires fixed unit fatigue weights.")
        root = Path(output_root).resolve()
        if (root / "summary.json").exists() or any((root / side / "weights.jsonl").exists() for side in ARMS):
            raise FileExistsError("Use a fresh output directory for each process RHO-PACE campaign")
        root.mkdir(parents=True, exist_ok=True)
        payload = {**self.payload, "cycles": cycles}
        local_config = dict(payload.get("muscle_pace", {}))
        local_config.setdefault("update_every_cycles", pace.config.update_every_cycles)
        if local_config["update_every_cycles"] != pace.config.update_every_cycles:
            raise ValueError("Muscle weights and work allocation must use the same block cadence")
        payload["muscle_pace"] = local_config
        for side, torque in pace.equivalent_mean_torques.items():
            payload[f"{side}_equivalent_mean_torque_nm"] = torque
        (root / "configuration.json").write_text(json.dumps({**payload, "resistance_pace": asdict(pace.config)}, indent=2) + "\n")
        context = mp.get_context("spawn")
        parents, processes = {}, {}
        for side in ARMS:
            parent, child = context.Pipe()
            process = context.Process(target=self._worker_target, args=(child, side, payload, str(root)),
                                      name=f"cocofest-{side}-rho")
            process.start()
            child.close()
            parents[side], processes[side] = parent, process
        history, preparation, initial_split_events = [], [], []
        from cocofest.optimization.pace_vr_async import BilateralPaceVrAsync
        vr_supervisor = BilateralPaceVrAsync(self.pace_vr_config) if self.pace_vr_config is not None else None
        started = time.perf_counter()
        failure = None
        completed = 0
        timeout = float(payload.get("worker_timeout_seconds", 600.))
        try:
            for completed in range(cycles + 1):
                reports = _receive_pair(parents, kind="boundary", completed_cycles=completed,
                                        timeout_seconds=timeout, processes=processes)
                if completed:
                    history.append({"cycle": completed, "arms": {side: reports[side]["metrics"] for side in ARMS},
                                    "pair_solve_and_transfer_wall_time_s": time.perf_counter() - solve_started})
                certified = all(reports[side]["metrics"].get("certified") is True for side in ARMS)
                if not certified or completed == cycles:
                    if not certified:
                        failure = f"At least one arm failed certification after cycle {completed}"
                    for connection in parents.values():
                        connection.send({"kind": "stop"})
                    break
                previous = pace.equivalent_mean_torques
                if (completed == 1
                        and pace.config.initial_split_policy == "capacity_fatigability_after_first_cycle"):
                    from cocofest.optimization.bilateral_initial_split import recommend_capacity_fatigability_split
                    decision = recommend_capacity_fatigability_split(
                        reports["right"]["metrics"].get("capacity_fatigability_initial_split", {}),
                        reports["left"]["metrics"].get("capacity_fatigability_initial_split", {}),
                        total_equivalent_mean_torque_nm=pace.config.total_equivalent_mean_torque_nm,
                        minimum_arm_equivalent_mean_torque_nm=pace.config.minimum_arm_equivalent_mean_torque_nm,
                    )
                    initial_split_events.append(pace.apply_initial_split_decision(decision))
                if completed:
                    event = pace.observe(completed - 1, reports["right"]["metrics"], reports["left"]["metrics"])
                    if event["status"] == "refused":
                        raise RuntimeError(f"Allocation refused: {event}")
                following = pace.equivalent_mean_torques
                vr_commands = vr_supervisor.boundary(completed, reports) if vr_supervisor is not None else {}
                if vr_supervisor is not None:
                    (root / "pace_vr_supervisor.json").write_text(json.dumps(vr_supervisor.events, indent=2, allow_nan=False) + "\n")
                for side, connection in parents.items():
                    connection.send({"kind": "prepare", "completed_cycles": completed,
                                     "equivalent_mean_torque_nm": following[side],
                                     "previous_equivalent_mean_torque_nm": previous[side],
                                     "pace_vr_decision": vr_commands.get(side)})
                prepared = _receive_pair(parents, kind="prepared", completed_cycles=completed,
                                         timeout_seconds=timeout, processes=processes)
                checkpoint_export = _publish_prepared_pair_export(root, completed, prepared)
                preparation.append({"completed_cycles": completed, "arms": prepared,
                                    "checkpoint_export": checkpoint_export})
                solve_started = time.perf_counter()
                for connection in parents.values():
                    connection.send({"kind": "solve", "completed_cycles": completed})
            finished = _receive_pair(parents, kind="finished", completed_cycles=completed,
                                     timeout_seconds=timeout, processes=processes)
            results = {side: message["result"] for side, message in finished.items()}
            summary = {"architecture": "two-process-unilateral-isokinetic-rho-pace", "parallel": True,
                       "cycle_synchronous": True, "requested_cycles": cycles,
                       "completed_rho_cycles": completed, "success": failure is None,
                       "failure": failure, "wall_time_s": time.perf_counter() - started,
                       "arms": results, "cycles": history, "preparations": preparation,
                       "initial_split_events": initial_split_events,
                       "resistance_pace": pace.audit(),
                       "pace_vr_supervisor": vr_supervisor.events if vr_supervisor is not None else None,
                       "pair_cycle_wall_timing_s": _timing_summary(item["pair_solve_and_transfer_wall_time_s"] for item in history)}
            (root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
            return summary
        except BaseException as error:
            (root / "failure.json").write_text(json.dumps({"success": False,
                "error": f"{type(error).__name__}: {error}", "cycles": history,
                "preparations": preparation, "initial_split_events": initial_split_events,
                "resistance_pace": pace.audit()}, indent=2, allow_nan=False) + "\n")
            raise
        finally:
            if vr_supervisor is not None:
                vr_supervisor.close()
            for connection in parents.values():
                try:
                    connection.send({"kind": "stop"})
                except (BrokenPipeError, EOFError, OSError):
                    pass
            for process in processes.values():
                process.join(timeout=2.)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2.)
            for connection in parents.values():
                connection.close()

    def run_to_directory(self, output_root, *, cycles=1):
        right = float(self.payload.get("right_equivalent_mean_torque_nm", .1))
        left = float(self.payload.get("left_equivalent_mean_torque_nm", .1))
        total = right + left
        config = IndependentArmRhoPaceConfig(total_equivalent_mean_torque_nm=total,
            initial_right_fraction=right / total if total else .5, capacity_feedback=False,
            update_every_cycles=self.payload.get("muscle_pace", {}).get("update_every_cycles", 10))
        return self.run_with_resistance_pace_to_directory(output_root, IndependentArmResistancePace(config), cycles=cycles)


def build_process_independent_arms(payload):
    """Public GUI/CLI factory; compilation occurs once per spawned child."""
    return IndependentArmProcessCoordinator(payload)
