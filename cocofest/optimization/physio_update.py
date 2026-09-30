"""Experimental state-conditioned physiological weight proposals.

This numerical kernel is deliberately not enabled in a public RHO runner.
It combines an exact, prescribed-force Ding fatigue challenge with an
isokinetic *work* contribution calculation. The caller must supply certified
current capacities and an independently validated available-force envelope.
The envelope is a work-opportunity approximation, not a viability certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import factorial, log

import numpy as np

from .endurance_weight_supervisor import _project_centered_logs


def build_isokinetic_max_pw_envelope(
    *,
    terminal_states,
    reference_force,
    intervals,
    pulse_width_parameters,
    angular_velocity_rad_s,
    integration_substeps=16,
):
    """Build Physio-U inputs from one certified isokinetic RHO boundary.

    ``terminal_states`` is the full five-state Ding state at the *end* of a
    certified RHO cycle, in ``(Cn, F, A, Tau1, Km)`` order.  The available
    force envelope is then obtained by continuing those exact states through
    one cycle with PW at its declared upper bound, independently for each
    muscle.  This is substantially stronger than inferring capacity from
    ``A/A_rest`` alone: it retains calcium, force, fatigue-modified Tau1/Km,
    the current geometry and the actual PW constraint.

    The envelope deliberately does not solve coupled torque constraints or
    find a PW-feasible movement.  It is therefore an individual force
    *opportunity* used only by the Shapley weight heuristic, never an
    attainable-work or endurance certificate.
    """
    # Local import avoids making the small NumPy physiological-weight module
    # import SciPy's scalar inversion machinery on ordinary static-weight use.
    from .adaptive_moment_rollout import (
        FULL_DING_STATE_NAMES,
        propagate_ding_pulse_width_interval,
    )

    states = np.asarray(terminal_states, dtype=float)
    force = np.asarray(reference_force, dtype=float)
    muscle_count = len(pulse_width_parameters)
    phase_count = len(intervals)
    if (states.shape != (muscle_count, len(FULL_DING_STATE_NAMES))
            or force.shape != (muscle_count, phase_count)
            or not np.all(np.isfinite(states)) or not np.all(np.isfinite(force))):
        raise ValueError("terminal_states and reference_force must be finite, with matching muscle/phase shapes.")
    if np.any(states[:, 2:] <= 0) or np.any(states[:, 1] < -1e-10) or np.any(force < -1e-10):
        raise ValueError("Certified Ding states must retain positive A, Tau1, Km and nonnegative force.")
    omega = float(angular_velocity_rad_s)
    if not np.isfinite(omega) or omega == 0:
        raise ValueError("angular_velocity_rad_s must be finite and nonzero.")
    if isinstance(integration_substeps, bool) or int(integration_substeps) != integration_substeps:
        raise ValueError("integration_substeps must be a positive integer.")
    integration_substeps = int(integration_substeps)
    if integration_substeps < 1:
        raise ValueError("integration_substeps must be a positive integer.")

    duration = np.empty(phase_count)
    moment_coefficients = np.empty((muscle_count, phase_count))
    available_force = np.empty_like(force)
    current = states.copy()
    for phase, interval in enumerate(intervals):
        if len(interval.calcium_amplitudes) != muscle_count:
            raise ValueError("Every interval must identify every muscle.")
        duration[phase] = float(interval.duration)
        moment_coefficients[:, phase] = np.asarray(interval.moment_coefficients, dtype=float)
        for muscle in range(muscle_count):
            current[muscle] = propagate_ding_pulse_width_interval(
                current[muscle],
                pulse_width=pulse_width_parameters[muscle].pulse_width_max,
                duration=duration[phase],
                calcium_amplitude=interval.calcium_amplitudes[muscle],
                mechanical_gain=interval.mechanical_gains[muscle],
                parameters=pulse_width_parameters[muscle],
                integration_substeps=integration_substeps,
            )
            available_force[muscle, phase] = current[muscle, 1]
    if not np.all(np.isfinite(moment_coefficients)) or not np.all(np.isfinite(available_force)):
        raise ValueError("Isokinetic mechanics or max-PW propagation produced a nonfinite envelope.")

    # The muscle contribution to the reduced model's E_prod_dot is
    # ``omega * moment_coefficient * F``.  This independent PW-max replay is
    # only a positive-work opportunity, not a certified attainable power.
    available_positive_power = np.maximum(
        0., omega * moment_coefficients * np.maximum(available_force, 0.)
    )
    reference_total_power = omega * np.sum(moment_coefficients * np.maximum(force, 0.), axis=0)
    required_active_work = float(np.dot(np.maximum(reference_total_power, 0.), duration))
    if not np.isfinite(required_active_work) or required_active_work <= 0:
        raise ValueError("The certified reference cycle must contain positive active isokinetic work.")
    return {
        "current_capacity": states[:, 2].copy(),
        "rest_capacity": np.asarray(
            [parameter.fatigue.a_rest for parameter in pulse_width_parameters], dtype=float,
        ),
        "alpha_a": np.asarray(
            [parameter.fatigue.alpha_a for parameter in pulse_width_parameters], dtype=float,
        ),
        "tau_fat": np.asarray(
            [parameter.fatigue.tau_fat for parameter in pulse_width_parameters], dtype=float,
        ),
        "reference_force": np.maximum(force, 0.).copy(),
        "available_positive_power": available_positive_power,
        "phase_durations": duration,
        "required_active_work": required_active_work,
        "envelope_force": available_force,
        "moment_coefficients": moment_coefficients,
        "force_envelope_model": "full_five_state_ding_one_cycle_pw_max_independent_muscles_v1",
        "mechanical_demand_model": "certified_cycle_positive_total_isokinetic_muscle_work_v1",
        "attainable_work_certified": False,
        "endurance_prediction": False,
    }


def build_isokinetic_max_pw_envelope_from_discrete_cycle(
    *,
    states,
    muscle_models,
    reduced_dynamics,
    stimulations_per_cycle,
    angular_velocity_rad_s,
    activate_force_length=True,
    activate_force_velocity=True,
    activate_passive_force=True,
    pulse_width_max_s=.0006,
    integration_substeps=16,
):
    """Extract a PW-max envelope from a certified discrete isokinetic cycle.

    This is the live-RHO bridge.  State vectors must include all collocation
    nodes and have exactly ``1 + n_phase * n_nodes_per_phase`` samples.  The
    end node of each phase supplies the certified reference force and the
    geometry used for the subsequent short PW-max replay.  Geometry is held
    piecewise constant at that end node; the returned audit names this
    approximation so it cannot be mistaken for a collocation replay.
    """
    from .adaptive_moment_rollout import DingPulseWidthParameters, MomentTrackingInterval

    models = tuple(muscle_models)
    names = tuple(str(model.muscle_name) for model in models)
    if not models or len(set(names)) != len(names):
        raise ValueError("muscle_models must contain uniquely named models.")
    if type(stimulations_per_cycle) is not int or stimulations_per_cycle < 1:
        raise ValueError("stimulations_per_cycle must be a positive integer.")
    pulse_width_max_s = float(pulse_width_max_s)
    if not np.isfinite(pulse_width_max_s) or pulse_width_max_s <= 0:
        raise ValueError("pulse_width_max_s must be finite and positive.")
    required = ("theta", *(
        f"{component}_{name}" for name in names for component in ("Cn", "F", "A", "Tau1", "Km")
    ))
    vectors = {}
    for key in required:
        if key not in states:
            raise ValueError(f"Certified cycle is missing state {key!r}.")
        value = np.asarray(states[key], dtype=float).reshape(-1)
        if not value.size or not np.all(np.isfinite(value)):
            raise ValueError(f"Certified cycle state {key!r} must be finite and nonempty.")
        vectors[key] = value
    columns = len(vectors["theta"])
    if any(len(value) != columns for value in vectors.values()):
        raise ValueError("Certified cycle states must share one node count.")
    if (columns - 1) % stimulations_per_cycle:
        raise ValueError("State nodes do not partition exactly into stimulation phases.")
    nodes_per_phase = (columns - 1) // stimulations_per_cycle
    if nodes_per_phase < 1:
        raise ValueError("Certified cycle has no state endpoint per stimulation phase.")
    endpoints = np.arange(1, stimulations_per_cycle + 1) * nodes_per_phase
    terminal = np.asarray([
        [vectors[f"{component}_{name}"][-1] for component in ("Cn", "F", "A", "Tau1", "Km")]
        for name in names
    ])
    reference_force = np.asarray([vectors[f"F_{name}"][endpoints] for name in names])
    omega = float(angular_velocity_rad_s)
    if not np.isfinite(omega) or omega == 0:
        raise ValueError("angular_velocity_rad_s must be finite and nonzero.")
    phase_duration = (2. * np.pi / abs(omega)) / stimulations_per_cycle
    intervals = []
    for endpoint in endpoints:
        theta = float(vectors["theta"][endpoint])
        force_length, force_velocity, passive_force = reduced_dynamics.muscle_relationships(theta, omega)
        force_length = np.asarray(force_length, dtype=float)
        force_velocity = np.asarray(force_velocity, dtype=float)
        passive_force = np.asarray(passive_force, dtype=float)
        if not activate_force_length:
            force_length = np.ones(len(models))
        if not activate_force_velocity:
            force_velocity = np.ones(len(models))
        if not activate_passive_force:
            passive_force = np.zeros(len(models))
        gains = force_length * force_velocity + passive_force
        coefficients = np.asarray(
            reduced_dynamics.coefficient_values(theta)["muscle_effectiveness"], dtype=float,
        )
        if coefficients.shape != (len(models),) or gains.shape != (len(models),):
            raise ValueError("Reduced dynamics does not match the four-muscle Ding model order.")
        intervals.append(MomentTrackingInterval(
            duration=phase_duration,
            calcium_amplitudes=tuple(float(model.post_stimulation_amplitude()) for model in models),
            mechanical_gains=tuple(float(value) for value in gains),
            moment_coefficients=tuple(float(value) for value in coefficients),
            target_moments=(0.,) * len(models),
        ))
    # In the live isokinetic OCP E_prod is the exact reduced inverse-dynamics
    # work state.  It is the task demand imposed by the NLP; summing only
    # individual muscle moments can omit model terms and is not equivalent.
    # Keep a direct-moment fallback solely for archive fixtures predating
    # E_prod, explicitly labelled as an approximation.
    eprod = states.get("E_prod")
    if eprod is not None:
        eprod = np.asarray(eprod, dtype=float).reshape(-1)
        if eprod.shape != (columns,) or not np.all(np.isfinite(eprod)):
            raise ValueError("Certified E_prod must be finite and share the state-node count.")
        required_active_work = float(eprod[-1] - eprod[0])
        if required_active_work <= 0:
            raise ValueError("Certified E_prod must show positive active isokinetic work.")
        demand_model = "certified_E_prod_state_difference_v1"
    else:
        required_active_work = None
        demand_model = "certified_cycle_positive_total_isokinetic_muscle_work_fallback_v1"
    envelope = build_isokinetic_max_pw_envelope(
        terminal_states=terminal,
        reference_force=reference_force,
        intervals=tuple(intervals),
        pulse_width_parameters=tuple(
            DingPulseWidthParameters.from_model(model, pulse_width_max=pulse_width_max_s) for model in models
        ),
        angular_velocity_rad_s=omega,
        integration_substeps=integration_substeps,
    )
    if required_active_work is not None:
        # The force envelope builder owns the other current-state inputs; only
        # the task work source changes for a live isokinetic OCP.
        envelope["required_active_work"] = required_active_work
        envelope["mechanical_demand_model"] = demand_model
    return {
        **envelope,
        "source_state_sampling": "certified_phase_endpoints",
        "envelope_geometry_sampling": "phase_endpoint_piecewise_constant_v1",
        "source_state_columns": columns,
        "source_nodes_per_phase": nodes_per_phase,
        "source_muscle_names": names,
    }


def _array(value, shape, name, *, positive=False, nonnegative=False):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    if positive and np.any(result <= 0):
        raise ValueError(f"{name} must be positive.")
    if nonnegative and np.any(result < 0):
        raise ValueError(f"{name} must be nonnegative.")
    return result


def prescribed_force_fatigue_challenge(
    current_capacity, rest_capacity, alpha_a, tau_fat, force, phase_durations,
):
    """Exact A-flow under one piecewise-constant force cycle and at zero force.

    Rest capacities are immutable physiology, never replaced with the current
    A. The recovery-only trajectory isolates force-induced fatigue from net
    change: a tired muscle can recover overall while stimulation still incurs
    a fatigue cost. The result includes every phase endpoint for domain checks.
    No future PW, task feasibility, or endurance is predicted here.
    """
    force = np.asarray(force, dtype=float)
    if force.ndim != 2 or min(force.shape) < 1:
        raise ValueError("force must have shape (muscles, phases).")
    count, phases = force.shape
    force = _array(force, force.shape, "force", nonnegative=True)
    current = _array(current_capacity, (count,), "current_capacity", positive=True)
    rest = _array(rest_capacity, (count,), "rest_capacity", positive=True)
    if np.any(current > rest * (1 + 1e-12)):
        raise ValueError("current_capacity must not exceed its rest capacity.")
    alpha = _array(alpha_a, (count,), "alpha_a")
    if np.any(alpha > 0):
        raise ValueError("alpha_a must be nonpositive.")
    tau = _array(tau_fat, (count,), "tau_fat", positive=True)
    dt = _array(phase_durations, (phases,), "phase_durations", positive=True)
    history = np.empty((phases + 1, count))
    history[0] = current
    for phase in range(phases):
        loss = -np.expm1(-dt[phase] / tau)
        history[phase + 1] = (
            history[phase] + loss * (rest - history[phase] + alpha * tau * force[:, phase])
        )
    free = rest + np.exp(-dt.sum() / tau) * (current - rest)
    decrement = (free - history[-1]) / rest
    damage_free = 1 - free / rest
    # Algebraically equivalent to damage_forced**2-damage_free**2, but stable
    # when the one-cycle stimulus causes only a very small capacity decrement.
    excess_squared_fatigue = decrement * (2 * damage_free + decrement)
    domain_valid = bool(np.all(history > 0))
    return {
        "domain_valid": domain_valid,
        "capacity_history": history,
        "recovery_only_terminal_capacity": free,
        "stimulus_capacity_decrement_ratio": decrement,
        "excess_squared_fatigue": excess_squared_fatigue,
        "duration_seconds": float(dt.sum()),
        "normalization": "immutable_A_rest",
        "endurance_prediction": False,
    }


def isokinetic_work_shapley(available_positive_power, phase_durations, required_work):
    """Give each muscle credit for useful work in all recruitment coalitions.

    The coalition game is v(S)=min(required_work, sum_i_in_S integral P_i dt).
    Four muscles require only 16 coalition evaluations. A muscle with positive
    work opportunity has positive credit whenever required_work > 0, including
    when no muscle is uniquely essential. Common passive work must be removed
    by the caller; it must not be attributed independently to each muscle.

    Instantaneous torque bounds, residual-force coupling and PW transitions
    are not solved here. Therefore this is a mechanical weighting heuristic,
    not attainable maximum work or a feasibility test.
    """
    power = np.asarray(available_positive_power, dtype=float)
    if power.ndim != 2 or not 1 <= power.shape[0] <= 10 or power.shape[1] < 1:
        raise ValueError("available_positive_power must have 1..10 muscles and at least one phase.")
    power = _array(power, power.shape, "available_positive_power", nonnegative=True)
    count, phases = power.shape
    dt = _array(phase_durations, (phases,), "phase_durations", positive=True)
    required_work = float(required_work)
    if not np.isfinite(required_work) or required_work <= 0:
        raise ValueError("required_work must be finite and positive.")
    opportunities = power @ dt
    values = np.zeros(1 << count)
    for mask in range(1, 1 << count):
        values[mask] = min(required_work, sum(opportunities[i] for i in range(count) if mask & (1 << i)))
    credit = np.zeros(count)
    for muscle in range(count):
        bit = 1 << muscle
        for mask in range(1 << count):
            if mask & bit:
                continue
            size = mask.bit_count()
            coefficient = factorial(size) * factorial(count - size - 1) / factorial(count)
            credit[muscle] += coefficient * (values[mask | bit] - values[mask])
    return {
        "work_opportunities_j": opportunities,
        "mechanical_credit_j": credit,
        "coalition_work_values_j": values,
        "required_active_work_j": required_work,
        "opportunity_to_demand_ratio": float(opportunities.sum() / required_work),
        "mechanical_model": "capped_additive_work_opportunity_shapley_v1",
        "attainable_work_certified": False,
    }


@dataclass(frozen=True)
class PhysioUpdateConfig:
    """Conservative proposal settings, distinct from the PACE QP controller."""

    update_every_cycles: int = 20
    smoothing: float = 0.2
    max_log_step: float = log(1.1)
    deadband_log: float = log(1.01)
    min_relative_weight: float = 0.25
    max_relative_weight: float = 4.0

    def __post_init__(self):
        if type(self.update_every_cycles) is not int or self.update_every_cycles < 1:
            raise ValueError("update_every_cycles must be a positive integer.")
        for name in ("smoothing", "max_log_step", "deadband_log", "min_relative_weight", "max_relative_weight"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.smoothing > 1 or not self.min_relative_weight <= 1 <= self.max_relative_weight:
            raise ValueError("smoothing must be <=1 and relative bounds must contain 1.")


def propose_mechanical_sensitivity_weights(
    *, available_positive_power, phase_durations, required_active_work,
    completed_cycles=None, certified=None, incumbent_weights=None, config=None,
):
    """Propose a mechanics-only ablation of the squared-fatigue weights.

    For ``sum_m w_m (1-A_m/A_rest,m)^2``, a diagonal approximation of the
    square of a mechanical-loss sensitivity gives ``w_m ∝ b_m²``.  ``b_m`` is
    the normalized Shapley work credit under the same isokinetic task.  Ding
    fatigability is deliberately *not* multiplied into this weight: it is
    already present in the propagated ``A`` state and hence in the squared
    objective.  This is an experimental alternative to ``physio_update``, not
    a replacement and not an endurance claim.
    """
    config = config or PhysioUpdateConfig()
    incumbent = None
    if incumbent_weights is not None:
        incumbent = np.asarray(incumbent_weights, dtype=float)
        if incumbent.ndim != 1 or not incumbent.size:
            raise ValueError("incumbent_weights must be a nonempty vector.")
        incumbent = _array(incumbent, incumbent.shape, "incumbent_weights", positive=True)
        if (np.any(incumbent < config.min_relative_weight) or np.any(incumbent > config.max_relative_weight)
                or abs(np.mean(np.log(incumbent))) > 1e-10):
            raise ValueError("incumbent_weights must have geometric mean one and respect configured bounds.")
    mechanical = isokinetic_work_shapley(
        available_positive_power, phase_durations, required_active_work,
    )
    credit = np.asarray(mechanical["mechanical_credit_j"], dtype=float)
    result = {
        "policy": "mechanical_sensitivity_squared_v1_experimental",
        "status": "held",
        "mechanical": mechanical,
        "endurance_improvement_validated": False,
    }
    if completed_cycles is not None:
        if type(completed_cycles) is not int or completed_cycles < 0:
            raise ValueError("completed_cycles must be a nonnegative integer.")
        result["completed_cycles"] = completed_cycles
        if certified is not True:
            return {**result, "weights": incumbent, "reason": "uncertified_source_state"}
        if completed_cycles % config.update_every_cycles:
            return {**result, "weights": incumbent, "reason": "update_not_due"}
    if not np.all(np.isfinite(credit)) or np.any(credit <= 0):
        return {**result, "weights": incumbent, "reason": "zero_or_invalid_mechanical_credit"}
    sensitivity = credit / credit.sum()
    raw = sensitivity ** 2
    max_normalized = raw / raw.max()
    logs = np.log(raw)
    target = np.asarray(_project_centered_logs(
        logs - logs.mean(), lower=log(config.min_relative_weight), upper=log(config.max_relative_weight),
    ))
    result = {
        **result,
        "status": "proposed",
        "reason": "mechanical_sensitivity_diagonal_square",
        "mechanical_sensitivity": sensitivity,
        "raw_scores": raw,
        "raw_max_normalized_scores": max_normalized,
        "projection_changed_ratios": bool(np.max(abs(target - (logs - logs.mean()))) > 1e-12),
        "projected_target_weights": np.exp(target),
    }
    if incumbent is None:
        return {**result, "weights": np.exp(target)}
    if incumbent.shape != target.shape:
        raise ValueError("incumbent_weights and mechanical inputs must use the same muscle count.")
    step = config.smoothing * (target - np.log(incumbent))
    step *= min(1.0, config.max_log_step / max(float(np.max(abs(step))), 1e-300))
    if np.max(abs(step)) < config.deadband_log:
        return {**result, "status": "held", "weights": incumbent, "reason": "log_deadband"}
    return {**result, "weights": np.exp(np.log(incumbent) + step)}


def propose_physio_update(
    *, completed_cycles, certified, incumbent_weights, current_capacity,
    rest_capacity, alpha_a, tau_fat, reference_force, available_positive_power,
    phase_durations, required_active_work, config=None,
):
    """Return a bounded experimental proposal; no OCP mutation is performed.

    Raw score = mechanical work credit * force-induced excess squared fatigue.
    At rest the latter is the squared normalized capacity decrement, matching
    the structure of the article's contribution-times-fatigability-squared
    heuristic. With fatigue it includes the current Ding state and recovery.

    This is a research hypothesis: weighting the existing squared-fatigue RHO
    cost by this score may over-protect damaged muscles and must be tested.
    A zero score is reported explicitly and holds the incumbent; this kernel
    does not invent a positive physiological contribution for a dummy muscle.
    """
    config = config or PhysioUpdateConfig()
    incumbent = np.asarray(incumbent_weights, dtype=float)
    if incumbent.ndim != 1 or not incumbent.size:
        raise ValueError("incumbent_weights must be a nonempty vector.")
    incumbent = _array(incumbent, incumbent.shape, "incumbent_weights", positive=True)
    if (np.any(incumbent < config.min_relative_weight) or np.any(incumbent > config.max_relative_weight)
            or abs(np.mean(np.log(incumbent))) > 1e-10):
        raise ValueError("incumbent_weights must have geometric mean one and respect configured bounds.")
    result = {"policy": "physio_update_work_v1_experimental", "weights": incumbent.copy(),
              "status": "held", "completed_cycles": completed_cycles,
              "ocp_cost_updated": False, "endurance_improvement_validated": False}
    if type(completed_cycles) is not int or completed_cycles < 0:
        raise ValueError("completed_cycles must be a nonnegative integer.")
    if certified is not True:
        return {**result, "reason": "uncertified_source_state"}
    if completed_cycles % config.update_every_cycles:
        return {**result, "reason": "update_not_due"}
    challenge = prescribed_force_fatigue_challenge(
        current_capacity, rest_capacity, alpha_a, tau_fat, reference_force, phase_durations)
    mechanical = isokinetic_work_shapley(available_positive_power, phase_durations, required_active_work)
    if (challenge["excess_squared_fatigue"].shape != incumbent.shape
            or mechanical["mechanical_credit_j"].shape != incumbent.shape):
        raise ValueError("Inputs must all use the same muscle order and count.")
    result.update({"challenge": challenge, "mechanical": mechanical})
    if not challenge["domain_valid"]:
        return {**result, "reason": "prescribed_force_challenge_outside_domain"}
    raw = mechanical["mechanical_credit_j"] * challenge["excess_squared_fatigue"]
    result["raw_scores"] = raw
    if np.any(raw <= 0) or not np.all(np.isfinite(raw)):
        return {**result, "reason": "zero_or_invalid_physiological_score"}
    result["raw_max_normalized_scores"] = raw / raw.max()
    logs = np.log(raw)
    unprojected = logs - logs.mean()
    target = np.asarray(_project_centered_logs(
        unprojected, lower=log(config.min_relative_weight), upper=log(config.max_relative_weight)))
    result["projected_target_weights"] = np.exp(target)
    result["projection_changed_ratios"] = bool(np.max(abs(target - unprojected)) > 1e-12)
    step = config.smoothing * (target - np.log(incumbent))
    step *= min(1.0, config.max_log_step / max(float(np.max(abs(step))), 1e-300))
    if np.max(abs(step)) < config.deadband_log:
        return {**result, "reason": "log_deadband"}
    return {**result, "status": "proposed", "reason": "state_conditioned_physiological_score",
            "weights": np.exp(np.log(incumbent) + step)}
