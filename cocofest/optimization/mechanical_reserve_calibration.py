"""Numerical, outside-NLP calibration of local isokinetic reserve proxies.

The force envelope and its local linearization are opportunities, not a
certificate of simultaneous torque, pulse-width, or future-cycle feasibility.
There is no optimization or fitting to full-horizon trajectories here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .mechanical_reserve_projection import LocalMechanicalMarginModel


# A lower numerical floor, not a physiological PW tolerance.  The effective
# per-muscle replay threshold is supplied by the RHO owner as
# ``nlp_tolerance * u_scaling`` and floored at this value.
PULSE_WIDTH_NUMERICAL_BOUND_TOLERANCE_FLOOR_S = 5e-12


def _array(value, name, shape=None, *, positive=False, nonnegative=False):
    result = np.array(value, dtype=float, copy=True)
    if (not np.all(np.isfinite(result)) or (shape is not None and result.shape != shape)
            or (positive and np.any(result <= 0)) or (nonnegative and np.any(result < 0))):
        raise ValueError(f"{name} has invalid shape, finite-value or positivity domain.")
    return result


def _positive_scalar(value, name):
    result = _array(value, name, (), positive=True)
    return float(result)


@dataclass(frozen=True)
class MechanicalMarginCalibration:
    """Frozen audit of a proxy; ``margin_model`` is ready for projection.

    Legacy margins contain net-power slack per phase and one net-work slack.
    ``cycle_work`` contains only the net-work slack, normalized by
    ``power_scale_w * cycle_duration_s``.
    ``normalized_state_jacobian`` differentiates with respect to state divided
    by ``state_scales``; the model itself uses derivatives in physical units.
    """

    margin_model: LocalMechanicalMarginModel
    cycle_duration_s: float
    power_scale_w: float
    state_scales: np.ndarray
    difference_steps: np.ndarray
    normalized_state_jacobian: np.ndarray
    power_coefficients: np.ndarray
    reference_envelope_force: np.ndarray
    reference_net_power_w: np.ndarray
    margin_aggregation: str = "phase_and_work"
    calibration_policy: str = "simultaneous_pwmax_v1"
    selected_pulse_widths: np.ndarray | None = None

    def __post_init__(self):
        for name in ("state_scales", "difference_steps", "normalized_state_jacobian",
                     "power_coefficients", "reference_envelope_force", "reference_net_power_w"):
            array = _array(getattr(self, name), name)
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if self.selected_pulse_widths is not None:
            values = _array(self.selected_pulse_widths, "selected_pulse_widths", nonnegative=True)
            values.setflags(write=False)
            object.__setattr__(self, "selected_pulse_widths", values)

    @property
    def attainable_work_certified(self):
        return False

    @property
    def endurance_prediction(self):
        return False

    @property
    def calibration_kind(self):
        if self.calibration_policy == "simultaneous_pwmax_v1":
            return "local_isokinetic_force_envelope_proxy"
        return "local_isokinetic_cycle_work_gated_proxy"


@dataclass(frozen=True)
class PulseWidthForceAffineModel:
    """Local causal endpoint-force map around one bounded PW schedule.

    ``force_jacobian[m, phase, n, command]`` is the derivative of muscle
    ``m`` force at a phase endpoint with respect to PW command ``command`` of
    muscle ``n``. The prescribed-geometry Ding replay makes cross-muscle
    blocks identically zero; earlier commands of the same muscle can affect
    later force through its fast and slow states.

    This is a differentiable local approximation, not a force-feasibility
    certificate and not a replacement for the full RHO dynamics.
    """

    reference_pulse_widths: np.ndarray
    reference_forces: np.ndarray
    force_jacobian: np.ndarray

    def __post_init__(self):
        widths = _array(self.reference_pulse_widths, "reference_pulse_widths", nonnegative=True)
        forces = _array(self.reference_forces, "reference_forces", widths.shape, nonnegative=True)
        if widths.ndim != 2 or not widths.shape[0] or not widths.shape[1]:
            raise ValueError("reference_pulse_widths must be a nonempty (muscles, phases) array.")
        jacobian = _array(self.force_jacobian, "force_jacobian",
                          (*widths.shape, *widths.shape))
        widths.setflags(write=False)
        forces.setflags(write=False)
        jacobian.setflags(write=False)
        object.__setattr__(self, "reference_pulse_widths", widths)
        object.__setattr__(self, "reference_forces", forces)
        object.__setattr__(self, "force_jacobian", jacobian)

    @property
    def muscle_count(self):
        return self.reference_pulse_widths.shape[0]

    @property
    def interval_count(self):
        return self.reference_pulse_widths.shape[1]


def clip_numerical_pulse_width_bound_violations(
    pulse_widths, pulse_width_parameters, *, tolerance_s, certified_source=False,
):
    """Clip only sub-tolerance numerical PW bound overshoots and audit them.

    This function is intentionally for controls extracted from a *certified*
    NLP solution before an external Ding replay. A correction is refused
    unless the caller explicitly attests ``certified_source=True``. It is not
    a relaxation of the physical bounds: an excursion strictly larger than
    ``tolerance_s`` is rejected, rather than silently projected into the
    admissible set.

    ``tolerance_s`` is an explicit scalar or a per-muscle vector in seconds;
    it is normally derived from the NLP's dimensionless stopping tolerance and
    physical control scaling. It is solely an IPOPT/CasADi endpoint
    representation guard.
    """
    parameters = tuple(pulse_width_parameters)
    widths = _array(pulse_widths, "pulse_widths", nonnegative=True)
    if widths.ndim != 2 or widths.shape[0] != len(parameters) or not widths.shape[1]:
        raise ValueError("pulse_widths must have shape (muscles, nonempty phases).")
    tolerance = _array(tolerance_s, "tolerance_s", nonnegative=True)
    if tolerance.shape == ():
        tolerance = np.full((len(parameters), 1), float(tolerance))
    elif tolerance.shape == (len(parameters),):
        tolerance = tolerance[:, None]
    else:
        raise ValueError("tolerance_s must be a scalar or one finite nonnegative value per muscle.")
    lower = np.asarray([float(parameter.pd0) for parameter in parameters])[:, None]
    upper = np.asarray([float(parameter.pulse_width_max) for parameter in parameters])[:, None]
    if (not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper))
            or np.any(lower < 0.0) or np.any(upper <= lower)):
        raise ValueError("pulse_width_parameters have invalid physical bounds.")

    below = np.maximum(lower - widths, 0.0)
    above = np.maximum(widths - upper, 0.0)
    correction = np.maximum(below, above)
    maximum = float(np.max(correction))
    excessive = correction > tolerance
    if np.any(excessive):
        muscle, phase = np.unravel_index(np.argmax(correction - tolerance), correction.shape)
        raise ValueError(
            "Certified pulse widths exceed physical Ding bounds by "
            f"{correction[muscle, phase]:.3e} s at muscle {muscle}, phase {phase}, above the numerical "
            f"replay tolerance {tolerance[muscle, 0]:.3e} s; no clipping was applied."
        )
    if np.any(correction > 0.0) and not certified_source:
        raise ValueError("Numerical PW clipping requires a certified NLP source; no clipping was applied.")
    clipped = np.clip(widths, lower, upper)
    corrected = correction > 0.0
    audit = {
        "units": "s",
        "source_certified": bool(certified_source),
        "numerical_bound_tolerance_s_by_muscle": tolerance[:, 0].tolist(),
        "raw_min_s": float(np.min(widths)),
        "raw_max_s": float(np.max(widths)),
        "corrected": bool(np.any(corrected)),
        "corrected_entries": int(np.count_nonzero(corrected)),
        "corrected_below_entries": int(np.count_nonzero(below > 0.0)),
        "corrected_above_entries": int(np.count_nonzero(above > 0.0)),
        "max_abs_correction_s": maximum,
        "max_abs_correction_us": maximum * 1e6,
    }
    return clipped, audit


def calibrate_isokinetic_ding_pulse_width_force_map(
    *, terminal_states, intervals, pulse_width_parameters, pulse_widths,
    integration_substeps=64, mechanics_mode="isokinetic",
):
    """Replay one PW schedule and its causal tangent force map.

    The initial full Ding state is fixed at a certified RHO boundary.  Each
    muscle is replayed independently at prescribed geometry. The Jacobian is
    exact for the fixed-step RK4 map up to its integration truncation, and is
    differentiated at a *fixed* schedule—there is no embedded PW selection or
    future OCP in this calibration.
    """
    from .adaptive_moment_rollout import (
        DingPulseWidthParameters,
        _propagate_ding_pulse_width_interval_with_slow_sensitivities,
    )

    if mechanics_mode != "isokinetic":
        raise ValueError("Pulse-width force calibration requires an isokinetic configuration.")
    intervals, parameters = tuple(intervals), tuple(pulse_width_parameters)
    if not intervals or not parameters or any(not isinstance(p, DingPulseWidthParameters) for p in parameters):
        raise ValueError("Nonempty intervals and DingPulseWidthParameters are required.")
    states = _array(terminal_states, "terminal_states", (len(parameters), 5))
    if np.any(states[:, :2] < 0) or np.any(states[:, 2:] <= 0):
        raise ValueError("terminal_states require nonnegative Cn/F and strictly positive A/Tau1/Km.")
    if (isinstance(integration_substeps, (bool, np.bool_))
            or not isinstance(integration_substeps, (int, np.integer)) or integration_substeps < 1):
        raise ValueError("integration_substeps must be a positive integer.")
    widths = _array(pulse_widths, "pulse_widths", (len(parameters), len(intervals)), nonnegative=True)
    for interval in intervals:
        if any(len(getattr(interval, name)) != len(parameters)
               for name in ("calcium_amplitudes", "mechanical_gains", "moment_coefficients")):
            raise ValueError("Every interval must describe every muscle.")
    for muscle, parameter in enumerate(parameters):
        if np.any(widths[muscle] < parameter.pd0) or np.any(widths[muscle] > parameter.pulse_width_max):
            raise ValueError("pulse_widths must lie within each muscle's [pd0, pulse_width_max] bounds.")

    forces = np.empty_like(widths)
    jacobian = np.zeros((*widths.shape, *widths.shape))
    for muscle, parameter in enumerate(parameters):
        current = states[muscle].copy()
        sensitivity = np.zeros((4, len(intervals)))
        for phase, interval in enumerate(intervals):
            command_direction = np.zeros(len(intervals))
            command_direction[phase] = 1.0
            current, sensitivity = _propagate_ding_pulse_width_interval_with_slow_sensitivities(
                current, pulse_width=widths[muscle, phase], duration=interval.duration,
                calcium_amplitude=interval.calcium_amplitudes[muscle],
                mechanical_gain=interval.mechanical_gains[muscle], parameters=parameter,
                integration_substeps=integration_substeps, slow_state_sensitivities=sensitivity,
                pulse_width_sensitivity=command_direction,
            )
            forces[muscle, phase] = current[1]
            jacobian[muscle, phase, muscle] = sensitivity[0]
    return PulseWidthForceAffineModel(widths, forces, jacobian)


def calibrate_local_mechanical_margin_model(
    *, reference_states, force_envelope, moment_coefficients, phase_durations,
    angular_velocity_rad_s, required_power_w, power_scale_w,
    required_work_j=None, nonmuscle_power_w=None, state_scales=None,
    relative_step=1e-4, mechanics_mode="isokinetic", margin_aggregation="phase_and_work",
    force_envelope_jacobian=None,
):
    """Calibrate signed power/work margins using normalized central differences.

    ``reference_states`` has shape (muscles, 3), ordered A, Tau1, Km.
    ``force_envelope(states)`` must deterministically return nonnegative force
    samples of shape (muscles, phases); all other boundary states and geometry
    must stay fixed during perturbations. Static force arrays alone cannot
    identify state sensitivity and are intentionally insufficient.

    ``moment_coefficients[m,k] = b_i`` follows the reduced dynamics convention.
    At equilibrium, ``E_prod_dot = omega * (sum(b_i*F_i) - g - c*omega**2)``.
    Thus ``p_i = omega*b_i`` with its sign intact. Supply the fixed gravity and
    velocity contribution as ``nonmuscle_power_w = -omega*(g+c*omega**2)``
    when calibrating total produced power; its default is explicitly zero.
    Negative muscle contributions are never discarded or sign-flipped.

    Durations must cover exactly one physical turn, ``2*pi/abs(omega)``.
    ``required_work_j`` defaults to the duration-weighted power demand.
    The same power scale normalizes phase margins and cycle-average work.
    Differences use steps ``relative_step*state_scales`` (default reference
    states); probes that cross zero are rejected rather than clipped.
    """
    if margin_aggregation not in ("phase_and_work", "cycle_work"):
        raise ValueError("margin_aggregation must be phase_and_work or cycle_work.")
    if mechanics_mode != "isokinetic":
        raise ValueError("Calibration requires an isokinetic configuration.")
    omega = float(_array(angular_velocity_rad_s, "angular_velocity_rad_s", ()))
    if omega == 0:
        raise ValueError("angular_velocity_rad_s must be nonzero for an isokinetic cycle.")
    period = 2 * math.pi / abs(omega)
    if not math.isfinite(period):
        raise ValueError("The isokinetic cycle duration must be finite.")
    durations = _array(phase_durations, "phase_durations", positive=True)
    if durations.ndim != 1 or not durations.size:
        raise ValueError("phase_durations must be a nonempty vector.")
    if not np.isclose(np.sum(durations), period, rtol=1e-9, atol=1e-12 * period):
        raise ValueError("phase_durations must span one isokinetic cycle: 2*pi/abs(omega).")
    states = _array(reference_states, "reference_states", positive=True)
    if states.ndim != 2 or states.shape[1] != 3 or not states.shape[0]:
        raise ValueError("reference_states must have shape (muscles, 3), ordered A, Tau1, Km.")
    shape = (states.shape[0], durations.size)
    coefficients = _array(moment_coefficients, "moment_coefficients", shape)
    demand = _array(required_power_w, "required_power_w", durations.shape)
    offset = (np.zeros_like(durations) if nonmuscle_power_w is None
              else _array(nonmuscle_power_w, "nonmuscle_power_w", durations.shape))
    scale = _positive_scalar(power_scale_w, "power_scale_w")
    step = _positive_scalar(relative_step, "relative_step")
    scales = states.copy() if state_scales is None else _array(
        state_scales, "state_scales", states.shape, positive=True)
    steps = _array(step * scales, "difference_steps", states.shape, positive=True)
    if np.any(states - steps <= 0) or np.any(states + steps == states) or np.any(states - steps == states):
        raise ValueError("relative_step and state_scales must yield representable, strictly positive central probes.")
    work = float(np.dot(demand, durations)) if required_work_j is None else float(
        _array(required_work_j, "required_work_j", ()))
    if not math.isfinite(work) or not math.isfinite(scale * period):
        raise ValueError("Work demand and normalization must be finite.")
    if not callable(force_envelope):
        raise ValueError("force_envelope must be callable to identify state sensitivities.")
    power_coefficients = _array(omega * coefficients, "power_coefficients", shape)

    def evaluate(probe):
        force = _array(force_envelope(probe.copy()), "force_envelope", shape, nonnegative=True)
        power = _array(np.sum(power_coefficients * force, axis=0) + offset, "net_power_w")
        work_margin = (np.dot(power, durations) - work) / (scale * period)
        margins = (np.r_[(power - demand) / scale, work_margin]
                   if margin_aggregation == "phase_and_work" else np.array([work_margin]))
        return _array(margins, "margins"), force, power

    margins, force, power = evaluate(states)
    if force_envelope_jacobian is None:
        jacobian = np.empty((margins.size, *states.shape))
        for muscle, component in np.ndindex(states.shape):
            plus, minus = states.copy(), states.copy()
            plus[muscle, component] += steps[muscle, component]
            minus[muscle, component] -= steps[muscle, component]
            jacobian[:, muscle, component] = (evaluate(plus)[0] - evaluate(minus)[0]) / (
                plus[muscle, component] - minus[muscle, component])
    else:
        if not callable(force_envelope_jacobian):
            raise ValueError("force_envelope_jacobian must be callable when supplied.")
        force_jacobian = _array(force_envelope_jacobian(states.copy()), "force_envelope_jacobian",
                                (*shape, *states.shape))
        phase_jacobian = np.einsum("mk,mknj->kmj", power_coefficients, force_jacobian) / scale
        work_jacobian = np.einsum("kmj,k->mj", phase_jacobian, durations) / period
        jacobian = (np.concatenate((phase_jacobian, work_jacobian[None]), axis=0)
                    if margin_aggregation == "phase_and_work" else work_jacobian[None])
    model = LocalMechanicalMarginModel(states, margins, jacobian)
    return MechanicalMarginCalibration(model, period, scale, scales, steps, jacobian * scales,
                                       power_coefficients, force, power,
                                       margin_aggregation=margin_aggregation)


def calibrate_isokinetic_ding_margin_model(
    *, terminal_states, intervals, pulse_width_parameters, angular_velocity_rad_s,
    required_power_w, power_scale_w, required_work_j=None, nonmuscle_power_w=None,
    state_scales=None, relative_step=1e-4, integration_substeps=64,
    mechanics_mode="isokinetic", calibration_policy="simultaneous_pwmax_v1",
    sensitivity_mode="tangent",
):
    """Calibrate a full five-state Ding PW-max force-opportunity replay.

    ``terminal_states`` follows (Cn, F, A, Tau1, Km). Each central probe keeps
    Cn and F fixed, perturbs one initial slow state, then propagates all five
    states over the same prescribed-geometry cycle. Forces are sampled at
    phase endpoints and held constant for the mechanical work quadrature.
    Neither this endpoint quadrature nor simultaneous PW-max is a certificate
    of an attainable load; no torque bounds or coupled feasibility are solved.

    Opt-in ``cycle_work_gated_v1`` scores five bounded schedules per muscle:
    PWmax, recruitment-threshold PW, and productive-phase gating with leads
    0, Tau1 and 2*Tau1. It retains the schedule with greatest signed work and
    differentiates that frozen schedule. Only cycle-work slack is returned;
    instantaneous power deficits are allowed by this proxy. This remains a
    lower bound on the best work among unconstrained replay schedules, not
    on work attainable under the full RHO constraints. It does not solve for
    optimal stimulation, reconstruct force closure, or certify feasibility.
    """
    from dataclasses import replace
    from .adaptive_moment_rollout import (
        DingPulseWidthParameters,
        _propagate_ding_pulse_width_interval_with_slow_sensitivities,
        propagate_ding_pulse_width_interval,
    )

    if calibration_policy not in ("simultaneous_pwmax_v1", "cycle_work_gated_v1"):
        raise ValueError("Unknown mechanical margin calibration_policy.")
    if sensitivity_mode not in ("tangent", "finite_difference"):
        raise ValueError("sensitivity_mode must be tangent or finite_difference.")

    intervals, parameters = tuple(intervals), tuple(pulse_width_parameters)
    if not intervals or not parameters or any(not isinstance(p, DingPulseWidthParameters) for p in parameters):
        raise ValueError("Nonempty intervals and DingPulseWidthParameters are required.")
    states = _array(terminal_states, "terminal_states", (len(parameters), 5))
    if np.any(states[:, :2] < 0) or np.any(states[:, 2:] <= 0):
        raise ValueError("terminal_states require nonnegative Cn/F and strictly positive A/Tau1/Km.")
    if (isinstance(integration_substeps, (bool, np.bool_))
            or not isinstance(integration_substeps, (int, np.integer)) or integration_substeps < 1):
        raise ValueError("integration_substeps must be a positive integer.")
    for interval in intervals:
        if any(len(getattr(interval, name)) != len(parameters)
               for name in ("calcium_amplitudes", "mechanical_gains", "moment_coefficients")):
            raise ValueError("Every interval must describe every muscle.")
    durations = np.asarray([interval.duration for interval in intervals])
    coefficients = np.asarray([interval.moment_coefficients for interval in intervals]).T

    # Unlike independent maxima of endpoint force, each candidate is one full
    # bounded PW schedule propagated through Ding's fast and slow states. The
    # winner is chosen at the reference and held fixed while differentiating.
    # A branch switch therefore never contaminates the local Jacobian.
    selected_widths = np.asarray([[p.pulse_width_max] * len(intervals) for p in parameters])

    def replay(muscle, slow, widths):
        current = states[muscle].copy()
        current[2:] = slow
        force = np.empty(len(intervals))
        for phase, interval in enumerate(intervals):
            current = propagate_ding_pulse_width_interval(
                current, pulse_width=widths[phase], duration=interval.duration,
                calcium_amplitude=interval.calcium_amplitudes[muscle],
                mechanical_gain=interval.mechanical_gains[muscle], parameters=parameters[muscle],
                integration_substeps=integration_substeps,
            )
            force[phase] = current[1]
        return force

    if calibration_policy == "cycle_work_gated_v1":
        # Endpoint geometry/quadrature matches the legacy calibration. Leads
        # approximately compensate the force lag without another OCP or QP.
        endpoint_times = np.cumsum(durations)
        period = float(endpoint_times[-1])
        for muscle, parameter in enumerate(parameters):
            productive = angular_velocity_rad_s * coefficients[muscle] > 0
            candidates = [selected_widths[muscle].copy(),
                          np.full(len(intervals), parameter.pd0)]
            for lead in (0.0, states[muscle, 3], 2 * states[muscle, 3]):
                future_times = np.mod(endpoint_times - .5 * durations + lead, period)
                indices = np.searchsorted(endpoint_times, future_times, side="right")
                candidates.append(np.where(productive[indices], parameter.pulse_width_max, parameter.pd0))
            candidates = np.unique(np.asarray(candidates), axis=0)
            work_scores = [np.dot(angular_velocity_rad_s * coefficients[muscle]
                                 * replay(muscle, states[muscle, 2:], widths), durations)
                           for widths in candidates]
            selected_widths[muscle] = candidates[int(np.argmax(work_scores))]

    # Muscle dynamics are independent at prescribed geometry. Cache unchanged
    # rows across central probes, reducing 6*M whole-model replays to 6*M
    # single-muscle replays (important at a 20-cycle update cadence).
    cache = {}

    def envelope(slow_states):
        forces = np.empty((len(parameters), len(intervals)))
        for muscle in range(len(parameters)):
            key = (muscle, slow_states[muscle].tobytes())
            if key not in cache:
                cache[key] = replay(muscle, slow_states[muscle], selected_widths[muscle])
            forces[muscle] = cache[key]
        return forces

    def envelope_jacobian(slow_states):
        if not np.array_equal(slow_states, states[:, 2:]):
            raise ValueError("Ding tangent calibration is defined at its reference slow state only.")
        derivatives = np.zeros((len(parameters), len(intervals), len(parameters), 3))
        for muscle, parameter in enumerate(parameters):
            current = states[muscle].copy()
            sensitivity = None
            for phase, interval in enumerate(intervals):
                current, sensitivity = _propagate_ding_pulse_width_interval_with_slow_sensitivities(
                    current, pulse_width=selected_widths[muscle, phase], duration=interval.duration,
                    calcium_amplitude=interval.calcium_amplitudes[muscle],
                    mechanical_gain=interval.mechanical_gains[muscle], parameters=parameter,
                    integration_substeps=integration_substeps, slow_state_sensitivities=sensitivity,
                )
                derivatives[muscle, phase, muscle] = sensitivity[0]
        return derivatives

    result = calibrate_local_mechanical_margin_model(
        reference_states=states[:, 2:], force_envelope=envelope, moment_coefficients=coefficients,
        phase_durations=durations, angular_velocity_rad_s=angular_velocity_rad_s,
        required_power_w=required_power_w, power_scale_w=power_scale_w,
        required_work_j=required_work_j, nonmuscle_power_w=nonmuscle_power_w,
        state_scales=state_scales, relative_step=relative_step, mechanics_mode=mechanics_mode,
        margin_aggregation=("cycle_work" if calibration_policy == "cycle_work_gated_v1" else "phase_and_work"),
        force_envelope_jacobian=(envelope_jacobian if sensitivity_mode == "tangent" else None),
    )
    return replace(result, calibration_policy=calibration_policy, selected_pulse_widths=selected_widths)
