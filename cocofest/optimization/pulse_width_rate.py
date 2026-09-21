"""Pulse widths as states, with bounded physical rates as controls.

The PW state interpolates *between* stimulus commands. Ding's current pulse
still has a fixed width throughout its shooting interval: subtracting the
elapsed-time times the rate recovers the state at the stimulus. This avoids
silently replacing a train of fixed-width pulses by continuously varying PW.
"""
from __future__ import annotations

import math
import numpy as np

CONTROL_MODE = "rate_state"
CONTROL_REPRESENTATION = "pw_state_rate_control_zoh_v1"


def validate_rate_mode(mode="direct", maximum_rate=None):
    if mode not in ("direct", CONTROL_MODE):
        raise ValueError("pulse_width_control_mode must be direct or rate_state.")
    if mode == "direct":
        if maximum_rate is not None:
            raise ValueError("pulse_width_max_rate_s_per_s requires rate_state mode.")
        return None
    if isinstance(maximum_rate, bool) or maximum_rate is None:
        raise ValueError("rate_state requires a finite positive pulse_width_max_rate_s_per_s.")
    maximum_rate = float(maximum_rate)
    if not math.isfinite(maximum_rate) or maximum_rate <= 0:
        raise ValueError("pulse_width_max_rate_s_per_s must be finite and strictly positive.")
    return maximum_rate


def rate_keys(muscle):
    return f"last_pulse_width_{muscle}", f"pulse_width_rate_{muscle}"


def rate_configuration(muscles, *, states):
    from bioptim import ConfigureVariables
    return [
        lambda ocp, nlp, key=rate_keys(muscle.muscle_name)[0 if states else 1]:
            ConfigureVariables.configure_new_variable(
                key, [key], ocp, nlp, as_states=states, as_controls=not states
            )
        for muscle in muscles
    ]


def sampled_pulse_width(pw_state, rate, time, interval_start):
    """Current fixed pulse width, including the left interval's endpoint.

    ``interval_start`` must come from the integrator's node-local numerical
    data, not floor(time/dt): Radau evaluates at the right endpoint too.
    """
    current_time = time[0] if hasattr(time, "shape") and time.shape[0] > 1 else time
    return pw_state - (current_time - interval_start) * rate


def rate_rhs(model, controls, nlp):
    from bioptim import DynamicsFunctions
    from casadi import vertcat
    return vertcat(*[
        DynamicsFunctions.get(nlp.controls[rate_keys(m.muscle_name)[1]], controls)
        for m in model.muscles_dynamics_model
    ])


def promote_pulse_width_to_state(model, x_bounds, x_init, x_scaling,
                                 u_bounds, u_init, u_scaling, *, n_shooting,
                                 scale, ode_solver):
    """Move physical PW seeds/bounds into states and return rate-only controls.

    The first window's initial PW is free within the physical range. Subsequent
    windows must carry the terminal PW state using the usual state continuity.
    The last rate leads to a free terminal PW, without periodic closure.
    """
    from bioptim import BoundsList, InitialGuessList, VariableScalingList, InterpolationType
    from casadi import collocation_points
    maximum_rate = validate_rate_mode(CONTROL_MODE, model.pulse_width_max_rate_s_per_s)
    dt = float(model.pulse_width_interval_s)
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(scale) or scale <= 0:
        raise ValueError("PW state-rate formulation requires positive finite interval and scaling.")
    if getattr(model, "pulse_width_max_step_s", None) is not None:
        raise ValueError("PW state-rate formulation and auxiliary delta-PW lift are mutually exclusive.")
    new_bounds, new_init, new_scaling = BoundsList(), InitialGuessList(), VariableScalingList()
    collocation = ode_solver.is_direct_collocation
    if collocation:
        fractions = [0.] + list(collocation_points(ode_solver.polynomial_degree, ode_solver.method))
        grid = np.concatenate([i + np.asarray(fractions) for i in range(n_shooting)] + [np.array([n_shooting])])
    else:
        grid = np.arange(n_shooting + 1)
    expected = {rate_keys(m.muscle_name)[0] for m in model.muscles_dynamics_model}
    if set(u_bounds.keys()) != expected:
        raise ValueError("PW state-rate formulation currently supports pulse-width-only controls.")
    for muscle in model.muscles_dynamics_model:
        pw_key, rate_key = rate_keys(muscle.muscle_name)
        lower, upper = np.asarray(u_bounds[pw_key].min), np.asarray(u_bounds[pw_key].max)
        if not np.all(lower == lower.flat[0]) or not np.all(upper == upper.flat[0]):
            raise ValueError("PW state-rate formulation requires stage-independent physical PW bounds.")
        physical = np.asarray(u_init[pw_key].init, dtype=float).reshape(-1)
        if physical.size == 1:
            physical = np.full(n_shooting, physical[0])
        if physical.size != n_shooting:
            raise ValueError(f"{pw_key}: expected {n_shooting} physical PW seed samples.")
        endpoints = np.r_[physical, physical[-1]]
        x_bounds.add(pw_key, min_bound=[float(lower.flat[0])], max_bound=[float(upper.flat[0])])
        x_init.add(pw_key, initial_guess=np.interp(grid, np.arange(n_shooting + 1), endpoints)[None, :],
                   interpolation=InterpolationType.ALL_POINTS if collocation else InterpolationType.EACH_FRAME)
        x_scaling.add(pw_key, scaling=[scale])
        new_bounds.add(rate_key, min_bound=[-maximum_rate], max_bound=[maximum_rate])
        new_init.add(rate_key, initial_guess=(np.diff(endpoints) / dt)[None, :],
                     interpolation=InterpolationType.EACH_FRAME)
        new_scaling.add(rate_key, scaling=[scale / dt])
    return new_bounds, new_init, new_scaling


def physical_pulse_controls(states, controls, *, shooting_stride=1):
    """Expose executed stimulus commands to archive/replay code as PW traces.

    ``states`` must include the terminal state; the terminal PW belongs to the
    next interval and is deliberately omitted. Rate controls remain available.
    """
    result = dict(controls)
    for key, values in states.items():
        if key.startswith("last_pulse_width_"):
            result[key] = np.asarray(values)[:, ::shooting_stride][:, :-1]
    return result
