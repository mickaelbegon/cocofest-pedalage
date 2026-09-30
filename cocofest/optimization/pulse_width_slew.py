"""Exact constraints on adjacent discrete pulse-width commands.

Physical PW controls remain zero-order held. The established ``lifting``
formulation introduces a carrier z and an increment control delta_pw, bounded
directly to +/-step. Its auxiliary ODE makes z[k+1]=u[k]+delta_pw[k]=u[k+1]
for both collocation and IRK. The free final increment never imposes an
artificial periodic PW closure.

The ``direct_constraints`` formulation instead constrains the difference of
the two successive physical controls directly. It has the same intra-window
admissible PW sequences as the lifting, while retaining only physical controls
and states. It uses the ordinary Bioptim penalty endpoint representation rather
than a multinode object so it remains compatible with Ding numerical time
series. Both forms enforce the executed RHO seam separately.

The current binding supports one-cycle RHO windows. The actual boundary to
the previous executed cycle is enforced separately through the first carrier
or first physical-control bound, respectively.
"""

from __future__ import annotations

import math
import numpy as np

PREFIX = "pw_slew_"
CONTROL_REPRESENTATION = "delta_pw_v1"
REGULARIZATION_NORMALIZATION = "mean_squared_intra_window_delta_v1"
LIFTING_FORMULATION = "lifting"
DIRECT_CONSTRAINTS_FORMULATION = "direct_constraints"


def validate_slew_formulation(value: str | None) -> str:
    """Return a supported transcription of the discrete PW-change bound.

    ``lifting`` is the established ACADOS-compatible formulation.  The
    direct form has precisely the same intra-window admissible physical PW
    sequences, but expresses their differences as sparse IPOPT/MadNLP
    constraints instead of introducing carrier states and delta controls.
    """
    formulation = LIFTING_FORMULATION if value is None else str(value).strip().lower()
    if formulation not in (LIFTING_FORMULATION, DIRECT_CONSTRAINTS_FORMULATION):
        raise ValueError(
            "pulse_width_slew_formulation must be 'lifting' or 'direct_constraints'."
        )
    return formulation


def validate_slew_regularization(weight=0.0, reference_us=100.0, *, max_step_s=None):
    """Validate independently of Bioptim, with an explicit hard-bound requirement."""
    if isinstance(weight, bool) or isinstance(reference_us, bool):
        raise ValueError("PW slew weight and reference must be finite real numbers, not booleans.")
    weight, reference_us = float(weight), float(reference_us)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("pulse_width_slew_weight must be finite and non-negative.")
    if not math.isfinite(reference_us) or reference_us <= 0:
        raise ValueError("pulse_width_slew_reference_us must be finite and strictly positive.")
    if weight > 0 and validate_max_step(max_step_s) is None:
        raise ValueError("Positive pulse_width_slew_weight requires pulse_width_max_step_s (a hard ΔPW bound).")
    return weight, reference_us


def validate_max_step(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("pulse_width_max_step_s must be finite and strictly positive.")
    return value


def add_pulse_width_slew_cli(parser) -> None:
    parser.add_argument(
        "--pulse-width-slew-formulation",
        choices=(LIFTING_FORMULATION, DIRECT_CONSTRAINTS_FORMULATION),
        default=LIFTING_FORMULATION,
        help=(
            "Transcription of --pulse-width-max-step-us. 'lifting' preserves "
            "the historical carrier/delta-PW NLP; 'direct_constraints' uses "
            "sparse adjacent physical-PW inequalities (IPOPT/MadNLP only)."
        ),
    )
    parser.add_argument(
        "--pulse-width-odd-interpolation", action="store_true",
        help="Experimental IPOPT/MadNLP constraint PW[1,3,...,47] = mean of adjacent even controls; "
             "PW[49] stays free. Requires 50 stimulations, reduced one-cycle windows and a hard ΔPW bound.",
    )
    parser.add_argument(
        "--pulse-width-max-step-us", type=float, default=None,
        help=(
            "Hard PW change bound between successive controls, including the "
            "cycle boundary, in microseconds (e.g. 100); requires reduced "
            "one-cycle windows."
        ),
    )
    parser.add_argument(
        "--pulse-width-slew-weight", type=float, default=0.0,
        help="Weight of mean squared normalized intra-window ΔPW; zero disables the cost. Requires a hard ΔPW bound.",
    )
    parser.add_argument(
        "--pulse-width-slew-reference-us", type=float, default=100.0,
        help="Positive ΔPW normalization reference in microseconds; independent of the hard bound (default: 100).",
    )


def pulse_width_slew_signature(args) -> dict:
    maximum_us = getattr(args, "pulse_width_max_step_us", None)
    weight = getattr(args, "pulse_width_slew_weight", 0.0)
    signature = {
        "pulse_width_odd_interpolation": getattr(args, "pulse_width_odd_interpolation", False),
        "pulse_width_max_step_us": maximum_us,
        "pulse_width_slew_weight": weight,
        "pulse_width_slew_reference_us": getattr(args, "pulse_width_slew_reference_us", 100.0),
        "pulse_width_slew_formulation": validate_slew_formulation(
            getattr(args, "pulse_width_slew_formulation", LIFTING_FORMULATION)
        ),
    }
    if weight > 0:
        signature["pulse_width_slew_normalization"] = REGULARIZATION_NORMALIZATION
    if maximum_us is not None:
        signature["pulse_width_slew_control_representation"] = (
            "physical_pw_v1"
            if signature["pulse_width_slew_formulation"] == DIRECT_CONSTRAINTS_FORMULATION
            else CONTROL_REPRESENTATION
        )
    return signature


def auxiliary_keys(muscle: str) -> tuple[str, str]:
    return f"{PREFIX}carrier_{muscle}", f"{PREFIX}delta_pw_{muscle}"


def auxiliary_configuration(muscles, *, states: bool):
    from bioptim import ConfigureVariables
    functions = []
    for muscle in muscles:
        carrier, increment = auxiliary_keys(muscle.muscle_name)
        for key in ((carrier,) if states else (increment,)):
            functions.append(lambda ocp, nlp, key=key: ConfigureVariables.configure_new_variable(
                key, [key], ocp, nlp, as_states=states, as_controls=not states
            ))
    return functions


def auxiliary_rhs(model, states, controls, nlp):
    from bioptim import DynamicsFunctions
    from casadi import vertcat
    values = []
    for muscle in model.muscles_dynamics_model:
        _, increment = auxiliary_keys(muscle.muscle_name)
        delta_pw = DynamicsFunctions.get(nlp.controls[increment], controls)
        values.append(delta_pw / model.pulse_width_interval_s)
    return vertcat(*values)


def slew_carrier_matches_control(controller, muscle: str, scale: float):
    carrier, _ = auxiliary_keys(muscle)
    return (controller.states[carrier].cx - controller.controls[f"last_pulse_width_{muscle}"].cx) / scale


def normalized_slew_residual(controller, reference_s: float):
    """Physical, stage-local increments; CONSTANT-control derivatives would be zero."""
    from casadi import vertcat
    return vertcat(*[
        controller.controls[auxiliary_keys(muscle.muscle_name)[1]].cx / reference_s
        for muscle in controller.model.muscles_dynamics_model
    ])


def normalized_direct_slew_residual(controller, reference_s: float):
    """Adjacent ZOH command differences without an auxiliary state or control.

    In the supported Bioptim version ``MINIMIZE_CONTROL, derivative=True``
    evaluates both ends with ``cx_start`` for CONSTANT controls, giving zero.
    Explicit endpoints express the intended discrete derivative while keeping
    Ding's physically constant command over each stimulation interval.
    """
    from casadi import vertcat
    return vertcat(*[
        adjacent_pulse_width_difference(
            controller, muscle=muscle.muscle_name, scale=reference_s,
        )
        for muscle in controller.model.muscles_dynamics_model
    ])


def add_slew_regularization(objectives, model, *, weight=0.0, reference_us=100.0,
                            n_shooting: int, interval_s: float):
    """Add λ mean((ΔPW/reference)²) with no extra variables in direct mode.

    Lagrange integration supplies dt. In lifting mode the last increment has no
    physical successor and is driven to zero by this cost; its squared value
    remains part of the solver cost away from the optimum. The real executed
    RHO seam is bounded separately and is not penalized here.
    """
    weight, reference_us = validate_slew_regularization(
        weight, reference_us, max_step_s=getattr(model, "pulse_width_max_step_s", None)
    )
    if weight == 0:
        return
    if n_shooting < 2 or not math.isfinite(interval_s) or interval_s <= 0:
        raise ValueError("PW slew regularization requires at least two shooting intervals and positive interval_s.")
    from bioptim import Node, ObjectiveFcn
    muscle_count = len(model.muscles_dynamics_model)
    if not muscle_count:
        raise ValueError("PW slew regularization requires at least one muscle.")
    formulation = validate_slew_formulation(getattr(model, "pulse_width_slew_formulation", None))
    if formulation == DIRECT_CONSTRAINTS_FORMULATION:
        # Only N-1 physical pairs exist; there is no terminal command and no
        # periodic wraparound penalty. The executed RHO seam is bounded alone.
        # Mayer penalties allow explicit nodes and have no dt multiplier.
        for node in range(n_shooting - 1):
            objectives.add(
                normalized_direct_slew_residual, custom_type=ObjectiveFcn.Mayer,
                node=node, quadratic=True, multi_thread=False,
                reference_s=reference_us * 1e-6,
                weight=weight / (muscle_count * (n_shooting - 1)),
            )
        return
    objectives.add(
        normalized_slew_residual, custom_type=ObjectiveFcn.Lagrange,
        node=Node.ALL_SHOOTING, quadratic=True, multi_thread=False,
        reference_s=reference_us * 1e-6,
        weight=weight / (muscle_count * (n_shooting - 1) * interval_s),
    )


def add_slew_constraints(constraints, muscles, *, max_step_s: float, scale: float):
    from bioptim import Node
    validate_max_step(max_step_s)
    for model in muscles:
        kwargs = {"muscle": model.muscle_name, "scale": scale}
        constraints.add(slew_carrier_matches_control, node=Node.ALL_SHOOTING, min_bound=0, max_bound=0, **kwargs)


def adjacent_pulse_width_difference(controller, *, muscle: str, scale: float):
    """Return ``(PW[k+1] - PW[k]) / scale`` for one shooting interval.

    The controls are exactly the values zero-order-held by Ding on the two
    neighbouring stimulation intervals.  No artificial state or continuous
    PW dynamics is involved.
    """
    # For CONSTANT controls Bioptim exposes the next shooting command through
    # ``cx_end``. This is the same mechanism used by the validated odd-PW
    # interpolation constraint, and unlike a MultinodeConstraint it remains
    # compatible with the periodic Ding numerical time series.
    key = f"last_pulse_width_{muscle}"
    return (controller.controls[key].cx_end - controller.controls[key].cx_start) / scale


def add_direct_slew_constraints(constraints, muscles, *, max_step_s: float,
                                 scale: float, n_shooting: int):
    """Bound all intra-window adjacent physical PW commands directly.

    A separate bound on the first control is updated between RHO windows by
    :func:`advance_direct_slew_bounds`, so this function intentionally only
    creates pairs that are both in the current NLP window.
    """
    validate_max_step(max_step_s)
    if not isinstance(n_shooting, int) or n_shooting < 2:
        raise ValueError("Direct PW slew constraints require at least two shooting intervals.")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Direct PW slew constraints require a positive finite PW scale.")
    bound = max_step_s / scale
    for muscle_model in muscles:
        for node in range(n_shooting - 1):
            constraints.add(
                adjacent_pulse_width_difference,
                node=node,
                min_bound=-bound,
                max_bound=bound,
                muscle=muscle_model.muscle_name,
                scale=scale,
            )


def add_auxiliary_bounds_and_guesses(model, x_bounds, x_init, x_scaling, u_bounds, u_init, u_scaling,
                                    *, n_shooting: int, scale: float, ode_solver=None):
    """Initialize a free auxiliary lift with finite, physically scaled bounds."""
    from bioptim import InterpolationType
    step = validate_max_step(model.pulse_width_max_step_s)
    if step is None:
        raise ValueError("Auxiliary PW increment bounds require pulse_width_max_step_s.")
    collocation = ode_solver is not None and ode_solver.is_direct_collocation
    state_stride = int(ode_solver.polynomial_degree) + 1 if collocation else 1
    state_count = n_shooting * state_stride + 1
    for muscle in model.muscles_dynamics_model:
        key = f"last_pulse_width_{muscle.muscle_name}"
        lower = float(np.min(u_bounds[key].min))
        upper = float(np.max(u_bounds[key].max))
        seed = float(np.mean(u_init[key].init))
        carrier, increment = auxiliary_keys(muscle.muscle_name)
        x_bounds.add(carrier, min_bound=[lower], max_bound=[upper])
        # A scalar CONSTANT guess cannot receive a nonconstant certified seed
        # or a transferred carrier trajectory. Match the physical state grid.
        x_init.add(carrier, initial_guess=np.full((1, state_count), seed),
                   interpolation=InterpolationType.ALL_POINTS if collocation else InterpolationType.EACH_FRAME)
        x_scaling.add(carrier, scaling=[scale])
        u_bounds.add(increment, min_bound=[-step], max_bound=[step])
        u_init.add(increment, initial_guess=np.zeros((1, n_shooting)), interpolation=InterpolationType.EACH_FRAME)
        u_scaling.add(increment, scaling=[scale])


def advance_auxiliary_bounds(nmpc, solution) -> None:
    """Keep the lift free and enforce the real last-executed→first-new seam."""
    phases = getattr(nmpc, "nlp", None)
    if not phases:
        return
    from bioptim import SolutionMerge
    model = getattr(phases[0], "model", None)
    if getattr(model, "pulse_width_slew_formulation", LIFTING_FORMULATION) != LIFTING_FORMULATION:
        return
    step = getattr(model, "pulse_width_max_step_s", None)
    if step is None:
        return
    controls = solution.decision_controls(to_merge=SolutionMerge.NODES)
    last = int(nmpc.time_idx_to_cycle) - 1
    for muscle in model.muscles_dynamics_model:
        carrier, _ = auxiliary_keys(muscle.muscle_name)
        previous = float(controls[f"last_pulse_width_{muscle.muscle_name}"][0, last])
        bounds = nmpc.nlp[0].x_bounds[carrier]
        bounds.min[:, 0] = np.maximum(bounds.min[:, 1], previous - step)
        bounds.max[:, 0] = np.minimum(bounds.max[:, 1], previous + step)


def advance_direct_slew_bounds(nmpc, solution) -> None:
    """Apply the last-executed → first-new PW bound for the direct NLP.

    The physical control at the first interval remains a normal direct PW
    variable; only its native bounds at node zero are narrowed for the next
    RHO solve.  The interior bounds are the immutable physiological bounds,
    hence successive calls do not accumulate an obsolete seam restriction.
    """
    phases = getattr(nmpc, "nlp", None)
    if not phases:
        return
    from bioptim import SolutionMerge
    model = getattr(phases[0], "model", None)
    if (
        getattr(model, "pulse_width_slew_formulation", LIFTING_FORMULATION)
        != DIRECT_CONSTRAINTS_FORMULATION
    ):
        return
    step = getattr(model, "pulse_width_max_step_s", None)
    if step is None:
        return
    controls = solution.decision_controls(to_merge=SolutionMerge.NODES)
    last = int(nmpc.time_idx_to_cycle) - 1
    for muscle in model.muscles_dynamics_model:
        key = f"last_pulse_width_{muscle.muscle_name}"
        previous = float(controls[key][0, last])
        bounds = nmpc.nlp[0].u_bounds[key]
        reference_column = 1 if bounds.min.shape[1] > 1 else 0
        native_lower = bounds.min[:, reference_column]
        native_upper = bounds.max[:, reference_column]
        bounds.min[:, 0] = np.maximum(native_lower, previous - step)
        bounds.max[:, 0] = np.minimum(native_upper, previous + step)


def circular_pw_differences(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim < 1 or values.shape[-1] < 2 or not np.isfinite(values).all():
        raise ValueError("PW values must have at least two finite samples on the last axis.")
    return np.roll(values, -1, axis=-1) - values


def _certified_cycle_prefix(summary: dict) -> int | None:
    """Resolve certification without treating exported/attempted cycles as executed."""
    for key in ("physically_validated_cycles", "validated_cycles"):
        if summary.get(key) is not None:
            return max(0, int(summary[key]))
    statuses = summary.get("window_statuses")
    if statuses is not None:
        feasibility = summary.get("window_feasibility") or []
        prefix = 0
        for index, status in enumerate(statuses):
            if status != 0 or index >= len(feasibility) or not feasibility[index].get("passes_tolerance", False):
                break
            prefix += 1
        # A fully certified multi-cycle horizon also exports its final tail.
        if prefix and prefix == len(statuses) and summary.get("solver_success"):
            return max(prefix, int(summary.get("covered_cycles") or prefix))
        return prefix
    return None


def attach_slew_audit(summary: dict, args) -> None:
    """Audit exported PW traces, distinguishing attempted and certified seams."""
    from .pulse_width_interpolation import attach_odd_interpolation_audit
    attach_odd_interpolation_audit(summary, args)
    maximum_us = getattr(args, "pulse_width_max_step_us", None)
    weight = getattr(args, "pulse_width_slew_weight", 0.0)
    reference_us = getattr(args, "pulse_width_slew_reference_us", 100.0)
    formulation = validate_slew_formulation(getattr(args, "pulse_width_slew_formulation", None))
    has_lift = maximum_us is not None and formulation == LIFTING_FORMULATION
    count = int(args.stimulations_per_cycle)
    certified_cycles = _certified_cycle_prefix(summary)
    muscles = {}
    for key, raw in (summary.get("control_traces") or {}).items():
        if not key.startswith("last_pulse_width_"):
            continue
        values = np.asarray(raw, dtype=float).reshape(-1)
        if count < 2 or values.size < count or values.size % count or not np.isfinite(values).all():
            continue
        cycles = values.reshape(-1, count)
        within_horizon = np.abs(np.diff(cycles, axis=1)) * 1e6
        periodic_wrap = np.abs(cycles[:, 0] - cycles[:, -1]) * 1e6
        attempted_seams = np.abs(cycles[1:, 0] - cycles[:-1, -1]) * 1e6
        executed_seams = attempted_seams[:max(0, certified_cycles - 1)] if certified_cycles is not None else np.array([])
        observed = max(float(within_horizon.max()), float(attempted_seams.max()) if attempted_seams.size else 0.0)
        muscles[key.removeprefix("last_pulse_width_")] = {
            "maximum_adjacent_change_us": observed,
            "maximum_intra_cycle_change_us": float(within_horizon.max()),
            "rms_intra_cycle_change_us": float(np.sqrt(np.mean(within_horizon ** 2))),
            "mean_squared_normalized_intra_cycle_change": float(np.mean((within_horizon / reference_us) ** 2)),
            "maximum_periodic_wrap_change_us_unconstrained": float(periodic_wrap.max()),
            "maximum_attempted_cycle_seam_change_us": float(attempted_seams.max()) if attempted_seams.size else None,
            "maximum_executed_cycle_seam_change_us": float(executed_seams.max()) if executed_seams.size else None,
            "minimum_margin_us": maximum_us - observed if maximum_us is not None else None,
            "maximum_bound_violation_us": max(0., observed - maximum_us) if maximum_us is not None else None,
        }
    summary["pulse_width_slew_audit"] = {
        "enabled": maximum_us is not None, "limit_us": maximum_us,
        "available": bool(muscles), "muscles": muscles,
        "trace_scope": "all_exported_attempts",
        "executed_seam_scope": "certified_cycle_prefix",
        "certified_cycle_count": certified_cycles,
        "control_representation": "physical PW held constant per shooting interval",
        "formulation": formulation,
        "auxiliary_control_representation": CONTROL_REPRESENTATION if has_lift else None,
        "bounded_pair": "successive_controls",
        "circular_within_cycle": False,
        "actual_rho_seam_bounded": maximum_us is not None,
    }
    normalized_mean = (
        float(np.mean([item["mean_squared_normalized_intra_cycle_change"] for item in muscles.values()]))
        if muscles else None
    )
    summary["pulse_width_slew_regularization_audit"] = {
        "enabled": weight > 0, "weight": weight, "reference_us": reference_us,
        "normalization": REGULARIZATION_NORMALIZATION,
        "scope": "intra_window_successive_controls",
        "actual_rho_seam_penalized": False, "circular_within_cycle": False,
        "mean_squared_normalized_intra_window_change": normalized_mean,
        "mean_weighted_intra_window_cost": weight * normalized_mean if normalized_mean is not None else None,
        "final_auxiliary_increment": (
            "penalized_toward_zero" if weight > 0 else "free_within_uniform_bounds"
        ) if has_lift else None,
        "reported_cost_excludes_final_auxiliary_increment": has_lift,
    }
