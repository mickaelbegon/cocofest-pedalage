"""NumPy reproduction of the article's physiological fixed-weight calculation.

Source: ``physiological_weight_calculation.py``, public commit
``31e064f4d80741c8b5444e6ce4d91bfe4eb78c3d``. Geometry is supplied by the
caller as signed torque profiles, with explicit parameters for each case.
The published one-second active/rest fatigue experiment is reproduced, not a
closed-loop endurance prediction. Integrals use the supplied sorted angular
samples without appending a periodic closing segment, exactly as the source.
"""

from collections.abc import Mapping

import numpy as np


SOURCE_COMMIT = "31e064f4d80741c8b5444e6ce4d91bfe4eb78c3d"
SUPPORT_THRESHOLD = 1e-9


def _finite(value, name, *, positive=False):
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0.):
        raise ValueError(f"{name} must be finite" + (" and positive." if positive else "."))
    return value


def _cycle_count(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("target_cycles must be a positive integer.")
    value = _finite(value, "target_cycles", positive=True)
    if int(value) != value:
        raise ValueError("target_cycles must be a positive integer.")
    return int(value)


def _parameter_values(parameters, muscle_names):
    if not isinstance(parameters, Mapping) or set(parameters) != set(muscle_names):
        raise ValueError("parameters must contain exactly the named muscles for this case.")
    output = {}
    for name in muscle_names:
        if not isinstance(parameters[name], Mapping):
            raise ValueError("Each muscle must have an explicit parameter mapping.")
        values = {}
        for key in ("Fmax", "a_scale", "alpha_a", "tau_fat"):
            if key not in parameters[name]:
                raise ValueError(f"Missing parameter {name}.{key}.")
            values[key] = _finite(parameters[name][key], f"{name}.{key}", positive=key != "alpha_a")
        if values["alpha_a"] > 0.:
            raise ValueError("alpha_a must be nonpositive for Ding fatigue.")
        output[name] = values
    return output


def simulate_fatigue_ratios(duty_cycles, parameters, *, muscle_names, target_cycles=1500, rho=.8):
    """Return active and post-rest capacity ratios, ordered (muscles,cycles).

    Each cycle lasts one second: activation for ``duty`` seconds at constant
    force ``rho*Fmax``, then recovery for ``1-duty`` seconds. The geometric
    series composes the exact scalar Ding flow using expm1 for short durations.
    Both endpoints are retained: a positive post-rest ratio cannot hide an
    unphysical nonpositive capacity reached during activation.
    """
    names = tuple(muscle_names)
    if not names or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("muscle_names must be unique nonempty strings.")
    params = _parameter_values(parameters, names)
    cycles = _cycle_count(target_cycles)
    rho = _finite(rho, "rho")
    if not 0. <= rho <= 1.:
        raise ValueError("rho must lie in [0,1].")
    duty = np.asarray(duty_cycles, dtype=float)
    if duty.shape != (len(names),) or not np.all(np.isfinite(duty)) or np.any((duty < 0.) | (duty > 1.)):
        raise ValueError("duty_cycles must have one finite entry in [0,1] per muscle.")
    capacity = np.array([params[n]["a_scale"] for n in names])[:, None]
    tau = np.array([params[n]["tau_fat"] for n in names])[:, None]
    beta = np.array([params[n]["alpha_a"] * rho * params[n]["Fmax"]
                     * params[n]["tau_fat"] / params[n]["a_scale"] for n in names])[:, None]
    active_duration = duty[:, None]
    active_decay = np.exp(-active_duration / tau)
    rest_decay = np.exp(-(1. - active_duration) / tau)
    active_input = beta * -np.expm1(-active_duration / tau)
    per_cycle_input = rest_decay * active_input
    index = np.arange(1, cycles + 1, dtype=float)[None, :]
    geometric_sum = np.expm1(-index / tau) / np.expm1(-1. / tau)
    ratios = 1. + per_cycle_input * geometric_sum
    previous = np.concatenate((np.ones((len(names), 1)), ratios[:, :-1]), axis=1)
    active_ratios = 1. + active_decay * (previous - 1.) + active_input
    invalid_active = ~np.isfinite(active_ratios) | (active_ratios <= 0.)
    invalid_rest = ~np.isfinite(ratios) | (ratios <= 0.)
    invalid = invalid_active | invalid_rest
    first_invalid = None
    if np.any(invalid):
        cycle, muscle = np.argwhere(invalid.T)[0]
        first_invalid = {
            "cycle_index": int(cycle), "muscle": names[muscle],
            "phase": "active" if invalid_active[muscle, cycle] else "rest",
            "active_ratio": float(active_ratios[muscle, cycle]),
            "rest_ratio": float(ratios[muscle, cycle]),
        }
    return {
        "ratios": ratios, "active_ratios": active_ratios,
        "active_capacities": active_ratios * capacity, "rest_capacities": ratios * capacity,
        "domain_valid": not bool(np.any(invalid)), "invalid_active": invalid_active,
        "invalid_rest": invalid_rest, "first_invalid": first_invalid,
    }


def build_pre_risk_mask(theta_deg, risk_mask, *, pre_risk_width_deg=90.):
    """Mask up to the declared angle before circular risk onsets, excluding risk.

    A cycle with risk at every angular sample has no onset and therefore no
    pre-risk samples, matching the published implementation.
    """
    theta = np.asarray(theta_deg, dtype=float)
    risk = np.asarray(risk_mask, dtype=bool)
    if theta.ndim != 1 or risk.shape[-1:] != theta.shape or not np.all(np.isfinite(theta)):
        raise ValueError("theta_deg must match the final risk-mask dimension.")
    width = _finite(pre_risk_width_deg, "pre_risk_width_deg")
    if not 0. <= width <= 360.:
        raise ValueError("pre_risk_width_deg must lie in [0,360].")
    onsets = risk & ~np.roll(risk, 1, axis=-1)
    # Rows index onset angles, columns index preceding sample angles.
    distance = np.mod(theta[:, None] - theta[None, :], 360.)
    preceding = (distance > 0.) & (distance <= width)
    return (onsets @ preceding) & ~risk


def calculate_physiological_muscle_weights(
    theta, torque_profiles, parameters, *, muscle_names, case_id,
    target_cycles=1500, rho=.8, pre_risk_width_deg=90., task_torque_threshold=.20,
    normalization="max",
):
    """Reproduce source factors and expose controller eligibility separately.

    ``theta`` is sorted radians; profiles have shape (muscles,angles). All
    parameter values are explicit per case; no global muscle table is used.
    ``fatigability=(ratio_cycle1-ratio_cycleN)/N``. Mechanical contribution
    averages pre-risk support plus unique support over every fatigue cycle.
    Raw weights equal mechanical contribution times squared fatigability.

    By default, weights are divided by their largest raw weight. This retains
    a strictly positive value for every muscle whose raw contribution is
    strictly positive; only a physically zero raw score stays zero.
    ``normalized_weights`` is None if fatigue leaves its physical domain or
    the raw weights are degenerate. ``legacy_normalized_weights`` always
    retains the source calculation for audit (never controller authorization).
    ``usable_for_controller`` means the calibration is admissible; it does
    not certify closed-loop RHO feasibility, endurance, or improvement.
    ``minmax`` remains available only to reproduce the published display and
    is stored in ``legacy_normalized_weights`` for audit.
    Arrays include all intermediate per-cycle factors and masks. ``context``
    is JSON-compatible and records exact per-case inputs and source semantics.
    """
    names = tuple(muscle_names)
    if not names or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("muscle_names must be unique nonempty strings.")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("case_id must identify the explicit parameter case.")
    params = _parameter_values(parameters, names)
    cycles = _cycle_count(target_cycles)
    angles = np.asarray(theta, dtype=float)
    profiles = np.asarray(torque_profiles, dtype=float)
    if (angles.ndim != 1 or angles.size < 2 or not np.all(np.isfinite(angles))
            or np.any(np.diff(angles) < 0.) or angles[0] < 0. or angles[-1] > 2 * np.pi
            or angles[-1] <= angles[0]):
        raise ValueError("theta must be sorted finite radians in [0,2*pi], with positive span.")
    if profiles.shape != (len(names), len(angles)) or not np.all(np.isfinite(profiles)):
        raise ValueError("torque_profiles must be finite with shape (muscles,angles).")
    threshold = _finite(task_torque_threshold, "task_torque_threshold")
    if threshold < 0.:
        raise ValueError("task_torque_threshold must be nonnegative.")
    if normalization not in ("minmax", "max"):
        raise ValueError("normalization must be 'max' or the legacy 'minmax' variant.")
    support = profiles > SUPPORT_THRESHOLD
    duty = np.mean(support, axis=1)
    fatigue = simulate_fatigue_ratios(duty, params, muscle_names=names, target_cycles=cycles, rho=rho)
    ratios = fatigue["ratios"]
    fatigability = (ratios[:, 0] - ratios[:, -1]) / cycles
    # No capacity clipping: on invalid trajectories this is a legacy audit,
    # including source sign flips caused by negative capacity ratios.
    positive = np.maximum(profiles[None, :, :] * ratios.T[:, :, None], 0.)
    redundancy = np.sum(positive > SUPPORT_THRESHOLD, axis=1)
    risk = np.sum(positive, axis=1) < threshold
    pre_risk = build_pre_risk_mask(np.degrees(angles), risk, pre_risk_width_deg=pre_risk_width_deg)
    # Explicit trapezoid weights match np.trapezoid without periodic closure.
    quadrature = np.r_[np.diff(angles) / 2, 0.] + np.r_[0., np.diff(angles) / 2]
    pre_support = np.sum(positive * pre_risk[:, None, :] * quadrature, axis=2)
    unique_support = np.sum(positive * (redundancy == 1)[:, None, :] * quadrature, axis=2)
    mean_pre, mean_unique = np.mean(pre_support, axis=0), np.mean(unique_support, axis=0)
    mechanical = mean_pre + mean_unique
    raw = mechanical * fatigability**2
    minimum, maximum = float(np.min(raw)), float(np.max(raw))
    legacy = np.zeros_like(raw) if maximum == minimum else (raw - minimum) / (maximum - minimum)
    max_normalized = np.zeros_like(raw) if maximum == 0. else raw / maximum
    normalized = max_normalized if normalization == "max" else legacy
    degenerate = not np.all(np.isfinite(normalized)) or not np.any(normalized > 0.)
    valid = fatigue["domain_valid"]
    usable = valid and not degenerate

    def keyed(values):
        return {name: float(value) for name, value in zip(names, values, strict=True)}

    return {
        "status": "invalid_fatigue_domain" if not valid else "degenerate_weights" if degenerate else "ok",
        "domain_valid": valid, "usable_for_controller": usable,
        "theta": angles.copy(), "torque_profiles": profiles.copy(),
        "positive_torque": np.maximum(profiles, 0.), "support_mask": support,
        "duty_cycles": keyed(duty), "fatigue_dynamics": {name: ratios[i].copy() for i, name in enumerate(names)},
        **fatigue,
        "fatigability": keyed(fatigability), "support_in_pre_risk": keyed(mean_pre),
        "unique_support": keyed(mean_unique), "mechanical_contribution": keyed(mechanical),
        "support_in_pre_risk_by_cycle": pre_support, "unique_support_by_cycle": unique_support,
        "positive_profiles_by_cycle": positive, "redundancy_count": redundancy,
        "risk_mask": risk, "pre_risk_mask": pre_risk,
        "raw_weights": keyed(raw), "min_raw_weight": minimum,
        "legacy_normalized_weights": keyed(legacy),
        "normalized_weights": keyed(normalized) if usable else None,
        "rho": float(rho), "target_cycles": cycles, "pre_risk_width_deg": float(pre_risk_width_deg),
        "context": {
            "case_id": case_id, "muscle_names": list(names), "parameters": params,
            "source_commit": SOURCE_COMMIT, "normalization": normalization,
            "published_normalization": normalization == "minmax", "cycle_duration_seconds": 1.,
            "rho": float(rho), "target_cycles": cycles, "task_torque_threshold": threshold,
            "support_threshold": SUPPORT_THRESHOLD, "pre_risk_width_deg": float(pre_risk_width_deg),
            "quadrature": "trapezoid_on_supplied_angles_without_periodic_closure",
            "fatigue_slope": "(post_rest_cycle_1-post_rest_cycle_N)/N",
            "closed_loop_endurance_certified": False,
        },
    }
