"""Opt-in odd PW interpolation through the existing exact increment lift.

With 50 zero-order-held physical controls, constrain indices 1, 3, ..., 47
to the mean of their even neighbours. Index 49 and the terminal carrier are
not coupled to index 0. These equalities restrict the admissible pulse trains;
they do not eliminate decision variables or reduce the shooting mesh.
"""

import numpy as np

from .pulse_width_slew import auxiliary_keys, validate_max_step

REPRESENTATION = "odd_pw_adjacent_increment_equality_v1"


def validate_odd_interpolation(enabled, *, max_step_s, mechanical_formulation,
                               cycles_per_window, stimulations_per_cycle, solver=None):
    if not isinstance(enabled, bool):
        raise ValueError("pulse_width_odd_interpolation must be a boolean.")
    if not enabled:
        return False
    if (validate_max_step(max_step_s) is None or mechanical_formulation != "reduced"
            or cycles_per_window != 1 or stimulations_per_cycle != 50):
        raise ValueError("PW odd interpolation requires a hard ΔPW bound, reduced mechanics, "
                         "one-cycle windows and exactly 50 stimulations per cycle.")
    if solver is not None and str(solver).lower() not in ("ipopt", "madnlp"):
        raise ValueError("PW odd interpolation currently requires IPOPT or MadNLP; "
                         "ACADOS does not export these cross-stage constraints.")
    return True


def adjacent_increment_equality(controller, *, muscle_names, scale):
    # Bioptim's ordinary NLP penalty binds cx_end to the next shooting
    # control even for CONSTANT controls. Do not use derivative=True: that
    # path deliberately substitutes cx_start for cx_end for CONSTANT PW.
    from casadi import vertcat
    return vertcat(*[
        (controller.controls[auxiliary_keys(name)[1]].cx_start
         - controller.controls[auxiliary_keys(name)[1]].cx_end) / scale
        for name in muscle_names
    ])


def odd_interpolation_constraints(muscles, *, scale, n_shooting=50, constraints=None):
    """delta[2j]=delta[2j+1] implies 2*PW[2j+1]=PW[2j]+PW[2j+2]."""
    from bioptim import ConstraintList
    if n_shooting != 50 or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Odd PW interpolation requires 50 shooting intervals and a positive finite scale.")
    if constraints is None:
        constraints = ConstraintList()
    names = tuple(muscle.muscle_name for muscle in muscles)
    for even in range(0, 48, 2):
        constraints.add(adjacent_increment_equality, node=even,
                        muscle_names=names, scale=scale,
                        min_bound=0, max_bound=0)
    return constraints


def attach_odd_interpolation_audit(summary, args):
    enabled = getattr(args, "pulse_width_odd_interpolation", False)
    muscles = {}
    if enabled:
        for key, raw in (summary.get("control_traces") or {}).items():
            if not key.startswith("last_pulse_width_"):
                continue
            values = np.asarray(raw, dtype=float).reshape(-1)
            if not values.size or values.size % 50 or not np.isfinite(values).all():
                continue
            cycles = values.reshape(-1, 50)
            residual = (cycles[:, 1:48:2] - .5 * (cycles[:, 0:47:2] + cycles[:, 2:49:2])) * 1e6
            muscles[key.removeprefix("last_pulse_width_")] = {
                "maximum_abs_residual_us": float(np.max(np.abs(residual))),
                "rms_residual_us": float(np.sqrt(np.mean(residual ** 2))),
            }
    summary["pulse_width_odd_interpolation_audit"] = {
        "enabled": enabled, "available": bool(muscles), "muscles": muscles,
        "representation": REPRESENTATION if enabled else None,
        "constrained_control_indices_zero_based": list(range(1, 48, 2)) if enabled else [],
        "last_control_index_49_free": True,
        "cycle_wrap_constrained": False,
        "trace_scope": "all_exported_attempts",
        "decision_variables_eliminated": False,
    }
