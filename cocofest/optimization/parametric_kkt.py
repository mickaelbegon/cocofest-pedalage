"""Small, solver-independent building blocks for advanced-step NMPC.

The routines in this module deliberately operate on numerical KKT blocks.
They do not alter an OCP and therefore can be audited against a certified
solution before being connected to a solver-specific warm-start path.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KktSensitivityPrediction:
    primal_step: np.ndarray
    dual_step: np.ndarray
    raw_primal_step_inf_norm: float
    applied_step_scale: float
    linear_residual_inf_norm: float
    condition_number: float
    used_least_squares: bool


def solve_parametric_kkt_sensitivity(
    hessian: np.ndarray,
    active_constraint_jacobian: np.ndarray,
    stationarity_parameter_jacobian: np.ndarray,
    constraint_parameter_jacobian: np.ndarray,
    parameter_step: np.ndarray,
    *,
    primal_regularization: float = 0.0,
    maximum_primal_step_inf_norm: float | None = None,
    least_squares_rcond: float | None = None,
) -> KktSensitivityPrediction:
    """Solve the fixed-active-set derivative of the nonlinear KKT system.

    The convention is ``c(z, p) = 0``. Consequently the right-hand side is
    ``-[d stationarity / dp; dc / dp] @ parameter_step``.
    """

    hessian = np.asarray(hessian, dtype=float)
    jacobian = np.asarray(active_constraint_jacobian, dtype=float)
    stationarity_p = np.asarray(stationarity_parameter_jacobian, dtype=float)
    constraints_p = np.asarray(constraint_parameter_jacobian, dtype=float)
    parameter_step = np.asarray(parameter_step, dtype=float).reshape(-1)

    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("The Lagrangian Hessian must be square.")
    variable_count = hessian.shape[0]
    if jacobian.ndim != 2 or jacobian.shape[1] != variable_count:
        raise ValueError("The active-constraint Jacobian has an invalid shape.")
    active_count = jacobian.shape[0]
    parameter_count = parameter_step.size
    if stationarity_p.shape != (variable_count, parameter_count):
        raise ValueError("The stationarity parameter Jacobian has an invalid shape.")
    if constraints_p.shape != (active_count, parameter_count):
        raise ValueError("The constraint parameter Jacobian has an invalid shape.")
    if not all(
        np.all(np.isfinite(values))
        for values in (
            hessian,
            jacobian,
            stationarity_p,
            constraints_p,
            parameter_step,
        )
    ):
        raise ValueError("KKT sensitivity inputs must be finite.")
    if not np.isfinite(primal_regularization) or primal_regularization < 0.0:
        raise ValueError("Primal KKT regularization must be finite and non-negative.")
    if maximum_primal_step_inf_norm is not None and (
        not np.isfinite(maximum_primal_step_inf_norm)
        or maximum_primal_step_inf_norm <= 0.0
    ):
        raise ValueError("The maximum primal KKT step must be finite and positive.")

    regularized_hessian = 0.5 * (hessian + hessian.T)
    if primal_regularization:
        regularized_hessian = regularized_hessian + (
            primal_regularization * np.eye(variable_count)
        )
    kkt_matrix = np.block(
        [
            [regularized_hessian, jacobian.T],
            [jacobian, np.zeros((active_count, active_count))],
        ]
    )
    parameter_jacobian = np.vstack((stationarity_p, constraints_p))
    right_hand_side = -(parameter_jacobian @ parameter_step)

    used_least_squares = False
    try:
        raw_step = np.linalg.solve(kkt_matrix, right_hand_side)
    except np.linalg.LinAlgError:
        raw_step = np.linalg.lstsq(
            kkt_matrix, right_hand_side, rcond=least_squares_rcond
        )[0]
        used_least_squares = True

    raw_primal = raw_step[:variable_count]
    raw_dual = raw_step[variable_count:]
    raw_norm = float(np.linalg.norm(raw_primal, ord=np.inf))
    step_scale = 1.0
    if (
        maximum_primal_step_inf_norm is not None
        and raw_norm > maximum_primal_step_inf_norm
    ):
        step_scale = maximum_primal_step_inf_norm / raw_norm
    primal_step = step_scale * raw_primal
    dual_step = step_scale * raw_dual
    applied_step = np.concatenate((primal_step, dual_step))
    applied_right_hand_side = step_scale * right_hand_side
    residual = kkt_matrix @ applied_step - applied_right_hand_side

    return KktSensitivityPrediction(
        primal_step=primal_step,
        dual_step=dual_step,
        raw_primal_step_inf_norm=raw_norm,
        applied_step_scale=float(step_scale),
        linear_residual_inf_norm=float(np.linalg.norm(residual, ord=np.inf)),
        condition_number=float(np.linalg.cond(kkt_matrix)),
        used_least_squares=used_least_squares,
    )


def kkt_prediction_passes_residual_guard(
    reference_residual: float,
    predicted_residual: float,
    *,
    maximum_ratio: float = 1.0,
    absolute_tolerance: float = 0.0,
) -> bool:
    """Accept a predictor only when it does not worsen the audited residual."""

    values = (reference_residual, predicted_residual, maximum_ratio, absolute_tolerance)
    if not all(np.isfinite(value) for value in values):
        return False
    if reference_residual < 0 or predicted_residual < 0:
        raise ValueError("KKT residual norms must be non-negative.")
    if maximum_ratio < 0 or absolute_tolerance < 0:
        raise ValueError("KKT residual guard tolerances must be non-negative.")
    threshold = max(absolute_tolerance, maximum_ratio * reference_residual)
    return predicted_residual <= threshold
