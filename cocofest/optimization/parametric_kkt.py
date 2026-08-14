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


@dataclass(frozen=True)
class ActiveKktRows:
    jacobian: np.ndarray
    sources: tuple[tuple[str, int, str], ...]


@dataclass(frozen=True)
class CanonicalNlpKktBlocks:
    """Numerical derivatives of the exact CasADi NLP submitted to a solver."""

    objective_gradient: np.ndarray
    lagrangian_hessian: object
    constraint_values: np.ndarray
    constraint_jacobian: object


@dataclass(frozen=True)
class SparseKktSensitivityPrediction:
    primal_step: np.ndarray
    dual_step: np.ndarray
    raw_primal_step_inf_norm: float
    applied_step_scale: float
    linear_residual_inf_norm: float
    used_least_squares: bool


class CanonicalNlpKktEvaluator:
    """Cache CasADi derivative graphs and return SciPy sparse numerical blocks.

    ``nlp`` must be the canonical ``{"x", "f", "g"}`` dictionary retained by
    the Bioptim IPOPT/MadNLP interface.  Building derivatives from this graph,
    rather than reconstructing the OCP, guarantees the same scaling and
    decision-vector ordering as the solver call.
    """

    def __init__(self, nlp: dict):
        from casadi import Function, dot, gradient, hessian, jacobian

        if not isinstance(nlp, dict) or not {"x", "f", "g"}.issubset(nlp):
            raise ValueError("A canonical NLP must define x, f, and g.")
        x = nlp["x"]
        f = nlp["f"]
        g = nlp["g"]
        self.variable_count = int(x.numel())
        self.constraint_count = int(g.numel())
        multiplier_type = type(x)
        lam_g = multiplier_type.sym("cocofest_kkt_lam_g", self.constraint_count)
        lagrangian = f + dot(lam_g, g)
        lagrangian_hessian, _ = hessian(lagrangian, x)
        self._function = Function(
            "cocofest_canonical_kkt_blocks",
            [x, lam_g],
            [
                gradient(f, x),
                lagrangian_hessian,
                g,
                jacobian(g, x),
            ],
            ["x", "lam_g"],
            ["grad_f", "hess_l", "g", "jac_g"],
        )
        self._constraint_function = Function(
            "cocofest_canonical_kkt_constraints",
            [x],
            [g],
            ["x"],
            ["g"],
        )

    @staticmethod
    def _as_sparse(matrix):
        from scipy.sparse import csc_matrix

        sparsity = matrix.sparsity()
        rows, columns = sparsity.get_triplet()
        values = np.asarray(matrix.nonzeros(), dtype=float)
        return csc_matrix(
            (values, (np.asarray(rows), np.asarray(columns))),
            shape=(sparsity.size1(), sparsity.size2()),
        )

    def evaluate(
        self, decision_vector: np.ndarray, constraint_multipliers: np.ndarray
    ) -> CanonicalNlpKktBlocks:
        z = np.asarray(decision_vector, dtype=float).reshape(-1)
        lam_g = np.asarray(constraint_multipliers, dtype=float).reshape(-1)
        if z.size != self.variable_count:
            raise ValueError("The decision vector has an incompatible dimension.")
        if lam_g.size != self.constraint_count:
            raise ValueError(
                "The constraint multipliers have an incompatible dimension."
            )
        if not np.all(np.isfinite(z)) or not np.all(np.isfinite(lam_g)):
            raise ValueError("Canonical KKT evaluation inputs must be finite.")
        values = self._function(x=z, lam_g=lam_g)
        return CanonicalNlpKktBlocks(
            objective_gradient=np.asarray(values["grad_f"], dtype=float).reshape(-1),
            lagrangian_hessian=self._as_sparse(values["hess_l"]),
            constraint_values=np.asarray(values["g"], dtype=float).reshape(-1),
            constraint_jacobian=self._as_sparse(values["jac_g"]),
        )

    def evaluate_constraints(self, decision_vector: np.ndarray) -> np.ndarray:
        z = np.asarray(decision_vector, dtype=float).reshape(-1)
        if z.size != self.variable_count or not np.all(np.isfinite(z)):
            raise ValueError("The decision vector is invalid for this canonical NLP.")
        return np.asarray(self._constraint_function(x=z)["g"], dtype=float).reshape(-1)


def active_kkt_rhs_from_bound_changes(
    sources: tuple[tuple[str, int, str], ...],
    old_variable_lower_bounds: np.ndarray,
    old_variable_upper_bounds: np.ndarray,
    new_variable_lower_bounds: np.ndarray,
    new_variable_upper_bounds: np.ndarray,
    old_constraint_lower_bounds: np.ndarray,
    old_constraint_upper_bounds: np.ndarray,
    new_constraint_lower_bounds: np.ndarray,
    new_constraint_upper_bounds: np.ndarray,
    *,
    equality_tolerance: float = 1e-12,
) -> np.ndarray:
    """Return the active-row target displacement for moving NLP bounds.

    For an active row written ``a(z) - b = 0``, linearization gives
    ``J dz = db``.  This routine produces that ``db`` in the same order as
    :func:`assemble_active_kkt_rows`.  It therefore avoids introducing fake
    symbolic parameters when a receding-horizon datum already enters CasADi
    through ``lbx/ubx/lbg/ubg``.
    """

    arrays = {
        "old_lbx": np.asarray(old_variable_lower_bounds, dtype=float).reshape(-1),
        "old_ubx": np.asarray(old_variable_upper_bounds, dtype=float).reshape(-1),
        "new_lbx": np.asarray(new_variable_lower_bounds, dtype=float).reshape(-1),
        "new_ubx": np.asarray(new_variable_upper_bounds, dtype=float).reshape(-1),
        "old_lbg": np.asarray(old_constraint_lower_bounds, dtype=float).reshape(-1),
        "old_ubg": np.asarray(old_constraint_upper_bounds, dtype=float).reshape(-1),
        "new_lbg": np.asarray(new_constraint_lower_bounds, dtype=float).reshape(-1),
        "new_ubg": np.asarray(new_constraint_upper_bounds, dtype=float).reshape(-1),
    }
    if (
        arrays["old_lbx"].shape != arrays["old_ubx"].shape
        or arrays["new_lbx"].shape != arrays["old_lbx"].shape
        or arrays["new_ubx"].shape != arrays["old_lbx"].shape
    ):
        raise ValueError("Old and new variable bounds must have identical shapes.")
    if (
        arrays["old_lbg"].shape != arrays["old_ubg"].shape
        or arrays["new_lbg"].shape != arrays["old_lbg"].shape
        or arrays["new_ubg"].shape != arrays["old_lbg"].shape
    ):
        raise ValueError("Old and new constraint bounds must have identical shapes.")
    if not np.isfinite(equality_tolerance) or equality_tolerance < 0.0:
        raise ValueError("The equality tolerance must be finite and non-negative.")

    target_steps = []
    for kind, index, side in sources:
        if kind == "variable":
            lower_key, upper_key = "lbx", "ubx"
        elif kind == "constraint":
            lower_key, upper_key = "lbg", "ubg"
        else:
            raise ValueError(f"Unknown active-row kind '{kind}'.")
        if index < 0 or index >= arrays[f"old_{lower_key}"].size:
            raise IndexError(
                f"Active-row index {index} is out of range for {kind} bounds."
            )
        if side == "lower":
            target_step = (
                arrays[f"new_{lower_key}"][index] - arrays[f"old_{lower_key}"][index]
            )
        elif side == "upper":
            target_step = (
                arrays[f"new_{upper_key}"][index] - arrays[f"old_{upper_key}"][index]
            )
        elif side == "equality":
            lower_step = (
                arrays[f"new_{lower_key}"][index] - arrays[f"old_{lower_key}"][index]
            )
            upper_step = (
                arrays[f"new_{upper_key}"][index] - arrays[f"old_{upper_key}"][index]
            )
            if not np.isfinite(lower_step) or not np.isfinite(upper_step):
                raise ValueError("A moving equality bound must remain finite.")
            if abs(lower_step - upper_step) > equality_tolerance:
                raise ValueError(
                    "Lower and upper displacements of a moving equality disagree."
                )
            target_step = 0.5 * (lower_step + upper_step)
        else:
            raise ValueError(f"Unknown active-row side '{side}'.")
        if not np.isfinite(target_step):
            raise ValueError("An active bound target displacement is not finite.")
        target_steps.append(float(target_step))
    return np.asarray(target_steps, dtype=float)


def active_kkt_rhs_to_bound_targets(
    sources: tuple[tuple[str, int, str], ...],
    variable_values: np.ndarray,
    constraint_values: np.ndarray,
    variable_lower_bounds: np.ndarray,
    variable_upper_bounds: np.ndarray,
    constraint_lower_bounds: np.ndarray,
    constraint_upper_bounds: np.ndarray,
) -> np.ndarray:
    """Return ``b - a(z)`` for each active row at a candidate primal."""

    z = np.asarray(variable_values, dtype=float).reshape(-1)
    g = np.asarray(constraint_values, dtype=float).reshape(-1)
    lbx = np.asarray(variable_lower_bounds, dtype=float).reshape(-1)
    ubx = np.asarray(variable_upper_bounds, dtype=float).reshape(-1)
    lbg = np.asarray(constraint_lower_bounds, dtype=float).reshape(-1)
    ubg = np.asarray(constraint_upper_bounds, dtype=float).reshape(-1)
    if z.shape != lbx.shape or z.shape != ubx.shape:
        raise ValueError("Variable values and bounds must have identical shapes.")
    if g.shape != lbg.shape or g.shape != ubg.shape:
        raise ValueError("Constraint values and bounds must have identical shapes.")
    residuals = []
    for kind, index, side in sources:
        if kind == "variable":
            values, lower, upper = z, lbx, ubx
        elif kind == "constraint":
            values, lower, upper = g, lbg, ubg
        else:
            raise ValueError(f"Unknown active-row kind '{kind}'.")
        if index < 0 or index >= values.size:
            raise IndexError(
                f"Active-row index {index} is out of range for {kind} values."
            )
        if side == "lower":
            target = lower[index]
        elif side == "upper":
            target = upper[index]
        elif side == "equality":
            if not np.isfinite(lower[index]) or not np.isfinite(upper[index]):
                raise ValueError("An equality target must remain finite.")
            target = 0.5 * (lower[index] + upper[index])
        else:
            raise ValueError(f"Unknown active-row side '{side}'.")
        residual = target - values[index]
        if not np.isfinite(residual):
            raise ValueError("An active target residual is not finite.")
        residuals.append(float(residual))
    return np.asarray(residuals, dtype=float)


def assemble_active_kkt_rows(
    variable_values: np.ndarray,
    constraint_values: np.ndarray,
    constraint_jacobian: np.ndarray,
    variable_lower_bounds: np.ndarray,
    variable_upper_bounds: np.ndarray,
    constraint_lower_bounds: np.ndarray,
    constraint_upper_bounds: np.ndarray,
    *,
    variable_multipliers: np.ndarray | None = None,
    constraint_multipliers: np.ndarray | None = None,
    primal_activity_tolerance: float = 1e-7,
    dual_activity_tolerance: float = 1e-8,
    equality_tolerance: float = 1e-12,
) -> ActiveKktRows:
    """Stack equality, active inequality, and active variable-bound rows.

    Activity always requires primal proximity to a bound. IPOPT/CasADi
    multiplier signs are only used to reject the opposite side: negative for
    a lower bound and positive for an upper bound. A multiplier alone must
    never activate a distant bound, which would violate complementarity.
    Equal lower and upper bounds are represented once.
    """

    z = np.asarray(variable_values, dtype=float).reshape(-1)
    g = np.asarray(constraint_values, dtype=float).reshape(-1)
    jacobian = np.asarray(constraint_jacobian, dtype=float)
    lbx = np.asarray(variable_lower_bounds, dtype=float).reshape(-1)
    ubx = np.asarray(variable_upper_bounds, dtype=float).reshape(-1)
    lbg = np.asarray(constraint_lower_bounds, dtype=float).reshape(-1)
    ubg = np.asarray(constraint_upper_bounds, dtype=float).reshape(-1)
    if jacobian.shape != (g.size, z.size):
        raise ValueError("The nonlinear-constraint Jacobian has an invalid shape.")
    if lbx.shape != z.shape or ubx.shape != z.shape:
        raise ValueError("Variable bounds must match the decision vector.")
    if lbg.shape != g.shape or ubg.shape != g.shape:
        raise ValueError("Constraint bounds must match the constraint vector.")
    for value, name in (
        (primal_activity_tolerance, "primal activity tolerance"),
        (dual_activity_tolerance, "dual activity tolerance"),
        (equality_tolerance, "equality tolerance"),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"The {name} must be finite and non-negative.")
    lam_x = (
        np.zeros(z.size)
        if variable_multipliers is None
        else np.asarray(variable_multipliers, dtype=float).reshape(-1)
    )
    lam_g = (
        np.zeros(g.size)
        if constraint_multipliers is None
        else np.asarray(constraint_multipliers, dtype=float).reshape(-1)
    )
    if lam_x.shape != z.shape or lam_g.shape != g.shape:
        raise ValueError("Multiplier vectors have incompatible dimensions.")

    rows = []
    sources = []
    for index in range(g.size):
        finite_lower = np.isfinite(lbg[index])
        finite_upper = np.isfinite(ubg[index])
        equality = (
            finite_lower
            and finite_upper
            and abs(ubg[index] - lbg[index]) <= equality_tolerance
        )
        if equality:
            rows.append(jacobian[index])
            sources.append(("constraint", index, "equality"))
            continue
        lower_active = (
            finite_lower
            and g[index] - lbg[index] <= primal_activity_tolerance
            and lam_g[index] <= dual_activity_tolerance
        )
        upper_active = (
            finite_upper
            and ubg[index] - g[index] <= primal_activity_tolerance
            and lam_g[index] >= -dual_activity_tolerance
        )
        if lower_active:
            rows.append(jacobian[index])
            sources.append(("constraint", index, "lower"))
        if upper_active:
            rows.append(jacobian[index])
            sources.append(("constraint", index, "upper"))

    identity = np.eye(z.size)
    for index in range(z.size):
        finite_lower = np.isfinite(lbx[index])
        finite_upper = np.isfinite(ubx[index])
        equality = (
            finite_lower
            and finite_upper
            and abs(ubx[index] - lbx[index]) <= equality_tolerance
        )
        if equality:
            rows.append(identity[index])
            sources.append(("variable", index, "equality"))
            continue
        lower_active = (
            finite_lower
            and z[index] - lbx[index] <= primal_activity_tolerance
            and lam_x[index] <= dual_activity_tolerance
        )
        upper_active = (
            finite_upper
            and ubx[index] - z[index] <= primal_activity_tolerance
            and lam_x[index] >= -dual_activity_tolerance
        )
        if lower_active:
            rows.append(identity[index])
            sources.append(("variable", index, "lower"))
        if upper_active:
            rows.append(identity[index])
            sources.append(("variable", index, "upper"))

    return ActiveKktRows(
        jacobian=(np.vstack(rows) if rows else np.empty((0, z.size), dtype=float)),
        sources=tuple(sources),
    )


def assemble_sparse_active_kkt_rows(
    variable_values: np.ndarray,
    constraint_values: np.ndarray,
    constraint_jacobian,
    variable_lower_bounds: np.ndarray,
    variable_upper_bounds: np.ndarray,
    constraint_lower_bounds: np.ndarray,
    constraint_upper_bounds: np.ndarray,
    *,
    variable_multipliers: np.ndarray | None = None,
    constraint_multipliers: np.ndarray | None = None,
    primal_activity_tolerance: float = 1e-7,
    dual_activity_tolerance: float = 1e-8,
    equality_tolerance: float = 1e-12,
) -> ActiveKktRows:
    """Sparse counterpart of :func:`assemble_active_kkt_rows`.

    The implementation deliberately avoids constructing a dense identity
    matrix. This matters for direct-collocation RHO problems, where fixed
    initial states alone can add dozens of active variable-bound rows.
    """

    from scipy.sparse import csc_matrix, eye, vstack

    z = np.asarray(variable_values, dtype=float).reshape(-1)
    g = np.asarray(constraint_values, dtype=float).reshape(-1)
    jacobian = csc_matrix(constraint_jacobian)
    lbx = np.asarray(variable_lower_bounds, dtype=float).reshape(-1)
    ubx = np.asarray(variable_upper_bounds, dtype=float).reshape(-1)
    lbg = np.asarray(constraint_lower_bounds, dtype=float).reshape(-1)
    ubg = np.asarray(constraint_upper_bounds, dtype=float).reshape(-1)
    if jacobian.shape != (g.size, z.size):
        raise ValueError("The nonlinear-constraint Jacobian has an invalid shape.")
    if lbx.shape != z.shape or ubx.shape != z.shape:
        raise ValueError("Variable bounds must match the decision vector.")
    if lbg.shape != g.shape or ubg.shape != g.shape:
        raise ValueError("Constraint bounds must match the constraint vector.")
    for value, name in (
        (primal_activity_tolerance, "primal activity tolerance"),
        (dual_activity_tolerance, "dual activity tolerance"),
        (equality_tolerance, "equality tolerance"),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"The {name} must be finite and non-negative.")
    lam_x = (
        np.zeros(z.size)
        if variable_multipliers is None
        else np.asarray(variable_multipliers, dtype=float).reshape(-1)
    )
    lam_g = (
        np.zeros(g.size)
        if constraint_multipliers is None
        else np.asarray(constraint_multipliers, dtype=float).reshape(-1)
    )
    if lam_x.shape != z.shape or lam_g.shape != g.shape:
        raise ValueError("Multiplier vectors have incompatible dimensions.")

    constraint_indices: list[int] = []
    variable_indices: list[int] = []
    sources: list[tuple[str, int, str]] = []
    for index in range(g.size):
        finite_lower = np.isfinite(lbg[index])
        finite_upper = np.isfinite(ubg[index])
        equality = (
            finite_lower
            and finite_upper
            and abs(ubg[index] - lbg[index]) <= equality_tolerance
        )
        if equality:
            constraint_indices.append(index)
            sources.append(("constraint", index, "equality"))
            continue
        if (
            finite_lower
            and g[index] - lbg[index] <= primal_activity_tolerance
            and lam_g[index] <= dual_activity_tolerance
        ):
            constraint_indices.append(index)
            sources.append(("constraint", index, "lower"))
        if (
            finite_upper
            and ubg[index] - g[index] <= primal_activity_tolerance
            and lam_g[index] >= -dual_activity_tolerance
        ):
            constraint_indices.append(index)
            sources.append(("constraint", index, "upper"))

    # A fixed initial state may be represented both by lbx == ubx and by an
    # explicit start constraint. Retaining both produces two collinear KKT
    # rows and a singular saddle-point matrix. Any selected constraint row
    # with one nonzero coefficient already fixes that same local direction.
    singleton_constraint_columns: set[int] = set()
    if constraint_indices:
        selected_constraints = jacobian[constraint_indices, :].tocsr()
        for row_index in range(selected_constraints.shape[0]):
            start = selected_constraints.indptr[row_index]
            stop = selected_constraints.indptr[row_index + 1]
            if stop - start == 1:
                singleton_constraint_columns.add(
                    int(selected_constraints.indices[start])
                )

    for index in range(z.size):
        finite_lower = np.isfinite(lbx[index])
        finite_upper = np.isfinite(ubx[index])
        equality = (
            finite_lower
            and finite_upper
            and abs(ubx[index] - lbx[index]) <= equality_tolerance
        )
        if equality:
            if index in singleton_constraint_columns:
                continue
            variable_indices.append(index)
            sources.append(("variable", index, "equality"))
            continue
        if (
            finite_lower
            and z[index] - lbx[index] <= primal_activity_tolerance
            and lam_x[index] <= dual_activity_tolerance
        ):
            variable_indices.append(index)
            sources.append(("variable", index, "lower"))
        if (
            finite_upper
            and ubx[index] - z[index] <= primal_activity_tolerance
            and lam_x[index] >= -dual_activity_tolerance
        ):
            variable_indices.append(index)
            sources.append(("variable", index, "upper"))

    rows = []
    if constraint_indices:
        rows.append(jacobian[constraint_indices, :])
    if variable_indices:
        rows.append(eye(z.size, format="csc")[variable_indices, :])
    active_jacobian = (
        vstack(rows, format="csc") if rows else csc_matrix((0, z.size), dtype=float)
    )
    return ActiveKktRows(jacobian=active_jacobian, sources=tuple(sources))


def solve_sparse_bound_kkt_sensitivity(
    hessian,
    active_constraint_jacobian,
    active_target_step: np.ndarray,
    *,
    primal_regularization: float = 0.0,
    maximum_primal_step_inf_norm: float | None = None,
    least_squares_tolerance: float = 1e-10,
) -> SparseKktSensitivityPrediction:
    """Solve ``[H J'; J 0] [dz, dl] = [0, db]`` sparsely."""

    from scipy.sparse import bmat, csc_matrix, eye
    from scipy.sparse.linalg import MatrixRankWarning, lsmr, spsolve
    import warnings

    hessian = csc_matrix(hessian)
    jacobian = csc_matrix(active_constraint_jacobian)
    target_step = np.asarray(active_target_step, dtype=float).reshape(-1)
    if hessian.shape[0] != hessian.shape[1]:
        raise ValueError("The Lagrangian Hessian must be square.")
    if jacobian.shape[1] != hessian.shape[0]:
        raise ValueError("The active-constraint Jacobian has an invalid shape.")
    if target_step.size != jacobian.shape[0]:
        raise ValueError("The active target step has an invalid dimension.")
    if not np.isfinite(primal_regularization) or primal_regularization < 0.0:
        raise ValueError("Primal KKT regularization must be finite and non-negative.")
    if maximum_primal_step_inf_norm is not None and (
        not np.isfinite(maximum_primal_step_inf_norm)
        or maximum_primal_step_inf_norm <= 0.0
    ):
        raise ValueError("The maximum primal KKT step must be finite and positive.")

    symmetric_hessian = 0.5 * (hessian + hessian.T)
    if primal_regularization:
        symmetric_hessian = symmetric_hessian + primal_regularization * eye(
            hessian.shape[0], format="csc"
        )
    kkt_matrix = bmat(
        [
            [symmetric_hessian, jacobian.T],
            [jacobian, csc_matrix((jacobian.shape[0], jacobian.shape[0]))],
        ],
        format="csc",
    )
    right_hand_side = np.concatenate((np.zeros(hessian.shape[0]), target_step))
    used_least_squares = False
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        try:
            raw_step = np.asarray(spsolve(kkt_matrix, right_hand_side)).reshape(-1)
            if not np.all(np.isfinite(raw_step)):
                raise np.linalg.LinAlgError(
                    "Sparse KKT solve returned non-finite values."
                )
        except (MatrixRankWarning, np.linalg.LinAlgError, RuntimeError, ValueError):
            raw_step = lsmr(
                kkt_matrix,
                right_hand_side,
                atol=least_squares_tolerance,
                btol=least_squares_tolerance,
            )[0]
            used_least_squares = True

    raw_primal = raw_step[: hessian.shape[0]]
    raw_dual = raw_step[hessian.shape[0] :]
    raw_norm = float(np.linalg.norm(raw_primal, ord=np.inf))
    step_scale = 1.0
    if (
        maximum_primal_step_inf_norm is not None
        and raw_norm > maximum_primal_step_inf_norm
    ):
        step_scale = maximum_primal_step_inf_norm / raw_norm
    applied_step = step_scale * raw_step
    residual = kkt_matrix @ applied_step - step_scale * right_hand_side
    return SparseKktSensitivityPrediction(
        primal_step=applied_step[: hessian.shape[0]],
        dual_step=applied_step[hessian.shape[0] :],
        raw_primal_step_inf_norm=raw_norm,
        applied_step_scale=float(step_scale),
        linear_residual_inf_norm=float(np.linalg.norm(residual, ord=np.inf)),
        used_least_squares=used_least_squares,
    )


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
