"""Short numerical muscle-allocation preview, outside the RHO NLP.

Slow states follow an internal nominal seed only while constructing the QP.
Known calcium and frozen slow states make F_next = a F + b recruitment affine;
residual forces couple phases. Only the first control is applied through the
original compact phase map, then the preview is rebuilt. A projected nominal
target is never treated as achieved physical tracking. No tracking band exists.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.linalg import qr
from scipy.optimize import lsq_linear, minimize

from .adaptive_moment_rollout import TotalMomentReferenceRolloutResult
from .smooth_muscle_moment_allocation import SmoothMomentAllocationOptions, solve_bounded_moment_qp_reference


def _physical(states):
    return (np.all(np.isfinite(states)) and np.all(states[:, :2] >= 0)
            and np.all(states[:, 2:] > 0))


@dataclass(frozen=True)
class PreviewAllocationProblem:
    depth: int
    initial_guess: np.ndarray
    upper_bounds: np.ndarray
    force_offsets: np.ndarray
    force_jacobians: np.ndarray
    equality_matrix: np.ndarray
    equality_target: np.ndarray
    reference_moments: np.ndarray
    moment_offsets: np.ndarray
    moment_jacobian: np.ndarray
    normalized_weights: np.ndarray
    hessian: np.ndarray
    linear_term: np.ndarray
    objective_constant: float
    nominal_projected_steps: int
    nominal_original_target_errors_nm: np.ndarray

    def objective(self, recruitment):
        x = np.asarray(recruitment)
        return float(.5 * x @ self.hessian @ x + self.linear_term @ x + self.objective_constant)


@dataclass(frozen=True)
class PreviewAllocationPlan:
    accepted: bool
    recruitment: np.ndarray | None
    problem: PreviewAllocationProblem | None
    diagnostics: dict


@dataclass(frozen=True)
class PreviewMomentRolloutResult(TotalMomentReferenceRolloutResult):
    preview_phases: int
    qp_solves: int
    qp_iterations: int
    qp_failures: int
    greedy_fallback_steps: int
    seed_projected_steps: int
    diagnostics: tuple[dict, ...]
    total_lower_bounds: np.ndarray
    total_upper_bounds: np.ndarray
    original_total_moments: np.ndarray
    signed_moment_errors: np.ndarray


class PreviewMuscleAllocation:
    """Small bounded convex QPs solved by SLSQP with analytic derivatives.

    The objective is the sum of squared per-muscle moment deviations weighted
    by the scalar greedy allocator's nominal flexibility. One common positive
    normalization bounds its Hessian scale; a fixed recruitment regularizer
    resolves invisible muscles. H1 bypasses this objective and uses the exact
    original greedy policy. Every QP is audited for full equality residual,
    recruitment bounds, and KKT stationarity with nonnegative active-bound
    multipliers. Optimizer success alone is never enough.
    """

    def __init__(self, predictor, *, preview_phases=2, moment_tolerance=1e-8,
                 max_iterations=60, fallback_to_greedy=False, optimality_tolerance=1e-6,
                 recruitment_regularization=1e-8, max_solve_time_s=.5):
        if isinstance(preview_phases, bool) or preview_phases not in (1, 2, 3):
            raise ValueError("preview_phases must be 1, 2 or 3.")
        if isinstance(max_iterations, bool) or int(max_iterations) != max_iterations or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer.")
        for name, value in (("moment_tolerance", moment_tolerance), ("optimality_tolerance", optimality_tolerance),
                            ("recruitment_regularization", recruitment_regularization), ("max_solve_time_s", max_solve_time_s)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if predictor.muscle_count * preview_phases > 12:
            raise ValueError("Preview is limited to at most 12 recruitment variables.")
        self.predictor = predictor
        self.preview_phases = int(preview_phases)
        self.moment_tolerance = float(moment_tolerance)
        self.max_iterations = int(max_iterations)
        self.fallback_to_greedy = bool(fallback_to_greedy)
        self.optimality_tolerance = float(optimality_tolerance)
        self.recruitment_regularization = float(recruitment_regularization)
        self.max_solve_time_s = float(max_solve_time_s)

    def _greedy_step(self, state, phase, *, seed_only=False):
        transition = self.predictor.phase_map(state, phase)
        interval = self.predictor.intervals[phase]
        coefficients = np.asarray(interval.moment_coefficients)
        baseline = coefficients * transition.intercept[:, 1]
        slope = coefficients * transition.slope[:, 1]
        maximal = baseline + slope * transition.maximum_recruitment
        lower, upper = np.minimum(baseline, maximal), np.maximum(baseline, maximal)
        required = float(np.sum(interval.target_moments))
        target = float(np.clip(required, lower.sum(), upper.sum())) if seed_only else required
        allocation = solve_bounded_moment_qp_reference(
            interval.target_moments, required_total_moment=target, lower_bounds=lower, upper_bounds=upper,
            feasibility_tolerance=self.moment_tolerance)
        if allocation.allocated_moments is None:
            raise ValueError(allocation.status)
        recruitment = np.divide(allocation.allocated_moments - baseline, slope,
                                out=np.zeros_like(slope), where=np.abs(slope) > 1e-14)
        if np.any(recruitment < -1e-12) or np.any(recruitment > transition.maximum_recruitment + 1e-12):
            raise ValueError("recruitment_inversion_outside_bounds")
        recruitment = np.clip(recruitment, 0., transition.maximum_recruitment)
        endpoint = transition.endpoint(recruitment)
        if not _physical(endpoint):
            raise ValueError("predicted_state_outside_domain")
        actual = float(np.sum(coefficients * endpoint[:, 1]))
        if abs(actual - target) > self.moment_tolerance * 1.01:
            raise ValueError("moment_inversion_residual")
        return transition, recruitment, endpoint, lower, upper, actual - required

    def build_preview_qp(self, states, interval_index, *, depth=None):
        depth = self.preview_phases if depth is None else depth
        if isinstance(depth, bool) or int(depth) != depth or not 1 <= depth <= self.preview_phases:
            raise ValueError("depth must be a positive integer at most preview_phases.")
        depth = int(depth)
        count, muscles = len(self.predictor.intervals), self.predictor.muscle_count
        current = np.asarray(states, float).copy()
        if current.shape != (muscles, 5) or not _physical(current):
            raise ValueError("states must be physical with shape (muscles, 5).")
        size = depth * muscles
        offsets, jacobians, seed, targets, refs, weights, coefficients, seed_errors = ([] for _ in range(8))
        offset, jacobian = current[:, 1].copy(), np.zeros((muscles, size))
        options = SmoothMomentAllocationOptions()
        for k in range(depth):
            phase = (interval_index + k) % count
            interval = self.predictor.intervals[phase]
            transition, recruitment, endpoint, lower, upper, seed_error = self._greedy_step(current, phase, seed_only=True)
            # The r=0 force response is homogeneous in incoming force. Use a
            # unit-force map to obtain its coefficient, including at F=0.
            unit = current.copy()
            unit[:, 1] = 1.
            decay = self.predictor.phase_map(unit, phase).intercept[:, 1]
            jacobian = decay[:, None] * jacobian
            jacobian[:, k * muscles:(k + 1) * muscles] += np.diag(transition.slope[:, 1])
            offset = decay * offset
            offsets.append(offset.copy())
            jacobians.append(jacobian.copy())
            seed.append(recruitment)
            targets.append(sum(interval.target_moments))
            reference = np.asarray(interval.target_moments)
            up = options.slack_temperature * np.logaddexp(0., (upper - reference) / options.slack_temperature)
            down = options.slack_temperature * np.logaddexp(0., (reference - lower) / options.slack_temperature)
            # Original required total equals sum(reference): directional gate=1/2.
            weights.append(1. / (.5 * up**2 + .5 * down**2 + options.flexibility_floor))
            refs.append(reference)
            coefficients.append(interval.moment_coefficients)
            seed_errors.append(seed_error)
            current = endpoint
        offsets, jacobians, coefficients = np.asarray(offsets), np.asarray(jacobians), np.asarray(coefficients)
        moment_offsets = (coefficients * offsets).reshape(-1)
        moment_jacobian = (coefficients[:, :, None] * jacobians).reshape(size, size)
        equality = (coefficients[:, :, None] * jacobians).sum(axis=1)
        equality_target = np.asarray(targets) - (coefficients * offsets).sum(axis=1)
        reference, weight = np.asarray(refs).reshape(-1), np.asarray(weights).reshape(-1)
        hessian = moment_jacobian.T @ (weight[:, None] * moment_jacobian)
        normalization = max(float(np.max(np.abs(hessian))), 1.)
        weight /= normalization
        hessian /= normalization
        hessian += self.recruitment_regularization * np.eye(size)
        deviation = moment_offsets - reference
        linear = moment_jacobian.T @ (weight * deviation)
        return PreviewAllocationProblem(
            depth, np.asarray(seed).reshape(-1), np.tile(self.predictor.maximum_recruitment, depth),
            offsets, jacobians, equality, equality_target, reference, moment_offsets, moment_jacobian,
            weight, hessian, linear, float(.5 * weight @ (deviation * deviation)),
            int(np.sum(np.abs(seed_errors) > self.moment_tolerance)), np.asarray(seed_errors))

    def plan(self, states, interval_index, *, depth=None):
        started = perf_counter()
        depth = self.preview_phases if depth is None else depth
        diagnostics = {"preview_depth": depth, "qp_solved": False, "optimizer_success": False,
                       "optimizer_status": None, "optimizer_iterations": 0, "fallback_reason": None}
        try:
            if depth == 1:
                _, recruitment, _, _, _, error = self._greedy_step(states, interval_index)
                diagnostics.update(accepted=True, reason="exact_greedy_depth_one", equality_residual_max_nm=abs(error),
                                   bound_violation_max=0., elapsed_s=perf_counter() - started, nominal_projected_steps=0)
                return PreviewAllocationPlan(True, recruitment, None, diagnostics)
            problem = self.build_preview_qp(states, interval_index, depth=depth)
        except ValueError as error:
            diagnostics.update(accepted=False, reason=str(error), elapsed_s=perf_counter() - started)
            return PreviewAllocationPlan(False, None, None, diagnostics)
        diagnostics["nominal_projected_steps"] = problem.nominal_projected_steps
        diagnostics["nominal_original_target_errors_nm"] = problem.nominal_original_target_errors_nm.tolist()
        matrix, target = problem.equality_matrix, problem.equality_target
        # Keep a linearly independent equality subset for SLSQP, while auditing
        # ALL original equalities afterward, including control-invisible rows.
        _, triangular, permutation = qr(matrix.T, pivoting=True, mode="economic")
        scale = np.linalg.norm(matrix, ord=2)
        rank = int(np.sum(np.abs(np.diag(triangular)) > max(matrix.shape) * np.finfo(float).eps * scale))
        selected = permutation[:rank]
        constraint_matrix, constraint_target = matrix[selected], target[selected]
        candidate = np.linalg.lstsq(matrix, target, rcond=None)[0]
        if np.max(np.abs(matrix @ candidate - target)) > self.moment_tolerance:
            diagnostics.update(accepted=False, reason="inconsistent_affine_equalities", elapsed_s=perf_counter() - started)
            return PreviewAllocationPlan(False, None, problem, diagnostics)
        deadline = perf_counter() + self.max_solve_time_s
        class TimeLimit(Exception):
            pass
        def objective(x):
            if perf_counter() > deadline:
                raise TimeLimit()
            return problem.objective(x)
        constraints = [] if rank == 0 else [{"type": "eq", "fun": lambda x: constraint_matrix @ x - constraint_target,
                                            "jac": lambda x: constraint_matrix}]
        try:
            diagnostics["qp_solved"] = True
            solved = minimize(objective, problem.initial_guess,
                              jac=lambda x: problem.hessian @ x + problem.linear_term,
                              method="SLSQP", bounds=list(zip(np.zeros_like(problem.upper_bounds), problem.upper_bounds)),
                              constraints=constraints, options={"maxiter": self.max_iterations, "ftol": 1e-12})
        except TimeLimit:
            diagnostics.update(accepted=False, reason="qp_time_limit", elapsed_s=perf_counter() - started)
            return PreviewAllocationPlan(False, None, problem, diagnostics)
        x = np.asarray(solved.x)
        if x.shape != problem.initial_guess.shape or not np.all(np.isfinite(x)):
            diagnostics.update(accepted=False, reason="nonfinite_or_invalid_qp_solution", elapsed_s=perf_counter() - started)
            return PreviewAllocationPlan(False, None, problem, diagnostics)
        equality_error = float(np.max(np.abs(matrix @ x - target)))
        bound_error = float(max(np.max(-x), np.max(x - problem.upper_bounds), 0.))
        gradient = problem.hessian @ x + problem.linear_term
        identity = np.eye(x.size)
        active_lower, active_upper = x <= 1e-7, problem.upper_bounds - x <= 1e-7
        multiplier_matrix = np.column_stack((constraint_matrix.T, -identity[:, active_lower], identity[:, active_upper]))
        if multiplier_matrix.shape[1]:
            lower_multipliers = np.r_[np.full(rank, -np.inf), np.zeros(active_lower.sum() + active_upper.sum())]
            multipliers = lsq_linear(multiplier_matrix, -gradient, bounds=(lower_multipliers, np.inf),
                                     tol=1e-12, max_iter=100)
            stationarity = gradient + multiplier_matrix @ multipliers.x
            bound_distances = np.r_[x[active_lower], problem.upper_bounds[active_upper] - x[active_upper]]
            complementarity_error = float(np.max(np.abs(multipliers.x[rank:] * bound_distances), initial=0.))
        else:
            stationarity = gradient
            complementarity_error = 0.
        kkt_error = float(np.max(np.abs(stationarity)))
        accepted = bool(np.all(np.isfinite(x)) and equality_error <= self.moment_tolerance
                        and bound_error <= 1e-12 and kkt_error <= self.optimality_tolerance
                        and complementarity_error <= self.optimality_tolerance)
        moment_deviation = problem.moment_offsets + problem.moment_jacobian @ x - problem.reference_moments
        diagnostics.update(accepted=accepted, reason="accepted" if accepted else "qp_feasibility_or_optimality_audit_failed",
                           optimizer_success=bool(solved.success), optimizer_status=int(solved.status),
                           optimizer_message=str(solved.message), optimizer_iterations=int(solved.nit),
                           optimizer_objective=problem.objective(x), equality_residual_max_nm=equality_error,
                           bound_violation_max=bound_error, kkt_stationarity_max=kkt_error,
                           kkt_complementarity_max=complementarity_error,
                           moment_deviation_objective=float(.5 * problem.normalized_weights @ (moment_deviation**2)),
                           recruitment_regularization_objective=float(.5 * self.recruitment_regularization * (x @ x)),
                           elapsed_s=perf_counter() - started)
        return PreviewAllocationPlan(accepted, np.clip(x[:self.predictor.muscle_count], 0.,
                                                       self.predictor.maximum_recruitment) if accepted else None,
                                     problem, diagnostics)

    def rollout(self, initial_states, *, horizon_cycles, moment_tolerance=None):
        if moment_tolerance is not None and moment_tolerance != self.moment_tolerance:
            raise ValueError("moment_tolerance must match the configured preview policy.")
        if isinstance(horizon_cycles, bool) or int(horizon_cycles) != horizon_cycles or horizon_cycles < 1:
            raise ValueError("horizon_cycles must be a positive integer.")
        horizon, count, muscles = int(horizon_cycles), len(self.predictor.intervals), self.predictor.muscle_count
        current = np.asarray(initial_states, float).copy()
        self.predictor.phase_map(current, 0)
        shape = (horizon, muscles, count)
        pws, allocated, achieved = (np.full(shape, np.nan) for _ in range(3))
        history = np.full((horizon * count + 1, muscles, 5), np.nan)
        history[0] = current
        lower, upper, original, errors = (np.full((horizon, count), np.nan) for _ in range(4))
        diagnostics, failure, completed = [], None, 0
        for step in range(horizon * count):
            cycle, phase = divmod(step, count)
            interval = self.predictor.intervals[phase]
            transition = self.predictor.phase_map(current, phase)
            coefficients = np.asarray(interval.moment_coefficients)
            baseline = coefficients * transition.intercept[:, 1]
            maximal = baseline + coefficients * transition.slope[:, 1] * transition.maximum_recruitment
            lower[cycle, phase], upper[cycle, phase] = np.minimum(baseline, maximal).sum(), np.maximum(baseline, maximal).sum()
            original[cycle, phase] = sum(interval.target_moments)
            plan = self.plan(current, phase, depth=min(self.preview_phases, horizon * count - step))
            record = dict(plan.diagnostics, cycle_index=cycle, interval_index=phase, greedy_fallback=False)
            recruitment = plan.recruitment
            if not plan.accepted and self.fallback_to_greedy:
                record["fallback_reason"] = record["reason"]
                try:
                    _, recruitment, _, _, _, _ = self._greedy_step(current, phase)
                    record["greedy_fallback"] = True
                except ValueError as error:
                    record["fallback_failure"] = str(error)
            diagnostics.append(record)
            if recruitment is None:
                failure = {"cycle_index": cycle, "interval_index": phase, "status": record["reason"]}
                break
            endpoint = transition.endpoint(recruitment)
            predicted = coefficients * endpoint[:, 1]
            error = float(predicted.sum() - original[cycle, phase])
            if not _physical(endpoint) or abs(error) > self.moment_tolerance * 1.01:
                failure = {"cycle_index": cycle, "interval_index": phase, "status": "applied_phase_domain_or_tracking_failure",
                           "signed_moment_error_nm": error}
                break
            p = self.predictor
            pws[cycle, :, phase] = np.minimum(p.pulse_width_max, p.pd0 - p.pdt * np.log1p(-recruitment))
            allocated[cycle, :, phase], achieved[cycle, :, phase] = predicted, predicted
            errors[cycle, phase] = error
            current = endpoint
            completed += 1
            history[completed] = current
        return PreviewMomentRolloutResult(
            "complete" if failure is None else "infeasible", horizon, completed // count, completed,
            pws, allocated, achieved, history, failure, 0, self.preview_phases,
            sum(r["qp_solved"] for r in diagnostics), sum(r["optimizer_iterations"] for r in diagnostics),
            sum(r["qp_solved"] and not r["accepted"] for r in diagnostics), sum(r["greedy_fallback"] for r in diagnostics),
            sum(r.get("nominal_projected_steps", 0) for r in diagnostics), tuple(diagnostics), lower, upper, original, errors)
