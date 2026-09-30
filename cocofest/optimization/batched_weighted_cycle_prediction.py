"""Vectorized independent weight candidates for the compact PACE rollout.

Only candidates are batched.  Time remains strictly sequential because every
phase starts from the preceding predicted state.  The equations are identical
to :mod:`weighted_cycle_prediction`; batching avoids repeated Python loops and
does not create CPU contention with the RHO solver.
"""

from dataclasses import dataclass

import numpy as np

from .batched_compact_muscle_prediction import BatchedCompactMusclePredictor, _physical
from .weighted_cycle_prediction import (
    ALLOCATION_OBJECTIVE_PREDICTED_DING_FATIGUE,
    ALLOCATION_OBJECTIVE_RECRUITMENT,
    POLICY_NAME,
    WeightedCycleRolloutResult,
    _weights,
    rollout_fatigue_metrics,
)


@dataclass(frozen=True)
class BatchedWeightedRecruitmentAllocation:
    statuses: tuple[str, ...]
    normalized_recruitment: np.ndarray
    normalized_weights: np.ndarray
    reference_recruitment: np.ndarray
    signed_margins: np.ndarray
    quadratic_hessian_diagonal: np.ndarray
    quadratic_linear: np.ndarray


def solve_weighted_recruitment_many(
    moment_intercept,
    moment_slope,
    reference_moments,
    weights,
    *,
    reference_regularization=1e-3,
    moment_tolerance=1e-8,
    project_infeasible=False,
    allocation_objective=ALLOCATION_OBJECTIVE_RECRUITMENT,
    capacity_intercept=None,
    capacity_slope=None,
    rest_capacity=None,
    phase_duration=1.0,
):
    """Vectorized equivalent of ``solve_weighted_recruitment``.

    With ``project_infeasible=True``, out-of-envelope rows receive the exact
    nearest signed-moment endpoint used by PACE's projected-capacity mode.
    Their status remains infeasible so the caller can retain the deficit audit.
    """

    intercept, slope, reference = [np.asarray(value, dtype=float) for value in (
        moment_intercept, moment_slope, reference_moments
    )]
    if (intercept.ndim != 2 or not intercept.shape[0] or not intercept.shape[1]
            or slope.shape != intercept.shape):
        raise ValueError("moment inputs must have matching nonempty (candidates, muscles) shape.")
    reference = np.broadcast_to(reference, intercept.shape)
    if not all(np.all(np.isfinite(value)) for value in (intercept, slope, reference)):
        raise ValueError("moment inputs must be finite.")
    if not np.isfinite(reference_regularization) or reference_regularization <= 0:
        raise ValueError("reference_regularization must be finite and strictly positive.")
    if not np.isfinite(moment_tolerance) or moment_tolerance <= 0:
        raise ValueError("moment_tolerance must be finite and strictly positive.")
    if allocation_objective not in {
            ALLOCATION_OBJECTIVE_RECRUITMENT, ALLOCATION_OBJECTIVE_PREDICTED_DING_FATIGUE}:
        raise ValueError(f"Unknown allocation_objective: {allocation_objective!r}")
    raw_weights = np.asarray(weights, dtype=float)
    if raw_weights.shape != intercept.shape:
        raise ValueError("weights must have shape (candidates, muscles).")
    normalized = np.asarray([_weights(row, intercept.shape[1]) for row in raw_weights])
    xref = np.divide(reference - intercept, slope, out=np.zeros_like(slope), where=slope != 0)
    xref = np.clip(xref, 0.0, 1.0)
    if allocation_objective == ALLOCATION_OBJECTIVE_RECRUITMENT:
        diagonal = normalized + reference_regularization
        linear = reference_regularization * xref
    else:
        capacity0 = np.asarray(capacity_intercept, dtype=float)
        capacity1 = np.asarray(capacity_slope, dtype=float)
        rest = np.asarray(rest_capacity, dtype=float)
        try:
            capacity0 = np.broadcast_to(capacity0, intercept.shape)
            capacity1 = np.broadcast_to(capacity1, intercept.shape)
            rest = np.broadcast_to(rest, intercept.shape)
        except ValueError as error:
            raise ValueError("fatigue objective capacity arrays must broadcast to (candidates, muscles).") from error
        if (not all(np.all(np.isfinite(value)) for value in (capacity0, capacity1, rest))
                or np.any(rest <= 0)):
            raise ValueError("fatigue objective requires finite capacity arrays and positive rest capacity.")
        if not np.isfinite(phase_duration) or phase_duration <= 0:
            raise ValueError("phase_duration must be finite and strictly positive.")
        residual0 = 1.0 - capacity0 / rest
        residual1 = -capacity1 / rest
        diagonal = phase_duration * normalized * residual1**2 + reference_regularization
        linear = (reference_regularization * xref
                  - phase_duration * normalized * residual0 * residual1)
    target = reference.sum(axis=1)
    lower = np.sum(intercept + np.minimum(slope, 0.0), axis=1)
    upper = np.sum(intercept + np.maximum(slope, 0.0), axis=1)
    margins = np.minimum(target - lower, upper - target)
    below = target < lower - moment_tolerance
    above = target > upper + moment_tolerance
    statuses = np.full(len(intercept), "ok", dtype=object)
    statuses[below] = "infeasible_total_below_bounds"
    statuses[above] = "infeasible_total_above_bounds"
    x = np.full_like(intercept, np.nan)

    infeasible = below | above
    if project_infeasible and np.any(infeasible):
        rows = np.flatnonzero(infeasible)
        projected = np.clip(linear[rows] / diagonal[rows], 0.0, 1.0)
        active = slope[rows] != 0
        toward_upper = above[rows, None]
        endpoint = np.where(toward_upper, slope[rows] > 0, slope[rows] < 0)
        projected[active] = endpoint[active].astype(float)
        x[rows] = projected

    feasible = ~infeasible
    rows = np.flatnonzero(feasible)
    if rows.size:
        local_intercept = intercept[rows]
        local_slope = slope[rows]
        local_target = target[rows]
        local_normalized = normalized[rows]
        local_xref = xref[rows]
        local_diagonal = diagonal[rows]
        local_linear = linear[rows]
        active = local_slope != 0
        answer = np.clip(local_linear / local_diagonal, 0.0, 1.0)
        at_lower = local_target <= lower[rows]
        at_upper = ~at_lower & (local_target >= upper[rows])
        answer[at_lower] = (local_slope[at_lower] < 0).astype(float)
        answer[at_upper] = (local_slope[at_upper] > 0).astype(float)
        interior = ~(at_lower | at_upper)
        if np.any(interior):
            ir = np.flatnonzero(interior)
            a_slope = local_slope[ir]
            scale = np.max(np.abs(a_slope), axis=1)
            a = a_slope / scale[:, None]
            rhs = (local_target[ir] - local_intercept[ir].sum(axis=1)) / scale
            diag = local_diagonal[ir]
            lin = local_linear[ir]
            is_active = active[ir]
            with np.errstate(divide="ignore", invalid="ignore"):
                zero_crossing = np.divide(-lin, a)
                one_crossing = np.divide(diag - lin, a)
            crossings = np.concatenate((zero_crossing, one_crossing), axis=1)
            # Inactive coordinates never change with the multiplier.  Zero is
            # a harmless duplicate breakpoint and, unlike infinity, cannot
            # contaminate the vectorized candidate evaluation through 0*inf.
            crossings[:, : a.shape[1]][~is_active] = 0.0
            crossings[:, a.shape[1] :][~is_active] = 0.0
            crossings.sort(axis=1)
            candidate = np.clip(
                (lin[:, None, :] + crossings[:, :, None] * a[:, None, :])
                / diag[:, None, :],
                0.0,
                1.0,
            )
            totals = np.einsum("bij,bj->bi", candidate, a)
            crossed = totals >= rhs[:, None]
            right_index = np.argmax(crossed, axis=1)
            right_index[~np.any(crossed, axis=1)] = crossings.shape[1] - 1
            left_index = np.maximum(right_index - 1, 0)
            row_index = np.arange(len(ir))
            left_lambda = crossings[row_index, left_index]
            right_lambda = crossings[row_index, right_index]
            left_total = totals[row_index, left_index]
            right_total = totals[row_index, right_index]
            fraction = np.divide(
                rhs - left_total,
                right_total - left_total,
                out=np.ones_like(rhs),
                where=right_total > left_total,
            )
            multiplier = left_lambda + fraction * (right_lambda - left_lambda)
            answer[ir] = np.clip((lin + multiplier[:, None] * a) / diag, 0.0, 1.0)
        residual = np.sum(local_intercept + local_slope * answer, axis=1) - local_target
        inaccurate = np.abs(residual) > moment_tolerance
        statuses[rows[inaccurate]] = "moment_allocation_residual"
        answer[inaccurate] = np.nan
        x[rows] = answer
    return BatchedWeightedRecruitmentAllocation(
        tuple(statuses), x, normalized, xref, margins, diagonal, linear
    )


class BatchedWeightedCyclePredictor:
    """Run independent PACE weight candidates in one NumPy phase loop."""

    def __init__(self, predictor):
        self.predictor = predictor
        self.batch_predictor = BatchedCompactMusclePredictor(predictor)

    def rollout_many(
        self,
        initial_states,
        weights,
        *,
        horizon_cycles,
        moment_tolerance=1e-8,
        tracking_mode="exact",
        score_block_cycles=20,
    ):
        if tracking_mode not in ("exact", "projected_capacity"):
            raise ValueError("tracking_mode must be exact or projected_capacity.")
        if isinstance(horizon_cycles, bool) or int(horizon_cycles) != horizon_cycles or horizon_cycles < 1:
            raise ValueError("horizon_cycles must be a positive integer.")
        if isinstance(score_block_cycles, bool) or int(score_block_cycles) != score_block_cycles or score_block_cycles < 1:
            raise ValueError("score_block_cycles must be a positive integer.")
        if not np.isfinite(moment_tolerance) or moment_tolerance <= 0:
            raise ValueError("moment_tolerance must be finite and strictly positive.")
        p = self.predictor
        weights = np.asarray(weights, dtype=float)
        if weights.ndim != 2 or weights.shape[1] != p.muscle_count or not len(weights):
            raise ValueError("weights must have nonempty shape (candidates, muscles).")
        normalized = np.asarray([_weights(row, p.muscle_count) for row in weights])
        initial = np.asarray(initial_states, dtype=float)
        if initial.shape != (p.muscle_count, 5):
            raise ValueError("initial_states must have shape (muscles, 5).")
        batch = len(weights)
        cycles, phases, muscles = int(horizon_cycles), len(p.intervals), p.muscle_count
        current = np.broadcast_to(initial, (batch, muscles, 5)).copy()
        shape = (batch, cycles, muscles, phases)
        pws, allocated, achieved, integrals = [np.full(shape, np.nan) for _ in range(4)]
        original_one = np.asarray([sum(interval.target_moments) for interval in p.intervals])
        original = np.broadcast_to(original_one, (batch, cycles, phases)).copy()
        errors, margins = [np.full((batch, cycles, phases), np.nan) for _ in range(2)]
        history = np.full((batch, cycles * phases + 1, muscles, 5), np.nan)
        cycle_integrals = np.full((batch, cycles, muscles), np.nan)
        completed = np.zeros(batch, dtype=int)
        duration = np.zeros(batch)
        failures = [None] * batch
        first_deficits = [None] * batch
        alive = _physical(current)
        history[alive, 0] = current[alive]
        for row in np.flatnonzero(~alive):
            failures[row] = {"cycle_index": 0, "interval_index": 0,
                             "status": "initial_state_outside_domain"}
        phase_durations = np.asarray([interval.duration for interval in p.intervals])
        reference_rms = float(np.sqrt(np.average(original_one**2, weights=phase_durations)))
        reference_scale = max(reference_rms, moment_tolerance)

        def fail(rows, cycle, phase, status):
            for row in rows:
                if failures[row] is None:
                    failures[row] = {"cycle_index": cycle, "interval_index": phase, "status": status}
            alive[rows] = False

        running_cycle_integral = np.zeros((batch, muscles))
        for step_index in range(cycles * phases):
            rows = np.flatnonzero(alive)
            if not rows.size:
                break
            cycle, phase = divmod(step_index, phases)
            if phase == 0:
                running_cycle_integral[rows] = 0.0
            interval = p.intervals[phase]
            transition = self.batch_predictor.phase_map_many(current[rows], phase)
            maximum = transition.maximum_recruitment
            maximum_states = transition.intercept + transition.slope * maximum[:, :, None]
            valid_envelope = _physical(transition.intercept) & _physical(maximum_states)
            fail(rows[~valid_envelope], cycle, phase, "envelope_domain_invalid")
            active_local = valid_envelope.copy()
            if not np.any(active_local):
                continue
            selected_rows = rows[active_local]
            coefficients = np.asarray(interval.moment_coefficients)
            moment0 = coefficients * transition.intercept[active_local, :, 1]
            slope = coefficients * transition.slope[active_local, :, 1] * maximum[active_local]
            allocation = solve_weighted_recruitment_many(
                moment0,
                slope,
                interval.target_moments,
                normalized[selected_rows],
                reference_regularization=p.reference_regularization,
                moment_tolerance=moment_tolerance,
                project_infeasible=tracking_mode == "projected_capacity",
                allocation_objective=p.allocation_objective,
                capacity_intercept=transition.intercept[active_local, :, 2],
                capacity_slope=(transition.slope[active_local, :, 2]
                                * maximum[active_local]),
                # ``rest`` stores [A_rest, Tau1_rest, Km_rest].  Keep this
                # vectorized path algebraically identical to the scalar one.
                rest_capacity=p.rest[:, 0],
                phase_duration=interval.duration,
            )
            # The scalar predictor records the signed reachability margin for
            # the first failed phase as well as for accepted phases.
            margins[selected_rows, cycle, phase] = allocation.signed_margins
            local_success = np.ones(len(selected_rows), dtype=bool)
            for local, status in enumerate(allocation.statuses):
                if status.startswith("infeasible_total_"):
                    if tracking_mode == "projected_capacity":
                        if first_deficits[selected_rows[local]] is None:
                            first_deficits[selected_rows[local]] = {
                                "cycle_index": cycle, "interval_index": phase,
                                "status": status,
                                "signed_margin_nm": float(allocation.signed_margins[local]),
                            }
                    else:
                        fail([selected_rows[local]], cycle, phase, status)
                        local_success[local] = False
                elif status != "ok":
                    fail([selected_rows[local]], cycle, phase, status)
                    local_success[local] = False
            if not np.any(local_success):
                continue
            accepted = selected_rows[local_success]
            fractions = allocation.normalized_recruitment[local_success]
            local_transition_indices = np.flatnonzero(active_local)[local_success]
            recruitment = fractions * maximum[local_transition_indices]
            next_states = (
                transition.intercept[local_transition_indices]
                + transition.slope[local_transition_indices] * recruitment[:, :, None]
            )
            physical = _physical(next_states)
            fail(accepted[~physical], cycle, phase, "predicted_state_outside_domain")
            if not np.any(physical):
                continue
            accepted = accepted[physical]
            next_states = next_states[physical]
            fractions = fractions[physical]
            recruitment = recruitment[physical]
            local_transition_indices = local_transition_indices[physical]
            moment = coefficients * next_states[:, :, 1]
            residual = moment.sum(axis=1) - original_one[phase]
            was_feasible = np.asarray(allocation.statuses, dtype=object)[local_success][physical] == "ok"
            accurate = ~was_feasible | (np.abs(residual) <= moment_tolerance)
            fail(accepted[~accurate], cycle, phase, "moment_inversion_residual")
            if not np.any(accurate):
                continue
            accepted = accepted[accurate]
            next_states = next_states[accurate]
            fractions = fractions[accurate]
            recruitment = recruitment[accurate]
            local_transition_indices = local_transition_indices[accurate]
            moment = moment[accurate]
            residual = residual[accurate]
            pws[accepted, cycle, :, phase] = np.clip(
                p.pd0 - p.pdt * np.log1p(-recruitment), p.pd0, p.pulse_width_max
            )
            allocated[accepted, cycle, :, phase] = moment0[local_success][physical][accurate] + (
                slope[local_success][physical][accurate] * fractions
            )
            achieved[accepted, cycle, :, phase] = moment
            errors[accepted, cycle, phase] = residual
            margins[accepted, cycle, phase] = allocation.signed_margins[local_success][physical][accurate]
            integral = (
                transition.weighted_force_intercept[local_transition_indices]
                + transition.weighted_force_slope[local_transition_indices] * recruitment
            )
            integrals[accepted, cycle, :, phase] = integral
            running_cycle_integral[accepted] = (
                np.exp(-interval.duration / p.tau_fat) * running_cycle_integral[accepted] + integral
            )
            current[accepted] = next_states
            completed[accepted] += 1
            duration[accepted] += interval.duration
            history[accepted, step_index + 1] = next_states
            if phase == phases - 1:
                cycle_integrals[accepted, cycle] = running_cycle_integral[accepted]

        results = []
        for row in range(batch):
            failure = failures[row]
            diagnostics = {}
            if tracking_mode == "projected_capacity":
                diagnostics = {
                    "first_target_deficit": first_deficits[row],
                    "normalization_reference_rms_nm": reference_rms,
                    "normalization_scale_nm": reference_scale,
                    "score_block_cycles": int(score_block_cycles),
                    "full_horizon_normalized_deficit": None,
                    "terminal_normalized_reserve": None,
                    "cycle_normalized_squared_deficit": [],
                    "block_diagnostics": [],
                    "continuation_is_feasible_movement": False,
                }
                if failure is None:
                    loss = np.average((errors[row] / reference_scale) ** 2,
                                      axis=1, weights=phase_durations)
                    reserve = np.average(margins[row] / reference_scale,
                                         axis=1, weights=phase_durations)
                    diagnostics.update({
                        "full_horizon_normalized_deficit": float(np.mean(loss)),
                        "terminal_normalized_reserve": float(np.mean(reserve[-int(score_block_cycles):])),
                        "cycle_normalized_squared_deficit": loss.tolist(),
                        "block_diagnostics": [
                            {"start_cycle": start, "stop_cycle": min(start + int(score_block_cycles), cycles),
                             "mean_squared_deficit": float(np.mean(loss[start:start + int(score_block_cycles)])),
                             "mean_signed_reserve": float(np.mean(reserve[start:start + int(score_block_cycles)]))}
                            for start in range(0, cycles, int(score_block_cycles))
                        ],
                    })
                    diagnostics.update(rollout_fatigue_metrics(
                        history[row], self.predictor.rest[:, 0], phase_durations, int(score_block_cycles)))
            status = (
                ("complete_with_deficit" if first_deficits[row] else "complete")
                if failure is None else
                "numerical_failure" if failure["status"] in (
                    "allocation_numerical_failure", "moment_allocation_residual", "moment_inversion_residual"
                ) else "infeasible"
            )
            results.append(WeightedCycleRolloutResult(
                status=status,
                requested_cycles=cycles,
                completed_cycles=int(completed[row]) // phases,
                completed_intervals=int(completed[row]),
                completed_duration=float(duration[row]),
                pulse_widths=pws[row],
                allocated_moments=allocated[row],
                achieved_moments=achieved[row],
                original_total_moments=original[row],
                signed_moment_errors=errors[row],
                signed_margins=margins[row],
                state_history=history[row],
                weighted_force_integrals=integrals[row],
                cycle_weighted_force_integrals=cycle_integrals[row],
                cycle_fatigue_forcing=cycle_integrals[row, :, :, None] * p.alpha[None, :, :],
                first_failure=failure,
                metadata={
                    "policy": POLICY_NAME,
                    "reference_regularization": p.reference_regularization,
                    "normalized_weights": normalized[row].tolist(),
                    "all_ones_is_old_greedy": False,
                    "force_model": "compact_phase_frozen_slow_states",
                    "moment_constraint_sampling": "phase_endpoints_only",
                    "full_ode_or_rho_feasibility_certified": False,
                    "future_cycles_explicitly_propagated": True,
                    "tracking_mode": ("exact_with_numerical_tolerance" if tracking_mode == "exact"
                                      else "projected_capacity"),
                    "moment_tolerance": float(moment_tolerance),
                    "allocation_objective": p.allocation_objective,
                    "fatigue_objective_approximation": (
                        "endpoint_A_plus_rectangular_phase_quadrature"
                        if p.allocation_objective == ALLOCATION_OBJECTIVE_PREDICTED_DING_FATIGUE
                        else None),
                    "fatigue_objective_full_ding_or_rho_certified": False,
                    "substeps": p.substeps,
                    "evaluation_backend": "candidate_batch",
                    **diagnostics,
                },
            ))
        return tuple(results)


__all__ = [
    "BatchedWeightedCyclePredictor",
    "BatchedWeightedRecruitmentAllocation",
    "solve_weighted_recruitment_many",
]
