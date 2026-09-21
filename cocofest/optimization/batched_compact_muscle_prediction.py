"""NumPy batches of independent compact policies; no additional NLP variables.

Equations and midpoint quadrature match CompactMusclePredictor. The bounded
QP uses its identical flexibility weights with sorted multiplier breakpoints.
A physical tracking band is separate from numerical solver tolerance. Failure
of this policy does not establish global infeasibility.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, exprel

from .adaptive_moment_rollout import TotalMomentReferenceRolloutResult
from .smooth_muscle_moment_allocation import SmoothMomentAllocationOptions


@dataclass(frozen=True)
class BatchedBoundedMomentQpResult:
    statuses: tuple[str, ...]
    allocated_moments: np.ndarray
    effective_total_targets: np.ndarray
    equality_residuals: np.ndarray
    relaxed: np.ndarray


def solve_bounded_moment_qp_many(reference_moments, *, required_total_moment,
                                 lower_bounds, upper_bounds,
                                 options=SmoothMomentAllocationOptions(),
                                 feasibility_tolerance=1e-10, tracking_band_nm=0.):
    """Solve independent diagonal bounded equality QPs with signed moments.

    ``lower_bounds`` and ``upper_bounds`` have shape (batch, muscles). The
    reference may broadcast from (muscles,), and targets may be scalar.
    Invalid/infeasible rows return NaNs while other rows remain usable.
    Outside-envelope targets are admitted within tracking_band_nm plus the
    separate numerical feasibility_tolerance, preserving scalar tolerance
    behavior when the physical band is zero. Only projections larger than
    feasibility_tolerance are labeled physical relaxation.
    """
    if not np.isfinite(feasibility_tolerance) or feasibility_tolerance <= 0:
        raise ValueError("feasibility_tolerance must be finite and positive.")
    if not np.isfinite(tracking_band_nm) or tracking_band_nm < 0:
        raise ValueError("tracking_band_nm must be finite and nonnegative.")
    lower, upper = np.asarray(lower_bounds, float), np.asarray(upper_bounds, float)
    if lower.ndim != 2 or not lower.shape[0] or not lower.shape[1] or upper.shape != lower.shape:
        raise ValueError("Bounds must have matching nonempty (batch, muscles) shape.")
    batch, muscles = lower.shape
    reference = np.broadcast_to(np.asarray(reference_moments, float), lower.shape)
    required = np.broadcast_to(np.asarray(required_total_moment, float), (batch,))
    allocated = np.full(lower.shape, np.nan)
    effective = np.full(batch, np.nan)
    residual = np.full(batch, np.nan)
    relaxed = np.zeros(batch, bool)
    statuses = np.full(batch, "invalid_bounds_or_target", dtype=object)
    valid = (np.all(np.isfinite(lower) & np.isfinite(upper) & np.isfinite(reference) & (lower <= upper), axis=1)
             & np.isfinite(required))
    low_total, high_total = lower.sum(axis=1), upper.sum(axis=1)
    distance = np.maximum(np.maximum(low_total - required, required - high_total), 0.)
    admission = tracking_band_nm + feasibility_tolerance
    statuses[valid & (required < low_total) & (distance > admission)] = "infeasible_total_below_bounds"
    statuses[valid & (required > high_total) & (distance > admission)] = "infeasible_total_above_bounds"
    rows = np.flatnonzero(valid & (distance <= admission))
    if rows.size:
        lo, hi, ref = lower[rows], upper[rows], reference[rows]
        target = np.clip(required[rows], low_total[rows], high_total[rows])
        effective[rows] = target
        relaxed[rows] = np.abs(target - required[rows]) > feasibility_tolerance
        correction = target - ref.sum(axis=1)
        up = options.slack_temperature * np.logaddexp(0., (hi - ref) / options.slack_temperature)
        down = options.slack_temperature * np.logaddexp(0., (ref - lo) / options.slack_temperature)
        gate = expit(correction / options.direction_temperature)[:, None]
        flexibility = gate * up**2 + (1 - gate) * down**2 + options.flexibility_floor
        # Rescaling all weights in a row leaves its minimizer unchanged and
        # keeps multiplier breakpoints away from unnecessarily large values.
        flexibility /= flexibility.max(axis=1, keepdims=True)
        breaks = np.sort(np.concatenate(((lo - ref) / flexibility,
                                         (hi - ref) / flexibility), axis=1), axis=1)
        candidates = np.clip(ref[:, None, :] + breaks[:, :, None] * flexibility[:, None, :],
                             lo[:, None, :], hi[:, None, :])
        totals = candidates.sum(axis=2)
        # First upper bracket. At exact/degenerate endpoints no division is needed.
        right_index = np.argmax(totals >= target[:, None], axis=1)
        missing = ~np.any(totals >= target[:, None], axis=1)
        right_index[missing] = 2 * muscles - 1
        left_index = np.maximum(right_index - 1, 0)
        row_index = np.arange(rows.size)
        left = candidates[row_index, left_index]
        right = candidates[row_index, right_index]
        total_delta = right.sum(axis=1) - left.sum(axis=1)
        fraction = np.divide(target - left.sum(axis=1), total_delta,
                             out=np.zeros(rows.size), where=total_delta > 0)
        answer = np.clip(left + fraction[:, None] * (right - left), lo, hi)
        # Exact endpoints avoid cancellation in reference + multiplier * weight.
        # Mirror the scalar allocator's tolerance-sized endpoint treatment.
        near_lower = np.abs(target - low_total[rows]) <= feasibility_tolerance
        near_upper = ~near_lower & (np.abs(target - high_total[rows]) <= feasibility_tolerance)
        answer[near_lower] = lo[near_lower]
        answer[near_upper] = hi[near_upper]
        errors = answer.sum(axis=1) - target
        success = np.all(np.isfinite(answer), axis=1) & (np.abs(errors) <= feasibility_tolerance)
        statuses[rows] = np.where(success, "ok", "allocation_numerical_residual")
        allocated[rows[success]] = answer[success]
        # Public residual is always relative to the ORIGINAL request. The
        # numerical check above remains relative to the admitted target.
        residual[rows] = answer.sum(axis=1) - required[rows]
    return BatchedBoundedMomentQpResult(tuple(statuses), allocated, effective, residual, relaxed)


@dataclass(frozen=True)
class BatchedAffineRecruitmentMap:
    intercept: np.ndarray
    slope: np.ndarray
    weighted_force_intercept: np.ndarray
    weighted_force_slope: np.ndarray
    maximum_recruitment: np.ndarray

    def endpoint(self, recruitment):
        recruitment = np.asarray(recruitment, float)
        if (recruitment.shape != self.intercept.shape[:2] or not np.all(np.isfinite(recruitment))
                or np.any(recruitment < 0) or np.any(recruitment > self.maximum_recruitment)):
            raise ValueError("Recruitment must have shape (batch, muscles) within its PW bounds.")
        return self.intercept + self.slope * recruitment[:, :, None]


@dataclass(frozen=True)
class BatchedMomentRolloutResult(TotalMomentReferenceRolloutResult):
    total_lower_bounds: np.ndarray
    total_upper_bounds: np.ndarray
    original_total_moments: np.ndarray
    effective_total_targets: np.ndarray
    signed_moment_errors: np.ndarray
    relaxed_step_mask: np.ndarray
    envelope_domain_valid: np.ndarray
    relaxed_steps: int

    @property
    def tracking_mode(self):
        return "bounded_tracking" if self.relaxed_steps else "exact_with_numerical_tolerance"


def _physical(states):
    return (np.all(np.isfinite(states), axis=(1, 2)) & np.all(states[:, :, :2] >= 0, axis=(1, 2))
            & np.all(states[:, :, 2:] > 0, axis=(1, 2)))


class BatchedCompactMusclePredictor:
    def __init__(self, predictor):
        self.predictor = predictor
        self.intervals = predictor.intervals
        self.parameters = predictor.parameters
        self.muscle_count = predictor.muscle_count
        self.substeps = predictor.substeps

    def phase_map_many(self, states, interval_index):
        states = np.asarray(states, float)
        if states.ndim != 3 or states.shape[1:] != (self.muscle_count, 5) or not np.all(_physical(states)):
            raise ValueError("States require (batch, muscles, 5), nonnegative Cn/F and positive A/Tau1/Km.")
        p = self.predictor
        interval = self.intervals[interval_index]
        amplitude = np.asarray(interval.calcium_amplitudes)
        if np.any(amplitude < 0):
            raise ValueError("Calcium amplitudes must be nonnegative.")
        dt = interval.duration / self.substeps
        force0 = states[:, :, 1].copy()
        force1, integral0, integral1 = (np.zeros_like(force0) for _ in range(3))
        admissible_recruitment = np.broadcast_to(p.maximum_recruitment, force0.shape).copy()
        slow_decay = np.exp(-dt / p.tau_fat)
        integral_constant = -p.tau_fat * np.expm1(-dt / p.tau_fat)
        for j in range(self.substeps):
            time = (j + .5) * dt
            cn = np.exp(-time / p.tauc) * (states[:, :, 0] + amplitude * time / p.tauc)
            activation = cn / (states[:, :, 4] + cn)
            relaxation = states[:, :, 3] + p.tau2 * activation
            rate = p.gains[interval_index, j] / relaxation
            equilibrium1 = states[:, :, 2] * activation * relaxation
            force_decay = np.exp(-rate * dt)
            kernel = slow_decay * dt * exprel((1. / p.tau_fat - rate) * dt)
            integral0 = slow_decay * integral0 + kernel * force0
            integral1 = slow_decay * integral1 + kernel * force1 + equilibrium1 * (integral_constant - kernel)
            force0 *= force_decay
            force1 = force_decay * force1 - np.expm1(-rate * dt) * equilibrium1
            force_bound = np.divide(force0, -force1, out=np.full_like(force0, np.inf),
                                    where=force1 < 0.0)
            admissible_recruitment = np.minimum(admissible_recruitment, force_bound)
        cn_end = np.exp(-interval.duration / p.tauc) * (states[:, :, 0] + amplitude * interval.duration / p.tauc)
        decay = np.exp(-interval.duration / p.tau_fat)
        slow0 = p.rest + decay[:, None] * (states[:, :, 2:] - p.rest) + p.alpha * integral0[:, :, None]
        slow1 = p.alpha * integral1[:, :, None]
        return BatchedAffineRecruitmentMap(
            np.concatenate((cn_end[:, :, None], force0[:, :, None], slow0), axis=2),
            np.concatenate((np.zeros_like(force1)[:, :, None], force1[:, :, None], slow1), axis=2),
            integral0, integral1, admissible_recruitment)

    def rollout_many(self, initial_states, *, horizon_cycles, moment_tolerance=1e-8, tracking_band_nm=0.):
        if isinstance(horizon_cycles, bool) or int(horizon_cycles) != horizon_cycles or horizon_cycles < 1:
            raise ValueError("horizon_cycles must be a positive integer.")
        if not np.isfinite(moment_tolerance) or moment_tolerance <= 0:
            raise ValueError("moment_tolerance must be finite and positive.")
        if not np.isfinite(tracking_band_nm) or tracking_band_nm < 0:
            raise ValueError("tracking_band_nm must be finite and nonnegative.")
        current = np.asarray(initial_states, float).copy()
        if current.ndim != 3 or not current.shape[0] or current.shape[1:] != (self.muscle_count, 5):
            raise ValueError("initial_states must have nonempty shape (batch, muscles, 5).")
        batch, count, horizon = current.shape[0], len(self.intervals), int(horizon_cycles)
        pws = np.full((batch, horizon, self.muscle_count, count), np.nan)
        allocated, achieved = np.full_like(pws, np.nan), np.full_like(pws, np.nan)
        history = np.full((batch, horizon * count + 1, self.muscle_count, 5), np.nan)
        lower, upper, original, effective, errors = (np.full((batch, horizon, count), np.nan) for _ in range(5))
        relaxed, envelope_valid = (np.zeros((batch, horizon, count), bool) for _ in range(2))
        completed = np.zeros(batch, int)
        failures = [None] * batch
        alive = _physical(current)
        history[alive, 0] = current[alive]
        for row in np.flatnonzero(~alive):
            failures[row] = {"cycle_index": 0, "interval_index": 0, "status": "initial_state_outside_domain"}

        def fail(rows, cycle, phase, status):
            for row in rows:
                failures[row] = {"cycle_index": cycle, "interval_index": phase, "status": status}
            alive[rows] = False

        for step in range(horizon * count):
            rows = np.flatnonzero(alive)
            if not rows.size:
                break
            cycle, k = divmod(step, count)
            interval = self.intervals[k]
            transition = self.phase_map_many(current[rows], k)
            coefficients = np.asarray(interval.moment_coefficients)
            moment0 = coefficients * transition.intercept[:, :, 1]
            slope = coefficients * transition.slope[:, :, 1]
            moment1 = moment0 + slope * transition.maximum_recruitment
            lo, hi = np.minimum(moment0, moment1), np.maximum(moment0, moment1)
            required = float(np.sum(interval.target_moments))
            lower[rows, cycle, k], upper[rows, cycle, k] = lo.sum(axis=1), hi.sum(axis=1)
            original[rows, cycle, k] = required
            maximal_endpoint = transition.intercept + transition.slope * transition.maximum_recruitment[:, :, None]
            envelope_valid[rows, cycle, k] = _physical(transition.intercept) & _physical(maximal_endpoint)
            allocation = solve_bounded_moment_qp_many(
                interval.target_moments, required_total_moment=required, lower_bounds=lo, upper_bounds=hi,
                feasibility_tolerance=moment_tolerance, tracking_band_nm=tracking_band_nm)
            effective[rows, cycle, k] = allocation.effective_total_targets
            for index, status in enumerate(allocation.statuses):
                if status != "ok":
                    fail([rows[index]], cycle, k, status)
            good = np.asarray(allocation.statuses) == "ok"
            if not np.any(good):
                continue
            selected = rows[good]
            recruitment = np.divide(allocation.allocated_moments[good] - moment0[good], slope[good],
                                    out=np.zeros_like(slope[good]), where=np.abs(slope[good]) > 1e-14)
            in_bounds = np.all((recruitment >= -1e-12) & (recruitment <= transition.maximum_recruitment[good] + 1e-12), axis=1)
            fail(selected[~in_bounds], cycle, k, "recruitment_inversion_outside_bounds")
            recruitment = np.clip(recruitment, 0., transition.maximum_recruitment[good])
            next_states = transition.intercept[good] + transition.slope[good] * recruitment[:, :, None]
            physical = _physical(next_states)
            fail(selected[in_bounds & ~physical], cycle, k, "predicted_state_outside_domain")
            predicted = coefficients * next_states[:, :, 1]
            numerical_error = predicted.sum(axis=1) - allocation.effective_total_targets[good]
            accurate = np.abs(numerical_error) <= moment_tolerance * 1.01
            fail(selected[in_bounds & physical & ~accurate], cycle, k, "moment_inversion_residual")
            success = in_bounds & physical & accurate
            selected = selected[success]
            p = self.predictor
            pws[selected, cycle, :, k] = np.minimum(p.pulse_width_max, p.pd0 - p.pdt * np.log1p(-recruitment[success]))
            allocated[selected, cycle, :, k] = allocation.allocated_moments[good][success]
            achieved[selected, cycle, :, k] = predicted[success]
            errors[selected, cycle, k] = predicted[success].sum(axis=1) - required
            relaxed[selected, cycle, k] = allocation.relaxed[good][success]
            current[selected] = next_states[success]
            completed[selected] += 1
            history[selected, step + 1] = current[selected]
        return tuple(BatchedMomentRolloutResult(
            "complete" if failures[row] is None else "infeasible", horizon, completed[row] // count,
            int(completed[row]), pws[row], allocated[row], achieved[row], history[row], failures[row], 0,
            lower[row], upper[row], original[row], effective[row], errors[row], relaxed[row],
            envelope_valid[row], int(relaxed[row].sum())) for row in range(batch))
