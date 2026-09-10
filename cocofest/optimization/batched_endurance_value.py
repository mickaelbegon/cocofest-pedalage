"""Batched local policy values with optional, explicitly bounded moment error.

No QP or rollout is introduced in the differentiable NLP. A nonzero band
changes the predicted task; it is neither a clinical tolerance nor a proof of
endurance improvement. All margins/errors retain the original moment target.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .batched_compact_muscle_prediction import BatchedCompactMusclePredictor
from .local_endurance_value import CompactEnduranceValueOracle, OracleEvaluation


@dataclass(frozen=True)
class BatchedOracleEvaluation(OracleEvaluation):
    margin_cost: float | None = None
    tracking_cost: float | None = None
    tracking_mode: str = "not_completed"
    max_abs_moment_error_nm: float | None = None
    relaxed_steps: int = 0
    signed_error_quadrature_nm_s: float | None = None
    absolute_error_quadrature_nm_s: float | None = None


class BatchedEnduranceValueOracle(CompactEnduranceValueOracle):
    """Same exact-task value as the scalar oracle when tracking_band_nm=0.

    With a band, only unavoidable out-of-envelope errors are admitted, never
    voluntary changes to an attainable original target. The additional cost is
    tracking_penalty_weight * time-averaged (error/moment_scale)^2. The scale
    and weight do not depend on the band, and the signed reserve is still
    computed against the original target. Weight1 may be numerically small;
    this penalty alone is not a guarantee against drift.
    """

    def __init__(self, predictor, coordinates, *, tracking_band_nm=0.,
                 tracking_penalty_weight=1., **kwargs):
        super().__init__(predictor, coordinates, **kwargs)
        if not np.isfinite(tracking_band_nm) or tracking_band_nm < 0:
            raise ValueError("tracking_band_nm must be finite and nonnegative.")
        if not np.isfinite(tracking_penalty_weight) or tracking_penalty_weight < 0:
            raise ValueError("tracking_penalty_weight must be finite and nonnegative.")
        self.tracking_band_nm = float(tracking_band_nm)
        self.tracking_penalty_weight = float(tracking_penalty_weight)
        self.batch_predictor = BatchedCompactMusclePredictor(predictor)

    @property
    def value_context_metadata(self):
        return {**super().value_context_metadata, "tracking_band_nm": self.tracking_band_nm,
                "tracking_penalty_weight": self.tracking_penalty_weight,
                "tracking_cost_definition": "duration_mean_squared_original_error_over_task_scale_v1"}

    def evaluate(self, xi):
        return self.evaluate_many(np.asarray(xi)[None, :])[0]

    def evaluate_many(self, points):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != self.coordinates.anchor.size:
            raise ValueError("points must have shape (candidates, 2M).")
        outcomes = [None] * len(points)
        rows, states = [], []
        for index, point in enumerate(points):
            try:
                states.append(self.coordinates.decode(point))
                rows.append(index)
            except (ValueError, FloatingPointError, OverflowError) as error:
                outcomes[index] = BatchedOracleEvaluation("domain_invalid", None, None, None, 0, str(error))
        if not rows:
            return tuple(outcomes)
        rollouts = self.batch_predictor.rollout_many(
            np.asarray(states), horizon_cycles=self.horizon_cycles,
            moment_tolerance=self.moment_tolerance, tracking_band_nm=self.tracking_band_nm,
        )
        durations = np.tile([it.duration for it in self.predictor.intervals], self.horizon_cycles)
        for index, rollout in zip(rows, rollouts):
            if rollout.status != "complete":
                outcomes[index] = BatchedOracleEvaluation(
                    "policy_failed", None, None, None, rollout.completed_intervals, str(rollout.first_failure),
                    relaxed_steps=rollout.relaxed_steps,
                )
                continue
            if not np.all(rollout.envelope_domain_valid):
                outcomes[index] = BatchedOracleEvaluation(
                    "envelope_domain_invalid", None, None, None, rollout.completed_intervals,
                    "An attainable-envelope endpoint is outside the predictor domain.",
                    relaxed_steps=rollout.relaxed_steps,
                )
                continue
            original = rollout.original_total_moments.ravel()
            margins = np.column_stack((original - rollout.total_lower_bounds.ravel(),
                                       rollout.total_upper_bounds.ravel() - original)).ravel() / self.moment_scale
            errors = rollout.signed_moment_errors.ravel()
            if not np.all(np.isfinite(margins)) or not np.all(np.isfinite(errors)):
                outcomes[index] = BatchedOracleEvaluation("domain_invalid", None, None, None,
                                                          rollout.completed_intervals, "Nonfinite margins or errors.")
                continue
            soft_minimum = -self.softmin_temperature * logsumexp(-margins / self.softmin_temperature)
            margin_cost = self.penalty_temperature * np.logaddexp(
                0., (self.margin_target - soft_minimum) / self.penalty_temperature)
            tracking_cost = (self.tracking_penalty_weight * np.dot(durations, (errors/self.moment_scale)**2)
                             / durations.sum() if self.tracking_band_nm > 0 else 0.)
            value = margin_cost + tracking_cost
            if not np.isfinite(value):
                outcomes[index] = BatchedOracleEvaluation("domain_invalid", None, None, None,
                                                          rollout.completed_intervals, "Nonfinite value.")
                continue
            outcomes[index] = BatchedOracleEvaluation(
                "complete", float(value), float(margins.min()), float(soft_minimum), rollout.completed_intervals,
                margin_cost=float(margin_cost), tracking_cost=float(tracking_cost),
                tracking_mode="bounded_tracking" if rollout.relaxed_steps else "exact_with_numerical_tolerance",
                max_abs_moment_error_nm=float(np.max(np.abs(errors))), relaxed_steps=rollout.relaxed_steps,
                # Endpoint rectangle quadratures, not continuous tracking integrals or work.
                signed_error_quadrature_nm_s=float(durations @ errors),
                absolute_error_quadrature_nm_s=float(durations @ np.abs(errors)),
            )
        return tuple(outcomes)
