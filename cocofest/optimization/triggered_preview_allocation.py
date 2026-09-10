"""Use the strict two-phase preview only near a next-phase moment boundary.

The inexpensive decision first applies the exact greedy candidate in the
compact model. It measures both signed distances from the ORIGINAL next
target to its attainable moment envelope. A fixed, declared threshold decides
whether to call the existing two-phase QP. The ordinary greedy path is not a
fallback, and a rejected preview never silently switches back to greedy.
"""

from dataclasses import dataclass, fields
from numbers import Real
from time import perf_counter

import numpy as np

from .preview_muscle_allocation import (
    PreviewAllocationPlan, PreviewMomentRolloutResult, PreviewMuscleAllocation, _physical,
)


def _finite_scalar(value, name, *, strictly_positive):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real scalar, not a boolean or string.")
    value = float(value)
    if not np.isfinite(value) or (value <= 0 if strictly_positive else value < 0):
        raise ValueError(f"{name} must be finite and {'positive' if strictly_positive else 'nonnegative'}.")
    return value


@dataclass(frozen=True)
class TriggeredPreviewRolloutResult(PreviewMomentRolloutResult):
    screenings: int
    triggers: int
    greedy_fast_path_steps: int
    screen_failures: int
    screening_time_s: float
    triggered_preview_time_s: float
    total_time_s: float
    allocation_policy_metadata: dict


class TriggeredPreviewMuscleAllocation(PreviewMuscleAllocation):
    """Greedy plus a declared signed-margin trigger for the existing K2 QP.

    ``moment_scale`` is a fixed positive task scale, normally max(abs(original
    cycle target)); it must not be the signed target or a candidate-dependent
    envelope width. ``trigger_margin_fraction`` is fixed at construction. The
    experimental comparisons use 0 and 0.01, without selecting a threshold
    from observed outcomes. ``screening_depth=1`` means one following phase.

    Timing named ``preview_elapsed_s`` includes building, solving and auditing
    the QP, not only SLSQP. The total rollout time also includes the original
    phase applications and reporting. No public RHO solver is changed.
    """

    def __init__(self, predictor, *, moment_scale, trigger_margin_fraction=.01, **preview_options):
        depth = preview_options.pop("preview_phases", 2)
        fallback = preview_options.pop("fallback_to_greedy", False)
        if isinstance(depth, (bool, np.bool_)) or not isinstance(depth, (int, np.integer)) or depth != 2:
            raise ValueError("Triggered preview requires preview_phases=2.")
        if not isinstance(fallback, (bool, np.bool_)) or fallback:
            raise ValueError("Triggered preview requires fallback_to_greedy=False.")
        self._moment_scale = _finite_scalar(moment_scale, "moment_scale", strictly_positive=True)
        self._trigger_margin_fraction = _finite_scalar(trigger_margin_fraction, "trigger_margin_fraction",
                                                        strictly_positive=False)
        if not np.isfinite(self._moment_scale * self._trigger_margin_fraction):
            raise ValueError("The trigger threshold must be finite.")
        super().__init__(predictor, preview_phases=2, fallback_to_greedy=False, **preview_options)

    @property
    def moment_scale(self):
        return self._moment_scale

    @property
    def trigger_margin_fraction(self):
        return self._trigger_margin_fraction

    @property
    def screening_depth(self):
        return 1

    @property
    def allocation_policy_metadata(self):
        """Fresh JSON-compatible value snapshot, including all policy settings."""
        return {
            "kind": "triggered_preview_signed_next_margin_v1",
            "preview_phases": self.preview_phases,
            "moment_tolerance": self.moment_tolerance,
            "max_iterations": self.max_iterations,
            "fallback_to_greedy": self.fallback_to_greedy,
            "optimality_tolerance": self.optimality_tolerance,
            "recruitment_regularization": self.recruitment_regularization,
            "max_solve_time_s": self.max_solve_time_s,
            "trigger_margin_fraction": self.trigger_margin_fraction,
            "moment_scale": self.moment_scale,
            "screening_depth": self.screening_depth,
        }

    def plan(self, states, interval_index, *, depth=None):
        depth = 2 if depth is None else depth
        if isinstance(depth, (bool, np.bool_)) or not isinstance(depth, (int, np.integer)) or depth not in (1, 2):
            raise ValueError("depth must be 1 or 2; the declared horizon must truncate screening.")
        if self.preview_phases != 2 or self.fallback_to_greedy:
            raise ValueError("Triggered preview requires unchanged depth=2 and no fallback.")
        start = perf_counter()
        threshold = self.trigger_margin_fraction * self.moment_scale
        diagnostics = {
            "preview_depth": int(depth), "qp_solved": False, "optimizer_success": False,
            "optimizer_status": None, "optimizer_iterations": 0, "fallback_reason": None,
            "nominal_projected_steps": 0, "screened": False, "triggered": False,
            "screen_failed": False, "screen_reason": None, "allocation_mode": None,
            "next_phase_index": None, "next_required_total_moment_nm": None,
            "next_lower_total_moment_nm": None, "next_upper_total_moment_nm": None,
            "next_lower_signed_margin_nm": None, "next_upper_signed_margin_nm": None,
            "next_signed_margin_nm": None, "next_normalized_signed_margin": None,
            "trigger_threshold_nm": threshold, "screening_elapsed_s": 0., "preview_elapsed_s": 0.,
        }

        def finish(accepted, recruitment, reason, *, problem=None):
            elapsed = perf_counter() - start
            diagnostics.update(accepted=accepted, reason=reason, elapsed_s=elapsed, plan_total_elapsed_s=elapsed)
            return PreviewAllocationPlan(accepted, recruitment, problem, diagnostics)

        try:
            state = np.asarray(states, float)
            if state.shape != (self.predictor.muscle_count, 5) or not _physical(state):
                raise ValueError("initial_state_outside_domain")
            _, greedy_recruitment, greedy_endpoint, _, _, greedy_error = self._greedy_step(state, interval_index)
        except (ValueError, FloatingPointError, OverflowError) as error:
            diagnostics["screen_reason"] = "current_greedy_failed"
            diagnostics["screening_elapsed_s"] = perf_counter() - start
            return finish(False, None, f"current_greedy_failed: {error}")
        diagnostics.update(equality_residual_max_nm=abs(greedy_error), bound_violation_max=0.)
        if depth == 1:
            # No next interval is accessed, even if the mechanics repeat.
            diagnostics.update(screen_reason="declared_horizon_end", allocation_mode="greedy",
                               screening_elapsed_s=perf_counter() - start)
            return finish(True, greedy_recruitment, "terminal_exact_greedy")

        diagnostics["screened"] = True
        next_phase = (interval_index + 1) % len(self.predictor.intervals)
        diagnostics["next_phase_index"] = next_phase
        try:
            interval = self.predictor.intervals[next_phase]
            transition = self.predictor.phase_map(greedy_endpoint, next_phase)
            maximal_endpoint = transition.endpoint(transition.maximum_recruitment)
            if not _physical(transition.intercept) or not _physical(maximal_endpoint):
                raise ValueError("next_envelope_endpoint_outside_domain")
            coefficients = np.asarray(interval.moment_coefficients, float)
            minimum_pw_moments = coefficients * transition.intercept[:, 1]
            maximum_pw_moments = coefficients * maximal_endpoint[:, 1]
            required = float(np.sum(interval.target_moments))
            lower = float(np.minimum(minimum_pw_moments, maximum_pw_moments).sum())
            upper = float(np.maximum(minimum_pw_moments, maximum_pw_moments).sum())
            lower_margin, upper_margin = required - lower, upper - required
            margin = min(lower_margin, upper_margin)
            normalized_margin = margin / self.moment_scale
            if not np.all(np.isfinite([required, lower, upper, lower_margin, upper_margin, margin, normalized_margin])):
                raise ValueError("nonfinite_next_signed_envelope")
            diagnostics.update(next_required_total_moment_nm=required, next_lower_total_moment_nm=lower,
                               next_upper_total_moment_nm=upper, next_lower_signed_margin_nm=lower_margin,
                               next_upper_signed_margin_nm=upper_margin, next_signed_margin_nm=margin,
                               next_normalized_signed_margin=normalized_margin)
        except (ValueError, FloatingPointError, OverflowError) as error:
            diagnostics.update(screen_failed=True, screen_reason="invalid_next_envelope",
                               screening_elapsed_s=perf_counter() - start)
            return finish(False, None, f"screen_failed: {error}")

        diagnostics["screening_elapsed_s"] = perf_counter() - start
        if margin > threshold:
            diagnostics.update(screen_reason="next_margin_above_threshold", allocation_mode="greedy")
            return finish(True, greedy_recruitment, "safe_exact_greedy")

        diagnostics.update(triggered=True, screen_reason="next_margin_at_or_below_threshold", allocation_mode="preview")
        preview_start = perf_counter()
        preview = super().plan(state, interval_index, depth=2)
        screening_fields = dict(diagnostics)
        diagnostics.update(preview.diagnostics)
        # Keep the original greedy candidate screen, while retaining QP audit
        # residuals/nominal projection counts from the delegated preview.
        for key in ("screened", "triggered", "screen_failed", "screen_reason", "allocation_mode",
                    "next_phase_index", "next_required_total_moment_nm", "next_lower_total_moment_nm",
                    "next_upper_total_moment_nm", "next_lower_signed_margin_nm", "next_upper_signed_margin_nm",
                    "next_signed_margin_nm", "next_normalized_signed_margin", "trigger_threshold_nm",
                    "screening_elapsed_s"):
            diagnostics[key] = screening_fields[key]
        diagnostics["preview_elapsed_s"] = perf_counter() - preview_start
        return finish(preview.accepted, preview.recruitment, preview.diagnostics["reason"], problem=preview.problem)

    def rollout(self, initial_states, *, horizon_cycles, moment_tolerance=None):
        if self.preview_phases != 2 or self.fallback_to_greedy:
            raise ValueError("Triggered preview requires unchanged depth=2 and no fallback.")
        started = perf_counter()
        snapshot = self.allocation_policy_metadata
        result = super().rollout(initial_states, horizon_cycles=horizon_cycles, moment_tolerance=moment_tolerance)
        if self.allocation_policy_metadata != snapshot:
            raise ValueError("Hybrid policy settings changed during rollout.")
        records = result.diagnostics
        return TriggeredPreviewRolloutResult(
            **{field.name: getattr(result, field.name) for field in fields(PreviewMomentRolloutResult)},
            screenings=sum(record["screened"] for record in records),
            triggers=sum(record["triggered"] for record in records),
            greedy_fast_path_steps=sum(record["allocation_mode"] == "greedy" for record in records[:result.completed_intervals]),
            screen_failures=sum(record["screen_failed"] for record in records),
            screening_time_s=sum(record["screening_elapsed_s"] for record in records),
            triggered_preview_time_s=sum(record["preview_elapsed_s"] for record in records),
            total_time_s=perf_counter() - started, allocation_policy_metadata=snapshot)
