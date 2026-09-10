"""Experimental local value of a short-preview muscle allocation policy.

The allocation QPs are numerical oracle work outside the differentiable RHO
graph. Only an independently audited polynomial may subsequently be bound to
the NLP. This adapter does not activate any public RHO campaign option.
"""

from .local_endurance_value import CompactEnduranceValueOracle


def _policy_context(policy):
    return {
        "kind": "frozen_slow_force_coupled_preview_v1",
        "preview_phases": policy.preview_phases,
        "moment_tolerance": policy.moment_tolerance,
        "max_iterations": policy.max_iterations,
        "fallback_to_greedy": policy.fallback_to_greedy,
        "optimality_tolerance": policy.optimality_tolerance,
        "recruitment_regularization": policy.recruitment_regularization,
        "max_solve_time_s": policy.max_solve_time_s,
    }


class _PreviewPredictorView:
    """Use the preview trajectory with the existing one-phase reserve audit."""

    def __init__(self, policy):
        self.policy = policy
        self._predictor = policy.predictor
        self.context = _policy_context(policy)
        self.parameters = policy.predictor.parameters
        self.intervals = policy.predictor.intervals
        self.gains = policy.predictor.gains

    def phase_map(self, states, interval_index):
        return self._predictor.phase_map(states, interval_index)

    def rollout(self, states, *, horizon_cycles, moment_tolerance):
        if self.policy.predictor is not self._predictor or _policy_context(self.policy) != self.context:
            raise ValueError("Preview policy settings changed after oracle construction.")
        if moment_tolerance != self.context["moment_tolerance"]:
            raise ValueError("Oracle and preview policy must use the same moment tolerance.")
        return self.policy.rollout(states, horizon_cycles=horizon_cycles)


class PreviewEnduranceValueOracle(CompactEnduranceValueOracle):
    """Score the preview policy with unchanged signed-envelope value semantics.

    All horizon phases must complete before a value is returned. The margins
    are recomputed using the original compact phase map at each *applied*
    boundary, not the frozen future QP envelopes. There is no tracking band.
    This backend is sequential; the speedups of the batched greedy oracle do
    not apply to it. No endurance or clinical performance claim is implied.
    """

    def __init__(self, policy, coordinates, **options):
        options.setdefault("moment_tolerance", policy.moment_tolerance)
        if options["moment_tolerance"] != policy.moment_tolerance:
            raise ValueError("Oracle and preview policy must use the same moment tolerance.")
        super().__init__(_PreviewPredictorView(policy), coordinates, **options)

    @property
    def value_context_metadata(self):
        return {
            **super().value_context_metadata,
            "allocation_policy": dict(self.predictor.context),
            "evaluation_backend": "sequential_preview_qp",
        }
