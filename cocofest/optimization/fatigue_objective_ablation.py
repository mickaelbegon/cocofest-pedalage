"""Opt-in fatigue costs for matched, one-cycle objective ablations.

The legacy objective is deliberately registered elsewhere without changes.
For a RHO whose initial state is fixed, terminal normalized fatigue and its
increment since the start differ by a constant.  The linear terminal cost
therefore tests the marginal cost of *new* fatigue without introducing an
initial-state parameter or changing the compiled NLP within a branch.
"""

from __future__ import annotations

from casadi import sum1, vertcat

from .parametric_fatigue_weights import FATIGUE_WEIGHT_PARAMETER_KEY


FATIGUE_OBJECTIVE_VARIANTS = frozenset(
    {"legacy", "integral_quadratic", "integral_linear", "terminal_quadratic", "terminal_linear"}
)


def validate_fatigue_objective_variant(
    variant: str, *, minimize_fatigue: bool, duration_s: float
) -> str:
    """Reject ambiguous or dimensionally invalid ablation requests."""
    import math

    if not isinstance(variant, str) or variant not in FATIGUE_OBJECTIVE_VARIANTS:
        raise ValueError(
            "fatigue_objective_variant must be one of "
            + ", ".join(sorted(FATIGUE_OBJECTIVE_VARIANTS))
        )
    if variant != "legacy" and not minimize_fatigue:
        raise ValueError("An explicit fatigue objective variant requires minimize_fatigue=True.")
    if not isinstance(duration_s, (int, float)) or not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("fatigue objective duration must be finite and positive.")
    return variant


def _normalized_fatigue(controller):
    muscles = controller.model.muscles_dynamics_model
    return vertcat(
        *(
            1 - controller.states[f"A_{muscle.muscle_name}"].cx / muscle.a_scale
            for muscle in muscles
        )
    )


def terminal_linear_fatigue(controller):
    """Sum normalized terminal deficits; initial deficits are fixed constants."""
    return sum1(_normalized_fatigue(controller))


def terminal_parameterized_linear_fatigue(controller):
    """Weighted terminal deficit, with weights held in fixed NLP parameters."""
    fatigue = _normalized_fatigue(controller)
    weights = controller.parameters[FATIGUE_WEIGHT_PARAMETER_KEY].cx
    if int(weights.numel()) != int(fatigue.numel()):
        raise ValueError("Fatigue weight parameter/model dimension mismatch")
    return sum1(weights * fatigue)
