"""Problem-architecture vocabulary shared by the desktop GUI and its tests.

The three entries deliberately describe *architectures*, rather than options of
one interchangeable OCP.  In particular, the independent-arm coordinator is
not a surrogate for the bilateral mechanical model: it has two unilateral
isokinetic NLPs and no shared crank state.
"""
from __future__ import annotations

from dataclasses import replace

from .config import SimulationConfig


UNILATERAL = "unilateral"
BILATERAL_COMBINED = "bilateral_combined"
INDEPENDENT_ARMS = "independent_arms"

PROBLEM_TYPE_LABELS = {
    UNILATERAL: "1 bras — un OCP",
    BILATERAL_COMBINED: "2 bras combinés — une manivelle, un OCP",
    INDEPENDENT_ARMS: "2 bras indépendants — deux OCPs isocinétiques",
}
PROBLEM_TYPE_BY_LABEL = {label: key for key, label in PROBLEM_TYPE_LABELS.items()}

PROBLEM_TYPE_DESCRIPTIONS = {
    UNILATERAL: (
        "Un bras, un état mécanique et quatre muscles. Dynamique ou isocinétique "
        "selon la formulation sélectionnée."
    ),
    BILATERAL_COMBINED: (
        "Deux chaînes de bras et huit muscles sur une même manivelle : angle, vitesse "
        "et bilan mécanique sont communs. Cette voie est actuellement dynamique."
    ),
    INDEPENDENT_ARMS: (
        "Deux OCPs unilatéraux isocinétiques séparés : états Ding, commandes et couples "
        "ne sont pas partagés; seule une allocation supervisée peut les coordonner."
    ),
}


def main_problem_type(config: SimulationConfig) -> str:
    """Return the architecture represented by a main-form configuration."""
    return BILATERAL_COMBINED if config.bilateral_reduced else UNILATERAL


def set_main_problem_type(config: SimulationConfig, problem_type: str) -> SimulationConfig:
    """Select a main-form architecture without modifying unrelated parameters.

    Independent arms have a separate request schema and launcher, so silently
    converting a :class:`SimulationConfig` to that schema would be scientifically
    misleading.  The GUI merely navigates to its dedicated form for that choice.
    """
    if problem_type == UNILATERAL:
        return replace(config, bilateral_reduced=False)
    if problem_type == BILATERAL_COMBINED:
        return replace(config, bilateral_reduced=True)
    if problem_type == INDEPENDENT_ARMS:
        raise ValueError("Les deux bras indépendants utilisent leur propre requête.")
    raise ValueError(f"Type de problème inconnu : {problem_type!r}")
