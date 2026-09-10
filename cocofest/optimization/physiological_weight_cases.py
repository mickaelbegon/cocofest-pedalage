"""Deterministic parameter cases for physiological-weight sensitivity studies.

This module only declares muscle-parameter inputs.  It does not calculate a
weight, run a simulation, or choose a target weighting.  The published-code
and current-repository parameter dictionaries are intentionally separate and
explicitly unreconciled.  Every sensitivity case is rebuilt from its named
baseline, so results from one case can never leak into the next case.

PCSA/fibre-type coupling is omitted: the repository comments and published
parameter source do not currently provide a reconciled protocol for applying
such a coupling across both baselines.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


MUSCLE_NAMES = ("Delt_ant", "Delt_post", "Biceps", "Triceps")
SENSITIVITY_FIELDS = ("alpha_a", "tau_fat", "a_scale", "Fmax")
OFAT_FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)
TARGETED_CROSS_FACTORS = (0.5, 2.0)
MAX_CASES_PER_PANEL = 100

PUBLISHED_CODE_BASELINE_ID = "published_code"
REPOSITORY_CURRENT_BASELINE_ID = "repository_current"
PUBLISHED_CODE_COMMIT = "31e064f4d80741c8b5444e6ce4d91bfe4eb78c3d"
GENERATION_PROTOCOL_ID = "independent-ofat-v1"


def _finite(value: Any, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _positive(value: Any, *, name: str) -> float:
    number = _finite(value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be strictly positive.")
    return number


@dataclass(frozen=True)
class MusclePhysiologyParameters:
    """Four independent inputs used by the first sensitivity panel."""

    muscle_name: str
    Fmax: float
    a_scale: float
    alpha_a: float
    tau_fat: float

    def __post_init__(self) -> None:
        muscle_name = str(self.muscle_name)
        if muscle_name not in MUSCLE_NAMES:
            raise ValueError(f"Unknown muscle_name {muscle_name!r}.")
        alpha_a = _finite(self.alpha_a, name="alpha_a")
        if alpha_a >= 0.0:
            raise ValueError("alpha_a must remain strictly negative.")
        object.__setattr__(self, "muscle_name", muscle_name)
        object.__setattr__(self, "Fmax", _positive(self.Fmax, name="Fmax"))
        object.__setattr__(self, "a_scale", _positive(self.a_scale, name="a_scale"))
        object.__setattr__(self, "alpha_a", alpha_a)
        object.__setattr__(self, "tau_fat", _positive(self.tau_fat, name="tau_fat"))

    def as_parameter_dict(self) -> dict[str, float]:
        """Return fresh mutable data for a downstream model constructor."""

        return {
            "Fmax": self.Fmax,
            "a_scale": self.a_scale,
            "alpha_a": self.alpha_a,
            "tau_fat": self.tau_fat,
        }


@dataclass(frozen=True)
class PhysiologicalBaseline:
    baseline_id: str
    name: str
    provenance_id: str
    source_path: str
    reconciliation_status: str
    parameters: tuple[MusclePhysiologyParameters, ...]

    def __post_init__(self) -> None:
        if not self.baseline_id or not self.name or not self.provenance_id or not self.source_path:
            raise ValueError("Baseline identifiers and source_path must not be empty.")
        if self.reconciliation_status != "not_reconciled":
            raise ValueError("The two nominal sources must remain explicitly not_reconciled.")
        names = tuple(parameter.muscle_name for parameter in self.parameters)
        if names != MUSCLE_NAMES:
            raise ValueError(f"Baseline parameters must follow muscle order {MUSCLE_NAMES}.")

    def as_parameter_dict(self) -> dict[str, dict[str, float]]:
        return {
            parameter.muscle_name: parameter.as_parameter_dict()
            for parameter in self.parameters
        }


@dataclass(frozen=True)
class ParameterFactor:
    muscle_name: str
    parameter_name: str
    factor: float
    operation: str = "multiply_baseline_value"

    def __post_init__(self) -> None:
        if self.muscle_name not in MUSCLE_NAMES:
            raise ValueError(f"Unknown muscle_name {self.muscle_name!r}.")
        if self.parameter_name not in SENSITIVITY_FIELDS:
            raise ValueError(f"Unknown parameter_name {self.parameter_name!r}.")
        object.__setattr__(self, "factor", _positive(self.factor, name="factor"))
        if self.operation != "multiply_baseline_value":
            raise ValueError("Only multiply_baseline_value is supported.")


@dataclass(frozen=True)
class PhysiologicalParameterCase:
    case_id: str
    name: str
    baseline_id: str
    provenance_id: str
    generation_protocol_id: str
    control_family: str
    factor: float | None
    factors: tuple[ParameterFactor, ...]
    changed_fields: tuple[str, ...]
    parameters: tuple[MusclePhysiologyParameters, ...]
    weight_recalculation_required: bool = True
    inherited_weights: bool = False
    clinical_interpretation: str = "none_parameter_perturbations_are_abstract"

    def __post_init__(self) -> None:
        if not self.case_id or not self.name or not self.baseline_id or not self.provenance_id:
            raise ValueError("Case identifiers must not be empty.")
        if self.generation_protocol_id != GENERATION_PROTOCOL_ID:
            raise ValueError("Unknown generation protocol.")
        if self.control_family not in {
            "nominal",
            "one_factor_at_a_time",
            "targeted_fatigability_cross",
        }:
            raise ValueError(f"Unknown control_family {self.control_family!r}.")
        if self.factor is not None:
            object.__setattr__(self, "factor", _positive(self.factor, name="factor"))
        if not self.weight_recalculation_required or self.inherited_weights:
            raise ValueError("Every case must request fresh weight recalculation without inheritance.")
        if self.clinical_interpretation != "none_parameter_perturbations_are_abstract":
            raise ValueError("Sensitivity cases must not be assigned a clinical interpretation.")
        names = tuple(parameter.muscle_name for parameter in self.parameters)
        if names != MUSCLE_NAMES:
            raise ValueError(f"Case parameters must follow muscle order {MUSCLE_NAMES}.")
        declared_fields = tuple(
            f"{factor.muscle_name}.{factor.parameter_name}" for factor in self.factors
        )
        if declared_fields != self.changed_fields:
            raise ValueError("changed_fields must exactly match factors in declaration order.")
        if self.control_family == "nominal" and (self.factors or self.changed_fields):
            raise ValueError("The nominal case cannot declare changed fields.")
        if self.control_family == "one_factor_at_a_time" and len(self.factors) != 1:
            raise ValueError("An OFAT case must contain exactly one factor.")
        if self.control_family == "targeted_fatigability_cross" and len(self.factors) != 2:
            raise ValueError("A targeted cross case must contain exactly two factors.")

    def as_parameter_dict(self) -> dict[str, dict[str, float]]:
        return {
            parameter.muscle_name: parameter.as_parameter_dict()
            for parameter in self.parameters
        }


def _parameters(values: dict[str, dict[str, float]]) -> tuple[MusclePhysiologyParameters, ...]:
    return tuple(
        MusclePhysiologyParameters(muscle_name=muscle_name, **values[muscle_name])
        for muscle_name in MUSCLE_NAMES
    )


_PUBLISHED_CODE = PhysiologicalBaseline(
    baseline_id=PUBLISHED_CODE_BASELINE_ID,
    name="Published supplementary-code parameter dictionary",
    provenance_id=f"published-supplement-code@{PUBLISHED_CODE_COMMIT}:PARAMETERS",
    source_path=".cache/article-weight-source/physiological_weight_calculation.py",
    reconciliation_status="not_reconciled",
    parameters=_parameters(
        {
            "Delt_ant": {"Fmax": 48.0, "a_scale": 1148.6, "alpha_a": -1.4e-1, "tau_fat": 445.5},
            "Delt_post": {"Fmax": 51.0, "a_scale": 1234.5, "alpha_a": -1.1e-1, "tau_fat": 342.7},
            "Biceps": {"Fmax": 149.0, "a_scale": 3314.7, "alpha_a": -5.6e-2, "tau_fat": 179.6},
            "Triceps": {"Fmax": 262.0, "a_scale": 4915.5, "alpha_a": -3.4e-2, "tau_fat": 109.1},
        }
    ),
)

_REPOSITORY_CURRENT = PhysiologicalBaseline(
    baseline_id=REPOSITORY_CURRENT_BASELINE_ID,
    name="Current repository set_fes_model parameter dictionary",
    provenance_id="repository-current-worktree:set_fes_model.parameter_dict:2026-09-10",
    source_path="examples/fes_multibody/cycling/cycling_pulse_width_mhe.py",
    reconciliation_status="not_reconciled",
    parameters=_parameters(
        {
            "Delt_ant": {"Fmax": 48.0, "a_scale": 1148.6, "alpha_a": -1.4, "tau_fat": 445.5},
            "Delt_post": {"Fmax": 51.0, "a_scale": 1234.5, "alpha_a": -1.1, "tau_fat": 342.7},
            "Biceps": {"Fmax": 149.0, "a_scale": 3314.7, "alpha_a": -0.56, "tau_fat": 179.6},
            "Triceps": {"Fmax": 617.0, "a_scale": 7036.3, "alpha_a": -0.24, "tau_fat": 76.2},
        }
    ),
)

_BASELINES = (_PUBLISHED_CODE, _REPOSITORY_CURRENT)


def nominal_baselines() -> tuple[PhysiologicalBaseline, ...]:
    """Return the two immutable, named, unreconciled nominal sources."""

    return _BASELINES


def get_baseline(baseline_id: str) -> PhysiologicalBaseline:
    for baseline in _BASELINES:
        if baseline.baseline_id == baseline_id:
            return baseline
    raise ValueError(
        f"Unknown baseline_id {baseline_id!r}; expected one of "
        f"{tuple(baseline.baseline_id for baseline in _BASELINES)}."
    )


def _factor_slug(factor: float) -> str:
    return f"{factor:g}".replace(".", "p")


def _case_parameters(
    baseline: PhysiologicalBaseline,
    factors: tuple[ParameterFactor, ...],
) -> tuple[MusclePhysiologyParameters, ...]:
    """Rebuild one case directly from nominal data, never from another case."""

    changes = {(factor.muscle_name, factor.parameter_name): factor.factor for factor in factors}
    if len(changes) != len(factors):
        raise ValueError("A case cannot scale the same muscle parameter more than once.")
    output = []
    for nominal in baseline.parameters:
        values = nominal.as_parameter_dict()
        for parameter_name in SENSITIVITY_FIELDS:
            factor = changes.get((nominal.muscle_name, parameter_name))
            if factor is not None:
                values[parameter_name] *= factor
        output.append(
            MusclePhysiologyParameters(muscle_name=nominal.muscle_name, **values)
        )
    return tuple(output)


def _make_case(
    baseline: PhysiologicalBaseline,
    *,
    case_id_suffix: str,
    name: str,
    control_family: str,
    factors: tuple[ParameterFactor, ...],
) -> PhysiologicalParameterCase:
    return PhysiologicalParameterCase(
        case_id=f"{baseline.baseline_id}--{case_id_suffix}",
        name=name,
        baseline_id=baseline.baseline_id,
        provenance_id=baseline.provenance_id,
        generation_protocol_id=GENERATION_PROTOCOL_ID,
        control_family=control_family,
        factor=factors[0].factor if len(factors) == 1 else (1.0 if not factors else None),
        factors=factors,
        changed_fields=tuple(
            f"{factor.muscle_name}.{factor.parameter_name}" for factor in factors
        ),
        parameters=_case_parameters(baseline, factors),
    )


def list_cases(
    baseline_id: str = REPOSITORY_CURRENT_BASELINE_ID,
    *,
    include_targeted_cross: bool = True,
) -> tuple[PhysiologicalParameterCase, ...]:
    """Build one bounded panel for exactly one named baseline.

    The unit factor is represented once by the nominal case.  OFAT cases scale
    the magnitude of negative ``alpha_a`` just as they scale positive fields,
    so its sign remains negative.  The optional cross panel is an abstract
    two-factor stress test, not a clinically motivated interaction model.
    """

    if not isinstance(include_targeted_cross, bool):
        raise ValueError("include_targeted_cross must be boolean.")
    baseline = get_baseline(baseline_id)
    cases = [
        _make_case(
            baseline,
            case_id_suffix="nominal",
            name=f"{baseline.baseline_id} nominal",
            control_family="nominal",
            factors=(),
        )
    ]
    for muscle_name in MUSCLE_NAMES:
        for parameter_name in SENSITIVITY_FIELDS:
            for factor in OFAT_FACTORS:
                if factor == 1.0:
                    continue
                declaration = ParameterFactor(muscle_name, parameter_name, factor)
                cases.append(
                    _make_case(
                        baseline,
                        case_id_suffix=(
                            f"ofat--{muscle_name.lower()}--{parameter_name.lower()}"
                            f"--x{_factor_slug(factor)}"
                        ),
                        name=(
                            f"{baseline.baseline_id}: {muscle_name} {parameter_name} "
                            f"x{factor:g}"
                        ),
                        control_family="one_factor_at_a_time",
                        factors=(declaration,),
                    )
                )

    if include_targeted_cross:
        for anterior_factor in TARGETED_CROSS_FACTORS:
            for triceps_factor in TARGETED_CROSS_FACTORS:
                factors = (
                    ParameterFactor("Delt_ant", "alpha_a", anterior_factor),
                    ParameterFactor("Triceps", "alpha_a", triceps_factor),
                )
                cases.append(
                    _make_case(
                        baseline,
                        case_id_suffix=(
                            "cross--delt-ant-alpha-a-"
                            f"x{_factor_slug(anterior_factor)}--triceps-alpha-a-"
                            f"x{_factor_slug(triceps_factor)}"
                        ),
                        name=(
                            f"{baseline.baseline_id}: abstract anterior/triceps "
                            f"fatigability cross x{anterior_factor:g}/x{triceps_factor:g}"
                        ),
                        control_family="targeted_fatigability_cross",
                        factors=factors,
                    )
                )

    identifiers = tuple(case.case_id for case in cases)
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Generated case identifiers are not unique.")
    if len(cases) > MAX_CASES_PER_PANEL:
        raise RuntimeError(
            f"Generated panel has {len(cases)} cases, above budget {MAX_CASES_PER_PANEL}."
        )
    return tuple(cases)


__all__ = [
    "GENERATION_PROTOCOL_ID",
    "MAX_CASES_PER_PANEL",
    "MUSCLE_NAMES",
    "MusclePhysiologyParameters",
    "OFAT_FACTORS",
    "PUBLISHED_CODE_BASELINE_ID",
    "PUBLISHED_CODE_COMMIT",
    "ParameterFactor",
    "PhysiologicalBaseline",
    "PhysiologicalParameterCase",
    "REPOSITORY_CURRENT_BASELINE_ID",
    "SENSITIVITY_FIELDS",
    "TARGETED_CROSS_FACTORS",
    "get_baseline",
    "list_cases",
    "nominal_baselines",
]
