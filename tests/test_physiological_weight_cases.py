import ast
from dataclasses import FrozenInstanceError
import math
from pathlib import Path

import pytest

from cocofest.optimization.physiological_weight_cases import (
    MAX_CASES_PER_PANEL,
    MUSCLE_NAMES,
    OFAT_FACTORS,
    PUBLISHED_CODE_BASELINE_ID,
    PUBLISHED_CODE_COMMIT,
    REPOSITORY_CURRENT_BASELINE_ID,
    SENSITIVITY_FIELDS,
    get_baseline,
    list_cases,
    nominal_baselines,
)


PUBLISHED_EXACT = {
    "Delt_ant": {"Fmax": 48.0, "a_scale": 1148.6, "alpha_a": -0.14, "tau_fat": 445.5},
    "Delt_post": {"Fmax": 51.0, "a_scale": 1234.5, "alpha_a": -0.11, "tau_fat": 342.7},
    "Biceps": {"Fmax": 149.0, "a_scale": 3314.7, "alpha_a": -0.056, "tau_fat": 179.6},
    "Triceps": {"Fmax": 262.0, "a_scale": 4915.5, "alpha_a": -0.034, "tau_fat": 109.1},
}

REPOSITORY_EXACT = {
    "Delt_ant": {"Fmax": 48.0, "a_scale": 1148.6, "alpha_a": -1.4, "tau_fat": 445.5},
    "Delt_post": {"Fmax": 51.0, "a_scale": 1234.5, "alpha_a": -1.1, "tau_fat": 342.7},
    "Biceps": {"Fmax": 149.0, "a_scale": 3314.7, "alpha_a": -0.56, "tau_fat": 179.6},
    "Triceps": {"Fmax": 617.0, "a_scale": 7036.3, "alpha_a": -0.24, "tau_fat": 76.2},
}

ROOT = Path(__file__).resolve().parents[1]
SET_FES_MODEL_SOURCE = (
    ROOT / "examples/fes_multibody/cycling/cycling_pulse_width_mhe.py"
)


def test_named_nominal_sources_are_exact_distinct_and_unreconciled():
    published, repository = nominal_baselines()

    assert published.baseline_id == PUBLISHED_CODE_BASELINE_ID
    assert repository.baseline_id == REPOSITORY_CURRENT_BASELINE_ID
    assert PUBLISHED_CODE_COMMIT in published.provenance_id
    assert published.reconciliation_status == "not_reconciled"
    assert repository.reconciliation_status == "not_reconciled"
    assert published.as_parameter_dict() == PUBLISHED_EXACT
    assert repository.as_parameter_dict() == REPOSITORY_EXACT
    assert published.as_parameter_dict()["Triceps"] != repository.as_parameter_dict()["Triceps"]


def test_repository_baseline_matches_safely_evaluated_local_parameter_dict():
    """Catch scientific-notation transcription errors without importing the example."""

    module = ast.parse(SET_FES_MODEL_SOURCE.read_text())
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "set_fes_model"
    )
    assignment = next(
        node
        for node in function.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "parameter_dict"
    )
    allowed_nodes = (
        ast.Dict,
        ast.Constant,
        ast.UnaryOp,
        ast.USub,
        ast.UAdd,
        ast.BinOp,
        ast.Mult,
    )
    unexpected = {
        type(node).__name__
        for node in ast.walk(assignment.value)
        if not isinstance(node, allowed_nodes)
    }
    assert unexpected == set(), f"Unsafe parameter_dict expression nodes: {sorted(unexpected)}"
    expression = ast.Expression(body=assignment.value)
    ast.fix_missing_locations(expression)
    extracted = eval(  # noqa: S307 - expression tree is restricted to numeric literals/dicts
        compile(expression, str(SET_FES_MODEL_SOURCE), "eval"),
        {"__builtins__": {}},
        {},
    )

    declared = get_baseline(REPOSITORY_CURRENT_BASELINE_ID).as_parameter_dict()
    assert set(declared) == set(extracted)
    for muscle_name in declared:
        assert set(declared[muscle_name]) == set(extracted[muscle_name])
        for parameter_name, source_value in extracted[muscle_name].items():
            assert declared[muscle_name][parameter_name] == pytest.approx(
                source_value, rel=0.0, abs=1e-15
            )


@pytest.mark.parametrize("baseline_id", (PUBLISHED_CODE_BASELINE_ID, REPOSITORY_CURRENT_BASELINE_ID))
def test_panel_is_deterministic_bounded_and_has_unique_ids(baseline_id):
    first = list_cases(baseline_id)
    second = list_cases(baseline_id)

    assert first == second
    assert len(first) == 69
    assert len(first) <= MAX_CASES_PER_PANEL
    assert len({case.case_id for case in first}) == len(first)
    assert sum(case.control_family == "nominal" for case in first) == 1
    assert sum(case.control_family == "one_factor_at_a_time" for case in first) == 64
    assert sum(case.control_family == "targeted_fatigability_cross" for case in first) == 4
    assert len(list_cases(baseline_id, include_targeted_cross=False)) == 65


def test_ofat_cases_recalculate_from_nominal_and_change_exactly_one_field():
    baseline = get_baseline(REPOSITORY_CURRENT_BASELINE_ID)
    nominal = baseline.as_parameter_dict()
    cases = list_cases(REPOSITORY_CURRENT_BASELINE_ID)

    for case in cases:
        assert tuple(parameter.muscle_name for parameter in case.parameters) == MUSCLE_NAMES
        assert case.weight_recalculation_required is True
        assert case.inherited_weights is False
        assert not hasattr(case, "weights")
        for parameter in case.parameters:
            assert parameter.Fmax > 0.0
            assert parameter.a_scale > 0.0
            assert parameter.tau_fat > 0.0
            assert parameter.alpha_a < 0.0

        if case.control_family != "one_factor_at_a_time":
            continue
        assert len(case.changed_fields) == 1
        change = case.factors[0]
        assert change.factor in OFAT_FACTORS
        assert change.factor != 1.0
        expected = {
            muscle: dict(parameters) for muscle, parameters in nominal.items()
        }
        expected[change.muscle_name][change.parameter_name] *= change.factor
        assert case.as_parameter_dict() == expected


def test_fmax_and_capacity_scale_are_independent_ofat_families():
    cases = list_cases(REPOSITORY_CURRENT_BASELINE_ID, include_targeted_cross=False)
    triceps_fmax = next(
        case
        for case in cases
        if case.changed_fields == ("Triceps.Fmax",) and case.factor == 2.0
    )
    triceps_capacity = next(
        case
        for case in cases
        if case.changed_fields == ("Triceps.a_scale",) and case.factor == 2.0
    )
    nominal = REPOSITORY_EXACT["Triceps"]

    assert triceps_fmax.as_parameter_dict()["Triceps"]["Fmax"] == 2.0 * nominal["Fmax"]
    assert triceps_fmax.as_parameter_dict()["Triceps"]["a_scale"] == nominal["a_scale"]
    assert triceps_capacity.as_parameter_dict()["Triceps"]["a_scale"] == 2.0 * nominal["a_scale"]
    assert triceps_capacity.as_parameter_dict()["Triceps"]["Fmax"] == nominal["Fmax"]


def test_targeted_cross_is_abstract_and_changes_only_two_alpha_magnitudes():
    nominal = get_baseline(PUBLISHED_CODE_BASELINE_ID).as_parameter_dict()
    cross_cases = [
        case
        for case in list_cases(PUBLISHED_CODE_BASELINE_ID)
        if case.control_family == "targeted_fatigability_cross"
    ]

    for case in cross_cases:
        assert case.factor is None
        assert case.changed_fields == ("Delt_ant.alpha_a", "Triceps.alpha_a")
        assert case.clinical_interpretation == "none_parameter_perturbations_are_abstract"
        expected = {muscle: dict(parameters) for muscle, parameters in nominal.items()}
        for change in case.factors:
            expected[change.muscle_name]["alpha_a"] *= change.factor
        assert case.as_parameter_dict() == expected


def test_returned_parameter_data_are_immutable_and_freshly_materialized():
    case = list_cases()[0]
    first = case.as_parameter_dict()
    second = case.as_parameter_dict()

    assert first == second
    assert first is not second
    assert first["Triceps"] is not second["Triceps"]
    first["Triceps"]["Fmax"] = math.nan
    assert case.as_parameter_dict()["Triceps"]["Fmax"] == REPOSITORY_EXACT["Triceps"]["Fmax"]
    with pytest.raises(FrozenInstanceError):
        case.parameters[0].Fmax = 0.0


def test_every_declared_ofat_field_and_factor_is_present_for_every_muscle():
    cases = list_cases(include_targeted_cross=False)
    declarations = {
        (case.factors[0].muscle_name, case.factors[0].parameter_name, case.factor)
        for case in cases
        if case.control_family == "one_factor_at_a_time"
    }
    expected = {
        (muscle, field, factor)
        for muscle in MUSCLE_NAMES
        for field in SENSITIVITY_FIELDS
        for factor in OFAT_FACTORS
        if factor != 1.0
    }

    assert declarations == expected
