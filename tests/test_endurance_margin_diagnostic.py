import numpy as np
import pytest

from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters
from cocofest.optimization.ding_slow_cycle import slow_cycle_map_from_collocation
from cocofest.optimization.endurance_margin_diagnostic import (
    LocalMarginDiagnostic, consumption_time_proxy, repeated_force_margin_path,
    witness_grid_consumption,
)


def test_no_depletion_is_unknown_not_infinite_endurance():
    assert consumption_time_proxy(.2, 0)["cycles"] is None
    assert consumption_time_proxy(.2, -.1)["cycles"] is None
    assert consumption_time_proxy(-.2, .1)["cycles"] is None
    assert consumption_time_proxy(.2, .01)["cycles"] == pytest.approx(20)


def test_witness_cap_is_preserved_and_not_claimed_as_maximum():
    rows = witness_grid_consumption([
        {"cycle": 140, "load_lower_bound": 1.15, "tested_maximum": 1.2},
        {"cycle": 120, "load_lower_bound": 1.2, "tested_maximum": 1.2},
        {"cycle": 160, "load_lower_bound": 1.05, "tested_maximum": 1.2},
    ])
    assert rows[0]["at_tested_load_cap"]
    assert not rows[1]["at_tested_load_cap"]
    assert rows[1]["time_proxy"]["cycles"] == pytest.approx(60)
    assert rows[2]["time_proxy"]["cycles"] == pytest.approx(10)


def test_conditionally_repeated_force_reports_extrapolation_before_zero_crossing():
    parameters = DingFatigueParameters(100, 1, 1, -1, .01, .01, 1000)
    cycle = slow_cycle_map_from_collocation([[10, 10]], [1], [0, 1], parameters)
    model = LocalMarginDiagnostic(("A_m", "Tau1_m", "Km_m", "F_m"),
        (1, 1, 1, 1), (100, 1, 1, 10), (0, 0, 0, 0), (1, 0, 0, 100),
        (.01, .1, .1, 1), 1.5)
    initial = {"A_m": 100, "Tau1_m": 1, "Km_m": 1, "F_m": 10}
    report = repeated_force_margin_path(model, initial, {"m": cycle}, horizon=10)
    assert report["first_coordinate_trust_exit"] == 1
    assert report["first_affine_zero_crossing"] == 6
    assert report["zero_crossing_inside_trust"] is False
    assert report["physiological_failure_certified"] is False
    assert report["rows"][0]["affine_margin"] == .5  # Fast F is frozen, no invented jump.
    assert initial["A_m"] == 100


def test_normalization_and_sign_are_physically_consistent():
    model = LocalMarginDiagnostic(("A_m",), (.8,), (100,), (20,), (2,), (.1,), 1.2)
    current = model.coordinates({"A_m": 100})
    improved = model.coordinates({"A_m": 105})
    assert model.evaluate(improved)["affine_margin"] > model.evaluate(current)["affine_margin"]
    # If used in a minimizing objective the sign would be negative.
    assert -np.dot(model.gradient, improved) < -np.dot(model.gradient, current)


def test_missing_slow_coordinates_and_duplicate_cycles_are_rejected():
    p = DingFatigueParameters(100, 1, 1, -1, .01, .01, 1000)
    cycle = slow_cycle_map_from_collocation([[10, 10]], [1], [0, 1], p)
    model = LocalMarginDiagnostic(("A_m",), (1,), (100,), (0,), (1,), (.1,), 1.2)
    with pytest.raises(ValueError, match="Exactly one"):
        repeated_force_margin_path(model, {"A_m": 100}, {"m": cycle}, horizon=20)
    row = {"cycle": 2, "load_lower_bound": 1., "tested_maximum": 2.}
    with pytest.raises(ValueError, match="distinct"):
        witness_grid_consumption([row, row])
