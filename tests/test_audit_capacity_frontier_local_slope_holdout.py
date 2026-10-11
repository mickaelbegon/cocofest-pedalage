import pytest

from scripts.audit_capacity_frontier_local_slope_holdout import audit


def summary(scale):
    rows = []
    for muscle, values in (("A", (.42, .36, .32)), ("B", (.39, .35, .33))):
        rows.append({"muscle": muscle, "cuts": [
            {"epsilon_upper": epsilon, "capacity_ratio": [.9, value]}
            for epsilon, value in zip((.4, .6, .8), (values[0] * scale, values[1] * scale, values[2] * scale))
        ]})
    return {"kind": "local_epsilon_exchange_replication_summary", "muscle_order": ["A", "B", "Triceps"],
            "rows": rows, "all_monotonicity_checks_passed": True}


def test_local_slope_holdout_uses_bracketed_not_global_secant_slope():
    value = audit(summary(1.), summary(1.01), summary(1.02), maximum_midpoint_relative_error=.10)
    assert value["passed"]
    assert value["muscle_order"] == ["A", "B", "Triceps"]
    assert value["muscles"]["A"]["intervals_overlap"]
    assert value["predicted_conservative_exchange_slopes"]["A"] > 0
    assert sum(value["diagnostic_fatigue_weights"].values()) / 3 == pytest.approx(1.)
