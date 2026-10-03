"""Focused tests for the archive-only marginal fatigue audit."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_fatigue_marginal_cost.py"
SPEC = importlib.util.spec_from_file_location("audit_fatigue_marginal_cost", MODULE_PATH)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def _result(cycles: int = 5):
    return {
        "configured_model": {
            "model_builds": [{"muscle_names": ["Biceps", "Triceps"]}],
            "configured_muscle_parameters": {
                "Biceps": {"a_scale": 100.0}, "Triceps": {"a_scale": 200.0},
            },
        },
        "cycles": [
            {
                "cycle": index + 1, "certified": True, "cost": float(index),
                "capacity_ratios": {"Biceps": 1 - .02 * index, "Triceps": 1 - .04 * index},
                "fatigue_objective_weights": [2.0, .5],
            }
            for index in range(cycles)
        ],
    }


def test_endpoint_derivative_matches_central_difference_and_physical_scale():
    terms = audit.endpoint_terms(.8, 2.0, 100.0)
    assert terms["endpoint_density"] == pytest.approx(800.0)
    assert terms["derivative_per_capacity_ratio"] == pytest.approx(-8000.0)
    assert terms["finite_difference_per_capacity_ratio"] == pytest.approx(-8000.0, rel=1e-10)
    assert terms["derivative_per_A_unit"] == pytest.approx(-80.0)


def test_audit_reads_muscle_order_and_certified_checkpoints(tmp_path):
    case = tmp_path / "case"
    for side in ("right", "left"):
        path = case / side
        path.mkdir(parents=True)
        (path / "result.json").write_text(json.dumps(_result()), encoding="utf-8")
    output = tmp_path / "new-audit"
    rows = audit.write_audit([("unit", case)], output)
    assert len(rows) == 12  # two sides, three checkpoints, two muscles
    assert [row["cycle"] for row in rows if row["side"] == "right" and row["muscle"] == "Biceps"] == [1, 3, 5]
    late = next(row for row in rows if row["side"] == "right" and row["checkpoint"] == "late" and row["muscle"] == "Triceps")
    assert late["objective_weight_measured"] == .5
    assert late["capacity_ratio_measured"] == pytest.approx(.84)
    assert late["derivative_per_capacity_ratio"] == pytest.approx(-1600)
    assert (output / "audit.csv").exists()
    assert "pas des costates" in (output / "report.md").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        audit.write_audit([("unit", case)], output)


def test_missing_objective_weight_is_reported(tmp_path):
    case = tmp_path / "case"
    for side in ("right", "left"):
        path = case / side
        path.mkdir(parents=True)
        result = _result()
        result["cycles"][0].pop("fatigue_objective_weights")
        (path / "result.json").write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing endpoint ratios or objective weights"):
        audit.audit_case("unit", case)
