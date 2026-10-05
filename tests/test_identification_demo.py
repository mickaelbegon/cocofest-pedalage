"""End-to-end checks of trial design, staged fitting and held-out prediction."""

import json

import pytest

from cocofest.identification.demo import run_demo


@pytest.mark.parametrize("two_axis", [False, True])
def test_synthetic_design_then_force_and_fatigue_fit_with_disjoint_holdouts(two_axis):
    result = run_demo(two_axis=two_axis)
    assert result["channels"] == (["tangential_n", "radial_n"] if two_axis else ["tangential_n"])
    assert len(result["force_design"]["selected"]) == 2
    assert len(result["fatigue_design"]["selected"]) == 1
    selected = {entry["name"] for stage in ("force", "fatigue")
                for entry in result[f"{stage}_design"]["selected"]}
    assert selected.isdisjoint(result["holdout_names"])
    truth = result["synthetic_truth"]
    force = result["force_fit"]
    fatigue = result["fatigue_fit"]
    assert force["converged"] and fatigue["converged"]
    assert force["fisher"]["rank"] == 2
    assert fatigue["fisher"]["rank"] == 1
    assert fatigue["parameters"]["a_rest"] == force["parameters"]["a_rest"]
    assert fatigue["parameters"]["pdt"] == force["parameters"]["pdt"]
    assert force["parameters"]["a_rest"] == pytest.approx(truth["a_rest"], rel=0.03)
    assert force["parameters"]["pdt"] == pytest.approx(truth["pdt"], rel=0.08)
    assert fatigue["parameters"]["alpha_a"] == pytest.approx(truth["alpha_a"], abs=0.02)
    for stage in ("force", "fatigue"):
        holdout = result[f"{stage}_fit"]["holdout_metrics"]
        assert len(holdout) == 1
        metric = next(iter(holdout.values()))
        assert len(metric["rmse_by_channel"]) == (2 if two_axis else 1)
        assert metric["noise_normalized_rmse"] < 2.
    json.dumps(result, allow_nan=False)
