"""Holdout partition and temporal extrapolation checks for the offline audit."""

import pytest

from scripts.assess_task_margin_value_dataset import (
    BRANCH_HOLDOUT, BRANCH_TRAIN, assess,
)


def _row(label, cycle, reserve, seed):
    state = [0.5 + 0.002 * seed * (index + 1) for index in range(20)]
    return {"label": label, "cycle": cycle, "reserve_witness": reserve,
            "normalized_ding_state": state}


def _dataset():
    time = [_row(f"c{cycle}", cycle, reserve, index) for index, (cycle, reserve) in
            enumerate(((80, .42), (100, .36), (120, .2725), (140, .1802), (160, .0569)))]
    branches = [_row(label, int(label[-3:]), .18 + .003 * index, index + 5)
                for index, label in enumerate((*BRANCH_TRAIN, *BRANCH_HOLDOUT))]
    return time, branches


def test_curved_time_holdout_rejects_recent_linear_extrapolation():
    time, branches = _dataset()
    report = assess(time, branches)
    linear = report["temporal"]["last_two_linear"]
    assert linear["prediction"] == pytest.approx(.0879)
    assert linear["absolute_error"] > .01
    assert linear["passes_0p01_abs_error_gate"] is False
    assert report["costate_activation_authorized"] is False


def test_holdout_policy_must_not_enter_the_training_partition():
    time, branches = _dataset()
    branches.pop()
    with pytest.raises(ValueError, match="partitions changed"):
        assess(time, branches)
