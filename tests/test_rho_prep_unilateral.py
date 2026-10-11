import numpy as np
import pytest

from scripts.run_rho_prep_unilateral import (
    _geometric_mean_one,
    _positive_weights,
    _scheduled_update_due,
)


def test_geometric_mean_normalization_preserves_weight_ratios():
    weights = np.array([0.05, 0.2, 0.13, 0.95])
    normalized = _geometric_mean_one(weights)
    assert np.exp(np.mean(np.log(normalized))) == pytest.approx(1.0)
    assert normalized / normalized[0] == pytest.approx(weights / weights[0])


@pytest.mark.parametrize("bad", ([1, 0], [1, -1], [1, float("nan")], []))
def test_geometric_mean_normalization_rejects_invalid_weights(bad):
    with pytest.raises(ValueError):
        _geometric_mean_one(bad)


def test_named_weight_order_is_explicit():
    names = ("Delt_ant", "Delt_post", "Biceps", "Triceps")
    values = _positive_weights({name: index + 1 for index, name in enumerate(names)}, names)
    assert values == pytest.approx([1, 2, 3, 4])


def test_named_weights_reject_missing_or_nonpositive_values():
    names = ("Delt_ant", "Delt_post")
    with pytest.raises(KeyError):
        _positive_weights({"Delt_ant": 1.0}, names)
    with pytest.raises(ValueError):
        _positive_weights({"Delt_ant": 1.0, "Delt_post": 0.0}, names)


def test_state_trigger_waits_for_next_scheduled_update_cycle():
    assert not _scheduled_update_due(392, 20)
    assert _scheduled_update_due(400, 20)
    assert _scheduled_update_due(1, 1)


def test_update_cadence_must_be_positive():
    with pytest.raises(ValueError):
        _scheduled_update_due(20, 0)
