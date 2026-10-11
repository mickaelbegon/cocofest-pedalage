from dataclasses import replace

import numpy as np
import pytest

from cocofest.cycling.slow import ConditionalRemainingCycles
from cocofest.cycling.slow.split_rule import SplitComfort, equalize_remaining_cycles


def estimates(left=50, right=100):
    return {name: ConditionalRemainingCycles(n, n+1, 200, (name,), str(i)*64)
            for i, (name, n) in enumerate((("left", left), ("right", right)))}


def test_transfer_moves_to_longer_lived_arm_preserves_total_and_receipt():
    comfort = SplitComfort((.5, .5), (1.5, 1.5))
    result = equalize_remaining_cycles((.96, .96), estimates(), comfort)
    assert result.torque[1] > .96 > result.torque[0]
    assert sum(result.torque) == pytest.approx(1.92)
    assert result.proposed_transfer_to_left == pytest.approx(-.05/3)
    assert result.reason == "adjusted"
    assert len(result.receipt_sha256) == 64
    assert result.receipt_sha256 == equalize_remaining_cycles((.96, .96), estimates(), comfort).receipt_sha256
    assert result.receipt_sha256 != equalize_remaining_cycles((.96, .96), estimates(50, 99), comfort).receipt_sha256


@pytest.mark.parametrize("kind,reason", [("censored", "hold_censored_estimate"),
    ("missing", "hold_invalid_estimate"), ("invalid", "hold_invalid_estimate"),
    ("zero", "hold_zero_remaining_cycles"), ("equal", "hold_equal_estimates")])
def test_conservative_hold(kind, reason):
    values = estimates()
    if kind == "censored":
        values["left"] = replace(values["left"], cycles_within_bounds=200, first_exceedance_cycle=None)
    elif kind == "missing":
        del values["left"]
    elif kind == "invalid":
        values["left"] = replace(values["left"], cycles_within_bounds=-1)
    elif kind == "zero":
        values = estimates(0, 100)
    else:
        values = estimates(100, 100)
    result = equalize_remaining_cycles((.86, 1.06), values, SplitComfort((.5, .5), (1.5, 1.5)))
    assert result.torque == (.86, 1.06)
    assert result.reason == reason


def test_projection_enforces_asymmetry_and_absolute_torque_limit():
    result = equalize_remaining_cycles((.82, 1.1), estimates(1, 100), SplitComfort((.5, .5), (1.5, 1.5)))
    assert result.torque == pytest.approx((.81, 1.11))
    result = equalize_remaining_cycles((.96, .96), estimates(1, 100), SplitComfort((.95, .5), (1.5, 1.5)))
    assert result.torque == pytest.approx((.95, .97))


def test_random_updates_always_meet_comfort_constraints():
    random = np.random.default_rng(102)
    comfort = SplitComfort((.6, .7), (1.2, 1.3), delta_max=.05, asymmetry_max=.3)
    for _ in range(200):
        left = random.uniform(.81, 1.11)
        current = (left, 1.92-left)
        result = equalize_remaining_cycles(current, estimates(*map(int, random.integers(1, 180, size=2))), comfort)
        assert sum(result.torque) == pytest.approx(sum(current))
        assert np.all(np.asarray(result.torque) >= comfort.minimum)
        assert np.all(np.asarray(result.torque) <= comfort.maximum)
        assert np.max(np.abs(np.asarray(result.torque)-current)) <= comfort.delta_max+1e-12
        assert abs(result.torque[0]-result.torque[1]) <= comfort.asymmetry_max+1e-12


def test_invalid_current_split_requires_recovery_not_silent_hold():
    with pytest.raises(ValueError, match="recovery policy"):
        equalize_remaining_cycles((.5, 1.5), {}, SplitComfort((.5, .5), (1.5, 1.5)))
