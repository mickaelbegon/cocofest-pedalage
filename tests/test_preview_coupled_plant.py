from types import SimpleNamespace

import numpy as np
import pytest

from scripts.validate_preview_coupled_plant import (
    _slow_comparison,
    _state_rows,
    flatten_preview_pulse_widths,
)


def test_flatten_preview_pulse_widths_preserves_cycle_then_phase_order_per_muscle():
    values = np.array([[[1., 2.], [10., 20.]], [[3., 4.], [30., 40.]]])
    assert np.array_equal(flatten_preview_pulse_widths(values), [[1., 2., 3., 4.], [10., 20., 30., 40.]])
    with pytest.raises(ValueError, match="finite"):
        flatten_preview_pulse_widths(np.array([np.nan]))


def test_state_rows_selects_named_initial_muscle_state_order():
    source = SimpleNamespace(
        state_names=("F_b", "Cn_a", "A_a", "F_a", "Cn_b", "A_b"),
        initial_state=np.array([4., 1., 3., 2., 5., 6.]),
    )
    assert np.array_equal(_state_rows(source, ("a", "b"), ("Cn", "F", "A")), [[1., 2., 3.], [5., 4., 6.]])


def test_slow_comparison_uses_ratio_and_shared_initial_boundary():
    parameters = {"m": {"a_scale": 100., "tau1_rest": 2., "km_rest": 4.}}
    normal = np.array([[[100., 2., 4.]], [[99., 2.1, 4.1]]])
    preview = np.array([[[100., 2., 4.]], [[98., 2.2, 4.2]]])
    report, delta = _slow_comparison(preview, normal, parameters, ("m",))
    assert report["initial_boundary_maximum_absolute_normalized_difference"] == 0.
    terminal = report["terminal_preview_minus_normal_rho"]
    assert terminal["mean_capacity_ratio"] == pytest.approx(-.01)
    assert terminal["mean_tau1_ratio"] == pytest.approx(.05)
    assert terminal["mean_km_ratio"] == pytest.approx(.025)
    assert delta.shape == (2, 1, 3)
