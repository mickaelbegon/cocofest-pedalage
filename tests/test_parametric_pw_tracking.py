import numpy as np
import pytest
from cocofest.optimization.parametric_pw_tracking import ParametricPulseWidthTrackingBinding


def test_tracking_update_is_inert_at_zero_and_rejects_bad_shapes():
    binding = ParametricPulseWidthTrackingBinding(("Biceps", "Triceps"), 3)
    summary = binding.update(target=[[.1, .2, .3], [.4, .5, .6]], activation=0.)
    assert summary["activation"] == 0. and not summary["objective_graph_rebuild_required"]
    with pytest.raises(ValueError, match="wrong shape"):
        binding.update(target=[[.1]], activation=1.)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        binding.update(target=np.zeros((2, 3)), activation=1.1)


def test_tracking_layout_is_muscle_major():
    binding = ParametricPulseWidthTrackingBinding(("Biceps", "Triceps"), 3)
    binding.update(target=[[1., 2., 3.], [4., 5., 6.]], activation=.5)
    assert binding.target.ravel().tolist() == [1., 2., 3., 4., 5., 6.]


def test_tracking_parameter_options_expose_target_and_activation():
    binding = ParametricPulseWidthTrackingBinding(("Biceps",), 2)
    options = binding.parameter_options()
    assert set(options) == {"parameters", "parameter_bounds", "parameter_init"}
