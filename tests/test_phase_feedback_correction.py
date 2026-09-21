from types import SimpleNamespace
import numpy as np
import pytest
from cocofest.optimization.phase_feedback_correction import SourcePhasePulseWidthFeedback


def make_policy(**kwargs):
    options = dict(phase_nodes=[0.1, 0.5, 0.1+2*np.pi], directed_omega_nodes=[1,2,1],
                   commands=[[0.0002,0.0003],[0.0004,0.0002]], period_s=4,
                   direction=-1, effectiveness=lambda theta: [-1,1],
                   lower_bounds=[0.0001,0.0001], upper_bounds=[0.0006,0.0006], maximum_slew_s=0.0005)
    options.update(kwargs)
    return SourcePhasePulseWidthFeedback(**options)


def observation(phase=0.6, time=1, speed=1):
    return SimpleNamespace(unwrapped_phase_rad=phase, theta_rad=-phase, time_s=time, omega_rad_s=-speed)


def test_source_nonuniform_phase_knots_and_origin_are_preserved():
    policy = make_policy()
    np.testing.assert_allclose(policy(observation(0.49)), [0.0002,0.0003])
    np.testing.assert_allclose(policy(observation(0.51)), [0.0004,0.0002])
    np.testing.assert_allclose(policy(observation(0.51+2*np.pi)), [0.0004,0.0002])


def test_feedback_torque_direction_and_application_slew():
    policy = make_policy(phase_gain_s_per_rad=0.01, maximum_slew_s=20e-6)
    first = policy(observation(phase=0.1, time=0))
    second = policy(observation(phase=0.1, time=1))
    assert second[0] == pytest.approx(first[0]+20e-6)
    assert second[1] == pytest.approx(first[1]-20e-6)
    assert np.max(np.abs(second-first)) <= 20e-6+1e-15


def test_source_phase_reversal_is_rejected():
    with pytest.raises(ValueError):
        make_policy(phase_nodes=[0.1,0.05,2*np.pi])
