import math

import casadi as ca
import numpy as np

from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters
from cocofest.optimization.endurance_rollout import (
    DingRolloutMuscleParameters,
    PeriodicRecruitmentProfile,
    rollout_periodic_ding_endurance,
)
from cocofest.optimization.endurance_rollout_objective import (
    RolloutObjectiveLayout,
    build_rollout_objective_function,
    pack_rollout_objective_parameters,
)
from cocofest.optimization.periodic_force_profile import PeriodicFourierForceProfile


def _muscles():
    return [
        DingRolloutMuscleParameters(
            fatigue=DingFatigueParameters(
                a_rest=1500.0 + 200.0 * index,
                tau1_rest=0.060601,
                km_rest=0.137,
                alpha_a=-0.2 - 0.03 * index,
                alpha_tau1=2.1e-5,
                alpha_km=1.9e-5,
                tau_fat=120.0 + 10.0 * index,
            ),
            tau2=0.001,
            pd0=0.000131405,
            pdt=0.000194138,
            pulse_width_max=0.0006,
        )
        for index in range(2)
    ]


def _profile(force_offset=0.0):
    interval_count = 4
    shape = (2, interval_count)
    return PeriodicRecruitmentProfile(
        force_profile=PeriodicFourierForceProfile(
            period=0.8,
            mean=np.array([12.0 + force_offset, 18.0 + force_offset]),
            cosine=np.array([[0.4], [0.2]]),
            sine=np.array([[0.2], [0.3]]),
        ),
        interval_count=interval_count,
        cn=np.full(shape, 0.42),
        force_length_relationship=np.full(shape, 0.95),
        force_velocity_relationship=np.full(shape, 0.98),
        passive_force_relationship=np.full(shape, 0.02),
    )


def test_symbolic_objective_matches_the_numerical_continuous_force_rollout():
    muscles = _muscles()
    profile = _profile()
    layout = RolloutObjectiveLayout(
        muscle_count=2,
        interval_count=4,
        horizon_cycles=3,
        smooth_max_temperature=0.02,
    )
    initial = np.array([[1450.0, 0.065, 0.142], [1620.0, 0.066, 0.143]])
    parameters = pack_rollout_objective_parameters(profile, muscles, layout)
    function = build_rollout_objective_function(muscles=muscles, layout=layout)

    symbolic = function(initial.reshape(-1), parameters)
    numerical = rollout_periodic_ding_endurance(
        profile,
        initial_slow_states=initial,
        muscles=muscles,
        horizon_cycles=layout.horizon_cycles,
    )

    np.testing.assert_allclose(
        np.asarray(symbolic[1]).ravel(),
        numerical.utilization.reshape(-1),
        rtol=2e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        np.asarray(symbolic[2]).ravel(),
        numerical.cycle_boundary_states[-1].reshape(-1),
        rtol=2e-13,
        atol=2e-13,
    )
    values = numerical.utilization.reshape(-1)
    expected_smooth_maximum = layout.smooth_max_temperature * (
        np.logaddexp.reduce(values / layout.smooth_max_temperature)
        - math.log(values.size)
    )
    np.testing.assert_allclose(
        float(symbolic[0]), expected_smooth_maximum, rtol=2e-13, atol=2e-13
    )


def test_one_graph_accepts_new_profile_parameters_and_has_finite_gradients():
    muscles = _muscles()
    layout = RolloutObjectiveLayout(2, 4, 5)
    first = pack_rollout_objective_parameters(_profile(), muscles, layout)
    second = pack_rollout_objective_parameters(_profile(force_offset=2.0), muscles, layout)
    function = build_rollout_objective_function(
        muscles=muscles,
        layout=layout,
        symbolic_type="MX",
    )
    initial = np.array([1450.0, 0.065, 0.142, 1620.0, 0.066, 0.143])

    first_value = float(function(initial, first)[0])
    second_value = float(function(initial, second)[0])
    assert second_value != first_value
    assert function.size1_in("profile") == layout.parameter_size

    state = ca.MX.sym("state", layout.initial_state_size)
    parameters = ca.MX.sym("parameters", layout.parameter_size)
    objective = function(state, parameters)[0]
    gradient = ca.Function(
        "rollout_gradient",
        [state, parameters],
        [ca.gradient(objective, state), ca.gradient(objective, parameters)],
    )
    evaluated = gradient(initial, first)
    assert all(np.all(np.isfinite(np.asarray(value))) for value in evaluated)


def test_fixed_size_rollout_objective_generates_c_code(tmp_path, monkeypatch):
    layout = RolloutObjectiveLayout(2, 4, 2)
    function = build_rollout_objective_function(muscles=_muscles(), layout=layout)

    monkeypatch.chdir(tmp_path)
    function.generate("compiled_endurance_rollout_objective.c", {"with_header": True})

    assert (tmp_path / "compiled_endurance_rollout_objective.c").is_file()
    assert (tmp_path / "compiled_endurance_rollout_objective.h").is_file()
