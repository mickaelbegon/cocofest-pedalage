"""Small contract tests for the opt-in RHO reserve wiring."""

from types import SimpleNamespace

import numpy as np
import pytest

from cocofest.simulation.independent_arms_process import (
    _certified_mechanical_reserve_profile,
    _pulse_width_replay_tolerance_from_nlp,
    _reserve_update_due,
)
from examples.fes_multibody.cycling.cycling_pulse_width_mhe_acados_periodic import (
    build_argument_parser,
    validate_experimental_mechanical_reserve_options,
)


def test_pw_replay_tolerance_uses_per_muscle_nlp_scaling_with_picosecond_floor():
    scaling = {
        "last_pulse_width_biceps": SimpleNamespace(scaling=np.array([[2.5e-3]])),
        "last_pulse_width_triceps": SimpleNamespace(scaling=np.array([[2e-4]])),
    }
    ocp = SimpleNamespace(nlp=[SimpleNamespace(u_scaling=scaling)])
    tolerance, audit = _pulse_width_replay_tolerance_from_nlp(
        ocp=ocp, muscle_names=("biceps", "triceps"), nlp_tolerance=1e-8,
    )
    np.testing.assert_allclose(tolerance, [25e-12, 5e-12])
    assert audit == {
        "units": "s",
        "derivation": "max(nlp_tolerance * u_scaling_s, floor_s)",
        "nlp_tolerance_normalized": 1e-8,
        "u_scaling_s_by_muscle": [2.5e-3, 2e-4],
        "floor_s": 5e-12,
        "effective_tolerance_s_by_muscle": [25e-12, 5e-12],
    }


def test_parse_defaults_to_cost_absent_and_validates_isolated_binding():
    parser = build_argument_parser()
    default = parser.parse_args([])
    assert default.experimental_mechanical_reserve_weight == 0.0
    assert validate_experimental_mechanical_reserve_options(default) is False
    args = parser.parse_args([
        "--experimental-mechanical-reserve-weight", "0.3",
        "--experimental-mechanical-reserve-horizons", "1", "5", "20",
        "--experimental-mechanical-reserve-calibration-policy", "cycle_work_gated_v1",
        "--solver", "ipopt", "--use-sx", "--cycles-per-window", "1",
        "--mechanical-formulation", "reduced", "--formulation", "isokinetic",
    ])
    assert validate_experimental_mechanical_reserve_options(args) is True
    assert args.experimental_mechanical_reserve_calibration_policy == "cycle_work_gated_v1"
    args.experimental_mechanical_reserve_horizons = (1, 20, 5)
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_experimental_mechanical_reserve_options(args)
    args.experimental_mechanical_reserve_horizons = (1, 5, 20)
    args.parametric_fatigue_weights = True
    with pytest.raises(ValueError, match="ParameterList"):
        validate_experimental_mechanical_reserve_options(args)
    args.parametric_fatigue_weights = False
    args._endurance_rollout_options = object()
    with pytest.raises(ValueError, match="ParameterList"):
        validate_experimental_mechanical_reserve_options(args)
    args._endurance_rollout_options = None
    args._muscle_horizon_options = object()
    with pytest.raises(ValueError, match="ParameterList"):
        validate_experimental_mechanical_reserve_options(args)
    args._muscle_horizon_options = None
    args.solver = "acados"
    with pytest.raises(ValueError, match="IPOPT SX"):
        validate_experimental_mechanical_reserve_options(args)


def test_local_pw_reserve_cost_requires_a_positive_trust_radius_and_excludes_global_graph():
    parser = build_argument_parser()
    base = [
        "--experimental-mechanical-reserve-weight", "0.3", "--solver", "ipopt", "--use-sx",
        "--cycles-per-window", "1", "--mechanical-formulation", "reduced", "--formulation", "isokinetic",
        "--experimental-mechanical-reserve-local-pw-cost",
    ]
    args = parser.parse_args(base)
    assert validate_experimental_mechanical_reserve_options(args) is True
    assert args.experimental_mechanical_reserve_local_pw_trust_us == 25.0
    args.experimental_mechanical_reserve_local_pw_trust_us = 0.0
    with pytest.raises(ValueError, match="trust-us"):
        validate_experimental_mechanical_reserve_options(args)
    args.experimental_mechanical_reserve_local_pw_trust_us = 25.0
    args.experimental_mechanical_reserve_pw_force_coupling = True
    with pytest.raises(ValueError, match="either global PW coupling"):
        validate_experimental_mechanical_reserve_options(args)
    args.experimental_mechanical_reserve_local_pw_cost = False
    with pytest.raises(ValueError, match="disabled"):
        validate_experimental_mechanical_reserve_options(args)


@pytest.mark.parametrize("cycle,expected", [(0, False), (1, True), (2, False),
                                             (19, False), (20, True), (21, False),
                                             (40, True), (41, False)])
def test_update_uses_physical_certified_boundaries(cycle, expected):
    assert _reserve_update_due(certified=True, has_solution=True, physical_cycle=cycle) is expected
    assert _reserve_update_due(certified=False, has_solution=True, physical_cycle=cycle) is False
    assert _reserve_update_due(certified=True, has_solution=False, physical_cycle=cycle) is False


def test_mayer_objective_preserves_symbolic_terminal_states_and_binding():
    ca = pytest.importorskip("casadi")
    from cocofest.custom_objectives import CustomObjective

    terminal = {key: ca.SX.sym(key) for key in ("A_biceps", "Tau1_biceps", "Km_biceps")}
    controller = SimpleNamespace(
        model=SimpleNamespace(muscles_dynamics_model=[SimpleNamespace(muscle_name="biceps")]),
        states={key: SimpleNamespace(cx=value) for key, value in terminal.items()},
    )
    class Binding:
        muscle_count = 1
        def objective(self, states, passed_controller):
            assert passed_controller is controller
            assert states.shape == (1, 3)
            return ca.sum2(states)

    expression = CustomObjective.minimize_terminal_projected_mechanical_reserve(controller, Binding())
    gradient = ca.Function("reserve_terminal_gradient", list(terminal.values()),
                           [ca.gradient(expression, ca.vertcat(*terminal.values()))])
    np.testing.assert_array_equal(np.asarray(gradient(2, 3, 4)), np.ones((3, 1)))


def test_certified_profile_uses_signed_phase_endpoints_and_fixed_work(monkeypatch):
    from cocofest.optimization import adaptive_moment_rollout, mechanical_reserve_calibration

    monkeypatch.setattr(adaptive_moment_rollout.DingPulseWidthParameters, "from_model",
                        lambda model, pulse_width_max: object())
    seen = {}
    margin = SimpleNamespace(reference_states=np.ones((1, 3)),
                             reference_margins=np.ones(3),
                             state_jacobian=np.zeros((3, 1, 3)))
    def calibrate(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(margin_model=margin, calibration_kind="local_proxy",
                               attainable_work_certified=False, endurance_prediction=False)
    monkeypatch.setattr(mechanical_reserve_calibration,
                        "calibrate_isokinetic_ding_margin_model", calibrate)
    states = {"theta": np.array([0., -np.pi, -2*np.pi]),
              "Cn_biceps": np.array([1., 1., 1.]),
              "F_biceps": np.array([3., 4., 5.]),
              "A_biceps": np.array([10., 9., 8.]),
              "Tau1_biceps": np.array([1., 1., 1.]),
              "Km_biceps": np.array([1., 1., 1.])}
    model = SimpleNamespace(muscle_name="biceps", post_stimulation_amplitude=lambda: 1.)
    dynamics = SimpleNamespace(
        source_model_sha256="fixture",
        muscle_relationships=lambda theta, omega: ([1.], [1.], [0.]),
        coefficient_values=lambda theta: {
            "muscle_effectiveness": [-0.2], "projected_gravity": 0.1,
            "projected_velocity_quadratic": 0.0},
    )
    binding = SimpleNamespace(muscle_count=1, interval_count=2, durations=(.5, .5))
    forces, returned_margin, receipt, pulse_width_force_model = _certified_mechanical_reserve_profile(
        states=states, models=[model], reduced_dynamics=dynamics,
        phase_count=2, omega=-2*np.pi, target_work_j=2.0,
        binding=binding, source_cycle=20, calibration_policy="cycle_work_gated_v1")
    np.testing.assert_array_equal(forces, [[4., 5.]])
    assert returned_margin is margin
    np.testing.assert_array_equal(seen["required_power_w"], [2., 2.])
    np.testing.assert_allclose(seen["nonmuscle_power_w"], [0.2*np.pi, 0.2*np.pi])
    assert receipt["application_cycle"] == 21
    assert receipt["calibration_policy"] == "cycle_work_gated_v1"
    assert seen["calibration_policy"] == "cycle_work_gated_v1"
    assert receipt["attainable_work_certified"] is False
    assert receipt["force_shape"] == [1, 2]
    assert pulse_width_force_model is None

    states["theta"][-1] = 2*np.pi
    with pytest.raises(ValueError, match="signed pedal turn"):
        _certified_mechanical_reserve_profile(
            states=states, models=[model], reduced_dynamics=dynamics,
            phase_count=2, omega=-2*np.pi, target_work_j=2.0,
            binding=binding, source_cycle=20)
