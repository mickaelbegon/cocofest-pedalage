"""Numerical/scientific contracts for the experimental Physio-U proposal."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from cocofest.optimization.physio_update import (
    PhysioUpdateConfig,
    build_isokinetic_max_pw_envelope,
    build_isokinetic_max_pw_envelope_from_discrete_cycle,
    isokinetic_work_shapley,
    prescribed_force_fatigue_challenge,
    propose_mechanical_sensitivity_weights,
    propose_physio_update,
)
from cocofest.optimization.adaptive_moment_rollout import (
    DingPulseWidthParameters,
    MomentTrackingInterval,
    propagate_ding_pulse_width_interval,
)
from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters


def fixture():
    return {
        "completed_cycles": 20, "certified": True,
        "incumbent_weights": np.ones(4),
        "current_capacity": np.array([800., 800., 800., 800.]),
        "rest_capacity": np.full(4, 1000.), "alpha_a": np.array([-.1, -.2, -.3, -.4]),
        "tau_fat": np.full(4, 100.),
        "reference_force": np.full((4, 4), 50.),
        "available_positive_power": np.full((4, 4), .4),
        "phase_durations": np.array([.1, .2, .3, .4]), "required_active_work": .7,
    }


def _pw_parameters(alpha_a=-.2):
    return DingPulseWidthParameters(
        fatigue=DingFatigueParameters(
            a_rest=1000., tau1_rest=.06, km_rest=.13, alpha_a=alpha_a,
            alpha_tau1=2e-5, alpha_km=2e-5, tau_fat=100.,
        ),
        tauc=.011, tau2=.001, pd0=.0001, pdt=.0002, pulse_width_max=.0006,
    )


def _isokinetic_envelope_fixture():
    parameters = (_pw_parameters(-.2), _pw_parameters(-.4))
    states = np.array([
        [.12, 18., 900., .063, .14],
        [.16, 24., 800., .065, .15],
    ])
    intervals = (
        MomentTrackingInterval(.02, (1.0, 1.1), (.95, .9), (-.02, -.03), (0., 0.)),
        MomentTrackingInterval(.03, (1.2, .9), (.92, .88), (-.025, -.02), (0., 0.)),
    )
    return {
        "terminal_states": states,
        "reference_force": np.array([[10., 12.], [20., 14.]]),
        "intervals": intervals,
        "pulse_width_parameters": parameters,
        "angular_velocity_rad_s": -2 * math.pi,
        "integration_substeps": 128,
    }


def test_exact_phase_fatigue_matches_independent_dop853_with_recovery():
    current = np.array([600., 950.])
    rest = np.array([1000., 1200.])
    alpha, tau = np.array([-.25, -.4]), np.array([76., 110.])
    dt = np.array([.01, .033333, .2, .156667])
    forces = np.array([[20., 0., 80., 10.], [15., 20., 100., 0.]])
    result = prescribed_force_fatigue_challenge(current, rest, alpha, tau, forces, dt)
    direct = current.copy()
    for phase, duration in enumerate(dt):
        direct = solve_ivp(lambda t, a: (rest - a) / tau + alpha * forces[:, phase],
                           (0., duration), direct, method="DOP853", rtol=1e-12, atol=1e-12).y[:, -1]
        np.testing.assert_allclose(result["capacity_history"][phase + 1], direct, rtol=2e-13)
    assert result["domain_valid"]
    assert result["normalization"] == "immutable_A_rest"


def test_isokinetic_max_pw_envelope_uses_full_terminal_ding_state_and_task_work():
    args = _isokinetic_envelope_fixture()
    envelope = build_isokinetic_max_pw_envelope(**args)
    assert envelope["force_envelope_model"].startswith("full_five_state_ding")
    assert not envelope["attainable_work_certified"]
    np.testing.assert_allclose(envelope["current_capacity"], args["terminal_states"][:, 2])
    np.testing.assert_allclose(envelope["rest_capacity"], [1000., 1000.])
    assert envelope["available_positive_power"].shape == (2, 2)
    assert np.all(envelope["available_positive_power"] > 0)
    # The work demand comes from the certified reference force and total
    # muscle moment, not from the PW-max counterfactual envelope.
    expected_work = 2 * math.pi * (
        .02 * 10. * .02 + .03 * 20. * .02 + .025 * 12. * .03 + .02 * 14. * .03
    )
    assert envelope["required_active_work"] == pytest.approx(expected_work)

    expected_first = propagate_ding_pulse_width_interval(
        args["terminal_states"][0], pulse_width=args["pulse_width_parameters"][0].pulse_width_max,
        duration=.02, calcium_amplitude=1., mechanical_gain=.95,
        parameters=args["pulse_width_parameters"][0], integration_substeps=128,
    )
    assert envelope["envelope_force"][0, 0] == pytest.approx(expected_first[1])
    np.testing.assert_allclose(
        envelope["available_positive_power"],
        np.maximum(0., args["angular_velocity_rad_s"]
                   * envelope["moment_coefficients"] * envelope["envelope_force"]),
    )


@pytest.mark.parametrize("omega", [-2 * math.pi, 2 * math.pi])
def test_isokinetic_envelope_only_credits_muscles_productive_in_rotation_direction(omega):
    args = _isokinetic_envelope_fixture()
    args["angular_velocity_rad_s"] = omega
    productive_sign = 1. if omega > 0 else -1.
    args["intervals"] = tuple(MomentTrackingInterval(
        interval.duration, interval.calcium_amplitudes, interval.mechanical_gains,
        (productive_sign * .02, -productive_sign * .001), interval.target_moments,
    ) for interval in args["intervals"])
    envelope = build_isokinetic_max_pw_envelope(**args)
    np.testing.assert_allclose(
        envelope["available_positive_power"][0],
        abs(omega) * .02 * envelope["envelope_force"][0],
    )
    np.testing.assert_array_equal(envelope["available_positive_power"][1], 0.)
    assert envelope["required_active_work"] > 0
    credit = isokinetic_work_shapley(
        envelope["available_positive_power"], envelope["phase_durations"],
        envelope["required_active_work"],
    )
    assert credit["mechanical_credit_j"][0] > 0
    assert credit["mechanical_credit_j"][1] == pytest.approx(0.)
    assert not envelope["attainable_work_certified"]


def test_isokinetic_envelope_is_equivariant_to_muscle_permutation():
    args = _isokinetic_envelope_fixture()
    original = build_isokinetic_max_pw_envelope(**args)
    order = [1, 0]
    args["terminal_states"] = args["terminal_states"][order]
    args["reference_force"] = args["reference_force"][order]
    args["pulse_width_parameters"] = tuple(args["pulse_width_parameters"][i] for i in order)
    args["intervals"] = tuple(MomentTrackingInterval(
        interval.duration,
        tuple(interval.calcium_amplitudes[i] for i in order),
        tuple(interval.mechanical_gains[i] for i in order),
        tuple(interval.moment_coefficients[i] for i in order),
        tuple(interval.target_moments[i] for i in order),
    ) for interval in args["intervals"])
    permuted = build_isokinetic_max_pw_envelope(**args)
    np.testing.assert_allclose(
        permuted["available_positive_power"], original["available_positive_power"][order]
    )
    np.testing.assert_allclose(permuted["envelope_force"], original["envelope_force"][order])
    assert permuted["required_active_work"] == pytest.approx(original["required_active_work"])


def test_isokinetic_envelope_refuses_capacity_only_or_nonproductive_inputs():
    args = _isokinetic_envelope_fixture()
    args["terminal_states"] = args["terminal_states"][:, 2]
    with pytest.raises(ValueError, match="terminal_states"):
        build_isokinetic_max_pw_envelope(**args)
    args = _isokinetic_envelope_fixture()
    args["reference_force"] = np.zeros((2, 2))
    with pytest.raises(ValueError, match="positive active isokinetic work"):
        build_isokinetic_max_pw_envelope(**args)


def test_discrete_cycle_bridge_extracts_certified_endpoints_and_labels_geometry_approximation():
    params = _pw_parameters()
    models = tuple(SimpleNamespace(
        muscle_name=f"m{index}", a_scale=params.fatigue.a_rest,
        alpha_a=params.fatigue.alpha_a, alpha_tau1=params.fatigue.alpha_tau1,
        alpha_km=params.fatigue.alpha_km, tau_fat=params.fatigue.tau_fat,
        tau1_rest=params.fatigue.tau1_rest, km_rest=params.fatigue.km_rest,
        tauc=params.tauc, tau2=params.tau2, pd0=params.pd0, pdt=params.pdt,
        post_stimulation_amplitude=lambda: 1.,
    ) for index in range(2))
    # Two phases, each with two collocation nodes after the initial node.
    states = {"theta": np.linspace(0., -2 * math.pi, 5)}
    for index in range(2):
        states.update({
            f"Cn_m{index}": np.full(5, .14), f"F_m{index}": np.full(5, 10. + index),
            f"A_m{index}": np.full(5, 900.), f"Tau1_m{index}": np.full(5, .063),
            f"Km_m{index}": np.full(5, .14),
        })
    reduced = SimpleNamespace(
        muscle_relationships=lambda theta, omega: (np.ones(2), np.ones(2), np.zeros(2)),
        coefficient_values=lambda theta: {"muscle_effectiveness": np.array([-.02, -.03])},
    )
    result = build_isokinetic_max_pw_envelope_from_discrete_cycle(
        states=states, muscle_models=models, reduced_dynamics=reduced,
        stimulations_per_cycle=2, angular_velocity_rad_s=-4 * math.pi,
    )
    assert result["source_state_sampling"] == "certified_phase_endpoints"
    assert result["source_nodes_per_phase"] == 2
    assert result["envelope_geometry_sampling"] == "phase_endpoint_piecewise_constant_v1"
    assert result["source_muscle_names"] == ("m0", "m1")
    np.testing.assert_allclose(result["phase_durations"], [.25, .25])
    assert result["phase_durations"].sum() == pytest.approx(.5)


def test_discrete_cycle_bridge_uses_certified_eprod_not_incomplete_muscle_moment_sum():
    args = _isokinetic_envelope_fixture()
    # Deliberately choose E_prod far from the sum one would infer from the
    # simplified synthetic muscle moments.  The live OCP state is authoritative.
    states = {"theta": np.linspace(0., .4, 5), "E_prod": np.linspace(2., 2.8, 5)}
    params = _pw_parameters()
    models = tuple(SimpleNamespace(
        muscle_name=f"m{index}", a_scale=params.fatigue.a_rest,
        alpha_a=params.fatigue.alpha_a, alpha_tau1=params.fatigue.alpha_tau1,
        alpha_km=params.fatigue.alpha_km, tau_fat=params.fatigue.tau_fat,
        tau1_rest=params.fatigue.tau1_rest, km_rest=params.fatigue.km_rest,
        tauc=params.tauc, tau2=params.tau2, pd0=params.pd0, pdt=params.pdt,
        post_stimulation_amplitude=lambda: 1.,
    ) for index in range(2))
    for index in range(2):
        states.update({
            f"Cn_m{index}": np.full(5, .14), f"F_m{index}": np.full(5, 10. + index),
            f"A_m{index}": np.full(5, 900.), f"Tau1_m{index}": np.full(5, .063),
            f"Km_m{index}": np.full(5, .14),
        })
    reduced = SimpleNamespace(
        muscle_relationships=lambda theta, omega: (np.ones(2), np.ones(2), np.zeros(2)),
        coefficient_values=lambda theta: {"muscle_effectiveness": np.array([-.02, -.03])},
    )
    result = build_isokinetic_max_pw_envelope_from_discrete_cycle(
        states=states, muscle_models=models, reduced_dynamics=reduced,
        stimulations_per_cycle=2, angular_velocity_rad_s=-2 * math.pi,
    )
    assert result["required_active_work"] == pytest.approx(.8)
    assert result["mechanical_demand_model"] == "certified_E_prod_state_difference_v1"


def test_damage_score_at_rest_equals_squared_decrement_and_tired_muscle_recovers():
    args = dict(rest_capacity=[1000.], alpha_a=[-.1], tau_fat=[100.],
                force=[[1.]], phase_durations=[1.])
    fresh = prescribed_force_fatigue_challenge([1000.], **args)
    tired = prescribed_force_fatigue_challenge([500.], **args)
    assert fresh["excess_squared_fatigue"][0] == pytest.approx(fresh["stimulus_capacity_decrement_ratio"][0] ** 2)
    assert tired["capacity_history"][-1, 0] > 500.  # Net recovery does not erase stimulus cost.
    assert tired["excess_squared_fatigue"][0] > fresh["excess_squared_fatigue"][0] > 0
    assert tired["stimulus_capacity_decrement_ratio"] == pytest.approx(fresh["stimulus_capacity_decrement_ratio"])


def test_zero_force_has_zero_incremental_cost_and_domain_failure_is_explicit():
    rested = prescribed_force_fatigue_challenge([500.], [1000.], [-1.], [100.], [[0.]], [1.])
    assert rested["excess_squared_fatigue"] == pytest.approx([0.])
    invalid = prescribed_force_fatigue_challenge([1.], [1000.], [-10.], [100.], [[100.]], [1.])
    assert not invalid["domain_valid"]
    with pytest.raises(ValueError, match="rest capacity"):
        prescribed_force_fatigue_challenge([1001.], [1000.], [-1.], [100.], [[1.]], [1.])


def test_work_credit_efficiency_symmetry_and_dummy_property():
    result = isokinetic_work_shapley([[2.], [2.], [0.]], [1.], 1.)
    assert result["mechanical_credit_j"] == pytest.approx([.5, .5, 0.])
    assert sum(result["mechanical_credit_j"]) == pytest.approx(1.)
    result = isokinetic_work_shapley([[.1], [.3], [.2]], [2.], 2.)
    assert result["mechanical_credit_j"] == pytest.approx([.2, .6, .4])
    assert not result["attainable_work_certified"]


def test_redundant_positive_muscle_never_gets_artificial_zero_or_minmax():
    result = isokinetic_work_shapley([[5.], [.01], [2.], [1.]], [1.], 1.)
    assert np.all(result["mechanical_credit_j"] > 0)
    assert sum(result["mechanical_credit_j"]) == pytest.approx(1.)


def test_work_credit_uses_time_quadrature_and_is_invariant_to_phase_refinement():
    coarse = isokinetic_work_shapley([[1., 2.], [.5, .2]], [.25, .75], 1.)
    fine = isokinetic_work_shapley([[1., 1., 2., 2.], [.5, .5, .2, .2]], [.125, .125, .375, .375], 1.)
    assert coarse["mechanical_credit_j"] == pytest.approx(fine["mechanical_credit_j"])


def test_proposal_limits_and_audit_do_not_claim_ocp_mutation_or_improvement():
    result = propose_physio_update(**fixture())
    assert result["status"] == "proposed"
    assert result["weights"].prod() == pytest.approx(1.)
    assert max(abs(np.log(result["weights"]))) <= math.log(1.1) + 1e-12
    assert np.all((result["weights"] >= .25) & (result["weights"] <= 4.))
    assert min(result["raw_max_normalized_scores"]) > 0
    assert not result["ocp_cost_updated"]
    assert not result["endurance_improvement_validated"]


@pytest.mark.parametrize("period", [10, 20])
def test_only_certified_block_boundaries_recompute(period):
    args = fixture()
    args["config"] = PhysioUpdateConfig(update_every_cycles=period)
    args["completed_cycles"] = period - 1
    args["current_capacity"] = None  # Unused outside a boundary.
    assert propose_physio_update(**args)["reason"] == "update_not_due"
    args["completed_cycles"] = period
    args["certified"] = False
    assert propose_physio_update(**args)["reason"] == "uncertified_source_state"


def test_no_drift_under_equal_physiology_and_mechanics():
    args = fixture()
    args["alpha_a"] = np.full(4, -.1)
    result = propose_physio_update(**args)
    assert result["status"] == "held"
    assert result["reason"] == "log_deadband"
    assert result["weights"] == pytest.approx(np.ones(4))


def test_mechanical_sensitivity_weights_match_the_diagonal_squared_fatigue_objective():
    result = propose_mechanical_sensitivity_weights(
        available_positive_power=np.array([[.2, .2], [.4, .4], [.6, .6], [.8, .8]]),
        phase_durations=np.array([.5, .5]), required_active_work=.7,
    )
    assert result["status"] == "proposed"
    assert result["weights"].prod() == pytest.approx(1.)
    assert result["raw_max_normalized_scores"].max() == pytest.approx(1.)
    # This proposal intentionally accepts no alpha_A: fatigue is already in
    # the propagated squared A-based objective and must not be squared again.
    assert "alpha_a" not in result


def test_mechanical_sensitivity_update_is_certified_bounded_and_causal():
    args = dict(
        available_positive_power=np.array([[.2, .2], [.4, .4], [.6, .6], [.8, .8]]),
        phase_durations=np.array([.5, .5]), required_active_work=.7,
        incumbent_weights=np.ones(4), completed_cycles=20, certified=True,
        config=PhysioUpdateConfig(update_every_cycles=20, smoothing=.2),
    )
    result = propose_mechanical_sensitivity_weights(**args)
    assert result["status"] == "proposed"
    assert result["weights"].prod() == pytest.approx(1.)
    assert max(abs(np.log(result["weights"]))) <= math.log(1.1) + 1e-12
    assert not result["endurance_improvement_validated"]
    assert propose_mechanical_sensitivity_weights(**{**args, "certified": False})["reason"] == "uncertified_source_state"
    assert propose_mechanical_sensitivity_weights(**{**args, "completed_cycles": 19})["reason"] == "update_not_due"


def test_muscle_permutation_is_equivariant():
    args = fixture()
    order = np.array([2, 0, 3, 1])
    original = propose_physio_update(**args)
    for name in ("incumbent_weights", "current_capacity", "rest_capacity", "alpha_a", "tau_fat",
                 "reference_force", "available_positive_power"):
        args[name] = args[name][order]
    permuted = propose_physio_update(**args)
    assert permuted["weights"] == pytest.approx(original["weights"][order])


def test_zero_score_and_out_of_domain_challenge_hold_incumbent():
    args = fixture()
    args["alpha_a"][0] = 0
    assert propose_physio_update(**args)["reason"] == "zero_or_invalid_physiological_score"
    args = fixture()
    args["current_capacity"][0] = 1
    args["reference_force"][0] = 1e6
    assert propose_physio_update(**args)["reason"] == "prescribed_force_challenge_outside_domain"


@pytest.mark.parametrize("field,value", [("update_every_cycles", True), ("smoothing", 2),
                                         ("max_log_step", 0), ("deadband_log", -1)])
def test_configuration_validation(field, value):
    with pytest.raises(ValueError):
        PhysioUpdateConfig(**{field: value})
