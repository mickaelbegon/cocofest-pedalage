from dataclasses import replace

import numpy as np
import pytest

from cocofest.optimization.adaptive_moment_rollout import DingPulseWidthParameters, MomentTrackingInterval
from cocofest.optimization.compact_muscle_prediction import CompactMusclePredictor
from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters
from cocofest.optimization.pace_vr import (
    PaceVrConfig, PaceVrSupervisor, create_pace_vr_snapshot, cyclic_phase_basis,
    evaluate_pace_vr_weight_candidates, run_pace_vr_snapshot,
)


def case(*, horizon=3, work=.07, signed=False):
    parameters = tuple(DingPulseWidthParameters(
        DingFatigueParameters(1200., .060601, .137, -1.4 * factor, 2.1e-5, 1.9e-5, 445.5),
        .011, .001, .000131405, .000194138, .0006,
    ) for factor in (1., 1.7))
    intervals = tuple(MomentTrackingInterval(
        1 / 30, (1.0597355478,) * 2, (.95, .85),
        (.05, -.03 if signed else .03), (0., 0.),
    ) for _ in range(3))
    states = np.array([[.16298215835, 10., 1150., .067, .15], [.16298215835, 10., 1100., .068, .151]])
    widths = np.full((2, 3), .00030)
    supervisor = PaceVrSupervisor(
        intervals=intervals, pulse_width_parameters=parameters, angular_velocity_rad_s=1.,
        required_work_j=work, config=PaceVrConfig(horizon_cycles=horizon, phase_knots=2),
    )
    return supervisor, states, widths


def test_periodic_basis_partition_and_dimension():
    basis = cyclic_phase_basis(30, 4)
    assert basis.shape == (30, 4)
    assert np.all(basis >= 0)
    np.testing.assert_allclose(basis.sum(axis=1), 1.)


def test_cycle_map_reference_matches_compact_predictor_all_states():
    supervisor, states, widths = case()
    _, recruitment = supervisor._inputs(states, widths, True)
    expected = states.copy()
    for k in range(3):
        expected = supervisor.predictor.phase_map(expected, k).endpoint(recruitment[:, k])
    actual = supervisor._cycle(states, recruitment)
    np.testing.assert_allclose(actual["state"], expected, rtol=1e-13, atol=1e-12)
    assert actual["work"] > 0


@pytest.mark.parametrize("frequency", [30, 50])
@pytest.mark.parametrize("substeps", [1, 4, 8, 16])
@pytest.mark.parametrize("linearize", [False, True])
def test_condensed_cycle_preserves_reference_force_calcium_fatigue_work_and_constraints(
        frequency, substeps, linearize):
    original, states, widths = case(signed=True)
    intervals = tuple(replace(interval, duration=1 / frequency,
                              mechanical_gains=(.95, -.35 if phase == 1 else .85))
                      for phase, interval in enumerate(original.intervals))
    supervisor = PaceVrSupervisor(
        intervals=intervals, pulse_width_parameters=original.parameters,
        angular_velocity_rad_s=1., required_work_j=.03,
        power_lower_w=-5., power_upper_w=5.,
        config=replace(original.config, integration_substeps=substeps))
    _, recruitment = supervisor._inputs(states, widths, True)
    # Mixed signs exercise the reference OCP's signed mechanical gain without
    # clipping it, including inactive recruitment and fatigued off-manifold states.
    recruitment[0, 1] = 0.
    states[:, 2:] *= [.65, 1.2, 1.15]
    expected = supervisor._cycle_reference(states, recruitment, linearize=linearize)
    actual = supervisor._cycle(states, recruitment, linearize=linearize)
    for key in ("state", "derivative", "work", "work_gradient"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=2e-12, atol=2e-12)
    for actual_row, expected_row in zip(actual["force_rows"], expected["force_rows"]):
        for actual_value, expected_value in zip(actual_row, expected_row):
            np.testing.assert_allclose(actual_value, expected_value, rtol=2e-12, atol=2e-12)
    if linearize:
        for actual_value, expected_value in zip(supervisor._constraints(actual, recruitment),
                                               supervisor._constraints(expected, recruitment)):
            np.testing.assert_allclose(actual_value, expected_value, rtol=2e-12, atol=2e-12)


def test_condensed_rollout_preserves_h100_reference_and_terminal_fit():
    from types import MethodType

    supervisor, states, widths = case(horizon=100)
    reference, _, _ = case(horizon=100)
    reference.config = replace(reference.config, reduce_redundant_force_constraints=False)
    reference._cycle = MethodType(PaceVrSupervisor._cycle_reference, reference)
    expected = reference.fit_terminal(states, widths, certified=True)
    actual = supervisor.fit_terminal(states, widths, certified=True)
    assert actual["accepted"] == expected["accepted"]
    assert actual["feasible_prefix_cycles"] == expected["feasible_prefix_cycles"] == 100
    np.testing.assert_allclose(actual["state_history"], expected["state_history"], rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(actual["work_per_cycle_j"], expected["work_per_cycle_j"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(actual["local_fit"]["gradient"], expected["local_fit"]["gradient"],
                               rtol=1e-7, atol=1e-9)


def test_force_row_reduction_preserves_signed_gain_affine_feasible_set_and_capacity():
    original, states, widths = case(signed=True)
    gains = ((.95, -.85), (-.5, -.4), (-.8, .7))
    intervals = tuple(replace(interval, mechanical_gains=gain)
                      for interval, gain in zip(original.intervals, gains))
    fast = PaceVrSupervisor(intervals=intervals, pulse_width_parameters=original.parameters,
                            angular_velocity_rad_s=1., required_work_j=.03, config=original.config)
    full = PaceVrSupervisor(intervals=intervals, pulse_width_parameters=original.parameters,
                            angular_velocity_rad_s=1., required_work_j=.03,
                            config=replace(original.config, reduce_redundant_force_constraints=False))
    _, recruitment = fast._inputs(states, widths, True)
    recruitment[:] = .06
    model = fast._cycle(states, recruitment)
    a_small, lo_small, hi_small = fast._constraints(model, recruitment)
    a_full, lo_full, hi_full = full._constraints(model, recruitment)
    assert a_small.shape[0] < a_full.shape[0] / 2
    small_margin = fast._capacity_margin(model, a_small, lo_small, hi_small)
    full_margin = full._capacity_margin(model, a_full, lo_full, hi_full)
    assert small_margin == pytest.approx(full_margin, rel=1e-10, abs=1e-10)
    rng = np.random.default_rng(871)
    feasible = []
    for step in rng.uniform(-.06, .9, size=(400, fast.variables)):
        small = bool(np.all(a_small @ step >= lo_small - 1e-12)
                     and np.all(a_small @ step <= hi_small + 1e-12))
        full_valid = bool(np.all(a_full @ step >= lo_full - 1e-12)
                          and np.all(a_full @ step <= hi_full + 1e-12))
        assert small == full_valid
        feasible.append(small)
    assert any(feasible) and not all(feasible)


def test_force_reduction_falls_back_if_frozen_coefficients_are_outside_domain():
    supervisor, states, widths = case()
    _, recruitment = supervisor._inputs(states, widths, True)
    model = supervisor._cycle(states, recruitment)
    reduced_rows = supervisor._constraints(model, recruitment)[0].shape[0]
    model["force_constraint_reduction_safe"] = False
    full_rows = supervisor._constraints(model, recruitment)[0].shape[0]
    assert full_rows - reduced_rows == supervisor.phases * supervisor.predictor.substeps * supervisor.muscles


def test_slsqp_bound_failure_retries_complete_force_constraints_before_refusing(monkeypatch):
    import cocofest.optimization.pace_vr as module
    from scipy.optimize import OptimizeResult

    supervisor, states, widths = case(horizon=1)
    # This regression specifically exercises the historical SLSQP retry path;
    # the default test path separately covers the persistent qpOASES backend.
    supervisor.config = replace(supervisor.config, qp_backend="scipy")
    real_minimize = module.minimize
    rows = []

    def fail_first(*args, **kwargs):
        rows.append(kwargs["constraints"][0].A.shape[0])
        if len(rows) == 1:
            return OptimizeResult(x=np.full(supervisor.variables, 2.), success=False, message="forced bound failure")
        return real_minimize(*args, **kwargs)

    monkeypatch.setattr(module, "minimize", fail_first)
    result = supervisor.evaluate(states, widths, certified=True)
    assert result["accepted"]
    assert result["reallocation"]["full_constraint_retries"] >= 1
    assert rows[1] > rows[0]
    assert result["constraint_violation_max"] <= supervisor.config.constraint_tolerance


def test_capacity_envelope_is_only_solved_for_the_returned_sqp_model(monkeypatch):
    import cocofest.optimization.pace_vr as module

    supervisor, states, widths = case()
    real_linprog = module.linprog
    calls = []

    def count_envelopes(*args, **kwargs):
        calls.append(1)
        return real_linprog(*args, **kwargs)

    monkeypatch.setattr(module, "linprog", count_envelopes)
    result = supervisor.evaluate(states, widths, certified=True)
    assert result["accepted"]
    assert sum(result["reallocation"]["sqp_iterations"]) > 3
    assert len(calls) == result["feasible_prefix_cycles"] == 3


def test_affine_cycle_jacobian_exact_without_fatigue_coefficient_dependence():
    supervisor, states, widths = case()
    params = tuple(replace(p, fatigue=replace(p.fatigue, alpha_a=0., alpha_tau1=0., alpha_km=0.))
                   for p in supervisor.parameters)
    states[:, 2:] = np.array([p.fatigue.rest_state for p in params])
    supervisor = PaceVrSupervisor(intervals=supervisor.intervals, pulse_width_parameters=params,
                                  angular_velocity_rad_s=1., required_work_j=.07,
                                  config=supervisor.config)
    _, recruitment = supervisor._inputs(states, widths, True)
    base = supervisor._cycle(states, recruitment)
    for index in range(supervisor.variables):
        delta = np.zeros(supervisor.variables)
        delta[index] = 1e-5
        direction = (supervisor.command_map @ delta).reshape(recruitment.shape)
        plus = supervisor._cycle(states, recruitment + direction, linearize=False)
        minus = supervisor._cycle(states, recruitment - direction, linearize=False)
        np.testing.assert_allclose((plus["work"] - minus["work"]) / 2e-5,
                                   base["work_gradient"][index], atol=1e-10)


@pytest.mark.parametrize("signed", [False, True])
def test_work_constrained_rollout_reallocates_pw_and_preserves_ding_domain(signed):
    supervisor, states, widths = case(work=.03 if signed else .07, signed=signed)
    before = states.copy()
    result = supervisor.evaluate(states, widths, certified=True)
    assert result["accepted"], result["first_failure"]
    assert result["feasible_prefix_cycles"] == 3
    np.testing.assert_allclose(result["work_per_cycle_j"], supervisor.required_work_j, rtol=1e-5)
    assert result["constraint_violation_max"] <= 1e-8
    assert set(result["reallocation"]["qp_backends"]) == {"casadi_qpoases"}
    assert not result["context"]["physiological_failure_certified"]
    pw = np.asarray(result["pulse_widths"])
    assert np.max(np.abs(pw[1] - pw[0])) > 1e-8
    assert np.all(pw >= supervisor.predictor.pd0[None, :, None])
    assert np.all(pw <= supervisor.predictor.pulse_width_max[None, :, None] + 1e-15)
    np.testing.assert_array_equal(states, before)


def test_rejects_uncertified_input_and_impossible_task_without_false_failure_certificate():
    supervisor, states, widths = case(work=100.)
    with pytest.raises(ValueError, match="certified"):
        supervisor.evaluate(states, widths)
    result = supervisor.evaluate(states, widths, certified=True)
    assert not result["accepted"]
    assert result["feasible_prefix_cycles"] == 0
    assert result["minimum_task_margin"] < 0
    assert not result["context"]["physiological_failure_certified"]


def test_explicit_power_limit_is_honored_and_none_is_invented():
    original, states, widths = case()
    assert not original.evaluate(states, widths, certified=True)["context"]["power_bounds_supplied"]
    bounded = PaceVrSupervisor(intervals=original.intervals, pulse_width_parameters=original.parameters,
                               angular_velocity_rad_s=1., required_work_j=.07,
                               power_upper_w=.1, config=original.config)
    result = bounded.evaluate(states, widths, certified=True)
    assert not result["accepted"]
    assert result["context"]["power_bounds_supplied"]


def test_local_fit_is_finite_restricted_and_does_not_differentiate_qp_in_rho():
    supervisor, states, widths = case(horizon=2)
    result = supervisor.fit_terminal(states, widths, certified=True)
    fit = result["local_fit"]
    assert fit["accepted"], fit["reasons"]
    assert np.asarray(fit["gradient"]).shape == (2, 3)
    assert np.all(np.isfinite(fit["gradient"]))
    assert np.all(np.asarray(fit["diagonal_hessian"]) >= 0)
    np.testing.assert_array_equal(np.asarray(fit["gradient"])[:, 1:], 0.)
    assert len(fit["samples"]) == 2
    assert fit["fit_mode"] == "forward_directional"
    assert fit["fit_rank"] == 2
    assert "offsets_fixed" in fit["context"]


@pytest.mark.parametrize("kwargs", [{"horizon_cycles": 101}, {"horizon_cycles": True},
                                    {"fit_step": 0}, {"work_relative_tolerance": float("nan")}])
def test_configuration_domain(kwargs):
    with pytest.raises(ValueError):
        PaceVrConfig(**kwargs)


def test_worker_snapshot_is_immutable_context_checked_and_json_serializable():
    import json
    from dataclasses import asdict

    supervisor, states, widths = case(horizon=2)
    snapshot = create_pace_vr_snapshot(supervisor, states, widths, request_id="test-40",
                                      source_cycle=40, deadline_seconds=20, certified=True)
    states[:] = 0
    widths[:] = 0
    result = run_pace_vr_snapshot(json.loads(json.dumps(asdict(snapshot))))
    assert result["accepted"]
    assert result["request_id"] == "test-40"
    assert result["source_cycle"] == 40
    assert result["context_digest"] == snapshot.context_digest
    assert result["deadline_met"]
    assert result["applied_cycle"] is None
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match="digest mismatch"):
        run_pace_vr_snapshot(replace(snapshot, context_digest="corrupted"))


def test_weight_candidates_replay_the_same_certified_snapshot():
    supervisor, states, widths = case(horizon=2)
    snapshot = create_pace_vr_snapshot(supervisor, states, widths, request_id="candidate-40",
                                      source_cycle=40, deadline_seconds=20, certified=True)
    weights = {"proposal": [1.25, .8], "half_step": [np.sqrt(1.25), np.sqrt(.8)]}
    evaluated = evaluate_pace_vr_weight_candidates(snapshot, weights)
    for name, candidate_weights in weights.items():
        direct = supervisor.evaluate(states, widths, certified=True, weights=candidate_weights)
        assert evaluated[name]["accepted"] == direct["accepted"]
        assert evaluated[name]["feasible_prefix_cycles"] == direct["feasible_prefix_cycles"]
        assert evaluated[name]["terminal_value"] == pytest.approx(direct["terminal_value"], abs=1e-12)
        assert evaluated[name]["deadline_met"] is True
        assert evaluated[name]["started_monotonic"] <= evaluated[name]["completed_monotonic"]
    with pytest.raises(ValueError, match="digest mismatch"):
        evaluate_pace_vr_weight_candidates(replace(snapshot, context_digest="corrupted"), weights)


def test_expired_worker_result_is_refused_without_mutating_last_value():
    import json

    supervisor, states, widths = case(horizon=2)
    snapshot = create_pace_vr_snapshot(supervisor, states, widths, request_id="late",
                                      source_cycle=40, deadline_seconds=20, certified=True)
    result = run_pace_vr_snapshot(replace(snapshot, deadline_monotonic=0.))
    assert not result["accepted"]
    assert not result["deadline_met"]
    assert result["status"] == "deadline_expired"
    assert result["feasible_prefix_cycles"] == 0
    json.dumps(result, allow_nan=False)


def test_fit_budget_does_not_start_an_unaffordable_direction(monkeypatch):
    from time import monotonic

    supervisor, states, widths = case(horizon=2)
    base = supervisor.evaluate(states, widths, certified=True)
    calls = []

    def expensive_sample(*args, **kwargs):
        calls.append(1)
        return {**base, "runtime_s": 100.}

    monkeypatch.setattr(supervisor, "evaluate", expensive_sample)
    result = supervisor.fit_terminal(states, widths, certified=True, deadline_monotonic=monotonic() + 1.)
    assert len(calls) == 1
    assert not result["local_fit"]["accepted"]
    assert result["local_fit"]["budget_limited"]
    assert result["local_fit"]["fit_mode"] == "constant_only"
    assert result["local_fit"]["sample_count"] == 0


def test_directional_fit_hard_sample_cap_is_explicit():
    supervisor, states, widths = case(horizon=2)
    supervisor.config = replace(supervisor.config, fit_max_samples=1)
    result = supervisor.fit_terminal(states, widths, certified=True)
    fit = result["local_fit"]
    assert fit["accepted"]
    assert fit["sample_count"] == fit["fit_rank"] == 1
    assert np.sum(np.asarray(fit["gradient"])[:, 0]) == pytest.approx(0., abs=1e-10)


def test_unreachable_base_refuses_terminal_fit_even_if_all_fd_branches_fail_identically():
    supervisor, states, widths = case(horizon=2, work=100.)
    result = supervisor.fit_terminal(states, widths, certified=True)
    assert not result["local_fit"]["accepted"]
    assert "base_rollout_incomplete" in result["local_fit"]["reasons"]


def test_exponential_work_integral_against_independent_dop853_ding_replay():
    from scipy.integrate import solve_ivp
    from cocofest.optimization.adaptive_moment_rollout import _four_state_rhs, periodic_calcium_state

    supervisor, initial, widths = case(horizon=3)
    result = supervisor.evaluate(initial, widths, certified=True)
    current = initial.copy()
    reference_work = []
    for schedule in result["pulse_widths"]:
        work = 0.
        for phase, interval in enumerate(supervisor.intervals):
            for muscle, parameter in enumerate(supervisor.parameters):
                cn0 = current[muscle, 0]

                def rhs(time, state):
                    return np.r_[_four_state_rhs(
                        time, state[:4], initial_cn=cn0,
                        pulse_width=schedule[muscle][phase],
                        calcium_amplitude=interval.calcium_amplitudes[muscle],
                        mechanical_gain=interval.mechanical_gains[muscle], parameters=parameter,
                    ), state[0]]

                integrated = solve_ivp(rhs, (0., interval.duration), np.r_[current[muscle, 1:], 0.],
                                       method="DOP853", rtol=1e-11, atol=1e-12).y[:, -1]
                current[muscle, 0] = periodic_calcium_state(
                    cn0, interval.duration, interval.calcium_amplitudes[muscle], parameter.tauc)
                current[muscle, 1:] = integrated[:4]
                work += supervisor.power_coefficients[phase, muscle] * integrated[4]
        reference_work.append(work)
    # The compact map is explicitly approximate; this bounds its error on a
    # fatigue-active, 30-Hz synthetic case, not on arbitrary clinical geometry.
    np.testing.assert_allclose(reference_work, result["work_per_cycle_j"], rtol=.01, atol=0.)
    np.testing.assert_allclose(current[:, 2:], np.asarray(result["terminal_state"]), rtol=.001)
