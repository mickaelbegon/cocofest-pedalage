from dataclasses import replace

import numpy as np
import pytest

from cocofest.optimization.rho_tick_fallback import (
    FallbackContract, OneCycleCandidate, TickMeasurement, config_fingerprint,
    prepare_tick_fallback, solve_tick_fallback,
)


def case(step=100e-6, tick=3):
    contract = FallbackContract(
        muscles=("a", "b"), frequency_hz=30., intervals_per_cycle=120,
        phase_origin_rad=.2, time_origin_s=0., cycle_displacement_rad=-2 * np.pi,
        dynamics_fingerprint=config_fingerprint({"profile": "verified-sha", "load_nm": .1}),
        max_delta_pw_s=step,
    )
    names = tuple(reversed(contract.state_names))  # never assume row ordering
    measured = {name: 1. for name in names}
    measured.update(theta=-.4, omega=-1.7, Cn_a=.3, F_a=23., A_a=901., Tau1_a=.05, Km_a=.2)
    measurement = TickMeasurement(tick, tick / 30., measured, {"a": 300e-6, "b": 200e-6}, contract)
    rows, count = len(names), contract.intervals_per_cycle
    values = np.ones((rows, count + 1))
    lower, upper = np.full((rows, 3), -1e4), np.full((rows, 3), 1e4)
    target_start = contract.phase_origin_rad + tick / count * contract.cycle_displacement_rad
    target_end = target_start + contract.cycle_displacement_rad
    lower[names.index("theta"), 2] = target_end - .1
    upper[names.index("theta"), 2] = target_end + .1
    candidate = OneCycleCandidate(
        contract, tick, names, np.linspace(0, 4, count + 1), values, lower, upper,
        np.full(rows, -1e4), np.full(rows, 1e4), np.full((2, count), 500e-6),
        np.full((2, count), 100e-6), np.full((2, count), 600e-6),
        np.full(2, 100e-6), np.full(2, 600e-6),
        target_start, target_end, {"archive_sha256": "abc", "config_basis": "explicit"},
    )
    return candidate, measurement


def test_transfer_is_exact_named_full_state_and_keeps_unwrapped_reference_and_bounds():
    candidate, measurement = case(tick=243)
    measurement = replace(measurement, states=dict(measurement.states, theta=-13.2))
    initial_before = candidate.states.copy()
    request = prepare_tick_fallback(candidate, measurement)
    updated = request.candidate
    expected = [measurement.states[name] for name in candidate.state_names]
    np.testing.assert_array_equal(updated.states[:, 0], expected)
    np.testing.assert_array_equal(updated.state_lower[:, 0], expected)
    np.testing.assert_array_equal(updated.state_upper[:, 0], expected)
    np.testing.assert_array_equal(updated.state_lower[:, 1:], candidate.state_lower[:, 1:])
    np.testing.assert_array_equal(updated.state_upper[:, 1:], candidate.state_upper[:, 1:])
    np.testing.assert_array_equal(updated.states[:, 1:], candidate.states[:, 1:])
    np.testing.assert_array_equal(candidate.states, initial_before)
    assert updated.reference_end_theta == candidate.reference_end_theta
    assert request.provenance["global_phase_origin_rad"] == .2
    assert request.provenance["start_tick"] == 243
    assert request.provenance["duration_s"] == 4.
    assert request.provenance["source"] == candidate.source_provenance


def test_delta_pw_seam_uses_last_applied_command_and_lift_represents_new_commands():
    candidate, measurement = case()
    request = prepare_tick_fallback(candidate, measurement)
    np.testing.assert_allclose(request.candidate.pw_lower_s[:, 0], [200e-6, 100e-6])
    np.testing.assert_allclose(request.candidate.pw_upper_s[:, 0], [400e-6, 300e-6])
    # Carrier at START equals the first NEW command; previous PW only bounds it.
    np.testing.assert_allclose(request.carrier_s[:, 0], [400e-6, 300e-6])
    np.testing.assert_allclose(request.carrier_s[:, :-1], request.candidate.pw_s)
    assert np.max(abs(request.increments_s)) <= 100e-6 + 1e-18
    np.testing.assert_array_equal(request.increments_s[:, -1], 0.)
    np.testing.assert_array_equal(request.candidate.pw_lower_s[:, 1:], candidate.pw_lower_s[:, 1:])


@pytest.mark.parametrize("change,match", [
    ({"time_s": .10001}, "fixed stimulation tick"),
    ({"tick_index": 3.1}, "nonnegative integer"),
    ({"tick_index": True}, "nonnegative integer"),
    ({"states": {"theta": 0.}}, "complete named physical state"),
    ({"last_pw_s": {"a": .0003}}, "every muscle"),
])
def test_invalid_measurement_is_rejected_before_solver(change, match):
    candidate, measurement = case()
    def forbidden(_):
        pytest.fail("Must not invoke the solver on an invalid transfer")
    with pytest.raises(ValueError, match=match):
        solve_tick_fallback(candidate, replace(measurement, **change), solve=forbidden)


@pytest.mark.parametrize("field,value", [
    ("frequency_hz", 50.), ("max_delta_pw_s", None),
    ("dynamics_fingerprint", "a" * 64), ("phase_origin_rad", 0.),
])
def test_incompatible_source_configuration_frequency_or_delta_pw_rejected(field, value):
    candidate, measurement = case()
    other = replace(measurement.contract, **{field: value})
    with pytest.raises(ValueError, match="contracts differ"):
        prepare_tick_fallback(candidate, replace(measurement, contract=other))


@pytest.mark.parametrize("field,value", [
    ("formulation", "isokinetic"), ("mechanics", "full"),
    ("ding_states", "reduced"), ("calcium_forcing", "nonperiodic"),
])
def test_unsupported_formulations_fail_explicitly(field, value):
    candidate, measurement = case()
    with pytest.raises(ValueError, match="full five-state"):
        prepare_tick_fallback(replace(candidate, contract=replace(candidate.contract, **{field: value})), measurement)


def test_physical_limits_are_checked_without_clipping_measured_state():
    candidate, measurement = case()
    upper = candidate.physical_state_upper.copy()
    upper[candidate.state_names.index("A_a")] = 900.
    state_upper = candidate.state_upper.copy()
    state_upper[candidate.state_names.index("A_a"), :] = 900.
    with pytest.raises(ValueError, match="outside physical bounds"):
        prepare_tick_fallback(replace(candidate, physical_state_upper=upper, state_upper=state_upper), measurement)
    with pytest.raises(ValueError, match="physiological domain"):
        prepare_tick_fallback(candidate, replace(measurement, states=dict(measurement.states, Tau1_a=0.)))


def test_phase_must_not_be_reanchored_to_measured_theta():
    candidate, measurement = case()
    with pytest.raises(ValueError, match="global phase reference"):
        prepare_tick_fallback(replace(candidate, reference_start_theta=measurement.states["theta"]), measurement)


def test_infeasible_first_delta_pw_seam_fails_without_relaxing_mask():
    candidate, measurement = case()
    upper = candidate.pw_upper_s.copy()
    upper[0, 0] = 100e-6
    with pytest.raises(ValueError, match="no feasible delta-PW seam"):
        prepare_tick_fallback(replace(candidate, pw_upper_s=upper), measurement)


def test_no_delta_pw_does_not_add_an_auxiliary_lift():
    candidate, measurement = case(step=None)
    request = prepare_tick_fallback(candidate, measurement)
    assert request.carrier_s is None and request.increments_s is None
    np.testing.assert_array_equal(request.candidate.pw_lower_s, candidate.pw_lower_s)


def test_previous_pw_is_checked_against_physical_envelope_not_new_window_mask():
    candidate, measurement = case()
    # Previous 300 us is physically valid although new window mask caps all
    # controls at 250 us; the first delta-PW seam remains feasible.
    candidate = replace(candidate, pw_upper_s=np.full(candidate.pw_s.shape, 250e-6))
    request = prepare_tick_fallback(candidate, measurement)
    assert request.candidate.pw_upper_s[0, 0] == 250e-6


def test_invalid_path_bounds_are_not_silently_preserved_outside_physical_envelope():
    candidate, measurement = case()
    lower = candidate.state_lower.copy()
    lower[0, 1] = -2e4
    with pytest.raises(ValueError, match="respect physical limits"):
        prepare_tick_fallback(replace(candidate, state_lower=lower), measurement)


def test_injection_seam_transfers_request_once_and_propagates_solver_failure():
    candidate, measurement = case()
    calls = []
    marker = object()
    def solver(request):
        calls.append(request)
        return marker
    assert solve_tick_fallback(candidate, measurement, solve=solver) is marker
    assert len(calls) == 1
    assert calls[0].measurement == measurement
    def failed_solver(_):
        raise RuntimeError("Real backend failed")
    with pytest.raises(RuntimeError, match="Real backend failed"):
        solve_tick_fallback(candidate, measurement, solve=failed_solver)


def test_configuration_fingerprint_is_order_independent_but_load_sensitive():
    assert config_fingerprint({"load": .1, "muscle": 1}) == config_fingerprint({"muscle": 1, "load": .1})
    assert config_fingerprint({"load": .1}) != config_fingerprint({"load": .2})
    with pytest.raises(ValueError):
        config_fingerprint({"load": np.nan})
