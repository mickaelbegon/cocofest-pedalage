import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from cocofest.cycling.slow import (
    OracleProvenance, SlowState, SlowStateBounds, propagate_prescribed_force,
    read_oracle_state, remaining_cycles,
)
from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters
from cocofest.optimization.ding_slow_cycle import slow_cycle_map_from_collocation


def fixture():
    p = DingFatigueParameters(4000, .06, .137, -.4, 2.1e-5, 1.9e-5, 127)
    provenance = OracleProvenance("witness-cycle-7.npz", "0"*64, "summary.json", "1"*64, "2"*64, 7)
    state = SlowState(("left", "right"), (3800, 3900), (.073, .062), (.142, .140), provenance)
    nodes = np.array([0, .2, .6, 1])
    maps = {name: slow_cycle_map_from_collocation([force + 20*nodes**2], [1], nodes, p)
            for name, force in zip(state.muscle_names, [50, 20])}
    bounds = SlowStateBounds(((3500, 0, 0), (3500, 0, 0)), ((np.inf, .2, .3), (np.inf, .2, .3)))
    return p, state, maps, bounds


def write_witness(tmp_path, certified=True):
    archive = tmp_path / "witness-cycle-7.npz"
    np.savez(archive, states__A_biceps=[[4000, 3990]], states__Tau1_biceps=[[.06, .061]],
             states__Km_biceps=[[.137, .138]])
    certificate = tmp_path / "summary.json"
    certificate.write_text(json.dumps({"cycles": [{"cycle": 7, "certified": certified}]}))
    return archive, certificate


def test_oracle_reads_exact_terminal_and_binds_provenance(tmp_path):
    archive, certificate = write_witness(tmp_path)
    state = read_oracle_state(archive, certificate, muscle_names=["biceps"], model_sha256="2"*64, cycle=7)
    np.testing.assert_array_equal(state.matrix, [[3990, .061, .138]])
    assert state.source == "simulator"
    assert state.provenance.archive_sha256 == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert state.provenance.certification_sha256 == hashlib.sha256(certificate.read_bytes()).hexdigest()
    assert replace(state, A=(3980,)).receipt_sha256 != state.receipt_sha256
    values = state.matrix
    values[0, 0] = 1
    assert state.A == (3990,)


@pytest.mark.parametrize("certified", [False, 1, "true", None])
def test_oracle_refuses_noncertified_records(tmp_path, certified):
    archive, certificate = write_witness(tmp_path, certified)
    with pytest.raises(ValueError, match="explicitly certified"):
        read_oracle_state(archive, certificate, muscle_names=["biceps"], model_sha256="2"*64, cycle=7)


def test_oracle_refuses_prepared_primal_and_bad_grid(tmp_path):
    archive, certificate = write_witness(tmp_path)
    with pytest.raises(ValueError, match="named solved witness"):
        read_oracle_state(tmp_path / "prepared.npz", certificate, muscle_names=["biceps"], model_sha256="2"*64, cycle=7)
    np.savez(archive, states__A_biceps=[4000, 3990], states__Tau1_biceps=[.06, .061, .062],
             states__Km_biceps=[.137, .138])
    with pytest.raises(ValueError, match="same node grid"):
        read_oracle_state(archive, certificate, muscle_names=["biceps"], model_sha256="2"*64, cycle=7)


def test_repeated_profile_against_independent_dop853_and_off_manifold_state():
    p, state, maps, _ = fixture()
    actual = propagate_prescribed_force(state, maps, 12)
    expected = []
    for row, force in zip(state.matrix, [50, 20]):
        for _ in range(12):
            solution = solve_ivp(lambda t, z: -(z-p.rest_state)/p.tau_fat+p.alpha*(force+20*t*t),
                                 (0, 1), row, method="DOP853", rtol=1e-12, atol=1e-13)
            assert solution.success
            row = solution.y[:, -1]
        expected.append(row)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert isinstance(actual, np.ndarray)  # A prediction is not an oracle observation.


def test_threshold_search_matches_direct_replay_and_reports_censoring():
    _, state, maps, bounds = fixture()
    current = state.matrix
    crossing = None
    for n in range(1, 301):
        current = np.stack([maps[name].propagate(row) for name, row in zip(state.muscle_names, current)])
        if np.any((current < bounds.lower) | (current > bounds.upper)):
            crossing = n
            break
    result = remaining_cycles(state, maps, bounds, horizon=300)
    assert result.first_exceedance_cycle == crossing
    assert result.cycles_within_bounds == crossing - 1
    assert result.limiting_muscles == ("left",)
    assert not result.censored
    assert "not_force_or_feasibility" in result.interpretation
    censored = remaining_cycles(state, maps, bounds, horizon=crossing-1)
    assert censored.first_exceedance_cycle is None
    assert censored.censored
    assert censored.cycles_within_bounds == crossing-1


def test_initial_violation_zero_horizon_and_inclusive_bounds():
    _, state, maps, bounds = fixture()
    initial = remaining_cycles(replace(state, A=(3400, 3900)), maps, bounds, horizon=100)
    assert initial.first_exceedance_cycle == initial.cycles_within_bounds == 0
    exact = SlowStateBounds(tuple(map(tuple, state.matrix)), tuple(map(tuple, state.matrix)))
    assert remaining_cycles(state, maps, exact, horizon=0).censored
    assert remaining_cycles(state, maps, exact, horizon=1).first_exceedance_cycle == 1


def test_recovery_crossing_upper_bound_and_no_fatigue_equilibrium():
    p, state, maps, _ = fixture()
    recovery = {name: slow_cycle_map_from_collocation([[0, 0]], [1], [0, 1], p) for name in maps}
    upper = SlowStateBounds(((0, 0, 0), (0, 0, 0)), ((3950, 1, 1), (3950, 1, 1)))
    result = remaining_cycles(state, recovery, upper, horizon=1000)
    expected = int(np.floor(-p.tau_fat*np.log((p.a_rest-3950)/(p.a_rest-state.A[1]))))+1
    assert result.first_exceedance_cycle == expected
    assert result.limiting_muscles == ("right",)
    unrestricted = SlowStateBounds(((0, 0, 0), (0, 0, 0)), ((np.inf, 1, 1), (np.inf, 1, 1)))
    assert remaining_cycles(state, recovery, unrestricted, horizon=10**7).censored


def test_invalid_contracts_rejected():
    _, state, maps, bounds = fixture()
    with pytest.raises(ValueError, match="source"):
        replace(state, source="prediction")
    with pytest.raises(ValueError, match="positive"):
        replace(state, Km=(np.nan, .1))
    with pytest.raises(ValueError, match="match"):
        propagate_prescribed_force(state, {"left": maps["left"]})
    with pytest.raises(ValueError, match="common cycle"):
        propagate_prescribed_force(state, dict(maps, right=replace(maps["right"], duration=2)))
    for value in (-1, True, 1.5):
        with pytest.raises(ValueError, match="horizon"):
            remaining_cycles(state, maps, bounds, horizon=value)
