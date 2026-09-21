"""Scientific regression checks for continuous cross-integrator evaluation."""

import json
import numpy as np
import pytest

from cocofest.optimization.solver_cross_rollout import (
    CommonCost, RolloutSource, collocation_tableau, compatibility,
    evaluate_rollout, implicit_step, load_source,
    _physical_rhs, ranking_physical_gate, qualified_velocity_comparability_gate,
)


PARAMETERS = dict(Fmax=100., a_scale=1000., alpha_a=0., tau_fat=179.6,
                  alpha_tau1=2.1e-5, alpha_km=1.9e-5, tau1_rest=.060601,
                  km_rest=.137, tauc=.011, tau2=.001, pd0=.000131405, pdt=.000194138)


class Profile:
    muscle_names = ("test",)

    def acceleration(self, theta, omega, forces, external_crank_torque=0):
        return 0.

    def coefficient_values(self, theta):
        return dict(external_torque_effectiveness=1., muscle_effectiveness=np.array([0.]),
                    projected_gravity=0., projected_velocity_quadratic=0.)


def make_source(cycles=1):
    state = np.array([.25, 0., 800., .060601, .137, 0., -2*np.pi])
    metadata = dict(model_formulation="periodic_node", mechanical_formulation="reduced",
                    formulation="dynamic", torque_application="constant", constant_crank_torque=0.,
                    activate_force_length_relationship=False, activate_force_velocity_relationship=False,
                    activate_passive_force_relationship=False, ding_sum_stim_truncation=6,
                    pulse_width_maximum_s=.0006, stimulations_per_cycle=2,
                    calcium_forcing_formulation="exact_exponential_periodic_node")
    names = tuple(f"{c}_test" for c in ("Cn", "F", "A", "Tau1", "Km")) + ("theta", "omega")
    return RolloutSource(metadata, {}, ("test",), {"test": PARAMETERS.copy()},
                         np.full((1, 2*cycles), PARAMETERS["pd0"]), np.tile(state[:, None], (1, 2*cycles+1)),
                         names, .04, 2, 0, cycles)


@pytest.mark.parametrize("method,stages", [("radau5", 5), ("gauss-legendre4x5", 4)])
def test_tableau_exactness_and_linear_endpoint(method, stages):
    c, a, b, _ = collocation_tableau(method)
    assert len(c) == stages
    assert np.allclose(a.sum(axis=1), c)
    for degree in range(stages):
        assert b @ c**degree == pytest.approx(1/(degree+1), abs=1e-12)
    end, dense, _, defect = implicit_step(lambda _t, x: -2*x, np.array([1.]), 0., .1, method, np.ones(1))
    assert end[0] == pytest.approx(np.exp(-.2), abs=2e-11)
    assert dense(np.array([0., .1]))[0] == pytest.approx([1., end[0]], abs=1e-12)
    assert defect < 1e-10


def test_continuous_dop853_fatigue_cost_matches_analytic_integral_and_does_not_reset():
    source = make_source(cycles=2)
    metrics, arrays = evaluate_rollout(source, Profile())
    total, tau = .08, PARAMETERS["tau_fat"]
    expected = 10000*.2**2*tau/2*(1-np.exp(-2*total/tau))/.04
    assert metrics["common_score"] == pytest.approx(expected, rel=2e-11)
    assert arrays["shooting_states"][2, -1] == pytest.approx(1000-200*np.exp(-total/tau), rel=1e-11)
    source.shooting_states[:, 1:] = 123.  # deliberately corrupt optimized observations
    same_metrics, same_arrays = evaluate_rollout(source, Profile())
    assert same_metrics["common_score"] == metrics["common_score"]
    assert np.array_equal(arrays["states"], same_arrays["states"])


@pytest.mark.parametrize("method", ["radau5", "gauss-legendre4x5"])
def test_implicit_maps_agree_with_reference_for_fatigue_cost(method):
    source = make_source()
    reference, ref_arrays = evaluate_rollout(source, Profile())
    candidate, arrays = evaluate_rollout(source, Profile(), evaluator=method)
    assert candidate["common_score"] == pytest.approx(reference["common_score"], rel=1e-10)
    assert np.max(np.abs(arrays["shooting_states"][2:]-ref_arrays["shooting_states"][2:])) < 1e-9


def test_dop853_preserves_exact_calcium_boundary_history():
    source = make_source()
    _, arrays = evaluate_rollout(source, Profile())
    p, h = PARAMETERS, source.dt
    decay = np.exp(-h/p["tauc"])
    amplitude = decay**5+(1+(p["km_rest"]+.04)*decay)*sum(decay**k for k in range(5))
    cn = source.initial_state[0]
    for value in arrays["shooting_states"][0, 1:]:
        cn = decay*(cn+amplitude*h/p["tauc"])
        assert value == pytest.approx(cn, rel=1e-10)


def test_compatibility_blocks_changed_initial_state_and_model():
    a, b = make_source(), make_source()
    assert compatibility([a, b])["comparable"]
    b.shooting_states[2, 0] += 1.
    assert not compatibility([a, b])["comparable"]
    b.parameters["test"]["alpha_a"] = -1.
    assert any("model parameters" in reason for reason in compatibility([a, b])["reasons"])


def test_loader_refuses_guessing_legacy_parameters_and_duration(tmp_path):
    source = make_source()
    metadata = dict(source.metadata)
    payload = {f"states__{key}": row[None, :] for key, row in zip(source.state_names, source.shooting_states)}
    payload["controls__last_pulse_width_test"] = source.controls
    path = tmp_path/"source.npz"
    np.savez(path, **payload, metadata__json=np.array(json.dumps(metadata)))
    with pytest.raises(ValueError, match="duration"):
        load_source(path, Profile())
    with pytest.raises(ValueError, match="parameters"):
        load_source(path, Profile(), cycle_duration=.04)
    metadata["configured_muscle_parameters"] = source.parameters
    np.savez(path, **payload, metadata__json=np.array(json.dumps(metadata)))
    loaded = load_source(path, Profile(), cycle_duration=.04)
    assert np.array_equal(loaded.initial_state, source.initial_state)
    assert loaded.provenance["parameters_basis"] == "embedded_parameters"


def test_loader_requires_an_explicit_audited_formulation_override_for_legacy_archive(tmp_path):
    source = make_source()
    metadata = dict(source.metadata)
    metadata.pop("formulation")
    metadata["configured_muscle_parameters"] = source.parameters
    payload = {f"states__{key}": row[None, :] for key, row in zip(source.state_names, source.shooting_states)}
    payload["controls__last_pulse_width_test"] = source.controls
    path = tmp_path / "legacy.npz"
    np.savez(path, **payload, metadata__json=np.array(json.dumps(metadata)))
    with pytest.raises(ValueError, match="declare dynamic"):
        load_source(path, Profile(), cycle_duration=.04)
    loaded = load_source(path, Profile(), cycle_duration=.04, formulation_override="dynamic")
    assert loaded.metadata["formulation"] == "dynamic"
    assert loaded.provenance["formulation_basis"] == "explicit_legacy_override"
    metadata["formulation"] = "dynamic"
    np.savez(path, **payload, metadata__json=np.array(json.dumps(metadata)))
    with pytest.raises(ValueError, match="conflicts"):
        load_source(path, Profile(), cycle_duration=.04, formulation_override="isokinetic")


def test_out_of_bounds_controls_fail_without_clipping():
    source = make_source()
    source.controls[0, 0] = 0.
    with pytest.raises(ValueError, match="pulse widths violate"):
        evaluate_rollout(source, Profile())


def test_common_cost_refuses_invalid_weights():
    with pytest.raises(ValueError):
        CommonCost(fatigue_weight=-1)
    with pytest.raises(ValueError):
        CommonCost(fatigue_shape="unknown")


def test_phase_and_velocity_metrics_use_declared_bounds():
    source = make_source()
    source.metadata.update(terminal_wheel_q_slack=.002, reduced_internal_crank_velocity_guard=True,
                           reduced_internal_crank_velocity_guard_target_rad_s=-5.,
                           reduced_internal_crank_velocity_guard_fast_margin_rad_s=.1,
                           reduced_internal_crank_velocity_guard_slow_margin_rad_s=.1)
    report, _ = evaluate_rollout(source, Profile())
    metrics = report["physical_metrics"]
    assert metrics["sampled_velocity_bound_violation_rad_s"] == pytest.approx(2*np.pi-5.1)
    assert metrics["sampled_phase_bound_violation_rad"] > 6.
    assert metrics["full_nlp_certified"] is False


def test_ranking_gate_refuses_sampled_dop853_constraint_violation():
    feasible = {"status": "success", "physical_metrics": {
        "physical_pw_bound_violation_s": 0.0,
        "sampled_phase_bound_violation_rad": 0.0,
        "sampled_velocity_bound_violation_rad_s": 0.0,
    }}
    assert ranking_physical_gate([feasible, feasible])["rankable"]
    infeasible = {"status": "success", "physical_metrics": {
        "sampled_velocity_bound_violation_rad_s": 1e-5,
    }}
    gate = ranking_physical_gate([feasible, infeasible])
    assert not gate["rankable"]
    assert "source 1" in gate["reasons"][0]


def test_qualified_gate_allows_small_similar_dop853_velocity_errors_only():
    entries = [
        {"status": "success", "physical_metrics": {"sampled_velocity_bound_violation_rad_s": 0.005}},
        {"status": "success", "physical_metrics": {"sampled_velocity_bound_violation_rad_s": 0.0075}},
    ]
    gate = qualified_velocity_comparability_gate(entries)
    assert gate["rankable"]
    assert gate["observed_velocity_violation_ratio"] == pytest.approx(1.5)

    not_comparable = qualified_velocity_comparability_gate([
        entries[0],
        {"status": "success", "physical_metrics": {"sampled_velocity_bound_violation_rad_s": 0.011}},
    ])
    assert not not_comparable["rankable"]


def test_vector_field_matches_native_periodic_node_model():
    ca = pytest.importorskip("casadi")
    pytest.importorskip("bioptim")
    from cocofest.models.ding2007.ding2007_with_fatigue_periodic_node import (
        DingModelPulseWidthFrequencyWithFatiguePeriodicNode,
    )
    from cocofest.optimization.configured_cycling_model import FIELD_ATTRIBUTES

    source = make_source()
    model = DingModelPulseWidthFrequencyWithFatiguePeriodicNode(
        muscle_name="test", stim_time=[0., source.dt], sum_stim_truncation=6,
    )
    for key, attribute in FIELD_ATTRIBUTES.items():
        setattr(model, attribute, PARAMETERS[key])
    model.a_rest = model.a_scale
    state = source.initial_state
    state[1] = 12.3
    control, time = np.array([.0003]), .013
    expected = model.system_dynamics(states=ca.DM(state[:5]), controls=ca.DM(control),
                                     time=ca.DM(time), numerical_timeseries=ca.DM([model.post_stimulation_amplitude(), 0.]))
    observed = _physical_rhs(source, Profile(), control)(time, state)
    assert observed[:5] == pytest.approx(np.array(expected).ravel(), rel=2e-14, abs=1e-12)
