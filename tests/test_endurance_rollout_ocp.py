"""Structural tests only: synthetic policies do not certify endurance gains."""

from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import casadi as ca
import numpy as np
import pytest

from bioptim import (
    BoundsList, ConfigureVariables, ConstraintList, DynamicsEvaluation, DynamicsOptions,
    InitialGuessList, Node, ObjectiveFcn, OptimalControlProgram, Solver, StateDynamics,
)
from bioptim.optimization.receding_horizon_optimization import RecedingHorizonOptimization
from cocofest import CustomObjective
from cocofest.optimization.endurance_rollout_ocp import (
    CertifiedRolloutParameters, EnduranceRolloutBinding, EnduranceRolloutOptions,
    ROLLOUT_ARTIFACT_SCHEMA, ROLLOUT_PARAMETER_KEY,
    endurance_rollout_signature_fields, resolve_endurance_rollout_options,
)
from cocofest.optimization.endurance_rollout_objective import (
    ROLLOUT_DOMAIN_MARGIN_NAMES, RolloutObjectiveLayout, pack_rollout_objective_parameters,
    rollout_domain_lower_bounds,
)
from examples.fes_multibody.cycling import cycling_pulse_width_mhe as mhe
from examples.fes_multibody.cycling import cycling_fes_solver_comparison as comparison
from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as periodic
from tests.test_endurance_rollout_objective import _muscles, _profile


def _payload(force_offset=0.0):
    layout = RolloutObjectiveLayout(2, 4, 2)
    payload = {
        "schema": ROLLOUT_ARTIFACT_SCHEMA,
        "method": "synthetic_test_fixture_only",
        "layout": asdict(layout), "muscles": [asdict(item) for item in _muscles()],
        "muscle_names": ["m0", "m1"],
        "profile_parameters": pack_rollout_objective_parameters(_profile(force_offset), _muscles(), layout).tolist(),
        "adapter_report": {
            "schema": "cocofest-rho-endurance-rollout-v2",
            "fixture_only": True,
            "status": "complete", "adapted_policy_fidelity": {"passed": True},
            "source": {"sha256": "0" * 64, "path": "synthetic_test_fixture_only"},
            "muscle_names": ["m0", "m1"],
        },
    }
    report = payload["adapter_report"]
    report["configuration"] = {"policy_representation": payload["method"]}
    report["rollout_objective_profile"] = {
        "schema": "cocofest-rollout-objective-profile-v1", "source_policy_gate_passed": True,
        "horizon_independent": True, "muscle_count": 2, "interval_count": 4,
        "parameter_size": layout.parameter_size, "parameters": list(payload["profile_parameters"]),
    }
    report["model_parameters"] = []
    for name, muscle in zip(payload["muscle_names"], _muscles()):
        fatigue = muscle.fatigue
        report["model_parameters"].append({
            "muscle": name, "A_rest": fatigue.a_rest, "Tau1_rest": fatigue.tau1_rest,
            "Km_rest": fatigue.km_rest, "alpha_A": fatigue.alpha_a,
            "alpha_Tau1": fatigue.alpha_tau1, "alpha_Km": fatigue.alpha_km, "tau_fat": fatigue.tau_fat,
            **{key: getattr(muscle, key) for key in ("tau2", "pd0", "pdt", "pulse_width_max")},
        })
    return payload


def _binding(weight=0.1):
    return EnduranceRolloutBinding(EnduranceRolloutOptions(
        CertifiedRolloutParameters.from_payload(_payload()), weight, 1e-8
    ))


def test_zero_weight_ignores_missing_profile_and_preserves_all_objective_entries():
    args = SimpleNamespace(endurance_rollout_weight=0.0, experimental_endurance_rollout=True,
                           endurance_rollout_profile="missing_profile.json")
    assert resolve_endurance_rollout_options(args) is None
    assert endurance_rollout_signature_fields(args) == {}
    model = SimpleNamespace(muscles_dynamics_model=[])
    kwargs = dict(model=model, minimize_force=False, minimize_fatigue=True, minimize_control=False,
                  cost_fun_weight=[0, 1, 0], target=0.0)
    baseline = mhe.set_objective_functions(**kwargs)
    disabled = mhe.set_objective_functions(**kwargs, endurance_rollout_binding=_binding(weight=0.0))
    for left, right in zip(baseline[0], disabled[0], strict=True):
        assert left.custom_function is right.custom_function
        assert left.type is right.type
        assert left.node == right.node
        assert left.quadratic == right.quadratic
        np.testing.assert_array_equal(left.weight, right.weight)
        assert left.extra_parameters == right.extra_parameters


@pytest.mark.parametrize("status, passed", [("rejected", True), ("complete", False), ("incomplete", True)])
def test_incomplete_fidelity_gate_never_builds_profile(status, passed):
    payload = _payload()
    payload["adapter_report"]["status"] = status
    payload["adapter_report"]["adapted_policy_fidelity"]["passed"] = passed
    with pytest.raises(ValueError, match="formulation_unavailable"):
        CertifiedRolloutParameters.from_payload(payload)


def test_explicit_opt_in_and_valid_domain_epsilon_required(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(_payload()))
    args = SimpleNamespace(endurance_rollout_weight=0.1, endurance_rollout_profile=path)
    with pytest.raises(ValueError, match="experimental-endurance-rollout"):
        resolve_endurance_rollout_options(args)
    args.experimental_endurance_rollout = True
    args.endurance_rollout_domain_epsilon = 0.0
    with pytest.raises(ValueError, match="epsilon"):
        resolve_endurance_rollout_options(args)


def test_compilation_signature_is_structural_but_seed_signature_tracks_numerical_profile():
    first = _binding().options
    second = replace(first, profile=CertifiedRolloutParameters.from_payload(_payload(2.0)))
    assert first.metadata(structure_only=True) == second.metadata(structure_only=True)
    assert first.metadata() != second.metadata()
    assert first.metadata(structure_only=True) != replace(first, domain_epsilon=1e-7).metadata(structure_only=True)
    assert first.metadata(structure_only=True) != replace(first, weight=0.2).metadata(structure_only=True)

    args = periodic.build_argument_parser().parse_args([])
    args.endurance_rollout_weight = first.weight
    args._endurance_rollout_options = first
    initial_codegen = periodic._codegen_signature(args)
    initial_seed = periodic._horizon_seed_cache_signature(args)
    args._endurance_rollout_options = second
    assert periodic._codegen_signature(args) == initial_codegen
    assert periodic._horizon_seed_cache_signature(args) != initial_seed


def test_adapter_report_ingestion_preserves_packed_values_and_records_provenance():
    payload = _payload()
    report = payload["adapter_report"]
    report["configuration"] = {"policy_representation": "collocation"}
    report["rollout_objective_profile"] = {
        "schema": "cocofest-rollout-objective-profile-v1", "source_policy_gate_passed": True,
        "horizon_independent": True, "muscle_count": 2, "interval_count": 4,
        "parameter_size": len(payload["profile_parameters"]), "parameters": payload["profile_parameters"],
    }
    report["model_parameters"] = []
    for name, muscle in zip(payload["muscle_names"], _muscles()):
        fatigue = muscle.fatigue
        report["model_parameters"].append({
            "muscle": name, "A_rest": fatigue.a_rest, "Tau1_rest": fatigue.tau1_rest,
            "Km_rest": fatigue.km_rest, "alpha_A": fatigue.alpha_a,
            "alpha_Tau1": fatigue.alpha_tau1, "alpha_Km": fatigue.alpha_km, "tau_fat": fatigue.tau_fat,
            **{key: getattr(muscle, key) for key in ("tau2", "pd0", "pdt", "pulse_width_max")},
        })
    profile = CertifiedRolloutParameters.from_adapter_report(report, horizon_cycles=10)
    assert profile.layout.horizon_cycles == 10
    np.testing.assert_array_equal(profile.parameters, payload["profile_parameters"])
    assert profile.provenance["adapter_status"] == "complete"
    assert profile.provenance["method"] == "collocation"
    report["adapted_policy_fidelity"]["passed"] = False
    with pytest.raises(ValueError, match="formulation_unavailable"):
        CertifiedRolloutParameters.from_adapter_report(report)


def test_mayer_binding_is_scalar_terminal_and_nonquadratic():
    binding = _binding()
    objectives = mhe.set_objective_functions(
        SimpleNamespace(muscles_dynamics_model=[]), False, True, False, [0, 1, 0], 0.0,
        endurance_rollout_binding=binding,
    )
    entry = objectives[0][0]
    assert entry.custom_function is CustomObjective.minimize_terminal_endurance_rollout
    assert entry.type == ObjectiveFcn.Mayer.CUSTOM
    assert entry.node == Node.END
    assert entry.quadratic is False
    assert float(entry.weight[0]) == pytest.approx(1000.0)


def test_model_validation_uses_actual_control_bounds_instead_of_a_hardcoded_pw_cap():
    profile = _binding().options.profile
    models, bounds = [], BoundsList()
    for name, muscle in zip(profile.muscle_names, profile.muscles):
        models.append(SimpleNamespace(
            muscle_name=name, **asdict(muscle.fatigue),
            tau2=muscle.tau2, pd0=muscle.pd0, pdt=muscle.pdt,
        ))
        bounds[f"last_pulse_width_{name}"] = [muscle.pd0], [muscle.pulse_width_max]
    model = SimpleNamespace(muscles_dynamics_model=models)
    profile.validate_model(model, control_bounds=bounds)
    bounds["last_pulse_width_m0"].max[:] = 0.0005
    with pytest.raises(ValueError, match="pulse-width bounds"):
        profile.validate_model(model, control_bounds=bounds)
    bounds["last_pulse_width_m0"].max[0, 1] = 0.0004
    with pytest.raises(ValueError, match="phase-independent"):
        profile.validate_model(model, control_bounds=bounds)


def test_both_public_clis_default_to_an_absent_rollout():
    for parser in (comparison.build_cli(), periodic.build_argument_parser()):
        args = parser.parse_args([])
        assert args.experimental_endurance_rollout is False
        assert args.endurance_rollout_weight == 0.0
        assert resolve_endurance_rollout_options(args) is None


@pytest.mark.parametrize("field", ["profile_parameters", "muscles", "method"])
def test_payload_cannot_replace_data_covered_by_a_certified_report(field):
    payload = _payload()
    if field == "profile_parameters":
        payload[field][0] += 1.0
    elif field == "muscles":
        payload[field][0]["fatigue"]["a_rest"] += 1.0
    else:
        payload[field] = "unrelated_method"
    with pytest.raises(ValueError, match="must match the certified report"):
        CertifiedRolloutParameters.from_payload(payload)


def test_payload_without_a_certified_packed_profile_is_rejected():
    payload = _payload()
    del payload["adapter_report"]["rollout_objective_profile"]
    with pytest.raises(ValueError, match="packed rollout profile"):
        CertifiedRolloutParameters.from_payload(payload)


def test_dataclass_replacement_cannot_bypass_certified_profile_binding():
    profile = _binding().options.profile
    with pytest.raises(ValueError, match="must match the certified report"):
        replace(profile, parameters=(profile.parameters[0] + 1.0, *profile.parameters[1:]))


def test_cli_rejects_wrapped_payload_and_applies_horizon_and_temperature_to_report(tmp_path):
    path = tmp_path / "profile.json"
    args = SimpleNamespace(
        endurance_rollout_weight=0.1, experimental_endurance_rollout=True,
        endurance_rollout_profile=path, endurance_rollout_horizon_cycles=7,
        endurance_rollout_temperature=0.035,
    )
    path.write_text(json.dumps(_payload()))
    with pytest.raises(ValueError, match="requires a complete adapter report"):
        resolve_endurance_rollout_options(args)
    path.write_text(json.dumps(_payload()["adapter_report"]))
    options = resolve_endurance_rollout_options(args)
    assert options.profile.layout.horizon_cycles == 7
    assert options.profile.layout.smooth_max_temperature == 0.035


def test_closed_and_strict_domain_bounds_follow_the_exact_symbolic_order():
    binding = _binding()
    expected = np.array([1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0]) * 1e-8
    assert ROLLOUT_DOMAIN_MARGIN_NAMES == (
        "midpoint_A", "midpoint_rest_minus_A", "midpoint_Tau1_minus_rest", "midpoint_Km_minus_rest",
        "midpoint_Tau1", "midpoint_Km_plus_Cn", "Cn", "mechanical_gain", "relaxation_time",
        "required_recruitment", "maximum_recruitment", "endpoint_A", "endpoint_rest_minus_A",
        "endpoint_Tau1_minus_rest", "endpoint_Km_minus_rest",
    )
    lower = binding.domain_lower_bounds
    assert lower.size == binding.function.size1_out("domain_margins") == 15 * 2 * 4 * 2
    np.testing.assert_array_equal(lower, np.tile(expected, 2 * 4 * 2))


@pytest.mark.parametrize("key, mismatch", [
    ("cycle_period_s", 1.0), ("signed_crank_torque_nm", 0.22),
    ("mechanical_formulation", "full"), ("formulation", "isokinetic"),
])
def test_explicit_profile_ocp_context_mismatch_is_rejected(key, mismatch):
    report = _payload()["adapter_report"]
    report["selection"] = {"cycle_period_s": 0.8}
    report["source_ocp_context"] = {
        "signed_crank_torque_nm": 0.1, "mechanical_formulation": "reduced", "formulation": "dynamic",
    }
    profile = CertifiedRolloutParameters.from_adapter_report(report)
    context = {"cycle_period_s": 0.8, **report["source_ocp_context"]}
    assert all(item["status"] == "matched" for item in profile.validate_context(**context).values())
    context[key] = mismatch
    with pytest.raises(ValueError, match=f"context mismatch for {key}"):
        profile.validate_context(**context)


def test_absent_profile_context_is_explicitly_not_verifiable():
    binding = _binding()
    binding.validate_context(cycle_period_s=1.0, signed_crank_torque_nm=0.22,
                             mechanical_formulation="reduced", formulation="dynamic")
    assert all(item["status"] == "not_verifiable" for item in binding.summary()["ocp_context_compatibility"].values())
    assert set(binding.options.profile.provenance["context_metadata_status"].values()) == {"not_verifiable"}


class _ConstantSlowStateModel(StateDynamics):
    """Small Bioptim transcription used only to exercise the real solver path."""

    def __init__(self):
        super().__init__()
        self.muscles_dynamics_model = [
            SimpleNamespace(muscle_name=f"m{i}", a_scale=muscle.fatigue.a_rest)
            for i, muscle in enumerate(_muscles())
        ]

    def serialize(self):
        return _ConstantSlowStateModel, {}

    @property
    def name(self):
        return "synthetic_rollout_binding_test"

    @property
    def name_dofs(self):
        return ["dummy"]

    @property
    def state_configuration_functions(self):
        def configure(ocp, nlp):
            for muscle in ("m0", "m1"):
                for field in ("A", "Tau1", "Km"):
                    ConfigureVariables.configure_new_variable(
                        f"{field}_{muscle}", [f"{field}_{muscle}"], ocp, nlp, as_states=True
                    )
        return [configure]

    @property
    def control_configuration_functions(self):
        return [lambda ocp, nlp: ConfigureVariables.configure_new_variable(
            "dummy", ["dummy"], ocp, nlp, as_controls=True
        )]

    @property
    def algebraic_configuration_functions(self):
        return []

    @property
    def extra_configuration_functions(self):
        return []

    def dynamics(self, time, states, controls, parameters, algebraic_states, numerical_timeseries, nlp):
        return DynamicsEvaluation(dxdt=0 * states, defects=None)


def _tiny_ocp(binding=None, *, receding=False, initial=None):
    model = _ConstantSlowStateModel()
    n_shooting = 2
    x_bounds, x_init, u_bounds, u_init = BoundsList(), InitialGuessList(), BoundsList(), InitialGuessList()
    initial = [1450.0, 0.065, 0.142, 1620.0, 0.066, 0.143] if initial is None else initial
    index = 0
    for muscle in ("m0", "m1"):
        for field in ("A", "Tau1", "Km"):
            key = f"{field}_{muscle}"
            value = initial[index]
            x_bounds[key] = [[value, value * 0.9, value * 0.9]], [[value, value * 1.1, value * 1.1]]
            x_init[key] = [value]
            index += 1
    u_bounds["dummy"] = [-1], [1]
    u_init["dummy"] = [0]
    objective = mhe.set_objective_functions(
        model, False, True, False, [0, 1, 0], 0.0,
        terminal_wheel_regularization_weight=0.0, endurance_rollout_binding=binding,
    )
    objective.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="dummy", weight=1.0)
    constraints = ConstraintList()
    parameter_options = {}
    if binding is not None and binding.options.weight > 0.0:
        parameter_options = binding.parameter_options()
        constraints.add(CustomObjective.terminal_endurance_rollout_domain, node=Node.END,
                        binding=binding, min_bound=binding.domain_lower_bounds, max_bound=np.inf)
    factory = RecedingHorizonOptimization if receding else OptimalControlProgram
    objective_options = {"common_objective_functions" if receding else "objective_functions": objective}
    ocp = factory(
        model, n_shooting, 0.8, dynamics=DynamicsOptions(), constraints=constraints,
        x_bounds=x_bounds, x_init=x_init, u_bounds=u_bounds, u_init=u_init, use_sx=True, n_threads=1,
        **parameter_options, **objective_options,
    )
    if binding is not None and binding.options.weight > 0.0:
        binding.attach(ocp)
    return ocp


def test_real_bioptim_nlp_compiles_once_across_numerical_profile_updates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    binding = _binding()
    ocp = _tiny_ocp(binding)
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_print_level(0)
    solver.set_c_compile(True)
    first = ocp.solve(solver=solver)
    binding.observe_solver(ocp)
    compiled = ocp.ocp_solver.shaked_ocp_solver
    nlp = ocp.nlp[0]
    profile_buffer = ocp.parameter_bounds[ROLLOUT_PARAMETER_KEY].min
    first_cost = float(first.cost)
    binding.update(ocp, CertifiedRolloutParameters.from_payload(_payload(2.0)))
    second = ocp.solve(solver=solver)
    binding.observe_solver(ocp)
    assert first.status == second.status == 0
    assert float(second.cost) != pytest.approx(first_cost)
    assert ocp.nlp[0] is nlp
    assert ocp.parameter_bounds[ROLLOUT_PARAMETER_KEY].min is profile_buffer
    assert ocp.ocp_solver.shaked_ocp_solver is compiled
    assert binding.build_count == 1
    assert binding.summary()["compiled_solver_build_count"] == 1
    assert binding.summary()["compiled_solver_reuse_verified"] is True
    assert (tmp_path / "nlp.c").is_file()


def test_zero_weight_bioptim_nlp_matches_the_baseline():
    baseline, disabled = _tiny_ocp(), _tiny_ocp(_binding(weight=0.0))
    assert baseline.nlp[0].numerical_data_timeseries == disabled.nlp[0].numerical_data_timeseries
    assert baseline.variables_vector.shape == disabled.variables_vector.shape
    assert len(baseline.nlp[0].J) == len(disabled.nlp[0].J)
    assert len(baseline.nlp[0].g) == len(disabled.nlp[0].g)
    for left, right in zip(baseline.nlp[0].J, disabled.nlp[0].J, strict=True):
        assert left.function[0].serialize() == right.function[0].serialize()


def test_zero_force_at_rest_is_feasible_with_the_mixed_domain_bounds():
    payload = _payload()
    layout = RolloutObjectiveLayout(**payload["layout"])
    packed = np.asarray(payload["profile_parameters"])
    for field in ("force", "force_derivative", "first_half_force_integral", "second_half_force_integral"):
        packed[layout.field_slice(field)] = 0.0
    payload["profile_parameters"] = packed.tolist()
    payload["adapter_report"]["rollout_objective_profile"]["parameters"] = packed.tolist()
    profile = CertifiedRolloutParameters.from_payload(payload)
    binding = EnduranceRolloutBinding(EnduranceRolloutOptions(profile, 0.1, 1e-8))
    initial = np.concatenate([muscle.fatigue.rest_state for muscle in profile.muscles])
    objective, _, _, margins = binding.function(initial, packed)
    assert float(objective) == pytest.approx(0.0, abs=1e-14)
    assert np.all(np.asarray(margins).ravel() >= binding.domain_lower_bounds)
    ocp = _tiny_ocp(binding, initial=initial)
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_print_level(0)
    solution = ocp.solve(solver=solver)
    assert solution.status == 0


def test_real_receding_horizon_loop_reuses_compiled_nlp_for_two_windows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    binding = _binding()
    rho = _tiny_ocp(binding, receding=True)
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_print_level(0)
    solver.set_c_compile(True)
    costs, compiled = [], []

    def update(ocp, index, previous):
        if previous is not None:
            assert previous.status == 0
            costs.append(float(previous.cost))
            binding.observe_solver(ocp)
            compiled.append(ocp.ocp_solver.shaked_ocp_solver)
        if index == 1:
            certified_parameters = previous.parameters[ROLLOUT_PARAMETER_KEY].copy()
            binding.update(ocp, CertifiedRolloutParameters.from_payload(_payload(2.0)))
            np.testing.assert_array_equal(previous.parameters[ROLLOUT_PARAMETER_KEY], certified_parameters)
        return index < 2

    rho.solve(update_function=update, solver=solver)
    assert rho.total_optimization_run == 2
    assert len(costs) == 2
    assert costs[0] != pytest.approx(costs[1])
    assert compiled[0] is compiled[1]
    assert binding.summary()["compiled_solver_build_count"] == 1
