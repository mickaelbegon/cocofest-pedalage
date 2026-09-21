from types import SimpleNamespace

import numpy as np
import pytest

from bioptim import (
    BoundsList,
    ConfigureVariables,
    ConstraintList,
    DynamicsEvaluation,
    DynamicsOptions,
    InitialGuessList,
    Node,
    ObjectiveFcn,
    OptimalControlProgram,
    Solver,
    StateDynamics,
)

from cocofest import CustomObjective
from cocofest.optimization.adaptive_moment_rollout import (
    DingPulseWidthParameters,
    MomentTrackingInterval,
    propagate_ding_pulse_width_interval,
)
from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters
from cocofest.optimization.muscle_horizon_ocp import (
    MUSCLE_HORIZON_FUTURE_PW_KEY,
    MUSCLE_HORIZON_PROFILE_KEY,
    CertifiedMuscleHorizonProfile,
    MuscleHorizonBinding,
    MuscleHorizonOptions,
    muscle_horizon_signature_fields,
    resolve_muscle_horizon_options,
)
from cocofest.optimization import muscle_horizon_ocp
from cocofest.optimization.rho_adaptive_moment_policy import RhoAdaptiveMomentPolicy
from examples.fes_multibody.cycling import cycling_pulse_width_mhe as mhe
from examples.fes_multibody.cycling import cycling_fes_solver_comparison as comparison
from examples.fes_multibody.cycling import (
    cycling_pulse_width_mhe_acados_periodic as periodic,
)


def _muscle():
    return DingPulseWidthParameters(
        fatigue=DingFatigueParameters(
            a_rest=4920.0,
            tau1_rest=0.060601,
            km_rest=0.137,
            alpha_a=-0.04,
            alpha_tau1=2.1e-6,
            alpha_km=1.9e-6,
            tau_fat=127.0,
        ),
        tauc=0.011,
        tau2=0.001,
        pd0=0.000131405,
        pdt=0.000194138,
        pulse_width_max=0.0006,
    )


def _policy(pulse_width=0.00035):
    muscle = _muscle()
    initial = np.array([[0.1629821583533315, 20.0, 4800.0, 0.065, 0.145]])
    endpoint = propagate_ding_pulse_width_interval(
        initial[0],
        pulse_width=pulse_width,
        duration=1.0 / 30.0,
        calcium_amplitude=1.0597355478114694,
        mechanical_gain=0.95,
        parameters=muscle,
        integration_substeps=1,
    )
    interval = MomentTrackingInterval(
        duration=1.0 / 30.0,
        calcium_amplitudes=(1.0597355478114694,),
        mechanical_gains=(0.95,),
        moment_coefficients=(0.05,),
        target_moments=(0.05 * endpoint[1],),
    )
    return RhoAdaptiveMomentPolicy(
        muscle_names=("m0",),
        source_cycle_index=0,
        source_cycle_count=1,
        period=1.0 / 30.0,
        initial_states=initial,
        intervals=(interval,),
        parameters=(muscle,),
        source_pulse_widths=np.array([[pulse_width]]),
        target_moments=np.array([[interval.target_moments[0]]]),
        certification_basis="synthetic_certified_test_fixture",
        period_basis="test",
    )


def _binding(weight=0.1, horizon_cycles=1):
    profile = CertifiedMuscleHorizonProfile.from_rho_policy(
        _policy(), horizon_cycles=horizon_cycles, integration_substeps=1
    )
    return MuscleHorizonBinding(MuscleHorizonOptions(profile, weight))


def test_binding_declares_fixed_profile_and_bounded_free_future_pw_parameters():
    binding = _binding(horizon_cycles=3)
    options = binding.parameter_options()
    profile = binding.options.profile

    assert options["parameters"][MUSCLE_HORIZON_PROFILE_KEY].size == profile.layout.parameter_size
    assert (
        options["parameters"][MUSCLE_HORIZON_FUTURE_PW_KEY].size
        == profile.layout.future_pulse_width_size
    )
    fixed = options["parameter_bounds"][MUSCLE_HORIZON_PROFILE_KEY]
    np.testing.assert_array_equal(fixed.min, fixed.max)
    future = options["parameter_bounds"][MUSCLE_HORIZON_FUTURE_PW_KEY]
    np.testing.assert_allclose(future.min, profile.muscles[0].pd0)
    np.testing.assert_allclose(future.max, profile.muscles[0].pulse_width_max)
    assert binding.total_moment_bounds[0].size == 3
    assert binding.domain_lower_bounds.size == profile.layout.domain_margin_size


def test_zero_weight_preserves_objective_list_structure():
    model = SimpleNamespace(muscles_dynamics_model=[])
    kwargs = dict(
        model=model,
        minimize_force=False,
        minimize_fatigue=True,
        minimize_control=False,
        cost_fun_weight=[0, 1, 0],
        target=0.0,
    )
    baseline = mhe.set_objective_functions(**kwargs)
    disabled = mhe.set_objective_functions(
        **kwargs, muscle_horizon_binding=_binding(weight=0.0)
    )

    assert len(baseline[0]) == len(disabled[0])
    for left, right in zip(baseline[0], disabled[0], strict=True):
        assert left.custom_function is right.custom_function
        assert left.type is right.type
        assert left.node == right.node
        np.testing.assert_array_equal(left.weight, right.weight)


def test_nonzero_binding_adds_one_scalar_terminal_objective():
    binding = _binding()
    objectives = mhe.set_objective_functions(
        SimpleNamespace(muscles_dynamics_model=[]),
        False,
        True,
        False,
        [0, 1, 0],
        0.0,
        muscle_horizon_binding=binding,
    )
    entry = objectives[0][0]

    assert entry.custom_function is CustomObjective.minimize_terminal_muscle_horizon
    assert entry.type == ObjectiveFcn.Mayer.CUSTOM
    assert entry.node == Node.END
    assert entry.quadratic is False
    assert float(entry.weight[0]) == pytest.approx(1000.0)


class _ConstantFullDingStateModel(StateDynamics):
    def __init__(self):
        super().__init__()
        muscle = _muscle()
        self.muscles_dynamics_model = [
            SimpleNamespace(
                muscle_name="m0",
                a_scale=muscle.fatigue.a_rest,
                tau1_rest=muscle.fatigue.tau1_rest,
                km_rest=muscle.fatigue.km_rest,
                alpha_a=muscle.fatigue.alpha_a,
                alpha_tau1=muscle.fatigue.alpha_tau1,
                alpha_km=muscle.fatigue.alpha_km,
                tau_fat=muscle.fatigue.tau_fat,
                tauc=muscle.tauc,
                tau2=muscle.tau2,
                pd0=muscle.pd0,
                pdt=muscle.pdt,
            )
        ]

    def serialize(self):
        return _ConstantFullDingStateModel, {}

    @property
    def name(self):
        return "synthetic_muscle_horizon_binding_test"

    @property
    def name_dofs(self):
        return ["dummy"]

    @property
    def state_configuration_functions(self):
        def configure(ocp, nlp):
            for field in ("Cn", "F", "A", "Tau1", "Km"):
                ConfigureVariables.configure_new_variable(
                    f"{field}_m0", [f"{field}_m0"], ocp, nlp, as_states=True
                )

        return [configure]

    @property
    def control_configuration_functions(self):
        return [
            lambda ocp, nlp: ConfigureVariables.configure_new_variable(
                "dummy", ["dummy"], ocp, nlp, as_controls=True
            )
        ]

    @property
    def algebraic_configuration_functions(self):
        return []

    @property
    def extra_configuration_functions(self):
        return []

    def dynamics(
        self,
        time,
        states,
        controls,
        parameters,
        algebraic_states,
        numerical_timeseries,
        nlp,
    ):
        return DynamicsEvaluation(dxdt=0 * states, defects=None)


def _tiny_ocp(binding):
    model = _ConstantFullDingStateModel()
    initial = _policy().initial_states[0]
    x_bounds, x_init = BoundsList(), InitialGuessList()
    for field, value in zip(("Cn", "F", "A", "Tau1", "Km"), initial, strict=True):
        x_bounds[f"{field}_m0"] = (
            [[value, 0.9 * value, 0.9 * value]],
            [[value, 1.1 * value, 1.1 * value]],
        )
        x_init[f"{field}_m0"] = [value]
    u_bounds, u_init = BoundsList(), InitialGuessList()
    u_bounds["dummy"] = [-1.0], [1.0]
    u_init["dummy"] = [0.0]
    objectives = mhe.set_objective_functions(
        model,
        False,
        True,
        False,
        [0, 1e-8, 0],
        0.0,
        terminal_wheel_regularization_weight=0.0,
        muscle_horizon_binding=binding,
    )
    objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="dummy", weight=1.0)
    constraints = ConstraintList()
    total_minimum, total_maximum = binding.total_moment_bounds
    constraints.add(
        CustomObjective.terminal_muscle_horizon_total_moment,
        node=Node.END,
        binding=binding,
        min_bound=total_minimum,
        max_bound=total_maximum,
    )
    constraints.add(
        CustomObjective.terminal_muscle_horizon_domain,
        node=Node.END,
        binding=binding,
        min_bound=binding.domain_lower_bounds,
        max_bound=np.inf,
    )
    ocp = OptimalControlProgram(
        model,
        2,
        0.1,
        dynamics=DynamicsOptions(),
        objective_functions=objectives,
        constraints=constraints,
        x_bounds=x_bounds,
        x_init=x_init,
        u_bounds=u_bounds,
        u_init=u_init,
        use_sx=True,
        n_threads=1,
        **binding.parameter_options(),
    )
    binding.attach(ocp)
    return ocp


def test_real_bioptim_nlp_solves_with_free_future_pw_and_exact_total_moment():
    binding = _binding()
    ocp = _tiny_ocp(binding)
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_print_level(0)
    solution = ocp.solve(solver=solver)

    assert solution.status == 0
    future = np.asarray(solution.parameters[MUSCLE_HORIZON_FUTURE_PW_KEY]).reshape(-1)
    profile = binding.options.profile
    outputs = binding.function(
        _policy().initial_states.reshape(-1),
        profile.packed_parameters,
        future,
    )
    np.testing.assert_allclose(np.asarray(outputs[4]), 0.0, atol=1e-8)
    assert future[0] == pytest.approx(profile.future_pulse_width_seed[0], abs=2e-8)


def test_compiled_nlp_is_reused_when_profile_and_future_pw_seed_are_updated(
    tmp_path, monkeypatch
):
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
    profile_buffer = ocp.parameter_bounds[MUSCLE_HORIZON_PROFILE_KEY].min

    updated_profile = CertifiedMuscleHorizonProfile.from_rho_policy(
        _policy(pulse_width=0.00037), horizon_cycles=1, integration_substeps=1
    )
    binding.update(ocp, updated_profile)
    second = ocp.solve(solver=solver)
    binding.observe_solver(ocp)

    assert first.status == second.status == 0
    assert ocp.nlp[0] is nlp
    assert ocp.parameter_bounds[MUSCLE_HORIZON_PROFILE_KEY].min is profile_buffer
    assert ocp.ocp_solver.shaked_ocp_solver is compiled
    assert binding.build_count == 1
    assert binding.update_count == 1
    assert binding.summary()["compiled_solver_reuse_verified"] is True
    assert (tmp_path / "nlp.c").is_file()


def test_disabled_cli_path_does_not_require_or_read_source_files():
    args = SimpleNamespace(muscle_horizon_weight=0.0)

    assert resolve_muscle_horizon_options(args) is None
    assert muscle_horizon_signature_fields(args) == {}


def test_both_public_launchers_expose_a_strictly_disabled_default():
    for parser in (comparison.build_cli(), periodic.build_argument_parser()):
        args = parser.parse_args([])
        assert args.experimental_muscle_horizon is False
        assert args.muscle_horizon_weight == 0.0
        assert args.muscle_horizon_cycles == 3
        assert resolve_muscle_horizon_options(args) is None


def test_nonzero_cli_weight_requires_explicit_experimental_opt_in():
    args = SimpleNamespace(
        muscle_horizon_weight=0.1,
        experimental_muscle_horizon=False,
    )

    with pytest.raises(ValueError, match="experimental-muscle-horizon"):
        resolve_muscle_horizon_options(args)


def test_cli_builds_certified_profile_and_infers_isokinetic_period(monkeypatch):
    observed = {}

    def fake_builder(source, reduced_profile, **kwargs):
        observed.update(source=source, reduced_profile=reduced_profile, **kwargs)
        return _policy()

    monkeypatch.setattr(
        muscle_horizon_ocp, "build_rho_adaptive_moment_policy", fake_builder
    )
    args = SimpleNamespace(
        muscle_horizon_weight=0.2,
        experimental_muscle_horizon=True,
        solver="ipopt",
        muscle_horizon_source="rho.npz",
        muscle_horizon_reduced_profile="reduced.npz",
        muscle_horizon_source_cycle_index=7,
        muscle_horizon_cycle_period=None,
        formulation="isokinetic",
        isokinetic_omega=-4.0 * np.pi,
        muscle_horizon_cycles=2,
        muscle_horizon_integration_substeps=1,
        muscle_horizon_temperature=0.03,
        muscle_horizon_allocation_weight=0.04,
        muscle_horizon_domain_epsilon=2e-8,
    )

    options = resolve_muscle_horizon_options(args)

    assert options.weight == pytest.approx(0.2)
    assert options.domain_epsilon == pytest.approx(2e-8)
    assert options.profile.layout.horizon_cycles == 2
    assert options.profile.layout.smooth_max_temperature == pytest.approx(0.03)
    assert options.profile.layout.allocation_weight == pytest.approx(0.04)
    assert observed["cycle_index"] == 7
    assert observed["cycle_period"] == pytest.approx(0.5)


def test_codegen_signature_excludes_numerical_profile_values():
    first = _binding().options
    second_profile = CertifiedMuscleHorizonProfile.from_rho_policy(
        _policy(pulse_width=0.00037), horizon_cycles=1, integration_substeps=1
    )
    second = MuscleHorizonOptions(
        second_profile,
        weight=first.weight,
        domain_epsilon=first.domain_epsilon,
    )
    first_args = SimpleNamespace(_muscle_horizon_options=first)
    second_args = SimpleNamespace(_muscle_horizon_options=second)

    assert muscle_horizon_signature_fields(
        first_args, structure_only=True
    ) == muscle_horizon_signature_fields(second_args, structure_only=True)
    assert muscle_horizon_signature_fields(first_args) != muscle_horizon_signature_fields(
        second_args
    )


def test_source_and_target_task_context_must_match_when_metadata_are_available():
    policy = _policy()
    policy = RhoAdaptiveMomentPolicy(
        **{
            **policy.__dict__,
            "source_signed_crank_torque_nm": 0.1,
            "source_mechanical_formulation": "reduced",
            "source_formulation": "dynamic",
        }
    )
    profile = CertifiedMuscleHorizonProfile.from_rho_policy(
        policy, horizon_cycles=1
    )
    audit = profile.validate_context(
        cycle_period_s=policy.period,
        signed_crank_torque_nm=0.1,
        mechanical_formulation="reduced",
        formulation="dynamic",
    )
    assert {item["status"] for item in audit.values()} == {"matched"}

    with pytest.raises(ValueError, match="signed_crank_torque_nm"):
        profile.validate_context(
            cycle_period_s=policy.period,
            signed_crank_torque_nm=0.2,
            mechanical_formulation="reduced",
            formulation="dynamic",
        )
