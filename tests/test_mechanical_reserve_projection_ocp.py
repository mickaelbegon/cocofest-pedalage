"""Tests for the isolated fixed-parameter mechanical reserve OCP binding."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters
from cocofest.optimization.mechanical_reserve_projection import (
    LocalMechanicalMarginModel,
    projected_mechanical_reserve,
)
from cocofest.optimization.mechanical_reserve_calibration import PulseWidthForceAffineModel
from cocofest.optimization.mechanical_reserve_projection_ocp import (
    ACTIVATION_PARAMETER_KEY,
    FORCE_PARAMETER_KEY,
    MARGIN_PARAMETER_KEY,
    LOCAL_PW_CURVATURE_PARAMETER_KEY,
    LOCAL_PW_GRADIENT_PARAMETER_KEY,
    LOCAL_PW_REFERENCE_PARAMETER_KEY,
    PW_JACOBIAN_PARAMETER_KEY,
    PW_REFERENCE_PARAMETER_KEY,
    MechanicalReserveProjectionBinding,
    _pack_margin,
    affine_candidate_force_from_pulse_widths_casadi,
    candidate_pulse_width_matrix_from_controller,
)


@pytest.fixture
def case():
    parameters = (
        DingFatigueParameters(4920., .060601, .137, -.04, 2.1e-6, 1.9e-6, 127.),
        DingFatigueParameters(4100., .070, .12, -.035, 2.8e-6, 1.5e-6, 110.),
    )
    states = np.array([p.rest_state for p in parameters])
    forces = np.array([[40., 110., 80.], [60., 55., 120.]])
    durations = [.07, .19, .37]
    model = LocalMechanicalMarginModel(
        states, [.5, .6],
        [[[1e-4, -1., -.5], [2e-5, -.3, -.1]],
         [[2e-5, -.2, -.4], [1e-4, -.8, -.6]]],
    )
    return states, forces, durations, parameters, model


@pytest.fixture
def fake_bioptim(monkeypatch):
    class ListBase(dict):
        def add(self, name, **kwargs):
            self[name] = SimpleNamespace(**kwargs)

    class ParameterList(ListBase):
        def __init__(self, *, use_sx):
            super().__init__()
            self.use_sx = use_sx

    class VariableScaling:
        def __init__(self, name, values):
            self.name, self.values = name, values

    class BoundsList(ListBase):
        def add(self, name, *, min_bound, max_bound, interpolation):
            self[name] = SimpleNamespace(
                min=np.array(min_bound, copy=True), max=np.array(max_bound, copy=True),
                interpolation=interpolation,
            )

    class InitialGuessList(ListBase):
        def add(self, name, *, initial_guess):
            self[name] = np.array(initial_guess, copy=True)

    module = SimpleNamespace(
        ParameterList=ParameterList,
        BoundsList=BoundsList,
        InitialGuessList=InitialGuessList,
        VariableScaling=VariableScaling,
        InterpolationType=SimpleNamespace(CONSTANT="constant"),
    )
    monkeypatch.setitem(sys.modules, "bioptim", module)
    return module


def _binding(case, **kwargs):
    _, forces, durations, parameters, model = case
    return MechanicalReserveProjectionBinding(
        forces=forces, durations=durations, parameters=parameters,
        horizons=(1, 20, 60), margin_model=model, weight=2.5, **kwargs,
    )


def _ocp(binding):
    options = binding.parameter_options()
    class Ocp:
        def __init__(self):
            self.nlp = [object()]
            self.parameter_bounds = options["parameter_bounds"]
            self.initial_updates = []
            self.ocp_solver = SimpleNamespace(shaked_ocp_solver=None)
            self.fail_initial_update = False

        def update_initial_guess(self, *, parameter_init):
            if self.fail_initial_update:
                raise RuntimeError("initial update failed")
            self.initial_updates.append(parameter_init)

    return Ocp(), options


@pytest.mark.parametrize("use_sx", [True, False])
def test_symbolic_cost_matches_numerical_surrogate_and_changes_with_parameters(case, use_sx):
    ca = pytest.importorskip("casadi")
    states, forces, durations, parameters, model = case
    binding = _binding(case, use_sx=use_sx)
    outputs = binding.function(states, forces.ravel(), _pack_margin(model), 1.0)
    assert float(binding.function(states, forces.ravel(), _pack_margin(model), 0.0)[0]) == 0.0
    numerical = projected_mechanical_reserve(states, forces, durations, parameters,
                                              (1, 20, 60), model)
    assert float(outputs[0]) == pytest.approx(2.5 * numerical.penalty, abs=1e-12)
    assert float(outputs[1]) == pytest.approx(numerical.penalty, abs=1e-12)
    assert float(outputs[2]) == pytest.approx(numerical.hard_minimum_margin, abs=1e-12)
    np.testing.assert_allclose(outputs[3], numerical.margins, atol=1e-12)

    increased = forces * 1.2
    shifted_model = LocalMechanicalMarginModel(model.reference_states,
                                               model.reference_margins + .05,
                                               model.state_jacobian)
    changed = binding.function(states, increased.ravel(), _pack_margin(shifted_model), 1.0)
    expected = projected_mechanical_reserve(states, increased, durations, parameters,
                                            (1, 20, 60), shifted_model)
    assert float(changed[0]) == pytest.approx(2.5 * expected.penalty, abs=1e-12)
    assert float(changed[0]) != pytest.approx(float(outputs[0]))

    symbol = (ca.SX if use_sx else ca.MX).sym("force", forces.size)
    objective = binding.function(states, symbol, _pack_margin(model), 1.0)[0]
    derivative = ca.Function("reserve_force_gradient", [symbol], [ca.gradient(objective, symbol)])
    assert np.all(np.isfinite(np.asarray(derivative(forces.ravel()))))


def test_affine_candidate_force_map_preserves_reference_and_pw_gradient(case):
    ca = pytest.importorskip("casadi")
    _, forces, _, _, _ = case
    widths = np.array([[.0002, .0003, .0004], [.00025, .00035, .00045]])
    jacobian = np.zeros((2, 3, 2, 3))
    jacobian[0, :, 0, :] = np.array([[2., 0., 0.], [.1, 1.8, 0.], [.2, .1, 1.5]])
    jacobian[1, :, 1, :] = np.array([[1.2, 0., 0.], [.2, 1.1, 0.], [.1, .2, 1.0]])
    candidate = ca.SX.sym("pw", 2, 3)
    mapped = affine_candidate_force_from_pulse_widths_casadi(
        candidate, reference_pulse_widths=widths, reference_forces=forces,
        force_jacobian=jacobian,
    )
    function = ca.Function("affine_pw_force", [candidate], [mapped])
    np.testing.assert_allclose(function(widths), forces, rtol=1e-12, atol=1e-12)
    perturbed = widths.copy()
    perturbed[0, 1] += 1e-5
    expected = forces + np.einsum("abij,ij->ab", jacobian, perturbed - widths)
    np.testing.assert_allclose(function(perturbed), expected, rtol=1e-9, atol=1e-9)
    gradient = ca.Function("affine_pw_force_gradient", [candidate],
                           [ca.jacobian(ca.vec(mapped.T), ca.vec(candidate.T))])
    np.testing.assert_allclose(gradient(widths), jacobian.reshape(6, 6), rtol=1e-9, atol=1e-9)


def test_candidate_pulse_width_matrix_reads_global_scaled_bioptim_layout():
    ca = pytest.importorskip("casadi")
    decision = ca.SX.sym("w", 6)
    names = ("first", "second")
    nlp = SimpleNamespace(
        controls={f"last_pulse_width_{name}": object() for name in names},
        u_scaling={
            "last_pulse_width_first": SimpleNamespace(scaling=np.array([[.0025]])),
            "last_pulse_width_second": SimpleNamespace(scaling=np.array([[.00125]])),
        },
    )
    controller = SimpleNamespace(ocp=SimpleNamespace(
        nlp=[nlp], variables_vector=decision,
        vector_layout=SimpleNamespace(index_map={
            (0, "controls", 0): (slice(0, 2), 1),
            (0, "controls", 1): (slice(2, 4), 1),
            (0, "controls", 2): (slice(4, 6), 1),
        }),
    ))
    matrix = candidate_pulse_width_matrix_from_controller(
        controller, muscle_names=names, interval_count=3,
    )
    extractor = ca.Function("physical_pw", [decision], [matrix])
    np.testing.assert_allclose(extractor([1., 2., 3., 4., 5., 6.]),
                               [[.0025, .0075, .0125], [.0025, .005, .0075]])


def test_opt_in_pw_coupling_uses_symbolic_pw_and_numeric_boundary_parameters(case, fake_bioptim):
    ca = pytest.importorskip("casadi")
    states, forces, durations, parameters, margin = case
    widths = np.full(forces.shape, .0003)
    jacobian = np.zeros((*forces.shape, *forces.shape))
    jacobian[0, :, 0, :] = np.eye(3) * 2e5
    jacobian[1, :, 1, :] = np.eye(3) * 1e5
    pw_model = PulseWidthForceAffineModel(widths, forces, jacobian)
    binding = MechanicalReserveProjectionBinding(
        forces=forces, durations=durations, parameters=parameters, horizons=(1, 20, 60),
        margin_model=margin, weight=2.5, pulse_width_force_model=pw_model,
    )
    options = binding.parameter_options()
    assert list(options["parameters"]) == [
        FORCE_PARAMETER_KEY, MARGIN_PARAMETER_KEY, ACTIVATION_PARAMETER_KEY,
        PW_REFERENCE_PARAMETER_KEY, PW_JACOBIAN_PARAMETER_KEY,
    ]
    candidate = widths.copy()
    candidate[0, 1] += 1e-5
    values = binding.function(
        states, candidate, widths.ravel(), forces.ravel(),
        binding._vectors()[PW_JACOBIAN_PARAMETER_KEY],
        _pack_margin(margin), 1.0,
    )
    affine_forces = np.asarray(affine_candidate_force_from_pulse_widths_casadi(
        ca.DM(candidate), reference_pulse_widths=widths, reference_forces=forces,
        force_jacobian=jacobian,
    ))
    expected = projected_mechanical_reserve(states, affine_forces, durations, parameters,
                                            (1, 20, 60), margin)
    assert float(values[0]) == pytest.approx(binding.weight * expected.penalty, abs=1e-10)
    # Evaluate the binding's dedicated audit at the reference; it must expose
    # a true candidate-PW direction, unlike the frozen-force implementation.
    ocp, _ = _ocp(binding)
    binding.update(ocp, forces=forces, margin_model=margin, pulse_width_force_model=pw_model)
    audit = binding.sensitivity_audit(states)
    assert audit["candidate_force_coupled_to_pw"] is True
    assert audit["candidate_pulse_width_gradient_l2"] > 0


def test_local_pw_terms_chain_the_force_gradient_without_global_control_graph(case, fake_bioptim):
    ca = pytest.importorskip("casadi")
    states, forces, durations, parameters, margin = case
    widths = np.array([[.0002, .0003, .0004], [.00025, .00035, .00045]])
    jacobian = np.zeros((*forces.shape, *forces.shape))
    jacobian[0, :, 0, :] = np.array([[2e5, 0., 0.], [1e4, 1.8e5, 0.], [2e4, 1e4, 1.5e5]])
    jacobian[1, :, 1, :] = np.array([[1.2e5, 0., 0.], [2e4, 1.1e5, 0.], [1e4, 2e4, 1e5]])
    pw_model = PulseWidthForceAffineModel(widths, forces, jacobian)
    binding = MechanicalReserveProjectionBinding(
        forces=forces, durations=durations, parameters=parameters, horizons=(1, 20, 60),
        margin_model=margin, weight=2.5, local_pulse_width_cost=True,
        local_pulse_width_trust_s=25e-6,
    )
    terms = binding.local_pulse_width_cost_terms(
        initial_states=states, forces=forces, margin_model=margin,
        pulse_width_force_model=pw_model,
    )
    _, force_gradient = binding.sensitivity_function(states, forces.ravel(), _pack_margin(margin), [1.0])
    expected = np.einsum("mkij,mk->ij", jacobian,
                         np.asarray(force_gradient).reshape(forces.shape))
    np.testing.assert_allclose(terms["gradient"], expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(terms["curvature"], np.abs(expected) / 25e-6)
    assert binding.candidate_force_coupling is False
    options = binding.parameter_options()
    assert list(options["parameters"])[-3:] == [
        LOCAL_PW_GRADIENT_PARAMETER_KEY, LOCAL_PW_REFERENCE_PARAMETER_KEY,
        LOCAL_PW_CURVATURE_PARAMETER_KEY,
    ]
    ocp, _ = _ocp(binding)
    receipt = binding.update(ocp, forces=forces, margin_model=margin,
                             pulse_width_force_model=pw_model, local_pulse_width_terms=terms)
    assert receipt["objective_graph_rebuild_required"] is False
    np.testing.assert_allclose(ocp.parameter_bounds[LOCAL_PW_GRADIENT_PARAMETER_KEY].min[:, 0],
                               expected.ravel())


def test_local_pw_objective_reads_only_current_control_node_with_physical_scaling(case):
    ca = pytest.importorskip("casadi")
    _, forces, durations, parameters, margin = case
    binding = MechanicalReserveProjectionBinding(
        forces=forces, durations=durations, parameters=parameters, horizons=(1, 20, 60),
        margin_model=margin, weight=2.5, local_pulse_width_cost=True,
        local_pulse_width_trust_s=20e-6,
    )
    command = ca.SX.sym("command", 2)
    gradient = np.arange(1., 7.)
    reference = np.full(6, .0003)
    curvature = np.full(6, 2e5)
    controller = SimpleNamespace(
        node_index=1,
        model=SimpleNamespace(muscles_dynamics_model=[
            SimpleNamespace(muscle_name="first"), SimpleNamespace(muscle_name="second"),
        ]),
        controls={
            "last_pulse_width_first": SimpleNamespace(cx=command[0]),
            "last_pulse_width_second": SimpleNamespace(cx=command[1]),
        },
        parameters={
            ACTIVATION_PARAMETER_KEY: SimpleNamespace(cx=ca.DM([1.])),
            LOCAL_PW_GRADIENT_PARAMETER_KEY: SimpleNamespace(cx=ca.DM(gradient)),
            LOCAL_PW_REFERENCE_PARAMETER_KEY: SimpleNamespace(cx=ca.DM(reference)),
            LOCAL_PW_CURVATURE_PARAMETER_KEY: SimpleNamespace(cx=ca.DM(curvature)),
        },
        ocp=SimpleNamespace(nlp=[SimpleNamespace(u_scaling={
            "last_pulse_width_first": SimpleNamespace(scaling=np.array([.001])),
            "last_pulse_width_second": SimpleNamespace(scaling=np.array([.002])),
        })]),
    )
    expression = binding.local_pulse_width_objective(controller)
    function = ca.Function("local_pw_node", [command], [expression])
    physical = np.array([.0004, .0008])
    delta = physical - reference[[1, 4]]
    expected = binding.weight * (
        np.dot(gradient[[1, 4]], delta) + .5 * np.dot(curvature[[1, 4]], delta**2)
    )
    assert float(function([.4, .4])) == pytest.approx(expected)


def test_fixed_parameter_options_and_same_nlp_updates_only_numeric_buffers(case, fake_bioptim):
    binding = _binding(case)
    ocp, options = _ocp(binding)
    keys = [FORCE_PARAMETER_KEY, MARGIN_PARAMETER_KEY, ACTIVATION_PARAMETER_KEY]
    assert list(options["parameters"]) == keys
    assert options["parameters"].use_sx is True
    assert options["parameters"][FORCE_PARAMETER_KEY].size == 6
    assert options["parameters"][MARGIN_PARAMETER_KEY].size == 20
    assert options["parameters"][ACTIVATION_PARAMETER_KEY].size == 1
    for key in keys:
        bound = ocp.parameter_bounds[key]
        np.testing.assert_array_equal(bound.min, bound.max)
        np.testing.assert_array_equal(bound.min, options["parameter_init"][key])
        assert bound.interpolation == "constant"

    graph = binding.function
    nlp = ocp.nlp[0]
    new_forces = case[1] * 1.1
    old_model = case[4]
    new_model = LocalMechanicalMarginModel(old_model.reference_states,
                                           old_model.reference_margins + .03,
                                           old_model.state_jacobian * .9)
    receipt = binding.update(ocp, forces=new_forces, margin_model=new_model)
    assert binding.function is graph and ocp.nlp[0] is nlp
    assert receipt["nlp_identity"] == id(nlp)
    assert receipt["objective_graph_build_count"] == 1
    assert receipt["parameter_update_count"] == 1
    assert receipt["objective_graph_rebuild_required"] is False
    assert all(item["previous_sha256"] != item["current_sha256"]
               for item in receipt["changed_parameters"].values())
    for key, values in ((FORCE_PARAMETER_KEY, new_forces.ravel()),
                        (MARGIN_PARAMETER_KEY, _pack_margin(new_model)),
                        (ACTIVATION_PARAMETER_KEY, np.array([1.0]))):
        np.testing.assert_array_equal(ocp.parameter_bounds[key].min[:, 0], values)
        np.testing.assert_array_equal(ocp.parameter_bounds[key].max[:, 0], values)
        np.testing.assert_array_equal(ocp.initial_updates[-1][key][:, 0], values)
    assert binding.summary()["horizons"] == [1, 20, 60]
    assert binding.summary()["objective_graph_rebuild_required"] is False
    assert not binding.forces.flags.writeable
    with pytest.raises(AttributeError, match="immutable"):
        binding.horizons = (1, 40)
    with pytest.raises(AttributeError, match="immutable"):
        binding.durations = (.1, .2, .3)
    with pytest.raises(ValueError, match="symbolic type"):
        binding.parameter_options(use_sx=False)
    updated = binding.function(case[0],
                               ocp.parameter_bounds[FORCE_PARAMETER_KEY].min[:, 0],
                               ocp.parameter_bounds[MARGIN_PARAMETER_KEY].min[:, 0],
                               ocp.parameter_bounds[ACTIVATION_PARAMETER_KEY].min[:, 0])
    expected = projected_mechanical_reserve(case[0], new_forces, case[2], case[3],
                                            (1, 20, 60), new_model)
    assert float(updated[0]) == pytest.approx(binding.weight * expected.penalty, abs=1e-12)
    audit = binding.sensitivity_audit(case[0])
    assert audit["activation"] == 1.0
    assert audit["terminal_slow_state_gradient_l2"] > 0.0
    assert audit["terminal_slow_state_gradient_normalized_l2"] > 0.0
    assert audit["frozen_force_parameter_gradient_l2"] > 0.0
    assert audit["candidate_force_coupled_to_pw"] is False


def test_rejects_changed_dimensions_nonfinite_values_and_rebuilt_nlp(case, fake_bioptim):
    binding = _binding(case)
    ocp, _ = _ocp(binding)
    binding.attach(ocp)
    model = case[4]
    wrong_constraints = LocalMechanicalMarginModel(model.reference_states, [.5], model.state_jacobian[:1])
    wrong_muscles = LocalMechanicalMarginModel(model.reference_states[:1], [.5, .6],
                                               model.state_jacobian[:, :1])
    invalid = [
        (np.zeros((2, 4)), model, "dimensions"),
        (np.full((2, 3), np.nan), model, "finite"),
        (-np.ones((2, 3)), model, "nonnegative"),
        (case[1], wrong_constraints, "constraint dimensions"),
        (case[1], wrong_muscles, "muscle dimensions"),
    ]
    for forces, margin, message in invalid:
        with pytest.raises(ValueError, match=message):
            binding.update(ocp, forces=forces, margin_model=margin)
    assert binding.update_count == 0
    rebuilt, _ = _ocp(binding)
    with pytest.raises(RuntimeError, match="rebuilt NLP"):
        binding.update(rebuilt, forces=case[1], margin_model=model)
    assert binding.update_count == 0


def test_rejects_wrong_parameter_layout_and_rolls_back_failed_initial_guess(case, fake_bioptim):
    binding = _binding(case)
    ocp, _ = _ocp(binding)
    key = MARGIN_PARAMETER_KEY
    ocp.parameter_bounds[key].max = ocp.parameter_bounds[key].max[:-1]
    with pytest.raises(ValueError, match="dimensions"):
        binding.attach(ocp)
    assert binding.summary()["nlp_attached"] is False

    ocp, _ = _ocp(binding)
    previous = {key: (bound.min.copy(), bound.max.copy())
                for key, bound in ocp.parameter_bounds.items()}
    ocp.fail_initial_update = True
    with pytest.raises(RuntimeError, match="initial update failed"):
        binding.update(ocp, forces=case[1] * 1.1, margin_model=case[4])
    assert binding.update_count == 0
    for key, bound in ocp.parameter_bounds.items():
        np.testing.assert_array_equal(bound.min, previous[key][0])
        np.testing.assert_array_equal(bound.max, previous[key][1])


def test_solver_identity_audit_and_zero_cost_has_no_configuration(case, fake_bioptim):
    assert MechanicalReserveProjectionBinding.configured(weight=0) is None
    for weight in (-1, np.nan, np.inf):
        with pytest.raises(ValueError, match="weight"):
            MechanicalReserveProjectionBinding.configured(weight=weight)
    binding = _binding(case)
    ocp, _ = _ocp(binding)
    assert binding.observe_solver(ocp)["solver_observed"] is False
    solver = object()
    ocp.ocp_solver.shaked_ocp_solver = solver
    assert binding.observe_solver(ocp)["compiled_solver_reuse_verified"] is False
    assert binding.observe_solver(ocp)["compiled_solver_reuse_verified"] is False
    receipt = binding.update(ocp, forces=case[1], margin_model=case[4])
    assert receipt["compiled_solver_reuse_verified"] is False
    assert binding.observe_solver(ocp)["compiled_solver_reuse_verified"] is True
    ocp.ocp_solver.shaked_ocp_solver = object()
    with pytest.raises(RuntimeError, match="solver changed"):
        binding.update(ocp, forces=case[1], margin_model=case[4])
    assert binding.update_count == 1
