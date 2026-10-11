from types import SimpleNamespace

import numpy as np
import pytest

from cocofest import CustomObjective
from cocofest.optimization.parametric_fatigue_weights import (
    FATIGUE_WEIGHT_PARAMETER_KEY,
    ParametricFatigueWeightBinding,
)
from examples.fes_multibody.cycling import cycling_pulse_width_mhe as mhe
from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as periodic


def test_binding_validates_and_describes_a_fixed_parameter_vector():
    binding = ParametricFatigueWeightBinding((1.0, 2.0, 0.5, 1.0))
    assert binding.size == 4
    assert binding.summary()["parameter_key"] == FATIGUE_WEIGHT_PARAMETER_KEY
    assert binding.summary()["objective_graph_rebuild_required"] is False
    with pytest.raises(ValueError, match="strictly positive"):
        ParametricFatigueWeightBinding((1.0, 0.0))
    with pytest.raises(ValueError, match="dimensions"):
        binding.update(SimpleNamespace(nlp=[object()]), (1.0, 1.0))


def test_binding_updates_acados_augmented_state_bounds_without_rebuilding():
    class ScaledBounds:
        def __init__(self, minimum, maximum):
            self.min = np.asarray(minimum, dtype=float)
            self.max = np.asarray(maximum, dtype=float)

        def scale(self, _scaling):
            return self

    class BoundsMap(dict):
        def __getitem__(self, key):
            return super().__getitem__(key)

    nlp = object()
    weights = np.asarray([[1.0], [1.0]])
    parameter_bounds = BoundsMap({
        FATIGUE_WEIGHT_PARAMETER_KEY: ScaledBounds(weights.copy(), weights.copy()),
    })
    parameters = {
        FATIGUE_WEIGHT_PARAMETER_KEY: SimpleNamespace(index=[0, 1], scaling=SimpleNamespace(scaling=np.ones(2))),
    }
    acados_interface = SimpleNamespace(
        x_bound_min=np.zeros((3, 3)),
        x_bound_max=np.zeros((3, 3)),
    )
    nmpc = SimpleNamespace(
        nlp=[nlp], parameter_bounds=parameter_bounds, parameters=parameters,
        ocp_solver=acados_interface,
        update_initial_guess=lambda **_: None,
    )
    recovery_nlp = object()
    recovery_bounds = BoundsMap({
        FATIGUE_WEIGHT_PARAMETER_KEY: ScaledBounds(weights.copy(), weights.copy()),
    })
    recovery_nmpc = SimpleNamespace(
        nlp=[recovery_nlp], parameter_bounds=recovery_bounds, parameters=parameters,
        update_initial_guess=lambda **_: None,
    )
    binding = ParametricFatigueWeightBinding((1.0, 1.0))
    binding.attach(recovery_nmpc)

    update = binding.update(nmpc, (0.5, 1.5))

    np.testing.assert_allclose(acados_interface.x_bound_min[:2, :], [[0.5] * 3, [1.5] * 3])
    np.testing.assert_allclose(acados_interface.x_bound_max[:2, :], [[0.5] * 3, [1.5] * 3])
    np.testing.assert_allclose(recovery_bounds[FATIGUE_WEIGHT_PARAMETER_KEY].min, [[0.5], [1.5]])
    np.testing.assert_allclose(recovery_bounds[FATIGUE_WEIGHT_PARAMETER_KEY].max, [[0.5], [1.5]])
    assert update["integration"] == "fixed_parameter_bounds_and_acados_constant_states"
    assert binding.update_count == 1
    assert nmpc.nlp[0] is nlp
    assert update["attached_nlp_count"] == 2


def test_parameterized_objective_replaces_only_the_fatigue_residual():
    model = SimpleNamespace(muscles_dynamics_model=[object(), object(), object(), object()])
    binding = ParametricFatigueWeightBinding((1.0, 1.0, 1.0, 1.0))
    objectives = mhe.set_objective_functions(
        model, False, True, False, [0.0, 1.0, 0.0], 0.0,
        fatigue_weight_binding=binding,
    )
    entry = objectives[0][0]
    assert entry.custom_function is CustomObjective.minimize_parameterized_overall_muscle_fatigue
    assert entry.quadratic is True
    np.testing.assert_allclose(entry.weight, [10000.0])


def test_parameterized_weights_have_a_distinct_codegen_cache_signature():
    baseline = periodic.build_argument_parser().parse_args([])
    parameterized = periodic.build_argument_parser().parse_args(["--parametric-fatigue-weights"])
    assert baseline.parametric_fatigue_weights is False
    assert parameterized.parametric_fatigue_weights is True
    assert periodic._codegen_signature(baseline) != periodic._codegen_signature(parameterized)
