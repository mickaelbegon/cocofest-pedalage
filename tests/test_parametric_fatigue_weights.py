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
