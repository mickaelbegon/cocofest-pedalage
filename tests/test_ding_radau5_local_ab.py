import casadi as ca
import numpy as np
import pytest

from scripts.benchmark_ding_radau5_local_ab import append_local_parameters


def test_local_adapter_preserves_original_nlp_parameters_before_offsets():
    x = ca.SX.sym("x")
    original_parameters = ca.SX.sym("weights", 2)
    offsets = ca.SX.sym("offsets", 2)
    nlp = {"p": original_parameters}
    objective = (x + offsets[0] - original_parameters[0]) ** 2
    objective += (x + offsets[1] - original_parameters[1]) ** 2
    constraints = ca.vertcat(x + offsets[0], x + offsets[1] - original_parameters[1])

    combined, reduced_objective, reduced_constraints, original_count = append_local_parameters(
        nlp, objective, constraints, offsets,
    )
    assert original_count == 2
    assert combined.numel() == 4
    evaluate = ca.Function("preserved_parameters", [x, combined], [reduced_objective, reduced_constraints])
    objective_value, constraint_values = evaluate(2.0, [3.0, 4.0, 0.1, -0.2])
    assert float(objective_value) == pytest.approx(5.65)
    assert np.asarray(constraint_values).reshape(-1) == pytest.approx([2.1, -2.2])


def test_local_adapter_without_original_parameters_uses_offsets_only():
    x = ca.SX.sym("x")
    offsets = ca.SX.sym("offsets", 1)
    combined, objective, constraints, original_count = append_local_parameters(
        {"p": ca.SX.zeros(0)}, (x + offsets[0]) ** 2, x + offsets[0], offsets,
    )
    assert original_count == 0
    assert combined.numel() == 1
    evaluate = ca.Function("offset_only", [x, combined], [objective, constraints])
    value, residual = evaluate(2.0, [0.5])
    assert float(value) == pytest.approx(6.25)
    assert float(residual) == pytest.approx(2.5)
