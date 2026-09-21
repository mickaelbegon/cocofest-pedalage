"""Guards for exact NLP stage elimination, including bounds and sensitivities."""
import casadi as ca
import numpy as np
import pytest

from scripts.benchmark_ding_radau5_symbolic_ab import eliminate_affine_stages
from scripts.validate_ding_radau5_condensation import radau5_tableau


def test_exact_substitution_retains_path_constraints_bounds_jacobian_and_hessian():
    _, tableau = radau5_tableau()
    x = ca.SX.sym("x", 12)
    a = ca.DM(np.eye(5) + tableau / 3)
    # Two coupled linear stage blocks, with retained initial states x[0:2].
    defects = ca.vertcat(a @ x[2:7] - x[0], a @ x[7:12] - x[1])
    objective = ca.sumsqr(ca.sin(x))
    path = x[2] * x[7] + x[0] ** 2
    nlp = {"x": x, "f": objective, "g": ca.vertcat(defects, path)}
    limits = {"lbg": np.r_[np.zeros(10), -1], "ubg": np.r_[np.zeros(10), 2]}
    result = eliminate_affine_stages(nlp, range(2, 12), limits)
    y = result["nlp"]["x"]
    reduced_g = result["nlp"]["g"]
    assert y.numel() == 2
    assert reduced_g.numel() == 11  # Original path + all 10 reconstructed bounds.
    full_x, full_g = result["reconstruct"]([0.3, 0.7])
    np.testing.assert_allclose(np.asarray(full_g)[:10], 0, atol=1e-15)
    residual, derivative = result["audit"]([0.3, 0.7])
    np.testing.assert_allclose(residual, 0, atol=1e-15)
    np.testing.assert_allclose(derivative, 0, atol=1e-15)
    # Independently construct the known affine physical-state map.
    ones = np.linalg.solve(np.array(a), np.ones(5))
    projection = np.zeros((12, 2))
    projection[:2] = np.eye(2)
    projection[2:7, 0] = ones
    projection[7:12, 1] = ones
    # Include a nonlinear path multiplier: the guard covers the Lagrangian
    # Hessian, not only an objective Hessian unaffected by constraint handling.
    lagrangian = objective + 0.73 * path
    reduced_lagrangian = result["nlp"]["f"] + 0.73 * reduced_g[0]
    reference = ca.Function("reference", [x], [ca.gradient(lagrangian, x), ca.hessian(lagrangian, x)[0]])
    transformed = ca.Function("transformed", [y], [ca.gradient(reduced_lagrangian, y),
                                                  ca.hessian(reduced_lagrangian, y)[0]])
    grad_full, hess_full = reference(full_x)
    grad_reduced, hess_reduced = transformed([0.3, 0.7])
    np.testing.assert_allclose(grad_reduced, projection.T @ grad_full, atol=1e-14)
    np.testing.assert_allclose(hess_reduced, projection.T @ hess_full @ projection, atol=1e-14)


def test_rejects_nonlinear_or_incomplete_elimination():
    x = ca.SX.sym("x", 6)
    g = ca.vertcat(*[ca.sum1(x[1:6]) + x[i + 1] + x[0] ** 2 for i in range(5)])
    with pytest.raises(ValueError, match="affine Radau rows"):
        eliminate_affine_stages({"x": x, "f": x[0], "g": g}, range(1, 6),
                                {"lbg": np.zeros(5), "ubg": np.zeros(5)})
