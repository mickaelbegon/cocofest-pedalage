"""Exact callback compilation after a parameterized affine elimination.

The incoming Radau offsets must stay dynamic between repeated NLP calls.
These tests catch compilation in the wrong decision space, stale offsets and
loss of the affine chain rule in Jacobians and Lagrangian Hessians.
"""
import casadi as ca
import numpy as np
import pytest

from scripts.benchmark_ding_local_compiled import compiled_factory


@pytest.mark.parametrize("callbacks", ["hessian", "all"])
def test_compiled_affine_reduction_callbacks_and_repeated_solves(tmp_path, callbacks):
    y, offsets = ca.SX.sym("y", 2), ca.SX.sym("offsets", 2)
    full = ca.vertcat(y[0], y[1], 0.15 * y[0] + offsets[0], -0.1 * y[1] + offsets[1])
    x = ca.SX.sym("x", 4)
    f_full = (x[0] - 0.7) ** 2 + (x[1] - 0.2) ** 2 + 0.1 * ca.exp(x[2]) + x[3] ** 2
    g_full = ca.vertcat(x[0] + ca.sin(x[2]) - x[1], x[1] ** 2 + x[3])
    nlp = dict(x=y, p=offsets, f=ca.substitute(f_full, x, full), g=ca.substitute(g_full, x, full))
    opts = {"ipopt.print_level": 0, "print_time": False, "ipopt.tol": 1e-10}
    vm = ca.nlpsol("reference", "ipopt", nlp, opts)
    report = {}
    native = compiled_factory(ca.nlpsol, cache_dir=tmp_path / "cache", callbacks=callbacks,
                              report=report)("compiled", "ipopt", nlp, opts)
    assert report["solver_nx"] == 2
    assert report["runtime_parameter_count"] == 2
    assert report["compiler_flags"] == ["-O1"]
    for name in report["compilation"]["native_callbacks"]:
        assert native.get_function(name).class_name() == "External"
    projection = ca.jacobian(full, y)
    lam = ca.SX.sym("lambda", 2)
    hess_full = ca.hessian(1.7 * f_full + ca.dot(lam, g_full), x)[0]
    hess_chain = ca.Function("chain", [y, offsets, lam],
                            [projection.T @ ca.substitute(hess_full, x, full) @ projection])
    for p in ([0.3, 0.2], [-0.2, 0.7], [0.8, -0.3]):
        point = [0.4, 0.6]
        for name in ("nlp_f", "nlp_g", "nlp_grad_f", "nlp_jac_g", "nlp_hess_l"):
            inputs = [point, p] + ([1.7, [0.8, -0.2]] if name == "nlp_hess_l" else [])
            expected = vm.get_function(name).call(inputs)
            got = native.get_function(name).call(inputs)
            for a, b in zip(expected, got):
                np.testing.assert_allclose(np.array(a), np.array(b), rtol=1e-13, atol=1e-13)
        upper = np.array(native.get_function("nlp_hess_l")(point, p, 1.7, [0.8, -0.2]))
        np.testing.assert_allclose(upper + upper.T - np.diag(np.diag(upper)),
                                   np.array(hess_chain(point, p, [0.8, -0.2])), rtol=1e-13, atol=1e-13)
        inputs = dict(x0=point, p=p, lbx=[-2, -2], ubx=[2, 2], lbg=[-5, -5], ubg=[5, 5])
        a, b = vm(**inputs), native(**inputs)
        assert vm.stats()["success"] and native.stats()["success"]
        np.testing.assert_allclose(np.array(a["x"]), np.array(b["x"]), rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(np.array(a["f"]), np.array(b["f"]), rtol=1e-10, atol=1e-10)
    # A second factory rebuild reuses the content-addressed native callbacks.
    reused = {}
    compiled_factory(ca.nlpsol, cache_dir=tmp_path / "cache", callbacks=callbacks,
                     report=reused)("compiled_again", "ipopt", nlp, opts)
    assert reused["compilation"]["hit"]
