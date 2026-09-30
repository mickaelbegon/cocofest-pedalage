"""Independent exact AD reference, cross-boundary terms, and IPOPT contract."""

import casadi as ca
import numpy as np
import pytest
import shutil
import subprocess

from cocofest.optimization.packet_hessian import HessianPacket, build_packet_hessian


def _problem():
    z = ca.MX.sym("z", 3)
    p = ca.MX.sym("p")
    # Bilinear boundary term plus a shared global variable: this is not block diagonal.
    kernel = ca.Function("local_kernel", [z, p],
                         [(z[0] - p[0]) ** 4 + z[2] ** 2,
                          ca.vertcat(z[0] * z[1] + ca.sin(z[2]))])
    packets = [HessianPacket(kernel, (0, 1, 4), (1,)),
               HessianPacket(kernel, (2, 1, 4), (0,)),
               HessianPacket(kernel, (3, 2, 4), (2,))]
    x = ca.MX.sym("x", 5)
    f, g = 0, ca.MX.zeros(3, 1)
    for packet in packets:
        local_f, local_g = kernel(x[list(packet.variables)], p)
        f += local_f
        g[list(packet.constraints)] = local_g
    return packets, x, p, f, g


@pytest.mark.parametrize("threads", [1, 3])
def test_matches_monolithic_exact_hessian_with_overlap_and_permuted_indices(threads):
    packets, x, p, f, g = _problem()
    block = build_packet_hessian(packets, n_variables=5, n_constraints=3, n_parameters=1, threads=threads)
    sigma, lam = ca.MX.sym("sigma"), ca.MX.sym("lam", 3)
    exact = ca.Function("exact", [x, p, sigma, lam], [ca.triu(ca.hessian(sigma * f + ca.dot(lam, g), x)[0])])
    assert block.function.sparsity_out(0) == exact.sparsity_out(0)
    assert block.group_sizes == (3,)
    random = np.random.default_rng(73)
    for _ in range(8):
        inputs = [random.normal(size=5), random.normal(size=1), random.normal(), random.normal(size=3)]
        np.testing.assert_allclose(block.function(*inputs), exact(*inputs), rtol=1e-12, atol=1e-12)
    # Off-diagonal entries bridging nominal packets really are retained.
    assert block.function.sparsity_out(0).has_nz(1, 2)


def test_rejects_duplicate_or_missing_constraint_ownership():
    packets, *_ = _problem()
    with pytest.raises(ValueError, match="every constraint exactly once"):
        build_packet_hessian(packets[:2], n_variables=5, n_constraints=3, n_parameters=1)
    with pytest.raises(ValueError, match="every constraint exactly once"):
        build_packet_hessian([*packets, packets[0]], n_variables=5, n_constraints=3, n_parameters=1)


def test_heterogeneous_packet_and_linear_packet_have_exact_sparse_union():
    x = ca.MX.sym("x", 2)
    p = ca.MX.sym("p", 0)
    z = ca.MX.sym("z", 2)
    nonlinear = ca.Function("nonlinear", [z, p], [z[0] ** 2, ca.vertcat(z[0] * z[1])])
    linear = ca.Function("linear", [z, p], [z[1], ca.vertcat(z[0] + z[1])])
    packets = [HessianPacket(nonlinear, (0, 1), (0,)), HessianPacket(linear, (0, 1), (1,))]
    result = build_packet_hessian(packets, n_variables=2, n_constraints=2, threads=2)
    np.testing.assert_allclose(result.function([2, 3], [], 4, [5, 6]), [[8, 5], [0, 0]])
    assert result.upper_nnz == 2


def test_ipopt_accepts_signature_and_solves_same_problem():
    z = ca.MX.sym("z", 2)
    p = ca.MX.sym("p", 0)
    f = (z[0] - 1) ** 2 + (z[1] - 2) ** 2
    g = ca.vertcat(z[0] * z[1] - 2)
    kernel = ca.Function("solve_kernel", [z, p], [f, g])
    callback = build_packet_hessian([HessianPacket(kernel, (0, 1), (0,))], n_variables=2, n_constraints=1)
    nlp = {"x": z, "p": p, "f": f, "g": g}
    common = {"ipopt.print_level": 0, "print_time": False, "ipopt.tol": 1e-10}
    exact = ca.nlpsol("reference", "ipopt", nlp, common)
    packet = ca.nlpsol("packets", "ipopt", nlp, {**common, "hess_lag": callback.function})
    args = {"x0": [0.8, 1.8], "lbg": 0, "ubg": 0}
    expected, actual = exact(**args), packet(**args)
    assert packet.stats()["success"]
    np.testing.assert_allclose(actual["x"], expected["x"], atol=1e-10)
    np.testing.assert_allclose(actual["f"], expected["f"], atol=1e-12)


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc is required for the local codegen test")
def test_ipopt_accepts_compiled_local_exact_hessian_without_compiling_global_mx(tmp_path):
    """The global NLP stays MX; only the already differentiated local block is C."""
    z = ca.MX.sym("z", 2)
    p = ca.MX.sym("p", 0)
    f = (z[0] - 1) ** 2 + (z[1] - 2) ** 2
    g = ca.vertcat(z[0] * z[1] - 2)
    kernel = ca.Function("compiled_solve_kernel", [z, p], [f, g])
    sigma = ca.MX.sym("sigma")
    lam = ca.MX.sym("lam", 1)
    local_lagrangian = sigma * f + ca.dot(lam, g)
    upper = ca.triu(ca.hessian(local_lagrangian, z)[0])
    values = ca.vertcat(*[upper.nz[index] for index in range(upper.nnz())])
    hessian = ca.Function("compiled_local_hess", [z, p, sigma, lam], [values])
    source = tmp_path / "compiled_local_hess.c"
    library = tmp_path / "compiled_local_hess.so"
    generator = ca.CodeGenerator(source.name)
    generator.add(hessian)
    generator.generate(str(tmp_path) + "/")
    subprocess.run(
        ["gcc", "-O3", "-fPIC", "-shared", str(source), "-o", str(library), "-lm"],
        check=True,
        capture_output=True,
        text=True,
    )
    compiled = ca.external(hessian.name(), str(library))
    callback = build_packet_hessian(
        [HessianPacket(kernel, (0, 1), (0,), local_hessian_kernel=compiled)],
        n_variables=2,
        n_constraints=1,
    )
    nlp = {"x": z, "p": p, "f": f, "g": g}
    common = {"ipopt.print_level": 0, "print_time": False, "ipopt.tol": 1e-10}
    reference = ca.nlpsol("compiled_reference", "ipopt", nlp, common)
    packet = ca.nlpsol("compiled_packets", "ipopt", nlp, {**common, "hess_lag": callback.function})
    args = {"x0": [0.8, 1.8], "lbg": 0, "ubg": 0}
    expected, actual = reference(**args), packet(**args)
    assert packet.stats()["success"]
    np.testing.assert_allclose(actual["x"], expected["x"], atol=1e-10)
    np.testing.assert_allclose(actual["f"], expected["f"], atol=1e-12)
