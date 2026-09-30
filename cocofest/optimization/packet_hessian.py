"""Experimental exact, sparse packet Hessian for a CasADi/IPOPT NLP.

This module has no production integration. Callers must supply a complete,
non-overlapping partition of constraint *terms*, and each objective term once.
Variables may overlap across packets, including arbitrary global variables.
"""

from dataclasses import dataclass
from typing import Sequence

import casadi as ca


@dataclass(frozen=True)
class HessianPacket:
    """A local ``kernel(z, p) -> (scalar objective, constraint column)``.

    Reuse the same Function object for structurally identical packets to obtain
    a single ThreadMap. ``variables`` indexes the global, scaled NLP vector;
    ``constraints`` indexes the global multiplier vector in NLP ordering.
    Boundary terms must belong to exactly one packet; their variables belong
    to that packet even when they are physically located in a neighboring one.
    """

    kernel: ca.Function
    variables: tuple[int, ...]
    constraints: tuple[int, ...]
    local_hessian_kernel: ca.Function | None = None
    """Optional already-differentiated local Hessian evaluator.

    Its contract is ``(z, p, sigma, lambda) -> hessian_upper_values``.  In
    particular it can be a ``casadi.external`` loaded from C code.  The
    callback builder never differentiates this function: it uses it only for
    numeric evaluation, while the sparse coordinates are recovered from the
    symbolic ``kernel``.  This is the narrow interface that lets a global MX
    NLP stay interpreted while repeated local exact Hessians are compiled.
    """


@dataclass(frozen=True)
class PacketHessian:
    function: ca.Function
    packet_count: int
    group_sizes: tuple[int, ...]
    upper_nnz: int
    parallelization: str
    threads: int


def build_packet_hessian(
    packets: Sequence[HessianPacket],
    *,
    n_variables: int,
    n_constraints: int,
    n_parameters: int = 0,
    threads: int = 1,
    name: str = "packet_hess_lag",
) -> PacketHessian:
    """Build exact ``hess_lag(x,p,lam_f,lam_g) -> triu_hess_gamma``.

    Differentiation happens *before* mapping. Only sparse local upper triangle
    values are mapped. A constant sparse scatter-add assembles their overlapping
    contributions in canonical global CSC order. No C generation or compilation
    is performed. The returned Function is an option to ``ca.nlpsol``:
    ``{"hess_lag": result.function}`` (not an ``ipopt.*`` option).

    Constraint coverage is checked; objective coverage and model equivalence
    require a caller audit against the original NLP before using this callback.
    """
    if n_variables < 1 or n_constraints < 0 or n_parameters < 0 or threads < 1:
        raise ValueError("Invalid NLP dimensions or thread count")
    if not packets:
        raise ValueError("At least one packet is required")
    coverage = []
    groups = {}
    for packet in packets:
        fun = packet.kernel
        if fun.n_in() != 2 or fun.n_out() != 2:
            raise ValueError("Kernel must have exactly two inputs and two outputs")
        expected = [(len(packet.variables), 1), (n_parameters, 1)]
        if any(fun.size_in(i) != shape for i, shape in enumerate(expected)):
            raise ValueError("Kernel inputs must be local variable and global parameter columns")
        if fun.size_out(0) != (1, 1) or fun.size_out(1) != (len(packet.constraints), 1):
            raise ValueError("Kernel outputs must be scalar objective and constraint column")
        if len(set(packet.variables)) != len(packet.variables):
            raise ValueError("Local variable indices must be unique")
        if any(i < 0 or i >= n_variables for i in packet.variables):
            raise ValueError("Variable index outside NLP")
        coverage.extend(packet.constraints)
        # Group by the evaluator actually mapped.  Two packets may share a
        # primal kernel but intentionally use different compiled libraries.
        groups.setdefault(id(packet.local_hessian_kernel or fun), []).append(packet)
    if sorted(coverage) != list(range(n_constraints)):
        raise ValueError("Packets must cover every constraint exactly once")

    x = ca.MX.sym("x", n_variables)
    p = ca.MX.sym("p", n_parameters)
    sigma = ca.MX.sym("lam_f")
    multipliers = ca.MX.sym("lam_g", n_constraints)
    coordinates, values = [], []
    for group_number, group in enumerate(groups.values()):
        first = group[0]
        z = ca.MX.sym("z", len(first.variables))
        local_p = ca.MX.sym("p", n_parameters)
        local_sigma = ca.MX.sym("sigma")
        local_lambda = ca.MX.sym("lambda", len(first.constraints))
        objective, constraints = first.kernel(z, local_p)
        hessian = ca.triu(ca.hessian(local_sigma * objective + ca.dot(local_lambda, constraints), z)[0])
        rows, cols = hessian.sparsity().get_triplet()
        symbolic_local = ca.Function(
            f"{name}_local_{group_number}",
            [z, local_p, local_sigma, local_lambda],
            [ca.vertcat(*[hessian.nz[k] for k in range(hessian.nnz())])],
        )
        local = first.local_hessian_kernel or symbolic_local
        expected_inputs = [
            (len(first.variables), 1),
            (n_parameters, 1),
            (1, 1),
            (len(first.constraints), 1),
        ]
        if local.n_in() != 4 or local.n_out() != 1:
            raise ValueError("Local Hessian evaluator must have four inputs and one output")
        if any(local.size_in(index) != shape for index, shape in enumerate(expected_inputs)):
            raise ValueError("Local Hessian evaluator inputs do not match its packet")
        if local.size_out(0) != (hessian.nnz(), 1):
            raise ValueError("Local Hessian evaluator output does not match symbolic upper Hessian")
        for packet in group:
            if (
                len(packet.variables) != len(first.variables)
                or len(packet.constraints) != len(first.constraints)
            ):
                raise ValueError("Mapped Hessian packets must have identical local dimensions")
            if (packet.local_hessian_kernel or symbolic_local) is not local:
                raise ValueError("Mapped Hessian packets must share an evaluator")
        count = len(group)
        mapped = local.map(count, "thread", min(threads, count)) if threads > 1 else local.map(count, "serial")
        result = mapped(
            ca.horzcat(*[x[list(packet.variables)] for packet in group]),
            ca.repmat(p, 1, count),
            ca.repmat(sigma, 1, count),
            ca.horzcat(*[multipliers[list(packet.constraints)] for packet in group]),
        )
        values.append(ca.vec(result))
        for packet in group:
            for row, col in zip(rows, cols):
                a, b = packet.variables[row], packet.variables[col]
                coordinates.append((min(a, b), max(a, b)))

    sparsity = ca.Sparsity.triplet(n_variables, n_variables, [a for a, _ in coordinates], [b for _, b in coordinates])
    rows, cols = sparsity.get_triplet()
    global_offsets = {position: index for index, position in enumerate(zip(rows, cols))}
    scatter_sparsity = ca.Sparsity.triplet(
        sparsity.nnz(), len(coordinates),
        [global_offsets[position] for position in coordinates], list(range(len(coordinates))),
    )
    scatter = ca.DM(scatter_sparsity, [1.0] * len(coordinates))
    upper = ca.MX(sparsity, scatter @ ca.vertcat(*values))
    callback = ca.Function(
        name, [x, p, sigma, multipliers], [upper],
        ["x", "p", "lam_f", "lam_g"], ["triu_hess_gamma"],
    )
    return PacketHessian(callback, len(packets), tuple(len(group) for group in groups.values()),
                         sparsity.nnz(), "thread" if threads > 1 else "serial", threads)
