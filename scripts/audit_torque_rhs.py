"""Compare full-model RHS values for external and generalized crank torques.

This diagnostic deliberately avoids solving an OCP.  It builds the two model
representations used by K1 and K2 and evaluates their continuous dynamics at
identical, perturbed admissible initial states.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = ROOT / "examples" / "fes_multibody" / "cycling"
sys.path.insert(0, str(EXAMPLE_DIR))

import cycling_pulse_width_mhe_acados_periodic as cycling  # noqa: E402


def _arguments(torque_application: str):
    return cycling.build_argument_parser().parse_args(
        [
            "--solver", "ipopt",
            "--model-formulation", "standard",
            "--mechanical-formulation", "full",
            "--torque-application", torque_application,
            "--stimulations-per-cycle", "50",
            "--cycles-per-window", "1",
            "--n-windows", "1",
            "--n-threads", "1",
            "--constant-crank-torque", "0.1",
            "--ode-solver", "collocation",
            "--collocation-degree", "5",
            "--collocation-method", "radau",
            "--disable-historical-ipopt-initial-guess",
        ]
    )


def _initial_vector(nlp, container, variables) -> np.ndarray:
    """Extract the first numerical initial-guess column from Bioptim."""
    first_key = next(iter(container.keys()))
    n_nodes = int(container[first_key].init.shape[1])
    return cycling._stack_initial_guess_values(container, variables, n_nodes)[:, :1]


def _rhs(nlp, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    data = cycling._numerical_timeseries_at_node(nlp, 0)
    return cycling._full_dynamics_rhs(nlp, 0.0, 1.0 / 50.0, x, u, data).reshape((-1, 1))


def main() -> None:
    external = cycling.build_unilateral_runtime(_arguments("external_forces"), echo=False)["nmpc"].nlp[0]
    constant = cycling.build_unilateral_runtime(_arguments("constant"), echo=False)["nmpc"].nlp[0]
    x0 = _initial_vector(external, external.x_init, external.states)
    u0 = _initial_vector(external, external.u_init, external.controls)
    if (
        x0.shape != _initial_vector(constant, constant.x_init, constant.states).shape
        or u0.shape != _initial_vector(constant, constant.u_init, constant.controls).shape
    ):
        raise RuntimeError("The two torque representations have incompatible state/control dimensions")

    rng = np.random.default_rng(20260922)
    # Small variations around the same admissible construction point.  The
    # dynamics are compared before any nonlinear optimization is performed.
    scales = np.maximum(np.abs(x0), 1.0)
    samples = [x0] + [x0 + 0.03 * scales * rng.standard_normal(x0.shape) for _ in range(31)]
    differences = []
    for x in samples:
        rhs_external = _rhs(external, x, u0)
        rhs_constant = _rhs(constant, x, u0)
        differences.append(float(np.max(np.abs(rhs_external - rhs_constant))))

    print({
        "sample_count": len(samples),
        "state_dimension": int(x0.size),
        "control_dimension": int(u0.size),
        "external_timeseries_dimension": int(cycling._numerical_timeseries_at_node(external, 0).size),
        "constant_timeseries_dimension": int(cycling._numerical_timeseries_at_node(constant, 0).size),
        "maximum_absolute_rhs_difference": max(differences),
        "median_absolute_rhs_difference": float(np.median(differences)),
        "per_sample_maximums": differences,
    })


if __name__ == "__main__":
    main()
