"""Guard the clinical RHO/FHO default mechanical formulation."""

import inspect

from examples.fes_multibody.cycling import cycling_fes_solver_comparison as benchmark
from examples.fes_multibody.cycling import (
    cycling_pulse_width_mhe_acados_periodic as periodic,
)


def test_benchmark_cli_and_api_default_to_reduced_mechanics():
    assert benchmark.build_cli().parse_args([]).mechanical_formulation == "reduced"
    assert (
        inspect.signature(benchmark.main)
        .parameters["mechanical_formulation"].default
        == "reduced"
    )


def test_periodic_rho_cli_defaults_to_reduced_mechanics():
    assert (
        periodic.build_argument_parser().parse_args([]).mechanical_formulation
        == "reduced"
    )
