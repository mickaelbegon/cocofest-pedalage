"""Defaults must not mutate an exact common IPOPT seed before SQP."""
import inspect

import pytest

from examples.fes_multibody.cycling import cycling_fes_solver_comparison as comparison
from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as periodic


@pytest.mark.parametrize("extra", [[], ["--disable-acados-assisted-hot-start"]])
def test_common_ipopt_default_disables_projection_phase_one_homotopy_and_refinement(extra):
    args = periodic.build_argument_parser().parse_args(
        ["--solver", "acados", "--formulation", "dynamic", "--common-initial-solution", "ipopt-cycle-1.npz", *extra]
    )
    assert args.acados_assisted_hot_start is False
    periodic.apply_assisted_hot_start_defaults(args)
    assert args.disable_periodic_fes_warmup_projection is True
    assert args.periodic_fes_warmup_projection_strategy == "sequential"  # no automatic rollout selection
    assert args.full_dynamics_phase_one is False
    assert args.acados_transfer_full_dynamics_rollout is False
    assert args.acados_transfer_irk_rollout is False
    assert args.acados_control_homotopy_radii is None
    assert args.acados_control_homotopy_keep_final_radius is False
    assert args.acados_bind_first_node_fes_states is False
    assert args.periodic_ipopt_refinement is False


def test_cli_and_programmatic_comparison_defaults_match_and_opt_in_remains_available():
    assert comparison.build_cli().parse_args([]).acados_assisted_hot_start is False
    assert inspect.signature(comparison.main).parameters["acados_assisted_hot_start"].default is False
    assert inspect.signature(comparison.main).parameters["periodic_ipopt_refinement"].default is False
    for parser in (comparison.build_cli(), periodic.build_argument_parser()):
        assert parser.parse_args([]).periodic_ipopt_refinement is False
        assert parser.parse_args(["--periodic-ipopt-refinement"]).periodic_ipopt_refinement is True
        assert parser.parse_args(["--acados-assisted-hot-start"]).acados_assisted_hot_start is True
        assert parser.parse_args(["--disable-acados-assisted-hot-start"]).acados_assisted_hot_start is False


def test_explicit_assisted_preparation_retains_its_existing_behavior():
    args = periodic.build_argument_parser().parse_args([
        "--common-initial-solution", "ipopt-cycle-1.npz", "--acados-assisted-hot-start",
    ])
    periodic.apply_assisted_hot_start_defaults(args)
    assert args.disable_periodic_fes_warmup_projection is False
    assert args.periodic_fes_warmup_projection_strategy == "rollout"
    assert args.full_dynamics_phase_one is True
    assert args.acados_transfer_irk_rollout is True
    assert args.acados_control_homotopy_radii == periodic.DEFAULT_ASSISTED_CONTROL_HOMOTOPY_RADII
    assert args.acados_control_homotopy_keep_final_radius is True


def test_no_assisted_attribute_does_not_implicitly_opt_in():
    args = periodic.build_argument_parser().parse_args(["--common-initial-solution", "ipopt-cycle-1.npz"])
    del args.acados_assisted_hot_start
    periodic.apply_assisted_hot_start_defaults(args)
    assert args.disable_periodic_fes_warmup_projection is True
    assert args.full_dynamics_phase_one is False
    assert args.acados_control_homotopy_radii is None


def test_ipopt_preparation_and_explicit_individual_options_are_not_reset():
    ipopt = periodic.build_argument_parser().parse_args(["--solver", "ipopt", "--common-initial-solution", "common.npz"])
    prior_projection = ipopt.disable_periodic_fes_warmup_projection
    periodic.apply_assisted_hot_start_defaults(ipopt)
    assert ipopt.disable_periodic_fes_warmup_projection is prior_projection
    acados = periodic.build_argument_parser().parse_args([
        "--common-initial-solution", "ipopt-cycle-1.npz", "--periodic-ipopt-refinement",
    ])
    acados.full_dynamics_phase_one = True
    acados.acados_control_homotopy_radii = (.01, .1)
    periodic.apply_assisted_hot_start_defaults(acados)
    assert acados.full_dynamics_phase_one is True
    assert acados.acados_control_homotopy_radii == (.01, .1)
    assert acados.periodic_ipopt_refinement is True
