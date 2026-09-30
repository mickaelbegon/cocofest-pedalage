"""Contract tests for the opt-in frozen-RHO MA57 restoration profile."""

import pytest

from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as periodic


def test_ma57_tuned_recovery_is_opt_in_and_numerically_distinct():
    baseline = periodic.build_argument_parser().parse_args([])
    assert periodic._ipopt_recovery_advanced_options(baseline) == {}

    args = periodic.build_argument_parser().parse_args([
        "--solver", "ipopt",
        "--ipopt-linear-solver", "ma57",
        "--nlp-ipopt-recovery",
        "--nlp-ipopt-recovery-linear-solver", "ma57",
        "--nlp-ipopt-recovery-ma57-tuned",
    ])

    target = periodic._ipopt_advanced_options(args)
    recovery = periodic._ipopt_recovery_advanced_options(args)
    assert recovery == {
        "linear_system_scaling": "mc19",
        "ma57_automatic_scaling": "yes",
        "ma57_pivtol": 1e-6,
        "ma57_pivtolmax": 1e-2,
        "ma57_pre_alloc": 1.2,
    }
    assert any(target.get(name) != value for name, value in recovery.items())
    periodic.validate_tuned_ma57_recovery_options(args)


def test_ma57_tuned_recovery_rejects_missing_recovery_or_identical_target_profile():
    missing_recovery = periodic.build_argument_parser().parse_args([
        "--nlp-ipopt-recovery-ma57-tuned",
    ])
    with pytest.raises(ValueError, match="requires --nlp-ipopt-recovery"):
        periodic.validate_tuned_ma57_recovery_options(missing_recovery)

    identical = periodic.build_argument_parser().parse_args([
        "--solver", "ipopt", "--ipopt-linear-solver", "ma57",
        "--nlp-ipopt-recovery", "--nlp-ipopt-recovery-linear-solver", "ma57",
        "--nlp-ipopt-recovery-ma57-tuned",
        "--ipopt-linear-system-scaling", "mc19",
        "--ipopt-ma57-automatic-scaling",
        "--ipopt-ma57-pivtol", "1e-6",
        "--ipopt-ma57-pivtolmax", "1e-2",
        "--ipopt-ma57-pre-alloc", "1.2",
    ])
    with pytest.raises(ValueError, match="must differ"):
        periodic.validate_tuned_ma57_recovery_options(identical)
