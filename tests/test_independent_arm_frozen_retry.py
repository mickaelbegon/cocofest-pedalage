"""Causal retries: no failed state transfer and no duplicate physical cycles."""
from types import SimpleNamespace

import numpy as np
import pytest

from cocofest.optimization.receding_horizon_initial_guess import snapshot_initial_guess
from cocofest.simulation.independent_arms_process import _FrozenRhoRetry, _driver_arguments


class Program:
    def __init__(self):
        self.nlp = [SimpleNamespace(
            x_init={"A_Biceps": SimpleNamespace(init=np.array([[1., .9]]))},
            u_init={"pw": SimpleNamespace(init=np.array([[.1]]))},
            x_bounds={"E_prod": SimpleNamespace(min=np.array([[0., 2.]]), max=np.array([[0., 2.]]))},
            u_bounds={"pw": SimpleNamespace(min=np.array([[.05]]), max=np.array([[.24]]))},
        )]
        self.parameter_init = {"weights": SimpleNamespace(init=np.ones((4, 1)))}
        self.parameter_bounds = {"weights": SimpleNamespace(min=np.ones((4, 1)), max=np.ones((4, 1)))}
        self.ocp_solver = SimpleNamespace(lam_x=np.ones(3), lam_g=np.ones(5))
        self.absolute_wheel_q_cycle_index = 0
        self.advances = []
        # A driver's instance closure must be bypassed because it owns a
        # stale checkpoint when its own preparation callback is never called.
        self.advance_window = lambda *_a, **_k: pytest.fail("stale driver retry closure called")

    def advance_window(self, solution, **kwargs):
        self.advances.append(solution)
        self.absolute_wheel_q_cycle_index += 1
        return "advanced"


class Driver:
    snapshot_initial_guess = staticmethod(snapshot_initial_guess)

    @staticmethod
    def snapshot_nlp_solver_stats(program):
        return {"return_status": "nominal_status"}

    @staticmethod
    def _restore_initial_guess_snapshot(program, checkpoint):
        for category, attr in (("states", "x_init"), ("controls", "u_init")):
            for key, values in checkpoint[category].items():
                getattr(program.nlp[0], attr)[key].init[:, :] = values

    @staticmethod
    def _solution_feasibility_summary(solution, tolerance):
        return {"passes_tolerance": solution.residual <= tolerance}

    @staticmethod
    def _rho_solution_is_certified(status, feasibility):
        return status == 0 and feasibility["passes_tolerance"]

    @staticmethod
    def apply_nlp_dual_warm_start(program, solution, **options):
        assert solution is None and options == {"solver_name": "ipopt", "mode": "off"}
        program.ocp_solver.lam_x = program.ocp_solver.lam_g = None
        return {"mode": "off", "reason": "disabled"}


def setup_retry(attempts=2):
    program = Program()
    retry = _FrozenRhoRetry(program, Driver, tolerance=1e-5, max_attempts=attempts)
    retry.capture()
    return program, retry


def solution(status=0, residual=0.):
    return SimpleNamespace(status=status, residual=residual)


def test_failed_solve_restores_exact_prepared_primal_and_duals_without_advancing():
    program, retry = setup_retry()
    program.nlp[0].x_init["A_Biceps"].init[:] = -99.
    program.nlp[0].u_init["pw"].init[:] = .23
    program.parameter_init["weights"].init[:] = 2.
    failed = solution(status=1, residual=.003)
    program.advance_window(failed)
    np.testing.assert_array_equal(program.nlp[0].x_init["A_Biceps"].init, [[1., .9]])
    np.testing.assert_array_equal(program.nlp[0].u_init["pw"].init, [[.1]])
    np.testing.assert_array_equal(program.parameter_init["weights"].init, np.ones((4, 1)))
    assert program.ocp_solver.lam_x is None and program.ocp_solver.lam_g is None
    assert program.absolute_wheel_q_cycle_index == 0 and program.advances == []
    assert program._cocofest_retry_same_rho_pending
    assert failed._cocofest_target_rho == 1
    assert failed._cocofest_frozen_retry["restored_primal_signature"] == retry.signature

    succeeded = solution()
    assert program.advance_window(succeeded) == "advanced"
    assert succeeded._cocofest_target_rho == 1
    assert succeeded._cocofest_attempt_in_physical_rho == 2
    assert succeeded._cocofest_advanced_physical_rho
    assert retry.completed == 1 and len(program.advances) == 1
    assert not program._cocofest_retry_same_rho_pending


def test_each_preparation_refreshes_checkpoint_including_current_work_target():
    program, retry = setup_retry()
    old_signature = retry.signature
    program.advance_window(solution())
    program.nlp[0].x_init["A_Biceps"].init[:] = [[.9, .8]]
    program.nlp[0].x_bounds["E_prod"].min[0, -1] = 3.
    program.nlp[0].x_bounds["E_prod"].max[0, -1] = 3.
    retry.capture()
    assert old_signature != retry.signature
    program.nlp[0].x_init["A_Biceps"].init[:] = -1.
    failed = solution(status=1)
    program.advance_window(failed)
    np.testing.assert_array_equal(program.nlp[0].x_init["A_Biceps"].init, [[.9, .8]])
    assert program.nlp[0].x_bounds["E_prod"].min[0, -1] == 3.
    assert failed._cocofest_target_rho == 2
    assert len(program.advances) == 1


@pytest.mark.parametrize("status,residual", [(1, 0.), (0, .003)])
def test_terminal_failure_is_visible_to_callback_and_never_advances(status, residual):
    program, retry = setup_retry()
    for attempt in range(1, 3):
        failed = solution(status=status, residual=residual)
        program.advance_window(failed)
        assert failed._cocofest_target_rho == 1
        assert failed._cocofest_attempt_in_physical_rho == attempt
        assert program._cocofest_retry_same_rho_pending is (attempt == 1)
    assert retry.completed == 0 and program.advances == []


def test_changed_physical_bounds_refuse_retry_instead_of_changing_the_problem():
    program, _ = setup_retry()
    program.nlp[0].x_bounds["E_prod"].max[0, -1] = 4.
    with pytest.raises(RuntimeError, match="frozen physical bounds"):
        program.advance_window(solution(status=1))
    assert program.advances == []


def test_secondary_solver_flags_are_rejected_until_adapter_is_validated():
    with pytest.raises(ValueError, match="same IPOPT backend"):
        _driver_arguments({"cycles": 2, "right_equivalent_mean_torque_nm": .1,
            "right_driver_arguments": ["--retry-failed-rho-without-advance", "--ipopt-madnlp-recovery"]}, "right")


class LadderDriver(Driver):
    events = None

    @classmethod
    def _ipopt_recovery_advanced_options(cls, args):
        return {"ma57_pivtol": 1e-6}

    @classmethod
    def reset_cached_nlp_solver_for_option_change(cls, program):
        cls.events.append("reset_capsule")
        return True

    @classmethod
    def run_periodic_nlp_recovery(cls, recovery, target, **options):
        assert recovery is target
        assert options["linear_solver"] == "ma57"
        assert options["ipopt_advanced_options"] == {"ma57_pivtol": 1e-6}
        np.testing.assert_array_equal(recovery.nlp[0].x_init["A_Biceps"].init, [[1., .9]])
        cls.events.append("ma57_tuned")
        recovery.nlp[0].u_init["pw"].init[:] = .12
        return None, {"accepted": True, "seed_injected": True}

    @classmethod
    def apply_failed_rho_micro_pulse_width_perturbation(cls, program, checkpoint, **options):
        assert options["retry_index"] == 0
        # The micro stage must start from the original checkpoint, not the
        # tuned solver's injected .12 seed.
        np.testing.assert_array_equal(program.nlp[0].u_init["pw"].init, [[.1]])
        cls.events.append("micro_pw")
        program.nlp[0].u_init["pw"].init[:] += .001
        return {"applied": True, "protected_states_exact": True}


def test_explicit_ladder_runs_reset_then_tuned_then_micro_with_nominal_certification():
    program = Program()
    LadderDriver.events = []
    args = SimpleNamespace(nlp_ipopt_recovery_ma57_tuned=True,
        ipopt_failed_rho_pw_micro_retry=True, nlp_ipopt_recovery_max_iterations=50)
    retry = _FrozenRhoRetry(program, LadderDriver, tolerance=1e-5, max_attempts=4, recovery_args=args)
    retry.capture()
    stages = []
    for attempt in range(1, 4):
        failed = solution(status=1)
        program.advance_window(failed)
        stages.append(failed._cocofest_frozen_retry["strategy"])
        assert failed._cocofest_frozen_retry["requires_nominal_certification"]
        assert failed._cocofest_nominal_solver_stats["return_status"] == "nominal_status"
        assert program._cocofest_retry_same_rho_pending
        assert program.advances == [] and retry.completed == 0
    assert stages == ["frozen_primal_reset_duals", "ma57_tuned", "direct_pw_micro_perturbation"]
    assert LadderDriver.events == ["reset_capsule", "ma57_tuned", "reset_capsule", "micro_pw"]
    np.testing.assert_array_equal(program.nlp[0].x_init["A_Biceps"].init, [[1., .9]])
    succeeded = solution()
    program.advance_window(succeeded)
    assert succeeded._cocofest_target_rho == 1
    assert succeeded._cocofest_attempt_in_physical_rho == 4
    assert retry.completed == 1 and len(program.advances) == 1


def test_flags_cannot_silently_enable_a_stage_outside_the_declared_attempt_budget():
    args = SimpleNamespace(nlp_ipopt_recovery_ma57_tuned=True, ipopt_failed_rho_pw_micro_retry=True)
    with pytest.raises(ValueError, match="requires max_consecutive_failing >= 4"):
        _FrozenRhoRetry(Program(), Driver, tolerance=1e-5, max_attempts=2, recovery_args=args)


def test_unsupported_micro_seed_exposes_terminal_failure_without_advancing():
    class RefusingDriver(Driver):
        @staticmethod
        def apply_failed_rho_micro_pulse_width_perturbation(*args, **kwargs):
            return {"applied": False, "reason": "unsupported_pulse_width_representation"}

    program = Program()
    args = SimpleNamespace(ipopt_failed_rho_pw_micro_retry=True)
    retry = _FrozenRhoRetry(program, RefusingDriver, tolerance=1e-5, max_attempts=3, recovery_args=args)
    retry.capture()
    program.advance_window(solution(status=1))
    failed = solution(status=1)
    program.advance_window(failed)
    assert failed._cocofest_frozen_retry["retry_refused"] == "unsupported_pulse_width_representation"
    assert not program._cocofest_retry_same_rho_pending
    assert program.advances == []


def test_only_explicit_tuned_ma57_profile_is_accepted_by_process_cli():
    args = _driver_arguments({"cycles": 2, "right_equivalent_mean_torque_nm": .1,
        "right_driver_arguments": ["--retry-failed-rho-without-advance", "--nlp-ipopt-recovery",
            "--nlp-ipopt-recovery-linear-solver", "ma57", "--nlp-ipopt-recovery-ma57-tuned",
            "--ipopt-failed-rho-pw-micro-retry", "--max-consecutive-failing", "4"]}, "right")
    assert args.nlp_ipopt_recovery_ma57_tuned
    assert args.ipopt_failed_rho_pw_micro_retry


def test_micro_invariant_exception_restores_checkpoint_and_reports_terminal_refusal():
    class UnsafeDriver(Driver):
        @staticmethod
        def apply_failed_rho_micro_pulse_width_perturbation(program, checkpoint, **kwargs):
            program.nlp[0].u_init["pw"].init[:] = 7.
            raise RuntimeError("outside active bounds")

    program = Program()
    retry = _FrozenRhoRetry(program, UnsafeDriver, tolerance=1e-5, max_attempts=3,
        recovery_args=SimpleNamespace(ipopt_failed_rho_pw_micro_retry=True))
    retry.capture()
    program.advance_window(solution(status=1))
    failed = solution(status=1)
    program.advance_window(failed)
    assert failed._cocofest_frozen_retry["retry_refused"] == "micro_pw_invariant_refused"
    np.testing.assert_array_equal(program.nlp[0].u_init["pw"].init, [[.1]])
    assert not program._cocofest_retry_same_rho_pending and not program.advances


def test_terminal_zero_objective_probe_is_diagnostic_and_restores_the_checkpoint():
    class ProbeDriver(Driver):
        @staticmethod
        def run_frozen_rho_zero_objective_feasibility_probe(program, **options):
            assert options["linear_solver"] == "ma57"
            program.nlp[0].u_init["pw"].init[:] = .23
            return {"objective": "identically_zero", "feasible_witness": False}

    program = Program()
    args = SimpleNamespace(ipopt_frozen_rho_feasibility_probe=True,
        nlp_ipopt_recovery_max_iterations=50)
    retry = _FrozenRhoRetry(program, ProbeDriver, tolerance=1e-5, max_attempts=1,
        recovery_args=args)
    retry.capture()
    failed = solution(status=1)
    program.advance_window(failed)
    assert failed._cocofest_zero_objective_feasibility_probe["objective"] == "identically_zero"
    np.testing.assert_array_equal(program.nlp[0].u_init["pw"].init, [[.1]])
    assert not program._cocofest_retry_same_rho_pending and not program.advances


def test_feasibility_probe_requires_a_frozen_rho():
    with pytest.raises(ValueError, match="feasibility probes require"):
        _driver_arguments({"cycles": 2, "right_equivalent_mean_torque_nm": .1,
            "right_driver_arguments": ["--ipopt-frozen-rho-feasibility-probe"]}, "right")
