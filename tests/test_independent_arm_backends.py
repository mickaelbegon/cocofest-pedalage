import numpy as np
import pytest
from types import SimpleNamespace

from cocofest.optimization.independent_arm_backends import (
    AcadosIndependentArmSolver,
    BioptimIndependentArmSolver,
    build_driver_ipopt_independent_arms,
    set_terminal_eprod_target,
)


class Bounds:
    def __init__(self):
        self.min = np.array([[0., -1e9, 1.]])
        self.max = np.array([[0., 1e9, 1.]])


class Init:
    def __init__(self): self.init = np.zeros((1, 4))


class Nlp:
    def __init__(self):
        self.x_bounds = {"E_prod": Bounds()}
        self.x_init = {"E_prod": Init()}
        self.states = {"E_prod": type("State", (), {"index": np.array([0])})()}
        self.x_scaling = {"E_prod": type("Scale", (), {"scaling": np.array([[2.]])})()}


class Solution:
    status = 0
    real_time_to_optimize = .01
    cost = 3.


class Nmpc:
    def __init__(self): self.nlp = [Nlp()]
    def solve(self, *, solver, warm_start):
        self.last_warm_start = warm_start
        return Solution()


def test_real_bioptim_adapter_updates_existing_terminal_bound_and_warmstarts():
    nmpc = Nmpc()
    adapter = BioptimIndependentArmSolver(nmpc, solver=object())
    adapter.set_isokinetic_work_target(2.5, .4)
    assert nmpc.nlp[0].x_bounds["E_prod"].min[0, 2] == 2.5
    assert nmpc.nlp[0].x_bounds["E_prod"].max[0, 2] == 2.5
    assert nmpc.nlp[0].x_init["E_prod"].init[0, -1] == 2.5
    outcome = adapter.solve_rho("previous-rho")
    assert nmpc.last_warm_start == "previous-rho"
    assert outcome.metrics["success"]


def test_terminal_update_requires_isokinetic_eprod_shape():
    nmpc = Nmpc()
    nmpc.nlp[0].x_bounds["E_prod"].min = np.zeros((1, 2))
    with pytest.raises(ValueError, match="initial/path/terminal"):
        set_terminal_eprod_target(nmpc, 1.)


def test_acados_pushes_terminal_bound_to_live_native_solver():
    class Native:
        def __init__(self): self.calls = []
        def get(self, stage, field): return np.zeros(1)
        def constraints_set(self, stage, field, value): self.calls.append((stage, field, value.copy()))
    nmpc = Nmpc()
    native = Native()
    nmpc.ocp_solver = type("Interface", (), {
        "ocp_solver": native, "nparams": 0,
        "acados_ocp": type("Ocp", (), {"solver_options": type("Opt", (), {"N_horizon": 4})()})(),
    })()
    adapter = AcadosIndependentArmSolver(nmpc, solver=object())
    adapter.set_isokinetic_work_target(4., .6)
    assert [(stage, field) for stage, field, _ in native.calls] == [(4, "lbx"), (4, "ubx")]
    assert native.calls[0][2][0] == pytest.approx(2.)


def test_public_driver_builder_constructs_two_handles_once_and_updates_in_memory(monkeypatch):
    from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as driver
    constructed = []

    def fake_runtime(args, *, echo=False):
        assert not echo
        nmpc = Nmpc()
        constructed.append(nmpc)
        return {"nmpc": nmpc, "solver": object()}

    monkeypatch.setattr(driver, "build_unilateral_runtime", fake_runtime)
    coordinator = build_driver_ipopt_independent_arms(
        SimpleNamespace(solver="ipopt"), SimpleNamespace(solver="ipopt"),
        right_equivalent_mean_torque_nm=.1, left_equivalent_mean_torque_nm=.2,
    )
    assert len(constructed) == 2
    coordinator.set_equivalent_mean_torques(right_nm=.3, left_nm=.4)
    assert len(constructed) == 2
    assert constructed[0].nlp[0].x_bounds["E_prod"].min[0, 2] == pytest.approx(.3 * 2 * np.pi)
    assert constructed[1].nlp[0].x_bounds["E_prod"].min[0, 2] == pytest.approx(.4 * 2 * np.pi)


def test_rho_session_uses_historical_multi_window_callback_not_repeated_single_solves():
    class SessionNmpc(Nmpc):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def solve_fes_nmpc(self, update, *, solver, total_cycles, **kwargs):
            windows = []
            for index in range(1, total_cycles + 1):
                self.calls += 1
                solution = Solution()
                windows.append(solution)
                if not update(self, index, solution):
                    break
            return windows

    nmpc = SessionNmpc()
    result = BioptimIndependentArmSolver(nmpc, object()).run_rho_cycles(3)
    assert nmpc.calls == 3
    assert result.metrics["requested_rho_cycles"] == 3
    assert result.metrics["returned_windows"] == 3
