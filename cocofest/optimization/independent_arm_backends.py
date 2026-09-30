"""Concrete IPOPT/ACADOS adapters for independent isokinetic arm OCPs.

The adapters operate on real Bioptim OCP/NMPC objects.  They do not own model
construction: callers supply a unilateral builder for each side, which lets a
right and a left model have distinct Ding parameters while keeping their
compiled solver instances alive.  Updating the per-arm target changes only
the terminal ``E_prod`` bound and its numerical initial guess.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar

import numpy as np

from cocofest.simulation.independent_arms import (
    ArmRunResult,
    IndependentArmConfig,
    IndependentArmCoordinator,
)


def set_terminal_eprod_target(nmpc: Any, target_work_j: float) -> None:
    """Update the existing terminal work equality without graph/codegen work.

    ``prepare_nmpc`` creates ``E_prod`` with three bound columns: initial,
    path and terminal.  Only terminal column 2 is changed.  The initial guess
    is linearly rescaled so the next IPOPT/ACADOS solve starts consistently.
    """
    target_work_j = float(target_work_j)
    if not np.isfinite(target_work_j) or target_work_j < 0:
        raise ValueError("target_work_j must be finite and non-negative.")
    nlp = nmpc.nlp[0]
    # Bioptim's VariableBoundsList is key-addressable but does not implement
    # membership consistently across supported versions.
    if "E_prod" not in nlp.x_bounds.keys() or "E_prod" not in nlp.x_init.keys():
        raise ValueError("The OCP has no E_prod state; use an isokinetic reduced OCP.")
    bounds = nlp.x_bounds["E_prod"]
    if bounds.min.shape[1] < 3 or bounds.max.shape[1] < 3:
        raise ValueError("E_prod bounds must have initial/path/terminal columns.")
    bounds.min[0, 2] = target_work_j
    bounds.max[0, 2] = target_work_j
    guess = nlp.x_init["E_prod"].init
    if guess.shape[1] > 1:
        guess[0, :] = np.linspace(0.0, target_work_j, guess.shape[1])
    else:
        guess[0, 0] = target_work_j


def _capacity_metrics(nmpc: Any, solution: Any) -> dict[str, Any]:
    """Expose the terminal Ding reserve needed by the paced supervisor.

    This reads the already-solved primal trajectory only.  A missing or
    non-standard solution deliberately yields no reserve rather than causing
    the supervisor to invent one; it will hold the last allocation.
    """
    try:
        states = solution.decision_states()
        nlp = nmpc.nlp[0]
        models = getattr(getattr(nlp, "model", None), "muscles_dynamics_model", ())
        scales = {f"A_{model.muscle_name}": float(model.a_scale) for model in models}
        ratios = {}
        for key, scale in scales.items():
            values = np.asarray(states[key], dtype=float).reshape(-1)
            if values.size and np.isfinite(scale) and scale > 0 and np.isfinite(values[-1]):
                ratios[key] = float(values[-1] / scale)
        if not ratios:
            return {}
        return {"capacity_ratios": ratios, "minimum_capacity_ratio": min(ratios.values())}
    except (AttributeError, KeyError, TypeError, ValueError, IndexError):
        return {}


def _solution_metrics(solution: Any, nmpc: Any = None) -> dict[str, Any]:
    status = getattr(solution, "status", None)
    cost = float(getattr(solution, "cost", float("nan")))
    metrics = {
        "status": str(status),
        "success": status in (0, "0", "SUCCESS", "success"),
        "solver_time_s": float(getattr(solution, "real_time_to_optimize", 0.0) or 0.0),
        "cost": cost if np.isfinite(cost) else None,
    }
    if nmpc is not None:
        metrics.update(_capacity_metrics(nmpc, solution))
    return metrics


@dataclass
class BioptimIndependentArmSolver:
    """One real unilateral Bioptim NLP solver, reused across RHO windows."""

    # A full historical session is supported below, but calling solve_rho()
    # independently does not implement the cyclic state/model transfer needed
    # by a paced coordinator. Native solver state also must not cross threads.
    requires_process_rho_pace: ClassVar[bool] = True

    nmpc: Any
    solver: Any
    advance_rho: Callable[[Any, Any], Any] | None = None
    external_force: Any = None
    retain_all_iterations: bool = False
    _solution: Any = None
    _target_work_j: float = 0.0

    def set_isokinetic_work_target(self, target_work_j_per_cycle: float, equivalent_mean_torque_nm: float) -> None:
        # ``equivalent_mean_torque_nm`` is intentionally accepted for audit
        # provenance; the OCP's physical target is its E_prod terminal bound.
        del equivalent_mean_torque_nm
        self._target_work_j = float(target_work_j_per_cycle)
        set_terminal_eprod_target(self.nmpc, self._target_work_j)

    def solve_rho(self, rho_state: Any) -> ArmRunResult:
        warm_start = rho_state if rho_state is not None else self._solution
        solution = self.nmpc.solve(solver=self.solver, warm_start=warm_start)
        self._solution = solution
        next_state = self.advance_rho(self.nmpc, solution) if self.advance_rho else solution
        return ArmRunResult(rho_state=next_state, metrics=_solution_metrics(solution, self.nmpc))

    def run_rho_cycles(self, cycles: int, rho_state: Any = None) -> ArmRunResult:
        """Use the historical NMPC update/advance loop for several RHO windows.

        This is deliberately not a Python loop of independent ``solve`` calls:
        ``solve_fes_nmpc`` invokes the model's established cyclic bound and
        initial-guess transfer after every certified window while retaining the
        same compiled NLP/ACADOS capsule.
        """
        if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles < 1:
            raise ValueError("cycles must be a strictly positive integer.")
        if not hasattr(self.nmpc, "solve_fes_nmpc"):
            # Small test doubles and custom Bioptim wrappers may only expose
            # solve(); preserve truthful one-window behaviour for them.
            result = None
            for _ in range(cycles):
                result = self.solve_rho(rho_state)
                rho_state = result.rho_state
            return result
        try:
            from bioptim import MultiCyclicCycleSolutions
            cycle_solutions = MultiCyclicCycleSolutions.ALL_CYCLES
        except ImportError:  # pragma: no cover - real driver always has Bioptim
            cycle_solutions = None
        completed = []

        def update_functions(_nmpc, cycle_idx, solution):
            completed.append(solution)
            # Compact output never uses the historical aggregate model list;
            # retaining every per-window model duplicates CasADi graphs and
            # makes two concurrent 100-cycle sessions unnecessarily memory
            # hungry. The active model and RHO transfer state remain on nmpc.
            models = getattr(_nmpc, "all_models", None)
            if isinstance(models, list):
                models.clear()
            return cycle_idx < cycles

        solution = self.nmpc.solve_fes_nmpc(
            update_functions,
            solver=self.solver,
            total_cycles=cycles,
            external_force=self.external_force,
            cycle_solutions=cycle_solutions,
            get_all_iterations=self.retain_all_iterations,
            cyclic_options={"states": {}},
            max_consecutive_failing=1,
            compact_solution_output=True,
        )
        # The periodic driver returns a list of per-window solutions.  Keep it
        # as the RHO state so a caller can retain audit/provenance information.
        # The compact aggregate returned by Bioptim deliberately has no NLP
        # status. The callback receives each actual window solution, including
        # its IPOPT/ACADOS status, so use that for certification metadata.
        final = completed[-1] if completed else (solution[-1] if isinstance(solution, (list, tuple)) else solution)
        self._solution = final
        metrics = _solution_metrics(final, self.nmpc)
        window_statuses = [str(getattr(item, "status", None)) for item in completed]
        metrics.update({
            "requested_rho_cycles": cycles,
            "callback_windows": len(completed),
            "returned_windows": len(completed),
            "all_callback_statuses_success": all(
                status in ("0", "SUCCESS", "success") for status in window_statuses
            ),
        })
        return ArmRunResult(rho_state=solution, metrics=metrics)


class AcadosIndependentArmSolver(BioptimIndependentArmSolver):
    """ACADOS variant; terminal E_prod bounds are pushed to its live solver."""

    def set_isokinetic_work_target(self, target_work_j_per_cycle: float, equivalent_mean_torque_nm: float) -> None:
        super().set_isokinetic_work_target(target_work_j_per_cycle, equivalent_mean_torque_nm)
        interface = getattr(self.nmpc, "ocp_solver", None)
        native = getattr(interface, "ocp_solver", None)
        if native is None:
            # Before the first solve Bioptim will export the OCP once.  This is
            # construction, not a target-induced regeneration.
            return
        nlp = self.nmpc.nlp[0]
        try:
            state_index = int(np.asarray(nlp.states["E_prod"].index).reshape(-1)[0])
            scale = float(np.asarray(nlp.x_scaling["E_prod"].scaling).reshape(-1)[0])
            horizon = int(interface.acados_ocp.solver_options.N_horizon)
            nparams = int(getattr(interface, "nparams", 0))
            lower = np.asarray(native.get(horizon, "lbx"), dtype=float).copy()
            upper = np.asarray(native.get(horizon, "ubx"), dtype=float).copy()
            index = nparams + state_index
            lower[index] = upper[index] = float(target_work_j_per_cycle) / scale
            native.constraints_set(horizon, "lbx", lower)
            native.constraints_set(horizon, "ubx", upper)
        except (AttributeError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Could not update the live ACADOS terminal E_prod bound.") from exc


def _unpack_builder(builder: Callable[[], Any]) -> tuple[Any, Any, Callable[[Any, Any], Any] | None, Any]:
    built = builder()
    if not isinstance(built, tuple) or len(built) not in (2, 3, 4):
        raise TypeError("A unilateral builder must return (nmpc, solver[, advance_rho[, external_force]]).")
    return (
        built[0], built[1], built[2] if len(built) >= 3 else None,
        built[3] if len(built) == 4 else None,
    )


def build_ipopt_independent_arms(
    right_builder: Callable[[], Any], left_builder: Callable[[], Any], *,
    config: IndependentArmConfig = IndependentArmConfig(),
    right_equivalent_mean_torque_nm: float = 0.0,
    left_equivalent_mean_torque_nm: float = 0.0,
) -> IndependentArmCoordinator:
    """Build the two real IPOPT OCPs once, then return their live coordinator."""
    right, left = _unpack_builder(right_builder), _unpack_builder(left_builder)
    return IndependentArmCoordinator(
        BioptimIndependentArmSolver(*right), BioptimIndependentArmSolver(*left), config=config,
        right_equivalent_mean_torque_nm=right_equivalent_mean_torque_nm,
        left_equivalent_mean_torque_nm=left_equivalent_mean_torque_nm,
    )


def build_acados_independent_arms(
    right_builder: Callable[[], Any], left_builder: Callable[[], Any], *,
    config: IndependentArmConfig = IndependentArmConfig(),
    right_equivalent_mean_torque_nm: float = 0.0,
    left_equivalent_mean_torque_nm: float = 0.0,
) -> IndependentArmCoordinator:
    """Build the two real ACADOS OCPs once, then return their live coordinator."""
    right, left = _unpack_builder(right_builder), _unpack_builder(left_builder)
    return IndependentArmCoordinator(
        AcadosIndependentArmSolver(*right), AcadosIndependentArmSolver(*left), config=config,
        right_equivalent_mean_torque_nm=right_equivalent_mean_torque_nm,
        left_equivalent_mean_torque_nm=left_equivalent_mean_torque_nm,
    )


def _driver_unilateral_builder(args: Any) -> Callable[[], tuple[Any, Any]]:
    """Adapt the historical driver public construction seam to a builder."""
    def build() -> tuple[Any, Any, None, Any]:
        from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as driver
        runtime = driver.build_unilateral_runtime(args, echo=False)
        return runtime["nmpc"], runtime["solver"], None, runtime.get("external_force")
    return build


def build_driver_ipopt_independent_arms(
    right_args: Any, left_args: Any, **kwargs: Any,
) -> IndependentArmCoordinator:
    """Build two real IPOPT unilateral OCPs through the historical driver."""
    if getattr(right_args, "solver", None) != "ipopt" or getattr(left_args, "solver", None) != "ipopt":
        raise ValueError("Both driver namespaces must select solver='ipopt'.")
    return build_ipopt_independent_arms(
        _driver_unilateral_builder(right_args), _driver_unilateral_builder(left_args), **kwargs
    )


def build_driver_acados_independent_arms(
    right_args: Any, left_args: Any, **kwargs: Any,
) -> IndependentArmCoordinator:
    """Build two real ACADOS unilateral OCPs through the historical driver."""
    if getattr(right_args, "solver", None) != "acados" or getattr(left_args, "solver", None) != "acados":
        raise ValueError("Both driver namespaces must select solver='acados'.")
    return build_acados_independent_arms(
        _driver_unilateral_builder(right_args), _driver_unilateral_builder(left_args), **kwargs
    )
