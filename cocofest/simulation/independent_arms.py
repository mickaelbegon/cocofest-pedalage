"""Two independent, synchronised isokinetic arm simulations.

This module deliberately does *not* use ``bilateral_reduced``.  A hand is a
complete unilateral OCP with its own Ding/RHO history and its own compiled
solver handle.  The only shared quantity is the prescribed crank speed, hence
the common wall-clock duration of a cycle.

The work target (represented by its equivalent mean resistive torque) is a
runtime parameter of a compiled arm handle.  Instantaneous load torque stays
the existing isokinetic inverse-dynamics quantity: it is inferred from the
equilibrium and bounded by the OCP.  An implementation of
:class:`ParametricArmSolver` must update its terminal-work parameter in the
NLP/ACADOS parameter vector (not rebuild a CasADi graph).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from threading import RLock
from typing import Any, Mapping, Protocol, runtime_checkable


ARM_NAMES = ("right", "left")


def _finite_equivalent_torque(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite, non-negative equivalent torque in N.m.")
    return value


@dataclass(frozen=True)
class IndependentArmConfig:
    """Scientific invariants of the independent-arm experiment."""

    formulation: str = "isokinetic"
    omega_rad_s: float = -2.0 * math.pi
    parallel: bool = True

    def __post_init__(self) -> None:
        if self.formulation != "isokinetic":
            raise ValueError("Independent arms are defined only for formulation='isokinetic'.")
        if not math.isfinite(float(self.omega_rad_s)) or self.omega_rad_s >= 0.0:
            raise ValueError("omega_rad_s must be finite and strictly negative.")

    @property
    def cycle_duration_s(self) -> float:
        return 2.0 * math.pi / abs(float(self.omega_rad_s))


@dataclass(frozen=True)
class ArmRuntimeParameters:
    """Mutable-between-runs target work of one arm.

    With the project convention ``Δtheta=-2π`` and negative cycling speed,
    positive ``equivalent_mean_torque_nm`` is resistive and requests positive
    mechanical work ``W = tau_bar * abs(Δtheta)`` per cycle.  It is *not* a
    prescribed instantaneous crank torque.
    """

    equivalent_mean_torque_nm: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "equivalent_mean_torque_nm", _finite_equivalent_torque(
            self.equivalent_mean_torque_nm, "equivalent_mean_torque_nm"))

    @property
    def target_work_j_per_cycle(self) -> float:
        return self.equivalent_mean_torque_nm * 2.0 * math.pi


@dataclass(frozen=True)
class ArmRunResult:
    """Backend-neutral result returned by one compiled unilateral OCP."""

    rho_state: Any
    metrics: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class ParametricArmSolver(Protocol):
    """A prebuilt unilateral solver with an in-place work-target parameter.

    ``set_isokinetic_work_target`` is called before every solve.  It must only
    update a numeric terminal-work parameter (for example IPOPT ``p`` or
    ACADOS stage parameters); compilation/code generation is prohibited at
    this point.  The solver keeps the native inferred load torque and its
    bounds in place.
    """

    def set_isokinetic_work_target(
        self, target_work_j_per_cycle: float, equivalent_mean_torque_nm: float
    ) -> None: ...

    def solve_rho(self, rho_state: Any) -> ArmRunResult: ...


def _jsonable(value: Any) -> Any:
    """Turn common numpy/scalar objects into JSON without importing numpy."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class IndependentArmCoordinator:
    """Own two compiled unilateral solvers and synchronise their RHO calls."""

    def __init__(
        self,
        right_solver: ParametricArmSolver,
        left_solver: ParametricArmSolver,
        *,
        config: IndependentArmConfig = IndependentArmConfig(),
        right_rho_state: Any = None,
        left_rho_state: Any = None,
        right_equivalent_mean_torque_nm: float = 0.0,
        left_equivalent_mean_torque_nm: float = 0.0,
    ) -> None:
        if right_solver is left_solver:
            raise ValueError("Right and left arms require distinct unilateral solver instances.")
        self.config = config
        self._solvers = {"right": right_solver, "left": left_solver}
        self._rho = {"right": right_rho_state, "left": left_rho_state}
        self._parameters = {
            "right": ArmRuntimeParameters(right_equivalent_mean_torque_nm),
            "left": ArmRuntimeParameters(left_equivalent_mean_torque_nm),
        }
        self._lock = RLock()
        # Establish the parameters exactly once during construction.  Future
        # calls change values in-place on these same solver objects.
        self._apply_parameters()

    @property
    def parameters(self) -> Mapping[str, ArmRuntimeParameters]:
        return dict(self._parameters)

    def set_equivalent_mean_torques(
        self, *, right_nm: float | None = None, left_nm: float | None = None
    ) -> None:
        """Change either arm's work target without recreating solvers."""
        with self._lock:
            candidate = dict(self._parameters)
            if right_nm is not None:
                candidate["right"] = ArmRuntimeParameters(right_nm)
            if left_nm is not None:
                candidate["left"] = ArmRuntimeParameters(left_nm)
            self._parameters = candidate
            self._apply_parameters()

    def _apply_parameters(self) -> None:
        for arm in ARM_NAMES:
            parameter = self._parameters[arm]
            self._solvers[arm].set_isokinetic_work_target(
                parameter.target_work_j_per_cycle,
                parameter.equivalent_mean_torque_nm,
            )

    def _solve_one(self, arm: str, cycles: int) -> ArmRunResult:
        solver = self._solvers[arm]
        if cycles > 1 and hasattr(solver, "run_rho_cycles"):
            return solver.run_rho_cycles(cycles, self._rho[arm])
        result = None
        for _ in range(cycles):
            result = solver.solve_rho(self._rho[arm])
            self._rho[arm] = result.rho_state
        return result

    def run(self, *, cycles: int = 1) -> dict[str, Any]:
        """Execute synchronised RHO progression for both independent arms."""
        if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles < 1:
            raise ValueError("cycles must be a strictly positive integer.")
        with self._lock:
            # Reassert values since third-party solvers may have changed their
            # current parameter vector while solving the previous window.
            self._apply_parameters()
            if self.config.parallel:
                with ThreadPoolExecutor(max_workers=2, thread_name_prefix="independent-arm") as pool:
                    futures = {arm: pool.submit(self._solve_one, arm, cycles) for arm in ARM_NAMES}
                    results = {arm: futures[arm].result() for arm in ARM_NAMES}
            else:
                results = {arm: self._solve_one(arm, cycles) for arm in ARM_NAMES}
            for arm, result in results.items():
                self._rho[arm] = result.rho_state
            summary = self._summary(results)
            summary["completed_rho_cycles"] = cycles
            return summary

    def _summary(self, results: Mapping[str, ArmRunResult]) -> dict[str, Any]:
        arms = {}
        for arm in ARM_NAMES:
            result = results[arm]
            arms[arm] = {
                "equivalent_mean_torque_nm": self._parameters[arm].equivalent_mean_torque_nm,
                "target_work_j_per_cycle": self._parameters[arm].target_work_j_per_cycle,
                "rho_state": _jsonable(result.rho_state),
                "metrics": _jsonable(result.metrics),
            }
        return {
            "architecture": "two-independent-unilateral-isokinetic-ocps",
            "formulation": "isokinetic",
            "omega_rad_s": self.config.omega_rad_s,
            "cycle_duration_s": self.config.cycle_duration_s,
            "parallel": self.config.parallel,
            "arms": arms,
        }

    def run_to_directory(self, output_root: str | Path, *, cycles: int = 1) -> dict[str, Any]:
        """Run and write the stable GUI/CLI result layout.

        ``right/result.json`` and ``left/result.json`` expose arm-local RHO
        state/metrics; ``summary.json`` is their synchronised comparison.
        """
        summary = self.run(cycles=cycles)
        root = Path(output_root)
        for arm in ARM_NAMES:
            path = root / arm / "result.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(summary["arms"][arm], indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return summary
