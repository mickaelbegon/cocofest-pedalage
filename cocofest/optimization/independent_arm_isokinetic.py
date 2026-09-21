"""Contracts for two independent, synchronized isokinetic arm simulations.

This module deliberately contains no OCP construction.  It is the small
boundary between two unilateral optimizers and a caller that launches them in
parallel.  Keeping the boundary numerical (rather than adding a bilateral
mechanical model) is what guarantees that a state or a control from one arm
cannot enter the other arm's NLP.

In the current isokinetic reduced-mechanics formulation the instantaneous load
is an *inverse-dynamics output*, not a prescribed external torque.  The
independently adjustable quantity is consequently the terminal work target,
represented equivalently by its cycle-mean resistive torque:
``W_i = tau_bar_i * 2*pi*n_turns``.  That target is implemented as bounds on
the existing ``E_prod`` state, so it can be changed between solves without
regenerating either solver.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isclose, isfinite
from typing import Literal, Mapping


ArmSide = Literal["right", "left"]


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class IndependentArmRequest:
    """Numerical input for one unilateral problem.

    ``equivalent_mean_resistance_nm`` means imposed work per cycle divided by
    ``2*pi``. It is not a prescribed instantaneous external torque.
    """

    side: ArmSide
    isokinetic_omega_rad_s: float
    phase_origin_rad: float = 0.0
    equivalent_mean_resistance_nm: float = 0.0
    turns: int = 1

    def __post_init__(self) -> None:
        if self.side not in ("right", "left"):
            raise ValueError("side must be 'right' or 'left'")
        omega = _finite(self.isokinetic_omega_rad_s, "isokinetic_omega_rad_s")
        if omega >= 0:
            raise ValueError("isokinetic_omega_rad_s must be negative in the cycling convention")
        object.__setattr__(self, "isokinetic_omega_rad_s", omega)
        object.__setattr__(self, "phase_origin_rad", _finite(self.phase_origin_rad, "phase_origin_rad"))
        resistance = _finite(self.equivalent_mean_resistance_nm, "equivalent_mean_resistance_nm")
        if resistance < 0:
            raise ValueError("equivalent_mean_resistance_nm must be nonnegative")
        if isinstance(self.turns, bool) or not isinstance(self.turns, int) or self.turns < 1:
            raise ValueError("turns must be a positive integer")
        object.__setattr__(self, "equivalent_mean_resistance_nm", resistance)

    @property
    def imposed_work_j(self) -> float:
        """Positive work requested for this arm over its RHO window."""
        from math import pi
        return self.equivalent_mean_resistance_nm * 2.0 * pi * self.turns


@dataclass(frozen=True)
class IndependentArmOutcome:
    """Minimal solver-neutral result used for a bilateral synthesis."""

    request: IndependentArmRequest
    produced_work_j: float
    final_capacity_ratio: float
    maximum_fatigue: float
    feasible: bool
    objective: float
    solver_time_s: float
    wall_time_s: float

    def __post_init__(self) -> None:
        for name in ("produced_work_j", "final_capacity_ratio", "maximum_fatigue",
                     "objective", "solver_time_s", "wall_time_s"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if not 0 <= self.final_capacity_ratio:
            raise ValueError("final_capacity_ratio must be nonnegative")
        if not 0 <= self.maximum_fatigue:
            raise ValueError("maximum_fatigue must be nonnegative")
        if self.solver_time_s < 0 or self.wall_time_s < 0:
            raise ValueError("timings must be nonnegative")


def synthesize_independent_arms(right: IndependentArmOutcome,
                                left: IndependentArmOutcome) -> dict:
    """Combine independent outcomes without inventing a coupled mechanics law.

    The elapsed parallel time is the larger wall time; ``serial_wall_time_s``
    is reported separately to make a lost parallelism visible.  Work and
    objectives are additive only because the two OCPs are independent.
    """
    if right.request.side != "right" or left.request.side != "left":
        raise ValueError("outcomes must be passed as (right, left)")
    r, l = right.request, left.request
    if not isclose(r.isokinetic_omega_rad_s, l.isokinetic_omega_rad_s, abs_tol=1e-12):
        raise ValueError("independent arms must use the same isokinetic angular velocity")
    if not isclose(r.phase_origin_rad, l.phase_origin_rad, abs_tol=1e-12):
        raise ValueError("independent arms must use the same phase origin")
    if r.turns != l.turns:
        raise ValueError("independent arms must use the same number of turns per synchronized window")
    return {
        "architecture": "two_independent_unilateral_isokinetic_ocps_v1",
        "synchronized": True,
        "isokinetic_omega_rad_s": r.isokinetic_omega_rad_s,
        "phase_origin_rad": r.phase_origin_rad,
        "right": asdict(right),
        "left": asdict(left),
        "both_feasible": bool(right.feasible and left.feasible),
        "total_produced_work_j": right.produced_work_j + left.produced_work_j,
        "total_objective": right.objective + left.objective,
        "worst_final_capacity_ratio": min(right.final_capacity_ratio, left.final_capacity_ratio),
        "maximum_fatigue": max(right.maximum_fatigue, left.maximum_fatigue),
        "total_solver_time_s": right.solver_time_s + left.solver_time_s,
        "parallel_wall_time_s": max(right.wall_time_s, left.wall_time_s),
        "serial_wall_time_s": right.wall_time_s + left.wall_time_s,
        "right_imposed_work_j": r.imposed_work_j,
        "left_imposed_work_j": l.imposed_work_j,
        "equivalent_resistance_imbalance_nm": (
            r.equivalent_mean_resistance_nm - l.equivalent_mean_resistance_nm),
    }


def runtime_resistance_contract(backend: str, *, terminal_work_bound_updatable: bool) -> Mapping[str, object]:
    """State precisely whether a resistance change avoids recompilation.

    The existing isokinetic OCP represents imposed work by fixing the terminal
    ``E_prod`` state bound. IPOPT receives fresh ``lbx/ubx`` at every solve;
    ACADOS exposes the same existing bounds through ``constraints_set``. No
    symbolic graph, C code, or solver instance need be rebuilt when only that
    terminal bound changes. This is explicitly not a claim that an arbitrary
    instantaneous load trace can be changed without recompilation.
    """
    backend = str(backend).lower()
    if backend not in {"ipopt", "acados"}:
        raise ValueError("backend must be 'ipopt' or 'acados'")
    update = ("replace the CasADi NLP parameter vector before solve"
              if backend == "ipopt" else
              "update the generated solver stage parameter p before solve")
    return {
        "backend": backend,
        "runtime_equivalent_resistance_change_supported": bool(terminal_work_bound_updatable),
        "requires_recompilation": not bool(terminal_work_bound_updatable),
        "required_update": update if terminal_work_bound_updatable else None,
        "current_model_status": (
            "requires mutable terminal E_prod bounds in the selected solver wrapper"
            if not terminal_work_bound_updatable else
            "replace terminal E_prod lbx/ubx and its initial guess; instantaneous load remains inverse-dynamics output"),
    }
