"""Serializable, dependency-free inputs shared by launchers and the desktop GUI.

Units are explicit: pulse-width slew is in microseconds, torque in N.m, and
angular velocity in rad/s. ``cycles`` is the executed cycle count; FHO uses
that entire count as its single optimization window.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class SimulationConfig:
    schema_version: int = 1
    mode: str = "rho"
    solver: str = "ipopt"
    mechanics: str = "reduced"
    bilateral_reduced: bool = False
    formulation: str = "dynamic"
    cycles: int = 100
    cycles_per_window: int = 1
    stimulations_per_cycle: int = 30
    signed_crank_torque: float = 0.1
    pulse_width_max_step_us: float | None = None
    pulse_width_slew_formulation: str = "lifting"
    pulse_width_slew_weight: float = 0.0
    pulse_width_slew_reference_us: float = 100.0
    reduced_internal_crank_velocity_guard: str = "auto"
    reduced_terminal_half_step_velocity_guard: bool = False
    terminal_q_slack: float = 0.002
    integration: str = "radau"
    collocation_degree: int = 5
    ipopt_enforce_start_constraints: bool = True
    acados_sim_stages: int = 4
    acados_sim_steps: int = 5
    acados_qp_solver: str = "auto"
    acados_ding_local_reduction: bool = False
    ipopt_ding_local_reduction: bool = False
    ipopt_linear_solver: str = "ma57"
    madnlp_linear_solver: str = "mumps"
    acados_ipopt_cycle1_seed: str | None = None
    common_initial_solution: str | None = None
    threads: int = 1
    numeric_threads: int = 1
    output_root: str = "pycharm-results"
    compile_evaluators: bool = True
    compile_hessian_only: bool = False
    compact_rho_output: bool = True
    madnlp_hot_max_iterations: int = 100
    madnlp_hot_max_wall_time: float = 20
    madnlp_recovery: bool = True
    isokinetic_omega: float = -6.283185307179586
    energy_equivalent_torque: float = 0.1
    load_torque_min: float = -10.0
    load_torque_max: float = 10.0
    weights_config: str | None = None
    model_config: str | None = None
    dry_run: bool = False
    extra_arguments: tuple[str, ...] = ()

    def __post_init__(self):
        # Keep the frozen configuration deeply stable at its only sequence field.
        if isinstance(self.extra_arguments, list):
            object.__setattr__(self, "extra_arguments", tuple(self.extra_arguments))
        for name in ("output_root", "acados_ipopt_cycle1_seed", "common_initial_solution",
                     "weights_config", "model_config"):
            value = getattr(self, name)
            if isinstance(value, Path):
                object.__setattr__(self, name, str(value))

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["extra_arguments"] = list(self.extra_arguments)
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SimulationConfig:
        if not isinstance(data, Mapping):
            raise ValueError("Simulation configuration must be an object")
        unknown = set(data) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown simulation fields: {sorted(unknown)}")
        if data.get("schema_version", 1) != 1 or isinstance(data.get("schema_version"), bool):
            raise ValueError("Unsupported simulation schema_version; expected 1")
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, text: str) -> SimulationConfig:
        return cls.from_dict(json.loads(text))
