"""Versioned launch configuration, shared by forms, JSON and campaign plans.

This identifies the launcher inputs, not an archive's physical equivalence:
file contents, environment versions and solver-produced defaults still belong
to the runtime scientific audit. Resolving a configuration never loads a solver.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, fields
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Mapping

from .capabilities import CapabilityRegistry
from .config import SimulationConfig


# These reproduce the feature increments in plan_incremental_solver_ablation.
# All other inputs (frequency, resistance, cycle count, seed) remain explicit.
_COMMON_PROFILE = dict(solver="ipopt", mechanics="reduced", integration="radau",
                       collocation_degree=5, compile_evaluators=False,
                       ipopt_ding_local_reduction=False, acados_ding_local_reduction=False)
PROFILES = {
    "K5": dict(_COMMON_PROFILE, compile_hessian_only=False, compact_rho_output=False,
               pulse_width_max_step_us=None, pulse_width_slew_weight=0.0),
    "K7": dict(_COMMON_PROFILE, compile_hessian_only=True, compact_rho_output=True,
               pulse_width_max_step_us=None, pulse_width_slew_weight=0.0),
    "K9": dict(_COMMON_PROFILE, compile_hessian_only=True, compact_rho_output=True,
               pulse_width_max_step_us=100.0, pulse_width_slew_weight=0.01,
               pulse_width_slew_reference_us=100.0),
}
_GROUPS = {
    "physical": "mechanics bilateral_reduced formulation stimulations_per_cycle signed_crank_torque "
                "isokinetic_omega energy_equivalent_torque load_torque_min load_torque_max weights_config model_config",
    "transcription": "mode cycles_per_window integration collocation_degree ipopt_enforce_start_constraints "
                     "acados_sim_stages acados_sim_steps acados_ding_local_reduction ipopt_ding_local_reduction "
                     "pulse_width_max_step_us pulse_width_slew_formulation pulse_width_slew_weight pulse_width_slew_reference_us "
                     "reduced_internal_crank_velocity_guard reduced_terminal_half_step_velocity_guard "
                     "terminal_q_slack",
    "solver": "solver ipopt_linear_solver madnlp_linear_solver acados_qp_solver compile_evaluators "
              "compile_hessian_only madnlp_hot_max_iterations madnlp_hot_max_wall_time madnlp_recovery",
    "execution": "cycles threads numeric_threads output_root compact_rho_output acados_ipopt_cycle1_seed "
                 "common_initial_solution extra_arguments",
}
_PATHS = ("output_root", "acados_ipopt_cycle1_seed", "common_initial_solution", "weights_config", "model_config")


@lru_cache(maxsize=1)
def _known_advanced_options() -> frozenset[str]:
    """Read literal parser declarations without importing the scientific stack.

    Legacy escape-hatch values remain engine-validated, but misspelled option
    names fail before spawning a process. No argparse abbreviation is accepted.
    """
    root = Path(__file__).resolve().parents[2]
    paths = (
        root / "examples/fes_multibody/cycling/cycling_fes_solver_comparison.py",
        root / "examples/fes_multibody/cycling/cycling_pulse_width_mhe_acados_periodic.py",
        root / "cocofest/optimization/solver_backends.py",
        root / "cocofest/optimization/pulse_width_slew.py",
        root / "cocofest/optimization/endurance_rollout_ocp.py",
        root / "cocofest/optimization/muscle_horizon_ocp.py",
    )
    options = set()
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
                options.update(arg.value for arg in node.args if isinstance(arg, ast.Constant)
                               and isinstance(arg.value, str) and arg.value.startswith("--"))
    return frozenset(options)


@dataclass(frozen=True)
class ResolvedSimulationConfig:
    config: SimulationConfig
    canonical_json: str
    profile: str | None = None

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return {"schema_version": 1, "config_hash": self.config_hash,
                "profile": self.profile, "effective": json.loads(self.canonical_json)}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"


def resolve_config(config: SimulationConfig | Mapping | None = None, *, profile: str | None = None,
                   root: Path | None = None) -> ResolvedSimulationConfig:
    """Resolve explicit fields over defaults; profile conflicts are errors.

    Pass a sparse mapping when selecting a profile. A SimulationConfig is
    already fully explicit, so none of its values is silently overwritten.
    The hash excludes profile provenance and dry-run, includes execution
    parameters, and canonicalizes managed paths relative to the project root.
    """
    if config is not None and not isinstance(config, (SimulationConfig, Mapping)):
        raise ValueError("Simulation configuration must be an object")
    explicit = config.to_dict() if isinstance(config, SimulationConfig) else dict(config or {})
    SimulationConfig.from_dict(explicit)  # Reject unknown keys before merging.
    if profile is not None and profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; choose {', '.join(PROFILES)}")
    preset = PROFILES.get(profile, {})
    conflicts = [key for key, value in preset.items() if key in explicit and explicit[key] != value]
    if conflicts:
        raise ValueError(f"Profile {profile} conflicts with explicit fields: {', '.join(sorted(conflicts))}")
    value = SimulationConfig.from_dict({**preset, **explicit})
    CapabilityRegistry.require_valid(value)
    unknown = [token.split("=", 1)[0] for token in value.extra_arguments
               if token.startswith("--") and token.split("=", 1)[0] not in _known_advanced_options()]
    if unknown:
        raise ValueError(f"Unknown advanced options: {', '.join(unknown)}")
    data = value.to_dict()
    root = (root or Path(__file__).resolve().parents[2]).expanduser().absolute()
    defaults = SimulationConfig()
    for field in fields(defaults):
        if field.type in (float, "float"):
            data[field.name] = float(data[field.name])
    if data["pulse_width_max_step_us"] is not None:
        data["pulse_width_max_step_us"] = float(data["pulse_width_max_step_us"])
    for field in _PATHS:
        if data[field] is not None:
            path = Path(data[field]).expanduser()
            data[field] = str((path if path.is_absolute() else root / path).resolve())
    produces_seed = any(arg.split("=", 1)[0] == "--common-initial-solution-output" for arg in value.extra_arguments)
    if (data["reduced_internal_crank_velocity_guard"] == "auto" and value.mechanics == "reduced"
            and value.formulation == "dynamic" and (value.solver == "acados" or produces_seed)):
        data["reduced_internal_crank_velocity_guard"] = "on"
    if value.mode == "fho":
        data["cycles_per_window"] = value.cycles
    sections = {group: {key: data[key] for key in names.split()} for group, names in _GROUPS.items()}
    if value.compile_hessian_only:
        sections["solver"].update(compiled_callbacks=["nlp_hess_l"], compiler_flags=["-O1"],
                                  cache_policy="output_root/native-cache/config_hash_prefix")
    sections["transcription"].update(symbolic_type="SX", muscle_graph="configured_model" if value.model_config else "periodic_node")
    canonical = json.dumps(sections, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return ResolvedSimulationConfig(value, canonical, profile)
