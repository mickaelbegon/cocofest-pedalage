"""Content-addressed P1/P2/P7 preparation for a simulated cycling session.

This deliberately prepares only quantities independent of the future RHO pulse
width decisions.  It does not create a feasibility map, a slow plan, or seeds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
from time import perf_counter

import numpy as np


_DING_2007_TAUC_S = 0.011
_DING_2007_KM_REST = 0.137
_DING_R0_KM_RELATIONSHIP = 1.04
_SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class PresessionConfig:
    model_left: Path
    model_right: Path
    horizon_cycles: int
    mode: str = "prescribed_load"
    cycle_duration_s: float = 1.0
    stimulation_hz: int = 30
    seed: int = 0
    reduced_profile: Path | None = None
    theta0_rad: float = 0.0
    omega_rad_s: float = -2 * math.pi

    def __post_init__(self) -> None:
        if isinstance(self.horizon_cycles, bool) or not isinstance(self.horizon_cycles, int) or self.horizon_cycles <= 0:
            raise ValueError("horizon_cycles must be a positive integer")
        if self.mode not in {"prescribed_load", "max_work"}:
            raise ValueError("mode must be prescribed_load or max_work")
        if isinstance(self.stimulation_hz, bool) or not isinstance(self.stimulation_hz, int) or self.stimulation_hz <= 0:
            raise ValueError("stimulation_hz must be a positive integer")
        if not math.isfinite(self.cycle_duration_s) or self.cycle_duration_s <= 0:
            raise ValueError("cycle_duration_s must be finite and positive")
        pulses_per_cycle = self.stimulation_hz * self.cycle_duration_s
        if not math.isclose(pulses_per_cycle, round(pulses_per_cycle), abs_tol=1e-10):
            raise ValueError("cycle duration must contain an integer number of stimulation intervals")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not math.isfinite(self.theta0_rad) or not math.isfinite(self.omega_rad_s) or self.omega_rad_s == 0:
            raise ValueError("theta0_rad and nonzero omega_rad_s must be finite")
        for name in ("model_left", "model_right"):
            path = Path(getattr(self, name))
            if not path.is_file():
                raise ValueError(f"{name} is not a file: {path}")
        if self.reduced_profile is not None and not Path(self.reduced_profile).is_file():
            raise ValueError(f"reduced_profile is not a file: {self.reduced_profile}")


def _models(config: PresessionConfig) -> dict[str, dict]:
    result = {}
    for side, path in (("left", config.model_left), ("right", config.model_right)):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("muscles"), dict) or not payload["muscles"]:
            raise ValueError(f"{side} model must contain a nonempty muscles object")
        for name, values in payload["muscles"].items():
            if not isinstance(values, dict):
                raise ValueError(f"{side}/{name} parameters must be an object")
            for key in ("tauc", "km_rest"):
                if key in values and (not isinstance(values[key], (int, float)) or not math.isfinite(values[key]) or values[key] <= 0):
                    raise ValueError(f"{side}/{name}/{key} must be finite and positive")
        result[side] = payload
    return result


def ding_cn_on_stimulation_grid(*, pulse_count: int, stimulation_hz: int, tauc_s: float, km_rest: float) -> np.ndarray:
    """Exact zero-initial-state flow of Ding's Cn equation at each pulse time.

    At t_i, the new pulse changes the exponential forcing, not Cn itself.
    The Ri formula is inherited from DingModelFrequency.cn_sum_fun.  Neither
    pulse width nor the evolving slow Km state appears in calculate_cn_dot.
    """
    if pulse_count < 0 or stimulation_hz <= 0 or tauc_s <= 0 or km_rest <= 0:
        raise ValueError("invalid pulse count, frequency or Ding calcium parameters")
    delta = 1.0 / stimulation_hz
    decay = math.exp(-delta / tauc_s)
    r0 = km_rest + _DING_R0_KM_RELATIONSHIP
    later_pulse_amplitude = 1.0 + (r0 - 1.0) * decay
    cn = np.empty(pulse_count + 1, dtype=float)
    cn[0] = 0.0
    forcing = 1.0  # Pulse at t=0.
    for index in range(1, pulse_count + 1):
        cn[index] = decay * (cn[index - 1] + forcing * delta / tauc_s)
        if index < pulse_count:
            forcing = forcing * decay + later_pulse_amplitude
    return cn


def _prepare_cn(directory: Path, config: PresessionConfig, models: dict[str, dict]) -> dict:
    pulse_count = round(config.horizon_cycles * config.cycle_duration_s * config.stimulation_hz)
    times = np.arange(pulse_count + 1, dtype=float) / config.stimulation_hz
    keys = []
    arrays = {"time_s": times}
    for side, model in models.items():
        for muscle, values in model["muscles"].items():
            key = f"{side}__{muscle}"
            if key in arrays:
                raise ValueError(f"duplicate calcium key: {key}")
            arrays[key] = ding_cn_on_stimulation_grid(
                pulse_count=pulse_count,
                stimulation_hz=config.stimulation_hz,
                tauc_s=float(values.get("tauc", _DING_2007_TAUC_S)),
                km_rest=float(values.get("km_rest", _DING_2007_KM_REST)),
            )
            keys.append(key)
    np.savez(directory / "cn_timeseries.npz", **arrays)
    return {"status": "computed", "samples": len(times), "muscles": keys,
            "model": "Ding 2007 calcium ODE, zero initial Cn, all pulses retained",
            "pulse_width_dependency": False,
            "parameter_defaults": {"tauc_s": _DING_2007_TAUC_S, "km_rest": _DING_2007_KM_REST,
                                   "r0_km_relationship": _DING_R0_KM_RELATIONSHIP}}


def _prepare_passive(directory: Path, config: PresessionConfig) -> dict:
    """Evaluate reduced mechanical g(theta)+c(theta)omega², if supplied.

    This is the mechanical demand with zero active force, before division by
    external torque effectiveness. It excludes passive muscle force. An
    unstimulated measured turn can replace this interface in clinical work.
    """
    if config.reduced_profile is None:
        return {"status": "not_measured", "reason": "no reduced profile or unstimulated turn supplied",
                "interface": "time_s, theta_rad, mechanical_passive_nm; source metadata required"}
    from cocofest.dynamics.reduced_cycling import ReducedCyclingDynamics

    dynamics = ReducedCyclingDynamics.load(config.reduced_profile)
    pulse_count = round(config.horizon_cycles * config.cycle_duration_s * config.stimulation_hz)
    times = np.arange(pulse_count + 1, dtype=float) / config.stimulation_hz
    theta = config.theta0_rad + config.omega_rad_s * times
    values = [dynamics.coefficient_values(float(angle)) for angle in theta]
    torque = np.array([v["projected_gravity"] + v["projected_velocity_quadratic"] * config.omega_rad_s**2
                       for v in values], dtype=float)
    if not np.isfinite(torque).all():
        raise ValueError("reduced profile produced nonfinite passive mechanical torque")
    np.savez(directory / "passive_torque.npz", time_s=times, theta_rad=theta, mechanical_passive_nm=torque)
    return {"status": "computed_from_reduced_profile", "samples": len(times),
            "definition": "projected_gravity(theta) + projected_velocity_quadratic(theta)*omega_rad_s**2",
            "muscle_passive_force_included": False, "file": "passive_torque.npz"}


def _versions() -> dict[str, str | None]:
    names = ("numpy", "casadi", "bioptim", "cyipopt", "ma57")
    result: dict[str, str | None] = {"python": platform.python_version()}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def prepare_presession(config: PresessionConfig, output_root: Path) -> Path:
    """Create an immutable, portable P1/P2/P7 directory and return its path."""
    started = perf_counter()
    models = _models(config)
    source_paths = {"model_left.json": Path(config.model_left), "model_right.json": Path(config.model_right)}
    if config.reduced_profile is not None:
        source_paths["reduced_profile.npz"] = Path(config.reduced_profile)
    input_hashes = {name: _sha256(path) for name, path in source_paths.items()}
    portable_config = asdict(config)
    portable_config["model_left"] = "model_left.json"
    portable_config["model_right"] = "model_right.json"
    portable_config["reduced_profile"] = "reduced_profile.npz" if config.reduced_profile else None
    identity = {"schema_version": _SCHEMA_VERSION, "config": portable_config, "input_sha256": input_hashes}
    run_hash = sha256(_canonical_json(identity)).hexdigest()[:16]
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    directory = root / f"presession-{run_hash}"
    if directory.exists():
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("identity") != identity:
            raise ValueError(f"existing presession directory has different inputs: {directory}")
        for name, digest in manifest["output_sha256"].items():
            if _sha256(directory / name) != digest:
                raise ValueError(f"existing presession artifact changed: {name}")
        return directory
    directory.mkdir()
    try:
        for name, path in source_paths.items():
            shutil.copyfile(path, directory / name)
        _write_json(directory / "config.json", portable_config)
        p1_started = perf_counter()
        cn = _prepare_cn(directory, config, models)
        p1_seconds = perf_counter() - p1_started
        p2_started = perf_counter()
        passive = _prepare_passive(directory, config)
        p2_seconds = perf_counter() - p2_started
        _write_json(directory / "passive_torque.json", passive)
        p7_started = perf_counter()
        input_files = [*source_paths, "config.json"]
        output_files = ["cn_timeseries.npz", "passive_torque.json"]
        if passive["status"] == "computed_from_reduced_profile":
            output_files.append("passive_torque.npz")
        manifest = {"identity": identity, "input_sha256": {name: _sha256(directory / name) for name in input_files},
                    "output_sha256": {name: _sha256(directory / name) for name in output_files},
                    "versions": _versions(), "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                    "seed": config.seed, "p1_cn": cn, "p2_passive_torque": passive,
                    "prepared_phases": ["P1", "P2", "P7"],
                    "pending_phases": ["P3", "P4", "P5", "P6"]}
        p7_seconds = perf_counter() - p7_started
        budget = {"seconds": {"P1_cn": p1_seconds, "P2_passive_torque": p2_seconds,
                              "P7_manifest": p7_seconds, "total": perf_counter() - started},
                  "target_total_seconds": 300.0, "target_met_for_prepared_phases_only": perf_counter() - started <= 300.0,
                  "pending_phases": ["P3", "P4", "P5", "P6"]}
        _write_json(directory / "manifest.json", manifest)
        _write_json(directory / "budget.json", budget)
        return directory
    except Exception:
        shutil.rmtree(directory)
        raise
