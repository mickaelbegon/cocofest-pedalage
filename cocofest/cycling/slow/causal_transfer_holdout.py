"""Immutable, out-of-distribution contracts for causal torque transfer.

The calibrated left-to-right policy is deliberately *not* re-fit here.  This
module only describes source trajectories needed to challenge it: a higher
total task load and a left/right-reversed Ding parameter pair.  A baseline
RHO must first create certified checkpoints; only then can a separate branch
manifest be sealed from their byte-pinned receipts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any


def _finite_positive(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


@dataclass(frozen=True)
class CausalTransferHoldout:
    """One source-RHO condition and its frozen action-policy constants."""

    identifier: str
    total_torque_nm: float
    left_model: str
    right_model: str
    checkpoint_cycles: tuple[int, ...]
    purpose: str
    left_activate_at_or_below: float = .62
    right_recipient_at_or_above: float = .78
    left_relief_nm: float = .10
    action_cycles: int = 10
    maximum_cycles: int = 240

    def validate(self) -> "CausalTransferHoldout":
        if not isinstance(self.identifier, str) or not self.identifier or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in self.identifier):
            raise ValueError("identifier must be lower-case kebab case")
        _finite_positive(self.total_torque_nm, "total_torque_nm")
        for label, value in (("left_model", self.left_model), ("right_model", self.right_model), ("purpose", self.purpose)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} is required")
        if not self.checkpoint_cycles or any(type(cycle) is not int or cycle <= 0 for cycle in self.checkpoint_cycles):
            raise ValueError("checkpoint_cycles must contain positive integer cycles")
        if tuple(sorted(set(self.checkpoint_cycles))) != self.checkpoint_cycles:
            raise ValueError("checkpoint_cycles must be unique and increasing")
        if self.checkpoint_cycles[-1] >= self.maximum_cycles:
            raise ValueError("checkpoint cycle must precede maximum_cycles")
        if type(self.action_cycles) is not int or self.action_cycles != 10:
            raise ValueError("the preregistered holdout action is exactly ten cycles")
        if not math.isclose(_finite_positive(self.left_relief_nm, "left_relief_nm"), .10, abs_tol=1e-12):
            raise ValueError("the preregistered holdout left relief is exactly 0.10 Nm")
        if not math.isclose(float(self.left_activate_at_or_below), .62, abs_tol=1e-12):
            raise ValueError("the preregistered left threshold is exactly 0.62")
        if not math.isclose(float(self.right_recipient_at_or_above), .78, abs_tol=1e-12):
            raise ValueError("the preregistered right threshold is exactly 0.78")
        if self.total_torque_nm / 2 <= self.left_relief_nm:
            raise ValueError("equal split must remain positive after the action")
        return self

    @property
    def baseline_per_arm_nm(self) -> float:
        self.validate()
        return self.total_torque_nm / 2

    def canonical_dict(self) -> dict[str, Any]:
        self.validate()
        return {**asdict(self), "checkpoint_cycles": list(self.checkpoint_cycles),
                "baseline_per_arm_nm": self.baseline_per_arm_nm,
                "action_left_nm": self.baseline_per_arm_nm - self.left_relief_nm,
                "action_right_nm": self.baseline_per_arm_nm + self.left_relief_nm}


def standard_holdouts(root: Path) -> dict[str, CausalTransferHoldout]:
    """Preregistered sources; none changes the learned policy values."""
    root = Path(root).resolve()
    original_left = root / "asymmetric-sides-r192-rho-bo-20260928/model-left.json"
    original_right = root / "asymmetric-sides-r192-rho-bo-20260928/model-right.json"
    return {
        # +8.3% total work versus calibration.  20-cycle checkpoints prevent
        # choosing the trigger only after seeing an endpoint.
        "higher-load-208": CausalTransferHoldout(
            "higher-load-208", 2.08, str(original_left), str(original_right),
            (40, 60, 80, 100, 120, 140, 160),
            "Load holdout: same Ding models, 2.08 Nm total instead of calibrated 1.92 Nm."),
        # New condition, deliberately distinct from the already observed
        # 2.08-Nm holdout.  The tighter grid is registered before execution,
        # so it cannot select a missed crossing retrospectively.
        "higher-load-200": CausalTransferHoldout(
            "higher-load-200", 2.00, str(original_left), str(original_right),
            (40, 60, 80, 90, 100, 110, 120, 130, 140, 150),
            "Load holdout: same Ding models, 2.00 Nm total; dense grid across the inherited transfer window."),
        # Swapping the two independently configured sides is a genuine OOD
        # direction: the policy donor is now the favourable arm.  A sensible
        # fixed left-relief policy should therefore abstain, not fabricate a
        # benefit by re-tuning its threshold or sign.
        "reversed-ding-sides-192": CausalTransferHoldout(
            "reversed-ding-sides-192", 1.92, str(original_right), str(original_left),
            (60, 80, 100, 120, 140, 160),
            "Ding holdout: favourable and unfavourable left/right parameter sets are reversed at fixed 1.92 Nm total."),
    }


def _reference_runtime_contract(condition: CausalTransferHoldout) -> dict[str, str]:
    """Pin the known-working RHO32 runtime and its MA57 HSL library.

    The holdout source is frequently launched from a clean GUI/base-conda
    process.  Relying on that process's ``CONDA_PREFIX`` silently removes MA57
    and produces IPOPT ``Invalid_Option`` before even one NLP callback.  The
    matched 1.92-Nm unit run is the numerical reference, so inherit its
    declared runtime prefix and resolve the exact shared library now.
    """
    reference = (Path(condition.left_model).resolve().parent / "factorial-weights-split-20261003"
                 / "unit_fixed" / "input.json")
    try:
        payload = json.loads(reference.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read the matched unit-reference runtime: {reference}") from error
    prefix_value = payload.get("runtime_prefix")
    if not isinstance(prefix_value, str) or not prefix_value:
        raise ValueError("Matched unit-reference does not declare runtime_prefix")
    prefix = Path(prefix_value).expanduser().resolve()
    if not prefix.is_dir():
        raise ValueError(f"Matched runtime prefix is unavailable: {prefix}")
    explicit = payload.get("ipopt_hsl_library")
    candidates = ([Path(explicit).expanduser()] if isinstance(explicit, str) and explicit
                  else sorted(prefix.glob("opt/libhsl/*/lib/libhsl.so")))
    usable = [candidate.resolve() for candidate in candidates if candidate.is_file()]
    if len(usable) != 1:
        raise ValueError(f"Could not resolve one MA57 HSL library below matched runtime {prefix}")
    hsl = usable[0]
    return {"reference_input": str(reference), "reference_input_sha256": sha256(reference.read_bytes()).hexdigest(),
            "runtime_prefix": str(prefix), "ipopt_hsl_library": str(hsl),
            "ipopt_hsl_library_sha256": sha256(hsl.read_bytes()).hexdigest()}


def validate_baseline_runtime(payload: dict[str, Any]) -> dict[str, str]:
    """Reject an unpinned/unloadable MA57 setup before a holdout is launched."""
    prefix = payload.get("runtime_prefix")
    library = payload.get("ipopt_hsl_library")
    if not isinstance(prefix, str) or not Path(prefix).is_dir():
        raise ValueError("Holdout baseline lacks a usable pinned runtime_prefix")
    if not isinstance(library, str) or not Path(library).is_file():
        raise ValueError("Holdout baseline lacks a usable pinned IPOPT MA57 HSL library")
    expected = payload.get("ipopt_hsl_library_sha256")
    observed = sha256(Path(library).read_bytes()).hexdigest()
    if not isinstance(expected, str) or expected != observed:
        raise ValueError("Pinned IPOPT MA57 HSL library changed or lacks its SHA-256 receipt")
    # Keep this independent of the current shell's CONDA_PREFIX.  This is the
    # exact precedence used by the process coordinator when it constructs
    # IPOPT's `hsllib` option.
    return {"runtime_prefix": str(Path(prefix).resolve()), "ipopt_hsl_library": str(Path(library).resolve()),
            "ipopt_hsl_library_sha256": observed}


def baseline_payload(condition: CausalTransferHoldout) -> dict[str, Any]:
    """The unit-weight source RHO requested before any holdout branching."""
    condition.validate()
    torque = condition.baseline_per_arm_nm
    weights = {name: 1.0 for name in ("Delt_ant", "Delt_post", "Biceps", "Triceps")}
    runtime = _reference_runtime_contract(condition)
    payload = {
        "cycles": condition.maximum_cycles,
        "cycles_per_window": 1,
        "factory": "cocofest.simulation.independent_arms_process:build_process_independent_arms",
        "formulation": "isokinetic",
        "left_driver_arguments": ["--retry-failed-rho-without-advance", "--max-consecutive-failing", "2",
                                  "--ipopt-frozen-rho-feasibility-probe"],
        "right_driver_arguments": ["--retry-failed-rho-without-advance", "--max-consecutive-failing", "2",
                                   "--ipopt-frozen-rho-feasibility-probe"],
        "left_equivalent_mean_torque_nm": torque,
        "right_equivalent_mean_torque_nm": torque,
        "left_model_config": condition.left_model,
        "right_model_config": condition.right_model,
        "omega_rad_s": -2 * math.pi,
        "parallel": True,
        "parametric_fatigue_weights": True,
        "prepared_restart_checkpoint_cycles": list(condition.checkpoint_cycles),
        "resistance_pace": {"total_equivalent_mean_torque_nm": condition.total_torque_nm,
                            "initial_right_fraction": .5, "initial_split_policy": "manual",
                            "minimum_arm_equivalent_mean_torque_nm": 0., "capacity_feedback": False,
                            "update_every_cycles": 20, "capacity_gain": 1., "smoothing": .25,
                            "max_fraction_step": .1},
        "muscle_pace": {"adaptation_enabled": False, "initial_weight_basis": "unit holdout baseline; no FHO data",
                         "left_initial_weights": weights, "right_initial_weights": weights,
                         "max_relative_weight": 4., "min_relative_weight": .25, "update_every_cycles": 20},
        "solver": "ipopt", "stimulations_per_cycle": 30,
        "worker_timeout_seconds": 1800,
        "holdout_contract": condition.canonical_dict(),
    }
    payload.update(runtime)
    validate_baseline_runtime(payload)
    return payload
