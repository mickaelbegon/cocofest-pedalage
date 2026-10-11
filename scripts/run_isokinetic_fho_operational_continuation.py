#!/usr/bin/env python3
"""Continue the prescribed-isokinetic FHO bridge under an explicit audit policy.

This is deliberately *not* a relaxation of the scientific certificate.  It is
an operational continuation policy requested for the 1.3 Nm campaign: a solve
whose only failed scientific gate is the DOP853 energy mismatch may seed the
next FHO when that mismatch is at most 0.1 percent.  RHO is a warm-start
generator only: when an end-of-horizon RHO fails, its finite output (or the
finite terminal FHO projection) can still seed the next FHO restoration
solve.  Every such decision is written to
``operational-continuation-audit.json``.

The resulting FHO is consequently labelled ``operational`` until it is
re-audited with the strict (or Radau-7 escalation) criterion.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python")
HSL = Path(
    "/home/mickaelbegon/miniforge3/envs/cocofest-rho32/opt/libhsl/"
    "v2025.7.21/lib/libhsl.so"
)
CYCLING = ROOT / "examples/fes_multibody/cycling/cycling_fes_solver_comparison.py"


def load_runner() -> Any:
    source = ROOT / ".github/scripts/run_full_horizon_benchmark.py"
    spec = importlib.util.spec_from_file_location("fho_runner", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {source}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses in the imported runner need the module registered first.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def result_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))["results"][0]


def audit_summary(result_path: Path, relative_limit: float) -> dict[str, Any]:
    """Return an auditable operational decision, never a strict certificate."""
    try:
        raw = json.loads(result_path.read_text(encoding="utf-8"))
        result = raw["results"][0]
        audit = (result.get("isokinetic_audits") or [{}])[0]
        config = raw["configurations"]["ipopt"]
        target = float(config["energy_target_j"])
        high_error = float(audit.get("high_accuracy_energy_target_error_j", float("inf")))
        collocation_error = float(audit.get("certification_energy_target_error_j", float("inf")))
        per_cycle_error = float(
            audit.get("high_accuracy_maximum_per_cycle_energy_error_j", float("inf"))
        )
        per_cycle_target = float(audit.get("per_cycle_energy_target_j", 0.0))
        per_cycle_enabled = bool(
            config.get("enforce_isokinetic_work_per_cycle", False)
        )
        relative_error = high_error / target if target else float("inf")
        mechanics = result.get("mechanical_equivalence_audit") or {}
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        # A partial JSON must become an explicit rejection, rather than an
        # exception which obscures whether the fallback stopped before FHO.
        return {
            "strict_scientific_success": False,
            "operational_continuation_accepted": False,
            "operational_relative_high_accuracy_energy_limit": relative_limit,
            "high_accuracy_energy_target_error_j": float("inf"),
            "energy_target_j": None,
            "relative_high_accuracy_energy_error": float("inf"),
            "checks": {"parseable_complete_solver_result": False},
            "reason": f"Operational continuation rejected: unreadable/incomplete solver result ({error}).",
        }

    # These checks retain every feasibility/mechanics condition that can be
    # checked independently of the strict high-accuracy energy gate.
    base_checks = {
        "ipopt_zero_status": bool(result.get("solver_success")),
        "finite_isokinetic_audit": bool(audit.get("finite")),
        "collocation_energy_matches_target": collocation_error <= 1e-7,
        "no_load_bound_violation": float(audit.get("maximum_load_torque_bound_violation_nm", float("inf"))) <= 1e-10,
        "zero_speed_error": float(audit.get("maximum_speed_error_rad_s", float("inf"))) <= 1e-10,
        "mechanical_equivalence": bool(mechanics.get("passes_tolerance")),
        "high_accuracy_error_within_operational_limit": relative_error <= relative_limit,
        "per_cycle_work_constraint_enabled": per_cycle_enabled,
        "per_cycle_high_accuracy_error_within_operational_limit": (
            per_cycle_error <= relative_limit * per_cycle_target
        ),
    }
    strict_success = bool(result.get("success"))
    accepted = all(base_checks.values())
    return {
        "strict_scientific_success": strict_success,
        "operational_continuation_accepted": accepted,
        "operational_relative_high_accuracy_energy_limit": relative_limit,
        "high_accuracy_energy_target_error_j": high_error,
        "energy_target_j": target,
        "relative_high_accuracy_energy_error": relative_error,
        "high_accuracy_maximum_per_cycle_energy_error_j": per_cycle_error,
        "per_cycle_energy_target_j": per_cycle_target,
        "checks": base_checks,
        "reason": (
            "User-approved operational continuation; this is not a strict scientific certificate."
            if accepted and not strict_success
            else "Strict certificate available." if strict_success else "Operational continuation rejected."
        ),
    }


def objective_arguments(objective: str, weights: tuple[float, float, float, float] | None) -> list[str]:
    """Keep the RHO and FHO objective definition exactly identical."""
    values = ["--objective", objective, "--objective-shape", "quadratic"]
    if weights is not None:
        if objective == "fatigue":
            values.extend(["--parametric-fatigue-weights", "--fatigue-weight-values"])
        else:
            values.extend(["--parametric-control-weights", "--control-weight-values"])
        values.extend(str(weight) for weight in weights)
    return values


def command_for_rho(
    seed: Path, solution: Path, result: Path, *, objective: str,
    weights: tuple[float, float, float, float] | None, collocation_degree: int,
) -> list[str]:
    return [
        "taskset", "-c", "0-11", str(PYTHON), str(CYCLING),
        "--solvers", "ipopt", *objective_arguments(objective, weights),
        "--formulation", "isokinetic", "--mechanical-formulation", "reduced",
        "--isokinetic-kinematics", "prescribed", "--isokinetic-omega", "-6.283185307179586",
        "--enforce-isokinetic-work-per-cycle",
        "--energy-equivalent-torque", "1.3", "--load-torque-min", "-4", "--load-torque-max", "4",
        "--stimulations-per-cycle", "30", "--ipopt-profile", "periodic_collocation",
        "--ipopt-collocation-method", "radau", "--ipopt-collocation-degree", str(collocation_degree),
        "--ipopt-use-sx", "--ipopt-linear-solver", "ma57", "--warmup-ipopt-linear-solver", "ma57",
        "--ipopt-hsl-library", str(HSL), "--ipopt-max-iter", "4000", "--nlp-tolerance", "1e-8",
        "--primal-feasibility-threshold", "1e-5", "--ipopt-enforce-start-constraints",
        "--ipopt-disable-standard-warmup", "--adopt-common-initial-solution-warmup-cycles",
        "--common-initial-solution-recenter-first-node-bounds",
        "--allow-primal-feasible-common-initial-solution-output",
        "--common-initial-solution", str(seed), "--common-initial-solution-output", str(solution),
        "--cycles-per-window", "1", "--n-windows", "1", "--n-threads", "12",
        "--compact-rho-output", "--output-json", str(result),
    ]


def command_for_fho(
    seed: Path, prefix: Path, solution: Path, result: Path, cycles: int, *, objective: str,
    weights: tuple[float, float, float, float] | None, collocation_degree: int, ipopt_max_iter: int,
    ding_local_sx: bool = False,
) -> list[str]:
    # The experimental adapter condenses the locally affine Ding states in an
    # SX graph and reconstructs/audits the full original solution.  It is not
    # a finite-difference approximation and it deliberately leaves the
    # isokinetic work constraint in the original audit space.
    entry = (
        [str(ROOT / "scripts/benchmark_ding_radau5_local_ab.py"), "local", "--"]
        if ding_local_sx
        else [str(CYCLING)]
    )
    representation = "--ipopt-use-sx" if ding_local_sx else "--ipopt-no-use-sx"
    return [
        "taskset", "-c", "0-11", str(PYTHON), *entry,
        "--solvers", "ipopt", *objective_arguments(objective, weights),
        "--formulation", "isokinetic", "--mechanical-formulation", "reduced",
        "--isokinetic-kinematics", "prescribed", "--isokinetic-omega", "-6.283185307179586",
        "--enforce-isokinetic-work-per-cycle",
        "--energy-equivalent-torque", "1.3", "--load-torque-min", "-4", "--load-torque-max", "4",
        "--stimulations-per-cycle", "30", "--ipopt-profile", "periodic_collocation",
        "--ipopt-collocation-method", "radau", "--ipopt-collocation-degree", str(collocation_degree),
        representation, "--ipopt-linear-solver", "ma57", "--warmup-ipopt-linear-solver", "ma57",
        "--ipopt-hsl-library", str(HSL), "--ipopt-max-iter", str(ipopt_max_iter), "--nlp-tolerance", "1e-8",
        "--primal-feasibility-threshold", "1e-5", "--ipopt-enforce-start-constraints",
        "--ipopt-disable-standard-warmup", "--adopt-common-initial-solution-warmup-cycles",
        "--optional-nlp-periodic-ipopt-hot-start", "--initial-guess-diagnostics",
        "--allow-primal-feasible-common-initial-solution-output",
        "--single-shot", "--cycles-per-window", str(cycles), "--n-windows", str(cycles), "--n-threads", "12",
        "--common-initial-solution", str(seed), "--full-horizon-prefix-solution", str(prefix),
        "--common-initial-solution-output", str(solution), "--output-json", str(result),
    ]


def _file_identity(path: Path) -> dict[str, Any]:
    """Return a cheap, reproducible identity without reading a large archive.

    The anticipated FHO cannot consume its predecessor's terminal state until
    that predecessor has passed its audit.  This identity is consequently
    recorded only for files that already exist; a missing expected FHO result
    is normal while it is still running.
    """
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "sha256": digest.hexdigest(),
    }


def _fho_protocol(
    *, objective: str, weights: tuple[float, float, float, float] | None,
    collocation_degree: int, ipopt_max_iter: int, ding_local_sx: bool,
) -> dict[str, Any]:
    """Fields which must match between anticipation and the later solve."""
    return {
        "formulation": "isokinetic",
        "mechanical_formulation": "reduced",
        "isokinetic_kinematics": "prescribed",
        "per_cycle_work_constraint": True,
        "graph": "SX with exact Ding-local condensation" if ding_local_sx else "MX",
        "ding_local_sx": ding_local_sx,
        "solver": "IPOPT",
        "linear_solver": "MA57",
        "cpu_affinity": "0-11",
        "bioptim_threads": 12,
        "numeric_threads": 1,
        "objective": objective,
        "objective_weights": weights,
        "radau_degree": collocation_degree,
        "ipopt_max_iter": ipopt_max_iter,
    }


def write_next_fho_preparation(
    *, output_dir: Path, output: Path, start_cycles: int, extension_cycles: int, objective: str,
    weights: tuple[float, float, float, float] | None, collocation_degree: int,
    ipopt_max_iter: int, ding_local_sx: bool,
) -> Path:
    """Write an auditable, non-solving next-FHO preparation manifest.

    CasADi graphs (MX and SX) are process-local in the current Bioptim entry
    point: a graph created by a helper process cannot be handed to the future
    IPOPT process.  The useful and safe overlap is therefore preparation of
    the immutable launch contract and filesystem layout.  The final terminal
    seed and command are bound only after FHO_i is certified.  Keeping this
    distinction in the manifest prevents a speculative state from ever being
    mistaken for a certified handoff.
    """
    if start_cycles < 1 or extension_cycles < 1:
        raise ValueError("start_cycles and extension_cycles must be positive.")
    end_cycles = start_cycles + extension_cycles
    output_dir = output_dir.resolve()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix_solution = output_dir / f"fho-{start_cycles:04d}-solution.npz"
    prefix_result = output_dir / f"fho-{start_cycles:04d}-result.json"
    # The manifest is intentionally path-only for future seed/result files.
    # A consumer substitutes only files produced after the source audit.
    planned_dir = output.parent / f"fho-{end_cycles:04d}-planned"
    planned_dir.mkdir(parents=True, exist_ok=True)
    placeholder_seed = planned_dir / "CERTIFIED_COMBINED_SEED_REQUIRED.npz"
    placeholder_solution = planned_dir / "fho-solution-after-solve.npz"
    placeholder_result = planned_dir / "fho-result-after-solve.json"
    command = command_for_fho(
        placeholder_seed,
        prefix_solution,
        placeholder_solution,
        placeholder_result,
        end_cycles,
        objective=objective,
        weights=weights,
        collocation_degree=collocation_degree,
        ipopt_max_iter=ipopt_max_iter,
        ding_local_sx=ding_local_sx,
    )
    payload = {
        "schema": "fho-next-horizon-preparation-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "prepared_waiting_for_certified_terminal",
        "start_cycles": start_cycles,
        "extension_cycles": extension_cycles,
        "end_cycles": end_cycles,
        "protocol": _fho_protocol(
            objective=objective,
            weights=weights,
            collocation_degree=collocation_degree,
            ipopt_max_iter=ipopt_max_iter,
            ding_local_sx=ding_local_sx,
        ),
        "expected_prefix": {
            "solution": _file_identity(prefix_solution),
            "result": _file_identity(prefix_result),
            "must_pass_operational_audit_before_consumption": True,
        },
        "planned_work_dir": str(planned_dir),
        "command_template": [
            "{CERTIFIED_COMBINED_SEED}" if item == str(placeholder_seed) else item
            for item in command
        ],
        "graph_reuse": {
            "available": False,
            "reason": (
                "The current Bioptim launcher creates and owns the CasADi graph "
                "inside the solve process; it cannot safely deserialize a graph from "
                "this preparatory process."
            ),
            "overlapped_work": [
                "launch contract validation",
                "protocol compatibility check",
                "attempt-directory allocation",
            ],
        },
    }
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return output


def load_prepared_next_fho(
    manifest_path: Path, *, start_cycles: int, end_cycles: int, objective: str,
    weights: tuple[float, float, float, float] | None, collocation_degree: int,
    ipopt_max_iter: int, ding_local_sx: bool,
) -> dict[str, Any]:
    """Validate a preflight manifest before associating it with a real solve."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "fho-next-horizon-preparation-v1":
        raise ValueError(f"Unsupported FHO preparation manifest: {manifest_path}")
    expected_protocol = _fho_protocol(
        objective=objective,
        weights=weights,
        collocation_degree=collocation_degree,
        ipopt_max_iter=ipopt_max_iter,
        ding_local_sx=ding_local_sx,
    )
    if (
        manifest.get("start_cycles") != start_cycles
        or manifest.get("end_cycles") != end_cycles
        or manifest.get("protocol") != expected_protocol
    ):
        raise ValueError(
            "Prepared FHO manifest does not match this continuation's horizon or protocol."
        )
    return manifest


def run(command: list[str], log_path: Path, *, allow_failure: bool = False) -> bool:
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode and not allow_failure:
        raise RuntimeError(f"Solver exited with {completed.returncode}; see {log_path}")
    return completed.returncode == 0


def rho_window_valid(result_path: Path) -> bool:
    """Accept a one-cycle stage only when IPOPT and its primal window agree."""
    result = result_payload(result_path)
    windows = result.get("windows") or []
    return bool(
        result.get("solver_success")
        and len(windows) == 1
        and windows[0].get("validated")
        and windows[0].get("primal_feasible")
    )


def finite_solution_archive(solution_path: Path) -> bool:
    """Return whether an archive is finite enough to be an FHO warm start.

    A solver process can leave a partial archive behind.  Its existence alone
    is not evidence that it is usable: NaNs/Infs would poison the subsequent
    FHO graph and make the failure hard to diagnose.  This deliberately does
    not inspect convergence, feasibility, or the scientific audit; those are
    precisely what make this a non-certified candidate rather than a RHO
    validation.
    """
    if not solution_path.exists():
        return False
    try:
        with np.load(solution_path, allow_pickle=False) as data:
            numeric = [
                np.asarray(data[key])
                for key in data.files
                if key.startswith(("states__", "controls__", "parameters__"))
            ]
        return bool(numeric) and all(np.all(np.isfinite(value)) for value in numeric)
    except (OSError, ValueError, KeyError):
        return False


def finite_rho_candidate(solution_path: Path, result_path: Path) -> bool:
    """An exported failed RHO is a candidate only when its result also exists."""
    return result_path.exists() and finite_solution_archive(solution_path)


def prescribed_compatible_reference(source: Path, output: Path) -> Path:
    """Remove redundant prescribed kinematic states from an RHO reference.

    A prescribed FHO has no theta/omega decision states; a standalone RHO
    archive can retain them for telemetry.  They are not part of the common
    continuation state space, so retaining them prevents a purely numerical
    initial-state homotopy.  Their prescribed values are audited separately.
    """
    with np.load(source, allow_pickle=False) as data:
        payload = {
            key: np.asarray(data[key])
            for key in data.files
            if key not in {
                "states__theta", "states__omega", "physical__initial_state", "physical__shooting_time_s"
            }
            and not key.startswith("applied_pulse_widths__")
        }
    metadata = json.loads(str(payload["metadata__json"]))
    # The physical replay contract specifically includes theta/omega.  This
    # projected seed intentionally has no such decision states, so it must be
    # a legacy warm-start archive rather than a falsely declared physical one.
    for key in (
        "physical_archive_schema", "physical_state_names", "physical_state_stride",
        "physical_time_origin", "physical_control_interpolation", "physical_cycles",
    ):
        metadata.pop(key, None)
    payload["metadata__json"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **payload)
    return output


def reset_local_cycle_integral(seed: Path, reference_trace: Path, cycle: int, runner: Any) -> None:
    """Reset E_prod in a local RHO seed without touching physical terminal states.

    ``E_prod`` is an accounting integral: FHO accumulates it across its whole
    single shot, whereas RHO resets it at every cycle boundary.  Carrying the
    FHO cumulative value into a local RHO makes its terminal energy equality
    impossible.  The reference supplies only that reset integral profile;
    fatigue, force, calcium, and parameter states remain the FHO terminal
    state exactly.
    """
    reference = seed.with_name(seed.stem + "-energy-reference.npz")
    # The reference cycle is used only to initialise the local accounting
    # integral E_prod.  Once continuation goes past the archived RHO horizon,
    # reusing its final *integral profile* is valid: no fatigue, force,
    # calcium, or parameter state is copied from that reference.  The current
    # FHO terminal remains the sole physical state carrier and IPOPT re-solves
    # the local energy trajectory.  This avoids an artificial hard stop at the
    # length of the initial RHO archive.
    with np.load(reference_trace, allow_pickle=False) as trace:
        metadata = json.loads(str(trace["metadata__json"]))
    source_cycles = int(metadata.get("physical_cycles") or metadata.get("producer_requested_cycles") or 0)
    if source_cycles < 1:
        raise ValueError(f"Could not determine the RHO reference horizon in {reference_trace}.")
    template_cycle = min(cycle, source_cycles)
    runner.write_rho_seed_cycle(reference_trace, reference, template_cycle)
    with np.load(seed, allow_pickle=False) as local, np.load(reference, allow_pickle=False) as rho:
        payload = {key: np.asarray(local[key]) for key in local.files}
        local_energy = np.asarray(payload["states__E_prod"], dtype=float)
        reference_energy = np.asarray(rho["states__E_prod"], dtype=float)
        if local_energy.shape != reference_energy.shape:
            # The accounting integral is the sole transferred quantity here.
            # A Radau-5 archive can seed a Radau-7 RHO provided that its
            # one-cycle E_prod profile is resampled on the target collocation
            # grid; no physical state is interpolated or transferred.
            if local_energy.ndim != 2 or reference_energy.ndim != 2 or (
                local_energy.shape[0] != 1 or reference_energy.shape[0] != 1
            ):
                raise ValueError("FHO and RHO E_prod layouts are incompatible.")
            reference_grid = np.linspace(0.0, 1.0, reference_energy.shape[1])
            local_grid = np.linspace(0.0, 1.0, local_energy.shape[1])
            local_energy = np.interp(
                local_grid, reference_grid, reference_energy[0]
            )[np.newaxis, :]
        payload["states__E_prod"] = local_energy
        metadata = json.loads(str(payload["metadata__json"]))
        metadata["cycle_boundary_reset_state_jumps"] = {
            "E_prod": float(payload["states__E_prod"][..., -1].reshape(-1)[0])
        }
        metadata["producer_e_prod_reset_from_reference_cycle"] = template_cycle
        metadata["producer_e_prod_reset_requested_cycle"] = cycle
        metadata["producer_e_prod_seed_resampled_to_target_grid"] = bool(
            local_energy.shape != reference_energy.shape
        )
        payload["metadata__json"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    np.savez(seed, **payload)


def terminal_handoff_error_excluding_cycle_integrals(source: Path, extension: Path) -> float:
    """Audit physical terminal states; E_prod is intentionally reset by RHO."""
    with np.load(source, allow_pickle=False) as left, np.load(extension, allow_pickle=False) as right:
        keys = {
            key for key in left.files if key.startswith("states__") and key != "states__E_prod"
        }
        if keys != {
            key for key in right.files if key.startswith("states__") and key != "states__E_prod"
        }:
            return float("inf")
        return max(
            float(np.max(np.abs(np.asarray(left[key])[..., -1] - np.asarray(right[key])[..., 0])))
            for key in keys
        )


def append_local_rho_to_cumulative_fho(
    runner: Any, prefix: Path, extension: Path, output: Path, *, nonoptimal_seed: bool = False
) -> None:
    """Append local RHO dynamics while restoring FHO's cumulative E_prod gauge."""
    runner.append_rho_extension_cycle(prefix, extension, output)
    with np.load(prefix, allow_pickle=False) as prefix_data, np.load(extension, allow_pickle=False) as extension_data, np.load(output, allow_pickle=False) as merged:
        payload = {key: np.asarray(merged[key]) for key in merged.files}
        prefix_terminal = np.asarray(prefix_data["states__E_prod"])[..., -1:]
        extension_energy = np.asarray(extension_data["states__E_prod"])
        prefix_count = np.asarray(prefix_data["states__E_prod"]).shape[-1]
        rebased_energy = prefix_terminal + extension_energy
        # RHO stores a reset at the boundary after its local cycle.  FHO uses
        # one cumulative accounting state, so carry the last pre-reset energy
        # increment through that common boundary instead.
        rebased_energy[..., -1:] = prefix_terminal + np.max(extension_energy, axis=-1, keepdims=True)
        payload["states__E_prod"][..., prefix_count - 1:] = rebased_energy
        metadata = json.loads(str(payload["metadata__json"]))
        metadata["producer_e_prod_local_reset_rebased_to_fho"] = True
        # Preserve this provenance through the FHO solve.  It is intentionally
        # separate from solver/certificate fields: a finite failed RHO can be
        # a useful warm start, never a validated trajectory.
        if nonoptimal_seed:
            metadata["producer_contains_noncertified_rho_seed"] = True
        payload["metadata__json"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    np.savez(output, **payload)


def repair_rho_by_initial_state_homotopy(
    runner: Any,
    *,
    carrier_solution: Path,
    reference_trace: Path,
    cycle: int,
    stage: Path,
    relative_limit: float,
    objective: str,
    weights: tuple[float, float, float, float] | None,
    collocation_degree: int,
) -> tuple[Path, Path, list[dict[str, Any]]]:
    """Move a known-good RHO cycle to the carrier terminal in certified steps.

    The direct terminal extrapolation is deliberately not reused after a
    continuity failure: it has an artificial E_prod defect.  Each accepted
    intermediate solve becomes the next warm start, and only fraction 1.0 is
    eligible to extend the FHO prefix.
    """
    raw_reference = stage / "reference-cycle-raw.npz"
    reference = stage / "reference-cycle.npz"
    # Homotopy needs a nominal cycle only as a shape/template.  Just like the
    # local E_prod reset, it must not make the continuation horizon depend on
    # how many cycles happened to be stored in the original RHO archive.
    with np.load(reference_trace, allow_pickle=False) as trace:
        reference_metadata = json.loads(str(trace["metadata__json"]))
    reference_cycles = int(
        reference_metadata.get("physical_cycles")
        or reference_metadata.get("producer_requested_cycles")
        or 0
    )
    if reference_cycles < 1:
        raise ValueError(f"Could not determine the RHO reference horizon in {reference_trace}.")
    template_cycle = min(cycle, reference_cycles)
    runner.write_rho_seed_cycle(reference_trace, raw_reference, template_cycle)
    prescribed_compatible_reference(raw_reference, reference)
    source = reference
    stages: list[dict[str, Any]] = []
    for index, fraction in enumerate((0.25, 0.50, 0.75, 1.00), start=1):
        homotopy_dir = stage / f"homotopy-{index:02d}-{fraction:.2f}"
        homotopy_dir.mkdir(exist_ok=True)
        seed = homotopy_dir / "seed.npz"
        solution = homotopy_dir / "solution.npz"
        result = homotopy_dir / "result.json"
        runner.write_rho_initial_state_homotopy_seed(
            source, reference, carrier_solution, seed, fraction
        )
        run(
            command_for_rho(
                seed, solution, result, objective=objective, weights=weights,
                collocation_degree=collocation_degree,
            ),
            homotopy_dir / "driver.log",
        )
        if solution.exists():
            # Standalone RHO exports retain theta/omega telemetry even though
            # prescribed FHO does not optimize those states.  Re-project each
            # accepted bridge before it becomes the next homotopy source.
            prescribed_compatible_reference(solution, solution)
        item = {
            "fraction": fraction,
            "result": str(result),
            "solver_window_valid": rho_window_valid(result),
        }
        if solution.exists():
            item.update(audit_summary(result, relative_limit))
        stages.append(item)
        # Intermediate fractions are deliberately nonphysical warm-start
        # bridges.  Requiring their terminal energy certificate would defeat
        # the homotopy; only the exact carrier fraction may extend FHO.
        final_stage_accepted = (
            fraction < 1.0 or item.get("operational_continuation_accepted", False)
        )
        if not item["solver_window_valid"] or not final_stage_accepted:
            raise RuntimeError(f"RHO_{cycle} homotopy stage {fraction:.2f} rejected.")
        source = solution
    return source, result, stages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--relative-energy-limit", type=float, default=0.001)
    parser.add_argument("--collocation-degree", type=int, default=5)
    parser.add_argument(
        "--fho-ipopt-max-iter", type=int, default=800,
        help="IPOPT iteration budget for the global FHO solve (RHO seed solves retain their own budget).",
    )
    parser.add_argument(
        "--fho-ding-local-sx",
        action="store_true",
        help=(
            "Use the exact SX Ding-local condensation adapter for FHO. It requires "
            "Radau-5, reconstructs the full solution, and audits original-space "
            "feasibility before promotion."
        ),
    )
    parser.add_argument("--start-cycles", type=int, default=3)
    parser.add_argument("--objective", choices=("fatigue", "control"), default="fatigue")
    parser.add_argument(
        "--objective-weight-values", type=float, nargs=4,
        metavar=("DELT_ANT", "DELT_POST", "BICEPS", "TRICEPS"),
        help="Fixed per-muscle fatigue or PW² weights, in the model's canonical order.",
    )
    parser.add_argument(
        "--reference-rho", type=Path,
        help="Optional RHO reference archive used only for local E_prod initialisation.",
    )
    parser.add_argument(
        "--extension-cycles", type=int, default=3,
        help="Number of RHO cycles to append before the next FHO (normally 3).",
    )
    parser.add_argument(
        "--attempt-label", default=None,
        help="Suffix for the audit of a +3/+2/+1 fallback attempt.",
    )
    parser.add_argument(
        "--attempt-dir",
        type=Path,
        help=(
            "Directory for this tentative +3/+2/+1 solve.  Sources are read "
            "from --output-dir and a successful FHO alone is promoted back "
            "there, preventing failed fallback artefacts from being reused."
        ),
    )
    parser.add_argument(
        "--allow-nonoptimal-single-rho-seed", action="store_true",
        help="For a one-cycle extension only, let a finite exported RHO candidate seed FHO despite RHO non-convergence.",
    )
    parser.add_argument(
        "--allow-rho-failure-fho-bridge", action="store_true",
        help=(
            "Treat RHO strictly as an FHO warm-start attempt: after a failed "
            "RHO and failed homotopy, continue with its finite output or the "
            "finite FHO-terminal projection.  Only the subsequent FHO audit "
            "can accept the extension."
        ),
    )
    parser.add_argument(
        "--audit-only-result",
        type=Path,
        help="Audit one completed FHO result under the operational policy; do not launch a continuation.",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        help="Where to write the audit-only JSON (defaults beside --audit-only-result).",
    )
    parser.add_argument(
        "--prepare-next-fho-only",
        action="store_true",
        help=(
            "Prepare the immutable launch contract for FHO_(i+extension) without "
            "reading a terminal state or launching IPOPT.  This is safe to run while "
            "FHO_i is still solving."
        ),
    )
    parser.add_argument(
        "--preparation-output",
        type=Path,
        help=(
            "JSON manifest produced by --prepare-next-fho-only. Defaults to "
            "<output-dir>/prepared/fho-<end>-preparation.json."
        ),
    )
    parser.add_argument(
        "--prepared-next-fho-manifest",
        type=Path,
        help=(
            "Optional manifest from --prepare-next-fho-only. Its horizon and protocol "
            "are checked before the real FHO is launched."
        ),
    )
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Keep the 0.1 % continuation decision explicit and reproducible for a
    # completed FHO as well as for a newly launched extension.  It never
    # changes the strict certificate stored by the solver result.
    if args.audit_only_result:
        result_path = args.audit_only_result.resolve()
        output = (args.audit_output or result_path.with_name("operational-audit.json")).resolve()
        payload = {
            "policy": "operational_high_accuracy_energy_limit",
            "result": str(result_path),
            **audit_summary(result_path, args.relative_energy_limit),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return

    weights = None if args.objective_weight_values is None else tuple(args.objective_weight_values)

    if args.start_cycles < 1:
        raise ValueError("--start-cycles must be positive.")
    start_cycles = args.start_cycles
    if args.extension_cycles < 1:
        raise ValueError("--extension-cycles must be positive.")
    if args.collocation_degree < 1:
        raise ValueError("--collocation-degree must be strictly positive.")
    if args.fho_ding_local_sx and args.collocation_degree != 5:
        raise ValueError("--fho-ding-local-sx requires the exact Radau-5 adapter.")
    if args.fho_ipopt_max_iter < 1:
        raise ValueError("--fho-ipopt-max-iter must be strictly positive.")
    if args.allow_nonoptimal_single_rho_seed and args.extension_cycles != 1:
        raise ValueError("A nonoptimal RHO seed is permitted only for --extension-cycles 1.")
    requested_end_cycles = start_cycles + args.extension_cycles
    if args.prepare_next_fho_only:
        preparation_output = (
            args.preparation_output.resolve()
            if args.preparation_output
            else out / "prepared" / f"fho-{requested_end_cycles:04d}-preparation.json"
        )
        written = write_next_fho_preparation(
            output_dir=out,
            output=preparation_output,
            start_cycles=start_cycles,
            extension_cycles=args.extension_cycles,
            objective=args.objective,
            weights=weights,
            collocation_degree=args.collocation_degree,
            ipopt_max_iter=args.fho_ipopt_max_iter,
            ding_local_sx=args.fho_ding_local_sx,
        )
        print(f"Prepared next-FHO launch contract: {written}")
        return

    runner = load_runner()
    # ``end_cycles`` may be shortened below when the first unrepaired RHO is
    # used only as a finite FHO warm start.  Continuing with subsequent RHOs
    # would propagate an already non-converged carrier and cannot improve the
    # seed for the immediate FHO bridge.
    end_cycles = requested_end_cycles
    if args.attempt_dir:
        work = args.attempt_dir.resolve()
        if work == out:
            raise ValueError("--attempt-dir must differ from --output-dir.")
        if work.exists() and any(work.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite continuation attempt artefacts in {work}."
            )
    else:
        # The default remains backward compatible, while making direct CLI
        # invocations collision-free across retries.
        label = args.attempt_label or f"plus{args.extension_cycles}"
        base = out / "attempts" / f"from-{start_cycles:04d}-{label}"
        work = base
        retry = 0
        while work.exists():
            retry += 1
            work = base.with_name(f"{base.name}-retry{retry:02d}")
    work.mkdir(parents=True, exist_ok=False)
    prefix_fho = out / f"fho-{start_cycles:04d}-solution.npz"
    prefix_result = out / f"fho-{start_cycles:04d}-result.json"
    reference_rho = (args.reference_rho.resolve() if args.reference_rho else out.parent / "rho-seed/validated-rho-trajectory.npz")
    if not prefix_fho.exists() or not prefix_result.exists():
        raise FileNotFoundError(f"FHO_{start_cycles} result and solution are required before continuation.")

    report: dict[str, Any] = {
        "policy": "operational_high_accuracy_energy_limit",
        "objective": args.objective,
        "objective_weight_values": weights,
        "reference_rho": str(reference_rho),
        "canonical_output_dir": str(out),
        "attempt_dir": str(work),
        "requested_end_cycles": requested_end_cycles,
        "entries": [],
    }
    if args.prepared_next_fho_manifest:
        report["prepared_next_fho_manifest"] = str(args.prepared_next_fho_manifest.resolve())
    suffix = "" if not args.attempt_label else f"-{args.attempt_label}"
    report_path = work / f"operational-continuation-from-{start_cycles:04d}{suffix}-audit.json"

    def persist_report() -> None:
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    entry = {"stage": f"FHO_{start_cycles}", "result": str(prefix_result), **audit_summary(prefix_result, args.relative_energy_limit)}
    report["entries"].append(entry)
    persist_report()
    if not entry["operational_continuation_accepted"]:
        raise RuntimeError(f"FHO_{start_cycles} does not meet the approved operational continuation policy.")

    source_solution = prefix_fho
    combined_seed = prefix_fho
    used_nonoptimal_rho_seed = False
    for cycle in range(start_cycles + 1, end_cycles + 1):
        stage = work / f"rho-{cycle:04d}"
        stage.mkdir(exist_ok=True)
        terminal_seed = stage / "terminal-seed.npz"
        rho_solution = stage / "solution.npz"
        rho_result = stage / "result.json"
        runner.write_fho_terminal_continuation_seed(source_solution, terminal_seed)
        reset_local_cycle_integral(terminal_seed, reference_rho, cycle, runner)
        rho_process_ok = run(
            command_for_rho(
                terminal_seed, rho_solution, rho_result, objective=args.objective, weights=weights,
                collocation_degree=args.collocation_degree,
            ),
            stage / "driver.log",
            allow_failure=(
                args.allow_nonoptimal_single_rho_seed
                or args.allow_rho_failure_fho_bridge
            ),
        )
        if rho_solution.exists():
            prescribed_compatible_reference(rho_solution, rho_solution)
        direct_ok = rho_process_ok and rho_solution.exists() and rho_window_valid(rho_result)
        allow_nonconverged_rho_seed = (
            args.allow_nonoptimal_single_rho_seed
            or args.allow_rho_failure_fho_bridge
        )
        failed_rho_solution_candidate = (
            allow_nonconverged_rho_seed
            and finite_rho_candidate(rho_solution, rho_result)
        )
        # IPOPT may fail before exporting a solution.  The pre-solve terminal
        # seed is nevertheless finite and has the exact FHO terminal physical
        # state (only E_prod is locally reset).  For +1 only, it can seed an
        # FHO restoration solve; it is emphatically not a RHO result.
        unresolved_terminal_seed_candidate = (
            allow_nonconverged_rho_seed
            and finite_solution_archive(terminal_seed)
        )
        nonoptimal_kind: str | None = None
        recovery_stages: list[dict[str, Any]] = []
        if not direct_ok:
            try:
                rho_solution, rho_result, recovery_stages = repair_rho_by_initial_state_homotopy(
                    runner,
                    carrier_solution=source_solution,
                    reference_trace=reference_rho,
                    cycle=cycle,
                    stage=stage,
                    relative_limit=args.relative_energy_limit,
                    objective=args.objective,
                    weights=weights,
                    collocation_degree=args.collocation_degree,
                )
            except RuntimeError:
                if failed_rho_solution_candidate:
                    nonoptimal_kind = (
                        "failed_rho_solution_fho_bridge"
                        if args.allow_rho_failure_fho_bridge
                        else "failed_rho_solution"
                    )
                elif unresolved_terminal_seed_candidate:
                    rho_solution = terminal_seed
                    nonoptimal_kind = (
                        "unresolved_rho_terminal_seed_fho_bridge"
                        if args.allow_rho_failure_fho_bridge
                        else "unresolved_rho_terminal_seed"
                    )
                else:
                    raise
        item = {"stage": f"RHO_{cycle}", "result": str(rho_result), **audit_summary(rho_result, args.relative_energy_limit)}
        item["direct_terminal_projection_accepted"] = direct_ok
        item["initial_state_homotopy"] = recovery_stages
        item["nonoptimal_rho_seed_used"] = nonoptimal_kind is not None
        item["nonoptimal_rho_seed_is_finite_only"] = item["nonoptimal_rho_seed_used"]
        item["nonoptimal_rho_seed_kind"] = nonoptimal_kind
        item["terminal_handoff_error_excluding_E_prod_reset"] = terminal_handoff_error_excluding_cycle_integrals(source_solution, rho_solution)
        report["entries"].append(item)
        persist_report()
        accepted_rho = item["operational_continuation_accepted"] or item["nonoptimal_rho_seed_used"]
        if not accepted_rho or item["terminal_handoff_error_excluding_E_prod_reset"] > 1e-8:
            raise RuntimeError(f"RHO_{cycle} rejected; no FHO extension was launched.")
        next_combined = work / f"fho-{start_cycles:04d}-plus-rho-through-{cycle:04d}.npz"
        append_local_rho_to_cumulative_fho(
            runner,
            combined_seed,
            rho_solution,
            next_combined,
            nonoptimal_seed=item["nonoptimal_rho_seed_used"],
        )
        used_nonoptimal_rho_seed = used_nonoptimal_rho_seed or item["nonoptimal_rho_seed_used"]
        source_solution, combined_seed = rho_solution, next_combined
        if item["nonoptimal_rho_seed_used"]:
            end_cycles = cycle
            report["extension_truncated_at_first_nonconverged_rho"] = True
            report["truncation_reason"] = (
                f"RHO_{cycle} did not converge; its finite output seeds FHO_{cycle} "
                "directly and no later RHO is solved."
            )
            persist_report()
            break

    fho_solution = work / f"fho-{end_cycles:04d}-solution.npz"
    fho_result = work / f"fho-{end_cycles:04d}-result.json"
    if args.prepared_next_fho_manifest:
        prepared = load_prepared_next_fho(
            args.prepared_next_fho_manifest.resolve(),
            start_cycles=start_cycles,
            end_cycles=end_cycles,
            objective=args.objective,
            weights=weights,
            collocation_degree=args.collocation_degree,
            ipopt_max_iter=args.fho_ipopt_max_iter,
            ding_local_sx=args.fho_ding_local_sx,
        )
        report["prepared_next_fho_consumed"] = {
            "manifest_created_at_utc": prepared.get("created_at_utc"),
            "graph_reuse_available": prepared.get("graph_reuse", {}).get("available"),
            "certified_terminal_bound_after_audit": True,
        }
        persist_report()
    run(
        command_for_fho(
            combined_seed, prefix_fho, fho_solution, fho_result, end_cycles,
            objective=args.objective, weights=weights,
            collocation_degree=args.collocation_degree, ipopt_max_iter=args.fho_ipopt_max_iter,
            ding_local_sx=args.fho_ding_local_sx,
        ),
        out / f"fho-{end_cycles:04d}-driver.log",
    )
    final_entry = {
        "stage": f"FHO_{end_cycles}",
        "result": str(fho_result),
        "contains_noncertified_rho_seed": used_nonoptimal_rho_seed,
        **audit_summary(fho_result, args.relative_energy_limit),
    }
    report["entries"].append(final_entry)
    persist_report()
    if not final_entry["operational_continuation_accepted"]:
        raise RuntimeError(
            f"FHO_{end_cycles} rejected by the operational continuation policy."
        )

    # Promote only a fully accepted operational FHO.  No rejected +3/+2/+1
    # archive can accidentally become the next prefix.  ``copy2`` then
    # ``replace`` keeps readers from observing a half-written canonical file.
    for source, target in (
        (fho_solution, out / f"fho-{end_cycles:04d}-solution.npz"),
        (fho_result, out / f"fho-{end_cycles:04d}-result.json"),
    ):
        temporary = target.with_name(target.name + ".tmp")
        shutil.copy2(source, temporary)
        temporary.replace(target)
    canonical_report = out / f"operational-continuation-from-{start_cycles:04d}{suffix}-audit.json"
    temporary_report = canonical_report.with_name(canonical_report.name + ".tmp")
    shutil.copy2(report_path, temporary_report)
    temporary_report.replace(canonical_report)
    provenance = {
        "fho_cycles": end_cycles,
        "requested_end_cycles": requested_end_cycles,
        "extension_truncated_at_first_nonconverged_rho": end_cycles != requested_end_cycles,
        "operational_continuation_accepted": True,
        "strict_scientific_success": bool(final_entry["strict_scientific_success"]),
        "contains_noncertified_rho_seed": used_nonoptimal_rho_seed,
        "meaning": (
            "A finite but non-converged RHO was used only as an FHO warm start; "
            "it is not a validated RHO result."
            if used_nonoptimal_rho_seed
            else "All appended RHO stages were accepted by the operational policy."
        ),
        "attempt_report": str(report_path),
    }
    provenance_path = out / f"fho-{end_cycles:04d}-continuation-provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
