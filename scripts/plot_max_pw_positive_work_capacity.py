#!/usr/bin/env python3
"""Plot one-cycle PW-max positive-work opportunities from saved fatigue states.

This deliberately is *not* a new endurance simulation or a coupled torque
feasibility solve.  At every archived cycle boundary it replays each muscle
independently for the following prescribed crank revolution with its pulse
width held at ``PW_max``.  The complete five-state Ding state is retained.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cocofest")

import matplotlib.pyplot as plt
import numpy as np

from cocofest.dynamics.reduced_cycling import ReducedCyclingDynamics
from cocofest.optimization.adaptive_moment_rollout import (
    DingPulseWidthParameters,
    MomentTrackingInterval,
)
from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters
from cocofest.optimization.max_pw_work_capacity import (
    MaxPwWorkCapacityLayout,
    build_max_pw_work_capacity_function,
    max_pw_work_domain_lower_bounds,
    pack_max_pw_work_profile,
)


OUT = ROOT / "local-results/isokinetic-prescribed-unit-1p3-pw2-20261001/analysis"
MUSCLES = ("Biceps", "Delt_ant", "Delt_post", "Triceps")
FATIGUE_FHO_DIR = ROOT / "local-results/isokinetic-prescribed-unit-1p3-fho-bo64-20260930/fho-continuation"
PW_FHO_DIR = ROOT / "local-results/isokinetic-prescribed-unit-1p3-pw2-20261001/fho-pw-bo-step1/continuation"


def latest_operational_fho(directory: Path) -> tuple[int, Path]:
    """Select a promoted continuation, excluding partial FHO attempts."""
    candidates: list[tuple[int, Path]] = []
    for provenance in directory.glob("fho-????-continuation-provenance.json"):
        try:
            record = json.loads(provenance.read_text(encoding="utf-8"))
            cycles = int(record["fho_cycles"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        solution = directory / f"fho-{cycles:04d}-solution.npz"
        if record.get("operational_continuation_accepted") and solution.is_file():
            candidates.append((cycles, solution))
    if not candidates:
        raise FileNotFoundError(f"No promoted operational FHO found in {directory}.")
    return max(candidates)


FATIGUE_FHO_CYCLES, FATIGUE_FHO = latest_operational_fho(FATIGUE_FHO_DIR)
PW_FHO_CYCLES, PW_FHO = latest_operational_fho(PW_FHO_DIR)
SOURCES = {
    "RHO min-fatigue": (
        ROOT / "local-results/isokinetic-prescribed-unit-1p3-fho-bo64-20260930/rho-seed/validated-rho-trajectory.npz",
        "#4c78a8", "--",
    ),
    "RHO_BO min-fatigue": (OUT / "replay-rho-bo-min-fatigue/trajectory.npz", "#1f4e79", "-"),
    f"FHO min-fatigue (op., {FATIGUE_FHO_CYCLES})": (FATIGUE_FHO, "#54a24b", "-"),
    "RHO min-PW²": (
        ROOT / "local-results/isokinetic-prescribed-unit-1p3-pw2-20261001/rho-unit-pw/validated-rho-trajectory.npz",
        "#e45756", "--",
    ),
    "RHO_BO min-PW²": (OUT / "replay-rho-bo-pw/trajectory.npz", "#b23a48", "-"),
    f"FHO min-PW² (op., {PW_FHO_CYCLES})": (PW_FHO, "#8c3d1c", "-"),
}
SUBSTEPS = 4
FORCE_ROUNDOFF_TOLERANCE_N = 1e-5


def _metadata(data: np.lib.npyio.NpzFile) -> dict:
    return json.loads(str(data["metadata__json"]))


def _profile_path(metadata: dict) -> Path:
    path = Path(metadata["reduced_profile"]["path"])
    return path if path.is_absolute() else ROOT / path


def _parameters(metadata: dict, reduced: ReducedCyclingDynamics) -> tuple[DingPulseWidthParameters, ...]:
    configured = metadata["configured_muscle_parameters"]
    pw_max = float(metadata["pulse_width_maximum_s"])
    result = []
    for name in reduced.muscle_names:
        values = configured[name]
        fatigue = DingFatigueParameters(
            a_rest=float(values["a_scale"]), tau1_rest=float(values["tau1_rest"]),
            km_rest=float(values["km_rest"]), alpha_a=float(values["alpha_a"]),
            alpha_tau1=float(values["alpha_tau1"]), alpha_km=float(values["alpha_km"]),
            tau_fat=float(values["tau_fat"]),
        )
        result.append(DingPulseWidthParameters(
            fatigue=fatigue, tauc=float(values["tauc"]), tau2=float(values["tau2"]),
            pd0=float(values["pd0"]), pdt=float(values["pdt"]), pulse_width_max=pw_max,
        ))
    return tuple(result)


def _calcium_amplitude(parameter: DingPulseWidthParameters, phase_duration: float, truncation: int) -> float:
    """Periodic-node post-stimulation amplitude used by the archived OCP."""
    decay = float(np.exp(-phase_duration / parameter.tauc))
    # Ding 2003's r0(Km_rest) = Km_rest + 1.04.
    increment = 1.0 + (parameter.fatigue.km_rest + 1.04 - 1.0) * decay
    if truncation == 1:
        return 1.0
    return float(decay ** (truncation - 1) + increment * sum(decay**age for age in range(truncation - 1)))


def _capacity_profile(
    *, theta0: float, omega: float, period: float, reduced: ReducedCyclingDynamics,
    parameters: tuple[DingPulseWidthParameters, ...], phases: int, truncation: int,
) -> np.ndarray:
    """Fixed future mechanics for exactly one prescribed revolution."""
    duration = period / phases
    intervals = []
    for phase in range(phases):
        start = phase * duration

        def gain(local_t: float, *, start=start) -> np.ndarray:
            theta = theta0 + omega * (start + local_t)
            fl, fv, passive = reduced.muscle_relationships(theta, omega)
            return np.asarray(fl, dtype=float) * np.asarray(fv, dtype=float) + np.asarray(passive, dtype=float)

        def coefficient(local_t: float, *, start=start) -> np.ndarray:
            theta = theta0 + omega * (start + local_t)
            return np.asarray(reduced.coefficient_values(theta)["muscle_effectiveness"], dtype=float)

        # The capacity packer accepts scalar or callable values per muscle.
        intervals.append(MomentTrackingInterval(
            duration=duration,
            calcium_amplitudes=tuple(_calcium_amplitude(item, duration, truncation) for item in parameters),
            mechanical_gains=tuple(
                lambda t, index=index: float(gain(t)[index]) for index in range(len(parameters))
            ),
            # ``MomentTrackingInterval`` stores a numerical midpoint value;
            # the packet below receives the time-dependent coefficients
            # separately and uses them for the positive-work integral.
            moment_coefficients=tuple(float(coefficient(.5 * duration)[index]) for index in range(len(parameters))),
            target_moments=(0.0,) * len(parameters),
        ))
    layout = MaxPwWorkCapacityLayout(len(parameters), phases, integration_substeps=SUBSTEPS)
    return pack_max_pw_work_profile(
        intervals, layout, angular_velocity_rad_s=omega,
        moment_coefficient_functions=[row.moment_coefficients for row in intervals],
        stimulation_policy="all_intervals_pw_max",
    )


def _boundary_states(data: np.lib.npyio.NpzFile, metadata: dict, muscle_names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    cycles = int(metadata["cycles_per_window"])
    # Full-horizon prescribed archives intentionally omit theta/omega because
    # they are algebraic functions of time.  Reconstruct their cycle-boundary
    # phase exactly from the declared prescribed velocity and period.
    if "states__theta" in data:
        theta = np.asarray(data["states__theta"], dtype=float).reshape(-1)
    else:
        reference = np.asarray(data[f"states__Cn_{muscle_names[0]}"], dtype=float).reshape(-1)
        stride, remainder = divmod(reference.size - 1, cycles)
        if remainder or stride < 1:
            raise ValueError("Archive state layout cannot be partitioned into complete cycles.")
        theta = np.linspace(
            0.0,
            float(metadata["isokinetic_omega"]) * float(metadata["cycle_duration_s"]) * cycles,
            reference.size,
        )
    stride, remainder = divmod(theta.size - 1, cycles)
    if remainder or stride < 1:
        raise ValueError("Archive state layout cannot be partitioned into complete cycles.")
    boundary = np.arange(cycles + 1) * stride
    states = np.empty((cycles + 1, len(muscle_names), 5))
    for muscle_index, name in enumerate(muscle_names):
        for state_index, field in enumerate(("Cn", "F", "A", "Tau1", "Km")):
            values = np.asarray(data[f"states__{field}_{name}"], dtype=float).reshape(-1)
            if values.shape != theta.shape:
                raise ValueError(f"{field}_{name} does not share theta's archive layout.")
            states[:, muscle_index, state_index] = values[boundary]
    return theta[boundary], states


def capacity_series(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as data:
        metadata = _metadata(data)
        omega = float(metadata["isokinetic_omega"])
        phases = int(metadata["stimulations_per_cycle"])
        period = float(metadata["cycle_duration_s"])
        reduced = ReducedCyclingDynamics.load(_profile_path(metadata))
        if tuple(reduced.muscle_names) != ("Delt_ant", "Delt_post", "Biceps", "Triceps"):
            raise ValueError(f"Unexpected reduced muscle order: {reduced.muscle_names}")
        parameters = _parameters(metadata, reduced)
        theta, states = _boundary_states(data, metadata, tuple(reduced.muscle_names))
    minimum_initial_force = float(np.min(states[:, :, 1]))
    if minimum_initial_force < -FORCE_ROUNDOFF_TOLERANCE_N:
        raise ValueError(
            f"PW-max capacity cannot start from materially negative force ({minimum_initial_force:.3e} N) in {path}."
        )
    # IPOPT's operational FHO export contains a single -4.9e-7 N force,
    # far below its primal tolerance.  Project only this declared roundoff to
    # the physical boundary; a materially negative state is rejected above.
    states[:, :, 1] = np.maximum(states[:, :, 1], 0.0)
    layout = MaxPwWorkCapacityLayout(len(parameters), phases, integration_substeps=SUBSTEPS)
    capacity = build_max_pw_work_capacity_function(muscles=parameters, layout=layout, symbolic_type="SX")
    lower = max_pw_work_domain_lower_bounds(layout)
    values = np.empty((states.shape[0], len(parameters)))
    truncation = int(metadata["ding_sum_stim_truncation"])
    for cycle, (theta0, current) in enumerate(zip(theta, states)):
        profile = _capacity_profile(
            theta0=float(theta0), omega=omega, period=period, reduced=reduced,
            parameters=parameters, phases=phases, truncation=truncation,
        )
        _, per_muscle, _, margins = capacity(current.ravel(), profile)
        margins = np.asarray(margins, dtype=float).reshape(-1)
        if np.any(~np.isfinite(margins)) or np.any(margins < lower - 1e-10):
            raise ValueError(f"PW-max capacity probe leaves the Ding domain at cycle {cycle} for {path}.")
        values[cycle] = np.asarray(per_muscle, dtype=float).reshape(-1)
    # Archives and figures use this anatomical display order, not the reduced-model order.
    index = [tuple(reduced.muscle_names).index(name) for name in MUSCLES]
    audit = {
        "archive": str(path), "cycles": int(values.shape[0] - 1), "reduced_muscle_order": list(reduced.muscle_names),
        "integration": "independent one-cycle RK4, 4 substeps per 30 stimulation intervals",
        "future_stimulation": "PW=PW_max in every interval; periodic-node calcium history retained",
        "work_formula": "W_i^+ = integral_0^T max(omega*b_i(theta(t)), 0) F_i(t) dt",
        "mechanics": "prescribed theta(t)=theta_boundary+omega*t; archive reduced force-length/velocity/passive geometry",
        "interpretation": "independent-muscle positive-work opportunity, not coupled-torque feasible work, endurance prediction, or certificate",
        "initial_force_roundoff_projection": {
            "minimum_exported_force_n": minimum_initial_force,
            "tolerance_n": FORCE_ROUNDOFF_TOLERANCE_N,
            "rule": "clip only force in [-tolerance, 0) to zero; reject a more negative force",
        },
        "parameters": {name: asdict(parameter) for name, parameter in zip(reduced.muscle_names, parameters)},
    }
    return np.arange(values.shape[0]), values[:, index], audit


def main() -> None:
    missing = [str(path) for path, _, _ in SOURCES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing trajectory archive(s): " + ", ".join(missing))
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.5), sharex=True, constrained_layout=True)
    summary: dict[str, object] = {"definition": {
        "quantity": "per-muscle independent one-cycle positive-work opportunity at PW_max",
        "formula": "W_i^+ = integral_0^T max(omega*b_i(theta(t)), 0) F_i(t) dt",
        "not": "actual optimized work, coupled torque-feasible work, endurance simulation, or physiological certificate",
    }, "sources": {}}
    for source, (path, color, style) in SOURCES.items():
        x, work, audit = capacity_series(path)
        summary["sources"][source] = {**audit, "work_j_by_cycle_boundary": {
            muscle: [float(value) for value in work[:, index]] for index, muscle in enumerate(MUSCLES)
        }}
        for index, axis in enumerate(axes.flat):
            axis.plot(x, work[:, index], label=source, color=color, linestyle=style, linewidth=1.6)
    for axis, muscle in zip(axes.flat, MUSCLES):
        axis.set_title(muscle); axis.grid(True, alpha=.25)
    axes[0, 0].set_ylabel("opportunité de travail positif (J/cycle)")
    axes[1, 0].set_ylabel("opportunité de travail positif (J/cycle)")
    axes[1, 0].set_xlabel("frontière de cycle"); axes[1, 1].set_xlabel("frontière de cycle")
    axes[0, 0].legend(fontsize=7.0, loc="best")
    fig.suptitle("Capacité individuelle de travail positif au PW maximal", fontsize=13)
    fig.text(
        .5, -.025,
        "À chaque frontière : replay Ding à PW=max sur un tour prescrit; Wᵢ⁺=∫max(ω bᵢ(θ),0)Fᵢdt. "
        "Opportunité indépendante, non une faisabilité couplée ni du travail optimisé.",
        ha="center", fontsize=8,
    )
    png = OUT / "endurance-max-pw-positive-work-capacity.png"
    figure_json = OUT / "endurance-max-pw-positive-work-capacity.json"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    figure_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(png)
    print(figure_json)


if __name__ == "__main__":
    main()
