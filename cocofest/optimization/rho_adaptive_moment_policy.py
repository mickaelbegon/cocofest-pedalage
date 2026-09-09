r"""Build a low-cost adaptive muscle-moment policy from one certified RHO cycle.

Only the selected RHO cycle is used.  Its individual muscle moments become
phase-dependent targets; future pulse widths are predicted by bounded scalar
inversions of the full five-state Ding dynamics.  No FHO trajectory, learned
terminal value, or a-priori muscle weighting is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from cocofest.dynamics.reduced_cycling import ReducedCyclingDynamics
from cocofest.optimization.adaptive_moment_rollout import (
    DingPulseWidthParameters,
    MomentTrackingInterval,
)
from cocofest.optimization.rho_rollout_adapter import (
    SelectedRhoCycle,
    _build_actual_muscle_models,
    _collocation_profile,
    _default_model_path,
    select_certified_rho_cycle,
)


@dataclass(frozen=True)
class RhoAdaptiveMomentPolicy:
    """All fixed data needed to predict future PWs without another OCP."""

    muscle_names: tuple[str, ...]
    source_cycle_index: int
    source_cycle_count: int
    period: float
    initial_states: np.ndarray
    intervals: tuple[MomentTrackingInterval, ...]
    parameters: tuple[DingPulseWidthParameters, ...]
    source_pulse_widths: np.ndarray
    target_moments: np.ndarray
    certification_basis: str
    period_basis: str


def _models_by_name(models: Sequence[Any], muscle_names: tuple[str, ...]) -> dict[str, Any]:
    by_name = {str(model.muscle_name): model for model in models}
    if set(by_name) != set(muscle_names):
        raise ValueError(
            "Ding model and reduced-profile muscle names differ: "
            f"models={sorted(by_name)}, reduced={sorted(muscle_names)}."
        )
    return by_name


def _validate_source_formulations(cycle: SelectedRhoCycle) -> None:
    expected = {
        "model_formulation": "periodic_node",
        "calcium_forcing_formulation": "exact_exponential_periodic_node",
        "mechanical_formulation": "reduced",
    }
    mismatches = {
        key: cycle.metadata.get(key)
        for key, value in expected.items()
        if cycle.metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "Adaptive moment prediction requires the same validated RHO formulations; "
            f"observed mismatches={mismatches}, expected={expected}."
        )


def _interval_gain(
    coefficients: np.ndarray,
    *,
    interval_duration: float,
    reduced: Any,
    muscle_index: int,
    activate_force_length: bool,
    activate_force_velocity: bool,
    activate_passive_force: bool,
):
    """Bind one muscle to one source collocation kinematic polynomial."""

    coefficients = np.asarray(coefficients, dtype=float).copy()

    def gain(local_time: float) -> float:
        normalized_time = float(local_time) / interval_duration
        theta = float(np.polynomial.polynomial.polyval(normalized_time, coefficients[0]))
        omega = float(np.polynomial.polynomial.polyval(normalized_time, coefficients[1]))
        force_length, force_velocity, passive_force = reduced.muscle_relationships(theta, omega)
        active_length = float(force_length[muscle_index]) if activate_force_length else 1.0
        active_velocity = float(force_velocity[muscle_index]) if activate_force_velocity else 1.0
        passive = float(passive_force[muscle_index]) if activate_passive_force else 0.0
        return active_length * active_velocity + passive

    return gain


def build_rho_adaptive_moment_policy(
    source: str | Path,
    reduced_profile: str | Path,
    *,
    cycle_index: int = 0,
    cycle_period: float | None = None,
    model_path: str | Path | None = None,
    muscle_models: Sequence[Any] | None = None,
    reduced_dynamics: Any | None = None,
) -> RhoAdaptiveMomentPolicy:
    """Convert one certified RHO cycle into repeated individual-moment targets."""

    cycle, period_basis = select_certified_rho_cycle(
        source,
        cycle_index=cycle_index,
        cycle_period=cycle_period,
    )
    if not cycle.certified:
        raise ValueError(f"The selected source is not a certified RHO trajectory: {cycle.certification_basis}.")
    _validate_source_formulations(cycle)

    reduced = (
        ReducedCyclingDynamics.load(reduced_profile)
        if reduced_dynamics is None
        else reduced_dynamics
    )
    muscle_names = tuple(str(name) for name in reduced.muscle_names)
    if reduced.muscle_geometry is None:
        raise ValueError("The reduced profile must contain muscle geometry.")
    actual_models = (
        _build_actual_muscle_models(
            cycle,
            _default_model_path() if model_path is None else Path(model_path),
        )
        if muscle_models is None
        else tuple(muscle_models)
    )
    by_name = _models_by_name(actual_models, muscle_names)
    pulse_width_max = float(cycle.metadata["pulse_width_maximum_s"])
    parameters = tuple(
        DingPulseWidthParameters.from_model(by_name[name], pulse_width_max=pulse_width_max)
        for name in muscle_names
    )
    amplitudes = tuple(float(by_name[name].post_stimulation_amplitude()) for name in muscle_names)

    state_names = ("Cn", "F", "A", "Tau1", "Km")
    initial_states = np.asarray(
        [
            [cycle.states[f"{state_name}_{name}"][cycle.start_column] for state_name in state_names]
            for name in muscle_names
        ],
        dtype=float,
    )
    interval_duration = cycle.period / cycle.stimulations_per_cycle
    kinematics = _collocation_profile(cycle, ("theta", "omega"))
    source_pulse_widths = np.empty((len(muscle_names), cycle.stimulations_per_cycle))
    target_moments = np.empty_like(source_pulse_widths)
    intervals = []
    first_control = cycle.cycle_index * cycle.stimulations_per_cycle
    activate_force_length = bool(cycle.metadata.get("activate_force_length_relationship", True))
    activate_force_velocity = bool(cycle.metadata.get("activate_force_velocity_relationship", True))
    activate_passive_force = bool(cycle.metadata.get("activate_passive_force_relationship", True))

    for interval_index in range(cycle.stimulations_per_cycle):
        endpoint_column = (
            cycle.start_column + (interval_index + 1) * cycle.state_columns_per_interval
        )
        theta_endpoint = float(cycle.states["theta"][endpoint_column])
        moment_coefficients = tuple(
            float(value)
            for value in reduced.coefficient_values(theta_endpoint)["muscle_effectiveness"]
        )
        gains = tuple(
            _interval_gain(
                kinematics.coefficients[:, interval_index, :],
                interval_duration=interval_duration,
                reduced=reduced,
                muscle_index=muscle_index,
                activate_force_length=activate_force_length,
                activate_force_velocity=activate_force_velocity,
                activate_passive_force=activate_passive_force,
            )
            for muscle_index in range(len(muscle_names))
        )
        for muscle_index, name in enumerate(muscle_names):
            source_pulse_widths[muscle_index, interval_index] = cycle.controls[
                f"last_pulse_width_{name}"
            ][first_control + interval_index]
            target_moments[muscle_index, interval_index] = (
                moment_coefficients[muscle_index]
                * cycle.states[f"F_{name}"][endpoint_column]
            )
        intervals.append(
            MomentTrackingInterval(
                duration=interval_duration,
                calcium_amplitudes=amplitudes,
                mechanical_gains=gains,
                moment_coefficients=moment_coefficients,
                target_moments=tuple(target_moments[:, interval_index]),
            )
        )

    return RhoAdaptiveMomentPolicy(
        muscle_names=muscle_names,
        source_cycle_index=cycle.cycle_index,
        source_cycle_count=cycle.cycle_count,
        period=cycle.period,
        initial_states=initial_states,
        intervals=tuple(intervals),
        parameters=parameters,
        source_pulse_widths=source_pulse_widths,
        target_moments=target_moments,
        certification_basis=cycle.certification_basis,
        period_basis=period_basis,
    )
