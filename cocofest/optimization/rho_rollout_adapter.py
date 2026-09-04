r"""Auditable adapter from a certified RHO NPZ export to endurance rollouts.

The adapter understands the collocation-node layout written by the cycling RHO
benchmark.  It selects the final certified cycle from metadata, reconstructs
the actual Radau sample times, fits explicit periodic Fourier representations
of ``F`` and ``Cn``, and evaluates reduced muscle geometry at interval
midpoints.  It never treats the dense collocation columns as uniformly spaced.

The periodic-policy assumption is a scientific gate.  A source without
certification metadata, an invalid crank winding, non-periodic ``omega``, ``F``
or ``Cn``, excessive Fourier residual, or negative reconstructed force produces
a rejected report with measured reasons.  Values are not clipped to make a
rollout run.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from cocofest.dynamics.reduced_cycling import ReducedCyclingDynamics
from cocofest.optimization.ding_fatigue_rollout import (
    ding_fatigue_parameters_from_model,
    physiological_ding_fatigue_state,
)
from cocofest.optimization.endurance_rollout import (
    DingRolloutMuscleParameters,
    PeriodicRecruitmentProfile,
    rollout_periodic_ding_endurance,
)
from cocofest.optimization.periodic_force_profile import PeriodicFourierForceProfile
from cocofest.optimization.recruitment_margin import ding_recruitment_margin


REPORT_SCHEMA = "cocofest-rho-endurance-rollout-v1"
DEFAULT_HORIZONS = (5, 10, 20)
DEFAULT_FORCE_HARMONICS = 12
DEFAULT_PERIODICITY_RELATIVE_TOLERANCE = 1e-3
DEFAULT_FORCE_SEAM_ABSOLUTE_TOLERANCE_N = 1e-3
DEFAULT_CN_SEAM_ABSOLUTE_TOLERANCE = 1e-6
DEFAULT_FOURIER_RELATIVE_RMSE_TOLERANCE = 5e-2
DEFAULT_FOURIER_RELATIVE_MAXIMUM_TOLERANCE = 0.35
DEFAULT_KINEMATIC_CONSISTENCY_TOLERANCE = 0.1
DEFAULT_NEGATIVE_FORCE_TOLERANCE = 1e-10
DEFAULT_PULSE_WIDTH_BOUND_TOLERANCE_S = 1e-10


@dataclass(frozen=True)
class SelectedRhoCycle:
    """The final cycle selected from an archive's declared collocation layout."""

    metadata: dict[str, Any]
    certified: bool
    certification_basis: str
    cycle_index: int
    cycle_count: int
    stimulations_per_cycle: int
    collocation_degree: int
    collocation_method: str
    period: float
    sample_times: np.ndarray
    sample_columns: np.ndarray
    start_column: int
    end_column: int
    state_columns_per_interval: int
    states: dict[str, np.ndarray]
    controls: dict[str, np.ndarray]


def _file_stamp(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a strictly positive integer.")
    try:
        integer = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a strictly positive integer.") from error
    if integer < 1 or integer != value:
        raise ValueError(f"{name} must be a strictly positive integer.")
    return integer


def _positive_float(value: Any, *, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and strictly positive.") from error
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive.")
    return numeric


def _cycle_period(metadata: dict[str, Any], override: float | None) -> tuple[float, str]:
    if override is not None:
        return _positive_float(override, name="cycle_period"), "cli_override"
    for key in ("cycle_duration_s", "cycle_duration"):
        if metadata.get(key) is not None:
            return _positive_float(metadata[key], name=key), f"metadata.{key}"
    stimulation_interval = metadata.get("calcium_stimulation_interval_s")
    if stimulation_interval is not None:
        stimulations = _positive_integer(metadata.get("stimulations_per_cycle"), name="stimulations_per_cycle")
        return (
            _positive_float(stimulation_interval, name="calcium_stimulation_interval_s") * stimulations,
            "metadata.calcium_stimulation_interval_s*stimulations_per_cycle",
        )
    nominal_omega = metadata.get("isokinetic_omega")
    if nominal_omega is not None:
        nominal_omega = float(nominal_omega)
        if math.isfinite(nominal_omega) and nominal_omega != 0.0:
            return 2.0 * math.pi / abs(nominal_omega), "metadata.isokinetic_omega"
    raise ValueError("Cycle duration is absent and cannot be inferred from a non-zero isokinetic_omega.")


def _certification(metadata: dict[str, Any]) -> tuple[bool, str, int]:
    mode = metadata.get("producer_mode")
    if mode == "receding_horizon_concatenation":
        cycles = _positive_integer(metadata.get("cycles_per_window"), name="cycles_per_window")
        return True, "producer_mode=receding_horizon_concatenation", cycles
    if mode == "rho_replay_checkpoint":
        completed = int(metadata.get("producer_completed_windows", 0))
        return completed >= 1, f"producer_mode=rho_replay_checkpoint;completed_windows={completed}", 1
    cycles = _positive_integer(metadata.get("cycles_per_window"), name="cycles_per_window")
    return False, f"producer_mode={mode!r} does not certify a RHO cycle", cycles


def _state_row(values: np.ndarray, *, key: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[0] != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{key} must be one finite scalar-state row.")
    return array[0].copy()


def _control_row(values: np.ndarray, *, key: str) -> np.ndarray:
    return _state_row(values, key=key)


def select_last_certified_rho_cycle(
    source: str | Path,
    *,
    cycle_period: float | None = None,
) -> tuple[SelectedRhoCycle, str]:
    """Load and structurally select the final metadata-declared RHO cycle.

    The returned object may have ``certified=False`` so callers can still write
    a complete rejection audit.  Malformed or ambiguous array layouts raise.
    """

    source = Path(source)
    with np.load(source, allow_pickle=False) as archive:
        if "metadata__json" not in archive.files:
            raise ValueError("RHO archive has no metadata__json entry.")
        metadata = json.loads(str(archive["metadata__json"].item()))
        certified, basis, cycle_count = _certification(metadata)
        stimulations = _positive_integer(metadata.get("stimulations_per_cycle"), name="stimulations_per_cycle")
        degree = _positive_integer(metadata.get("producer_collocation_degree"), name="producer_collocation_degree")
        method = str(metadata.get("producer_collocation_method") or "").lower()
        if not method:
            raise ValueError("producer_collocation_method is required for dense collocation exports.")
        period, period_basis = _cycle_period(metadata, cycle_period)
        state_keys = sorted(key for key in archive.files if key.startswith("states__"))
        control_keys = sorted(key for key in archive.files if key.startswith("controls__"))
        if not state_keys or not control_keys:
            raise ValueError("RHO archive must contain state and control traces.")
        states = {key.removeprefix("states__"): _state_row(archive[key], key=key) for key in state_keys}
        controls = {key.removeprefix("controls__"): _control_row(archive[key], key=key) for key in control_keys}

    interval_count = cycle_count * stimulations
    expected_controls = interval_count
    expected_states = interval_count * (degree + 1) + 1
    observed_state_lengths = {values.size for values in states.values()}
    observed_control_lengths = {values.size for values in controls.values()}
    if observed_state_lengths != {expected_states}:
        raise ValueError(
            "State trace length does not match cycles*stimulations*(collocation_degree+1)+1: "
            f"observed={sorted(observed_state_lengths)}, expected={expected_states}."
        )
    if observed_control_lengths != {expected_controls}:
        raise ValueError(
            "Control trace length does not match cycles*stimulations: "
            f"observed={sorted(observed_control_lengths)}, expected={expected_controls}."
        )

    import casadi as ca

    try:
        collocation_points = np.asarray(ca.collocation_points(degree, method), dtype=float)
    except Exception as error:
        raise ValueError(f"Unsupported collocation rule {method!r} of degree {degree}.") from error
    if collocation_points.shape != (degree,) or np.any(collocation_points <= 0.0):
        raise ValueError("Collocation abscissae are inconsistent with the declared degree.")

    cycle_index = cycle_count - 1
    first_interval = cycle_index * stimulations
    start_column = first_interval * (degree + 1)
    end_column = (first_interval + stimulations) * (degree + 1)
    sample_columns = [start_column]
    normalized_times = [0.0]
    for local_interval in range(stimulations):
        base = start_column + local_interval * (degree + 1)
        sample_columns.extend(base + np.arange(1, degree + 1))
        normalized_times.extend((local_interval + collocation_points) / stimulations)
    sample_columns = np.asarray(sample_columns, dtype=int)
    normalized_times = np.asarray(normalized_times, dtype=float)
    # A Radau endpoint is also stored as the next shooting node.  Retain the
    # shooting endpoint only for seam audits and remove t=T from the Fourier fit.
    fit_mask = normalized_times < 1.0 - 1e-14
    return (
        SelectedRhoCycle(
            metadata=metadata,
            certified=certified,
            certification_basis=basis,
            cycle_index=cycle_index,
            cycle_count=cycle_count,
            stimulations_per_cycle=stimulations,
            collocation_degree=degree,
            collocation_method=method,
            period=period,
            sample_times=period * normalized_times[fit_mask],
            sample_columns=sample_columns[fit_mask],
            start_column=start_column,
            end_column=end_column,
            state_columns_per_interval=degree + 1,
            states=states,
            controls=controls,
        ),
        period_basis,
    )


def _fit_periodic_at_times(
    samples: np.ndarray,
    times: np.ndarray,
    *,
    period: float,
    harmonic_count: int,
) -> PeriodicFourierForceProfile:
    samples = np.asarray(samples, dtype=float)
    times = np.asarray(times, dtype=float)
    if samples.ndim == 1:
        samples = samples[None, :]
    if samples.ndim != 2 or times.ndim != 1 or samples.shape[1] != times.size:
        raise ValueError("Fourier samples and times have incompatible shapes.")
    harmonic_count = _positive_integer(harmonic_count, name="harmonic_count")
    if 2 * harmonic_count + 1 > times.size:
        raise ValueError("Fourier order is underdetermined by the selected collocation samples.")
    phase = 2.0 * math.pi * times / period
    design = np.column_stack(
        [np.ones_like(phase)]
        + [np.cos(harmonic * phase) for harmonic in range(1, harmonic_count + 1)]
        + [np.sin(harmonic * phase) for harmonic in range(1, harmonic_count + 1)]
    )
    coefficients, *_ = np.linalg.lstsq(design, samples.T, rcond=None)
    return PeriodicFourierForceProfile(
        period=period,
        mean=coefficients[0],
        cosine=coefficients[1 : harmonic_count + 1].T,
        sine=coefficients[harmonic_count + 1 :].T,
    )


def square_periodic_fourier_profile(
    profile: PeriodicFourierForceProfile,
) -> PeriodicFourierForceProfile:
    r"""Return the exact Fourier coefficients of ``profile(t) ** 2``.

    If the input contains ``H`` harmonics, the returned profile contains
    ``2H`` harmonics.  Coefficients are convolved in their complex Fourier
    representation; this is an algebraic conversion, not a second fit.
    """

    harmonic_count = profile.harmonic_count
    signal_count = profile.signal_count
    squared_mean = np.empty(signal_count)
    squared_cosine = np.empty((signal_count, 2 * harmonic_count))
    squared_sine = np.empty_like(squared_cosine)
    for signal_index in range(signal_count):
        complex_coefficients = np.zeros(2 * harmonic_count + 1, dtype=complex)
        center = harmonic_count
        complex_coefficients[center] = profile.mean[signal_index]
        for harmonic in range(1, harmonic_count + 1):
            positive = 0.5 * (
                profile.cosine[signal_index, harmonic - 1]
                - 1j * profile.sine[signal_index, harmonic - 1]
            )
            complex_coefficients[center + harmonic] = positive
            complex_coefficients[center - harmonic] = np.conjugate(positive)
        squared = np.convolve(complex_coefficients, complex_coefficients)
        squared_center = 2 * harmonic_count
        squared_mean[signal_index] = float(squared[squared_center].real)
        for harmonic in range(1, 2 * harmonic_count + 1):
            positive = squared[squared_center + harmonic]
            squared_cosine[signal_index, harmonic - 1] = 2.0 * positive.real
            squared_sine[signal_index, harmonic - 1] = -2.0 * positive.imag
    return PeriodicFourierForceProfile(
        period=profile.period,
        mean=squared_mean,
        cosine=squared_cosine,
        sine=squared_sine,
    )


def fit_nonnegative_periodic_fourier_force_profile(
    samples: np.ndarray,
    times: np.ndarray,
    *,
    period: float,
    root_harmonic_count: int,
) -> PeriodicFourierForceProfile:
    r"""Fit ``sqrt(F)`` and exactly square its Fourier series.

    The source samples must already satisfy ``F >= 0``.  No negative sample is
    clipped or replaced.  The resulting analytical Fourier series is the exact
    square of the fitted root series and is therefore non-negative apart from
    floating-point evaluation round-off.
    """

    samples = np.asarray(samples, dtype=float)
    if np.any(samples < 0.0):
        raise ValueError("A non-negative Fourier force fit requires non-negative source samples; no clipping is used.")
    root_profile = _fit_periodic_at_times(
        np.sqrt(samples),
        times,
        period=period,
        harmonic_count=root_harmonic_count,
    )
    return square_periodic_fourier_profile(root_profile)


def _signal_scales(samples: np.ndarray) -> np.ndarray:
    return np.maximum.reduce(
        (
            np.max(np.abs(samples), axis=1),
            np.ptp(samples, axis=1),
            np.full(samples.shape[0], 1e-12),
        )
    )


def _fourier_audit(
    profile: PeriodicFourierForceProfile,
    samples: np.ndarray,
    times: np.ndarray,
    *,
    dense_count: int,
) -> tuple[dict[str, Any], np.ndarray]:
    reconstructed = profile.evaluate(times)
    error = reconstructed - samples
    scales = _signal_scales(samples)
    dense_times = np.linspace(0.0, profile.period, dense_count, endpoint=False)
    dense = profile.evaluate(dense_times)
    rmse = np.sqrt(np.mean(error**2, axis=1))
    maximum = np.max(np.abs(error), axis=1)
    return (
        {
            "harmonic_count": profile.harmonic_count,
            "sample_count": int(times.size),
            "rmse": rmse.tolist(),
            "relative_rmse": (rmse / scales).tolist(),
            "maximum_absolute_error": maximum.tolist(),
            "relative_maximum_absolute_error": (maximum / scales).tolist(),
            "dense_minimum": np.min(dense, axis=1).tolist(),
            "dense_maximum": np.max(dense, axis=1).tolist(),
        },
        dense,
    )


def _endpoint_audit(samples: np.ndarray, endpoints: np.ndarray) -> dict[str, Any]:
    scales = _signal_scales(samples)
    differences = endpoints[:, 1] - endpoints[:, 0]
    return {
        "start": endpoints[:, 0].tolist(),
        "end": endpoints[:, 1].tolist(),
        "difference": differences.tolist(),
        "relative_difference": (np.abs(differences) / scales).tolist(),
    }


def _state_samples(cycle: SelectedRhoCycle, prefix: str, muscle_names: Sequence[str]) -> np.ndarray:
    rows = []
    for muscle_name in muscle_names:
        key = f"{prefix}_{muscle_name}"
        if key not in cycle.states:
            raise ValueError(f"RHO archive is missing state {key!r}.")
        rows.append(cycle.states[key][cycle.sample_columns])
    return np.vstack(rows)


def _state_endpoints(cycle: SelectedRhoCycle, prefix: str, muscle_names: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [
            [
                cycle.states[f"{prefix}_{muscle_name}"][cycle.start_column],
                cycle.states[f"{prefix}_{muscle_name}"][cycle.end_column],
            ]
            for muscle_name in muscle_names
        ],
        dtype=float,
    )


def _lagrange_weights(nodes: np.ndarray, query: float) -> np.ndarray:
    """Return polynomial interpolation weights at one normalized interval time."""

    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 1 or nodes.size < 2 or not np.all(np.isfinite(nodes)):
        raise ValueError("Collocation interpolation nodes must be a finite vector.")
    if np.unique(nodes).size != nodes.size:
        raise ValueError("Collocation interpolation nodes must be distinct.")
    weights = np.ones(nodes.size)
    for index, node in enumerate(nodes):
        others = np.delete(nodes, index)
        weights[index] = np.prod((float(query) - others) / (node - others))
    return weights


def _state_midpoints_from_collocation(
    cycle: SelectedRhoCycle,
    prefix: str,
    muscle_names: Sequence[str],
) -> np.ndarray:
    """Interpolate a selected cycle's state at every interval midpoint.

    Each interval uses its shooting node and its declared degree-specific
    collocation stages.  In particular, dense archive columns are not treated
    as uniformly spaced samples.
    """

    import casadi as ca

    try:
        stages = np.asarray(
            ca.collocation_points(cycle.collocation_degree, cycle.collocation_method),
            dtype=float,
        )
    except Exception as error:
        raise ValueError(
            f"Unsupported collocation rule {cycle.collocation_method!r} "
            f"of degree {cycle.collocation_degree}."
        ) from error
    nodes = np.concatenate(([0.0], stages))
    weights = _lagrange_weights(nodes, 0.5)
    midpoints = np.empty((len(muscle_names), cycle.stimulations_per_cycle), dtype=float)
    for muscle_index, muscle_name in enumerate(muscle_names):
        key = f"{prefix}_{muscle_name}"
        if key not in cycle.states:
            raise ValueError(f"RHO archive is missing state {key!r}.")
        trace = cycle.states[key]
        for interval_index in range(cycle.stimulations_per_cycle):
            base = cycle.start_column + interval_index * cycle.state_columns_per_interval
            columns = base + np.arange(nodes.size)
            midpoints[muscle_index, interval_index] = float(trace[columns] @ weights)
    return midpoints


def _default_model_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "examples/msk_models/Wu/Modified_Wu_Shoulder_Model_Cycling.bioMod"
    )


def _build_actual_muscle_models(
    cycle: SelectedRhoCycle,
    model_path: Path,
) -> Sequence[Any]:
    from examples.fes_multibody.cycling.cycling_pulse_width_mhe import set_fes_model

    formulation = cycle.metadata.get("model_formulation")
    if formulation not in {"standard", "periodic", "periodic_node"}:
        raise ValueError(f"Unsupported model_formulation {formulation!r}.")
    stim_time = np.linspace(
        0.0,
        cycle.period,
        cycle.stimulations_per_cycle,
        endpoint=False,
    ).tolist()
    model = set_fes_model(
        str(model_path),
        stim_time,
        periodic_cn_sum_approximation=formulation == "periodic",
        periodic_node_forcing=formulation == "periodic_node",
    )
    return model.muscles_dynamics_model


def _rollout_parameters(
    muscle_names: Sequence[str],
    muscle_models: Sequence[Any],
    pulse_width_maximum: float,
) -> tuple[list[DingRolloutMuscleParameters], list[dict[str, Any]]]:
    by_name = {str(model.muscle_name): model for model in muscle_models}
    if set(by_name) != set(muscle_names):
        raise ValueError(
            "Muscle names in the instantiated Ding models and reduced profile differ: "
            f"models={sorted(by_name)}, reduced={sorted(muscle_names)}."
        )
    parameters = []
    report = []
    for name in muscle_names:
        model = by_name[name]
        fatigue = ding_fatigue_parameters_from_model(model)
        parameter = DingRolloutMuscleParameters(
            fatigue=fatigue,
            tau2=float(model.tau2),
            pd0=float(model.pd0),
            pdt=float(model.pdt),
            pulse_width_max=pulse_width_maximum,
        )
        parameters.append(parameter)
        report.append(
            {
                "muscle": name,
                "model_class": type(model).__name__,
                "A_rest": fatigue.a_rest,
                "Tau1_rest": fatigue.tau1_rest,
                "Km_rest": fatigue.km_rest,
                "alpha_A": fatigue.alpha_a,
                "alpha_Tau1": fatigue.alpha_tau1,
                "alpha_Km": fatigue.alpha_km,
                "tau_fat": fatigue.tau_fat,
                "tau2": parameter.tau2,
                "pd0": parameter.pd0,
                "pdt": parameter.pdt,
                "pulse_width_max": parameter.pulse_width_max,
            }
        )
    return parameters, report


def _finite_json_number(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def adapter_completion_status(
    rejection_reasons: Sequence[dict[str, Any]],
    *,
    reference_policy_reproducible: bool | None,
) -> str:
    """Classify adaptation independently of future endurance failures.

    Input adaptation failures take precedence.  Once the input is accepted,
    the selected source cycle must itself admit the bounded pulse-width inverse
    under its own collocation-interpolated slow states.  Feasibility loss only
    in later rollout cycles is deliberately not an adapter failure; it is the
    endurance result being measured.
    """

    if rejection_reasons:
        return "rejected"
    if reference_policy_reproducible is not True:
        return "reference_policy_not_reproducible"
    return "complete"


def _error_summary(errors: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(errors)
    values = errors[finite]
    return {
        "finite_sample_count": int(values.size),
        "maximum_absolute_error": None if values.size == 0 else float(np.max(np.abs(values))),
        "rmse": None if values.size == 0 else float(np.sqrt(np.mean(values**2))),
    }


def _reference_policy_audit(
    *,
    source_cycle_index: int,
    midpoint_times: np.ndarray,
    muscle_names: Sequence[str],
    force: np.ndarray,
    force_derivative: np.ndarray,
    cn: np.ndarray,
    source_slow_states: np.ndarray,
    force_length: np.ndarray,
    force_velocity: np.ndarray,
    passive_force: np.ndarray,
    source_pulse_widths: np.ndarray,
    parameters: Sequence[DingRolloutMuscleParameters],
    pulse_width_tolerance_s: float,
) -> dict[str, Any]:
    """Invert the adapted policy on the selected RHO cycle's own slow state."""

    muscle_count = len(muscle_names)
    interval_count = midpoint_times.size
    expected_phase_shape = (muscle_count, interval_count)
    arrays = {
        "force": force,
        "force_derivative": force_derivative,
        "cn": cn,
        "force_length": force_length,
        "force_velocity": force_velocity,
        "passive_force": passive_force,
        "source_pulse_widths": source_pulse_widths,
    }
    for name, values in arrays.items():
        if np.asarray(values).shape != expected_phase_shape:
            raise ValueError(f"Reference-policy {name} must have shape {expected_phase_shape}.")
    if source_slow_states.shape != (muscle_count, interval_count, 3):
        raise ValueError(
            "Reference-policy source_slow_states must have shape "
            f"({muscle_count}, {interval_count}, 3)."
        )
    if len(parameters) != muscle_count:
        raise ValueError("Reference-policy parameters must contain one entry per muscle.")

    status_counts: Counter[str] = Counter()
    per_muscle_counts = {name: Counter() for name in muscle_names}
    utilization = np.full(expected_phase_shape, np.nan)
    derivative_error = np.full(expected_phase_shape, np.nan)
    pulse_width_error = np.full(expected_phase_shape, np.nan)
    first_failure: dict[str, Any] | None = None

    for interval_index in range(interval_count):
        for muscle_index, (muscle_name, parameter) in enumerate(
            zip(muscle_names, parameters, strict=True)
        ):
            capacity, tau1, km = source_slow_states[muscle_index, interval_index]
            diagnostic = ding_recruitment_margin(
                cn=cn[muscle_index, interval_index],
                force=force[muscle_index, interval_index],
                force_derivative=force_derivative[muscle_index, interval_index],
                capacity=capacity,
                tau1=tau1,
                km=km,
                tau2=parameter.tau2,
                pd0=parameter.pd0,
                pdt=parameter.pdt,
                pulse_width_max=parameter.pulse_width_max,
                force_length_relationship=force_length[muscle_index, interval_index],
                force_velocity_relationship=force_velocity[muscle_index, interval_index],
                passive_force_relationship=passive_force[muscle_index, interval_index],
                pulse_width_tolerance=pulse_width_tolerance_s,
            )
            status = diagnostic.status.value
            status_counts[status] += 1
            per_muscle_counts[muscle_name][status] += 1
            utilization[muscle_index, interval_index] = diagnostic.utilization
            if math.isfinite(diagnostic.reconstructed_force_derivative):
                derivative_error[muscle_index, interval_index] = (
                    diagnostic.reconstructed_force_derivative
                    - force_derivative[muscle_index, interval_index]
                )
            if math.isfinite(diagnostic.required_pulse_width):
                pulse_width_error[muscle_index, interval_index] = (
                    diagnostic.required_pulse_width
                    - source_pulse_widths[muscle_index, interval_index]
                )
            if first_failure is None and not diagnostic.feasible:
                first_failure = {
                    "source_cycle_index": int(source_cycle_index),
                    "interval_index": int(interval_index),
                    "midpoint_time_s": float(midpoint_times[interval_index]),
                    "muscle_index": int(muscle_index),
                    "muscle": muscle_name,
                    "status": status,
                    "utilization": _finite_json_number(diagnostic.utilization),
                    "required_pulse_width_s": _finite_json_number(
                        diagnostic.required_pulse_width
                    ),
                }

    finite_utilization = utilization[np.isfinite(utilization)]
    slow_state_ranges = {}
    for state_index, state_name in enumerate(("A", "Tau1", "Km")):
        values = source_slow_states[:, :, state_index]
        slow_state_ranges[state_name] = {
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
    return {
        "evaluated": True,
        "reproducible": first_failure is None,
        "source_cycle_index": int(source_cycle_index),
        "sample_count": int(muscle_count * interval_count),
        "failure_count": int(sum(count for status, count in status_counts.items() if status != "ok")),
        "status_counts": dict(sorted(status_counts.items())),
        "status_counts_by_muscle": {
            name: dict(sorted(counts.items())) for name, counts in per_muscle_counts.items()
        },
        "first_failure": first_failure,
        "maximum_finite_utilization": (
            None if finite_utilization.size == 0 else float(np.max(finite_utilization))
        ),
        "force_derivative_reconstruction_error_n_per_s": _error_summary(derivative_error),
        "inferred_vs_exported_pulse_width_error_s": _error_summary(pulse_width_error),
        "source_slow_state_ranges": slow_state_ranges,
        "slow_state_source": "degree-specific_piecewise_collocation_polynomial_at_shared_midpoints",
        "force_derivative_source": "analytical_derivative_of_adapted_periodic_fourier_force_profile",
    }


def _location_payload(location, muscle_names: Sequence[str]) -> dict[str, Any] | None:
    if location is None:
        return None
    return {
        "cycle_index": int(location.cycle_index),
        "interval_index": int(location.interval_index),
        "muscle_index": int(location.muscle_index),
        "muscle": muscle_names[location.muscle_index],
    }


def _horizon_payload(result, horizon: int, muscle_names: Sequence[str], parameters) -> dict[str, Any]:
    final_states = result.cycle_boundary_states[-1]
    capacity_ratios = np.asarray(
        [final_states[index, 0] / parameter.fatigue.a_rest for index, parameter in enumerate(parameters)]
    )
    statuses = Counter(
        diagnostic.status.value
        for cycle in result.diagnostics
        for interval in cycle
        for diagnostic in interval
    )
    first_failure = result.first_failure
    return {
        "horizon_cycles": horizon,
        "feasible": result.feasible,
        "worst_utilization": _finite_json_number(result.worst_utilization),
        "worst_utilization_location": _location_payload(result.worst_utilization_location, muscle_names),
        "first_failure": (
            None
            if first_failure is None
            else {
                **(_location_payload(first_failure, muscle_names) or {}),
                "status": first_failure.status.value,
                "utilization": _finite_json_number(first_failure.utilization),
                "required_pulse_width_s": _finite_json_number(first_failure.required_pulse_width),
            }
        ),
        "status_counts": dict(sorted(statuses.items())),
        "final_slow_states": final_states.tolist(),
        "final_capacity_ratios": capacity_ratios.tolist(),
        "minimum_final_capacity_ratio": float(np.min(capacity_ratios)),
        "maximum_marginal_damage_rate_s_inv": float(np.max(result.marginal_damage_rate)),
    }


def build_rho_endurance_rollout_report(
    source: str | Path,
    reduced_profile: str | Path,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    cycle_period: float | None = None,
    force_harmonics: int = DEFAULT_FORCE_HARMONICS,
    cn_harmonics: int | None = None,
    kinematic_harmonics: int = DEFAULT_FORCE_HARMONICS,
    periodicity_relative_tolerance: float = DEFAULT_PERIODICITY_RELATIVE_TOLERANCE,
    force_seam_absolute_tolerance_n: float = DEFAULT_FORCE_SEAM_ABSOLUTE_TOLERANCE_N,
    cn_seam_absolute_tolerance: float = DEFAULT_CN_SEAM_ABSOLUTE_TOLERANCE,
    fourier_relative_rmse_tolerance: float = DEFAULT_FOURIER_RELATIVE_RMSE_TOLERANCE,
    fourier_relative_maximum_tolerance: float = DEFAULT_FOURIER_RELATIVE_MAXIMUM_TOLERANCE,
    kinematic_consistency_tolerance: float = DEFAULT_KINEMATIC_CONSISTENCY_TOLERANCE,
    negative_force_tolerance: float = DEFAULT_NEGATIVE_FORCE_TOLERANCE,
    pulse_width_bound_tolerance_s: float = DEFAULT_PULSE_WIDTH_BOUND_TOLERANCE_S,
    model_path: str | Path | None = None,
    muscle_models: Sequence[Any] | None = None,
    reduced_dynamics: Any | None = None,
) -> dict[str, Any]:
    """Build a compact, solver-independent rollout or an explicit rejection.

    ``muscle_models`` and ``reduced_dynamics`` are injectable for tests.  The
    production path loads the reduced cache, validates it against the bioMod,
    and calls the benchmark's actual ``set_fes_model`` factory before extracting
    Ding parameters from those instantiated models.
    """

    source = Path(source)
    reduced_profile = Path(reduced_profile)
    horizons = tuple(_positive_integer(value, name="horizon") for value in horizons)
    if not horizons:
        raise ValueError("At least one rollout horizon is required.")
    if len(set(horizons)) != len(horizons):
        raise ValueError("Rollout horizons must be unique.")
    thresholds = {
        "periodicity_relative": _positive_float(
            periodicity_relative_tolerance, name="periodicity_relative_tolerance"
        ),
        "force_seam_absolute_n": _positive_float(
            force_seam_absolute_tolerance_n, name="force_seam_absolute_tolerance_n"
        ),
        "cn_seam_absolute": _positive_float(
            cn_seam_absolute_tolerance, name="cn_seam_absolute_tolerance"
        ),
        "fourier_relative_rmse": _positive_float(
            fourier_relative_rmse_tolerance, name="fourier_relative_rmse_tolerance"
        ),
        "fourier_relative_maximum": _positive_float(
            fourier_relative_maximum_tolerance, name="fourier_relative_maximum_tolerance"
        ),
        "kinematic_consistency_rad_s": _positive_float(
            kinematic_consistency_tolerance, name="kinematic_consistency_tolerance"
        ),
        "negative_force_tolerance_n": _positive_float(
            negative_force_tolerance, name="negative_force_tolerance"
        ),
        "pulse_width_bound_tolerance_s": _positive_float(
            pulse_width_bound_tolerance_s, name="pulse_width_bound_tolerance_s"
        ),
    }
    cycle, period_basis = select_last_certified_rho_cycle(source, cycle_period=cycle_period)
    model_path = _default_model_path() if model_path is None else Path(model_path)
    reduced = ReducedCyclingDynamics.load(reduced_profile) if reduced_dynamics is None else reduced_dynamics
    if reduced_dynamics is None:
        reduced.validate_source_model(model_path)
    if getattr(reduced, "muscle_geometry", object()) is None:
        raise ValueError("Reduced cycling profile has no muscle geometry.")
    muscle_names = tuple(str(name) for name in reduced.muscle_names)
    if not muscle_names:
        raise ValueError("Reduced cycling profile has no muscle names.")

    sample_count = cycle.sample_times.size
    force_harmonics = _positive_integer(force_harmonics, name="force_harmonics")
    kinematic_harmonics = _positive_integer(kinematic_harmonics, name="kinematic_harmonics")
    cn_harmonics = (
        min(2 * cycle.stimulations_per_cycle, (sample_count - 1) // 2)
        if cn_harmonics is None
        else _positive_integer(cn_harmonics, name="cn_harmonics")
    )
    reasons: list[dict[str, Any]] = []

    def reject(code: str, message: str, **metrics: Any) -> None:
        reasons.append({"code": code, "message": message, **metrics})

    if not cycle.certified:
        reject("source_not_certified", cycle.certification_basis)

    force_samples = _state_samples(cycle, "F", muscle_names)
    cn_samples = _state_samples(cycle, "Cn", muscle_names)
    force_endpoints = _state_endpoints(cycle, "F", muscle_names)
    cn_endpoints = _state_endpoints(cycle, "Cn", muscle_names)
    force_endpoint_audit = _endpoint_audit(force_samples, force_endpoints)
    cn_endpoint_audit = _endpoint_audit(cn_samples, cn_endpoints)
    for kind, audit, absolute_tolerance in (
        ("force", force_endpoint_audit, thresholds["force_seam_absolute_n"]),
        ("cn", cn_endpoint_audit, thresholds["cn_seam_absolute"]),
    ):
        absolute = np.abs(np.asarray(audit["difference"], dtype=float))
        relative = np.asarray(audit["relative_difference"], dtype=float)
        violations = (absolute > absolute_tolerance) & (relative > thresholds["periodicity_relative"])
        if np.any(violations):
            reject(
                f"non_periodic_{kind}",
                f"{kind} endpoint mismatch exceeds both declared seam tolerances.",
                maximum_absolute_difference=float(np.max(absolute)),
                maximum_relative_difference=float(np.max(relative)),
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=thresholds["periodicity_relative"],
            )

    force_profile = fit_nonnegative_periodic_fourier_force_profile(
        force_samples,
        cycle.sample_times,
        period=cycle.period,
        root_harmonic_count=force_harmonics,
    )
    cn_profile = _fit_periodic_at_times(
        cn_samples,
        cycle.sample_times,
        period=cycle.period,
        harmonic_count=cn_harmonics,
    )
    dense_count = max(10 * sample_count, 1000)
    force_fourier_audit, dense_force = _fourier_audit(
        force_profile,
        force_samples,
        cycle.sample_times,
        dense_count=dense_count,
    )
    force_fourier_audit.update(
        {
            "fit_space": "sqrt_force",
            "root_harmonic_count": force_harmonics,
            "nonnegative_construction": "exact_complex_fourier_convolution_of_squared_root_series",
            "clipping_applied": False,
        }
    )
    cn_fourier_audit, dense_cn = _fourier_audit(
        cn_profile,
        cn_samples,
        cycle.sample_times,
        dense_count=dense_count,
    )
    for kind, audit in (("force", force_fourier_audit), ("cn", cn_fourier_audit)):
        maximum_rmse = max(audit["relative_rmse"])
        maximum_error = max(audit["relative_maximum_absolute_error"])
        if maximum_rmse > thresholds["fourier_relative_rmse"]:
            reject(
                f"{kind}_fourier_rmse_too_large",
                f"{kind} Fourier relative RMSE exceeds tolerance.",
                maximum_relative_rmse=maximum_rmse,
                tolerance=thresholds["fourier_relative_rmse"],
            )
        if maximum_error > thresholds["fourier_relative_maximum"]:
            reject(
                f"{kind}_fourier_maximum_error_too_large",
                f"{kind} Fourier maximum relative error exceeds tolerance.",
                maximum_relative_error=maximum_error,
                tolerance=thresholds["fourier_relative_maximum"],
            )
    minimum_dense_force = float(np.min(dense_force))
    if minimum_dense_force < -thresholds["negative_force_tolerance_n"]:
        reject(
            "negative_fourier_force",
            "The direct Fourier force reconstruction is negative; no clipping is permitted.",
            minimum_reconstructed_force_n=minimum_dense_force,
            tolerance_n=thresholds["negative_force_tolerance_n"],
        )
    if float(np.min(dense_cn)) <= 0.0:
        reject(
            "non_positive_fourier_cn",
            "The Fourier Cn reconstruction leaves the positive activation domain.",
            minimum_reconstructed_cn=float(np.min(dense_cn)),
        )

    theta = cycle.states.get("theta")
    omega = cycle.states.get("omega")
    if theta is None or omega is None:
        raise ValueError("Reduced rollout requires scalar theta and omega state traces.")
    theta_samples = theta[cycle.sample_columns][None, :]
    omega_samples = omega[cycle.sample_columns][None, :]
    direction = int(reduced.kinematics.direction)
    expected_shift = direction * 2.0 * math.pi
    normalized_time = cycle.sample_times / cycle.period
    theta_residual_samples = theta_samples - expected_shift * normalized_time[None, :]
    theta_residual_profile = _fit_periodic_at_times(
        theta_residual_samples,
        cycle.sample_times,
        period=cycle.period,
        harmonic_count=kinematic_harmonics,
    )
    omega_profile = _fit_periodic_at_times(
        omega_samples,
        cycle.sample_times,
        period=cycle.period,
        harmonic_count=kinematic_harmonics,
    )
    theta_shift = float(theta[cycle.end_column] - theta[cycle.start_column])
    theta_winding_error = theta_shift - expected_shift
    theta_tolerance = max(
        1e-8,
        float(cycle.metadata.get("terminal_wheel_q_slack") or 0.0) + 1e-8,
    )
    if abs(theta_winding_error) > theta_tolerance:
        reject(
            "invalid_theta_winding",
            "Crank-angle shift is inconsistent with one reduced-profile revolution.",
            observed_shift_rad=theta_shift,
            expected_shift_rad=expected_shift,
            absolute_error_rad=abs(theta_winding_error),
            tolerance_rad=theta_tolerance,
        )
    omega_endpoint_audit = _endpoint_audit(
        omega_samples,
        np.array([[omega[cycle.start_column], omega[cycle.end_column]]]),
    )
    omega_endpoint_error = omega_endpoint_audit["relative_difference"][0]
    if omega_endpoint_error > thresholds["periodicity_relative"]:
        reject(
            "non_periodic_omega",
            "omega endpoint mismatch exceeds the declared relative tolerance.",
            relative_difference=omega_endpoint_error,
            tolerance=thresholds["periodicity_relative"],
        )

    midpoint_times = (np.arange(cycle.stimulations_per_cycle) + 0.5) * (
        cycle.period / cycle.stimulations_per_cycle
    )
    force_midpoints = force_profile.evaluate(midpoint_times)
    force_derivative_midpoints = force_profile.derivative(midpoint_times)
    cn_midpoints = cn_profile.evaluate(midpoint_times)
    theta_midpoints = (
        expected_shift * midpoint_times / cycle.period + theta_residual_profile.evaluate(midpoint_times)[0]
    )
    omega_midpoints = omega_profile.evaluate(midpoint_times)[0]
    theta_derivative_midpoints = (
        expected_shift / cycle.period + theta_residual_profile.derivative(midpoint_times)[0]
    )
    theta_omega_error = theta_derivative_midpoints - omega_midpoints
    maximum_kinematic_error = float(np.max(np.abs(theta_omega_error)))
    if maximum_kinematic_error > thresholds["kinematic_consistency_rad_s"]:
        reject(
            "inconsistent_theta_omega_interpolation",
            "The separate periodic theta/omega interpolants violate theta_dot=omega at midpoints.",
            maximum_absolute_error_rad_s=maximum_kinematic_error,
            tolerance_rad_s=thresholds["kinematic_consistency_rad_s"],
        )

    force_length = np.empty((len(muscle_names), cycle.stimulations_per_cycle))
    force_velocity = np.empty_like(force_length)
    passive_force = np.empty_like(force_length)
    for index, (theta_value, omega_value) in enumerate(zip(theta_midpoints, omega_midpoints, strict=True)):
        fl, fv, fp = reduced.muscle_relationships(float(theta_value), float(omega_value))
        force_length[:, index] = np.asarray(fl, dtype=float)
        force_velocity[:, index] = np.asarray(fv, dtype=float)
        passive_force[:, index] = np.asarray(fp, dtype=float)
    raw_gains = {
        "force_length": force_length.copy(),
        "force_velocity": force_velocity.copy(),
        "passive_force": passive_force.copy(),
    }
    if not bool(cycle.metadata.get("activate_force_length_relationship", True)):
        force_length.fill(1.0)
    if not bool(cycle.metadata.get("activate_force_velocity_relationship", True)):
        force_velocity.fill(1.0)
    if not bool(cycle.metadata.get("activate_passive_force_relationship", True)):
        passive_force.fill(0.0)
    expected_midpoint_shape = (len(muscle_names), cycle.stimulations_per_cycle)
    for name, values in (
        ("force", force_midpoints),
        ("cn", cn_midpoints),
        ("force_length", force_length),
        ("force_velocity", force_velocity),
        ("passive_force", passive_force),
    ):
        if values.shape != expected_midpoint_shape:
            raise ValueError(
                f"Aligned midpoint {name} values have shape {values.shape}; expected {expected_midpoint_shape}."
            )

    pulse_width_maximum = _positive_float(
        cycle.metadata.get("pulse_width_maximum_s"), name="pulse_width_maximum_s"
    )
    actual_models = (
        _build_actual_muscle_models(cycle, model_path) if muscle_models is None else muscle_models
    )
    parameters, parameter_report = _rollout_parameters(muscle_names, actual_models, pulse_width_maximum)
    source_slow_states = np.stack(
        [
            _state_midpoints_from_collocation(cycle, state_name, muscle_names)
            for state_name in ("A", "Tau1", "Km")
        ],
        axis=2,
    )
    initial_slow_states = np.column_stack(
        [
            _state_endpoints(cycle, state_name, muscle_names)[:, 1]
            for state_name in ("A", "Tau1", "Km")
        ]
    )
    for index, parameter in enumerate(parameters):
        try:
            physiological_ding_fatigue_state(initial_slow_states[index], parameter.fatigue)
        except ValueError as error:
            reject(
                "non_physiological_terminal_slow_state",
                str(error),
                muscle=muscle_names[index],
                state=initial_slow_states[index].tolist(),
            )

    control_audit = {}
    source_pulse_widths = np.empty(expected_midpoint_shape)
    for index, (name, parameter) in enumerate(zip(muscle_names, parameters, strict=True)):
        key = f"last_pulse_width_{name}"
        if key not in cycle.controls:
            raise ValueError(f"RHO archive is missing control {key!r}.")
        first = cycle.cycle_index * cycle.stimulations_per_cycle
        last = first + cycle.stimulations_per_cycle
        values = cycle.controls[key][first:last]
        source_pulse_widths[index] = values
        control_audit[name] = {"minimum_s": float(np.min(values)), "maximum_s": float(np.max(values))}
        pw_tolerance = thresholds["pulse_width_bound_tolerance_s"]
        if np.min(values) < parameter.pd0 - pw_tolerance or np.max(values) > parameter.pulse_width_max + pw_tolerance:
            reject(
                "pulse_width_control_out_of_domain",
                "Exported pulse-width controls leave the instantiated model bounds.",
                muscle=name,
                minimum_s=float(np.min(values)),
                maximum_s=float(np.max(values)),
                pd0_s=parameter.pd0,
                pulse_width_max_s=parameter.pulse_width_max,
                tolerance_s=pw_tolerance,
            )

    reference_policy: dict[str, Any] = {
        "evaluated": False,
        "reproducible": None,
        "reason": "input_adaptation_rejected",
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": adapter_completion_status(
            reasons,
            reference_policy_reproducible=reference_policy["reproducible"],
        ),
        "source": _file_stamp(source),
        "reduced_profile": _file_stamp(reduced_profile),
        "selection": {
            "certified": cycle.certified,
            "certification_basis": cycle.certification_basis,
            "selected_cycle_index": cycle.cycle_index,
            "declared_cycle_count": cycle.cycle_count,
            "stimulations_per_cycle": cycle.stimulations_per_cycle,
            "collocation_degree": cycle.collocation_degree,
            "collocation_method": cycle.collocation_method,
            "state_columns_per_interval": cycle.state_columns_per_interval,
            "fourier_sample_count": sample_count,
            "cycle_period_s": cycle.period,
            "cycle_period_basis": period_basis,
        },
        "configuration": {
            "horizons": list(horizons),
            "force_sqrt_harmonics": force_harmonics,
            "force_resulting_harmonics": force_profile.harmonic_count,
            "cn_harmonics": cn_harmonics,
            "kinematic_harmonics": kinematic_harmonics,
            "thresholds": thresholds,
        },
        "muscle_names": list(muscle_names),
        "model_parameters": parameter_report,
        "audits": {
            "force_endpoint_periodicity": force_endpoint_audit,
            "cn_endpoint_periodicity": cn_endpoint_audit,
            "force_fourier": force_fourier_audit,
            "cn_fourier": cn_fourier_audit,
            "theta_winding": {
                "observed_shift_rad": theta_shift,
                "expected_shift_rad": expected_shift,
                "error_rad": theta_winding_error,
                "tolerance_rad": theta_tolerance,
            },
            "omega_endpoint_periodicity": omega_endpoint_audit,
            "theta_dot_vs_omega_midpoints": {
                "maximum_absolute_error_rad_s": maximum_kinematic_error,
                "rmse_rad_s": float(np.sqrt(np.mean(theta_omega_error**2))),
            },
            "midpoint_alignment": {
                "policy": "shared_equal_stimulation_interval_midpoints",
                "times_s": midpoint_times.tolist(),
                "force_source": "direct_periodic_fourier_evaluation",
                "cn_source": "direct_periodic_fourier_evaluation",
                "theta_source": "linear_winding_plus_periodic_fourier_residual",
                "omega_source": "periodic_fourier_evaluation",
                "muscle_relationship_source": "ReducedCyclingDynamics.muscle_relationships(theta,omega)",
                "source_slow_state_source": (
                    "degree-specific_piecewise_collocation_polynomial_at_shared_midpoints"
                ),
                "all_array_shapes": list(expected_midpoint_shape),
                "source_slow_state_shape": list(source_slow_states.shape),
            },
            "raw_reduced_muscle_relationship_ranges": {
                name: {
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
                for name, values in raw_gains.items()
            },
            "pulse_width_controls": control_audit,
        },
        "reference_policy": reference_policy,
        "rejection_reasons": reasons,
        "horizons": [],
    }
    if reasons:
        return report

    reference_policy = _reference_policy_audit(
        source_cycle_index=cycle.cycle_index,
        midpoint_times=midpoint_times,
        muscle_names=muscle_names,
        force=force_midpoints,
        force_derivative=force_derivative_midpoints,
        cn=cn_midpoints,
        source_slow_states=source_slow_states,
        force_length=force_length,
        force_velocity=force_velocity,
        passive_force=passive_force,
        source_pulse_widths=source_pulse_widths,
        parameters=parameters,
        pulse_width_tolerance_s=thresholds["pulse_width_bound_tolerance_s"],
    )
    report["reference_policy"] = reference_policy
    report["status"] = adapter_completion_status(
        reasons,
        reference_policy_reproducible=reference_policy["reproducible"],
    )

    recruitment_profile = PeriodicRecruitmentProfile(
        force_profile=force_profile,
        interval_count=cycle.stimulations_per_cycle,
        cn=cn_midpoints,
        force_length_relationship=force_length,
        force_velocity_relationship=force_velocity,
        passive_force_relationship=passive_force,
    )
    report["horizons"] = [
        _horizon_payload(
            rollout_periodic_ding_endurance(
                recruitment_profile,
                initial_slow_states=initial_slow_states,
                muscles=parameters,
                horizon_cycles=horizon,
            ),
            horizon,
            muscle_names,
            parameters,
        )
        for horizon in horizons
    ]
    return report


def write_rho_endurance_rollout_report(output: str | Path, report: dict[str, Any]) -> Path:
    """Atomically write a compact standards-compliant JSON report."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    )
    temporary.replace(output)
    return output
