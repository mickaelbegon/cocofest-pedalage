r"""Auditable adapter from a certified RHO NPZ export to endurance rollouts.

The adapter understands the collocation-node layout written by the cycling RHO
benchmark.  It selects the final certified cycle from metadata, reconstructs
the actual Radau sample times, retains the local collocation polynomials
of ``F`` and ``Cn``, and evaluates reduced muscle geometry at interval
midpoints.  It never treats the dense collocation columns as uniformly spaced.

The periodic-policy assumption is a scientific gate.  A source without
certification metadata, an invalid crank winding, non-periodic ``omega``, ``F``
or ``Cn``, excessive representation residual, or negative reconstructed force produces
a rejected report with measured reasons.  The source transcription is checked
where Radau actually enforces the ODE.  Midpoint inversion is reported as a
separate approximation-quality diagnostic because midpoint states are not NLP
constraint points.  Values are not clipped to make a rollout run.
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
from cocofest.optimization.periodic_collocation_profile import PeriodicCollocationProfile
from cocofest.optimization.recruitment_margin import ding_recruitment_margin


REPORT_SCHEMA = "cocofest-rho-endurance-rollout-v2"
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
DEFAULT_COLLOCATION_FORCE_ODE_TOLERANCE_N_PER_S = 1e-4
# Midpoints are interpolation points, not direct-collocation constraints.  The
# defaults below are engineering acceptance scales for exporting the compact
# policy: 10 us in pulse width and 10 N/s in the vector field.  They are
# intentionally much looser than the exact stage-transcription tolerance and
# are exposed by both the Python API and CLI.
DEFAULT_MIDPOINT_PULSE_WIDTH_ERROR_TOLERANCE_S = 1e-5
DEFAULT_MIDPOINT_FORCE_ODE_TOLERANCE_N_PER_S = 10.0
DEFAULT_MIDPOINT_INVERSE_COVERAGE_MINIMUM = 0.90


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


def select_certified_rho_cycle(
    source: str | Path,
    *,
    cycle_index: int,
    cycle_period: float | None = None,
) -> tuple[SelectedRhoCycle, str]:
    """Load and structurally select one metadata-declared RHO cycle.

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

    if isinstance(cycle_index, bool):
        raise ValueError("cycle_index must be an integer in the declared cycle range.")
    try:
        integer_cycle_index = int(cycle_index)
    except (TypeError, ValueError) as error:
        raise ValueError("cycle_index must be an integer in the declared cycle range.") from error
    if integer_cycle_index != cycle_index:
        raise ValueError("cycle_index must be an integer in the declared cycle range.")
    cycle_index = integer_cycle_index
    if cycle_index < 0 or cycle_index >= cycle_count:
        raise ValueError(
            f"cycle_index={cycle_index} is outside the declared range [0, {cycle_count - 1}]."
        )
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


def select_last_certified_rho_cycle(
    source: str | Path,
    *,
    cycle_period: float | None = None,
) -> tuple[SelectedRhoCycle, str]:
    """Load the final metadata-declared RHO cycle.

    This compatibility wrapper retains the original public behavior.  Use
    :func:`select_certified_rho_cycle` for retrospective anchor selection.
    """

    source = Path(source)
    with np.load(source, allow_pickle=False) as archive:
        if "metadata__json" not in archive.files:
            raise ValueError("RHO archive has no metadata__json entry.")
        metadata = json.loads(str(archive["metadata__json"].item()))
    _, _, cycle_count = _certification(metadata)
    return select_certified_rho_cycle(
        source,
        cycle_index=cycle_count - 1,
        cycle_period=cycle_period,
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
        raise ValueError(
            "A non-negative Fourier force fit requires non-negative source samples; "
            f"minimum={float(np.min(samples)):.17g} N, "
            f"negative_sample_count={int(np.count_nonzero(samples < 0.0))}; "
            "no clipping is used."
        )
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
            "harmonic_count": getattr(profile, "harmonic_count", None),
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


def _lagrange_derivative_weights(nodes: np.ndarray, query: float) -> np.ndarray:
    """Return derivatives of the Lagrange basis at a normalized query time.

    The implementation uses the defining polynomial products directly.  This
    is deliberately degree agnostic: the archive's declared Radau/Legendre
    rule and degree determine ``nodes``, rather than a hard-coded coefficient
    table that could silently disagree with the NLP transcription.
    """

    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 1 or nodes.size < 2 or not np.all(np.isfinite(nodes)):
        raise ValueError("Collocation interpolation nodes must be a finite vector.")
    if np.unique(nodes).size != nodes.size:
        raise ValueError("Collocation interpolation nodes must be distinct.")
    query = float(query)
    weights = np.zeros(nodes.size)
    for index, node in enumerate(nodes):
        denominator = np.prod(node - np.delete(nodes, index))
        numerator_derivative = 0.0
        other_indices = [other for other in range(nodes.size) if other != index]
        for omitted in other_indices:
            factors = [query - nodes[other] for other in other_indices if other != omitted]
            numerator_derivative += float(np.prod(factors))
        weights[index] = numerator_derivative / denominator
    return weights


def _collocation_nodes(cycle: SelectedRhoCycle) -> np.ndarray:
    """Return the shooting node followed by the archive-declared stages."""

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
    return np.concatenate(([0.0], stages))


def _collocation_profile(cycle: SelectedRhoCycle, state_names: Sequence[str]) -> PeriodicCollocationProfile:
    """Retain each interval's actual shooting/stage polynomial without projection."""
    nodes = _collocation_nodes(cycle)
    columns = (cycle.start_column
        + np.arange(cycle.stimulations_per_cycle)[:, None] * cycle.state_columns_per_interval
        + np.arange(nodes.size)[None, :])
    values = np.stack([cycle.states[name][columns] for name in state_names])
    return PeriodicCollocationProfile.from_samples(period=cycle.period, nodes=nodes, values=values)


def _state_midpoints_and_derivatives_from_collocation(
    cycle: SelectedRhoCycle,
    prefix: str,
    muscle_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one state and its time derivative from each local NLP polynomial.

    Both quantities use the *same* degree-specific polynomial through the
    interval shooting node and collocation stages.  The derivative is scaled
    from normalized interval time to seconds.  This is the only reconstruction
    suitable for checking whether the exported pulse width reproduces the
    source collocation equation.
    """

    nodes = _collocation_nodes(cycle)
    value_weights = _lagrange_weights(nodes, 0.5)
    derivative_weights = _lagrange_derivative_weights(nodes, 0.5)
    interval_duration = cycle.period / cycle.stimulations_per_cycle
    midpoints = np.empty((len(muscle_names), cycle.stimulations_per_cycle), dtype=float)
    derivatives = np.empty_like(midpoints)
    for muscle_index, muscle_name in enumerate(muscle_names):
        key = f"{prefix}_{muscle_name}"
        if key not in cycle.states:
            raise ValueError(f"RHO archive is missing state {key!r}.")
        trace = cycle.states[key]
        for interval_index in range(cycle.stimulations_per_cycle):
            base = cycle.start_column + interval_index * cycle.state_columns_per_interval
            columns = base + np.arange(nodes.size)
            local_values = trace[columns]
            midpoints[muscle_index, interval_index] = float(local_values @ value_weights)
            derivatives[muscle_index, interval_index] = float(
                local_values @ derivative_weights / interval_duration
            )
    return midpoints, derivatives


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

    midpoints, _ = _state_midpoints_and_derivatives_from_collocation(
        cycle, prefix, muscle_names
    )
    return midpoints


def _scalar_state_midpoints_from_collocation(
    cycle: SelectedRhoCycle,
    state_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Scalar-state counterpart of the muscle-prefixed Radau evaluator."""

    nodes = _collocation_nodes(cycle)
    value_weights = _lagrange_weights(nodes, 0.5)
    derivative_weights = _lagrange_derivative_weights(nodes, 0.5)
    interval_duration = cycle.period / cycle.stimulations_per_cycle
    if state_name not in cycle.states:
        raise ValueError(f"RHO archive is missing state {state_name!r}.")
    trace = cycle.states[state_name]
    values = np.empty(cycle.stimulations_per_cycle)
    derivatives = np.empty_like(values)
    for interval_index in range(cycle.stimulations_per_cycle):
        base = cycle.start_column + interval_index * cycle.state_columns_per_interval
        columns = base + np.arange(nodes.size)
        local_values = trace[columns]
        values[interval_index] = float(local_values @ value_weights)
        derivatives[interval_index] = float(local_values @ derivative_weights / interval_duration)
    return values, derivatives


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


def _raw_effective_recruitment(capacity: float, pulse_width: float, pd0: float, pdt: float) -> float:
    """Evaluate the NLP formula without silently projecting a near-bound control.

    This private evaluator is used only for transcription-residual audits.  A
    control outside the declared bound tolerance is rejected separately; a
    solver-scale violation inside that tolerance retains its signed effect so
    the audit neither clips nor hides it.
    """

    return float(capacity) * (-math.expm1(-(float(pulse_width) - float(pd0)) / float(pdt)))


def _raw_force_derivative(
    *,
    cn: float,
    force: float,
    capacity: float,
    tau1: float,
    km: float,
    tau2: float,
    pulse_width: float,
    pd0: float,
    pdt: float,
    force_length: float,
    force_velocity: float,
    passive_force: float,
) -> float:
    activation = float(cn) / (float(km) + float(cn))
    relaxation = float(tau1) + float(tau2) * activation
    recruitment = _raw_effective_recruitment(capacity, pulse_width, pd0, pdt)
    gain = float(force_length) * float(force_velocity) + float(passive_force)
    return gain * (recruitment * activation - float(force) / relaxation)


def _collocation_force_ode_audit(
    cycle: SelectedRhoCycle,
    *,
    muscle_names: Sequence[str],
    parameters: Sequence[DingRolloutMuscleParameters],
    reduced: Any,
    source_pulse_widths: np.ndarray,
    tolerance_n_per_s: float,
) -> dict[str, Any]:
    """Check force defects at the Radau stages where the NLP enforces dynamics."""

    nodes = _collocation_nodes(cycle)
    interval_duration = cycle.period / cycle.stimulations_per_cycle
    residuals = np.empty(
        (len(muscle_names), cycle.stimulations_per_cycle, cycle.collocation_degree),
        dtype=float,
    )
    for interval_index in range(cycle.stimulations_per_cycle):
        base = cycle.start_column + interval_index * cycle.state_columns_per_interval
        columns = base + np.arange(nodes.size)
        theta_trace = cycle.states["theta"]
        omega_trace = cycle.states["omega"]
        for stage_offset, normalized_time in enumerate(nodes[1:], start=1):
            derivative_weights = (
                _lagrange_derivative_weights(nodes, float(normalized_time)) / interval_duration
            )
            stage_column = base + stage_offset
            fl, fv, fp = reduced.muscle_relationships(
                float(theta_trace[stage_column]), float(omega_trace[stage_column])
            )
            fl = np.asarray(fl, dtype=float)
            fv = np.asarray(fv, dtype=float)
            fp = np.asarray(fp, dtype=float)
            if not bool(cycle.metadata.get("activate_force_length_relationship", True)):
                fl.fill(1.0)
            if not bool(cycle.metadata.get("activate_force_velocity_relationship", True)):
                fv.fill(1.0)
            if not bool(cycle.metadata.get("activate_passive_force_relationship", True)):
                fp.fill(0.0)
            for muscle_index, (muscle_name, parameter) in enumerate(
                zip(muscle_names, parameters, strict=True)
            ):
                polynomial_derivative = float(
                    cycle.states[f"F_{muscle_name}"][columns] @ derivative_weights
                )
                ode_derivative = _raw_force_derivative(
                    cn=cycle.states[f"Cn_{muscle_name}"][stage_column],
                    force=cycle.states[f"F_{muscle_name}"][stage_column],
                    capacity=cycle.states[f"A_{muscle_name}"][stage_column],
                    tau1=cycle.states[f"Tau1_{muscle_name}"][stage_column],
                    km=cycle.states[f"Km_{muscle_name}"][stage_column],
                    tau2=parameter.tau2,
                    pulse_width=source_pulse_widths[muscle_index, interval_index],
                    pd0=parameter.pd0,
                    pdt=parameter.pdt,
                    force_length=fl[muscle_index],
                    force_velocity=fv[muscle_index],
                    passive_force=fp[muscle_index],
                )
                residuals[muscle_index, interval_index, stage_offset - 1] = (
                    polynomial_derivative - ode_derivative
                )
    per_muscle = {}
    for muscle_index, muscle_name in enumerate(muscle_names):
        values = residuals[muscle_index]
        per_muscle[muscle_name] = {
            "maximum_absolute_error_n_per_s": float(np.max(np.abs(values))),
            "rmse_n_per_s": float(np.sqrt(np.mean(values**2))),
        }
    maximum = float(np.max(np.abs(residuals)))
    return {
        "evaluation_location": "every_archive_declared_collocation_stage",
        "role": "exact_source_NLP_transcription_gate",
        "sample_count": int(residuals.size),
        "maximum_absolute_error_n_per_s": maximum,
        "rmse_n_per_s": float(np.sqrt(np.mean(residuals**2))),
        "tolerance_n_per_s": float(tolerance_n_per_s),
        "passed": maximum <= tolerance_n_per_s,
        "per_muscle": per_muscle,
        "control_bound_projection_applied": False,
    }


def _finite_json_number(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def adapter_completion_status(
    rejection_reasons: Sequence[dict[str, Any]],
    *,
    midpoint_approximation_acceptable: bool | None,
) -> str:
    """Classify adaptation independently of future endurance failures.

    Input/transcription failures take precedence.  Once they are accepted, the
    compact midpoint policy must meet its declared numerical approximation
    tolerances.  Individual bounded-inverse failures at non-enforced midpoints
    remain visible diagnostics but do not, by themselves, fail the adapter.
    Feasibility loss only in later rollout cycles is deliberately not an
    adapter failure; it is the endurance result being measured.
    """

    if rejection_reasons:
        return "rejected"
    if midpoint_approximation_acceptable is not True:
        return "midpoint_approximation_out_of_tolerance"
    return "complete"


def _error_summary(errors: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(errors)
    values = errors[finite]
    return {
        "finite_sample_count": int(values.size),
        "maximum_absolute_error": None if values.size == 0 else float(np.max(np.abs(values))),
        "p95_absolute_error": (
            None if values.size == 0 else float(np.percentile(np.abs(values), 95.0))
        ),
        "rmse": None if values.size == 0 else float(np.sqrt(np.mean(values**2))),
    }


def _error_summary_by_muscle(
    errors: np.ndarray, muscle_names: Sequence[str]
) -> dict[str, Any]:
    errors = np.asarray(errors, dtype=float)
    if errors.ndim != 2 or errors.shape[0] != len(muscle_names):
        raise ValueError("errors must have shape (muscles, samples).")
    return {
        **_error_summary(errors),
        "per_muscle": {
            muscle_name: _error_summary(errors[muscle_index])
            for muscle_index, muscle_name in enumerate(muscle_names)
        },
    }


def _midpoint_policy_fidelity_audit(
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
    midpoint_pulse_width_error_tolerance_s: float,
    midpoint_force_ode_tolerance_n_per_s: float,
    midpoint_inverse_coverage_minimum: float,
    policy_source: str,
    used_for_global_acceptance: bool,
) -> dict[str, Any]:
    """Audit the source policy at non-enforced interval midpoints.

    The raw exported-control ODE residual is defined for every midpoint.  The
    inverse pulse-width error is defined only where the bounded inverse exists;
    its finite coverage and every failure status are retained.  Approximation
    acceptance uses the two declared numerical error tolerances, never a
    requirement that all midpoint inversions be feasible.
    """

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
    normalized_pulse_width_error = np.full(expected_phase_shape, np.nan)
    exported_policy_derivative_error = np.full(expected_phase_shape, np.nan)
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
                normalized_pulse_width_error[muscle_index, interval_index] = (
                    pulse_width_error[muscle_index, interval_index]
                    / (parameter.pulse_width_max - parameter.pd0)
                )
            exported_derivative = _raw_force_derivative(
                cn=cn[muscle_index, interval_index],
                force=force[muscle_index, interval_index],
                capacity=capacity,
                tau1=tau1,
                km=km,
                tau2=parameter.tau2,
                pulse_width=source_pulse_widths[muscle_index, interval_index],
                pd0=parameter.pd0,
                pdt=parameter.pdt,
                force_length=force_length[muscle_index, interval_index],
                force_velocity=force_velocity[muscle_index, interval_index],
                passive_force=passive_force[muscle_index, interval_index],
            )
            exported_policy_derivative_error[muscle_index, interval_index] = (
                exported_derivative - force_derivative[muscle_index, interval_index]
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
    force_ode_summary = _error_summary_by_muscle(
        exported_policy_derivative_error, muscle_names
    )
    pulse_width_summary = _error_summary_by_muscle(pulse_width_error, muscle_names)
    normalized_pulse_width_summary = _error_summary_by_muscle(
        normalized_pulse_width_error, muscle_names
    )
    force_ode_maximum = force_ode_summary["maximum_absolute_error"]
    pulse_width_maximum = pulse_width_summary["maximum_absolute_error"]
    force_ode_passed = (
        force_ode_summary["finite_sample_count"] == muscle_count * interval_count
        and force_ode_maximum is not None
        and force_ode_maximum <= midpoint_force_ode_tolerance_n_per_s
    )
    inverse_coverage = pulse_width_summary["finite_sample_count"] / (
        muscle_count * interval_count
    )
    inverse_coverage_passed = inverse_coverage >= midpoint_inverse_coverage_minimum
    pulse_width_passed = (
        inverse_coverage_passed
        and pulse_width_maximum is not None
        and pulse_width_maximum <= midpoint_pulse_width_error_tolerance_s
    )
    approximation_passed = force_ode_passed and pulse_width_passed
    interval_duration = (
        float(midpoint_times[1] - midpoint_times[0])
        if interval_count > 1
        else 2.0 * float(midpoint_times[0])
    )
    slow_state_ranges = {}
    for state_index, state_name in enumerate(("A", "Tau1", "Km")):
        values = source_slow_states[:, :, state_index]
        slow_state_ranges[state_name] = {
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
    return {
        "evaluated": True,
        "passed": approximation_passed,
        "policy_source": policy_source,
        "acceptance_semantics": (
            "force_ode_error_and_pulse_width_error_and_minimum_inverse_coverage; "
            "not_all_midpoint_inversions_feasible"
        ),
        "source_cycle_index": int(source_cycle_index),
        "sample_count": int(muscle_count * interval_count),
        "failure_count": int(sum(count for status, count in status_counts.items() if status != "ok")),
        "feasible_inversion_count": int(status_counts.get("ok", 0)),
        "all_midpoint_inversions_feasible": first_failure is None,
        "bounded_midpoint_inversion_is_acceptance_gate": False,
        "status_counts": dict(sorted(status_counts.items())),
        "status_counts_by_muscle": {
            name: dict(sorted(counts.items())) for name, counts in per_muscle_counts.items()
        },
        "first_failure": first_failure,
        "maximum_finite_utilization": (
            None if finite_utilization.size == 0 else float(np.max(finite_utilization))
        ),
        "force_derivative_reconstruction_error_n_per_s": _error_summary(derivative_error),
        "exported_policy_force_ode_residual_n_per_s": {
            **force_ode_summary,
            "evaluation_location": "interval_midpoints_not_enforced_by_Radau_NLP",
            "interpretation": (
                "difference_between_local_polynomial_derivative_and_Ding_vector_field; "
                "the separately reported collocation-stage residual is the transcription gate"
            ),
            "control_bound_projection_applied": False,
        },
        "inferred_vs_exported_pulse_width_error_s": {
            **pulse_width_summary,
            "population": "finite_bounded_inverse_results_only",
            "unavailable_sample_count": int(
                muscle_count * interval_count - pulse_width_summary["finite_sample_count"]
            ),
        },
        "inferred_vs_exported_pulse_width_error_normalized_by_available_range": {
            **normalized_pulse_width_summary,
            "normalization": "abs(error)/(pulse_width_max-pd0)_per_muscle",
        },
        "approximation_quality": {
            "passed": approximation_passed,
            "role": (
                "adapted_rollout_policy_export_gate_not_NLP_transcription_gate"
                if used_for_global_acceptance
                else "source_local_midpoint_diagnostic_not_global_acceptance_gate"
            ),
            "used_for_global_acceptance": used_for_global_acceptance,
            "midpoint_force_ode": {
                "maximum_absolute_error_n_per_s": force_ode_maximum,
                "tolerance_n_per_s": float(midpoint_force_ode_tolerance_n_per_s),
                "passed": force_ode_passed,
                "equivalent_one_interval_force_error_tolerance_n": float(
                    midpoint_force_ode_tolerance_n_per_s * interval_duration
                ),
            },
            "inferred_vs_exported_pulse_width": {
                "maximum_absolute_error_s": pulse_width_maximum,
                "tolerance_s": float(midpoint_pulse_width_error_tolerance_s),
                "passed": pulse_width_passed,
                "finite_sample_count": pulse_width_summary["finite_sample_count"],
                "unavailable_sample_count": int(
                    muscle_count * interval_count - pulse_width_summary["finite_sample_count"]
                ),
                "coverage_semantics": (
                    "only_midpoints_with_a_finite_bounded_inverse; "
                    "all_other_statuses_are_retained"
                ),
            },
            "bounded_inverse_coverage": {
                "fraction": float(inverse_coverage),
                "minimum_fraction": float(midpoint_inverse_coverage_minimum),
                "passed": inverse_coverage_passed,
                "ambiguous_sample_count": int(
                    muscle_count * interval_count - pulse_width_summary["finite_sample_count"]
                ),
                "interpretation": (
                    "nonfinite_inverse_samples_are_an_explicit_ambiguity_zone_not_clipped_values"
                ),
            },
        },
        "source_slow_state_ranges": slow_state_ranges,
        "slow_state_source": "degree-specific_piecewise_collocation_polynomial_at_shared_midpoints",
        "force_derivative_source": policy_source,
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


def _cycle_utilization_payload(
    utilization: np.ndarray,
    muscle_names: Sequence[str],
    *,
    saturation_threshold: float = 0.95,
) -> dict[str, Any]:
    """Summarize one ``(phase, muscle)`` utilization matrix without clipping."""

    values = np.asarray(utilization, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(muscle_names):
        raise ValueError("Cycle utilization must have shape (intervals, muscles).")
    finite = np.isfinite(values)
    finite_values = values[finite]
    location = None
    maximum = None
    if finite_values.size:
        masked = np.where(finite, values, -math.inf)
        interval_index, muscle_index = np.unravel_index(np.argmax(masked), values.shape)
        maximum = float(masked[interval_index, muscle_index])
        location = {
            "interval_index": int(interval_index),
            "muscle_index": int(muscle_index),
            "muscle": muscle_names[muscle_index],
        }
    saturated = finite & (values >= saturation_threshold)
    return {
        "maximum_finite_utilization": maximum,
        "critical_location": location,
        "finite_sample_count": int(finite_values.size),
        "saturation_threshold": float(saturation_threshold),
        "saturated_sample_count": int(np.count_nonzero(saturated)),
        "saturated_sample_fraction": (
            None if finite_values.size == 0 else float(np.count_nonzero(saturated) / finite_values.size)
        ),
    }


def _horizon_payload(
    result,
    horizon: int,
    muscle_names: Sequence[str],
    parameters,
    *,
    saturation_threshold: float = 0.95,
) -> dict[str, Any]:
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
        "last_cycle_recruitment": _cycle_utilization_payload(
            result.utilization[-1], muscle_names, saturation_threshold=saturation_threshold
        ),
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
    source_cycle_index: int | None = None,
    cycle_period: float | None = None,
    policy_representation: str = "collocation",
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
    collocation_force_ode_tolerance_n_per_s: float = DEFAULT_COLLOCATION_FORCE_ODE_TOLERANCE_N_PER_S,
    midpoint_pulse_width_error_tolerance_s: float = DEFAULT_MIDPOINT_PULSE_WIDTH_ERROR_TOLERANCE_S,
    midpoint_force_ode_tolerance_n_per_s: float = DEFAULT_MIDPOINT_FORCE_ODE_TOLERANCE_N_PER_S,
    midpoint_inverse_coverage_minimum: float = DEFAULT_MIDPOINT_INVERSE_COVERAGE_MINIMUM,
    saturation_utilization_threshold: float = 0.95,
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
    if policy_representation not in {"collocation", "fourier"}:
        raise ValueError("policy_representation must be 'collocation' or 'fourier'.")
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
        "collocation_force_ode_n_per_s": _positive_float(
            collocation_force_ode_tolerance_n_per_s,
            name="collocation_force_ode_tolerance_n_per_s",
        ),
        "midpoint_pulse_width_error_s": _positive_float(
            midpoint_pulse_width_error_tolerance_s,
            name="midpoint_pulse_width_error_tolerance_s",
        ),
        "midpoint_force_ode_n_per_s": _positive_float(
            midpoint_force_ode_tolerance_n_per_s,
            name="midpoint_force_ode_tolerance_n_per_s",
        ),
    }
    midpoint_inverse_coverage_minimum = float(midpoint_inverse_coverage_minimum)
    if not math.isfinite(midpoint_inverse_coverage_minimum) or not (
        0.0 < midpoint_inverse_coverage_minimum <= 1.0
    ):
        raise ValueError("midpoint_inverse_coverage_minimum must be in (0, 1].")
    thresholds["midpoint_inverse_coverage_minimum"] = midpoint_inverse_coverage_minimum
    saturation_utilization_threshold = float(saturation_utilization_threshold)
    if not math.isfinite(saturation_utilization_threshold) or not (
        0.0 < saturation_utilization_threshold <= 1.0
    ):
        raise ValueError("saturation_utilization_threshold must be in (0, 1].")
    if source_cycle_index is None:
        cycle, period_basis = select_last_certified_rho_cycle(source, cycle_period=cycle_period)
    else:
        cycle, period_basis = select_certified_rho_cycle(
            source,
            cycle_index=source_cycle_index,
            cycle_period=cycle_period,
        )
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

    if policy_representation == "collocation":
        force_profile = _collocation_profile(cycle, [f"F_{name}" for name in muscle_names])
        cn_profile = _collocation_profile(cycle, [f"Cn_{name}" for name in muscle_names])
    else:
        force_profile = fit_nonnegative_periodic_fourier_force_profile(
            force_samples, cycle.sample_times, period=cycle.period,
            root_harmonic_count=force_harmonics,
        )
        cn_profile = _fit_periodic_at_times(
            cn_samples, cycle.sample_times, period=cycle.period, harmonic_count=cn_harmonics,
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
            "fit_space": "source_collocation_values" if policy_representation == "collocation" else "sqrt_force",
            "root_harmonic_count": force_harmonics if policy_representation == "fourier" else None,
            "nonnegative_construction": ("unmodified_polynomial_extrema_audit"
                if policy_representation == "collocation" else "exact_complex_fourier_convolution_of_squared_root_series"),
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
                f"{kind}_{policy_representation}_rmse_too_large",
                f"{kind} {policy_representation} relative RMSE exceeds tolerance.",
                maximum_relative_rmse=maximum_rmse,
                tolerance=thresholds["fourier_relative_rmse"],
            )
        if maximum_error > thresholds["fourier_relative_maximum"]:
            reject(
                f"{kind}_{policy_representation}_maximum_error_too_large",
                f"{kind} {policy_representation} maximum relative error exceeds tolerance.",
                maximum_relative_error=maximum_error,
                tolerance=thresholds["fourier_relative_maximum"],
            )
    minimum_dense_force = float(np.min(dense_force))
    if minimum_dense_force < -thresholds["negative_force_tolerance_n"]:
        reject(
            f"negative_{policy_representation}_force",
            "The force reconstruction is negative; no clipping is permitted.",
            minimum_reconstructed_force_n=minimum_dense_force,
            tolerance_n=thresholds["negative_force_tolerance_n"],
        )
    if float(np.min(dense_cn)) <= 0.0:
        reject(
            f"non_positive_{policy_representation}_cn",
            "The Cn reconstruction leaves the positive activation domain.",
            minimum_reconstructed_cn=float(np.min(dense_cn)),
        )
    positivity = force_profile.force_positivity_certificate(tolerance=thresholds["negative_force_tolerance_n"])
    force_fourier_audit["continuous_extrema"] = {
        "minimum": positivity.minimum_force.tolist(),
        "minimum_time_s": positivity.minimum_time.tolist(),
        "certified_nonnegative": positivity.certified_nonnegative.tolist(),
        "tolerance_n": positivity.tolerance,
        "certification_semantics": "numerical_extrema_audit_not_interval_arithmetic_proof",
        "method": ("recursive_derivative_root_isolation_on_each_closed_polynomial_interval"
            if policy_representation == "collocation" else "stationary_points_of_trigonometric_polynomial"),
    }
    if not np.all(positivity.certified_nonnegative):
        reject("negative_continuous_force", "The continuous force interpolant is not nonnegative.",
            minimum_force_n=float(np.min(positivity.minimum_force)))
    if policy_representation == "collocation":
        for kind, profile, audit, samples, tolerance in (
            ("force", force_profile, force_fourier_audit, force_samples, thresholds["force_seam_absolute_n"]),
            ("cn", cn_profile, cn_fourier_audit, cn_samples, thresholds["cn_seam_absolute"]),
        ):
            audit["seams"] = profile.seam_audit()
            jumps = np.asarray(audit["seams"]["maximum_absolute_value_jump"])
            if np.any((jumps > tolerance) & (jumps / _signal_scales(samples) > thresholds["periodicity_relative"])):
                reject(f"discontinuous_{kind}_collocation_policy", "Polynomial value jumps exceed declared seam tolerances.",
                    maximum_absolute_jump=float(np.max(jumps)))
        cn_minima = cn_profile.force_positivity_certificate(tolerance=0.0).minimum_force
        cn_fourier_audit["continuous_minimum"] = cn_minima.tolist()
        if np.any(cn_minima <= 0.0):
            reject("non_positive_continuous_cn", "The continuous Cn interpolant leaves the positive domain.",
                minimum_cn=float(np.min(cn_minima)))

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
    if policy_representation == "collocation":
        theta_profile = _collocation_profile(cycle, ("theta",))
        residual_coefficients = theta_profile.coefficients.copy()
        residual_coefficients[:, :, 0] -= expected_shift * np.arange(cycle.stimulations_per_cycle) / cycle.stimulations_per_cycle
        residual_coefficients[:, :, 1] -= expected_shift / cycle.stimulations_per_cycle
        theta_residual_profile = PeriodicCollocationProfile(cycle.period, residual_coefficients)
        omega_profile = _collocation_profile(cycle, ("omega",))
    else:
        theta_residual_profile = _fit_periodic_at_times(
            theta_residual_samples, cycle.sample_times, period=cycle.period, harmonic_count=kinematic_harmonics,
        )
        omega_profile = _fit_periodic_at_times(
            omega_samples, cycle.sample_times, period=cycle.period, harmonic_count=kinematic_harmonics,
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
    # Future rollouts use the explicitly audited periodic profiles below.  The
    # source-policy gate must instead interrogate exactly the local state
    # polynomial used by the direct-collocation NLP.
    force_midpoints = force_profile.evaluate(midpoint_times)
    force_derivative_midpoints = force_profile.derivative(midpoint_times)
    cn_midpoints = cn_profile.evaluate(midpoint_times)
    source_force_midpoints, source_force_derivative_midpoints = (
        _state_midpoints_and_derivatives_from_collocation(cycle, "F", muscle_names)
    )
    source_cn_midpoints = _state_midpoints_from_collocation(cycle, "Cn", muscle_names)
    source_theta_midpoints, source_theta_derivative_midpoints = (
        _scalar_state_midpoints_from_collocation(cycle, "theta")
    )
    source_omega_midpoints, _ = _scalar_state_midpoints_from_collocation(cycle, "omega")
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
    source_theta_omega_error = source_theta_derivative_midpoints - source_omega_midpoints
    source_maximum_kinematic_error = float(np.max(np.abs(source_theta_omega_error)))
    source_midpoint_kinematic_diagnostic_passed = (
        source_maximum_kinematic_error <= thresholds["kinematic_consistency_rad_s"]
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
    source_force_length = np.empty_like(force_length)
    source_force_velocity = np.empty_like(force_velocity)
    source_passive_force = np.empty_like(passive_force)
    for index, (theta_value, omega_value) in enumerate(
        zip(source_theta_midpoints, source_omega_midpoints, strict=True)
    ):
        fl, fv, fp = reduced.muscle_relationships(float(theta_value), float(omega_value))
        source_force_length[:, index] = np.asarray(fl, dtype=float)
        source_force_velocity[:, index] = np.asarray(fv, dtype=float)
        source_passive_force[:, index] = np.asarray(fp, dtype=float)
    if not bool(cycle.metadata.get("activate_force_length_relationship", True)):
        source_force_length.fill(1.0)
    if not bool(cycle.metadata.get("activate_force_velocity_relationship", True)):
        source_force_velocity.fill(1.0)
    if not bool(cycle.metadata.get("activate_passive_force_relationship", True)):
        source_passive_force.fill(0.0)
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
        pw_tolerance = thresholds["pulse_width_bound_tolerance_s"]
        below_minimum = values < parameter.pd0
        above_maximum = values > parameter.pulse_width_max
        control_audit[name] = {
            "minimum_s": float(np.min(values)),
            "maximum_s": float(np.max(values)),
            "nominal_below_pd0_count": int(np.count_nonzero(below_minimum)),
            "nominal_above_maximum_count": int(np.count_nonzero(above_maximum)),
            "maximum_lower_bound_violation_s": float(
                max(0.0, parameter.pd0 - float(np.min(values)))
            ),
            "maximum_upper_bound_violation_s": float(
                max(0.0, float(np.max(values)) - parameter.pulse_width_max)
            ),
            "bound_tolerance_s": pw_tolerance,
            "bound_projection_applied": False,
        }
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

    collocation_force_ode_audit = _collocation_force_ode_audit(
        cycle,
        muscle_names=muscle_names,
        parameters=parameters,
        reduced=reduced,
        source_pulse_widths=source_pulse_widths,
        tolerance_n_per_s=thresholds["collocation_force_ode_n_per_s"],
    )
    if not collocation_force_ode_audit["passed"]:
        reject(
            "source_collocation_force_ode_residual_too_large",
            "The exported force polynomial does not reproduce the Ding ODE at its enforced collocation stages.",
            maximum_absolute_error_n_per_s=collocation_force_ode_audit[
                "maximum_absolute_error_n_per_s"
            ],
            tolerance_n_per_s=thresholds["collocation_force_ode_n_per_s"],
        )

    adapted_policy_fidelity: dict[str, Any] = {
        "evaluated": False,
        "passed": None,
        "reason": "input_adaptation_rejected",
    }
    reconstruction_errors = {
        name: {
            "maximum_absolute_error": float(np.max(np.abs(actual - reference))),
            "rmse": float(np.sqrt(np.mean((actual - reference) ** 2))),
        }
        for name, actual, reference in (
            ("force_n", force_midpoints, source_force_midpoints),
            ("force_derivative_n_per_s", force_derivative_midpoints, source_force_derivative_midpoints),
            ("cn", cn_midpoints, source_cn_midpoints),
            ("theta_rad", theta_midpoints, source_theta_midpoints),
            ("omega_rad_per_s", omega_midpoints, source_omega_midpoints),
            ("force_length", force_length, source_force_length),
            ("force_velocity", force_velocity, source_force_velocity),
            ("passive_force", passive_force, source_passive_force),
        )
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": adapter_completion_status(
            reasons,
            midpoint_approximation_acceptable=adapted_policy_fidelity["passed"],
        ),
        "source": _file_stamp(source),
        "source_ocp_context": {
            "signed_crank_torque_nm": cycle.metadata.get(
                "signed_crank_torque_nm", cycle.metadata.get("constant_crank_torque")
            ),
            "mechanical_formulation": cycle.metadata.get("mechanical_formulation"),
            "formulation": cycle.metadata.get("formulation"),
        },
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
            "policy_representation": policy_representation,
            "force_sqrt_harmonics": force_harmonics if policy_representation == "fourier" else None,
            "force_resulting_harmonics": getattr(force_profile, "harmonic_count", None),
            "cn_harmonics": cn_harmonics if policy_representation == "fourier" else None,
            "kinematic_harmonics": kinematic_harmonics if policy_representation == "fourier" else None,
            "force_polynomial_coefficient_shape": (list(force_profile.coefficients.shape)
                if policy_representation == "collocation" else None),
            "thresholds": thresholds,
            "saturation_utilization_threshold": saturation_utilization_threshold,
        },
        "muscle_names": list(muscle_names),
        "model_parameters": parameter_report,
        "audits": {
            "policy_vs_source_collocation_midpoints": {
                "role": "representation_fidelity_diagnostic; independent_Ding_ODE_gate_also_required",
                "clipping_applied": False,
                "errors": reconstruction_errors,
            },
            "force_endpoint_periodicity": force_endpoint_audit,
            "cn_endpoint_periodicity": cn_endpoint_audit,
            "force_policy": force_fourier_audit,
            "cn_policy": cn_fourier_audit,
            **({"force_fourier": force_fourier_audit, "cn_fourier": cn_fourier_audit}
                if policy_representation == "fourier" else {}),
            "theta_winding": {
                "observed_shift_rad": theta_shift,
                "expected_shift_rad": expected_shift,
                "error_rad": theta_winding_error,
                "tolerance_rad": theta_tolerance,
            },
            "omega_endpoint_periodicity": omega_endpoint_audit,
            **({"theta_residual_polynomial_seams": theta_residual_profile.seam_audit(),
                "omega_polynomial_seams": omega_profile.seam_audit()}
                if policy_representation == "collocation" else {}),
            "theta_dot_vs_omega_midpoints": {
                "maximum_absolute_error_rad_s": maximum_kinematic_error,
                "rmse_rad_s": float(np.sqrt(np.mean(theta_omega_error**2))),
            },
            "source_collocation_theta_dot_vs_omega_midpoints": {
                "maximum_absolute_error_rad_s": source_maximum_kinematic_error,
                "rmse_rad_s": float(np.sqrt(np.mean(source_theta_omega_error**2))),
                "tolerance_rad_s": thresholds["kinematic_consistency_rad_s"],
                "passed": source_midpoint_kinematic_diagnostic_passed,
                "role": "diagnostic_at_non_enforced_midpoints_not_transcription_gate",
            },
            "midpoint_alignment": {
                "policy": "shared_equal_stimulation_interval_midpoints",
                "times_s": midpoint_times.tolist(),
                "force_source": f"direct_periodic_{policy_representation}_evaluation",
                "cn_source": f"direct_periodic_{policy_representation}_evaluation",
                "theta_source": f"linear_winding_plus_periodic_{policy_representation}_residual",
                "omega_source": f"periodic_{policy_representation}_evaluation",
                "muscle_relationship_source": "ReducedCyclingDynamics.muscle_relationships(theta,omega)",
                "source_slow_state_source": (
                    "degree-specific_piecewise_collocation_polynomial_at_shared_midpoints"
                ),
                "reference_policy_source": (
                    f"adapted_periodic_{policy_representation}_F,F_dot,Cn_and_geometry; "
                    "source_A,Tau1,Km_from_local_collocation_polynomials"
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
            "source_transcription": {
                "passed": collocation_force_ode_audit["passed"],
                "role": "exact_gate_at_NLP_enforced_Radau_stages",
                "force_ode": collocation_force_ode_audit,
            },
        },
        "adapted_policy_fidelity": adapted_policy_fidelity,
        "reference_policy": {
            "deprecated_alias_of": "adapted_policy_fidelity",
            "schema_note": "use adapted_policy_fidelity in schema v2",
        },
        "rejection_reasons": reasons,
        "horizons": [],
        "rollout_outcome": {"status": "not_evaluated", "reason": "input_adaptation_rejected"},
    }
    if reasons:
        return report

    source_midpoint_interpolation = _midpoint_policy_fidelity_audit(
        source_cycle_index=cycle.cycle_index,
        midpoint_times=midpoint_times,
        muscle_names=muscle_names,
        force=source_force_midpoints,
        force_derivative=source_force_derivative_midpoints,
        cn=source_cn_midpoints,
        source_slow_states=source_slow_states,
        force_length=source_force_length,
        force_velocity=source_force_velocity,
        passive_force=source_passive_force,
        source_pulse_widths=source_pulse_widths,
        parameters=parameters,
        pulse_width_tolerance_s=thresholds["pulse_width_bound_tolerance_s"],
        midpoint_pulse_width_error_tolerance_s=thresholds[
            "midpoint_pulse_width_error_s"
        ],
        midpoint_force_ode_tolerance_n_per_s=thresholds[
            "midpoint_force_ode_n_per_s"
        ],
        midpoint_inverse_coverage_minimum=thresholds[
            "midpoint_inverse_coverage_minimum"
        ],
        policy_source="degree-specific_local_collocation_polynomial_at_midpoints",
        used_for_global_acceptance=False,
    )
    report["audits"]["source_midpoint_interpolation"] = source_midpoint_interpolation

    adapted_policy_fidelity = _midpoint_policy_fidelity_audit(
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
        midpoint_pulse_width_error_tolerance_s=thresholds[
            "midpoint_pulse_width_error_s"
        ],
        midpoint_force_ode_tolerance_n_per_s=thresholds[
            "midpoint_force_ode_n_per_s"
        ],
        midpoint_inverse_coverage_minimum=thresholds[
            "midpoint_inverse_coverage_minimum"
        ],
        policy_source=f"periodic_{policy_representation}_policy_used_by_endurance_rollout",
        used_for_global_acceptance=True,
    )
    report["adapted_policy_fidelity"] = adapted_policy_fidelity
    report["status"] = adapter_completion_status(
        reasons,
        midpoint_approximation_acceptable=adapted_policy_fidelity["passed"],
    )

    recruitment_profile = PeriodicRecruitmentProfile(
        force_profile=force_profile,
        interval_count=cycle.stimulations_per_cycle,
        cn=cn_midpoints,
        force_length_relationship=force_length,
        force_velocity_relationship=force_velocity,
        passive_force_relationship=passive_force,
    )
    if adapted_policy_fidelity["passed"]:
        from cocofest.optimization.endurance_rollout_objective import (
            RolloutObjectiveLayout, pack_rollout_objective_parameters,
        )

        layout = RolloutObjectiveLayout(len(muscle_names), cycle.stimulations_per_cycle, horizons[0])
        packed = pack_rollout_objective_parameters(recruitment_profile, parameters, layout)
        report["rollout_objective_profile"] = {
            "schema": "cocofest-rollout-objective-profile-v1",
            "muscle_count": layout.muscle_count,
            "interval_count": layout.interval_count,
            "parameter_size": layout.parameter_size,
            "packing": "pack_rollout_objective_parameters; field-major then muscle-major C order",
            "parameters": packed.tolist(),
            "initial_slow_states": initial_slow_states.tolist(),
            "source_policy_gate_passed": True,
            "certification_scope": "source_cycle_representation_fidelity_only; future_feasibility_not_certified",
            "horizon_independent": True,
        }
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
            saturation_threshold=saturation_utilization_threshold,
        )
        for horizon in horizons
    ]
    all_feasible = all(row["feasible"] for row in report["horizons"])
    report["rollout_outcome"] = {
        "status": "feasible" if all_feasible else "infeasible",
        "all_requested_horizons_feasible": all_feasible,
        "first_failure": next((row["first_failure"] for row in report["horizons"] if row["first_failure"]), None),
        "interpretation": "frozen_policy_model_outcome; distinct_from_source_fidelity_and_clinical_endurance",
    }
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
