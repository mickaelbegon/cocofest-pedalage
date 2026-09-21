"""Audit fixed-PW Ding collocation with a prescribed Radau-5 mechanical trace.

This is deliberately *not* an OCP: the pulse widths and the mechanical
trajectory are both frozen from a successful Radau-5 one-cycle seed.  It
therefore separates the Ding transcription error from changes to the motion
or from optimizer decisions.  The prescribed mechanics are a degree-5
polynomial on each stimulation interval, reconstructed from the seed's
Radau-5 theta/omega stages; force-length, force-velocity and passive-force
relations are re-evaluated at every integration stage.

The independent reference is DOP853.  Results quantify a one-cycle numerical
error only; they do not establish endurance or OCP feasibility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.dynamics.reduced_cycling import ReducedCyclingDynamics


STATE_NAMES = ("Cn", "F", "A", "Tau1", "Km")
R0_KM_RELATIONSHIP = 1.04


def radau_nodes(degree: int) -> np.ndarray:
    """Return the Radau-right collocation nodes, including the initial zero."""

    if degree < 1:
        raise ValueError("degree must be positive")
    from casadi import collocation_points

    return np.asarray([0.0, *collocation_points(degree, "radau")], dtype=float)


def collocation_coefficients(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return derivative and endpoint coefficients for Lagrange nodes."""

    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 1 or nodes.size < 2 or nodes[0] != 0.0:
        raise ValueError("nodes must be a one-dimensional grid starting at zero")
    if np.any(np.diff(nodes) <= 0.0):
        raise ValueError("collocation nodes must be strictly increasing")
    count = nodes.size
    derivative = np.empty((count, count - 1), dtype=float)
    endpoint = np.empty(count, dtype=float)
    for index, node in enumerate(nodes):
        polynomial = np.poly1d([1.0])
        denominator = 1.0
        for other_index, other_node in enumerate(nodes):
            if other_index == index:
                continue
            polynomial *= np.poly1d([1.0, -other_node])
            denominator *= node - other_node
        polynomial /= denominator
        derivative[index] = np.polyder(polynomial)(nodes[1:])
        endpoint[index] = polynomial(1.0)
    return derivative, endpoint


def periodic_node_history_amplitude(
    *, interval_s: float, tauc: float, km_rest: float, truncation: int
) -> float:
    """Reproduce periodic-node's truncated post-stimulation history ``H0``."""

    if interval_s <= 0.0 or tauc <= 0.0 or truncation < 1:
        raise ValueError("interval_s, tauc and truncation must be positive")
    decay = float(np.exp(-interval_s / tauc))
    if truncation == 1:
        return 1.0
    increment = 1.0 + (km_rest + R0_KM_RELATIONSHIP - 1.0) * decay
    return float(decay ** (truncation - 1) + increment * sum(decay**age for age in range(truncation - 1)))


def radau_collocation_step(rhs, state: np.ndarray, *, duration: float, degree: int,
                            stage_guess: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """One implicit Radau step for a fixed smooth RHS.

    The routine intentionally has no NLP/optimizer dependency.  It is the
    controlled numerical experiment used by this validation script.
    """

    state = np.asarray(state, dtype=float)
    nodes = radau_nodes(degree)
    derivative, endpoint = collocation_coefficients(nodes)
    if stage_guess is None:
        stage_guess = np.repeat(state[None, :], degree, axis=0)
    stage_guess = np.asarray(stage_guess, dtype=float)
    if stage_guess.shape != (degree, state.size):
        raise ValueError("stage_guess must have shape (degree, state dimension)")

    def residual(flat_stages):
        stages = flat_stages.reshape((degree, state.size))
        values = np.vstack((state, stages))
        defects = [
            derivative[:, stage_index] @ values - duration * rhs(nodes[stage_index + 1] * duration, stages[stage_index])
            for stage_index in range(degree)
        ]
        return np.concatenate(defects)

    solved = root(residual, stage_guess.ravel(), method="hybr", options={"xtol": 1e-11})
    maximum_residual = float(np.max(np.abs(residual(solved.x))))
    if not solved.success or not np.isfinite(maximum_residual) or maximum_residual > 2e-9:
        raise RuntimeError(
            f"Radau-{degree} fixed-input step failed: {solved.message}; "
            f"maximum residual={maximum_residual:.3e}."
        )
    stages = solved.x.reshape((degree, state.size))
    return endpoint @ np.vstack((state, stages)), stages, int(solved.nfev)


def _seed_metadata(seed: np.lib.npyio.NpzFile) -> dict:
    if "metadata__json" not in seed.files:
        raise ValueError("Seed has no metadata__json; use a common periodic initial solution.")
    return json.loads(str(seed["metadata__json"]))


def _source_layout(seed: np.lib.npyio.NpzFile, metadata: dict) -> tuple[tuple[str, ...], int, int, int]:
    # NPZ controls preserve the OCP muscle order whereas JSON object order is
    # not a physical contract (and can be alphabetized by other tools).
    control_prefix = "controls__last_pulse_width_"
    muscles = tuple(key.removeprefix(control_prefix) for key in seed.files if key.startswith(control_prefix))
    if not muscles:
        raise ValueError("Seed has no last-pulse-width controls.")
    if set(muscles) != set(metadata["configured_muscle_parameters"]):
        raise ValueError("Seed control muscles and configured muscle parameters differ.")
    controls = int(metadata["stimulations_per_cycle"])
    first = np.asarray(seed[f"states__Cn_{muscles[0]}"], dtype=float).reshape(-1)
    if controls < 1 or (first.size - 1) % controls:
        raise ValueError("Seed state grid is incompatible with its stimulation count.")
    stride = (first.size - 1) // controls
    degree = int(metadata.get("producer_collocation_degree") or stride - 1)
    if stride != degree + 1:
        raise ValueError(
            "Expected a direct Radau state trace with one shooting node plus "
            "degree collocation nodes per interval."
        )
    return muscles, controls, degree, stride


def _interval_polynomials(values: np.ndarray, *, interval: int, stride: int,
                          nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Polynomial coefficients for theta and omega on one source interval."""

    begin = interval * stride
    local_values = values[:, begin : begin + nodes.size]
    if local_values.shape[1] != nodes.size:
        raise ValueError("Source trace ended before the requested interval.")
    return tuple(np.polyfit(nodes, row, deg=nodes.size - 1) for row in local_values)


def _mechanical_gain_function(profile: ReducedCyclingDynamics, theta_polynomial, omega_polynomial,
                              *, use_force_length: bool, use_force_velocity: bool,
                              use_passive_force: bool, muscle_index: int):
    def gain(local_time: float) -> float:
        tau = float(np.clip(local_time, 0.0, 1.0))
        theta = float(np.polyval(theta_polynomial, tau))
        omega = float(np.polyval(omega_polynomial, tau))
        force_length, force_velocity, passive_force = profile.muscle_relationships(theta, omega)
        value = (
            (float(force_length[muscle_index]) if use_force_length else 1.0)
            * (float(force_velocity[muscle_index]) if use_force_velocity else 1.0)
            + (float(passive_force[muscle_index]) if use_passive_force else 0.0)
        )
        if not np.isfinite(value) or value <= 0.0:
            raise RuntimeError(f"Non-positive prescribed mechanical gain {value}.")
        return value

    return gain


def _ding_rhs(*, parameters: dict, pulse_width: float, history_amplitude: float, gain):
    tauc = float(parameters["tauc"])
    tau2 = float(parameters["tau2"])
    pd0 = float(parameters["pd0"])
    pdt = float(parameters["pdt"])
    a_rest = float(parameters["a_scale"])
    tau1_rest = float(parameters["tau1_rest"])
    km_rest = float(parameters["km_rest"])
    tau_fat = float(parameters["tau_fat"])
    alpha_a = float(parameters["alpha_a"])
    alpha_tau1 = float(parameters["alpha_tau1"])
    alpha_km = float(parameters["alpha_km"])
    recruitment_fraction = 1.0 - np.exp(-(float(pulse_width) - pd0) / pdt)

    def rhs(local_time: float, state: np.ndarray) -> np.ndarray:
        cn, force, capacity, tau1, km = state
        if capacity <= 0.0 or tau1 <= 0.0 or km + cn <= 0.0:
            # A large finite value lets the nonlinear solve reject an invalid
            # stage without hiding it as an overflow or a NaN.
            return np.full(5, 1e15)
        calcium_history = history_amplitude * np.exp(-local_time / tauc)
        activation = cn / (km + cn)
        relaxation = tau1 + tau2 * activation
        return np.array(
            [
                (calcium_history - cn) / tauc,
                gain(local_time) * (capacity * recruitment_fraction * activation - force / relaxation),
                -(capacity - a_rest) / tau_fat + alpha_a * force,
                -(tau1 - tau1_rest) / tau_fat + alpha_tau1 * force,
                -(km - km_rest) / tau_fat + alpha_km * force,
            ],
            dtype=float,
        )

    return rhs


def _error_summary(errors: np.ndarray, parameters_by_muscle: dict, muscles: tuple[str, ...]) -> dict:
    scale_keys = (None, "Fmax", "a_scale", "tau1_rest", "km_rest")
    result = {}
    for muscle_index, muscle in enumerate(muscles):
        parameters = parameters_by_muscle[muscle]
        result[muscle] = {}
        for state_index, state_name in enumerate(STATE_NAMES):
            scale = 1.0 if scale_keys[state_index] is None else float(parameters[scale_keys[state_index]])
            values = np.abs(errors[:, muscle_index, state_index])
            result[muscle][state_name] = {
                "maximum_absolute": float(np.max(values)),
                "final_absolute": float(values[-1]),
                "maximum_scaled": float(np.max(values) / scale),
                "scale": scale,
            }
    return result


def validate(seed_path: Path, reduced_profile: Path, *, degrees: tuple[int, ...] = (3, 5),
             dop_rtol: float = 1e-11, dop_atol: float = 1e-13) -> dict:
    """Run the fixed-PW, fixed-mechanics Radau/DOP853 comparison."""

    with np.load(seed_path, allow_pickle=False) as seed:
        metadata = _seed_metadata(seed)
        muscles, interval_count, source_degree, stride = _source_layout(seed, metadata)
        parameters_by_muscle = metadata["configured_muscle_parameters"]
        source_nodes = radau_nodes(source_degree)
        state_traces = np.stack(
            [np.stack([np.asarray(seed[f"states__{state}_{muscle}"], dtype=float).reshape(-1)
                       for state in STATE_NAMES]) for muscle in muscles]
        )
        theta_omega = np.stack(
            [np.asarray(seed["states__theta"], dtype=float).reshape(-1),
             np.asarray(seed["states__omega"], dtype=float).reshape(-1)]
        )
        pulse_widths = np.stack(
            [np.asarray(seed[f"controls__last_pulse_width_{muscle}"], dtype=float).reshape(-1)
             for muscle in muscles]
        )
    if pulse_widths.shape != (len(muscles), interval_count):
        raise ValueError("Seed controls do not provide one pulse width per muscle and interval.")
    profile = ReducedCyclingDynamics.load(reduced_profile)
    if tuple(profile.muscle_names) != muscles:
        raise ValueError(f"Reduced profile muscle order {profile.muscle_names} differs from seed {muscles}.")
    if profile.muscle_geometry is None:
        raise ValueError("Reduced profile has no muscle geometry.")
    duration = 1.0 / interval_count
    truncation = int(metadata["ding_sum_stim_truncation"])
    flags = {key: bool(metadata.get(key, True)) for key in (
        "activate_force_length_relationship", "activate_force_velocity_relationship",
        "activate_passive_force_relationship",
    )}
    current_reference = state_traces[:, :, 0].copy()
    current_by_degree = {degree: current_reference.copy() for degree in degrees}
    source_endpoint = state_traces[:, :, np.arange(interval_count + 1) * stride].transpose(2, 0, 1)
    errors = {degree: [] for degree in degrees}
    source_reproduction_errors = []
    gain_samples = [[] for _ in muscles]
    function_evaluations = {degree: 0 for degree in degrees}

    for interval in range(interval_count):
        theta_polynomial, omega_polynomial = _interval_polynomials(
            theta_omega, interval=interval, stride=stride, nodes=source_nodes
        )
        gains = [
            _mechanical_gain_function(
                profile, theta_polynomial, omega_polynomial,
                use_force_length=flags["activate_force_length_relationship"],
                use_force_velocity=flags["activate_force_velocity_relationship"],
                use_passive_force=flags["activate_passive_force_relationship"], muscle_index=index,
            )
            for index in range(len(muscles))
        ]
        for index, gain in enumerate(gains):
            gain_samples[index].extend(gain(float(time)) for time in np.linspace(0.0, duration, 31))
        references = []
        for muscle_index, muscle in enumerate(muscles):
            parameters = parameters_by_muscle[muscle]
            amplitude = periodic_node_history_amplitude(
                interval_s=duration, tauc=float(parameters["tauc"]),
                km_rest=float(parameters["km_rest"]), truncation=truncation,
            )
            rhs = _ding_rhs(parameters=parameters, pulse_width=pulse_widths[muscle_index, interval],
                             history_amplitude=amplitude, gain=gains[muscle_index])
            solution = solve_ivp(rhs, (0.0, duration), current_reference[muscle_index],
                                 method="DOP853", rtol=dop_rtol, atol=dop_atol)
            if not solution.success:
                raise RuntimeError(f"DOP853 failed for {muscle}, interval {interval}: {solution.message}")
            references.append(solution.y[:, -1])
        current_reference = np.asarray(references)
        for degree in degrees:
            next_states = []
            for muscle_index, muscle in enumerate(muscles):
                parameters = parameters_by_muscle[muscle]
                amplitude = periodic_node_history_amplitude(
                    interval_s=duration, tauc=float(parameters["tauc"]),
                    km_rest=float(parameters["km_rest"]), truncation=truncation,
                )
                rhs = _ding_rhs(parameters=parameters, pulse_width=pulse_widths[muscle_index, interval],
                                 history_amplitude=amplitude, gain=gains[muscle_index])
                # Seed stages give Radau-5 its exact primal initialisation.
                # Other degrees use a physically timed interpolation of those stages.
                source_stage = state_traces[muscle_index, :, interval * stride : interval * stride + source_nodes.size]
                stage_guess = np.vstack(
                    [
                        np.polyval(
                            np.polyfit(source_nodes, row, source_degree),
                            radau_nodes(degree)[1:],
                        )
                        for row in source_stage
                    ]
                ).T
                endpoint, _, evaluations = radau_collocation_step(
                    rhs, current_by_degree[degree][muscle_index], duration=duration,
                    degree=degree, stage_guess=stage_guess,
                )
                next_states.append(endpoint)
                function_evaluations[degree] += evaluations
            current_by_degree[degree] = np.asarray(next_states)
            errors[degree].append(current_by_degree[degree] - current_reference)
        source_reproduction_errors.append(current_by_degree.get(source_degree, current_reference) - source_endpoint[interval + 1])

    return {
        "schema": "fixed-pw-prescribed-radau5-mechanics-v1",
        "scope": (
            "one_cycle_fixed_PW_and_prescribed_mechanics; isolates_Ding_transcription; "
            "not_an_OCP_or_endurance_certificate"
        ),
        "seed": str(seed_path.resolve()), "reduced_profile": str(reduced_profile.resolve()),
        "interval_count": interval_count, "interval_duration_s": duration,
        "source_collocation_degree": source_degree, "tested_collocation_degrees": list(degrees),
        "mechanics": {
            "source": "piecewise_degree5_Radau_stage_polynomial_from_seed",
            "force_length_enabled": flags["activate_force_length_relationship"],
            "force_velocity_enabled": flags["activate_force_velocity_relationship"],
            "passive_force_enabled": flags["activate_passive_force_relationship"],
            "gain_minimum_by_muscle": {muscle: float(np.min(gain_samples[index])) for index, muscle in enumerate(muscles)},
            "gain_maximum_by_muscle": {muscle: float(np.max(gain_samples[index])) for index, muscle in enumerate(muscles)},
        },
        "dop853": {"rtol": dop_rtol, "atol": dop_atol},
        "results": {
            str(degree): {
                "function_evaluations": function_evaluations[degree],
                "endpoint_error_vs_DOP853": _error_summary(np.asarray(errors[degree]), parameters_by_muscle, muscles),
            }
            for degree in degrees
        },
        # This is diagnostic only: the stored common seed can have been
        # captured during a continuation/warm-up pathway and is not used as a
        # numerical accuracy pass criterion.  The Radau/DOP comparison above
        # is the controlled experiment.
        "raw_seed_state_recurrence_mismatch": (
            _error_summary(np.asarray(source_reproduction_errors), parameters_by_muscle, muscles)
            if source_degree in degrees else None
        ),
        "scientific_limitations": [
            "The mechanical trace is prescribed, so this omits feedback from Ding force error to theta and omega.",
            "The trace is reconstructed from a finite Radau-5 solution; it is not an independently measured motion.",
            "Endpoint error alone does not certify path-constraint, torque, or endurance accuracy.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--reduced-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--degrees", type=int, nargs="+", default=(3, 5))
    parser.add_argument("--dop-rtol", type=float, default=1e-11)
    parser.add_argument("--dop-atol", type=float, default=1e-13)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    result = validate(args.seed.resolve(strict=True), args.reduced_profile.resolve(strict=True),
                      degrees=tuple(args.degrees), dop_rtol=args.dop_rtol, dop_atol=args.dop_atol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for degree, row in result["results"].items():
        maximum = max(
            item["maximum_scaled"] for muscle in row["endpoint_error_vs_DOP853"].values()
            for item in muscle.values()
        )
        print(f"Radau-{degree}: maximum scaled Ding endpoint error vs DOP853 = {maximum:.3e}")
    print(f"Detailed coupled fixed-input audit: {args.output}")


if __name__ == "__main__":
    main()
