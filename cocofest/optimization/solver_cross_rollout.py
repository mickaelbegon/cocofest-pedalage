"""Independent open-loop evaluation of exported cycling commands.

The physical state is propagated continuously; optimized shooting nodes are
used only as observations. Numerical Radau/GL maps share the same vector field
and score definition as DOP853. They reproduce tableaux, not native solver
code, cost transcription, SQP settings, or truncated Newton iterations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.special import roots_jacobi, roots_legendre

from cocofest.dynamics.reduced_cycling import ReducedCyclingDynamics
from cocofest.optimization.isokinetic_cycling import inverse_load_torque, produced_mechanical_power


EVALUATORS = ("dop853", "radau5", "gauss-legendre4x5")
COMPONENTS = ("Cn", "F", "A", "Tau1", "Km")
PARAMETERS = ("Fmax", "a_scale", "alpha_a", "tau_fat", "alpha_tau1", "alpha_km",
              "tau1_rest", "km_rest", "tauc", "tau2", "pd0", "pdt")


def file_stamp(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}


@dataclass(frozen=True)
class CommonCost:
    """Explicit physical score, independent of either original NLP objective."""

    fatigue_weight: float = 10000.0
    fatigue_shape: str = "quadratic"
    speed_weight: float = 0.0
    target_omega: float = -2 * np.pi
    pulse_width_weight: float = 0.0
    terminal_capacity_weight: float = 0.0

    def __post_init__(self):
        if self.fatigue_shape not in ("quadratic", "linear"):
            raise ValueError("fatigue_shape must be quadratic or linear")
        for key, value in asdict(self).items():
            if key == "fatigue_shape":
                continue
            if not np.isfinite(value) or (key != "target_omega" and value < 0):
                raise ValueError(f"Invalid common cost field {key}")


@dataclass
class RolloutSource:
    metadata: dict
    provenance: dict
    muscles: tuple[str, ...]
    parameters: dict
    controls: np.ndarray
    shooting_states: np.ndarray
    state_names: tuple[str, ...]
    duration: float
    intervals_per_cycle: int
    cycle_start: int
    cycles: int

    @property
    def dt(self):
        return self.duration / self.intervals_per_cycle

    @property
    def initial_state(self):
        return self.shooting_states[:, 0].copy()


def _row(values, name):
    values = np.asarray(values, dtype=float)
    if values.ndim == 2 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite scalar trace")
    return values


def _positive_integer(value, name, *, minimum=1):
    if isinstance(value, bool) or not np.isfinite(value) or int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def load_source(path, profile, *, model_config=None, cycle_start=0, cycles=1, cycle_duration=None,
                formulation_override=None):
    """Load complete commands and initial physical state; reject guessed models.

    A legacy archive without embedded muscle parameters requires an explicitly
    supplied model config. Its provenance is labelled user-declared, not
    verified against the original run. Full five-state reconstructed exports
    from local Ding reduction are accepted, compact optimization states are not.
    """
    cycle_start = _positive_integer(cycle_start, "cycle_start", minimum=0)
    cycles = _positive_integer(cycles, "cycles")
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        if "metadata__json" not in data:
            raise ValueError("Source archive has no metadata__json")
        metadata = json.loads(str(data["metadata__json"].item()))
        # JSON permits NaN by default in Python; scientific provenance does not.
        json.dumps(metadata, allow_nan=False)
        if metadata.get("model_formulation") != "periodic_node":
            raise ValueError("Only explicitly declared periodic_node forcing is supported")
        if metadata.get("mechanical_formulation") != "reduced" or metadata.get("bilateral_reduced", False):
            raise ValueError("This evaluator requires a unilateral reduced mechanical model")
        declared_formulation = metadata.get("formulation")
        if declared_formulation is None and formulation_override in ("dynamic", "isokinetic"):
            metadata = dict(metadata, formulation=formulation_override)
        elif declared_formulation is None and formulation_override is not None:
            raise ValueError("formulation_override must be 'dynamic' or 'isokinetic'.")
        elif declared_formulation is not None and formulation_override is not None:
            if formulation_override != declared_formulation:
                raise ValueError("formulation_override conflicts with the archive declaration.")
        if metadata.get("formulation") not in ("dynamic", "isokinetic"):
            raise ValueError("Source must declare dynamic or isokinetic formulation")
        if metadata.get("torque_application") != "constant":
            raise ValueError("Only a declared constant torque application is supported")
        for field in ("activate_force_length_relationship", "activate_force_velocity_relationship",
                      "activate_passive_force_relationship"):
            if type(metadata.get(field)) is not bool:
                raise ValueError(f"Source must explicitly declare {field}")
        if metadata.get("calcium_forcing_formulation") != "exact_exponential_periodic_node":
            raise ValueError("Unsupported or undeclared calcium forcing")
        count = _positive_integer(metadata.get("stimulations_per_cycle", 0), "stimulations_per_cycle")
        muscles = tuple(profile.muscle_names)
        control_keys = {f"controls__last_pulse_width_{name}" for name in muscles}
        actual_pw = {key for key in data.files if key.startswith("controls__last_pulse_width_")}
        if actual_pw != control_keys:
            raise ValueError("Profile muscle names and physical pulse-width controls differ")
        extra_controls = {key for key in data.files if key.startswith("controls__")} - control_keys
        # ``pw_slew_*`` controls belong to an auxiliary lifting used solely to
        # express a pulse-width increment penalty.  The physical vector field
        # consumes only ``last_pulse_width_*``; keeping the lift in provenance
        # but excluding it from the replay is exact, not an approximation.
        auxiliary_controls = {
            key for key in extra_controls if key.startswith("controls__pw_slew_")
        }
        unsupported_controls = extra_controls - auxiliary_controls
        if unsupported_controls:
            raise ValueError(
                "Extra controls require an explicit evaluator adapter: "
                f"{sorted(unsupported_controls)}"
            )
        controls = np.stack([_row(data[f"controls__last_pulse_width_{m}"], m) for m in muscles])
        total = controls.shape[1]
        if total % count:
            raise ValueError("Control trace does not contain whole declared cycles")
        start, end = cycle_start * count, (cycle_start + cycles) * count
        if end > total:
            raise ValueError("Requested cycle range exceeds saved controls")
        names = tuple(f"{component}_{m}" for m in muscles for component in COMPONENTS) + ("theta", "omega")
        rows = []
        stride = None
        for name in names:
            key = f"states__{name}"
            if key not in data:
                raise ValueError(f"Missing reconstructed physical state {name}")
            trace = _row(data[key], name)
            candidate, remainder = divmod(len(trace) - 1, total)
            if remainder or candidate < 1 or (stride is not None and stride != candidate):
                raise ValueError("State and control layouts do not determine one common shooting stride")
            stride = candidate
            rows.append(trace[start * stride : end * stride + 1 : stride])
        # A producer may carry inactive collocation metadata in an ACADOS
        # archive; stride 1 is unambiguously a shooting-node trace.
        if stride != 1 and (metadata.get("producer_collocation_method") != "radau" or
                           stride != int(metadata.get("producer_collocation_degree", 0)) + 1):
            raise ValueError("Dense state layout does not match declared Radau collocation")

    declared_period = metadata.get("cycle_duration_s", metadata.get("cycle_duration"))
    if declared_period is None and metadata.get("calcium_stimulation_interval_s") is not None:
        declared_period = float(metadata["calcium_stimulation_interval_s"]) * count
    if cycle_duration is None:
        if declared_period is not None:
            cycle_duration = declared_period
        elif metadata["formulation"] == "isokinetic" and metadata.get("isokinetic_omega"):
            cycle_duration = 2 * np.pi / abs(float(metadata["isokinetic_omega"]))
        else:
            raise ValueError("Legacy dynamic source omits cycle duration: provide --cycle-duration explicitly")
    cycle_duration = float(cycle_duration)
    if not np.isfinite(cycle_duration) or cycle_duration <= 0:
        raise ValueError("cycle_duration must be finite and positive")
    if declared_period is not None and not np.isclose(float(declared_period), cycle_duration, rtol=1e-12, atol=1e-14):
        raise ValueError("Requested cycle duration conflicts with archive metadata")
    if metadata["formulation"] == "isokinetic":
        omega = float(metadata["isokinetic_omega"])
        if omega >= 0 or not np.isclose(cycle_duration * omega, -2 * np.pi, atol=1e-10):
            raise ValueError("Isokinetic speed and cycle duration are incompatible")

    parameters = metadata.get("configured_muscle_parameters")
    basis = "embedded_parameters"
    supplied = None
    if model_config is not None:
        supplied = json.loads(Path(model_config).read_text())
        if not isinstance(supplied.get("provenance"), str) or not supplied["provenance"].strip():
            raise ValueError("Explicit model config must contain a nonempty provenance")
        if supplied.get("schema_version") == 1:
            # Canonical repository configs may contain four calibrated fields
            # and rely on the installed Ding defaults for the remaining ones.
            # Resolve them explicitly and retain all effective values below.
            from cocofest.optimization.configured_cycling_model import resolve_model_config

            supplied = resolve_model_config(supplied)
        if parameters is not None and supplied.get("muscles") != parameters:
            raise ValueError("Explicit model config differs from embedded parameters")
        if parameters is None:
            parameters = supplied.get("muscles")
            basis = "explicit_user_declared_model_not_verified_against_legacy_source"
    if not isinstance(parameters, dict) or set(parameters) != set(muscles):
        raise ValueError("Complete embedded muscle parameters or --model-config are required")
    for name, values in parameters.items():
        if not set(PARAMETERS) <= set(values):
            raise ValueError(f"{name}: full effective parameters required; missing {set(PARAMETERS) - set(values)}")
        for key in PARAMETERS:
            value = values[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"Nonfinite or invalid parameter {name}.{key}")
            valid = value <= 0 if key == "alpha_a" else value >= 0 if key in ("alpha_km", "alpha_tau1", "pd0") else value > 0
            if not valid:
                raise ValueError(f"Invalid sign for {name}.{key}")
    encoded = json.dumps({"schema_version": 1, "muscles": parameters}, sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    fingerprint = sha256(encoded.encode()).hexdigest()
    if metadata.get("muscle_parameter_fingerprint") not in (None, fingerprint):
        raise ValueError("Effective muscle parameters do not match archive fingerprint")
    provenance = {"source": file_stamp(path), "parameters_basis": basis, "state_stride": stride,
                  "cycle_duration_basis": "archive" if declared_period is not None else "explicit_or_isokinetic",
                  "effective_muscle_parameters": parameters, "muscle_parameter_fingerprint": fingerprint,
                  "source_solver": metadata.get("producer_solver"), "source_transcription": metadata.get("producer_transcription_profile")}
    if declared_formulation is None and formulation_override is not None:
        provenance["formulation_basis"] = "explicit_legacy_override"
        provenance["legacy_formulation_override"] = formulation_override
    else:
        provenance["formulation_basis"] = "archive"
    if auxiliary_controls:
        provenance["ignored_auxiliary_controls"] = sorted(auxiliary_controls)
    if model_config is not None:
        provenance["model_config"] = file_stamp(model_config)
    return RolloutSource(metadata, provenance, muscles, parameters, controls[:, start:end], np.stack(rows),
                         names, cycle_duration, count, cycle_start, cycles)


def collocation_tableau(method):
    """Five-stage Radau IIA (order 9) or four-stage Gauss (order 8).

    Project 'Radau-5' denotes five stages, not the classical three-stage
    fifth-order RADAU5 solver. Abscissae are obtained from orthogonal roots.
    """
    if method == "radau5":
        nodes = np.r_[(roots_jacobi(4, 1, 0)[0] + 1) / 2, 1.0]
    elif method == "gauss-legendre4x5":
        nodes = (roots_legendre(4)[0] + 1) / 2
    else:
        raise ValueError(f"Unsupported collocation method {method}")
    primitives = []
    for j, node in enumerate(nodes):
        polynomial = np.poly1d([1.0])
        for k, other in enumerate(nodes):
            if j != k:
                polynomial *= np.poly1d([1, -other]) / (node - other)
        primitives.append(np.polyint(polynomial))
    matrix = np.array([[p(c) - p(0) for p in primitives] for c in nodes])
    weights = np.array([p(1) - p(0) for p in primitives])
    return nodes, matrix, weights, primitives


def implicit_step(rhs, state, t0, dt, method, scale):
    """Solve the complete implicit map to a scaled residual below 1e-9."""
    nodes, matrix, weights, primitives = collocation_tableau(method)
    scale = np.asarray(scale)
    guess = np.tile(state, (len(nodes), 1)) / scale

    def residual(flat):
        stages = flat.reshape(guess.shape) * scale
        rates = np.stack([rhs(t0 + c * dt, x) for c, x in zip(nodes, stages)])
        return ((stages - state - dt * matrix @ rates) / scale).ravel()

    solved = root(residual, guess.ravel(), method="hybr", options={"xtol": 1e-10})
    defect = float(np.max(np.abs(residual(solved.x))))
    if not np.isfinite(defect) or defect > 1e-9:
        raise RuntimeError(f"{method}: implicit solve failed (scaled defect={defect:.3g}): {solved.message}")
    stages = solved.x.reshape(guess.shape) * scale
    rates = np.stack([rhs(t0 + c * dt, x) for c, x in zip(nodes, stages)])
    end = state + dt * weights @ rates

    def dense(times):
        tau = (np.asarray(times) - t0) / dt
        basis = np.array([p(tau) - p(0) for p in primitives])
        return state[:, None] + dt * rates.T @ basis

    return end, dense, int(solved.nfev), defect


def _physical_rhs(source, profile, control):
    metadata, muscles = source.metadata, source.muscles
    truncation = _positive_integer(metadata.get("ding_sum_stim_truncation", 0), "ding_sum_stim_truncation")
    pars = [source.parameters[name] for name in muscles]
    n = len(muscles)
    amplitudes = []
    for p in pars:
        decay = np.exp(-source.dt / p["tauc"])
        amplitude = decay ** (truncation - 1) + (1 + (p["km_rest"] + 0.04) * decay) * sum(decay**k for k in range(truncation - 1))
        amplitudes.append(amplitude)
    torque = float(metadata.get("signed_crank_torque_nm", metadata.get("constant_crank_torque", np.nan)))
    if not np.isfinite(torque):
        raise ValueError("Explicit finite signed crank torque is required")
    isokinetic = metadata["formulation"] == "isokinetic"
    if isokinetic and torque != 0:
        raise ValueError("Isokinetic formulation must have zero prescribed external torque")

    def rhs(time, state):
        theta, omega = state[5*n:5*n+2]
        if isokinetic:
            omega = float(metadata["isokinetic_omega"])
        fl, fv, fp = profile.muscle_relationships(theta, omega) if any(
            metadata[key] for key in ("activate_force_length_relationship", "activate_force_velocity_relationship", "activate_passive_force_relationship")
        ) else (np.ones(n), np.ones(n), np.zeros(n))
        if not metadata["activate_force_length_relationship"]:
            fl = np.ones(n)
        if not metadata["activate_force_velocity_relationship"]:
            fv = np.ones(n)
        if not metadata["activate_passive_force_relationship"]:
            fp = np.zeros(n)
        rates = np.empty(5*n + 2)
        for j, p in enumerate(pars):
            cn, force, capacity, tau1, km = state[5*j:5*j+5]
            if tau1 <= 0 or km + cn <= 0 or capacity <= 0:
                raise FloatingPointError(f"Ding domain left for {muscles[j]} (A, Tau1 or Km+Cn <= 0)")
            activation = cn / (km + cn)
            history = amplitudes[j] * np.exp(-time / p["tauc"])
            fraction = -np.expm1(-(control[j] - p["pd0"]) / p["pdt"])
            rates[5*j:5*j+5] = (
                (history-cn) / p["tauc"],
                (capacity*fraction*activation - force/(tau1+p["tau2"]*activation)) * (fl[j]*fv[j]+fp[j]),
                -(capacity-p["a_scale"])/p["tau_fat"] + p["alpha_a"]*force,
                -(tau1-p["tau1_rest"])/p["tau_fat"] + p["alpha_tau1"]*force,
                -(km-p["km_rest"])/p["tau_fat"] + p["alpha_km"]*force,
            )
        forces = state[np.arange(n)*5+1]
        rates[-2:] = (omega, 0 if isokinetic else profile.acceleration(theta, omega, forces, external_crank_torque=torque))
        return rates

    return rhs


def evaluate_rollout(source, profile, *, evaluator="dop853", cost=None, samples_per_interval=9,
                     rtol=1e-11, atol=1e-13):
    """Evaluate all commands from the first saved state, never resetting physiology."""
    if evaluator not in EVALUATORS:
        raise ValueError(f"Unsupported evaluator {evaluator}")
    samples_per_interval = _positive_integer(samples_per_interval, "samples_per_interval", minimum=2)
    if not (0 < rtol < 1 and 0 < atol < 1):
        raise ValueError("Integration tolerances must be between 0 and 1")
    cost = cost or CommonCost()
    n = len(source.muscles)
    nx = len(source.state_names)
    arows = np.arange(n)*5+2
    frows = np.arange(n)*5+1
    rest = np.array([source.parameters[m]["a_scale"] for m in source.muscles])
    pd0 = np.array([source.parameters[m]["pd0"] for m in source.muscles])
    pwmax = float(source.metadata.get("pulse_width_maximum_s", np.nan))
    if not np.isfinite(pwmax) or np.any(pwmax <= pd0):
        raise ValueError("Explicit physical PW upper bound required")
    pw_violation = max(float(np.max(pd0[:, None]-source.controls)), float(np.max(source.controls-pwmax)), 0.0)
    if pw_violation > 1e-9:
        raise ValueError(f"Saved physical pulse widths violate bounds by {pw_violation} seconds")
    initial = source.initial_state
    if source.metadata["formulation"] == "isokinetic" and not np.isclose(initial[-1], source.metadata["isokinetic_omega"], atol=1e-9):
        raise ValueError("Initial omega disagrees with isokinetic speed")
    # States plus per-muscle fatigue AUC and squared AUC, speed, PW, work.
    current = np.r_[initial, np.zeros(2*n+3)]
    scales = np.r_[np.array([[1, source.parameters[m]["Fmax"], source.parameters[m]["a_scale"],
                            source.parameters[m]["tau1_rest"], source.parameters[m]["km_rest"]]
                           for m in source.muscles]).ravel(), 2*np.pi, 2*np.pi, np.ones(2*n+3)]
    nodes = [initial]
    dense_states, dense_times = [], []
    load_values = []
    work_by_cycle = []
    previous_work = 0.0
    calls, defect = 0, 0.0
    started = perf_counter()
    for interval, control in enumerate(source.controls.T):
        physical_rhs = _physical_rhs(source, profile, control)

        def rhs(local_time, augmented):
            state = augmented[:nx]
            fatigue = 1-state[arows]/rest
            speed = (state[-1]-cost.target_omega)**2
            pw = float(np.sum(((control-pd0)/(pwmax-pd0))**2))
            theta, omega = state[-2:]
            load = inverse_load_torque(profile, theta, omega, state[frows]) if source.metadata["formulation"] == "isokinetic" else float(source.metadata.get("signed_crank_torque_nm", source.metadata.get("constant_crank_torque", np.nan)))
            bext = profile.coefficient_values(theta)["external_torque_effectiveness"]
            power = float(produced_mechanical_power(load, bext, omega))
            return np.r_[physical_rhs(local_time, state), fatigue/source.duration,
                         fatigue**2/source.duration, speed/source.duration, pw/source.duration, power]

        substeps = 5 if evaluator == "gauss-legendre4x5" else 1
        grid = np.linspace(0, source.dt, samples_per_interval)
        sample_columns = np.empty((nx, samples_per_interval))
        for substep in range(substeps):
            t0, step = substep*source.dt/substeps, source.dt/substeps
            if evaluator == "dop853":
                solved = solve_ivp(rhs, (t0, t0+step), current, method="DOP853", rtol=rtol,
                                   atol=atol*scales, dense_output=True)
                if not solved.success:
                    raise RuntimeError(f"DOP853 interval {interval}: {solved.message}")
                current, dense, evaluations = solved.y[:, -1], solved.sol, solved.nfev
            else:
                current, dense, evaluations, local_defect = implicit_step(rhs, current, t0, step, evaluator, scales)
                defect = max(defect, local_defect)
            mask = (grid >= t0 - 1e-15) & (grid <= t0+step+1e-15)
            sample_columns[:, mask] = dense(grid[mask])[:nx]
            calls += evaluations
        nodes.append(current[:nx].copy())
        dense_states.append(sample_columns)
        dense_times.append(interval*source.dt + grid)
        if source.metadata["formulation"] == "isokinetic":
            load_values.extend(inverse_load_torque(profile, x[-2], x[-1], x[frows]) for x in sample_columns.T)
        if (interval+1) % source.intervals_per_cycle == 0:
            work_by_cycle.append(float(current[-1]-previous_work))
            previous_work = current[-1]

    sampled = np.hstack(dense_states)
    nodes = np.stack(nodes, axis=1)
    fatigue_auc, fatigue_squared = current[nx:nx+n], current[nx+n:nx+2*n]
    components = {
        "fatigue": float(cost.fatigue_weight*np.sum(fatigue_squared if cost.fatigue_shape == "quadratic" else fatigue_auc)),
        "speed": float(cost.speed_weight*current[-3]),
        "pulse_width": float(cost.pulse_width_weight*current[-2]),
        "terminal_capacity": float(cost.terminal_capacity_weight*np.sum((1-current[arows]/rest)**2)),
    }
    capacity = sampled[arows]/rest[:, None]
    theta_cycle = nodes[-2, ::source.intervals_per_cycle]
    target_theta = initial[-2] - 2*np.pi*np.arange(source.cycles+1)
    checks = {"physical_pw_bound_violation_s": pw_violation,
              "minimum_force_n": float(np.min(sampled[frows])),
              "minimum_capacity_ratio": float(np.min(capacity)),
              "maximum_capacity_ratio": float(np.max(capacity)),
              "omega_min_rad_s": float(np.min(sampled[-1])), "omega_max_rad_s": float(np.max(sampled[-1])),
              "cycle_boundary_phase_error_rad": (theta_cycle-target_theta).tolist(),
              "maximum_cycle_boundary_phase_error_rad": float(np.max(np.abs(theta_cycle-target_theta))),
              "produced_work_per_cycle_j": work_by_cycle,
              "constraint_scope": "sampled physical metrics; not a certificate of all original NLP constraints"}
    slack = source.metadata.get("terminal_wheel_q_slack")
    checks["phase_slack_rad"] = slack
    checks["sampled_phase_bound_violation_rad"] = (
        None if slack is None else max(checks["maximum_cycle_boundary_phase_error_rad"]-float(slack), 0.0)
    )
    checks["phase_target_basis"] = "one negative revolution per declared cycle from common incoming theta"
    guard = source.metadata.get("reduced_internal_crank_velocity_guard")
    checks["velocity_guard_declared"] = guard
    checks["sampled_velocity_bound_violation_rad_s"] = None
    if guard:
        fields = ("reduced_internal_crank_velocity_guard_target_rad_s",
                  "reduced_internal_crank_velocity_guard_fast_margin_rad_s",
                  "reduced_internal_crank_velocity_guard_slow_margin_rad_s")
        if any(source.metadata.get(field) is None for field in fields):
            raise ValueError("Active source velocity guard omits target or bound margins")
        target, fast, slow = (float(source.metadata[field]) for field in fields)
        checks["velocity_lower_bound_rad_s"] = target-fast
        checks["velocity_upper_bound_rad_s"] = target+slow
        checks["sampled_velocity_bound_violation_rad_s"] = max(
            target-fast-checks["omega_min_rad_s"], checks["omega_max_rad_s"]-target-slow, 0.0)
    checks["full_nlp_certified"] = False
    if load_values:
        low, high = float(source.metadata["load_torque_min"]), float(source.metadata["load_torque_max"])
        checks.update(load_min_nm=float(min(load_values)), load_max_nm=float(max(load_values)),
                      sampled_load_bound_violation_nm=max(low-min(load_values), max(load_values)-high, 0.0),
                      energy_target_per_cycle_j=float(source.metadata["energy_equivalent_torque"])*2*np.pi)
    metrics = {"status": "success", "evaluator": evaluator,
               "implementation": "scipy_DOP853" if evaluator == "dop853" else "independent_converged_implicit_tableau_not_native_solver",
               "score_definition": asdict(cost), "score_components": components, "common_score": float(sum(components.values())),
               "cumulative_normalized_fatigue_cycles_by_muscle": dict(zip(source.muscles, fatigue_auc.tolist())),
               "final_capacity_ratio_by_muscle": dict(zip(source.muscles, (current[arows]/rest).tolist())),
               "physical_metrics": checks, "elapsed_s": perf_counter()-started,
               "rhs_or_nonlinear_residual_evaluations": int(calls), "maximum_scaled_implicit_defect": defect,
               "samples_per_interval": samples_per_interval, "rtol": rtol, "atol_scaled": atol,
               "maximum_endpoint_discrepancy_by_state": dict(zip(source.state_names, np.max(np.abs(nodes-source.shooting_states), axis=1).tolist()))}
    arrays = {"time": np.concatenate(dense_times), "states": sampled, "shooting_states": nodes,
              "state_names": np.asarray(source.state_names), "controls": source.controls}
    return metrics, arrays


def compatibility(sources):
    """Give explicit reasons when a cross-source ranking is not meaningful."""
    fields = ("model_formulation", "mechanical_formulation", "formulation", "torque_application",
              "constant_crank_torque", "signed_crank_torque_nm", "isokinetic_omega",
              "activate_force_length_relationship", "activate_force_velocity_relationship",
              "activate_passive_force_relationship", "ding_sum_stim_truncation", "pulse_width_maximum_s",
              "energy_equivalent_torque", "load_torque_min", "load_torque_max")
    reasons = []
    first = sources[0]
    for index, other in enumerate(sources[1:], 1):
        for field in fields:
            if first.metadata.get(field) != other.metadata.get(field):
                reasons.append(f"source {index}: different {field}")
        if first.parameters != other.parameters or first.muscles != other.muscles:
            reasons.append(f"source {index}: different model parameters or muscles")
        if (first.duration, first.cycles, first.intervals_per_cycle) != (other.duration, other.cycles, other.intervals_per_cycle):
            reasons.append(f"source {index}: different horizon or stimulation calendar")
        if not np.allclose(first.initial_state, other.initial_state, rtol=1e-10, atol=1e-10):
            reasons.append(f"source {index}: different incoming physical state")
    return {"comparable": not reasons, "reasons": reasons, "initial_state_rtol": 1e-10, "initial_state_atol": 1e-10}


def ranking_physical_gate(entries):
    """Reject a score ranking when the DOP853 physical reference is infeasible.

    The numerical maps are useful sensitivity checks, but DOP853 is the stated
    continuous reference.  A candidate that violates a sampled common bound
    under that rollout must be reported, never silently ordered by cost.
    """

    reasons = []
    for index, entry in enumerate(entries):
        if entry.get("status") != "success":
            reasons.append(f"source {index}: DOP853 evaluation failed")
            continue
        metrics = entry.get("physical_metrics", {})
        for field in (
            "physical_pw_bound_violation_s",
            "sampled_phase_bound_violation_rad",
            "sampled_velocity_bound_violation_rad_s",
            "sampled_load_bound_violation_nm",
        ):
            value = metrics.get(field)
            if value is not None and float(value) > 0.0:
                reasons.append(f"source {index}: {field}={float(value):.12g}")
    return {"rankable": not reasons, "reasons": reasons, "reference": "dop853"}


def qualified_velocity_comparability_gate(
    entries, *, max_velocity_violation_rad_s=0.01, max_velocity_ratio=2.0
):
    """Allow a labeled solver comparison when only small, similar speed errors remain.

    This is deliberately distinct from :func:`ranking_physical_gate`: it is a
    model-discretization comparability criterion, never a continuous-feasibility
    certificate.  Pulse-width, phase, and load violations remain hard rejects.
    """
    if max_velocity_violation_rad_s < 0 or max_velocity_ratio < 1:
        raise ValueError("Qualified velocity thresholds must be non-negative and ratio >= 1.")
    reasons = []
    velocity_violations = []
    hard_fields = (
        "physical_pw_bound_violation_s",
        "sampled_phase_bound_violation_rad",
        "sampled_load_bound_violation_nm",
    )
    for index, entry in enumerate(entries):
        if entry.get("status") != "success":
            reasons.append(f"source {index}: DOP853 status={entry.get('status')}")
            continue
        metrics = entry.get("physical_metrics") or {}
        for field in hard_fields:
            value = metrics.get(field)
            if value is not None and float(value) > 0.0:
                reasons.append(f"source {index}: {field}={float(value):.12g}")
        value = float(metrics.get("sampled_velocity_bound_violation_rad_s", 0.0) or 0.0)
        velocity_violations.append(value)
        if value > max_velocity_violation_rad_s:
            reasons.append(
                f"source {index}: sampled_velocity_bound_violation_rad_s={value:.12g} "
                f"> {max_velocity_violation_rad_s:.12g}"
            )
    ratio = None
    if len(velocity_violations) > 1 and all(value > 0.0 for value in velocity_violations):
        ratio = max(velocity_violations) / min(velocity_violations)
        if ratio > max_velocity_ratio:
            reasons.append(
                f"velocity violation ratio={ratio:.12g} > {max_velocity_ratio:.12g}"
            )
    return {
        "rankable": not reasons,
        "reasons": reasons,
        "reference": "dop853",
        "classification": "qualified_numerical_comparison_not_continuous_certificate",
        "maximum_velocity_violation_rad_s": max_velocity_violation_rad_s,
        "maximum_velocity_violation_ratio": max_velocity_ratio,
        "observed_velocity_violations_rad_s": velocity_violations,
        "observed_velocity_violation_ratio": ratio,
    }


def run_matrix(paths, profile_path, output_dir, *, model_config=None, evaluators=EVALUATORS,
               cycle_start=0, cycles=1, cycle_duration=None, samples_per_interval=9, cost=None,
               rtol=1e-11, atol=1e-13, qualified_max_velocity_violation_rad_s=0.01,
               qualified_max_velocity_ratio=2.0):
    """Save an auditable source × evaluator matrix and corresponding trajectories."""
    if not paths or not evaluators or "dop853" not in evaluators or len(set(evaluators)) != len(evaluators):
        raise ValueError("At least one source and unique evaluators including dop853 are required")
    if set(evaluators) - set(EVALUATORS):
        raise ValueError("Unknown evaluators")
    profile = ReducedCyclingDynamics.load(profile_path)
    sources = [load_source(path, profile, model_config=model_config, cycle_start=cycle_start,
                           cycles=cycles, cycle_duration=cycle_duration) for path in paths]
    gate = compatibility(sources)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "matrix.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}; select a new output directory")
    report = {"schema": "cocofest-solver-cross-rollout-v1", "mode": "open_loop_no_reoptimization",
              "reference": "dop853", "profile": file_stamp(profile_path), "comparison": gate,
              "cost_scope": "common physical score, not reconstructed original NLP total",
              "cycle_start": cycle_start, "cycles": cycles, "sources": [], "rankings": {},
              "qualified_rankings": {}}
    for index, source in enumerate(sources):
        row = {"source_index": index, "provenance": source.provenance, "evaluations": {}}
        for evaluator in evaluators:
            try:
                metrics, arrays = evaluate_rollout(source, profile, evaluator=evaluator, cost=cost,
                                                  samples_per_interval=samples_per_interval, rtol=rtol, atol=atol)
                name = f"source-{index}-{evaluator}.npz"
                if (output/name).exists():
                    raise FileExistsError(f"Refusing to overwrite {output/name}")
                np.savez_compressed(output/name, **arrays)
                metrics["trajectory_path"] = str((output/name).resolve())
                row["evaluations"][evaluator] = metrics
            except (RuntimeError, FloatingPointError, ValueError) as error:
                row["evaluations"][evaluator] = {"status": "failed", "error": str(error)}
        reference = row["evaluations"]["dop853"]
        if reference["status"] == "success":
            for metric in row["evaluations"].values():
                if metric["status"] == "success":
                    metric["score_difference_from_dop853"] = metric["common_score"]-reference["common_score"]
        report["sources"].append(row)
    physical_gate = ranking_physical_gate(
        [row["evaluations"]["dop853"] for row in report["sources"]]
    )
    report["ranking_physical_gate"] = physical_gate
    qualified_gate = qualified_velocity_comparability_gate(
        [row["evaluations"]["dop853"] for row in report["sources"]],
        max_velocity_violation_rad_s=qualified_max_velocity_violation_rad_s,
        max_velocity_ratio=qualified_max_velocity_ratio,
    )
    report["qualified_comparability_gate"] = qualified_gate
    if gate["comparable"] and physical_gate["rankable"]:
        for evaluator in evaluators:
            entries = [row["evaluations"][evaluator] for row in report["sources"]]
            report["rankings"][evaluator] = (sorted(range(len(sources)), key=lambda k: entries[k]["common_score"])
                                              if all(row["status"] == "success" for row in entries) else None)
    if gate["comparable"] and qualified_gate["rankable"]:
        for evaluator in evaluators:
            entries = [row["evaluations"][evaluator] for row in report["sources"]]
            report["qualified_rankings"][evaluator] = (sorted(range(len(sources)), key=lambda k: entries[k]["common_score"])
                                                        if all(row["status"] == "success" for row in entries) else None)
    report["ranking_caveat"] = (
        "rankings require compatible sources and zero sampled common-bound "
        "violation under the DOP853 physical reference"
    )
    report["qualified_ranking_caveat"] = (
        "qualified_rankings require compatible sources, no sampled pulse-width/phase/load violation, "
        "and DOP853 speed errors below the declared absolute and relative thresholds; they are numerical "
        "solver comparisons, not continuous-feasibility certificates"
    )
    destination.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report
