"""Bounded, noise-weighted parameter estimation for staged Ding experiments.

This module deliberately depends only on NumPy/SciPy. A predictor supplies the
Ding integration and calibrated sensor projection. Force experiments and fatigue
experiments are fitted separately; fitted fresh-force parameters are frozen in
the second stage. Fisher diagnostics exclude the MAP prior, so a strong prior
cannot be mistaken for information supplied by an experiment.
"""

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    initial: float
    lower: float
    upper: float
    scale: float
    prior_mean: float | None = None
    prior_std: float | None = None

    def __post_init__(self):
        values = (self.initial, self.lower, self.upper, self.scale)
        if not self.name or not np.all(np.isfinite(values)):
            raise ValueError("Parameter names and finite bounds/scales are required")
        if not self.lower < self.upper or not self.lower <= self.initial <= self.upper or self.scale <= 0:
            raise ValueError(f"Invalid bounds, initial value or scale for {self.name}")
        if (self.prior_mean is None) != (self.prior_std is None):
            raise ValueError("Specify both prior_mean and prior_std, or neither")
        if self.prior_std is not None and (
            not np.isfinite(self.prior_mean) or not np.isfinite(self.prior_std) or self.prior_std <= 0
        ):
            raise ValueError("Prior mean must be finite and prior_std positive")


@dataclass(frozen=True)
class TrialObservation:
    """One complete trial; its context is passed unchanged to the predictor.

    ``observed`` has shape (samples,) or (samples, channels). ``noise_std`` is
    broadcastable to it, in the same units (e.g. N for tangential/radial force).
    The diagonal noise model assumes independent samples and channels. Correlated
    data must be prewhitened in observations AND predictor, or downsampled.
    """

    name: str
    observed: np.ndarray
    noise_std: Any
    context: Any = None

    def __post_init__(self):
        observed = np.array(self.observed, dtype=float, copy=True)
        if not self.name or observed.ndim not in (1, 2) or observed.size == 0 or not np.all(np.isfinite(observed)):
            raise ValueError("A named trial needs nonempty finite 1D/2D observations")
        std = np.array(np.broadcast_to(self.noise_std, observed.shape), dtype=float, copy=True)
        if not np.all(np.isfinite(std)) or np.any(std <= 0):
            raise ValueError("Observation noise standard deviations must be finite and positive")
        observed.flags.writeable = False
        std.flags.writeable = False
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "noise_std", std)


Predictor = Callable[[Mapping[str, float], Any], np.ndarray]


def _prediction(predictor, parameters, trial):
    prediction = np.asarray(predictor(parameters, trial.context), dtype=float)
    if prediction.shape != trial.observed.shape or not np.all(np.isfinite(prediction)):
        raise ValueError(f"Nonfinite or incorrectly shaped prediction for {trial.name}")
    return prediction


def _check_trials(training, holdout):
    if not training:
        raise ValueError("At least one training trial is required")
    names = [trial.name for trial in (*training, *holdout)]
    if len(set(names)) != len(names):
        raise ValueError("Training and holdout trial names must be unique and disjoint")


def fisher_diagnostics(jacobian: np.ndarray, *, relative_rank_tolerance: float = 1e-7) -> dict:
    """Data-only local Fisher diagnostics for whitened, scaled sensitivities.

    The tolerance defines *numerical* identifiability, not a clinical acceptance
    threshold. Fisher condition is the square of sensitivity condition. Null
    directions are explicitly retained when there are fewer rows than parameters.
    No pseudo-inverse confidence intervals are returned for deficient designs.
    """
    jacobian = np.asarray(jacobian, dtype=float)
    if jacobian.ndim != 2 or jacobian.shape[1] == 0 or not np.all(np.isfinite(jacobian)):
        raise ValueError("Expected a finite observations-by-parameters Jacobian")
    if not 0 < relative_rank_tolerance < 1:
        raise ValueError("relative_rank_tolerance must lie between zero and one")
    # Retain the parameter nullspace without allocating a samples-by-samples U
    # for long acquisition records.
    _, singular, right = np.linalg.svd(jacobian, full_matrices=jacobian.shape[0] < jacobian.shape[1])
    n_parameters = jacobian.shape[1]
    singular = np.pad(singular, (0, max(0, n_parameters - singular.size)))
    cutoff = singular[0] * relative_rank_tolerance if singular.size else 0.
    rank = int(np.count_nonzero(singular > cutoff))
    full_rank = rank == n_parameters
    condition = float(singular[0] / singular[-1]) if full_rank else None
    covariance = None
    if full_rank:
        covariance = (right.T * (1. / singular**2)) @ right
    return {
        "n_observations": jacobian.shape[0],
        "n_parameters": n_parameters,
        "rank": rank,
        "full_rank": full_rank,
        "relative_rank_tolerance": relative_rank_tolerance,
        "singular_values": singular.tolist(),
        "sensitivity_condition_number": condition,
        "fisher_condition_number": condition**2 if condition is not None else None,
        "weakest_scaled_parameter_direction": right[-1].tolist(),
        "scaled_covariance_if_interior": covariance.tolist() if covariance is not None else None,
        "information_source": "measurement_only_excludes_priors",
    }


def _bounded_jacobian(function, point, lower, upper, relative_step=1e-5):
    """Finite differences in scaled coordinates, without crossing bounds."""
    center = function(point)
    jacobian = np.empty((center.size, point.size))
    for index in range(point.size):
        step = relative_step * max(1., abs(point[index]))
        left = max(lower[index], point[index] - step)
        right = min(upper[index], point[index] + step)
        if right <= left:
            raise ValueError("Parameter bounds are too narrow for sensitivity differences")
        minus, plus = point.copy(), point.copy()
        minus[index], plus[index] = left, right
        jacobian[:, index] = (function(plus) - function(minus)) / (right - left)
    return jacobian


def evaluate_trials(predictor: Predictor, parameters: Mapping[str, float], trials: Sequence[TrialObservation]) -> dict:
    """Unfitted validation errors per trial/channel, in observation units."""
    metrics = {}
    for trial in trials:
        residual = _prediction(predictor, parameters, trial) - trial.observed
        matrix = residual[:, None] if residual.ndim == 1 else residual
        metrics[trial.name] = {
            "rmse_by_channel": np.sqrt(np.mean(matrix**2, axis=0)).tolist(),
            "max_abs_by_channel": np.max(np.abs(matrix), axis=0).tolist(),
            "noise_normalized_rmse": float(np.sqrt(np.mean((residual / trial.noise_std)**2))),
            "n_measurements": residual.size,
        }
    return metrics


@dataclass(frozen=True)
class IdentificationResult:
    parameters: dict[str, float]
    fitted_parameter_names: tuple[str, ...]
    converged: bool
    message: str
    objective: float
    n_function_evaluations: int
    active_bounds: dict[str, int]
    fisher: dict
    training_metrics: dict
    holdout_metrics: dict

    def to_dict(self) -> dict:
        return asdict(self)


def fit_parameters(
    predictor: Predictor,
    training: Sequence[TrialObservation],
    specs: Sequence[ParameterSpec],
    *,
    fixed_parameters: Mapping[str, float] | None = None,
    holdout: Sequence[TrialObservation] = (),
    max_nfev: int = 300,
    relative_rank_tolerance: float = 1e-7,
) -> IdentificationResult:
    """Gaussian MAP (or least squares without priors), using bounded TRF.

    Parameters are optimized as physical_value / scale. Noise, bounds and prior
    standard deviations retain physical units. Holdout trials never enter the
    optimization. Local covariance assumes independent known Gaussian noise and
    is conditional on fixed mechanical gains and all other fixed parameters.
    """
    training, holdout, specs = tuple(training), tuple(holdout), tuple(specs)
    _check_trials(training, holdout)
    names = tuple(spec.name for spec in specs)
    if not names or len(set(names)) != len(names):
        raise ValueError("Fitted parameter names must be nonempty and unique")
    fixed = dict(fixed_parameters or {})
    if set(fixed).intersection(names):
        raise ValueError("A parameter cannot be simultaneously fitted and fixed")
    if not all(np.isfinite(value) for value in fixed.values()):
        raise ValueError("Fixed parameters must be finite")
    scale = np.array([spec.scale for spec in specs])
    lower = np.array([spec.lower for spec in specs]) / scale
    upper = np.array([spec.upper for spec in specs]) / scale
    initial = np.array([spec.initial for spec in specs]) / scale

    def unpack(point):
        return {**fixed, **dict(zip(names, (point * scale).tolist()))}

    def measurement_residual(point):
        parameters = unpack(point)
        return np.concatenate([
            ((_prediction(predictor, parameters, trial) - trial.observed) / trial.noise_std).ravel()
            for trial in training
        ])

    def map_residual(point):
        prior = [(point[index] * spec.scale - spec.prior_mean) / spec.prior_std
                 for index, spec in enumerate(specs) if spec.prior_std is not None]
        return np.r_[measurement_residual(point), prior]

    solution = least_squares(map_residual, initial, bounds=(lower, upper), method="trf", jac="3-point",
                             max_nfev=max_nfev, ftol=1e-9, xtol=1e-9, gtol=1e-9)
    parameters = unpack(solution.x)
    jacobian = _bounded_jacobian(measurement_residual, solution.x, lower, upper)
    fisher = fisher_diagnostics(jacobian, relative_rank_tolerance=relative_rank_tolerance)
    fisher["parameter_names"] = list(names)
    fisher["parameter_scales"] = scale.tolist()
    fisher["at_parameter_bound"] = bool(np.any(solution.active_mask))
    if fisher["at_parameter_bound"]:
        fisher["scaled_covariance_if_interior"] = None
    return IdentificationResult(
        parameters=parameters, fitted_parameter_names=names,
        converged=bool(solution.success), message=solution.message,
        objective=float(solution.cost), n_function_evaluations=solution.nfev,
        active_bounds=dict(zip(names, solution.active_mask.astype(int).tolist())), fisher=fisher,
        training_metrics=evaluate_trials(predictor, parameters, training),
        holdout_metrics=evaluate_trials(predictor, parameters, holdout),
    )


def fit_force_then_fatigue(
    predictor: Predictor,
    force_trials: Sequence[TrialObservation],
    fatigue_trials: Sequence[TrialObservation],
    force_specs: Sequence[ParameterSpec],
    fatigue_specs: Sequence[ParameterSpec],
    *,
    fixed_parameters: Mapping[str, float] | None = None,
    force_holdout: Sequence[TrialObservation] = (),
    fatigue_holdout: Sequence[TrialObservation] = (),
    max_nfev: int = 300,
) -> dict[str, IdentificationResult]:
    """Fit force first, then freeze its estimates while fitting fatigue.

    The predictor/context must disable fatigue during fresh-force trials, or
    these trials must be short enough that the fixed initial fatigue parameters
    have negligible influence. All physical state initialization lives in the
    trial context: consecutive fatigue/recovery blocks must preserve continuity.
    This procedure does not propagate stage-one uncertainty into stage two;
    resampling complete trials is needed for unconditional confidence intervals.
    """
    force_specs, fatigue_specs = tuple(force_specs), tuple(fatigue_specs)
    force_names = {spec.name for spec in force_specs}
    fatigue_names = {spec.name for spec in fatigue_specs}
    fixed = dict(fixed_parameters or {})
    if force_names & fatigue_names or set(fixed) & (force_names | fatigue_names):
        raise ValueError("Force, fatigue and fixed parameter groups must be disjoint")
    all_training = tuple(force_trials) + tuple(fatigue_trials)
    all_holdout = tuple(force_holdout) + tuple(fatigue_holdout)
    _check_trials(all_training, all_holdout)
    force = fit_parameters(
        predictor, force_trials, force_specs,
        fixed_parameters={**fixed, **{spec.name: spec.initial for spec in fatigue_specs}},
        holdout=force_holdout, max_nfev=max_nfev,
    )
    if not force.converged:
        raise RuntimeError(f"Fresh-force fit did not converge; fatigue fit not started: {force.message}")
    fatigue = fit_parameters(
        predictor, fatigue_trials, fatigue_specs,
        fixed_parameters={name: value for name, value in force.parameters.items() if name not in fatigue_names},
        holdout=fatigue_holdout, max_nfev=max_nfev,
    )
    return {"force": force, "fatigue": fatigue}
