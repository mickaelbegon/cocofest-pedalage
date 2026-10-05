"""Local robust experimental design for staged Ding identification.

Sensitivities refer to dimensionless parameter coordinates and whitened sensor
observations. Trial information is additive only for independent trial noise and
declared initial states (e.g. recovered force trials or explicitly simulated
fatigue/recovery blocks). A selected list is not a continuous fatigue schedule:
recompute candidates after each actual experiment and posterior update.
"""

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


def scaled_sensitivities(
    predict: Callable[[np.ndarray], np.ndarray],
    parameters: np.ndarray,
    scales: np.ndarray,
    noise_std: float | np.ndarray = 1.0,
    *,
    covariance: np.ndarray | None = None,
    step: float = 1e-4,
) -> np.ndarray:
    """Central differences d(y/sigma)/d(p/scales), including signed parameters.

    ``predict`` must return consistent observation ordering. Full covariance can
    model temporal/channel correlations; its units are sensor units squared.
    Compare derivatives at step and step/2 before accepting stiff-model designs.
    Bounds and threshold crossings must be handled by the candidate generator.
    """
    p, scale = np.asarray(parameters, float), np.asarray(scales, float)
    if p.ndim != 1 or scale.shape != p.shape or not np.all(np.isfinite(p)):
        raise ValueError("parameters/scales must be finite vectors of equal length")
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0) or not np.isfinite(step) or step <= 0:
        raise ValueError("scales and difference step must be strictly positive")
    baseline = np.asarray(predict(p), float).ravel()
    columns = []
    for j in range(p.size):
        delta = np.zeros_like(p)
        delta[j] = step * scale[j]
        high, low = np.asarray(predict(p + delta), float).ravel(), np.asarray(predict(p - delta), float).ravel()
        if high.shape != baseline.shape or low.shape != baseline.shape:
            raise ValueError("prediction shape changes under parameter perturbation")
        columns.append((high - low) / (2 * step))
    jac = np.column_stack(columns)
    if not np.all(np.isfinite(jac)) or not np.all(np.isfinite(baseline)):
        raise ValueError("non-finite prediction or sensitivity")
    if covariance is not None:
        cov = np.asarray(covariance, float)
        if cov.shape != (jac.shape[0], jac.shape[0]) or not np.all(np.isfinite(cov)) or not np.allclose(cov, cov.T):
            raise ValueError("covariance must be finite, symmetric and match observations")
        return np.linalg.solve(np.linalg.cholesky(cov), jac)
    sigma = np.broadcast_to(np.asarray(noise_std, float), baseline.shape)
    if np.any(sigma <= 0) or not np.all(np.isfinite(sigma)):
        raise ValueError("noise standard deviations must be finite and positive")
    return jac / sigma[:, None]


def fisher_information(sensitivities: np.ndarray, nuisance: np.ndarray | None = None) -> np.ndarray:
    """Project out unconstrained per-trial nuisance columns before forming J'J.

    Examples are an unknown trial force offset or gain. Shared nuisance parameters
    must instead be included in the full cross-trial parameter vector; projecting
    them independently would discard information recoverable across trials.
    """
    jac = np.asarray(sensitivities, float)
    if jac.ndim != 2 or not np.all(np.isfinite(jac)):
        raise ValueError("sensitivities must be a finite observation x parameter matrix")
    if nuisance is not None:
        extra = np.asarray(nuisance, float)
        if extra.ndim != 2 or extra.shape[0] != jac.shape[0] or not np.all(np.isfinite(extra)):
            raise ValueError("nuisance must have one finite row per whitened observation")
        jac = jac - extra @ np.linalg.lstsq(extra, jac, rcond=None)[0]
    return jac.T @ jac


def information_diagnostics(information: np.ndarray, relative_tolerance: float = 1e-8) -> dict:
    """Data-only local rank, weakest direction and CRLB in scaled coordinates.

    A regularizing prior must not be passed to claim data identifiability.
    Rank is local/practical, not a proof of structural/global identifiability.
    """
    matrix = _validated_information(information)
    values, vectors = np.linalg.eigh(matrix)
    if not np.isfinite(relative_tolerance) or relative_tolerance <= 0:
        raise ValueError("relative_tolerance must be positive")
    threshold = relative_tolerance * max(float(values[-1]), np.finfo(float).tiny)
    rank = int(np.count_nonzero(values > threshold))
    full_rank = rank == matrix.shape[0]
    return {
        "parameter_count": matrix.shape[0],
        "rank": rank,
        "full_rank": full_rank,
        "eigenvalues": values.tolist(),
        "weakest_direction": vectors[:, 0].tolist(),
        "condition_number": float(values[-1] / values[0]) if full_rank else None,
        "scaled_crlb_std": np.sqrt(np.maximum(np.diag(np.linalg.inv(matrix)), 0)).tolist() if full_rank else None,
        "relative_rank_tolerance": relative_tolerance,
    }


def _validated_information(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("information must be a nonempty square matrix")
    if not np.all(np.isfinite(matrix)) or not np.allclose(matrix, matrix.T, atol=1e-10):
        raise ValueError("information must be finite and symmetric")
    if np.linalg.eigvalsh(matrix)[0] < -1e-9 * max(1.0, np.linalg.norm(matrix, 2)):
        raise ValueError("information must be positive semidefinite")
    return (matrix + matrix.T) / 2


@dataclass(frozen=True)
class TrialInformation:
    """A candidate's Fisher matrices at matching parameter-ensemble members.

    Duration includes recovery/overhead. Fatigue cost is a declared conservative
    budget metric, not a clinical limit or an additive physiological state.
    """

    name: str
    information: np.ndarray
    duration_s: float
    fatigue_cost: float = 0.0
    charge_cost: float = 0.0
    admissible: bool = True


def select_trials(
    candidates: Sequence[TrialInformation],
    count: int,
    *,
    prior_precision: np.ndarray | None = None,
    ridge: float = 1e-8,
    duration_budget_s: float = np.inf,
    fatigue_budget: float = np.inf,
    charge_budget: float = np.inf,
    fatigue_penalty: float = 0.0,
    charge_penalty: float = 0.0,
    time_normalized: bool = True,
) -> dict:
    """Greedy ensemble D-optimal allocation without replacement.

    Score = mean posterior logdet gain minus costs, optionally per elapsed second.
    The ensemble-average criterion reduces dependence on one nominal parameter
    set; it is not a worst-case guarantee. All matrices use identical parameter
    ordering/scales and ensemble ordering. Rank diagnostics exclude prior/ridge.
    ``admissible`` reflects externally defined experimental limits.
    """
    if not isinstance(count, int) or count < 0 or not candidates:
        raise ValueError("count must be nonnegative and candidates nonempty")
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    budgets = np.asarray([duration_budget_s, fatigue_budget, charge_budget], float)
    penalties = np.asarray([fatigue_penalty, charge_penalty], float)
    if np.any(np.isnan(budgets)) or np.any(budgets < 0) or np.any(~np.isfinite(penalties)) or np.any(penalties < 0):
        raise ValueError("budgets/penalties must be nonnegative")
    matrices, costs = [], []
    names = [candidate.name for candidate in candidates]
    if len(set(names)) != len(names):
        raise ValueError("candidate names must be unique")
    for candidate in candidates:
        value = np.asarray(candidate.information, float)
        if value.ndim == 2:
            value = value[None, :, :]
        if value.ndim != 3 or value.shape[0] == 0:
            raise ValueError("information must be matrix or ensemble x parameter x parameter")
        value = np.stack([_validated_information(member) for member in value])
        if matrices and value.shape != matrices[0].shape:
            raise ValueError("all candidates must share ensemble and parameter dimensions")
        cost = np.asarray([candidate.duration_s, candidate.fatigue_cost, candidate.charge_cost], float)
        if np.any(~np.isfinite(cost)) or cost[0] <= 0 or np.any(cost[1:] < 0):
            raise ValueError("duration must be positive; all trial costs finite and nonnegative")
        matrices.append(value)
        costs.append(cost)
    dimension = matrices[0].shape[-1]
    prior = np.zeros((dimension, dimension)) if prior_precision is None else _validated_information(prior_precision)
    if prior.shape != (dimension, dimension):
        raise ValueError("prior dimension does not match target parameters")
    data = np.zeros_like(matrices[0])
    regularizer = prior + ridge * np.eye(dimension)
    spent = np.zeros(3)
    selected, remaining = [], list(range(len(candidates)))
    for _ in range(count):
        base = np.linalg.slogdet(data + regularizer)[1]
        choices = []
        for i in remaining:
            if not candidates[i].admissible or np.any(spent + costs[i] > budgets):
                continue
            gain = float(np.mean(np.linalg.slogdet(data + matrices[i] + regularizer)[1] - base))
            score = gain - fatigue_penalty * costs[i][1] - charge_penalty * costs[i][2]
            if time_normalized:
                score /= costs[i][0]
            choices.append((score, -i, gain))
        if not choices:
            break
        score, minus_i, gain = max(choices)
        if score <= 0:
            break
        i = -minus_i
        data += matrices[i]
        spent += costs[i]
        remaining.remove(i)
        selected.append({"name": names[i], "score": score, "mean_logdet_gain": gain})
    return {
        "criterion": "ensemble_mean_logdet_gain_per_second" if time_normalized else "ensemble_mean_logdet_gain",
        "selected": selected,
        "spent": dict(zip(("duration_s", "fatigue_cost", "charge_cost"), spent.tolist())),
        "data_information": data.tolist(),
        "data_diagnostics": [information_diagnostics(member) for member in data],
        "regularization_ridge": ridge,
        "uses_prior": prior_precision is not None,
        "requires_sequential_replanning": True,
    }
