"""Local polynomial of a compact future muscle policy, sampled outside the NLP.

This is a conditional policy value, not a viability or endurance certificate.
The task, phase, calcium regime and exact fatigue offsets are fixed context.
Finite differences and held-out checks cannot certify the entire trust box.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
from scipy.special import logsumexp

from .compact_muscle_prediction import (
    CompactMusclePredictor,
    fatigue_memory_coordinates,
    reconstruct_fatigue_memory,
)


def _vector(value, size, name, *, positive=False):
    result = np.broadcast_to(np.asarray(value, dtype=float), (size,)).copy()
    if not np.all(np.isfinite(result)) or (positive and np.any(result <= 0)):
        raise ValueError(f"{name} must be finite" + (" and strictly positive." if positive else "."))
    return result


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and strictly positive.")
    return value


@dataclass(frozen=True)
class LocalEnduranceCoordinates:
    """Order: all muscle damages, then all forces divided by force_scale.

    ``offsets`` are the two exact independent Tau1/Km memory coordinates,
    evaluated at the anchor boundary, not projected to their resting manifold.
    """

    parameters: tuple
    anchor: np.ndarray
    fixed_cn: np.ndarray
    offsets: np.ndarray
    force_scale: np.ndarray

    @property
    def context_signature(self):
        """Stable digest for binding a fitted value to its exact coordinate context."""
        payload = {"parameters": [asdict(p) for p in self.parameters],
                   "anchor": self.anchor.tolist(), "fixed_cn": self.fixed_cn.tolist(),
                   "offsets": self.offsets.tolist(), "force_scale": self.force_scale.tolist()}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False,
                                         separators=(",", ":")).encode("utf-8")).hexdigest()

    @classmethod
    def from_state(cls, anchor_states, parameters, *, force_scale):
        parameters = tuple(parameters)
        count = len(parameters)
        state = np.asarray(anchor_states, dtype=float)
        if (not count or state.shape != (count, 5) or not np.all(np.isfinite(state))
                or np.any(state[:, :2] < 0) or np.any(state[:, 2:] <= 0)):
            raise ValueError("Anchor requires nonnegative Cn/F and positive A/Tau1/Km for each muscle.")
        scale = _vector(force_scale, count, "force_scale", positive=True)
        reduced = [fatigue_memory_coordinates(s[2:], p.fatigue) for s, p in zip(state, parameters)]
        return cls(parameters, np.r_[[r[0] for r in reduced], state[:, 1] / scale],
                   state[:, 0].copy(), np.asarray([r[1] for r in reduced]), scale)

    def decode(self, coordinates):
        count = len(self.parameters)
        xi = np.asarray(coordinates, dtype=float)
        if xi.shape != (2 * count,) or not np.all(np.isfinite(xi)):
            raise ValueError("coordinates must be a finite 2M vector.")
        slow = np.asarray([reconstruct_fatigue_memory(d, e, p.fatigue)
                           for d, e, p in zip(xi[:count], self.offsets, self.parameters)])
        state = np.column_stack((self.fixed_cn, xi[count:] * self.force_scale, slow))
        if np.any(state[:, :2] < 0) or np.any(state[:, 2:] <= 0) or not np.all(np.isfinite(state)):
            raise ValueError("Decoded state lies outside the physical predictor domain.")
        return state

    def encode(self, states, *, context_tolerance=1e-10):
        """Reject changed calcium/offset context rather than silently discarding it."""
        tolerance = _positive(context_tolerance, "context_tolerance")
        other = self.from_state(states, self.parameters, force_scale=self.force_scale)
        if (not np.allclose(other.fixed_cn, self.fixed_cn, rtol=0, atol=tolerance)
                or not np.allclose(other.offsets, self.offsets, rtol=0, atol=tolerance)):
            raise ValueError("State calcium or fatigue offsets differ from fixed local context.")
        return other.anchor


@dataclass(frozen=True)
class OracleEvaluation:
    status: str
    value: float | None
    minimum_signed_margin: float | None
    soft_minimum_margin: float | None
    completed_intervals: int
    message: str = ""


class CompactEnduranceValueOracle:
    """Adaptive rollout scored by both signed distances to total-moment bounds.

    For each phase, attainable signed total moment lies in [sum lower,
    sum upper]. Both distances to the requested total enter a conservative
    soft minimum, -temperature * logsumexp(-margins / temperature).
    Thus the smoothing bias depends on the fixed horizon/phase count; values
    computed with different settings are not directly comparable. A stable
    softplus penalizes margins below margin_target. PW saturation is not the
    score. Any failed policy rollout has no usable scalar value.
    """

    def __init__(self, predictor: CompactMusclePredictor, coordinates: LocalEnduranceCoordinates,
                 *, moment_scale, horizon_cycles=10, softmin_temperature=0.05,
                 margin_target=0.1, penalty_temperature=0.05, moment_tolerance=1e-8):
        if predictor.parameters != coordinates.parameters:
            raise ValueError("Predictor and coordinates must use identical muscle parameters.")
        if isinstance(horizon_cycles, bool) or int(horizon_cycles) != horizon_cycles or horizon_cycles < 1:
            raise ValueError("horizon_cycles must be a positive integer.")
        self.predictor = predictor
        self.coordinates = coordinates
        self.horizon_cycles = int(horizon_cycles)
        self.moment_scale = _positive(moment_scale, "moment_scale")
        self.softmin_temperature = _positive(softmin_temperature, "softmin_temperature")
        self.penalty_temperature = _positive(penalty_temperature, "penalty_temperature")
        self.moment_tolerance = _positive(moment_tolerance, "moment_tolerance")
        self.margin_target = float(margin_target)
        if not np.isfinite(self.margin_target):
            raise ValueError("margin_target must be finite.")

    def evaluate(self, xi):
        try:
            state = self.coordinates.decode(xi)
            rollout = self.predictor.rollout(state, horizon_cycles=self.horizon_cycles,
                                             moment_tolerance=self.moment_tolerance)
            if rollout.status != "complete":
                return OracleEvaluation("policy_failed", None, None, None,
                                        rollout.completed_intervals, str(rollout.first_failure))
            margins = []
            for step, state in enumerate(rollout.state_history[:-1]):
                index = step % len(self.predictor.intervals)
                interval = self.predictor.intervals[index]
                transition = self.predictor.phase_map(state, index)
                # An invalid maximal-recruitment slow state cannot count as
                # available reserve, even if the chosen lower-recruitment
                # policy succeeds. Conservatively reject this value instead
                # of claiming global infeasibility or counting invalid force.
                # This guard is at phase endpoints, not a continuous-time proof.
                for endpoint in (transition.intercept, transition.endpoint(transition.maximum_recruitment)):
                    if (not np.all(np.isfinite(endpoint)) or np.any(endpoint[:, :2] < 0)
                            or np.any(endpoint[:, 2:] <= 0)):
                        return OracleEvaluation("envelope_domain_invalid", None, None, None,
                                                rollout.completed_intervals,
                                                f"Attainable-envelope endpoint outside domain at step {step}.")
                coefficients = np.asarray(interval.moment_coefficients)
                moment0 = coefficients * transition.intercept[:, 1]
                moment1 = moment0 + coefficients * transition.slope[:, 1] * transition.maximum_recruitment
                required = float(np.sum(interval.target_moments))
                lower = float(np.minimum(moment0, moment1).sum())
                upper = float(np.maximum(moment0, moment1).sum())
                margins.extend(((required - lower) / self.moment_scale,
                                (upper - required) / self.moment_scale))
            margins = np.asarray(margins)
            if not np.all(np.isfinite(margins)):
                raise ValueError("Nonfinite attainable moment margins.")
            soft_minimum = -self.softmin_temperature * logsumexp(-margins / self.softmin_temperature)
            value = self.penalty_temperature * np.logaddexp(
                0., (self.margin_target - soft_minimum) / self.penalty_temperature
            )
            if not np.isfinite(value):
                raise ValueError("Nonfinite margin penalty.")
            return OracleEvaluation("complete", float(value), float(margins.min()),
                                    float(soft_minimum), rollout.completed_intervals)
        except (ValueError, FloatingPointError, OverflowError) as error:
            return OracleEvaluation("domain_invalid", None, None, None, 0, str(error))


@dataclass(frozen=True)
class LocalEndurancePolynomial:
    center: np.ndarray
    constant: float
    gradient: np.ndarray
    diagonal_hessian: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    finite_difference_step: np.ndarray

    @property
    def coefficients(self):
        """Fixed size 1 + 4M, including zero curvature for a linear model."""
        return np.r_[self.constant, self.gradient, self.diagonal_hessian]

    def evaluate(self, xi):
        xi = np.asarray(xi, dtype=float)
        if (xi.shape != self.center.shape or not np.all(np.isfinite(xi))
                or np.any(xi < self.lower_bounds) or np.any(xi > self.upper_bounds)):
            raise ValueError("Polynomial evaluation lies outside its explicit trust box.")
        delta = xi - self.center
        return float(self.constant + self.gradient @ delta + 0.5 * self.diagonal_hessian @ (delta * delta))


@dataclass(frozen=True)
class LocalEnduranceAudit:
    accepted: bool
    reason: str
    heldout_count: int
    maximum_absolute_error: float | None = None
    maximum_scaled_error: float | None = None
    ranking_pairs: int = 0
    ranking_failures: int = 0
    records: tuple[dict, ...] = ()


@dataclass(frozen=True)
class LocalEnduranceFit:
    accepted: bool
    model: LocalEndurancePolynomial | None
    audit: LocalEnduranceAudit
    metadata: dict


def audit_local_endurance_value(model, oracle, heldout_points, *, absolute_tolerance=1e-3,
                               relative_tolerance=0.05, ranking_tolerance=1e-7):
    """Independent residual and pairwise directional ranking gate.

    Held-out points must differ from the anchor and every central-difference
    training point. At least one oracle pair must differ by ranking_tolerance;
    an uninformative flat audit fails closed. Every informative pair must have
    the same strict ordering under the polynomial. This checks local policy
    ranking only and does not validate compact dynamics against full Ding.
    """
    absolute_tolerance = _positive(absolute_tolerance, "absolute_tolerance")
    relative_tolerance = float(relative_tolerance)
    ranking_tolerance = _positive(ranking_tolerance, "ranking_tolerance")
    if not np.isfinite(relative_tolerance) or relative_tolerance < 0:
        raise ValueError("relative_tolerance must be finite and nonnegative.")
    points = np.asarray(heldout_points, dtype=float)
    count = len(points) if points.ndim else 0
    if points.ndim != 2 or points.shape[1] != model.center.size or count < 2:
        return LocalEnduranceAudit(False, "At least two independent held-out points are required.", count)
    training = np.r_[model.center[None, :], model.center + np.diag(model.finite_difference_step),
                     model.center - np.diag(model.finite_difference_step)]
    actual, predicted, records = [], [], []
    for point in points:
        if np.any(np.all(np.isclose(training, point, rtol=0, atol=1e-13), axis=1)):
            return LocalEnduranceAudit(False, "Held-out point overlaps a fitted sample.", count)
        try:
            predicted.append(model.evaluate(point))
        except ValueError as error:
            return LocalEnduranceAudit(False, str(error), count)
        result = oracle.evaluate(point)
        records.append({"coordinates": point.tolist(), "prediction": predicted[-1],
                        "value": result.value, "status": result.status,
                        "minimum_signed_margin": result.minimum_signed_margin,
                        "soft_minimum_margin": result.soft_minimum_margin})
        if result.status != "complete" or result.value is None or not np.isfinite(result.value):
            return LocalEnduranceAudit(False, f"Held-out oracle {result.status}: {result.message}", count,
                                       records=tuple(records))
        actual.append(result.value)
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    error = np.abs(actual - predicted)
    scaled = error / (absolute_tolerance + relative_tolerance * np.abs(actual))
    pairs = failures = 0
    for i in range(count):
        for j in range(i):
            difference = actual[i] - actual[j]
            if abs(difference) > ranking_tolerance:
                pairs += 1
                failures += int((predicted[i] - predicted[j]) * difference <= 0)
    accepted = bool(np.all(scaled <= 1) and pairs > 0 and failures == 0)
    reason = ("accepted" if accepted else "Held-out residual, directional ranking, or informativeness gate failed.")
    return LocalEnduranceAudit(accepted, reason, count, float(error.max()), float(scaled.max()),
                               pairs, failures, tuple(records))


def fit_local_endurance_value(oracle, *, trust_radius, fd_step=None, kind="linear",
                             heldout_points=None, absolute_tolerance=1e-3,
                             relative_tolerance=0.05, ranking_tolerance=1e-7):
    """Fit central differences and expose a model only after independent audit.

    Default FD steps are one quarter of each trust radius. Default held-outs
    are every +/- trust-axis endpoint plus four full-radius joint directions,
    the gradient direction and its linear-box optimum. No boundary clipping, domain projection or infeasibility
    penalties are used to manufacture gradients. All 2M radii must be positive;
    anchors with zero force may therefore require another parameterization.
    """
    if kind not in ("linear", "diagonal_quadratic"):
        raise ValueError("kind must be linear or diagonal_quadratic.")
    center = oracle.coordinates.anchor.copy()
    size = center.size
    radius = _vector(trust_radius, size, "trust_radius", positive=True)
    step = radius / 4 if fd_step is None else _vector(fd_step, size, "fd_step", positive=True)
    if np.any(step > radius):
        raise ValueError("fd_step must lie within trust_radius.")
    metadata = {
        "kind": kind, "coordinate_order": "all damage, then all F/force_scale",
        "horizon_cycles": oracle.horizon_cycles, "moment_scale": oracle.moment_scale,
        "softmin_temperature": oracle.softmin_temperature, "margin_target": oracle.margin_target,
        "penalty_temperature": oracle.penalty_temperature, "moment_tolerance": oracle.moment_tolerance,
        "absolute_tolerance": absolute_tolerance, "relative_tolerance": relative_tolerance,
        "ranking_tolerance": ranking_tolerance, "training_evaluations": 0,
        "policy_conditional": True, "whole_box_certified": False, "training_records": [],
        "trust_radius": radius.tolist(), "finite_difference_step": step.tolist(),
        "coordinate_context_sha256": oracle.coordinates.context_signature,
    }

    def failure(reason):
        return LocalEnduranceFit(False, None, LocalEnduranceAudit(False, reason, 0), metadata)

    # Reconstruction is affine and separable in each coordinate, hence checking
    # both box endpoints suffices for the initial physical domain, not rollout feasibility.
    try:
        oracle.coordinates.decode(center - radius)
        oracle.coordinates.decode(center + radius)
    except ValueError as error:
        return failure(f"Trust domain invalid: {error}")
    values = []
    points = np.r_[center[None, :], center + np.diag(step), center - np.diag(step)]
    for point in points:
        sample = oracle.evaluate(point)
        metadata["training_evaluations"] += 1
        metadata["training_records"].append({"coordinates": point.tolist(), "value": sample.value,
                                             "status": sample.status,
                                             "minimum_signed_margin": sample.minimum_signed_margin})
        if sample.status != "complete" or sample.value is None or not np.isfinite(sample.value):
            return failure(f"Training oracle {sample.status}: {sample.message}")
        values.append(sample.value)
    constant = values[0]
    plus, minus = np.asarray(values[1:1 + size]), np.asarray(values[1 + size:])
    gradient = (plus - minus) / (2 * step)
    curvature = (plus - 2 * constant + minus) / (step * step) if kind == "diagonal_quadratic" else np.zeros(size)
    if not np.all(np.isfinite(np.r_[constant, gradient, curvature])):
        return failure("Nonfinite polynomial coefficients.")
    model = LocalEndurancePolynomial(center, constant, gradient, curvature, center - radius,
                                    center + radius, step)
    if heldout_points is None:
        alternating = np.where(np.arange(size) % 2, -1., 1.)
        directions = np.asarray([np.ones(size), -np.ones(size), alternating, -alternating])
        heldout_points = np.r_[center + np.diag(radius), center - np.diag(radius),
                              center + radius * directions]
        # Audit along the steepest direction in normalized trust coordinates,
        # independently of central-difference samples and mixed sign probes.
        direction = radius * gradient
        if np.max(np.abs(direction)) > 0:
            direction /= np.max(np.abs(direction))
            heldout_points = np.r_[heldout_points, center + 0.75 * radius * direction[None, :],
                                  center - 0.75 * radius * direction[None, :],
                                  center + radius * np.sign(gradient)[None, :],
                                  center - radius * np.sign(gradient)[None, :]]
    audit = audit_local_endurance_value(
        model, oracle, heldout_points, absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance, ranking_tolerance=ranking_tolerance,
    )
    metadata["heldout_evaluations_requested"] = audit.heldout_count
    return LocalEnduranceFit(audit.accepted, model if audit.accepted else None, audit, metadata)
