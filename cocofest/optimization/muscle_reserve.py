r"""Dimensionless, differentiable proxies for terminal muscle reserve.

The quantities in this module deliberately depend only on the fatigue-capacity
states ``A`` and their individual resting scales ``A_scale``.  They therefore
do not require a full-horizon solution, a muscle-specific hand-tuned weight, or
an assumed future force-sharing policy.  They are *reserve proxies*, not a
prediction of endurance: a later rollout must additionally account for the
torque demand and the muscles' force-generation capacities.

The normalized soft minimum is used as the aggregate reserve:

.. math::

    R_\tau(r) = -\tau \log\left(\frac{1}{n}
    \sum_i \exp(-r_i/\tau)\right), \qquad r_i=A_i/A_{scale,i}.

It approaches the least available muscle capacity as ``temperature`` tends to
zero, treats muscles symmetrically, and stays exactly equal to ``r`` if all
muscles have the same capacity ratio.  At finite temperature it is an
*optimistic* approximation:

.. math::

    0 \le R_\tau(r)-\min_i(r_i) \le \tau\log(n).

The associated penalty is ``1-R``.  Scientific reporting must retain the hard
minimum and the smoothing gap alongside this optimization surrogate.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def _as_capacity_vector(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    """Return a finite, non-empty one-dimensional float vector."""
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    return vector


def _validate_temperature(temperature: float) -> float:
    """Validate the dimensionless soft-min temperature."""
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and strictly positive.")
    return temperature


def capacity_ratios(
    capacities: Sequence[float] | np.ndarray,
    capacity_scales: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Normalize fatigue capacities by their strictly positive resting scales."""
    capacities = _as_capacity_vector(capacities, name="capacities")
    capacity_scales = _as_capacity_vector(capacity_scales, name="capacity_scales")
    if capacities.shape != capacity_scales.shape:
        raise ValueError("capacities and capacity_scales must have the same shape.")
    if np.any(capacity_scales <= 0.0):
        raise ValueError("capacity_scales must be strictly positive.")
    return capacities / capacity_scales


DEFAULT_SMOOTH_MIN_TEMPERATURE = 0.005


@dataclass(frozen=True)
class CapacityReserveMetrics:
    """Hard and smooth reserve values intended for scientific reporting."""

    minimum_ratio: float
    smooth_minimum_ratio: float
    optimism_gap: float
    maximum_optimism_gap: float
    penalty: float


def smooth_minimum_optimism_bound(number_of_muscles: int, temperature: float) -> float:
    """Return the exact upper bound ``temperature * log(number_of_muscles)``."""
    if not isinstance(number_of_muscles, (int, np.integer)) or number_of_muscles <= 0:
        raise ValueError("number_of_muscles must be a strictly positive integer.")
    return _validate_temperature(temperature) * math.log(number_of_muscles)


def smooth_minimum_capacity_ratio(
    ratios: Sequence[float] | np.ndarray,
    *,
    temperature: float = DEFAULT_SMOOTH_MIN_TEMPERATURE,
) -> float:
    """Return a numerically stable, normalized smooth minimum of ``ratios``.

    Normalizing the log-sum-exp by the number of muscles preserves the reserve
    when the complete muscle population is duplicated.
    """
    ratios = _as_capacity_vector(ratios, name="ratios")
    temperature = _validate_temperature(temperature)
    scaled_negated_ratios = -ratios / temperature
    # np.logaddexp.reduce is a stable log-sum-exp implementation even when a
    # direct exp() would overflow.
    log_mean_exp = np.logaddexp.reduce(scaled_negated_ratios) - math.log(ratios.size)
    return float(-temperature * log_mean_exp)


def smooth_minimum_capacity_penalty(
    capacities: Sequence[float] | np.ndarray,
    capacity_scales: Sequence[float] | np.ndarray,
    *,
    temperature: float = DEFAULT_SMOOTH_MIN_TEMPERATURE,
) -> float:
    """Return ``1 - R_tau(A / A_scale)`` without muscle-specific weights.

    A larger value means a smaller aggregate terminal reserve. ``R_tau`` is an
    optimistic smooth approximation of the hard minimum, not the exact worst
    muscle reserve. The output is dimensionless and is intended as a terminal
    objective contribution.
    """
    ratios = capacity_ratios(capacities, capacity_scales)
    return 1.0 - smooth_minimum_capacity_ratio(ratios, temperature=temperature)


def capacity_reserve_metrics(
    capacities: Sequence[float] | np.ndarray,
    capacity_scales: Sequence[float] | np.ndarray,
    *,
    temperature: float = DEFAULT_SMOOTH_MIN_TEMPERATURE,
) -> CapacityReserveMetrics:
    """Return the smooth surrogate together with its mandatory hard-min audit."""
    ratios = capacity_ratios(capacities, capacity_scales)
    smooth_minimum = smooth_minimum_capacity_ratio(ratios, temperature=temperature)
    hard_minimum = float(np.min(ratios))
    return CapacityReserveMetrics(
        minimum_ratio=hard_minimum,
        smooth_minimum_ratio=smooth_minimum,
        optimism_gap=smooth_minimum - hard_minimum,
        maximum_optimism_gap=smooth_minimum_optimism_bound(ratios.size, temperature),
        penalty=1.0 - smooth_minimum,
    )


def physiological_capacity_ratios(
    capacities: Sequence[float] | np.ndarray,
    capacity_scales: Sequence[float] | np.ndarray,
    *,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """Validate and return ratios in the physical interval ``[0, 1]``.

    The optimization primitives intentionally accept arbitrary finite ratios
    for numerical testing. This separate audit is for scientific diagnostics.
    """
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative.")
    ratios = capacity_ratios(capacities, capacity_scales)
    if np.any(ratios < -tolerance) or np.any(ratios > 1.0 + tolerance):
        raise ValueError("capacity ratios must lie in [0, 1] within tolerance.")
    return ratios


def _casadi_module():
    """Import CasADi lazily so NumPy-only users do not require it at import time."""
    import casadi as ca

    return ca


def _casadi_vector(value, *, name: str):
    ca = _casadi_module()
    vector = ca.vec(value)
    if vector.numel() == 0:
        raise ValueError(f"{name} must be non-empty.")
    return vector


def capacity_ratios_casadi(capacities, capacity_scales):
    """CasADi counterpart of :func:`capacity_ratios` for MX, SX, or DM inputs.

    Constant scales are checked eagerly. Symbolic scales must be constrained
    strictly positive by the caller.
    """
    ca = _casadi_module()
    capacities = _casadi_vector(capacities, name="capacities")
    capacity_scales = _casadi_vector(capacity_scales, name="capacity_scales")
    if capacities.numel() != capacity_scales.numel():
        raise ValueError("capacities and capacity_scales must have the same size.")
    if capacity_scales.is_constant():
        constant_scales = np.asarray(ca.DM(capacity_scales), dtype=float).ravel()
        if not np.all(np.isfinite(constant_scales)) or np.any(constant_scales <= 0.0):
            raise ValueError("constant capacity_scales must be finite and strictly positive.")
    return capacities / capacity_scales


def smooth_minimum_capacity_ratio_casadi(
    ratios,
    *,
    temperature: float = DEFAULT_SMOOTH_MIN_TEMPERATURE,
):
    """Differentiable CasADi expression for the normalized smooth minimum."""
    ca = _casadi_module()
    ratios = _casadi_vector(ratios, name="ratios")
    temperature = _validate_temperature(temperature)
    return -temperature * (ca.logsumexp(-ratios / temperature) - math.log(ratios.numel()))


def smooth_minimum_capacity_penalty_casadi(
    capacities,
    capacity_scales,
    *,
    temperature: float = DEFAULT_SMOOTH_MIN_TEMPERATURE,
):
    """Differentiable CasADi expression for :func:`smooth_minimum_capacity_penalty`."""
    ratios = capacity_ratios_casadi(capacities, capacity_scales)
    return 1.0 - smooth_minimum_capacity_ratio_casadi(ratios, temperature=temperature)
