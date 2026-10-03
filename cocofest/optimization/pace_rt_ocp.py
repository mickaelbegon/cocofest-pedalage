"""Fixed-graph terminal reserve-target cost for the asynchronous PACE rollout.

PACE-RT deliberately does *not* maximise an affine local value.  The slow
worker supplies a local prediction of its reduced future-work score around a
nominal next-cycle terminal state.  The fast RHO only penalises falling below
a bounded target and moving too far from that nominal terminal state.  The
model is conditional on the compact rollout and is never an endurance or
feasibility certificate.

The parameter vector is numerical, so accepted supervisor updates reuse the
same SX/IPOPT graph:

``[objective_activation, target, constant, center(n), gradient(n), radius(n),
proximal, shortage, shortage_scale, constraint_activation]``.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
from typing import Mapping, Sequence

import numpy as np

from .task_reserve_ocp import TaskReserveObjectiveBinding, TaskReserveStateCoordinate


PACE_RT_MODE = "terminal_reserve_target_experimental"


def _finite_vector(value, size, name, *, positive=False):
    values = np.asarray(value, dtype=float).reshape(-1)
    if values.shape != (size,) or not np.all(np.isfinite(values)) or (positive and np.any(values <= 0)):
        raise ValueError(f"{name} must be a finite vector of length {size}" + (" with positive entries." if positive else "."))
    return values


class PaceRtObjectiveBinding(TaskReserveObjectiveBinding):
    """A target-seeking, proximal local terminal-value cost.

    Coordinates are currently the capacity ``A`` states normalised by their
    resting capacity.  The compact rollout still propagates Cn/F/A/Tau1/Km;
    Tau1/Km memory offsets are held fixed while fitting this first low-rank
    model.  This limitation is made explicit in every summary.
    """

    def __init__(self, coordinates: Sequence[TaskReserveStateCoordinate], *, task_context: Mapping,
                 model_sha256: str, target_fraction: float = .25, smoothing: float = 1e-3,
                 maximum_age_cycles: int = 20, normalize_shortage: bool = False,
                 enforce_target_constraint: bool = False):
        if not math.isfinite(target_fraction) or not 0 <= float(target_fraction) <= 1:
            raise ValueError("target_fraction must lie in [0, 1].")
        super().__init__(coordinates, task_context=task_context, model_sha256=model_sha256,
                         target=0., smoothing=smoothing, maximum_age_cycles=maximum_age_cycles)
        self.target_fraction = float(target_fraction)
        if type(normalize_shortage) is not bool:
            raise ValueError("normalize_shortage must be a boolean.")
        self.normalize_shortage = normalize_shortage
        if type(enforce_target_constraint) is not bool:
            raise ValueError("enforce_target_constraint must be a boolean.")
        self.enforce_target_constraint = enforce_target_constraint
        # CasADi evaluates every algebraic subexpression even when activation
        # is zero.  Therefore the inactive bootstrap must still contain a
        # finite, nonzero radius; otherwise ``0 * (delta / 0)**2`` propagates
        # NaN into IPOPT before the first asynchronous update.
        self.values = np.r_[0., 0., 0., np.zeros(self.dimension), np.zeros(self.dimension),
                            np.ones(self.dimension), 0., 1., 1., 0.]
        # Final scalars are the proximal/shortage multipliers and the (strictly
        # positive) shortage scale.  Smoothing remains graph structure so a
        # live update cannot silently change its curvature.
        self.graph_signature_sha256 = sha256(json.dumps({
            "version": "pace-reserve-target-terminal-v2-experimental",
            "coordinate_layout": [dict(state_key=c.state_key, index=c.index, scale=c.scale, offset=c.offset)
                                  for c in self.coordinates],
            "smoothing": self.smoothing, "normalize_shortage": self.normalize_shortage,
            "enforce_target_constraint": self.enforce_target_constraint,
            "parameter_size": int(self.values.size),
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.source_completed_cycles = None
        self.last_model = None

    @property
    def _slices(self):
        n = self.dimension
        return dict(activation=0, target=1, constant=2,
                    center=slice(3, 3 + n), gradient=slice(3 + n, 3 + 2 * n),
                    radius=slice(3 + 2 * n, 3 + 3 * n), proximal=3 + 3 * n,
                    shortage=4 + 3 * n, shortage_scale=5 + 3 * n,
                    constraint_activation=6 + 3 * n)

    def _prediction(self, controller):
        """Return the local terminal value and its fixed numerical parameters."""
        from casadi import vertcat

        point = vertcat(*[
            (controller.states[c.state_key].cx[c.index] - c.offset) / c.scale
            for c in self.coordinates
        ])
        p = controller.parameters["rho_task_reserve"].cx
        s = self._slices
        prediction = p[s["constant"]]
        for index in range(self.dimension):
            prediction += p[s["gradient"]][index] * (point[index] - p[s["center"]][index])
        return point, prediction, p, s

    def objective(self, controller):
        from casadi import sqrt

        point, prediction, p, s = self._prediction(controller)
        proximity = 0
        for index in range(self.dimension):
            delta = point[index] - p[s["center"]][index]
            proximity += (delta / p[s["radius"]][index]) ** 2
        # Scaling by the locally attainable decrease makes the shortage
        # curvature commensurate with the dimensionless proximity term.  The
        # smoothing constant is scaled too, preserving the smooth-positive
        # shape.  ``shortage_scale`` is always finite and positive, including
        # before the first asynchronous update.
        shortage_scale = p[s["shortage_scale"]] if self.normalize_shortage else 1.
        shortage = (prediction - p[s["target"]]) / shortage_scale
        smoothing = self.smoothing / shortage_scale
        smooth_positive = .5 * (shortage + sqrt(shortage * shortage + smoothing * smoothing))
        return p[s["activation"]] * (
            p[s["shortage"]] * smooth_positive * smooth_positive + p[s["proximal"]] * proximity
        )

    def target_constraint(self, controller):
        """A compiled terminal target constraint, inactive when its activation is zero.

        The residual is constrained to be at most zero.  This is deliberately
        separate from the objective activation so a subsequent two-solve
        protocol can minimise fatigue while retaining a numerically updated
        reserve target.  It is experimental until its target is obtained from
        an actual first RHO solve rather than a coordinate-box surrogate.
        """
        _, prediction, p, s = self._prediction(controller)
        return p[s["constraint_activation"]] * (prediction - p[s["target"]])

    def update_from_rollout(self, nmpc, *, local_fit: Mapping, source_completed_cycles: int,
                            completed_cycles: int, proximal_weight: float, shortage_weight: float = 1.,
                            objective_activation: float = 1., constraint_activation: float | None = None):
        """Install a validated compact-rollout fit without rebuilding the NLP."""
        if type(source_completed_cycles) is not int or type(completed_cycles) is not int:
            raise ValueError("PACE-RT cycle indices must be integers.")
        if completed_cycles < source_completed_cycles or completed_cycles - source_completed_cycles > self.maximum_age_cycles:
            raise ValueError("PACE-RT rollout is stale or belongs to a future boundary.")
        if not isinstance(local_fit, Mapping) or local_fit.get("accepted") is not True:
            raise ValueError("PACE-RT requires an accepted local rollout fit.")
        n = self.dimension
        reference = np.asarray(local_fit.get("reference_state"), dtype=float)
        scales = np.asarray(local_fit.get("state_scales"), dtype=float)
        gradient_3 = np.asarray(local_fit.get("gradient"), dtype=float)
        if reference.shape != (n, 3) or scales.shape != (n, 3) or gradient_3.shape != (n, 3):
            raise ValueError("PACE-RT local fit must contain one [A,Tau1,Km] row per muscle.")
        if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
            raise ValueError("PACE-RT local fit state reference/scales are invalid.")
        # The PACE-VR finite difference is in A/a_rest.  The other slow
        # coordinates are propagated by the rollout but not independently
        # identified by this low-rank fit, hence their gradients are refused
        # instead of accidentally assigning them a different normalization.
        if not np.allclose(gradient_3[:, 1:], 0., atol=1e-12, rtol=0):
            raise ValueError("PACE-RT v1 accepts only the explicitly identified A-state gradient.")
        declared_scales = np.asarray([coordinate.scale for coordinate in self.coordinates], dtype=float)
        if not np.allclose(scales[:, 0], declared_scales, rtol=0., atol=1e-12):
            raise ValueError("PACE-RT rollout A scales differ from the compiled terminal coordinates.")
        center = reference[:, 0] / scales[:, 0]
        gradient = gradient_3[:, 0]
        trust = float(local_fit.get("trust_bounds", [np.nan, np.nan])[1])
        radius = np.full(n, trust)
        constant = float(local_fit.get("constant", np.nan))
        proximal_weight = float(proximal_weight)
        shortage_weight = float(shortage_weight)
        objective_activation = float(objective_activation)
        if constraint_activation is None:
            constraint_activation = 1. if self.enforce_target_constraint else 0.
        constraint_activation = float(constraint_activation)
        if (not math.isfinite(constant) or not math.isfinite(trust) or trust <= 0
                or not math.isfinite(proximal_weight) or proximal_weight < 0
                or not math.isfinite(shortage_weight) or shortage_weight < 0
                or not math.isfinite(objective_activation) or not 0 <= objective_activation <= 1
                or not math.isfinite(constraint_activation) or not 0 <= constraint_activation <= 1):
            raise ValueError("PACE-RT fit target, trust radius or objective weights are invalid.")
        # The predicted best decrease inside the announced coordinate box is
        # bounded.  Asking for only a fraction avoids driving a linear model
        # to the box edge, while the proximal term absorbs residual curvature.
        attainable_decrease = float(np.sum(np.abs(gradient) * radius))
        target = constant - self.target_fraction * attainable_decrease
        # The lower bound prevents an uninformative zero-gradient local fit
        # from creating a singular objective.  The scale is a numerical
        # property of this specific fit, not a new RHO decision variable.
        shortage_scale = max(attainable_decrease, self.smoothing) if self.normalize_shortage else 1.
        values = np.r_[1., target, constant, center, gradient, radius, proximal_weight,
                       shortage_weight, shortage_scale, constraint_activation]
        values[0] = objective_activation
        self._write(nmpc, values)
        self.source_completed_cycles = source_completed_cycles
        self.last_model = {
            "constant": constant, "target": target, "center": center.tolist(),
            "gradient": gradient.tolist(), "radius": radius.tolist(),
            "proximity_weight": proximal_weight, "shortage_weight": shortage_weight,
            "shortage_scale": shortage_scale, "shortage_normalized": self.normalize_shortage,
            "objective_activation": objective_activation, "constraint_activation": constraint_activation,
            "fit_rank": local_fit.get("fit_rank"), "sample_count": local_fit.get("sample_count"),
        }
        return self.summary()

    def activate_fatigue_secondary_stage(self, nmpc, *, reserve_target: float):
        """Keep a compiled reserve target while removing its objective cost.

        This is the numerical transition between the two solves of the
        offline lexicographic experiment.  ``reserve_target`` must be supplied
        by the first, actual RHO solve plus its selected tolerance; this class
        intentionally does not infer it from a coordinate box.
        """
        if not self.enforce_target_constraint or self.last_model is None:
            raise ValueError("A fitted PACE-RT target constraint is required before entering the fatigue-secondary stage.")
        reserve_target = float(reserve_target)
        if not math.isfinite(reserve_target):
            raise ValueError("reserve_target must be finite.")
        values = self.values.copy()
        values[self._slices["activation"]] = 0.
        values[self._slices["target"]] = reserve_target
        values[self._slices["constraint_activation"]] = 1.
        self._write(nmpc, values)
        self.last_model = {**self.last_model, "target": reserve_target,
                           "objective_activation": 0., "constraint_activation": 1.}
        return {**self.summary(), "priority_stage": "fatigue_secondary"}

    def validate_terminal_point(self, coordinates, *, completed_cycles: int):
        if (self.values[0] == 0 and self.values[self._slices["constraint_activation"]] == 0) or self.last_model is None:
            return {"active": False, "terminal_trust_validated": False}
        point = _finite_vector(coordinates, self.dimension, "PACE-RT terminal coordinates")
        center, radius = np.asarray(self.last_model["center"]), np.asarray(self.last_model["radius"])
        age = int(completed_cycles) - int(self.source_completed_cycles)
        fraction = np.abs(point - center) / radius
        return {"active": True, "terminal_trust_validated": bool(age >= 0 and age <= self.maximum_age_cycles
                and np.all(fraction <= 1.)), "maximum_trust_fraction": float(np.max(fraction)),
                "source_age_cycles": age,
                "predicted_terminal_value": float(self.last_model["constant"] + np.dot(
                    np.asarray(self.last_model["gradient"]), point - center))}

    def summary(self):
        base = super().summary()
        return {**base, "mode": PACE_RT_MODE, "objective": "smooth_reserve_shortage_plus_terminal_proximity",
                "target_fraction": self.target_fraction,
                "shortage_normalized_by_local_attainable_decrease": self.normalize_shortage,
                "terminal_target_constraint_compiled": self.enforce_target_constraint,
                "identified_terminal_coordinates": "A/a_rest only; Cn/F/A/Tau1/Km remain in compact rollout",
                "last_model": self.last_model}
