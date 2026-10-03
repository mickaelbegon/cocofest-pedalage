"""Design reachable, local reserve experiments without solving or fitting an OCP.

The historical fatigue trajectory is a discovery set. It cannot identify how
redistributing future stimulation changes the reserve, even if its curved
trajectory happens to yield a full-rank affine design matrix. This planner
keeps that distinction explicit and never turns unsuccessful NLPs into an
upper bound on attainable work.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .task_reserve import LocalReserveSample, TaskReserveProbe


@dataclass(frozen=True)
class ReserveDesignSample:
    sample_id: str
    sample: LocalReserveSample
    partition: str
    receipt: str


def _positive(value, name):
    if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def observed_grid_report(sample: LocalReserveSample) -> dict:
    """Recompute grid censoring from witnessed constraints, never solver status.

    A non-witnessed factor above the best witness remains undetermined; even
    a non-censored observed maximum is not a measured feasibility boundary.
    """
    estimate = sample.estimate
    scales, feasible = [], []
    for evidence in estimate.evidence:
        scale = _positive(evidence.work_scale, "work_scale")
        if scale in scales:
            raise ValueError("Probe evidence contains duplicate work scales")
        scales.append(scale)
        if evidence.witness_is_valid(TaskReserveProbe(estimate.checkpoint, scale)):
            feasible.append(scale)
    if 1. not in scales:
        raise ValueError("A nominal work probe is required")
    best = max(feasible) if feasible else None
    if (tuple(feasible) != estimate.feasible_work_scales
            or (1. in feasible) != estimate.nominal_task_witnessed
            or best != estimate.work_scale_lower_bound):
        raise ValueError("Reserve summary disagrees with probe evidence")
    return {
        "tested_scales": sorted(scales), "feasible_scales": sorted(feasible),
        "undetermined_scales": sorted(set(scales) - set(feasible)),
        "nominal_witnessed": 1. in feasible,
        "observed_reserve_lower_bound": None if best is None else best - 1.,
        "grid_ceiling_witnessed": max(scales) in feasible,
        "observation_kind": ("nominal_undetermined" if 1. not in feasible
                             else "grid_ceiling_censored" if max(scales) in feasible
                             else "witness_lower_bound_with_undetermined_larger_loads"),
        "maximum_measured": False, "global_upper_bound": None,
    }


def _design_diagnostics(points, center, radii):
    dimension = len(center)
    if not points:
        return {"count": 0, "affine_rank": 0, "required_rank": dimension + 1,
                "condition_number": None, "singular_values": []}
    matrix = np.column_stack((np.ones(len(points)), (np.asarray(points) - center) / radii))
    singular = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.linalg.matrix_rank(matrix))
    condition = float(singular[0] / singular[-1]) if rank == dimension + 1 else None
    return {"count": len(points), "affine_rank": rank, "required_rank": dimension + 1,
            "condition_number": condition, "singular_values": singular.tolist()}


def local_neighborhood(samples: Sequence[ReserveDesignSample], *, anchor_id: str,
                       coordinate_radius: Sequence[float]) -> dict:
    """Select a physical-coordinate trust box without enlarging it to fit data.

    All samples here belong to the existing baseline trajectory. Keeping the
    original train/holdout partition avoids retrospectively selecting a fit
    on its holdout. Neither partition is a confirmatory experiment after its
    results have informed this new design.
    """
    samples = tuple(samples)
    ids = [item.sample_id for item in samples]
    if not samples or len(set(ids)) != len(ids) or anchor_id not in ids:
        raise ValueError("Unique sample ids and an existing anchor are required")
    anchor = samples[ids.index(anchor_id)]
    center, radii = np.asarray(anchor.sample.coordinates, float), np.asarray(coordinate_radius, float)
    if (center.ndim != 1 or not center.size or radii.shape != center.shape
            or not np.all(np.isfinite(center)) or not np.all(np.isfinite(radii)) or np.any(radii <= 0)):
        raise ValueError("Coordinate radius must match finite coordinates and be positive")
    source = anchor.sample.estimate.checkpoint
    selected, excluded, points = [], [], {"train": [], "holdout": []}
    seen = set()
    for item in samples:
        checkpoint = item.sample.estimate.checkpoint
        if (checkpoint.task_context_sha256 != source.task_context_sha256
                or checkpoint.model_sha256 != source.model_sha256):
            raise ValueError("Neighborhood mixes physical contexts or models")
        if checkpoint.prepared_problem_sha256 in seen:
            raise ValueError("Prepared checkpoint cannot be duplicated across samples")
        seen.add(checkpoint.prepared_problem_sha256)
        x = np.asarray(item.sample.coordinates, float)
        if x.shape != center.shape or not np.all(np.isfinite(x)) or item.partition not in points:
            raise ValueError("Invalid coordinates or discovery partition")
        distance = float(np.max(np.abs(x - center) / radii))
        report = observed_grid_report(item.sample)
        if distance > 1. + 1e-12 or not report["nominal_witnessed"]:
            excluded.append({"id": item.sample_id, "normalized_box_distance": distance,
                             "reason": "outside_local_box" if distance > 1. + 1e-12
                             else "nominal_undetermined"})
            continue
        selected.append({"id": item.sample_id, "receipt": item.receipt,
                         "completed_cycles": checkpoint.completed_cycles, "partition": item.partition,
                         "coordinates": x.tolist(), "normalized_box_distance": distance,
                         "grid": report})
        points[item.partition].append(x)
    if not any(row["id"] == anchor_id for row in selected):
        raise ValueError("Anchor must have a nominal feasible witness")
    bounds = [row["grid"]["observed_reserve_lower_bound"] for row in selected]
    return {"anchor_id": anchor_id, "anchor_receipt": anchor.receipt,
            "completed_cycles": source.completed_cycles, "center": center.tolist(),
            "trust_radius": radii.tolist(), "selected": selected, "excluded": excluded,
            "discovery_design": {key: _design_diagnostics(value, center, radii)
                                 for key, value in points.items()},
            "observed_lower_bound_range": [min(bounds), max(bounds)],
            "observed_lower_bound_contrast": max(bounds) - min(bounds),
            "frontier_contrast_demonstrated": False,
            "activation_allowed": False,
            "reason": "baseline_trajectory_does_not_identify_policy_induced_state_directions"}


def common_followup_grid(reports: Sequence[dict], *, refinement_step: float = .025,
                         exploration_ceiling: float = 1.5, extension_step: float = .1,
                         maximum_points: int = 40) -> dict:
    """Predeclare the same complete grid for every new branch endpoint.

    Refine every original interval equally. Extend beyond a witnessed grid
    ceiling up to a finite computational budget. This is an experiment cap,
    never an upper feasibility bound. Failed points are retained and no binary
    search or feasibility monotonicity is assumed.
    """
    refinement_step = _positive(refinement_step, "refinement_step")
    extension_step = _positive(extension_step, "extension_step")
    exploration_ceiling = _positive(exploration_ceiling, "exploration_ceiling")
    if type(maximum_points) is not int or maximum_points < 2 or not reports:
        raise ValueError("Nonempty reports and a point budget >=2 are required")
    old_grids = [tuple(report["tested_scales"]) for report in reports]
    if any(grid != old_grids[0] for grid in old_grids):
        raise ValueError("Discovery reports must use the same grid")
    original = old_grids[0]
    if not original or original[0] != 1. or exploration_ceiling < original[-1]:
        raise ValueError("Planning requires work factors >=1 and a ceiling >= the old ceiling")
    planned = set(original)
    for lo, hi in zip(original, original[1:]):
        intervals = max(1, math.ceil((hi - lo) / refinement_step - 1e-10))
        planned.update(round(float(v), 12) for v in np.linspace(lo, hi, intervals + 1))
    censored = any(report["grid_ceiling_witnessed"] for report in reports)
    if censored and exploration_ceiling > original[-1]:
        intervals = math.ceil((exploration_ceiling - original[-1]) / extension_step - 1e-10)
        planned.update(round(float(v), 12) for v in
                       np.linspace(original[-1], exploration_ceiling, intervals + 1))
    if len(planned) > maximum_points:
        raise ValueError("Requested refinement exceeds the explicit work-grid point budget")
    return {"work_scales": sorted(planned), "previous_work_scales": list(original),
            "extension_trigger": "grid_ceiling_witnessed" if censored else None,
            "exploration_ceiling": exploration_ceiling, "global_upper_bound": None,
            "failed_scales_remain_undetermined": True, "evaluate_every_factor": True,
            "grid_spacing_is_not_a_boundary_error_bound": True}


def reachable_branch_design(muscle_names: Sequence[str], *, log_amplitude: float = .25) -> dict:
    """Short nominal-task policies that can excite reachable fatigue directions.

    Paired zero-sum log-weight directions redistribute cost while the arithmetic
    mean of weights remains one. Their endpoints are proposals, not synthetic
    fatigue states: the full RHO must generate and certify each endpoint.
    """
    names = tuple(muscle_names)
    if len(names) < 2 or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("At least two distinct explicit muscles are required")
    amplitude = _positive(log_amplitude, "log_amplitude")
    if amplitude > 1.:
        raise ValueError("Local pilot log-amplitude must not exceed 1")
    dimension = len(names)
    directions = []
    # Orthonormal Helmert contrasts: common scaling is intentionally excluded.
    for j in range(1, dimension):
        direction = np.zeros(dimension)
        direction[:j], direction[j] = 1., -j
        directions.append(direction / np.linalg.norm(direction))
    branches = []

    def branch(name, direction, amplitude, partition, endpoints):
        weights = np.exp(amplitude * direction)
        weights /= np.mean(weights)
        branches.append({"id": name, "partition": partition, "endpoint_cycles": endpoints,
                         "relative_log_direction": direction.tolist(),
                         "absolute_fatigue_weights_from_unit": dict(zip(names, weights.tolist())),
                         "starting_checkpoint": "exact_anchor_receipt", "nominal_work_scale": 1.})

    branch("unit", np.zeros(dimension), 0., "train", [1, 3])
    for index, direction in enumerate(directions):
        for sign, label in ((1, "plus"), (-1, "minus")):
            branch(f"contrast-{index + 1}-{label}", sign * direction, amplitude, "train", [1, 3])
    # Mixed directions and unseen durations are reserved before any new probes.
    mixed = np.sum(directions, axis=0)
    mixed /= np.linalg.norm(mixed)
    for sign, label in ((1, "plus"), (-1, "minus")):
        branch(f"mixed-{label}-holdout", sign * mixed, .6 * amplitude, "holdout", [2, 4])
    return {"muscle_order": list(names), "branches": branches,
            "estimated_nominal_rho_cycles_per_anchor": sum(max(b["endpoint_cycles"]) for b in branches),
            "endpoint_count_per_anchor": sum(len(b["endpoint_cycles"]) for b in branches),
            "shared_prefix_endpoints_stay_in_same_partition": True,
            "state_excitation_guaranteed": False,
            "reject_on": ["failed_nominal_cycle", "inexact_checkpoint_restoration",
                          "outside_declared_local_box", "rank_deficient_reachable_state_design",
                          "ill_conditioned_reachable_state_design", "unpredictable_heldout_reserve",
                          "no_reproducible_observed_reserve_contrast"],
            "prohibited": ["edit_A_without_reachable_RHO_trajectory",
                           "count_multiple_load_probes_as_independent_state_samples",
                           "count_successive_endpoints_as_independent_trajectories",
                           "treat_NLP_failure_as_upper_bound"]}
