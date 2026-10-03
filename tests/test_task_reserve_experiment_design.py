"""Scientific guards for a local experiment; no OCP solves in unit tests."""

from dataclasses import replace
import math

import numpy as np
import pytest

from cocofest.optimization.task_reserve import (
    DEFAULT_CONSTRAINT_GROUPS, LocalReserveSample, ProbeEvidence, ReserveEstimate, TaskReserveCheckpoint,
)
from cocofest.optimization.task_reserve_experiment_design import (
    ReserveDesignSample, common_followup_grid, local_neighborhood, observed_grid_report,
    reachable_branch_design,
)


def sample(name, coordinates, *, feasible=(1., 1.1, 1.2), grid=(1., 1.1, 1.2), partition="train"):
    checkpoint = TaskReserveCheckpoint("unused.npz", "archive-" + name, "prepared-" + name,
        20, "unused.json", "model", '{"task":"test"}', "context")
    evidence = tuple(ProbeEvidence(factor, checkpoint.prepared_problem_sha256,
        checkpoint.task_context_sha256, checkpoint.model_sha256,
        "Infeasible_Problem_Detected" if factor not in feasible else "Solve_Succeeded",
        0. if factor in feasible else 1., 1e-5, factor in feasible,
        "witness.npz" if factor in feasible else None, .1, DEFAULT_CONSTRAINT_GROUPS, True)
        for factor in grid)
    best = max(feasible) if feasible else None
    estimate = ReserveEstimate(checkpoint, evidence, tuple(feasible), 1. in feasible, best,
        None if best is None else best - 1., "nominal_and_reserve_witnessed")
    return ReserveDesignSample(name, LocalReserveSample(tuple(coordinates), estimate), partition, name + ".json")


def test_ceiling_feasibility_is_censoring_not_a_measured_maximum():
    report = observed_grid_report(sample("a", [.8, .7]).sample)
    assert report["observation_kind"] == "grid_ceiling_censored"
    assert report["observed_reserve_lower_bound"] == pytest.approx(.2)
    assert not report["maximum_measured"]
    assert report["global_upper_bound"] is None


def test_failed_larger_factors_do_not_bracket_feasibility_boundary():
    report = observed_grid_report(sample("a", [.8, .7], feasible=(1., 1.1)).sample)
    assert not report["grid_ceiling_witnessed"]
    assert report["undetermined_scales"] == [1.2]
    assert report["global_upper_bound"] is None
    assert not report["maximum_measured"]
    assert report["observation_kind"] == "witness_lower_bound_with_undetermined_larger_loads"


def test_nonmonotone_witnesses_are_retained_and_solver_status_is_ignored():
    point = sample("a", [.8, .7], feasible=(1., 1.2)).sample
    evidence = tuple(replace(e, solver_status="failed") if e.work_scale == 1.2 else e
                     for e in point.estimate.evidence)
    point = replace(point, estimate=replace(point.estimate, evidence=evidence))
    report = observed_grid_report(point)
    assert report["feasible_scales"] == [1., 1.2]
    assert report["undetermined_scales"] == [1.1]
    assert report["grid_ceiling_witnessed"]


def test_tampered_summary_cannot_override_witnesses():
    point = sample("a", [.8]).sample
    point = replace(point, estimate=replace(point.estimate, work_scale_lower_bound=2.))
    with pytest.raises(ValueError, match="disagrees"):
        observed_grid_report(point)


def test_locality_does_not_expand_to_force_full_rank():
    samples = [sample("a", [.8, .7]), sample("b", [.79, .69]),
               sample("c", [.75, .6], partition="holdout")]
    report = local_neighborhood(samples, anchor_id="a", coordinate_radius=(.03, .03))
    assert [s["id"] for s in report["selected"]] == ["a", "b"]
    assert report["excluded"][0]["id"] == "c"
    assert report["discovery_design"]["train"]["affine_rank"] == 2
    assert report["discovery_design"]["train"]["required_rank"] == 3
    assert report["discovery_design"]["train"]["condition_number"] is None
    assert report["discovery_design"]["holdout"]["count"] == 0
    assert not report["activation_allowed"]


def test_full_rank_curved_trajectory_still_does_not_identify_policy_gradient():
    samples = [sample("a", [.8, .7]), sample("b", [.79, .68]), sample("c", [.78, .695]),
               sample("d", [.795, .696], partition="holdout")]
    report = local_neighborhood(samples, anchor_id="a", coordinate_radius=(.03, .03))
    assert report["discovery_design"]["train"]["affine_rank"] == 3
    assert report["discovery_design"]["train"]["condition_number"] < 100
    assert report["observed_lower_bound_contrast"] == 0
    assert not report["activation_allowed"]
    assert not report["frontier_contrast_demonstrated"]


def test_observed_lower_bound_contrast_is_not_frontier_contrast():
    samples = [sample("a", [.8, .7]), sample("b", [.79, .69], feasible=(1., 1.1))]
    report = local_neighborhood(samples, anchor_id="a", coordinate_radius=(.03, .03))
    assert report["observed_lower_bound_contrast"] == pytest.approx(.1)
    assert not report["frontier_contrast_demonstrated"]


@pytest.mark.parametrize("field", ["model_sha256", "task_context_sha256"])
def test_context_mismatch_is_rejected(field):
    a, b = sample("a", [.8]), sample("b", [.81])
    source = replace(b.sample.estimate.checkpoint, **{field: "other"})
    b = replace(b, sample=replace(b.sample, estimate=replace(b.sample.estimate, checkpoint=source)))
    with pytest.raises(ValueError, match="contexts or models"):
        local_neighborhood([a, b], anchor_id="a", coordinate_radius=[.05])


def test_duplicate_prepared_checkpoint_cannot_be_another_observation():
    a = sample("a", [.8])
    b = replace(a, sample_id="b", partition="holdout")
    with pytest.raises(ValueError, match="duplicated"):
        local_neighborhood([a, b], anchor_id="a", coordinate_radius=[.05])


def test_common_grid_extends_censored_data_and_retains_all_failed_factors():
    reports = [observed_grid_report(sample("a", [.8]).sample),
               observed_grid_report(sample("b", [.81], feasible=(1.,)).sample)]
    grid = common_followup_grid(reports, refinement_step=.025, exploration_ceiling=1.5)
    assert grid["work_scales"] == pytest.approx([1., 1.025, 1.05, 1.075, 1.1, 1.125,
                                                1.15, 1.175, 1.2, 1.3, 1.4, 1.5])
    assert grid["global_upper_bound"] is None
    assert grid["evaluate_every_factor"]
    assert grid["grid_spacing_is_not_a_boundary_error_bound"]


def test_no_ceiling_witness_refines_full_original_grid_without_claiming_boundary():
    report = observed_grid_report(sample("a", [.8], feasible=(1., 1.1)).sample)
    grid = common_followup_grid([report], refinement_step=.025)
    assert grid["work_scales"][-1] == 1.2
    assert grid["extension_trigger"] is None
    assert grid["global_upper_bound"] is None


def test_grid_budget_and_common_measurement_protocol_enforced():
    report = observed_grid_report(sample("a", [.8]).sample)
    with pytest.raises(ValueError, match="point budget"):
        common_followup_grid([report], refinement_step=.001, maximum_points=20)
    with pytest.raises(ValueError, match="same grid"):
        common_followup_grid([report, {**report, "tested_scales": [1., 1.05]}])


def test_branches_keep_objective_scale_and_grouped_holdout_before_measurement():
    design = reachable_branch_design(["Delt_ant", "Delt_post", "Biceps", "Triceps"])
    branches = design["branches"]
    assert len(branches) == 9
    assert design["endpoint_count_per_anchor"] == 18
    assert design["estimated_nominal_rho_cycles_per_anchor"] == 29
    directions = []
    for branch in branches:
        weights = np.array(list(branch["absolute_fatigue_weights_from_unit"].values()))
        assert weights.min() > 0
        assert weights.mean() == pytest.approx(1.)
        assert sum(branch["relative_log_direction"]) == pytest.approx(0.)
        assert branch["nominal_work_scale"] == 1.
        if branch["id"].endswith("plus"):
            directions.append(branch["relative_log_direction"])
        assert branch["endpoint_cycles"] == ([1, 3] if branch["partition"] == "train" else [2, 4])
    assert np.array(directions) @ np.array(directions).T == pytest.approx(np.eye(3))
    assert not design["state_excitation_guaranteed"]
    assert design["shared_prefix_endpoints_stay_in_same_partition"]


@pytest.mark.parametrize("names,amplitude", [(["a"], .25), (["a", "a"], .25),
    (["a", "b"], math.nan), (["a", "b"], 0.), (["a", "b"], 2.)])
def test_invalid_local_branch_design_rejected(names, amplitude):
    with pytest.raises(ValueError):
        reachable_branch_design(names, log_amplitude=amplitude)
