import json

from scripts.summarize_capacity_frontier_pilot import summarize


def write_report(tmp_path, name, capacity, *, context="a" * 64, work=1e-4, endpoint=2e-4,
                 epsilon_violation=0.):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({
        "status": "fresh_discrete_rebuild_and_DOP853_completed_not_global_front_certified",
        "condition_sha256": context, "muscle_order": ["A", "B"], "support": [.5, .5],
        "support_smoke": {"initial_capacity_ratio": capacity, "support_value": sum(capacity) / 2,
                          "epsilon_capacity_max_violation": epsilon_violation},
        "independent_rebuild": {"discrete_audit_passed": True, "continuous_DOP853_replay": "complete",
                                "DOP853_relative_work_error": work,
                                "DOP853": {"maximum_scaled_endpoint_error": endpoint},
                                "discrete_audit": {"constraint_normalized_violation": 0.}},
    }))
    return path


def test_summary_filters_numeric_gate_and_computes_minimum_capacity_dominance(tmp_path):
    best = write_report(tmp_path, "best", [.2, .2])
    dominated = write_report(tmp_path, "dominated", [.3, .4])
    rejected = write_report(tmp_path, "rejected", [.1, .1], work=.1)
    result = summarize([best, dominated, rejected], maximum_relative_work_error=.01,
                       maximum_scaled_endpoint_error=.01)
    rows = {row["id"]: row for row in result["reports"]}
    assert rows["best"]["provisional_nondominated"]
    assert rows["dominated"]["provisionally_dominated_by"] == ["best"]
    assert not rows["rejected"]["numeric_gate_passed"]
    assert result["provisional_nondominated_count"] == 1


def test_summary_rejects_a_witness_that_violates_its_epsilon_constraint(tmp_path):
    valid = write_report(tmp_path, "valid", [.3, .3])
    malformed = write_report(tmp_path, "malformed", [.2, .2], epsilon_violation=.2)
    result = summarize([valid, malformed], maximum_relative_work_error=.01,
                       maximum_scaled_endpoint_error=.01)
    rows = {row["id"]: row for row in result["reports"]}
    assert rows["valid"]["numeric_gate_passed"]
    assert not rows["malformed"]["numeric_gate_passed"]
