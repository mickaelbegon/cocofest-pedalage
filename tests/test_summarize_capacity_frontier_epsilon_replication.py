import json

from scripts.summarize_capacity_frontier_epsilon_replication import summarize


ORDER = ["Delt_ant", "Delt_post", "Biceps", "Triceps"]


def report(tmp_path, *, cycle, epsilon_upper, capacity):
    receipt = tmp_path / f"cycle-{cycle}" / "receipt.json"
    receipt.parent.mkdir(exist_ok=True); receipt.write_text("{}")
    active = next((i for i, value in enumerate(epsilon_upper) if value is not None), "minimum")
    value = {"status": "fresh_discrete_rebuild_and_DOP853_completed_not_global_front_certified",
             "receipt": str(receipt), "condition_sha256": f"condition-{cycle}", "muscle_order": ORDER,
             "epsilon_upper": epsilon_upper,
             "support_smoke": {"initial_capacity_ratio": capacity, "epsilon_capacity_max_violation": 0.},
             "independent_rebuild": {"discrete_audit_passed": True, "continuous_DOP853_replay": "complete",
                                     "DOP853_relative_work_error": 1e-4,
                                     "DOP853": {"maximum_scaled_endpoint_error": 1e-4}}}
    suffix = "minimum" if active == "minimum" else epsilon_upper[active]
    path = tmp_path / f"report-{cycle}-{active}-{suffix}.json"; path.write_text(json.dumps(value))
    return path


def test_summary_requires_three_ordered_cuts_and_checks_triceps_monotonicity(tmp_path):
    minima = {}
    reports = []
    for cycle in (120, 160):
        minima[cycle] = report(tmp_path, cycle=cycle, epsilon_upper=[None] * 4, capacity=[.99, .99, .96, .32])
        for epsilon, tri in ((.4, .37), (.6, .35), (.8, .33)):
            reports.append(report(tmp_path, cycle=cycle, epsilon_upper=[epsilon, None, None, None],
                                  capacity=[epsilon, .99, .96, tri]))
        for index in (1, 2):
            for epsilon, tri in ((.4, .37), (.6, .35), (.8, .33)):
                upper = [None] * 4; upper[index] = epsilon
                capacity = [.99, .99, .96, tri]; capacity[index] = epsilon
                reports.append(report(tmp_path, cycle=cycle, epsilon_upper=upper, capacity=capacity))
    summary = summarize(minimum_reports=minima, epsilon_reports=reports)
    assert len(summary["rows"]) == 6
    assert summary["all_monotonicity_checks_passed"] is True


def test_summary_accepts_one_complete_holdout_checkpoint(tmp_path):
    minimum = report(tmp_path, cycle=140, epsilon_upper=[None] * 4, capacity=[.99, .99, .96, .32])
    reports = []
    for index in range(3):
        for epsilon, tri in ((.4, .37), (.6, .35), (.8, .33)):
            upper = [None] * 4; upper[index] = epsilon
            capacity = [.99, .99, .96, tri]; capacity[index] = epsilon
            reports.append(report(tmp_path, cycle=140, epsilon_upper=upper, capacity=capacity))
    value = summarize(minimum_reports={140: minimum}, epsilon_reports=reports)
    assert {row["checkpoint_cycle"] for row in value["rows"]} == {140}
