import copy
import math

from scripts.validate_radau_frequency import build_cases, summarize_result


def document(theta_error=0.02, closure_error=0.02):
    return {"configurations": {"ipopt": {
        "collocation_degree": 3, "calcium_stimulation_interval_s": 0.02,
        "control_decisions_per_cycle": 50}}, "results": [{
            "solver": "ipopt", "solver_success": True, "nlp_validated_cycles": 1,
            "physical_success": True,
            "state_boundary_snapshots": {"cycle_1": {"states": {
                "theta": {"start": [0.0], "end": [-2 * math.pi]}}}},
            "high_accuracy_trace_rollout": {
                "available": True, "maximum_absolute_endpoint_error_by_state": {
                    "theta": theta_error, "omega": 0.1},
                "final_reference_state": {"theta": [-2 * math.pi + closure_error]},
                "relative_tolerance": 1e-11, "absolute_tolerance": 1e-13}}]}


def test_discrete_success_cannot_hide_replay_closure_or_accuracy_failure():
    row = summarize_result(document())
    assert row["solver_success"]
    assert not row["transcription_accuracy_pass"]
    assert not row["task_closure_pass"]
    assert math.isclose(row["dop853_cycle_closure_error_rad"], 0.02)


def test_task_tolerance_and_numerical_accuracy_budget_are_independent():
    row = summarize_result(document(theta_error=0.001, closure_error=0.001))
    assert row["task_closure_pass"]
    assert not row["transcription_accuracy_pass"]
    good = summarize_result(document(theta_error=0.0001, closure_error=0.0001))
    assert good["transcription_accuracy_pass"]
    assert good["task_closure_pass"]
    missing = copy.deepcopy(document())
    del missing["results"][0]["high_accuracy_trace_rollout"]
    assert not summarize_result(missing)["transcription_accuracy_pass"]


def test_pair_changes_degree_without_changing_stimulation_or_control_grid(tmp_path):
    source = tmp_path / "source"
    source.touch()
    cases = build_cases(python=source, model_config=source, reduced_profile=source,
                        hsl_library=source, output_directory=tmp_path / "runs")
    paired = cases[1:]
    for case in paired:
        argv = case["argv"]
        assert argv[argv.index("--stimulations-per-cycle") + 1] == "50"
        assert argv[argv.index("--cycles-per-window") + 1] == "1"
        assert "--validate-integrator-maps" in argv
        assert "--common-initial-solution-recenter-first-node-bounds" in argv
        assert argv[argv.index("--ipopt-linear-solver") + 1] == "ma57"
    seeds = [case["argv"][case["argv"].index("--common-initial-solution") + 1] for case in paired]
    assert seeds[0] == seeds[1]
    assert [case["degree"] for case in paired] == [3, 5]
