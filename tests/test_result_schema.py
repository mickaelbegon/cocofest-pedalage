"""Regression coverage for result units and optional FHO diagnostic evidence."""

from types import SimpleNamespace
import json

from cocofest.evaluation.result_schema import audit_registry, timing_populations


def test_hot_populations_exclude_first_window_failure_and_hybrid_from_target():
    rows = [
        {"window": 0, "solver_time_s": 5., "wall_time_s": 6., "iterations": 10, "certifier": "target_solver"},
        {"window": 1, "solver_time_s": .2, "wall_time_s": .3, "iterations": 2, "certifier": "target_solver"},
        {"window": 2, "solver_time_s": .4, "wall_time_s": None, "iterations": 3, "certifier": "ipopt_recovery"},
        {"window": 3, "solver_time_s": 7., "wall_time_s": 8., "iterations": 100, "certifier": "target_solver"},
    ]
    metadata = timing_populations(rows, 3)
    assert metadata["hot"]["window_indices"] == [1, 2]
    assert metadata["hot"]["solver_time_s"]["sample_count"] == 2
    assert metadata["hot"]["wall_time_s"]["sample_count"] == 1
    assert metadata["target_solver_only_hot"]["window_indices"] == [1]
    assert metadata["per_cycle_time"]["includes_first_window"] is True


def test_monolithic_fho_has_no_hot_population_even_when_it_covers_200_cycles():
    metadata = timing_populations([
        {"window": 0, "solver_time_s": 20., "wall_time_s": 25., "iterations": 100, "certifier": "target_solver"},
    ], 1)
    assert metadata["hot"]["window_indices"] == []
    assert metadata["hot"]["solver_time_s"]["sample_count"] == 0


def test_audit_registry_distinguishes_missing_partial_failure_and_diagnostic():
    registry = audit_registry({
        "window_feasibility": [{"passes_tolerance": True}, {"passes_tolerance": False}],
        "integrator_map_final_solution": [{"node": 0}, {"available": False, "error": "failed"}],
        "high_accuracy_trace_rollout": {"available": False, "reason": "unsupported"},
    })
    assert registry["window_feasibility"]["passes_tolerance"] is False
    assert registry["integrator_map_final_solution"]["status"] == "partial"
    assert registry["integrator_map_final_solution"]["role"] == "diagnostic"
    assert registry["high_accuracy_trace_rollout"]["status"] == "unavailable"
    assert registry["parametric_kkt_audits"]["status"] == "not_recorded"


def test_single_shot_requested_audits_are_collected_without_console_dependency(monkeypatch):
    from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as example

    calls = []
    monkeypatch.setattr(example, "apply_solution_directly_to_periodic_nmpc_initial_guess", lambda nmpc, sol: calls.append("apply"))
    def maps(nmpc, *, nodes):
        assert nodes == (0, 29, 150, 299)
        return [{"node": 0, "trajectory_vs_reference": 1e-8}]
    monkeypatch.setattr(example, "high_accuracy_integrator_map_diagnostics", maps)
    monkeypatch.setattr(example, "canonical_solution_kkt_audit", lambda sol, **kwargs: {"available": True, "variable_count": 200})
    summary = {"solver_success": True, "covered_cycles": 10}
    nmpc = SimpleNamespace(nlp=[SimpleNamespace(ns=300)], cycle_len=30)
    example.attach_single_shot_diagnostics(summary, nmpc, object(), SimpleNamespace(
        validate_integrator_maps=True, parametric_kkt_audit=True,
    ))
    assert calls == ["apply"]
    assert summary["integrator_map_final_solution"][0]["trajectory_vs_reference"] == 1e-8
    assert summary["parametric_kkt_audits"][0]["window"] == 0
    assert summary["parametric_kkt_audits"][0]["covered_cycles"] == 10
    assert summary["high_accuracy_trace_rollout"]["available"] is False
    assert summary["integrator_map_final_solution_wall_time_s"] >= 0


def test_single_shot_audit_errors_remain_reported_without_changing_certification(monkeypatch):
    from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as example

    def fail(*args, **kwargs):
        raise ValueError("invalid dynamics")

    monkeypatch.setattr(example, "apply_solution_directly_to_periodic_nmpc_initial_guess", fail)
    monkeypatch.setattr(example, "canonical_solution_kkt_audit", fail)
    summary = {"solver_success": True, "physical_success": True, "success": True}
    example.attach_single_shot_diagnostics(summary, object(), object(), SimpleNamespace(
        validate_integrator_maps=True, parametric_kkt_audit=True,
    ))
    assert summary["integrator_map_final_solution"][0]["error"] == "ValueError: invalid dynamics"
    assert summary["parametric_kkt_audits"][0]["available"] is False
    assert summary["success"] is True


def test_failed_fho_setup_retains_one_window_and_requested_cycles():
    from examples.fes_multibody.cycling import cycling_fes_solver_comparison as example

    result = example._failed_solver_result(SimpleNamespace(single_shot=True, cycles_per_window=20, n_windows=1), RuntimeError("missing plugin"), 2.)
    assert result["requested_windows"] == 1
    assert result["requested_cycles"] == 20
    assert result["attempted_windows"] == 0
    assert result["covered_cycles"] == 0


def test_serialized_fho_retains_diagnostic_failure_and_empty_hot_population(tmp_path):
    from examples.fes_multibody.cycling import cycling_fes_solver_comparison as example

    args = SimpleNamespace(single_shot=True, cycles_per_window=20, n_windows=1,
                           stimulations_per_cycle=30, solver="ipopt")
    result = example._failed_solver_result(args, RuntimeError("missing plugin"), 2.)
    result["integrator_map_final_solution"] = [{"available": False, "error": "diagnostic failed"}]
    result["parametric_kkt_audits"] = [{"available": False, "reason": "solution_not_certified"}]
    output = example.write_benchmark_summary(tmp_path / "result.json", {"ipopt": result})
    payload = json.loads(output.read_text())
    assert payload["schema_version"] == 4
    row = payload["results"][0]
    assert row["requested_windows"] == 1
    assert row["requested_cycles"] == 20
    assert row["hot_solver_time_median_s"] is None
    assert row["timing_populations"]["hot"]["window_indices"] == []
    assert row["integrator_map_final_solution"][0]["error"] == "diagnostic failed"
    assert row["audit_registry"]["parametric_kkt_audits"]["status"] == "unavailable"
