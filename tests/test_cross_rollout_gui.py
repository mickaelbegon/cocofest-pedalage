"""Headless GUI contracts for the independent cross-rollout engine."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from cocofest.simulation.cross_rollout_model import CrossRolloutConfig, build_cross_rollout_plan, cross_rollout_summary, cross_rollout_report
from cocofest.simulation.gui import SimulationApp


def test_request_roundtrip_and_argument_transport(tmp_path):
    config = CrossRolloutConfig(sources=("solution a.npz", "solution b.npz"), output_root=str(tmp_path / "out"),
                              model_config="models/exact.json", cost_config="cost.json", samples_per_interval=65)
    assert CrossRolloutConfig.from_json(config.to_json()) == config
    plan = build_cross_rollout_plan(config, tmp_path / "environment", tmp_path)
    assert str(tmp_path / "solution a.npz") in plan.argv
    assert "solution a.npz'" in plan.command
    assert plan.argv.count("--source") == 2
    assert plan.result_json == tmp_path / "out/matrix.json"
    assert plan.argv[plan.argv.index("--cost-config")+1] == str(tmp_path / "cost.json")
    assert plan.argv[plan.argv.index("--samples-per-interval")+1] == "65"
    assert "--solvers" not in plan.argv
    assert "evaluate_solver_cross_rollout.py" in plan.command


@pytest.mark.parametrize("changes", [dict(evaluators=("radau5",)), dict(cycles=0), dict(start_cycle=-1),
                                    dict(dop853_rtol=float("nan")), dict(samples_per_interval=1),
                                    dict(evaluators=("dop853", "unknown")), dict(sources=("a", "a")),
                                    dict(model_config=True), dict(sources="source.npz"), dict(evaluators="dop853")])
def test_invalid_request_is_refused(changes):
    with pytest.raises(ValueError):
        replace(CrossRolloutConfig(sources=("source.npz",)), **changes).validate()


def test_unknown_fields_and_nonlists_are_refused():
    with pytest.raises(ValueError, match="inconnus"):
        CrossRolloutConfig.from_json('{"unexpected": true}')
    with pytest.raises(ValueError, match="liste"):
        CrossRolloutConfig.from_json('{"sources": "source.npz"}')


def test_gui_import_remains_standard_library_only():
    subprocess.run([sys.executable, "-c", "import sys; import cocofest.simulation.gui; "
                    "assert not any(name in sys.modules for name in ('numpy', 'casadi', 'bioptim', 'scipy', 'tkinter'))"], check=True)


def test_summary_does_not_infer_feasibility_from_numerical_success():
    payload = {"schema": "cocofest-solver-cross-rollout-v1", "sources": [
        {"evaluations": {"dop853": {"status": "success"}, "radau5": {"status": "failed"}}}],
        "rankings": {"dop853": [0]}}
    summary = cross_rollout_summary(payload)
    assert "1/2" in summary and "ne certifie pas la faisabilité" in summary
    assert "[0]" in summary
    payload["rankings"] = {}
    assert "indisponible" in cross_rollout_summary(payload)


def test_report_displays_phase_violation_independently_of_lower_cost():
    payload = {"schema": "cocofest-solver-cross-rollout-v1", "sources": [
        {"provenance": {"source": {"path": "solution.npz"}}, "evaluations": {
            "dop853": {"status": "success", "common_score": .25, "score_difference_from_dop853": 0.,
                       "physical_metrics": {"sampled_phase_bound_violation_rad": .01}},
            "radau5": {"status": "failed", "error": "implicit residual"}}}],
        "comparison": {"comparable": False, "reasons": ["different incoming state"]}, "rankings": {}}
    report = cross_rollout_report(payload)
    assert "solution.npz" in report and "0.01" in report and "0.25" in report
    assert "INCOMPATIBLES" in report and "different incoming state" in report
    assert "ÉCHEC : implicit residual" in report and "ne certifie pas" in report


class Value:
    def __init__(self, value):
        self.value = value
    def get(self):
        return self.value
    def set(self, value):
        self.value = value


def test_gui_replay_form_preserves_every_request_field():
    config = CrossRolloutConfig(sources=("source.npz",), cost_config="cost.json", samples_per_interval=33,
                              dop853_rtol=1e-12, dop853_atol=1e-14, cycles=5)
    data = json.loads(config.to_json())
    variables = {name: Value(json.dumps(value) if name == "sources" else ", ".join(value) if name == "evaluators"
                            else "" if value is None else str(value)) for name, value in data.items() if name != "schema_version"}
    assert SimulationApp.current_rollout(SimpleNamespace(rollout_variables=variables)) == config


def test_reoptimized_rho_selects_existing_controller_form_without_loading_replay_sources():
    selections = []
    app = SimpleNamespace(variables={"mode": Value("fho")}, form_notebook=SimpleNamespace(select=selections.append),
                          scientific_status=Value(""))
    SimulationApp.select_reoptimized_rho(app)
    assert app.variables["mode"].get() == "rho" and selections == [0]
    assert "recalculées" in app.scientific_status.get()
