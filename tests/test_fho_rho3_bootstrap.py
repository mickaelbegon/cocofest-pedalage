from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/run_full_horizon_benchmark.py"
SPEC = importlib.util.spec_from_file_location("fho_rho3_bootstrap_driver", SCRIPT)
driver = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = driver
SPEC.loader.exec_module(driver)


def test_direct_rho3_bootstrap_records_a_certified_start(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(driver, "_seed_cycle_count", lambda path: 5)
    monkeypatch.setattr(driver, "_next_horizon_chance", lambda *args: 1)
    monkeypatch.setattr(driver, "_write_report", lambda *args: None)
    monkeypatch.setattr(driver, "_write_markdown", lambda *args: None)

    def run_attempt(args, **kwargs):
        calls.append(kwargs)
        return {
            "cycles": 3,
            "success": True,
            "certificate_valid": True,
            "solution_path": str(tmp_path / "full-solution.npz"),
        }

    monkeypatch.setattr(driver, "_run_horizon_attempt", run_attempt)
    args = SimpleNamespace(output_dir=tmp_path, max_cycles=100)
    report = {"full_horizon_attempts": [], "homotopy_constructed_cycles": 0}
    result = driver._bootstrap_direct_three_cycle_horizon(
        args,
        report=report,
        report_path=tmp_path / "report.json",
        markdown_path=tmp_path / "report.md",
        rho_seed_path=tmp_path / "rho-reduced.npz",
        rss_limit_bytes=123,
    )

    assert calls[0]["cycles"] == 3
    assert calls[0]["rho_seed"] == tmp_path / "rho-reduced.npz"
    assert "prefix_solution_path" not in calls[0]
    assert calls[0]["heartbeat_seed_label"] == "RHO_1..RHO_3"
    assert result["adaptive_source_cycles"] == 0
    assert result["accepted_for_continuation"] is True
    assert report["largest_successful_cycles"] == 3
    assert report["bootstrap_state"]["status"] == "certified"
