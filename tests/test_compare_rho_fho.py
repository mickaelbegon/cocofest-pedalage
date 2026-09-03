from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest


os.environ.setdefault("MPLBACKEND", "Agg")
SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "scripts"
    / "compare_rho_fho.py"
)
SPEC = importlib.util.spec_from_file_location("compare_rho_fho", SCRIPT_PATH)
comparison = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = comparison
SPEC.loader.exec_module(comparison)


def _write_solution(path: Path, cycles: int, offset: float = 0.0) -> None:
    intervals = 4
    controls = 3
    state_samples = cycles * intervals + 1
    control_samples = cycles * controls
    metadata = {
        "cycles_per_window": cycles,
        "mechanical_formulation": "reduced",
        "stimulations_per_cycle": controls,
        "fatigue_capacity_scales": {
            "A_Biceps": 100.0,
            "A_Triceps": 200.0,
            "A_Delt_ant": 300.0,
            "A_Delt_post": 400.0,
        },
    }
    payload = {
        "states__theta": np.linspace(0.0, -2.0 * np.pi * cycles, state_samples)[None, :],
        "states__omega": np.full((1, state_samples), -6.0 + offset),
    }
    for index, muscle in enumerate(comparison.MUSCLES, start=1):
        scale = metadata["fatigue_capacity_scales"][f"A_{muscle}"]
        payload[f"states__A_{muscle}"] = np.linspace(
            scale, 0.9 * scale + offset, state_samples
        )[None, :]
        payload[f"states__F_{muscle}"] = np.linspace(
            10.0 * index, 20.0 * index + offset, state_samples
        )[None, :]
        payload[f"controls__last_pulse_width_{muscle}"] = np.full(
            (1, control_samples), 200e-6 + offset * 1e-6
        )
    payload["metadata__json"] = np.asarray(json.dumps(metadata))
    np.savez(path, **payload)


def test_compute_comparison_reports_cyclewise_physical_differences(tmp_path):
    rho_path = tmp_path / "rho.npz"
    fho_path = tmp_path / "fho.npz"
    _write_solution(rho_path, 3)
    _write_solution(fho_path, 3, offset=2.0)
    rho, rho_metadata = comparison.load_trajectory(rho_path)
    fho, fho_metadata = comparison.load_trajectory(fho_path)

    summary, plot_data = comparison.compute_comparison(
        rho, fho, rho_metadata, fho_metadata, 3
    )

    assert summary["cycles_compared"] == 3
    assert summary["headline"]["omega_rmse_rad_s"] == pytest.approx(2.0)
    assert summary["headline"]["pulse_width_rmse_us"] == pytest.approx(2.0)
    assert plot_data["per_cycle"]["states__omega"].shape == (3,)
    assert plot_data["pulse_width"]["Biceps"]["rho"].shape == (3, 3)


def test_main_discovers_artifacts_and_writes_all_figures(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    rho_path = results / "rho.npz"
    fho_path = results / "fho.npz"
    _write_solution(rho_path, 3)
    _write_solution(fho_path, 3, offset=1.0)
    report = {
        "rho": {"seed_path": str(rho_path), "elapsed_s": 20.0},
        "full_horizon_attempts": [
            {
                "cycles": 3,
                "success": True,
                "accepted_for_continuation": True,
                "solution_path": str(fho_path),
                "elapsed_s": 30.0,
            }
        ],
        "extension_rho_attempts": [
            {"target_cycle": 3, "elapsed_s": 2.0, "success": True}
        ],
    }
    (results / "full-horizon-report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )

    return_code = comparison.main(
        ["--results-dir", str(results), "--cycles", "3"]
    )

    output = results / "rho-vs-fho-0003"
    assert return_code == 0
    assert (output / "comparison.json").is_file()
    assert (output / "comparison.md").is_file()
    assert len(list(output.glob("*.png"))) == 7
    persisted = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert persisted["timing"]["iterative_fho_construction_elapsed_s"] == 32.0


def test_discovery_rejects_an_uncertified_target(tmp_path):
    report = tmp_path / "full-horizon-report.json"
    report.write_text(
        json.dumps(
            {
                "rho": {"seed_path": "rho.npz"},
                "full_horizon_attempts": [
                    {"cycles": 100, "success": False, "solution_path": "fho.npz"}
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="No accepted FHO_100"):
        comparison.discover_artifacts(report, 100)
