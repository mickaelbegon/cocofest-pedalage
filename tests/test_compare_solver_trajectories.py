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
    / "compare_solver_trajectories.py"
)
SPEC = importlib.util.spec_from_file_location("compare_solver_trajectories", SCRIPT_PATH)
comparison = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = comparison
SPEC.loader.exec_module(comparison)


def _write_trajectory(
    path: Path,
    *,
    cycles: int,
    formulation: str,
    state_intervals: int,
    offset: float,
) -> None:
    state_samples = cycles * state_intervals + 1
    controls_per_cycle = 3
    control_samples = cycles * controls_per_cycle
    metadata = {
        "cycles_per_window": cycles,
        "mechanical_formulation": formulation,
        "stimulations_per_cycle": controls_per_cycle,
        "fatigue_capacity_scales": {
            "A_Biceps": 100.0,
            "A_Triceps": 200.0,
            "A_Delt_ant": 300.0,
            "A_Delt_post": 400.0,
        },
    }
    payload = {}
    if formulation == "reduced":
        payload["states__theta"] = np.linspace(
            0.0, -2.0 * np.pi * cycles, state_samples
        )[None, :]
        payload["states__omega"] = np.full((1, state_samples), -6.0 + offset)
    else:
        payload["states__q"] = np.vstack(
            (
                np.zeros(state_samples),
                np.ones(state_samples),
                np.linspace(1.0, 1.0 - 2.0 * np.pi * cycles, state_samples),
            )
        )
        payload["states__qdot"] = np.vstack(
            (
                np.zeros(state_samples),
                np.ones(state_samples),
                np.full(state_samples, -6.0 + offset),
            )
        )
    for index, muscle in enumerate(comparison.MUSCLES, start=1):
        scale = metadata["fatigue_capacity_scales"][f"A_{muscle}"]
        for state, base in (("Cn", 0.1), ("F", 10.0 * index), ("Tau1", 0.2), ("Km", 0.3)):
            payload[f"states__{state}_{muscle}"] = np.full(
                (1, state_samples), base + offset
            )
        payload[f"states__A_{muscle}"] = np.linspace(
            scale, 0.9 * scale + offset, state_samples
        )[None, :]
        payload[f"controls__last_pulse_width_{muscle}"] = np.full(
            (1, control_samples), 200e-6 + offset * 1e-6
        )
    payload["metadata__json"] = np.asarray(json.dumps(metadata))
    np.savez(path, **payload)


def test_compute_comparison_maps_full_mechanics_and_resamples(tmp_path):
    paths = {
        "IPOPT R5": tmp_path / "ipopt.npz",
        "MadNLP R5": tmp_path / "madnlp.npz",
        "ACADOS IRK": tmp_path / "acados.npz",
    }
    _write_trajectory(
        paths["IPOPT R5"], cycles=3, formulation="reduced", state_intervals=4, offset=0.0
    )
    _write_trajectory(
        paths["MadNLP R5"], cycles=3, formulation="reduced", state_intervals=4, offset=1.0
    )
    _write_trajectory(
        paths["ACADOS IRK"], cycles=3, formulation="full", state_intervals=2, offset=2.0
    )
    trajectories = {}
    metadata = {}
    for name, path in paths.items():
        trajectories[name], metadata[name] = comparison.load_trajectory(path)

    summary, plot_data = comparison.compute_comparison(trajectories, metadata, 3)

    assert summary["common_variable_count"] == 26
    assert summary["headline"]["MadNLP R5"]["wheel_speed_rmse_rad_s"] == pytest.approx(1.0)
    assert summary["headline"]["ACADOS IRK"]["wheel_speed_rmse_rad_s"] == pytest.approx(2.0)
    assert summary["headline"]["ACADOS IRK"]["pulse_width_rmse_us"] == pytest.approx(2.0)
    assert plot_data["aligned"]["states__wheel_speed"]["ACADOS IRK"].shape == (3, 1, 5)


def test_main_writes_figures_and_summary_without_solver_dependencies(tmp_path):
    paths = {name: tmp_path / f"{name.split()[0].lower()}.npz" for name in comparison.COLORS}
    _write_trajectory(
        paths["IPOPT R5"], cycles=3, formulation="reduced", state_intervals=4, offset=0.0
    )
    _write_trajectory(
        paths["MadNLP R5"], cycles=3, formulation="reduced", state_intervals=4, offset=0.2
    )
    _write_trajectory(
        paths["ACADOS IRK"], cycles=3, formulation="full", state_intervals=2, offset=0.4
    )
    output = tmp_path / "comparison"

    return_code = comparison.main(
        [
            "--cycles",
            "3",
            "--ipopt-trajectory",
            str(paths["IPOPT R5"]),
            "--madnlp-trajectory",
            str(paths["MadNLP R5"]),
            "--acados-trajectory",
            str(paths["ACADOS IRK"]),
            "--output-dir",
            str(output),
        ]
    )

    assert return_code == 0
    assert len(list(output.glob("*.png"))) == 8
    assert (output / "comparison.md").is_file()
    persisted = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert persisted["cycles_compared"] == 3
    assert persisted["metadata"]["ACADOS IRK"]["mechanical_formulation"] == "full"
