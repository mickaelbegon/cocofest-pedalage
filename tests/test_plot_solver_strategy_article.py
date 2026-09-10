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
    / "plot_solver_strategy_article.py"
)
SPEC = importlib.util.spec_from_file_location("plot_solver_strategy_article", SCRIPT_PATH)
article = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = article
SPEC.loader.exec_module(article)


def _write_trajectory(
    path: Path,
    *,
    cycles: int = 2,
    offset_us: float = 0.0,
    mechanical_formulation: str = "reduced",
) -> None:
    controls_per_cycle = 30
    state_intervals_per_cycle = 4
    state_samples = cycles * state_intervals_per_cycle + 1
    metadata = {
        "cycles_per_window": cycles,
        "stimulations_per_cycle": controls_per_cycle,
        "formulation": "isokinetic",
        "mechanical_formulation": mechanical_formulation,
        "isokinetic_omega": -2.0 * np.pi,
        "energy_equivalent_torque": 0.2,
        "load_torque_min": -3.0,
        "load_torque_max": 3.0,
        "absolute_wheel_q_origin_reference": None,
        "absolute_wheel_q_start_cycle_index": None,
        "fatigue_capacity_scales": {
            "A_Delt_ant": 100.0,
            "A_Delt_post": 200.0,
            "A_Biceps": 300.0,
            "A_Triceps": 400.0,
        },
    }
    payload: dict[str, np.ndarray] = {}
    for muscle_index, muscle in enumerate(article.MUSCLES):
        scale = metadata["fatigue_capacity_scales"][f"A_{muscle}"]
        payload[f"states__A_{muscle}"] = np.linspace(
            scale,
            scale * (0.99 - 0.001 * muscle_index),
            state_samples,
        )[None, :]
        pulse_width = np.full(
            cycles * controls_per_cycle,
            (130.0 + offset_us + muscle_index) * 1e-6,
        )
        pulse_width[::controls_per_cycle] = (250.0 + offset_us) * 1e-6
        payload[f"controls__last_pulse_width_{muscle}"] = pulse_width[None, :]
    payload["metadata__json"] = np.asarray(json.dumps(metadata))
    np.savez(path, **payload)


def test_activation_metrics_preserves_circular_and_disjoint_windows():
    pulse_width = np.zeros(30)
    pulse_width[[28, 29, 0, 1, 10]] = 200.0

    metrics = article.activation_metrics(pulse_width, threshold=150.0)

    assert metrics["onset_deg"] == pytest.approx(336.0)
    assert metrics["offset_deg"] == pytest.approx(24.0)
    assert metrics["longest_arc_deg"] == pytest.approx(48.0)
    assert metrics["total_arc_deg"] == pytest.approx(60.0)
    assert metrics["arc_count"] == 2


@pytest.mark.parametrize(
    ("value", "arc", "count"),
    [(0.0, 0.0, 0), (200.0, 360.0, 1)],
)
def test_activation_metrics_handles_empty_and_full_cycles(value, arc, count):
    metrics = article.activation_metrics(np.full(30, value), threshold=150.0)

    assert np.isnan(metrics["onset_deg"])
    assert np.isnan(metrics["offset_deg"])
    assert metrics["total_arc_deg"] == pytest.approx(arc)
    assert metrics["longest_arc_deg"] == pytest.approx(arc)
    assert metrics["arc_count"] == count


def test_main_generates_article_figures_and_provenance(tmp_path):
    ipopt = tmp_path / "ipopt.npz"
    acados = tmp_path / "acados.npz"
    output = tmp_path / "figures"
    _write_trajectory(ipopt)
    _write_trajectory(acados, offset_us=10.0)

    return_code = article.main(
        [
            "--ipopt",
            str(ipopt),
            "--acados",
            str(acados),
            "--output-dir",
            str(output),
            "--selected-cycles",
            "1",
            "2",
            "--dpi",
            "30",
        ]
    )

    assert return_code == 0
    assert len(list(output.glob("*.png"))) == 4
    assert len(list(output.glob("*.svg"))) == 4
    assert (output / "pw_window_metrics.csv").is_file()
    provenance = json.loads((output / "figure_provenance.json").read_text(encoding="utf-8"))
    assert provenance["cycles"] == 2
    assert provenance["control_representation"].startswith("ZOH")
    assert provenance["pw_delta_acados_minus_ipopt"]["Biceps"]["rmse_us"] == pytest.approx(10.0)


def test_load_comparison_rejects_different_mechanical_formulations(tmp_path):
    ipopt = tmp_path / "ipopt.npz"
    acados = tmp_path / "acados.npz"
    _write_trajectory(ipopt, mechanical_formulation="reduced")
    _write_trajectory(acados, mechanical_formulation="full")

    with pytest.raises(ValueError, match="mechanical_formulation"):
        article.load_comparison(ipopt, acados, cycles=2)
