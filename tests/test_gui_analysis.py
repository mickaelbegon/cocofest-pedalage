"""GUI analysis reads published artifacts without importing or running solvers."""
import json

import pytest

from cocofest.simulation.gui_analysis import LiveArtifacts, cycle_series, generate_analysis_figures


def test_live_reader_keeps_partial_line_and_reports_only_published_certification(tmp_path):
    side = tmp_path / "right"
    side.mkdir()
    journal = side / "weights.jsonl"
    journal.write_bytes(b'{"completed_cycles": 1, "capacity_ratios": [0.95, 0.9]}\n{"completed_cycles": 2')
    live = LiveArtifacts(tmp_path)
    assert "cycle 1" in live.update()
    assert "90.00 %" in live.text()
    assert "certifié" not in live.text()
    first_offset = live.offsets[journal]
    assert "cycle 1" in live.update()
    assert live.offsets[journal] == first_offset
    with journal.open("ab") as stream:
        stream.write(b', "certified": false, "solver_time_s": 0.4}\n')
    text = live.update()
    assert "cycle 2" in text and "non certifié" in text and "solveur 0.4 s" in text
    journal.write_bytes(b'{"completed_cycles": 0}\n')
    assert "cycle 0" in live.update()


def test_figures_use_physical_cycles_and_keep_failed_attempts(tmp_path):
    side = tmp_path / "right"
    side.mkdir()
    rows = [{"physical_cycle": 1, "certified": False, "solver_time_s": 1.2},
            {"physical_cycle": 1, "certified": True, "solver_time_s": .4}]
    (side / "attempts.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n{partial")
    assert cycle_series(tmp_path)["right/result.json"] == rows
    paths = generate_analysis_figures(tmp_path)
    assert paths[0].read_bytes().startswith(b"\x89PNG")
    (side / "result.json").write_text(json.dumps({"cycles": [rows[-1]]}))
    assert cycle_series(tmp_path)["right/result.json"] == [rows[-1]]


def test_trajectory_figure_can_be_generated_without_mechanical_states(tmp_path):
    import numpy as np
    np.savez(tmp_path / "validated-rho-trajectory.npz",
             controls__last_pulse_width_Biceps=np.array([[.0003, .0004]]),
             states__F_Biceps=np.array([[0., 10., 20.]]),
             states__A_Biceps=np.array([[1., .99, .98]]))
    paths = generate_analysis_figures(tmp_path)
    assert len(paths) == 1
    assert paths[0].read_bytes().startswith(b"\x89PNG")
    with pytest.raises(ValueError, match="Aucune métrique"):
        generate_analysis_figures(tmp_path / "absent")
