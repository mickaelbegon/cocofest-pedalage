import json

import numpy as np
import pytest

from scripts.benchmark_numeric_vector_dispatch_ab import summarize


def _write_run(root, *, complete, ipopt, perturbation=0.0):
    audit = root / "symbolic-audit"
    audit.mkdir(parents=True)
    samples = [1.0, complete]
    result = {
        "results": [{
            "mode": "rho_deferred_summary",
            "execution_timing": {
                "offline_finalizer_required": True,
                "rho_complete_iteration_wall_time_samples_s": samples,
            },
        }],
    }
    (root / "result.json").write_text(json.dumps(result), encoding="utf-8")
    for call, wall in enumerate((1.0, ipopt)):
        row = {"call": call, "timings": {"t_wall_total": wall}}
        with (audit / "nlp0_metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        np.savez(audit / f"nlp0_call{call}.npz", x=[call + perturbation], g=[call],
                 lam_x=[0.], lam_g=[0.])


def test_summary_requires_exact_archives_and_reports_hot_timing(tmp_path):
    for mode, complete, ipopt in (("scalar", .2, .1), ("cached", .1, .11)):
        _write_run(tmp_path / "replicate-01" / mode, complete=complete, ipopt=ipopt)
    report = summarize(tmp_path, 1)
    assert report["pooled"]["hot_complete_relative_change"] == pytest.approx(-.5)
    assert report["replicates"][0]["exact_archive_max_abs_error"] == {
        "x": 0.0, "g": 0.0, "lam_x": 0.0, "lam_g": 0.0,
    }


def test_summary_refuses_a_numerically_different_cached_archive(tmp_path):
    _write_run(tmp_path / "replicate-01" / "scalar", complete=.2, ipopt=.1)
    _write_run(tmp_path / "replicate-01" / "cached", complete=.1, ipopt=.1, perturbation=.01)
    with pytest.raises(ValueError, match="changed an archived solver vector"):
        summarize(tmp_path, 1)
