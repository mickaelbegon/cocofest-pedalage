"""The paired report must reject unequal IPOPT inputs before timing claims."""
import json

import numpy as np
import pytest

from scripts.summarize_prescribed_ding_local_ab import summarize


def test_summary_requires_identical_frozen_solver_inputs(tmp_path):
    metric = {
        "success": True, "original_nx": 2, "original_ng": 1,
        "solver_nx": 2, "solver_ng": 1, "mapping_s": 0., "solver_build_s": 0.,
        "iterations": 3, "timings": {"t_wall_total": .1}, "call_wall_s": .1,
        "original_max_bound_constraint_violation": 0.,
        "discrete_equivalence_certificate_max": 0., "objective": 1.,
    }
    for mode, initial in (("baseline", 0.), ("local", 1.)):
        path = tmp_path / mode / "symbolic-audit"
        path.mkdir(parents=True)
        (path / "nlp0_metrics.json").write_text(json.dumps([metric]))
        np.savez(path / "nlp0_call0.npz", x=[.1, .2, .3], x0=[initial, 0.],
                 lbx=[0., 0.], ubx=[2., 2.], lbg=[0.], ubg=[0.])
    (tmp_path / "baseline" / "result.json").write_text(json.dumps({
        "configurations": {"ipopt": {"pulse_width_scaling": 1.}},
        "results": [{"stimulation_patterns": {"cycle_1": {
            "muscles": {"test": {"pulse_width_s": [.1, .2, .3]}}
        }}}],
    }))
    with pytest.raises(ValueError, match="Frozen NLP input mismatch"):
        summarize(tmp_path)
