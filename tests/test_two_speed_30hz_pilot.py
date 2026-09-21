"""Scientific comparability gates for the isolated two-speed pilot."""

import json
import sys

import pytest

from scripts.run_two_speed_30hz_pilot import build_manifest, summarize


def manifest(tmp_path):
    source = tmp_path / "model.json"
    source.write_text(json.dumps({"muscles": {"Biceps": {}}}))
    return build_manifest(python=sys.executable, model_config=source,
                          reduced_profile=source, hsl_library=source,
                          output_directory=tmp_path / "run", cycles=3, update_every_cycles=2)


def write_results(m, *, missing_states=False, replay_cycles=3, state_offset=0.0):
    from pathlib import Path
    for case in m["cases"]:
        if case["kind"] != "ocp":
            continue
        state_names = ["theta", "omega"] + [f"{c}_Biceps" for c in ("Cn", "F", "A", "Tau1", "Km")]
        if missing_states:
            state_names = ["theta"]
        states = {name: {"start": [state_offset if case["name"] == "slow-pace" else 0.0]}
                  for name in state_names}
        doc = {"configurations": {"ipopt": {"collocation_degree": 5,
               "calcium_stimulation_interval_s": 1. / 30., "ipopt_linear_solver": "ma57"}},
               "results": [{"solver": "ipopt", "solver_success": True, "physical_success": True,
               "nlp_validated_cycles": 3, "state_boundary_snapshots": {"cycle_1": {"states": states}},
               "high_accuracy_trace_rollout": {"available": True, "covered_cycles": 3, "cycle_count": replay_cycles,
               "maximum_absolute_endpoint_error_by_state": {"theta": 1e-5}}}]}
        path = Path(case["result"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc))
        if case["name"] == "slow-pace":
            (path.parent / "weights.jsonl").write_text(json.dumps(
                {"event": "boundary", "status": "applied", "cycle_index": 2}) + "\n")


def test_manifest_requires_observable_slow_update_and_identical_ocp_grids(tmp_path):
    m = manifest(tmp_path)
    ocps = m["cases"][:3]
    for case in ocps:
        argv = case["argv"]
        for flag, value in (("--solvers", "ipopt"), ("--ipopt-linear-solver", "ma57"),
                            ("--stimulations-per-cycle", "30"), ("--ipopt-collocation-degree", "5")):
            assert argv[argv.index(flag) + 1] == value
        assert "--single-shot" not in argv
    assert ocps[1]["argv"][ocps[1]["argv"].index("--common-initial-solution") + 1] == (
        ocps[2]["argv"][ocps[2]["argv"].index("--common-initial-solution") + 1])
    assert "--model-config" in m["cases"][3]["argv"]
    assert not m["projection_drives_ocp"]


@pytest.mark.parametrize("kwargs", [{"missing_states": True}, {"replay_cycles": 1}, {"state_offset": 1e-3}])
def test_incomplete_or_unmatched_evidence_never_passes(tmp_path, kwargs):
    m = manifest(tmp_path)
    write_results(m, **kwargs)
    assert not summarize(m)["closed_loop_pilot_gate_passed"]


def test_gate_requires_real_slow_update_and_complete_matching_state(tmp_path):
    from pathlib import Path
    m = manifest(tmp_path)
    write_results(m)
    assert summarize(m)["closed_loop_pilot_gate_passed"]
    journal = Path(m["cases"][2]["directory"]) / "weights.jsonl"
    journal.write_text(json.dumps({"event": "boundary", "status": "applied", "cycle_index": 0}) + "\n")
    assert not summarize(m)["closed_loop_pilot_gate_passed"]


def test_missing_runs_cannot_pass(tmp_path):
    assert not summarize(manifest(tmp_path))["closed_loop_pilot_gate_passed"]
