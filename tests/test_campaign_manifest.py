"""Campaign scheduling and artifact protection without launching solvers."""

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from cocofest.simulation.campaign_manifest import plan_campaign


def manifest():
    return {
        "schema_version": 1,
        "campaign_id": "pilot",
        "output_root": "results",
        "reserved_cpus": list(range(12)),
        "lanes": {
            "ma57": {"cpus": [12, 13], "runtime_prefix": "runtime"},
            "mumps": {"cpus": [16, 17], "runtime_prefix": "runtime"},
        },
        "cases": [
            {
                "id": "K5-ma57",
                "lane": "ma57",
                "profile": "K5",
                "config": {"cycles": 100},
            },
            {
                "id": "K7-ma57",
                "lane": "ma57",
                "profile": "K7",
                "config": {"cycles": 100},
            },
            {
                "id": "K5-mumps",
                "lane": "mumps",
                "profile": "K5",
                "config": {"cycles": 100, "ipopt_linear_solver": "mumps"},
            },
        ],
    }


def test_independent_lanes_and_sequential_steps_are_read_only(tmp_path):
    plan = plan_campaign(manifest(), root=tmp_path)
    first, second, third = plan["cases"]
    assert [case["status"] for case in plan["cases"]] == ["ready", "waiting", "ready"]
    assert second["depends_on"] == [first["id"]]
    assert first["argv"][:3] == ["taskset", "--cpu-list", "12,13"]
    assert first["config_hash"] != third["config_hash"]
    assert second["output_root"] != first["output_root"]
    for case in plan["cases"]:
        for key in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "JULIA_NUM_THREADS",
        ):
            assert case["environment_updates"][key] == "1"
        assert Path(case["result_json"]).is_relative_to(case["output_root"])
    assert list(tmp_path.iterdir()) == []


def test_existing_matching_record_unblocks_successor_without_certifying(tmp_path):
    first = plan_campaign(manifest(), root=tmp_path)["cases"][0]
    result = Path(first["result_json"])
    result.parent.mkdir(parents=True)
    (result.parent / "effective-configuration.json").write_text(
        json.dumps(first["resolved_configuration"])
    )
    result.write_text(json.dumps({"error": "native failure retained"}))
    cases = plan_campaign(manifest(), root=tmp_path)["cases"]
    assert cases[0]["status"] == "recorded"
    assert cases[1]["status"] == "ready"
    assert json.loads(result.read_text())["error"] == "native failure retained"


def test_partial_and_changed_artifacts_block_reuse(tmp_path):
    first = plan_campaign(manifest(), root=tmp_path)["cases"][0]
    result = Path(first["result_json"])
    result.parent.mkdir(parents=True)
    effective = result.parent / "effective-configuration.json"
    effective.write_text(json.dumps(first["resolved_configuration"]))
    assert (
        plan_campaign(manifest(), root=tmp_path)["cases"][0]["status"] == "incomplete"
    )
    effective.write_text(json.dumps({"config_hash": "different"}))
    cases = plan_campaign(manifest(), root=tmp_path)["cases"]
    assert cases[0]["status"] == "conflict"
    assert cases[1]["status"] == "waiting"


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda m: m["lanes"]["ma57"].update(cpus=[0, 12]), "overlaps"),
        (lambda m: m["lanes"]["mumps"].update(cpus=[12]), "overlaps"),
        (lambda m: m["lanes"]["ma57"].update(cpus=[True]), "integer"),
        (lambda m: m["cases"][0].update(id="../outside"), "identifier"),
        (lambda m: m["cases"][1].update(id="K5-ma57"), "Duplicate"),
        (lambda m: m["cases"][0].update(depends_on=["K7-ma57"]), "earlier"),
        (lambda m: m["cases"][0]["config"].update(output_root="shared"), "managed"),
        (lambda m: m["cases"][0]["config"].update(numeric_threads=2), "managed"),
        (lambda m: m["cases"][0]["config"].update(threads=3), "allocated"),
        (lambda m: m["cases"][0]["config"].update(collocation_degree=3), "conflicts"),
        (lambda m: m.update(unknown="ignored"), "unknown"),
        (lambda m: m.update(schema_version=True), "Unsupported"),
    ],
)
def test_invalid_manifest_rejected_before_launch(tmp_path, mutation, match):
    data = copy.deepcopy(manifest())
    mutation(data)
    with pytest.raises(ValueError, match=match):
        plan_campaign(data, root=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_cli_emits_only_plan(tmp_path):
    source = tmp_path / "manifest.json"
    source.write_text(json.dumps(manifest()))
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "cocofest.simulation.campaign_manifest",
            str(source),
            "--root",
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["dry_run"] is True
    assert list(tmp_path.iterdir()) == [source]
