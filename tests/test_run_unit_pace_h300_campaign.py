import json

import pytest

from scripts.run_unit_pace_h300_campaign import prepare


def _source_manifest(tmp_path):
    models, arms = [], []
    for index, factor in enumerate((.25, .5, 1., 2.)):
        identifier = f"model-{index}"
        seed = tmp_path / identifier / "seed.npz"
        model = tmp_path / identifier / "model.json"
        seed.parent.mkdir(parents=True)
        seed.write_bytes(b"seed")
        model.write_text("{}")
        models.append({"model_id": identifier, "factor": factor, "seed": str(seed),
                       "model_config": str(model), "cpu_id": index})
        for condition in ("rho", "rho-physio"):
            result = tmp_path / identifier / condition / "result.json"
            result.parent.mkdir(parents=True)
            result.write_text("{}")
            arms.append({"id": f"{identifier}/{condition}", "result_path": str(result)})
    manifest = tmp_path / "source.json"
    manifest.write_text(json.dumps({"cycles": 3000, "resistance_nm": .15,
                                    "models": models, "arms": arms}))
    return manifest


def test_prepare_h300_campaign_allocates_distinct_fast_and_slow_cores(tmp_path):
    output = tmp_path / "campaign"
    manifest_path = prepare(source_manifest=_source_manifest(tmp_path), run_directory=output,
                            slow_update_cycles=60, projection_budget_fraction=.9,
                            worker_cpu_offset=4)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["protocol"]["fast_rho_cpu_ids"] == [0, 1, 2, 3]
    assert manifest["protocol"]["projection_worker_cpu_ids"] == [4, 5, 6, 7]
    for index, model in enumerate(manifest["models"]):
        weights = json.loads(open(model["weights_config"]).read())
        policy = weights["policy"]
        assert policy["update_every_cycles"] == 60
        assert policy["projection_horizon_cycles"] == 300
        assert policy["projection_budget_seconds"] == pytest.approx(54.)
        assert policy["projection_worker_cpu_ids"] == [4 + index]


def test_prepare_rejects_overlapping_worker_cores(tmp_path):
    with pytest.raises(ValueError, match="must not overlap"):
        prepare(source_manifest=_source_manifest(tmp_path), run_directory=tmp_path / "campaign",
                slow_update_cycles=60, projection_budget_fraction=.9, worker_cpu_offset=3)
