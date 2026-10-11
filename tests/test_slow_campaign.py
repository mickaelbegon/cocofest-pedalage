from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from cocofest.cycling.slow.campaign import SlowCampaignInputs, build_slow_campaign_plan


@pytest.fixture
def inputs(tmp_path):
    paths = {}
    for name in ("model_left", "model_right", "rho_config", "baseline_source", "split_config"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"source": name}))
        paths[name] = path
    return SlowCampaignInputs(**paths, horizon_cycles=228)


def test_ablation_factors_reference_and_closed_gate(inputs):
    plan = build_slow_campaign_plan(inputs)
    a0, a1 = plan["cases"]
    assert a0["torque_nm"] == a1["initial_torque_nm"] == {"left": .96, "right": .96}
    assert a1["total_torque_nm"] == 1.92
    assert a0["muscle_weights"] == a1["muscle_weights"] == {"policy": "unit", "value_for_every_muscle": 1.0}
    assert not a0["slow_split_enabled"] and a1["slow_split_enabled"]
    assert a1["update_every_cycles"] == 10
    assert a1["predeclared_criteria"]["reference_certified_cycles"] == 211
    assert plan["execution"]["online_activation"] is False
    assert plan["execution"]["launch_supported"] is False
    assert plan["results"] is None
    assert plan["sources"]["baseline_source"]["sha256"] == hashlib.sha256(inputs.baseline_source.read_bytes()).hexdigest()


def test_plan_is_read_only_reproducible_and_sensitive_to_inputs(inputs):
    before = sorted(inputs.rho_config.parent.iterdir())
    first = build_slow_campaign_plan(inputs)
    assert first == build_slow_campaign_plan(inputs)
    assert before == sorted(inputs.rho_config.parent.iterdir())
    inputs.rho_config.write_text('{"online_activation": true, "phi_validated": true}')
    changed = build_slow_campaign_plan(inputs)
    assert first["plan_sha256"] != changed["plan_sha256"]
    assert changed["execution"]["online_activation"] is False
    assert changed["execution"]["validation_status"]["phi"] == "not_assessed"


def test_short_horizon_cannot_claim_reference_reproduction(inputs):
    plan = build_slow_campaign_plan(replace(inputs, horizon_cycles=211))
    criteria = plan["cases"][1]["predeclared_criteria"]
    assert criteria["horizon_sufficient_to_observe_reference_endpoint"] is False
    assert criteria["reference_is_observed_by_builder"] is False


@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_invalid_cycle_counts(inputs, value):
    with pytest.raises(ValueError, match="positive integer"):
        build_slow_campaign_plan(replace(inputs, horizon_cycles=value))
    with pytest.raises(ValueError, match="positive integer"):
        build_slow_campaign_plan(replace(inputs, update_every_cycles=value))


@pytest.mark.parametrize("contents", ["[]", "{}", '{"value": NaN}'])
def test_invalid_config(inputs, contents):
    inputs.rho_config.write_text(contents)
    with pytest.raises(ValueError):
        build_slow_campaign_plan(inputs)


def test_missing_reference(inputs):
    with pytest.raises(FileNotFoundError):
        build_slow_campaign_plan(replace(inputs, baseline_source=inputs.baseline_source.parent / "absent"))


def test_cli_only_prints_plan(inputs):
    root = Path(__file__).resolve().parents[1]
    argv = [sys.executable, str(root / "scripts/slow_layer/build_campaign.py")]
    for name in ("model_left", "model_right", "rho_config", "baseline_source", "split_config", "horizon_cycles"):
        argv.extend(["--" + name.replace("_", "-"), str(getattr(inputs, name))])
    before = sorted(inputs.rho_config.parent.iterdir())
    result = subprocess.run(argv, cwd=inputs.rho_config.parent, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == build_slow_campaign_plan(inputs)
    assert before == sorted(inputs.rho_config.parent.iterdir())
