from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / ".github" / "scripts" / "run_full_horizon.py"
)
SPEC = importlib.util.spec_from_file_location("run_full_horizon_driver", SCRIPT_PATH)
driver = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = driver
SPEC.loader.exec_module(driver)


def _write_report(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "max_cycles": 100,
                "largest_successful_cycles": 42,
                "full_horizon_solver": "ipopt",
                "continuation_step_cycles": 1,
                "jump_objective_relative_tolerance": 0.002,
            }
        ),
        encoding="utf-8",
    )


def test_resume_from_accepts_a_nested_fho_checkpoint_directory(tmp_path):
    campaign = tmp_path / "downloaded-campaign"
    checkpoint = campaign / "full-horizon-0042" / "chance-1"
    checkpoint.mkdir(parents=True)
    _write_report(campaign / driver.REPORT_JSON_NAME)

    assert driver.find_resume_directory(checkpoint) == campaign


def test_resume_defaults_preserve_the_stored_campaign_settings(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    _write_report(campaign / driver.REPORT_JSON_NAME)
    args = driver.parse_arguments(["--resume-from", str(campaign)])
    args.resume = True

    report = driver.apply_run_defaults(args, campaign)

    assert report is not None
    assert args.max_cycles == 100
    assert args.solver == "ipopt"
    assert args.continuation_step_cycles == 1
    assert args.jump_objective_relative_tolerance == 0.002


def test_explicit_resume_target_overrides_the_old_ceiling(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    _write_report(campaign / driver.REPORT_JSON_NAME)
    args = driver.parse_arguments(
        ["--resume-from", str(campaign), "--max-cycles", "150"]
    )
    args.resume = True

    driver.apply_run_defaults(args, campaign)

    assert args.max_cycles == 150
