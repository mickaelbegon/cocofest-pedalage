from pathlib import Path

import pytest

from scripts.plan_capacity_frontier_epsilon_replication import plan


def test_plan_registers_every_missing_epsilon_cut_with_immutable_receipt_hashes(tmp_path):
    c120, c160 = tmp_path / "c120.json", tmp_path / "c160.json"
    c120.write_text("c120")
    c160.write_text("c160")
    result = plan(receipts={120: c120, 160: c160}, output_directory=tmp_path / "results",
                  epsilon_values=(.4, .8), lower_ratio=.1, upper_ratio=1., max_iterations=800)
    assert len(result["jobs"]) == 12
    assert {row["checkpoint_cycle"] for row in result["jobs"]} == {120, 160}
    assert all(row["support"] == [0., 0., 0., 1.] for row in result["jobs"])
    assert len(result["plan_sha256"]) == 64


def test_plan_rejects_an_endpoint_epsilon_value(tmp_path):
    c120, c160 = tmp_path / "c120.json", tmp_path / "c160.json"
    c120.write_text("c120")
    c160.write_text("c160")
    with pytest.raises(ValueError, match="strictly"):
        plan(receipts={120: c120, 160: c160}, output_directory=Path(tmp_path / "results"),
             epsilon_values=(.1,), lower_ratio=.1, upper_ratio=1., max_iterations=800)


def test_plan_accepts_one_explicit_holdout_checkpoint(tmp_path):
    receipt = tmp_path / "c140.json"
    receipt.write_text("c140")
    result = plan(receipts={140: receipt}, output_directory=tmp_path / "results",
                  epsilon_values=(.4, .6, .8), lower_ratio=.1, upper_ratio=1., max_iterations=800)
    assert len(result["jobs"]) == 9
    assert {row["checkpoint_cycle"] for row in result["jobs"]} == {140}
