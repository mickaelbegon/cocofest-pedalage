from pathlib import Path

from scripts.run_capacity_frontier_epsilon_replication import command_for


def test_command_preserves_the_physical_capacity_box_and_unconstrained_coordinates(tmp_path):
    receipt, python = tmp_path / "receipt.json", tmp_path / "python"
    receipt.write_text("{}"); python.write_text("")
    command = command_for({"receipt": str(receipt), "side": "left", "capacity_domain": [.1, 1.],
                           "support": [0., 0., 0., 1.], "epsilon_upper": [None, .6, None, None],
                           "max_iterations": 800, "output": str(tmp_path / "result.json")}, python=python, cpu=27)
    assert command[command.index("--cpu") + 1] == "27"
    assert command[command.index("--epsilon-upper") + 1:command.index("--epsilon-upper") + 5] == ["nan", "0.59999999999999998", "nan", "nan"]
    assert command[command.index("--lower-ratio") + 1] == "0.10000000000000001"
