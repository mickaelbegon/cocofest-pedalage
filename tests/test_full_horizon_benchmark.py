from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "scripts"
    / "run_full_horizon_benchmark.py"
)
SPEC = importlib.util.spec_from_file_location("run_full_horizon_benchmark", SCRIPT_PATH)
full_horizon = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = full_horizon
SPEC.loader.exec_module(full_horizon)


def test_rho_initial_state_homotopy_minimum_step_is_finer_than_one_over_32():
    assert full_horizon.RHO_INITIAL_STATE_HOMOTOPY_MIN_STEP == pytest.approx(1 / 256)
    assert full_horizon.RHO_INITIAL_STATE_HOMOTOPY_MIN_STEP < 1 / 32


@pytest.mark.parametrize(
    ("maximum", "expected_tail"),
    (
        (2, [2]),
        (3, [2, 3]),
        (32, [30, 31, 32]),
        (60, [58, 59, 60]),
        (100, [98, 99, 100]),
    ),
)
def test_horizon_sweep_targets_grow_one_cycle_at_a_time(maximum, expected_tail):
    targets = full_horizon.horizon_sweep_targets(maximum)

    assert targets == sorted(set(targets))
    assert targets[-1] == maximum
    assert targets[-len(expected_tail) :] == expected_tail


def test_horizon_sweep_requires_the_two_rho_bootstrap_cycles():
    with pytest.raises(ValueError, match="at least two"):
        full_horizon.horizon_sweep_targets(1)


def test_rho_only_is_an_explicit_benchmark_mode(tmp_path):
    args = full_horizon.build_parser().parse_args(
        [
            "--workspace",
            str(tmp_path),
            "--seed-dir",
            str(tmp_path / "seed"),
            "--output-dir",
            str(tmp_path / "output"),
            "--max-cycles",
            "150",
            "--n-threads",
            "4",
            "--rho-only",
        ]
    )

    assert args.rho_only is True


def test_full_horizon_cli_accepts_an_unlimited_attempt_timeout(tmp_path):
    arguments = [
        "--workspace",
        str(tmp_path),
        "--seed-dir",
        str(tmp_path / "seed"),
        "--output-dir",
        str(tmp_path / "output"),
        "--max-cycles",
        "200",
        "--n-threads",
        "4",
    ]

    assert full_horizon.build_parser().parse_args(arguments).attempt_timeout_s is None
    assert (
        full_horizon.build_parser().parse_args(
            [*arguments, "--attempt-timeout-s", "none"]
        ).attempt_timeout_s
        is None
    )


def test_rho_only_writes_a_complete_report_and_skips_full_horizon(
    tmp_path, monkeypatch
):
    output = tmp_path / "output"
    args = full_horizon.build_parser().parse_args(
        [
            "--workspace",
            str(tmp_path),
            "--seed-dir",
            str(tmp_path / "seed"),
            "--output-dir",
            str(output),
            "--max-cycles",
            "5",
            "--n-threads",
            "2",
            "--rho-only",
        ]
    )
    monitored_commands = []

    def fake_run_monitored(command, **kwargs):
        monitored_commands.append(command)
        return full_horizon.MonitoredRun(
            command=command,
            return_code=0,
            peak_rss_bytes=1024,
            elapsed_s=0.1,
            memory_limit_exceeded=False,
            timed_out=False,
            log_path=str(kwargs["log_path"]),
        )

    monkeypatch.setattr(full_horizon, "available_memory_bytes", lambda: 16 * full_horizon.GIB)
    monkeypatch.setattr(full_horizon, "run_monitored", fake_run_monitored)
    monkeypatch.setattr(full_horizon, "_benchmark_validated_cycles", lambda *_args, **_kwargs: 5)
    monkeypatch.setattr(full_horizon, "_seed_cycle_count", lambda _path: 5)
    monkeypatch.setattr(full_horizon, "_log_has_unknown_mumps_warning", lambda _path: False)

    assert full_horizon.run(args) == 0
    report = json.loads((output / "full-horizon-report.json").read_text())
    assert report["stop_reason"] == "rho_only_completed"
    assert report["rho_available_cycles"] == 5
    assert report["full_horizon_attempts"] == []
    assert len(monitored_commands) == 1


def test_full_horizon_cli_preserves_an_explicit_resistive_signed_torque():
    args = full_horizon.build_parser().parse_args(
        [
            "--workspace", ".", "--seed-dir", ".", "--output-dir", ".",
            "--max-cycles", "50", "--n-threads", "1", "--signed-crank-torque", "0.22",
        ]
    )
    command = full_horizon._common_solver_options(args)
    assert command[command.index("--signed-crank-torque") + 1] == "0.22"
    assert "--crank-assistance" not in command


def test_rho_only_rejects_a_partial_reference(tmp_path, monkeypatch):
    args = full_horizon.build_parser().parse_args(
        [
            "--workspace",
            str(tmp_path),
            "--seed-dir",
            str(tmp_path / "seed"),
            "--output-dir",
            str(tmp_path / "output"),
            "--max-cycles",
            "5",
            "--n-threads",
            "2",
            "--rho-only",
        ]
    )
    monkeypatch.setattr(full_horizon, "available_memory_bytes", lambda: 16 * full_horizon.GIB)
    monkeypatch.setattr(
        full_horizon,
        "run_monitored",
        lambda command, **kwargs: full_horizon.MonitoredRun(
            command=command,
            return_code=0,
            peak_rss_bytes=1024,
            elapsed_s=0.1,
            memory_limit_exceeded=False,
            timed_out=False,
            log_path=str(kwargs["log_path"]),
        ),
    )
    monkeypatch.setattr(full_horizon, "_benchmark_validated_cycles", lambda *_args, **_kwargs: 3)
    monkeypatch.setattr(full_horizon, "_seed_cycle_count", lambda _path: 3)
    monkeypatch.setattr(full_horizon, "_log_has_unknown_mumps_warning", lambda _path: False)
    monkeypatch.setattr(full_horizon, "_benchmark_payload_is_readable", lambda _path: True)

    assert full_horizon.run(args) == 2
    report = json.loads((args.output_dir / "full-horizon-report.json").read_text())
    assert report["stop_reason"] == "rho_incomplete"


@pytest.fixture
def bootstrap_campaign(tmp_path, monkeypatch):
    """Exercise real commands, seed slicing, certificates and reports sans NLP."""

    args = full_horizon.build_parser().parse_args(
        [
            "--workspace", str(tmp_path), "--seed-dir", str(tmp_path / "seed"),
            "--output-dir", str(tmp_path / "output"), "--max-cycles", "5",
            "--n-threads", "12", "--continuation-step-cycles", "3",
        ]
    )
    state = SimpleNamespace(args=args, outcomes=[], commands=[], continuations=[])
    args.seed_dir.mkdir(parents=True)
    np.savez(args.seed_dir / "common-reduced.npz", **{
        "metadata__json": np.asarray(json.dumps({
            "cycles_per_window": 1, "mechanical_formulation": "reduced",
            "warmup_cycles_consumed": 1, "producer_mode": "certified_common_seed",
        })),
        # Same first state, intentionally different trajectory than RHO_1.
        "states__theta": np.array([[0.0, 10.0, 20.0]]),
        "controls__pulse_width_Biceps": np.array([[0.3, 0.4]]),
    })

    def fake_run(command, **kwargs):
        state.commands.append(command)
        is_fho = "--single-shot" in command
        cycles = int(command[command.index("--n-windows") + 1])
        outcome = state.outcomes.pop(0) if is_fho else "success"
        success = outcome == "success"
        result_path = Path(command[command.index("--output-json") + 1])
        result_path.parent.mkdir(parents=True, exist_ok=True)
        error = (
            "ValueError: Common initial solution has incompatible warmup_cycles_consumed: expected 0, got 1"
            if outcome == "seed_handoff_error" else None
        )
        if is_fho:
            assert "--ipopt-disable-standard-warmup" in command
            assert "--adopt-common-initial-solution-warmup-cycles" in command
            assert "--standard-warmup-seed" not in command
            assert "--legacy-standard-warmup-seed-signed-torque" not in command
            assert "--standard-warmup-seed-continuation" not in command
            with np.load(command[command.index("--common-initial-solution") + 1]) as data:
                metadata = full_horizon._load_metadata(data)
                assert metadata["warmup_cycles_consumed"] == 1
                assert metadata["cycles_per_window"] == cycles
                if cycles == 1:
                    assert metadata["producer_mode"] == "certified_common_seed"
                    np.testing.assert_array_equal(data["states__theta"], [[0.0, 10.0, 20.0]])
                    np.testing.assert_array_equal(data["controls__pulse_width_Biceps"], [[0.3, 0.4]])
                else:
                    # FHO_2 retains the second RHO cycle around its separately
                    # applied certified FHO_1 prefix.
                    np.testing.assert_array_equal(data["states__theta"][..., 2:], [[2.0, 3.0, 4.0]])
        result_path.write_text(json.dumps({
            "results": [{
                "success": success, "solver_success": success,
                "physical_success": success, "solver": "ipopt",
                "mode": "single_shot" if is_fho else "rho",
                "covered_cycles": cycles if success else 0,
                "physically_validated_cycles": cycles if success else 0,
                "error": error,
            }],
            "configurations": {"ipopt": {
                "single_shot": is_fho, "mechanical_formulation": "reduced",
                "cycles_per_window": cycles if is_fho else 1,
                "n_windows": cycles, "use_sx": not is_fho,
                "ipopt_linear_solver": "ma57",
            }},
        }))
        if success:
            flag = "--common-initial-solution-output" if is_fho else "--receding-horizon-solution-output"
            solution_path = Path(command[command.index(flag) + 1])
            np.savez(solution_path, **{
                "metadata__json": np.asarray(json.dumps({
                    "cycles_per_window": cycles, "mechanical_formulation": "reduced",
                    "warmup_cycles_consumed": 1,
                })),
                "states__theta": np.arange(cycles * 2 + 1, dtype=float)[None, :],
                "controls__pulse_width_Biceps": np.ones((1, cycles * 2)),
            })
        Path(kwargs["log_path"]).write_text(error or "solver finished\n")
        return full_horizon.MonitoredRun(
            command=command, return_code=1 if error else 0,
            peak_rss_bytes=1024, elapsed_s=0.1,
            memory_limit_exceeded=outcome == "memory_limit",
            timed_out=outcome == "timeout", log_path=str(kwargs["log_path"]),
        )

    def fake_continue(continuation_args, **kwargs):
        state.continuations.append((continuation_args, kwargs))
        return 0

    monkeypatch.setattr(full_horizon, "available_memory_bytes", lambda: 16 * full_horizon.GIB)
    monkeypatch.setattr(full_horizon, "run_monitored", fake_run)
    monkeypatch.setattr(full_horizon, "_continue_adaptively", fake_continue)
    return state


def test_bootstrap_recovers_two_cycles_with_the_certified_one_cycle_prefix(bootstrap_campaign):
    campaign = bootstrap_campaign
    campaign.outcomes[:] = ["solver_failure", "success", "success"]

    assert full_horizon.run(campaign.args) == 0

    report = json.loads((campaign.args.output_dir / "full-horizon-report.json").read_text())
    attempts = report["full_horizon_attempts"]
    assert [(item["cycles"], item["success"]) for item in attempts] == [(2, False), (1, True), (2, True)]
    assert attempts[-1]["prefix_solution_path"] == attempts[1]["solution_path"]
    assert attempts[1]["seed_origin"] == "common_initial_solution"
    assert attempts[1]["seed_source_path"] == str(campaign.args.seed_dir / "common-reduced.npz")
    assert attempts[-1]["chance"] == 2
    assert report["largest_successful_cycles"] == 2
    assert report["solver_gap_cycles"] == []
    continuation_args, continuation = campaign.continuations[0]
    assert continuation_args.continuation_step_cycles == 3
    assert continuation["current_cycles"] == 2
    assert continuation["current_full_solution"] == Path(attempts[-1]["solution_path"])
    assert full_horizon.adaptive_continuation_target(2, 5, continuation_args.continuation_step_cycles) == 5


@pytest.mark.parametrize(
    ("outcomes", "stop_reason", "failing_cycles", "largest", "exit_code"),
    [
        (["solver_failure", "seed_handoff_error"], "seed_handoff_error", 1, 0, 3),
        (["solver_failure", "solver_failure"], "solver_failure", 1, 0, 0),
        (["solver_failure", "success", "seed_handoff_error"], "seed_handoff_error", 2, 1, 3),
        (["memory_limit"], "memory_limit", 2, 0, 0),
        (["timeout"], "timeout", 2, 0, 0),
    ],
)
def test_bootstrap_reports_the_actual_failure_without_claiming_fatigue(
    bootstrap_campaign, outcomes, stop_reason, failing_cycles, largest, exit_code
):
    campaign = bootstrap_campaign
    campaign.outcomes[:] = outcomes

    assert full_horizon.run(campaign.args) == exit_code

    report = json.loads((campaign.args.output_dir / "full-horizon-report.json").read_text())
    assert report["stop_reason"] == stop_reason
    assert report["largest_successful_cycles"] == largest
    assert report["full_horizon_attempts"][-1]["cycles"] == failing_cycles
    assert report["full_horizon_attempts"][-1]["physiological_limit_certified"] is False
    assert report["solver_gap_cycles"] == ([failing_cycles] if stop_reason == "solver_failure" else [])
    assert not campaign.continuations


@pytest.mark.parametrize("has_certified_one", [False, True])
def test_resume_bootstrap_reuses_reference_and_last_certificate_without_overwriting(
    bootstrap_campaign, has_certified_one
):
    campaign = bootstrap_campaign
    campaign.outcomes[:] = (
        ["solver_failure", "success", "solver_failure"]
        if has_certified_one else ["solver_failure", "seed_handoff_error"]
    )
    full_horizon.run(campaign.args)
    report_path = campaign.args.output_dir / "full-horizon-report.json"
    old_report = json.loads(report_path.read_text())
    old_logs = {item["log_path"]: Path(item["log_path"]).read_text() for item in old_report["full_horizon_attempts"]}
    original_command_count = len(campaign.commands)
    campaign.args.resume = True
    campaign.outcomes[:] = ["success"] if has_certified_one else ["success", "success"]

    assert full_horizon.run(campaign.args) == 0

    new_commands = campaign.commands[original_command_count:]
    assert all("--single-shot" in command for command in new_commands)
    assert [int(command[command.index("--n-windows") + 1]) for command in new_commands] == ([2] if has_certified_one else [1, 2])
    report = json.loads(report_path.read_text())
    assert report["largest_successful_cycles"] == 2
    assert report["full_horizon_attempts"][:len(old_report["full_horizon_attempts"])] == old_report["full_horizon_attempts"]
    assert report["full_horizon_attempts"][-1]["chance"] == (3 if has_certified_one else 2)
    assert all(Path(path).read_text() == content for path, content in old_logs.items())
    if has_certified_one:
        assert report["bootstrap_reuse_events"][-1]["cycles"] == 1
    assert len(campaign.continuations) == 1


def test_rho_prefix_stop_assessment_does_not_promote_solver_failure_to_physiology(
    tmp_path,
):
    result = tmp_path / "rho.json"
    result.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "first_failed_rho": 133,
                        "fatigue_endurance_outcome": {"accepted": False},
                        "physical_crank_diagnostics": {
                            "issues": ["wheel_cycle_grid_mismatch"]
                        },
                        "windows": [
                            {
                                "validated": False,
                                "solver_converged": False,
                                "primal_feasible": False,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assessment = full_horizon._rho_prefix_stop_assessment(result)

    assert assessment["classification"] == "numerical_or_optimization_failure"
    assert assessment["failed_cycle"] == 133
    assert assessment["grid_or_transfer_diagnostic_present"] is True
    assert assessment["physiological_limit_certified"] is False


def test_rho_prefix_stop_assessment_requires_explicit_endurance_acceptance(tmp_path):
    result = tmp_path / "rho.json"
    result.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "first_failed_rho": 18,
                        "fatigue_endurance_outcome": {"accepted": True},
                        "physical_crank_diagnostics": {"issues": []},
                        "windows": [
                            {
                                "validated": False,
                                "solver_converged": True,
                                "primal_feasible": True,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assessment = full_horizon._rho_prefix_stop_assessment(result)

    assert assessment["classification"] == "physiological_endurance_limit"
    assert assessment["physiological_limit_certified"] is True


def test_rho_prefix_stop_assessment_marks_an_isolated_grid_issue_inconclusive(
    tmp_path,
):
    result = tmp_path / "rho.json"
    result.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "first_failed_rho": 18,
                        "fatigue_endurance_outcome": {"accepted": False},
                        "physical_crank_diagnostics": {
                            "issues": ["wheel_cycle_grid_mismatch"]
                        },
                        "windows": [
                            {
                                "validated": False,
                                "solver_converged": True,
                                "primal_feasible": True,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assessment = full_horizon._rho_prefix_stop_assessment(result)

    assert assessment["classification"] == "grid_or_transfer_diagnostic"
    assert assessment["physiological_limit_certified"] is False


def test_resume_rebases_artifact_paths_after_the_campaign_is_moved(tmp_path):
    campaign = tmp_path / "downloaded-campaign"
    rho = campaign / "rho-reduced" / "concatenated-solution.npz"
    fho = campaign / "full-horizon-0042" / "chance-1" / "full-solution.npz"
    rho.parent.mkdir(parents=True)
    fho.parent.mkdir(parents=True)
    rho.touch()
    fho.touch()
    report = {
        "rho": {"seed_path": "/old/runner/results/rho-reduced/concatenated-solution.npz"},
        "full_horizon_attempts": [
            {
                "solution_path": (
                    "/old/runner/results/full-horizon-0042/chance-1/full-solution.npz"
                )
            }
        ],
        "external_seed_path": "/unrelated/benchmark-seed/common-reduced.npz",
    }

    relocated = full_horizon.rebase_report_artifact_paths(report, campaign)

    assert relocated == 2
    assert Path(report["rho"]["seed_path"]) == rho
    assert Path(report["full_horizon_attempts"][0]["solution_path"]) == fho
    assert report["external_seed_path"] == "/unrelated/benchmark-seed/common-reduced.npz"


def test_resume_attempt_numbers_include_unreported_checkpoint_directories(tmp_path):
    fho_attempt = tmp_path / "full-horizon-0050" / "chance-3"
    rho_attempt = tmp_path / "rho-extension-after-fho-0049" / "retry-04" / "stage-01"
    fho_attempt.mkdir(parents=True)
    rho_attempt.mkdir(parents=True)

    assert full_horizon._next_horizon_chance([], 50, tmp_path) == 4
    assert full_horizon._next_extension_run_number([], 50, tmp_path) == 5


def test_adaptive_target_uses_three_cycles_and_clips_the_tail():
    assert full_horizon.adaptive_continuation_target(2, 20, 3) == 5
    assert full_horizon.adaptive_continuation_target(20, 22, 3) == 22


def test_jump_objective_gate_rejects_excessive_degradation():
    accepted = full_horizon.objective_gate(100.4, 100.0, 0.005)
    rejected = full_horizon.objective_gate(100.6, 100.0, 0.005)

    assert accepted["passes"]
    assert accepted["relative_degradation"] == pytest.approx(0.004)
    assert not rejected["passes"]


def test_jump_objective_gate_marks_missing_measurement_incomparable():
    gate = full_horizon.objective_gate(None, 100.0, 0.005)

    assert not gate["comparable"]


def test_adaptive_continuation_falls_back_to_one_then_retries_jump(
    tmp_path, monkeypatch
):
    objectives = tmp_path / "objectives"
    objectives.mkdir()

    def result(name, objective):
        path = objectives / f"{name}.json"
        path.write_text(
            json.dumps({"results": [{"window_objective_sum": objective}]}),
            encoding="utf-8",
        )
        return path

    bootstrap_solution = tmp_path / "fho-2.npz"
    bootstrap_solution.touch()
    bootstrap_result = result("fho-2", 2.0)
    report = {
        "full_horizon_attempts": [
            {
                "cycles": 2,
                "success": True,
                "accepted_for_continuation": True,
                "result_path": str(bootstrap_result),
                "solution_path": str(bootstrap_solution),
            }
        ],
        "extension_rho_attempts": [],
        "adaptive_fallback_events": [],
        "largest_successful_cycles": 2,
        "homotopy_constructed_cycles": 2,
    }
    args = SimpleNamespace(
        output_dir=tmp_path / "output",
        continuation_step_cycles=3,
        jump_objective_relative_tolerance=0.005,
        max_cycles=5,
    )
    extension_number = 0
    extension_runs = []

    def fake_extension(call_args, **kwargs):
        nonlocal extension_number
        extension_number += 1
        extension_runs.append(
            (kwargs["after_cycles"] + 1, kwargs.get("run_number", 1))
        )
        solution = tmp_path / f"rho-{extension_number}.npz"
        solution.touch()
        return {
            "target_cycle": kwargs["after_cycles"] + 1,
            "run_number": kwargs.get("run_number", 1),
            "success": True,
            "infrastructure_error": False,
            "failure_kind": None,
            "solution_path": str(solution),
            "result_path": str(result(f"rho-{extension_number}", 1.0)),
        }

    horizon_targets = []
    horizon_chances = []

    def fake_horizon(call_args, **kwargs):
        cycles = kwargs["cycles"]
        horizon_targets.append(cycles)
        horizon_chances.append(kwargs["chance"])
        objective = 10.0 if horizon_targets == [5] else float(cycles)
        solution = tmp_path / f"fho-{cycles}-{len(horizon_targets)}.npz"
        solution.touch()
        return {
            "cycles": cycles,
            "success": True,
            "infrastructure_error": False,
            "failure_kind": None,
            "solution_path": str(solution),
            "result_path": str(
                result(f"fho-{cycles}-{len(horizon_targets)}", objective)
            ),
        }

    monkeypatch.setattr(full_horizon, "_run_extension_rho", fake_extension)
    monkeypatch.setattr(full_horizon, "_run_horizon_attempt", fake_horizon)
    monkeypatch.setattr(
        full_horizon,
        "append_rho_extension_cycle",
        lambda source, extension, output: (
            output.parent.mkdir(parents=True, exist_ok=True), output.touch()
        ),
    )
    monkeypatch.setattr(full_horizon, "_write_report", lambda *args: None)
    monkeypatch.setattr(full_horizon, "_write_markdown", lambda *args: None)

    return_code = full_horizon._continue_adaptively(
        args,
        report=report,
        report_path=tmp_path / "report.json",
        markdown_path=tmp_path / "report.md",
        rho_seed_path=tmp_path / "rho-reference.npz",
        effective_max_cycles=5,
        current_cycles=2,
        current_full_solution=bootstrap_solution,
        rss_limit_bytes=1024,
    )

    assert return_code == 0
    assert horizon_targets == [5, 3, 5]
    assert horizon_chances == [1, 1, 2]
    assert extension_runs == [(3, 1), (4, 1), (5, 1), (3, 2), (4, 2), (5, 2)]
    assert report["largest_successful_cycles"] == 5
    assert report["adaptive_fallback_events"] == [
        {
            "from_cycles": 2,
            "rejected_target_cycles": 5,
            "rejected_step_cycles": 3,
            "reason": "jump_objective_degradation",
            "fallback_step_cycles": 1,
        }
    ]


def test_adaptive_continuation_can_bypass_a_failed_rho_extension_with_fho_bridge(
    tmp_path, monkeypatch
):
    """A RHO ceiling must not itself be reported as an FHO ceiling."""

    bootstrap_solution = tmp_path / "fho-2.npz"
    bootstrap_solution.touch()
    bridge_solution = tmp_path / "fho-3.npz"
    bridge_solution.touch()
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps({"results": [{"window_objective_sum": 3.0}]}),
        encoding="utf-8",
    )
    report = {
        "full_horizon_attempts": [
            {
                "cycles": 2,
                "success": True,
                "accepted_for_continuation": True,
                "result_path": str(result),
                "solution_path": str(bootstrap_solution),
            }
        ],
        "extension_rho_attempts": [],
        "adaptive_fallback_events": [],
        "largest_successful_cycles": 2,
        "homotopy_constructed_cycles": 2,
    }
    args = SimpleNamespace(
        output_dir=tmp_path / "output",
        continuation_step_cycles=1,
        jump_objective_relative_tolerance=0.005,
        max_cycles=3,
        allow_fho_terminal_bridge=True,
    )

    monkeypatch.setattr(
        full_horizon,
        "_run_extension_rho",
        lambda *_args, **_kwargs: {
            "success": False,
            "infrastructure_error": False,
            "failure_kind": "solver_failure",
        },
    )
    observed = {}

    def bridge(call_args, **kwargs):
        observed.update(kwargs)
        return {
            "cycles": 3,
            "success": True,
            "infrastructure_error": False,
            "failure_kind": None,
            "result_path": str(result),
            "solution_path": str(bridge_solution),
        }

    monkeypatch.setattr(full_horizon, "_run_fho_terminal_bridge", bridge)
    monkeypatch.setattr(full_horizon, "_write_report", lambda *args: None)
    monkeypatch.setattr(full_horizon, "_write_markdown", lambda *args: None)

    assert full_horizon._continue_adaptively(
        args,
        report=report,
        report_path=tmp_path / "report.json",
        markdown_path=tmp_path / "report.md",
        rho_seed_path=tmp_path / "rho-reference.npz",
        effective_max_cycles=3,
        current_cycles=2,
        current_full_solution=bootstrap_solution,
        rss_limit_bytes=1024,
    ) == 0
    assert observed["source_full_solution"] == bootstrap_solution
    assert observed["target_cycles"] == 3
    assert report["largest_successful_cycles"] == 3
    assert report["terminal_bridge_attempts"] == [
        {
            "from_cycles": 2,
            "to_cycles": 3,
            "success": True,
            "failure_kind": None,
            "rho_extension_failure": "rho_extension_solver_failure",
            "result_path": str(result),
            "solution_path": str(bridge_solution),
        }
    ]


def test_reference_ceiling_does_not_prevent_plus_three_chain(tmp_path, monkeypatch):
    """FHO65 + local RHO66/67/68 reaches FHO68 despite the reference ceiling."""

    reference = tmp_path / "rho-reference.npz"
    np.savez(reference, metadata__json=np.asarray(json.dumps({"cycles_per_window": 67})))
    source = tmp_path / "fho-65.npz"
    source.touch()
    report = {
        "full_horizon_attempts": [], "extension_rho_attempts": [],
        "adaptive_fallback_events": [], "largest_successful_cycles": 65,
        "homotopy_constructed_cycles": 65,
    }
    args = SimpleNamespace(
        output_dir=tmp_path / "output", continuation_step_cycles=3,
        jump_objective_relative_tolerance=0.005, max_cycles=68,
        allow_fho_terminal_bridge=True,
    )
    horizon_targets, bridge_targets = [], []
    extension_sources = []

    def extension(call_args, **kwargs):
        cycle = kwargs["after_cycles"] + 1
        extension_sources.append((cycle, kwargs["source_full_solution"]))
        return {
            "target_cycle": cycle, "success": True, "failure_kind": None,
            "infrastructure_error": False,
            "result_path": str(tmp_path / f"rho-{cycle}.json"),
            "solution_path": str(tmp_path / f"rho-{cycle}.npz"),
        }

    def horizon(call_args, **kwargs):
        cycles = kwargs["cycles"]
        horizon_targets.append(cycles)
        return {
            "cycles": cycles, "success": True, "infrastructure_error": False,
            "failure_kind": None, "result_path": str(tmp_path / f"fho-{cycles}.json"),
            "solution_path": str(tmp_path / f"fho-{cycles}.npz"),
        }

    def bridge(call_args, **kwargs):
        bridge_targets.append(kwargs["target_cycles"])
        assert kwargs["source_full_solution"] == tmp_path / "fho-67.npz"
        return {
            "cycles": 68, "success": True, "infrastructure_error": False,
            "failure_kind": None, "result_path": str(tmp_path / "fho-68.json"),
            "solution_path": str(tmp_path / "fho-68.npz"),
        }

    monkeypatch.setattr(full_horizon, "_run_extension_rho", extension)
    monkeypatch.setattr(full_horizon, "_run_horizon_attempt", horizon)
    monkeypatch.setattr(full_horizon, "_run_fho_terminal_bridge", bridge)
    monkeypatch.setattr(full_horizon, "append_rho_extension_cycle", lambda *args: {})
    monkeypatch.setattr(full_horizon, "_benchmark_window_objective", lambda *args: 1.0)
    monkeypatch.setattr(full_horizon, "_certified_fho_objective", lambda *args: 10.0)
    monkeypatch.setattr(full_horizon, "_write_report", lambda *args: None)
    monkeypatch.setattr(full_horizon, "_write_markdown", lambda *args: None)
    monkeypatch.setattr(full_horizon, "run_monitored", lambda *_args, **_kwargs: pytest.fail("no numerical solve expected"))

    assert full_horizon._continue_adaptively(
        args, report=report, report_path=tmp_path / "report.json",
        markdown_path=tmp_path / "report.md", rho_seed_path=reference,
        effective_max_cycles=68, current_cycles=65, current_full_solution=source,
        rss_limit_bytes=1024,
    ) == 0
    assert horizon_targets == [68]
    assert bridge_targets == []
    assert extension_sources == [(66, source), (67, tmp_path / "rho-66.npz"), (68, tmp_path / "rho-67.npz")]
    assert report["largest_successful_cycles"] == 68
    assert report["stop_reason"] == "requested_ceiling_reached"
    assert report["solver_gap_cycles"] == []


@pytest.mark.parametrize("reference_present", [False, True])
@pytest.mark.parametrize("direct_fails", [False, True])
def test_unavailable_reference_still_solves_local_terminal_rho(tmp_path, monkeypatch, reference_present, direct_fails):
    reference = tmp_path / "rho-reference.npz"
    if reference_present:
        np.savez(reference, metadata__json=np.asarray(json.dumps({"cycles_per_window": 67})))
    source = tmp_path / "fho67.npz"
    # Two intervals per cycle, with absolute phase and all initial data copied.
    np.savez(source, states__theta=np.arange(135.).reshape(1, -1),
             states__A_Biceps=np.linspace(1, .7, 135).reshape(1, -1),
             controls__pulse=np.ones((1, 134)),
             metadata__json=np.asarray(json.dumps({"cycles_per_window": 67, "mechanical_formulation": "reduced"})))
    calls = []
    def command(args, result, solution, **kwargs):
        calls.append(kwargs["common_initial_solution"])
        full_horizon.shutil.copyfile(kwargs["common_initial_solution"], solution)
        result.write_text(json.dumps({"results": [{}]}))
        return []
    monkeypatch.setattr(full_horizon, "_rho_command", command)
    monkeypatch.setattr(full_horizon, "_rho_extension_success", lambda *args: not (direct_fails and len(calls) == 1))
    monkeypatch.setattr(full_horizon, "run_monitored", lambda *_args, **kwargs: SimpleNamespace(
        log_path=str(kwargs["log_path"]), return_code=0, memory_limit_exceeded=False,
        timed_out=False, elapsed_s=1, peak_rss_bytes=10))

    result = full_horizon._run_extension_rho(
        SimpleNamespace(output_dir=tmp_path, workspace=tmp_path, poll_interval_s=.1, attempt_timeout_s=None), source_full_solution=source,
        reference_rho_seed=reference, after_cycles=67, rss_limit_bytes=1024,
    )

    assert result["success"] is True
    assert result["failure_kind"] is None
    assert result["target_cycle"] == 68
    assert result["infrastructure_error"] is False
    assert result["physiological_limit_certified"] is False
    assert result["certificate_valid"] is True
    assert len(calls) == (5 if direct_fails else 1)
    assert result["homotopy_stages"][0]["recipe"] == "terminal_projection"
    assert result["homotopy_stages"][0]["terminal_handoff_max_abs_error"] == 0
    assert result["provenance"]["initial_state_origin"] == "FHO_67:terminal"
    if direct_fails:
        assert [stage["fraction"] for stage in result["homotopy_stages"]] == [1, .25, .5, .75, 1]
        assert result["homotopy_stages"][-1]["terminal_handoff_max_abs_error"] == 0


def test_terminal_handoff_checks_fatigue_and_stimulation_memory(tmp_path):
    source, extension = tmp_path / "source.npz", tmp_path / "next.npz"
    np.savez(source, states__theta=[[0., 1.]], states__A_Biceps=[[.8, .7]], states__last_pulse_width_Biceps=[[.001, .002]])
    np.savez(extension, states__theta=[[1., 2.]], states__A_Biceps=[[.7, .6]], states__last_pulse_width_Biceps=[[.003, .004]])
    assert full_horizon._terminal_handoff_error(source, extension) == pytest.approx(.001)
    np.savez(extension, states__theta=[[1., 2.]], states__A_Biceps=[[float("nan"), .6]], states__last_pulse_width_Biceps=[[.002, .004]])
    assert np.isinf(full_horizon._terminal_handoff_error(source, extension))


@pytest.mark.parametrize("problem", ["missing_file", "missing_cycles", "zero_cycles", "invalid_layout"])
def test_extension_does_not_hide_invalid_certified_carrier(tmp_path, problem):
    reference = tmp_path / "fho.npz"
    if problem != "missing_file":
        metadata = {} if problem == "missing_cycles" else {"cycles_per_window": 0 if problem == "zero_cycles" else 3}
        np.savez(reference, metadata__json=np.asarray(json.dumps(metadata)), states__theta=np.zeros((1, 11)))
    expected = {"missing_file": FileNotFoundError, "missing_cycles": KeyError}.get(problem, ValueError)

    with pytest.raises(expected) as error:
        full_horizon._run_extension_rho(
            SimpleNamespace(output_dir=tmp_path), source_full_solution=reference,
            reference_rho_seed=tmp_path / "optional-reference.npz", after_cycles=2, rss_limit_bytes=1024,
        )
    assert not isinstance(error.value, full_horizon.RhoReferenceCycleUnavailable)


def test_refinement_fills_only_the_last_coarse_interval():
    assert full_horizon.refinement_targets(60, 70) == list(range(61, 70))
    assert full_horizon.refinement_targets(12, 13) == []


def test_automatic_rss_limits_match_the_two_ci_machine_classes():
    assert full_horizon.automatic_rss_limit_gib(16 * full_horizon.GIB) == 12.5
    assert full_horizon.automatic_rss_limit_gib(128 * full_horizon.GIB) == 97.5


def test_rho_seed_prefix_preserves_complete_cycle_layout_and_metadata(
    tmp_path,
):
    source = tmp_path / "rho.npz"
    output = tmp_path / "prefix.npz"
    metadata = {
        "schema": "cocofest-common-periodic-initial-solution-v2",
        "cycles_per_window": 4,
        "mechanical_formulation": "reduced",
    }
    np.savez(
        source,
        states__theta=np.arange(17, dtype=float).reshape(1, 17),
        states__A_Biceps=np.arange(34, dtype=float).reshape(2, 17),
        controls__last_pulse_width_Biceps=np.arange(16, dtype=float).reshape(1, 16),
        metadata__json=np.asarray(json.dumps(metadata)),
    )

    written_metadata = full_horizon.write_rho_seed_prefix(source, output, 3)

    with np.load(output, allow_pickle=False) as data:
        assert data["states__theta"].shape == (1, 13)
        assert data["states__A_Biceps"].shape == (2, 13)
        assert data["controls__last_pulse_width_Biceps"].shape == (1, 12)
        persisted_metadata = json.loads(str(data["metadata__json"].item()))

    assert written_metadata == persisted_metadata
    assert persisted_metadata["cycles_per_window"] == 3
    assert persisted_metadata["producer_source_cycles"] == 4
    assert persisted_metadata["producer_mode"] == "receding_horizon_prefix"


def test_rho_seed_prefix_rejects_non_integral_cycle_layout(tmp_path):
    source = tmp_path / "rho.npz"
    np.savez(
        source,
        states__theta=np.zeros((1, 10)),
        controls__last_pulse_width_Biceps=np.zeros((1, 8)),
        metadata__json=np.asarray(json.dumps({"cycles_per_window": 4})),
    )

    with pytest.raises(ValueError, match="State seed"):
        full_horizon.write_rho_seed_prefix(source, tmp_path / "prefix.npz", 2)


def test_rho_seed_cycle_extracts_one_based_cycle(tmp_path):
    source = tmp_path / "rho.npz"
    output = tmp_path / "cycle-2.npz"
    np.savez(
        source,
        states__theta=np.arange(13, dtype=float).reshape(1, 13),
        controls__pulse=np.arange(12, dtype=float).reshape(1, 12),
        metadata__json=np.asarray(
            json.dumps({"cycles_per_window": 3, "mechanical_formulation": "reduced"})
        ),
    )

    metadata = full_horizon.write_rho_seed_cycle(source, output, 2)

    with np.load(output, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["states__theta"], [[4, 5, 6, 7, 8]])
        np.testing.assert_array_equal(data["controls__pulse"], [[4, 5, 6, 7]])
    assert metadata["cycles_per_window"] == 1
    assert metadata["producer_cycle_number"] == 2


def test_rho_initial_state_homotopy_hits_requested_fraction(tmp_path):
    source = tmp_path / "source.npz"
    reference = tmp_path / "reference.npz"
    target = tmp_path / "fho-2.npz"
    output = tmp_path / "stage.npz"
    one_cycle_metadata = {
        "cycles_per_window": 1,
        "mechanical_formulation": "reduced",
    }
    np.savez(
        source,
        states__theta=np.asarray([[0.0, -1.0, -2.0]]),
        states__F_Biceps=np.asarray([[10.0, 9.0, 8.0]]),
        controls__pulse=np.asarray([[0.2, 0.3]]),
        metadata__json=np.asarray(json.dumps(one_cycle_metadata)),
    )
    np.savez(
        reference,
        states__theta=np.asarray([[0.0, -1.0, -2.0]]),
        states__F_Biceps=np.asarray([[10.0, 9.0, 8.0]]),
        controls__pulse=np.asarray([[0.2, 0.3]]),
        metadata__json=np.asarray(
            json.dumps({**one_cycle_metadata, "producer_cycle_number": 3})
        ),
    )
    np.savez(
        target,
        states__theta=np.asarray([[0.0, -2.0, -4.0, -6.0, -8.0]]),
        states__F_Biceps=np.asarray([[10.0, 20.0, 30.0, 40.0, 50.0]]),
        controls__pulse=np.asarray([[0.2, 0.3, 0.4, 0.5]]),
        metadata__json=np.asarray(
            json.dumps({"cycles_per_window": 2, "mechanical_formulation": "reduced"})
        ),
    )

    metadata = full_horizon.write_rho_initial_state_homotopy_seed(
        source, reference, target, output, 0.5
    )

    with np.load(output, allow_pickle=False) as data:
        np.testing.assert_allclose(data["states__theta"], [[-4.0, -5.0, -6.0]])
        assert data["states__F_Biceps"][0, 0] == pytest.approx(30.0)
        assert data["states__F_Biceps"][0, -1] == pytest.approx(8.0)
        np.testing.assert_array_equal(data["controls__pulse"], [[0.2, 0.3]])
    assert metadata["homotopy_fraction"] == 0.5
    assert metadata["homotopy_reference_cycle_number"] == 3


def test_rho_initial_state_homotopy_aligns_equivalent_theta_winding(tmp_path):
    source = tmp_path / "source.npz"
    reference = tmp_path / "reference.npz"
    target = tmp_path / "fho-3.npz"
    output = tmp_path / "stage.npz"
    metadata = {"cycles_per_window": 1, "mechanical_formulation": "reduced"}
    reference_theta = np.asarray([[-8.0 * np.pi, -9.0 * np.pi, -10.0 * np.pi]])
    np.savez(
        source,
        states__theta=reference_theta,
        metadata__json=np.asarray(json.dumps(metadata)),
    )
    np.savez(
        reference,
        states__theta=reference_theta,
        metadata__json=np.asarray(json.dumps(metadata)),
    )
    np.savez(
        target,
        states__theta=np.asarray([[0.0, -2.0 * np.pi, -6.0 * np.pi]]),
        metadata__json=np.asarray(
            json.dumps({"cycles_per_window": 3, "mechanical_formulation": "reduced"})
        ),
    )

    written = full_horizon.write_rho_initial_state_homotopy_seed(
        source, reference, target, output, 0.25
    )

    with np.load(output, allow_pickle=False) as data:
        theta = data["states__theta"]
    np.testing.assert_allclose(theta[:, 0], [-6.0 * np.pi])
    np.testing.assert_allclose(np.diff(theta), np.diff(reference_theta))
    assert written["homotopy_theta_winding_shift"] == pytest.approx(2.0 * np.pi)


def test_fho_terminal_continuation_starts_exactly_at_fho_terminal_state(tmp_path):
    source = tmp_path / "fho-2.npz"
    output = tmp_path / "next-cycle.npz"
    metadata = {
        "cycles_per_window": 2,
        "mechanical_formulation": "reduced",
    }
    theta = np.asarray(
        [[0.0, -2.0 * np.pi / 3.0, -4.0 * np.pi / 3.0, -2.0 * np.pi,
          -8.0 * np.pi / 3.0, -10.0 * np.pi / 3.0, -4.0 * np.pi]]
    )
    fatigue = np.asarray([[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]])
    controls = np.arange(6, dtype=float).reshape(1, 6)
    np.savez(
        source,
        states__theta=theta,
        states__A_Biceps=fatigue,
        controls__last_pulse_width_Biceps=controls,
        metadata__json=np.asarray(json.dumps(metadata)),
    )

    written = full_horizon.write_fho_terminal_continuation_seed(source, output)

    with np.load(output, allow_pickle=False) as data:
        next_theta = data["states__theta"]
        next_fatigue = data["states__A_Biceps"]
        next_controls = data["controls__last_pulse_width_Biceps"]
        persisted = json.loads(str(data["metadata__json"].item()))

    np.testing.assert_allclose(next_theta[:, 0], theta[:, -1])
    np.testing.assert_allclose(next_fatigue[:, 0], fatigue[:, -1])
    assert next_theta[-1, -1] == pytest.approx(-6.0 * np.pi)
    assert next_fatigue[0, -1] == pytest.approx(0.1)
    np.testing.assert_array_equal(next_controls, controls[:, -3:])
    assert written == persisted
    assert persisted["cycles_per_window"] == 1
    assert persisted["producer_source_cycles"] == 2


def test_fho_terminal_continuation_projects_terminal_phase_to_one_winding(tmp_path):
    source = tmp_path / "fho-phase-1.npz"
    output = tmp_path / "next-phase-cycle.npz"
    slack = 0.002
    theta = np.asarray([[0.0, -2.0, -4.0, -2.0 * np.pi + slack]])
    np.savez(
        source,
        states__theta=theta,
        metadata__json=np.asarray(
            json.dumps({"cycles_per_window": 1, "mechanical_formulation": "reduced"})
        ),
    )

    metadata = full_horizon.write_fho_terminal_continuation_seed(source, output)

    with np.load(output, allow_pickle=False) as data:
        continuation = data["states__theta"]
    np.testing.assert_allclose(continuation[:, 0], theta[:, -1])
    np.testing.assert_allclose(
        continuation[:, -1], theta[:, -1] - 2.0 * np.pi
    )
    assert metadata["producer_terminal_theta_phase_projection_rad"] == pytest.approx(
        slack
    )


def test_fho_terminal_continuation_fades_fast_force_boundary_correction(tmp_path):
    """Fast force must not receive an unbounded whole-cycle translation."""

    source = tmp_path / "fho-force-2.npz"
    output = tmp_path / "next-force-cycle.npz"
    metadata = {"cycles_per_window": 2, "mechanical_formulation": "reduced"}
    # Last cycle begins at 32 N and ends at zero.  The previous construction
    # translated every node by -32 N and consequently finished at -32 N.
    force = np.asarray([[10.0, 8.0, 32.0, 36.0, 20.0, 4.0, 0.0]])
    np.savez(
        source,
        states__F_Delt_post=force,
        metadata__json=np.asarray(json.dumps(metadata)),
    )

    full_horizon.write_fho_terminal_continuation_seed(source, output)

    with np.load(output, allow_pickle=False) as data:
        continuation = data["states__F_Delt_post"]

    # The exact FHO terminal force remains the RHO initial state.  The
    # correction is local and therefore cannot manufacture the previous
    # whole-cycle negative terminal force.
    assert continuation[0, 0] == pytest.approx(0.0)
    assert continuation[0, -1] == pytest.approx(0.0)
    assert float(np.min(continuation)) >= 0.0


def test_rho_extension_replaces_the_carrier_boundary_before_appending(tmp_path):
    prefix = tmp_path / "rho-prefix.npz"
    extension = tmp_path / "rho-extension.npz"
    output = tmp_path / "fho-plus-rho.npz"
    prefix_metadata = {
        "cycles_per_window": 2,
        "mechanical_formulation": "reduced",
    }
    extension_metadata = {
        "cycles_per_window": 1,
        "mechanical_formulation": "reduced",
    }
    np.savez(
        prefix,
        states__theta=np.arange(7, dtype=float).reshape(1, 7),
        controls__pulse=np.arange(6, dtype=float).reshape(1, 6),
        metadata__json=np.asarray(json.dumps(prefix_metadata)),
    )
    np.savez(
        extension,
        states__theta=np.asarray([[20.0, 21.0, 22.0, 23.0]]),
        controls__pulse=np.asarray([[30.0, 31.0, 32.0]]),
        metadata__json=np.asarray(json.dumps(extension_metadata)),
    )

    written = full_horizon.append_rho_extension_cycle(prefix, extension, output)

    with np.load(output, allow_pickle=False) as data:
        theta = data["states__theta"]
        pulse = data["controls__pulse"]
        persisted = json.loads(str(data["metadata__json"].item()))

    assert theta.shape == (1, 10)
    assert pulse.shape == (1, 9)
    np.testing.assert_array_equal(theta[0, 6:], [20.0, 21.0, 22.0, 23.0])
    np.testing.assert_array_equal(pulse[0, -3:], [30.0, 31.0, 32.0])
    assert written == persisted
    assert persisted["cycles_per_window"] == 3
    assert persisted["replaced_reduced_boundary_maximum_absolute_change"] == 14.0


def test_benchmark_success_requires_the_complete_physical_horizon(tmp_path):
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "success": True,
                        "solver_success": True,
                        "physical_success": True,
                        "solver": "madnlp",
                        "mode": "single_shot",
                        "covered_cycles": 30,
                        "physically_validated_cycles": 30,
                    }
                ],
                "configurations": {
                    "madnlp": {
                        "single_shot": True,
                        "mechanical_formulation": "full",
                        "cycles_per_window": 30,
                        "n_windows": 30,
                        "use_sx": False,
                        "madnlp_linear_solver": "mumps",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert full_horizon._benchmark_success(
        result_path,
        expected_mode="single_shot",
        expected_cycles=30,
        expected_solver="madnlp",
    )
    assert not full_horizon._benchmark_success(
        result_path,
        expected_mode="single_shot",
        expected_cycles=100,
        expected_solver="madnlp",
    )
    assert not full_horizon._benchmark_success(
        result_path,
        expected_mode="rho",
        expected_cycles=30,
        expected_solver="madnlp",
    )


def test_validated_cycles_retains_a_shorter_rho_prefix(tmp_path):
    result_path = tmp_path / "rho-result.json"
    result_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "solver": "ipopt",
                        "mode": "rho",
                        "success": False,
                        "covered_cycles": 43,
                        "physically_validated_cycles": 42,
                    }
                ],
                "configurations": {
                    "ipopt": {
                        "single_shot": False,
                        "mechanical_formulation": "reduced",
                        "cycles_per_window": 1,
                        "n_windows": 100,
                        "use_sx": True,
                        "ipopt_linear_solver": "ma57",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert (
        full_horizon._benchmark_validated_cycles(
            result_path,
            expected_mode="rho",
            expected_solver="ipopt",
            expected_requested_cycles=100,
        )
        == 42
    )


def test_rho_extension_accepts_a_valid_cycle_rejected_by_endurance_semantics(
    tmp_path,
):
    result_path = tmp_path / "extension.json"
    result_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "success": False,
                        "solver_success": True,
                        "physical_success": False,
                        "solver": "ipopt",
                        "mode": "rho",
                        "covered_cycles": 1,
                        "physically_validated_cycles": 1,
                        "windows": [{"validated": True}],
                        "fatigue_endurance_outcome": {
                            "accepted": False,
                            "evidence": ["ding_force_capacity_decreased"],
                        },
                    }
                ],
                "configurations": {
                    "ipopt": {
                        "single_shot": False,
                        "mechanical_formulation": "reduced",
                        "cycles_per_window": 1,
                        "n_windows": 1,
                        "use_sx": True,
                        "ipopt_linear_solver": "ma57",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert full_horizon._rho_extension_success(result_path)


def test_unknown_mumps_warning_accepts_monitored_string_path(tmp_path):
    log_path = tmp_path / "solver.log"
    log_path.write_text(
        "libMAD WARNING: option linear_solver is of unknown type mumps, ignoring\n",
        encoding="utf-8",
    )

    assert full_horizon._log_has_unknown_mumps_warning(str(log_path))


def test_solver_chances_keep_independent_logs_and_results(tmp_path, monkeypatch):
    observed = {}
    args = SimpleNamespace(
        output_dir=tmp_path / "output",
        workspace=tmp_path,
        poll_interval_s=0.5,
        attempt_timeout_s=30.0,
    )
    monkeypatch.setattr(
        full_horizon,
        "write_rho_seed_prefix",
        lambda source, destination, cycles: destination.parent.mkdir(
            parents=True, exist_ok=True
        ),
    )
    monkeypatch.setattr(
        full_horizon,
        "_full_horizon_command",
        lambda *command_args, **command_kwargs: ["solver"],
    )

    def fake_run_monitored(command, **kwargs):
        observed["log_path"] = kwargs["log_path"]
        return full_horizon.MonitoredRun(
            command=command,
            return_code=1,
            peak_rss_bytes=0,
            elapsed_s=1.0,
            memory_limit_exceeded=False,
            timed_out=False,
            log_path=str(kwargs["log_path"]),
        )

    monkeypatch.setattr(full_horizon, "run_monitored", fake_run_monitored)

    attempt = full_horizon._run_horizon_attempt(
        args,
        rho_seed=tmp_path / "rho.npz",
        cycles=2,
        phase="coarse",
        chance=2,
        rss_limit_bytes=1024,
    )

    expected_dir = tmp_path / "output" / "full-horizon-0002" / "chance-2"
    assert observed["log_path"] == expected_dir / "solver.log"
    assert attempt["result_path"] == str(expected_dir / "result.json")
    assert attempt["seed_origin"] == "rho_prefix"


def test_horizon_attempt_passes_the_certified_full_prefix(tmp_path, monkeypatch):
    observed = {}
    args = SimpleNamespace(
        output_dir=tmp_path / "output",
        workspace=tmp_path,
        poll_interval_s=0.5,
        attempt_timeout_s=30.0,
    )
    prefix = tmp_path / "certified-full.npz"
    monkeypatch.setattr(
        full_horizon,
        "write_rho_seed_prefix",
        lambda source, destination, cycles: destination.parent.mkdir(
            parents=True, exist_ok=True
        ),
    )

    def fake_command(*command_args, **command_kwargs):
        observed["prefix"] = command_kwargs["prefix_solution_path"]
        return ["solver"]

    monkeypatch.setattr(full_horizon, "_full_horizon_command", fake_command)
    monkeypatch.setattr(
        full_horizon,
        "run_monitored",
        lambda command, **kwargs: full_horizon.MonitoredRun(
            command=command,
            return_code=1,
            peak_rss_bytes=0,
            elapsed_s=1.0,
            memory_limit_exceeded=False,
            timed_out=False,
            log_path=str(kwargs["log_path"]),
        ),
    )

    attempt = full_horizon._run_horizon_attempt(
        args,
        rho_seed=tmp_path / "rho.npz",
        cycles=2,
        phase="coarse",
        chance=1,
        rss_limit_bytes=1024,
        prefix_solution_path=prefix,
    )

    assert observed["prefix"] == prefix
    assert attempt["seed_origin"] == "rho_plus_certified_fho_prefix"
    assert attempt["prefix_solution_path"] == str(prefix)


def test_workflow_has_an_isolated_mx_mumps_full_horizon_mode():
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "cycling_solver_benchmark_linux.yml"
    ).read_text(encoding="utf-8")
    full_job = workflow.split("\n  full-horizon:", maxsplit=1)[1].split(
        "\n  screen-report:", maxsplit=1
    )[0]

    assert "inputs.cycles == 'full_horizon'" in full_job
    assert "--single-shot" in SCRIPT_PATH.read_text(encoding="utf-8")
    assert '"--madnlp-linear-solver"' in SCRIPT_PATH.read_text(encoding="utf-8")
    assert '"mumps"' in SCRIPT_PATH.read_text(encoding="utf-8")
    assert '"--ipopt-no-use-sx"' in SCRIPT_PATH.read_text(encoding="utf-8")
    assert '"--optional-nlp-periodic-ipopt-hot-start"' in SCRIPT_PATH.read_text(
        encoding="utf-8"
    )
    assert "--memory-limit-gib" in full_job
    assert "--continuation-step-cycles 3" in full_job
    assert "--jump-objective-relative-tolerance 0.005" in full_job
    assert "full_horizon_max_cycles" in workflow
    assert "cycling-full-horizon-${{ github.run_id }}" in full_job


def test_rho_and_full_horizon_use_the_intended_solver_contract(tmp_path):
    args = SimpleNamespace(
        python="python",
        workspace=tmp_path,
        seed_dir=tmp_path / "seed",
        output_dir=tmp_path,
        n_threads=4,
        crank_assistance=0.0,
        max_iterations=2000,
        terminal_wheel_q_slack=0.002,
        max_cycles=100,
    )

    rho = full_horizon._rho_command(args, tmp_path / "rho.json", tmp_path / "rho.npz")
    full = full_horizon._full_horizon_command(
        args,
        60,
        tmp_path / "prefix.npz",
        tmp_path / "full.json",
        tmp_path / "full.npz",
        prefix_solution_path=tmp_path / "previous-full.npz",
    )
    one_cycle_full = full_horizon._full_horizon_command(
        args,
        1,
        tmp_path / "one-cycle-prefix.npz",
        tmp_path / "one-cycle-full.json",
        tmp_path / "one-cycle-full.npz",
    )
    paired_reduced = full_horizon._full_horizon_command(
        args,
        2,
        tmp_path / "two-cycle-prefix.npz",
        tmp_path / "two-cycle-reduced.json",
        tmp_path / "two-cycle-reduced.npz",
        mechanical_formulation="reduced",
    )

    assert rho[rho.index("--solvers") + 1] == "ipopt"
    assert full[full.index("--solvers") + 1] == "ipopt"
    assert "--single-shot" not in rho
    assert "--allow-partial-receding-horizon-solution-output" in rho
    assert "--retry-failed-rho-without-advance" in rho
    assert "--ipopt-use-sx" in rho
    assert "--ipopt-c-compile" in rho
    assert "--ipopt-enable-periodic-fes-warmup-projection" in rho
    assert rho[rho.index("--periodic-fes-warmup-projection-strategy") + 1] == "rollout"
    assert "--common-initial-solution-recenter-first-node-bounds" in rho
    assert "--ipopt-no-use-sx" not in rho
    assert rho[rho.index("--ipopt-max-iter") + 1] == "2000"
    one_cycle_rho = full_horizon._rho_command(
        args,
        tmp_path / "one-cycle-rho.json",
        tmp_path / "one-cycle-rho.npz",
        n_windows=1,
    )
    # A one-cycle transition is a separate child process.  It compiles SX
    # once into a campaign-level cache so later transitions can reuse the
    # native evaluator while changing only their numerical warm start.
    assert "--ipopt-c-compile" in one_cycle_rho
    assert one_cycle_rho[
        one_cycle_rho.index("--ipopt-c-cache-dir") + 1
    ] == str(tmp_path / "rho-transition-ipopt-c-cache")
    assert one_cycle_rho[
        one_cycle_rho.index("--ipopt-c-cache-name") + 1
    ] == "rho-transition-sx"
    changed_state_rho = full_horizon._rho_command(
        args,
        tmp_path / "changed-state-rho.json",
        tmp_path / "changed-state-rho.npz",
        n_windows=1,
        common_initial_solution=tmp_path / "different-initial-state.npz",
    )
    assert changed_state_rho[
        changed_state_rho.index("--ipopt-c-cache-dir") + 1
    ] == one_cycle_rho[one_cycle_rho.index("--ipopt-c-cache-dir") + 1]
    assert changed_state_rho[
        changed_state_rho.index("--common-initial-solution") + 1
    ] == str(tmp_path / "different-initial-state.npz")
    assert "--ipopt-disable-standard-warmup" in changed_state_rho
    assert "--standard-warmup-seed" not in changed_state_rho
    assert "--single-shot" in full
    assert full[full.index("--mechanical-formulation") + 1] == "reduced"
    assert full[full.index("--full-horizon-prefix-solution") + 1] == str(
        tmp_path / "previous-full.npz"
    )
    assert (
        paired_reduced[paired_reduced.index("--mechanical-formulation") + 1]
        == "reduced"
    )
    assert "--ipopt-no-use-sx" in paired_reduced
    assert full[full.index("--ipopt-linear-solver") + 1] == "ma57"
    assert "--ipopt-no-use-sx" in full
    assert "--ipopt-c-compile" not in full
    assert "--ipopt-disable-standard-warmup" in full
    assert "--adopt-common-initial-solution-warmup-cycles" in full
    assert "--ipopt-disable-standard-warmup" in paired_reduced
    assert "--adopt-common-initial-solution-warmup-cycles" in paired_reduced
    assert "--ipopt-disable-standard-warmup" in one_cycle_full
    assert "--adopt-common-initial-solution-warmup-cycles" in one_cycle_full
    assert "--optional-nlp-periodic-ipopt-hot-start" in full
    assert "--initial-guess-diagnostics" in full
    assert "--exact-initial-nlp-audit" not in full
    assert "--acados-diagnostics" not in full
    assert "--periodic-ipopt-refinement-use-sx" in full
    assert full[full.index("--periodic-ipopt-refinement-iterations") + 1] == "2000"


def test_full_horizon_can_use_ipopt_locally_with_mx(tmp_path):
    args = SimpleNamespace(
        python="python",
        workspace=tmp_path,
        seed_dir=tmp_path / "seed",
        n_threads=4,
        crank_assistance=0.0,
        max_iterations=2000,
        terminal_wheel_q_slack=0.002,
        max_cycles=3,
        full_horizon_solver="ipopt",
    )

    command = full_horizon._full_horizon_command(
        args,
        3,
        tmp_path / "prefix.npz",
        tmp_path / "full.json",
        tmp_path / "full.npz",
    )

    assert command[command.index("--solvers") + 1] == "ipopt"
    assert "--single-shot" in command
    assert "--ipopt-no-use-sx" in command
    assert command[command.index("--ipopt-linear-solver") + 1] == "ma57"
    assert "--exact-initial-nlp-audit" not in command


def test_reference_rho_and_full_horizon_backends_default_to_radau5(tmp_path):
    common = dict(
        python="python",
        workspace=tmp_path,
        seed_dir=tmp_path / "seed",
        n_threads=4,
        crank_assistance=0.0,
        max_iterations=2000,
        terminal_wheel_q_slack=0.002,
        max_cycles=5,
    )
    ipopt_args = SimpleNamespace(**common, full_horizon_solver="ipopt")
    madnlp_args = SimpleNamespace(**common, full_horizon_solver="madnlp")
    commands = (
        full_horizon._rho_command(
            ipopt_args,
            tmp_path / "rho.json",
            tmp_path / "rho.npz",
        ),
        full_horizon._full_horizon_command(
            ipopt_args,
            5,
            tmp_path / "ipopt-prefix.npz",
            tmp_path / "ipopt.json",
            tmp_path / "ipopt.npz",
        ),
        full_horizon._full_horizon_command(
            madnlp_args,
            5,
            tmp_path / "madnlp-prefix.npz",
            tmp_path / "madnlp.json",
            tmp_path / "madnlp.npz",
        ),
    )

    assert commands[0][commands[0].index("--solvers") + 1] == "ipopt"
    assert commands[1][commands[1].index("--solvers") + 1] == "ipopt"
    assert commands[2][commands[2].index("--solvers") + 1] == "madnlp"
    for command in commands:
        assert command[command.index("--ipopt-collocation-degree") + 1] == "5"
        assert command[command.index("--ipopt-collocation-method") + 1] == "radau"
