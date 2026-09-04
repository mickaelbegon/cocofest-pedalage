from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github" / "scripts" / "run_prospective_rho_matrix.py"
SPEC = importlib.util.spec_from_file_location("run_prospective_rho_matrix", SCRIPT)
matrix = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = matrix
SPEC.loader.exec_module(matrix)


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        python=Path("/env/bin/python"),
        output_root=tmp_path,
        n_windows=150,
        stimulations_per_cycle=30,
        signed_crank_torque=0.2,
        isokinetic_omega=-2.0 * math.pi,
        ipopt_max_iter=5000,
        hsl_library=Path("/opt/libhsl.so"),
        seed=Path("/data/common.npz"),
    )


def test_matrix_predeclares_two_policies_and_censors_rollout_arms():
    cases = matrix.build_cases(0.03, 0.005)

    assert [case.slug for case in cases] == [
        "historical",
        "terminal-reserve",
        "rollout-h5",
        "rollout-h10",
        "rollout-h20",
    ]
    assert [case.executable for case in cases] == [True, True, False, False, False]
    assert cases[1].terminal_reserve_weight == 0.03
    assert "intentionally unspecified" in matrix.ROLLOUT_ACTIVATION_REQUIREMENT


def test_paired_commands_freeze_seed_load_cadence_and_ma57(tmp_path):
    args = _args(tmp_path)
    historical, reserve, rollout, *_ = matrix.build_cases(0.03, 0.005)
    commands = [
        matrix.build_case_command(args, historical),
        matrix.build_case_command(args, reserve),
    ]

    for command in commands:
        assert command[command.index("--common-initial-solution") + 1] == "/data/common.npz"
        assert command[command.index("--signed-crank-torque") + 1] == "0.2"
        assert command[command.index("--isokinetic-omega") + 1] == str(-2.0 * math.pi)
        assert command[command.index("--stimulations-per-cycle") + 1] == "30"
        assert command[command.index("--ipopt-linear-solver") + 1] == "ma57"
        assert command[command.index("--ipopt-dual-warm-start-mode") + 1] == "off"
        assert "--ipopt-c-compile" in command
    assert commands[0][commands[0].index("--terminal-reserve-weight") + 1] == "0.0"
    assert commands[1][commands[1].index("--terminal-reserve-weight") + 1] == "0.03"
    assert matrix.build_case_command(args, rollout) is None


def test_unavailable_rollout_is_censored_not_counted_as_failure(tmp_path):
    args = _args(tmp_path)
    case = matrix.build_cases(0.03, 0.005)[2]

    row = matrix.summarize_case(args, case, "formulation_unavailable")

    assert row["censored"] is True
    assert row["complete"] is False
    assert row["censor_reason"] == "formulation_unavailable"
    assert row["rollout_horizon"] == 5


def test_dry_run_writes_immutable_protocol_and_censored_summary(
    tmp_path, monkeypatch, capsys
):
    seed = tmp_path / "seed.npz"
    hsl = tmp_path / "libhsl.so"
    seed.write_bytes(b"seed")
    hsl.write_bytes(b"hsl")
    output = tmp_path / "matrix"
    monkeypatch.setattr(
        matrix._sweep,
        "hsl_preflight",
        lambda path: {
            "static_inspection_success": True,
            "ma57_symbol": "ma57id_",
        },
    )
    probe = {
        "schema": "cocofest-ipopt-ma57-runtime-probe-v2",
        "functional_success": True,
        "success": True,
        "production_ready": True,
        "production_readiness_reasons": [],
        "process_return_code": 0,
    }
    monkeypatch.setattr(
        matrix._sweep, "run_ma57_runtime_probe", lambda *args, **kwargs: probe
    )

    code = matrix.main(
        [
            "--seed",
            str(seed),
            "--hsl-library",
            str(hsl),
            "--output-root",
            str(output),
            "--python",
            sys.executable,
            "--n-windows",
            "2",
            "--dry-run",
        ]
    )

    assert code == 0
    assert capsys.readouterr().out.count("cycling_fes_solver_comparison") == 2
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["contract"]["protocol"] == "paired-rho-only; no FHO data"
    assert manifest["contract"]["ma57_runtime_probe"] == matrix._sweep.stable_runtime_probe_identity(probe)
    summary = json.loads((output / "summary.json").read_text())
    assert [row["censor_reason"] for row in summary["rows"][2:]] == [
        "formulation_unavailable"
    ] * 3

    with pytest.raises(SystemExit, match="already exists"):
        matrix.main(
            [
                "--seed",
                str(seed),
                "--hsl-library",
                str(hsl),
                "--output-root",
                str(output),
                "--python",
                sys.executable,
                "--n-windows",
                "2",
                "--dry-run",
            ]
        )


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--n-windows", "1"], "at least 2"),
        (["--isokinetic-omega", "0"], "strictly negative"),
        (["--terminal-reserve-weight", "0"], "strictly positive"),
        (["--ma57-probe-timeout", "0"], "limits must be positive"),
    ],
)
def test_invalid_protocol_is_rejected_before_native_probe(tmp_path, extra, message):
    seed = tmp_path / "seed.npz"
    hsl = tmp_path / "libhsl.so"
    seed.write_bytes(b"seed")
    hsl.write_bytes(b"hsl")

    with pytest.raises(SystemExit, match=message):
        matrix.main(
            [
                "--seed",
                str(seed),
                "--hsl-library",
                str(hsl),
                "--output-root",
                str(tmp_path / "matrix"),
                "--python",
                sys.executable,
                *extra,
            ]
        )
