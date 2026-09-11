"""Manifest and gate tests: no solver or biomechanical runtime is imported."""

import ast
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from scripts.build_resistance_comparison_campaign import (
    ROOT, INITIAL_GATES, ORDER, artifact, build_manifest, main, next_command,
)


@pytest.fixture
def inputs(tmp_path):
    seed, profile, config = (tmp_path / name for name in ("seed.npz", "profile.npz", "pace.json"))
    seed.write_bytes(b"fixture: certification is supplied separately")
    profile.write_bytes(b"fixture geometry")
    library = tmp_path / "libcoinhsl.so"
    library.write_bytes(b"fixture: not an actual HSL shared library")
    config.write_text(json.dumps({"initial_weight_basis": "uniform_start_test_control",
                                  "policy": {"max_cycles": 100}}))
    return dict(campaign_id="fixed-R-test", resistance_nm=.22, cycles=3,
                seed=seed, reduced_profile=profile, pace_config=config,
                output_directory=tmp_path / "campaign", python=sys.executable, hsl_library=library)


def initial_review(manifest, tmp_path):
    audit = tmp_path / "initial-review.json"
    audit.write_text('{"fixture_only": true}')
    return {"manifest_sha256": manifest["manifest_sha256"],
            "initial": {"gates": dict.fromkeys(INITIAL_GATES, True),
                        "evidence": [artifact(audit)]}}


def completed_review(arm, cycles, *, outcome="completed"):
    result = Path(arm["result_path"])
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text('{"fixture_only": true}')
    evidence = [artifact(result)]
    if arm["pace_journal_path"]:
        journal = Path(arm["pace_journal_path"])
        journal.write_text('{"fixture_receipt_only": true}\n')
        evidence.append(artifact(journal))
    return {"gates": dict.fromkeys(arm["required_review_gates"], True),
            "evidence": evidence, "outcome": outcome,
            "certified_executed_cycles": cycles}


def value(argv, option):
    return argv[argv.index(option) + 1]


def test_three_arms_lock_dynamics_resistance_initial_inputs_and_horizon(inputs):
    manifest = build_manifest(**inputs)
    assert tuple(arm["id"] for arm in manifest["arms"]) == ORDER
    assert not manifest["execution_started"]
    for arm in manifest["arms"]:
        argv = arm["argv"]
        assert value(argv, "--formulation") == "dynamic"
        assert value(argv, "--signed-crank-torque") == "0.22"
        assert value(argv, "--common-initial-solution") == str(inputs["seed"])
        assert value(argv, "--reduced-cycling-profile") == str(inputs["reduced_profile"])
        assert "isokinetic" not in argv and "--crank-assistance" not in argv
        assert value(argv, "--n-windows") == "3"
        assert value(argv, "--ipopt-linear-solver") == "ma57"
        assert value(argv, "--ipopt-hsl-library") == str(inputs["hsl_library"])
    assert manifest["environment"]["IPOPT_HSL_LIBRARY"] == str(inputs["hsl_library"])
    assert manifest["inputs"]["hsl_library"] == artifact(inputs["hsl_library"])
    assert [value(a["argv"], "--cycles-per-window") for a in manifest["arms"]] == ["1", "1", "3"]
    assert "--single-shot" in manifest["arms"][2]["argv"]
    assert all("--single-shot" not in a["argv"] for a in manifest["arms"][:2])
    assert "run_rho_pace_benchmark.py" in manifest["arms"][1]["argv"][1]
    assert not inputs["output_directory"].exists()


@pytest.mark.parametrize("resistance", [0., -.2, float("nan"), float("inf"), True])
def test_no_assistance_zero_or_unresolved_resistance(inputs, resistance):
    with pytest.raises(ValueError, match="resistance"):
        build_manifest(**{**inputs, "resistance_nm": resistance})


@pytest.mark.parametrize("cycles", [0, 101, 1.5, True])
def test_campaign_respects_pace_cycle_limit(inputs, cycles):
    with pytest.raises(ValueError, match="cycles"):
        build_manifest(**{**inputs, "cycles": cycles})


def test_commands_use_existing_benchmark_flags_without_importing_solver(inputs):
    source = ROOT / "examples/fes_multibody/cycling/cycling_fes_solver_comparison.py"
    tree = ast.parse(source.read_text())
    flags = {arg.value for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
             for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)}
    for arm in build_manifest(**inputs)["arms"]:
        argv = arm["argv"]
        argv = argv[argv.index("--") + 1:] if "--" in argv else argv[2:]
        assert {item for item in argv if item.startswith("--")} <= flags


def test_ma57_requires_explicit_existing_file_not_ambient_environment(inputs, monkeypatch):
    monkeypatch.setenv("IPOPT_HSL_LIBRARY", str(inputs["hsl_library"]))
    with pytest.raises(ValueError, match="explicit existing --hsl-library"):
        build_manifest(**{**inputs, "hsl_library": None})
    with pytest.raises(FileNotFoundError):
        build_manifest(**{**inputs, "hsl_library": inputs["hsl_library"].with_name("absent.so")})
    with pytest.raises(ValueError, match="Expected file"):
        build_manifest(**{**inputs, "hsl_library": inputs["hsl_library"].parent})


def test_hsl_content_change_refuses_next_arm(inputs, tmp_path):
    manifest = build_manifest(**inputs)
    reviews = initial_review(manifest, tmp_path)
    inputs["hsl_library"].write_bytes(b"different HSL build")
    with pytest.raises(ValueError, match="input or code changed"):
        next_command(manifest, reviews)


def test_mumps_has_no_hsl_dependency_or_claim(inputs):
    manifest = build_manifest(**{**inputs, "linear_solver": "mumps", "hsl_library": None})
    assert "hsl_library" not in manifest["inputs"]
    assert "IPOPT_HSL_LIBRARY" not in manifest["environment"]
    assert all("--ipopt-hsl-library" not in arm["argv"] for arm in manifest["arms"])
    with pytest.raises(ValueError, match="only valid for an MA57"):
        build_manifest(**{**inputs, "linear_solver": "mumps"})


def test_sequential_gates_require_initial_then_rho_then_pace_with_journal(inputs, tmp_path):
    manifest = build_manifest(**inputs)
    with pytest.raises(ValueError, match="exact campaign"):
        next_command(manifest, {})
    reviews = initial_review(manifest, tmp_path)
    assert next_command(manifest, reviews) == manifest["arms"][0]["argv"]
    for index, arm in enumerate(manifest["arms"]):
        reviews[arm["id"]] = completed_review(arm, 3)
        expected = manifest["arms"][index + 1]["argv"] if index < 2 else None
        assert next_command(manifest, reviews) == expected


def test_missing_physical_gate_and_missing_pace_receipt_refuse_progress(inputs, tmp_path):
    manifest = build_manifest(**inputs)
    reviews = initial_review(manifest, tmp_path)
    reviews["initial"]["gates"]["seed_physically_certified"] = False
    with pytest.raises(ValueError, match="gates"):
        next_command(manifest, reviews)
    reviews["initial"]["gates"]["seed_physically_certified"] = True
    for arm in manifest["arms"][:2]:
        reviews[arm["id"]] = completed_review(arm, 3)
    reviews["rho_pace"]["gates"]["pace_objective_update_receipts_verified"] = False
    with pytest.raises(ValueError, match="gates"):
        next_command(manifest, reviews)


def test_numeric_stop_is_unresolved_and_never_fatigue_or_full_completion(inputs, tmp_path):
    manifest = build_manifest(**inputs)
    reviews = initial_review(manifest, tmp_path)
    review = completed_review(manifest["arms"][0], 1, outcome="numerical_or_unresolved_stop")
    reviews["baseline_rho"] = review
    assert next_command(manifest, reviews) == manifest["arms"][1]["argv"]
    review["outcome"] = "fatigue"
    with pytest.raises(ValueError, match="Never infer fatigue"):
        next_command(manifest, reviews)
    review["outcome"] = "completed"
    with pytest.raises(ValueError, match="all requested cycles"):
        next_command(manifest, reviews)
    review.update(outcome="numerical_or_unresolved_stop", certified_executed_cycles=0)
    with pytest.raises(ValueError, match="nonempty certified"):
        next_command(manifest, reviews)


def test_manifest_change_input_change_and_unreviewed_existing_output_are_refused(inputs, tmp_path):
    manifest = build_manifest(**inputs)
    reviews = initial_review(manifest, tmp_path)
    changed = deepcopy(manifest)
    changed["resistance_nm"] = .23
    with pytest.raises(ValueError, match="Manifest changed"):
        next_command(changed, reviews)
    original = inputs["seed"].read_bytes()
    inputs["seed"].write_bytes(b"new state")
    with pytest.raises(ValueError, match="input or code changed"):
        next_command(manifest, reviews)
    inputs["seed"].write_bytes(original)
    completed_review(manifest["arms"][0], 3)
    with pytest.raises(ValueError, match="Existing arm artifacts"):
        next_command(manifest, reviews)
    with pytest.raises(ValueError, match="already contains results"):
        build_manifest(**inputs)


def test_cli_only_creates_manifest_and_refuses_overwrite(inputs, tmp_path, capsys):
    target = tmp_path / "manifest.json"
    argv = ["--campaign-id", "test", "--resistance-nm", ".22", "--cycles", "3",
            "--seed", str(inputs["seed"]), "--reduced-profile", str(inputs["reduced_profile"]),
            "--pace-config", str(inputs["pace_config"]),
            "--hsl-library", str(inputs["hsl_library"]),
            "--output-directory", str(inputs["output_directory"]), "--manifest", str(target)]
    main(argv)
    assert "no solver started" in capsys.readouterr().out
    assert len(json.loads(target.read_text())["arms"]) == 3
    assert not inputs["output_directory"].exists()
    with pytest.raises(FileExistsError):
        main(argv)
