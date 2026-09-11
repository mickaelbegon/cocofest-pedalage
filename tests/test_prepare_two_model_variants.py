"""Descriptor bootstrap can declare pending inputs, but never certify them."""

import json
from pathlib import Path

import pytest

from scripts import prepare_two_model_variants as prepare
from scripts import build_two_model_resistance_campaign as campaign


def test_default_descriptor_uses_actual_bundled_cases_and_absolute_paths(tmp_path):
    variants = prepare.build_variants(run_directory=tmp_path, reduced_profile=tmp_path / "profile.npz")
    for variant, suffix, alpha_a in zip(variants, ("x0p5", "x2"), (-.12, -.48), strict=True):
        model = json.loads(Path(variant["model_config"]).read_text())
        weights = json.loads(Path(variant["weights_config"]).read_text())
        assert model["muscles"]["Triceps"]["alpha_a"] == alpha_a
        assert all(value > 0 for value in weights["initial_weights"].values())
        assert variant["seed"] == str(tmp_path / suffix / "seed.npz")
        assert variant["seed_certificate"] == str(tmp_path / suffix / "seed-certificate.json")
        assert all(Path(value).is_absolute() for key, value in variant.items() if key != "model_id")


def test_creation_reports_pending_inputs_without_creating_certificates(tmp_path, capsys):
    profile = tmp_path / "profile.npz"
    profile.write_bytes(b"existing profile")
    prepare.main(["--run-directory", str(tmp_path), "--reduced-profile", str(profile)])
    descriptor = tmp_path / "variants.json"
    variants = json.loads(descriptor.read_text())
    output = capsys.readouterr().out
    assert "preparation is incomplete" in output
    for variant in variants:
        for key in ("seed", "seed_certificate"):
            assert variant[key] in output
            assert not Path(variant[key]).exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["profile.npz", "variants.json"]
    original = descriptor.read_bytes()
    with pytest.raises(SystemExit) as failure:
        prepare.main(["--run-directory", str(tmp_path), "--reduced-profile", str(profile)])
    assert failure.value.code == 2
    assert descriptor.read_bytes() == original


def test_explicit_seed_and_certificate_locations_are_preserved(tmp_path):
    overrides = {f"{field}_{suffix}": tmp_path / f"{suffix}-{field}.custom"
                 for field in ("seed", "seed_certificate") for suffix in ("x0p5", "x2")}
    variants = prepare.build_variants(run_directory=tmp_path, reduced_profile=tmp_path / "p", **overrides)
    for variant, suffix in zip(variants, ("x0p5", "x2"), strict=True):
        for field in ("seed", "seed_certificate"):
            assert variant[field] == str(overrides[f"{field}_{suffix}"])
    with pytest.raises(ValueError, match="distinct seed"):
        prepare.build_variants(run_directory=tmp_path, reduced_profile=tmp_path / "p",
                               seed_x0p5=tmp_path / "shared", seed_x2=tmp_path / "shared")


def test_missing_descriptor_gives_actionable_error_before_loading_ocp(tmp_path, capsys):
    with pytest.raises(SystemExit) as failure:
        campaign.main(["--campaign-id", "test", "--resistance-nm", ".1", "--cycles", "100",
                       "--variants", str(tmp_path / "missing.json"), "--output-directory", str(tmp_path / "out"),
                       "--manifest", str(tmp_path / "campaign.json"), "--hsl-library", str(tmp_path / "hsl.so"),
                       "--cores-per-job", "4"])
    assert failure.value.code == 2
    error = capsys.readouterr().err
    assert "Variants descriptor not found" in error
    assert "prepare_two_model_variants.py" in error
    assert "Traceback" not in error
