import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cocofest.optimization.configured_cycling_model import (
    FIELD_ATTRIBUTES, FINGERPRINT_KEY, annotate_generated_seed, apply_model_config,
    configured_model_factories, require_seed_fingerprint, resolve_model_config,
)


def declared(alpha=-.24):
    return {"schema_version": 1, "case_id": "triceps-alpha-test", "provenance": "unit fixture",
            "muscles": {"Triceps": {"Fmax": 617., "a_scale": 7036.3, "alpha_a": alpha, "tau_fat": 76.2}}}


def model():
    from cocofest.models.ding2007.ding2007_with_fatigue import DingModelPulseWidthFrequencyWithFatigue
    return SimpleNamespace(muscles_dynamics_model=[
        DingModelPulseWidthFrequencyWithFatigue(muscle_name="Triceps", stim_time=[0., .1])])


def test_complete_effective_parameter_fingerprint_changes_only_with_parameters():
    nominal = resolve_model_config(declared())
    changed = resolve_model_config(declared(-.12))
    renamed = declared()
    renamed["case_id"] = "same-parameters-different-name"
    assert nominal[FINGERPRINT_KEY] != changed[FINGERPRINT_KEY]
    assert nominal[FINGERPRINT_KEY] == resolve_model_config(renamed)[FINGERPRINT_KEY]
    assert set(nominal["muscles"]["Triceps"]) == set(FIELD_ATTRIBUTES)


@pytest.mark.parametrize("key,value", [("unknown", 1.), ("alpha_a", .2), ("a_scale", 0.),
                                      ("tau_fat", float("inf")), ("Fmax", True)])
def test_unsafe_or_unknown_model_parameters_are_refused(key, value):
    config = declared()
    config["muscles"]["Triceps"][key] = value
    with pytest.raises(ValueError):
        resolve_model_config(config)


def test_all_factory_aliases_apply_real_ding_parameters_and_restore_on_failure():
    config = resolve_model_config(declared(-.12))
    factory = lambda *a, **kw: model()
    modules = (SimpleNamespace(set_fes_model=factory), SimpleNamespace(set_fes_model=factory))
    records = []
    with pytest.raises(RuntimeError, match="stop"):
        with configured_model_factories(config, records, modules=modules):
            for module in modules:
                muscle = module.set_fes_model().muscles_dynamics_model[0]
                assert muscle.alpha_a == -.12
                assert muscle.a_scale == muscle.a_rest == 7036.3
                assert muscle.fmax == 617.
            raise RuntimeError("stop")
    assert all(module.set_fes_model is factory for module in modules)
    assert len(records) == 2
    assert all(record[FINGERPRINT_KEY] == config[FINGERPRINT_KEY] for record in records)


def test_model_muscle_names_must_match_before_mutation():
    current = model()
    config = declared()
    config["muscles"]["Biceps"] = config["muscles"].pop("Triceps")
    with pytest.raises(ValueError, match="muscle names"):
        apply_model_config(current, resolve_model_config(config))
    assert current.muscles_dynamics_model[0].a_scale == 4920.


def test_real_cycling_factory_applies_triceps_variant_before_symbolic_fatigue_evaluation():
    import casadi as ca
    from examples.fes_multibody.cycling import cycling_pulse_width_mhe as mhe
    from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as periodic
    path = Path(__file__).resolve().parents[1] / "examples/msk_models/Wu/Modified_Wu_Shoulder_Model_Cycling.bioMod"
    nominal = mhe.set_fes_model(str(path), [0., .1], periodic_node_forcing=True)
    values = {m.muscle_name: {key: float(getattr(m, FIELD_ATTRIBUTES[key])) for key in
                             ("Fmax", "a_scale", "alpha_a", "tau_fat")}
              for m in nominal.muscles_dynamics_model}
    values["Triceps"]["alpha_a"] = -.12
    config = resolve_model_config({"schema_version": 1, "case_id": "real-factory-test",
                                   "provenance": "nominal factory plus Triceps alpha_a x0.5", "muscles": values})
    records = []
    with configured_model_factories(config, records):
        changed = periodic.set_fes_model(str(path), [0., .1], periodic_node_forcing=True)
    muscle = next(m for m in changed.muscles_dynamics_model if m.muscle_name == "Triceps")
    force = ca.SX.sym("force")
    derivative = ca.Function("configured_fatigue", [force], [muscle.a_dot_fun(muscle.a_scale, force)])
    assert float(derivative(10.)) == pytest.approx(-1.2)
    assert records[0]["actual_parameters"] == config["muscles"]


def test_seed_requires_exact_fingerprint_and_generated_annotation_preserves_data(tmp_path):
    path = tmp_path / "seed.npz"
    trajectory = np.array([[1., 2., 3.]])
    np.savez(path, states__A_Triceps=trajectory,
             metadata__json=np.asarray(json.dumps({"producer_mode": "test"})))
    nominal = resolve_model_config(declared())
    changed = resolve_model_config(declared(-.12))
    with pytest.raises(ValueError, match="incompatible"):
        require_seed_fingerprint(path, nominal[FINGERPRINT_KEY])
    annotate_generated_seed(path, nominal, condition="rho")
    assert require_seed_fingerprint(path, nominal[FINGERPRINT_KEY])["compatible"]
    with pytest.raises(ValueError, match="incompatible"):
        require_seed_fingerprint(path, changed[FINGERPRINT_KEY])
    with np.load(path) as seed:
        np.testing.assert_array_equal(seed["states__A_Triceps"], trajectory)
        assert json.loads(seed["metadata__json"].item())["producer_mode"] == "test"


@pytest.mark.parametrize("condition,single_shot", [("rho", False), ("fho", True)])
def test_common_runner_applies_same_model_to_rho_and_fho(tmp_path, monkeypatch, condition, single_shot):
    from scripts.run_configured_cycling_benchmark import main
    from examples.fes_multibody.cycling import cycling_fes_solver_comparison as benchmark
    from examples.fes_multibody.cycling import cycling_pulse_width_mhe as mhe
    from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as periodic
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(declared(-.12)))
    output_path = tmp_path / "result.json"
    seed_path = tmp_path / "generated.npz"
    monkeypatch.setattr(mhe, "set_fes_model", lambda *a, **kw: model())
    monkeypatch.setattr(periodic, "set_fes_model", lambda *a, **kw: model())

    def fake_benchmark(**kwargs):
        assert kwargs["single_shot"] is single_shot
        assert kwargs["ipopt_disable_historical_initial_guess"] is True
        for module in (mhe, periodic):
            assert module.set_fes_model().muscles_dynamics_model[0].alpha_a == -.12
        output_path.write_text("{}")
        np.savez(seed_path, states__A_Triceps=np.array([[1.]]))

    monkeypatch.setattr(benchmark, "main", fake_benchmark)
    argv = ["--model-config", str(model_path), "--condition", condition, "--", "--solvers", "ipopt",
            "--formulation", "dynamic", "--mechanical-formulation", "reduced", "--signed-crank-torque", ".1",
            "--n-windows", "2", "--cycles-per-window", "2" if single_shot else "1",
            "--output-json", str(output_path), "--common-initial-solution-output", str(seed_path)]
    if single_shot:
        argv.append("--single-shot")
    main(argv)
    audit = json.loads(output_path.with_suffix(".configuration.json").read_text())
    assert audit["status"] == "completed"
    assert audit["weights_applied"] is False
    assert len(audit["model_builds"]) == 2
    require_seed_fingerprint(seed_path, audit["model"][FINGERPRINT_KEY])


def test_common_runner_rejects_unfingerprinted_seed_before_solver(tmp_path, monkeypatch):
    from scripts.run_configured_cycling_benchmark import main
    from examples.fes_multibody.cycling import cycling_fes_solver_comparison as benchmark
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(declared()))
    seed_path = tmp_path / "old.npz"
    np.savez(seed_path, states__A_Triceps=np.array([[1.]]))
    called = []
    monkeypatch.setattr(benchmark, "main", lambda **kwargs: called.append(True))
    with pytest.raises(ValueError, match="incompatible"):
        main(["--model-config", str(model_path), "--condition", "rho", "--", "--solvers", "ipopt",
              "--mechanical-formulation", "reduced", "--signed-crank-torque", ".1",
              "--output-json", str(tmp_path / "result.json"), "--common-initial-solution", str(seed_path)])
    assert not called


@pytest.mark.parametrize("condition,adaptive", [("rho-physio", False), ("rho-pace", True)])
def test_common_runner_dispatches_static_and_adaptive_with_same_model_and_weights(tmp_path, monkeypatch, condition, adaptive):
    from scripts.run_configured_cycling_benchmark import main
    from scripts import run_rho_pace_benchmark as pace
    from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as periodic
    model_path, weights_path = tmp_path / "model.json", tmp_path / "weights.json"
    model_path.write_text(json.dumps(declared(-.12)))
    weights_path.write_text(json.dumps({"initial_weights": {"Triceps": 1.},
                                        "initial_weight_basis": "explicit test fixture"}))
    result = tmp_path / "result.json"
    calls = []
    monkeypatch.setattr(periodic, "set_fes_model", lambda *a, **kw: model())
    def fake_weighted(argv, *, adaptation_enabled):
        calls.append((argv, adaptation_enabled))
        assert periodic.set_fes_model().muscles_dynamics_model[0].alpha_a == -.12
        result.write_text("{}")
    monkeypatch.setattr(pace, "main", fake_weighted)
    main(["--model-config", str(model_path), "--condition", condition, "--weights-config", str(weights_path),
          "--weights-journal", str(tmp_path / "weights.jsonl"), "--", "--solvers", "ipopt",
          "--mechanical-formulation", "reduced", "--signed-crank-torque", ".1", "--output-json", str(result)])
    assert calls[0][1] is adaptive
    assert str(weights_path) in calls[0][0]
    audit = json.loads(result.with_suffix(".configuration.json").read_text())
    assert audit["condition"] == condition
    assert audit["adaptation_enabled"] is adaptive
    assert audit["weights_applied"] is True
