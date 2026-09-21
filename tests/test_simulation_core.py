"""Configuration contracts without scientific imports or solver execution."""
from dataclasses import FrozenInstanceError, replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from cocofest.simulation import (
    CapabilityRegistry, ConfigurationError, SimulationConfig, build_launch_plan,
)

ROOT = Path(__file__).resolve().parents[1]


def test_lightweight_import_in_fresh_interpreter():
    program = """
import sys
from cocofest.simulation import SimulationConfig, build_launch_plan
assert not any(name.split('.')[0] in ('bioptim', 'casadi', 'biorbd', 'numpy', 'tkinter', 'PySide6') for name in sys.modules)
assert SimulationConfig().schema_version == 1
"""
    subprocess.run([sys.executable, "-c", program], cwd=ROOT, check=True)


def test_config_roundtrip_is_lossless_and_sequence_is_frozen():
    config = SimulationConfig(extra_arguments=["--standard-warmup-seed", "a path/$(no).npz"],
                              acados_ipopt_cycle1_seed=Path("seed.npz"))
    assert SimulationConfig.from_json(config.to_json()) == config
    assert config.extra_arguments == ("--standard-warmup-seed", "a path/$(no).npz")
    with pytest.raises(FrozenInstanceError):
        config.cycles = 3


@pytest.mark.parametrize("data", [{"unknown_option": 1}, {"schema_version": 2},
                                  {"schema_version": True}, [], None])
def test_invalid_documents_are_rejected(data):
    with pytest.raises(ValueError):
        SimulationConfig.from_dict(data)


@pytest.mark.parametrize("field,value", [
    ("solver", "unsupported"), ("mode", "unknown"), ("mechanics", "other"),
    ("formulation", "free"), ("integration", "euler"), ("ipopt_linear_solver", "bad"),
    ("madnlp_linear_solver", "bad"), ("cycles", 0), ("cycles", True), ("threads", 1.5),
    ("numeric_threads", -1), ("cycles_per_window", 0), ("stimulations_per_cycle", 3.5),
    ("collocation_degree", 0), ("acados_sim_stages", False), ("acados_sim_steps", 0),
    ("madnlp_hot_max_iterations", 0), ("madnlp_hot_max_wall_time", 0),
    ("signed_crank_torque", float("nan")), ("signed_crank_torque", "0.1"),
    ("terminal_q_slack", -1), ("isokinetic_omega", 0), ("energy_equivalent_torque", -1),
    ("load_torque_min", 0), ("load_torque_max", 0), ("pulse_width_max_step_us", -1),
    ("pulse_width_max_step_us", float("inf")), ("pulse_width_max_step_us", True),
    ("output_root", ""), ("weights_config", 42), ("acados_ipopt_cycle1_seed", "\0"),
    ("extra_arguments", "bad"), ("extra_arguments", ["\0"]), ("compile_evaluators", "false"),
    ("madnlp_recovery", 1), ("dry_run", 0), ("schema_version", 1.0),
    ("reduced_internal_crank_velocity_guard", "invalid"),
    ("acados_qp_solver", "invalid"),
])
def test_invalid_field_returns_structured_issue(field, value):
    config = replace(SimulationConfig(), **{field: value})
    issues = CapabilityRegistry.validate(config)
    assert any(issue.field == field for issue in issues)
    with pytest.raises(ConfigurationError) as caught:
        build_launch_plan(config, Path("/runtime"))
    assert caught.value.issues == issues


@pytest.mark.parametrize("option", sorted(CapabilityRegistry.MANAGED_OPTIONS))
def test_extra_arguments_cannot_override_visible_configuration(option):
    for extras in ((option, "value"), (option + "=value",)):
        config = replace(SimulationConfig(), extra_arguments=extras)
        assert any(i.field == "extra_arguments" for i in CapabilityRegistry.validate(config))


@pytest.mark.parametrize("extra", ["--solv=madnlp", "--n-wind", "--"])
def test_argparse_abbreviations_cannot_bypass_managed_options(extra):
    assert not CapabilityRegistry.supports(SimulationConfig(extra_arguments=(extra,)))


@pytest.mark.parametrize("mode", ["rho", "rho-physio", "rho-pace", "fho"])
@pytest.mark.parametrize("solver", ["ipopt", "madnlp", "fatrop", "acados"])
def test_capability_matrix_tracks_connected_adapters(mode, solver):
    config = SimulationConfig(mode=mode, solver=solver, compile_evaluators=False,
                              weights_config="weights.json", acados_ipopt_cycle1_seed="seed.npz",
                              integration="irk" if solver == "acados" else "radau")
    supported = mode == "rho" or solver == "ipopt" or (mode == "fho" and solver in {"madnlp", "fatrop"})
    assert CapabilityRegistry.supports(config) == supported


@pytest.mark.parametrize("change", [
    {"formulation": "isokinetic", "mechanics": "full"},
    {"pulse_width_max_step_us": 100, "mechanics": "full"},
    {"pulse_width_max_step_us": 100, "cycles_per_window": 2},
    {"pulse_width_max_step_us": 100, "mode": "fho"},
    {"cycles": 1, "cycles_per_window": 2},
    {"solver": "acados", "integration": "irk"},
    {"solver": "acados", "acados_ipopt_cycle1_seed": "seed.npz"},
    {"solver": "acados", "integration": "irk", "acados_ipopt_cycle1_seed": "seed.npz", "cycles_per_window": 2},
    {"energy_equivalent_torque": 11},
    {"mode": "rho-pace"},
    {"model_config": "model.json", "signed_crank_torque": -0.1},
])
def test_cross_field_invalid_combinations(change):
    assert not CapabilityRegistry.supports(replace(SimulationConfig(), **change))


def test_dynamic_free_torque_is_signed():
    assert CapabilityRegistry.supports(SimulationConfig(signed_crank_torque=-0.25))
    assert CapabilityRegistry.supports(SimulationConfig(signed_crank_torque=0))


def test_weighted_configured_rho_accepts_2000_cycles_but_not_more():
    base = SimulationConfig(mode="rho-physio", solver="ipopt", mechanics="reduced",
                            formulation="dynamic", cycles_per_window=1,
                            compile_evaluators=False, signed_crank_torque=.2,
                            weights_config="weights.json", model_config="model.json")
    assert CapabilityRegistry.supports(replace(base, cycles=2000))
    assert not CapabilityRegistry.supports(replace(base, cycles=2001))


def test_plan_does_not_touch_files_environment_or_processes(tmp_path, monkeypatch):
    before = dict(os.environ)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: pytest.fail("unexpected launch"))
    plan = build_launch_plan(SimulationConfig(output_root="out", threads=6, numeric_threads=2,
                                             integration="irk", pulse_width_max_step_us=100),
                             tmp_path / "env", tmp_path)
    assert dict(os.environ) == before
    assert not list(tmp_path.iterdir())
    assert plan.argv[6] == "irk"
    assert plan.environment_updates["BENCHMARK_THREADS"] == "6"
    assert plan.environment_updates["OPENBLAS_NUM_THREADS"] == "2"
    assert json.loads(plan.environment_updates["BENCHMARK_EXTRA_ARGUMENTS_JSON"]) == [
        "--reduced-internal-crank-velocity-guard", "auto",
        "--acados-qp-solver", "auto",
        "--pulse-width-slew-weight", "0.0",
        "--pulse-width-slew-reference-us", "100.0",
        "--pulse-width-max-step-us", "100.0"]


def test_exact_cycle1_launches_share_the_reduced_velocity_guard(tmp_path):
    producer = build_launch_plan(
        SimulationConfig(
            extra_arguments=("--common-initial-solution-output", "cycle1.npz"),
        ),
        tmp_path / "env",
        tmp_path,
    )
    producer_extras = json.loads(
        producer.environment_updates["BENCHMARK_EXTRA_ARGUMENTS_JSON"]
    )
    guard_index = producer_extras.index(
        "--reduced-internal-crank-velocity-guard"
    )
    assert producer_extras[guard_index + 1] == "on"

    consumer = build_launch_plan(
        SimulationConfig(
            solver="acados",
            integration="irk",
            acados_ipopt_cycle1_seed="cycle1.npz",
        ),
        tmp_path / "env",
        tmp_path,
    )
    guard_index = consumer.argv.index("--reduced-internal-crank-velocity-guard")
    assert consumer.argv[guard_index + 1] == "on"


def test_explicit_reduced_velocity_guard_override_is_preserved(tmp_path):
    plan = build_launch_plan(
        SimulationConfig(reduced_internal_crank_velocity_guard="off"),
        tmp_path / "env",
        tmp_path,
    )
    extras = json.loads(plan.environment_updates["BENCHMARK_EXTRA_ARGUMENTS_JSON"])
    guard_index = extras.index("--reduced-internal-crank-velocity-guard")
    assert extras[guard_index + 1] == "off"


@pytest.mark.parametrize(
    "qp_solver",
    ("auto", "PARTIAL_CONDENSING_HPIPM", "FULL_CONDENSING_HPIPM"),
)
def test_launch_plan_forwards_acados_qp_solver(tmp_path, qp_solver):
    plan = build_launch_plan(
        SimulationConfig(
            solver="acados",
            integration="irk",
            acados_ipopt_cycle1_seed="cycle1.npz",
            acados_qp_solver=qp_solver,
        ),
        tmp_path / "env",
        tmp_path,
    )

    option_index = plan.argv.index("--acados-qp-solver")
    assert plan.argv[option_index + 1] == qp_solver


@pytest.mark.parametrize("solver", ["ipopt", "madnlp"])
def test_fho_plan_has_one_window_spanning_all_cycles(tmp_path, solver):
    config = SimulationConfig(mode="fho", solver=solver, cycles=12,
                              integration="irk" if solver == "acados" else "radau",
                              acados_ipopt_cycle1_seed="seed.npz")
    plan = build_launch_plan(config, tmp_path / "env", tmp_path)
    assert plan.environment_updates["BENCHMARK_CYCLES_PER_WINDOW"] == "12"
    assert "--single-shot" in json.loads(plan.environment_updates["BENCHMARK_EXTRA_ARGUMENTS_JSON"])


def test_fho_acados_is_rejected_before_engine_launch(tmp_path):
    config = SimulationConfig(mode="fho", solver="acados", integration="irk",
                              acados_ipopt_cycle1_seed="cycle-1.npz")
    with pytest.raises(ConfigurationError, match="FHO is currently connected only"):
        build_launch_plan(config, tmp_path)


@pytest.mark.parametrize("mode", ["rho-physio", "rho-pace"])
def test_weighted_plans_use_distinct_mode_and_journal(tmp_path, mode):
    plan = build_launch_plan(SimulationConfig(mode=mode, weights_config="weights.json",
                                              compile_evaluators=False), tmp_path / "env", tmp_path)
    assert plan.argv[plan.argv.index("--mode") + 1] == mode
    assert plan.argv[plan.argv.index("--pace-config") + 1] == str(tmp_path / "weights.json")
    assert plan.argv[plan.argv.index("--pace-journal") + 1].endswith("weights.jsonl")
    assert "--ipopt-c-compile" not in plan.argv


def test_configured_fho_plan_preserves_model_adapter(tmp_path):
    plan = build_launch_plan(SimulationConfig(mode="fho", model_config="model.json"),
                             tmp_path / "env", tmp_path)
    assert plan.argv[1].endswith("scripts/run_configured_cycling_benchmark.py")
    assert plan.argv[plan.argv.index("--condition") + 1] == "fho"
    assert "--single-shot" in plan.argv


@pytest.mark.parametrize("solver", ["ipopt", "madnlp", "fatrop"])
def test_bilateral_reduced_plan_selects_each_nlp_backend(tmp_path, solver):
    plan = build_launch_plan(
        SimulationConfig(solver=solver, bilateral_reduced=True), tmp_path / "env", tmp_path,
    )
    extras = json.loads(plan.environment_updates["BENCHMARK_EXTRA_ARGUMENTS_JSON"])
    assert plan.argv[3] == solver
    assert "--bilateral-reduced" in extras


def test_bilateral_reduced_acados_plan_requires_and_forwards_seed(tmp_path):
    plan = build_launch_plan(
        SimulationConfig(solver="acados", integration="irk", bilateral_reduced=True,
                         acados_ipopt_cycle1_seed="cycle-1.npz"),
        tmp_path / "env", tmp_path,
    )
    assert plan.argv[plan.argv.index("--solvers") + 1] == "acados"
    assert "--bilateral-reduced" in plan.argv
    assert plan.argv[plan.argv.index("--common-initial-solution") + 1] == str(tmp_path / "cycle-1.npz")


@pytest.mark.parametrize("solver", ["ipopt", "madnlp", "fatrop"])
def test_configured_bilateral_plan_preserves_selected_nlp_solver(tmp_path, solver):
    plan = build_launch_plan(
        SimulationConfig(solver=solver, bilateral_reduced=True, model_config="model.json"),
        tmp_path / "env", tmp_path,
    )
    assert plan.argv[1].endswith("scripts/run_configured_cycling_benchmark.py")
    assert plan.argv[plan.argv.index("--solvers") + 1] == solver
    assert "--bilateral-reduced" in plan.argv


def test_configured_bilateral_acados_plan_uses_certified_rti_seed(tmp_path):
    plan = build_launch_plan(
        SimulationConfig(solver="acados", integration="irk", bilateral_reduced=True,
                         model_config="model.json", acados_ipopt_cycle1_seed="cycle-1.npz"),
        tmp_path / "env", tmp_path,
    )
    assert plan.argv[plan.argv.index("--solvers") + 1] == "acados"
    assert plan.argv[plan.argv.index("--acados-nlp-solver-type") + 1] == "SQP_RTI"
    assert plan.argv[plan.argv.index("--common-initial-solution") + 1] == str(tmp_path / "cycle-1.npz")


@pytest.mark.parametrize("kwargs", [
    {"mechanics": "full", "bilateral_reduced": True},
    {"formulation": "isokinetic", "bilateral_reduced": True},
    {"bilateral_reduced": 1},
])
def test_bilateral_reduced_validation_rejects_incompatible_modes(kwargs):
    issues = CapabilityRegistry.validate(SimulationConfig(**kwargs))
    assert any(issue.field == "bilateral_reduced" for issue in issues)


def test_lazy_public_exports_preserve_historical_names_and_cache(monkeypatch):
    import cocofest
    names = ("CustomObjective", "DingModelPulseWidthFrequency", "OcpFesMsk", "ReducedCyclingDynamics")
    assert all(name in cocofest.__all__ and name in dir(cocofest) for name in names)
    import importlib
    for name in names:
        module, symbol = cocofest._EXPORTS[name]
        # This test exercises real scientific imports only when runtime dependencies exist.
        pytest.importorskip("bioptim")
        expected = getattr(importlib.import_module(module, "cocofest"), symbol)
        assert getattr(cocofest, name) is expected
        assert name in vars(cocofest)
    with pytest.raises(AttributeError):
        getattr(cocofest, "nonexistent_public_symbol")
