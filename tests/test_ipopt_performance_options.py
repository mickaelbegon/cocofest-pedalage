"""Evaluator acceleration is opt-in and leaves physical seed signatures intact."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from cocofest.optimization.solver_backends import configure_nlp_solver, SolverBackendUnavailable
from examples.fes_multibody.cycling import cycling_fes_solver_comparison as comparison
from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as periodic


OPTIONS = [
    "--ipopt-function-transform", "--ipopt-c-compile",
    "--ipopt-c-compiler-flag=-O1", "--ipopt-c-compiler-flag=-g",
    "--ipopt-c-cache-dir", "/tmp/cocofest-test-cache", "--ipopt-c-cache-name", "rho",
]
SPLIT_OPTIONS = [
    "--ipopt-function-transform",
    "--ipopt-c-compile-callback", "nlp_f",
    "--ipopt-c-compile-callback", "nlp_g",
    "--ipopt-c-compile-callback", "nlp_grad_f",
    "--ipopt-c-cache-dir", "/tmp/cocofest-test-cache",
]


@pytest.mark.parametrize("parser_factory", [comparison.build_cli, periodic.build_argument_parser])
def test_cli_defaults_and_explicit_performance_options(parser_factory):
    parser = parser_factory()
    defaults = parser.parse_args([])
    assert defaults.ipopt_function_transform is False
    assert defaults.ipopt_c_compile is False
    assert defaults.ipopt_c_compile_callbacks is None
    assert defaults.ipopt_c_compiler_flags is None
    assert defaults.ipopt_c_cache_dir is None
    assert defaults.ipopt_c_cache_name is None
    args = parser.parse_args(OPTIONS)
    assert args.ipopt_function_transform is True
    assert args.ipopt_c_compiler_flags == ["-O1", "-g"]
    assert args.ipopt_c_cache_name == "rho"
    split_args = parser.parse_args(SPLIT_OPTIONS)
    assert split_args.ipopt_c_compile_callbacks == ["nlp_f", "nlp_g", "nlp_grad_f"]
    assert split_args.ipopt_c_compile is False


@pytest.mark.parametrize("split", [False, True])
def test_periodic_runner_forwards_performance_options(monkeypatch, split):
    options = SPLIT_OPTIONS + ["--ipopt-c-compiler-flag=-O1", "--ipopt-c-compiler-flag=-g", "--ipopt-c-cache-name", "rho"] if split else OPTIONS
    args = periodic.build_argument_parser().parse_args(["--solver", "ipopt", *options])
    captured = {}

    def record(solver_name, **kwargs):
        captured.update(kwargs)
        return solver_name

    monkeypatch.setattr(periodic, "configure_nlp_solver", record)
    assert periodic.configure_cycle_nlp_solver(args) == "ipopt"
    assert captured["ipopt_function_transform"] is True
    assert captured["ipopt_c_compiler_flags"] == ["-O1", "-g"]
    assert captured["ipopt_c_cache_dir"] == "/tmp/cocofest-test-cache"
    assert captured["ipopt_c_cache_name"] == "rho"
    assert captured["ipopt_c_compile_callbacks"] == (args.ipopt_c_compile_callbacks if split else None)
    assert periodic.nlp_c_compile_enabled(args) is True


@pytest.mark.parametrize("split", [False, True])
def test_comparison_main_forwards_options_and_resolves_cache_from_invocation(monkeypatch, tmp_path, split):
    captured = {}

    def record(solver_name, args, **kwargs):
        captured[solver_name] = args
        return {}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(comparison, "_run_benchmark_case", record)
    monkeypatch.setattr(comparison, "print_solver_overview", lambda result: None)
    comparison.main(
        solvers=("ipopt", "madnlp"), n_windows=1,
        ipopt_function_transform=True, ipopt_c_compile=not split,
        ipopt_c_compile_callbacks=["nlp_f", "nlp_g"] if split else None,
        ipopt_c_compiler_flags=["-O1"], ipopt_c_cache_dir="relative-cache", ipopt_c_cache_name="rho",
    )
    args = captured["ipopt"]
    assert args.ipopt_function_transform is True
    assert args.ipopt_c_compile is (not split)
    assert args.ipopt_c_compile_callbacks == (["nlp_f", "nlp_g"] if split else None)
    assert args.ipopt_c_compiler_flags == ["-O1"]
    assert args.ipopt_c_cache_dir == str(tmp_path / "relative-cache")
    assert args.ipopt_c_cache_name == "rho"
    assert captured["madnlp"].ipopt_function_transform is False
    assert captured["madnlp"].ipopt_c_cache_dir is None
    assert captured["madnlp"].ipopt_c_compile_callbacks is None


class FakeSolver:
    def __init__(self, **kwargs):
        self.calls = []

    def __getattr__(self, name):
        if not name.startswith("set_"):
            raise AttributeError(name)

        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        return record


def _configure(**kwargs):
    return configure_nlp_solver(
        "ipopt", max_iterations=20, ipopt_linear_solver="mumps",
        solver_namespace=SimpleNamespace(IPOPT=FakeSolver), check_availability=False, **kwargs,
    )


def test_backend_passes_acceleration_to_bioptim_not_ipopt_options():
    solver = _configure(
        ipopt_c_compile=True, ipopt_function_transform=True,
        ipopt_c_compiler_flags=["-O1"], ipopt_c_cache_dir="/tmp/cache", ipopt_c_cache_name="rho",
    )
    assert ("set_function_transform", (True,), {}) in solver.calls
    assert ("set_c_compile", (True,), {
        "compiler_flags": ["-O1"], "cache_dir": "/tmp/cache", "cache_name": "rho",
    }) in solver.calls
    assert not any(call[0] == "set_option_unsafe" for call in solver.calls)


def test_backend_legacy_default_never_calls_transform():
    solver = _configure()
    assert ("set_c_compile", (False,), {}) in solver.calls
    assert not any(call[0] == "set_function_transform" for call in solver.calls)


def test_backend_split_compile_keeps_bioptim_default_flags_and_vm_for_unselected_callbacks():
    solver = _configure(ipopt_c_compile_callbacks=["nlp_f", "nlp_g", "nlp_grad_f"], ipopt_c_cache_dir="/tmp/cache")
    assert ("set_c_compile_callbacks", (True,), {
        "callbacks": ("nlp_f", "nlp_g", "nlp_grad_f"), "cache_dir": "/tmp/cache",
    }) in solver.calls
    assert not any(call[0] == "set_c_compile" for call in solver.calls)


@pytest.mark.parametrize("callbacks,options,message", [
    (["nlp_f"], {}, "requires ipopt_c_cache_dir"),
    (["nlp_f"], {"ipopt_c_cache_dir": "/tmp/cache", "ipopt_c_compile": True}, "mutually exclusive"),
    ([], {"ipopt_c_cache_dir": "/tmp/cache"}, "distinct IPOPT"),
    (["f"], {"ipopt_c_cache_dir": "/tmp/cache"}, "distinct IPOPT"),
    (["nlp_f", "nlp_f"], {"ipopt_c_cache_dir": "/tmp/cache"}, "distinct IPOPT"),
    ("nlp_f", {"ipopt_c_cache_dir": "/tmp/cache"}, "distinct IPOPT"),
])
def test_split_compile_rejects_invalid_combinations(callbacks, options, message):
    with pytest.raises(ValueError, match=message):
        _configure(ipopt_c_compile_callbacks=callbacks, **options)


def test_missing_bioptim_split_compile_support_has_actionable_error(monkeypatch):
    monkeypatch.setattr(FakeSolver, "set_c_compile_callbacks", None, raising=False)
    with pytest.raises(SolverBackendUnavailable, match="set_c_compile_callbacks"):
        _configure(ipopt_c_compile_callbacks=["nlp_f"], ipopt_c_cache_dir="/tmp/cache")


@pytest.mark.parametrize("option", [
    {"ipopt_c_compiler_flags": ["-O1"]}, {"ipopt_c_cache_dir": "/tmp/cache"},
    {"ipopt_c_cache_name": "rho"},
])
def test_compile_options_require_explicit_compilation(option):
    with pytest.raises(ValueError, match="require ipopt_c_compile"):
        _configure(**option)


def test_missing_bioptim_transform_support_has_actionable_error(monkeypatch):
    monkeypatch.setattr(FakeSolver, "set_function_transform", None, raising=False)
    with pytest.raises(SolverBackendUnavailable, match="set_function_transform and CasADi 3.8"):
        _configure(ipopt_function_transform=True)


def test_missing_bioptim_compiler_options_has_actionable_error(monkeypatch):
    monkeypatch.setattr(FakeSolver, "set_c_compile", lambda self, enabled: None, raising=False)
    with pytest.raises(SolverBackendUnavailable, match="compiler/cache options"):
        _configure(ipopt_c_compile=True, ipopt_c_compiler_flags=["-O1"])


@pytest.mark.parametrize("options", [OPTIONS, SPLIT_OPTIONS])
def test_performance_options_preserve_scientific_seed_and_codegen_signatures(options):
    args = periodic.build_argument_parser().parse_args([])
    accelerated = deepcopy(args)
    for name, value in vars(periodic.build_argument_parser().parse_args(options)).items():
        if name.startswith("ipopt_"):
            setattr(accelerated, name, value)
    for signature in (
        periodic._continuation_cache_signature, periodic._horizon_seed_cache_signature,
        periodic._codegen_signature,
    ):
        assert signature(args) == signature(accelerated)
