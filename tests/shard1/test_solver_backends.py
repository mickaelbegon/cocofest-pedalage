from pathlib import Path
from types import SimpleNamespace

import pytest

import cocofest.optimization.solver_backends as solver_backends
from cocofest.optimization.solver_backends import (
    MADNLP_QUIET_PRINT_LEVEL,
    SolverBackendUnavailable,
    configure_nlp_solver,
    effective_ipopt_options,
    file_provenance,
    ipopt_hsl_diagnostics,
    nlp_solver_availability,
    solve_ipopt_ma57_probe,
)


class _FakeSolver:
    def __init__(self, **constructor_options):
        self.constructor_options = constructor_options
        self.calls = []

    def __getattr__(self, name):
        if not name.startswith("set_"):
            raise AttributeError(name)

        def record(*args):
            self.calls.append((name, *args))

        return record


class _Factory:
    def __init__(self):
        self.instances = []

    def __call__(self, **kwargs):
        instance = _FakeSolver(**kwargs)
        self.instances.append(instance)
        return instance


class _IpoptWithHessianDefaults(_FakeSolver):
    """Model the Bioptim IPOPT attributes which unsafe assignment cannot replace."""

    def __init__(self, **constructor_options):
        super().__init__(**constructor_options)
        self._hessian_approximation = "exact"
        self._limited_memory_max_history = 50

    def set_hessian_approximation(self, value):
        self._hessian_approximation = value
        self.calls.append(("set_hessian_approximation", value))

    def set_limited_memory_max_history(self, value):
        self._limited_memory_max_history = value
        self.calls.append(("set_limited_memory_max_history", value))


class _TypedFactory:
    def __init__(self, solver_type):
        self.solver_type = solver_type
        self.instances = []

    def __call__(self, **kwargs):
        instance = self.solver_type(**kwargs)
        self.instances.append(instance)
        return instance


def _solver_namespace(*names):
    return SimpleNamespace(**{name: _Factory() for name in names})


def test_solver_availability_checks_bioptim_factory_and_casadi_plugin():
    namespace = _solver_namespace("IPOPT", "MADNLP")
    plugins = SimpleNamespace(has_nlpsol=lambda name: name == "madnlp")

    assert nlp_solver_availability(
        "madnlp", solver_namespace=namespace, casadi_module=plugins
    ) == (True, None)
    available, reason = nlp_solver_availability(
        "alpaqa", solver_namespace=namespace, casadi_module=plugins
    )
    assert available is False
    assert "Solver.ALPAQA" in reason


def test_configure_madnlp_uses_supported_primal_hot_start():
    namespace = _solver_namespace("MADNLP")

    solver = configure_nlp_solver(
        "madnlp",
        max_iterations=321,
        tolerance=2e-6,
        madnlp_linear_solver="umfpack",
        madnlp_max_wall_time=20.0,
        solver_namespace=namespace,
        check_availability=False,
    )

    assert ("set_convergence_tolerance", 2e-6) in solver.calls
    assert ("set_constraint_tolerance", 2e-6) in solver.calls
    assert ("set_maximum_iterations", 321) in solver.calls
    assert MADNLP_QUIET_PRINT_LEVEL == 6
    assert (
        "set_option_unsafe",
        MADNLP_QUIET_PRINT_LEVEL,
        "print_level",
    ) in solver.calls
    assert not any(call[0] == "set_print_level" for call in solver.calls)
    assert not any(
        call[0] == "set_option_unsafe" and call[-1] == "mu_init"
        for call in solver.calls
    )
    assert not any(call[0] == "set_warm_start_options" for call in solver.calls)
    assert ("set_option_unsafe", "umfpack", "linear_solver") in solver.calls
    assert ("set_option_unsafe", 20.0, "max_wall_time") in solver.calls
    assert ("set_c_compile", False) in solver.calls


@pytest.mark.parametrize(
    ("requested_level", "runtime_level"),
    [(0, 6), (-1, 1), (1, 1), (5, 5), (6, 6), (9, 6)],
)
def test_configure_madnlp_maps_print_level_to_runtime_enum(
    requested_level, runtime_level
):
    solver = configure_nlp_solver(
        "madnlp",
        max_iterations=10,
        print_level=requested_level,
        solver_namespace=_solver_namespace("MADNLP"),
        check_availability=False,
    )

    assert ("set_option_unsafe", runtime_level, "print_level") in solver.calls


def test_configure_madnlp_maps_pardiso_mkl_to_native_libmad_type():
    namespace = _solver_namespace("MADNLP")

    solver = configure_nlp_solver(
        "madnlp",
        max_iterations=321,
        tolerance=2e-6,
        madnlp_linear_solver="pardiso_mkl",
        solver_namespace=namespace,
        check_availability=False,
    )

    assert (
        "set_option_unsafe",
        "PardisoMKLSolver",
        "linear_solver",
    ) in solver.calls


def test_configure_madnlp_maps_mumps_to_native_libmad_type():
    solver = configure_nlp_solver(
        "madnlp",
        max_iterations=321,
        madnlp_linear_solver="mumps",
        solver_namespace=_solver_namespace("MADNLP"),
        check_availability=False,
    )

    assert ("set_option_unsafe", "MumpsSolver", "linear_solver") in solver.calls


def test_configure_madnlp_maps_ma57_to_native_libmad_type():
    solver = configure_nlp_solver(
        "madnlp",
        max_iterations=321,
        madnlp_linear_solver="ma57",
        solver_namespace=_solver_namespace("MADNLP"),
        check_availability=False,
    )

    assert ("set_option_unsafe", "Ma57Solver", "linear_solver") in solver.calls


def test_configure_fatrop_uses_time_structured_native_options():
    namespace = _solver_namespace("FATROP")

    solver = configure_nlp_solver(
        "fatrop",
        max_iterations=432,
        tolerance=3e-7,
        print_level=2,
        fatrop_c_compile=True,
        fatrop_structure_detection="auto",
        solver_namespace=namespace,
        check_availability=False,
    )

    assert solver.constructor_options == {
        "_structure_detection": "auto",
        "_c_compile": True,
    }
    assert ("set_convergence_tolerance", 3e-7) in solver.calls
    assert ("set_constraint_tolerance", 3e-7) in solver.calls
    assert ("set_maximum_iterations", 432) in solver.calls
    assert ("set_print_level", 2) in solver.calls
    assert ("set_bound_tightening_factor", 1e-8) in solver.calls
    assert ("set_c_compile", True) in solver.calls
    assert not any(call[0] == "set_warm_start_options" for call in solver.calls)


def test_configure_alpaqa_sets_both_iteration_budgets_and_lbfgs():
    namespace = _solver_namespace("ALPAQA")

    solver = configure_nlp_solver(
        "alpaqa",
        max_iterations=500,
        tolerance=1e-5,
        alpaqa_alm_max_iterations=80,
        alpaqa_lbfgs_memory=30,
        alpaqa_max_wall_time=0.75,
        alpaqa_initial_penalty=5.0,
        alpaqa_initial_tolerance=1e-3,
        alpaqa_penalty_update_factor=5.0,
        alpaqa_maximum_penalty=1e7,
        alpaqa_panoc_max_wall_time=0.25,
        alpaqa_max_no_progress=25,
        solver_namespace=namespace,
        check_availability=False,
    )

    assert ("set_maximum_iterations", 500) in solver.calls
    assert ("set_alm_maximum_iterations", 80) in solver.calls
    assert ("set_lbfgs_memory", 30) in solver.calls
    assert ("set_maximum_wall_time", 0.75) in solver.calls
    assert ("set_initial_penalty", 5.0) in solver.calls
    assert ("set_initial_tolerance", 1e-3) in solver.calls
    assert ("set_penalty_update_factor", 5.0) in solver.calls
    assert ("set_maximum_penalty", 1e7) in solver.calls
    assert ("set_option_unsafe", "0.25s", "panoc.max_time") in solver.calls
    assert ("set_option_unsafe", 25, "panoc.max_no_progress") in solver.calls


def test_configure_ipopt_retains_robust_cocofest_settings():
    namespace = _solver_namespace("IPOPT")

    solver = configure_nlp_solver(
        "ipopt",
        max_iterations=1000,
        tolerance=1e-6,
        ipopt_linear_solver="mumps",
        ipopt_hsl_library="/opt/coinhsl/libcoinhsl.dylib",
        ipopt_c_compile=True,
        ipopt_options={
            "linear_system_scaling": "none",
            "ma57_automatic_scaling": "yes",
        },
        solver_namespace=namespace,
        check_availability=False,
    )

    assert solver.constructor_options["_max_iter"] == 1000
    assert ("set_warm_start_init_point", "yes") in solver.calls
    assert ("set_mu_init", 1e-2) in solver.calls
    assert ("set_tol", 1e-6) in solver.calls
    assert ("set_dual_inf_tol", 1e-6) in solver.calls
    assert ("set_constr_viol_tol", 1e-6) in solver.calls
    assert ("set_linear_solver", "mumps") in solver.calls
    assert ("set_print_level", 0) in solver.calls
    assert (
        "set_option_unsafe",
        "/opt/coinhsl/libcoinhsl.dylib",
        "hsllib",
    ) in solver.calls
    assert ("set_option_unsafe", "none", "linear_system_scaling") in solver.calls
    assert ("set_option_unsafe", "yes", "ma57_automatic_scaling") in solver.calls
    assert ("set_c_compile", True) in solver.calls


def test_configure_ipopt_overrides_bioptim_hessian_defaults_with_setters():
    factory = _TypedFactory(_IpoptWithHessianDefaults)
    solver = configure_nlp_solver(
        "ipopt",
        max_iterations=10,
        ipopt_options={
            "hessian_approximation": "limited-memory",
            "limited_memory_max_history": 10,
        },
        solver_namespace=SimpleNamespace(IPOPT=factory),
        check_availability=False,
    )

    assert solver._hessian_approximation == "limited-memory"
    assert solver._limited_memory_max_history == 10
    assert ("set_hessian_approximation", "limited-memory") in solver.calls
    assert ("set_limited_memory_max_history", 10) in solver.calls


def test_configure_ipopt_serializes_an_hsl_path_for_casadi():
    solver = configure_nlp_solver(
        "ipopt",
        max_iterations=10,
        ipopt_hsl_library=Path("/opt/coinhsl/libhsl.so"),
        solver_namespace=_solver_namespace("IPOPT"),
        check_availability=False,
    )

    assert (
        "set_option_unsafe",
        "/opt/coinhsl/libhsl.so",
        "hsllib",
    ) in solver.calls


def test_file_provenance_records_resolved_path_size_and_sha256(tmp_path):
    seed = tmp_path / "common-seed.npz"
    seed.write_bytes(b"shared deterministic seed")

    provenance = file_provenance(seed)

    assert provenance == {
        "requested_path": str(seed),
        "resolved_path": str(seed.resolve()),
        "exists": True,
        "is_file": True,
        "sha256": "a0a48e306bc09bc7c29df91c7c8fd97fe06a06b1ac520a57bcba84e9d209e6d8",
        "size_bytes": 25,
    }


def test_actual_ma57_probe_submits_hsl_and_validates_the_solution(tmp_path, monkeypatch):
    library = tmp_path / "libhsl.so"
    library.write_bytes(b"not needed by fake CasADi")
    submitted = {}

    class Expression:
        def __getitem__(self, key):
            return self

        def __add__(self, other):
            return self

        def __radd__(self, other):
            return self

        def __sub__(self, other):
            return self

        def __pow__(self, other):
            return self

    class Dense:
        def reshape(self, *args):
            return [1.0, 2.0]

    class ProbeSolver:
        def __call__(self, **kwargs):
            return {"x": SimpleNamespace(full=lambda: Dense())}

        def stats(self):
            return {
                "success": True,
                "return_status": "Solve_Succeeded",
                "iter_count": 1,
            }

    def nlpsol(name, plugin, nlp, options):
        submitted.update(options)
        assert plugin == "ipopt"
        return ProbeSolver()

    fake_casadi = SimpleNamespace(
        __version__="test",
        __file__=None,
        MX=SimpleNamespace(sym=lambda *args: Expression()),
        nlpsol=nlpsol,
    )
    monkeypatch.setattr(
        "cocofest.optimization.solver_backends.loaded_solver_runtime_provenance",
        lambda: [file_provenance(library)],
    )

    report = solve_ipopt_ma57_probe(library, casadi_module=fake_casadi)

    assert report["success"] is True
    assert report["functional_success"] is True
    assert report["selected_hsl_mapped"] is True
    assert report["constraint_residual"] == 0.0
    assert report["solution_error_inf"] == 0.0
    assert submitted["ipopt.linear_solver"] == "ma57"
    assert submitted["ipopt.hsllib"] == str(library)


def test_runtime_abi_audit_exposes_multiple_metis_providers():
    report = solver_backends._runtime_abi_audit(
        {
            "loaded_solver_libraries": [
                {"resolved_path": "/env/lib/libcoinmetis.so.2"},
                {"resolved_path": "/env/lib/libmetis.so"},
            ],
            "hsl_library": {"elf": {"class_bits": 64, "endianness": "little"}},
            "process_runtime": {"pointer_bits": 64, "byteorder": "little"},
        }
    )

    assert report["loaded_metis_provider_names"] == [
        "libcoinmetis.so.2",
        "libmetis.so",
    ]
    assert report["multiple_metis_providers"] is True
    assert any("Multiple METIS providers" in warning for warning in report["warnings"])


def test_hsl_diagnostics_report_missing_explicit_library(tmp_path):
    missing = tmp_path / "libcoinhsl.so"

    diagnostics = ipopt_hsl_diagnostics(
        "ma57", missing, probe_loadability=True
    )

    assert diagnostics["hsl_required"] is True
    assert diagnostics["library_source"] == "argument"
    assert diagnostics["file"]["resolved_path"] == str(missing)
    assert diagnostics["file"]["exists"] is False
    assert diagnostics["load_probe_attempted"] is True
    assert diagnostics["loadable"] is False
    assert "No such file" in diagnostics["load_error"]
    assert "actual MA57 solve" in diagnostics["abi_note"]


def test_configure_ipopt_uses_environment_hsl_and_keeps_effective_options(
    tmp_path, monkeypatch
):
    library = tmp_path / "libcoinhsl.so"
    library.touch()
    monkeypatch.setenv("IPOPT_HSL_LIBRARY", str(library))

    solver = configure_nlp_solver(
        "ipopt",
        max_iterations=17,
        tolerance=2e-7,
        print_level=3,
        ipopt_linear_solver="ma57",
        ipopt_options={"ma57_pivtol": 1e-8},
        solver_namespace=_solver_namespace("IPOPT"),
        check_availability=False,
    )

    assert ("set_option_unsafe", str(library), "hsllib") in solver.calls
    diagnostics = ipopt_hsl_diagnostics("ma57")
    assert diagnostics["library_source"] == "IPOPT_HSL_LIBRARY"
    assert effective_ipopt_options(
        max_iterations=17,
        tolerance=2e-7,
        print_level=3,
        linear_solver="ma57",
        hsl_library=str(library),
        advanced_options={"ma57_pivtol": 1e-8},
    )["ma57_pivtol"] == 1e-8


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"max_iterations": 0}, "max_iterations"),
        ({"max_iterations": 1, "tolerance": 0}, "tolerance"),
        (
            {"max_iterations": 1, "alpaqa_lbfgs_memory": 0},
            "alpaqa_lbfgs_memory",
        ),
        (
            {"max_iterations": 1, "madnlp_linear_solver": "MumpsSolver"},
            "madnlp_linear_solver",
        ),
        (
            {"max_iterations": 1, "madnlp_max_wall_time": 0},
            "madnlp_max_wall_time",
        ),
        (
            {"max_iterations": 1, "fatrop_structure_detection": "none"},
            "fatrop_structure_detection",
        ),
        (
            {"max_iterations": 1, "fatrop_bound_tightening_factor": -1e-8},
            "fatrop_bound_tightening_factor",
        ),
        (
            {"max_iterations": 1, "alpaqa_penalty_update_factor": 1},
            "alpaqa_penalty_update_factor",
        ),
        (
            {"max_iterations": 1, "alpaqa_panoc_max_wall_time": 0},
            "alpaqa_panoc_max_wall_time",
        ),
    ],
)
def test_solver_configuration_rejects_invalid_common_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        configure_nlp_solver(
            "alpaqa",
            solver_namespace=_solver_namespace("ALPAQA"),
            check_availability=False,
            **kwargs,
        )


def test_fatrop_rejects_iteration_budget_above_native_limit():
    with pytest.raises(ValueError, match="bounded to 1000"):
        configure_nlp_solver(
            "fatrop",
            max_iterations=1001,
            solver_namespace=_solver_namespace("FATROP"),
            check_availability=False,
        )


def test_missing_optional_solver_has_actionable_error():
    with pytest.raises(SolverBackendUnavailable, match="Solver.MADNLP"):
        configure_nlp_solver(
            "madnlp",
            max_iterations=10,
            solver_namespace=_solver_namespace("IPOPT"),
            check_availability=False,
        )
