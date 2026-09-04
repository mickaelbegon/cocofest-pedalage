"""Optional nonlinear solver backends used by Cocofest benchmarks.

Fatrop, MadNLP and Alpaqa are exposed by compatible Bioptim revisions through
CasADi's ``nlpsol`` API.  They are optional on purpose: importing Cocofest must
continue to work with a standard Bioptim/CasADi installation.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import os
from pathlib import Path
from typing import Any

NLP_SOLVER_NAMES = ("ipopt", "fatrop", "madnlp", "alpaqa")
MADNLP_LINEAR_SOLVER_NAMES = (
    "mumps",
    "ma57",
    "umfpack",
    "lapack_cpu",
    "pardiso_mkl",
    "cudss",
    "lapack_gpu",
    "cucholesky",
)
MADNLP_LINEAR_SOLVER_RUNTIME_NAMES = {
    # libMad transports ``linear_solver`` as a Julia ``Type``.  Its type
    # registry is case-sensitive, so the user-facing alias must never be sent
    # literally (``mumps`` is rejected and silently falls back to the MadNLP
    # default after emitting an "unknown type" warning).
    "mumps": "MumpsSolver",
    "ma57": "Ma57Solver",
    "pardiso_mkl": "PardisoMKLSolver",
}
# MadNLP 0.9.2 defines LogLevels as TRACE=1 through ERROR=6.  The libMad
# interface transports that enum as an integer and rejects IPOPT's conventional
# quiet value 0.
MADNLP_QUIET_PRINT_LEVEL = 6


class SolverBackendUnavailable(RuntimeError):
    """Raised when an optional Bioptim solver or CasADi plugin is unavailable."""


_IPOPT_HSL_LINEAR_SOLVERS = frozenset({"ma27", "ma57", "ma77", "ma86", "ma97"})
_MA57_SYMBOL_NAMES = ("ma57id_", "ma57id", "MA57ID")


def file_provenance(path: str | os.PathLike[str] | None) -> dict[str, Any] | None:
    """Return stable, JSON-safe provenance for an input file.

    Missing inputs are described instead of raising so failed benchmark setup
    still leaves actionable evidence in ``result.json``.
    """

    if path is None:
        return None
    requested = os.fspath(path)
    resolved = Path(path).expanduser().resolve()
    provenance: dict[str, Any] = {
        "requested_path": requested,
        "resolved_path": str(resolved),
        "exists": resolved.exists(),
        "is_file": resolved.is_file(),
        "sha256": None,
        "size_bytes": None,
    }
    if not resolved.is_file():
        return provenance
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        provenance["sha256"] = digest.hexdigest()
        provenance["size_bytes"] = resolved.stat().st_size
    except OSError as error:
        provenance["read_error"] = f"{type(error).__name__}: {error}"
    return provenance


def ipopt_hsl_diagnostics(
    linear_solver: str,
    hsl_library: str | os.PathLike[str] | None = None,
    *,
    probe_loadability: bool = False,
) -> dict[str, Any]:
    """Describe how an IPOPT HSL dependency is supplied and whether it loads.

    A concrete path is never guessed from a workstation layout. Callers may
    provide one directly or through ``IPOPT_HSL_LIBRARY``. When neither is
    set, the platform dynamic-loader lookup is reported as evidence, but IPOPT
    remains free to use an HSL implementation linked into its own binary.
    """

    linear_solver = str(linear_solver).lower()
    explicit_library = hsl_library is not None
    if hsl_library is None:
        hsl_library = os.environ.get("IPOPT_HSL_LIBRARY")
    source = "argument" if explicit_library else (
        "IPOPT_HSL_LIBRARY" if hsl_library is not None else None
    )
    discovered = None
    if hsl_library is None:
        for library_name in ("coinhsl", "hsl"):
            discovered = ctypes.util.find_library(library_name)
            if discovered:
                source = f"dynamic_loader_lookup:{library_name}"
                break

    requested = None if hsl_library is None else os.fspath(hsl_library)
    load_target = requested or discovered
    concrete_path = None
    if requested is not None:
        candidate = Path(requested).expanduser()
        if candidate.is_absolute() or os.sep in requested or candidate.exists():
            concrete_path = candidate.resolve()

    diagnostics: dict[str, Any] = {
        "linear_solver": linear_solver,
        "hsl_required": linear_solver in _IPOPT_HSL_LINEAR_SOLVERS,
        "library_source": source,
        "requested_library": requested,
        "dynamic_loader_candidate": discovered,
        "load_target": load_target,
        "file": file_provenance(concrete_path),
        "load_probe_attempted": False,
        "loadable": None,
        "ma57_symbol_detected": None,
        "load_error": None,
        "abi_note": (
            "A successful dlopen/symbol probe checks immediate loader dependencies "
            "only; IPOPT, integer-width, BLAS and Fortran-runtime ABI compatibility "
            "is established only by an actual MA57 solve."
        ),
    }
    if not probe_loadability or load_target is None:
        return diagnostics

    diagnostics["load_probe_attempted"] = True
    try:
        library = ctypes.CDLL(os.fspath(load_target))
    except OSError as error:
        diagnostics["loadable"] = False
        diagnostics["ma57_symbol_detected"] = False
        diagnostics["load_error"] = f"{type(error).__name__}: {error}"
        return diagnostics
    diagnostics["loadable"] = True
    diagnostics["ma57_symbol_detected"] = any(
        hasattr(library, symbol) for symbol in _MA57_SYMBOL_NAMES
    )
    return diagnostics


def effective_ipopt_options(
    *,
    max_iterations: int,
    tolerance: float,
    print_level: int,
    linear_solver: str,
    hsl_library: str | os.PathLike[str] | None,
    advanced_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the final IPOPT option values submitted by Cocofest."""

    options: dict[str, Any] = {
        "max_iter": int(max_iterations),
        "warm_start_init_point": "yes",
        "mu_init": 1e-2,
        "tol": float(tolerance),
        "dual_inf_tol": float(tolerance),
        "constr_viol_tol": float(tolerance),
        "linear_solver": str(linear_solver),
        "print_level": int(print_level),
    }
    if hsl_library is not None:
        options["hsllib"] = os.fspath(hsl_library)
    # Unsafe options are applied last by configure_nlp_solver and therefore
    # intentionally win if an advanced caller overrides a common setting.
    options.update(advanced_options or {})
    return options


def nlp_solver_availability(
    solver_name: str,
    *,
    solver_namespace=None,
    casadi_module=None,
) -> tuple[bool, str | None]:
    """Check both the Bioptim factory and the underlying CasADi plugin."""

    solver_name = solver_name.lower()
    if solver_name not in NLP_SOLVER_NAMES:
        raise ValueError(f"Unsupported NLP solver '{solver_name}'.")

    if solver_namespace is None:
        from bioptim import Solver as solver_namespace

    factory_name = solver_name.upper()
    if not hasattr(solver_namespace, factory_name):
        return (
            False,
            f"Bioptim does not expose Solver.{factory_name}; install its "
            f"{solver_name} integration branch.",
        )

    if casadi_module is None:
        import casadi as casadi_module

    has_nlpsol = getattr(casadi_module, "has_nlpsol", None)
    if has_nlpsol is None:
        return False, "The installed CasADi does not expose has_nlpsol()."
    try:
        available = bool(has_nlpsol(solver_name))
    except RuntimeError as error:
        return False, str(error)
    if not available:
        return (
            False,
            f"CasADi nlpsol plugin '{solver_name}' is unavailable in this build.",
        )
    return True, None


def configure_nlp_solver(
    solver_name: str,
    *,
    max_iterations: int,
    tolerance: float = 1e-6,
    print_level: int = 0,
    ipopt_linear_solver: str = "ma57",
    ipopt_hsl_library: str | None = None,
    ipopt_c_compile: bool = False,
    ipopt_options: dict[str, Any] | None = None,
    fatrop_c_compile: bool = False,
    fatrop_structure_detection: str = "auto",
    fatrop_bound_tightening_factor: float = 1e-8,
    madnlp_c_compile: bool = False,
    madnlp_linear_solver: str | None = None,
    madnlp_max_wall_time: float | None = None,
    alpaqa_alm_max_iterations: int | None = None,
    alpaqa_lbfgs_memory: int = 20,
    alpaqa_max_wall_time: float | None = None,
    alpaqa_initial_penalty: float | None = None,
    alpaqa_initial_tolerance: float | None = None,
    alpaqa_penalty_update_factor: float | None = None,
    alpaqa_maximum_penalty: float | None = None,
    alpaqa_panoc_max_wall_time: float | None = None,
    alpaqa_max_no_progress: int | None = None,
    solver_namespace=None,
    check_availability: bool = True,
) -> Any:
    """Create a consistently configured Bioptim NLP solver.

    The common tolerance and iteration budget make solver comparisons
    interpretable.  Solver-specific settings only cover features that have
    been validated by the corresponding Bioptim integration branch.
    """

    solver_name = solver_name.lower()
    if max_iterations < 1:
        raise ValueError("max_iterations must be strictly positive.")
    if solver_name == "fatrop" and max_iterations > 1000:
        raise ValueError(
            "Fatrop's native max_iter option is bounded to 1000 in CasADi 3.7.2."
        )
    if tolerance <= 0:
        raise ValueError("tolerance must be strictly positive.")
    if fatrop_structure_detection not in {"auto", "manual"}:
        raise ValueError("fatrop_structure_detection must be 'auto' or 'manual'.")
    if fatrop_bound_tightening_factor < 0:
        raise ValueError("fatrop_bound_tightening_factor must be non-negative.")
    if alpaqa_lbfgs_memory < 1:
        raise ValueError("alpaqa_lbfgs_memory must be strictly positive.")
    if (
        madnlp_linear_solver is not None
        and madnlp_linear_solver not in MADNLP_LINEAR_SOLVER_NAMES
    ):
        raise ValueError(
            "madnlp_linear_solver must be one of "
            f"{', '.join(MADNLP_LINEAR_SOLVER_NAMES)}."
        )
    if madnlp_max_wall_time is not None and madnlp_max_wall_time <= 0:
        raise ValueError("madnlp_max_wall_time must be strictly positive.")
    if alpaqa_initial_tolerance is not None and alpaqa_initial_tolerance <= 0:
        raise ValueError("alpaqa_initial_tolerance must be strictly positive.")
    if (
        alpaqa_penalty_update_factor is not None
        and alpaqa_penalty_update_factor <= 1
    ):
        raise ValueError("alpaqa_penalty_update_factor must be greater than one.")
    if alpaqa_maximum_penalty is not None and alpaqa_maximum_penalty <= 0:
        raise ValueError("alpaqa_maximum_penalty must be strictly positive.")
    if (
        alpaqa_panoc_max_wall_time is not None
        and alpaqa_panoc_max_wall_time <= 0
    ):
        raise ValueError("alpaqa_panoc_max_wall_time must be strictly positive.")
    if alpaqa_max_no_progress is not None and alpaqa_max_no_progress < 1:
        raise ValueError("alpaqa_max_no_progress must be strictly positive.")

    if solver_namespace is None:
        from bioptim import Solver as solver_namespace

    if check_availability:
        available, reason = nlp_solver_availability(
            solver_name, solver_namespace=solver_namespace
        )
        if not available:
            raise SolverBackendUnavailable(reason)

    factory_name = solver_name.upper()
    factory = getattr(solver_namespace, factory_name, None)
    if factory is None:
        raise SolverBackendUnavailable(
            f"Bioptim does not expose Solver.{factory_name}."
        )

    if solver_name == "ipopt":
        configured_hsl_library = ipopt_hsl_library
        if configured_hsl_library is None:
            configured_hsl_library = os.environ.get("IPOPT_HSL_LIBRARY")
        hsl_diagnostics = ipopt_hsl_diagnostics(
            ipopt_linear_solver,
            ipopt_hsl_library,
            probe_loadability=(
                bool(check_availability)
                and str(ipopt_linear_solver).lower() in _IPOPT_HSL_LINEAR_SOLVERS
            ),
        )
        hsl_file = hsl_diagnostics.get("file") or {}
        if (
            check_availability
            and hsl_diagnostics["hsl_required"]
            and configured_hsl_library is not None
            and not hsl_file.get("exists", True)
        ):
            raise SolverBackendUnavailable(
                "The configured IPOPT HSL library does not exist: "
                f"{hsl_file.get('resolved_path')}. Set IPOPT_HSL_LIBRARY to the "
                "CoinHSL shared library built for this environment."
            )
        if (
            hsl_diagnostics["hsl_required"]
            and hsl_diagnostics.get("loadable") is False
        ):
            raise SolverBackendUnavailable(
                "The configured IPOPT HSL library could not be loaded before "
                f"constructing the NLP: {hsl_diagnostics['load_error']}. Rebuild "
                "CoinHSL against this environment's Fortran/BLAS runtime."
            )
        solver = factory(
            show_online_optim=False,
            _max_iter=max_iterations,
            show_options={"show_bounds": True},
        )
        solver.set_warm_start_init_point("yes")
        solver.set_mu_init(1e-2)
        solver.set_tol(tolerance)
        solver.set_dual_inf_tol(tolerance)
        solver.set_constr_viol_tol(tolerance)
        solver.set_linear_solver(ipopt_linear_solver)
        solver.set_print_level(print_level)
        if configured_hsl_library is not None:
            # The comparison front-end resolves invocation paths to ``Path``
            # objects. CasADi accepts only scalar option values, so pass the
            # HSL shared-library location as a string.
            solver.set_option_unsafe(str(configured_hsl_library), "hsllib")
        for name, value in (ipopt_options or {}).items():
            solver.set_option_unsafe(value, name)
        solver.set_c_compile(ipopt_c_compile)
        # Do not attach diagnostic attributes to Bioptim's solver wrapper.
        # Its attribute forwarding treats unknown attributes as IPOPT options,
        # so Python-only diagnostic names are later submitted to CasADi and
        # make IPOPT fail during setup.
        return solver

    if solver_name == "fatrop":
        solver = factory(
            _structure_detection=fatrop_structure_detection,
            _c_compile=fatrop_c_compile,
        )
    else:
        solver = factory()
    solver.set_convergence_tolerance(tolerance)
    solver.set_constraint_tolerance(tolerance)
    solver.set_maximum_iterations(max_iterations)

    if solver_name == "fatrop":
        solver.set_print_level(print_level)
        # Fatrop relaxes each bound relatively. For large fatigue capacity
        # states (~7000), the native 1e-8 relaxation permits about 7e-5
        # absolute overshoot. The Bioptim integration tightens only the solver
        # call bounds while retaining the physical limits for the audit.
        set_bound_tightening = getattr(
            solver, "set_bound_tightening_factor", None
        )
        if set_bound_tightening is None:
            raise SolverBackendUnavailable(
                "This Fatrop benchmark requires the Bioptim interface with "
                "set_bound_tightening_factor()."
            )
        set_bound_tightening(fatrop_bound_tightening_factor)
        solver.set_c_compile(fatrop_c_compile)
        return solver

    if solver_name == "madnlp":
        # Bypass Bioptim's generic print-level mapping. The pinned libMad
        # runtime embeds MadNLP 0.9.2, whose LogLevels enum accepts only 1..6
        # and uses ERROR=6 as the quiet benchmark setting.
        madnlp_print_level = (
            MADNLP_QUIET_PRINT_LEVEL
            if int(print_level) == 0
            else max(1, min(int(print_level), MADNLP_QUIET_PRINT_LEVEL))
        )
        solver.set_option_unsafe(madnlp_print_level, "print_level")
        # The audited madnlp_c runtimes either reject or silently ignore
        # ``dual_initialized``/``mu_init`` and lam_g0/lam_x0. The reliable hot
        # start is therefore the shifted, projected primal trajectory supplied
        # by Cocofest, without claiming a solver-level dual initialization.
        if madnlp_linear_solver is not None:
            runtime_linear_solver = MADNLP_LINEAR_SOLVER_RUNTIME_NAMES.get(
                madnlp_linear_solver, madnlp_linear_solver
            )
            solver.set_option_unsafe(runtime_linear_solver, "linear_solver")
        if madnlp_max_wall_time is not None:
            solver.set_option_unsafe(float(madnlp_max_wall_time), "max_wall_time")
        solver.set_c_compile(madnlp_c_compile)
        return solver

    solver.set_print_level(print_level)
    solver.set_alm_maximum_iterations(
        max_iterations
        if alpaqa_alm_max_iterations is None
        else alpaqa_alm_max_iterations
    )
    solver.set_lbfgs_memory(alpaqa_lbfgs_memory)
    if alpaqa_max_wall_time is not None:
        if alpaqa_max_wall_time <= 0:
            raise ValueError("alpaqa_max_wall_time must be strictly positive.")
        solver.set_maximum_wall_time(alpaqa_max_wall_time)
    if alpaqa_initial_penalty is not None:
        if alpaqa_initial_penalty < 0:
            raise ValueError("alpaqa_initial_penalty must be non-negative.")
        solver.set_initial_penalty(alpaqa_initial_penalty)
    if alpaqa_initial_tolerance is not None:
        solver.set_initial_tolerance(alpaqa_initial_tolerance)
    if alpaqa_penalty_update_factor is not None:
        solver.set_penalty_update_factor(alpaqa_penalty_update_factor)
    if alpaqa_maximum_penalty is not None:
        solver.set_maximum_penalty(alpaqa_maximum_penalty)
    if alpaqa_panoc_max_wall_time is not None:
        solver.set_option_unsafe(
            f"{alpaqa_panoc_max_wall_time}s", "panoc.max_time"
        )
    if alpaqa_max_no_progress is not None:
        solver.set_option_unsafe(alpaqa_max_no_progress, "panoc.max_no_progress")
    return solver
