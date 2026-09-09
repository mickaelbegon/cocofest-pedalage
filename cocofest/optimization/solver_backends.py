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
import platform
from pathlib import Path
import struct
import sys
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


def _elf_identity(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read the ABI-relevant fields of an ELF header without executing it."""

    resolved = Path(path).expanduser().resolve()
    identity: dict[str, Any] = {
        "format": None,
        "class_bits": None,
        "endianness": None,
        "machine": None,
    }
    try:
        header = resolved.read_bytes()[:20]
    except OSError as error:
        identity["error"] = f"{type(error).__name__}: {error}"
        return identity
    if len(header) < 20 or header[:4] != b"\x7fELF":
        identity["format"] = "not_elf"
        return identity
    identity["format"] = "ELF"
    identity["class_bits"] = {1: 32, 2: 64}.get(header[4])
    identity["endianness"] = {1: "little", 2: "big"}.get(header[5])
    byte_order = "<" if header[5] == 1 else ">" if header[5] == 2 else None
    if byte_order is not None:
        identity["machine"] = struct.unpack(f"{byte_order}H", header[18:20])[0]
    return identity


def process_runtime_provenance() -> dict[str, Any]:
    """Describe the Python process ABI used to load CasADi, IPOPT and HSL."""

    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": file_provenance(sys.executable),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "pointer_bits": struct.calcsize("P") * 8,
        "byteorder": sys.byteorder,
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
    }


def loaded_solver_runtime_provenance() -> list[dict[str, Any]]:
    """Fingerprint solver-relevant shared objects mapped in this process.

    ``/proc/self/maps`` is Linux-specific.  Other platforms return an empty
    list rather than pretending that the loaded Fortran/BLAS ABI was audited.
    """

    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return []
    relevant = (
        "hsl",
        "ipopt",
        "gfortran",
        "quadmath",
        "blas",
        "lapack",
        "openblas",
        "mkl",
        "metis",
    )
    paths: set[Path] = set()
    try:
        for line in maps.read_text().splitlines():
            candidate = line.rsplit(maxsplit=1)[-1]
            if not candidate.startswith("/"):
                continue
            path = Path(candidate)
            if any(token in path.name.lower() for token in relevant) and path.is_file():
                paths.add(path.resolve())
    except OSError:
        return []
    return [
        {**(file_provenance(path) or {}), "elf": _elf_identity(path)}
        for path in sorted(paths, key=str)
    ]


def _runtime_abi_audit(report: dict[str, Any]) -> dict[str, Any]:
    libraries = report.get("loaded_solver_libraries") or []
    fortran_names = sorted(
        {
            Path(record.get("resolved_path", "")).name
            for record in libraries
            if "gfortran" in Path(record.get("resolved_path", "")).name.lower()
        }
    )
    fortran_majors = sorted(
        {
            name.rsplit(".so.", 1)[1].split(".", 1)[0]
            for name in fortran_names
            if ".so." in name
        }
    )
    # CasADi's IPOPT wheel carries its own ``libcoinmetis`` while a legacy
    # CoinHSL can load an external ``libmetis`` at run time.  They export some
    # of the same unversioned symbols.  Keep that fact in the probe evidence:
    # it explains otherwise cryptic METIS diagnostics without claiming that a
    # small MA57 solve necessarily exercised a conflicting METIS call.
    metis_provider_names = sorted(
        {
            Path(record.get("resolved_path", "")).name
            for record in libraries
            if "metis" in Path(record.get("resolved_path", "")).name.lower()
        }
    )
    hsl_elf = (report.get("hsl_library") or {}).get("elf") or {}
    process = report.get("process_runtime") or {}
    class_matches = hsl_elf.get("class_bits") == process.get("pointer_bits")
    endian_matches = hsl_elf.get("endianness") == process.get("byteorder")
    warnings: list[str] = []
    if len(fortran_majors) > 1:
        warnings.append(
            "Multiple libgfortran ABI majors are mapped in the successful probe; "
            "rebuilding CoinHSL against the active environment remains recommended."
        )
    if len(metis_provider_names) > 1:
        warnings.append(
            "Multiple METIS providers are mapped; a CoinHSL rebuild must not emit "
            "a METIS runtime diagnostic in the isolated MA57 probe."
        )
    if not class_matches or not endian_matches:
        warnings.append("The HSL ELF class or endianness differs from the Python process ABI.")
    return {
        "hsl_process_class_matches": class_matches,
        "hsl_process_endianness_matches": endian_matches,
        "loaded_fortran_runtime_names": fortran_names,
        "loaded_fortran_abi_majors": fortran_majors,
        "multiple_fortran_abi_majors": len(fortran_majors) > 1,
        "loaded_metis_provider_names": metis_provider_names,
        "multiple_metis_providers": len(metis_provider_names) > 1,
        "warnings": warnings,
    }


def _ma57_probe_production_readiness(report: dict[str, Any]) -> tuple[bool, list[str]]:
    """Separate a successful toy solve from a production-clean native stack."""

    reasons: list[str] = []
    if report.get("functional_success") is not True:
        reasons.append("probe_solve_failed")
    abi_audit = report.get("abi_audit") or {}
    if abi_audit.get("hsl_process_class_matches") is not True:
        reasons.append("hsl_process_class_mismatch")
    if abi_audit.get("hsl_process_endianness_matches") is not True:
        reasons.append("hsl_process_endianness_mismatch")
    if abi_audit.get("multiple_fortran_abi_majors") is True:
        reasons.append("multiple_libgfortran_abi_majors")
    return not reasons, reasons


def solve_ipopt_ma57_probe(
    hsl_library: str | os.PathLike[str],
    *,
    casadi_module=None,
) -> dict[str, Any]:
    """Run a tiny constrained NLP that must factorize through IPOPT/MA57.

    This is deliberately separate from :func:`ipopt_hsl_diagnostics`: a
    successful ``dlopen`` cannot detect integer-width, Fortran-runtime or BLAS
    ABI incompatibilities.  Production runners should execute this function
    in a disposable subprocess because an incompatible native library may
    terminate the interpreter before Python can raise an exception.
    """

    if casadi_module is None:
        import casadi as casadi_module

    library = Path(hsl_library).expanduser().resolve()
    report: dict[str, Any] = {
        "schema": "cocofest-ipopt-ma57-runtime-probe-v2",
        "functional_success": False,
        "success": False,
        "hsl_library": {
            **(file_provenance(library) or {}),
            "elf": _elf_identity(library),
        },
        "process_runtime": process_runtime_provenance(),
        "casadi": {
            "version": getattr(casadi_module, "__version__", None),
            "module": file_provenance(getattr(casadi_module, "__file__", None)),
        },
        "submitted_options": {
            "linear_solver": "ma57",
            "hsllib": str(library),
            "max_iter": 20,
            "tol": 1e-10,
        },
        "return_status": None,
        "iteration_count": None,
        "constraint_residual": None,
        "solution_error_inf": None,
        "loaded_solver_libraries": [],
        "production_ready": False,
        "production_readiness_reasons": [],
        "error": None,
    }
    try:
        x = casadi_module.MX.sym("x", 2)
        nlp = {
            "x": x,
            "f": (x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2,
            "g": x[0] + x[1],
        }
        solver = casadi_module.nlpsol(
            "cocofest_ma57_runtime_probe",
            "ipopt",
            nlp,
            {
                "ipopt.linear_solver": "ma57",
                "ipopt.hsllib": str(library),
                "ipopt.max_iter": 20,
                "ipopt.tol": 1e-10,
                "ipopt.print_level": 0,
                "print_time": False,
            },
        )
        solution = solver(x0=[0.0, 0.0], lbg=3.0, ubg=3.0)
        values = [float(value) for value in solution["x"].full().reshape(-1)]
        constraint_residual = abs(sum(values) - 3.0)
        solution_error = max(abs(values[0] - 1.0), abs(values[1] - 2.0))
        stats = solver.stats()
        report.update(
            {
                "return_status": stats.get("return_status"),
                "iteration_count": stats.get("iter_count"),
                "constraint_residual": constraint_residual,
                "solution_error_inf": solution_error,
                "loaded_solver_libraries": loaded_solver_runtime_provenance(),
            }
        )
        hsl_path = str(library)
        hsl_sha256 = (report.get("hsl_library") or {}).get("sha256")
        loaded_hsl = next(
            (
                item
                for item in report["loaded_solver_libraries"]
                if item.get("resolved_path") == hsl_path
                and item.get("sha256") == hsl_sha256
            ),
            None,
        )
        report["selected_hsl_mapped"] = loaded_hsl is not None
        report["functional_success"] = bool(
            stats.get("success")
            and stats.get("return_status") in {"Solve_Succeeded", "Solved_To_Acceptable_Level"}
            and constraint_residual <= 1e-8
            and solution_error <= 1e-6
            and report["selected_hsl_mapped"]
        )
        report["success"] = report["functional_success"]
    except Exception as error:  # native/plugin failures vary by CasADi build
        report["error"] = f"{type(error).__name__}: {error}"
        report["loaded_solver_libraries"] = loaded_solver_runtime_provenance()
    report["abi_audit"] = _runtime_abi_audit(report)
    (
        report["production_ready"],
        report["production_readiness_reasons"],
    ) = _ma57_probe_production_readiness(report)
    report["abi_audit"]["status"] = (
        "clean"
        if report["production_ready"]
        else "warning"
        if report["functional_success"]
        else "incompatible"
    )
    return report


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
