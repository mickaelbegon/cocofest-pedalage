"""Read-only runtime evidence before a solve; no native plugin is loaded.

Call in the scientific worker after its imports and patch installation. Calling
from a launcher describes the launcher's interpreter, not the child runtime.
The report deliberately does not certify a native ABI or successful solve.
"""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import struct
import sys
from typing import Mapping

from cocofest.optimization.solver_backends import MADNLP_LINEAR_SOLVER_RUNTIME_NAMES

_PACKAGES = {
    "casadi": ("casadi",),
    "bioptim": ("bioptim",),
    "biorbd_casadi": ("biorbd-casadi", "biorbd"),
    "acados_template": ("acados-template",),
    "numpy": ("numpy",),
    "scipy": ("scipy",),
}
_ENVIRONMENT_KEYS = (
    "ACADOS_SOURCE_DIR",
    "JULIA_DEPOT_PATH",
    "JULIA_HSL_LIBRARY_PATH",
    "JULIA_CUDA_USE_COMPAT",
    "LD_LIBRARY_PATH",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "JULIA_NUM_THREADS",
)
_LIBRARY_TOKENS = (
    "casadi",
    "acados",
    "hpipm",
    "blasfeo",
    "ipopt",
    "madnlp",
    "libmad.",
    "libjulia",
    "hsl",
    "mumps",
    "gfortran",
    "openblas",
    "libblas",
    "lapack",
    "metis",
)


def _package_record(name: str) -> dict:
    module = sys.modules.get(name)
    # vars() avoids lazy module __getattr__ hooks and therefore new imports.
    namespace = vars(module) if module is not None else {}
    distributions = []
    for distribution in _PACKAGES[name]:
        try:
            distributions.append(
                {"name": distribution, "version": metadata.version(distribution)}
            )
        except metadata.PackageNotFoundError:
            continue
    version = namespace.get("__version__")
    source = namespace.get("__file__")
    return {
        "loaded": module is not None,
        "loaded_version": version if isinstance(version, (str, int, float)) else None,
        "loaded_path": (
            os.fspath(source) if isinstance(source, (str, os.PathLike)) else None
        ),
        "distributions": distributions,
        "availability": (
            "loaded"
            if module is not None
            else "installed_not_loaded" if distributions else "unknown"
        ),
    }


def _mapped_libraries() -> dict:
    """Only inspect current Linux mappings; never dlopen a candidate library."""
    try:
        lines = Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return {"status": "unavailable", "libraries": [], "reason": str(error)}
    paths = set()
    for line in lines:
        parts = line.split(maxsplit=5)
        if len(parts) != 6 or not parts[5].startswith("/"):
            continue
        path = parts[5]
        if any(token in Path(path).name.lower() for token in _LIBRARY_TOKENS):
            paths.add(path)
    records = []
    for path in sorted(paths):
        record = {"path": path, "elf_pointer_bits": None}
        try:
            with Path(path).open("rb") as handle:
                header = handle.read(6)
            if header[:4] == b"\x7fELF" and len(header) >= 6:
                record["elf_pointer_bits"] = {1: 32, 2: 64}.get(header[4])
        except OSError as error:
            record["inspection_error"] = str(error)
        records.append(record)
    return {"status": "observed", "libraries": records}


def collect_runtime_preflight(
    *,
    solver: str,
    linear_solver: str | None = None,
    expected_versions: Mapping[str, str] | None = None,
    require_acados_ding_patch: bool = False,
) -> dict:
    """Return JSON evidence and fail only on an observed contradiction.

    ``expected_versions`` are exact loaded-module versions supplied by the
    caller's campaign; no guessed supported-version ranges are imposed.
    Unknown versions or unloaded modules are unresolved checks, never passes.
    ``linear_solver`` records requested configuration, not a native selection.
    """
    if solver not in ("ipopt", "madnlp", "fatrop", "alpaqa", "acados"):
        raise ValueError(f"Unsupported solver: {solver!r}")
    if linear_solver is not None and (
        not isinstance(linear_solver, str) or not linear_solver.strip()
    ):
        raise ValueError("linear_solver must be a nonempty string or None")
    expected_versions = dict(expected_versions or {})
    if set(expected_versions) - set(_PACKAGES):
        raise ValueError("Expected versions contain an unknown package")
    if any(
        not isinstance(value, str) or not value for value in expected_versions.values()
    ):
        raise ValueError("Expected versions must be nonempty exact version strings")
    packages = {name: _package_record(name) for name in _PACKAGES}
    checks = []
    for name, expected in expected_versions.items():
        observed = packages[name]["loaded_version"]
        checks.append(
            {
                "name": f"version:{name}",
                "expected": expected,
                "observed": observed,
                "status": (
                    "unknown"
                    if observed is None
                    else "pass" if str(observed) == expected else "fail"
                ),
            }
        )
    bioptim = sys.modules.get("bioptim")
    factory_namespace = vars(bioptim).get("Solver") if bioptim is not None else None
    factory = (
        getattr(factory_namespace, solver.upper(), None)
        if factory_namespace is not None
        else None
    )
    checks.append(
        {
            "name": "bioptim_solver_factory",
            "expected": solver.upper(),
            "status": (
                "unknown"
                if factory_namespace is None
                else "pass" if callable(factory) else "fail"
            ),
        }
    )
    acados_interface = sys.modules.get("bioptim.interfaces.acados_interface")
    interface_type = (
        vars(acados_interface).get("AcadosInterface")
        if acados_interface is not None
        else None
    )
    patched = (
        getattr(interface_type, "_cocofest_ding_local_patch", False)
        if interface_type is not None
        else None
    )
    patch = {
        "name": "acados_ding_local_reduction",
        "required": require_acados_ding_patch,
        "observed": patched,
        "status": (
            "unknown"
            if patched is None
            else "installed" if patched else "not_installed"
        ),
    }
    if require_acados_ding_patch:
        checks.append(
            {
                "name": "required_acados_ding_patch",
                "status": (
                    "unknown" if patched is None else "pass" if patched else "fail"
                ),
            }
        )
    mappings = _mapped_libraries()
    pointer_bits = struct.calcsize("P") * 8
    for library in mappings["libraries"]:
        if (
            library["elf_pointer_bits"] is not None
            and library["elf_pointer_bits"] != pointer_bits
        ):
            checks.append(
                {
                    "name": "mapped_library_pointer_width",
                    "status": "fail",
                    "path": library["path"],
                }
            )
    # Both a discovered plugin filename and a configured name fall short of an
    # actual solver banner. Do not call has_nlpsol: it can load native code.
    checks.append(
        {
            "name": "native_backend_and_abi",
            "status": "unknown",
            "reason": "Requires a separate isolated native smoke solve and solver banner.",
        }
    )
    failures = any(check["status"] == "fail" for check in checks)
    return {
        "schema": "cocofest-runtime-preflight-v1",
        "status": "incompatible" if failures else "incomplete",
        "runtime": {
            "executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "pointer_bits": pointer_bits,
            "pid": os.getpid(),
        },
        "requested_backend": {
            "solver": solver,
            "linear_solver": linear_solver,
            "interface_linear_solver": (
                MADNLP_LINEAR_SOLVER_RUNTIME_NAMES.get(linear_solver, linear_solver)
                if solver == "madnlp"
                else linear_solver
            ),
        },
        "observed_backend": {
            "solver_version": None,
            "linear_solver": None,
            "reason": "Native solver identity cannot be inferred from configuration or mapped libraries.",
        },
        "packages": packages,
        "mapped_native_libraries": mappings,
        "patches": [patch],
        "checks": checks,
        "environment": {
            key: os.environ[key] for key in _ENVIRONMENT_KEYS if key in os.environ
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--solver",
        required=True,
        choices=("ipopt", "madnlp", "fatrop", "alpaqa", "acados"),
    )
    parser.add_argument("--linear-solver")
    parser.add_argument(
        "--expect-version", action="append", default=[], metavar="PACKAGE=VERSION"
    )
    parser.add_argument("--require-acados-ding-patch", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        help="Create a new report; refuse overwriting existing evidence",
    )
    args = parser.parse_args(argv)
    try:
        versions = {}
        for item in args.expect_version:
            name, separator, value = item.partition("=")
            if not separator or (name in versions and versions[name] != value):
                raise ValueError(
                    f"Invalid or conflicting version requirement: {item!r}"
                )
            versions[name] = value
        report = collect_runtime_preflight(
            solver=args.solver,
            linear_solver=args.linear_solver,
            expected_versions=versions,
            require_acados_ding_patch=args.require_acados_ding_patch,
        )
        document = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as output:
                output.write(document)
        else:
            print(document, end="")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    # 0 means the observation completed without a detected contradiction;
    # inspect status: incomplete is never a production-readiness certificate.
    return 1 if report["status"] == "incompatible" else 0


if __name__ == "__main__":
    raise SystemExit(main())
