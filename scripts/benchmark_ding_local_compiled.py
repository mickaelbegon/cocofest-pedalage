"""Compile IPOPT callbacks *after* the exact local Ding substitution.

This experimental entry point deliberately leaves Bioptim's outer compilation
disabled: the native callbacks must be generated from the reduced NLP, whose
Radau offsets remain run-time parameters. All original-space audits, bounds,
dual transport and frozen replay are inherited from the local adapter.

Usage::

    python scripts/benchmark_ding_local_compiled.py \
        --callbacks hessian --cache-dir RUN/cache local -- COMPARISON_ARGUMENTS

Use ``--callbacks all`` to compile all five IPOPT evaluation callbacks. The
production capability guard is not relaxed by this benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import benchmark_ding_radau5_local_ab as local


def compiled_factory(factory, *, cache_dir, callbacks="hessian", cache_name="ding-local", report=None):
    """Wrap the NLP factory at the post-reduction boundary, with exact derivatives.

    No ``fast-math`` or approximation is used. The cache implementation includes
    generated source and toolchain in its content signature. A new solver and
    derivatives are prepared once, then reused with different offsets ``p``.
    """
    from bioptim.interfaces.c_compile_cache import cached_nlpsol_callbacks

    if callbacks not in {"hessian", "all"}:
        raise ValueError("callbacks must be hessian or all")
    selected = ("nlp_hess_l",) if callbacks == "hessian" else None

    def build(name, plugin, nlp, options):
        if plugin != "ipopt":
            raise ValueError("Local compiled benchmark requires IPOPT")
        tic = time.perf_counter()
        interpreted = factory(name, plugin, nlp, options)
        interpreted_build_s = time.perf_counter() - tic
        tic = time.perf_counter()
        solver, information = cached_nlpsol_callbacks(
            plugin, nlp, options, compiler_flags=("-O1",), cache_dir=cache_dir,
            cache_name=cache_name, vm_solver=interpreted, callbacks=selected,
        )
        if report is not None:
            report.update(
                interpreted_build_s=interpreted_build_s,
                compilation_and_reload_s=time.perf_counter() - tic,
                compiler_flags=["-O1"], compilation=information,
                solver_nx=nlp["x"].numel(), solver_ng=nlp["g"].numel(),
                runtime_parameter_count=nlp.get("p", local.ca.SX.zeros(0)).numel(),
            )
        return solver

    return build


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--callbacks", choices=("hessian", "all"), default="hessian")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--cache-name", default="ding-local")
    parser.add_argument("mode", choices=("baseline", "local"))
    parser.add_argument("comparison_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    # The comparison entry point changes cwd before constructing the NLP.
    # Resolve caller-owned paths now, before delegating to that entry point.
    args.cache_dir = args.cache_dir.resolve()
    comparison_args = args.comparison_arguments
    if comparison_args and comparison_args[0] == "--":
        comparison_args = comparison_args[1:]
    if "--output-json" not in comparison_args:
        parser.error("comparison arguments must include --output-json")
    # This entry point owns compilation after reduction. Let the standard
    # interpreted adapter reject any accidental outer compilation/transform.
    class CompiledLocalSolver(local.LocalSolver):
        def __init__(self, factory, name, plugin, nlp, options, interface, mode, output, index):
            self.compilation_report = {}
            wrapped = compiled_factory(
                factory, cache_dir=args.cache_dir.resolve(), callbacks=args.callbacks,
                cache_name=f"{args.cache_name}-nlp{index}", report=self.compilation_report,
            )
            super().__init__(wrapped, name, plugin, nlp, options, interface, mode, output, index)
            output.mkdir(parents=True, exist_ok=True)
            (output / f"nlp{index}_compilation.json").write_text(
                json.dumps(self.compilation_report, indent=2) + "\n"
            )
            print("DING_LOCAL_COMPILED " + json.dumps(self.compilation_report), flush=True)

    local.base.AuditedSolver = CompiledLocalSolver
    sys.argv = [sys.argv[0], "condensed" if args.mode == "local" else "baseline", *comparison_args]
    local.base.main()


if __name__ == "__main__":
    main()
