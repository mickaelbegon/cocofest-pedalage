"""Profile construction of an isolated FHO up to (but excluding) IPOPT.

This runs the same application entry point as benchmark_fho_threaded_solve.py.
It intercepts Bioptim at generic_solve, before NLP dispatch or solver build, so
the report isolates application and OCP construction. No solve is performed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import resource
import runpy
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ProfileFinished(BaseException):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-command", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--map-first-node-only", action="store_true",
                        help="Experimental Bioptim continuity optimization for this audit")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    base = json.loads(args.base_command.read_text())
    cli = base[2:]
    for flag, value in (("--n-threads", str(args.threads)),
                        ("--output-json", str(output / "unused-result.json"))):
        if flag in cli:
            cli[cli.index(flag) + 1] = value
        else:
            cli.extend([flag, value])
    (output / "command.json").write_text(json.dumps([*base[:2], *cli], indent=2))

    import casadi as ca
    import bioptim.interfaces.ipopt_interface as ipopt_interface
    from bioptim.optimization.optimal_control_program import OptimalControlProgram
    from bioptim.limits.penalty_option import PenaltyOption

    started = time.perf_counter()
    stages: dict[str, dict[str, float | int]] = defaultdict(lambda: {"seconds": 0.0, "calls": 0})

    def wrap(cls, name: str) -> None:
        original = getattr(cls, name)

        def timed(self, *method_args, **method_kwargs):
            tic = time.perf_counter()
            try:
                return original(self, *method_args, **method_kwargs)
            finally:
                record = stages[f"{cls.__name__}.{name}"]
                record["seconds"] += time.perf_counter() - tic
                record["calls"] += 1

        setattr(cls, name, timed)

    for name in (
        "_prepare_all_decision_variables", "_check_arguments_and_build_nlp",
        "_prepare_dynamics", "_prepare_bounds_and_init",
        "_declare_multi_node_penalties", "_finalize_penalties",
        "_declare_continuity", "update_constraints", "update_objectives",
        "_prepare_vector_layout",
    ):
        wrap(OptimalControlProgram, name)
    for name in ("set_penalty", "_set_penalty_function", "get_variable_inputs"):
        wrap(PenaltyOption, name)

    if args.map_first_node_only:
        original_set = PenaltyOption._set_penalty_function

        def map_first_node_only(self, controllers, fcn):
            controller = controllers[-1] if isinstance(controllers, list) else controllers
            # Bioptim's threaded dispatch calls weighted_function[0] for all
            # nodes. Retain every per-node nonthreaded function, while avoiding
            # thousands of unused ThreadMap wrappers at nodes 1...N-1.
            skip = bool(self.multi_thread and len(self.node_idx) > 1
                        and controller.node_index != self.node_idx[0])
            if not skip:
                return original_set(self, controllers, fcn)
            self.multi_thread = False
            try:
                return original_set(self, controllers, fcn)
            finally:
                self.multi_thread = True

        PenaltyOption._set_penalty_function = map_first_node_only

    def finish(interface, expand_during_shake_tree=False):
        report = {
            "threads": args.threads,
            "map_first_node_only": args.map_first_node_only,
            "casadi_version": ca.__version__,
            "affinity": sorted(os.sched_getaffinity(0)),
            "numeric_environment": {k: os.environ.get(k) for k in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "elapsed_before_generic_solve_s": time.perf_counter() - started,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
            "stages": dict(stages),
        }
        (output / "report.json").write_text(json.dumps(report, indent=2))
        print("CONSTRUCTION_PROFILE " + json.dumps(report), flush=True)
        raise ProfileFinished()

    ipopt_interface.generic_solve = finish
    script = Path(base[1]).resolve()
    sys.argv = [str(script), *cli]
    sys.path.insert(0, str(script.parent))
    try:
        runpy.run_path(str(script), run_name="__main__")
    except ProfileFinished:
        return
    raise RuntimeError("The expected IPOPT entry point was not reached")


if __name__ == "__main__":
    main()
