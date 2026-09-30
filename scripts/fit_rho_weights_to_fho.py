#!/usr/bin/env python3
"""Identify four bounded RHO fatigue weights from FHO windows.

The command template is intentionally explicit: it must create
``{rho_solution}`` and use ``{fho_window_solution}`` to initialise the RHO at
the beginning of that exact FHO window.  It receives weights in both CSV and
named JSON form.  No active campaign directory is ever modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from cocofest.optimization.fho_rho_weight_fit import (  # noqa: E402
    MUSCLE_NAMES, SCHEMA_VERSION, TrajectoryArchive, fit_window, window_archive, write_archive,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fho-solution", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--window-cycles", type=int, default=10)
    p.add_argument("--max-cycles", type=int, default=None)
    p.add_argument("--evaluations", type=int, default=96,
                   help="Global Latin-hypercube candidates per FHO block on the L1 simplex.")
    p.add_argument("--refine-best", type=int, default=4,
                   help="Number of best feasible global candidates to refine locally.")
    p.add_argument("--refine-evaluations", type=int, default=8,
                   help="Additional local candidates per retained global candidate.")
    p.add_argument("--refine-concentration", type=float, default=30.0,
                   help="Dirichlet concentration for local simplex refinement (higher is more local).")
    p.add_argument(
        "--parallel-workers",
        type=int,
        default=8,
        help="Number of independent RHO candidates evaluated concurrently per FHO window.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rho-command-template", required=True,
                   help="Shell-style argv template; placeholders: {weights_csv}, {weights_json}, "
                        "{weight_delt_ant}, {weight_delt_post}, {weight_biceps}, {weight_triceps}, "
                        "{fho_window_solution}, {rho_solution}, {window_start}, {window_end}, "
                        "{window_cycles}, {evaluation}.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if (args.window_cycles < 1 or args.evaluations < 5 or args.parallel_workers < 1
            or args.refine_best < 0 or args.refine_evaluations < 0 or args.refine_concentration <= 0):
        raise ValueError(
            "--window-cycles and --parallel-workers must be positive and "
            "--evaluations must be at least five."
        )
    reference = TrajectoryArchive.load(args.fho_solution)
    total = reference.cycles(controls_per_cycle=30)
    total = min(total, args.max_cycles) if args.max_cycles is not None else total
    out = args.output_dir.expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": SCHEMA_VERSION, "method": "simplex_latin_hypercube_plus_dirichlet_refinement_v2",
              "fho_solution": str(reference.path), "muscle_names": list(MUSCLE_NAMES),
              "weight_bounds": [0.0, 1.0], "weight_constraint": "w_i >= 0; sum_i(w_i) = 1",
              "global_evaluations": args.evaluations, "refine_best": args.refine_best,
              "refine_evaluations_per_best": args.refine_evaluations,
              "refine_concentration": args.refine_concentration, "window_cycles": args.window_cycles,
              "normalization": "per-variable robust FHO 5-95 span; controls divided by 0.0006 s; equal state/control groups",
              "windows": []}
    for window_index, start in enumerate(range(1, total + 1, args.window_cycles)):
        end = min(start + args.window_cycles - 1, total)
        window = window_archive(reference, start, end)
        window_dir = out / f"cycles-{start:04d}-{end:04d}"; window_dir.mkdir(exist_ok=True)
        window_path = write_archive(window, window_dir / "fho-window.npz")
        def solve(weights, evaluation):
            rho_solution = window_dir / f"rho-eval-{evaluation:03d}.npz"
            named = dict(zip(MUSCLE_NAMES, map(float, weights)))
            values = {"weights_csv": ",".join(f"{v:.17g}" for v in weights),
                      "weights_json": json.dumps(named, sort_keys=True),
                      "fho_window_solution": str(window_path), "rho_solution": str(rho_solution),
                      "window_start": start, "window_end": end,
                      "window_cycles": end - start + 1, "evaluation": evaluation}
            values.update({f"weight_{name.lower()}": f"{value:.17g}"
                           for name, value in named.items()})
            command = [part.format(**values) for part in shlex.split(args.rho_command_template)]
            completed = subprocess.run(command, cwd=REPO, check=False, text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (window_dir / f"rho-eval-{evaluation:03d}.log").write_text(completed.stdout)
            if completed.returncode:
                raise RuntimeError(f"RHO returned {completed.returncode}; see its log.")
            if not rho_solution.is_file():
                raise FileNotFoundError(f"RHO did not create {rho_solution}.")
            return rho_solution
        result = fit_window(
            window,
            solve,
            window_index=window_index,
            evaluations=args.evaluations,
            seed=args.seed,
            parallel_workers=args.parallel_workers,
            refine_best=args.refine_best,
            refine_evaluations=args.refine_evaluations,
            refine_concentration=args.refine_concentration,
        )
        result.update({"cycle_start": start, "cycle_end": end,
                       "fho_window_solution": str(window_path)})
        report["windows"].append(result)
        (window_dir / "fit-result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
        (out / "fit-results.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
