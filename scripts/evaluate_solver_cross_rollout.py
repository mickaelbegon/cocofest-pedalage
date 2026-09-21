#!/usr/bin/env python3
"""Évaluer des commandes IPOPT/ACADOS avec DOP853, Radau-5 et Gauss-Legendre 4×5."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.solver_cross_rollout import CommonCost, EVALUATORS, run_matrix


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--reduced-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, help="Paramètres effectifs complets, obligatoires si absents du NPZ")
    parser.add_argument("--cost-config", type=Path, help="JSON CommonCost partagé par toutes les sources")
    parser.add_argument("--evaluators", nargs="+", choices=EVALUATORS, default=list(EVALUATORS))
    parser.add_argument("--cycle-start", type=int, default=0)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--cycle-duration", type=float, help="Durée explicite obligatoire pour une archive dynamique legacy")
    parser.add_argument("--samples-per-interval", type=int, default=9)
    parser.add_argument("--rtol", type=float, default=1e-11)
    parser.add_argument("--atol", type=float, default=1e-13)
    parser.add_argument("--qualified-max-velocity-violation-rad-s", type=float, default=0.01,
                        help="Maximum DOP853 speed-bound excursion for a qualified, non-certifying ranking.")
    parser.add_argument("--qualified-max-velocity-ratio", type=float, default=2.0,
                        help="Maximum ratio between nonzero DOP853 speed excursions for qualified ranking.")
    args = parser.parse_args(argv)
    try:
        cost = CommonCost(**json.loads(args.cost_config.read_text())) if args.cost_config else CommonCost()
        report = run_matrix(args.source, args.reduced_profile, args.output_dir, model_config=args.model_config,
                            evaluators=args.evaluators, cycle_start=args.cycle_start, cycles=args.cycles,
                            cycle_duration=args.cycle_duration, samples_per_interval=args.samples_per_interval,
                            cost=cost, rtol=args.rtol, atol=args.atol,
                            qualified_max_velocity_violation_rad_s=args.qualified_max_velocity_violation_rad_s,
                            qualified_max_velocity_ratio=args.qualified_max_velocity_ratio)
    except (ValueError, OSError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({"report": str((args.output_dir/"matrix.json").resolve()),
                      "comparison": report["comparison"], "rankings": report["rankings"],
                      "qualified_rankings": report["qualified_rankings"]}, indent=2))
    return 0 if all(cell["status"] == "success" for row in report["sources"] for cell in row["evaluations"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
