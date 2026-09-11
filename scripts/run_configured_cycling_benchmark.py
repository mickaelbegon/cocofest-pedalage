#!/usr/bin/env python3
"""Common config-driven runner for rho, rho-physio, rho-pace and fho.

All four conditions use the same parameter factory adapter. FHO requires the
benchmark's --single-shot form; no weights or FHO trajectory are passed to
the adaptive policy. Every supplied seed must carry this model's fingerprint.
"""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ.setdefault("MPLBACKEND", "Agg")
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

CONDITIONS = ("rho", "rho-physio", "rho-pace", "fho")
SEED_ARGUMENTS = ("common_initial_solution", "standard_warmup_seed", "full_horizon_prefix_solution")
OUTPUT_ARGUMENTS = ("common_initial_solution_output", "receding_horizon_solution_output",
                    "rho_replay_checkpoint_output")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--weights-config", type=Path)
    parser.add_argument("--weights-journal", type=Path)
    parser.add_argument("--configuration-audit", type=Path)
    parser.add_argument("benchmark_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    caller_directory = Path.cwd()
    benchmark_argv = args.benchmark_args[1:] if args.benchmark_args[:1] == ["--"] else args.benchmark_args

    from cocofest.optimization.configured_cycling_model import (
        FINGERPRINT_KEY, annotate_generated_seed, configured_model_factories,
        require_seed_fingerprint, resolve_model_config,
    )
    from examples.fes_multibody.cycling import cycling_fes_solver_comparison as benchmark
    from scripts.run_rho_pace_benchmark import main as run_weighted

    model_path = args.model_config.expanduser().resolve()
    model_bytes = model_path.read_bytes()
    config = resolve_model_config(json.loads(model_bytes))
    parsed = benchmark.build_cli().parse_args(benchmark_argv)
    if parsed.output_json is None:
        raise ValueError("Configured conditions require an explicit --output-json")
    if parsed.formulation != "dynamic" or parsed.mechanical_formulation != "reduced":
        raise ValueError("Configured comparison requires dynamic reduced mechanics")
    if tuple(parsed.solvers) != ("ipopt",):
        raise ValueError("Configured comparison currently supports IPOPT only")
    if parsed.single_shot != (args.condition == "fho"):
        raise ValueError("Only condition=fho requires --single-shot; RHO conditions must omit it")
    if not 1 <= parsed.n_windows <= 100:
        raise ValueError("Configured conditions are limited to 1..100 cycles")
    if args.condition == "fho" and parsed.cycles_per_window != parsed.n_windows:
        raise ValueError("FHO cycles-per-window must equal n-windows")
    if args.condition != "fho" and parsed.cycles_per_window != 1:
        raise ValueError("RHO conditions require cycles-per-window=1")
    if parsed.resistive_torque is None or not 0 < parsed.resistive_torque < float("inf"):
        raise ValueError("Configured comparison requires explicit positive --signed-crank-torque")
    if parsed.rho_prepared_checkpoint_output_template:
        raise ValueError("Templated checkpoint outputs are not fingerprinted by this runner; use explicit output paths")
    # Historical pickles have no parameter fingerprint. The configured model
    # must generate its own warmup unless an explicitly compatible seed exists.
    if not parsed.ipopt_disable_historical_initial_guess:
        benchmark_argv = [*benchmark_argv, "--ipopt-disable-historical-initial-guess"]
        parsed.ipopt_disable_historical_initial_guess = True
    weighted = args.condition in {"rho-physio", "rho-pace"}
    if weighted and (args.weights_config is None or args.weights_journal is None):
        raise ValueError("Weighted conditions require --weights-config and --weights-journal")
    if weighted:
        weights_path = args.weights_config.expanduser().resolve()
        declared_weights = json.loads(weights_path.read_text())
        if not isinstance(declared_weights.get("initial_weights"), dict) or not declared_weights["initial_weights"]:
            raise ValueError("Configured physiological conditions require explicit named initial_weights")
        if set(declared_weights["initial_weights"]) != set(config["muscles"]):
            raise ValueError("Weight and model configuration muscle names must match")

    output_json = Path(parsed.output_json).expanduser().resolve()
    audit_path = (args.configuration_audit.expanduser().resolve() if args.configuration_audit
                  else output_json.with_suffix(".configuration.json"))
    generated_outputs = [Path(getattr(parsed, name)).expanduser().resolve()
                         for name in OUTPUT_ARGUMENTS if getattr(parsed, name)]
    destinations = [output_json, audit_path, *generated_outputs]
    if weighted:
        destinations.append(args.weights_journal.expanduser().resolve())
    if len(set(destinations)) != len(destinations):
        raise ValueError("Configured result, audit, weights journal and seed outputs must use distinct paths")
    for path in destinations:
        if path.exists():
            raise FileExistsError(f"Configured benchmark requires a fresh output: {path}")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    audit = {"schema_version": 1, "condition": args.condition, "status": "prepared",
             "model_config_path": str(model_path), "model_config_sha256": sha256(model_bytes).hexdigest(),
             "model": config, "weights_requested": weighted, "weights_applied": False,
             "adaptation_enabled": args.condition == "rho-pace", "uses_fho_data_for_weights": False,
             "arguments": benchmark_argv, "seed_checks": [], "model_builds": records}
    if weighted:
        audit["weights_config_path"] = str(weights_path)
        audit["weights_config_sha256"] = sha256(weights_path.read_bytes()).hexdigest()
        audit["weights_journal_path"] = str(args.weights_journal.expanduser().resolve())

    def write_audit():
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n")

    write_audit()
    try:
        for name in SEED_ARGUMENTS:
            path = getattr(parsed, name)
            if path is not None:
                audit["seed_checks"].append(require_seed_fingerprint(path, config[FINGERPRINT_KEY]))
        with configured_model_factories(config, records):
            if weighted:
                run_weighted(["--pace-config", str(weights_path), "--pace-journal",
                              str(args.weights_journal.expanduser().resolve()), "--", *benchmark_argv],
                             adaptation_enabled=args.condition == "rho-pace")
            else:
                benchmark.main(**vars(parsed))
        if not records:
            raise RuntimeError("Benchmark never constructed the explicitly configured model")
        audit["weights_applied"] = weighted
        for path in generated_outputs:
            if path.is_file():
                annotate_generated_seed(path, config, condition=args.condition)
        audit["status"] = "completed"
        audit["physical_outcome"] = "see_benchmark_result; launcher completion is not physical certification"
    except BaseException as error:
        audit.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        os.chdir(caller_directory)
        write_audit()


if __name__ == "__main__":
    main()
