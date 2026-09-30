#!/usr/bin/env python3
"""Print an incremental ablation manifest without importing or launching solvers.

Commands are candidates requiring the declared smoke/compatibility gates. Run in
the activated rho32 environment. CPU allocation is a coordinator responsibility;
this planner does not claim that taskset makes a CPU exclusive.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def build_manifest(output: Path, cpus: str, python: str, windows: int) -> dict:
    common = [
        "--objective", "fatigue", "--cycles-per-window", "1",
        "--stimulations-per-cycle", "50", "--n-windows", str(windows),
        "--n-threads", "1", "--signed-crank-torque", "0.1",
        "--nlp-tolerance", "1e-6", "--primal-feasibility-threshold", "1e-5",
        "--formulation", "dynamic", "--state-scaling", "full",
        "--first-node-wheel-q-slack", "0", "--terminal-wheel-q-slack", "0.002",
        "--rho-pulse-width-transfer-mode", "repeat",
        "--rho-pulse-width-extrapolation-factor", "1",
        "--reduced-internal-crank-velocity-guard", "off",
        "--ipopt-disable-historical-initial-guess",
        "--ipopt-linear-solver", "ma57",
    ]
    rows = []

    def add(identifier, parent, change, cli, gates=(), adapter=None):
        destination = output / identifier
        entry = ROOT / "examples/fes_multibody/cycling/cycling_fes_solver_comparison.py"
        prefix = [str(entry)] if adapter is None else [str(ROOT / adapter), "local", "--"]
        args = [*common, *cli, "--codegen-tag", "ablation_" + identifier,
                "--output-json", str(destination / "result.json"),
                "--receding-horizon-solution-output", str(destination / "trajectory.npz")]
        rows.append({"id": identifier, "parent": parent, "change": change,
                     "status": "planned_unvalidated", "gates": list(gates),
                     "working_directory": str(destination),
                     "command": ["taskset", "-c", cpus, python, *prefix, *args]})

    # The scientific-radau profiles reject these intentional model/SX overrides.
    # historical is used only as a CLI base; explicit options define the case.
    base = ["--solvers", "ipopt", "--ipopt-profile", "historical",
            "--ipopt-ode-solver", "collocation", "--ipopt-collocation-method", "radau",
            "--ipopt-enforce-start-constraints", "--ipopt-fatigue-warmstart-mode", "continuous",
            "--ipopt-disable-periodic-fes-warmup-projection"]
    state = {"degree": "3", "mechanics": "full", "model": "standard",
             "torque": "external_forces", "sx": False, "compile": False,
             "compact": False, "slew": False, "weight": "0"}

    def ipopt_cli(identifier):
        cli = [*base, "--ipopt-collocation-degree", state["degree"],
               "--mechanical-formulation", state["mechanics"],
               "--ipopt-model-formulation", state["model"],
               "--ipopt-torque-application", state["torque"],
               "--ipopt-use-sx" if state["sx"] else "--ipopt-no-use-sx"]
        if state["compile"]:
            cli += ["--ipopt-c-compile-callback", "nlp_hess_l",
                    "--ipopt-c-cache-dir", str(output / identifier / "native-cache"),
                    "--ipopt-c-cache-name", "rho", "--ipopt-c-compiler-flag=-O1"]
        if state["compact"]:
            cli += ["--compact-rho-output"]
        if state["slew"]:
            cli += ["--pulse-width-max-step-us", "100", "--pulse-width-slew-formulation", "lifting",
                    "--pulse-width-slew-weight", state["weight"], "--pulse-width-slew-reference-us", "100"]
        return cli

    changes = [
        ("K0", {}, "Article-method baseline in current code"),
        ("K1", {"degree": "5"}, "Radau degree 3 to 5"),
        ("K2", {"torque": "constant"}, "Constant generalized crank torque bridge"),
        ("K3", {"mechanics": "reduced"}, "Reduced mechanical coordinates"),
        ("K4", {"model": "periodic_node"}, "Periodic-node Ding calcium"),
        ("K5", {"sx": True}, "SX graph"),
        ("K6", {"compile": True}, "Compile exact Lagrangian Hessian callback"),
        ("K7", {"compact": True}, "Compact RHO output"),
        ("K8", {"slew": True}, "Hard 100 us PW increment bound"),
        ("K9", {"weight": "0.01"}, "Normalized squared PW increment penalty, weight 0.01"),
    ]
    parent = None
    k5_cli = None
    for identifier, patch, description in changes:
        state.update(patch)
        cli = ipopt_cli(identifier)
        add(identifier, parent, description, cli, ["2-window native smoke", "scientific configuration audit"])
        if identifier == "K5":
            k5_cli = cli
        parent = identifier
    add("K5-local", "K5", "Local Ding condensation side branch", k5_cli,
        ["Affine manifold and reconstructed bound audit", "2-window native smoke"],
        adapter="scripts/benchmark_ding_radau5_local_ab.py")

    acados = ["--solvers", "acados", "--acados-integrator-type", "IRK",
              "--acados-collocation-type", "GAUSS_LEGENDRE", "--acados-sim-stages", "4",
              "--acados-sim-steps", "5", "--acados-newton-iter", "5",
              "--acados-nlp-solver-type", "SQP", "--acados-hessian-approx", "GAUSS_NEWTON",
              "--acados-max-iter", "100", "--acados-tolerance", "1e-6"]
    add("A0", None, "ACADOS native full-mechanics reference", [*acados, "--mechanical-formulation", "full"],
        ["Certified common physical seed", "2-window native smoke", "Canonical objective/constraint audit"])
    acados += ["--mechanical-formulation", "reduced", "--experimental-reduced-acados"]
    add("A1", "A0", "Reduced mechanics", acados, ["2-window native smoke", "DOP853 matched accuracy"])
    acados += ["--acados-ding-local-reduction"]
    add("A2", "A1", "Local Ding native IRK reduction", acados, ["2-window native smoke", "Full native map audit"])
    acados += ["--compact-rho-output"]
    add("A3", "A2", "Compact RHO output", acados)
    acados += ["--pulse-width-max-step-us", "100", "--pulse-width-slew-formulation", "lifting"]
    add("A4", "A3", "Hard 100 us PW increment bound", acados, ["Native reduction/lifting compatibility smoke"])
    acados += ["--pulse-width-slew-weight", "0.01", "--pulse-width-slew-reference-us", "100"]
    add("A5", "A4", "Normalized PW increment penalty, weight 0.01", acados)
    return {"schema_version": 1, "status": "proposal_not_executed", "cpu_set": cpus,
            "allocation_verified": False, "environment": {
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "NUMERIC_THREADS": "1", "BENCHMARK_THREADS": "1", "CMAKE_BUILD_PARALLEL_LEVEL": "1"},
            "protocol": str(ROOT / "docs/cycling_solver_benchmark/incremental_ablation_protocol_20260921.md"),
            "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cpus", required=True, help="Coordinator-verified allocation; does not reserve cores")
    parser.add_argument("--python", required=True, help="Absolute rho32 interpreter path")
    parser.add_argument("--n-windows", type=int, default=100)
    args = parser.parse_args()
    if args.n_windows < 1:
        parser.error("n-windows must be positive")
    print(json.dumps(build_manifest(args.output_root.resolve(), args.cpus, args.python, args.n_windows), indent=2))


if __name__ == "__main__":
    main()
