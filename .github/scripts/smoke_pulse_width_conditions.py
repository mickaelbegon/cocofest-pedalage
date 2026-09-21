#!/usr/bin/env python3
"""Build six real cycling OCPs and export Acados constraints, without solving.

This bounded smoke check deliberately stops after prepare_nmpc: no warmup or
RHO campaign is run. The JSON records actual model dimensions and options.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as benchmark


class ModelBuilt(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "pulse-width-conditions-smoke/build-audit.json")
    parser.add_argument("--profile", type=Path, default=ROOT / "resistance-fho-pilots-20260910/seed-0p10/reduced-cycling-fourier12.npz")
    args = parser.parse_args()
    real_prepare = benchmark.prepare_nmpc
    rows = []
    for frequency, bound in ((50, None), (30, 100), (50, 100)):
        for solver in ("ipopt", "acados"):
            captured = {}
            def capture(*values, **kwargs):
                captured["nmpc"] = real_prepare(*values, **kwargs)
                raise ModelBuilt()
            argv = ["--solver", solver, "--single-shot", "--n-windows", "1", "--cycles-per-window", "1",
                    "--mechanical-formulation", "reduced", "--model-formulation", "periodic_node",
                    "--formulation", "dynamic", "--torque-application", "constant", "--constant-crank-torque", "0.1",
                    "--n-threads", "1", "--stimulations-per-cycle", str(frequency),
                    "--reduced-cycling-profile", str(args.profile), "--disable-standard-ipopt-warmup",
                    "--ipopt-linear-solver", "ma57", "--use-sx"]
            argv += (["--ode-solver", "collocation", "--collocation-degree", "5", "--collocation-method", "radau"]
                     if solver == "ipopt" else ["--ode-solver", "rk4", "--experimental-reduced-acados", "--acados-integrator-type", "IRK"])
            if solver == "acados":
                # solve_case validates the mandatory cycle-1 seed policy before
                # prepare_nmpc. The construction-only capture stops inside
                # prepare_nmpc, so this sentinel is declared but never read.
                argv += [
                    "--common-initial-solution",
                    str(ROOT / "construction-only-certified-cycle1-sentinel.npz"),
                ]
            if bound is not None:
                argv += ["--pulse-width-max-step-us", str(bound)]
            config = benchmark.build_argument_parser().parse_args(argv)
            start = perf_counter()
            with patch.object(benchmark, "prepare_nmpc", capture):
                try:
                    benchmark.solve_case(config, echo=False)
                except ModelBuilt:
                    pass
            nmpc = captured["nmpc"]
            nlp = nmpc.nlp[0]
            row = {"solver": solver, "stimulations_per_cycle": frequency, "pulse_width_max_step_us": bound,
                   "signed_crank_torque_nm": nlp.model.external_crank_torque, "mechanical_formulation": "reduced",
                   "n_threads": config.n_threads, "ode_solver": config.ode_solver,
                   "collocation_degree": config.collocation_degree if solver == "ipopt" else None,
                   "acados_integrator_type": config.acados_integrator_type if solver == "acados" else None,
                   "ipopt_linear_solver": config.ipopt_linear_solver if solver == "ipopt" else None,
                   "state_dimension": nlp.states.shape, "control_dimension": nlp.controls.shape,
                   "shooting_intervals": nlp.ns, "stimulation_interval_s": config.calcium_stimulation_interval_s,
                   "state_keys": list(nlp.states.keys()), "control_keys": list(nlp.controls.keys()),
                   "built": True, "solved": False, "arguments": argv}
            if solver == "acados":
                from bioptim import Solver
                from bioptim.interfaces.acados_interface import AcadosInterface
                benchmark.patch_bioptim_acados_interface()
                options = Solver.ACADOS()
                options.set_integrator_type("IRK")
                interface = AcadosInterface(nmpc, options)
                interface._AcadosInterface__set_constraints(nmpc)
                row["acados_constraint_dimensions"] = {
                    "initial": interface.acados_model.con_h_expr_0.shape[0],
                    "path": interface.acados_model.con_h_expr.shape[0],
                    "terminal": interface.acados_model.con_h_expr_e.shape[0],
                }
            row["wall_time_s"] = perf_counter() - start
            rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"kind": "construction_only_no_campaign", "cases": rows}, indent=2) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "cases_built": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
