"""Pure subprocess plans: no scientific imports, environment discovery or writes."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shlex

from .config import SimulationConfig
from .capabilities import CapabilityRegistry

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class LaunchPlan:
    argv: tuple[str, ...]
    cwd: Path
    environment_updates: dict[str, str]
    suite: str
    result_json: Path | None = None

    @property
    def command(self) -> str:
        """Shell-quoted display only; execute argv with shell=False."""
        return shlex.join(self.argv)


def build_launch_plan(config: SimulationConfig | dict, prefix: Path, root: Path = ROOT) -> LaunchPlan:
    """Build and validate without discovering runtimes or touching the filesystem."""
    legacy_dictionary = not isinstance(config, SimulationConfig)
    if legacy_dictionary:
        data = dict(config)
        # The historical dictionary had no integration selector.
        if data.get("solver") == "acados" and "integration" not in data:
            data["integration"] = "irk"
        config = SimulationConfig.from_dict(data)
    CapabilityRegistry.require_valid(config)
    typed_config = config
    config = config.to_dict()
    prefix, root = Path(prefix).expanduser().absolute(), Path(root).expanduser().absolute()
    solver = config["solver"]
    max_step_us = config["pulse_width_max_step_us"]
    acados_seed = config["acados_ipopt_cycle1_seed"]
    if acados_seed and not legacy_dictionary:
        seed_path = Path(acados_seed).expanduser()
        acados_seed = str(seed_path if seed_path.is_absolute() else root / seed_path)
    backend = config["ipopt_linear_solver"]
    extras = config["extra_arguments"]
    suite = "madnlp32" if solver == "madnlp" else "rho32"
    configured_extras = list(extras)
    produces_cycle1_seed = any(
        argument == "--common-initial-solution-output"
        or argument.startswith("--common-initial-solution-output=")
        for argument in configured_extras
    )
    guard_mode = config["reduced_internal_crank_velocity_guard"]
    if (
        guard_mode == "auto"
        and config["mechanics"] == "reduced"
        and config["formulation"] == "dynamic"
        and (solver == "acados" or produces_cycle1_seed)
    ):
        # A cycle-1 IPOPT producer and its ACADOS consumer must describe the
        # same target constraint set. Standalone runs retain legacy ``auto``.
        guard_mode = "on"
    configured_extras.extend(
        ["--reduced-internal-crank-velocity-guard", guard_mode]
    )
    configured_extras.extend(["--acados-qp-solver", config["acados_qp_solver"]])
    if config["bilateral_reduced"]:
        configured_extras.append("--bilateral-reduced")
    configured_extras.extend([
        "--pulse-width-slew-weight", str(float(config["pulse_width_slew_weight"])),
        "--pulse-width-slew-reference-us", str(float(config["pulse_width_slew_reference_us"])),
    ])
    if max_step_us is not None:
        configured_extras.extend(
            ["--pulse-width-max-step-us", str(float(max_step_us))]
        )
    output = Path(config["output_root"]).expanduser()
    if not output.is_absolute():
        output = root / output
    updates = {
        "BENCHMARK_THREADS": str(config["threads"]), "BENCHMARK_CYCLES_PER_WINDOW": str(config["cycles"] if config["mode"] == "fho" else config["cycles_per_window"]), "NUMERIC_THREADS": str(config["numeric_threads"]),
        "BENCHMARK_ASSISTANCE": "0", "BENCHMARK_Q_SLACK": str(config["terminal_q_slack"]),
        "BENCHMARK_MAX_ITER": "2000", "BENCHMARK_FORMULATION": config["formulation"],
        "BENCHMARK_SIGNED_CRANK_TORQUE": str(config["signed_crank_torque"]),
        "BENCHMARK_STIMULATIONS_PER_CYCLE": str(config["stimulations_per_cycle"]),
        "IPOPT_LINEAR_SOLVER": backend, "WARMUP_IPOPT_LINEAR_SOLVER": backend,
        "PYTHON_EXECUTABLE": str(prefix / "bin/python"),
        "GITHUB_WORKSPACE": str(root),
    }
    if config["ipopt_ding_local_reduction"]:
        updates["BENCHMARK_IPOPT_DING_LOCAL_REDUCTION"] = "true"
    updates.update({
        "OMP_NUM_THREADS": str(config["numeric_threads"]),
        "OPENBLAS_NUM_THREADS": str(config["numeric_threads"]),
        "MKL_NUM_THREADS": str(config["numeric_threads"]),
        "OMP_THREAD_LIMIT": str(config["numeric_threads"]),
        "BLIS_NUM_THREADS": str(config["numeric_threads"]),
        "NUMEXPR_NUM_THREADS": str(config["numeric_threads"]),
        "JULIA_NUM_THREADS": str(config["numeric_threads"]),
        "BENCHMARK_ISOKINETIC_OMEGA": str(config["isokinetic_omega"]),
        "BENCHMARK_ENERGY_EQUIVALENT_TORQUE": str(config["energy_equivalent_torque"]),
        "BENCHMARK_LOAD_TORQUE_MIN": str(config["load_torque_min"]),
        "BENCHMARK_LOAD_TORQUE_MAX": str(config["load_torque_max"]),
    })
    if config["mode"] in ("rho-physio", "rho-pace") or config["model_config"]:
        return _build_adapted_plan(typed_config, prefix, root, output, updates, suite)
    if config["mode"] == "fho":
        configured_extras.append("--single-shot")
    if solver != "acados":
        slug = f"{solver}-pycharm" if config["mode"] == "rho" else f"{solver}-{config['mode']}"
        if solver == "madnlp" and config["madnlp_recovery"]:
            slug += "-fatigue-endurance"
        updates.update({
            "MADNLP_FAST_MAX_ITERATIONS": str(config["madnlp_hot_max_iterations"]),
            "MADNLP_FAST_MAX_WALL_TIME": str(config["madnlp_hot_max_wall_time"]),
            "MADNLP_FIRST_MAX_ITERATIONS": "2000",
            "BENCHMARK_EXTRA_ARGUMENTS_JSON": json.dumps(configured_extras),
        })
        command = ["bash", str(root / ".github/scripts/run_cycling_benchmark_case.sh"),
                   slug, solver, config["mechanics"], backend if solver == "ipopt" else config["madnlp_linear_solver"],
                   "collocation" if config["integration"] == "radau" else "irk", str(output), str(config["cycles"]),
                   str(bool(config["compile_evaluators"])).lower(), "sx", "none",
                   str(config["collocation_degree"]), "periodic_collocation", "auto", "auto"]
        cwd = root
        directory = f"{slug}-{config['mechanics']}"
        if config["formulation"] == "isokinetic":
            torque, omega, low, high = (format(float(config[name]), ".12g") for name in (
                "energy_equivalent_torque", "isokinetic_omega", "load_torque_min", "load_torque_max"))
            directory += f"-isokinetic-torque-{torque}-omega-{omega}-load-{low}-to-{high}"
        result_json = output / directory / "result.json"
    else:
        case = output / ("acados-pycharm" if config["mode"] == "rho" else f"acados-{config['mode']}")
        cwd = case / "codegen"
        command = [str(prefix / "bin/python"),
                   str(root / "examples/fes_multibody/cycling/cycling_fes_solver_comparison.py"),
                   "--solvers", "acados", "--objective", "fatigue", "--ipopt-use-sx",
                   "--ipopt-linear-solver", backend, "--warmup-ipopt-linear-solver", backend,
                   "--cycles-per-window", str(config["cycles"] if config["mode"] == "fho" else config["cycles_per_window"]), "--n-windows", str(config["cycles"]),
                   "--stimulations-per-cycle", str(config["stimulations_per_cycle"]),
                   "--n-threads", str(config["threads"]),
                   "--signed-crank-torque", str(config["signed_crank_torque"]),
                   "--terminal-wheel-q-slack", str(config["terminal_q_slack"]),
                   "--acados-dir", str(prefix), "--acados-nlp-solver-type", "SQP",
                   "--acados-integrator-type", "IRK", "--acados-sim-stages", str(config["acados_sim_stages"]),
                   "--acados-sim-steps", str(config["acados_sim_steps"]), "--acados-max-iter", "100",
                   "--mechanical-formulation", config["mechanics"], "--compact-rho-output",
                   "--output-json", str(case / "result.json")]
        if config["mechanics"] == "reduced":
            command.append("--experimental-reduced-acados")
        if config["acados_ding_local_reduction"]:
            command.append("--acados-ding-local-reduction")
        command.extend([
            "--common-initial-solution", str(acados_seed),
            "--adopt-common-initial-solution-warmup-cycles",
            "--acados-disable-standard-ipopt-warmup",
        ])
        command.extend(["--formulation", config["formulation"],
                        "--isokinetic-omega", str(config["isokinetic_omega"]),
                        "--energy-equivalent-torque", str(config["energy_equivalent_torque"]),
                        "--load-torque-min", str(config["load_torque_min"]),
                        "--load-torque-max", str(config["load_torque_max"])])
        command.extend(configured_extras)
        result_json = case / "result.json"
    return LaunchPlan(tuple(command), cwd, updates, suite, result_json)


def _build_adapted_plan(config, prefix, root, output, updates, suite):
    """Connect validated weighted/model adapters without loading their engines."""
    case = output / f"{config.solver}-{config.mode}-{config.mechanics}"
    def path(value):
        candidate = Path(value).expanduser()
        return str(candidate if candidate.is_absolute() else root / candidate)
    benchmark = [
        "--solvers", config.solver, "--objective", "fatigue", "--objective-shape", "quadratic",
        "--ipopt-profile", "periodic_collocation", "--ipopt-use-sx",
        "--ipopt-linear-solver", config.ipopt_linear_solver,
        "--warmup-ipopt-linear-solver", config.ipopt_linear_solver,
        "--ipopt-ode-solver", "collocation" if config.integration == "radau" else "irk",
        "--ipopt-collocation-degree", str(config.collocation_degree),
        "--ipopt-collocation-method", "radau", "--ipopt-enforce-start-constraints",
        "--ipopt-disable-historical-initial-guess",
        "--cycles-per-window", str(config.cycles if config.mode == "fho" else config.cycles_per_window),
        "--n-windows", str(config.cycles), "--n-threads", str(config.threads),
        "--stimulations-per-cycle", str(config.stimulations_per_cycle),
        "--signed-crank-torque", str(config.signed_crank_torque),
        "--terminal-wheel-q-slack", str(config.terminal_q_slack),
        "--mechanical-formulation", config.mechanics, "--formulation", config.formulation,
        "--acados-qp-solver", config.acados_qp_solver,
        "--madnlp-linear-solver", config.madnlp_linear_solver,
        "--state-scaling", "full", "--compact-rho-output",
        "--output-json", str(case / "result.json"),
    ]
    if config.mode == "fho":
        benchmark.append("--single-shot")
    if config.compile_evaluators:
        benchmark.append("--ipopt-c-compile")
    if config.pulse_width_max_step_us is not None:
        benchmark += ["--pulse-width-max-step-us", str(float(config.pulse_width_max_step_us))]
    if config.bilateral_reduced:
        benchmark.append("--bilateral-reduced")
    benchmark += [
        "--pulse-width-slew-weight", str(float(config.pulse_width_slew_weight)),
        "--pulse-width-slew-reference-us", str(float(config.pulse_width_slew_reference_us)),
    ]
    if config.solver == "acados":
        # The configured runner enforces the same certified one-cycle IPOPT
        # transfer as its direct CLI. Keep this explicit rather than relying
        # on ACADOS' generic warmup, whose unilateral state dimension is not
        # compatible with the bilateral reduced model.
        benchmark += [
            "--experimental-reduced-acados",
            "--acados-dir", str(prefix),
            "--acados-nlp-solver-type", "SQP_RTI",
            "--acados-integrator-type", "IRK",
            "--acados-sim-stages", str(config.acados_sim_stages),
            "--acados-sim-steps", str(config.acados_sim_steps),
            "--acados-max-iter", "100",
            "--common-initial-solution", path(config.acados_ipopt_cycle1_seed),
            "--adopt-common-initial-solution-warmup-cycles",
            "--acados-disable-standard-ipopt-warmup",
        ]
    benchmark.extend(config.extra_arguments)
    weighted = config.mode in ("rho-physio", "rho-pace")
    if config.model_config:
        command = [str(prefix / "bin/python"), str(root / "scripts/run_configured_cycling_benchmark.py"),
                   "--model-config", path(config.model_config), "--condition", config.mode]
        if weighted:
            command += ["--weights-config", path(config.weights_config),
                        "--weights-journal", str(case / "weights.jsonl")]
    else:
        command = [str(prefix / "bin/python"), str(root / "cocofest/simulation/weighted_runner.py"),
                   "--mode", config.mode, "--pace-config", path(config.weights_config),
                   "--pace-journal", str(case / "weights.jsonl")]
    command += ["--", *benchmark]
    return LaunchPlan(tuple(command), case / "codegen", updates, suite, case / "result.json")
