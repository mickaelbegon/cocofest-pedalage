"""Central compatibility rules reflecting the actually connected engines.

Validation is pure: runtime availability, seed contents/provenance and weight
files are checked again by the scientific runners before any optimization.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .config import SimulationConfig


@dataclass(frozen=True)
class ValidationIssue:
    field: str
    message: str
    code: str = "invalid"


class ConfigurationError(ValueError):
    def __init__(self, issues):
        self.issues = tuple(issues)
        super().__init__("; ".join(f"{i.field}: {i.message}" for i in self.issues))


class CapabilityRegistry:
    """One set of restrictions for interactive forms, saved JSON and CLI plans."""

    CHOICES = {
        "mode": ("rho", "rho-physio", "rho-pace", "fho"),
        "solver": ("ipopt", "madnlp", "fatrop", "acados"),
        "mechanics": ("reduced", "full"),
        "formulation": ("dynamic", "isokinetic"),
        "integration": ("radau", "irk"),
        "reduced_internal_crank_velocity_guard": ("auto", "on", "off"),
        "acados_qp_solver": (
            "auto", "PARTIAL_CONDENSING_HPIPM", "FULL_CONDENSING_HPIPM",
            "FULL_CONDENSING_QPOASES",
        ),
        "ipopt_linear_solver": ("mumps", "ma27", "ma57", "ma77", "ma86", "ma97", "pardiso"),
        "madnlp_linear_solver": ("mumps", "ma27", "ma57", "ma86", "ma97"),
    }
    FIELD_HELP = {
        "formulation": "Dynamic: free crank dynamics; isokinetic: prescribed angular velocity.",
        "pulse_width_max_step_us": "Hard ΔPW bound between successive controls, including executed RHO boundaries (µs).",
        "pulse_width_slew_weight": "Weight of mean squared normalized intra-window ΔPW; requires a hard bound. Executed RHO seams are only bounded.",
        "pulse_width_slew_reference_us": "Positive reference for ΔPW normalization (µs), independent of the hard bound.",
        "signed_crank_torque": "Signed crank torque (N.m): positive resists negative angular velocity.",
        "bilateral_reduced": "Use the bilateral Wu bioMod to build a reduced profile from two physical arm chains and one shared crank; the online OCP remains theta/omega only.",
        "acados_ipopt_cycle1_seed": "Required ACADOS initialization: certified IPOPT solution of exactly cycle 1, matching the target problem.",
        "reduced_internal_crank_velocity_guard": "Internal reduced-mechanics cadence guard: auto preserves legacy behavior; exact ACADOS seed transfers use on.",
        "acados_qp_solver": "ACADOS QP backend: auto uses full-condensing HPIPM with a ΔPW bound and partial-condensing HPIPM otherwise.",
        "acados_ding_local_reduction": "Experimental local Ding reconstruction: ACADOS retains F and A as NLP states and reconstructs Cn, Tau1 and Km from the fixed initial state and the current pulse-width profile. It requires SQP IRK Gauss-Legendre 4×5, reduced dynamic mechanics, 50 stimulations/cycle, an active cadence guard and no slew constraint. Fixed initial states are checked by the engine; full NLP duals are not reconstructed.",
        "ipopt_ding_local_reduction": "Experimental IPOPT local Ding reconstruction: F and A remain decision states while Cn, Tau1 and Km are reconstructed with the original discrete Radau-5 operator. It requires the interpreted SX IPOPT path, one-cycle dynamic reduced RHO windows, 30 stimulations/cycle and no pulse-width slew constraint. The complete NLP is audited after each solve.",
        "cycles": "Number of executed cycles; FHO optimizes these cycles in one window.",
        "threads": "Solver worker threads; independent from numerical library threads.",
        "numeric_threads": "Threads per numerical library (BLAS/OpenMP).",
    }
    MANAGED_OPTIONS = frozenset({
        "--solvers", "--mechanical-formulation", "--formulation", "--stimulations-per-cycle",
        "--n-windows", "--cycles-per-window", "--single-shot", "--n-threads",
        "--signed-crank-torque", "--resistive-torque", "--crank-assistance",
        "--ipopt-linear-solver", "--warmup-ipopt-linear-solver", "--madnlp-linear-solver",
        "--terminal-wheel-q-slack", "--acados-terminal-wheel-q-slack", "--output-json",
        "--common-initial-solution", "--adopt-common-initial-solution-warmup-cycles",
        "--acados-disable-standard-ipopt-warmup", "--pulse-width-max-step-us",
        "--pulse-width-slew-weight", "--pulse-width-slew-reference-us",
        "--ipopt-ode-solver", "--ipopt-collocation-degree", "--ipopt-collocation-method",
        "--ipopt-c-compile", "--madnlp-c-compile", "--acados-integrator-type",
        "--acados-sim-stages", "--acados-sim-steps", "--energy-equivalent-torque",
        "--reduced-internal-crank-velocity-guard",
        "--acados-qp-solver",
        "--acados-ding-local-reduction", "--bilateral-reduced",
        "--isokinetic-omega", "--load-torque-min", "--load-torque-max",
        "--objective", "--objective-shape", "--compact-rho-output",
        "--pace-config", "--pace-journal", "--model-config", "--condition",
    })

    @classmethod
    def choices(cls, field: str) -> tuple[str, ...]:
        return cls.CHOICES.get(field, ())

    @classmethod
    def validate(cls, config: SimulationConfig) -> tuple[ValidationIssue, ...]:
        issues = []
        def issue(field, message, code="invalid"):
            issues.append(ValidationIssue(field, message, code))
        def finite(field):
            value = getattr(config, field)
            valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            if not valid:
                issue(field, "must be a finite number")
            return valid
        if type(config.schema_version) is not int or config.schema_version != 1:
            issue("schema_version", "must be 1")
        for field, choices in cls.CHOICES.items():
            if getattr(config, field) not in choices:
                issue(field, f"must be one of {', '.join(choices)}")
        for field in ("cycles", "cycles_per_window", "threads", "numeric_threads", "stimulations_per_cycle",
                      "collocation_degree", "acados_sim_stages", "acados_sim_steps", "madnlp_hot_max_iterations"):
            value = getattr(config, field)
            if type(value) is not int or value < 1:
                issue(field, "must be a positive integer")
        for field in ("compile_evaluators", "madnlp_recovery", "dry_run", "acados_ding_local_reduction",
                      "ipopt_ding_local_reduction",
                      "bilateral_reduced"):
            if type(getattr(config, field)) is not bool:
                issue(field, "must be a boolean")
        for field in ("signed_crank_torque", "terminal_q_slack", "isokinetic_omega",
                      "energy_equivalent_torque", "load_torque_min", "load_torque_max", "madnlp_hot_max_wall_time"):
            if not finite(field):
                continue
            value = getattr(config, field)
            if field in ("terminal_q_slack", "energy_equivalent_torque") and value < 0:
                issue(field, "must be nonnegative")
            if field in ("isokinetic_omega", "load_torque_min") and value >= 0:
                issue(field, "must be strictly negative")
            if field in ("load_torque_max", "madnlp_hot_max_wall_time") and value <= 0:
                issue(field, "must be strictly positive")
        if finite("pulse_width_slew_weight"):
            if config.pulse_width_slew_weight < 0:
                issue("pulse_width_slew_weight", "must be nonnegative")
            elif config.pulse_width_slew_weight > 0:
                if config.pulse_width_max_step_us is None:
                    issue("pulse_width_slew_weight", "requires a hard pulse_width_max_step_us bound")
                if type(config.stimulations_per_cycle) is int and config.stimulations_per_cycle < 2:
                    issue("pulse_width_slew_weight", "requires at least two controls per cycle")
        if finite("pulse_width_slew_reference_us") and config.pulse_width_slew_reference_us <= 0:
            issue("pulse_width_slew_reference_us", "must be strictly positive")
        if config.pulse_width_max_step_us is not None:
            if finite("pulse_width_max_step_us") and config.pulse_width_max_step_us <= 0:
                issue("pulse_width_max_step_us", "must be positive or None")
            if config.mechanics != "reduced":
                issue("pulse_width_max_step_us", "currently requires reduced mechanics", "unsupported")
            window = config.cycles if config.mode == "fho" else config.cycles_per_window
            if window != 1:
                issue("pulse_width_max_step_us", "currently requires one-cycle windows", "unsupported")
        bounds = (config.load_torque_min, config.energy_equivalent_torque, config.load_torque_max)
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in bounds):
            if not bounds[0] <= bounds[1] <= bounds[2]:
                issue("energy_equivalent_torque", "must lie inside the load-torque bounds")
        for field in ("output_root", "acados_ipopt_cycle1_seed", "weights_config", "model_config"):
            value = getattr(config, field)
            if value is None and field != "output_root":
                continue
            if not isinstance(value, str) or not value.strip() or "\0" in value:
                issue(field, "must be a nonempty path without NUL")
        extras = config.extra_arguments
        if not isinstance(extras, (tuple, list)) or any(not isinstance(x, str) or "\0" in x for x in extras):
            issue("extra_arguments", "must be a list of strings without NUL")
        elif any(x.startswith("--") and any(option.startswith(x.split("=", 1)[0])
                                             for option in cls.MANAGED_OPTIONS) for x in extras):
            issue("extra_arguments", "set managed options in the configuration")
        if config.formulation == "isokinetic" and config.mechanics != "reduced":
            issue("formulation", "isokinetic currently requires reduced mechanics", "unsupported")
        if config.bilateral_reduced:
            if config.mechanics != "reduced":
                issue("bilateral_reduced", "requires reduced mechanics", "unsupported")
            if config.formulation != "dynamic":
                issue("bilateral_reduced", "currently requires dynamic free-crank mechanics", "unsupported")
        if config.solver == "acados" and not config.acados_ipopt_cycle1_seed:
            issue("acados_ipopt_cycle1_seed", "ACADOS requires the exact IPOPT cycle-1 OCP seed", "required")
        if config.solver == "acados" and config.integration != "irk":
            issue("integration", "ACADOS uses IRK; select irk", "unsupported")
        if config.solver == "acados" and config.mode == "fho":
            issue("mode", "FHO is currently connected only to IPOPT and MadNLP; ACADOS requires a cycle-1 seed", "unsupported")
        if config.solver == "acados" and config.cycles_per_window != 1:
            issue("cycles_per_window", "ACADOS seed metadata currently requires one-cycle windows", "unsupported")
        if config.acados_ding_local_reduction:
            requirements = {"solver": "acados", "mode": "rho", "mechanics": "reduced",
                            "formulation": "dynamic", "integration": "irk",
                            "acados_sim_stages": 4, "acados_sim_steps": 5,
                            "stimulations_per_cycle": 50, "pulse_width_max_step_us": None}
            for field, expected in requirements.items():
                if getattr(config, field) != expected:
                    issue(field, f"Réduction Ding expérimentale : exige {expected!r} (ACADOS IRK Gauss-Legendre 4×5).", "unsupported")
            if config.reduced_internal_crank_velocity_guard == "off":
                issue("reduced_internal_crank_velocity_guard", "Réduction Ding expérimentale : guard vitesse actif requis (on ou auto).", "unsupported")
            if isinstance(extras, (tuple, list)):
                blocked = ("--acados-initial-irk-rollout", "--acados-transfer-irk-rollout")
                for argument in extras:
                    if isinstance(argument, str) and argument.startswith("--"):
                        option = argument.split("=", 1)[0]
                        if any(name.startswith(option) for name in blocked):
                            issue("extra_arguments", "Réduction Ding expérimentale : les rollouts IRK initial/transfert ne sont pas adaptés au proxy réduit.", "unsupported")
                        if "--acados-nlp-solver-type".startswith(option):
                            issue("extra_arguments", "Réduction Ding expérimentale : le GUI fixe le solveur ACADOS à SQP.", "unsupported")
            if config.bilateral_reduced:
                issue(
                    "acados_ding_local_reduction",
                    "the experimental local Ding reduction is not yet certified for bilateral muscles",
                    "unsupported",
                )
        if config.ipopt_ding_local_reduction:
            requirements = {
                "solver": "ipopt", "mode": "rho", "mechanics": "reduced",
                "formulation": "dynamic", "integration": "radau", "collocation_degree": 5,
                "cycles_per_window": 1, "stimulations_per_cycle": 30,
                "pulse_width_max_step_us": None, "pulse_width_slew_weight": 0.0,
                "compile_evaluators": False, "model_config": None,
            }
            for field, expected in requirements.items():
                if getattr(config, field) != expected:
                    issue(field, f"Réduction Ding IPOPT expérimentale : exige {expected!r}.", "unsupported")
            if config.reduced_internal_crank_velocity_guard != "off":
                issue(
                    "reduced_internal_crank_velocity_guard",
                    "Réduction Ding IPOPT expérimentale : le pilote validé exige le guard vitesse interne désactivé.",
                    "unsupported",
                )
        if config.mode in ("rho-physio", "rho-pace"):
            requirements = {"solver": "ipopt", "mechanics": "reduced", "formulation": "dynamic",
                            "cycles_per_window": 1, "compile_evaluators": False}
            for field, expected in requirements.items():
                if getattr(config, field) != expected:
                    issue(field, f"{config.mode} currently requires {expected!r}", "unsupported")
            if not config.weights_config:
                issue("weights_config", "weighted RHO requires weights and their provenance", "required")
            if isinstance(config.signed_crank_torque, (int, float)) and config.signed_crank_torque <= 0:
                issue("signed_crank_torque", "weighted RHO requires positive resistance", "unsupported")
            if type(config.cycles) is int and config.cycles > 2000:
                issue("cycles", "weighted RHO is limited to 2000 cycles", "unsupported")
        if config.model_config:
            for field, expected in {"mechanics": "reduced", "formulation": "dynamic"}.items():
                if getattr(config, field) != expected:
                    issue(field, f"configured models currently require {expected!r}", "unsupported")
            if config.mode != "fho" and config.cycles_per_window != 1:
                issue("cycles_per_window", "configured RHO requires 1", "unsupported")
            if type(config.cycles) is int and config.cycles > 2000:
                issue("cycles", "configured models are limited to 2000 cycles", "unsupported")
            if isinstance(config.signed_crank_torque, (int, float)) and config.signed_crank_torque <= 0:
                issue("signed_crank_torque", "configured models require positive resistance", "unsupported")
        if (type(config.cycles) is int and type(config.cycles_per_window) is int
                and config.mode != "fho" and config.cycles_per_window > config.cycles):
            issue("cycles_per_window", "must not exceed executed cycles")
        return tuple(issues)

    @classmethod
    def require_valid(cls, config: SimulationConfig) -> SimulationConfig:
        issues = cls.validate(config)
        if issues:
            raise ConfigurationError(issues)
        return config

    @classmethod
    def supports(cls, config: SimulationConfig) -> bool:
        return not cls.validate(config)
