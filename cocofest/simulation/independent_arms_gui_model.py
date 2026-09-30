"""Pure GUI-facing contracts for independent isokinetic arm simulations.

This is intentionally separate from :class:`SimulationConfig`.  A bilateral
reduced model has one crank and is a different scientific problem: this model
describes two unconnected, unilateral optimisation problems sharing only a
clock.  In particular, a target *work* is not an instantaneous load law.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Any, Mapping


_TAU = 2.0 * math.pi
PROCESS_FACTORY = "cocofest.simulation.independent_arms_process:build_process_independent_arms"


def independent_solver_choices(factory: str | None) -> tuple[str, ...]:
    """Advertise only the solvers connected to the selected runtime factory."""
    return ("ipopt",) if factory == PROCESS_FACTORY else ("ipopt", "acados")


@dataclass(frozen=True)
class IndependentArmsGuiConfig:
    """Serializable request submitted by the desktop GUI to the coordinator.

    ``*_work_j_per_cycle`` is positive dissipated/mechanical work magnitude.
    ``*_equivalent_mean_torque_nm`` is its full-turn equivalent, namely
    ``work / (2*pi)``.  Both may be supplied, but then must agree.  The
    coordinator, not the GUI, derives the instantaneous isokinetic balance.
    ``*_runner_config`` is an opaque path/configuration reference owned by the
    scientific runtime, permitting two distinct arm models without coupling
    them here.
    """

    schema_version: int = 1
    formulation: str = "isokinetic"
    solver: str = "ipopt"
    cycles: int = 100
    cycles_per_window: int = 1
    stimulations_per_cycle: int = 30
    isokinetic_omega: float = -2.0 * math.pi
    right_work_j_per_cycle: float | None = None
    left_work_j_per_cycle: float | None = None
    right_equivalent_mean_torque_nm: float | None = 0.1
    left_equivalent_mean_torque_nm: float | None = 0.1
    factory: str | None = None
    right_runner_config: str | None = None
    left_runner_config: str | None = None
    output_root: str = "gui-results/independent-arms"
    parallel: bool = True
    reuse_compiled_artifacts: bool = True
    parametric_fatigue_weights: bool = False
    muscle_weight_policy: str = "unit"
    physio_update_every_cycles: int = 20
    physio_update_smoothing: float = 0.2
    physio_update_max_log_step: float = math.log(1.1)
    physio_update_deadband_log: float = math.log(1.01)
    physio_update_min_relative_weight: float = 0.25
    physio_update_max_relative_weight: float = 4.0
    right_solver_cpu: int | None = None
    left_solver_cpu: int | None = None
    resistance_pace_policy: str = "fixed"
    resistance_pace_initial_split_policy: str = "manual"
    resistance_pace_update_every_cycles: int = 10
    resistance_pace_capacity_gain: float = 1.0
    resistance_pace_smoothing: float = 0.25
    resistance_pace_max_fraction_step: float = 0.10
    resistance_pace_minimum_arm_torque_nm: float = 0.0

    def __post_init__(self):
        for field in ("factory", "right_runner_config", "left_runner_config"):
            value = getattr(self, field)
            if value is not None:
                if not isinstance(value, str) or not value.strip() or "\0" in value:
                    raise ValueError(f"{field} must be a nonempty string or null")

    @staticmethod
    def _target(work: float | None, torque: float | None, side: str) -> float:
        if work is None and torque is None:
            raise ValueError(f"{side}: indiquez un travail cible ou un couple moyen équivalent.")
        def finite_positive(value, label):
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{side}: {label} doit être un nombre fini strictement positif.")
            return float(value)
        if work is not None:
            work = finite_positive(work, "travail par cycle")
        if torque is not None:
            torque = finite_positive(torque, "couple moyen équivalent")
        inferred = torque * _TAU if work is None else work
        if work is not None and torque is not None and not math.isclose(work, torque * _TAU, rel_tol=1e-9, abs_tol=1e-10):
            raise ValueError(
                f"{side}: travail et couple ne concordent pas (W = 2π·τ sur un tour complet)."
            )
        return inferred

    def validate(self) -> "IndependentArmsGuiConfig":
        if self.schema_version != 1:
            raise ValueError("Version de requête indépendante non prise en charge.")
        if self.formulation != "isokinetic":
            raise ValueError("Les deux bras indépendants sont disponibles uniquement en isocinétique.")
        if self.solver not in independent_solver_choices(self.factory):
            if self.factory == PROCESS_FACTORY:
                raise ValueError("La fabrique deux-bras par défaut prend en charge seulement IPOPT/MA57 ; les mises à jour ACADOS ne sont pas validées.")
            raise ValueError("Le solveur indépendant doit être IPOPT ou ACADOS.")
        for name in ("cycles", "cycles_per_window", "stimulations_per_cycle"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} doit être un entier strictement positif.")
        if self.cycles_per_window > self.cycles:
            raise ValueError("La fenêtre RHO ne peut excéder le nombre de cycles.")
        if self.factory == PROCESS_FACTORY:
            if self.cycles_per_window != 1:
                raise ValueError("La fabrique deux-bras par défaut exige des fenêtres d'un cycle.")
            if self.parallel is not True:
                raise ValueError("La fabrique deux-bras par défaut exige une exécution parallèle.")
            if self.right_runner_config is not None or self.left_runner_config is not None:
                raise ValueError("La fabrique deux-bras par défaut ne prend pas en charge les références de configuration worker ; utilisez une fabrique personnalisée.")
        if (not isinstance(self.isokinetic_omega, (int, float)) or isinstance(self.isokinetic_omega, bool)
                or not math.isfinite(self.isokinetic_omega) or self.isokinetic_omega >= 0):
            raise ValueError("La vitesse isocinétique doit être finie et négative (rad/s).")
        if any(type(getattr(self, name)) is not bool for name in (
            "parallel", "reuse_compiled_artifacts", "parametric_fatigue_weights",
        )):
            raise ValueError(
                "parallel, reuse_compiled_artifacts et parametric_fatigue_weights doivent être booléens."
            )
        assigned_cpus = (self.right_solver_cpu, self.left_solver_cpu)
        if (assigned_cpus[0] is None) != (assigned_cpus[1] is None):
            raise ValueError("Sélectionnez soit les deux CPU solveur, soit le mode automatique pour les deux bras.")
        if any(type(cpu) is not int or cpu < 0 for cpu in assigned_cpus if cpu is not None):
            raise ValueError("Les CPU solveur doivent être des entiers non négatifs.")
        if assigned_cpus[0] is not None and assigned_cpus[0] == assigned_cpus[1]:
            raise ValueError("Les bras droit et gauche doivent utiliser deux CPU solveur distincts.")
        if self.resistance_pace_policy not in ("fixed", "capacity_feedback"):
            raise ValueError("La politique de répartition doit être 'fixed' ou 'capacity_feedback'.")
        if self.resistance_pace_initial_split_policy not in ("manual", "capacity_fatigability_after_first_cycle"):
            raise ValueError("La politique initiale D/G est inconnue.")
        if (self.resistance_pace_initial_split_policy == "capacity_fatigability_after_first_cycle"
                and self.cycles < 2):
            raise ValueError("Le calibrage capacité-fatigabilité exige au moins deux cycles RHO.")
        if (type(self.resistance_pace_update_every_cycles) is not int
                or self.resistance_pace_update_every_cycles < 1):
            raise ValueError("La période de mise à jour de la répartition doit être un entier strictement positif.")
        for name in ("resistance_pace_capacity_gain", "resistance_pace_smoothing",
                     "resistance_pace_max_fraction_step", "resistance_pace_minimum_arm_torque_nm"):
            value = getattr(self, name)
            if (not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)):
                raise ValueError(f"{name} doit être fini.")
        if self.resistance_pace_capacity_gain <= 0:
            raise ValueError("Le gain de capacité doit être strictement positif.")
        if not 0 <= self.resistance_pace_smoothing <= 1 or not 0 <= self.resistance_pace_max_fraction_step <= 1:
            raise ValueError("Le lissage et le pas maximal doivent appartenir à [0, 1].")
        if self.resistance_pace_minimum_arm_torque_nm < 0:
            raise ValueError("Le couple minimal par bras doit être non négatif.")
        if self.muscle_weight_policy not in ("unit", "physio_u", "mechanical_sensitivity_squared_v1_experimental"):
            raise ValueError("La politique de poids musculaires doit être 'unit', 'physio_u' ou 'mechanical_sensitivity_squared_v1_experimental'.")
        if type(self.physio_update_every_cycles) is not int or self.physio_update_every_cycles < 1:
            raise ValueError("La cadence Physio-U doit être un entier strictement positif.")
        for name in ("physio_update_smoothing", "physio_update_max_log_step",
                     "physio_update_deadband_log", "physio_update_min_relative_weight",
                     "physio_update_max_relative_weight"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{name} doit être fini.")
        if not 0.0 <= self.physio_update_smoothing <= 1.0:
            raise ValueError("Le lissage Physio-U doit appartenir à [0, 1].")
        if self.physio_update_max_log_step <= 0 or self.physio_update_deadband_log < 0:
            raise ValueError("Le pas maximal Physio-U doit être positif et le seuil mort non négatif.")
        if not 0 < self.physio_update_min_relative_weight <= 1 <= self.physio_update_max_relative_weight:
            raise ValueError("La boîte des poids Physio-U doit être positive et contenir 1.")
        if self.muscle_weight_policy != "unit":
            if not self.parametric_fatigue_weights:
                raise ValueError("Cette adaptation de poids exige des poids de fatigue paramétriques pour préserver le NLP compilé.")
            if self.cycles_per_window != 1:
                raise ValueError("Cette adaptation de poids exige une fenêtre RHO d'un cycle certifié.")
            if (self.resistance_pace_policy == "capacity_feedback"
                    and self.physio_update_every_cycles != self.resistance_pace_update_every_cycles):
                raise ValueError("Physio-U et la répartition D/G doivent utiliser la même cadence de mise à jour.")
        if not isinstance(self.output_root, str) or not self.output_root.strip() or "\0" in self.output_root:
            raise ValueError("Le dossier de sortie est requis.")
        self._target(self.right_work_j_per_cycle, self.right_equivalent_mean_torque_nm, "Bras droit")
        self._target(self.left_work_j_per_cycle, self.left_equivalent_mean_torque_nm, "Bras gauche")
        return self

    def target_work_j(self, side: str) -> float:
        self.validate()
        if side not in ("right", "left"):
            raise ValueError("side must be 'right' or 'left'")
        return self._target(getattr(self, f"{side}_work_j_per_cycle"),
                            getattr(self, f"{side}_equivalent_mean_torque_nm"),
                            "Bras droit" if side == "right" else "Bras gauche")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_runtime_dict(self) -> dict[str, Any]:
        """Translate the GUI request to the coordinator's small stable schema."""
        self.validate()
        runtime = {
            "schema_version": self.schema_version,
            "factory": self.factory,
            "formulation": "isokinetic",
            "omega_rad_s": self.isokinetic_omega,
            "solver": self.solver,
            "cycles": self.cycles,
            "cycles_per_window": self.cycles_per_window,
            "stimulations_per_cycle": self.stimulations_per_cycle,
            "parallel": self.parallel,
            "reuse_compiled_artifacts": self.reuse_compiled_artifacts,
            "parametric_fatigue_weights": self.parametric_fatigue_weights,
            "right_equivalent_mean_torque_nm": self.target_work_j("right") / _TAU,
            "left_equivalent_mean_torque_nm": self.target_work_j("left") / _TAU,
            "right_target_work_j_per_cycle": self.target_work_j("right"),
            "left_target_work_j_per_cycle": self.target_work_j("left"),
            "right_runner_config": self.right_runner_config,
            "left_runner_config": self.left_runner_config,
        }
        if self.right_solver_cpu is not None:
            runtime["solver_cpu_affinity"] = {
                "right": self.right_solver_cpu,
                "left": self.left_solver_cpu,
            }
        if (self.resistance_pace_policy == "capacity_feedback"
                or self.resistance_pace_initial_split_policy != "manual"):
            if self.cycles_per_window != 1:
                raise ValueError("La répartition RHO synchronisée exige une fenêtre d'un cycle.")
            total = runtime["right_equivalent_mean_torque_nm"] + runtime["left_equivalent_mean_torque_nm"]
            if 2.0 * self.resistance_pace_minimum_arm_torque_nm > total:
                raise ValueError("Deux fois le couple minimal par bras dépasse le couple total.")
            runtime["resistance_pace"] = {
                "total_equivalent_mean_torque_nm": total,
                "initial_right_fraction": runtime["right_equivalent_mean_torque_nm"] / total,
                "minimum_arm_equivalent_mean_torque_nm": self.resistance_pace_minimum_arm_torque_nm,
                "capacity_feedback": self.resistance_pace_policy == "capacity_feedback",
                "initial_split_policy": self.resistance_pace_initial_split_policy,
                "update_every_cycles": self.resistance_pace_update_every_cycles,
                "capacity_gain": self.resistance_pace_capacity_gain,
                "smoothing": self.resistance_pace_smoothing,
                "max_fraction_step": self.resistance_pace_max_fraction_step,
            }
        if self.muscle_weight_policy != "unit":
            unit_weights = {name: 1.0 for name in ("Delt_ant", "Delt_post", "Biceps", "Triceps")}
            strategy = ("physio_update" if self.muscle_weight_policy == "physio_u"
                        else "mechanical_sensitivity")
            runtime["muscle_pace"] = {
                "initial_weight_basis": (
                    "GUI Physio-U unit start; no FHO or BO data"
                    if strategy == "physio_update"
                    else "GUI unit cold start; mechanics-only update after certified live envelope; no FHO or BO data"
                ),
                "right_initial_weights": dict(unit_weights),
                "left_initial_weights": dict(unit_weights),
                "adaptation_enabled": True,
                "adaptation_strategy": strategy,
                "update_every_cycles": self.physio_update_every_cycles,
                "smoothing": float(self.physio_update_smoothing),
                "max_log_step": float(self.physio_update_max_log_step),
                "physio_deadband_log": float(self.physio_update_deadband_log),
                "min_relative_weight": float(self.physio_update_min_relative_weight),
                "max_relative_weight": float(self.physio_update_max_relative_weight),
            }
        return runtime

    def to_json(self) -> str:
        self.validate()
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IndependentArmsGuiConfig":
        if not isinstance(data, Mapping):
            raise ValueError("La requête deux-bras doit être un objet JSON.")
        allowed = set(cls.__dataclass_fields__)
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"Champs deux-bras inconnus : {sorted(unknown)}")
        return cls(**dict(data)).validate()

    @classmethod
    def from_json(cls, text: str) -> "IndependentArmsGuiConfig":
        return cls.from_dict(json.loads(text))


def independent_arms_summary(payload: Mapping[str, Any], requested_cycles: int) -> str:
    """Render only coordinator declarations; never infer success from a PID."""
    if not isinstance(payload, Mapping):
        return "Synthèse deux-bras indisponible : résultat JSON non reconnu."
    lines = []
    arms = payload.get("arms", payload)
    if not isinstance(arms, Mapping):
        return "Synthèse deux-bras indisponible : résultats des bras non reconnus."
    for side, label in (("right", "Bras droit"), ("left", "Bras gauche")):
        result = arms.get(side)
        if not isinstance(result, Mapping):
            return "Synthèse deux-bras indisponible : résultat d'un bras manquant."
        metrics = result.get("metrics") if isinstance(result.get("metrics"), Mapping) else {}
        cycles = result.get("validated_cycles", metrics.get("validated_cycles"))
        success = result.get("success", metrics.get("success"))
        if success is True and type(cycles) is int and cycles >= requested_cycles:
            state = "réussite déclarée"
        elif success is False:
            state = "échec déclaré"
        else:
            state = "résultat partiel ou non certifié"
        parts = [f"{label} : {state} ; cycles validés {cycles if cycles is not None else '?'} / {requested_cycles}"]
        for key, text in (("target_work_j_per_cycle", "cible"), ("achieved_work_j_per_cycle", "travail"),
                          ("equivalent_mean_torque_nm", "τeq"), ("solver_time_s", "temps solveur")):
            value = result.get(key, metrics.get(key))
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                unit = " J" if "work" in key else " N.m" if "torque" in key else " s"
                parts.append(f"{text} {value:.6g}{unit}")
        lines.append(" ; ".join(parts) + ".")
    adjustment = payload.get("recommended_adjustment")
    if isinstance(adjustment, Mapping):
        values = []
        for side, label in (("right", "droite"), ("left", "gauche")):
            value = adjustment.get(f"{side}_equivalent_mean_torque_nm")
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                values.append(f"{label} τeq={value:.6g} N.m")
        if values:
            lines.append("Proposition du coordinateur (à confirmer avant relance) : " + ", ".join(values) + ".")
    return "\n".join(lines)
