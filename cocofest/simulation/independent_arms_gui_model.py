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
        if self.solver not in ("ipopt", "acados"):
            raise ValueError("Le solveur indépendant doit être IPOPT ou ACADOS.")
        for name in ("cycles", "cycles_per_window", "stimulations_per_cycle"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} doit être un entier strictement positif.")
        if self.cycles_per_window > self.cycles:
            raise ValueError("La fenêtre RHO ne peut excéder le nombre de cycles.")
        if (not isinstance(self.isokinetic_omega, (int, float)) or isinstance(self.isokinetic_omega, bool)
                or not math.isfinite(self.isokinetic_omega) or self.isokinetic_omega >= 0):
            raise ValueError("La vitesse isocinétique doit être finie et négative (rad/s).")
        if type(self.parallel) is not bool or type(self.reuse_compiled_artifacts) is not bool:
            raise ValueError("parallel et reuse_compiled_artifacts doivent être booléens.")
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
        return {
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
            "right_equivalent_mean_torque_nm": self.target_work_j("right") / _TAU,
            "left_equivalent_mean_torque_nm": self.target_work_j("left") / _TAU,
            "right_target_work_j_per_cycle": self.target_work_j("right"),
            "left_target_work_j_per_cycle": self.target_work_j("left"),
            "right_runner_config": self.right_runner_config,
            "left_runner_config": self.left_runner_config,
        }

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
