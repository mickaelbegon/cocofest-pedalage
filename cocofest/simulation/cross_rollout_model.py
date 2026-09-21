"""Standard-library request model and launch plan for cross-integrator replay."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

from .launch import LaunchPlan, ROOT


EVALUATORS = ("dop853", "radau5", "gauss-legendre4x5")


@dataclass(frozen=True)
class CrossRolloutConfig:
    sources: tuple[str, ...] = ()
    reduced_profile: str = "benchmark-seed/reduced-cycling-fourier12.npz"
    model_config: str | None = None
    cost_config: str | None = None
    evaluators: tuple[str, ...] = EVALUATORS
    output_root: str = "gui-results/cross-rollout"
    start_cycle: int = 0
    cycles: int = 1
    cycle_duration_s: float = 1.0
    dop853_rtol: float = 1e-11
    dop853_atol: float = 1e-13
    samples_per_interval: int = 65
    schema_version: int = 1

    def validate(self):
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise ValueError("Version de configuration de rejeu inconnue.")
        if not isinstance(self.sources, (tuple, list)) or not self.sources or any(not isinstance(path, str) or not path.strip() for path in self.sources):
            raise ValueError("Sélectionnez au moins une archive NPZ de solution.")
        if len(set(self.sources)) != len(self.sources):
            raise ValueError("Les archives sources doivent être distinctes.")
        if not isinstance(self.evaluators, (tuple, list)) or not self.evaluators or any(not isinstance(item, str) for item in self.evaluators):
            raise ValueError("Les évaluateurs doivent être une liste de noms.")
        if set(self.evaluators) - set(EVALUATORS) or len(set(self.evaluators)) != len(self.evaluators):
            raise ValueError("Évaluateurs autorisés : dop853, radau5, gauss-legendre4x5, sans doublon.")
        if "dop853" not in self.evaluators:
            raise ValueError("DOP853 est la référence commune obligatoire.")
        if type(self.start_cycle) is not int or self.start_cycle < 0 or type(self.cycles) is not int or self.cycles < 1:
            raise ValueError("Le cycle de départ doit être ≥ 0 et le nombre de cycles ≥ 1.")
        if type(self.samples_per_interval) is not int or self.samples_per_interval < 2:
            raise ValueError("Le nombre d'échantillons de contraintes doit être un entier ≥ 2.")
        for field in ("cycle_duration_s", "dop853_rtol", "dop853_atol"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field} doit être fini et strictement positif.")
        if not isinstance(self.reduced_profile, str) or not self.reduced_profile.strip():
            raise ValueError("Le profil de mécanique réduite est requis.")
        if not isinstance(self.output_root, str) or not self.output_root.strip():
            raise ValueError("Le dossier de résultats est requis.")
        for field in ("model_config", "cost_config"):
            value = getattr(self, field)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field} invalide.")
        return self

    def to_json(self):
        self.validate()
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, value):
        data = json.loads(value)
        if not isinstance(data, dict) or set(data) - set(cls.__dataclass_fields__):
            raise ValueError("Champs de configuration de rejeu inconnus.")
        for field in ("sources", "evaluators"):
            if field in data:
                if not isinstance(data[field], list):
                    raise ValueError(f"{field} doit être une liste JSON.")
                data[field] = tuple(data[field])
        return cls(**data).validate()


def build_cross_rollout_plan(config: CrossRolloutConfig, prefix: Path, root: Path = ROOT):
    config.validate()
    root = Path(root).expanduser().absolute()
    def absolute(value):
        path = Path(value).expanduser()
        return str(path if path.is_absolute() else root / path)
    output = Path(absolute(config.output_root))
    command = [str(Path(prefix).expanduser().absolute() / "bin/python"),
               str(root / "scripts/evaluate_solver_cross_rollout.py"),
               "--reduced-profile", absolute(config.reduced_profile), "--evaluators", *config.evaluators,
               "--output-dir", str(output), "--cycle-start", str(config.start_cycle), "--cycles", str(config.cycles),
               "--cycle-duration", str(config.cycle_duration_s), "--rtol", str(config.dop853_rtol),
               "--atol", str(config.dop853_atol), "--samples-per-interval", str(config.samples_per_interval)]
    for source in config.sources:
        command.extend(["--source", absolute(source)])
    if config.model_config:
        command.extend(["--model-config", absolute(config.model_config)])
    if config.cost_config:
        command.extend(["--cost-config", absolute(config.cost_config)])
    return LaunchPlan(tuple(command), root, {}, "rho32", output / "matrix.json")


def cross_rollout_summary(payload):
    if not isinstance(payload, dict) or payload.get("schema") != "cocofest-solver-cross-rollout-v1":
        return "Rejeu : résultat indisponible ou non reconnu."
    sources = payload.get("sources", [])
    rows = [cell for source in sources for cell in source.get("evaluations", {}).values()]
    count = sum(row.get("status") == "success" for row in rows)
    ranking = payload.get("rankings", {}).get("dop853", [])
    best = f" Ordre des scores DOP853 (index sources) : {ranking}." if ranking else " Classement DOP853 indisponible : vérifier les conditions communes."
    return (f"Rejeu ouvert : {count}/{len(rows)} intégrations réussies.{best} "
            "Un score inférieur ne certifie pas la faisabilité. Matrice et diagnostics : matrix.json.")


def cross_rollout_report(payload):
    """Readable source × evaluator table, with validity separate from the score."""
    if not isinstance(payload, dict) or payload.get("schema") != "cocofest-solver-cross-rollout-v1":
        return cross_rollout_summary(payload)
    lines = [cross_rollout_summary(payload), "", "Sources (index commençant à 0) :"]
    for index, source in enumerate(payload.get("sources", [])):
        provenance = source.get("provenance", {})
        path = provenance.get("source", provenance.get("archive", {}))
        if isinstance(path, dict):
            path = path.get("path", "voir provenance JSON")
        lines.append(f"  {index} : {path or 'voir provenance JSON'}")
    comparison = payload.get("comparison", {})
    lines.extend(["", "Conditions communes : " + ("compatibles" if comparison.get("comparable") else "INCOMPATIBLES")])
    lines.extend(f"  - {reason}" for reason in comparison.get("reasons", []))
    lines.extend(["", "Source  Évaluateur             Coût commun      Δ vs DOP853     Violation phase  Violation vitesse",
                  "                                                           (rad)            (rad/s)"])
    def number(value):
        return "indisponible" if value is None else f"{value:.8g}"
    for index, source in enumerate(payload.get("sources", [])):
        for evaluator, cell in source.get("evaluations", {}).items():
            if cell.get("status") != "success":
                lines.append(f"{index:<7} {evaluator:<22} ÉCHEC : {cell.get('error', 'raison indisponible')}")
                continue
            metrics = cell.get("physical_metrics", {})
            lines.append(f"{index:<7} {evaluator:<22} {number(cell.get('common_score')):<16} "
                         f"{number(cell.get('score_difference_from_dop853')):<16} "
                         f"{number(metrics.get('sampled_phase_bound_violation_rad')):<16} "
                         f"{number(metrics.get('sampled_velocity_bound_violation_rad_s'))}")
    lines.extend(["", "Classement par coût :"])
    for evaluator, ranking in payload.get("rankings", {}).items():
        lines.append(f"  {evaluator} : {ranking if ranking is not None else 'indisponible'}")
    lines.extend(["", "Les contraintes sont échantillonnées ; ce résultat n'est pas un certificat NLP complet.",
                  "Radau-5 = 5 stages, ordre 9. GL4×5 = 4 stages de Gauss, 5 sous-pas.",
                  "Les temps des intégrateurs indépendants ne mesurent pas les performances IPOPT/ACADOS.",
                  "Coût natif du solveur et score physique commun sont deux quantités distinctes."])
    return "\n".join(lines)
