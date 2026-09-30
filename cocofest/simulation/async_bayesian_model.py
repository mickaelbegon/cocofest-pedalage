"""Dependency-free contract for persistent asynchronous Bayesian campaigns."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .config import SimulationConfig

STATUSES = ("observed", "horizon_censored", "administrative_censored", "technical_failure")
TIME_KEYS = {"solver_time": "solver_time_per_cycle_s", "hot_time": "hot_solver_time_median_s",
             "p90_time": "hot_solver_time_p90_s"}
SOLVER_FIELDS = {"solver", "collocation_degree", "threads", "ipopt_linear_solver", "madnlp_linear_solver",
                 "madnlp_hot_max_iterations", "madnlp_hot_max_wall_time", "madnlp_recovery",
                 "acados_qp_solver", "acados_sim_stages", "acados_sim_steps"}
CONTROLLER_FIELDS = {"cycles_per_window", "pulse_width_slew_weight", "pulse_width_max_step_us"}
MUSCLE_WEIGHT_PREFIX = "muscle_weight__"
MUSCLE_WEIGHT_COORDINATE_PREFIX = "muscle_weight_coordinate__"


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def muscle_weight_parameter_name(muscle_name: str) -> str:
    """Return the retired v1 multiplier parameter name.

    The v1 scheme was intentionally kept as a named helper so old result
    folders remain readable.  It must not be used in new studies: multiplying
    a calibrated input and letting RHO-Physio normalize/project it makes the
    space seen by Optuna many-to-one.
    """
    if not isinstance(muscle_name, str) or not muscle_name.strip():
        raise ValueError("Le nom de muscle doit être une chaîne non vide.")
    return f"{MUSCLE_WEIGHT_PREFIX}{muscle_name}"


def muscle_weight_coordinate_name(muscle_name: str) -> str:
    """Return the bounded latent-coordinate name for one non-reference muscle."""
    if not isinstance(muscle_name, str) or not muscle_name.strip():
        raise ValueError("Le nom de muscle doit être une chaîne non vide.")
    return f"{MUSCLE_WEIGHT_COORDINATE_PREFIX}{muscle_name}"


def configured_muscle_names(model_config: str) -> tuple[str, ...]:
    """Read the explicit model order used to validate BO muscle-weight knobs."""
    path = Path(model_config).expanduser()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"model_config illisible pour les poids musculaires : {path}") from exc
    muscles = payload.get("muscles") if isinstance(payload, dict) else None
    if not isinstance(muscles, dict) or not muscles or any(not isinstance(name, str) or not name for name in muscles):
        raise ValueError("model_config doit déclarer un objet non vide 'muscles' pour le BO des poids.")
    return tuple(muscles)


def muscle_weight_reference(muscle_names: tuple[str, ...]) -> str:
    """Choose the reproducible gauge for centred relative log weights.

    Biceps is the natural clinical reference in the cycling model.  Small test
    and future models may not contain it, in which case the declared model
    order supplies a deterministic reference rather than an implicit sort.
    """
    if not muscle_names:
        raise ValueError("Au moins un muscle est requis.")
    return "Biceps" if "Biceps" in muscle_names else muscle_names[0]


def relative_weight_bounds(weights_config: str) -> tuple[float, float]:
    """Read the actual RHO-PACE box instead of duplicating its defaults.

    The policy defaults are deliberately mirrored here because this module is
    dependency-free.  The configured runner remains the authority that
    validates the policy at solve time.
    """
    source = Path(weights_config).expanduser()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"weights_config illisible pour la géométrie BO : {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError("weights_config doit être un objet JSON.")
    policy = payload.get("policy", {})
    if not isinstance(policy, dict):
        raise ValueError("weights_config.policy doit être un objet quand il est fourni.")
    lower = policy.get("min_relative_weight", 0.25)
    upper = policy.get("max_relative_weight", 4.0)
    if (not _finite(lower) or not _finite(upper) or lower <= 0 or upper <= 0
            or lower > 1 or upper < 1 or lower > upper):
        raise ValueError("Les bornes de poids relatifs doivent être finies, positives et contenir 1.")
    return float(lower), float(upper)


def muscle_weight_parameters(parameters: Mapping[str, Any]) -> dict[str, float]:
    """Extract and validate named candidate weights from an Optuna parameter map."""
    selected = {}
    for name, value in parameters.items():
        if name.startswith(MUSCLE_WEIGHT_PREFIX):
            muscle = name.removeprefix(MUSCLE_WEIGHT_PREFIX)
            if not muscle or not _finite(value) or value <= 0:
                raise ValueError(f"Poids musculaire BO invalide : {name}")
            selected[muscle] = float(value)
    return selected


def muscle_weight_coordinates(parameters: Mapping[str, Any], muscle_names: tuple[str, ...],
                              *, min_weight: float, max_weight: float) -> dict[str, Any]:
    """Map independent coordinates to a unique, admissible relative cost.

    The controller works in centred log-weight space, where the common scale
    is unidentifiable.  We remove that gauge by deriving the Biceps log weight
    (or the first declared muscle when Biceps is absent).  The first ``n-2``
    logs are confined to a symmetric interior slab.  The final coordinate then
    selects its *feasible interval* exactly, so the reference closes the zero
    sum.  This is one-to-one, has geometric mean one, and cannot later be
    altered by RHO-Physio's [min,max] projection.

    With asymmetric policy bounds we use their largest symmetric log box about
    one.  That conservative interior is still admissible and avoids inventing
    an asymmetric normalization convention for a relative objective.
    """
    names = tuple(muscle_names)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("La géométrie BO exige au moins deux muscles nommés distincts.")
    reference = muscle_weight_reference(names)
    free = tuple(name for name in names if name != reference)
    expected = {muscle_weight_coordinate_name(name) for name in free}
    supplied = {name: value for name, value in parameters.items()
                if name.startswith(MUSCLE_WEIGHT_COORDINATE_PREFIX)}
    if set(supplied) != expected:
        raise ValueError("Les coordonnées BO doivent identifier exactement les muscles hors référence.")
    coordinates = {}
    for name in free:
        value = supplied[muscle_weight_coordinate_name(name)]
        if not _finite(value) or not -1.0 <= float(value) <= 1.0:
            raise ValueError(f"Coordonnée BO hors [-1,1] : {name}")
        coordinates[name] = float(value)
    log_lower, log_upper = math.log(min_weight), math.log(max_weight)
    radius = min(-log_lower, log_upper)
    if not radius > 0:
        raise ValueError("La boîte de poids relatifs doit contenir un voisinage de 1.")
    # Reserve enough room for the last free muscle and the reference.  The
    # resulting last interval has a strictly positive width for every point.
    direct_count = len(free) - 1
    direct_radius = radius / max(1, direct_count)
    logs = {}
    partial_sum = 0.0
    for name in free[:-1]:
        value = direct_radius * coordinates[name]
        logs[name] = value
        partial_sum += value
    last = free[-1]
    feasible_low = max(-radius, -radius - partial_sum)
    feasible_high = min(radius, radius - partial_sum)
    if not feasible_low < feasible_high:  # defensive: the slab above guarantees this.
        raise AssertionError("Intervalle final de coordonnées BO dégénéré.")
    logs[last] = .5 * (feasible_low + feasible_high) + .5 * (feasible_high - feasible_low) * coordinates[last]
    logs[reference] = -sum(logs.values())
    if any(value < log_lower - 1e-12 or value > log_upper + 1e-12 for value in logs.values()):
        raise AssertionError("La géométrie BO a produit un poids hors boîte.")
    weights = {name: math.exp(logs[name]) for name in names}
    return {
        "schema": "centered_log_relative_weights_v2",
        "reference_muscle": reference,
        "coordinates": {name: coordinates[name] for name in free},
        "effective_centered_log_weights": {name: logs[name] for name in names},
        "effective_initial_weights": weights,
        "min_relative_weight": float(min_weight),
        "max_relative_weight": float(max_weight),
        "geometric_mean": math.prod(weights.values()) ** (1.0 / len(weights)),
    }


def validate_muscle_weight_search(base_config: Mapping[str, Any], search_space: Mapping[str, Any]):
    """Check the exact, non-redundant relative-log BO contract.

    ACADOS currently has no connected physiological-weight objective adapter;
    keep this contract on the already audited IPOPT RHO-Physio/PACE routes.
    """
    legacy_fields = {name for name in search_space if name.startswith(MUSCLE_WEIGHT_PREFIX)}
    fields = {name for name in search_space if name.startswith(MUSCLE_WEIGHT_COORDINATE_PREFIX)}
    if legacy_fields:
        raise ValueError("Les multiplicateurs muscle_weight__ v1 sont ambigus après normalisation RHO ; "
                         "utilisez les coordonnées centrées muscle_weight_coordinate__ v2.")
    if not fields:
        return ()
    config = SimulationConfig.from_dict(base_config)
    if config.mode not in ("rho-physio", "rho-pace"):
        raise ValueError("Le BO de poids musculaires exige le mode rho-physio ou rho-pace.")
    if config.solver == "acados":
        raise ValueError("Les poids musculaires ne sont pas encore connectés à ACADOS ; utilisez IPOPT.")
    if not config.model_config or not config.weights_config:
        raise ValueError("Le BO de poids musculaires exige model_config et weights_config de référence.")
    names = configured_muscle_names(config.model_config)
    if len(names) < 2:
        raise ValueError("Le BO de poids musculaires exige au moins deux muscles.")
    reference = muscle_weight_reference(names)
    expected = {muscle_weight_coordinate_name(name) for name in names if name != reference}
    if fields != expected:
        missing, unexpected = sorted(expected - fields), sorted(fields - expected)
        details = []
        if missing:
            details.append(f"manquants={missing}")
        if unexpected:
            details.append(f"inconnus={unexpected}")
        raise ValueError("Le BO doit rechercher exactement une coordonnée par muscle hors référence "
                         f"({reference}) (" + ", ".join(details) + ").")
    min_weight, max_weight = relative_weight_bounds(config.weights_config)
    for name in fields:
        spec = search_space[name]
        if (spec.get("type") != "float" or spec.get("log", False)
                or spec.get("low") != -1.0 or spec.get("high") != 1.0):
            raise ValueError("Chaque coordonnée musculaire BO doit être un flottant linéaire dans [-1, 1].")
    return tuple(name.removeprefix(MUSCLE_WEIGHT_COORDINATE_PREFIX) for name in names if name != reference), min_weight, max_weight


@dataclass(frozen=True)
class AsyncBayesianCampaignConfig:
    base_config: dict[str, Any]
    output_root: str
    study_name: str = "cycling"
    study_kind: str = "solver"
    phase: str = "screening"
    max_cycles: int = 10
    metric: str = "solver_time"
    n_trials: int = 16
    workers: int = 4
    n_startup_trials: int = 6
    seed: int = 42
    timeout_s: float | None = None
    stop_grace_s: float = 120.0
    search_space: dict[str, dict[str, Any]] = field(default_factory=lambda: {
        "solver": {"type": "categorical", "choices": ["ipopt", "madnlp"]},
        "collocation_degree": {"type": "int", "low": 3, "high": 5},
        "threads": {"type": "categorical", "choices": [1]},
    })
    enqueue_trials: tuple[dict[str, Any], ...] = ()

    @property
    def direction(self):
        return "maximize" if self.metric == "continuous_endurance" else "minimize"

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]):
        if not isinstance(data, Mapping):
            raise ValueError("La campagne doit être un objet JSON.")
        data = dict(data)
        if "enqueue_trials" in data:
            data["enqueue_trials"] = tuple(data["enqueue_trials"])
        return cls(**data)

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text))

    def validate(self):
        SimulationConfig.from_dict(self.base_config)
        if self.study_kind not in ("solver", "controller"):
            raise ValueError("study_kind doit être solver ou controller.")
        if self.phase not in ("screening", "confirmation"):
            raise ValueError("phase doit être screening ou confirmation.")
        if self.metric not in (*TIME_KEYS, "continuous_endurance"):
            raise ValueError("Métrique inconnue.")
        if self.study_kind == "controller" and self.metric != "continuous_endurance":
            raise ValueError("Une étude contrôleur utilise continuous_endurance.")
        for name in ("max_cycles", "n_trials", "workers", "n_startup_trials"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} doit être un entier positif.")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed doit être un entier positif ou nul.")
        if self.timeout_s is not None and (not _finite(self.timeout_s) or self.timeout_s <= 0):
            raise ValueError("timeout_s doit être positif.")
        if not _finite(self.stop_grace_s) or self.stop_grace_s < 0:
            raise ValueError("stop_grace_s doit être positif ou nul.")
        if not isinstance(self.output_root, str) or not self.output_root.strip() or "\0" in self.output_root:
            raise ValueError("Dossier de campagne invalide.")
        if not isinstance(self.study_name, str) or not self.study_name.strip():
            raise ValueError("Nom d'étude absent.")
        allowed = SOLVER_FIELDS if self.study_kind == "solver" else CONTROLLER_FIELDS
        dynamic_muscle_fields = {name for name in self.search_space
                                 if name.startswith(MUSCLE_WEIGHT_PREFIX)
                                 or name.startswith(MUSCLE_WEIGHT_COORDINATE_PREFIX)}
        invalid = set(self.search_space) - allowed - dynamic_muscle_fields
        if not self.search_space or invalid:
            raise ValueError(f"Paramètres autorisés dans l'étude {self.study_kind}: {sorted(allowed)}")
        if dynamic_muscle_fields:
            if self.study_kind != "controller":
                raise ValueError("Les poids musculaires sont réservés à une étude contrôleur.")
            validate_muscle_weight_search(self.base_config, self.search_space)
        for name, spec in self.search_space.items():
            kind = spec.get("type")
            if kind == "categorical":
                values = spec.get("choices")
                if not isinstance(values, list) or not values or any(not isinstance(x, (str, int, float, bool, type(None))) for x in values):
                    raise ValueError(f"Choix invalides pour {name}.")
                if any(isinstance(x, float) and not math.isfinite(x) for x in values):
                    raise ValueError(f"Choix non finis pour {name}.")
            elif kind in ("int", "float"):
                low, high = spec.get("low"), spec.get("high")
                if not _finite(low) or not _finite(high) or low > high:
                    raise ValueError(f"Bornes invalides pour {name}.")
                if kind == "int" and (type(low) is not int or type(high) is not int):
                    raise ValueError(f"Bornes entières requises pour {name}.")
                if spec.get("log", False) and low <= 0:
                    raise ValueError(f"Borne logarithmique positive requise pour {name}.")
            else:
                raise ValueError(f"Distribution inconnue pour {name}.")
        for point in self.enqueue_trials:
            if set(point) != set(self.search_space):
                raise ValueError("Chaque essai initial doit contenir tous les paramètres recherchés.")
            for name, value in point.items():
                spec = self.search_space[name]
                if spec["type"] == "categorical":
                    valid = value in spec["choices"]
                else:
                    valid = _finite(value) and spec["low"] <= value <= spec["high"]
                    valid = valid and (spec["type"] != "int" or type(value) is int)
                if not valid:
                    raise ValueError(f"Essai initial hors domaine: {name}.")
        self.to_json()
        return self

    def scientific_contract(self):
        """Fields that cannot change on resume; budget/workers may change."""
        data = self.to_dict()
        for key in ("n_trials", "workers", "output_root", "enqueue_trials", "stop_grace_s"):
            data.pop(key)
        return data


@dataclass(frozen=True)
class SimResult:
    status: str
    score: float | None = None
    lower_bound: float | None = None
    validated_cycles: int = 0
    reason: str = ""
    method: str | None = None

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"Statut inconnu: {self.status}")
        if self.score is not None and (not _finite(self.score) or self.score < 0):
            raise ValueError("Le score doit être fini et positif ou nul.")
        if self.status != "observed" and self.score is not None:
            raise ValueError("Seule une observation exacte peut avoir un score.")

    def to_dict(self):
        return asdict(self)


def classify_result(result, requested_cycles, metric, *, administrative=False, error=None, returncode=0):
    """Certify an observation before sending it to the Bayesian sampler.

    A completed endurance horizon gives only a lower bound. A completed timing
    benchmark gives an exact time observation. An accepted fatigue endpoint is
    an exact endurance observation. A capacity-plus-saturation endpoint is a
    separately labelled *proxy* observation: it is usable for controller BO,
    but must never be presented as a terminal-cycle feasibility certificate.
    """
    result = result if isinstance(result, dict) else {}
    cycles = result.get("validated_cycles", 0)
    cycles = cycles if type(cycles) is int and 0 <= cycles <= requested_cycles else 0
    if administrative:
        return SimResult("administrative_censored", lower_bound=float(cycles), validated_cycles=cycles,
                         reason=error or "administrative_stop")
    if error or returncode is not None and returncode < 0:
        return SimResult("technical_failure", validated_cycles=cycles, reason=error or f"signal:{returncode}")
    completed = result.get("success") is True and cycles == requested_cycles
    if metric in TIME_KEYS:
        value = result.get(TIME_KEYS[metric])
        if completed and returncode == 0 and _finite(value) and value >= 0:
            return SimResult("observed", score=float(value), validated_cycles=cycles, reason="certified_timing")
        return SimResult("technical_failure", validated_cycles=cycles, reason="uncertified_or_missing_timing")
    if metric != "continuous_endurance":
        raise ValueError(f"Métrique inconnue: {metric}")
    if completed and returncode == 0:
        return SimResult("horizon_censored", lower_bound=float(cycles), validated_cycles=cycles,
                         reason="requested_horizon_completed")
    outcome = result.get("fatigue_endurance_outcome") or {}
    estimate = result.get("continuous_endurance") or {}
    value = estimate.get("continuous_cycle_equivalent")
    if (cycles > 0 and outcome.get("accepted") is True
            and outcome.get("label") == "fatigue_limited_candidate" and estimate.get("available") is True
            and estimate.get("censored") is not True and _finite(value) and cycles <= value <= cycles + 1):
        return SimResult("observed", score=float(value), validated_cycles=cycles,
                         reason="accepted_fatigue_endpoint", method=estimate.get("method"))
    evidence = set(outcome.get("evidence") or ())
    if (cycles > 0 and outcome.get("label") == "unconfirmed_endurance_stop"
            and {"ding_force_capacity_materially_decreased", "pulse_width_upper_bound_active"} <= evidence
            and estimate.get("available") is True and estimate.get("censored") is not True
            and _finite(value) and cycles <= value <= cycles + 1):
        return SimResult("observed", score=float(value), validated_cycles=cycles,
                         reason="capacity_saturation_proxy_endpoint", method=estimate.get("method"))
    return SimResult("technical_failure", validated_cycles=cycles, reason="unconfirmed_endurance_stop")
