"""Portable configuration for Bayesian solver-configuration campaigns.

This module deliberately contains no optimisation or scientific dependency so
that JSON can be inspected and a campaign can be prepared from the GUI.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping


_SOLVERS = ("ipopt", "madnlp", "fatrop")
_METRICS = ("solver_time", "hot_time", "p90_time", "continuous_endurance")


@dataclass(frozen=True)
class BayesianCampaignConfig:
    """Search only solver choices that are already exposed by the workbench."""

    base_config: dict[str, Any]
    output_root: str
    solvers: tuple[str, ...] = _SOLVERS
    n_calls: int = 16
    n_initial_points: int = 6
    random_state: int = 42
    collocation_degree_min: int = 3
    collocation_degree_max: int = 5
    thread_choices: tuple[int, ...] = (1, 2, 4)
    metric: str = "solver_time"

    def __post_init__(self):
        if isinstance(self.solvers, list):
            object.__setattr__(self, "solvers", tuple(self.solvers))
        if isinstance(self.thread_choices, list):
            object.__setattr__(self, "thread_choices", tuple(self.thread_choices))
        if isinstance(self.output_root, Path):
            object.__setattr__(self, "output_root", str(self.output_root))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["solvers"] = list(self.solvers)
        data["thread_choices"] = list(self.thread_choices)
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BayesianCampaignConfig":
        if not isinstance(data, Mapping):
            raise ValueError("La campagne doit être un objet JSON.")
        expected = set(cls.__dataclass_fields__)
        unknown = set(data) - expected
        if unknown:
            raise ValueError(f"Champs de campagne inconnus : {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_json(cls, text: str) -> "BayesianCampaignConfig":
        return cls.from_dict(json.loads(text))

    def validate(self) -> "BayesianCampaignConfig":
        from .config import SimulationConfig
        from .capabilities import CapabilityRegistry

        if not isinstance(self.base_config, dict):
            raise ValueError("Configuration de base absente ou invalide.")
        base = SimulationConfig.from_dict(self.base_config)
        if not isinstance(self.output_root, str) or not self.output_root.strip() or "\0" in self.output_root:
            raise ValueError("Dossier de campagne invalide.")
        if not self.solvers or len(set(self.solvers)) != len(self.solvers) or any(s not in _SOLVERS for s in self.solvers):
            raise ValueError(f"Solveurs BO : choisissez une liste sans doublon parmi {', '.join(_SOLVERS)}.")
        if type(self.n_calls) is not int or self.n_calls < 2:
            raise ValueError("Budget BO : au moins 2 évaluations.")
        if type(self.n_initial_points) is not int or not 1 <= self.n_initial_points < self.n_calls:
            raise ValueError("Points initiaux BO : entier entre 1 et budget − 1.")
        if type(self.random_state) is not int:
            raise ValueError("Graine BO : entier attendu.")
        if type(self.collocation_degree_min) is not int or type(self.collocation_degree_max) is not int or not 1 <= self.collocation_degree_min <= self.collocation_degree_max:
            raise ValueError("Bornes de collocation invalides.")
        if not self.thread_choices or any(type(v) is not int or v < 1 for v in self.thread_choices) or len(set(self.thread_choices)) != len(self.thread_choices):
            raise ValueError("Threads BO : entiers positifs sans doublon attendus.")
        cardinality = len(self.solvers) * (self.collocation_degree_max - self.collocation_degree_min + 1) * len(self.thread_choices)
        if self.n_calls > cardinality:
            raise ValueError(f"Budget BO : {self.n_calls} essais mais seulement {cardinality} configurations distinctes.")
        if self.metric not in _METRICS:
            raise ValueError(f"Métrique BO : choisissez parmi {', '.join(_METRICS)}.")
        # Validate every selectable solver before a potentially expensive run.
        for solver in self.solvers:
            CapabilityRegistry.require_valid(SimulationConfig.from_dict({**base.to_dict(), "solver": solver}))
        return self


def campaign_summary(config: BayesianCampaignConfig) -> str:
    return (f"BO GP/EI · {config.n_calls} essais ({config.n_initial_points} initiaux), graine {config.random_state}\n"
            f"Solveurs : {', '.join(config.solvers)} · collocation {config.collocation_degree_min}–{config.collocation_degree_max} · "
            f"threads : {', '.join(map(str, config.thread_choices))}\n"
            "Seuls les essais certifiés (succès et tous les cycles validés) sont classés ; les échecs restent journalisés.")
