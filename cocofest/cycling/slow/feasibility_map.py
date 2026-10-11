"""Conservative, conditional interpolation of authenticated RHO experiments.

Phi is an experimental map, never a feasibility certificate. It conditions on
all declared slow states, torque and muscle weights and checks the fast-state
envelope. Its output may only be enabled after a predeclared holdout gate.
Numerical failures and right censoring remain unidentified outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from .oracle import AUDIT_SCOPE


OUTCOMES = ("certified_feasible", "physiological_failure_certified", "right_censored", "numerical_failure")


class PhiOutOfSupport(ValueError):
    """The query has no supported conditional interpolation."""


@dataclass(frozen=True)
class PhiPoint:
    sample_id: str
    family_id: str
    partition: str
    side: str
    model_sha256: str
    task_sha256: str
    state_names: tuple[str, ...]
    slow_state: tuple[float, ...]
    fast_names: tuple[str, ...]
    fast_state: tuple[float, ...]
    command_names: tuple[str, ...]
    command: tuple[float, ...]
    outcome: str
    force_integrals_n_s: tuple[float, ...] | None
    artifacts: tuple[tuple[str, str], ...]

    @property
    def coordinates(self):
        return np.asarray((*self.slow_state, *self.command), float)

    def verify_files(self):
        if not self.artifacts:
            raise ValueError("Phi point lacks archived oracle evidence")
        for path, expected in self.artifacts:
            if sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Phi oracle evidence changed: {path}")


def _vector(document, names_key, values_key):
    names, values = document.get(names_key), document.get(values_key)
    if (not isinstance(names, list) or not names or any(not isinstance(n, str) or not n for n in names)
            or len(set(names)) != len(names) or not isinstance(values, list) or len(names) != len(values)
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
        raise ValueError(f"Incomplete finite coordinate contract: {values_key}")
    return tuple(names), tuple(float(v) for v in values)


def _artifact(reference, path, expected):
    if not isinstance(path, str) or not path:
        raise ValueError("Missing oracle evidence path")
    path = Path(path)
    path = (path if path.is_absolute() else reference.parent / path).resolve(strict=True)
    actual = sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError("Oracle artifact digest mismatch")
    return str(path), actual


def _audit(audit, tolerance):
    if not isinstance(audit, dict) or audit.get("scope") != AUDIT_SCOPE:
        raise ValueError("Full discrete RHO audit is required")
    for key in ("constraint_normalized_violation", "decision_normalized_violation", "terminal_work_normalized_violation"):
        value = audit.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= tolerance:
            raise ValueError("RHO audit does not pass tolerance")


def load_phi_point(path: Path, *, maximum_tolerance: float = 1e-5) -> PhiPoint:
    """Read an explicit v1 oracle receipt; bare model predictions are rejected.

    The receipt records *initial* slow and fast states for its frozen solve.
    ``state_names`` must contain A/Tau1/Km for every commanded muscle, and
    ``fast_names`` the full declared fast/history state of that RHO experiment.
    ``task_sha256`` pins the shared discretization and physical task contract.
    """
    if not math.isfinite(maximum_tolerance) or maximum_tolerance <= 0:
        raise ValueError("maximum_tolerance must be positive")
    path = Path(path).resolve(strict=True)
    d = json.loads(path.read_text())
    if (d.get("schema") != "slow-phi-rho-oracle-v1" or d.get("partition") not in ("train", "holdout")
            or d.get("side") not in ("left", "right")
            or d.get("outcome") not in OUTCOMES or d.get("full_fast_state_declared") is not True):
        raise ValueError("Unknown or incomplete Phi oracle receipt")
    for field in ("sample_id", "family_id", "model_sha256", "task_sha256"):
        if not isinstance(d.get(field), str) or not d[field]:
            raise ValueError(f"Missing Phi provenance: {field}")
    for field in ("model_sha256", "task_sha256"):
        if len(d[field]) != 64 or any(c not in "0123456789abcdef" for c in d[field]):
            raise ValueError(f"Invalid SHA-256: {field}")
    states, slow = _vector(d, "state_names", "slow_state")
    fast_names, fast = _vector(d, "fast_names", "fast_state")
    commands, command = _vector(d, "command_names", "command")
    muscles = tuple(name.removeprefix("weight:") for name in commands if name.startswith("weight:"))
    if (not muscles or set(states) != {f"{s}_{m}" for m in muscles for s in ("A", "Tau1", "Km")}
            or "torque_nm" not in commands or any(v <= 0 for v in slow)
            or any(value <= 0 for name, value in zip(commands, command) if name.startswith("weight:"))):
        raise ValueError("Full slow states, torque and positive muscle weights are required")
    tolerance = d.get("audit_tolerance")
    if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or not 0 < tolerance <= maximum_tolerance:
        raise ValueError("Invalid declared audit tolerance")
    artifacts = [(str(path), sha256(path.read_bytes()).hexdigest()),
                 _artifact(path, d.get("prepared_archive"), d.get("prepared_archive_sha256"))]
    # Tie input coordinates to frozen physical equalities and saved history,
    # rather than trusting a manually copied slow-state annotation.
    with np.load(artifacts[1][0], allow_pickle=False) as archive:
        for key, expected in zip(states, slow):
            lower = np.asarray(archive[f"problem__x_bounds:{key}:min"], float)
            upper = np.asarray(archive[f"problem__x_bounds:{key}:max"], float)
            if lower.ndim != 2 or lower.shape != upper.shape or lower[0, 0] != expected or upper[0, 0] != expected:
                raise ValueError("Phi slow input differs from the frozen RHO equality")
        sources = d.get("fast_sources")
        if not isinstance(sources, list) or len(sources) != len(fast):
            raise ValueError("Every fast/history coordinate requires an archived location")
        for expected, source in zip(fast, sources):
            if (not isinstance(source, dict) or not isinstance(source.get("key"), str)
                    or type(source.get("row")) is not int or type(source.get("column")) is not int
                    or source["row"] < 0 or source["column"] < 0):
                raise ValueError("Invalid fast/history archived location")
            array = np.asarray(archive[source["key"]], float)
            if array.ndim != 2 or array[source["row"], source["column"]] != expected:
                raise ValueError("Phi fast/history state differs from the frozen RHO archive")
        sources = d.get("command_sources")
        if not isinstance(sources, list) or len(sources) != len(command):
            raise ValueError("Every command requires an archived numerical location")
        for expected, source in zip(command, sources):
            if (not isinstance(source, dict) or not isinstance(source.get("key"), str)
                    or type(source.get("row")) is not int or type(source.get("column")) is not int
                    or source["row"] < 0 or source["column"] < 0
                    or type(source.get("scale")) not in (int, float)
                    or not math.isfinite(source["scale"]) or source["scale"] == 0):
                raise ValueError("Invalid command archived location")
            array = np.asarray(archive[source["key"]], float)
            if array.ndim != 2 or not math.isclose(float(array[source["row"], source["column"]])*source["scale"], expected,
                                                 rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Phi command differs from the prepared RHO archive")
    integrals = None
    if d["outcome"] == "certified_feasible":
        _audit(d.get("independent_audit"), tolerance)
        artifact = _artifact(path, d.get("witness"), d.get("witness_sha256"))
        with np.load(artifact[0], allow_pickle=False) as witness:
            metadata = json.loads(str(witness["metadata__json"].item()))
            if metadata.get("independent_audit") != d["independent_audit"]:
                raise ValueError("Witness audit differs from Phi receipt")
            quadrature = np.asarray(d.get("force_quadrature_weights"), float)
            if quadrature.ndim != 1 or not quadrature.size or not np.isfinite(quadrature).all():
                raise ValueError("Force integral requires the declared archived-node quadrature")
            measured_integrals = []
            for muscle in muscles:
                force = np.asarray(witness[f"states__F_{muscle}"], float)
                if force.shape != (1, quadrature.size) or not np.isfinite(force).all():
                    raise ValueError("Force witness and quadrature node grids differ")
                measured_integrals.append(float(force[0] @ quadrature))
        artifacts.append(artifact)
        raw = d.get("force_integrals_n_s")
        if not isinstance(raw, list) or len(raw) != len(muscles) or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in raw):
            raise ValueError("Certified Phi point needs nonnegative force integrals per muscle")
        integrals = tuple(raw)
        if not np.allclose(integrals, measured_integrals, rtol=1e-12, atol=1e-12):
            raise ValueError("Reported force integrals differ from archived RHO forces")
    elif d["outcome"] == "physiological_failure_certified":
        failure = d.get("failure_certificate", {})
        if (failure.get("scope") != "protocol_conditioned_local_frozen_endpoint"
                or failure.get("prepared_archive_sha256") != d["prepared_archive_sha256"]
                or failure.get("zero_objective_status") != "Infeasible_Problem_Detected"
                or failure.get("zero_objective_feasible_witness") is not False):
            raise ValueError("Physiological outcome lacks a frozen local endpoint certificate")
        _audit(failure.get("fatigue_rest_audit"), tolerance)
        artifacts.append(_artifact(path, failure.get("report"), failure.get("report_sha256")))
        report = json.loads(Path(artifacts[-1][0]).read_text())
        if report != {k: v for k, v in failure.items() if k not in ("report", "report_sha256")}:
            raise ValueError("Frozen endpoint certificate differs from its archived report")
        artifact = _artifact(path, failure.get("fatigue_rest_witness"), failure.get("fatigue_rest_witness_sha256"))
        with np.load(artifact[0], allow_pickle=False) as witness:
            metadata = json.loads(str(witness["metadata__json"].item()))
            if metadata.get("independent_audit") != failure["fatigue_rest_audit"]:
                raise ValueError("Fatigue-rest witness lacks the archived passing audit")
        artifacts.append(artifact)
    point = PhiPoint(d["sample_id"], d["family_id"], d["partition"], d["side"], d["model_sha256"], d["task_sha256"],
                     states, slow, fast_names, fast, commands, command, d["outcome"], integrals, tuple(artifacts))
    point.verify_files()
    return point


@dataclass(frozen=True)
class PhiEstimate:
    outcome: str
    force_integrals_n_s: tuple[float, ...] | None
    supporting_samples: tuple[str, ...]
    weights: tuple[float, ...]
    certified: bool = False


class ConditionalPhiMap:
    """Convex interpolation, with explicit abstention in unidentified cells."""

    def __init__(self, points):
        self.points = tuple(points)
        if not self.points or any(p.partition != "train" for p in self.points):
            raise ValueError("Phi fitting accepts nonempty training data only")
        if len({p.sample_id for p in self.points}) != len(self.points):
            raise ValueError("Duplicate Phi sample")
        self.contract = self._contract(self.points[0])
        if any(self._contract(p) != self.contract for p in self.points):
            raise ValueError("Phi points differ in model, task or coordinate contract")
        for point in self.points:
            point.verify_files()
            if point != load_phi_point(Path(point.artifacts[0][0])):
                raise ValueError("Phi point differs from its validated oracle receipt")

    @staticmethod
    def _contract(p):
        return p.side, p.model_sha256, p.task_sha256, p.state_names, p.fast_names, p.command_names

    def predict(self, slow_state, command, fast_state) -> PhiEstimate:
        from scipy.optimize import linprog

        for point in self.points:
            point.verify_files()
        target = np.asarray((*slow_state, *command), float)
        fast = np.asarray(fast_state, float)
        coords = np.stack([p.coordinates for p in self.points])
        fast_support = np.asarray([p.fast_state for p in self.points])
        if target.shape != coords.shape[1:] or fast.shape != fast_support.shape[1:] or not np.isfinite(target).all() or not np.isfinite(fast).all():
            raise ValueError("Phi query coordinate contract differs")
        if np.any(fast < fast_support.min(axis=0)) or np.any(fast > fast_support.max(axis=0)):
            raise PhiOutOfSupport("Fast covariates are outside the sampled envelope")
        scale = np.ptp(coords, axis=0)
        scale[scale == 0] = 1.
        delta = (coords-target)/scale
        result = linprog(np.sum(delta**2, axis=1), A_eq=np.vstack([delta.T, np.ones(len(coords))]),
                         b_eq=np.r_[np.zeros(target.size), 1.], bounds=(0., 1.), method="highs")
        if not result.success or np.max(np.abs(delta.T@result.x)) > 1e-8:
            raise PhiOutOfSupport("Phi query lies outside sampled convex support")
        selected = np.flatnonzero(result.x > 1e-9)
        points = [self.points[i] for i in selected]
        local_fast = np.asarray([p.fast_state for p in points])
        if np.any(fast < local_fast.min(axis=0)) or np.any(fast > local_fast.max(axis=0)):
            raise PhiOutOfSupport("Fast covariates are outside the supporting cell envelope")
        weights = result.x[selected]
        outcomes = {p.outcome for p in points}
        outcome, integrals = "abstain", None
        if outcomes == {"certified_feasible"}:
            outcome = "estimated_feasible"
            integrals = tuple(weights @ np.asarray([p.force_integrals_n_s for p in points]))
        elif outcomes == {"physiological_failure_certified"}:
            outcome = "estimated_local_failure"
        return PhiEstimate(outcome, integrals, tuple(p.sample_id for p in points), tuple(weights))

    def validate_holdout(self, holdout, *, maximum_false_feasible_rate=0., minimum_coverage=1., maximum_force_relative_error=.05):
        """Gate interpolation against separate families; censorship cannot pass."""
        if (not 0 <= maximum_false_feasible_rate <= 1 or not 0 <= minimum_coverage <= 1
                or not math.isfinite(maximum_force_relative_error) or maximum_force_relative_error < 0):
            raise ValueError("Invalid holdout acceptance thresholds")
        holdout = tuple(holdout)
        if not holdout:
            raise ValueError("A nonempty holdout set is required")
        families = {p.family_id for p in self.points}
        if len({p.sample_id for p in holdout}) != len(holdout):
            raise ValueError("Duplicate holdout sample")
        rows, covered, false_feasible, feasible_predictions, max_error = [], 0, 0, 0, 0.
        for p in holdout:
            p.verify_files()
            if p != load_phi_point(Path(p.artifacts[0][0])):
                raise ValueError("Holdout point differs from its validated oracle receipt")
            if p.partition != "holdout" or p.family_id in families or self._contract(p) != self.contract:
                raise ValueError("Holdout leaks a training family or differs in contract")
            try:
                prediction = self.predict(p.slow_state, p.command, p.fast_state)
                outcome = prediction.outcome
            except PhiOutOfSupport:
                outcome = "outside_support"
                prediction = None
            identified = p.outcome in ("certified_feasible", "physiological_failure_certified")
            covered += identified and outcome in ("estimated_feasible", "estimated_local_failure")
            if outcome == "estimated_feasible":
                feasible_predictions += 1
                false_feasible += p.outcome != "certified_feasible"
                if p.outcome == "certified_feasible":
                    actual = np.asarray(p.force_integrals_n_s)
                    error = float(np.max(np.abs(np.asarray(prediction.force_integrals_n_s)-actual)/np.maximum(np.abs(actual), 1e-12)))
                    max_error = max(max_error, error)
            rows.append({"sample_id": p.sample_id, "observed": p.outcome, "predicted": outcome})
        coverage = covered/len(holdout)
        rate = false_feasible/max(feasible_predictions, 1)
        return {"passed": coverage >= minimum_coverage and rate <= maximum_false_feasible_rate
                and max_error <= maximum_force_relative_error and feasible_predictions > 0,
                "coverage": coverage, "false_feasible_rate": rate, "maximum_force_relative_error": max_error,
                "holdout_count": len(holdout), "rows": rows,
                "training_receipts": [p.artifacts[0] for p in self.points],
                "holdout_receipts": [p.artifacts[0] for p in holdout],
                "thresholds": {"maximum_false_feasible_rate": maximum_false_feasible_rate,
                               "minimum_coverage": minimum_coverage,
                               "maximum_force_relative_error": maximum_force_relative_error},
                "interpretation": "conditional interpolation validation; no RHO feasibility certification"}
