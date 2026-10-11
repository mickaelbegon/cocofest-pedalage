from hashlib import sha256
import json

import numpy as np
import pytest

from cocofest.cycling.slow.feasibility_map import ConditionalPhiMap, PhiOutOfSupport, load_phi_point
from cocofest.cycling.slow.oracle import AUDIT_SCOPE


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def point_factory(tmp_path):
    def make(name, a, *, torque=1., fast=.4, partition="train", outcome="certified_feasible", family=None, force=None):
        states = ["A_Biceps", "Tau1_Biceps", "Km_Biceps"]
        slow = [a, .06, .14]
        prepared = tmp_path / f"{name}-prepared.npz"
        arrays = {f"problem__x_bounds:{s}:{bound}": np.asarray([[value]])
                  for s, value in zip(states, slow) for bound in ("min", "max")}
        arrays.update(fast=np.array([[fast]]), commands=np.array([[torque, 1.]]))
        np.savez_compressed(prepared, **arrays)
        audit = {"scope": AUDIT_SCOPE, "constraint_normalized_violation": 1e-10,
                 "decision_normalized_violation": 1e-10, "terminal_work_normalized_violation": 0.}
        force = a/100 if force is None else force
        witness = tmp_path / f"{name}-witness.npz"
        np.savez_compressed(witness, states__F_Biceps=np.array([[force, force]]),
                            metadata__json=np.asarray(json.dumps({"independent_audit": audit})))
        document = {"schema": "slow-phi-rho-oracle-v1", "sample_id": name, "family_id": family or name,
                    "partition": partition, "side": "left", "model_sha256": "a"*64, "task_sha256": "b"*64,
                    "state_names": states, "slow_state": slow, "fast_names": ["Cn_Biceps"],
                    "fast_state": [fast], "full_fast_state_declared": True,
                    "fast_sources": [{"key": "fast", "row": 0, "column": 0}],
                    "command_names": ["torque_nm", "weight:Biceps"], "command": [torque, 1.],
                    "command_sources": [{"key": "commands", "row": 0, "column": i, "scale": 1.} for i in (0, 1)],
                    "outcome": outcome, "prepared_archive": str(prepared), "prepared_archive_sha256": digest(prepared),
                    "audit_tolerance": 1e-5, "independent_audit": audit,
                    "witness": str(witness), "witness_sha256": digest(witness),
                    "force_integrals_n_s": [force], "force_quadrature_weights": [.5, .5]}
        if outcome == "physiological_failure_certified":
            failure = {"scope": "protocol_conditioned_local_frozen_endpoint",
                       "prepared_archive_sha256": digest(prepared), "zero_objective_status": "Infeasible_Problem_Detected",
                       "zero_objective_feasible_witness": False, "fatigue_rest_audit": audit,
                       "fatigue_rest_witness": str(witness), "fatigue_rest_witness_sha256": digest(witness)}
            report = tmp_path / f"{name}-failure.json"
            report.write_text(json.dumps(failure))
            document["failure_certificate"] = {**failure, "report": str(report), "report_sha256": digest(report)}
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(document))
        return load_phi_point(path)
    return make


def test_interpolation_remains_estimate_and_passes_independent_holdout(point_factory):
    phi = ConditionalPhiMap([point_factory("low", 2000), point_factory("high", 4000)])
    estimate = phi.predict([3000, .06, .14], [1., 1.], [.4])
    assert estimate.outcome == "estimated_feasible"
    assert estimate.force_integrals_n_s == pytest.approx((30.,))
    assert estimate.certified is False
    assert sum(estimate.weights) == pytest.approx(1.)
    gate = phi.validate_holdout([point_factory("held", 3000, partition="holdout")])
    assert gate["passed"] is True
    assert gate["coverage"] == 1.


def test_slow_extrapolation_and_fast_covariates_are_rejected(point_factory):
    phi = ConditionalPhiMap([point_factory("low", 2000), point_factory("high", 4000)])
    with pytest.raises(PhiOutOfSupport, match="convex"):
        phi.predict([5000, .06, .14], [1., 1.], [.4])
    with pytest.raises(PhiOutOfSupport, match="Fast"):
        phi.predict([3000, .06, .14], [1., 1.], [.41])
    with pytest.raises(PhiOutOfSupport, match="convex"):
        phi.predict([3000, .07, .14], [1., 1.], [.4])


def test_joint_convex_support_is_stricter_than_component_ranges(point_factory):
    phi = ConditionalPhiMap([point_factory("a", 2000, torque=1.), point_factory("b", 4000, torque=1.),
                             point_factory("c", 2000, torque=2.)])
    with pytest.raises(PhiOutOfSupport):
        phi.predict([4000, .06, .14], [2., 1.], [.4])


@pytest.mark.parametrize("outcome", ["physiological_failure_certified", "numerical_failure", "right_censored"])
def test_mixed_or_unidentified_outcomes_abstain(point_factory, outcome):
    phi = ConditionalPhiMap([point_factory("low", 2000), point_factory("high", 4000, outcome=outcome)])
    assert phi.predict([3000, .06, .14], [1., 1.], [.4]).outcome == "abstain"


def test_false_feasible_and_force_error_fail_holdout(point_factory):
    phi = ConditionalPhiMap([point_factory("low", 2000), point_factory("high", 4000)])
    failed = point_factory("held-failed", 3000, partition="holdout", outcome="physiological_failure_certified")
    assert phi.validate_holdout([failed])["false_feasible_rate"] == 1.
    assert phi.validate_holdout([failed])["passed"] is False
    wrong_force = point_factory("held-force", 3000, partition="holdout", force=40.)
    assert phi.validate_holdout([wrong_force])["passed"] is False


def test_family_leak_and_file_mutation_are_rejected(point_factory):
    low, high = point_factory("low", 2000), point_factory("high", 4000)
    phi = ConditionalPhiMap([low, high])
    with pytest.raises(ValueError, match="leaks"):
        phi.validate_holdout([point_factory("held", 3000, partition="holdout", family="low")])
    from pathlib import Path
    Path(low.artifacts[0][0]).write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        phi.predict([3000, .06, .14], [1., 1.], [.4])


@pytest.mark.parametrize("field,value", [("full_fast_state_declared", False), ("force_integrals_n_s", [999.]),
                                         ("fast_state", [.9]), ("command", [2., 1.]), ("slow_state", [1000., .06, .14])])
def test_receipt_coordinates_and_response_must_match_archives(point_factory, field, value):
    from pathlib import Path
    point = point_factory("source", 2000)
    path = Path(point.artifacts[0][0])
    document = json.loads(path.read_text())
    document[field] = value
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        load_phi_point(path)
