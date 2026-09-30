import numpy as np
import pytest

from cocofest.optimization.rho_async_audit import (
    AsyncRhoAudit, RhoAuditFailure, audit_numeric_snapshot,
)


def snapshot(cycle=1):
    return dict(cycle=cycle, status=0, threshold=1e-6,
                x=np.array([0.1, 0.2]), lbx=np.zeros(2), ubx=np.ones(2),
                g=np.zeros(2), lbg=np.zeros(2), ubg=np.zeros(2), inf_pr=1e-10,
                states=np.array([[0.1, 0.2]]), controls=np.array([[0.0002, 0.00025]]),
                pw_lower=0.000131405, pw_upper=0.0006, pw_max_step=0.0001)


def test_numeric_audit_rejects_constraints_and_slew():
    data = snapshot()
    assert audit_numeric_snapshot(data)["passed"]
    data["g"][1] = 1e-3
    assert not audit_numeric_snapshot(data)["passed"]
    data["controls"][0, 1] = 0.0004
    with pytest.raises(RhoAuditFailure, match="difference"):
        audit_numeric_snapshot(data)


def test_pretransfer_state_and_pw_seam_checks():
    data = snapshot()
    data["previous_state"] = np.array([0.5])
    with pytest.raises(RhoAuditFailure, match="State seam"):
        audit_numeric_snapshot(data)
    data["previous_state"] = np.array([0.1])
    data["previous_pw"] = np.array([0.0005])
    with pytest.raises(RhoAuditFailure, match="difference"):
        audit_numeric_snapshot(data)


def test_persistent_worker_immutable_snapshots_and_ordered_release():
    with AsyncRhoAudit() as worker:
        first = snapshot()
        worker.submit(first)
        first["g"][:] = 100  # mutation after submission must not alter worker input
        worker.submit(snapshot(2))
        with pytest.raises(ValueError, match="next"):
            worker.certify_before_release(2)
        assert worker.certify_before_release(1)["passed"]
        pid = worker._process.pid
        assert worker.certify_before_release(2)["passed"]
        worker.submit(snapshot(3))
        assert worker.certify_before_release(3)["passed"]
        assert worker._process.pid == pid
        assert worker.last_certified_checkpoint["cycle"] == 3


def test_late_failure_poisoning_and_last_certified_checkpoint():
    with AsyncRhoAudit() as worker:
        worker.submit(snapshot())
        worker.certify_before_release(1)
        invalid = snapshot(2)
        invalid["g"][0] = 0.1
        worker.submit(invalid)
        worker.submit(snapshot(3))  # speculative solve completed before cycle2 audit
        with pytest.raises(RhoAuditFailure, match="cycle 2"):
            worker.certify_before_release(2)
        with pytest.raises(RhoAuditFailure):
            worker.certify_before_release(3)
        assert worker.last_certified_checkpoint["cycle"] == 1


def test_reject_objects_worker_death_and_unbounded_speculation():
    with AsyncRhoAudit(max_pending=1) as worker:
        invalid = snapshot()
        invalid["ocp"] = object()
        with pytest.raises(TypeError, match="Unsupported"):
            worker.submit(invalid)
        worker.submit(snapshot())
        with pytest.raises(RhoAuditFailure, match="depth"):
            worker.submit(snapshot(2))
        worker.certify_before_release(1)
        worker._process.terminate()
        worker._process.join()
        worker.submit(snapshot(2))
        with pytest.raises(RhoAuditFailure, match="exited"):
            worker.certify_before_release(2)


def test_timeout_cannot_be_followed_by_release():
    with AsyncRhoAudit() as worker:
        worker.submit(snapshot())
        with pytest.raises(RhoAuditFailure, match="timeout"):
            worker.certify_before_release(1, timeout=0)
        with pytest.raises(RhoAuditFailure):
            worker.certify_before_release(1)
        assert worker.last_certified_checkpoint is None
