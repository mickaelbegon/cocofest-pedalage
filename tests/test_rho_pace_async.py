from queue import Queue
import time

import pytest

from cocofest.optimization.rho_pace import RhoPaceConfig, RhoPaceController
from cocofest.optimization import rho_pace_async
from cocofest.optimization.rho_pace_async import AsyncPaceWorker


class FakeQueue(Queue):
    def cancel_join_thread(self):
        pass

    def close(self):
        pass


class FakeProcess:
    def __init__(self, **kwargs):
        self.alive = False
        self.terminated = False

    def start(self):
        self.alive = True

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.alive = False
        self.terminated = True

    def join(self, timeout):
        assert timeout == 0, "RHO must never wait for the projection"

    def close(self):
        pass


class FakeContext:
    Queue = FakeQueue
    Process = FakeProcess


def worker_fixture():
    now = [100.]
    worker = AsyncPaceWorker(None, horizon_cycles=100, budget_seconds=16., max_age_cycles=20,
                             context=FakeContext(), clock=lambda: now[0])
    worker.submit({}, cycle_index=20, incumbent_weights=(1., 1.))
    return worker, now


def test_busy_projection_is_nonblocking_and_never_queues_backlog():
    worker, _ = worker_fixture()
    assert worker.poll(cycle_index=21, incumbent_weights=(1., 1.)) == (None, None)
    assert not worker.submit({}, cycle_index=40, incumbent_weights=(1., 1.))
    worker.close()
    assert not worker.busy


def test_deadline_terminates_worker_and_reduces_next_horizon():
    worker, now = worker_fixture()
    process = worker.process
    now[0] = 117.
    weights, audit = worker.poll(cycle_index=30, incumbent_weights=(1., 1.))
    assert weights is None
    assert audit["discard_reason"] == "projection_deadline_exceeded"
    assert process.terminated
    assert worker.horizon_cycles == 50


@pytest.mark.parametrize("cycle,weights,completed_at,reason", [
    (41, (1., 1.), 104., "stale_projection"),
    (24, (2., .5), 104., "incumbent_weights_changed"),
    (24, (1., 1.), 117., "projection_deadline_exceeded"),
    (20, (1., 1.), 104., "stale_projection"),
])
def test_late_or_wrong_version_results_never_reach_ocp(cycle, weights, completed_at, reason):
    worker, _ = worker_fixture()
    worker.results.put(((2., .5), {"has_actionable_evidence": True}, completed_at))
    proposed, audit = worker.poll(cycle_index=cycle, incumbent_weights=weights)
    assert proposed is None
    assert audit["discard_reason"] == reason
    worker.close()


def test_ready_proposal_applies_at_next_certified_boundary_between_slow_calls():
    worker, _ = worker_fixture()
    worker.results.put(((1.05, 1 / 1.05), {"has_actionable_evidence": True,
                                        "selection_basis": "guarded_fatigue_improvement"}, 104.))
    weights, audit = worker.poll(cycle_index=24, incumbent_weights=(1., 1.))
    config = RhoPaceConfig(adaptation_strategy="predictive_moment", update_every_cycles=20)
    controller = RhoPaceController(("a", "b"), (1., 1.), signed_crank_torque_nm=.15,
                                   parameters={"a": {}, "b": {}}, initial_weight_basis="fixture",
                                   config=config)
    writer = lambda _: {"ocp_cost_updated": True}
    controller.boundary(0, (1., 1.), certified=True, signed_crank_torque_nm=.15, apply_weights=writer)
    refused = controller.boundary(23, (.9, .95), certified=False, signed_crank_torque_nm=.15,
                                  apply_weights=writer, predictive_weights=weights, predictive_audit=audit)
    assert refused["status"] == "refused"
    assert controller.weights == (1., 1.)
    applied = controller.boundary(24, (.9, .95), certified=True, signed_crank_torque_nm=.15,
                                  apply_weights=writer, predictive_weights=weights, predictive_audit=audit)
    assert applied["status"] == "applied"
    assert controller.weights == pytest.approx((1.05, 1 / 1.05))
    worker.close()


def test_effective_budget_cannot_exceed_call_period():
    assert RhoPaceConfig(projection_budget_seconds=600).effective_projection_budget_seconds == 16.
    assert RhoPaceConfig(projection_budget_seconds=3).effective_projection_budget_seconds == 3.
    assert RhoPaceConfig(update_every_cycles=5, target_cycle_seconds=.5).effective_projection_budget_seconds == 2.
    with pytest.raises(ValueError, match="projection_budget_fraction"):
        RhoPaceConfig(projection_budget_fraction=1.1)


def test_worker_retains_requested_cpu_affinity_in_process_configuration():
    worker = AsyncPaceWorker(None, horizon_cycles=100, budget_seconds=16., max_age_cycles=20,
                             cpu_ids=(4,), context=FakeContext())
    assert worker.cpu_ids == (4,)
    worker.submit({}, cycle_index=20, incumbent_weights=(1., 1.))
    assert worker.process is not None
    worker.close()


def test_child_sets_requested_affinity_before_evaluating(monkeypatch):
    applied = []
    monkeypatch.setattr(rho_pace_async.os, "sched_setaffinity",
                        lambda pid, cpus: applied.append((pid, cpus)))
    monkeypatch.setattr(rho_pace_async.os, "nice", lambda _: None)
    results = Queue()
    rho_pace_async._evaluate_in_process(
        lambda _: ((1.,), {"finished": True}), {}, results, cpu_ids=(4, 5),
    )
    weights, audit, _ = results.get_nowait()
    assert applied == [(0, {4, 5})]
    assert weights == (1.,)
    assert audit == {"finished": True}


def _slow_evaluator(payload):
    time.sleep(payload["delay"])
    return (2., .5), {"worker_finished": True}


def test_real_process_slow_evaluator_does_not_block_rho():
    worker = AsyncPaceWorker(_slow_evaluator, horizon_cycles=100, budget_seconds=10, max_age_cycles=20)
    started = time.monotonic()
    worker.submit({"delay": 5.}, cycle_index=20, incumbent_weights=(1., 1.))
    for cycle in range(21, 40):
        assert worker.poll(cycle_index=cycle, incumbent_weights=(1., 1.)) == (None, None)
    worker.close()
    assert time.monotonic() - started < 1., "Nineteen RHO boundaries must not await the 5-second rollout"


def test_real_process_result_is_received_and_audited():
    worker = AsyncPaceWorker(_slow_evaluator, horizon_cycles=100, budget_seconds=10, max_age_cycles=20)
    worker.submit({"delay": .01}, cycle_index=20, incumbent_weights=(1., 1.))
    deadline = time.monotonic() + 8
    try:
        while time.monotonic() < deadline:
            weights, audit = worker.poll(cycle_index=21, incumbent_weights=(1., 1.))
            if audit is not None:
                assert weights == (2., .5)
                assert audit["worker_finished"]
                assert audit["snapshot_age_cycles"] == 1
                break
            time.sleep(.02)
        else:
            pytest.fail("Worker result did not arrive")
    finally:
        worker.close()
