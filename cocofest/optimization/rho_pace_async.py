"""A bounded, disposable PACE process; the RHO thread never waits for a rollout.

Only immutable numerical snapshots enter this worker. The child has no OCP
writer; polling validates the incumbent version and snapshot age before the
main thread can apply a result at its next certified cycle boundary.
"""

from dataclasses import dataclass
import math
import multiprocessing
import os
from queue import Empty
import time


def _evaluate_in_process(evaluator, payload, results, cpu_ids=None):
    try:
        if cpu_ids is not None:
            if not hasattr(os, "sched_setaffinity"):
                raise RuntimeError("CPU affinity was requested but is unavailable on this platform")
            os.sched_setaffinity(0, set(cpu_ids))
        if hasattr(os, "nice"):
            os.nice(5)
        weights, audit = evaluator(payload)
        results.put((weights, audit, time.monotonic()))
    except Exception as error:
        results.put((None, {"error": f"{type(error).__name__}: {error}"}, time.monotonic()))


@dataclass(frozen=True)
class ProjectionRequest:
    source_cycle: int
    incumbent_weights: tuple
    submitted_at: float
    horizon_cycles: int
    budget_seconds: float


class AsyncPaceWorker:
    """One job at a time, no backlog and no blocking joins in the RHO loop.

    The budget includes process startup and input transport. Expired work is
    terminated on the next poll, and a late result is never applied. Thus the
    deadline is an acceptance deadline, not a hard OS scheduling guarantee.
    """

    def __init__(self, evaluator, *, horizon_cycles, budget_seconds, max_age_cycles,
                 cpu_ids=None, context=None, clock=time.monotonic):
        self.evaluator = evaluator
        self.max_horizon_cycles = int(horizon_cycles)
        self.horizon_cycles = int(horizon_cycles)
        self.budget_seconds = float(budget_seconds)
        self.max_age_cycles = int(max_age_cycles)
        self.cpu_ids = None if cpu_ids is None else tuple(cpu_ids)
        if (self.max_horizon_cycles < 1 or not math.isfinite(self.budget_seconds)
                or self.budget_seconds <= 0 or self.max_age_cycles < 1):
            raise ValueError("Worker horizon, budget and maximum age must be positive")
        if self.cpu_ids is not None and (not self.cpu_ids or any(
                isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0 for cpu in self.cpu_ids)):
            raise ValueError("Worker CPU ids must be nonnegative integers")
        self.context = context or multiprocessing.get_context("spawn")
        self.clock = clock
        self.request = None
        self.process = None
        self.results = None
        self.retired = []

    @property
    def busy(self):
        return self.request is not None

    def submit(self, payload, *, cycle_index, incumbent_weights):
        if self.busy:
            return False
        self._reap()
        self.request = ProjectionRequest(
            int(cycle_index), tuple(incumbent_weights), self.clock(),
            self.horizon_cycles, self.budget_seconds,
        )
        self.results = self.context.Queue(maxsize=1)
        self.process = self.context.Process(
            target=_evaluate_in_process,
            # A supervisor may itself fan out independent arm projections.
            # It remains disposable and is always terminated by ``close``.
            args=(self.evaluator, payload, self.results, self.cpu_ids), daemon=False,
        )
        try:
            self.process.start()
        except Exception:
            self.results.cancel_join_thread()
            self.results.close()
            self.process.close()
            self.process = self.results = self.request = None
            raise
        return True

    def _reap(self):
        alive = []
        for process in self.retired:
            process.join(timeout=0)
            if process.is_alive():
                alive.append(process)
            else:
                process.close()
        self.retired = alive

    def _retire(self, *, terminate=False):
        if terminate and self.process.is_alive():
            self.process.terminate()
        self.retired.append(self.process)
        self.results.cancel_join_thread()
        self.results.close()
        self.process = self.results = self.request = None
        self._reap()

    def poll(self, *, cycle_index, incumbent_weights):
        """Return (weights, audit), or (None, None) while the child is working."""
        self._reap()
        if not self.busy:
            return None, None
        request = self.request
        now = self.clock()
        age = int(cycle_index) - request.source_cycle
        audit = {
            "asynchronous": True, "source_cycle_index": request.source_cycle,
            "application_cycle_index": int(cycle_index), "snapshot_age_cycles": age,
            "snapshot_incumbent_weights": request.incumbent_weights,
            "horizon_cycles": request.horizon_cycles,
            "budget_seconds": request.budget_seconds,
        }
        try:
            weights, details, completed_at = self.results.get_nowait()
        except Empty:
            if now - request.submitted_at <= request.budget_seconds and age <= self.max_age_cycles:
                if self.process.is_alive():
                    return None, None
                reason = "projection_worker_failed"
            else:
                reason = "projection_deadline_exceeded" if age <= self.max_age_cycles else "stale_projection"
            self.horizon_cycles = max(1, request.horizon_cycles // 2)
            self._retire(terminate=True)
            return None, {**audit, "discard_reason": reason, "elapsed_s": now - request.submitted_at,
                          "next_horizon_cycles": self.horizon_cycles}
        elapsed = completed_at - request.submitted_at
        reason = None
        if elapsed > request.budget_seconds:
            reason = "projection_deadline_exceeded"
        elif age < 1 or age > self.max_age_cycles:
            reason = "stale_projection"
        elif tuple(incumbent_weights) != request.incumbent_weights:
            reason = "incumbent_weights_changed"
        # Target 70% of the deadline. Shrink quickly; grow by at most 25%.
        desired = int(request.horizon_cycles * .7 * request.budget_seconds / max(elapsed, 1e-6))
        self.horizon_cycles = max(1, min(self.max_horizon_cycles, desired,
                                        max(request.horizon_cycles + 1, int(1.25 * request.horizon_cycles))))
        self._retire()
        audit = {**details, **audit, "worker_wall_seconds": elapsed,
                 "next_horizon_cycles": self.horizon_cycles}
        if reason:
            audit["discard_reason"] = reason
            weights = None
        return weights, audit

    def close(self):
        """Cancel work without waiting for an outstanding projection."""
        if self.busy:
            self._retire(terminate=True)
        for process in self.retired:
            if process.is_alive():
                process.terminate()
        self._reap()
