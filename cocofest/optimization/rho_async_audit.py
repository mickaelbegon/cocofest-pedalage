"""Experimental numerical RHO certification in a persistent spawn process.

Solving a descendant may proceed speculatively. Releasing a command requires
``certify_before_release`` for its cycle, in order. No OCP, CasADi object, or
actuator callback crosses the process boundary. This module does not alter
the production runner's synchronous certification policy.
"""
from __future__ import annotations

import multiprocessing
import time
from queue import Empty, Full

import numpy as np


class RhoAuditFailure(RuntimeError):
    """A speculative branch cannot be released or used for actuation."""


def _numeric_copy(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf":
            raise TypeError("Audit arrays must be real numeric arrays")
        copied = np.array(value, copy=True)
        copied.setflags(write=False)
        return copied
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Audit metadata keys must be strings")
        return {key: _numeric_copy(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_numeric_copy(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported audit payload type: {type(value).__name__}")


def _bound_violation(values, lower, upper):
    values = np.asarray(values, dtype=float).reshape(-1)
    lower = np.broadcast_to(np.asarray(lower, dtype=float).reshape(-1), values.shape)
    upper = np.broadcast_to(np.asarray(upper, dtype=float).reshape(-1), values.shape)
    if not np.all(np.isfinite(values)) or np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
        return float("inf")
    if np.any(lower > upper):
        return float("inf")
    return float(max(0.0, np.max(lower - values, initial=0), np.max(values - upper, initial=0)))


def synchronous_command_checks(snapshot):
    """Cheap pre-transfer gates; they do not replace full NLP certification.

Run these before using a candidate's terminal state/PW as a speculative seed.
Seam references must come from the immediate parent snapshot, never mutable OCP
bounds already shifted to the next RHO. Actuation still requires the barrier.
"""
    tolerance = float(snapshot["threshold"])
    pw_tolerance = float(snapshot.get("pw_tolerance", 1e-10))
    states = np.asarray(snapshot["states"], dtype=float)
    controls = np.asarray(snapshot["controls"], dtype=float)
    if states.ndim != 2 or controls.ndim != 2 or not states.size or not controls.size:
        raise ValueError("Nonempty 2D state and control snapshots are required")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(controls)):
        raise RhoAuditFailure("Nonfinite trajectory")
    if _bound_violation(snapshot["x"], snapshot["lbx"], snapshot["ubx"]) > tolerance:
        raise RhoAuditFailure("Decision bounds violated")
    if _bound_violation(controls, snapshot["pw_lower"], snapshot["pw_upper"]) > pw_tolerance:
        raise RhoAuditFailure("PW bounds violated")
    previous_pw = snapshot.get("previous_pw")
    joined = controls if previous_pw is None else np.column_stack((previous_pw, controls))
    max_step = snapshot.get("pw_max_step")
    if max_step is not None and np.max(np.abs(np.diff(joined, axis=1)), initial=0) > max_step + pw_tolerance:
        raise RhoAuditFailure("PW difference bound violated")
    previous_state = snapshot.get("previous_state")
    if previous_state is not None and not np.allclose(
        states[:, 0], previous_state, rtol=0, atol=float(snapshot.get("state_seam_tolerance", tolerance))
    ):
        raise RhoAuditFailure("State seam mismatch")


def audit_numeric_snapshot(snapshot):
    """Independent bounds/residual audit on the frozen numerical NLP output."""
    start = time.perf_counter()
    synchronous_command_checks(snapshot)
    if snapshot.get("g") is None:
        raise RhoAuditFailure("Constraint vector unavailable")
    constraint_violation = _bound_violation(snapshot["g"], snapshot["lbg"], snapshot["ubg"])
    decision_violation = _bound_violation(snapshot["x"], snapshot["lbx"], snapshot["ubx"])
    inf_pr = snapshot.get("inf_pr")
    if inf_pr is None:
        inf_pr = 0.0  # independent g and x checks remain mandatory
    if not np.isfinite(inf_pr):
        raise RhoAuditFailure("Nonfinite solver primal residual")
    maximum = max(constraint_violation, decision_violation, abs(float(inf_pr)))
    passed = int(snapshot["status"]) == 0 and maximum <= float(snapshot["threshold"])
    return {
        "cycle": int(snapshot["cycle"]), "passed": passed,
        "constraint_violation": constraint_violation,
        "decision_violation": decision_violation,
        "maximum_primal_violation": maximum,
        "worker_audit_s": time.perf_counter() - start,
    }


def _worker_loop(requests, responses):
    responses.put({"ready": True})
    while True:
        snapshot = requests.get()
        if snapshot is None:
            return
        try:
            result = audit_numeric_snapshot(snapshot)
        except Exception as error:
            result = {"cycle": snapshot["cycle"], "passed": False,
                      "error": f"{type(error).__name__}: {error}"}
        responses.put(result)


class AsyncRhoAudit:
    """Persistent, bounded worker with ordered fail-closed release barriers.

The maximum speculative depth is ``max_pending``. A failed audit poisons all
descendants. ``last_certified_checkpoint`` remains available for restart from
the most recently released cycle; callers must discard speculative warm starts.
Startup belongs before the online RHO timer. No child process can actuate.
"""
    def __init__(self, *, max_pending=2, startup_timeout=20.0):
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        context = multiprocessing.get_context("spawn")
        self._requests = context.Queue(maxsize=max_pending)
        self._responses = context.Queue(maxsize=max_pending + 1)
        self._process = context.Process(target=_worker_loop, args=(self._requests, self._responses), daemon=True)
        self._max_pending = int(max_pending)
        self._pending = {}
        self._completed = {}
        self._last_submitted = 0
        self._last_released = 0
        self._failed = None
        self._closed = False
        self.last_certified_checkpoint = None
        self._process.start()
        try:
            if self._responses.get(timeout=startup_timeout) != {"ready": True}:
                raise RhoAuditFailure("Unexpected worker handshake")
        except Empty as error:
            self.close()
            raise RhoAuditFailure("Audit worker startup timed out") from error

    def submit(self, snapshot):
        if self._closed or self._failed is not None:
            raise RhoAuditFailure("Audit pipeline is closed or failed")
        frozen = _numeric_copy(snapshot)
        cycle = frozen["cycle"]
        if cycle != self._last_submitted + 1:
            raise ValueError("Submit cycles consecutively, starting at one")
        if len(self._pending) >= self._max_pending:
            raise RhoAuditFailure("Speculative depth exhausted: certify a parent first")
        synchronous_command_checks(frozen)
        try:
            self._requests.put_nowait(frozen)
        except Full as error:
            raise RhoAuditFailure("Audit queue full") from error
        self._pending[cycle] = frozen
        self._last_submitted = cycle

    def certify_before_release(self, cycle, *, timeout=10.0):
        if self._failed is not None:
            raise RhoAuditFailure(self._failed)
        if cycle != self._last_released + 1 or cycle not in self._pending:
            raise ValueError("Only the next submitted cycle can cross the release barrier")
        deadline = time.monotonic() + timeout
        while cycle not in self._completed:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._failed = f"Certification timeout for cycle {cycle}; discard descendants"
                raise RhoAuditFailure(self._failed)
            try:
                result = self._responses.get(timeout=min(remaining, 0.1))
            except Empty:
                if not self._process.is_alive():
                    self._failed = "Audit worker exited; discard speculative descendants"
                    raise RhoAuditFailure(self._failed)
                continue
            if result["cycle"] not in self._pending:
                self._failed = "Unexpected audit result cycle"
                raise RhoAuditFailure(self._failed)
            self._completed[result["cycle"]] = result
        result = self._completed.pop(cycle)
        if not result["passed"]:
            self._failed = f"Audit failed for cycle {cycle}; discard speculative descendants: {result}"
            raise RhoAuditFailure(self._failed)
        self.last_certified_checkpoint = self._pending.pop(cycle)
        self._last_released = cycle
        return result

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._requests.put_nowait(None)
        except Full:
            pass
        self._process.join(timeout=0.5)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=0.5)
        self._requests.cancel_join_thread()
        self._responses.cancel_join_thread()
        self._requests.close()
        self._responses.close()
        self._process.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()
