"""Nonblocking bilateral PACE-VR transport and experimental weight adapter.

The adapter is deliberately named: fitting a rollout value and converting its
gradient to positive fatigue weights is not injecting that terminal value in
the RHO. The existing fixed-parameter weight graph is reused without rebuild.
"""
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
import multiprocessing
import os
import math
from queue import Empty, Full
import time

import numpy as np

from .pace_vr import PaceVrConfig, PaceVrSnapshot, evaluate_pace_vr_weight_candidates, run_pace_vr_snapshot
from .pace_rt_ocp import PACE_RT_MODE
from .rho_pace_async import AsyncPaceWorker


APPLICATION_MODE = "derived_fatigue_weights_experimental"
APPLICATION_MODES = frozenset((APPLICATION_MODE, PACE_RT_MODE))
_PROCESS_SUPERVISOR_CACHE = {}


def validate_pace_vr_configuration(payload):
    declared = payload.get("experimental_pace_vr")
    if not declared or declared.get("enabled") is not True:
        return None
    config = dict(declared)
    if config.get("asynchronous") is not True:
        raise ValueError("experimental_pace_vr requires asynchronous=true; synchronous RHO blocking is forbidden.")
    if config.get("application_mode") not in APPLICATION_MODES:
        raise ValueError("PACE-VR terminal-value binding is not wired for this application_mode.")
    if payload.get("parametric_fatigue_weights") is not True:
        raise ValueError("PACE-VR weight adapter requires parametric_fatigue_weights=true for compiled NLP reuse.")
    if payload.get("muscle_pace", {}).get("adaptation_enabled", True) is not False:
        raise ValueError("PACE-VR weight adapter requires disabled competing muscle-weight feedback.")
    if float(payload.get("experimental_mechanical_reserve_weight", 0.)) > 0:
        raise ValueError("PACE-VR weight adapter cannot coexist with the mechanical-reserve parameter graph.")
    if config["application_mode"] == PACE_RT_MODE:
        if not np.isfinite(config.get("terminal_proximal_weight", .03)) or config.get("terminal_proximal_weight", .03) < 0:
            raise ValueError("PACE-RT terminal_proximal_weight must be finite and nonnegative.")
        if not np.isfinite(config.get("target_fraction", .25)) or not 0 <= config.get("target_fraction", .25) <= 1:
            raise ValueError("PACE-RT target_fraction must lie in [0, 1].")
        if not np.isfinite(config.get("terminal_shortage_weight", 1.)) or config.get("terminal_shortage_weight", 1.) < 0:
            raise ValueError("PACE-RT terminal_shortage_weight must be finite and nonnegative.")
        config.setdefault("normalize_terminal_shortage", False)
        if type(config["normalize_terminal_shortage"]) is not bool:
            raise ValueError("PACE-RT normalize_terminal_shortage must be a boolean.")
        config.setdefault("enforce_terminal_target_constraint", False)
        if type(config["enforce_terminal_target_constraint"]) is not bool:
            raise ValueError("PACE-RT enforce_terminal_target_constraint must be a boolean.")
        config.setdefault("lexicographic_reserve_priority", False)
        if type(config["lexicographic_reserve_priority"]) is not bool:
            raise ValueError("PACE-RT lexicographic_reserve_priority must be a boolean.")
        config.setdefault("lexicographic_reserve_epsilon", 1e-4)
        epsilon = config["lexicographic_reserve_epsilon"]
        if (isinstance(epsilon, bool) or not isinstance(epsilon, (int, float))
                or not np.isfinite(epsilon) or epsilon < 0):
            raise ValueError("PACE-RT lexicographic_reserve_epsilon must be finite and nonnegative.")
        if config["lexicographic_reserve_priority"] and not config["enforce_terminal_target_constraint"]:
            raise ValueError("PACE-RT reserve priority requires the compiled terminal target constraint.")
        if config["lexicographic_reserve_priority"]:
            primary_weights = payload.get("fatigue_weight_values")
            if (not isinstance(primary_weights, (list, tuple)) or not primary_weights
                    or any(float(value) != 0. for value in primary_weights)):
                raise ValueError("PACE-RT reserve priority requires zero fatigue_weight_values for the primary solve.")
    config.setdefault("horizon_cycles", 20)
    ladder = config.get("horizon_ladder_cycles", (config["horizon_cycles"],))
    if isinstance(ladder, (str, bytes)):
        raise ValueError("PACE-VR horizon_ladder_cycles must be a sequence of integer horizons.")
    try:
        horizons = tuple(ladder)
    except TypeError as error:
        raise ValueError("PACE-VR horizon_ladder_cycles must be a sequence of integer horizons.") from error
    if (not horizons or any(type(value) is not int or value < 1 for value in horizons)
            or tuple(sorted(set(horizons), reverse=True)) != horizons):
        raise ValueError("PACE-VR horizons must be distinct positive integers in decreasing order.")
    # A bilateral rollout owns two processes.  A ladder gives each horizon its
    # own pair, so no candidate steals CPU time from another candidate or RHO.
    declared_cpus = config.get("supervisor_cpu_ids", (config.get("supervisor_cpu_id"),))
    if isinstance(declared_cpus, (str, bytes)):
        raise ValueError("PACE-VR supervisor_cpu_ids must be a sequence of CPU ids.")
    try:
        cpus = tuple(declared_cpus)
    except TypeError as error:
        raise ValueError("PACE-VR supervisor_cpu_ids must be a sequence of CPU ids.") from error
    if not cpus or any(isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0 for cpu in cpus):
        raise ValueError("PACE-VR requires nonnegative supervisor CPU ids.")
    if len(cpus) != 2 * len(horizons) or len(set(cpus)) != len(cpus):
        raise ValueError("PACE-VR requires two distinct supervisor CPUs per rollout horizon.")
    config["supervisor_cpu_ids"] = cpus
    config["supervisor_cpu_id"] = cpus[0]
    config["horizon_ladder_cycles"] = horizons
    config.setdefault("persistent_supervisor", False)
    if type(config["persistent_supervisor"]) is not bool:
        raise ValueError("PACE-VR persistent_supervisor must be a boolean.")
    if config["persistent_supervisor"] and len(horizons) != 1:
        raise ValueError("PACE-VR persistent_supervisor currently requires one rollout horizon and two arm CPUs.")
    affinity = payload.get("solver_cpu_affinity")
    if not isinstance(affinity, dict) or set(affinity) != {"right", "left"}:
        raise ValueError("PACE-VR requires explicit right/left solver_cpu_affinity for CPU isolation.")
    if set(cpus) & set(affinity.values()):
        raise ValueError("PACE-VR supervisor CPU must be distinct from both RHO CPUs.")
    config.setdefault("update_every_cycles", 20)
    config.setdefault("deadline_seconds", 16.)
    config.setdefault("maximum_log_weight_step", math.log(1.25))
    # For complete rollouts, score gain equals tanh(margin) gain divided by H.
    # Express the threshold in normalized margin units, invariant across H.
    if "candidate_minimum_improvement" in config:
        if "candidate_minimum_margin_improvement" in config:
            raise ValueError("Specify only one PACE-VR candidate improvement threshold.")
        config["candidate_minimum_margin_improvement"] = (
            float(config["candidate_minimum_improvement"]) * config["horizon_cycles"])
    config.setdefault("candidate_minimum_margin_improvement", 2e-4)
    config.setdefault("candidate_publication_reserve_seconds", 1.)
    config.setdefault("candidate_fit_budget_fraction", .65)
    config.setdefault("maximum_age_cycles", config["update_every_cycles"])
    config.setdefault("core", {})
    for horizon in horizons:
        PaceVrConfig(horizon_cycles=horizon, **config["core"])
    for name in ("update_every_cycles", "maximum_age_cycles"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"PACE-VR {name} must be a positive integer.")
    for name in ("deadline_seconds", "maximum_log_weight_step"):
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"PACE-VR {name} must be finite and positive.")
    if (not np.isfinite(config["candidate_minimum_margin_improvement"])
            or config["candidate_minimum_margin_improvement"] < 0):
        raise ValueError("PACE-VR candidate_minimum_margin_improvement must be finite and nonnegative.")
    if (not np.isfinite(config["candidate_publication_reserve_seconds"])
            or not 0 < config["candidate_publication_reserve_seconds"] < config["deadline_seconds"]):
        raise ValueError("PACE-VR candidate_publication_reserve_seconds must lie inside the deadline.")
    if (not np.isfinite(config["candidate_fit_budget_fraction"])
            or not 0 < config["candidate_fit_budget_fraction"] < 1):
        raise ValueError("PACE-VR candidate_fit_budget_fraction must lie strictly between zero and one.")
    if config["deadline_seconds"] > config["update_every_cycles"]:
        raise ValueError("PACE-VR deadline_seconds must fit the one-second-per-cycle call period.")
    return config


def derived_fatigue_weights(result, incumbent_weights, maximum_log_step):
    """Bounded relative reweighting from the fitted marginal cost of damage.

    This is an experimental sign/direction adapter, not an exact match between
    the rollout value and the original squared fatigue objective.
    """
    fit = result.get("local_fit") or {}
    if result.get("accepted") is not True or fit.get("accepted") is not True:
        return None
    current = np.asarray(incumbent_weights, float)
    gradient = np.asarray(fit["gradient"], float)
    if gradient.shape != (current.size, 3) or np.any(current <= 0) or not np.all(np.isfinite(gradient)):
        raise ValueError("Invalid local value gradient or incumbent fatigue weights.")
    damage_cost = -gradient[:, 0]
    direction = damage_cost - np.mean(damage_cost)
    scale = float(np.max(np.abs(direction)))
    if scale <= 1e-12:
        return None
    logs = np.log(current) + float(maximum_log_step) * direction / scale
    from .endurance_weight_supervisor import _project_centered_logs
    return np.exp(_project_centered_logs(logs, lower=math.log(.25), upper=math.log(4.))).tolist()


def _choose_verified_weights(base, trials, incumbent, proposed, minimum_margin_improvement):
    """Accept only a feasible replay with a meaningful gain over incumbent.

    The score combines validated prefix length and minimum task margin. It is
    a screening surrogate for endurance, not a predicted failure time.
    """
    incumbent_value = float(base["terminal_value"])
    horizon = int(base["feasible_prefix_cycles"])
    if base["accepted"] is not True or horizon < 1:
        raise ValueError("Candidate screening requires a complete incumbent rollout.")
    score_threshold = float(minimum_margin_improvement) / horizon
    audit = dict(score_kind="reduced_prefix_and_minimum_margin", incumbent_value=incumbent_value,
                 horizon_cycles=horizon,
                 minimum_normalized_margin_improvement=float(minimum_margin_improvement),
                 equivalent_score_improvement=score_threshold, candidates={"incumbent": {
                     "accepted": bool(base["accepted"]), "terminal_value": incumbent_value,
                     "feasible_prefix_cycles": base["feasible_prefix_cycles"],
                     "minimum_task_margin": base["minimum_task_margin"]}})
    weights = _candidate_weight_sets(incumbent, proposed)
    chosen, best_value = None, incumbent_value - score_threshold
    for name in ("proposal", "opposite", "half_step"):
        candidate = trials.get(name)
        if candidate is None:
            audit["candidates"][name] = {"status": "not_evaluated"}
            continue
        audit["candidates"][name] = candidate
        value = float(candidate["terminal_value"])
        if (candidate["accepted"] is True and candidate.get("deadline_met", True) is True
                and np.isfinite(value) and value < best_value):
            chosen, best_value = name, value
    audit["chosen"] = chosen or "incumbent"
    audit["predicted_improvement"] = max(0., incumbent_value - best_value) if chosen else 0.
    if chosen is None:
        audit["reason"] = "no_feasible_candidate_exceeds_improvement_threshold"
    return (weights[chosen] if chosen else None), audit


def _candidate_weight_sets(incumbent, proposed):
    """Bounded local policy directions around the current log weights."""
    from .endurance_weight_supervisor import _project_centered_logs
    current = np.log(np.asarray(incumbent, float))
    step = np.log(np.asarray(proposed, float)) - current
    return {"proposal": np.asarray(proposed, float).tolist(),
            "opposite": np.exp(_project_centered_logs(
                current - step, lower=math.log(.25), upper=math.log(4.))).tolist(),
            "half_step": np.exp(_project_centered_logs(
                current + .5 * step, lower=math.log(.25), upper=math.log(4.))).tolist()}


def _evaluate_pace_vr_side(task):
    """Evaluate one independent arm in a separately pinned child process."""
    if len(task) == 10:  # Backward-compatible direct unit-test / worker payload.
        (side, horizon, snapshot_doc, incumbent, maximum_log_step, minimum_margin_improvement,
         publication_reserve_s, fit_budget_fraction, deadline, cpu_id) = task
        application_mode, terminal_proximal_weight, terminal_shortage_weight = APPLICATION_MODE, .03, 1.
    else:
        (side, horizon, snapshot_doc, incumbent, maximum_log_step, minimum_margin_improvement,
         publication_reserve_s, fit_budget_fraction, deadline, cpu_id, application_mode,
         terminal_proximal_weight, terminal_shortage_weight) = task
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cpu_id})
    source = PaceVrSnapshot(**snapshot_doc)
    payload = json.loads(source.payload_json)
    original_digest = hashlib.sha256(json.dumps(
        payload["context"], allow_nan=False, sort_keys=True).encode()).hexdigest()
    if original_digest != source.context_digest:
        raise ValueError("Original PACE-VR snapshot context digest mismatch.")
    # The incumbent score comes from fit_terminal's base rollout. Its weights
    # must be the same ones used to derive the two proposed policy changes.
    incumbent_weights = np.asarray(incumbent, float)
    snapshot_weights = (np.ones_like(incumbent_weights) if payload["weights"] is None
                        else np.asarray(payload["weights"], float))
    if (snapshot_weights.shape != incumbent_weights.shape
            or not np.allclose(snapshot_weights, incumbent_weights, rtol=0., atol=1e-12)):
        raise ValueError("PACE-VR snapshot and async incumbent weights disagree.")
    payload["context"]["config"]["horizon_cycles"] = horizon
    digest = hashlib.sha256(json.dumps(payload["context"], allow_nan=False, sort_keys=True).encode()).hexdigest()
    publication_deadline = min(source.deadline_monotonic, deadline) - publication_reserve_s
    rollout_snapshot = PaceVrSnapshot(f"{source.request_id}:h{horizon}", source.source_cycle, digest,
                                      publication_deadline,
                                      json.dumps(payload, allow_nan=False, sort_keys=True))
    remaining = max(0., publication_deadline - time.monotonic())
    fit_snapshot = PaceVrSnapshot(rollout_snapshot.request_id, rollout_snapshot.source_cycle,
                                  rollout_snapshot.context_digest,
                                  time.monotonic() + fit_budget_fraction * remaining,
                                  rollout_snapshot.payload_json)
    outcome = run_pace_vr_snapshot(fit_snapshot, supervisor_cache=_PROCESS_SUPERVISOR_CACHE)
    # The horizon is a slow-policy setting, not part of the certified RHO
    # snapshot provenance.  The owner therefore validates the original digest.
    outcome["rollout_context_digest"] = outcome["context_digest"]
    outcome["context_digest"] = source.context_digest
    outcome["request_id"] = source.request_id
    outcome["rollout_horizon_cycles"] = horizon
    proposal = None
    if application_mode == PACE_RT_MODE:
        fit = outcome.get("local_fit") or {}
        if outcome.get("accepted") is True and fit.get("accepted") is True:
            proposal = {"local_fit": fit, "terminal_proximal_weight": float(terminal_proximal_weight),
                        "terminal_shortage_weight": float(terminal_shortage_weight)}
            outcome["candidate_screen"] = {"chosen": "terminal_reserve_target",
                                           "reason": "accepted_local_fit_with_bounded_target"}
        else:
            outcome["candidate_screen"] = {"chosen": "incumbent", "reason": "no_valid_terminal_fit"}
    else:
        proposal = derived_fatigue_weights(outcome, incumbent, maximum_log_step)
    if proposal is not None and application_mode == APPLICATION_MODE:
        incumbent_array = np.asarray(incumbent, float)
        proposed_array = np.asarray(proposal, float)
        candidate_weights = _candidate_weight_sets(incumbent_array, proposed_array)
        fit = outcome["local_fit"]
        estimated_runtime = outcome["runtime_s"] / (1 + fit["sample_count"])
        trials = evaluate_pace_vr_weight_candidates(
            rollout_snapshot, candidate_weights, deadline_monotonic=publication_deadline,
            estimated_candidate_runtime_s=estimated_runtime, supervisor_cache=_PROCESS_SUPERVISOR_CACHE)
        proposal, screen = _choose_verified_weights(
            outcome, trials, incumbent_array, proposed_array, minimum_margin_improvement)
        screen["publication_deadline_monotonic"] = publication_deadline
        screen["fit_deadline_monotonic"] = fit_snapshot.deadline_monotonic
        outcome["candidate_screen"] = screen
    elif application_mode == APPLICATION_MODE:
        outcome["candidate_screen"] = {"chosen": "incumbent", "reason": "no_valid_gradient_proposal"}
    outcome["policy_selected"] = proposal is not None
    outcome["completed_monotonic"] = time.monotonic()
    outcome["deadline_met"] = outcome["completed_monotonic"] <= min(source.deadline_monotonic, deadline)
    return side, horizon, outcome, proposal


def evaluate_bilateral_pace_vr(payload):
    """AsyncPaceWorker-compatible top-level callable: (proposal, audit)."""
    results, proposed = {}, {}
    deadline = time.monotonic() + float(payload["budget_seconds"])
    order = tuple(payload.get("evaluation_order", ("right", "left")))
    if set(order) != {"right", "left"} or len(order) != 2:
        raise ValueError("The bilateral evaluation order must contain each arm once.")
    horizons = tuple(payload.get("horizon_ladder_cycles", (payload["horizon_cycles"],)))
    cpus = tuple(payload.get("supervisor_cpu_ids", ()))
    if len(cpus) != 2 * len(horizons):
        raise ValueError("Bilateral PACE-VR requires two dedicated CPUs per rollout horizon.")
    tasks = [(side, horizon, payload["snapshots"][side], payload["incumbent_weights"][side],
              payload["maximum_log_step"], payload.get("candidate_minimum_margin_improvement", 2e-4),
              payload.get("candidate_publication_reserve_seconds", 1.),
              payload.get("candidate_fit_budget_fraction", .65), deadline,
              cpus[2 * horizon_index + side_index], payload["application_mode"],
              payload.get("terminal_proximal_weight", .03), payload.get("terminal_shortage_weight", 1.))
             for horizon_index, horizon in enumerate(horizons)
             for side_index, side in enumerate(order)]
    # Process isolation is required: the Python/SLSQP path does not reliably
    # release the GIL, whereas two spawned processes were benchmarked faster.
    with ProcessPoolExecutor(max_workers=len(tasks), mp_context=multiprocessing.get_context("spawn")) as pool:
        evaluated = list(pool.map(_evaluate_pace_vr_side, tasks))
    by_side = {side: [] for side in order}
    for side, horizon, outcome, proposal in evaluated:
        by_side[side].append((horizon, outcome, proposal))
        # The full trajectories stay in standalone validation outputs, not IPC.
        results.setdefault("attempts", {}).setdefault(str(horizon), {})[side] = {
            key: value for key, value in outcome.items() if key not in {"state_history", "pulse_widths"}}
    selected_horizons = {}
    for side, candidates in by_side.items():
        candidates.sort(key=lambda value: value[0], reverse=True)
        chosen = next(((horizon, outcome, proposal) for horizon, outcome, proposal in candidates
                       if proposal is not None and outcome["deadline_met"]), None)
        if chosen is not None:
            horizon, outcome, proposal = chosen
            selected_horizons[side] = horizon
            decision = ({"weights": proposal} if payload["application_mode"] == APPLICATION_MODE
                        else {key: value for key, value in proposal.items() if key != "local_fit"})
            proposed[side] = dict(**decision, source_cycle=outcome["source_cycle"],
                                  request_id=outcome["request_id"], context_digest=outcome["context_digest"],
                                  local_fit=outcome["local_fit"], deadline_met=outcome["deadline_met"],
                                  rollout_horizon_cycles=horizon, application_mode=payload["application_mode"])
            results.setdefault("arms", {})[side] = results["attempts"][str(horizon)][side]
        else:
            # Retain the longest audit as the representative outcome when no
            # candidate passed; the complete ladder remains in ``attempts``.
            results.setdefault("arms", {})[side] = results["attempts"][str(candidates[0][0])][side]
    return proposed or None, dict(arms=results["arms"], attempts=results["attempts"], application_mode=payload["application_mode"],
                                  evaluation_order=list(order), horizon_ladder_cycles=list(horizons),
                                  selected_horizon_cycles=selected_horizons,
                                  terminal_value_in_nlp=payload["application_mode"] == PACE_RT_MODE,
                                  uses_fho_data=False)


def _persistent_arm_loop(side, cpu_id, requests, results, latest_sequence, evaluator):
    """One pinned arm process; stale queued requests never enter the rollout."""
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cpu_id})
    if hasattr(os, "nice"):
        os.nice(5)
    # Imports and the CasADi backend are loaded before the first snapshot.
    from .pace_vr import _casadi_qpoases_available
    _casadi_qpoases_available()
    while True:
        item = requests.get()
        if item is None:
            return
        sequence, task, submitted_at = item
        if sequence != latest_sequence.value:
            continue
        started_at = time.monotonic()
        try:
            _, _, outcome, proposal = evaluator(task)
            error = None
        except Exception as exception:
            outcome, proposal = None, None
            error = f"{type(exception).__name__}: {exception}"
        finished_at = time.monotonic()
        results.put_nowait((sequence, side, outcome, proposal, submitted_at,
                            started_at, finished_at, error))


class PersistentBilateralPaceWorker:
    """Two warm arm processes with replaceable, one-slot request mailboxes.

    ``submit`` and ``poll`` only perform nonblocking queue operations. Running
    evaluations may finish, but their sequence is invalidated immediately by
    a newer snapshot and their result cannot reach the RHO owner.
    """

    replaces_pending = True

    def __init__(self, *, horizon_cycles, budget_seconds, max_age_cycles, cpu_ids,
                 context=None, clock=time.monotonic, evaluator=_evaluate_pace_vr_side):
        self.horizon_cycles = int(horizon_cycles)
        self.budget_seconds = float(budget_seconds)
        self.max_age_cycles = int(max_age_cycles)
        self.cpu_ids = tuple(cpu_ids)
        if self.horizon_cycles < 1 or self.budget_seconds <= 0 or self.max_age_cycles < 1:
            raise ValueError("Persistent worker horizon, budget and age must be positive.")
        if len(self.cpu_ids) != 2 or len(set(self.cpu_ids)) != 2:
            raise ValueError("Persistent bilateral worker requires two distinct CPU ids.")
        self.context = context or multiprocessing.get_context("spawn")
        self.clock = clock
        self.results = self.context.Queue()
        self.slots = {}
        self.pending = {}
        self.active = {}
        self.closed = False
        for side, cpu_id in zip(("right", "left"), self.cpu_ids):
            requests = self.context.Queue(maxsize=1)
            sequence = self.context.RawValue("q", 0)
            process = self.context.Process(
                target=_persistent_arm_loop,
                args=(side, cpu_id, requests, self.results, sequence, evaluator),
                daemon=True)
            process.start()
            self.slots[side] = (process, requests, sequence)

    @property
    def busy(self):
        return bool(self.active)

    def _flush_pending(self):
        for side, item in tuple(self.pending.items()):
            _, queue, _ = self.slots[side]
            try:
                queue.put_nowait(item)
            except Full:
                continue
            del self.pending[side]

    def submit(self, payload, *, cycle_index, incumbent_weights):
        if self.closed:
            return False
        now = self.clock()
        order = tuple(payload.get("evaluation_order", ("right", "left")))
        if set(order) != {"right", "left"} or len(order) != 2:
            raise ValueError("The bilateral evaluation order must contain each arm once.")
        if tuple(payload["horizon_ladder_cycles"]) != (self.horizon_cycles,):
            raise ValueError("Persistent worker requires its configured single horizon.")
        deadline = now + float(payload["budget_seconds"])
        for side in order:
            _, _, sequence = self.slots[side]
            next_sequence = sequence.value + 1
            sequence.value = next_sequence
            task = (side, self.horizon_cycles, deepcopy(payload["snapshots"][side]),
                    tuple(payload["incumbent_weights"][side]), payload["maximum_log_step"],
                    payload["candidate_minimum_margin_improvement"],
                    payload["candidate_publication_reserve_seconds"],
                    payload["candidate_fit_budget_fraction"], deadline,
                    self.cpu_ids[0 if side == "right" else 1], payload["application_mode"],
                    payload["terminal_proximal_weight"], payload["terminal_shortage_weight"])
            self.pending[side] = (next_sequence, task, now)
            self.active[side] = dict(sequence=next_sequence, source_cycle=int(cycle_index),
                                     incumbent=tuple(payload["incumbent_weights"][side]),
                                     submitted_at=now, deadline=deadline,
                                     application_mode=payload["application_mode"])
        self._flush_pending()
        return True

    def poll(self, *, cycle_index, incumbent_weights):
        if self.closed:
            return None, None
        self._flush_pending()
        proposals, arms, timings, source_cycles = {}, {}, {}, []
        while True:
            try:
                sequence, side, outcome, proposal, submitted_at, started_at, finished_at, error = self.results.get_nowait()
            except Empty:
                break
            current = self.active.get(side)
            if current is None or sequence != current["sequence"]:
                continue
            del self.active[side]
            source_cycles.append(current["source_cycle"])
            age = int(cycle_index) - current["source_cycle"]
            timing = dict(queue_wait_s=started_at - submitted_at,
                          compute_s=finished_at - started_at,
                          result_transport_s=max(0., self.clock() - finished_at),
                          total_s=finished_at - submitted_at)
            timings[side] = timing
            reason = None
            if error:
                reason = "projection_worker_failed"
            elif finished_at > current["deadline"]:
                reason = "projection_deadline_exceeded"
            elif age < 1 or age > self.max_age_cycles:
                reason = "stale_projection"
            elif (tuple(incumbent_weights[:len(current["incumbent"])]) if side == "right" else
                  tuple(incumbent_weights[-len(current["incumbent"]):])) != current["incumbent"]:
                reason = "incumbent_weights_changed"
            if reason:
                arms[side] = {"discard_reason": reason, "error": error}
                continue
            arms[side] = {key: value for key, value in outcome.items() if key not in {"state_history", "pulse_widths"}}
            if proposal is not None and outcome.get("deadline_met") is True:
                decision = ({"weights": proposal} if current["application_mode"] == APPLICATION_MODE
                            else {key: value for key, value in proposal.items() if key != "local_fit"})
                proposals[side] = dict(**decision, source_cycle=outcome["source_cycle"],
                                       request_id=outcome["request_id"], context_digest=outcome["context_digest"],
                                       local_fit=outcome["local_fit"], deadline_met=True,
                                       rollout_horizon_cycles=self.horizon_cycles,
                                       application_mode=current["application_mode"])
        for side, current in tuple(self.active.items()):
            if self.clock() > current["deadline"] or int(cycle_index) - current["source_cycle"] > self.max_age_cycles:
                _, _, sequence = self.slots[side]
                sequence.value += 1
                del self.active[side]
                self.pending.pop(side, None)
                source_cycles.append(current["source_cycle"])
                arms[side] = {"discard_reason": "projection_deadline_exceeded"}
        if not arms:
            return None, None
        return proposals or None, dict(arms=arms, timings=timings, asynchronous=True,
                                       persistent_supervisor=True,
                                       source_cycle_index=min(source_cycles))

    def close(self):
        if self.closed:
            return
        self.closed = True
        for process, requests, sequence in self.slots.values():
            sequence.value += 1
            if process.is_alive():
                process.terminate()
            process.join(timeout=0)
            if not process.is_alive():
                process.close()
            requests.cancel_join_thread()
            requests.close()
        self.results.cancel_join_thread()
        self.results.close()


class BilateralPaceVrAsync:
    """Bilateral async owner; boundary calls do not wait for completion."""

    def __init__(self, config, *, worker=None):
        self.config = config
        if worker is not None:
            self.worker = worker
        elif config.get("persistent_supervisor", False):
            self.worker = PersistentBilateralPaceWorker(
                horizon_cycles=config["horizon_cycles"], budget_seconds=config["deadline_seconds"],
                max_age_cycles=config["maximum_age_cycles"], cpu_ids=config["supervisor_cpu_ids"])
        else:
            self.worker = AsyncPaceWorker(
                evaluate_bilateral_pace_vr, horizon_cycles=config["horizon_cycles"],
                budget_seconds=config["deadline_seconds"], max_age_cycles=config["maximum_age_cycles"],
                cpu_ids=config["supervisor_cpu_ids"])
        self.contexts = {}
        self.last_weights = {"right": (), "left": ()}
        self.events = []

    def boundary(self, cycle, reports):
        metrics = {side: reports[side]["metrics"] for side in ("right", "left")}
        if not all(m.get("certified") is True for m in metrics.values()):
            return {}
        for side, values in metrics.items():
            if values.get("weights_used") is not None:
                self.last_weights[side] = tuple(values["weights_used"])
        token = self.last_weights["right"] + self.last_weights["left"]
        snapshot_docs = {side: values.get("pace_vr_snapshot") for side, values in metrics.items()}
        for side, snapshot in snapshot_docs.items():
            if snapshot is not None:
                self.contexts[side] = snapshot["context_digest"]
        proposal, audit = self.worker.poll(cycle_index=cycle, incumbent_weights=token)
        commands = {}
        if audit is not None:
            event = {**audit, "applied_cycle": cycle, "status": "held"}
            if proposal:
                for side, decision in proposal.items():
                    age = cycle - decision["source_cycle"]
                    reason = None
                    if decision.get("context_digest") != self.contexts.get(side):
                        reason = "context_mismatch"
                    elif age < 1 or age > self.config["maximum_age_cycles"]:
                        reason = "stale_source"
                    elif decision.get("deadline_met") is not True:
                        reason = "deadline_exceeded"
                    if reason:
                        event.setdefault("refused", {})[side] = reason
                    else:
                        commands[side] = {**decision, "applied_cycle": cycle, "application_lag_cycles": age,
                                          "application_mode": decision.get("application_mode"),
                                          "terminal_value_in_nlp": decision.get("application_mode") == PACE_RT_MODE}
                if commands:
                    event["status"] = "proposed"
            event["applied_arms"] = list(commands)
            self.events.append(event)
        due = cycle == 1 or cycle > 0 and cycle % self.config["update_every_cycles"] == 0
        if due and not commands and (not self.worker.busy or getattr(self.worker, "replaces_pending", False)) and all(snapshot_docs.values()):
            # AsyncPaceWorker's older policy adapts its bookkeeping horizon;
            # this request already contains a frozen fixed-horizon snapshot.
            # Keep the declared horizon truthful until snapshot adaptation is
            # implemented explicitly rather than silently changing the task.
            self.worker.horizon_cycles = self.config["horizon_cycles"]
            submitted = self.worker.submit(
                dict(snapshots=snapshot_docs, incumbent_weights={side: list(self.last_weights[side]) for side in self.last_weights},
                     budget_seconds=self.config["deadline_seconds"], maximum_log_step=self.config["maximum_log_weight_step"],
                     candidate_minimum_margin_improvement=self.config["candidate_minimum_margin_improvement"],
                     candidate_publication_reserve_seconds=self.config["candidate_publication_reserve_seconds"],
                     candidate_fit_budget_fraction=self.config["candidate_fit_budget_fraction"],
                     application_mode=self.config["application_mode"],
                     terminal_proximal_weight=self.config.get("terminal_proximal_weight", .03),
                     terminal_shortage_weight=self.config.get("terminal_shortage_weight", 1.),
                     horizon_cycles=self.config["horizon_cycles"],
                     horizon_ladder_cycles=list(self.config["horizon_ladder_cycles"]),
                     supervisor_cpu_ids=list(self.config["supervisor_cpu_ids"]),
                     evaluation_order=["right", "left"] if cycle // self.config["update_every_cycles"] % 2 == 0 else ["left", "right"]),
                cycle_index=cycle, incumbent_weights=token)
            self.events.append(dict(status="submitted" if submitted else "held", source_cycle=cycle,
                                    horizon_cycles=self.config["horizon_cycles"],
                                    horizon_ladder_cycles=list(self.config["horizon_ladder_cycles"]),
                                    horizon_adaptation="parallel_largest_valid",
                                    application_mode=self.config["application_mode"],
                                    terminal_value_in_nlp=self.config["application_mode"] == PACE_RT_MODE))
        return commands

    def close(self):
        self.worker.close()
