from types import SimpleNamespace

import numpy as np
import pytest

from cocofest.optimization.pace_vr_async import (
    APPLICATION_MODE, BilateralPaceVrAsync, _choose_verified_weights, _evaluate_pace_vr_side,
    derived_fatigue_weights, validate_pace_vr_configuration,
)
from cocofest.simulation.independent_arms_process import _apply_pace_vr_weight_decision


def payload():
    return dict(parametric_fatigue_weights=True, muscle_pace={"adaptation_enabled": False},
                solver_cpu_affinity={"right": 2, "left": 3},
                experimental_pace_vr=dict(enabled=True, asynchronous=True, supervisor_cpu_id=4,
                                          supervisor_cpu_ids=[4, 5],
                                          application_mode=APPLICATION_MODE))


def decision():
    return dict(weights=[1.1, 1 / 1.1], source_cycle=1, applied_cycle=3,
                context_digest="same", deadline_met=True, application_mode=APPLICATION_MODE,
                local_fit=dict(reference_state=[[1000., .06, .14]] * 2,
                               state_scales=[[1000., .06, .14]] * 2,
                               trust_bounds=[-.01, .01]))


def test_enabled_contract_demands_separate_cpu_and_compiled_numerical_updates():
    declared = payload()
    config = validate_pace_vr_configuration(declared)
    assert config["deadline_seconds"] <= config["update_every_cycles"]
    for mutate, pattern in [
        (lambda p: p["experimental_pace_vr"].update(supervisor_cpu_ids=[2, 5]), "distinct"),
        (lambda p: p["experimental_pace_vr"].update(asynchronous=False), "asynchronous"),
        (lambda p: p.update(parametric_fatigue_weights=False), "compiled"),
        (lambda p: p["experimental_pace_vr"].update(application_mode="terminal_value"), "not wired"),
        (lambda p: p["experimental_pace_vr"].update(deadline_seconds=21), "call period"),
    ]:
        declared = payload()
        mutate(declared)
        with pytest.raises(ValueError, match=pattern):
            validate_pace_vr_configuration(declared)


class NoWaitWorker:
    def __init__(self):
        self.busy = False
        self.answer = (None, None)
        self.submissions = []

    def poll(self, **kwargs):
        answer, self.answer = self.answer, (None, None)
        return answer

    def submit(self, payload, **kwargs):
        self.submissions.append((payload, kwargs))
        self.busy = True
        return True

    def close(self):
        self.busy = False


def reports(*, snapshot=False, certified=True):
    values = dict(certified=certified, weights_used=[1., 1.])
    if snapshot:
        values["pace_vr_snapshot"] = dict(context_digest="same")
    return {side: {"metrics": dict(values)} for side in ("right", "left")}


def test_busy_supervisor_keeps_rho_nonblocking_and_no_backlog():
    worker = NoWaitWorker()
    supervisor = BilateralPaceVrAsync(validate_pace_vr_configuration(payload()), worker=worker)
    assert supervisor.boundary(1, reports(snapshot=True)) == {}
    assert supervisor.boundary(2, reports()) == {}
    assert supervisor.boundary(20, reports(snapshot=True)) == {}
    assert len(worker.submissions) == 1
    supervisor.close()


def test_parallel_horizon_ladder_requires_a_private_cpu_pair_per_horizon():
    declared = payload()
    declared["experimental_pace_vr"].update(
        horizon_cycles=100, horizon_ladder_cycles=[100, 50, 25, 20, 5],
        supervisor_cpu_ids=list(range(4, 14)),
    )
    config = validate_pace_vr_configuration(declared)
    assert config["horizon_ladder_cycles"] == (100, 50, 25, 20, 5)
    assert len(config["supervisor_cpu_ids"]) == 10

    declared["experimental_pace_vr"]["supervisor_cpu_ids"] = list(range(4, 12))
    with pytest.raises(ValueError, match="per rollout horizon"):
        validate_pace_vr_configuration(declared)


def test_ladder_configuration_is_sent_to_the_nonblocking_worker():
    declared = payload()
    declared["experimental_pace_vr"].update(
        horizon_cycles=100, horizon_ladder_cycles=[100, 50, 25, 20, 5],
        supervisor_cpu_ids=list(range(4, 14)),
    )
    worker = NoWaitWorker()
    supervisor = BilateralPaceVrAsync(validate_pace_vr_configuration(declared), worker=worker)
    supervisor.boundary(1, reports(snapshot=True))
    assert worker.submissions[0][0]["horizon_ladder_cycles"] == [100, 50, 25, 20, 5]
    assert worker.submissions[0][0]["supervisor_cpu_ids"] == list(range(4, 14))
    supervisor.close()


@pytest.mark.parametrize("patch,expected", [
    ({"context_digest": "changed"}, "context_mismatch"),
    ({"source_cycle": 0}, "stale_source"),
    ({"deadline_met": False}, "deadline_exceeded"),
])
def test_invalid_async_result_cannot_replace_last_weights(patch, expected):
    worker = NoWaitWorker()
    supervisor = BilateralPaceVrAsync(validate_pace_vr_configuration(payload()), worker=worker)
    supervisor.boundary(1, reports(snapshot=True))
    worker.answer = ({"right": {**decision(), **patch}}, {})
    cycle = 25 if expected == "stale_source" else 3
    assert supervisor.boundary(cycle, reports()) == {}
    assert supervisor.last_weights["right"] == (1., 1.)
    assert supervisor.events[-1]["refused"]["right"] == expected


def test_valid_async_proposal_only_on_later_certified_boundary():
    worker = NoWaitWorker()
    supervisor = BilateralPaceVrAsync(validate_pace_vr_configuration(payload()), worker=worker)
    supervisor.boundary(1, reports(snapshot=True))
    worker.answer = ({"right": decision()}, {})
    assert supervisor.boundary(2, reports(certified=False)) == {}
    commands = supervisor.boundary(3, reports())
    assert commands["right"]["applied_cycle"] == 3
    assert commands["right"]["application_lag_cycles"] == 2
    assert commands["right"]["terminal_value_in_nlp"] is False


def test_gradient_weight_adapter_is_bounded_and_explicitly_heuristic():
    result = {"accepted": True, "local_fit": {"accepted": True, "gradient": [[-2., 0., 0.], [-1., 0., 0.]]}}
    weights = derived_fatigue_weights(result, [1., 1.], np.log(1.25))
    np.testing.assert_allclose(weights, [1.25, .8])
    assert derived_fatigue_weights({**result, "accepted": False}, [1., 1.], .2) is None


def test_candidate_screen_keeps_incumbent_without_a_meaningful_feasible_gain():
    base = {"accepted": True, "terminal_value": -.003, "feasible_prefix_cycles": 50,
            "minimum_task_margin": .15}
    trials = {"proposal": {"accepted": False, "terminal_value": .1},
              "half_step": {"accepted": True, "terminal_value": -.003005}}
    chosen, audit = _choose_verified_weights(base, trials, np.ones(2),
                                             np.array([1.25, .8]), 5e-4)
    assert chosen is None
    assert audit["chosen"] == "incumbent"
    assert audit["predicted_improvement"] == 0.


def test_candidate_screen_can_select_half_step_over_full_proposal():
    base = {"accepted": True, "terminal_value": -.003, "feasible_prefix_cycles": 50,
            "minimum_task_margin": .15}
    trials = {"proposal": {"accepted": True, "terminal_value": -.00304},
              "half_step": {"accepted": True, "terminal_value": -.00308}}
    chosen, audit = _choose_verified_weights(base, trials, np.ones(2),
                                             np.array([1.25, .8]), 5e-4)
    np.testing.assert_allclose(chosen, [np.sqrt(1.25), np.sqrt(.8)])
    assert audit["chosen"] == "half_step"
    assert audit["predicted_improvement"] == pytest.approx(.00008)


def test_candidate_screen_can_reverse_a_harmful_gradient_direction():
    base = {"accepted": True, "terminal_value": -.003, "feasible_prefix_cycles": 50,
            "minimum_task_margin": .15}
    trials = {"proposal": {"accepted": True, "terminal_value": -.0028},
              "half_step": {"accepted": True, "terminal_value": -.0029},
              "opposite": {"accepted": True, "terminal_value": -.0032}}
    chosen, audit = _choose_verified_weights(base, trials, np.ones(2),
                                             np.array([1.25, .8]), 5e-4)
    np.testing.assert_allclose(chosen, [.8, 1.25])
    assert audit["chosen"] == "opposite"
    assert audit["predicted_improvement"] == pytest.approx(.0002)


@pytest.mark.parametrize("horizon", [5, 20, 100])
def test_normalized_margin_threshold_is_comparable_across_horizons(horizon):
    base = {"accepted": True, "terminal_value": 0., "feasible_prefix_cycles": horizon,
            "minimum_task_margin": 0.}
    trials = {"proposal": {"accepted": True, "deadline_met": True,
                           "terminal_value": -3e-4 / horizon}}
    chosen, audit = _choose_verified_weights(base, trials, np.ones(2),
                                             np.array([1.25, .8]), 2e-4)
    assert chosen is not None
    assert audit["equivalent_score_improvement"] == pytest.approx(2e-4 / horizon)
    trials["proposal"]["terminal_value"] = -1e-4 / horizon
    chosen, _ = _choose_verified_weights(base, trials, np.ones(2),
                                         np.array([1.25, .8]), 2e-4)
    assert chosen is None


def test_late_candidate_cannot_displace_or_cancel_prior_valid_candidate():
    base = {"accepted": True, "terminal_value": 0., "feasible_prefix_cycles": 20,
            "minimum_task_margin": 0.}
    trials = {"proposal": {"accepted": True, "deadline_met": True,
                           "completed_monotonic": 9., "terminal_value": -2e-5},
              "opposite": {"accepted": True, "deadline_met": False,
                           "completed_monotonic": 11., "terminal_value": -9e-5}}
    chosen, audit = _choose_verified_weights(base, trials, np.ones(2),
                                             np.array([1.25, .8]), 2e-4)
    np.testing.assert_allclose(chosen, [1.25, .8])
    assert audit["chosen"] == "proposal"
    assert audit["candidates"]["opposite"]["completed_monotonic"] == 11.


def test_candidate_improvement_threshold_is_validated_and_sent_to_worker():
    declared = payload()
    declared["experimental_pace_vr"].update(candidate_minimum_margin_improvement=2e-4,
                                             candidate_publication_reserve_seconds=1.5)
    config = validate_pace_vr_configuration(declared)
    worker = NoWaitWorker()
    supervisor = BilateralPaceVrAsync(config, worker=worker)
    supervisor.boundary(1, reports(snapshot=True))
    assert worker.submissions[0][0]["candidate_minimum_margin_improvement"] == 2e-4
    assert worker.submissions[0][0]["candidate_publication_reserve_seconds"] == 1.5
    declared["experimental_pace_vr"]["candidate_minimum_margin_improvement"] = -1.
    with pytest.raises(ValueError, match="candidate_minimum_margin_improvement"):
        validate_pace_vr_configuration(declared)


def test_original_snapshot_digest_is_checked_before_horizon_change(monkeypatch):
    import json
    import time

    monkeypatch.setattr("cocofest.optimization.pace_vr_async.os.sched_setaffinity", lambda *_: None)
    source = dict(request_id="right:1", source_cycle=1, context_digest="corrupt",
                  deadline_monotonic=time.monotonic() + 5,
                  payload_json=json.dumps({"context": {"config": {"horizon_cycles": 100}},
                                           "weights": [1., 1.]}))
    task = ("right", 20, source, [1., 1.], np.log(1.25), 2e-4,
            1., .65, time.monotonic() + 5, 0)
    with pytest.raises(ValueError, match="Original PACE-VR snapshot context digest mismatch"):
        _evaluate_pace_vr_side(task)


def test_apply_changes_only_fixed_parameter_binding_and_preserves_native_solver():
    class Binding:
        def update(self, ocp, weights):
            self.weights = weights
            return {"integration": "fixed_parameter_bounds", "ocp_cost_updated": True}

    graph, solver = object(), object()
    ocp = SimpleNamespace(nlp=[graph], ocp_solver=SimpleNamespace(shaked_ocp_solver=solver),
                          fatigue_weight_binding=Binding())
    pace = SimpleNamespace(weights=(1., 1.))
    outcome = _apply_pace_vr_weight_decision(ocp, pace, decision(), certified=True, current_cycle=3)
    assert outcome["compiled_nlp_reused"]
    assert ocp.nlp[0] is graph and ocp.ocp_solver.shaked_ocp_solver is solver
    assert pace.weights == tuple(decision()["weights"])
    assert outcome["terminal_value_in_nlp"] is False
    invalid = {**decision(), "weights": [1.1, 0.]}
    with pytest.raises(ValueError, match="strictly positive"):
        _apply_pace_vr_weight_decision(ocp, pace, invalid, certified=True, current_cycle=3)
