from types import SimpleNamespace

import casadi as ca
import numpy as np
import pytest

from cocofest.optimization.capacity_frontier_ocp import CapacityFrontierSupportProblem
from cocofest.optimization.ding_slow_manifold import DingSlowFatigueManifold


class Adapter:
    def __init__(self):
        self.muscle_order = ("Biceps",)
        self.manifolds = (DingSlowFatigueManifold(100., .05, .1, -.4, 2.1e-6, 1.9e-6),)
        self.indices = ((0, 1, 2),)
        self.coefficients = ((100., .05, .1),)
        self.center = (.8,)
        self.condition_sha256 = "c" * 64
        self.limits = {"lbx": np.array([.8, 1., 1., -1.]),
                       "ubx": np.array([.8, 1., 1., 1.]),
                       "x0": np.array([.8, 1., 1., 0.]),
                       "lbg": np.array([0.]), "ubg": np.array([0.])}

    def verify(self):
        return None


def problem():
    adapter = Adapter()
    x = ca.SX.sym("x", 4)
    return CapacityFrontierSupportProblem(graph={"x": x, "g": x[3], "f": x[3] ** 2},
                                         adapter=adapter, support=(1.,))


def test_support_problem_relaxes_only_slow_initial_bounds_and_adds_two_equalities():
    value = problem()
    limits = value.limits_for_domain(lower=(.4,), upper=(1.,))
    assert value.graph["g"].numel() == 3
    assert limits["lbg"].shape == (3,)
    assert limits["lbx"][0] == pytest.approx(.4)
    assert limits["ubx"][0] == pytest.approx(1.)
    # Tau1/Km retain sign-aware compatible ranges; their equality coupling is
    # in the auxiliary graph rather than inferred from independent bounds.
    assert limits["lbx"][1] < limits["ubx"][1]
    assert limits["lbx"][2] < limits["ubx"][2]
    assert limits["lbx"][3] == -1.
    assert limits["ubx"][3] == 1.


def test_support_problem_audit_reports_auxiliary_not_certified_result():
    value = problem()
    manifold = value.adapter.manifolds[0]
    states = manifold.states_from_capacity_ratio(.7)
    vector = np.array([states[0] / 100., states[1] / .05, states[2] / .1, 0.])
    audit = value.audit(vector)
    assert audit["initial_capacity_ratio"] == pytest.approx([.7])
    assert audit["slow_manifold_max_residual"] <= 1e-12
    assert audit["passed_discrete_auxiliary_audit"]
    assert not audit["certified_feasibility"]


def test_support_problem_rejects_invalid_domain_and_support():
    with pytest.raises(ValueError, match="support"):
        CapacityFrontierSupportProblem(graph={"x": ca.SX.sym("x", 4), "g": ca.SX(0), "f": ca.SX(0)},
                                       adapter=Adapter(), support=(0.,))
    with pytest.raises(ValueError, match="physical"):
        problem().limits_for_domain(lower=(-.1,), upper=(1.,))


def test_epsilon_constraint_is_added_and_audited():
    adapter = Adapter()
    x = ca.SX.sym("x", 4)
    value = CapacityFrontierSupportProblem(graph={"x": x, "g": x[3], "f": x[3] ** 2},
                                           adapter=adapter, support=(1.,), epsilon_upper=(.6,))
    assert value.graph["g"].numel() == 4
    limits = value.limits_for_domain(lower=(.2,), upper=(1.,))
    assert limits["ubg"][-1] == pytest.approx(.6)
    states = adapter.manifolds[0].states_from_capacity_ratio(.7)
    vector = np.array([states[0] / 100., states[1] / .05, states[2] / .1, 0.])
    assert value.audit(vector)["epsilon_capacity_max_violation"] == pytest.approx(.1)


def test_fixed_capacity_is_added_as_an_equality_and_audited():
    value = CapacityFrontierSupportProblem(graph={"x": ca.SX.sym("x", 4), "g": ca.SX(0), "f": ca.SX(0)},
                                           adapter=Adapter(), support=(1.,), fixed_capacity=(.6,))
    assert value.graph["g"].numel() == 4
    assert value.limits_for_domain(lower=(.2,), upper=(1.,))["lbg"][-1] == pytest.approx(.6)
    states = value.adapter.manifolds[0].states_from_capacity_ratio(.7)
    vector = np.array([states[0] / 100., states[1] / .05, states[2] / .1, 0.])
    assert value.audit(vector)["fixed_capacity_max_violation"] == pytest.approx(.1)
