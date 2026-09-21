from pathlib import Path

import numpy as np
import pytest

from cocofest.dynamics.reduced_cycling import (
    build_reduced_cycling_dynamics,
    solve_bilateral_cycling_kinematics,
)


MODEL = (
    Path(__file__).resolve().parents[2]
    / "examples/msk_models/Wu/Modified_Wu_Shoulder_Model_Cycling_Bilateral.bioMod"
)


def test_bilateral_biomod_reduces_two_hands_on_one_crank():
    pytest.importorskip("biorbd_casadi")
    theta, q, audit = solve_bilateral_cycling_kinematics(MODEL, sample_count=17)

    assert theta.shape == (17,)
    assert q.shape == (5, 17)
    assert audit["kinematic_model"] == "bilateral_hand_crank"
    assert audit["maximum_hand_handle_residual_m"] < 1e-9
    assert audit["maximum_cycle_closure_error"] < 1e-9
    # The unique crank coordinate winds once; arms close at their own initial
    # configuration, proving this is a true two-chain kinematic construction.
    np.testing.assert_allclose(q[:4, 0], q[:4, -1], atol=1e-9)
    np.testing.assert_allclose(q[4, -1] - q[4, 0], -2.0 * np.pi, atol=1e-12)


def test_bilateral_biomod_profile_contains_eight_physical_muscles():
    pytest.importorskip("biorbd_casadi")
    reduced, audit = build_reduced_cycling_dynamics(
        MODEL, sample_count=41, kinematic_order=8, dynamics_order=8
    )

    assert audit["kinematic_model"] == "bilateral_hand_crank"
    assert len(reduced.muscle_names) == 8
    assert {name.split("_", 1)[0] for name in reduced.muscle_names} == {
        "right",
        "left",
    }
    assert reduced.crank_torque_dof_index == 4
    assert reduced.coefficient_values(0.0)["effective_inertia"] > 0.0
