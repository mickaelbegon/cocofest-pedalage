"""Local-bound equivalence and impossible-manifold guards."""
import numpy as np
import pytest
import casadi as ca
from types import SimpleNamespace as NS

from scripts.benchmark_ding_radau5_local_ab import (intersect_affine_bounds, build_local_map, frozen_limits,
                                                     recorded_outer_solution)
from scripts.validate_ding_radau5_condensation import radau5_tableau


def test_affine_bound_intersection_handles_both_signs_and_constants():
    lower, upper = intersect_affine_bounds(
        [-10.], [10.], [0, 0, 0], [-2., 3., 0.], [1., -1., 2.],
        [-3., -4., 1.], [5., 8., 3.])
    np.testing.assert_allclose(lower, [-1.])
    np.testing.assert_allclose(upper, [2.])
    for a in np.linspace(-4, 4, 161):
        original_feasible = (-10 <= a <= 10 and -3 <= -2*a+1 <= 5
                             and -4 <= 3*a-1 <= 8)
        assert original_feasible == bool(lower[0] <= a <= upper[0])


def test_impossible_intersection_and_calcium_are_rejected():
    with pytest.raises(ValueError, match="Infeasible"):
        intersect_affine_bounds([0.], [1.], [0], [1.], [0.], [2.], [3.])
    with pytest.raises(ValueError, match="Fixed reconstructed"):
        intersect_affine_bounds([0.], [1.], [0], [0.], [2.], [0.], [1.])


def test_frozen_limits_replace_only_recorded_solver_inputs(tmp_path):
    np.savez(tmp_path / "nlp2_call4.npz", x0=[1., 2.], lbx=[0., 0.], ubx=[3., 3.],
             lbg=[0.], ubg=[0.], lam_x0=[.1, .2], lam_g0=[.3])
    original = {"x0": np.zeros(2), "lbx": -np.ones(2), "ubx": np.ones(2),
                "lbg": -np.ones(1), "ubg": np.ones(1), "unrelated": "kept"}
    replaced, path = frozen_limits(original, tmp_path, nlp_index=2, call=3, offset=1)
    np.testing.assert_allclose(replaced["x0"], [1., 2.])
    np.testing.assert_allclose(replaced["lam_g0"], [.3])
    assert replaced["unrelated"] == "kept"
    assert path.name == "nlp2_call4.npz"


def test_recorded_solution_is_only_an_outer_loop_transport(tmp_path):
    path = tmp_path / "recorded.npz"
    np.savez(path, x=[2.], g=[3.], lam_x=[4.], lam_g=[5.])
    x = ca.SX.sym("x")
    fg = ca.Function("fg", [x], [x**2, x + 1])
    returned = recorded_outer_solution({"x": ca.DM([9.]), "g": ca.DM([10.]),
                                        "lam_x": ca.DM([11.]), "lam_g": ca.DM([12.])}, path, fg)
    np.testing.assert_allclose(np.asarray(returned["x"]), [[2.]])
    np.testing.assert_allclose(np.asarray(returned["g"]), [[3.]])
    np.testing.assert_allclose(np.asarray(returned["f"]), [[4.]])


def test_reconstruction_is_discrete_radau_with_arbitrary_incoming_offsets():
    _, tableau = radau5_tableau()
    h, rest, alpha = 1/30, np.array([4920., .060601, .137]), np.array([-.4, 2.1e-5, 1.9e-5])
    stages, end = ca.SX.sym("stages", 5, 6), ca.SX.sym("end", 5, 1)
    x = ca.vertcat(ca.vec(stages), end)
    rhs = []
    for j in range(1, 6):
        cn, f, a, tau, km = (stages[i, j] for i in range(5))
        rhs.append(ca.vertcat((.4-cn)/.011, a*cn/(km+cn)-f/tau,
                   -(a-rest[0])/127+alpha[0]*f,
                   -(tau-rest[1])/127+alpha[1]*f,
                   -(km-rest[2])/127+alpha[2]*f))
    defects = ca.vertcat(*(stages[:,j+1]-stages[:,0]-h*sum(
        (float(tableau[j,k])*rhs[k] for k in range(5)), ca.SX.zeros(5)) for j in range(5)))
    g = ca.vertcat(defects, end-stages[:,-1])
    keys = {k+"_test": NS(index=[i]) for i,k in enumerate(("Cn","F","A","Tau1","Km"))}
    muscle = NS(muscle_name="test", alpha_a=alpha[0], alpha_tau1=alpha[1], alpha_km=alpha[2])
    phase = NS(dynamics_type=NS(ode_solver=NS(method="radau", polynomial_degree=5)),
               model=NS(muscles_dynamics_model=[muscle]), states=keys,
               x_scaling={k:NS(scaling=np.ones((1,1))) for k in keys}, X_scaled=[stages,end])
    limits = dict(lbg=np.zeros(30), ubg=np.zeros(30))
    nlp = dict(x=x, f=ca.sumsqr(ca.sin(x)), g=g)
    m = build_local_map(nlp, NS(ocp=NS(nlp=[phase]),limits=limits))
    incoming = np.array([.23,4920*.77,.072,.18])
    linear = m["factor"].solve(np.r_[-m["constant"], incoming])
    offset = linear[m["zpos"]]-m["coefficient"]*linear[m["apos"]]
    y = np.zeros(len(m["r"]))
    # A stages from the same Radau linear operator and arbitrary force stages.
    force = np.array([20.,40.,45.,42.,37.])
    capacity = np.linalg.solve(np.eye(5)+h*tableau/127,
                              np.full(5,incoming[1])+h*tableau@(np.full(5,rest[0]/127)+alpha[0]*force))
    full_seed = np.zeros(35)
    full_seed[1],full_seed[2] = 12.,incoming[1]
    for j in range(5):
        full_seed[5*(j+1)+1],full_seed[5*(j+1)+2] = force[j],capacity[j]
    full_seed[31],full_seed[32] = force[-1],capacity[-1]
    y[:] = full_seed[m["r"]]
    full = m["reconstruct"](y,offset)
    original = np.array(ca.Function("g_test",[x],[g])(full)).ravel()
    np.testing.assert_allclose(original[m["removed"]],0,atol=2e-13)
    assert m["nlp"]["x"].numel() == 14
    assert m["nlp"]["g"].numel() == 12
