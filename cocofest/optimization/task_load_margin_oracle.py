"""Full one-cycle load objective and independently evaluated solver evidence.

The load variable is eliminated exactly: lambda = E_prod(T)/nominal_work.
Only the terminal work equality is relaxed. All other constraints (including
the next-half-cycle condition) retain their original meaning and bounds.
No successful solve proves a global maximum or physiological exhaustion.
"""
from copy import copy, deepcopy
import math

import numpy as np

from .task_reserve_probe_adapter import IndependentAudit, _normalized_bound_violation
from .task_reserve import DEFAULT_CONSTRAINT_GROUPS


def configure_load_objective(program, *, nominal_work_j, upper_bound):
    """Mutate an exclusively owned, restored supervisor OCP before its solve."""
    from bioptim import Objective, ObjectiveFcn, Node

    if not math.isfinite(nominal_work_j) or nominal_work_j <= 0:
        raise ValueError("nominal work must be positive")
    if not math.isfinite(upper_bound) or upper_bound <= 1:
        raise ValueError("load cap must exceed nominal")
    for scope in [program, *program.nlp]:
        for collection in ("J", "J_internal"):
            for penalty in getattr(scope, collection, []):
                if not penalty:
                    continue
                if collection == "J_internal":
                    # Internal objective semantics cannot be silently retained.
                    raise ValueError("Internal objectives need an explicit supervisor adapter")
                replacement = copy(penalty)
                replacement.weight = deepcopy(penalty.weight)
                replacement.weight[...] = 0.
                if isinstance(replacement.node, tuple) and len(replacement.node) == 1:
                    replacement.node = replacement.node[0]
                program.update_objectives(replacement)
    bounds = program.nlp[0].x_bounds["E_prod"]
    if float(bounds.min[0, 2]) != nominal_work_j or float(bounds.max[0, 2]) != nominal_work_j:
        raise ValueError("Restored nominal terminal equality differs from source")
    bounds.min[0, 2] = 0.
    bounds.max[0, 2] = nominal_work_j * upper_bound
    slot = len(program.nlp[0].J)
    program.update_objectives(Objective(ObjectiveFcn.Mayer.MINIMIZE_STATE,
        key="E_prod", node=Node.END, phase=0, list_index=slot,
        quadratic=False, weight=-1. / nominal_work_j))


def _dual_residuals(values, lower, upper, dual):
    values, lower, upper, dual = [np.asarray(v, float).reshape(-1) for v in (values, lower, upper, dual)]
    if not (values.shape == lower.shape == upper.shape == dual.shape) or not np.all(np.isfinite(dual)):
        raise ValueError("Missing or invalid complete multipliers")
    comp = sign = 0.
    for v, lb, ub, lam in zip(values, lower, upper, dual):
        if lb == ub:
            continue
        bound = ub if lam >= 0 else lb
        if not np.isfinite(bound):
            sign = max(sign, abs(lam))
        else:
            comp = max(comp, abs(lam * (v - bound)))
    return comp, sign


def audit_load_solution(solution, program, *, nominal_work_j, upper_bound, tolerance):
    """Recompute every discrete constraint/bound and actual KKT residuals.

    KKT residuals concern the full *scaled transcription*, including variable
    bounds. They are optional diagnostics, never fabricated from IPOPT status.
    """
    import casadi as ca
    from bioptim import SolutionMerge

    graph, limits = program.ocp_solver.nlp, program.ocp_solver.limits
    vector = np.asarray(solution.vector, float).reshape(-1)
    graph_objects = (graph["x"], graph["g"], graph["f"])
    cache = getattr(program, "_task_load_margin_oracle_cache", None)
    if (cache is None or len(cache.get("graph_objects", ())) != len(graph_objects)
            or any(left is not right for left, right in
                   zip(cache["graph_objects"], graph_objects))):
        cache = {"graph_objects": graph_objects,
                 "constraints": ca.Function("load_margin_audit", [graph["x"]], [graph["g"]])}
        setattr(program, "_task_load_margin_oracle_cache", cache)
    evaluator = cache["constraints"]
    constraint = np.asarray(evaluator(vector), float).reshape(-1)
    violation = max(_normalized_bound_violation(constraint, limits["lbg"], limits["ubg"]),
                    _normalized_bound_violation(vector, limits["lbx"], limits["ubx"]))
    states = solution.decision_states(to_merge=SolutionMerge.NODES)
    controls = solution.decision_controls(to_merge=SolutionMerge.NODES)
    if not controls or not any(k.startswith("last_pulse_width_") for k in controls):
        raise ValueError("Missing direct PW controls/history")
    if not all(np.all(np.isfinite(v)) for data in (states, controls) for v in data.values()):
        raise ValueError("Nonfinite trajectories")
    load = float(np.asarray(states["E_prod"]).reshape(-1)[-1]) / nominal_work_j
    if not math.isfinite(load):
        raise ValueError("Nonfinite terminal load")
    violation = max(violation, max(0., -load, load - upper_bound))
    detail = {"load_factor": load, "constraint_rows": constraint.size,
              "decision_variables": vector.size, "scope": "complete_discrete_NLP_no_continuous_replay"}
    kkt = {name: None for name in ("stationarity_residual", "complementarity_residual", "dual_feasibility_residual")}
    try:
        lam_g, lam_x = np.asarray(solution.lam_g, float).reshape(-1), np.asarray(solution.lam_x, float).reshape(-1)
        if "stationarity" not in cache:
            symbol = ca.MX if isinstance(graph["x"], ca.MX) else ca.SX
            symbolic_lam_g = symbol.sym("load_lam_g", lam_g.size)
            symbolic_lam_x = symbol.sym("load_lam_x", lam_x.size)
            cache["stationarity"] = ca.Function("load_margin_stationarity",
                [graph["x"], symbolic_lam_g, symbolic_lam_x],
                [ca.gradient(graph["f"], graph["x"])
                 + ca.jacobian(graph["g"], graph["x"]).T @ symbolic_lam_g + symbolic_lam_x])
        residual = float(np.max(np.abs(np.asarray(cache["stationarity"](vector, lam_g, lam_x)))))
        comp_g, sign_g = _dual_residuals(constraint, limits["lbg"], limits["ubg"], lam_g)
        comp_x, sign_x = _dual_residuals(vector, limits["lbx"], limits["ubx"], lam_x)
        if not math.isfinite(residual):
            raise ValueError("Nonfinite KKT residual")
        kkt.update(stationarity_residual=residual, complementarity_residual=max(comp_g, comp_x),
                   dual_feasibility_residual=max(sign_g, sign_x))
    except (ValueError, TypeError, RuntimeError, AttributeError, KeyError) as error:
        detail["kkt_unavailable"] = str(error)
    return load, IndependentAudit(violation <= tolerance, violation, DEFAULT_CONSTRAINT_GROUPS, detail), kkt


def reoptimized_central_gradient(center, steps, solve_at):
    """Finite differences of independently reoptimized feasible load witnesses.

    ``solve_at`` must restore the complete source checkpoint, modify only the
    declared coordinates and maximize load anew. Missing witnesses invalidate
    the derivative. This derivative alone never authorizes RHO activation;
    independent smaller-step and held-out validation remain mandatory.
    """
    center, steps = np.asarray(center, float), np.asarray(steps, float)
    if center.ndim != 1 or steps.shape != center.shape or not np.all(np.isfinite(center)) or not np.all(np.isfinite(steps)) or np.any(steps <= 0):
        raise ValueError("Finite center and positive coordinate steps required")
    derivative = []
    for index, step in enumerate(steps):
        offset = np.zeros_like(center)
        offset[index] = step
        pair = [solve_at(center + sign * offset) for sign in (-1, 1)]
        if any(value is None or not math.isfinite(value) for value in pair):
            raise ValueError("Reoptimized feasible load witness missing")
        derivative.append((pair[1] - pair[0]) / (2 * step))
    return np.asarray(derivative)


def initial_bound_envelope_gradient(solution, program, coordinates):
    """Return d(lambda)/d(normalized initial state) from complete bound duals.

    For min -lambda with L=f+lam_x*(z-b), df*/db=-lam_x. Thus
    d(lambda)/d((physical-offset)/scale)=lam_x*scale/scaling_of_z.
    The symbolic mapping is checked explicitly; fixed-node vector offsets are
    never assumed. This derivative is conditional on all other data and still
    requires primal/KKT and independent local validation before RHO use.
    """
    import casadi as ca
    nlp = program.nlp[0]
    graph, limits = program.ocp_solver.nlp, program.ocp_solver.limits
    signature = tuple((item.state_key, item.index, item.scale, item.offset)
                      for item in coordinates)
    cache = getattr(program, "_task_load_margin_mapping_cache", None)
    if cache is None or cache.get("signature") != signature or cache.get("graph_x") is not graph["x"]:
        normalized = ca.vertcat(*[
            (nlp.X_scaled[0][nlp.states[item.state_key].index[item.index], 0]
             * float(np.asarray(nlp.x_scaling[item.state_key].scaling).reshape(-1)[item.index])
             - item.offset)/item.scale for item in coordinates])
        mapping = ca.jacobian(normalized, graph["x"])
        mapper = ca.Function("initial_load_coordinate_mapping", [graph["x"]], [mapping, normalized])
        cache = {"signature": signature, "graph_x": graph["x"], "mapper": mapper}
        setattr(program, "_task_load_margin_mapping_cache", cache)
    mapper = cache["mapper"]
    jacobian, actual = mapper(solution.vector)
    jacobian, actual = np.asarray(jacobian, float), np.asarray(actual, float).reshape(-1)
    lower, upper = (np.asarray(limits[key], float).reshape(-1) for key in ("lbx", "ubx"))
    multipliers = np.asarray(solution.lam_x, float).reshape(-1)
    vector = np.asarray(solution.vector, float).reshape(-1)
    if not np.all(np.isfinite(multipliers)) or multipliers.shape != lower.shape:
        raise ValueError("Complete finite bound multipliers required")
    derivative, indices = [], []
    for row, item in zip(jacobian, coordinates):
        selected = np.flatnonzero(row)
        if selected.size != 1 or row[selected[0]] <= 0:
            raise ValueError("Each normalized initial state must map to one scaled decision variable")
        index = int(selected[0])
        if (lower[index] != upper[index] or not np.isfinite(lower[index])
                or not np.isclose(vector[index], lower[index], atol=1e-8, rtol=0)):
            raise ValueError("Initial-state envelope requires a verified fixed bound")
        bound = nlp.x_bounds[item.state_key]
        if bound.min[item.index, 0] != bound.max[item.index, 0]:
            raise ValueError("Initial physical coordinate is not fixed")
        expected = (float(bound.min[item.index, 0])-item.offset)/item.scale
        if not np.isclose(actual[len(indices)], expected, atol=1e-8, rtol=0):
            raise ValueError("Initial coordinate mapping disagrees with physical fixed bound")
        derivative.append(multipliers[index]/row[index])
        indices.append(index)
    return {"gradient": derivative, "decision_variable_indices": indices,
            "coordinate_names": [item.state_key for item in coordinates],
            "normalized_coordinates": actual.tolist(),
            "sensitivity_method": "kkt_envelope",
            "scope": "conditional_initial_state_bound_envelope"}


def validate_reoptimized_direction(center, gradient, center_load, steps, offsets, solve_at):
    """Independent smaller-step derivatives and mixed-coordinate holdouts.

    The caller must enforce full primal/KKT checks in ``solve_at``. No point
    here may have been used to fit ``gradient``. This is local empirical
    validation conditional on every unselected source state and history.
    """
    center, gradient = np.asarray(center, float), np.asarray(gradient, float)
    offsets = np.asarray(offsets, float)
    if (center.ndim != 1 or gradient.shape != center.shape or offsets.ndim != 2
            or offsets.shape[1] != center.size or offsets.shape[0] < 1
            or not np.all(np.isfinite(gradient)) or not np.all(np.isfinite(offsets))
            or not math.isfinite(center_load) or np.any(np.max(np.abs(offsets), axis=1) == 0)):
        raise ValueError("Finite gradient and noncentral holdouts of matching dimension required")
    finer = reoptimized_central_gradient(center, steps, solve_at)
    heldout = []
    for delta in offsets:
        actual = solve_at(center + delta)
        if actual is None or not math.isfinite(actual):
            raise ValueError("Independently reoptimized holdout witness missing")
        prediction = float(center_load + gradient @ delta)
        heldout.append({"coordinates": (center + delta).tolist(),
                        "load_factor": float(actual), "prediction": prediction,
                        "absolute_error": abs(float(actual) - prediction)})
    return {"smaller_step_gradient": finer.tolist(),
            "gradient_validation_max_abs_error": float(np.max(np.abs(finer - gradient))),
            "local_validation_max_abs_error": max(item["absolute_error"] for item in heldout),
            "independent_validation_points": len(heldout), "holdouts": heldout}
