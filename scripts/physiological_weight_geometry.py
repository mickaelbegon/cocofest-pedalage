"""Geometry adapter for the published physiological-weight calculation.

Preserves the source's single-active-muscle total joint torque, including
other muscles' passive forces, finite-difference hand Jacobian, sorted modulo
angles and first-order endpoint velocity differences. This is not the RHO's
force model. Source: pyomeca/cocofest commit
31e064f4d80741c8b5444e6ce4d91bfe4eb78c3d, physiological_weight_calculation.py.
The implementation uses a CasADi graph to recompute torque for each case's
Fmax values without repeating inverse kinematics or assuming independent
column scaling in the presence of passive forces. No FHO data are used.
"""

from dataclasses import dataclass
import ast
from hashlib import sha256
from pathlib import Path
import re
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "examples/msk_models/Wu/Modified_Wu_Shoulder_Model_Cycling.bioMod"
IK_MODEL = ROOT / "examples/msk_models/Wu/Modified_Wu_Shoulder_Model_Cycling_for_IK.bioMod"


def _range_scalar(expression):
    """Evaluate only signed numeric constants and multiplication by pi."""
    def visit(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id == "pi":
            return np.pi
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return (-1 if isinstance(node.op, ast.USub) else 1) * visit(node.operand)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            return visit(node.left) * visit(node.right)
        raise ValueError("Unsupported Wu model range expression.")
    value = visit(ast.parse(expression, mode="eval").body)
    if not np.isfinite(value):
        raise ValueError("Nonfinite model range.")
    return value


def _wu_bounds(path):
    # Reading ranges from this simple bioMod avoids a native QRanges binding
    # crash in the available biorbd_casadi runtime. No dependency is patched.
    content = re.sub(r"//[^\n]*", "", Path(path).read_text())
    names, bounds = [], []
    for name, body in re.findall(r"\bsegment\s+(\S+)(.*?)\bendsegment\b", content, re.S):
        if re.search(r"(?m)^\s*translations\s+", body):
            raise ValueError("Only the three-rotation Wu IK model is supported.")
        rotation = re.search(r"(?m)^\s*rotations\s+(\S+)", body)
        if rotation is None:
            continue
        limits = re.search(r"(?m)^\s*(?:rangesQ|ranges)\s+(\S+)\s+(\S+)", body)
        if rotation[1] != "z" or limits is None:
            raise ValueError("Each active Wu segment must have one z rotation and explicit ranges.")
        names.append(name + "_RotZ")
        bounds.append([_range_scalar(limits[1]), _range_scalar(limits[2])])
    limits = np.asarray(bounds, float)
    if limits.shape != (3, 2) or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("Expected three nonempty Wu joint bounds.")
    return tuple(names), limits[:, 0], limits[:, 1]


def _source_wu_ik(path, n_shooting):
    import biorbd_casadi as biorbd
    import casadi as ca
    from scipy.optimize import least_squares

    model = biorbd.Model(str(path))
    dofs, lower, upper = _wu_bounds(path)
    if tuple(name.to_string() for name in model.nameDof()) != dofs:
        raise ValueError("Parsed ranges do not match model DoFs.")
    marker_names = [name.to_string() for name in model.markerNames()]
    if marker_names != ["hand", "wheel_center"]:
        raise ValueError("Expected the source's two ordered IK markers.")
    q = ca.MX.sym("ik_q", 3)
    markers = ca.vertcat(*[marker.to_mx() for marker in model.markers(q)])
    marker_fn = ca.Function("weight_ik_markers", [q], [markers])
    jac_fn = ca.Function("weight_ik_jac", [q], [ca.jacobian(markers, q)])
    z = float(np.asarray(marker_fn(np.zeros(3))).ravel()[2])
    angles = -2 * np.pi * np.linspace(0., 1., n_shooting + 1)
    result = np.empty((3, n_shooting + 1))
    residuals, xy_residuals = [], []
    for index, angle in enumerate(angles):
        target = np.asarray([.35 + .1 * np.cos(angle), .1 * np.sin(angle), 0., .35, 0., z])
        guess = (lower + upper) / 2 if index == 0 else result[:, index - 1]
        solution = least_squares(
            lambda value: np.asarray(marker_fn(value), float).ravel() - target,
            guess, jac=lambda value: np.asarray(jac_fn(value), float),
            bounds=(lower, upper), method="trf", xtol=1e-6,
        )
        if not solution.success or not np.all(np.isfinite(solution.x)):
            raise ValueError(f"IK failed at sample {index}: {solution.message}")
        result[:, index] = solution.x
        residuals.append(float(np.max(np.abs(solution.fun))))
        xy_residuals.append(float(np.max(np.abs(solution.fun[[0, 1, 3, 4]]))))
    return result, {"maximum_marker_error_m": max(residuals),
                    "maximum_xy_marker_error_m": max(xy_residuals),
                    "requested_marker_geometry_within_10um": max(residuals) <= 1e-5,
                    "source_least_squares_result_retained_without_exact_geometry_claim": True,
                    "bounds_source": "parsed_active_bioMod_ranges",
                    "solver": "scipy_trf_source_settings_with_casadi_marker_jacobian"}


def _positive_maxima(parameters, muscle_names):
    maxima = np.asarray([parameters[name]["Fmax"] for name in muscle_names], float)
    if not np.all(np.isfinite(maxima)) or np.any(maxima <= 0):
        raise ValueError("Each case must supply finite positive Fmax for every muscle.")
    return maxima


@dataclass
class CrankWeightGeometry:
    muscle_names: tuple
    theta: np.ndarray
    q: np.ndarray
    qdot: np.ndarray
    projectors: np.ndarray
    residual_projectors: np.ndarray
    torque_function: object
    mapped_torque_function: object
    metadata: dict

    def profiles(self, parameters):
        maxima = _positive_maxima(parameters, self.muscle_names)
        started = perf_counter()
        outputs = self.mapped_torque_function(
            self.q, self.qdot, np.repeat(maxima[:, None], len(self.theta), axis=1)
        )
        joint = [np.asarray(value, float) for value in outputs]
        profiles = np.asarray([np.sum(self.projectors * value, axis=0) for value in joint[:-1]])
        passive = np.sum(self.projectors * joint[-1], axis=0)
        joint_residuals = [np.linalg.norm(np.einsum("nij,jn->ni", self.residual_projectors, value), axis=1)
                           for value in joint[:-1]]
        if not np.all(np.isfinite(profiles)) or not np.all(np.isfinite(passive)):
            raise ValueError("Nonfinite biomechanical torque profile.")
        return profiles, {
            "profile_evaluation_s": perf_counter() - started,
            "Fmax": dict(zip(self.muscle_names, maxima.tolist())),
            "maximum_absolute_all_inactive_crank_torque_nm": float(np.max(np.abs(passive))),
            "maximum_actual_joint_torque_projection_residual_nm": float(np.max(joint_residuals)),
            "passive_torque_subtracted": False,
            "profiles_recomputed_for_case_Fmax": True,
        }


def build_geometry(muscle_names, *, n_shooting=120, model_path=MODEL, ik_model_path=IK_MODEL):
    """Build the published one-second kinematics and signed torque graph.

    IK uses the source's marker least-squares problem and joint bounds with
    a numerical CasADi-backend adapter. Endpoint velocities use the source's
    np.gradient(q, 1/n), edge_order=1.
    Model file hashes and backend are recorded; matching numerical output is
    checked separately, not assumed from using this adapter.
    """
    if isinstance(n_shooting, bool) or int(n_shooting) != n_shooting or n_shooting < 4:
        raise ValueError("n_shooting must be an integer >= 4.")
    import biorbd_casadi as biorbd
    import casadi as ca

    started = perf_counter()
    names = tuple(muscle_names)
    model_path, ik_model_path = Path(model_path), Path(ik_model_path)
    model = biorbd.Model(str(model_path))
    actual_names = tuple(name.to_string() for name in model.muscleNames())
    if actual_names != names:
        raise ValueError(f"Source-order muscle names do not match model: {actual_names} vs {names}.")
    q_ref, ik_audit = _source_wu_ik(ik_model_path, int(n_shooting))
    qdot_ref = np.gradient(q_ref, 1. / int(n_shooting), axis=1, edge_order=1)
    theta = np.mod(np.unwrap(q_ref[2]), 2 * np.pi)
    order = np.argsort(theta)
    theta, q_ref, qdot_ref = theta[order], q_ref[:, order], qdot_ref[:, order]

    q = ca.MX.sym("q", model.nbQ())
    velocity = ca.MX.sym("velocity", model.nbQdot())
    maxima = ca.MX.sym("Fmax", len(names))
    for index in range(len(names)):
        model.muscle(index).setForceIsoMax(maxima[index])
    marker_names = [name.to_string() for name in model.markerNames()]
    markers = model.markers(q)
    hand = markers[marker_names.index("hand")].to_mx()[:2]
    center = markers[marker_names.index("global_wheel_center")].to_mx()[:2]
    positions = ca.Function("weight_positions", [q], [hand, center])
    torques = []
    for active_index in (*range(len(names)), -1):
        states = model.stateSet()
        for index, state in enumerate(states):
            state.setExcitation(float(index == active_index))
            state.setActivation(float(index == active_index))
        torques.append(model.muscularJointTorque(states, q, velocity).to_mx())
    torque_function = ca.Function("weight_joint_torques", [q, velocity, maxima], torques)
    projectors = np.empty_like(q_ref)
    residual_projectors = np.empty((len(theta), model.nbQ(), model.nbQ()))
    residuals, condition_numbers, radii, hand_angles = [], [], [], []
    epsilon = 1e-7
    for sample, configuration in enumerate(q_ref.T):
        hand_xy, center_xy = [np.asarray(value, float).ravel() for value in positions(configuration)]
        radial = hand_xy - center_xy
        radius = np.linalg.norm(radial)
        if radius < 1e-10:
            raise ValueError("Hand too close to wheel center.")
        tangent = np.asarray([-radial[1], radial[0]]) / radius
        radii.append(float(radius))
        hand_angles.append(float(np.arctan2(radial[1], radial[0])))
        jacobian = np.empty((2, model.nbQ()))
        for index in range(model.nbQ()):
            change = np.zeros(model.nbQ())
            change[index] = epsilon
            plus = np.asarray(positions(configuration + change)[0], float).ravel()
            minus = np.asarray(positions(configuration - change)[0], float).ravel()
            jacobian[:, index] = (plus - minus) / (2 * epsilon)
        force_map, _, rank, _ = np.linalg.lstsq(jacobian.T, np.eye(model.nbQ()), rcond=None)
        if rank != 2:
            raise ValueError("Rank deficient hand Jacobian.")
        projectors[:, sample] = radius * tangent @ force_map
        condition_numbers.append(np.linalg.cond(jacobian.T))
        residual_projectors[sample] = jacobian.T @ force_map - np.eye(model.nbQ())
        residuals.append(np.linalg.norm(residual_projectors[sample]))
    metadata = {
        "geometry": "published_hand_force_projection_with_passive_contributions",
        "ik_audit": ik_audit,
        "backend": "biorbd_casadi", "casadi_version": ca.__version__,
        "n_shooting": int(n_shooting), "sample_count": len(theta),
        "cycle_period_s": 1., "requested_radius_m": .1, "requested_center_xy_m": [.35, 0.],
        "actual_hand_to_global_center_radius_range_m": [min(radii), max(radii)],
        "mean_hand_angular_velocity_rad_s": float(np.mean(np.diff(np.unwrap(np.asarray(hand_angles)[np.argsort(order)]))) * n_shooting),
        "mean_crank_coordinate_velocity_rad_s": float(np.mean(qdot_ref[2])),
        "pre_risk_angular_convention_not_temporally_validated": True,
        "velocity_endpoint_rule": "source_gradient_edge_order_1",
        "angle_rule": "source_sort_modulo_2pi_no_explicit_quadrature_closure",
        "minimum_angle_spacing_rad": float(np.min(np.diff(theta))),
        "angular_span_rad": float(theta[-1] - theta[0]),
        "max_hand_jacobian_condition": float(max(condition_numbers)),
        "max_full_joint_projection_residual_norm": float(max(residuals)),
        "joint_projection_residual_interpretation": "2D hand force cannot span every 3D generalized torque",
        "model": {"path": str(model_path.resolve()), "sha256": sha256(model_path.read_bytes()).hexdigest()},
        "ik_model": {"path": str(ik_model_path.resolve()), "sha256": sha256(ik_model_path.read_bytes()).hexdigest()},
        "setup_s": perf_counter() - started,
    }
    return CrankWeightGeometry(names, theta, q_ref, qdot_ref, projectors, residual_projectors,
                               torque_function, torque_function.map(len(theta)), metadata)
