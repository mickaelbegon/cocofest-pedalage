"""Observation geometry from the cycling OCP's saved reduced bioMod profile.

Loading/evaluating a profile needs only NumPy. Its generation and mechanical
validation remain the responsibility of ``dynamics.reduced_cycling``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import numpy as np

from cocofest.dynamics.reduced_cycling import ReducedCyclingDynamics
from cocofest.identification.synthetic_cycling import BILATERAL_MUSCLE_NAMES, CyclingTrialSpec


@dataclass(frozen=True)
class ReducedGeometryAdapter:
    """Map a sensor trial to a reduced profile without assuming side symmetry.

    Names match the bilateral bioMod exactly by default. A unilateral profile
    requires an explicit mapping, e.g. ``{"right_Biceps": "Biceps"}``. A
    separate adapter/profile can represent the other arm. ``angle_offset_rad``
    converts measured physical crank angle to the profile's convention; no
    half-cycle shift or sign reversal is silently applied.

    Signed moment arms (including zeros) are valid. The Ding ODE gain is
    ``f_length * f_velocity + f_passive``, exactly as in the cycling model.
    This is distinct from the moment arm used to project force to torque.
    """

    profile: ReducedCyclingDynamics
    muscle_map: Mapping[str, str] | None = None
    angle_offset_rad: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.angle_offset_rad):
            raise ValueError("angle_offset_rad must be finite.")
        names = tuple(self.profile.muscle_names)
        if len(names) != len(set(names)) or not names:
            raise ValueError("Profile muscle names must be unique and nonempty.")
        if self.profile.muscle_geometry is None:
            raise ValueError("Reduced profile needs muscle geometry; rebuild the profile.")
        mapping = ({name: name for name in BILATERAL_MUSCLE_NAMES if name in names}
                   if self.muscle_map is None else dict(self.muscle_map))
        if not mapping or set(mapping) - set(BILATERAL_MUSCLE_NAMES) or set(mapping.values()) - set(names):
            raise ValueError("Provide an explicit valid bilateral-to-profile muscle_map.")
        object.__setattr__(self, "muscle_map", mapping)

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "ReducedGeometryAdapter":
        """Load a saved profile; no biorbd, CasADi or Bioptim import is needed."""
        return cls(ReducedCyclingDynamics.load(path), **kwargs)

    @property
    def provenance(self) -> dict:
        return {
            "kind": "reduced_biomod_profile",
            "source_model_sha256": self.profile.source_model_sha256,
            "muscle_map": dict(self.muscle_map),
            "angle_offset_rad": float(self.angle_offset_rad),
            "mechanical_gain": "active_force_length * force_velocity + passive_force",
            "moment_arm": "muscle_effectiveness = -d_length/d_theta",
            "radial_gain": "requires independent calibration or reaction-force model",
        }

    def apply(
        self,
        spec: CyclingTrialSpec,
        *,
        omega_rad_s: float | np.ndarray,
        radial_gain: np.ndarray | None = None,
    ) -> CyclingTrialSpec:
        """Return a trial using angle/speed-dependent OCP coefficients.

        Supply measured angular velocity (or a scalar for an isokinetic trial).
        The profile yields net muscle generalized torque, not sensor reaction
        forces. Baseline subtraction and mechanical calibration must therefore
        be validated experimentally. Radial gain is deliberately never inferred
        from the tangential moment arm.
        """
        if spec.muscle_name not in self.muscle_map:
            raise ValueError(f"No explicit profile mapping for {spec.muscle_name}.")
        if spec.radial_gain is not None and radial_gain is None:
            raise ValueError("Supply calibrated radial_gain explicitly for a two-axis trial.")
        angle = np.asarray(spec.angle_rad, dtype=float) + self.angle_offset_rad
        velocity = np.asarray(omega_rad_s, dtype=float)
        if velocity.ndim == 0:
            velocity = np.full(angle.shape, velocity)
        if velocity.shape != angle.shape or not np.all(np.isfinite(velocity)):
            raise ValueError("omega_rad_s must be finite and scalar or aligned with time_s.")
        index = self.profile.muscle_names.index(self.muscle_map[spec.muscle_name])
        count = len(self.profile.muscle_names)
        moment_arm, gain = np.empty(angle.size), np.empty(angle.size)
        for sample, (theta, omega) in enumerate(zip(angle, velocity)):
            coefficients = self.profile.coefficient_values(float(theta))
            inertia = float(coefficients["effective_inertia"])
            effectiveness = np.asarray(coefficients["muscle_effectiveness"], dtype=float)
            relationships = [np.asarray(value, dtype=float) for value in
                             self.profile.muscle_relationships(float(theta), float(omega))]
            if (not np.isfinite(inertia) or inertia <= 0
                    or effectiveness.shape != (count,)
                    or not np.all(np.isfinite(effectiveness))
                    or any(value.shape != (count,) or not np.all(np.isfinite(value)) for value in relationships)):
                raise ValueError(f"Nonphysical or malformed reduced geometry at sample {sample}.")
            force_length, force_velocity, passive = (value[index] for value in relationships)
            # The repository's passive law can be slightly negative below l/lopt=1.
            # Retain it exactly, while rejecting a nonpositive total ODE multiplier.
            gain[sample] = force_length * force_velocity + passive
            if force_length < 0 or force_velocity < 0 or not np.isfinite(gain[sample]) or gain[sample] <= 0:
                raise ValueError(f"Nonphysical Ding mechanical gain at sample {sample}.")
            moment_arm[sample] = effectiveness[index]
        return replace(spec, moment_arm_m=moment_arm, mechanical_gain=gain, radial_gain=radial_gain)


def planar_endpoint_force(
    endpoint_jacobian_xy: np.ndarray,
    generalized_torque: np.ndarray,
    *,
    condition_limit: float = 1e8,
) -> np.ndarray:
    """Recover a planar endpoint force from the corresponding joint torque.

    ``endpoint_jacobian_xy`` maps the selected joint velocity to the endpoint
    x/y velocity.  Consequently ``J.T @ force == generalized_torque``.  The
    helper deliberately rejects an under-determined endpoint instead of
    silently applying a pseudo-inverse: the radial component is meaningful
    only when the two arm DoFs span the two measured Cartesian components.
    """
    jacobian = np.asarray(endpoint_jacobian_xy, dtype=float)
    torque = np.asarray(generalized_torque, dtype=float).reshape(-1)
    if jacobian.ndim != 2 or jacobian.shape[0] != 2:
        raise ValueError("endpoint_jacobian_xy must have shape (2, selected_dofs).")
    if jacobian.shape[1] != torque.size or not np.all(np.isfinite(jacobian)) or not np.all(np.isfinite(torque)):
        raise ValueError("Endpoint Jacobian and generalized torque must be finite and aligned.")
    if not np.isfinite(condition_limit) or condition_limit <= 1:
        raise ValueError("condition_limit must be finite and greater than one.")
    map_to_torque = jacobian.T
    if map_to_torque.shape != (2, 2):
        raise ValueError(
            "A two-axis endpoint force requires exactly two selected arm DoFs; "
            "use a constrained inverse-dynamics model for redundant arms."
        )
    condition = np.linalg.cond(map_to_torque)
    if not np.isfinite(condition) or condition > condition_limit:
        raise ValueError(f"Endpoint Jacobian is singular or ill-conditioned (condition={condition:.3g}).")
    force = np.linalg.solve(map_to_torque, torque)
    if not np.all(np.isfinite(force)):
        raise ValueError("Endpoint-force solve returned non-finite values.")
    return force


@dataclass(frozen=True)
class PlanarEndpointForceAdapter:
    """Derive tangential and radial pedal-force observations from a bioMod.

    The full bioMod supplies the hand-marker Jacobian and the muscle-length
    Jacobian.  For each point on the reduced contact manifold, a unit muscle
    force generates a joint torque ``-dL/dq``.  Solving ``J_hand.T F = tau``
    over the two arm DoFs yields the force *applied by the arm to the pedal*.
    This is a physical two-axis observation map, unlike a free radial gain.

    It is intentionally limited to an independently identified arm (two arm
    DoFs and one planar endpoint).  For the bilateral coupled model, create a
    dedicated adapter per arm with the matching two DoFs and hand marker.  A
    dynamic total contact force additionally requires all muscle forces and
    residual torques; ``force_from_full_dynamics`` exposes that calculation.
    """

    geometry: ReducedGeometryAdapter
    model_path: Path
    endpoint_marker: str = "hand"
    center_marker: str = "global_wheel_center"
    arm_dof_indices: tuple[int, int] | None = None
    condition_limit: float = 1e8
    radius_relative_tolerance: float = 1e-4

    def __post_init__(self) -> None:
        path = Path(self.model_path)
        if not path.is_file():
            raise ValueError(f"bioMod model does not exist: {path}")
        if not np.isfinite(self.condition_limit) or self.condition_limit <= 1:
            raise ValueError("condition_limit must be finite and greater than one.")
        if not np.isfinite(self.radius_relative_tolerance) or not 0 < self.radius_relative_tolerance < 1:
            raise ValueError("radius_relative_tolerance must lie strictly between zero and one.")
        # Fail early if a profile from another bioMod is accidentally paired
        # with this endpoint geometry.
        self.geometry.profile.validate_source_model(path)
        object.__setattr__(self, "model_path", path)

    @classmethod
    def from_paths(
        cls,
        profile_path: str | Path,
        model_path: str | Path,
        **kwargs,
    ) -> "PlanarEndpointForceAdapter":
        geometry_kwargs = {
            name: kwargs.pop(name)
            for name in ("muscle_map", "angle_offset_rad")
            if name in kwargs
        }
        return cls(
            ReducedGeometryAdapter.load(profile_path, **geometry_kwargs),
            Path(model_path),
            **kwargs,
        )

    @classmethod
    def bilateral_arm_from_paths(
        cls,
        profile_path: str | Path,
        model_path: str | Path,
        *,
        side: str,
        **kwargs,
    ) -> "PlanarEndpointForceAdapter":
        """Build the safe endpoint mapping for one arm of the bilateral Wu model.

        The mapping is intentionally explicit: it prevents a right-hand
        Jacobian being combined with a left-arm muscle or the shared crank DoF.
        The caller must still use a profile created from the same bioMod.
        """
        if side not in ("right", "left"):
            raise ValueError("side must be 'right' or 'left'.")
        defaults = {
            "endpoint_marker": f"{side}_hand",
            "center_marker": "wheel_center",
            "arm_dof_indices": (0, 1) if side == "right" else (2, 3),
        }
        overlap = set(defaults) & set(kwargs)
        if overlap:
            raise ValueError(
                "bilateral_arm_from_paths fixes " + ", ".join(sorted(overlap))
                + "; instantiate PlanarEndpointForceAdapter directly for a custom bioMod."
            )
        return cls.from_paths(profile_path, model_path, **defaults, **kwargs)

    @property
    def provenance(self) -> dict:
        return {
            **self.geometry.provenance,
            "kind": "biomod_planar_endpoint_inverse_dynamics",
            "model_path": str(self.model_path),
            "endpoint_marker": self.endpoint_marker,
            "center_marker": self.center_marker,
            "arm_dof_indices": None if self.arm_dof_indices is None else list(self.arm_dof_indices),
            "radius_relative_tolerance": float(self.radius_relative_tolerance),
            "force_convention": "force applied by arm to pedal",
            "radial_gain": "derived from endpoint Jacobian and muscle torque",
        }

    def _model_and_indices(self):
        try:
            import biorbd
        except ImportError:  # source-built Linux uses the CasADi backend
            import biorbd_casadi as biorbd
        model = biorbd.Model(str(self.model_path))
        marker_names = [model.markerNames()[index].to_string() for index in range(model.nbMarkers())]
        dof_names = [model.nameDof()[index].to_string() for index in range(model.nbQ())]
        if self.endpoint_marker not in marker_names or self.center_marker not in marker_names:
            raise ValueError(
                f"bioMod must contain endpoint '{self.endpoint_marker}' and centre '{self.center_marker}'."
            )
        arm_indices = (
            tuple(index for index in range(model.nbQ()) if index != self.geometry.profile.crank_torque_dof_index)
            if self.arm_dof_indices is None
            else tuple(self.arm_dof_indices)
        )
        if len(arm_indices) != 2 or len(set(arm_indices)) != 2 or min(arm_indices) < 0 or max(arm_indices) >= model.nbQ():
            raise ValueError("arm_dof_indices must identify exactly two valid, distinct arm DoFs.")
        if self.geometry.profile.crank_torque_dof_index in arm_indices:
            raise ValueError("arm_dof_indices must not include the crank DoF.")
        crank_index = self.geometry.profile.crank_torque_dof_index
        if not 0 <= crank_index < len(dof_names):
            raise ValueError("The reduced profile has an invalid crank DoF index.")
        crank_axis = dof_names[crank_index].rsplit("Rot", maxsplit=1)[-1].lower()
        axis_indices = {"x": 0, "y": 1, "z": 2}
        if crank_axis not in axis_indices:
            raise ValueError(
                "The endpoint-force adapter requires a Cartesian rotational "
                f"crank DoF, received '{dof_names[crank_index]}'."
            )
        plane_indices = tuple(
            index for index in range(3) if index != axis_indices[crank_axis]
        )
        names = [model.muscle(index).name().to_string() for index in range(model.nbMuscles())]
        profile_names = self.geometry.profile.muscle_names
        missing = set(self.geometry.muscle_map.values()) - set(names)
        if missing:
            raise ValueError(f"Profile muscles missing from bioMod: {', '.join(sorted(missing))}.")
        evaluator = None
        # biorbd_casadi otherwise rebuilds the same expression graph for every
        # measurement.  Compile the four required quantities once; retaining a
        # numerical-biorbd fallback keeps this module usable in both setups.
        try:
            from casadi import Function, MX

            q = MX.sym("endpoint_q", model.nbQ())
            qdot = MX.sym("endpoint_qdot", model.nbQdot())
            qddot = MX.sym("endpoint_qddot", model.nbQddot())

            def symbolic(value):
                return value.to_mx() if hasattr(value, "to_mx") else value

            markers = model.markers(q)
            jacobian = model.markersJacobian(q)[marker_names.index(self.endpoint_marker)]
            evaluator = Function(
                "planar_endpoint_geometry",
                [q, qdot, qddot],
                [
                    symbolic(markers[marker_names.index(self.endpoint_marker)]),
                    symbolic(markers[marker_names.index(self.center_marker)]),
                    symbolic(jacobian),
                    symbolic(model.musclesLengthJacobian(q)),
                    symbolic(model.InverseDynamics(q, qdot, qddot)),
                ],
            )
        except (ImportError, AttributeError, RuntimeError, TypeError):
            pass
        return (
            model,
            marker_names.index(self.endpoint_marker),
            marker_names.index(self.center_marker),
            arm_indices,
            plane_indices,
            names,
            evaluator,
        )

    @staticmethod
    def _array(value) -> np.ndarray:
        if hasattr(value, "to_array"):
            value = value.to_array()
        elif hasattr(value, "to_mx"):
            from casadi import evalf
            value = evalf(value.to_mx())
        return np.asarray(value, dtype=float)

    def _sample_maps(self, theta: float, omega: float, theta_acceleration: float, *, model, endpoint_index: int,
                     center_index: int, arm_indices: tuple[int, int], plane_indices: tuple[int, int], muscle_names: list[str], evaluator=None):
        q, qdot, qddot = self.geometry.profile.kinematics.generalized_kinematics(theta, omega, theta_acceleration)
        if evaluator is None:
            markers = model.markers(q)
            endpoint = self._array(markers[endpoint_index]).reshape(-1)[list(plane_indices)]
            centre = self._array(markers[center_index]).reshape(-1)[list(plane_indices)]
            full_marker_jacobian = self._array(model.markersJacobian(q)[endpoint_index])
            full_length_jacobian = self._array(model.musclesLengthJacobian(q))
            inverse_dynamics_full = self._array(model.InverseDynamics(q, qdot, qddot)).reshape(-1)
        else:
            endpoint, centre, full_marker_jacobian, full_length_jacobian, inverse_dynamics_full = (
                self._array(value) for value in evaluator(q, qdot, qddot)
            )
            endpoint = endpoint.reshape(-1)[list(plane_indices)]
            centre = centre.reshape(-1)[list(plane_indices)]
        radial = endpoint - centre
        radius = float(np.linalg.norm(radial))
        if not np.isfinite(radius) or radius <= 1e-9:
            raise ValueError("Endpoint and crank centre coincide; radial force is undefined.")
        radial /= radius
        marker_jacobian = full_marker_jacobian[list(plane_indices), :][:, list(arm_indices)]
        endpoint_velocity_per_theta = full_marker_jacobian[list(plane_indices), :] @ self.geometry.profile.kinematics.tangent(theta)
        tangent_norm = float(np.linalg.norm(endpoint_velocity_per_theta))
        if not np.isfinite(tangent_norm) or tangent_norm <= 1e-9:
            raise ValueError("Endpoint has zero crank-tangent velocity; tangential force is undefined.")
        tangent = endpoint_velocity_per_theta / tangent_norm
        length_jacobian = full_length_jacobian[:, list(arm_indices)]
        inverse_dynamics = inverse_dynamics_full[list(arm_indices)]
        return marker_jacobian, length_jacobian, tangent, radial, radius, inverse_dynamics

    def unit_force_gains(
        self,
        muscle_name: str,
        angle_rad: np.ndarray,
        *,
        omega_rad_s: float | np.ndarray,
        theta_acceleration_rad_s2: float | np.ndarray = 0.0,
    ) -> dict:
        """Return the two endpoint-force gains for one bioMod muscle.

        This method needs only the mechanical Jacobians.  It therefore remains
        useful for diagnosing a bioMod whose Ding force--length geometry is
        invalid; :meth:`apply` still refuses such a profile before it can be
        used for parameter identification.
        """
        angle = np.asarray(angle_rad, dtype=float).reshape(-1)
        omega = np.asarray(omega_rad_s, dtype=float)
        alpha = np.asarray(theta_acceleration_rad_s2, dtype=float)
        if omega.ndim == 0:
            omega = np.full(angle.shape, omega)
        if alpha.ndim == 0:
            alpha = np.full(angle.shape, alpha)
        if (angle.size == 0 or omega.shape != angle.shape or alpha.shape != angle.shape
                or not all(np.all(np.isfinite(value)) for value in (angle, omega, alpha))):
            raise ValueError("angle_rad, omega_rad_s and theta_acceleration_rad_s2 must be finite and aligned.")
        if muscle_name not in self.geometry.muscle_map:
            raise ValueError(f"No explicit profile mapping for {muscle_name}.")
        model, endpoint_index, center_index, arm_indices, plane_indices, muscle_names, evaluator = self._model_and_indices()
        muscle_index = muscle_names.index(self.geometry.muscle_map[muscle_name])
        tangent_gain, radial_gain, radii = (
            np.empty(angle.size), np.empty(angle.size), np.empty(angle.size),
        )
        for index, (theta, speed, acceleration) in enumerate(zip(angle + self.geometry.angle_offset_rad, omega, alpha)):
            jacobian, length_jacobian, tangent, radial, radii[index], _ = self._sample_maps(
                float(theta), float(speed), float(acceleration), model=model,
                endpoint_index=endpoint_index, center_index=center_index,
                arm_indices=arm_indices, plane_indices=plane_indices,
                muscle_names=muscle_names, evaluator=evaluator,
            )
            # tau_muscle = -dL/dq * F.  This is the action exerted by the arm
            # on the pedal, so its sign agrees with the sensor convention.
            endpoint_force = planar_endpoint_force(jacobian, -length_jacobian[muscle_index], condition_limit=self.condition_limit)
            tangent_gain[index] = float(tangent @ endpoint_force)
            radial_gain[index] = float(radial @ endpoint_force)
        return {
            "tangential_gain_n_per_n": tangent_gain,
            "radial_gain_n_per_n": radial_gain,
            "endpoint_radius_m": radii,
            "crank_radius_m": float(np.mean(radii)),
            "provenance": self.provenance,
        }

    def audit_unit_force_gains(
        self,
        muscle_name: str,
        angle_rad: np.ndarray,
        *,
        omega_rad_s: float | np.ndarray,
    ) -> dict:
        """Check endpoint virtual work against the saved reduced torque map."""
        angle = np.asarray(angle_rad, dtype=float).reshape(-1)
        gains = self.unit_force_gains(
            muscle_name, angle, omega_rad_s=omega_rad_s,
        )
        profile_name = self.geometry.muscle_map.get(muscle_name)
        if profile_name is None:
            raise ValueError(f"No explicit profile mapping for {muscle_name}.")
        index = self.geometry.profile.muscle_names.index(profile_name)
        reference = np.array(
            [
                self.geometry.profile.coefficient_values(value + self.geometry.angle_offset_rad)["muscle_effectiveness"][index]
                for value in angle
            ],
            dtype=float,
        )
        endpoint_torque = gains["tangential_gain_n_per_n"] * gains["endpoint_radius_m"]
        error = endpoint_torque - reference
        return {
            "muscle_name": muscle_name,
            "samples": int(angle.size),
            "maximum_absolute_torque_error_nm_per_n": float(np.max(np.abs(error))),
            "root_mean_square_torque_error_nm_per_n": float(np.sqrt(np.mean(error**2))),
            "endpoint_torque_nm_per_n": endpoint_torque,
            "reduced_profile_torque_nm_per_n": reference,
            "provenance": gains["provenance"],
        }

    def audit_fibre_lengths(self, angle_rad: np.ndarray) -> dict:
        """Audit actual bioMod fibre lengths along the reduced crank manifold.

        A negative fibre length is a model/path error, not a value to clip or
        take the absolute value of: doing so would change both force--length
        and the derivative used by the Ding ODE.  This audit makes that gate
        explicit before a two-arm identification campaign is launched.
        """
        angle = np.asarray(angle_rad, dtype=float).reshape(-1)
        if angle.size < 2 or not np.all(np.isfinite(angle)):
            raise ValueError("angle_rad must contain at least two finite samples.")
        model, _, _, _, _, names, _ = self._model_and_indices()
        lengths = np.empty((angle.size, len(names)))
        optimal = np.empty(len(names))
        for sample, value in enumerate(angle + self.geometry.angle_offset_rad):
            q = self.geometry.profile.kinematics.q(float(value))
            kinematics = model.UpdateKinematicsCustom(q)
            for index in range(len(names)):
                muscle = model.muscle(index)
                lengths[sample, index] = float(self._array(muscle.length(kinematics, q, True)).reshape(-1)[0])
                if sample == 0:
                    optimal[index] = float(self._array(muscle.characteristics().optimalLength()).reshape(-1)[0])
        if np.any(~np.isfinite(lengths)) or np.any(~np.isfinite(optimal)) or np.any(optimal <= 0):
            raise ValueError("bioMod returned a non-finite or nonpositive optimal muscle length.")
        normalized = lengths / optimal
        per_muscle = {
            name: {
                "minimum_fibre_length_m": float(np.min(lengths[:, index])),
                "maximum_fibre_length_m": float(np.max(lengths[:, index])),
                "minimum_normalized_fibre_length": float(np.min(normalized[:, index])),
                "maximum_normalized_fibre_length": float(np.max(normalized[:, index])),
                "strictly_positive": bool(np.all(lengths[:, index] > 0.0)),
            }
            for index, name in enumerate(names)
        }
        invalid = [name for name, entry in per_muscle.items() if not entry["strictly_positive"]]
        return {
            "samples": int(angle.size),
            "all_fibre_lengths_strictly_positive": not invalid,
            "invalid_muscles": invalid,
            "per_muscle": per_muscle,
            "provenance": self.provenance,
        }

    def apply(
        self,
        spec: CyclingTrialSpec,
        *,
        omega_rad_s: float | np.ndarray,
        theta_acceleration_rad_s2: float | np.ndarray = 0.0,
        include_radial: bool = True,
    ) -> CyclingTrialSpec:
        """Add model-derived two-axis gains to a Ding-identification trial.

        The tangential gain replaces the torque/radius surrogate in the
        synthetic observation, while the radial gain is calculated from the
        same endpoint force vector.  No inertial baseline is invented: fit or
        subtract the no-stimulation trace for the subject separately.
        """
        base = self.geometry.apply(spec, omega_rad_s=omega_rad_s, radial_gain=np.zeros_like(spec.time_s))
        gains = self.unit_force_gains(
            spec.muscle_name, spec.angle_rad, omega_rad_s=omega_rad_s,
            theta_acceleration_rad_s2=theta_acceleration_rad_s2,
        )
        radii = gains["endpoint_radius_m"]
        mean_radius = float(np.mean(radii))
        radius_variation = float(np.max(np.abs(radii - mean_radius)) / mean_radius)
        if radius_variation > self.radius_relative_tolerance:
            raise ValueError(
                "bioMod endpoint radius changes along the trial beyond the "
                f"allowed {self.radius_relative_tolerance:.3g} relative fit error."
            )
        return replace(
            base,
            # Use the physical radius, not the generic synthetic default.  It
            # keeps Ft * r == muscle torque by construction.
            crank_radius_m=mean_radius,
            moment_arm_m=gains["tangential_gain_n_per_n"] * mean_radius,
            radial_gain=gains["radial_gain_n_per_n"] if include_radial else None,
        )

    def force_from_full_dynamics(
        self,
        theta_rad: np.ndarray,
        omega_rad_s: np.ndarray,
        muscle_forces_n: np.ndarray,
        *,
        theta_acceleration_rad_s2: np.ndarray | float = 0.0,
    ) -> dict:
        """Recover total two-axis pedal force from a measured/modelled muscle trace.

        The returned force uses the ``arm -> pedal`` convention.  It includes
        the inertial/gravitational baseline from inverse dynamics and should be
        used for comparing an actual two-axis sensor trace.  All muscles in
        the bioMod must be supplied; unmodelled voluntary/residual torques are
        deliberately not hidden in the result.
        """
        theta = np.asarray(theta_rad, dtype=float).reshape(-1)
        omega = np.asarray(omega_rad_s, dtype=float)
        alpha = np.asarray(theta_acceleration_rad_s2, dtype=float)
        if omega.ndim == 0:
            omega = np.full(theta.shape, omega)
        if alpha.ndim == 0:
            alpha = np.full(theta.shape, alpha)
        model, endpoint_index, center_index, arm_indices, plane_indices, names, evaluator = self._model_and_indices()
        forces = np.asarray(muscle_forces_n, dtype=float)
        if theta.size == 0 or omega.shape != theta.shape or alpha.shape != theta.shape or forces.shape != (theta.size, len(names)):
            raise ValueError("theta, omega, acceleration and all bioMod muscle forces must align by sample.")
        if not all(np.all(np.isfinite(value)) for value in (theta, omega, alpha, forces)):
            raise ValueError("Dynamics inputs must be finite.")
        cartesian = np.empty((theta.size, 2))
        tangent_values, radial_values = np.empty(theta.size), np.empty(theta.size)
        for index, values in enumerate(zip(theta, omega, alpha, forces)):
            angle, speed, acceleration, sample_forces = values
            jacobian, length_jacobian, tangent, radial, _, inverse_dynamics = self._sample_maps(
                float(angle + self.geometry.angle_offset_rad), float(speed), float(acceleration), model=model,
                endpoint_index=endpoint_index, center_index=center_index,
                arm_indices=arm_indices, plane_indices=plane_indices,
                muscle_names=names, evaluator=evaluator,
            )
            muscle_torque = -length_jacobian.T @ sample_forces
            cartesian[index] = planar_endpoint_force(
                jacobian, muscle_torque - inverse_dynamics, condition_limit=self.condition_limit
            )
            tangent_values[index] = tangent @ cartesian[index]
            radial_values[index] = radial @ cartesian[index]
        return {
            "force_arm_on_pedal_n": cartesian,
            "tangential_n": tangent_values,
            "radial_n": radial_values,
            "provenance": self.provenance,
        }
