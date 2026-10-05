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
