"""Bioptim model combining reduced cycling mechanics with exact Ding states."""

from __future__ import annotations

import math
from collections.abc import Sequence
from copy import deepcopy

import numpy as np

from casadi import MX, SX, vertcat
from bioptim import (
    ConfigureVariables,
    DynamicsEvaluation,
    DynamicsFunctions,
    NonLinearProgram,
    OdeSolver,
    StateDynamics,
)

from cocofest.dynamics.reduced_cycling import PeriodicFourierSeries, ReducedCyclingDynamics
from cocofest.models.ding2007.ding2007 import DingModelPulseWidthFrequency
from cocofest.models.fes_model import FesModel
from cocofest.models.state_configure import StateConfigure
from cocofest.optimization.pulse_width_slew import (
    LIFTING_FORMULATION, auxiliary_configuration, auxiliary_rhs,
    validate_max_step, validate_slew_formulation,
)
from cocofest.optimization.pulse_width_rate import (
    CONTROL_MODE as PW_RATE_MODE, validate_rate_mode, rate_configuration,
    rate_rhs, rate_keys, sampled_pulse_width,
)


def _bilateral_muscle_names(names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(names)
    if len(names) != 4 or len(set(names)) != 4 or any(
        not isinstance(name, str) or not name or name.startswith(("right_", "left_"))
        for name in names
    ):
        raise ValueError("Bilateral duplication requires four unique, unsided muscle names.")
    return tuple(f"{side}_{name}" for side in ("right", "left") for name in names)


def duplicate_bilateral_muscles(muscles_model: Sequence[FesModel]) -> list[FesModel]:
    """Clone a configured arm into independent right/left Ding models.

    Copying the objects retains their exact class, effective parameters and
    stimulation history, including periodic Ding variants. No state, control or
    mutable stimulation history is shared between the two arms.
    """
    muscles = list(muscles_model)
    names = _bilateral_muscle_names([muscle.muscle_name for muscle in muscles])
    result = []
    for index, name in enumerate(names):
        muscle = deepcopy(muscles[index % len(muscles)])
        muscle._muscle_name = name
        result.append(muscle)
    return result


def make_bilateral_reduced_dynamics(
    reduced_dynamics: ReducedCyclingDynamics, *, phase_offset_rad: float = math.pi,
) -> ReducedCyclingDynamics:
    """Add a phase-shifted symmetric arm to a shared reduced mechanical profile.

    Right-arm effectiveness and Hill geometry use ``theta``; the left uses
    ``theta + phase_offset_rad`` with the same angular velocity and torque sign.
    The default puts the two handles half a revolution apart. Fourier shifting
    is exact and preserves smooth numeric and symbolic dynamics.

    This first approximation retains the original inertia, gravity, quadratic
    velocity term, external torque and reference-arm kinematics ONCE. It models
    two active arms acting on shared reference mechanics, not the full mass and
    gravity of a bilateral multibody assembly. Replace these mechanics with a
    calibrated bilateral profile for biomechanical validation. In particular,
    ``kinematics`` remains the reference arm used by existing trajectory plots.
    """
    names = _bilateral_muscle_names(reduced_dynamics.muscle_names)
    if not math.isfinite(phase_offset_rad):
        raise ValueError("The bilateral phase offset must be finite.")

    def expanded_series(series, rows, shifted_rows):
        offset = series.offset[rows].copy()
        cosine = series.cosine[rows].copy()
        sine = series.sine[rows].copy()
        phase = reduced_dynamics.kinematics.direction * float(phase_offset_rad)
        angles = np.arange(1, series.order + 1) * phase
        for row in shifted_rows:
            original_cosine = cosine[row].copy()
            original_sine = sine[row].copy()
            cosine[row] = original_cosine * np.cos(angles) + original_sine * np.sin(angles)
            sine[row] = original_sine * np.cos(angles) - original_cosine * np.sin(angles)
        return PeriodicFourierSeries(offset, cosine, sine)

    # Shared M/g/c, four right muscles, four left muscles, one external load.
    coefficients = expanded_series(
        reduced_dynamics.coefficients, [0, 1, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7], range(7, 11),
    )
    geometry = None
    if reduced_dynamics.muscle_geometry is not None:
        # All normalized lengths followed by all velocities per crank speed.
        geometry = expanded_series(
            reduced_dynamics.muscle_geometry,
            [0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7],
            [4, 5, 6, 7, 12, 13, 14, 15],
        )
    return ReducedCyclingDynamics(
        kinematics=reduced_dynamics.kinematics,
        coefficients=coefficients,
        muscle_names=names,
        crank_torque_dof_index=reduced_dynamics.crank_torque_dof_index,
        muscle_geometry=geometry,
        source_model_sha256=reduced_dynamics.source_model_sha256,
    )


class ReducedFesCyclingModel(StateDynamics):
    """Ding states plus the physical crank mechanics.

    The five-state Ding models and their pulse-width controls are retained
    exactly.  The three-coordinate constrained multibody subsystem is replaced
    by the tangent-projected two-state system ``theta_dot = omega`` and
    ``omega_dot = f(theta, omega, muscle forces)``.

    When ``isokinetic`` is enabled, ``omega`` is constant by construction,
    ``theta`` advances at ``isokinetic_omega``. The required load torque is
    eliminated analytically from the inverse mechanical balance and evaluated
    at every integrator stage. ``E_prod`` integrates the net work produced
    against that load. The default remains the original 22-state
    forward-dynamics model.

    Optional PW slew carriers and bounded delta_pw controls only constrain
    successive physical PW controls. Ding dynamics keep using the physical
    zero-order-held controls, never the carrier or its increment.
    """

    def __init__(
        self,
        *,
        reduced_dynamics: ReducedCyclingDynamics,
        muscles_model: Sequence[FesModel],
        external_crank_torque: float = 0.0,
        isokinetic: bool = False,
        isokinetic_omega: float = -2.0 * math.pi,
        activate_force_length_relationship: bool = True,
        activate_force_velocity_relationship: bool = True,
        activate_passive_force_relationship: bool = True,
        name: str = "reduced_fes_cycling",
        pulse_width_max_step_s: float | None = None,
        pulse_width_slew_formulation: str = LIFTING_FORMULATION,
        pulse_width_interval_s: float | None = None,
        pulse_width_control_mode: str = "direct",
        pulse_width_max_rate_s_per_s: float | None = None,
        dynamic_mechanical_residual: str = "direct",
    ):
        super().__init__()
        self.reduced_dynamics = reduced_dynamics
        self.muscles_dynamics_model = list(muscles_model)
        self.external_crank_torque = float(external_crank_torque)
        self.isokinetic = bool(isokinetic)
        self.isokinetic_omega = float(isokinetic_omega)
        self.activate_force_length_relationship = bool(
            activate_force_length_relationship
        )
        self.activate_force_velocity_relationship = bool(
            activate_force_velocity_relationship
        )
        self.activate_passive_force_relationship = bool(
            activate_passive_force_relationship
        )
        self._name = str(name)
        self.pulse_width_max_step_s = validate_max_step(pulse_width_max_step_s)
        self.pulse_width_slew_formulation = validate_slew_formulation(
            pulse_width_slew_formulation
        )
        self.pulse_width_interval_s = pulse_width_interval_s
        self.pulse_width_control_mode = pulse_width_control_mode
        self.pulse_width_max_rate_s_per_s = validate_rate_mode(
            pulse_width_control_mode, pulse_width_max_rate_s_per_s
        )
        self.dynamic_mechanical_residual = str(dynamic_mechanical_residual)
        if self.dynamic_mechanical_residual not in ("direct", "implicit_inverse"):
            raise ValueError(
                "dynamic_mechanical_residual must be 'direct' or "
                "'implicit_inverse'."
            )
        if self.isokinetic and self.dynamic_mechanical_residual != "direct":
            raise ValueError(
                "The isokinetic reduced formulation already eliminates the "
                "load by inverse dynamics; dynamic_mechanical_residual applies "
                "only to the dynamic formulation."
            )
        if pulse_width_control_mode == PW_RATE_MODE:
            from cocofest.models.ding2007.ding2007_with_fatigue_periodic_node import (
                DingModelPulseWidthFrequencyWithFatiguePeriodicNode,
            )
            if self.pulse_width_max_step_s is not None:
                raise ValueError("PW rate-state mode and auxiliary slew lift are mutually exclusive.")
            if not all(isinstance(m, DingModelPulseWidthFrequencyWithFatiguePeriodicNode)
                       for m in self.muscles_dynamics_model):
                raise ValueError("PW rate-state mode requires periodic-node Ding models with interval-local timing.")
        if (self.pulse_width_max_step_s is not None or pulse_width_control_mode == PW_RATE_MODE) and (
            pulse_width_interval_s is None or not math.isfinite(pulse_width_interval_s)
            or pulse_width_interval_s <= 0
        ):
            raise ValueError("The PW slew lift requires a finite positive stimulation interval.")

        if not math.isfinite(self.isokinetic_omega):
            raise ValueError("isokinetic_omega must be finite.")
        if self.isokinetic and self.isokinetic_omega >= 0.0:
            raise ValueError(
                "isokinetic_omega must be strictly negative for the current "
                "cycling convention."
            )
        if self.isokinetic and self.external_crank_torque != 0.0:
            raise ValueError(
                "external_crank_torque cannot be used together with the "
                "analytically eliminated isokinetic load."
            )

        model_names = tuple(
            str(model.muscle_name) for model in self.muscles_dynamics_model
        )
        if model_names != self.reduced_dynamics.muscle_names:
            raise ValueError(
                "Reduced profile muscles and Ding models must have identical "
                f"ordering; received {model_names} and "
                f"{self.reduced_dynamics.muscle_names}."
            )
        if len(model_names) not in (4, 8):
            raise ValueError(
                "The cycling reduction expects four unilateral or eight bilateral "
                f"muscles, received {len(model_names)}."
            )
        if len(set(model_names)) != len(model_names):
            raise ValueError("Reduced cycling muscle names must be unique.")
        invalid_state_models = [
            f"{model.muscle_name}:{model.nb_state}"
            for model in self.muscles_dynamics_model
            if int(model.nb_state) != 5
        ]
        if invalid_state_models:
            raise ValueError(
                "Reduced cycling requires exactly five Ding states per muscle "
                "(Cn, F, A, Tau1, Km); incompatible models: "
                + ", ".join(invalid_state_models)
                + "."
            )
        if (
            self.activate_force_length_relationship
            or self.activate_force_velocity_relationship
            or self.activate_passive_force_relationship
        ) and self.reduced_dynamics.muscle_geometry is None:
            raise ValueError(
                "The reduced profile must include muscle geometry when Hill "
                "relationships are enabled."
            )
        self.dynamic_inverse_reference_inertia = 1.0
        if self.dynamic_mechanical_residual == "implicit_inverse":
            # A positive, fixed reference makes the experimental implicit
            # balance a state-dependent row scaling of the current omega
            # collocation defect. It therefore retains exactly the direct-model
            # feasible set.
            progress = np.linspace(0.0, 1.0, 512, endpoint=False)
            inertias = np.asarray(
                self.reduced_dynamics.coefficients.evaluate(progress)[
                    self.reduced_dynamics._inertia_index
                ],
                dtype=float,
            )
            self.dynamic_inverse_reference_inertia = float(np.median(inertias))
            if not math.isfinite(self.dynamic_inverse_reference_inertia) or (
                self.dynamic_inverse_reference_inertia <= 0.0
            ):
                raise ValueError("The reduced effective inertia reference must be positive.")

    @property
    def name(self) -> str:
        return self._name

    @property
    def name_dofs(self) -> list[str]:
        return ["theta"]

    @property
    def nb_state(self) -> int:
        return 5 * len(self.muscles_dynamics_model) + 2 + int(self.isokinetic) + (
            len(self.muscles_dynamics_model)
            if self.uses_pulse_width_slew_lifting or self.pulse_width_control_mode == PW_RATE_MODE else 0
        )

    @property
    def uses_pulse_width_slew_lifting(self) -> bool:
        """Whether this model carries the historical auxiliary ΔPW variables."""
        return (
            self.pulse_width_max_step_s is not None
            and self.pulse_width_slew_formulation == LIFTING_FORMULATION
        )

    @property
    def contact_types(self) -> tuple:
        return ()

    @property
    def state_configuration_functions(self):
        state_dictionary = StateConfigure().state_dictionary
        functions = []
        for muscle_model in self.muscles_dynamics_model:
            for state_key in muscle_model.name_dof:
                if state_key not in state_dictionary:
                    continue
                functions.append(
                    lambda ocp,
                    nlp,
                    state_key=state_key,
                    muscle_model=muscle_model: state_dictionary[state_key](
                        ocp=ocp,
                        nlp=nlp,
                        as_states=True,
                        as_controls=False,
                        muscle_name=muscle_model.muscle_name,
                    )
                )
        functions.extend(
            (
                lambda ocp, nlp: ConfigureVariables.configure_new_variable(
                    "theta", ["physical_crank_angle"], ocp, nlp, as_states=True
                ),
                lambda ocp, nlp: ConfigureVariables.configure_new_variable(
                    "omega", ["physical_crank_angular_velocity"], ocp, nlp, as_states=True
                ),
            )
        )
        if self.isokinetic:
            functions.append(
                lambda ocp, nlp: ConfigureVariables.configure_new_variable(
                    "E_prod", ["net_external_work"], ocp, nlp, as_states=True
                )
            )
        if self.uses_pulse_width_slew_lifting:
            functions.extend(auxiliary_configuration(self.muscles_dynamics_model, states=True))
        if self.pulse_width_control_mode == PW_RATE_MODE:
            functions.extend(rate_configuration(self.muscles_dynamics_model, states=True))
        return functions

    @property
    def control_configuration_functions(self):
        if self.pulse_width_control_mode == PW_RATE_MODE:
            return rate_configuration(self.muscles_dynamics_model, states=False)
        functions = []
        for muscle_model in self.muscles_dynamics_model:
            if isinstance(muscle_model, DingModelPulseWidthFrequency):
                functions.append(
                    lambda ocp,
                    nlp,
                    muscle_model=muscle_model: StateConfigure().configure_last_pulse_width(
                        ocp, nlp, muscle_model.muscle_name
                    )
                )
        if self.uses_pulse_width_slew_lifting:
            functions.extend(auxiliary_configuration(self.muscles_dynamics_model, states=False))
        return functions

    @property
    def algebraic_configuration_functions(self):
        return []

    @property
    def extra_configuration_functions(self):
        return []

    @property
    def extra_dynamics(self):
        return None

    def serialize(self):
        return (
            ReducedFesCyclingModel,
            {
                "reduced_dynamics": self.reduced_dynamics,
                "muscles_model": self.muscles_dynamics_model,
                "external_crank_torque": self.external_crank_torque,
                "isokinetic": self.isokinetic,
                "isokinetic_omega": self.isokinetic_omega,
                "activate_force_length_relationship": self.activate_force_length_relationship,
                "activate_force_velocity_relationship": self.activate_force_velocity_relationship,
                "activate_passive_force_relationship": self.activate_passive_force_relationship,
                "name": self._name,
                "pulse_width_max_step_s": self.pulse_width_max_step_s,
                "pulse_width_slew_formulation": self.pulse_width_slew_formulation,
                "pulse_width_interval_s": self.pulse_width_interval_s,
                "pulse_width_control_mode": self.pulse_width_control_mode,
                "pulse_width_max_rate_s_per_s": self.pulse_width_max_rate_s_per_s,
                "dynamic_mechanical_residual": self.dynamic_mechanical_residual,
            },
        )

    def external_torque_effectiveness(self, theta):
        """Return ``b_ext(theta)`` from the fitted reduced profile."""

        values = self.reduced_dynamics.coefficients.casadi(
            self.reduced_dynamics.kinematics.progress(theta)
        )
        return values[self.reduced_dynamics._external_index]

    def mechanical_equilibrium_residual(
        self,
        theta,
        omega,
        muscle_forces,
        tau_load,
    ):
        """Return the isokinetic inverse-dynamics balance residual.

        This is exactly the numerator used by
        :meth:`ReducedCyclingDynamics.casadi_acceleration`, exposed separately
        so an OCP constraint can impose zero acceleration without introducing
        the effective inertia.
        """

        from casadi import dot

        values = self.reduced_dynamics.coefficients.casadi(
            self.reduced_dynamics.kinematics.progress(theta)
        )
        return (
            dot(values[self.reduced_dynamics._muscle_slice], muscle_forces)
            + values[self.reduced_dynamics._external_index] * tau_load
            - values[self.reduced_dynamics._gravity_index]
            - values[self.reduced_dynamics._velocity_index] * omega**2
        )

    def required_load_torque(self, theta, omega, muscle_forces):
        """Return the load torque that makes the reduced acceleration zero."""

        from casadi import dot

        values = self.reduced_dynamics.coefficients.casadi(
            self.reduced_dynamics.kinematics.progress(theta)
        )
        return (
            values[self.reduced_dynamics._gravity_index]
            + values[self.reduced_dynamics._velocity_index] * omega**2
            - dot(values[self.reduced_dynamics._muscle_slice], muscle_forces)
        ) / values[self.reduced_dynamics._external_index]

    def dynamics(
        self,
        time: MX | SX,
        states: MX | SX,
        controls: MX | SX,
        parameters: MX | SX,
        algebraic_states: MX | SX,
        numerical_data_timeseries: MX | SX,
        nlp: NonLinearProgram,
    ) -> DynamicsEvaluation:
        theta = DynamicsFunctions.get(nlp.states["theta"], states)
        omega = DynamicsFunctions.get(nlp.states["omega"], states)
        mechanical_omega = self.isokinetic_omega if self.isokinetic else omega
        if (
            self.activate_force_length_relationship
            or self.activate_force_velocity_relationship
            or self.activate_passive_force_relationship
        ):
            force_length, force_velocity, passive_force = (
                self.reduced_dynamics.casadi_muscle_relationships(
                    theta, mechanical_omega
                )
            )
        else:
            force_length = [1.0] * len(self.muscles_dynamics_model)
            force_velocity = [1.0] * len(self.muscles_dynamics_model)
            passive_force = [0.0] * len(self.muscles_dynamics_model)

        muscle_derivatives = []
        muscle_forces = []
        for muscle_index, muscle_model in enumerate(
            self.muscles_dynamics_model
        ):
            muscle_states = vertcat(
                *[
                    DynamicsFunctions.get(
                        nlp.states[f"{state_key}_{muscle_model.muscle_name}"],
                        states,
                    )
                    for state_key in muscle_model.name_dof
                ]
            )
            control_key = f"last_pulse_width_{muscle_model.muscle_name}"
            if self.pulse_width_control_mode == PW_RATE_MODE:
                _, rate_key = rate_keys(muscle_model.muscle_name)
                pulse_width = sampled_pulse_width(
                    DynamicsFunctions.get(nlp.states[control_key], states),
                    DynamicsFunctions.get(nlp.controls[rate_key], controls),
                    time, numerical_data_timeseries[1],
                )
            else:
                pulse_width = DynamicsFunctions.get(nlp.controls[control_key], controls)
            muscle_derivatives.append(
                muscle_model.dynamics(
                    time,
                    muscle_states,
                    pulse_width,
                    parameters,
                    algebraic_states,
                    numerical_data_timeseries,
                    nlp,
                    fes_model=muscle_model,
                    force_length_relationship=(
                        force_length[muscle_index]
                        if self.activate_force_length_relationship
                        else 1.0
                    ),
                    force_velocity_relationship=(
                        force_velocity[muscle_index]
                        if self.activate_force_velocity_relationship
                        else 1.0
                    ),
                    passive_force_relationship=(
                        passive_force[muscle_index]
                        if self.activate_passive_force_relationship
                        else 0.0
                    ),
                ).dxdt
            )
            muscle_forces.append(
                DynamicsFunctions.get(
                    nlp.states[f"F_{muscle_model.muscle_name}"], states
                )
            )

        if self.isokinetic:
            # Substitute the inverse balance into -tau_load*b_ext*omega.
            # This avoids a needless division/remultiplication by b_ext while
            # retaining tau_load itself for bounds and post-solve reporting.
            energy_dot = (
                self.mechanical_equilibrium_residual(
                    theta,
                    mechanical_omega,
                    vertcat(*muscle_forces),
                    0.0,
                )
                * mechanical_omega
            )
            dxdt = vertcat(
                *muscle_derivatives,
                self.isokinetic_omega,
                0.0,
                energy_dot,
            )
        else:
            omega_dot = self.reduced_dynamics.casadi_acceleration(
                theta,
                omega,
                vertcat(*muscle_forces),
                self.external_crank_torque,
            )
            dxdt = vertcat(*muscle_derivatives, omega, omega_dot)
        if self.uses_pulse_width_slew_lifting:
            dxdt = vertcat(dxdt, auxiliary_rhs(self, states, controls, nlp))
        if self.pulse_width_control_mode == PW_RATE_MODE:
            dxdt = vertcat(dxdt, rate_rhs(self, controls, nlp))
        defects = None
        if isinstance(nlp.dynamics_type.ode_solver, OdeSolver.COLLOCATION):
            defects = (
                nlp.states_dot.scaled.cx * nlp.dt - dxdt * nlp.dt
            )
            if (
                not self.isokinetic
                and self.dynamic_mechanical_residual == "implicit_inverse"
            ):
                # Current direct defect: alpha - N/I.  Multiplication by
                # I(theta)/I_ref gives (I*alpha-N)/I_ref, namely the implicit
                # inverse-balance residual, without an extra state/control.
                # Applying the factor to the existing scaled defect preserves
                # the exact direct collocation roots even when omega itself is
                # decision-scaled by Bioptim.
                values = self.reduced_dynamics.coefficients.casadi(
                    self.reduced_dynamics.kinematics.progress(theta)
                )
                factor = (
                    values[self.reduced_dynamics._inertia_index]
                    / self.dynamic_inverse_reference_inertia
                )
                omega_defect_index = 5 * len(self.muscles_dynamics_model) + 1
                defects = vertcat(
                    *[
                        factor * defects[index]
                        if index == omega_defect_index
                        else defects[index]
                        for index in range(defects.shape[0])
                    ]
                )
        return DynamicsEvaluation(dxdt=dxdt, defects=defects)
