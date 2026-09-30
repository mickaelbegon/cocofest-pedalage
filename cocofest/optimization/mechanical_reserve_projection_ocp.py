"""Fixed-parameter Bioptim binding for the projected mechanical reserve cost.

The candidate force profile and the externally computed affine margin model
are numerical parameters of one symbolic objective.  A window update changes
only their equal bounds and parameter initial guesses.  Durations, Ding
parameters, horizons, smoothing and vector dimensions belong to the graph.
This is a local surrogate, not a certificate of future mechanical feasibility.
"""

from __future__ import annotations

from hashlib import sha256
import math

import numpy as np

from .mechanical_reserve_projection import (
    LocalMechanicalMarginModel,
    _projection_constants,
    _temperature,
    projected_mechanical_reserve_casadi,
)


FORCE_PARAMETER_KEY = "rho_reserve_candidate_forces"
MARGIN_PARAMETER_KEY = "rho_reserve_affine_margin"
ACTIVATION_PARAMETER_KEY = "rho_reserve_activation"
PW_REFERENCE_PARAMETER_KEY = "rho_reserve_reference_pulse_widths"
PW_JACOBIAN_PARAMETER_KEY = "rho_reserve_force_pulse_width_jacobian"
LOCAL_PW_GRADIENT_PARAMETER_KEY = "rho_reserve_local_pw_gradient"
LOCAL_PW_REFERENCE_PARAMETER_KEY = "rho_reserve_local_pw_reference"
LOCAL_PW_CURVATURE_PARAMETER_KEY = "rho_reserve_local_pw_curvature"


def _digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.float64)
    digest = sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _forces(value, shape: tuple[int, int]) -> np.ndarray:
    values = np.array(value, dtype=float, copy=True)
    if values.shape != shape:
        raise ValueError(f"Candidate force dimensions must remain {shape}.")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Candidate forces must be finite and nonnegative.")
    values.setflags(write=False)
    return values


def affine_candidate_force_from_pulse_widths_casadi(
    candidate_pulse_widths, *, reference_pulse_widths, reference_forces,
    force_jacobian, nonnegative_force_smoothing_n: float = 1e-6,
):
    """Return a smooth nonnegative affine PW-to-endpoint-force surrogate.

    All numerical arrays use ``(muscles, phases)`` order. The Jacobian has
    ``(force_muscle, force_phase, pw_muscle, pw_phase)`` order. This helper is
    deliberately independent from the current reserve binding: it is the
    audited algebraic bridge to install only after the enclosing Bioptim
    controller has provided all physical PW decision variables.
    """
    import casadi as ca

    reference_widths = _forces(reference_pulse_widths, np.asarray(reference_forces).shape)
    reference_forces = _forces(reference_forces, reference_widths.shape)
    muscles, phases = reference_widths.shape
    jacobian = np.asarray(force_jacobian, dtype=float)
    if jacobian.shape != (muscles, phases, muscles, phases) or not np.all(np.isfinite(jacobian)):
        raise ValueError("force_jacobian must be finite with shape (M, K, M, K).")
    smoothing = float(nonnegative_force_smoothing_n)
    if not math.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("nonnegative_force_smoothing_n must be finite and strictly positive.")
    if int(candidate_pulse_widths.numel()) != muscles * phases:
        raise ValueError("candidate_pulse_widths dimension does not match the PW affine model.")
    # Bioptim's native vector layout and NumPy's C order both list all muscle
    # commands of a phase together. ``vec(PW.T)`` retains this convention.
    candidate = ca.vec(candidate_pulse_widths.T)
    reference = ca.DM(reference_widths.ravel())
    reference_force = ca.DM(reference_forces.ravel())
    derivative = ca.reshape(ca.DM(jacobian.reshape(muscles * phases, muscles * phases).ravel()),
                            muscles * phases, muscles * phases).T
    raw = reference_force + ca.mtimes(derivative, candidate - reference)
    guarded = .5 * (raw + ca.sqrt(raw**2 + smoothing**2))
    return ca.reshape(guarded, phases, muscles).T


def candidate_pulse_width_matrix_from_controller(controller, *, muscle_names, interval_count: int):
    """Extract physical PW controls from a one-phase Bioptim decision vector.

    This intentionally accepts only the audited reduced one-cycle layout:
    one scalar ``last_pulse_width_<muscle>`` control for every muscle at every
    interval. It converts Bioptim's scaled decision variables to seconds and
    returns a ``(muscles, intervals)`` CasADi matrix.
    """
    import casadi as ca

    names = tuple(str(name) for name in muscle_names)
    if not names or len(set(names)) != len(names) or type(interval_count) is not int or interval_count < 1:
        raise ValueError("muscle_names and interval_count must identify a nonempty PW grid.")
    try:
        ocp, nlp = controller.ocp, controller.ocp.nlp[0]
        controls = tuple(nlp.controls.keys())
        scalings = nlp.u_scaling
    except (AttributeError, IndexError, TypeError) as error:
        raise ValueError("Candidate PW coupling requires a one-phase Bioptim controller layout.") from error
    expected = tuple(f"last_pulse_width_{name}" for name in names)
    if controls != expected:
        raise ValueError("Candidate PW coupling requires direct PW controls in model-muscle order.")
    scales = []
    for key in expected:
        scaling = np.asarray(scalings[key].scaling, dtype=float).reshape(-1)
        if scaling.shape != (1,) or not math.isfinite(float(scaling[0])) or scaling[0] <= 0:
            raise ValueError("Candidate PW coupling requires finite positive scalar control scaling.")
        scales.append(float(scaling[0]))
    columns = []
    layout = getattr(ocp, "vector_layout", None)
    index_map = getattr(layout, "index_map", None)
    if isinstance(index_map, dict):
        vector = ocp.variables_vector
        for node in range(interval_count):
            try:
                block, columns_count = index_map[(0, "controls", node)]
            except (KeyError, TypeError) as error:
                raise ValueError("Candidate PW coupling could not locate every global control block.") from error
            if columns_count != 1 or block.stop - block.start != len(names):
                raise ValueError("Candidate PW coupling found an unsupported control block shape.")
            scaled = vector[block]
            columns.append(ca.vertcat(*[scaled[row] * scales[row] for row in range(len(names))]))
    else:
        # Penalties are assembled before the OCP creates ``vector_layout``.
        # The control container nevertheless owns the symbolic variable at
        # every node; visit it explicitly and restore the caller's node.
        try:
            prior_node = nlp.controls.node_index
            for node in range(interval_count):
                nlp.controls.node_index = node
                columns.append(ca.vertcat(*[
                    nlp.controls[key].cx * scales[row] for row, key in enumerate(expected)
                ]))
        except (AttributeError, IndexError, KeyError, TypeError) as error:
            raise ValueError("Candidate PW coupling could not access every nodal control symbol.") from error
        finally:
            if "prior_node" in locals():
                nlp.controls.node_index = prior_node
    return ca.horzcat(*columns)


def _pulse_width_force_model(value, muscles: int, intervals: int):
    from .mechanical_reserve_calibration import PulseWidthForceAffineModel

    if not isinstance(value, PulseWidthForceAffineModel):
        raise TypeError("pulse_width_force_model must be a PulseWidthForceAffineModel.")
    if value.reference_pulse_widths.shape != (muscles, intervals):
        raise ValueError("Pulse-width force model dimensions cannot change.")
    return value


def _affine_candidate_force_from_parameter_vectors(candidate_pulse_widths, *, reference_widths,
                                                    reference_forces, jacobian_vector,
                                                    muscles: int, phases: int, lag: int, smoothing: float):
    """Symbolic counterpart of :func:`affine_candidate_force_from_pulse_widths_casadi`."""
    import casadi as ca

    candidate = ca.vec(candidate_pulse_widths.T)
    size = muscles * phases
    if any(int(value.numel()) != expected for value, expected in (
        (candidate, size), (reference_widths, size), (reference_forces, size),
        (jacobian_vector, muscles * phases * lag),
    )):
        raise ValueError("Pulse-width affine parameter dimensions do not match the reserve graph.")
    rows, index = [], 0
    for muscle in range(muscles):
        for phase in range(phases):
            row = [0] * size
            for delay in range(lag):
                source_phase = phase - delay
                if source_phase >= 0:
                    row[muscle * phases + source_phase] = jacobian_vector[index]
                index += 1
            rows.append(ca.horzcat(*row))
    jacobian = ca.vertcat(*rows)
    raw = reference_forces + ca.mtimes(jacobian, candidate - reference_widths)
    guarded = .5 * (raw + ca.sqrt(raw**2 + smoothing**2))
    return ca.reshape(guarded, phases, muscles).T


def _pack_causal_pw_jacobian(model, *, lag: int) -> np.ndarray:
    values = []
    for muscle in range(model.muscle_count):
        for phase in range(model.interval_count):
            for delay in range(lag):
                source_phase = phase - delay
                values.append(0.0 if source_phase < 0 else model.force_jacobian[
                    muscle, phase, muscle, source_phase
                ])
    return np.asarray(values, dtype=float)


def _margin_model(value, muscles: int, constraints: int | None = None) -> LocalMechanicalMarginModel:
    if not isinstance(value, LocalMechanicalMarginModel):
        raise TypeError("margin_model must be a LocalMechanicalMarginModel.")
    if value.reference_states.shape != (muscles, 3):
        raise ValueError("Affine margin muscle dimensions cannot change.")
    if constraints is not None and value.reference_margins.size != constraints:
        raise ValueError("Affine margin constraint dimensions cannot change.")
    return value


def _pack_margin(model: LocalMechanicalMarginModel) -> np.ndarray:
    # All blocks use NumPy C order. The CasADi adapter below reverses each
    # matrix's reshape dimensions to preserve this exact ordering.
    return np.concatenate((model.reference_states.ravel(),
                           model.reference_margins,
                           model.state_jacobian.ravel()))


class _SymbolicMarginModel:
    def __init__(self, vector, muscles: int, constraints: int):
        import casadi as ca

        self.reference_states = ca.reshape(vector[:3 * muscles], 3, muscles).T
        self.reference_margins = vector[3 * muscles:3 * muscles + constraints]
        self.state_jacobian = ca.reshape(vector[3 * muscles + constraints:], 3 * muscles, constraints).T

    def evaluate_casadi(self, projected_states):
        import casadi as ca

        return ca.vertcat(*[
            (self.reference_margins + ca.mtimes(
                self.state_jacobian, ca.vec((state - self.reference_states).T)
            )).T
            for state in projected_states
        ])


class MechanicalReserveProjectionBinding:
    """Own one reserve objective graph and its two fixed numerical parameters."""

    _STRUCTURE_FIELDS = frozenset({
        "weight", "durations", "parameters", "horizons", "margin_temperature",
        "horizon_temperature", "muscle_count", "interval_count", "constraint_count",
        "margin_parameter_size", "use_sx", "function", "sensitivity_function", "build_count",
        "candidate_force_coupling", "pulse_width_parameter_size",
        "nonnegative_force_smoothing_n", "pulse_width_force_lag_intervals",
        "local_pulse_width_cost", "local_pulse_width_trust_s",
    })

    def __setattr__(self, name, value):
        if getattr(self, "_structure_locked", False) and name in self._STRUCTURE_FIELDS:
            raise AttributeError(f"Reserve graph structure is immutable: {name}.")
        super().__setattr__(name, value)

    def __init__(self, *, forces, durations, parameters, horizons, margin_model,
                 weight: float, margin_temperature=0.02, horizon_temperature=0.02,
                 use_sx=True, pulse_width_force_model=None,
                 nonnegative_force_smoothing_n: float = 1e-6,
                 pulse_width_force_lag_intervals: int = 3,
                 local_pulse_width_cost: bool = False,
                 local_pulse_width_trust_s: float | None = None):
        self.weight = float(weight)
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("An active reserve cost needs a finite positive weight; use configured() for zero.")
        durations, parameters, horizons, _, _ = _projection_constants(durations, parameters, horizons)
        self.durations = tuple(float(value) for value in durations)
        self.parameters = tuple(parameters)
        self.horizons = tuple(horizons)
        self.margin_temperature = _temperature(margin_temperature)
        self.horizon_temperature = _temperature(horizon_temperature)
        self.muscle_count = len(parameters)
        self.interval_count = len(durations)
        self.forces = _forces(forces, (self.muscle_count, self.interval_count))
        self.margin_model = _margin_model(margin_model, self.muscle_count)
        self.candidate_force_coupling = pulse_width_force_model is not None
        self.local_pulse_width_cost = bool(local_pulse_width_cost)
        if self.candidate_force_coupling and self.local_pulse_width_cost:
            raise ValueError("Global candidate PW coupling and local PW cost are mutually exclusive.")
        if self.local_pulse_width_cost:
            if local_pulse_width_trust_s is None:
                raise ValueError("local_pulse_width_trust_s is required for the local PW cost.")
            self.local_pulse_width_trust_s = float(local_pulse_width_trust_s)
            if not math.isfinite(self.local_pulse_width_trust_s) or self.local_pulse_width_trust_s <= 0:
                raise ValueError("local_pulse_width_trust_s must be finite and strictly positive.")
        elif local_pulse_width_trust_s is not None:
            raise ValueError("local_pulse_width_trust_s requires local_pulse_width_cost.")
        else:
            self.local_pulse_width_trust_s = None
        self.pulse_width_force_model = (
            _pulse_width_force_model(pulse_width_force_model, self.muscle_count, self.interval_count)
            if self.candidate_force_coupling else None
        )
        if self.candidate_force_coupling and not np.array_equal(
                self.forces, self.pulse_width_force_model.reference_forces):
            raise ValueError("forces must equal the PW force model reference forces.")
        if type(pulse_width_force_lag_intervals) is not int or not 1 <= pulse_width_force_lag_intervals <= self.interval_count:
            raise ValueError("pulse_width_force_lag_intervals must lie in [1, interval_count].")
        self.pulse_width_force_lag_intervals = pulse_width_force_lag_intervals
        self.pulse_width_parameter_size = self.muscle_count * self.interval_count
        self.local_pulse_width_gradient = np.zeros((self.muscle_count, self.interval_count))
        self.local_pulse_width_reference = np.zeros((self.muscle_count, self.interval_count))
        self.local_pulse_width_curvature = np.zeros((self.muscle_count, self.interval_count))
        self.nonnegative_force_smoothing_n = float(nonnegative_force_smoothing_n)
        if not math.isfinite(self.nonnegative_force_smoothing_n) or self.nonnegative_force_smoothing_n <= 0:
            raise ValueError("nonnegative_force_smoothing_n must be finite and strictly positive.")
        self.constraint_count = self.margin_model.reference_margins.size
        self.margin_parameter_size = 3 * self.muscle_count + self.constraint_count * (1 + 3 * self.muscle_count)
        self.use_sx = bool(use_sx)
        self.function = self._build_function()
        self.sensitivity_function = self._build_sensitivity_function()
        self.build_count = 1
        self.update_count = 0
        self.activation = 0.0
        self._nlp = None
        self._solver = None
        self.solver_observation_count = 0
        self._solver_observed_at_updates = set()
        self._structure_locked = True

    @classmethod
    def configured(cls, *, weight, **kwargs):
        """A zero weight omits the entire cost, graph and ParameterList entries."""
        value = float(weight)
        if not math.isfinite(value) or value < 0:
            raise ValueError("Reserve weight must be finite and nonnegative.")
        return None if value == 0 else cls(weight=value, **kwargs)

    def _build_function(self):
        import casadi as ca

        symbol = ca.SX if self.use_sx else ca.MX
        initial_states = symbol.sym("initial_states", self.muscle_count, 3)
        force_vector = symbol.sym(FORCE_PARAMETER_KEY, self.muscle_count * self.interval_count)
        margin_vector = symbol.sym(MARGIN_PARAMETER_KEY, self.margin_parameter_size)
        activation = symbol.sym(ACTIVATION_PARAMETER_KEY)
        if self.candidate_force_coupling:
            candidate_pulse_widths = symbol.sym(
                "candidate_pulse_widths", self.muscle_count, self.interval_count
            )
            reference_widths = symbol.sym(PW_REFERENCE_PARAMETER_KEY, self.pulse_width_parameter_size)
            jacobian = symbol.sym(PW_JACOBIAN_PARAMETER_KEY,
                                  self.pulse_width_parameter_size * self.pulse_width_force_lag_intervals)
            forces = _affine_candidate_force_from_parameter_vectors(
                candidate_pulse_widths, reference_widths=reference_widths, reference_forces=force_vector,
                jacobian_vector=jacobian, muscles=self.muscle_count, phases=self.interval_count,
                lag=self.pulse_width_force_lag_intervals, smoothing=self.nonnegative_force_smoothing_n,
            )
        else:
            candidate_pulse_widths = reference_widths = jacobian = None
            forces = ca.reshape(force_vector, self.interval_count, self.muscle_count).T
        model = _SymbolicMarginModel(margin_vector, self.muscle_count, self.constraint_count)
        result = projected_mechanical_reserve_casadi(
            initial_states, forces, self.durations, self.parameters, self.horizons, model,
            margin_temperature=self.margin_temperature,
            horizon_temperature=self.horizon_temperature,
        )
        inputs = [initial_states]
        input_names = ["initial_states"]
        if self.candidate_force_coupling:
            inputs.extend((candidate_pulse_widths, reference_widths, force_vector, jacobian))
            input_names.extend(("candidate_pulse_widths", PW_REFERENCE_PARAMETER_KEY,
                                FORCE_PARAMETER_KEY, PW_JACOBIAN_PARAMETER_KEY))
        else:
            inputs.append(force_vector)
            input_names.append(FORCE_PARAMETER_KEY)
        inputs.extend((margin_vector, activation))
        input_names.extend((MARGIN_PARAMETER_KEY, ACTIVATION_PARAMETER_KEY))
        return ca.Function(
            "rho_mechanical_reserve_projection",
            inputs,
            [activation * self.weight * result.penalty, result.penalty, result.hard_minimum_margin, result.margins],
            input_names,
            ["weighted_cost", "penalty", "hard_minimum_margin", "margins"],
        )

    def _build_sensitivity_function(self):
        """Differentiate the fixed reserve graph for an auditable local sensitivity.

        The force derivative is deliberately labelled *frozen*: candidate
        forces are numerical parameters in the current RHO integration, so it
        is not yet a pulse-width derivative.  The state derivative is the
        path that is presently connected to candidate PW through the Ding
        dynamics.
        """
        import casadi as ca

        symbol = ca.SX if self.use_sx else ca.MX
        states = symbol.sym("initial_states", self.muscle_count, 3)
        forces = symbol.sym(FORCE_PARAMETER_KEY, self.muscle_count * self.interval_count)
        margins = symbol.sym(MARGIN_PARAMETER_KEY, self.margin_parameter_size)
        activation = symbol.sym(ACTIVATION_PARAMETER_KEY)
        if self.candidate_force_coupling:
            candidate = symbol.sym("candidate_pulse_widths", self.muscle_count, self.interval_count)
            reference_widths = symbol.sym(PW_REFERENCE_PARAMETER_KEY, self.pulse_width_parameter_size)
            jacobian = symbol.sym(PW_JACOBIAN_PARAMETER_KEY,
                                  self.pulse_width_parameter_size * self.pulse_width_force_lag_intervals)
            cost = self.function(states, candidate, reference_widths, forces, jacobian, margins, activation)[0]
            inputs = [states, candidate, reference_widths, forces, jacobian, margins, activation]
            names = ["initial_states", "candidate_pulse_widths", PW_REFERENCE_PARAMETER_KEY,
                     FORCE_PARAMETER_KEY, PW_JACOBIAN_PARAMETER_KEY, MARGIN_PARAMETER_KEY,
                     ACTIVATION_PARAMETER_KEY]
            gradients = [ca.jacobian(cost, states), ca.jacobian(cost, ca.vec(candidate.T))]
            output_names = ["terminal_slow_state_gradient", "candidate_pulse_width_gradient"]
        else:
            cost = self.function(states, forces, margins, activation)[0]
            inputs = [states, forces, margins, activation]
            names = ["initial_states", FORCE_PARAMETER_KEY, MARGIN_PARAMETER_KEY, ACTIVATION_PARAMETER_KEY]
            gradients = [ca.jacobian(cost, states), ca.jacobian(cost, forces)]
            output_names = ["terminal_slow_state_gradient", "frozen_force_parameter_gradient"]
        return ca.Function(
            "rho_mechanical_reserve_projection_sensitivity",
            inputs, gradients, names, output_names,
        )

    def objective(self, initial_states, controller):
        """Return the scalar cost for an OCP objective callback.

        The caller supplies its (muscles, 3) slow-state matrix; this binding
        makes no assumptions about the enclosing model's state variable names.
        """
        if self.candidate_force_coupling:
            names = [model.muscle_name for model in controller.model.muscles_dynamics_model]
            candidate = candidate_pulse_width_matrix_from_controller(
                controller, muscle_names=names, interval_count=self.interval_count,
            )
            return self.function(
                initial_states, candidate, controller.parameters[PW_REFERENCE_PARAMETER_KEY].cx,
                controller.parameters[FORCE_PARAMETER_KEY].cx,
                controller.parameters[PW_JACOBIAN_PARAMETER_KEY].cx,
                controller.parameters[MARGIN_PARAMETER_KEY].cx,
                controller.parameters[ACTIVATION_PARAMETER_KEY].cx,
            )[0]
        return self.function(initial_states, controller.parameters[FORCE_PARAMETER_KEY].cx,
                             controller.parameters[MARGIN_PARAMETER_KEY].cx,
                             controller.parameters[ACTIVATION_PARAMETER_KEY].cx)[0]

    def local_pulse_width_objective(self, controller):
        """Convex one-node PW surrogate from a boundary-updated reserve tangent.

        Unlike the rejected global candidate-PW Mayer graph, this expression
        reads only the current node's controls.  The vector parameters retain
        the C-order ``(muscle, phase)`` convention used by the Ding replay.
        """
        if not self.local_pulse_width_cost:
            raise RuntimeError("The reserve binding has no local pulse-width cost.")
        import casadi as ca

        names = tuple(str(model.muscle_name) for model in controller.model.muscles_dynamics_model)
        if len(names) != self.muscle_count:
            raise ValueError("Local PW reserve binding/model muscle count mismatch.")
        phase = int(controller.node_index)
        if not 0 <= phase < self.interval_count:
            raise ValueError("Local PW reserve cost was requested outside its control grid.")
        try:
            nlp = controller.ocp.nlp[0]
            controls = ca.vertcat(*[
                controller.controls[f"last_pulse_width_{name}"].cx
                * float(np.asarray(nlp.u_scaling[f"last_pulse_width_{name}"].scaling, dtype=float).reshape(-1)[0])
                for name in names
            ])
        except (AttributeError, KeyError, IndexError, TypeError, ValueError) as error:
            raise ValueError("Local PW reserve cost requires direct scalar pulse-width controls.") from error
        indices = [muscle * self.interval_count + phase for muscle in range(self.muscle_count)]
        gradient = controller.parameters[LOCAL_PW_GRADIENT_PARAMETER_KEY].cx[indices]
        reference = controller.parameters[LOCAL_PW_REFERENCE_PARAMETER_KEY].cx[indices]
        curvature = controller.parameters[LOCAL_PW_CURVATURE_PARAMETER_KEY].cx[indices]
        delta = controls - reference
        # Keep the one-node surrogate on the same calibrated scale as the
        # binding's terminal reserve objective.  Omitting ``self.weight`` here
        # silently decouples the local PW trade-off from
        # ``--experimental-mechanical-reserve-weight``.
        return controller.parameters[ACTIVATION_PARAMETER_KEY].cx[0] * self.weight * (
            ca.dot(gradient, delta) + .5 * ca.dot(curvature, delta**2)
        )

    def parameter_options(self, *, use_sx: bool | None = None) -> dict:
        from bioptim import BoundsList, InitialGuessList, InterpolationType, ParameterList, VariableScaling

        if use_sx is not None and bool(use_sx) != self.use_sx:
            raise ValueError("ParameterList symbolic type must match the reserve objective graph.")
        parameters = ParameterList(use_sx=self.use_sx)
        bounds, initial = BoundsList(), InitialGuessList()
        for key, values in self._vectors().items():
            parameters.add(name=key, function=None, size=values.size,
                           scaling=VariableScaling(key, np.ones(values.size)))
            column = values[:, None]
            bounds.add(key, min_bound=column.copy(), max_bound=column.copy(),
                       interpolation=InterpolationType.CONSTANT)
            initial.add(key, initial_guess=column.copy())
        return {"parameters": parameters, "parameter_bounds": bounds, "parameter_init": initial}

    def _vectors(self) -> dict[str, np.ndarray]:
        values = {FORCE_PARAMETER_KEY: self.forces.ravel(),
                  MARGIN_PARAMETER_KEY: _pack_margin(self.margin_model),
                  ACTIVATION_PARAMETER_KEY: np.array([self.activation])}
        if self.candidate_force_coupling:
            values[PW_REFERENCE_PARAMETER_KEY] = self.pulse_width_force_model.reference_pulse_widths.ravel()
            values[PW_JACOBIAN_PARAMETER_KEY] = _pack_causal_pw_jacobian(
                self.pulse_width_force_model, lag=self.pulse_width_force_lag_intervals
            )
        if self.local_pulse_width_cost:
            values[LOCAL_PW_GRADIENT_PARAMETER_KEY] = self.local_pulse_width_gradient.ravel()
            values[LOCAL_PW_REFERENCE_PARAMETER_KEY] = self.local_pulse_width_reference.ravel()
            values[LOCAL_PW_CURVATURE_PARAMETER_KEY] = self.local_pulse_width_curvature.ravel()
        return values

    def local_pulse_width_cost_terms(self, *, initial_states, forces, margin_model,
                                     pulse_width_force_model):
        """Return a causal convex local PW model at a certified boundary.

        A frozen reserve derivative with respect to endpoint force is chained
        through the Ding PW tangent. Curvature limits the unconstrained step
        to the declared trust radius; no term reads another RHO control node.
        """
        if not self.local_pulse_width_cost:
            raise RuntimeError("Local pulse-width terms require local_pulse_width_cost.")
        states = np.asarray(initial_states, dtype=float)
        if states.shape != (self.muscle_count, 3) or not np.all(np.isfinite(states)):
            raise ValueError("initial_states must be finite with shape (muscles, 3).")
        force_values = _forces(forces, (self.muscle_count, self.interval_count))
        model = _margin_model(margin_model, self.muscle_count, self.constraint_count)
        pw_model = _pulse_width_force_model(
            pulse_width_force_model, self.muscle_count, self.interval_count
        )
        if not np.array_equal(force_values, pw_model.reference_forces):
            raise ValueError("forces must equal the PW tangent reference forces.")
        _, force_gradient = self.sensitivity_function(
            states, force_values.ravel(), _pack_margin(model), np.array([1.0])
        )
        force_gradient = np.asarray(force_gradient, dtype=float).reshape(
            self.muscle_count, self.interval_count
        )
        gradient = np.einsum("mkij,mk->ij", pw_model.force_jacobian, force_gradient)
        if not np.all(np.isfinite(gradient)):
            raise ValueError("Local pulse-width reserve gradient must be finite.")
        return {
            "gradient": gradient,
            "reference": np.array(pw_model.reference_pulse_widths, dtype=float, copy=True),
            "curvature": np.abs(gradient) / self.local_pulse_width_trust_s,
            "force_gradient": force_gradient,
        }

    def sensitivity_audit(self, initial_states) -> dict:
        """Return scale-aware diagnostics without changing the compiled NLP.

        This does not infer a PW gradient.  It records whether the active
        Mayer proxy is locally sensitive to the terminal Ding slow states and
        whether a future candidate-force coupling would have a nonzero
        direction to exploit.
        """
        states = np.asarray(initial_states, dtype=float)
        if states.shape != (self.muscle_count, 3) or not np.all(np.isfinite(states)):
            raise ValueError("initial_states must be finite with shape (muscles, 3).")
        vectors = self._vectors()
        if self.candidate_force_coupling:
            state_gradient, pulse_width_gradient = self.sensitivity_function(
                states, self.pulse_width_force_model.reference_pulse_widths,
                vectors[PW_REFERENCE_PARAMETER_KEY], vectors[FORCE_PARAMETER_KEY],
                vectors[PW_JACOBIAN_PARAMETER_KEY], vectors[MARGIN_PARAMETER_KEY],
                vectors[ACTIVATION_PARAMETER_KEY],
            )
            force_gradient = None
        else:
            state_gradient, force_gradient = self.sensitivity_function(
                states, vectors[FORCE_PARAMETER_KEY], vectors[MARGIN_PARAMETER_KEY],
                vectors[ACTIVATION_PARAMETER_KEY],
            )
            pulse_width_gradient = None
        state_gradient = np.asarray(state_gradient, dtype=float).reshape(
            self.muscle_count, 3, order="F"
        )
        state_scales = np.asarray([parameter.rest_state for parameter in self.parameters], dtype=float)
        normalized_state_gradient = state_gradient * state_scales
        audit = {
            "activation": float(self.activation),
            "terminal_slow_state_gradient_l2": float(np.linalg.norm(state_gradient)),
            "terminal_slow_state_gradient_normalized_l2": float(np.linalg.norm(normalized_state_gradient)),
            "candidate_force_coupled_to_pw": self.candidate_force_coupling,
        }
        if self.candidate_force_coupling:
            pulse_width_gradient = np.asarray(pulse_width_gradient, dtype=float).reshape(
                self.muscle_count, self.interval_count
            )
            audit.update({
                "candidate_pulse_width_gradient_l2": float(np.linalg.norm(pulse_width_gradient)),
                "interpretation": "candidate PW gradient includes the fixed local Ding PW-to-force tangent",
            })
        else:
            force_gradient = np.asarray(force_gradient, dtype=float).reshape(
                self.muscle_count, self.interval_count
            )
            audit.update({
                "frozen_force_parameter_gradient_l2": float(np.linalg.norm(force_gradient)),
                "interpretation": "terminal-state gradient is connected through Ding dynamics; force gradient remains a frozen-parameter diagnostic",
            })
        return audit

    def attach(self, nmpc) -> dict:
        """Check that this is the same single NLP and parameter layout."""
        try:
            nlp, = nmpc.nlp
            parameter_bounds = nmpc.parameter_bounds
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("Reserve binding requires an OCP with exactly one NLP.") from exc
        if self._nlp is not None and nlp is not self._nlp:
            raise RuntimeError("Reserve binding cannot attach to a rebuilt NLP.")
        for key, values in self._vectors().items():
            try:
                bound = parameter_bounds[key]
            except (KeyError, TypeError) as exc:
                raise ValueError(f"Missing fixed reserve parameter {key}.") from exc
            if bound.min.shape != (values.size, 1) or bound.max.shape != (values.size, 1):
                raise ValueError(f"Reserve parameter dimensions changed for {key}.")
            if not np.array_equal(bound.min[:, 0], values) or not np.array_equal(bound.max[:, 0], values):
                raise ValueError(f"Reserve parameter {key} must have the expected fixed equal bounds.")
        self._nlp = nlp
        return {"nlp_reused": True, "nlp_identity": id(nlp),
                "objective_graph_build_count": self.build_count,
                "parameter_keys": list(self._vectors())}

    def observe_solver(self, nmpc) -> dict:
        """Record a compiled solver identity after a solve, when available."""
        self.attach(nmpc)
        interface = getattr(nmpc, "ocp_solver", None)
        solver = getattr(interface, "shaked_ocp_solver", None)
        if solver is None:
            solver = getattr(interface, "acados_solver", None)
        if solver is None:
            return {"solver_observed": False, "compiled_solver_reuse_verified": False}
        if self._solver is not None and solver is not self._solver:
            raise RuntimeError("Compiled NLP solver changed between reserve windows.")
        self._solver = solver
        self.solver_observation_count += 1
        self._solver_observed_at_updates.add(self.update_count)
        return {"solver_observed": True,
                "compiled_solver_reuse_verified": len(self._solver_observed_at_updates) > 1,
                "solver_identity": id(solver)}

    def update(self, nmpc, *, forces, margin_model, activation: float = 1.0,
               pulse_width_force_model=None, local_pulse_width_terms=None) -> dict:
        """Refresh only numerical equality bounds and initial guesses."""
        new_forces = _forces(forces, (self.muscle_count, self.interval_count))
        new_model = _margin_model(margin_model, self.muscle_count, self.constraint_count)
        new_pw_model = None
        if self.candidate_force_coupling:
            new_pw_model = _pulse_width_force_model(
                pulse_width_force_model, self.muscle_count, self.interval_count
            )
            if not np.array_equal(new_forces, new_pw_model.reference_forces):
                raise ValueError("forces must equal the PW force model reference forces.")
        elif pulse_width_force_model is not None and not self.local_pulse_width_cost:
            raise ValueError("pulse_width_force_model requires a PW reserve coupling at construction.")
        if self.local_pulse_width_cost:
            if local_pulse_width_terms is None:
                raise ValueError("Local PW reserve cost requires boundary-updated local PW terms.")
            try:
                gradient = np.asarray(local_pulse_width_terms["gradient"], dtype=float)
                reference = np.asarray(local_pulse_width_terms["reference"], dtype=float)
                curvature = np.asarray(local_pulse_width_terms["curvature"], dtype=float)
            except (KeyError, TypeError) as error:
                raise ValueError("Local PW reserve terms require gradient, reference and curvature.") from error
            shape = (self.muscle_count, self.interval_count)
            if any(value.shape != shape or not np.all(np.isfinite(value)) for value in
                   (gradient, reference, curvature)) or np.any(reference < 0) or np.any(curvature < 0):
                raise ValueError("Local PW reserve terms must be finite with nonnegative reference and curvature.")
        elif local_pulse_width_terms is not None:
            raise ValueError("local_pulse_width_terms requires local_pulse_width_cost at construction.")
        if activation != 1.0:
            raise ValueError("A certified reserve profile must activate the objective exactly once.")
        new_vectors = {FORCE_PARAMETER_KEY: new_forces.ravel(),
                       MARGIN_PARAMETER_KEY: _pack_margin(new_model),
                       ACTIVATION_PARAMETER_KEY: np.array([1.0])}
        if new_pw_model is not None:
            new_vectors[PW_REFERENCE_PARAMETER_KEY] = new_pw_model.reference_pulse_widths.ravel()
            new_vectors[PW_JACOBIAN_PARAMETER_KEY] = _pack_causal_pw_jacobian(
                new_pw_model, lag=self.pulse_width_force_lag_intervals
            )
        if self.local_pulse_width_cost:
            new_vectors.update({
                LOCAL_PW_GRADIENT_PARAMETER_KEY: gradient.ravel(),
                LOCAL_PW_REFERENCE_PARAMETER_KEY: reference.ravel(),
                LOCAL_PW_CURVATURE_PARAMETER_KEY: curvature.ravel(),
            })
        self.attach(nmpc)
        if self._solver is not None:
            interface = getattr(nmpc, "ocp_solver", None)
            solver = getattr(interface, "shaked_ocp_solver", None)
            if solver is None:
                solver = getattr(interface, "acados_solver", None)
            if solver is not self._solver:
                raise RuntimeError("Compiled NLP solver changed between reserve windows.")
        previous = self._vectors()
        old_bounds = {}
        for key, values in new_vectors.items():
            bound = nmpc.parameter_bounds[key]
            if not np.all(np.isfinite(bound.min)) or not np.all(np.isfinite(bound.max)):
                raise ValueError(f"Existing reserve bounds for {key} must be finite.")
            old_bounds[key] = (bound.min.copy(), bound.max.copy())
        from bioptim import InitialGuessList

        initial = InitialGuessList()
        for key, values in new_vectors.items():
            initial.add(key, initial_guess=values[:, None].copy())
        try:
            for key, values in new_vectors.items():
                bound = nmpc.parameter_bounds[key]
                bound.min[...] = values[:, None]
                bound.max[...] = values[:, None]
            nmpc.update_initial_guess(parameter_init=initial)
        except Exception:
            for key, (minimum, maximum) in old_bounds.items():
                bound = nmpc.parameter_bounds[key]
                bound.min[...] = minimum
                bound.max[...] = maximum
            raise
        self.forces = new_forces
        self.margin_model = new_model
        if new_pw_model is not None:
            self.pulse_width_force_model = new_pw_model
        if self.local_pulse_width_cost:
            self.local_pulse_width_gradient = gradient.copy()
            self.local_pulse_width_reference = reference.copy()
            self.local_pulse_width_curvature = curvature.copy()
        self.activation = 1.0
        self.update_count += 1
        return {
            "nlp_reused": True,
            "nlp_identity": id(self._nlp),
            "objective_graph_rebuild_required": False,
            "objective_graph_build_count": self.build_count,
            "parameter_update_count": self.update_count,
            "changed_parameters": {key: {"previous_sha256": _digest(previous[key]),
                                         "current_sha256": _digest(values)}
                                   for key, values in new_vectors.items()},
            "compiled_solver_reuse_verified": len(self._solver_observed_at_updates) > 1,
        }

    def summary(self) -> dict:
        return {
            "weight": self.weight,
            "parameter_keys": list(self._vectors()),
            "activation": self.activation,
            "force_parameter_size": self.muscle_count * self.interval_count,
            "margin_parameter_size": self.margin_parameter_size,
            "horizons": list(self.horizons),
            "durations": list(self.durations),
            "muscle_count": self.muscle_count,
            "constraint_count": self.constraint_count,
            "objective_graph_build_count": self.build_count,
            "objective_graph_rebuild_required": False,
            "parameter_update_count": self.update_count,
            "nlp_attached": self._nlp is not None,
            "compiled_solver_observation_count": self.solver_observation_count,
            "compiled_solver_reuse_verified": len(self._solver_observed_at_updates) > 1,
            "candidate_force_coupling": self.candidate_force_coupling,
            "local_pulse_width_cost": self.local_pulse_width_cost,
            "local_pulse_width_trust_s": self.local_pulse_width_trust_s,
            "pulse_width_force_lag_intervals": self.pulse_width_force_lag_intervals,
        }
