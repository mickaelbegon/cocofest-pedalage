"""Experimental fixed-graph Bioptim binding for an audited local terminal value.

The numerical rollout/QP never appears in this graph. Only its local polynomial
and validity guards are differentiated. This does not enable a production RHO
mode: the caller must additionally validate the source/target task, kinematics,
phase and period, and recompute the local fit between windows as needed.
"""

from __future__ import annotations

from hashlib import sha256

import numpy as np


LOCAL_ENDURANCE_PARAMETER_KEY = "local_endurance_value"


def terminal_local_endurance_cost(controller, binding):
    return binding.controller_outputs(controller)[0]


def terminal_local_endurance_trust_margins(controller, binding):
    return binding.controller_outputs(controller)[1]


def terminal_local_endurance_context_residuals(controller, binding):
    return binding.controller_outputs(controller)[2]


class LocalEnduranceValueBinding:
    """One polynomial, a trust box and a nonzero-offset/calcium context guard.

    Parameter order is constant, weight, center, gradient, diagonal Hessian,
    lower, upper, force scales, numerical terminal Cn, and two offsets/muscle.
    All are fixed-bound parameters, not future free controls. Bioptim still
    includes these parameters in its generic NLP vector: 2 + 14*M entries.
    Muscle fatigue constants, names, and the normalized context tolerance are
    structural. Changing any of them requires a new binding/compiled graph.
    """

    def __init__(self, fit, coordinates, muscle_names, *, weight=1.0,
                 use_sx=True, context_tolerance=1e-7, allow_tracking_band=False):
        self.muscle_names = tuple(muscle_names)
        self.muscle_count = len(self.muscle_names)
        if not self.muscle_count or len(set(self.muscle_names)) != self.muscle_count:
            raise ValueError("Unique nonempty muscle names are required.")
        self.parameters = tuple(coordinates.parameters)
        if len(self.parameters) != self.muscle_count:
            raise ValueError("Coordinate parameters and muscle names must agree.")
        self.context_tolerance = float(context_tolerance)
        if not np.isfinite(self.context_tolerance) or self.context_tolerance <= 0:
            raise ValueError("context_tolerance must be finite and positive.")
        self.dimension = 2 * self.muscle_count
        self.parameter_size = 2 + 14 * self.muscle_count
        self._allow_tracking_band = bool(allow_tracking_band)
        self._tracking_contract = self._tracking_settings(fit)
        if self._tracking_contract[0] > 0 and not self._allow_tracking_band:
            raise ValueError("A relaxed future task requires explicit allow_tracking_band=True.")
        self._value_context = fit.metadata.get("value_context")
        self.values = self._pack(fit, coordinates, weight)
        self.function = self._build_function(use_sx=use_sx)
        self._ocp = None
        self._nlp = None
        self._compiled_solver = None
        self._last_solution = None
        self.successful_solve_count = 0
        self.update_count = 0

    @staticmethod
    def _tracking_settings(fit):
        context = fit.metadata.get("value_context") or {}
        settings = tuple(float(context.get(key, 0.)) for key in ("tracking_band_nm", "tracking_penalty_weight"))
        if not np.all(np.isfinite(settings)) or min(settings) < 0:
            raise ValueError("Tracking band and penalty settings must be finite and nonnegative.")
        # With a zero band the batch oracle disables the tracking penalty, so
        # switching scalar/batch exact backends must not change this contract.
        return settings[0], settings[1] if settings[0] > 0 else 0.

    def _pack(self, fit, coordinates, weight):
        if not fit.accepted or fit.model is None:
            raise ValueError("Only an accepted, independently audited local fit can be bound.")
        if fit.metadata.get("coordinate_context_sha256") != coordinates.context_signature:
            raise ValueError("The fitted value and supplied coordinate context do not match.")
        if self._tracking_settings(fit) != self._tracking_contract:
            raise ValueError("A local-value update cannot silently change the future tracking task.")
        if tuple(coordinates.parameters) != self.parameters:
            raise ValueError("An update cannot change muscle parameters or order.")
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("weight must be finite and nonnegative.")
        model = fit.model
        vectors = [np.asarray(getattr(model, name), dtype=float) for name in
                   ("center", "gradient", "diagonal_hessian", "lower_bounds", "upper_bounds")]
        if any(x.shape != (self.dimension,) or not np.all(np.isfinite(x)) for x in vectors):
            raise ValueError("Local coefficient/bound dimensions must match the muscle count.")
        center, _, _, lower, upper = vectors
        if np.any(lower >= upper) or np.any(center < lower) or np.any(center > upper):
            raise ValueError("The local model must have a nonempty trust box containing its center.")
        # Domain checks are separable in the d/F coordinates. Do not clip an
        # invalid trust box, which would silently alter the audited model.
        coordinates.decode(center)
        for i in range(self.dimension):
            for bound in (lower, upper):
                point = center.copy()
                point[i] = bound[i]
                coordinates.decode(point)
        scale = np.asarray(coordinates.force_scale, dtype=float)
        cn = np.asarray(coordinates.fixed_cn, dtype=float)
        offsets = np.asarray(coordinates.offsets, dtype=float)
        if scale.shape != (self.muscle_count,) or np.any(scale <= 0):
            raise ValueError("force_scale must be strictly positive for every muscle.")
        if cn.shape != (self.muscle_count,) or offsets.shape != (self.muscle_count, 2):
            raise ValueError("The fixed calcium and offset dimensions must match the muscles.")
        values = np.r_[float(model.constant), float(weight), *vectors, scale, cn, offsets.ravel()]
        if values.shape != (self.parameter_size,) or not np.all(np.isfinite(values)):
            raise ValueError("Packed local-value data must be finite with the fixed graph size.")
        return values

    def _build_function(self, *, use_sx):
        from casadi import Function, MX, SX, dot, vertcat

        sym = SX if use_sx else MX
        states = sym.sym("muscle_states", 5 * self.muscle_count)
        p = sym.sym("local_value_data", self.parameter_size)
        n, m = self.dimension, self.muscle_count
        center, gradient, diagonal, lower, upper = [p[2 + i*n:2 + (i+1)*n] for i in range(5)]
        scale = p[2 + 5*n:2 + 5*n + m]
        cn = p[2 + 5*n + m:2 + 5*n + 2*m]
        offsets = p[2 + 5*n + 2*m:]
        damage, forces, residuals = [], [], []
        for i, params in enumerate(self.parameters):
            fatigue = params.fatigue
            if fatigue.alpha_a >= 0:
                raise ValueError("The damage reduction requires alpha_a < 0.")
            calcium, force, amplitude, tau1, km = (states[5*i+j] for j in range(5))
            damage.append(1 - amplitude / fatigue.a_rest)
            forces.append(force / scale[i])
            delta_a = amplitude - fatigue.a_rest
            residuals.extend([
                calcium - cn[i],
                (tau1 - fatigue.tau1_rest - fatigue.alpha_tau1/fatigue.alpha_a * delta_a
                 - offsets[2*i]) / fatigue.tau1_rest,
                (km - fatigue.km_rest - fatigue.alpha_km/fatigue.alpha_a * delta_a
                 - offsets[2*i+1]) / fatigue.km_rest,
            ])
        xi = vertcat(*damage, *forces)
        delta = xi - center
        value = p[1] * (p[0] + dot(gradient, delta) + 0.5 * dot(diagonal, delta**2))
        return Function("local_endurance_terminal_value", [states, p],
                        [value, vertcat(xi-lower, upper-xi), vertcat(*residuals), xi])

    def controller_outputs(self, controller):
        from casadi import vertcat

        states = vertcat(*[controller.states[f"{key}_{name}"].cx
                          for name in self.muscle_names for key in ("Cn", "F", "A", "Tau1", "Km")])
        return self.function(states, controller.parameters[LOCAL_ENDURANCE_PARAMETER_KEY].cx)

    def add_penalties(self, objectives, constraints):
        """Call during OCP construction, before the one-time compilation."""
        from bioptim import Node, ObjectiveFcn

        objectives.add(terminal_local_endurance_cost, custom_type=ObjectiveFcn.Mayer,
                       node=Node.END, weight=1.0, quadratic=False, binding=self)
        constraints.add(terminal_local_endurance_trust_margins, node=Node.END,
                        min_bound=0.0, max_bound=np.inf, binding=self)
        constraints.add(terminal_local_endurance_context_residuals, node=Node.END,
                        min_bound=-self.context_tolerance, max_bound=self.context_tolerance, binding=self)

    def parameter_options(self, *, use_sx=True):
        from bioptim import BoundsList, InitialGuessList, InterpolationType, ParameterList, VariableScaling

        key = LOCAL_ENDURANCE_PARAMETER_KEY
        parameters = ParameterList(use_sx=use_sx)
        parameters.add(name=key, function=None, size=self.parameter_size,
                       scaling=VariableScaling(key, np.ones(self.parameter_size)))
        bounds, initial = BoundsList(), InitialGuessList()
        values = self.values[:, None].copy()
        bounds.add(key, min_bound=values.copy(), max_bound=values.copy(),
                   interpolation=InterpolationType.CONSTANT)
        initial.add(key, initial_guess=values.copy())
        return {"parameters": parameters, "parameter_bounds": bounds, "parameter_init": initial}

    def attach(self, ocp):
        if self._ocp is not None and (self._ocp is not ocp or self._nlp is not ocp.nlp[0]):
            raise RuntimeError("This binding cannot be moved to a rebuilt NLP.")
        self._ocp, self._nlp = ocp, ocp.nlp[0]

    def update(self, ocp, fit, coordinates, *, weight=None):
        """Validate first; update numeric buffers only, never graph expressions.

        This must run between solves. A rejected fit leaves the previous
        parameter values intact; the caller must not silently keep using that
        old model outside its original validity box.
        """
        values = self._pack(fit, coordinates, self.values[1] if weight is None else weight)
        self.attach(ocp)
        compiled = getattr(getattr(ocp, "ocp_solver", None), "shaked_ocp_solver", None)
        if self._compiled_solver is not None and compiled is not self._compiled_solver:
            raise RuntimeError("The compiled NLP solver has changed.")
        bounds = ocp.parameter_bounds[LOCAL_ENDURANCE_PARAMETER_KEY]
        if bounds.min.shape != (self.parameter_size, 1) or bounds.max.shape != bounds.min.shape:
            raise ValueError("The fixed-parameter buffer dimensions changed.")
        from bioptim import InitialGuessList

        initial = InitialGuessList()
        initial.add(LOCAL_ENDURANCE_PARAMETER_KEY, initial_guess=values[:, None].copy())
        ocp.update_initial_guess(parameter_init=initial)
        bounds.min[...] = values[:, None]
        bounds.max[...] = values[:, None]
        self.values = values
        self._value_context = fit.metadata.get("value_context")
        self.update_count += 1

    def audit_terminal_state(self, states, *, tolerance=1e-8):
        states = np.asarray(states, dtype=float)
        if states.shape != (self.muscle_count, 5) or not np.all(np.isfinite(states)):
            raise ValueError("Terminal states must be finite in (muscles, 5) order.")
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Audit tolerance must be finite and nonnegative.")
        value, margins, context, _ = self.function(states.ravel(), self.values)
        minimum = float(np.min(margins))
        residual = float(np.max(np.abs(context)))
        valid = (np.all(states[:, :2] >= 0) and np.all(states[:, 2:] > 0)
                 and minimum >= -tolerance and residual <= self.context_tolerance + tolerance)
        return {"valid": bool(valid), "value": float(value), "minimum_trust_margin": minimum,
                "maximum_context_residual": residual}

    def record_solution(self, ocp, solution):
        """Record distinct successful solves, including numeric/context checks."""
        if solution is self._last_solution:
            raise ValueError("The same solution cannot be counted twice as compiled reuse.")
        if solution.status != 0:
            raise ValueError("Only successful solves can establish compiled reuse.")
        self.attach(ocp)
        actual = np.asarray(solution.parameters[LOCAL_ENDURANCE_PARAMETER_KEY]).ravel()
        if actual.shape != self.values.shape or not np.allclose(actual, self.values, rtol=0, atol=1e-9):
            raise ValueError("The solution does not contain the current local-value coefficients.")
        from bioptim import SolutionMerge

        states = solution.decision_states(to_merge=SolutionMerge.NODES)
        terminal = np.asarray([[states[f"{key}_{name}"][0, -1]
                                for key in ("Cn", "F", "A", "Tau1", "Km")]
                               for name in self.muscle_names])
        audit = self.audit_terminal_state(terminal)
        if not audit["valid"]:
            raise ValueError("The solution extrapolates beyond the audited local model context.")
        compiled = getattr(getattr(ocp, "ocp_solver", None), "shaked_ocp_solver", None)
        if compiled is None or not getattr(ocp.ocp_solver, "c_compile", False):
            raise ValueError("No compiled NLP solver is available to verify reuse.")
        if self._compiled_solver is not None and compiled is not self._compiled_solver:
            raise RuntimeError("The compiled solver was rebuilt between solutions.")
        self._compiled_solver = compiled
        self._last_solution = solution
        self.successful_solve_count += 1
        return audit

    def summary(self):
        return {"status": "experimental_binding_not_prospective_endurance_validation",
                "muscle_count": self.muscle_count, "local_coordinates": self.dimension,
                "fixed_nlp_parameters": self.parameter_size, "future_free_variables": 0,
                "trust_inequalities": 2*self.dimension, "context_inequalities": 3*self.muscle_count,
                "objective_graph_build_count": 1, "numeric_update_count": self.update_count,
                "successful_compiled_solves": self.successful_solve_count,
                "same_compiled_solver_verified": self.successful_solve_count >= 2,
                "parameters_sha256": sha256(self.values.tobytes()).hexdigest(),
                "future_value_context": self._value_context,
                "tracking_band_explicitly_allowed": self._allow_tracking_band,
                "requires_external_task_and_kinematic_context_validation": True}
