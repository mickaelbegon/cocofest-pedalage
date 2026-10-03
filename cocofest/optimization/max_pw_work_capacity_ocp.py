"""Fixed-graph terminal Mayer binding for max-PW positive-work capacity.

This experimental binding does not alter the RHO dynamics or add a future
OCP. The separately validated one-cycle proxy and its gradient are evaluated
only at certified cycle boundaries. The next RHO receives their local
first-order model as equality-bound numerical parameters, so its graph stays
small. Changing geometry, cadence or the fixed stimulation mask still
requires a new binding and a rebuild.
"""
from __future__ import annotations

import math
from hashlib import sha256
import numpy as np

PROFILE_KEY = "rho_max_pw_work_profile"
ACTIVATION_KEY = "rho_max_pw_work_activation"
LINEAR_REFERENCE_KEY = "rho_max_pw_work_linear_reference"
LINEAR_GRADIENT_KEY = "rho_max_pw_work_linear_gradient"
GRADIENT_FILTERS = ("full", "slow_fatigue_states")
STATE_NAMES = ("Cn", "F", "A", "Tau1", "Km")


class MaxPwWorkCapacityBinding:
    def __init__(self, *, function, profile, reference_work_j: float, weight: float, layout=None,
                 muscle_names=None, gradient_filter="full"):
        self.function = function
        self.profile = np.asarray(profile, float).reshape(-1)
        self.reference_work_j, self.weight = float(reference_work_j), float(weight)
        if (not self.profile.size or not np.all(np.isfinite(self.profile))
                or not math.isfinite(self.reference_work_j) or self.reference_work_j <= 0
                or not math.isfinite(self.weight) or self.weight <= 0):
            raise ValueError("Capacity binding needs finite profile, positive reference work and weight")
        if gradient_filter not in GRADIENT_FILTERS:
            raise ValueError(f"gradient_filter must be one of {GRADIENT_FILTERS}")
        self.gradient_filter = gradient_filter
        self.activation = 0.0
        self.reference_state = np.zeros(int(function.numel_in(0)), dtype=float)
        self.gradient = np.zeros_like(self.reference_state)
        self.reference_capacity_j = None
        if self.reference_state.size % len(STATE_NAMES):
            raise ValueError("Capacity terminal state must contain five Ding states per muscle")
        self.gradient_mask = np.tile(
            [1., 1., 1., 1., 1.] if gradient_filter == "full" else [0., 0., 1., 1., 1.],
            self.reference_state.size // len(STATE_NAMES),
        )
        self._nlp = None
        self.update_count = 0
        self.layout = layout
        self.stimulation_policy = None
        self.muscle_names = tuple(muscle_names or ())
        if layout is not None and (len(self.muscle_names) != layout.muscle_count
                                   or len(set(self.muscle_names)) != layout.muscle_count):
            raise ValueError("Capacity binding requires the ordered muscle names")
        self.last_audit = None
        if int(function.numel_in(1)) != self.profile.size:
            raise ValueError("Capacity profile has incompatible dimensions")
        self.profile = self._validated_profile(self.profile)
        self.profile.setflags(write=False)
        self.profile_sha256 = sha256(self.profile.tobytes()).hexdigest()
        # This expensive derivative is evaluated only at certified boundaries.
        # The RHO graph below contains the resulting 20-vector, not this rollout.
        import casadi as ca
        state = ca.SX.sym("max_pw_work_boundary_state", self.reference_state.size)
        self._gradient_function = ca.Function("max_pw_work_boundary_gradient", [state],
            [ca.gradient(function(state, self.profile)[0], state)])

    def validate_muscle_names(self, names):
        if tuple(names) != self.muscle_names:
            raise ValueError("Capacity proxy muscle order differs from the RHO model")

    def audit(self, terminal_state, *, profile=None):
        """Numerical RK-stage audit, deliberately outside the NLP constraints."""
        from .max_pw_work_capacity import max_pw_work_domain_lower_bounds
        if self.layout is None:
            raise ValueError("A layout is required to audit capacity domain margins")
        state = np.asarray(terminal_state, float).reshape(-1)
        values = self.profile if profile is None else self._validated_profile(profile)
        if state.size != self.layout.initial_state_size or not np.all(np.isfinite(state)):
            raise ValueError("Capacity terminal state has incompatible dimensions or nonfinite values")
        result = self.function(state, values)
        margins = np.asarray(result[3], float).reshape(-1)
        lower = max_pw_work_domain_lower_bounds(self.layout)
        finite = all(np.all(np.isfinite(np.asarray(output, float))) for output in result)
        valid = bool(finite and margins.shape == lower.shape and np.all(margins >= lower))
        capacity = float(result[0]) if finite else None
        gradient = None
        if valid:
            gradient = np.asarray(self._gradient_function(state), float).reshape(-1)
            valid = bool(gradient.shape == state.shape and np.all(np.isfinite(gradient)))
        names = self.muscle_names or tuple(f"muscle_{i}" for i in range(state.size // 5))
        gradient_by_muscle = None
        gradient_l2 = None
        filtered_gradient_l2 = None
        if valid:
            gradient_by_muscle = {}
            for i, name in enumerate(names):
                part = gradient[5*i:5*(i+1)]
                gradient_by_muscle[name] = {
                    "derivative_j_per_state_unit": dict(zip(STATE_NAMES, map(float, part))),
                    "fast_cn_force_l2_raw": float(np.linalg.norm(part[:2])),
                    "slow_a_tau1_km_l2_raw": float(np.linalg.norm(part[2:])),
                }
            gradient_l2 = float(np.linalg.norm(gradient))
            filtered_gradient_l2 = float(np.linalg.norm(gradient * self.gradient_mask))
        taylor = None
        if valid and self.reference_capacity_j is not None and self.activation:
            delta = state - self.reference_state
            prediction = self.reference_capacity_j + float(np.dot(self.gradient, delta))
            contributions = {}
            for i, name in enumerate(names):
                direction = self.gradient[5*i:5*(i+1)] * delta[5*i:5*(i+1)]
                contributions[name] = {
                    "fast_cn_force_j": float(np.sum(direction[:2])),
                    "slow_a_tau1_km_j": float(np.sum(direction[2:])),
                }
            error = capacity - prediction
            taylor = {
                "prediction_j": prediction, "error_j": error,
                "relative_error": error / max(abs(capacity), self.reference_work_j),
                "state_step_l2_raw": float(np.linalg.norm(delta)),
                "directional_contributions_by_muscle_j": contributions,
            }
        return {"valid": valid, "capacity_j": capacity,
                "work_by_muscle_j": (dict(zip(names, map(float, np.asarray(result[1]).reshape(-1))))
                                     if finite else None),
                "gradient_l2_raw": gradient_l2,
                "filtered_gradient_l2_raw": filtered_gradient_l2,
                "gradient_by_muscle": gradient_by_muscle,
                "taylor_from_previous_checkpoint": taylor,
                "minimum_domain_slack": float(np.min(margins-lower)) if finite else None,
                "physiological_feasibility_certified": False}

    def _validated_profile(self, profile):
        values = np.asarray(profile, float).reshape(-1)
        if values.shape != self.profile.shape or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Capacity profile update is nonfinite, negative or changes dimensions")
        if self.layout is not None:
            for name, section in self.layout.slices().items():
                if (name == "duration" or name.startswith("gain_")) and np.any(values[section] <= 0):
                    raise ValueError("Capacity duration and gain must be positive")
                if name == "stimulation_mask" and not np.all(np.isin(values[section], [0., 1.])):
                    raise ValueError("Capacity stimulation mask must be binary")
        return values.copy()

    def _write(self, nmpc, profile, activation):
        self.attach(nmpc)
        from bioptim import InitialGuessList
        entries = ((ACTIVATION_KEY, [activation]), (LINEAR_REFERENCE_KEY, self.reference_state),
                   (LINEAR_GRADIENT_KEY, self.gradient))
        initial = InitialGuessList()
        backups = {}
        for key, value in entries:
            column = np.asarray(value, float).reshape(-1, 1)
            bound = nmpc.parameter_bounds[key]
            if bound.min.shape != column.shape or bound.max.shape != column.shape:
                raise ValueError("Capacity parameter bounds have an unexpected shape")
            backups[key] = (bound.min.copy(), bound.max.copy(), nmpc.parameter_init[key].init.copy())
            initial.add(key, initial_guess=column.copy())
        try:
            nmpc.update_initial_guess(parameter_init=initial)
            for key, value in entries:
                column = np.asarray(value, float).reshape(-1, 1)
                nmpc.parameter_bounds[key].min[...] = column
                nmpc.parameter_bounds[key].max[...] = column
        except Exception:
            for key, (lower, upper, guess) in backups.items():
                nmpc.parameter_bounds[key].min[...] = lower
                nmpc.parameter_bounds[key].max[...] = upper
                nmpc.parameter_init[key].init[...] = guess
            raise
        self.activation = float(activation)
        self.update_count += 1

    def update(self, nmpc, *, terminal_state, profile=None):
        values = self._validated_profile(self.profile if profile is None else profile)
        if not np.array_equal(values, self.profile):
            raise ValueError("Changing the fixed capacity profile requires a graph rebuild")
        audit = self.audit(terminal_state, profile=values)
        if not audit["valid"]:
            raise ValueError("Capacity update refused: invalid rollout domain")
        state = np.asarray(terminal_state, float).reshape(-1).copy()
        gradient = np.asarray(self._gradient_function(state), float).reshape(-1)
        if gradient.shape != state.shape or not np.all(np.isfinite(gradient)):
            raise ValueError("Capacity gradient is nonfinite or has incompatible dimensions")
        old = (self.reference_state, self.gradient, self.reference_capacity_j)
        self.reference_state = state
        self.gradient = gradient * self.gradient_mask
        self.reference_capacity_j = audit["capacity_j"]
        try:
            self._write(nmpc, values, 1.)
        except Exception:
            self.reference_state, self.gradient, self.reference_capacity_j = old
            raise
        self.last_audit = audit
        return self.summary()

    def deactivate(self, nmpc):
        self._write(nmpc, self.profile, 0.)
        return self.summary()

    def validate_terminal_point(self, terminal_state):
        self.last_audit = self.audit(terminal_state)
        return {**self.last_audit, "active": bool(self.activation),
                "hold_required": bool(self.activation and not self.last_audit["valid"])}

    def attach(self, nmpc):
        if len(nmpc.nlp) != 1:
            raise ValueError("Max-PW work capacity requires one RHO phase")
        if self._nlp is not None and self._nlp is not nmpc.nlp[0]:
            raise RuntimeError("Capacity binding cannot be attached to a rebuilt NLP")
        self._nlp = nmpc.nlp[0]

    def objective(self, terminal_state, controller):
        # First-order model of W+ about the last certified boundary.  The
        # constant W+(x_ref) is irrelevant to the minimizer and omitted.
        delta = terminal_state - controller.parameters[LINEAR_REFERENCE_KEY].cx
        value = controller.parameters[LINEAR_GRADIENT_KEY].cx.T @ delta
        return -controller.parameters[ACTIVATION_KEY].cx[0] * self.weight * value / self.reference_work_j

    def parameter_options(self, *, use_sx=True):
        return max_pw_work_parameter_options(self, use_sx=use_sx)

    def summary(self):
        return {"experimental": True, "activation": self.activation, "updates": self.update_count,
                "objective_graph_rebuild_required": False, "last_domain_audit": self.last_audit,
                "fixed_profile_sha256": self.profile_sha256, "profile_change_requires_rebuild": True,
                "muscle_names": list(self.muscle_names), "reference_work_j": self.reference_work_j,
                "weight": self.weight, "coupling": "boundary_first_order",
                "gradient_filter": self.gradient_filter,
                "integration_substeps": (self.layout.integration_substeps if self.layout is not None else None),
                "stimulation_policy": self.stimulation_policy,
                "reference_capacity_j": self.reference_capacity_j,
                "interpretation": "max-PW positive-work opportunity proxy; not endurance or feasibility"}


def max_pw_work_parameter_options(binding, *, fatigue_weight_binding=None, use_sx=True):
    from bioptim import BoundsList, InitialGuessList, InterpolationType, ParameterList, VariableScaling
    from .parametric_fatigue_weights import FATIGUE_WEIGHT_PARAMETER_KEY, ParametricFatigueWeightBinding
    if not isinstance(binding, MaxPwWorkCapacityBinding):
        raise TypeError("binding must be a MaxPwWorkCapacityBinding")
    entries = [(ACTIVATION_KEY, [binding.activation]),
               (LINEAR_REFERENCE_KEY, binding.reference_state),
               (LINEAR_GRADIENT_KEY, binding.gradient)]
    if fatigue_weight_binding is not None:
        if not isinstance(fatigue_weight_binding, ParametricFatigueWeightBinding):
            raise TypeError("Only ParametricFatigueWeightBinding can coexist with max-PW capacity")
        entries.append((FATIGUE_WEIGHT_PARAMETER_KEY, fatigue_weight_binding.weights))
    p, b, x = ParameterList(use_sx=use_sx), BoundsList(), InitialGuessList()
    for key, values in entries:
        values = np.asarray(values, float).reshape((-1, 1))
        p.add(name=key, function=None, size=values.size, scaling=VariableScaling(key, np.ones(values.size)))
        b.add(key, min_bound=values.copy(), max_bound=values.copy(), interpolation=InterpolationType.CONSTANT)
        x.add(key, initial_guess=values.copy())
    return {"parameters": p, "parameter_bounds": b, "parameter_init": x}


def build_isokinetic_max_pw_work_binding(model, *, interval_count, omega, reference_work_j,
                                        weight, integration_substeps=16,
                                        stimulation_policy="all_intervals_pw_max", pulse_width_maxima=None,
                                        gradient_filter="full"):
    """Freeze the exact prescribed geometry at RK stages for one future turn."""
    from .adaptive_moment_rollout import DingPulseWidthParameters, MomentTrackingInterval
    from .max_pw_work_capacity import (MaxPwWorkCapacityLayout,
        build_max_pw_work_capacity_function, pack_max_pw_work_profile)
    models = tuple(model.muscles_dynamics_model)
    layout = MaxPwWorkCapacityLayout(len(models), interval_count, integration_substeps)
    duration = 2 * math.pi / abs(omega)
    dt = duration / interval_count
    theta0 = float(model.isokinetic_theta0)
    dynamics = model.reduced_dynamics
    if pulse_width_maxima is None or len(pulse_width_maxima) != len(models):
        raise ValueError("Max-PW capacity needs actual model-ordered pulse-width upper bounds")
    parameters = tuple(DingPulseWidthParameters.from_model(m, pulse_width_max=float(pw))
                       for m, pw in zip(models, pulse_width_maxima))
    intervals, coefficients = [], []
    for k in range(interval_count):
        def gains(time, k=k):
            fl, fv, passive = (np.asarray(v, float) for v in
                              dynamics.muscle_relationships(theta0 + omega*(k*dt+time), omega))
            return ((fl if model.activate_force_length_relationship else np.ones_like(fl))
                    * (fv if model.activate_force_velocity_relationship else np.ones_like(fv))
                    + (passive if model.activate_passive_force_relationship else np.zeros_like(passive)))
        functions = tuple(lambda t, m=m, k=k: float(dynamics.coefficient_values(
            theta0 + omega*(k*dt+t))["muscle_effectiveness"][m]) for m in range(len(models)))
        intervals.append(MomentTrackingInterval(dt,
            tuple(float(m.post_stimulation_amplitude()) for m in models),
            tuple(lambda t, m=m, gains=gains: float(gains(t)[m]) for m in range(len(models))),
            tuple(f(dt/2) for f in functions), (0.,)*len(models)))
        coefficients.append(functions)
    profile = pack_max_pw_work_profile(intervals, layout, angular_velocity_rad_s=omega,
        moment_coefficient_functions=coefficients, stimulation_policy=stimulation_policy)
    function = build_max_pw_work_capacity_function(muscles=parameters, layout=layout)
    binding = MaxPwWorkCapacityBinding(function=function, profile=profile, layout=layout,
        muscle_names=[m.muscle_name for m in models], reference_work_j=reference_work_j,
        weight=weight, gradient_filter=gradient_filter)
    binding.stimulation_policy = stimulation_policy
    return binding


def max_pw_work_boundary(nmpc, solution, *, certified):
    """Audit a solved terminal state and activate only after a valid bootstrap.

    Call before committing/advancing the solved cycle. A false ``accepted``
    means the supervisor must hold at the last accepted physical boundary.
    """
    from bioptim import SolutionMerge
    binding = getattr(nmpc, "max_pw_work_binding", None)
    if binding is None or solution is None:
        return {"enabled": binding is not None, "accepted": True, "bootstrap": solution is None}
    if not certified:
        return {"enabled": True, "accepted": False, "reason": "uncertified_rho"}
    states = solution.decision_states(to_merge=SolutionMerge.NODES)
    state = [float(np.asarray(states[f"{key}_{name}"]).reshape(-1)[-1])
             for name in binding.muscle_names for key in ("Cn", "F", "A", "Tau1", "Km")]
    audit = binding.validate_terminal_point(state)
    if not audit["valid"]:
        return {"enabled": True, "accepted": False, "reason": "invalid_capacity_domain", **audit}
    # Equality bounds are refreshed each time: cyclic warm-start transfer may
    # otherwise replace the parameter guess with the previous solution.
    binding.update(nmpc, terminal_state=state)
    return {"enabled": True, "accepted": True, "activation_next": binding.activation, **audit}
