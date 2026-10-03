"""Causal, cycle-work-constrained value-of-reserve rollout for PACE-VR.

All optimization is outside the RHO graph.  Each projected cycle changes its
PW schedule through a small recruitment-space QP; it does not repeat forces
or PW.  Calcium is analytic and force uses the exponential midpoint map of
``CompactMusclePredictor``.  Fatigue receives the corresponding force
convolution.  The slow states are frozen only in the force coefficients over
one stimulation interval.  Geometry is held at the supplied interval sample.

Consequently neither a successful rollout nor a rejected QP certifies physical
endurance.  The full RHO/independent feasibility problem remains the authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import contextmanager
from time import monotonic, perf_counter
import hashlib
import json
import os
import ctypes

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize
from scipy.special import exprel

from .compact_muscle_prediction import CompactMusclePredictor
from .adaptive_moment_rollout import DingPulseWidthParameters, MomentTrackingInterval
from .ding_fatigue_rollout import DingFatigueParameters


def _casadi_qpoases_available():
    """Whether the bundled CasADi qpOASES interface can be constructed.

    This deliberately checks the interface actually used by PACE-VR.  An
    ``acados_template`` import alone is insufficient: PACE-VR is a dense,
    condensed QP, rather than an acados OCP.  The local CasADi/HPIPM bridge
    segfaults after repeated calls, so it is intentionally not a backend.
    """
    try:
        import casadi as ca
        with _silence_native_solver_output():
            return ca if ca.has_conic("qpoases") else None
    except (ImportError, RuntimeError):
        return None


@contextmanager
def _silence_native_solver_output():
    """Keep native QP diagnostics out of asynchronous RHO logs.

    PACE-VR runs in an isolated process. Redirecting its file descriptors for
    one QP call is therefore safe and, unlike ``redirect_stdout``, also hides
    messages printed directly by qpOASES.
    """
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        yield
    finally:
        # Native solvers may retain their banner in the C stdio buffer.
        ctypes.CDLL(None).fflush(None)
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(null_fd)


class _PersistentQpoasesQp:
    """Small data-updated qpOASES QP cache, with one solver per row layout.

    Its symbolic sparsity and native active-set solver are created once; H, g
    and all bounds are updated numerically at each SQP step. The final variable
    is a normalized work slack, needed because the first trust-region SQP step
    can make the exact linearized work equality infeasible.
    """

    def __init__(self):
        self.ca = _casadi_qpoases_available()
        self._solvers = {}
        self._warm_starts = {}

    @property
    def available(self):
        return self.ca is not None

    def solve(self, hessian, gradient, matrix, lower, upper, work_gradient,
              work_delta, trust, required_work_j):
        if not self.available:
            raise RuntimeError("CasADi qpOASES is unavailable.")
        n_recruitment = gradient.size
        def call_qp(label, h, g, a, lo, hi, lbx, ubx):
            n = g.size
            key = (label, n, a.shape[0])
            solver = self._solvers.get(key)
            if solver is None:
                ca = self.ca
                with _silence_native_solver_output():
                    solver = ca.conic(
                        f"pace_vr_qpoases_{label}_{n}_{a.shape[0]}", "qpoases",
                        {"h": ca.Sparsity.dense(n, n), "a": ca.Sparsity.dense(a.shape[0], n)},
                        {"error_on_fail": False, "print_time": False, "printLevel": "none",
                         "terminationTolerance": 1e-12, "boundTolerance": 1e-12,
                         "numRefinementSteps": 3, "enableDriftCorrection": 1, "nWSR": 250},
                    )
                self._solvers[key] = solver
                self._warm_starts[key] = np.zeros(n)
            ca = self.ca
            try:
                with _silence_native_solver_output():
                    output = solver.call(dict(
                        h=ca.DM(0.5 * (h + h.T)), g=ca.DM(g), a=ca.DM(a),
                        lba=ca.DM(lo), uba=ca.DM(hi), lbx=ca.DM(lbx), ubx=ca.DM(ubx),
                        x0=ca.DM(self._warm_starts[key]),
                    ))
                stats = solver.stats()
                x = np.asarray(output["x"], dtype=float).reshape(-1)
                success = bool(stats.get("success", False)) and np.all(np.isfinite(x))
                if success:
                    self._warm_starts[key] = x
                return success, x, str(stats.get("unified_return_status", stats.get("return_status", "unknown"))), stats
            except (RuntimeError, ValueError, OSError) as error:
                return False, np.zeros(n), f"qpOASES error: {error}", {}

        strict_matrix = np.vstack((matrix, work_gradient))
        strict_lower = np.r_[lower, work_delta]
        strict_upper = np.r_[upper, work_delta]
        success, x, message, stats = call_qp(
            "strict", hessian, gradient, strict_matrix, strict_lower, strict_upper,
            np.full(n_recruitment, -trust), np.full(n_recruitment, trust))
        if success:
            return dict(success=True, x=x, message=message, backend="casadi_qpoases",
                        iterations=stats.get("iter_count"), work_slack_normalized=0.)

        # A strict linear work equality may be outside the *first* trust
        # region.  Use a strongly penalized normalized slack only to obtain a
        # feasible SQP progress step; later feasible iterations stay strict.
        n = n_recruitment + 1
        soft_hessian = np.zeros((n, n))
        soft_hessian[:-1, :-1] = hessian
        soft_hessian[-1, -1] = 1e4
        soft_matrix = np.vstack((np.column_stack((matrix, np.zeros(matrix.shape[0]))),
                                 np.r_[work_gradient / required_work_j, 1.]))
        success, x, soft_message, stats = call_qp(
            "soft", soft_hessian, np.r_[gradient, 0.], soft_matrix,
            np.r_[lower, work_delta / required_work_j], np.r_[upper, work_delta / required_work_j],
            np.r_[np.full(n_recruitment, -trust), -10.], np.r_[np.full(n_recruitment, trust), 10.])
        if not success:
            return dict(success=False, x=np.zeros(n_recruitment),
                        message=f"strict={message}; soft={soft_message}",
                        backend="casadi_qpoases", iterations=stats.get("iter_count"))
        return dict(success=True, x=x[:-1], message=soft_message, backend="casadi_qpoases",
                    iterations=stats.get("iter_count"), work_slack_normalized=float(x[-1]))


@dataclass(frozen=True)
class PaceVrConfig:
    horizon_cycles: int = 100
    phase_knots: int = 4
    integration_substeps: int = 8
    maximum_sqp_iterations: int = 6
    recruitment_trust: float = .25
    work_relative_tolerance: float = 1e-5
    constraint_tolerance: float = 1e-8
    qp_regularization: float = 1e-6
    fit_step: float = .002
    fit_trust: float = .01
    fit_max_samples: int = 3
    reduce_redundant_force_constraints: bool = True
    # ``auto`` uses the bundled CasADi/qpOASES interface when present. ``scipy``
    # remains an explicit reproducibility fallback for historical studies.
    qp_backend: str = "auto"

    def __post_init__(self):
        if not isinstance(self.reduce_redundant_force_constraints, bool):
            raise ValueError("reduce_redundant_force_constraints must be a boolean.")
        if self.qp_backend not in {"auto", "casadi_qpoases", "scipy"}:
            raise ValueError("qp_backend must be 'auto', 'casadi_qpoases', or 'scipy'.")
        for name in ("horizon_cycles", "phase_knots", "integration_substeps", "maximum_sqp_iterations", "fit_max_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.horizon_cycles > 100:
            raise ValueError("PACE-VR horizon_cycles must not exceed 100.")
        for name in ("recruitment_trust", "work_relative_tolerance", "constraint_tolerance",
                     "qp_regularization", "fit_step", "fit_trust"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")


def cyclic_phase_basis(interval_count, knot_count):
    """Nonnegative periodic linear basis with partition of unity."""
    knots = min(interval_count, knot_count)
    coordinate = np.arange(interval_count) * knots / interval_count
    left = np.floor(coordinate).astype(int)
    fraction = coordinate - left
    basis = np.zeros((interval_count, knots))
    basis[np.arange(interval_count), left] += 1 - fraction
    basis[np.arange(interval_count), (left + 1) % knots] += fraction
    return basis


class PaceVrSupervisor:
    """Small work-constrained allocator and a frozen terminal-value sampler.

    ``nonmuscle_power_w`` must follow exactly the sign convention of the RHO:
    net power = omega * sum(moment_coefficient * F) + nonmuscle_power.
    Optional power bounds must come from that same problem; none are invented.
    Every public evaluation requires explicit certification of its source.
    """

    def __init__(self, *, intervals, pulse_width_parameters, angular_velocity_rad_s,
                 required_work_j, nonmuscle_power_w=None, power_lower_w=None,
                 power_upper_w=None, config=None):
        self.config = config or PaceVrConfig()
        self.predictor = CompactMusclePredictor(
            intervals, pulse_width_parameters, substeps=self.config.integration_substeps)
        self.parameters = self.predictor.parameters
        self.intervals = self.predictor.intervals
        self.muscles = len(self.parameters)
        self.phases = len(self.intervals)
        self.omega = float(angular_velocity_rad_s)
        self.required_work_j = float(required_work_j)
        if not np.isfinite(self.omega) or self.omega == 0:
            raise ValueError("angular_velocity_rad_s must be finite and nonzero.")
        if not np.isfinite(self.required_work_j) or self.required_work_j <= 0:
            raise ValueError("required_work_j must be finite and positive.")
        self.power_coefficients = self.omega * np.asarray([i.moment_coefficients for i in self.intervals])
        self.nonmuscle_power = self._phases(nonmuscle_power_w, 0., "nonmuscle_power_w")
        self.lower_power = self._phases(power_lower_w, -np.inf, "power_lower_w")
        self.upper_power = self._phases(power_upper_w, np.inf, "power_upper_w")
        if np.any(self.lower_power > self.upper_power):
            raise ValueError("power bounds are reversed.")
        self.basis = cyclic_phase_basis(self.phases, self.config.phase_knots)
        self.variables = self.muscles * self.basis.shape[1]
        self.command_map = np.kron(np.eye(self.muscles), self.basis)
        negative_gain = self.predictor.gains.reshape(-1, self.muscles) < 0
        negative_after = np.vstack((negative_gain[1:], np.zeros((1, self.muscles), dtype=bool)))
        self._necessary_force_rows = (negative_gain & ~negative_after).ravel()
        self._force_reduction_calcium_safe = all(
            np.all(np.asarray(interval.calcium_amplitudes) >= 0) for interval in self.intervals)
        # These depend only on the sampled problem, not on a projected state.
        # Keeping all eight exponential substeps preserves the reference map.
        self._phase_constants = []
        self._qpoases_qp = _PersistentQpoasesQp()
        if self.config.qp_backend == "casadi_qpoases" and not self._qpoases_qp.available:
            raise RuntimeError("qp_backend=casadi_qpoases was requested, but CasADi qpOASES is unavailable.")
        for phase, interval in enumerate(self.intervals):
            p = self.predictor
            dt = interval.duration / p.substeps
            times = (np.arange(p.substeps) + .5)[:, None] * dt
            cn_decay = np.exp(-times / p.tauc)
            cn_input = np.asarray(interval.calcium_amplitudes) * times / p.tauc
            slow_decay = np.exp(-dt / p.tau_fat)
            slow_constant = -p.tau_fat * np.expm1(-dt / p.tau_fat)
            # Contribution from each substep to the final slow-state endpoint.
            slow_tail = np.exp(-np.arange(p.substeps - 1, -1, -1)[:, None] * dt / p.tau_fat)
            direction = self.command_map.reshape(self.muscles, self.phases, self.variables)[:, phase]
            self._phase_constants.append((dt, cn_decay, cn_input, slow_decay, slow_constant,
                                          slow_tail, direction))

    def _phases(self, value, default, name):
        if value is None:
            return np.full(self.phases, default)
        array = np.broadcast_to(np.asarray(value, float), (self.phases,)).copy()
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite when explicitly supplied.")
        return array

    def _inputs(self, initial_states, pulse_widths, certified):
        if certified is not True:
            raise ValueError("PACE-VR requires an explicitly certified source state and PW schedule.")
        states = np.asarray(initial_states, float).copy()
        self.predictor.phase_map(states, 0)
        widths = np.asarray(pulse_widths, float)
        if widths.shape != (self.muscles, self.phases) or not np.all(np.isfinite(widths)):
            raise ValueError("pulse_widths must be finite (muscles, phases), in seconds.")
        lower, upper = self.predictor.pd0[:, None], self.predictor.pulse_width_max[:, None]
        if np.any(widths < lower) or np.any(widths > upper):
            raise ValueError("pulse_widths outside physical bounds; numerical clipping belongs to the certified caller.")
        return states, -np.expm1(-(widths - lower) / self.predictor.pdt[:, None])

    def _cycle(self, initial_states, recruitment, *, linearize=True):
        """Same frozen-coefficient map as ``_cycle_reference``, condensed by phase.

        The force recurrence is affine during a stimulation interval. Forming
        its small triangular propagator eliminates the Python substep loop,
        while retaining every force/power constraint and fatigue convolution.
        No integration steps, constraints, or SQP derivatives are dropped.
        """
        p = self.predictor
        state = np.asarray(initial_states, float).copy()
        size = self.variables if linearize else 0
        derivative = np.zeros((self.muscles, 5, size))
        work, work_gradient = 0., np.zeros(size)
        force_rows, power_rows = [], []
        reduction_safe = self._force_reduction_calcium_safe and bool(np.all(state[:, 1] >= 0))
        triangular = np.tri(p.substeps, dtype=bool)
        for phase, interval in enumerate(self.intervals):
            dt, cn_decay, cn_input, slow_decay, slow_constant, slow_tail, direction = self._phase_constants[phase]
            start = state.copy()
            reduction_safe = reduction_safe and bool(np.all(start[:, 0] >= 0) and np.all(start[:, 2:] > 0))
            cn = cn_decay * (start[:, 0] + cn_input)
            activation = cn / (start[:, 4] + cn)
            relaxation = start[:, 3] + p.tau2 * activation
            rate_dt = p.gains[phase] / relaxation * dt
            equilibrium = start[:, 2] * activation * relaxation
            cumulative_rate = np.cumsum(rate_dt, axis=0)
            homogeneous = np.exp(-cumulative_rate)
            # exp(-sum of rates after source substep i up to endpoint j).
            # Mask before exponentiating: unused upper-triangle rates must
            # never overflow for otherwise well-defined decaying dynamics.
            differences = cumulative_rate[None, :, :] - cumulative_rate[:, None, :]
            differences[~triangular] = -np.inf
            propagation = np.exp(differences)
            forcing = equilibrium * (-np.expm1(-rate_dt))
            response = np.einsum("jim,im->jm", propagation, forcing, optimize=False)
            force = homogeneous * start[:, 1] + response * recruitment[:, phase]
            previous_force = np.vstack((start[:, 1], force[:-1]))
            force_kernel = dt * exprel(-rate_dt)
            fatigue_kernel = slow_decay * dt * exprel(dt / p.tau_fat - rate_dt)
            integral = force_kernel * previous_force + equilibrium * (dt - force_kernel) * recruitment[:, phase]
            fatigue_integral = (fatigue_kernel * previous_force
                                + equilibrium * (slow_constant - fatigue_kernel) * recruitment[:, phase])
            work += float(self.power_coefficients[phase] @ integral.sum(axis=0)
                          + self.nonmuscle_power[phase] * interval.duration)
            state[:, 1] = force[-1]
            total_decay = np.exp(-interval.duration / p.tau_fat)
            state[:, 2:] = (p.rest + total_decay[:, None] * (start[:, 2:] - p.rest)
                           + p.alpha * np.sum(slow_tail * fatigue_integral, axis=0)[:, None])
            if linearize:
                dforce = (homogeneous[:, :, None] * derivative[None, :, 1]
                          + response[:, :, None] * direction)
                previous_dforce = np.concatenate((derivative[None, :, 1], dforce[:-1]), axis=0)
                dintegral = (force_kernel[:, :, None] * previous_dforce
                             + (equilibrium * (dt - force_kernel))[:, :, None] * direction)
                dfatigue = (fatigue_kernel[:, :, None] * previous_dforce
                            + (equilibrium * (slow_constant - fatigue_kernel))[:, :, None] * direction)
                work_gradient += self.power_coefficients[phase] @ dintegral.sum(axis=0)
                derivative[:, 2:] = (total_decay[:, None, None] * derivative[:, 2:]
                                      + p.alpha[:, :, None] * np.sum(
                                          slow_tail[:, :, None] * dfatigue, axis=0)[:, None, :])
                derivative[:, 1] = dforce[-1]
                dpower = np.einsum("m,jmv->jv", self.power_coefficients[phase], dforce, optimize=False)
            else:
                dforce = np.empty((p.substeps, self.muscles, 0))
                dpower = np.empty((p.substeps, 0))
            power = force @ self.power_coefficients[phase] + self.nonmuscle_power[phase]
            force_rows.extend(zip(force, dforce))
            power_rows.extend((float(power[j]), dpower[j], phase) for j in range(p.substeps))
            state[:, 0] = np.exp(-interval.duration / p.tauc) * (
                start[:, 0] + np.asarray(interval.calcium_amplitudes) * interval.duration / p.tauc)
        return dict(state=state, derivative=derivative, work=work, work_gradient=work_gradient,
                    force_rows=force_rows, power_rows=power_rows, force_constraint_reduction_safe=reduction_safe)

    def _cycle_reference(self, initial_states, recruitment, *, linearize=True):
        """Affine condensation with exact force integrals for frozen rates.

        The reference trajectory matches CompactMusclePredictor. Sensitivities
        freeze the force coefficients at each reference interval, and are
        corrected by a new replay after each QP step (sequential QP).
        """
        p = self.predictor
        state = initial_states.copy()
        size = self.variables if linearize else 0
        derivative = np.zeros((self.muscles, 5, size))
        work, work_gradient = 0., np.zeros(size)
        force_rows, power_rows = [], []
        reduction_safe = self._force_reduction_calcium_safe and bool(np.all(state[:, 1] >= 0))
        for phase, interval in enumerate(self.intervals):
            start = state.copy()
            reduction_safe = reduction_safe and bool(np.all(start[:, 0] >= 0) and np.all(start[:, 2:] > 0))
            dt = interval.duration / p.substeps
            direction = np.zeros((self.muscles, size))
            if linearize:
                for muscle in range(self.muscles):
                    offset = muscle * self.basis.shape[1]
                    direction[muscle, offset:offset + self.basis.shape[1]] = self.basis[phase]
            rec = recruitment[:, phase]
            for substep in range(p.substeps):
                midpoint = (substep + .5) * dt
                cn = np.exp(-midpoint / p.tauc) * (
                    start[:, 0] + np.asarray(interval.calcium_amplitudes) * midpoint / p.tauc)
                activation = cn / (start[:, 4] + cn)
                relaxation = start[:, 3] + p.tau2 * activation
                rate = p.gains[phase, substep] / relaxation
                equilibrium = start[:, 2] * activation * relaxation
                decay = np.exp(-rate * dt)
                force_kernel = dt * exprel(-rate * dt)
                slow_decay = np.exp(-dt / p.tau_fat)
                slow_constant = -p.tau_fat * np.expm1(-dt / p.tau_fat)
                fatigue_kernel = slow_decay * dt * exprel((1 / p.tau_fat - rate) * dt)
                force = state[:, 1].copy()
                dforce = derivative[:, 1].copy()
                integral = force_kernel * force + equilibrium * (dt - force_kernel) * rec
                dintegral = force_kernel[:, None] * dforce + (equilibrium * (dt - force_kernel))[:, None] * direction
                work += float(self.power_coefficients[phase] @ integral + self.nonmuscle_power[phase] * dt)
                work_gradient += self.power_coefficients[phase] @ dintegral
                fatigue_integral = fatigue_kernel * force + equilibrium * (slow_constant - fatigue_kernel) * rec
                dfatigue = fatigue_kernel[:, None] * dforce + (equilibrium * (slow_constant - fatigue_kernel))[:, None] * direction
                state[:, 2:] = p.rest + slow_decay[:, None] * (state[:, 2:] - p.rest) + p.alpha * fatigue_integral[:, None]
                derivative[:, 2:] = slow_decay[:, None, None] * derivative[:, 2:] + p.alpha[:, :, None] * dfatigue[:, None, :]
                state[:, 1] = decay * force + equilibrium * (-np.expm1(-rate * dt)) * rec
                derivative[:, 1] = decay[:, None] * dforce + (equilibrium * (-np.expm1(-rate * dt)))[:, None] * direction
                force_rows.append((state[:, 1].copy(), derivative[:, 1].copy()))
                power_rows.append((float(self.power_coefficients[phase] @ state[:, 1] + self.nonmuscle_power[phase]),
                                   self.power_coefficients[phase] @ derivative[:, 1], phase))
            state[:, 0] = np.exp(-interval.duration / p.tauc) * (
                start[:, 0] + np.asarray(interval.calcium_amplitudes) * interval.duration / p.tauc)
        return dict(state=state, derivative=derivative, work=work, work_gradient=work_gradient,
                    force_rows=force_rows, power_rows=power_rows, force_constraint_reduction_safe=reduction_safe)

    def _constraints(self, model, recruitment, *, force_all=False):
        # All rows are inequalities lower <= matrix @ x <= upper.
        matrices = [self.command_map]
        lower = [-recruitment.ravel()]
        upper = [(self.predictor.maximum_recruitment[:, None] - recruitment).ravel()]
        forces = np.asarray([force for force, _ in model["force_rows"]]).ravel()
        jacobian = np.asarray([jac for _, jac in model["force_rows"]]).reshape(-1, self.variables)
        # With nonnegative recruitment and positive frozen Ding coefficients,
        # a positive-gain step preserves F>=0. A negative-gain step cannot
        # recover a negative F, so constraining the end of every negative run
        # implies all omitted lower bounds. Replay still checks every substep.
        retained = (self._necessary_force_rows if not force_all and self.config.reduce_redundant_force_constraints
                    and model.get("force_constraint_reduction_safe") else np.ones(forces.size, dtype=bool))
        matrices.append(jacobian[retained])
        lower.append(-forces[retained])
        upper.append(np.full(np.count_nonzero(retained), np.inf))
        powers = np.asarray([power for power, _, _ in model["power_rows"]])
        phases = np.asarray([phase for _, _, phase in model["power_rows"]])
        finite = np.isfinite(self.lower_power[phases]) | np.isfinite(self.upper_power[phases])
        if np.any(finite):
            matrices.append(np.asarray([jac for _, jac, _ in model["power_rows"]])[finite])
            lower.append(self.lower_power[phases[finite]] - powers[finite])
            upper.append(self.upper_power[phases[finite]] - powers[finite])
        return np.vstack(matrices), np.concatenate(lower), np.concatenate(upper)

    def _violation(self, model, recruitment):
        violation = max(0., float(-recruitment.min()),
                        float(np.max(recruitment - self.predictor.maximum_recruitment[:, None])))
        forces = np.asarray([force for force, _ in model["force_rows"]])
        powers = np.asarray([power for power, _, _ in model["power_rows"]])
        phases = np.asarray([phase for _, _, phase in model["power_rows"]])
        violation = max(violation, float(-forces.min()), float(np.max(self.lower_power[phases] - powers)),
                        float(np.max(powers - self.upper_power[phases])))
        if not np.all(np.isfinite(model["state"])) or np.any(model["state"][:, 2:] <= 0):
            return float("inf")
        return float(violation)

    @staticmethod
    def _lp_inequalities(matrix, lower, upper):
        finite_upper, finite_lower = np.isfinite(upper), np.isfinite(lower)
        return (np.vstack((matrix[finite_upper], -matrix[finite_lower])),
                np.r_[upper[finite_upper], -lower[finite_lower]])

    def _capacity_margin(self, model, matrix, lower, upper):
        """Envelope for the last SQP model; intermediate envelopes are unused."""
        a_ub, b_ub = self._lp_inequalities(matrix, lower, upper)
        capacity = linprog(-model["work_gradient"], A_ub=a_ub, b_ub=b_ub,
                           bounds=[(None, None)] * self.variables, method="highs")
        return ((model["work"] - capacity.fun) / self.required_work_j - 1
                if capacity.success else float("nan"))

    def _solve_recruitment_qp(self, hessian, gradient, matrix, lower, upper,
                              work_gradient, work_delta):
        """Solve one condensed SQP QP without changing its replay contract."""
        use_qpoases = self.config.qp_backend == "casadi_qpoases" or (
            self.config.qp_backend == "auto" and self._qpoases_qp.available)
        if use_qpoases:
            result = self._qpoases_qp.solve(
                hessian, gradient, matrix, lower, upper, work_gradient,
                work_delta, self.config.recruitment_trust, self.required_work_j)
            if result["success"] or self.config.qp_backend == "casadi_qpoases":
                return result
            # ``auto`` must never turn an optional native backend regression
            # into a failed PACE projection. Keep the exact SciPy reference as
            # a documented numerical fallback and record it in the audit.
            native_message = result["message"]
        else:
            native_message = None
        constraints = [LinearConstraint(matrix, lower, upper),
                       LinearConstraint(work_gradient[None], work_delta, work_delta)]
        solution = minimize(lambda x: .5 * x @ hessian @ x + gradient @ x,
                            np.zeros(self.variables), jac=lambda x: hessian @ x + gradient,
                            method="SLSQP", bounds=Bounds(-self.config.recruitment_trust,
                                                           self.config.recruitment_trust),
                            constraints=constraints, options={"ftol": 1e-10, "maxiter": 100})
        return dict(success=bool(solution.success), x=np.asarray(solution.x, float),
                    message=str(solution.message), backend="scipy_slsqp",
                    iterations=getattr(solution, "nit", None),
                    native_qp_fallback_message=native_message)

    def _one_cycle(self, state, recruitment, weights):
        residual_tolerance = self.config.work_relative_tolerance * self.required_work_j
        current = recruitment.copy()
        full_constraint_retries = 0
        for iteration in range(self.config.maximum_sqp_iterations):
            model = self._cycle(state, current)
            matrix, lower, upper = self._constraints(model, current)
            scales = self.predictor.rest[:, 0]
            damage = (state[:, 2] - model["state"][:, 2]) / scales
            jac = -model["derivative"][:, 2] / scales[:, None]
            hessian = jac.T @ (weights[:, None] * jac)
            gradient = jac.T @ (weights * damage)
            scale = max(float(np.max(np.abs(hessian))), float(np.max(np.abs(gradient))), 1e-14)
            hessian = hessian / scale + self.config.qp_regularization * np.eye(self.variables)
            gradient /= scale
            work_delta = self.required_work_j - model["work"]
            solution = self._solve_recruitment_qp(hessian, gradient, matrix, lower, upper,
                                                  model["work_gradient"], work_delta)
            candidate = current + (self.command_map @ solution["x"]).reshape(current.shape)
            if (self.config.reduce_redundant_force_constraints and model.get("force_constraint_reduction_safe")
                    and (np.any(candidate < -self.config.constraint_tolerance)
                         or np.any(candidate > self.predictor.maximum_recruitment[:, None]
                                   + self.config.constraint_tolerance))):
                # Equivalent constraints can change SLSQP's active-set path
                # near a boundary. Retry the complete historical rows before
                # refusing an out-of-bounds candidate. Other SLSQP statuses
                # retain the reference policy: the numerical replay decides.
                local_margin = self._capacity_margin(model, matrix, lower, upper)
                # A negative LP envelope proves this *affine QP* cannot meet
                # the work equality even without its trust bounds. Equivalent
                # force rows cannot repair it. Keep the ordinary SQP replay/
                # iteration policy, but omit that redundant recovery solve.
                if not np.isfinite(local_margin) or local_margin >= -self.config.work_relative_tolerance:
                    matrix, lower, upper = self._constraints(model, current, force_all=True)
                    solution = self._solve_recruitment_qp(hessian, gradient, matrix, lower, upper,
                                                          model["work_gradient"], work_delta)
                    candidate = current + (self.command_map @ solution["x"]).reshape(current.shape)
                    full_constraint_retries += 1
            # Only roundoff from the bounded QP may be projected.
            if np.any(candidate < -self.config.constraint_tolerance) or np.any(
                    candidate > self.predictor.maximum_recruitment[:, None] + self.config.constraint_tolerance):
                margin = self._capacity_margin(model, matrix, lower, upper)
                return dict(accepted=False, status="qp_bound_failure", margin=margin, iterations=iteration + 1,
                            full_constraint_retries=full_constraint_retries,
                            recruitment_bound_violation=max(float(-candidate.min()), float(np.max(
                                candidate - self.predictor.maximum_recruitment[:, None]))),
                            qp_success=bool(solution["success"]), qp_message=str(solution["message"]),
                            qp_backend=solution["backend"])
            candidate = np.clip(candidate, 0., self.predictor.maximum_recruitment[:, None])
            replay = self._cycle(state, candidate, linearize=False)
            violation = self._violation(replay, candidate)
            residual = abs(replay["work"] - self.required_work_j)
            if violation <= self.config.constraint_tolerance and residual <= residual_tolerance:
                margin = self._capacity_margin(model, matrix, lower, upper)
                return dict(accepted=True, status="projected_cycle_validated", state=replay["state"],
                            recruitment=candidate, work=replay["work"], work_residual=residual,
                            constraint_violation=violation, margin=margin, iterations=iteration + 1,
                            qp_success=bool(solution["success"]), qp_message=str(solution["message"]),
                            qp_backend=solution["backend"], qp_inequality_rows=matrix.shape[0],
                            full_constraint_retries=full_constraint_retries)
            current = candidate
        margin = self._capacity_margin(model, matrix, lower, upper)
        return dict(accepted=False, status="projected_cycle_not_validated", margin=margin,
                    work_residual=residual, constraint_violation=violation,
                    iterations=self.config.maximum_sqp_iterations, qp_message=str(solution["message"]),
                    qp_backend=solution["backend"],
                    full_constraint_retries=full_constraint_retries)

    def evaluate(self, initial_states, pulse_widths, *, certified=False, weights=None,
                 candidate_name="unit", deadline_monotonic=None):
        """Return an audited work-feasible prefix; never call it physiological failure."""
        started = perf_counter()
        current, recruitment = self._inputs(initial_states, pulse_widths, certified)
        weights = np.ones(self.muscles) if weights is None else np.asarray(weights, float)
        if weights.shape != (self.muscles,) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("weights must contain one finite positive value per muscle.")
        weights = weights / np.exp(np.mean(np.log(weights)))
        cycles, histories, widths, margins = [], [current.tolist()], [], []
        for _ in range(self.config.horizon_cycles):
            if deadline_monotonic is not None and monotonic() >= deadline_monotonic:
                cycle = dict(accepted=False, status="deadline_expired", margin=float("nan"))
                break
            cycle = self._one_cycle(current, recruitment, weights)
            margins.append(float(cycle["margin"]))
            if not cycle["accepted"]:
                break
            current, recruitment = cycle["state"], cycle["recruitment"]
            histories.append(current.tolist())
            widths.append((self.predictor.pd0[:, None] - self.predictor.pdt[:, None] * np.log1p(-recruitment)).tolist())
            cycles.append(cycle)
        completed = len(cycles)
        finite_margins = [m for m in margins if np.isfinite(m)]
        minimum_margin = min(finite_margins) if finite_margins else None
        # Censored horizon progress + continuous margin. Lower is preferable.
        value = 1 - completed / self.config.horizon_cycles
        if minimum_margin is not None:
            value -= float(np.tanh(minimum_margin)) / self.config.horizon_cycles
        return dict(status="complete" if completed == self.config.horizon_cycles else cycle["status"],
                    accepted=completed == self.config.horizon_cycles,
                    feasible_prefix_cycles=completed, minimum_task_margin=minimum_margin,
                    terminal_value=float(value), terminal_state=current[:, 2:].tolist(),
                    terminal_full_state=current.tolist(),
                    work_residual_max=max((c["work_residual"] for c in cycles), default=0.),
                    constraint_violation_max=max((c["constraint_violation"] for c in cycles), default=0.),
                    work_per_cycle_j=[c["work"] for c in cycles], state_history=histories,
                    pulse_widths=widths, runtime_s=perf_counter() - started, local_fit=None,
                    candidate=dict(name=str(candidate_name), weights=weights.tolist()),
                    reallocation=dict(variables=self.variables, phase_knots=self.basis.shape[1],
                                      updates=completed, sqp_iterations=[c["iterations"] for c in cycles],
                                      qp_backends=[c["qp_backend"] for c in cycles],
                                      qp_inequality_rows=[c["qp_inequality_rows"] for c in cycles],
                                      full_constraint_retries=sum(c["full_constraint_retries"] for c in cycles),
                                      full_force_rows=self.phases * self.predictor.substeps * self.muscles,
                                      necessary_force_rows=int(np.count_nonzero(self._necessary_force_rows))),
                    context=dict(config=asdict(self.config), required_work_j=self.required_work_j,
                                 uses_fho_data=False, physiological_failure_certified=False,
                                 quadrature="exponential_force_integral_piecewise_geometry",
                                 envelope="local_condensed_recruitment_subspace",
                                 power_bounds_supplied=bool(np.any(np.isfinite(self.lower_power)) or
                                                            np.any(np.isfinite(self.upper_power)))),
                    first_failure=None if completed == self.config.horizon_cycles else
                    {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                     for k, v in cycle.items() if k not in {"state", "recruitment"}})

    def fit_terminal(self, initial_states, pulse_widths, *, certified=False, weights=None,
                     deadline_monotonic=None):
        """Fit a budgeted low-rank value in exact Ding fatigue-memory coordinates.

        Tau1/Km offsets from the A-driven fatigue manifold are held fixed.
        At most ``fit_max_samples`` one-sided directional rollouts are added
        to the baseline. Orthogonal muscle-redistribution directions precede
        the common-capacity direction. The gradient acts on normalized A only;
        unsampled directions are zero and its rank is explicitly reported.
        There is no claim of a full state Hessian. A branch change refuses the
        fit. Insufficient remaining time yields a refused constant-only result.
        """
        started = perf_counter()
        states, _ = self._inputs(initial_states, pulse_widths, certified)
        base = self.evaluate(states, pulse_widths, certified=True, weights=weights,
                             deadline_monotonic=deadline_monotonic)
        gradient = np.zeros((self.muscles, 3))
        curvature = np.zeros_like(gradient)
        samples, reasons = [], []
        if not base["accepted"]:
            reasons.append("base_rollout_incomplete")
        step = self.config.fit_step
        if any(p.fatigue.alpha_a >= 0 for p in self.parameters):
            reasons.append("damage_coordinate_requires_negative_alpha_a")
        if self.muscles > 1:
            contrasts = np.eye(self.muscles)[:, :-1] - np.eye(self.muscles)[:, -1, None]
            orthogonal, _ = np.linalg.qr(contrasts)
            directions = list(orthogonal.T)
        else:
            directions = []
        directions.append(np.ones(self.muscles) / np.sqrt(self.muscles))
        durations = [base["runtime_s"]]
        budget_limited = False
        if not reasons:
            for index, direction in enumerate(directions[:self.config.fit_max_samples]):
                if deadline_monotonic is not None and deadline_monotonic - monotonic() < 1.2 * max(durations):
                    budget_limited = True
                    break
                perturbed = states.copy()
                for muscle, parameter in enumerate(self.parameters):
                    fatigue = parameter.fatigue
                    perturbed[muscle, 2:] += step * direction[muscle] * fatigue.a_rest * fatigue.alpha / fatigue.alpha_a
                if np.any(perturbed[:, 2:] <= 0):
                    reasons.append(f"direction_{index}_perturbation_outside_domain")
                    break
                outcome = self.evaluate(perturbed, pulse_widths, certified=True, weights=weights,
                                        deadline_monotonic=deadline_monotonic)
                durations.append(outcome["runtime_s"])
                samples.append(dict(direction=direction.tolist(), value=outcome["terminal_value"],
                                    feasible_prefix_cycles=outcome["feasible_prefix_cycles"]))
                if outcome["status"] == "deadline_expired":
                    reasons.append("deadline_expired")
                    break
                if outcome["feasible_prefix_cycles"] != base["feasible_prefix_cycles"]:
                    reasons.append(f"direction_{index}_rollout_branch_changed")
                    break
                gradient[:, 0] += (outcome["terminal_value"] - base["terminal_value"]) / step * direction
        if not samples:
            reasons.append("no_directional_sample_available")
        base["local_fit"] = dict(accepted=not reasons, reasons=reasons,
                                 fit_mode="forward_directional" if samples else "constant_only",
                                 sample_count=len(samples), fit_rank=len(samples),
                                 sample_budget=self.config.fit_max_samples, budget_limited=budget_limited,
                                 constant=base["terminal_value"], gradient=gradient.tolist(),
                                 diagonal_hessian=curvature.tolist(),
                                 reference_state=states[:, 2:].tolist(),
                                 state_scales=self.predictor.rest.tolist(),
                                 trust_bounds=[-self.config.fit_trust, self.config.fit_trust],
                                 context="normalized_A_with_exact_Ding_memory_offsets_fixed",
                                 samples=samples)
        base["runtime_s"] = perf_counter() - started
        return base

    def fit_predicted_terminal(self, initial_states, pulse_widths, *, certified=False, weights=None,
                               deadline_monotonic=None):
        """Fit the local value at the nominal *next* terminal state.

        The fast RHO optimises its next terminal state, not the boundary from
        which the slow calculation was launched.  Centering the local model at
        this predicted terminal removes the deterministic one-cycle fatigue
        drift from its trust region.  It costs one compact rollout outside the
        RHO and retains the same work-constrained allocator for every step.
        """
        started = perf_counter()
        preview = self.evaluate(initial_states, pulse_widths, certified=certified, weights=weights,
                                deadline_monotonic=deadline_monotonic)
        if not preview["accepted"] or len(preview["state_history"]) < 2 or not preview["pulse_widths"]:
            preview["local_fit"] = dict(accepted=False, reasons=["nominal_next_terminal_unavailable"],
                                        fit_mode="constant_only", sample_count=0, fit_rank=0)
            preview["runtime_s"] = perf_counter() - started
            return preview
        fit = self.fit_terminal(np.asarray(preview["state_history"][1], dtype=float),
                                np.asarray(preview["pulse_widths"][0], dtype=float), certified=True,
                                weights=weights, deadline_monotonic=deadline_monotonic)
        fit["reference_terminal"] = "nominal_next_cycle_projected_state"
        fit["source_preview"] = {key: preview[key] for key in (
            "accepted", "feasible_prefix_cycles", "minimum_task_margin", "terminal_value",
            "work_residual_max", "constraint_violation_max")}
        fit["runtime_s"] = perf_counter() - started
        return fit


@dataclass(frozen=True)
class PaceVrSnapshot:
    """Immutable, JSON/pickle-safe request for a dedicated asynchronous worker.

    The owner must check request identity/context, deadline, source age and fit
    acceptance before installation at a certified RHO boundary. A late result
    never authorizes replacing the last accepted value.
    """

    request_id: str
    source_cycle: int
    context_digest: str
    deadline_monotonic: float
    payload_json: str


def create_pace_vr_snapshot(supervisor, initial_states, pulse_widths, *, request_id,
                           source_cycle, deadline_seconds, certified=False, weights=None,
                           fit_reference="source_terminal"):
    """Copy certified inputs, including sampled geometry, into an immutable request."""
    states, _ = supervisor._inputs(initial_states, pulse_widths, certified)
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("request_id must be a nonempty string.")
    if isinstance(source_cycle, bool) or not isinstance(source_cycle, int) or source_cycle < 0:
        raise ValueError("source_cycle must be a nonnegative integer.")
    if not np.isfinite(deadline_seconds) or deadline_seconds <= 0:
        raise ValueError("deadline_seconds must be finite and positive.")
    if fit_reference not in {"source_terminal", "predicted_next_terminal"}:
        raise ValueError("Unknown PACE-VR fit_reference.")
    phases = []
    for phase, interval in enumerate(supervisor.intervals):
        phases.append(dict(duration=interval.duration,
                           calcium_amplitudes=list(interval.calcium_amplitudes),
                           mechanical_gain_samples=supervisor.predictor.gains[phase].T.tolist(),
                           moment_coefficients=list(interval.moment_coefficients),
                           target_moments=list(interval.target_moments)))
    context = dict(config=asdict(supervisor.config), intervals=phases,
                   pulse_width_parameters=[asdict(p) for p in supervisor.parameters],
                   angular_velocity_rad_s=supervisor.omega, required_work_j=supervisor.required_work_j,
                   nonmuscle_power_w=supervisor.nonmuscle_power.tolist(),
                   power_lower_w=supervisor.lower_power.tolist() if np.all(np.isfinite(supervisor.lower_power)) else None,
                   power_upper_w=supervisor.upper_power.tolist() if np.all(np.isfinite(supervisor.upper_power)) else None,
                   fit_reference=fit_reference)
    digest = hashlib.sha256(json.dumps(context, allow_nan=False, sort_keys=True).encode()).hexdigest()
    payload = dict(context=context, initial_states=states.tolist(),
                   pulse_widths=np.asarray(pulse_widths).tolist(),
                   weights=None if weights is None else np.asarray(weights).tolist())
    return PaceVrSnapshot(request_id, source_cycle, digest, monotonic() + deadline_seconds,
                          json.dumps(payload, allow_nan=False, sort_keys=True))


@dataclass(frozen=True)
class _SampledGain:
    values: tuple
    duration: float

    def __call__(self, time):
        return self.values[min(len(self.values) - 1, max(0, int(time / self.duration * len(self.values))))]


def _supervisor_from_snapshot(snapshot, *, cache=None):
    """Rebuild one rollout model from a certified, immutable snapshot."""
    if isinstance(snapshot, dict):
        snapshot = PaceVrSnapshot(**snapshot)
    if not isinstance(snapshot, PaceVrSnapshot):
        raise TypeError("snapshot must be a PaceVrSnapshot or its dataclass dictionary.")
    payload = json.loads(snapshot.payload_json)
    context = payload["context"]
    digest = hashlib.sha256(json.dumps(context, allow_nan=False, sort_keys=True).encode()).hexdigest()
    if digest != snapshot.context_digest:
        raise ValueError("Snapshot context digest mismatch.")
    if cache is not None and digest in cache:
        return cache[digest], payload
    parameters = []
    for entry in context["pulse_width_parameters"]:
        entry = dict(entry)
        entry["fatigue"] = DingFatigueParameters(**entry["fatigue"])
        parameters.append(DingPulseWidthParameters(**entry))
    intervals = []
    for phase in context["intervals"]:
        intervals.append(MomentTrackingInterval(
            phase["duration"], tuple(phase["calcium_amplitudes"]),
            tuple(_SampledGain(tuple(samples), phase["duration"]) for samples in phase["mechanical_gain_samples"]),
            tuple(phase["moment_coefficients"]), tuple(phase["target_moments"])))
    supervisor = PaceVrSupervisor(
        intervals=intervals, pulse_width_parameters=parameters, config=PaceVrConfig(**context["config"]),
        **{key: context[key] for key in ("angular_velocity_rad_s", "required_work_j", "nonmuscle_power_w",
                                       "power_lower_w", "power_upper_w")})
    if cache is not None:
        # A worker keeps only the current geometry/configuration.  The QP
        # layouts and warm starts owned by this supervisor survive snapshots.
        cache.clear()
        cache[digest] = supervisor
    return supervisor, payload


def evaluate_pace_vr_weight_candidates(snapshot, candidates, *, deadline_monotonic=None,
                                       estimated_candidate_runtime_s=0., supervisor_cache=None):
    """Replay candidate policies from the same state, geometry and horizon.

    The value is the reduced rollout's censored prefix/margin score. It is a
    relative screening score, not a certificate of physiological endurance.
    """
    if isinstance(snapshot, dict):
        snapshot = PaceVrSnapshot(**snapshot)
    supervisor, payload = _supervisor_from_snapshot(snapshot, cache=supervisor_cache)
    deadline = snapshot.deadline_monotonic if deadline_monotonic is None else min(
        snapshot.deadline_monotonic, deadline_monotonic)
    evaluated = {}
    runtime_estimate = max(0., float(estimated_candidate_runtime_s))
    for name, weights in candidates.items():
        started = monotonic()
        if started >= deadline or (runtime_estimate > 0 and deadline - started < 1.2 * runtime_estimate):
            break
        outcome = supervisor.evaluate(payload["initial_states"], payload["pulse_widths"],
                                      certified=True, weights=weights, candidate_name=name,
                                      deadline_monotonic=deadline)
        completed = monotonic()
        runtime_estimate = max(runtime_estimate, outcome["runtime_s"])
        evaluated[name] = {key: outcome[key] for key in (
            "accepted", "status", "feasible_prefix_cycles", "minimum_task_margin",
            "terminal_value", "runtime_s", "work_residual_max", "constraint_violation_max")}
        evaluated[name].update(started_monotonic=started, completed_monotonic=completed,
                               deadline_met=completed <= deadline)
        if outcome["status"] == "deadline_expired" or completed > deadline:
            break
    return evaluated


def run_pace_vr_snapshot(snapshot, *, supervisor_cache=None):
    """Top-level worker callable; CPU affinity and process lifecycle are owner concerns."""
    if isinstance(snapshot, dict):
        snapshot = PaceVrSnapshot(**snapshot)
    supervisor, payload = _supervisor_from_snapshot(snapshot, cache=supervisor_cache)
    fitting = (supervisor.fit_predicted_terminal if payload["context"].get("fit_reference") == "predicted_next_terminal"
               else supervisor.fit_terminal)
    result = fitting(payload["initial_states"], payload["pulse_widths"], certified=True,
                     weights=payload["weights"], deadline_monotonic=snapshot.deadline_monotonic)
    result.update(request_id=snapshot.request_id, source_cycle=snapshot.source_cycle,
                  context_digest=snapshot.context_digest, deadline_monotonic=snapshot.deadline_monotonic,
                  completed_monotonic=monotonic(), applied_cycle=None)
    result["deadline_met"] = result["completed_monotonic"] <= snapshot.deadline_monotonic
    result["accepted"] = bool(result["accepted"] and result["local_fit"]["accepted"] and result["deadline_met"])
    return result
