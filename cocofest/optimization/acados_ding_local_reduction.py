"""Opt-in native ACADOS reduction of periodic-node Ding muscle states.

Bioptim retains its original state-space contract. A native capsule is built
from a transformed copy of its AcadosOcp and this adapter maps node variables,
bounds and numerical inputs in both directions. IRK stages use the *discrete*
affine profiles of the selected tableau, not continuous exponential solutions.
Native pi/lam and flat iterates belong to the reduced formulation exclusively.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter

import casadi as ca
import numpy as np


def irk_tableau(stages: int, family: str):
    nodes = np.asarray(ca.collocation_points(stages, family))
    basis = []
    for j, node in enumerate(nodes):
        poly = np.poly1d([1.0])
        for k, other in enumerate(nodes):
            if j != k:
                poly *= np.poly1d([1.0, -other]) / (node - other)
        basis.append(np.polyint(poly))
    matrix = np.array([[poly(c) - poly(0) for poly in basis] for c in nodes])
    weights = np.array([poly(1) - poly(0) for poly in basis])
    return nodes, matrix, weights


def affine_irk_interval(value, rest, tau, amplitude, duration, substeps, tableau):
    """Eliminate Cn and fatigue-offset equations for one shooting interval."""
    nodes, matrix, weights = tableau
    value = np.array(value, dtype=float, copy=True)
    rest, tau = np.asarray(rest), np.asarray(tau)
    step = duration / substeps
    samples = []
    for index in range(substeps):
        forcing = np.repeat((rest / tau)[:, None], len(nodes), axis=1)
        forcing[0] = amplitude * np.exp(-(index + nodes) * step / tau[0]) / tau[0]
        stages = np.array([
            np.linalg.solve(np.eye(len(nodes)) + step / tau[k] * matrix,
                            np.full(len(nodes), value[k]) + step * matrix @ forcing[k])
            for k in range(3)
        ])
        value += step * ((forcing - stages / tau[:, None]) @ weights)
        samples.append(stages)
    return np.array(samples), value


class AffineIRKKernel:
    """Cache tableau solves: RHO refresh performs only small affine products."""

    def __init__(self, rest, tau, duration, substeps, tableau):
        nodes, matrix, weights = tableau
        rest, tau = np.asarray(rest), np.asarray(tau)
        step = duration / substeps
        inverses = np.array([np.linalg.solve(np.eye(len(nodes)) + step / decay * matrix,
                                             np.eye(len(nodes))) for decay in tau])
        self.stage_slope = inverses @ np.ones(len(nodes))
        self.endpoint_slope = 1 - step / tau * (self.stage_slope @ weights)
        self.stage_offset, self.endpoint_offset = [], []
        for index in range(substeps):
            forcing = np.repeat((rest / tau)[:, None], len(nodes), axis=1)
            forcing[0] = np.exp(-(index + nodes) * step / tau[0]) / tau[0]
            offset = np.einsum("kij,kj->ki", inverses, step * (forcing @ matrix.T))
            self.stage_offset.append(offset)
            self.endpoint_offset.append(step * ((forcing - offset / tau[:, None]) @ weights))

    def interval(self, value, amplitude):
        value = np.asarray(value).copy()
        factor = np.array([amplitude, 1.0, 1.0])
        samples = []
        for stage_offset, endpoint_offset in zip(self.stage_offset, self.endpoint_offset):
            samples.append(self.stage_slope * value[:, None] + stage_offset * factor[:, None])
            value = self.endpoint_slope * value + endpoint_offset * factor
        return np.asarray(samples), value


def interpolate_irk_stages(time, parameters, duration, substeps, nodes):
    step = duration / substeps
    pieces = []
    cursor = 0
    for index in range(substeps):
        local = time / step - index
        values = []
        for _ in range(3):
            value = 0
            for j, node in enumerate(nodes):
                basis = 1
                for k, other in enumerate(nodes):
                    if j != k:
                        basis *= (local - other) / (node - other)
                value += parameters[cursor + j] * basis
            cursor += len(nodes)
            values.append(value)
        pieces.append(ca.vertcat(*values))
    result = pieces[-1]
    for index in reversed(range(substeps - 1)):
        result = ca.if_else(time <= (index + 1) * step, pieces[index], result)
    return result


@dataclass(frozen=True)
class MuscleRows:
    model: object
    cn: int
    a: int
    tau1: int
    km: int

    @property
    def ratios(self):
        return np.array([self.model.alpha_tau1, self.model.alpha_km]) / self.model.alpha_a


class DingLocalMapping:
    """All dimensions are explicit; native decisions use Bioptim scaling."""

    def __init__(self, interface):
        from cocofest.models.ding2007.ding2007_with_fatigue_periodic_node import (
            DingModelPulseWidthFrequencyWithFatiguePeriodicNode,
        )

        self.interface = interface
        self.nlp = interface.ocp.nlp[0]
        original = interface.acados_ocp
        options = original.solver_options
        if interface.nparams or interface.ocp.n_phases != 1:
            raise ValueError("Ding ACADOS reduction requires one phase and no optimized parameters")
        if options.integrator_type != "IRK":
            raise ValueError("Ding ACADOS reduction requires an IRK integrator")
        for name in ("idxsbx", "idxsbx_0", "idxsbx_e", "C", "C_e"):
            if np.asarray(getattr(original.constraints, name, np.empty(0))).size:
                raise ValueError("Ding reduction does not support softened state bounds or linear state constraints")
        families = {"GAUSS_LEGENDRE": "legendre", "GAUSS_RADAU_IIA": "radau"}
        if options.collocation_type not in families:
            raise ValueError("Unsupported Ding ACADOS collocation tableau")
        self.family = families[options.collocation_type]
        def constant_integer(value, label):
            array = np.asarray(value).reshape(-1)
            if not array.size or np.any(array != array[0]) or array[0] < 1:
                raise ValueError(f"Ding reduction requires constant positive {label}")
            return int(array[0])
        self.stages = constant_integer(options.sim_method_num_stages, "IRK stage count")
        self.substeps = constant_integer(options.sim_method_num_steps, "IRK substep count")
        self.tableau = irk_tableau(self.stages, self.family)
        self.horizon = int(options.N_horizon)
        self.duration = float(options.tf) / self.horizon
        if options.time_steps is not None and np.asarray(options.time_steps).size:
            if not np.allclose(options.time_steps, self.duration, rtol=0, atol=1e-14):
                raise ValueError("Ding reduction requires a uniform fixed shooting mesh")
        self.nx = int(original.model.x.numel())
        self.np = int(original.model.p.numel())
        if self.np != 2 or set(self.nlp.numerical_data_timeseries) != {"periodic_calcium"}:
            raise ValueError("Ding reduction currently requires only periodic_calcium runtime data")
        self.scaling = np.ones(self.nx)
        for key in self.nlp.states.keys():
            self.scaling[self.nlp.states[key].index] = np.asarray(self.nlp.x_scaling[key].scaling).reshape(-1)
        if np.any(self.scaling <= 0) or not np.isfinite(self.scaling).all():
            raise ValueError("Ding reduction requires positive finite state scaling")
        self.muscles = []
        for model in self.nlp.model.muscles_dynamics_model:
            if type(model) is not DingModelPulseWidthFrequencyWithFatiguePeriodicNode:
                raise ValueError("Ding reduction supports the exact periodic-node fatigue model only")
            if model.alpha_a == 0 or not np.isclose(model._stim_interval, self.duration):
                raise ValueError("Ding reduction requires nonzero alpha_A and one stimulation per interval")
            rows = [int(self.nlp.states[f"{key}_{model.muscle_name}"].index[0])
                    for key in ("Cn", "A", "Tau1", "Km")]
            self.muscles.append(MuscleRows(model, *rows))
        if not self.muscles:
            raise ValueError("No periodic-node Ding muscles to reduce")
        removed = {row for muscle in self.muscles for row in (muscle.cn, muscle.tau1, muscle.km)}
        self.keep = np.array([row for row in range(self.nx) if row not in removed], dtype=int)
        self.inverse = {int(row): index for index, row in enumerate(self.keep)}
        self.parameters_per_muscle = 3 + 3 * self.stages * self.substeps
        self.extra_np = self.parameters_per_muscle * len(self.muscles)
        self.physiology_signature = self._physiology_signature()
        self.kernels = []
        for muscle in self.muscles:
            rest = np.array([0.0, *(np.array([muscle.model.tau1_rest, muscle.model.km_rest]) - muscle.ratios * muscle.model.a_scale)])
            tau = np.array([muscle.model.tauc, muscle.model.tau_fat, muscle.model.tau_fat])
            self.kernels.append(AffineIRKKernel(rest, tau, self.duration, self.substeps, self.tableau))
        self.node_offsets = np.empty((self.horizon + 1, len(self.muscles), 3))
        self.profiles = np.empty((self.horizon + 1, self.extra_np))
        self.refresh()

    def _physiology_signature(self):
        return tuple(tuple(float(getattr(muscle.model, key)) for key in
                           ("alpha_a", "alpha_tau1", "alpha_km", "a_scale", "tau1_rest", "km_rest", "tauc", "tau_fat", "_stim_interval"))
                     for muscle in self.muscles)

    def refresh(self):
        start = perf_counter()
        if self._physiology_signature() != self.physiology_signature:
            raise ValueError("Changed Ding physiological parameters require rebuilding the local capsule")
        nlp = self.nlp
        physical_lower = np.empty(self.nx)
        physical_upper = np.empty(self.nx)
        for key in nlp.states.keys():
            rows = nlp.states[key].index
            physical_lower[rows] = np.asarray(nlp.x_bounds[key].min[:, 0]).reshape(-1)
            physical_upper[rows] = np.asarray(nlp.x_bounds[key].max[:, 0]).reshape(-1)
        removed = [row for muscle in self.muscles for row in (muscle.cn, muscle.a, muscle.tau1, muscle.km)]
        if not np.allclose(physical_lower[removed], physical_upper[removed], rtol=0, atol=1e-12):
            raise ValueError("Ding reduction requires fixed initial Cn,A,Tau1,Km bounds at each RHO")
        runtime = np.asarray(nlp.numerical_data_timeseries["periodic_calcium"])
        if runtime.shape != (2, 1, self.horizon + 1):
            raise ValueError("periodic_calcium runtime shape changed")
        for j, muscle in enumerate(self.muscles):
            ratios = muscle.ratios
            initial = physical_lower
            value = np.array([initial[muscle.cn], *(initial[[muscle.tau1, muscle.km]] - ratios * initial[muscle.a])])
            section = slice(j * self.parameters_per_muscle, (j + 1) * self.parameters_per_muscle)
            for node in range(self.horizon + 1):
                self.node_offsets[node, j] = value
                values, endpoint = self.kernels[j].interval(value, runtime[0, 0, node])
                self.profiles[node, section] = np.concatenate((value, values.reshape(-1)))
                value = endpoint
        self.preparation_seconds = perf_counter() - start

    def lift(self, reduced, node):
        reduced = np.asarray(reduced).reshape(-1)
        result = np.empty(self.nx)
        result[self.keep] = reduced
        for j, muscle in enumerate(self.muscles):
            offsets = self.node_offsets[node, j]
            physical_a = result[muscle.a] * self.scaling[muscle.a]
            result[muscle.cn] = offsets[0] / self.scaling[muscle.cn]
            result[[muscle.tau1, muscle.km]] = (muscle.ratios * physical_a + offsets[1:]) / self.scaling[[muscle.tau1, muscle.km]]
        return result

    def bounds(self, lower, upper, node):
        lower, upper = np.asarray(lower).reshape(-1), np.asarray(upper).reshape(-1)
        if lower.size != self.nx or upper.size != self.nx:
            raise ValueError("Expected original-space state bounds")
        lo, hi = lower[self.keep].copy(), upper[self.keep].copy()
        for j, muscle in enumerate(self.muscles):
            offsets = self.node_offsets[node, j]
            cn = offsets[0] / self.scaling[muscle.cn]
            if cn < lower[muscle.cn] - 1e-9 or cn > upper[muscle.cn] + 1e-9:
                raise ValueError(f"Reconstructed Cn violates original bounds at node {node}")
            a_index = self.inverse[muscle.a]
            for row, ratio, offset in zip((muscle.tau1, muscle.km), muscle.ratios, offsets[1:]):
                coefficient = ratio * self.scaling[muscle.a] / self.scaling[row]
                offset /= self.scaling[row]
                if coefficient == 0:
                    if offset < lower[row] - 1e-9 or offset > upper[row] + 1e-9:
                        raise ValueError("Reconstructed constant violates original fatigue bounds")
                    continue
                endpoints = [(lower[row] - offset) / coefficient, (upper[row] - offset) / coefficient]
                lo[a_index] = max(lo[a_index], min(endpoints))
                hi[a_index] = min(hi[a_index], max(endpoints))
        if np.any(lo > hi + 1e-10):
            raise ValueError(f"Empty reconstructed fatigue-bound intersection at node {node}")
        return lo, np.maximum(hi, lo)

    def transform(self, original):
        reduced = deepcopy(original)
        model = reduced.model
        x = ca.SX.sym("ding_local_x", self.keep.size)
        parameters = ca.SX.sym("ding_local_profiles", self.extra_np)
        node_lift, stage_lift = ca.SX.zeros(self.nx), ca.SX.zeros(self.nx)
        node_lift[self.keep], stage_lift[self.keep] = x, x
        for j, muscle in enumerate(self.muscles):
            cursor = j * self.parameters_per_muscle
            node_values = parameters[cursor:cursor + 3]
            stage_values = interpolate_irk_stages(model.t, parameters[cursor + 3:cursor + self.parameters_per_muscle],
                self.duration, self.substeps, self.tableau[0])
            for lift, values in ((node_lift, node_values), (stage_lift, stage_values)):
                lift[muscle.cn] = values[0] / self.scaling[muscle.cn]
                for row, ratio, offset in zip((muscle.tau1, muscle.km), muscle.ratios, (values[1], values[2])):
                    lift[row] = (ratio * lift[muscle.a] * self.scaling[muscle.a] + offset) / self.scaling[row]
        old_x = model.x
        model.f_expl_expr = ca.substitute(model.f_expl_expr, old_x, stage_lift)[self.keep]
        model.x = x
        model.xdot = ca.SX.sym("ding_local_xdot", self.keep.size)
        model.f_impl_expr = model.xdot - model.f_expl_expr
        model.p = ca.vertcat(model.p, parameters)
        for suffix in ("", "_0", "_e"):
            if getattr(reduced.cost, "cost_type" + suffix) != "NONLINEAR_LS":
                raise ValueError("Ding ACADOS reduction currently supports NONLINEAR_LS costs only")
            for prefix in ("cost_y_expr", "con_h_expr"):
                name = prefix + suffix
                value = getattr(model, name)
                if isinstance(value, (ca.SX, ca.MX)) and value.numel():
                    setattr(model, name, ca.substitute(value, old_x, node_lift))
            stage = {"_0": 0, "": min(1, self.horizon), "_e": self.horizon}[suffix]
            lo, hi = self.bounds(getattr(original.constraints, "lbx" + suffix),
                                 getattr(original.constraints, "ubx" + suffix), stage)
            setattr(reduced.constraints, "lbx" + suffix, lo)
            setattr(reduced.constraints, "ubx" + suffix, hi)
            setattr(reduced.constraints, "Jbx" + suffix, np.eye(self.keep.size))
            setattr(reduced.constraints, "idxbx" + suffix, np.arange(self.keep.size))
            setattr(reduced.dims, "nbx" + suffix, self.keep.size)
        reduced.dims.nx = self.keep.size
        reduced.parameter_values = np.concatenate((original.parameter_values, self.profiles[0]))
        model.name += "_dinglocal_v1"
        return reduced


class DingLocalAcadosSolver:
    """Full-state facade over a native reduced ACADOS capsule.

    Only x and node bounds are full-space views. All dual and flat-iterate APIs
    remain native reduced APIs, so a baseline capsule state cannot be restored.
    """

    def __init__(self, original, interface, factory, **kwargs):
        self.mapping = DingLocalMapping(interface)
        self.acados_ocp = original
        self.native_ocp = self.mapping.transform(original)
        self.native_solver = factory(self.native_ocp, **kwargs)
        self._lower = np.stack([original.constraints.lbx_0 if n == 0 else original.constraints.lbx_e
                              if n == self.mapping.horizon else original.constraints.lbx
                              for n in range(self.mapping.horizon + 1)]).copy()
        self._upper = np.stack([original.constraints.ubx_0 if n == 0 else original.constraints.ubx_e
                              if n == self.mapping.horizon else original.constraints.ubx
                              for n in range(self.mapping.horizon + 1)]).copy()
        self.summary = {"enabled": True, "nx_original": self.mapping.nx,
                        "nx_native": self.mapping.keep.size, "extra_parameters": self.mapping.extra_np,
                        "family": self.mapping.family, "stages": self.mapping.stages,
                        "substeps": self.mapping.substeps,
                        "duals": "native reduced capsule only",
                        "last_profile_preparation_seconds": self.mapping.preparation_seconds}

    def __getattr__(self, name):
        return getattr(self.native_solver, name)

    def refresh(self):
        self.mapping.refresh()
        # Profiles can change even when periodic_calcium itself stays identical.
        runtime = self.mapping.nlp.numerical_data_timeseries["periodic_calcium"][:, 0, :]
        for node in range(self.mapping.horizon + 1):
            self.set(node, "p", runtime[:, node])
        self.summary["last_profile_preparation_seconds"] = self.mapping.preparation_seconds

    def set(self, stage, field, value):
        values = np.asarray(value).reshape(-1)
        if field in ("x", "xdot"):
            if values.size != self.mapping.nx:
                raise ValueError("Ding capsule facade expects original-space x")
            value = values[self.mapping.keep]
        elif field == "p":
            if values.size != self.mapping.np:
                raise ValueError("Ding capsule facade expects original runtime parameters")
            value = np.concatenate((values, self.mapping.profiles[stage]))
        self.native_solver.set(stage, field, value)

    def set_params_sparse(self, stage, indices, values):
        indices = np.asarray(indices, dtype=int)
        if np.any(indices < 0) or np.any(indices >= self.mapping.np):
            raise ValueError("Sparse parameter updates must address the original runtime prefix")
        self.native_solver.set_params_sparse(stage, indices, values)

    def get(self, stage, field):
        value = self.native_solver.get(stage, field)
        if field == "x":
            return self.mapping.lift(value, stage)
        if field == "p":
            return value[:self.mapping.np]
        return value

    def constraints_set(self, stage, field, value, **kwargs):
        if field not in ("lbx", "ubx"):
            return self.native_solver.constraints_set(stage, field, value, **kwargs)
        (self._lower if field == "lbx" else self._upper)[stage] = np.asarray(value).reshape(-1)
        # Bioptim updates lower/upper separately. The final pair is applied when
        # initialization is complete, before diagnostics, warm starts and solve.

    def constraints_get(self, stage, field):
        if field == "lbx":
            return self._lower[stage].copy()
        if field == "ubx":
            return self._upper[stage].copy()
        return self.native_solver.constraints_get(stage, field)

    def get_iterate(self, iteration):
        native = self.native_solver.get_iterate(iteration)
        mapping = self.mapping

        class FullStateIterateView:
            # Retry/seed code reads x_traj in the original model dimensions.
            # A flattened primal-dual checkpoint stays an opaque native object.
            x_traj = [mapping.lift(x, node) for node, x in enumerate(native.x_traj)]

            def __getattr__(self, name):
                return getattr(native, name)

        return FullStateIterateView()

    def set_new_time_steps(self, *_args, **_kwargs):
        raise ValueError("Changing the mesh requires rebuilding the Ding local capsule")

    def flush_bounds(self):
        for node in range(self.mapping.horizon + 1):
            lower, upper = self.mapping.bounds(self._lower[node], self._upper[node], node)
            self.native_solver.constraints_set(node, "lbx", lower)
            self.native_solver.constraints_set(node, "ubx", upper)

    def solve(self):
        self.flush_bounds()
        return self.native_solver.solve()


def install_acados_ding_local_reduction():
    """Install an opt-in adapter without modifying the Bioptim dependency."""
    import bioptim.interfaces.acados_interface as module
    interface_type = module.AcadosInterface
    if getattr(interface_type, "_cocofest_ding_local_patch", False):
        return
    original_factory = module.AcadosOcpSolver
    original_initialize = interface_type.initialize_solver
    original_diagnostics = interface_type.get_diagnostics

    def factory(acados_ocp, **kwargs):
        interface = getattr(acados_ocp, "_cocofest_ding_local_interface", None)
        if interface is None:
            return original_factory(acados_ocp, **kwargs)
        # Do not deepcopy the owning OCP/interface into generated model metadata.
        del acados_ocp._cocofest_ding_local_interface
        return DingLocalAcadosSolver(acados_ocp, interface, original_factory, **kwargs)

    def initialize(self):
        enabled = getattr(self.ocp, "_cocofest_acados_ding_local_reduction", False)
        if enabled:
            if self.ocp_solver is None:
                self.acados_ocp._cocofest_ding_local_interface = self
            else:
                if not isinstance(self.ocp_solver, DingLocalAcadosSolver):
                    raise ValueError("Cannot enable Ding reduction on an existing full capsule")
                self.ocp_solver.refresh()
        result = original_initialize(self)
        if enabled:
            result.flush_bounds()
        return result

    def diagnostics(self):
        result = original_diagnostics(self)
        if isinstance(self.ocp_solver, DingLocalAcadosSolver):
            result["ding_local_reduction"] = dict(self.ocp_solver.summary)
        return result

    module.AcadosOcpSolver = factory
    interface_type.initialize_solver = initialize
    interface_type.get_diagnostics = diagnostics
    interface_type._cocofest_ding_local_patch = True
