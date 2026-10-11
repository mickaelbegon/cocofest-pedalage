"""Fixed-graph pulse-width tracking objective for offline causal branches."""
from __future__ import annotations
import math
from typing import Sequence
import numpy as np

PW_TRACKING_PARAMETER_KEY = "rho_pw_tracking_target"
PW_TRACKING_ACTIVATION_KEY = "rho_pw_tracking_activation"
PW_TRACKING_GRAPH_VERSION = "direct_pw_tracking_lagrange_v1"


def pw_tracking_parameter_options(binding, *, fatigue_weight_binding=None, use_sx=True):
    """Explicitly merge the supported PW-tracking/fatigue parameter pair."""
    from bioptim import BoundsList, InitialGuessList, InterpolationType, ParameterList, VariableScaling
    entries = [(PW_TRACKING_PARAMETER_KEY, binding.target.ravel()),
               (PW_TRACKING_ACTIVATION_KEY, np.asarray([binding.activation]))]
    if fatigue_weight_binding is not None:
        from .parametric_fatigue_weights import FATIGUE_WEIGHT_PARAMETER_KEY
        entries.insert(0, (FATIGUE_WEIGHT_PARAMETER_KEY, fatigue_weight_binding.weights))
    parameters, bounds, initial = ParameterList(use_sx=use_sx), BoundsList(), InitialGuessList()
    for key, value in entries:
        column = np.asarray(value, float).reshape((-1, 1))
        parameters.add(name=key, function=None, size=column.size, scaling=VariableScaling(key, np.ones(column.size)))
        bounds.add(key, min_bound=column.copy(), max_bound=column.copy(), interpolation=InterpolationType.CONSTANT)
        initial.add(key, initial_guess=column.copy())
    return {"parameters": parameters, "parameter_bounds": bounds, "parameter_init": initial}


class ParametricPulseWidthTrackingBinding:
    """Numerical PW target/activation buffers; zero activation is inert."""
    def __init__(self, muscle_names: Sequence[str], nodes_per_cycle: int, *, scale_s: float = 1e-4):
        self.muscle_names, self.nodes_per_cycle, self.scale_s = tuple(muscle_names), int(nodes_per_cycle), float(scale_s)
        if (not self.muscle_names or len(set(self.muscle_names)) != len(self.muscle_names) or self.nodes_per_cycle < 1
                or not math.isfinite(self.scale_s) or self.scale_s <= 0.):
            raise ValueError("PW tracking requires unique muscles, positive nodes and a positive scale.")
        self.target = np.zeros((len(self.muscle_names), self.nodes_per_cycle)); self.activation = 0.
        self._nlp = None
    @property
    def size(self): return int(self.target.size)
    def parameter_options(self, *, use_sx=True):
        from bioptim import BoundsList, InitialGuessList, InterpolationType, ParameterList, VariableScaling
        parameters, bounds, initial = ParameterList(use_sx=use_sx), BoundsList(), InitialGuessList()
        for key, value in ((PW_TRACKING_PARAMETER_KEY, self.target.ravel()),
                           (PW_TRACKING_ACTIVATION_KEY, np.asarray([self.activation]))):
            column = np.asarray(value, float).reshape((-1, 1))
            parameters.add(name=key, function=None, size=column.size, scaling=VariableScaling(key, np.ones(column.size)))
            bounds.add(key, min_bound=column.copy(), max_bound=column.copy(), interpolation=InterpolationType.CONSTANT)
            initial.add(key, initial_guess=column.copy())
        return {"parameters": parameters, "parameter_bounds": bounds, "parameter_init": initial}
    def update(self, *, target, activation: float):
        value, activation = np.asarray(target, float), float(activation)
        if value.shape != self.target.shape or not np.all(np.isfinite(value)):
            raise ValueError("PW tracking target has the wrong shape or nonfinite values.")
        if not math.isfinite(activation) or not 0. <= activation <= 1.:
            raise ValueError("PW tracking activation must lie in [0, 1].")
        self.target, self.activation = value.copy(), activation
        return self.summary()
    def attach(self, nmpc):
        if self._nlp is not None and self._nlp is not nmpc.nlp[0]:
            raise RuntimeError("PW tracking binding cannot attach to a rebuilt NLP.")
        self._nlp = nmpc.nlp[0]
    def update_nmpc(self, nmpc, *, target, activation: float):
        self.attach(nmpc); self.update(target=target, activation=activation)
        from bioptim import InitialGuessList
        values = {PW_TRACKING_PARAMETER_KEY: self.target.ravel()[:, None],
                  PW_TRACKING_ACTIVATION_KEY: np.asarray([[self.activation]])}
        initial = InitialGuessList()
        for key, value in values.items():
            bounds = nmpc.parameter_bounds[key]
            if bounds.min.shape != value.shape or bounds.max.shape != value.shape:
                raise ValueError("PW tracking parameter shape changed.")
            bounds.min[:, :], bounds.max[:, :] = value, value
            initial.add(key, initial_guess=value.copy())
        nmpc.update_initial_guess(parameter_init=initial)
        return self.summary()
    def objective(self, controller):
        from casadi import vertcat, sumsqr
        values = vertcat(*[controller.controls[f"last_pulse_width_{name}"].cx for name in self.muscle_names])
        target, activation = controller.parameters[PW_TRACKING_PARAMETER_KEY].cx, controller.parameters[PW_TRACKING_ACTIVATION_KEY].cx[0]
        node = int(controller.node_index)
        if (not 0 <= node < self.nodes_per_cycle or int(values.numel()) != len(self.muscle_names)
                or int(target.numel()) != self.size):
            raise ValueError("PW tracking control/parameter layout mismatch.")
        indexes = [muscle * self.nodes_per_cycle + node for muscle in range(len(self.muscle_names))]
        return activation * sumsqr((values - target[indexes]) / self.scale_s)
    def summary(self):
        return {"parameter_key": PW_TRACKING_PARAMETER_KEY, "activation_key": PW_TRACKING_ACTIVATION_KEY,
                "muscle_order": list(self.muscle_names), "nodes_per_cycle": self.nodes_per_cycle,
                "scale_s": self.scale_s, "activation": self.activation, "objective_graph_rebuild_required": False,
                "graph_version": PW_TRACKING_GRAPH_VERSION}


def migrate_prepared_checkpoint(source_path, destination_path, program, *, completed_cycles: int):
    """Migrate a legacy prepared RHO archive into the PW-extended graph.

    The source archive is fully verified and compared with the target runtime
    before the first mutation. Only the six PW target/activation arrays may be
    absent. All primal, physical-bound, fatigue-parameter and runtime arrays
    must match byte-for-byte after numerical conversion to float arrays.
    """
    from hashlib import sha256
    import json
    from pathlib import Path
    from cocofest.simulation.rho_restart_checkpoint import (
        _read_prepared_checkpoint, _runtime_state, export_prepared_checkpoint,
        prepared_problem_arrays, prepared_problem_digest, _restore_runtime_state,
    )
    from cocofest.optimization.receding_horizon_initial_guess import (
        initial_guess_signature, snapshot_initial_guess,
    )

    source_path, destination_path = Path(source_path).resolve(strict=True), Path(destination_path).resolve()
    if destination_path.exists():
        raise FileExistsError(destination_path)
    binding = getattr(program, "pw_tracking_binding", None)
    if binding is None or getattr(program, "fatigue_weight_binding", None) is None:
        raise ValueError("Checkpoint migration requires the PW tracking and fatigue bindings.")
    metadata, source_snapshot, source_arrays = _read_prepared_checkpoint(source_path, completed_cycles=completed_cycles)
    source_digest = prepared_problem_digest(source_snapshot, source_arrays, metadata.get("runtime_state", {}))
    if source_digest != metadata.get("prepared_problem_sha256"):
        raise ValueError("Source prepared checkpoint digest is invalid.")
    current_snapshot = snapshot_initial_guess(program)
    current_arrays = prepared_problem_arrays(program)
    new_keys = {
        f"parameter_bounds:{PW_TRACKING_PARAMETER_KEY}:min",
        f"parameter_bounds:{PW_TRACKING_PARAMETER_KEY}:max",
        f"parameter_bounds:{PW_TRACKING_ACTIVATION_KEY}:min",
        f"parameter_bounds:{PW_TRACKING_ACTIVATION_KEY}:max",
        f"parameter_init:{PW_TRACKING_PARAMETER_KEY}",
        f"parameter_init:{PW_TRACKING_ACTIVATION_KEY}",
    }
    if set(current_snapshot["states"]) != set(source_snapshot["states"]):
        raise ValueError("Checkpoint migration state layout mismatch.")
    if set(current_snapshot["controls"]) != set(source_snapshot["controls"]):
        raise ValueError("Checkpoint migration control layout mismatch.")
    for category in ("states", "controls"):
        for key, source in source_snapshot[category].items():
            if current_snapshot[category][key].shape != source.shape:
                raise ValueError(f"Checkpoint migration primal shape mismatch for {category}:{key}.")
    projected_current = {key: value for key, value in current_arrays.items() if key not in new_keys}
    if (set(projected_current) != set(source_arrays)
            or set(current_arrays) - set(source_arrays) != new_keys):
        raise ValueError("Checkpoint migration numerical-key mismatch outside PW parameters.")
    for key, value in source_arrays.items():
        if projected_current[key].shape != value.shape:
            raise ValueError(f"Checkpoint migration array shape mismatch for {key}.")
    source_runtime = metadata.get("runtime_state", {})
    if not isinstance(source_runtime, dict) or set(source_runtime) - set(_runtime_state(program)):
        raise ValueError("Checkpoint migration runtime state contains unsupported fields.")
    names = tuple(binding.muscle_names)
    if tuple(key.removeprefix("last_pulse_width_") for key in sorted(source_snapshot["controls"])
             if key.startswith("last_pulse_width_")) != tuple(sorted(names)):
        raise ValueError("Checkpoint migration direct-PW muscle layout mismatch.")
    if binding.nodes_per_cycle != next(iter(source_snapshot["controls"].values())).shape[1] and any(
            key.startswith("last_pulse_width_") and source_snapshot["controls"][key].shape[1] != binding.nodes_per_cycle
            for key in source_snapshot["controls"]):
        raise ValueError("Checkpoint migration PW node count mismatch.")
    # All possible compatibility failures have been checked. Now install the
    # exact source primal and old numerical arrays, then the canonical PW data.
    nlp = program.nlp[0]
    for key, value in source_snapshot["states"].items(): nlp.x_init[key].init[:, :] = value
    for key, value in source_snapshot["controls"].items(): nlp.u_init[key].init[:, :] = value
    for prefix, container in (("x_bounds", nlp.x_bounds), ("u_bounds", nlp.u_bounds),
                              ("parameter_bounds", program.parameter_bounds)):
        for key in container.keys():
            for field in ("min", "max"):
                name = f"{prefix}:{key}:{field}"
                if name in source_arrays:
                    getattr(container[key], field)[:, :] = source_arrays[name]
    for key, value in source_arrays.items():
        if key.startswith("parameter_init:"):
            program.parameter_init[key.split(":", 1)[1]].init[:, :] = value
    _restore_runtime_state(program, source_runtime)
    target = np.vstack([source_snapshot["controls"][f"last_pulse_width_{name}"] for name in names])
    if target.shape != binding.target.shape:
        raise ValueError("Canonical PW migration target shape mismatch.")
    binding.update_nmpc(program, target=target, activation=0.)
    for muscle_index, name in enumerate(names):
        if not np.array_equal(program.nlp[0].u_init[f"last_pulse_width_{name}"].init, target[muscle_index:muscle_index + 1]):
            raise RuntimeError("Canonical PW target migration altered the source control trace.")
    projected_digest = prepared_problem_digest(snapshot_initial_guess(program),
        {key: value for key, value in prepared_problem_arrays(program).items() if key not in new_keys},
        _runtime_state(program))
    if projected_digest != source_digest:
        raise RuntimeError("Removing the PW extension did not reproduce the source prepared digest.")
    graph_signature = {**binding.summary(), "fatigue_objective_variant": getattr(program, "fatigue_objective_variant", "legacy"),
                       "quadratic_callback": False, "node_scope": "all_shooting"}
    enriched = export_prepared_checkpoint(destination_path, program, completed_cycles=completed_cycles,
                                          model_path=Path(metadata.get("model_path", "")) if metadata.get("model_path") else None)
    receipt = {"schema_version": 1, "kind": "pw_tracking_checkpoint_migration", "source_archive": str(source_path),
              "source_archive_sha256": sha256(source_path.read_bytes()).hexdigest(),
              "source_prepared_problem_sha256": source_digest, "projected_back_to_source_sha256": projected_digest,
              "enriched_archive": enriched["primal_path"], "enriched_archive_sha256": enriched["sha256"],
              "enriched_prepared_problem_sha256": enriched["prepared_problem_sha256"],
              "enriched_primal_signature": enriched["prepared_primal_signature"],
              "completed_cycles": completed_cycles, "pw_target_source": "source_prepared_direct_pw_initial_guess",
              "pw_activation": 0.0, "pw_graph_signature": graph_signature,
              "migration_checks": {"only_six_pw_arrays_added": True, "old_primal_exact": True,
                                   "fatigue_parameters_exact": True, "runtime_clock_exact": True,
                                   "source_projection_digest_matches": True,
                                   "fresh_worker_replay_verified": False}}
    receipt_path = destination_path.with_suffix(".migration.json")
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
    return receipt
