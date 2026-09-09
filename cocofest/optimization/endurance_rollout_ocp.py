"""Opt-in Bioptim binding for a certified, fixed-size numerical rollout policy.

The input artifact contains already packed samples and force convolutions; its
representation may be Fourier or piecewise polynomial. This module never fits
or invents a policy. A passing adapter report is mandatory. The pinned Bioptim
substitutes numerical timeseries into NLP graphs, so this prototype uses fixed
ParameterList entries instead. Updates only change their equal bounds and
initial values, never OCP penalties. These entries increase the NLP dimension.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .ding_fatigue_rollout import DingFatigueParameters, ding_fatigue_parameters_from_model
from .endurance_rollout import DingRolloutMuscleParameters
from .endurance_rollout_objective import (
    ROLLOUT_DOMAIN_MARGIN_NAMES, RolloutObjectiveLayout, build_rollout_objective_function,
    rollout_domain_lower_bounds,
)


ROLLOUT_ARTIFACT_SCHEMA = "cocofest-certified-rollout-parameters-v1"
ROLLOUT_PARAMETER_KEY = "endurance_rollout_profile"


def _digest(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _require_complete_gate(report: dict) -> None:
    if (
        not isinstance(report, dict)
        or report.get("status") != "complete"
        or report.get("adapted_policy_fidelity", {}).get("passed") is not True
    ):
        raise ValueError(
            "endurance_rollout formulation_unavailable: adapted_policy_fidelity "
            "must pass and the adapter report must have status=complete; no fallback profile is generated."
        )
    source = report.get("source", {})
    if not isinstance(source.get("sha256"), str) or len(source["sha256"]) != 64:
        raise ValueError("The certified rollout report must identify its source by SHA-256.")


@dataclass(frozen=True)
class CertifiedRolloutParameters:
    layout: RolloutObjectiveLayout
    muscles: tuple[DingRolloutMuscleParameters, ...]
    muscle_names: tuple[str, ...]
    parameters: tuple[float, ...]
    provenance_json: str
    method: str

    @classmethod
    def _payload_from_adapter_report(cls, report: dict, *, horizon_cycles=5, temperature=0.02):
        _require_complete_gate(report)
        if report.get("schema") != "cocofest-rho-endurance-rollout-v2":
            raise ValueError("A complete adapter report of schema cocofest-rho-endurance-rollout-v2 is required.")
        packed = report.get("rollout_objective_profile", {})
        if (
            packed.get("schema") != "cocofest-rollout-objective-profile-v1"
            or packed.get("source_policy_gate_passed") is not True
            or packed.get("horizon_independent") is not True
        ):
            raise ValueError("A complete, horizon-independent packed rollout profile is required.")
        layout = RolloutObjectiveLayout(
            packed["muscle_count"], packed["interval_count"], horizon_cycles, temperature
        )
        if packed.get("parameter_size") != layout.parameter_size:
            raise ValueError("Adapter packed parameter_size does not match its declared dimensions.")
        muscles = []
        for item in report["model_parameters"]:
            muscles.append({
                "fatigue": {
                    "a_rest": item["A_rest"], "tau1_rest": item["Tau1_rest"], "km_rest": item["Km_rest"],
                    "alpha_a": item["alpha_A"], "alpha_tau1": item["alpha_Tau1"], "alpha_km": item["alpha_Km"],
                    "tau_fat": item["tau_fat"],
                },
                **{key: item[key] for key in ("tau2", "pd0", "pdt", "pulse_width_max")},
            })
        if [item["muscle"] for item in report["model_parameters"]] != report["muscle_names"]:
            raise ValueError("Adapter model parameter order differs from its muscle names.")
        return {
            "schema": ROLLOUT_ARTIFACT_SCHEMA, "adapter_report": report,
            "layout": asdict(layout), "muscles": muscles, "muscle_names": report["muscle_names"],
            "profile_parameters": packed["parameters"],
            "method": report["configuration"]["policy_representation"],
        }

    @classmethod
    def from_adapter_report(cls, report: dict, *, horizon_cycles=5, temperature=0.02):
        """Ingest the adapter's existing packed vector, without reconstructing it."""
        return cls.from_payload(cls._payload_from_adapter_report(
            report, horizon_cycles=horizon_cycles, temperature=temperature
        ))

    def __post_init__(self):
        """Bind all executable data to the same certified adapter report."""
        canonical = self._payload_from_adapter_report(
            json.loads(self.provenance_json), horizon_cycles=self.layout.horizon_cycles,
            temperature=self.layout.smooth_max_temperature,
        )
        if (
            asdict(self.layout) != canonical["layout"]
            or [asdict(muscle) for muscle in self.muscles] != canonical["muscles"]
            or list(self.muscle_names) != canonical["muscle_names"]
            or self.method != canonical["method"]
            or not np.array_equal(np.asarray(self.parameters), np.asarray(canonical["profile_parameters"]))
        ):
            raise ValueError("The executable rollout profile, model, method and dimensions must match the certified report.")

    @classmethod
    def from_payload(cls, payload: dict):
        if payload.get("schema") != ROLLOUT_ARTIFACT_SCHEMA:
            raise ValueError(f"Expected rollout artifact schema {ROLLOUT_ARTIFACT_SCHEMA}.")
        report = payload.get("adapter_report")
        _require_complete_gate(report)
        layout = RolloutObjectiveLayout(**payload["layout"])
        muscles = tuple(
            DingRolloutMuscleParameters(
                fatigue=DingFatigueParameters(**entry["fatigue"]),
                **{key: value for key, value in entry.items() if key != "fatigue"},
            )
            for entry in payload["muscles"]
        )
        names = tuple(payload["muscle_names"])
        if any(muscle.pdt <= 0.0 or muscle.tau2 < 0.0 or muscle.pulse_width_max <= muscle.pd0 for muscle in muscles):
            raise ValueError("The rollout needs positive pdt, nonnegative tau2 and PW_max > pd0.")
        if (
            len(muscles) != layout.muscle_count
            or len(names) != layout.muscle_count
            or len(set(names)) != len(names)
            or list(names) != report.get("muscle_names")
        ):
            raise ValueError("Packed rollout muscle names/order must match the certified adapter report.")
        values = np.asarray(payload["profile_parameters"], dtype=float)
        if values.shape != (layout.parameter_size,) or not np.all(np.isfinite(values)):
            raise ValueError("The packed rollout profile must be a finite fixed-size vector.")
        for field in ("force", "first_half_force_integral", "second_half_force_integral"):
            if np.any(values[layout.field_slice(field)] < 0.0):
                raise ValueError(f"Packed rollout {field} must be non-negative.")
        if np.any(values[layout.field_slice("cn")] <= 0.0):
            raise ValueError("Packed rollout cn must be positive.")
        gain = (
            values[layout.field_slice("force_length")] * values[layout.field_slice("force_velocity")]
            + values[layout.field_slice("passive_force")]
        )
        decay = values[layout.field_slice("half_decay")]
        if np.any(gain <= 0.0) or np.any(decay <= 0.0) or np.any(decay > 1.0):
            raise ValueError("Packed rollout requires positive mechanical gain and decay in (0, 1].")
        method = payload.get("method")
        if not isinstance(method, str) or not method.strip():
            raise ValueError("The packed rollout artifact must name its reconstruction method.")
        return cls(layout, muscles, names, tuple(values), json.dumps(report, sort_keys=True), method)

    @property
    def structure(self) -> dict:
        return {
            "schema": ROLLOUT_ARTIFACT_SCHEMA,
            "layout": asdict(self.layout),
            "muscles": [asdict(muscle) for muscle in self.muscles],
            "muscle_names": list(self.muscle_names),
            "objective": "Mayer.END.normalized_smooth_maximum_utilization.nonquadratic",
            "runtime_parameter": ROLLOUT_PARAMETER_KEY,
            "binding": "Bioptim.ParameterList.fixed_equal_bounds",
            "additional_fixed_parameters": self.layout.parameter_size,
            "policy_reconstruction_method": self.method,
        }

    @property
    def provenance(self) -> dict:
        report = json.loads(self.provenance_json)
        return {
            "method": self.method,
            "source": report["source"],
            "adapter_report_sha256": _digest(report),
            "profile_parameters_sha256": _digest(self.parameters),
            "adapted_policy_fidelity": report["adapted_policy_fidelity"],
            "adapter_status": report["status"],
            "source_ocp_context": self.source_context,
            "context_metadata_status": {
                key: "present" if value is not None else "not_verifiable"
                for key, value in self.source_context.items()
            },
        }

    @property
    def source_context(self) -> dict:
        report = json.loads(self.provenance_json)
        context = report.get("source_ocp_context", {})
        return {
            "cycle_period_s": report.get("selection", {}).get("cycle_period_s"),
            "signed_crank_torque_nm": context.get("signed_crank_torque_nm"),
            "mechanical_formulation": context.get("mechanical_formulation"),
            "formulation": context.get("formulation"),
        }

    def validate_context(self, **ocp_context) -> dict:
        audit = {}
        for key, source_value in self.source_context.items():
            target_value = ocp_context.get(key)
            if source_value is None or target_value is None:
                status = "not_verifiable"
            else:
                matches = (
                    np.isfinite(float(source_value)) and np.isfinite(float(target_value))
                    and np.isclose(float(source_value), float(target_value), rtol=0.0, atol=1e-12)
                    if key in ("cycle_period_s", "signed_crank_torque_nm")
                    else source_value == target_value
                )
                if not matches:
                    raise ValueError(f"Rollout/OCP context mismatch for {key}: source={source_value}, OCP={target_value}.")
                status = "matched"
            audit[key] = {"status": status, "source": source_value, "ocp": target_value}
        return audit

    def validate_model(self, model, *, control_bounds) -> None:
        models = model.muscles_dynamics_model
        if tuple(item.muscle_name for item in models) != self.muscle_names:
            raise ValueError("The rollout and OCP must use identical muscle order.")
        for packed, current in zip(self.muscles, models):
            upper = np.asarray(control_bounds[f"last_pulse_width_{current.muscle_name}"].max, dtype=float)
            if not np.all(np.isfinite(upper)) or not np.all(upper == upper.flat[0]):
                raise ValueError("The rollout requires phase-independent pulse-width upper bounds.")
            expected = DingRolloutMuscleParameters(
                fatigue=ding_fatigue_parameters_from_model(current),
                tau2=float(current.tau2), pd0=float(current.pd0), pdt=float(current.pdt),
                pulse_width_max=float(upper.flat[0]),
            )
            if packed != expected:
                raise ValueError("The rollout and OCP must use identical Ding parameters and pulse-width bounds.")


@dataclass(frozen=True)
class EnduranceRolloutOptions:
    profile: CertifiedRolloutParameters
    weight: float
    domain_epsilon: float

    def metadata(self, *, structure_only=False) -> dict:
        payload = {
            **self.profile.structure,
            "weight": self.weight,
            "effective_weight": 10000.0 * self.weight,
            "domain_epsilon": self.domain_epsilon,
            "domain_lower_bound_policy": "closed_domains_zero_strict_domains_epsilon_v1",
            "status": "experimental_structure_only",
            "scientific_validation": "not_run",
        }
        if not structure_only:
            payload["provenance"] = self.profile.provenance
        return payload


def resolve_endurance_rollout_options(args) -> EnduranceRolloutOptions | None:
    weight = float(getattr(args, "endurance_rollout_weight", 0.0))
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("endurance_rollout_weight must be finite and non-negative.")
    # No file access, graph construction, parameter or constraint at zero.
    if weight == 0.0:
        return None
    if not getattr(args, "experimental_endurance_rollout", False):
        raise ValueError("Nonzero rollout weight requires --experimental-endurance-rollout.")
    epsilon = float(getattr(args, "endurance_rollout_domain_epsilon", 1e-8))
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("endurance_rollout_domain_epsilon must be finite and positive.")
    path = getattr(args, "endurance_rollout_profile", None)
    if path is None:
        raise ValueError("endurance_rollout formulation_unavailable: a certified packed profile is required.")
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != "cocofest-rho-endurance-rollout-v2":
        raise ValueError("The CLI requires a complete adapter report, not an independent packed payload.")
    profile = CertifiedRolloutParameters.from_adapter_report(
        payload, horizon_cycles=getattr(args, "endurance_rollout_horizon_cycles", 5),
        temperature=getattr(args, "endurance_rollout_temperature", 0.02),
    )
    return EnduranceRolloutOptions(profile, weight, epsilon)


def endurance_rollout_signature_fields(args, *, structure_only=False) -> dict:
    options = getattr(args, "_endurance_rollout_options", None)
    if options is None:
        options = resolve_endurance_rollout_options(args)
    return {} if options is None else {"endurance_rollout": options.metadata(structure_only=structure_only)}


def add_endurance_rollout_cli(parser) -> None:
    parser.add_argument("--experimental-endurance-rollout", action="store_true",
                        help="Enable the structural rollout binding; requires a complete fidelity gate.")
    parser.add_argument("--endurance-rollout-weight", type=float, default=0.0,
                        help="Dimensionless terminal rollout weight; zero removes the entire binding.")
    parser.add_argument("--endurance-rollout-profile", type=Path, default=None,
                        help="Certified fixed-size packed-policy JSON; no synthetic fallback is allowed.")
    parser.add_argument("--endurance-rollout-domain-epsilon", type=float, default=1e-8,
                        help="Positive lower bound only on strict rollout domains; closed domains retain zero.")
    parser.add_argument("--endurance-rollout-horizon-cycles", type=int, default=5)
    parser.add_argument("--endurance-rollout-temperature", type=float, default=0.02)


class EnduranceRolloutBinding:
    """One objective graph and an updater for fixed numerical parameter bounds."""

    def __init__(self, options: EnduranceRolloutOptions, *, use_sx=True):
        self.options = options
        self.function = build_rollout_objective_function(
            muscles=options.profile.muscles, layout=options.profile.layout,
            symbolic_type="SX" if use_sx else "MX",
        )
        self.build_count = 1
        self.update_count = 0
        self._nlp = None
        self._solver = None
        self.solver_observation_count = 0
        self._ocp_context = {}
        self.context_compatibility = options.profile.validate_context()

    @property
    def domain_lower_bounds(self) -> np.ndarray:
        bounds = rollout_domain_lower_bounds(self.options.profile.layout, self.options.domain_epsilon)
        if self.function.size1_out("domain_margins") != bounds.size:
            raise RuntimeError("Rollout domain bounds do not match the symbolic margin count.")
        return bounds

    def validate_context(self, **ocp_context) -> None:
        audit = self.options.profile.validate_context(**ocp_context)
        self._ocp_context = dict(ocp_context)
        self.context_compatibility = audit

    def parameter_options(self, *, use_sx=True) -> dict:
        from bioptim import BoundsList, InitialGuessList, InterpolationType, ParameterList, VariableScaling

        profile = self.options.profile
        parameters = ParameterList(use_sx=use_sx)
        parameters.add(
            name=ROLLOUT_PARAMETER_KEY, function=None, size=profile.layout.parameter_size,
            scaling=VariableScaling(ROLLOUT_PARAMETER_KEY, np.ones(profile.layout.parameter_size)),
        )
        values = np.asarray(profile.parameters)[:, None]
        bounds, initial = BoundsList(), InitialGuessList()
        bounds.add(ROLLOUT_PARAMETER_KEY, min_bound=values, max_bound=values,
                   interpolation=InterpolationType.CONSTANT)
        initial.add(ROLLOUT_PARAMETER_KEY, initial_guess=values)
        return {"parameters": parameters, "parameter_bounds": bounds, "parameter_init": initial}

    def attach(self, nmpc) -> None:
        if self._nlp is not None and self._nlp is not nmpc.nlp[0]:
            raise RuntimeError("Rollout binding cannot be attached to a rebuilt NLP.")
        self._nlp = nmpc.nlp[0]

    def update(self, nmpc, profile: CertifiedRolloutParameters) -> None:
        self.attach(nmpc)
        context_compatibility = profile.validate_context(**self._ocp_context)
        if profile.structure != self.options.profile.structure:
            raise ValueError("A rollout profile update cannot change the compiled graph dimensions or model.")
        self.observe_solver(nmpc, record_observation=False)
        bounds = nmpc.parameter_bounds[ROLLOUT_PARAMETER_KEY]
        new_values = np.asarray(profile.parameters)[:, None]
        if bounds.min.shape != new_values.shape or bounds.max.shape != new_values.shape:
            raise ValueError("Rollout parameter dimensions cannot change between windows.")
        bounds.min[...] = new_values
        bounds.max[...] = new_values
        # RHO may have initialized parameters with a view of the previous
        # Solution's array. Replacing this buffer preserves certified history.
        from bioptim import InitialGuessList

        parameter_init = InitialGuessList()
        parameter_init.add(ROLLOUT_PARAMETER_KEY, initial_guess=new_values.copy())
        nmpc.update_initial_guess(parameter_init=parameter_init)
        self.update_count += 1
        self.options = EnduranceRolloutOptions(profile, self.options.weight, self.options.domain_epsilon)
        self.context_compatibility = context_compatibility

    def observe_solver(self, nmpc, *, record_observation=True) -> None:
        self.attach(nmpc)
        interface = getattr(nmpc, "ocp_solver", None)
        solver = getattr(interface, "shaked_ocp_solver", None)
        if solver is None:
            solver = getattr(interface, "acados_solver", None)
        if solver is None:
            return
        if self._solver is not None and solver is not self._solver:
            raise RuntimeError("Compiled NLP solver changed between rollout windows.")
        self._solver = solver
        if record_observation:
            self.solver_observation_count += 1

    def summary(self) -> dict:
        return {**self.options.metadata(), "objective_graph_build_count": self.build_count,
                "ocp_context_compatibility": self.context_compatibility,
                "domain_margin_names_per_sample": list(ROLLOUT_DOMAIN_MARGIN_NAMES),
                "domain_lower_bounds_per_sample": self.domain_lower_bounds[:len(ROLLOUT_DOMAIN_MARGIN_NAMES)].tolist(),
                "profile_update_count": self.update_count,
                "compiled_solver_observation_count": self.solver_observation_count,
                "compiled_solver_build_count": int(self._solver is not None),
                "compiled_solver_reuse_verified": self.solver_observation_count > 1}
