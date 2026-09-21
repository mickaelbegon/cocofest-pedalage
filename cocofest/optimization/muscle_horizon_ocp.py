"""Opt-in Bioptim binding for the differentiable muscle-only horizon."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import math
from pathlib import Path

import numpy as np

from .adaptive_moment_rollout import DingPulseWidthParameters
from .ding_fatigue_rollout import ding_fatigue_parameters_from_model
from .muscle_horizon_objective import (
    MUSCLE_HORIZON_DOMAIN_MARGIN_NAMES,
    MuscleHorizonLayout,
    MuscleHorizonPackedProfile,
    build_muscle_horizon_function,
    pack_muscle_horizon_profile,
)
from .rho_adaptive_moment_policy import (
    RhoAdaptiveMomentPolicy,
    build_rho_adaptive_moment_policy,
)


MUSCLE_HORIZON_PROFILE_KEY = "muscle_horizon_profile"
MUSCLE_HORIZON_FUTURE_PW_KEY = "muscle_horizon_future_pulse_widths"
MUSCLE_HORIZON_FUTURE_PW_SCALING = 1e-3
_STRICT_DOMAIN_INDICES = (0, 2, 6, 7, 8)


def _array_digest(values) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    digest = sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CertifiedMuscleHorizonProfile:
    layout: MuscleHorizonLayout
    muscles: tuple[DingPulseWidthParameters, ...]
    muscle_names: tuple[str, ...]
    packed_parameters: tuple[float, ...]
    future_pulse_width_seed: tuple[float, ...]
    certification_basis: str
    source_cycle_index: int
    source_bound_projection_count: int
    source_bound_projection_maximum_absolute_s: float
    source_cycle_period_s: float
    source_signed_crank_torque_nm: float | None
    source_mechanical_formulation: str | None
    source_formulation: str | None

    @classmethod
    def from_rho_policy(
        cls,
        policy: RhoAdaptiveMomentPolicy,
        *,
        horizon_cycles: int = 3,
        integration_substeps: int = 1,
        smooth_max_temperature: float = 0.02,
        allocation_weight: float = 0.05,
        source_bound_tolerance_s: float = 1e-10,
    ) -> "CertifiedMuscleHorizonProfile":
        layout = MuscleHorizonLayout(
            muscle_count=len(policy.muscle_names),
            interval_count=len(policy.intervals),
            horizon_cycles=horizon_cycles,
            integration_substeps=integration_substeps,
            smooth_max_temperature=smooth_max_temperature,
            allocation_weight=allocation_weight,
        )
        bounded_source = np.asarray(policy.source_pulse_widths, dtype=float).copy()
        for muscle_index, muscle in enumerate(policy.parameters):
            bounded_source[muscle_index] = np.clip(
                bounded_source[muscle_index], muscle.pd0, muscle.pulse_width_max
            )
        projection = bounded_source - policy.source_pulse_widths
        maximum_projection = float(np.max(np.abs(projection)))
        if maximum_projection > float(source_bound_tolerance_s):
            raise ValueError(
                "The source PW seed violates the instantiated Ding bounds beyond tolerance."
            )
        packed = pack_muscle_horizon_profile(policy.intervals, bounded_source, layout)
        return cls(
            layout=layout,
            muscles=tuple(policy.parameters),
            muscle_names=tuple(policy.muscle_names),
            packed_parameters=tuple(packed.parameters),
            future_pulse_width_seed=tuple(packed.future_pulse_width_seed),
            certification_basis=policy.certification_basis,
            source_cycle_index=policy.source_cycle_index,
            source_bound_projection_count=int(np.count_nonzero(projection)),
            source_bound_projection_maximum_absolute_s=maximum_projection,
            source_cycle_period_s=float(policy.period),
            source_signed_crank_torque_nm=(
                None
                if policy.source_signed_crank_torque_nm is None
                else float(policy.source_signed_crank_torque_nm)
            ),
            source_mechanical_formulation=policy.source_mechanical_formulation,
            source_formulation=policy.source_formulation,
        )

    def __post_init__(self) -> None:
        if len(self.muscles) != self.layout.muscle_count or len(self.muscle_names) != self.layout.muscle_count:
            raise ValueError("Muscles and names must match the horizon layout.")
        if len(set(self.muscle_names)) != len(self.muscle_names):
            raise ValueError("Muscle names must be unique.")
        packed = np.asarray(self.packed_parameters, dtype=float)
        seed = np.asarray(self.future_pulse_width_seed, dtype=float)
        if packed.shape != (self.layout.parameter_size,) or not np.all(np.isfinite(packed)):
            raise ValueError("packed_parameters do not match the horizon layout.")
        if seed.shape != (self.layout.future_pulse_width_size,) or not np.all(np.isfinite(seed)):
            raise ValueError("future_pulse_width_seed does not match the horizon layout.")
        for cycle in range(self.layout.horizon_cycles):
            for muscle_index, muscle in enumerate(self.muscles):
                start = cycle * self.layout.phase_size + muscle_index * self.layout.interval_count
                values = seed[start : start + self.layout.interval_count]
                if np.any(values < muscle.pd0) or np.any(values > muscle.pulse_width_max):
                    raise ValueError("future_pulse_width_seed lies outside the Ding bounds.")
        if not isinstance(self.certification_basis, str) or not self.certification_basis:
            raise ValueError("A non-empty RHO certification basis is required.")
        if not math.isfinite(self.source_cycle_period_s) or self.source_cycle_period_s <= 0.0:
            raise ValueError("The source RHO cycle period must be finite and positive.")

    @property
    def packed(self) -> MuscleHorizonPackedProfile:
        return MuscleHorizonPackedProfile(
            parameters=np.asarray(self.packed_parameters),
            future_pulse_width_seed=np.asarray(self.future_pulse_width_seed),
        )

    @property
    def structure(self) -> dict:
        return {
            "method": "outer_nlp_future_pw_full_ding_total_moment_constraints",
            "muscle_count": self.layout.muscle_count,
            "interval_count": self.layout.interval_count,
            "horizon_cycles": self.layout.horizon_cycles,
            "integration_substeps": self.layout.integration_substeps,
            "smooth_max_temperature": self.layout.smooth_max_temperature,
            "allocation_weight": self.layout.allocation_weight,
            "fixed_profile_parameters": self.layout.parameter_size,
            "free_future_pw_parameters": self.layout.future_pulse_width_size,
            "total_moment_constraints": self.layout.total_moment_constraint_size,
            "domain_constraints": self.layout.domain_margin_size,
            "muscle_names": list(self.muscle_names),
            "ding_parameters": [asdict(muscle) for muscle in self.muscles],
        }

    def validate_model(self, model, *, control_bounds) -> None:
        models = model.muscles_dynamics_model
        if tuple(str(item.muscle_name) for item in models) != self.muscle_names:
            raise ValueError("The muscle horizon and OCP must use identical muscle order.")
        for expected, current in zip(self.muscles, models, strict=True):
            upper = np.asarray(
                control_bounds[f"last_pulse_width_{current.muscle_name}"].max,
                dtype=float,
            )
            if not np.all(np.isfinite(upper)) or not np.all(upper == upper.flat[0]):
                raise ValueError("The muscle horizon requires phase-independent PW upper bounds.")
            observed = DingPulseWidthParameters(
                fatigue=ding_fatigue_parameters_from_model(current),
                tauc=float(current.tauc),
                tau2=float(current.tau2),
                pd0=float(current.pd0),
                pdt=float(current.pdt),
                pulse_width_max=float(upper.flat[0]),
            )
            if observed != expected:
                raise ValueError("The muscle horizon and OCP use different Ding parameters or PW bounds.")

    def validate_context(self, **ocp_context) -> dict:
        source_context = {
            "cycle_period_s": self.source_cycle_period_s,
            "signed_crank_torque_nm": self.source_signed_crank_torque_nm,
            "mechanical_formulation": self.source_mechanical_formulation,
            "formulation": self.source_formulation,
        }
        audit = {}
        for key, source_value in source_context.items():
            target_value = ocp_context.get(key)
            if source_value is None or target_value is None:
                status = "not_verifiable"
            elif key in ("cycle_period_s", "signed_crank_torque_nm"):
                if not np.isclose(
                    float(source_value), float(target_value), rtol=0.0, atol=1e-12
                ):
                    raise ValueError(
                        f"Muscle-horizon/OCP context mismatch for {key}: "
                        f"source={source_value}, OCP={target_value}."
                    )
                status = "matched"
            else:
                if source_value != target_value:
                    raise ValueError(
                        f"Muscle-horizon/OCP context mismatch for {key}: "
                        f"source={source_value}, OCP={target_value}."
                    )
                status = "matched"
            audit[key] = {
                "status": status,
                "source": source_value,
                "ocp": target_value,
            }
        return audit


@dataclass(frozen=True)
class MuscleHorizonOptions:
    profile: CertifiedMuscleHorizonProfile
    weight: float
    domain_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        weight = float(self.weight)
        epsilon = float(self.domain_epsilon)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("weight must be finite and non-negative.")
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("domain_epsilon must be finite and positive.")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "domain_epsilon", epsilon)

    def metadata(self, *, structure_only: bool = False) -> dict:
        payload = {
            **self.profile.structure,
            "weight": self.weight,
            "effective_weight": 10000.0 * self.weight,
            "domain_epsilon": self.domain_epsilon,
            "domain_lower_bound_policy": "closed_domains_zero_strict_domains_epsilon_v1",
            "status": "experimental_not_yet_prospectively_validated",
        }
        if not structure_only:
            payload.update(
                {
                    "packed_parameters_sha256": _array_digest(
                        self.profile.packed_parameters
                    ),
                    "future_pulse_width_seed_sha256": _array_digest(
                        self.profile.future_pulse_width_seed
                    ),
                    "source_cycle_index": self.profile.source_cycle_index,
                    "certification_basis": self.profile.certification_basis,
                    "source_context": {
                        "cycle_period_s": self.profile.source_cycle_period_s,
                        "signed_crank_torque_nm": (
                            self.profile.source_signed_crank_torque_nm
                        ),
                        "mechanical_formulation": (
                            self.profile.source_mechanical_formulation
                        ),
                        "formulation": self.profile.source_formulation,
                    },
                    "source_bound_projection_count": (
                        self.profile.source_bound_projection_count
                    ),
                    "source_bound_projection_maximum_absolute_s": (
                        self.profile.source_bound_projection_maximum_absolute_s
                    ),
                }
            )
        return payload


def _resolved_cycle_period(args) -> float:
    override = getattr(args, "muscle_horizon_cycle_period", None)
    if override is not None:
        period = float(override)
    elif getattr(args, "formulation", "dynamic") == "isokinetic":
        omega = abs(float(getattr(args, "isokinetic_omega", -2.0 * np.pi)))
        if not math.isfinite(omega) or omega <= 0.0:
            raise ValueError("isokinetic_omega must be finite and nonzero.")
        period = 2.0 * np.pi / omega
    else:
        period = 1.0
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError("muscle_horizon_cycle_period must be finite and positive.")
    return period


def resolve_muscle_horizon_options(args) -> MuscleHorizonOptions | None:
    """Build the fixed-size differentiable horizon from one certified RHO cycle."""

    weight = float(getattr(args, "muscle_horizon_weight", 0.0))
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("muscle_horizon_weight must be finite and non-negative.")
    # Preserve the historical NLP exactly: no files, graph, parameters or constraints.
    if weight == 0.0:
        return None
    if not getattr(args, "experimental_muscle_horizon", False):
        raise ValueError("Nonzero muscle horizon weight requires --experimental-muscle-horizon.")
    if getattr(args, "solver", "ipopt") != "ipopt":
        raise ValueError(
            "The experimental muscle horizon is currently certified only with --solver ipopt."
        )
    source = getattr(args, "muscle_horizon_source", None)
    if source is None:
        raise ValueError("--muscle-horizon-source must identify a certified compact RHO NPZ.")
    reduced_profile = getattr(args, "muscle_horizon_reduced_profile", None)
    if reduced_profile is None:
        reduced_profile = getattr(args, "reduced_cycling_profile", None)
    if reduced_profile is None:
        raise ValueError(
            "Provide --muscle-horizon-reduced-profile or --reduced-cycling-profile; "
            "automatic profile generation occurs too late to certify this objective."
        )
    policy = build_rho_adaptive_moment_policy(
        Path(source),
        Path(reduced_profile),
        cycle_index=int(getattr(args, "muscle_horizon_source_cycle_index", 0)),
        cycle_period=_resolved_cycle_period(args),
    )
    profile = CertifiedMuscleHorizonProfile.from_rho_policy(
        policy,
        horizon_cycles=int(getattr(args, "muscle_horizon_cycles", 3)),
        integration_substeps=int(
            getattr(args, "muscle_horizon_integration_substeps", 1)
        ),
        smooth_max_temperature=float(
            getattr(args, "muscle_horizon_temperature", 0.02)
        ),
        allocation_weight=float(
            getattr(args, "muscle_horizon_allocation_weight", 0.05)
        ),
    )
    return MuscleHorizonOptions(
        profile=profile,
        weight=weight,
        domain_epsilon=float(getattr(args, "muscle_horizon_domain_epsilon", 1e-8)),
    )


def muscle_horizon_signature_fields(args, *, structure_only: bool = False) -> dict:
    options = getattr(args, "_muscle_horizon_options", None)
    if options is None:
        options = resolve_muscle_horizon_options(args)
    return (
        {}
        if options is None
        else {"muscle_horizon": options.metadata(structure_only=structure_only)}
    )


def add_muscle_horizon_cli(parser) -> None:
    parser.add_argument(
        "--experimental-muscle-horizon",
        action="store_true",
        help=(
            "Enable differentiable future muscle-PW allocation inside the IPOPT NLP."
        ),
    )
    parser.add_argument(
        "--muscle-horizon-weight",
        type=float,
        default=0.0,
        help="Dimensionless terminal horizon weight; zero removes the binding exactly.",
    )
    parser.add_argument(
        "--muscle-horizon-source",
        type=Path,
        default=None,
        help="Certified compact RHO NPZ providing one reference cycle.",
    )
    parser.add_argument(
        "--muscle-horizon-reduced-profile",
        type=Path,
        default=None,
        help=(
            "Reduced cycling NPZ used to reconstruct phase-dependent muscle moments; "
            "defaults to --reduced-cycling-profile."
        ),
    )
    parser.add_argument("--muscle-horizon-source-cycle-index", type=int, default=0)
    parser.add_argument(
        "--muscle-horizon-cycle-period",
        type=float,
        default=None,
        help="Cycle period in seconds; inferred from the active formulation when omitted.",
    )
    parser.add_argument("--muscle-horizon-cycles", type=int, default=3)
    parser.add_argument("--muscle-horizon-integration-substeps", type=int, default=1)
    parser.add_argument("--muscle-horizon-temperature", type=float, default=0.02)
    parser.add_argument("--muscle-horizon-allocation-weight", type=float, default=0.05)
    parser.add_argument("--muscle-horizon-domain-epsilon", type=float, default=1e-8)


class MuscleHorizonBinding:
    """Bioptim parameters, objective and constraints for one fixed graph."""

    def __init__(self, options: MuscleHorizonOptions, *, use_sx: bool = True):
        self.options = options
        self.function = build_muscle_horizon_function(
            muscles=options.profile.muscles,
            layout=options.profile.layout,
            symbolic_type="SX" if use_sx else "MX",
        )
        self._nlp = None
        self._solver = None
        self.build_count = 1
        self.update_count = 0
        self.solver_observation_count = 0
        self._ocp_context = {}
        self.context_compatibility = options.profile.validate_context()

    @property
    def total_moment_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        zeros = np.zeros(self.options.profile.layout.total_moment_constraint_size)
        return zeros.copy(), zeros.copy()

    @property
    def future_pulse_width_scaling(self) -> float:
        return MUSCLE_HORIZON_FUTURE_PW_SCALING

    @property
    def domain_lower_bounds(self) -> np.ndarray:
        per_sample = np.zeros(len(MUSCLE_HORIZON_DOMAIN_MARGIN_NAMES))
        per_sample[list(_STRICT_DOMAIN_INDICES)] = self.options.domain_epsilon
        return np.tile(per_sample, self.options.profile.layout.future_pulse_width_size)

    def parameter_options(self, *, use_sx: bool = True) -> dict:
        from bioptim import BoundsList, InitialGuessList, InterpolationType, ParameterList, VariableScaling

        profile = self.options.profile
        layout = profile.layout
        parameters = ParameterList(use_sx=use_sx)
        parameters.add(
            name=MUSCLE_HORIZON_PROFILE_KEY,
            function=None,
            size=layout.parameter_size,
            scaling=VariableScaling(MUSCLE_HORIZON_PROFILE_KEY, np.ones(layout.parameter_size)),
        )
        parameters.add(
            name=MUSCLE_HORIZON_FUTURE_PW_KEY,
            function=None,
            size=layout.future_pulse_width_size,
            scaling=VariableScaling(
                MUSCLE_HORIZON_FUTURE_PW_KEY,
                np.full(
                    layout.future_pulse_width_size,
                    MUSCLE_HORIZON_FUTURE_PW_SCALING,
                ),
            ),
        )
        fixed = np.asarray(profile.packed_parameters)[:, None]
        lower = np.empty(layout.future_pulse_width_size)
        upper = np.empty_like(lower)
        for cycle in range(layout.horizon_cycles):
            for muscle_index, muscle in enumerate(profile.muscles):
                start = cycle * layout.phase_size + muscle_index * layout.interval_count
                lower[start : start + layout.interval_count] = muscle.pd0
                upper[start : start + layout.interval_count] = muscle.pulse_width_max
        bounds = BoundsList()
        bounds.add(
            MUSCLE_HORIZON_PROFILE_KEY,
            min_bound=fixed,
            max_bound=fixed,
            interpolation=InterpolationType.CONSTANT,
        )
        bounds.add(
            MUSCLE_HORIZON_FUTURE_PW_KEY,
            min_bound=lower[:, None],
            max_bound=upper[:, None],
            interpolation=InterpolationType.CONSTANT,
        )
        initial = InitialGuessList()
        initial.add(MUSCLE_HORIZON_PROFILE_KEY, initial_guess=fixed)
        initial.add(
            MUSCLE_HORIZON_FUTURE_PW_KEY,
            initial_guess=np.asarray(profile.future_pulse_width_seed)[:, None],
        )
        return {
            "parameters": parameters,
            "parameter_bounds": bounds,
            "parameter_init": initial,
        }

    def validate_context(self, **ocp_context) -> None:
        self._ocp_context = dict(ocp_context)
        self.context_compatibility = self.options.profile.validate_context(**ocp_context)

    def attach(self, nmpc) -> None:
        if self._nlp is not None and self._nlp is not nmpc.nlp[0]:
            raise RuntimeError("Muscle horizon cannot be attached to a rebuilt NLP.")
        self._nlp = nmpc.nlp[0]

    def observe_solver(self, nmpc, *, record_observation: bool = True) -> None:
        self.attach(nmpc)
        interface = getattr(nmpc, "ocp_solver", None)
        solver = getattr(interface, "shaked_ocp_solver", None)
        if solver is None:
            solver = getattr(interface, "acados_solver", None)
        if solver is None:
            return
        if self._solver is not None and solver is not self._solver:
            raise RuntimeError("The compiled muscle-horizon solver changed between windows.")
        self._solver = solver
        if record_observation:
            self.solver_observation_count += 1

    def update(self, nmpc, profile: CertifiedMuscleHorizonProfile) -> None:
        """Update only numerical profile values and the free-PW warm start."""

        self.attach(nmpc)
        if (
            profile.layout != self.options.profile.layout
            or profile.muscles != self.options.profile.muscles
            or profile.muscle_names != self.options.profile.muscle_names
        ):
            raise ValueError("A muscle-horizon update cannot change the compiled graph structure.")
        self.observe_solver(nmpc, record_observation=False)
        fixed_bounds = nmpc.parameter_bounds[MUSCLE_HORIZON_PROFILE_KEY]
        fixed_values = np.asarray(profile.packed_parameters)[:, None]
        if fixed_bounds.min.shape != fixed_values.shape or fixed_bounds.max.shape != fixed_values.shape:
            raise ValueError("Updated muscle-horizon profile dimensions changed.")
        fixed_bounds.min[...] = fixed_values
        fixed_bounds.max[...] = fixed_values

        from bioptim import InitialGuessList

        parameter_init = InitialGuessList()
        parameter_init.add(MUSCLE_HORIZON_PROFILE_KEY, initial_guess=fixed_values.copy())
        parameter_init.add(
            MUSCLE_HORIZON_FUTURE_PW_KEY,
            initial_guess=np.asarray(profile.future_pulse_width_seed)[:, None],
        )
        nmpc.update_initial_guess(parameter_init=parameter_init)
        context_compatibility = profile.validate_context(**self._ocp_context)
        self.options = MuscleHorizonOptions(
            profile,
            weight=self.options.weight,
            domain_epsilon=self.options.domain_epsilon,
        )
        self.context_compatibility = context_compatibility
        self.update_count += 1

    def summary(self) -> dict:
        return {
            **self.options.profile.structure,
            "weight": self.options.weight,
            "domain_epsilon": self.options.domain_epsilon,
            "objective_graph_build_count": self.build_count,
            "profile_update_count": self.update_count,
            "compiled_solver_observation_count": self.solver_observation_count,
            "compiled_solver_build_count": int(self._solver is not None),
            "compiled_solver_reuse_verified": self.solver_observation_count > 1,
            "source_cycle_index": self.options.profile.source_cycle_index,
            "certification_basis": self.options.profile.certification_basis,
            "ocp_context_compatibility": self.context_compatibility,
            "source_bound_projection_count": self.options.profile.source_bound_projection_count,
            "source_bound_projection_maximum_absolute_s": (
                self.options.profile.source_bound_projection_maximum_absolute_s
            ),
            "status": "experimental_not_yet_prospectively_validated",
        }
