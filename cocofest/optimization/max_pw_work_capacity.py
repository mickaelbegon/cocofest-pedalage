"""One-cycle positive-work capacity under fixed maximum pulse widths.

This is a terminal-state-dependent proxy, not an endurance or feasibility
certificate. The prescribed isokinetic mechanics are numerical parameters.
Calcium is propagated analytically; the four remaining Ding states and work
use the same RK4 stages. No future controls or nested optimization are added.
"""

from dataclasses import dataclass
import math

import numpy as np

from .adaptive_moment_rollout import DingPulseWidthParameters
from .muscle_horizon_objective import _four_state_rhs, _periodic_cn


DOMAIN_NAMES = ("Cn", "F", "A", "Tau1", "Km_plus_Cn", "relaxation_time")


@dataclass(frozen=True)
class MaxPwWorkCapacityLayout:
    muscle_count: int
    interval_count: int
    integration_substeps: int = 16

    def __post_init__(self):
        for name in ("muscle_count", "interval_count", "integration_substeps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")

    @property
    def parameter_size(self):
        return self.interval_count + self.muscle_count * self.interval_count * (
            2 + 6 * self.integration_substeps
        )

    @property
    def initial_state_size(self):
        return 5 * self.muscle_count

    @property
    def domain_margin_size(self):
        return 6 * 5 * self.muscle_count * self.interval_count * self.integration_substeps

    def slices(self):
        k, mk, mks = (self.interval_count, self.muscle_count * self.interval_count,
                      self.muscle_count * self.interval_count * self.integration_substeps)
        result, start = {}, 0
        for name, size in (("duration", k), ("calcium_amplitude", mk), ("stimulation_mask", mk),
                           ("gain_start", mks), ("gain_midpoint", mks), ("gain_endpoint", mks),
                           ("power_start", mks), ("power_midpoint", mks), ("power_endpoint", mks)):
            result[name] = slice(start, start + size)
            start += size
        return result


def pack_max_pw_work_profile(intervals, layout, *, angular_velocity_rad_s,
                             moment_coefficient_functions=None, stimulation_mask=None,
                             stimulation_policy="all_intervals_pw_max"):
    """Pack fixed mechanics in C order; each callable uses interval-local time.

    ``moment_coefficient_functions[k][m](t)`` optionally supplies b_i(t)
    rather than the interval's frozen coefficient. Positive power is
    ``max(omega*b_i(t), 0)*F_i(t)``, matching reduced-model E_prod_dot.
    The maximum is applied to numerical geometry, never to an NLP state.
    ``all_intervals_pw_max`` uses PW max throughout. The alternative
    ``selected_propulsive_intervals`` applies PW max only on intervals whose
    midpoint geometry is propulsive; elsewhere PW=pd0 gives zero recruitment.
    A custom binary
    ``stimulation_mask`` must have shape (muscles, intervals). The selection
    remains fixed for all candidate terminal states. Mask and positive-power
    coefficients are separate: residual force can persist outside selected
    intervals but braking work is never credited.
    Calcium amplitudes must describe the future stimulation history and
    frequency, not be reset from rest at the terminal boundary.
    """
    intervals = tuple(intervals)
    if stimulation_policy not in ("all_intervals_pw_max", "selected_propulsive_intervals"):
        raise ValueError("Unknown stimulation_policy.")
    if len(intervals) != layout.interval_count:
        raise ValueError("intervals must match layout.interval_count.")
    omega = float(angular_velocity_rad_s)
    if not math.isfinite(omega) or omega == 0:
        raise ValueError("angular_velocity_rad_s must be finite and nonzero.")
    if moment_coefficient_functions is not None and (
        len(moment_coefficient_functions) != layout.interval_count or
        any(len(row) != layout.muscle_count for row in moment_coefficient_functions)
    ):
        raise ValueError("moment_coefficient_functions must have interval-by-muscle shape.")
    shape = (layout.muscle_count, layout.interval_count, layout.integration_substeps)
    fields = {key: np.empty(shape) for key in layout.slices() if key.startswith(("gain_", "power_"))}
    fields["duration"] = np.asarray([row.duration for row in intervals])
    if any(len(row.calcium_amplitudes) != layout.muscle_count for row in intervals):
        raise ValueError("interval muscle count must match layout.")
    fields["calcium_amplitude"] = np.asarray([row.calcium_amplitudes for row in intervals]).T
    fields["stimulation_mask"] = np.empty((layout.muscle_count, layout.interval_count))
    for k, row in enumerate(intervals):
        for m in range(layout.muscle_count):
            gain = row.mechanical_gains[m]
            coefficient = row.moment_coefficients[m] if moment_coefficient_functions is None else moment_coefficient_functions[k][m]
            midpoint_coefficient = float(coefficient(.5 * row.duration) if callable(coefficient) else coefficient)
            if not math.isfinite(midpoint_coefficient):
                raise ValueError("Moment coefficients must be finite.")
            fields["stimulation_mask"][m, k] = float(omega * midpoint_coefficient > 0.)
            for s in range(layout.integration_substeps):
                for label, fraction in (("start", 0.), ("midpoint", .5), ("endpoint", 1.)):
                    time = (s + fraction) * row.duration / layout.integration_substeps
                    fields[f"gain_{label}"][m, k, s] = gain(time) if callable(gain) else gain
                    b = float(coefficient(time) if callable(coefficient) else coefficient)
                    if not math.isfinite(b):
                        raise ValueError("Moment coefficients must be finite.")
                    fields[f"power_{label}"][m, k, s] = max(omega * b, 0.)
    if stimulation_mask is not None:
        mask = np.asarray(stimulation_mask, dtype=float)
        if mask.shape != fields["stimulation_mask"].shape or not np.all((mask == 0.) | (mask == 1.)):
            raise ValueError("stimulation_mask must be binary with shape (muscles, intervals).")
        fields["stimulation_mask"] = mask.copy()
    elif stimulation_policy == "all_intervals_pw_max":
        fields["stimulation_mask"][:] = 1.
    packed = np.empty(layout.parameter_size)
    for name, section in layout.slices().items():
        value = fields[name]
        if not np.all(np.isfinite(value)) or np.any(value < 0):
            raise ValueError(f"{name} must be finite and nonnegative.")
        if (name == "duration" or name.startswith("gain_")) and np.any(value <= 0):
            raise ValueError(f"{name} must be strictly positive.")
        packed[section] = value.ravel()
    return packed


def max_pw_work_domain_lower_bounds(layout, *, epsilon=1e-10):
    """Bounds for every stage, retaining zero-force/zero-calcium admissibility."""
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive.")
    return np.tile([0., 0., epsilon, epsilon, epsilon, epsilon], layout.domain_margin_size // 6)


def build_max_pw_work_capacity_function(*, muscles, layout, symbolic_type="SX"):
    """Build once; terminal state and profile values can change without rebuild.

    Output domain margins cover the four RK stages and endpoint of every
    substep. Callers must enforce these domains or reject invalid probes;
    no clipping hides negative force, depleted capacity, or singular states.
    The experimental rollout uses exact calcium plus RK4, independently of
    the enclosing RHO's scientific Radau-5 transcription.
    """
    import casadi as ca

    muscles = tuple(muscles)
    if len(muscles) != layout.muscle_count or any(not isinstance(p, DingPulseWidthParameters) for p in muscles):
        raise ValueError("muscles must match layout and contain DingPulseWidthParameters.")
    if symbolic_type not in ("SX", "MX"):
        raise ValueError("symbolic_type must be SX or MX.")
    symbol = getattr(ca, symbolic_type).sym
    states = symbol("terminal_state", layout.initial_state_size)
    profile = symbol("profile", layout.parameter_size)
    slices = layout.slices()

    def field(name, m, k, s=0):
        offset = k if name == "duration" else m * layout.interval_count + k
        if name.startswith(("gain_", "power_")):
            offset = offset * layout.integration_substeps + s
        return profile[slices[name].start + offset]

    margins, end_states, work = [], [], []
    for m, muscle in enumerate(muscles):
        cn = states[5*m]
        current = states[5*m+1:5*m+5]
        produced_work = 0.

        def domain(cn_value, state):
            force, a, tau1, km = (state[i] for i in range(4))
            margins.extend((cn_value, force, a, tau1, km + cn_value,
                            tau1 + muscle.tau2 * cn_value / (km + cn_value)))

        for k in range(layout.interval_count):
            duration = field("duration", m, k)
            h = duration / layout.integration_substeps
            amplitude = field("calcium_amplitude", m, k)
            pw = muscle.pd0 + field("stimulation_mask", m, k)*(muscle.pulse_width_max - muscle.pd0)
            for s in range(layout.integration_substeps):
                c0, cm, ce = [_periodic_cn(cn, (s + f)*h, amplitude, muscle.tauc) for f in (0., .5, 1.)]

                def rhs(x, c, phase):
                    return _four_state_rhs(x, cn=c, pulse_width=pw,
                                           mechanical_gain=field(f"gain_{phase}", m, k, s), muscle=muscle)

                k1 = rhs(current, c0, "start")
                x2 = current + .5*h*k1
                k2 = rhs(x2, cm, "midpoint")
                x3 = current + .5*h*k2
                k3 = rhs(x3, cm, "midpoint")
                x4 = current + h*k3
                k4 = rhs(x4, ce, "endpoint")
                produced_work += h/6 * (field("power_start", m, k, s)*current[0]
                    + 2*field("power_midpoint", m, k, s)*(x2[0] + x3[0])
                    + field("power_endpoint", m, k, s)*x4[0])
                for c, x in ((c0, current), (cm, x2), (cm, x3), (ce, x4)):
                    domain(c, x)
                current = current + h/6*(k1 + 2*k2 + 2*k3 + k4)
                domain(ce, current)
            cn = _periodic_cn(cn, duration, amplitude, muscle.tauc)
        work.append(produced_work)
        end_states.append(ca.vertcat(cn, current))
    per_muscle = ca.vertcat(*work)
    return ca.Function("max_pw_work_capacity", [states, profile],
                       [ca.sum1(per_muscle), per_muscle, ca.vertcat(*end_states), ca.vertcat(*margins)],
                       ["terminal_state", "profile"],
                       ["capacity_j", "muscle_work_j", "final_state", "domain_margins"])
