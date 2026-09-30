#!/usr/bin/env python3
"""Plot the applied four-muscle pulse-width patterns from certified RHO archives.

The default inputs are the IPOPT/MA57 30-Hz K5--K9 archives.  A cycle holds
``stimulations_per_cycle`` consecutive controls (30 in the reported campaign),
so cycles 10, 55, and 100 are extracted without interpolation.  The script
intentionally fails loudly when an archive is partial or has a different
control layout rather than silently drawing a non-comparable trace.

Example
-------
MPLCONFIGDIR=/tmp/mpl \
  /home/mickaelbegon/miniforge3/envs/cocofest-rho32/bin/python3.11 \
  scripts/plot_pulse_width_patterns.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MUSCLES = ("Delt_ant", "Delt_post", "Biceps", "Triceps")
DEFAULT_CASES = {
    "K5: SX": "local-results/frequency-30hz-100cycles-20260922/K5/ipopt-ma57/trajectory.npz",
    "K6: compiled Hessian": "local-results/frequency-30hz-100cycles-20260922/K6/ipopt-ma57/trajectory.npz",
    "K7: compact output": "local-results/frequency-30hz-100cycles-20260922/K7/ipopt-ma57/trajectory.npz",
    "K8: hard $\\Delta PW$": "local-results/frequency-30hz-100cycles-20260922/K8/ipopt-ma57/trajectory.npz",
    "K9: bound + penalty": "local-results/frequency-30hz-100cycles-20260922/K9/ipopt-ma57/trajectory.npz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("docs/scientific_appendix/figures/pulse_width_patterns_k5_k9"),
                        help="Output prefix, without suffix.")
    parser.add_argument("--cycles", type=int, nargs="+", default=[10, 55, 100])
    parser.add_argument("--stimulations-per-cycle", type=int, default=30)
    return parser.parse_args()


def control_key(muscle: str, archive: np.lib.npyio.NpzFile) -> str:
    """Prefer the actually applied control, otherwise use the decision trace."""
    applied = f"applied_pulse_widths__last_pulse_width_{muscle}"
    decision = f"controls__last_pulse_width_{muscle}"
    if applied in archive:
        return applied
    if decision in archive:
        return decision
    raise KeyError(f"No pulse-width archive for {muscle}; checked {applied} and {decision}.")


def load_case(path: Path, cycles: list[int], stimulations_per_cycle: int) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing archive: {path}")
    required = max(cycles) * stimulations_per_cycle
    with np.load(path, allow_pickle=False) as archive:
        data: dict[str, np.ndarray] = {}
        for muscle in MUSCLES:
            trace = np.asarray(archive[control_key(muscle, archive)]).reshape(-1)
            if trace.size < required:
                raise ValueError(
                    f"{path} has only {trace.size} pulse-width controls for {muscle}; "
                    f"{required} are needed through cycle {max(cycles)}."
                )
            data[muscle] = trace
    return data


def main() -> None:
    args = parse_args()
    if any(cycle < 1 for cycle in args.cycles):
        raise ValueError("Cycle numbers are one-based positive integers.")

    cases = {label: load_case(Path(path), args.cycles, args.stimulations_per_cycle)
             for label, path in DEFAULT_CASES.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(len(MUSCLES), len(args.cycles), figsize=(13.5, 8.8),
                                sharex=True, sharey="row")
    colors = plt.get_cmap("tab10").colors
    sample = np.arange(1, args.stimulations_per_cycle + 1)
    for row, muscle in enumerate(MUSCLES):
        for col, cycle in enumerate(args.cycles):
            axis = axes[row, col]
            start = (cycle - 1) * args.stimulations_per_cycle
            stop = start + args.stimulations_per_cycle
            for color, (label, data) in zip(colors, cases.items()):
                axis.step(sample, 1e6 * data[muscle][start:stop], where="mid", label=label,
                          color=color, linewidth=1.35)
            if row == 0:
                axis.set_title(f"cycle {cycle}")
            if col == 0:
                axis.set_ylabel(f"{muscle}\\nPW (µs)")
            if row == len(MUSCLES) - 1:
                axis.set_xlabel("stimulation within cycle")
            axis.grid(alpha=0.22)
            axis.set_xlim(1, args.stimulations_per_cycle)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=len(cases), frameon=False,
                  bbox_to_anchor=(0.5, 0.995))
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(args.output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Wrote {args.output.with_suffix('.png')}")
    print(f"Wrote {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
