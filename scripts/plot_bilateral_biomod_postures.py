"""Render the bilateral Wu bones and muscle paths at four crank positions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cocofest.dynamics.reduced_cycling import solve_bilateral_cycling_kinematics


MUSCLE_COLORS = {
    "Delt_ant": "#d55e00",
    "Delt_post": "#cc79a7",
    "Biceps": "#0072b2",
    "Triceps": "#009e73",
}
SIDES = {"right": "solid", "left": "dashed"}


def _biorbd():
    try:
        import biorbd
    except ImportError:
        import biorbd_casadi as biorbd
    return biorbd


def _array(value) -> np.ndarray:
    if hasattr(value, "to_array"):
        value = value.to_array()
    elif hasattr(value, "to_mx"):
        from casadi import evalf

        value = evalf(value.to_mx())
    return np.asarray(value, dtype=float)


def _marker_map(model, q: np.ndarray) -> dict[str, np.ndarray]:
    names = [name.to_string() for name in model.markerNames()]
    return {
        name: _array(marker).reshape(-1)
        for name, marker in zip(names, model.markers(q))
    }


def _segment_origins(model, q: np.ndarray) -> dict[str, np.ndarray]:
    return {
        model.segment(index).name().to_string(): _array(model.globalJCS(q, index))[:2, 3]
        for index in range(model.nbSegment())
    }


def _muscle_paths(model, q: np.ndarray) -> dict[str, np.ndarray]:
    kinematics = model.UpdateKinematicsCustom(q)
    paths = {}
    for index in range(model.nbMuscles()):
        name = model.muscle(index).name().to_string()
        paths[name] = np.asarray(
            [_array(point).reshape(-1)[:2] for point in model.muscle(index).musclesPointsInGlobal(kinematics, q)]
        )
    return paths


def _draw_bones(axis, origins: dict[str, np.ndarray], markers: dict[str, np.ndarray]) -> None:
    center = markers["wheel_center"][:2]
    axis.add_patch(plt.Circle(center, 0.1, fill=False, color="0.35", linewidth=1.4))
    for side in ("right", "left"):
        shoulder = origins[f"{side}_humerus"]
        elbow = origins[f"{side}_ulna"]
        hand = markers[f"{side}_hand"][:2]
        handle = markers[f"{side}_handle"][:2]
        axis.plot(*np.vstack((shoulder, elbow, hand)).T, color="0.12", linewidth=4.0, solid_capstyle="round")
        axis.plot([center[0], handle[0]], [center[1], handle[1]], color="0.12", linewidth=2.2)
        axis.scatter(*np.vstack((shoulder, elbow, hand, handle)).T, color="0.12", s=16, zorder=4)


def plot_postures(model_path: Path, output: Path) -> None:
    """Create four exact contact postures, at 0°, 90°, 180° and 270°."""
    theta, q, _ = solve_bilateral_cycling_kinematics(model_path, sample_count=5)
    model = _biorbd().Model(str(model_path))
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for phase_deg, sample, axis in zip((0, 90, 180, 270), range(4), axes.flat):
        coordinates = q[:, sample]
        _draw_bones(axis, _segment_origins(model, coordinates), _marker_map(model, coordinates))
        for name, points in _muscle_paths(model, coordinates).items():
            side, muscle = name.split("_", 1)
            axis.plot(
                points[:, 0],
                points[:, 1],
                color=MUSCLE_COLORS[muscle],
                linestyle=SIDES[side],
                linewidth=1.5,
                alpha=0.9,
            )
        axis.set_title(f"Manivelle {phase_deg}°")
        axis.set_aspect("equal")
        axis.set_xlim(-0.43, 0.49)
        axis.set_ylim(-0.43, 0.43)
        axis.grid(alpha=0.2)
    for axis in axes[-1, :]:
        axis.set_xlabel("x (m)")
    for axis in axes[:, 0]:
        axis.set_ylabel("y (m)")
    legend = [Line2D([0], [0], color="0.12", linewidth=4, label="os / manivelle")]
    legend.extend(Line2D([0], [0], color=color, linewidth=2, label=name) for name, color in MUSCLE_COLORS.items())
    legend.extend(
        [
            Line2D([0], [0], color="0.12", linewidth=2, linestyle="solid", label="bras droit"),
            Line2D([0], [0], color="0.12", linewidth=2, linestyle="dashed", label="bras gauche"),
        ]
    )
    figure.legend(handles=legend, loc="lower center", ncol=4, frameon=False)
    figure.suptitle("Wu bilatéral — os et chemins musculaires sur un tour de pédalage")
    figure.tight_layout(rect=(0, 0.08, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--biomod",
        type=Path,
        default=Path("examples/msk_models/Wu/Modified_Wu_Shoulder_Model_Cycling_Bilateral.bioMod"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    plot_postures(args.biomod, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
