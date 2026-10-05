"""Plot exact bilateral Wu fibre-length trajectories over one crank cycle."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cocofest.dynamics.reduced_cycling import solve_bilateral_cycling_kinematics


MUSCLES = ("Delt_ant", "Delt_post", "Biceps", "Triceps")


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


def fibre_trajectories(model_path: Path, samples: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return normalised fibre lengths on the exact bilateral contact manifold."""
    theta, q, _ = solve_bilateral_cycling_kinematics(model_path, sample_count=samples)
    model = _biorbd().Model(str(model_path))
    names = [model.muscle(index).name().to_string() for index in range(model.nbMuscles())]
    trajectories = {}
    for index, name in enumerate(names):
        values = np.empty(theta.size)
        optimal = float(_array(model.muscle(index).characteristics().optimalLength()).reshape(-1)[0])
        for sample in range(theta.size):
            kinematics = model.UpdateKinematicsCustom(q[:, sample])
            values[sample] = float(
                _array(model.muscle(index).length(kinematics, q[:, sample], True)).reshape(-1)[0]
            ) / optimal
        trajectories[name] = values
    return np.linspace(0.0, 360.0, theta.size), trajectories


def plot(model_path: Path, output: Path, samples: int) -> None:
    phase, trajectories = fibre_trajectories(model_path, samples)
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for axis, muscle in zip(axes.flat, MUSCLES):
        right, left = trajectories[f"right_{muscle}"], trajectories[f"left_{muscle}"]
        axis.axhspan(-1.0, 0.0, color="tab:red", alpha=0.10, label="non physique (< 0)")
        axis.axhline(0.0, color="black", linewidth=0.9)
        axis.plot(phase, right, color="tab:blue", label="droite")
        axis.plot(phase, left, color="tab:orange", linestyle="--", label="gauche")
        axis.set_title(muscle)
        axis.set_ylabel("Longueur de fibre / longueur optimale")
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    for axis in axes[1, :]:
        axis.set_xlabel("Phase de manivelle (degrés)")
    figure.suptitle("BioMod Wu bilatéral — trajectoires exactes des fibres")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--biomod", type=Path,
        default=Path("examples/msk_models/Wu/Modified_Wu_Shoulder_Model_Cycling_Bilateral.bioMod"),
    )
    parser.add_argument("--samples", type=int, default=181)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.samples < 5:
        parser.error("--samples must be at least 5")
    plot(args.biomod, args.output, args.samples)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
