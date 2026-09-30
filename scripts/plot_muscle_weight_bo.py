"""Render compact diagnostics for an asynchronous muscle-weight BO campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


WEIGHTS = ("Biceps", "Delt_ant", "Delt_post", "Triceps")


def effective_weights(entry):
    """Prefer the controller receipt over latent BO coordinates.

    v2 campaigns optimise centred-log coordinates, not independent raw
    multipliers.  Plotting their coordinates as weights would recreate the
    ambiguity the runner removed.  The post-run receipt is authoritative;
    fall back to the requested effective candidate when a run did not produce
    a journal (for example a technical failure retained for audit).
    """
    audit = entry.get("effective_weight_audit") or {}
    weights = audit.get("controller_applied_initial_weights")
    if not isinstance(weights, dict):
        weights = audit.get("requested_effective_initial_weights")
    if isinstance(weights, dict) and set(weights) == set(WEIGHTS):
        return [float(weights[name]) for name in WEIGHTS]
    # Historical v1 campaigns only recorded requested multipliers.  Retain a
    # readable plot, but never label it as an effective v2 cost.
    params = entry.get("parameters") or {}
    if all(f"muscle_weight__{name}" in params for name in WEIGHTS):
        return [float(params[f"muscle_weight__{name}"]) for name in WEIGHTS]
    return None


def load_rows(root: Path):
    rows = []
    for path in sorted((root / "trials").glob("trial_*/observation.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        observation = entry.get("sim_result") or {}
        if observation.get("status") != "observed" or not isinstance(observation.get("score"), (int, float)):
            continue
        weights = effective_weights(entry)
        if weights is not None:
            rows.append((int(entry["trial"]), float(observation["score"]), weights))
    if not rows:
        raise ValueError("Aucune observation BO valide.")
    rows.sort()
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows]), np.array([r[2] for r in rows])


def save_overview(destination: Path, trials, scores, weights):
    best = np.maximum.accumulate(scores)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes[0, 0].plot(trials, scores, ".", ms=4, alpha=.55, label="proxy observé")
    axes[0, 0].plot(trials, best, lw=2.3, color="#d55e00", label="meilleur cumulé")
    axes[0, 0].set(title="Convergence du BO", xlabel="Essai", ylabel="Cycles proxy")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].hist(scores, bins=18, color="#0072b2", alpha=.85)
    axes[0, 1].axvline(scores.max(), color="#d55e00", lw=2, label=f"max = {scores.max():.3f}")
    axes[0, 1].set(title="Distribution des proxys", xlabel="Cycles proxy", ylabel="Nombre d'essais")
    axes[0, 1].legend(frameon=False)
    axes[0, 2].axis("off")
    winner = weights[scores.argmax()]
    axes[0, 2].text(.04, .86, "Meilleur candidat", fontsize=15, weight="bold")
    axes[0, 2].text(.04, .65, f"{scores.max():.3f} cycles proxy", fontsize=17, color="#d55e00")
    axes[0, 2].text(.04, .42, "\n".join(f"{name}: ×{value:.3f}" for name, value in zip(WEIGHTS, winner)), fontsize=12)
    for axis, name, values in zip(axes[1], WEIGHTS, weights.T):
        axis.scatter(values, scores, c=scores, cmap="viridis", s=24, alpha=.75, edgecolors="none")
        axis.set_xscale("log")
        axis.set(title=name, xlabel="Poids relatif appliqué (échelle log)", ylabel="Cycles proxy")
        axis.axvline(1, color="0.4", lw=1, ls="--")
    fig.suptitle("BO IPOPT — poids musculaires, résistance 0,2 Nm", fontsize=16)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def save_pairwise(destination: Path, scores, weights):
    fig, axes = plt.subplots(4, 4, figsize=(11, 10), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(scores.min(), scores.max())
    for row in range(4):
        for col in range(4):
            axis = axes[row, col]
            if row == col:
                axis.hist(weights[:, row], bins=16, color="#56b4e9")
            else:
                axis.scatter(weights[:, col], weights[:, row], c=scores, cmap=cmap, norm=norm, s=17, alpha=.72, edgecolors="none")
                axis.set_xscale("log"); axis.set_yscale("log")
            if row == 3: axis.set_xlabel(WEIGHTS[col], fontsize=9)
            if col == 0: axis.set_ylabel(WEIGHTS[row], fontsize=9)
            axis.tick_params(labelsize=7)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=.8, label="Cycles proxy")
    fig.suptitle("Relations entre poids relatifs appliqués et endurance proxy", fontsize=15)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    root = args.campaign.resolve()
    output = root / "figures"; output.mkdir(exist_ok=True)
    trials, scores, weights = load_rows(root)
    save_overview(output / "bo-overview.png", trials, scores, weights)
    save_pairwise(output / "bo-pairwise.png", scores, weights)
    print(output)


if __name__ == "__main__":
    main()
