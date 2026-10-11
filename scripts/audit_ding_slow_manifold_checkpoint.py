#!/usr/bin/env python3
"""Audit the Ding slow-manifold invariant at a certified checkpoint boundary."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.models.ding2007.ding2007_with_fatigue import DingModelPulseWidthFrequencyWithFatigue
from cocofest.optimization.ding_slow_manifold import DingSlowFatigueManifold
from scripts.probe_independent_rho_task_reserve import _load_arm


def manifolds_from_model(path: Path) -> dict[str, DingSlowFatigueManifold]:
    """Build model-faithful manifolds, retaining defaults explicitly in output."""
    payload = json.loads(path.read_text())
    result = {}
    allowed = {"a_scale", "alpha_a", "alpha_tau1", "alpha_km", "tau_fat", "tau1_rest", "km_rest"}
    for name, overrides in payload["muscles"].items():
        model = DingModelPulseWidthFrequencyWithFatigue(muscle_name=name)
        for key, value in overrides.items():
            if key in allowed:
                setattr(model, key, float(value))
        result[name] = DingSlowFatigueManifold.from_model(model)
    return result


def audit(receipt: Path, *, side: str, tolerance: float = 1e-8) -> dict:
    source, _ = _load_arm(Path(receipt).resolve(strict=True), side)
    source.verify_files()
    models = manifolds_from_model(Path(source.model_path))
    rows = []
    with np.load(source.archive_path, allow_pickle=False) as archive:
        for name, manifold in models.items():
            states = tuple(float(np.asarray(archive[f"states__{state}_{name}"], float)[0, 0])
                           for state in ("A", "Tau1", "Km"))
            residual = manifold.invariant_residual(*states)
            ratio = states[0] / manifold.a_rest
            rows.append({"muscle": name, "initial_capacity_ratio": ratio,
                         "slow_manifold_residual": residual,
                         "on_exact_slow_manifold": residual <= tolerance,
                         "a_rest": manifold.a_rest, "tau1_rest": manifold.tau1_rest,
                         "km_rest": manifold.km_rest, "alpha_a": manifold.alpha_a,
                         "alpha_tau1": manifold.alpha_tau1, "alpha_km": manifold.alpha_km})
    return {"schema_version": 1, "receipt": str(receipt),
            "receipt_sha256": sha256(Path(receipt).read_bytes()).hexdigest(), "side": side,
            "checkpoint_archive": str(source.archive_path), "archive_sha256": source.archive_sha256,
            "model_sha256": source.model_sha256, "tolerance": tolerance,
            "muscles": rows, "maximum_residual": max(row["slow_manifold_residual"] for row in rows),
            "all_on_exact_slow_manifold": all(row["slow_manifold_residual"] <= tolerance for row in rows),
            "scope": "initial archived slow states only; fast states/history remain separately conditioned"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args(argv))
    output = args.pop("output").resolve()
    report = audit(**args)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(output)


if __name__ == "__main__":
    main()
