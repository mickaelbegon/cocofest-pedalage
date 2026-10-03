#!/usr/bin/env python3
"""Audit the local fatigue objective at archived RHO cycle endpoints.

This is an endpoint calculation, not a replay of the NLP quadrature.  Its
inputs are archived endpoint capacity ratios, objective weights, and the
model's muscle order/scale.  No OCP is solved and no archive is modified.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date
import json
import math
from pathlib import Path
from typing import Any


OBJECTIVE_SCALE = 10_000.0
MUSCLE_COST_SOURCE = "cocofest/custom_objectives.py:minimize_parameterized_overall_muscle_fatigue"
WEIGHT_SOURCE = "examples/fes_multibody/cycling/cycling_pulse_width_mhe.py:set_objective_functions"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def endpoint_terms(ratio: float, weight: float, a_scale: float, *, epsilon: float = 1e-6) -> dict[str, float]:
    """Evaluate 10^4*w*(1-A/a_scale)^2 and its endpoint derivatives.

    The finite difference perturbs the normalized capacity ratio directly.
    The derivative with respect to physical A follows by the chain rule.
    """
    if not all(math.isfinite(x) for x in (ratio, weight, a_scale, epsilon)):
        raise ValueError("Ratio, weight, a_scale, and epsilon must be finite.")
    if ratio < 0 or weight < 0 or a_scale <= 0 or epsilon <= 0:
        raise ValueError("Invalid endpoint fatigue inputs.")
    loss = 1.0 - ratio
    density = OBJECTIVE_SCALE * weight * loss * loss
    analytic_ratio = -2.0 * OBJECTIVE_SCALE * weight * loss
    plus = OBJECTIVE_SCALE * weight * (1.0 - (ratio + epsilon)) ** 2
    minus = OBJECTIVE_SCALE * weight * (1.0 - (ratio - epsilon)) ** 2
    finite_difference_ratio = (plus - minus) / (2.0 * epsilon)
    return {
        "capacity_loss": loss,
        "endpoint_density": density,
        "derivative_per_capacity_ratio": analytic_ratio,
        "finite_difference_per_capacity_ratio": finite_difference_ratio,
        "derivative_per_A_unit": analytic_ratio / a_scale,
    }


def _muscle_data(result: dict[str, Any], source: Path) -> tuple[list[str], dict[str, float]]:
    model = result.get("configured_model")
    if not isinstance(model, dict):
        raise ValueError(f"Missing configured_model: {source}")
    builds = model.get("model_builds")
    parameters = model.get("configured_muscle_parameters")
    if not isinstance(builds, list) or not builds or not isinstance(builds[0], dict):
        raise ValueError(f"Missing model_builds/muscle order: {source}")
    names = builds[0].get("muscle_names")
    if not isinstance(names, list) or not names or len(set(names)) != len(names):
        raise ValueError(f"Invalid muscle order: {source}")
    if not isinstance(parameters, dict):
        raise ValueError(f"Missing configured_muscle_parameters: {source}")
    scale = {}
    for name in names:
        try:
            scale[name] = float(parameters[name]["a_scale"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Missing a_scale for {name}: {source}") from error
        if not math.isfinite(scale[name]) or scale[name] <= 0:
            raise ValueError(f"Invalid a_scale for {name}: {source}")
    return names, scale


def _certified_cycles(result: dict[str, Any], source: Path) -> list[dict[str, Any]]:
    cycles = result.get("cycles")
    if not isinstance(cycles, list) or not cycles:
        raise ValueError(f"Missing cycle records: {source}")
    certified = [record for record in cycles if isinstance(record, dict) and record.get("certified") is True]
    if not certified:
        raise ValueError(f"No certified cycle records: {source}")
    indexes = [int(record["cycle"]) for record in certified]
    if indexes != sorted(set(indexes)):
        raise ValueError(f"Non-monotonic or duplicate certified cycles: {source}")
    return certified


def audit_case(label: str, case_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for side in ("right", "left"):
        source = case_dir / side / "result.json"
        result = _read_json(source)
        names, scales = _muscle_data(result, source)
        certified = _certified_cycles(result, source)
        checkpoints = [("early", 0), ("mid", (len(certified) - 1) // 2), ("late", len(certified) - 1)]
        for checkpoint, index in checkpoints:
            cycle = certified[index]
            ratios = cycle.get("capacity_ratios")
            weights = cycle.get("fatigue_objective_weights")
            if not isinstance(ratios, dict) or not isinstance(weights, list) or len(weights) != len(names):
                raise ValueError(f"Missing endpoint ratios or objective weights: {source}, cycle {cycle['cycle']}")
            try:
                measured_cost = float(cycle["cost"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Missing measured NLP cost: {source}, cycle {cycle['cycle']}") from error
            if not math.isfinite(measured_cost):
                raise ValueError(f"Non-finite measured NLP cost: {source}, cycle {cycle['cycle']}")
            for name, weight in zip(names, weights):
                if name not in ratios:
                    raise ValueError(f"Missing ratio for {name}: {source}, cycle {cycle['cycle']}")
                ratio = float(ratios[name])
                terms = endpoint_terms(ratio, float(weight), scales[name])
                rows.append({
                    "case": label, "side": side, "checkpoint": checkpoint,
                    "cycle": int(cycle["cycle"]), "certified_cycles_in_archive": len(certified),
                    "muscle": name, "capacity_ratio_measured": ratio,
                    "a_scale_configured": scales[name], "objective_weight_measured": float(weight),
                    "nlp_cost_measured": measured_cost, "result_json": str(source),
                    **terms,
                })
    return rows


def _write_report(output: Path, rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["case"], row["side"], row["checkpoint"]), []).append(row)
    lines = [
        f"# Audit du coût marginal de fatigue — {date.today().isoformat()}", "",
        "L’archive mesure le rapport de capacité en fin de cycle, le poids de l’objectif et le coût NLP total. "
        "Le tableau calcule ensuite le coût et la pente **locaux à cette extrémité** selon "
        "`10 000 × poids × (1 − capacité)^2`. Ce calcul est exact pour le terme ponctuel "
        "quadratique et a été contrôlé par différence finie centrale. Il ne reconstitue pas "
        "l’intégrale sur les nœuds temporels du RHO ni les autres termes de coût.", "",
        f"Sources du calcul dans le dépôt : `{MUSCLE_COST_SOURCE}` et `{WEIGHT_SOURCE}`.", "",
        "| Cas | Bras | Checkpoint | Cycle | Coût NLP mesuré | Densité terminale calculée | "
        "Pente absolue totale par rapport à la capacité normalisée |", "|---|---|---|---:|---:|---:|---:|",
    ]
    for (case, side, checkpoint), group in groups.items():
        cycle = group[0]["cycle"]
        cost = group[0]["nlp_cost_measured"]
        density = sum(row["endpoint_density"] for row in group)
        slope = sum(abs(row["derivative_per_capacity_ratio"]) for row in group)
        lines.append(f"| {case} | {side} | {checkpoint} | {cycle} | {cost:.4g} | {density:.4g} | {slope:.4g} |")
    lines += [
        "", "## Interprétation et limites", "",
        "- La pente locale vaut `−20 000 × poids × (1 − capacité)` par unité de capacité normalisée. "
        "Son amplitude augmente donc avec la fatigue accumulée, même lorsque les poids restent fixes. "
        "Pour une variation `δ` de capacité normalisée, la variation exacte du terme ponctuel "
        "est `pente × δ + 10 000 × poids × δ²`.",
        "- Une variation de capacité d’un muscle modifie aussi la trajectoire future des états Ding et la faisabilité. "
        "Les pentes ci-dessus sont des dérivées partielles du seul terme de coût, pas des costates ni des effets causaux sur l’endurance.",
        "- Le coût NLP mesuré couvre l’horizon courant; la densité terminale est ponctuelle. "
        "Leurs amplitudes ne doivent pas être comparées directement.",
        "- Les checkpoints sont le premier, le milieu et le dernier cycle certifié de chaque bras; "
        "si la simulation continue, le dernier n’est pas nécessairement proche de l’épuisement.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_audit(cases: list[tuple[str, Path]], output: Path) -> list[dict[str, Any]]:
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    labels = [label for label, _ in cases]
    if len(labels) != len(set(labels)):
        raise ValueError("Case labels must be unique.")
    rows = [row for label, path in cases for row in audit_case(label, path)]
    output.mkdir(parents=True)
    (output / "audit.json").write_text(json.dumps({
        "method": "archived_endpoint_quadratic_fatigue_expansion_v1",
        "objective_scale": OBJECTIVE_SCALE,
        "quantity_types": {"capacity_ratio_measured": "measured", "objective_weight_measured": "measured",
                           "nlp_cost_measured": "measured", "endpoint_density": "derived",
                           "derivative_per_capacity_ratio": "derived",
                           "derivative_per_A_unit": "derived"},
        "rows": rows,
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_report(output, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", required=True, metavar="LABEL=CASE_DIR")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cases: list[tuple[str, Path]] = []
    for case in args.case:
        label, separator, path = case.partition("=")
        if not separator or not label.strip() or not path.strip():
            parser.error("Each --case must be LABEL=CASE_DIR.")
        cases.append((label.strip(), Path(path).resolve(strict=True)))
    rows = write_audit(cases, args.output)
    print(f"Wrote {len(rows)} muscle-checkpoint records to {args.output}")


if __name__ == "__main__":
    main()
