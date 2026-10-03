#!/usr/bin/env python3
"""Read-only archive diagnostic of task margin / consumption-rate proxies.

No solve is run and no future force profile is invented. The repeated-force
counterfactual is explicitly separated from the recorded continuation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cocofest.optimization.ding_fatigue_rollout import DingFatigueParameters
from cocofest.optimization.ding_slow_cycle import DingSlowCycleMap
from cocofest.optimization.endurance_margin_diagnostic import (
    LocalMarginDiagnostic, repeated_force_margin_path, witness_grid_consumption,
)


def read(path):
    return json.loads(Path(path).read_text())


def provenance(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def diagnose(campaign: Path, slow_report_path: Path, validation_path: Path) -> dict:
    slow = read(slow_report_path)
    validation = read(validation_path)
    source_result_path = Path(validation["source_result"])
    source_request_path = Path(validation["source_request"])
    source_result, source_request = read(source_result_path), read(source_request_path)
    model_path = Path(slow["model_result_path"])
    if provenance(model_path)["sha256"] != slow["model_result_sha256"]:
        raise ValueError("Slow replay parameter source changed")
    if source_request["checkpoint"]["model_sha256"] != slow["source_model_sha256"]:
        raise ValueError("Margin and slow replay model fingerprints differ")
    if validation.get("validation_passed") is not True:
        raise ValueError("Local gradient has no independent validation")
    names = tuple(item["state_key"] for item in validation["coordinate_layout"])
    if any(item["index"] != 0 for item in validation["coordinate_layout"]):
        raise ValueError("Only scalar muscle coordinates supported")
    model = LocalMarginDiagnostic(names, tuple(validation["center"]),
        tuple(item["scale"] for item in validation["coordinate_layout"]),
        tuple(item["offset"] for item in validation["coordinate_layout"]),
        tuple(validation["gradient"]), tuple(validation["trust_radius"]),
        source_result["load_witness"]["work_scale"])
    parameters = {name: DingFatigueParameters(value["a_scale"], value["tau1_rest"],
        value["km_rest"], value["alpha_a"], value["alpha_tau1"], value["alpha_km"], value["tau_fat"])
        for name, value in read(model_path)["configured_model"]["configured_muscle_parameters"].items()}
    directory = Path(slow["witness_directory"])
    continuation = read(directory / "summary.json")
    certified = {row["cycle"] for row in continuation["cycles"] if row.get("certified") is True}
    by_cycle = {}
    for row in slow["rows"]:
        by_cycle.setdefault(row["cycle"], {})[row["muscle"]] = row
    measured_rows, forecasts = [], []
    slow_mask = np.asarray([name.startswith(("A_", "Tau1_", "Km_")) for name in names])
    start = None
    for artifact in slow["source_archives"]:
        cycle = artifact["cycle"]
        if cycle not in certified or provenance(artifact["path"])["sha256"] != artifact["sha256"]:
            raise ValueError("Solved cycle missing certification or changed since slow replay")
        with np.load(artifact["path"], allow_pickle=False) as archive:
            physical_begin = {name: float(archive[f"states__{name}"].reshape(-1)[0]) for name in names}
            physical_end = {name: float(archive[f"states__{name}"].reshape(-1)[-1]) for name in names}
        begin, end = model.coordinates(physical_begin), model.coordinates(physical_end)
        if start is None:
            start = begin
            if not np.allclose(start, model.center, atol=1e-8, rtol=0):
                raise ValueError("First solved cycle does not begin at the locally validated checkpoint")
        delta = end - begin
        measured_rows.append({"completed_cycle": cycle, **model.evaluate(end),
            "normalized_state": end.tolist(),
            "slow_contribution_to_local_margin_change": float(np.dot(np.asarray(model.gradient)[slow_mask], delta[slow_mask])),
            "fast_contribution_to_local_margin_change": float(np.dot(np.asarray(model.gradient)[~slow_mask], delta[~slow_mask])),
            "remaining_recorded_certified_cycles": continuation["last_certified_cycle"] - cycle,
            "actual_future_margin_measured": False})
        if cycle in (141, 150, 160, 169):
            maps = {muscle: DingSlowCycleMap(slow["assumptions"]["cycle_duration_seconds"], parameters[muscle],
                row["weighted_force_integral_n_s"], row["mean_force_n"]*slow["assumptions"]["cycle_duration_seconds"])
                for muscle, row in by_cycle[cycle].items()}
            forecasts.append({"after_completed_cycle": cycle,
                "uses_only_this_cycle_observed_force": True,
                **repeated_force_margin_path(model, physical_end, maps, horizon=300)})

    grids, sources = {}, []
    for side in ("left", "right"):
        points = []
        for path in (campaign / f"task-reserve-calibration-20260930/{side}/probes").glob(f"c*/{side}-task-reserve.json"):
            doc = read(path)
            sources.append(provenance(path))
            if not doc.get("nominal_task_witnessed"):
                continue
            points.append({"cycle": doc["checkpoint"]["completed_cycles"],
                "load_lower_bound": doc["work_scale_lower_bound"],
                "tested_maximum": max(row["work_scale"] for row in doc["evidence"]),
                "global_upper_bound": doc.get("global_upper_bound")})
        grids[side] = witness_grid_consumption(points)
    unique = set()
    oracle_files = []
    for path in campaign.glob("task-load-margin*/result.json"):
        request = read(path.parent / "request.json")
        result = read(path)
        unique.add((request["checkpoint"]["archive_sha256"], request["checkpoint"]["prepared_problem_sha256"]))
        oracle_files.append({**provenance(path), "checkpoint": request["checkpoint"]["completed_cycles"],
            "request_id": request["request_id"], "local_load_witness": (result.get("load_witness") or {}).get("work_scale")})
    return {"schema": "endurance-margin-archive-diagnostic-v1", "read_only_no_ocp_solve": True,
        "sources": [provenance(slow_report_path), provenance(validation_path), provenance(source_result_path),
                    provenance(source_request_path), provenance(directory/"summary.json"), *sources],
        "availability": {"coarse_grid_checkpoints_per_side": {side: len(rows) for side, rows in grids.items()},
            "load_maximization_artifact_count": len(oracle_files), "unique_maximized_checkpoints": len(unique),
            "independently_observed_future_policy_load_maxima": 0,
            "full_solved_force_cycles": len(measured_rows), "physiological_failure_certified": False},
        "oracle_artifacts": oracle_files, "coarse_witness_grid": grids,
        "local_model": {"source_completed_cycle": source_request["checkpoint"]["completed_cycles"],
            "source_load_factor": model.load_factor, "validation_passed": validation["validation_passed"],
            "coordinate_names": model.names, "gradient": model.gradient,
            "local_validation_max_abs_error": validation["diagnostics"]["local_validation_max_abs_error"],
            "gradient_validation_max_abs_error": validation["diagnostics"]["gradient_validation_max_abs_error"],
            "activation_blockers": validation["activation_blockers"]},
        "observed_continuation": {"last_certified_cycle": continuation["last_certified_cycle"],
            "active_margin_cost_cycles": continuation["active_cost_committed_cycles"],
            "endpoint_interpretation": "observed policy continuation only; numerical stop is not a physiological endpoint",
            "rows": measured_rows}, "fixed_force_counterfactuals": forecasts,
        "verdict": {"endurance_costate_activation_supported": False,
            "locally_validated_load_gradient_available": True,
            "reasons": ["single_independent_maximized_checkpoint", "coarse_grid_load_cap_censoring",
                "no_observed_future_policy_margin_values", "no_reliable_physiological_endpoint",
                "repeated_force_does_not_certify_realizable_future_PW", "future_points_exit_local_trust_region"]},
        "predeclared_next_validation": {
            "minimum_independent_maximized_checkpoints_per_policy": 3,
            "sampling": "early, middle, late; same model/task/split and certified frozen state/history",
            "scope": "reoptimized witnessed load, not global optimum",
            "proposed_margin_error_gate": "absolute error <= max(0.01, 0.05 * measured positive reserve); prospectively chosen engineering gate, not clinical validation",
            "falsification": ["negative or undefined depletion near a witnessed decline", "zero crossing after coordinate trust exit",
                "incorrect ranking of held-out policy continuations", "material fast-state contribution omitted",
                "sign-reversed causal branch improves as much as the proposed direction"],
            "control_activation": "Only after held-out margin prediction and a causal 1/5/20-cycle branch comparison; no endurance assertion from an affine crossing"}}


def markdown(report):
    availability = report["availability"]
    local = report["local_model"]
    lines = ["# Diagnostic archive : marge de travail et durée restante", "",
        "Verdict : les archives permettent de tester la cohérence d'un signal local de marge, mais pas de valider un costate d'endurance. Aucun nouvel OCP n'a été lancé.", "",
        "## Ce qui est mesuré", "",
        f"{availability['load_maximization_artifact_count']} fichiers d'optimisation de charge correspondent à **{availability['unique_maximized_checkpoints']} seul checkpoint indépendant**, gauche c140. La charge réalisable locale vaut {local['source_load_factor']:.9f} fois la charge nominale. La réserve témoin est donc {local['source_load_factor']-1:.6f}. Ce témoin réalisable n'est pas un maximum global.", "",
        f"Le gradient sur les 20 états Ding dispose de validations indépendantes locales : erreur maximale sur la charge {local['local_validation_max_abs_error']:.3g}, différence de gradient {local['gradient_validation_max_abs_error']:.3g}. La validation reste conditionnelle aux états mécaniques et à l'historique de stimulation.", "",
        "Les sondages à charge fixée couvrent 8 checkpoints par côté. À droite, tous atteignent la charge maximale testée 1,20 : la réserve réelle n'est pas identifiée. À gauche, le même plafond est atteint de c20 à c120. Les échecs aux charges supérieures ne fournissent aucune borne supérieure démontrée.", "",
        "| Checkpoint gauche | Réserve témoin | Plafond de test atteint | Durée marge / pente des témoins |", "|---:|---:|:---:|---:|"]
    for row in report["coarse_witness_grid"]["left"]:
        estimate = row["time_proxy"]["cycles"]
        lines.append(f"| {row['cycle']} | {row['reserve_lower_bound']:.3f} | {'oui' if row['at_tested_load_cap'] else 'non'} | {'indéterminée' if estimate is None else f'{estimate:.1f} cycles'} |")
    lines += ["", "Ces durées sont des extrapolations de bornes inférieures échantillonnées, sans garantie. La suite enregistrée atteint c169; comparer 60 cycles restants à c140 et 10 à c160 aux 29 et 9 cycles encore enregistrés montre la sensibilité au checkpoint, pas une validation de la durée jusqu'à épuisement. Cette suite a reçu le coût de marge pendant un seul cycle. Son arrêt numérique ne constitue pas un endpoint physiologique.", "",
        "## Ce que prédit l'approximation locale sur les états réellement enregistrés", "",
        "La colonne réserve est évaluée par le modèle affine de c140; elle n'est pas une nouvelle résolution de charge à chacun des cycles. Les états proviennent de trajectoires NLP certifiées et la propagation lente est vérifiée séparément. Une fraction de confiance >1 signifie que le modèle est extrapolé.", "",
        "| Cycle atteint | Réserve affine | Fraction de confiance | Variation lente / cycle | Variation rapide / cycle |", "|---:|---:|---:|---:|---:|"]
    for row in report["observed_continuation"]["rows"]:
        if row["completed_cycle"] in (141,142,145,150,160,169):
            lines.append(f"| {row['completed_cycle']} | {row['affine_margin']:.6f} | {row['trust_fraction']:.2f} | {row['slow_contribution_to_local_margin_change']:.6f} | {row['fast_contribution_to_local_margin_change']:.6f} |")
    lines += ["", "## Répéter la force observée : expérience contrefactuelle économique", "",
        "Après chaque cycle sélectionné, on répète uniquement son profil de force observé pendant 300 cycles dans la carte exponentielle des états A, Tau1 et Km. Cn et F au raccord sont figés. Aucune PW future n'est construite, aucun travail futur n'est certifié. L'âge du gradient et le dépassement de son domaine sont conservés dans le diagnostic.", "",
        "| Départ après cycle | Durée réserve / consommation | Premier zéro affine | Sortie du domaine à +n cycles |", "|---:|---:|---:|---:|"]
    for forecast in report["fixed_force_counterfactuals"]:
        estimate = forecast["linear_consumption_time"]["cycles"]
        lines.append(f"| {forecast['after_completed_cycle']} | {'indéterminée' if estimate is None else f'{estimate:.1f}'} | {forecast['first_affine_zero_crossing']} | {forecast['first_coordinate_trust_exit']} |")
    lines += ["", "Un zéro calculé après la sortie du domaine n'est pas une durée prédite exploitable. Une expression réserve / consommation peut donner une unité en cycles tout en conservant une mauvaise direction de commande, particulièrement si le taux de consommation dépend de la politique future.", "",
        "## Validation à réaliser avant un costate", "",
        "1. Réoptimiser la charge sur au moins trois checkpoints indépendants (début, milieu, fin), par politique appariée; conserver le gel complet et vérifier les témoins. Déplafonner les sondages précoces si nécessaire.",
        "2. Réserver des checkpoints à la validation, comparer les valeurs de marge prévues aux valeurs réellement réoptimisées et quantifier séparément l'effet des états rapides. Le seuil prospectif proposé est une erreur absolue ≤ max(0,01; 5 % de la réserve mesurée), seuil d'ingénierie à discuter, sans statut clinique.",
        "3. Vérifier le classement de continuations de politiques distinctes. Le classement chronologique d'une seule trajectoire, avec des réserves plafonnées, ne valide pas le choix d'une meilleure commande.",
        "4. Tester ensuite les signes opposés du gradient sur branches 1/5/20 cycles identiques, puis seulement une continuation jusqu'au critère de faisabilité gelée.", "",
        "Le code n'active aucune nouvelle fonction coût. Les données et leurs SHA-256 sont consignés dans report.json; les archives sources restent intactes.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--slow-report", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(args.campaign, args.slow_report, args.validation)
    args.output_directory.mkdir(parents=True, exist_ok=False)
    (args.output_directory/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    (args.output_directory/"rapport_fr.md").write_text(markdown(report))
    print(json.dumps({"availability": report["availability"], "verdict": report["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
