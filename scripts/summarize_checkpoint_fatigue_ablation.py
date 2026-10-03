#!/usr/bin/env python3
"""Audit and summarize the predeclared fixed-task objective ablation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cocofest.simulation.independent_arms_process import _atomic_json
from scripts.run_checkpoint_fatigue_ablation import VARIANTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    campaign = json.loads((args.directory / "campaign.json").read_text())
    if not (args.directory / "campaign-completion.json").exists():
        raise RuntimeError("Wait for short branch campaign completion before margin probes")
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MPLCONFIGDIR": "/tmp/cocofest-fatigue-ablation-mpl"}
    if args.probe:
        def one_side(side):
            for job in [j for j in campaign["jobs"] if j["side"] == side]:
                result_path = Path(job["output"]) / "result.json"
                if not result_path.exists():
                    continue
                result = json.loads(result_path.read_text())
                if not result["success"] or (result_path.parent / "margin-cycle-20.json").exists():
                    continue
                with open(result_path.parent / "margin.log", "a") as log:
                    proc = subprocess.run([sys.executable, str(ROOT / "scripts/probe_fatigue_ablation_endpoint.py"),
                        "--result", str(result_path), "--cpu", str(job["cpu"])], cwd=ROOT,
                        env=env, stdout=log, stderr=subprocess.STDOUT)
                print(f"Margin {side} c{job['source_cycle']} {job['variant']}: {proc.returncode}", flush=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(one_side, ("left", "right")))
    rows = []
    for job in campaign["jobs"]:
        path = Path(job["output"]) / "result.json"
        if not path.exists():
            rows.append({**job, "certified_cycles": 0, "margin": None, "complete": False})
            continue
        result = json.loads(path.read_text())
        cycles = result["cycles"]
        margin_path = path.parent / "margin-cycle-20.json"
        margin = json.loads(margin_path.read_text()) if margin_path.exists() else {}
        timings = [r["solve_wall_seconds"] for r in cycles[1:]]
        rows.append({**job, "certified_cycles": sum(r["certified"] for r in cycles),
            "complete": result["success"],
            "margin": margin.get("margin") if margin.get("audit", {}).get("passed") else None,
            "mean_solve_seconds_excluding_first": sum(timings) / len(timings) if timings else None,
            "prefix_observations": [{"offset": r["offset"], "completed_cycles": r["completed_cycles"],
                "terminal_normalized_A": r["terminal_normalized_A"],
                "terminal_slow_states": r["terminal_slow_states"],
                "maximum_normalized_violation": r["maximum_normalized_violation"],
                "solve_wall_seconds": r["solve_wall_seconds"]}
                for r in cycles if r["offset"] in (1, 5, 20)],
            "terminal_normalized_A": cycles[-1]["terminal_normalized_A"]})
    baseline = {(r["side"], r["source_cycle"]): r for r in rows if r["variant"] == "integral_quadratic"}
    for row in rows:
        base = baseline[row["side"], row["source_cycle"]]
        row["margin_delta_vs_integral_quadratic"] = row["margin"] - base["margin"] if row["margin"] is not None and base["margin"] is not None else None
    promoted = []
    for variant in VARIANTS[1:]:
        subset = [r for r in rows if r["variant"] == variant]
        left = [r for r in subset if r["side"] == "left"]
        if (all(r["complete"] and r["margin_delta_vs_integral_quadratic"] is not None and r["margin_delta_vs_integral_quadratic"] >= 0 for r in subset)
                and all(r["margin_delta_vs_integral_quadratic"] > .01 for r in left)):
            promoted.append(variant)
    reproduction = []
    for side in ("left", "right"):
        first = next(r for r in rows if r["side"] == side and r["source_cycle"] == 120 and r["variant"] == "integral_quadratic")
        later = next(r for r in rows if r["side"] == side and r["source_cycle"] == 140 and r["variant"] == "integral_quadratic")
        repeated = Path(first["output"]) / "cycle-20.npz"
        original = Path(later["receipt"]).parent / f"{side}.npz"
        if repeated.exists():
            protocol = json.loads((Path(first["output"]) / "protocol.json").read_text())
            model = json.loads(Path(protocol["model_path"]).read_text())
            with np.load(repeated, allow_pickle=False) as actual, np.load(original, allow_pickle=False) as expected:
                errors = {name: abs(float(actual[f"problem__x_bounds:A_{name}:min"][0, 0] - expected[f"problem__x_bounds:A_{name}:min"][0, 0])) / parameters["a_scale"]
                          for name, parameters in model["muscles"].items()}
            reproduction.append({"side": side, "scope": "normalized A after c120->140 compared with original archive c140",
                                 "max_absolute_error": max(errors.values()), "by_muscle": errors})
    _atomic_json(args.directory / "assessment.json", {"rows": rows, "promoted_variants": promoted,
        "baseline_archive_reproduction": reproduction,
        "promotion_rule": "All branches complete; left limiting arm gain >0.01 at both anchors; no right-arm regression",
        "endurance_claim_allowed": False})
    lines = ["# Ablation des coûts de fatigue aux checkpoints c120/c140", "",
        "Les quatre coûts sont évalués depuis les mêmes états, mécanique et historique de stimulation archivés. Chaque bras reçoit exactement 0,96 Nm équivalent (travail constant par cycle), à 30 Hz, Radau-5 et IPOPT/MA57. Les poids sont unitaires. Les sorties à 1, 5 et 20 cycles sont des préfixes emboîtés, pas des répétitions indépendantes.", "",
        "Le coefficient de fatigue reste celui de la source (10000 fois son coefficient multiplicatif). Les coûts terminaux sont multipliés par la durée du cycle. Aucun ajustement des pentes entre linéaire et quadratique n'est appliqué : cette ablation inclut donc un changement de magnitude marginale.", "",
        "| Départ | Bras | Coût | Cycles audités | Marge témoin finale | Δ marge | Temps solve moyen hors première résolution (s) |", "|---:|---|---|---:|---:|---:|---:|"]
    def fmt(value):
        return "indisponible" if value is None else f"{value:.5f}"
    for row in rows:
        lines.append(f"| {row['source_cycle']} | {row['side']} | {row['variant']} | {row['certified_cycles']} | {fmt(row['margin'])} | {fmt(row['margin_delta_vs_integral_quadratic'])} | {fmt(row['mean_solve_seconds_excluding_first'])} |")
    lines.extend(["", "Reproduction du témoin c120→140 face à l'archive historique c140 : " + "; ".join(f"{r['side']}, erreur absolue max A/a_scale = {r['max_absolute_error']:.3g}" for r in reproduction) + ".", "",
        "La marge est réoptimisée dans un processus frais à partir du checkpoint final exact. C'est une charge localement réalisable et auditée, pas une preuve de maximum global. Le temps solve inclut l'appel solve et exclut l'audit; la première résolution inclut la préparation du solveur. Les temps mesurés sous charge parallèle ne constituent pas une latence clinique garantie.", "",
        "Promotion vers une continuation longue : " + (", ".join(promoted) if promoted else "aucune variante ne franchit actuellement le critère pré-déclaré") + ".", "",
        "Une amélioration de fatigue ou de marge sur 20 cycles n'est pas une preuve de gain d'endurance. Un échec NLP dans ce protocole court est indéterminé. Le protocole de faisabilité gelée reste requis pour les continuations longues.", ""])
    (args.directory / "rapport_fr.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
