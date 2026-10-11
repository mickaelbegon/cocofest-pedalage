#!/usr/bin/env python3
"""Prepare and assess audited, slow-matched fast-state pair evidence.

This utility does not invent fast histories or launch a matching solve.  A
separate audited matching-RHO runner must produce admissible pairs.  In its
absence ``assess`` emits an explicit inconclusive report and refuses to issue
a fast-state gate receipt.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cocofest.cycling.slow.fast_state_gate import (
    assess_fast_state_pairs,
    digest,
    make_fast_state_gate_receipt,
    verify_preregistration,
)
from scripts.slow_layer import run_causal_torque_pilot as worker


def _write(path: Path, document: dict) -> None:
    worker.write_json(path, document)


def _sources(plan: dict) -> dict[str, str]:
    paths = set(ROOT.glob("cocofest/**/*.py")) | {Path(__file__).resolve()}
    for source in plan["sources"]:
        for artifact in source["artifacts"]:
            paths.add(Path(artifact["path"]))
    return {str(path): digest(path) for path in sorted(paths)}


def _verify_source_artifacts(plan: dict) -> None:
    for source in plan["sources"]:
        for artifact in source["artifacts"]:
            if digest(artifact["path"]) != artifact["sha256"]:
                raise ValueError(f"Preregistered source artifact changed: {artifact['path']}")


def prepare(preregistration: Path, output: Path) -> Path:
    preregistration = Path(preregistration).resolve(strict=True)
    plan = json.loads(preregistration.read_text())
    verify_preregistration(plan)
    _verify_source_artifacts(plan)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    protocol = {
        "schema": "slow-phi-fast-pair-validation-plan-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "preregistration": str(preregistration),
        "preregistration_sha256": digest(preregistration),
        "preregistration_plan_sha256": plan["plan_sha256"],
        "code_sha256": _sources(plan),
        "evidence_schema": "slow-phi-fast-pair-evidence-v1",
        "required": {
            "source_coverage": "one slow-matched, distinct-fast pair for every preregistered source",
            "controls": f"at least {plan['controls']['independent_repeats_per_action']} independently audited frozen controls for each source/action",
            "actions": plan["actions"]["left_transfer_nm"],
            "fast_groups": ["Cn", "F", "PW_history", "mechanics"],
            "outcome": "audited frozen-local endpoint with finite weighted-force integral per declared muscle",
            "decision_rule": "no pairwise action-ranking change and no endpoint/feasibility change beyond stratum control resolution",
        },
        "not_a_claim": "Passing is a finite paired local robustness gate only. It never establishes Markov closure, global state sufficiency, clinical safety, or online activation.",
        "activation_allowed": False,
    }
    _write(output / "validation-plan.json", protocol)
    _write(output / "validation-plan-seal.json", {"validation_plan_sha256": digest(output / "validation-plan.json")})
    return output / "validation-plan.json"


def _authenticated(path: Path) -> tuple[dict, dict]:
    path = Path(path).resolve(strict=True)
    seal = json.loads((path.parent / "validation-plan-seal.json").read_text())
    if seal.get("validation_plan_sha256") != digest(path):
        raise ValueError("Validation plan changed after preparation")
    document = json.loads(path.read_text())
    if document.get("schema") != "slow-phi-fast-pair-validation-plan-v1":
        raise ValueError("Unexpected validation plan")
    for source, expected in document["code_sha256"].items():
        if digest(source) != expected:
            raise ValueError(f"Validation dependency changed: {source}")
    prereg = Path(document["preregistration"]).resolve(strict=True)
    if digest(prereg) != document["preregistration_sha256"]:
        raise ValueError("Preregistration file changed")
    plan = json.loads(prereg.read_text())
    if plan.get("plan_sha256") != document["preregistration_plan_sha256"]:
        raise ValueError("Preregistration contents changed")
    verify_preregistration(plan)
    _verify_source_artifacts(plan)
    return document, plan


def _bilateral_anchors(plan: dict) -> dict[str, dict[str, str]]:
    """Bind every left-side fast probe to the exact bilateral checkpoint pair."""
    from scripts.probe_independent_rho_task_reserve import _load_arm

    bindings = {}
    for source in plan["sources"]:
        receipt = Path(source["receipt"]).resolve(strict=True)
        receipt_sha = next(item["sha256"] for item in source["artifacts"] if item["path"] == str(receipt))
        models = {}
        for side in ("left", "right"):
            arm, _ = _load_arm(receipt, side)
            arm.verify_files()
            models[side] = arm.model_sha256
        bindings[receipt_sha] = models
    return bindings


def assess(validation_plan: Path, evidence: Path | None, output: Path, *, gate_output: Path | None = None) -> dict:
    _, prereg = _authenticated(validation_plan)
    if evidence is None:
        evidence_doc = {"schema": "slow-phi-fast-pair-evidence-v1", "plan_sha256": prereg["plan_sha256"],
                        "controls": [], "pairs": []}
    else:
        evidence_doc = json.loads(Path(evidence).resolve(strict=True).read_text())
    report = assess_fast_state_pairs(prereg, evidence_doc)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    _write(output, report)
    if gate_output is not None:
        if report["status"] != "validated_local_robustness":
            raise ValueError("Evidence is not sufficient for a fast-state gate receipt")
        gate_output = Path(gate_output).resolve()
        if gate_output.exists():
            raise FileExistsError(gate_output)
        _write(gate_output, make_fast_state_gate_receipt(report, _bilateral_anchors(prereg)))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "assess"))
    parser.add_argument("--preregistration", type=Path)
    parser.add_argument("--validation-plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--gate-output", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.preregistration is None:
            parser.error("--preregistration is required for prepare")
        print(prepare(args.preregistration, args.output))
    else:
        if args.validation_plan is None:
            parser.error("--validation-plan is required for assess")
        result = assess(args.validation_plan, args.evidence, args.output, gate_output=args.gate_output)
        print(json.dumps({"status": result["status"], "fast_state_gate": result["fast_state_gate"],
                          "closure_established": False, "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
