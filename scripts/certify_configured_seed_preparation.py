#!/usr/bin/env python3
"""Numerically certify one configured dynamic-RHO seed for a sealed campaign.

The script certifies the finite one-cycle seed solve only.  It does not infer
an endurance limit or certify a later weighted/FHO campaign.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")

from cocofest.optimization.configured_cycling_model import FINGERPRINT_KEY, require_seed_fingerprint, resolve_model_config
from scripts.build_resistance_comparison_campaign import INITIAL_GATES, artifact


def _read(path, label):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not read {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _same_number(actual, expected, label, *, tolerance=1e-12):
    if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual):
        raise ValueError(f"{label} must be finite")
    if not math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=tolerance):
        raise ValueError(f"{label}={actual!r}; expected {expected!r}")


def _required(mapping, key, label):
    if not isinstance(mapping, dict) or key not in mapping:
        raise ValueError(f"Missing {label}.{key}")
    return mapping[key]


def _validate_result(result_document, resistance_nm):
    configurations = _required(result_document, "configurations", "result")
    ipopt = _required(configurations, "ipopt", "result.configurations")
    if not isinstance(result_document.get("results"), list) or len(result_document["results"]) != 1:
        raise ValueError("Seed result must contain exactly one solver result")
    result = result_document["results"][0]
    expected_config = {
        "formulation": "dynamic", "mechanical_formulation": "reduced",
        "ipopt_linear_solver": "ma57", "ipopt_c_compile": False,
    }
    for key, expected in expected_config.items():
        if ipopt.get(key) != expected:
            raise ValueError(f"result.configurations.ipopt.{key} must be {expected!r}")
    _same_number(ipopt.get("constant_crank_torque"), resistance_nm, "constant_crank_torque")
    if ipopt.get("crank_torque_role") != "resistive":
        raise ValueError("Seed must use a positive resistive crank torque")
    if ipopt.get("cycles_per_window") != 1 or result.get("requested_cycles") != 1:
        raise ValueError("Seed certification is restricted to one requested RHO cycle")
    for key in ("success", "solver_success", "physical_success"):
        if result.get(key) is not True:
            raise ValueError(f"Seed result did not pass {key}")
    if result.get("solver") != "ipopt" or result.get("error") is not None:
        raise ValueError("Seed must be a successful IPOPT result without an error")
    if result.get("validated_cycles") != 1 or result.get("physically_validated_cycles") != 1:
        raise ValueError("Seed must physically validate its one requested cycle")
    stats = result.get("nlp_solver_stats")
    if not isinstance(stats, list) or len(stats) != 1 or stats[0].get("return_status") != "Solve_Succeeded":
        raise ValueError("IPOPT did not report Solve_Succeeded")
    if stats[0].get("success") is not True:
        raise ValueError("IPOPT solver statistics are not successful")
    windows = result.get("windows")
    if not isinstance(windows, list) or len(windows) != 1:
        raise ValueError("Seed must have exactly one RHO window")
    feasibility = _required(windows[0], "feasibility", "window")
    if feasibility.get("passes_tolerance") is not True or feasibility.get("trajectories_finite") is not True:
        raise ValueError("Seed window failed feasibility or finite-trajectory validation")
    if float(feasibility.get("maximum_bound_violation", math.inf)) > 1e-7:
        raise ValueError("Seed bound violation exceeds declared 1e-7 threshold")
    for diagnostic_name in ("nlp_crank_diagnostics", "physical_crank_diagnostics"):
        diagnostic = _required(result, diagnostic_name, "result")
        if diagnostic.get("is_physical") is not True or diagnostic.get("issues"):
            raise ValueError(f"{diagnostic_name} did not pass")
    mechanics = _required(result, "mechanical_equivalence_audit", "result")
    if any(mechanics.get(key) is not True for key in (
            "available", "passes_tolerance", "passes_configuration_tolerance",
            "passes_velocity_tolerance", "passes_physical_crank_velocity_bounds")):
        raise ValueError("Reduced mechanical replay validation did not pass")
    reserve = _required(result, "terminal_capacity_reserve", "result")
    if reserve.get("physiological_domain_valid") is not True or float(result.get("min_A_capacity_ratio", 0)) <= 0:
        raise ValueError("Seed leaves the physiological capacity domain")
    saturation = result.get("control_saturation")
    if not isinstance(saturation, list) or not saturation:
        raise ValueError("Pulse-width bounds are not reported")
    for entry in saturation:
        lower, maximum, upper = entry.get("lower"), entry.get("maximum"), entry.get("upper")
        if not all(isinstance(value, (int, float)) and math.isfinite(value)
                   for value in (lower, maximum, upper)) or lower > maximum + 1e-12 or maximum > upper + 1e-12:
            raise ValueError("Reported pulse-width bounds are violated")
    return {
        "solver_return_status": stats[0]["return_status"],
        "iterations": stats[0].get("iter_count"),
        "maximum_bound_violation": feasibility["maximum_bound_violation"],
        "minimum_capacity_ratio": result["min_A_capacity_ratio"],
        "maximum_configuration_projection_error_rad": mechanics["maximum_configuration_projection_error_rad"],
        "maximum_physical_velocity_bound_violation_rad_s": mechanics["maximum_physical_crank_velocity_bound_violation_rad_s"],
        "validated_cycles": result["physically_validated_cycles"],
    }


def certify(*, model_config, seed, result, configuration_audit, reduced_profile,
            resistance_nm, validation_report, certificate):
    """Validate immutable evidence and write a report plus campaign certificate."""
    destinations = [Path(validation_report).resolve(), Path(certificate).resolve()]
    if destinations[0] == destinations[1] or any(path.exists() for path in destinations):
        raise FileExistsError("Validation report and certificate must be distinct fresh paths")
    declared_model = _read(model_config, "model configuration")
    model = resolve_model_config(declared_model)
    seed_check = require_seed_fingerprint(seed, model[FINGERPRINT_KEY])
    audit = _read(configuration_audit, "configuration audit")
    if (audit.get("status") != "completed" or audit.get("condition") != "rho"
            or audit.get("weights_requested") is not False or audit.get("uses_fho_data_for_weights") is not False
            or audit.get("model", {}).get(FINGERPRINT_KEY) != model[FINGERPRINT_KEY]
            or not audit.get("model_builds")):
        raise ValueError("Configuration audit does not establish an unweighted configured RHO seed")
    metrics = _validate_result(_read(result, "seed result"), resistance_nm)
    model_record, seed_record, profile_record = artifact(model_config), artifact(seed), artifact(reduced_profile)
    evidence = [artifact(result), artifact(configuration_audit)]
    report = {
        "schema_version": 1, "stage": "per_model_seed_preparation_validation",
        "model_case_id": model["case_id"], FINGERPRINT_KEY: model[FINGERPRINT_KEY],
        "seed": seed_record, "model_config": model_record, "reduced_profile": profile_record,
        "resistance_nm": float(resistance_nm), "seed_check": seed_check,
        "thresholds": {"maximum_bound_violation": 1e-7, "minimum_capacity_ratio_exclusive": 0.0,
                       "requires_dynamic_reduced_ma57": True, "requires_one_physically_validated_cycle": True},
        "metrics": metrics, "validated": True,
        "scope": "finite configured one-cycle seed only; no endurance or FHO claim",
        "evidence": evidence,
    }
    destinations[0].parent.mkdir(parents=True, exist_ok=True)
    destinations[0].write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    certificate_document = {
        "schema_version": 1, "stage": "per_model_seed_preparation",
        FINGERPRINT_KEY: model[FINGERPRINT_KEY], "seed": seed_record,
        "model_config": model_record, "reduced_profile": profile_record,
        "resistance_nm": float(resistance_nm),
        "gates": {gate: True for gate in INITIAL_GATES},
        "evidence": [*evidence, artifact(destinations[0])],
        "scope": report["scope"],
    }
    destinations[1].parent.mkdir(parents=True, exist_ok=True)
    destinations[1].write_text(json.dumps(certificate_document, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report, certificate_document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--configuration-audit", required=True, type=Path)
    parser.add_argument("--reduced-profile", required=True, type=Path)
    parser.add_argument("--resistance-nm", required=True, type=float)
    parser.add_argument("--validation-report", required=True, type=Path)
    parser.add_argument("--certificate", required=True, type=Path)
    args = vars(parser.parse_args(argv))
    try:
        report, _ = certify(**args)
    except (OSError, ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(json.dumps({"validated": report["validated"], "metrics": report["metrics"],
                      "certificate": str(args["certificate"].resolve())}, indent=2))


if __name__ == "__main__":
    main()
