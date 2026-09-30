"""Read-only indexing of benchmark artifacts without importing scientific solvers.

The catalog records reported evidence; it does not certify a simulation. Missing
fields stay missing, and source files are never migrated or rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

CATALOG_SCHEMA_VERSION = 1
_METRICS = (
    "requested_cycles",
    "requested_windows",
    "attempted_windows",
    "successful_windows",
    "validated_windows",
    "validated_cycles",
    "physically_validated_cycles",
    "solver_time_s",
    "wall_time_s",
    "end_to_end_wall_time_s",
    "solver_time_per_cycle_s",
    "wall_time_per_cycle_s",
    "hot_window_count",
    "hot_solver_time_median_s",
    "hot_solver_time_p90_s",
    "hot_wall_time_median_s",
    "hot_wall_time_p90_s",
    "hot_complete_iteration_wall_time_median_s",
    "hot_complete_iteration_wall_time_p90_s",
    "target_solver_only_hot_solver_time_mean_s",
    "target_solver_only_hot_solver_time_median_s",
    "target_solver_only_hot_solver_time_p90_s",
    "reduced_profile_build_time_s",
    "initial_guess_preparation_time_s",
    "execution_timing",
    "timing_populations",
)
_AUDITS = (
    "audit_registry",
    "mechanical_equivalence_audit",
    "high_accuracy_trace_rollout",
    "high_accuracy_cycle_milestones",
    "integrator_map_final_solution",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _frequency(config: dict) -> tuple[float | None, str | None]:
    for field in ("stimulation_frequency_hz", "frequency_hz", "stimulation_frequency"):
        value = config.get(field)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value > 0
        ):
            return float(value), field
    interval = config.get("calcium_stimulation_interval_s")
    if (
        isinstance(interval, (int, float))
        and not isinstance(interval, bool)
        and math.isfinite(interval)
        and interval > 0
    ):
        return 1.0 / interval, "1/calcium_stimulation_interval_s"
    # A stimulation count is not a frequency unless cycle duration is known.
    count, duration = config.get("stimulations_per_cycle"), config.get("cycle_duration")
    if (
        isinstance(count, (int, float))
        and isinstance(duration, (int, float))
        and not isinstance(count, bool)
        and not isinstance(duration, bool)
    ):
        if (
            math.isfinite(count)
            and math.isfinite(duration)
            and count > 0
            and duration > 0
        ):
            return count / duration, "stimulations_per_cycle/cycle_duration"
    return None, None


def _stage(config: dict, source: Path) -> tuple[str | None, str | None]:
    for field in ("stage", "benchmark_stage", "ablation_stage"):
        if isinstance(config.get(field), str):
            return config[field], field
    for part in reversed(source.parent.parts):
        if re.fullmatch(r"K\d+(?:[_-].+)?", part):
            return part, "path_component"
    return None, None


def _reported_status(row: dict) -> str:
    if row.get("error"):
        return "error"
    if row.get("success") is True:
        return "reported_success"
    if row.get("success") is False:
        return "reported_failure"
    return "unknown"


def _configuration_evidence(source: Path) -> tuple[dict, dict, dict | None]:
    """Read only the sidecar published alongside this result, never ancestors.

    A campaign-wide sidecar could describe a different solver or case. A valid
    canonical hash verifies bytes/inputs, not scientific or ABI equivalence.
    """
    sidecar = source.with_name("effective-configuration.json")
    evidence = {"status": "not_recorded", "uri": None, "sha256": None}
    if not sidecar.exists():
        return evidence, {}, None
    evidence["uri"] = sidecar.resolve().as_uri()
    try:
        raw = sidecar.read_bytes()
        evidence["sha256"] = hashlib.sha256(raw).hexdigest()
        payload = json.loads(raw)
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("Unsupported effective configuration schema")
        effective = payload.get("effective")
        if not isinstance(effective, dict) or any(
            not isinstance(effective.get(group), dict)
            for group in ("physical", "transcription", "solver", "execution")
        ):
            raise ValueError("Effective configuration requires four object sections")
        canonical = json.dumps(
            effective, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        computed_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        evidence.update(
            config_hash=payload.get("config_hash"),
            computed_config_hash=computed_hash,
            profile=payload.get("profile"),
            schema_version=payload["schema_version"],
        )
        if payload.get("config_hash") != computed_hash:
            raise ValueError("Effective configuration config_hash mismatch")
        evidence["status"] = "hash_verified"
        # Keep the source sections for provenance, including model and seed
        # references. Referenced files are never opened or claimed as verified.
        evidence["effective"] = effective
        flattened = {
            key: value
            for name in ("physical", "transcription", "solver", "execution")
            for key, value in effective[name].items()
        }
        return evidence, flattened, None
    except (OSError, ValueError, TypeError) as exc:
        evidence["status"] = "invalid"
        evidence["error"] = str(exc)
        return (
            evidence,
            {},
            {"uri": evidence["uri"], "sha256": evidence["sha256"], "error": str(exc)},
        )


def _identity(source: Path, roots: list[Path], document: dict, row: dict) -> dict:
    candidates = [
        root for root in roots if root.is_dir() and source.is_relative_to(root)
    ]
    root = (
        max(candidates, key=lambda path: len(path.parts))
        if candidates
        else source.parent
    )
    identity = {
        "campaign": root.name,
        "case": source.parent.relative_to(root).as_posix(),
        "campaign_root_uri": root.as_uri(),
        "sources": {"campaign": "index_root", "case": "relative_result_directory"},
    }
    for field in ("campaign", "case"):
        for label, payload in (("result_entry", row), ("result_document", document)):
            value = payload.get(f"{field}_id")
            if isinstance(value, str) and value:
                identity[field] = value
                identity["sources"][field] = f"{label}.{field}_id"
                break
    return identity


def index_campaigns(roots: list[str | Path]) -> dict:
    """Build a deterministic catalog from result.json files under each root.

    One record corresponds to one entry in ``results`` (a list or keyed map),
    or to a standalone result object. Parse failures are retained in ``errors``.
    Hashes cover original file bytes, including all unindexed fields.
    """
    paths: set[Path] = set()
    errors = []
    resolved_roots = sorted({Path(root).resolve() for root in roots})
    for root in resolved_roots:
        if root.is_file():
            paths.add(root)
        elif root.is_dir():
            paths.update(
                path.resolve() for path in root.rglob("result.json") if path.is_file()
            )
        else:
            errors.append({"uri": root.as_uri(), "error": "Root does not exist"})
    records = []
    for source in sorted(paths):
        digest = None
        try:
            raw = source.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            document = json.loads(raw)
            if not isinstance(document, dict):
                raise ValueError("Result document must be a JSON object")
            results = document.get("results", [document])
            if isinstance(results, dict):
                entries = list(results.items())
            elif isinstance(results, list):
                entries = list(enumerate(results))
            else:
                raise ValueError("results must be a list or object")
            if not entries:
                raise ValueError("results contains no entries")
            config_evidence, effective_config, config_error = _configuration_evidence(
                source
            )
            if config_error:
                errors.append(config_error)
            configurations = document.get("configurations", {})
            for key, row in entries:
                if not isinstance(row, dict):
                    errors.append(
                        {
                            "uri": source.as_uri(),
                            "sha256": digest,
                            "entry": key,
                            "error": "Result entry must be a JSON object",
                        }
                    )
                    continue
                solver = row.get("solver", key if isinstance(key, str) else None)
                config = (
                    configurations.get(solver, {})
                    if isinstance(configurations, dict)
                    else {}
                )
                if not isinstance(config, dict):
                    config = {}
                if not config and isinstance(row.get("configuration"), dict):
                    config = row["configuration"]
                # The launcher's sidecar is only usable for its recorded solver.
                # Comparison documents may contain several unrelated solvers.
                compatible = bool(effective_config) and solver in (
                    None,
                    effective_config.get("solver"),
                )
                fallback = effective_config if compatible else {}
                conflicts = {
                    field: {"result": value, "effective_configuration": fallback[field]}
                    for field, value in config.items()
                    if field in fallback and value != fallback[field]
                }
                if solver is None:
                    solver = fallback.get("solver")
                config = {**fallback, **config}
                frequency, frequency_source = _frequency(config)
                stage, stage_source = _stage(config, source)
                if stage_source == "path_component" or stage is None:
                    profile = config_evidence.get("profile") if compatible else None
                    if isinstance(profile, str) and profile:
                        stage, stage_source = profile, "effective_configuration.profile"
                records.append(
                    _json_safe(
                        {
                            "id": f"{source.as_uri()}#results/{key}",
                            "artifact": {
                                "uri": source.as_uri(),
                                "sha256": digest,
                                "size_bytes": len(raw),
                                "result_entry": key,
                                "result_schema_version": document.get("schema_version"),
                            },
                            "identity": _identity(
                                source, resolved_roots, document, row
                            ),
                            "configuration_provenance": {
                                **config_evidence,
                                "solver_matches_result": (
                                    compatible if effective_config else None
                                ),
                                "conflicts": conflicts,
                                "dimension_precedence": "result_configuration_then_verified_sidecar",
                            },
                            "dimensions": {
                                "solver": solver,
                                "linear_solver": config.get(f"{solver}_linear_solver"),
                                "formulation": config.get("formulation"),
                                "mechanical_formulation": config.get(
                                    "mechanical_formulation", fallback.get("mechanics")
                                ),
                                "ding_formulation": config.get(
                                    "calcium_forcing_formulation"
                                ),
                                "frequency_hz": frequency,
                                "stage": stage,
                            },
                            "dimension_sources": {
                                "frequency_hz": frequency_source,
                                "stage": stage_source,
                            },
                            "status": {
                                "catalog_status": _reported_status(row),
                                **{
                                    field: row[field]
                                    for field in (
                                        "status",
                                        "error",
                                        "success",
                                        "solver_success",
                                        "physical_success",
                                    )
                                    if field in row
                                },
                            },
                            "metrics": {
                                field: row[field] for field in _METRICS if field in row
                            },
                            "audits": {
                                field: row[field] for field in _AUDITS if field in row
                            },
                        }
                    )
                )
        except (OSError, ValueError, TypeError) as exc:
            errors.append({"uri": source.as_uri(), "sha256": digest, "error": str(exc)})
    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "roots": [root.as_uri() for root in resolved_roots],
        "scanned_file_count": len(paths),
        "records": records,
        "errors": errors,
    }


def query_catalog(
    catalog: dict,
    *,
    solver: str | None = None,
    formulation: str | None = None,
    frequency_hz: float | None = None,
    stage: str | None = None,
    campaign: str | None = None,
    case: str | None = None,
    config_hash: str | None = None,
) -> list[dict]:
    """Filter observed dimensions, never interpreting missing values as matches."""
    if catalog.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError("Unsupported experiment catalog schema version")

    def matches(record: dict) -> bool:
        for key, requested in (("campaign", campaign), ("case", case)):
            if (
                requested is not None
                and record.get("identity", {}).get(key) != requested
            ):
                return False
        provenance = record.get("configuration_provenance", {})
        if config_hash is not None and (
            provenance.get("status") != "hash_verified"
            or provenance.get("solver_matches_result") is not True
            or bool(provenance.get("conflicts"))
            or provenance.get("config_hash") != config_hash
        ):
            return False
        dimensions = record["dimensions"]
        for key, requested in (
            ("solver", solver),
            ("formulation", formulation),
            ("stage", stage),
        ):
            if requested is not None and dimensions.get(key) != requested:
                return False
        observed = dimensions.get("frequency_hz")
        return frequency_hz is None or (
            observed is not None and math.isclose(observed, frequency_hz, rel_tol=1e-9)
        )

    return [record for record in catalog["records"] if matches(record)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    index = subparsers.add_parser(
        "index", help="Index artifacts; prints JSON unless --output is supplied"
    )
    index.add_argument("roots", nargs="+")
    index.add_argument("--output", type=Path)
    index.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the complete proposed catalog without writing",
    )
    query = subparsers.add_parser("query", help="Query an existing catalog")
    query.add_argument("catalog", type=Path)
    query.add_argument("--solver")
    query.add_argument("--formulation")
    query.add_argument("--frequency-hz", type=float)
    query.add_argument("--stage")
    query.add_argument("--campaign")
    query.add_argument("--case")
    query.add_argument("--config-hash")
    args = parser.parse_args(argv)
    if args.command == "query":
        catalog = json.loads(args.catalog.read_text())
        output = query_catalog(
            catalog,
            solver=args.solver,
            formulation=args.formulation,
            frequency_hz=args.frequency_hz,
            stage=args.stage,
            campaign=args.campaign,
            case=args.case,
            config_hash=args.config_hash,
        )
        print(json.dumps(output, indent=2, allow_nan=False))
        return 0
    catalog = index_campaigns(args.roots)
    serialized = json.dumps(catalog, indent=2, allow_nan=False) + "\n"
    if args.output is None or args.dry_run:
        print(serialized, end="")
    else:
        destination = args.output.resolve()
        if (
            destination.name in ("result.json", "effective-configuration.json")
            or any(
                destination.as_uri()
                in (
                    record["artifact"]["uri"],
                    record["configuration_provenance"].get("uri"),
                )
                for record in catalog["records"]
            )
            or any(destination.as_uri() == error["uri"] for error in catalog["errors"])
        ):
            parser.error(
                "Catalog output must not overwrite a result or configuration artifact"
            )
        # Atomic publication protects readers when a campaign is being indexed.
        with tempfile.NamedTemporaryFile(
            mode="w", dir=destination.parent, prefix=".catalog-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
    return 1 if catalog["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
