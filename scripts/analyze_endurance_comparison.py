#!/usr/bin/env python3
"""Produce a conservative, reproducible comparison of endurance campaigns.

The repository contains several generations of RHO campaigns.  Their output
schemas deliberately differ: a cold two-arm run writes ``summary.json``, a
continuous causal run writes a live ``progress.json`` until it has a terminal
receipt, and checkpoint pilots write one record per branch.  This reader makes
those differences visible rather than pretending that every archive is an
identical endurance trial.

In particular, an IPOPT error, a missing file, or a completed solver summary
without an explicit frozen-feasibility certificate is *not* reported as a
physiological endpoint.  The resulting JSON and Markdown files are therefore
safe to refresh while a campaign is still executing.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FACTORIAL = ROOT / "asymmetric-sides-r192-rho-bo-20260928/factorial-weights-split-20261003"
DEFAULT_SLOW = ROOT / "slow-layer-sim-20261003"


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        stream.write(value)
        temporary = Path(stream.name)
    temporary.replace(path)


def _timing(values: Iterable[object]) -> dict[str, float] | None:
    sample = sorted(value for raw in values if (value := _finite(raw)) is not None)
    if not sample:
        return None
    def percentile(q: float) -> float:
        index = (len(sample) - 1) * q
        lower, upper = int(math.floor(index)), int(math.ceil(index))
        return sample[lower] + (sample[upper] - sample[lower]) * (index - lower)
    return {"count": len(sample), "mean_s": math.fsum(sample) / len(sample),
            "median_s": percentile(.5), "p90_s": percentile(.9),
            "min_s": sample[0], "max_s": sample[-1]}


def _shared_rows(cycles: object) -> list[dict[str, Any]]:
    """Return only rows with both arms explicitly marked certified."""
    if not isinstance(cycles, list):
        return []
    rows: list[dict[str, Any]] = []
    for row in cycles:
        if not isinstance(row, dict) or not isinstance(row.get("cycle"), int):
            continue
        arms = row.get("arms")
        if not isinstance(arms, dict) or not all(isinstance(arms.get(side), dict) and arms[side].get("certified") is True
                                                  for side in ("left", "right")):
            continue
        rows.append(row)
    return sorted(rows, key=lambda row: row["cycle"])


def _work_from_shared(rows: list[dict[str, Any]]) -> tuple[float | None, int]:
    work: list[float] = []
    for row in rows:
        arms = row["arms"]
        left = _finite(arms["left"].get("achieved_work_j_per_cycle", arms["left"].get("achieved_work_j")))
        right = _finite(arms["right"].get("achieved_work_j_per_cycle", arms["right"].get("achieved_work_j")))
        if left is None or right is None:
            continue
        work.append(left + right)
    return (math.fsum(work) if work else None), len(work)


def _limiting_sides(arms: object) -> list[str]:
    if not isinstance(arms, dict):
        return []
    return [side for side in ("left", "right") if isinstance(arms.get(side), dict)
            and arms[side].get("status") == "physiological_failure_certified"]


def _base(label: str, source: Path, *, scope: str, state: str) -> dict[str, Any]:
    return {"label": label, "source": str(source.resolve()), "scope": scope, "state": state,
            "last_shared_certified_cycle": None, "endpoint": {"classification": "not_available", "certified": False,
            "cycle": None, "limiting_sides": []}, "bilateral_work_j": None, "work_cycle_count": 0,
            "rho_timing": None, "trigger_events": [], "fast_state_status": "not_reported",
            "weights_status": "not_reported", "limitations": []}


def _summary_record(label: str, path: Path, data: dict[str, Any]) -> dict[str, Any]:
    record = _base(label, path, scope="cold_start", state="available")
    rows = _shared_rows(data.get("cycles"))
    record["last_shared_certified_cycle"] = rows[-1]["cycle"] if rows else None
    work, count = _work_from_shared(rows)
    record["bilateral_work_j"], record["work_cycle_count"] = work, count
    timing = data.get("pair_cycle_wall_timing_s")
    if isinstance(timing, dict):
        record["rho_timing"] = {"kind": "pair_cycle_wall", **{key: value for key, value in timing.items() if _finite(value) is not None}}
    else:
        record["rho_timing"] = _timing(row.get("pair_solve_and_transfer_wall_time_s") for row in rows)
    record["weights_status"] = "reported_by_arm_cycles" if rows else "not_available"
    if data.get("success") is not True:
        record["limitations"].append("No explicit frozen-feasibility terminal certificate in summary; endpoint intentionally not inferred.")
    record["limitations"].append("Cold-start campaign summary: work covers only shared rows explicitly marked certified.")
    return record


def _continuous_record(label: str, root: Path) -> dict[str, Any]:
    manifest_path, completion_path, progress_path = root / "manifest.json", root / "campaign-completion.json", root / "progress.json"
    if not manifest_path.is_file():
        return _base(label, root, scope="cold_start", state="missing_manifest")
    manifest = _read(manifest_path)
    data_path = completion_path if completion_path.is_file() else progress_path
    if not data_path.is_file():
        record = _base(label, root, scope="cold_start", state="pending")
        record["fast_state_status"] = str(manifest.get("fast_state_status", "not_reported"))
        record["limitations"].append("No progress or completion receipt yet.")
        return record
    data = _read(data_path)
    record = _base(label, root, scope="cold_start", state="completed" if completion_path.is_file() else "in_progress")
    rows = _shared_rows(data.get("cycles"))
    record["last_shared_certified_cycle"] = rows[-1]["cycle"] if rows else None
    work, count = _work_from_shared(rows)
    record["bilateral_work_j"], record["work_cycle_count"] = work, count
    timing = data.get("pair_cycle_wall_timing_s")
    record["rho_timing"] = (dict(timing, kind="pair_cycle_wall") if isinstance(timing, dict)
                            else _timing(row.get("pair_solve_and_transfer_wall_time_s") for row in rows))
    record["fast_state_status"] = str(data.get("fast_state_status", manifest.get("fast_state_status", "not_reported")))
    decisions = data.get("decisions")
    if isinstance(decisions, list):
        record["trigger_events"] = [{key: decision.get(key) for key in
                                     ("completed_cycles", "phase", "left_torque_nm", "right_torque_nm", "reason", "fast_state_status")}
                                    for decision in decisions if isinstance(decision, dict) and decision.get("phase") not in (None, "hold")]
    record["weights_status"] = "fixed_unit" if isinstance(manifest.get("payload"), dict) and isinstance(manifest["payload"].get("muscle_pace"), dict) and manifest["payload"]["muscle_pace"].get("adaptation_enabled") is False else "not_reported"
    if completion_path.is_file():
        status = data.get("status")
        exact = data.get("exact_label_allowed") is True and status == "physiological_failure_certified"
        record["endpoint"] = {"classification": "frozen_feasibility_exhaustion" if exact else str(status or "unresolved"),
                              "certified": exact, "cycle": data.get("last_certified_cycle") if exact else None,
                              "limiting_sides": _limiting_sides((data.get("terminal") or {}).get("arms"))}
        if not exact:
            record["limitations"].append("Completion exists but does not provide an exact frozen-feasibility endpoint.")
    else:
        record["limitations"].append("Live progress is a certified prefix, not an endpoint.")
    if record["fast_state_status"] != "validated":
        record["limitations"].append("Fast/history-state closure is not validated; this record is not clinical activation evidence.")
    return record


def _phase_rows(result_path: Path) -> dict[int, dict[str, dict[str, Any]]]:
    """Read available phase rows for one checkpoint branch, without auditing them."""
    rows: dict[int, dict[str, dict[str, Any]]] = {}
    for side in ("left", "right"):
        candidates = [result_path]
        if result_path.parent.name == "continuation":
            candidates.append(result_path.parent.parent / "intervention" / "result.json")
        for candidate in candidates:
            if not candidate.is_file():
                continue
            try:
                payload = _read(candidate)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            for row in payload.get("cycles", []):
                if isinstance(row, dict) and isinstance(row.get("cycle"), int) and row.get("certified") is True:
                    rows.setdefault(row["cycle"], {})[side] = row
    return rows


def _checkpoint_records(label: str, path: Path, data: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = data.get("jobs")
    if not isinstance(jobs, list):
        record = _base(label, path, scope="checkpoint_continuation", state="unusable")
        record["limitations"].append("Campaign completion has no jobs list.")
        return [record]
    result = []
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            continue
        job_id = str(job.get("job_id", f"job-{index + 1}"))
        record = _base(f"{label}/{job_id}", path, scope="checkpoint_continuation", state="completed" if data.get("completed") is True else "partial")
        endpoint_exact = job.get("status") == "physiological_failure_certified" and job.get("exact_label_allowed") is True
        record["endpoint"] = {"classification": "frozen_feasibility_exhaustion" if endpoint_exact else str(job.get("status", "unresolved")),
                              "certified": endpoint_exact, "cycle": job.get("last_certified_cycle") if endpoint_exact else None,
                              "limiting_sides": list(job.get("limiting_sides", _limiting_sides(job.get("arms"))))}
        record["last_shared_certified_cycle"] = job.get("last_certified_cycle") if isinstance(job.get("last_certified_cycle"), int) else None
        phase: dict[int, dict[str, dict[str, Any]]] = {}
        if isinstance(job.get("arms"), dict):
            for side in ("left", "right"):
                arm = job["arms"].get(side)
                if isinstance(arm, dict) and isinstance(arm.get("result"), str):
                    source = _phase_rows(Path(arm["result"]))
                    for cycle, sides in source.items():
                        phase.setdefault(cycle, {}).update(sides)
        common = [sides for _, sides in sorted(phase.items()) if set(sides) == {"left", "right"}]
        values = []
        for sides in common:
            left = _finite(sides["left"].get("achieved_work_j", sides["left"].get("achieved_work_j_per_cycle")))
            right = _finite(sides["right"].get("achieved_work_j", sides["right"].get("achieved_work_j_per_cycle")))
            if left is not None and right is not None:
                values.append(left + right)
        record["bilateral_work_j"], record["work_cycle_count"] = (math.fsum(values) if values else None), len(values)
        record["rho_timing"] = _timing(row.get("solver_time_s") for sides in common for row in sides.values())
        record["trigger_events"] = [{"anchor_cycle": job.get("anchor_cycle"), "delta_left_nm": job.get("delta_left_nm")}]
        record["fast_state_status"] = "not_reported_by_checkpoint_completion"
        record["weights_status"] = "not_reported_by_checkpoint_completion"
        record["limitations"].append("Checkpoint continuation; it is not an uninterrupted cold-start RHO trial.")
        record["limitations"].append("Phase rows are reported certified but are not re-authenticated by this comparison reader.")
        result.append(record)
    return result


def collect_source(label: str, source: Path) -> list[dict[str, Any]]:
    """Normalize a source path.  Missing sources become explicit pending rows."""
    source = source.resolve()
    if not source.exists():
        record = _base(label, source, scope="unknown", state="missing")
        record["limitations"].append("Source does not exist; no result is inferred.")
        return [record]
    if source.is_dir():
        if (source / "results" / "summary.json").is_file():
            return [_summary_record(label, source / "results" / "summary.json", _read(source / "results" / "summary.json"))]
        if (source / "summary.json").is_file():
            return [_summary_record(label, source / "summary.json", _read(source / "summary.json"))]
        if (source / "manifest.json").is_file() and (_read(source / "manifest.json").get("kind") == "continuous_causal_left_to_right_transfer_rho_benchmark"):
            return [_continuous_record(label, source)]
        if (source / "campaign-completion.json").is_file():
            return _checkpoint_records(label, source / "campaign-completion.json", _read(source / "campaign-completion.json"))
        record = _base(label, source, scope="unknown", state="pending")
        record["limitations"].append("No recognized summary, continuous manifest, or campaign completion receipt.")
        return [record]
    data = _read(source)
    if source.name == "summary.json":
        return [_summary_record(label, source, data)]
    if source.name == "campaign-completion.json":
        return _checkpoint_records(label, source, data)
    record = _base(label, source, scope="unknown", state="unrecognized")
    record["limitations"].append("Recognized JSON syntax but unsupported campaign schema.")
    return [record]


def default_sources() -> dict[str, Path]:
    sources = {"RHO_unitaire": DEFAULT_FACTORIAL / "unit_fixed",
               "RHO_BO_fixe": DEFAULT_FACTORIAL / "bo_fixed",
               "RHO_unitaire_adaptatif": DEFAULT_FACTORIAL / "unit_adaptive",
               "RHO_BO_adaptatif": DEFAULT_FACTORIAL / "bo_adaptive",
               "RHO_physio": ROOT / "asymmetric-sides-r192-rho-bo-20260928/physio-results-rerun",
               "RHO_physio_U": ROOT / "asymmetric-sides-r192-rho-bo-20260928/physio-u-results-rerun",
               "RHO_PACE_VR": ROOT / "asymmetric-sides-r192-rho-bo-20260928/pace-vr-until-failure-3000-20260930/results",
               "pilote_dose_duree": DEFAULT_SLOW / "torque-transfer-dose-duration",
               "superviseur_par_reprise": DEFAULT_SLOW / "causal-transfer-supervisor-experimental"}
    for name in ("continuous-causal-transfer-experimental-rerun1", "continuous-causal-transfer-experimental"):
        path = DEFAULT_SLOW / name
        if path.exists():
            sources[f"RHO_continu_{name.rsplit('-', 1)[-1]}"] = path
    return sources


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Comparaison d’endurance RHO", "", "Ce rapport ne déduit jamais un échec physiologique d’un arrêt numérique ou d’un fichier absent.",
             "Les reprises depuis un checkpoint sont listées séparément des essais RHO démarrant à froid.", ""]
    for scope, title in (("cold_start", "RHO depuis le repos"), ("checkpoint_continuation", "Reprises depuis checkpoint"), ("unknown", "Sources en attente ou non reconnues")):
        rows = [row for row in report["records"] if row["scope"] == scope]
        if not rows:
            continue
        lines += [f"## {title}", "", "| Condition | État | Dernier cycle certifié | Endpoint | Travail bilatéral [J] | Bras limitant | Temps RHO médian [s] | Déclenchement | États rapides |", "|---|---|---:|---|---:|---|---:|---|---|"]
        for row in rows:
            endpoint = row["endpoint"]
            endpoint_text = endpoint["classification"] if endpoint["certified"] else f"non certifié: {endpoint['classification']}"
            timing = row.get("rho_timing") or {}
            trigger = "; ".join(str(event) for event in row["trigger_events"]) if row["trigger_events"] else "—"
            def fmt(value: object, digits: int = 2) -> str:
                return "—" if _finite(value) is None else f"{float(value):.{digits}f}"
            lines.append("| " + " | ".join((row["label"], row["state"], str(row["last_shared_certified_cycle"] or "—"), endpoint_text,
                                               fmt(row["bilateral_work_j"]), ", ".join(endpoint["limiting_sides"]) or "—",
                                               fmt(timing.get("median_s", timing.get("median"))), trigger, row["fast_state_status"])) + " |")
        lines.append("")
    lines += ["## Limites de lecture", ""]
    unique = sorted({limit for row in report["records"] for limit in row["limitations"]})
    lines.extend(f"- {limit}" for limit in unique)
    lines.append("")
    return "\n".join(lines)


def build_report(sources: dict[str, Path]) -> dict[str, Any]:
    records = [record for label, path in sources.items() for record in collect_source(label, path)]
    for record in records:
        record["limitations"].append("Cross-archive mechanical/model equivalence is not verified here; comparisons are descriptive until protocol contracts are matched.")
    return {"schema_version": 1, "purpose": "descriptive comparison only; no endpoint is inferred", "records": records}


def _source_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("Use nonempty LABEL=PATH")
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", default=[], type=_source_argument, metavar="LABEL=PATH",
                        help="Add or replace an archived campaign source. Repeatable.")
    parser.add_argument("--output", type=Path, default=DEFAULT_SLOW / "endurance-comparison.json")
    parser.add_argument("--markdown", type=Path, default=DEFAULT_SLOW / "endurance-comparison.md")
    args = parser.parse_args()
    sources = default_sources()
    sources.update(dict(args.source))
    report = build_report(sources)
    _atomic_write(args.output, json.dumps(report, indent=2, allow_nan=False) + "\n")
    _atomic_write(args.markdown, render_markdown(report))
    print(json.dumps({"records": len(report["records"]), "output": str(args.output), "markdown": str(args.markdown)}, indent=2))


if __name__ == "__main__":
    main()
