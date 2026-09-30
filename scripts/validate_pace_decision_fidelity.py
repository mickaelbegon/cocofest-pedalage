#!/usr/bin/env python3
"""Test whether archived PACE decisions agree with short real RHO continuations.

The PACE slow prediction is deliberately *not* changed here.  This script
uses the immutable ``rho_replay_checkpoint`` saved after a certified RHO
boundary, freezes each candidate's muscle weights, and launches matched
one-cycle RHO continuations.  It therefore tests the decision made by PACE,
rather than comparing a compact rollout with a different initial state.

The output has two stages:

* ``--prepare`` writes a manifest, immutable weight files and exact commands;
* ``--execute`` runs the pending commands and writes a machine-readable
  ranking/regret summary (and a small PNG when matplotlib is available).

PACE workers and their pending asynchronous proposals are intentionally not
restored.  A prepared checkpoint is an exact primal for the *next* RHO, but
not a complete controller checkpoint.  Consequently every continuation here
uses ``rho-physio`` with one frozen candidate weight vector.  This is the
controlled experiment needed to evaluate a single PACE decision.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import NamedTemporaryFile
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MUSCLE_NAMES = ("Delt_ant", "Delt_post", "Biceps", "Triceps")
DEFAULT_CHECKPOINTS = (60, 180, 300, 420)
DEFAULT_HORIZONS = (1, 5, 20)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _read_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return document


def _write_new_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def _weights(value: Any, *, source: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != len(MUSCLE_NAMES):
        raise ValueError(f"{source} must contain {len(MUSCLE_NAMES)} weights")
    converted = tuple(_finite(item) for item in value)
    if any(item is None or item <= 0 for item in converted):
        raise ValueError(f"{source} must contain finite positive weights")
    return tuple(float(item) for item in converted)


def _load_journal(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Journal record {line_number} is not an object: {path}")
        records.append(value)
    return records


def _boundary_weights(records: Iterable[dict[str, Any]], checkpoint: int) -> tuple[float, ...]:
    """Weights already connected to the OCP at the saved next-RHO primal."""
    for record in records:
        if record.get("event") == "boundary" and record.get("cycle_index") == checkpoint:
            return _weights(record.get("weights"), source=f"PACE boundary {checkpoint}")
    raise ValueError(f"PACE journal has no boundary record at checkpoint {checkpoint}")


def _pace_proposal(records: Iterable[dict[str, Any]], checkpoint: int) -> tuple[tuple[float, ...], dict[str, Any]] | None:
    """Return the completed slow proposal originating at ``checkpoint``.

    ``projection_completed`` is the durable asynchronous receipt.  A later
    boundary may apply the same proposal; prefer the receipt so latency does
    not silently turn a decision-at-checkpoint comparison into one at the
    later application state.
    """
    for record in records:
        if record.get("event") != "projection_completed" or record.get("source_cycle_index") != checkpoint:
            continue
        selected = record.get("selected_candidate")
        candidate = selected.get("weights") if isinstance(selected, dict) else None
        if candidate is None:
            continue
        return _weights(candidate, source=f"PACE proposal from {checkpoint}"), {
            "source_cycle_index": checkpoint,
            "application_cycle_index": record.get("application_cycle_index"),
            "selection_basis": record.get("selection_basis"),
            "predicted_full_horizon_mean_squared_fatigue": _finite(
                (record.get("selected_evidence") or {}).get("full_horizon_mean_squared_fatigue")
                if isinstance(record.get("selected_evidence"), dict) else None
            ),
            "predicted_terminal_minimum_capacity": _finite(
                (record.get("selected_evidence") or {}).get("terminal_minimum_capacity")
                if isinstance(record.get("selected_evidence"), dict) else None
            ),
        }
    return None


def _option_value_pairs(arguments: list[str], replacements: dict[str, str]) -> list[str]:
    """Clone benchmark arguments while replacing only continuation-owned fields."""
    paired = {
        "--n-windows", "--output-json", "--common-initial-solution",
        "--rho-prepared-checkpoint-output-template", "--rho-prepared-checkpoint-windows",
    }
    drop_flags = {
        "--common-initial-solution-recenter-first-node-bounds",
        "--adopt-common-initial-solution-warmup-cycles",
    }
    result: list[str] = []
    index = 0
    while index < len(arguments):
        item = arguments[index]
        if item in paired:
            if index + 1 >= len(arguments):
                raise ValueError(f"Source benchmark argument {item} is missing a value")
            if item in replacements:
                result += [item, replacements[item]]
            index += 2
            continue
        if item in drop_flags:
            index += 1
            continue
        result.append(item)
        index += 1
    for option, value in replacements.items():
        if option not in arguments:
            result += [option, value]
    return result


def _candidate_document(*, candidate_id: str, weights: tuple[float, ...], checkpoint: int,
                        source: str, proposal: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "initial_weight_basis": (
            "PACE decision-fidelity validation; frozen static continuation; "
            f"candidate={candidate_id}; source_checkpoint_completed_cycles={checkpoint}; {source}; no_FHO_data"
        ),
        "initial_weights": dict(zip(MUSCLE_NAMES, weights, strict=True)),
        "policy": {"adaptation_enabled": False, "max_cycles": 20,
                   "min_relative_weight": 0.001, "max_relative_weight": 64.0},
        # ``run_rho_pace_benchmark`` deliberately permits only the established
        # top-level schema.  Put provenance in its existing calibration block
        # so static RHO-Physio can consume this file without a special case.
        "calibration": {"decision_fidelity": {"candidate_id": candidate_id,
                                                "checkpoint_completed_cycles": checkpoint,
                                                "source": source, "pace_projection": proposal}},
    }


def prepare_validation(*, pace_directory: Path, bo_weights_config: Path, output_directory: Path,
                       python: Path, checkpoints: tuple[int, ...] = DEFAULT_CHECKPOINTS,
                       horizons: tuple[int, ...] = DEFAULT_HORIZONS,
                       candidate_configs: dict[str, Path] | None = None) -> Path:
    """Create all static continuation jobs from existing certified checkpoints."""
    pace_directory = pace_directory.expanduser().resolve(strict=True)
    bo_weights_config = bo_weights_config.expanduser().resolve(strict=True)
    output_directory = output_directory.expanduser().resolve()
    python = python.expanduser().resolve(strict=True)
    if output_directory.exists():
        raise FileExistsError(f"Output directory must be fresh: {output_directory}")
    if not checkpoints or any(not isinstance(item, int) or item < 1 for item in checkpoints):
        raise ValueError("checkpoints must be positive integers")
    if not horizons or any(not isinstance(item, int) or item < 1 or item > 20 for item in horizons):
        raise ValueError("horizons must be positive integers no greater than 20")

    audit = _read_json(pace_directory / "configuration-audit.json")
    source_arguments = audit.get("arguments")
    if not isinstance(source_arguments, list) or not all(isinstance(item, str) for item in source_arguments):
        raise ValueError("PACE configuration audit has no serialised benchmark arguments")
    model_path = Path(audit.get("model_config_path", "")).expanduser().resolve(strict=True)
    journal_path = Path(audit.get("weights_journal_path", "")).expanduser().resolve(strict=True)
    journal = _load_journal(journal_path)
    bo = _read_json(bo_weights_config)
    bo_weights = tuple(float(bo["initial_weights"][name]) for name in MUSCLE_NAMES)
    _weights(bo_weights, source="BO fixed-weight config")
    extra_candidates = []
    reserved_ids = {"pace_current", "unit", "bo_fixed", "pace_proposal"}
    for candidate_id, candidate_config in (candidate_configs or {}).items():
        if (not candidate_id or candidate_id in reserved_ids or candidate_id in {".", ".."}
                or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                       for character in candidate_id)):
            raise ValueError(f"Invalid or reserved candidate id: {candidate_id!r}")
        candidate_config = candidate_config.expanduser().resolve(strict=True)
        document = _read_json(candidate_config)
        weights = _weights([document["initial_weights"][name] for name in MUSCLE_NAMES],
                           source=f"extra candidate {candidate_id}")
        extra_candidates.append((candidate_id, weights, f"experimental fixed candidate from {candidate_config}", None))

    jobs: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        checkpoint_path = pace_directory / "checkpoints" / f"cycle-{checkpoint}.npz"
        receipt_path = checkpoint_path.with_suffix(".receipt.json")
        if not checkpoint_path.is_file() or not receipt_path.is_file():
            raise FileNotFoundError(f"Certified checkpoint/receipt missing for cycle {checkpoint}: {checkpoint_path}")
        receipt = _read_json(receipt_path)
        if receipt.get("completed_windows") != checkpoint:
            raise ValueError(f"Checkpoint receipt does not certify exactly {checkpoint} completed cycles")
        pace_current = _boundary_weights(journal, checkpoint)
        proposal = _pace_proposal(journal, checkpoint)
        candidates: list[tuple[str, tuple[float, ...], str, dict[str, Any] | None]] = [
            ("pace_current", pace_current, "weights already applied by PACE at saved boundary", None),
            ("unit", (1.0, 1.0, 1.0, 1.0), "uniform unit comparator", None),
            ("bo_fixed", bo_weights, f"fixed BO comparator from {bo_weights_config}", None),
        ]
        if proposal is not None:
            candidates.append(("pace_proposal", proposal[0], "slow PACE proposal computed from this checkpoint", proposal[1]))
        candidates.extend(extra_candidates)
        for candidate_id, weights, source, projection in candidates:
            candidate_path = output_directory / f"checkpoint-{checkpoint}" / candidate_id / "weights.json"
            _write_new_json(candidate_path, _candidate_document(
                candidate_id=candidate_id, weights=weights, checkpoint=checkpoint, source=source, proposal=projection,
            ))
            for horizon in horizons:
                result_path = output_directory / f"checkpoint-{checkpoint}" / candidate_id / f"horizon-{horizon}" / "result.json"
                replacements = {
                    "--n-windows": str(horizon), "--output-json": str(result_path),
                    "--common-initial-solution": str(checkpoint_path),
                }
                benchmark = _option_value_pairs(list(source_arguments), replacements)
                benchmark += ["--common-initial-solution-recenter-first-node-bounds",
                              "--adopt-common-initial-solution-warmup-cycles"]
                command = [str(python), str(ROOT / "scripts/run_configured_cycling_benchmark.py"),
                           "--model-config", str(model_path), "--condition", "rho-physio",
                           "--weights-config", str(candidate_path), "--weights-journal",
                           str(result_path.parent / "weights.jsonl"), "--configuration-audit",
                           str(result_path.parent / "configuration-audit.json"), "--", *benchmark]
                jobs.append({
                    "checkpoint_completed_cycles": checkpoint, "candidate_id": candidate_id,
                    "horizon_cycles": horizon, "weights": list(weights), "weights_config": str(candidate_path),
                    "checkpoint_path": str(checkpoint_path), "checkpoint_receipt": str(receipt_path),
                    "result_path": str(result_path), "command": command,
                    "pace_projection": projection,
                    "continuation_semantics": "exact_certified_shifted_primal; frozen_static_weights; no_PACE_worker_restore",
                })
    manifest = {
        "schema_version": 1, "kind": "pace_decision_fidelity_validation",
        "pace_directory": str(pace_directory), "model_config": str(model_path),
        "journal": str(journal_path), "bo_weights_config": str(bo_weights_config),
        "extra_candidate_configs": {name: str(path.expanduser().resolve(strict=True))
                                    for name, path in (candidate_configs or {}).items()},
        "checkpoints": list(checkpoints), "horizons": list(horizons), "muscle_names": list(MUSCLE_NAMES),
        "interpretation": {
            "decision_target": "frozen PACE choice evaluated by exact short RHO continuations",
            "not_tested": "future asynchronous PACE worker/controller-clock restoration",
            "ranking_metric": "lower continuation fatigue_auc_cycles among fully validated candidates",
        },
        "jobs": jobs,
    }
    manifest_path = output_directory / "manifest.json"
    _write_new_json(manifest_path, manifest)
    return manifest_path


def _result_metrics(path: Path, horizon: int) -> dict[str, Any]:
    if not path.is_file():
        return {"state": "pending"}
    try:
        document = _read_json(path)
        rows = document.get("results")
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
            raise ValueError("result has no single solver row")
        row = rows[0]
        fatigue = _finite(row.get("fatigue_auc_cycles"))
        capacity = _finite(row.get("min_A_capacity_ratio"))
        validated = row.get("validated_cycles")
        validated = validated if isinstance(validated, int) and not isinstance(validated, bool) else 0
        saturation = row.get("control_saturation")
        terminal_upper = []
        if isinstance(saturation, list):
            terminal_upper = [_finite(item.get("terminal_upper_fraction")) for item in saturation if isinstance(item, dict)]
            terminal_upper = [item for item in terminal_upper if item is not None]
        crank = row.get("physical_crank_diagnostics")
        margin = None
        if isinstance(crank, dict):
            tolerance = _finite(crank.get("absolute_cycle_tolerance"))
            errors = crank.get("absolute_cycle_errors")
            if tolerance is not None and isinstance(errors, list):
                finite_errors = [_finite(item) for item in errors]
                finite_errors = [abs(item) for item in finite_errors if item is not None]
                if finite_errors:
                    margin = tolerance - max(finite_errors)
        return {
            "state": "available", "solver_success": row.get("solver_success") is True,
            "physical_success": row.get("physical_success") is True,
            "validated_cycles": validated, "horizon_completed": validated >= horizon,
            "fatigue_auc_cycles": fatigue, "min_A_capacity_ratio": capacity,
            "mean_terminal_upper_pw_saturation": (sum(terminal_upper) / len(terminal_upper)
                                                  if terminal_upper else None),
            "mechanical_absolute_cycle_margin_rad": margin,
            "status": row.get("status"), "error": row.get("error"),
        }
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        return {"state": "unreadable", "error": f"{type(error).__name__}: {error}"}


def _quality_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
    """Higher is better; incomplete trajectories never outrank completed ones."""
    return (
        1.0 if metrics.get("horizon_completed") and metrics.get("physical_success") else 0.0,
        -float(metrics["fatigue_auc_cycles"]) if _finite(metrics.get("fatigue_auc_cycles")) is not None else -math.inf,
        float(metrics["min_A_capacity_ratio"]) if _finite(metrics.get("min_A_capacity_ratio")) is not None else -math.inf,
    )


def summarize(manifest_path: Path, *, plot: bool = True) -> Path:
    manifest_path = manifest_path.expanduser().resolve(strict=True)
    manifest = _read_json(manifest_path)
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("Manifest needs jobs")
    entries = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        entry = {key: job.get(key) for key in ("checkpoint_completed_cycles", "candidate_id", "horizon_cycles", "weights", "pace_projection", "result_path")}
        entry["metrics"] = _result_metrics(Path(job["result_path"]), int(job["horizon_cycles"]))
        entries.append(entry)
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for entry in entries:
        groups.setdefault((int(entry["checkpoint_completed_cycles"]), int(entry["horizon_cycles"])), []).append(entry)
    comparisons = []
    for (checkpoint, horizon), group in sorted(groups.items()):
        available = [entry for entry in group if entry["metrics"].get("state") == "available"]
        ordered = sorted(available, key=lambda entry: _quality_key(entry["metrics"]), reverse=True)
        for rank, entry in enumerate(ordered, start=1):
            entry["actual_rank"] = rank
        unit = next((entry for entry in available if entry["candidate_id"] == "unit"), None)
        selected = next((entry for entry in available if entry["candidate_id"] == "pace_current"), None)
        best = ordered[0] if ordered else None
        for entry in available:
            fatigue = _finite(entry["metrics"].get("fatigue_auc_cycles"))
            unit_fatigue = None if unit is None else _finite(unit["metrics"].get("fatigue_auc_cycles"))
            entry["signed_fatigue_auc_gain_vs_unit"] = (
                None if fatigue is None or unit_fatigue is None else unit_fatigue - fatigue
            )
        selected_fatigue = None if selected is None else _finite(selected["metrics"].get("fatigue_auc_cycles"))
        best_fatigue = None if best is None else _finite(best["metrics"].get("fatigue_auc_cycles"))
        comparisons.append({
            "checkpoint_completed_cycles": checkpoint, "horizon_cycles": horizon,
            "available_candidates": len(available),
            "best_candidate_id": None if best is None else best["candidate_id"],
            "pace_selected_candidate_id": None if selected is None else selected["candidate_id"],
            "pace_selected_actual_rank": None if selected is None else selected.get("actual_rank"),
            "pace_selection_regret_fatigue_auc_cycles": (
                None if selected_fatigue is None or best_fatigue is None else selected_fatigue - best_fatigue
            ),
            "interpretation": "positive signed gain/regret means lower fatigue AUC than unit / worse than the observed best, respectively",
        })
    summary = {
        "schema_version": 1, "kind": "pace_decision_fidelity_summary", "manifest": str(manifest_path),
        "entries": entries, "comparisons": comparisons,
        "caveat": (
            "A checkpoint restores the certified shifted RHO primal, not the asynchronous PACE worker or its clock. "
            "These are static short-continuation decision tests, not full PACE campaign restarts."
        ),
    }
    output = manifest_path.parent / "summary.json"
    _atomic_json(output, summary)
    _write_markdown(manifest_path.parent / "summary.md", comparisons, entries)
    if plot:
        _plot(manifest_path.parent / "decision-fidelity.png", entries)
    return output


def _write_markdown(path: Path, comparisons: list[dict[str, Any]], entries: list[dict[str, Any]]) -> None:
    lines = ["# PACE decision-fidelity validation", "", "Lower fatigue AUC is better; results not yet available are omitted.", "",
             "| Checkpoint | Horizon | Best | PACE-current rank | PACE regret fatigue AUC |", "|---:|---:|---|---:|---:|"]
    for item in comparisons:
        regret = item["pace_selection_regret_fatigue_auc_cycles"]
        lines.append(f"| {item['checkpoint_completed_cycles']} | {item['horizon_cycles']} | {item['best_candidate_id'] or '—'} | "
                     f"{item['pace_selected_actual_rank'] or '—'} | {'—' if regret is None else f'{regret:.6g}'} |")
    lines += ["", "## Continuation metrics", "", "| Checkpoint | Horizon | Candidate | Rank | Fatigue AUC | Min capacity | PW upper saturation | Mechanical margin [rad] |", "|---:|---:|---|---:|---:|---:|---:|---:|"]
    for item in entries:
        metrics = item["metrics"]
        if metrics.get("state") != "available":
            continue
        values = (metrics.get("fatigue_auc_cycles"), metrics.get("min_A_capacity_ratio"),
                  metrics.get("mean_terminal_upper_pw_saturation"), metrics.get("mechanical_absolute_cycle_margin_rad"))
        rendered = ["—" if value is None else f"{value:.6g}" for value in values]
        lines.append(f"| {item['checkpoint_completed_cycles']} | {item['horizon_cycles']} | {item['candidate_id']} | {item.get('actual_rank', '—')} | "
                     + " | ".join(rendered) + " |")
    _atomic_text(path, "\n".join(lines) + "\n")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    temporary.replace(path)


def _plot(path: Path, entries: list[dict[str, Any]]) -> None:
    available = [entry for entry in entries if entry["metrics"].get("state") == "available"]
    if not available:
        return
    try:
        os.environ.setdefault("MPLBACKEND", "Agg")
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/cocofest-matplotlib")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    labels = [f"c{item['checkpoint_completed_cycles']}\nh{item['horizon_cycles']}\n{item['candidate_id']}" for item in available]
    fatigue = [item["metrics"].get("fatigue_auc_cycles") for item in available]
    capacity = [item["metrics"].get("min_A_capacity_ratio") for item in available]
    fig, axes = plt.subplots(2, 1, figsize=(max(9, len(labels) * .38), 6), layout="constrained")
    axes[0].bar(range(len(labels)), [value if value is not None else math.nan for value in fatigue], color="#4c78a8")
    axes[0].set_ylabel("Fatigue AUC [cycles]\n(lower is better)")
    axes[1].bar(range(len(labels)), [value if value is not None else math.nan for value in capacity], color="#54a24b")
    axes[1].set_ylabel("Minimum A capacity ratio")
    axes[1].set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    for axis in axes:
        axis.grid(axis="y", alpha=.25)
    fig.suptitle("Exact short RHO continuations from PACE checkpoints")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def execute(manifest_path: Path, *, workers: int) -> None:
    manifest_path = manifest_path.expanduser().resolve(strict=True)
    manifest = _read_json(manifest_path)
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("Manifest needs jobs")
    if workers < 1:
        raise ValueError("workers must be positive")
    pending = [job for job in jobs if isinstance(job, dict) and not Path(job["result_path"]).is_file()]

    def run(job: dict[str, Any]) -> str:
        result = Path(job["result_path"])
        result.parent.mkdir(parents=True, exist_ok=True)
        log = result.parent / "launcher-output.log"
        with log.open("x", encoding="utf-8") as stream:
            completed = subprocess.run(job["command"], cwd=ROOT, text=True, stdout=stream,
                                       stderr=subprocess.STDOUT, check=False)
        if completed.returncode:
            raise RuntimeError(f"exit={completed.returncode}; inspect {log}")
        return f"checkpoint={job['checkpoint_completed_cycles']} candidate={job['candidate_id']} horizon={job['horizon_cycles']}"

    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run, job): job for job in pending}
        for future in as_completed(futures):
            try:
                print(f"Completed: {future.result()}", flush=True)
            except BaseException as error:
                job = futures[future]
                errors.append(f"checkpoint={job['checkpoint_completed_cycles']} candidate={job['candidate_id']} horizon={job['horizon_cycles']}: {error}")
    summarize(manifest_path)
    if errors:
        raise SystemExit("; ".join(errors))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pace-directory", type=Path)
    parser.add_argument("--bo-weights-config", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--manifest", type=Path, help="Existing manifest for --execute or --summarize")
    parser.add_argument("--checkpoints", default=",".join(map(str, DEFAULT_CHECKPOINTS)))
    parser.add_argument("--horizons", default=",".join(map(str, DEFAULT_HORIZONS)))
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--candidate-config", action="append", default=[], metavar="ID=JSON",
                        help="Additional frozen weight candidate; repeat for multiple candidates")
    args = parser.parse_args(argv)
    if not (args.prepare or args.execute or args.summarize):
        parser.error("Select --prepare, --execute, and/or --summarize")
    manifest = args.manifest
    if args.prepare:
        if None in (args.pace_directory, args.bo_weights_config, args.output_directory, args.python):
            parser.error("--prepare requires --pace-directory, --bo-weights-config, --output-directory and --python")
        checkpoints = tuple(int(item) for item in args.checkpoints.split(",") if item.strip())
        horizons = tuple(int(item) for item in args.horizons.split(",") if item.strip())
        candidate_configs = {}
        for item in args.candidate_config:
            candidate_id, separator, path = item.partition("=")
            if not separator or not path or candidate_id in candidate_configs:
                parser.error("--candidate-config requires unique ID=JSON pairs")
            candidate_configs[candidate_id] = Path(path)
        manifest = prepare_validation(pace_directory=args.pace_directory, bo_weights_config=args.bo_weights_config,
                                      output_directory=args.output_directory, python=args.python,
                                      checkpoints=checkpoints, horizons=horizons,
                                      candidate_configs=candidate_configs)
        print(f"Prepared: {manifest}", flush=True)
    if manifest is None:
        parser.error("--execute/--summarize requires --manifest (or --prepare in this invocation)")
    if args.execute:
        execute(manifest, workers=args.workers)
    elif args.summarize:
        print(f"Summary: {summarize(manifest)}", flush=True)


if __name__ == "__main__":
    main()
