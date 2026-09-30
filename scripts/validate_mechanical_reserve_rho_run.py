"""Validate the runtime wiring of a bilateral mechanical-reserve RHO run.

This checks the recorded execution contract, not endurance or future
mechanical feasibility. Use a run with at least two certified cycles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


ARMS = ("right", "left")


def validate_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact, machine-readable audit of the first activation."""
    failures: list[str] = []
    completed = summary.get("completed_rho_cycles")
    if not isinstance(completed, int) or completed < 2:
        failures.append("At least two completed RHO cycles are required.")
    if summary.get("success") is not True or summary.get("failure") is not None:
        failures.append("The bilateral run did not finish successfully.")
    if summary.get("requested_cycles") != completed:
        failures.append("Requested and completed cycle counts differ.")

    arm_reports: dict[str, dict[str, Any]] = {}
    arms = summary.get("arms")
    if not isinstance(arms, Mapping):
        arms = {}
    for side in ARMS:
        arm = arms.get(side)
        if not isinstance(arm, Mapping):
            failures.append(f"{side}: missing arm result.")
            continue
        cycles = arm.get("cycles")
        if not isinstance(cycles, list) or len(cycles) != completed:
            failures.append(f"{side}: missing or incomplete cycle records.")
            continue
        if any(not isinstance(cycle, Mapping) or cycle.get("cycle") != index or
               cycle.get("certified") is not True
               for index, cycle in enumerate(cycles, start=1)):
            failures.append(f"{side}: cycles are not consecutively certified.")

        binding = arm.get("mechanical_reserve_binding")
        if not isinstance(binding, Mapping):
            failures.append(f"{side}: missing mechanical-reserve binding.")
            continue
        if binding.get("objective_graph_build_count") != 1 or binding.get("objective_graph_rebuild_required") is not False:
            failures.append(f"{side}: objective graph was not built exactly once.")
        if binding.get("nlp_attached") is not True or binding.get("compiled_solver_reuse_verified") is not True:
            failures.append(f"{side}: compiled NLP solver reuse was not verified.")
        if binding.get("compiled_solver_observation_count") != len(cycles):
            failures.append(f"{side}: solver observation count differs from cycle count.")

        solver_ids = []
        activations = []
        for index, cycle in enumerate(cycles, start=1):
            cost = cycle.get("mechanical_reserve_cost")
            solver = cycle.get("mechanical_reserve_solver")
            if not isinstance(cost, Mapping) or not isinstance(solver, Mapping):
                failures.append(f"{side}: missing reserve audit in cycle {index}.")
                continue
            if cost.get("attainable_work_certified") is not False:
                failures.append(f"{side}: cycle {index} mislabels the proxy as certified work.")
            activations.append(cost.get("activation"))
            if solver.get("solver_observed") is not True or not isinstance(solver.get("solver_identity"), int):
                failures.append(f"{side}: cycle {index} has no compiled solver identity.")
            else:
                solver_ids.append(solver["solver_identity"])
            if index >= 2 and solver.get("compiled_solver_reuse_verified") is not True:
                failures.append(f"{side}: solver reuse not verified in cycle {index}.")
        if not activations or activations[0] != 0.0 or len(activations) < 2 or activations[1] != 1.0:
            failures.append(f"{side}: expected activation 0 in cycle 1 and 1 in cycle 2.")
        if any(activation != 1.0 for activation in activations[1:]):
            failures.append(f"{side}: activation must stay at 1 after cycle 1.")
        if len(solver_ids) != len(cycles) or len(set(solver_ids)) != 1:
            failures.append(f"{side}: compiled solver identity changed or was missing.")

        events = arm.get("mechanical_reserve_events")
        if not isinstance(events, list):
            failures.append(f"{side}: missing reserve update events.")
            events = []
        for index, event in enumerate(events, start=1):
            if not isinstance(event, Mapping):
                failures.append(f"{side}: malformed reserve event {index}.")
                continue
            if event.get("status") == "updated" and (
                event.get("attainable_work_certified") is not False or
                event.get("endurance_prediction") is not False
            ):
                failures.append(f"{side}: event {index} mislabels the reserve proxy as a certification or endurance prediction.")
        initial = events[0] if events and isinstance(events[0], Mapping) else {}
        update = initial.get("parameter_update") if isinstance(initial.get("parameter_update"), Mapping) else {}
        if (initial.get("status"), initial.get("source_cycle"), initial.get("application_cycle")) != ("updated", 1, 2):
            failures.append(f"{side}: missing certified cycle 1 to cycle 2 update.")
        if initial.get("attainable_work_certified") is not False or initial.get("endurance_prediction") is not False:
            failures.append(f"{side}: reserve event must remain a noncertified proxy, not an endurance prediction.")
        if (update.get("nlp_reused") is not True or
                update.get("objective_graph_rebuild_required") is not False or
                update.get("objective_graph_build_count") != 1 or
                update.get("parameter_update_count") != 1):
            failures.append(f"{side}: first update did not reuse the parameterized NLP graph.")
        arm_reports[side] = {
            "certified_cycles": sum(cycle.get("certified") is True for cycle in cycles if isinstance(cycle, Mapping)),
            "activation_first_two": activations[:2],
            "objective_graph_build_count": binding.get("objective_graph_build_count"),
            "compiled_solver_reuse_verified": binding.get("compiled_solver_reuse_verified"),
            "first_update_source_cycle": initial.get("source_cycle"),
            "first_update_application_cycle": initial.get("application_cycle"),
        }

    return {
        "passed": not failures,
        "completed_rho_cycles": completed,
        "arms": arm_reports,
        "failures": failures,
        "scope": "runtime wiring and certified solved cycles only; no future feasibility or endurance claim",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="Bilateral campaign summary.json")
    args = parser.parse_args()
    try:
        summary = json.loads(args.summary.read_text())
        if not isinstance(summary, dict):
            raise ValueError("The summary must contain a JSON object.")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    report = validate_summary(summary)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
