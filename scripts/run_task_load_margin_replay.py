#!/usr/bin/env python3
"""Offline guarded intervention from one validated load-margin source.

Uses one fixed, locally validated direction, at most 20 cycles old, with ordinary
RHO fallback. This does not implement an online refreshing supervisor. Source
timestamps are preserved; the explicitly offline wall-age allowance is 1e9 s.
No solve failure is labeled physiological exhaustion.
"""
from copy import copy, deepcopy
from dataclasses import asdict, replace
import argparse
import json
import os
from pathlib import Path
import sys
from time import monotonic, perf_counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cocofest.optimization.task_load_margin import TaskLoadMarginRequest, TaskLoadMarginResult, TaskLoadMarginDirection, TaskLoadMarginPolicy
from cocofest.optimization.task_load_margin_ocp import TaskLoadMarginObjectiveBinding, periodic_ding_conditional_context, restore_source_with_inactive_load_binding, physical_problem_view_without_load_binding
from cocofest.optimization.task_load_margin_oracle import configure_load_objective, audit_load_solution, initial_bound_envelope_gradient
from cocofest.optimization.task_reserve import TaskReserveCheckpoint, ProbeEvidence
from cocofest.optimization.task_reserve_ocp import TaskReserveStateCoordinate
from cocofest.optimization.task_reserve_probe_adapter import independent_full_nlp_audit
from cocofest.simulation.independent_arms_process import _atomic_json, _driver_arguments, _configured_payload_model
from cocofest.simulation.rho_restart_checkpoint import export_prepared_checkpoint, restore_prepared_checkpoint
from scripts.probe_independent_rho_task_reserve import _load_arm
from scripts.run_local_task_reserve_branch import _advance


def disable_fatigue_objective(program):
    """Set only the declared fatigue Lagrange term to zero in an ablation.

    The parameter channel is intentionally retained: it is part of the exact
    prepared physical problem and must remain restorable.  The function never
    touches a load-margin Mayer objective or any hard constraint.
    """
    fatigue_names = {
        "minimize_overall_muscle_fatigue",
        "minimize_parameterized_overall_muscle_fatigue",
    }
    disabled = []
    for nlp in program.nlp:
        for penalty in nlp.J:
            if not penalty or getattr(penalty, "name", None) not in fatigue_names:
                continue
            replacement = copy(penalty)
            replacement.weight = deepcopy(penalty.weight)
            replacement.weight[...] = 0.
            if isinstance(replacement.node, tuple) and len(replacement.node) == 1:
                replacement.node = replacement.node[0]
            program.update_objectives(replacement)
            disabled.append({"name": penalty.name, "list_index": penalty.list_index})
    if not disabled:
        raise ValueError("No declared fatigue objective found for margin-only ablation")
    return disabled


def impose_terminal_trust_box(program, coordinates, center, radii, *, reference_bounds=None):
    """Impose the local load-model domain at the RHO terminal node.

    This is used only by the margin-only ablation.  It turns the previously
    post-solve trust check into a differentiable bound constraint so an affine
    reward cannot run to an arbitrary remote feasible endpoint.
    """
    center, radii = np.asarray(center, float), np.asarray(radii, float)
    if center.shape != radii.shape or center.shape != (len(coordinates),):
        raise ValueError("Trust center/radii do not match the terminal coordinates")
    changed = []
    for coordinate, value, radius in zip(coordinates, center, radii):
        bounds = program.nlp[0].x_bounds[coordinate.state_key]
        lower = value * coordinate.scale + coordinate.offset - radius * coordinate.scale
        upper = value * coordinate.scale + coordinate.offset + radius * coordinate.scale
        if reference_bounds is None:
            previous_lower, previous_upper = float(bounds.min[coordinate.index, 2]), float(bounds.max[coordinate.index, 2])
        else:
            previous_lower, previous_upper = reference_bounds[coordinate.state_key]
        lower, upper = max(previous_lower, lower), min(previous_upper, upper)
        if lower > upper:
            raise ValueError(f"Trust box conflicts with physical terminal bounds for {coordinate.state_key}")
        bounds.min[coordinate.index, 2] = lower
        bounds.max[coordinate.index, 2] = upper
        changed.append({"state_key": coordinate.state_key, "lower": lower, "upper": upper})
    return changed


def run(receipt, side, source_directory, validation_path, output_directory, *, cycles=3, weight=10000.,
        sweep_weights=None, verify_next_load=False, exploratory_refresh=False, margin_only=False,
        hard_terminal_trust=False):
    from bioptim import SolutionMerge
    from cocofest.optimization.configured_cycling_model import configured_model_factories
    from examples.fes_multibody.cycling import cycling_pulse_width_mhe_acados_periodic as driver
    if cycles < 1 or not np.isfinite(weight) or weight <= 0:
        raise ValueError("Positive cycles and intervention weight required")
    if verify_next_load and sweep_weights is None:
        raise ValueError("Exact next-load verification requires a weight sweep")
    if exploratory_refresh and sweep_weights is not None:
        raise ValueError("Weight sweep and exploratory refresh are distinct modes")
    if margin_only and not exploratory_refresh:
        raise ValueError("Margin-only ablation currently requires --exploratory-refresh")
    if hard_terminal_trust and not margin_only:
        raise ValueError("Hard terminal trust is defined only for the margin-only ablation")
    output = Path(output_directory).resolve()
    source_directory = Path(source_directory).resolve()
    raw_request = json.loads((source_directory/"request.json").read_text())
    raw_result = json.loads((source_directory/"result.json").read_text())
    validation = json.loads(Path(validation_path).read_text())
    if (validation.get("validation_passed") is not True or validation.get("coordinates") != "full Ding state"
            or Path(validation["source_result"]).resolve() != source_directory/"result.json"):
        raise ValueError("Matching independent full-state validation required")
    observations = json.loads(Path(validation_path).with_name("observations.json").read_text())["observations"]
    if len(observations) < 48 or not all(item["passed"] and Path(item["artifact"]).is_file() for item in observations):
        raise ValueError("Missing complete independent validation artifacts")
    source, payload = _load_arm(Path(receipt), side)
    original = TaskReserveCheckpoint(**raw_request["checkpoint"])
    original.verify_files()
    if source.prepared_problem_sha256 != original.prepared_problem_sha256:
        raise ValueError("Receipt does not identify the validated source")
    layout = tuple(TaskReserveStateCoordinate(**item) for item in validation["coordinate_layout"])
    context = json.loads(original.task_context_json)
    context.update(coordinate_layout=[asdict(item) for item in layout],
                   coordinate_definition="complete_ding_state_normalization_v1")
    source = TaskReserveCheckpoint.from_archive(Path(original.archive_path), completed_cycles=original.completed_cycles,
        model_path=Path(original.model_path), task_context=context)
    # Reannotation only: the physical source and measured trajectories are
    # identical. Preserve old reports and link them in the replay manifest.
    request = TaskLoadMarginRequest(raw_request["request_id"]+"-full-state-offline", source,
        tuple(item.state_key for item in layout), tuple(validation["center"]),
        tuple(validation["trust_radius"]), (0.,)*len(layout),
        tuple(1. if item.state_key.startswith("A_") else max(10.,value*2.)
              for item,value in zip(layout,validation["center"])),
        raw_request["issued_monotonic_seconds"], raw_request["load_factor_upper_bound"])
    nominal, load = [replace(ProbeEvidence(**raw_result[key]), task_context_sha256=source.task_context_sha256)
                     for key in ("nominal_witness","load_witness")]
    diagnostics = validation["diagnostics"]
    result = TaskLoadMarginResult(request.request_id, raw_result["completed_monotonic_seconds"],
        nominal_witness=nominal, load_witness=load, gradient=tuple(validation["gradient"]),
        sensitivity_method="kkt_envelope", sensitivity_regular=True,
        **{key:raw_result[key] for key in ("stationarity_residual","complementarity_residual","dual_feasibility_residual")},
        **{key:diagnostics[key] for key in ("gradient_validation_max_abs_error","local_validation_max_abs_error","independent_validation_points")},
        solution_artifact=raw_result["solution_artifact"], multipliers_artifact=raw_result["multipliers_artifact"])
    direction = TaskLoadMarginDirection(request,result)
    args = copy(_driver_arguments(payload,side))
    args.nlp_ipopt_recovery = args.nlp_ipopt_recovery_ma57_tuned = False
    configured = _configured_payload_model(payload,side)
    def build():
        with configured_model_factories(configured[1],[]):
            runtime = driver.build_unilateral_runtime(args,echo=False)
        runtime["nmpc"]._initialize_state_idx_to_cycle({"states":{}})
        return runtime
    base_runtime = build()
    restore_prepared_checkpoint(Path(source.archive_path),base_runtime["nmpc"],completed_cycles=source.completed_cycles)
    conditional_sha,conditional_context = periodic_ding_conditional_context(base_runtime["nmpc"],layout)
    physical_context = dict(cycle_period_s=abs(2*np.pi/args.isokinetic_omega),
        cycle_len=args.stimulations_per_cycle, formulation="isokinetic", mechanical_formulation="reduced",
        nominal_work_j=context["nominal_work_j"], terminal_half_step_guard=bool(args.reduced_terminal_half_step_velocity_guard))
    binding = TaskLoadMarginObjectiveBinding(layout,task_context=context,model_sha256=source.model_sha256,
        physical_context=physical_context,conditional_context_sha256=conditional_sha,
        policy=TaskLoadMarginPolicy(maximum_age_cycles=20, maximum_age_seconds=1e9))
    args.experimental_task_load_margin_binding = binding
    args.experimental_task_load_margin_model_path = source.model_path
    runtime = build()
    program,solver = runtime["nmpc"],runtime["solver"]
    restored = restore_source_with_inactive_load_binding(Path(source.archive_path),program,completed_cycles=source.completed_cycles)
    fatigue_ablation = disable_fatigue_objective(program) if margin_only else ()
    output.mkdir(parents=True,exist_ok=False)
    protocol_mode=("offline_exploratory_margin_only_hard_trust_kkt_refresh_uncertified_gradient" if exploratory_refresh and margin_only and hard_terminal_trust
        else "offline_exploratory_margin_only_kkt_refresh_uncertified_gradient" if exploratory_refresh and margin_only
        else "offline_exploratory_kkt_refresh_each_cycle_uncertified_gradient" if exploratory_refresh
        else "single_boundary_weight_sweep_no_decision_committed" if sweep_weights is not None
        else "offline_one_local_direction_intervention")
    _atomic_json(output/"protocol.json",{"mode":protocol_mode,
        "source_receipt":str(Path(receipt).resolve()),"source_request":str(source_directory/"request.json"),
        "source_result":str(source_directory/"result.json"),"validation":str(Path(validation_path).resolve()),
        "context_reannotation":"only coordinate normalization metadata; same physical source and trajectories",
        "source_completed_cycles":source.completed_cycles,"requested_additional_cycles":cycles,"weight":weight,
        "wall_age_policy_seconds":1e9,"timestamps_refreshed":False,"conditional_context":conditional_context,
        "source_restore":restored,"request":asdict(request),"result":asdict(result),
        "fatigue_objective_disabled":fatigue_ablation,
        "terminal_trust_box_hard_constraint":bool(hard_terminal_trust)})
    def coordinates(states=None):
        return tuple((float(program.nlp[0].x_bounds[c.state_key].min[c.index,0])
            if states is None else float(states[c.state_key][c.index,-1]))/c.scale-c.offset/c.scale for c in layout)
    if sweep_weights is not None:
        if not sweep_weights or any(not np.isfinite(w) or w < 0 for w in sweep_weights):
            raise ValueError("Sweep weights must be finite and nonnegative")
        from bioptim import SolutionMerge
        center = np.asarray(request.center,float)
        gradient = np.asarray(result.gradient,float)
        records=[]
        supervisor_program, supervisor_solver = base_runtime["nmpc"], base_runtime["solver"]
        supervisor_configured = False
        for candidate_weight in sweep_weights:
            binding._write(program,np.zeros_like(binding.values))
            restore_source_with_inactive_load_binding(Path(source.archive_path),program,
                                                       completed_cycles=source.completed_cycles)
            signature,_=periodic_ding_conditional_context(program,layout)
            if candidate_weight == 0:
                binding.deactivate(program)
                activation={"accepted":False,"reasons":["zero_weight_reference"]}
            else:
                activation=binding.update(program,direction,weight=candidate_weight,coordinates=tuple(center),
                    completed_cycles=source.completed_cycles,now_monotonic_seconds=monotonic(),
                    conditional_context_sha256=signature)
            solution=super(driver.RecedingHorizonOptimization,program).solve(solver=solver,warm_start=None)
            states=solution.decision_states(to_merge=SolutionMerge.NODES)
            terminal=np.asarray([(float(states[c.state_key][c.index,-1])-c.offset)/c.scale for c in layout])
            terminal_signature,_=periodic_ding_conditional_context(program,layout,states=states)
            gate=binding.validate_terminal_point(tuple(terminal),completed_cycles=source.completed_cycles+1,
                now_monotonic_seconds=monotonic(),conditional_context_sha256=terminal_signature)
            audit=independent_full_nlp_audit(solution,program,target_work_j=context["nominal_work_j"],tolerance=1e-6)
            margin_delta=float(gradient @ (terminal-center))
            objective=float(np.asarray(solution.cost).reshape(-1)[0])
            record={"weight":candidate_weight,"status":int(solution.status),"audit":asdict(audit),
                "activation":activation,"terminal_gate":gate,"solver_time_s":float(solution.real_time_to_optimize),
                "objective_total":objective,"predicted_margin_delta":margin_delta,
                "margin_cost_term":-candidate_weight*margin_delta,
                "base_cost_on_same_trajectory":objective+candidate_weight*margin_delta,
                "maximum_normalized_terminal_shift":float(np.max(np.abs(terminal-center))),
                "maximum_trust_fraction":float(np.max(np.abs(terminal-center)/np.asarray(request.trust_radius))),
                "terminal_coordinates":terminal.tolist()}
            if verify_next_load and audit.passed:
                _advance(program,solution,context["nominal_work_j"])
                prepared=output/f"next-boundary-weight-{candidate_weight:g}.npz"
                export_prepared_checkpoint(prepared,physical_problem_view_without_load_binding(program),
                                           completed_cycles=source.completed_cycles+1)
                restored_next=restore_prepared_checkpoint(prepared,supervisor_program,
                    completed_cycles=source.completed_cycles+1)
                if restored_next["stimulation_history_complete"] is not True:
                    raise RuntimeError("Next-cycle load oracle lost stimulation history")
                if not supervisor_configured:
                    configure_load_objective(supervisor_program,nominal_work_j=context["nominal_work_j"],
                                             upper_bound=request.load_factor_upper_bound)
                    supervisor_configured=True
                else:
                    work_bounds=supervisor_program.nlp[0].x_bounds["E_prod"]
                    work_bounds.min[0,2]=0.
                    work_bounds.max[0,2]=context["nominal_work_j"]*request.load_factor_upper_bound
                next_solution=super(driver.RecedingHorizonOptimization,supervisor_program).solve(
                    solver=supervisor_solver,warm_start=None)
                next_load,next_audit,next_kkt=audit_load_solution(next_solution,supervisor_program,
                    nominal_work_j=context["nominal_work_j"],upper_bound=request.load_factor_upper_bound,
                    tolerance=1e-6)
                record["next_cycle_exact_load_factor"]=next_load
                record["next_cycle_load_audit"]=asdict(next_audit)
                record["next_cycle_load_kkt"]=next_kkt
                record["next_cycle_load_solver_seconds"]=float(next_solution.real_time_to_optimize)
            records.append(record)
            _atomic_json(output/"progress.json",{"sweep":records})
            print(json.dumps({k:record[k] for k in ("weight","objective_total","predicted_margin_delta",
                "maximum_trust_fraction","solver_time_s")}),flush=True)
        report={"mode":"single_boundary_weight_sweep_no_decision_committed",
            "source_completed_cycles":source.completed_cycles,"sweep":records}
        _atomic_json(output/"summary.json",report)
        return report
    if exploratory_refresh:
        # Offline ablation only: later KKT directions have *not* received the
        # complete 40-axis + eight-holdout independent certification required
        # by the production binding.update contract. Never label this mode a
        # validated online controller or physiological failure certificate.
        supervisor_program,supervisor_solver=base_runtime["nmpc"],base_runtime["solver"]
        supervisor_configured=False
        trust=np.asarray(request.trust_radius,float)
        records=[]
        current_archive=Path(source.archive_path)
        started=perf_counter()
        physical_terminal_bounds={c.state_key:(float(program.nlp[0].x_bounds[c.state_key].min[c.index,2]),
                                                float(program.nlp[0].x_bounds[c.state_key].max[c.index,2]))
                                  for c in layout}
        for offset in range(cycles):
            completed=source.completed_cycles+offset
            restored_supervisor=restore_prepared_checkpoint(current_archive,supervisor_program,
                completed_cycles=completed)
            if restored_supervisor["stimulation_history_complete"] is not True:
                raise RuntimeError("Supervisor refresh lost stimulation history")
            if not supervisor_configured:
                configure_load_objective(supervisor_program,nominal_work_j=context["nominal_work_j"],
                                         upper_bound=request.load_factor_upper_bound)
                supervisor_configured=True
            else:
                work_bounds=supervisor_program.nlp[0].x_bounds["E_prod"]
                work_bounds.min[0,2]=0.
                work_bounds.max[0,2]=context["nominal_work_j"]*request.load_factor_upper_bound
            sup_start=perf_counter()
            sup_solution=super(driver.RecedingHorizonOptimization,supervisor_program).solve(
                solver=supervisor_solver,warm_start=None)
            load,load_audit,kkt=audit_load_solution(sup_solution,supervisor_program,
                nominal_work_j=context["nominal_work_j"],upper_bound=request.load_factor_upper_bound,
                tolerance=1e-6)
            sup_seconds=perf_counter()-sup_start
            valid_kkt=load_audit.passed and all(value is not None and np.isfinite(value)
                and 0<=value<=1e-5 for value in kkt.values())
            margin_can_support_nominal=bool(valid_kkt and load >= 1.-1e-6)
            gradient=None
            if margin_can_support_nominal:
                gradient=initial_bound_envelope_gradient(sup_solution,supervisor_program,layout)
                center=np.asarray(gradient["normalized_coordinates"],float)
                binding._write(program,np.r_[weight,0.,center,gradient["gradient"]])
                hard_trust_bounds=(impose_terminal_trust_box(program,layout,center,trust,
                                      reference_bounds=physical_terminal_bounds)
                                   if hard_terminal_trust else ())
            else:
                binding._write(program,np.zeros_like(binding.values))
                center=coordinates()
                hard_trust_bounds=()
                if margin_only:
                    record={"cycle":completed+1,"load_factor_at_start":load,"load_audit":asdict(load_audit),
                        "load_kkt":kkt,"supervisor_seconds":sup_seconds,"candidate_accepted":False,
                        "termination":"local_load_margin_below_nominal_or_unverified",
                        "physiological_failure_certified":False}
                    records.append(record)
                    _atomic_json(output/"progress.json",{"cycles":records})
                    break
            solution=super(driver.RecedingHorizonOptimization,program).solve(solver=solver,warm_start=None)
            states=solution.decision_states(to_merge=SolutionMerge.NODES)
            proposed=np.asarray(coordinates(states),float)
            trust_fraction=float(np.max(np.abs(proposed-center)/trust))
            fallback=bool(valid_kkt and trust_fraction>1.+1e-12)
            if fallback:
                if margin_only and not hard_terminal_trust:
                    # The ablation must not silently revert to fatigue.  It
                    # stops when the locally validated margin is unavailable.
                    record={"cycle":completed+1,"load_factor_at_start":load,"load_audit":asdict(load_audit),
                        "load_kkt":kkt,"supervisor_seconds":sup_seconds,"candidate_trust_fraction":trust_fraction,
                        "candidate_fallback":True,"candidate_accepted":False,
                        "termination":"local_margin_domain_exhausted_without_fatigue_fallback",
                        "physiological_failure_certified":False}
                    records.append(record)
                    _atomic_json(output/"progress.json",{"cycles":records})
                    break
                binding._write(program,np.zeros_like(binding.values))
                solution=super(driver.RecedingHorizonOptimization,program).solve(solver=solver,warm_start=None)
                states=solution.decision_states(to_merge=SolutionMerge.NODES)
            audit=independent_full_nlp_audit(solution,program,target_work_j=context["nominal_work_j"],tolerance=1e-6)
            record={"cycle":completed+1,"load_factor_at_start":load,"load_audit":asdict(load_audit),
                "load_kkt":kkt,"supervisor_seconds":sup_seconds,"candidate_trust_fraction":trust_fraction,
                "candidate_fallback":fallback,"candidate_accepted":bool(valid_kkt and not fallback and audit.passed),
                "rho_audit":asdict(audit),"rho_solver_seconds":float(solution.real_time_to_optimize),
                "gradient":gradient,"terminal_coordinates":coordinates(states),
                "status":int(solution.status),"hard_terminal_trust_bounds":hard_trust_bounds}
            records.append(record)
            _atomic_json(output/"progress.json",{"cycles":records})
            print(json.dumps({key:record[key] for key in ("cycle","load_factor_at_start",
                "supervisor_seconds","candidate_trust_fraction","candidate_fallback","candidate_accepted",
                "rho_solver_seconds")}),flush=True)
            if not audit.passed:
                break
            _advance(program,solution,context["nominal_work_j"])
            current_archive=output/f"physical-prepared-cycle-{completed+1}.npz"
            export_prepared_checkpoint(current_archive,physical_problem_view_without_load_binding(program),
                completed_cycles=completed+1)
        report={"mode":("offline_exploratory_margin_only_hard_trust_kkt_refresh_uncertified_gradient" if margin_only and hard_terminal_trust
                        else "offline_exploratory_margin_only_kkt_refresh_uncertified_gradient" if margin_only
                        else "offline_exploratory_kkt_refresh_each_cycle_uncertified_gradient"),
            "source_completed_cycles":source.completed_cycles,
            "certified_additional_rho_cycles":sum(row.get("rho_audit", {}).get("passed", False) for row in records),
            "last_certified_cycle":source.completed_cycles+sum(row.get("rho_audit", {}).get("passed", False) for row in records),
            "candidate_accepted_cycles":sum(row.get("candidate_accepted", False) for row in records),
            "candidate_fallback_cycles":sum(row.get("candidate_fallback", False) for row in records),
            "refreshes_with_primal_kkt_evidence":sum(row["load_audit"]["passed"] for row in records),
            "fatigue_objective_disabled":bool(margin_only),
            "terminal_trust_box_hard_constraint":bool(hard_terminal_trust),
            "independent_gradient_validation_each_cycle":False,
            "physiological_failure_certified":False,
            "wall_time_s":perf_counter()-started,"cycles":records}
        _atomic_json(output/"summary.json",report)
        return report
    records, failure = [], None
    start = perf_counter()
    def solve():
        return super(driver.RecedingHorizonOptimization,program).solve(solver=solver,warm_start=None)
    for offset in range(cycles):
        completed = source.completed_cycles+offset
        signature,_ = periodic_ding_conditional_context(program,layout)
        activated = binding.update(program,direction,weight=weight,coordinates=coordinates(),
            completed_cycles=completed,now_monotonic_seconds=monotonic(),conditional_context_sha256=signature)
        solution = solve()
        states = solution.decision_states(to_merge=SolutionMerge.NODES)
        signature,_ = periodic_ding_conditional_context(program,layout,states=states)
        gate = binding.validate_terminal_point(coordinates(states),completed_cycles=completed+1,
            now_monotonic_seconds=monotonic(),conditional_context_sha256=signature)
        proposed_coordinates = coordinates(states)
        if gate["fallback_resolve_required"]:
            binding.deactivate(program)
            solution = solve()
            states = solution.decision_states(to_merge=SolutionMerge.NODES)
        audit = independent_full_nlp_audit(solution,program,target_work_j=context["nominal_work_j"],tolerance=1e-6)
        record = {"cycle":completed+1,"certified":audit.passed,"activation":activated,"terminal_gate":gate,
            "status":int(solution.status),"audit":asdict(audit),"solver_time_s":float(solution.real_time_to_optimize),
            "coordinates":coordinates(states),"proposed_coordinates":proposed_coordinates,
            "active_cost_committed":bool(binding.values[0])}
        records.append(record)
        _atomic_json(output/"progress.json",{"cycles":records})
        if not audit.passed:
            failure="complete_discrete_NLP_failure_not_physiological_certificate"
            break
        np.savez_compressed(output/f"witness-cycle-{completed+1}.npz",vector=np.asarray(solution.vector,float),
            **{f"states__{key}":np.asarray(value,float) for key,value in states.items()})
        _advance(program,solution,context["nominal_work_j"])
    report={"source_completed_cycles":source.completed_cycles,"certified_additional_cycles":sum(r["certified"] for r in records),
        "last_certified_cycle":source.completed_cycles+sum(r["certified"] for r in records),
        "active_cost_committed_cycles":sum(r["certified"] and r["active_cost_committed"] for r in records),
        "failure_reason":failure,"physiological_failure_certified":False,"wall_time_s":perf_counter()-start,"cycles":records}
    _atomic_json(output/"summary.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt",type=Path,required=True)
    parser.add_argument("--side",choices=("left","right"),required=True)
    parser.add_argument("--source-directory",type=Path,required=True)
    parser.add_argument("--validation",type=Path,required=True)
    parser.add_argument("--output-directory",type=Path,required=True)
    parser.add_argument("--cycles",type=int,default=3)
    parser.add_argument("--weight",type=float,default=10000.)
    parser.add_argument("--weight-sweep",help="Comma-separated nonnegative weights; compare single-cycle solutions without advancing")
    parser.add_argument("--verify-next-load",action="store_true",
        help="Re-solve the load oracle from each candidate's exact prepared next boundary")
    parser.add_argument("--exploratory-refresh",action="store_true",
        help="Offline every-cycle KKT refresh ablation; later gradients lack full independent validation")
    parser.add_argument("--margin-only", action="store_true",
        help="With --exploratory-refresh, remove the fatigue cost; stop rather than fall back when local margin trust is exhausted")
    parser.add_argument("--hard-terminal-trust", action="store_true",
        help="For margin-only, impose the local margin trust box as terminal state bounds")
    parser.add_argument("--cpu",type=int,required=True)
    args=parser.parse_args()
    if args.cpu not in os.sched_getaffinity(0):
        parser.error("CPU outside allowed affinity")
    os.sched_setaffinity(0,{args.cpu})
    sweep=None if args.weight_sweep is None else tuple(float(value) for value in args.weight_sweep.split(","))
    report=run(args.receipt,args.side,args.source_directory,args.validation,args.output_directory,
        cycles=args.cycles,weight=args.weight,sweep_weights=sweep,
        verify_next_load=args.verify_next_load,exploratory_refresh=args.exploratory_refresh,
        margin_only=args.margin_only,hard_terminal_trust=args.hard_terminal_trust)
    print(json.dumps({key:value for key,value in report.items() if key!="cycles"},indent=2))


if __name__=="__main__":
    main()
