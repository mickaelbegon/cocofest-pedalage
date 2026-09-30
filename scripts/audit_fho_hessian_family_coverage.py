"""Measure exact Hessian sparsity coverage of post-shake Bioptim penalty families.

This diagnostic builds a saved FHO command up to the final uncompiled MX NLP,
then stops before IPOPT iterates.  It is deliberately read-only with respect to
campaigns: it neither calls the solver nor replaces a callback.  The registry
comes from the post-shake-provenance prototype and lets us answer a narrower
question than a timing benchmark: *which part of the native Hessian could a
future pre-shake fragment callback possibly own?*
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import runpy
import shlex
import sys
import time

import casadi as ca


class _CapturedNlp(BaseException):
    pass


class _StoppingSolver:
    def __init__(self, solver):
        self.solver = solver

    def call(self, _limits):
        raise _CapturedNlp()

    def stats(self):  # pragma: no cover - the controlled stop occurs first
        return self.solver.stats()


def _read_command(path: Path) -> list[str]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        return list(payload.get("command", payload))
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("command: "):
            return shlex.split(line.removeprefix("command: "))
    raise ValueError(f"No command in {path}")


def _replace_or_append(cli: list[str], flag: str, value: str) -> None:
    if flag in cli:
        cli[cli.index(flag) + 1] = value
    else:
        cli.extend((flag, value))


def _family(term) -> str:
    """A transparent, name-based grouping; raw names remain in the JSON."""
    if term.metadata.kind == "objective":
        return "objective"
    if term.metadata.thread_map_fragment or term.metadata.multi_thread:
        return "thread_map"
    name = term.metadata.penalty_name.lower()
    if any(token in name for token in ("dynamics", "collocation", "defect", "integrated_value")):
        return "dynamics_or_collocation"
    if any(token in name for token in ("continuity", "transition")):
        return "continuity_or_transition"
    return "other_constraint"


def _canonical_pairs(rows, cols) -> set[tuple[int, int]]:
    # CasADi may expose either triangular orientation.  Hessians are symmetric,
    # so compare structural ownership in orientation-independent coordinates.
    return {(min(int(r), int(c)), max(int(r), int(c))) for r, c in zip(rows, cols)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", type=Path, required=True, help="FHO command JSON or log")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hsl-library", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    command = _read_command(args.command.resolve())
    script, cli = Path(command[1]).resolve(), command[2:]
    _replace_or_append(cli, "--ipopt-hsl-library", str(args.hsl_library.resolve()))
    _replace_or_append(cli, "--output-json", str(output / "unused-result.json"))

    # The first path is the isolated Bioptim prototype, keeping the production
    # checkout and every active campaign untouched.
    branch = Path(__file__).resolve().parents[1] / ".benchmark-deps" / "bioptim-postshake-registry"
    sys.path.insert(0, str(branch))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import bioptim.interfaces.interface_utils as iu
    import bioptim.interfaces.ipopt_interface as ii

    original_nlpsol, original_solve = iu.nlpsol, ii.generic_solve
    captured: dict[str, object] = {}

    def nlpsol(name, plugin, nlp, options):
        solver = original_nlpsol(name, plugin, nlp, options)
        captured["solver"] = solver
        return _StoppingSolver(solver)

    def generic_solve(interface, expand_during_shake_tree=False):
        captured["interface"] = interface
        return original_solve(interface, expand_during_shake_tree)

    iu.nlpsol, ii.generic_solve = nlpsol, generic_solve
    sys.argv = [str(script), *cli]
    sys.path.insert(0, str(script.parent))
    try:
        runpy.run_path(str(script), run_name="__main__")
    except _CapturedNlp:
        pass
    finally:
        iu.nlpsol, ii.generic_solve = original_nlpsol, original_solve

    solver = captured["solver"]
    native = solver.get_function("nlp_hess_l")
    native_pairs = _canonical_pairs(*native.sparsity_out(0).get_triplet())
    owner = captured["interface"]
    tic = time.perf_counter()
    registry = owner.build_post_shake_penalty_registry()
    registry_seconds = time.perf_counter() - tic

    family_pairs: dict[str, set[tuple[int, int]]] = defaultdict(set)
    raw_entries: Counter[str] = Counter()
    terms_by_family: Counter[str] = Counter()
    names: dict[str, Counter[str]] = defaultdict(Counter)
    outside_native: Counter[str] = Counter()
    all_pairs: set[tuple[int, int]] = set()
    for term in registry.terms:
        family = _family(term)
        terms_by_family[family] += 1
        names[family][term.metadata.penalty_name] += 1
        local_rows, local_cols = term.hessian_sparsity.get_triplet()
        global_pairs = {
            (min(term.decision_indices[int(r)], term.decision_indices[int(c)]),
             max(term.decision_indices[int(r)], term.decision_indices[int(c)]))
            for r, c in zip(local_rows, local_cols)
        }
        raw_entries[family] += len(global_pairs)
        outside_native[family] += len(global_pairs - native_pairs)
        family_pairs[family].update(global_pairs & native_pairs)
        all_pairs.update(global_pairs & native_pairs)

    total = len(native_pairs)
    overlap = {
        left: {right: len(family_pairs[left] & family_pairs[right])
               for right in sorted(family_pairs) if right > left}
        for left in sorted(family_pairs)
    }
    result = {
        "mode": "post_shake_family_hessian_sparsity_coverage_no_solve",
        "command": [command[0], str(script), *cli],
        "casadi_version": ca.__version__,
        "nx": native.size1_in(0),
        "ng": native.size1_in(3),
        "native_hessian_triangular_nnz": native.sparsity_out(0).nnz(),
        "native_hessian_symmetric_pairs": total,
        "registry_build_s": registry_seconds,
        "registry_terms": len(registry.terms),
        "families": {
            family: {
                "terms": terms_by_family[family],
                "raw_local_symmetric_pairs_before_native_intersection": raw_entries[family],
                "pairs_in_native_hessian": len(family_pairs[family]),
                "fraction_of_native": len(family_pairs[family]) / max(total, 1),
                "pairs_not_present_in_native": outside_native[family],
                "penalty_names": dict(names[family]),
            }
            for family in sorted(terms_by_family)
        },
        "union_pairs_covered_by_registry": len(all_pairs),
        "union_fraction_of_native": len(all_pairs) / max(total, 1),
        "native_pairs_unattributed_by_registry": total - len(all_pairs),
        "family_pair_overlaps": overlap,
        "interpretation": (
            "Coverage is a structural upper bound only.  It does not measure runtime, "
            "does not establish that a term is repeated, and does not authorize replacing nlp_hess_l."
        ),
    }
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
