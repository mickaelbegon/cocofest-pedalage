"""Run inventoried application tests without importing excluded solver modules.

The manifest is checked before collection so a new test module cannot disappear
from PR coverage silently. This entry point uses only the standard library until
pytest starts; the numerical dependencies belong to the selected CI job.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / ".github/test-suites.json"
SUITE_GROUPS = {"pr": ("pr_unit", "pr_integration"), "scientific": ("scientific_root",)}


def validate_inventory(root=ROOT, manifest_path=MANIFEST):
    """Fail closed for stale, duplicate, unsafe, or unclassified modules."""
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported CI test inventory schema")
    owners = {}
    for group, definition in manifest["groups"].items():
        if not definition.get("job") or not definition.get("reason"):
            raise ValueError(f"Missing CI job or classification reason for {group}")
        for module in definition["modules"]:
            path = Path(module)
            if path.is_absolute() or ".." in path.parts or path.parts[0] != "tests":
                raise ValueError(f"Unsafe test module: {module}")
            if module in owners:
                raise ValueError(f"Duplicate classification: {module}")
            owners[module] = group
    actual = {path.relative_to(root).as_posix() for path in Path(root).joinpath("tests").rglob("test*.py")}
    missing, stale = sorted(actual - owners.keys()), sorted(owners.keys() - actual)
    if missing or stale:
        raise ValueError(f"CI inventory mismatch; unclassified={missing}; stale={stale}")
    for nodeid in manifest["mandatory_pr_tests"]:
        module = nodeid.split("::", 1)[0]
        if owners.get(module) not in SUITE_GROUPS["pr"]:
            raise ValueError(f"Mandatory PR test is outside the PR suite: {nodeid}")
    overrides = manifest.get("scientific_tests", [])
    if len(overrides) != len(set(overrides)):
        raise ValueError("Duplicate scientific test override")
    for nodeid in overrides:
        module, separator, test = nodeid.partition("::")
        if not separator or not test or owners.get(module) not in SUITE_GROUPS["pr"]:
            raise ValueError(f"Scientific override must name a test in a PR module: {nodeid}")
        if nodeid in manifest["mandatory_pr_tests"]:
            raise ValueError(f"Mandatory PR test cannot be moved to scientific suite: {nodeid}")
    return manifest, owners


class TestReport:
    def __init__(self, suite, manifest, owners):
        self.suite, self.manifest, self.owners = suite, manifest, owners
        self.nodeids = []
        self.reports = []
        self.collection_errors = []

    def pytest_configure(self, config):
        for name, description in (
            ("unit", "Headless configuration, launch and fixture contracts"),
            ("integration", "Replay, numerical, subprocess or scientific integration"),
            ("solver", "Production scientific stack or NLP/code generation required"),
        ):
            config.addinivalue_line("markers", f"{name}: {description}")

    def pytest_collection_modifyitems(self, config, items):
        import pytest

        overrides = set(self.manifest.get("scientific_tests", []))
        seen = {item.nodeid.split("[", 1)[0] for item in items}
        missing = overrides - seen
        if missing:
            raise pytest.UsageError(f"Stale scientific test overrides: {sorted(missing)}")
        selected, deselected = [], []
        for item in items:
            group = self.owners[item.path.relative_to(ROOT).as_posix()]
            scientific = group == "scientific_root" or item.nodeid.split("[", 1)[0] in overrides
            if scientific != (self.suite == "scientific"):
                deselected.append(item)
                continue
            selected.append(item)
            markers = ("integration", "solver") if scientific else self.manifest["groups"][group]["markers"]
            for marker in markers:
                item.add_marker(getattr(pytest.mark, marker))
        items[:] = selected
        if deselected:
            config.hook.pytest_deselected(items=deselected)
        self.nodeids = [item.nodeid for item in items]

    def pytest_runtest_logreport(self, report):
        if report.when == "call" or report.outcome != "passed":
            reason = str(report.longrepr) if report.outcome == "skipped" else None
            dependency_skip = reason and any(
                text in reason.lower()
                for text in ("could not import", "no module named", "not installed", "unavailable", "requires", "hsl")
            )
            self.reports.append(
                {
                    "nodeid": report.nodeid,
                    "phase": report.when,
                    "outcome": report.outcome,
                    "skip_kind": "dependency" if dependency_skip else "other" if reason else None,
                    "skip_reason": reason,
                }
            )

    def pytest_collectreport(self, report):
        if report.failed:
            self.collection_errors.append({"nodeid": report.nodeid, "error": str(report.longrepr)})
        elif report.skipped:
            self.reports.append(
                {
                    "nodeid": report.nodeid,
                    "phase": "collection",
                    "outcome": "skipped",
                    "skip_kind": "dependency" if "could not import" in str(report.longrepr).lower() else "other",
                    "skip_reason": str(report.longrepr),
                }
            )

    def result(self, exit_code, collect_only):
        counts = Counter(report["outcome"] for report in self.reports)
        passed = {
            report["nodeid"] for report in self.reports if report["phase"] == "call" and report["outcome"] == "passed"
        }
        mandatory_missing = []
        if self.suite == "pr" and not collect_only:
            mandatory_missing = sorted(set(self.manifest["mandatory_pr_tests"]) - passed)
        return {
            "suite": self.suite,
            "pytest_exit_code": int(exit_code),
            "collected": len(self.nodeids),
            "executed": sum(report["phase"] == "call" and report["outcome"] != "skipped" for report in self.reports),
            "passed": counts["passed"],
            "failed": counts["failed"],
            "collection_errors": self.collection_errors,
            "skipped": counts["skipped"],
            "dependency_skips": [report for report in self.reports if report["skip_kind"] == "dependency"],
            "other_skips": [report for report in self.reports if report["skip_kind"] == "other"],
            "mandatory_pr_tests_not_passed": mandatory_missing,
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITE_GROUPS, default="pr")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "local-results/ci-application")
    args, pytest_args = parser.parse_known_args(argv)
    try:
        manifest, owners = validate_inventory()
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    inventory = {
        group: {"modules": len(definition["modules"]), "job": definition["job"], "reason": definition["reason"]}
        for group, definition in manifest["groups"].items()
    }
    print(json.dumps({"inventory": inventory}, indent=2))
    if args.inventory_only:
        return 0
    # Scientific drivers and headless subprocess tests import this checkout.
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    import pytest

    report = TestReport(args.suite, manifest, owners)
    modules = [module for group in SUITE_GROUPS[args.suite] for module in manifest["groups"][group]["modules"]]
    if args.suite == "scientific":
        modules.extend(sorted({nodeid.split("::", 1)[0] for nodeid in manifest.get("scientific_tests", [])}))
    args.report_dir.mkdir(parents=True, exist_ok=True)
    exit_code = pytest.main(
        [*modules, "-ra", "--strict-markers", f"--junitxml={args.report_dir / 'junit.xml'}", *pytest_args],
        plugins=[report],
    )
    result = report.result(exit_code, "--collect-only" in pytest_args or "--co" in pytest_args)
    (args.report_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        lines = [
            f"### Application test suite: {args.suite}",
            "",
            "| Collected | Executed | Passed | Failed | Collection errors | Dependency skips | Other skips |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| {result['collected']} | {result['executed']} | {result['passed']} | {result['failed']} | "
            f"{len(result['collection_errors'])} | {len(result['dependency_skips'])} | {len(result['other_skips'])} |",
            "",
        ]
        for kind in ("dependency_skips", "other_skips"):
            for entry in result[kind]:
                lines.append(f"- {kind}: `{entry['nodeid']}` — {entry['skip_reason']}")
        if result["mandatory_pr_tests_not_passed"]:
            lines.extend(["", f"Required PR tests not passed: {result['mandatory_pr_tests_not_passed']}"])
        with Path(step_summary).open("a") as stream:
            stream.write("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))
    if result["mandatory_pr_tests_not_passed"]:
        print("Mandatory configuration/GUI/replay tests did not pass", file=sys.stderr)
        return 1
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
