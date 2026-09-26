"""Coverage inventory gates are checked without the scientific dependencies."""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_application_tests.py"
SPEC = importlib.util.spec_from_file_location("application_ci", SCRIPT)
ci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ci)


class InventoryTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.root.joinpath("tests").mkdir()
        self.root.joinpath("tests/test_config.py").write_text("def test_roundtrip(): pass\n")
        self.manifest_path = self.root / "manifest.json"
        self.manifest = {
            "schema_version": 1,
            "mandatory_pr_tests": ["tests/test_config.py::test_roundtrip"],
            "groups": {
                "pr_unit": {
                    "modules": ["tests/test_config.py"],
                    "job": "application-pr",
                    "reason": "Configuration contract",
                    "markers": ["unit"],
                }
            },
        }

    def validate(self):
        self.manifest_path.write_text(json.dumps(self.manifest))
        return ci.validate_inventory(self.root, self.manifest_path)

    def test_new_test_module_cannot_be_omitted_from_ci(self):
        self.validate()
        self.root.joinpath("tests/test_new.py").write_text("def test_new(): pass\n")
        with self.assertRaisesRegex(ValueError, "unclassified=.*test_new"):
            self.validate()

    def test_stale_module_and_duplicate_are_rejected(self):
        self.manifest["groups"]["pr_unit"]["modules"].append("tests/test_deleted.py")
        with self.assertRaisesRegex(ValueError, "stale=.*test_deleted"):
            self.validate()
        self.manifest["groups"]["pr_unit"]["modules"] = ["tests/test_config.py"] * 2
        with self.assertRaisesRegex(ValueError, "Duplicate classification"):
            self.validate()

    def test_required_pr_test_cannot_be_classified_as_scientific(self):
        self.manifest["groups"]["scientific_root"] = self.manifest["groups"].pop("pr_unit")
        with self.assertRaisesRegex(ValueError, "Mandatory PR test"):
            self.validate()

    def test_scientific_override_cannot_move_a_required_pr_test(self):
        self.manifest["scientific_tests"] = ["tests/test_config.py::test_roundtrip"]
        with self.assertRaisesRegex(ValueError, "Mandatory PR test"):
            self.validate()

    def test_required_pr_test_must_pass_and_cannot_be_hidden_by_a_skip(self):
        report = ci.TestReport("pr", self.manifest, {"tests/test_config.py": "pr_unit"})
        report.nodeids = self.manifest["mandatory_pr_tests"]
        report.reports = [
            {
                "nodeid": report.nodeids[0],
                "phase": "call",
                "outcome": "skipped",
                "skip_kind": "dependency",
                "skip_reason": "No module named bioptim",
            }
        ]
        result = report.result(0, False)
        self.assertEqual(result["mandatory_pr_tests_not_passed"], report.nodeids)
        self.assertEqual(result["executed"], 0)
        report.reports[0].update(outcome="passed", skip_kind=None, skip_reason=None)
        self.assertEqual(report.result(0, False)["mandatory_pr_tests_not_passed"], [])

    def test_report_hooks_separate_collection_failure_from_execution_and_skips(self):
        report = ci.TestReport("scientific", self.manifest, {})
        report.pytest_runtest_logreport(
            SimpleNamespace(nodeid="tests/test_config.py::test_roundtrip", when="call", outcome="passed", longrepr=None)
        )
        report.pytest_collectreport(
            SimpleNamespace(
                nodeid="tests/test_missing.py", failed=True, skipped=False, longrepr="ImportError: dependency missing"
            )
        )
        report.pytest_collectreport(
            SimpleNamespace(
                nodeid="tests/test_optional.py",
                failed=False,
                skipped=True,
                longrepr="could not import optional_backend",
            )
        )
        result = report.result(2, False)
        self.assertEqual(result["passed"], 1)
        self.assertEqual(result["executed"], 1)
        self.assertEqual(len(result["collection_errors"]), 1)
        self.assertEqual(len(result["dependency_skips"]), 1)


if __name__ == "__main__":
    unittest.main()
