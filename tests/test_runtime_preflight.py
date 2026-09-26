"""Runtime evidence must remain usable before optional scientific imports."""

from importlib import metadata
import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

import pytest

from cocofest import runtime_preflight as preflight


@pytest.fixture
def empty_runtime(monkeypatch):
    for name in (*preflight._PACKAGES, "bioptim.interfaces.acados_interface"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    def absent(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(preflight.metadata, "version", absent)
    monkeypatch.setattr(
        preflight, "_mapped_libraries", lambda: {"status": "observed", "libraries": []}
    )


def test_absent_metadata_is_unknown_not_a_false_missing_dependency(empty_runtime):
    report = preflight.collect_runtime_preflight(
        solver="ipopt", expected_versions={"bioptim": "3.4.0"}
    )
    assert report["status"] == "incomplete"
    assert report["packages"]["bioptim"]["availability"] == "unknown"
    assert report["checks"][0]["status"] == "unknown"
    json.dumps(report, allow_nan=False)


def test_loaded_version_is_used_instead_of_installed_metadata(
    empty_runtime, monkeypatch
):
    monkeypatch.setattr(preflight.metadata, "version", lambda name: "9.9")
    monkeypatch.setitem(
        sys.modules,
        "casadi",
        SimpleNamespace(__version__="3.7.2", __file__="/custom/casadi.py"),
    )
    report = preflight.collect_runtime_preflight(
        solver="ipopt", expected_versions={"casadi": "3.8.0"}
    )
    assert report["status"] == "incompatible"
    record = report["packages"]["casadi"]
    assert record["loaded_version"] == "3.7.2"
    assert record["distributions"][0]["version"] == "9.9"
    assert report["checks"][0]["observed"] == "3.7.2"


@pytest.mark.parametrize(
    "alias,wire", [("ma57", "Ma57Solver"), ("mumps", "MumpsSolver")]
)
def test_configured_madnlp_backend_never_becomes_observed_evidence(
    empty_runtime, alias, wire
):
    report = preflight.collect_runtime_preflight(solver="madnlp", linear_solver=alias)
    assert report["requested_backend"]["interface_linear_solver"] == wire
    assert report["observed_backend"]["linear_solver"] is None
    assert report["observed_backend"]["solver_version"] is None


def test_loaded_factory_and_required_patch_contradictions(empty_runtime, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "bioptim",
        SimpleNamespace(Solver=SimpleNamespace(IPOPT=lambda: None)),
    )
    monkeypatch.setitem(
        sys.modules,
        "bioptim.interfaces.acados_interface",
        SimpleNamespace(AcadosInterface=type("Interface", (), {})),
    )
    report = preflight.collect_runtime_preflight(
        solver="acados", require_acados_ding_patch=True
    )
    assert report["status"] == "incompatible"
    assert {
        check["name"] for check in report["checks"] if check["status"] == "fail"
    } == {
        "bioptim_solver_factory",
        "required_acados_ding_patch",
    }
    monkeypatch.setitem(
        sys.modules,
        "bioptim",
        SimpleNamespace(Solver=SimpleNamespace(ACADOS=lambda: None)),
    )
    monkeypatch.setitem(
        sys.modules,
        "bioptim.interfaces.acados_interface",
        SimpleNamespace(
            AcadosInterface=type("Interface", (), {"_cocofest_ding_local_patch": True})
        ),
    )
    report = preflight.collect_runtime_preflight(
        solver="acados", require_acados_ding_patch=True
    )
    assert report["status"] == "incomplete"  # native ABI remains untested
    assert report["patches"][0]["status"] == "installed"


def test_process_maps_unavailable_is_explicit(empty_runtime, monkeypatch):
    def unavailable(self, *args, **kwargs):
        raise OSError("no procfs")

    monkeypatch.undo()
    monkeypatch.setattr(Path, "read_text", unavailable)
    assert preflight._mapped_libraries()["status"] == "unavailable"


def test_report_cli_works_without_importing_scientific_stack():
    script = """
import sys
class RejectImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'casadi', 'bioptim', 'biorbd', 'biorbd_casadi', 'acados_template', 'examples'}:
            raise AssertionError('Scientific import: ' + fullname)
sys.meta_path.insert(0, RejectImports())
from cocofest.runtime_preflight import main
raise SystemExit(main(['--solver', 'madnlp', '--linear-solver', 'ma57']))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["schema"] == "cocofest-runtime-preflight-v1"
    assert report["status"] == "incomplete"


def test_cli_preserves_existing_evidence(empty_runtime, tmp_path):
    output = tmp_path / "preflight.json"
    assert preflight.main(["--solver", "ipopt", "--output", str(output)]) == 0
    original = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        preflight.main(["--solver", "madnlp", "--output", str(output)])
    assert error.value.code == 2
    assert output.read_bytes() == original


@pytest.mark.parametrize(
    "settings",
    [
        {"solver": "invalid"},
        {"solver": "ipopt", "linear_solver": ""},
        {"solver": "ipopt", "expected_versions": {"unknown": "1"}},
        {"solver": "ipopt", "expected_versions": {"casadi": 3}},
    ],
)
def test_invalid_requests_rejected(settings):
    with pytest.raises(ValueError):
        preflight.collect_runtime_preflight(**settings)


def test_worker_sidecars_are_unique_and_record_actual_worker(empty_runtime, tmp_path):
    path = tmp_path / "case" / "result.json"
    first = preflight.record_worker_preflight(
        result_path=path, solver="madnlp", linear_solver="ma57"
    )
    original = Path(first["path"]).read_bytes()
    second = preflight.record_worker_preflight(
        result_path=path, solver="madnlp", linear_solver="mumps"
    )
    assert first["status"] == second["status"] == "recorded"
    assert first["path"] != second["path"]
    assert Path(first["path"]).read_bytes() == original
    report = json.loads(original)
    assert report["runtime"]["pid"] == os.getpid()
    assert report["observation_stage"] == "before_solve_case"
    assert report["requested_backend"]["interface_linear_solver"] == "Ma57Solver"
    assert report["observed_backend"]["linear_solver"] is None
    assert first["report_status"] == "incomplete"
    assert first["wall_time_s"] >= 0
    assert not path.exists()


def test_worker_records_collection_failure_without_stopping_solve(
    monkeypatch, tmp_path
):
    def fail(**kwargs):
        raise RuntimeError("metadata unavailable")

    monkeypatch.setattr(preflight, "collect_runtime_preflight", fail)
    result = preflight.record_worker_preflight(
        result_path=tmp_path / "result.json", solver="ipopt"
    )
    assert result["status"] == "unavailable"
    assert result["path"] is None
    assert result["error"] == "RuntimeError: metadata unavailable"


def test_worker_records_write_failure_without_stopping_solve(empty_runtime, tmp_path):
    parent = tmp_path / "file-not-directory"
    parent.write_text("preserve me")
    result = preflight.record_worker_preflight(
        result_path=parent / "result.json", solver="ipopt"
    )
    assert result["status"] == "unavailable"
    assert "FileExistsError" in result["error"]
    assert result["report_status"] == "incomplete"
    assert parent.read_text() == "preserve me"


@pytest.mark.parametrize(
    "fails,codegen", [(False, False), (True, False), (False, True)]
)
def test_benchmark_worker_records_before_native_entry_and_retains_evidence(
    empty_runtime, tmp_path, fails, codegen
):
    # Execute the real worker seam without importing the scientific example.
    # The fake solve asserts evidence exists before any native work can begin.
    source = (
        Path(__file__).resolve().parents[1]
        / "examples/fes_multibody/cycling/cycling_fes_solver_comparison.py"
    )
    node = next(
        item
        for item in ast.parse(source.read_text()).body
        if isinstance(item, ast.FunctionDef) and item.name == "_run_benchmark_case"
    )
    destination = tmp_path / "result.json"
    arguments = argparse.Namespace(ipopt_linear_solver="ma57", ipopt_c_compile=codegen)
    seen = []

    def solve(args, *, echo):
        reports = list(tmp_path.glob("result.runtime-preflight.ipopt.*.json"))
        assert len(reports) == 1
        report = json.loads(reports[0].read_text())
        assert report["requested_backend"]["linear_solver"] == "ma57"
        assert args is arguments
        seen.append("solve")
        if fails:
            raise RuntimeError("native setup failure")
        return {"success": True}

    namespace = {
        "argparse": argparse,
        "Path": Path,
        "perf_counter": perf_counter,
        "record_worker_preflight": preflight.record_worker_preflight,
        "solve_case": solve,
        "_failed_solver_result": lambda args, error, elapsed: {"error": str(error)},
        "os": os,
        "TemporaryDirectory": TemporaryDirectory,
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    cwd = Path.cwd()
    result = namespace["_run_benchmark_case"](
        "ipopt", arguments, echo=False, runtime_preflight_result_path=destination
    )
    assert seen == ["solve"]
    assert Path.cwd() == cwd
    assert result["runtime_preflight"]["status"] == "recorded"
    if fails:
        assert result["error"] == "native setup failure"
    else:
        assert result["success"] is True
