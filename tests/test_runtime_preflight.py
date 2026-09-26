"""Runtime evidence must remain usable before optional scientific imports."""

from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
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
