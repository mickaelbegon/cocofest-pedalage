"""Artifact catalog behavior with no scientific solver dependency."""

import hashlib
import json
from pathlib import Path

import pytest

from cocofest.evaluation.experiment_catalog import index_campaigns, main, query_catalog

FIXTURES = Path(__file__).parent / "fixtures" / "experiment_catalog"


def test_existing_campaign_is_indexed_without_implying_certification():
    catalog = index_campaigns([FIXTURES])
    assert catalog["errors"] == []
    (record,) = catalog["records"]
    artifact = FIXTURES / "K7" / "ipopt-ma57" / "result.json"
    assert (
        record["artifact"]["sha256"]
        == hashlib.sha256(artifact.read_bytes()).hexdigest()
    )
    assert record["artifact"]["uri"] == artifact.resolve().as_uri()
    assert record["dimensions"]["frequency_hz"] == pytest.approx(30)
    assert record["dimensions"]["linear_solver"] == "ma57"
    assert record["dimension_sources"]["stage"] == "path_component"
    assert record["status"]["catalog_status"] == "reported_success"
    assert record["status"]["physical_success"] is False
    assert record["audits"]["mechanical_equivalence_audit"]["passes_tolerance"] is False
    assert "wall_time_s" not in record["metrics"]


def test_deduplication_order_and_hash_track_source_changes(tmp_path):
    source = tmp_path / "result.json"
    source.write_text(json.dumps({"solver": "ipopt", "success": True}))
    first = index_campaigns([tmp_path, source, tmp_path])
    assert first["scanned_file_count"] == 1
    assert first == index_campaigns([source, tmp_path])
    source.write_text(json.dumps({"solver": "ipopt", "success": False}))
    second = index_campaigns([tmp_path])
    assert (
        first["records"][0]["artifact"]["sha256"]
        != second["records"][0]["artifact"]["sha256"]
    )
    assert second["records"][0]["status"]["catalog_status"] == "reported_failure"


def test_multiple_results_and_unknown_frequency_are_preserved(tmp_path):
    source = tmp_path / "result.json"
    source.write_text(
        json.dumps(
            {
                "configurations": {"acados": {"stimulations_per_cycle": 35}},
                "results": {
                    "acados": {"solver_success": True},
                    "madnlp": {"error": "failed"},
                },
            }
        )
    )
    catalog = index_campaigns([tmp_path])
    acados, madnlp = catalog["records"]
    assert acados["dimensions"]["frequency_hz"] is None
    assert acados["status"]["catalog_status"] == "unknown"
    assert madnlp["status"]["catalog_status"] == "error"
    assert query_catalog(catalog, frequency_hz=35) == []
    assert query_catalog(catalog, solver="acados") == [acados]


def test_frequency_uses_cycle_duration_and_stage_uses_explicit_config(tmp_path):
    source = tmp_path / "result.json"
    source.write_text(
        json.dumps(
            {
                "solver": "ipopt",
                "configuration": {
                    "stimulations_per_cycle": 35,
                    "cycle_duration": 2,
                    "stage": "K9",
                    "formulation": "isokinetic",
                },
            }
        )
    )
    catalog = index_campaigns([source])
    assert (
        len(
            query_catalog(
                catalog,
                solver="ipopt",
                formulation="isokinetic",
                stage="K9",
                frequency_hz=17.5,
            )
        )
        == 1
    )
    assert query_catalog(catalog, formulation="dynamic") == []


def test_parse_failures_invalid_entries_and_missing_roots_are_visible(tmp_path):
    bad = tmp_path / "bad" / "result.json"
    bad.parent.mkdir()
    bad.write_text("{")
    partial = tmp_path / "partial" / "result.json"
    partial.parent.mkdir()
    partial.write_text(json.dumps({"results": [None, {"solver": "ipopt"}]}))
    catalog = index_campaigns([tmp_path, tmp_path / "missing"])
    assert len(catalog["errors"]) == 3
    assert len(catalog["records"]) == 1
    assert any(error.get("entry") == 0 for error in catalog["errors"])
    assert next(error for error in catalog["errors"] if error["uri"] == bad.as_uri())[
        "sha256"
    ]


def test_dry_run_is_non_destructive_and_query_cli_matches_library(tmp_path, capsys):
    output = tmp_path / "catalog.json"
    output.write_text("existing catalog")
    assert main(["index", str(FIXTURES), "--output", str(output), "--dry-run"]) == 0
    catalog = json.loads(capsys.readouterr().out)
    assert output.read_text() == "existing catalog"
    assert main(["index", str(FIXTURES), "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == catalog
    assert (
        main(
            [
                "query",
                str(output),
                "--solver",
                "ipopt",
                "--stage",
                "K7",
                "--frequency-hz",
                "30",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == catalog["records"]
    assert list(tmp_path.glob(".catalog-*")) == []


def test_output_cannot_overwrite_source_even_for_malformed_artifact(tmp_path):
    source = tmp_path / "result.json"
    source.write_text("{")
    with pytest.raises(SystemExit, match="2"):
        main(["index", str(tmp_path), "--output", str(source)])
    assert source.read_text() == "{"


def test_nonfinite_legacy_metrics_become_explicit_null_and_errors_set_exit_code(
    tmp_path, capsys
):
    source = tmp_path / "result.json"
    source.write_text('{"solver": "ipopt", "hot_solver_time_median_s": NaN}')
    assert main(["index", str(tmp_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["records"][0]["metrics"]["hot_solver_time_median_s"] is None
    assert main(["index", str(tmp_path / "missing")]) == 1
    capsys.readouterr()


def test_unknown_catalog_version_is_rejected():
    with pytest.raises(ValueError, match="schema version"):
        query_catalog({"schema_version": 999, "records": []})


def test_sidecar_provenance_identity_and_hash_queries():
    catalog = index_campaigns([FIXTURES])
    (record,) = catalog["records"]
    evidence = record["configuration_provenance"]
    assert evidence["status"] == "hash_verified"
    assert evidence["config_hash"] == evidence["computed_config_hash"]
    assert evidence["effective"]["physical"]["model_config"] == "/recorded/model.json"
    assert record["identity"]["case"] == "K7/ipopt-ma57"
    assert query_catalog(
        catalog,
        campaign=FIXTURES.name,
        case="K7/ipopt-ma57",
        config_hash=evidence["config_hash"],
    ) == [record]
    assert query_catalog(catalog, config_hash="unknown") == []


def _write_sidecar(directory, *, corrupt=False):
    source = FIXTURES / "K7/ipopt-ma57/effective-configuration.json"
    payload = json.loads(source.read_text())
    payload["profile"] = "K7"
    if corrupt:
        payload["effective"]["physical"]["mechanics"] = "full"
    sidecar = directory / "effective-configuration.json"
    sidecar.write_text(json.dumps(payload))
    return sidecar


def test_sidecar_fallback_obeys_result_configuration_and_solver_scope(tmp_path):
    source = tmp_path / "result.json"
    source.write_text(
        json.dumps(
            {
                "campaign_id": "declared",
                "results": {
                    "ipopt": {
                        "case_id": "case-a",
                        "configuration": {"ipopt_linear_solver": "mumps"},
                    },
                    "acados": {},
                },
            }
        )
    )
    _write_sidecar(tmp_path)
    catalog = index_campaigns([tmp_path])
    ipopt, acados = catalog["records"]
    assert ipopt["dimensions"]["linear_solver"] == "mumps"
    assert ipopt["dimensions"]["mechanical_formulation"] == "reduced"
    assert ipopt["dimensions"]["stage"] == "K7"
    assert ipopt["identity"]["campaign"] == "declared"
    assert ipopt["identity"]["case"] == "case-a"
    assert ipopt["configuration_provenance"]["conflicts"]["ipopt_linear_solver"] == {
        "result": "mumps",
        "effective_configuration": "ma57",
    }
    assert (
        query_catalog(
            catalog, config_hash=ipopt["configuration_provenance"]["config_hash"]
        )
        == []
    )
    assert acados["dimensions"]["stage"] is None
    assert acados["configuration_provenance"]["solver_matches_result"] is False
    assert (
        query_catalog(
            catalog,
            solver="acados",
            config_hash=ipopt["configuration_provenance"]["config_hash"],
        )
        == []
    )


@pytest.mark.parametrize("damage", ["hash", "json", "schema", "sections"])
def test_invalid_sidecars_retain_result_and_report_error_without_modification(
    tmp_path, damage
):
    source = tmp_path / "result.json"
    source.write_text('{"solver": "ipopt", "success": true}')
    sidecar = _write_sidecar(tmp_path, corrupt=damage == "hash")
    if damage == "json":
        sidecar.write_text("{")
    elif damage in ("schema", "sections"):
        payload = json.loads(sidecar.read_text())
        if damage == "schema":
            payload["schema_version"] = 999
        else:
            payload["effective"]["physical"] = []
        sidecar.write_text(json.dumps(payload))
    before = (source.read_bytes(), sidecar.read_bytes())
    catalog = index_campaigns([tmp_path])
    assert len(catalog["records"]) == 1
    assert len(catalog["errors"]) == 1
    assert catalog["errors"][0]["uri"] == sidecar.as_uri()
    assert catalog["records"][0]["configuration_provenance"]["status"] == "invalid"
    assert query_catalog(catalog, config_hash="unknown") == []
    assert (source.read_bytes(), sidecar.read_bytes()) == before


def test_sidecar_in_ancestor_is_not_associated_and_output_is_protected(tmp_path):
    sidecar = _write_sidecar(tmp_path)
    case = tmp_path / "case"
    case.mkdir()
    (case / "result.json").write_text('{"solver": "ipopt"}')
    catalog = index_campaigns([tmp_path])
    assert catalog["records"][0]["configuration_provenance"]["status"] == "not_recorded"
    before = sidecar.read_bytes()
    with pytest.raises(SystemExit, match="2"):
        main(["index", str(tmp_path), "--output", str(sidecar)])
    assert sidecar.read_bytes() == before


def test_sidecar_symlink_target_cannot_be_replaced_by_catalog(tmp_path):
    (tmp_path / "result.json").write_text('{"solver": "ipopt"}')
    sidecar = _write_sidecar(tmp_path)
    original = tmp_path / "inputs.json"
    sidecar.rename(original)
    sidecar.symlink_to(original)
    before = original.read_bytes()
    with pytest.raises(SystemExit, match="2"):
        main(["index", str(tmp_path), "--output", str(original)])
    assert original.read_bytes() == before


def test_sidecar_hash_matches_real_resolver(tmp_path):
    from cocofest.simulation.resolved_config import resolve_config

    resolved = resolve_config({}, profile="K7", root=tmp_path)
    (tmp_path / "result.json").write_text('{"solver": "ipopt"}')
    (tmp_path / "effective-configuration.json").write_text(resolved.to_json())
    catalog = index_campaigns([tmp_path])
    assert catalog["errors"] == []
    assert len(query_catalog(catalog, config_hash=resolved.config_hash, stage="K7")) == 1
