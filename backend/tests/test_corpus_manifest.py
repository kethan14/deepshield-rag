import json
from datetime import datetime
from pathlib import Path

import pytest

from deepshield_api.corpus.cli import main
from deepshield_api.corpus.collection import CollectionPolicy, FetchedDocument
from deepshield_api.corpus.manifest import (
    ManifestError,
    load_source_manifest,
    render_source_manifest,
    render_source_manifest_schema,
    write_source_manifest,
    write_source_manifest_schema,
)
from deepshield_api.corpus.models import SourceRecord

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
VALID_FIXTURE = REPOSITORY_ROOT / "data" / "fixtures" / "source-manifest.valid.json"
F1_SOURCE_MANIFEST = REPOSITORY_ROOT / "data" / "manifests" / "f1-policy-source-manifest.v1.json"


def test_committed_synthetic_fixture_is_valid() -> None:
    manifest = load_source_manifest(VALID_FIXTURE)

    assert manifest.manifest_version == 1
    assert len(manifest.sources) == 1


def test_committed_f1_source_registry_is_valid_and_planned() -> None:
    manifest = load_source_manifest(F1_SOURCE_MANIFEST)

    assert manifest.manifest_version == 1
    assert len(manifest.sources) == 17
    assert all(source.collection_status.value == "planned" for source in manifest.sources)
    assert all(source.retrieved_at is None for source in manifest.sources)
    assert all(source.raw_path is None for source in manifest.sources)
    assert all(source.sha256 is None for source in manifest.sources)

    publishers = {source.publisher.value for source in manifest.sources}
    assert publishers == {
        "department-of-state",
        "ecfr",
        "ice-sevp",
        "study-in-the-states",
        "uscis",
    }

    covered_topics = {tag for source in manifest.sources for tag in source.scope_tags}
    assert {
        "cpt",
        "employment",
        "full-course-of-study",
        "maintaining-status",
        "opt",
        "reinstatement",
        "stem-opt",
        "transfer",
        "travel",
        "visa-application",
    } <= covered_topics


def test_load_manifest_rejects_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.json"

    with pytest.raises(ManifestError, match="unable to read"):
        load_source_manifest(missing_path)


def test_load_manifest_rejects_invalid_json(tmp_path: Path) -> None:
    manifest_path = tmp_path / "invalid.json"
    manifest_path.write_text("{invalid", encoding="utf-8")

    with pytest.raises(ManifestError, match="not valid JSON"):
        load_source_manifest(manifest_path)


def test_load_manifest_rejects_invalid_schema(tmp_path: Path) -> None:
    manifest_path = tmp_path / "invalid-schema.json"
    manifest_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ManifestError, match="validation failed"):
        load_source_manifest(manifest_path)


def test_schema_render_is_deterministic_json() -> None:
    first_render = render_source_manifest_schema()
    second_render = render_source_manifest_schema()

    assert first_render == second_render
    assert first_render.endswith("\n")
    assert json.loads(first_render)["title"] == "SourceManifest"


def test_schema_write_creates_parent_directory(tmp_path: Path) -> None:
    schema_path = tmp_path / "nested" / "source-manifest.schema.json"

    write_source_manifest_schema(schema_path)

    assert schema_path.read_text(encoding="utf-8") == render_source_manifest_schema()


def test_manifest_render_and_write_are_deterministic(tmp_path: Path) -> None:
    manifest = load_source_manifest(VALID_FIXTURE)
    manifest_path = tmp_path / "nested" / "manifest.json"

    first_render = render_source_manifest(manifest)
    second_render = render_source_manifest(manifest)
    write_source_manifest(manifest_path, manifest)

    assert first_render == second_render
    assert first_render.endswith("\n")
    assert manifest_path.read_text(encoding="utf-8") == first_render

    with pytest.raises(ManifestError, match="refusing to replace"):
        write_source_manifest(manifest_path, manifest)


def test_collect_cli_writes_new_manifest_and_raw_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "collected-manifest.json"

    def fetcher(source: SourceRecord, policy: CollectionPolicy) -> FetchedDocument:
        return FetchedDocument(
            final_url="https://www.uscis.gov/synthetic-test-fixture",
            media_type="text/html",
            content=b"<html>Synthetic fixture</html>",
        )

    assert (
        main(
            [
                "collect",
                str(VALID_FIXTURE),
                "synthetic-uscis-guidance",
                "--document-version",
                "1",
                "--output",
                str(output_path),
                "--repository-root",
                str(tmp_path),
            ],
            fetcher=fetcher,
            clock=lambda: datetime.fromisoformat("2026-09-22T12:00:00+00:00"),
        )
        == 0
    )
    output = capsys.readouterr().out
    collected = load_source_manifest(output_path)

    assert "wrote manifest version 2" in output
    assert collected.sources[0].collection_status.value == "collected"
    assert collected.sources[0].raw_path is not None
    assert (tmp_path / Path(collected.sources[0].raw_path)).is_file()


def test_collect_cli_rejects_in_place_output(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "collect",
                str(VALID_FIXTURE),
                "synthetic-uscis-guidance",
                "--document-version",
                "1",
                "--output",
                str(VALID_FIXTURE),
            ]
        )
        == 1
    )
    assert "must differ" in capsys.readouterr().err


def test_schema_cli_writes_and_checks_schema(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    schema_path = tmp_path / "source-manifest.schema.json"

    assert main(["schema", "--output", str(schema_path)]) == 0
    assert "Wrote schema" in capsys.readouterr().out
    assert main(["schema", "--output", str(schema_path), "--check"]) == 0
    assert "Schema is current" in capsys.readouterr().out


def test_schema_cli_detects_missing_or_stale_schema(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    schema_path = tmp_path / "source-manifest.schema.json"

    assert main(["schema", "--output", str(schema_path), "--check"]) == 1
    assert "missing or out of date" in capsys.readouterr().err
    schema_path.write_text("{}\n", encoding="utf-8")
    assert main(["schema", "--output", str(schema_path), "--check"]) == 1
    assert "missing or out of date" in capsys.readouterr().err


def test_validate_cli_accepts_valid_manifest(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate", str(VALID_FIXTURE)]) == 0

    output = capsys.readouterr().out
    assert "Validated manifest version 1 with 1 source(s)" in output


def test_validate_cli_reports_invalid_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path = tmp_path / "invalid.json"
    manifest_path.write_text("{}\n", encoding="utf-8")

    assert main(["validate", str(manifest_path)]) == 1
    assert "validation failed" in capsys.readouterr().err
