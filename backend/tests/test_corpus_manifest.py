import json
from pathlib import Path

import pytest

from deepshield_api.corpus.cli import main
from deepshield_api.corpus.manifest import (
    ManifestError,
    load_source_manifest,
    render_source_manifest_schema,
    write_source_manifest_schema,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
VALID_FIXTURE = REPOSITORY_ROOT / "data" / "fixtures" / "source-manifest.valid.json"


def test_committed_synthetic_fixture_is_valid() -> None:
    manifest = load_source_manifest(VALID_FIXTURE)

    assert manifest.manifest_version == 1
    assert len(manifest.sources) == 1


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
