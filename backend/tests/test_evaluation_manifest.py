import json
from pathlib import Path

import pytest

from deepshield_api.evaluation.cli import main
from deepshield_api.evaluation.manifest import (
    BenchmarkManifestError,
    load_benchmark_manifest,
    render_benchmark_manifest,
    render_benchmark_manifest_schema,
    validate_corpus_references,
    write_benchmark_manifest,
    write_benchmark_manifest_schema,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
VALID_FIXTURE = REPOSITORY_ROOT / "data" / "fixtures" / "evaluation-benchmark.valid.json"
COMMITTED_SCHEMA = REPOSITORY_ROOT / "data" / "evaluation" / "benchmark-manifest.schema.json"


def test_committed_fixture_and_corpus_references_are_valid() -> None:
    manifest = load_benchmark_manifest(VALID_FIXTURE)

    validate_corpus_references(manifest, REPOSITORY_ROOT)

    assert manifest.manifest_version == 1
    assert len(manifest.cases) == 4
    assert manifest.target_case_count == 600


def test_load_rejects_missing_and_invalid_json(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkManifestError, match="unable to read"):
        load_benchmark_manifest(tmp_path / "missing.json")

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{invalid", encoding="utf-8")
    with pytest.raises(BenchmarkManifestError, match="not valid JSON"):
        load_benchmark_manifest(invalid)


def test_schema_is_deterministic_and_writable(tmp_path: Path) -> None:
    first = render_benchmark_manifest_schema()
    second = render_benchmark_manifest_schema()
    output = tmp_path / "nested" / "schema.json"

    write_benchmark_manifest_schema(output)

    assert first == second
    assert json.loads(first)["title"] == "BenchmarkManifest"
    assert output.read_text(encoding="utf-8") == first


def test_committed_benchmark_schema_is_current() -> None:
    assert COMMITTED_SCHEMA.read_text(encoding="utf-8") == render_benchmark_manifest_schema()


def test_manifest_render_is_deterministic_and_immutable(tmp_path: Path) -> None:
    manifest = load_benchmark_manifest(VALID_FIXTURE)
    output = tmp_path / "nested" / "benchmark.json"

    write_benchmark_manifest(output, manifest)

    assert output.read_text(encoding="utf-8") == render_benchmark_manifest(manifest)
    with pytest.raises(BenchmarkManifestError, match="refusing to replace"):
        write_benchmark_manifest(output, manifest)


def test_corpus_reference_validation_rejects_checksum_drift(tmp_path: Path) -> None:
    manifest = load_benchmark_manifest(VALID_FIXTURE)
    corpus_path = tmp_path / manifest.corpus_manifest_path
    corpus_path.parent.mkdir(parents=True)
    corpus_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(BenchmarkManifestError, match="checksum does not match"):
        validate_corpus_references(manifest, tmp_path)


def test_corpus_reference_validation_rejects_unknown_source(tmp_path: Path) -> None:
    payload = json.loads(VALID_FIXTURE.read_text(encoding="utf-8"))
    payload["cases"][0]["reference_source_ids"] = ["missing-official-source"]
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps(payload), encoding="utf-8")
    manifest = load_benchmark_manifest(fixture)

    with pytest.raises(BenchmarkManifestError, match="unknown corpus source IDs"):
        validate_corpus_references(manifest, REPOSITORY_ROOT)


def test_corpus_reference_validation_rejects_impossible_chunk(tmp_path: Path) -> None:
    payload = json.loads(VALID_FIXTURE.read_text(encoding="utf-8"))
    payload["cases"][0]["reference_chunk_ids"] = ["uscis-pm-f-chapter-5-v1-c99999"]
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps(payload), encoding="utf-8")
    manifest = load_benchmark_manifest(fixture)

    with pytest.raises(BenchmarkManifestError, match="unknown chunk ID"):
        validate_corpus_references(manifest, REPOSITORY_ROOT)


def test_cli_writes_checks_and_validates_schema(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    schema_path = tmp_path / "schema.json"

    assert main(["schema", "--output", str(schema_path)]) == 0
    assert main(["schema", "--output", str(schema_path), "--check"]) == 0
    assert "Schema is current" in capsys.readouterr().out

    assert (
        main(
            [
                "validate",
                str(VALID_FIXTURE),
                "--repository-root",
                str(REPOSITORY_ROOT),
            ]
        )
        == 0
    )
    assert "Validated draft benchmark" in capsys.readouterr().out


def test_cli_requires_frozen_status(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "validate",
                str(VALID_FIXTURE),
                "--repository-root",
                str(REPOSITORY_ROOT),
                "--require-frozen",
            ]
        )
        == 1
    )
    assert "must be frozen" in capsys.readouterr().err
