import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from deepshield_api.corpus.chunking import (
    ChunkingError,
    ChunkingPolicy,
    ChunkingReport,
    ChunkingSourceSummary,
    ChunkRecord,
    chunk_extraction_report,
    chunk_source_text,
    render_chunk_dataset,
    render_chunking_report,
    write_chunk_dataset,
    write_chunking_report,
)
from deepshield_api.corpus.cli import main
from deepshield_api.corpus.extraction import ExtractionRecord, ExtractionReport
from deepshield_api.corpus.manifest import write_source_manifest
from deepshield_api.corpus.models import SourceManifest, SourceRecord

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
F1_EXTRACTION_REPORT = (
    REPOSITORY_ROOT / "data" / "manifests" / "f1-policy-extraction-report.v1.json"
)
F1_CHUNKING_REPORT = REPOSITORY_ROOT / "data" / "manifests" / "f1-policy-chunking-report.v1.json"


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def source_data(**overrides: Any) -> dict[str, Any]:
    raw = b"synthetic raw source"
    payload: dict[str, Any] = {
        "source_id": "synthetic-uscis-guidance",
        "document_version": 1,
        "title": "Synthetic USCIS Guidance",
        "publisher": "uscis",
        "canonical_url": "https://www.uscis.gov/example",
        "document_type": "guidance-page",
        "media_type": "text/plain",
        "scope_tags": ["status", "employment"],
        "rights_notes": "Synthetic test fixture.",
        "collection_status": "collected",
        "retrieved_at": "2026-09-25T12:00:00Z",
        "raw_path": "data/raw/uscis/synthetic-uscis-guidance/v1.txt",
        "sha256": sha256(raw),
    }
    payload.update(overrides)
    return payload


def make_source(**overrides: Any) -> SourceRecord:
    return SourceRecord.model_validate(source_data(**overrides))


def make_extraction(source: SourceRecord, text: str) -> ExtractionRecord:
    assert source.raw_path is not None
    assert source.sha256 is not None
    content = (text + "\n").encode()
    return ExtractionRecord(
        source_id=source.source_id,
        document_version=source.document_version,
        media_type=source.media_type,
        raw_path=source.raw_path,
        raw_sha256=source.sha256,
        extracted_path=(
            f"data/interim/{source.publisher.value}/{source.source_id}/"
            f"v{source.document_version}/extractor-1.1.txt"
        ),
        extracted_sha256=sha256(content),
        extraction_method="utf8-text-v1",
        extraction_profile="plain-text-v1",
        source_encoding="utf-8",
        character_count=len(text),
        line_count=text.count("\n") + 1,
    )


def write_chunking_inputs(root: Path, text: str) -> tuple[Path, Path, SourceRecord]:
    source = make_source()
    manifest = SourceManifest(
        manifest_version=1,
        created_at=datetime.fromisoformat("2026-09-25T12:00:00+00:00"),
        sources=[source],
    )
    manifest_path = root / "data/manifests/source.json"
    write_source_manifest(manifest_path, manifest)
    extraction = make_extraction(source, text)
    extracted_path = root / extraction.extracted_path
    extracted_path.parent.mkdir(parents=True)
    extracted_path.write_text(text + "\n", encoding="utf-8", newline="\n")
    report = ExtractionReport(
        report_version=1,
        source_manifest_path="data/manifests/source.json",
        source_manifest_sha256=sha256(manifest_path.read_bytes()),
        records=[extraction],
    )
    report_path = root / "data/manifests/extraction.json"
    report_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report_path, extracted_path, source


def test_chunking_policy_rejects_inconsistent_limits() -> None:
    with pytest.raises(ValidationError, match="smaller than"):
        ChunkingPolicy(max_characters=500, overlap_characters=500)
    with pytest.raises(ValidationError, match="must not exceed"):
        ChunkingPolicy(max_characters=500, min_characters=501)


def test_short_source_produces_one_exact_traceable_chunk() -> None:
    text = "F-1 students must maintain status under the applicable rules."
    source = make_source()
    extraction = make_extraction(source, text)

    chunks = chunk_source_text(text, source, extraction)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "synthetic-uscis-guidance-v1-c00000"
    assert chunks[0].text == text
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == len(text)
    assert chunks[0].text_sha256 == sha256(text.encode())


def test_long_source_is_deterministic_bounded_and_has_no_content_gaps() -> None:
    paragraphs = [
        f"Section {index}. This is official synthetic policy guidance for an F-1 student. "
        "It contains enough detail to test deterministic boundary selection."
        for index in range(30)
    ]
    text = "\n".join(paragraphs)
    source = make_source()
    extraction = make_extraction(source, text)
    policy = ChunkingPolicy(max_characters=500, overlap_characters=80, min_characters=120)

    first = chunk_source_text(text, source, extraction, policy=policy)
    second = chunk_source_text(text, source, extraction, policy=policy)

    assert first == second
    assert len(first) > 1
    assert first[0].char_start == 0
    assert first[-1].char_end == len(text)
    assert all(chunk.character_count <= policy.max_characters for chunk in first)
    assert all(chunk.text == text[chunk.char_start : chunk.char_end] for chunk in first)
    assert all(chunk.chunk_index == index for index, chunk in enumerate(first))
    for previous, current in zip(first, first[1:], strict=False):
        assert current.char_start - previous.char_start >= policy.min_characters
        assert current.char_start <= previous.char_end


def test_chunking_uses_sentence_boundary_for_single_long_line() -> None:
    text = " ".join(f"Sentence {index} has policy detail." for index in range(60))
    source = make_source()
    extraction = make_extraction(source, text)
    policy = ChunkingPolicy(max_characters=240, overlap_characters=40, min_characters=100)

    chunks = chunk_source_text(text, source, extraction, policy=policy)

    assert len(chunks) > 1
    assert all(chunk.text.endswith(".") for chunk in chunks)


def test_chunk_record_rejects_inconsistent_text_hash() -> None:
    text = "Traceable policy text."
    source = make_source()
    extraction = make_extraction(source, text)
    payload = chunk_source_text(text, source, extraction)[0].model_dump(mode="json")
    payload["text_sha256"] = "0" * 64

    with pytest.raises(ValidationError, match="must match the chunk text"):
        ChunkRecord.model_validate(payload)


def test_chunking_report_rejects_inconsistent_totals() -> None:
    with pytest.raises(ValidationError, match="total_chunks"):
        ChunkingReport(
            report_version=1,
            extraction_report_path="data/manifests/extraction.json",
            extraction_report_sha256="a" * 64,
            chunks_path="data/processed/chunks.jsonl",
            chunks_sha256="b" * 64,
            policy=ChunkingPolicy(),
            total_sources=1,
            total_chunks=2,
            total_source_characters=100,
            total_chunk_characters=100,
            sources=[
                ChunkingSourceSummary(
                    source_id="synthetic-source",
                    document_version=1,
                    extracted_sha256="c" * 64,
                    source_character_count=100,
                    chunk_count=1,
                    chunk_character_count=100,
                    min_chunk_characters=100,
                    max_chunk_characters=100,
                )
            ],
        )


def test_chunk_extraction_report_writes_deterministic_immutable_artifacts(
    tmp_path: Path,
) -> None:
    text = "\n".join(
        f"Policy section {index}. Synthetic guidance for deterministic testing."
        for index in range(20)
    )
    report_path, _, _ = write_chunking_inputs(tmp_path, text)
    chunks_path = tmp_path / "data/processed/chunks.jsonl"
    output_report_path = tmp_path / "data/manifests/chunking.json"
    policy = ChunkingPolicy(max_characters=400, overlap_characters=60, min_characters=100)

    chunks, report = chunk_extraction_report(
        report_path,
        chunks_path,
        tmp_path,
        policy=policy,
    )
    write_chunk_dataset(chunks_path, chunks)
    write_chunking_report(output_report_path, report)

    assert report.total_sources == 1
    assert report.total_chunks == len(chunks)
    assert report.chunks_sha256 == sha256(chunks_path.read_bytes())
    assert chunks_path.read_bytes() == render_chunk_dataset(chunks)
    assert output_report_path.read_text(encoding="utf-8") == render_chunking_report(report)
    write_chunk_dataset(chunks_path, chunks)
    with pytest.raises(ChunkingError, match="refusing to replace"):
        write_chunk_dataset(chunks_path, [chunks[0]])
    with pytest.raises(ChunkingError, match="refusing to replace"):
        write_chunking_report(output_report_path, report)


def test_chunk_extraction_report_rejects_tampered_text(tmp_path: Path) -> None:
    report_path, extracted_path, _ = write_chunking_inputs(tmp_path, "Official policy text.")
    extracted_path.write_text("Tampered policy text.\n", encoding="utf-8")

    with pytest.raises(ChunkingError, match="checksum mismatch"):
        chunk_extraction_report(
            report_path,
            tmp_path / "data/processed/chunks.jsonl",
            tmp_path,
        )


def test_committed_f1_chunking_report_is_complete_and_traceable() -> None:
    report = ChunkingReport.model_validate_json(F1_CHUNKING_REPORT.read_text(encoding="utf-8"))

    assert report.report_version == 1
    assert report.chunker_version == "1.0"
    assert report.extraction_report_sha256 == sha256(F1_EXTRACTION_REPORT.read_bytes())
    assert report.policy == ChunkingPolicy(
        max_characters=1800,
        overlap_characters=250,
        min_characters=200,
    )
    assert report.total_sources == 17
    assert report.total_chunks == 1122
    assert report.total_source_characters == 1_367_316
    assert report.total_chunk_characters == 1_724_524
    assert report.chunks_sha256 == (
        "6dcb7c03719437406fca66a215c004583bc223d3d8c5bf5a58e51ad12bc41852"
    )
    assert len({(source.source_id, source.document_version) for source in report.sources}) == 17
    assert min(source.min_chunk_characters for source in report.sources) >= 200
    assert max(source.max_chunk_characters for source in report.sources) <= 1800


def test_chunk_cli_writes_canonical_dataset_and_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    text = " ".join(f"Official policy sentence {index}." for index in range(50))
    extraction_report, _, _ = write_chunking_inputs(tmp_path, text)
    chunks_path = tmp_path / "data/processed/chunks.jsonl"
    report_path = tmp_path / "data/manifests/chunking.json"

    result = main(
        [
            "chunk",
            str(extraction_report),
            "--output-chunks",
            str(chunks_path),
            "--output-report",
            str(report_path),
            "--repository-root",
            str(tmp_path),
            "--max-characters",
            "300",
            "--overlap-characters",
            "50",
            "--min-characters",
            "100",
        ]
    )

    assert result == 0
    report = ChunkingReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    assert report.total_chunks > 1
    assert report.chunks_sha256 == sha256(chunks_path.read_bytes())
    assert "Chunked 1 source(s)" in capsys.readouterr().out


def test_chunk_cli_refuses_same_output_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    extraction_report, _, _ = write_chunking_inputs(tmp_path, "Official policy text.")
    output = tmp_path / "data/processed/output.json"

    result = main(
        [
            "chunk",
            str(extraction_report),
            "--output-chunks",
            str(output),
            "--output-report",
            str(output),
            "--repository-root",
            str(tmp_path),
        ]
    )

    assert result == 1
    assert "outputs must differ" in capsys.readouterr().err
