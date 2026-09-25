import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from deepshield_api.corpus.cli import main
from deepshield_api.corpus.extraction import (
    ExtractionError,
    ExtractionPolicy,
    ExtractionReport,
    extract_html,
    extract_manifest,
    extract_pdf,
    extract_source,
    normalize_text,
    render_extraction_report,
)
from deepshield_api.corpus.manifest import write_source_manifest
from deepshield_api.corpus.models import SourceManifest, SourceRecord

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
F1_COLLECTED_MANIFEST = (
    REPOSITORY_ROOT / "data" / "manifests" / "f1-policy-collected-manifest.v18.json"
)
F1_EXTRACTION_REPORT = (
    REPOSITORY_ROOT / "data" / "manifests" / "f1-policy-extraction-report.v1.json"
)


def make_pdf(text: str) -> bytes:
    """Build a small standards-compliant synthetic PDF without external fixtures."""

    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    content = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, item in enumerate(objects, start=1):
        offsets.append(len(content))
        content.extend(f"{number} 0 obj\n".encode())
        content.extend(item)
        content.extend(b"\nendobj\n")
    xref_offset = len(content)
    content.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    content.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode())
    content.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode()
    )
    return bytes(content)


def source_data(**overrides: Any) -> dict[str, Any]:
    content = overrides.pop(
        "content",
        b"<html><main>Official synthetic F-1 policy content for deterministic extraction.</main></html>",
    )
    media_type = overrides.get("media_type", "text/html")
    extension = "pdf" if media_type == "application/pdf" else "html"
    payload: dict[str, Any] = {
        "source_id": "synthetic-uscis-guidance",
        "document_version": 1,
        "title": "Synthetic USCIS Guidance",
        "publisher": "uscis",
        "canonical_url": "https://www.uscis.gov/example",
        "document_type": "guidance-page",
        "media_type": media_type,
        "scope_tags": ["status"],
        "rights_notes": "Synthetic test fixture.",
        "collection_status": "collected",
        "retrieved_at": "2026-09-25T12:00:00Z",
        "raw_path": f"data/raw/uscis/synthetic-uscis-guidance/v1.{extension}",
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    payload.update(overrides)
    payload["_content"] = content
    return payload


def write_source(root: Path, **overrides: Any) -> SourceRecord:
    payload = source_data(**overrides)
    content = payload.pop("_content")
    source = SourceRecord.model_validate(payload)
    assert source.raw_path is not None
    raw_path = root / source.raw_path
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(content)
    return source


def test_normalize_text_is_deterministic() -> None:
    assert normalize_text("  F‐1\u00a0 policy\r\n\r\n\r\n  text\u00ad  ") == "F‐1 policy\n\ntext"


def test_extract_html_removes_executable_and_navigation_content() -> None:
    content = b"""
    <html><body>
      <nav>Navigation text</nav><script>secret()</script>
      <main><h1>F-1 Policy</h1><p>Visible guidance.</p><p hidden>Hidden text</p></main>
    </body></html>
    """

    text = normalize_text(extract_html(content))

    assert text == "F-1 Policy\nVisible guidance."
    assert "Navigation" not in text
    assert "secret" not in text
    assert "Hidden" not in text


def test_extract_html_supports_declared_iso_8859_1_as_windows_1252() -> None:
    content = (
        b'<html><head><meta charset="iso-8859-1"></head>'
        b"<main>Official \x93F-1\x94 guidance.</main></html>"
    )

    assert normalize_text(extract_html(content)) == "Official “F-1” guidance."


def test_extract_html_uses_required_primary_selector() -> None:
    content = (
        b"<html><body>Noise<div class='policy'><p>(<span>a</span>) Official policy.</p>"
        b"</div></body></html>"
    )

    assert extract_html(content, primary_selector=".policy") == "(a) Official policy."
    with pytest.raises(ExtractionError, match="did not match"):
        extract_html(content, primary_selector=".missing")


def test_extract_html_rejects_unapproved_legacy_encoding() -> None:
    content = b'<html><head><meta charset="shift_jis"></head><main>\x80</main></html>'

    with pytest.raises(ExtractionError, match="not UTF-8 or approved"):
        extract_html(content)


def test_extract_pdf_reads_synthetic_page() -> None:
    assert "Synthetic F-1 policy" in normalize_text(extract_pdf(make_pdf("Synthetic F-1 policy")))


def test_extract_source_verifies_and_writes_immutable_text(tmp_path: Path) -> None:
    source = write_source(tmp_path)

    record = extract_source(source, tmp_path)
    repeated = extract_source(source, tmp_path)

    extracted_path = tmp_path / record.extracted_path
    assert record == repeated
    assert extracted_path.read_text(encoding="utf-8").endswith("\n")
    assert record.extracted_sha256 == hashlib.sha256(extracted_path.read_bytes()).hexdigest()
    assert record.character_count >= 50


def test_extract_source_rejects_checksum_mismatch(tmp_path: Path) -> None:
    source = write_source(tmp_path)
    assert source.raw_path is not None
    (tmp_path / source.raw_path).write_text("tampered", encoding="utf-8")

    with pytest.raises(ExtractionError, match="checksum mismatch"):
        extract_source(source, tmp_path)


def test_extract_source_rejects_different_existing_interim_file(tmp_path: Path) -> None:
    source = write_source(tmp_path)
    interim_path = tmp_path / "data/interim/uscis/synthetic-uscis-guidance/v1/extractor-1.1.txt"
    interim_path.parent.mkdir(parents=True)
    interim_path.write_text("different", encoding="utf-8")

    with pytest.raises(ExtractionError, match="refusing to replace"):
        extract_source(source, tmp_path)


def test_extract_source_rejects_planned_source(tmp_path: Path) -> None:
    payload = source_data()
    payload.pop("_content")
    payload.update(
        collection_status="planned",
        retrieved_at=None,
        raw_path=None,
        sha256=None,
    )
    source = SourceRecord.model_validate(payload)

    with pytest.raises(ExtractionError, match="only collected"):
        extract_source(source, tmp_path)


def test_extraction_report_rejects_duplicate_source_versions(tmp_path: Path) -> None:
    source = write_source(tmp_path)
    record = extract_source(source, tmp_path)

    with pytest.raises(ValidationError, match="duplicate source version"):
        ExtractionReport(
            report_version=1,
            source_manifest_path="data/manifests/source.json",
            source_manifest_sha256="a" * 64,
            records=[record, record],
        )


def test_extract_manifest_rejects_uncollected_sources_before_writing(tmp_path: Path) -> None:
    collected = write_source(tmp_path)
    planned_payload = source_data(source_id="second-source")
    planned_payload.pop("_content")
    planned_payload.update(
        collection_status="planned",
        retrieved_at=None,
        raw_path=None,
        sha256=None,
    )
    planned = SourceRecord.model_validate(planned_payload)
    manifest = SourceManifest(
        manifest_version=1,
        created_at=datetime.fromisoformat("2026-09-25T12:00:00+00:00"),
        sources=[collected, planned],
    )
    manifest_path = tmp_path / "data/manifests/collected.json"
    write_source_manifest(manifest_path, manifest)

    with pytest.raises(ExtractionError, match="second-source"):
        extract_manifest(manifest, manifest_path, tmp_path)

    assert not (tmp_path / "data/interim").exists()


def test_committed_f1_extraction_report_is_complete_and_traceable() -> None:
    report = ExtractionReport.model_validate_json(F1_EXTRACTION_REPORT.read_text(encoding="utf-8"))
    source_manifest = SourceManifest.model_validate_json(
        F1_COLLECTED_MANIFEST.read_text(encoding="utf-8")
    )

    assert report.report_version == 1
    assert report.extractor_version == "1.1"
    assert (
        report.source_manifest_sha256
        == hashlib.sha256(F1_COLLECTED_MANIFEST.read_bytes()).hexdigest()
    )
    assert len(report.records) == 17
    assert {(record.source_id, record.document_version) for record in report.records} == {
        (source.source_id, source.document_version) for source in source_manifest.sources
    }
    assert all(record.character_count >= 50 for record in report.records)
    assert sum(record.character_count for record in report.records) == 1_367_316
    assert (
        sum(record.extraction_method == "beautifulsoup4-html-v1" for record in report.records) == 15
    )
    assert sum(record.extraction_method == "pypdf-v1" for record in report.records) == 2
    assert sum(record.extraction_profile == "ecfr-section-v1" for record in report.records) == 2
    assert sum(record.extraction_profile == "state-main-copy-v1" for record in report.records) == 1


def test_extract_cli_writes_canonical_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = write_source(tmp_path)
    manifest = SourceManifest(
        manifest_version=1,
        created_at=datetime.fromisoformat("2026-09-25T12:00:00+00:00"),
        sources=[source],
    )
    manifest_path = tmp_path / "data/manifests/collected.json"
    write_source_manifest(manifest_path, manifest)
    report_path = tmp_path / "data/manifests/extraction.json"

    result = main(
        [
            "extract",
            str(manifest_path),
            "--output",
            str(report_path),
            "--repository-root",
            str(tmp_path),
        ]
    )

    assert result == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    report = ExtractionReport.model_validate(payload)
    assert report_path.read_text(encoding="utf-8") == render_extraction_report(report)
    assert report.source_manifest_sha256 == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert "Extracted 1 source(s)" in capsys.readouterr().out


def test_extract_cli_reports_too_little_text(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = write_source(tmp_path)
    manifest = SourceManifest(
        manifest_version=1,
        created_at=datetime.fromisoformat("2026-09-25T12:00:00+00:00"),
        sources=[source],
    )
    manifest_path = tmp_path / "data/manifests/collected.json"
    write_source_manifest(manifest_path, manifest)

    result = main(
        [
            "extract",
            str(manifest_path),
            "--output",
            str(tmp_path / "report.json"),
            "--repository-root",
            str(tmp_path),
            "--min-characters",
            "10000",
        ]
    )

    assert result == 1
    assert "minimum is 10000" in capsys.readouterr().err


def test_extraction_policy_rejects_invalid_minimum() -> None:
    with pytest.raises(ValidationError):
        ExtractionPolicy(min_characters=0)
