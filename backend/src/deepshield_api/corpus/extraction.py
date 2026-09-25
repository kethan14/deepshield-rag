"""Deterministically extract normalized text from verified corpus artifacts."""

import hashlib
import json
import re
import unicodedata
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from bs4 import BeautifulSoup
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pypdf import PdfReader

from deepshield_api.corpus.models import (
    CollectionStatus,
    Sha256Digest,
    Slug,
    SourceManifest,
    SourceMediaType,
    SourcePublisher,
    SourceRecord,
)

EXTRACTOR_VERSION: Literal["1.1"] = "1.1"
DEFAULT_MIN_CHARACTERS = 50
HTML_REMOVAL_TAGS = (
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "canvas",
    "nav",
    "header",
    "footer",
    "form",
)
HTML_TEXT_BLOCK_TAGS = (
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "li",
    "dt",
    "dd",
    "caption",
    "th",
    "td",
    "blockquote",
    "pre",
)
HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n]+")
BLANK_LINES = re.compile(r"\n{3,}")
HTML_CHARSET = re.compile(r"charset\s*=\s*[\"']?([a-z0-9._-]+)", re.IGNORECASE)
INLINE_PARENTHETICAL = re.compile(r"\(\s+([^()]+?)\s+\)")
SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.;:!?])")


class ExtractionError(RuntimeError):
    """Raised when a raw artifact cannot be safely or usefully extracted."""


class ExtractionPolicy(BaseModel):
    """Limits applied to deterministic text extraction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_characters: int = Field(default=DEFAULT_MIN_CHARACTERS, ge=1)


class ExtractionRecord(BaseModel):
    """Traceability metadata for one extracted source artifact."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: Literal["1.0"] = "1.0"
    source_id: Slug
    document_version: int = Field(ge=1)
    media_type: SourceMediaType
    raw_path: str
    raw_sha256: Sha256Digest
    extracted_path: str
    extracted_sha256: Sha256Digest
    extraction_method: Literal["beautifulsoup4-html-v1", "pypdf-v1", "utf8-text-v1"]
    extraction_profile: Literal[
        "html-main-v1",
        "ecfr-section-v1",
        "state-main-copy-v1",
        "pdf-pages-v1",
        "plain-text-v1",
    ]
    source_encoding: Literal["utf-8", "windows-1252"] | None
    character_count: int = Field(ge=1)
    line_count: int = Field(ge=1)

    @model_validator(mode="after")
    def require_safe_paths(self) -> Self:
        """Keep portable input and output paths inside their data stages."""

        _require_data_path(self.raw_path, "raw")
        _require_data_path(self.extracted_path, "interim")
        return self


class ExtractionReport(BaseModel):
    """Versioned, deterministic record of a complete extraction run."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: Literal["1.0"] = "1.0"
    corpus_id: Literal["deepshield-rag-f1-policy"] = "deepshield-rag-f1-policy"
    report_version: int = Field(ge=1)
    extractor_version: Literal["1.1"] = EXTRACTOR_VERSION
    source_manifest_path: str
    source_manifest_sha256: Sha256Digest
    records: list[ExtractionRecord] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_sources(self) -> Self:
        """Prevent ambiguous duplicate source versions in one report."""

        _require_data_path(self.source_manifest_path, "manifests")
        source_versions = [(record.source_id, record.document_version) for record in self.records]
        if len(source_versions) != len(set(source_versions)):
            raise ValueError("extraction report contains a duplicate source version")
        return self


def extract_manifest(
    manifest: SourceManifest,
    manifest_path: Path,
    repository_root: Path,
    *,
    report_version: int = 1,
    policy: ExtractionPolicy | None = None,
) -> ExtractionReport:
    """Extract every collected source and return a traceable report."""

    root = repository_root.resolve()
    manifest_file = _resolve_regular_file(manifest_path, root, "source manifest")
    effective_policy = policy or ExtractionPolicy()
    uncollected = [
        source.source_id
        for source in manifest.sources
        if source.collection_status is not CollectionStatus.COLLECTED
    ]
    if uncollected:
        raise ExtractionError(
            "manifest contains uncollected sources: " + ", ".join(sorted(uncollected))
        )

    prepared = [
        _prepare_source(source, root, policy=effective_policy) for source in manifest.sources
    ]
    for _, extracted_path, content in prepared:
        _write_immutable(extracted_path, content)
    records = [record for record, _, _ in prepared]
    return ExtractionReport(
        report_version=report_version,
        source_manifest_path=_repository_relative_path(manifest_file, root),
        source_manifest_sha256=_sha256(manifest_file.read_bytes()),
        records=records,
    )


def extract_source(
    source: SourceRecord,
    repository_root: Path,
    *,
    policy: ExtractionPolicy | None = None,
) -> ExtractionRecord:
    """Verify, extract, normalize, and persist one immutable source artifact."""

    root = repository_root.resolve()
    record, extracted_path, content = _prepare_source(source, root, policy=policy)
    _write_immutable(extracted_path, content)
    return record


def _prepare_source(
    source: SourceRecord,
    repository_root: Path,
    *,
    policy: ExtractionPolicy | None = None,
) -> tuple[ExtractionRecord, Path, bytes]:
    """Prepare one verified extraction without mutating the interim directory."""

    if source.collection_status is not CollectionStatus.COLLECTED:
        raise ExtractionError("only collected sources can be extracted")
    if source.raw_path is None or source.sha256 is None:
        raise ExtractionError("collected source is missing raw-artifact provenance")

    root = repository_root.resolve()
    raw_path = _resolve_regular_file(root / Path(source.raw_path), root / "data" / "raw", "raw")
    raw_bytes = _read_bytes(raw_path, "raw artifact")
    actual_raw_sha256 = _sha256(raw_bytes)
    if actual_raw_sha256 != source.sha256:
        raise ExtractionError(f"raw checksum mismatch for {source.source_id}")

    method: Literal["beautifulsoup4-html-v1", "pypdf-v1", "utf8-text-v1"]
    profile: Literal[
        "html-main-v1",
        "ecfr-section-v1",
        "state-main-copy-v1",
        "pdf-pages-v1",
        "plain-text-v1",
    ]
    source_encoding: Literal["utf-8", "windows-1252"] | None
    if source.media_type is SourceMediaType.HTML:
        decoded, source_encoding = _decode_html(raw_bytes)
        selector: str | None = None
        profile = "html-main-v1"
        if source.publisher is SourcePublisher.ECFR:
            selector = "div.section[id]"
            profile = "ecfr-section-v1"
        elif source.source_id == "state-student-visa":
            selector = ".tsg-rwd-main-copy-frame"
            profile = "state-main-copy-v1"
        text = _extract_html_text(decoded, primary_selector=selector)
        method = "beautifulsoup4-html-v1"
    elif source.media_type is SourceMediaType.PDF:
        text = extract_pdf(raw_bytes)
        method = "pypdf-v1"
        profile = "pdf-pages-v1"
        source_encoding = None
    elif source.media_type is SourceMediaType.PLAIN_TEXT:
        text = _decode_utf8(raw_bytes, "plain-text source")
        method = "utf8-text-v1"
        profile = "plain-text-v1"
        source_encoding = "utf-8"
    else:  # pragma: no cover - enum prevents unsupported values
        raise ExtractionError(f"unsupported media type: {source.media_type}")

    normalized = normalize_text(text)
    effective_policy = policy or ExtractionPolicy()
    if len(normalized) < effective_policy.min_characters:
        raise ExtractionError(
            f"extracted text for {source.source_id} has {len(normalized)} characters; "
            f"minimum is {effective_policy.min_characters}"
        )

    content = (normalized + "\n").encode("utf-8")
    extracted_relative = build_interim_path(source)
    extracted_path = root / Path(extracted_relative.as_posix())
    _validate_immutable_target(extracted_path, content)

    return (
        ExtractionRecord(
            source_id=source.source_id,
            document_version=source.document_version,
            media_type=source.media_type,
            raw_path=source.raw_path,
            raw_sha256=source.sha256,
            extracted_path=extracted_relative.as_posix(),
            extracted_sha256=_sha256(content),
            extraction_method=method,
            extraction_profile=profile,
            source_encoding=source_encoding,
            character_count=len(normalized),
            line_count=normalized.count("\n") + 1,
        ),
        extracted_path,
        content,
    )


def extract_html(content: bytes, *, primary_selector: str | None = None) -> str:
    """Extract visible primary HTML text while removing executable UI content."""

    decoded, _ = _decode_html(content)
    return _extract_html_text(decoded, primary_selector=primary_selector)


def _extract_html_text(decoded: str, *, primary_selector: str | None = None) -> str:
    """Extract visible primary text from a decoded HTML document."""

    soup = BeautifulSoup(decoded, "html.parser")
    for element in soup.find_all(HTML_REMOVAL_TAGS):
        element.decompose()
    for element in soup.select('[aria-hidden="true"], [hidden]'):
        element.decompose()

    container = soup.select_one(primary_selector) if primary_selector else None
    if primary_selector and container is None:
        raise ExtractionError(f"HTML primary selector did not match: {primary_selector}")
    container = (
        container
        or soup.find("main")
        or soup.find(attrs={"role": "main"})
        or soup.find("article")
        or soup.body
        or soup
    )
    blocks = []
    for element in container.find_all(HTML_TEXT_BLOCK_TAGS):
        if element.find(HTML_TEXT_BLOCK_TAGS):
            continue
        text = element.get_text(separator=" ", strip=True)
        if text:
            text = INLINE_PARENTHETICAL.sub(r"(\1)", text)
            blocks.append(SPACE_BEFORE_PUNCTUATION.sub(r"\1", text))
    return "\n".join(blocks) if blocks else container.get_text(separator=" ", strip=True)


def extract_pdf(content: bytes) -> str:
    """Extract text from every page of an unencrypted PDF."""

    try:
        reader = PdfReader(BytesIO(content), strict=True)
        if reader.is_encrypted:
            raise ExtractionError("encrypted PDFs are not supported")
        pages = [page.extract_text() or "" for page in reader.pages]
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError("unable to extract PDF text") from exc
    return "\n\n".join(pages)


def normalize_text(text: str) -> str:
    """Apply deterministic Unicode and whitespace normalization."""

    normalized = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = HORIZONTAL_WHITESPACE.sub(" ", normalized)
    lines = [line.strip() for line in normalized.split("\n")]
    normalized = "\n".join(lines)
    return BLANK_LINES.sub("\n\n", normalized).strip()


def build_interim_path(source: SourceRecord) -> PurePosixPath:
    """Build the deterministic repository-relative path for normalized text."""

    return PurePosixPath(
        "data",
        "interim",
        source.publisher.value,
        source.source_id,
        f"v{source.document_version}",
        f"extractor-{EXTRACTOR_VERSION}.txt",
    )


def render_extraction_report(report: ExtractionReport) -> str:
    """Return canonical deterministic extraction-report JSON."""

    return json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def write_extraction_report(path: Path, report: ExtractionReport) -> None:
    """Write a new extraction report without replacing an existing version."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ExtractionError(f"refusing to replace an existing extraction report: {path}")
    path.write_text(render_extraction_report(report), encoding="utf-8", newline="\n")


def _require_data_path(
    value: str,
    stage: Literal["raw", "interim", "manifests"],
) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) < 3 or path.parts[:2] != ("data", stage):
        raise ValueError(f"path must be a relative POSIX path inside data/{stage}")
    if ".." in path.parts:
        raise ValueError(f"path must not escape data/{stage}")
    return path


def _resolve_regular_file(path: Path, boundary: Path, label: str) -> Path:
    try:
        resolved_boundary = boundary.resolve(strict=True)
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ExtractionError(f"unable to resolve {label} file: {path}") from exc
    if not resolved.is_relative_to(resolved_boundary) or not resolved.is_file():
        raise ExtractionError(f"{label} path must be a regular file inside {resolved_boundary}")
    if path.is_symlink():
        raise ExtractionError(f"{label} path must not be a symbolic link")
    return resolved


def _repository_relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ExtractionError("source manifest must be inside the repository root") from exc


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ExtractionError(f"unable to read {label}: {path}") from exc


def _decode_utf8(content: bytes, label: str) -> str:
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ExtractionError(f"{label} is not valid UTF-8") from exc


def _decode_html(content: bytes) -> tuple[str, Literal["utf-8", "windows-1252"]]:
    """Decode HTML using UTF-8 or an explicitly declared legacy Western encoding."""

    try:
        return content.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError as exc:
        header = content[:4096].decode("ascii", errors="ignore")
        match = HTML_CHARSET.search(header)
        charset = match.group(1).lower() if match else None
        if charset not in {"iso-8859-1", "latin-1", "latin1", "windows-1252", "cp1252"}:
            raise ExtractionError("HTML source is not UTF-8 or approved Windows-1252 text") from exc
        return content.decode("windows-1252"), "windows-1252"


def _write_immutable(path: Path, content: bytes) -> None:
    _validate_immutable_target(path, content)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _validate_immutable_target(path: Path, content: bytes) -> None:
    if path.exists() and (not path.is_file() or path.read_bytes() != content):
        raise ExtractionError(f"refusing to replace a different interim artifact: {path}")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
