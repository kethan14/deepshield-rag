"""Deterministically chunk normalized corpus text with complete provenance."""

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError, model_validator

from deepshield_api.corpus.extraction import ExtractionRecord, ExtractionReport
from deepshield_api.corpus.manifest import load_source_manifest
from deepshield_api.corpus.models import (
    Sha256Digest,
    Slug,
    SourceDocumentType,
    SourcePublisher,
    SourceRecord,
)

CHUNKER_VERSION: Literal["1.0"] = "1.0"
DEFAULT_MAX_CHARACTERS = 1800
DEFAULT_OVERLAP_CHARACTERS = 250
DEFAULT_MIN_CHUNK_CHARACTERS = 200
SENTENCE_BOUNDARIES = (". ", "? ", "! ")
ChunkId = Annotated[
    str,
    Field(min_length=1, max_length=120, pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$"),
]


class ChunkingError(RuntimeError):
    """Raised when normalized text cannot be safely or consistently chunked."""


class ChunkingPolicy(BaseModel):
    """Model-independent character limits for retrieval chunks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_characters: int = Field(default=DEFAULT_MAX_CHARACTERS, ge=200, le=10_000)
    overlap_characters: int = Field(default=DEFAULT_OVERLAP_CHARACTERS, ge=0)
    min_characters: int = Field(default=DEFAULT_MIN_CHUNK_CHARACTERS, ge=1)

    @model_validator(mode="after")
    def require_consistent_limits(self) -> Self:
        """Prevent overlap or minimum sizes that cannot fit in a chunk."""

        if self.overlap_characters >= self.max_characters:
            raise ValueError("overlap_characters must be smaller than max_characters")
        if self.min_characters > self.max_characters:
            raise ValueError("min_characters must not exceed max_characters")
        return self


class ChunkRecord(BaseModel):
    """One retrieval-ready passage with source and offset provenance."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    schema_version: Literal["1.0"] = "1.0"
    chunk_id: ChunkId
    source_id: Slug
    document_version: int = Field(ge=1)
    chunk_index: int = Field(ge=0)
    title: str = Field(min_length=1, max_length=300)
    publisher: SourcePublisher
    canonical_url: HttpUrl
    document_type: SourceDocumentType
    scope_tags: list[Slug] = Field(min_length=1)
    raw_sha256: Sha256Digest
    extracted_sha256: Sha256Digest
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    character_count: int = Field(gt=0)
    text_sha256: Sha256Digest
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_text_provenance(self) -> Self:
        """Keep offsets, length, and checksum synchronized with the stored text."""

        if self.text != self.text.strip():
            raise ValueError("chunk text must not have leading or trailing whitespace")
        if self.char_end - self.char_start != len(self.text):
            raise ValueError("chunk offsets must match the text length")
        if self.character_count != len(self.text):
            raise ValueError("character_count must match the text length")
        if self.text_sha256 != _sha256(self.text.encode("utf-8")):
            raise ValueError("text_sha256 must match the chunk text")
        if len(self.scope_tags) != len(set(self.scope_tags)):
            raise ValueError("scope_tags must not contain duplicates")
        return self


class ChunkingSourceSummary(BaseModel):
    """Per-source measurements for one chunking run."""

    model_config = ConfigDict(extra="forbid")

    source_id: Slug
    document_version: int = Field(ge=1)
    extracted_sha256: Sha256Digest
    source_character_count: int = Field(gt=0)
    chunk_count: int = Field(gt=0)
    chunk_character_count: int = Field(gt=0)
    min_chunk_characters: int = Field(gt=0)
    max_chunk_characters: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_summary_range(self) -> Self:
        """Keep per-source range statistics internally consistent."""

        if self.min_chunk_characters > self.max_chunk_characters:
            raise ValueError("minimum chunk size cannot exceed maximum chunk size")
        return self


class ChunkingReport(BaseModel):
    """Canonical measurements and checksums for a complete chunk dataset."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: Literal["1.0"] = "1.0"
    corpus_id: Literal["deepshield-rag-f1-policy"] = "deepshield-rag-f1-policy"
    report_version: int = Field(ge=1)
    chunker_version: Literal["1.0"] = CHUNKER_VERSION
    extraction_report_path: str
    extraction_report_sha256: Sha256Digest
    chunks_path: str
    chunks_sha256: Sha256Digest
    policy: ChunkingPolicy
    total_sources: int = Field(gt=0)
    total_chunks: int = Field(gt=0)
    total_source_characters: int = Field(gt=0)
    total_chunk_characters: int = Field(gt=0)
    sources: list[ChunkingSourceSummary] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_report_totals(self) -> Self:
        """Validate paths, unique sources, and aggregate measurements."""

        _require_stage_path(self.extraction_report_path, "manifests")
        _require_stage_path(self.chunks_path, "processed")
        source_versions = [(source.source_id, source.document_version) for source in self.sources]
        if len(source_versions) != len(set(source_versions)):
            raise ValueError("chunking report contains a duplicate source version")
        if self.total_sources != len(self.sources):
            raise ValueError("total_sources must match the source summaries")
        if self.total_chunks != sum(source.chunk_count for source in self.sources):
            raise ValueError("total_chunks must match the source summaries")
        if self.total_source_characters != sum(
            source.source_character_count for source in self.sources
        ):
            raise ValueError("total_source_characters must match the source summaries")
        if self.total_chunk_characters != sum(
            source.chunk_character_count for source in self.sources
        ):
            raise ValueError("total_chunk_characters must match the source summaries")
        return self


def load_extraction_report(path: Path) -> ExtractionReport:
    """Load and validate a UTF-8 extraction report."""

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ChunkingError(f"unable to read extraction report: {path}") from exc
    try:
        return ExtractionReport.model_validate_json(content)
    except ValidationError as exc:
        raise ChunkingError(f"extraction report validation failed:\n{exc}") from exc


def chunk_extraction_report(
    extraction_report_path: Path,
    chunks_path: Path,
    repository_root: Path,
    *,
    report_version: int = 1,
    policy: ChunkingPolicy | None = None,
) -> tuple[list[ChunkRecord], ChunkingReport]:
    """Verify extracted artifacts and build a deterministic chunk dataset and report."""

    root = repository_root.resolve()
    report_file = _resolve_regular_file(
        extraction_report_path,
        root / "data" / "manifests",
        "extraction report",
    )
    extraction_report = load_extraction_report(report_file)
    extraction_report_hash = _sha256(report_file.read_bytes())
    source_manifest_file = _resolve_regular_file(
        root / Path(extraction_report.source_manifest_path),
        root / "data" / "manifests",
        "source manifest",
    )
    if _sha256(source_manifest_file.read_bytes()) != extraction_report.source_manifest_sha256:
        raise ChunkingError("source manifest checksum does not match the extraction report")
    source_manifest = load_source_manifest(source_manifest_file)
    sources = {
        (source.source_id, source.document_version): source for source in source_manifest.sources
    }
    effective_policy = policy or ChunkingPolicy()

    all_chunks: list[ChunkRecord] = []
    summaries: list[ChunkingSourceSummary] = []
    for extraction in extraction_report.records:
        key = (extraction.source_id, extraction.document_version)
        source = sources.get(key)
        if source is None:
            raise ChunkingError(
                f"extraction record has no source-manifest entry: {extraction.source_id}"
            )
        source_chunks, summary = _chunk_extracted_source(
            extraction,
            source,
            root,
            effective_policy,
        )
        all_chunks.extend(source_chunks)
        summaries.append(summary)

    if set(sources) != {
        (record.source_id, record.document_version) for record in extraction_report.records
    }:
        raise ChunkingError(
            "source manifest and extraction report contain different source versions"
        )

    dataset = render_chunk_dataset(all_chunks)
    chunks_relative = _repository_relative_path(chunks_path, root)
    report = ChunkingReport(
        report_version=report_version,
        extraction_report_path=_repository_relative_path(report_file, root),
        extraction_report_sha256=extraction_report_hash,
        chunks_path=chunks_relative,
        chunks_sha256=_sha256(dataset),
        policy=effective_policy,
        total_sources=len(summaries),
        total_chunks=len(all_chunks),
        total_source_characters=sum(summary.source_character_count for summary in summaries),
        total_chunk_characters=sum(summary.chunk_character_count for summary in summaries),
        sources=summaries,
    )
    return all_chunks, report


def chunk_source_text(
    text: str,
    source: SourceRecord,
    extraction: ExtractionRecord,
    *,
    policy: ChunkingPolicy | None = None,
) -> list[ChunkRecord]:
    """Split one normalized document into deterministic overlapping chunks."""

    if not text or text != text.strip():
        raise ChunkingError("source text must be non-empty and stripped")
    effective_policy = policy or ChunkingPolicy()
    chunks: list[ChunkRecord] = []
    start = 0
    while start < len(text):
        maximum_end = min(start + effective_policy.max_characters, len(text))
        end = _choose_chunk_end(text, start, maximum_end, effective_policy.min_characters)
        raw = text[start:end]
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw) - len(raw.rstrip())
        actual_start = start + leading
        actual_end = end - trailing
        chunk_text = text[actual_start:actual_end]
        if not chunk_text:
            raise ChunkingError(f"chunking produced empty text for {source.source_id}")
        chunk_index = len(chunks)
        chunks.append(
            ChunkRecord(
                chunk_id=f"{source.source_id}-v{source.document_version}-c{chunk_index:05d}",
                source_id=source.source_id,
                document_version=source.document_version,
                chunk_index=chunk_index,
                title=source.title,
                publisher=source.publisher,
                canonical_url=source.canonical_url,
                document_type=source.document_type,
                scope_tags=source.scope_tags,
                raw_sha256=extraction.raw_sha256,
                extracted_sha256=extraction.extracted_sha256,
                char_start=actual_start,
                char_end=actual_end,
                character_count=len(chunk_text),
                text_sha256=_sha256(chunk_text.encode("utf-8")),
                text=chunk_text,
            )
        )
        if end >= len(text):
            break
        start = _choose_next_start(text, start, end, effective_policy)

    _validate_chunk_sequence(text, chunks, effective_policy)
    return chunks


def render_chunk_dataset(chunks: list[ChunkRecord]) -> bytes:
    """Render canonical newline-delimited JSON for a chunk dataset."""

    if not chunks:
        raise ChunkingError("chunk dataset must not be empty")
    lines = [
        json.dumps(chunk.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        for chunk in chunks
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def render_chunking_report(report: ChunkingReport) -> str:
    """Render canonical deterministic chunking-report JSON."""

    return json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def write_chunk_dataset(path: Path, chunks: list[ChunkRecord]) -> None:
    """Write an immutable canonical chunk dataset."""

    _write_immutable(path, render_chunk_dataset(chunks), "chunk dataset")


def write_chunking_report(path: Path, report: ChunkingReport) -> None:
    """Write a new chunking report without replacing an existing version."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ChunkingError(f"refusing to replace an existing chunking report: {path}")
    path.write_text(render_chunking_report(report), encoding="utf-8", newline="\n")


def _chunk_extracted_source(
    extraction: ExtractionRecord,
    source: SourceRecord,
    root: Path,
    policy: ChunkingPolicy,
) -> tuple[list[ChunkRecord], ChunkingSourceSummary]:
    if extraction.raw_sha256 != source.sha256:
        raise ChunkingError(f"raw checksum provenance mismatch for {source.source_id}")
    extracted_path = _resolve_regular_file(
        root / Path(extraction.extracted_path),
        root / "data" / "interim",
        "extracted text",
    )
    content = extracted_path.read_bytes()
    if _sha256(content) != extraction.extracted_sha256:
        raise ChunkingError(f"extracted checksum mismatch for {source.source_id}")
    try:
        decoded = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ChunkingError(f"extracted text is not UTF-8 for {source.source_id}") from exc
    if not decoded.endswith("\n") or decoded.endswith("\n\n"):
        raise ChunkingError(f"extracted text must end with exactly one newline: {source.source_id}")
    text = decoded.removesuffix("\n")
    if len(text) != extraction.character_count or text.count("\n") + 1 != extraction.line_count:
        raise ChunkingError(f"extracted text measurements mismatch for {source.source_id}")

    chunks = chunk_source_text(text, source, extraction, policy=policy)
    counts = [chunk.character_count for chunk in chunks]
    summary = ChunkingSourceSummary(
        source_id=source.source_id,
        document_version=source.document_version,
        extracted_sha256=extraction.extracted_sha256,
        source_character_count=len(text),
        chunk_count=len(chunks),
        chunk_character_count=sum(counts),
        min_chunk_characters=min(counts),
        max_chunk_characters=max(counts),
    )
    return chunks, summary


def _choose_chunk_end(text: str, start: int, maximum_end: int, minimum: int) -> int:
    if maximum_end >= len(text):
        return len(text)
    minimum_end = min(start + minimum, maximum_end)
    newline = text.rfind("\n", minimum_end, maximum_end + 1)
    if newline >= minimum_end:
        return newline
    for marker in SENTENCE_BOUNDARIES:
        boundary = text.rfind(marker, minimum_end, maximum_end + 1)
        if boundary >= minimum_end:
            return boundary + 1
    space = text.rfind(" ", minimum_end, maximum_end + 1)
    return space if space >= minimum_end else maximum_end


def _choose_next_start(text: str, previous_start: int, end: int, policy: ChunkingPolicy) -> int:
    minimum_progress = min(end, previous_start + policy.min_characters)
    if len(text) - end < policy.min_characters:
        desired = max(minimum_progress, len(text) - policy.max_characters)
    else:
        desired = max(minimum_progress, end - policy.overlap_characters)
    earliest = max(minimum_progress, desired - policy.overlap_characters)
    newline = text.rfind("\n", earliest, desired)
    if newline != -1:
        return newline + 1
    for marker in SENTENCE_BOUNDARIES:
        boundary = text.rfind(marker, earliest, desired)
        if boundary != -1:
            return boundary + len(marker)
    space = text.rfind(" ", earliest, desired)
    if space != -1:
        return space + 1
    return desired


def _validate_chunk_sequence(
    text: str,
    chunks: list[ChunkRecord],
    policy: ChunkingPolicy,
) -> None:
    if not chunks or chunks[0].char_start != 0 or chunks[-1].char_end != len(text):
        raise ChunkingError("chunks do not cover the complete source text")
    for index, chunk in enumerate(chunks):
        if chunk.chunk_index != index or chunk.text != text[chunk.char_start : chunk.char_end]:
            raise ChunkingError("chunk sequence provenance is inconsistent")
        if chunk.character_count > policy.max_characters:
            raise ChunkingError("chunk exceeds max_characters")
        if index and chunk.char_start <= chunks[index - 1].char_start:
            raise ChunkingError("chunk sequence does not make forward progress")
        if index and chunk.char_start > chunks[index - 1].char_end:
            skipped = text[chunks[index - 1].char_end : chunk.char_start]
            if skipped.strip():
                raise ChunkingError("chunk sequence contains a non-whitespace text gap")


def _require_stage_path(value: str, stage: Literal["manifests", "processed"]) -> PurePosixPath:
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
        raise ChunkingError(f"unable to resolve {label}: {path}") from exc
    if not resolved.is_relative_to(resolved_boundary) or not resolved.is_file():
        raise ChunkingError(f"{label} must be a regular file inside {resolved_boundary}")
    if path.is_symlink():
        raise ChunkingError(f"{label} must not be a symbolic link")
    return resolved


def _repository_relative_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ChunkingError("output path must be inside the repository root") from exc


def _write_immutable(path: Path, content: bytes, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (not path.is_file() or path.read_bytes() != content):
        raise ChunkingError(f"refusing to replace a different {label}: {path}")
    if path.exists():
        return
    path.write_bytes(content)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
