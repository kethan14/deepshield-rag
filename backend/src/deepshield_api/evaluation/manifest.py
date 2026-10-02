"""Read, validate, and export evaluation benchmark artifacts."""

import hashlib
import json
import re
from json import JSONDecodeError
from pathlib import Path

from pydantic import ValidationError

from deepshield_api.corpus.chunking import ChunkingError, load_chunking_report
from deepshield_api.corpus.manifest import ManifestError, load_source_manifest
from deepshield_api.evaluation.models import BenchmarkManifest


class BenchmarkManifestError(ValueError):
    """Raised when a benchmark artifact or its corpus reference is invalid."""


def load_benchmark_manifest(path: Path) -> BenchmarkManifest:
    """Load and validate a UTF-8 JSON benchmark manifest."""

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BenchmarkManifestError(f"unable to read benchmark manifest: {path}") from exc
    try:
        payload = json.loads(content)
    except JSONDecodeError as exc:
        raise BenchmarkManifestError(
            f"benchmark manifest is not valid JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc
    try:
        return BenchmarkManifest.model_validate(payload)
    except ValidationError as exc:
        raise BenchmarkManifestError(f"benchmark manifest validation failed:\n{exc}") from exc


def validate_corpus_references(manifest: BenchmarkManifest, repository_root: Path) -> None:
    """Verify frozen corpus, chunking identity, and every reference label."""

    root = repository_root.resolve()
    corpus_path = (root / manifest.corpus_manifest_path).resolve()
    try:
        corpus_path.relative_to(root / "data" / "manifests")
    except ValueError as exc:  # pragma: no cover - model validation is the primary boundary
        raise BenchmarkManifestError("corpus manifest resolves outside data/manifests") from exc
    try:
        corpus_bytes = corpus_path.read_bytes()
    except OSError as exc:
        raise BenchmarkManifestError(
            f"unable to read referenced corpus manifest: {corpus_path}"
        ) from exc
    actual_digest = hashlib.sha256(corpus_bytes).hexdigest()
    if actual_digest != manifest.corpus_manifest_sha256:
        raise BenchmarkManifestError("referenced corpus manifest checksum does not match")
    try:
        corpus = load_source_manifest(corpus_path)
    except ManifestError as exc:
        raise BenchmarkManifestError(str(exc)) from exc

    available_ids = {source.source_id for source in corpus.sources}
    referenced_ids = {
        source_id for case in manifest.cases for source_id in case.reference_source_ids
    }
    missing_ids = sorted(referenced_ids - available_ids)
    if missing_ids:
        raise BenchmarkManifestError(
            "benchmark references unknown corpus source IDs: " + ", ".join(missing_ids)
        )

    chunking_path = (root / manifest.chunking_report_path).resolve()
    try:
        chunking_path.relative_to(root / "data" / "manifests")
    except ValueError as exc:  # pragma: no cover - model validation is the primary boundary
        raise BenchmarkManifestError("chunking report resolves outside data/manifests") from exc
    try:
        chunking_bytes = chunking_path.read_bytes()
    except OSError as exc:
        raise BenchmarkManifestError(
            f"unable to read referenced chunking report: {chunking_path}"
        ) from exc
    if hashlib.sha256(chunking_bytes).hexdigest() != manifest.chunking_report_sha256:
        raise BenchmarkManifestError("referenced chunking report checksum does not match")
    try:
        chunking_report = load_chunking_report(chunking_path)
    except ChunkingError as exc:
        raise BenchmarkManifestError(str(exc)) from exc

    chunk_counts = {
        (summary.source_id, summary.document_version): summary.chunk_count
        for summary in chunking_report.sources
    }
    for case in manifest.cases:
        for chunk_id in case.reference_chunk_ids:
            match = re.fullmatch(
                r"(?P<source>[a-z0-9]+(?:-[a-z0-9]+)*)-v(?P<version>[1-9][0-9]*)-c(?P<index>[0-9]{5})",
                chunk_id,
            )
            if match is None:  # pragma: no cover - ChunkId validates the outer format
                raise BenchmarkManifestError(f"invalid reference chunk ID: {chunk_id}")
            source_id = match.group("source")
            document_version = int(match.group("version"))
            chunk_index = int(match.group("index"))
            chunk_count = chunk_counts.get((source_id, document_version))
            if source_id not in case.reference_source_ids:
                raise BenchmarkManifestError(
                    f"chunk {chunk_id} is not covered by its case reference_source_ids"
                )
            if chunk_count is None or chunk_index >= chunk_count:
                raise BenchmarkManifestError(f"benchmark references unknown chunk ID: {chunk_id}")


def render_benchmark_manifest_schema() -> str:
    """Return the canonical deterministic benchmark JSON Schema."""

    return json.dumps(BenchmarkManifest.model_json_schema(), indent=2, sort_keys=True) + "\n"


def render_benchmark_manifest(manifest: BenchmarkManifest) -> str:
    """Return a canonical deterministic benchmark manifest."""

    return json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def write_benchmark_manifest_schema(path: Path) -> None:
    """Write the benchmark schema with normalized line endings."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_benchmark_manifest_schema(), encoding="utf-8", newline="\n")


def write_benchmark_manifest(path: Path, manifest: BenchmarkManifest) -> None:
    """Write a new benchmark artifact without replacing an existing version."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise BenchmarkManifestError(f"refusing to replace an existing benchmark: {path}")
    path.write_text(render_benchmark_manifest(manifest), encoding="utf-8", newline="\n")
