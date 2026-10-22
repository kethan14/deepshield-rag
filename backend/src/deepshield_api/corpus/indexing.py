"""Build and search a small, exact vector index with complete corpus provenance."""

import hashlib
import json
from collections.abc import Iterable, Sequence
from importlib.metadata import version as package_version
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Protocol, Self, cast

import numpy as np
from fastembed import TextEmbedding
from huggingface_hub import snapshot_download
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError, model_validator

from deepshield_api.corpus.chunking import ChunkRecord, load_chunking_report
from deepshield_api.corpus.models import EvidencePublisher, Sha256Digest, Slug

INDEXER_VERSION: Literal["1.0"] = "1.0"
DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_MODEL_REPOSITORY = "Qdrant/bge-small-en-v1.5-onnx-Q"
DEFAULT_MODEL_REVISION = "aa8f8b060edb00e03bfdd08813a2949946c8ba55"
DEFAULT_DIMENSION = 384
DEFAULT_BATCH_SIZE = 64
MODEL_FILES = (
    "config.json",
    "model_optimized.onnx",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)
GitCommit = Annotated[str, Field(pattern=r"^[a-f0-9]{40}$")]
FloatMatrix = NDArray[np.float32]


class IndexingError(RuntimeError):
    """Raised when a vector index cannot be built or safely loaded."""


class EmbeddingBackend(Protocol):
    """Small interface that keeps indexing testable without downloading a model."""

    model_name: str
    model_repository: str
    model_revision: str
    dimension: int
    engine: str
    engine_version: str

    def embed_documents(self, texts: Sequence[str], *, batch_size: int) -> FloatMatrix:
        """Embed document passages in their current order."""

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """Embed one retrieval query."""


class FastEmbedBackend:
    """Pinned CPU-only FastEmbed implementation for the production corpus."""

    model_name = DEFAULT_MODEL_NAME
    model_repository = DEFAULT_MODEL_REPOSITORY
    model_revision = DEFAULT_MODEL_REVISION
    dimension = DEFAULT_DIMENSION
    engine = "fastembed"

    def __init__(self, cache_dir: Path, *, threads: int = 1) -> None:
        if threads < 1:
            raise ValueError("threads must be at least 1")
        self.engine_version = package_version("fastembed")
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            model_path = snapshot_download(
                repo_id=self.model_repository,
                revision=self.model_revision,
                allow_patterns=list(MODEL_FILES),
                cache_dir=str(cache_dir),
                max_workers=1,
            )
            self._model = TextEmbedding(
                model_name=self.model_name,
                cache_dir=str(cache_dir),
                threads=threads,
                providers=["CPUExecutionProvider"],
                specific_model_path=model_path,
                local_files_only=True,
            )
        except Exception as exc:
            raise IndexingError(f"unable to load embedding model {self.model_name}") from exc

    def embed_documents(self, texts: Sequence[str], *, batch_size: int) -> FloatMatrix:
        try:
            generated = self._model.embed(texts, batch_size=batch_size, parallel=None)
            return _stack_vectors(generated, self.dimension)
        except IndexingError:
            raise
        except Exception as exc:
            raise IndexingError("document embedding failed") from exc

    def embed_query(self, text: str) -> NDArray[np.float32]:
        try:
            vectors = list(self._model.query_embed(text))
        except Exception as exc:
            raise IndexingError("query embedding failed") from exc
        if len(vectors) != 1:
            raise IndexingError("query embedder did not return exactly one vector")
        vector = np.asarray(vectors[0], dtype=np.float32)
        if vector.shape != (self.dimension,):
            raise IndexingError("query embedding has an unexpected dimension")
        return vector


class VectorIndexReport(BaseModel):
    """Versioned identity and integrity metadata for one dense-vector index."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: Literal["1.0"] = "1.0"
    corpus_id: Literal["deepshield-rag-f1-policy"] = "deepshield-rag-f1-policy"
    report_version: int = Field(ge=1)
    indexer_version: Literal["1.0"] = INDEXER_VERSION
    chunking_report_path: str
    chunking_report_sha256: Sha256Digest
    chunks_path: str
    chunks_sha256: Sha256Digest
    vectors_path: str
    vectors_sha256: Sha256Digest
    chunk_ids_sha256: Sha256Digest
    model_name: str = Field(min_length=1)
    model_repository: str = Field(min_length=1)
    model_revision: GitCommit
    embedding_engine: str = Field(min_length=1)
    embedding_engine_version: str = Field(min_length=1)
    vector_count: int = Field(gt=0)
    dimension: int = Field(gt=0)
    dtype: Literal["float32"] = "float32"
    similarity: Literal["cosine"] = "cosine"
    normalized: Literal[True] = True

    @model_validator(mode="after")
    def require_safe_paths(self) -> Self:
        """Keep every linked artifact inside its intended repository data stage."""

        _require_stage_path(self.chunking_report_path, "manifests")
        _require_stage_path(self.chunks_path, "processed")
        _require_stage_path(self.vectors_path, "indexes")
        return self


class RetrievalHit(BaseModel):
    """One ranked, cited passage returned by exact cosine search."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1)
    score: float = Field(ge=-1.000001, le=1.000001)
    chunk_id: str
    source_id: Slug
    title: str
    publisher: EvidencePublisher
    canonical_url: HttpUrl
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    text: str = Field(min_length=1)


def build_vector_index(
    chunking_report_path: Path,
    vectors_path: Path,
    repository_root: Path,
    embedder: EmbeddingBackend,
    *,
    report_version: int = 1,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[bytes, VectorIndexReport]:
    """Embed a verified chunk dataset and return immutable index bytes and its report."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    root = repository_root.resolve()
    report_file = _resolve_regular_file(
        chunking_report_path,
        root / "data" / "manifests",
        "chunking report",
    )
    chunking_report = load_chunking_report(report_file)
    chunks_file = _resolve_regular_file(
        root / Path(chunking_report.chunks_path),
        root / "data" / "processed",
        "chunk dataset",
    )
    chunk_bytes = chunks_file.read_bytes()
    if _sha256(chunk_bytes) != chunking_report.chunks_sha256:
        raise IndexingError("chunk dataset checksum does not match the chunking report")
    chunks = load_chunk_dataset(chunks_file)
    if len(chunks) != chunking_report.total_chunks:
        raise IndexingError("chunk count does not match the chunking report")

    vectors = embedder.embed_documents(
        [chunk.text for chunk in chunks],
        batch_size=batch_size,
    )
    if vectors.shape != (len(chunks), embedder.dimension):
        raise IndexingError("document embeddings have an unexpected shape")
    normalized = _normalize_rows(vectors)
    vector_bytes = render_vector_index(normalized)
    report = VectorIndexReport(
        report_version=report_version,
        chunking_report_path=_repository_relative_path(report_file, root),
        chunking_report_sha256=_sha256(report_file.read_bytes()),
        chunks_path=chunking_report.chunks_path,
        chunks_sha256=chunking_report.chunks_sha256,
        vectors_path=_repository_relative_path(vectors_path, root),
        vectors_sha256=_sha256(vector_bytes),
        chunk_ids_sha256=_chunk_ids_sha256(chunks),
        model_name=embedder.model_name,
        model_repository=embedder.model_repository,
        model_revision=embedder.model_revision,
        embedding_engine=embedder.engine,
        embedding_engine_version=embedder.engine_version,
        vector_count=len(chunks),
        dimension=embedder.dimension,
    )
    return vector_bytes, report


def load_chunk_dataset(path: Path) -> list[ChunkRecord]:
    """Load and validate canonical JSONL chunk records."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise IndexingError(f"unable to read chunk dataset: {path}") from exc
    if not lines:
        raise IndexingError("chunk dataset must not be empty")
    try:
        chunks = [ChunkRecord.model_validate_json(line) for line in lines]
    except ValidationError as exc:
        raise IndexingError(f"chunk dataset validation failed:\n{exc}") from exc
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise IndexingError("chunk dataset contains duplicate chunk identifiers")
    return chunks


def load_vector_index(
    index_report_path: Path,
    repository_root: Path,
) -> tuple[VectorIndexReport, list[ChunkRecord], FloatMatrix]:
    """Load a vector index only after verifying every linked artifact and checksum."""

    root = repository_root.resolve()
    report_file = _resolve_regular_file(
        index_report_path,
        root / "data" / "manifests",
        "vector index report",
    )
    report = load_vector_index_report(report_file)
    chunking_report_file = _resolve_regular_file(
        root / Path(report.chunking_report_path),
        root / "data" / "manifests",
        "chunking report",
    )
    if _sha256(chunking_report_file.read_bytes()) != report.chunking_report_sha256:
        raise IndexingError("chunking report checksum does not match the index report")
    chunking_report = load_chunking_report(chunking_report_file)
    if chunking_report.chunks_sha256 != report.chunks_sha256:
        raise IndexingError("chunk dataset provenance differs between reports")
    chunks_file = _resolve_regular_file(
        root / Path(report.chunks_path),
        root / "data" / "processed",
        "chunk dataset",
    )
    if _sha256(chunks_file.read_bytes()) != report.chunks_sha256:
        raise IndexingError("chunk dataset checksum does not match the index report")
    chunks = load_chunk_dataset(chunks_file)
    if len(chunks) != report.vector_count or _chunk_ids_sha256(chunks) != report.chunk_ids_sha256:
        raise IndexingError("chunk order or count does not match the vector index")
    vectors_file = _resolve_regular_file(
        root / Path(report.vectors_path),
        root / "data" / "indexes",
        "vector index",
    )
    vector_bytes = vectors_file.read_bytes()
    if _sha256(vector_bytes) != report.vectors_sha256:
        raise IndexingError("vector index checksum does not match the index report")
    vectors = parse_vector_index(vector_bytes)
    if vectors.shape != (report.vector_count, report.dimension):
        raise IndexingError("vector index shape does not match the index report")
    _require_normalized(vectors)
    return report, chunks, vectors


def search_vector_index(
    query: str,
    index_report_path: Path,
    repository_root: Path,
    embedder: EmbeddingBackend,
    *,
    top_k: int = 5,
    source_ids: set[str] | None = None,
) -> list[RetrievalHit]:
    """Return stable exact-cosine rankings from a verified local vector index."""

    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("query must not be empty")
    if not 1 <= top_k <= 100:
        raise ValueError("top_k must be between 1 and 100")
    report, chunks, vectors = load_vector_index(index_report_path, repository_root)
    _require_matching_embedder(report, embedder)
    query_vector = _normalize_vector(embedder.embed_query(cleaned_query), report.dimension)
    scores = vectors @ query_vector
    candidates = [
        index
        for index, chunk in enumerate(chunks)
        if source_ids is None or chunk.source_id in source_ids
    ]
    ranked = sorted(candidates, key=lambda index: (-float(scores[index]), chunks[index].chunk_id))
    hits = []
    for rank, index in enumerate(ranked[:top_k], start=1):
        chunk = chunks[index]
        hits.append(
            RetrievalHit(
                rank=rank,
                score=float(scores[index]),
                chunk_id=chunk.chunk_id,
                source_id=chunk.source_id,
                title=chunk.title,
                publisher=chunk.publisher,
                canonical_url=chunk.canonical_url,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                text=chunk.text,
            )
        )
    return hits


def render_vector_index(vectors: FloatMatrix) -> bytes:
    """Serialize one contiguous float32 matrix without pickle data."""

    if vectors.ndim != 2 or not vectors.shape[0] or not vectors.shape[1]:
        raise IndexingError("vector index must be a non-empty two-dimensional matrix")
    if vectors.dtype != np.float32:
        raise IndexingError("vector index must use float32")
    buffer = BytesIO()
    np.save(buffer, np.ascontiguousarray(vectors), allow_pickle=False)
    return buffer.getvalue()


def parse_vector_index(content: bytes) -> FloatMatrix:
    """Parse a safe NumPy matrix and reject unsupported shapes or data types."""

    try:
        loaded = np.load(BytesIO(content), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise IndexingError("unable to parse vector index") from exc
    if not isinstance(loaded, np.ndarray) or loaded.ndim != 2 or loaded.dtype != np.float32:
        raise IndexingError("vector index must be a two-dimensional float32 array")
    if not loaded.shape[0] or not loaded.shape[1] or not np.isfinite(loaded).all():
        raise IndexingError("vector index must contain finite values and non-empty dimensions")
    return cast(FloatMatrix, np.ascontiguousarray(loaded))


def load_vector_index_report(path: Path) -> VectorIndexReport:
    """Load and validate a UTF-8 vector-index report."""

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise IndexingError(f"unable to read vector index report: {path}") from exc
    try:
        return VectorIndexReport.model_validate_json(content)
    except ValidationError as exc:
        raise IndexingError(f"vector index report validation failed:\n{exc}") from exc


def render_vector_index_report(report: VectorIndexReport) -> str:
    """Render canonical deterministic vector-index report JSON."""

    return json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def render_retrieval_hits(hits: list[RetrievalHit]) -> str:
    """Render deterministic human-inspectable retrieval results."""

    return json.dumps([hit.model_dump(mode="json") for hit in hits], indent=2) + "\n"


def write_vector_index(path: Path, content: bytes) -> None:
    """Write an immutable vector-index artifact."""

    _write_immutable(path, content, "vector index")


def write_vector_index_report(path: Path, report: VectorIndexReport) -> None:
    """Write a new index report without replacing an existing version."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise IndexingError(f"refusing to replace an existing vector index report: {path}")
    path.write_text(render_vector_index_report(report), encoding="utf-8", newline="\n")


def _stack_vectors(vectors: Iterable[NDArray[np.generic]], dimension: int) -> FloatMatrix:
    items = [np.asarray(vector, dtype=np.float32) for vector in vectors]
    if not items:
        raise IndexingError("embedding backend returned no vectors")
    if any(item.shape != (dimension,) for item in items):
        raise IndexingError("embedding backend returned an unexpected dimension")
    return cast(FloatMatrix, np.ascontiguousarray(np.vstack(items), dtype=np.float32))


def _normalize_rows(vectors: FloatMatrix) -> FloatMatrix:
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise IndexingError("embeddings must be a finite two-dimensional matrix")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 0):
        raise IndexingError("embeddings must not contain zero-length vectors")
    return cast(
        FloatMatrix,
        np.ascontiguousarray(vectors / norms[:, np.newaxis], dtype=np.float32),
    )


def _normalize_vector(vector: NDArray[np.float32], dimension: int) -> NDArray[np.float32]:
    value = np.asarray(vector, dtype=np.float32)
    if value.shape != (dimension,) or not np.isfinite(value).all():
        raise IndexingError("query embedding has an invalid shape or value")
    norm = float(np.linalg.norm(value))
    if norm <= 0:
        raise IndexingError("query embedding must not be a zero-length vector")
    return cast(NDArray[np.float32], np.ascontiguousarray(value / norm, dtype=np.float32))


def _require_normalized(vectors: FloatMatrix) -> None:
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
        raise IndexingError("vector index rows are not L2-normalized")


def _require_matching_embedder(report: VectorIndexReport, embedder: EmbeddingBackend) -> None:
    identity = (
        embedder.model_name,
        embedder.model_repository,
        embedder.model_revision,
        embedder.engine,
        embedder.engine_version,
        embedder.dimension,
    )
    expected = (
        report.model_name,
        report.model_repository,
        report.model_revision,
        report.embedding_engine,
        report.embedding_engine_version,
        report.dimension,
    )
    if identity != expected:
        raise IndexingError("query embedder identity does not match the vector index")


def _chunk_ids_sha256(chunks: Sequence[ChunkRecord]) -> str:
    content = "".join(f"{chunk.chunk_id}\n" for chunk in chunks).encode("utf-8")
    return _sha256(content)


def _require_stage_path(
    value: str,
    stage: Literal["manifests", "processed", "indexes"],
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
        raise IndexingError(f"unable to resolve {label}: {path}") from exc
    if not resolved.is_relative_to(resolved_boundary) or not resolved.is_file():
        raise IndexingError(f"{label} must be a regular file inside {resolved_boundary}")
    if path.is_symlink():
        raise IndexingError(f"{label} must not be a symbolic link")
    return resolved


def _repository_relative_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise IndexingError("output path must be inside the repository root") from exc


def _write_immutable(path: Path, content: bytes, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (not path.is_file() or path.read_bytes() != content):
        raise IndexingError(f"refusing to replace a different {label}: {path}")
    if not path.exists():
        path.write_bytes(content)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
