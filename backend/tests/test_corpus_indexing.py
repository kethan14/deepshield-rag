import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from deepshield_api.corpus.chunking import (
    ChunkingPolicy,
    ChunkingReport,
    ChunkingSourceSummary,
    ChunkRecord,
    render_chunk_dataset,
    render_chunking_report,
)
from deepshield_api.corpus.cli import main
from deepshield_api.corpus.indexing import (
    FastEmbedBackend,
    IndexingError,
    VectorIndexReport,
    build_vector_index,
    load_vector_index,
    load_vector_index_report,
    parse_vector_index,
    render_vector_index,
    render_vector_index_report,
    search_vector_index,
    write_vector_index,
    write_vector_index_report,
)


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class FakeEmbedder:
    def __init__(self) -> None:
        self.model_name = "synthetic-embedding-model"
        self.model_repository = "synthetic/model-repository"
        self.model_revision = "a" * 40
        self.dimension = 3
        self.engine = "synthetic-engine"
        self.engine_version = "1.0.0"

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
    ) -> NDArray[np.float32]:
        del batch_size
        basis = (
            np.array([2.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 3.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 4.0], dtype=np.float32),
        )
        return np.vstack([basis[index % len(basis)] for index, _ in enumerate(texts)])

    def embed_query(self, text: str) -> NDArray[np.float32]:
        if "employment" in text.lower():
            return np.array([5.0, 0.0, 0.0], dtype=np.float32)
        if "travel" in text.lower():
            return np.array([0.0, 5.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 5.0], dtype=np.float32)


class DifferentFakeEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.model_revision = "b" * 40


def make_chunk(index: int, text: str, *, source_id: str = "synthetic-policy") -> ChunkRecord:
    return ChunkRecord.model_validate(
        {
            "chunk_id": f"{source_id}-v1-c{index:05d}",
            "source_id": source_id,
            "document_version": 1,
            "chunk_index": index,
            "title": "Synthetic F-1 Policy",
            "publisher": "uscis",
            "canonical_url": "https://www.uscis.gov/example",
            "document_type": "guidance-page",
            "scope_tags": ["employment", "travel"],
            "raw_sha256": "1" * 64,
            "extracted_sha256": "2" * 64,
            "char_start": index * 100,
            "char_end": index * 100 + len(text),
            "character_count": len(text),
            "text_sha256": sha256(text.encode()),
            "text": text,
        }
    )


def write_indexing_inputs(root: Path) -> tuple[Path, Path, list[ChunkRecord]]:
    chunks = [
        make_chunk(0, "F-1 on-campus employment may be permitted under defined conditions."),
        make_chunk(1, "Travel requires valid documents and may require an endorsed Form I-20."),
        make_chunk(2, "Students normally must maintain a full course of study."),
    ]
    chunks_path = root / "data/processed/chunks.jsonl"
    chunks_path.parent.mkdir(parents=True)
    chunk_bytes = render_chunk_dataset(chunks)
    chunks_path.write_bytes(chunk_bytes)
    report = ChunkingReport(
        report_version=1,
        extraction_report_path="data/manifests/extraction.json",
        extraction_report_sha256="3" * 64,
        chunks_path="data/processed/chunks.jsonl",
        chunks_sha256=sha256(chunk_bytes),
        policy=ChunkingPolicy(),
        total_sources=1,
        total_chunks=len(chunks),
        total_source_characters=sum(chunk.character_count for chunk in chunks),
        total_chunk_characters=sum(chunk.character_count for chunk in chunks),
        sources=[
            ChunkingSourceSummary(
                source_id="synthetic-policy",
                document_version=1,
                extracted_sha256="2" * 64,
                source_character_count=sum(chunk.character_count for chunk in chunks),
                chunk_count=len(chunks),
                chunk_character_count=sum(chunk.character_count for chunk in chunks),
                min_chunk_characters=min(chunk.character_count for chunk in chunks),
                max_chunk_characters=max(chunk.character_count for chunk in chunks),
            )
        ],
    )
    report_path = root / "data/manifests/chunking.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(render_chunking_report(report), encoding="utf-8", newline="\n")
    return report_path, chunks_path, chunks


def write_built_index(
    root: Path,
    embedder: FakeEmbedder | None = None,
) -> tuple[Path, Path, VectorIndexReport]:
    report_path, _, _ = write_indexing_inputs(root)
    vectors_path = root / "data/indexes/vectors.npy"
    index_report_path = root / "data/manifests/index.json"
    vector_bytes, index_report = build_vector_index(
        report_path,
        vectors_path,
        root,
        embedder or FakeEmbedder(),
        batch_size=2,
    )
    write_vector_index(vectors_path, vector_bytes)
    write_vector_index_report(index_report_path, index_report)
    return vectors_path, index_report_path, index_report


def test_build_vector_index_normalizes_and_links_every_input(tmp_path: Path) -> None:
    chunking_report_path, _, chunks = write_indexing_inputs(tmp_path)
    vectors_path = tmp_path / "data/indexes/vectors.npy"

    vector_bytes, report = build_vector_index(
        chunking_report_path,
        vectors_path,
        tmp_path,
        FakeEmbedder(),
        batch_size=2,
    )
    vectors = parse_vector_index(vector_bytes)

    assert vectors.shape == (3, 3)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert report.vector_count == len(chunks)
    assert report.dimension == 3
    assert report.vectors_sha256 == sha256(vector_bytes)
    assert report.chunks_sha256 == sha256((tmp_path / "data/processed/chunks.jsonl").read_bytes())


def test_search_vector_index_returns_stable_exact_cosine_ranking(tmp_path: Path) -> None:
    _, index_report_path, _ = write_built_index(tmp_path)

    hits = search_vector_index(
        "What employment is allowed?",
        index_report_path,
        tmp_path,
        FakeEmbedder(),
        top_k=2,
    )

    assert [hit.rank for hit in hits] == [1, 2]
    assert hits[0].chunk_id == "synthetic-policy-v1-c00000"
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].chunk_id == "synthetic-policy-v1-c00001"


def test_search_vector_index_supports_source_filter(tmp_path: Path) -> None:
    _, index_report_path, _ = write_built_index(tmp_path)

    hits = search_vector_index(
        "employment",
        index_report_path,
        tmp_path,
        FakeEmbedder(),
        source_ids={"not-present"},
    )

    assert hits == []


def test_search_rejects_mismatched_embedder_identity(tmp_path: Path) -> None:
    _, index_report_path, _ = write_built_index(tmp_path)

    with pytest.raises(IndexingError, match="identity does not match"):
        search_vector_index(
            "employment",
            index_report_path,
            tmp_path,
            DifferentFakeEmbedder(),
        )


def test_load_vector_index_rejects_tampered_chunk_dataset(tmp_path: Path) -> None:
    _, index_report_path, _ = write_built_index(tmp_path)
    chunks_path = tmp_path / "data/processed/chunks.jsonl"
    chunks_path.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(IndexingError, match="chunk dataset checksum"):
        load_vector_index(index_report_path, tmp_path)


def test_load_vector_index_rejects_tampered_vectors(tmp_path: Path) -> None:
    vectors_path, index_report_path, _ = write_built_index(tmp_path)
    vectors_path.write_bytes(b"tampered")

    with pytest.raises(IndexingError, match="vector index checksum"):
        load_vector_index(index_report_path, tmp_path)


def test_vector_artifacts_are_immutable(tmp_path: Path) -> None:
    vectors_path, index_report_path, report = write_built_index(tmp_path)
    original = vectors_path.read_bytes()

    write_vector_index(vectors_path, original)
    with pytest.raises(IndexingError, match="refusing to replace"):
        write_vector_index(vectors_path, b"different")
    with pytest.raises(IndexingError, match="refusing to replace"):
        write_vector_index_report(index_report_path, report)


def test_parse_vector_index_rejects_wrong_dtype() -> None:
    buffer = render_vector_index(np.ones((2, 3), dtype=np.float32))
    assert parse_vector_index(buffer).dtype == np.float32

    with pytest.raises(IndexingError, match="must use float32"):
        render_vector_index(np.ones((2, 3), dtype=np.float64))


def test_vector_index_report_rejects_unsafe_path() -> None:
    embedder = FakeEmbedder()
    with pytest.raises(ValidationError, match="data/indexes"):
        VectorIndexReport(
            report_version=1,
            chunking_report_path="data/manifests/chunking.json",
            chunking_report_sha256="1" * 64,
            chunks_path="data/processed/chunks.jsonl",
            chunks_sha256="2" * 64,
            vectors_path="../vectors.npy",
            vectors_sha256="3" * 64,
            chunk_ids_sha256="4" * 64,
            model_name=embedder.model_name,
            model_repository=embedder.model_repository,
            model_revision=embedder.model_revision,
            embedding_engine=embedder.engine,
            embedding_engine_version=embedder.engine_version,
            vector_count=3,
            dimension=3,
        )


def test_fastembed_backend_rejects_invalid_thread_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        FastEmbedBackend(tmp_path, threads=0)


def test_index_and_search_cli_with_injected_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    chunking_report_path, _, _ = write_indexing_inputs(tmp_path)
    vectors_path = tmp_path / "data/indexes/vectors.npy"
    index_report_path = tmp_path / "data/manifests/index.json"
    monkeypatch.setattr(
        "deepshield_api.corpus.cli.FastEmbedBackend",
        lambda cache_dir, threads=1: FakeEmbedder(),
    )

    build_result = main(
        [
            "build-index",
            str(chunking_report_path),
            "--output-vectors",
            str(vectors_path),
            "--output-report",
            str(index_report_path),
            "--repository-root",
            str(tmp_path),
            "--model-cache",
            str(tmp_path / "data/indexes/models"),
        ]
    )
    build_output = capsys.readouterr().out
    search_result = main(
        [
            "search",
            str(index_report_path),
            "employment rules",
            "--repository-root",
            str(tmp_path),
            "--model-cache",
            str(tmp_path / "data/indexes/models"),
            "--top-k",
            "1",
        ]
    )
    search_output = capsys.readouterr().out

    assert build_result == 0
    assert "Embedded 3 chunk(s)" in build_output
    assert search_result == 0
    assert '"chunk_id": "synthetic-policy-v1-c00000"' in search_output
    assert index_report_path.read_text(encoding="utf-8") == render_vector_index_report(
        VectorIndexReport.model_validate_json(index_report_path.read_text(encoding="utf-8"))
    )


def test_committed_f1_vector_index_report_has_expected_identity() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    report_path = repository_root / "data/manifests/f1-policy-vector-index-report.v1.json"
    report = load_vector_index_report(report_path)

    assert report.model_dump(mode="json") == {
        "schema_version": "1.0",
        "corpus_id": "deepshield-rag-f1-policy",
        "report_version": 1,
        "indexer_version": "1.0",
        "chunking_report_path": "data/manifests/f1-policy-chunking-report.v1.json",
        "chunking_report_sha256": (
            "55e2d4f85c52511b47b9848983c78a00ef9280e4233c4e9ff2e4093a1bc89807"
        ),
        "chunks_path": "data/processed/f1-policy-chunks.v1.jsonl",
        "chunks_sha256": ("6dcb7c03719437406fca66a215c004583bc223d3d8c5bf5a58e51ad12bc41852"),
        "vectors_path": "data/indexes/f1-policy-v1/vectors.npy",
        "vectors_sha256": ("b0f8271590136227a1635d3e2ec39dddc21f70f65cd7be4632ad42fb2425df87"),
        "chunk_ids_sha256": ("286068f47eca16e2e03820164ff2532d438467e675114179dabb1663dcb38aaa"),
        "model_name": "BAAI/bge-small-en-v1.5",
        "model_repository": "Qdrant/bge-small-en-v1.5-onnx-Q",
        "model_revision": "aa8f8b060edb00e03bfdd08813a2949946c8ba55",
        "embedding_engine": "fastembed",
        "embedding_engine_version": "0.8.1",
        "vector_count": 1122,
        "dimension": 384,
        "dtype": "float32",
        "similarity": "cosine",
        "normalized": True,
    }
    assert report_path.read_text(encoding="utf-8") == render_vector_index_report(report)
