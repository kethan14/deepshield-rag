"""Safe, reproducible collection of pinned classifier source files."""

import csv
import hashlib
import os
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile

from huggingface_hub import hf_hub_download

from deepshield_api.classifier.manifest import load_source_registry
from deepshield_api.classifier.models import (
    ClassifierCollectionReceipt,
    ClassifierSource,
    ClassifierSourceRegistry,
    CollectedClassifierFile,
)

DEFAULT_MAX_FILE_BYTES = 64 * 1024 * 1024
Clock = Callable[[], datetime]
SourceDownloader = Callable[[ClassifierSource, str], Path]


class ClassifierCollectionError(RuntimeError):
    """Raised when pinned classifier data cannot be collected safely."""


def utc_now() -> datetime:
    """Return an aware UTC timestamp for receipt metadata."""

    return datetime.now(UTC)


def download_registered_file(source: ClassifierSource, source_file: str) -> Path:
    """Download one file from the source's immutable Hugging Face revision."""

    try:
        downloaded = hf_hub_download(
            repo_id=source.dataset_id,
            filename=source_file,
            repo_type="dataset",
            revision=source.revision,
        )
    except Exception as exc:
        raise ClassifierCollectionError(
            f"unable to download registered classifier source file: {source.source_id}/{source_file}"
        ) from exc
    return Path(downloaded)


def collect_registered_sources(
    registry_path: Path,
    repository_root: Path,
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    downloader: SourceDownloader = download_registered_file,
    clock: Clock = utc_now,
) -> ClassifierCollectionReceipt:
    """Collect every registered file and return a validated evidence receipt."""

    if max_file_bytes <= 0:
        raise ValueError("max_file_bytes must be greater than zero")
    root = repository_root.resolve()
    resolved_registry = registry_path.resolve()
    expected_registry = (root / "data" / "classifier" / "source-registry.v2.json").resolve()
    if resolved_registry != expected_registry:
        raise ClassifierCollectionError("collection requires source-registry.v2.json")
    registry = load_source_registry(resolved_registry)
    registry_hash = _sha256(resolved_registry.read_bytes())
    collected_at = clock()
    if collected_at.utcoffset() is None:
        raise ClassifierCollectionError("collection clock must return a timezone-aware timestamp")

    records: list[CollectedClassifierFile] = []
    for source in registry.sources:
        for source_file in source.files:
            raw_path = _raw_path(source, source_file)
            target = _prepare_raw_target(root, raw_path)
            if target.exists():
                content = _read_existing_artifact(target, max_file_bytes)
            else:
                downloaded = downloader(source, source_file)
                content = _read_download(downloaded, max_file_bytes)
                _write_immutable_artifact(target, content, _sha256(content))
            records.append(_inspect_csv(source, source_file, raw_path, content))

    receipt = ClassifierCollectionReceipt(
        receipt_version=1,
        collected_at=collected_at,
        source_registry_path="data/classifier/source-registry.v2.json",
        source_registry_sha256=registry_hash,
        files=records,
    )
    validate_collection_receipt(receipt, root, require_files=True, registry=registry)
    return receipt


def validate_collection_receipt(
    receipt: ClassifierCollectionReceipt,
    repository_root: Path,
    *,
    require_files: bool,
    registry: ClassifierSourceRegistry | None = None,
) -> None:
    """Validate registry references, expected counts, and optionally local raw bytes."""

    root = repository_root.resolve()
    registry_path = (root / receipt.source_registry_path).resolve()
    classifier_root = (root / "data" / "classifier").resolve()
    if not registry_path.is_relative_to(classifier_root):
        raise ClassifierCollectionError("source registry resolves outside data/classifier")
    try:
        registry_bytes = registry_path.read_bytes()
    except OSError as exc:
        raise ClassifierCollectionError(f"unable to read source registry: {registry_path}") from exc
    if _sha256(registry_bytes) != receipt.source_registry_sha256:
        raise ClassifierCollectionError("collection source registry checksum does not match")
    effective_registry = registry or load_source_registry(registry_path)
    expected = {
        (source.source_id, source_file): (source, source_file)
        for source in effective_registry.sources
        for source_file in source.files
    }
    actual = {(item.source_id, item.source_file): item for item in receipt.files}
    if set(actual) != set(expected):
        raise ClassifierCollectionError("collection receipt files do not match the source registry")

    row_counts: Counter[str] = Counter()
    for identity, item in actual.items():
        source, source_file = expected[identity]
        if item.dataset_id != source.dataset_id or item.revision != source.revision:
            raise ClassifierCollectionError(
                "collection receipt source identity does not match registry"
            )
        required_columns = {
            source.columns.text,
            source.columns.label,
            *(
                column
                for column in (
                    source.columns.group,
                    source.columns.context,
                    source.columns.severity,
                )
                if column is not None
            ),
        }
        if not required_columns.issubset(item.columns):
            raise ClassifierCollectionError(
                f"collected CSV is missing registered columns: {source.source_id}/{source_file}"
            )
        if set(item.label_counts) != set(source.label_mapping):
            raise ClassifierCollectionError(
                f"collected CSV labels do not match registry: {source.source_id}/{source_file}"
            )
        expected_hash = source.lfs_sha256.get(source_file)
        if expected_hash is not None and item.sha256 != expected_hash:
            raise ClassifierCollectionError(
                f"collected checksum does not match registry: {source.source_id}/{source_file}"
            )
        row_counts[source.source_id] += item.row_count
        if require_files:
            _verify_local_file(item, root)

    for source in effective_registry.sources:
        if (
            source.expected_rows is not None
            and row_counts[source.source_id] != source.expected_rows
        ):
            raise ClassifierCollectionError(
                f"collected row count does not match registry for {source.source_id}: "
                f"expected {source.expected_rows}, found {row_counts[source.source_id]}"
            )


def _inspect_csv(
    source: ClassifierSource,
    source_file: str,
    raw_path: PurePosixPath,
    content: bytes,
) -> CollectedClassifierFile:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ClassifierCollectionError(
            f"classifier source is not valid UTF-8: {source.source_id}/{source_file}"
        ) from exc
    reader = csv.DictReader(StringIO(text, newline=""))
    columns = reader.fieldnames
    if columns is None:
        raise ClassifierCollectionError(
            f"classifier source has no CSV header: {source.source_id}/{source_file}"
        )
    if any(column is None or not column.strip() for column in columns):
        raise ClassifierCollectionError(
            f"classifier source has an empty CSV column: {source.source_id}/{source_file}"
        )
    label_counts: Counter[str] = Counter()
    row_count = 0
    empty_text_count = 0
    try:
        for row in reader:
            if None in row:
                raise ClassifierCollectionError(
                    f"classifier source row has extra CSV fields: {source.source_id}/{source_file}"
                )
            row_count += 1
            label = row.get(source.columns.label)
            if label is None:
                raise ClassifierCollectionError(
                    f"classifier source is missing label column: {source.source_id}/{source_file}"
                )
            label_counts[label] += 1
            if not (row.get(source.columns.text) or "").strip():
                empty_text_count += 1
    except csv.Error as exc:
        raise ClassifierCollectionError(
            f"classifier source is not valid CSV: {source.source_id}/{source_file}"
        ) from exc
    if row_count == 0:
        raise ClassifierCollectionError(
            f"classifier source contains no records: {source.source_id}/{source_file}"
        )
    if set(label_counts) != set(source.label_mapping):
        raise ClassifierCollectionError(
            f"collected CSV labels do not match registry: {source.source_id}/{source_file}"
        )
    return CollectedClassifierFile(
        source_id=source.source_id,
        dataset_id=source.dataset_id,
        revision=source.revision,
        source_file=source_file,
        raw_path=raw_path.as_posix(),
        sha256=_sha256(content),
        size_bytes=len(content),
        row_count=row_count,
        columns=list(columns),
        label_counts=dict(sorted(label_counts.items())),
        empty_text_count=empty_text_count,
    )


def _raw_path(source: ClassifierSource, source_file: str) -> PurePosixPath:
    return PurePosixPath("data", "classifier", "raw", source.source_id, source_file)


def _prepare_raw_target(root: Path, raw_path: PurePosixPath) -> Path:
    data_root = _create_safe_directory(root, root / "data")
    classifier_root = _create_safe_directory(data_root, data_root / "classifier")
    raw_root = _create_safe_directory(classifier_root, classifier_root / "raw")
    parent = raw_root
    for part in raw_path.parts[3:-1]:
        parent = _create_safe_directory(parent, parent / part)
    target = parent / raw_path.name
    if target.is_symlink():
        raise ClassifierCollectionError("classifier raw target must not be a symbolic link")
    return target


def _create_safe_directory(parent: Path, candidate: Path) -> Path:
    candidate.mkdir(parents=False, exist_ok=True)
    if candidate.is_symlink():
        raise ClassifierCollectionError("classifier raw directory must not be a symbolic link")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(parent.resolve()):
        raise ClassifierCollectionError("classifier raw path escapes data/classifier/raw")
    return resolved


def _read_existing_artifact(path: Path, max_file_bytes: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ClassifierCollectionError("existing classifier raw artifact is not a regular file")
    return _read_limited(path, max_file_bytes, "existing classifier raw artifact")


def _read_download(path: Path, max_file_bytes: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ClassifierCollectionError("downloaded classifier source is not a regular file")
    return _read_limited(path, max_file_bytes, "downloaded classifier source")


def _read_limited(path: Path, max_file_bytes: int, label: str) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ClassifierCollectionError(f"unable to inspect {label}") from exc
    if size <= 0:
        raise ClassifierCollectionError(f"{label} is empty")
    if size > max_file_bytes:
        raise ClassifierCollectionError(f"{label} exceeds the {max_file_bytes}-byte limit")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ClassifierCollectionError(f"unable to read {label}") from exc


def _verify_local_file(item: CollectedClassifierFile, root: Path) -> None:
    candidate = root / item.raw_path
    raw_root = (root / "data" / "classifier" / "raw").resolve()
    path = candidate.resolve()
    if candidate.is_symlink() or not path.is_relative_to(raw_root) or not candidate.is_file():
        raise ClassifierCollectionError(f"collected raw file is unavailable: {item.raw_path}")
    content = candidate.read_bytes()
    if len(content) != item.size_bytes or _sha256(content) != item.sha256:
        raise ClassifierCollectionError(
            f"collected raw file no longer matches receipt: {item.raw_path}"
        )


def _write_immutable_artifact(target: Path, content: bytes, digest: str) -> None:
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, target)
        if _sha256(target.read_bytes()) != digest:
            raise ClassifierCollectionError("stored classifier raw artifact checksum changed")
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
