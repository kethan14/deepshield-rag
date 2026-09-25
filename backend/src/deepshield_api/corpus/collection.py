"""Safely collect immutable raw artifacts for approved manifest records."""

import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from deepshield_api.corpus.models import (
    PUBLISHER_DOMAINS,
    CollectionStatus,
    SourceManifest,
    SourceMediaType,
    SourceRecord,
)

DEFAULT_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 30.0
READ_CHUNK_BYTES = 64 * 1024
DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; DeepShield-RAG/0.1; academic corpus research)"


class CollectionError(RuntimeError):
    """Raised when a source cannot be collected safely and reproducibly."""


@dataclass(frozen=True, slots=True)
class CollectionPolicy:
    """Network and artifact limits applied to every source collection."""

    max_bytes: int = DEFAULT_MAX_BYTES
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    user_agent: str = DEFAULT_USER_AGENT

    def __post_init__(self) -> None:
        if self.max_bytes <= 0:
            raise ValueError("max_bytes must be greater than zero")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if not self.user_agent.strip():
            raise ValueError("user_agent must not be empty")


@dataclass(frozen=True, slots=True)
class FetchedDocument:
    """Bytes and response metadata returned by a source fetcher."""

    final_url: str
    media_type: str
    content: bytes


SourceFetcher = Callable[[SourceRecord, CollectionPolicy], FetchedDocument]
Clock = Callable[[], datetime]


class _ApprovedRedirectHandler(HTTPRedirectHandler):
    """Reject a redirect before requesting an unapproved destination."""

    def __init__(self, source: SourceRecord) -> None:
        super().__init__()
        self._source = source

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request | None:
        _validate_source_url(self._source, newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def utc_now() -> datetime:
    """Return an aware UTC timestamp for collection metadata."""

    return datetime.now(UTC)


def download_source(source: SourceRecord, policy: CollectionPolicy) -> FetchedDocument:
    """Download one source while enforcing timeout and byte limits."""

    request = Request(
        str(source.canonical_url),
        headers={
            "Accept": source.media_type.value,
            "User-Agent": policy.user_agent,
        },
    )

    opener = build_opener(_ApprovedRedirectHandler(source))
    try:
        with opener.open(request, timeout=policy.timeout_seconds) as response:
            content_type = response.headers.get("Content-Type")
            if content_type is None:
                raise CollectionError("source response is missing a Content-Type header")
            declared_length = _parse_content_length(response.headers.get("Content-Length"))
            if declared_length is not None and declared_length > policy.max_bytes:
                raise CollectionError(
                    f"source declares {declared_length} bytes, exceeding the "
                    f"{policy.max_bytes}-byte limit"
                )

            content = bytearray()
            while True:
                remaining = policy.max_bytes - len(content)
                chunk = response.read(min(READ_CHUNK_BYTES, remaining + 1))
                if not chunk:
                    break
                content.extend(chunk)
                if len(content) > policy.max_bytes:
                    raise CollectionError(f"source body exceeds the {policy.max_bytes}-byte limit")

            if declared_length is not None and len(content) != declared_length:
                raise CollectionError("source body length does not match the Content-Length header")

            return FetchedDocument(
                final_url=response.geturl(),
                media_type=content_type,
                content=bytes(content),
            )
    except CollectionError:
        raise
    except HTTPError as exc:
        raise CollectionError(f"source request failed with HTTP status {exc.code}") from exc
    except (OSError, TimeoutError, URLError) as exc:
        raise CollectionError(
            "source request failed before a complete response was received"
        ) from exc


def collect_source(
    source: SourceRecord,
    repository_root: Path,
    *,
    policy: CollectionPolicy | None = None,
    fetcher: SourceFetcher = download_source,
    clock: Clock = utc_now,
) -> SourceRecord:
    """Collect a planned source and return its validated collected record."""

    if source.collection_status is not CollectionStatus.PLANNED:
        raise CollectionError("only planned sources can be collected")

    effective_policy = policy or CollectionPolicy()
    fetched = fetcher(source, effective_policy)
    return _store_fetched_source(
        source,
        fetched,
        repository_root,
        policy=effective_policy,
        clock=clock,
    )


def import_source_file(
    source: SourceRecord,
    input_path: Path,
    repository_root: Path,
    *,
    policy: CollectionPolicy | None = None,
    clock: Clock = utc_now,
) -> SourceRecord:
    """Validate a browser-downloaded file and store it as an immutable raw artifact."""

    if source.collection_status is not CollectionStatus.PLANNED:
        raise CollectionError("only planned sources can be collected")
    if input_path.is_symlink() or not input_path.is_file():
        raise CollectionError("import input must be a regular non-symbolic-link file")

    effective_policy = policy or CollectionPolicy()
    try:
        declared_size = input_path.stat().st_size
    except OSError as exc:
        raise CollectionError("unable to inspect import input") from exc
    if declared_size > effective_policy.max_bytes:
        raise CollectionError(f"import input exceeds the {effective_policy.max_bytes}-byte limit")
    try:
        content = input_path.read_bytes()
    except OSError as exc:
        raise CollectionError("unable to read import input") from exc

    _validate_import_signature(source, content)
    fetched = FetchedDocument(
        final_url=str(source.canonical_url),
        media_type=source.media_type.value,
        content=content,
    )
    return _store_fetched_source(
        source,
        fetched,
        repository_root,
        policy=effective_policy,
        clock=clock,
    )


def _store_fetched_source(
    source: SourceRecord,
    fetched: FetchedDocument,
    repository_root: Path,
    *,
    policy: CollectionPolicy,
    clock: Clock,
) -> SourceRecord:
    """Validate fetched bytes and persist one collected source record."""

    _validate_fetched_document(source, fetched, policy)

    retrieved_at = clock()
    if retrieved_at.utcoffset() is None:
        raise CollectionError("collection clock must return a timezone-aware timestamp")

    raw_path = build_raw_path(source)
    target = _prepare_raw_target(repository_root, raw_path)
    digest = hashlib.sha256(fetched.content).hexdigest()
    _write_immutable_artifact(target, fetched.content, digest)

    payload = source.model_dump(mode="json")
    payload.update(
        {
            "collection_status": CollectionStatus.COLLECTED.value,
            "retrieved_at": retrieved_at,
            "raw_path": raw_path.as_posix(),
            "sha256": digest,
        }
    )
    return SourceRecord.model_validate(payload)


def collect_manifest_source(
    manifest: SourceManifest,
    source_id: str,
    document_version: int,
    repository_root: Path,
    *,
    policy: CollectionPolicy | None = None,
    fetcher: SourceFetcher = download_source,
    clock: Clock = utc_now,
) -> SourceManifest:
    """Collect one selected source and produce the next manifest version."""

    matches = [
        (index, source)
        for index, source in enumerate(manifest.sources)
        if source.source_id == source_id and source.document_version == document_version
    ]
    if not matches:
        raise CollectionError(f"manifest source not found: {source_id} v{document_version}")

    index, source = matches[0]
    collected_at = clock()
    collected = collect_source(
        source,
        repository_root,
        policy=policy,
        fetcher=fetcher,
        clock=lambda: collected_at,
    )

    sources = [item.model_dump(mode="json") for item in manifest.sources]
    sources[index] = collected.model_dump(mode="json")
    payload = manifest.model_dump(mode="json")
    payload.update(
        {
            "manifest_version": manifest.manifest_version + 1,
            "created_at": collected_at,
            "sources": sources,
        }
    )
    return SourceManifest.model_validate(payload)


def import_manifest_source_file(
    manifest: SourceManifest,
    source_id: str,
    document_version: int,
    input_path: Path,
    repository_root: Path,
    *,
    policy: CollectionPolicy | None = None,
    clock: Clock = utc_now,
) -> SourceManifest:
    """Import one browser-downloaded source and produce the next manifest version."""

    matches = [
        (index, source)
        for index, source in enumerate(manifest.sources)
        if source.source_id == source_id and source.document_version == document_version
    ]
    if not matches:
        raise CollectionError(f"manifest source not found: {source_id} v{document_version}")

    index, source = matches[0]
    collected_at = clock()
    collected = import_source_file(
        source,
        input_path,
        repository_root,
        policy=policy,
        clock=lambda: collected_at,
    )

    sources = [item.model_dump(mode="json") for item in manifest.sources]
    sources[index] = collected.model_dump(mode="json")
    payload = manifest.model_dump(mode="json")
    payload.update(
        {
            "manifest_version": manifest.manifest_version + 1,
            "created_at": collected_at,
            "sources": sources,
        }
    )
    return SourceManifest.model_validate(payload)


def build_raw_path(source: SourceRecord) -> PurePosixPath:
    """Build the deterministic repository-relative path for a raw artifact."""

    suffixes = {
        SourceMediaType.HTML: ".html",
        SourceMediaType.PDF: ".pdf",
        SourceMediaType.PLAIN_TEXT: ".txt",
    }
    filename = f"v{source.document_version}{suffixes[source.media_type]}"
    return PurePosixPath(
        "data",
        "raw",
        source.publisher.value,
        source.source_id,
        filename,
    )


def _parse_content_length(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        length = int(value)
    except ValueError as exc:
        raise CollectionError("source returned an invalid Content-Length header") from exc
    if length < 0:
        raise CollectionError("source returned a negative Content-Length header")
    return length


def _validate_fetched_document(
    source: SourceRecord,
    fetched: FetchedDocument,
    policy: CollectionPolicy,
) -> None:
    if not fetched.content:
        raise CollectionError("source returned an empty body")
    if len(fetched.content) > policy.max_bytes:
        raise CollectionError(f"source body exceeds the {policy.max_bytes}-byte limit")

    media_type = fetched.media_type.split(";", maxsplit=1)[0].strip().lower()
    if media_type != source.media_type.value:
        raise CollectionError(
            f"source returned {media_type or 'no media type'}; expected {source.media_type.value}"
        )

    _validate_source_url(source, fetched.final_url)


def _validate_import_signature(source: SourceRecord, content: bytes) -> None:
    """Reject local files whose bytes do not match the declared media type."""

    if not content:
        raise CollectionError("import input is empty")
    if source.media_type is SourceMediaType.PDF:
        if not content.startswith(b"%PDF-"):
            raise CollectionError("import input does not have a PDF file signature")
        return

    try:
        decoded = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CollectionError("import input is not valid UTF-8 text") from exc
    if source.media_type is SourceMediaType.HTML:
        signature_window = decoded[:4096].lower()
        if "<!doctype html" not in signature_window and "<html" not in signature_window:
            raise CollectionError("import input does not have an HTML document signature")


def _validate_source_url(source: SourceRecord, url: str) -> None:
    parsed = urlsplit(url)
    hostname = parsed.hostname.lower().rstrip(".") if parsed.hostname else None
    allowed_domains = PUBLISHER_DOMAINS[source.publisher]
    is_allowed = hostname is not None and any(
        hostname == domain or hostname.endswith(f".{domain}") for domain in allowed_domains
    )
    if (
        parsed.scheme.lower() != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
        or parsed.fragment
        or not is_allowed
    ):
        raise CollectionError("source redirected outside its approved HTTPS publisher domain")


def _prepare_raw_target(repository_root: Path, raw_path: PurePosixPath) -> Path:
    root = repository_root.resolve()
    data_directory = _create_safe_directory(root, root / "data")
    raw_root = _create_safe_directory(data_directory, data_directory / "raw")
    publisher_directory = _create_safe_directory(raw_root, raw_root / raw_path.parts[2])
    source_directory = _create_safe_directory(
        publisher_directory,
        publisher_directory / raw_path.parts[3],
    )
    target = source_directory / raw_path.name
    if target.is_symlink():
        raise CollectionError("raw artifact target must not be a symbolic link")
    return target


def _create_safe_directory(parent: Path, candidate: Path) -> Path:
    candidate.mkdir(parents=True, exist_ok=True)
    resolved = candidate.resolve()
    try:
        resolved.relative_to(parent)
    except ValueError as exc:
        raise CollectionError("raw artifact path escapes its repository directory") from exc
    return resolved


def _write_immutable_artifact(target: Path, content: bytes, digest: str) -> None:
    if target.exists():
        if not target.is_file():
            raise CollectionError(f"raw artifact target is not a regular file: {target}")
        existing_digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if existing_digest != digest:
            raise CollectionError(
                f"immutable raw artifact already exists with different bytes: {target}"
            )
        return

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
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
