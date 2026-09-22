import hashlib
from datetime import datetime
from email.message import Message
from io import BytesIO
from pathlib import Path
from types import TracebackType
from typing import Any, Self
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

from deepshield_api.corpus import collection
from deepshield_api.corpus.collection import (
    CollectionError,
    CollectionPolicy,
    FetchedDocument,
    build_raw_path,
    collect_manifest_source,
    collect_source,
    download_source,
)
from deepshield_api.corpus.models import SourceManifest, SourceRecord

CONTENT = b"<html><body>Synthetic policy fixture</body></html>"
COLLECTED_AT = datetime.fromisoformat("2026-09-22T12:00:00+00:00")


def source_data(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "source_id": "synthetic-uscis-guidance",
        "document_version": 1,
        "title": "Synthetic USCIS guidance fixture",
        "publisher": "uscis",
        "canonical_url": "https://www.uscis.gov/synthetic-test-fixture",
        "document_type": "guidance-page",
        "media_type": "text/html",
        "scope_tags": ["status"],
        "rights_notes": "Synthetic metadata fixture; not an approved corpus source.",
    }
    data.update(overrides)
    return data


def planned_source(**overrides: Any) -> SourceRecord:
    return SourceRecord.model_validate(source_data(**overrides))


def fetched_document(**overrides: Any) -> FetchedDocument:
    data: dict[str, Any] = {
        "final_url": "https://www.uscis.gov/synthetic-test-fixture",
        "media_type": "text/html; charset=utf-8",
        "content": CONTENT,
    }
    data.update(overrides)
    return FetchedDocument(**data)


def fake_fetcher(source: SourceRecord, policy: CollectionPolicy) -> FetchedDocument:
    assert source.source_id == "synthetic-uscis-guidance"
    assert policy.max_bytes > len(CONTENT)
    return fetched_document()


def fixed_clock() -> datetime:
    return COLLECTED_AT


def test_collection_policy_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="max_bytes"):
        CollectionPolicy(max_bytes=0)
    with pytest.raises(ValueError, match="timeout_seconds"):
        CollectionPolicy(timeout_seconds=0)
    with pytest.raises(ValueError, match="user_agent"):
        CollectionPolicy(user_agent="  ")


def test_collect_source_writes_deterministic_immutable_artifact(tmp_path: Path) -> None:
    source = planned_source()

    collected = collect_source(source, tmp_path, fetcher=fake_fetcher, clock=fixed_clock)

    expected_path = Path(*build_raw_path(source).parts)
    artifact = tmp_path / expected_path
    assert artifact.read_bytes() == CONTENT
    assert collected.collection_status.value == "collected"
    assert collected.raw_path == expected_path.as_posix()
    assert collected.retrieved_at == COLLECTED_AT
    assert collected.sha256 == hashlib.sha256(CONTENT).hexdigest()

    repeated = collect_source(source, tmp_path, fetcher=fake_fetcher, clock=fixed_clock)
    assert repeated == collected


def test_collect_source_rejects_changed_existing_artifact(tmp_path: Path) -> None:
    source = planned_source()
    target = tmp_path / Path(*build_raw_path(source).parts)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"different")

    with pytest.raises(CollectionError, match="already exists with different bytes"):
        collect_source(source, tmp_path, fetcher=fake_fetcher, clock=fixed_clock)


def test_collect_source_rejects_directory_at_artifact_path(tmp_path: Path) -> None:
    source = planned_source()
    target = tmp_path / Path(*build_raw_path(source).parts)
    target.mkdir(parents=True)

    with pytest.raises(CollectionError, match="not a regular file"):
        collect_source(source, tmp_path, fetcher=fake_fetcher, clock=fixed_clock)


def test_collect_source_rejects_symlink_artifact(tmp_path: Path) -> None:
    source = planned_source()
    target = tmp_path / Path(*build_raw_path(source).parts)
    target.parent.mkdir(parents=True)
    external = tmp_path / "external.html"
    external.write_bytes(CONTENT)
    try:
        target.symlink_to(external)
    except OSError:
        pytest.skip("symbolic links are unavailable in this environment")

    with pytest.raises(CollectionError, match="symbolic link"):
        collect_source(source, tmp_path, fetcher=fake_fetcher, clock=fixed_clock)


def test_collect_source_rejects_symlinked_data_directory(tmp_path: Path) -> None:
    external = tmp_path.parent / f"{tmp_path.name}-external"
    external.mkdir()
    try:
        (tmp_path / "data").symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("symbolic links are unavailable in this environment")

    with pytest.raises(CollectionError, match="escapes its repository"):
        collect_source(planned_source(), tmp_path, fetcher=fake_fetcher, clock=fixed_clock)


@pytest.mark.parametrize(
    ("fetched", "message"),
    [
        (fetched_document(content=b""), "empty body"),
        (fetched_document(media_type="application/pdf"), "expected text/html"),
        (fetched_document(final_url="http://www.uscis.gov/example"), "approved HTTPS"),
        (fetched_document(final_url="https://uscis.gov.example.com/example"), "approved HTTPS"),
        (fetched_document(final_url="https://user@www.uscis.gov/example"), "approved HTTPS"),
        (fetched_document(final_url="https://www.uscis.gov/example#fragment"), "approved HTTPS"),
    ],
)
def test_collect_source_rejects_invalid_response(
    tmp_path: Path,
    fetched: FetchedDocument,
    message: str,
) -> None:
    def fetcher(source: SourceRecord, policy: CollectionPolicy) -> FetchedDocument:
        return fetched

    with pytest.raises(CollectionError, match=message):
        collect_source(planned_source(), tmp_path, fetcher=fetcher, clock=fixed_clock)


def test_collect_source_rejects_oversized_response(tmp_path: Path) -> None:
    def fetcher(source: SourceRecord, policy: CollectionPolicy) -> FetchedDocument:
        return fetched_document(content=b"12345")

    with pytest.raises(CollectionError, match="exceeds the 4-byte limit"):
        collect_source(
            planned_source(),
            tmp_path,
            policy=CollectionPolicy(max_bytes=4),
            fetcher=fetcher,
            clock=fixed_clock,
        )


def test_collect_source_rejects_collected_record_and_naive_clock(tmp_path: Path) -> None:
    collected = SourceRecord.model_validate(
        source_data(
            collection_status="collected",
            retrieved_at="2026-09-22T12:00:00Z",
            raw_path="data/raw/uscis/example.html",
            sha256="a" * 64,
        )
    )
    with pytest.raises(CollectionError, match="only planned"):
        collect_source(collected, tmp_path, fetcher=fake_fetcher)

    with pytest.raises(CollectionError, match="timezone-aware"):
        collect_source(
            planned_source(),
            tmp_path,
            fetcher=fake_fetcher,
            clock=lambda: datetime(2026, 9, 22, 12, 0),
        )


def test_collect_manifest_source_versions_manifest_and_preserves_other_sources(
    tmp_path: Path,
) -> None:
    second = source_data(
        source_id="synthetic-uscis-form",
        title="Second synthetic source",
        canonical_url="https://www.uscis.gov/second-synthetic-fixture",
    )
    manifest = SourceManifest.model_validate(
        {
            "manifest_version": 4,
            "created_at": "2026-09-21T12:00:00Z",
            "sources": [source_data(), second],
        }
    )

    collected = collect_manifest_source(
        manifest,
        "synthetic-uscis-guidance",
        1,
        tmp_path,
        fetcher=fake_fetcher,
        clock=fixed_clock,
    )

    assert collected.manifest_version == 5
    assert collected.created_at == COLLECTED_AT
    assert collected.sources[0].collection_status.value == "collected"
    assert collected.sources[1] == manifest.sources[1]


def test_collect_manifest_source_rejects_unknown_source(tmp_path: Path) -> None:
    manifest = SourceManifest.model_validate(
        {
            "manifest_version": 1,
            "created_at": "2026-09-21T12:00:00Z",
            "sources": [source_data()],
        }
    )

    with pytest.raises(CollectionError, match="source not found"):
        collect_manifest_source(manifest, "missing", 1, tmp_path, fetcher=fake_fetcher)


class FakeResponse:
    def __init__(
        self,
        content: bytes,
        *,
        media_type: str | None = "text/html; charset=utf-8",
        content_length: str | None = None,
        final_url: str = "https://www.uscis.gov/synthetic-test-fixture",
    ) -> None:
        self._body = BytesIO(content)
        self._final_url = final_url
        self.headers = Message()
        if media_type is not None:
            self.headers["Content-Type"] = media_type
        if content_length is not None:
            self.headers["Content-Length"] = content_length

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    def geturl(self) -> str:
        return self._final_url


class FakeOpener:
    def __init__(self, response: FakeResponse | Exception) -> None:
        self._response = response

    def open(self, request: Request, timeout: float) -> FakeResponse:
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def test_download_source_sets_headers_and_reads_response(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, Any] = {}
    response = FakeResponse(CONTENT, content_length=str(len(CONTENT)))

    class ObservedOpener(FakeOpener):
        def open(self, request: Request, timeout: float) -> FakeResponse:
            response_value = super().open(request, timeout)
            observed["accept"] = request.get_header("Accept")
            observed["user_agent"] = request.get_header("User-agent")
            observed["timeout"] = timeout
            return response_value

    def fake_build_opener(handler: object) -> FakeOpener:
        assert isinstance(handler, collection._ApprovedRedirectHandler)
        return ObservedOpener(response)

    monkeypatch.setattr(collection, "build_opener", fake_build_opener)
    policy = CollectionPolicy(timeout_seconds=7.5, user_agent="DeepShield-Test/1.0")

    fetched = download_source(planned_source(), policy)

    assert fetched == fetched_document()
    assert observed == {
        "accept": "text/html",
        "user_agent": "DeepShield-Test/1.0",
        "timeout": 7.5,
    }


def test_redirect_handler_rejects_unapproved_destination() -> None:
    handler = collection._ApprovedRedirectHandler(planned_source())

    with pytest.raises(CollectionError, match="approved HTTPS"):
        handler.redirect_request(
            Request("https://www.uscis.gov/example"),
            BytesIO(),
            302,
            "Found",
            Message(),
            "http://127.0.0.1/private",
        )

    with pytest.raises(CollectionError, match="approved HTTPS"):
        handler.redirect_request(
            Request("https://www.uscis.gov/example"),
            BytesIO(),
            302,
            "Found",
            Message(),
            "https://www.uscis.gov:8443/unexpected-service",
        )


def test_redirect_handler_allows_approved_destination() -> None:
    handler = collection._ApprovedRedirectHandler(planned_source())
    redirected = handler.redirect_request(
        Request("https://www.uscis.gov/example"),
        BytesIO(),
        302,
        "Found",
        Message(),
        "https://www.uscis.gov/new-location",
    )

    assert redirected is not None
    assert redirected.full_url == "https://www.uscis.gov/new-location"


@pytest.mark.parametrize(
    ("content_length", "content", "max_bytes", "message"),
    [
        ("invalid", CONTENT, 100, "invalid Content-Length"),
        ("-1", CONTENT, 100, "negative Content-Length"),
        ("101", CONTENT, 100, "declares 101 bytes"),
        ("1", CONTENT, 100, "does not match"),
        (None, b"12345", 4, "body exceeds"),
    ],
)
def test_download_source_rejects_invalid_or_oversized_lengths(
    monkeypatch: pytest.MonkeyPatch,
    content_length: str | None,
    content: bytes,
    max_bytes: int,
    message: str,
) -> None:
    response = FakeResponse(content, content_length=content_length)
    monkeypatch.setattr(collection, "build_opener", lambda handler: FakeOpener(response))

    with pytest.raises(CollectionError, match=message):
        download_source(planned_source(), CollectionPolicy(max_bytes=max_bytes))


def test_download_source_requires_declared_media_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = FakeResponse(CONTENT, media_type=None)
    monkeypatch.setattr(collection, "build_opener", lambda handler: FakeOpener(response))

    with pytest.raises(CollectionError, match="missing a Content-Type"):
        download_source(planned_source(), CollectionPolicy())


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (
            HTTPError(
                "https://www.uscis.gov/example",
                403,
                "Forbidden",
                Message(),
                None,
            ),
            "HTTP status 403",
        ),
        (URLError("network unavailable"), "complete response"),
    ],
)
def test_download_source_wraps_network_errors(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    message: str,
) -> None:
    monkeypatch.setattr(collection, "build_opener", lambda handler: FakeOpener(error))

    with pytest.raises(CollectionError, match=message):
        download_source(planned_source(), CollectionPolicy())
